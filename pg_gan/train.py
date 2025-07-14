"""
Application entry point for PG-GAN.
Author: Sara Mathieson, Zhanpeng Wang, Jiaping Wang, Rebecca Riley
Date 9/27/22
"""

# python imports
import datetime
import numpy as np
import os
import random
import sys
import tensorflow as tf
from sklearn.metrics import roc_auc_score
import wandb

# our imports
from pg_gan import discriminator
from pg_gan import global_vars
from pg_gan import util

# globals for simulated annealing
NUM_ITER = 500
NUM_BATCH = 100
print("NUM_ITER", NUM_ITER)
print("BATCH_SIZE", global_vars.BATCH_SIZE)
print("NUM_BATCH", NUM_BATCH)

# globals for data
NUM_CLASSES = 2     # "real" vs "simulated"
NUM_CHANNELS = 2    # SNPs and distances
print("NUM_SNPS", global_vars.NUM_SNPS)
print("L", global_vars.L)
print("NUM_CLASSES", NUM_CLASSES)
print("NUM_CHANNELS", NUM_CHANNELS)

def reset_random_seeds(seed):
   os.environ['PYTHONHASHSEED']=str(seed)
   tf.random.set_seed(seed)
   np.random.seed(seed)
   random.seed(seed)
   
def main():
    """Parse args and run simulated annealing"""

    opts = util.parse_args()
    print(opts)

    if opts.wandb:
        wandb.init(project="pg-gan", config=vars(opts))


    # set up seeds
    reset_random_seeds(opts.seed)

    generator, iterator, iterable_params, sample_sizes = util.process_opts(opts)
    #disc = discriminator.MultiPopModel(sample_sizes)
    disc = get_discriminator(sample_sizes, opts.seed)

    # grid search
    if opts.grid:
        print("Grid search not supported right now")
        sys.exit()
        #posterior, loss_lst = grid_search(disc, samples, simulator,
        #    iterator, parameters, opts.seed)
    # simulated annealing
    else:
        posterior, loss_lst = simulated_annealing(generator, disc,
            iterator, iterable_params, opts.seed, toy=opts.toy, prefix=opts.out_prefix,
            opts=opts)

    print(posterior)
    print(loss_lst)

################################################################################
# SIMULATED ANNEALING
################################################################################

def simulated_annealing(generator, disc, iterator, iterable_params, seed,
    toy=False, prefix=None, opts=None):
    """Main function that drives GAN updates"""

    # main object for pg-gan
    pg_gan = PG_GAN(generator, disc, iterator, iterable_params, seed,
                    disc_lr=opts.disc_lr, gp_weight=opts.gp_weight)

    # start
    s_current = iterable_params.clone(start=True)
    
    pg_gan.generator.update_params(s_current)
    loss_curr = pg_gan.generator_loss(s_current)

    print("params, loss", s_current.to_list(), loss_curr)

    posterior = [s_current.to_list()]
    loss_lst = [loss_curr]

    # simulated-annealing iterations
    num_iter = NUM_ITER
    # for toy example
    if toy:
        num_iter = 2

    s_current_set = s_current.param_set

    # main pg-gan loop
    for i in range(num_iter):
        print(f"ITER: {i} | TIME: {datetime.datetime.now().time()}")

        T = temperature(i, num_iter) # cool down

        # train disc first
        pg_gan.train_sa(opts.num_critic)

        # then propose multiple updates to the generator
        num_gen_steps = opts.num_gen
        for j in range(num_gen_steps):
            # propose an update
            s_proposal = s_current.clone()
            param_names = list(s_proposal.param_set.keys())
            chosen_param = np.random.choice(param_names)
            s_proposal.propose_param(chosen_param, s_proposal.get(chosen_param), multiplier=T)
            
            # acceptance
            loss_proposal = pg_gan.generator_loss(s_proposal)

            # decide whether to accept or not (reduce accepting bad state later on)
            if loss_proposal < loss_curr: # unsure about this equal here
                p_accept = 1.0
            else:
                p_accept = min(np.exp(-(loss_proposal - loss_curr) / T), 1.0)
            
            rand = np.random.rand()
            accept = rand < p_accept

            # if accept, update generator
            if accept:
                s_current = s_proposal
                loss_curr = loss_proposal
                pg_gan.generator.update_params(s_current)
                
            if wandb.run:
                wandb.log({
                    "iteration": i,
                    "temperature": T,
                    "p_accept": p_accept,
                    "accepted": float(accept),
                    "generator_loss": loss_curr,
                }, commit=False)

            posterior.append(s_current.to_list())
            loss_lst.append(loss_curr)

    if prefix is not None:
        pg_gan.discriminator.save(prefix + ".keras")
    return posterior, loss_lst

def temperature(i, num_iter):
    """Temperature controls the width of the proposal and acceptance prob."""
    return 1 - i/num_iter # start at 1, end at 0

# not used right now
"""
def grid_search(model_type, samples, demo_file, simulator, iterator, parameters,
    is_range, seed):

    # can only do one param right now
    assert len(parameters) == 1
    param = parameters[0]

    all_values = []
    all_likelihood = []
    for fake_value in np.linspace(param.min, param.max, num=30):
        fake_params = [fake_value]
        model = TrainingModel(model_type, samples, demo_file, simulator,
            iterator, parameters, is_range, seed)

        # train more for grid search
        model.train(fake_params, NUM_BATCH*10, globals.BATCH_SIZE)
        test_acc, conf_mat = model.test(fake_params, NUM_TEST)
        like_curr = likelihood(test_acc)
        print("params, test_acc, likelihood", fake_value, test_acc, like_curr)

        all_values.append(fake_value)
        all_likelihood.append(like_curr)

    return all_values, all_likelihood
"""

################################################################################
# TRAINING
################################################################################

class PG_GAN:

    def __init__(self, generator, disc, iterator, iterable_params, seed, disc_lr=1e-4, gp_weight=10.0):
        """Setup the model and training framework"""

        # set up generator and discriminator
        self.generator = generator
        self.discriminator = disc
        self.iterator = iterator # for training data (real or simulated)
        self.iterable_params = iterable_params

        # this checks and prints the model (1 is for the batch size)
        self.discriminator.build_graph((1, iterator.num_samples,
            global_vars.NUM_SNPS, NUM_CHANNELS))
        self.discriminator.summary()

        self.gp_weight = gp_weight
        self.disc_optimizer = tf.keras.optimizers.Adam(learning_rate=disc_lr, beta_1=0, beta_2=0.9)

    def train_sa(self, num_batches):
        """Train using fake_values for the simulated data"""

        for epoch in range(num_batches):

            real_regions = self.iterator.real_batch(neg1 = True)
            disc_loss, logs = self.train_step(real_regions)

            if wandb.run:
                wandb.log(logs)

            # if (epoch+1) % 1 == 0:
            #     template = 'Epoch {}, Loss: {}'
            #     print(template.format(epoch + 1, disc_loss))

    def generator_loss(self, proposed_params):
        """ Generator loss (Wasserstein) """
        generated_regions = self.generator.simulate_batch(params=proposed_params)
        fake_output = self.discriminator(generated_regions, training=False)
        loss = -tf.reduce_mean(fake_output)
        return loss.numpy()

    def discriminator_loss(self, real_output, fake_output, gp):
        """ Discriminator loss (Wasserstein + gradient penalty) """
        real_score = tf.reduce_mean(real_output)
        fake_score = tf.reduce_mean(fake_output)
        wasserstein_loss = fake_score - real_score
        total_loss = wasserstein_loss + self.gp_weight * gp

        return total_loss, {
            "real_score": real_score,
            "fake_score": fake_score,
            "wasserstein_distance": real_score - fake_score,
            "gradient_penalty": gp,
            "discriminator_loss": total_loss,
        }

    def gradient_penalty(self, real_samples, fake_samples):
        batch_size = tf.shape(real_samples)[0]
        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
        interpolated = real_samples + alpha * (fake_samples - real_samples)
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)
        grads = tape.gradient(pred, interpolated)
        grads = tf.reshape(grads, [batch_size, -1])
        gp = tf.reduce_mean((tf.norm(grads, axis=1) - 1.0) ** 2)
        return gp
    
    def train_step(self, real_regions):
        with tf.GradientTape() as disc_tape:
            generated_regions = self.generator.simulate_batch()
            real_output = self.discriminator(real_regions, training=True)
            fake_output = self.discriminator(generated_regions, training=True)

            gp = self.gradient_penalty(real_regions, generated_regions)
            disc_loss, logs = self.discriminator_loss(real_output, fake_output, gp)
        gradients_of_discriminator = disc_tape.gradient(disc_loss,
            self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator,
            self.discriminator.trainable_variables))
        return disc_loss, logs

################################################################################
# EXTRA UTILITIES
################################################################################

def get_discriminator(sample_sizes, seed):
    num_pops = len(sample_sizes)
    if num_pops == 1:
        return discriminator.OnePopModel(pop=sample_sizes[0], fc_size=64)
    if num_pops == 2:
        return discriminator.TwoPopModel(sample_sizes[0], sample_sizes[1])
    if num_pops == 3:
        return discriminator.ThreePopModel(sample_sizes[0], sample_sizes[1],
            sample_sizes[2])

    raise NotImplementedError

if __name__ == "__main__":
    main()
