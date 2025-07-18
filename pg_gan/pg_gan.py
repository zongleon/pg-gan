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
import scipy.stats

# our imports
from . import discriminator
from . import global_vars
from . import util

# globals for simulated annealing
NUM_ITER = 300
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

DISC_PATH = "/homes/smathieson/Documents/pg_gan_interpret/"

def reset_random_seeds(seed):
   os.environ['PYTHONHASHSEED']=str(seed)
   tf.random.set_seed(seed)
   np.random.seed(seed)
   random.seed(seed)
   
def main():
    """Parse args and run simulated annealing"""

    opts = util.parse_args()
    print(opts)

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
            iterator, iterable_params, opts.seed, toy=opts.toy)

    print(posterior)
    print(loss_lst)

################################################################################
# SIMULATED ANNEALING
################################################################################

def simulated_annealing(generator, disc, iterator, iterable_params, seed,
    toy=False, prefix=None):
    """Main function that drives GAN updates"""

    # main object for pg-gan
    pg_gan = PG_GAN(generator, disc, iterator, iterable_params, seed)

    # find starting point through pre-training (update generator in method)
    if not toy:
        s_current = pg_gan.disc_pretraining(800)
    else:
        pg_gan.disc_pretraining(1) # for testing purposes
        s_current = iterable_params.clone(start=True)
        pg_gan.generator.update_params(s_current)

    loss_curr = pg_gan.generator_loss(s_current)
    print("params, loss", s_current.to_list(), loss_curr)

    posterior = [s_current.to_list()]
    loss_lst = [loss_curr]
    real_acc_lst = []
    fake_acc_lst = []

    # simulated-annealing iterations
    num_iter = NUM_ITER
    # for toy example
    if toy:
        num_iter = 2

    s_current_set = s_current.param_set

    # main pg-gan loop
    for i in range(num_iter):
        print("\nITER", i)
        print("time", datetime.datetime.now().time())
        T = temperature(i, num_iter) # reduce width of proposal over time

        # propose 10 updates per param and pick the best one
        s_best = None
        loss_best = float('inf')

        for param_name in s_current_set: # trying all params!
            value = s_current_set[param_name].value
            s_proposal = s_current.clone()

            #k = random.choice(range(len(parameters))) # random param
            for j in range(10): # trying 10

                # can update all the parameters at once, or choose one at a time
                # s_proposal.proposal_all(multiplier=T, value_dict=parameters.param_set)
                s_proposal.propose_param(param_name, value, T)
                loss_proposal = pg_gan.generator_loss(s_proposal)

                print(j, "proposal", s_proposal.to_list(), loss_proposal)
                if loss_proposal < loss_best: # minimizing loss
                    loss_best = loss_proposal
                    s_best = s_proposal.clone()

        # decide whether to accept or not (reduce accepting bad state later on)
        if loss_best <= loss_curr: # unsure about this equal here
            p_accept = 1
        else:
            p_accept = (loss_curr / loss_best) * T
        rand = np.random.rand()
        accept = rand < p_accept

        # if accept, retrain
        if accept:
            print("ACCEPTED")
            s_current = s_best
            pg_gan.generator.update_params(s_current)
            # train only if accept
            real_acc, fake_acc = pg_gan.train_sa(NUM_BATCH)
            loss_curr = loss_best

        # don't retrain
        else:
            print("NOT ACCEPTED")

        print("T, p_accept, rand, s_current, loss_curr", end=" ")
        print(T, p_accept, rand, s_current.to_list(), loss_curr)
        posterior.append(s_current.to_list())
        loss_lst.append(loss_curr)

    if prefix is not None:
        pg_gan.discriminator.save(DISC_PATH + prefix + ".keras")
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

    def __init__(self, generator, disc, iterator, iterable_params, seed):
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

        self.cross_entropy =tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.disc_optimizer = tf.keras.optimizers.Adam()

    def disc_pretraining(self, num_batches):
        """Pre-train so discriminator has a chance to learn before generator"""
        s_best = []
        max_acc = 0
        k = 0 if num_batches > 1 else 9 # limit iterations for toy/testing

        # try with several random sets at first
        while max_acc < 0.9 and k < 10:
            s_trial = self.iterable_params.clone(start=True)
            print("trial", k+1, s_trial)
            self.generator.update_params(s_trial)
            real_acc, fake_acc = self.train_sa(num_batches)
            avg_acc = (real_acc + fake_acc)/2
            if avg_acc > max_acc:
                max_acc = avg_acc
                s_best = s_trial
            k += 1

        # now start!
        self.generator.update_params(s_best)
        return s_best

    def train_sa(self, num_batches):
        """Train using fake_values for the simulated data"""

        for epoch in range(num_batches):

            real_regions = self.iterator.real_batch(neg1 = True)
            real_acc, fake_acc, disc_loss = self.train_step(real_regions)

            if (epoch+1) % 100 == 0:
                template = 'Epoch {}, Loss: {}, Real Acc: {}, Fake Acc: {}'
                print(template.format(epoch + 1,
                                disc_loss,
                                real_acc/global_vars.BATCH_SIZE * 100,
                                fake_acc/global_vars.BATCH_SIZE * 100))

        return real_acc/global_vars.BATCH_SIZE, fake_acc/global_vars.BATCH_SIZE

    def generator_loss(self, proposed_params):
        """ Generator loss """
        generated_regions = self.generator.simulate_batch(params=proposed_params)
        # not training when we use the discriminator here
        fake_output = self.discriminator(generated_regions, training=False)
        loss = self.cross_entropy(tf.ones_like(fake_output), fake_output)

        return loss.numpy()

    def discriminator_loss(self, real_output, fake_output):
        """ Discriminator loss """
        # accuracy
        real_acc = np.sum(real_output >= 0) # positive logit => pred 1
        fake_acc = np.sum(fake_output <  0) # negative logit => pred 0

        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss

        # add on entropy regularization (small penalty)
        real_entropy = scipy.stats.entropy(tf.nn.sigmoid(real_output))
        fake_entropy = scipy.stats.entropy(tf.nn.sigmoid(fake_output))
        entropy = tf.math.scalar_mul(0.001/2, tf.math.add(real_entropy,
            fake_entropy)) # can I just use +,*? TODO experiement with constant

        return total_loss, real_acc, fake_acc

    def train_step(self, real_regions):
        """One mini-batch for the discriminator"""

        with tf.GradientTape() as disc_tape:
            # use current params
            generated_regions = self.generator.simulate_batch()

            real_output = self.discriminator(real_regions, training=True)
            fake_output = self.discriminator(generated_regions, training=True)

            disc_loss, real_acc, fake_acc = self.discriminator_loss(
                real_output, fake_output)

        # gradient descent
        gradients_of_discriminator = disc_tape.gradient(disc_loss,
            self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator,
            self.discriminator.trainable_variables))

        return real_acc, fake_acc, disc_loss

################################################################################
# EXTRA UTILITIES
################################################################################

def get_discriminator(sample_sizes, seed):
    num_pops = len(sample_sizes)
    if num_pops == 1:
        return discriminator.OnePopModel()# sample_sizes[0], seed) SM: removing for now to work with saving disc
    if num_pops == 2:
        return discriminator.TwoPopModel(sample_sizes[0], sample_sizes[1])
    # else
    return discriminator.ThreePopModel(sample_sizes[0], sample_sizes[1],
        sample_sizes[2])

if __name__ == "__main__":
    main()
