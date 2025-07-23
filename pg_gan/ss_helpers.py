"""
Summary stat helpers for computing and plotting summary statistics.
Note: "real" should alwasy be first, followed by simulated.
Author: Sara Mathieson, Rebecca Riley
Date: 9/27/22
"""

# TODO some of these could be replaced with tskit
# https://tskit.readthedocs.io/en/stable/python-api.html#tskit.TreeSequence

# python imports
import allel
import pylibseq as libsequence
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# our imports
from pg_gan import global_vars

# GLOBALS
NUM_SFS = 10
NUM_LD  = 15

################################################################################
# PARSE PG-GAN OUTPUT
################################################################################

def parse_mini_lst(mini_lst):
    return [float(remove_numpy(x.replace("[",'').replace("]",'').replace(",",''))) for x in
        mini_lst]

def remove_numpy(string):
    if "(" in string:
        return string[string.index("(")+1:string.index(")")]
    return string

def add_to_lst(total_lst, mini_lst):
    assert len(total_lst) == len(mini_lst)
    for i in range(len(total_lst)):
        total_lst[i].append(mini_lst[i])

def parse_output(filename, return_acc=False):
    """Parse pg-gan output to find the inferred parameters"""

    def clean_param_tkn(s):
        if s == 'None,':
            return None # this is a common result (not an edge case)

        if s[:-1].isnumeric(): # probably the seed
            # no need to remove quotation marks, just comma
            return int(s[:-1]) # only used as a label, so ok to leave as str

        return s[1:-2]

    f = open(filename,'r')

    # list of lists, one for each param
    param_lst_all = []

    # evaluation metrics
    disc_loss_lst = []
    real_acc_lst = []
    fake_acc_lst = []

    num_param = None

    trial_data = {}

    for line in f:

        if line.startswith("{"):
            tokens = line.split()
            print(tokens)
            param_str = tokens[3][1:-2]
            print("PARAMS", param_str)
            param_names = param_str.split(",")
            num_param = len(param_names)
            for i in range(num_param):
                param_lst_all.append([])

            trial_data['model'] = clean_param_tkn(tokens[1])
            trial_data['params'] = param_str
            trial_data['data_h5'] = clean_param_tkn(tokens[5])
            trial_data['bed_file'] = clean_param_tkn(tokens[7])
            trial_data['reco_folder'] = clean_param_tkn(tokens[9])
            trial_data['seed'] = clean_param_tkn(tokens[15])
            trial_data['sample_sizes'] = clean_param_tkn(tokens[17])
            
        elif "Epoch 100" in line:
            tokens = line.split()
            disc_loss = float(tokens[3][:-1])
            real_acc = float(tokens[6][:-1])/100
            fake_acc = float(tokens[9])/100
            disc_loss_lst.append(disc_loss)
            real_acc_lst.append(real_acc)
            fake_acc_lst.append(fake_acc)

        if "T, p_accept" in line:
            tokens = line.split()
            # parse current params and add to each list
            mini_lst = parse_mini_lst(tokens[-1-num_param:-1])
            add_to_lst(param_lst_all, mini_lst)

    f.close()

    # Use -1 instead of iter for the last iteration
    final_params = [param_lst_all[i][-1] for i in range(num_param)]
    if return_acc:
        return final_params, disc_loss_lst, real_acc_lst, fake_acc_lst, \
            trial_data
    else:
        return final_params, trial_data
        
################################################################################
# COMPUTE STATS
################################################################################

def compute_sfs(vm):
    """Show the beginning of the SFS"""
    ac = vm.count_alleles()
    return [variant_counts(ac,i) for i in range(0,NUM_SFS)]

def variant_counts(ac,c):
    """Helper for SFS"""
    count = 0
    for site in np.array(ac):
        # this should not happen if VCF is pre-processed correctly
        if len(site) > 2:
            print("non-bi-allelic?", site)

        # non-seg can happen after splitting into separate pops
        elif len(site) == 1:
            if c == 0:
                count += 1

        # folded but it shouldn't matter b/c we do major=0 and minor=1
        elif site[0] == c or site[1] == c:
            count += 1

    return count

def compute_ld(vm, L):
    """Compute LD as a function of inter-SNP distance"""

    stringy_data = [''.join(map(str, row)) for row in vm.data.transpose()]
    sd = libsequence.SimData(vm.positions, stringy_data)
    ld = libsequence.ld(sd)

    # num bins
    nbin = NUM_LD
    max_dist = 20000
    dist_bins = np.linspace(0,max_dist,nbin)
    rsquared = [0]*nbin
    counts = [0]*nbin

    # compute bin distances and add ld values to the distance class
    for dict in ld:
        snp_dist = abs(dict['i'] - dict['j'])*L # L is region length
        rsq = dict['rsq']
        idx = np.digitize(snp_dist, dist_bins, right=False)

        # after last num is okay, just goes in last bin
        if idx < 0:
            print(idx, "LD problem!")
        rsquared[idx-1] += rsq
        counts[idx-1] += 1

    # average rsq
    for i in range(nbin):
        if counts[i] > 0:
            rsquared[i] = rsquared[i]/counts[i]
    return rsquared

def compute_stats(vm, vm_region=None):
    """Generic stats for vm (fixed num SNPs) and vm_region (fixed region len)"""

    stats = []
    ac = vm.count_alleles()

    if vm_region is not None:
        ac_region = vm_region.count_alleles()

        # Tajima's D (use region here - not fixed num SNPs)
        stats.append(libsequence.tajd(ac_region))

    # pi
    stats.append(libsequence.thetapi(ac))

    # wattersons
    #stats.append(libsequence.thetaw(ac))

    # num haps
    stats.append(libsequence.number_of_haplotypes(vm))

    return stats

def compute_fst(raw):
    """
    FST (for two populations)
    https://scikit-allel.readthedocs.io/en/stable/stats/fst.html
    """
    # raw has been transposed
    nvar = raw.shape[0]
    nsam = raw.shape[1]
    raw = np.expand_dims(raw, axis=2).astype('i')

    g = allel.GenotypeArray(raw)
    subpops = [range(nsam//2), range(nsam//2, nsam)]

    # for each pop
    ac1 = g.count_alleles(subpop=subpops[0])
    ac2 = g.count_alleles(subpop=subpops[1])

    # compute average fst
    num, den = allel.hudson_fst(ac1, ac2)
    fst = np.sum(num) / np.sum(den)
    return fst

################################################################################
# PLOTTING FUNCTIONS
################################################################################

def plot_generic(ax, name, real, sim, real_color, sim_color, pop="",
    sim_label="", single=False):
    """Plot a generic statistic."""

    # SFS
    if name == "minor allele count (SFS)":
        # average over regions
        num_sfs = len(real)
        real_sfs = [sum(rs)/num_sfs for rs in real]
        sim_sfs = [sum(ss)/num_sfs for ss in sim]

        # plotting (0.3 for offset)
        ax.bar([x-0.3 for x in range(num_sfs)], real_sfs, label=pop, width=0.4,
            color=real_color)
        ax.bar(range(num_sfs), sim_sfs, label=sim_label, width=0.4,
            color=sim_color)
        # SFS x-axis label starts from 0 (monomorphic)
        ax.set_xlim(-1,len(real_sfs))
        ax.set_ylabel("frequency per region")

    # LD
    elif name == "distance between SNPs":
        nbin = NUM_LD
        max_dist = 20000
        dist_bins = np.linspace(0,max_dist,nbin)
        real_mean = [np.mean(rs) for rs in real]
        sim_mean = [np.mean(ss) for ss in sim]
        real_stddev = [np.std(rs) for rs in real]
        sim_stddev = [np.std(ss) for ss in sim]

        # plotting
        ax.errorbar(dist_bins, real_mean, yerr=real_stddev, color=real_color,
            label=pop)
        ax.errorbar([x+150 for x in dist_bins], sim_mean, yerr=sim_stddev,
            color=sim_color, label=sim_label)
        ax.set_ylabel(r'LD ($r^2$)')

    # all other stats
    else:
        sns.histplot(real, ax=ax, color=real_color, label=pop, kde=True,
            stat="density", edgecolor=None)
        sns.histplot(sim, ax=ax, color=sim_color, label=sim_label, kde=True,
            stat="density", edgecolor=None)

    # inter-SNP distances
    if name == "inter-SNP distances":
        ax.set_xlim(-50,1250)
    ax.set(xlabel=name)

    # legend
    if single or name == "Hudson's Fst":
        ax.legend()
    else:
        if len(pop) > 3:
            x_spacing = 0.83
        else:
            x_spacing = 0.85

        ax.text(x_spacing, 0.85, pop, horizontalalignment='center',
            transform=ax.transAxes, fontsize=18)

################################################################################
# COLLECT STATISTICS
################################################################################

def stats_all(matrices, matrices_region=None):
    """Set up and compute stats"""

    # sfs
    pop_sfs = []
    for j in range(NUM_SFS):
        pop_sfs.append([])
    # inter-snp
    pop_dist = []
    # LD
    pop_ld = []
    for j in range(NUM_LD):
        pop_ld.append([])
    # Taj D, pi, num haps
    pop_stats = []
    for j in range(3):
        pop_stats.append([])

    all_per_region = []
        
    # go through each region
    for i in range(len(matrices)):

        per_region_stats = []
        
        # fixed SNPs
        matrix = matrices[i]
        raw = matrix[:,:,0].transpose()
        intersnp = matrix[:,:,1][0] # all the same
        pos = [sum(intersnp[:i]) for i in range(len(intersnp))]
        assert len(pos) == len(intersnp)
        vm = libsequence.VariantMatrix(raw, pos)

        # fixed region
        if matrices_region != None:
            matrix_region = matrices_region[i]
            raw_region = matrix_region[:,:,0].transpose()
            intersnp_region = matrix_region[:,:,1][0] # all the same
            pos_region = [sum(intersnp_region[:i]) for i in
                          range(len(intersnp_region))]
            assert len(pos_region) == len(intersnp_region)
            vm_region = libsequence.VariantMatrix(raw_region, pos_region)
        else:
            vm_region = None
            
        # sfs
        sfs = compute_sfs(vm)
        for s in range(len(sfs)):
            pop_sfs[s].append(sfs[s])
        per_region_stats.extend(sfs)
            
        # inter-snp
        intersnp_rescaled = [round(x*global_vars.L) for x in intersnp]
        pop_dist.extend(intersnp_rescaled)
        per_region_stats.extend(intersnp_rescaled)
        
        # LD
        ld = compute_ld(vm, global_vars.L)
        for l in range(len(ld)):
            pop_ld[l].append(ld[l])
        per_region_stats.extend(ld)
            
        # rest of stats
        stats = compute_stats(vm, vm_region)
        for s in range(len(stats)):
            pop_stats[s].append(stats[s])
        per_region_stats.extend(stats)

        all_per_region.append(per_region_stats)
            
    if matrices_region is None:
        to_np = np.array(all_per_region)
        return to_np

    return [pop_sfs, pop_dist, pop_ld] + pop_stats

def fst_all(matrices):
    """Fst for all regions"""
    real_fst = []
    for i in range(len(matrices)):
        matrix = matrices[i]

        raw = matrix[:,:,0].transpose()
        intersnp = matrix[:,:,1][0] # all the same

        fst = compute_fst(raw)
        real_fst.append(fst)

    return real_fst

STATS = [f'SFS_{i}' for i in range(0, 10)] + \
             [f'inter-SNP_{i}' for i in range(1, 37)] + \
             [f'LD_{i}' for i in range(1, 16)] + ['$\pi$', '#haps']

EXTRA_STATS = ['ihs_maxabs', "tajimas_d", 'garud_h1', 'garud_h12', 'garud_h123', 'garud_h2_h1']

ALL_STATS = STATS + EXTRA_STATS