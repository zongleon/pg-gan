'''For collecting global values'''
# section A: general -----------------------------------------------------------
NUM_SNPS = 36       # number of seg sites, should be divisible by 4
L = 50000           # heuristic to get enough SNPs for simulations (50,000 or fifty-thousand)
BATCH_SIZE = 50

DEFAULT_SEED = 1833
DEFAULT_SAMPLE_SIZE = 198

FRAC_TEST = 0.1 # depricated

# section B: overwriting in-file data-------------------------------------------

# to use custom trial data, switch OVERWRITE_TRIAL_DATA to True and
# change the TRIAL_DATA dictionary to have the values desired.
# Model, params, and param_values must be defined
OVERWRITE_TRIAL_DATA = True
TRIAL_DATA = { 'model': 'exp', 'params': 'N1,N2,growth,T1,T2', 'data_h5': 'data/genomes/CEU.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.h5',
               'bed_file': 'data/genomes/20120824_strict_mask.bed', 'reco_folder': None, 'param_values': '21558.869742709805, 5896.142076568601, 0.0010594146557898422, 3650.272293584133, 1014.233835438603',
               'seed': None, 'sample_sizes': '198'}

# section C: summary stats customization----------------------------------------
SS_SHOW_TITLE = False

COLOR_DICT = {"YRI": "darkorange", "ESN": "darkorange", "CEU": "blue",
              "GBR": "blue", "CHB": "green", "CHS": "green", "MXL": "red",
              "simulation": "gray", "msprime": "purple",
              "GHIST-bottleneck": "green", "GHIST-split-isolation": "blue"}

SS_LABELS = []
SS_COLORS = []
'''
Override by commenting out the function body,
and adding in your definitions. Leave the assert
at the end.
'''
def update_ss_labels(pop_names, num_pops = 1):
    # SS_LABELS is a list of string labels, ex ["CEU", "YRI", "CHB", "simulation"]
    # or ["msprime", "SLiM"]
    if pop_names == "":
        if num_pops == 1:
            pop_labels = ["msprime"]
        else: # works for up to 7 populations
            pop_labels = [None  for i in range(num_pops)]
            ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVQXYZ"
            ALT_COLORS = ["red", "blue", "darkorange", "green",
                          "purple", "pink", "navy"]

            for i in range(num_pops):
                name = "POP_"+ALPHABET[i]
                pop_labels[i] = name
                COLOR_DICT[name] = ALT_COLORS[i]

    else:
        pop_labels = pop_names.split("_")

    SS_LABELS.extend(pop_labels[:1]) # TODO hack
    SS_LABELS.append("simulation")
    #print(SS_LABELS)

    # colors for plotting, ex ["blue", "darkorange", "green", "gray"] (last is traditionally gray)
    for label in SS_LABELS:
        SS_COLORS.append(COLOR_DICT[label])

    assert len(SS_LABELS) == len(SS_COLORS)

# Section D: alternate data format options--------------------------------------

HUMAN_CHROM_RANGE = range(1, 23) # Human chroms, 1000G doesn't use XY

'''
Rewrite this function to appropriately collect a list of
reco files. Not called if reco_folder isn't provided.

The file list can be defined directly for ease, i.e.
files = ["file1", "file2", ... ]
'''
def get_reco_files(reco_folder):
    # DEFAULT IS FOR hg19 FORMAT
    files = [reco_folder + "genetic_map_GRCh37_chr" + str(i) +
             ".txt" for i in HUMAN_CHROM_RANGE]

    # for high coverage/ hg38, comment the above line, and uncomment the following:
    # pop = reco_folder[-4: -1]
    # files = [reco_folder + pop + "_recombination_map_hapmap_format_hg38_chr_" + str(i) +
    #          ".txt" for i in HUMAN_CHROM_RANGE]

    return files

'''
Likewise, overwrite for parsing for your datafile
'''
def parse_chrom(chrom):
    if isinstance(chrom, bytes):
        return chrom.decode("utf-8")

    return chrom # hg19 option

    # for hg38, replace the above with
    # return chrom[3:]

'''The high-coverage data ("new data") appears to have partial filtering on
singletons. It is recommended, if using the high-coverage data, to enable
singleton filtering for both real and simulated data. It may be necessary to
experiment with different filtering rates.'''
FILTER_SIMULATED = False
FILTER_REAL_DATA = False
FILTER_RATE = 0.50
NUM_SNPS_ADJ = NUM_SNPS * 3
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # testing
    print(SS_LABELS)
    print(SS_COLORS)
    update_ss_labels("CEU")
    print(SS_LABELS)
    print(SS_COLORS)
