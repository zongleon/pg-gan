"""
Allows us to read in real data regions randomly, and also use a mask (bed
format) file so we don't use regions that are uncallable.
Author: Sara Mathieson, Rebecca Riley
Date: 9/27/22
"""

# python imports
from collections import defaultdict
import h5py
import numpy as np
from numpy.random import default_rng
import sys

# our imports
from . import global_vars
from . import util

class Region:

    def __init__(self, chrom, start_pos, end_pos):
        self.chrom = chrom
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.region_len = end_pos - start_pos # L

    def __str__(self):
        s = str(self.chrom) + ":" + str(self.start_pos) + "-" +str(self.end_pos)
        return s

    def inside_mask(self, mask_dict, frac_callable = 0.5):
        if mask_dict is None:
            return True

        mask_lst = mask_dict[self.chrom] # restrict to this chrom
        region_start_idx, start_inside = binary_search(self.start_pos, mask_lst)
        region_end_idx, end_inside = binary_search(self.end_pos, mask_lst)

        # same region index
        if region_start_idx == region_end_idx:
            if start_inside and end_inside: # both inside
                return True
            elif (not start_inside) and (not end_inside): # both outside
                return False
            elif start_inside:
                part_inside = mask_lst[region_start_idx][1] - self.start_pos
            else:
                part_inside = self.end_pos - mask_lst[region_start_idx][0]
            return part_inside/self.region_len >= frac_callable

        # different region index
        part_inside = 0
        # conservatively add at first
        for region_idx in range(region_start_idx+1, region_end_idx):
            part_inside += (mask_lst[region_idx][1] - mask_lst[region_idx][0])

        # add on first if inside
        if start_inside:
            part_inside += (mask_lst[region_start_idx][1] - self.start_pos)
        elif self.start_pos >= mask_lst[region_start_idx][1]:
            # start after closest region, don't add anything
            pass
        else:
            part_inside += (mask_lst[region_start_idx][1] -
                mask_lst[region_start_idx][0])

        # add on last if inside
        if end_inside:
            part_inside += (self.end_pos - mask_lst[region_end_idx][0])
        elif self.end_pos <= mask_lst[region_end_idx][0]:
            # end before closest region, don't add anything
            pass
        else:
            part_inside += (mask_lst[region_end_idx][1] -
                mask_lst[region_end_idx][0])

        return part_inside/self.region_len >= frac_callable

def read_mask(filename):
    """Read from bed file"""

    mask_dict = {}
    f = open(filename,'r')

    for line in f:
        tokens = line.split()
        chrom_str = tokens[0][3:]
        if chrom_str != 'X' and chrom_str != 'Y':
            begin = int(tokens[1])
            end = int(tokens[2])

            if chrom_str in mask_dict:
                mask_dict[chrom_str].append([begin,end])
            else:
                mask_dict[chrom_str] = [[begin,end]]

    f.close()
    return mask_dict

def binary_search(q, lst):
    low = 0
    high = len(lst)-1

    while low <= high:

        mid = (low+high)//2
        if lst[mid][0] <= q <= lst[mid][1]: # inside region
            return mid, True
        elif q < lst[mid][0]:
            high = mid-1
        else:
            low = mid+1

    return mid, False # something close

class RealDataRandomIterator:

    def __init__(self, filename, seed, bed_file=None, chrom_starts=False):
        callset = h5py.File(filename, mode='r')
        print(list(callset.keys()))
        # output: ['GT'] ['CHROM', 'POS']
        print(list(callset['calldata'].keys()),list(callset['variants'].keys()))

        raw = callset['calldata/GT']
        print("raw", raw.shape)

        # does the right thing for unphased data
        newshape = (raw.shape[0], -1)
        self.haps_all = np.reshape(raw, newshape)

        # check counts
        #print("less than 0", np.count_nonzero(self.haps_all[self.haps_all<0]))
        #print("greater than 1", np.count_nonzero(self.haps_all[self.haps_all>1]))
        #input('enter')

        self.haps_all[self.haps_all<0] = 0 # -1 is missing, replacing w/ 0 for now but need a better way! TODO
        self.pos_all = callset['variants/POS']

        # same length as pos_all, noting chrom for each variant (sorted)
        self.chrom_all = callset['variants/CHROM']
        print("after haps", self.haps_all.shape)
        self.num_samples = self.haps_all.shape[1]
        #if not phased:
        #    assert self.num_samples % 2 == 0
        #    self.num_samples = self.num_samples//2

        '''print(self.pos_all.shape)
        print(self.pos_all.chunks)
        print(self.chrom_all.shape)
        print(self.chrom_all.chunks)'''
        self.num_snps = len(self.pos_all) # total for all chroms
        #self.phased = phased

        # mask
        self.mask_dict = read_mask(bed_file) if bed_file is not None else None

        self.rng = default_rng(seed)

        # useful for fastsimcoal and msmc
        if chrom_starts:
            self.chrom_counts = defaultdict(int)
            for x in list(self.chrom_all):
                self.chrom_counts[int(x)] += 1

    def find_end(self, start_idx):
        """
        Based on the given start_idx and the region_len, find the end index
        """
        ln = 0
        chr = self.chrom_all[start_idx]
        i = start_idx
        curr_pos = self.pos_all[start_idx]
        while ln < global_vars.L:

            if len(self.pos_all) <= i+1:
                chr_str = chr.decode("utf-8") if isinstance(chr, bytes) else chr
                print("not enough on chrom", chr_str)
                return -1 # not enough on last chrom

            next_pos = self.pos_all[i+1]
            if self.chrom_all[i+1] == chr:
                diff = next_pos - curr_pos
                ln += diff
            else:
                chr_str = chr.decode("utf-8") if isinstance(chr, bytes) else chr
                print("not enough on chrom", chr_str)
                return -1 # not enough on this chrom
            i += 1
            curr_pos = next_pos

        return i # exclusive

    def find_endpoints(self, mid_idx, region_L):
        """
        Based on the given mid_idx and the region_L, find the start/end
        """
        chr = self.chrom_all[mid_idx]

        ln_after = 0
        end_idx = mid_idx
        curr_pos = self.pos_all[mid_idx]
        while ln_after < region_L/2:

            if len(self.pos_all) <= end_idx+1:
                chr_str = chr.decode("utf-8") if isinstance(chr, bytes) else chr
                print("not enough on chrom", chr_str)
                return -1, -1 # not enough on last chrom

            next_pos = self.pos_all[end_idx+1]
            if self.chrom_all[end_idx+1] == chr:
                diff = next_pos - curr_pos
                ln_after += diff
            else:
                chr_str = chr.decode("utf-8") if isinstance(chr, bytes) else chr
                print("not enough on chrom", chr_str)
                return -1, -1 # not enough on this chrom
            end_idx += 1
            curr_pos = next_pos

        ln_before = 0
        start_idx = mid_idx
        curr_pos = self.pos_all[mid_idx]
        while ln_before < region_L/2:

            if start_idx-1 < 0:
                chr_str = chr.decode("utf-8") if isinstance(chr, bytes) else chr
                print("not enough on chrom", chr_str)
                return -1, -1 # not enough on last chrom

            next_pos = self.pos_all[start_idx-1]
            if self.chrom_all[start_idx-1] == chr:
                diff = curr_pos - next_pos
                ln_before += diff
            else:
                chr_str = chr.decode("utf-8") if isinstance(chr, bytes) else chr
                print("not enough on chrom", chr_str)
                return -1, -1 # not enough on this chrom
            start_idx -= 1
            curr_pos = next_pos

        return start_idx, end_idx # exclusive

    def real_region(self, neg1, region_len, start_idx=None, mid_idx=None,
        region_L=None):
        # use region_len = True and mid_idx/region_L together for logit_tajD!

        if start_idx is None and mid_idx is None:
            # inclusive
            start_idx = self.rng.integers(self.num_snps - global_vars.NUM_SNPS)

        if region_len:
            start_idx, end_idx = self.find_endpoints(mid_idx, region_L)
            #end_idx = self.find_end(start_idx)
            if end_idx == -1 or start_idx == -1:
                if start_idx is None:
                    return self.real_region(neg1, region_len) # try again
                else:
                    #print('start or end bad')
                    return None # no recursion if walking through the genome
        else:
            end_idx = start_idx + global_vars.NUM_SNPS # exclusive

        # make sure we don't span two chroms
        start_chrom = self.chrom_all[start_idx]
        end_chrom = self.chrom_all[end_idx-1] # inclusive here

        if start_chrom != end_chrom:
            #print("bad chrom", start_chrom, end_chrom)
            if start_idx is None:
                return self.real_region(neg1, region_len) # try again
            else:
                #print('start end mismatch')
                return None # no recursion if walking through the genome

        #print("start mid end", start_idx, mid_idx, end_idx)
        hap_data = np.copy(self.haps_all[start_idx:end_idx, :])
        start_base = self.pos_all[start_idx]
        end_base = self.pos_all[end_idx]
        positions = self.pos_all[start_idx:end_idx]

        chrom = global_vars.parse_chrom(start_chrom)
        region = Region(chrom, start_base, end_base)
        result = region.inside_mask(self.mask_dict)

        # if we do have an accessible region
        if result:
            # if region_len, then positions_S is actually positions_len
            dist_vec = [0] + [(positions[j+1] - positions[j])/global_vars.L
                for j in range(len(positions)-1)]

            # check neg1
            #unique, counts = np.unique(hap_data, return_counts=True)
            #print("hap_data", dict(zip(unique, counts)))

            after = util.process_gt_dist(hap_data, dist_vec,
                region_len=region_len, real=True, neg1=neg1)
            return after

        # try again if not in accessible region
        if start_idx is None:
            return self.real_region(neg1, region_len) # try again
        else:
            #print('not accessible')
            return None # no recursion if walking through the genome

    def real_batch(self, batch_size = global_vars.BATCH_SIZE, neg1=True,
        region_len=False):
        """Use region_len=True for fixed region length, not by SNPs"""

        if not region_len:
            regions = np.zeros((batch_size, self.num_samples,
                global_vars.NUM_SNPS, 2), dtype=np.float32)

            for i in range(batch_size):
                regions[i] = self.real_region(neg1, region_len)

        else:
            regions = []
            for i in range(batch_size):
                regions.append(self.real_region(neg1, region_len))

        return regions

    def real_chrom(self, chrom, samples):
        """Mostly used for msmc - gather all data for a given chrom int"""
        start_idx = 0
        for i in range(1, chrom):
            start_idx += self.chrom_counts[i]
        end_idx = start_idx + self.chrom_counts[chrom]
        print(chrom, start_idx, end_idx)
        positions = self.pos_all[start_idx:end_idx]

        assert len(samples) == 2 # two populations
        n = self.haps_all.shape[1]
        half = n//2
        pop1_data = self.haps_all[start_idx:end_idx, 0:samples[0]]
        pop2_data = self.haps_all[start_idx:end_idx, half:half+samples[1]]
        hap_data = np.concatenate((pop1_data, pop2_data), axis=1)
        assert len(hap_data) == len(positions)

        return hap_data.transpose(), positions

if __name__ == "__main__":
    # testing

    # test file
    filename = sys.argv[1]
    #bed_file = sys.argv[2]
    iterator = RealDataRandomIterator(filename, global_vars.DEFAULT_SEED) 
        #bed_file)#, phased=False)

    #start_time = datetime.datetime.now()
    #for i in range(10):
    batch = iterator.real_batch()
    print(batch.shape)

    #end_time = datetime.datetime.now()
    #elapsed = end_time - start_time
    #print("time s:ms", elapsed.seconds,":",elapsed.microseconds)

    # test find_end
    #for i in range(10):
    #    start_idx = iterator.rng.integers(iterator.num_snps-global_vars.NUM_SNPS)
    #    iterator.find_end(start_idx)
