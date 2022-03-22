from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import tensorflow as tf
import h5py
import random
import sys
import datetime

from . import pg_util as util


# globals
FRAC_CALLABLE = 0.5
L = 50000

NUM_SNPS = 36
print("NUM_SNPS", NUM_SNPS)



def pg_load_data(
    data_h5, bed, batch_size, cut32=False, frac_test=None, \
        chrom_starts=False):

    dataset = PGDataset(
        S=NUM_SNPS,
        filename=data_h5,
        bed_file=bed,
        cut32=cut32,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
    )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class Region:

    def __init__(self, chrom, start_pos, end_pos):
        self.chrom = chrom
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.region_len = end_pos - start_pos # L

    def __str__(self):
        s = str(self.chrom) + ":" + str(self.start_pos) + "-" +str(self.end_pos)
        return s

    def inside_mask(self, mask_dict):
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
            return part_inside/self.region_len >= FRAC_CALLABLE

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
            part_inside += (mask_lst[region_start_idx][1] - \
                mask_lst[region_start_idx][0])

        # add on last if inside
        if end_inside:
            part_inside += (self.end_pos - mask_lst[region_end_idx][0])
        elif self.end_pos <= mask_lst[region_start_idx][0]:
            # end before closest region, don't add anything
            pass
        else:
            part_inside += (mask_lst[region_end_idx][1] - \
                mask_lst[region_end_idx][0])

        return part_inside/self.region_len >= FRAC_CALLABLE

def read_mask(filename):
    """Read from bed file"""

    mask_dict = {}
    f = open(filename,'r')

    for line in f:
        tokens = line.split()
        chrom_str = tokens[0][3:]
        if chrom_str != 'X' and chrom_str != 'Y':
            chrom = int(chrom_str)
            begin = int(tokens[1])
            end = int(tokens[2])

            if chrom in mask_dict:
                mask_dict[chrom].append([begin,end])
            else:
                mask_dict[chrom] = [[begin,end]]

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



class PGDataset(Dataset):

    def __init__(self, S, filename, bed_file, cut32 = False, frac_test=None, \
        chrom_starts=False):
        callset = h5py.File(filename, mode='r')
        # output: ['GT'] ['CHROM', 'POS']
        print(list(callset['calldata'].keys()),list(callset['variants'].keys()))

        self.S = S
        raw = callset['calldata/GT']
        print("raw", raw.shape)
        newshape = (raw.shape[0], -1)
        self.haps_all = np.reshape(raw, newshape)
        self.pos_all = callset['variants/POS']
        # same length as pos_all, noting chrom for each variant (sorted)
        self.chrom_all = callset['variants/CHROM']
        print("after haps", self.haps_all.shape)
        self.num_samples = self.haps_all.shape[1]

        '''print(self.pos_all.shape)
        print(self.pos_all.chunks)
        print(self.chrom_all.shape)
        print(self.chrom_all.chunks)'''
        self.num_snps = len(self.pos_all) # total for all chroms

        self.mask_dict = read_mask(bed_file) # mask

        self.cut32 = cut32

        # useful for fastsimcoal and msmc
        if chrom_starts:
            self.chrom_counts = defaultdict(int)
            for x in list(self.chrom_all):
                self.chrom_counts[int(x)] += 1

    def find_end(self, start_idx, region_len):
        """
        Based on the given start_idx and the region_len, find the end index
        """
        ln = 0
        chr = self.chrom_all[start_idx]
        i = start_idx
        curr_pos = self.pos_all[start_idx]
        while ln < region_len:

            if len(self.pos_all) <= i+1:
                print("not enough on chrom", chr)
                return -1 # not enough on last chrom

            next_pos = self.pos_all[i+1]
            if self.chrom_all[i+1] == chr:
                diff = next_pos - curr_pos
                ln += diff
            else:
                print("not enough on chrom", chr)
                return -1 # not enough on this chrom
            i += 1
            curr_pos = next_pos

        return i # exclusive

    def real_region(self, neg1, region_len):
        start_idx = random.randrange(self.num_snps-self.S) # inclusive

        # go by region len or by SNPs
        if region_len:
            end_idx = self.find_end(start_idx, L)
            if end_idx == -1:
                return self.real_region(neg1, region_len) # try again
        else:
            end_idx = start_idx + self.S # exclusive

        hap_data = self.haps_all[start_idx:end_idx, :]
        start_base = self.pos_all[start_idx]
        end_base = self.pos_all[end_idx]
        positions = self.pos_all[start_idx:end_idx]

        # make sure we don't span two chroms
        start_chrom = self.chrom_all[start_idx]
        end_chrom = self.chrom_all[end_idx-1] # inclusive here
        if start_chrom != end_chrom:
            #print("bad chrom", start_chrom, end_chrom)
            return self.real_region(neg1, region_len) # try again

        region = Region(int(start_chrom), start_base, end_base)
        result = region.inside_mask(self.mask_dict)

        # if we do have an accessible region
        if result:
            dist_vec = [0] + [(positions[j+1] - positions[j])/L for j in \
                range(len(positions)-1)]

            after = util.process_gt_dist(hap_data, dist_vec, len(dist_vec), \
                neg1=neg1)
            return after

        # try again if not in accessible region
        return self.real_region(neg1, region_len)

    def real_batch(self, batch_size, neg1=True, region_len=False):
        """Use region_len=True for fixed region length, not by SNPs"""

        # if not region_len:
        #     regions = np.zeros((batch_size, self.num_samples, self.S, 2), \
        #         dtype=np.float32)

        #     for i in range(batch_size):
        #         regions[i] = self.real_region(neg1, region_len)

        # else:
        #     regions = []
        #     for i in range(batch_size):
        #         regions.append(self.real_region(neg1, region_len))

        return self.real_region(neg1, region_len)

        # return regions

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

    def __len__(self):
        return L

    def __getitem__(self, idx):
        neg1=True
        region_len=False
        out = self.real_batch(1, True)
        if self.cut32: out = out[:32,:32,:]
        # print("originalshape",out.shape)
        newOut = np.transpose(out, [2, 0, 1])
        # newOut = tf.transpose(out, perm=[2,0,1])
        # newOut = newOut.numpy()

        # ------- make a third channel --------
        # third = []
        # third.append(newOut[0])
        # newOut = np.concatenate((newOut,third),axis=0)

        # for x in np.nditer(newOut, op_flags=['readwrite']):
        #     x[...] = np.round(0.5* np.round(x,1),1)
        newOut = newOut.astype(np.float32)
        out_dict = {}
        # print("outputshape",newOut.shape)
        return newOut, out_dict
