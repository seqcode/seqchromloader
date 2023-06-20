"""
Utilities for iterating constructing data sets and iterating over
DNA sequence data.
"""
import math
import pandas as pd
import numpy as np
import logging
import pysam
import pyBigWig
from multiprocessing import Pool
from pybedtools import Interval, BedTool
from pybedtools.helpers import chromsizes

def get_genome_sizes(gs=None, genome=None, to_filter=None, to_keep=None):
    """
    Loads the genome sizes file, filter or keep chromosomes
    
    :param gs: genome size file path, only one of `gs` and `genome` is required
    :type gs: string
    :param genome: genome build name, only one of `gs` and `genome` is required
    :type genome: string
    :param to_filter: list of chrmosomes to filter, mutually exclusive with `to_keep`
    :type to_filter: list
    :param to_keep: list of chromosomes to keep, mutually exclusive with `to_filter`
    :type to_keep: list
    """
    
    if gs:
        genome_sizes = pd.read_table(gs, header=None, usecols=[0,1], names=['chrom', 'length'])
    elif genome:
        genome_sizes = (pd.DataFrame(chromsizes(genome))
                        .T
                        .reset_index()
                        .rename(columns={"index":"chrom", 0:"start", 1:"end"})
                        .assign(length=lambda x: x["end"] - x["start"]))[["chrom", "length"]]
    else:
        raise Exception("Either gs or genome should be provided!")

    genome_sizes_filt = filter_chromosomes(genome_sizes, to_filter=to_filter, to_keep=to_keep)

    return genome_sizes_filt

def filter_chromosomes(coords, to_filter=None, to_keep=None):
    """
    Filter or keep the specified chromosomes
    
    :param coords: input coordinate dataframe, first column should be `chrom`
    :type coords: pandas.DataFrame
    :param to_filter: list of chrmosomes to filter, mutually exclusive with `to_keep`
    :type to_filter: list
    :param to_keep: list of chromosomes to keep, mutually exclusive with `to_filter`
    :type to_keep: list
    :rtype: filtered coordinate dataframe
    """
    
    if to_filter and to_keep:
        print("Both to_filter and to_keep are provided, only to_filter will work under this circumstance!")
    
    if to_filter:
        corods_out = coords.copy()
        for chromosome in to_filter:
            # note: using the str.contains method to remove all
            # contigs; for example: chrUn_JH584304
            bool_filter = ~(corods_out['chrom'].str.contains(chromosome))
            corods_out = corods_out[bool_filter]
    elif to_keep:
        # keep only the to_keep chromosomes:
        # note: this is slightly different from to_filter, because
        # at a time, if only one chromosome is retained, it can be used
        # sequentially.
        filtered_chromosomes = []
        for chromosome in to_keep:
            filtered_record = coords[(coords['chrom'] == chromosome)]
            filtered_chromosomes.append(filtered_record)
        # merge the retained chromosomes
        corods_out = pd.concat(filtered_chromosomes)
    else:
        corods_out = coords
    return corods_out

def make_random_shift(coords, L, buffer=0):
    """
    This function takes as input a set of bed coordinates dataframe 
    It finds the mid-point for each record or Interval in the bed file,
    shifts the mid-point, and generates a windows of length L.

    If training window length is L, then we must ensure that the
    peak center is still within the training window.
    Therefore: -L/2 < shift < L/2
    To add in a buffer: -L/2 + buffer <= shift <= L/2 - 25
    
    :param coords: input coordinate dataframe, first 3 columns are: [chrom, start, end]
    :type coords: pandas.DataFrame
    :param L: training/shifting window size, the shifted midpoint should fall into the range -L/2 + buffer ~ L/2 + buffer
    :type L: integer
    :param buffer: buffer size
    :type buffer: integer
    :rtype: shifted coordinate dataframe
    
    """
    low = int(-L/2 + buffer)
    high = int(L/2 - buffer + 1)

    return (coords.assign(midpoint=lambda x: (x["start"]+x["end"])/2)
            .astype({"midpoint": int})
            .assign(midpoint=lambda x: x["midpoint"] + np.random.randint(low=low, high=high, size=len(coords)))
            .apply(lambda s: pd.Series([s["chrom"], int(s["midpoint"]-L/2), int(s["midpoint"]+L/2)],
                                        index=["chrom", "start", "end"]), axis=1))

def make_flank(coords, L, d):
    """
    Make flanking regions by:
    1. Shift midpoint by d
    2. Expand midpoint to upstream/downstream by L/2
    
    :param coords: input coordinate dataframe, first 3 columns are: [chrom, start, end]
    :type coords: pandas.DataFrame
    :param L: window size of output regions
    :type L: integer
    :param d: shifting distance
    :type d: integer
    :rtype: shifted coordinate dataframe
    """
    return (coords.assign(midpoint=lambda x: (x["start"]+x["end"])/2)
                .astype({"midpoint": int})
                .assign(midpoint=lambda x: x["midpoint"] + d)
                .apply(lambda s: pd.Series([s["chrom"], int(s["midpoint"]-L/2), int(s["midpoint"]+L/2)],
                                            index=["chrom", "start", "end"]), axis=1))

    
def random_coords(gs:str=None, genome:str=None, incl:BedTool=None, excl:BedTool=None, l=500, n=1000):
    """
    Randomly sample n intervals of length l from the genome,
    shuffle to make all intervals inside the desired regions 
    and outside exclusion regions
    
    :param gs: genome size file path, only one of `gs` and `genome` is required
    :type gs: string
    :param genome: genome build name, only one of `gs` and `genome` is required
    :type genome: string
    :param excl: regions that chopped regions should overlap with
    :type incl: BedTool object
    :param incl: regions that chopped regions shouldn't overlap with
    :type excl: BedTool object
    :param l: random interval size
    :type l: integer
    :param n: number of random intervals generated
    :type n: integer
    """
    random_kwargs = {}
    shuffle_kwargs = {}
    if genome:
        random_kwargs.update({"genome": genome}); shuffle_kwargs.update({"genome": genome})
    elif gs:
        random_kwargs.update({"g": gs}); shuffle_kwargs.update({"g": gs})
    else:
        raise Exception("Either gs or genome should be provided!")
    
    if incl: shuffle_kwargs.update({"incl": incl.fn})
    if excl: shuffle_kwargs.update({"excl": excl.fn})
    
    return (BedTool()
            .random(l=l, n=n, **random_kwargs)
            .shuffle(**shuffle_kwargs)
            .to_dataframe()[["chrom", "start", "end"]])

def chop_genome(chroms:list=None, incl:BedTool=None, excl:BedTool=None, gs=None, genome=None, stride=500, l=500):
    """
    Given a genome size file and chromosome list,
    chop these chromosomes into intervals of length l,
    with include/exclude regions specified
    
    :param chroms: list of chromosomes to be chopped
    :type chroms: list of strings
    :param excl: regions that chopped regions should overlap with
    :type incl: BedTool object
    :param incl: regions that chopped regions shouldn't overlap with
    :type excl: BedTool object
    :param gs: genome size file path, only one of `gs` and `genome` is required
    :type gs: string
    :param genome: genome build name, only one of `gs` and `genome` is required
    :type genome: string
    :param stride: step size that everytime chopped region start coordinate moving forward by
    :type stride: int
    :param l: interval window size
    :type l: int
    :rtype: chopped intervals coordinate dataframe
    """
    def intervals_loop(chrom, start, stride, l, size):
        intervals = []
        while True:
            if (start + l) < size:
                intervals.append((chrom, start, start+l))
            else:
                break
            start += stride
        return pd.DataFrame(intervals, columns=["chrom", "start", "end"])
    
    genome_sizes = get_genome_sizes(gs=gs, genome=genome, to_keep=chroms)
    
    genome_chops = pd.concat([intervals_loop(i.chrom, 0, stride, l, i.length) 
                                for i in genome_sizes.itertuples()])
    genome_chops_bdt = BedTool.from_dataframe(genome_chops)
    
    if incl: genome_chops_bdt = genome_chops_bdt.intersect(incl, wa=True)
    if excl: genome_chops_bdt = genome_chops_bdt.intersect(excl, v=True)

    return genome_chops_bdt.to_dataframe()[["chrom", "start", "end"]]

def compute_mean_std_bigwig(bigwig):
    """
    
    Compute the overall mean and standard deviation of a given bigwig file

    :param bigwig: bigwig file path
    :type bigwig: str
    :rtype: (mean, stddev)
    """
    bw = pyBigWig.open(bigwig)
    
    # get chrom length list
    chroms = bw.chroms()
    
    # iterate chrom list to get mean and std
    ns = []; means = []; stds = []
    for chrom, length in chroms.items():
        # iterate all intervals to get the covered length
        length = sum([i[1]-i[0] for i in bw.intervals(chrom)])
        ns.append(length)
        means.append(bw.stats(chrom, type="mean", exact=True)[0])
        try:
            stds.append(bw.stats(chrom, type="std", exact=True)[0])
        except RuntimeError:
            print(chrom)
            print(length)
            raise Exception
    
    # compute overall metrics
    std_all = 0
    ns_sum = sum(ns); mean_all = sum([n*means[i] for i, n in enumerate(ns)])/ns_sum
    for i, n in enumerate(ns):
        std_all += n* math.pow(means[i]- mean_all, 2)
        std_all += n* math.pow(stds[i], 2)
    std_all /= ns_sum
    std_all = math.sqrt(std_all)
    
    return mean_all, std_all

class DNA2OneHot(object):
    def __init__(self):
        self.DNA2Index = {
            "A": 0,
            "C": 1, 
            "G": 2,
            "T": 3,
        }
    
    def __call__(self, dnaSeq):
        seqLen = len(dnaSeq)
        # initialize the matrix as 4 x len(dnaSeq)
        seqMatrix = np.zeros((4, len(dnaSeq)), dtype=np.float32)
        # change the value to matrix
        dnaSeq = dnaSeq.upper()
        for j in range(0, seqLen):
            if dnaSeq[j] == "N": continue
            try:
                seqMatrix[self.DNA2Index[dnaSeq[j]], j] = 1
            except KeyError as e:
                print(f"Keyerror happened at position {j}: {dnaSeq[j]}, legal keys are: [A, C, G, T, N]")
                continue
        return seqMatrix

DNA2Index = {
        "A": 0,
        "C": 1, 
        "G": 2,
        "T": 3,
    }
    
def dna2OneHot(dnaSeq):
    """
    One-hot code input DNA sequence into an array of shape (4, len)
    Mapping is in the order of ACGT
    
    :param dnaSeq: input DNA sequence
    :type dnaSeq: string
    :rtype: numpy array of shape (4, len)
    """
    seqLen = len(dnaSeq)
    # initialize the matrix as 4 x len(dnaSeq)
    seqMatrix = np.zeros((4, len(dnaSeq)), dtype=np.float32)
    # change the value to matrix
    dnaSeq = dnaSeq.upper()
    for j in range(0, seqLen):
        if dnaSeq[j] == "N": continue
        try:
            seqMatrix[DNA2Index[dnaSeq[j]], j] = 1
        except KeyError as e:
            print(f"Keyerror happened at position {j}: {dnaSeq[j]}, legal keys are: [A, C, G, T, N]")
            continue
    return seqMatrix

def rev_comp(inp_str):
    """
    Return reverse complemented sequence
    
    :param input_str: input DNA sequence
    :type input_str: string
    :rtype: reverse complemented DNA sequence
    """
    rc_dict = {'A': 'T', 'G': 'C', 'T': 'A', 'C': 'G', 'c': 'g',
               'g': 'c', 't': 'a', 'a': 't', 'n': 'n', 'N': 'N'}
    outp_str = list()
    for nucl in inp_str:
        outp_str.append(rc_dict[nucl])
    return ''.join(outp_str)[::-1]  

class BigWigInaccessible(Exception):
    def __init__(self, chrom, start, end, *args):
        super().__init__(*args)
        self.chrom = chrom
        self.start = start
        self.end = end

    def __str__(self) -> str:
        return f'Chromatin Info Inaccessible in region {self.chrom}:{self.start}-{self.end}'
    
def extract_bw(chrom, start, end, bigwigs, strand="+"):
    chroms_array = []
    try:
        for idx, bigwig in enumerate(bigwigs):
            c = (np.nan_to_num(bigwig.values(chrom, start, end))).astype(np.float32)
            if strand=="-":
                c = c[::-1] 
            chroms_array.append(c)
    except RuntimeError as e:
        logging.warning(e)
        logging.warning(f"RuntimeError happened when accessing {chrom}:{start}-{end}, it's probably due to at least one chromatin track bigwig doesn't have information in this region")
        raise BigWigInaccessible(chrom, start, end)
    chroms_array = np.vstack(chroms_array)  # create the chromatin track array, shape (num_tracks, length)
    
    return chroms_array

def extract_dnaOneHot(chrom, start, end, strand, genome_pyfaidx):
    seq = genome_pyfaidx[chrom][int(start):int(end)].seq
    if strand=="-":
        seq = rev_comp(seq)
    seq_array = dna2OneHot(seq)
    
    return seq_array

def extract_target(chrom, start, end, strand, target):
    if isinstance(target, pysam.AlignmentFile):
        target_array = np.array(target.count(chrom, start, end), dtype=np.float32)[np.newaxis]
    elif isinstance(target, pyBigWig.pyBigWig):
        target_array = np.nan_to_num(target.values(chrom, start, end)).astype(np.float32)
        if strand=="-":
            target_array = target_array[::-1]
    else:
        target_array = None
    return target_array

def extract_info(chrom, start, end, label, genome_pyfaidx, bigwigs, target, strand="+", transforms:dict=None):
    seq_array = extract_dnaOneHot(chrom, start, end, strand, genome_pyfaidx)

    #chromatin track
    if bigwigs is not None and len(bigwigs)>0:
        chroms_array = extract_bw(chrom, start, end, strand)
    else:
        chroms_array = None
    # label
    label_array = np.array(label, dtype=np.int32)[np.newaxis]
    # counts
    target_array = extract_target(chrom, start, end, strand, target)

    feature = {
        'seq': seq_array,
        'chrom': chroms_array,
        'target': target_array,
        'label': label_array
    }

    if transforms is not None:
        for k,t in transforms.items(): 
            feature[k] = t(feature[k])
    
    return feature