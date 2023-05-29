"""
Utilities for iterating constructing data sets and iterating over
DNA sequence data.
"""
import pandas as pd
import numpy as np
import logging
from multiprocessing import Pool
from pybedtools import Interval, BedTool
from pybedtools.helpers import chromsizes

def filter_chromosomes(coords, to_filter=None, to_keep=None):
    """
    Filter or keep the specified chromosomes
    
    :param coords: input coordinate dataframe, first 3 columns are: [chrom, start, end]
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

def make_random_shift(coords, L, buffer=25):
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
    high = int(L/2 - buffer)

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


def chop_genome(chroms:list=None, excl:BedTool=None, gs=None, genome=None, stride=500, l=500):
    """
    Given a genome size file and chromosome list,
    chop these chromosomes into intervals of length l,
    with include/exclude regions specified
    
    :param chroms: list of chromosomes to be chopped
    :type chroms: list of strings
    :param excl: regions that chopped regions shouldn't overlap with
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
    
    if genome:
        genome_sizes = (pd.DataFrame(chromsizes(genome))
                        .T
                        .rename(columns={0:"chrom", 1:"len"})
                        .set_index("chrom")
                        .loc[chroms])
    elif gs:
        genome_sizes = (pd.read_csv(gs, sep="\t", usecols=[0,1], names=["chrom", "len"])
                        .set_index("chrom")
                        .loc[chroms])
    genome_chops = pd.concat([intervals_loop(i.Index, 0, stride, l, i.len) 
                                for i in genome_sizes.itertuples()])
    genome_chops_bdt = BedTool.from_dataframe(genome_chops)

    return (genome_chops_bdt.intersect(excl, v=True)
                            .to_dataframe()[["chrom", "start", "end"]])

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

def extract_info(chrom, start, end, label, genome_pyfasta, bigwigs, target_bam, strand="+", transforms:dict=None):
    seq = genome_pyfasta[chrom][int(start):int(end)]
    if strand=="-":
        seq = rev_comp(seq)
    seq_array = dna2OneHot(seq)

    #chromatin track
    chroms_array = []
    if bigwigs is not None and len(bigwigs)>0:
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
    else:
        chroms_array = None
    # label
    label_array = np.array(label, dtype=np.int32)[np.newaxis]
    # counts
    target_array = target_bam.count(chrom, start, end) if target_bam is not None else np.nan
    target_array = np.array(target_array, dtype=np.float32)[np.newaxis]

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