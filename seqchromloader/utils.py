"""
Utilities for iterating constructing data sets and iterating over
DNA sequence data.
"""
import pandas as pd
import numpy as np
import logging
from multiprocessing import Pool
from pybedtools import Interval, BedTool


def filter_chromosomes(input_df, to_filter=None, to_keep=None):
    """
    This function takes as input a pandas DataFrame
    Parameters:
        input_df (dataFrame): A pandas dataFrame, the first column is expected to
        be a chromosome. Example: chr1.
        to_filter (list): Default None (bool = False), will iterate over list
        objects and filter the listed chromosomes.
        ( Default: None, i.e. this condition will not be triggered unless a list
        is supplied)
        to_keep (list): Default None, will iterate over list objects and only
        retain the listed chromosomes.
    Returns:
          output_df (dataFrame): The filtered pandas dataFrame
    """
    if to_filter:
        output_df = input_df.copy()
        for chromosome in to_filter:
            # note: using the str.contains method to remove all
            # contigs; for example: chrUn_JH584304
            bool_filter = ~(output_df['chrom'].str.contains(chromosome))
            output_df = output_df[bool_filter]
    elif to_keep:
        # keep only the to_keep chromosomes:
        # note: this is slightly different from to_filter, because
        # at a time, if only one chromosome is retained, it can be used
        # sequentially.
        filtered_chromosomes = []
        for chromosome in to_keep:
            filtered_record = input_df[(input_df['chrom'] == chromosome)]
            filtered_chromosomes.append(filtered_record)
        # merge the retained chromosomes
        output_df = pd.concat(filtered_chromosomes)
    else:
        output_df = input_df
    return output_df

def get_genome_sizes(genome_sizes_file, to_filter=None, to_keep=None):
    """
    Loads the genome sizes file which should look like this:
    chr1    45900011
    chr2    10001401
    ...
    chrX    9981013
    This function parses this file, and saves the resulting intervals file
    as a BedTools object.
    "Random" contigs, chrUns and chrMs are filtered out.
    Parameters:
        genome_sizes_file (str): (Is in an input to the class,
        can be downloaded from UCSC genome browser)
        to_filter (list): Default None (bool = False), will iterate over list
        objects and filter the listed chromosomes.
        ( Default: None, i.e. this condition will not be triggered unless a list
        is supplied)
        to_keep (list): Default None, will iterate over list objects and only
        retain the listed chromosomes.
    Returns:
        A BedTools (from pybedtools) object containing all the chromosomes,
        start (0) and stop (chromosome size) positions
    """
    genome_sizes = pd.read_csv(genome_sizes_file, sep='\t',
                               header=None, names=['chrom', 'length'])

    genome_sizes_filt = filter_chromosomes(genome_sizes, to_filter=to_filter,
                                           to_keep=to_keep)

    genome_bed_data = []
    # Note: Modifying this to deal with unexpected (incorrect) edge case \
    # BedTools shuffle behavior.
    # While shuffling data, BedTools shuffle is placing certain windows at the \
    # edge of a chromosome
    # Why it's doing that is unclear; will open an issue on GitHub.
    # It's probably placing the "start" co-ordinate within limits of the genome,
    # with the end coordinate not fitting.
    # This leads to the fasta file returning an incomplete sequence \
    # (< 500 base pairs)
    # This breaks the generator feeding into Model.fit.
    # Therefore, in the genome sizes file, buffering 550 from the edges
    # to allow for BedTools shuffle to place window without running of the
    # chromosome.
    for chrom, sizes in genome_sizes_filt.values:
        genome_bed_data.append(Interval(chrom, 0 + 550, sizes - 550))
    genome_bed_data = BedTool(genome_bed_data)
    return genome_bed_data

def load_chipseq_data(chip_peaks_file, genome_sizes_file, to_filter=None,
                      to_keep=None):
    """
    Loads the ChIP-seq peaks data.
    The chip peaks file is an events bed file:
    chr1:451350
    chr2:91024
    ...
    chrX:870000
    This file can be constructed using a any peak-caller. We use multiGPS.
    Also constructs a 1 bp long bedfile for each coordinate and a
    BedTools object which can be later used to generate
    negative sets.
    """
    chip_seq_data = pd.read_csv(chip_peaks_file, delimiter=':', header=None,
                             names=['chrom', 'start'])
    chip_seq_data['end'] = chip_seq_data['start'] + 1

    chip_seq_data = filter_chromosomes(chip_seq_data, to_filter=to_filter,
                                       to_keep=to_keep)

    sizes = pd.read_csv(genome_sizes_file, names=['chrom', 'chrsize'],
                        sep='\t')

    # filtering out any regions that are close enough to the edges to
    # result in out-of-range windows when applying data augmentation.
    chrom_sizes_dict = (dict(zip(sizes.chrom, sizes.chrsize)))
    chip_seq_data['window_max'] = chip_seq_data['end'] + 500
    chip_seq_data['window_min'] = chip_seq_data['start'] - 500

    chip_seq_data['chr_limits_upper'] = chip_seq_data['chrom'].map(
        chrom_sizes_dict)
    chip_seq_data = chip_seq_data[chip_seq_data['window_max'] <=
                                  chip_seq_data['chr_limits_upper']]
    chip_seq_data = chip_seq_data[chip_seq_data['window_min'] >= 0]
    chip_seq_data = chip_seq_data[['chrom', 'start', 'end']]

    return chip_seq_data

def exclusion_regions(blacklist_file, chip_seq_data):
    """
    This function takes as input a bound bed file (from multiGPS).
    The assumption is that the bed file reports the peak center
    For example: chr2   45  46
    It converts these peak centers into 501 base pair windows, and adds them to
    the exclusion list which will be used when constructing negative sets.
    It also adds the mm10 blacklisted windows to the exclusion list.
    Parameters:
        blacklist_file (str): Path to the blacklist file.
        chip_seq_data (dataFrame): The pandas chip-seq data loaded by load_chipseq_data
    Returns:
        exclusion_windows (BedTool): A bedtools object containing all exclusion windows.
        bound_exclusion_windows (BedTool): A bedtool object containing only
        those exclusion windows where there exists a binding site.
    """
    temp_chip_file = chip_seq_data.copy()  # Doesn't modify OG array.
    temp_chip_file['start'] = temp_chip_file['start'] - 250
    temp_chip_file['end'] = temp_chip_file['end'] + 250

    if blacklist_file is None:
        print('No blacklist file specified ...')
        exclusion_windows = BedTool.from_dataframe(temp_chip_file[['chrom', 'start','end']])
    else:
        bound_exclusion_windows = BedTool.from_dataframe(temp_chip_file[['chrom', 'start','end']])
        blacklist_exclusion_windows = BedTool(blacklist_file)
        exclusion_windows = BedTool.cat(
            *[blacklist_exclusion_windows, bound_exclusion_windows])
    return exclusion_windows

def make_random_shift(coords, L, buffer=25):
    """
    This function takes as input a set of bed coordinates dataframe 
    It finds the mid-point for each record or Interval in the bed file,
    shifts the mid-point, and generates a windows of length L.

    If training window length is L, then we must ensure that the
    peak center is still within the training window.
    Therefore: -L/2 < shift < L/2
    To add in a buffer: -L/2 + 25 <= shift <= L/2 + 25
    # Note: The 50 here is a tunable hyper-parameter.
    Parameters:
        coords(pandas dataFrame): This is an input bedfile (first 3 column names: "chr", "start", "end")
    Returns:
        shifted_coords(pandas dataFrame): The output bedfile with shifted coords
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
    """
    return (coords.assign(midpoint=lambda x: (x["start"]+x["end"])/2)
                .astype({"midpoint": int})
                .assign(midpoint=lambda x: x["midpoint"] + d)
                .apply(lambda s: pd.Series([s["chrom"], int(s["midpoint"]-L/2), int(s["midpoint"]+L/2)],
                                            index=["chrom", "start", "end"]), axis=1))

def random_coords(gs, incl, excl, l=500, n=1000):
    """
    Randomly sample n intervals of length l from the genome,
    shuffle to make all intervals inside the desired regions 
    and outside exclusion regions
    """
    return (BedTool()
            .random(l=l, n=n, g=gs)
            .shuffle(g=gs, incl=incl.fn, excl=excl.fn)
            .to_dataframe()[["chrom", "start", "end"]])

def chop_genome(gs, chroms, excl, stride=500, l=500):
    """
    Given a genome size file and chromosome list,
    chop these chromosomes into intervals of length l,
    with include/exclude regions specified
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
     
    genome_sizes = (pd.read_csv(gs, sep="\t", names=["chrom", "len"])
                        .set_index("chrom")
                        .loc[chroms])
    genome_chops = pd.concat([intervals_loop(i.Index, 0, stride, l, i.len) 
                                for i in genome_sizes.itertuples()])
    genome_chops_bdt = BedTool.from_dataframe(genome_chops)

    return (genome_chops_bdt.intersect(excl, v=True)
                            .to_dataframe()[["chrom", "start", "end"]])

def clean_bed(coords):
    """
    Clean the bed file:
    1. Remove intervals with start < 0
    """
    return coords.loc[coords["start"]>=0]

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

def dna2OneHot(dnaSeq):
    DNA2Index = {
            "A": 0,
            "C": 1, 
            "G": 2,
            "T": 3,
        }
    
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

if __name__ == "__main__":
    chip_seq_coordinates = load_chipseq_data("Bichrom/sample_data/Ascl1.peaks", genome_sizes_file="Bichrom/sample_data/mm10.info",
                                                    to_keep=None, to_filter=['chr17', 'chr11', 'chrM', 'chrUn'])
