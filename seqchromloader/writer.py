description = """

    Write seqchrom dataset to disk

"""

import os
import numpy as np
import functools
import math
import logging
from collections import defaultdict
from multiprocessing import Pool

import pyfaidx
import pysam
import pyBigWig
import webdataset as wds

from . import utils
from .loader import _SeqChromDatasetByWds

def convert_data_webdataset(wds_in, wds_out, transforms=None, compress=False):
    """
    Transform the provided webdataset
    
    :param wds_in: input webdataset file
    :type wds_in: string
    :param wds_out: output webdataset file
    :type wds_out: string
    :param transforms: A dictionary of functions to transform the output data, accepted keys are *["seq", "chrom", "target", "label"]*
    :type transforms: dict of functions
    :param compress: whether to compress the output file
    :type compress: boolean
    """
    
    ds = _SeqChromDatasetByWds(wds_in, transforms=transforms, keep_key=True)
    sink = wds.TarWriter(wds_out, compress=compress)
    for (key, seq, chrom, target, label) in ds:
        feature_dict = defaultdict()
        feature_dict["__key__"] = key
        
        feature_dict["seq.npy"] = seq
        feature_dict["chrom.npy"] = chrom
        feature_dict["target.npy"] = target
        feature_dict["label.npy"] = label
        sink.write(feature_dict)
    sink.close()
    
def dump_data_webdataset(coords, genome_fasta, bigwig_filelist,
                        target_bam=None, 
                        outdir="dataset/", outprefix="seqchrom", 
                        compress=True, 
                        numProcessors=1,
                        transforms=None,
                        braceexpand=False,
                        DALI=False):
    """
    Given coordinates dataframe, extract the sequence and chromatin signal, save in webdataset format

    :param coords: pandas DataFrame containing genomic coordinates with columns **[chrom, start, end, label]**
    :type coords: pandas DataFrame
    :param genome_fasta: Genome fasta file.
    :type genome_fasta: str
    :param bigwig_filelist: A list of bigwig files containing track information (e.g., histone modifications)
    :type bigwig_filelist: list of str or None
    :param target_bam: bam file to get # reads in each region
    :type target_bam: str or None
    :param transforms: A dictionary of functions to transform the output data, accepted keys are *["seq", "chrom", "target", "label"]*
    :type transforms: dict of functions
    :param outdir: output directory to save files in
    :type outdir: str
    :param outprefix: prefix of output files
    :type outprefix: str
    :param compress: whether to compress the output files
    :type compress: boolean
    :param numProcessors: number of processors
    :type numProcessors: int
    :param braceexpand: if use brace to simplify the wds file list into a string
    :param braceexpand: boolean
    :param DALI: Set to True if you want to use the dataset for NVIDIA DALI, it would save all arrays in bytes, which results in losing the array shape info
    :param DALI: boolean
    """

    # split coordinates and assign chunks to workers
    num_chunks = math.ceil(len(coords) / 7000)
    chunks = np.array_split(coords, num_chunks)
    
    # freeze the common parameters
    ## create a scaler to get statistics for normalizing chromatin marks input
    ## also create a multiprocessing lock
    dump_data_worker_freeze = functools.partial(dump_data_webdataset_worker, 
                                                    fasta=genome_fasta, 
                                                    bigwig_files=bigwig_filelist,
                                                    target_bam=target_bam,
                                                    compress=compress,
                                                    outdir=outdir,
                                                    transforms=transforms,
                                                    DALI=DALI)
    
    count_of_digits = 0
    nc = num_chunks
    while nc > 0:
       nc = int(nc/10)
       count_of_digits += 1

    pool = Pool(numProcessors)
    res = pool.starmap_async(dump_data_worker_freeze, zip(chunks, [outprefix + "_" + format(i, f'0{count_of_digits}d') for i in range(num_chunks)]))
    files = res.get()
    
    if braceexpand:
        begin = format(0, f'0{count_of_digits}d')
        end = format(range(num_chunks)[-1], f'0{count_of_digits}d')
        return os.path.join(outdir, f"{outprefix}_{{{begin}..{end}}}.tar.gz" if compress else f"{outprefix}_{{{begin}..{end}}}.tar")
    else:
        return files

def dump_data_webdataset_worker(coords, 
                                outprefix, 
                                fasta, 
                                bigwig_files,
                                target_bam=None, 
                                outdir="dataset/", 
                                compress=True,
                                transforms=None,
                                DALI=False):
    # get handlers
    genome_pyfaidx = pyfaidx.Fasta(fasta)
    bigwigs = [pyBigWig.open(bw) for bw in bigwig_files]
    target_pysam = pysam.AlignmentFile(target_bam) if target_bam is not None else None

    # iterate all records
    filename = os.path.join(outdir, f"{outprefix}.tar.gz" if compress else f"{outprefix}.tar")
    sink = wds.TarWriter(filename, compress=compress)
    for rindex, item in enumerate(coords.itertuples()):
        feature_dict = defaultdict()
        feature_dict["__key__"] = f"{rindex}_{item.chrom}:{item.start}-{item.end}_{item.strand}" 

        try:
            feature = utils.extract_info(
                item.chrom,
                item.start,
                item.end,
                item.label,
                genome_pyfaidx=genome_pyfaidx,
                bigwigs=bigwigs,
                target_bam=target_pysam,
                strand=item.strand,
                transforms=transforms,
            )
        except utils.BigWigInaccessible as e:
            continue
        
        if not DALI:
            feature_dict["seq.npy"] = feature['seq']
            feature_dict["chrom.npy"] = feature['chrom']
            feature_dict["target.npy"] = feature['target']
            feature_dict["label.npy"] = feature['label']
        else:
            feature_dict["seq.npy"] = feature['seq'].tobytes()
            feature_dict["chrom.npy"] = feature['chrom'].tobytes()
            feature_dict["target.npy"] = feature['target'].tobytes()
            feature_dict["label.npy"] = feature['label'].tobytes()

        sink.write(feature_dict)

    sink.close()
    for bw in bigwigs: bw.close()

    return filename
