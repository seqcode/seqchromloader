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

import pyfasta
import pysam
import pyBigWig
import webdataset as wds

from seqchromloader import utils

def dump_data_webdataset(coords, genome_fasta, bigwig_filelist,
                        target_bam=None, 
                        outdir="dataset/", outprefix="seqchrom", 
                        compress=True, 
                        numProcessors=1,
                        transforms=None):
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
                                                    transforms=transforms)

    pool = Pool(numProcessors)
    res = pool.starmap_async(dump_data_worker_freeze, zip(chunks, [outprefix + "_" + str(i) for i in range(num_chunks)]))
    files = res.get()

    return files

def dump_data_webdataset_worker(coords, 
                                outprefix, 
                                fasta, 
                                bigwig_files,
                                target_bam=None, 
                                outdir="dataset/", 
                                compress=True,
                                transforms=None):
    # get handlers
    genome_pyfasta = pyfasta.Fasta(fasta)
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
                genome_pyfasta=genome_pyfasta,
                bigwigs=bigwigs,
                target_bam=target_pysam,
                strand=item.strand,
                transforms=transforms,
            )
        except utils.BigWigInaccessible as e:
            continue

        feature_dict["seq.npy"] = feature['seq']
        feature_dict["chrom.npy"] = feature['chrom']
        feature_dict["target.npy"] = feature['target']
        feature_dict["label.npy"] = feature['label']

        sink.write(feature_dict)

    sink.close()
    for bw in bigwigs: bw.close()

    return filename
