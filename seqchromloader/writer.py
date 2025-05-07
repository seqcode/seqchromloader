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

logger = logging.getLogger(__name__)

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
                        target_bw=None, 
                        patch_left=0, patch_right=0,
                        outdir="dataset/", outprefix="seqchrom", 
                        compress=True, 
                        numProcessors=1,
                        transforms=None,
                        braceexpand=False,
                        samples_per_tar=10000,
                        batch_size=None):
    """
    Given coordinates dataframe, extract the sequence and chromatin signal, save in webdataset format

    :param coords: pandas DataFrame containing genomic coordinates with columns **[chrom, start, end, label]**
    :type coords: pandas DataFrame
    :param genome_fasta: Genome fasta file.
    :type genome_fasta: str
    :param bigwig_filelist: A list of bigwig files containing track information (e.g., histone modifications)
    :type bigwig_filelist: list of str or None
    :param target_bam: bam file to get # reads in each region, mutually exclusive with `target_bw`
    :type target_bam: str or None
    :param target_bw: bigwig file to get # reads in each region, mutually exclusive with `target_bam`
    :type target_bw: str or None
    :param patch_left: extend the seq and chrom inputs on the left by `patch_left`bp
    :type patch_left: int
    :param patch_right: extend the seq and chrom inputs on the right by `patch_right`bp
    :type patch_right: int
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
    :param samples_per_tar: Number of samples included per tar file
    :param samples_per_tar: int
    """
    # check parameters
    if (target_bam is not None and target_bw is not None):
        raise Exception("Only one of target_bam and target_bw should be provided!")

    # split coordinates and assign chunks to workers
    num_chunks = math.ceil(len(coords) / samples_per_tar)
    chunks = np.array_split(coords, num_chunks)
    
    # freeze the common parameters
    ## create a scaler to get statistics for normalizing chromatin marks input
    ## also create a multiprocessing lock
    dump_data_worker_freeze = functools.partial(dump_data_webdataset_worker, 
                                                    fasta=genome_fasta, 
                                                    bigwig_files=bigwig_filelist,
                                                    target_bam=target_bam,
                                                    target_bw=target_bw,
                                                    patch_left=patch_left,
                                                    patch_right=patch_right,
                                                    compress=compress,
                                                    outdir=outdir,
                                                    transforms=transforms,
                                                    batch_size=batch_size)
    
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
                                bigwig_files=None,
                                target_bam=None,
                                target_bw=None,
                                patch_left=0, patch_right=0,
                                outdir="dataset/", 
                                compress=True,
                                transforms=None,
                                batch_size=None,
                                ):
    #get handlers
    genome_pyfaidx = pyfaidx.Fasta(fasta)
    bigwigs = [pyBigWig.open(bw) for bw in bigwig_files] if bigwig_files is not None else None
    if target_bam is not None:
        target = pysam.AlignmentFile(target_bam)
    elif target_bw is not None:
        target = pyBigWig.open(target_bw)
    else:
        target = None

    # iterate all records
    filename = os.path.join(outdir, f"{outprefix}.tar.gz" if compress else f"{outprefix}.tar")
    sink = wds.TarWriter(filename, compress=compress)
    counter = 0; keys = []; seq_arr = []; chrom_arr = []; target_arr = []; label_arr = []
    for rindex, item in enumerate(coords.itertuples()):

        try:
            feature = utils.extract_info(
                item.chrom,
                item.start,
                item.end,
                item.label,
                genome_pyfaidx=genome_pyfaidx,
                bigwigs=bigwigs,
                target=target,
                strand=item.strand,
                transforms=transforms,
                patch_left=patch_left,
                patch_right=patch_right
            )
        except utils.BigWigInaccessible as e:
            logger.warning(f"Skip the region {item.chrom}:{item.start}-{item.end} due to BigWigInaccessible Exception")
            continue
        except pyfaidx.FetchError as e:
            logger.warning(f"Skip the region {item.chrom}:{item.start}-{item.end} due to pyfaidx FetchError")
            continue
        except AssertionError as e:
            logger.warning(f"Skip the region {item.chrom}:{item.start}-{item.end} due to AssertionError")
            continue
        
        feature_dict = defaultdict()
        if batch_size is None:    
            feature_dict["__key__"] = f"{rindex}_{item.chrom}:{item.start-patch_left}-{item.end+patch_right}_{item.strand}" 
            feature_dict["seq.npy"] = feature['seq']
            feature_dict["chrom.npy"] = feature['chrom']
            feature_dict["target.npy"] = feature['target']
            feature_dict["label.npy"] = feature['label']
            sink.write(feature_dict)
            feature_dict = defaultdict()
        else:
            counter += 1
            keys.append(f"{rindex}_{item.chrom}:{item.start}-{item.end}_{item.strand}")
            seq_arr.append(feature['seq']); chrom_arr.append(feature['chrom']); target_arr.append(feature['target']); label_arr.append(feature['label'])
            
            if counter>=batch_size:
                feature_dict["__key__"] = ','.join(keys)
                feature_dict["seq.npy"] = np.array(seq_arr)
                feature_dict["chrom.npy"] = np.array(chrom_arr)
                feature_dict["target.npy"] = np.array(target_arr)
                feature_dict["label.npy"] = np.array(label_arr)
                sink.write(feature_dict)
                keys, seq_arr, chrom_arr, target_arr, label_arr = [], [], [], [] ,[]
                counter = 0 
                feature_dict = defaultdict() 

    if batch_size is not None:
        feature_dict["__key__"] = ','.join(keys)
        feature_dict["seq.npy"] = np.array(seq_arr)
        feature_dict["chrom.npy"] = np.array(chrom_arr)
        feature_dict["target.npy"] = np.array(target_arr)
        feature_dict["label.npy"] = np.array(label_arr)
        sink.write(feature_dict)

    sink.close()
    if bigwigs is not None:
        for bw in bigwigs: bw.close()

    return filename
