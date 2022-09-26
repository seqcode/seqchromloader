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

def get_data_webdataset(coords, genome_fasta, chromatin_tracks,
                        tf_bam=None, 
                        nbins=None, 
                        outdir="dataset/", outprefix="seqchrom", 
                        reverse=False, compress=False, 
                        numProcessors=1, chroms_scaler=None):
    """
    Given coordinates dataframe, extract the sequence and chromatin signal,
    Then save in **TFReocrd** format
    """

    # split coordinates and assign chunks to workers
    num_chunks = math.ceil(len(coords) / 7000)
    chunks = np.array_split(coords, num_chunks)

    # freeze the common parameters
    ## create a scaler to get statistics for normalizing chromatin marks input
    ## also create a multiprocessing lock
    get_data_worker_freeze = functools.partial(get_data_webdataset_worker, 
                                                    fasta=genome_fasta, nbins=nbins, 
                                                    bigwig_files=chromatin_tracks,
                                                    tf_bam=tf_bam,
                                                    reverse=reverse, 
                                                    compress=compress,
                                                    outdir=outdir)

    pool = Pool(numProcessors)
    res = pool.starmap_async(get_data_worker_freeze, zip(chunks, [outprefix + "_" + str(i) for i in range(num_chunks)]))
    res = res.get()

    # fit the scaler if provided
    files = []
    for file, mss in res:
        if chroms_scaler: 
            chroms_scaler.partial_fit(mss)
        files.append(file)

    return files

def get_data_webdataset_worker(coords, outprefix, fasta, bigwig_files,
                                tf_bam=None, 
                                outdir="dataset/", 
                                nbins=None, reverse=False, compress=True):
    # get handlers
    genome_pyfasta = pyfasta.Fasta(fasta)
    bigwigs = [pyBigWig.open(bw) for bw in bigwig_files]
    tfbam = pysam.AlignmentFile(tf_bam) if tf_bam else None

    # iterate all records
    filename = os.path.join(outdir, f"{outprefix}.tar.gz" if compress else f"{outprefix}.tar")
    sink = wds.TarWriter(filename, compress=compress)
    mss = []
    dna2onehot = utils.DNA2OneHot()
    for rindex, item in enumerate(coords.itertuples()):
        feature_dict = defaultdict()
        feature_dict["__key__"] = f"{item.chrom}:{item.start}-{item.end}" 

        # seq
        seq = genome_pyfasta[item.chrom][int(item.start):int(item.end)]
        if reverse:
            seq = utils.rev_comp(seq)
        seq_array = dna2onehot(seq)
        feature_dict["seq.npy"] = seq_array

        #chromatin track
        ms = []
        try:
            for idx, bigwig in enumerate(bigwigs):
                m = (np.nan_to_num(bigwig.values(item.chrom, item.start, item.end)))
                if nbins:
                    m = (m.reshape((nbins, -1))
                          .mean(axis=1, dtype=np.float32))
                if reverse:
                    m = m[::-1] 
                ms.append(m)
        except RuntimeError as e:
            logging.warning(e)
            logging.warning(f"Chromatin track {bigwig_files[idx]} doesn't have information in {item} Skip this region...")
            continue
        ms = np.vstack(ms)  # create the chromatin track array, shape (num_tracks, length)
        feature_dict["chrom.npy"] = ms
        mss.append(ms)
        # label
        feature_dict["label.npy"] = np.array(item.label, dtype=np.int32)[np.newaxis]
        # counts
        target = tfbam.count(item.chrom, item.start, item.end) if tfbam else np.nan
        feature_dict["target.npy"] = np.array(target, dtype=np.float32)[np.newaxis]

        sink.write(feature_dict)

    sink.close()
    for bw in bigwigs: bw.close()

    mss = np.hstack(mss).T

    return filename, mss

def get_data_TFRecord_worker(coords, outprefix, fasta, bigwig_files, tf_bam, nbins, reverse=False):

    genome_pyfasta = pyfasta.Fasta(fasta)
    bigwigs = [pyBigWig.open(bw) for bw in bigwig_files]
    tfbam = pysam.AlignmentFile(tf_bam)

    TFRecord_file = outprefix + ".TFRecord"
    mss = []
    with tf.io.TFRecordWriter(TFRecord_file) as writer:
        for item in coords.itertuples():
            feature_dict = defaultdict()

            # seq
            seq = genome_pyfasta[item.chrom][int(item.start):int(item.end)]
            if reverse:
                seq = rev_comp(seq)
            feature_dict["seq"] = tf.train.Feature(float_list=tf.train.FloatList(value=dna2onehot(seq).flatten()))

            # chromatin track
            ms = []
            try:
                for idx, bigwig in enumerate(bigwigs):
                    m = (np.nan_to_num(bigwig.values(item.chrom, item.start, item.end))
                                            .reshape((nbins, -1))
                                            .mean(axis=1, dtype=float))
                    if reverse:
                        m = m[::-1] 
                    ms.append(m)
                    feature_dict[bigwig_files[idx]] = tf.train.Feature(float_list=tf.train.FloatList(value=m))
            except RuntimeError as e:
                logging.warning(e)
                logging.warning(f"Chromatin track {bigwig_files[idx]} doesn't have information in {item} Skip this region...")
                continue
            ms = np.vstack(ms)  # create the chromatin track array, shape (num_tracks, length)
            mss.append(ms)
            # label
            feature_dict["label"] = tf.train.Feature(int64_list=tf.train.Int64List(value=[item.label]))
            # counts
            # Jianyu: Instead of labels, here using counts as prediction target
            target = tfbam.count(item.chrom, item.start, item.end)
            feature_dict["target"] = tf.train.Feature(float_list=tf.train.FloatList(value=[target]))

            example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
            writer.write(example.SerializeToString())

    for bw in bigwigs: bw.close()

    mss = np.hstack(mss).T

    return TFRecord_file, mss

def get_data(coords, genome_fasta, chromatin_tracks, nbins, reverse=False, numProcessors=1):
    """
    Given coordinates dataframe, extract the sequence and chromatin signal
    """
    y = coords["label"]

    # get pointer
    genome_pyfasta = pyfasta.Fasta(genome_fasta)

    # split coordinates and assign chunks to workers
    chunks = np.array_split(coords, numProcessors)
    get_coverage_worker_freeze = functools.partial(get_coverage_worker, nbins=nbins, 
                                                    bigwig_files=chromatin_tracks, reverse=reverse)
    pool = Pool(numProcessors)
    res = pool.map_async(get_coverage_worker_freeze, chunks)

    # let's take care of sequence
    X_seq = get_sequence_worker(coords, genome_pyfasta, reverse=reverse)

    # gather the results
    chromatin_out_lists = res.get()
    chromatin_out_lists = np.concatenate(chromatin_out_lists, axis=1)

    return X_seq, chromatin_out_lists, y

def get_sequence_worker(coords, fasta, reverse=False):
    """
    Get the sequence in provided regions
    """
    seqs = []
    for item in coords.itertuples():
        seq = fasta[item.chrom][int(item.start):int(item.end)]
        if reverse:
            seq = rev_comp(seq)
        seqs.append(seq)
    return seqs

def get_coverage_worker(coords, bigwig_files, nbins, reverse=False):
    """
    Get the signal coverage in provided regions, summarize mean in each bin
    """
    bigwigs = [pyBigWig.open(bw) for bw in bigwig_files]

    ms = [[] for x in bigwigs]
    for idx, bigwig in enumerate(bigwigs):
        for item in coords.itertuples():
            try:
                m = (np.nan_to_num(bigwig.values(item.chrom, item.start, item.end))
                                    .reshape((nbins, -1))
                                    .mean(axis=1))
            except RuntimeError as e:
                logging.warning(e)
                logging.warning(f"Skip region: {item}")
                continue
            if reverse:
                m = m[::-1]
            ms[idx].append(m)
    return np.array(ms)
