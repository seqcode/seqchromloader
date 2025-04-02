
description = """
    Lightning Data Module for model training
    Given bed file, return sequence and chromatin info
"""

import math
import logging
import torch
import random
import pysam
import pyfaidx
import pyBigWig
import numpy as np
import pandas as pd
import webdataset as wds
from math import sqrt, ceil
from itertools import islice
from torch.utils.data import Dataset, IterableDataset, DataLoader
from pybedtools import BedTool
from pytorch_lightning import LightningDataModule

from seqchromloader import utils

class SeqChromLoader():
    """
    :param dataloader_kws: keyword arguments passed to ``torch.utils.data.DataLoader``
    :type dataloader_kws: dict of kwargs
    """
    def __init__(self, SeqChromDataset):
        self.SeqChromDataset = SeqChromDataset
        self.__doc__ = self.__doc__ + self.SeqChromDataset.__doc__

    def __call__(self, *args, dataloader_kws:dict={}, **kwargs):
        # default dataloader kws
        if dataloader_kws is not None:
            num_workers = dataloader_kws.pop("num_workers", 1) 
        else:
            num_workers = 1

        return DataLoader(self.SeqChromDataset(*args, **kwargs),
                          num_workers=num_workers, **dataloader_kws)

def seqChromLoaderCurry(SeqChromDataset):

    return SeqChromLoader(SeqChromDataset)

class _SeqChromDatasetByWds(IterableDataset):
    """
    :param wds: list of webdataset files to get samples from
    :type wds: list of str
    :param transforms: A dictionary of functions to transform the output data, accepted keys are **["seq", "chrom", "target", "label"]**
    :type transforms: dict of functions
    """
    def __init__(self, wds, transforms:dict=None, rank=0, world_size=1, keep_key=False):
        self.wds = wds
        self.transforms = transforms

        self.rank = rank
        self.world_size = world_size
        self.keep_key = keep_key

    def initialize(self):
        # this function will be called by worker_init_function in DataLoader
        pass

    def __iter__(self):
        pipeline = [
            wds.SimpleShardList(self.wds),
            split_by_node(self.rank, self.world_size),
            wds.split_by_worker,
            wds.tarfile_to_samples(),
            wds.decode(),
            wds.rename(seq="seq.npy",
                       chrom="chrom.npy",
                       target="target.npy",
                       label="label.npy")
        ]
        
        # transform
        if self.transforms is not None: 
            pipeline.append(wds.map_dict(**self.transforms))

        if self.keep_key:
            pipeline.append(wds.to_tuple("__key__", "seq", "chrom", "target", "label"))
        else:
            pipeline.append(wds.to_tuple("seq", "chrom", "target", "label"))
            
        ds = wds.DataPipeline(*pipeline)

        return iter(ds)

SeqChromDatasetByWds = seqChromLoaderCurry(_SeqChromDatasetByWds)

class _SeqChromDatasetByDataFrame(IterableDataset):
    """
    :param dataframe: pandas dataframe describing genomics regions to extract info from, every region has to be of the same length.
    :type dataframe: pd.DataFrame
    :param genome_fasta: Genome fasta file.
    :type genome_fasta: str
    :param bigwig_filelist: A list of bigwig files containing track information (e.g., histone modifications)
    :type bigwig_filelist: list of str or None
    :param target_bam: bam file to get # reads in each region
    :type target_bam: str or None
    :param transforms: A dictionary of functions to transform the output data, accepted keys are *["seq", "chrom", "target", "label"]*
    :type transforms: dict of functions
    """
    def __init__(self, 
                 dataframe: pd.DataFrame, 
                 genome_fasta: str, 
                 bigwig_filelist:list=None, 
                 target_bam=None, 
                 transforms:dict=None, 
                 return_region=False,
                 patch_left=0, patch_right=0,
                 shuffle=False):
        
        self.dataframe = dataframe        
        self.genome_fasta = genome_fasta
        self.genome_pyfaidx = None
        self.bigwig_filelist = bigwig_filelist
        self.bigwigs = None
        self.target_bam = target_bam
        self.target_pysam = None

        self.transforms = transforms
        self.return_region = return_region
        self.patch_left = patch_left
        self.patch_right = patch_right
        self.shuffle = shuffle
        
        self.start = 0; self.end = len(self.dataframe)

    def __len__(self):
        return len(self.dataframe)
    
    def initialize(self):
        # create the stream handler after child processes spawned to enable parallel reading
        # this function will be called by worker_init_function in DataLoader
        self.genome_pyfaidx = pyfaidx.Fasta(self.genome_fasta)
        self.bigwigs = [pyBigWig.open(bw) for bw in self.bigwig_filelist] if self.bigwig_filelist is not None else None
        if self.target_bam is not None:
            self.target_pysam = pysam.AlignmentFile(self.target_bam)
        # shuffle the dataframe if need
        if self.shuffle:
            self.dataframe.sample(frac=1., replace=False)
    
    def __iter__(self):
        self.initialize()
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:  # single-process data loading, return the full iterator
            # split workload
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
            # replace start and end
            self.start = iter_start; self.end = iter_end

        for idx in range(self.start, self.end):
            item = self.dataframe.iloc[idx,]
            try:
                feature = utils.extract_info(
                    item.chrom,
                    item.start,
                    item.end,
                    item.label,
                    genome_pyfaidx=self.genome_pyfaidx,
                    bigwigs=self.bigwigs,
                    target=self.target_pysam,
                    strand=item.strand,
                    transforms=self.transforms,
                    patch_left=self.patch_left,
                    patch_right=self.patch_right
                )
            except utils.BigWigInaccessible as e:
                logging.warn(f"Inaccessible bigwig error detected in region {item.chrom}:{item.start}-{item.end}, Skipping...")
                continue
            except AssertionError as e:
                logging.warn(f"AssertionError detected in region {item.chrom}:{item.start}-{item.end}, Skipping")
                continue

            if not self.return_region:
                yield feature['seq'], feature['chrom'], feature['target'], feature['label']
            else:
                yield f'{item.chrom}:{item.start}-{item.end}', feature['seq'], feature['chrom'], feature['target'], feature['label']


SeqChromDatasetByDataFrame = seqChromLoaderCurry(_SeqChromDatasetByDataFrame)

class _SeqChromDatasetByBed(_SeqChromDatasetByDataFrame):
    """
    :param bed: Bed file describing genomics regions to extract info from, every region has to be of the same length.
    :type bed: str
    :param genome_fasta: Genome fasta file.
    :type genome_fasta: str
    :param bigwig_filelist: A list of bigwig files containing track information (e.g., histone modifications)
    :type bigwig_filelist: list of str or None
    :param target_bam: bam file to get # reads in each region
    :type target_bam: str or None
    :param transforms: A dictionary of functions to transform the output data, accepted keys are *["seq", "chrom", "target", "label"]*
    :type transforms: dict of functions
    """
    def __init__(self, bed: str, genome_fasta: str, bigwig_filelist:list=None, target_bam=None, 
                 transforms:dict=None, return_region=False,
                 patch_left=0, patch_right=0, shuffle=False):
        dataframe = BedTool(bed).to_dataframe().iloc[:, :6]
        dataframe = dataframe.rename(columns={'name': 'label'})
        # assign fake labels and strands if missing
        if not 'label' in dataframe.columns:
            dataframe['label'] = -1
        if not 'strand' in dataframe.columns:
            dataframe['strand'] = '+'
        super().__init__(dataframe,
                         genome_fasta,
                         bigwig_filelist,
                         target_bam,
                         transforms,
                         return_region,
                         patch_left, patch_right, shuffle)

SeqChromDatasetByBed = seqChromLoaderCurry(_SeqChromDatasetByBed)

def count_lines(fp):
    with open(fp, 'r') as f:
        for count, line in enumerate(f):
            pass
    return count+1

def _split_by_node(src, global_rank, world_size):
    if world_size > 1:
        for s in islice(src, global_rank, None, world_size):
            yield s
    else:
        for s in src:
            yield s

split_by_node = wds.pipelinefilter(_split_by_node)

def _scale_chrom(sample, scaler_mean, scaler_std):
    # standardize chrom by provided mean and std
    seq, chrom, target, label = sample
    
    chrom = np.divide(chrom - scaler_mean, scaler_std, dtype=np.float32)

    return seq, chrom, target, label

scale_chrom = wds.pipelinefilter(_scale_chrom)

def _target_vlog(sample):
    # take log(n+1) on target
    seq, chrom, target, label = sample

    target = np.log(target + 1, dtype=np.float32)

    return seq, chrom, target, label

target_vlog = wds.pipelinefilter(_target_vlog)

class SeqChromDataModule(LightningDataModule):
    def __init__(self, train_wds, val_wds, test_wds, train_dataset_size:int=None, transforms:dict=None, num_workers=1, batch_size=512, patch_last=True):
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.train_dataset_size = train_dataset_size
        self.patch_last = patch_last

        self.train_wds = train_wds
        self.val_wds = val_wds
        self.test_wds = test_wds

        self.transforms = transforms

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        try:
            device_id = self.trainer.device_ids[self.trainer.local_rank]
        
            global_rank = self.trainer.global_rank
            world_size = self.trainer.world_size
            print(f"device id {device_id}, local rank {self.trainer.local_rank}, global rank {self.trainer.global_rank} in world {world_size}")
        except AttributeError:
            print(f"Error when trying to fetch device and rank info")
            print(f"Assume dataset is being setup without a trainer, set device id as 0, global rank as 0, world size as 1")
            device_id = 0
            global_rank = 0
            world_size = 1

        self.batch_size_per_rank = int(self.batch_size/world_size)

        if stage in ["fit", "validate", "test"] or stage is None:
            train_pipeline = [
                wds.SimpleShardList(self.train_wds),
                wds.shuffle(100, rng=random.Random(1)),
                split_by_node(global_rank, world_size),
                wds.split_by_worker,
                wds.tarfile_to_samples(),
                wds.shuffle(1000, rng=random.Random(1)),
                wds.decode(),
                wds.rename(seq="seq.npy",
                       chrom="chrom.npy",
                       target="target.npy",
                       label="label.npy")
            ]

            val_pipeline = [
                wds.SimpleShardList(self.val_wds),
                split_by_node(global_rank, world_size),
                wds.split_by_worker,
                wds.tarfile_to_samples(),
                wds.decode(),
                wds.rename(seq="seq.npy",
                       chrom="chrom.npy",
                       target="target.npy",
                       label="label.npy")
            ]

            test_pipeline = [
                wds.SimpleShardList(self.test_wds),
                split_by_node(global_rank, world_size),
                wds.split_by_worker,
                wds.tarfile_to_samples(),
                wds.decode(),
                wds.rename(seq="seq.npy",
                       chrom="chrom.npy",
                       target="target.npy",
                       label="label.npy")
            ]

            if self.transforms is not None:
                train_pipeline.append(wds.map_dict(**self.transforms))
                val_pipeline.append(wds.map_dict(**self.transforms))
                test_pipeline.append(wds.map_dict(**self.transforms))

            self.train_loader = wds.DataPipeline([
                *train_pipeline,
                wds.to_tuple("seq", "chrom", "target", "label")
            ]) 

            self.val_loader = wds.DataPipeline([
                *val_pipeline,
                wds.to_tuple("seq", "chrom", "target", "label"),
            ])

            self.test_loader = wds.DataPipeline([
                *test_pipeline,
                wds.to_tuple("seq", "chrom", "target", "label"),
            ])

    def train_dataloader(self):
        num_workers = self.num_workers if self.num_workers < len(self.train_wds) else len(self.train_wds)
        if self.patch_last:
            return wds.WebLoader(self.train_loader.repeat(2), num_workers=num_workers, batch_size=self.batch_size_per_rank).with_epoch(ceil(self.train_dataset_size/self.batch_size)) # pad the last batch if there is remainder
        else:
            return wds.WebLoader(self.train_loader, num_workers=num_workers, batch_size=self.batch_size_per_rank)

    def val_dataloader(self):
        num_workers = self.num_workers if self.num_workers < len(self.val_wds) else len(self.val_wds)
        return wds.WebLoader(self.val_loader, num_workers=num_workers, batch_size=self.batch_size_per_rank)

    def test_dataloader(self):
        num_workers = self.num_workers if self.num_workers < len(self.test_wds) else len(self.test_wds)
        return wds.WebLoader(self.test_loader, num_workers=num_workers, batch_size=self.batch_size_per_rank)
