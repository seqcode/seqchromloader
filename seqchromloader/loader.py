
description = """
    Lightning Data Module for model training
    Given bed file, return sequence and chromatin info
"""

import logging
import torch
import random
import pysam
import pyfasta
import pyBigWig
import numpy as np
import pandas as pd
import webdataset as wds
from math import sqrt, ceil
from itertools import islice
from torch.utils.data import Dataset, IterableDataset, DataLoader
from pytorch_lightning import LightningDataModule

from seqchromloader import utils

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.initialize()

class SeqChromLoader():
    def __init__(self, SeqChromDataset):
        self.SeqChromDataset = SeqChromDataset

    def __call__(self, *args, dataloader_kws:dict={}, **kwargs):
        # default dataloader kws
        if dataloader_kws is not None:
            wif = dataloader_kws.pop("worker_init_fn", worker_init_fn)
            num_workers = dataloader_kws.pop("num_workers", 1) 
        else:
            wif = worker_init_fn
            num_workers = 1

        return DataLoader(self.SeqChromDataset(*args, **kwargs),
                            worker_init_fn=wif, num_workers=num_workers, **dataloader_kws)

def seqChromLoaderCurry(SeqChromDataset):

    return SeqChromLoader(SeqChromDataset)

class _SeqChromDatasetByWds(IterableDataset):
    def __init__(self, wds, transforms:dict=None):
        self.transforms = transforms

        self.wds = wds

    def initialize(self):
        # this function will be called by worker_init_function in DataLoader
        pass

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        pipeline = [
            wds.SimpleShardList(self.wds),
            wds.split_by_worker,
            wds.tarfile_to_samples(),
            wds.decode(),
            wds.rename(seq="seq.npy",
                       chrom="chrom.npy",
                       target="target.npy",
                       label="label.npy")
        ]
        if worker_info is None:
            logging.info("Worker info not found, won't split dataset across subprocesses, are you using custom dataloader?")
            logging.info("Ignore the message if you are not using multiprocessing on data loading")
            del pipeline[2]
        
        # transform
        if self.transforms is not None: 
            pipeline.append(wds.map_dict(**self.transforms))

        pipeline.append(wds.to_tuple("seq", "chrom", "target", "label"))
            
        ds = wds.DataPipeline(*pipeline)

        return iter(ds)

SeqChromDatasetByWds = seqChromLoaderCurry(_SeqChromDatasetByWds)

class _SeqChromDatasetByBed(Dataset):
    def __init__(self, bed, genome_fasta, bigwig_filelist:list, target_bam=None, transforms:dict=None, initialize_first=False):
        self.bed = pd.read_table(bed, header=None, names=['chrom', 'start', 'end', 'label', 'score', 'strand' ])

        self.genome_fasta = genome_fasta
        self.genome_pyfasta = None
        self.bigwig_filelist = bigwig_filelist
        self.bigwigs = None
        self.target_bam = target_bam
        self.target_pysam = None

        self.transforms = transforms

        if initialize_first: self.initialize()
    
    def initialize(self):
        # this function will be called by worker_init_function in DataLoader
        self.genome_pyfasta = pyfasta.Fasta(self.genome_fasta)
        self.bigwigs = [pyBigWig.open(bw) for bw in self.bigwig_filelist]
        if self.target_bam is not None:
            self.target_pysam = pysam.AlignmentFile(self.target_bam)
    
    def __len__(self):
        return len(self.bed)

    def __getitem__(self, idx):
        item = self.bed.iloc[idx,]
        try:
            feature = utils.extract_info(
                item.chrom,
                item.start,
                item.end,
                item.label,
                genome_pyfasta=self.genome_pyfasta,
                bigwigs=self.bigwigs,
                target_bam=self.target_pysam,
                strand=item.strand,
                transforms=self.transforms
            )
        except utils.BigWigInaccessible as e:
            raise e

        return feature['seq'], feature['chrom'], feature['target'], feature['label']

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
        if self.patch_last:
            return wds.WebLoader(self.train_loader.repeat(2), num_workers=self.num_workers, batch_size=self.batch_size_per_rank).with_epoch(ceil(self.train_dataset_size/self.batch_size)) # pad the last batch if there is remainder
        else:
            return wds.WebLoader(self.train_loader, num_workers=self.num_workers, batch_size=self.batch_size_per_rank)

    def val_dataloader(self):
        return wds.WebLoader(self.val_loader, num_workers=self.num_workers, batch_size=self.batch_size_per_rank)

    def test_dataloader(self):
        return wds.WebLoader(self.test_loader, num_workers=self.num_workers, batch_size=self.batch_size_per_rank)
