
description = """
    Lightning Data Module for model training
    Given bed file, return sequence and chromatin info
"""

import logging
import torch
import random
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

    def __call__(self, *args, worker_init_fn=worker_init_fn, dataloader_kws:dict=None, **kwargs):
        # default dataloader kws
        wif = dataloader_kws.pop("worker_init_fn", worker_init_fn) if dataloader_kws is not None else worker_init_fn

        return DataLoader(self.SeqChromDataset(*args, **kwargs),
                            worker_init_fn=wif, **dataloader_kws)

def seqChromLoaderCurry(SeqChromDataset):

    return SeqChromLoader(SeqChromDataset)

class _SeqChromDatasetByWds(IterableDataset):
    def __init__(self, wds, seq_transform:list=None, chrom_transform:list=None, target_transform:list=None):
        self.seq_transform = seq_transform
        self.chrom_transform = chrom_transform
        self.target_transform = target_transform

        self.wds = wds

    def initialize(self):
        # this function will be called by worker_init_function in DataLoader
        pass

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        pipeline = [
            wds.SimpleShardList(self.wds),
            wds.tarfile_to_samples(),
            wds.split_by_worker,
            wds.decode(),
            wds.to_tuple("seq.npy", "chrom.npy", "target.npy", "label.npy"),
        ]
        if worker_info is None:
            logging.info("Worker info not found, won't split dataset across subprocesses, are you using custom dataloader?")
            logging.info("Ignore the message if you are not using multiprocessing on data loading")
            del pipeline[2]
        
        # transform
        if self.seq_transform is not None: pipeline.extend(self.seq_transform)
        if self.chrom_transform is not None: pipeline.extend(self.chrom_transform)
        if self.target_transform is not None: pipeline.extend(self.target_transform)

        ds = wds.DataPipeline(*pipeline)

        return iter(ds)

SeqChromDatasetByWds = seqChromLoaderCurry(_SeqChromDatasetByWds)

class _SeqChromDatasetByBed(Dataset):
    def __init__(self, bed, fasta, bigwig_files, seq_transform:list=None, chrom_transform:list=None, target_transform:list=None):
        self.bed = pd.read_table(bed, header=None, names=['chrom', 'start', 'end', 'name', 'score', 'strand' ])

        self.fasta = fasta
        self.bigwig_files = bigwig_files
        self.seq_transform = [utils.DNA2OneHot()] + seq_transform # prepend default DNA one hot coding transform
        self.chrom_transfrom = chrom_transform
        self.target_transform = target_transform
    
    def initialize(self):
        # this function will be called by worker_init_function in DataLoader
        self.genome_pyfasta = pyfasta.Fasta(self.config["train_bichrom"]["fasta"])
        #self.tfbam = pysam.AlignmentFile(self.config["train_bichrom"]["tf_bam"])
        self.bigwigs = [pyBigWig.open(bw) for bw in self.bigwig_files]
    
    def __len__(self):
        return len(self.bed)

    def __getitem__(self, idx):
        entry = self.bed.iloc[idx,]
        # get info in the each entry region
        ## sequence
        sequence = self.genome_pyfasta[entry.chrom][int(entry.start):int(entry.end)]
        sequence = self.rev_comp(sequence) if entry.strand=="-" else sequence
        ## chromatin
        ms = []
        try:
            for idx, bigwig in enumerate(self.bigwigs):
                m = (np.nan_to_num(bigwig.values(entry.chrom, entry.start, entry.end))).astype(np.float32)
                if entry.strand == "-": m = m[::-1] # reverse if needed
                ms.append(m)
        except RuntimeError as e:
            print(e)
            raise Exception(f"Failed to extract chromatin {self.bigwig_files[idx]} information in region {entry}")
        ms = np.vstack(ms)
        ## target: read count in region
        #target = self.tfbam.count(entry.chrom, entry.start, entry.end)
        
        # transform
        if self.seq_transform:
            seq = [t(sequence) for t in self.seq_transform]
        if self.chrom_transfrom:
            ms = [t(ms) for t in self.chrom_transfrom]

        return seq, ms

    def rev_comp(self, inp_str):
        rc_dict = {'A': 'T', 'G': 'C', 'T': 'A', 'C': 'G', 'c': 'g',
                   'g': 'c', 't': 'a', 'a': 't', 'n': 'n', 'N': 'N'}
        outp_str = list()
        for nucl in inp_str:
            outp_str.append(rc_dict[nucl])
        return ''.join(outp_str)[::-1] 

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
    def __init__(self, train_wds, val_wds, test_wds, train_dataset_size:int=None, transform:list=None, num_workers=8, batch_size=512):
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.train_dataset_size = train_dataset_size

        self.train_wds = train_wds
        self.val_wds = val_wds
        self.test_wds = test_wds

        self.transform = transform

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
                wds.to_tuple("seq.npy", "chrom.npy", "target.npy", "label.npy"),
            ]

            val_pipeline = [
                wds.SimpleShardList(self.val_wds),
                split_by_node(global_rank, world_size),
                wds.split_by_worker,
                wds.tarfile_to_samples(),
                wds.decode(),
                wds.to_tuple("seq.npy", "chrom.npy", "target.npy", "label.npy"),
            ]

            test_pipeline = [
                wds.SimpleShardList(self.test_wds),
                split_by_node(global_rank, world_size),
                wds.split_by_worker,
                wds.tarfile_to_samples(),
                wds.decode(),
                wds.to_tuple("seq.npy", "chrom.npy", "target.npy", "label.npy"),
            ]

            if self.transform is not None:
                train_pipeline.extend(self.transform)
                val_pipeline.extend(self.transform)
                test_pipeline.extend(self.transform)

            self.train_loader = wds.DataPipeline(
                *train_pipeline
            ) 

            self.val_loader = wds.DataPipeline(
                *val_pipeline
            )

            self.test_loader = wds.DataPipeline(
                *test_pipeline
            )

    def train_dataloader(self):
        return wds.WebLoader(self.train_loader.repeat(2), num_workers=self.num_workers, batch_size=self.batch_size_per_rank).with_epoch(ceil(self.train_dataset_size/self.batch_size)) # pad the last batch if there is remainder

    def val_dataloader(self):
        return wds.WebLoader(self.val_loader, num_workers=self.num_workers, batch_size=self.batch_size_per_rank)

    def test_dataloader(self):
        return wds.WebLoader(self.test_loader, num_workers=self.num_workers, batch_size=self.batch_size_per_rank)