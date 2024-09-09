import os
import sys
import numpy as np
import pandas as pd
from functools import partial

from torch import threshold
from seqchromloader import SeqChromDatasetByDataFrame, SeqChromDatasetByBed, SeqChromDatasetByWds, SeqChromDataModule
from seqchromloader import dump_data_webdataset, convert_data_webdataset
from seqchromloader import get_genome_sizes, make_random_shift, make_flank, random_coords, chop_genome, make_gc_match, make_motif_match

import unittest
import tempfile
import shutil
import pathlib as pl
import webdataset as wds
from pybedtools import BedTool
from pyfaidx import Fasta
from pyjaspar import jaspardb

class Test(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    @classmethod
    def setUpClass(cls) -> None:
        cls.tempdir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tempdir)

    def assertIsFile(self, path):
        if not pl.Path(path).resolve().is_file():
            raise AssertionError("File does not exist: %s" % str(path))
        
    def test_get_genome_sizes(self):
        genome_sizes_nochr10=get_genome_sizes(genome="mm10", to_filter=["chr10"])
        genome_sizes_chr19chr11=get_genome_sizes(genome="mm10", to_keep=["chr19", "chr11"])
        
        self.assertFalse(np.any(genome_sizes_nochr10.chrom.unique()=="chr10"))
        self.assertTrue([i in ["chr11", "chr19"] for i in genome_sizes_chr19chr11.chrom.unique()])
        
    def test_make_random_shift(self):
        coords = pd.DataFrame({
            'chrom': ["chr1", "chr2"],
            'start': [30, 100],
            'end':[50, 150]
        })
        for i in range(1000):
            shifted_window = make_random_shift(coords=coords, L=10)
            self.assertTrue(max(abs((shifted_window.start + shifted_window.end)/2 - (coords.start + coords.end)/2)) <=5)
            self.assertTrue(np.all((shifted_window.end-shifted_window.start) == 10))
        
    def test_make_flank(self):
        coords = pd.DataFrame({
            'chrom': ["chr1", "chr2"],
            'start': [30, 100],
            'end':[50, 150]
        })
        coords_flank = make_flank(coords, L=20, d=30)
        self.assertTrue(np.all(coords_flank.start == [60, 145]))
        self.assertTrue(np.all(coords_flank.end == [80, 165]))
        
    def test_random_coords(self):
        interval = BedTool().from_dataframe(pd.DataFrame({'chrom': ['chr1', 'chr3'],
                                                      'start': [0, 500],
                                                      'end': [50000, 20000]}))
        coords_incl = random_coords(genome="mm10", incl=interval)
        coords_excl = random_coords(genome="mm10", excl=interval)
        
        self.assertTrue(BedTool().from_dataframe(coords_incl).intersect(interval).count()==len(coords_incl))
        self.assertTrue(BedTool().from_dataframe(coords_excl).intersect(interval).count()==0)
        
    def test_chop_genome(self):
        interval = BedTool().from_dataframe(pd.DataFrame({'chrom': ['chr2', 'chr12'],
                                                      'start': [0, 500],
                                                      'end': [50000, 20000]}))
        coords_incl = chop_genome(chroms=["chr2", "chr12"], genome="mm10", stride=1000, l=500, incl=interval)
        coords_excl = chop_genome(chroms=["chr2", "chr12"], genome="mm10", stride=1000, l=500, excl=interval)
        for c in ['chr2', 'chr12']:
            df = coords_incl[coords_incl.chrom==c]
            self.assertTrue(np.all([df.start.iloc[i] - df.start.iloc[i-1] == 1000 for i in range(1, len(df))]))
        self.assertTrue(BedTool().from_dataframe(coords_incl).intersect(interval).count()==len(coords_incl))
        self.assertTrue(BedTool().from_dataframe(coords_excl).intersect(interval).count()==0)

    def test_write_load_batched_target_bam(self):
        coords = pd.DataFrame({
            'chrom': ["chr19", "chr19"],
            'start': [2, 3],
            'end': [7, 8],
            'label': [0, 1],
            'score': [".", "."],
            'strand': ["+", "+"]
        })
        huge_coords = pd.concat([coords] * 6001, axis=0).reset_index()
        dump_data_webdataset(huge_coords, 
                    genome_fasta='data/sample.fa', 
                    bigwig_filelist=['data/sample.bw'],
                    target_bam='data/sample.bam',
                    outdir=self.tempdir,
                    outprefix='test',
                    compress=True,
                    numProcessors=2,
                    batch_size=2,
                    patch_left=2, patch_right=2)
        self.assertIsFile(os.path.join(self.tempdir, "test_0.tar.gz"))
        ds = wds.DataPipeline(
            wds.SimpleShardList([os.path.join(self.tempdir, "test_0.tar.gz")]),
            wds.tarfile_to_samples(),
            wds.decode(),
            wds.to_tuple("seq.npy", "chrom.npy", "target.npy", "label.npy"),
            wds.batched(2)
        )
        seq, chrom, target, label = next(iter(ds))
        self.assertEqual(seq.shape, (2, 2, 4, 9))
        ## iterate through all samples
        it = iter(ds)
        while True:
            try:
                next(it)
            except StopIteration:
                break

    def test_write_load_target_bam(self):
        coords = pd.DataFrame({
            'chrom': ["chr19", "chr19"],
            'start': [0, 3],
            'end': [5, 8],
            'label': [0, 1],
            'score': [".", "."],
            'strand': ["+", "+"]
        })
        huge_coords = pd.concat([coords] * 6000, axis=0).reset_index()
        dump_data_webdataset(huge_coords, 
                    genome_fasta='data/sample.fa', 
                    bigwig_filelist=['data/sample.bw'],
                    target_bam='data/sample.bam',
                    outdir=self.tempdir,
                    outprefix='test',
                    compress=True,
                    numProcessors=2)
        self.assertIsFile(os.path.join(self.tempdir, "test_0.tar.gz"))
        wds_files = dump_data_webdataset(huge_coords, 
                    genome_fasta='data/sample.fa', 
                    bigwig_filelist=['data/sample.bw'],
                    target_bam='data/sample.bam',
                    outdir=self.tempdir,
                    outprefix='test',
                    compress=True,
                    numProcessors=2,
                    braceexpand=True)
        self.assertTrue(wds_files == os.path.join(self.tempdir, "test_{0..1}.tar.gz"))

        ds = wds.DataPipeline(
            wds.SimpleShardList([os.path.join(self.tempdir, "test_0.tar.gz")]),
            wds.tarfile_to_samples(),
            wds.decode(),
            wds.to_tuple("seq.npy", "chrom.npy", "target.npy", "label.npy"),
            wds.batched(2)
        )
        seq, chrom, target, label = next(iter(ds))
        self.assertEqual(seq[1,0,4].item(), 1.0)
        self.assertEqual(chrom[0,0,3].item(), 999.0)
        self.assertEqual(target[0].item(), 2.0)
        self.assertEqual(label[1].item(), 1)
    
    def test_write_load_target_bw(self):
        coords = pd.DataFrame({
            'chrom': ["chr19", "chr19"],
            'start': [0, 3],
            'end': [5, 8],
            'label': [0, 1],
            'score': [".", "."],
            'strand': ["+", "+"]
        })
        huge_coords = pd.concat([coords] * 5000, axis=0).reset_index()
        dump_data_webdataset(huge_coords, 
                    genome_fasta='data/sample.fa', 
                    bigwig_filelist=['data/sample.bw'],
                    target_bw='data/sample.bw',
                    outdir=self.tempdir,
                    outprefix='test',
                    compress=True,
                    transforms={'target': partial(np.sum, keepdims=True)},
                    numProcessors=2)
        self.assertIsFile(os.path.join(self.tempdir, "test_0.tar.gz"))
        ds = wds.DataPipeline(
            wds.SimpleShardList([os.path.join(self.tempdir, "test_0.tar.gz")]),
            wds.tarfile_to_samples(),
            wds.decode(),
            wds.to_tuple("seq.npy", "chrom.npy", "target.npy", "label.npy"),
            wds.batched(2)
        )
        seq, chrom, target, label = next(iter(ds))
        self.assertEqual(seq[1,0,4].item(), 1.0)
        self.assertEqual(chrom[0,0,3].item(), 999.0)
        self.assertEqual(target[0].item(), 999.0)
        self.assertEqual(label[1].item(), 1)

    def test_write_load_chrom_target_none(self):
        coords = pd.DataFrame({
            'chrom': ["chr19", "chr19"],
            'start': [0, 3],
            'end': [5, 8],
            'label': [0, 1],
            'score': [".", "."],
            'strand': ["+", "+"]
        })
        huge_coords = pd.concat([coords] * 5000, axis=0).reset_index()
        dump_data_webdataset(huge_coords, 
                    genome_fasta='data/sample.fa', 
                    bigwig_filelist=None,
                    outdir=self.tempdir,
                    outprefix='test_target_none',
                    compress=True,
                    numProcessors=2)
        self.assertIsFile(os.path.join(self.tempdir, "test_target_none_0.tar.gz"))
        ds = wds.DataPipeline(
            wds.SimpleShardList([os.path.join(self.tempdir, "test_target_none_0.tar.gz")]),
            wds.tarfile_to_samples(),
            wds.decode(), wds.to_tuple("seq.npy", "chrom.npy", "target.npy", "label.npy"),
            wds.batched(2)
        )
        seq, chrom, target, label = next(iter(ds))
        self.assertEqual(seq[1,0,4].item(), 1.0)
        self.assertEqual(label[1].item(), 1)

    def test_wds_loader(self):
        it = iter(SeqChromDatasetByWds(["data/test_0.tar.gz"], dataloader_kws={"batch_size":3}))
        seq, chrom, target, label = next(it)

        self.assertEqual(seq[0,0,3].item(), 1.0)
        self.assertEqual(chrom[0,0,3].item(), 4.0)
        self.assertEqual(target[0].item(), 0.0)
        self.assertEqual(label[1].item(), 1)
    
    def test_wds_loader_transform(self):
        it = iter(SeqChromDatasetByWds(["data/test_0.tar.gz"],
                  transforms={"seq": test_seq_transform,
                              "chrom": test_chrom_transform,
                              "target": test_target_transform},
                  dataloader_kws={"batch_size":3}))
        seq, chrom, target, label = next(it)

        self.assertEqual(seq[0,0,3].item(), 2.0)
        self.assertAlmostEqual(chrom[0,0,3].item(), 4.0/3)
        self.assertEqual(target[0].item(), 0.0)
        self.assertEqual(label[1].item(), 1)
    
    def test_wds_convert_loader(self):
        convert_data_webdataset("data/test_0.tar.gz", "test_0_convert.tar.gz",
                                        transforms={"seq": test_seq_transform,
                                                    "chrom": test_chrom_transform,
                                                    "target": test_target_transform})
        it = iter(SeqChromDatasetByWds(["test_0_convert.tar.gz"],
                  dataloader_kws={"batch_size":3}))
        seq, chrom, target, label = next(it)

        self.assertEqual(seq[0,0,3].item(), 2.0)
        self.assertAlmostEqual(chrom[0,0,3].item(), 4.0/3)
        self.assertEqual(target[0].item(), 0.0)
        self.assertEqual(label[1].item(), 1)
    
    def test_df_loader(self):
        dataframe = pd.read_table("data/sample.bed", header=None, sep="\t", names=['chrom', 'start', 'end', 'label', 'score', 'strand' ])
        it = iter(SeqChromDatasetByDataFrame(
            dataframe=dataframe,
            genome_fasta="data/sample.fa",
            bigwig_filelist=["data/sample.bw"],
            target_bam="data/sample.bam",
            dataloader_kws={"batch_size":2,
                            "shuffle":False}
        ))
        seq, chrom, target, label = next(it)
        self.assertEqual(seq[1,0,4].item(), 1.0)
        self.assertEqual(chrom[0,0,3].item(), 999.0)
        self.assertEqual(target[0].item(), 2.0)
        self.assertEqual(label[1].item(), 1)

    def test_df_loader_no_bigwig(self):
        dataframe = pd.read_table("data/sample.bed", header=None, sep="\t", names=['chrom', 'start', 'end', 'label', 'score', 'strand' ])
        it = iter(SeqChromDatasetByDataFrame(
            dataframe=dataframe,
            genome_fasta="data/sample.fa",
            bigwig_filelist=None,
            target_bam="data/sample.bam",
            dataloader_kws={"batch_size":2,
                            "shuffle":False}
        ))
        seq, chrom, target, label = next(it)
        self.assertEqual(seq[1,0,4].item(), 1.0)
        self.assertEqual(target[0].item(), 2.0)
        self.assertEqual(label[1].item(), 1)

    def test_bed_loader(self):
        it = iter(SeqChromDatasetByBed(
            bed="data/sample.bed",
            genome_fasta="data/sample.fa",
            bigwig_filelist=["data/sample.bw"],
            target_bam="data/sample.bam",
            dataloader_kws={"batch_size":2,
                            "shuffle":False}
        ))
        seq, chrom, target, label = next(it)
        self.assertEqual(seq[1,0,4].item(), 1.0)
        self.assertEqual(chrom[0,0,3].item(), 999.0)
        self.assertEqual(target[0].item(), 2.0)
        self.assertEqual(label[1].item(), 1)

    def test_bed_loader_transform(self):

        it = iter(SeqChromDatasetByBed(
            bed="data/sample.bed",
            genome_fasta="data/sample.fa",
            bigwig_filelist=["data/sample.bw"],
            target_bam="data/sample.bam",
            transforms={"seq": test_seq_transform,
                        "chrom": test_chrom_transform,
                        "target": test_target_transform},
            dataloader_kws={"batch_size":2,
                            "shuffle":False}
        ))
        seq, chrom, target, label = next(it)
        self.assertEqual(seq[1,0,4].item(), 2.0)
        self.assertEqual(chrom[0,0,3].item(), 333.0)
        self.assertEqual(target[0].item(), 6.0)
        self.assertEqual(label[1].item(), 1)

    def test_bed_loader_return_region(self):

        it = iter(SeqChromDatasetByBed(
            bed="data/sample.bed",
            genome_fasta="data/sample.fa",
            bigwig_filelist=["data/sample.bw"],
            target_bam="data/sample.bam",
            transforms={"seq": test_seq_transform,
                        "chrom": test_chrom_transform,
                        "target": test_target_transform},
            dataloader_kws={"batch_size":2,
                            "shuffle":False},
            return_region=True
        ))
        region, seq, chrom, target, label = next(it)
        self.assertEqual(region[0], "chr19:0-5")

    def test_lightning_datamodule(self):
        dm = SeqChromDataModule(
            train_wds="data/test_0.tar.gz",
            val_wds="data/test_0.tar.gz",
            test_wds="data/test_0.tar.gz",
            train_dataset_size=100,
            batch_size=3,
            num_workers=1,
            patch_last=False,
        )
        dm.setup()
        val_dl = iter(dm.val_dataloader())
        seq, chrom, target, label = next(val_dl)
        self.assertEqual(seq[0,0,3].item(), 1.0)
        self.assertEqual(chrom[0,0,3].item(), 4.0)
        self.assertEqual(target[0].item(), 0.0)
        self.assertEqual(label[1].item(), 1)

    def test_lightning_datamodule_transform(self):
        dm = SeqChromDataModule(
            train_wds="data/test_0.tar.gz",
            val_wds="data/test_0.tar.gz",
            test_wds="data/test_0.tar.gz",
            transforms={"seq": test_seq_transform,
                        "chrom": test_chrom_transform,
                        "target": test_target_transform},
            train_dataset_size=100,
            batch_size=3,
            num_workers=1,
            patch_last=False,
        )
        dm.setup()
        val_dl = iter(dm.val_dataloader())
        seq, chrom, target, label = next(val_dl)
        
        self.assertEqual(seq[0,0,3].item(), 2.0)
        self.assertAlmostEqual(chrom[0,0,3].item(), 4.0/3)
        self.assertEqual(target[0].item(), 0.0)
        self.assertEqual(label[1].item(), 1)

    def test_make_gc_match_regions(self):
        # input region gc content is 0.345
        coords = pd.DataFrame({
                'chrom': ['chr1'],
                'start': [6000000],
                'end': [6000200]
            })
        g = Fasta('data/chr1.fa')
        input_gc = g['chr1'][6000000:6000200].gc
        gc_diff_max = 0.05
        output_gc_match = make_gc_match(coords, 'data/chr1.fa', n=100, gc_diff_max=gc_diff_max)
        for item in output_gc_match.itertuples():
            self.assertTrue(abs(input_gc - g[item.chrom][item.start:item.end].gc)<=gc_diff_max)

    def test_make_motif_match_regions(self):
        gc_content = 0.4; threshold=2.0
        jdb_obj = jaspardb()
        jdb_obj.fetch_motif_by_id('MA0095.2')
        motif = jdb_obj.fetch_motif_by_id('MA0095.2')
        pwm = motif.counts.normalize(pseudocounts={'A':1-gc_content, 'C': gc_content, 'G': gc_content, 'T': 1-gc_content})
        pssm = pwm.log_odds({'A':(1-gc_content)/2,'C':gc_content/2,'G':gc_content/2,'T':(1-gc_content)/2})
        rpssm = pssm.reverse_complement()

        g = Fasta('data/chr1.fa')
        output_motif_match = make_motif_match(motif, "data/chr1.fa", threshold=threshold, gc_content=gc_content)
        for item in output_motif_match.itertuples():
            subseq = g[item.chrom][item.start:item.end].seq
            s = max(max(pssm.calculate(subseq)), max(rpssm.calculate(subseq)))
            self.assertTrue(s > threshold)

def test_seq_transform(seq):
    return seq + 1

def test_chrom_transform(chrom):
    return chrom / 3

def test_target_transform(target):
    return target * 3

if __name__ == "__main__":
    unittest.main(verbosity=2)
