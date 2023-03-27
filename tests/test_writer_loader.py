import os
import sys
sys.path.insert(0, "./")
import pandas as pd
from seqchromloader import SeqChromDatasetByDataFrame, SeqChromDatasetByBed, SeqChromDatasetByWds, SeqChromDataModule
from seqchromloader import dump_data_webdataset

import unittest
import tempfile
import shutil
import pathlib as pl
import webdataset as wds

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

    def test_writer(self):
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
                    target_bam='data/sample.bam',
                    outdir=self.tempdir,
                    outprefix='test',
                    compress=True,
                    numProcessors=5)
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
        self.assertEqual(target[0].item(), 2.0)
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

def test_seq_transform(seq):
    return seq + 1

def test_chrom_transform(chrom):
    return chrom / 3

def test_target_transform(target):
    return target * 3

if __name__ == "__main__":
    unittest.main()