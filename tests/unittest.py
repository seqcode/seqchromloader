import os
import pandas as pd
from seqchromloader import SeqChromDatasetByBed, SeqChromDatasetByWds, SeqChromDataModule
from seqchromloader import get_data_webdataset

import unittest
import tempfile
import shutil
import pathlib as pl
import numpy as np

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
            'start': [60002000, 60005000],
            'end': [60002200, 60005200],
            'label': [0, 1]
        })
        huge_coords = pd.concat([coords] * 5000, axis=0).reset_index()
        get_data_webdataset(huge_coords, 
                    genome_fasta='data/chr19.fa', 
                    chromatin_tracks=['data/sample.bw'],
                    tf_bam='data/sample.bam',
                    nbins=None,
                    outdir=self.tempdir,
                    outprefix='test',
                    reverse=False,
                    compress=True,
                    numProcessors=5,
                    chroms_scaler=None)
        self.assertIsFile(os.path.join(self.tempdir, "test_0.tar.gz"))
    
    def test_wds_loader(self):
        it = iter(SeqChromDatasetByWds(["data/test_0.tar.gz"], batch_size=3))
        seq, chrom, target, label = next(it)

        self.assertEqual(seq[0,0,3].item(), 0.0)
        self.assertEqual(chrom[0,0,3].item(), 2.0)
        self.assertEqual(target[0].item(), 0.0)
        self.assertEqual(label[0].item(), 1)

if __name__ == "__main__":
    unittest.main()