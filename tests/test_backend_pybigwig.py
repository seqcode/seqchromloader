from test_writer_loader import Test
import unittest, seqchromloader

class TestBackendPyBigWig(Test, unittest.TestCase):
    seqchromloader.config.set_bigwig_backend("pyBigWig")

if __name__ == "__main__":
    unittest.main(verbosity=2)

