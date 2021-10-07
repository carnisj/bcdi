# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#   (c) 06/2021-present : DESY CFEL
#       authors:
#         Steven Leake, leake@esrf.fr

import numpy as np
import unittest
import bcdi.simulation.supportMaker as sM


def run_tests(test_class):
    suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)
    
class Test(unittest.TestCase):
    """Tests."""

    def test_dummy(self):
        self.assertTrue(True)


if __name__ == "__main__":
    run_tests(Test)
