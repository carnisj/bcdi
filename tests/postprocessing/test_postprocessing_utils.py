# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import unittest
import bcdi.postprocessing.postprocessing_utils as pu
from tests.config import run_tests


class Test(unittest.TestCase):
    """Tests."""

    def test_dummy(self):
        self.assertTrue(True)


if __name__ == "__main__":
    run_tests(Test)
