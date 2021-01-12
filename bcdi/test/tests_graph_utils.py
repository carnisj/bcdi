# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import unittest
from numbers import Real
import numpy as np
import bcdi.utils.validation as valid


def run_tests(test_class):
    suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


class TestGraphUtils(unittest.TestCase):
    # def setUp(self):
    #     executed before each test
    #
    # def tearDown(self):
    #     executed after each test

    ########################
    # tests on save_to_vti #
    ########################
    pass


if __name__ == 'main':
    result = run_tests(TestGraphUtils)
    print(result)
