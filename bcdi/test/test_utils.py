# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import unittest
from numbers import Real
import numpy as np
import bcdi.utils.utilities as util


def run_tests(test_class):
    suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


class TestUtils(unittest.TestCase):

    def setUp(self):
        # executed before each test
        self.extent = (-10, 99, -20, 89, 10, 119)

    #
    # def tearDown(self):
    #     executed after each test

    #####################
    # tests on in_range #
    #####################
    def test_inrange_in_range(self):
        self.assertTrue(util.in_range(point=(0, 0, 20), extent=self.extent))

    def test_inrange_not_in_range_low(self):
        self.assertFalse(util.in_range(point=(-11, 0, 20), extent=self.extent))

    def test_inrange_not_in_range_high(self):
        self.assertFalse(util.in_range(point=(100, 0, 20), extent=self.extent))

    def test_inrange_lower_edge(self):
        self.assertTrue(util.in_range(point=(-10, -20, 10), extent=self.extent))

    def test_inrange_larger_edge(self):
        self.assertTrue(util.in_range(point=(99, 89, 119), extent=self.extent))


if __name__ == 'main':
    result = run_tests(TestUtils)
    print(result)
