# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np
import os
from pyfakefs import fake_filesystem_unittest
import unittest
import bcdi.utils.utilities as util


def run_tests(test_class):
    suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


class TestFindFile(fake_filesystem_unittest.TestCase):
    """
    Tests on the function utilities.find_file.

    def find_file(filename: str, default_folder: str) -> str:
    """

    def setUp(self):
        self.setUpPyfakefs()
        self.valid_path = "/gpfs/bcdi/data/"
        os.makedirs(self.valid_path)
        with open(self.valid_path + "dummy.spec", "w") as f:
            f.write("dummy")

    def test_filename_none(self):
        with self.assertRaises(TypeError):
            util.find_file(filename=None, default_folder=None)

    def test_full_path_to_file(self):
        output = util.find_file(
            filename=self.valid_path + "dummy.spec", default_folder=None
        )
        self.assertTrue(output == self.valid_path + "dummy.spec")

    def test_filename_file_name(self):
        output = util.find_file(filename="dummy.spec", default_folder=self.valid_path)
        self.assertTrue(output == self.valid_path + "dummy.spec")

    def test_filename_file_name_missing_backslash(self):
        output = util.find_file(
            filename="dummy.spec", default_folder=self.valid_path[:-1]
        )
        self.assertTrue(output == self.valid_path + "dummy.spec")

    def test_filename_file_name_default_dir_none(self):
        with self.assertRaises(TypeError):
            util.find_file(filename="dummy.spec", default_folder=None)

    def test_filename_file_name_default_dir_inexisting(self):
        with self.assertRaises(ValueError):
            util.find_file(filename="dummy.spec", default_folder="/wrong/path")

    def test_filename_file_name_inexisting_default_dir_existing(self):
        with self.assertRaises(ValueError):
            util.find_file(filename="dum.spec", default_folder=self.valid_path)


class TestInRange(unittest.TestCase):
    """Tests on the function utilities.in_range."""

    def setUp(self):
        # executed before each test
        self.extent = (-10, 99, -20, 89, 10, 119)

    # def tearDown(self):
    #     executed after each test

    def test_inrange_in_range(self):
        self.assertTrue(util.in_range(point=(0, 0, 20), extent=self.extent))

    def test_inrange_not_in_range_low_z(self):
        self.assertFalse(util.in_range(point=(-11, 0, 20), extent=self.extent))

    def test_inrange_not_in_range_high_z(self):
        self.assertFalse(util.in_range(point=(100, 0, 20), extent=self.extent))

    def test_inrange_not_in_range_low_y(self):
        self.assertFalse(util.in_range(point=(0, -21, 20), extent=self.extent))

    def test_inrange_not_in_range_high_y(self):
        self.assertFalse(util.in_range(point=(0, 90, 20), extent=self.extent))

    def test_inrange_not_in_range_low_x(self):
        self.assertFalse(util.in_range(point=(0, 0, 9), extent=self.extent))

    def test_inrange_not_in_range_high_x(self):
        self.assertFalse(util.in_range(point=(0, 0, 120), extent=self.extent))

    def test_inrange_lower_edge_z(self):
        self.assertTrue(util.in_range(point=(-10, 0, 20), extent=self.extent))

    def test_inrange_larger_edge_z(self):
        self.assertTrue(util.in_range(point=(99, 0, 20), extent=self.extent))

    def test_inrange_lower_edge_y(self):
        self.assertTrue(util.in_range(point=(0, -20, 20), extent=self.extent))

    def test_inrange_larger_edge_y(self):
        self.assertTrue(util.in_range(point=(0, 89, 20), extent=self.extent))

    def test_inrange_lower_edge_x(self):
        self.assertTrue(util.in_range(point=(0, 0, 10), extent=self.extent))

    def test_inrange_larger_edge_x(self):
        self.assertTrue(util.in_range(point=(0, 0, 119), extent=self.extent))


class TestIsFloat(unittest.TestCase):
    """
    Tests on the function utilities.is_float.

    def is_float(string)
    """

    def test_string_float(self):
        self.assertTrue(util.is_float("12.0"))

    def test_string_int(self):
        self.assertTrue(util.is_float("12"))

    def test_string_complex(self):
        self.assertFalse(util.is_float("12 + 1j"))

    def test_string_none(self):
        self.assertFalse(util.is_float("None"))

    def test_string_not_numeric(self):
        self.assertFalse(util.is_float("abc"))

    def test_none(self):
        with self.assertRaises(TypeError):
            util.is_float(None)

    def test_array(self):
        with self.assertRaises(TypeError):
            util.is_float(np.ones(3))


if __name__ == "__main__":
    run_tests(TestInRange)
    run_tests(TestIsFloat)
    run_tests(TestFindFile)
