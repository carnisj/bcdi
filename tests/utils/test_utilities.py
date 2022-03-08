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


class TestCast(unittest.TestCase):
    """
    Tests on the function utilities.cast

    """

    def test_not_type_None(self):
        with self.assertRaises(TypeError):
            util.cast(2, target_type=None)

    def test_not_type_str(self):
        with self.assertRaises(TypeError):
            util.cast(2, target_type="float")

    def test_wrong_type_str(self):
        with self.assertRaises(ValueError):
            util.cast(2, target_type=str)

    def test_wrong_type_list(self):
        with self.assertRaises(ValueError):
            util.cast(2, target_type=list)

    def test_list(self):
        out = util.cast([1, 2, 3], target_type=float)
        self.assertTrue(np.allclose(out, [1.0, 2.0, 3.0]))

    def test_list_of_list(self):
        out = util.cast([[1.2, 2.6, -3.1], [8.2, 0, 4.9]], target_type=int)
        self.assertTrue(np.allclose(out, [[1, 2, -3], [8, 0, 4]]))

    def test_array_int(self):
        out = util.cast(np.ones((3, 3), dtype=int), target_type=float)
        self.assertTrue(np.allclose(out, 1.0))

    def test_array_float(self):
        out = util.cast(np.ones((3, 3)) * 1.6, target_type=int)
        self.assertTrue(np.allclose(out, 1))

    def test_number_float(self):
        out = util.cast(-1.6, target_type=int)
        self.assertEqual(out, -1)

    def test_number_int(self):
        out = util.cast(-2, target_type=float)
        self.assertTrue(np.isclose(out, -2.0))

    def test_number_complex(self):
        with self.assertRaises(TypeError):
            util.cast(1-2*1j, target_type=float)

    def test_number_str(self):
        out = util.cast("2.0", target_type=float)
        self.assertTrue(np.isclose(out, 2.0))


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


class TestGaussianWindow(unittest.TestCase):
    """
    Tests on the function utilities.gaussian_window.

    def gaussian_window(
        window_shape,
        sigma=0.3,
        mu=0.0,
        voxel_size=None,
        debugging=False
    )
    """

    def test_2d(self):
        window = util.gaussian_window(window_shape=(13, 13))
        self.assertTrue(np.unravel_index(abs(window).argmax(), window.shape) == (6, 6))

    def test_2d_pad(self):
        data = np.zeros((32, 32))
        data[-13:, 17:30] = util.gaussian_window(window_shape=(13, 13))
        self.assertTrue(np.unravel_index(abs(data).argmax(), data.shape) == (25, 23))

    def test_3d(self):
        window = util.gaussian_window(window_shape=(3, 13, 13))
        self.assertTrue(
            np.unravel_index(abs(window).argmax(), window.shape) == (1, 6, 6)
        )

    def test_3d_pad(self):
        data = np.zeros((4, 32, 32))
        data[:-1, -13:, 17:30] = util.gaussian_window(window_shape=(3, 13, 13))
        self.assertTrue(np.unravel_index(abs(data).argmax(), data.shape) == (1, 25, 23))


if __name__ == "__main__":
    run_tests(TestInRange)
    run_tests(TestIsFloat)
    run_tests(TestFindFile)
    run_tests(TestGaussianWindow)
