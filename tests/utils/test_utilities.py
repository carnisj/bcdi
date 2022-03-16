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
from tests.config import run_tests

from bcdi.experiment.detector import create_detector, Detector
from bcdi.experiment.setup import Setup


class TestCast(unittest.TestCase):
    """
    Tests on the function utilities.cast.

    def cast(
    val: Union[float, List, np.ndarray], target_type: type = float
    ) -> Union[float, List, np.ndarray]:
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
            util.cast(1 - 2 * 1j, target_type=float)

    def test_number_str(self):
        out = util.cast("2.0", target_type=float)
        self.assertTrue(np.isclose(out, 2.0))

    def test_not_a_number_str(self):
        with self.assertRaises(ValueError):
            util.cast("two", target_type=float)


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


class TestCreateRepr(unittest.TestCase):
    """
    Tests on the function utilities.create_repr.

    def create_repr(obj: type) -> str
    """

    def test_detector(self):
        det = create_detector(name="Maxipix")
        valid = (
            'Maxipix(name="Maxipix", rootdir=None, datadir=None, savedir=None, '
            'template_file=None, template_imagefile=None, specfile=None, '
            'sample_name=None, roi=[0, 516, 0, 516], sum_roi=[0, 516, 0, 516], '
            'binning=(1, 1, 1), )'
        )
        out = util.create_repr(obj=det, cls=Detector)
        self.assertEqual(out, valid)

    def test_setup(self):
        setup = Setup(beamline_name="34ID", detector_name="Timepix")
        valid = (
            'Setup(beamline_name="34ID", detector_name="Timepix", '
            'beam_direction=[1.0, 0.0, 0.0], energy=None, distance=None, '
            'outofplane_angle=None, inplane_angle=None, tilt_angle=None, '
            'rocking_angle=None, grazing_angle=None, )'
        )
        out = util.create_repr(obj=setup, cls=Setup)
        self.assertEqual(out, valid)

    def test_not_a_class(self):
        det = create_detector(name="Maxipix")
        with self.assertRaises(TypeError):
            util.create_repr(obj=det, cls="Detector")

    def test_empty_init(self):
        valid = 'Empty()'

        class Empty:
            """This is an empty class"""

        out = util.create_repr(obj=Empty(), cls=Empty)
        self.assertEqual(out, valid)


class TestFormatRepr(unittest.TestCase):
    """
    Tests on the function utilities.format_repr.

    def format_repr(field: str, value: Optional[Any]) -> str
    """

    def test_field_undefined(self):
        with self.assertRaises(TypeError):
            util.format_repr(None, "test")

    def test_str(self):
        out = util.format_repr("field", "test")
        self.assertEqual(out, 'field="test", ')

    def test_float(self):
        out = util.format_repr("field", 0.4)
        self.assertEqual(out, 'field=0.4, ')

    def test_none(self):
        out = util.format_repr("field", None)
        self.assertEqual(out, 'field=None, ')

    def test_tuple(self):
        out = util.format_repr("field", (1.0, 2.0))
        self.assertEqual(out, 'field=(1.0, 2.0), ')


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


class TestUnpackArray(unittest.TestCase):
    """
    Tests on the function utilities.unpack_array.

    def unpack_array(array: Any) -> Any
    """

    def test_array_longer_than_one(self):
        arr = util.unpack_array(np.ones(2))
        self.assertTrue(np.array_equal(arr, np.ones(2)))

    def test_array_length_one(self):
        arr = util.unpack_array(np.array([33]))
        self.assertEqual(arr, 33)

    def test_list_length_one(self):
        arr = util.unpack_array([2])
        self.assertEqual(arr, 2)

    def test_str_length_one(self):
        val = util.unpack_array("s")
        self.assertEqual(val, "s")

    def test_none(self):
        val = util.unpack_array(None)
        self.assertEqual(val, None)

    def test_number(self):
        val = util.unpack_array(5)
        self.assertEqual(val, 5)


class TestNdarrayToList(unittest.TestCase):
    """
    Tests on the function utilities.ndarray_to_list.

    def ndarray_to_list(array: np.ndarray) -> List
    """

    def test_not_an_array(self):
        with self.assertRaises(TypeError):
            util.ndarray_to_list(array=2.3)

    def test_none(self):
        with self.assertRaises(TypeError):
            util.ndarray_to_list(array=None)

    def test_1d_array_int(self):
        valid = [1, 2, 3]
        out = util.ndarray_to_list(array=np.array(valid))
        self.assertTrue(out == valid)

    def test_1d_array_float(self):
        valid = [1.12333333333333333333333333, 2.77, 3.5]
        out = util.ndarray_to_list(array=np.array(valid))
        self.assertTrue(out == valid)

    def test_2d_array_int(self):
        valid = [[1, 2, 3], [1.2, 3.333333333, 0]]
        out = util.ndarray_to_list(array=np.array(valid))
        self.assertTrue(out == valid)


if __name__ == "__main__":
    run_tests(TestInRange)
    run_tests(TestIsFloat)
    run_tests(TestFindFile)
    run_tests(TestGaussianWindow)
    run_tests(TestUnpackArray)
    run_tests(TestCreateRepr)
    run_tests(TestFormatRepr)
    run_tests(TestNdarrayToList)
