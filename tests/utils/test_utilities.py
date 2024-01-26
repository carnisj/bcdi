# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import os
import unittest

import numpy as np
from pyfakefs import fake_filesystem_unittest

import bcdi.utils.utilities as util
from tests.config import load_config, run_tests

parameters, skip_tests = load_config("preprocessing")


class TestConvertStrTarget(unittest.TestCase):
    """
    Tests on the function utilities.convert_str_target.

    def convert_str_target(
        value: Any, target: str, conversion_table: Optional[Dict[str, Any]] = None
    ) -> Any:
    """

    def test_conversion_table_not_provided(self):
        value = "some object"
        with self.assertRaises(ValueError):
            util.convert_str_target(value, target="blue")

    def test_conversion_table_provided(self):
        target = "blue"
        conversion_table = {"blue": "sky"}
        out = util.convert_str_target(
            "blue", target=target, conversion_table=conversion_table
        )
        self.assertEqual(out, conversion_table[target])

    def test_value_none_str(self):
        target = "none"
        out = util.convert_str_target("none", target=target)
        self.assertEqual(out, None)

    def test_value_true_str(self):
        target = "tRue"
        out = util.convert_str_target("true", target=target)
        self.assertEqual(out, True)

    def test_value_false_str(self):
        target = "fALSe"
        out = util.convert_str_target("False", target=target)
        self.assertEqual(out, False)

    def test_value_none_mixed(self):
        target = "none"
        expected = [
            [None, "blade", 1],
            None,
            [None, 2, [1, 2, None]],
            {
                1: {
                    "time": None,
                    "value": None,
                    "blade": [1, None, int],
                }
            },
        ]
        out = util.convert_str_target(
            [
                ["NONE", "blade", 1],
                "noNe",
                (None, 2, [1, 2, "None"]),
                {
                    1: {
                        "time": None,
                        "value": "noNe",
                        "blade": (1, "None", int),
                    }
                },
            ],
            target=target,
        )
        self.assertEqual(out, expected)


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


class TestUpsample(unittest.TestCase):
    """
    Tests on the function utilities.upsample.

    def upsample(
    array: Union[np.ndarray, List], factor: int = 2, interp_order: int = 1
    ) -> np.ndarray:
    """

    def test_1d_array_factor_1_odd(self):
        array = np.arange(9)
        output = util.upsample(array=array, factor=1)
        self.assertTrue(np.allclose(output, array))

    def test_1d_array_factor_1_even(self):
        array = np.arange(8)
        output = util.upsample(array=array, factor=1)
        self.assertTrue(np.allclose(output, array))

    def test_1d_array_factor_2_odd(self):
        array = np.arange(3)
        expected = np.linspace(0, 2, num=6, endpoint=True)
        output = util.upsample(array=array, factor=2)
        self.assertTrue(np.allclose(output, expected))

    def test_1d_array_factor_2_even(self):
        array = np.arange(6)
        expected = np.linspace(0, 5, num=12, endpoint=True)
        output = util.upsample(array=array, factor=2)
        self.assertTrue(np.allclose(output, expected))

    def test_1d_array_complex_factor_2_even(self):
        array = np.arange(6) * (1 + 1j)
        expected = np.linspace(0, 5, num=12, endpoint=True) * (1 + 1j)
        output = util.upsample(array=array, factor=2)
        self.assertTrue(np.allclose(output, expected))

    def test_2d_array(self):
        array = np.asarray([np.arange(6) for _ in range(5)])
        expected = np.asarray([np.linspace(0, 5, num=12) for _ in range(10)])
        output = util.upsample(array=array, factor=2)
        self.assertTrue(np.allclose(output, expected))

    def test_2d_array_vertical(self):
        array = np.asarray([np.ones(6) * idx for idx in range(5)])
        expected = np.asarray([np.ones(12) * idx for idx in np.linspace(0, 4, num=10)])
        output = util.upsample(array=array, factor=2)
        self.assertTrue(np.allclose(output, expected))

    def test_3d_array(self):
        array = np.asarray([[np.arange(6) for _ in range(5)] for _ in range(4)])
        expected = np.asarray(
            [[np.linspace(0, 5, num=12) for _ in range(10)] for _ in range(8)]
        )
        output = util.upsample(array=array, factor=2)
        self.assertTrue(np.allclose(output, expected))


class TestGenerateFramesLogical(unittest.TestCase):
    """
    Tests on the function utilities.generate_frames_logical.

    def generate_frames_logical(
        nb_images: int, frames_pattern: Optional[List[int]]
    ) -> np.ndarray:
    """

    def test_nb_image_none(self) -> None:
        with self.assertRaises(ValueError):
            util.generate_frames_logical(nb_images=None, frames_pattern=[128])

    def test_nb_image_null(self) -> None:
        with self.assertRaises(ValueError):
            util.generate_frames_logical(nb_images=0, frames_pattern=[128])

    def test_frames_pattern_none(self) -> None:
        nb_images = 12
        expected = np.ones(nb_images, dtype=int)
        out = util.generate_frames_logical(nb_images=nb_images, frames_pattern=None)
        self.assertTrue(np.array_equal(expected, out))

    def test_frames_pattern_binary(self) -> None:
        nb_images = 6
        frames_pattern = [1, 0, 0, 1, 1, 1]
        expected = np.array([1, 0, 0, 1, 1, 1], dtype=int)
        out = util.generate_frames_logical(
            nb_images=nb_images, frames_pattern=frames_pattern
        )
        self.assertTrue(np.array_equal(expected, out))

    def test_frames_pattern_binary_wrong_length(self) -> None:
        nb_images = 6
        frames_pattern = [1, 0, 1, 1, 1]
        with self.assertRaises(ValueError):
            util.generate_frames_logical(
                nb_images=nb_images, frames_pattern=frames_pattern
            )

    def test_frames_pattern_list_of_indices_too_long(self) -> None:
        nb_images = 6
        frames_pattern = [0, 1, 2, 3, 4, 5, 6]
        with self.assertRaises(ValueError):
            util.generate_frames_logical(
                nb_images=nb_images, frames_pattern=frames_pattern
            )

    def test_frames_pattern_list_of_indices(self) -> None:
        nb_images = 6
        frames_pattern = [0, 3]
        expected = np.array([0, 1, 1, 0, 1, 1], dtype=int)
        out = util.generate_frames_logical(
            nb_images=nb_images, frames_pattern=frames_pattern
        )
        self.assertTrue(np.array_equal(expected, out))

    def test_frames_pattern_index_too_large(self) -> None:
        nb_images = 6
        frames_pattern = [0, 6]
        with self.assertRaises(ValueError):
            util.generate_frames_logical(
                nb_images=nb_images, frames_pattern=frames_pattern
            )

    def test_frames_pattern_duplicated_indices(self) -> None:
        nb_images = 6
        frames_pattern = [0, 2, 2]
        with self.assertRaises(ValueError):
            util.generate_frames_logical(
                nb_images=nb_images, frames_pattern=frames_pattern
            )


class TestUpdateFramesLogical(unittest.TestCase):
    """
    Tests on the function utilities.update_frames_logical.

    def update_frames_logical(
        frames_logical: np.ndarray, logical_subset: np.ndarray
    ) -> np.ndarray:
    """

    def test_inconsitency_with_length_of_logical_subset(self):
        frames_logical = np.array([0, 1, 1, 1])
        logical_subset = np.array([1, 1])
        with self.assertRaises(ValueError):
            util.update_frames_logical(
                frames_logical=frames_logical, logical_subset=logical_subset
            )

    def test_remove_1_frame(self):
        frames_logical = np.array([0, 1, 1, 1])
        logical_subset = np.array([1, 1, 0])
        expected = np.array([0, 1, 1, 0])
        out = util.update_frames_logical(
            frames_logical=frames_logical, logical_subset=logical_subset
        )
        self.assertTrue(np.array_equal(expected, out))


class TestApplyLogicalArray(unittest.TestCase):
    """
    Tests on the function utilities.apply_logical_array.

    def apply_logical_array(
        arrays: Union[np.ndarray, Tuple[np.ndarray, ...]],
        frames_logical: Optional[np.ndarray],
    ) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    """

    def setUp(self) -> None:
        self.frames_logical = np.array([1, 0, 1, 1, 1, 1, 0])

    def test_single_array(self):
        expected = np.array([0, 2, 3, 4, 5])
        out = util.apply_logical_array(
            arrays=np.arange(len(self.frames_logical)),
            frames_logical=self.frames_logical,
        )
        self.assertTrue(np.array_equal(expected, out))

    def test_tuple_of_arrays(self):
        expected = np.array([0, 2, 3, 4, 5]), np.array([0, -2, -3, -4, -5])
        out = util.apply_logical_array(
            arrays=(
                np.arange(len(self.frames_logical)),
                -np.arange(len(self.frames_logical)),
            ),
            frames_logical=self.frames_logical,
        )
        self.assertIsInstance(out, tuple)
        for idx, val in enumerate(out):
            self.assertTrue(np.array_equal(expected[idx], val))

    def test_input_is_a_number(self):
        expected = 3
        out = util.apply_logical_array(
            arrays=expected,
            frames_logical=self.frames_logical,
        )
        self.assertEqual(out, expected)

    def test_mixed_tuple(self):
        expected = (3, np.array([0, 2, 3, 4, 5]))
        out = util.apply_logical_array(
            arrays=(3, np.arange(len(self.frames_logical))),
            frames_logical=self.frames_logical,
        )
        self.assertIsInstance(out, tuple)
        self.assertEqual(out[0], expected[0])
        self.assertTrue(np.array_equal(expected[1], out[1]))


if __name__ == "__main__":
    run_tests(TestInRange)
    run_tests(TestFindFile)
    run_tests(TestGaussianWindow)
    run_tests(TestUnpackArray)
    run_tests(TestUpsample)
    run_tests(TestGenerateFramesLogical)
    run_tests(TestUpdateFramesLogical)
    run_tests(TestApplyLogicalArray)
