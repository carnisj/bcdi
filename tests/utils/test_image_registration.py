# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import unittest

import numpy as np

import bcdi.utils.image_registration as reg
from tests.config import run_tests


class TestCalcNewPositions(unittest.TestCase):
    """
    Tests on the function calc_new_positions.

    def calc_new_positions(old_positions: list, shift: Sequence[float]) -> np.ndarray
    """

    def setUp(self):
        self.shapes = ((3,), (4,), (2, 2), (1, 2, 2))
        self.shifts = ((1,), (-0.2,), (2.3, -0.1), (1.1, 0.3, 0))

    def test_ndim_no_shift(self):
        correct = (
            np.array([[-2], [-1], [0]]),
            np.array([[-2], [-1], [0], [1]]),
            np.array(
                [
                    [-1, -1],
                    [-1, 0],
                    [0, -1],
                    [0, 0],
                ]
            ),
            np.array([[-1, -1, -1], [-1, -1, 0], [-1, 0, -1], [-1, 0, 0]]),
        )
        for index, shape in enumerate(self.shapes):
            with self.subTest():
                old_pos = [np.arange(-val // 2, val // 2) for val in shape]
                new_pos = reg.calc_new_positions(old_pos, shift=(0,) * len(shape))
                self.assertTrue(np.allclose(new_pos, correct[index]))

    def test_ndim_with_shift(self):
        correct = (
            np.array([[-3], [-2], [-1]]),
            np.array([[-1.8], [-0.8], [0.2], [1.2]]),
            np.array(
                [
                    [-3.3, -0.9],
                    [-3.3, 0.1],
                    [-2.3, -0.9],
                    [-2.3, 0.1],
                ]
            ),
            np.array(
                [[-2.1, -1.3, -1], [-2.1, -1.3, 0], [-2.1, -0.3, -1], [-2.1, -0.3, 0]]
            ),
        )
        for index, shape in enumerate(self.shapes):
            with self.subTest():
                old_pos = [np.arange(-val // 2, val // 2) for val in shape]
                new_pos = reg.calc_new_positions(old_pos, shift=self.shifts[index])
                self.assertTrue(np.allclose(new_pos, correct[index]))

    def test_empty_positions(self):
        with self.assertRaises(ValueError):
            reg.calc_new_positions([], shift=[])

    def test_wrong_shift_length(self):
        with self.assertRaises(ValueError):
            reg.calc_new_positions([np.arange(-2, 1)], shift=[1, 2])

    def test_wrong_shift_none(self):
        with self.assertRaises(ValueError):
            reg.calc_new_positions([np.arange(-2, 1)], shift=None)


class TestGetShift2D(unittest.TestCase):
    """
    Tests on the function image_registration.get_shift for 2D arrays.

    def get_shift(
    reference_array: np.ndarray,
    shifted_array: np.ndarray,
    shift_method: str = "modulus",
    precision: int = 1000,
    support_threshold: Union[None, float] = None,
    verbose: bool = True,
    ) -> Sequence[float]:
    """

    def setUp(self):
        # executed before each test
        reference_array = np.zeros((5, 5), dtype=complex)
        reference_array[1:4, 1:4] = 1 + 1j
        shifted_array = np.zeros((5, 5), dtype=complex)
        shifted_array[2:, 2:] = 1 + 1j
        self.reference_array = reference_array
        self.shifted_array = shifted_array

    def test_method_modulus(self):
        shifts = reg.get_shift(
            reference_array=self.reference_array,
            shifted_array=self.shifted_array,
            shift_method="modulus",
        )
        self.assertTrue(
            np.allclose(
                np.asarray(shifts),
                np.array([-1.0, -1.0]),
                rtol=1e-09,
                atol=1e-09,
            )
        )

    def test_method_raw(self):
        shifts = reg.get_shift(
            reference_array=self.reference_array,
            shifted_array=self.shifted_array,
            shift_method="raw",
        )
        self.assertTrue(
            np.allclose(
                np.asarray(shifts),
                np.array([-1.0, -1.0]),
                rtol=1e-09,
                atol=1e-09,
            )
        )

    def test_method_support(self):
        shifts = reg.get_shift(
            reference_array=self.reference_array,
            shifted_array=self.shifted_array,
            shift_method="support",
            support_threshold=0.5,
        )
        self.assertTrue(
            np.allclose(
                np.asarray(shifts),
                np.array([-1.0, -1.0]),
                rtol=1e-09,
                atol=1e-09,
            )
        )

    def test_method_support_none(self):
        with self.assertRaises(ValueError):
            reg.get_shift(
                reference_array=self.reference_array,
                shifted_array=self.shifted_array,
                shift_method="support",
            )

    def test_precision_float(self):
        with self.assertRaises(TypeError):
            reg.get_shift(
                reference_array=self.reference_array,
                shifted_array=self.shifted_array,
                precision=2.3,
            )

    def test_precision_null(self):
        with self.assertRaises(ValueError):
            reg.get_shift(
                reference_array=self.reference_array,
                shifted_array=self.shifted_array,
                precision=0,
            )

    def test_precision_None(self):
        with self.assertRaises(ValueError):
            reg.get_shift(
                reference_array=self.reference_array,
                shifted_array=self.shifted_array,
                precision=None,
            )

    def test_precision_min_allowed(self):
        shifts = reg.get_shift(
            reference_array=self.reference_array,
            shifted_array=self.shifted_array,
            precision=1,
        )
        self.assertTrue(
            np.allclose(
                np.asarray(shifts),
                np.array([-1.0, -1.0]),
                rtol=1e-09,
                atol=1e-09,
            )
        )

    def test_wrong_method_name(self):
        with self.assertRaises(ValueError):
            reg.get_shift(
                reference_array=self.reference_array,
                shifted_array=self.shifted_array,
                shift_method="wrong",
            )


class TestGetShift3D(unittest.TestCase):
    """
    Tests on the function image_registration.get_shift for 3D arrays.

    def get_shift(
    reference_array: np.ndarray,
    shifted_array: np.ndarray,
    shift_method: str = "modulus",
    precision: int = 1000,
    support_threshold: Union[None, float] = None,
    verbose: bool = True,
    ) -> Sequence[float]:
    """

    def setUp(self):
        # executed before each test
        reference_array = np.zeros((5, 5, 5), dtype=complex)
        reference_array[1:4, 1:4, 1:4] = 1 + 1j
        shifted_array = np.zeros((5, 5, 5), dtype=complex)
        shifted_array[2:, 2:, 0:3] = 1 + 1j
        self.reference_array = reference_array
        self.shifted_array = shifted_array

    def test_method_modulus(self):
        shifts = reg.get_shift(
            reference_array=self.reference_array,
            shifted_array=self.shifted_array,
            shift_method="modulus",
        )
        self.assertTrue(
            np.allclose(
                np.asarray(shifts),
                np.array([-1.0, -1.0, 1.0]),
                rtol=1e-09,
                atol=1e-09,
            )
        )

    def test_method_raw(self):
        shifts = reg.get_shift(
            reference_array=self.reference_array,
            shifted_array=self.shifted_array,
            shift_method="raw",
        )
        self.assertTrue(
            np.allclose(
                np.asarray(shifts),
                np.array([-1.0, -1.0, 1.0]),
                rtol=1e-09,
                atol=1e-09,
            )
        )

    def test_method_support(self):
        shifts = reg.get_shift(
            reference_array=self.reference_array,
            shifted_array=self.shifted_array,
            shift_method="support",
            support_threshold=0.5,
        )
        self.assertTrue(
            np.allclose(
                np.asarray(shifts),
                np.array([-1.0, -1.0, 1.0]),
                rtol=1e-09,
                atol=1e-09,
            )
        )

    def test_method_support_none(self):
        with self.assertRaises(ValueError):
            reg.get_shift(
                reference_array=self.reference_array,
                shifted_array=self.shifted_array,
                shift_method="support",
            )

    def test_precision_float(self):
        with self.assertRaises(TypeError):
            reg.get_shift(
                reference_array=self.reference_array,
                shifted_array=self.shifted_array,
                precision=2.3,
            )

    def test_precision_null(self):
        with self.assertRaises(ValueError):
            reg.get_shift(
                reference_array=self.reference_array,
                shifted_array=self.shifted_array,
                precision=0,
            )

    def test_precision_None(self):
        with self.assertRaises(ValueError):
            reg.get_shift(
                reference_array=self.reference_array,
                shifted_array=self.shifted_array,
                precision=None,
            )

    def test_precision_min_allowed(self):
        shifts = reg.get_shift(
            reference_array=self.reference_array,
            shifted_array=self.shifted_array,
            precision=1,
        )
        self.assertTrue(
            np.allclose(
                np.asarray(shifts),
                np.array([-1.0, -1.0, 1.0]),
                rtol=1e-09,
                atol=1e-09,
            )
        )

    def test_wrong_method_name(self):
        with self.assertRaises(ValueError):
            reg.get_shift(
                reference_array=self.reference_array,
                shifted_array=self.shifted_array,
                shift_method="wrong",
            )


class TestInterpRgiTranslation2D(unittest.TestCase):
    """
    Tests on the function image_registration.interp_rgi_translation for 2D arrays.

    def interp_rgi_translation(array: np.ndarray, shift: Sequence[float]) -> np.ndarray
    """

    def setUp(self):
        # executed before each test
        reference_array = np.zeros((5, 5), dtype=complex)
        reference_array[1:4, 1:4] = 1 + 1j
        self.reference_array = reference_array
        shifted_array = np.zeros((5, 5), dtype=complex)
        shifted_array[2:, 2:] = 1 + 1j
        self.shifted_array = shifted_array
        self.shifts = (-1.0, -1.0)

    def test_output_dtype(self):
        aligned_array = reg.interp_rgi_translation(
            array=self.shifted_array,
            shift=self.shifts,
        )
        self.assertEqual(aligned_array.dtype, self.shifted_array.dtype)

    def test_method_rgi(self):
        aligned_array = reg.interp_rgi_translation(
            array=self.shifted_array,
            shift=self.shifts,
        )
        self.assertTrue(
            np.allclose(
                self.reference_array,
                aligned_array,
                rtol=1e-09,
                atol=1e-09,
            )
        )


class TestInterpRgiTranslation3D(unittest.TestCase):
    """
    Tests on the function image_registration.interp_rgi_translation for 3D arrays.

    def interp_rgi_translation(array: np.ndarray, shift: Sequence[float]) -> np.ndarray
    """

    def setUp(self):
        # executed before each test
        reference_array = np.zeros((5, 5, 5), dtype=complex)
        reference_array[1:4, 1:4, 1:4] = 1 + 1j
        self.reference_array = reference_array
        shifted_array = np.zeros((5, 5, 5), dtype=complex)
        shifted_array[2:, 2:, 0:3] = 1 + 1j
        self.shifted_array = shifted_array
        self.shifts = (-1.0, -1.0, 1.0)

    def test_output_dtype(self):
        aligned_array = reg.interp_rgi_translation(
            array=self.shifted_array,
            shift=self.shifts,
        )
        self.assertEqual(aligned_array.dtype, self.shifted_array.dtype)

    def test_method_rgi(self):
        aligned_array = reg.interp_rgi_translation(
            array=self.shifted_array,
            shift=self.shifts,
        )
        self.assertTrue(
            np.allclose(
                self.reference_array,
                aligned_array,
                rtol=1e-09,
                atol=1e-09,
            )
        )


class TestShiftArray2D(unittest.TestCase):
    """
    Tests on the function image_registration.shift_array for 2D arrays.

    def shift_array(
    array: np.ndarray, shift: Sequence[float], interpolation_method: str = "subpixel"
    ) -> np.ndarray:
    """

    def setUp(self):
        # executed before each test
        reference_array = np.zeros((5, 5), dtype=complex)
        reference_array[1:4, 1:4] = 1 + 1j
        shifted_array = np.zeros((5, 5), dtype=complex)
        shifted_array[2:, 2:] = 1 + 1j
        self.reference_array = reference_array
        self.shifted_array = shifted_array
        self.shifts = reg.get_shift(
            reference_array=self.reference_array, shifted_array=self.shifted_array
        )

    def test_output_dtype(self):
        aligned_array = reg.shift_array(
            array=self.shifted_array,
            shift=self.shifts,
            interpolation_method="subpixel",
        )
        self.assertEqual(aligned_array.dtype, self.shifted_array.dtype)

    def test_method_subpixel(self):
        aligned_array = reg.shift_array(
            array=self.shifted_array,
            shift=self.shifts,
            interpolation_method="subpixel",
        )
        self.assertTrue(
            np.allclose(
                self.reference_array,
                aligned_array,
                rtol=1e-09,
                atol=1e-09,
            )
        )

    def test_method_rgi(self):
        aligned_array = reg.shift_array(
            array=self.shifted_array,
            shift=self.shifts,
            interpolation_method="rgi",
        )
        self.assertTrue(
            np.allclose(
                self.reference_array,
                aligned_array,
                rtol=1e-09,
                atol=1e-09,
            )
        )

    def test_method_roll(self):
        aligned_array = reg.shift_array(
            array=self.shifted_array,
            shift=self.shifts,
            interpolation_method="roll",
        )
        self.assertTrue(
            np.allclose(
                self.reference_array,
                aligned_array,
                rtol=1e-09,
                atol=1e-09,
            )
        )


class TestShiftArray3D(unittest.TestCase):
    """
    Tests on the function image_registration.shift_array for 3D arrays.

    def shift_array(
    array: np.ndarray, shift: Sequence[float], interpolation_method: str = "subpixel"
    ) -> np.ndarray:
    """

    def setUp(self):
        # executed before each test
        reference_array = np.zeros((5, 5, 5), dtype=complex)
        reference_array[1:4, 1:4, 1:4] = 1 + 1j
        shifted_array = np.zeros((5, 5, 5), dtype=complex)
        shifted_array[2:, 2:, 0:3] = 1 + 1j
        self.reference_array = reference_array
        self.shifted_array = shifted_array
        self.shifts = reg.get_shift(
            reference_array=self.reference_array, shifted_array=self.shifted_array
        )

    def test_output_dtype(self):
        aligned_array = reg.shift_array(
            array=self.shifted_array,
            shift=self.shifts,
            interpolation_method="subpixel",
        )
        self.assertEqual(aligned_array.dtype, self.shifted_array.dtype)

    def test_method_subpixel(self):
        aligned_array = reg.shift_array(
            array=self.shifted_array,
            shift=self.shifts,
            interpolation_method="subpixel",
        )
        self.assertTrue(
            np.allclose(
                self.reference_array,
                aligned_array,
                rtol=1e-09,
                atol=1e-09,
            )
        )

    def test_method_rgi(self):
        aligned_array = reg.shift_array(
            array=self.shifted_array,
            shift=self.shifts,
            interpolation_method="rgi",
        )
        self.assertTrue(
            np.allclose(
                self.reference_array,
                aligned_array,
                rtol=1e-09,
                atol=1e-09,
            )
        )

    def test_method_roll(self):
        aligned_array = reg.shift_array(
            array=self.shifted_array,
            shift=self.shifts,
            interpolation_method="roll",
        )
        self.assertTrue(
            np.allclose(
                self.reference_array,
                aligned_array,
                rtol=1e-09,
                atol=1e-09,
            )
        )


if __name__ == "__main__":
    run_tests(TestCalcNewPositions)
    run_tests(TestGetShift2D)
    run_tests(TestGetShift2D)
    run_tests(TestShiftArray2D)
    run_tests(TestShiftArray3D)
