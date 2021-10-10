# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#   (c) 06/2021-present : DESY CFEL
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np
import unittest
import bcdi.utils.image_registration as reg


def run_tests(test_class):
    suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


class TestGetShift(unittest.TestCase):
    """
    Tests on the function image_registration.get_shift.

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
        shifted_array[2:, 2:] = 1+1j
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
            support_threshold=0.5
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
                precision=2.3
            )

    def test_precision_null(self):
        with self.assertRaises(ValueError):
            reg.get_shift(
                reference_array=self.reference_array,
                shifted_array=self.shifted_array,
                precision=0
            )

    def test_precision_None(self):
        with self.assertRaises(ValueError):
            reg.get_shift(
                reference_array=self.reference_array,
                shifted_array=self.shifted_array,
                precision=None
            )

    def test_precision_min_allowed(self):
        shifts = reg.get_shift(
                reference_array=self.reference_array,
                shifted_array=self.shifted_array,
                precision=1
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
                shift_method="wrong"
            )


class TestShiftArray(unittest.TestCase):
    """
    Tests on the function image_registration.shift_array.

    def shift_array(
    array: np.ndarray, shift: Sequence[float], interpolation_method: str = "subpixel"
    ) -> np.ndarray:
    """

    def setUp(self):
        # executed before each test
        reference_array = np.zeros((5, 5), dtype=complex)
        reference_array[1:4, 1:4] = 1 + 1j
        shifted_array = np.zeros((5, 5), dtype=complex)
        shifted_array[2:, 2:] = 1+1j
        self.reference_array = reference_array
        self.shifted_array = shifted_array
        self.shifts = reg.get_shift(
                reference_array=self.reference_array,
                shifted_array=self.shifted_array
        )

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
    run_tests(TestGetShift)
