# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np
import unittest

from bcdi.preprocessing.bcdi_utils import find_bragg
from bcdi.utils.utilities import gaussian_window


def run_tests(test_class):
    suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


class TestFindBragg(unittest.TestCase):
    """
    Tests related to the function find_bragg.

    def find_bragg(
            data: np.ndarray,
            peak_method: str,
            roi: Optional[Tuple[int, int, int, int]] = None,
            binning: Optional[Tuple[int, ...]] = None,
    ) -> Tuple[int, ...]:
    """
    def setUp(self) -> None:
        data = np.zeros((4, 32, 32))
        data[:-1, -12:, 18:30] = gaussian_window(window_shape=(3, 12, 12))
        self.data = data

    def test_no_method(self):
        with self.assertRaises(ValueError):
            find_bragg(data=self.data, peak_method="wrong")

    def test_max(self):
        peak = find_bragg(data=self.data, peak_method="max")
        self.assertTrue(peak == (1, 26, 24))

    def test_com(self):
        peak = find_bragg(data=self.data, peak_method="com")
        self.assertTrue(peak == (1, 26, 24))

    def test_maxcom(self):
        peak = find_bragg(data=self.data, peak_method="maxcom")
        self.assertTrue(peak == (1, 26, 24))

    def test_binning_wrong_length(self):
        with self.assertRaises(ValueError):
            find_bragg(data=self.data, peak_method="max", binning=(2, 2))

    def test_binning_wrong_type(self):
        with self.assertRaises(TypeError):
            find_bragg(data=self.data, peak_method="max", binning=(2, 2, 2.3))

    def test_binning(self):
        peak = find_bragg(data=self.data, peak_method="max", binning=(2, 1, 1))
        self.assertTrue(peak == (2, 26, 24))

    def test_binning_2(self):
        peak = find_bragg(data=self.data, peak_method="max", binning=(3, 3, 4))
        self.assertTrue(peak == (3, 78, 96))

    def test_roi_wrong_length(self):
        with self.assertRaises(ValueError):
            find_bragg(data=self.data, peak_method="max", roi=(2, 1, 1))

    def test_roi_wrong_type(self):
        with self.assertRaises(TypeError):
            find_bragg(data=self.data, peak_method="max", roi=(2, 1, 1, 1.2))

    def test_roi(self):
        peak = find_bragg(data=self.data, peak_method="max", roi=(3, 3, 4, 2))
        self.assertTrue(peak == (1, 29, 28))

    def test_bin_roi(self):
        peak = find_bragg(
            data=self.data,
            peak_method="max",
            roi=(3, 3, 4, 2),
            binning=(3, 3, 4)
        )
        self.assertTrue(peak == (3, 81, 100))


if __name__ == "__main__":
    run_tests(TestFindBragg)
