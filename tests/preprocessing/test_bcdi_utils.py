# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import unittest

import numpy as np

from bcdi.preprocessing.bcdi_utils import find_bragg
from bcdi.utils.utilities import gaussian_window
from tests.config import run_tests


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
        data[:-1, -13:, 17:30] = gaussian_window(window_shape=(3, 13, 13))
        self.data = data

    def test_max(self):
        peaks = find_bragg(data=self.data)
        self.assertTrue(peaks["max"] == (1, 25, 23))

    def test_com(self):
        peaks = find_bragg(data=self.data)
        self.assertTrue(peaks["com"] == (1, 25, 23))

    def test_maxcom(self):
        peaks = find_bragg(data=self.data)
        self.assertTrue(peaks["max_com"] == (1, 25, 23))

    def test_binning_wrong_length(self):
        with self.assertRaises(ValueError):
            find_bragg(data=self.data, binning=(2, 2))

    def test_binning_wrong_type(self):
        with self.assertRaises(TypeError):
            find_bragg(data=self.data, peak_method="max", binning=(2, 2, 2.3))

    def test_binning(self):
        peaks = find_bragg(data=self.data, binning=(2, 1, 1))
        self.assertTrue(peaks["max"] == (2, 25, 23))

    def test_binning_2(self):
        peaks = find_bragg(data=self.data, binning=(3, 3, 4))
        self.assertTrue(peaks["max"] == (3, 75, 92))

    def test_roi_wrong_length(self):
        with self.assertRaises(ValueError):
            find_bragg(data=self.data, roi=(2, 1, 1))

    def test_roi_wrong_type(self):
        with self.assertRaises(TypeError):
            find_bragg(data=self.data, roi=(2, 1, 1, 1.2))

    def test_roi(self):
        peaks = find_bragg(data=self.data, roi=(3, 3, 4, 2))
        self.assertTrue(peaks["max"] == (1, 28, 27))

    def test_bin_roi(self):
        peaks = find_bragg(data=self.data, roi=(3, 3, 4, 2), binning=(3, 3, 4))
        self.assertTrue(peaks["max"] == (3, 78, 96))


if __name__ == "__main__":
    run_tests(TestFindBragg)
