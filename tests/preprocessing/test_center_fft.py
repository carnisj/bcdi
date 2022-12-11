# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import logging
import pathlib
import tempfile
import unittest
from typing import Tuple
import numpy as np

from bcdi.preprocessing.center_fft import CenteringFactory
from bcdi.utils.utilities import gaussian_window
from tests.config import run_tests

module_logger = logging.getLogger(__name__)


def create_data(shape: Tuple[int, int, int]) -> np.ndarray:
    """Create a fake data where the max is not in the center."""
    if any(val < 3 for val in shape):
        raise ValueError("shape should be >= 3 in all dimensions")
    data = np.zeros(shape)
    data[shape[0] // 2 + 1, shape[1] // 2 + 1, shape[2] // 2 + 1] = 1
    data[shape[0] // 2 - 3 : shape[0] // 2, shape[1] // 2 + 1, shape[2] // 2 + 1] = 0.5
    data[shape[0] // 2 + 1, shape[1] // 2 - 3 : shape[1] // 2, shape[2] // 2 + 1] = 0.5

    return data


class TestCenteringFactory(unittest.TestCase):
    def setUp(self) -> None:
        self.data_shape = (7, 7, 7)
        self.factory = CenteringFactory(
            data=create_data(self.data_shape),
            binning=(1, 1, 1),
            preprocessing_binning=(1, 1, 1),
            roi=(0, self.data_shape[1], 0, self.data_shape[2]),
            fix_bragg=None,
            fft_option="crop_sym_ZYX",
            pad_size=None,
            centering_method="max",
            q_values=None,
            logger=module_logger,
        )

    def test_find_center_max(self):
        self.assertEqual(
            self.factory.center_position,
            (
                4,
                4,
                4,
            ),
        )

    def test_find_center_com(self):
        center = self.factory.find_center(
            data=create_data(self.data_shape), method="com"
        )
        self.assertEqual(
            center,
            (3, 3, 4),
        )

    def test_find_center_max_com(self):
        center = self.factory.find_center(
            data=create_data(self.data_shape), method="max_com"
        )
        self.assertEqual(
            center,
            (4, 2, 4),
        )

    def test_get_max_symmetrical_window(self):
        self.assertEqual(
            self.factory.max_symmetrical_window,
            (6, 6, 6),
        )

    def test_check_center_position(self):
        self.factory.max_symmetrical_window = (0, 6, 6)
        self.factory.check_center_position()
        self.assertEqual(
            self.factory.fft_option,
            "skip",
        )


if __name__ == "__main__":
    run_tests(TestCenteringFactory)
