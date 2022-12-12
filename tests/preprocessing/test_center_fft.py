# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import logging
import unittest
from typing import Tuple
import numpy as np

from bcdi.preprocessing import center_fft
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
        self.factory = center_fft.CenteringFactory(
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

    def test_data_shape(self):
        with self.assertRaises(ValueError):
            self.factory.data_shape = (2, 2)

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

    def test_find_center_fix_bragg(self):
        self.factory.fix_bragg = [3, 3, 3]
        center = self.factory.find_center(
            data=create_data(self.data_shape), method="max_com"
        )
        self.assertEqual(
            center,
            (3, 3, 3),
        )

    def test_find_center_fix_bragg_wrong_length(self):
        self.factory.fix_bragg = [3, 3]
        with self.assertRaises(ValueError):
            self.factory.find_center(
                data=create_data(self.data_shape), method="max_com"
            )

    def test_find_center_fix_bragg_binning(self):
        self.factory.fix_bragg = [6, 6, 6]
        self.factory.binning = [1, 2, 1]
        center = self.factory.find_center(
            data=create_data(self.data_shape), method="max_com"
        )
        self.assertEqual(
            center,
            (6, 3, 6),
        )

    def test_find_center_fix_bragg_preprocessing_binning(self):
        self.factory.fix_bragg = [6, 6, 6]
        self.factory.binning = [1, 2, 1]
        self.factory.preprocessing_binning = [1, 1, 3]
        center = self.factory.find_center(
            data=create_data(self.data_shape), method="max_com"
        )
        self.assertEqual(
            center,
            (6, 3, 2),
        )

    def test_find_center_fix_bragg_roi(self):
        self.factory.fix_bragg = [6, 6, 6]
        self.factory.roi = (1, self.data_shape[1], 3, self.data_shape[2])
        center = self.factory.find_center(
            data=create_data(self.data_shape), method="max_com"
        )
        self.assertEqual(
            center,
            (6, 5, 3),
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

    def test_round_sequence_to_int(self):
        self.assertEqual(self.factory.round_sequence_to_int((2.3, -1.1)), (2, -1))

    def test_round_sequence_to_int_not_a_sequence(self):
        with self.assertRaises(TypeError):
            self.factory.round_sequence_to_int(2.3)

    def test_round_sequence_to_int_not_numbers(self):
        with self.assertRaises(ValueError):
            self.factory.round_sequence_to_int([2.3, "c"])

    def test_round_sequence_to_int_none(self):
        with self.assertRaises(ValueError):
            self.factory.round_sequence_to_int([2.3, None])

    def test_round_sequence_to_int_nan(self):
        with self.assertRaises(ValueError):
            self.factory.round_sequence_to_int([2.3, np.nan])

    def test_get_centering_instance(self):
        self.assertIsInstance(
            self.factory.get_centering_instance(), center_fft.CenterFFT
        )

    def test_get_centering_instance_crop_sym_ZYX(self):
        self.factory.fft_option = "crop_sym_ZYX"
        self.assertIsInstance(
            self.factory.get_centering_instance(), center_fft.CenterFFTCropSymZYX
        )

    def test_get_centering_instance_crop_asym_ZYX(self):
        self.factory.fft_option = "crop_asym_ZYX"
        self.assertIsInstance(
            self.factory.get_centering_instance(), center_fft.CenterFFTCropAsymZYX
        )

    def test_get_centering_instance_pad_sym_Z_crop_sym_YX(self):
        self.factory.fft_option = "pad_sym_Z_crop_sym_YX"
        self.assertIsInstance(
            self.factory.get_centering_instance(), center_fft.CenterFFTPadSymZCropSymYX
        )

    def test_get_centering_instance_pad_sym_Z_crop_asym_YX(self):
        self.factory.fft_option = "pad_sym_Z_crop_asym_YX"
        self.assertIsInstance(
            self.factory.get_centering_instance(), center_fft.CenterFFTPadSymZCropAsymYX
        )

    def test_get_centering_instance_pad_asym_Z_crop_sym_YX(self):
        self.factory.fft_option = "pad_asym_Z_crop_sym_YX"
        self.assertIsInstance(
            self.factory.get_centering_instance(), center_fft.CenterFFTPadAsymZCropSymYX
        )

    def test_get_centering_instance_pad_asym_Z_crop_asym_YX(self):
        self.factory.fft_option = "pad_asym_Z_crop_asym_YX"
        self.assertIsInstance(
            self.factory.get_centering_instance(),
            center_fft.CenterFFTPadAsymZCropAsymYX,
        )

    def test_get_centering_instance_pad_sym_Z(self):
        self.factory.fft_option = "pad_sym_Z"
        self.assertIsInstance(
            self.factory.get_centering_instance(), center_fft.CenterFFTPadSymZ
        )

    def test_get_centering_instance_pad_asym_Z(self):
        self.factory.fft_option = "pad_asym_Z"
        self.assertIsInstance(
            self.factory.get_centering_instance(), center_fft.CenterFFTPadAsymZ
        )

    def test_get_centering_instance_pad_sym_ZYX(self):
        self.factory.fft_option = "pad_sym_ZYX"
        self.assertIsInstance(
            self.factory.get_centering_instance(), center_fft.CenterFFTPadSymZYX
        )

    def test_get_centering_instance_pad_asym_ZYX(self):
        self.factory.fft_option = "pad_asym_ZYX"
        self.assertIsInstance(
            self.factory.get_centering_instance(), center_fft.CenterFFTPadAsymZYX
        )

    def test_get_centering_instance_skip(self):
        self.factory.fft_option = "skip"
        self.assertIsInstance(
            self.factory.get_centering_instance(), center_fft.SkipCentering
        )

    def test_get_centering_instance_not_implemented(self):
        self.factory.fft_option = "unknown"
        with self.assertRaises(ValueError):
            self.factory.get_centering_instance()


if __name__ == "__main__":
    run_tests(TestCenteringFactory)
