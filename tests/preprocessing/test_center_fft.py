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


class TestRoundSequenceToInt(unittest.TestCase):
    def test_round_sequence_to_int(self):
        self.assertEqual(center_fft.round_sequence_to_int((2.3, -1.1)), (2, -1))

    def test_round_sequence_to_int_not_a_sequence(self):
        with self.assertRaises(TypeError):
            center_fft.round_sequence_to_int(2.3)

    def test_round_sequence_to_int_not_numbers(self):
        with self.assertRaises(ValueError):
            center_fft.round_sequence_to_int([2.3, "c"])

    def test_round_sequence_to_int_none(self):
        with self.assertRaises(ValueError):
            center_fft.round_sequence_to_int([2.3, None])

    def test_round_sequence_to_int_nan(self):
        with self.assertRaises(ValueError):
            center_fft.round_sequence_to_int([2.3, np.nan])


class TestCenteringFactory(unittest.TestCase):
    def setUp(self) -> None:
        self.data_shape = (7, 7, 7)
        self.factory = center_fft.CenteringFactory(
            data=create_data(self.data_shape),
            binning=(1, 1, 1),
            preprocessing_binning=(1, 1, 1),
            roi=(0, self.data_shape[1], 0, self.data_shape[2]),
            bragg_peak=None,
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

    def test_find_center_bragg_peak(self):
        self.factory.bragg_peak = [3, 3, 3]
        center = self.factory.find_center(
            data=create_data(self.data_shape), method="max_com"
        )
        self.assertEqual(
            center,
            (3, 3, 3),
        )

    def test_find_center_bragg_peak_wrong_length(self):
        self.factory.bragg_peak = [3, 3]
        with self.assertRaises(ValueError):
            self.factory.find_center(
                data=create_data(self.data_shape), method="max_com"
            )

    def test_find_center_bragg_peak_binning(self):
        self.factory.bragg_peak = [6, 6, 6]
        self.factory.binning = [1, 2, 1]
        center = self.factory.find_center(
            data=create_data(self.data_shape), method="max_com"
        )
        self.assertEqual(
            center,
            (6, 3, 6),
        )

    def test_find_center_bragg_peak_preprocessing_binning(self):
        self.factory.bragg_peak = [6, 6, 6]
        self.factory.binning = [1, 2, 1]
        self.factory.preprocessing_binning = [1, 1, 3]
        center = self.factory.find_center(
            data=create_data(self.data_shape), method="max_com"
        )
        self.assertEqual(
            center,
            (6, 3, 2),
        )

    def test_find_center_bragg_peak_roi(self):
        self.factory.bragg_peak = [6, 6, 6]
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

    def test_log_q_values_at_center_bragg_peak(self):
        self.factory.bragg_peak = (3, 3, 3)
        expected = (
            "Peak intensity position with detector ROI and binning in the "
            f"detector plane: ({self.factory.center_position})"
        )
        with self.assertLogs() as captured:
            self.factory.log_q_values_at_center("max")
        self.assertEqual(len(captured.records), 1)
        self.assertEqual(captured.records[0].getMessage(), expected)

    def test_log_q_values_at_center_max(self):
        expected = f"Max at pixel (Z, Y, X): ({self.factory.center_position})"
        with self.assertLogs() as captured:
            self.factory.log_q_values_at_center("max")
        self.assertEqual(len(captured.records), 1)
        self.assertEqual(captured.records[0].getMessage(), expected)

    def test_log_q_values_at_center_com(self):
        expected = (
            f"Center of mass at pixel (Z, Y, X): ({self.factory.center_position})"
        )
        with self.assertLogs() as captured:
            self.factory.log_q_values_at_center("com")
        self.assertEqual(len(captured.records), 1)
        self.assertEqual(captured.records[0].getMessage(), expected)

    def test_log_q_values_at_center_maxcom(self):
        expected = f"Max_com at pixel (Z, Y, X): ({self.factory.center_position})"
        with self.assertLogs() as captured:
            self.factory.log_q_values_at_center("max_com")
        self.assertEqual(len(captured.records), 1)
        self.assertEqual(captured.records[0].getMessage(), expected)

    def test_log_q_values_at_center_maxcom_q_values(self):
        z0, y0, x0 = self.factory.center_position
        self.factory.q_values = [
            np.ones(self.data_shape[0]),
            2 * np.ones(self.data_shape[1]),
            3 * np.ones(self.data_shape[2]),
        ]
        expected = (
            f"Max_com at (qx, qz, qy): {self.factory.q_values[0][z0]:.5f}, "
            f"{self.factory.q_values[1][y0]:.5f}, {self.factory.q_values[2][x0]:.5f}"
        )
        with self.assertLogs() as captured:
            self.factory.log_q_values_at_center("max_com")
        self.assertEqual(len(captured.records), 1)
        self.assertEqual(captured.records[0].getMessage(), expected)


class TestCenterFFT(unittest.TestCase):
    def setUp(self) -> None:
        self.data_shape = (7, 7, 7)
        self.data = create_data(self.data_shape)
        self.instance = center_fft.CenteringFactory(
            data=self.data,
            binning=(1, 1, 1),
            preprocessing_binning=(1, 1, 1),
            roi=(0, self.data_shape[1], 0, self.data_shape[2]),
            bragg_peak=None,
            fft_option="crop_sym_ZYX",
            pad_size=None,
            centering_method="max",
            q_values=[
                np.ones(self.data_shape[0]),
                2 * np.ones(self.data_shape[1]),
                3 * np.ones(self.data_shape[2]),
            ],
            logger=module_logger,
        ).get_centering_instance()

    def test_init(self):
        self.assertTrue(all(val == 0 for val in self.instance.pad_width))
        self.assertEqual(
            self.instance.start_stop_indices,
            (
                0,
                self.instance.data_shape[0],
                0,
                self.instance.data_shape[1],
                0,
                self.instance.data_shape[2],
            ),
        )

    def test_pad_size_not_a_sequence(self):
        with self.assertRaises(TypeError):
            self.instance.pad_size = 3

    def test_pad_size_none(self):
        self.assertIsNone(self.instance.pad_size)

    def test_pad_size_sequence_wrong_length(self):
        with self.assertRaises(ValueError):
            self.instance.pad_size = (3, 3)

    def test_pad_size_violate_fft_requirements(self):
        with self.assertRaises(ValueError):
            self.instance.pad_size = (9, 8, 8)

    def test_pad_size_not_integers(self):
        with self.assertRaises(TypeError):
            self.instance.pad_size = (128, 128, 128.0)

    def test_start_stop_indices_not_a_sequence(self):
        with self.assertRaises(TypeError):
            self.instance.start_stop_indices = 3

    def test_start_stop_indices_sequence_wrong_length(self):
        with self.assertRaises(ValueError):
            self.instance.start_stop_indices = (3, 3)

    def test_start_stop_indices_sequence_not_integers(self):
        with self.assertRaises(TypeError):
            self.instance.start_stop_indices = (0, 6, 0, 6, 0, 6.0)

    def test_start_stop_indices_negative_index(self):
        with self.assertRaises(ValueError):
            self.instance.start_stop_indices = (0, 6, 0, 6, -1, 6)

    def test_start_stop_indices_index_larger_than_shape(self):
        with self.assertRaises(ValueError):
            self.instance.start_stop_indices = (
                0,
                6,
                0,
                6,
                0,
                self.instance.data_shape[2] + 1,
            )

    def test_crop_array(self):
        self.instance.start_stop_indices = (0, 6, 0, 6, 1, 6)
        max_indices = center_fft.round_sequence_to_int(
            np.unravel_index(abs(self.data).argmax(), self.data.shape)
        )
        out = self.instance.crop_array(self.data)
        self.assertTrue(
            out.shape
            == (
                self.instance.start_stop_indices[1]
                - self.instance.start_stop_indices[0],
                self.instance.start_stop_indices[3]
                - self.instance.start_stop_indices[2],
                self.instance.start_stop_indices[5]
                - self.instance.start_stop_indices[4],
            )
        )
        out_max_indices = center_fft.round_sequence_to_int(
            np.unravel_index(abs(out).argmax(), out.shape)
        )
        self.assertTrue(
            out_max_indices
            == (
                max_indices[0] - self.instance.start_stop_indices[0],
                max_indices[1] - self.instance.start_stop_indices[2],
                max_indices[2] - self.instance.start_stop_indices[4],
            )
        )

    def test_get_data_shape_wrong_length(self):
        with self.assertRaises(ValueError):
            self.instance.data_shape = (5, 3)

    def test_center_fft_minimum_config(self):
        self.data_shape = (16, 16, 16)
        self.data = create_data(self.data_shape)
        self.instance = center_fft.CenteringFactory(
            data=self.data,
            binning=(1, 1, 1),
            preprocessing_binning=(1, 1, 1),
            roi=(0, self.data_shape[1], 0, self.data_shape[2]),
            bragg_peak=None,
            fft_option="crop_sym_ZYX",
            pad_size=None,
            centering_method="max",
            q_values=None,
            logger=module_logger,
        ).get_centering_instance()
        data, mask, pad_width, q_values, frames_logical = self.instance.center_fft(
            data=self.data, mask=None, frames_logical=None
        )
        self.assertIsNone(mask)
        self.assertIsNone(frames_logical)
        self.assertIsNone(q_values)
        self.assertTrue(all(val == 0 for val in pad_width))
        self.assertEqual(self.instance.center_position, (9, 9, 9))
        self.assertEqual(self.instance.start_stop_indices, (2, 16, 2, 16, 2, 16))
        self.assertEqual(data.shape, (14, 14, 14))

    def test_center_fft_mask_frames_logical_not_none(self):
        self.data_shape = (16, 16, 16)
        self.data = create_data(self.data_shape)
        self.instance = center_fft.CenteringFactory(
            data=self.data,
            binning=(1, 1, 1),
            preprocessing_binning=(1, 1, 1),
            roi=(0, self.data_shape[1], 0, self.data_shape[2]),
            bragg_peak=None,
            fft_option="crop_sym_ZYX",
            pad_size=None,
            centering_method="max",
            q_values=None,
            logger=module_logger,
        ).get_centering_instance()
        _, mask, _, _, frames_logical = self.instance.center_fft(
            data=self.data, mask=self.data, frames_logical=np.ones(self.data.shape[0])
        )
        self.assertEqual(mask.shape, (14, 14, 14))
        self.assertTrue(
            all(
                val == 0
                for val in frames_logical[0 : self.instance.start_stop_indices[0]]
            )
        )

    def test_center_fft_q_values_not_none(self):
        self.data_shape = (16, 16, 16)
        self.data = create_data(self.data_shape)
        self.instance = center_fft.CenteringFactory(
            data=self.data,
            binning=(1, 1, 1),
            preprocessing_binning=(1, 1, 1),
            roi=(0, self.data_shape[1], 0, self.data_shape[2]),
            bragg_peak=None,
            fft_option="crop_sym_ZYX",
            pad_size=None,
            centering_method="max",
            q_values=[
                np.arange(self.data.shape[0], dtype=int),
                -np.arange(self.data.shape[1], dtype=int),
                10 + np.arange(self.data.shape[2], dtype=int),
            ],
            logger=module_logger,
        ).get_centering_instance()
        _, _, _, q_values, _ = self.instance.center_fft(
            data=self.data, mask=self.data, frames_logical=np.ones(self.data.shape[0])
        )
        self.assertTrue(q_values[0][0] == 2)
        self.assertTrue(q_values[1][0] == -2)
        self.assertTrue(q_values[2][0] == 12)


if __name__ == "__main__":
    run_tests(TestRoundSequenceToInt)
    run_tests(TestCenteringFactory)
    run_tests(TestCenterFFT)
