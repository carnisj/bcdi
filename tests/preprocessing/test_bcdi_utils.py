# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import pathlib
import tempfile
import unittest

import matplotlib
import numpy as np

from bcdi.preprocessing.bcdi_utils import PeakFinder, find_bragg
from bcdi.utils.utilities import gaussian_window
from tests.config import run_tests

matplotlib.use("Agg")


class TestPeakFinder(unittest.TestCase):
    def setUp(self) -> None:
        data = np.zeros((4, 32, 32))
        data[:-1, -13:, 17:30] = gaussian_window(window_shape=(3, 13, 13))
        self.peakfinder = PeakFinder(
            array=data,
            region_of_interest=None,
            binning=None,
            peak_method="max_com",
        )

    def test_max(self):
        peaks = self.peakfinder.find_peak()
        self.assertTrue(peaks["max"] == (1, 25, 23))

    def test_com(self):
        peaks = self.peakfinder.find_peak()
        self.assertTrue(peaks["com"] == (1, 25, 23))

    def test_no_user_defined_peak(self):
        user_defined_peak = None
        peaks = self.peakfinder.find_peak(user_defined_peak=user_defined_peak)
        self.assertTrue("user" not in peaks)

    def test_user_defined_peak(self):
        user_defined_peak = (1, 22, 21)
        peaks = self.peakfinder.find_peak(user_defined_peak=user_defined_peak)
        self.assertTrue(peaks["user"] == user_defined_peak)

    def test_maxcom(self):
        peaks = self.peakfinder.find_peak()
        self.assertTrue(peaks["max_com"] == (1, 25, 23))

    def test_unbin_no_binning(self):
        position = [5, 6, 7]
        output = self.peakfinder._unbin(position)
        self.assertTrue(all(val[0] == val[1] for val in zip(position, output)))

    def test_unbin_binning(self):
        position = [5, 6, 7]
        self.peakfinder.binning = [1, 2, 2]
        expected = [5, 12, 14]
        output = self.peakfinder._unbin(position)
        self.assertTrue(all(val[0] == val[1] for val in zip(expected, output)))

    def test_bin(self):
        position = [5, 6, 7]
        self.peakfinder.binning = [1, 2, 2]
        expected = [5, 3, 3]
        output = self.peakfinder._bin(position)
        self.assertTrue(all(val[0] == val[1] for val in zip(expected, output)))

    def test_offset_full_detector_no_region_of_interest(self):
        position = [5, 6, 7]
        output = self.peakfinder._offset(position, frame="full_detector")
        self.assertTrue(all(val[0] == val[1] for val in zip(position, output)))

    def test_offset_full_detector_region_of_interest(self):
        position = [5, 6, 7]
        self.peakfinder.region_of_interest = [1, 32, 2, 30]
        expected = [5, 7, 9]
        output = self.peakfinder._offset(position, frame="full_detector")
        self.assertTrue(all(val[0] == val[1] for val in zip(expected, output)))

    def test_offset_in_region_of_interest(self):
        position = [5, 6, 7]
        self.peakfinder.region_of_interest = [1, 32, 2, 30]
        expected = [5, 5, 5]
        output = self.peakfinder._offset(position, frame="region_of_interest")
        self.assertTrue(all(val[0] == val[1] for val in zip(expected, output)))

    def test_get_indices_full_detector(self):
        position = [5, 6, 7]
        self.peakfinder.binning = [1, 2, 2]
        self.peakfinder.region_of_interest = [1, 32, 2, 30]
        expected = [5, 13, 16]
        output = self.peakfinder.get_indices_full_detector(position)
        self.assertTrue(all(val[0] == val[1] for val in zip(expected, output)))

    def test_get_indices_cropped_binned_detector(self):
        position = [5, 13, 17]
        self.peakfinder.region_of_interest = [3, 32, 2, 30]
        self.peakfinder.binning = [1, 2, 2]
        expected = [5, 5, 7]
        output = self.peakfinder.get_indices_cropped_binned_detector(position)
        self.assertTrue(all(val[0] == val[1] for val in zip(expected, output)))

    def test_binning_wrong_length(self):
        with self.assertRaises(ValueError):
            self.peakfinder.binning = (2, 2)

    def test_binning_wrong_type(self):
        with self.assertRaises(TypeError):
            self.peakfinder.binning = (2, 2, 2.3)

    def test_binning(self):
        self.peakfinder.binning = [2, 1, 1]
        peaks = self.peakfinder.find_peak()
        self.assertTrue(peaks["max"] == (2, 25, 23))

    def test_binning_2(self):
        self.peakfinder.binning = [3, 3, 4]
        peaks = self.peakfinder.find_peak()
        self.assertTrue(peaks["max"] == (3, 75, 92))

    def test_roi_wrong_length(self):
        with self.assertRaises(ValueError):
            self.peakfinder.region_of_interest = [2, 1, 1]

    def test_roi_wrong_type(self):
        with self.assertRaises(TypeError):
            self.peakfinder.region_of_interest = [2, 1, 1, 1.2]

    def test_roi(self):
        self.peakfinder.region_of_interest = [3, 3, 4, 2]
        peaks = self.peakfinder.find_peak()
        self.assertTrue(peaks["max"] == (1, 28, 27))

    def test_bin_roi(self):
        self.peakfinder.region_of_interest = [3, 3, 4, 2]
        self.peakfinder.binning = [3, 3, 4]
        peaks = self.peakfinder.find_peak()
        self.assertTrue(peaks["max"] == (3, 78, 96))

    def test_unknown_peak_method(self):
        with self.assertRaises(ValueError):
            self.peakfinder.peak_method = "fancy_method"

    def test_plot_peaks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.peakfinder.plot_peaks(savedir=tmpdir)
            path = pathlib.Path(tmpdir) / "centering_method.png"
            self.assertTrue(path.is_file())

    def test_get_rocking_curve(self):
        self.peakfinder._get_rocking_curve()
        output = self.peakfinder.metadata["rocking_curve"]
        expected = np.array([47548.6935939, 12299450.55212367, 47548.6935939, 0.0])
        self.assertTrue(output.shape[0] == self.peakfinder.array.shape[0])
        self.assertTrue(output.ndim == 1)
        self.assertTrue(np.allclose(output, expected))

    def test_fit_rocking_curve_tilt_values_undefined(self):
        self.peakfinder.fit_rocking_curve()
        self.assertIsNone(self.peakfinder.metadata["tilt_values"])
        self.assertIsNone(self.peakfinder.metadata["tilt_value_at_peak"])

    def test_fit_rocking_curve_tilt_values_defined(self):
        tilt_values = np.linspace(5.1, 5.25, 4)
        self.peakfinder.fit_rocking_curve(tilt_values=tilt_values)
        self.assertTrue(
            np.allclose(self.peakfinder.metadata["tilt_values"], tilt_values)
        )
        self.assertAlmostEqual(self.peakfinder.metadata["tilt_value_at_peak"], 5.15)

    def test_fit_rocking_curve_keys(self):
        expected_keys = {
            "bragg_peak",
            "peaks",
            "rocking_curve",
            "detector_data_at_peak",
            "tilt_values",
            "interp_tilt_values",
            "interp_rocking_curve",
            "interp_fwhm",
            "tilt_value_at_peak",
        }
        self.peakfinder.fit_rocking_curve()
        self.assertTrue(key in self.peakfinder.metadata for key in expected_keys)

    def test_plot_peaks_image_saved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.peakfinder.plot_peaks(savedir=tmpdir)
            path = pathlib.Path(tmpdir) / "centering_method.png"
            self.assertTrue(path.is_file())

    def test_bragg_peak(self):
        expected = (1, 25, 23)
        self.assertTrue(self.peakfinder.bragg_peak == expected)

    def test_roi_center_should_be_identical_to_bragg_peak(self):
        expected = (1, 25, 23)
        self.assertTrue(self.peakfinder._roi_center == expected)

    def test_roi_center_with_binning(self):
        self.peakfinder.binning = [1, 2, 2]
        expected = (1, 12, 11)
        self.assertTrue(self.peakfinder._roi_center == expected)

    def test_roi_center_with_roi_and_binning(self):
        self.peakfinder.binning = [1, 2, 2]
        self.peakfinder.region_of_interest = [1, 32, 2, 32]
        expected = (1, 12, 10)
        self.assertTrue(self.peakfinder._roi_center == expected)

    def test_fit_rocking_curve(self):
        self.peakfinder.fit_rocking_curve(
            tilt_values=np.arange(self.peakfinder.array.shape[0]),
        )
        self.assertEqual(self.peakfinder.metadata["tilt_value_at_peak"], 1)

    def test_plot_rocking_curve_file_saved(self):
        self.peakfinder.fit_rocking_curve(
            tilt_values=0.1 * np.arange(self.peakfinder.array.shape[0]),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            self.peakfinder.plot_rocking_curve(savedir=tmpdir)
            path = pathlib.Path(tmpdir) / "rocking_curve.png"
            self.assertTrue(path.is_file())


class TestFindBragg(unittest.TestCase):
    def setUp(self) -> None:
        self.data = np.zeros((4, 32, 32))
        self.data[:-1, -13:, 17:30] = gaussian_window(window_shape=(3, 13, 13))

    def test_check_return_type(self):
        output = find_bragg(array=self.data)
        self.assertIsInstance(output, dict)

    def test_check_return_keys(self):
        output = find_bragg(array=self.data)
        self.assertTrue("detector_data_at_peak" in output)

    def test_run_only_peak_finding(self):
        expected = (1, 25, 23)
        metadata = find_bragg(array=self.data, plot_fit=False)
        self.assertTrue(metadata["bragg_peak"] == expected)

    def test_with_binning(self):
        expected = (3, 50, 46)
        metadata = find_bragg(array=self.data, binning=[3, 2, 2], plot_fit=False)
        self.assertTrue(metadata["bragg_peak"] == expected)

    def test_with_binning_and_roi(self):
        expected = (3, 62, 55)
        metadata = find_bragg(
            array=self.data, binning=[3, 2, 2], roi=[12, 44, 9, 41], plot_fit=False
        )
        self.assertTrue(metadata["bragg_peak"] == expected)


if __name__ == "__main__":
    run_tests(TestPeakFinder)
    run_tests(TestFindBragg)
