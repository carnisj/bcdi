# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr
import unittest

import numpy as np

import bcdi.postprocessing.postprocessing_utils as pu
from tests.config import run_tests


def generate_binary_array(array_shape, nonzero_slice):
    """"""
    array = np.zeros(array_shape)
    array[nonzero_slice] = 1
    return array


class TestFindDataRange(unittest.TestCase):
    """Tests on the function postprocessing_utils.find_datarange."""

    def setUp(self) -> None:
        self.plot_margin = 2
        self.amplitude_threshold = 0.25

    def test_keep_size(self):
        array = generate_binary_array((10, 10, 10), nonzero_slice=np.s_[3:9, 1:6, 0:2])
        output = pu.find_datarange(
            array, self.plot_margin, self.amplitude_threshold, keep_size=True
        )
        self.assertTrue(output == list(array.shape))

    def test_no_margin_even_shape_full_array(self):
        array = generate_binary_array((10, 10, 8), nonzero_slice=np.s_[3:10, 1:10, 0:2])
        expected = [10, 10, 8]
        output = pu.find_datarange(
            array, plot_margin=0, amplitude_threshold=self.amplitude_threshold
        )
        self.assertTrue(output == expected)

    def test_no_margin_even_shape(self):
        array = generate_binary_array((10, 10, 10), nonzero_slice=np.s_[3:9, 1:6, 0:2])
        expected = [8, 8, 10]
        output = pu.find_datarange(
            array, plot_margin=0, amplitude_threshold=self.amplitude_threshold
        )
        self.assertTrue(output == expected)

    def test_no_margin_odd_shape_full_array(self):
        array = generate_binary_array((10, 11, 9), nonzero_slice=np.s_[3:10, 1:11, 0:2])
        expected = [10, 11, 9]
        output = pu.find_datarange(
            array, plot_margin=0, amplitude_threshold=self.amplitude_threshold
        )
        self.assertTrue(output == expected)

    def test_no_margin_odd_shape(self):
        array = generate_binary_array((10, 11, 1), nonzero_slice=np.s_[3:5, 6:8, :])
        expected = [4, 5, 1]
        output = pu.find_datarange(
            array, plot_margin=0, amplitude_threshold=self.amplitude_threshold
        )
        self.assertTrue(output == expected)

    def test_below_threshold(self):
        array = generate_binary_array((3, 3, 3), nonzero_slice=np.s_[:, :, :])
        with self.assertRaises(pu.ModulusBelowThreshold):
            pu.find_datarange(array, plot_margin=0, amplitude_threshold=2)

    def test_margin_odd_shape(self):
        array = generate_binary_array((10, 11, 1), nonzero_slice=np.s_[3:5, 6:8, :])
        expected = [8, 9, 5]
        output = pu.find_datarange(
            array,
            plot_margin=self.plot_margin,
            amplitude_threshold=self.amplitude_threshold,
        )
        self.assertTrue(output == expected)

    def test_2d_array(self):
        array = generate_binary_array((10, 11), nonzero_slice=np.s_[3:5, 6:8:])
        expected = [8, 9]
        output = pu.find_datarange(
            array,
            plot_margin=self.plot_margin,
            amplitude_threshold=self.amplitude_threshold,
        )
        self.assertTrue(output == expected)

    def test_1d_array(self):
        array = generate_binary_array((11,), nonzero_slice=np.s_[6:8:])
        expected = [9]
        output = pu.find_datarange(
            array,
            plot_margin=self.plot_margin,
            amplitude_threshold=self.amplitude_threshold,
        )
        self.assertTrue(output == expected)

    def test_empty_array(self):
        with self.assertRaises(ValueError):
            pu.find_datarange(
                np.empty(0),
                plot_margin=self.plot_margin,
                amplitude_threshold=self.amplitude_threshold,
            )


if __name__ == "__main__":
    run_tests(TestFindDataRange)
