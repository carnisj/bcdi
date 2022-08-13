# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import tempfile
import unittest

import numpy as np

import bcdi.graph.linecut as lc
import bcdi.utils.utilities as util
from tests.config import run_tests


class TestLinecut(unittest.TestCase):
    """
    Tests on linecut.linecut.

    def linecut(
        array: np.ndarray,
        indices: List[Tuple[int, int]],
        interp_order: int = 3,
    ) -> np.ndarray:
    """

    def test_interp_order_wrong_type(self):
        with self.assertRaises(TypeError):
            lc.linecut(np.ones(3), indices=[(1, 2)], interp_order=1.2)

    def test_interp_order_none(self):
        with self.assertRaises(ValueError):
            lc.linecut(np.ones(3), indices=[(1, 2)], interp_order=None)

    def test_interp_order_null(self):
        with self.assertRaises(ValueError):
            lc.linecut(np.ones(3), indices=[(1, 2)], interp_order=0)

    def test_1d(self):
        array = np.arange(5)
        expected = np.array([1.0, 2.0, 3.0])
        output = lc.linecut(array, indices=[(1, 3)], interp_order=1)
        self.assertTrue(np.allclose(output, expected))

    def test_2d_diagonal(self):
        array = np.asarray([np.arange(6) for _ in range(6)])
        expected = np.array(
            [
                0,
                0.71428571,
                1.42857143,
                2.14285714,
                2.85714286,
                3.57142857,
                4.28571429,
                5.0,
            ]
        )
        output = lc.linecut(array, indices=[(0, 5), (0, 5)], interp_order=1)
        self.assertTrue(np.allclose(output, expected))

    def test_2d_horizontal(self):
        array = np.asarray([np.arange(6) for _ in range(5)])
        expected = np.arange(1, 5)
        output = lc.linecut(array, indices=[(1, 1), (1, 4)], interp_order=1)
        self.assertTrue(np.allclose(output, expected))

    def test_2d_vertical(self):
        array = np.asarray([np.ones(6) * idx for idx in range(5)])
        expected = np.arange(1, 5)
        output = lc.linecut(array, indices=[(1, 4), (2, 2)], interp_order=1)
        self.assertTrue(np.allclose(output, expected))

    def test_3d_diagonal(self):
        array = np.asarray([[np.arange(6) for _ in range(6)] for _ in range(6)])
        expected = np.array(
            [
                0,
                0.55555556,
                1.11111111,
                1.66666667,
                2.22222222,
                2.77777778,
                3.33333333,
                3.88888889,
                4.44444444,
                5.0,
            ]
        )
        output = lc.linecut(array, indices=[(0, 5), (0, 5), (0, 5)], interp_order=1)
        self.assertTrue(np.allclose(output, expected))


class TestLinecutGenerator(unittest.TestCase):
    """Tests on graphs_utils.LinecutGenerator."""

    def setUp(self) -> None:
        array = util.gaussian_window(window_shape=(20, 20, 20))
        self.linecut_generator = lc.LinecutGenerator(
            array=array,
            indices=None,
            filename=f"{tempfile.TemporaryDirectory()}linecut_amp.png",
            fit_derivative=True,
            voxel_sizes=None,
            support_threshold=0.1,
            label="modulus",
        )

    def test_instantiation(self):
        expected_indices = (
            ((0, 19), (10, 10), (10, 10)),
            ((10, 10), (0, 19), (10, 10)),
            ((10, 10), (10, 10), (0, 19)),
        )
        self.assertEqual(self.linecut_generator.support_threshold, 0.1)
        self.assertTrue(self.linecut_generator.fit_derivative)
        self.assertTrue(self.linecut_generator.voxel_sizes is None)
        self.assertEqual(self.linecut_generator.user_label, "modulus")
        self.assertTrue(
            len(self.linecut_generator.indices) == self.linecut_generator.array.ndim
        )
        self.assertTrue(
            all(
                tuple(self.linecut_generator.indices[idx]) == expected_indices[idx]
                for idx in range(self.linecut_generator.array.ndim)
            )
        )
        self.assertEqual(self.linecut_generator.unit, "pixels")


if __name__ == "__main__":
    run_tests(TestLinecut)
    run_tests(TestLinecutGenerator)
