# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import os
import tempfile
import unittest

import matplotlib
import numpy as np

import bcdi.graph.linecut as lc
from bcdi.postprocessing.postprocessing_utils import tukey_window
from tests.config import run_tests

# matplotlib.use("Agg")


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
        window = tukey_window(shape=(20, 20, 20), alpha=(0.8, 0.5, 0.25))
        array = np.zeros((30, 30, 30))
        array[5:25, 5:25, 5:25] = window
        self.linecut_generator = lc.LinecutGenerator(
            array=array,
            indices=None,
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

    def test_plot_linecuts(self):
        self.linecut_generator.generate_linecuts()
        with tempfile.TemporaryDirectory() as tmpdir:
            self.linecut_generator.filename = f"{tmpdir}/linecut_amp.png"
            self.linecut_generator.plot_linecuts()
            self.assertTrue(os.path.isfile(f"{tmpdir}/linecut_amp.png"))

    def test_plot_fits(self):
        self.linecut_generator.generate_linecuts()
        with tempfile.TemporaryDirectory() as tmpdir:
            self.linecut_generator.filename = f"{tmpdir}/linecut_modulus.png"
            self.linecut_generator.plot_fits()
            self.assertTrue(os.path.isfile(f"{tmpdir}/linecut_modulus_fits.png"))

    def test_fit_boundaries(self):
        expected = [
            [
                {
                    "amp": 0.14704430853118566,
                    "sig": 2.790988805946599,
                    "cen": 6.999999999558211,
                },
                {"amp": 0.19575654101589068, "sig": 2.080391353842167, "cen": 21.0},
            ],
            [
                {
                    "amp": 0.20048506942012878,
                    "sig": 2.0599789064845133,
                    "cen": 5.999999999999737,
                },
                {"amp": 0.3051742945613928, "sig": 1.3305499607185045, "cen": 22.0},
            ],
            [
                {
                    "amp": 0.4763863222906689,
                    "sig": 0.8577322453010738,
                    "cen": 5.999999999999927,
                },
                {
                    "amp": 0.4877979921287945,
                    "sig": 0.83749284517319,
                    "cen": 22.817276027361604,
                },
            ],
        ]
        self.linecut_generator.generate_linecuts()
        result = self.linecut_generator.result
        for idx, val in enumerate(expected):
            for idy in range(2):
                self.assertDictEqual(
                    result[f"dimension_{idx}"][f"param_{idy}"], val[idy]
                )


if __name__ == "__main__":
    run_tests(TestLinecut)
    run_tests(TestLinecutGenerator)
