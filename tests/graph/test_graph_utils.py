# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import unittest
import numpy as np
import os
import pathlib
import bcdi.graph.graph_utils as gu
from tests.config import run_tests


class TestLinecut(unittest.TestCase):
    """
    Tests on graphs_utils.linecut.

    def linecut(
        array: np.ndarray,
        indices: List[Tuple[int, int]],
        interp_order: int = 3,
    ) -> np.ndarray:
    """

    def test_interp_order_wrong_type(self):
        with self.assertRaises(TypeError):
            gu.linecut(np.ones(3), indices=[(1, 2)], interp_order=1.2)

    def test_interp_order_none(self):
        with self.assertRaises(ValueError):
            gu.linecut(np.ones(3), indices=[(1, 2)], interp_order=None)

    def test_interp_order_null(self):
        with self.assertRaises(ValueError):
            gu.linecut(np.ones(3), indices=[(1, 2)], interp_order=0)

    def test_1d(self):
        array = np.arange(5)
        expected = np.array([1.0, 2.0, 3.0])
        output = gu.linecut(array, indices=[(1, 3)], interp_order=1)
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
        output = gu.linecut(array, indices=[(0, 5), (0, 5)], interp_order=1)
        self.assertTrue(np.allclose(output, expected))

    def test_2d_horizontal(self):
        array = np.asarray([np.arange(6) for _ in range(5)])
        expected = np.arange(1, 5)
        output = gu.linecut(array, indices=[(1, 1), (1, 4)], interp_order=1)
        self.assertTrue(np.allclose(output, expected))

    def test_2d_vertical(self):
        array = np.asarray([np.ones(6) * idx for idx in range(5)])
        expected = np.arange(1, 5)
        output = gu.linecut(array, indices=[(1, 4), (2, 2)], interp_order=1)
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
        output = gu.linecut(array, indices=[(0, 5), (0, 5), (0, 5)], interp_order=1)
        self.assertTrue(np.allclose(output, expected))


class TestSaveToVti(unittest.TestCase):
    """Tests on save_to_vti."""

    def __init__(self, *args, **kwargs):
        super(TestSaveToVti, self).__init__(*args, **kwargs)
        self.saving_dir = os.getcwd() + "/test_output/"
        pathlib.Path(self.saving_dir).mkdir(parents=True, exist_ok=True)

    def setUp(self):
        # executed before each test
        self.amp = np.zeros((5, 5, 5))
        self.amp[1:4, 1:4, 1:4] = 1
        self.phase = np.zeros((5, 5, 5))
        self.phase[:4, :4, :4] = 1

    # def tearDown(self):
    #     executed after each test

    def test_savetovti_amp(self):
        self.assertIsNone(
            gu.save_to_vti(
                filename=self.saving_dir + "test.vti",
                voxel_size=(1, 1, 1),
                tuple_array=(self.amp, self.phase),
                tuple_fieldnames=("amp", "phase"),
            )
        )

    def test_savetovti_no_amp(self):
        self.assertIsNone(
            gu.save_to_vti(
                filename=self.saving_dir + "test.vti",
                voxel_size=(1, 1, 1),
                tuple_array=(self.amp, self.phase),
                tuple_fieldnames=("other", "phase"),
            )
        )

    def test_savetovti_voxelsize_wrong_shape(self):
        self.assertRaises(
            ValueError,
            gu.save_to_vti,
            filename=self.saving_dir + "test.vti",
            voxel_size=(1, 1),
            tuple_array=(self.amp, self.phase),
            tuple_fieldnames=("amp", "phase"),
        )

    def test_savetovti_voxelsize_negative(self):
        self.assertRaises(
            ValueError,
            gu.save_to_vti,
            filename=self.saving_dir + "test.vti",
            voxel_size=(1, 1, -1),
            tuple_array=(self.amp, self.phase),
            tuple_fieldnames=("amp", "phase"),
        )

    def test_savetovti_array_dim(self):
        self.amp = np.ones((2, 2))
        self.assertRaises(
            ValueError,
            gu.save_to_vti,
            filename=self.saving_dir + "test.vti",
            voxel_size=(1, 1, 1),
            tuple_array=(self.amp, self.phase),
            tuple_fieldnames=("amp", "phase"),
        )

    def test_savetovti_array_shape(self):
        self.amp = np.ones((2, 2, 2))
        self.assertRaises(
            ValueError,
            gu.save_to_vti,
            filename=self.saving_dir + "test.vti",
            voxel_size=(1, 1, 1),
            tuple_array=(self.amp, self.phase),
            tuple_fieldnames=("amp", "phase"),
        )

    def test_savetovti_fieldnames_wrongtype(self):
        self.assertRaises(
            TypeError,
            gu.save_to_vti,
            filename=self.saving_dir + "test.vti",
            voxel_size=(1, 1, 1),
            tuple_array=(self.amp, self.phase),
            tuple_fieldnames=("amp", 0),
        )


if __name__ == "__main__":
    run_tests(TestSaveToVti)
