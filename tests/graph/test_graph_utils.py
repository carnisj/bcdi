# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import os
import pathlib
import unittest

import numpy as np

import bcdi.graph.graph_utils as gu
from tests.config import run_tests


class TestSaveToVti(unittest.TestCase):
    """Tests on save_to_vti."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
