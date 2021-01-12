# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import unittest
from numbers import Real
import numpy as np
import os
import pathlib
import bcdi.graph.graph_utils as gu


def run_tests(test_class):
    suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


class TestGraphUtils(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestGraphUtils, self).__init__(*args, **kwargs)
        self.saving_dir = os.getcwd() + '/test_output/'
        pathlib.Path(self.saving_dir).mkdir(parents=True, exist_ok=True)

    def setUp(self):
        # executed before each test
        self.amp = np.zeros((5, 5, 5))
        self.amp[1:4, 1:4, 1:4] = 1
        self.phase = np.zeros((5, 5, 5))
        self.phase[:4, :4, :4] = 1

    # def tearDown(self):
    #     executed after each test

    ########################
    # tests on save_to_vti #
    ########################
    def test_savetovti_voxelsize(self):
        self.assertIsNone(gu.save_to_vti(filename=self.saving_dir + 'test.vti', voxel_size=(1, 1, 1),
                                         tuple_array=(self.amp, self.phase), tuple_fieldnames=('amp', 'phase')))


if __name__ == 'main':
    result = run_tests(TestGraphUtils)
    print(result)
