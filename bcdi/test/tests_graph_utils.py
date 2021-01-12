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
        pathlib.Path(os.getcwd() + '/test_output/').mkdir(parents=True, exist_ok=True)

    def setUp(self):
        # executed before each test
        amp = np.zeros((5, 5, 5))
        amp[1:4, 1:4, 1:4] = 1
        phase = np.zeros((5, 5, 5))
        phase[:4, :4, :4] = 1

    # def tearDown(self):
    #     executed after each test

    ########################
    # tests on save_to_vti #
    ########################
    def test_savetovti_voxelsize(self):
        pass
        # self.assertTrue(gu.save_to_vti(filename='', allowed_types=Real, max_included=0))


if __name__ == 'main':
    result = run_tests(TestGraphUtils)
    print(result)
