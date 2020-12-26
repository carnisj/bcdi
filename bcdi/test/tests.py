# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import unittest
import bcdi.utils.validation as valid


def run_tests(test_class):
    suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


class TestValidation(unittest.TestCase):
    def setUp(self):
        print('Running setup...')

    def tearDown(self):
        print('Running tear down...')

    def test_list(self):
        self.assertTrue(valid.valid_container(list(), container_types=list))

    def test_tuple(self):
        self.assertTrue(valid.valid_container(tuple(), container_types=tuple))

    def test_set(self):
        self.assertTrue(valid.valid_container(set(), container_types=set))

    # self.assertNotEqual(tz, test_tz)


if __name__ == 'main':
    result = run_tests(TestValidation)
    print(result)