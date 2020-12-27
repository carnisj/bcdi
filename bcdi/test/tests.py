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
    # def setUp(self):
    #     print('Running setup...')
    #
    # def tearDown(self):
    #     print('Running tear down...')

    ############################
    # tests on valid.container #
    ############################
    def test_container_type(self):
        self.assertRaises(TypeError, valid.valid_container, obj=list(), container_types=1)

    def test_container_list(self):
        self.assertTrue(valid.valid_container(list(), container_types=list))

    def test_container_tuple(self):
        self.assertTrue(valid.valid_container(tuple(), container_types=tuple))

    def test_container_set(self):
        self.assertTrue(valid.valid_container(set(), container_types=set))

    def test_container_none(self):
        self.assertRaises(ValueError, valid.valid_container, obj=list(), container_types=None)

    def test_container_dict(self):
        self.assertRaises(TypeError, valid.valid_container, obj=list(), container_types=dict)

    def test_container_types(self):
        self.assertTrue(valid.valid_container(obj=list(), container_types=list, item_types=(list, tuple)))

    def test_container_length_1(self):
        self.assertRaises(TypeError, valid.valid_container, obj=list(), container_types=list, length=2.3)

    def test_container_length_2(self):
        self.assertRaises(ValueError, valid.valid_container, obj=list(), container_types=list, length=-2)

    def test_container_length_3(self):
        self.assertRaises(ValueError, valid.valid_container, obj=list(), container_types=list, length=0)

    def test_container_minlength_1(self):
        self.assertRaises(TypeError, valid.valid_container, obj=list(), container_types=list, min_length=2.3)

    def test_container_minlength_2(self):
        self.assertRaises(ValueError, valid.valid_container, obj=list(), container_types=list, length=-2)

    def test_container_itemtype_1(self):
        self.assertRaises(TypeError, valid.valid_container, obj=list(), container_types=list, item_types=2)

    def test_container_itemtype_2(self):
        self.assertTrue(valid.valid_container(obj=list(), container_types=list, item_types=int))

    def test_container_itemtype_3(self):
        self.assertTrue(valid.valid_container(obj=list(), container_types=list, item_types=(int, float)))

    def test_container_min_included_1(self):
        self.assertTrue(valid.valid_container(obj=list(), container_types=list, min_included=0))

    def test_container_min_included_2(self):
        self.assertRaises(TypeError, valid.valid_container, obj=list(), container_types=list, min_included=1+1j)

    def test_container_min_excluded_1(self):
        self.assertTrue(valid.valid_container(obj=list(), container_types=list, min_excluded=0))

    def test_container_min_excluded_2(self):
        self.assertRaises(TypeError, valid.valid_container, obj=list(), container_types=list, min_excluded=1+1j)

    def test_container_max_included_1(self):
        self.assertTrue(valid.valid_container(obj=list(), container_types=list, max_included=0))

    def test_container_max_included_2(self):
        self.assertRaises(TypeError, valid.valid_container, obj=list(), container_types=list, max_included=1+1j)

    def test_container_max_excluded_1(self):
        self.assertTrue(valid.valid_container(obj=list(), container_types=list, max_excluded=0))

    def test_container_max_excluded_2(self):
        self.assertRaises(TypeError, valid.valid_container, obj=list(), container_types=list, max_excluded=1+1j)

    def test_container_allownone_1(self):
        self.assertTrue(valid.valid_container(obj=list(), container_types=list, allow_none=True))

    def test_container_allownone_2(self):
        self.assertRaises(TypeError, valid.valid_container, obj=list(), container_types=list, allow_none=0)


if __name__ == 'main':
    result = run_tests(TestValidation)
    print(result)
