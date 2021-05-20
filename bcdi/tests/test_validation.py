# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import unittest
from numbers import Real
import numpy as np
from bcdi.context import valid


def run_tests(test_class):
    suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


class TestValidContainer(unittest.TestCase):
    ############################
    # tests on valid_container #
    ############################
    # valid_container(obj, container_types, length=None, min_length=None, item_types=None, min_included=None,
    #                 min_excluded=None, max_included=None, max_excluded=None, allow_none=False, name=None)
    # def setUp(self):
    #     executed before each test
    #
    # def tearDown(self):
    #     executed after each test

    def test_validcontainer_container_type(self):
        self.assertRaises(TypeError, valid.valid_container, obj=list(), container_types=1)

    def test_validcontainer_container_list(self):
        self.assertTrue(valid.valid_container(list(), container_types=list))

    def test_validcontainer_container_dict(self):
        self.assertTrue(valid.valid_container(dict(), container_types=dict))

    def test_validcontainer_container_tuple(self):
        self.assertTrue(valid.valid_container(tuple(), container_types=tuple))

    def test_validcontainer_container_set(self):
        self.assertTrue(valid.valid_container(set(), container_types=set))

    def test_validcontainer_container_string(self):
        self.assertTrue(valid.valid_container('test', container_types=str))

    def test_validcontainer_container_types_none(self):
        self.assertRaises(ValueError, valid.valid_container, obj=list(), container_types=None)

    def test_validcontainer_container_none_not_allowed(self):
        self.assertRaises(ValueError, valid.valid_container, obj=None, container_types={list, tuple}, allow_none=False)

    def test_validcontainer_container_none_allowed(self):
        self.assertTrue(valid.valid_container(obj=None, container_types={list, tuple}, allow_none=True))

    def test_validcontainer_container_wrong_type(self):
        self.assertRaises(TypeError, valid.valid_container, obj=list(), container_types={dict, tuple})

    def test_validcontainer_container_wrong_type_real(self):
        self.assertRaises(TypeError, valid.valid_container, obj=list(), container_types=Real)

    def test_validcontainer_container_length_float(self):
        self.assertRaises(TypeError, valid.valid_container, obj=list(), container_types=list, length=2.3)

    def test_validcontainer_container_length_negative(self):
        self.assertRaises(ValueError, valid.valid_container, obj=list(), container_types=list, length=-2)

    def test_validcontainer_container_length_null(self):
        self.assertTrue(valid.valid_container(obj=list(), container_types=list, length=0))

    def test_validcontainer_container_string_length(self):
        self.assertTrue(valid.valid_container('test', container_types=str, length=4))

    def test_validcontainer_container_string_wrong_length(self):
        self.assertRaises(ValueError, valid.valid_container, obj='test', container_types=str, length=2)

    def test_validcontainer_container_dict_wrong_length(self):
        self.assertRaises(ValueError, valid.valid_container, obj={0: {'x': 1}}, container_types=dict, min_length=2)

    def test_validcontainer_container_minlength_float(self):
        self.assertRaises(TypeError, valid.valid_container, obj=list(), container_types=list, min_length=2.3)

    def test_validcontainer_container_maxlength_float(self):
        self.assertRaises(TypeError, valid.valid_container, obj=list(), container_types=list, max_length=2.3)

    def test_validcontainer_container_minlength_negative(self):
        self.assertRaises(ValueError, valid.valid_container, obj=list(), container_types=list, min_length=-2)

    def test_validcontainer_container_maxlength_negative(self):
        self.assertRaises(ValueError, valid.valid_container, obj=list(), container_types=list, max_length=-2)

    def test_validcontainer_container_maxlength_wrong_length(self):
        self.assertRaises(ValueError, valid.valid_container, obj=[4, 3, 2], container_types=list, max_length=2)

    def test_validcontainer_container_maxlength_smaller_than_minlength(self):
        self.assertRaises(ValueError, valid.valid_container, obj=list(), container_types=list, min_length=2,
                          max_length=1)

    def test_validcontainer_container_maxlength_larger_than_minlength(self):
        self.assertTrue(valid.valid_container(obj=[1, 2], container_types=list, min_length=1, max_length=2))

    def test_validcontainer_container_maxlength_equal_minlength(self):
        self.assertTrue(valid.valid_container(obj=[1], container_types=list, min_length=1, max_length=1))

    def test_validcontainer_container_maxlength_equal_minlength_zero(self):
        self.assertTrue(valid.valid_container(obj=list(), container_types=list, min_length=0, max_length=0))

    def test_validcontainer_container_itemtype(self):
        self.assertTrue(valid.valid_container(obj=list(), container_types=list, item_types=int))

    def test_validcontainer_container_dict_itemtype_int(self):
        self.assertTrue(valid.valid_container(obj={0: {'x': 1}}, container_types=dict, item_types=int))

    def test_validcontainer_container_itemtype_none(self):
        self.assertTrue(valid.valid_container(obj=list(), container_types=list, item_types=None))

    def test_validcontainer_container_itemtype_int(self):
        self.assertRaises(TypeError, valid.valid_container, obj=list(), container_types=list, item_types=2)

    def test_validcontainer_container_itemtype_ndarray(self):
        self.assertTrue(valid.valid_container(obj=(np.ones(3), np.zeros(4)), container_types=(tuple, list),
                                              item_types=np.ndarray))

    def test_validcontainer_container_itemtype_collection(self):
        self.assertTrue(valid.valid_container(obj=list(), container_types=list, item_types=(int, float)))

    def test_validcontainer_container_min_included(self):
        self.assertTrue(valid.valid_container(obj=list(), container_types=list, min_included=0))

    def test_validcontainer_container_min_included_complex(self):
        self.assertRaises(TypeError, valid.valid_container, obj=list(), container_types=list, min_included=1+1j)

    def test_validcontainer_container_min_excluded(self):
        self.assertTrue(valid.valid_container(obj=list(), container_types=list, min_excluded=0))

    def test_validcontainer_container_min_excluded_complex(self):
        self.assertRaises(TypeError, valid.valid_container, obj=list(), container_types=list, min_excluded=1+1j)

    def test_validcontainer_container_max_included(self):
        self.assertTrue(valid.valid_container(obj=list(), container_types=list, max_included=0))

    def test_validcontainer_container_max_included_complex(self):
        self.assertRaises(TypeError, valid.valid_container, obj=list(), container_types=list, max_included=1+1j)

    def test_validcontainer_container_max_excluded(self):
        self.assertTrue(valid.valid_container(obj=list(), container_types=list, max_excluded=0))

    def test_validcontainer_container_max_excluded_complex(self):
        self.assertRaises(TypeError, valid.valid_container, obj=list(), container_types=list, max_excluded=1+1j)

    def test_validcontainer_container_allownone(self):
        self.assertTrue(valid.valid_container(obj=list(), container_types=list, allow_none=True))

    def test_validcontainer_container_allownone_int(self):
        self.assertRaises(TypeError, valid.valid_container, obj=list(), container_types=list, allow_none=0)

    def test_validcontainer_container_allownone_none(self):
        self.assertRaises(TypeError, valid.valid_container, obj=list(), container_types=list, allow_none=None)

    def test_validcontainer_length(self):
        self.assertTrue(valid.valid_container(obj=[1, 2], container_types=list, length=2))

    def test_validcontainer_wrong_length(self):
        self.assertRaises(ValueError, valid.valid_container, obj=[1, 2], container_types=list, length=3)

    def test_validcontainer_minlength(self):
        self.assertTrue(valid.valid_container(obj=[1, 2], container_types=list, min_length=2))

    def test_validcontainer_wrong_minlength(self):
        self.assertRaises(ValueError, valid.valid_container, obj=[1, 2], container_types=list, min_length=3)

    def test_validcontainer_allownone(self):
        self.assertTrue(valid.valid_container(obj=[1, None], container_types=list, allow_none=True))

    def test_validcontainer_notallownone(self):
        self.assertRaises(ValueError, valid.valid_container, obj=[1, None], container_types=list, allow_none=False)


class TestValidKwargs(unittest.TestCase):
    #########################
    # tests on valid_kwargs #
    #########################
    # valid_kwargs(kwargs, allowed_kwargs, name=None)

    def test_validkwargs_kwargs_type_dict(self):
        self.assertTrue(valid.valid_kwargs(kwargs=dict(), allowed_kwargs='test'))

    def test_validkwargs_wrong_kwargs_none(self):
        self.assertRaises(TypeError, valid.valid_kwargs, kwargs=None, allowed_kwargs='test')

    def test_validkwargs_wrong_kwargs_type(self):
        self.assertRaises(TypeError, valid.valid_kwargs, kwargs=[1, 2, 3], allowed_kwargs='test')

    def test_validkwargs_allowed_kwargs_empty(self):
        self.assertRaises(ValueError, valid.valid_kwargs, kwargs=dict(), allowed_kwargs=list())

    def test_validkwargs_allowed_kwargs_dict(self):
        self.assertRaises(TypeError, valid.valid_kwargs, kwargs=dict(), allowed_kwargs={'test': 1})

    def test_validkwargs_allowed_kwargs_length(self):
        self.assertRaises(ValueError, valid.valid_kwargs, kwargs=dict(), allowed_kwargs='')

    def test_validkwargs_allowed_kwargs_items_length(self):
        self.assertRaises(ValueError, valid.valid_kwargs, kwargs=dict(), allowed_kwargs=('', 'test', 'moon'))

    def test_validkwargs_allowed_kwargs_items_type(self):
        self.assertRaises(TypeError, valid.valid_kwargs, kwargs=dict(), allowed_kwargs=(1, 'test', 'moon'))

    def test_validkwargs_allowed_kwargs_items_none(self):
        self.assertRaises(TypeError, valid.valid_kwargs, kwargs=dict(), allowed_kwargs=('test', None))

    def test_validkwargs_valid_kwargs(self):
        self.assertTrue(valid.valid_kwargs(kwargs={'test': 0, 'red': None}, allowed_kwargs={'test', 'red'}))

    def test_validkwargs_invalid_kwargs(self):
        self.assertRaises(KeyError, valid.valid_kwargs, kwargs={'test': 0, 'red': None},
                          allowed_kwargs={'test', 'blue'})


class TestValidItem(unittest.TestCase):
    #######################
    # tests on valid_item #
    #######################
    # valid_item(value, allowed_types, min_included=None, min_excluded=None, max_included=None, max_excluded=None,
    #            allow_none=False, name=None)

    def test_validitem_allowedtypes_none(self):
        self.assertRaises(ValueError, valid.valid_item, value=0, allowed_types=None)

    def test_validitem_allowedtypes_not_type(self):
        self.assertRaises(TypeError, valid.valid_item, value=0, allowed_types=(list, 0))

    def test_validitem_allowedtypes_bool(self):
        self.assertTrue(valid.valid_item(value=True, allowed_types=bool))

    def test_validitem_allowedtypes_ndarray(self):
        self.assertTrue(valid.valid_item(value=np.zeros(4), allowed_types=np.ndarray))

    def test_validitem_allowedtypes_bool_wrongtype(self):
        self.assertRaises(TypeError, valid.valid_item, value=0, allowed_types=bool)

    def test_validitem_min_included(self):
        self.assertTrue(valid.valid_item(value=0, allowed_types=Real, min_included=0))

    def test_validitem_min_included_complex(self):
        self.assertRaises(TypeError, valid.valid_item, value=0, allowed_types=Real, min_included=1+1j)

    def test_validitem_min_included_string(self):
        self.assertRaises(TypeError, valid.valid_item, value='c', allowed_types=str, min_included='a')

    def test_validitem_min_excluded(self):
        self.assertTrue(valid.valid_item(value=1, allowed_types=Real, min_excluded=0))

    def test_validitem_min_excluded_complex(self):
        self.assertRaises(TypeError, valid.valid_item, value=0, allowed_types=Real, min_excluded=1+1j)

    def test_validitem_max_included(self):
        self.assertTrue(valid.valid_item(value=0, allowed_types=Real, max_included=0))

    def test_validitem_max_included_complex(self):
        self.assertRaises(TypeError, valid.valid_container, value=0, allowed_types=Real, max_included=1+1j)

    def test_validitem_max_excluded(self):
        self.assertTrue(valid.valid_item(value=-1, allowed_types=Real, max_excluded=0))

    def test_validitem_max_excluded_complex(self):
        self.assertRaises(TypeError, valid.valid_item, value=0, allowed_types=Real, max_excluded=1+1j)

    def test_validitem_allownone(self):
        self.assertTrue(valid.valid_item(value=None, allowed_types=Real, allow_none=True))

    def test_validitem_allownone_int(self):
        self.assertRaises(TypeError, valid.valid_item, value=0, allowed_types=Real, allow_none=0)

    def test_validitem_allownone_none(self):
        self.assertRaises(TypeError, valid.valid_item, value=0, allowed_types=Real, allow_none=None)

    def test_validitem_allownone_false(self):
        self.assertRaises(ValueError, valid.valid_item, value=None, allowed_types=Real, allow_none=False)

    def test_validitem_allownone_true(self):
        self.assertTrue(valid.valid_item(value=None, allowed_types=Real, allow_none=True))

    def test_validitem_min_included_valid(self):
        self.assertTrue(valid.valid_item(value=1, allowed_types=Real, min_included=0))

    def test_validitem_min_included_equal(self):
        self.assertTrue(valid.valid_item(value=0, allowed_types=Real, min_included=0))

    def test_validitem_min_included_invalid(self):
        self.assertRaises(ValueError, valid.valid_item, value=-1, allowed_types=Real, min_included=0)

    def test_validitem_min_excluded_valid(self):
        self.assertTrue(valid.valid_item(value=1, allowed_types=Real, min_excluded=0))

    def test_validitem_min_excluded_equal(self):
        self.assertRaises(ValueError, valid.valid_item, value=0, allowed_types=Real, min_excluded=0)

    def test_validitem_min_excluded_invalid(self):
        self.assertRaises(ValueError, valid.valid_item, value=-1, allowed_types=Real, min_excluded=0)

    def test_validitem_max_included_valid(self):
        self.assertTrue(valid.valid_item(value=1, allowed_types=Real, max_included=2))

    def test_validitem_max_included_equal(self):
        self.assertTrue(valid.valid_item(value=0, allowed_types=Real, max_included=0))

    def test_validitem_max_included_invalid(self):
        self.assertRaises(ValueError, valid.valid_item, value=1, allowed_types=Real, max_included=0)

    def test_validitem_max_excluded_valid(self):
        self.assertTrue(valid.valid_item(value=1, allowed_types=Real, max_excluded=2))

    def test_validitem_max_excluded_equal(self):
        self.assertRaises(ValueError, valid.valid_item, value=0, allowed_types=Real, max_excluded=0)

    def test_validitem_max_excluded_invalid(self):
        self.assertRaises(ValueError, valid.valid_item, value=1, allowed_types=Real, max_excluded=0)


if __name__ == 'main':
    run_tests(TestValidContainer)
    run_tests(TestValidKwargs)
    run_tests(TestValidItem)
