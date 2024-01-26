# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import unittest
from numbers import Integral, Real

import numpy as np

import bcdi.utils.validation as valid
from tests.config import run_tests


class TestIsFloat(unittest.TestCase):
    """
    Tests on the function utilities.is_float.

    def is_float(string)
    """

    def test_string_float(self):
        self.assertTrue(valid.is_float("12.0"))

    def test_string_int(self):
        self.assertTrue(valid.is_float("12"))

    def test_string_complex(self):
        self.assertFalse(valid.is_float("12 + 1j"))

    def test_string_none(self):
        self.assertFalse(valid.is_float("None"))

    def test_string_not_numeric(self):
        self.assertFalse(valid.is_float("abc"))

    def test_none(self):
        with self.assertRaises(TypeError):
            valid.is_float(None)

    def test_array(self):
        with self.assertRaises(TypeError):
            valid.is_float(np.ones(3))


class TestValidContainer(unittest.TestCase):
    """
    Tests on valid_container.

    valid_container(obj, container_types, length=None, min_length=None,
                item_types=None, min_included=None,
                min_excluded=None, max_included=None, max_excluded=None,
                allow_none=False, name=None)
    """

    def test_validcontainer_container_type(self):
        self.assertRaises(TypeError, valid.valid_container, obj=[], container_types=1)

    def test_validcontainer_container_list(self):
        self.assertTrue(valid.valid_container([], container_types=list))

    def test_validcontainer_container_dict(self):
        self.assertTrue(valid.valid_container({}, container_types=dict))

    def test_validcontainer_container_tuple(self):
        self.assertTrue(valid.valid_container((), container_types=tuple))

    def test_validcontainer_container_set(self):
        self.assertTrue(valid.valid_container(set(), container_types=set))

    def test_validcontainer_container_string(self):
        self.assertTrue(valid.valid_container("test", container_types=str))

    def test_validcontainer_container_types_none(self):
        self.assertRaises(
            ValueError, valid.valid_container, obj=[], container_types=None
        )

    def test_validcontainer_container_none_not_allowed(self):
        self.assertRaises(
            ValueError,
            valid.valid_container,
            obj=None,
            container_types={list, tuple},
            allow_none=False,
        )

    def test_validcontainer_container_none_allowed(self):
        self.assertTrue(
            valid.valid_container(
                obj=None, container_types={list, tuple}, allow_none=True
            )
        )

    def test_validcontainer_container_wrong_type(self):
        self.assertRaises(
            TypeError, valid.valid_container, obj=[], container_types={dict, tuple}
        )

    def test_validcontainer_container_wrong_type_real(self):
        self.assertRaises(
            TypeError, valid.valid_container, obj=[], container_types=Real
        )

    def test_validcontainer_container_length_float(self):
        self.assertRaises(
            TypeError, valid.valid_container, obj=[], container_types=list, length=2.3
        )

    def test_validcontainer_container_length_negative(self):
        self.assertRaises(
            ValueError, valid.valid_container, obj=[], container_types=list, length=-2
        )

    def test_validcontainer_container_length_null(self):
        self.assertTrue(valid.valid_container(obj=[], container_types=list, length=0))

    def test_validcontainer_container_string_length(self):
        self.assertTrue(valid.valid_container("test", container_types=str, length=4))

    def test_validcontainer_container_string_wrong_length(self):
        self.assertRaises(
            ValueError, valid.valid_container, obj="test", container_types=str, length=2
        )

    def test_validcontainer_container_dict_wrong_length(self):
        self.assertRaises(
            ValueError,
            valid.valid_container,
            obj={0: {"x": 1}},
            container_types=dict,
            min_length=2,
        )

    def test_validcontainer_container_minlength_float(self):
        self.assertRaises(
            TypeError,
            valid.valid_container,
            obj=[],
            container_types=list,
            min_length=2.3,
        )

    def test_validcontainer_container_maxlength_float(self):
        self.assertRaises(
            TypeError,
            valid.valid_container,
            obj=[],
            container_types=list,
            max_length=2.3,
        )

    def test_validcontainer_container_minlength_negative(self):
        self.assertRaises(
            ValueError,
            valid.valid_container,
            obj=[],
            container_types=list,
            min_length=-2,
        )

    def test_validcontainer_container_maxlength_negative(self):
        self.assertRaises(
            ValueError,
            valid.valid_container,
            obj=[],
            container_types=list,
            max_length=-2,
        )

    def test_validcontainer_container_maxlength_wrong_length(self):
        self.assertRaises(
            ValueError,
            valid.valid_container,
            obj=[4, 3, 2],
            container_types=list,
            max_length=2,
        )

    def test_validcontainer_container_maxlength_smaller_than_minlength(self):
        self.assertRaises(
            ValueError,
            valid.valid_container,
            obj=[],
            container_types=list,
            min_length=2,
            max_length=1,
        )

    def test_validcontainer_container_maxlength_larger_than_minlength(self):
        self.assertTrue(
            valid.valid_container(
                obj=[1, 2], container_types=list, min_length=1, max_length=2
            )
        )

    def test_validcontainer_container_maxlength_equal_minlength(self):
        self.assertTrue(
            valid.valid_container(
                obj=[1], container_types=list, min_length=1, max_length=1
            )
        )

    def test_validcontainer_container_maxlength_equal_minlength_zero(self):
        self.assertTrue(
            valid.valid_container(
                obj=[], container_types=list, min_length=0, max_length=0
            )
        )

    def test_validcontainer_container_itemtype(self):
        self.assertTrue(
            valid.valid_container(obj=[], container_types=list, item_types=int)
        )

    def test_validcontainer_container_dict_itemtype_int(self):
        self.assertTrue(
            valid.valid_container(
                obj={0: {"x": 1}}, container_types=dict, item_types=int
            )
        )

    def test_validcontainer_container_itemtype_none(self):
        self.assertTrue(
            valid.valid_container(obj=[], container_types=list, item_types=None)
        )

    def test_validcontainer_container_itemtype_int(self):
        self.assertRaises(
            TypeError, valid.valid_container, obj=[], container_types=list, item_types=2
        )

    def test_validcontainer_container_itemtype_ndarray(self):
        self.assertTrue(
            valid.valid_container(
                obj=(np.ones(3), np.zeros(4)),
                container_types=(tuple, list),
                item_types=np.ndarray,
            )
        )

    def test_validcontainer_container_itemtype_collection(self):
        self.assertTrue(
            valid.valid_container(obj=[], container_types=list, item_types=(int, float))
        )

    def test_validcontainer_container_min_included(self):
        self.assertTrue(
            valid.valid_container(obj=[], container_types=list, min_included=0)
        )

    def test_validcontainer_container_min_included_complex(self):
        self.assertRaises(
            TypeError,
            valid.valid_container,
            obj=[],
            container_types=list,
            min_included=1 + 1j,
        )

    def test_validcontainer_container_min_excluded(self):
        self.assertTrue(
            valid.valid_container(obj=[], container_types=list, min_excluded=0)
        )

    def test_validcontainer_container_min_excluded_complex(self):
        self.assertRaises(
            TypeError,
            valid.valid_container,
            obj=[],
            container_types=list,
            min_excluded=1 + 1j,
        )

    def test_validcontainer_container_max_included(self):
        self.assertTrue(
            valid.valid_container(obj=[], container_types=list, max_included=0)
        )

    def test_validcontainer_container_max_included_complex(self):
        self.assertRaises(
            TypeError,
            valid.valid_container,
            obj=[],
            container_types=list,
            max_included=1 + 1j,
        )

    def test_validcontainer_container_max_excluded(self):
        self.assertTrue(
            valid.valid_container(obj=[], container_types=list, max_excluded=0)
        )

    def test_validcontainer_container_max_excluded_complex(self):
        self.assertRaises(
            TypeError,
            valid.valid_container,
            obj=[],
            container_types=list,
            max_excluded=1 + 1j,
        )

    def test_validcontainer_container_allownone(self):
        self.assertTrue(
            valid.valid_container(obj=[], container_types=list, allow_none=True)
        )

    def test_validcontainer_container_allownone_int(self):
        self.assertRaises(
            TypeError, valid.valid_container, obj=[], container_types=list, allow_none=0
        )

    def test_validcontainer_container_allownone_none(self):
        self.assertRaises(
            TypeError,
            valid.valid_container,
            obj=[],
            container_types=list,
            allow_none=None,
        )

    def test_validcontainer_length(self):
        self.assertTrue(
            valid.valid_container(obj=[1, 2], container_types=list, length=2)
        )

    def test_validcontainer_wrong_length(self):
        self.assertRaises(
            ValueError,
            valid.valid_container,
            obj=[1, 2],
            container_types=list,
            length=3,
        )

    def test_validcontainer_minlength(self):
        self.assertTrue(
            valid.valid_container(obj=[1, 2], container_types=list, min_length=2)
        )

    def test_validcontainer_wrong_minlength(self):
        self.assertRaises(
            ValueError,
            valid.valid_container,
            obj=[1, 2],
            container_types=list,
            min_length=3,
        )

    def test_validcontainer_allownone(self):
        self.assertTrue(
            valid.valid_container(obj=[1, None], container_types=list, allow_none=True)
        )

    def test_validcontainer_notallownone(self):
        self.assertRaises(
            ValueError,
            valid.valid_container,
            obj=[1, None],
            container_types=list,
            allow_none=False,
        )

    def test_validcontainer_invalid_name(self):
        self.assertRaises(
            TypeError, valid.valid_container, obj=(), container_types=tuple, name=0
        )


class TestValidKwargs(unittest.TestCase):
    """
    Tests on valid_kwargs.

    valid_kwargs(kwargs, allowed_kwargs, name=None)
    """

    def test_validkwargs_kwargs_type_dict(self):
        self.assertTrue(valid.valid_kwargs(kwargs={}, allowed_kwargs="test"))

    def test_validkwargs_wrong_kwargs_none(self):
        self.assertRaises(
            TypeError, valid.valid_kwargs, kwargs=None, allowed_kwargs="test"
        )

    def test_validkwargs_wrong_kwargs_type(self):
        self.assertRaises(
            TypeError, valid.valid_kwargs, kwargs=[1, 2, 3], allowed_kwargs="test"
        )

    def test_validkwargs_allowed_kwargs_empty(self):
        self.assertRaises(ValueError, valid.valid_kwargs, kwargs={}, allowed_kwargs=[])

    def test_validkwargs_allowed_kwargs_dict(self):
        self.assertRaises(
            TypeError, valid.valid_kwargs, kwargs={}, allowed_kwargs={"test": 1}
        )

    def test_validkwargs_allowed_kwargs_length(self):
        self.assertRaises(ValueError, valid.valid_kwargs, kwargs={}, allowed_kwargs="")

    def test_validkwargs_allowed_kwargs_items_length(self):
        self.assertRaises(
            ValueError,
            valid.valid_kwargs,
            kwargs={},
            allowed_kwargs=("", "test", "moon"),
        )

    def test_validkwargs_allowed_kwargs_items_type(self):
        self.assertRaises(
            TypeError, valid.valid_kwargs, kwargs={}, allowed_kwargs=(1, "test", "moon")
        )

    def test_validkwargs_allowed_kwargs_items_none(self):
        self.assertRaises(
            TypeError, valid.valid_kwargs, kwargs={}, allowed_kwargs=("test", None)
        )

    def test_validkwargs_valid_kwargs(self):
        self.assertTrue(
            valid.valid_kwargs(
                kwargs={"test": 0, "red": None}, allowed_kwargs={"test", "red"}
            )
        )

    def test_validkwargs_invalid_kwargs(self):
        self.assertRaises(
            KeyError,
            valid.valid_kwargs,
            kwargs={"test": 0, "red": None},
            allowed_kwargs={"test", "blue"},
        )

    def test_validkwargs_invalid_name(self):
        self.assertRaises(
            TypeError,
            valid.valid_kwargs,
            kwargs={"test": 0, "red": None},
            allowed_kwargs={"test", "red"},
            name=0,
        )


class TestValidItem(unittest.TestCase):
    """
    Tests on valid_item.

    valid_item(value, allowed_types, min_included=None, min_excluded=None,
           max_included=None, max_excluded=None,
           allow_none=False, name=None)
    """

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
        self.assertRaises(
            TypeError,
            valid.valid_item,
            value=0,
            allowed_types=Real,
            min_included=1 + 1j,
        )

    def test_validitem_min_included_string(self):
        self.assertRaises(
            TypeError, valid.valid_item, value="c", allowed_types=str, min_included="a"
        )

    def test_validitem_min_excluded(self):
        self.assertTrue(valid.valid_item(value=1, allowed_types=Real, min_excluded=0))

    def test_validitem_min_excluded_complex(self):
        self.assertRaises(
            TypeError,
            valid.valid_item,
            value=0,
            allowed_types=Real,
            min_excluded=1 + 1j,
        )

    def test_validitem_max_included(self):
        self.assertTrue(valid.valid_item(value=0, allowed_types=Real, max_included=0))

    def test_validitem_max_included_complex(self):
        self.assertRaises(
            TypeError,
            valid.valid_container,
            value=0,
            allowed_types=Real,
            max_included=1 + 1j,
        )

    def test_validitem_max_excluded(self):
        self.assertTrue(valid.valid_item(value=-1, allowed_types=Real, max_excluded=0))

    def test_validitem_max_excluded_complex(self):
        self.assertRaises(
            TypeError,
            valid.valid_item,
            value=0,
            allowed_types=Real,
            max_excluded=1 + 1j,
        )

    def test_validitem_allownone(self):
        self.assertTrue(
            valid.valid_item(value=None, allowed_types=Real, allow_none=True)
        )

    def test_validitem_allownone_int(self):
        self.assertRaises(
            TypeError, valid.valid_item, value=0, allowed_types=Real, allow_none=0
        )

    def test_validitem_allownone_none(self):
        self.assertRaises(
            TypeError, valid.valid_item, value=0, allowed_types=Real, allow_none=None
        )

    def test_validitem_allownone_false(self):
        self.assertRaises(
            ValueError,
            valid.valid_item,
            value=None,
            allowed_types=Real,
            allow_none=False,
        )

    def test_validitem_allownone_true(self):
        self.assertTrue(
            valid.valid_item(value=None, allowed_types=Real, allow_none=True)
        )

    def test_validitem_min_included_valid(self):
        self.assertTrue(valid.valid_item(value=1, allowed_types=Real, min_included=0))

    def test_validitem_min_included_equal(self):
        self.assertTrue(valid.valid_item(value=0, allowed_types=Real, min_included=0))

    def test_validitem_min_included_invalid(self):
        self.assertRaises(
            ValueError, valid.valid_item, value=-1, allowed_types=Real, min_included=0
        )

    def test_validitem_min_excluded_valid(self):
        self.assertTrue(valid.valid_item(value=1, allowed_types=Real, min_excluded=0))

    def test_validitem_min_excluded_equal(self):
        self.assertRaises(
            ValueError, valid.valid_item, value=0, allowed_types=Real, min_excluded=0
        )

    def test_validitem_min_excluded_invalid(self):
        self.assertRaises(
            ValueError, valid.valid_item, value=-1, allowed_types=Real, min_excluded=0
        )

    def test_validitem_max_included_valid(self):
        self.assertTrue(valid.valid_item(value=1, allowed_types=Real, max_included=2))

    def test_validitem_max_included_equal(self):
        self.assertTrue(valid.valid_item(value=0, allowed_types=Real, max_included=0))

    def test_validitem_max_included_invalid(self):
        self.assertRaises(
            ValueError, valid.valid_item, value=1, allowed_types=Real, max_included=0
        )

    def test_validitem_max_excluded_valid(self):
        self.assertTrue(valid.valid_item(value=1, allowed_types=Real, max_excluded=2))

    def test_validitem_max_excluded_equal(self):
        self.assertRaises(
            ValueError, valid.valid_item, value=0, allowed_types=Real, max_excluded=0
        )

    def test_validitem_max_excluded_invalid(self):
        self.assertRaises(
            ValueError, valid.valid_item, value=1, allowed_types=Real, max_excluded=0
        )

    def test_validitem_invalid_name(self):
        self.assertRaises(
            TypeError, valid.valid_item, value=1, allowed_types=Real, name=0
        )


class TestValid1dArray(unittest.TestCase):
    """
    Tests on valid_1darray.

    valid_1d_array(array, length=None, min_length=None, allow_none=True,
    allowed_types=None, allowed_values=None, name=None)
    """

    def setUp(self) -> None:
        self.data = np.ones(7, dtype=int)

    def test_length_wrong_type_float(self):
        self.assertRaises(TypeError, valid.valid_1d_array, self.data, legnth=1.5)

    def test_length_negative(self):
        self.assertRaises(ValueError, valid.valid_1d_array, self.data, length=-1)

    def test_length_null(self):
        self.assertTrue(valid.valid_1d_array(np.empty(0), length=0))

    def test_length_None(self):
        self.assertTrue(valid.valid_1d_array(self.data, length=None))

    def test_minlength_wrong_type_float(self):
        self.assertRaises(TypeError, valid.valid_1d_array, self.data, min_length=1.5)

    def test_minlength_negative(self):
        self.assertRaises(ValueError, valid.valid_1d_array, self.data, min_length=-1)

    def test_minlength_null(self):
        self.assertTrue(valid.valid_1d_array(self.data, min_length=0))

    def test_minlength_None(self):
        self.assertTrue(valid.valid_1d_array(self.data, min_length=None))

    def test_minlength_equal_length(self):
        self.assertTrue(valid.valid_1d_array(self.data, min_length=7))

    def test_allow_none_wrong_type_string(self):
        self.assertRaises(TypeError, valid.valid_1d_array, self.data, allow_none="true")

    def test_allow_none_wrong_type_float(self):
        self.assertRaises(TypeError, valid.valid_1d_array, self.data, allow_none=0)

    def test_allow_none_true(self):
        self.assertTrue(valid.valid_1d_array(None, allow_none=True))

    def test_allow_none_false(self):
        self.assertRaises(ValueError, valid.valid_1d_array, None, allow_none=False)

    def test_allowedtypes_none(self):
        self.assertTrue(valid.valid_1d_array(array=self.data, allowed_types=None))

    def test_allowedtypes_not_type(self):
        self.assertRaises(
            TypeError, valid.valid_1d_array, array=self.data, allowed_types=(list, 0)
        )

    def test_allowedtypes_integral(self):
        self.assertTrue(valid.valid_1d_array(array=self.data, allowed_types=Integral))

    def test_allowedtypes_float(self):
        self.assertTrue(
            valid.valid_1d_array(
                array=np.ones(4, dtype=float), allowed_types=(int, float)
            )
        )

    def test_allowedtypes_wrongtype(self):
        self.assertRaises(
            TypeError, valid.valid_1d_array, value=self.data, allowed_types=float
        )

    def test_allowedvalues_wrong_type(self):
        self.assertRaises(
            TypeError, valid.valid_1d_array, array=self.data, allowed_values="a"
        )

    def test_allowedvalues_list(self):
        self.assertTrue(valid.valid_1d_array(array=self.data, allowed_values=[0, 1]))

    def test_allowedvalues_set(self):
        self.assertTrue(valid.valid_1d_array(array=self.data, allowed_values={0, 1}))

    def test_allowedvalues_array(self):
        self.assertTrue(
            valid.valid_1d_array(array=self.data, allowed_values=np.array([0, 1]))
        )

    def test_allowedvalues_correct(self):
        self.assertTrue(valid.valid_1d_array(array=self.data, allowed_values=1))

    def test_allowedvalues_wrong_value(self):
        self.assertRaises(
            ValueError, valid.valid_1d_array, array=self.data, allowed_values=2
        )

    def test_invalid_name(self):
        self.assertRaises(TypeError, valid.valid_1d_array, arrays=self.data, name=0)


class TestValidNdArray(unittest.TestCase):
    """
    Tests on valid_ndarray.

    valid_ndarray(arrays, ndim=None, shape=None, fix_ndim=True, fix_shape=True,
                  name=None)
    """

    def setUp(self) -> None:
        self.data = np.ones((7, 7))
        self.mask = np.zeros(self.data.shape)
        self.mask[3, 3] = 1

    def test_ndim_wrong_type_float(self):
        self.assertRaises(
            TypeError, valid.valid_ndarray, arrays=(self.data, self.mask), ndim=2.5
        )

    def test_ndim_type_list(self):
        self.assertTrue(valid.valid_ndarray(arrays=(self.data, self.mask), ndim=[2, 3]))

    def test_ndim_type_tuple(self):
        self.assertTrue(valid.valid_ndarray(arrays=(self.data, self.mask), ndim=(2, 3)))

    def test_ndim_type_set(self):
        self.assertRaises(
            TypeError, valid.valid_ndarray, arrays=(self.data, self.mask), ndim={2, 3}
        )

    def test_ndim_negative(self):
        self.assertRaises(
            ValueError, valid.valid_ndarray, arrays=(self.data, self.mask), ndim=-1
        )

    def test_ndim_null(self):
        self.assertRaises(
            ValueError, valid.valid_ndarray, arrays=(self.data, self.mask), ndim=0
        )

    def test_ndim_none(self):
        self.assertTrue(valid.valid_ndarray(arrays=(self.data, self.mask), ndim=None))

    def test_shape_wrong_type_number(self):
        self.assertRaises(
            TypeError, valid.valid_ndarray, arrays=(self.data, self.mask), shape=1
        )

    def test_ndim_shape_none(self):
        self.assertTrue(
            valid.valid_ndarray(arrays=(self.data, self.mask), ndim=None, shape=None)
        )

    def test_shape_none(self):
        self.assertTrue(valid.valid_ndarray(arrays=(self.data, self.mask), shape=None))

    def test_shape_item_none(self):
        self.assertTrue(
            valid.valid_ndarray(arrays=(self.data, self.mask), shape=(1, None))
        )

    def test_arrays_is_ndarray(self):
        self.assertTrue(valid.valid_ndarray(arrays=self.data))

    def test_arrays_empty(self):
        self.assertRaises(ValueError, valid.valid_ndarray, arrays=())

    def test_arrays_None(self):
        self.assertRaises(ValueError, valid.valid_ndarray, arrays=None)

    def test_arrays_mixed_types_int(self):
        self.assertRaises(TypeError, valid.valid_ndarray, arrays=(self.data, 1))

    def test_arrays_mixed_types_none(self):
        self.assertRaises(ValueError, valid.valid_ndarray, arrays=(self.data, None))

    def test_fix_ndim_wrong_type(self):
        self.assertRaises(TypeError, valid.valid_ndarray, arrays=self.data, fix_ndim=0)

    def test_fix_ndim_wrong_none(self):
        self.assertRaises(
            TypeError, valid.valid_ndarray, arrays=self.data, fix_ndim=None
        )

    def test_fix_shape_wrong_type(self):
        self.assertRaises(TypeError, valid.valid_ndarray, arrays=self.data, fix_shape=0)

    def test_fix_shape_wrong_none(self):
        self.assertRaises(
            TypeError, valid.valid_ndarray, arrays=self.data, fix_shape=None
        )

    def test_incompatible_dim(self):
        self.assertRaises(
            ValueError,
            valid.valid_ndarray,
            arrays=(self.data, np.zeros((1, 1, 1))),
            ndim=2,
        )

    def test_wrong_dim(self):
        self.assertRaises(
            ValueError,
            valid.valid_ndarray,
            arrays=self.data,
            ndim=(3, 4),
        )

    def test_incompatible_mixed_ndim(self):
        self.assertRaises(
            ValueError,
            valid.valid_ndarray,
            arrays=(self.data, np.ones(1)),
            ndim=(2, 1),
        )

    def test_compatible_mixed_ndim(self):
        self.assertTrue(
            valid.valid_ndarray(
                arrays=(self.data, np.ones(1)), ndim=(2, 1), fix_ndim=False
            )
        )

    def test_incompatible_shape(self):
        self.assertRaises(
            ValueError,
            valid.valid_ndarray,
            arrays=(self.data, np.zeros((1, 1))),
            ndim=2,
        )

    def test_fix_shape_false(self):
        self.assertTrue(
            valid.valid_ndarray(arrays=(self.data, np.zeros((1, 1))), fix_shape=False)
        )

    def test_invalid_name(self):
        self.assertRaises(TypeError, valid.valid_ndarray, arrays=self.data, name=0)


if __name__ == "__main__":
    run_tests(TestIsFloat)
    run_tests(TestValidContainer)
    run_tests(TestValidKwargs)
    run_tests(TestValidItem)
    run_tests(TestValid1dArray)
    run_tests(TestValidNdArray)
