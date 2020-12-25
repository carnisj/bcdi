# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

from numbers import Number, Real


def valid_container(obj, container_type, length=None, item_type=None, allow_none=False, strictly_positive=False,
                    name=None):
    """
    Check that the input object as three elements fulfilling the defined requirements.

    :param obj: the object to be tested
    :param container_type: list of the allowed types for obj
    :param length: required length
    :param item_type: allowed type of the object values
    :param allow_none: True if the container items are allowed to be None
    :param strictly_positive: True is object values must all be strictly positive.
    :param name: name of the object appearing in exception messages
    """
    # check the validity of the requirements
    if container_type is None:
        raise ValueError('at least one type must be specified for the container')
    container_type = tuple(container_type)
    if not len(container_type):
        raise ValueError('at least one type must be specified for the container')
    if not all(isinstance(val, type) for val in container_type):
        raise TypeError('container_type should be a collection of valid types')

    if length is not None:
        if not isinstance(length, int) or length <= 0:
            raise ValueError('length should be a strictly positive integer')

    if item_type is not None:
        item_type = tuple(item_type)
        if not all(isinstance(val, type) for val in item_type):
            raise TypeError('type_elements should be a collection of valid types')

    if not isinstance(allow_none, bool):
        raise TypeError('allow_none should be a boolean')

    if not isinstance(strictly_positive, bool):
        raise TypeError('strictly_positive should be a boolean')

    if allow_none and strictly_positive:
        raise TypeError("'>' not supported between instances of 'NoneType' and 'Number'")

    name = name or 'obj'

    # check the type of obj
    if not isinstance(obj, container_type):
        raise TypeError(f'type({name})={type(obj)}, allowed is {container_type}')

    # check the length of obj
    if length is not None:
        try:
            if len(obj) != length:
                raise ValueError(f'{name} should be of length {length}')
        except TypeError as ex:
            raise TypeError(f'method __len__ not defined for the type(s) {container_type}') from ex

    # check the type of the items in obj
    if item_type is not None:
        if allow_none:
            for val in obj:
                if val is not None and not isinstance(val, item_type):
                    raise TypeError(f'{name}: wrong type for items, allowed is {item_type} or None')
        else:
            if not all(isinstance(val, item_type) for val in obj):
                raise TypeError(f'{name}: wrong type for items, allowed is {item_type}')

    # check the presence of None in the items
    if not allow_none:
        if any(val is None for val in obj):
            raise ValueError(f'{name}: None is not allowed')

    # check the positivity of the items in obj
    if all(isinstance(val, Real) for val in obj) and strictly_positive:
        if not all(val > 0 for val in obj):
            raise ValueError(f'all items in {name} should be strictly positive')
