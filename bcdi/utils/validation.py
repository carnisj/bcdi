# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

from numbers import Number, Real


def valid_container(obj, container_types, length=None, min_length=None, item_types=None, min_included=None,
                    min_excluded=None, max_included=None, max_excluded=None, allow_none=False, name=None):
    """
    Check that the input object as three elements fulfilling the defined requirements.

    :param obj: the object to be tested
    :param container_types: list of the allowed types for obj
    :param length: required length
    :param min_length: mininum length (inclusive)
    :param item_types: list of the allowed types for the object items
    :param min_included: minimum allowed value (inclusive)
    :param min_excluded: minimum allowed value (exclusive)
    :param max_included: maximum allowed value (inclusive)
    :param max_excluded: maximum allowed value (exclusive)
    :param allow_none: True if the container items are allowed to be None
    :param name: name of the calling object appearing in exception messages
    """
    # check the validity of the requirements
    if container_types is None:
        raise ValueError('at least one type must be specified for the container')
    if type(container_types) == type:
        container_types = (container_types,)
    if not all(isinstance(val, type) for val in container_types):
        raise TypeError('container_types should be a collection of valid types')
    if not all(val in {list, tuple, set} for val in container_types):
        raise TypeError('container_types should be a collection of types inheriting from {tuple, list, set}')

    if length is not None:
        if not isinstance(length, int):
            raise TypeError('length should be an integer')
        if length <= 0:
            raise ValueError('length should be a strictly positive integer')

    if min_length is not None:
        if not isinstance(min_length, int):
            raise TypeError('min_length should be an integer')
        if min_length < 0:
            raise ValueError('min_length should be a positive integer')

    if item_types is not None:
        if type(item_types) == type:
            item_types = (item_types,)
        if not all(isinstance(val, type) for val in item_types):
            raise TypeError('type_elements should be a collection of valid types')

    if min_included is not None:
        if not isinstance(min_included, Real):
            raise TypeError('min_included should be a real number')

    if min_excluded is not None:
        if not isinstance(min_excluded, Real):
            raise TypeError('min_excluded should be a real number')

    if max_included is not None:
        if not isinstance(max_included, Real):
            raise TypeError('max_included should be a real number')

    if max_excluded is not None:
        if not isinstance(max_excluded, Real):
            raise TypeError('max_excluded should be a real number')

    if not isinstance(allow_none, bool):
        raise TypeError('allow_none should be a boolean')

    name = name or 'obj'

    # check the type of obj
    if not isinstance(obj, container_types):
        raise TypeError(f'type({name})={type(obj)}, allowed is {container_types}')

    # check the length of obj
    if length is not None:
        try:
            if len(obj) != length:
                raise ValueError(f'{name} should be of length {length}')
        except TypeError as ex:
            raise TypeError(f'method __len__ not defined for the type(s) {container_types}') from ex

    # check the min_length of obj
    if min_length is not None:
        try:
            if len(obj) < min_length:
                raise ValueError(f'{name}: the container should be of length >= {min_length}')
        except TypeError as ex:
            raise TypeError(f'method __len__ not defined for the type(s) {container_types}') from ex

    # check the presence of None in the items
    if not allow_none:
        if any(val is None for val in obj):
            raise ValueError(f'{name}: None is not allowed')

    # check the type and value of each items in obj
    if item_types is not None:
        for val in obj:
            valid_item(value=val, allowed_types=item_types, min_included=min_included, min_excluded=min_excluded,
                       max_included=max_included, max_excluded=max_excluded, allow_none=allow_none, name=name)

    # every tests passed, return True
    return True


def valid_kwargs(kwargs, allowed_kwargs, name=None):
    """
    Check if the provided parameters belong to the set of allowed kwargs.

    :param kwargs: dictionnary of kwargs to check
    :param allowed_kwargs: set of allowed keys
    :param name: name of the calling object appearing in exception messages
    """
    # check the validity of the parameters
    if not isinstance(kwargs, dict):
        raise TypeError('kwargs should be a dictionnary')

    valid_container(obj=allowed_kwargs, container_types=(tuple, list, set), min_length=1, name='valid_kwargs')

    # check the kwargs
    for k in kwargs.keys():
        if k not in allowed_kwargs:
            raise Exception(f"{name}: unknown keyword argument given:", k)

    # every tests passed, return True
    return True


def valid_item(value, allowed_types, min_included=None, min_excluded=None, max_included=None, max_excluded=None,
               allow_none=False, name=None):
    """
    Check that the input object as three elements fulfilling the defined requirements.

    :param value: the value to be tested
    :param allowed_types: allowed types of the object values
    :param min_included: minimum allowed value (inclusive)
    :param min_excluded: minimum allowed value (exclusive)
    :param max_included: maximum allowed value (inclusive)
    :param max_excluded: maximum allowed value (exclusive)
    :param allow_none: True if the container items are allowed to be None
    :param name: name of the calling object appearing in exception messages
    """
    # check the validity of the requirements
    if allowed_types is None:
        raise ValueError('at least one allowed type must be specified for the value')
    if type(allowed_types) == type:
        allowed_types = (allowed_types,)
    if not len(allowed_types):
        raise ValueError('at least one allowed type must be specified for the value')
    if not all(isinstance(val, type) for val in allowed_types):
        raise TypeError('allowed_types should be a collection of valid types')

    if min_included is not None:
        if not isinstance(min_included, Real):
            raise TypeError('min_included should be a real number')

    if min_excluded is not None:
        if not isinstance(min_excluded, Real):
            raise TypeError('min_excluded should be a real number')

    if max_included is not None:
        if not isinstance(max_included, Real):
            raise TypeError('max_included should be a real number')

    if max_excluded is not None:
        if not isinstance(max_excluded, Real):
            raise TypeError('max_excluded should be a real number')

    if not isinstance(allow_none, bool):
        raise TypeError('allow_none should be a boolean')

    name = name or 'obj'

    # check if value is None
    if not allow_none and value is None:
        raise ValueError(f'{name}: None is not allowed')

    # check the type of obj
    if value is not None and not isinstance(value, allowed_types):
        raise TypeError(f'{name}: wrong type for value, allowed is {allowed_types}')

    # check min_included
    if min_included is not None and value is not None:
        try:
            if value < min_included:
                raise ValueError(f'{name}: value should be larger or equal to {min_included}')
        except TypeError as ex:
            raise TypeError(f"{name}: '<' not supported between instances of "
                            f"'{type(value)}' and '{type(min_included)}'") from ex

    # check min_excluded
    if min_excluded is not None and value is not None:
        try:
            if value <= min_excluded:
                raise ValueError(f'{name}: value should be strictly larger than {min_excluded}')
        except TypeError as ex:
            raise TypeError(f"{name}: '<=' not supported between instances of "
                            f"'{type(value)}' and '{type(min_excluded)}'") from ex

    # check max_included
    if max_included is not None and value is not None:
        try:
            if value > max_included:
                raise ValueError(f'{name}: value should be smaller or equal to {max_included}')
        except TypeError as ex:
            raise TypeError(f"{name}: '>' not supported between instances of "
                            f"'{type(value)}' and '{type(max_included)}'") from ex

    # check max_excluded
    if max_excluded is not None and value is not None:
        try:
            if value >= max_excluded:
                raise ValueError(f'{name}: value should be strictly smaller than {max_excluded}')
        except TypeError as ex:
            raise TypeError(f"{name}: '>=' not supported between instances of "
                            f"'{type(value)}' and '{type(max_excluded)}'") from ex

    # every tests passed, return True
    return True
