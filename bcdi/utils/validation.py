# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#   (c) 06/2021-present : DESY CFEL
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr
"""Functions related to the validation of input parameters."""

from collections import Sequence
from numbers import Real
import numpy as np


def valid_container(
    obj,
    container_types,
    length=None,
    min_length=None,
    max_length=None,
    item_types=None,
    name=None,
    min_included=None,
    min_excluded=None,
    max_included=None,
    max_excluded=None,
    allow_none=False,
):
    """
    Check that the input object as three elements fulfilling the defined requirements.

    :param obj: the object to be tested
    :param container_types: list of the allowed types for obj
    :param length: required length
    :param min_length: mininum length (inclusive)
    :param max_length: maximum length (inclusive)
    :param item_types: list of the allowed types for the object items
    :param min_included: minimum allowed value (inclusive)
    :param min_excluded: minimum allowed value (exclusive)
    :param max_included: maximum allowed value (inclusive)
    :param max_excluded: maximum allowed value (exclusive)
    :param allow_none: True if the container items are allowed to be None
    :param name: name of the calling object appearing in exception messages
    :return: True if checks pass, raise some error otherwise
    """
    supported_containers = {list, tuple, set, str, np.ndarray, dict}

    # check the validity of the requirements
    if container_types is None:
        raise ValueError("at least one type must be specified for the container")
    if isinstance(container_types, type):
        container_types = (container_types,)
    if not all(isinstance(val, type) for val in container_types):
        raise TypeError("container_types should be a collection of valid types")
    if not all(val in supported_containers for val in container_types):
        raise TypeError(f"non supported container type {container_types}")

    if length is not None:
        if not isinstance(length, int):
            raise TypeError("length should be an integer")
        if length < 0:
            raise ValueError("length should be a strictly positive integer")

    if min_length is not None:
        if not isinstance(min_length, int):
            raise TypeError("min_length should be an integer")
        if min_length < 0:
            raise ValueError("min_length should be a positive integer")

    if max_length is not None:
        if not isinstance(max_length, int):
            raise TypeError("max_length should be an integer")
        if max_length < 0:
            raise ValueError("max_length should be a positive integer")
        if min_length is not None and max_length < min_length:
            raise ValueError("max_length should be larger or equal to min_length")

    if item_types is not None:
        if isinstance(item_types, type):
            item_types = (item_types,)
        if not all(isinstance(val, type) for val in item_types):
            raise TypeError("type_elements should be a collection of valid types")

    if min_included is not None:
        if not isinstance(min_included, Real):
            raise TypeError("min_included should be a real number")

    if min_excluded is not None:
        if not isinstance(min_excluded, Real):
            raise TypeError("min_excluded should be a real number")

    if max_included is not None:
        if not isinstance(max_included, Real):
            raise TypeError("max_included should be a real number")

    if max_excluded is not None:
        if not isinstance(max_excluded, Real):
            raise TypeError("max_excluded should be a real number")

    if not isinstance(allow_none, bool):
        raise TypeError("allow_none should be a boolean")

    name = name or "obj"

    # check if the object is None
    if not allow_none and obj is None:
        raise ValueError(f"{name}: None for the container is not allowed")

    # check the type of obj
    if obj is not None and not isinstance(obj, container_types):
        raise TypeError(
            f"{name}: type(container)={type(obj)}, allowed is {container_types}"
        )

    # check the length of obj
    if obj is not None and length is not None:
        try:
            if len(obj) != length:
                raise ValueError(f"{name}: the container should be of length {length}")
        except TypeError as ex:
            raise TypeError(
                f"method __len__ not defined for the type(s) {container_types}"
            ) from ex

    # check the min_length of obj
    if obj is not None and min_length is not None:
        try:
            if len(obj) < min_length:
                raise ValueError(
                    f"{name}: the container should be of length >= {min_length}"
                )
        except TypeError as ex:
            raise TypeError(
                f"method __len__ not defined for the type(s) {container_types}"
            ) from ex

    # check the max_length of obj
    if obj is not None and max_length is not None:
        try:
            if len(obj) > max_length:
                raise ValueError(
                    f"{name}: the container should be of length <= {max_length}"
                )
        except TypeError as ex:
            raise TypeError(
                f"method __len__ not defined for the type(s) {container_types}"
            ) from ex

    # check the presence of None in the items
    if not allow_none and obj is not None:
        if any(val is None for val in obj):
            raise ValueError(f"{name}: None for the items is not allowed")

    # check the type and value of each items in obj
    if obj is not None and item_types is not None:
        for val in obj:
            valid_item(
                value=val,
                allowed_types=item_types,
                min_included=min_included,
                min_excluded=min_excluded,
                max_included=max_included,
                max_excluded=max_excluded,
                allow_none=allow_none,
                name=name,
            )

    # every tests passed, return True
    return True


def valid_kwargs(kwargs, allowed_kwargs, name=None):
    """
    Check if the provided parameters belong to the set of allowed kwargs.

    :param kwargs: dictionnary of kwargs to check
    :param allowed_kwargs: set of allowed keys
    :param name: name of the calling object appearing in exception messages
    :return: True if checks pass, raise some error otherwise
    """
    # check the validity of the parameters
    if not isinstance(kwargs, dict):
        raise TypeError("kwargs should be a dictionnary")

    if isinstance(allowed_kwargs, str):
        allowed_kwargs = (allowed_kwargs,)
    if not isinstance(allowed_kwargs, (tuple, list, set)):
        raise TypeError("allowed_kwargs should be a collection of kwargs keys")
    if not all(isinstance(val, str) for val in allowed_kwargs):
        raise TypeError("the keys in allowed_kwargs should be strings")
    if not all(len(val) > 0 for val in allowed_kwargs):
        raise ValueError(
            "the length of all keys in allowed_kwargs should be larger than 0"
        )

    valid_container(
        obj=allowed_kwargs,
        container_types=(tuple, list, set),
        min_length=1,
        name="valid_kwargs",
    )

    # check the kwargs
    for k in kwargs.keys():
        if k not in allowed_kwargs:
            raise KeyError(f"{name}: unknown keyword argument given:", k)

    # every tests passed, return True
    return True


def valid_item(
    value,
    allowed_types,
    min_included=None,
    min_excluded=None,
    max_included=None,
    max_excluded=None,
    allow_none=False,
    name=None,
):
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
    :return: True if checks pass, raise some error otherwise
    """
    # check the validity of the requirements
    if allowed_types is None:
        raise ValueError("at least one allowed type must be specified for the value")
    if isinstance(allowed_types, type):
        allowed_types = (allowed_types,)
    if not all(isinstance(val, type) for val in allowed_types):
        raise TypeError("allowed_types should be a collection of valid types")

    if min_included is not None:
        if not isinstance(min_included, Real):
            raise TypeError("min_included should be a real number")

    if min_excluded is not None:
        if not isinstance(min_excluded, Real):
            raise TypeError("min_excluded should be a real number")

    if max_included is not None:
        if not isinstance(max_included, Real):
            raise TypeError("max_included should be a real number")

    if max_excluded is not None:
        if not isinstance(max_excluded, Real):
            raise TypeError("max_excluded should be a real number")

    if not isinstance(allow_none, bool):
        raise TypeError("allow_none should be a boolean")

    name = name or "obj"

    # check if value is None
    if not allow_none and value is None:
        raise ValueError(f"{name}: None is not allowed")

    # check the type of obj
    if value is not None and not isinstance(value, allowed_types):
        raise TypeError(f"{name}: wrong type for value, allowed is {allowed_types}")

    # check min_included
    if min_included is not None and value is not None:
        try:
            if value < min_included:
                raise ValueError(
                    f"{name}: value should be larger or equal to {min_included}"
                )
        except TypeError as ex:
            raise TypeError(
                f"{name}: '<' not supported between instances of "
                f"'{type(value)}' and '{type(min_included)}'"
            ) from ex

    # check min_excluded
    if min_excluded is not None and value is not None:
        try:
            if value <= min_excluded:
                raise ValueError(
                    f"{name}: value should be strictly larger than {min_excluded}"
                )
        except TypeError as ex:
            raise TypeError(
                f"{name}: '<=' not supported between instances of "
                f"'{type(value)}' and '{type(min_excluded)}'"
            ) from ex

    # check max_included
    if max_included is not None and value is not None:
        try:
            if value > max_included:
                raise ValueError(
                    f"{name}: value should be smaller or equal to {max_included}"
                )
        except TypeError as ex:
            raise TypeError(
                f"{name}: '>' not supported between instances of "
                f"'{type(value)}' and '{type(max_included)}'"
            ) from ex

    # check max_excluded
    if max_excluded is not None and value is not None:
        try:
            if value >= max_excluded:
                raise ValueError(
                    f"{name}: value should be strictly smaller than {max_excluded}"
                )
        except TypeError as ex:
            raise TypeError(
                f"{name}: '>=' not supported between instances of "
                f"'{type(value)}' and '{type(max_excluded)}'"
            ) from ex

    # every tests passed, return True
    return True


def valid_ndarray(arrays, ndim=None, shape=None):
    """
    Check that arrays have the same shape and the correct number of dimensions.

    :param arrays: a sequence of numpy ndarrays
    :param ndim: int, the number of dimensions to be compared with
    :param shape: sequence of int, shape to be comared with
    :return: True if checks pass, raise some error otherwise
    """
    # check the validity of the requirements
    valid_item(ndim, allowed_types=int, min_excluded=0, allow_none=True, name="ndim")
    valid_container(
        shape,
        container_types=(tuple, list),
        item_types=int,
        min_excluded=0,
        allow_none=True,
        name="shape",
    )
    if isinstance(arrays, np.ndarray):
        arrays = (arrays,)
    valid_container(
        arrays,
        container_types=(tuple, list),
        item_types=np.ndarray,
        min_length=1,
        name="arrays"
    )

    # check arrays
    if ndim is None:
        ndim = arrays[0].ndim
    if not all(array.ndim == ndim for array in arrays):
        raise ValueError(f"all arrays should have the same dimension {ndim}")
    if shape is None or any(val is None for val in shape):
        shape = arrays[0].shape
    if not all(array.shape == shape for array in arrays):
        raise ValueError(f"all arrays should have the same shape {shape}")

    # every tests passed, return True
    return True
