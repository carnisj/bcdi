# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr
"""Functions related to formatting for string representation."""

import json
import logging
from inspect import signature
from typing import Any, List, Optional, Tuple

import numpy as np

module_logger = logging.getLogger(__name__)


class CustomEncoder(json.JSONEncoder):
    """Class to handle the serialization of np.ndarrays, sets."""

    def default(self, obj):
        """Override the JSONEncoder.default method to support more types."""
        if isinstance(obj, np.ndarray):
            return ndarray_to_list(obj)
            # Let the base class default method raise the TypeError
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


def create_repr(obj: Any, cls: type) -> str:
    """
    Generate the string representation of the object.

    It uses the parameters given to __init__, except self, args and kwargs.

    :param obj: the object for which the string representation should be generated
    :param cls: the cls from which __init__ parameters should be extracted (e.g., base
     class in case of inheritance)
    :return: the string representation
    """
    if not isinstance(cls, type):
        raise TypeError(f"'cls' should be a class, for {type(cls)}")
    output = obj.__class__.__name__ + "("
    for _, param in enumerate(
        signature(cls.__init__).parameters.keys()  # type: ignore
    ):
        if param not in ["self", "args", "kwargs"]:
            out, quote_mark = format_value(getattr(obj, param))
            output += f"{param}=" + format_repr(out, quote_mark)

    output += ")"
    return str(output)


def format_value(value: Any) -> Tuple[Any, bool]:
    """Format the value for a proper representation."""
    quote_mark = True
    if isinstance(value, np.ndarray):
        out = ndarray_to_list(value)
        quote_mark = True
    elif callable(value):
        out = value.__module__ + "." + value.__name__
        # it's a string, but we don't want to put it in quote mark in order to
        # be able to call it directly
        quote_mark = False
    elif isinstance(value, dict):
        out = {}  # type: ignore
        for key, val in value.items():
            if not isinstance(key, str):
                raise NotImplementedError(f"key {key} should be a string")
            out[key] = format_value(val)[0]  # type: ignore
        quote_mark = False
    else:
        out = value
    return out, quote_mark


def format_repr(value: Optional[Any], quote_mark: bool = True) -> str:
    """
    Format strings for the __repr__ method.

    :param value: string or None
    :param quote_mark: True to put quote marks around strings
    :return: a string
    """
    if isinstance(value, str) and quote_mark:
        return f'"{value}", '.replace("\\", "/")
    return f"{value}, ".replace("\\", "/")


def ndarray_to_list(array: np.ndarray) -> List:
    """
    Convert a numpy ndarray of any dimension to a nested list.

    :param array: the array to be converted
    """
    if not isinstance(array, np.ndarray):
        raise TypeError("a numpy ndarray is expected")
    if array.ndim == 1:
        return list(array)
    output = []
    for idx in range(array.shape[0]):
        output.append(ndarray_to_list(array[idx]))
    return output
