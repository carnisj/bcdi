# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

"""
Definition of the correct parameters for the config files.

Parameter validation is performed over the excepted parameters.
"""
from numbers import Number, Real
import numpy as np
from typing import Any
import bcdi.utils.validation as valid


class ParameterError(Exception):
    def __init__(self, key, value, allowed):
        super().__init__(f"Incorrect value {value} for parameter {key}\n"
                         f"Allowed are {allowed}")


def valid_param(key: str, value: Any) -> bool:
    """
    Validate a key value pair corresponding to an input parameter.

    It will raise an exception if the check fails.

    :param key: name of the parameter
    :param value: the value of the parameter
    :return: True if the check is sucessful, False if the key is not expected
    """
    # test the booleans first
    if key in {"align_q", "bin_during_loading",
               "custom_scan", "debug", "flag_interact", "is_series", "mask_zero_event",
               "reload_orthogonal", "reload_previous",
               "save_asint", "save_rawdata", "save_to_mat", "save_to_npz",
               "save_to_vti", "use_rawdata"}:
        valid.valid_item(value, allowed_types=bool, name=key)
    elif key == "absorption":
        valid.valid_item(value, allowed_types=float, min_excluded=0, name=key)
    elif key == "actuators":
        valid.valid_container(value, container_types=dict, allow_none=True, name=key)
    elif key == "beam_direction":
        valid.valid_container(
            value, container_types=(tuple, list), length=3, item_types=Real, name=key
        )
    elif key == "beamline":
        valid.valid_container(value, container_types=str, min_length=1, name=key)
    elif key == "custom_motors":
        valid.valid_container(value, container_types=dict, allow_none=True, name=key)
    elif key == "detector":
        valid.valid_container(value, container_types=str, min_length=1, name=key)
    elif key == "energy":
        if isinstance(value, Number):
            valid.valid_item(value, allowed_types=Real, min_excluded=0, name=key)
        else:
            valid.valid_container(
                value, container_types=(tuple, list, np.ndarray),
                min_length=1, item_types=Real, min_excluded=0, name=key
            )
    elif key == "fill_value_mask":
        allowed = {0, 1}
        if value not in allowed:
            raise ParameterError(key, value, allowed)
    elif key == "rocking_angle":
        allowed = {"outofplane", "inplane", "energy"}
        if value not in allowed:
            raise ParameterError(key, value, allowed)
    elif key == "sample_offsets":
        valid.valid_container(
            value, container_types=(tuple, list), allow_none=True, name=key
        )
    elif key == "scan":
        valid.valid_item(value, allowed_types=int, min_included=0, name=key)
    elif key == "scans":
        valid.valid_container(
            value, container_types=(tuple, list, np.ndarray), min_length=1, name=key
        )
    elif key == "sdd":
        valid.valid_item(value, allowed_types=Real, min_excluded=0, name=key)
    elif key == "specfile_name":
        valid.valid_container(value, container_types=str, name=key)
    elif key == "tilt_angle":
        valid.valid_item(value, allowed_types=Real, name=key)
    else:
        # this key is not in the known parameters
        return False
    return True
