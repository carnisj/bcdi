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
from typing import Any, Tuple
import bcdi.utils.validation as valid


class ParameterError(Exception):
    def __init__(self, key, value, allowed):
        super().__init__(
            f"Incorrect value {value} for parameter {key}\n" f"Allowed are {allowed}"
        )


def valid_param(key: str, value: Any) -> Tuple[Any, bool]:
    """
    Validate a key value pair corresponding to an input parameter.

    It will raise an exception if the check fails.

    :param key: name of the parameter
    :param value: the value of the parameter
    :return: a tuple (formatted_value, is_valid). is_valid is True if the key
     is valid, False otherwise.
    """
    is_valid = True
    # test the booleans first
    if key in {
        "align_axis",
        "align_q",
        "apodize",
        "bin_during_loading",
        "correct_refraction",
        "custom_scan",
        "debug",
        "flag_interact",
        "flip_reconstruction",
        "get_temperature",
        "grey_background",
        "invert_phase",
        "is_series",
        "keep_size",
        "mask_zero_event",
        "reload_orthogonal",
        "reload_previous",
        "save",
        "save_asint",
        "save_rawdata",
        "save_support",
        "save_to_mat",
        "save_to_npz",
        "save_to_vti",
        "simulation",
        "use_rawdata",
    }:
        valid.valid_item(value, allowed_types=bool, name=key)
    elif key == "absorption":
        valid.valid_item(value, allowed_types=Real, min_excluded=0, name=key)
    elif key == "actuators":
        valid.valid_container(value, container_types=dict, allow_none=True, name=key)
    elif key == "apodization_alpha":
        valid.valid_container(
            value,
            container_types=(tuple, list, np.ndarray),
            length=3,
            item_types=Real,
            min_included=0,
            name=key,
        )
        value = np.asarray(value)
    elif key == "apodization_mu":
        valid.valid_container(
            value,
            container_types=(tuple, list, np.ndarray),
            length=3,
            item_types=Real,
            min_included=0,
            name=key,
        )
        value = np.asarray(value)
    elif key == "apodization_sigma":
        valid.valid_container(
            value,
            container_types=(tuple, list, np.ndarray),
            length=3,
            item_types=Real,
            min_included=0,
            name=key,
        )
        value = np.asarray(value)
    elif key == "apodization_window":
        allowed = {"blackman", "tukey", "normal"}
        if value not in allowed:
            raise ParameterError(key, value, allowed)
    elif key == "avg_space":
        allowed = {"reciprocal_space", "direct_space"}
        if value not in allowed:
            raise ParameterError(key, value, allowed)
    elif key == "axis_to_align":
        valid.valid_container(
            value,
            container_types=(tuple, list, np.ndarray),
            length=3,
            item_types=Real,
            name=key,
        )
        value = np.asarray(value)
    elif key == "beam_direction":
        valid.valid_container(
            value,
            container_types=(tuple, list, np.ndarray),
            length=3,
            item_types=Real,
            name=key,
        )
    elif key == "beamline":
        valid.valid_container(value, container_types=str, min_length=1, name=key)
    elif key == "centering_method":
        allowed = {"com", "max", "max_com", "do_nothing"}
        if value not in allowed:
            raise ParameterError(key, value, allowed)
    elif key == "comment":
        valid.valid_container(value, container_types=str, name=key)
        if value and not value.startwith("_"):
            value += "_"
    elif key == "correlation_threshold":
        valid.valid_item(value, allowed_types=Real, min_included=0, name=key)
    elif key == "custom_motors":
        valid.valid_container(value, container_types=dict, allow_none=True, name=key)
    elif key == "data_frame":
        allowed = {"detector", "crystal", "laboratory"}
        if value not in allowed:
            raise ParameterError(key, value, allowed)
    elif key == "detector":
        valid.valid_container(value, container_types=str, min_length=1, name=key)
    elif key == "dispersion":
        valid.valid_item(value, allowed_types=Real, min_excluded=0, name=key)
    elif key == "energy":
        if isinstance(value, Number):
            valid.valid_item(value, allowed_types=Real, min_excluded=0, name=key)
        else:
            valid.valid_container(
                value,
                container_types=(tuple, list, np.ndarray),
                min_length=1,
                item_types=Real,
                min_excluded=0,
                name=key,
            )
    elif key == "fill_value_mask":
        allowed = {0, 1}
        if value not in allowed:
            raise ParameterError(key, value, allowed)
    elif key == "fix_voxel":
        valid.valid_item(
            value, allowed_types=Real, min_excluded=0, allow_none=True, name=key
        )
    elif key == "half_width_avg_phase":
        valid.valid_item(value, allowed_types=int, min_included=0, name=key)
    elif key == "inplane_angle":
        valid.valid_item(value, allowed_types=Real, name=key)
    elif key == "interp_method":
        allowed = {"xrayutilities", "linearization"}
        if value not in allowed:
            raise ParameterError(key, value, allowed)
    elif key == "isosurface_strain":
        valid.valid_item(value, allowed_types=Real, min_excluded=0, name=key)
    elif key == "offset_method":
        allowed = {"com", "mean"}
        if value not in allowed:
            raise ParameterError(key, value, allowed)
    elif key == "optical_path_method":
        allowed = {"threshold", "defect"}
        if value not in allowed:
            raise ParameterError(key, value, allowed)
    elif key == "original_size":
        valid.valid_container(
            value,
            container_types=(tuple, list, np.ndarray),
            length=3,
            item_types=int,
            min_excluded=0,
            allow_none=True,
            name=key,
        )
    elif key == "outofplane_angle":
        valid.valid_item(value, allowed_types=Real, name=key)
    elif key == "output_size":
        valid.valid_container(
            value,
            container_types=(tuple, list, np.ndarray),
            length=3,
            item_types=int,
            min_excluded=0,
            allow_none=True,
            name=key,
        )
    elif key == "phase_offset_origin":
        valid.valid_item(value, allowed_types=Real, allow_none=True, name=key)
    elif key == "phase_ramp_removal":
        allowed = {"gradient", "upsampling"}
        if value not in allowed:
            raise ParameterError(key, value, allowed)
    elif key == "phase_range":
        valid.valid_item(value, allowed_types=Real, min_excluded=0, name=key)
    elif key == "phasing_binning":
        valid.valid_container(
            value,
            container_types=(tuple, list, np.ndarray),
            length=3,
            item_types=int,
            min_excluded=0,
            name=key,
        )
    elif key == "pixel_size":
        valid.valid_item(
            value, allowed_types=Real, min_excluded=0, allow_none=True, name=key
        )
    elif key == "preprocessing_binning":
        valid.valid_container(
            value,
            container_types=(tuple, list, np.ndarray),
            length=3,
            item_types=int,
            min_excluded=0,
            name=key,
        )
    elif key == "ref_axis_q" or key == "ref_axis":
        allowed = {"x", "y", "z"}
        if value not in allowed:
            raise ParameterError(key, value, allowed)
    elif key == "reference_spacing":
        valid.valid_item(
            value, allowed_types=Real, min_included=0, allow_none=True, name=key
        )
    elif key == "reference_temperature":
        valid.valid_item(
            value, allowed_types=Real, min_included=0, allow_none=True, name=key
        )
    elif key == "reflection":
        valid.valid_container(
            value,
            container_types=(tuple, list, np.ndarray),
            length=3,
            item_types=Real,
            name=key,
        )
        value = np.asarray(value)
    elif key == "rocking_angle":
        allowed = {"outofplane", "inplane", "energy"}
        if value not in allowed:
            raise ParameterError(key, value, allowed)
    elif key == "roll_modes":
        valid.valid_container(
            value,
            container_types=(tuple, list, np.ndarray),
            length=3,
            item_types=int,
            name=key,
        )
    elif key == "root_folder":
        valid.valid_container(value, container_types=str, min_length=1, name=key)
    elif key == "sample_name":
        valid.valid_container(value, container_types=str, min_length=1, name=key)
    elif key == "sample_offsets":
        valid.valid_container(
            value, container_types=(tuple, list, np.ndarray), allow_none=True, name=key
        )
    elif key == "save_dir":
        valid.valid_container(
            value, container_types=str, min_length=1, allow_none=True, name=key
        )
        if isinstance(value, str) and not value.endswith("/"):
            value += "/"
    elif key == "save_frame":
        allowed = {"laboratory", "crystal", "lab_flat_sample"}
        if value not in allowed:
            raise ParameterError(key, value, allowed)
    elif key == "scan":
        valid.valid_item(value, allowed_types=int, min_included=0, name=key)
    elif key == "scans":
        valid.valid_container(
            value, container_types=(tuple, list, np.ndarray), min_length=1, name=key
        )
    elif key == "sdd":
        valid.valid_item(value, allowed_types=Real, min_excluded=0, name=key)
    elif key == "sort_method":
        allowed = {"mean_amplitude", "variance", "variance/mean", "volume"}
        if value not in allowed:
            raise ParameterError(key, value, allowed)
    elif key == "specfile_name":
        valid.valid_container(value, container_types=str, allow_none=True, name=key)
    elif key == "strain_method":
        allowed = {"default", "defect"}
        if value not in allowed:
            raise ParameterError(key, value, allowed)
    elif key == "strain_range":
        valid.valid_item(value, allowed_types=Real, min_excluded=0, name=key)
    elif key == "threshold_avg":
        valid.valid_item(
            value, allowed_types=Real, min_included=0, max_included=1, name=key
        )
    elif key == "template_imagefile":
        valid.valid_container(value, container_types=str, min_length=1, name=key)
    elif key == "threshold_gradient":
        valid.valid_item(value, allowed_types=Real, min_excluded=0, name=key)
    elif key == "threshold_unwrap_refraction":
        valid.valid_item(value, allowed_types=Real, min_included=0, name=key)
    elif key == "tick_direction":
        allowed = {"out", "in", "inout"}
        if value not in allowed:
            raise ParameterError(key, value, allowed)
    elif key == "tick_length":
        valid.valid_item(value, allowed_types=int, min_included=1, name=key)
    elif key == "tick_spacing":
        valid.valid_item(value, allowed_types=Real, min_excluded=0, name=key)
    elif key == "tick_width":
        valid.valid_item(value, allowed_types=int, min_included=1, name=key)
    elif key == "tilt_angle":
        valid.valid_item(value, allowed_types=Real, name=key)
    else:
        # this key is not in the known parameters
        is_valid = False

    return value, is_valid
