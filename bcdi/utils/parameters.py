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
import os
from typing import Any, Tuple
import bcdi.utils.validation as valid


class ParameterError(Exception):
    """
    Custom Exception for a dictionary of parsed parameters.

    :param key: a key of the dictionary
    :param value: the corresponding value
    :param allowed: allowed values for value
    """

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
    allowed: Any = None

    # convert 'None' to None
    if value == "None":
        value = None

    # convert 'True' to True
    if value in ("True", "true", "TRUE"):
        value = True

    # convert 'False' to False
    if value in ("False", "false", "FALSE"):
        value = False

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
        "save_as_int",
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
    elif key == "averaging_space":
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
    elif key == "backend":
        allowed = {"Agg", "Qt5Agg", "module://matplotlib_inline.backend_inline"}
        if value not in allowed:
            raise ParameterError(key, value, allowed)
    elif key == "background_file":
        valid.valid_container(
            value, container_types=str, min_length=1, allow_none=True, name=key
        )
    elif key == "background_plot":
        valid.valid_container(value, container_types=str, min_length=1, name=key)
    elif key == "beam_direction":
        valid.valid_container(
            value,
            container_types=(tuple, list, np.ndarray),
            length=3,
            item_types=Real,
            name=key,
        )
        value = np.asarray(value)
    elif key == "beamline":
        valid.valid_container(value, container_types=str, min_length=1, name=key)
    elif key == "bragg_peak":
        valid.valid_container(
            value,
            container_types=(tuple, list),
            item_types=int,
            min_included=0,
            length=3,
            allow_none=True,
            name=key,
        )
    elif key == "cch1":
        valid.valid_item(value, allowed_types=Real, name=key)
    elif key == "cch2":
        valid.valid_item(value, allowed_types=Real, name=key)
    elif key == "center_fft":
        allowed = {
            "crop_sym_ZYX",
            "crop_asym_ZYX",
            "pad_asym_Z_crop_sym_YX",
            "pad_sym_Z_crop_asym_YX",
            "pad_sym_Z",
            "pad_asym_Z",
            "pad_sym_ZYX",
            "pad_asym_ZYX",
            "skip",
        }
        if value not in allowed:
            raise ParameterError(key, value, allowed)
    elif key == "centering_method":
        allowed = {"com", "max", "max_com", "do_nothing"}
        if value not in allowed:
            raise ParameterError(key, value, allowed)
    elif key == "comment":
        valid.valid_container(value, container_types=str, name=key)
        if value and not value.startwith("_"):
            value += "_"
    elif key == "config_file":
        valid.valid_container(value, container_types=str, min_length=1, name=key)
        if not os.path.isfile(value):
            raise ValueError(f"The directory {value} does not exist")
    elif key == "correlation_threshold":
        valid.valid_item(value, allowed_types=Real, min_included=0, name=key)
    elif key == "custom_images":
        valid.valid_container(
            value,
            container_types=(tuple, list, np.ndarray),
            item_types=int,
            min_included=0,
            allow_none=True,
            name=key,
        )
    elif key == "custom_monitor":
        valid.valid_container(
            value,
            container_types=(tuple, list, np.ndarray),
            item_types=Real,
            min_included=0,
            allow_none=True,
            name=key,
        )
    elif key == "custom_motors":
        valid.valid_container(value, container_types=dict, allow_none=True, name=key)
    elif key == "data_dir":
        if value is not None:
            valid.valid_container(value, container_types=str, min_length=1, name=key)
            if not os.path.isdir(value):
                raise ValueError(f"The directory {value} does not exist")
    elif key == "data_frame":
        allowed = {"detector", "crystal", "laboratory"}
        if value not in allowed:
            raise ParameterError(key, value, allowed)
    elif key == "dirbeam_detector_angles":
        valid.valid_container(
            value,
            container_types=(list, tuple),
            item_types=Real,
            length=2,
            allow_none=True,
            name=key
        )
    elif key == "direct_beam":
        valid.valid_container(
            value,
            container_types=(list, tuple),
            item_types=Real,
            length=2,
            allow_none=True,
            name=key
        )
    elif key == "detector":
        valid.valid_container(value, container_types=str, min_length=1, name=key)
    elif key == "detrot":
        valid.valid_item(value, allowed_types=Real, name=key)
    elif key == "dispersion":
        valid.valid_item(value, allowed_types=Real, min_excluded=0, name=key)
    elif key == "energy":
        if value is None or isinstance(value, Number):
            valid.valid_item(
                value, allowed_types=Real, min_excluded=0, allow_none=True, name=key
            )
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
        allowed = (0, 1)
        if value not in allowed:
            raise ParameterError(key, value, allowed)
    elif key == "fix_bragg":
        valid.valid_container(
            value,
            container_types=(tuple, list),
            length=3,
            item_types=int,
            allow_none=True,
            name=key,
        )
    elif key == "fix_size":
        valid.valid_container(
            value,
            container_types=(tuple, list),
            length=6,
            item_types=int,
            allow_none=True,
            name=key,
        )
    elif key == "fix_voxel":
        valid.valid_item(
            value, allowed_types=Real, min_excluded=0, allow_none=True, name=key
        )
    elif key == "flatfield_file":
        valid.valid_container(
            value, container_types=str, min_length=1, allow_none=True, name=key
        )
    elif key == "frames_pattern":
        if value is not None:
            value = np.asarray(value)
            valid.valid_1d_array(
                value, allow_none=False, allowed_values={0, 1}, name=key
            )
    elif key == "half_width_avg_phase":
        valid.valid_item(value, allowed_types=int, min_included=0, name=key)
    elif key == "hotpixels_file":
        valid.valid_container(
            value, container_types=str, min_length=1, allow_none=True, name=key
        )
    elif key == "inplane_angle":
        valid.valid_item(value, allowed_types=Real, allow_none=True, name=key)
    elif key == "interpolation_method":
        allowed = {"xrayutilities", "linearization"}
        if value not in allowed:
            raise ParameterError(key, value, allowed)
    elif key == "isosurface_strain":
        valid.valid_item(value, allowed_types=Real, min_excluded=0, name=key)
    elif key == "linearity_func":
        valid.valid_container(
            value,
            container_types=(tuple, list, np.ndarray),
            length=5,
            item_types=Real,
            allow_none=True,
            name=key,
        )
    elif key == "median_filter":
        allowed = {"median", "interp_isolated", "mask_isolated", "skip"}
        if value not in allowed:
            raise ParameterError(key, value, allowed)
    elif key == "median_filter_order":
        valid.valid_item(value, allowed_types=int, min_included=0, name=key)
    elif key == "normalize_flux":
        allowed = {"monitor", "skip"}
        if value not in allowed:
            raise ParameterError(key, value, allowed)
    elif key == "offset_inplane":
        valid.valid_item(value, allowed_types=Real, name=key)
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
        valid.valid_item(value, allowed_types=Real, allow_none=True, name=key)
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
    elif key == "pad_size":
        valid.valid_container(
            value,
            container_types=(tuple, list),
            length=3,
            item_types=int,
            allow_none=True,
            name=key,
        )
    elif key == "phase_offset":
        valid.valid_item(value, allowed_types=Real, allow_none=True, name=key)
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
    elif key == "photon_filter":
        allowed = {"loading", "postprocessing"}
        if value not in allowed:
            raise ParameterError(key, value, allowed)
    elif key == "photon_threshold":
        valid.valid_item(value, allowed_types=Real, min_included=0, name=key)
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
    elif key == "reconstruction_file":
        valid.valid_container(
            value, container_types=str, min_length=1, allow_none=True, name=key
        )
    elif key in {"ref_axis_q", "ref_axis"}:
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
    elif key == "roi_detector":
        valid.valid_container(
            value,
            container_types=(tuple, list, np.ndarray),
            length=4,
            item_types=int,
            allow_none=True,
            name=key,
        )
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
        if not os.path.isdir(value):
            raise ValueError(f"The directory {value} does not exist")
    elif key == "sample_inplane":
        valid.valid_container(
            value,
            container_types=(tuple, list),
            length=3,
            item_types=Real,
            name=key,
        )
    elif key == "sample_name":
        valid.valid_container(value, container_types=str, min_length=1, name=key)
    elif key == "sample_offsets":
        valid.valid_container(
            value, container_types=(tuple, list, np.ndarray), allow_none=True, name=key
        )
    elif key == "sample_outofplane":
        valid.valid_container(
            value,
            container_types=(tuple, list),
            length=3,
            item_types=Real,
            name=key,
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
        if isinstance(value, Real):
            value = (value,)
        valid.valid_container(
            value, container_types=(tuple, list, np.ndarray), min_length=1, name=key
        )

    elif key == "sdd":
        valid.valid_item(
            value, allowed_types=Real, min_excluded=0, allow_none=True, name=key
        )
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
        valid.valid_container(
            value, container_types=str, min_length=0, allow_none=True, name=key
        )
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
        valid.valid_item(value, allowed_types=Real, allow_none=True, name=key)
    elif key == "tiltazimuth":
        valid.valid_item(value, allowed_types=Real, name=key)
    elif key == "tilt_detector":
        valid.valid_item(value, allowed_types=Real, name=key)
    elif key == "x_bragg":
        valid.valid_item(value, allowed_types=int, allow_none=True, name=key)
    elif key == "y_bragg":
        valid.valid_item(value, allowed_types=int, allow_none=True, name=key)
    else:
        # this key is not in the known parameters
        is_valid = False

    return value, is_valid
