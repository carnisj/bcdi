# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

"""
Validation of configuration parameters.

The validation is performed only on the expected parameters. Other parameters are simply
discarded.
"""
import copy
import logging
import os
import pathlib
from abc import ABC, abstractmethod
from numbers import Number, Real
from typing import Any, Dict, List, Optional, Set, Tuple

import colorcet as cc
import matplotlib
import numpy as np

import bcdi.utils.utilities as util
import bcdi.utils.validation as valid
from bcdi.graph.colormap import ColormapFactory

logger = logging.getLogger(__name__)


class MissingKeyError(Exception):
    """Custom Exception for a missing required key in the config dictionary."""


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


class ConfigChecker(ABC):
    """
    Validate and configure parameters.

    :param initial_params: the dictionary of parameters to validate and configure
    :param default_values: an optional dictionary of default values for keys in
     initial_params
    :param match_length_params: a tuple of keys from initial_params which should match
     a certain length (e.g. the number of scans)
    :param required_params: a tuple of keys that have to be present in initial_params
    """

    def __init__(
        self,
        initial_params: Dict[str, Any],
        default_values: Optional[Dict[str, Any]] = None,
        match_length_params: Tuple = (),
        required_params: Tuple = (),
    ) -> None:
        self.initial_params = initial_params
        self.default_values = default_values
        self.match_length_params = match_length_params
        self.required_params = required_params
        self._checked_params = copy.deepcopy(self.initial_params)
        self._nb_scans: Optional[int] = None

    def check_config(self) -> Dict[str, Any]:
        """Check if the provided config is consistent."""
        if self._checked_params.get("scans") is None:
            raise ValueError("no scan provided")
        self._nb_scans = len(self._checked_params["scans"])

        for key in self.match_length_params:
            self._check_length(key, self._nb_scans)

        self._assign_default_value()
        self._check_mandatory_params()
        self._configure_params()
        self._check_backend()
        self._create_dirs()
        self._create_colormap()
        return self._checked_params

    def _create_dirs(self) -> None:
        """Check if the directories exist and create them if needed."""
        if not isinstance(self._checked_params.get("save_dir"), list):
            raise TypeError(
                "save_dir should be a list, got a "
                f"{type(self._checked_params.get('save_dir'))}"
            )
        for _, val in enumerate(self._checked_params["save_dir"]):
            if val is not None:
                pathlib.Path(val).mkdir(parents=True, exist_ok=True)

    def _create_roi(self) -> Optional[List[int]]:
        """
        Load "roi_detector" from the dictionary of parameters and update it.

        If the keys "center_roi_x" or "center_roi_y" are defined, it will consider that
        the current values in roi_detector define a window around the Bragg peak
        position and the final output will be:
        [center_roi_y - roi_detector[0], center_roi_y + roi_detector[1],
        center_roi_x - roi_detector[2], center_roi_x + roi_detector[3]].

        If a key is not defined, it will consider that the values of roi_detector are
        absolute pixels positions, e.g. if only "center_roi_y" is defined, the output
        will be:
        [center_roi_y - roi_detector[0], center_roi_y + roi_detector[1],
        roi_detector[2], roi_detector[3]].

        Accordingly, if none of the keys are defined, the output will be:
        [roi_detector[0], roi_detector[1], roi_detector[2], roi_detector[3]].

        :return: the calculated region of interest [Vstart, Vstop, Hstart, Hstop] or
         None.
        """
        roi = copy.deepcopy(self._checked_params.get("roi_detector"))

        # update the ROI
        if roi is not None:
            center_roi_y = self._checked_params.get("center_roi_y")
            if center_roi_y is not None:
                valid.valid_item(center_roi_y, allowed_types=int, name="center_roi_y")
                roi[0] = center_roi_y - self._checked_params["roi_detector"][0]
                roi[1] = center_roi_y + self._checked_params["roi_detector"][1]

            center_roi_x = self._checked_params.get("center_roi_x")
            if center_roi_x is not None:
                valid.valid_item(center_roi_x, allowed_types=int, name="center_roi_x")
                roi[2] = center_roi_x - self._checked_params["roi_detector"][2]
                roi[3] = center_roi_x + self._checked_params["roi_detector"][3]

            return [int(val) for val in roi]
        return None

    def _assign_default_value(self) -> None:
        """Assign default values to parameters."""
        if self.default_values is not None:
            for key, value in self.default_values.items():
                self._checked_params[key] = self._checked_params.get(key, value)

    def _check_backend(self) -> None:
        """Check if the backend is supported."""
        try:
            matplotlib.use(self._checked_params["backend"])
        except ModuleNotFoundError:
            raise ValueError(
                f"{self._checked_params['backend']} backend is not supported."
            )
        except ImportError:
            raise ValueError(f"cannot load backend {self._checked_params['backend']}")

    def _check_length(self, param_name: str, length: int) -> None:
        """Ensure that a parameter as the correct type and length."""
        initial_param = self._checked_params.get(param_name)
        if initial_param is None:
            initial_param = (None,) * length
        elif not isinstance(initial_param, (tuple, list)):
            raise TypeError(
                f"'{param_name}' shold be a tuple or a list, got {type(param_name)}"
            )
        if len(initial_param) == 1:
            self._checked_params[param_name] = initial_param * length
        elif len(initial_param) != length:
            raise ValueError(
                f"'{param_name}' should be of length {length}, "
                f"got {len(param_name)} elements"
            )
        else:
            self._checked_params[param_name] = initial_param

    def _check_mandatory_params(self) -> None:
        """Check if mandatory parameters are provided."""
        for key in self.required_params:
            try:
                _ = self._checked_params[key]
            except KeyError:
                raise MissingKeyError(f"Required parameter {key} not defined")

    @abstractmethod
    def _configure_params(self) -> None:
        """
        Configure preprocessing-dependent parameters.

        Override this method in the child class
        """
        raise NotImplementedError

    def _create_colormap(self) -> None:
        """Create a colormap instance."""
        if self._checked_params.get("grey_background"):
            bad_color = "0.7"
        else:
            bad_color = "1.0"  # white background
        self._checked_params["colormap"] = ColormapFactory(
            bad_color=bad_color, colormap=self._checked_params["colormap"]
        )


class CDIPreprocessingChecker(ConfigChecker):
    """Configure preprocessing-dependent parameters for the 'CDI' case."""

    def _configure_params(self) -> None:
        """Hard-coded processing-dependent parameter configuration."""
        self._checked_params["roi_detector"] = self._create_roi()
        if self._checked_params["photon_filter"] == "loading":
            self._checked_params["loading_threshold"] = self._checked_params[
                "photon_threshold"
            ]
        else:
            self._checked_params["loading_threshold"] = 0

        if self._checked_params["reload_previous"]:
            self._checked_params["comment"] += "_reloaded"
            logger.info(
                "Reloading... update the direct beam position "
                "taking into account preprocessing_binning"
            )
            self._checked_params["direct_beam"] = (
                self._checked_params["direct_beam"][0]
                // self._checked_params["preprocessing_binning"][1],
                self._checked_params["direct_beam"][1]
                // self._checked_params["preprocessing_binning"][2],
            )
        else:
            self._checked_params["preprocessing_binning"] = (1, 1, 1)
            self._checked_params["reload_orthogonal"] = False
        if self._checked_params["reload_orthogonal"]:
            self._checked_params["use_rawdata"] = False

        if self._checked_params["use_rawdata"]:
            self._checked_params["save_dirname"] = "pynxraw"
            self._checked_params["plot_title"] = ["YZ", "XZ", "XY"]
            logger.info("Output will be non orthogonal, in the detector frame")
        else:
            if (
                self._checked_params["reload_orthogonal"]
                and self._checked_params["preprocessing_binning"][0] != 1
            ):
                raise ValueError(
                    "preprocessing_binning along axis 0 should be 1"
                    " when gridding reloaded data (angles won't match)"
                )
            logger.info(
                "use_rawdata=False: defaulting the binning factor "
                "along the stacking dimension to 1"
            )
            # data in the detector frame, one cannot bin the first axis because it is
            # done during interpolation. The vertical axis y being the rotation axis,
            # binning along z downstream and x outboard will be the same
            self._checked_params["phasing_binning"][0] = 1

            self._checked_params["save_dirname"] = "pynx"
            self._checked_params["plot_title"] = ["QzQx", "QyQx", "QyQz"]
            logger.info("Output will be interpolated in the (Qx, Qy, Qz) frame.")
        if (
            self._checked_params["backend"].lower() == "agg"
            and self._checked_params["flag_interact"]
        ):
            raise ValueError(
                "non-interactive backend 'agg' not compatible with the "
                "interactive masking GUI"
            )
        if (
            self._checked_params["flag_interact"]
            or self._checked_params["reload_previous"]
        ):
            self._checked_params["multiprocessing"] = False


class PreprocessingChecker(ConfigChecker):
    """Configure preprocessing-dependent parameters."""

    def _configure_params(self) -> None:
        """Hard-coded processing-dependent parameter configuration."""
        if self._nb_scans is not None and self._nb_scans > 1:
            if self._checked_params["center_fft"] not in [
                "crop_asymmetric_ZYX",
                "pad_Z",
                "pad_asymmetric_ZYX",
                "skip",
            ]:
                self._checked_params["center_fft"] = "skip"
                # avoid croping the detector plane XY while centering the Bragg peak
                # otherwise outputs may have a different size,
                # which will be problematic for combining or comparing them
        else:
            self._checked_params["multiprocessing"] = False

        self._checked_params["roi_detector"] = self._create_roi()

        if self._checked_params["photon_filter"] == "loading":
            self._checked_params["loading_threshold"] = self._checked_params[
                "photon_threshold"
            ]
        else:
            self._checked_params["loading_threshold"] = 0

        if self._checked_params["reload_previous"]:
            self._checked_params["comment"] += "_reloaded"
        else:
            self._checked_params["preprocessing_binning"] = (1, 1, 1)
            self._checked_params["reload_orthogonal"] = False

        if self._checked_params["rocking_angle"] == "energy":
            self._checked_params["use_rawdata"] = False
            # you need to interpolate the data in QxQyQz for energy scans
            logger.info(
                "Energy scan: defaulting use_rawdata to False,"
                " the data will be interpolated using xrayutilities"
            )

        if self._checked_params["reload_orthogonal"]:
            self._checked_params["use_rawdata"] = False

        if self._checked_params["use_rawdata"]:
            self._checked_params["save_dirname"] = "pynxraw"
            logger.info("Output will be non orthogonal, in the detector frame")
        else:
            if self._checked_params["interpolation_method"] not in {
                "xrayutilities",
                "linearization",
            }:
                raise ValueError(
                    "Incorrect value for interp_method,"
                    ' allowed values are "xrayutilities" and "linearization"'
                )
            if self._checked_params["rocking_angle"] == "energy":
                self._checked_params["interpolation_method"] = "xrayutilities"
                logger.info(
                    "Defaulting interp_method to "
                    f"{self._checked_params['interpolation_method'] }"
                )
            if (
                self._checked_params["reload_orthogonal"]
                and self._checked_params["preprocessing_binning"][0] != 1
            ):
                raise ValueError(
                    "preprocessing_binning along axis 0 should be 1"
                    " when gridding reloaded data (angles won't match)"
                )
            self._checked_params["save_dirname"] = "pynx"
            logger.info(
                "Output will be orthogonalized using "
                f"{self._checked_params['interpolation_method']}"
            )

        if self._checked_params["align_q"]:
            if self._checked_params["ref_axis_q"] not in {"x", "y", "z"}:
                raise ValueError("ref_axis_q should be either 'x', 'y' or 'z'")
            if (
                not self._checked_params["use_rawdata"]
                and self._checked_params["interpolation_method"] == "linearization"
            ):
                self._checked_params[
                    "comment"
                ] += f"_align-q-{self._checked_params['ref_axis_q']}"

        if (
            self._checked_params["backend"].lower() == "agg"
            and self._checked_params["flag_interact"]
        ):
            raise ValueError(
                "non-interactive backend 'agg' not compatible with the "
                "interactive masking GUI"
            )
        if (
            self._checked_params["flag_interact"]
            or self._checked_params["reload_previous"]
        ):
            self._checked_params["multiprocessing"] = False
        if self._checked_params["bragg_peak"] is not None:
            self._checked_params["centering_method"]["reciprocal_space"] = "user"


class PostprocessingChecker(ConfigChecker):
    """Configure postprocessing-dependent parameters."""

    def check_config(self) -> Dict[str, Any]:
        """Check if the provided config is consistent."""
        super().check_config()
        if (
            self._checked_params["rocking_angle"] == "energy"
            and self._checked_params["data_frame"] == "detector"
        ):
            raise NotImplementedError(
                "Energy scans must be interpolated during preprocessing."
            )
        return self._checked_params

    def _configure_params(self) -> None:
        """Hard-coded processing-dependent parameter configuration."""
        if self._nb_scans is not None and self._nb_scans > 1:
            self._checked_params["backend"] = "Agg"
            if self._checked_params["multiprocessing"] and any(
                val is None for val in self._checked_params["reconstruction_files"]
            ):
                raise ValueError(
                    "provide a list of files in 'reconstruction_files' "
                    "with multiprocessing ON"
                )
        else:
            self._checked_params["multiprocessing"] = False
        if self._checked_params["simulation"]:
            self._checked_params["invert_phase"] = False
            self._checked_params["correct_refraction"] = False
        if self._checked_params["invert_phase"]:
            self._checked_params["phase_fieldname"] = "disp"
        else:
            self._checked_params["phase_fieldname"] = "phase"

        if self._checked_params["data_frame"] == "detector":
            self._checked_params["is_orthogonal"] = False
        else:
            self._checked_params["is_orthogonal"] = True

        if (
            self._checked_params["data_frame"] == "crystal"
            and self._checked_params["save_frame"] != "crystal"
        ):
            logger.info(
                "data already in the crystal frame before phase retrieval,"
                " it is impossible to come back to the laboratory "
                "frame, parameter 'save_frame' defaulted to 'crystal'"
            )
            self._checked_params["save_frame"] = "crystal"
        self._checked_params["roi_detector"] = self._create_roi()


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
    allowed: Optional[Set] = None

    # convert 'None' to None
    value = util.convert_str_target(value, target="none")

    # convert 'True' to True
    value = util.convert_str_target(value, target="true")

    # convert 'False' to False
    value = util.convert_str_target(value, target="false")

    # test the booleans first
    if key in {
        "align_axis",
        "align_q",
        "apodize",
        "bin_during_loading",
        "correct_curvature",
        "correct_refraction",
        "custom_scan",
        "debug",
        "detector_on_goniometer",
        "fit_datarange",
        "flag_interact",
        "flip_reconstruction",
        "get_temperature",
        "grey_background",
        "invert_phase",
        "is_series",
        "keep_size",
        "mask_beamstop",
        "mask_zero_event",
        "multiprocessing",
        "reload_orthogonal",
        "reload_previous",
        "save",
        "save_as_int",
        "save_rawdata",
        "save_support",
        "save_to_mat",
        "save_to_npz",
        "save_to_vti",
        "skip_unwrap",
        "simulation",
        "use_rawdata",
    }:
        valid.valid_item(value, allowed_types=bool, name=key)
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
            item_types=Real,
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
        allowed_keys = {"direct_space", "reciprocal_space"}
        allowed_values = {"com", "max", "max_com", "skip", "user"}
        if isinstance(value, dict):
            if any(
                subkey not in allowed_keys or val not in allowed_values
                for (subkey, val) in value.items()
            ):
                raise ValueError(
                    f"Invalid value {value} for '{key}'. "
                    f"Allowed keys: {allowed_keys}, allowed values: {allowed_values}"
                )
        elif isinstance(value, str):
            if value not in allowed_values:
                raise ParameterError(key, value, allowed_values)
            value = {"direct_space": value, "reciprocal_space": value}
        else:
            raise TypeError(
                f"'{key}' should be a dictionary or a string, " f"got {type(value)}"
            )
    elif key == "center_roi_x":
        valid.valid_item(value, allowed_types=int, allow_none=True, name=key)
    elif key == "center_roi_y":
        valid.valid_item(value, allowed_types=int, allow_none=True, name=key)
    elif key == "colormap":
        if value not in ["turbo", "custom"] and value not in cc.cm:
            raise ValueError(f"unknow colormap '{value}'")
    elif key == "comment":
        valid.valid_container(value, container_types=str, name=key)
        if value and not value.startswith("_"):
            value = "_" + value
    elif key == "config_file":
        valid.valid_container(value, container_types=str, min_length=1, name=key)
        if not os.path.isfile(value):
            raise ValueError(f"The file '{value}' does not exist")
    elif key == "correlation_threshold":
        valid.valid_item(
            value, allowed_types=Real, min_included=0, max_included=1, name=key
        )
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
    elif key == "custom_pixelsize":
        valid.valid_item(
            value, allowed_types=Real, min_excluded=0, allow_none=True, name=key
        )
    elif key == "data_dir":
        if value is not None:
            if isinstance(value, str):
                value = (value,)
            valid.valid_container(
                value,
                container_types=(tuple, list),
                item_types=str,
                min_length=1,
                allow_none=True,
                name=key,
            )
            for val in value:
                if val is not None and not os.path.isdir(val):
                    raise ValueError(f"The directory {val} does not exist")
    elif key == "data_frame":
        allowed = {"detector", "crystal", "laboratory"}
        if value not in allowed:
            raise ParameterError(key, value, allowed)
    elif key == "detector_distance":
        valid.valid_item(
            value, allowed_types=Real, min_excluded=0, allow_none=True, name=key
        )
    elif key == "dirbeam_detector_angles":
        valid.valid_container(
            value,
            container_types=(list, tuple),
            item_types=Real,
            length=2,
            allow_none=True,
            name=key,
        )
    elif key == "dirbeam_detector_position":
        valid.valid_container(
            value,
            container_types=(list, tuple),
            item_types=Real,
            length=3,
            allow_none=True,
            name=key,
        )
    elif key == "direct_beam":
        valid.valid_container(
            value,
            container_types=(list, tuple),
            item_types=Real,
            length=2,
            allow_none=True,
            name=key,
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
        allowed = {0, 1}
        if value not in allowed:
            raise ParameterError(key, value, allowed)
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
            valid.valid_container(
                value,
                container_types=list,
                item_types=int,
                allow_none=False,
                min_included=0,
                name=key,
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
    elif key == "mask":
        valid.valid_container(value, container_types=str, allow_none=True, name=key)
        if value is not None and not os.path.isfile(value):
            raise ValueError(f"The file '{value}' does not exist")
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
    elif key == "plot_margin":
        valid.valid_item(value, allowed_types=int, min_included=0, name=key)
    elif key == "preprocessing_binning":
        valid.valid_container(
            value,
            container_types=(tuple, list, np.ndarray),
            length=3,
            item_types=int,
            min_excluded=0,
            name=key,
        )
    elif key == "reconstruction_files":
        if isinstance(value, str):
            value = (value,)
        valid.valid_container(
            value,
            container_types=(tuple, list),
            item_types=str,
            min_length=1,
            allow_none=True,
            name=key,
        )
        if value is not None:
            for val in value:
                if val is not None and not os.path.isfile(val):
                    raise ValueError(f"The file '{val}' does not exist")
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
            raise ValueError(f"The directory '{value}' does not exist")
    elif key == "sample_inplane":
        valid.valid_container(
            value,
            container_types=(tuple, list),
            length=3,
            item_types=Real,
            name=key,
        )
    elif key == "sample_name":
        if value is None:
            value = ""
        if isinstance(value, str):
            value = (value,)
        valid.valid_container(
            value,
            container_types=(tuple, list),
            item_types=str,
            name=key,
        )
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
        if isinstance(value, str) or value is None:
            value = [
                value,
            ]
        valid.valid_container(
            value,
            container_types=(tuple, list),
            item_types=str,
            min_length=1,
            allow_none=True,
            name=key,
        )
        value = list(value)
        for idx, val in enumerate(value):
            if isinstance(val, str) and not val.endswith("/"):
                value[idx] += "/"
    elif key == "save_frame":
        allowed = {"laboratory", "crystal", "lab_flat_sample"}
        if value not in allowed:
            raise ParameterError(key, value, allowed)
    elif key == "scans":
        if isinstance(value, Real):
            value = (value,)
        valid.valid_container(
            value, container_types=(tuple, list, np.ndarray), min_length=1, name=key
        )
    elif key == "sort_method":
        allowed = {"mean_amplitude", "variance", "variance/mean", "volume"}
        if value not in allowed:
            raise ParameterError(key, value, allowed)
    elif key == "strain_method":
        allowed = {"default", "defect"}
        if value not in allowed:
            raise ParameterError(key, value, allowed)
    elif key == "strain_range":
        valid.valid_item(value, allowed_types=Real, min_excluded=0, name=key)
    elif key in ["template_imagefile", "specfile_name"]:
        if isinstance(value, str):
            value = (value,)
        valid.valid_container(
            value,
            container_types=(tuple, list),
            item_types=str,
            min_length=1,
            allow_none=True,
            name=key,
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
    else:
        # this key is not in the known parameters
        is_valid = False

    return value, is_valid
