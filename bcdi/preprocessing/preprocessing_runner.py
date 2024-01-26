# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, jerome.carnis@esrf.fr
"""Main runner for BCDI data preprocessing, before phase retrieval."""

import logging
import multiprocessing as mp
from typing import Any, Dict

import bcdi.utils.utilities as util
from bcdi.preprocessing.process_scan import process_scan
from bcdi.preprocessing.process_scan_cdi import process_scan_cdi
from bcdi.utils.parameters import CDIPreprocessingChecker, PreprocessingChecker

logger = logging.getLogger(__name__)


def initialize_parameters_bcdi(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Configure and validate the existing dictionary of parameters for BCDI."""
    return PreprocessingChecker(
        initial_params=parameters,
        default_values={
            "actuators": None,
            "align_q": True,
            "backend": "Qt5Agg",
            "background_file": None,
            "background_plot": 0.5,
            "beam_direction": [1, 0, 0],
            "bin_during_loading": False,
            "bragg_peak": None,
            "center_fft": "skip",
            "centering_method": "max_com",
            "colormap": "turbo",
            "comment": "",
            "custom_monitor": None,
            "custom_motors": None,
            "custom_images": None,
            "custom_scan": False,
            "data_dir": None,
            "debug": False,
            "detector_distance": None,
            "direct_beam": None,
            "dirbeam_detector_angles": None,
            "energy": None,
            "fill_value_mask": 0,
            "flag_interact": True,
            "flatfield_file": None,
            "frames_pattern": None,
            "hotpixels_file": None,
            "inplane_angle": None,
            "interpolation_method": "linearization",
            "is_series": False,
            "linearity_func": None,
            "mask_zero_event": False,
            "median_filter": "skip",
            "median_filter_order": 7,
            "multiprocessing": True,
            "normalize_flux": False,
            "offset_inplane": 0,
            "outofplane_angle": None,
            "pad_size": None,
            "photon_filter": "loading",
            "photon_threshold": 0,
            "preprocessing_binning": [1, 1, 1],
            "ref_axis_q": "y",
            "reload_orthogonal": False,
            "reload_previous": False,
            "sample_inplane": [1, 0, 0],
            "sample_offsets": None,
            "sample_outofplane": [0, 0, 1],
            "save_as_int": False,
            "save_rawdata": False,
            "save_to_mat": False,
            "save_to_npz": True,
            "save_to_vti": False,
        },
        match_length_params=(
            "data_dir",
            "sample_name",
            "save_dir",
            "specfile_name",
            "template_imagefile",
        ),
        required_params=(
            "beamline",
            "detector",
            "phasing_binning",
            "rocking_angle",
            "root_folder",
            "sample_name",
            "scans",
            "use_rawdata",
        ),
    ).check_config()


def initialize_parameters_cdi(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Configure and validate the existing dictionary of parameters for CDI."""
    return CDIPreprocessingChecker(
        initial_params=parameters,
        default_values={
            "actuators": None,
            "backend": "Qt5Agg",
            "background_file": None,
            "background_plot": 0.5,
            "beam_direction": [1, 0, 0],
            "bin_during_loading": False,
            "centering_method": "max_com",
            "colormap": "turbo",
            "correct_curvature": True,
            "comment": "",
            "custom_monitor": None,
            "custom_motors": None,
            "custom_images": None,
            "custom_scan": False,
            "data_dir": None,
            "debug": False,
            "detector_distance": None,
            "energy": None,
            "fill_value_mask": 0,
            "fit_datarange": False,
            "flag_interact": True,
            "flatfield_file": None,
            "frames_pattern": None,
            "hotpixels_file": None,
            "is_series": False,
            "linearity_func": None,
            "mask_beamstop": False,
            "mask_zero_event": False,
            "median_filter": "skip",
            "median_filter_order": 7,
            "multiprocessing": True,
            "normalize_flux": False,
            "photon_filter": "loading",
            "photon_threshold": 0,
            "preprocessing_binning": [1, 1, 1],
            "reload_orthogonal": False,
            "reload_previous": False,
            "sample_offsets": None,
            "save_as_int": False,
            "save_rawdata": False,
            "save_to_mat": False,
            "save_to_npz": True,
            "save_to_vti": False,
        },
        match_length_params=(
            "data_dir",
            "sample_name",
            "save_dir",
            "specfile_name",
            "template_imagefile",
        ),
        required_params=(
            "beamline",
            "detector",
            "dirbeam_detector_position",
            "direct_beam",
            "phasing_binning",
            "root_folder",
            "sample_name",
            "scans",
            "use_rawdata",
        ),
    ).check_config()


def run(prm: Dict[str, Any]) -> None:
    """
    Run the preprocessing.

    :param prm: the parsed parameters
    """
    if prm.get("detector_on_goniometer") is None or prm["detector_on_goniometer"]:
        func = process_scan
        prm = initialize_parameters_bcdi(prm)
    else:
        func = process_scan_cdi
        prm = initialize_parameters_cdi(prm)

    ############################
    # start looping over scans #
    ############################
    nb_scans = len(prm["scans"])
    if prm["multiprocessing"]:
        mp.freeze_support()
        pool = mp.Pool(
            processes=min(mp.cpu_count(), nb_scans)
        )  # use this number of processes

        for scan_idx, scan_nb in enumerate(prm["scans"]):
            tmp_str = (
                f"Scan {scan_idx + 1}/{len(prm['scans'])}: "
                f"{prm['sample_name'][scan_idx]}{scan_nb}"
            )
            logger.info(
                f'\n{"#" * len(tmp_str)}\n' + tmp_str + "\n" + f'{"#" * len(tmp_str)}'
            )
            pool.apply_async(
                func,
                args=(scan_idx, prm),
                callback=util.move_log,
                error_callback=util.catch_error,
            )
        pool.close()
        pool.join()  # postpones the execution of next line of code
        # until all processes in the queue are done.
    else:
        for scan_idx in range(nb_scans):
            result = func(scan_idx=scan_idx, prm=prm)
            util.move_log(result)
