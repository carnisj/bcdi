# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, jerome.carnis@esrf.fr
"""Main runner for BCDI data postprocessing, after phase retrieval."""

import logging
import multiprocessing as mp
from typing import Any, Dict

import numpy as np

import bcdi.utils.utilities as util
from bcdi.postprocessing.process_scan import process_scan
from bcdi.postprocessing.raw_orthogonalization import orthogonalize
from bcdi.utils.parameters import PostprocessingChecker

logger = logging.getLogger(__name__)


def initialize_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Configure and validate the existing dictionary of parameters."""
    return PostprocessingChecker(
        initial_params=parameters,
        default_values={
            "actuators": None,
            "align_axis": False,
            "apodize": False,
            "apodization_alpha": [1.0, 1.0, 1.0],
            "apodization_mu": [0.0, 0.0, 0.0],
            "apodization_sigma": [0.30, 0.30, 0.30],
            "apodization_window": "blackman",
            "averaging_space": "reciprocal_space",
            "backend": "Qt5Agg",
            "background_file": None,
            "beam_direction": [1, 0, 0],
            "bragg_peak": None,
            "centering_method": {
                "direct_space": "max_com",
                "reciprocal_space": "max_com",
            },
            "colormap": "turbo",
            "comment": "",
            "correct_refraction": False,
            "correlation_threshold": 0.90,
            "custom_motors": None,
            "custom_pixelsize": None,
            "custom_scan": False,
            "data_dir": None,
            "debug": False,
            "detector_distance": None,
            "direct_beam": None,
            "dirbeam_detector_angles": None,
            "energy": None,
            "fix_voxel": None,
            "flatfield_file": None,
            "frames_pattern": None,
            "get_temperature": False,
            "half_width_avg_phase": 0,
            "hotpixels_file": None,
            "inplane_angle": None,
            "invert_phase": True,
            "is_series": False,
            "keep_size": False,
            "multiprocessing": True,
            "normalize_flux": "skip",
            "offset_inplane": 0,
            "offset_method": "mean",
            "optical_path_method": "threshold",
            "original_size": None,
            "outofplane_angle": None,
            "phase_offset": 0,
            "phase_offset_origin": None,
            "phase_ramp_removal": "gradient",
            "phase_range": np.pi / 2,
            "phasing_binning": [1, 1, 1],
            "plot_margin": 10,
            "preprocessing_binning": [1, 1, 1],
            "ref_axis_q": "y",
            "reference_spacing": None,
            "reference_temperature": None,
            "roll_modes": [0, 0, 0],
            "sample_inplane": [1, 0, 0],
            "sample_offsets": None,
            "sample_outofplane": [0, 0, 1],
            "save": True,
            "save_dirname": "result",
            "save_rawdata": False,
            "save_support": False,
            "skip_unwrap": False,
            "sort_method": "variance/mean",
            "strain_method": "default",
            "strain_range": 0.002,
            "threshold_gradient": 1.0,
            "threshold_unwrap_refraction": 0.05,
            "tilt_angle": None,
            "tick_direction": "inout",
            "tick_length": 10,
            "tick_spacing": 50,
            "tick_width": 2,
        },
        match_length_params=(
            "data_dir",
            "reconstruction_files",
            "sample_name",
            "save_dir",
            "specfile_name",
            "template_imagefile",
        ),
        required_params=(
            "beamline",
            "data_frame",
            "detector",
            "isosurface_strain",
            "output_size",
            "rocking_angle",
            "root_folder",
            "sample_name",
            "save_frame",
            "scans",
        ),
    ).check_config()


def run(prm: Dict[str, Any], procedure: str = "strain_computation") -> None:
    """
    Run the postprocessing defined by the configuration parameters.

    It assumes that the dictionary of parameters was validated via a ConfigChecker
    instance.

    :param prm: the parsed parameters
    :param procedure: "orthogonalization" to do only the interpolation,
     "strain_computation" to use the full workflow
    """
    if procedure == "strain_computation":
        process = process_scan
    elif procedure == "orthogonalization":
        process = orthogonalize
    else:
        raise NotImplementedError(
            f"procedure {procedure} unknown, should be either "
            "'strain_computation' or  'orthogonalize'"
        )

    prm = initialize_parameters(prm)

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
                process,
                args=(scan_idx, prm),
                callback=util.move_log,
                error_callback=util.catch_error,
            )
        pool.close()
        pool.join()  # postpones the execution of next line of code
        # until all processes in the queue are done.
    else:
        for scan_idx in range(nb_scans):
            result = process(scan_idx=scan_idx, prm=prm)
            util.move_log(result)
