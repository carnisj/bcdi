# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, jerome.carnis@esrf.fr
#         Clement Atlan, c.atlan@outlook.com

"""Workflow for BCDI data orthogonalization of a single scan, after phase retrieval."""
import logging
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np
import yaml
from matplotlib import pyplot as plt

import bcdi.utils.utilities as util
from bcdi.constants import AXIS_TO_ARRAY
from bcdi.experiment.setup import Setup
from bcdi.postprocessing.analysis import create_analysis
from bcdi.utils.snippets_logging import FILE_FORMATTER

logger = logging.getLogger(__name__)


def orthogonalize(
    scan_idx: int, prm: Dict[str, Any]
) -> Tuple[Path, Path, Optional[Logger]]:
    """
    Run the orthogonalization defined by the configuration parameters for a single scan.

    This function is meant to be run as a process in multiprocessing, although it can
    also be used as a normal function for a single scan. It assumes that the dictionary
    of parameters was validated via a ConfigChecker instance.

    :param scan_idx: index of the scan to be processed in prm["scans"]
    :param prm: the parsed parameters
    """
    scan_nb = prm["scans"][scan_idx]
    tmpfile = Path(
        prm["save_dir"][scan_idx]
        if prm["save_dir"][scan_idx] is not None
        else prm["root_folder"]
    ) / (f"interpolation_run{scan_idx}_{prm['sample_name'][scan_idx]}" f"{scan_nb}.log")

    filehandler = logging.FileHandler(tmpfile, mode="w", encoding="utf-8")
    filehandler.setFormatter(FILE_FORMATTER)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(filehandler)
    if not prm["multiprocessing"] or len(prm["scans"]) == 1:
        logger.propagate = True

    # prm["sample"] = f"{prm['sample_name']}+{scan_nb}"
    tmp_str = f"Scan {scan_idx + 1}/{len(prm['scans'])}: S{scan_nb}"
    logger.info(f"Start {orthogonalize.__name__} at {datetime.now()}")
    logger.info(f'\n{"#" * len(tmp_str)}\n' + tmp_str + "\n" + f'{"#" * len(tmp_str)}')

    #################################
    # define the experimental setup #
    #################################
    setup = Setup(
        parameters=prm,
        scan_index=scan_idx,
        logger=logger,
    )

    logger.info(f"##############\nSetup instance\n##############\n{setup.params}")
    logger.info(
        "#################\nDetector instance\n#################\n"
        f"{setup.detector.params}"
    )

    ######################
    # start the analysis #
    ######################
    logger.info("###############\nProcessing data\n###############")

    analysis = create_analysis(
        scan_index=scan_idx, parameters=prm, setup=setup, logger=logger
    )

    ##########################################################
    # correct the detector angles for the direct beam offset #
    ##########################################################
    if analysis.detector_angles_correction_needed:
        logger.info("Trying to correct detector angles using the direct beam")

        if analysis.undefined_bragg_peak_but_retrievable:
            metadata = analysis.retrieve_bragg_peak()
            analysis.update_parameters({"bragg_peak": metadata["bragg_peak"]})

        analysis.update_detector_angles(bragg_peak_position=prm["bragg_peak"])

    #########################################################
    # calculate q of the Bragg peak in the laboratory frame #
    #########################################################
    q_lab = analysis.get_normalized_q_bragg_laboratory_frame
    if q_lab is None:
        raise ValueError("q_lab is None")
    logger.info(
        "Normalized diffusion vector in the laboratory frame (z*, y*, x*): "
        f"{[f'{val:.4f}' for _, val in enumerate(q_lab)]} (1/A)"
    )
    logger.info(f"Wavevector transfer: {analysis.get_norm_q_bragg:.4f} 1/A")
    logger.info(f"Atomic planar distance: {analysis.get_interplanar_distance:.4f} nm")

    #######################
    #  orthogonalize data #
    #######################
    logger.info(f"Shape before interpolation {analysis.data.shape}")
    analysis.interpolate_into_crystal_frame()

    if analysis.voxel_sizes is None:
        raise ValueError("voxel sizes undefined")
    logger.info(f"Voxel size: {analysis.voxel_sizes}")

    ######################################################
    # center the object (centering based on the modulus) #
    ######################################################
    logger.info("Centering the crystal")
    analysis.center_object_based_on_modulus(centering_method="com")

    #######################################
    # optionally rotates back the crystal #
    # into the laboratory frame           #
    #######################################
    if analysis.get_normalized_q_bragg_laboratory_frame is None:
        raise ValueError("analysis.get_normalized_q_bragg_laboratory_frame is None")
    if prm["save_frame"] in ["laboratory", "lab_flat_sample"]:
        amplitude, phase = util.rotate_crystal(
            arrays=(np.abs(analysis.data), np.angle(analysis.data)),
            axis_to_align=AXIS_TO_ARRAY[prm["ref_axis_q"]],
            voxel_size=analysis.voxel_sizes,
            is_orthogonal=prm["is_orthogonal"],
            reciprocal_space=False,
            reference_axis=analysis.get_normalized_q_bragg_laboratory_frame[::-1],
            debugging=(False, False),
        )
        q_bragg_in_saving_frame = q_lab

        if prm["save_frame"] == "lab_flat_sample":
            if setup.q_laboratory is None:
                raise ValueError("setup.q_laboratory is None")
            (amplitude, phase), q_bragg_in_saving_frame = setup.beamline.flatten_sample(
                arrays=(amplitude, phase),
                voxel_size=analysis.voxel_sizes,
                q_bragg=setup.q_laboratory / float(np.linalg.norm(setup.q_laboratory)),
                is_orthogonal=prm["is_orthogonal"],
                reciprocal_space=False,
                rocking_angle=setup.rocking_angle,
                debugging=(False, False),
            )

        complex_object = amplitude * np.exp(1j * phase)

    else:  # crystal frame
        complex_object = analysis.data
        q_bragg_in_saving_frame = np.asarray(
            util.rotate_vector(
                vectors=analysis.get_normalized_q_bragg_laboratory_frame[::-1],
                axis_to_align=AXIS_TO_ARRAY[prm["ref_axis_q"]],
                reference_axis=analysis.get_normalized_q_bragg_laboratory_frame[::-1],
            )
        )

    # rescale q (it is normalized)
    if analysis.get_interplanar_distance is None:
        raise ValueError("analysis.get_interplanar_distanc is None")
    q_bragg_in_saving_frame *= 2 * np.pi / (10 * analysis.get_interplanar_distance)

    # Save the complex object in the desired output file
    output_file_path_template = (
        f"{prm['save_dir'][scan_idx]}/S{scan_nb}"
        f"_orthogonolized_reconstruction_{prm['save_frame']}"
    )
    np.savez(
        f"{output_file_path_template}.npz",
        data=complex_object,
        voxel_sizes=analysis.voxel_sizes,
        q_bragg=q_bragg_in_saving_frame,
        detector=str(yaml.dump(setup.detector.params)),
        setup=str(yaml.dump(setup.params)),
        params=str(yaml.dump(prm)),
    )

    with h5py.File(f"{output_file_path_template}.h5", "w") as hf:
        output = hf.create_group("output")
        parameters = hf.create_group("parameters")
        output.create_dataset("data", data=complex_object)
        output.create_dataset("q_bragg", data=q_bragg_in_saving_frame)
        output.create_dataset("voxel_sizes", data=analysis.voxel_sizes)
        parameters.create_dataset("detector", data=str(setup.detector.params))
        parameters.create_dataset("setup", data=str(setup.params))
        parameters.create_dataset("parameters", data=str(prm))

    if len(prm["scans"]) > 1:
        plt.close("all")

    logger.removeHandler(filehandler)
    filehandler.close()

    return tmpfile, Path(setup.detector.savedir), logger
