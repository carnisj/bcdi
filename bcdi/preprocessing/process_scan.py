# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, jerome.carnis@esrf.fr
"""
Workflow for BCDI data preprocessing of a single scan, before phase retrieval.

The detector is expected to be on a goniometer.
"""

import logging
import os
import tkinter as tk
from logging import Logger
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from bcdi.experiment.setup import Setup
from bcdi.preprocessing.analysis import create_analysis
from bcdi.utils.snippets_logging import FILE_FORMATTER

logger = logging.getLogger(__name__)


def process_scan(
    scan_idx: int, prm: Dict[str, Any]
) -> Tuple[Path, Path, Optional[Logger]]:
    """
    Run the BCDI preprocessing with the configuration parameters for a single scan.

    This function is meant to be run as a process in multiprocessing, although it can
    also be used as a normal function for a single scan. It assumes that the dictionary
    of parameters was validated via a ConfigChecker instance. Interactive masking and
    reloading of previous masking are not compatible with multiprocessing.

    :param scan_idx: index of the scan to be processed in prm["scans"]
    :param prm: the parsed parameters
    """
    ####################################################
    # Initialize parameters for the callback functions #
    ####################################################
    plt.rcParams["keymap.fullscreen"] = [""]
    plt.rcParams["keymap.quit"] = [
        "ctrl+w",
        "cmd+w",
    ]  # this one to avoid that q closes window (matplotlib default)
    if prm["reload_previous"]:
        root = tk.Tk()
        root.withdraw()
    plt.ion()

    ####################
    # Setup the logger #
    ####################
    scan_nb = prm["scans"][scan_idx]
    matplotlib.use(prm["backend"])

    tmpfile = (
        Path(
            prm["save_dir"][scan_idx]
            if prm["save_dir"][scan_idx] is not None
            else prm["root_folder"]
        )
        / f"preprocessing_run{scan_idx}_{prm['sample_name'][scan_idx]}{scan_nb}.log"
    )
    filehandler = logging.FileHandler(tmpfile, mode="w", encoding="utf-8")
    filehandler.setFormatter(FILE_FORMATTER)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(filehandler)
    if not prm["multiprocessing"] or len(prm["scans"]) == 1:
        logger.propagate = True

    prm["sample"] = f"{prm['sample_name']}+{scan_nb}"
    tmp_str = f"Scan {scan_idx + 1}/{len(prm['scans'])}: S{scan_nb}"
    from datetime import datetime

    logger.info(f"Start {process_scan.__name__} at {datetime.now()}")
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
    comment = analysis.comment

    #############
    # Load data #
    #############
    logger.info(f"Input data shape: {np.shape(analysis.data)}")

    binning_comment = (
        f"_{setup.detector.preprocessing_binning[0] * setup.detector.binning[0]}"
        f"_{setup.detector.preprocessing_binning[1] * setup.detector.binning[1]}"
        f"_{setup.detector.preprocessing_binning[2] * setup.detector.binning[2]}"
    )

    ##############################################################
    # correct detector angles and save values for postprocessing #
    ##############################################################
    if analysis.detector_angles_correction_needed:
        logger.info("Trying to correct detector angles using the direct beam")

        analysis.retrieve_bragg_peak()

        analysis.update_detector_angles(bragg_peak_position=prm["bragg_peak"])
        analysis.update_parameters(
            {
                "inplane_angle": setup.inplane_angle,
                "outofplane_angle": setup.outofplane_angle,
            }
        )

    ##############################################################
    # optional interpolation of the data onto an orthogonal grid #
    ##############################################################
    if analysis.is_raw_data_available and prm["save_rawdata"]:
        analysis.save_data(
            filename=setup.detector.savedir
            + f"S{scan_nb}"
            + "_data_before_masking_stack",
        )

    if analysis.interpolation_needed:
        data_shape = analysis.data.shape
        analysis.show_masked_data(
            title="Data before gridding\n",
            filename=setup.detector.savedir + f"data_before_gridding_S{scan_nb}"
            f"_{data_shape[0]}_{data_shape[1]}_{data_shape[2]}"
            + binning_comment
            + ".png",
        )
        analysis.interpolate_data()

    analysis.calculate_q_bragg()

    if analysis.q_bragg is not None:
        # expressed in the laboratory frame z downstream, y vertical, x outboard
        logger.info(
            f"Wavevector transfer of Bragg peak: {analysis.q_bragg}, "
            f"Qnorm={analysis.q_norm:.4f}"
        )
        logger.info(f"Interplanar distance: {analysis.planar_distance:.6f} angstroms")

    ########################
    # crop/pad/center data #
    ########################
    analysis.apply_mask_to_data()
    analysis.center_fft()

    #########################################################
    # optional masking of points when there is no intensity #
    # along the whole rocking curve (probably dead pixels)  #
    #########################################################
    if prm["mask_zero_event"]:
        analysis.mask_zero_events()

    #####################################
    # save figures before alien removal #
    #####################################
    nz, ny, nx = analysis.data.shape
    analysis.show_masked_data(
        title="Data before aliens removal\n",
        filename=setup.detector.savedir
        + f"data_before_masking_sum_S{scan_nb}_{nz}_{ny}_{nx}_"
        f"{setup.detector.binning[0]}_"
        f"{setup.detector.binning[1]}_{setup.detector.binning[2]}.png",
    )

    analysis.show_masked_data_at_max(
        title="data at max\n",
        filename=setup.detector.savedir
        + f"data_before_masking_S{scan_nb}_{nz}_{ny}_{nx}_"
        f"{setup.detector.binning[0]}"
        f"_{setup.detector.binning[1]}_{setup.detector.binning[2]}.png",
    )

    analysis.show_mask(
        title="Mask before aliens removal\n",
        filename=setup.detector.savedir
        + f"mask_before_masking_S{scan_nb}_{nz}_{ny}_{nx}_"
        f"{setup.detector.binning[0]}"
        f"_{setup.detector.binning[1]}_{setup.detector.binning[2]}.png",
        vmax=(nz, ny, nx),
    )

    if analysis.is_orthogonal and analysis.data_loader.q_values is not None:
        analysis.save_to_vti(
            filename=os.path.join(
                setup.detector.savedir,
                f"S{scan_nb}_ortho_int" + comment.text + ".vti",
            )
        )

    ########################################################
    # load an optional mask from the config and combine it #
    ########################################################
    mask_file = prm.get("mask")
    if isinstance(mask_file, str):
        analysis.update_mask(mask_file)

    if prm["flag_interact"]:
        interactive_mask = analysis.get_interactive_masker()
        ##################
        # aliens removal #
        ##################
        interactive_mask.interactive_masking_aliens()

        analysis.show_masked_data(
            title="Data after aliens removal\n",
            filename=setup.detector.savedir
            + f"data_after_aliens_removal_S{scan_nb}_{nz}_{ny}_{nx}_"
            f"{setup.detector.binning[0]}"
            f"_{setup.detector.binning[1]}_{setup.detector.binning[2]}.png",
        )

        analysis.show_mask(
            title="Mask after aliens removal\n",
            filename=setup.detector.savedir
            + f"mask_after_aliens_removal_S{scan_nb}_{nz}_{ny}_{nx}_"
            f"{setup.detector.binning[0]}"
            f"_{setup.detector.binning[1]}_{setup.detector.binning[2]}.png",
            vmax=(nz, ny, nx),
        )

        interactive_mask.refine_mask()  # (remove remaining hotpixels, ...)
        if interactive_mask.mask is None:
            raise ValueError("mask is undefined")
        analysis.mask = np.copy(interactive_mask.mask)
        del interactive_mask

    analysis.set_binary_mask()
    analysis.apply_mask_to_data()

    #############################################
    # save the interactive mask for later reuse #
    #############################################
    analysis.save_hotpixels(
        filename=setup.detector.savedir + f"S{scan_nb}_interactive_mask"
    )

    ###############################################
    # mask or median filter isolated empty pixels #
    ###############################################
    if analysis.is_filtering_needed:
        analysis.filter_data()
    else:
        logger.info("Skipping median filtering")

    analysis.apply_photon_threshold()
    analysis.remove_nan()

    ####################
    # debugging plots  #
    ####################
    analysis.show_masked_data_at_com(
        title="Masked data\n",
        filename=setup.detector.savedir
        + f"middle_frame_S{scan_nb}_{nz}_{ny}_{nx}_{setup.detector.binning[0]}_"
        f"{setup.detector.binning[1]}_{setup.detector.binning[2]}"
        + comment.text
        + ".png",
    )
    analysis.show_masked_data(
        title="Masked data\n",
        filename=setup.detector.savedir
        + f"sum_S{scan_nb}_{nz}_{ny}_{nx}_{setup.detector.binning[0]}_"
        f"{setup.detector.binning[1]}_{setup.detector.binning[2]}"
        + comment.text
        + ".png",
    )
    analysis.show_mask(
        title="Mask\n",
        filename=setup.detector.savedir + f"mask_S{scan_nb}_{nz}_{ny}_{nx}_"
        f"{setup.detector.binning[0]}_{setup.detector.binning[1]}_"
        f"{setup.detector.binning[2]}" + comment.text + ".png",
        vmax=(nz, ny, nx),
    )

    ##################################################
    # bin the stacking axis if needed, the detector  #
    # plane was already binned when loading the data #
    ##################################################
    if analysis.is_binning_rocking_axis_needed:
        analysis.bin_rocking_axis()
    analysis.check_binning()

    ##################################################################
    # final check of the shape to comply with FFT shape requirements #
    ##################################################################
    analysis.crop_to_fft_compliant_shape()

    nz, ny, nx = analysis.data.shape
    comment.concatenate(f"{nz}_{ny}_{nx}" + binning_comment)

    ############################
    # save final data and mask #
    ############################
    logger.info(f"Saving directory: {setup.detector.savedir}")
    if prm["save_as_int"]:
        analysis.cast_data_to_int()
    logger.info(f"Data type before saving: {analysis.data.dtype}")
    logger.info(f"Mask type before saving: {analysis.mask.dtype}")

    if analysis.is_orthogonal and analysis.data_loader.q_values is not None:
        analysis.save_q_values(
            filename=setup.detector.savedir + f"QxQzQy_S{scan_nb}" + comment.text
        )
        analysis.contour_data(
            title="Final data\n",
            filename=setup.detector.savedir
            + f"final_reciprocal_space_S{scan_nb}"
            + comment.text
            + ".png",
        )

    analysis.save_data(
        filename=setup.detector.savedir + f"S{scan_nb}_pynx" + comment.text
    )
    analysis.save_mask(
        filename=setup.detector.savedir + f"S{scan_nb}_maskpynx" + comment.text
    )
    analysis.save_results_as_h5(
        filename=f"{setup.detector.savedir}S{scan_nb}_preprocessing{comment.text}.h5"
    )

    ############################
    # plot final data and mask #
    ############################
    analysis.apply_mask_to_data()
    analysis.show_masked_data(
        title="Final data\n",
        filename=setup.detector.savedir
        + f"finalsum_S{scan_nb}"
        + comment.text
        + ".png",
    )
    analysis.show_mask(
        title="Final mask\n",
        filename=setup.detector.savedir
        + f"finalmask_S{scan_nb}"
        + comment.text
        + ".png",
        vmax=(nz, ny, nx),
    )

    if len(prm["scans"]) > 1:
        plt.close("all")

    logger.removeHandler(filehandler)
    filehandler.close()

    return tmpfile, Path(setup.detector.savedir), logger
