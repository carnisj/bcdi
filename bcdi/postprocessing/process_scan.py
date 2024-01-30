# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, jerome.carnis@esrf.fr
"""Workflow for BCDI data postprocessing of a single scan, after phase retrieval."""

import logging
import os
from logging import Logger
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

import bcdi.graph.graph_utils as gu
import bcdi.postprocessing.postprocessing_utils as pu
from bcdi.constants import AXIS_TO_ARRAY
from bcdi.experiment.setup import Setup
from bcdi.postprocessing.analysis import create_analysis
from bcdi.utils.snippets_logging import FILE_FORMATTER

logger = logging.getLogger(__name__)


def process_scan(
    scan_idx: int, prm: Dict[str, Any]
) -> Tuple[Path, Path, Optional[Logger]]:
    """
    Run the postprocessing defined by the configuration parameters for a single scan.

    This function is meant to be run as a process in multiprocessing, although it can
    also be used as a normal function for a single scan. It assumes that the dictionary
    of parameters was validated via a ConfigChecker instance.

    :param scan_idx: index of the scan to be processed in prm["scans"]
    :param prm: the parsed parameters
    """
    scan_nb = prm["scans"][scan_idx]
    matplotlib.use(prm["backend"])

    tmpfile = (
        Path(
            prm["save_dir"][scan_idx]
            if prm["save_dir"][scan_idx] is not None
            else prm["root_folder"]
        )
        / f"postprocessing_run{scan_idx}_{prm['sample_name'][scan_idx]}{scan_nb}.log"
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

    analysis.find_data_range(amplitude_threshold=0.05, plot_margin=prm["plot_margin"])

    analysis.find_best_reconstruction()

    analysis.average_reconstructions()

    phase_manipulator = analysis.get_phase_manipulator()

    if not prm["skip_unwrap"]:
        phase_manipulator.unwrap_phase()
        phase_manipulator.center_phase()
        if prm["debug"]:
            phase_manipulator.plot_phase(
                plot_title="Phase after unwrap + wrap", save_plot=True
            )

    phase_manipulator.remove_ramp()
    if prm["debug"]:
        phase_manipulator.plot_phase(
            plot_title="Phase after ramp removal", save_plot=True
        )

    phase_manipulator.remove_offset()
    phase_manipulator.center_phase()

    #################################################################
    # average the phase over a window to reduce noise in the strain #
    #################################################################
    if prm["half_width_avg_phase"] != 0:
        phase_manipulator.average_phase()
        comment.concatenate("avg" + str(2 * prm["half_width_avg_phase"] + 1))

    #############################################################
    # put back the phase ramp otherwise the diffraction pattern #
    # will be shifted and the prtf messed up                    #
    #############################################################
    phase_manipulator.add_ramp()

    if prm["apodize"]:
        phase_manipulator.apodize()
        comment.concatenate("apodize_" + prm["apodization_window"])

    np.savez_compressed(
        setup.detector.savedir + "S" + str(scan_nb) + "_avg_obj_prtf" + comment.text,
        obj=phase_manipulator.modulus * np.exp(1j * phase_manipulator.phase),
    )

    ####################################################
    # remove again phase ramp before orthogonalization #
    ####################################################
    phase_manipulator.add_ramp(sign=-1)

    analysis.update_data(
        modulus=phase_manipulator.modulus, phase=phase_manipulator.phase
    )
    analysis.extent_phase = phase_manipulator.extent_phase
    del phase_manipulator

    analysis.center_object_based_on_modulus()

    if prm["save_support"]:
        # to be used as starting support in phasing (still in the detector frame)
        # low threshold 0.1 because support will be cropped by shrinkwrap during phasing
        analysis.save_support(
            filename=setup.detector.savedir + f"S{scan_nb}_support{comment.text}.npz",
            modulus_threshold=0.1,
        )

    if prm["save_rawdata"]:
        analysis.save_modulus_phase(
            filename=setup.detector.savedir
            + f"S{scan_nb}_raw_amp-phase{comment.text}.npz",
        )
        analysis.save_to_vti(
            filename=os.path.join(
                setup.detector.savedir,
                "S" + str(scan_nb) + "_raw_amp-phase" + comment.text + ".vti",
            )
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

    #######################
    #  orthogonalize data #
    #######################
    logger.info(f"Shape before interpolation {analysis.data.shape}")
    analysis.interpolate_into_crystal_frame()

    ######################################################
    # center the object (centering based on the modulus) #
    ######################################################
    logger.info("Centering the crystal")
    analysis.center_object_based_on_modulus(centering_method="com")

    ####################
    # Phase unwrapping #
    ####################
    phase_manipulator = analysis.get_phase_manipulator()

    if not prm["skip_unwrap"]:
        logger.info("Phase unwrapping")
        phase_manipulator.unwrap_phase()

    #############################################
    # invert phase: -1*phase = displacement * q #
    #############################################
    if prm["invert_phase"]:
        phase_manipulator.invert_phase()

    #########################
    # refraction correction #
    #########################
    if prm["correct_refraction"]:
        phase_manipulator.compensate_refraction(analysis.get_optical_path())

    ##############################################
    # phase ramp and offset removal (mean value) #
    ##############################################
    phase_manipulator.remove_ramp()
    phase_manipulator.remove_offset()

    # Wrap the phase around 0 (no more offset)
    phase_manipulator.center_phase()

    ################################################################
    # calculate the strain depending on which axis q is aligned on #
    ################################################################
    interpolated_crystal = phase_manipulator.calculate_strain(
        planar_distance=analysis.get_interplanar_distance,
        voxel_sizes=analysis.voxel_sizes,
    )
    del phase_manipulator

    ################################################
    # optionally rotates back the crystal into the #
    # laboratory frame (for debugging purpose)     #
    ################################################
    if prm["save_frame"] in ["laboratory", "lab_flat_sample"]:
        if analysis.get_normalized_q_bragg_laboratory_frame is None:
            raise ValueError("analysis.get_normalized_q_bragg_laboratory_frame is None")
        comment.concatenate("labframe")
        logger.info("Rotating back the crystal in laboratory frame")
        interpolated_crystal.rotate_crystal(
            axis_to_align=AXIS_TO_ARRAY[prm["ref_axis_q"]],
            reference_axis=analysis.get_normalized_q_bragg_laboratory_frame[::-1],
        )
        # q_lab is already in the laboratory frame
        interpolated_crystal.q_bragg_in_saving_frame = (
            analysis.get_normalized_q_bragg_laboratory_frame[::-1]
        )

        if prm["save_frame"] == "lab_flat_sample":
            comment.concatenate("flat")
            interpolated_crystal.flatten_sample_circles(setup=setup)
    else:  # "save_frame" = "crystal"
        # rotate also q_lab to have it along ref_axis_q,
        # as a cross-checkm, vectors needs to be in xyz order
        comment.concatenate("crystalframe")
        interpolated_crystal.set_q_bragg_to_saving_frame(analysis)

    ###############################################
    # rotates the crystal e.g. for easier slicing #
    # of the result along a particular direction  #
    ###############################################
    # typically this is an inplane rotation, q should stay aligned with the axis
    # along which the strain was calculated
    if prm["align_axis"]:
        logger.info("Rotating arrays for visualization")
        interpolated_crystal.rotate_crystal(
            axis_to_align=prm["axis_to_align"],
            reference_axis=AXIS_TO_ARRAY[prm["ref_axis"]],
        )
        # rotate q accordingly, vectors needs to be in xyz order
        interpolated_crystal.rotate_vector_to_saving_frame(
            vector=interpolated_crystal.q_bragg_in_saving_frame[::-1],
            axis_to_align=AXIS_TO_ARRAY[prm["ref_axis"]],
            reference_axis=prm["axis_to_align"],
        )

    q_final = interpolated_crystal.q_bragg_in_saving_frame
    logger.info(
        f"q_final = ({q_final[0]:.4f} 1/A,"
        f" {q_final[1]:.4f} 1/A, {q_final[2]:.4f} 1/A)"
    )

    ##############################################
    # pad array to fit the output_size parameter #
    ##############################################
    interpolated_crystal.crop_pad_arrays()
    logger.info(f"Final data shape: {interpolated_crystal.strain.shape}")

    ######################
    # save result to vtk #
    ######################
    voxel_sizes = analysis.voxel_sizes
    if voxel_sizes is None:
        raise ValueError("voxel sizes are undefined")
    voxel_sizes_text = (
        f"Voxel sizes: ({voxel_sizes[0]:.2f} nm, {voxel_sizes[1]:.2f} nm,"
        f" {voxel_sizes[2]:.2f} nm)"
    )
    logger.info(voxel_sizes_text)

    if prm["save"]:
        interpolated_crystal.save_results_as_npz(
            filename=f"{setup.detector.savedir}S{scan_nb}_"
            f"amp{prm['phase_fieldname']}strain{comment.text}",
            setup=setup,
        )

        interpolated_crystal.save_results_as_h5(
            filename=f"{setup.detector.savedir}S{scan_nb}_"
            f"amp{prm['phase_fieldname']}strain{comment.text}.h5",
            setup=setup,
        )

        # save amp & phase to VTK
        # in VTK, x is downstream, y vertical, z inboard,
        # thus need to flip the last axis
        interpolated_crystal.save_results_as_vti(
            filename=f"{setup.detector.savedir}S{scan_nb}_"
            f"amp-{prm['phase_fieldname']}-strain{comment.text}.vti"
        )

    ######################################
    # estimate the volume of the crystal #
    ######################################
    interpolated_crystal.normalize_modulus()
    interpolated_crystal.estimate_crystal_volume()

    ################################
    # plot linecuts of the results #
    ################################
    interpolated_crystal.fit_linecuts_through_crystal_edges(
        filename=setup.detector.savedir + "linecut_amp.png"
    )

    #############################################
    # prepare the phase and strain for plotting #
    #############################################
    pixel_spacing = [prm["tick_spacing"] / vox for vox in voxel_sizes]
    logger.info(
        "Phase extent with thresholding the modulus "
        f"(threshold={prm['isosurface_strain']}): "
        f"{interpolated_crystal.find_phase_extent_within_crystal():.2f} rad"
    )

    interpolated_crystal.find_max_phase(
        filename=f"{setup.detector.savedir}S{scan_nb}"
        f"_phase_at_max{comment.text}.png"
    )
    interpolated_crystal.threshold_phase_strain()

    ##############################
    # plot slices of the results #
    ##############################
    volume = interpolated_crystal.estimated_crystal_volume
    planar_dist = interpolated_crystal.planar_distance
    nb_phasing = analysis.nb_reconstructions
    modulus = interpolated_crystal.modulus
    # amplitude
    fig, _, _ = gu.multislices_plot(
        modulus,
        sum_frames=False,
        title="Normalized orthogonal amp",
        vmin=0,
        vmax=1,
        tick_direction=prm["tick_direction"],
        tick_width=prm["tick_width"],
        tick_length=prm["tick_length"],
        pixel_spacing=pixel_spacing,
        plot_colorbar=True,
        is_orthogonal=True,
        reciprocal_space=False,
        cmap=prm["colormap"].cmap,
    )
    fig.text(0.60, 0.45, f"Scan {scan_nb}", size=20)
    fig.text(0.60, 0.40, voxel_sizes_text, size=20)
    fig.text(0.60, 0.35, f"Ticks spacing={prm['tick_spacing']} nm", size=20)
    fig.text(0.60, 0.30, f"Volume={int(volume)} nm3", size=20)
    fig.text(0.60, 0.25, "Sorted by " + prm["sort_method"], size=20)
    fig.text(
        0.60, 0.20, f"correlation threshold={prm['correlation_threshold']}", size=20
    )
    fig.text(0.60, 0.15, f"average over {nb_phasing} reconstruction(s)", size=20)
    fig.text(0.60, 0.10, f"Planar distance={planar_dist:.5f} nm", size=20)
    if prm["get_temperature"]:
        temperature = pu.bragg_temperature(
            spacing=planar_dist * 10,
            reflection=prm["reflection"],
            spacing_ref=prm["reference_spacing"],
            temperature_ref=prm["reference_temperature"],
            use_q=False,
            material="Pt",
        )
        fig.text(0.60, 0.05, f"Estimated T={temperature} C", size=20)
    if prm["save"]:
        fig.savefig(setup.detector.savedir + f"S{scan_nb}_amp" + comment.text + ".png")

    # amplitude histogram
    fig, ax = plt.subplots(1, 1)
    ax.hist(modulus[modulus > 0.05 * modulus.max()].flatten(), bins=250)
    ax.set_ylim(bottom=1)
    ax.tick_params(
        labelbottom=True,
        labelleft=True,
        direction="out",
        length=prm["tick_length"],
        width=prm["tick_width"],
    )
    ax.spines["right"].set_linewidth(1.5)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["top"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    fig.savefig(
        setup.detector.savedir + f"S{scan_nb}_histo_amp" + comment.text + ".png"
    )

    # phase
    fig, _, _ = gu.multislices_plot(
        interpolated_crystal.phase,
        sum_frames=False,
        title="Orthogonal displacement",
        vmin=-prm["phase_range"],
        vmax=prm["phase_range"],
        tick_direction=prm["tick_direction"],
        cmap=prm["colormap"].cmap,
        tick_width=prm["tick_width"],
        tick_length=prm["tick_length"],
        pixel_spacing=pixel_spacing,
        plot_colorbar=True,
        is_orthogonal=True,
        reciprocal_space=False,
    )
    fig.text(0.60, 0.30, f"Scan {scan_nb}", size=20)
    fig.text(
        0.60,
        0.25,
        f"Voxel size=({voxel_sizes[0]:.1f}, {voxel_sizes[1]:.1f}, "
        f"{voxel_sizes[2]:.1f}) (nm)",
        size=20,
    )
    fig.text(0.60, 0.20, f"Ticks spacing={prm['tick_spacing']} nm", size=20)
    fig.text(0.60, 0.15, f"average over {nb_phasing} reconstruction(s)", size=20)
    if prm["half_width_avg_phase"] > 0:
        fig.text(
            0.60,
            0.10,
            f"Averaging over {2 * prm['half_width_avg_phase'] + 1} pixels",
            size=20,
        )
    else:
        fig.text(0.60, 0.10, "No phase averaging", size=20)
    if prm["save"]:
        fig.savefig(
            setup.detector.savedir + f"S{scan_nb}_displacement" + comment.text + ".png"
        )

    # strain
    fig, _, _ = gu.multislices_plot(
        interpolated_crystal.strain,
        sum_frames=False,
        title="Orthogonal strain",
        vmin=-prm["strain_range"],
        vmax=prm["strain_range"],
        tick_direction=prm["tick_direction"],
        tick_width=prm["tick_width"],
        tick_length=prm["tick_length"],
        plot_colorbar=True,
        cmap=prm["colormap"].cmap,
        pixel_spacing=pixel_spacing,
        is_orthogonal=True,
        reciprocal_space=False,
    )
    fig.text(0.60, 0.30, f"Scan {scan_nb}", size=20)
    fig.text(
        0.60,
        0.25,
        f"Voxel size=({voxel_sizes[0]:.1f}, "
        f"{voxel_sizes[1]:.1f}, {voxel_sizes[2]:.1f}) (nm)",
        size=20,
    )
    fig.text(0.60, 0.20, f"Ticks spacing={prm['tick_spacing']} nm", size=20)
    fig.text(0.60, 0.15, f"average over {nb_phasing} reconstruction(s)", size=20)
    if prm["half_width_avg_phase"] > 0:
        fig.text(
            0.60,
            0.10,
            f"Averaging over {2 * prm['half_width_avg_phase'] + 1} pixels",
            size=20,
        )
    else:
        fig.text(0.60, 0.10, "No phase averaging", size=20)
    if prm["save"]:
        fig.savefig(
            setup.detector.savedir + f"S{scan_nb}_strain" + comment.text + ".png"
        )

    if len(prm["scans"]) > 1:
        plt.close("all")

    logger.removeHandler(filehandler)
    filehandler.close()

    return tmpfile, Path(setup.detector.savedir), logger
