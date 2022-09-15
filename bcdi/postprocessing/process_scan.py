# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, jerome.carnis@esrf.fr
"""Workflow for BCDI data postprocessing of a single scan, after phase retrieval."""

import gc
from functools import reduce

try:
    import hdf5plugin  # for P10, should be imported before h5py or PyTables
except ModuleNotFoundError:
    pass
import logging
import os
from logging import Logger
from pathlib import Path
from tkinter import filedialog
from typing import Any, Dict, Optional, Tuple

import h5py
import matplotlib
import numpy as np
import yaml
from matplotlib import pyplot as plt

import bcdi.graph.graph_utils as gu
import bcdi.graph.linecut as lc
import bcdi.postprocessing.postprocessing_utils as pu
import bcdi.utils.utilities as util
from bcdi.experiment.setup import Setup
from bcdi.postprocessing.analysis import create_analysis
from bcdi.utils.constants import AXIS_TO_ARRAY
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
        beamline_name=prm["beamline"],
        energy=prm["energy"],
        outofplane_angle=prm["outofplane_angle"],
        inplane_angle=prm["inplane_angle"],
        tilt_angle=prm["tilt_angle"],
        rocking_angle=prm["rocking_angle"],
        distance=prm["detector_distance"],
        sample_offsets=prm["sample_offsets"],
        actuators=prm["actuators"],
        custom_scan=prm["custom_scan"],
        custom_motors=prm["custom_motors"],
        dirbeam_detector_angles=prm["dirbeam_detector_angles"],
        direct_beam=prm["direct_beam"],
        is_series=prm["is_series"],
        detector_name=prm["detector"],
        template_imagefile=prm["template_imagefile"][scan_idx],
        roi=prm["roi_detector"],
        binning=prm["phasing_binning"],
        preprocessing_binning=prm["preprocessing_binning"],
        custom_pixelsize=prm["custom_pixelsize"],
        logger=logger,
    )

    ########################################
    # Initialize the paths and the logfile #
    ########################################
    setup.init_paths(
        sample_name=prm["sample_name"][scan_idx],
        scan_number=scan_nb,
        root_folder=prm["root_folder"],
        data_dir=prm["data_dir"][scan_idx],
        save_dir=prm["save_dir"][scan_idx],
        specfile_name=prm["specfile_name"][scan_idx],
        template_imagefile=prm["template_imagefile"][scan_idx],
    )

    setup.create_logfile(
        scan_number=scan_nb,
        root_folder=prm["root_folder"],
        filename=setup.detector.specfile,
    )

    # load the goniometer positions needed in the calculation
    # of the transformation matrix
    setup.read_logfile(scan_number=scan_nb)

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

    analysis.find_data_range(amplitude_threshold=0.05, plot_margin=10)

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
        comment.concatenate("_apodize_" + prm["apodization_window"])

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
        analysis.update_parameters(
            {
                "inplane_angle": setup.inplane_angle,
                "outofplane_angle": setup.outofplane_angle,
            }
        )

    #########################################################
    # calculate q of the Bragg peak in the laboratory frame #
    #########################################################
    logger.info(
        "Normalized diffusion vector in the laboratory frame (z*, y*, x*): "
        f"{[f'{val:.4f}' for _, val in enumerate(analysis.get_normalized_q_bragg_laboratory_frame)]} (1/A)"
    )
    logger.info(f"Wavevector transfer: {analysis.get_norm_q_bragg:.4f} 1/A")
    logger.info(f"Atomic planar distance: {analysis.get_interplanar_distance:.4f} A")

    #######################
    #  orthogonalize data #
    #######################
    logger.info(f"Shape before interpolation {analysis.data.shape}")
    analysis.interpolate_into_crystal_frame()

    ######################################################
    # center the object (centering based on the modulus) #
    ######################################################
    logger.info("Centering the crystal")
    analysis.center_object_based_on_modulus()

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

    ########################################
    # refraction and absorption correction #
    ########################################
    # TODO remove below placeholder
    planar_dist = analysis.get_interplanar_distance / 10  # switch to nm  # TODO
    original_size = analysis.original_shape
    numz, numy, numx = analysis.optimized_range
    avg_counter = analysis.nb_reconstructions
    voxel_size = analysis.voxel_sizes
    if voxel_size is None:
        raise ValueError("voxel sizes undefined")
    q_lab = analysis.get_normalized_q_bragg_laboratory_frame
    qnorm = analysis.get_norm_q_bragg
    amp = phase_manipulator.modulus
    phase = phase_manipulator.phase
    extent_phase = analysis.extent_phase  # todo test if this is needed with skip_unwrap

    # TODO

    if prm["correct_refraction"]:  # or correct_absorption:
        bulk = pu.find_bulk(
            amp=amp,
            support_threshold=prm["threshold_unwrap_refraction"],
            method=prm["optical_path_method"],
            debugging=prm["debug"],
            cmap=prm["colormap"].cmap,
        )

        kin = setup.incident_wavevector
        kout = setup.exit_wavevector
        # kin and kout were calculated in the laboratory frame,
        # but after the geometric transformation of the crystal, this
        # latter is always in the crystal frame (for simpler strain calculation).
        # We need to transform kin and kout back
        # into the crystal frame (also, xrayutilities output is in crystal frame)
        kin = util.rotate_vector(
            vectors=[kin[2], kin[1], kin[0]],
            axis_to_align=AXIS_TO_ARRAY[prm["ref_axis_q"]],
            reference_axis=[q_lab[2], q_lab[1], q_lab[0]],
        )
        kout = util.rotate_vector(
            vectors=[kout[2], kout[1], kout[0]],
            axis_to_align=AXIS_TO_ARRAY[prm["ref_axis_q"]],
            reference_axis=[q_lab[2], q_lab[1], q_lab[0]],
        )

        # calculate the optical path of the incoming wavevector
        path_in = pu.get_opticalpath(
            support=bulk,
            direction="in",
            k=kin,
            debugging=prm["debug"],
            cmap=prm["colormap"].cmap,
        )  # path_in already in nm

        # calculate the optical path of the outgoing wavevector
        path_out = pu.get_opticalpath(
            support=bulk,
            direction="out",
            k=kout,
            debugging=prm["debug"],
            cmap=prm["colormap"].cmap,
        )  # path_our already in nm

        optical_path = path_in + path_out
        del path_in, path_out
        gc.collect()

        if prm["correct_refraction"]:
            if setup.wavelength is None:
                raise ValueError("X-ray energy undefined")
            phase_correction = (
                2 * np.pi / (1e9 * setup.wavelength) * prm["dispersion"] * optical_path
            )
            phase = phase + phase_correction

            gu.multislices_plot(
                np.multiply(phase_correction, bulk),
                width_z=numz,
                width_y=numy,
                width_x=numx,
                sum_frames=False,
                plot_colorbar=True,
                vmin=0,
                vmax=np.nan,
                title="Refraction correction on the support",
                is_orthogonal=True,
                reciprocal_space=False,
                cmap=prm["colormap"].cmap,
            )
        correct_absorption = False
        if correct_absorption:
            if setup.wavelength is None:
                raise ValueError("X-ray energy undefined")
            amp_correction = np.exp(
                2 * np.pi / (1e9 * setup.wavelength) * prm["absorption"] * optical_path
            )
            amp = amp * amp_correction

            gu.multislices_plot(
                np.multiply(amp_correction, bulk),
                width_z=numz,
                width_y=numy,
                width_x=numx,
                sum_frames=False,
                plot_colorbar=True,
                vmin=1,
                vmax=1.1,
                title="Absorption correction on the support",
                is_orthogonal=True,
                reciprocal_space=False,
                cmap=prm["colormap"].cmap,
            )

        del bulk, optical_path
        gc.collect()

    ##############################################
    # phase ramp and offset removal (mean value) #
    ##############################################
    logger.info("Phase ramp removal")
    amp, phase, _, _, _ = pu.remove_ramp(
        amp=amp,
        phase=phase,
        initial_shape=original_size,
        method=prm["phase_ramp_removal"],
        amplitude_threshold=prm["isosurface_strain"],
        threshold_gradient=prm["threshold_gradient"],
        debugging=prm["debug"],
        cmap=prm["colormap"].cmap,
        logger=logger,
    )

    ########################
    # phase offset removal #
    ########################
    logger.info("Phase offset removal")
    support = np.zeros(amp.shape)
    support[amp > prm["isosurface_strain"] * amp.max()] = 1
    phase = pu.remove_offset(
        array=phase,
        support=support,
        offset_method=prm["offset_method"],
        phase_offset=prm["phase_offset"],
        offset_origin=prm["phase_offset_origin"],
        title="Orthogonal phase",
        debugging=prm["debug"],
        reciprocal_space=False,
        is_orthogonal=True,
        cmap=prm["colormap"].cmap,
    )
    del support
    gc.collect()
    # Wrap the phase around 0 (no more offset)
    if extent_phase is not None:
        phase = util.wrap(
            obj=phase, start_angle=-extent_phase / 2, range_angle=extent_phase
        )

    ################################################################
    # calculate the strain depending on which axis q is aligned on #
    ################################################################
    logger.info(f"Calculation of the strain along {prm['ref_axis_q']}")
    strain = pu.get_strain(
        phase=phase,
        planar_distance=planar_dist,
        voxel_size=voxel_size,
        reference_axis=prm["ref_axis_q"],
        extent_phase=extent_phase,
        method=prm["strain_method"],
        debugging=prm["debug"],
        cmap=prm["colormap"].cmap,
    )

    ################################################
    # optionally rotates back the crystal into the #
    # laboratory frame (for debugging purpose)     #
    ################################################
    if prm["save_frame"] in ["laboratory", "lab_flat_sample"]:
        comment.concatenate("labframe")
        logger.info("Rotating back the crystal in laboratory frame")
        amp, phase, strain = util.rotate_crystal(
            arrays=(amp, phase, strain),
            axis_to_align=AXIS_TO_ARRAY[prm["ref_axis_q"]],
            voxel_size=voxel_size,
            is_orthogonal=True,
            reciprocal_space=False,
            reference_axis=[q_lab[2], q_lab[1], q_lab[0]],
            debugging=(True, False, False),
            title=("amp", "phase", "strain"),
            cmap=prm["colormap"].cmap,
        )
        # q_lab is already in the laboratory frame
        q_final = q_lab

        if prm["save_frame"] == "lab_flat_sample":
            comment.concatenate("flat")
            logger.info("Sending sample stage circles to 0")
            (amp, phase, strain), q_final = setup.beamline.flatten_sample(
                arrays=(amp, phase, strain),
                voxel_size=voxel_size,
                q_bragg=q_lab[::-1],  # q_bragg needs to be in xyz order
                is_orthogonal=True,
                reciprocal_space=False,
                rocking_angle=setup.rocking_angle,
                debugging=(True, False, False),
                title=("amp", "phase", "strain"),
                cmap=prm["colormap"].cmap,
            )
    else:  # "save_frame" = "crystal"
        # rotate also q_lab to have it along ref_axis_q,
        # as a cross-checkm, vectors needs to be in xyz order
        comment.concatenate("crystalframe")
        q_final = util.rotate_vector(
            vectors=q_lab[::-1],
            axis_to_align=AXIS_TO_ARRAY[prm["ref_axis_q"]],
            reference_axis=q_lab[::-1],
        )

    ###############################################
    # rotates the crystal e.g. for easier slicing #
    # of the result along a particular direction  #
    ###############################################
    # typically this is an inplane rotation, q should stay aligned with the axis
    # along which the strain was calculated
    if prm["align_axis"]:
        logger.info("Rotating arrays for visualization")
        amp, phase, strain = util.rotate_crystal(
            arrays=(amp, phase, strain),
            reference_axis=AXIS_TO_ARRAY[prm["ref_axis"]],
            axis_to_align=prm["axis_to_align"],
            voxel_size=voxel_size,
            debugging=(True, False, False),
            is_orthogonal=True,
            reciprocal_space=False,
            title=("amp", "phase", "strain"),
            cmap=prm["colormap"].cmap,
        )
        # rotate q accordingly, vectors needs to be in xyz order
        if q_final is not None:
            q_final = util.rotate_vector(
                vectors=q_final[::-1],
                axis_to_align=AXIS_TO_ARRAY[prm["ref_axis"]],
                reference_axis=prm["axis_to_align"],
            )

    q_final = q_final * qnorm
    logger.info(
        f"q_final = ({q_final[0]:.4f} 1/A,"
        f" {q_final[1]:.4f} 1/A, {q_final[2]:.4f} 1/A)"
    )

    ##############################################
    # pad array to fit the output_size parameter #
    ##############################################
    if prm["output_size"] is not None:
        amp = util.crop_pad(
            array=amp, output_shape=prm["output_size"], cmap=prm["colormap"].cmap
        )
        phase = util.crop_pad(
            array=phase, output_shape=prm["output_size"], cmap=prm["colormap"].cmap
        )
        strain = util.crop_pad(
            array=strain, output_shape=prm["output_size"], cmap=prm["colormap"].cmap
        )
    logger.info(f"Final data shape: {amp.shape}")

    ######################
    # save result to vtk #
    ######################
    logger.info(
        f"Voxel size: ({voxel_size[0]:.2f} nm, {voxel_size[1]:.2f} nm,"
        f" {voxel_size[2]:.2f} nm)"
    )
    bulk = pu.find_bulk(
        amp=amp,
        support_threshold=prm["isosurface_strain"],
        method="threshold",
        cmap=prm["colormap"].cmap,
    )
    if prm["save"]:
        prm["comment"] = comment.text
        np.savez_compressed(
            f"{setup.detector.savedir}S{scan_nb}_"
            f"amp{prm['phase_fieldname']}strain{comment.text}",
            amp=amp,
            phase=phase,
            bulk=bulk,
            strain=strain,
            q_bragg=q_final,
            voxel_sizes=voxel_size,
            detector=setup.detector.params,
            setup=str(yaml.dump(setup.params)),
            params=str(yaml.dump(prm)),
        )

        # save results in hdf5 file
        with h5py.File(
            f"{setup.detector.savedir}S{scan_nb}_"
            f"amp{prm['phase_fieldname']}strain{comment.text}.h5",
            "w",
        ) as hf:
            out = hf.create_group("output")
            par = hf.create_group("params")
            out.create_dataset("amp", data=amp)
            out.create_dataset("bulk", data=bulk)
            out.create_dataset("phase", data=phase)
            out.create_dataset("strain", data=strain)
            out.create_dataset("q_bragg", data=q_final)
            out.create_dataset("voxel_sizes", data=voxel_size)
            par.create_dataset("detector", data=str(setup.detector.params))
            par.create_dataset("setup", data=str(setup.params))
            par.create_dataset("parameters", data=str(prm))

        # save amp & phase to VTK
        # in VTK, x is downstream, y vertical, z inboard,
        # thus need to flip the last axis
        gu.save_to_vti(
            filename=os.path.join(
                setup.detector.savedir,
                "S"
                + str(scan_nb)
                + "_amp-"
                + prm["phase_fieldname"]
                + "-strain"
                + comment.text
                + ".vti",
            ),
            voxel_size=voxel_size,
            tuple_array=(amp, bulk, phase, strain),
            tuple_fieldnames=("amp", "bulk", prm["phase_fieldname"], "strain"),
            amplitude_threshold=0.01,
        )

    ######################################
    # estimate the volume of the crystal #
    ######################################
    amp = amp / amp.max()
    temp_amp = np.copy(amp)
    temp_amp[amp < prm["isosurface_strain"]] = 0
    temp_amp[np.nonzero(temp_amp)] = 1
    volume = temp_amp.sum() * reduce(lambda x, y: x * y, voxel_size)  # in nm3
    del temp_amp
    gc.collect()

    ################################
    # plot linecuts of the results #
    ################################
    lc.fit_linecut(
        array=amp,
        fit_derivative=True,
        filename=setup.detector.savedir + "linecut_amp.png",
        voxel_sizes=voxel_size,
        label="modulus",
        logger=logger,
    )

    ##############################
    # plot slices of the results #
    ##############################
    pixel_spacing = [prm["tick_spacing"] / vox for vox in voxel_size]
    logger.info(
        "Phase extent without / with thresholding the modulus "
        f"(threshold={prm['isosurface_strain']}): "
        f"{phase.max() - phase.min():.2f} rad, "
        f"{phase[np.nonzero(bulk)].max() - phase[np.nonzero(bulk)].min():.2f} rad"
    )
    piz, piy, pix = np.unravel_index(phase.argmax(), phase.shape)
    logger.info(
        f"phase.max() = {phase[np.nonzero(bulk)].max():.2f} "
        f"at voxel ({piz}, {piy}, {pix})"
    )
    strain[bulk == 0] = np.nan
    phase[bulk == 0] = np.nan

    # plot the slice at the maximum phase
    fig = gu.combined_plots(
        (phase[piz, :, :], phase[:, piy, :], phase[:, :, pix]),
        tuple_sum_frames=False,
        tuple_sum_axis=0,
        tuple_width_v=None,
        tuple_width_h=None,
        tuple_colorbar=True,
        tuple_vmin=np.nan,
        tuple_vmax=np.nan,
        tuple_title=(
            "phase at max in xy",
            "phase at max in xz",
            "phase at max in yz",
        ),
        tuple_scale="linear",
        cmap=prm["colormap"].cmap,
        is_orthogonal=True,
        reciprocal_space=False,
    )
    plt.pause(0.1)
    if prm["save"]:
        fig.savefig(
            setup.detector.savedir
            + "S"
            + str(scan_nb)
            + "_phase_at_max"
            + comment.text
            + ".png"
        )
    plt.close(fig)

    # bulk support
    fig, _, _ = gu.multislices_plot(
        bulk,
        sum_frames=False,
        title="Orthogonal bulk",
        vmin=0,
        vmax=1,
        is_orthogonal=True,
        reciprocal_space=False,
        cmap=prm["colormap"].cmap,
    )
    fig.text(0.60, 0.45, "Scan " + str(scan_nb), size=20)
    fig.text(
        0.60,
        0.40,
        f"Bulk - isosurface= {prm['isosurface_strain']:.2f}",
        size=20,
    )
    plt.pause(0.1)
    if prm["save"]:
        fig.savefig(
            setup.detector.savedir
            + "S"
            + str(scan_nb)
            + "_bulk"
            + comment.text
            + ".png"
        )
    plt.close(fig)

    # amplitude
    fig, _, _ = gu.multislices_plot(
        amp,
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
    fig.text(
        0.60,
        0.40,
        f"Voxel size=({voxel_size[0]:.1f}, {voxel_size[1]:.1f}, "
        f"{voxel_size[2]:.1f}) (nm)",
        size=20,
    )
    fig.text(0.60, 0.35, f"Ticks spacing={prm['tick_spacing']} nm", size=20)
    fig.text(0.60, 0.30, f"Volume={int(volume)} nm3", size=20)
    fig.text(0.60, 0.25, "Sorted by " + prm["sort_method"], size=20)
    fig.text(
        0.60, 0.20, f"correlation threshold={prm['correlation_threshold']}", size=20
    )
    fig.text(0.60, 0.15, f"average over {avg_counter} reconstruction(s)", size=20)
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
    ax.hist(amp[amp > 0.05 * amp.max()].flatten(), bins=250)
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
        phase,
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
        f"Voxel size=({voxel_size[0]:.1f}, {voxel_size[1]:.1f}, "
        f"{voxel_size[2]:.1f}) (nm)",
        size=20,
    )
    fig.text(0.60, 0.20, f"Ticks spacing={prm['tick_spacing']} nm", size=20)
    fig.text(0.60, 0.15, f"average over {avg_counter} reconstruction(s)", size=20)
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
        strain,
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
        f"Voxel size=({voxel_size[0]:.1f}, "
        f"{voxel_size[1]:.1f}, {voxel_size[2]:.1f}) (nm)",
        size=20,
    )
    fig.text(0.60, 0.20, f"Ticks spacing={prm['tick_spacing']} nm", size=20)
    fig.text(0.60, 0.15, f"average over {avg_counter} reconstruction(s)", size=20)
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
