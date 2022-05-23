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
import tkinter as tk
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
import bcdi.postprocessing.postprocessing_utils as pu
import bcdi.preprocessing.bcdi_utils as bu
import bcdi.simulation.simulation_utils as simu
import bcdi.utils.image_registration as reg
import bcdi.utils.utilities as util
from bcdi.experiment.setup import Setup
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
        / f"run{scan_idx}_{prm['sample_name'][scan_idx]}{scan_nb}.log"
    )
    filehandler = logging.FileHandler(tmpfile, mode="w", encoding="utf-8")
    filehandler.setFormatter(FILE_FORMATTER)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(filehandler)
    if not prm["multiprocessing"] or len(prm["scans"]) == 1:
        logger.propagate = True

    prm["sample"] = f"{prm['sample_name']}+{scan_nb}"
    comment = prm["comment"]  # re-initialize comment
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

    ###################
    # print instances #
    ###################
    logger.info(f"##############\nSetup instance\n##############\n{setup.params}")
    logger.info(
        "#################\nDetector instance\n#################\n"
        f"{setup.detector.params}"
    )

    ################
    # preload data #
    ################
    if prm["reconstruction_files"][scan_idx] is not None:
        file_path = prm["reconstruction_files"][scan_idx]
        if isinstance(file_path, str):
            file_path = (file_path,)
    else:
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilenames(
            initialdir=setup.detector.scandir
            if prm["data_dir"] is None
            else setup.detector.datadir,
            filetypes=[
                ("HDF5", "*.h5"),
                ("CXI", "*.cxi"),
                ("NPZ", "*.npz"),
                ("NPY", "*.npy"),
            ],
        )

    nbfiles = len(file_path)
    plt.ion()

    obj, extension = util.load_file(file_path[0])
    if extension == ".h5":
        comment = comment + "_mode"

    logger.info("###############\nProcessing data\n###############")
    nz, ny, nx = obj.shape
    logger.info(f"Initial data size: ({nz}, {ny}, {nx})")
    original_size = prm["original_size"] if prm["original_size"] else obj.shape
    logger.info(f"FFT size before accounting for phasing_binning: {original_size}")
    original_size = tuple(
        [
            original_size[index] // prm["phasing_binning"][index]
            for index in range(len(prm["phasing_binning"]))
        ]
    )
    logger.info(f"Binning used during phasing: {setup.detector.binning}")
    logger.info(f"Padding back to original FFT size: {original_size}")
    obj = util.crop_pad(
        array=obj, output_shape=original_size, cmap=prm["colormap"].cmap
    )

    ###########################################################################
    # define range for orthogonalization and plotting - speed up calculations #
    ###########################################################################
    zrange, yrange, xrange = pu.find_datarange(
        array=obj, amplitude_threshold=0.05, keep_size=prm["keep_size"]
    )

    numz = zrange * 2
    numy = yrange * 2
    numx = xrange * 2
    logger.info(
        "Data shape used for orthogonalization and plotting: "
        f"({numz}, {numy}, {numx})"
    )

    ######################################################################
    # find the best reconstruction, based on mean amplitude and variance #
    ######################################################################
    if nbfiles > 1:
        logger.info(
            "Trying to find the best reconstruction\nSorting by "
            f"{prm['sort_method']}"
        )
        sorted_obj = pu.sort_reconstruction(
            file_path=file_path,
            amplitude_threshold=prm["isosurface_strain"],
            data_range=(zrange, yrange, xrange),
            sort_method=prm["sort_method"],
        )
    else:
        sorted_obj = [0]

    #######################################
    # load reconstructions and average it #
    #######################################
    avg_obj = np.zeros((numz, numy, numx))
    ref_obj = np.zeros((numz, numy, numx))
    avg_counter = 1
    logger.info(f"Averaging using {nbfiles} candidate reconstructions")
    for counter, value in enumerate(sorted_obj):
        obj, extension = util.load_file(file_path[value])
        logger.info(f"Opening {file_path[value]}")
        prm[f"from_file_{counter}"] = file_path[value]

        if prm["flip_reconstruction"]:
            obj = pu.flip_reconstruction(obj, debugging=True, cmap=prm["colormap"].cmap)

        if extension == ".h5":
            prm[
                "centering_method"
            ] = "do_nothing"  # do not center, data is already cropped
            # just on support for mode decomposition
            # correct a roll after the decomposition into modes in PyNX
            obj = np.roll(obj, prm["roll_modes"], axis=(0, 1, 2))
            fig, _, _ = gu.multislices_plot(
                abs(obj),
                sum_frames=True,
                plot_colorbar=True,
                title="1st mode after centering",
                cmap=prm["colormap"].cmap,
            )

        # use the range of interest defined above
        obj = util.crop_pad(
            obj,
            [2 * zrange, 2 * yrange, 2 * xrange],
            debugging=False,
            cmap=prm["colormap"].cmap,
        )

        # align with average reconstruction
        if counter == 0:  # the fist array loaded will serve as reference object
            logger.info("This reconstruction will be used as reference.")
            ref_obj = obj

        avg_obj, flag_avg = reg.average_arrays(
            avg_obj=avg_obj,
            ref_obj=ref_obj,
            obj=obj,
            support_threshold=0.25,
            correlation_threshold=prm["correlation_threshold"],
            aligning_option="dft",
            space=prm["averaging_space"],
            reciprocal_space=False,
            is_orthogonal=prm["is_orthogonal"],
            debugging=prm["debug"],
            cmap=prm["colormap"].cmap,
        )
        avg_counter = avg_counter + flag_avg

    avg_obj = avg_obj / avg_counter
    if avg_counter > 1:
        logger.info(f"Average performed over {avg_counter} reconstructions\n")
    del obj, ref_obj
    gc.collect()

    ################
    # unwrap phase #
    ################
    if prm["skip_unwrap"]:
        phase = np.angle(avg_obj)
    else:
        phase, extent_phase = pu.unwrap(
            avg_obj,
            support_threshold=prm["threshold_unwrap_refraction"],
            debugging=prm["debug"],
            reciprocal_space=False,
            is_orthogonal=prm["is_orthogonal"],
            cmap=prm["colormap"].cmap,
        )

        logger.info(
            "Extent of the phase over an extended support (ceil(phase range)) ~ "
            f"{int(extent_phase)} (rad)",
        )
        phase = util.wrap(
            phase, start_angle=-extent_phase / 2, range_angle=extent_phase
        )
        if prm["debug"]:
            gu.multislices_plot(
                phase,
                width_z=2 * zrange,
                width_y=2 * yrange,
                width_x=2 * xrange,
                plot_colorbar=True,
                title="Phase after unwrap + wrap",
                reciprocal_space=False,
                is_orthogonal=prm["is_orthogonal"],
                cmap=prm["colormap"].cmap,
            )

    #############################################
    # phase ramp removal before phase filtering #
    #############################################
    amp, phase, rampz, rampy, rampx = pu.remove_ramp(
        amp=abs(avg_obj),
        phase=phase,
        initial_shape=original_size,
        method="gradient",
        amplitude_threshold=prm["isosurface_strain"],
        threshold_gradient=prm["threshold_gradient"],
        cmap=prm["colormap"].cmap,
        logger=logger,
    )
    del avg_obj
    gc.collect()

    if prm["debug"]:
        gu.multislices_plot(
            phase,
            width_z=2 * zrange,
            width_y=2 * yrange,
            width_x=2 * xrange,
            plot_colorbar=True,
            title="Phase after ramp removal",
            reciprocal_space=False,
            is_orthogonal=prm["is_orthogonal"],
            cmap=prm["colormap"].cmap,
        )

    ########################
    # phase offset removal #
    ########################
    support = np.zeros(amp.shape)
    support[amp > prm["isosurface_strain"] * amp.max()] = 1
    phase = pu.remove_offset(
        array=phase,
        support=support,
        offset_method=prm["offset_method"],
        phase_offset=prm["phase_offset"],
        offset_origin=prm["phase_offset_origin"],
        title="Phase",
        debugging=prm["debug"],
        cmap=prm["colormap"].cmap,
    )
    del support
    gc.collect()

    phase = util.wrap(
        obj=phase, start_angle=-extent_phase / 2, range_angle=extent_phase
    )

    ##############################################################################
    # average the phase over a window or apodize to reduce noise in strain plots #
    ##############################################################################
    if prm["half_width_avg_phase"] != 0:
        bulk = pu.find_bulk(
            amp=amp,
            support_threshold=prm["isosurface_strain"],
            method="threshold",
            cmap=prm["colormap"].cmap,
        )
        # the phase should be averaged only in the support defined by the isosurface
        phase = pu.mean_filter(
            array=phase,
            support=bulk,
            half_width=prm["half_width_avg_phase"],
            cmap=prm["colormap"].cmap,
        )
        del bulk
        gc.collect()

    if prm["half_width_avg_phase"] != 0:
        comment = comment + "_avg" + str(2 * prm["half_width_avg_phase"] + 1)

    gridz, gridy, gridx = np.meshgrid(
        np.arange(0, numz, 1),
        np.arange(0, numy, 1),
        np.arange(0, numx, 1),
        indexing="ij",
    )

    phase = (
        phase + gridz * rampz + gridy * rampy + gridx * rampx
    )  # put back the phase ramp otherwise the diffraction
    # pattern will be shifted and the prtf messed up

    if prm["apodize"]:
        amp, phase = pu.apodize(
            amp=amp,
            phase=phase,
            initial_shape=original_size,
            window_type=prm["apodization_window"],
            sigma=prm["apodization_sigma"],
            mu=prm["apodization_mu"],
            alpha=prm["apodization_alpha"],
            is_orthogonal=prm["is_orthogonal"],
            debugging=True,
            cmap=prm["colormap"].cmap,
        )
        comment = comment + "_apodize_" + prm["apodization_window"]

    ################################################################
    # save the phase with the ramp for PRTF calculations,          #
    # otherwise the object will be misaligned with the measurement #
    ################################################################
    np.savez_compressed(
        setup.detector.savedir + "S" + str(scan_nb) + "_avg_obj_prtf" + comment,
        obj=amp * np.exp(1j * phase),
    )

    ####################################################
    # remove again phase ramp before orthogonalization #
    ####################################################
    phase = phase - gridz * rampz - gridy * rampy - gridx * rampx

    avg_obj = amp * np.exp(1j * phase)  # here the phase is again wrapped in [-pi pi[

    del amp, phase, gridz, gridy, gridx, rampz, rampy, rampx
    gc.collect()

    ######################
    # centering of array #
    ######################
    if prm["centering_method"] == "max":
        avg_obj = pu.center_max(avg_obj)
        # shift based on max value,
        # required if it spans across the edge of the array before COM
    elif prm["centering_method"] == "com":
        avg_obj = pu.center_com(avg_obj)
    elif prm["centering_method"] == "max_com":
        avg_obj = pu.center_max(avg_obj)
        avg_obj = pu.center_com(avg_obj)

    #######################
    #  save support & vti #
    #######################
    if prm["save_support"]:
        # to be used as starting support in phasing,
        # hence still in the detector frame
        support = np.zeros((numz, numy, numx))
        support[abs(avg_obj) / abs(avg_obj).max() > 0.01] = 1
        # low threshold because support will be cropped by shrinkwrap during phasing
        np.savez_compressed(
            setup.detector.savedir + "S" + str(scan_nb) + "_support" + comment,
            obj=support,
        )
        del support
        gc.collect()

    if prm["save_rawdata"]:
        np.savez_compressed(
            setup.detector.savedir + "S" + str(scan_nb) + "_raw_amp-phase" + comment,
            amp=abs(avg_obj),
            phase=np.angle(avg_obj),
        )

        # voxel sizes in the detector frame
        voxel_z, voxel_y, voxel_x = setup.voxel_sizes_detector(
            array_shape=original_size,
            tilt_angle=(
                prm["tilt_angle"]
                * setup.detector.prm["preprocessing_binning"][0]
                * setup.detector.binning[0]
            ),
            pixel_x=setup.detector.pixelsize_x,
            pixel_y=setup.detector.pixelsize_y,
            verbose=True,
        )
        # save raw amp & phase to VTK
        # in VTK, x is downstream, y vertical, z inboard,
        # thus need to flip the last axis
        gu.save_to_vti(
            filename=os.path.join(
                setup.detector.savedir,
                "S" + str(scan_nb) + "_raw_amp-phase" + comment + ".vti",
            ),
            voxel_size=(voxel_z, voxel_y, voxel_x),
            tuple_array=(abs(avg_obj), np.angle(avg_obj)),
            tuple_fieldnames=("amp", "phase"),
            amplitude_threshold=0.01,
        )

    #########################################################
    # calculate q of the Bragg peak in the laboratory frame #
    #########################################################
    q_lab = (
        setup.q_laboratory
    )  # (1/A), in the laboratory frame z downstream, y vertical, x outboard
    qnorm = np.linalg.norm(q_lab)
    q_lab = q_lab / qnorm

    angle = simu.angle_vectors(
        ref_vector=[q_lab[2], q_lab[1], q_lab[0]],
        test_vector=AXIS_TO_ARRAY[prm["ref_axis_q"]],
    )
    logger.info(
        f"Normalized diffusion vector in the laboratory frame (z*, y*, x*): "
        f"({q_lab[0]:.4f} 1/A, {q_lab[1]:.4f} 1/A, {q_lab[2]:.4f} 1/A)"
    )

    planar_dist = 2 * np.pi / qnorm  # qnorm should be in angstroms
    logger.info(f"Wavevector transfer: {qnorm:.4f} 1/A")
    logger.info(f"Atomic planar distance: {planar_dist:.4f} A")
    logger.info(f"Angle between q_lab and {prm['ref_axis_q']} = {angle:.2f} deg")
    if prm["debug"]:
        logger.info(
            "Angle with y in zy plane = "
            f"{np.arctan(q_lab[0] / q_lab[1]) * 180 / np.pi:.2f} deg"
        )
        logger.info(
            "Angle with y in xy plane = "
            f"{np.arctan(-q_lab[2] / q_lab[1]) * 180 / np.pi:.2f} deg"
        )
        logger.info(
            "Angle with z in xz plane = "
            f"{180 + np.arctan(q_lab[2] / q_lab[0]) * 180 / np.pi:.2f} deg\n"
        )

    planar_dist = planar_dist / 10  # switch to nm

    #######################
    #  orthogonalize data #
    #######################
    logger.info(f"Shape before orthogonalization {avg_obj.shape}\n")
    if prm["data_frame"] == "detector":
        if prm["debug"] and not prm["skip_unwrap"]:
            phase, _ = pu.unwrap(
                avg_obj,
                support_threshold=prm["threshold_unwrap_refraction"],
                debugging=True,
                reciprocal_space=False,
                is_orthogonal=False,
                cmap=prm["colormap"].cmap,
            )
            gu.multislices_plot(
                phase,
                width_z=2 * zrange,
                width_y=2 * yrange,
                width_x=2 * xrange,
                sum_frames=False,
                plot_colorbar=True,
                reciprocal_space=False,
                is_orthogonal=False,
                title="unwrapped phase before orthogonalization",
                cmap=prm["colormap"].cmap,
            )
            del phase
            gc.collect()

        if not prm["outofplane_angle"] and not prm["inplane_angle"]:
            logger.info("Trying to correct detector angles using the direct beam")
            # corrected detector angles not provided
            if (
                prm["bragg_peak"] is None
                and setup.detector.template_imagefile is not None
            ):
                # Bragg peak position not provided, find it from the data
                data, _, _, _ = setup.loader.load_check_dataset(
                    scan_number=scan_nb,
                    setup=setup,
                    frames_pattern=prm["frames_pattern"],
                    bin_during_loading=False,
                    flatfield=prm["flatfield_file"],
                    hotpixels=prm["hotpixels_file"],
                    background=prm["background_file"],
                    normalize=prm["normalize_flux"],
                )
                bragg_peak = bu.find_bragg(
                    data=data,
                    peak_method="maxcom",
                    roi=setup.detector.roi,
                    binning=None,
                    logger=logger,
                )
                roi_center = (
                    bragg_peak[0],
                    bragg_peak[1]
                    - setup.detector.roi[0],  # no binning as in bu.find_bragg
                    bragg_peak[2]
                    - setup.detector.roi[2],  # no binning as in bu.find_bragg
                )
                bu.show_rocking_curve(
                    data,
                    roi_center=roi_center,
                    tilt_values=setup.incident_angles,
                    savedir=setup.detector.savedir,
                    logger=logger,
                )
                prm["bragg_peak"] = bragg_peak
            setup.correct_detector_angles(bragg_peak_position=prm["bragg_peak"])
            prm["outofplane_angle"] = setup.outofplane_angle
            prm["inplane_angle"] = setup.inplane_angle

        obj_ortho, voxel_size, transfer_matrix = setup.ortho_directspace(
            arrays=avg_obj,
            q_com=np.array([q_lab[2], q_lab[1], q_lab[0]]),
            initial_shape=original_size,
            voxel_size=prm["fix_voxel"],
            reference_axis=AXIS_TO_ARRAY[prm["ref_axis_q"]],
            fill_value=0,
            debugging=True,
            title="amplitude",
            cmap=prm["colormap"].cmap,
        )
        prm["transformation_matrix"] = transfer_matrix
    else:  # data already orthogonalized using xrayutilities
        # or the linearized transformation matrix
        obj_ortho = avg_obj
        try:
            logger.info("Select the file containing QxQzQy")
            file_path = filedialog.askopenfilename(
                title="Select the file containing QxQzQy",
                initialdir=setup.detector.savedir,
                filetypes=[("NPZ", "*.npz")],
            )
            npzfile = np.load(file_path)
            qx = npzfile["qx"]
            qy = npzfile["qy"]
            qz = npzfile["qz"]
        except FileNotFoundError:
            raise FileNotFoundError(
                "q values not provided, the voxel size cannot be calculated"
            )
        dy_real = (
            2 * np.pi / abs(qz.max() - qz.min()) / 10
        )  # in nm qz=y in nexus convention
        dx_real = (
            2 * np.pi / abs(qy.max() - qy.min()) / 10
        )  # in nm qy=x in nexus convention
        dz_real = (
            2 * np.pi / abs(qx.max() - qx.min()) / 10
        )  # in nm qx=z in nexus convention
        logger.info(
            f"direct space voxel size from q values: ({dz_real:.2f} nm,"
            f" {dy_real:.2f} nm, {dx_real:.2f} nm)"
        )
        if prm["fix_voxel"]:
            voxel_size = prm["fix_voxel"]
            logger.info(
                f"Direct space pixel size for the interpolation: {voxel_size} (nm)"
            )
            logger.info("Interpolating...\n")
            obj_ortho = pu.regrid(
                array=obj_ortho,
                old_voxelsize=(dz_real, dy_real, dx_real),
                new_voxelsize=voxel_size,
            )
        else:
            # no need to interpolate
            voxel_size = dz_real, dy_real, dx_real  # in nm

        if (
            prm["data_frame"] == "laboratory"
        ):  # the object must be rotated into the crystal frame
            # before the strain calculation
            logger.info(
                "Rotating the object in the crystal frame " "for the strain calculation"
            )

            amp, phase = util.rotate_crystal(
                arrays=(abs(obj_ortho), np.angle(obj_ortho)),
                is_orthogonal=True,
                reciprocal_space=False,
                voxel_size=voxel_size,
                debugging=(True, False),
                axis_to_align=q_lab[::-1],
                reference_axis=AXIS_TO_ARRAY[prm["ref_axis_q"]],
                title=("amp", "phase"),
                cmap=prm["colormap"].cmap,
            )

            obj_ortho = amp * np.exp(
                1j * phase
            )  # here the phase is again wrapped in [-pi pi[
            del amp, phase

    del avg_obj
    gc.collect()

    ######################################################
    # center the object (centering based on the modulus) #
    ######################################################
    logger.info("Centering the crystal")
    obj_ortho = pu.center_com(obj_ortho)

    ####################
    # Phase unwrapping #
    ####################
    if prm["skip_unwrap"]:
        phase = np.angle(obj_ortho)
    else:
        logger.info("Phase unwrapping")
        phase, extent_phase = pu.unwrap(
            obj_ortho,
            support_threshold=prm["threshold_unwrap_refraction"],
            debugging=True,
            reciprocal_space=False,
            is_orthogonal=True,
            cmap=prm["colormap"].cmap,
        )
    amp = abs(obj_ortho)
    del obj_ortho
    gc.collect()

    #############################################
    # invert phase: -1*phase = displacement * q #
    #############################################
    if prm["invert_phase"]:
        phase = -1 * phase

    ########################################
    # refraction and absorption correction #
    ########################################
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
            phase_correction = (
                2 * np.pi / (1e9 * setup.wavelength) * prm["dispersion"] * optical_path
            )
            phase = phase + phase_correction

            gu.multislices_plot(
                np.multiply(phase_correction, bulk),
                width_z=2 * zrange,
                width_y=2 * yrange,
                width_x=2 * xrange,
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
            amp_correction = np.exp(
                2 * np.pi / (1e9 * setup.wavelength) * prm["absorption"] * optical_path
            )
            amp = amp * amp_correction

            gu.multislices_plot(
                np.multiply(amp_correction, bulk),
                width_z=2 * zrange,
                width_y=2 * yrange,
                width_x=2 * xrange,
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
        comment = comment + "_labframe"
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
            comment = comment + "_flat"
            logger.info("Sending sample stage circles to 0")
            (amp, phase, strain), q_final = setup.beamline.flatten_sample(
                arrays=(amp, phase, strain),
                voxel_size=voxel_size,
                q_com=q_lab[::-1],  # q_com needs to be in xyz order
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
        comment = comment + "_crystalframe"
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
        prm["comment"] = comment
        np.savez_compressed(
            f"{setup.detector.savedir}S{scan_nb}_"
            f"amp{prm['phase_fieldname']}strain{comment}",
            amp=amp,
            phase=phase,
            bulk=bulk,
            strain=strain,
            q_com=q_final,
            voxel_sizes=voxel_size,
            detector=setup.detector.params,
            setup=str(yaml.dump(setup.params)),
            params=str(yaml.dump(prm)),
        )

        # save results in hdf5 file
        with h5py.File(
            f"{setup.detector.savedir}S{scan_nb}_"
            f"amp{prm['phase_fieldname']}strain{comment}.h5",
            "w",
        ) as hf:
            out = hf.create_group("output")
            par = hf.create_group("params")
            out.create_dataset("amp", data=amp)
            out.create_dataset("bulk", data=bulk)
            out.create_dataset("phase", data=phase)
            out.create_dataset("strain", data=strain)
            out.create_dataset("q_com", data=q_final)
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
                + comment
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
    gu.fit_linecut(
        array=amp,
        fit_derivative=True,
        filename=setup.detector.savedir + "linecut_amp.png",
        voxel_sizes=voxel_size,
        label="modulus",
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
    gu.combined_plots(
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
        "Bulk - isosurface=" + str("{:.2f}".format(prm["isosurface_strain"])),
        size=20,
    )
    plt.pause(0.1)
    if prm["save"]:
        fig.savefig(
            setup.detector.savedir + "S" + str(scan_nb) + "_bulk" + comment + ".png"
        )

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
        fig.savefig(setup.detector.savedir + f"S{scan_nb}_amp" + comment + ".png")

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
    fig.savefig(setup.detector.savedir + f"S{scan_nb}_histo_amp" + comment + ".png")

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
            setup.detector.savedir + f"S{scan_nb}_displacement" + comment + ".png"
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
        fig.savefig(setup.detector.savedir + f"S{scan_nb}_strain" + comment + ".png")

    if len(prm["scans"]) > 1:
        plt.close("all")

    logger.removeHandler(filehandler)
    filehandler.close()

    return tmpfile, Path(setup.detector.savedir), logger
