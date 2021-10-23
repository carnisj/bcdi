#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

from datetime import datetime
from functools import reduce
import gc

try:
    import hdf5plugin  # for P10, should be imported before h5py or PyTables
except ModuleNotFoundError:
    pass
import h5py
from matplotlib import pyplot as plt
import numpy as np
import os
import pprint
import tkinter as tk
from tkinter import filedialog
import argparse

import bcdi.graph.graph_utils as gu
from bcdi.experiment.detector import create_detector
from bcdi.experiment.setup import Setup
import bcdi.postprocessing.postprocessing_utils as pu
import bcdi.simulation.simulation_utils as simu
import bcdi.utils.image_registration as reg
from bcdi.utils.parser import ConfigParser
import bcdi.utils.utilities as util

helptext = """
Interpolate the output of the phase retrieval into an orthonormal frame,
and calculate the strain component along the direction of the experimental diffusion
vector q.

Input: complex amplitude array, output from a phase retrieval program.
Output: data in an orthonormal frame (laboratory or crystal frame), amp_disp_strain
array.The disp array should be divided by q to get the displacement (disp = -1*phase
here).

Laboratory frame: z downstream, y vertical, x outboard (CXI convention)
Crystal reciprocal frame: qx downstream, qz vertical, qy outboard
Detector convention: when out_of_plane angle=0   Y=-y , when in_plane angle=0   X=x

In arrays, when plotting the first parameter is the row (vertical axis), and the
second the column (horizontal axis). Therefore the data structure is data[qx, qz,
qy] for reciprocal space, or data[z, y, x] for real space
"""


def run(prm):
    """
    Run the postprocessing.

    :param prm: the parsed parameters
    """
    pretty = pprint.PrettyPrinter(indent=4)

    ################################
    # assign often used parameters #
    ################################
    debug = prm["debug"]
    comment = prm["comment"]
    centering_method = prm["centering_method"]
    original_size = prm["original_size"]
    phasing_binning = prm["phasing_binning"]
    preprocessing_binning = prm["preprocessing_binning"]
    isosurface_strain = prm["isosurface_strain"]
    ref_axis_q = prm["ref_axis_q"]
    fix_voxel = prm["fix_voxel"]
    output_size = prm["output_size"]
    save = prm["save"]
    tick_spacing = prm["tick_spacing"]
    tick_direction = prm["tick_direction"]
    tick_length = prm["tick_length"]
    tick_width = prm["tick_width"]
    invert_phase = prm["invert_phase"]
    correct_refraction = prm["correct_refraction"]
    save_frame = prm["save_frame"]
    data_frame = prm["data_frame"]
    scan = prm["scan"]
    sample_name = prm["sample_name"]
    root_folder = prm["root_folder"]

    prm["sample"] = f"{sample_name}+{scan}",
    #########################
    # Check some parameters #
    #########################
    if prm["simulation"]:
        invert_phase = False
        correct_refraction = 0
    if invert_phase:
        phase_fieldname = "disp"
    else:
        phase_fieldname = "phase"

    if data_frame == "detector":
        is_orthogonal = False
    else:
        is_orthogonal = True

    if data_frame == "crystal" and save_frame != "crystal":
        print(
            "data already in the crystal frame before phase retrieval,"
            " it is impossible to come back to the laboratory "
            "frame, parameter 'save_frame' defaulted to 'crystal'"
        )
        save_frame = "crystal"

    axis_to_array_xyz = {
        "x": np.array([1, 0, 0]),
        "y": np.array([0, 1, 0]),
        "z": np.array([0, 0, 1]),
    }  # in xyz order

    ###################
    # define colormap #
    ###################
    if prm["grey_background"]:
        bad_color = "0.7"
    else:
        bad_color = "1.0"  # white background
    colormap = gu.Colormap(bad_color=bad_color)
    my_cmap = colormap.cmap

    #######################
    # Initialize detector #
    #######################
    detector = create_detector(
        name=prm["detector"],
        template_imagefile=prm["template_imagefile"],
        binning=phasing_binning,
        preprocessing_binning=preprocessing_binning,
        pixel_size=prm["pixel_size"],
    )

    ####################################
    # define the experimental geometry #
    ####################################
    # correct the tilt_angle for binning
    tilt_angle = prm["tilt_angle"] * preprocessing_binning[0] * phasing_binning[0]
    setup = Setup(
        beamline=prm["beamline"],
        detector=detector,
        energy=prm["energy"],
        outofplane_angle=prm["outofplane_angle"],
        inplane_angle=prm["inplane_angle"],
        tilt_angle=tilt_angle,
        rocking_angle=prm["rocking_angle"],
        distance=prm["sdd"],
        sample_offsets=prm["sample_offsets"],
        actuators=prm["actuators"],
        custom_scan=prm["custom_scan"],
        custom_motors=prm["custom_motors"],
    )

    ########################################
    # Initialize the paths and the logfile #
    ########################################
    setup.init_paths(
        sample_name=sample_name,
        scan_number=scan,
        root_folder=root_folder,
        save_dir=prm["save_dir"],
        specfile_name=prm["specfile_name"],
        template_imagefile=prm["template_imagefile"],
    )

    logfile = setup.create_logfile(
        scan_number=scan, root_folder=root_folder, filename=detector.specfile
    )

    #########################################################
    # get the motor position of goniometer circles which    #
    # are below the rocking angle (e.g., chi for eta/omega) #
    #########################################################
    _, setup.grazing_angle, _, _ = setup.diffractometer.goniometer_values(
        logfile=logfile, scan_number=scan, setup=setup
    )

    ###################
    # print instances #
    ###################
    print(f'{"#"*(5+len(str(scan)))}\nScan {scan}\n{"#"*(5+len(str(scan)))}')
    print("\n##############\nSetup instance\n##############")
    pretty.pprint(setup.params)
    print("\n#################\nDetector instance\n#################")
    pretty.pprint(detector.params)

    ################
    # preload data #
    ################
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilenames(
        initialdir=detector.scandir,
        filetypes=[
            ("NPZ", "*.npz"),
            ("NPY", "*.npy"),
            ("CXI", "*.cxi"),
            ("HDF5", "*.h5"),
        ],
    )
    nbfiles = len(file_path)
    plt.ion()

    obj, extension = util.load_file(file_path[0])
    if extension == ".h5":
        comment = comment + "_mode"

    print("\n###############\nProcessing data\n###############")
    nz, ny, nx = obj.shape
    print("Initial data size: (", nz, ",", ny, ",", nx, ")")
    if len(original_size) == 0:
        original_size = obj.shape
    print("FFT size before accounting for phasing_binning", original_size)
    original_size = tuple(
        [
            original_size[index] // phasing_binning[index]
            for index in range(len(phasing_binning))
        ]
    )
    print("Binning used during phasing:", detector.binning)
    print("Padding back to original FFT size", original_size)
    obj = util.crop_pad(array=obj, output_shape=original_size)

    ###########################################################################
    # define range for orthogonalization and plotting - speed up calculations #
    ###########################################################################
    zrange, yrange, xrange = pu.find_datarange(
        array=obj, amplitude_threshold=0.05, keep_size=prm["keep_size"]
    )

    numz = zrange * 2
    numy = yrange * 2
    numx = xrange * 2
    print(
        f"Data shape used for orthogonalization and plotting: ({numz}, {numy}, {numx})"
    )

    ####################################################################################
    # find the best reconstruction from the list, based on mean amplitude and variance #
    ####################################################################################
    if nbfiles > 1:
        print(
            "\nTrying to find the best reconstruction\nSorting by ", prm["sort_method"]
        )
        sorted_obj = pu.sort_reconstruction(
            file_path=file_path,
            amplitude_threshold=isosurface_strain,
            data_range=(zrange, yrange, xrange),
            sort_method="variance/mean",
        )
    else:
        sorted_obj = [0]

    #######################################
    # load reconstructions and average it #
    #######################################
    avg_obj = np.zeros((numz, numy, numx))
    ref_obj = np.zeros((numz, numy, numx))
    avg_counter = 1
    print("\nAveraging using", nbfiles, "candidate reconstructions")
    for counter, value in enumerate(sorted_obj):
        obj, extension = util.load_file(file_path[value])
        print("\nOpening ", file_path[value])
        prm[f"from_file_{counter}"] = file_path[value]

        if prm["flip_reconstruction"]:
            obj = pu.flip_reconstruction(obj, debugging=True)

        if extension == ".h5":
            centering_method = "do_nothing"  # do not center, data is already cropped
            # just on support for mode decomposition
            # correct a roll after the decomposition into modes in PyNX
            obj = np.roll(obj, prm["roll_modes"], axis=(0, 1, 2))
            fig, _, _ = gu.multislices_plot(
                abs(obj),
                sum_frames=True,
                plot_colorbar=True,
                title="1st mode after centering",
            )
            fig.waitforbuttonpress()
            plt.close(fig)
        # use the range of interest defined above
        obj = util.crop_pad(obj, [2 * zrange, 2 * yrange, 2 * xrange], debugging=False)

        # align with average reconstruction
        if counter == 0:  # the fist array loaded will serve as reference object
            print("This reconstruction will be used as reference.")
            ref_obj = obj

        avg_obj, flag_avg = reg.average_arrays(
            avg_obj=avg_obj,
            ref_obj=ref_obj,
            obj=obj,
            support_threshold=0.25,
            correlation_threshold=prm["threshold_avg"],
            aligning_option="dft",
            space=prm["averaging_space"],
            reciprocal_space=False,
            is_orthogonal=is_orthogonal,
            debugging=debug,
        )
        avg_counter = avg_counter + flag_avg

    avg_obj = avg_obj / avg_counter
    if avg_counter > 1:
        print("\nAverage performed over ", avg_counter, "reconstructions\n")
    del obj, ref_obj
    gc.collect()

    ################
    # unwrap phase #
    ################
    phase, extent_phase = pu.unwrap(
        avg_obj,
        support_threshold=prm["threshold_unwrap_refraction"],
        debugging=debug,
        reciprocal_space=False,
        is_orthogonal=is_orthogonal,
    )

    print(
        "Extent of the phase over an extended support (ceil(phase range)) ~ ",
        int(extent_phase),
        "(rad)",
    )
    phase = util.wrap(phase, start_angle=-extent_phase / 2, range_angle=extent_phase)
    if debug:
        gu.multislices_plot(
            phase,
            width_z=2 * zrange,
            width_y=2 * yrange,
            width_x=2 * xrange,
            plot_colorbar=True,
            title="Phase after unwrap + wrap",
            reciprocal_space=False,
            is_orthogonal=is_orthogonal,
        )

    #############################################
    # phase ramp removal before phase filtering #
    #############################################
    amp, phase, rampz, rampy, rampx = pu.remove_ramp(
        amp=abs(avg_obj),
        phase=phase,
        initial_shape=original_size,
        method="gradient",
        amplitude_threshold=isosurface_strain,
        threshold_gradient=prm["threshold_gradient"],
    )
    del avg_obj
    gc.collect()

    if debug:
        gu.multislices_plot(
            phase,
            width_z=2 * zrange,
            width_y=2 * yrange,
            width_x=2 * xrange,
            plot_colorbar=True,
            title="Phase after ramp removal",
            reciprocal_space=False,
            is_orthogonal=is_orthogonal,
        )

    ########################
    # phase offset removal #
    ########################
    support = np.zeros(amp.shape)
    support[amp > isosurface_strain * amp.max()] = 1
    phase = pu.remove_offset(
        array=phase,
        support=support,
        offset_method=prm["offset_method"],
        phase_offset=prm["phase_offset"],
        offset_origin=prm["phase_offset_origin"],
        title="Phase",
        debugging=debug,
    )
    del support
    gc.collect()

    phase = util.wrap(
        obj=phase, start_angle=-extent_phase / 2, range_angle=extent_phase
    )

    ##############################################################################
    # average the phase over a window or apodize to reduce noise in strain plots #
    ##############################################################################
    half_width_avg_phase = prm["half_width_avg_phase"]
    if half_width_avg_phase != 0:
        bulk = pu.find_bulk(
            amp=amp, support_threshold=isosurface_strain, method="threshold"
        )
        # the phase should be averaged only in the support defined by the isosurface
        phase = pu.mean_filter(
            array=phase, support=bulk, half_width=half_width_avg_phase
        )
        del bulk
        gc.collect()

    if half_width_avg_phase != 0:
        comment = comment + "_avg" + str(2 * half_width_avg_phase + 1)

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
            is_orthogonal=is_orthogonal,
            debugging=True,
        )
        comment = comment + "_apodize_" + prm["apodization_window"]

    ################################################################
    # save the phase with the ramp for PRTF calculations,          #
    # otherwise the object will be misaligned with the measurement #
    ################################################################
    np.savez_compressed(
        detector.savedir + "S" + str(scan) + "_avg_obj_prtf" + comment,
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
    if centering_method == "max":
        avg_obj = pu.center_max(avg_obj)
        # shift based on max value,
        # required if it spans across the edge of the array before COM
    elif centering_method == "com":
        avg_obj = pu.center_com(avg_obj)
    elif centering_method == "max_com":
        avg_obj = pu.center_max(avg_obj)
        avg_obj = pu.center_com(avg_obj)

    #######################
    #  save support & vti #
    #######################
    if prm["save_support"]:
        # to be used as starting support in phasing, hence still in the detector frame
        support = np.zeros((numz, numy, numx))
        support[abs(avg_obj) / abs(avg_obj).max() > 0.01] = 1
        # low threshold because support will be cropped by shrinkwrap during phasing
        np.savez_compressed(
            detector.savedir + "S" + str(scan) + "_support" + comment, obj=support
        )
        del support
        gc.collect()

    if prm["save_rawdata"]:
        np.savez_compressed(
            detector.savedir + "S" + str(scan) + "_raw_amp-phase" + comment,
            amp=abs(avg_obj),
            phase=np.angle(avg_obj),
        )

        # voxel sizes in the detector frame
        voxel_z, voxel_y, voxel_x = setup.voxel_sizes_detector(
            array_shape=original_size,
            tilt_angle=tilt_angle,
            pixel_x=detector.pixelsize_x,
            pixel_y=detector.pixelsize_y,
            verbose=True,
        )
        # save raw amp & phase to VTK
        # in VTK, x is downstream, y vertical, z inboard,
        # thus need to flip the last axis
        gu.save_to_vti(
            filename=os.path.join(
                detector.savedir, "S" + str(scan) + "_raw_amp-phase" + comment + ".vti"
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
        test_vector=axis_to_array_xyz[ref_axis_q],
    )
    print(
        f"\nNormalized diffusion vector in the laboratory frame (z*, y*, x*): "
        f"({q_lab[0]:.4f} 1/A, {q_lab[1]:.4f} 1/A, {q_lab[2]:.4f} 1/A)"
    )

    planar_dist = 2 * np.pi / qnorm  # qnorm should be in angstroms
    print(f"Wavevector transfer: {qnorm:.4f} 1/A")
    print(f"Atomic planar distance: {planar_dist:.4f} A")
    print(f"\nAngle between q_lab and {ref_axis_q} = {angle:.2f} deg")
    if debug:
        print(
            "Angle with y in zy plane = "
            f"{np.arctan(q_lab[0]/q_lab[1])*180/np.pi:.2f} deg"
        )
        print(
            "Angle with y in xy plane = "
            f"{np.arctan(-q_lab[2]/q_lab[1])*180/np.pi:.2f} deg"
        )
        print(
            "Angle with z in xz plane = "
            f"{180+np.arctan(q_lab[2]/q_lab[0])*180/np.pi:.2f} deg\n"
        )

    planar_dist = planar_dist / 10  # switch to nm

    #######################
    #  orthogonalize data #
    #######################
    print("\nShape before orthogonalization", avg_obj.shape)
    if data_frame == "detector":
        if debug:
            phase, _ = pu.unwrap(
                avg_obj,
                support_threshold=prm["threshold_unwrap_refraction"],
                debugging=True,
                reciprocal_space=False,
                is_orthogonal=False,
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
            )
            del phase
            gc.collect()

        obj_ortho, voxel_size = setup.ortho_directspace(
            arrays=avg_obj,
            q_com=np.array([q_lab[2], q_lab[1], q_lab[0]]),
            initial_shape=original_size,
            voxel_size=fix_voxel,
            reference_axis=axis_to_array_xyz[ref_axis_q],
            fill_value=0,
            debugging=True,
            title="amplitude",
        )

    else:  # data already orthogonalized using xrayutilities
        # or the linearized transformation matrix
        obj_ortho = avg_obj
        try:
            print("Select the file containing QxQzQy")
            file_path = filedialog.askopenfilename(
                title="Select the file containing QxQzQy",
                initialdir=detector.savedir,
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
        print(
            f"direct space voxel size from q values: ({dz_real:.2f} nm,"
            f" {dy_real:.2f} nm, {dx_real:.2f} nm)"
        )
        if fix_voxel:
            voxel_size = fix_voxel
            print(f"Direct space pixel size for the interpolation: {voxel_size} (nm)")
            print("Interpolating...\n")
            obj_ortho = pu.regrid(
                array=obj_ortho,
                old_voxelsize=(dz_real, dy_real, dx_real),
                new_voxelsize=voxel_size,
            )
        else:
            # no need to interpolate
            voxel_size = dz_real, dy_real, dx_real  # in nm

        if (
            data_frame == "laboratory"
        ):  # the object must be rotated into the crystal frame
            # before the strain calculation
            print("Rotating the object in the crystal frame for the strain calculation")

            amp, phase = util.rotate_crystal(
                arrays=(abs(obj_ortho), np.angle(obj_ortho)),
                is_orthogonal=True,
                reciprocal_space=False,
                voxel_size=voxel_size,
                debugging=(True, False),
                axis_to_align=q_lab[::-1],
                reference_axis=axis_to_array_xyz[ref_axis_q],
                title=("amp", "phase"),
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
    print("\nCentering the crystal")
    obj_ortho = pu.center_com(obj_ortho)

    ####################
    # Phase unwrapping #
    ####################
    print("\nPhase unwrapping")
    phase, extent_phase = pu.unwrap(
        obj_ortho,
        support_threshold=prm["threshold_unwrap_refraction"],
        debugging=True,
        reciprocal_space=False,
        is_orthogonal=True,
    )
    amp = abs(obj_ortho)
    del obj_ortho
    gc.collect()

    #############################################
    # invert phase: -1*phase = displacement * q #
    #############################################
    if invert_phase:
        phase = -1 * phase

    ########################################
    # refraction and absorption correction #
    ########################################
    if correct_refraction:  # or correct_absorption:
        bulk = pu.find_bulk(
            amp=amp,
            support_threshold=prm["threshold_unwrap_refraction"],
            method=prm["optical_path_method"],
            debugging=debug,
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
            axis_to_align=axis_to_array_xyz[ref_axis_q],
            reference_axis=[q_lab[2], q_lab[1], q_lab[0]],
        )
        kout = util.rotate_vector(
            vectors=[kout[2], kout[1], kout[0]],
            axis_to_align=axis_to_array_xyz[ref_axis_q],
            reference_axis=[q_lab[2], q_lab[1], q_lab[0]],
        )

        # calculate the optical path of the incoming wavevector
        path_in = pu.get_opticalpath(
            support=bulk, direction="in", k=kin, debugging=debug
        )  # path_in already in nm

        # calculate the optical path of the outgoing wavevector
        path_out = pu.get_opticalpath(
            support=bulk, direction="out", k=kout, debugging=debug
        )  # path_our already in nm

        optical_path = path_in + path_out
        del path_in, path_out
        gc.collect()

        if correct_refraction:
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
            )

        del bulk, optical_path
        gc.collect()

    ##############################################
    # phase ramp and offset removal (mean value) #
    ##############################################
    print("\nPhase ramp removal")
    amp, phase, _, _, _ = pu.remove_ramp(
        amp=amp,
        phase=phase,
        initial_shape=original_size,
        method=prm["phase_ramp_removal"],
        amplitude_threshold=isosurface_strain,
        threshold_gradient=prm["threshold_gradient"],
        debugging=debug,
    )

    ########################
    # phase offset removal #
    ########################
    print("\nPhase offset removal")
    support = np.zeros(amp.shape)
    support[amp > isosurface_strain * amp.max()] = 1
    phase = pu.remove_offset(
        array=phase,
        support=support,
        offset_method=prm["offset_method"],
        phase_offset=prm["phase_offset"],
        offset_origin=prm["phase_offset_origin"],
        title="Orthogonal phase",
        debugging=debug,
        reciprocal_space=False,
        is_orthogonal=True,
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
    print(f"\nCalculation of the strain along {ref_axis_q}")
    strain = pu.get_strain(
        phase=phase,
        planar_distance=planar_dist,
        voxel_size=voxel_size,
        reference_axis=ref_axis_q,
        extent_phase=extent_phase,
        method=prm["strain_method"],
        debugging=debug,
    )

    ################################################
    # optionally rotates back the crystal into the #
    # laboratory frame (for debugging purpose)     #
    ################################################
    q_final = None
    if save_frame in {"laboratory", "lab_flat_sample"}:
        comment = comment + "_labframe"
        print("\nRotating back the crystal in laboratory frame")
        amp, phase, strain = util.rotate_crystal(
            arrays=(amp, phase, strain),
            axis_to_align=axis_to_array_xyz[ref_axis_q],
            voxel_size=voxel_size,
            is_orthogonal=True,
            reciprocal_space=False,
            reference_axis=[q_lab[2], q_lab[1], q_lab[0]],
            debugging=(True, False, False),
            title=("amp", "phase", "strain"),
        )
        # q_lab is already in the laboratory frame
        q_final = q_lab

    if save_frame == "lab_flat_sample":
        comment = comment + "_flat"
        print("\nSending sample stage circles to 0")
        sample_angles = setup.diffractometer.goniometer_values(
            logfile=logfile, scan_number=scan, setup=setup, stage_name="sample"
        )
        (amp, phase, strain), q_final = setup.diffractometer.flatten_sample(
            arrays=(amp, phase, strain),
            voxel_size=voxel_size,
            angles=sample_angles,
            q_com=q_lab[::-1],  # q_com needs to be in xyz order
            is_orthogonal=True,
            reciprocal_space=False,
            rocking_angle=prm["rocking_angle"],
            debugging=(True, False, False),
            title=("amp", "phase", "strain"),
        )
    if save_frame == "crystal":
        # rotate also q_lab to have it along ref_axis_q,
        # as a cross-checkm, vectors needs to be in xyz order
        comment = comment + "_crystalframe"
        q_final = util.rotate_vector(
            vectors=q_lab[::-1],
            axis_to_align=axis_to_array_xyz[ref_axis_q],
            reference_axis=q_lab[::-1],
        )

    ###############################################
    # rotates the crystal e.g. for easier slicing #
    # of the result along a particular direction  #
    ###############################################
    # typically this is an inplane rotation, q should stay aligned with the axis
    # along which the strain was calculated
    if prm["align_axis"]:
        print("\nRotating arrays for visualization")
        amp, phase, strain = util.rotate_crystal(
            arrays=(amp, phase, strain),
            reference_axis=axis_to_array_xyz[prm["ref_axis"]],
            axis_to_align=prm["axis_to_align"],
            voxel_size=voxel_size,
            debugging=(True, False, False),
            is_orthogonal=True,
            reciprocal_space=False,
            title=("amp", "phase", "strain"),
        )
        # rotate q accordingly, vectors needs to be in xyz order
        q_final = util.rotate_vector(
            vectors=q_final[::-1],
            axis_to_align=axis_to_array_xyz[prm["ref_axis"]],
            reference_axis=prm["axis_to_align"],
        )

    print(
        f"\nq_final = ({q_final[0]:.4f} 1/A,"
        f" {q_final[1]:.4f} 1/A, {q_final[2]:.4f} 1/A)"
    )

    ##############################################
    # pad array to fit the output_size parameter #
    ##############################################
    if output_size is not None:
        amp = util.crop_pad(array=amp, output_shape=output_size)
        phase = util.crop_pad(array=phase, output_shape=output_size)
        strain = util.crop_pad(array=strain, output_shape=output_size)
    print(f"\nFinal data shape: {amp.shape}")

    ######################
    # save result to vtk #
    ######################
    print(
        f"\nVoxel size: ({voxel_size[0]:.2f} nm, {voxel_size[1]:.2f} nm,"
        f" {voxel_size[2]:.2f} nm)"
    )
    bulk = pu.find_bulk(
        amp=amp, support_threshold=isosurface_strain, method="threshold"
    )
    if save:
        prm["comment"] = comment
        np.savez_compressed(
            f"{detector.savedir}S{scan}_amp{phase_fieldname}strain{comment}",
            amp=amp,
            phase=phase,
            bulk=bulk,
            strain=strain,
            q_com=q_final,
            voxel_sizes=voxel_size,
            detector=detector.params,
            setup=setup.params,
            params=prm,
        )

        # save results in hdf5 file
        with h5py.File(
            f"{detector.savedir}S{scan}_amp{phase_fieldname}strain{comment}.h5", "w"
        ) as hf:
            out = hf.create_group("output")
            par = hf.create_group("params")
            out.create_dataset("amp", data=amp)
            out.create_dataset("bulk", data=bulk)
            out.create_dataset("phase", data=phase)
            out.create_dataset("strain", data=strain)
            out.create_dataset("q_com", data=q_final)
            out.create_dataset("voxel_sizes", data=voxel_size)
            par.create_dataset("detector", data=str(detector.params))
            par.create_dataset("setup", data=str(setup.params))
            par.create_dataset("parameters", data=str(prm))

        # save amp & phase to VTK
        # in VTK, x is downstream, y vertical, z inboard,
        # thus need to flip the last axis
        gu.save_to_vti(
            filename=os.path.join(
                detector.savedir,
                "S"
                + str(scan)
                + "_amp-"
                + phase_fieldname
                + "-strain"
                + comment
                + ".vti",
            ),
            voxel_size=voxel_size,
            tuple_array=(amp, bulk, phase, strain),
            tuple_fieldnames=("amp", "bulk", phase_fieldname, "strain"),
            amplitude_threshold=0.01,
        )

    ######################################
    # estimate the volume of the crystal #
    ######################################
    amp = amp / amp.max()
    temp_amp = np.copy(amp)
    temp_amp[amp < isosurface_strain] = 0
    temp_amp[np.nonzero(temp_amp)] = 1
    volume = temp_amp.sum() * reduce(lambda x, y: x * y, voxel_size)  # in nm3
    del temp_amp
    gc.collect()

    ##############################
    # plot slices of the results #
    ##############################
    pixel_spacing = [tick_spacing / vox for vox in voxel_size]
    print(
        "\nPhase extent without / with thresholding the modulus "
        f"(threshold={isosurface_strain}): {phase.max()-phase.min():.2f} rad, "
        f"{phase[np.nonzero(bulk)].max()-phase[np.nonzero(bulk)].min():.2f} rad"
    )
    piz, piy, pix = np.unravel_index(phase.argmax(), phase.shape)
    print(
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
        tuple_title=("phase at max in xy", "phase at max in xz", "phase at max in yz"),
        tuple_scale="linear",
        cmap=my_cmap,
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
    )
    fig.text(0.60, 0.45, "Scan " + str(scan), size=20)
    fig.text(
        0.60,
        0.40,
        "Bulk - isosurface=" + str("{:.2f}".format(isosurface_strain)),
        size=20,
    )
    plt.pause(0.1)
    if save:
        plt.savefig(detector.savedir + "S" + str(scan) + "_bulk" + comment + ".png")

    # amplitude
    fig, _, _ = gu.multislices_plot(
        amp,
        sum_frames=False,
        title="Normalized orthogonal amp",
        vmin=0,
        vmax=1,
        tick_direction=tick_direction,
        tick_width=tick_width,
        tick_length=tick_length,
        pixel_spacing=pixel_spacing,
        plot_colorbar=True,
        is_orthogonal=True,
        reciprocal_space=False,
    )
    fig.text(0.60, 0.45, f"Scan {scan}", size=20)
    fig.text(
        0.60,
        0.40,
        f"Voxel size=({voxel_size[0]:.1f}, {voxel_size[1]:.1f}, "
        f"{voxel_size[2]:.1f}) (nm)",
        size=20,
    )
    fig.text(0.60, 0.35, f"Ticks spacing={tick_spacing} nm", size=20)
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
    if save:
        plt.savefig(detector.savedir + f"S{scan}_amp" + comment + ".png")

    # amplitude histogram
    fig, ax = plt.subplots(1, 1)
    ax.hist(amp[amp > 0.05 * amp.max()].flatten(), bins=250)
    ax.set_ylim(bottom=1)
    ax.tick_params(
        labelbottom=True,
        labelleft=True,
        direction="out",
        length=tick_length,
        width=tick_width,
    )
    ax.spines["right"].set_linewidth(1.5)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["top"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    fig.savefig(detector.savedir + f"S{scan}_histo_amp" + comment + ".png")

    # phase
    fig, _, _ = gu.multislices_plot(
        phase,
        sum_frames=False,
        title="Orthogonal displacement",
        vmin=-prm["phase_range"],
        vmax=prm["phase_range"],
        tick_direction=tick_direction,
        cmap=my_cmap,
        tick_width=tick_width,
        tick_length=tick_length,
        pixel_spacing=pixel_spacing,
        plot_colorbar=True,
        is_orthogonal=True,
        reciprocal_space=False,
    )
    fig.text(0.60, 0.30, f"Scan {scan}", size=20)
    fig.text(
        0.60,
        0.25,
        f"Voxel size=({voxel_size[0]:.1f}, {voxel_size[1]:.1f}, "
        f"{voxel_size[2]:.1f}) (nm)",
        size=20,
    )
    fig.text(0.60, 0.20, f"Ticks spacing={tick_spacing} nm", size=20)
    fig.text(0.60, 0.15, f"average over {avg_counter} reconstruction(s)", size=20)
    if half_width_avg_phase > 0:
        fig.text(
            0.60, 0.10, f"Averaging over {2*half_width_avg_phase+1} pixels", size=20
        )
    else:
        fig.text(0.60, 0.10, "No phase averaging", size=20)
    if save:
        plt.savefig(detector.savedir + f"S{scan}_displacement" + comment + ".png")

    # strain
    fig, _, _ = gu.multislices_plot(
        strain,
        sum_frames=False,
        title="Orthogonal strain",
        vmin=-prm["strain_range"],
        vmax=prm["strain_range"],
        tick_direction=tick_direction,
        tick_width=tick_width,
        tick_length=tick_length,
        plot_colorbar=True,
        cmap=my_cmap,
        pixel_spacing=pixel_spacing,
        is_orthogonal=True,
        reciprocal_space=False,
    )
    fig.text(0.60, 0.30, f"Scan {scan}", size=20)
    fig.text(
        0.60,
        0.25,
        f"Voxel size=({voxel_size[0]:.1f}, "
        f"{voxel_size[1]:.1f}, {voxel_size[2]:.1f}) (nm)",
        size=20,
    )
    fig.text(0.60, 0.20, f"Ticks spacing={tick_spacing} nm", size=20)
    fig.text(0.60, 0.15, f"average over {avg_counter} reconstruction(s)", size=20)
    if half_width_avg_phase > 0:
        fig.text(
            0.60, 0.10, f"Averaging over {2*half_width_avg_phase+1} pixels", size=20
        )
    else:
        fig.text(0.60, 0.10, "No phase averaging", size=20)
    if save:
        plt.savefig(detector.savedir + f"S{scan}_strain" + comment + ".png")


if __name__ == "__main__":
    # construct the argument parser and parse the command-line arguments
    ap = argparse.ArgumentParser()
    cli_args = vars(ap.parse_args())
    # print arguments
    print("Arguments parsed:")
    for k, v in cli_args.items():
        print("argument: {}".format(k), "value: {}".format(v))

    # load the config file
    config_file = "../../conf/config_postprocessing.yml"
    parser = ConfigParser(config_file, cli_args)
    args = parser.load_arguments()
    args["time"] = f"{datetime.now()}"
    run(prm=args)

    print("\nEnd of script")
    plt.ioff()
    plt.show()
