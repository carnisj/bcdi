# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, jerome.carnis@esrf.fr
"""Workflow for BCDI data preprocessing of a single scan, before phase retrieval."""

import gc

try:
    import hdf5plugin  # for P10, should be imported before h5py or PyTables
except ModuleNotFoundError:
    pass

import h5py
import logging
from logging import Logger
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import scipy.signal  # for medfilt2d
from scipy.ndimage.measurements import center_of_mass
from scipy.io import savemat
import tkinter as tk
from tkinter import filedialog
from typing import Any, Dict, List, Optional, Tuple
import xrayutilities as xu

import bcdi.graph.graph_utils as gu
from bcdi.experiment.setup import Setup
import bcdi.postprocessing.postprocessing_utils as pu
import bcdi.preprocessing.bcdi_utils as bu
from bcdi.utils.constants import AXIS_TO_ARRAY
from bcdi.utils.snippets_logging import FILE_FORMATTER
import bcdi.utils.utilities as util

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

    def on_click(event):
        """
        Interact with a plot, return the position of clicked pixel.

        If flag_pause==1 or if the mouse is out of plot axes, it will not register
        the click.

        :param event: mouse click event
        """
        nonlocal xy, flag_pause, previous_axis

        if not event.inaxes:
            return
        if not flag_pause:

            if (previous_axis == event.inaxes) or (
                previous_axis is None
            ):  # collect points
                _x, _y = int(np.rint(event.xdata)), int(np.rint(event.ydata))
                xy.append([_x, _y])
                if previous_axis is None:
                    previous_axis = event.inaxes
            else:  # the click is not in the same subplot, restart collecting points
                print(
                    "Select mask polygon vertices within "
                    "the same subplot: restart masking..."
                )
                xy = []
                previous_axis = None

    def press_key(event):
        """
        Interact with a plot for masking parasitic intensity or detector gaps.

        :param event: button press event
        """
        nonlocal original_data, original_mask, updated_mask, data, mask, frame_index
        nonlocal flag_aliens, flag_mask, flag_pause, xy, fig_mask, max_colorbar
        nonlocal ax0, ax1, ax2, ax3, previous_axis, info_text, my_cmap, width

        try:
            if event.inaxes == ax0:
                dim = 0
                inaxes = True
            elif event.inaxes == ax1:
                dim = 1
                inaxes = True
            elif event.inaxes == ax2:
                dim = 2
                inaxes = True
            else:
                dim = -1
                inaxes = False

            if inaxes:
                if flag_aliens:
                    (
                        data,
                        mask,
                        width,
                        max_colorbar,
                        frame_index,
                        stop_masking,
                    ) = gu.update_aliens_combined(
                        key=event.key,
                        pix=int(np.rint(event.xdata)),
                        piy=int(np.rint(event.ydata)),
                        original_data=original_data,
                        original_mask=original_mask,
                        updated_data=data,
                        updated_mask=mask,
                        axes=(ax0, ax1, ax2, ax3),
                        width=width,
                        dim=dim,
                        frame_index=frame_index,
                        vmin=0,
                        vmax=max_colorbar,
                        cmap=my_cmap,
                        invert_yaxis=not prm["use_rawdata"],
                    )
                elif flag_mask:
                    if previous_axis == ax0:
                        click_dim = 0
                        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
                        points = np.stack((x.flatten(), y.flatten()), axis=0).T
                    elif previous_axis == ax1:
                        click_dim = 1
                        x, y = np.meshgrid(np.arange(nx), np.arange(nz))
                        points = np.stack((x.flatten(), y.flatten()), axis=0).T
                    elif previous_axis == ax2:
                        click_dim = 2
                        x, y = np.meshgrid(np.arange(ny), np.arange(nz))
                        points = np.stack((x.flatten(), y.flatten()), axis=0).T
                    else:
                        click_dim = None
                        points = None

                    (
                        data,
                        updated_mask,
                        flag_pause,
                        xy,
                        width,
                        max_colorbar,
                        click_dim,
                        stop_masking,
                        info_text,
                    ) = gu.update_mask_combined(
                        key=event.key,
                        pix=int(np.rint(event.xdata)),
                        piy=int(np.rint(event.ydata)),
                        original_data=original_data,
                        original_mask=mask,
                        updated_data=data,
                        updated_mask=updated_mask,
                        axes=(ax0, ax1, ax2, ax3),
                        flag_pause=flag_pause,
                        points=points,
                        xy=xy,
                        width=width,
                        dim=dim,
                        click_dim=click_dim,
                        info_text=info_text,
                        vmin=0,
                        vmax=max_colorbar,
                        cmap=my_cmap,
                        invert_yaxis=not prm["use_rawdata"],
                    )
                    if click_dim is None:
                        previous_axis = None
                else:
                    stop_masking = False

                if stop_masking:
                    plt.close("all")

        except AttributeError:  # mouse pointer out of axes
            pass

    ####################################################
    # Initialize parameters for the callback functions #
    ####################################################
    flag_mask = False
    flag_aliens = False
    my_cmap = prm["colormap"].cmap
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
        Path(prm["root_folder"])
        / f"run{scan_idx}_{prm['sample_name'][scan_idx]}{scan_nb}.log"
    )
    filehandler = logging.FileHandler(tmpfile, mode="w", encoding="utf-8")
    filehandler.setFormatter(FILE_FORMATTER)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(filehandler)
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
        rocking_angle=prm["rocking_angle"],
        distance=prm["detector_distance"],
        beam_direction=prm["beam_direction"],
        sample_inplane=prm["sample_inplane"],
        sample_outofplane=prm["sample_outofplane"],
        offset_inplane=prm["offset_inplane"],
        custom_scan=prm["custom_scan"],
        custom_images=prm["custom_images"],
        sample_offsets=prm["sample_offsets"],
        custom_monitor=prm["custom_monitor"],
        custom_motors=prm["custom_motors"],
        actuators=prm["actuators"],
        is_series=prm["is_series"],
        outofplane_angle=prm["outofplane_angle"],
        inplane_angle=prm["inplane_angle"],
        dirbeam_detector_angles=prm["dirbeam_detector_angles"],
        direct_beam=prm["direct_beam"],
        detector_name=prm["detector"],
        template_imagefile=prm["template_imagefile"][scan_idx],
        roi=prm["roi_detector"],
        binning=prm["phasing_binning"],
        preprocessing_binning=prm["preprocessing_binning"],
        linearity_func=prm["linearity_func"],
    )

    # initialize the paths
    setup.init_paths(
        sample_name=prm["sample_name"][scan_idx],
        scan_number=scan_nb,
        data_dir=prm["data_dir"][scan_idx],
        root_folder=prm["root_folder"],
        save_dir=prm["save_dir"][scan_idx],
        save_dirname=prm["save_dirname"],
        specfile_name=prm["specfile_name"][scan_idx],
        template_imagefile=prm["template_imagefile"][scan_idx],
    )

    logfile = setup.create_logfile(
        scan_number=scan_nb,
        root_folder=prm["root_folder"],
        filename=setup.detector.specfile,
    )

    # load the goniometer positions needed for the calculation of the corrected
    # detector angles
    setup.read_logfile(scan_number=scan_nb)

    ###################
    # print instances #
    ###################
    logger.info(f"##############\nSetup instance\n##############\n{setup.params}")
    logger.info(
        "#################\nDetector instance\n#################\n"
        f"{setup.detector.params}"
    )

    if not prm["use_rawdata"]:
        comment += "_ortho"
        if prm["interpolation_method"] == "linearization":
            comment += "_lin"
            # load the goniometer positions needed in the calculation
            # of the transformation matrix
            setup.read_logfile(scan_number=scan_nb)
        else:  # 'xrayutilities'
            comment += "_xrutil"
    if prm["normalize_flux"]:
        comment = comment + "_norm"

    #############
    # Load data #
    #############
    if prm["reload_previous"]:  # resume previous masking
        logger.info("Resuming previous masking")
        file_path = filedialog.askopenfilename(
            initialdir=setup.detector.scandir,
            title="Select data file",
            filetypes=[("NPZ", "*.npz")],
        )
        data = np.load(file_path)
        npz_key = data.files
        data = data[npz_key[0]]
        nz, ny, nx = np.shape(data)

        # check that the ROI is correctly defined
        setup.detector.roi = prm["roi_detector"] or [0, ny, 0, nx]
        logger.info(f"Detector ROI: {setup.detector.roi}")
        # update savedir to save the data in the same directory as the reloaded data
        if not prm["save_dir"]:
            setup.detector.savedir = os.path.dirname(file_path) + "/"
            logger.info(f"Updated saving directory: {setup.detector.savedir}")

        file_path = filedialog.askopenfilename(
            initialdir=os.path.dirname(file_path) + "/",
            title="Select mask file",
            filetypes=[("NPZ", "*.npz")],
        )
        mask = np.load(file_path)
        npz_key = mask.files
        mask = mask[npz_key[0]]

        if prm["reload_orthogonal"]:
            # the data is gridded in the orthonormal laboratory frame
            prm["use_rawdata"] = False
            try:
                file_path = filedialog.askopenfilename(
                    initialdir=setup.detector.savedir,
                    title="Select q values",
                    filetypes=[("NPZ", "*.npz")],
                )
                reload_qvalues = np.load(file_path)
                q_values = [
                    reload_qvalues["qx"],
                    reload_qvalues["qz"],
                    reload_qvalues["qy"],
                ]
            except FileNotFoundError:
                q_values = []

            prm["normalize_flux"] = "skip"
            # we assume that normalization was already performed
            monitor = []  # we assume that normalization was already performed
            prm["center_fft"] = "skip"
            # we assume that crop/pad/centering was already performed
            prm[
                "fix_size"
            ] = []  # we assume that crop/pad/centering was already performed

            # bin data and mask if needed
            if (
                (setup.detector.binning[0] != 1)
                or (setup.detector.binning[1] != 1)
                or (setup.detector.binning[2] != 1)
            ):
                logger.info(
                    f"Binning the reloaded orthogonal data by {setup.detector.binning}"
                )
                data = util.bin_data(
                    data,
                    binning=setup.detector.binning,
                    debugging=False,
                    cmap=prm["colormap"].cmap,
                )
                mask = util.bin_data(
                    mask,
                    binning=setup.detector.binning,
                    debugging=False,
                    cmap=prm["colormap"].cmap,
                )
                mask[np.nonzero(mask)] = 1
                if len(q_values) != 0:
                    qx = q_values[0]
                    qz = q_values[1]
                    qy = q_values[2]
                    numz, numy, numx = len(qx), len(qz), len(qy)
                    qx = qx[
                        : numz
                        - (numz % setup.detector.binning[0]) : setup.detector.binning[0]
                    ]  # along z downstream
                    qz = qz[
                        : numy
                        - (numy % setup.detector.binning[1]) : setup.detector.binning[1]
                    ]  # along y vertical
                    qy = qy[
                        : numx
                        - (numx % setup.detector.binning[2]) : setup.detector.binning[2]
                    ]  # along x outboard
                    del numz, numy, numx
        else:  # the data is in the detector frame
            data, mask, frames_logical, monitor = bu.reload_bcdi_data(
                logfile=logfile,
                scan_number=scan_nb,
                data=data,
                mask=mask,
                setup=setup,
                debugging=prm["debug"],
                normalize=prm["normalize_flux"],
                photon_threshold=prm["loading_threshold"],
            )

    else:  # new masking process
        prm["reload_orthogonal"] = False  # the data is in the detector plane
        flatfield = util.load_flatfield(prm["flatfield_file"])
        hotpix_array = util.load_hotpixels(prm["hotpixels_file"])
        background = util.load_background(prm["background_file"])

        data, mask, frames_logical, monitor = bu.load_bcdi_data(
            scan_number=scan_nb,
            setup=setup,
            frames_pattern=prm["frames_pattern"],
            bin_during_loading=prm["bin_during_loading"],
            flatfield=flatfield,
            hotpixels=hotpix_array,
            background=background,
            normalize=prm["normalize_flux"],
            debugging=prm["debug"],
            photon_threshold=prm["loading_threshold"],
        )

    nz, ny, nx = np.shape(data)
    logger.info(f"Input data shape: {np.shape(data)}")

    binning_comment = (
        f"_{setup.detector.preprocessing_binning[0] * setup.detector.binning[0]}"
        f"_{setup.detector.preprocessing_binning[1] * setup.detector.binning[1]}"
        f"_{setup.detector.preprocessing_binning[2] * setup.detector.binning[2]}"
    )

    ##############################################################
    # correct detector angles and save values for postprocessing #
    ##############################################################
    metadata = None
    if not prm["outofplane_angle"] and not prm["inplane_angle"]:
        # corrected detector angles not provided
        if prm["bragg_peak"] is None:
            # Bragg peak position not provided, find it from the data
            prm["bragg_peak"] = bu.find_bragg(
                data=data,
                peak_method="maxcom",
                roi=setup.detector.roi,
                binning=setup.detector.binning,
            )

        if prm["bragg_peak"] is None:
            raise ValueError("bragg_peak undefined")
        roi_center = (
            prm["bragg_peak"][0],
            (prm["bragg_peak"][1] - setup.detector.roi[0]) // setup.detector.binning[1],
            (prm["bragg_peak"][2] - setup.detector.roi[2]) // setup.detector.binning[2],
        )

        metadata = bu.show_rocking_curve(
            data,
            roi_center=roi_center,
            tilt_values=setup.incident_angles,
            savedir=setup.detector.savedir,
        )
        setup.correct_detector_angles(bragg_peak_position=prm["bragg_peak"])
        prm["outofplane_angle"] = setup.outofplane_angle
        prm["inplane_angle"] = setup.inplane_angle

    ####################################
    # wavevector transfer calculations #
    ####################################
    kin = (
        2 * np.pi / setup.wavelength * np.asarray(setup.beam_direction)
    )  # in lab frame z downstream, y vertical, x outboard
    kout = setup.exit_wavevector  # in lab.frame z downstream, y vertical, x outboard
    q = (kout - kin) / 1e10  # convert from 1/m to 1/angstrom
    qnorm = np.linalg.norm(q)
    dist_plane = 2 * np.pi / qnorm
    logger.info(f"Wavevector transfer of Bragg peak: {q}, Qnorm={qnorm:.4f}")
    logger.info(f"Interplanar distance: {dist_plane:.6f} angstroms")

    ##############################################################
    # optional interpolation of the data onto an orthogonal grid #
    ##############################################################
    if not prm["reload_orthogonal"]:
        if prm["save_rawdata"]:
            np.savez_compressed(
                setup.detector.savedir + f"S{scan_nb}" + "_data_before_masking_stack",
                data=data,
            )
            if prm["save_to_mat"]:
                # save to .mat, the new order is x y z
                # (outboard, vertical up, downstream)
                savemat(
                    setup.detector.savedir
                    + "S"
                    + str(scan_nb)
                    + "_data_before_masking_stack.mat",
                    {"data": np.moveaxis(data, [0, 1, 2], [-1, -2, -3])},
                )

        if prm["use_rawdata"]:
            q_values = []
            # binning along axis 0 is done after masking
            data[np.nonzero(mask)] = 0
        else:
            tmp_data = np.copy(
                data
            )  # do not modify the raw data before the interpolation
            tmp_data[mask == 1] = 0
            fig, _, _ = gu.multislices_plot(
                tmp_data,
                sum_frames=True,
                scale="log",
                plot_colorbar=True,
                vmin=0,
                title="Data before gridding\n",
                is_orthogonal=False,
                reciprocal_space=True,
                cmap=prm["colormap"].cmap,
            )
            fig.savefig(
                setup.detector.savedir
                + f"data_before_gridding_S{scan_nb}_{nz}_{ny}_{nx}"
                + binning_comment
                + ".png"
            )
            plt.close(fig)
            del tmp_data
            gc.collect()

            if prm["interpolation_method"] == "xrayutilities":
                qconv, offsets = setup.init_qconversion()
                setup.detector.offsets = offsets
                hxrd = xu.experiment.HXRD(
                    prm["sample_inplane"],
                    prm["sample_outofplane"],
                    en=setup.energy,
                    qconv=qconv,
                )
                # the first 2 arguments in HXRD are the inplane reference direction
                # along the beam and surface normal of the sample

                # Update the direct beam vertical position,
                # take into account the roi and binning
                cch1 = (prm["cch1"] - setup.detector.roi[0]) / (
                    setup.detector.preprocessing_binning[1] * setup.detector.binning[1]
                )
                # Update the direct beam horizontal position,
                # take into account the roi and binning
                cch2 = (prm["cch2"] - setup.detector.roi[2]) / (
                    setup.detector.preprocessing_binning[2] * setup.detector.binning[2]
                )
                # number of pixels after taking into account the roi and binning
                nch1 = (setup.detector.roi[1] - setup.detector.roi[0]) // (
                    setup.detector.preprocessing_binning[1] * setup.detector.binning[1]
                ) + (setup.detector.roi[1] - setup.detector.roi[0]) % (
                    setup.detector.preprocessing_binning[1] * setup.detector.binning[1]
                )
                nch2 = (setup.detector.roi[3] - setup.detector.roi[2]) // (
                    setup.detector.preprocessing_binning[2] * setup.detector.binning[2]
                ) + (setup.detector.roi[3] - setup.detector.roi[2]) % (
                    setup.detector.preprocessing_binning[2] * setup.detector.binning[2]
                )
                # detector init_area method, pixel sizes are the binned ones
                hxrd.Ang2Q.init_area(
                    setup.detector_ver_xrutil,
                    setup.detector_hor_xrutil,
                    cch1=cch1,
                    cch2=cch2,
                    Nch1=nch1,
                    Nch2=nch2,
                    pwidth1=setup.detector.pixelsize_y,
                    pwidth2=setup.detector.pixelsize_x,
                    distance=setup.distance,
                    detrot=prm["detrot"],
                    tiltazimuth=prm["tiltazimuth"],
                    tilt=prm["tilt_detector"],
                )
                # the first two arguments in init_area are
                # the direction of the detector

                data, mask, q_values, frames_logical = bu.grid_bcdi_xrayutil(
                    data=data,
                    mask=mask,
                    scan_number=scan_nb,
                    setup=setup,
                    frames_logical=frames_logical,
                    hxrd=hxrd,
                    debugging=prm["debug"],
                    cmap=prm["colormap"].cmap,
                )
            else:  # 'linearization'
                # for q values, the frame used is
                # (qx downstream, qy outboard, qz vertical up)
                # for reference_axis, the frame is z downstream, y vertical up,
                # x outboard but the order must be x,y,z
                data, mask, q_values, transfer_matrix = bu.grid_bcdi_labframe(
                    data=data,
                    mask=mask,
                    detector=setup.detector,
                    setup=setup,
                    align_q=prm["align_q"],
                    reference_axis=AXIS_TO_ARRAY[prm["ref_axis_q"]],
                    debugging=prm["debug"],
                    fill_value=(0, prm["fill_value_mask"]),
                    cmap=prm["colormap"].cmap,
                )
                prm["transformation_matrix"] = transfer_matrix
            nz, ny, nx = data.shape
            logger.info(
                "Data size after interpolation into an orthonormal frame:"
                f"{nz}, {ny}, {nx}"
            )

            # plot normalization by incident monitor for the gridded data
            if prm["normalize_flux"]:
                plt.ion()
                tmp_data = np.copy(
                    data
                )  # do not modify the raw data before the interpolation
                tmp_data[tmp_data < 5] = 0  # threshold the background
                tmp_data[mask == 1] = 0
                fig = gu.combined_plots(
                    tuple_array=(monitor, tmp_data),
                    tuple_sum_frames=(False, True),
                    tuple_sum_axis=(0, 1),
                    tuple_width_v=None,
                    tuple_width_h=None,
                    tuple_colorbar=(False, False),
                    tuple_vmin=(np.nan, 0),
                    tuple_vmax=(np.nan, np.nan),
                    tuple_title=(
                        "monitor.min() / monitor",
                        "Gridded normed data (threshold 5)\n",
                    ),
                    tuple_scale=("linear", "log"),
                    xlabel=("Frame number", "Q$_y$"),
                    ylabel=("Counts (a.u.)", "Q$_x$"),
                    position=(323, 122),
                    is_orthogonal=not prm["use_rawdata"],
                    reciprocal_space=True,
                    cmap=prm["colormap"].cmap,
                )

                fig.savefig(
                    setup.detector.savedir
                    + f"monitor_gridded_S{scan_nb}_{nz}_{ny}_{nx}"
                    + binning_comment
                    + ".png"
                )
                if prm["flag_interact"]:
                    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
                    cid = plt.connect("close_event", gu.close_event)
                    fig.waitforbuttonpress()
                    plt.disconnect(cid)
                plt.close(fig)
                plt.ioff()
                del tmp_data
                gc.collect()

    ########################
    # crop/pad/center data #
    ########################
    data, mask, pad_width, q_values, frames_logical = bu.center_fft(
        data=data,
        mask=mask,
        detector=setup.detector,
        frames_logical=frames_logical,
        centering=prm["centering_method"],
        fft_option=prm["center_fft"],
        pad_size=prm["pad_size"],
        fix_bragg=prm["bragg_peak"],
        fix_size=prm["fix_size"],
        q_values=q_values,
    )

    starting_frame = [
        pad_width[0],
        pad_width[2],
        pad_width[4],
    ]  # no need to check padded frames
    logger.info(f"Pad width: {pad_width}")
    nz, ny, nx = data.shape
    logger.info(f"Data size after cropping / padding: {data.shape}")

    ##########################################
    # optional masking of zero photon events #
    ##########################################
    if prm["mask_zero_event"]:
        # mask points when there is no intensity along the whole rocking curve
        # probably dead pixels
        temp_mask = np.zeros((ny, nx))
        temp_mask[np.sum(data, axis=0) == 0] = 1
        mask[np.repeat(temp_mask[np.newaxis, :, :], repeats=nz, axis=0) == 1] = 1
        del temp_mask

    ###########################################
    # save data and mask before alien removal #
    ###########################################
    fig, _, _ = gu.multislices_plot(
        data,
        sum_frames=True,
        scale="log",
        plot_colorbar=True,
        vmin=0,
        title="Data before aliens removal\n",
        is_orthogonal=not prm["use_rawdata"],
        reciprocal_space=True,
        cmap=prm["colormap"].cmap,
    )
    if prm["debug"]:
        fig.savefig(
            setup.detector.savedir
            + f"data_before_masking_sum_S{scan_nb}_{nz}_{ny}_{nx}_"
            f"{setup.detector.binning[0]}_"
            f"{setup.detector.binning[1]}_{setup.detector.binning[2]}.png"
        )
    if prm["flag_interact"]:
        fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
        cid = plt.connect("close_event", gu.close_event)
        fig.waitforbuttonpress()
        plt.disconnect(cid)
    plt.close(fig)

    piz, piy, pix = np.unravel_index(data.argmax(), data.shape)
    fig = gu.combined_plots(
        (data[piz, :, :], data[:, piy, :], data[:, :, pix]),
        tuple_sum_frames=False,
        tuple_sum_axis=0,
        tuple_width_v=None,
        tuple_width_h=None,
        tuple_colorbar=True,
        tuple_vmin=0,
        tuple_vmax=np.nan,
        tuple_scale="log",
        tuple_title=("data at max in xy", "data at max in xz", "data at max in yz"),
        is_orthogonal=not prm["use_rawdata"],
        reciprocal_space=False,
        cmap=prm["colormap"].cmap,
    )
    if prm["debug"]:
        fig.savefig(
            setup.detector.savedir + f"data_before_masking_S{scan_nb}_{nz}_{ny}_{nx}_"
            f"{setup.detector.binning[0]}"
            f"_{setup.detector.binning[1]}_{setup.detector.binning[2]}.png"
        )
    if prm["flag_interact"]:
        fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
        cid = plt.connect("close_event", gu.close_event)
        fig.waitforbuttonpress()
        plt.disconnect(cid)
    plt.close(fig)

    fig, _, _ = gu.multislices_plot(
        mask,
        sum_frames=True,
        scale="linear",
        plot_colorbar=True,
        vmin=0,
        vmax=(nz, ny, nx),
        title="Mask before aliens removal\n",
        is_orthogonal=not prm["use_rawdata"],
        reciprocal_space=True,
        cmap=prm["colormap"].cmap,
    )
    if prm["debug"]:
        fig.savefig(
            setup.detector.savedir + f"mask_before_masking_S{scan_nb}_{nz}_{ny}_{nx}_"
            f"{setup.detector.binning[0]}"
            f"_{setup.detector.binning[1]}_{setup.detector.binning[2]}.png"
        )

    if prm["flag_interact"]:
        fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
        cid = plt.connect("close_event", gu.close_event)
        fig.waitforbuttonpress()
        plt.disconnect(cid)
    plt.close(fig)

    ###############################################
    # save the orthogonalized diffraction pattern #
    ###############################################
    if not prm["use_rawdata"] and len(q_values) != 0:
        qx = q_values[0]
        qz = q_values[1]
        qy = q_values[2]

        if prm["save_to_vti"]:
            # save diffraction pattern to vti
            (
                nqx,
                nqz,
                nqy,
            ) = (
                data.shape
            )  # in nexus z downstream, y vertical / in q z vertical, x downstream
            logger.info(
                f"dqx, dqy, dqz = ({qx[1] - qx[0]}, {qy[1] - qy[0]}, {qz[1] - qz[0]})"
            )
            # in nexus z downstream, y vertical / in q z vertical, x downstream
            qx0 = qx.min()
            dqx = (qx.max() - qx0) / nqx
            qy0 = qy.min()
            dqy = (qy.max() - qy0) / nqy
            qz0 = qz.min()
            dqz = (qz.max() - qz0) / nqz

            gu.save_to_vti(
                filename=os.path.join(
                    setup.detector.savedir,
                    f"S{scan_nb}_ortho_int" + comment + ".vti",
                ),
                voxel_size=(dqx, dqz, dqy),
                tuple_array=data,
                tuple_fieldnames="int",
                origin=(qx0, qz0, qy0),
            )

    if prm["flag_interact"]:
        plt.ioff()
        #############################################
        # remove aliens
        #############################################
        nz, ny, nx = np.shape(data)
        width = 5
        max_colorbar = 5
        flag_mask = False
        flag_aliens = True

        fig_mask, ((ax0, ax1), (ax2, ax3)) = plt.subplots(
            nrows=2, ncols=2, figsize=(12, 6)
        )
        fig_mask.canvas.mpl_disconnect(fig_mask.canvas.manager.key_press_handler_id)
        original_data = np.copy(data)
        original_mask = np.copy(mask)
        frame_index = starting_frame
        ax0.imshow(
            data[frame_index[0], :, :],
            vmin=0,
            vmax=max_colorbar,
            cmap=prm["colormap"].cmap,
        )
        ax1.imshow(
            data[:, frame_index[1], :],
            vmin=0,
            vmax=max_colorbar,
            cmap=prm["colormap"].cmap,
        )
        ax2.imshow(
            data[:, :, frame_index[2]],
            vmin=0,
            vmax=max_colorbar,
            cmap=prm["colormap"].cmap,
        )
        ax3.set_visible(False)
        ax0.axis("scaled")
        ax1.axis("scaled")
        ax2.axis("scaled")
        if not prm["use_rawdata"]:
            ax0.invert_yaxis()  # detector Y is vertical down
        ax0.set_title(f"XY - Frame {frame_index[0] + 1} / {nz}")
        ax1.set_title(f"XZ - Frame {frame_index[1] + 1} / {ny}")
        ax2.set_title(f"YZ - Frame {frame_index[2] + 1} / {nx}")
        fig_mask.text(
            0.60,
            0.30,
            "m mask ; b unmask ; u next frame ; d previous frame",
            size=12,
        )
        fig_mask.text(
            0.60,
            0.25,
            "up larger ; down smaller ; right darker ; left brighter",
            size=12,
        )
        fig_mask.text(0.60, 0.20, "p plot full image ; q quit", size=12)
        plt.tight_layout()
        plt.connect("key_press_event", press_key)
        fig_mask.set_facecolor(prm["background_plot"])
        plt.show()
        del fig_mask, original_data, original_mask
        gc.collect()

        mask[np.nonzero(mask)] = 1

        fig, _, _ = gu.multislices_plot(
            data,
            sum_frames=True,
            scale="log",
            plot_colorbar=True,
            vmin=0,
            title="Data after aliens removal\n",
            is_orthogonal=not prm["use_rawdata"],
            reciprocal_space=True,
            cmap=prm["colormap"].cmap,
        )

        fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
        cid = plt.connect("close_event", gu.close_event)
        fig.waitforbuttonpress()
        plt.disconnect(cid)
        plt.close(fig)

        fig, _, _ = gu.multislices_plot(
            mask,
            sum_frames=True,
            scale="linear",
            plot_colorbar=True,
            vmin=0,
            vmax=(nz, ny, nx),
            title="Mask after aliens removal\n",
            is_orthogonal=not prm["use_rawdata"],
            reciprocal_space=True,
            cmap=prm["colormap"].cmap,
        )

        fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
        cid = plt.connect("close_event", gu.close_event)
        fig.waitforbuttonpress()
        plt.disconnect(cid)
        plt.close(fig)

        #############################################
        # define mask
        #############################################
        width = 0
        max_colorbar = 5
        flag_aliens = False
        flag_mask = True
        flag_pause = False  # press x to pause for pan/zoom
        previous_axis = None
        xy: List[int] = []  # list of points for mask

        fig_mask, ((ax0, ax1), (ax2, ax3)) = plt.subplots(
            nrows=2, ncols=2, figsize=(12, 6)
        )
        fig_mask.canvas.mpl_disconnect(fig_mask.canvas.manager.key_press_handler_id)
        original_data = np.copy(data)
        updated_mask = np.zeros((nz, ny, nx))
        data[mask == 1] = 0  # will appear as grey in the log plot (nan)
        ax0.imshow(
            np.log10(abs(data).sum(axis=0)),
            vmin=0,
            vmax=max_colorbar,
            cmap=prm["colormap"].cmap,
        )
        ax1.imshow(
            np.log10(abs(data).sum(axis=1)),
            vmin=0,
            vmax=max_colorbar,
            cmap=prm["colormap"].cmap,
        )
        ax2.imshow(
            np.log10(abs(data).sum(axis=2)),
            vmin=0,
            vmax=max_colorbar,
            cmap=prm["colormap"].cmap,
        )
        ax3.set_visible(False)
        ax0.axis("scaled")
        ax1.axis("scaled")
        ax2.axis("scaled")
        if not prm["use_rawdata"]:
            ax0.invert_yaxis()  # detector Y is vertical down
        ax0.set_title("XY")
        ax1.set_title("XZ")
        ax2.set_title("YZ")
        fig_mask.text(
            0.60, 0.45, "click to select the vertices of a polygon mask", size=12
        )
        fig_mask.text(
            0.60, 0.40, "x to pause/resume polygon masking for pan/zoom", size=12
        )
        fig_mask.text(0.60, 0.35, "p plot mask ; r reset current points", size=12)
        fig_mask.text(
            0.60,
            0.30,
            "m square mask ; b unmask ; right darker ; left brighter",
            size=12,
        )
        fig_mask.text(
            0.60, 0.25, "up larger masking box ; down smaller masking box", size=12
        )
        fig_mask.text(0.60, 0.20, "a restart ; q quit", size=12)
        info_text = fig_mask.text(0.60, 0.05, "masking enabled", size=16)
        plt.tight_layout()
        plt.connect("key_press_event", press_key)
        plt.connect("button_press_event", on_click)
        fig_mask.set_facecolor(prm["background_plot"])
        plt.show()

        mask[np.nonzero(updated_mask)] = 1
        data = original_data

        del fig_mask, flag_pause, flag_mask, original_data, updated_mask
        gc.collect()

    mask[np.nonzero(mask)] = 1
    data[mask == 1] = 0

    ########################################################
    # save the projected mask as hotpixels for later reuse #
    ########################################################
    hotpixels = mask.sum(axis=0)
    hotpixels[np.nonzero(hotpixels)] = 1
    np.savez_compressed(
        setup.detector.savedir + f"S{scan_nb}_hotpixels",
        hotpixels=hotpixels.astype(int),
    )

    ###############################################
    # mask or median filter isolated empty pixels #
    ###############################################
    if prm["median_filter"] in {"mask_isolated", "interp_isolated"}:
        logger.info("Filtering isolated pixels")
        nb_pix = 0
        for idx in range(
            pad_width[0], nz - pad_width[1]
        ):  # filter only frames whith data (not padded)
            data[idx, :, :], processed_pix, mask[idx, :, :] = util.mean_filter(
                data=data[idx, :, :],
                nb_neighbours=prm["median_filter_order"],
                mask=mask[idx, :, :],
                interpolate=prm["median_filter"],
                min_count=3,
                debugging=prm["debug"],
                cmap=prm["colormap"].cmap,
            )
            nb_pix += processed_pix
        logger.info(f"Total number of filtered pixels: {nb_pix}")
    elif prm["median_filter"] == "median":  # apply median filter
        logger.info("Applying median filtering")
        for idx in range(
            pad_width[0], nz - pad_width[1]
        ):  # filter only frames whith data (not padded)
            data[idx, :, :] = scipy.signal.medfilt2d(data[idx, :, :], [3, 3])
    else:
        logger.info("Skipping median filtering")

    ##########################
    # apply photon threshold #
    ##########################
    if prm["photon_threshold"] != 0:
        mask[data < prm["photon_threshold"]] = 1
        data[data < prm["photon_threshold"]] = 0
        logger.info(f"Applying photon threshold < {prm['photon_threshold']}")

    ################################################
    # check for nans and infs in the data and mask #
    ################################################
    nz, ny, nx = data.shape
    logger.info(f"Data size after masking: {data.shape}")

    data, mask = util.remove_nan(data=data, mask=mask)

    data[mask == 1] = 0

    ####################
    # debugging plots  #
    ####################
    plt.ion()
    if prm["debug"]:
        z0, y0, x0 = center_of_mass(data)
        fig, _, _ = gu.multislices_plot(
            data,
            sum_frames=False,
            scale="log",
            plot_colorbar=True,
            vmin=0,
            title="Masked data",
            slice_position=[int(z0), int(y0), int(x0)],
            is_orthogonal=not prm["use_rawdata"],
            reciprocal_space=True,
            cmap=prm["colormap"].cmap,
        )
        fig.savefig(
            setup.detector.savedir
            + f"middle_frame_S{scan_nb}_{nz}_{ny}_{nx}_{setup.detector.binning[0]}_"
            f"{setup.detector.binning[1]}_{setup.detector.binning[2]}"
            + comment
            + ".png"
        )
        if not prm["flag_interact"]:
            plt.close(fig)

        fig, _, _ = gu.multislices_plot(
            data,
            sum_frames=True,
            scale="log",
            plot_colorbar=True,
            vmin=0,
            title="Masked data",
            is_orthogonal=not prm["use_rawdata"],
            reciprocal_space=True,
            cmap=prm["colormap"].cmap,
        )
        fig.savefig(
            setup.detector.savedir
            + f"sum_S{scan_nb}_{nz}_{ny}_{nx}_{setup.detector.binning[0]}_"
            f"{setup.detector.binning[1]}_{setup.detector.binning[2]}"
            + comment
            + ".png"
        )
        if not prm["flag_interact"]:
            plt.close(fig)

        fig, _, _ = gu.multislices_plot(
            mask,
            sum_frames=True,
            scale="linear",
            plot_colorbar=True,
            vmin=0,
            vmax=(nz, ny, nx),
            title="Mask",
            is_orthogonal=not prm["use_rawdata"],
            reciprocal_space=True,
            cmap=prm["colormap"].cmap,
        )
        fig.savefig(
            setup.detector.savedir + f"mask_S{scan_nb}_{nz}_{ny}_{nx}_"
            f"{setup.detector.binning[0]}_{setup.detector.binning[1]}_"
            f"{setup.detector.binning[2]}" + comment + ".png"
        )
        if not prm["flag_interact"]:
            plt.close(fig)

    ##################################################
    # bin the stacking axis if needed, the detector  #
    # plane was already binned when loading the data #
    ##################################################
    if (
        setup.detector.binning[0] != 1 and not prm["reload_orthogonal"]
    ):  # data was already binned for reload_orthogonal
        data = util.bin_data(
            data,
            (setup.detector.binning[0], 1, 1),
            debugging=False,
            cmap=prm["colormap"].cmap,
        )
        mask = util.bin_data(
            mask,
            (setup.detector.binning[0], 1, 1),
            debugging=False,
            cmap=prm["colormap"].cmap,
        )
        mask[np.nonzero(mask)] = 1
        if not prm["use_rawdata"] and len(q_values) != 0:
            numz = len(qx)
            qx = qx[
                : numz - (numz % setup.detector.binning[0]) : setup.detector.binning[0]
            ]  # along Z
            del numz
    logger.info(f"Data size after binning the stacking dimension: {data.shape}")

    ##################################################################
    # final check of the shape to comply with FFT shape requirements #
    ##################################################################
    final_shape = util.smaller_primes(data.shape, maxprime=7, required_dividers=(2,))
    com = tuple(map(lambda x: int(np.rint(x)), center_of_mass(data)))
    crop_center = pu.find_crop_center(
        array_shape=data.shape, crop_shape=final_shape, pivot=com
    )
    data = util.crop_pad(
        data,
        output_shape=final_shape,
        crop_center=crop_center,
        cmap=prm["colormap"].cmap,
    )
    mask = util.crop_pad(
        mask,
        output_shape=final_shape,
        crop_center=crop_center,
        cmap=prm["colormap"].cmap,
    )
    logger.info(f"Data size after considering FFT shape requirements: {data.shape}")
    nz, ny, nx = data.shape
    comment = f"{comment}_{nz}_{ny}_{nx}" + binning_comment

    ############################
    # save final data and mask #
    ############################
    logger.info(f"Saving directory: {setup.detector.savedir}")
    if prm["save_as_int"]:
        data = data.astype(int)
    logger.info(f"Data type before saving: {data.dtype}")
    mask[np.nonzero(mask)] = 1
    mask = mask.astype(int)
    logger.info(f"Mask type before saving: {mask.dtype}")
    if not prm["use_rawdata"] and len(q_values) != 0:
        if prm["save_to_npz"]:
            np.savez_compressed(
                setup.detector.savedir + f"QxQzQy_S{scan_nb}" + comment,
                qx=qx,
                qz=qz,
                qy=qy,
            )
        if prm["save_to_mat"]:
            savemat(setup.detector.savedir + f"S{scan_nb}_qx.mat", {"qx": qx})
            savemat(setup.detector.savedir + f"S{scan_nb}_qz.mat", {"qz": qz})
            savemat(setup.detector.savedir + f"S{scan_nb}_qy.mat", {"qy": qy})
        max_z = data.sum(axis=0).max()
        fig, _, _ = gu.contour_slices(
            data,
            (qx, qz, qy),
            sum_frames=True,
            title="Final data",
            plot_colorbar=True,
            scale="log",
            is_orthogonal=True,
            levels=np.linspace(0, np.ceil(np.log10(max_z)), 150, endpoint=False),
            reciprocal_space=True,
            cmap=prm["colormap"].cmap,
        )
        fig.savefig(
            setup.detector.savedir
            + f"final_reciprocal_space_S{scan_nb}"
            + comment
            + ".png"
        )
        plt.close(fig)

    if prm["save_to_npz"]:
        np.savez_compressed(
            setup.detector.savedir + f"S{scan_nb}_pynx" + comment, data=data
        )
        np.savez_compressed(
            setup.detector.savedir + f"S{scan_nb}_maskpynx" + comment, mask=mask
        )

    if prm["save_to_mat"]:
        # save to .mat, the new order is x y z (outboard, vertical up, downstream)
        savemat(
            setup.detector.savedir + f"S{scan_nb}_data.mat",
            {"data": np.moveaxis(data.astype(np.float32), [0, 1, 2], [-1, -2, -3])},
        )
        savemat(
            setup.detector.savedir + f"S{scan_nb}_mask.mat",
            {"data": np.moveaxis(mask.astype(np.int8), [0, 1, 2], [-1, -2, -3])},
        )

    # save results in hdf5 file
    with h5py.File(
        f"{setup.detector.savedir}S{scan_nb}_preprocessing{comment}.h5", "w"
    ) as hf:
        out = hf.create_group("output")
        par = hf.create_group("params")
        out.create_dataset("data", data=data)
        out.create_dataset("mask", data=mask)

        if metadata is not None:
            out.create_dataset("tilt_values", data=metadata["tilt_values"])
            out.create_dataset("rocking_curve", data=metadata["rocking_curve"])
            out.create_dataset("interp_tilt", data=metadata["interp_tilt_values"])
            out.create_dataset("interp_curve", data=metadata["interp_rocking_curve"])
            out.create_dataset("COM_rocking_curve", data=metadata["COM_rocking_curve"])
            out.create_dataset("detector_data_COM", data=metadata["detector_data_COM"])
            out.create_dataset("interp_fwhm", data=metadata["interp_fwhm"])
        try:
            out.create_dataset("bragg_peak", data=prm["bragg_peak"])
        except TypeError:
            logger.info("Bragg peak not computed.")
        out.create_dataset("q", data=q)
        out.create_dataset("qnorm", data=qnorm)
        out.create_dataset("dist_plane", data=dist_plane)
        out.create_dataset("bragg_inplane", data=prm["inplane_angle"])
        out.create_dataset("bragg_outofplane", data=prm["outofplane_angle"])

        par.create_dataset("detector", data=str(setup.detector.params))
        par.create_dataset("setup", data=str(setup.params))
        par.create_dataset("parameters", data=str(prm))

    ############################
    # plot final data and mask #
    ############################
    data[np.nonzero(mask)] = 0
    fig, _, _ = gu.multislices_plot(
        data,
        sum_frames=True,
        scale="log",
        plot_colorbar=True,
        vmin=0,
        title="Final data",
        is_orthogonal=not prm["use_rawdata"],
        reciprocal_space=True,
        cmap=prm["colormap"].cmap,
    )
    fig.savefig(setup.detector.savedir + f"finalsum_S{scan_nb}" + comment + ".png")
    if not prm["flag_interact"]:
        plt.close(fig)

    fig, _, _ = gu.multislices_plot(
        mask,
        sum_frames=True,
        scale="linear",
        plot_colorbar=True,
        vmin=0,
        vmax=(nz, ny, nx),
        title="Final mask",
        is_orthogonal=not prm["use_rawdata"],
        reciprocal_space=True,
        cmap=prm["colormap"].cmap,
    )
    fig.savefig(setup.detector.savedir + f"finalmask_S{scan_nb}" + comment + ".png")
    if not prm["flag_interact"]:
        plt.close(fig)

    del data, mask
    gc.collect()

    if len(prm["scans"]) > 1:
        plt.close("all")

    logger.removeHandler(filehandler)
    filehandler.close()

    return tmpfile, Path(setup.detector.savedir), logger
