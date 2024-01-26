# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, jerome.carnis@esrf.fr
"""
Workflow for CDI data preprocessing of a single scan, before phase retrieval.

The detector is expected to be fixed, its plane being always perpendicular to the direct
beam independently of the detector position.
"""
# mypy: ignore-errors
import gc
import logging
import os
import tkinter as tk
from logging import Logger
from pathlib import Path
from tkinter import filedialog
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal  # for medfilt2d
from scipy.io import savemat
from scipy.ndimage.measurements import center_of_mass

import bcdi.graph.graph_utils as gu
import bcdi.preprocessing.cdi_utils as cdi
import bcdi.utils.utilities as util
from bcdi.experiment.setup import Setup
from bcdi.utils.snippets_logging import FILE_FORMATTER

logger = logging.getLogger(__name__)


def process_scan_cdi(
    scan_idx: int, prm: Dict[str, Any]
) -> Tuple[Path, Path, Optional[Logger]]:
    """
    Run the CDI preprocessing defined by the configuration parameters for a single scan.

    This function is meant to be run as a process in multiprocessing, although it can
    also be used as a normal function for a single scan. It assumes that the dictionary
    of parameters was validated via a ConfigChecker instance. Interactive masking and
    reloading of previous masking are not compatible with multiprocessing.

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
    min_range: Optional[float] = None
    # used to define the crop size when fit_datarange is True

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
    if not prm["multiprocessing"] or len(prm["scans"]) == 1:
        logger.propagate = True

    prm["sample"] = f"{prm['sample_name']}+{scan_nb}"
    comment = prm["comment"]  # re-initialize comment
    tmp_str = f"Scan {scan_idx + 1}/{len(prm['scans'])}: S{scan_nb}"
    from datetime import datetime

    logger.info(f"Start {process_scan_cdi.__name__} at {datetime.now()}")
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

    if not prm["use_rawdata"]:
        comment += "_ortho"
    if prm["normalize_flux"]:
        comment = comment + "_norm"

    if prm["reload_previous"]:  # resume previous masking
        logger.info("Resuming previous masking")
        file_path = filedialog.askopenfilename(
            initialdir=setup.detector.scandir,
            title="Select data file",
            filetypes=[("NPZ", "*.npz")],
        )
        data, _ = util.load_file(file_path)
        nz, ny, nx = np.shape(data)

        # update savedir to save the data in the same directory as the reloaded data
        setup.detector.savedir = os.path.dirname(file_path) + "/"

        file_path = filedialog.askopenfilename(
            initialdir=setup.detector.savedir,
            title="Select mask file",
            filetypes=[("NPZ", "*.npz")],
        )
        mask, _ = util.load_file(file_path)

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
                prm["q_values"] = [
                    reload_qvalues["qx"],
                    reload_qvalues["qz"],
                    reload_qvalues["qy"],
                ]
            except FileNotFoundError:
                prm["q_values"] = []

            prm["normalize_flux"] = "skip"
            # we assume that normalization was already performed
            monitor = []  # we assume that normalization was already performed
            if prm["fit_datarange"]:
                min_range = (nx / 2) * np.sqrt(2)
                # used when fit_datarange is True, keep the full array because
                # we do not know the position of the origin of reciprocal space
            frames_logical = np.ones(nz)

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
                    data, binning=setup.detector.binning, debugging=False
                )
                mask = util.bin_data(
                    mask, binning=setup.detector.binning, debugging=False
                )
                mask[np.nonzero(mask)] = 1
                if len(prm["q_values"]) == 3:
                    qx, qz, qy = prm["q_values"]  # downstream, vertical up, outboard
                    numz, numy, numx = len(qx), len(qz), len(qy)
                    qx = qx[
                        : numz
                        - (numz % setup.detector.binning[2]) : setup.detector.binning[2]
                    ]
                    # along z downstream, same binning as along x
                    qz = qz[
                        : numy
                        - (numy % setup.detector.binning[1]) : setup.detector.binning[1]
                    ]
                    # along y vertical, the axis of rotation
                    qy = qy[
                        : numx
                        - (numx % setup.detector.binning[2]) : setup.detector.binning[2]
                    ]
                    # along x outboard
                    del numz, numy, numx
        else:  # the data is in the detector frame
            data, mask, frames_logical, monitor = cdi.reload_cdi_data(
                scan_number=scan_nb,
                data=data,
                mask=mask,
                setup=setup,
                debugging=prm["debug"],
                normalize_method=prm["normalize_flux"],
                photon_threshold=prm["loading_threshold"],
            )

    else:  # new masking process
        flatfield = util.load_flatfield(prm["flatfield_file"])
        hotpix_array = util.load_hotpixels(prm["hotpixels_file"])
        background = util.load_background(prm["background_file"])

        data, mask, frames_logical, monitor = cdi.load_cdi_data(
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

    nz, ny, nx = data.shape
    logger.info(f"Input data shape: {data.shape}")

    if not prm["reload_orthogonal"]:
        dirbeam = int(
            (setup.direct_beam[1] - setup.detector.roi[2]) / setup.detector.binning[2]
        )
        # updated horizontal direct beam
        if prm["fit_datarange"]:
            min_range = min(dirbeam, nx - dirbeam)
            # crop at the maximum symmetrical range
            logger.info(
                "Maximum symmetrical range with defined data along the "
                f"detector horizontal direction: 2*{min_range} pixels"
            )
            if min_range <= 0:
                raise ValueError(
                    "error in calculating min_range, check the direct beam " "position"
                )

        if prm["save_rawdata"]:
            np.savez_compressed(
                setup.detector.savedir
                + "S"
                + str(scan_nb)
                + "_data_before_masking_stack",
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

        if prm["flag_interact"]:
            # masking step in the detector plane
            plt.ioff()
            width = 0
            max_colorbar = 5
            flag_aliens = False
            flag_mask = True
            flag_pause = False  # press x to pause for pan/zoom
            previous_axis = None
            xy: List[List[int]] = []  # list of points for mask

            fig_mask = plt.figure(figsize=(12, 9))
            ax0 = fig_mask.add_subplot(121)
            ax1 = fig_mask.add_subplot(322)
            ax2 = fig_mask.add_subplot(324)
            ax3 = None
            fig_mask.canvas.mpl_disconnect(fig_mask.canvas.manager.key_press_handler_id)
            original_data = np.copy(data)
            updated_mask = np.zeros((nz, ny, nx))
            data[mask == 1] = 0  # will appear as grey in the log plot (nan)
            ax0.imshow(
                np.log10(abs(data).sum(axis=0)), vmin=0, vmax=max_colorbar, cmap=my_cmap
            )
            ax1.imshow(
                np.log10(abs(data).sum(axis=1)), vmin=0, vmax=max_colorbar, cmap=my_cmap
            )
            ax2.imshow(
                np.log10(abs(data).sum(axis=2)), vmin=0, vmax=max_colorbar, cmap=my_cmap
            )
            ax0.axis("scaled")
            ax1.axis("scaled")
            ax2.axis("scaled")
            ax0.set_title("XY")
            ax1.set_title("XZ")
            ax2.set_title("YZ")
            fig_mask.text(
                0.60, 0.27, "click to select the vertices of a polygon mask", size=10
            )
            fig_mask.text(
                0.60, 0.24, "x to pause/resume polygon masking for pan/zoom", size=10
            )
            fig_mask.text(0.60, 0.21, "p plot mask ; r reset current points", size=10)
            fig_mask.text(
                0.60,
                0.18,
                "m square mask ; b unmask ; right darker ; left brighter",
                size=10,
            )
            fig_mask.text(
                0.60, 0.15, "up larger masking box ; down smaller masking box", size=10
            )
            fig_mask.text(0.60, 0.12, "a restart ; q quit", size=10)
            info_text = fig_mask.text(0.60, 0.05, "masking enabled", size=16)
            plt.tight_layout()
            plt.connect("key_press_event", press_key)
            plt.connect("button_press_event", on_click)
            fig_mask.set_facecolor(prm["background_plot"])
            plt.show()

            mask[np.nonzero(updated_mask)] = 1
            data = original_data
            del fig_mask, original_data, updated_mask
            gc.collect()

        if prm["use_rawdata"]:
            prm["q_values"] = []
            binning_comment = (
                f"_{setup.detector.preprocessing_binning[0]*setup.detector.binning[0]}"
                f"_{setup.detector.preprocessing_binning[1]*setup.detector.binning[1]}"
                f"_{setup.detector.preprocessing_binning[2]*setup.detector.binning[2]}"
            )
            # binning along axis 0 is done after masking
            data[np.nonzero(mask)] = 0
        else:  # the data will be gridded, binning[0] is already set to 1
            # sample rotation around the vertical direction at P10:
            # the effective binning in axis 0 is preprocessing_binning[2]*binning[2]
            binning_comment = (
                f"_{setup.detector.preprocessing_binning[2]*setup.detector.binning[2]}"
                f"_{setup.detector.preprocessing_binning[1]*setup.detector.binning[1]}"
                f"_{setup.detector.preprocessing_binning[2]*setup.detector.binning[2]}"
            )

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

            logger.info("Gridding the data in the orthonormal laboratory frame")
            data, mask, prm["q_values"], frames_logical = cdi.grid_cdi(
                data=data,
                mask=mask,
                setup=setup,
                frames_logical=frames_logical,
                correct_curvature=prm["correct_curvature"],
                fill_value=(0, prm["fill_value_mask"]),
                debugging=prm["debug"],
            )

            # plot normalization by incident monitor for the gridded data
            if prm["normalize_flux"] != "skip":
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

    else:  # reload_orthogonal=True, the data is already gridded,
        # binning was realized along each axis
        binning_comment = (
            f"_{setup.detector.preprocessing_binning[0]*setup.detector.binning[0]}"
            f"_{setup.detector.preprocessing_binning[1]*setup.detector.binning[1]}"
            f"_{setup.detector.preprocessing_binning[2]*setup.detector.binning[2]}"
        )

    nz, ny, nx = np.shape(data)
    plt.ioff()

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

    #####################################
    # save data and mask before masking #
    #####################################
    fig, _, _ = gu.multislices_plot(
        data,
        sum_frames=True,
        scale="log",
        plot_colorbar=True,
        vmin=0,
        title="Data before aliens removal\n",
        is_orthogonal=not prm["use_rawdata"],
        reciprocal_space=True,
    )
    if prm["debug"]:
        fig.savefig(
            setup.detector.savedir
            + f"data_before_masking_S{scan_nb}_{nz}_{ny}_{nx}"
            + binning_comment
            + ".png"
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
    )
    if prm["debug"]:
        fig.savefig(
            setup.detector.savedir
            + f"mask_before_masking_S{scan_nb}_{nz}_{ny}_{nx}"
            + binning_comment
            + ".png"
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
    if not prm["use_rawdata"] and len(prm["q_values"]) == 3:
        qx, qz, qy = prm["q_values"]  # downstream, vertical up, outboard

        if prm["save_to_vti"]:
            (
                nqx,
                nqz,
                nqy,
            ) = (
                data.shape
            )  # in nexus z downstream, y vertical / in q z vertical, x downstream
            logger.info(
                f"(dqx, dqy, dqz) = {qx[1] - qx[0]}, {qy[1] - qy[0]}, {qz[1] - qz[0]}"
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
                    "S" + str(scan_nb) + "_ortho_int" + comment + ".vti",
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
        frame_index = [0, 0, 0]
        ax0.imshow(data[frame_index[0], :, :], vmin=0, vmax=max_colorbar, cmap=my_cmap)
        ax1.imshow(data[:, frame_index[1], :], vmin=0, vmax=max_colorbar, cmap=my_cmap)
        ax2.imshow(data[:, :, frame_index[2]], vmin=0, vmax=max_colorbar, cmap=my_cmap)
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
            0.60, 0.30, "m mask ; b unmask ; u next frame ; d previous frame", size=12
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
        )

        fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
        cid = plt.connect("close_event", gu.close_event)
        fig.waitforbuttonpress()
        plt.disconnect(cid)
        plt.close(fig)

        ####################
        # GUI for the mask #
        ####################
        width = 0
        max_colorbar = 5
        flag_aliens = False
        flag_mask = True
        flag_pause = False  # press x to pause for pan/zoom
        previous_axis = None
        xy = []  # list of points for mask

        fig_mask, ((ax0, ax1), (ax2, ax3)) = plt.subplots(
            nrows=2, ncols=2, figsize=(12, 6)
        )
        fig_mask.canvas.mpl_disconnect(fig_mask.canvas.manager.key_press_handler_id)
        original_data = np.copy(data)
        updated_mask = np.zeros((nz, ny, nx))
        data[mask == 1] = 0  # will appear as grey in the log plot (nan)
        ax0.imshow(
            np.log10(abs(data).sum(axis=0)), vmin=0, vmax=max_colorbar, cmap=my_cmap
        )
        ax1.imshow(
            np.log10(abs(data).sum(axis=1)), vmin=0, vmax=max_colorbar, cmap=my_cmap
        )
        ax2.imshow(
            np.log10(abs(data).sum(axis=2)), vmin=0, vmax=max_colorbar, cmap=my_cmap
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
        fig_mask.text(0.60, 0.40, "then p to apply and see the result", size=12)
        fig_mask.text(0.60, 0.30, "x to pause/resume masking for pan/zoom", size=12)
        fig_mask.text(
            0.60, 0.25, "up larger masking box ; down smaller masking box", size=12
        )
        fig_mask.text(
            0.60, 0.20, "m mask ; b unmask ; right darker ; left brighter", size=12
        )
        fig_mask.text(
            0.60, 0.15, "p plot full masked data ; a restart ; q quit", size=12
        )
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

    ###############################################
    # mask or median filter isolated empty pixels #
    ###############################################
    if prm["median_filter"] in {"mask_isolated", "interp_isolated"}:
        logger.info("Filtering isolated pixels")
        nb_pix = 0
        for idx in range(nz):  # filter only frames whith data (not padded)
            data[idx, :, :], numb_pix, mask[idx, :, :] = util.mean_filter(
                data=data[idx, :, :],
                nb_neighbours=prm["median_filter_order"],
                mask=mask[idx, :, :],
                interpolate=prm["median_filter"],
                min_count=3,
                debugging=prm["debug"],
                cmap=prm["colormap"].cmap,
            )
            nb_pix = nb_pix + numb_pix
            logger.info(f"Total number of filtered pixels: {nb_pix}")
    elif prm["median_filter"] == "median":  # apply median filter
        logger.info("Applying median filtering")
        for idx in range(nz):
            # filter only frames whith data (not padded)
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
        )
        fig.savefig(
            setup.detector.savedir
            + f"middle_frame_S{scan_nb}_{nz}_{ny}_{nx}"
            + binning_comment
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
        )
        fig.savefig(
            setup.detector.savedir
            + f"sum_S{scan_nb}_{nz}_{ny}_{nx}"
            + binning_comment
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
        )
        fig.savefig(
            setup.detector.savedir
            + f"mask_S{scan_nb}_{nz}_{ny}_{nx}"
            + binning_comment
            + ".png"
        )
        if not prm["flag_interact"]:
            plt.close(fig)

    ############################################################
    # select the largest cubic array fitting inside data range #
    ############################################################
    # this is to avoid having large masked areas near the corner of the area
    # which is a side effect of regridding the data from cylindrical coordinates
    if not prm["use_rawdata"] and prm["fit_datarange"] and min_range is not None:
        final_nxz = int(np.floor(min_range * 2 / np.sqrt(2)))
        if (final_nxz % 2) != 0:
            final_nxz = final_nxz - 1  # we want the number of pixels to be even
        data = data[
            (nz - final_nxz) // 2 : (nz - final_nxz) // 2 + final_nxz,
            :,
            (nz - final_nxz) // 2 : (nz - final_nxz) // 2 + final_nxz,
        ]
        mask = mask[
            (nz - final_nxz) // 2 : (nz - final_nxz) // 2 + final_nxz,
            :,
            (nz - final_nxz) // 2 : (nz - final_nxz) // 2 + final_nxz,
        ]
        logger.info(
            f"Data size after taking the largest data-defined area: {data.shape}"
        )
        if len(prm["q_values"]) != 0:
            qx = qx[
                (nz - final_nxz) // 2 : (nz - final_nxz) // 2 + final_nxz
            ]  # along Z
            qy = qy[
                (nz - final_nxz) // 2 : (nz - final_nxz) // 2 + final_nxz
            ]  # along X
            # qz (along Y) keeps the same number of pixels
        else:
            logger.info("fit_datarange: q values are not provided")

    ##############################################################
    # only for non gridded data, bin the stacking axis           #
    # the detector plane was already binned during data loading  #
    ##############################################################
    if setup.detector.binning[0] != 1 and not prm["reload_orthogonal"]:
        # for data to be gridded, binning[0] is set to 1
        data = util.bin_data(data, (setup.detector.binning[0], 1, 1), debugging=False)
        mask = util.bin_data(mask, (setup.detector.binning[0], 1, 1), debugging=False)
        mask[np.nonzero(mask)] = 1

    nz, ny, nx = data.shape
    logger.info(f"Data size after binning the stacking dimension: {data.shape}")
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
    if not prm["use_rawdata"] and len(prm["q_values"]) != 0:
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
        fig, _, _ = gu.contour_slices(
            data,
            (qx, qz, qy),
            sum_frames=True,
            title="Final data",
            levels=np.linspace(0, int(np.log10(data.max())), 150, endpoint=False),
            plot_colorbar=True,
            scale="log",
            is_orthogonal=True,
            reciprocal_space=True,
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
