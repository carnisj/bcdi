#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

try:
    import hdf5plugin  # for P10, should be imported before h5py or PyTables
except ModuleNotFoundError:
    pass
import gc
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.signal  # for medfilt2d
from scipy.ndimage.measurements import center_of_mass
import sys
from scipy.io import savemat
import tkinter as tk
from tkinter import filedialog
import bcdi.graph.graph_utils as gu
from bcdi.experiment.detector import create_detector
from bcdi.experiment.setup import Setup
import bcdi.utils.utilities as util
import bcdi.preprocessing.cdi_utils as cdi
import bcdi.utils.validation as valid

helptext = """
Prepare experimental data for forward CDI phasing: crop/pad, center, mask, normalize,
filter and regrid the data. Beamlines currently supported: ESRF ID01, SOLEIL CRISTAL,
SOLEIL SIXS and PETRAIII P10.
Output: data and mask as numpy .npz or Matlab .mat 3D arrays for phasing
File structure should be (e.g. scan 1):
specfile, background, hotpixels file and flatfield file in:    /rootdir/

The data is expected in: /rootdir/S1/data/
Output files are saved in:
/rootdir/S1/pynxraw/ or /rootdir/S1/pynx/ depending on the 'use_rawdata' parameter.
"""

scans = [22]  # list or array of scan numbers
root_folder = "C:/Users/Jerome/Documents/data/dataset_P10_saxs/"
save_dir = root_folder + "test/"
# images will be saved here, leave it to None otherwise
# (default to data directory's parent)
sample_name = "gold_2_2_2"
# "S"  # # list of sample names. If only one name is indicated,
# it will be repeated to match the number of scans
user_comment = ""  # string, should start with "_"
debug = False  # set to True to see plots
binning = [1, 2, 2]  # binning that will be used for phasing
# (stacking dimension, detector vertical axis, detector horizontal axis)
bin_during_loading = False  # True to bin during loading, require less memory
##############################
# parameters used in masking #
##############################
flag_interact = True  # True to interact with plots, False to close it automatically
background_plot = (
    "0.5"  # in level of grey in [0,1], 0 being dark. For visual comfort during masking
)
##############################################
# parameters used in intensity normalization #
##############################################
normalize_method = "monitor"  # 'skip' for no normalization,
# 'monitor' to use the default monitor, 'sum_roi' to normalize
# by the intensity summed in normalize_roi
normalize_roi = None
# roi for the integration of intensity used as a monitor for data normalization
# [Vstart, Vstop, Hstart, Hstop]
#################################
# parameters for data filtering #
#################################
mask_zero_event = False  # mask pixels where the sum along the rocking curve is zero
# may be dead pixels
flag_medianfilter = "skip"
# set to 'median' for applying med2filter [3,3]
# set to 'interp_isolated' to interpolate isolated empty pixels based on
# 'medfilt_order' parameter
# set to 'mask_isolated' it will mask isolated empty pixels
# set to 'skip' will skip filtering
medfilt_order = 8  # for custom median filter,
# number of pixels with intensity surrounding the empty pixel
#################################################
# parameters used when reloading processed data #
#################################################
reload_previous = False  # True to resume a previous masking (load data and mask)
reload_orthogonal = False
# True if the reloaded data is already intepolated in an orthonormal frame
preprocessing_binning = (1, 1, 1)
# binning factors in each dimension of the binned data to be reloaded
##################
# saving options #
##################
save_rawdata = False  # save also the raw data when use_rawdata is False
save_to_npz = True  # True to save the processed data in npz format
save_to_mat = False  # True to save the processed data in mat format
save_to_vti = False  # save the orthogonalized diffraction pattern to VTK file
save_asint = False
# if True, the result will be saved as an array of integers (save space)

###############################
# beamline related parameters #
###############################
beamline = "P10_SAXS"
# name of the beamline, used for data loading and normalization by monitor
# supported beamlines: 'ID01', 'SIXS_2018', 'SIXS_2019', 'CRISTAL', 'P10', 'P10_SAXS'
is_series = True  # specific to series measurement at P10

custom_scan = False  # set it to True for a stack of images acquired without scan,
# e.g. with ct in a macro, or when
# there is no spec/log file available
custom_images = None  # [10*i+929+j for i in range(92) for j in range(8)]
# custom_images.append(1849)  # np.arange(11353, 11453, 1)
# list of image numbers for the custom_scan
custom_monitor = None  # np.ones(len(custom_images))
# monitor values for normalization for the custom_scan

specfile_name = ""
# template for ID01: name of the spec file without '.spec'
# template for SIXS_2018: full path of the alias dictionnary,
# typically root_folder + 'alias_dict_2019.txt'
# template for all other beamlines: ''
###############################
# detector related parameters #
###############################
detector = "Eiger4M"  # "Eiger2M" or "Maxipix" or "Eiger4M"
direct_beam = (1349, 1321)
# tuple of int (vertical, horizontal): position of the direct beam in pixels, in the
# unbinned detector.
# This parameter is important for gridding the data onto the laboratory frame.
roi_detector = [
    direct_beam[0] - 250,
    direct_beam[0] + 250,
    direct_beam[1] - 354,
    direct_beam[1] + 354,
]
# [Vstart, Vstop, Hstart, Hstop]
# leave it as None to use the full detector.
photon_threshold = 0  # data[data < photon_threshold] = 0
photon_filter = "loading"  # 'loading' or 'postprocessing',
# when the photon threshold should be applied
# if 'loading', it is applied before binning;
# if 'postprocessing', it is applied at the end of the script before saving
background_file = None  # root_folder + 'background.npz'  # non empty file path or None
hotpixels_file = root_folder + "hotpixels.npz"  # non empty file path or None
flatfield_file = None
# root_folder + "flatfield_maxipix_8kev.npz"  # non empty file path or None
template_imagefile = "_master.h5"  # ''_data_%06d.h5'
# template for ID01: 'data_mpx4_%05d.edf.gz' or 'align_eiger2M_%05d.edf.gz'
# template for SIXS_2018: 'align.spec_ascan_mu_%05d.nxs'
# template for SIXS_2019: 'spare_ascan_mu_%05d.nxs'
# template for Cristal: 'S%d.nxs'
# template for P10: '_master.h5'
# template for NANOMAX: '%06d.h5'
# template for 34ID: 'Sample%dC_ES_data_51_256_256.npz'
######################################################################
# parameters used for interpolating the data in an orthonormal frame #
######################################################################
use_rawdata = False  # False for using data gridded in laboratory frame
# True for using data in detector frame
fill_value_mask = 1  # 0 (not masked) or 1 (masked).
# It will define how the pixels outside of the data range are
# processed during the interpolation. Because of the large number of masked pixels,
# phase retrieval converges better if the pixels are not masked (0 intensity
# imposed). The data is by default set to 0 outside of the defined range.
correct_curvature = False
# True to correcture q values for the curvature of Ewald sphere
fit_datarange = True  # if True, crop the final array within data range,
# avoiding areas at the corners of the window viewed from the top, data is circular,
# but the interpolation window is rectangular, with nan values outside of data
sdd = 4.95  # sample to detector distance in m, used only if use_rawdata is False
energy = 8700  # x-ray energy in eV, used only if use_rawdata is False
custom_motors = None  # {"hprz": np.linspace(0, 184, num=737, endpoint=True)}
# use this to declare motor positions if there is not log file
# example: {"hprz": np.linspace(16.989, 18.989, num=100, endpoint=False)}
# P10: hprz for the inplane rotation
##################################
# end of user-defined parameters #
##################################


def close_event(event):
    """This function handles closing events on plots."""
    print(event, "Click on the figure instead of closing it!")
    sys.exit()


def on_click(event):
    """
    Function to interact with a plot, return the position of clicked pixel. If
    flag_pause==1 or if the mouse is out of plot axes, it will not register the click

    :param event: mouse click event
    """
    global xy, flag_pause, previous_axis
    if not event.inaxes:
        return
    if not flag_pause:

        if (previous_axis == event.inaxes) or (previous_axis is None):  # collect points
            _x, _y = int(np.rint(event.xdata)), int(np.rint(event.ydata))
            xy.append([_x, _y])
            if previous_axis is None:
                previous_axis = event.inaxes
        else:  # the click is not in the same subplot, restart collecting points
            print(
                "Please select mask polygon vertices within the same subplot: "
                "restart masking..."
            )
            xy = []
            previous_axis = None


def press_key(event):
    """
    Interact with a plot for masking parasitic diffraction intensity or detector gaps

    :param event: button press event
    """
    global original_data, original_mask, updated_mask, data, mask, frame_index, width
    global flag_aliens, flag_mask, flag_pause, xy, fig_mask, max_colorbar, ax0, ax1
    global ax2, previous_axis, detector_plane, info_text, my_cmap

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
            invert_yaxis = (not use_rawdata) and (not detector_plane)
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
                    axes=(ax0, ax1, ax2),
                    width=width,
                    dim=dim,
                    frame_index=frame_index,
                    vmin=0,
                    vmax=max_colorbar,
                    cmap=my_cmap,
                    invert_yaxis=invert_yaxis,
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
                    axes=(ax0, ax1, ax2),
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
                    invert_yaxis=invert_yaxis,
                )

                if click_dim is None:
                    previous_axis = None
            else:
                stop_masking = False

            if stop_masking:
                plt.close("all")

    except AttributeError:  # mouse pointer out of axes
        pass


#########################
# check some parameters #
#########################
min_range = None
q_values = None
qx = None
qy = None
qz = None

if isinstance(scans, int):
    scans = (scans,)

if photon_filter == "loading":
    loading_threshold = photon_threshold
else:
    loading_threshold = 0

if reload_previous:
    create_savedir = False
    user_comment += "_reloaded"
    print(
        "\nReloading... update the direct beam position "
        "taking into account preprocessing_binning"
    )
    direct_beam = (
        direct_beam[0] // preprocessing_binning[1],
        direct_beam[1] // preprocessing_binning[2],
    )
else:
    create_savedir = True
    preprocessing_binning = (1, 1, 1)
    reload_orthogonal = False

if reload_orthogonal:
    use_rawdata = False

if use_rawdata:
    save_dirname = "pynxraw"
    print("Output will be non orthogonal, in the detector frame")
    plot_title = ["YZ", "XZ", "XY"]
else:
    save_dirname = "pynx"
    print("Output will interpolated in the orthogonal laboratory frame")
    plot_title = ["QzQx", "QyQx", "QyQz"]
    if reload_orthogonal:  # data already gridded, one can bin the first axis
        pass
    else:  # data in the detector frame,
        # one cannot bin the first axis because it is done during interpolation
        print(
            "\nuse_rawdata=False: defaulting the binning factor "
            "along the stacking dimension to 1"
        )
        # the vertical axis y being the rotation axis,
        # binning along z downstream and x outboard will be the same
        binning[0] = 1
        if preprocessing_binning[0] != 1:
            print(
                "preprocessing_binning along axis 0 should be 1 "
                "for reloaded data to be gridded (angles will not match)"
            )
            sys.exit()

if isinstance(sample_name, str):
    sample_name = [sample_name for idx in range(len(scans))]
valid.valid_container(
    sample_name,
    container_types=(tuple, list),
    length=len(scans),
    item_types=str,
    name="preprocess_bcdi",
)

###################
# define colormap #
###################
colormap = gu.Colormap()
my_cmap = colormap.cmap
plt.rcParams["keymap.fullscreen"] = [""]

#######################
# Initialize detector #
#######################
detector = create_detector(
    name=detector,
    roi=roi_detector,
    sum_roi=normalize_roi,
    binning=binning,
    preprocessing_binning=preprocessing_binning,
)

####################
# Initialize setup #
####################
setup = Setup(
    beamline=beamline,
    detector=detector,
    energy=energy,
    rocking_angle="inplane",
    distance=sdd,
    direct_beam=direct_beam,
    custom_scan=custom_scan,
    custom_images=custom_images,
    custom_monitor=custom_monitor,
    custom_motors=custom_motors,
    is_series=is_series,
)

########################################
# print the current setup and detector #
########################################
print("\n##############\nSetup instance\n##############")
print(setup)
print("\n#################\nDetector instance\n#################")
print(detector)

############################################
# Initialize values for callback functions #
############################################
detector_plane = False
flag_mask = False
flag_aliens = False
plt.rcParams["keymap.quit"] = [
    "ctrl+w",
    "cmd+w",
]  # this one to avoid that q closes window (matplotlib default)

############################
# start looping over scans #
############################
root = tk.Tk()
root.withdraw()

for scan_idx, scan_nb in enumerate(scans, start=1):
    plt.ion()

    comment = user_comment  # re-initialize comment
    tmp_str = f"Scan {scan_idx}/{len(scans)}: S{scan_nb}"
    print(f'\n{"#" * len(tmp_str)}\n' + tmp_str + "\n" + f'{"#" * len(tmp_str)}')

    # initialize the paths
    setup.init_paths(
        sample_name=sample_name[scan_idx - 1],
        scan_number=scan_nb,
        root_folder=root_folder,
        save_dir=save_dir,
        save_dirname=save_dirname,
        verbose=True,
        specfile_name=specfile_name,
        template_imagefile=template_imagefile,
    )

    logfile = setup.create_logfile(
        scan_number=scan_nb, root_folder=root_folder, filename=detector.specfile
    )

    if normalize_method != "skip":
        comment = comment + "_norm"

    #############
    # Load data #
    #############
    if reload_previous:  # resume previous masking
        print("Resuming previous masking")
        file_path = filedialog.askopenfilename(
            initialdir=detector.scandir,
            title="Select data file",
            filetypes=[("NPZ", "*.npz")],
        )
        data, _ = util.load_file(file_path)
        nz, ny, nx = np.shape(data)

        # update savedir to save the data in the same directory as the reloaded data
        detector.savedir = os.path.dirname(file_path) + "/"

        file_path = filedialog.askopenfilename(
            initialdir=detector.savedir,
            title="Select mask file",
            filetypes=[("NPZ", "*.npz")],
        )
        mask, _ = util.load_file(file_path)

        if reload_orthogonal:  # the data is gridded in the orthonormal laboratory frame
            use_rawdata = False
            try:
                file_path = filedialog.askopenfilename(
                    initialdir=detector.savedir,
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

            normalize_method = (
                "skip"  # we assume that normalization was already performed
            )
            monitor = []  # we assume that normalization was already performed
            min_range = (nx / 2) * np.sqrt(
                2
            )  # used when fit_datarange is True, keep the full array because
            # we do not know the position of the origin of reciprocal space
            frames_logical = np.ones(nz)

            # bin data and mask if needed
            if (
                (detector.binning[0] != 1)
                or (detector.binning[1] != 1)
                or (detector.binning[2] != 1)
            ):
                print("Binning the reloaded orthogonal data by", detector.binning)
                data = util.bin_data(data, binning=detector.binning, debugging=False)
                mask = util.bin_data(mask, binning=detector.binning, debugging=False)
                mask[np.nonzero(mask)] = 1
                if len(q_values) != 0:
                    qx = q_values[0]
                    qz = q_values[1]
                    qy = q_values[2]
                    numz, numy, numx = len(qx), len(qz), len(qy)
                    qx = qx[: numz - (numz % detector.binning[2]) : detector.binning[2]]
                    # along z downstream, same binning as along x
                    qz = qz[: numy - (numy % detector.binning[1]) : detector.binning[1]]
                    # along y vertical, the axis of rotation
                    qy = qy[: numx - (numx % detector.binning[2]) : detector.binning[2]]
                    # along x outboard
                    del numz, numy, numx
        else:  # the data is in the detector frame
            data, mask, frames_logical, monitor = cdi.reload_cdi_data(
                logfile=logfile,
                scan_number=scan_nb,
                data=data,
                mask=mask,
                detector=detector,
                setup=setup,
                debugging=debug,
                normalize_method=normalize_method,
                photon_threshold=loading_threshold,
            )

    else:  # new masking process
        reload_orthogonal = False  # the data is in the detector plane
        flatfield = util.load_flatfield(flatfield_file)
        hotpix_array = util.load_hotpixels(hotpixels_file)
        background = util.load_background(background_file)

        data, mask, frames_logical, monitor = cdi.load_cdi_data(
            logfile=logfile,
            scan_number=scan_nb,
            detector=detector,
            setup=setup,
            bin_during_loading=bin_during_loading,
            flatfield=flatfield,
            hotpixels=hotpix_array,
            background=background,
            normalize=normalize_method,
            debugging=debug,
            photon_threshold=loading_threshold,
        )

    nz, ny, nx = np.shape(data)
    print("\nInput data shape:", nz, ny, nx)

    if not reload_orthogonal:
        dirbeam = int((setup.direct_beam[1] - detector.roi[2]) / detector.binning[2])
        # updated horizontal direct beam
        min_range = min(dirbeam, nx - dirbeam)  # crop at the maximum symmetrical range
        print(
            "\nMaximum symmetrical range with defined data along"
            " detector horizontal direction: 2*{0} pixels".format(min_range)
        )
        if min_range <= 0:
            raise ValueError(
                "error in calculating min_range, check the direct beam " "position"
            )

        if save_rawdata:
            np.savez_compressed(
                detector.savedir + "S" + str(scan_nb) + "_data_before_masking_stack",
                data=data,
            )
            if save_to_mat:
                # save to .mat, the new order is x y z
                # (outboard, vertical up, downstream)
                savemat(
                    detector.savedir
                    + "S"
                    + str(scan_nb)
                    + "_data_before_masking_stack.mat",
                    {"data": np.moveaxis(data, [0, 1, 2], [-1, -2, -3])},
                )

        if flag_interact:
            # masking step in the detector plane
            plt.ioff()
            width = 0
            max_colorbar = 5
            detector_plane = True
            flag_aliens = False
            flag_mask = True
            flag_pause = False  # press x to pause for pan/zoom
            previous_axis = None
            xy = []  # list of points for mask

            fig_mask = plt.figure(figsize=(12, 9))
            ax0 = fig_mask.add_subplot(121)
            ax1 = fig_mask.add_subplot(322)
            ax2 = fig_mask.add_subplot(324)
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
            fig_mask.set_facecolor(background_plot)
            plt.show()

            mask[np.nonzero(updated_mask)] = 1
            data = original_data
            detector_plane = False
            del fig_mask, original_data, updated_mask
            gc.collect()

        if use_rawdata:
            q_values = []
            binning_comment = (
                f"_{detector.preprocessing_binning[0]*detector.binning[0]}"
                f"_{detector.preprocessing_binning[1]*detector.binning[1]}"
                f"_{detector.preprocessing_binning[2]*detector.binning[2]}"
            )
            # binning along axis 0 is done after masking
            data[np.nonzero(mask)] = 0
        else:  # the data will be gridded, binning[0] is already set to 1
            # sample rotation around the vertical direction at P10:
            # the effective binning in axis 0 is preprocessing_binning[2]*binning[2]
            binning_comment = (
                f"_{detector.preprocessing_binning[2]*detector.binning[2]}"
                f"_{detector.preprocessing_binning[1]*detector.binning[1]}"
                f"_{detector.preprocessing_binning[2]*detector.binning[2]}"
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
            plt.savefig(
                detector.savedir
                + f"data_before_gridding_S{scan_nb}_{nz}_{ny}_{nx}"
                + binning_comment
                + ".png"
            )
            plt.close(fig)
            del tmp_data
            gc.collect()

            print("\nGridding the data in the orthonormal laboratory frame")
            data, mask, q_values, frames_logical = cdi.grid_cdi(
                data=data,
                mask=mask,
                logfile=logfile,
                detector=detector,
                setup=setup,
                frames_logical=frames_logical,
                correct_curvature=correct_curvature,
                fill_value=(0, fill_value_mask),
                debugging=debug,
            )

            # plot normalization by incident monitor for the gridded data
            if normalize_method != "skip":
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
                    is_orthogonal=not use_rawdata,
                    reciprocal_space=True,
                )

                fig.savefig(
                    detector.savedir
                    + f"monitor_gridded_S{scan_nb}_{nz}_{ny}_{nx}"
                    + binning_comment
                    + ".png"
                )
                if flag_interact:
                    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
                    cid = plt.connect("close_event", close_event)
                    fig.waitforbuttonpress()
                    plt.disconnect(cid)
                plt.close(fig)
                plt.ioff()
                del tmp_data
                gc.collect()

    else:  # reload_orthogonal=True, the data is already gridded,
        # binning was realized along each axis
        binning_comment = (
            f"_{detector.preprocessing_binning[0]*detector.binning[0]}"
            f"_{detector.preprocessing_binning[1]*detector.binning[1]}"
            f"_{detector.preprocessing_binning[2]*detector.binning[2]}"
        )

    nz, ny, nx = np.shape(data)
    plt.ioff()

    ##########################################
    # optional masking of zero photon events #
    ##########################################
    if mask_zero_event:
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
        is_orthogonal=not use_rawdata,
        reciprocal_space=True,
    )
    if debug:
        plt.savefig(
            detector.savedir
            + f"data_before_masking_S{scan_nb}_{nz}_{ny}_{nx}"
            + binning_comment
            + ".png"
        )

    if flag_interact:
        fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
        cid = plt.connect("close_event", close_event)
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
        is_orthogonal=not use_rawdata,
        reciprocal_space=True,
    )
    if debug:
        plt.savefig(
            detector.savedir
            + f"mask_before_masking_S{scan_nb}_{nz}_{ny}_{nx}"
            + binning_comment
            + ".png"
        )

    if flag_interact:
        fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
        cid = plt.connect("close_event", close_event)
        fig.waitforbuttonpress()
        plt.disconnect(cid)
    plt.close(fig)

    ###############################################
    # save the orthogonalized diffraction pattern #
    ###############################################
    if not use_rawdata and len(q_values) != 0:
        qx = q_values[0]  # downstream
        qz = q_values[1]  # vertical up
        qy = q_values[2]  # outboard

        if save_to_vti:
            (
                nqx,
                nqz,
                nqy,
            ) = (
                data.shape
            )  # in nexus z downstream, y vertical / in q z vertical, x downstream
            print("\ndqx, dqy, dqz = ", qx[1] - qx[0], qy[1] - qy[0], qz[1] - qz[0])
            # in nexus z downstream, y vertical / in q z vertical, x downstream
            qx0 = qx.min()
            dqx = (qx.max() - qx0) / nqx
            qy0 = qy.min()
            dqy = (qy.max() - qy0) / nqy
            qz0 = qz.min()
            dqz = (qz.max() - qz0) / nqz

            gu.save_to_vti(
                filename=os.path.join(
                    detector.savedir,
                    "S" + str(scan_nb) + "_ortho_int" + comment + ".vti",
                ),
                voxel_size=(dqx, dqz, dqy),
                tuple_array=data,
                tuple_fieldnames="int",
                origin=(qx0, qz0, qy0),
            )

    if flag_interact:
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
        if not use_rawdata:
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
        fig_mask.set_facecolor(background_plot)
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
            is_orthogonal=not use_rawdata,
            reciprocal_space=True,
        )

        fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
        cid = plt.connect("close_event", close_event)
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
            is_orthogonal=not use_rawdata,
            reciprocal_space=True,
        )

        fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
        cid = plt.connect("close_event", close_event)
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
        if not use_rawdata:
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
        fig_mask.set_facecolor(background_plot)
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
    if flag_medianfilter in {"mask_isolated", "interp_isolated"}:
        print("\nFiltering isolated pixels")
        nb_pix = 0
        for idx in range(nz):  # filter only frames whith data (not padded)
            data[idx, :, :], numb_pix, mask[idx, :, :] = util.mean_filter(
                data=data[idx, :, :],
                nb_neighbours=medfilt_order,
                mask=mask[idx, :, :],
                interpolate=flag_medianfilter,
                min_count=3,
                debugging=debug,
            )
            nb_pix = nb_pix + numb_pix
            print("Processed image nb: ", idx)
        if flag_medianfilter == "mask_isolated":
            print("\nTotal number of masked isolated pixels: ", nb_pix)
        if flag_medianfilter == "interp_isolated":
            print("\nTotal number of interpolated isolated pixels: ", nb_pix)

    elif flag_medianfilter == "median":  # apply median filter
        for idx in range(nz):  # filter only frames whith data (not padded)
            data[idx, :, :] = scipy.signal.medfilt2d(data[idx, :, :], [3, 3])
        print("\nApplying median filtering")
    else:
        print("\nSkipping median filtering")

    ##########################
    # apply photon threshold #
    ##########################
    if photon_threshold != 0:
        mask[data < photon_threshold] = 1
        data[data < photon_threshold] = 0
        print("\nApplying photon threshold < ", photon_threshold)

    ########################################
    # check for nans / inf, convert to int #
    ########################################
    plt.ion()
    nz, ny, nx = np.shape(data)
    print("\nData size after masking:", nz, ny, nx)

    # check for Nan
    mask[np.isnan(data)] = 1
    data[np.isnan(data)] = 0
    mask[np.isnan(mask)] = 1
    # check for Inf
    mask[np.isinf(data)] = 1
    data[np.isinf(data)] = 0
    mask[np.isinf(mask)] = 1

    data[mask == 1] = 0
    if save_asint:
        data = data.astype(int)

    ####################
    # debugging plots  #
    ####################
    if debug:
        z0, y0, x0 = center_of_mass(data)
        fig, _, _ = gu.multislices_plot(
            data,
            sum_frames=False,
            scale="log",
            plot_colorbar=True,
            vmin=0,
            title="Masked data",
            slice_position=[int(z0), int(y0), int(x0)],
            is_orthogonal=not use_rawdata,
            reciprocal_space=True,
        )
        plt.savefig(
            detector.savedir
            + f"middle_frame_S{scan_nb}_{nz}_{ny}_{nx}"
            + binning_comment
            + ".png"
        )
        if not flag_interact:
            plt.close(fig)

        fig, _, _ = gu.multislices_plot(
            data,
            sum_frames=True,
            scale="log",
            plot_colorbar=True,
            vmin=0,
            title="Masked data",
            is_orthogonal=not use_rawdata,
            reciprocal_space=True,
        )
        plt.savefig(
            detector.savedir
            + f"sum_S{scan_nb}_{nz}_{ny}_{nx}"
            + binning_comment
            + ".png"
        )
        if not flag_interact:
            plt.close(fig)

        fig, _, _ = gu.multislices_plot(
            mask,
            sum_frames=True,
            scale="linear",
            plot_colorbar=True,
            vmin=0,
            vmax=(nz, ny, nx),
            title="Mask",
            is_orthogonal=not use_rawdata,
            reciprocal_space=True,
        )
        plt.savefig(
            detector.savedir
            + f"mask_S{scan_nb}_{nz}_{ny}_{nx}"
            + binning_comment
            + ".png"
        )
        if not flag_interact:
            plt.close(fig)

    if not use_rawdata and fit_datarange:
        ############################################################
        # select the largest cubic array fitting inside data range #
        ############################################################
        # this is to avoid having large masked areas near the corner of the area
        # which is a side effect of regridding the data from cylindrical coordinates
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
        print("\nData size after taking the largest data-defined area:", data.shape)
        if len(q_values) != 0:
            qx = qx[
                (nz - final_nxz) // 2 : (nz - final_nxz) // 2 + final_nxz
            ]  # along Z
            qy = qy[
                (nz - final_nxz) // 2 : (nz - final_nxz) // 2 + final_nxz
            ]  # along X
            # qz (along Y) keeps the same number of pixels
        else:
            print("fit_datarange: q values are not provided")

    ##############################################################
    # only for non gridded data, bin the stacking axis           #
    # the detector plane was already binned during data loading  #
    ##############################################################
    if (
        detector.binning[0] != 1 and not reload_orthogonal
    ):  # for data to be gridded, binning[0] is set to 1
        data = util.bin_data(data, (detector.binning[0], 1, 1), debugging=False)
        mask = util.bin_data(mask, (detector.binning[0], 1, 1), debugging=False)
        mask[np.nonzero(mask)] = 1

    nz, ny, nx = data.shape
    print("\nData size after binning the stacking dimension:", data.shape)
    comment = f"{comment}_{nz}_{ny}_{nx}" + binning_comment

    ############################
    # save final data and mask #
    ############################
    print("\nSaving directory:", detector.savedir)
    print("Data type before saving:", data.dtype)
    mask[np.nonzero(mask)] = 1
    mask = mask.astype(int)
    print("Mask type before saving:", mask.dtype)
    if not use_rawdata and len(q_values) != 0:
        if save_to_npz:
            np.savez_compressed(
                detector.savedir + f"QxQzQy_S{scan_nb}" + comment, qx=qx, qz=qz, qy=qy
            )
        if save_to_mat:
            savemat(detector.savedir + f"S{scan_nb}_qx.mat", {"qx": qx})
            savemat(detector.savedir + f"S{scan_nb}_qy.mat", {"qy": qy})
            savemat(detector.savedir + f"S{scan_nb}_qz.mat", {"qz": qz})
        fig, _, _ = gu.contour_slices(
            data,
            (qx, qz, qy),
            sum_frames=True,
            title="Final data",
            levels=np.linspace(
                0, int(np.log10(data.max(initial=None))), 150, endpoint=False
            ),
            plot_colorbar=True,
            scale="log",
            is_orthogonal=True,
            reciprocal_space=True,
        )
        fig.savefig(
            detector.savedir + f"final_reciprocal_space_S{scan_nb}" + comment + ".png"
        )
        plt.close(fig)

    if save_to_npz:
        np.savez_compressed(detector.savedir + f"S{scan_nb}_pynx" + comment, data=data)
        np.savez_compressed(
            detector.savedir + f"S{scan_nb}_maskpynx" + comment, mask=mask
        )

    if save_to_mat:
        # save to .mat, the new order is x y z (outboard, vertical up, downstream)
        savemat(
            detector.savedir + "S" + str(scan_nb) + "_data.mat",
            {"data": np.moveaxis(data.astype(np.float32), [0, 1, 2], [-1, -2, -3])},
        )
        savemat(
            detector.savedir + "S" + str(scan_nb) + "_mask.mat",
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
        is_orthogonal=not use_rawdata,
        reciprocal_space=True,
    )
    plt.savefig(detector.savedir + f"finalsum_S{scan_nb}" + comment + ".png")
    if not flag_interact:
        plt.close(fig)

    fig, _, _ = gu.multislices_plot(
        mask,
        sum_frames=True,
        scale="linear",
        plot_colorbar=True,
        vmin=0,
        vmax=(nz, ny, nx),
        title="Final mask",
        is_orthogonal=not use_rawdata,
        reciprocal_space=True,
    )
    plt.savefig(detector.savedir + f"finalmask_S{scan_nb}" + comment + ".png")
    if not flag_interact:
        plt.close(fig)

    del data, mask
    gc.collect()

    if len(scans) > 1:
        plt.close("all")

print("\nEnd of script")
plt.ioff()
plt.show()
