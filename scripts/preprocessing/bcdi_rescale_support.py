#!/usr/bin/env python3

import gc
import sys
import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr
import numpy as np
from scipy.interpolate import RegularGridInterpolator

import bcdi.algorithms.algorithms_utils as algu
import bcdi.graph.graph_utils as gu
import bcdi.postprocessing.postprocessing_utils as pu
import bcdi.utils.utilities as util
from bcdi.experiment.setup import Setup
from bcdi.graph.colormap import ColormapFactory

helptext = """
Create a support from a reconstruction, using the indicated threshold. The support
can be cropped/padded to a desired shape. In real space the CXI convention is used:
z downstream, y vertical up, x outboard. In reciprocal space, the following
convention is used: qx downtream, qz vertical up, qy outboard.
"""

root_folder = "D:/data/P10_August2020_CDI/data/mag_3_macro1/BCDI/"
support_threshold = 0.01  # in % of the normalized absolute value
pynx_shape = (
    90,
    90,
    90,
)  # shape of the array used for phasing and finding the support (after binning_pynx)
binning_pynx = (1, 1, 1)  # binning that was used in PyNX during phasing
output_shape = (
    540,
    540,
    540,
)  # shape of the array for later phasing (before binning_output)
# if the data and q-values were binned beforehand,
# use the binned shape and binning_output=(1,1,1)
binning_output = (1, 1, 1)  # binning that will be used in PyNX for later phasing
qvalues_binned = True  # if True, the q values provided are expected to be binned
# (binning_pynx & binning_output)
flag_interact = True  # if False, will skip thresholding and masking
binary_support = True  # True to save the support as an array of 0 and 1
save_intermediate = False  # if True, will save the masked data just after the
# interactive masking, before applying
# other filtering and interpolation
is_ortho = True  # True if the data is already orthogonalized
center = True  # will center the support based on the center of mass
flip_reconstruction = False  # True if you want to get the conjugate object
roll_modes = (
    0,
    0,
    0,
)  # correct a roll of few pixels after the decomposition into modes in PyNX.
# axis=(0, 1, 2)
roll_centering = (
    0,
    0,
    0,
)  # roll applied after masking when centering by center of mass is not optimal.
# axis=(0, 1, 2)
background_plot = (
    "0.5"  # in level of grey in [0,1], 0 being dark. For visual comfort during masking
)
save_fig = True  # if True, will save the figure of the final support
comment = ""  # should start with _
#####################################
# parameters for gaussian filtering #
#####################################
filter_name = (
    "skip"  # apply a filtering kernel to the support, 'skip' or 'gaussian_highpass'
)
gaussian_sigma = 4.0  # sigma of the gaussian filter
######################################################################
# parameters for image deconvolution using Richardson-Lucy algorithm #
######################################################################
psf_iterations = 0  # number of iterations of Richardson-Lucy deconvolution,
# leave it to 0 if unwanted
psf_shape = (10, 10, 10)
psf = util.gaussian_window(window_shape=psf_shape, sigma=0.3, mu=0.0, debugging=False)
###########################
# experimental parameters #
###########################
energy = 10235  # in eV
tilt_angle = 0.25  # in degrees
distance = 5  # in m
pixel_x = 75e-06  # in m, horizontal pixel size of the detector,
# including an eventual preprocessing binning
pixel_y = 75e-06  # in m, vertical pixel size of the detector,
# including an eventual preprocessing binning
###########################################################################
# parameters used only when the data is in the detector frame (Bragg CDI) #
###########################################################################
beamline = "ID01"  # name of the beamline, used for data loading and
# normalization by monitor and orthogonalisation
# supported beamlines: 'ID01', 'SIXS_2018', 'SIXS_2019', 'CRISTAL', 'P10', '34ID'
rocking_angle = "outofplane"  # "outofplane" or "inplane"
outofplane_angle = 35.2694  # detector delta ID01, delta SIXS, gamma 34ID
inplane_angle = -2.5110  # detector nu ID01, gamma SIXS, tth 34ID
grazing_angle = 0  # in degrees, incident angle for in-plane rocking curves
# (eta ID01, th 34ID, beta SIXS)
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
    flag_pause==1 or if the mouse is out of plot axes, it will not register the click.

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
                "Please select mask polygon vertices within the same subplot:"
                " restart masking..."
            )
            xy = []
            previous_axis = None


def press_key(event):
    """
    Interact with a plot for masking parasitic diffraction intensity or detector gaps

    :param event: button press event
    """
    global original_data, original_mask, data, mask, frame_index, width, flag_aliens
    global flag_mask, flag_pause, xy, fig_mask, max_colorbar, ax0, ax1, ax2
    global previous_axis, info_text, is_ortho, my_cmap

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
            invert_yaxis = is_ortho
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
                    mask,
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
                    original_mask=original_mask,
                    updated_data=data,
                    updated_mask=mask,
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
                plt.close(fig_mask)

    except AttributeError:  # mouse pointer out of axes
        pass


###################
# define colormap #
###################
my_cmap = ColormapFactory().cmap
plt.rcParams["keymap.fullscreen"] = [""]

#################
# load the data #
#################
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    initialdir=root_folder,
    title="Select the reconstruction",
    filetypes=[("HDF5", "*.h5"), ("NPZ", "*.npz"), ("CXI", "*.cxi")],
)
data, _ = util.load_file(file_path)
nz, ny, nx = data.shape
data = np.roll(data, roll_modes, axis=(0, 1, 2))

if flip_reconstruction:
    data = pu.flip_reconstruction(data, debugging=True)

data = abs(data)  # take the real part

###################################
# clean interactively the support #
###################################
if flag_interact:
    data = data / data.max(initial=None)  # normalize
    data[data < support_threshold] = 0

    fig, _, _ = gu.multislices_plot(
        data,
        sum_frames=False,
        scale="linear",
        plot_colorbar=True,
        vmin=0,
        vmax=1,
        title="Support before masking",
        is_orthogonal=True,
        reciprocal_space=False,
    )
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
    cid = plt.connect("close_event", close_event)
    fig.waitforbuttonpress()
    plt.disconnect(cid)
    plt.close(fig)

    #############################################
    # mask the projected data in each dimension #
    #############################################
    plt.ioff()
    width = 0
    max_colorbar = np.rint(np.log10(max(data.shape) * data.max(initial=None)))
    flag_aliens = False
    flag_mask = True
    flag_pause = False  # press x to pause for pan/zoom
    previous_axis = None
    mask = np.zeros(data.shape)
    xy = []  # list of points for mask

    fig_mask, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))
    fig_mask.canvas.mpl_disconnect(fig_mask.canvas.manager.key_press_handler_id)
    original_data = np.copy(data)
    original_mask = np.copy(mask)
    data[mask == 1] = 0  # will appear as grey in the log plot (nan)
    ax0.imshow(np.log10(abs(data).sum(axis=0)), vmin=0, vmax=max_colorbar, cmap=my_cmap)
    ax1.imshow(np.log10(abs(data).sum(axis=1)), vmin=0, vmax=max_colorbar, cmap=my_cmap)
    ax2.imshow(np.log10(abs(data).sum(axis=2)), vmin=0, vmax=max_colorbar, cmap=my_cmap)
    ax3.set_visible(False)
    ax0.axis("scaled")
    ax1.axis("scaled")
    ax2.axis("scaled")
    if is_ortho:
        ax0.invert_yaxis()  # detector Y is vertical down
    ax0.set_title("XY")
    ax1.set_title("XZ")
    ax2.set_title("YZ")
    fig_mask.text(0.60, 0.45, "click to select the vertices of a polygon mask", size=12)
    fig_mask.text(
        0.60, 0.40, "then p to apply and see the result; r to reset points", size=12
    )
    fig_mask.text(0.60, 0.30, "x to pause/resume masking for pan/zoom", size=12)
    fig_mask.text(0.60, 0.25, "up/down larger/smaller masking box ; f fill", size=12)
    fig_mask.text(
        0.60, 0.20, "m mask ; b unmask ; right darker ; left brighter", size=12
    )
    fig_mask.text(0.60, 0.15, "p plot full masked data ; a restart ; q quit", size=12)
    info_text = fig_mask.text(0.60, 0.05, "masking enabled", size=16)
    plt.tight_layout()
    plt.connect("key_press_event", press_key)
    plt.connect("button_press_event", on_click)
    fig_mask.set_facecolor(background_plot)
    plt.show()

    mask[mask == -1] = 0
    # clear the filled points from the mask since we do not want to mask them later
    mask[np.nonzero(mask)] = 1  # ensure that masked voxels appear as 1 in the mask
    data[np.nonzero(mask)] = 0
    del fig_mask, flag_pause, flag_mask, original_data, original_mask
    gc.collect()

    ############################################
    # mask individual frames in each dimension #
    ############################################
    nz, ny, nx = np.shape(data)
    width = 5
    max_colorbar = 1
    flag_mask = False
    flag_aliens = True

    fig_mask, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))
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
    if is_ortho:
        ax0.invert_yaxis()  # detector Y is vertical down
    ax0.set_title("XY - Frame " + str(frame_index[0] + 1) + "/" + str(nz))
    ax1.set_title("XZ - Frame " + str(frame_index[1] + 1) + "/" + str(ny))
    ax2.set_title("YZ - Frame " + str(frame_index[2] + 1) + "/" + str(nx))
    fig_mask.text(
        0.60, 0.30, "m mask ; b unmask ; u next frame ; d previous frame", size=12
    )
    fig_mask.text(
        0.60, 0.25, "up larger ; down smaller ; right darker ; left brighter", size=12
    )
    fig_mask.text(0.60, 0.20, "p plot full image ; q quit", size=12)
    plt.tight_layout()
    plt.connect("key_press_event", press_key)
    fig_mask.set_facecolor(background_plot)
    plt.show()

    mask[mask == -1] = 0
    # clear the filled points from the mask since we do not want to mask them later
    mask[np.nonzero(mask)] = 1  # ensure that masked voxels appear as 1 in the mask
    data[np.nonzero(mask)] = 0
    del fig_mask, original_data, original_mask, mask
    gc.collect()

#######################################################
# optional: save the masked data for future reloading #
#######################################################
if save_intermediate:
    filename = (
        "intermediate_pynx shape_"
        + str(pynx_shape)
        + "_binning pynx_"
        + str(binning_pynx)
        + comment
    )
    np.savez_compressed(root_folder + filename + ".npz", obj=data)

############################################
# plot the support with the original shape #
############################################
fig, _, _ = gu.multislices_plot(
    data,
    sum_frames=False,
    scale="linear",
    plot_colorbar=True,
    vmin=0,
    title="Support after masking\n",
    is_orthogonal=True,
    reciprocal_space=False,
)
fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
cid = plt.connect("close_event", close_event)
fig.waitforbuttonpress()
plt.disconnect(cid)
plt.close(fig)

###########################################
# optional: Richardson-Lucy deconvolution #
###########################################
if psf_iterations > 0:
    data = algu.deconvolution_rl(
        data, psf=psf, iterations=psf_iterations, debugging=True
    )

############################
# optional: apply a filter #
############################
if filter_name != "skip":
    comment = comment + "_" + filter_name
    data = pu.filter_3d(
        data, filter_name=filter_name, sigma=gaussian_sigma, debugging=True
    )
    fig, _, _ = gu.multislices_plot(
        data,
        sum_frames=False,
        scale="linear",
        plot_colorbar=True,
        vmin=0,
        title="Support after filtering\n",
        is_orthogonal=True,
        reciprocal_space=False,
    )
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
    cid = plt.connect("close_event", close_event)
    fig.waitforbuttonpress()
    plt.disconnect(cid)
    plt.close(fig)

#################################################################
# normalize, threshold and convert the data to a binary support #
#################################################################
data = data / data.max(initial=None)  # normalize
data[data < support_threshold] = 0
if binary_support:
    data[np.nonzero(data)] = 1  # change data into a support

###################################################
# calculate shapes considering binning parameters #
###################################################
unbinned_shape = [
    int(pynx_shape[idx] * binning_pynx[idx]) for idx in range(0, len(binning_pynx))
]
print(
    "Original data shape after considering PyNX binning and PyNX shape:", unbinned_shape
)
print(
    "Original voxel sizes in detector coordinates based on "
    "experimental parameters (ver, hor): "
    f"{12.398 * 1e-7 / energy * distance / (unbinned_shape[1] * pixel_y) * 1e9:.2f} nm,"
    f" {12.398 * 1e-7 / energy * distance / (unbinned_shape[2] * pixel_x) * 1e9:.2f} nm"
)

rebinned_shape = [
    int(output_shape[idx] / binning_output[idx])
    for idx in range(0, len(binning_output))
]
print(
    "Final data shape after considering output_shape and binning_output :",
    rebinned_shape,
)

######################
# center the support #
######################
if center:
    data = pu.center_com(data)
# Use user-defined roll when the center by COM is not optimal
data = np.roll(data, roll_centering, axis=(0, 1, 2))

#################################
# rescale the support if needed #
#################################
if not all(
    i == j for i, j in zip(output_shape, unbinned_shape)
):  # accomodate for different object types
    print("\nCalculating voxel sizes...")
    if is_ortho:
        # load the original q values to calculate actual real space voxel sizes
        file_path = filedialog.askopenfilename(
            initialdir=root_folder,
            title="Select original q values",
            filetypes=[("NPZ", "*.npz")],
        )
        q_values = np.load(file_path)
        qx = q_values["qx"]  # 1D array
        qy = q_values["qy"]  # 1D array
        qz = q_values["qz"]  # 1D array
        # crop q to accomodate a shape change of the original array
        # (e.g. cropping to fit FFT shape requirement)
        if qvalues_binned:
            if len(qx) < pynx_shape[0]:
                raise ValueError(
                    "qx declared binned, its length should be >= " "pynx_shape[0]"
                )
            if len(qy) < pynx_shape[2]:
                raise ValueError(
                    "qy declared binned, its length should be >= " "pynx_shape[2]"
                )
            if len(qz) < pynx_shape[1]:
                raise ValueError(
                    "qz declared binned, its length should be >= " "pynx_shape[1]"
                )
            qx = util.crop_pad_1d(qx, pynx_shape[0])  # qx along z
            qy = util.crop_pad_1d(qy, pynx_shape[2])  # qy along x
            qz = util.crop_pad_1d(qz, pynx_shape[1])  # qz along y
        else:
            if len(qx) < unbinned_shape[0]:
                raise ValueError(
                    "qx declared unbinned, its length should be >= " "unbinned_shape[0]"
                )
            if len(qy) < unbinned_shape[2]:
                raise ValueError(
                    "qy declared unbinned, its length should be >= " "unbinned_shape[2]"
                )
            if len(qz) < unbinned_shape[1]:
                raise ValueError(
                    "qz declared unbinned, its length should be >= " "unbinned_shape[1]"
                )
            qx = util.crop_pad_1d(qx, unbinned_shape[0])  # qx along z
            qy = util.crop_pad_1d(qy, unbinned_shape[2])  # qy along x
            qz = util.crop_pad_1d(qz, unbinned_shape[1])  # qz along y

        print("Length(q_original)=", len(qx), len(qz), len(qy), "(qx, qz, qy)")
        voxelsize_z = 2 * np.pi / (qx.max() - qx.min())  # qx along z
        voxelsize_x = 2 * np.pi / (qy.max() - qy.min())  # qy along x
        voxelsize_y = 2 * np.pi / (qz.max() - qz.min())  # qz along y

        # load the q values of the desired shape
        # and calculate corresponding real space voxel sizes
        file_path = filedialog.askopenfilename(
            initialdir=root_folder,
            title="Select q values for the new shape",
            filetypes=[("NPZ", "*.npz")],
        )
        q_values = np.load(file_path)
        newqx = q_values["qx"]  # 1D array
        newqy = q_values["qy"]  # 1D array
        newqz = q_values["qz"]  # 1D array
        # crop q to accomodate a shape change of the original array
        # (e.g. cropping to fit FFT shape requirement)
        if qvalues_binned:
            if len(newqx) < rebinned_shape[0]:
                raise ValueError(
                    "newqx declared binned, its length should be >= "
                    "rebinned_shape[0]"
                )
            if len(newqy) < rebinned_shape[2]:
                raise ValueError(
                    "newqy declared binned, its length should be >= "
                    "rebinned_shape[2]"
                )
            if len(newqz) < rebinned_shape[1]:
                raise ValueError(
                    "newqz declared binned, its length should be >= "
                    "rebinned_shape[1]"
                )
        else:
            if len(newqx) < output_shape[0]:
                raise ValueError(
                    "newqx declared binned, its length should be >= " "output_shape[0]"
                )
            if len(newqy) < output_shape[2]:
                raise ValueError(
                    "newqy declared binned, its length should be >=" " output_shape[2]"
                )
            if len(newqz) < output_shape[1]:
                raise ValueError(
                    "newqz declared binned, its length should be >=" " output_shape[1]"
                )
            newqx = util.crop_pad_1d(newqx, output_shape[0])  # qx along z
            newqy = util.crop_pad_1d(newqy, output_shape[2])  # qy along x
            newqz = util.crop_pad_1d(newqz, output_shape[1])  # qz along y

        print("Length(q_output)=", len(newqx), len(newqz), len(newqy), "(qx, qz, qy)")
        newvoxelsize_z = 2 * np.pi / (newqx.max() - newqx.min())  # qx along z
        newvoxelsize_x = 2 * np.pi / (newqy.max() - newqy.min())  # qy along x
        newvoxelsize_y = 2 * np.pi / (newqz.max() - newqz.min())  # qz along y

    else:  # data in detector frame
        setup = Setup(
            parameters={
                "beamline": beamline,
                "energy": energy,
                "outofplane_angle": outofplane_angle,
                "inplane_angle": inplane_angle,
                "tilt_angle": tilt_angle,
                "rocking_angle": rocking_angle,
                "detector_distance": distance,
                "grazing_angle": grazing_angle,
            },
        )

        voxelsize_z, voxelsize_y, voxelsize_x = setup.voxel_sizes(
            unbinned_shape, tilt_angle=tilt_angle, pixel_x=pixel_x, pixel_y=pixel_y
        )
        newvoxelsize_z, newvoxelsize_y, newvoxelsize_x = setup.voxel_sizes(
            unbinned_shape, tilt_angle=tilt_angle, pixel_x=pixel_x, pixel_y=pixel_y
        )

    print(
        "Original voxel sizes zyx (nm):",
        f"{voxelsize_z:.2f}, {voxelsize_y:.2f}, {voxelsize_x:.2f}",
    )
    print(
        "Output voxel sizes zyx (nm):",
        f"{newvoxelsize_z:.2f}, {newvoxelsize_y:.2f}, {newvoxelsize_x:.2f}",
    )

    # Interpolate the support
    print("\nInterpolating the support...")
    data = util.crop_pad(data, pynx_shape)  # the data could be cropped near the support
    fig, _, _ = gu.multislices_plot(
        data,
        sum_frames=True,
        scale="linear",
        plot_colorbar=True,
        vmin=0,
        title="Support before interpolation\n",
        is_orthogonal=True,
        reciprocal_space=False,
    )

    rgi = RegularGridInterpolator(
        (
            np.arange(-pynx_shape[0] // 2, pynx_shape[0] // 2, 1) * voxelsize_z,
            np.arange(-pynx_shape[1] // 2, pynx_shape[1] // 2, 1) * voxelsize_y,
            np.arange(-pynx_shape[2] // 2, pynx_shape[2] // 2, 1) * voxelsize_x,
        ),
        data,
        method="linear",
        bounds_error=False,
        fill_value=0,
    )

    new_z, new_y, new_x = np.meshgrid(
        np.arange(-rebinned_shape[0] // 2, rebinned_shape[0] // 2, 1) * newvoxelsize_z,
        np.arange(-rebinned_shape[1] // 2, rebinned_shape[1] // 2, 1) * newvoxelsize_y,
        np.arange(-rebinned_shape[2] // 2, rebinned_shape[2] // 2, 1) * newvoxelsize_x,
        indexing="ij",
    )

    new_support = rgi(
        np.concatenate(
            (
                new_z.reshape((1, new_z.size)),
                new_y.reshape((1, new_z.size)),
                new_x.reshape((1, new_z.size)),
            )
        ).transpose()
    )
    new_support = new_support.reshape(rebinned_shape).astype(data.dtype)

    print("Shape after interpolating the support:", new_support.shape)

else:  # no need for interpolation, the data may be cropped near the support
    new_support = util.crop_pad(data, rebinned_shape)

if binary_support:
    new_support[np.nonzero(new_support)] = 1

###################################################
# save and plot the support with the output shape #
###################################################
filename = (
    "support_"
    + str(rebinned_shape[0])
    + "_"
    + str(rebinned_shape[1])
    + "_"
    + str(rebinned_shape[2])
    + "_"
    + str(binning_output[0])
    + "_"
    + str(binning_output[1])
    + "_"
    + str(binning_output[2])
    + comment
)
np.savez_compressed(root_folder + filename + ".npz", obj=new_support)
fig, _, _ = gu.multislices_plot(
    new_support,
    sum_frames=False,
    scale="linear",
    plot_colorbar=True,
    vmin=0,
    title="Support after interpolation\n",
    is_orthogonal=True,
    reciprocal_space=False,
)
if save_fig:
    fig.savefig(root_folder + filename + ".png")

print("support shape", new_support.shape)
print("End of script")
plt.ioff()
plt.show()
