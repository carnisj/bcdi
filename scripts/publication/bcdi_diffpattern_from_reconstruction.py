#!/usr/bin/env python3

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import pathlib
import tkinter as tk
from functools import reduce
from numbers import Real
from tkinter import filedialog

import matplotlib as mpl
import matplotlib.ticker as ticker
import numpy as np
from matplotlib import pyplot as plt
from numpy.fft import fftn, fftshift
from scipy.ndimage.measurements import center_of_mass

import bcdi.graph.graph_utils as gu
import bcdi.postprocessing.postprocessing_utils as pu
import bcdi.utils.utilities as util
import bcdi.utils.validation as valid
from bcdi.graph.colormap import ColormapFactory

helptext = """
Calculate the diffraction pattern corresponding to a reconstructed 3D crystal
(output of phase retrieval). The crystal is expected to be expressed in an orthonormal
frame, and voxel sizes must be provided (voxel sizes can be different in each
dimension).

If q values are provided, the crystal will be resampled so that the extent in q given
by the direct space voxel sizes matches the extent defined by q values. If q values are
not provided, the crystal is padded to the user-defined shape before calculating the
diffraction pattern.

The reconstructed crystal file should be a .NPZ with field names 'amp' for the modulus
and 'displacement' for the phase. Corresponding q values can be loaded optionally.
"""

scan = 1  # scan number
root_folder = "D:/data/P10_2nd_test_isosurface_Dec2020/data_nanolab/"
sample_name = "dataset_"
datadir = root_folder + sample_name + str(scan) + "_newpsf/result/"
voxel_sizes = (
    5  # number (if identical for all dimensions) or tuple of 3 voxel sizes in nm
)
flip_phase = True  # True to flip the phase
# (-1*phase is saved in the field 'displacement' of the NPZ file)
peak_value = 189456
# 189456  # dataset 1
# 242428  # dataset 2
# correction due to the loss of normalization with the mode decomposition,
# leave None otherwise.
# The diffraction intensity will be normalized so that the integrated intensity
# in a ROI of 7x7x7 voxels around the
# center of mass of the Bragg peak equals this value.
# mode_factor = 0.2740 dataset_1_newpsf
# mode_factor = 0.2806 dataset_1_nopsf
# mode_factor = 0.2744 dataset_2_pearson97.5_newpsf
load_qvalues = True  # True to load the q values.
# It expects a single npz file with fieldnames 'qx', 'qy' and 'qz'
padding_shape = (
    300,
    512,
    400,
)  # the object is padded to that shape before calculating its diffraction pattern.
# It will be overrident if it does not match the shape defined by q values.
##############################
# settings related to saving #
##############################
savedir = (
    datadir + "diffpattern_from_reconstruction/"
)  # results will be saved here, if None it will default to datadir
save_qyqz = True  # True to save the strain in QyQz plane
save_qyqx = True  # True to save the strain in QyQx plane
save_qzqx = True  # True to save the strain in QzQx plane
save_sum = False  # True to save the summed diffraction pattern in the detector,
# False to save the central slice only
comment = ""  # string to add to the filename when saving, should start with "_"
##########################
# settings for the plots #
##########################
tick_direction = "out"  # 'out', 'in', 'inout'
tick_length = 8  # in plots
tick_width = 2  # in plots
tick_spacing = (
    0.025,
    0.025,
    0.025,
)  # tuple of three numbers, in 1/A. Leave None for default.
num_ticks = 5  # number of ticks to use in axes when tick_spacing is not defined
colorbar_range = (
    -1,
    3,
)  # (vmin, vmax) log scale in photon counts, leave None for default.
debug = False  # True to see more plots
grey_background = False  # True to set nans to grey in the plots
##################################
# end of user-defined parameters #
##################################

#########################
# check some parameters #
#########################
valid_name = "diffpattern_from_reconstruction"
savedir = savedir or datadir
pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)

if isinstance(voxel_sizes, Real):
    voxel_sizes = (voxel_sizes,) * 3
valid.valid_container(
    voxel_sizes,
    container_types=(list, tuple, np.ndarray),
    length=3,
    item_types=Real,
    min_excluded=0,
    name=valid_name,
)

valid.valid_container(
    padding_shape,
    container_types=(tuple, list, np.ndarray),
    item_types=int,
    min_excluded=0,
    length=3,
    name=valid_name,
)

valid.valid_item(
    peak_value, allowed_types=Real, min_excluded=0, allow_none=True, name=valid_name
)

valid.valid_container(
    (
        load_qvalues,
        flip_phase,
        save_qyqz,
        save_qyqx,
        save_qzqx,
        save_sum,
        debug,
        grey_background,
    ),
    container_types=tuple,
    item_types=bool,
    name=valid_name,
)

if len(comment) != 0 and not comment.startswith("_"):
    comment = "_" + comment
if save_sum:
    comment = comment + "_sum"

if tick_direction not in {"out", "in", "inout"}:
    raise ValueError("tick_direction should be 'out', 'in' or 'inout'")
valid.valid_item(tick_length, allowed_types=int, min_excluded=0, name=valid_name)
valid.valid_item(tick_width, allowed_types=int, min_excluded=0, name=valid_name)
if isinstance(tick_spacing, Real) or tick_spacing is None:
    tick_spacing = (tick_spacing,) * 3
valid.valid_container(
    tick_spacing,
    container_types=(tuple, list, np.ndarray),
    allow_none=True,
    item_types=Real,
    min_excluded=0,
    name=valid_name,
)
valid.valid_item(num_ticks, allowed_types=int, min_excluded=0, name=valid_name)
valid.valid_container(
    colorbar_range,
    container_types=(tuple, list, np.ndarray),
    item_types=Real,
    length=2,
    allow_none=True,
    name=valid_name,
)

#############################
# define default parameters #
#############################
mpl.rcParams["axes.linewidth"] = tick_width  # set the linewidth globally

if grey_background:
    bad_color = "0.7"
else:
    bad_color = "1.0"  # white background

my_cmap = ColormapFactory(bad_color=bad_color).cmap

labels = ("Qx", "Qz", "Qy")

if load_qvalues:
    draw_ticks = True
    cbar_pad = (
        0.2  # pad value for the offset of the colorbar, to avoid overlapping with ticks
    )
    unit = " 1/A"
else:
    draw_ticks = False
    cbar_pad = (
        0.1  # pad value for the offset of the colorbar, to avoid overlapping with ticks
    )
    unit = " pixels"
    tick_spacing = (None, None, None)

##################################
# load the reconstructed crystal #
##################################
plt.ion()
root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename(
    initialdir=datadir,
    title="Select the reconstruction file",
    filetypes=[("NPZ", "*.npz")],
)
npzfile = np.load(file_path)
phase = npzfile["displacement"]
if flip_phase:
    phase = -1 * phase
amp = npzfile["amp"]
if amp.ndim != 3:
    raise ValueError("3D arrays are expected")

gu.multislices_plot(
    array=amp,
    sum_frames=False,
    scale="linear",
    plot_colorbar=True,
    reciprocal_space=False,
    is_orthogonal=True,
    title="Modulus",
)

################################
# optionally load the q values #
################################
if load_qvalues:
    file_path = filedialog.askopenfilename(
        initialdir=datadir, title="Select the q values", filetypes=[("NPZ", "*.npz")]
    )
    q_values = np.load(file_path)
    qx = q_values["qx"]
    qz = q_values["qz"]
    qy = q_values["qy"]
    qvalues_shape = (*qx.shape, *qz.shape, *qy.shape)

    q_range = (qx.min(), qx.max(), qz.min(), qz.max(), qy.min(), qy.max())
else:
    q_range = (0, padding_shape[0], 0, padding_shape[1], 0, padding_shape[2])
    qvalues_shape = padding_shape

print("\nq range:", [f"{val:.4f}" for val in q_range])

################################################
# resample the object to match the extent in q #
################################################
obj = amp * np.exp(1j * phase)
if load_qvalues:
    new_voxelsizes = [
        2 * np.pi / (10 * q_range[2 * idx + 1] - 10 * q_range[2 * idx])
        for idx in range(3)
    ]
    print(
        f"\nRegridding with the new voxel sizes = "
        f"({new_voxelsizes[0]:.2f} nm, {new_voxelsizes[1]:.2f} nm, "
        f"{new_voxelsizes[2]:.2f} nm)"
    )
    obj = pu.regrid(array=obj, old_voxelsize=voxel_sizes, new_voxelsize=new_voxelsizes)

#######################################
# pad the object to the desired shape #
#######################################
if qvalues_shape != padding_shape:
    print(
        f"\nThe shape defined by q_values {qvalues_shape} "
        f"is different from padding_shape {padding_shape}"
    )
    print(f"Overriding padding_shape with {qvalues_shape}")
    padding_shape = qvalues_shape
obj = util.crop_pad(array=obj, output_shape=padding_shape, debugging=debug)

#####################################
# calculate the diffraction pattern #
#####################################
data = fftshift(fftn(obj)) / np.sqrt(
    reduce(lambda x, y: x * y, padding_shape)
)  # complex diffraction amplitude
data = abs(np.multiply(data, np.conjugate(data)))  # diffraction intensity

zcom, ycom, xcom = com = tuple(map(lambda x: int(np.rint(x)), center_of_mass(data)))
print("Center of mass of the diffraction pattern at pixel:", zcom, ycom, xcom)
print(
    f"\nintensity in a ROI of 7x7x7 voxels centered on the COM:"
    f" {int(data[zcom-3:zcom+4, ycom-3:ycom+4, xcom-3:xcom+4].sum())}"
)

if peak_value is not None:
    data = (
        data
        / data[zcom - 3 : zcom + 4, ycom - 3 : ycom + 4, xcom - 3 : xcom + 4].sum()
        * peak_value
    )
    print(
        f"Normalizing the data to peak_value, new intensity in the 7x7x7 ROI = "
        f"{int(data[zcom-3:zcom+4, ycom-3:ycom+4, xcom-3:xcom+4].sum())}"
    )
    # correction due to the loss of the normalization with mode decomposition

################
# contour plot #
################
if colorbar_range is None:  # use rounded acceptable values
    colorbar_range = (
        np.ceil(np.median(np.log10(data[np.logical_and(data != 0, ~np.isnan(data))]))),
        np.ceil(np.log10(data[np.logical_and(data != 0, ~np.isnan(data))].max())),
    )
if load_qvalues:
    fig, _, _ = gu.contour_slices(
        data,
        (qx, qz, qy),
        sum_frames=save_sum,
        title="Diffraction pattern",
        levels=np.linspace(colorbar_range[0], colorbar_range[1], 150, endpoint=True),
        plot_colorbar=True,
        scale="log",
        is_orthogonal=True,
        reciprocal_space=True,
    )
else:
    fig, _, _ = gu.multislices_plot(
        data,
        sum_frames=save_sum,
        scale="log",
        plot_colorbar=True,
        vmin=colorbar_range[0],
        vmax=colorbar_range[1],
        title="Diffraction pattern",
        is_orthogonal=True,
        reciprocal_space=True,
    )

#############################################################
# define the positions of the axes ticks and colorbar ticks #
#############################################################
# use 5 ticks by default if tick_spacing is None for the axis
tick_spacing = (
    (tick_spacing[0] or (q_range[1] - q_range[0]) / num_ticks),
    (tick_spacing[1] or (q_range[3] - q_range[2]) / num_ticks),
    (tick_spacing[2] or (q_range[5] - q_range[4]) / num_ticks),
)

print("\nTick spacing:", [f"{val:.3f} {unit}" for val in tick_spacing])

numticks_colorbar = int(np.floor(colorbar_range[1] - colorbar_range[0] + 1))

############################
# plot views in QyQz plane #
############################
if save_qyqz:
    fig, ax0 = plt.subplots(1, 1, figsize=(9, 6))
    if save_sum:
        # extent (left, right, bottom, top)
        plt0 = ax0.imshow(
            np.log10(data.sum(axis=0)),
            cmap=my_cmap,
            vmin=colorbar_range[0],
            vmax=colorbar_range[1],
            extent=[q_range[4], q_range[5], q_range[3], q_range[2]],
        )
    else:
        plt0 = ax0.imshow(
            np.log10(data[padding_shape[0] // 2, :, :]),
            cmap=my_cmap,
            vmin=colorbar_range[0],
            vmax=colorbar_range[1],
            extent=[q_range[4], q_range[5], q_range[3], q_range[2]],
        )
    ax0.invert_yaxis()  # qz is pointing up
    ax0.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing[2]))
    ax0.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing[1]))
    gu.colorbar(plt0, numticks=numticks_colorbar, pad=cbar_pad)
    gu.savefig(
        savedir=savedir,
        figure=fig,
        axes=ax0,
        tick_width=tick_width,
        tick_length=tick_length,
        tick_direction=tick_direction,
        label_size=16,
        xlabels=labels[2],
        ylabels=labels[1],
        filename=sample_name + str(scan) + comment + "_fromrec_qyqz",
        labelbottom=draw_ticks,
        labelleft=draw_ticks,
        labelright=False,
        labeltop=False,
        left=draw_ticks,
        right=draw_ticks,
        bottom=draw_ticks,
        top=draw_ticks,
    )

############################
# plot views in QyQx plane #
############################
if save_qyqx:
    fig, ax0 = plt.subplots(1, 1, figsize=(9, 6))
    if save_sum:
        # extent (left, right, bottom, top)
        plt0 = ax0.imshow(
            np.log10(data.sum(axis=1)),
            cmap=my_cmap,
            vmin=colorbar_range[0],
            vmax=colorbar_range[1],
            extent=[q_range[4], q_range[5], q_range[1], q_range[0]],
        )
    else:
        plt0 = ax0.imshow(
            np.log10(data[:, padding_shape[1] // 2, :]),
            cmap=my_cmap,
            vmin=colorbar_range[0],
            vmax=colorbar_range[1],
            extent=[q_range[4], q_range[5], q_range[1], q_range[0]],
        )
    ax0.invert_yaxis()  # qx is pointing up
    ax0.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing[2]))
    ax0.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing[0]))
    gu.colorbar(plt0, numticks=numticks_colorbar, pad=cbar_pad)
    gu.savefig(
        savedir=savedir,
        figure=fig,
        axes=ax0,
        tick_width=tick_width,
        tick_length=tick_length,
        tick_direction=tick_direction,
        label_size=16,
        xlabels=labels[2],
        ylabels=labels[0],
        filename=sample_name + str(scan) + comment + "_fromrec_qyqx",
        labelbottom=draw_ticks,
        labelleft=draw_ticks,
        labelright=False,
        labeltop=False,
        left=draw_ticks,
        right=draw_ticks,
        bottom=draw_ticks,
        top=draw_ticks,
    )

############################
# plot views in QzQx plane #
############################
if save_qzqx:
    fig, ax0 = plt.subplots(1, 1, figsize=(9, 6))
    if save_sum:
        # extent (left, right, bottom, top)
        plt0 = ax0.imshow(
            np.log10(data.sum(axis=2)),
            cmap=my_cmap,
            vmin=colorbar_range[0],
            vmax=colorbar_range[1],
            extent=[q_range[2], q_range[3], q_range[1], q_range[0]],
        )
    else:
        plt0 = ax0.imshow(
            np.log10(data[:, :, padding_shape[2] // 2]),
            cmap=my_cmap,
            vmin=colorbar_range[0],
            vmax=colorbar_range[1],
            extent=[q_range[2], q_range[3], q_range[1], q_range[0]],
        )
    # qx is pointing down (the image will be rotated manually by 90 degrees)
    ax0.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing[1]))
    ax0.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing[0]))
    gu.colorbar(plt0, numticks=numticks_colorbar, pad=cbar_pad)
    gu.savefig(
        savedir=savedir,
        figure=fig,
        axes=ax0,
        tick_width=tick_width,
        tick_length=tick_length,
        tick_direction=tick_direction,
        label_size=16,
        xlabels=labels[1],
        ylabels=labels[0],
        filename=sample_name + str(scan) + comment + "_fromrec_qzqx",
        labelbottom=draw_ticks,
        labelleft=draw_ticks,
        labelright=False,
        labeltop=False,
        left=draw_ticks,
        right=draw_ticks,
        bottom=draw_ticks,
        top=draw_ticks,
    )

plt.ioff()
plt.show()
