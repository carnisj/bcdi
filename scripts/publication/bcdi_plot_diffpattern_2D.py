#!/usr/bin/env python3

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import pathlib
import tkinter as tk
from numbers import Real
from tkinter import filedialog

import matplotlib as mpl
import matplotlib.ticker as ticker
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.measurements import center_of_mass

import bcdi.graph.graph_utils as gu
import bcdi.utils.utilities as util
import bcdi.utils.validation as valid
from bcdi.graph.colormap import ColormapFactory

helptext = """
Template for figures of the following article:
Carnis et al. Scientific Reports 9, 17357 (2019)
https://doi.org/10.1038/s41598-019-53774-2
For simulated data or experimental data, open a npz file (3D diffraction pattern) and
save individual figures. q values can be provided optionally.
For everything else than q values, the convention is the CXI convention:
(z downstream, y vertical up, x outboard).
For q values, the convention is (qx downstream, qz vertical up, qy outboard).
"""

scan = 292  # spec scan number
root_folder = "C:/Users/Jerome/Documents/data/P10_Longfei_Nov2020/data/"
sample_name = "B10_syn_S1"
datadir = root_folder + sample_name + f"_{scan:05d}" + "/pynx/"
photon_threshold = 0  # everything < this value will be set to 0
load_qvalues = True  # True to load the q values.
# It expects a single npz file with fieldnames 'qx', 'qy' and 'qz'
is_orthogonal = (
    True  # True if the data is in the qx qy qz orthogonal frame. Used for plot labels
)
##############################
# settings related to saving #
##############################
savedir = (
    datadir + "diffpattern/"
)  # results will be saved here, if None it will default to datadir
save_qyqz = True  # True to save the strain in QyQz plane
save_qyqx = True  # True to save the strain in QyQx plane
save_qzqx = True  # True to save the strain in QzQx plane
save_sum = True  # True to save the summed diffraction pattern in the detector,
# False to save the central slice only
comment = ""  # should start with _
##########################
# settings for the plots #
##########################
plot_symmetrical = False  # if False, will not use the parameter half_range
half_range = (
    None,
    None,
    None,
)  # tuple of three pixel numbers, half-range in each direction. Use None to use the
# maximum symmetrical data range along one direction e.g. [20, None, None]
colorbar_range = (
    0,
    6,
)  # [vmin, vmax] log scale in photon counts, leave None for default.
grey_background = False  # True to set nans to grey in the plots
tick_direction = "out"  # 'out', 'in', 'inout'
tick_length = 10  # in plots
tick_width = 2  # in plots
tick_spacing = (
    0.025,
    0.025,
    0.025,
)  # tuple of three numbers, in 1/A. Leave None for default.
num_ticks = 5  # number of ticks to use in axes when tick_spacing is not defined
##################################
# end of user-defined parameters #
##################################

####################
# Check parameters #
####################
valid_name = "bcdi_plot_diffpattern_2D"
valid.valid_item(save_sum, allowed_types=bool, name=valid_name)
if save_sum:
    comment = comment + "_sum"
valid.valid_container(
    colorbar_range,
    container_types=(tuple, list, np.ndarray),
    item_types=Real,
    length=2,
    allow_none=True,
    name=valid_name,
)
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
    (
        load_qvalues,
        save_qyqz,
        save_qyqx,
        save_qzqx,
        save_sum,
        is_orthogonal,
        grey_background,
    ),
    container_types=tuple,
    item_types=bool,
    name=valid_name,
)

savedir = savedir or datadir
if not savedir.endswith("/"):
    savedir += "/"
pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)

#############################
# define default parameters #
#############################
mpl.rcParams["axes.linewidth"] = tick_width  # set the linewidth globally

if grey_background:
    bad_color = "0.7"
else:
    bad_color = "1.0"  # white background

my_cmap = ColormapFactory(bad_color=bad_color).cmap

if is_orthogonal:
    labels = ("Qx", "Qz", "Qy")
else:
    labels = ("rocking angle", "detector Y", "detector X")

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

#############
# load data #
#############
plt.ion()
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    initialdir=datadir,
    title="Select the diffraction pattern",
    filetypes=[("NPZ", "*.npz")],
)
data, _ = util.load_file(file_path)
print("Initial data shape:", data.shape)
print("Data type", data.dtype)
data[data < photon_threshold] = 0

############################
# Check the plotting range #
############################
try:
    if len(half_range) != 3:
        raise ValueError("half-range should be a tuple of three pixel numbers")
except TypeError:
    raise TypeError("half-range should be a tuple of three pixel numbers")

nbz, nby, nbx = data.shape
zcom, ycom, xcom = com = tuple(map(lambda x: int(np.rint(x)), center_of_mass(data)))
print("Center of mass of the diffraction pattern at pixel:", zcom, ycom, xcom)
print(
    f"\nintensity in a ROI of 7x7x7 voxels centered on the COM:"
    f" {int(data[zcom-3:zcom+4, ycom-3:ycom+4, xcom-3:xcom+4].sum())}"
)
plot_range = []
if plot_symmetrical:
    max_range = (
        min(zcom, nbz - zcom),
        min(zcom, nbz - zcom),
        min(ycom, nby - ycom),
        min(ycom, nby - ycom),
        min(xcom, nbx - xcom),
        min(xcom, nbx - xcom),
    )  # maximum symmetric half ranges
else:
    max_range = (
        zcom,
        nbz - zcom,
        ycom,
        nby - ycom,
        xcom,
        nbx - xcom,
    )  # asymmetric half ranges

for idx, val in enumerate(half_range):
    plot_range.append(min(val or max_range[2 * idx], max_range[2 * idx]))
    plot_range.append(min(val or max_range[2 * idx + 1], max_range[2 * idx + 1]))
print("\nPlotting symmetrical ranges:", plot_symmetrical)
print("Plotting range from the center of mass:", plot_range)

gu.multislices_plot(
    array=data,
    sum_frames=True,
    scale="log",
    cmap=my_cmap,
    reciprocal_space=True,
    is_orthogonal=is_orthogonal,
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
    print("Loaded: qx shape:", qx.shape, "qz shape:", qz.shape, "qy shape:", qy.shape)
    if (
        not (
            *qx.shape,
            *qz.shape,
            *qy.shape,
        )
        == data.shape
    ):
        raise ValueError("q values and data shape are incompatible")

    # crop the q values to the region of interest used in plots
    qx = util.crop_pad_1d(
        array=qx, output_length=plot_range[0] + plot_range[1], crop_center=zcom
    )
    qz = util.crop_pad_1d(
        array=qz, output_length=plot_range[2] + plot_range[3], crop_center=ycom
    )
    qy = util.crop_pad_1d(
        array=qy, output_length=plot_range[4] + plot_range[5], crop_center=xcom
    )
    print("Cropped: qx shape:", qx.shape, "qz shape:", qz.shape, "qy shape:", qy.shape)

    q_range = (qx.min(), qx.max(), qz.min(), qz.max(), qy.min(), qy.max())
else:
    # crop the q values to the region of interest used in plots
    q_range = (
        0,
        plot_range[0] + plot_range[1],
        0,
        plot_range[2] + plot_range[3],
        0,
        plot_range[4] + plot_range[5],
    )

print("q range:", [f"{val:.4f}" for val in q_range])

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

if colorbar_range is None:  # use rounded acceptable values
    colorbar_range = (
        np.ceil(np.median(np.log10(data[np.logical_and(data != 0, ~np.isnan(data))]))),
        np.ceil(np.log10(data[np.logical_and(data != 0, ~np.isnan(data))].max())),
    )
numticks_colorbar = int(np.floor(colorbar_range[1] - colorbar_range[0] + 1))

############################
# plot views in QyQz plane #
############################
if save_qyqz:
    fig, ax0 = plt.subplots(1, 1, figsize=(9, 6))
    if save_sum:
        # extent (left, right, bottom, top)
        plt0 = ax0.imshow(
            np.log10(
                data[
                    :,
                    ycom - plot_range[2] : ycom + plot_range[3],
                    xcom - plot_range[4] : xcom + plot_range[5],
                ].sum(axis=0)
            ),
            cmap=my_cmap,
            vmin=colorbar_range[0],
            vmax=colorbar_range[1],
            extent=[q_range[4], q_range[5], q_range[3], q_range[2]],
        )
    else:
        plt0 = ax0.imshow(
            np.log10(
                data[
                    zcom,
                    ycom - plot_range[2] : ycom + plot_range[3],
                    xcom - plot_range[4] : xcom + plot_range[5],
                ]
            ),
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
        filename=sample_name + str(scan) + comment + "_qyqz",
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
            np.log10(
                data[
                    zcom - plot_range[0] : zcom + plot_range[1],
                    :,
                    xcom - plot_range[4] : xcom + plot_range[5],
                ].sum(axis=1)
            ),
            cmap=my_cmap,
            vmin=colorbar_range[0],
            vmax=colorbar_range[1],
            extent=[q_range[4], q_range[5], q_range[1], q_range[0]],
        )
    else:
        plt0 = ax0.imshow(
            np.log10(
                data[
                    zcom - plot_range[0] : zcom + plot_range[1],
                    ycom,
                    xcom - plot_range[4] : xcom + plot_range[5],
                ]
            ),
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
        filename=sample_name + str(scan) + comment + "_qyqx",
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
            np.log10(
                data[
                    zcom - plot_range[0] : zcom + plot_range[1],
                    ycom - plot_range[2] : ycom + plot_range[3],
                    :,
                ].sum(axis=2)
            ),
            cmap=my_cmap,
            vmin=colorbar_range[0],
            vmax=colorbar_range[1],
            extent=[q_range[2], q_range[3], q_range[1], q_range[0]],
        )
    else:
        plt0 = ax0.imshow(
            np.log10(
                data[
                    zcom - plot_range[0] : zcom + plot_range[1],
                    ycom - plot_range[2] : ycom + plot_range[3],
                    xcom,
                ]
            ),
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
        filename=sample_name + str(scan) + comment + "_qzqx",
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
