#!/usr/bin/env python3

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import json
import os
import pathlib
import tkinter as tk
from numbers import Real
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

import bcdi.graph.graph_utils as gu
import bcdi.utils.format as fmt
import bcdi.utils.utilities as util
import bcdi.utils.validation as valid

helptext = """
This script allow to plot and save linecuts through a 2D or 3D object in function of a
modulus threshold defining the object from the background. Must be given as input:
the voxel size (possibly different in all directions),  the direction of the cuts and
a list of points where to apply the cut along this direction.
"""

datadir = (
    "D:/data/P10_2nd_test_isosurface_Dec2020/data_nanolab/dataset_1_newpsf/result/"
)
# data folder
savedir = datadir + "linecuts/test/"
# results will be saved here, if None it will default to datadir
threshold = np.linspace(0, 1.0, num=20)
# number or list of numbers between 0 and 1,
# modulus threshold defining the normalized object from the background
direction = (
    0,
    1,
    0,
)  # tuple of 2 or 3 numbers (2 for 2D object, 3 for 3D)
# defining the direction of the cut in the orthonormal reference frame is given by
# the array axes. It will be corrected for anisotropic voxel sizes.
points = {
    (25, 37, 23),
    (25, 37, 24),
    (25, 37, 25),
    (25, 37, 26),
    (26, 37, 23),
    (26, 37, 24),
    (26, 37, 25),
    (26, 37, 26),
    (27, 37, 24),
    (27, 37, 25),
}
# {(0, 5), (0, 25), (0, 50), (0, 75), (0, 100),
# (0, 125), (0, 150), (0, 175), (0, 200), (0, 225)}w

# list/tuple/set of 2 or 3 indices (2 for 2D object, 3 for 3D)
# corresponding to the points where the cut alond direction should be performed. The
# reference frame is given by the array axes.
voxel_size = 5  # 4.140786749482402  # positive real number  or
# tuple of 2 or 3 positive real number (2 for 2D object, 3 for 3D)
width_lines = (
    98,
    100,
    102,
)  # list of vertical lines that will appear in the plot width vs threshold,
# None otherwise
styles = {
    0: (0, (2, 6)),
    1: "dashed",
    2: (0, (2, 6)),
}  # line style for the width_lines, 1 for each line
debug = False  # True to print the output dictionary and plot the legend
comment = ""  # string to add to the filename when saving
tick_length = 10  # in plots
tick_width = 2  # in plots
fig_size = (
    7,
    9,
)  # figure size for the plot of linecuts and plot of width vs threshold.
# If None, default to (12, 9)
cuts_limits = [
    63,
    212,
    0,
    1,
]  # [xmin, xmax, ymin, ymax] list of axes limits for the plot of the linecuts.
# Leave None for the default range.
thres_limits = [
    0.1,
    0.7,
    85,
    120,
]  # [xmin, xmax, ymin, ymax] list of axes limits for the plot width vs threshold.
# Leave None for the default range.
zoom = None  # [xmin, xmax, ymin, ymax] list of axes limits
# for a zoom on a ROI of the plot. Leave None to skip.
##################################
# end of user-defined parameters #
##################################

###############################
# list of colors for the plot #
###############################
colors = ("b", "g", "r", "c", "m", "y", "k")
markers = (".", "v", "^", "<", ">")

#################
# load the data #
#################
plt.ion()
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    initialdir=datadir,
    filetypes=[
        ("NPZ", "*.npz"),
        ("NPY", "*.npy"),
        ("CXI", "*.cxi"),
        ("HDF5", "*.h5"),
        ("all files", "*.*"),
    ],
)

_, ext = os.path.splitext(file_path)
if ext in {".png", ".jpg", ".tif"}:
    obj = util.image_to_ndarray(filename=file_path, convert_grey=True)
else:
    obj, _ = util.load_file(file_path)
ndim = obj.ndim

#########################
# check some parameters #
#########################
valid_name = "bcdi_line_profile"
if ndim not in {2, 3}:
    raise ValueError(f"Number of dimensions = {ndim}, expected 2 or 3")

valid.valid_container(
    direction,
    container_types=(list, tuple, np.ndarray),
    length=ndim,
    item_types=Real,
    name=valid_name,
)

valid.valid_container(
    points, container_types=(list, tuple, set), min_length=1, name=valid_name
)
for point in points:
    valid.valid_container(
        point,
        container_types=(list, tuple, np.ndarray),
        length=ndim,
        item_types=Real,
        min_included=0,
        name=valid_name,
    )

if isinstance(voxel_size, Real):
    voxel_size = (voxel_size,) * ndim
valid.valid_container(
    voxel_size,
    container_types=(list, tuple, np.ndarray),
    length=ndim,
    item_types=Real,
    min_excluded=0,
    name=valid_name,
)

savedir = savedir or datadir
pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)

if isinstance(threshold, Real):
    threshold = (threshold,)
valid.valid_container(
    threshold,
    container_types=(list, tuple, np.ndarray),
    item_types=Real,
    min_included=0,
    max_included=1,
    name=valid_name,
)

if isinstance(width_lines, Real):
    width_lines = (width_lines,)
valid.valid_container(
    width_lines,
    container_types=(list, tuple, np.ndarray),
    item_types=Real,
    min_excluded=0,
    allow_none=True,
    name=valid_name,
)

if width_lines is not None:
    tmp_thres = np.zeros((len(width_lines), len(points)))
    if not isinstance(styles, dict):
        raise TypeError("styles should be a dictionnary")
    if len(styles) != len(width_lines):
        raise ValueError(
            "styles should have as many entries as the number of width_lines"
        )

if fig_size is None:
    fig_size = (12, 9)
valid.valid_container(
    fig_size, container_types=(list, tuple), item_types=Real, length=2, name=valid_name
)
valid.valid_container(
    cuts_limits,
    container_types=(list, tuple),
    item_types=Real,
    length=4,
    allow_none=True,
    name=valid_name,
)
valid.valid_container(
    thres_limits,
    container_types=(list, tuple),
    item_types=Real,
    length=4,
    allow_none=True,
    name=valid_name,
)
valid.valid_container(
    zoom,
    container_types=(list, tuple),
    item_types=Real,
    length=4,
    allow_none=True,
    name=valid_name,
)

if ndim == 3:
    comment = f"_direction{direction[0]}_{direction[1]}_{direction[2]}_{comment}"
else:
    comment = f"_direction{direction[0]}_{direction[1]}_{comment}"

#########################
# normalize the modulus #
#########################
obj = abs(obj) / abs(obj).max()  # normalize the modulus to 1
obj[np.isnan(obj)] = 0  # remove nans
if ndim == 2:
    gu.imshow_plot(
        array=obj, plot_colorbar=True, reciprocal_space=False, is_orthogonal=True
    )
else:
    gu.multislices_plot(
        array=obj,
        sum_frames=False,
        plot_colorbar=True,
        reciprocal_space=False,
        is_orthogonal=True,
        slice_position=(25, 37, 25),
    )

#####################################
# create the linecut for each point #
#####################################
result = {}
for point in points:
    # get the distances and the modulus values along the linecut
    distance, cut = util.linecut(
        array=obj, point=point, direction=direction, voxel_size=voxel_size
    )
    # store the result in a dictionary (cuts can have different lengths
    # depending on the direction)
    result[f"voxel {point}"] = {"distance": distance, "cut": cut}

######################
#  plot the linecuts #
######################
fig = plt.figure(figsize=fig_size)
ax = plt.subplot(111)
plot_nb = 0
for key, value in result.items():
    # value is a dictionary {'distance': 1D array, 'cut': 1D array}
    (line,) = ax.plot(
        value["distance"],
        value["cut"],
        color=colors[plot_nb % len(colors)],
        marker=markers[(plot_nb // len(colors)) % len(markers)],
        fillstyle="none",
        markersize=10,
        linestyle="-",
        linewidth=1,
    )
    line.set_label(f"cut through {key}")
    plot_nb += 1

if cuts_limits is not None:
    ax.set_xlim(left=cuts_limits[0], right=cuts_limits[1])
    ax.set_ylim(bottom=cuts_limits[2], top=cuts_limits[3])

legend = False
if plot_nb < 15:
    legend = True
gu.savefig(
    savedir=savedir,
    figure=fig,
    axes=ax,
    tick_width=tick_width,
    tick_length=tick_length,
    tick_labelsize=16,
    xlabels="width (nm)",
    ylabels="modulus",
    label_size=20,
    legend=legend,
    legend_labelsize=14,
    filename="cuts" + comment,
    only_labels=False,
)

#################################################################################
# calculate the evolution of the width of the object depending on the threshold #
#################################################################################
idx_point = 0
for key, value in result.items():  # loop over the linecuts
    fit = interp1d(value["distance"], value["cut"])
    dist_interp = np.linspace(
        value["distance"].min(), value["distance"].max(), num=10000
    )
    cut_interp = fit(dist_interp)
    width = np.empty(len(threshold))

    # calculate the function width vs threshold
    for idx, thres in enumerate(threshold):
        # calculate the distances where the modulus is equal to threshold
        crossings = np.argwhere(cut_interp > thres)
        if len(crossings) > 1:
            width[idx] = dist_interp[crossings.max()] - dist_interp[crossings.min()]
        else:
            width[idx] = 0
    # update the dictionary value
    value["threshold"] = threshold
    value["width"] = width

    if width_lines is not None:
        # fit the function width vs threshold and estimate
        # where it crosses the expected widths
        fit = interp1d(
            width, threshold
        )  # width vs threshold is monotonic (decreasing with increasing threshold)
        for idx_line, val in enumerate(width_lines):
            tmp_thres[idx_line, idx_point] = fit(val)
        idx_point += 1

#################################################
# calculate statistics on the fitted thresholds #
#################################################
if width_lines is not None:
    mean_thres = np.mean(tmp_thres, axis=1)
    std_thres = np.std(tmp_thres, axis=1)
    # update the dictionary
    result["fitted_thresholds"] = tmp_thres
    result["expected_width"] = width_lines
    result["mean_thres"] = np.round(mean_thres, decimals=3)
    result["std_thres"] = np.round(std_thres, decimals=3)

#################################
#  plot the widths vs threshold #
#################################
result["direction"] = direction

fig = plt.figure(figsize=fig_size)
ax = plt.subplot(111)
plot_nb = 0
for key, value in result.items():
    if isinstance(value, dict):  # iterating over points, value is a dictionary
        (line,) = ax.plot(
            value["threshold"],
            value["width"],
            color=colors[plot_nb % len(colors)],
            marker=markers[(plot_nb // len(colors)) % len(markers)],
            fillstyle="none",
            markersize=10,
            linestyle="-",
            linewidth=1,
        )
        line.set_label(f"cut through {key}")
        plot_nb += 1

if width_lines is not None:
    for index, hline in enumerate(width_lines):
        ax.axhline(y=hline, linestyle=styles[index], color="k", linewidth=1.5)

xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()

if zoom is not None:
    ax.set_xlim(left=zoom[0], right=zoom[1])
    ax.set_ylim(bottom=zoom[2], top=zoom[3])
    gu.savefig(
        savedir=savedir,
        figure=fig,
        axes=ax,
        tick_width=tick_width,
        tick_length=tick_length,
        tick_labelsize=16,
        xlabels="threshold",
        ylabels="width (nm)",
        titles=f"Width vs threshold in the direction {result['direction']}\n",
        title_size=20,
        label_size=20,
        legend_labelsize=14,
        filename="width_vs_threshold" + comment + "_zoom",
    )

if thres_limits is not None:
    ax.set_xlim(left=thres_limits[0], right=thres_limits[1])
    ax.set_ylim(bottom=thres_limits[2], top=thres_limits[3])
else:
    ax.set_xlim(left=xmin, right=xmax)
    ax.set_ylim(bottom=ymin, top=ymax)

legend = False
if debug and plot_nb < 15:
    legend = True
if width_lines is not None:
    text = {
        0: {
            "x": 0.3,
            "y": 0.80,
            "s": f"expected widths: {result['expected_width']}",
            "fontsize": 14,
        },
        1: {
            "x": 0.3,
            "y": 0.75,
            "s": f"fitted thresholds: {result['mean_thres']}",
            "fontsize": 14,
        },
        2: {"x": 0.3, "y": 0.70, "s": f"stds: {result['std_thres']}", "fontsize": 14},
    }
else:
    text = ""
gu.savefig(
    savedir=savedir,
    figure=fig,
    axes=ax,
    tick_width=tick_width,
    tick_length=tick_length,
    tick_labelsize=16,
    xlabels="threshold",
    ylabels="width (nm)",
    legend=legend,
    text=text,
    titles=f"Width vs threshold in the direction {result['direction']}\n",
    title_size=20,
    label_size=20,
    legend_labelsize=14,
    filename="width_vs_threshold" + comment,
)

###################
# save the result #
###################
if debug:
    print("output dictionary:\n", json.dumps(result, cls=fmt.CustomEncoder, indent=4))

with open(savedir + "cut" + comment + ".json", "w", encoding="utf-8") as file:
    json.dump(result, file, cls=fmt.CustomEncoder, ensure_ascii=False, indent=4)

plt.ioff()
plt.show()
