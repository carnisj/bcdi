#!/usr/bin/env python3

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import os
import pathlib
import tkinter as tk
from numbers import Real
from pprint import pprint
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
from lmfit.models import RectangleModel

import bcdi.graph.graph_utils as gu
import bcdi.utils.utilities as util
import bcdi.utils.validation as valid

helptext = """
This script allow to plot and save linecuts through a 2D SEM image of a calibration
grid. Must be given as input: the voxel size, the direction of the cuts and a list of
points where to apply the cut along this direction. A rectangular profile is fitted
to the maxima.
"""

datadir = (
    "D:/data/P10_2nd_test_isosurface_Dec2020/data_nanolab/AFM-SEM/SEM calibration/"
)
# data folder
savedir = (
    "D:/data/P10_2nd_test_isosurface_Dec2020/data_nanolab/AFM-SEM/SEM calibration/test/"
)
# results will be saved here, if None it will default to datadir
direction = (0, 1)  # tuple of 2 numbers defining the direction of the cut
# in the orthonormal reference frame is given by the array axes.
# It will be corrected for anisotropic voxel sizes.
points = [
    (5, 0),
    (25, 0),
    (50, 0),
    (75, 0),
    (100, 0),
    (125, 0),
    (150, 0),
    (175, 0),
    (200, 0),
    (225, 0),
]
# points for MCS_03.tif,  MCS_06.tif and MCS_07.tif
# list/tuple of 2 indices corresponding to the points where
# the cut alond direction should be performed.
# The reference frame is given by the array axes.
# fit_roi = [[(360, 480), (1070, 1190)],  # ROIs for MCS_07.tif
#            [(475, 590), (1175, 1290)],
#            [(575, 685), (1275, 1390)],
#            [(675, 785), (1375, 1490)],
#            [(360, 480), (1070, 1190)],
#            [(475, 590), (1175, 1290)],
#            [(575, 685), (1275, 1390)],
#            [(675, 785), (1375, 1490)],
#            [(360, 480), (1070, 1190)],
#            [(475, 590), (1175, 1290)]]
# fit_roi = [[(200, 320), (1700, 1820)],  # ROIs for MCS_06.tif
#            [(300, 420), (1800, 1920)],
#            [(400, 520), (1900, 2015)],
#            [(500, 620), (2000, 2120)],
#            [(600, 720), (2100, 2220)],
#            [(700, 820), (2200, 2320)],
#            [(800, 920), (2300, 2420)],
#            [(900, 1020), (2400, 2520)],
#            [(1000, 1120), (2500, 2620)],
#            [(1100, 1220), (2600, 2720)]]
fit_roi = [
    [(350, 495), (5660, 5800)],
    [(350, 495), (5660, 5800)],
    [(350, 495), (5660, 5780)],
    [(350, 495), (5660, 5780)],  # ROIs for MCS_03.tif
    [(350, 495), (5650, 5790)],
    [(350, 495), (5650, 5790)],
    [(350, 495), (5650, 5790)],
    [(350, 485), (5640, 5780)],
    [(350, 485), (5640, 5780)],
    [(350, 485), (5640, 5780)],
]  # ROIs that should be fitted for each point. There should be as many
# sublists as the number of points. Leave None otherwise.
# background_roi = [0, 400, 465, 485]  # background_roi for MCS_07.tif
# background_roi = [0, 400, 150, 156]  # background_roi for MCS_06.tif
background_roi = [0, 400, 112, 118]  # background_roi for MCS_03.tif
# [ystart, ystop, xstart, xstop], the mean intensity in this ROI will be
# subtracted from the data. Leave None otherwise
# list of tuples [(start, stop), ...] of regions to be fitted,
# in the unit of length along the linecut, None otherwise
voxel_size = 4.140786749482402  # * 0.96829786
# positive real number, voxel size of the SEM image
nb_lines = 52
expected_width = 100.4 * (nb_lines - 1)  # 5120  # in nm, real positive number or None
debug = False  # True to print the output dictionary and plot the legend
comment = ""  # string to add to the filename when saving
tick_length = 10  # in plots
tick_width = 2  # in plots
##################################
# end of user-defined parameters #
##################################

#############################
# define default parameters #
#############################
colors = ("b", "g", "r", "c", "m", "y", "k")  # for plots
markers = (".", "v", "^", "<", ">")  # for plots
validation_name = "calibration_grid_SEM"

#################
# load the data #
#################
plt.ion()
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    initialdir=datadir,
    filetypes=[
        ("TIFF", "*.tif"),
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
if ndim != 2:
    raise ValueError(f"Number of dimensions = {ndim}, expected 2")

valid.valid_container(
    direction,
    container_types=(list, tuple, np.ndarray),
    length=ndim,
    item_types=Real,
    name=validation_name,
)

valid.valid_container(
    points, container_types=(list, tuple), min_length=1, name=validation_name
)
for point in points:
    valid.valid_container(
        point,
        container_types=(list, tuple, np.ndarray),
        length=ndim,
        item_types=Real,
        min_included=0,
        name=validation_name,
    )

if isinstance(voxel_size, Real):
    voxel_size = (voxel_size,) * ndim
valid.valid_container(
    voxel_size,
    container_types=(list, tuple, np.ndarray),
    length=ndim,
    item_types=Real,
    min_excluded=0,
    name=validation_name,
)

savedir = savedir or datadir
pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)

valid.valid_container(
    fit_roi, container_types=(list, tuple), allow_none=True, name=validation_name
)
if fit_roi is not None:
    if len(fit_roi) != len(points):
        raise ValueError(
            "There should be as many ROIs sublists as the number of points (None "
            "allowed)"
        )
    for sublist in fit_roi:
        valid.valid_container(
            sublist,
            container_types=(list, tuple),
            allow_none=True,
            name=validation_name,
        )
        if sublist is not None:
            for roi in sublist:
                valid.valid_container(
                    roi,
                    container_types=(list, tuple),
                    length=ndim,
                    item_types=Real,
                    min_included=0,
                    name=validation_name,
                )
valid.valid_container(
    background_roi,
    container_types=(list, tuple),
    allow_none=True,
    item_types=int,
    min_included=0,
    name=validation_name,
)

valid.valid_item(
    value=expected_width,
    allowed_types=Real,
    min_excluded=0,
    allow_none=True,
    name=validation_name,
)

valid.valid_container(comment, container_types=str, name=validation_name)
if comment.startswith("_"):
    comment = comment[1:]
comment = f"_direction{direction[0]}_{direction[1]}_{comment}"

#########################
# normalize the modulus #
#########################
obj = abs(obj) / abs(obj).max()  # normalize the modulus to 1
obj[np.isnan(obj)] = 0  # remove nans
if background_roi is not None:
    background = obj[
        background_roi[0] : background_roi[1] + 1,
        background_roi[2 : background_roi[3] + 1],
    ].mean()
    print(f"removing background = {background:.2f} from the data")
    obj = obj - background

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
    # store the result in a dictionary
    # (cuts can have different lengths depending on the direction)
    result[f"pixel {point}"] = {"distance": distance, "cut": cut}

######################
#  plot the linecuts #
######################
fig = plt.figure(figsize=(12, 9))
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

ax.tick_params(
    labelbottom=False,
    labelleft=False,
    direction="out",
    length=tick_length,
    width=tick_width,
)
ax.spines["right"].set_linewidth(tick_width)
ax.spines["left"].set_linewidth(tick_width)
ax.spines["top"].set_linewidth(tick_width)
ax.spines["bottom"].set_linewidth(tick_width)
fig.savefig(savedir + "cut" + comment + ".png")

ax.set_xlabel("width (nm)", fontsize=20)
ax.set_ylabel("modulus", fontsize=20)
if debug:
    ax.legend(fontsize=14)
ax.tick_params(
    labelbottom=True, labelleft=True, axis="both", which="major", labelsize=16
)
fig.savefig(savedir + "cut" + comment + "_labels.png")

###############################
# fit the peaks with gaussian #
###############################
if fit_roi is not None:
    width = np.empty(len(points))

    idx_point = 0
    for key, value in result.items():  # loop over linecuts
        # value is a dictionary {'distance': 1D array, 'cut': 1D array}
        tmp_str = f"{key}"
        print(f'\n{"#" * len(tmp_str)}\n' + tmp_str + "\n" + f'{"#" * len(tmp_str)}')
        for idx_roi, roi in enumerate(
            fit_roi[idx_point]
        ):  # loop over the ROIs, roi is a tuple of two number
            # define the fit initial center
            tmp_str = f"{roi}"
            indent = 2
            print(
                f'\n{" " * indent}{"-" * len(tmp_str)}\n'
                + f'{" " * indent}'
                + tmp_str
                + "\n"
                + f'{" " * indent}{"-" * len(tmp_str)}'
            )
            # find linecut indices falling into the roi
            ind_start, ind_stop = util.find_nearest(value["distance"], roi)

            # fit a RectangleModel from lmfit to the peaks
            midpoint = (roi[0] + roi[1]) / 2
            offset = (roi[1] - roi[0]) / 8
            # initialize fit parameters (guess does not perform well)
            rect_mod = RectangleModel(form="erf")
            rect_params = rect_mod.make_params()
            rect_params["amplitude"].set(0.75, min=0.5, max=1)
            rect_params["center1"].set(midpoint - offset, min=roi[0], max=roi[1])
            rect_params["sigma1"].set(1, min=0.001, max=10)
            rect_params["center2"].set(midpoint + offset, min=roi[0], max=roi[1])
            rect_params["sigma2"].set(1, min=0.001, max=10)
            # run the fit
            rect_result = rect_mod.fit(
                value["cut"][ind_start : ind_stop + 1],
                rect_params,
                x=value["distance"][ind_start : ind_stop + 1],
            )
            print("\n" + rect_result.fit_report())

            value[f"roi {roi}"] = rect_result

        # calculate the mean distance between the first and last peaks
        width[idx_point] = (
            value[f"roi {fit_roi[idx_point][-1]}"].params["midpoint"].value
            - value[f"roi {fit_roi[idx_point][0]}"].params["midpoint"].value
        )
        idx_point += 1

    # update the dictionnary
    print(f"\n widths: {width}")
    result["mean_width"] = np.mean(width)
    result["std_width"] = np.std(width)
    result["fitting_rois"] = fit_roi

    tmp_str = "mean width"
    print(f'\n{"#" * len(tmp_str)}\n' + tmp_str + "\n" + f'{"#" * len(tmp_str)}')
    print(
        f"mean width: {result['mean_width']:.2f}nm,"
        f" std width: {result['std_width']:.2f}"
    )
    print(f"mean pitch: {result['mean_width']/(nb_lines-1):.2f}nm")
    if expected_width is not None:
        correction_factor = expected_width / result["mean_width"]
        print(f"correction factor to apply to the voxel size: {correction_factor}")

    #####################################################################
    # plot an overlay of the first and last peaks for the first linecut #
    #####################################################################
    fig = plt.figure(figsize=(12, 9))
    # area around the first peak
    ax0 = plt.subplot(121)
    ind_start, ind_stop = util.find_nearest(
        result[f"pixel {points[0]}"]["distance"], fit_roi[0][0]
    )
    x_axis = result[f"pixel {points[0]}"]["distance"][ind_start : ind_stop + 1]
    (line0,) = ax0.plot(
        x_axis, result[f"pixel {points[0]}"]["cut"][ind_start : ind_stop + 1], "-or"
    )
    line0.set_label("linecut")
    fit_axis = np.linspace(x_axis.min(), x_axis.max(), num=200)
    result_first = result[f"pixel {points[0]}"][
        f"roi {fit_roi[0][0]}"
    ]  # results of the fit with rect. model
    fit_first = result_first.eval(x=fit_axis)
    (fit0,) = ax0.plot(fit_axis, fit_first, "b-")
    fit0.set_label("RectModel")
    ax0.set_ylim(-0.05, 0.9)

    ax1 = plt.subplot(122)
    ind_start, ind_stop = util.find_nearest(
        result[f"pixel {points[0]}"]["distance"], fit_roi[0][-1]
    )
    x_axis = result[f"pixel {points[0]}"]["distance"][ind_start : ind_stop + 1]
    (line1,) = ax1.plot(
        x_axis, result[f"pixel {points[0]}"]["cut"][ind_start : ind_stop + 1], "-or"
    )
    line1.set_label("linecut")
    fit_axis = np.linspace(x_axis.min(), x_axis.max(), num=200)
    result_last = result[f"pixel {points[0]}"][
        f"roi {fit_roi[0][-1]}"
    ]  # results of the fit with rect. model
    fit_last = result_last.eval(x=fit_axis)
    (fit1,) = ax1.plot(fit_axis, fit_last, "b-")
    fit1.set_label("RectModel")
    ax1.set_ylim(-0.05, 0.9)

    ax0.spines["right"].set_linewidth(tick_width)
    ax0.spines["left"].set_linewidth(tick_width)
    ax0.spines["top"].set_linewidth(tick_width)
    ax0.spines["bottom"].set_linewidth(tick_width)
    ax1.spines["right"].set_linewidth(tick_width)
    ax1.spines["left"].set_linewidth(tick_width)
    ax1.spines["top"].set_linewidth(tick_width)
    ax1.spines["bottom"].set_linewidth(tick_width)
    ax0.tick_params(
        labelbottom=False,
        labelleft=False,
        direction="out",
        length=tick_length,
        width=tick_width,
        labelsize=16,
    )
    ax1.tick_params(
        labelbottom=False,
        labelleft=False,
        direction="out",
        length=tick_length,
        width=tick_width,
        labelsize=16,
    )
    fig.savefig(savedir + "fit_roi" + comment + ".png")

    ax0.set_xlabel("width (nm)", fontsize=20)
    ax0.set_ylabel("modulus", fontsize=20)
    ax0.set_title("first roi", fontsize=20)
    ax1.set_xlabel("width (nm)", fontsize=20)
    ax1.set_ylabel("modulus", fontsize=20)
    ax1.set_title("last roi", fontsize=20)
    ax0.legend(fontsize=14)
    ax1.legend(fontsize=14)
    ax0.tick_params(
        labelbottom=True, labelleft=True, axis="both", which="major", labelsize=16
    )
    ax1.tick_params(
        labelbottom=True, labelleft=True, axis="both", which="major", labelsize=16
    )
    fig.savefig(savedir + "fit_roi" + comment + "_labels.png")

#############################
# print and save the result #
#############################
if debug:
    tmp_str = "output dictionnary"
    print(f'\n{"#" * len(tmp_str)}\n' + tmp_str + "\n" + f'{"#" * len(tmp_str)}')
    pprint(result, indent=2)

plt.ioff()
plt.show()
print("")
