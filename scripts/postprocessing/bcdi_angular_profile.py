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

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage.measurements import center_of_mass

import bcdi.graph.graph_utils as gu
import bcdi.postprocessing.facet_recognition as fu
import bcdi.utils.format as fmt
import bcdi.utils.utilities as util
import bcdi.utils.validation as valid

helptext = """
This script allow to plot the width of a 2D object in function of the angle and a
modulus threshold defining the object from the background. Must be given as input:
the voxel size (possibly different in all directions), the angular step size and an
origin point where all linecuts pass by.
"""

datadir = (
    "D:/data/P10_2nd_test_isosurface_Dec2020/data_nanolab/dataset_1_no_psf/result/"
)
# "D:/data/P10_2nd_test_isosurface_Dec2020/data_nanolab/dataset_1_newpsf/result/"
# "D:/data/P10_2nd_test_isosurface_Dec2020/data_nanolab/" \
# "AFM-SEM/P10 beamtime P2 particle size SEM/"
# "D:/data/P10_2nd_test_isosurface_Dec2020/data_nanolab/dataset_1_newpsf/" \
# "PtNP1_00128/result/"  # data folder  #
savedir = (
    datadir + "linecuts/refined0.25-0.55/test/"
)  # 'linecuts_P2_001a/valid_range/'  #
# "D:/data/P10_2nd_test_isosurface_Dec2020/data_nanolab/AFM-SEM/" \
# "P10 beamtime P2 particle size SEM/linecuts_P2_001a/"
# results will be saved here, if None it will default to datadir
upsampling_factor = (
    5  # integer, 1=no upsampling_factor, 2=voxel size divided by 2 etc...
)
threshold = np.linspace(
    0.25, 0.55, num=11
)  # [0.471, 0.5, 0.526]  # np.round(np.linspace(0.2, 0.5, num=10), decimals=3)
# number or list of numbers between 0 and 1,
# modulus threshold defining the normalized object from the background
angular_step = 1  # in degrees, the linecut directions will be automatically calculated
# in the orthonormal reference frame is given by the array axes.
# It will be corrected for anisotropic voxel sizes.
roi = None  # (470, 550, 710, 790)  # P2_001a.tif
# (470, 550, 710, 790)  # P2_001a.tif
# (220, 680, 620, 1120)  # P2_018.tif
# ROI centered around the crystal of interest in the 2D image
# the center of mass will be determined within this ROI when origin is not defined.
# Leave None to use the full array.
origin = None  # origin where all the line cuts pass by
# (indices considering the array cropped to roi).
# If None, it will use the center of mass of the modulus in the region defined by roi
voxel_size = 5
# 2.070393374741201 * 0.96829786  # P2_001a.tif
# 0.3448275862068966 * 0.96829786  # P2_018.tif
# positive real number or tuple of 2 or 3 positive real number
# (2 for 2D object, 3 for 3D) (in nm)
sum_axis = 1  # if the object is 3D, it will be summed along that axis
debug = False  # True to print the output dictionary and plot the legend
tick_length = 8  # in plots
tick_width = 2  # in plots
comment = ""  # string to add to the filename when saving
##################################
# end of user-defined parameters #
##################################

#############################
# define default parameters #
#############################
colors = ("b", "g", "r", "c", "m", "y", "k")  # for plots
markers = (".", "v", "^", "<", ">")  # for plots
validation_name = "angular_profile"
mpl.rcParams["axes.linewidth"] = tick_width  # set the linewidth globally

#########################
# check some parameters #
#########################
valid.valid_item(
    value=upsampling_factor, allowed_types=int, min_included=1, name=validation_name
)
valid.valid_container(comment, container_types=str, name=validation_name)
if comment.startswith("_"):
    comment = comment[1:]

##################################################
# create the list of directions for the linecuts #
##################################################
angles = np.arange(0, 180, angular_step)
nb_dir = len(angles)
directions = []
for idx in range(nb_dir):
    directions.append(
        (np.sin(angles[idx] * np.pi / 180), np.cos(angles[idx] * np.pi / 180))
    )

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

obj = abs(obj)
ndim = obj.ndim
if isinstance(voxel_size, Real):
    voxel_size = (voxel_size,) * ndim

print(f"Object shape = {obj.shape}, voxel size = {voxel_size}")
if upsampling_factor > 1:
    obj, voxel_size = fu.upsample(
        array=obj,
        upsampling_factor=upsampling_factor,
        voxelsizes=voxel_size,
        title="modulus",
        debugging=debug,
    )
    print(f"Upsampled object shape = {obj.shape}, upsampled voxel size = {voxel_size}")
else:
    valid.valid_container(
        voxel_size,
        container_types=(list, tuple, np.ndarray),
        length=ndim,
        item_types=Real,
        min_excluded=0,
        name="angular_profile",
    )

#########################
# check some parameters #
#########################
if ndim == 3:
    nbz, nby, nbx = obj.shape
elif ndim == 2:
    nby, nbx = obj.shape
else:
    raise ValueError(f"obj should be either 2D or 3D, ndim={ndim}")

if roi is None:
    roi = (0, nby, 0, nbx)
valid.valid_container(
    roi,
    container_types=(list, tuple, np.ndarray),
    length=4,
    item_types=int,
    min_included=0,
    name="angular_profile",
)
if not (roi[0] < roi[1] <= nby and roi[2] < roi[3] <= nbx):
    raise ValueError("roi incompatible with the array shape")

obj = obj[roi[0] : roi[1], roi[2] : roi[3]].astype(float)

if origin is None:
    if ndim == 3:
        piy, pix = center_of_mass(obj.sum(axis=sum_axis))
    else:
        piy, pix = center_of_mass(obj)
    origin = int(np.rint(piy)), int(np.rint(pix))
valid.valid_container(
    origin,
    container_types=(list, tuple),
    length=2,
    item_types=int,
    name="angular_profile",
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
    name="angular_profile",
)

comment = f"_origin_{origin}_{comment}"

#########################
# normalize the modulus #
#########################
obj = abs(obj) / abs(obj).max()  # normalize the modulus to 1
obj[np.isnan(obj)] = 0  # remove nans
fig, axs, _ = gu.imshow_plot(
    array=obj,
    sum_frames=True,
    sum_axis=1,
    plot_colorbar=True,
    reciprocal_space=False,
    vmin=0,
    vmax=np.nan,
    is_orthogonal=True,
)

gu.savefig(
    savedir=savedir,
    figure=fig,
    axes=axs,
    tick_width=tick_width,
    tick_length=tick_length,
    tick_labelsize=14,
    xlabels=axs.get_xlabel(),
    ylabels=axs.get_ylabel(),
    titles=axs.get_title(),
    label_size=16,
    filename=f"roi{roi}" + comment,
)

comment = comment + f"_{angular_step}deg"
result = {}

#########################################################
# 3D case (BCDI): loop over thresholds first            #
# (the threshold needs to be applied before projecting) #
#########################################################
if ndim == 3:
    # remove the voxel size along the projection axis
    voxel_size = list(voxel_size)
    voxel_size.pop(sum_axis)
    valid.valid_container(
        voxel_size,
        container_types=list,
        length=2,
        item_types=Real,
        min_excluded=0,
        name="angular_profile",
    )

    ang_width = np.empty((len(threshold), nb_dir))
    for idx, thres in enumerate(threshold):
        # apply the threshold
        tmp_obj = np.copy(obj)
        tmp_obj[tmp_obj < thres] = 0
        if ndim == 3:  # project the object
            tmp_obj = tmp_obj.sum(axis=sum_axis)
            tmp_obj = abs(tmp_obj) / abs(tmp_obj).max()  # normalize the modulus to 1

        for idy, direction in enumerate(directions):
            # get the distances and the modulus values along the linecut
            distance, cut = util.linecut(
                array=tmp_obj, point=origin, direction=direction, voxel_size=voxel_size
            )
            # get the indices where the linecut is non_zero
            indices = np.nonzero(cut)
            # get the width along the cut for that threshold
            ang_width[idx, idy] = distance[max(indices[0])] - distance[min(indices[0])]

    # store the result in the dictionary
    result["ang_width_threshold"] = ang_width

###################################################
# 2D case (SEM): one can create the linecut for   #
# each direction first and apply thresholds later #
###################################################
else:
    ##############################################################################
    # calculate the evolution of the width vs threshold for different directions #
    ##############################################################################
    for idx, direction in enumerate(directions):
        # get the distances and the modulus values along the linecut
        distance, cut = util.linecut(
            array=obj, point=origin, direction=direction, voxel_size=voxel_size
        )

        fit = interp1d(distance, cut)
        dist_interp = np.linspace(distance.min(), distance.max(), num=10000)
        cut_interp = fit(dist_interp)
        width = np.empty(len(threshold))

        # calculate the function width vs threshold
        for idy, thres in enumerate(threshold):
            # calculate the distances where the modulus is equal to threshold
            crossings = np.argwhere(cut_interp > thres)
            if len(crossings) > 1:
                width[idy] = dist_interp[crossings.max()] - dist_interp[crossings.min()]
            else:
                width[idy] = 0
        # store the result in a dictionary
        # (cuts can have different lengths depending on the direction)
        result[f"direction ({direction[0]:.4f},{direction[1]:.4f})"] = {
            "angle": angles[idx],
            "distance": distance,
            "cut": cut,
            "threshold": threshold,
            "width": width,
        }

    if debug:  # plot all line cuts
        fig = plt.figure(figsize=(12, 9))
        ax = plt.subplot(111)
        plot_nb = 0
        for key, value in result.items():
            # value is a dictionary
            # {'angle': angles[idx], 'distance': distance, 'cut': cut}
            (line,) = ax.plot(
                value["distance"],
                value["cut"],
                color=colors[plot_nb % len(colors)],
                marker=markers[(plot_nb // len(colors)) % len(markers)],
                fillstyle="none",
                markersize=6,
                linestyle="-",
                linewidth=1,
            )
            line.set_label(f"{key}")
            plot_nb += 1
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
            only_labels=True,
        )

    ##########################################################################
    # calculate the evolution of the width vs angle for different thresholds #
    ##########################################################################
    ang_width_threshold = np.empty((len(threshold), nb_dir))
    for idx, thres in enumerate(threshold):
        tmp_angles = np.empty(nb_dir)  # will be used to reorder the angles
        angular_width = np.empty(nb_dir)
        count = 0
        for key, value in result.items():  # iterating over the directions
            # value is a dictionary
            # {'angle': angles[idx], 'distance': distance, 'cut': cut}
            tmp_angles[count] = value["angle"]  # index related to the angle/direction
            if thres != value["threshold"][idx]:
                raise ValueError("ordering error in threshold")
            angular_width[count] = value["width"][idx]  # index related to the threshold
            count += 1
        if not np.all(np.isclose(tmp_angles, angles)):
            raise ValueError("ordering error in angles")
        ang_width_threshold[idx, :] = angular_width

    # update the dictionary
    result["ang_width_threshold"] = ang_width_threshold

#####################################################
#  plot the width vs angle for different thresholds #
#####################################################
fig = plt.figure(figsize=(12, 9))
ax = plt.subplot(111)
for idx, thres in enumerate(threshold):
    (line,) = ax.plot(
        angles,
        result["ang_width_threshold"][idx],
        color=colors[idx % len(colors)],
        marker=markers[(idx // len(colors)) % len(markers)],
        fillstyle="none",
        markersize=6,
        linestyle="-",
        linewidth=1,
    )
    line.set_label(f"threshold {thres}")

legend = False
if len(threshold) < 15:
    legend = True

gu.savefig(
    savedir=savedir,
    figure=fig,
    axes=ax,
    tick_width=tick_width,
    tick_length=tick_length,
    tick_labelsize=14,
    xlabels="angle (deg)",
    ylabels="width (nm)",
    label_size=20,
    legend=legend,
    legend_labelsize=14,
    filename="width_vs_ang" + comment,
    only_labels=True,
)

###################
# save the result #
###################
result["threshold"] = threshold
result["angles"] = angles
result["origin"] = origin
result["roi"] = roi

if debug:
    print("output dictionary:\n", json.dumps(result, cls=fmt.CustomEncoder, indent=4))

with open(savedir + "ang_width" + comment + ".json", "w", encoding="utf-8") as file:
    json.dump(result, file, cls=fmt.CustomEncoder, ensure_ascii=False, indent=4)

plt.ioff()
plt.show()
