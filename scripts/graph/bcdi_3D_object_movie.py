#!/usr/bin/env python3

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import sys
import tkinter as tk
from tkinter import filedialog

import matplotlib
import matplotlib.animation as manimation
import numpy as np
from matplotlib import pyplot as plt

import bcdi.utils.utilities as util
from bcdi.graph.colormap import ColormapFactory

helptext = """
Create a movie from a 3D real space reconstruction in each direction. Requires
imagemagick (https://imagemagick.org) or ffmpeg (http://ffmpeg.zeranoe.com/builds/).
"""

scan = 22
root_folder = "D:/data/P10_August2019_CDI/data/"  # location of the .spec or log file
sample_name = "gold_2_2_2_000"  # "SN"  #
datadir = (
    root_folder + sample_name + str(scan) + "/pynx/1000_1000_1000_1_1_1/current_paper/"
)
comment = ""  # should start with _
movie_z = False  # save movie along z axis (downstream)
movie_y = True  # save movie along y axis (vertical up)
movie_x = False  # save movie along x axis (outboard)
frame_spacing = 1  # spacing between consecutive slices in voxel
frame_per_second = 2  # number of frames per second, 5 is a good default
vmin_vmax = [0, 1]  # scale for plotting the data
roi = []  # ROI to be plotted, leave it as [] to use all the reconstruction
# [zstart, ztop, ystart, ystop, xstart, xstop]
field_name = ""  # name or ''
# load the field name in a .npz file, if '' load the complex object
# and plot the normalized modulus
align_axes = ((0.2, 1, 0.02), (1, 0, -0.1))
# sequence of vectors of 3 elements each in the order xyz,
# e.g. ((x1,y1,z1), ...). None otherwise.
ref_axes = ((0, 1, 0), (1, 0, 0))
# sequence of reference vectors, same length as align_axes.
# Each vector in align_axes will be aligned to the
# corresponding reference axis of ref_axes
threshold = 0.05  # threshold apply on the object, if np.nan nothing happens
output_format = "gif"  # 'gif', 'mp4'
##################################
# end of user-defined parameters #
##################################

###################
# define colormap #
###################
my_cmap = ColormapFactory().cmap

###############
# load FFMpeg #
###############
if output_format == "gif":
    plt.rcParams["animation.convert_path"] = "D:/Python/imagemagick/magick.exe"
    FFMpegWriter = None
else:
    try:
        FFMpegWriter = manimation.writers["ffmpeg"]
    except KeyError:
        print("KeyError: 'ffmpeg'")
        sys.exit()
    except RuntimeError:
        print("Could not import FFMpeg writer for movie generation")
        sys.exit()

#############################
# load reconstructed object #
#############################
plt.ion()
root = tk.Tk()
root.withdraw()

if len(field_name) == 0:
    file_path = filedialog.askopenfilename(
        initialdir=datadir,
        title="Select the reconstructed object",
        filetypes=[
            ("NPZ", "*.npz"),
            ("NPY", "*.npy"),
            ("CXI", "*.cxi"),
            ("HDF5", "*.h5"),
        ],
    )
    obj, extension = util.load_file(file_path)
    obj = abs(obj)
    obj = obj / obj.max()
    if extension == ".h5":
        comment = comment + "_mode"
else:
    file_path = filedialog.askopenfilename(
        initialdir=datadir,
        title="Select the reconstructed object",
        filetypes=[("NPZ", "*.npz")],
    )
    obj = np.load(file_path)[field_name]
nbz, nby, nbx = obj.shape

#################
# rotate object #
#################
if align_axes:
    if len(align_axes) != len(ref_axes):
        raise ValueError("align_axes and ref_axes should have the same length")
    new_shape = [int(1.2 * nbz), int(1.2 * nby), int(1.2 * nbx)]
    obj = util.crop_pad(array=obj, output_shape=new_shape, debugging=False)
    nbz, nby, nbx = obj.shape
    print("Rotating object to have the crystallographic axes along array axes")
    for axis, ref_axis in zip(align_axes, ref_axes):
        print("axis to align, reference axis:", axis, ref_axis)
        obj = util.rotate_crystal(
            arrays=obj, axis_to_align=axis, reference_axis=ref_axis, debugging=True
        )  # out of plane alignement

###################
# apply threshold #
###################
if not np.isnan(threshold):
    obj[obj < threshold] = 0

#############
# check ROI #
#############
if len(roi) == 6:
    print("Crop/pad the reconstruction to accommodate the ROI")
    obj = util.crop_pad(
        array=obj, output_shape=[roi[1] - roi[0], roi[3] - roi[2], roi[5] - roi[4]]
    )

#################
# movie along z #
#################
if movie_z:
    metadata = dict(title="S" + str(scan) + comment)
    if output_format == "gif":
        writer = matplotlib.animation.ImageMagickFileWriter(
            fps=frame_per_second, metadata=metadata
        )
    else:
        writer = FFMpegWriter(fps=frame_per_second, metadata=metadata)
    fontsize = 10

    fig = plt.figure()
    with writer.saving(
        fig, datadir + "S" + str(scan) + "_z_movie." + output_format, dpi=100
    ):
        for index in range(nbz // frame_spacing):
            img = obj[index * frame_spacing, :, :]
            plt.clf()
            plt.imshow(img, vmin=vmin_vmax[0], vmax=vmin_vmax[1], cmap=my_cmap)
            ax = plt.gca()
            ax.invert_yaxis()
            ax.set_title("slice # %3d" % index, fontsize=fontsize)
            writer.grab_frame()

#################
# movie along y #
#################
if movie_y:
    metadata = dict(title="S" + str(scan) + comment)
    if output_format == "gif":
        writer = matplotlib.animation.ImageMagickFileWriter(
            fps=frame_per_second, metadata=metadata
        )
    else:
        writer = FFMpegWriter(fps=frame_per_second, metadata=metadata)
    fontsize = 10

    fig = plt.figure()
    with writer.saving(
        fig, datadir + "S" + str(scan) + "_y_movie." + output_format, dpi=100
    ):
        for index in range(nby // frame_spacing):
            img = obj[:, index * frame_spacing, :]
            plt.clf()
            plt.imshow(img, vmin=vmin_vmax[0], vmax=vmin_vmax[1], cmap=my_cmap)
            ax = plt.gca()
            ax.invert_yaxis()
            ax.set_title("slice # %3d" % index, fontsize=fontsize)
            writer.grab_frame()

#################
# movie along x #
#################
if movie_x:
    metadata = dict(title="S" + str(scan) + comment)
    if output_format == "gif":
        writer = matplotlib.animation.ImageMagickFileWriter(
            fps=frame_per_second, metadata=metadata
        )
    else:
        writer = FFMpegWriter(fps=frame_per_second, metadata=metadata)
    fontsize = 10

    fig = plt.figure()
    with writer.saving(
        fig, datadir + "S" + str(scan) + "_x_movie." + output_format, dpi=100
    ):
        for index in range(nbx // frame_spacing):
            img = obj[:, :, index * frame_spacing]
            plt.clf()
            plt.imshow(img, vmin=vmin_vmax[0], vmax=vmin_vmax[1], cmap=my_cmap)
            ax = plt.gca()
            ax.invert_yaxis()
            ax.set_title("slice # %3d" % index, fontsize=fontsize)
            writer.grab_frame()
