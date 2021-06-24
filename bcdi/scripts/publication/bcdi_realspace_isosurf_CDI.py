#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from mayavi import mlab
import numpy as np
import sys
import tkinter as tk
from tkinter import filedialog

import bcdi.graph.graph_utils as gu
import bcdi.utils.utilities as util

helptext = """
Template for 3d isosurface figures of a real space forward CDI reconstruction.

Open a CDI reconstruction file and save individual figures including a length scale.
"""

scan = 22  # spec scan number
root_folder = "D:/data/P10_August2019_CDI/data/"
sample_name = "gold_2_2_2"
homedir = (
    root_folder
    + sample_name
    + "_"
    + str("{:05d}".format(scan))
    + "/pynx/1000_1000_1000_1_1_1/current_paper//"
)
comment = "_current_color"  # should start with _

save_YZ = True  # True to save the modulus in YZ plane
save_XZ = True  # True to save the modulus in XZ plane
save_XY = True  # True to save the modulus in XY plane
grey_background = False  # True to set the background to grey in 2D plots
tick_direction = "in"  # 'out', 'in', 'inout'
tick_length = 10  # in plots
tick_width = 2  # in plots
cmap = "custom"  # matplotlib colormap name, or 'custom'
vmax = 0.8  # number or 'max_slice', maximum value of imshow in 2D slices

voxel_size = 9.42  # in nm, supposed isotropic
tick_spacing = 500  # for plots, in nm
field_of_view = [
    2000,
    2000,
    2000,
]  # [z,y,x] in nm, can be larger than the total width (the array will be padded)
# the number of labels of mlab.axes() is an integer and is be calculated as:
# field_of_view[0]/tick_spacing
# therefore it is better to use an isotropic field_of_view
threshold_isosurface = 0.4  # threshold for the 3D isosurface plot  #0.4 without ML
threshold_modulus = 0.06  # threshold for 2D plots  # 0.06 without ML
axis_outofplane = (
    None  # in order x y z for rotate_crystal(), axis to align on y vertical up
)
# leave it to None if you do not need to rotate the object
axis_inplane = (
    None  # in order x y z for rotate_crystal(), axis to align on x downstream
)
# leave it to None if you do not need to rotate the object
##########################
# end of user parameters #
##########################

#########################
# check some parameters #
#########################
if grey_background:
    bad_color = "0.7"
else:
    bad_color = "1.0"  # white background

if cmap == "custom":
    # define colormap
    colormap = gu.Colormap(bad_color=bad_color)
    cmap = colormap.cmap

#############
# load data #
#############
plt.ion()
root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename(
    initialdir=homedir,
    title="Select reconstruction file",
    filetypes=[("NPZ", "*.npz"), ("CXI", "*.cxi"), ("HDF5", "*.h5")],
)
obj, _ = util.load_file(file_path)

if obj.ndim != 3:
    print("a 3D reconstruction array is expected")
    sys.exit()

obj = abs(obj)
numz, numy, numx = obj.shape
print("Initial data size: (", numz, ",", numy, ",", numx, ")")

#############################
# rotate the reconstruction #
#############################
if axis_outofplane is not None or axis_inplane is not None:
    new_shape = [int(1.2 * numz), int(1.2 * numy), int(1.2 * numx)]
    obj = util.crop_pad(array=obj, output_shape=new_shape, debugging=False)
    numz, numy, numx = obj.shape

    print(
        "Cropped/padded data size before rotating: (", numz, ",", numy, ",", numx, ")"
    )
    print("Rotating object to have the crystallographic axes along array axes")
    if axis_outofplane is not None:
        obj = util.rotate_crystal(
            arrays=obj,
            axis_to_align=axis_outofplane,
            reference_axis=np.array([0, 1, 0]),
            debugging=True,
        )  # out of plane alignement
    if axis_inplane is not None:
        obj = util.rotate_crystal(
            arrays=obj,
            axis_to_align=axis_inplane,
            reference_axis=np.array([1, 0, 0]),
            debugging=True,
        )  # inplane alignement

#################################################
#  pad array to obtain the desired field of view #
##################################################
amp = np.copy(obj)
amp = np.flip(
    amp, 2
)  # mayavi expect xyz, but we provide downstream/upward/outboard
# which is not in the correct order
amp = amp / amp.max()
amp[amp < threshold_isosurface] = 0

z_pixel_FOV = int(
    np.rint((field_of_view[0] / voxel_size) / 2)
)  # half-number of pixels corresponding to the FOV
y_pixel_FOV = int(
    np.rint((field_of_view[1] / voxel_size) / 2)
)  # half-number of pixels corresponding to the FOV
x_pixel_FOV = int(
    np.rint((field_of_view[2] / voxel_size) / 2)
)  # half-number of pixels corresponding to the FOV
new_shape = [
    max(numz, 2 * z_pixel_FOV),
    max(numy, 2 * y_pixel_FOV),
    max(numx, 2 * x_pixel_FOV),
]
amp = util.crop_pad(array=amp, output_shape=new_shape, debugging=False)
numz, numy, numx = amp.shape
print(
    "Cropped/padded data size for 3D isosurface plot: (",
    numz,
    ",",
    numy,
    ",",
    numx,
    ")",
)

#########################################
# plot 3D isosurface (perspective view) #
#########################################
grid_qx, grid_qz, grid_qy = np.mgrid[
    0 : 2 * z_pixel_FOV * voxel_size : voxel_size,
    0 : 2 * y_pixel_FOV * voxel_size : voxel_size,
    0 : 2 * x_pixel_FOV * voxel_size : voxel_size,
]
extent = [
    0,
    2 * z_pixel_FOV * voxel_size,
    0,
    2 * y_pixel_FOV * voxel_size,
    0,
    2 * x_pixel_FOV * voxel_size,
]
# in CXI convention, z is downstream, y vertical and x outboard

myfig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
mlab.contour3d(
    grid_qx,
    grid_qz,
    grid_qy,
    amp[
        numz // 2 - z_pixel_FOV : numz // 2 + z_pixel_FOV,
        numy // 2 - y_pixel_FOV : numy // 2 + y_pixel_FOV,
        numx // 2 - x_pixel_FOV : numx // 2 + x_pixel_FOV,
    ],
    contours=[threshold_isosurface],
    color=(0.7, 0.7, 0.7),
)
mlab.view(
    azimuth=0, elevation=50, distance=3 * field_of_view[0]
)  # azimut is the rotation around z axis of mayavi (x)
mlab.roll(0)

ax = mlab.axes(
    extent=extent, line_width=2.0, nb_labels=int(1 + field_of_view[0] / tick_spacing)
)
mlab.savefig(homedir + "S" + str(scan) + "-perspective_labels.png", figure=myfig)
ax.label_text_property.opacity = 0.0
ax.title_text_property.opacity = 0.0
mlab.savefig(homedir + "S" + str(scan) + "-perspective.png", figure=myfig)
# mlab.close(myfig)

#################
# plot 2D views #
#################
amp = np.copy(obj)
amp = amp / amp.max()
amp[amp < threshold_modulus] = 0

pixel_spacing = tick_spacing / voxel_size
pixel_FOV = [int(np.rint((fov / voxel_size) / 2)) for fov in field_of_view]
# half-number of pixels corresponding to the FOV
new_shape = [
    max(numz, 2 * pixel_FOV[0]),
    max(numy, 2 * pixel_FOV[1]),
    max(numx, 2 * pixel_FOV[2]),
]
amp = util.crop_pad(array=amp, output_shape=new_shape, debugging=False)
numz, numy, numx = amp.shape
print("Cropped/padded data size for 2D plots: (", numz, ",", numy, ",", numx, ")")

# middle slice in YZ plane
fig, ax0 = plt.subplots(1, 1)
try:
    plt0 = ax0.imshow(
        amp[
            numz // 2 - pixel_FOV[0] : numz // 2 + pixel_FOV[0],
            numy // 2 - pixel_FOV[1] : numy // 2 + pixel_FOV[1],
            numx // 2,
        ],
        vmin=0,
        vmax=vmax,
        cmap=cmap,
    )
except ValueError:
    if vmax == "max_slice":
        slice_data = amp[
            numz // 2 - pixel_FOV[0] : numz // 2 + pixel_FOV[0],
            numy // 2 - pixel_FOV[1] : numy // 2 + pixel_FOV[1],
            numx // 2,
        ]
        plt0 = ax0.imshow(slice_data, vmin=0, vmax=slice_data.max(), cmap=cmap)
    else:
        print("Incorrect value for vmax parameter")
        sys.exit()
ax0.xaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
ax0.yaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
ax0.tick_params(
    labelbottom=False,
    labelleft=False,
    top=True,
    right=True,
    direction=tick_direction,
    length=tick_length,
    width=tick_width,
)
if save_YZ:
    fig.savefig(homedir + "amp_YZ" + comment + ".png", bbox_inches="tight")
    if vmax == "max_slice":
        plt.colorbar(plt0, ax=ax0)
        fig.savefig(homedir + "amp_YZ" + comment + "_colorbar.png", bbox_inches="tight")

# middle slice in XZ plane
fig, ax1 = plt.subplots(1, 1)
try:
    plt1 = ax1.imshow(
        amp[
            numz // 2 - pixel_FOV[0] : numz // 2 + pixel_FOV[0],
            numy // 2,
            numx // 2 - pixel_FOV[2] : numx // 2 + pixel_FOV[2],
        ],
        vmin=0,
        vmax=vmax,
        cmap=cmap,
    )
except ValueError:  # vmax = 'max_slice'
    slice_data = amp[
        numz // 2 - pixel_FOV[0] : numz // 2 + pixel_FOV[0],
        numy // 2,
        numx // 2 - pixel_FOV[2] : numx // 2 + pixel_FOV[2],
    ]
    plt1 = ax1.imshow(slice_data, vmin=0, vmax=slice_data.max(), cmap=cmap)

ax1.invert_yaxis()
ax1.xaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
ax1.yaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
ax1.tick_params(
    labelbottom=False,
    labelleft=False,
    top=True,
    right=True,
    direction=tick_direction,
    length=tick_length,
    width=tick_width,
)
if save_XZ:
    fig.savefig(homedir + "amp_XZ" + comment + ".png", bbox_inches="tight")
    if vmax == "max_slice":
        plt.colorbar(plt1, ax=ax1)
        fig.savefig(homedir + "amp_XZ" + comment + "_colorbar.png", bbox_inches="tight")

# middle slice in XY plane
fig, ax2 = plt.subplots(1, 1)
try:
    plt2 = ax2.imshow(
        amp[
            numz // 2,
            numy // 2 - pixel_FOV[1] : numy // 2 + pixel_FOV[1],
            numx // 2 - pixel_FOV[2] : numx // 2 + pixel_FOV[2],
        ],
        vmin=0,
        vmax=vmax,
        cmap=cmap,
    )
except ValueError:  # vmax = 'max_slice'
    slice_data = amp[
        numz // 2,
        numy // 2 - pixel_FOV[1] : numy // 2 + pixel_FOV[1],
        numx // 2 - pixel_FOV[2] : numx // 2 + pixel_FOV[2],
    ]
    plt2 = ax2.imshow(slice_data, vmin=0, vmax=slice_data.max(), cmap=cmap)
ax2.invert_yaxis()
ax2.xaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
ax2.yaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
ax2.tick_params(
    labelbottom=False,
    labelleft=False,
    top=True,
    right=True,
    direction=tick_direction,
    length=tick_length,
    width=tick_width,
)
if save_XY:
    fig.savefig(homedir + "amp_XY" + comment + ".png", bbox_inches="tight")
    plt.colorbar(plt2, ax=ax2)
    fig.savefig(homedir + "amp_XY" + comment + "_colorbar.png", bbox_inches="tight")

plt.ioff()
plt.show()
