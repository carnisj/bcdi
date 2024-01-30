#!/usr/bin/env python3

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import gc
import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

import bcdi.graph.graph_utils as gu
import bcdi.utils.utilities as util

helptext = """
Rotate a 3D reciprocal space map around some axis. The data is expected to be in an
orthonormal frame.
"""

scan = 22  # scan number
datadir = (
    f"/nfs/fs/fscxi/experiments/2020/PETRA/P10/11008562/raw/ht_pillar3_{scan:05d}"
    + "/pynx/"
)
tilt = 0.0239082357814962
# rotation angle in radians to be applied counter-clockwise around rotation_axis
# 0.086318314 -0.177905782 0.980254396   qy qx qz Matlab
# -0.177905782 0.980254396 0.086318314   z/qx y/qz x/qy Python CXI/qlab
# 0.086318314 0.980254396 -0.177905782   x/qy y/qz z/qx for the rotation
rotation_axis = (
    0.086318314,
    0.980254396,
    -0.177905782,
)  # in the order (x y z), z axis 0, y axis 1, x axis 2
crop_shape = (1400, 1600, 1450)  # None of a tuple of 3 voxels numbers.
# The data will be cropped symmetrically around origin.
origin = (
    1161,
    912,
    1161,
)  # position in voxels of the origin of the reciprocal space (origin of the rotation)
save = True  # True to save the rotated data
plots = False  # if True, will show plot
comment = (
    ""  # should start with _, comment for the filename when saving the rotated data
)
##################################
# end of user-defined parameters #
##################################

#########################################
# load the data and the mask (optional) #
#########################################
plt.ion()
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    initialdir=datadir, title="Select 3D data", filetypes=[("NPZ", "*.npz")]
)
data, _ = util.load_file(file_path)
nbz, nby, nbx = data.shape
print("data shape:", data.shape)

if crop_shape:
    if len(crop_shape) != 3:
        raise ValueError("crop should be a sequence of 3 voxels numbers")
    if not np.all(np.asarray(origin) - np.asarray(crop_shape) // 2 >= 0):
        raise ValueError("origin incompatible with crop_shape")
    if not (
        origin[0] + crop_shape[0] // 2 <= nbz
        and origin[1] + crop_shape[1] // 2 <= nby
        and origin[2] + crop_shape[2] // 2 <= nbx
    ):
        raise ValueError("origin incompatible with crop_shape")

    data = util.crop_pad(array=data, output_shape=crop_shape, crop_center=origin)
    gc.collect()
    nbz, nby, nbx = data.shape
    print("data shape after cropping:", data.shape)
    # calculate the new position of origin
    new_origin = (crop_shape[0] // 2, crop_shape[1] // 2, crop_shape[2] // 2)
else:
    new_origin = origin

if plots:
    gu.multislices_plot(
        data,
        sum_frames=True,
        scale="log",
        plot_colorbar=True,
        title="S" + str(scan) + "\n Data before rotation",
        vmin=0,
        reciprocal_space=True,
        is_orthogonal=True,
    )

###################
# rotate the data #
###################
# define the rotation matrix in the order (x, y, z)
rotation_matrix = np.array(
    [
        [
            np.cos(tilt) + (1 - np.cos(tilt)) * rotation_axis[0] ** 2,
            rotation_axis[0] * rotation_axis[1] * (1 - np.cos(tilt))
            - rotation_axis[2] * np.sin(tilt),
            rotation_axis[0] * rotation_axis[2] * (1 - np.cos(tilt))
            + rotation_axis[1] * np.sin(tilt),
        ],
        [
            rotation_axis[1] * rotation_axis[0] * (1 - np.cos(tilt))
            + rotation_axis[2] * np.sin(tilt),
            np.cos(tilt) + (1 - np.cos(tilt)) * rotation_axis[1] ** 2,
            rotation_axis[1] * rotation_axis[2] * (1 - np.cos(tilt))
            - rotation_axis[0] * np.sin(tilt),
        ],
        [
            rotation_axis[2] * rotation_axis[0] * (1 - np.cos(tilt))
            - rotation_axis[1] * np.sin(tilt),
            rotation_axis[2] * rotation_axis[1] * (1 - np.cos(tilt))
            + rotation_axis[0] * np.sin(tilt),
            np.cos(tilt) + (1 - np.cos(tilt)) * rotation_axis[2] ** 2,
        ],
    ]
)

transfer_matrix = rotation_matrix.transpose()
old_z = np.arange(-new_origin[0], -new_origin[0] + nbz, 1)
old_y = np.arange(-new_origin[1], -new_origin[1] + nby, 1)
old_x = np.arange(-new_origin[2], -new_origin[2] + nbx, 1)

myz, myy, myx = np.meshgrid(old_z, old_y, old_x, indexing="ij")

new_x = (
    transfer_matrix[0, 0] * myx
    + transfer_matrix[0, 1] * myy
    + transfer_matrix[0, 2] * myz
)
new_y = (
    transfer_matrix[1, 0] * myx
    + transfer_matrix[1, 1] * myy
    + transfer_matrix[1, 2] * myz
)
new_z = (
    transfer_matrix[2, 0] * myx
    + transfer_matrix[2, 1] * myy
    + transfer_matrix[2, 2] * myz
)
del myx, myy, myz
gc.collect()

rgi = RegularGridInterpolator(
    (old_z, old_y, old_x), data, method="linear", bounds_error=False, fill_value=0
)
rot_data = rgi(
    np.concatenate(
        (
            new_z.reshape((1, new_z.size)),
            new_y.reshape((1, new_z.size)),
            new_x.reshape((1, new_z.size)),
        )
    ).transpose()
)
rot_data = rot_data.reshape((nbz, nby, nbx)).astype(data.dtype)
del data
gc.collect()

if plots:
    gu.multislices_plot(
        rot_data,
        sum_frames=True,
        scale="log",
        plot_colorbar=True,
        title="S" + str(scan) + "\n Data after rotation",
        vmin=0,
        reciprocal_space=True,
        is_orthogonal=True,
    )
if save:
    comment = comment + "_" + str(nbz) + "_" + str(nby) + "_" + str(nbx)
    np.savez_compressed(
        datadir + "S" + str(scan) + "_pynx_rotated" + comment + ".npz", data=rot_data
    )

del rot_data
gc.collect()

###########################
# optional: rotate a mask #
###########################
try:
    file_path = filedialog.askopenfilename(
        initialdir=datadir, title="Select 3D mask", filetypes=[("NPZ", "*.npz")]
    )
    mask, _ = util.load_file(file_path)
    skip_mask = False
except ValueError:
    print("skip mask")
    mask = None
    skip_mask = True

if not skip_mask:
    if crop_shape:
        mask = util.crop_pad(array=mask, output_shape=crop_shape, crop_center=origin)
        gc.collect()

    if plots:
        gu.multislices_plot(
            mask,
            sum_frames=False,
            scale="linear",
            plot_colorbar=True,
            title="S" + str(scan) + "\n Mask before rotation",
            vmin=0,
            reciprocal_space=True,
            is_orthogonal=True,
        )

    rgi = RegularGridInterpolator(
        (old_z, old_y, old_x),
        mask,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    rot_mask = rgi(
        np.concatenate(
            (
                new_z.reshape((1, new_z.size)),
                new_y.reshape((1, new_z.size)),
                new_x.reshape((1, new_z.size)),
            )
        ).transpose()
    )
    rot_mask = rot_mask.reshape((nbz, nby, nbx)).astype(mask.dtype)
    del mask
    gc.collect()

    rot_mask[np.isnan(rot_mask)] = 1
    rot_mask[np.nonzero(rot_mask)] = 1

    if plots:
        gu.multislices_plot(
            rot_mask,
            sum_frames=False,
            scale="linear",
            plot_colorbar=True,
            title="S" + str(scan) + "\n Mask after rotation",
            vmin=0,
            reciprocal_space=True,
            is_orthogonal=True,
        )
    if save:
        np.savez_compressed(
            datadir + "S" + str(scan) + "_maskpynx_rotated" + comment + ".npz",
            mask=rot_mask,
        )

plt.ioff()
plt.show()
