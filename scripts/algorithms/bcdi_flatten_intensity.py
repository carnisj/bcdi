#!/usr/bin/env python3

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import gc
import os
import tkinter as tk
from tkinter import filedialog

import numpy as np
import scipy.optimize as optimize
from matplotlib import pyplot as plt

import bcdi.graph.graph_utils as gu
import bcdi.postprocessing.postprocessing_utils as pu
import bcdi.utils.utilities as util

helptext = """
Apply intensity normalization of additive spatially multiplexed patterns with n
encoded phases, as described in:
R. Jaurez-Salazar et al. Optics and Lasers in Engineering 77, 225-229 (2016).

Input: a 3D real intensity array
"""

datadir = ""
savedir = "D:/data/P10_August2019_CDI/data/gold_2_2_2_00022/pynx/scratch/"
comment = "_gaussian_21_5"  # should start with _
threshold = 0.075
# threshold used to define the support for background fitting
# (intensity normalized to 1)
roll_modes = (0, -10, 0)  # axis=(0, 1, 2), correct a misalignement of the data
save = True  # True to save the result as a NPZ file
nb_phases = 1  # number of encoded phases, generally 1 if this is a single measurement
#########################################
# parameters for the background fitting #
#########################################
background_method = "gaussian"
# 'gaussian', 'polyfit' or 'skip': 'gaussian' will convolve a gaussian with the object,
# 'polyfit' will fit a polynomial or order background_order to the object.
# 'skip' defines a zero background
background_order = 1  # degree of the polynomial for background fitting 1~4.
background_kernel = (
    21  # size of the kernel for the 'gaussian' method  # 2*ceil(2*sigma)+1 in matlab
)
background_sigma = 5  # standard deviation of the gaussian for the 'gaussian' method
#########################################
# parameters for the modulation fitting #
#########################################
modulation_method = "gaussian"
# 'gaussian' or 'polyfit': 'gaussian' will convolve a gaussian with the object,
# 'polyfit' will fit a polynomial or order modulation_order to the object.
modulation_order = 4  # degree of the polynomial for modulation fitting 1~4
modulation_kernel = (
    21  # size of the kernel for the 'gaussian' method  # 2*ceil(2*sigma)+1 in matlab
)
modulation_sigma = 5  # standard deviation of the gaussian for the 'gaussian' method
##########################
# end of user parameters #
##########################

####################
# check parameters #
####################
if not 1 <= background_order <= 4:
    raise ValueError("polynomial fitting of order > 4 not implemented")
if not 1 <= modulation_order <= 4:
    raise ValueError("polynomial fitting of order > 4 not implemented")
if background_method not in [
    "gaussian",
    "polyfit",
    "skip",
]:
    raise ValueError("invalid setting for background_method")
if modulation_method not in [
    "gaussian",
    "polyfit",
]:
    raise ValueError("invalid setting for modulation_method")

#################
# load the data #
#################
plt.ion()
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    initialdir=datadir,
    title="Select the CCF file",
    filetypes=[("HDF5", "*.h5"), ("NPZ", "*.npz"), ("NPY", "*.npy"), ("CXI", "*.cxi")],
)
filename = os.path.splitext(os.path.basename(file_path))[
    0
]  # the extension .npz is removed
filename = filename + comment
print("filename:", filename)
obj, extension = util.load_file(file_path)
nz, ny, nx = obj.shape

######################
# data preprocessing #
######################
obj[np.isnan(obj)] = 0
obj = abs(obj)
obj = obj / obj.max()  # normalize to 1

# correct a misalignement of the object
obj = np.roll(obj, roll_modes, axis=(0, 1, 2))
gu.multislices_plot(
    abs(obj),
    sum_frames=True,
    plot_colorbar=True,
    reciprocal_space=False,
    is_orthogonal=True,
    title="obj after centering",
)
original_obj = np.copy(obj)

############################
# determine the background #
############################
support = np.zeros((nz, ny, nx))
support[obj >= threshold] = 1
support_points = np.nonzero(support)
nb_nonzero = support_points[0].size
temp_obj = np.copy(obj)
temp_obj[obj < threshold] = np.nan
gu.multislices_plot(
    temp_obj,
    sum_frames=False,
    plot_colorbar=True,
    reciprocal_space=False,
    is_orthogonal=True,
    title="points belonging to the support",
)
del temp_obj
gc.collect()

if background_method == "polyfit":
    grid_z, grid_y, grid_x = np.meshgrid(
        np.arange(0, nz, 1), np.arange(0, ny, 1), np.arange(0, nx, 1), indexing="ij"
    )
    grid = np.concatenate(
        (
            grid_z.reshape((1, obj.size)),
            grid_y.reshape((1, obj.size)),
            grid_x.reshape((1, obj.size)),
        ),
        axis=0,
    )  # xdata should have a 3xN array
    if background_order == 1:
        guess = np.ones(4)
        params, cov = optimize.curve_fit(
            util.fit3d_poly1,
            xdata=support_points,
            ydata=obj[support_points].reshape(nb_nonzero),
            p0=guess,
        )
        background = util.fit3d_poly1(grid, params[0], params[1], params[2], params[3])
    elif background_order == 2:
        guess = np.ones(7)
        params, cov = optimize.curve_fit(
            util.fit3d_poly2,
            xdata=support_points,
            ydata=obj[support_points].reshape(nb_nonzero),
            p0=guess,
        )
        background = util.fit3d_poly2(
            grid,
            params[0],
            params[1],
            params[2],
            params[3],
            params[4],
            params[5],
            params[6],
        )
    elif background_order == 3:
        guess = np.ones(10)
        params, cov = optimize.curve_fit(
            util.fit3d_poly3,
            xdata=support_points,
            ydata=obj[support_points].reshape(nb_nonzero),
            p0=guess,
        )
        background = util.fit3d_poly3(
            grid,
            params[0],
            params[1],
            params[2],
            params[3],
            params[4],
            params[5],
            params[6],
            params[7],
            params[8],
            params[9],
        )
    else:
        guess = np.ones(13)
        params, cov = optimize.curve_fit(
            util.fit3d_poly4,
            xdata=support_points,
            ydata=obj[support_points].reshape(nb_nonzero),
            p0=guess,
        )
        background = util.fit3d_poly4(
            grid,
            params[0],
            params[1],
            params[2],
            params[3],
            params[4],
            params[5],
            params[6],
            params[7],
            params[8],
            params[9],
            params[10],
            params[11],
            params[12],
        )
    del grid_z, grid_y, grid_x, grid
    gc.collect()

elif background_method == "gaussian":
    obj[obj < threshold] = threshold  # avoid steps at the boundaries of the support
    background = pu.filter_3d(
        obj,
        filter_name="gaussian",
        kernel_length=background_kernel,
        sigma=background_sigma,
        debugging=False,
    )
else:  # skip
    print("skipping background determination")
    background = np.zeros((nz, ny, nx))

##############################
# plot the fitted background #
##############################
background = background.reshape((nz, ny, nx))
gu.multislices_plot(
    background,
    sum_frames=False,
    plot_colorbar=True,
    reciprocal_space=False,
    is_orthogonal=True,
    title="fitted background",
)

#######################################################
# subtract the background to the intensity and square #
#######################################################
obj_bck = np.square(obj - background)
obj_bck[np.isnan(obj_bck)] = 0
gu.multislices_plot(
    obj_bck,
    sum_frames=False,
    plot_colorbar=True,
    reciprocal_space=False,
    title="(obj-background)**2",
)

#############################################################
# fit the modulation to the points belonging to the support #
#############################################################
if background_method == "polyfit":
    grid_z, grid_y, grid_x = np.meshgrid(
        np.arange(0, nz, 1), np.arange(0, ny, 1), np.arange(0, nx, 1), indexing="ij"
    )
    grid = np.concatenate(
        (
            grid_z.reshape((1, obj.size)),
            grid_y.reshape((1, obj.size)),
            grid_x.reshape((1, obj.size)),
        ),
        axis=0,
    )  # xdata should have a 3xN array
    if modulation_order == 1:
        guess = np.ones(4)
        params, cov = optimize.curve_fit(
            util.fit3d_poly1,
            xdata=support_points,
            ydata=obj_bck[support_points].reshape(nb_nonzero),
            p0=guess,
        )
        modulation = util.fit3d_poly1(grid, params[0], params[1], params[2], params[3])
    elif modulation_order == 2:
        guess = np.ones(7)
        params, cov = optimize.curve_fit(
            util.fit3d_poly2,
            xdata=support_points,
            ydata=obj_bck[support_points].reshape(nb_nonzero),
            p0=guess,
        )
        modulation = util.fit3d_poly2(
            grid,
            params[0],
            params[1],
            params[2],
            params[3],
            params[4],
            params[5],
            params[6],
        )
    elif modulation_order == 3:
        guess = np.ones(10)
        params, cov = optimize.curve_fit(
            util.fit3d_poly3,
            xdata=support_points,
            ydata=obj_bck[support_points].reshape(nb_nonzero),
            p0=guess,
        )
        modulation = util.fit3d_poly3(
            grid,
            params[0],
            params[1],
            params[2],
            params[3],
            params[4],
            params[5],
            params[6],
            params[7],
            params[8],
            params[9],
        )
    else:  # 4th order
        guess = np.ones(13)
        params, cov = optimize.curve_fit(
            util.fit3d_poly4,
            xdata=support_points,
            ydata=obj_bck[support_points].reshape(nb_nonzero),
            p0=guess,
        )
        modulation = util.fit3d_poly4(
            grid,
            params[0],
            params[1],
            params[2],
            params[3],
            params[4],
            params[5],
            params[6],
            params[7],
            params[8],
            params[9],
            params[10],
            params[11],
            params[12],
        )
else:  # 'gaussian'
    modulation = pu.filter_3d(
        obj_bck,
        filter_name="gaussian",
        kernel_length=modulation_kernel,
        sigma=modulation_sigma,
        debugging=False,
    )

##############################
# plot the fitted modulation #
##############################
modulation = modulation.reshape((nz, ny, nx))
modulation = np.sqrt(2 * nb_phases * modulation)
modulation = modulation / modulation[support_points].max()
modulation[np.isnan(modulation)] = 1
gu.multislices_plot(
    modulation,
    sum_frames=False,
    plot_colorbar=True,
    reciprocal_space=False,
    is_orthogonal=True,
    title="fitted modulation",
)

############################################
# calculate and plot the normalized object #
############################################
result = np.divide(original_obj - background, modulation)
result[np.isnan(result)] = 0
result = (
    result - result[support_points].min()
)  # offset intensities such that the min value is 0
result = result / result[support_points].max()
result[support == 0] = 0
piz, piy, pix = np.unravel_index(result.argmax(), shape=(nz, ny, nx))
print("maximum at voxel:", piz, piy, pix, "   value=", result.max())
gu.multislices_plot(
    result,
    slice_position=(piz, piy, pix),
    sum_frames=False,
    plot_colorbar=True,
    vmin=0,
    vmax=1,
    reciprocal_space=False,
    is_orthogonal=True,
    title="result at max",
)


fig = gu.combined_plots(
    tuple_array=(original_obj, original_obj, original_obj, result, result, result),
    tuple_sum_frames=False,
    tuple_sum_axis=(0, 1, 2, 0, 1, 2),
    tuple_colorbar=True,
    tuple_title=("Original", "Original", "Original", "Result", "Result", "Result"),
    tuple_scale="linear",
    is_orthogonal=True,
    reciprocal_space=False,
    tuple_vmin=0,
    tuple_vmax=1,
    position=(321, 323, 325, 322, 324, 326),
)

if save:
    np.savez_compressed(savedir + filename + ".npz", obj=result)
    fig.savefig(savedir + filename + ".png")
plt.ioff()
plt.show()
