#!/usr/bin/env python3

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import pathlib
import tkinter as tk
from collections import OrderedDict
from numbers import Real
from tkinter import filedialog

import matplotlib as mpl
import numpy as np
from lmfit import Parameters, minimize, report_fit
from matplotlib import pyplot as plt
from scipy.ndimage.measurements import center_of_mass
from scipy.signal import find_peaks

import bcdi.algorithms.algorithms_utils as algo
import bcdi.graph.graph_utils as gu
import bcdi.postprocessing.facet_recognition as fu
import bcdi.utils.utilities as util
import bcdi.utils.validation as valid

helptext = """
Load a 3D BCDI reconstruction (.npz file) containing the field 'amp'. After defining a
support using a threshold on the normalized amplitude, calculate the blurring function
by Richardson-Lucy deconvolution. Extract the resolution from this blurring function in
arbitrary direction. See M. Cherukara et al. Anisotropic nano-scale resolution in 3D
Bragg coherent diffraction imaging.
Appl. Phys. Lett. 113, 203101 (2018); https://doi.org/10.1063/1.5055235
"""

datadir = "D:/data/P10_2nd_test_isosurface_Dec2020/"
savedir = datadir + "blurring_function/test/"
isosurface_threshold = 0.2
phasing_shape = None
# shape of the dataset used during phase retrieval (after an eventual binning in PyNX).
# tuple of 3 positive integers or None, if None the actual shape will be considered.
upsampling_factor = (
    2  # integer, 1=no upsampling_factor, 2=voxel size divided by 2 etc...
)
voxel_size = 5
# number or list of three numbers corresponding to the voxel size in each dimension.
# If a single number is provided, it will use it for all dimensions
sigma_guess = (
    15  # in nm, sigma of the gaussian guess for the blurring function (e.g. mean PRTF)
)
rl_iterations = 50  # number of iterations for the Richardson-Lucy algorithm
center_method = "max"
# 'com' or 'max', method to determine the center of the blurring function for line cuts
comment = ""  # string to add to the filename when saving, should start with "_"
tick_length = 8  # in plots
tick_width = 2  # in plots
roi_width = 20  # in pixels, width of the central region of the psf to plot
debug = True  # True to see more plots
min_offset = 1e-6
# object and support voxels with null value will be set to this number,
# in order to avoid divisions by zero
##########################
# end of user parameters #
##########################

#############################
# define default parameters #
#############################
colors = ("b", "g", "r", "c", "m", "y", "k")  # for plots
markers = (".", "v", "^", "<", ">", "x", "+", "o")  # for plots
mpl.rcParams["axes.linewidth"] = tick_width  # set the linewidth globally
validation_name = "bcdi_blurring_function"

#########################
# check some parameters #
#########################
if not datadir.endswith("/"):
    datadir += "/"
savedir = savedir or datadir
if not savedir.endswith("/"):
    savedir += "/"
pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)
valid.valid_item(
    isosurface_threshold,
    allowed_types=Real,
    min_included=0,
    max_excluded=1,
    name=validation_name,
)
valid.valid_container(
    phasing_shape,
    container_types=(tuple, list, np.ndarray),
    allow_none=True,
    item_types=int,
    min_excluded=0,
    name=validation_name,
)
valid.valid_item(
    value=upsampling_factor, allowed_types=int, min_included=1, name=validation_name
)
valid.valid_container(comment, container_types=str, name=validation_name)
if len(comment) != 0 and not comment.startswith("_"):
    comment = "_" + comment
valid.valid_item(tick_length, allowed_types=int, min_excluded=0, name=validation_name)
valid.valid_item(tick_width, allowed_types=int, min_excluded=0, name=validation_name)
valid.valid_item(debug, allowed_types=bool, name=validation_name)
valid.valid_item(min_offset, allowed_types=Real, min_included=0, name=validation_name)
if isinstance(voxel_size, Real):
    voxel_size = [voxel_size] * 3
voxel_size = list(voxel_size)
valid.valid_container(
    voxel_size,
    container_types=(list, np.ndarray),
    item_types=Real,
    min_excluded=0,
    name=validation_name,
)
valid.valid_item(sigma_guess, allowed_types=Real, min_excluded=0, name=validation_name)
valid.valid_item(rl_iterations, allowed_types=int, min_excluded=0, name=validation_name)
valid.valid_item(roi_width, allowed_types=int, min_excluded=0, name=validation_name)
if center_method not in {"max", "com"}:
    raise ValueError('center_method should be either "com" or "max"')

#########################################################
# load the 3D recontruction , output of phase retrieval #
#########################################################
plt.ion()
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    initialdir=datadir,
    title="Select the reconstructed object",
    filetypes=[("NPZ", "*.npz"), ("NPY", "*.npy"), ("CXI", "*.cxi"), ("HDF5", "*.h5")],
)
if file_path.endswith("npz"):
    kwarg = {"fieldname": "amp"}
else:
    kwarg = {}

obj, extension = util.load_file(file_path, **kwarg)
if extension == ".h5":
    comment = comment + "_mode"

if phasing_shape is None:
    phasing_shape = obj.shape

if upsampling_factor > 1:
    obj, _ = fu.upsample(
        array=obj, upsampling_factor=upsampling_factor, debugging=debug
    )
    print(f"Upsampled object shape = {obj.shape}")
    voxel_size = [vox / upsampling_factor for vox in voxel_size]
    comment += f"_ups{upsampling_factor}"

######################
# define the support #
######################
obj = abs(obj)
min_obj = obj[np.nonzero(obj)].min()
obj = obj / min_obj  # normalize to the non-zero min to avoid dividing by small numbers
obj[obj == 0] = min_offset  # avoid dividing by 0
support = np.zeros(obj.shape)
support[obj >= isosurface_threshold * obj.max()] = 1
support[support == 0] = min_offset

if debug:
    gu.multislices_plot(
        obj,
        sum_frames=False,
        reciprocal_space=False,
        is_orthogonal=True,
        plot_colorbar=True,
        title="normalized modulus",
    )
    gu.multislices_plot(
        support,
        sum_frames=False,
        reciprocal_space=False,
        is_orthogonal=True,
        vmin=0,
        vmax=1,
        plot_colorbar=True,
        title=f"support at threshold {isosurface_threshold}",
    )

###################################
# calculate the blurring function #
###################################
psf_guess = util.gaussian_window(
    window_shape=obj.shape,
    sigma=sigma_guess,
    mu=0.0,
    voxel_size=voxel_size,
    debugging=debug,
)
psf_guess = psf_guess / min_obj  # rescale to the object original min
psf_partial_coh, error = algo.partial_coherence_rl(
    measured_intensity=obj,
    coherent_intensity=support,
    iterations=rl_iterations,
    debugging=False,
    scale="linear",
    is_orthogonal=True,
    reciprocal_space=False,
    guess=psf_guess,
)

denoised_obj, _ = algo.richardson_lucy(
    image=obj, psf=psf_partial_coh, iterations=20, clip=False
)
gu.multislices_plot(
    denoised_obj,
    sum_frames=False,
    reciprocal_space=False,
    is_orthogonal=True,
    plot_colorbar=True,
    title="denoised modulus",
)

psf_partial_coh = abs(psf_partial_coh) / abs(psf_partial_coh).max()
min_error_idx = np.unravel_index(error.argmin(), shape=(rl_iterations,))[0]

peaks, _ = find_peaks(-1 * error)
if peaks.size == 1 and peaks[0] == rl_iterations - 1:
    print("no local minimum for this number of iterations")
else:
    print(f"error local minima at iterations {peaks}")
print(f"min error={error.min():.6f} at iteration {min_error_idx}\n")

###############################################
# plot the retrieved psf and the error metric #
###############################################
if center_method == "com":
    psf_cen = list(map(lambda x: int(np.rint(x)), center_of_mass(psf_partial_coh)))
else:  # 'max'
    psf_cen = list(
        map(
            int,
            np.unravel_index(psf_partial_coh.argmax(), shape=psf_partial_coh.shape),
        )
    )

fig, _, _ = gu.multislices_plot(
    psf_partial_coh,
    scale="linear",
    sum_frames=False,
    title="psf",
    reciprocal_space=False,
    is_orthogonal=True,
    plot_colorbar=True,
    width_z=roi_width,
    width_y=roi_width,
    width_x=roi_width,
    tick_width=tick_width,
    tick_length=tick_length,
    tick_direction="out",
    slice_position=psf_cen,
)
fig.savefig(savedir + f"psf_slices_{rl_iterations}" + comment + ".png")

fig, ax = plt.subplots(figsize=(12, 9))
ax.plot(error, "r.")
ax.set_yscale("log")
ax.set_xlabel("iteration number")
ax.set_ylabel("difference between consecutive iterates")
fig.savefig(savedir + f"error_metric_{rl_iterations}" + comment + ".png")

#######################################################################################
# calculate linecuts of the blurring function through its center of mass and fit them #
#######################################################################################
# create linecuts in the three orthogonal directions
width_z, cut_z = util.linecut(
    array=psf_partial_coh, point=psf_cen, direction=(1, 0, 0), voxel_size=voxel_size
)
width_y, cut_y = util.linecut(
    array=psf_partial_coh, point=psf_cen, direction=(0, 1, 0), voxel_size=voxel_size
)
width_x, cut_x = util.linecut(
    array=psf_partial_coh, point=psf_cen, direction=(0, 0, 1), voxel_size=voxel_size
)
linecuts_dict = OrderedDict(
    [
        ("width_z", width_z),
        ("width_y", width_y),
        ("width_x", width_x),
        ("cut_z", cut_z),
        ("cut_y", cut_y),
        ("cut_x", cut_x),
    ]
)

# define the maximum identical length that can share the linecuts
# (we need to concatenate them)
width_length = min(width_z.size, width_y.size, width_x.size)
linecuts = np.empty(
    (6, width_length)
)  # rows 0:3 for the widths, rows 3:6 for the corresponding cuts
idx = 0
for key in linecuts_dict.keys():  # order is maintained in OrderedDict
    linecuts[idx, :] = util.crop_pad_1d(
        linecuts_dict[key], output_length=width_length
    )  # implicit crop from the center
    idx += 1

# create nb_fit sets of parameters, one per data set
fit_params = Parameters()
for idx in range(3):  # 3 linecuts in orthogonal directions to be fitted by a gaussian
    fit_params.add("amp_%i" % idx, value=1, min=0.1, max=100)
    fit_params.add(
        "cen_%i" % idx,
        value=linecuts[idx, :].mean(),
        min=linecuts[idx, :].min(),
        max=linecuts[idx, :].max(),
    )
    fit_params.add("sig_%i" % idx, value=5, min=0.1, max=100)

# run the global fit to all the data sets
minimization = minimize(
    util.objective_lmfit,
    fit_params,
    args=(linecuts[0:3, :], linecuts[3:6, :], "gaussian"),
)
report_fit(minimization.params)

#######################################################
# plot the linecuts of the blurring function and fits #
#######################################################
# check if the desired roi for plotting is smaller than the available range
if roi_width > width_length // 2:
    print("roi_width larger than the available range, ignoring it")
    roi_width = width_length // 2

fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))
axes_dict = OrderedDict([(0, (ax0, "z")), (1, (ax1, "y")), (2, (ax2, "x"))])
for idx in axes_dict.keys():
    (line,) = axes_dict[idx][0].plot(
        linecuts[
            idx, width_length // 2 - roi_width : width_length // 2 + roi_width
        ],  # x axis
        linecuts[
            idx + 3, width_length // 2 - roi_width : width_length // 2 + roi_width
        ],  # cut
        "ro",
        fillstyle="none",
    )
    line.set_label(f"cut along {axes_dict[idx][1]}")
    fit_x_axis = np.linspace(
        linecuts[
            idx, width_length // 2 - roi_width : width_length // 2 + roi_width
        ].min(),
        linecuts[
            idx, width_length // 2 - roi_width : width_length // 2 + roi_width
        ].max(),
        num=200,
    )
    y_fit = util.function_lmfit(
        params=minimization.params,
        iterator=idx,
        x_axis=fit_x_axis,
        distribution="gaussian",
    )
    (line,) = axes_dict[idx][0].plot(fit_x_axis, y_fit, "b-")
    sig_name = f"sig_{idx+1}"
    line.set_label(
        f"fit, FWHM={2*np.sqrt(2*np.log(2))*minimization.params[sig_name].value:.2f}nm"
    )

gu.savefig(
    savedir=savedir,
    figure=fig,
    axes=(ax0, ax1, ax2),
    tick_width=tick_width,
    tick_length=tick_length,
    tick_labelsize=16,
    xlabels="width (nm)",
    ylabels="psf (a.u.)",
    label_size=20,
    legend=True,
    legend_labelsize=14,
    filename=f"cuts_{rl_iterations}" + comment,
    only_labels=False,
)

################
# save the psf #
################
np.savez_compressed(
    savedir + f"psf_{rl_iterations}" + comment + ".npz",
    psf=psf_partial_coh,
    nb_iter=rl_iterations,
    isosurface_threshold=isosurface_threshold,
    upsampling_factor=upsampling_factor,
)

plt.ioff()
plt.show()
