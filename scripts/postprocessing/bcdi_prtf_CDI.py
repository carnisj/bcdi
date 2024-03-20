#!/usr/bin/env python3

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import gc
import pathlib
import sys
import tkinter as tk
from tkinter import filedialog

import numpy as np
from matplotlib import pyplot as plt
from numpy.fft import fftn, fftshift
from scipy.interpolate import interp1d
from scipy.ndimage.measurements import center_of_mass

import bcdi.graph.graph_utils as gu
import bcdi.utils.utilities as util
from bcdi.graph.colormap import ColormapFactory

helptext = """
Calculate the resolution of a forward CDI reconstruction using the phase retrieval
transfer function (PRTF). The diffraction pattern and reconstructions should be in
the orthogonal laboratory frame. Q values need to be provided.

For the laboratory frame, the CXI convention is used: z downstream, y vertical,
x outboard. For q, the usual convention is used: qx downstream, qz vertical, qy outboard
"""

scan = 22
root_folder = "D:/data/P10_August2019/data/"  # location of the .spec or log file
sample_name = "gold_2_2_2_000"  # "SN"  #
datadir = root_folder + sample_name + str(scan) + "/pynx/1000_2_debug/"
savedir = None
comment = "_hotpixel"  # should start with _
binning = (
    1,
    1,
    1,
)  # binning factor used during phasing: axis0=downstream,
# axis1=vertical up, axis2=outboard. Leave it to (1, 1, 1) if the binning factor is the
# same between the input data and the phasing output
original_shape = (
    500,
    500,
    500,
)  # shape of the array used during phasing, before an eventual crop of the result
###########
# options #
###########
normalize_prtf = False  # set to True when the solution is the first mode
# then the intensity needs to be normalized
debug = False  # True to show more plots
q_max = None  # in 1/nm, PRTF normalization using only points smaller than q_max.
# Leave it to None otherwise.
##########################
# end of user parameters #
##########################

###################
# define colormap #
###################
my_cmap = ColormapFactory().cmap

#########################
# check some parameters #
#########################
savedir = savedir or datadir
pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)

##############################
# load reciprocal space data #
##############################
print("\nScan", scan)
print("Datadir:", datadir)

plt.ion()
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    initialdir=datadir,
    title="Select the diffraction pattern",
    filetypes=[("NPZ", "*.npz")],
)
npzfile = np.load(file_path)
diff_pattern = npzfile[list(npzfile.files)[0]].astype(float)
nz, ny, nx = diff_pattern.shape
print("Data shape:", nz, ny, nx)

#############
# load mask #
#############
file_path = filedialog.askopenfilename(
    initialdir=datadir, title="Select the mask", filetypes=[("NPZ", "*.npz")]
)
npzfile = np.load(file_path)
mask = npzfile[list(npzfile.files)[0]]
if debug:
    gu.multislices_plot(
        diff_pattern,
        sum_frames=False,
        plot_colorbar=True,
        cmap=my_cmap,
        title="measured amplitude",
        scale="log",
        vmin=np.nan,
        vmax=np.nan,
        reciprocal_space=True,
        is_orthogonal=True,
    )

    gu.multislices_plot(
        mask,
        sum_frames=False,
        plot_colorbar=True,
        cmap=my_cmap,
        title="mask",
        scale="linear",
        vmin=0,
        vmax=1,
        reciprocal_space=True,
        is_orthogonal=True,
    )

#################
# load q values #
#################
file_path = filedialog.askopenfilename(
    initialdir=datadir, title="Select q values", filetypes=[("NPZ", "*.npz")]
)
npzfile = np.load(file_path)
qx = npzfile["qx"]  # downstream
qz = npzfile["qz"]  # vertical up
qy = npzfile["qy"]  # outboard

###################################
# bin data and q values if needed #
###################################
if any(bin_factor != 1 for bin_factor in binning):
    diff_pattern = util.bin_data(array=diff_pattern, binning=binning, debugging=False)
    mask = util.bin_data(array=mask, binning=binning, debugging=False)
    mask[np.nonzero(mask)] = 1
    qx = qx[:: binning[0]]
    qy = qy[:: binning[1]]
    qz = qz[:: binning[2]]

############################
# plot diffraction pattern #
############################
nz, ny, nx = diff_pattern.shape
print(
    "Data shape after binning=",
    nz,
    ny,
    nx,
    " Max(measured amplitude)=",
    np.sqrt(diff_pattern).max(),
    " at voxel # ",
    np.unravel_index(diff_pattern.argmax(), diff_pattern.shape),
)
# print(diff_pattern[434, 54, 462])
mask[diff_pattern < 1.0] = (
    1  # do not use interpolated points with a low photon count in PRTF calculation.
)
# These points results in overshoots in the PRTF
diff_pattern[np.nonzero(mask)] = 0

z0, y0, x0 = center_of_mass(diff_pattern)
z0, y0, x0 = [int(z0), int(y0), int(x0)]
print(
    "COM of measured pattern after masking: ",
    z0,
    y0,
    x0,
    " Number of unmasked photons =",
    diff_pattern.sum(),
)

plt.figure()
plt.imshow(np.log10(np.sqrt(diff_pattern).sum(axis=0)), cmap=my_cmap, vmin=0)
plt.title("abs(binned measured amplitude).sum(axis=0)")
plt.colorbar()
plt.pause(0.1)

if debug:
    gu.multislices_plot(
        diff_pattern,
        sum_frames=False,
        plot_colorbar=True,
        cmap=my_cmap,
        title="abs(binned measured amplitude)",
        scale="log",
        vmin=0,
        reciprocal_space=True,
        is_orthogonal=True,
    )
    gu.multislices_plot(
        mask,
        sum_frames=False,
        plot_colorbar=True,
        cmap=my_cmap,
        title="binned mask",
        scale="linear",
        vmin=0,
        vmax=1,
        reciprocal_space=True,
        is_orthogonal=True,
    )

##########################################################
# calculate the distances in q space relative to the COM #
##########################################################
qxCOM = qx[z0]
qzCOM = qz[y0]
qyCOM = qy[x0]
print("COM[qx, qz, qy] = ", qxCOM, qzCOM, qyCOM)

qx = qx[:, np.newaxis, np.newaxis]  # broadcast array
qy = qy[np.newaxis, np.newaxis, :]  # broadcast array
qz = qz[np.newaxis, :, np.newaxis]  # broadcast array

distances_q = np.sqrt((qx - qxCOM) ** 2 + (qy - qyCOM) ** 2 + (qz - qzCOM) ** 2)
del qx, qy, qz
gc.collect()

if debug:
    gu.multislices_plot(
        distances_q,
        sum_frames=False,
        plot_colorbar=True,
        cmap=my_cmap,
        title="distances_q",
        scale="linear",
        vmin=np.nan,
        vmax=np.nan,
        reciprocal_space=True,
        is_orthogonal=True,
    )

#############################
# load reconstructed object #
#############################
file_path = filedialog.askopenfilename(
    initialdir=datadir,
    title="Select the reconstructed object",
    filetypes=[("NPZ", "*.npz"), ("NPY", "*.npy"), ("CXI", "*.cxi"), ("HDF5", "*.h5")],
)

obj, extension = util.load_file(file_path)
print("Opening ", file_path)

if extension == ".h5":
    comment = comment + "_mode"

# check if the shape is the same as the measured diffraction pattern
if obj.shape != original_shape:
    print(
        "Reconstructed object shape = ",
        obj.shape,
        "different from the experimental diffraction pattern: crop/pad",
    )
    new_shape = (
        int(original_shape[0] / binning[0]),
        int(original_shape[1] / binning[1]),
        int(original_shape[2] / binning[2]),
    )
    obj = util.crop_pad(array=obj, output_shape=new_shape, debugging=False)

if obj.shape != diff_pattern.shape:
    print(
        "Reconstructed object shape = ",
        obj.shape,
        "different from diffraction pattern shape = ",
        diff_pattern.shape,
    )
    sys.exit()

#################################################
# calculate the retrieved diffraction amplitude #
#################################################
phased_fft = fftshift(fftn(obj)) / (
    np.sqrt(nz) * np.sqrt(ny) * np.sqrt(nx)
)  # complex amplitude
del obj
gc.collect()

plt.figure()
plt.imshow(np.log10(abs(phased_fft).sum(axis=0)), cmap=my_cmap, vmin=0)
plt.title("abs(retrieved amplitude).sum(axis=0)")
plt.colorbar()
plt.pause(0.1)

phased_fft[np.nonzero(mask)] = 0  # do not take mask voxels into account
print("Max(retrieved amplitude) =", abs(phased_fft).max())
print(
    "COM of the retrieved diffraction pattern after masking: ",
    center_of_mass(abs(phased_fft)),
)
del mask
gc.collect()

gu.combined_plots(
    tuple_array=(diff_pattern, phased_fft),
    tuple_sum_frames=False,
    tuple_sum_axis=(0, 0),
    tuple_width_v=None,
    tuple_width_h=None,
    tuple_colorbar=False,
    tuple_vmin=(-1, -1),
    tuple_vmax=np.nan,
    tuple_title=("measurement", "phased_fft"),
    tuple_scale="log",
)

###########################################
# check alignment of diffraction patterns #
###########################################
z1, y1, x1 = center_of_mass(diff_pattern)
z1, y1, x1 = [int(z1), int(y1), int(x1)]
print(
    "COM of retrieved pattern after masking: ",
    z1,
    y1,
    x1,
    " Number of unmasked photons =",
    abs(phased_fft).sum(),
)

#########################
# calculate the 3D PRTF #
#########################
diff_pattern[diff_pattern == 0] = np.nan  # discard zero valued pixels
prtf_matrix = abs(phased_fft) / np.sqrt(diff_pattern)
del phased_fft  # , diff_pattern
gc.collect()

copy_prtf = np.copy(prtf_matrix)
copy_prtf[np.isnan(copy_prtf)] = 0
piz, piy, pix = np.unravel_index(copy_prtf.argmax(), copy_prtf.shape)
print("Max(3D PRTF)=", copy_prtf.max(), " at voxel # ", (piz, piy, pix))
gu.multislices_plot(
    prtf_matrix,
    sum_frames=False,
    plot_colorbar=True,
    cmap=my_cmap,
    title="prtf_matrix",
    scale="linear",
    vmin=0,
    vmax=np.nan,
    reciprocal_space=True,
    is_orthogonal=True,
)
plt.figure()
plt.imshow(copy_prtf[piz, :, :], vmin=0)
plt.colorbar()
plt.title(
    "PRTF at max in Qx (frame "
    + str(piz)
    + ") \nMax in QyQz plane: vertical "
    + str(piy)
    + ", horizontal "
    + str(pix)
)
print(diff_pattern[piz, piy, pix])
if debug:
    copy_prtf = np.copy(prtf_matrix)
    copy_prtf[np.isnan(prtf_matrix)] = 0
    copy_prtf[copy_prtf < 5] = 0
    copy_prtf[np.nonzero(copy_prtf)] = 1

    gu.multislices_plot(
        copy_prtf,
        sum_frames=False,
        plot_colorbar=True,
        cmap=my_cmap,
        title="hotpix_prtf",
        scale="linear",
        vmin=0,
        vmax=1,
        reciprocal_space=True,
        is_orthogonal=True,
    )
del copy_prtf
gc.collect()
#################################
# average over spherical shells #
#################################
print(
    "Distance max:",
    distances_q.max(),
    "(1/nm) at: ",
    np.unravel_index(abs(distances_q).argmax(), distances_q.shape),
)
nb_bins = nz // 5
prtf_avg = np.zeros(nb_bins)
nb_points = np.zeros(nb_bins)
dq = distances_q.max() / nb_bins  # in 1/nm
q_axis = np.linspace(0, distances_q.max(), endpoint=True, num=nb_bins + 1)  # in 1/nm

for index in range(nb_bins):
    logical_array = np.logical_and(
        (distances_q < q_axis[index + 1]), (distances_q >= q_axis[index])
    )
    temp = prtf_matrix[logical_array]
    nb_points[index] = logical_array.sum()
    prtf_avg[index] = temp[~np.isnan(temp)].mean()
q_axis = q_axis[:-1]

plt.figure()
plt.plot(q_axis, nb_points, ".")
plt.xlabel("q (1/nm)")
plt.ylabel("nb of points in the average")

if q_max is None:
    q_max = q_axis.max() + 1

prtf_avg = prtf_avg[q_axis < q_max]
q_axis = q_axis[q_axis < q_max]


if normalize_prtf:
    print("Normalizing the PRTF to 1 ...")
    prtf_avg = prtf_avg / prtf_avg[~np.isnan(prtf_avg)].max()  # normalize to 1

#############################
# plot and save the 1D PRTF #
#############################
defined_q = q_axis[~np.isnan(prtf_avg)]

# create a new variable 'arc_length' to predict q and prtf parametrically
# (because prtf is not monotonic)
arc_length = np.concatenate(
    (
        np.zeros(1),
        np.cumsum(
            np.diff(prtf_avg[~np.isnan(prtf_avg)]) ** 2 + np.diff(defined_q) ** 2
        ),
    ),
    axis=0,
)  # cumulative linear arc length, used as the parameter

fit_prtf = interp1d(prtf_avg[~np.isnan(prtf_avg)], arc_length, kind="linear")
try:
    arc_length_res = fit_prtf(1 / np.e)
    fit_q = interp1d(arc_length, defined_q, kind="linear")
    q_resolution = fit_q(arc_length_res)
except ValueError:
    if (prtf_avg[~np.isnan(prtf_avg)] > 1 / np.e).all():
        print("Resolution limited by the 1 photon counts only (min(prtf)>1/e)")
        print(f"min(PRTF) = {prtf_avg[~np.isnan(prtf_avg)].min()}")
        q_resolution = defined_q.max()
    else:  # PRTF always below 1/e
        print("PRTF < 1/e for all q values, problem of normalization")
        q_resolution = np.nan

print(f"q resolution = {q_resolution:.5f} (1/nm)")
print(f"resolution d = {2*np.pi / q_resolution:.1f} nm")

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 9))
ax.plot(defined_q, prtf_avg[~np.isnan(prtf_avg)], "or")  # q_axis in 1/nm
ax.axhline(
    y=1 / np.e, linestyle="dashed", color="k", linewidth=1
)  # horizontal line at PRTF=1/e
ax.set_xlim(defined_q.min(), defined_q.max())
ax.set_ylim(0, 1.1)

gu.savefig(
    savedir=savedir,
    figure=fig,
    axes=ax,
    tick_width=2,
    tick_length=10,
    tick_labelsize=14,
    label_size=16,
    xlabels="q (1/nm)",
    ylabels="PRTF",
    filename=f"S{scan}_prtf" + comment,
    text={
        0: {"x": 0.15, "y": 0.30, "s": "Scan " + str(scan) + comment, "fontsize": 16},
        1: {
            "x": 0.15,
            "y": 0.25,
            "s": f"q at PRTF=1/e: {q_resolution:.5f} (1/nm)",
            "fontsize": 16,
        },
        2: {
            "x": 0.15,
            "y": 0.20,
            "s": f"resolution d = {2*np.pi / q_resolution:.3f} nm",
            "fontsize": 16,
        },
    },
)

plt.ioff()
plt.show()
