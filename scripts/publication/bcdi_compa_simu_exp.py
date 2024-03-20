#!/usr/bin/env python3

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import tkinter as tk
from tkinter import filedialog

import matplotlib.ticker as ticker
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import interp1d
from scipy.signal import convolve

from bcdi.utils import image_registration as reg

helptext = """
Template for figures of the following article:
Carnis et al. Scientific Reports 9, 17357 (2019)
https://doi.org/10.1038/s41598-019-53774-2
Open the simulated amp_dist_strain.npz file and the reconstructed
amp_dist_strain.npz, and plot difference maps
"""


scan = 2227  # spec scan number
datadir = (
    "G:/review paper/BCDI_isosurface/S"
    + str(scan)
    + "/simu/crop400phase/no_apodization/avg1/"
)
savedir = (
    "G:/review paper/BCDI_isosurface/New figures/isosurface/no_apodization/avg1_new/"
)
voxel_size = 3.0  # in nm
tick_spacing = 50  # for plots, in nm
planar_dist = 0.2269735  # in nm, for strain calculation
field_of_view = (
    500  # in nm, should not be larger than the total width at the moment (no padding)
)
tick_direction = "in"  # 'out', 'in', 'inout'
tick_length = 6  # in plots
tick_width = 2  # in plots
strain_range = 0.002  # for plots
phase_range = np.pi  # for plots
support_threshold = 0.7  # threshold for support determination
min_amp = 0.01  # everything with lower amplitude will be set to np.nan in plots
debug = 0  # 1 to show all plots
save_YZ = 0  # 1 to save the strain in YZ plane
save_XZ = 1  # 1 to save the strain in XZ plane
save_XY = 1  # 1 to save the strain in XY plane
comment = "_iso" + str(support_threshold)  # should start with _
comment = comment + "_strainrange_" + str(strain_range)
######################################
# define a colormap
cdict = {
    "red": (
        (0.0, 1.0, 1.0),
        (0.11, 0.0, 0.0),
        (0.36, 0.0, 0.0),
        (0.62, 1.0, 1.0),
        (0.87, 1.0, 1.0),
        (1.0, 0.0, 0.0),
    ),
    "green": (
        (0.0, 1.0, 1.0),
        (0.11, 0.0, 0.0),
        (0.36, 1.0, 1.0),
        (0.62, 1.0, 1.0),
        (0.87, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    ),
    "blue": (
        (0.0, 1.0, 1.0),
        (0.11, 1.0, 1.0),
        (0.36, 1.0, 1.0),
        (0.62, 0.0, 0.0),
        (0.87, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    ),
}
my_cmap = LinearSegmentedColormap("my_colormap", cdict, 256)
my_cmap.set_bad(color="0.7")


def calc_coordination(mysupport, debugging=0):
    """Calculate the coordination number of the support using a 3x3x3 kernel."""
    nbz, nby, nbx = mysupport.shape

    mykernel = np.ones((3, 3, 3))
    mycoord = np.rint(convolve(mysupport, mykernel, mode="same"))
    mycoord = mycoord.astype(int)

    if debugging == 1:
        plt.figure(figsize=(18, 15))
        plt.subplot(2, 2, 1)
        plt.imshow(mycoord[:, :, nbx // 2])
        plt.colorbar()
        plt.axis("scaled")
        plt.title("Coordination matrix in middle slice in YZ")
        plt.subplot(2, 2, 2)
        plt.imshow(mycoord[:, nby // 2, :])
        plt.colorbar()
        plt.title("Coordination matrix in middle slice in XZ")
        plt.axis("scaled")
        plt.subplot(2, 2, 3)
        plt.imshow(mycoord[nbz // 2, :, :])
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.title("Coordination matrix in middle slice in XY")
        plt.axis("scaled")
        plt.pause(0.1)
    return mycoord


def crop_pad(myobj, myshape, debugging=0):
    """
    Crop or pad my obj depending on myshape.

    :param myobj: 3d complex array to be padded
    :param myshape: list of desired output shape [z, y, x]
    :param debugging: to plot myobj before and after rotation
    :return: myobj padded with zeros
    """
    nbz, nby, nbx = myobj.shape
    newz, newy, newx = myshape
    if debugging == 1:
        plt.figure(figsize=(18, 15))
        plt.subplot(2, 2, 1)
        plt.imshow(abs(myobj)[:, :, nbx // 2], vmin=0, vmax=1)
        plt.colorbar()
        plt.axis("scaled")
        plt.title("Middle slice in YZ before padding")
        plt.subplot(2, 2, 2)
        plt.imshow(abs(myobj)[:, nby // 2, :], vmin=0, vmax=1)
        plt.colorbar()
        plt.title("Middle slice in XZ before padding")
        plt.axis("scaled")
        plt.subplot(2, 2, 3)
        plt.imshow(abs(myobj)[nbz // 2, :, :], vmin=0, vmax=1)
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.title("Middle slice in XY before padding")
        plt.axis("scaled")
        plt.pause(0.1)
    # z
    if newz >= nbz:  # pad
        temp_z = np.zeros((myshape[0], nby, nbx), dtype=myobj.dtype)
        temp_z[(newz - nbz) // 2 : (newz + nbz) // 2, :, :] = myobj
    else:  # crop
        temp_z = myobj[(nbz - newz) // 2 : (newz + nbz) // 2, :, :]
    # y
    if newy >= nby:  # pad
        temp_y = np.zeros((newz, newy, nbx), dtype=myobj.dtype)
        temp_y[:, (newy - nby) // 2 : (newy + nby) // 2, :] = temp_z
    else:  # crop
        temp_y = temp_z[:, (nby - newy) // 2 : (newy + nby) // 2, :]
    # x
    if newx >= nbx:  # pad
        newobj = np.zeros((newz, newy, newx), dtype=myobj.dtype)
        newobj[:, :, (newx - nbx) // 2 : (newx + nbx) // 2] = temp_y
    else:  # crop
        newobj = temp_y[:, :, (nbx - newx) // 2 : (newx + nbx) // 2]

    if debugging == 1:
        plt.figure(figsize=(18, 15))
        plt.subplot(2, 2, 1)
        plt.imshow(abs(newobj)[:, :, newx // 2], vmin=0, vmax=1)
        plt.colorbar()
        plt.axis("scaled")
        plt.title("Middle slice in YZ after padding")
        plt.subplot(2, 2, 2)
        plt.imshow(abs(newobj)[:, newy // 2, :], vmin=0, vmax=1)
        plt.colorbar()
        plt.title("Middle slice in XZ after padding")
        plt.axis("scaled")
        plt.subplot(2, 2, 3)
        plt.imshow(abs(newobj)[newz // 2, :, :], vmin=0, vmax=1)
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.title("Middle slice in XY after padding")
        plt.axis("scaled")
        plt.pause(0.1)
    return newobj


plt.ion()
root = tk.Tk()
root.withdraw()
#######################################
pixel_spacing = tick_spacing / voxel_size
pixel_FOV = int(
    np.rint((field_of_view / voxel_size) / 2)
)  # half-number of pixels corresponding to the FOV

##########################
# open simulated amp_phase_strain.npz
##########################
file_path = filedialog.askopenfilename(
    initialdir=datadir, title="Select simulation file", filetypes=[("NPZ", "*.npz")]
)
print("Opening ", file_path)
npzfile = np.load(file_path)
amp_simu = npzfile["amp"]
bulk_simu = npzfile["bulk"]
strain_simu = npzfile["strain"]
phase_simu = npzfile["phase"]  # ['displacement']
numz, numy, numx = amp_simu.shape
print("SIMU: Initial data size: (", numz, ",", numy, ",", numx, ")")
strain_simu[amp_simu == 0] = np.nan
phase_simu[amp_simu == 0] = np.nan

##########################
# open phased amp_phase_strain.npz
##########################
file_path = filedialog.askopenfilename(
    initialdir=datadir, title="Select phased file", filetypes=[("NPZ", "*.npz")]
)
print("Opening ", file_path)
npzfile = np.load(file_path)
amp = npzfile["amp"]
phase = npzfile["phase"]
obj = amp * np.exp(1j * phase)
del amp, phase
numz, numy, numx = obj.shape
print("Phased: Initial data size: (", numz, ",", numy, ",", numx, ")")
obj = crop_pad(obj, amp_simu.shape)
numz, numy, numx = obj.shape
print("Cropped/padded size: (", numz, ",", numy, ",", numx, ")")
plt.figure()
plt.imshow(
    np.angle(obj)[numz // 2, :, :], cmap=my_cmap, vmin=-phase_range, vmax=phase_range
)
plt.title("Phase before subpixel shift")
plt.pause(0.1)

##############################
# align datasets
##############################
# dft registration and subpixel shift (see Matlab code)
shiftz, shifty, shiftx = reg.getimageregistration(amp_simu, abs(obj), precision=1000)
obj = reg.subpixel_shift(obj, shiftz, shifty, shiftx)
print(
    "Shift calculated from dft registration: (",
    str(f"{shiftz:.2f}"),
    ",",
    str(f"{shifty:.2f}"),
    ",",
    str(f"{shiftx:.2f}"),
    ") pixels",
)
new_amp = abs(obj)
new_phase = np.angle(obj)
del obj
_, new_strain, _ = np.gradient(planar_dist / (2 * np.pi) * new_phase, voxel_size)
# q is along y after rotating the crystal
plt.figure()
plt.imshow(
    new_phase[numz // 2, :, :], cmap=my_cmap, vmin=-phase_range, vmax=phase_range
)
plt.title("Phase after subpixel shift")
plt.pause(0.1)
del new_phase
plt.figure()
plt.imshow(
    new_strain[numz // 2, :, :], cmap=my_cmap, vmin=-strain_range, vmax=strain_range
)
plt.title("Strain after subpixel shift")
plt.pause(0.1)
new_amp = (
    new_amp / new_amp.max()
)  # need to renormalize after subpixel shift interpolation

new_amp[new_amp < min_amp] = 0
new_strain[new_amp == 0] = 0

##############################
# plot simulated phase
##############################
masked_array = np.ma.array(phase_simu, mask=np.isnan(phase_simu))
fig, ax0 = plt.subplots(1, 1)
plt0 = ax0.imshow(
    masked_array[
        numz // 2 - pixel_FOV : numz // 2 + pixel_FOV,
        numy // 2 - pixel_FOV : numy // 2 + pixel_FOV,
        numx // 2,
    ],
    vmin=-phase_range,
    vmax=phase_range,
    cmap=my_cmap,
)
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
if save_YZ == 1:
    plt.savefig(savedir + "simu_phase_YZ.png", bbox_inches="tight")

fig, ax1 = plt.subplots(1, 1)
plt1 = ax1.imshow(
    masked_array[
        numz // 2 - pixel_FOV : numz // 2 + pixel_FOV,
        numy // 2,
        numx // 2 - pixel_FOV : numx // 2 + pixel_FOV,
    ],
    vmin=-phase_range,
    vmax=phase_range,
    cmap=my_cmap,
)
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
if save_XZ == 1:
    plt.savefig(savedir + "simu_phase_XZ.png", bbox_inches="tight")

fig, ax2 = plt.subplots(1, 1)
plt2 = ax2.imshow(
    masked_array[
        numz // 2,
        numy // 2 - pixel_FOV : numy // 2 + pixel_FOV,
        numx // 2 - pixel_FOV : numx // 2 + pixel_FOV,
    ],
    vmin=-phase_range,
    vmax=phase_range,
    cmap=my_cmap,
)
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

if save_XY == 1:
    plt.savefig(savedir + "simu_phase_XY.png", bbox_inches="tight")
plt.colorbar(plt2, ax=ax2)
plt.savefig(savedir + "simu_phase_XY_colorbar.png", bbox_inches="tight")

##############################
# plot amplitudes
##############################
hist, bin_edges = np.histogram(new_amp[new_amp > min_amp].flatten(), bins=250)
bin_step = (bin_edges[1] - bin_edges[0]) / 2
bin_axis = bin_edges + bin_step
bin_axis = bin_axis[0 : len(hist)]
# hist = medfilt(hist, kernel_size=3)

interpolation = interp1d(bin_axis, hist, kind="cubic")
interp_points = 1 * len(hist)
interp_axis = np.linspace(bin_axis.min(), bin_axis.max(), interp_points)
inter_step = interp_axis[1] - interp_axis[0]
interp_curve = interpolation(interp_axis)


fig, ax = plt.subplots(1, 1)
plt.hist(new_amp[new_amp > min_amp].flatten(), bins=250)
plt.xlim(left=min_amp)
plt.ylim(bottom=1)
ax.set_yscale("log")
ax.tick_params(
    labelbottom=False,
    labelleft=False,
    direction="out",
    length=tick_length,
    width=tick_width,
)
plt.savefig(savedir + "phased_histogram_amp.png", bbox_inches="tight")
ax.tick_params(
    labelbottom=True,
    labelleft=True,
    direction="out",
    length=tick_length,
    width=tick_width,
)
ax.spines["right"].set_linewidth(1.5)
ax.spines["left"].set_linewidth(1.5)
ax.spines["top"].set_linewidth(1.5)
ax.spines["bottom"].set_linewidth(1.5)
plt.savefig(savedir + "phased_histogram_amp_labels.png", bbox_inches="tight")


fig, ax0 = plt.subplots(1, 1)
plt.plot(amp_simu[numz // 2, 183, 128:136], "r")
plt.plot(new_amp[numz // 2, 183, 128:136], "k")
ax0.tick_params(
    labelbottom=False,
    labelleft=False,
    direction="out",
    top=True,
    bottom=False,
    length=tick_length,
    width=tick_width,
)
ax0.spines["right"].set_linewidth(1.5)
ax0.spines["left"].set_linewidth(1.5)
ax0.spines["top"].set_linewidth(1.5)
ax0.spines["bottom"].set_linewidth(1.5)
plt.savefig(savedir + "linecut_amp.png", bbox_inches="tight")

if debug:
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)
    plt0 = ax0.imshow(
        new_amp[
            numz // 2 - pixel_FOV : numz // 2 + pixel_FOV,
            numy // 2 - pixel_FOV : numy // 2 + pixel_FOV,
            numx // 2,
        ],
        cmap=my_cmap,
        vmin=0,
        vmax=1,
    )
    plt1 = ax1.imshow(
        new_amp[
            numz // 2 - pixel_FOV : numz // 2 + pixel_FOV,
            numy // 2,
            numx // 2 - pixel_FOV : numx // 2 + pixel_FOV,
        ],
        cmap=my_cmap,
        vmin=0,
        vmax=1,
    )
    plt2 = ax2.imshow(
        new_amp[
            numz // 2,
            numy // 2 - pixel_FOV : numy // 2 + pixel_FOV,
            numx // 2 - pixel_FOV : numx // 2 + pixel_FOV,
        ],
        cmap=my_cmap,
        vmin=0,
        vmax=1,
    )
    ax2.invert_yaxis()
    plt.title("new_amp")

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(
        2,
        2,
    )
    plt0 = ax0.imshow(
        amp_simu[
            numz // 2 - pixel_FOV : numz // 2 + pixel_FOV,
            numy // 2 - pixel_FOV : numy // 2 + pixel_FOV,
            numx // 2,
        ],
        cmap=my_cmap,
        vmin=0,
        vmax=1,
    )
    plt1 = ax1.imshow(
        amp_simu[
            numz // 2 - pixel_FOV : numz // 2 + pixel_FOV,
            numy // 2,
            numx // 2 - pixel_FOV : numx // 2 + pixel_FOV,
        ],
        cmap=my_cmap,
        vmin=0,
        vmax=1,
    )
    plt2 = ax2.imshow(
        amp_simu[
            numz // 2,
            numy // 2 - pixel_FOV : numy // 2 + pixel_FOV,
            numx // 2 - pixel_FOV : numx // 2 + pixel_FOV,
        ],
        cmap=my_cmap,
        vmin=0,
        vmax=1,
    )
    ax2.invert_yaxis()
    plt.title("amp_simu")

diff_amp = (amp_simu - new_amp) * 100
diff_amp_copy = np.copy(diff_amp)
support = np.zeros(amp_simu.shape)

support[np.nonzero(amp_simu)] = 1
support[np.nonzero(new_amp)] = 1
# the support will have the size of the largest object
# between the simulation and the reconstruction
diff_amp_copy[support == 0] = np.nan
masked_array = np.ma.array(diff_amp_copy, mask=np.isnan(diff_amp_copy))
if debug:
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)
    plt0 = ax0.imshow(
        masked_array[
            numz // 2 - pixel_FOV : numz // 2 + pixel_FOV,
            numy // 2 - pixel_FOV : numy // 2 + pixel_FOV,
            numx // 2,
        ],
        cmap=my_cmap,
        vmin=-100,
        vmax=100,
    )
    plt.colorbar(plt0, ax=ax0)
    plt1 = ax1.imshow(
        masked_array[
            numz // 2 - pixel_FOV : numz // 2 + pixel_FOV,
            numy // 2,
            numx // 2 - pixel_FOV : numx // 2 + pixel_FOV,
        ],
        cmap=my_cmap,
        vmin=-100,
        vmax=100,
    )
    plt.colorbar(plt1, ax=ax1)
    plt2 = ax2.imshow(
        masked_array[
            numz // 2,
            numy // 2 - pixel_FOV : numy // 2 + pixel_FOV,
            numx // 2 - pixel_FOV : numx // 2 + pixel_FOV,
        ],
        cmap=my_cmap,
        vmin=-100,
        vmax=100,
    )
    ax2.invert_yaxis()
    plt.colorbar(plt2, ax=ax2)
    plt.title("(amp_simu - new_amp)*100")

fig, ax0 = plt.subplots(1, 1)
plt0 = ax0.imshow(
    masked_array[
        numz // 2 - pixel_FOV : numz // 2 + pixel_FOV,
        numy // 2 - pixel_FOV : numy // 2 + pixel_FOV,
        numx // 2,
    ],
    vmin=-100,
    vmax=100,
    cmap=my_cmap,
)
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
if save_YZ == 1:
    plt.savefig(savedir + "diff_amp_YZ.png", bbox_inches="tight")

fig, ax1 = plt.subplots(1, 1)
plt1 = ax1.imshow(
    masked_array[
        numz // 2 - pixel_FOV : numz // 2 + pixel_FOV,
        numy // 2,
        numx // 2 - pixel_FOV : numx // 2 + pixel_FOV,
    ],
    vmin=-100,
    vmax=100,
    cmap=my_cmap,
)
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
if save_XZ == 1:
    plt.savefig(savedir + "diff_amp_XZ.png", bbox_inches="tight")

fig, ax2 = plt.subplots(1, 1)
plt2 = ax2.imshow(
    masked_array[
        numz // 2,
        numy // 2 - pixel_FOV : numy // 2 + pixel_FOV,
        numx // 2 - pixel_FOV : numx // 2 + pixel_FOV,
    ],
    vmin=-100,
    vmax=100,
    cmap=my_cmap,
)
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

if save_XY == 1:
    plt.savefig(savedir + "diff_amp_XY.png", bbox_inches="tight")
plt.colorbar(plt2, ax=ax2)
plt.savefig(savedir + "diff_amp_XY_colorbar.png", bbox_inches="tight")

del diff_amp_copy
support[amp_simu == 0] = 0  # redefine the support as the simulated object

##############################
# plot individual strain maps
##############################
new_strain_copy = np.copy(new_strain)
new_strain_copy[new_amp == 0] = np.nan
masked_array = np.ma.array(new_strain_copy, mask=np.isnan(new_strain_copy))

if debug:
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)
    plt0 = ax0.imshow(
        masked_array[
            numz // 2 - pixel_FOV : numz // 2 + pixel_FOV,
            numy // 2 - pixel_FOV : numy // 2 + pixel_FOV,
            numx // 2,
        ],
        cmap=my_cmap,
        vmin=-strain_range,
        vmax=strain_range,
    )
    plt.colorbar(plt0, ax=ax0)
    plt1 = ax1.imshow(
        masked_array[
            numz // 2 - pixel_FOV : numz // 2 + pixel_FOV,
            numy // 2,
            numx // 2 - pixel_FOV : numx // 2 + pixel_FOV,
        ],
        cmap=my_cmap,
        vmin=-strain_range,
        vmax=strain_range,
    )
    plt.colorbar(plt1, ax=ax1)
    plt2 = ax2.imshow(
        masked_array[
            numz // 2,
            numy // 2 - pixel_FOV : numy // 2 + pixel_FOV,
            numx // 2 - pixel_FOV : numx // 2 + pixel_FOV,
        ],
        cmap=my_cmap,
        vmin=-strain_range,
        vmax=strain_range,
    )
    ax2.invert_yaxis()
    plt.colorbar(plt2, ax=ax2)
    plt.title("new_strain")

fig, ax0 = plt.subplots(1, 1)
plt0 = ax0.imshow(
    masked_array[
        numz // 2 - pixel_FOV : numz // 2 + pixel_FOV,
        numy // 2 - pixel_FOV : numy // 2 + pixel_FOV,
        numx // 2,
    ],
    vmin=-strain_range,
    vmax=strain_range,
    cmap=my_cmap,
)
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
if save_YZ == 1:
    plt.savefig(savedir + "phased_strain_YZ" + comment + ".png", bbox_inches="tight")

fig, ax1 = plt.subplots(1, 1)
plt1 = ax1.imshow(
    masked_array[
        numz // 2 - pixel_FOV : numz // 2 + pixel_FOV,
        numy // 2,
        numx // 2 - pixel_FOV : numx // 2 + pixel_FOV,
    ],
    vmin=-strain_range,
    vmax=strain_range,
    cmap=my_cmap,
)
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
if save_XZ == 1:
    plt.savefig(savedir + "phased_strain_XZ" + comment + ".png", bbox_inches="tight")

fig, ax2 = plt.subplots(1, 1)
plt2 = ax2.imshow(
    masked_array[
        numz // 2,
        numy // 2 - pixel_FOV : numy // 2 + pixel_FOV,
        numx // 2 - pixel_FOV : numx // 2 + pixel_FOV,
    ],
    vmin=-strain_range,
    vmax=strain_range,
    cmap=my_cmap,
)
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

if save_XY == 1:
    plt.savefig(savedir + "phased_strain_XY" + comment + ".png", bbox_inches="tight")
plt.colorbar(plt2, ax=ax2)
plt.savefig(
    savedir + "phased_strain_XY" + comment + "_colorbar.png", bbox_inches="tight"
)

del new_strain_copy

strain_simu[bulk_simu == 0] = (
    np.nan
)  # remove the non-physical outer layer for simulated strain
masked_array = np.ma.array(strain_simu, mask=np.isnan(strain_simu))

fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)
plt0 = ax0.imshow(
    masked_array[
        numz // 2 - pixel_FOV : numz // 2 + pixel_FOV,
        numy // 2 - pixel_FOV : numy // 2 + pixel_FOV,
        numx // 2,
    ],
    cmap=my_cmap,
    vmin=-strain_range,
    vmax=strain_range,
)
plt.colorbar(plt0, ax=ax0)
plt1 = ax1.imshow(
    masked_array[
        numz // 2 - pixel_FOV : numz // 2 + pixel_FOV,
        numy // 2,
        numx // 2 - pixel_FOV : numx // 2 + pixel_FOV,
    ],
    cmap=my_cmap,
    vmin=-strain_range,
    vmax=strain_range,
)
plt.colorbar(plt1, ax=ax1)
plt2 = ax2.imshow(
    masked_array[
        numz // 2,
        numy // 2 - pixel_FOV : numy // 2 + pixel_FOV,
        numx // 2 - pixel_FOV : numx // 2 + pixel_FOV,
    ],
    cmap=my_cmap,
    vmin=-strain_range,
    vmax=strain_range,
)
plt.colorbar(plt2, ax=ax2)
ax2.invert_yaxis()
plt.title("strain_simu")

fig, ax0 = plt.subplots(1, 1)
plt0 = ax0.imshow(
    masked_array[
        numz // 2 - pixel_FOV : numz // 2 + pixel_FOV,
        numy // 2 - pixel_FOV : numy // 2 + pixel_FOV,
        numx // 2,
    ],
    vmin=-strain_range,
    vmax=strain_range,
    cmap=my_cmap,
)
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
if save_YZ == 1:
    plt.savefig(savedir + "simu_strain_YZ" + comment + ".png", bbox_inches="tight")

fig, ax1 = plt.subplots(1, 1)
plt1 = ax1.imshow(
    masked_array[
        numz // 2 - pixel_FOV : numz // 2 + pixel_FOV,
        numy // 2,
        numx // 2 - pixel_FOV : numx // 2 + pixel_FOV,
    ],
    vmin=-strain_range,
    vmax=strain_range,
    cmap=my_cmap,
)
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
if save_XZ == 1:
    plt.savefig(savedir + "simu_strain_XZ" + comment + ".png", bbox_inches="tight")

fig, ax2 = plt.subplots(1, 1)
plt2 = ax2.imshow(
    masked_array[
        numz // 2,
        numy // 2 - pixel_FOV : numy // 2 + pixel_FOV,
        numx // 2 - pixel_FOV : numx // 2 + pixel_FOV,
    ],
    vmin=-strain_range,
    vmax=strain_range,
    cmap=my_cmap,
)
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

if save_XY == 1:
    plt.savefig(savedir + "simu_strain_XY" + comment + ".png", bbox_inches="tight")
plt.colorbar(plt2, ax=ax2)
plt.savefig(savedir + "simu_strain_XY" + comment + "_colorbar.png", bbox_inches="tight")

##############################
# plot difference strain maps
##############################
diff_strain = strain_simu - new_strain
diff_strain[support == 0] = (
    np.nan
)  # the support is 0 outside of the simulated object, strain is not defined there
masked_array = np.ma.array(diff_strain, mask=np.isnan(diff_strain))
if debug:
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)
    plt0 = ax0.imshow(
        masked_array[
            numz // 2 - pixel_FOV : numz // 2 + pixel_FOV,
            numy // 2 - pixel_FOV : numy // 2 + pixel_FOV,
            numx // 2,
        ],
        cmap=my_cmap,
        vmin=-strain_range,
        vmax=strain_range,
    )
    plt.colorbar(plt0, ax=ax0)
    plt1 = ax1.imshow(
        masked_array[
            numz // 2 - pixel_FOV : numz // 2 + pixel_FOV,
            numy // 2,
            numx // 2 - pixel_FOV : numx // 2 + pixel_FOV,
        ],
        cmap=my_cmap,
        vmin=-strain_range,
        vmax=strain_range,
    )
    plt.colorbar(plt1, ax=ax1)
    plt2 = ax2.imshow(
        masked_array[
            numz // 2,
            numy // 2 - pixel_FOV : numy // 2 + pixel_FOV,
            numx // 2 - pixel_FOV : numx // 2 + pixel_FOV,
        ],
        cmap=my_cmap,
        vmin=-strain_range,
        vmax=strain_range,
    )
    ax2.invert_yaxis()
    plt.colorbar(plt2, ax=ax2)
    plt.title("(strain_simu - new_strain) on full data")

phased_support = np.ones(amp_simu.shape)
phased_support[new_amp < support_threshold] = 0
if debug:
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)
    plt0 = ax0.imshow(
        phased_support[
            numz // 2 - pixel_FOV : numz // 2 + pixel_FOV,
            numy // 2 - pixel_FOV : numy // 2 + pixel_FOV,
            numx // 2,
        ]
    )
    plt.colorbar(plt0, ax=ax0)
    plt1 = ax1.imshow(
        phased_support[
            numz // 2 - pixel_FOV : numz // 2 + pixel_FOV,
            numy // 2,
            numx // 2 - pixel_FOV : numx // 2 + pixel_FOV,
        ]
    )
    plt.colorbar(plt1, ax=ax1)
    plt2 = ax2.imshow(
        phased_support[
            numz // 2,
            numy // 2 - pixel_FOV : numy // 2 + pixel_FOV,
            numx // 2 - pixel_FOV : numx // 2 + pixel_FOV,
        ]
    )
    ax2.invert_yaxis()
    plt.colorbar(plt2, ax=ax2)
    plt.title("Phased support")

diff_strain[phased_support == 0] = (
    np.nan
)  # exclude also layers outside of the isosurface for the reconstruction
masked_array = np.ma.array(diff_strain, mask=np.isnan(diff_strain))

if debug:
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)
    plt0 = ax0.imshow(
        masked_array[
            numz // 2 - pixel_FOV : numz // 2 + pixel_FOV,
            numy // 2 - pixel_FOV : numy // 2 + pixel_FOV,
            numx // 2,
        ],
        cmap=my_cmap,
        vmin=-strain_range,
        vmax=strain_range,
    )
    plt.colorbar(plt0, ax=ax0)
    plt1 = ax1.imshow(
        masked_array[
            numz // 2 - pixel_FOV : numz // 2 + pixel_FOV,
            numy // 2,
            numx // 2 - pixel_FOV : numx // 2 + pixel_FOV,
        ],
        cmap=my_cmap,
        vmin=-strain_range,
        vmax=strain_range,
    )
    plt.colorbar(plt1, ax=ax1)
    plt2 = ax2.imshow(
        masked_array[
            numz // 2,
            numy // 2 - pixel_FOV : numy // 2 + pixel_FOV,
            numx // 2 - pixel_FOV : numx // 2 + pixel_FOV,
        ],
        cmap=my_cmap,
        vmin=-strain_range,
        vmax=strain_range,
    )
    ax2.invert_yaxis()
    plt.colorbar(plt2, ax=ax2)
    plt.title("(strain_simu - new_strain) with isosurface " + str(support_threshold))

fig, ax0 = plt.subplots(1, 1)
plt0 = ax0.imshow(
    masked_array[
        numz // 2 - pixel_FOV : numz // 2 + pixel_FOV,
        numy // 2 - pixel_FOV : numy // 2 + pixel_FOV,
        numx // 2,
    ],
    vmin=-strain_range,
    vmax=strain_range,
    cmap=my_cmap,
)
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
if save_YZ == 1:
    plt.savefig(savedir + "diff_strain_YZ" + comment + ".png", bbox_inches="tight")

fig, ax1 = plt.subplots(1, 1)
plt1 = ax1.imshow(
    masked_array[
        numz // 2 - pixel_FOV : numz // 2 + pixel_FOV,
        numy // 2,
        numx // 2 - pixel_FOV : numx // 2 + pixel_FOV,
    ],
    vmin=-strain_range,
    vmax=strain_range,
    cmap=my_cmap,
)
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
if save_XZ == 1:
    plt.savefig(savedir + "diff_strain_XZ" + comment + ".png", bbox_inches="tight")

fig, ax2 = plt.subplots(1, 1)
plt2 = ax2.imshow(
    masked_array[
        numz // 2,
        numy // 2 - pixel_FOV : numy // 2 + pixel_FOV,
        numx // 2 - pixel_FOV : numx // 2 + pixel_FOV,
    ],
    vmin=-strain_range,
    vmax=strain_range,
    cmap=my_cmap,
)
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

if save_XY == 1:
    plt.savefig(savedir + "diff_strain_XY" + comment + ".png", bbox_inches="tight")
plt.colorbar(plt2, ax=ax2)
plt.savefig(savedir + "diff_strain_XY" + comment + "_colorbar.png", bbox_inches="tight")

coordination_matrix = calc_coordination(
    phased_support, debugging=0
)  # the surface is defined for the reconstruction
surface = np.copy(phased_support)
surface[coordination_matrix > 22] = 0  # remove the bulk 22
bulk = phased_support - surface
bulk[np.nonzero(bulk)] = 1

plt.figure()
plt.subplot(4, 3, 1)
plt.imshow(
    surface[
        numz // 2 - pixel_FOV : numz // 2 + pixel_FOV,
        numy // 2 - pixel_FOV : numy // 2 + pixel_FOV,
        numx // 2,
    ]
)
plt.colorbar()
plt.axis("scaled")
plt.title("Surface matrix in middle slice in YZ")
plt.subplot(4, 3, 2)
plt.imshow(
    surface[
        numz // 2 - pixel_FOV : numz // 2 + pixel_FOV,
        numy // 2,
        numx // 2 - pixel_FOV : numx // 2 + pixel_FOV,
    ]
)
plt.colorbar()
plt.title("Surface matrix in middle slice in XZ")
plt.axis("scaled")
plt.subplot(4, 3, 3)
plt.imshow(
    surface[
        numz // 2,
        numy // 2 - pixel_FOV : numy // 2 + pixel_FOV,
        numx // 2 - pixel_FOV : numx // 2 + pixel_FOV,
    ]
)
plt.gca().invert_yaxis()
plt.colorbar()
plt.title("Surface matrix in middle slice in XY")
plt.axis("scaled")

surface[surface == 0] = np.nan
print("Total number of points in surface = ", (~np.isnan(surface)).sum())
rms_strain = np.sqrt(np.mean(np.ndarray.flatten(surface[~np.isnan(surface)]) ** 2))

surface = np.multiply(surface, diff_strain)
bulk[bulk == 0] = np.nan
bulk = np.multiply(bulk, diff_strain)

plt.subplot(4, 3, 4)
plt.imshow(
    surface[
        numz // 2 - pixel_FOV : numz // 2 + pixel_FOV,
        numy // 2 - pixel_FOV : numy // 2 + pixel_FOV,
        numx // 2,
    ],
    vmin=-strain_range,
    vmax=strain_range,
    cmap=my_cmap,
)
plt.colorbar()
plt.axis("scaled")
plt.title("Surface difference strain in middle slice in YZ")
plt.subplot(4, 3, 5)
plt.imshow(
    surface[
        numz // 2 - pixel_FOV : numz // 2 + pixel_FOV,
        numy // 2,
        numx // 2 - pixel_FOV : numx // 2 + pixel_FOV,
    ],
    vmin=-strain_range,
    vmax=strain_range,
    cmap=my_cmap,
)
plt.colorbar()
plt.title("Surface difference strain in middle slice in XZ")
plt.axis("scaled")
plt.subplot(4, 3, 6)
plt.imshow(
    surface[
        numz // 2,
        numy // 2 - pixel_FOV : numy // 2 + pixel_FOV,
        numx // 2 - pixel_FOV : numx // 2 + pixel_FOV,
    ],
    vmin=-strain_range,
    vmax=strain_range,
    cmap=my_cmap,
)
plt.gca().invert_yaxis()
plt.colorbar()
plt.title("Surface difference strain in middle slice in XY")
plt.axis("scaled")

plt.subplot(4, 3, 7)
plt.imshow(
    bulk[
        numz // 2 - pixel_FOV : numz // 2 + pixel_FOV,
        numy // 2 - pixel_FOV : numy // 2 + pixel_FOV,
        numx // 2,
    ],
    vmin=-strain_range,
    vmax=strain_range,
    cmap=my_cmap,
)
plt.colorbar()
plt.axis("scaled")
plt.title("Bulk difference strain in middle slice in YZ")
plt.subplot(4, 3, 8)
plt.imshow(
    bulk[
        numz // 2 - pixel_FOV : numz // 2 + pixel_FOV,
        numy // 2,
        numx // 2 - pixel_FOV : numx // 2 + pixel_FOV,
    ],
    vmin=-strain_range,
    vmax=strain_range,
    cmap=my_cmap,
)
plt.colorbar()
plt.title("Bulk difference strain in middle slice in XZ")
plt.axis("scaled")
plt.subplot(4, 3, 9)
plt.imshow(
    bulk[
        numz // 2,
        numy // 2 - pixel_FOV : numy // 2 + pixel_FOV,
        numx // 2 - pixel_FOV : numx // 2 + pixel_FOV,
    ],
    vmin=-strain_range,
    vmax=strain_range,
    cmap=my_cmap,
)
plt.gca().invert_yaxis()
plt.colorbar()
plt.title("Bulk difference strain in middle slice in XY")
plt.axis("scaled")

plt.subplot(4, 3, 10)
plt.imshow(
    support[
        numz // 2 - pixel_FOV : numz // 2 + pixel_FOV,
        numy // 2 - pixel_FOV : numy // 2 + pixel_FOV,
        numx // 2,
    ]
)
plt.colorbar()
plt.axis("scaled")
plt.title("Simulation support in middle slice in YZ")
plt.subplot(4, 3, 11)
plt.imshow(
    support[
        numz // 2 - pixel_FOV : numz // 2 + pixel_FOV,
        numy // 2,
        numx // 2 - pixel_FOV : numx // 2 + pixel_FOV,
    ]
)
plt.colorbar()
plt.title("Simulation support in middle slice in XZ")
plt.axis("scaled")
plt.subplot(4, 3, 12)
plt.imshow(
    support[
        numz // 2,
        numy // 2 - pixel_FOV : numy // 2 + pixel_FOV,
        numx // 2 - pixel_FOV : numx // 2 + pixel_FOV,
    ]
)
plt.gca().invert_yaxis()
plt.colorbar()
plt.title("Simulation support in middle slice in XY")
plt.axis("scaled")

print("Number of defined points in surface = ", (~np.isnan(surface)).sum())
rms_strain = np.sqrt(np.mean(np.ndarray.flatten(surface[~np.isnan(surface)]) ** 2))
print("RMS of the difference in surface strain = ", str(f"{rms_strain:.4e}"))
rms_strain = np.sqrt(np.mean(np.ndarray.flatten(bulk[~np.isnan(bulk)]) ** 2))
print("RMS of the difference in bulk strain = ", str(f"{rms_strain:.4e}"))
mean_strain = np.mean(np.ndarray.flatten(surface[~np.isnan(surface)]))
print("Mean difference in surface strain = ", str(f"{mean_strain:.4e}"))
mean_strain = np.mean(np.ndarray.flatten(bulk[~np.isnan(bulk)]))
print("Mean difference in bulk strain = ", str(f"{mean_strain:.4e}"))
plt.ioff()
plt.show()
