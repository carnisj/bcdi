#!/usr/bin/env python3

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import pathlib
import tkinter as tk
from numbers import Real
from tkinter import filedialog

import matplotlib.ticker as ticker
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.measurements import center_of_mass

import bcdi.graph.graph_utils as gu
import bcdi.utils.utilities as util
import bcdi.utils.validation as valid
from bcdi.graph.colormap import ColormapFactory

helptext = """
Template for figures of the following article:
Carnis et al. Scientific Reports 9, 17357 (2019)
https://doi.org/10.1038/s41598-019-53774-2

Open an amp_dist_strain.npz file and save individual figures.

In the reconstruction file, the following fieldnames are expected: 'amp', 'bulk',
phase' for simulated data or 'disp' for experimental data, 'strain'.

It is necessary to know the voxel size of the reconstruction in order to put ticks
at the correct position.
"""


datadir = (
    "C:/Users/Jerome/Documents/data/P10_Longfei_Nov2020/data/B10_syn_S1_00292/result/"
)
savedir = datadir + "/figures/"
comment = ""  # should start with _
simulated_data = False
# if yes, it will look for a field 'phase' in the reconstructed file,
# otherwise for field 'disp'
strain_isosurface = 0.7  # amplitude below this value will be set to 0

voxel_size = 10.0  # in nm
tick_spacing = 50  # for plots, in nm
field_of_view = (
    650  # in nm, can be larger than the total width (the array will be padded)
)

tick_direction = "in"  # 'out', 'in', 'inout'
tick_length = 10  # in plots
tick_width = 2  # in plots

strain_range = "minmax"  # 0.008
# for plots, if float it represents the half range, if 'minmax' if will use the full
# data range
phase_range = "minmax"
# for plots, if float it represents the half range, if 'minmax' if will use the full
# data range
grey_background = True  # True to set the background to grey in phase and strain plots

save_YZ = True  # True to save the view in YZ plane
save_XZ = True  # True to save the view in XZ plane
save_XY = True  # True to save the view in XY plane

flag_strain = True  # True to plot and save the strain
flag_phase = True  # True to plot and save the phase
flag_amp = True  # True to plot and save the amplitude
flag_support = False  # True to plot and save the support

amp_histogram_Yaxis = (
    "linear"  # 'log' or 'linear', Y axis scale for the amplitude histogram
)
xmin_histo = 0.02
# array values <= xmin_histo*array.max() will not be plotted to avoid the peak at 0
ylim_histo = (
    1,
    800,
)
# tuple of two numbers (ymin, ymax) for the limits of the y axis in the histogram plot
vline_hist = None  # [0.375, 0.451, 0.505, 0.541]
# list of vertical lines to plot in the amplitude histogram, leave None otherwise

flag_linecut = False  # True to plot and save a linecut of the phase
y_linecut = 257  # in pixels

background_phase = (
    np.nan
)  # value outside of the crystal, np.nan will give grey if grey_background = True
background_strain = (
    np.nan
)  # value outside of the crystal, np.nan will give grey if grey_background = True
# -2 * strain_range
##########################
# end of user parameters #
##########################

###################
# define colormap #
###################
if grey_background:
    bad_color = "0.7"
else:
    bad_color = "1.0"  # white background
my_cmap = ColormapFactory(bad_color=bad_color).cmap

#########################
# check some parameters #
#########################
if not isinstance(strain_range, Real):
    if strain_range != "minmax":
        raise ValueError(
            f"Incorrect setting {strain_range} " 'for the parameter "strain_range"'
        )
    strain_min, strain_max = -np.inf, np.inf
else:
    strain_min, strain_max = -1 * strain_range, strain_range

if not isinstance(phase_range, Real):
    if phase_range != "minmax":
        raise ValueError(
            f"Incorrect setting {phase_range} " 'for the parameter "phase_range"'
        )
    phase_min, phase_max = -np.inf, np.inf
else:
    phase_min, phase_max = -1 * phase_range, phase_range

valid.valid_item(xmin_histo, allowed_types=Real, min_included=0, name="xmin_histo")
valid.valid_container(
    obj=vline_hist,
    container_types=(list, tuple, np.ndarray, set),
    min_excluded=0,
    allow_none=True,
    item_types=Real,
    name="vline_hist",
)

if amp_histogram_Yaxis == "linear":
    valid.valid_container(
        ylim_histo,
        container_types=(tuple, list, np.ndarray),
        item_types=Real,
        length=2,
        min_included=0,
        name="ylim_histo",
    )
else:
    valid.valid_container(
        ylim_histo,
        container_types=(tuple, list, np.ndarray),
        item_types=Real,
        length=2,
        min_excluded=0,
        name="ylim_histo",
    )
if ylim_histo[1] <= ylim_histo[0]:
    raise ValueError("ylim_histo[0] should be strictly smaller than ylim_histo[1]")
pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)

#############
# load data #
#############
plt.ion()
root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename(
    initialdir=datadir, title="Select data file", filetypes=[("NPZ", "*.npz")]
)
npzfile = np.load(file_path)
strain = npzfile["strain"]
amp = npzfile["amp"]

amp = amp / amp.max()  # normalize amplitude
ref_amp = np.copy(amp)
amp[amp < strain_isosurface] = 0

if simulated_data:
    phase = npzfile["phase"]
else:
    phase = npzfile["displacement"]

numz, numy, numx = amp.shape
print("Initial data size: (", numz, ",", numy, ",", numx, ")")
comment = comment + "_iso" + str(strain_isosurface)

###################################################
#  pad arrays to obtain the desired field of view #
###################################################
pixel_spacing = tick_spacing / voxel_size
pixel_FOV = int(
    np.rint((field_of_view / voxel_size) / 2)
)  # half-number of pixels corresponding to the FOV
new_shape = [
    max(numz, 2 * pixel_FOV),
    max(numy, 2 * pixel_FOV),
    max(numx, 2 * pixel_FOV),
]
strain = util.crop_pad(array=strain, output_shape=new_shape, debugging=False)
phase = util.crop_pad(array=phase, output_shape=new_shape, debugging=False)
amp = util.crop_pad(array=amp, output_shape=new_shape, debugging=False)
numz, numy, numx = amp.shape
print("Cropped/padded data size: (", numz, ",", numy, ",", numx, ")")

######################################
# center arrays based on the support #
######################################
support = np.zeros((numz, numy, numx))
support[np.nonzero(amp)] = 1

zcom, ycom, xcom = center_of_mass(support)
zcom, ycom, xcom = [int(np.rint(zcom)), int(np.rint(ycom)), int(np.rint(xcom))]
strain = np.roll(
    strain, (numz // 2 - zcom, numy // 2 - ycom, numx // 2 - xcom), axis=(0, 1, 2)
)
phase = np.roll(
    phase, (numz // 2 - zcom, numy // 2 - ycom, numx // 2 - xcom), axis=(0, 1, 2)
)
amp = np.roll(
    amp, (numz // 2 - zcom, numy // 2 - ycom, numx // 2 - xcom), axis=(0, 1, 2)
)

################################################
# assign default values outside of the crystal #
################################################
support[25, 24:26, 35] = 1
support[19:27, 25, 32:36] = 1
gu.multislices_plot(
    support, sum_frames=False, is_orthogonal=True, reciprocal_space=False
)
strain[support == 0] = background_strain
phase[support == 0] = background_phase

###########
# Support #
###########
if flag_support:
    fig, ax0 = plt.subplots(1, 1)
    ax0.imshow(
        support[
            numz // 2 - pixel_FOV : numz // 2 + pixel_FOV,
            numy // 2 - pixel_FOV : numy // 2 + pixel_FOV,
            numx // 2,
        ],
        vmin=0,
        vmax=1,
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
    if save_YZ:
        fig.savefig(savedir + "support_YZ" + comment + ".png", bbox_inches="tight")

    fig, ax1 = plt.subplots(1, 1)
    ax1.imshow(
        support[
            numz // 2 - pixel_FOV : numz // 2 + pixel_FOV,
            numy // 2,
            numx // 2 - pixel_FOV : numx // 2 + pixel_FOV,
        ],
        vmin=0,
        vmax=1,
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
    if save_XZ:
        fig.savefig(savedir + "support_XZ" + comment + ".png", bbox_inches="tight")

    fig, ax2 = plt.subplots(1, 1)
    ax2.imshow(
        support[
            numz // 2,
            numy // 2 - pixel_FOV : numy // 2 + pixel_FOV,
            numx // 2 - pixel_FOV : numx // 2 + pixel_FOV,
        ],
        vmin=0,
        vmax=1,
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

    if save_XY:
        fig.savefig(savedir + "support_XY" + comment + ".png", bbox_inches="tight")

#############
# Amplitude #
#############
if flag_amp:
    fig, ax0 = plt.subplots(1, 1)
    plt0 = ax0.imshow(
        ref_amp[
            numz // 2 - pixel_FOV : numz // 2 + pixel_FOV,
            numy // 2 - pixel_FOV : numy // 2 + pixel_FOV,
            numx // 2,
        ],
        vmin=0,
        vmax=1,
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
    if save_YZ:
        fig.savefig(savedir + "amp_YZ" + comment + ".png", bbox_inches="tight")
        plt.colorbar(plt0, ax=ax0)
        fig.savefig(savedir + "amp_YZ" + comment + "_colorbar.png", bbox_inches="tight")

    fig, ax1 = plt.subplots(1, 1)
    plt1 = ax1.imshow(
        ref_amp[
            numz // 2 - pixel_FOV : numz // 2 + pixel_FOV,
            numy // 2,
            numx // 2 - pixel_FOV : numx // 2 + pixel_FOV,
        ],
        vmin=0,
        vmax=1,
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
    if save_XZ:
        fig.savefig(savedir + "amp_XZ" + comment + ".png", bbox_inches="tight")
        plt.colorbar(plt1, ax=ax1)
        fig.savefig(savedir + "amp_XZ" + comment + "_colorbar.png", bbox_inches="tight")

    fig, ax2 = plt.subplots(1, 1)
    plt2 = ax2.imshow(
        ref_amp[
            numz // 2,
            numy // 2 - pixel_FOV : numy // 2 + pixel_FOV,
            numx // 2 - pixel_FOV : numx // 2 + pixel_FOV,
        ],
        vmin=0,
        vmax=1,
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

    if save_XY:
        fig.savefig(savedir + "amp_XY" + comment + ".png", bbox_inches="tight")
        plt.colorbar(plt2, ax=ax2)
        fig.savefig(savedir + "amp_XY" + comment + "_colorbar.png", bbox_inches="tight")

    fig, ax = plt.subplots(1, 1)
    ax.hist(ref_amp[ref_amp > xmin_histo * ref_amp.max()].flatten(), bins=250)
    # avoid the peak for very low noise amplitudes
    ax.set_xlim(left=xmin_histo)
    if amp_histogram_Yaxis == "log":
        ax.set_yscale("log")
    ax.set_ylim(bottom=ylim_histo[0], top=ylim_histo[1])
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
    if vline_hist is not None:
        for line in vline_hist:
            ax.axvline(
                x=line, linestyle=(0, (1, 300)), color="k", linewidth=0.5
            )  # vertical line
    fig.savefig(
        savedir + "phased_histogram_amp" + comment + ".png", bbox_inches="tight"
    )
    ax.tick_params(
        labelbottom=True,
        labelleft=True,
        direction="out",
        length=tick_length,
        width=tick_width,
    )
    if vline_hist is not None:
        ax.set_title(f"vlines: {vline_hist}", size=12)
    fig.savefig(
        savedir + "phased_histogram_amp" + comment + "_labels.png", bbox_inches="tight"
    )

##########
# Strain #
##########
if flag_strain:
    fig, ax0 = plt.subplots(1, 1)
    if not isinstance(strain_range, Real):
        tmp_strain = strain[
            numz // 2 - pixel_FOV : numz // 2 + pixel_FOV,
            numy // 2 - pixel_FOV : numy // 2 + pixel_FOV,
            numx // 2,
        ]
        strain_min, strain_max = (
            tmp_strain[~np.isnan(tmp_strain)].min(),
            tmp_strain[~np.isnan(tmp_strain)].max(),
        )
    plt0 = ax0.imshow(
        strain[
            numz // 2 - pixel_FOV : numz // 2 + pixel_FOV,
            numy // 2 - pixel_FOV : numy // 2 + pixel_FOV,
            numx // 2,
        ],
        vmin=strain_min,
        vmax=strain_max,
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
    if save_YZ:
        fig.savefig(savedir + "strain_YZ" + comment + ".png", bbox_inches="tight")
        plt.colorbar(plt0, ax=ax0)
        fig.savefig(
            savedir + "strain_YZ" + comment + "_colorbar.png", bbox_inches="tight"
        )
    fig, ax1 = plt.subplots(1, 1)
    if not isinstance(strain_range, Real):
        tmp_strain = strain[
            numz // 2 - pixel_FOV : numz // 2 + pixel_FOV,
            numy // 2,
            numx // 2 - pixel_FOV : numx // 2 + pixel_FOV,
        ]
        strain_min, strain_max = (
            tmp_strain[~np.isnan(tmp_strain)].min(),
            tmp_strain[~np.isnan(tmp_strain)].max(),
        )
    plt1 = ax1.imshow(
        strain[
            numz // 2 - pixel_FOV : numz // 2 + pixel_FOV,
            numy // 2,
            numx // 2 - pixel_FOV : numx // 2 + pixel_FOV,
        ],
        vmin=strain_min,
        vmax=strain_max,
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
    if save_XZ:
        fig.savefig(savedir + "strain_XZ" + comment + ".png", bbox_inches="tight")
        plt.colorbar(plt1, ax=ax1)
        fig.savefig(
            savedir + "strain_XZ" + comment + "_colorbar.png", bbox_inches="tight"
        )

    fig, ax2 = plt.subplots(1, 1)
    if not isinstance(strain_range, Real):
        tmp_strain = strain[
            numz // 2,
            numy // 2 - pixel_FOV : numy // 2 + pixel_FOV,
            numx // 2 - pixel_FOV : numx // 2 + pixel_FOV,
        ]
        strain_min, strain_max = (
            tmp_strain[~np.isnan(tmp_strain)].min(),
            tmp_strain[~np.isnan(tmp_strain)].max(),
        )
    plt2 = ax2.imshow(
        strain[
            numz // 2,
            numy // 2 - pixel_FOV : numy // 2 + pixel_FOV,
            numx // 2 - pixel_FOV : numx // 2 + pixel_FOV,
        ],
        vmin=strain_min,
        vmax=strain_max,
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

    if save_XY:
        fig.savefig(savedir + "strain_XY" + comment + ".png", bbox_inches="tight")
        plt.colorbar(plt2, ax=ax2)
        fig.savefig(
            savedir + "strain_XY" + comment + "_colorbar.png", bbox_inches="tight"
        )

#########
# Phase #
#########
if flag_phase:
    fig, ax0 = plt.subplots(1, 1)
    if not isinstance(phase_range, Real):
        tmp_phase = phase[
            numz // 2 - pixel_FOV : numz // 2 + pixel_FOV,
            numy // 2 - pixel_FOV : numy // 2 + pixel_FOV,
            numx // 2,
        ]
        phase_min, phase_max = (
            tmp_phase[~np.isnan(tmp_phase)].min(),
            tmp_phase[~np.isnan(tmp_phase)].max(),
        )
    plt0 = ax0.imshow(
        phase[
            numz // 2 - pixel_FOV : numz // 2 + pixel_FOV,
            numy // 2 - pixel_FOV : numy // 2 + pixel_FOV,
            numx // 2,
        ],
        vmin=phase_min,
        vmax=phase_max,
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
    if save_YZ:
        fig.savefig(savedir + "phase_YZ" + comment + ".png", bbox_inches="tight")
        plt.colorbar(plt0, ax=ax0)
        fig.savefig(
            savedir + "phase_YZ" + comment + "_colorbar.png", bbox_inches="tight"
        )

    fig, ax1 = plt.subplots(1, 1)
    if not isinstance(phase_range, Real):
        tmp_phase = phase[
            numz // 2 - pixel_FOV : numz // 2 + pixel_FOV,
            numy // 2,
            numx // 2 - pixel_FOV : numx // 2 + pixel_FOV,
        ]
        phase_min, phase_max = (
            tmp_phase[~np.isnan(tmp_phase)].min(),
            tmp_phase[~np.isnan(tmp_phase)].max(),
        )
    plt1 = ax1.imshow(
        phase[
            numz // 2 - pixel_FOV : numz // 2 + pixel_FOV,
            numy // 2,
            numx // 2 - pixel_FOV : numx // 2 + pixel_FOV,
        ],
        vmin=phase_min,
        vmax=phase_max,
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
    if save_XZ:
        fig.savefig(savedir + "phase_XZ" + comment + ".png", bbox_inches="tight")
        plt.colorbar(plt1, ax=ax1)
        fig.savefig(
            savedir + "phase_XZ" + comment + "_colorbar.png", bbox_inches="tight"
        )

    fig, ax2 = plt.subplots(1, 1)
    if not isinstance(phase_range, Real):
        tmp_phase = phase[
            numz // 2,
            numy // 2 - pixel_FOV : numy // 2 + pixel_FOV,
            numx // 2 - pixel_FOV : numx // 2 + pixel_FOV,
        ]
        phase_min, phase_max = (
            tmp_phase[~np.isnan(tmp_phase)].min(),
            tmp_phase[~np.isnan(tmp_phase)].max(),
        )
    plt2 = ax2.imshow(
        phase[
            numz // 2,
            numy // 2 - pixel_FOV : numy // 2 + pixel_FOV,
            numx // 2 - pixel_FOV : numx // 2 + pixel_FOV,
        ],
        vmin=phase_min,
        vmax=phase_max,
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

    if save_XY:
        fig.savefig(savedir + "phase_XY" + comment + ".png", bbox_inches="tight")
        plt.colorbar(plt2, ax=ax2)
        fig.savefig(
            savedir + "phase_XY" + comment + "_colorbar.png", bbox_inches="tight"
        )

    ############################################################
    # example of a line cut on the phase,
    # can also load more data for the lineplot for comparison  #
    ############################################################
    if flag_linecut:
        # file_path = filedialog.askopenfilename(initialdir=datadir,
        #                                        title="Select avg7 file",
        #                                        filetypes=[("NPZ", "*.npz")])
        # npzfile = np.load(file_path)
        # phase2 = npzfile['displacement']
        # bulk = npzfile['bulk']
        # if phase2.shape != phase.shape:
        #     print('array2 shape not compatible')
        #     sys.exit()
        # phase2[bulk == 0] = np.nan
        #
        # file_path = filedialog.askopenfilename(initialdir=datadir,
        #  title="Select post_processing apodization file",
        #                                        filetypes=[("NPZ", "*.npz")])
        # npzfile = np.load(file_path)
        # phase3 = npzfile['displacement']
        # bulk = npzfile['bulk']
        # if phase3.shape != phase.shape:
        #     print('array3 shape not compatible')
        #     sys.exit()
        # phase3[bulk == 0] = np.nan
        #
        # file_path = filedialog.askopenfilename(initialdir=datadir,
        # title="Select pre_processingapodization file",
        #                                        filetypes=[("NPZ", "*.npz")])
        # npzfile = np.load(file_path)
        # phase4 = npzfile['displacement']
        # bulk = npzfile['bulk']
        # if phase3.shape != phase.shape:
        #     print('array4 shape not compatible')
        #     sys.exit()
        # phase4[bulk == 0] = np.nan

        fig, ax0 = plt.subplots(1, 1)
        plt0 = ax0.imshow(
            phase[numz // 2, :, :], vmin=-phase_min, vmax=phase_max, cmap=my_cmap
        )

        fig, ax3 = plt.subplots(1, 1)
        plt.plot(phase[numz // 2, y_linecut, :], "r", linestyle="-")  # (1, (4, 4)))  #
        # plt.plot(phase2[numz // 2, y_linecut, :], 'k', linestyle='-.')
        # , marker='D', fillstyle='none'
        # plt.plot(phase3[numz // 2, y_linecut, :], 'b', linestyle=':')
        # , marker='^', fillstyle='none'
        # plt.plot(phase4[numz // 2, y_linecut, :], 'g', linestyle='--')
        # , marker='^', fillstyle='none'
        ax3.spines["right"].set_linewidth(1)
        ax3.spines["left"].set_linewidth(1)
        ax3.spines["top"].set_linewidth(1)
        ax3.spines["bottom"].set_linewidth(1)
        ax3.set_xlim(155, 255)
        ax3.set_ylim(-np.pi, np.pi)
        ax3.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax3.tick_params(
            labelbottom=False,
            labelleft=False,
            top=False,
            bottom=True,
            direction="inout",
            length=tick_length,
            width=tick_width,
        )
        fig.savefig(
            savedir + "Linecut_phase_X_Y=" + str(y_linecut) + comment + ".png",
            bbox_inches="tight",
        )

plt.ioff()
plt.show()
