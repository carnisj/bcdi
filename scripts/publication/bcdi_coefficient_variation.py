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
from scipy.ndimage.measurements import center_of_mass

import bcdi.graph.graph_utils as gu
import bcdi.postprocessing.postprocessing_utils as pu
import bcdi.utils.utilities as util
from bcdi.graph.colormap import ColormapFactory
from bcdi.utils import image_registration as reg

helptext = """
Load several reconstructed complex objects and calculate the coefficient variation
(CV = std/mean) on the modulus, the phase or the strain.

In the reconstruction file, the following fieldnames are expected: 'amp', 'bulk',
'phase' for simulated data or 'disp' for experimental data, 'strain'.

It is necessary to know the voxel size of the reconstruction in order to put ticks at
the correct position. Laboratory frame: z downstream, y vertical, x outboard (CXI
convention)
"""


scans = [1301, 1304]  # spec scan number
rootfolder = "D:/data/SIXS_2019_Ni/"
savedir = "D:/data/SIXS_2019_Ni/comparison_S1301_S1304/"
comment = ""  # should start with _

voxel_size = 9.74  # in nm
planar_distance = 0.39242 / np.sqrt(
    3
)  # crystallographic interplanar distance, in nm (for strain calculation)
ref_axis_q = "z"  # axis along which q is aligned (for strain calculation)
# 'z', 'y' or 'x' in CXI convention
isosurface = 0.30  # threshold use for removing the outer layer
# (strain is undefined at the exact surface voxel)
isosurface_method = "threshold"  # 'threshold' or 'defect',
# for 'defect' it tries to remove only outer layers even if
# the amplitude is low inside the crystal

tick_spacing = 100  # for plots, in nm
field_of_view = (
    900  # in nm, can be larger than the total width (the array will be padded)
)

tick_direction = "in"  # 'out', 'in', 'inout'
tick_length = 10  # in plots
tick_width = 2  # in plots

strain_range = 1  # for coefficient of variation plots,
# the colorbar will be in the range [-strain_range, strain_range]
phase_range = (
    np.pi
)  # for coefficient of variation plots, the colorbar will be in the range
# [-phase_range, phase_range]
grey_background = True  # True to set the background to grey in phase and strain plots
background_color = (
    np.nan
)  # value outside of the crystal, np.nan will give grey if grey_background = True

save_YZ = True  # True to save the view in YZ plane
save_XZ = True  # True to save the view in XZ plane
save_XY = True  # True to save the view in XY plane

flag_strain = True  # True to plot and save the strain
flag_phase = True  # True to plot and save the strain
flag_amp = True  # True to plot and save the amplitude

center_object = (
    True  # if True, will center the first object based on the COM of its support
)
# all other objects will be aligned on the first one
debug = False  # True to see more plots for debugging
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

#############
# load data #
#############
plt.ion()
root = tk.Tk()
root.withdraw()

nb_scans = len(scans)
fieldnames = ["scans", "modulus", "phase", "strain"]
datasets = {new_list: [] for new_list in fieldnames}  # create dictionnary
datasets["scans"].append(scans)

for index, item in enumerate(scans):
    file_path = filedialog.askopenfilename(
        initialdir=rootfolder,
        title="Select amp-disp-strain file for S" + str(item),
        filetypes=[("NPZ", "*.npz")],
    )
    npzfile = np.load(file_path)

    amp = npzfile["amp"]
    phase = npzfile["displacement"]

    numz, numy, numx = amp.shape

    # pad the dataset to the desired the field #
    pixel_spacing = tick_spacing / voxel_size
    pixel_FOV = int(
        np.rint((field_of_view / voxel_size) / 2)
    )  # half-number of pixels corresponding to the FOV

    #  pad arrays to obtain the desired field of view
    new_shape = [
        max(numz, 2 * pixel_FOV),
        max(numy, 2 * pixel_FOV),
        max(numx, 2 * pixel_FOV),
    ]
    amp = util.crop_pad(array=amp, output_shape=new_shape, debugging=False)
    phase = util.crop_pad(array=phase, output_shape=new_shape, debugging=False)
    numz, numy, numx = amp.shape

    if debug:
        gu.multislices_plot(
            amp,
            sum_frames=False,
            title="Input modulus",
            vmin=0,
            tick_direction=tick_direction,
            tick_width=tick_width,
            tick_length=tick_length,
            pixel_spacing=pixel_spacing,
            plot_colorbar=True,
            is_orthogonal=True,
            reciprocal_space=False,
        )
        gu.multislices_plot(
            phase,
            sum_frames=False,
            title="Input phase",
            vmin=-phase_range,
            vmax=phase_range,
            tick_direction=tick_direction,
            cmap=my_cmap,
            tick_width=tick_width,
            tick_length=tick_length,
            pixel_spacing=pixel_spacing,
            plot_colorbar=True,
            is_orthogonal=True,
            reciprocal_space=False,
        )
    if index == 0:
        # center the first object if needed
        if center_object is True:
            support = np.zeros(amp.shape)
            support[amp > isosurface * amp.max()] = 1

            zcom, ycom, xcom = center_of_mass(support)
            zcom, ycom, xcom = [
                int(np.rint(zcom)),
                int(np.rint(ycom)),
                int(np.rint(xcom)),
            ]
            obj = amp * np.exp(1j * phase)
            amp = np.roll(
                amp,
                (numz // 2 - zcom, numy // 2 - ycom, numx // 2 - xcom),
                axis=(0, 1, 2),
            )
            phase = np.roll(
                phase,
                (numz // 2 - zcom, numy // 2 - ycom, numx // 2 - xcom),
                axis=(0, 1, 2),
            )

        support = np.zeros(amp.shape)
        support[amp > isosurface * amp.max()] = 1
        zcom, ycom, xcom = center_of_mass(support)
        zcom, ycom, xcom = [
            int(np.rint(zcom)),
            int(np.rint(ycom)),
            int(np.rint(xcom)),
        ]  # used for the reference phase
        phase, extent_phase = pu.unwrap(
            amp * np.exp(1j * phase), support_threshold=0.05, debugging=debug
        )
        phase = util.wrap(
            phase, start_angle=-extent_phase / 2, range_angle=extent_phase
        )
        phase = phase - phase[zcom, ycom, xcom]
        phase, extent_phase = pu.unwrap(
            amp * np.exp(1j * phase), support_threshold=0.05, debugging=debug
        )
        phase = util.wrap(
            phase, start_angle=-extent_phase / 2, range_angle=extent_phase
        )  # rewrap after modifying phase
        ref_amp = np.copy(amp)
    else:  # align it with the reference object
        shiftz, shifty, shiftx = reg.getimageregistration(ref_amp, amp, precision=1000)
        print(
            "Shift of array", index, "with the reference array:", shiftz, shifty, shiftx
        )
        obj = amp * np.exp(1j * phase)
        obj = reg.subpixel_shift(obj, shiftz, shifty, shiftx)
        amp = abs(obj)
        phase = np.angle(obj)
        phase, extent_phase = pu.unwrap(
            amp * np.exp(1j * phase), support_threshold=0.05, debugging=False
        )
        phase = util.wrap(
            phase, start_angle=-extent_phase / 2, range_angle=extent_phase
        )
        phase = phase - phase[zcom, ycom, xcom]
        phase, extent_phase = pu.unwrap(
            amp * np.exp(1j * phase), support_threshold=0.05, debugging=False
        )
        phase = util.wrap(
            phase, start_angle=-extent_phase / 2, range_angle=extent_phase
        )  # rewrap after modifying phase
    print(
        "Array",
        str(index),
        ":  Extent of the phase over an extended support (ceil(phase range))~ ",
        int(extent_phase),
        "(rad)",
    )
    print(
        "Array",
        str(index),
        ":  Phase = ",
        phase[zcom, ycom, xcom],
        "at (zcom, ycom, xcom)=",
        zcom,
        ycom,
        xcom,
    )
    if debug:
        gu.multislices_plot(
            amp,
            sum_frames=False,
            title="Amp after matching",
            vmin=0,
            tick_direction=tick_direction,
            tick_width=tick_width,
            tick_length=tick_length,
            pixel_spacing=pixel_spacing,
            plot_colorbar=True,
            is_orthogonal=True,
            reciprocal_space=False,
        )
        gu.multislices_plot(
            phase,
            sum_frames=False,
            title="Phase after matching",
            vmin=-phase_range,
            vmax=phase_range,
            tick_direction=tick_direction,
            cmap=my_cmap,
            tick_width=tick_width,
            tick_length=tick_length,
            pixel_spacing=pixel_spacing,
            plot_colorbar=True,
            is_orthogonal=True,
            reciprocal_space=False,
        )

    datasets["modulus"].append(amp)
    datasets["phase"].append(phase)

    strain = pu.get_strain(
        phase=phase,
        planar_distance=planar_distance,
        voxel_size=voxel_size,
        reference_axis=ref_axis_q,
    )
    datasets["strain"].append(strain)

    bulk = pu.find_bulk(amp=amp, support_threshold=isosurface, method=isosurface_method)

    np.savez_compressed(
        savedir + "S" + str(item) + "_amp_disp_strain_matched" + comment,
        amp=amp,
        displacement=phase,
        bulk=bulk,
        strain=strain,
    )

numz, numy, numx = ref_amp.shape

#############
# Amplitude #
#############
if flag_amp:
    amp_concat = np.zeros((nb_scans, ref_amp.size))
    for idx in range(nb_scans):
        amp_concat[idx, :] = datasets["modulus"][idx].flatten()

    CV_amp = np.divide(np.std(amp_concat, axis=0), np.mean(amp_concat, axis=0)).reshape(
        ref_amp.shape
    )
    # do not apply amplitude threshold, in order to see if the sizes are different
    fig, ax0 = plt.subplots(1, 1)
    plt0 = ax0.imshow(
        CV_amp[
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
        fig.savefig(savedir + "CV_amp_YZ" + comment + ".png", bbox_inches="tight")

    fig, ax1 = plt.subplots(1, 1)
    plt1 = ax1.imshow(
        CV_amp[
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
        fig.savefig(savedir + "CV_amp_XZ" + comment + ".png", bbox_inches="tight")

    fig, ax2 = plt.subplots(1, 1)
    plt2 = ax2.imshow(
        CV_amp[
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
        fig.savefig(savedir + "CV_amp_XY" + comment + ".png", bbox_inches="tight")
    plt.colorbar(plt2, ax=ax2)
    fig.savefig(savedir + "CV_amp_XY" + comment + "_colorbar.png", bbox_inches="tight")

#########
# Phase #
#########
if flag_phase:
    phase_concat = np.zeros((nb_scans, ref_amp.size))
    for idx in range(nb_scans):
        phase_concat[idx, :] = datasets["phase"][idx].flatten()
        # do not apply amplitude threshold before CV calculation,
        # it creates artefacts in the coefficient of variation

    CV_phase = np.divide(
        np.std(phase_concat, axis=0), np.mean(phase_concat, axis=0)
    ).reshape(ref_amp.shape)
    CV_phase[ref_amp < isosurface * ref_amp.max()] = background_color

    fig, ax0 = plt.subplots(1, 1)
    plt0 = ax0.imshow(
        CV_phase[
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
    if save_YZ:
        fig.savefig(savedir + "CV_phase_YZ" + comment + ".png", bbox_inches="tight")

    fig, ax1 = plt.subplots(1, 1)
    plt1 = ax1.imshow(
        CV_phase[
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
    if save_XZ:
        fig.savefig(savedir + "CV_phase_XZ" + comment + ".png", bbox_inches="tight")

    fig, ax2 = plt.subplots(1, 1)
    plt2 = ax2.imshow(
        CV_phase[
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

    if save_XY:
        fig.savefig(savedir + "CV_phase_XY" + comment + ".png", bbox_inches="tight")
    plt.colorbar(plt2, ax=ax2)
    fig.savefig(
        savedir + "CV_phase_XY" + comment + "_colorbar.png", bbox_inches="tight"
    )

    ###################
    # difference maps #
    ###################
    if nb_scans == 2:
        diff_phase = (phase_concat[1, :] - phase_concat[0, :]).reshape(ref_amp.shape)
        diff_phase[ref_amp < isosurface * ref_amp.max()] = background_color

        temp_diff = diff_phase[
            numz // 2 - pixel_FOV : numz // 2 + pixel_FOV,
            numy // 2 - pixel_FOV : numy // 2 + pixel_FOV,
            numx // 2,
        ]
        min_diff, max_diff = (
            temp_diff[~np.isnan(temp_diff)].min(),
            temp_diff[~np.isnan(temp_diff)].max(),
        )
        fig, ax0 = plt.subplots(1, 1)
        plt0 = ax0.imshow(temp_diff, vmin=min_diff, vmax=max_diff, cmap=my_cmap)
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
            fig.savefig(
                savedir + "diff_phase_YZ" + comment + ".png", bbox_inches="tight"
            )
        plt.colorbar(plt0, ax=ax0)
        fig.savefig(
            savedir + "diff_phase_YZ" + comment + "_colorbar.png", bbox_inches="tight"
        )

        temp_diff = diff_phase[
            numz // 2 - pixel_FOV : numz // 2 + pixel_FOV,
            numy // 2,
            numx // 2 - pixel_FOV : numx // 2 + pixel_FOV,
        ]
        min_diff, max_diff = (
            temp_diff[~np.isnan(temp_diff)].min(),
            temp_diff[~np.isnan(temp_diff)].max(),
        )
        fig, ax1 = plt.subplots(1, 1)
        plt1 = ax1.imshow(temp_diff, vmin=min_diff, vmax=max_diff, cmap=my_cmap)
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
            fig.savefig(
                savedir + "diff_phase_XZ" + comment + ".png", bbox_inches="tight"
            )
        plt.colorbar(plt1, ax=ax1)
        fig.savefig(
            savedir + "diff_phase_XZ" + comment + "_colorbar.png", bbox_inches="tight"
        )

        temp_diff = diff_phase[
            numz // 2,
            numy // 2 - pixel_FOV : numy // 2 + pixel_FOV,
            numx // 2 - pixel_FOV : numx // 2 + pixel_FOV,
        ]
        min_diff, max_diff = (
            temp_diff[~np.isnan(temp_diff)].min(),
            temp_diff[~np.isnan(temp_diff)].max(),
        )
        fig, ax2 = plt.subplots(1, 1)
        plt2 = ax2.imshow(temp_diff, vmin=min_diff, vmax=max_diff, cmap=my_cmap)
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
            fig.savefig(
                savedir + "diff_phase_XY" + comment + ".png", bbox_inches="tight"
            )
        plt.colorbar(plt2, ax=ax2)
        fig.savefig(
            savedir + "diff_phase_XY" + comment + "_colorbar.png", bbox_inches="tight"
        )

##########
# Strain #
##########
if flag_strain:
    strain_concat = np.zeros((nb_scans, ref_amp.size))
    for idx in range(nb_scans):
        strain_concat[idx, :] = datasets["strain"][idx].flatten()
        # do not apply amplitude threshold before CV calculation,
        # it creates artefacts in the coefficient of variation

    CV_strain = np.divide(
        np.std(strain_concat, axis=0), np.mean(strain_concat, axis=0)
    ).reshape(ref_amp.shape)
    CV_strain[ref_amp < isosurface * ref_amp.max()] = background_color

    fig, ax0 = plt.subplots(1, 1)
    plt0 = ax0.imshow(
        CV_strain[
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
    if save_YZ:
        fig.savefig(savedir + "CV_strain_YZ" + comment + ".png", bbox_inches="tight")

    fig, ax1 = plt.subplots(1, 1)
    plt1 = ax1.imshow(
        CV_strain[
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
    if save_XZ:
        fig.savefig(savedir + "CV_strain_XZ" + comment + ".png", bbox_inches="tight")

    fig, ax2 = plt.subplots(1, 1)
    plt2 = ax2.imshow(
        CV_strain[
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
    if save_XY:
        fig.savefig(savedir + "CV_strain_XY" + comment + ".png", bbox_inches="tight")
    plt.colorbar(plt2, ax=ax2)
    fig.savefig(
        savedir + "CV_strain_XY" + comment + "_colorbar.png", bbox_inches="tight"
    )

    ###################
    # difference maps #
    ###################
    if nb_scans == 2:
        diff_strain = (strain_concat[1, :] - strain_concat[0, :]).reshape(ref_amp.shape)
        diff_strain[ref_amp < isosurface * ref_amp.max()] = background_color

        temp_diff = diff_strain[
            numz // 2 - pixel_FOV : numz // 2 + pixel_FOV,
            numy // 2 - pixel_FOV : numy // 2 + pixel_FOV,
            numx // 2,
        ]
        min_diff, max_diff = (
            temp_diff[~np.isnan(temp_diff)].min(),
            temp_diff[~np.isnan(temp_diff)].max(),
        )
        fig, ax0 = plt.subplots(1, 1)
        plt0 = ax0.imshow(temp_diff, vmin=min_diff, vmax=max_diff, cmap=my_cmap)
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
            fig.savefig(
                savedir + "diff_strain_YZ" + comment + ".png", bbox_inches="tight"
            )
        plt.colorbar(plt0, ax=ax0)
        fig.savefig(
            savedir + "diff_strain_YZ" + comment + "_colorbar.png", bbox_inches="tight"
        )

        temp_diff = diff_strain[
            numz // 2 - pixel_FOV : numz // 2 + pixel_FOV,
            numy // 2,
            numx // 2 - pixel_FOV : numx // 2 + pixel_FOV,
        ]
        min_diff, max_diff = (
            temp_diff[~np.isnan(temp_diff)].min(),
            temp_diff[~np.isnan(temp_diff)].max(),
        )
        fig, ax1 = plt.subplots(1, 1)
        plt1 = ax1.imshow(temp_diff, vmin=min_diff, vmax=max_diff, cmap=my_cmap)
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
            fig.savefig(
                savedir + "diff_strain_XZ" + comment + ".png", bbox_inches="tight"
            )
        plt.colorbar(plt1, ax=ax1)
        fig.savefig(
            savedir + "diff_strain_XZ" + comment + "_colorbar.png", bbox_inches="tight"
        )

        temp_diff = diff_strain[
            numz // 2,
            numy // 2 - pixel_FOV : numy // 2 + pixel_FOV,
            numx // 2 - pixel_FOV : numx // 2 + pixel_FOV,
        ]
        min_diff, max_diff = (
            temp_diff[~np.isnan(temp_diff)].min(),
            temp_diff[~np.isnan(temp_diff)].max(),
        )
        fig, ax2 = plt.subplots(1, 1)
        plt2 = ax2.imshow(temp_diff, vmin=min_diff, vmax=max_diff, cmap=my_cmap)
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
            fig.savefig(
                savedir + "diff_strain_XY" + comment + ".png", bbox_inches="tight"
            )
        plt.colorbar(plt2, ax=ax2)
        fig.savefig(
            savedir + "diff_strain_XY" + comment + "_colorbar.png", bbox_inches="tight"
        )

plt.ioff()
plt.show()
