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

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fftn, fftshift

import bcdi.graph.graph_utils as gu
import bcdi.utils.validation as valid
from bcdi.graph.colormap import ColormapFactory

helptext = """
Open and plot the point-spread function (PSF) from a .cxi reconstruction file (from
PyNX). The PSF or the mutual coherence function (FFT of the PSF) can be to be plotted.
"""

datadir = (
    "D:/data/P10_2nd_test_isosurface_Dec2020/data_nanolab/dataset_2_pearson97.5_newpsf/"
)
savedir = datadir + "psf_Run0020/"
is_orthogonal = False  # True if the data was orthogonalized before phasing
comment = ""  # should start with _
width = 30
# integer or tuple of three integers (one for each dimension), the psf will be plotted
# for +/- this number of pixels from center of the array.
# Leave None to use the full array
vmin = -6  # min of the colorbar for plots (log scale). Use np.nan for default.
vmax = 0  # max of the colorbar for plots (log scale). Use np.nan for default.
plot_mcf = False  # True to plot the Fourier transform of the PSF
save_slices = True  # True to save individual 2D slices (in z, y, x)
tick_direction = "out"  # 'out', 'in', 'inout'
tick_length = 8  # in plots
tick_width = 2  # in plots
linewidth = 2  # linewidth for the plot frame
###################
# define colormap #
###################
bad_color = "1.0"  # white background
my_cmap = ColormapFactory(bad_color=bad_color).cmap
mpl.rcParams["axes.linewidth"] = tick_width  # set the linewidth globally

#########################
# check some parameters #
#########################
valid_name = "bcdi_view_psf"
savedir = savedir or datadir
pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)
if width is not None and isinstance(width, Real):
    width = (width,) * 3
valid.valid_container(
    width,
    container_types=(tuple, list, np.ndarray),
    length=3,
    item_types=int,
    min_excluded=0,
    allow_none=True,
    name=valid_name,
)
valid.valid_item(vmin, allowed_types=Real, name=valid_name)
valid.valid_item(vmax, allowed_types=Real, name=valid_name)
valid.valid_item(tick_length, allowed_types=int, min_excluded=0, name=valid_name)
valid.valid_item(tick_width, allowed_types=int, min_excluded=0, name=valid_name)
valid.valid_item(linewidth, allowed_types=int, min_excluded=0, name=valid_name)
valid.valid_item(plot_mcf, allowed_types=bool, name=valid_name)
valid.valid_item(save_slices, allowed_types=bool, name=valid_name)
valid.valid_item(is_orthogonal, allowed_types=bool, name=valid_name)
if tick_direction not in {"out", "in", "inout"}:
    raise ValueError("allowed values for tick_direction: 'out', 'in', 'inout'")

if is_orthogonal:
    title = "log(psf) in laboratory frame"
else:
    title = "log(psf) in detector frame"

valid.valid_container(comment, container_types=str, name=valid_name)
if comment and not comment.startswith("_"):
    comment = "_" + comment

#####################################################
# load the CXI file, output of PyNX phase retrieval #
#####################################################
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(initialdir=datadir, filetypes=[("CXI", "*.cxi")])

h5file = h5py.File(file_path, "r")
try:
    if plot_mcf:
        dataset = fftshift(
            fftn(
                h5file[
                    "/entry_1/image_1/instrument_1/detector_1/point_spread_function"
                ].value
            )
        )
        fname = "mcf"
        plot_scale = "linear"
    else:
        dataset = h5file[
            "/entry_1/image_1/instrument_1/detector_1/point_spread_function"
        ].value
        fname = "psf"
        plot_scale = "log"
except KeyError as ex:
    print("The PSF was not saved in the CXI file")
    raise KeyError from ex

# normalize the psf to 1
dataset = abs(dataset) / abs(dataset).max()

###############################################################
# check if the plotting region of interest matches data shape #
###############################################################
nbz, nby, nbx = np.shape(dataset)
print(f"psf shape = {dataset.shape}")
cen_z, cen_y, cen_x = nbz // 2, nby // 2, nbx // 2
if width is None:
    width = (cen_z, cen_y, cen_x)
if any(
    (
        cen_z - width[0] < 0,
        cen_z + width[0] > nbz,
        cen_y - width[1] < 0,
        cen_y + width[1] > nby,
        cen_x - width[2] < 0,
        cen_x + width[2] > nbx,
    )
):
    print("width is not compatible with the psf shape")

    width = (min(cen_z, cen_y, cen_x),) * 3

#########################
# plot and save the psf #
#########################
fig, _, _ = gu.multislices_plot(
    dataset,
    scale=plot_scale,
    sum_frames=False,
    title=title,
    vmin=vmin,
    vmax=vmax,
    reciprocal_space=False,
    is_orthogonal=is_orthogonal,
    plot_colorbar=True,
    width_z=2 * width[0],
    width_y=2 * width[1],
    width_x=2 * width[2],
)
fig.savefig(savedir + fname + "_centralslice" + comment + ".png")

if save_slices:
    if is_orthogonal:  # orthogonal laboratory frame, CXI convention:
        # z downstream, y vertical up, x outboard
        labels = (
            ("x", "y", "z"),  # labels for x axis, y axis, title
            ("x", "z", "y"),
            ("y", "z", "x"),
        )
    else:  # non-orthogonal detector frame stacking axis,
        # detector vertical Y down, detector horizontal X inboard
        labels = (
            (
                "detector X",
                "detector Y",
                "stacking axis",
            ),  # labels for x axis, y axis, title
            ("detector X", "stacking axis", "detector Y"),
            ("detector Y", "stacking axis", "detector X"),
        )

    fig, ax, _ = gu.imshow_plot(
        dataset[cen_z, :, :],
        sum_frames=False,
        width_v=2 * width[1],
        width_h=2 * width[2],
        scale=plot_scale,
        vmin=vmin,
        vmax=vmax,
        reciprocal_space=False,
        is_orthogonal=is_orthogonal,
        plot_colorbar=True,
    )

    gu.savefig(
        savedir=savedir,
        figure=fig,
        axes=ax,
        tick_width=tick_width,
        tick_length=tick_length,
        tick_labelsize=16,
        xlabels=labels[0][0],
        ylabels=labels[0][1],
        label_size=20,
        titles=fname + " central slice in " + labels[0][2],
        title_size=20,
        legend_labelsize=14,
        filename=fname + "_centralslice_z" + comment,
    )

    fig, ax, _ = gu.imshow_plot(
        dataset[:, cen_y, :],
        sum_frames=False,
        width_v=2 * width[0],
        width_h=2 * width[2],
        scale=plot_scale,
        vmin=vmin,
        vmax=vmax,
        reciprocal_space=False,
        is_orthogonal=is_orthogonal,
        plot_colorbar=True,
    )

    gu.savefig(
        savedir=savedir,
        figure=fig,
        axes=ax,
        tick_width=tick_width,
        tick_length=tick_length,
        tick_labelsize=16,
        xlabels=labels[1][0],
        ylabels=labels[1][1],
        label_size=20,
        titles=fname + " central slice in " + labels[1][2],
        title_size=20,
        legend_labelsize=14,
        filename=fname + "_centralslice_y" + comment,
    )

    fig, ax, _ = gu.imshow_plot(
        dataset[:, :, cen_x],
        sum_frames=False,
        width_v=2 * width[0],
        width_h=2 * width[1],
        scale=plot_scale,
        vmin=vmin,
        vmax=vmax,
        reciprocal_space=False,
        is_orthogonal=is_orthogonal,
        plot_colorbar=True,
    )

    gu.savefig(
        savedir=savedir,
        figure=fig,
        axes=ax,
        tick_width=tick_width,
        tick_length=tick_length,
        tick_labelsize=16,
        xlabels=labels[2][0],
        ylabels=labels[2][1],
        label_size=20,
        titles=fname + " central slice in " + labels[2][2],
        title_size=20,
        legend_labelsize=14,
        filename=fname + "_centralslice_x" + comment,
    )

plt.show()
