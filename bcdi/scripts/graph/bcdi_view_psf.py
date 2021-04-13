#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np
from numpy.fft import fftshift
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
import pathlib
import tkinter as tk
from tkinter import filedialog
import sys
sys.path.append('D:/myscripts/bcdi/')
import bcdi.graph.graph_utils as gu

helptext = """
Open and plot the psf from a .cxi reconstruction file (from PyNX). The psf has to be fftshifted.
"""

datadir = "D:/data/P10_2nd_test_isosurface_Dec2020/data_nanolab/dataset_1_newpsf/"
save_dir = datadir + "psf/"
is_orthogonal = False  # True if the data was orthogonalized before phasing
comment = ''  # should start with _
width = 30  # the psf will be plotted for +/- this number of pixels from center of the array
vmin = -6  # min of the colorbar for plots (log scale). Use np.nan for default.
vmax = 0  # max of the colorbar for plots (log scale). Use np.nan for default.
fft_shift = False  # True to apply fftshift to the loaded psf
save_slices = True  # True to save individual 2D slices (in z, y, x)
tick_direction = 'out'  # 'out', 'in', 'inout'
tick_length = 8  # in plots
tick_width = 2  # in plots
linewidth = 2  # linewidth for the plot frame
###################
# define colormap #
###################
bad_color = '1.0'  # white background
colormap = gu.Colormap(bad_color=bad_color)
my_cmap = colormap.cmap
mpl.rcParams['axes.linewidth'] = tick_width  # set the linewidth globally

#####################################################
# load the CXI file, output of PyNX phase retreival #
#####################################################
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(initialdir=datadir, filetypes=[("CXI", "*.cxi")])

h5file = h5py.File(file_path, 'r')
try:
    if fft_shift:
        dataset = fftshift(h5file['/entry_1/image_1/instrument_1/detector_1/point_spread_function'].value)
    else:
        dataset = h5file['/entry_1/image_1/instrument_1/detector_1/point_spread_function'].value
except KeyError as ex:
    print('The PSF was not saved in the CXI file')
    raise KeyError from ex

# normalize the psf to 1
dataset = abs(dataset) / abs(dataset).max()

#########################
# check some parameters #
#########################
save_dir = save_dir or datadir
pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
nbz, nby, nbx = np.shape(dataset)
print(f'psf shape = {dataset.shape}')
cen_z, cen_y, cen_x = nbz // 2, nby // 2, nbx // 2
if any((cen_z-width < 0, cen_z+width > nbz, cen_y-width < 0, cen_y+width > nby, cen_x-width < 0, cen_x+width > nbx)):
    print('width is not compatible with the psf shape')
    width = min(cen_z, cen_y, cen_x)

if is_orthogonal:
    title = 'log(psf) in laboratory frame'
else:
    title = 'log(psf) in detector frame'

if comment and not comment.startswith('_'):
    comment = '_' + comment

#########################
# plot and save the psf #
#########################
fig, _, _ = gu.multislices_plot(dataset[cen_z-width:cen_z+width, cen_y-width:cen_y+width, cen_x-width:cen_x+width],
                                scale='log', sum_frames=False, title=title, vmin=vmin, vmax=vmax,
                                reciprocal_space=False, is_orthogonal=is_orthogonal, plot_colorbar=True)
fig.savefig(save_dir + 'psf_centralslice' + comment + '.png')

if save_slices:
    if is_orthogonal:  # orthogonal laboratory frame, CXI convention z downstream, y vertical up, x outboard
        labels = (('x', 'y', 'z'),  # labels for x axis, y axis, title
                  ('x', 'z', 'y'),
                  ('y', 'z', 'x'))
    else:  # non-orthogonal detector frame stacking axis, detector vertical Y down, detector horizontal X inboard
        labels = (('detector X', 'detector Y', 'stacking axis'),  # labels for x axis, y axis, title
                  ('detector X', 'stacking axis', 'detector Y'),
                  ('detector Y', 'stacking axis', 'detector X'))

    fig, ax, _ = gu.imshow_plot(dataset[cen_z, cen_y-width:cen_y+width, cen_x-width:cen_x+width], sum_frames=False,
                                scale='log', vmin=vmin, vmax=vmax, reciprocal_space=False, is_orthogonal=is_orthogonal,
                                plot_colorbar=True)

    gu.savefig(savedir=save_dir, figure=fig, axes=ax, tick_width=tick_width, tick_length=tick_length,
               tick_labelsize=16, xlabels=labels[0][0], ylabels=labels[0][1], label_size=20,
               titles='psf central slice in '+labels[0][2], title_size=20,
               legend_labelsize=14, filename='psf_centralslice_z' + comment)

    fig, ax, _ = gu.imshow_plot(dataset[cen_z-width:cen_z+width, cen_y, cen_x-width:cen_x+width], sum_frames=False,
                                scale='log', vmin=vmin, vmax=vmax, reciprocal_space=False, is_orthogonal=is_orthogonal,
                                plot_colorbar=True)

    gu.savefig(savedir=save_dir, figure=fig, axes=ax, tick_width=tick_width, tick_length=tick_length,
               tick_labelsize=16, xlabels=labels[1][0], ylabels=labels[1][1], label_size=20,
               titles='psf central slice in '+labels[1][2], title_size=20,
               legend_labelsize=14, filename='psf_centralslice_y' + comment)

    fig, ax, _ = gu.imshow_plot(dataset[cen_z-width:cen_z+width, cen_y-width:cen_y+width, cen_x], sum_frames=False,
                                scale='log', vmin=vmin, vmax=vmax, reciprocal_space=False, is_orthogonal=is_orthogonal,
                                plot_colorbar=True)

    gu.savefig(savedir=save_dir, figure=fig, axes=ax, tick_width=tick_width, tick_length=tick_length,
               tick_labelsize=16, xlabels=labels[2][0], ylabels=labels[2][1], label_size=20,
               titles='psf central slice in '+labels[2][2], title_size=20,
               legend_labelsize=14, filename='psf_centralslice_x' + comment)

plt.show()
