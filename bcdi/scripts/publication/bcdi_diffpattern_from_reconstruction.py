#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

from functools import reduce
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from numbers import Real
import numpy as np
from numpy.fft import fftn, fftshift
import pathlib
import tkinter as tk
from tkinter import filedialog
import sys
sys.path.append('D:/myscripts/bcdi/')
import bcdi.graph.graph_utils as gu
import bcdi.postprocessing.postprocessing_utils as pu
import bcdi.utils.validation as valid

helptext = """
Calculate the diffraction pattern corresponding to a reconstructed 3D crystal (output of phase retrieval), after 
padding it to the desired shape. The reconstructed crystal file should be a .NPZ with field names 'amp' for the modulus 
and 'displacement' for the phase. Corresponding q values can be loaded optionally.
"""

scan = 2  # scan number
root_folder = "D:/data/P10_2nd_test_isosurface_Dec2020/data_nanolab/"
sample_name = "dataset_"
datadir = root_folder + sample_name + str(scan) + '_pearson97.5_newpsf/pynx/'
original_shape = (392, 420, 256)  # the reconstruction will be padded to that shape before calculating its diffraction
# pattern (shape of the diffraction pattern before phase retrieval)
# first dataset (168, 1024, 800)
# second dataset (168, 1024, 800)
mode_factor = 0.2740  # correction factor due to mode decomposition, leave None if no correction is needed
# the diffraction intensity will be multiplied by the square of this factor
# mode_factor = 0.2740 dataset_1_newpsf
# mode_factor = 0.2806 dataset_1_nopsf
# mode_factor = 0.2744 dataset_2_pearson97.5_newpsf
is_orthogonal = True  # True if the recosntructed crystal is in n orthonormal frame
load_qvalues = True  # True to load the q values. It expects a single npz file with fieldnames 'qx', 'qy' and 'qz'
##############################
# settings related to saving #
##############################
savedir = datadir + 'test/'  # results will be saved here, if None it will default to datadir
save_qyqz = True  # True to save the strain in QyQz plane
save_qyqx = True  # True to save the strain in QyQx plane
save_qzqx = True  # True to save the strain in QzQx plane
save_sum = True  # True to save the summed diffraction pattern in the detector, False to save the central slice only
comment = ''  # string to add to the filename when saving, should start with "_"
##########################
# settings for the plots #
##########################
tick_direction = 'out'  # 'out', 'in', 'inout'
tick_length = 10  # in plots
tick_width = 2  # in plots
tick_spacing = (0.025, 0.025, 0.025)  # tuple of three numbers, in 1/A. Leave None for default.
num_ticks = 5  # number of ticks to use in axes when tick_spacing is not defined
colorbar_range = (0, 4.5)  # (vmin, vmax) log scale in photon counts, leave None for default.
debug = False  # True to see more plots
grey_background = False  # True to set nans to grey in the plots
##################################
# end of user-defined parameters #
##################################

#########################
# check some parameters #
#########################
valid_name = 'diffpattern_from_reconstruction'
savedir = savedir or datadir
pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)

valid.valid_container(original_shape, container_types=(tuple, list, np.ndarray), item_types=int, min_excluded=0,
                      length=3, name=valid_name)
if mode_factor is None:
    mode_factor = 1
valid.valid_item(mode_factor, allowed_types=Real, min_excluded=0, name=valid_name)

valid.valid_item(save_qyqz, allowed_types=bool, name=valid_name)
valid.valid_item(save_qyqx, allowed_types=bool, name=valid_name)
valid.valid_item(save_qzqx, allowed_types=bool, name=valid_name)
valid.valid_item(save_sum, allowed_types=bool, name=valid_name)
if len(comment) != 0 and not comment.startswith('_'):
    comment = '_' + comment

if tick_direction not in {'out', 'in', 'inout'}:
    raise ValueError("tick_direction should be 'out', 'in' or 'inout'")
valid.valid_item(tick_length, allowed_types=int, min_excluded=0, name=valid_name)
valid.valid_item(tick_width, allowed_types=int, min_excluded=0, name=valid_name)
if isinstance(tick_spacing, Real):
    tick_spacing = (tick_spacing,) * 3
valid.valid_container(tick_spacing, container_types=(tuple, list, np.ndarray), allow_none=True, item_types=Real,
                      min_excluded=0, name=valid_name)
valid.valid_item(num_ticks, allowed_types=int, min_excluded=0, name=valid_name)
valid.valid_container(colorbar_range, container_types=(tuple, list, np.ndarray), item_types=Real, length=2,
                      allow_none=True, name=valid_name)
valid.valid_item(debug, allowed_types=bool, name=valid_name)
valid.valid_item(grey_background, allowed_types=bool, name=valid_name)

#############################
# define default parameters #
#############################
mpl.rcParams['axes.linewidth'] = tick_width  # set the linewidth globally

if grey_background:
    bad_color = '0.7'
else:
    bad_color = '1.0'  # white background

colormap = gu.Colormap(bad_color=bad_color)
my_cmap = colormap.cmap

if is_orthogonal:
    labels = ('Qx', 'Qz', 'Qy')
else:
    labels = ('rocking angle', 'detector Y', 'detector X')

if load_qvalues:
    draw_ticks = True
else:
    draw_ticks = False

##################################
# load the reconstructed crystal #
##################################
plt.ion()
root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename(initialdir=datadir, title="Select the reconstruction file",
                                       filetypes=[("NPZ", "*.npz")])
npzfile = np.load(file_path)
phase = npzfile['displacement']
amp = npzfile['amp']
if amp.ndim != 3:
    raise ValueError('3D arrays are expected')

gu.multislices_plot(array=amp, sum_frames=False, scale='linear', plot_colorbar=True, reciprocal_space=False,
                    is_orthogonal=is_orthogonal, title='Modulus')

####################################################################
# calculate the complex amplitude  and pad it to the desired shape #
####################################################################
obj = amp * np.exp(1j * phase)
obj = pu.crop_pad(array=obj, output_shape=original_shape, debugging=debug)

#####################################
# calculate the diffraction pattern #
#####################################
data = fftshift(fftn(obj)) / np.sqrt(reduce(lambda x, y: x*y, original_shape))  # complex diffraction amplitude
data = abs(np.multiply(data, np.conjugate(data)))  # diffraction intensity
data = data * mode_factor**2  # correction due to the loss of the normalization with mode decomposition

################################
# optionally load the q values #
################################
nbz, nby, nbx = data.shape
if load_qvalues:
    file_path = filedialog.askopenfilename(initialdir=datadir, title="Select the q values",
                                           filetypes=[("NPZ", "*.npz")])
    q_values = np.load(file_path)
    qx = q_values['qx']
    qz = q_values['qz']
    qy = q_values['qy']
    print('Loaded: qx shape:', qx.shape, 'qz shape:', qz.shape, 'qy shape:', qy.shape)
    if not (*qx.shape, *qz.shape, *qy.shape) == data.shape:
        raise ValueError('q values and data shape are incompatible')
    q_range = (qx.min(), qx.max(), qz.min(), qz.max(), qy.min(), qy.max())
else:
    q_range = (0, nbz, 0, nby, 0, nbx)

print('q range:', [f'{val:.4f}' for val in q_range])

#############################################################
# define the positions of the axes ticks and colorbar ticks #
#############################################################
# use 5 ticks by default if tick_spacing is None for the axis
pixel_spacing = ((tick_spacing[0] or (q_range[1]-q_range[0])/num_ticks),
                 (tick_spacing[1] or (q_range[3]-q_range[2])/num_ticks),
                 (tick_spacing[2] or (q_range[5]-q_range[4])/num_ticks))
print('Pixel spacing:', pixel_spacing)

if colorbar_range is None:  # use rounded acceptable values
    colorbar_range = (np.ceil(np.median(np.log10(data[np.logical_and(data != 0, ~np.isnan(data))]))),
                      np.ceil(np.log10(data[np.logical_and(data != 0, ~np.isnan(data))].max())))
numticks_colorbar = int(np.floor(colorbar_range[1] - colorbar_range[0] + 1))

############################
# plot views in QyQz plane #
############################
if save_qyqz:
    fig, ax0 = plt.subplots(1, 1)
    if save_sum:
        # extent (left, right, bottom, top)
        plt0 = ax0.imshow(np.log10(data.sum(axis=0)), cmap=my_cmap, vmin=colorbar_range[0], vmax=colorbar_range[1],
                          extent=[q_range[4], q_range[5], q_range[3], q_range[2]])
    else:
        plt0 = ax0.imshow(np.log10(data[nbz//2, :, :]), cmap=my_cmap, vmin=colorbar_range[0], vmax=colorbar_range[1],
                          extent=[q_range[4], q_range[5], q_range[3], q_range[2]])
    ax0.invert_yaxis()  # qz is pointing up
    ax0.xaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing[2]))
    ax0.yaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing[1]))
    gu.colorbar(plt0, numticks=numticks_colorbar)
    gu.savefig(savedir=savedir, figure=fig, axes=ax0, tick_width=tick_width, tick_length=tick_length,
               tick_direction=tick_direction, label_size=16, xlabels=labels[2], ylabels=labels[1],
               filename=sample_name + str(scan) + comment + '_fromrec_qyqz', labelbottom=True, labelleft=True,
               labelright=False, labeltop=False, left=draw_ticks, right=draw_ticks, bottom=draw_ticks, top=draw_ticks)

############################
# plot views in QyQx plane #
############################
if save_qyqx:
    fig, ax0 = plt.subplots(1, 1)
    if save_sum:
        # extent (left, right, bottom, top)
        plt0 = ax0.imshow(np.log10(data.sum(axis=1)), cmap=my_cmap, vmin=colorbar_range[0], vmax=colorbar_range[1],
                          extent=[q_range[4], q_range[5], q_range[1], q_range[0]])
    else:
        plt0 = ax0.imshow(np.log10(data[:, nby//2, :]), cmap=my_cmap, vmin=colorbar_range[0], vmax=colorbar_range[1],
                          extent=[q_range[4], q_range[5], q_range[1], q_range[0]])

    ax0.xaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing[2]))
    ax0.yaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing[0]))
    gu.colorbar(plt0, numticks=numticks_colorbar)
    gu.savefig(savedir=savedir, figure=fig, axes=ax0, tick_width=tick_width, tick_length=tick_length,
               tick_direction=tick_direction, label_size=16, xlabels=labels[2], ylabels=labels[0],
               filename=sample_name + str(scan) + comment + '_fromrec_qyqx', labelbottom=True, labelleft=True,
               labelright=False, labeltop=False, left=draw_ticks, right=draw_ticks, bottom=draw_ticks, top=draw_ticks)

############################
# plot views in QzQx plane #
############################
if save_qzqx:
    fig, ax0 = plt.subplots(1, 1)
    if save_sum:
        # extent (left, right, bottom, top)
        plt0 = ax0.imshow(np.log10(data.sum(axis=2)), cmap=my_cmap, vmin=colorbar_range[0], vmax=colorbar_range[1],
                          extent=[q_range[2], q_range[3], q_range[1], q_range[0]])
    else:
        plt0 = ax0.imshow(np.log10(data[:, :, nbx//2]), cmap=my_cmap, vmin=colorbar_range[0], vmax=colorbar_range[1],
                          extent=[q_range[2], q_range[3], q_range[1], q_range[0]])

    ax0.xaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing[1]))
    ax0.yaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing[0]))
    gu.colorbar(plt0, numticks=numticks_colorbar)
    gu.savefig(savedir=savedir, figure=fig, axes=ax0, tick_width=tick_width, tick_length=tick_length,
               tick_direction=tick_direction, label_size=16, xlabels=labels[1], ylabels=labels[0],
               filename=sample_name + str(scan) + comment + '_fromrec_qzqx', labelbottom=True, labelleft=True,
               labelright=False, labeltop=False, left=draw_ticks, right=draw_ticks, bottom=draw_ticks, top=draw_ticks)

plt.ioff()
plt.show()

