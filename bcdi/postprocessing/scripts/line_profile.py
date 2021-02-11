# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import json
import matplotlib.pyplot as plt
from numbers import Real
import numpy as np
import pathlib
from scipy.interpolate import interp1d
import sys
import tkinter as tk
from tkinter import filedialog
sys.path.append('D:/myscripts/bcdi/')
import bcdi.graph.graph_utils as gu
import bcdi.utils.utilities as util
import bcdi.utils.validation as valid

helptext = """
This script allow to plot and save linecuts through a 2D or 3D object in function of a modulus threshold 
defining the object from the background. Must be given as input: the voxel size (possibly different in all directions), 
the direction of the cuts and a list of points where to apply the cut along this direction.   
"""

datadir = "D:/data/P10_2nd_test_isosurface_Dec2020/data_nanolab/dataset_1/PtNP1_00128/result/"  # data folder
savedir = "D:/data/P10_2nd_test_isosurface_Dec2020/data_nanolab/dataset_1/PtNP1_00128/result/linecuts/"
# results will be saved here, if None it will default to datadir
threshold = np.linspace(0, 1.0, num=20)
# number or list of numbers between 0 and 1, modulus threshold defining the normalized object from the background
direction = (0, 1, 0)  # tuple of 2 or 3 numbers (2 for 2D object, 3 for 3D) defining the direction of the cut
# in the orthonormal reference frame is given by the array axes. It will be corrected for anisotropic voxel sizes.
points = {(24, 26, 23), (24, 26, 24), (24, 26, 25),
          (25, 26, 23), (25, 26, 24), (25, 26, 25),
          (26, 26, 23), (26, 26, 24), (26, 26, 25),}
# list/tuple/set of 2 or 3 indices (2 for 2D object, 3 for 3D) corresponding to the points where
# the cut alond direction should be performed. The reference frame is given by the array axes.
voxel_size = 5  # positive real number  or tuple of 2 or 3 positive real number (2 for 2D object, 3 for 3D)
width_lines = (100, 101, 102)  # list of vertical lines that will appear in the plot width vs threshold
debug = False  # True to print the output dictionary and plot the legend
comment = ''  # string to add to the filename when saving
##################################
# end of user-defined parameters #
##################################

###############################
# list of colors for the plot #
###############################
colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')
markers = ('.', 'v', '^', '<', '>')

#################
# load the data #
#################
plt.ion()
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(initialdir=datadir,
                                       filetypes=[("NPZ", "*.npz"), ("NPY", "*.npy"),
                                                  ("CXI", "*.cxi"), ("HDF5", "*.h5")])

obj, _ = util.load_file(file_path)
ndim = obj.ndim

#########################
# check some parameters #
#########################
if ndim not in {2, 3}:
    raise ValueError(f'Number of dimensions = {ndim}, expected 2 or 3')

valid.valid_container(direction, container_types=(list, tuple, np.ndarray), length=ndim, item_types=Real,
                      name='line_profile')

valid.valid_container(points, container_types=(list, tuple, set), min_length=1, name='line_profile')
for point in points:
    valid.valid_container(point, container_types=(list, tuple, np.ndarray), length=ndim, item_types=Real,
                          min_included=0, name='line_profile')

if isinstance(voxel_size, Real):
    voxel_size = (voxel_size,) * ndim
valid.valid_container(voxel_size, container_types=(list, tuple, np.ndarray), length=ndim, item_types=Real,
                      min_excluded=0, name='line_profile')

savedir = savedir or datadir
pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)

if isinstance(threshold, Real):
    threshold = (threshold,)
valid.valid_container(threshold, container_types=(list, tuple, np.ndarray), item_types=Real,
                      min_included=0, max_included=1, name='line_profile')

if isinstance(width_lines, Real):
    width_lines = (width_lines,)
valid.valid_container(width_lines, container_types=(list, tuple, np.ndarray), item_types=Real,
                      min_excluded=0, name='line_profile')

comment = f'_direction{direction[0]}_{direction[1]}_{direction[2]}_{comment}'

#########################
# normalize the modulus #
#########################
obj = abs(obj) / abs(obj).max()  # normalize the modulus to 1
obj[np.isnan(obj)] = 0  # remove nans
if ndim == 2:
    gu.imshow_plot(array=obj, plot_colorbar=True, reciprocal_space=False, is_orthogonal=True)
else:
    gu.multislices_plot(array=obj, sum_frames=False, plot_colorbar=True, reciprocal_space=False, is_orthogonal=True)

#####################################
# create the linecut for each point #
#####################################
result = dict()
result['direction'] = direction
for point in points:
    # get the distances and the modulus values along the linecut
    distance, cut = util.linecut(array=obj, point=point, direction=direction, voxel_size=voxel_size)
    # store the result in a dictionary (cuts can have different lengths depending on the direction)
    result[f'voxel {point}'] = {'distance': distance, 'cut': cut}

######################
#  plot the linecuts #
######################
fig = plt.figure(figsize=(12, 9))
ax = plt.subplot(111)
plot_nb = 0
for key, value in result.items():
    if key != 'direction':  # value is a dictionary {'distance': 1D array, 'cut': 1D array}
        line, = ax.plot(value['distance'], value['cut'], color=colors[plot_nb % len(colors)],
                        marker=markers[(plot_nb // len(colors)) % len(markers)], fillstyle='none', markersize=6,
                        linestyle='-', linewidth=1)
        line.set_label(f'cut through {key}')
        plot_nb += 1

ax.set_xlabel('width (nm)', fontsize=20)
ax.set_ylabel('modulus', fontsize=20)
if debug:
    ax.legend(fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=16)
fig.savefig(savedir + 'cut' + comment + '.png')

#################################################################################
# calculate the evolution of the width of the object depending on the threshold #
#################################################################################
for key, value in result.items():
    if key != 'direction':
        fit = interp1d(value['distance'], value['cut'])
        dist_interp = np.linspace(value['distance'].min(), value['distance'].max(), num=10000)
        cut_interp = fit(dist_interp)
        width = np.empty(len(threshold))

        # calculate the function width vs threshold
        for idx, thres in enumerate(threshold):
            # calculate the distances where the modulus is equal to threshold
            crossings = np.argwhere(cut_interp > thres)
            if len(crossings) > 1:
                width[idx] = dist_interp[crossings.max()] - dist_interp[crossings.min()]
            else:
                width[idx] = 0

        # fit the function width vs threshold and estimate where it crosses the expected widths
        fit = interp1d(width, threshold)  # width vs threshold is monotonic (decreasing with increasing threshold)
        fit_thresh = np.empty(len(width_lines))
        for idx, val in enumerate(width_lines):
            fit_thresh[idx] = fit(val)
        # update the dictionary value
        value['threshold'] = threshold
        value['width'] = width
        value['fitted_threshold'] = fit_thresh

#################################################
# calculate statistics on the fitted thresholds #
#################################################
count = 0
tmp_thres = np.zeros((len(width_lines), len(points)))
for key, value in result.items():
    if key != 'direction':  # iterating over points, value is a dictionary
        for idx in range(len(width_lines)):
            tmp_thres[idx, count] = value['fitted_threshold'][idx]
        count += 1
mean_thres = np.mean(tmp_thres, axis=1)
std_thres = np.std(tmp_thres, axis=1)

# update the dictionary
result['expected_width'] = width_lines
result['mean_thres'] = np.round(mean_thres, decimals=3)
result['std_thres'] = np.round(std_thres, decimals=3)

#################################
#  plot the widths vs threshold #
#################################
fig = plt.figure(figsize=(12, 9))
ax = plt.subplot(111)
plot_nb = 0
for key, value in result.items():
    if key == 'direction':
        ax.set_title(f'Width vs threshold in the direction {value}\n', fontsize=20)
    elif key == 'expected_width':
        fig.text(0.15, 0.30, f'expected widths: {value}', size=16)
    elif key == 'mean_thres':
        fig.text(0.15, 0.25, f'fitted thresholds: {value}', size=16)
    elif key == 'std_thres':
        fig.text(0.15, 0.20, f'stds: {value}', size=16)
    else:  # iterating over points, value is a dictionary
        line, = ax.plot(value['threshold'], value['width'], color=colors[plot_nb % len(colors)],
                        marker=markers[(plot_nb // len(colors)) % len(markers)], fillstyle='none', markersize=6,
                        linestyle='-', linewidth=1)
        line.set_label(f'cut through {key}')
        plot_nb += 1

ax.set_xlabel('threshold', fontsize=20)
ax.set_ylabel('width (nm)', fontsize=20)
if debug:
    ax.legend(fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=16)
for hline in width_lines:
    ax.axhline(y=hline, linestyle='dashed', color='k', linewidth=1)
fig.savefig(savedir + 'width_vs_threshold' + comment + '.png')

###################
# save the result #
###################
if debug:
    print('output dictionary:\n', json.dumps(result, cls=util.CustomEncoder, indent=4))

with open(savedir+'cut' + comment + '.json', 'w', encoding='utf-8') as file:
    json.dump(result, file, cls=util.CustomEncoder, ensure_ascii=False, indent=4)

np.savez_compressed(savedir + 'cut' + comment + '.npz', result=result)
plt.ioff()
plt.show()
