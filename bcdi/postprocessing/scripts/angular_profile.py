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
from scipy.ndimage.measurements import center_of_mass
import sys
import tkinter as tk
from tkinter import filedialog
sys.path.append('D:/myscripts/bcdi/')
import bcdi.graph.graph_utils as gu
import bcdi.utils.utilities as util
import bcdi.utils.validation as valid

helptext = """
This script allow to plot the width of a 2D object in function of the angle and a modulus threshold defining the object 
from the background. Must be given as input: the voxel size (possibly different in all directions), the angular step 
size and an origin point where all linecuts pass by.   
"""

datadir = "D:/data/P10_2nd_test_isosurface_Dec2020/data_nanolab/dataset_1/PtNP1_00128/result/"  # data folder
savedir = "D:/data/P10_2nd_test_isosurface_Dec2020/data_nanolab/dataset_1/PtNP1_00128/result/linecuts/"
# results will be saved here, if None it will default to datadir
threshold = np.round(np.linspace(0.3, 0.6, num=10), decimals=3)
# number or list of numbers between 0 and 1, modulus threshold defining the normalized object from the background
angular_step = 10  # in degrees, the linecut directions will be automatically calculated
# in the orthonormal reference frame is given by the array axes. It will be corrected for anisotropic voxel sizes.
origin = None  # origin where all the line cuts pass by. If None, it will use the center of mass of the modulus
voxel_size = 5  # positive real number  or tuple of 2 or 3 positive real number (2 for 2D object, 3 for 3D)
width_lines = (100, 101, 102)  # list of vertical lines that will appear in the plot width vs threshold
sum_axis = 1  # if the object is 3D, it will be summed along that axis
debug = True  # True to print the output dictionary and plot the legend
comment = ''  # string to add to the filename when saving
##################################
# end of user-defined parameters #
##################################

###############################
# list of colors for the plot #
###############################
colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')
markers = ('.', 'v', '^', '<', '>')

##################################################
# create the list of directions for the linecuts #
##################################################
angles = np.arange(0, 180, angular_step)
nb_dir = len(angles)
directions = []
for idx in range(nb_dir):
    directions.append((np.sin(angles[idx] * np.pi / 180), np.cos(angles[idx] * np.pi / 180)))

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
if obj.ndim == 3:
    obj = obj.sum(axis=sum_axis)

ndim = obj.ndim

#########################
# check some parameters #
#########################
if ndim != 2:
    raise ValueError(f'Number of dimensions = {ndim}, expected 2 or 3')

if origin is None:
    piy, pix = center_of_mass(obj)
    origin = int(np.rint(piy)), int(np.rint(pix))
valid.valid_container(origin, container_types=(list, tuple), length=2, item_types=int, name='line_profile')

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

comment = f'_origin_{origin}_ang_{angular_step}deg_{comment}'

#########################
# normalize the modulus #
#########################
obj = abs(obj) / abs(obj).max()  # normalize the modulus to 1
obj[np.isnan(obj)] = 0  # remove nans
gu.imshow_plot(array=obj, plot_colorbar=True, reciprocal_space=False, is_orthogonal=True)

#########################################
# create the linecut for each direction #
#########################################
result = dict()
result['origin'] = origin
for idx, direction in enumerate(directions):
    # get the distances and the modulus values along the linecut
    distance, cut = util.linecut(array=obj, point=origin, direction=direction, voxel_size=voxel_size)
    # store the result in a dictionary (cuts can have different lengths depending on the direction)
    result[f'direction {direction}'] = {'angle': angles[idx], 'distance': distance, 'cut': cut}

#######################
#  plot all line cuts #
#######################
fig = plt.figure(figsize=(12, 9))
ax = plt.subplot(111)
plot_nb = 0
for key, value in result.items():
    if key != 'origin':  # value is a dictionary {'angle': angles[idx], 'distance': distance, 'cut': cut}
        line, = ax.plot(value['distance'], value['cut'], color=colors[plot_nb % len(colors)],
                        marker=markers[plot_nb // len(colors)], fillstyle='none', markersize=6,
                        linestyle='-', linewidth=1)
        line.set_label(f'direction {key}')
        plot_nb += 1

ax.set_xlabel('width (nm)', fontsize=20)
ax.set_ylabel('modulus', fontsize=20)
if debug:
    ax.legend(fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=16)
fig.savefig(savedir + 'cuts' + comment + '.png')

##############################################################################
# calculate the evolution of the width vs threshold for different directions #
##############################################################################
for key, value in result.items():  # iterating over the directions (except the key 'origin')
    if key != 'origin':  # value is a dictionary {'angle': angles[idx], 'distance': distance, 'cut': cut}
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
        # update the dictionary value
        value['threshold'] = threshold
        value['width'] = width

##########################################################################
# calculate the evolution of the width vs angle for different thresholds #
##########################################################################
ang_width_threshold = np.empty((len(threshold), nb_dir))
for idx, thres in enumerate(threshold):
    tmp_angles = np.empty(nb_dir)  # will be used to reorder the angles
    angular_width = np.empty(nb_dir)
    count = 0
    for key, value in result.items():  # iterating over the directions (except the key 'origin')
        if key != 'origin':  # value is a dictionary {'angle': angles[idx], 'distance': distance, 'cut': cut}
            tmp_angles[count] = value['angle']  # index related to the angle/direction
            assert thres == value['threshold'][idx], 'ordering error in threshold'
            angular_width[count] = value['width'][idx]  # index related to the threshold
            count += 1
    assert np.all(np.isclose(tmp_angles, angles)), 'ordering error in angles'
    ang_width_threshold[idx, :] = angular_width

# update the dictionary
result['threshold'] = threshold
result['ang_width_threshold'] = ang_width_threshold

#####################################################
#  plot the width vs angle for different thresholds #
#####################################################
fig = plt.figure(figsize=(12, 9))
ax = plt.subplot(111)
for idx, thres in enumerate(threshold):
    line, = ax.plot(angles, result[f'ang_width_threshold'][idx], color=colors[idx % len(colors)],
                    marker=markers[idx // len(colors)], fillstyle='none', markersize=6,
                    linestyle='-', linewidth=1)
    line.set_label(f'threshold {thres}')

ax.set_xlabel('angle (deg)', fontsize=20)
ax.set_ylabel('width (nm)', fontsize=20)
if debug:
    ax.legend(fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=16)
fig.savefig(savedir + 'width_vs_ang' + comment + '.png')

###################
# save the result #
###################
if debug:
    print('output dictionary:\n', json.dumps(result, cls=util.CustomEncoder, indent=4))

with open(savedir+'ang_width' + comment + '.json', 'w', encoding='utf-8') as file:
    json.dump(result, file, cls=util.CustomEncoder, ensure_ascii=False, indent=4)

np.savez_compressed(savedir + 'ang_width' + comment + '.npz', result=result)
plt.ioff()
plt.show()
