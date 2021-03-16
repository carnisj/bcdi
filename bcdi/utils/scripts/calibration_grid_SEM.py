# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import json
from lmfit import minimize, Parameters, report_fit
import matplotlib.pyplot as plt
from numbers import Real
import numpy as np
import os
import pathlib
from pprint import pprint
from scipy.interpolate import interp1d
import sys
import tkinter as tk
from tkinter import filedialog
sys.path.append('D:/myscripts/bcdi/')
import bcdi.graph.graph_utils as gu
import bcdi.utils.utilities as util
import bcdi.utils.validation as valid

helptext = """
This script allow to plot and save linecuts through a 2D SEM image of a calibration grid. 
Must be given as input: the voxel size, the direction of the cuts and a list of points where to apply the cut along 
this direction. Optionally Gaussian can be fitted to the grid maxima.
"""

datadir = "D:/data/P10_2nd_test_isosurface_Dec2020/data_nanolab/AFM-SEM/SEM calibration/"  # data folder
savedir = "D:/data/P10_2nd_test_isosurface_Dec2020/data_nanolab/AFM-SEM/SEM calibration/test/"
# results will be saved here, if None it will default to datadir
direction = (0, 1)  # tuple of 2 numbers defining the direction of the cut
# in the orthonormal reference frame is given by the array axes. It will be corrected for anisotropic voxel sizes.
points = [(5, 0), (300, 0)]  # MCS_06.tif
# [(5, 0), (25, 0), (50, 0), (75, 0), (100, 0), (125, 0), (150, 0), (175, 0), (200, 0), (225, 0)]  # MCS_03.tif
# list/tuple of 2 indices corresponding to the points where
# the cut alond direction should be performed. The reference frame is given by the array axes.
fit_roi = None
# fit_roi = [[(350, 495), (5660, 5800)],
#            [(350, 495), (5660, 5800)],
#            [(350, 495), (5660, 5780)],
#            [(350, 495), (5660, 5780)],              # ROIs for MCS_03.tif
#            [(350, 495), (5650, 5790)],
#            [(350, 495), (5650, 5790)],
#            [(350, 495), (5650, 5790)],
#            [(350, 485), (5640, 5780)],
#            [(350, 485), (5640, 5780)],
#            [(350, 485), (5640, 5780)]]  # ROIs that should be fitted for each point. There should be as many


# sublists as the number of points. Leave None otherwise.
background_roi = [0, 400, 112, 118]  # [ystart, ystop, xstart, xstop], the mean intensity in this ROI will be
# subtracted from the data. Leave None otherwise
# list of tuples [(start, stop), ...] of regions to be fitted, in the unit of length along the linecut, None otherwise
voxel_size = 2.070393374741201 * 0.96829786  # positive real number, voxel size of the SEM image
expected_width = 5120  # in nm, real positive number or None
debug = False  # True to print the output dictionary and plot the legend
comment = ''  # string to add to the filename when saving
tick_length = 10  # in plots
tick_width = 2  # in plots
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
                                       filetypes=[("TIFF", "*.tif"), ("NPZ", "*.npz"), ("NPY", "*.npy"),
                                                  ("CXI", "*.cxi"), ("HDF5", "*.h5"), ("all files", "*.*")])

_, ext = os.path.splitext(file_path)
if ext in {'.png', '.jpg', '.tif'}:
    obj = util.image_to_ndarray(filename=file_path, convert_grey=True, cmap='gray', debug=False)
else:
    obj, _ = util.load_file(file_path)
ndim = obj.ndim

#########################
# check some parameters #
#########################
if ndim != 2:
    raise ValueError(f'Number of dimensions = {ndim}, expected 2')

valid.valid_container(direction, container_types=(list, tuple, np.ndarray), length=ndim, item_types=Real,
                      name='calibration_grid_SEM')

valid.valid_container(points, container_types=(list, tuple), min_length=1, name='calibration_grid_SEM')
for point in points:
    valid.valid_container(point, container_types=(list, tuple, np.ndarray), length=ndim, item_types=Real,
                          min_included=0, name='calibration_grid_SEM')

if isinstance(voxel_size, Real):
    voxel_size = (voxel_size,) * ndim
valid.valid_container(voxel_size, container_types=(list, tuple, np.ndarray), length=ndim, item_types=Real,
                      min_excluded=0, name='calibration_grid_SEM')

savedir = savedir or datadir
pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)

valid.valid_container(fit_roi, container_types=(list, tuple), allow_none=True, name='calibration_grid_SEM')
if fit_roi is not None:
    if len(fit_roi) != len(points):
        raise ValueError('There should be as many ROIs sublists as the number of points (None allowed)')
    for sublist in fit_roi:
        valid.valid_container(sublist, container_types=(list, tuple), allow_none=True, name='calibration_grid_SEM')
        if sublist is not None:
            for roi in sublist:
                valid.valid_container(roi, container_types=(list, tuple), length=ndim, item_types=Real,
                                      min_included=0, name='calibration_grid_SEM')
valid.valid_container(background_roi, container_types=(list, tuple), allow_none=True, item_types=int, min_included=0,
                      name='calibration_grid_SEM')

valid.valid_item(value=expected_width, allowed_types=Real, min_excluded=0, allow_none=True, name='calibration_grid_SEM')
comment = f'_direction{direction[0]}_{direction[1]}_{comment}'

#########################
# normalize the modulus #
#########################
obj = abs(obj) / abs(obj).max()  # normalize the modulus to 1
obj[np.isnan(obj)] = 0  # remove nans
if background_roi is not None:
    background = obj[background_roi[0]:background_roi[1]+1, background_roi[2:background_roi[3]+1]].mean()
    print(f'removing background = {background:.2f} from the data')
    obj = obj - background

if ndim == 2:
    gu.imshow_plot(array=obj, plot_colorbar=True, reciprocal_space=False, is_orthogonal=True)
else:
    gu.multislices_plot(array=obj, sum_frames=False, plot_colorbar=True, reciprocal_space=False, is_orthogonal=True)

#####################################
# create the linecut for each point #
#####################################
result = dict()
for point in points:
    # get the distances and the modulus values along the linecut
    distance, cut = util.linecut(array=obj, point=point, direction=direction, voxel_size=voxel_size)
    # store the result in a dictionary (cuts can have different lengths depending on the direction)
    result[f'pixel {point}'] = {'distance': distance, 'cut': cut}

######################
#  plot the linecuts #
######################
fig = plt.figure(figsize=(12, 9))
ax = plt.subplot(111)
plot_nb = 0
for key, value in result.items():
    # value is a dictionary {'distance': 1D array, 'cut': 1D array}
    line, = ax.plot(value['distance'], value['cut'], color=colors[plot_nb % len(colors)],
                    marker=markers[(plot_nb // len(colors)) % len(markers)], fillstyle='none', markersize=10,
                    linestyle='-', linewidth=1)
    line.set_label(f'cut through {key}')
    plot_nb += 1

ax.tick_params(labelbottom=False, labelleft=False, direction='out', length=tick_length, width=tick_width)
ax.spines['right'].set_linewidth(tick_width)
ax.spines['left'].set_linewidth(tick_width)
ax.spines['top'].set_linewidth(tick_width)
ax.spines['bottom'].set_linewidth(tick_width)
fig.savefig(savedir + 'cut' + comment + '.png')

ax.set_xlabel('width (nm)', fontsize=20)
ax.set_ylabel('modulus', fontsize=20)
if debug:
    ax.legend(fontsize=14)
ax.tick_params(labelbottom=True, labelleft=True, axis='both', which='major', labelsize=16)
fig.savefig(savedir + 'cut' + comment + '_labels.png')

###############################
# fit the peaks with gaussian #
###############################
if fit_roi is not None:
    # peaks = np.empty((len(points), len(fit_roi)))  # array where the peaks positions will be saved
    width = np.empty(len(points))
    # define the fit initial parameters
    fit_params = Parameters()

    idx_point = 0
    for key, value in result.items():
        # value is a dictionary {'distance': 1D array, 'cut': 1D array}
        tmp_str = f'{key}'
        print(f'\n{"#" * len(tmp_str)}\n' + tmp_str + '\n' + f'{"#" * len(tmp_str)}')
        for idx_roi, roi in enumerate(fit_roi[idx_point]):  # loop over the ROIs, roi is a tuple of two number
            # define the fit initial center
            tmp_str = f'{roi}'
            indent = 2
            print(f'\n{" " * indent}{"-" * len(tmp_str)}\n' + f'{" " * indent}' + tmp_str + '\n' +
                  f'{" " * indent}{"-" * len(tmp_str)}')
            fit_params.add('amp_1', value=50, min=1, max=100)
            fit_params.add('sig_1', value=25, min=15, max=35)
            fit_params.add('ratio_1', value=0.5, min=0, max=1)
            fit_params.add('cen_1', value=(roi[0]+roi[1])/2, min=roi[0], max=roi[1])
            # find linecut indices falling into the roi
            ind_start, ind_stop = util.find_nearest(value['distance'], roi)
            # run the fit
            minimization = minimize(util.objective_lmfit, fit_params,
                                    args=(value['distance'][ind_start:ind_stop+1],
                                          value['cut'][ind_start:ind_stop+1],
                                          'pseudovoigt'))
            report_fit(minimization.params)
            value[f'roi {roi}'] = minimization.params
            # peak_fit = util.function_lmfit(params=minimization.params,
            #                                x_axis=value['distance'][ind_start:ind_stop+1],
            #                                distribution='pseudovoigt')
            # peaks[idx_point, idx_roi] = minimization.params['cen_1'].value
            # idx_point += 1

        # calculate the mean distance between the first and last peaks
        width[idx_point] = (value[f'roi {fit_roi[idx_point][-1]}']['cen_1'].value -
                            value[f'roi {fit_roi[idx_point][0]}']['cen_1'].value)
        idx_point += 1

    # update the dictionnary
    print(f'\n widths: {width}')
    result['mean_width'] = np.mean(width)
    result['std_width'] = np.std(width)
    result['fitting_rois'] = fit_roi

    tmp_str = 'mean width'
    print(f'\n{"#" * len(tmp_str)}\n' + tmp_str + '\n' + f'{"#" * len(tmp_str)}')
    print(f"mean width: {result['mean_width']}, std width: {result['std_width']}")
    if expected_width is not None:
        correction_factor = expected_width / result['mean_width']
        print(f"correction factor to apply to the voxel size: {correction_factor}")

    #####################################################################
    # plot an overlay of the first and last peaks for the first linecut #
    #####################################################################
    fig = plt.figure(figsize=(12, 9))
    # area around the first peak
    ax0 = plt.subplot(121)
    ind_start, ind_stop = util.find_nearest(result[f'pixel {points[0]}']['distance'], fit_roi[0][0])
    x_axis = result[f'pixel {points[0]}']['distance'][ind_start:ind_stop+1]
    ax0.plot(x_axis, result[f'pixel {points[0]}']['cut'][ind_start:ind_stop+1], '-r')
    params_first = result[f'pixel {points[0]}'][f'roi {fit_roi[0][0]}']
    fit_first = util.function_lmfit(params=params_first, x_axis=x_axis, distribution='pseudovoigt')
    ax0.plot(x_axis, fit_first, '.b')

    ax1 = plt.subplot(122)
    ind_start, ind_stop = util.find_nearest(result[f'pixel {points[0]}']['distance'], fit_roi[0][-1])
    x_axis = result[f'pixel {points[0]}']['distance'][ind_start:ind_stop+1]
    ax1.plot(x_axis, result[f'pixel {points[0]}']['cut'][ind_start:ind_stop+1], '-r')
    params_last = result[f'pixel {points[0]}'][f'roi {fit_roi[0][-1]}']
    fit_last = util.function_lmfit(params=params_last, x_axis=x_axis, distribution='pseudovoigt')
    ax1.plot(x_axis, fit_last, '.b')

#############################
# print and save the result #
#############################
if debug:
    tmp_str = 'output dictionnary'
    print(f'\n{"#" * len(tmp_str)}\n' + tmp_str + '\n' + f'{"#" * len(tmp_str)}')
    pprint(result, indent=2)
# if debug:
#     print('output dictionary:\n', json.dumps(result, cls=util.CustomEncoder, indent=4))
#
# with open(savedir+'SEM_calib' + comment + '.json', 'w', encoding='utf-8') as file:
#     json.dump(result, file, cls=util.CustomEncoder, ensure_ascii=False, indent=4)

plt.ioff()
plt.show()
print('')
