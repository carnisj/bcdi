# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
from scipy.stats import pearsonr
import sys
import tkinter as tk
from tkinter import filedialog
sys.path.append('D:/myscripts/bcdi/')
import bcdi.utils.utilities as util
import bcdi.utils.validation as valid

helptext = """
This script can be used to compare the lateral sizes of an object measured by CDI and scanning electron micrscopy. Two
dictionary should be provided as input (one for each technique). The dictionary should contain the following items:
{'threshold': 1D array-like values of thresholds,
 'angles':D array-like values of angles 
 'ang_width_threshold': 2D array-like values (one row for each threshold, the row is the width vs angle of the linecut)}

These dictionaries can be produced by the script angular_profile.py

After aligning the traces of the width vs angle (e.g. if the object was slightly rotated in one of the measurements),
the traces are overlaid in order to determine which threshold is correct.     
"""

datadir = "D:/data/P10_2nd_test_isosurface_Dec2020/data_nanolab/dataset_1_newpsf/result/linecuts/"
# "D:/data/P10_2nd_test_isosurface_Dec2020/data_nanolab/AFM-SEM/P10 beamtime P2 particle size SEM/linecuts_P2_001a/"
# "D:/data/P10_2nd_test_isosurface_Dec2020/data_nanolab/dataset_1_newpsf/result/linecuts/"
# "D:/data/P10_2nd_test_isosurface_Dec2020/data_nanolab/dataset_1/PtNP1_00128/result/"  # data folder
savedir = datadir + 'comparison_SEM/valid_range/'
# results will be saved here, if None it will default to datadir
index_sem = 1  # index of the threshold to use for the SEM profile. Leave None to print the available thresholds.
plot_sem = 'fill'  # if 'single', will plot only index_sem, if 'fill', it will fill the area between the first
# and the last SEM thresholds in grey
comment = 'valid_range'  # string to add to the filename when saving, should start with "_"
tick_length = 10  # in plots
tick_width = 2  # in plots
##################################
# end of user-defined parameters #
##################################

#############################
# define default parameters #
#############################
colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')  # for plots
markers = ('.', 'v', '^', '<', '>', 'x', '+', 'o')  # for plots
mpl.rcParams['axes.linewidth'] = tick_width  # set the linewidth globally
validation_name = 'compare_CDI_SEM'

#########################
# check some parameters #
#########################
valid.valid_item(value=index_sem, allowed_types=int, min_included=0, allow_none=True, name=validation_name)
valid.valid_container(plot_sem, container_types=str, name=validation_name)
if plot_sem not in {'single', 'fill'}:
    raise ValueError("allowed values for plot_sem are 'single' and 'all'")
valid.valid_container(comment, container_types=str, name=validation_name)
if len(comment) != 0 and not comment.startswith('_'):
    comment = '_' + comment

savedir = savedir or datadir
pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)

########################
# load the SEM profile #
########################
plt.ion()
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(initialdir=datadir, title='select the dictionary containing SEM profiles',
                                       filetypes=[("JSON", "*.json"), ("NPZ", "*.npz")])

_, ext = os.path.splitext(file_path)
if ext == '.json':
    with open(file_path, 'r') as f:
        json_string = f.read()
    sem_dict = json.loads(json_string, object_hook=util.decode_json)
else:
    raise ValueError(f'expecting as JSON file, got a {ext}')

#########################
# load the BCDI profile #
#########################
plt.ion()
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(initialdir=datadir, title='select the dictionary containing BCDI profiles',
                                       filetypes=[("JSON", "*.json"), ("NPZ", "*.npz")])

_, ext = os.path.splitext(file_path)
if ext == '.json':
    with open(file_path, 'r') as f:
        json_string = f.read()
    bcdi_dict = json.loads(json_string, object_hook=util.decode_json)
else:
    raise ValueError(f'expecting as JSON file, got a {ext}')

try:
    if not np.all(sem_dict['angles'] == bcdi_dict['angles']):
        print('angular steps are not equal:')
        print(f"SEM angles: {sem_dict['angles']}")
        print(f"BCDI angles: {bcdi_dict['angles']}")
        raise NotImplementedError
except KeyError:
    print('missing key angles in one of the dictionary')
    raise

####################################################################
# get the angular shift between SEM and BCDI traces and align them #
####################################################################
if index_sem is None:
    print(f"thresholds SEM: {sem_dict['threshold']}")
    sys.exit()

sem_trace = sem_dict['ang_width_threshold'][index_sem]
bcdi_trace = bcdi_dict['ang_width_threshold'][0]
cross_corr = np.correlate(sem_trace - np.mean(sem_trace), bcdi_trace - np.mean(bcdi_trace), mode="full")
lags = np.arange(-sem_trace.size + 1, bcdi_trace.size)
lag = lags[np.argmax(cross_corr)]

# plot the cross-correlation
fig = plt.figure(figsize=(12, 9))
ax0 = plt.subplot(121)
line, = ax0.plot(sem_trace)
line.set_label("SEM trace")
line, = ax0.plot(bcdi_trace)
line.set_label(f"BCDI trace (threshold={bcdi_dict['threshold'][0]})")

print(f'lag between the two traces = {lag}')
print('aligning the SEM trace on the BCDI traces')
sem_trace = np.roll(sem_trace, -lag)

line, = ax0.plot(sem_trace)
line.set_label("SEM trace aligned")
ax0.set_xlabel('angle (deg)', fontsize=20)
ax0.set_ylabel('width (nm)', fontsize=20)
ax0.legend(fontsize=14)
ax1 = plt.subplot(122)
ax1.plot(cross_corr)
ax1.set_title("cross-correlated signal", fontsize=14)
plt.tight_layout()  # avoids the overlap of subplots with axes labels
fig.savefig(savedir + 'crosscorr_alignement' + comment + '.png')

##########################################################################
# plot the aligned BDCI traces for different thresholds vs the SEM trace #
##########################################################################
thres_bcdi = bcdi_dict['threshold']
angles_bcdi = bcdi_dict['angles']
correlation = np.empty(thres_bcdi.size)
residuals = np.empty(thres_bcdi.size)
fig = plt.figure(figsize=(12, 9))
ax0 = plt.subplot(111)
if plot_sem == 'single':
    line, = ax0.plot(sem_dict['angles'], sem_trace, color='k', marker='.', markersize=15, linestyle='-', linewidth=1)
    line.set_label(f"SEM thres {sem_dict['threshold'][index_sem]}")
else:  # fill area between the first and the last SEM thresholds
    ax0.fill_between(x=sem_dict['angles'],
                     y1=np.roll(sem_dict['ang_width_threshold'][0], -lag),
                     y2=np.roll(sem_dict['ang_width_threshold'][-1], -lag), color='grey')

for idx, thres in enumerate(thres_bcdi, start=0):
    bcdi_trace = bcdi_dict['ang_width_threshold'][idx]
    cross_corr = np.correlate(sem_trace - np.mean(sem_trace), bcdi_trace - np.mean(bcdi_trace), mode="full")
    lag = lags[np.argmax(cross_corr)]
    sem_trace = np.roll(sem_trace, -lag)
    print(f'bcdi trace {idx}, threshold = {thres}, lag = {lag}')

    correlation[idx] = pearsonr(sem_trace, bcdi_trace)[0]
    residuals[idx] = (np.subtract(sem_trace, bcdi_trace)**2).sum()
    line, = ax0.plot(angles_bcdi, bcdi_trace, color=colors[idx % len(colors)],
                     marker=markers[idx % len(markers)],
                     fillstyle='none', markersize=9,
                     linestyle='-', linewidth=1)
    line.set_label(f'threshold {thres}')

ax0.tick_params(labelbottom=False, labelleft=False, direction='out', length=tick_length, width=tick_width,
                labelsize=16)
fig.savefig(savedir + 'compa_width_vs_ang' + comment + '.png')
ax0.set_xlabel('angle (deg)', fontsize=20)
ax0.set_ylabel('width (nm)', fontsize=20)
ax0.tick_params(labelbottom=True, labelleft=True, axis='both', which='major', labelsize=16)
ax0.legend(fontsize=14)
if plot_sem == 'fill':
    ax0.set_title(f"fill between SEM_thres {sem_dict['threshold'][0]} and {sem_dict['threshold'][-1]}")
fig.savefig(savedir + 'compa_width_vs_ang' + comment + '_legend.png')

##############################################################################################################
# Plot the evolution of the Pearson correlation coefficient and squared residuals depending on the threshold #
##############################################################################################################
min_thres_idx = np.unravel_index(residuals.argmin(), shape=residuals.shape)[0]
fig = plt.figure(figsize=(12, 9))
ax0 = plt.subplot(111)
ax0.plot(thres_bcdi, correlation, color='b', marker='.', fillstyle='none', markersize=10, markeredgewidth=2,
         linestyle='solid', linewidth=2)
ax0.tick_params(labelbottom=False, labelleft=False, direction='out', length=tick_length, width=tick_width,
                labelsize=16)

ax1 = ax0.twinx()
ax1.plot(thres_bcdi, residuals, color='r', marker='v', fillstyle='none', markersize=10, markeredgewidth=2,
         linestyle='dashed', linewidth=2)
ax1.tick_params(labelbottom=False, labelright=False, direction='out', length=tick_length, width=tick_width,
                labelsize=16)
fig.savefig(savedir + 'Pearson_vs_threshold' + comment + '.png')
ax0.set_xlabel('threshold', fontsize=20)
ax0.set_ylabel('Pearson correlation coeff.', fontsize=20)
ax1.set_ylabel('Squared residuals', fontsize=20)
ax0.tick_params(labelbottom=True, labelleft=True, axis='both', which='major', labelsize=16)
ax1.tick_params(labelright=True, axis='both', which='major', labelsize=16)
fig.tight_layout()
fig.text(0.4, 0.4, f'Min SR at threshold={thres_bcdi[min_thres_idx]}', size=12)
fig.savefig(savedir + 'Pearson_vs_threshold' + comment + '_legend.png')

plt.ioff()
plt.show()
