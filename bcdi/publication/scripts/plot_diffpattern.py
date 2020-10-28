# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import filedialog
from scipy.ndimage.measurements import center_of_mass
import sys
sys.path.append('D:/myscripts/bcdi/')
import bcdi.graph.graph_utils as gu
import bcdi.utils.utilities as util

helptext = """
Template for figures of the following article: 
Carnis et al. Scientific Reports 9, 17357 (2019) https://doi.org/10.1038/s41598-019-53774-2
For simulated data or experimental data, open a npz file (3D diffraction pattern) and save individual figures.
q values can be provided optionally.
"""

scan = 85  # spec scan number
root_folder = "D:/data/test_FuzzyGridder/"
sample_name = "S"
datadir = root_folder + sample_name + str(scan) + '/pynx/'
savedir = datadir
load_qvalues = True  # True to load the q values. It expects a single npz file with fieldnames 'qx', 'qy' and 'qz'
colorbar_range = [-1, 6]  # [vmin, vmax] log scale in photon counts
grey_background = True  # True to set nans to grey in the plots
save_YZ = True  # True to save the strain in YZ plane
save_XZ = True  # True to save the strain in XZ plane
save_XY = True  # True to save the strain in XY plane
save_sum = False  # True to save the summed diffraction pattern in the detector, False to save the central slice only
plot_symmetrical = False  # if False, will not use the parameter half_range
half_range = (None, None, None)  # tuple of three pixel numbers, half-range in each direction. Use None to use the
# maximum symmetrical data range along one direction e.g. [20, None, None]
comment = '_' + str(colorbar_range)  # should start with _
##################################
# end of user-defined parameters #
##################################

####################
# Check parameters #
####################
if save_sum:
    comment = comment + '_sum'
numticks_colorbar = colorbar_range[1] - colorbar_range[0] + 1
###################
# define colormap #
###################
if grey_background:
    bad_color = '0.7'
else:
    bad_color = '1.0'  # white background
colormap = gu.Colormap(bad_color=bad_color)
my_cmap = colormap.cmap

#############
# load data #
#############
plt.ion()
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(initialdir=datadir, title="Select the diffraction pattern",
                                       filetypes=[("NPZ", "*.npz")])
data, _ = util.load_file(file_path)
print('Initial data shape:', data.shape)

############################
# Check the plotting range #
############################
try:
    assert len(half_range) == 3, 'half-range should be a tuple of three pixel numbers'
except TypeError:
    raise TypeError('half-range should be a tuple of three pixel numbers')

nbz, nby, nbx = data.shape
zcom, ycom, xcom = center_of_mass(data)
zcom, ycom, xcom = int(np.rint(zcom)), int(np.rint(ycom)), int(np.rint(xcom))

plot_range = []
if plot_symmetrical:
    max_range = (min(zcom, nbz-zcom), min(zcom, nbz-zcom),
                 min(ycom, nby-ycom), min(ycom, nby-ycom),
                 min(xcom, nbx-xcom), min(xcom, nbx-xcom))  # maximum symmetric half ranges
else:
    max_range = (zcom, nbz-zcom, ycom, nby-ycom, xcom, nbx-xcom)  # asymmetric half ranges

for idx, val in enumerate(half_range):
    plot_range.append(min(val or max_range[2*idx], max_range[2*idx]))
    plot_range.append(min(val or max_range[2*idx+1], max_range[2*idx+1]))
print('Plotting symmetrical ranges:', plot_symmetrical)
print('Plotting range:', plot_range)

################################
# optionally load the q values #
################################
if load_qvalues:
    file_path = filedialog.askopenfilename(initialdir=datadir, title="Select the q values",
                                           filetypes=[("NPZ", "*.npz")])
    q_values = np.load(file_path)
    qx = q_values['qx']
    qz = q_values['qz']
    qy = q_values['qy']
    print('qx shape:', qx.shape, 'qz shape:', qz.shape, 'qy shape:', qy.shape)
    assert (*qx.shape, *qz.shape, *qy.shape) == data.shape, 'q values and data shape are incompatible'

############################
# plot the different views #
############################
if save_XY:
    fig, ax0 = plt.subplots(1, 1)
    if save_sum:
        plt0 = ax0.imshow(np.log10(data[ycom-plot_range[2]:ycom+plot_range[3],
                                        xcom-plot_range[4]:xcom+plot_range[5]]),
                          cmap=my_cmap, vmin=colorbar_range[0], vmax=colorbar_range[1])
    else:
        plt0 = ax0.imshow(np.log10(data[zcom, ycom - plot_range[2]:ycom + plot_range[3],
                                        xcom - plot_range[4]:xcom + plot_range[5]]),
                          cmap=my_cmap, vmin=colorbar_range[0], vmax=colorbar_range[1])
    ax0.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                    labelbottom=False, labelleft=False)
    plt.savefig(savedir + 'diffpattern' + comment + '_XY.png', bbox_inches="tight")
    gu.colorbar(plt0, numticks=numticks_colorbar)
    plt.savefig(savedir + 'diffpattern' + comment + '_XY_colorbar.png', bbox_inches="tight")

if save_XZ:
    fig, ax0 = plt.subplots(1, 1)
    if save_sum:
        plt0 = ax0.imshow(np.log10(data[zcom-plot_range[0]:zcom+plot_range[1],
                                        xcom-plot_range[4]:xcom+plot_range[5]]),
                          cmap=my_cmap, vmin=colorbar_range[0], vmax=colorbar_range[1])
    else:
        plt0 = ax0.imshow(np.log10(data[zcom - plot_range[0]:zcom + plot_range[1], ycom,
                                        xcom - plot_range[4]:xcom + plot_range[5]]),
                          cmap=my_cmap, vmin=colorbar_range[0], vmax=colorbar_range[1])
    ax0.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                    labelbottom=False, labelleft=False)
    plt.savefig(savedir + 'diffpattern' + comment + '_XZ.png', bbox_inches="tight")
    gu.colorbar(plt0, numticks=numticks_colorbar)
    plt.savefig(savedir + 'diffpattern' + comment + '_XZ_colorbar.png', bbox_inches="tight")

if save_YZ:
    fig, ax0 = plt.subplots(1, 1)
    if save_sum:
        plt0 = ax0.imshow(np.log10(data[zcom-plot_range[0]:zcom+plot_range[1],
                                        ycom-plot_range[2]:ycom+plot_range[3]]),
                          cmap=my_cmap, vmin=colorbar_range[0], vmax=colorbar_range[1])
    else:
        plt0 = ax0.imshow(np.log10(data[zcom - plot_range[0]:zcom + plot_range[1],
                                        ycom - plot_range[2]:ycom + plot_range[3], xcom]),
                          cmap=my_cmap, vmin=colorbar_range[0], vmax=colorbar_range[1])
    ax0.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                    labelbottom=False, labelleft=False)
    plt.savefig(savedir + 'diffpattern' + comment + '_YZ.png', bbox_inches="tight")
    gu.colorbar(plt0, numticks=numticks_colorbar)
    plt.savefig(savedir + 'diffpattern' + comment + '_YZ_colorbar.png', bbox_inches="tight")

plt.ioff()
plt.show()
