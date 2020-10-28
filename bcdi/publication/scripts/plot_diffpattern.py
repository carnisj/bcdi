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
save_YZ = False  # True to save the strain in YZ plane
save_XZ = False  # True to save the strain in XZ plane
save_XY = True  # True to save the strain in XY plane
save_sum = False  # True to save the summed diffraction pattern in the detector, 0 to save the central slice only
zoom_halfwidth_XY = 20  # 25  # number of pixels to crop around the Bragg peak in the detector plane, put 0 if no zoom
zoom_halfwidth_Z = 20  # 25  # number of pixels to crop around the Bragg peak along the rocking angle, put 0 if no zoom
comment = '_' + str(colorbar_range)  # should start with _
if zoom_halfwidth_XY != 0:
    comment = comment + '_zoom_' + str(zoom_halfwidth_XY)
##################################
# end of user-defined parameters #
##################################

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
rawdata, _ = util.load_file(file_path)
numz, numy, numx = rawdata.shape
zcom, ycom, xcom = center_of_mass(rawdata)
zcom, ycom, xcom = [int(np.rint(zcom)), int(np.rint(ycom)), int(np.rint(xcom))]
print('Initial data shape:', rawdata.shape)

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

############################
# plot the different views #
############################
if save_sum:
    comment = comment + '_sum'
    if save_XY:
        data = np.copy(rawdata)
        data = data.sum(axis=0)
        fig, ax0 = plt.subplots(1, 1)
        if zoom_halfwidth_XY != 0:
            plt0 = ax0.imshow(np.log10(data[ycom-zoom_halfwidth_XY:ycom+zoom_halfwidth_XY,
                                            xcom-zoom_halfwidth_XY:xcom+zoom_halfwidth_XY]),
                              cmap=my_cmap, vmin=colorbar_range[0], vmax=colorbar_range[1])
        else:
            plt0 = ax0.imshow(np.log10(data), cmap=my_cmap, vmin=colorbar_range[0], vmax=colorbar_range[1])
        ax0.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labelleft=False)
        plt.savefig(savedir + 'diffpattern' + comment + '_XY.png', bbox_inches="tight")
        plt.colorbar(plt0, ax=ax0)
        plt.savefig(savedir + 'diffpattern' + comment + '_XY_colorbar.png', bbox_inches="tight")

    if save_XZ:
        data = np.copy(rawdata)
        data = data.sum(axis=1)
        fig, ax0 = plt.subplots(1, 1)
        if zoom_halfwidth_XY != 0:
            plt0 = ax0.imshow(np.log10(data[zcom-zoom_halfwidth_Z:zcom+zoom_halfwidth_Z,
                                            xcom-zoom_halfwidth_XY:xcom+zoom_halfwidth_XY]),
                              cmap=my_cmap, vmin=colorbar_range[0], vmax=colorbar_range[1])
        else:
            plt0 = ax0.imshow(np.log10(data), cmap=my_cmap, vmin=colorbar_range[0], vmax=colorbar_range[1])
        ax0.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labelleft=False)
        plt.savefig(savedir + 'diffpattern' + comment + '_XZ.png', bbox_inches="tight")
        plt.colorbar(plt0, ax=ax0)
        plt.savefig(savedir + 'diffpattern' + comment + '_XZ_colorbar.png', bbox_inches="tight")

    if save_YZ:
        data = np.copy(rawdata)
        data = data.sum(axis=2)
        fig, ax0 = plt.subplots(1, 1)
        if zoom_halfwidth_XY != 0:
            plt0 = ax0.imshow(np.log10(data[zcom-zoom_halfwidth_Z:zcom+zoom_halfwidth_Z,
                                            ycom-zoom_halfwidth_XY:ycom+zoom_halfwidth_XY]),
                              cmap=my_cmap, vmin=colorbar_range[0], vmax=colorbar_range[1])
        else:
            plt0 = ax0.imshow(np.log10(data), cmap=my_cmap, vmin=colorbar_range[0], vmax=colorbar_range[1])
        ax0.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labelleft=False)
        plt.savefig(savedir + 'diffpattern' + comment + '_YZ.png', bbox_inches="tight")
        plt.colorbar(plt0, ax=ax0)
        plt.savefig(savedir + 'diffpattern' + comment + '_YZ_colorbar.png', bbox_inches="tight")
else:
    if save_XY:
        fig, ax0 = plt.subplots(1, 1)
        if zoom_halfwidth_XY != 0:
            plt0 = ax0.imshow(np.log10(rawdata[zcom, ycom-zoom_halfwidth_XY:ycom+zoom_halfwidth_XY,
                                               xcom-zoom_halfwidth_XY:xcom+zoom_halfwidth_XY]),
                              cmap=my_cmap, vmin=colorbar_range[0], vmax=colorbar_range[1])
        else:
            plt0 = ax0.imshow(np.log10(rawdata[zcom, :, :]),
                              cmap=my_cmap, vmin=colorbar_range[0], vmax=colorbar_range[1])
        ax0.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labelleft=False)
        plt.savefig(savedir + 'diffpattern' + comment + '_XY.png', bbox_inches="tight")
        plt.colorbar(plt0, ax=ax0)
        plt.savefig(savedir + 'diffpattern' + comment + '_XY_colorbar.png', bbox_inches="tight")

    if save_XZ:
        fig, ax0 = plt.subplots(1, 1)
        if zoom_halfwidth_XY != 0:
            plt0 = ax0.imshow(np.log10(rawdata[zcom - zoom_halfwidth_Z:zcom + zoom_halfwidth_Z, ycom,
                                               xcom - zoom_halfwidth_XY:xcom + zoom_halfwidth_XY]),
                              cmap=my_cmap, vmin=colorbar_range[0], vmax=colorbar_range[1])
        else:
            plt0 = ax0.imshow(np.log10(rawdata[:, ycom, :]),
                              cmap=my_cmap, vmin=colorbar_range[0], vmax=colorbar_range[1])
        ax0.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labelleft=False)
        plt.savefig(savedir + 'diffpattern' + comment + '_XZ.png', bbox_inches="tight")
        plt.colorbar(plt0, ax=ax0)
        plt.savefig(savedir + 'diffpattern' + comment + '_XZ_colorbar.png', bbox_inches="tight")

    if save_YZ:
        fig, ax0 = plt.subplots(1, 1)
        if zoom_halfwidth_XY != 0:
            plt0 = ax0.imshow(np.log10(rawdata[zcom - zoom_halfwidth_Z:zcom + zoom_halfwidth_Z,
                                               ycom - zoom_halfwidth_XY:ycom + zoom_halfwidth_XY, xcom]),
                              cmap=my_cmap, vmin=colorbar_range[0], vmax=colorbar_range[1])
        else:
            plt0 = ax0.imshow(np.log10(rawdata[:, :, xcom]),
                              cmap=my_cmap, vmin=colorbar_range[0], vmax=colorbar_range[1])
        ax0.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labelleft=False)
        plt.savefig(savedir + 'diffpattern' + comment + '_YZ.png', bbox_inches="tight")
        plt.colorbar(plt0, ax=ax0)
        plt.savefig(savedir + 'diffpattern' + comment + '_YZ_colorbar.png', bbox_inches="tight")
plt.ioff()
plt.show()
