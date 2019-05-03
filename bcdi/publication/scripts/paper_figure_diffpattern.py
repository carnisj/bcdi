# -*- coding: utf-8 -*-
"""
Template for making figure for paper
Open an npz file (3D diffraction pattern) and save individual figures
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import tkinter as tk
from tkinter import filedialog
from scipy.ndimage.measurements import center_of_mass

scan = 978  # spec scan number
datadir = "C:/Users/carnis/Work Folders/Documents/data/HC3207/SN"+str(scan)+"/pynxraw/"
savedir = "C:/Users/carnis/Work Folders/Documents/data/HC3207/SN"+str(scan)+"/figures/"
colorbar_range = [0, 5]  # [vmin, vmax] log scale in photon counts
save_YZ = 1  # 1 to save the strain in YZ plane
save_XZ = 0  # 1 to save the strain in XZ plane
save_XY = 0  # 1 to save the strain in XY plane
save_sum = 0  # 1 to save the summed diffraction pattern in the detector, 0 to save the central slice only
save_colorbar = 1  # to save the colorbar
zoom_halfwidth_XY = 25  # number of pixels to crop around the Bragg peak in the detector plane, put 0 if no zoom
zoom_halfwidth_Z = 25  # number of pixels to crop around the Bragg peak along the rocking angle, put 0 if no zoom
comment = '_' + str(colorbar_range)  # should start with _
if zoom_halfwidth_XY != 0:
    comment = comment + '_zoom_' + str(zoom_halfwidth_XY)
######################################
# define a colormap
cdict = {'red':  ((0.0, 1.0, 1.0),
                  (0.11, 0.0, 0.0),
                  (0.36, 0.0, 0.0),
                  (0.62, 1.0, 1.0),
                  (0.87, 1.0, 1.0),
                  (1.0, 0.0, 0.0)),
         'green': ((0.0, 1.0, 1.0),
                   (0.11, 0.0, 0.0),
                   (0.36, 1.0, 1.0),
                   (0.62, 1.0, 1.0),
                   (0.87, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),
         'blue': ((0.0, 1.0, 1.0),
                  (0.11, 1.0, 1.0),
                  (0.36, 1.0, 1.0),
                  (0.62, 0.0, 0.0),
                  (0.87, 0.0, 0.0),
                  (1.0, 0.0, 0.0))}
my_cmap = LinearSegmentedColormap('my_colormap', cdict, 256)

#######################################
plt.ion()
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(initialdir=datadir, filetypes=[("NPZ", "*.npz")])
npzfile = np.load(file_path)
rawdata = npzfile['data']  # bulk is the amplitude minus the surface voxel layer were the strain is not defined
numz, numy, numx = rawdata.shape
zcom, ycom, xcom = center_of_mass(rawdata)
zcom, ycom, xcom = [int(np.rint(zcom)), int(np.rint(ycom)), int(np.rint(xcom))]
print("Initial data size: (", numz, ',', numy, ',', numx, ')')

if save_sum == 1:
    comment = comment + '_sum'
    if save_XY == 1:
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
        if save_colorbar == 1:
            plt.colorbar(plt0, ax=ax0)
            plt.savefig(savedir + 'diffpattern' + comment + '_XY_colorbar.png', bbox_inches="tight")

    if save_XZ == 1:
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
        if save_colorbar == 1:
            plt.colorbar(plt0, ax=ax0)
            plt.savefig(savedir + 'diffpattern' + comment + '_XZ_colorbar.png', bbox_inches="tight")

    if save_YZ == 1:
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
        if save_colorbar == 1:
            plt.colorbar(plt0, ax=ax0)
            plt.savefig(savedir + 'diffpattern' + comment + '_YZ_colorbar.png', bbox_inches="tight")
else:
    if save_XY == 1:
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
        if save_colorbar == 1:
            plt.colorbar(plt0, ax=ax0)
            plt.savefig(savedir + 'diffpattern' + comment + '_XY_colorbar.png', bbox_inches="tight")

    if save_XZ == 1:
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
        if save_colorbar == 1:
            plt.colorbar(plt0, ax=ax0)
            plt.savefig(savedir + 'diffpattern' + comment + '_XZ_colorbar.png', bbox_inches="tight")

    if save_YZ == 1:
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
        if save_colorbar == 1:
            plt.colorbar(plt0, ax=ax0)
            plt.savefig(savedir + 'diffpattern' + comment + '_YZ_colorbar.png', bbox_inches="tight")
plt.ioff()
plt.show()
