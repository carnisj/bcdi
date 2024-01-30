#!/usr/bin/env python3

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import sys
import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

import bcdi.postprocessing.postprocessing_utils as pu

helptext = """
Apodization applied directly on reciprocal space data, using a 3d Gaussian, Tukey or
Blackman window.
"""

scan = 2227
datadir = (
    "G:/review paper/BCDI_isosurface/S"
    + str(scan)
    + "/simu/crop400phase/pre_apod_blackman/"
)
comment = "_blackman"
debug = True

tick_direction = "out"  # 'out', 'in', 'inout'
tick_length = 6  # in plots
tick_width = 2  # in plots

window_type = "blackman"  # 'normal' or 'tukey' or 'blackman'
#############################
# parameters for a gaussian #
#############################
mu = np.array([0.0, 0.0, 0.0])
sigma = np.array([0.30, 0.30, 0.30])
covariance = np.diag(sigma**2)
################################
# parameter for a tukey window #
################################
alpha = np.array([0.70, 0.70, 0.70])  # shape parameter of the tukey window
##################################
# end of user-defined parameters #
##################################
plt.ion()
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    initialdir=datadir, filetypes=[("NPZ", "*.npz"), ("NPY", "*.npy")]
)
data = np.load(file_path)["data"]
nbz, nby, nbx = data.shape
print(data.max())
maxdata = data.max()

plt.figure()
plt.imshow(np.log10(data.sum(axis=0)), vmin=0, vmax=6)
plt.colorbar()
plt.title("Initial diffraction pattern")
plt.pause(0.1)

if window_type == "normal":
    comment = comment + "normal"
    grid_z, grid_y, grid_x = np.meshgrid(
        np.linspace(-1, 1, nbz),
        np.linspace(-1, 1, nby),
        np.linspace(-1, 1, nbx),
        indexing="ij",
    )
    window = multivariate_normal.pdf(
        np.column_stack([grid_z.flat, grid_y.flat, grid_x.flat]),
        mean=mu,
        cov=covariance,
    )
    window = window.reshape((nbz, nby, nbx))
elif window_type == "tukey":
    comment = comment + "_tukey"
    window = pu.tukey_window(data.shape, alpha=alpha)
elif window_type == "blackman":
    comment = comment + "_blackman"
    window = pu.blackman_window(data.shape)
else:
    print("invalid window type")
    sys.exit()

if debug:
    fig, ax0 = plt.subplots(1, 1)
    plt.imshow(window[:, :, nbx // 2], vmin=0, vmax=window.max())
    plt.title("Window at middle frame")

    fig, ax0 = plt.subplots(1, 1)
    plt.plot(window[nbz // 2, nby // 2, :])
    plt.plot(window[:, nby // 2, nbx // 2])
    plt.plot(window[nbz // 2, :, nbx // 2])
    plt.title("Window linecuts at array center")

    #########################################################
    # the plot below compare different appodization windows #
    #########################################################
    # window2 = pu.blackman_window(data.shape)
    # window3 = pu.tukey_window(data.shape, alpha=alpha)
    # fig, ax = plt.subplots(1, 1)
    # plt.plot(window2[nbz // 2, nby // 2, :], '.r')
    # plt.plot(window3[nbz // 2, nby // 2, :], 'sb')
    # ax.tick_params(labelbottom=False, labelleft=False, direction=tick_direction,
    # length=tick_length, width=tick_width)
    # plt.savefig(datadir + 'windows.png', bbox_inches="tight")
    # ax.tick_params(labelbottom=True, labelleft=True, direction=tick_direction,
    # length=tick_length, width=tick_width)
    # ax.spines['right'].set_linewidth(1.5)
    # ax.spines['left'].set_linewidth(1.5)
    # ax.spines['top'].set_linewidth(1.5)
    # ax.spines['bottom'].set_linewidth(1.5)
    # plt.savefig(datadir + 'windows_labels.png', bbox_inches="tight")
new_data = np.multiply(data, window)
new_data = new_data * maxdata / new_data.max()

print(new_data.max())
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(np.log10(new_data.sum(axis=0)), vmin=0, vmax=6)
plt.colorbar()
plt.title("Apodized diffraction pattern")
plt.subplot(1, 2, 2)
plt.imshow((new_data - data).sum(axis=0))
plt.colorbar()
plt.title("(Apodized - initial) diffraction pattern")
plt.pause(0.1)

np.savez_compressed(datadir + comment + ".npz", data=new_data)

plt.ioff()
plt.show()
