# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import sys
sys.path.append('D:/myscripts/bcdi/')
import bcdi.graph.graph_utils as gu
import bcdi.utils.utilities as util

helptext = """
Graphical interface to visualize 2D slices through a 3D dataset in an easy way.
"""

datadir = "D:/data/CH5309/S614/pynxraw/"
savedir = "D:/data/CH5309/S614/test/"
scale = 'linear'  # 'linear' or 'log', scale of the 2D plots
field = 'angle'  # data field name. Leave it to None for default.
# It will take abs() for 'modulus', numpy.angle() for 'angle'
half_range = 1  # colorbar range will be [-half_range half-range]
grey_background = True
background_plot = '0.5'  # in level of grey in [0,1], 0 being dark. For visual comfort
vmin = None  # lower boundary of the colorbar, leave it to None for default
vmax = None  # higher boundary of the colorbar, leave it to None for default


def press_key(event):
    """
    Interact with a plot for masking parasitic diffraction intensity or detector gaps

    :param event: button press event
    :return: updated controls
    """
    global data, dim, idx, max_colorbar, scale, savedir

    try:
        max_colorbar, idx, exit_flag = \
            gu.loop_thru_scan(key=event.key, data=data, figure=fig_loop, scale=scale, dim=dim, idx=idx, vmin=0, vmax=max_colorbar,
                              savedir=savedir)

        if exit_flag:
            plt.close(fig_loop)

    except AttributeError:  # mouse pointer out of axes
        pass


###################
# define colormap #
###################
if grey_background:
    bad_color = '0.7'
else:
    bad_color = '1.0'  # white background
colormap = gu.Colormap(bad_color=bad_color)
my_cmap = colormap.cmap

if field == 'angle' or field == 'modulus':
    scale = 'linear'
    
#############
# load data #
#############
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(initialdir=datadir, filetypes=[("NPZ", "*.npz"), ("NPY", "*.npy"),
                                                                      ("CXI", "*.cxi"), ("HDF5", "*.h5")])
nbfiles = len(file_path)
plt.ion()

data, extension = util.load_file(file_path, fieldname=field)

if vmin is None:
    vmin = data.min()
if vmax is None:
    vmax = data.max()
#########################
# loop through the data #
#########################
plt.ioff()
nz, ny, nx = np.shape(data)
max_colorbar = vmax

# in XY
dim = 0
fig_loop = plt.figure()
idx = 0
original_data = np.copy(data)
if scale == 'linear':
    plt.imshow(data[idx, :, :], vmin=vmin, vmax=max_colorbar)
else:  # 'log'
    plt.imshow(np.log10(data[idx, :, :]), vmin=vmin, vmax=max_colorbar)
plt.title("Frame " + str(idx + 1) + "/" + str(nz) + "\n"
          "q quit ; u next frame ; d previous frame ; p unzoom\n"
          "right darker ; left brighter ; r save 2D frame")
plt.connect('key_press_event', press_key)
fig_loop.set_facecolor(background_plot)
plt.show()
del dim, fig_loop

# in XZ
dim = 1
fig_loop = plt.figure()
idx = 0
if scale == 'linear':
    plt.imshow(data[:, idx, :], vmin=vmin, vmax=max_colorbar)
else:  # 'log'
    plt.imshow(np.log10(data[:, idx, :]), vmin=vmin, vmax=max_colorbar)
plt.title("Frame " + str(idx + 1) + "/" + str(ny) + "\n"
          "q quit ; u next frame ; d previous frame ; p unzoom\n"
          "right darker ; left brighter ; r save 2D frame")
plt.connect('key_press_event', press_key)
fig_loop.set_facecolor(background_plot)
plt.show()
del dim, fig_loop

# in YZ
dim = 2
fig_loop = plt.figure()
idx = 0
if scale == 'linear':
    plt.imshow(data[:, :, idx], vmin=vmin, vmax=max_colorbar)
else:  # 'log'
    plt.imshow(np.log10(data[:, :, idx]), vmin=vmin, vmax=max_colorbar)
plt.title("Frame " + str(idx + 1) + "/" + str(nx) + "\n"
          "q quit ; u next frame ; d previous frame ; p unzoom\n"
          "right darker ; left brighter ; r save 2D frame")
plt.connect('key_press_event', press_key)
fig_loop.set_facecolor(background_plot)
plt.show()
