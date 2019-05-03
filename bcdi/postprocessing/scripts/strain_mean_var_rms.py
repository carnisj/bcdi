# -*- coding: utf-8 -*-
# Calculate the mean and variance of the strain, for all voxels in the support

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import tkinter as tk
from tkinter import filedialog
import gc

scan = 2227
datadir = "C:/Users/carnis/Work Folders/Documents/data/CH4760_Pt/S"+str(scan)+"/simu/crop400phase/no_apodization/avg1/"  # no_apodization"  # apodize_during_phasing # apodize_postprocessing
strain_range = 0.001  # for plots
support_threshold = 0.73  # threshold for support determination
flag_plot = 0  # 1 to show plots of data
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
print('Opening ', file_path)
npzfile = np.load(file_path)
bulk = npzfile['bulk']
nz, ny, nx = bulk.shape
print("Initial data size: (", nz, ',', ny, ',', nx, ')')
support = np.ones((nz, ny, nx))
support[abs(bulk) < support_threshold * abs(bulk).max()] = 0

########################################
# plot data
########################################
if flag_plot == 1:
    plt.figure(figsize=(18, 15))
    plt.subplot(2, 2, 1)
    plt.imshow(bulk[:, :, nx//2], cmap=my_cmap)
    plt.colorbar()
    plt.axis('scaled')
    plt.title('Amplitude at middle frame in YZ')
    plt.subplot(2, 2, 2)
    plt.imshow(bulk[:, ny//2, :], cmap=my_cmap)
    plt.colorbar()
    plt.axis('scaled')
    plt.title('Amplitude at middle frame in XZ')
    plt.subplot(2, 2, 3)
    plt.imshow(bulk[nz//2, :, :], cmap=my_cmap)
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.axis('scaled')
    plt.title('Amplitude at middle frame in XY')
    plt.pause(0.1)
del bulk
gc.collect()

strain = npzfile['strain']
if flag_plot == 1:
    plt.figure(figsize=(18, 15))
    plt.subplot(2, 2, 1)
    plt.imshow(strain[:, :, nx//2], vmin=-strain_range, vmax=strain_range, cmap=my_cmap)
    plt.colorbar()
    plt.axis('scaled')
    plt.title("Strain at middle frame in YZ")
    plt.subplot(2, 2, 2)
    plt.imshow(strain[:, ny//2, :], vmin=-strain_range, vmax=strain_range, cmap=my_cmap)
    plt.colorbar()
    plt.title("Strain at middle frame in XZ")
    plt.axis('scaled')
    plt.subplot(2, 2, 3)
    plt.imshow(strain[nz//2, :, :], vmin=-strain_range, vmax=strain_range, cmap=my_cmap)
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.title("Strain at middle frame in XY")
    plt.axis('scaled')
    plt.pause(0.1)

    plt.figure(figsize=(18, 15))
    plt.subplot(2, 2, 1)
    plt.imshow(support[:, :, nx//2], vmin=0, vmax=1, cmap=my_cmap)
    plt.colorbar()
    plt.axis('scaled')
    plt.title("Support at middle frame in YZ")
    plt.subplot(2, 2, 2)
    plt.imshow(support[:, ny//2, :], vmin=0, vmax=1, cmap=my_cmap)
    plt.colorbar()
    plt.title("Support at middle frame in XZ")
    plt.axis('scaled')
    plt.subplot(2, 2, 3)
    plt.imshow(support[nz//2, :, :], vmin=0, vmax=1, cmap=my_cmap)
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.title("Support at middle frame in XY")
    plt.axis('scaled')
    plt.pause(0.1)

########################################
# calculate mean and variance of the strain
########################################
mean_strain = strain[np.nonzero(support)].mean()
var_strain = strain[np.nonzero(support)].var()
rms_strain = np.sqrt(np.mean(np.ndarray.flatten(strain[np.nonzero(support)])**2))
print('Mean strain = ', str('{:.4e}'.format(mean_strain)),
      '\nVariance strain = ', str('{:.4e}'.format(var_strain)),
      '\nRMS strain = ', str('{:.4e}'.format(rms_strain)))

plt.ioff()
plt.show()
