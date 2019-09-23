# -*- coding: utf-8 -*-
# Calculate the mean and variance of the strain, for all voxels in the support

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import tkinter as tk
from tkinter import filedialog
import gc
import sys
sys.path.append('//win.desy.de/home/carnisj/My Documents/myscripts/bcdi/')
import bcdi.postprocessing.postprocessing_utils as pu

scan = 2227
datadir = 'D:/data/PtRh/ArCOO2(102x92x140)/'  # "D:/review paper/BCDI_isosurface/S"+str(scan)+"/"
strain_range = 0.001  # for plots
support_threshold = 0.55  # threshold for support determination
use_bulk = True
flag_plot = 1  # 1 to show plots of data
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
if use_bulk:
    try:
        bulk = npzfile['bulk']
    except KeyError:
        amp = npzfile['amp']
        bulk = pu.find_bulk(amp=amp, support_threshold=support_threshold, method='threshold')
        del amp
        gc.collect()
    nz, ny, nx = bulk.shape
    support = bulk
else:  # use amplitude
    amp = npzfile['amp']
    nz, ny, nx = amp.shape
    support = np.ones((nz, ny, nx))
    support[abs(amp) < support_threshold * abs(amp).max()] = 0

print("Initial data size: (", nz, ',', ny, ',', nx, ')')

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
print('Mean strain = ', str('{:.4e}'.format(mean_strain)).replace('.', ','),
      # '\nVariance strain = ', str('{:.4e}'.format(var_strain)),
      '\nRMS strain = ', str('{:.4e}'.format(rms_strain)).replace('.', ','))

plt.ioff()
plt.show()
