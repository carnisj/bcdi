# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import tkinter as tk
from tkinter import filedialog
import gc
import sys
sys.path.append('//win.desy.de/home/carnisj/My Documents/myscripts/bcdi/')
import bcdi.postprocessing.postprocessing_utils as pu
import bcdi.graph.graph_utils as gu

helptext = """
Calculate the mean and variance of the strain, for all voxels in the support
"""

scan = 2227
datadir = 'D:/data/BCDI_isosurface/S2227/oversampling/real_space_interpolation/sdd_1,01/'  # "D:/review paper/BCDI_isosurface/S"+str(scan)+"/"
strain_range = 0.001  # for plots
support_threshold = 0.6  # threshold for support determination
use_bulk = False
flag_plot = True  # True to show plots of data


###################
# define colormap #
###################
bad_color = '1.0'  # white background
colormap = gu.Colormap(bad_color=bad_color)
my_cmap = colormap.cmap

#############
# load data #
#############
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

#############
# plot data #
#############
if flag_plot and not use_bulk:
    gu.multislices_plot(amp, sum_frames=False, invert_yaxis=True, title='Amplitude', plot_colorbar=True, cmap=my_cmap,
                        is_orthogonal=True, reciprocal_space=False)
    del amp
    gc.collect()

strain = npzfile['strain']
strain[support == 0] = 0

if flag_plot:
    gu.multislices_plot(strain, sum_frames=False, invert_yaxis=True, title='Strain', plot_colorbar=True,
                        vmin=-strain_range, vmax=strain_range, cmap=my_cmap, is_orthogonal=True, reciprocal_space=False)

    gu.multislices_plot(support, sum_frames=False, invert_yaxis=True, title='Support', plot_colorbar=True,
                        vmin=0, vmax=1, cmap=my_cmap, is_orthogonal=True, reciprocal_space=False)

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
