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
sys.path.append('D:/myscripts/bcdi/')
import bcdi.postprocessing.postprocessing_utils as pu
import bcdi.graph.graph_utils as gu

helptext = """
Calculate the mean and variance of the strain, for all voxels in the support
"""

scan = 1301  # spec scan number
root_folder = "D:/data/SIXS_2019_Ni/"
sample_name = "S"  # "S"
datadir = root_folder + sample_name + str(scan) + "/pynxraw/"
strain_range = 0.001  # for plots
support_threshold = 0.6  # threshold for support determination
use_bulk = False  # True to use the bulk array as support,
# if False it will use support_threshold on the modulus to define the support
debug = True  # True to see data plots
##########################
# end of user parameters #
##########################

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

######################
# define the support #
######################
if use_bulk:
    try:
        bulk = npzfile['bulk']
    except KeyError:
        print('Bulk is not of key of the npz file')
        print('Using the modulus and support_threshold to define the bulk')
        amp = npzfile['amp']
        bulk = pu.find_bulk(amp=amp, support_threshold=support_threshold, method='threshold')
        if debug:
            gu.multislices_plot(amp, sum_frames=False, title='Amplitude', plot_colorbar=True, cmap=my_cmap,
                                is_orthogonal=True, reciprocal_space=False)
        del amp
    nz, ny, nx = bulk.shape
    support = bulk
else:  # use amplitude
    print('Using the modulus and support_threshold to define the support')
    amp = npzfile['amp']
    nz, ny, nx = amp.shape
    support = np.ones((nz, ny, nx))
    support[abs(amp) < support_threshold * abs(amp).max()] = 0
    if debug:
        gu.multislices_plot(amp, sum_frames=False, title='Amplitude', plot_colorbar=True, cmap=my_cmap,
                            is_orthogonal=True, reciprocal_space=False)
    del amp

strain = npzfile['strain']
strain[support == 0] = 0
print('Data size: ({:d},{:d},{:d})'.format(nz, ny, nx))

if debug:
    gu.multislices_plot(strain, sum_frames=False, title='Strain', plot_colorbar=True,
                        vmin=-strain_range, vmax=strain_range, cmap=my_cmap, is_orthogonal=True, reciprocal_space=False)

    gu.multislices_plot(support, sum_frames=False, title='Support', plot_colorbar=True,
                        vmin=0, vmax=1, cmap=my_cmap, is_orthogonal=True, reciprocal_space=False)

#####################################################################
# calculate the mean, variance and RMS of the strain on the support #
#####################################################################
mean_strain = strain[np.nonzero(support)].mean()
var_strain = strain[np.nonzero(support)].var()
rms_strain = np.sqrt(np.mean(np.ndarray.flatten(strain[np.nonzero(support)])**2))
print('Mean strain = ', str('{:.4e}'.format(mean_strain)).replace('.', ','),
      '\nVariance strain = ', str('{:.4e}'.format(var_strain)).replace('.', ','),
      '\nRMS strain = ', str('{:.4e}'.format(rms_strain)).replace('.', ','))

plt.ioff()
plt.show()
