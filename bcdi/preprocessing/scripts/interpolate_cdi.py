# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np
import matplotlib.pyplot as plt
import sys
import tkinter as tk
from tkinter import filedialog
sys.path.append('D:/myscripts/bcdi/')
import bcdi.graph.graph_utils as gu
import bcdi.utils.utilities as util
import bcdi.utils.validation as valid

helptext = """
This script can be used to interpolate the intensity of masked voxels suing the centrosymmetry property of a 3D 
diffraction pattern in the forward CDI geometry. The diffraction pattern should be in an orthonormal frame with 
identical voxel sizes in all directions. The mask should be an array of integers (0 or 1) of the same shape as the 
diffraction pattern. 
"""

datadir = ''  # location of the data and mask
savedir = None  # path where to save the result, will default to datadir if None
comment = ''  # comment for the file name when saving, should start with _
origin = (12, 23, 15)  # tuple of three integers, position in pixels of the origin of reciprocal space
plot = True  # True to show plots of the data and mask, before and after the interpolation
##################################
# end of user-defined parameters #
##################################

###################
# define colormap #
###################
colormap = gu.Colormap()
my_cmap = colormap.cmap

###################################
# load experimental data and mask #
###################################
plt.ion()
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(initialdir=datadir, title="Select the diffraction pattern",
                                       filetypes=[("NPZ", "*.npz"), ("NPY", "*.npy")])
data, _ = util.load_file(file_path)
data = data.astype(float)

file_path = filedialog.askopenfilename(initialdir=datadir, title="Select the mask",
                                       filetypes=[("NPZ", "*.npz"), ("NPY", "*.npy")])
mask, _ = util.load_file(file_path)
mask = mask.astype(int)

#########################
# check some parameters #
#########################
savedir = savedir or datadir

if data.shape != mask.shape:
    raise ValueError(f'Incompatible shape for the data: {data.shape} and the mask: {mask.shape}')

if data.ndim != 3:
    raise ValueError('only 3D data is supported')

valid.valid_container(obj=origin, container_types=(tuple, list, np.ndarray), item_types=int, length=3,
                      name='interpolate_cdi.py')

nbz, nby, nbx = data.shape
# calculate the range of pixels indices covered by the data, taking into account the origin of reciprocal space
data_extent = (-origin[0], nbz-origin[0]-1, -origin[1], nby-origin[1]-1, -origin[2], nbx-origin[2]-1)

###################################################
# plot the data and mask before the interpolation #
###################################################
if plot:
    gu.multislices_plot(array=data, sum_frames=False, plot_colorbar=True, scale='log', slice_position=origin,
                        is_orthogonal=True, reciprocal_space=True, vmin=0, title='data before interpolation')
    gu.multislices_plot(array=mask, sum_frames=False, plot_colorbar=False, scale='linear', slice_position=origin,
                        is_orthogonal=True, reciprocal_space=True, vmin=0, vmax=1, title='mask before interpolation')

####################################################################################################################
# loop over masked points to see if the centrosymmetric voxel is also masked, if not copy its intensity and unmask #
####################################################################################################################
ind_z, ind_y, ind_x = np.nonzero(mask)

for idx in range(len(ind_z)):
    # calculate the position of the centrosymmetric voxel
    sym_z, sym_y, sym_x = 2 * origin[0] - ind_z[idx], 2 * origin[1] - ind_y[idx], 2 * origin[2] - ind_x[idx]

    # check if this voxel is masked. Copy its intensity if not.
    if util.in_range(point=(sym_z, sym_y, sym_x), extent=data_extent) and not mask[sym_z, sym_y, sym_x]:
        data[ind_z[idx], ind_y[idx], ind_x[idx]] = data[sym_z, sym_y, sym_x]
        mask[ind_z[idx], ind_y[idx], ind_x[idx]] = 0

##################################################
# plot the data and mask after the interpolation #
##################################################
if plot:
    gu.multislices_plot(array=data, sum_frames=False, plot_colorbar=True, scale='log', slice_position=origin,
                        is_orthogonal=True, reciprocal_space=True, vmin=0, title='data after interpolation')
    gu.multislices_plot(array=mask, sum_frames=False, plot_colorbar=False, scale='linear', slice_position=origin,
                        is_orthogonal=True, reciprocal_space=True, vmin=0, vmax=1, title='mask after interpolation')

##################################
# save the updated data and mask #
##################################
np.savez_compressed(savedir + 'centrosym_data' + comment, data=data)
np.savez_compressed(savedir + 'centrosym_data' + comment, mask=mask)
plt.ioff()
plt.show()
