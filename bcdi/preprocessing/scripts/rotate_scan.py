# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import tkinter as tk
from tkinter import filedialog
import sys
import gc
sys.path.append('D:/myscripts/bcdi/')
import bcdi.graph.graph_utils as gu
import bcdi.utils.utilities as util

helptext = """
Rotate a 3D reciprocal space map around some axis. The data is expected to be in an orthonormal frame.
"""

scan = 22  # scan number
datadir = 'D:/data/P10_March2020_CDI/data/ht_pillar3/test/'  # S' + str(scan) + '/pynxraw/'
tilt = 5*3.1415/180  # rotation angle in radians to be applied counter-clockwise around rotation_axis
rotation_axis = (0, 0, 1)  # in the order (x y z), z axis 0, y axis 1, x axis 0
origin = (150, 150, 150)  # position in voxels of the origin of the reciprocal space (origin of the rotation)
save = False  # True to save the rotated data
comment = ''  # should start with _, comment for the filename when saving the rotated data
##################################
# end of user-defined parameters #
##################################

#########################################
# load the data and the mask (optional) #
#########################################
plt.ion()
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(initialdir=datadir,
                                       title="Select 3D data", filetypes=[("NPZ", "*.npz")])
data, _ = util.load_file(file_path)
nbz, nby, nbx = data.shape
print('data shape:', data.shape)

try:
    file_path = filedialog.askopenfilename(initialdir=datadir,
                                           title="Select 3D mask", filetypes=[("NPZ", "*.npz")])
    mask, _ = util.load_file(file_path)
    skip_mask = False
except ValueError:
    print('skip mask')
    mask = None
    skip_mask = True

gu.multislices_plot(data, sum_frames=True, scale='log', plot_colorbar=True,
                    title='S' + str(scan) + '\n Data before rotation', vmin=0,
                    reciprocal_space=True, is_orthogonal=True)
if not skip_mask:
    gu.multislices_plot(mask, sum_frames=True, scale='linear', plot_colorbar=True,
                        title='S' + str(scan) + '\n Mask before rotation', vmin=0,
                        reciprocal_space=True, is_orthogonal=True)

###################
# rotate the data #
###################
# define the rotation matrix in the order (x, y, z)
rotation_matrix = np.array([[np.cos(tilt) + (1-np.cos(tilt)) * rotation_axis[0]**2,
                             rotation_axis[0]*rotation_axis[1]*(1-np.cos(tilt))-rotation_axis[2]*np.sin(tilt),
                             rotation_axis[0]*rotation_axis[2]*(1-np.cos(tilt))+rotation_axis[1]*np.sin(tilt)],
                            [rotation_axis[1]*rotation_axis[0]*(1-np.cos(tilt))+rotation_axis[2]*np.sin(tilt),
                             np.cos(tilt) + (1-np.cos(tilt)) * rotation_axis[1]**2,
                             rotation_axis[1]*rotation_axis[2]*(1-np.cos(tilt))-rotation_axis[0]*np.sin(tilt)],
                            [rotation_axis[2]*rotation_axis[0]*(1-np.cos(tilt))-rotation_axis[1]*np.sin(tilt),
                             rotation_axis[2]*rotation_axis[1]*(1-np.cos(tilt))+rotation_axis[0]*np.sin(tilt),
                             np.cos(tilt) + (1-np.cos(tilt)) * rotation_axis[2]**2]])

transfer_matrix = rotation_matrix.transpose()
old_z = np.arange(-origin[0], -origin[0] + nbz, 1)
old_y = np.arange(-origin[1], -origin[1] + nby, 1)
old_x = np.arange(-origin[2], -origin[2] + nbx, 1)

myz, myy, myx = np.meshgrid(old_z, old_y, old_x, indexing='ij')

new_x = transfer_matrix[0, 0] * myx + transfer_matrix[0, 1] * myy + transfer_matrix[0, 2] * myz
new_y = transfer_matrix[1, 0] * myx + transfer_matrix[1, 1] * myy + transfer_matrix[1, 2] * myz
new_z = transfer_matrix[2, 0] * myx + transfer_matrix[2, 1] * myy + transfer_matrix[2, 2] * myz
del myx, myy, myz
gc.collect()

rgi = RegularGridInterpolator((old_z, old_y, old_x), data, method='linear', bounds_error=False, fill_value=0)
rot_data = rgi(np.concatenate((new_z.reshape((1, new_z.size)), new_y.reshape((1, new_z.size)),
                               new_x.reshape((1, new_z.size)))).transpose())
rot_data = rot_data.reshape((nbz, nby, nbx)).astype(data.dtype)
gu.multislices_plot(rot_data, sum_frames=True, scale='log', plot_colorbar=True,
                    title='S' + str(scan) + '\n Data after rotation', vmin=0,
                    reciprocal_space=True, is_orthogonal=True)
if save:
    np.savez_compressed(datadir + 'S' + str(scan) + '_data_rotated' + comment + '.npz', data=rot_data)

if not skip_mask:
    rgi = RegularGridInterpolator((old_z, old_y, old_x), mask, method='linear', bounds_error=False, fill_value=0)
    rot_mask = rgi(np.concatenate((new_z.reshape((1, new_z.size)), new_y.reshape((1, new_z.size)),
                                   new_x.reshape((1, new_z.size)))).transpose())
    rot_mask = rot_mask.reshape((nbz, nby, nbx)).astype(mask.dtype)
    gu.multislices_plot(rot_mask, sum_frames=True, scale='linear', plot_colorbar=True,
                        title='S' + str(scan) + '\n Mask after rotation', vmin=0,
                        reciprocal_space=True, is_orthogonal=True)
    if save:
        np.savez_compressed(datadir + 'S' + str(scan) + '_mask_rotated' + comment + '.npz', mask=rot_mask)

plt.ioff()
plt.show()
