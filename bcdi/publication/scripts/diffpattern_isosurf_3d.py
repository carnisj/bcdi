# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np
from matplotlib import pyplot as plt
# import mayavi
from mayavi import mlab
import tkinter as tk
from tkinter import filedialog
from scipy.interpolate import RegularGridInterpolator
import gc
import sys
sys.path.append('D:/myscripts/bcdi/')
import bcdi.utils.utilities as util
import bcdi.postprocessing.postprocessing_utils as pu

helptext = """
Template for 3d isosurface figures of a diffraction pattern.

The data will be interpolated on an isotropic range in order to fulfill the desired tick spacing.

Note that qy values (axis 2) are opposite to the correct ones, because there is no easy way to flip an axis in mayvi.

The diffraction pattern is supposed to be in an orthonormal frame and q values need to be provided.
"""

scan = 76    # spec scan number
root_folder = "D:/data/P10_March2020_CDI/data/"
sample_name = "non_ht_sphere_c"
homedir = root_folder + sample_name + '_' + str('{:05d}'.format(scan)) + '/pynx/'
comment = ""  # should start with _
binning = [1, 1, 1]  # binning for the measured diffraction pattern in each dimension
tick_spacing = 0.2  # in 1/nm, spacing between ticks
threshold_isosurface = 4.5  # log scale
##########################
# end of user parameters #
##########################

#############
# load data #
#############
plt.ion()
root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename(initialdir=homedir, title="Select the diffraction pattern",
                                       filetypes=[("NPZ", "*.npz")])
data = util.load_file(file_path)
assert data.ndim == 3, 'data should be a 3D array'

nz, ny, nx = data.shape
print("Initial data size: (", nz, ',', ny, ',', nx, ')')

file_path = filedialog.askopenfilename(initialdir=homedir, title="Select q values",
                                       filetypes=[("NPZ", "*.npz")])
q_values = np.load(file_path)
qx = q_values['qx']  # 1D array
qy = q_values['qy']  # 1D array
qz = q_values['qz']  # 1D array

############
# bin data #
############
qx = qx[:nz - (nz % binning[0]):binning[0]]
qz = qz[:ny - (ny % binning[1]):binning[1]]
qy = qy[:nx - (nx % binning[2]):binning[2]]
data = pu.bin_data(data, (binning[0], binning[1], binning[2]), debugging=False)
print('Diffraction data shape after binning', data.shape)

##########################################
# take the largest data symmetrical in q #
##########################################
min_range = min(min(abs(qx.min()), qx.max()), min(abs(qz.min()), qz.max()), min(abs(qy.min()), qy.max()))
indices_qx = np.argwhere(abs(qx) < min_range)[:, 0]
indices_qz = np.argwhere(abs(qz) < min_range)[:, 0]
indices_qy = np.argwhere(abs(qy) < min_range)[:, 0]
qx = qx[indices_qx]
qz = qz[indices_qz]
qy = qy[indices_qy]
data = data[indices_qx.min():indices_qx.max()+1,
            indices_qz.min():indices_qz.max()+1,
            indices_qy.min():indices_qy.max()+1]
nz, ny, nx = data.shape
print("Shape of the largest symmetrical dataset:", nz, ny, nx)

##############################################################
# interpolate the data to have ticks at the desired location #
##############################################################
half_labels = int(min_range // tick_spacing)  # the number of labels is (2*half_labels+1)
rgi = RegularGridInterpolator((np.linspace(qx.min(), qx.max(), num=nz),
                               np.linspace(qz.min(), qz.max(), num=ny),
                               np.linspace(qy.min(), qy.max(), num=nx)),
                              data, method='linear', bounds_error=False, fill_value=0)

new_z, new_y, new_x = np.meshgrid(np.linspace(-tick_spacing*half_labels, tick_spacing*half_labels, num=nz),
                                  np.linspace(-tick_spacing*half_labels, tick_spacing*half_labels, num=ny),
                                  np.linspace(-tick_spacing*half_labels, tick_spacing*half_labels, num=nx),
                                  indexing='ij')

newdata = rgi(np.concatenate((new_z.reshape((1, new_z.size)), new_y.reshape((1, new_z.size)),
                              new_x.reshape((1, new_z.size)))).transpose())
newdata = newdata.reshape((nz, ny, nx)).astype(data.dtype)

del data, new_z, new_y, new_x, rgi
gc.collect()

#########################################
# plot 3D isosurface (perspective view) #
#########################################
newdata = np.flip(newdata, 2)  # mayavi expects xyz, data order is downstream/upward/outboard
newdata[newdata == 0] = np.nan
grid_qx, grid_qz, grid_qy = np.mgrid[-tick_spacing*half_labels:tick_spacing*half_labels:1j * nz,
                                     -tick_spacing*half_labels:tick_spacing*half_labels:1j * ny,
                                     -tick_spacing*half_labels:tick_spacing*half_labels:1j * nx]
# in CXI convention, z is downstream, y vertical and x outboard
# for q: classical convention qx downstream, qz vertical and qy outboard
myfig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(1000, 1000))
mlab.contour3d(grid_qx, grid_qz, grid_qy, np.log10(newdata),
               contours=[0.5*threshold_isosurface, 0.6*threshold_isosurface, 0.7*threshold_isosurface, 0.8*threshold_isosurface, 0.9*threshold_isosurface, threshold_isosurface, 1.1*threshold_isosurface, 1.2*threshold_isosurface],
               opacity=0.2, colormap='hsv', vmin=3.5, vmax=5.5)  # , color=(0.7, 0.7, 0.7))

mlab.view(azimuth=38, elevation=63, distance=5*np.sqrt(grid_qx**2+grid_qz**2+grid_qy**2).max())
# azimut is the rotation around z axis of mayavi (x)
mlab.roll(0)

ax = mlab.axes(line_width=2.0, nb_labels=2*half_labels+1)
mlab.xlabel('Qx (1/nm)')
mlab.ylabel('Qz (1/nm)')
mlab.zlabel('-Qy (1/nm)')
mlab.savefig(homedir + 'S' + str(scan) + comment + '_labels.png', figure=myfig)
ax.label_text_property.opacity = 0.0
ax.title_text_property.opacity = 0.0
mlab.savefig(homedir + 'S' + str(scan) + comment + '_axes.png', figure=myfig)
ax.axes.x_axis_visibility = False
ax.axes.y_axis_visibility = False
ax.axes.z_axis_visibility = False
mlab.savefig(homedir + 'S' + str(scan) + comment + '.png', figure=myfig)
mlab.show()
