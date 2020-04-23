# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import time
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RegularGridInterpolator
import tkinter as tk
from tkinter import filedialog
import gc
import sys
sys.path.append('C:/Users/Jerome/Documents/myscripts/bcdi/')
import bcdi.graph.graph_utils as gu
import bcdi.utils.utilities as util

helptext = """
Calculate the angular cross-correlation in a 3D reciprocal space dataset at the same q value or between two different q
values. The 3D dataset is expected to be interpolated on an orthonormal grid.

Input: the 3D dataset, an optional 3D mask, (qx, qy, qz) values

Laboratory frame convention (CXI): z downstream, y vertical up, x outboard.
Reciprocal space basis:            qx downstream, qz vertical up, qy outboard."""

datadir = "D:/data/P10_March2020_CDI/test_april/data/align_06_00248/pynx_not_masked/"
savedir = "D:/data/P10_March2020_CDI/test_april/data/align_06_00248/simu/"
comment = ''  # should start with _
interp_factor = 2  # the number of point for the interpolation on a sphere will be the number of voxels in the q range
# divided by interp_factor
plot_avg = False  # True to plot the angular average of the data
debug = False  # set to True to see more plots
origin_qspace = (281, 216, 236)  # origin of the reciprocal space in pixels in the order (qx, qz, qy)
q_xcca = (0.45, 0.45)  # q values in 1/nm where to calculate the angular cross-correlation
##################################
# end of user-defined parameters #
##################################

##########################
# check input parameters #
##########################
assert len(q_xcca) == 2, "Two q values should be provided (it can be the same value)"
assert len(origin_qspace) == 3, "origin_qspace should be a tuple of 3 integer pixel values"
if q_xcca[0] == q_xcca[1]:
    same_q = True
else:
    same_q = False

###################
# define colormap #
###################
bad_color = '1.0'  # white background
colormap = gu.Colormap(bad_color=bad_color)
my_cmap = colormap.cmap
plt.ion()

###################################
# load experimental data and mask #
###################################
plt.ion()
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(initialdir=datadir, title="Select the 3D reciprocal space map",
                                       filetypes=[("NPZ", "*.npz")])
data = np.load(file_path)['data']
nz, ny, nx = data.shape

try:
    file_path = filedialog.askopenfilename(initialdir=datadir, title="Select the 3D mask",
                                           filetypes=[("NPZ", "*.npz")])
    mask = np.load(file_path)['mask']

    data[np.nonzero(mask)] = 0
except FileNotFoundError:
    mask = None

file_path = filedialog.askopenfilename(initialdir=datadir, title="Select q values",
                                       filetypes=[("NPZ", "*.npz")])
qvalues = np.load(file_path)
qx = qvalues['qx']
qz = qvalues['qz']
qy = qvalues['qy']

#########################################################
# plot the angular average using mean and median values #
#########################################################
if plot_avg:
    q_axis, y_mean_masked, y_median_masked = util.angular_avg(data=data, q_values=(qx, qz, qy), origin=origin_qspace,
                                                              nb_bins=nz//4, debugging=debug)
    fig, ax = plt.subplots(1, 1)
    ax.plot(q_axis, np.log10(y_mean_masked), 'r', label='mean')
    ax.plot(q_axis, np.log10(y_median_masked), 'b', label='median')
    ax.axvline(x=q_xcca[0], ymin=0, ymax=1, color='g', linestyle='--', label='q1')
    ax.axvline(x=q_xcca[1], ymin=0, ymax=1, color='r', linestyle=':', label='q2')
    ax.set_xlabel('q (1/nm)')
    ax.set_ylabel('Angular average (A.U.)')
    ax.legend()

##############################################################
# interpolate the data onto spheres at user-defined q values #
##############################################################
# calculate the matrix of distances from the origin of reciprocal space
distances = np.sqrt((qx[:, np.newaxis, np.newaxis] - qx[origin_qspace[0]]) ** 2 +
                    (qz[np.newaxis, :, np.newaxis] - qz[origin_qspace[1]]) ** 2 +
                    (qy[np.newaxis, np.newaxis, :] - qy[origin_qspace[2]]) ** 2)
dq = min(qx[1]-qx[0], qz[1]-qz[0], qy[1]-qy[0])

for counter, value in enumerate(q_xcca):
    if (counter == 1) and not same_q:
        nb_pixels = int((np.logical_and((distances < q_xcca[counter]+dq), (distances > q_xcca[counter]-dq))).sum()
                        / interp_factor)
        print('Number of voxels for the sphere of radius q ={:.3f} 1/nm'.format(q_xcca[counter]), nb_pixels)
        indices = np.arange(0, nb_pixels, dtype=float) + 0.5

        theta = np.arccos(1 - 2*indices/nb_pixels)  # theta, phi are the angles from the spherical coordinates
        phi = np.pi * (1 + np.sqrt(5)) * indices

        qx_sphere = q_xcca[counter] * np.cos(phi) * np.sin(theta)
        qz_sphere = q_xcca[counter] * np.cos(theta)
        qy_sphere = q_xcca[counter] * np.sin(phi) * np.sin(theta)

        # interpolate the data onto the new points
        rgi = RegularGridInterpolator((qx, qz, qy), data, method='linear', bounds_error=False, fill_value=np.nan)
        sphere_int = rgi(np.concatenate((qx_sphere.reshape((1, nb_pixels)), qz_sphere.reshape((1, nb_pixels)),
                                         qy_sphere.reshape((1, nb_pixels)))).transpose())

        if debug:
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(qx_sphere, qz_sphere, qy_sphere, c=np.log10(sphere_int), cmap=my_cmap)
            ax.set_xlabel('qx')
            ax.set_ylabel('qz')
            ax.set_zlabel('qy')
            plt.title('Intensity interpolated on a sphere of radius q ={:.3f} 1/nm'.format(q_xcca[0]))
            plt.pause(0.1)

        # TODO: save qx_sphere, qz_sphere, qy_sphere, sphere_int in a structure?

plt.ioff()
plt.show()

# start = time.time()
# end = time.time()
# print('Time ellapsed for the interpolation:', int(end - start), 's')
