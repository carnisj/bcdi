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
interp_factor = 50  # the number of point for the interpolation on a sphere will be the number of voxels in the q range
# divided by interp_factor
plot_avg = False  # True to plot the angular average of the data
debug = False  # set to True to see more plots
origin_qspace = (281, 216, 236)  # origin of the reciprocal space in pixels in the order (qx, qz, qy)
q_xcca = (0.479, 0.479)  # q values in 1/nm where to calculate the angular cross-correlation
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

    data[np.nonzero(mask)] = np.nan
    del mask
    gc.collect()
except FileNotFoundError:
    pass

file_path = filedialog.askopenfilename(initialdir=datadir, title="Select q values",
                                       filetypes=[("NPZ", "*.npz")])
qvalues = np.load(file_path)
qx = qvalues['qx']
qz = qvalues['qz']
qy = qvalues['qy']

del qvalues
gc.collect()

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

theta_phi_int = dict()  # create dictionnary
dict_fields = ['q1', 'q2']
nb_points = []
for counter, value in enumerate(q_xcca):
    if (counter == 0) or ((counter == 1) and not same_q):
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

        # remove nan values here, then we do not need to care about it anymore in the for loop following
        nan_indices = np.argwhere(np.isnan(sphere_int))
        print('removing', nan_indices.size, 'nan values')
        # for idx in range(len(nan_indices)):
        theta = np.delete(theta, nan_indices)
        phi = np.delete(phi, nan_indices)
        sphere_int = np.delete(sphere_int, nan_indices)
        theta_phi_int[dict_fields[counter]] = np.concatenate((theta[:, np.newaxis],
                                                              phi[:, np.newaxis],
                                                              sphere_int[:, np.newaxis]), axis=1)
        # update the number of points without nan
        nb_points.append(len(theta))

        if debug:
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(qx_sphere, qz_sphere, qy_sphere, c=np.log10(sphere_int), cmap=my_cmap)
            ax.set_xlabel('qx')
            ax.set_ylabel('qz')
            ax.set_zlabel('qy')
            plt.title('Intensity interpolated on a sphere of radius q ={:.3f} 1/nm'.format(q_xcca[0]))
            plt.pause(0.1)

        del qx_sphere, qz_sphere, qy_sphere, theta, phi, sphere_int
        gc.collect()
del qx, qy, qz
gc.collect()

############################################
# calculate the cross-correlation function #
############################################
# the ideal would be to calculate angles between all points, but this would end up quickly in a memory error due to the
# large number of points. An iterative approach without storing values seems the best approach.
# TODO: implement the Gram matrix for non-cubic unit cells
if same_q:
    key_q2 = 'q1'
else:
    key_q2 = 'q2'

ang_corr_count = np.zeros((nb_points[0], 3))  # the first column contains the angular values, the second column the
# correlations, the third column the number of points for the averaging
ang_corr_count[:, 0] = np.linspace(start=0, stop=np.pi, num=nb_points[0])
delta_step = (ang_corr_count[1, 0] - ang_corr_count[0, 0]) / 2
# TODO: how to choose the bin width for the CCF? This will define the resolution.

start = time.time()
for idx in range(nb_points[0]):  # loop over the points of the first q value
    # calculate the angle between the current point and all points from the second q value (delta in [0 pi])
    delta = np.arccos(np.sin(theta_phi_int['q1'][idx, 0]) * np.sin(theta_phi_int[key_q2][:, 0]) *
                      np.cos(theta_phi_int[key_q2][:, 1] - theta_phi_int['q1'][idx, 1]) +
                      np.cos(theta_phi_int['q1'][idx, 0]) * np.cos(theta_phi_int[key_q2][:, 0]))

    # find the nearest angular bin value for each value of the array delta
    indices = util.find_nearest(test_values=delta, reference_array=ang_corr_count[:, 0])

    # update the cross-correlation function with correlations for the current point. Nan values are already removed.
    ang_corr_count[indices, 1] = theta_phi_int['q1'][idx, 2] * theta_phi_int[key_q2][indices, 2]

    # update the counter of bin indices
    index, counts = np.unique(indices, return_counts=True)
    ang_corr_count[index, 2] = ang_corr_count[index, 2] + counts

    del index, counts, indices, delta
    gc.collect()
end = time.time()
print('Time ellapsed for the calculation of the CCF:', int(end - start), 's')
# normalize the cross-correlation by the counter
ang_corr_count[:, 1] = ang_corr_count[:, 1] / ang_corr_count[:, 2]

#######################################
# save the cross-correlation function #
#######################################
np.savez_compressed(savedir + 'CCF_q1={:.3f}_q2={:.3f}'.format(q_xcca[0], q_xcca[1]) + '.npz', obj=ang_corr_count)

#######################################
# plot the cross-correlation function #
#######################################
fig, ax = plt.subplots()
ax.plot(180*ang_corr_count[:, 0]/np.pi, ang_corr_count[:, 1])
ax.set_xlim(0, 180)
ax.set_xlabel('Angle (deg)')
ax.set_ylabel('Cross-correlation')
ax.set_title('CCF at q1 ={:.3f} 1/nm  and q2={:.3f} 1/nm '.format(q_xcca[0], q_xcca[1]))
plt.savefig(savedir + 'CCF_q1={:.3f}_q2={:.3f}'.format(q_xcca[0], q_xcca[1]) + '.png')
plt.ioff()
plt.show()
