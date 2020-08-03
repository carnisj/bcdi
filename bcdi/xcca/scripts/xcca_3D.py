# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import time
import datetime
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RegularGridInterpolator
import tkinter as tk
from tkinter import filedialog
import gc
import multiprocessing as mp
import sys
sys.path.append('C:/Users/Jerome/Documents/myscripts/bcdi/')
import bcdi.graph.graph_utils as gu
import bcdi.utils.utilities as util
import bcdi.xcca.xcca_utils as xcca

helptext = """
Calculate the angular cross-correlation in a 3D reciprocal space dataset at the same q value or between two different q
values. The 3D dataset is expected to be interpolated on an orthonormal grid.

Input: the 3D dataset, an optional 3D mask, (qx, qy, qz) values

Laboratory frame convention (CXI): z downstream, y vertical up, x outboard.
Reciprocal space basis:            qx downstream, qz vertical up, qy outboard."""

datadir = "D:/data/P10_March2020_CDI/test_april/data/align_06_00248/pynx_not_masked/"
savedir = "D:/data/P10_March2020_CDI/test_april/data/align_06_00248/pynx_not_masked/"
comment = '_trash'  # should start with _
interp_factor = 100  # the number of points for the interpolation on a sphere will be the number of voxels
# at the defined q value divided by interp_factor
angular_resolution = 0.5  # in degrees, angle between to adjacent points for the calculation of the cross-correlation
debug = False  # set to True to see more plots
origin_qspace = (281, 216, 236)  # origin of the reciprocal space in pixels in the order (qx, qz, qy)
q_xcca = (0.479, 0.479)  # q values in 1/nm where to calculate the angular cross-correlation
hotpix_threshold = 1e6  # data above this threshold will be masked
single_proc = False  # do not use multiprocessing if True
plot_meandata = False  # if True, will plot the 1D average of the data
##################################################################
# end of user-defined parameters, do not change parameters below #
##################################################################
corr_count = np.zeros((int(180/angular_resolution), 2))  # initialize the cross-correlation array
current_point = 0  # do not change this number, it is used as counter in the callback
#############################################
# define multiprocessing callback functions #
#############################################


def collect_result(result):
    """
    Callback processing the result after asynchronous multiprocessing. Update the global arrays corr_count, corr_point.

    :param result: the output of ccf_val, containing the sorted cross-correlation values, the angular bins indices and
     the number of points contributing to the angular bins
    """
    global corr_count, current_point
    # result is a tuple: ccf_uniq_val, counter_val, counter_indices
    corr_count[result[2], 0] = corr_count[result[2], 0] + result[0]

    corr_count[result[2], 1] = corr_count[result[2], 1] + result[1]  # this line is ok

    current_point += 1
    if (current_point % 100) == 0:
        sys.stdout.write('\rPoint {:d}'.format(current_point))
        sys.stdout.flush()


def collect_result_debug(ccf_uniq_val, counter_val, counter_indices):
    """
    Similar behaviour as collect_result() when multiprocessing is not used, useful for debugging.

    :param ccf_uniq_val: the sorted cross-correlation values
    :param counter_val: the number of points contributing to the angular bins defined by counter_indices
    :param counter_indices: the indices of angular bins where to update the cross-correlation
    """
    global corr_count, current_point
    # result is a tuple: ccf_uniq_val, counter_val, counter_indices
    corr_count[counter_indices, 0] = corr_count[counter_indices, 0] + ccf_uniq_val

    corr_count[counter_indices, 1] = corr_count[counter_indices, 1] + counter_val  # this line is ok

    current_point += 1
    if (current_point % 100) == 0:
        sys.stdout.write('\rPoint {:d}'.format(current_point))
        sys.stdout.flush()


def main():
    if not single_proc:
        print("Number of processors: ", mp.cpu_count())
        mp.freeze_support()
        pool = mp.Pool(mp.cpu_count())  # use this number of processes

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

    file_path = filedialog.askopenfilename(initialdir=datadir, title="Select the 3D mask",
                                           filetypes=[("NPZ", "*.npz")])
    mask = np.load(file_path)['mask']

    print((data > hotpix_threshold).sum(), ' hotpixels masked')
    mask[data > hotpix_threshold] = 1
    data[np.nonzero(mask)] = np.nan
    del mask
    gc.collect()

    file_path = filedialog.askopenfilename(initialdir=datadir, title="Select q values",
                                           filetypes=[("NPZ", "*.npz")])
    qvalues = np.load(file_path)
    qx = qvalues['qx']
    qz = qvalues['qz']
    qy = qvalues['qy']

    del qvalues
    gc.collect()

    ##############################################################
    # calculate the angular average using mean and median values #
    ##############################################################
    if plot_meandata:
        q_axis, y_mean_masked, y_median_masked = xcca.angular_avg(data=data, q_values=(qx, qz, qy),
                                                                  origin=origin_qspace, nb_bins=250, debugging=debug)
        fig, ax = plt.subplots(1, 1)
        ax.plot(q_axis, np.log10(y_mean_masked), 'r', label='mean')
        ax.plot(q_axis, np.log10(y_median_masked), 'b', label='median')
        ax.axvline(x=q_xcca[0], ymin=0, ymax=1, color='g', linestyle='--', label='q1')
        ax.axvline(x=q_xcca[1], ymin=0, ymax=1, color='r', linestyle=':', label='q2')
        ax.set_xlabel('q (1/nm)')
        ax.set_ylabel('Angular average (A.U.)')
        ax.legend()
        fig.savefig(savedir + '1D_average.png')

        del q_axis, y_median_masked, y_mean_masked

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
            nb_pixels = (np.logical_and((distances < q_xcca[counter]+dq), (distances > q_xcca[counter]-dq))).sum()

            print('Number of voxels for the sphere of radius q ={:.3f} 1/nm:'.format(q_xcca[counter]), nb_pixels)

            nb_pixels = int(nb_pixels / interp_factor)
            print('Dividing the number of voxels by interp_factor: {:d} voxels remaining'.format(nb_pixels))

            indices = np.arange(0, nb_pixels, dtype=float) + 0.5

            # angles for interpolation are chosen using the 'golden spiral method', so that the corresponding points
            # are evenly distributed on the sphere
            theta = np.arccos(1 - 2*indices/nb_pixels)  # theta is the polar angle of the spherical coordinates
            phi = np.pi * (1 + np.sqrt(5)) * indices  # phi is the azimuthal angle of the spherical coordinates

            qx_sphere = q_xcca[counter] * np.cos(phi) * np.sin(theta)
            qz_sphere = q_xcca[counter] * np.cos(theta)
            qy_sphere = q_xcca[counter] * np.sin(phi) * np.sin(theta)

            # interpolate the data onto the new points
            rgi = RegularGridInterpolator((qx, qz, qy), data, method='linear', bounds_error=False, fill_value=np.nan)
            sphere_int = rgi(np.concatenate((qx_sphere.reshape((1, nb_pixels)), qz_sphere.reshape((1, nb_pixels)),
                                             qy_sphere.reshape((1, nb_pixels)))).transpose())

            # remove nan values here, then we do not need to care about it anymore in the for loop following
            nan_indices = np.argwhere(np.isnan(sphere_int))
            theta = np.delete(theta, nan_indices)
            phi = np.delete(phi, nan_indices)
            sphere_int = np.delete(sphere_int, nan_indices)
            theta_phi_int[dict_fields[counter]] = np.concatenate((theta[:, np.newaxis],
                                                                  phi[:, np.newaxis],
                                                                  sphere_int[:, np.newaxis]), axis=1)
            # update the number of points without nan
            nb_points.append(len(theta))
            print('q={:.3f}:'.format(q_xcca[counter]), ' removing', nan_indices.size, 'nan values,',
                  nb_points[counter], 'remain')
            if debug:
                fig = plt.figure()
                ax = Axes3D(fig)
                ax.scatter(qx_sphere, qz_sphere, qy_sphere, c=np.log10(sphere_int), cmap=my_cmap)
                ax.set_xlabel('qx')
                ax.set_ylabel('qz')
                ax.set_zlabel('qy')
                plt.title('Intensity interpolated on a sphere of radius q ={:.3f} 1/nm'.format(q_xcca[0]))
                plt.pause(0.1)

            del qx_sphere, qz_sphere, qy_sphere, theta, phi, sphere_int, indices, nan_indices
            gc.collect()
    del qx, qy, qz, distances, data
    gc.collect()

    ############################################
    # calculate the cross-correlation function #
    ############################################
    # TODO: implement the Gram matrix for non-cubic unit cells
    if same_q:
        key_q2 = 'q1'
    else:
        key_q2 = 'q2'

    print('The CCF will be calculated over', nb_points[0], 'points and', corr_count.shape[0], 'angular bins\n')

    angular_bins = np.linspace(start=0, stop=np.pi, num=corr_count.shape[0], endpoint=False)

    start = time.time()

    for idx in range(nb_points[0]):
        if single_proc:
            ccf_uniq_val, counter_val, counter_indices = \
                 xcca.calc_ccf(point=idx, q2_name=key_q2, bin_values=angular_bins, polar_azi_int=theta_phi_int)
            collect_result_debug(ccf_uniq_val, counter_val, counter_indices)
        else:

            pool.apply_async(xcca.calc_ccf, args=(idx, key_q2, angular_bins, theta_phi_int), callback=collect_result,
                             error_callback=util.catch_error)

    # close the pool and let all the processes complete
    pool.close()
    pool.join()  # postpones the execution of next line of code until all processes in the queue are done.
    end = time.time()
    print('\nTime ellapsed for the calculation of the CCF:', str(datetime.timedelta(seconds=int(end - start))))

    # normalize the cross-correlation by the counter
    indices = np.nonzero(corr_count[:, 1])
    corr_count[indices, 0] = corr_count[indices, 0] / corr_count[indices, 1]

    #######################################
    # save the cross-correlation function #
    #######################################
    filename = 'CCF_q1={:.3f}_q2={:.3f}'.format(q_xcca[0], q_xcca[1]) +\
               '_points{:d}_interp{:d}_res{:.3f}'.format(nb_points[0], interp_factor, angular_resolution) + comment
    np.savez_compressed(savedir + filename + '.npz', obj=corr_count)

    #######################################
    # plot the cross-correlation function #
    #######################################
    # find the y limit excluding the peaks at 0 and 180 degrees
    indices = np.argwhere(np.logical_and((angular_bins >= 5*np.pi/180), (angular_bins <= 175*np.pi/180)))
    ymax = 1.2 * corr_count[indices, 0].max()
    corr_count[(corr_count[:, 1] == 0), 1] = np.nan  # discard these values of the CCF

    fig, ax = plt.subplots()
    ax.plot(180*angular_bins/np.pi, corr_count[:, 0], color='red', linestyle="-", markerfacecolor='blue',
            marker='.')
    ax.set_xlim(0, 180)
    ax.set_ylim(0, ymax)
    ax.set_xlabel('Angle (deg)')
    ax.set_ylabel('Cross-correlation')
    ax.set_xticks(np.arange(0, 181, 30))
    ax.set_title('CCF at q1={:.3f} 1/nm  and q2={:.3f} 1/nm'.format(q_xcca[0], q_xcca[1]))
    fig.savefig(savedir + filename + '.png')

    _, ax = plt.subplots()
    ax.plot(180*angular_bins/np.pi, corr_count[:, 1], linestyle="None", markerfacecolor='blue',
            marker='.')
    ax.set_xlim(0, 180)
    ax.set_xlabel('Angle (deg)')
    ax.set_ylabel('Number of points')
    ax.set_xticks(np.arange(0, 181, 30))
    ax.set_title('Points per angular bin')
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
