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
import multiprocessing as mp
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
interp_factor = 50  # the number of point for the interpolation on a sphere will be the number of voxels at the defined
# q value divided by interp_factor
debug = False  # set to True to see more plots
origin_qspace = (281, 216, 236)  # origin of the reciprocal space in pixels in the order (qx, qz, qy)
q_xcca = (0.479, 0.479)  # q values in 1/nm where to calculate the angular cross-correlation
# results = []
corr_count = np.zeros((16997, 2))  # put the shape as (x, 2) with x the number of points without nans.
# You have to run the script one time to know this number. Declaring corr_count here is required for multiprocessing.
current_point = 0  # do not change this number, it is used as counter in the callback
##################################
# end of user-defined parameters #
##################################


def calc_ccf(point, q2_name, bin_values, polar_azi_int):
    """


    :param point:
    :param q2_name:
    :param bin_values:
    :param polar_azi_int:
    :return:
    """

    # initialize the cross-correlation and bin counter arrays
    ccf_val = np.zeros(bin_values.shape)
    counter_array = np.zeros(bin_values.shape)

    # calculate the angle between the current point and all points from the second q value (delta in [0 pi])
    delta_val = np.arccos(np.sin(polar_azi_int['q1'][point, 0]) * np.sin(polar_azi_int[q2_name][:, 0]) *
                          np.cos(polar_azi_int[q2_name][:, 1] - polar_azi_int['q1'][point, 1]) +
                          np.cos(polar_azi_int['q1'][point, 0]) * np.cos(polar_azi_int[q2_name][:, 0]))

    # find the nearest angular bin value for each value of the array delta
    nearest_indices = util.find_nearest(test_values=delta_val, reference_array=bin_values)

    # update the cross-correlation function for the current point. Nan values are already removed.
    ccf_val[nearest_indices] = polar_azi_int['q1'][point, 2] * polar_azi_int[q2_name][nearest_indices, 2]

    # update the counter of bin indices
    counter_indices, counter_val = np.unique(nearest_indices, return_counts=True)
    counter_array[counter_indices] = counter_val

    return ccf_val, counter_array


def collect_result(result):
    global corr_count, current_point
    corr_count[:, 0] = corr_count[:, 0] + result[0]
    corr_count[:, 1] = corr_count[:, 1] + result[1]
    current_point += 1
    if (current_point % 100) == 0:
        sys.stdout.write('\rPoint {:d}'.format(current_point))
        sys.stdout.flush()


def catch_error(exception):
    print(exception)


def main():
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

    ##############################################################
    # calculate the angular average using mean and median values #
    ##############################################################
    # TODO: calculate the optimal number of bins
    # q_axis, y_mean_masked, y_median_masked = util.angular_avg(data=data, q_values=(qx, qz, qy), origin=origin_qspace,
    #                                                           nb_bins=nz//4, debugging=debug)
    # fig, ax = plt.subplots(1, 1)
    # ax.plot(q_axis, np.log10(y_mean_masked), 'r', label='mean')
    # ax.plot(q_axis, np.log10(y_median_masked), 'b', label='median')
    # ax.axvline(x=q_xcca[0], ymin=0, ymax=1, color='g', linestyle='--', label='q1')
    # ax.axvline(x=q_xcca[1], ymin=0, ymax=1, color='r', linestyle=':', label='q2')
    # ax.set_xlabel('q (1/nm)')
    # ax.set_ylabel('Angular average (A.U.)')
    # ax.legend()

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
            print('Number of voxels for the sphere of radius q ={:.3f} 1/nm:'.format(q_xcca[counter]), nb_pixels)
            indices = np.arange(0, nb_pixels, dtype=float) + 0.5

            # angles for interpolation are chosen using the 'golden spiral method', so that the corresponding points are
            # evenly distributed on the sphere
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
            print('Removing', nan_indices.size, 'nan values,', nb_points[counter], 'remain')
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

    print('The CCF will be calculated over', nb_points[0], 'bins\n')

    # check if corr_count was initialized with the correct number of points
    assert corr_count.shape[0] == nb_points[0],\
        '\nYou need to initialize corr_count.shape[0] with this value: {:d}'.format(nb_points[0])

    angular_bins = np.linspace(start=0, stop=np.pi, num=nb_points[0])

    # delta_step = (ang_corr_count[1, 0] - ang_corr_count[0, 0])

    # # try:  # try to calculate the CCF in one round using vectorization
    # if False:
    #     start = time.time()
    #     # calculate the angle between the all points from both q values (delta in [0 pi])
    #     # values for q1 will be in raw, values for q2 in column
    #     theta1 = theta_phi_int['q1'][:, 0]
    #     theta2 = theta_phi_int[key_q2][:, 0]
    #     phi1 = theta_phi_int['q1'][:, 1]
    #     phi2 = theta_phi_int[key_q2][:, 1]
    #     int1 = theta_phi_int['q1'][:, 2]
    #     int2 = theta_phi_int[key_q2][:, 2]
    #
    #     delta = np.arccos(np.sin(theta1[:, np.newaxis]) * np.sin(theta2[np.newaxis, :]) *
    #                       np.cos(phi2[np.newaxis, :] - phi1[:, np.newaxis]) +
    #                       np.cos(theta1[:, np.newaxis]) * np.cos(theta2[np.newaxis, :]))
    #
    #     for angle in range(nb_points[0]):  # loop over the bins of the CCF
    #         if angle != 0 and (angle % 100) == 0:
    #             sys.stdout.write('\rPoint {:d} / {:d}'.format(angle, nb_points[0]))
    #             sys.stdout.flush()
    #
    #         raw, col = np.nonzero(np.logical_and((delta < ang_corr_count[angle, 0] + delta_step / 2),
    #                                              (delta >= ang_corr_count[angle, 0] - delta_step / 2)))
    #
    #         # update the cross-correlation function. Nan values are already removed.
    #         ang_corr_count[angle, 1] = np.multiply(int1[raw], int2[col]).sum()
    #
    #         # update the counter of bin indices
    #         ang_corr_count[angle, 2] = len(raw)  # or equivalently len(col)
    #
    #     del theta1, theta2, phi1, phi2, int1, int2, delta
    #     gc.collect()
    #     end = time.time()
    #     print('\nTime ellapsed for the calculation of the CCF using vectorization:', int(end - start), 's')
    # if True:
    # # except MemoryError:  # switch to the for loop, not enough memory to calculate the CCF using vectorization
    #     print('Not enough memory, switching to the iterative calculation')

    start = time.time()

    for idx in range(nb_points[0]):
        pool.apply_async(calc_ccf, args=(idx, key_q2, angular_bins, theta_phi_int), callback=collect_result,
                         error_callback=catch_error)

    # close the pool and let all the processes complete
    pool.close()
    pool.join()  # postpones the execution of next line of code until all processes in the queue are done.
    end = time.time()
    print('\nTime ellapsed for the calculation of the CCF:', int(end - start), 's')

    # normalize the cross-correlation by the counter
    corr_count[(corr_count[:, 1] == 0), 1] = np.nan  # discard these values of the CCF
    indices = np.nonzero(corr_count[:, 1])
    corr_count[indices, 0] = corr_count[indices, 0] / corr_count[indices, 1]
    # ang_corr_count[(ang_corr_count[:, 2] == 0), 1] = np.nan  # discard these values of the CCF
    # indices = np.nonzero(ang_corr_count[:, 2])
    # ang_corr_count[indices, 1] = ang_corr_count[indices, 1] / ang_corr_count[indices, 2]

    #######################################
    # save the cross-correlation function #
    #######################################
    np.savez_compressed(savedir + 'CCF_q1={:.3f}_q2={:.3f}_interp{:d}'.format(q_xcca[0], q_xcca[1], interp_factor)
                        + '.npz', obj=corr_count)

    #######################################
    # plot the cross-correlation function #
    #######################################
    fig, ax = plt.subplots()
    ax.plot(180*angular_bins/np.pi, corr_count[:, 0])
    ax.set_xlim(0, 180)
    ax.set_xlabel('Angle (deg)')
    ax.set_ylabel('Cross-correlation')
    ax.set_xticks(np.arange(0, 181, 30))
    ax.set_title('CCF at q1={:.3f} 1/nm  and q2={:.3f} 1/nm'.format(q_xcca[0], q_xcca[1]))
    fig.savefig(savedir + 'CCF_q1={:.3f}_q2={:.3f}'.format(q_xcca[0], q_xcca[1]) + '.png')

    _, ax = plt.subplots()
    ax.plot(180*angular_bins/np.pi, corr_count[:, 1])
    ax.set_xlim(0, 180)
    ax.set_xlabel('Angle (deg)')
    ax.set_ylabel('Number of points')
    ax.set_xticks(np.arange(0, 181, 30))
    ax.set_title('Points per angular bin')
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()

