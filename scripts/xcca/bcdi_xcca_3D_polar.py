#!/usr/bin/env python3

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import datetime
import gc
import multiprocessing as mp
import sys
import time
import tkinter as tk
import warnings
from tkinter import filedialog

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import RegularGridInterpolator

import bcdi.graph.graph_utils as gu
import bcdi.postprocessing.facet_recognition as fu
import bcdi.utils.utilities as util
import bcdi.xcca.xcca_utils as xcca
from bcdi.graph.colormap import ColormapFactory

helptext = """
Calculate the angular cross-correlation in a 3D reciprocal space dataset at the same
q value or between two different q values. The 3D dataset is expected to be
interpolated on an orthonormal grid. The intensity used for cross-correlation
calculation is interpolated using the golden spiral method on a sphere of the desired
q radius, using original voxels belonging to a slice at this q value. Downsampling
can be applied for faster calculation.

Input: the 3D dataset, an optional 3D mask, (qx, qy, qz) values

Laboratory frame convention (CXI): z downstream, y vertical up, x outboard.
Reciprocal space basis:            qx downstream, qz vertical up, qy outboard.
"""

datadir = "D:/data/P10_August2019_CDI/data/gold_2_2_2_00022/pynx/1_4_4_fullrange_xcca/"
savedir = "D:/data/P10_August2019_CDI/data/gold_2_2_2_00022/pynx/1_4_4_fullrange_xcca/"
comment = "_q1q3"  # should start with _
interp_factor = 10  # the number of points for the interpolation on a sphere
# will be the number of voxels
# at the defined q value divided by interp_factor
angular_resolution = 0.1  # in degrees, angle between to adjacent points
# for the calculation of the cross-correlation
debug = True  # set to True to see more plots
origin_qspace = (
    330,
    204,
    330,
)  # origin of the reciprocal space in pixels in the order (qx, qz, qy)
q_xcca = [
    0.104,
    0.172,
]  # q values in 1/nm where to calculate the angular cross-correlation
hotpix_threshold = 1e9  # data above this threshold will be masked
# check once in the stereographic projection that usefull data is not masked,
# otherwise correlations will be wrong
single_proc = False  # do not use multiprocessing if True
plot_meandata = False  # if True, will plot the 1D average of the data
##################################################################
# end of user-defined parameters, do not change parameters below #
##################################################################
corr_count = np.zeros(
    (int(180 / angular_resolution), 2)
)  # initialize the cross-correlation array
current_point = 0  # do not change this number, it is used as counter in the callback
#############################################
# define multiprocessing callback functions #
#############################################


def collect_result(result):
    """
    Callback processing the result after asynchronous multiprocessing.

    Update the global arrays corr_count, corr_point.

    :param result: the output of ccf_val, containing the sorted cross-correlation
     values, the angular bins indices and the number of points contributing to the
     angular bins
    """
    global corr_count, current_point
    # result is a tuple: ccf_uniq_val, counter_val, counter_indices
    corr_count[result[2], 0] = corr_count[result[2], 0] + result[0]

    corr_count[result[2], 1] = corr_count[result[2], 1] + result[1]  # this line is ok

    current_point += 1
    if (current_point % 100) == 0:
        sys.stdout.write(f"\rPoint {current_point:d}")
        sys.stdout.flush()


def collect_result_debug(ccf_uniq_val, counter_val, counter_indices):
    """
    Similar behaviour as collect_result() when multiprocessing is not used.

    It is usefull for debugging.

    :param ccf_uniq_val: the sorted cross-correlation values
    :param counter_val: the number of points contributing to the angular bins defined
     by counter_indices
    :param counter_indices: the indices of angular bins where to update
     the cross-correlation
    """
    global corr_count, current_point
    # result is a tuple: ccf_uniq_val, counter_val, counter_indices
    corr_count[counter_indices, 0] = corr_count[counter_indices, 0] + ccf_uniq_val

    corr_count[counter_indices, 1] = (
        corr_count[counter_indices, 1] + counter_val
    )  # this line is ok

    current_point += 1
    if (current_point % 100) == 0:
        sys.stdout.write(f"\rPoint {current_point:d}")
        sys.stdout.flush()


def main(user_comment):
    """
    Protection for multiprocessing.

    :param user_comment: comment to include in the filename when saving results
    """
    ##########################
    # check input parameters #
    ##########################
    global corr_count

    if len(q_xcca) != 2:
        raise ValueError("Two q values should be provided (it can be the same value)")
    if len(origin_qspace) != 3:
        raise ValueError("origin_qspace should be a tuple of 3 integer pixel values")
    q_xcca.sort()
    same_q = q_xcca[0] == q_xcca[1]
    warnings.filterwarnings("ignore")

    ###################
    # define colormap #
    ###################
    bad_color = "1.0"  # white background
    my_cmap = ColormapFactory(bad_color=bad_color).cmap
    plt.ion()

    ###################################
    # load experimental data and mask #
    ###################################
    plt.ion()
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        initialdir=datadir,
        title="Select the 3D reciprocal space map",
        filetypes=[("NPZ", "*.npz")],
    )
    data = np.load(file_path)["data"]

    file_path = filedialog.askopenfilename(
        initialdir=datadir, title="Select the 3D mask", filetypes=[("NPZ", "*.npz")]
    )
    mask = np.load(file_path)["mask"]

    print((data > hotpix_threshold).sum(), " hotpixels masked")
    mask[data > hotpix_threshold] = 1
    data[np.nonzero(mask)] = np.nan
    del mask
    gc.collect()

    file_path = filedialog.askopenfilename(
        initialdir=datadir, title="Select q values", filetypes=[("NPZ", "*.npz")]
    )
    qvalues = np.load(file_path)
    qx = qvalues["qx"]
    qz = qvalues["qz"]
    qy = qvalues["qy"]

    del qvalues
    gc.collect()

    ##############################################################
    # calculate the angular average using mean and median values #
    ##############################################################
    if plot_meandata:
        q_axis, y_mean_masked, y_median_masked = xcca.angular_avg(
            data=data,
            q_values=(qx, qz, qy),
            origin=origin_qspace,
            nb_bins=250,
            debugging=debug,
        )
        fig, ax = plt.subplots(1, 1)
        ax.plot(q_axis, np.log10(y_mean_masked), "r", label="mean")
        ax.plot(q_axis, np.log10(y_median_masked), "b", label="median")
        ax.axvline(x=q_xcca[0], ymin=0, ymax=1, color="g", linestyle="--", label="q1")
        ax.axvline(x=q_xcca[1], ymin=0, ymax=1, color="r", linestyle=":", label="q2")
        ax.set_xlabel("q (1/nm)")
        ax.set_ylabel("Angular average (A.U.)")
        ax.legend()
        plt.pause(0.1)
        fig.savefig(savedir + "1D_average.png")

        del q_axis, y_median_masked, y_mean_masked

    ##############################################################
    # interpolate the data onto spheres at user-defined q values #
    ##############################################################
    # calculate the matrix of distances from the origin of reciprocal space
    distances = np.sqrt(
        (qx[:, np.newaxis, np.newaxis] - qx[origin_qspace[0]]) ** 2
        + (qz[np.newaxis, :, np.newaxis] - qz[origin_qspace[1]]) ** 2
        + (qy[np.newaxis, np.newaxis, :] - qy[origin_qspace[2]]) ** 2
    )
    dq = min(qx[1] - qx[0], qz[1] - qz[0], qy[1] - qy[0])

    theta_phi_int = {}  # create dictionnary
    dict_fields = ["q1", "q2"]
    nb_points = []

    for counter, q_value in enumerate(q_xcca):
        if (counter == 0) or ((counter == 1) and not same_q):
            nb_pixels = (
                np.logical_and((distances < q_value + dq), (distances > q_value - dq))
            ).sum()

            print(
                f"\nNumber of voxels for the sphere of radius q ={q_value:.3f} 1/nm:",
                nb_pixels,
            )

            nb_pixels = int(nb_pixels / interp_factor)
            print(
                f"Dividing the number of voxels by interp_factor: {nb_pixels:d} "
                "voxels remaining"
            )

            indices = np.arange(0, nb_pixels, dtype=float) + 0.5

            # angles for interpolation are chosen using the 'golden spiral method',
            # so that the corresponding points are evenly distributed on the sphere
            theta = np.arccos(
                1 - 2 * indices / nb_pixels
            )  # theta is the polar angle of the spherical coordinates
            phi = (
                np.pi * (1 + np.sqrt(5)) * indices
            )  # phi is the azimuthal angle of the spherical coordinates

            qx_sphere = q_value * np.cos(phi) * np.sin(theta)
            qz_sphere = q_value * np.cos(theta)
            qy_sphere = q_value * np.sin(phi) * np.sin(theta)

            # interpolate the data onto the new points
            rgi = RegularGridInterpolator(
                (qx, qz, qy),
                data,
                method="linear",
                bounds_error=False,
                fill_value=np.nan,
            )
            sphere_int = rgi(
                np.concatenate(
                    (
                        qx_sphere.reshape((1, nb_pixels)),
                        qz_sphere.reshape((1, nb_pixels)),
                        qy_sphere.reshape((1, nb_pixels)),
                    )
                ).transpose()
            )

            # look for nan values
            nan_indices = np.argwhere(np.isnan(sphere_int))
            if debug:
                sphere_debug = np.copy(
                    sphere_int
                )  # create a copy to see also nans in the debugging plot

            #  remove nan values before calculating the cross-correlation function
            theta = np.delete(theta, nan_indices)
            phi = np.delete(phi, nan_indices)
            sphere_int = np.delete(sphere_int, nan_indices)

            # normalize the intensity by the median value
            # (remove the influence of the form factor)
            print(
                f"q={q_value:.3f}:",
                " normalizing by the median value",
                np.median(sphere_int),
            )
            sphere_int = sphere_int / np.median(sphere_int)

            theta_phi_int[dict_fields[counter]] = np.concatenate(
                (theta[:, np.newaxis], phi[:, np.newaxis], sphere_int[:, np.newaxis]),
                axis=1,
            )
            # update the number of points without nan
            nb_points.append(len(theta))
            print(
                f"q={q_value:.3f}:",
                " removing",
                nan_indices.size,
                "nan values,",
                nb_points[counter],
                "remain",
            )

            if debug:
                # calculate the stereographic projection
                stereo_proj, uv_labels = fu.calc_stereoproj_facet(
                    projection_axis=1,
                    radius_mean=q_value,
                    stereo_center=0,
                    vectors=np.concatenate(
                        (
                            qx_sphere[:, np.newaxis],
                            qz_sphere[:, np.newaxis],
                            qy_sphere[:, np.newaxis],
                        ),
                        axis=1,
                    ),
                )
                # plot the projection from the South pole
                fig, _ = gu.scatter_stereographic(
                    euclidian_u=stereo_proj[:, 0],
                    euclidian_v=stereo_proj[:, 1],
                    color=sphere_debug,
                    title="Projection from the South pole"
                    f" at q={q_value:.3f} (1/nm)",
                    uv_labels=uv_labels,
                    cmap=my_cmap,
                )
                fig.savefig(savedir + f"South pole_q={q_value:.3f}.png")
                plt.close(fig)

                # plot the projection from the North pole
                fig, _ = gu.scatter_stereographic(
                    euclidian_u=stereo_proj[:, 2],
                    euclidian_v=stereo_proj[:, 3],
                    color=sphere_debug,
                    title="Projection from the North pole"
                    f" at q={q_value:.3f} (1/nm)",
                    uv_labels=uv_labels,
                    cmap=my_cmap,
                )
                fig.savefig(savedir + f"North pole_q={q_value:.3f}.png")
                plt.close(fig)
                del sphere_debug

            del (
                qx_sphere,
                qz_sphere,
                qy_sphere,
                theta,
                phi,
                sphere_int,
                indices,
                nan_indices,
            )
            gc.collect()
    del qx, qy, qz, distances, data
    gc.collect()

    ############################################
    # calculate the cross-correlation function #
    ############################################
    if same_q:
        key_q2 = "q1"
        print(
            f"\nThe CCF will be calculated over {nb_points[0]:d} * {nb_points[0]:d}"
            f" points and {corr_count.shape[0]:d} angular bins"
        )
    else:
        key_q2 = "q2"
        print(
            f"\nThe CCF will be calculated over {nb_points[0]:d} * {nb_points[1]:d}"
            f" points and {corr_count.shape[0]:d} angular bins"
        )

    angular_bins = np.linspace(
        start=0, stop=np.pi, num=corr_count.shape[0], endpoint=False
    )

    start = time.time()
    if single_proc:
        for idx in range(nb_points[0]):
            ccf_uniq_val, counter_val, counter_indices = xcca.calc_ccf_polar(
                point=idx,
                q1_name="q1",
                q2_name=key_q2,
                bin_values=angular_bins,
                polar_azi_int=theta_phi_int,
            )
            collect_result_debug(ccf_uniq_val, counter_val, counter_indices)
    else:
        print("\nNumber of processors: ", mp.cpu_count())
        mp.freeze_support()
        pool = mp.Pool(mp.cpu_count())  # use this number of processes
        for idx in range(nb_points[0]):
            pool.apply_async(
                xcca.calc_ccf_polar,
                args=(idx, "q1", key_q2, angular_bins, theta_phi_int),
                callback=collect_result,
                error_callback=util.catch_error,
            )
        # close the pool and let all the processes complete
        pool.close()
        pool.join()  # postpones the execution of next line of code
        # until all processes in the queue are done.
    end = time.time()
    print(
        "\nTime ellapsed for the calculation of the CCF:",
        str(datetime.timedelta(seconds=int(end - start))),
    )

    # normalize the cross-correlation by the counter
    indices = np.nonzero(corr_count[:, 1])
    corr_count[indices, 0] = corr_count[indices, 0] / corr_count[indices, 1]

    #######################################
    # save the cross-correlation function #
    #######################################
    filename = (
        f"CCF_q1={q_xcca[0]:.3f}_q2={q_xcca[1]:.3f}"
        + f"_points{nb_points[0]:d}_interp{interp_factor:d}_res{angular_resolution:.3f}"
        + user_comment
    )
    np.savez_compressed(
        savedir + filename + ".npz",
        angles=180 * angular_bins / np.pi,
        ccf=corr_count[:, 0],
        points=corr_count[:, 1],
    )

    #######################################
    # plot the cross-correlation function #
    #######################################
    # find the y limit excluding the peaks at 0 and 180 degrees
    indices = np.argwhere(
        np.logical_and(
            (angular_bins >= 5 * np.pi / 180), (angular_bins <= 175 * np.pi / 180)
        )
    )
    ymax = 1.2 * corr_count[indices, 0].max()
    print(
        "Discarding CCF values with a zero counter:",
        (corr_count[:, 1] == 0).sum(),
        "points masked",
    )
    corr_count[(corr_count[:, 1] == 0), 0] = np.nan  # discard these values of the CCF

    fig, ax = plt.subplots()
    ax.plot(
        180 * angular_bins / np.pi,
        corr_count[:, 0],
        color="red",
        linestyle="-",
        markerfacecolor="blue",
        marker=".",
    )
    ax.set_xlim(0, 180)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel("Cross-correlation")
    ax.set_xticks(np.arange(0, 181, 30))
    ax.set_title(f"CCF at q1={q_xcca[0]:.3f} 1/nm  and q2={q_xcca[1]:.3f} 1/nm")
    fig.savefig(savedir + filename + ".png")

    _, ax = plt.subplots()
    ax.plot(
        180 * angular_bins / np.pi,
        corr_count[:, 1],
        linestyle="None",
        markerfacecolor="blue",
        marker=".",
    )
    ax.set_xlim(0, 180)
    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel("Number of points")
    ax.set_xticks(np.arange(0, 181, 30))
    ax.set_title("Points per angular bin")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main(user_comment=comment)
