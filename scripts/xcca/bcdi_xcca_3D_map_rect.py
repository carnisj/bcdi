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

import bcdi.graph.graph_utils as gu
import bcdi.postprocessing.facet_recognition as fu
import bcdi.utils.utilities as util
import bcdi.xcca.xcca_utils as xcca
from bcdi.graph.colormap import ColormapFactory

helptext = """
Calculate the angular cross-correlation in a 3D reciprocal space dataset over a range
in q values, at the same q value or between two different q values. The 3D dataset is
expected to be interpolated on an orthonormal grid.  The voxels belonging to a slice at
the defined q value are used for the calculation without further interpolation.

Input: the 3D dataset, an optional 3D mask, (qx, qy, qz) values

Laboratory frame convention (CXI): z downstream, y vertical up, x outboard.
Reciprocal space basis:            qx downstream, qz vertical up, qy outboard.
"""

datadir = "D:/data/P10_August2019_CDI/data/gold_2_2_2_00022/pynx/1_4_4_fullrange_xcca/"
savedir = "D:/data/P10_August2019_CDI/data/gold_2_2_2_00022/pynx/1_4_4_fullrange_xcca/"
comment = ""  # should start with _
angular_resolution = 0.5  # in degrees, angle between to adjacent points
# for the calculation of the cross-correlation
debug = True  # set to True to see more plots
origin_qspace = (
    330,
    204,
    330,
)  # origin of the reciprocal space in pixels in the order (qx, qz, qy)
q_range = np.arange(
    start=0.104, stop=0.205, step=0.004
)  # q values in 1/nm where to calculate the cross-correlation
same_q = False  # True if you want to calculate the cross-correlation at the same q.
# If False, it will calculate the
# cross-correlation between the first q value and all others
# the stop value is not included in np.arange()
hotpix_threshold = 1e9  # data above this threshold will be masked
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


def main(calc_self, user_comment):
    """
    Protection for multiprocessing.

    :param calc_self: if True, the cross-correlation will be calculated between
     same q-values
    :param user_comment: comment to include in the filename when saving results
    """
    ##########################
    # check input parameters #
    ##########################
    global corr_count, current_point
    if len(origin_qspace) != 3:
        raise ValueError("origin_qspace should be a tuple of 3 integer pixel values")
    if not isinstance(calc_self, bool):
        raise TypeError("unexpected type for calc_self")
    if len(q_range) <= 1:
        raise ValueError("at least 2 values are needed for q_range")

    print(f"the CCF map will be calculated for {len(q_range):d} q values: ")
    for _, item in enumerate(q_range):
        if calc_self:
            print(f"q1 = {item:.3f}  q2 = {item:.3f}")
        else:
            print(f"q1 = {q_range[0]:.3f}  q2 = {item:.3f}")
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
        ax.axvline(
            x=q_range[0], ymin=0, ymax=1, color="g", linestyle="--", label="q_start"
        )
        ax.axvline(
            x=q_range[-1], ymin=0, ymax=1, color="r", linestyle=":", label="q_stop"
        )
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

    q_int = {}  # create dictionnary
    dict_fields = ["q" + str(idx + 1) for idx, _ in enumerate(q_range)]
    # ['q1', 'q2', 'q3', ...]
    nb_points = []

    for counter, q_value in enumerate(q_range):
        indices = np.nonzero(
            np.logical_and((distances < q_value + dq), (distances > q_value - dq))
        )
        nb_voxels = indices[0].shape
        print(
            f"\nNumber of voxels for the sphere of radius q ={q_value:.3f} 1/nm:",
            nb_voxels,
        )

        qx_voxels = qx[indices[0]]  # qx downstream, axis 0
        qz_voxels = qz[indices[1]]  # qz vertical up, axis 1
        qy_voxels = qy[indices[2]]  # qy outboard, axis 2
        int_voxels = data[indices]

        if debug:
            # calculate the stereographic projection
            stereo_proj, uv_labels = fu.calc_stereoproj_facet(
                projection_axis=1,
                radius_mean=q_value,
                stereo_center=0,
                vectors=np.concatenate(
                    (
                        qx_voxels[:, np.newaxis],
                        qz_voxels[:, np.newaxis],
                        qy_voxels[:, np.newaxis],
                    ),
                    axis=1,
                ),
            )
            # plot the projection from the South pole
            fig, _ = gu.scatter_stereographic(
                euclidian_u=stereo_proj[:, 0],
                euclidian_v=stereo_proj[:, 1],
                color=int_voxels,
                title="Projection from the South pole" f" at q={q_value:.3f} (1/nm)",
                uv_labels=uv_labels,
                cmap=my_cmap,
            )
            fig.savefig(savedir + f"South pole_q={q_value:.3f}.png")
            plt.close(fig)

            # plot the projection from the North pole
            fig, _ = gu.scatter_stereographic(
                euclidian_u=stereo_proj[:, 2],
                euclidian_v=stereo_proj[:, 3],
                color=int_voxels,
                title="Projection from the North pole" f" at q={q_value:.3f} (1/nm)",
                uv_labels=uv_labels,
                cmap=my_cmap,
            )
            fig.savefig(savedir + f"North pole_q={q_value:.3f}.png")
            plt.close(fig)

        # look for nan values
        nan_indices = np.argwhere(np.isnan(int_voxels))

        #  remove nan values before calculating the cross-correlation function
        qx_voxels = np.delete(qx_voxels, nan_indices)
        qz_voxels = np.delete(qz_voxels, nan_indices)
        qy_voxels = np.delete(qy_voxels, nan_indices)
        int_voxels = np.delete(int_voxels, nan_indices)

        # normalize the intensity by the median value
        # (remove the influence of the form factor)
        print(
            f"q={q_value:.3f}:",
            " normalizing by the median value",
            np.median(int_voxels),
        )
        int_voxels = int_voxels / np.median(int_voxels)

        q_int[dict_fields[counter]] = np.concatenate(
            (
                qx_voxels[:, np.newaxis],
                qz_voxels[:, np.newaxis],
                qy_voxels[:, np.newaxis],
                int_voxels[:, np.newaxis],
            ),
            axis=1,
        )
        # update the number of points without nan
        nb_points.append(len(qx_voxels))
        print(
            f"q={q_value:.3f}:",
            " removing",
            nan_indices.size,
            "nan values,",
            nb_points[counter],
            "remain",
        )

        del qx_voxels, qz_voxels, qy_voxels, int_voxels, indices, nan_indices
        gc.collect()
    del qx, qy, qz, distances, data
    gc.collect()

    ############################################
    # calculate the cross-correlation function #
    ############################################
    cross_corr = np.empty((len(q_range), int(180 / angular_resolution), 2))
    angular_bins = np.linspace(
        start=0, stop=np.pi, num=corr_count.shape[0], endpoint=False
    )

    start = time.time()
    print("\nNumber of processors: ", mp.cpu_count())
    mp.freeze_support()

    for ind_q in range(len(q_range)):
        pool = mp.Pool(mp.cpu_count())  # use this number of processes
        if calc_self:
            key_q1 = "q" + str(ind_q + 1)
            key_q2 = key_q1
            print(
                "\n" + key_q2 + ": the CCF will be calculated over {:d} * {:d}"
                " points and {:d} angular bins".format(
                    nb_points[ind_q], nb_points[ind_q], corr_count.shape[0]
                )
            )
            for ind_point in range(nb_points[ind_q]):
                pool.apply_async(
                    xcca.calc_ccf_rect,
                    args=(ind_point, key_q1, key_q2, angular_bins, q_int),
                    callback=collect_result,
                    error_callback=util.catch_error,
                )
        else:
            key_q1 = "q1"
            key_q2 = "q" + str(ind_q + 1)
            print(
                "\n" + key_q2 + ": the CCF will be calculated over {:d} * {:d}"
                " points and {:d} angular bins".format(
                    nb_points[0], nb_points[ind_q], corr_count.shape[0]
                )
            )
            for ind_point in range(nb_points[0]):
                pool.apply_async(
                    xcca.calc_ccf_rect,
                    args=(ind_point, key_q1, key_q2, angular_bins, q_int),
                    callback=collect_result,
                    error_callback=util.catch_error,
                )

        # close the pool and let all the processes complete
        pool.close()
        pool.join()  # postpones the execution of next line of code
        # until all processes in the queue are done.

        # normalize the cross-correlation by the counter
        indices = np.nonzero(corr_count[:, 1])
        corr_count[indices, 0] = corr_count[indices, 0] / corr_count[indices, 1]
        cross_corr[ind_q, :, :] = corr_count

        # initialize the globals for the next q value
        corr_count = np.zeros(
            (int(180 / angular_resolution), 2)
        )  # corr_count is declared as a global, this should work
        current_point = 0

    end = time.time()
    print(
        "\nTime ellapsed for the calculation of the CCF map:",
        str(datetime.timedelta(seconds=int(end - start))),
    )

    #######################################
    # save the cross-correlation function #
    #######################################
    if calc_self:
        user_comment = user_comment + "_self"
    else:
        user_comment = user_comment + "_cross"
    filename = (
        f"CCFmap_qstart={q_range[0]:.3f}_qstop={q_range[-1]:.3f}"
        + f"_res{angular_resolution:.3f}"
        + user_comment
    )
    np.savez_compressed(
        savedir + filename + ".npz",
        angles=180 * angular_bins / np.pi,
        q_range=q_range,
        ccf=cross_corr[:, :, 0],
        points=cross_corr[:, :, 1],
    )

    #######################################
    # plot the cross-correlation function #
    #######################################
    # find the y limit excluding the peaks at 0 and 180 degrees
    indices = np.argwhere(
        np.logical_and(
            (angular_bins >= 20 * np.pi / 180), (angular_bins <= 160 * np.pi / 180)
        )
    )
    vmax = 1.2 * cross_corr[:, indices, 0].max()
    print(
        "Discarding CCF values with a zero counter:",
        (cross_corr[:, :, 1] == 0).sum(),
        "points masked",
    )
    cross_corr[(cross_corr[:, :, 1] == 0), 0] = (
        np.nan
    )  # discard these values of the CCF

    dq = q_range[1] - q_range[0]
    fig, ax = plt.subplots()
    plt0 = ax.imshow(
        cross_corr[:, :, 0],
        cmap=my_cmap,
        vmin=0,
        vmax=vmax,
        extent=[0, 180, q_range[-1] + dq / 2, q_range[0] - dq / 2],
    )  # extent (left, right, bottom, top)
    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel("q (nm$^{-1}$)")
    ax.set_xticks(np.arange(0, 181, 30))
    ax.set_yticks(q_range)
    ax.set_aspect("auto")
    if calc_self:
        ax.set_title(
            f"self CCF from q={q_range[0]:.3f} 1/nm  to q={q_range[-1]:.3f} 1/nm"
        )
    else:
        ax.set_title(
            f"cross CCF from q={q_range[0]:.3f} 1/nm  to q={q_range[-1]:.3f} 1/nm"
        )
    gu.colorbar(plt0, scale="linear", numticks=5)
    fig.savefig(savedir + filename + ".png")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main(calc_self=same_q, user_comment=comment)
