# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr
"""
Functions related to reciprocal space averaging and XCCA.

XCCA stands for X-ray cross-correlation analysis.
"""

import numpy as np

from ..graph import graph_utils as gu
from ..utils import utilities as util
from ..utils import validation as valid


def angular_avg(
    data, q_values, mask=None, origin=None, nb_bins=np.nan, debugging=False
):
    """
    Calculate an angular average of a 3D reciprocal space dataset.

    It needs q values and the position of the origin of the reciprocal space.

    :param data: 3D reciprocal space data gridded in the orthonormal frame
     (qx downstream, qz vertical up, qy outboard)
    :param q_values: tuple of 3 1-D arrays: (qx downstream, qz vertical up, qy outboard)
    :param mask: 3D array of the same shape as data. 1 for a masked voxel, 0 otherwise
    :param origin: position in pixels of the origin of the reciprocal space
    :param nb_bins: number of points where to calculate the average
    :param debugging: True to see plots
    :return: q_axis, angular mean average, angular median average
    """
    if len(q_values) != 3:
        raise ValueError("q_values should be a tuple of three 1D arrays")
    valid.valid_ndarray(data, ndim=3)
    if mask is None:
        mask = np.zeros(data.shape)
    valid.valid_ndarray(mask, shape=data.shape)
    qx, qz, qy = q_values
    nz, ny, nx = data.shape

    if len(qx) != nz or len(qz) != ny or len(qy) != nx:
        raise ValueError("size of q values incompatible with data shape")

    if origin is None:
        origin = (int(nz // 2), int(ny // 2), int(nx // 2))
    elif len(origin) != 3:
        raise ValueError("origin should be a tuple of 3 elements")

    if np.isnan(nb_bins):
        nb_bins = 250

    # calculate the matrix of distances from the origin of reciprocal space
    distances = np.sqrt(
        (qx[:, np.newaxis, np.newaxis] - qx[origin[0]]) ** 2
        + (qz[np.newaxis, :, np.newaxis] - qz[origin[1]]) ** 2
        + (qy[np.newaxis, np.newaxis, :] - qy[origin[2]]) ** 2
    )
    if debugging:
        gu.multislices_plot(
            distances,
            sum_frames=False,
            plot_colorbar=True,
            title="distances_q",
            scale="linear",
            vmin=np.nan,
            vmax=np.nan,
            reciprocal_space=True,
            is_orthogonal=True,
        )

    # average over spherical shells
    print(
        "Distance max:",
        distances.max(),
        " (1/nm) at voxel:",
        np.unravel_index(abs(distances).argmax(), distances.shape),
    )
    print(
        "Distance:",
        distances[origin[0], origin[1], origin[2]],
        " (1/nm) at voxel:",
        origin,
    )
    ang_avg = np.zeros(nb_bins)  # angular average using the mean value
    ang_median = np.zeros(nb_bins)  # angular average using the median value
    q_axis = np.linspace(
        0, distances.max(), endpoint=True, num=nb_bins + 1
    )  # in pixels or 1/nm

    for index in range(nb_bins):
        indices = np.logical_and(
            (distances < q_axis[index + 1]), (distances >= q_axis[index])
        )
        temp_data = data[indices]
        temp_mask = mask[indices]

        ang_avg[index] = temp_data[
            np.logical_and((~np.isnan(temp_data)), (temp_mask != 1))
        ].mean()
        ang_median[index] = np.median(
            temp_data[np.logical_and((~np.isnan(temp_data)), (temp_mask != 1))]
        )

    q_axis = q_axis[:-1] + (q_axis[1] - q_axis[0]) / 2

    # prepare for masking arrays - 'conventional' arrays won't do it
    y_mean = np.ma.array(ang_avg)
    y_median = np.ma.array(ang_median)
    # mask nan values
    y_mean_masked = np.ma.masked_where(np.isnan(y_mean), y_mean)
    y_median_masked = np.ma.masked_where(np.isnan(y_median), y_median)

    return q_axis, y_mean_masked, y_median_masked


def calc_ccf_polar(point, q1_name, q2_name, bin_values, polar_azi_int):
    """
    Cross-correlate intensities at two q values, in polar coordinates.

    It calculates the cross-correlation of point with all other points at the second q
    value and sort the result.

    :param point: the reference point
    :param q1_name: key for the first q value in the dictionnary polar_azi_int
    :param q2_name: key for the second q value in the dictionnary polar_azi_int
    :param bin_values: in radians, angular bin values where to calculate the
     cross-correlation
    :param polar_azi_int: a dictionnary with fields 'q1', 'q2', ... Each field contains
     three 1D arrays: polar angle, azimuthal angle and intensity values for each point
    :return: the sorted cross-correlation values, angular bins indices and number of
     points contributing to the angular bins
    """
    # calculate the angle between the current point and all points from the second
    # q value (delta in [0 pi])
    delta_val = np.arccos(
        np.sin(polar_azi_int[q1_name][point, 0])
        * np.sin(polar_azi_int[q2_name][:, 0])
        * np.cos(polar_azi_int[q2_name][:, 1] - polar_azi_int[q1_name][point, 1])
        + np.cos(polar_azi_int[q1_name][point, 0])
        * np.cos(polar_azi_int[q2_name][:, 0])
    )

    # It can happen that the value in the arccos is outside [-1, 1] because of
    # the limited floating precision of Python, which result in delta_val = nan.
    # These points would contribute to the 0 and 180 degrees CCF, and can be neglected.

    # find the nearest angular bin value for each value of the array delta
    nearest_indices = util.find_nearest(
        test_values=delta_val,
        reference_array=bin_values,
        width=bin_values[1] - bin_values[0],
    )

    # update the counter of bin indices
    counter_indices, counter_val = np.unique(
        nearest_indices, return_counts=True
    )  # counter_indices are sorted

    # filter out -1 indices which correspond to no neighbour in the range defined by
    # width in find_nearest()
    counter_val = np.delete(counter_val, np.argwhere(counter_indices == -1))
    counter_indices = np.delete(counter_indices, np.argwhere(counter_indices == -1))

    # calculate the contribution to the cross-correlation for bins in counter_indices
    ccf_uniq_val = np.zeros(len(counter_indices))
    for idx, item in enumerate(counter_indices):
        ccf_uniq_val[idx] = (
            polar_azi_int[q1_name][point, 2]
            * polar_azi_int[q2_name][nearest_indices == item, 2]
        ).sum()

    return ccf_uniq_val, counter_val, counter_indices


def calc_ccf_rect(point, q1_name, q2_name, bin_values, q_int):
    """
    Cross-correlate intensities at two q values, in cartesian coordinates.

    It calculates the cross-correlation of point with all other points at the second q
    value and sort the result.

    :param point: the reference point
    :param q1_name: key for the first q value in the dictionnary polar_azi_int
    :param q2_name: key for the second q value in the dictionnary polar_azi_int
    :param bin_values: in radians, angular bin values where to calculate the
     cross-correlation
    :param q_int: a dictionnary with fields 'q1', 'q2', ... Each field contains four 1D
     arrays: qx, qy, qz and intensity values for each point
    :return: the sorted cross-correlation values, angular bins indices and number of
     points contributing to the angular bins
    """
    # calculate the angle between the current point and all points from the second
    # q value (delta in [0 pi])
    delta_val = np.arccos(
        np.divide(
            np.dot(q_int[q2_name][:, 0:3], q_int[q1_name][point, 0:3]),
            np.linalg.norm(q_int[q2_name][:, 0:3], axis=1)
            * np.linalg.norm(q_int[q1_name][point, 0:3]),
        )
    )
    # It can happen that the value in the arccos is outside [-1, 1] because of
    # the limited floating precision of Python, which result in delta_val = nan.
    # These points would contribute to the 0 and 180 degrees CCF, and can be neglected.

    # find the nearest angular bin value for each value of the array delta
    nearest_indices = util.find_nearest(
        test_values=delta_val,
        reference_array=bin_values,
        width=bin_values[1] - bin_values[0],
    )

    # update the counter of bin indices
    counter_indices, counter_val = np.unique(
        nearest_indices, return_counts=True
    )  # counter_indices are sorted

    # filter out -1 indices which correspond to no neighbour in the range defined by
    # width in find_nearest()
    counter_val = np.delete(counter_val, np.argwhere(counter_indices == -1))
    counter_indices = np.delete(counter_indices, np.argwhere(counter_indices == -1))

    # calculate the contribution to the cross-correlation for bins in counter_indices
    ccf_uniq_val = np.zeros(len(counter_indices))
    for idx, item in enumerate(counter_indices):
        ccf_uniq_val[idx] = (
            q_int[q1_name][point, 3] * q_int[q2_name][nearest_indices == item, 3]
        ).sum()

    return ccf_uniq_val, counter_val, counter_indices
