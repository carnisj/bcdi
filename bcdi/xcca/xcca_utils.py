# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np
import sys
sys.path.append('C:/Users/Jerome/Documents/myscripts/bcdi/')
sys.path.append('D:/myscripts/bcdi/')
import bcdi.graph.graph_utils as gu


def angular_avg(data, q_values, mask=None, origin=None, nb_bins=np.nan, debugging=False):
    """
    Calculate an angular average of a 3D reciprocal space dataset given q values and the position of the origin of the
    reciprocal space.

    :param data: 3D reciprocal space data gridded in the orthonormal frame (qx downstream, qz vertical up, qy outboard)
    :param q_values: tuple of 3 1-D arrays: (qx downstream, qz vertical up, qy outboard)
    :param mask: 3D array of the same shape as data. 1 for a masked voxel, 0 otherwise
    :param origin: position in pixels of the origin of the reciprocal space
    :param nb_bins: number of points where to calculate the average
    :param debugging: True to see plots
    :return: q_axis, angular mean average, angular median average
    """
    if len(q_values) != 3:
        raise ValueError("q_values should be a tuple of three 1D arrays")
    if data.ndim != 3:
        raise ValueError("data should be a 3D array")
    if mask is None:
        mask = np.zeros(data.shape)
    else:
        assert mask.shape == data.shape, "mask should have the same shape as data"
    qx, qz, qy = q_values
    nz, ny, nx = data.shape

    if len(qx) != nz or len(qz) != ny or len(qy) != nx:
        raise ValueError("size of q values incompatible with data shape")

    if origin is None:
        origin = (int(nz // 2), int(ny // 2), int(nx // 2))
    elif len(origin) != 3:
        raise ValueError("origin should be a tuple of 3 elements")

    if np.isnan(nb_bins):
        nb_bins = nz // 4

    # calculate the matrix of distances from the origin of reciprocal space
    distances = np.sqrt((qx[:, np.newaxis, np.newaxis] - qx[origin[0]]) ** 2 +
                        (qz[np.newaxis, :, np.newaxis] - qz[origin[1]]) ** 2 +
                        (qy[np.newaxis, np.newaxis, :] - qy[origin[2]]) ** 2)
    if debugging:
        gu.multislices_plot(distances, sum_frames=False, plot_colorbar=True, title='distances_q', scale='linear',
                            vmin=np.nan, vmax=np.nan, reciprocal_space=True, is_orthogonal=True)

    # average over spherical shells
    print('Distance max:', distances.max(), ' (1/nm) at voxel:',
          np.unravel_index(abs(distances).argmax(), distances.shape))
    print('Distance:', distances[origin[0], origin[1], origin[2]], ' (1/nm) at voxel:',
          origin)
    ang_avg = np.zeros(nb_bins)  # angular average using the mean value
    ang_median = np.zeros(nb_bins)  # angular average using the median value
    q_axis = np.linspace(0, distances.max(), endpoint=True, num=nb_bins + 1)  # in pixels or 1/nm

    for index in range(nb_bins):
        indices = np.logical_and((distances < q_axis[index + 1]), (distances >= q_axis[index]))
        temp_data = data[indices]
        temp_mask = mask[indices]

        ang_avg[index] = temp_data[np.logical_and((~np.isnan(temp_data)), (temp_mask != 1))].mean()
        ang_median[index] = np.median(temp_data[np.logical_and((~np.isnan(temp_data)), (temp_mask != 1))])

    q_axis = q_axis[:-1]

    # prepare for masking arrays - 'conventional' arrays won't do it
    y_mean = np.ma.array(ang_avg)
    y_median = np.ma.array(ang_median)
    # mask nan values
    y_mean_masked = np.ma.masked_where(np.isnan(y_mean), y_mean)
    y_median_masked = np.ma.masked_where(np.isnan(y_median), y_median)

    return q_axis, y_mean_masked, y_median_masked


# if __name__ == "__main__":
