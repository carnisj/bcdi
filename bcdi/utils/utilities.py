# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#   (c) 06/2021-present : DESY CFEL
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr
"""Functions related to data loading, encoding, fitting, data manipulation."""

from collections import OrderedDict
import ctypes
from functools import reduce
import gc
import json
import h5py
from matplotlib import pyplot as plt
from numbers import Real
import numpy as np
import os
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy.stats import multivariate_normal

from ..graph import graph_utils as gu
from ..utils import validation as valid


class CustomEncoder(json.JSONEncoder):
    """Class to handle the serialization of np.ndarrays, sets."""

    def default(self, obj):
        """Override the JSONEncoder.default method to support more types."""
        if isinstance(obj, np.ndarray):
            return CustomEncoder.ndarray_to_list(obj)
            # Let the base class default method raise the TypeError
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

    @staticmethod
    def ndarray_to_list(obj):
        """Convert a numpy ndarray of any dimension to a nested list."""
        if not isinstance(obj, np.ndarray):
            raise TypeError("a numpy ndarray is expected")
        if obj.ndim == 1:
            return list(obj)
        output = []
        for idx in range(obj.shape[0]):
            output.append(CustomEncoder.ndarray_to_list(obj[idx]))
        return output


def bin_data(array, binning, debugging=False):
    """
    Rebin a 1D, 2D or 3D array.

    If its dimensions are not a multiple of binning, the array will be cropped.
    Adapted from PyNX.

    :param array: the array to resize
    :param binning: the rebin factor - pixels will be summed by groups of
     binning (x binning (x binning)). This can also be a tuple/list of rebin values
     along each axis, e.g. binning=(4,1,2) for a 3D array
    :param debugging: boolean, True to see plots
    :return: the binned array
    """
    ndim = array.ndim
    if isinstance(binning, int):
        binning = [binning] * ndim
    else:
        if ndim != len(binning):
            raise ValueError(
                "Rebin: number of dimensions does not agree with number "
                f"of rebin values: {binning}"
            )

    if ndim == 1:
        nx = len(array)
        array = array[: nx - (nx % binning[0])]
        sh = nx // binning[0], binning[0]
        newarray = array.reshape(sh).sum(axis=1)
    elif ndim == 2:
        ny, nx = array.shape
        array = array[: ny - (ny % binning[0]), : nx - (nx % binning[1])]
        sh = ny // binning[0], binning[0], nx // binning[1], binning[1]
        newarray = array.reshape(sh).sum(axis=(1, 3))
    elif ndim == 3:
        nz, ny, nx = array.shape
        array = array[
            : nz - (nz % binning[0]), : ny - (ny % binning[1]), : nx - (nx % binning[2])
        ]
        sh = (
            nz // binning[0],
            binning[0],
            ny // binning[1],
            binning[1],
            nx // binning[2],
            binning[2],
        )
        newarray = array.reshape(sh).sum(axis=(1, 3, 5))
    else:
        raise ValueError("Array should be 1D, 2D, or 3D")

    if debugging:
        print("array shape after cropping but before binning:", array.shape)
        print("array shape after binning:", newarray.shape)
        gu.combined_plots(
            tuple_array=(array, newarray),
            tuple_sum_frames=False,
            tuple_sum_axis=(1, 1),
            tuple_colorbar=True,
            tuple_width_v=None,
            tuple_width_h=None,
            tuple_vmin=0,
            tuple_vmax=np.nan,
            tuple_title=("array", "binned array"),
            tuple_scale="log",
            reciprocal_space=True,
        )
    return newarray


def catch_error(exception):
    """
    Process exception in asynchronous multiprocessing.

    :param exception: the arisen exception
    """
    print(exception)


def crop_pad(
    array, output_shape, pad_value=0, pad_start=None, crop_center=None, debugging=False
):
    """
    Crop or pad the 3D object depending on output_shape.

    :param array: 3D complex array to be padded
    :param output_shape: desired output shape (3D)
    :param pad_value: will pad using this value
    :param pad_start: for padding, tuple of 3 positions in pixel where the original
     array should be placed. If None, padding is symmetric along the respective axis
    :param crop_center: for cropping, [z, y, x] position in the original array
     (in pixels) of the center of the output array. If None, it will be set to the
     center of the original array
    :param debugging: set to True to see plots
    :type debugging: bool
    :return: myobj cropped or padded with zeros
    """
    if array.ndim != 3:
        raise ValueError("array should be a 3D array")

    nbz, nby, nbx = array.shape
    newz, newy, newx = output_shape

    if pad_start is None:
        pad_start = [(newz - nbz) // 2, (newy - nby) // 2, (newx - nbx) // 2]
    if len(pad_start) != 3:
        raise ValueError("pad_start should be a list or tuple of three indices")

    if crop_center is None:
        crop_center = [nbz // 2, nby // 2, nbx // 2]
    if len(crop_center) != 3:
        raise ValueError("crop_center should be a list or tuple of three indices")

    if debugging:
        print(f"array shape before crop/pad = {array.shape}")
        gu.multislices_plot(
            abs(array), sum_frames=True, scale="log", title="Before crop/pad"
        )

    # crop/pad along axis 0
    if newz >= nbz:  # pad
        temp_z = np.ones((output_shape[0], nby, nbx), dtype=array.dtype) * pad_value
        temp_z[pad_start[0] : pad_start[0] + nbz, :, :] = array
    else:  # crop
        if (crop_center[0] - output_shape[0] // 2 < 0) or (
            crop_center[0] + output_shape[0] // 2 > nbz
        ):
            raise ValueError("crop_center[0] incompatible with output_shape[0]")
        temp_z = array[
            crop_center[0] - newz // 2 : crop_center[0] + newz // 2 + newz % 2, :, :
        ]

    # crop/pad along axis 1
    if newy >= nby:  # pad
        temp_y = np.ones((newz, newy, nbx), dtype=array.dtype) * pad_value
        temp_y[:, pad_start[1] : pad_start[1] + nby, :] = temp_z
    else:  # crop
        if (crop_center[1] - output_shape[1] // 2 < 0) or (
            crop_center[1] + output_shape[1] // 2 > nby
        ):
            raise ValueError("crop_center[1] incompatible with output_shape[1]")
        temp_y = temp_z[
            :, crop_center[1] - newy // 2 : crop_center[1] + newy // 2 + newy % 2, :
        ]

    # crop/pad along axis 2
    if newx >= nbx:  # pad
        newobj = np.ones((newz, newy, newx), dtype=array.dtype) * pad_value
        newobj[:, :, pad_start[2] : pad_start[2] + nbx] = temp_y
    else:  # crop
        if (crop_center[2] - output_shape[2] // 2 < 0) or (
            crop_center[2] + output_shape[2] // 2 > nbx
        ):
            raise ValueError("crop_center[2] incompatible with output_shape[2]")
        newobj = temp_y[
            :, :, crop_center[2] - newx // 2 : crop_center[2] + newx // 2 + newx % 2
        ]

    if debugging:
        print(f"array shape after crop/pad = {newobj.shape}")
        gu.multislices_plot(
            abs(newobj), sum_frames=True, scale="log", title="After crop/pad"
        )
    return newobj


def crop_pad_2d(
    array, output_shape, pad_value=0, pad_start=None, crop_center=None, debugging=False
):
    """
    Crop or pad the 2D object depending on output_shape.

    :param array: 2D complex array to be padded
    :param output_shape: list of desired output shape [y, x]
    :param pad_value: will pad using this value
    :param pad_start: for padding, tuple of 2 positions in pixel where the original
     array should be placed. If None, padding is symmetric along the respective axis
    :param crop_center: for cropping, [y, x] position in the original array (in pixels)
     of the center of the ourput array. If None, it will be set to the center of the
     original array
    :param debugging: set to True to see plots
    :type debugging: bool
    :return: myobj cropped or padded with zeros
    """
    if array.ndim != 2:
        raise ValueError("array should be a 2D array")

    nby, nbx = array.shape
    newy, newx = output_shape

    if pad_start is None:
        pad_start = [(newy - nby) // 2, (newx - nbx) // 2]
    if len(pad_start) != 2:
        raise ValueError("pad_start should be a list or tuple of two indices")

    if crop_center is None:
        crop_center = [nby // 2, nbx // 2]
    if len(crop_center) != 2:
        raise ValueError("crop_center should be a list or tuple of two indices")

    if debugging:
        gu.imshow_plot(
            abs(array), sum_frames=True, scale="log", title="Before crop/pad"
        )
    # crop/pad along axis 0
    if newy >= nby:  # pad
        temp_y = np.ones((output_shape[0], nbx), dtype=array.dtype) * pad_value
        temp_y[pad_start[0] : pad_start[0] + nby, :] = array
    else:  # crop
        if (crop_center[0] - output_shape[0] // 2 < 0) or (
            crop_center[0] + output_shape[0] // 2 > nby
        ):
            raise ValueError("crop_center[0] incompatible with output_shape[0]")
        temp_y = array[
            crop_center[0] - newy // 2 : crop_center[0] + newy // 2 + newy % 2, :
        ]

    # crop/pad along axis 1
    if newx >= nbx:  # pad
        newobj = np.ones((newy, newx), dtype=array.dtype) * pad_value
        newobj[:, pad_start[1] : pad_start[1] + nbx] = temp_y
    else:  # crop
        if (crop_center[1] - output_shape[1] // 2 < 0) or (
            crop_center[1] + output_shape[1] // 2 > nbx
        ):
            raise ValueError("crop_center[1] incompatible with output_shape[1]")
        newobj = temp_y[
            :, crop_center[1] - newx // 2 : crop_center[1] + newx // 2 + newx % 2
        ]

    if debugging:
        gu.imshow_plot(abs(array), sum_frames=True, scale="log", title="After crop/pad")
    return newobj


def crop_pad_1d(
    array,
    output_length,
    pad_value=0,
    pad_start=None,
    crop_center=None,
    extrapolate=False,
):
    """
    Crop or pad the 2D object depending on output_shape.

    :param array: 1D complex array to be padded
    :param output_length: int desired output length
    :param pad_value: will pad using this value
    :param pad_start: for padding, position in pixel where the original array should be
     placed. If None, padding is symmetric
    :param crop_center: for cropping, position in pixels in the original array of the
     center of the ourput array. If None, it will be set to the center of the
     original array
    :param extrapolate: set to True to extrapolate using the current spacing
     (supposed constant)
    :return: myobj cropped or padded
    """
    if array.ndim != 1:
        raise ValueError("array should be 1D")

    nbx = array.shape[0]
    newx = output_length

    if pad_start is None:
        pad_start = (newx - nbx) // 2

    if crop_center is None:
        crop_center = nbx // 2

    if newx >= nbx:  # pad
        if not extrapolate:
            newobj = np.ones(output_length, dtype=array.dtype) * pad_value
            newobj[pad_start : pad_start + nbx] = array
        else:
            spacing = array[1] - array[0]
            pad_start = array[0] - ((newx - nbx) // 2) * spacing
            newobj = pad_start + np.arange(newx) * spacing
    else:  # crop
        if (crop_center - output_length // 2 < 0) or (
            crop_center + output_length // 2 > nbx
        ):
            raise ValueError("crop_center incompatible with output_length")
        newobj = array[crop_center - newx // 2 : crop_center + newx // 2 + newx % 2]

    return newobj


def decode_json(dct):
    """
    Define the parameter object_hook in json.load function, supporting various types.

    :param dct: the input dictionary of strings
    :return: a dictionary
    """
    for key, val in dct.items():
        if isinstance(val, list):
            try:
                dct[key] = np.asarray(val)
            except TypeError:
                pass
    return dct


def dos2unix(input_file, output_file):
    """
    Convert DOS linefeeds (crlf) to UNIX (lf).

    :param input_file: the original filename (absolute path)
    :param output_file: the output filename (absolute path) where to save
    """
    with open(input_file, "rb") as infile:
        content = infile.read()
    with open(output_file, "wb") as output:
        for row in content.splitlines():
            output.write(row + str.encode("\n"))


def find_nearest(reference_array, test_values, width=None):
    """
    Find the indices where original_array is nearest to array_values.

    :param reference_array: a 1D array where to look for the nearest values
    :param test_values: a number or a 1D array of numbers to be tested
    :param width: if not None, it will look for the nearest element within the range
     [x-width/2, x+width/2[
    :return: index or indices from original_array nearest to values of length
     len(test_values). Returns (index - 1). If there is no nearest neighbour in the
     range defined by width.
    """
    original_array, test_values = np.asarray(reference_array), np.asarray(test_values)

    if original_array.ndim != 1:
        raise ValueError("original_array should be 1D")
    if test_values.ndim > 1:
        raise ValueError("array_values should be a number or a 1D array")
    if test_values.ndim == 0:
        nearest_index = (np.abs(original_array - test_values)).argmin()
        return nearest_index

    nb_values = len(test_values)
    nearest_index = np.zeros(nb_values, dtype=int)
    for idx in range(nb_values):
        nearest_index[idx] = (np.abs(original_array - test_values[idx])).argmin()
    if width is not None:
        for idx in range(nb_values):
            if (
                reference_array[nearest_index[idx]] >= test_values[idx] + width / 2
            ) or (reference_array[nearest_index[idx]] < test_values[idx] - width / 2):
                # no neighbour in the range defined by width
                nearest_index[idx] = -1
    return nearest_index


def fit3d_poly1(x_axis, a, b, c, d):
    """
    Calculate the 1st order polynomial function on points in a 3D grid.

    :param x_axis: (3xN) tuple or array of 3D coordinates
    :param a: offset
    :param b: 1st order parameter for the 1st coordinate
    :param c: 1st order parameter for the 2nd coordinate
    :param d: 1st order parameter for the 3rd coordinate
    :return: the 1st order polynomial calculated on x_axis
    """
    return a + b * x_axis[0] + c * x_axis[1] + d * x_axis[2]


def fit3d_poly2(x_axis, a, b, c, d, e, f, g):
    """
    Calculate the 2nd order polynomial function on points in a 3D grid.

    :param x_axis: (3xN) tuple or array of 3D coordinates
    :param a: offset
    :param b: 1st order parameter for the 1st coordinate
    :param c: 1st order parameter for the 2nd coordinate
    :param d: 1st order parameter for the 3rd coordinate
    :param e: 2nd order parameter for the 1st coordinate
    :param f: 2nd order parameter for the 2nd coordinate
    :param g: 2nd order parameter for the 3rd coordinate
    :return: the 2nd order polynomial calculated on x_axis
    """
    return (
        a
        + b * x_axis[0]
        + c * x_axis[1]
        + d * x_axis[2]
        + e * x_axis[0] ** 2
        + f * x_axis[1] ** 2
        + g * x_axis[2] ** 2
    )


def fit3d_poly3(x_axis, a, b, c, d, e, f, g, h, i, j):
    """
    Calculate the 3rd order polynomial function on points in a 3D grid.

    :param x_axis: (3xN) tuple or array of 3D coordinates
    :param a: offset
    :param b: 1st order parameter for the 1st coordinate
    :param c: 1st order parameter for the 2nd coordinate
    :param d: 1st order parameter for the 3rd coordinate
    :param e: 2nd order parameter for the 1st coordinate
    :param f: 2nd order parameter for the 2nd coordinate
    :param g: 2nd order parameter for the 3rd coordinate
    :param h: 3rd order parameter for the 1st coordinate
    :param i: 3th order parameter for the 2nd coordinate
    :param j: 3th order parameter for the 3rd coordinate
    :return: the 3rd order polynomial calculated on x_axis
    """
    return (
        a
        + b * x_axis[0]
        + c * x_axis[1]
        + d * x_axis[2]
        + e * x_axis[0] ** 2
        + f * x_axis[1] ** 2
        + g * x_axis[2] ** 2
        + h * x_axis[0] ** 3
        + i * x_axis[1] ** 3
        + j * x_axis[2] ** 3
    )


def fit3d_poly4(x_axis, a, b, c, d, e, f, g, h, i, j, k, m, n):
    """
    Calculate the 4th order polynomial function on points in a 3D grid.

    :param x_axis: (3xN) tuple or array of 3D coordinates
    :param a: offset
    :param b: 1st order parameter for the 1st coordinate
    :param c: 1st order parameter for the 2nd coordinate
    :param d: 1st order parameter for the 3rd coordinate
    :param e: 2nd order parameter for the 1st coordinate
    :param f: 2nd order parameter for the 2nd coordinate
    :param g: 2nd order parameter for the 3rd coordinate
    :param h: 3rd order parameter for the 1st coordinate
    :param i: 3th order parameter for the 2nd coordinate
    :param j: 3th order parameter for the 3rd coordinate
    :param k: 4th order parameter for the 1st coordinate
    :param m: 4th order parameter for the 2nd coordinate
    :param n: 4th order parameter for the 3rd coordinate
    :return: the 4th order polynomial calculated on x_axis
    """
    return (
        a
        + b * x_axis[0]
        + c * x_axis[1]
        + d * x_axis[2]
        + e * x_axis[0] ** 2
        + f * x_axis[1] ** 2
        + g * x_axis[2] ** 2
        + h * x_axis[0] ** 3
        + i * x_axis[1] ** 3
        + j * x_axis[2] ** 3
        + k * x_axis[0] ** 4
        + m * x_axis[1] ** 4
        + n * x_axis[2] ** 4
    )


def function_lmfit(params, x_axis, distribution, iterator=0):
    """
    Calculate distribution defined by lmfit Parameters.

    :param params: a lmfit Parameters object
    :param x_axis: where to calculate the function
    :param distribution: the distribution to use
    :param iterator: the index of the relevant parameters
    :return: the gaussian function calculated at x_axis positions
    """
    if distribution == "gaussian":
        amp = params["amp_%i" % (iterator + 1)].value
        cen = params["cen_%i" % (iterator + 1)].value
        sig = params["sig_%i" % (iterator + 1)].value
        return gaussian(x_axis=x_axis, amp=amp, cen=cen, sig=sig)
    if distribution == "skewed_gaussian":
        amp = params["amp_%i" % (iterator + 1)].value
        loc = params["loc_%i" % (iterator + 1)].value
        sig = params["sig_%i" % (iterator + 1)].value
        alpha = params["alpha_%i" % (iterator + 1)].value
        return skewed_gaussian(x_axis=x_axis, amp=amp, loc=loc, sig=sig, alpha=alpha)
    if distribution == "lorentzian":
        amp = params["amp_%i" % (iterator + 1)].value
        cen = params["cen_%i" % (iterator + 1)].value
        sig = params["sig_%i" % (iterator + 1)].value
        return lorentzian(x_axis=x_axis, amp=amp, cen=cen, sig=sig)
    if distribution == "pseudovoigt":
        amp = params["amp_%i" % (iterator + 1)].value
        cen = params["cen_%i" % (iterator + 1)].value
        sig = params["sig_%i" % (iterator + 1)].value
        ratio = params["ratio_%i" % (iterator + 1)].value
        return pseudovoigt(x_axis, amp=amp, cen=cen, sig=sig, ratio=ratio)
    raise ValueError(distribution + " not implemented")


def gaussian(x_axis, amp, cen, sig):
    """
    Gaussian line shape.

    :param x_axis: where to calculate the function
    :param amp: the amplitude of the Gaussian
    :param cen: the position of the center
    :param sig: HWHM of the Gaussian
    :return: the Gaussian line shape at x_axis
    """
    return amp * np.exp(-((x_axis - cen) ** 2) / (2.0 * sig ** 2))


def gaussian_window(window_shape, sigma=0.3, mu=0.0, voxel_size=None, debugging=False):
    """
    Create a 2D or 3D Gaussian window using scipy.stats.multivariate_normal.

    :param window_shape: shape of the window
    :param sigma: float, sigma of the distribution
    :param mu: float, mean of the distribution
    :param voxel_size: tuple, voxel size in each dimension corresponding to
     window_shape. If None, it will default to 1/window_shape[ii] for each dimension
     so that it is independent of the shape of the window
    :param debugging: True to see plots
    :return: the Gaussian window
    """
    # check parameters
    valid.valid_container(
        window_shape,
        container_types=(tuple, list, np.ndarray),
        min_length=2,
        max_length=3,
        min_excluded=0,
        name="window_shape",
    )
    ndim = len(window_shape)
    valid.valid_item(sigma, allowed_types=Real, min_excluded=0, name="sigma")
    valid.valid_item(mu, allowed_types=Real, name="mu")
    valid.valid_container(
        voxel_size,
        container_types=(tuple, list, np.ndarray),
        length=ndim,
        allow_none=True,
        item_types=Real,
        min_excluded=0,
        name="voxel_size",
    )
    valid.valid_item(debugging, allowed_types=bool, name="debugging")

    # define sigma and mu in ndim
    sigma = np.repeat(sigma, ndim)
    mu = np.repeat(mu, ndim)

    # check the voxel size
    if voxel_size is None:
        voxel_size = [1 / pixel_nb for pixel_nb in window_shape]

    if ndim == 2:
        nby, nbx = window_shape
        grid_y, grid_x = np.meshgrid(
            np.linspace(-nby, nby, nby) * voxel_size[0],
            np.linspace(-nbx, nbx, nbx) * voxel_size[1],
            indexing="ij",
        )
        covariance = np.diag(sigma ** 2)
        window = multivariate_normal.pdf(
            np.column_stack([grid_y.flat, grid_x.flat]), mean=mu, cov=covariance
        )
        del grid_y, grid_x
        gc.collect()
        window = window.reshape((nby, nbx))

    else:  # 3D
        nbz, nby, nbx = window_shape
        grid_z, grid_y, grid_x = np.meshgrid(
            np.linspace(-nbz, nbz, nbz) * voxel_size[0],
            np.linspace(-nby, nby, nby) * voxel_size[1],
            np.linspace(-nbx, nbx, nbx) * voxel_size[2],
            indexing="ij",
        )
        covariance = np.diag(sigma ** 2)
        window = multivariate_normal.pdf(
            np.column_stack([grid_z.flat, grid_y.flat, grid_x.flat]),
            mean=mu,
            cov=covariance,
        )
        del grid_z, grid_y, grid_x
        gc.collect()
        window = window.reshape((nbz, nby, nbx))

    # rescale the gaussian if voxel size was provided
    if voxel_size is not None:
        window = window * reduce((lambda x, y: x * y), window_shape) ** 2

    if debugging:
        gu.multislices_plot(
            array=window,
            sum_frames=False,
            plot_colorbar=True,
            scale="linear",
            title="Gaussian window",
            reciprocal_space=False,
            is_orthogonal=True,
        )

    return window


def image_to_ndarray(filename, convert_grey=True, cmap=None, debug=False):
    """
    Convert an image to a numpy array using pillow.

    Matplotlib only supports the PNG format.

    :param filename: absolute path of the image to open
    :param convert_grey: if True and the number of layers is 3, it will be converted
     to a single layer of grey
    :param cmap: colormap for the plots
    :param debug: True to see plots
    :return:
    """
    from PIL import Image

    if cmap is None:
        cmap = gu.Colormap(bad_color="1.0").cmap

    im = Image.open(filename)

    array = np.asarray(im)
    if array.ndim == 3 and convert_grey:
        print("converting image to gray")
        array = rgb2gray(array)

    print(f"Image shape after conversion to ndarray: {array.shape}")
    if debug:
        gu.imshow_plot(
            array, sum_axis=2, plot_colorbar=True, cmap=cmap, reciprocal_space=False
        )
    return array


def in_range(point, extent):
    """
    Check if a point in within a certain range.

    It returns a boolean depending on whether point is in the indices range defined by
    extent or not.

    :param point: tuple of two real numbers (2D case) (y, x) or three real numbers
     (3D case) (z, y, x) representing
     the voxel indices to be tested
    :param extent: tuple of four integers (2D case) (y_start, y_stop, x_tart, x_stop)
     or six integers (3D case)
     (z_start, z_stop, y_start, y_stop, x_tart, x_stop) representing the range of
     valid indices
    :return: True if point belongs to extent, False otherwise
    """
    # check parameters
    valid.valid_container(
        point,
        container_types=(list, tuple, np.ndarray),
        item_types=Real,
        name="utilities.in_range",
    )
    ndim = len(point)
    if ndim not in {2, 3}:
        raise ValueError("point should be 2D or 3D")
    valid.valid_container(
        extent,
        container_types=(list, tuple, np.ndarray),
        length=2 * ndim,
        item_types=int,
        name="utilities.in_range",
    )

    # check the appartenance to the defined extent
    if ndim == 2:
        if (extent[0] <= point[0] <= extent[1]) and (
            extent[2] <= point[1] <= extent[3]
        ):
            return True
    else:
        if (
            (extent[0] <= point[0] <= extent[1])
            and (extent[2] <= point[1] <= extent[3])
            and (extent[4] <= point[2] <= extent[5])
        ):
            return True
    return False


def is_numeric(string):
    """
    Return True is the string represents a number.

    :param string: the string to be checked
    :return: True of False
    """
    try:
        float(string)
        return True
    except ValueError:
        return False


def linecut(array, point, direction, direction_basis="voxel", voxel_size=1):
    """
    Calculate the linecut through a 2D or 3D array in some direction passing by a point.

    :param array: 2D or 3D numpy array from which the linecut will be extracted
    :param point: tuple of three integral indices, point by which the linecut pass.
    :param direction: list of 2 (for 2D) or 3 (for 3D) vector components,
     direction of the linecut in units of pixels
    :param direction_basis: 'orthonormal' if the vector direction is expressed in
     an orthonormal basis. In that case it
     will be corrected for the different voxel sizes in each direction. 'voxel' if
     direction is expressed in the non-orthonormal basis defined by the voxel sizes
     in each direction.
    :param voxel_size: real positive number or tuple of 2 (for 2D) or 3 (for 3D)
     real positive numbers representing the voxel size in each dimension.
    :return: distances (1D array, distance along the linecut in the unit given by
     voxel_size) and cut (1D array, linecut through array in direction passing by point)
    """
    # check parameters
    ndim = array.ndim
    if ndim not in {2, 3}:
        raise ValueError(f"Number of dimensions = {ndim}, expected 2 or 3")
    if ndim == 2:
        nby, nbx = array.shape
        nbz = 0
    else:
        nbz, nby, nbx = array.shape
    direction = list(direction)
    valid.valid_container(
        direction,
        container_types=(list, tuple, np.ndarray),
        length=ndim,
        item_types=Real,
        name="utilities.linecut",
    )
    valid.valid_container(
        point,
        container_types=(list, tuple, np.ndarray),
        length=ndim,
        item_types=int,
        min_included=0,
        name="utilities.linecut",
    )
    point = tuple(point)  # point needs to be hashable
    if direction_basis not in {"orthonormal", "voxel"}:
        raise ValueError(
            f"unknown value {direction_basis} for direction_basis,"
            ' allowed are "voxel" and "orthonormal"'
        )
    if isinstance(voxel_size, Real):
        voxel_size = (voxel_size,) * ndim
    valid.valid_container(
        voxel_size,
        container_types=(list, tuple, np.ndarray),
        length=ndim,
        item_types=Real,
        min_excluded=0,
        name="utilities.linecut",
    )

    # normalize the vector direction, eventually correct it for anisotropic voxel sizes
    if direction_basis == "orthonormal":
        direction = [direction[i] * voxel_size[i] for i in range(ndim)]
    direction = direction / np.linalg.norm(direction)

    # initialize parameters
    cut_points = []
    # calculate the indices of the voxels on one side of the linecut
    go_on = True
    n = 1
    while go_on:
        if ndim == 2:
            next_point = (
                int(np.rint(point[0] - n * direction[0])),
                int(np.rint(point[1] - n * direction[1])),
            )
            go_on = in_range(next_point, (0, nby - 1, 0, nbx - 1))
        else:
            next_point = (
                int(np.rint(point[0] - n * direction[0])),
                int(np.rint(point[1] - n * direction[1])),
                int(np.rint(point[2] - n * direction[2])),
            )
            go_on = in_range(next_point, (0, nbz - 1, 0, nby - 1, 0, nbx - 1))
        if go_on:
            cut_points.append(next_point)
            n += 1
    # flip the indices so that the increasing direction is consistent with
    # the second half of the linecut
    cut_points = cut_points[::-1]
    # append the point by which the linecut pass
    cut_points.append(point)
    # calculate the indices of the voxels on the other side of the linecut
    go_on = True
    n = 1
    while go_on:
        if ndim == 2:
            next_point = (
                int(np.rint(point[0] + n * direction[0])),
                int(np.rint(point[1] + n * direction[1])),
            )
            go_on = in_range(next_point, (0, nby - 1, 0, nbx - 1))
        else:
            next_point = (
                int(np.rint(point[0] + n * direction[0])),
                int(np.rint(point[1] + n * direction[1])),
                int(np.rint(point[2] + n * direction[2])),
            )
            go_on = in_range(next_point, (0, nbz - 1, 0, nby - 1, 0, nbx - 1))
        if go_on:
            cut_points.append(next_point)
            n += 1

    # remove duplicates
    cut_points = list(OrderedDict.fromkeys(cut_points))
    # transform cut_points in an appropriate way for slicing array
    if ndim == 2:
        indices = ([item[0] for item in cut_points], [item[1] for item in cut_points])
    else:
        indices = (
            [item[0] for item in cut_points],
            [item[1] for item in cut_points],
            [item[2] for item in cut_points],
        )
    # indices is a tuple of ndim ndarrays that can be used to directly slice obj
    cut = array[indices]  # cut is now 1D

    # calculate the distance along the linecut given the voxel size
    distances = []
    for idx in range(len(cut)):
        if ndim == 2:
            distance = np.sqrt(
                ((indices[0][idx] - indices[0][0]) * voxel_size[0]) ** 2
                + ((indices[1][idx] - indices[1][0]) * voxel_size[1]) ** 2
            )
        else:  # 3D
            distance = np.sqrt(
                ((indices[0][idx] - indices[0][0]) * voxel_size[0]) ** 2
                + ((indices[1][idx] - indices[1][0]) * voxel_size[1]) ** 2
                + ((indices[2][idx] - indices[2][0]) * voxel_size[2]) ** 2
            )
        distances.append(distance)

    return np.asarray(distances), cut


def load_file(file_path, fieldname=None):
    """
    Load a file.

    In case of .cxi or .h5 file, it will use a default path to the data. 'fieldname'
    is used only for .npz files.

    :param file_path: the path of the reconstruction to load.
     Format supported: .npy .npz .cxi .h5
    :param fieldname: the name of the field to be loaded
    :return: the loaded data and the extension of the file
    """
    _, extension = os.path.splitext(file_path)
    if extension == ".npz":  # could be anything
        if fieldname is None:  # output of PyNX phasing
            npzfile = np.load(file_path)
            dataset = npzfile[list(npzfile.files)[0]]
        else:  # could be anything
            try:
                dataset = np.load(file_path)[fieldname]
                return dataset, extension
            except KeyError:
                npzfile = np.load(file_path)
                dataset = npzfile[list(npzfile.files)[0]]
    elif extension == ".npy":  # could be anything
        dataset = np.load(file_path)
    elif extension == ".cxi":  # output of PyNX phasing
        h5file = h5py.File(file_path, "r")
        # group_key = list(h5file.keys())[1]
        # subgroup_key = list(h5file[group_key])
        # dataset = h5file['/'+group_key+'/'+subgroup_key[0]+'/data'].value
        dataset = h5file["/entry_1/data_1/data"][()]
    elif extension == ".h5":  # modes.h5
        h5file = h5py.File(file_path, "r")
        group_key = list(h5file.keys())[0]
        if group_key == "mask":  # mask object for Nanomax data
            dataset = h5file["/" + group_key][:]
        else:  # modes.h5 file output of PyNX phase retrieval
            subgroup_key = list(h5file[group_key])
            dataset = h5file["/" + group_key + "/" + subgroup_key[0] + "/data"][
                0
            ]  # select only first mode
    else:
        raise ValueError(
            "File format not supported: "
            "can load only '.npy', '.npz', '.cxi' or '.h5' files"
        )

    if fieldname == "modulus":
        dataset = abs(dataset)
    elif fieldname == "angle":
        dataset = np.angle(dataset)
    elif fieldname is None:
        pass
    else:
        raise ValueError('"field" parameter settings is not valid')
    return dataset, extension


def lorentzian(x_axis, amp, cen, sig):
    """
    Lorentzian line shape.

    :param x_axis: where to calculate the function
    :param amp: the amplitude of the Lorentzian
    :param cen: the position of the center
    :param sig: HWHM of the Lorentzian
    :return: the Lorentzian line shape at x_axis
    """
    return amp / (sig * np.pi) / (1 + (x_axis - cen) ** 2 / (sig ** 2))


def objective_lmfit(params, x_axis, data, distribution):
    """
    Calculate the total residual for fits to several data sets.

    Data sets should be held in a 2-D array (1 row per dataset).

    :param params: a lmfit Parameters object
    :param x_axis: where to calculate the distribution
    :param data: data to fit
    :param distribution: distribution to use for fitting
    :return: the residuals of the fit of data using the parameters
    """
    if len(data.shape) == 1:  # single dataset
        data = data[np.newaxis, :]
        x_axis = x_axis[np.newaxis, :]
    if data.ndim != 2:
        raise ValueError("Data should be a 2D stack of 1D datasets (1 per row)")
    ndata, nx = data.shape
    resid = 0.0 * data[:]
    # make residual per data set
    for idx in range(ndata):
        resid[idx, :] = data[idx, :] - function_lmfit(
            params=params,
            iterator=idx,
            x_axis=x_axis[idx, :],
            distribution=distribution,
        )
    # now flatten this to a 1D array, as minimize() needs
    return resid.flatten()


def line(x_array, a, b):
    """
    Return y values such that y = a*x + b.

    :param x_array: a 1D numpy array of length N
    :param a: coefficient for x values
    :param b: constant offset
    :return: an array of length N containing the y values
    """
    return a * x_array + b


def plane(xy_array, a, b, c):
    """
    Return z values such that z = a*x + b*y + c.

    :param xy_array: a (2xN) numpy array, x values being the first row and
     y values the second row
    :param a: coefficient for x values
    :param b:  coefficient for y values
    :param c: constant offset
    :return: an array of length N containing the z values
    """
    return a * xy_array[0, :] + b * xy_array[1, :] + c


def plane_dist(indices, params):
    """
    Calculate the distance of an ensemble of voxels to a plane given by its parameters.

    :param indices: a (3xN) numpy array, x values being the 1st row,
     y values the 2nd row and z values the 3rd row
    :param params: a tuple of coefficient (a, b, c, d) such that ax+by+cz+d=0
    :return: a array of shape (N,) containing the distance to the plane for each voxel
    """
    distance = np.zeros(len(indices[0]))
    plane_normal = np.array(
        [params[0], params[1], params[2]]
    )  # normal is [a, b, c] if ax+by+cz+d=0
    for point in range(len(indices[0])):
        distance[point] = abs(
            params[0] * indices[0, point]
            + params[1] * indices[1, point]
            + params[2] * indices[2, point]
            + params[3]
        ) / np.linalg.norm(plane_normal)
    return distance


def plane_fit(indices, label="", threshold=1, debugging=False):
    """
    Fit a plane to the voxels defined by indices.

    :param indices: a (3xN) numpy array, x values being the 1st row,
     y values the 2nd row and z values the 3rd row
    :param label: int, label of the plane used for the title in the debugging plot
    :param threshold: the fit will be considered as good if the mean distance
     of the voxels to the plane is smaller than this value
    :param debugging: True to see plots
    :return: a tuple of coefficient (a, b, c, d) such that ax+by+cz+d=0,
     the matrix of covariant values
    """
    valid_plane = True
    indices = np.asarray(indices)
    params3d, pcov3d = curve_fit(plane, indices[0:2, :], indices[2, :])
    std_param3d = np.sqrt(np.diag(pcov3d))
    params = (-params3d[0], -params3d[1], 1, -params3d[2])
    std_param = (std_param3d[0], std_param3d[1], 0, std_param3d[2])

    if debugging:
        _, ax = gu.scatter_plot(
            np.transpose(indices),
            labels=("axis 0", "axis 1", "axis 2"),
            title="Points and fitted plane " + str(label),
        )
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        meshx, meshy = np.meshgrid(
            np.arange(xlim[0], xlim[1] + 1, 1), np.arange(ylim[0], ylim[1] + 1, 1)
        )
        # meshx varies horizontally, meshy vertically
        meshz = plane(
            np.vstack((meshx.flatten(), meshy.flatten())),
            params3d[0],
            params3d[1],
            params3d[2],
        ).reshape(meshx.shape)
        ax.plot_wireframe(meshx, meshy, meshz, color="k")
        plt.pause(0.1)

    # calculate the mean distance to the fitted plane
    distance = plane_dist(indices=indices, params=params)
    print(
        f"plane fit using z=a*x+b*y+c: dist.mean()={distance.mean():.2f},"
        f"  dist.std()={distance.std():.2f}"
    )

    if distance.mean() > threshold and distance.mean() / distance.std() > 1:
        # probably z does not depend on x and y, try to fit  y = a*x + b
        print("z=a*x+b*y+c: z may not depend on x and y")
        params2d, pcov2d = curve_fit(line, indices[0, :], indices[1, :])
        std_param2d = np.sqrt(np.diag(pcov2d))
        params = (-params2d[0], 1, 0, -params2d[1])
        std_param = (std_param2d[0], 0, 0, std_param2d[1])
        if debugging:
            _, ax = gu.scatter_plot(
                np.transpose(indices),
                labels=("axis 0", "axis 1", "axis 2"),
                title="Points and fitted plane " + str(label),
            )
            xlim = ax.get_xlim()
            zlim = ax.get_zlim()
            meshx, meshz = np.meshgrid(
                np.arange(xlim[0], xlim[1] + 1, 1), np.arange(zlim[0], zlim[1] + 1, 1)
            )
            meshy = line(x_array=meshx.flatten(), a=params2d[0], b=params2d[1]).reshape(
                meshx.shape
            )
            ax.plot_wireframe(meshx, meshy, meshz, color="k")
            plt.pause(0.1)
        # calculate the mean distance to the fitted plane
        distance = plane_dist(indices=indices, params=params)
        print(
            f"plane fit using y=a*x+b: dist.mean()={distance.mean():.2f},"
            f"  dist.std()={distance.std():.2f}"
        )

        if distance.mean() > threshold and distance.mean() / distance.std() > 1:
            # probably y does not depend on x, that means x = constant
            print("y=a*x+b: y may not depend on x")
            constant = indices[0, :].mean()
            params = (1, 0, 0, -constant)
            std_param = (0, 0, 0, indices[0, :].std())
            if debugging:
                _, ax = gu.scatter_plot(
                    np.transpose(indices),
                    labels=("axis 0", "axis 1", "axis 2"),
                    title="Points and fitted plane " + str(label),
                )
                ylim = ax.get_ylim()
                zlim = ax.get_zlim()
                meshy, meshz = np.meshgrid(
                    np.arange(ylim[0], ylim[1] + 1, 1),
                    np.arange(zlim[0], zlim[1] + 1, 1),
                )
                meshx = np.ones(meshy.shape) * constant
                ax.plot_wireframe(meshx, meshy, meshz, color="k")
                plt.pause(0.1)
            # calculate the mean distance to the fitted plane
            distance = plane_dist(indices=indices, params=params)
            print(
                f"plane fit using x=constant: dist.mean()={distance.mean():.2f},"
                f"  dist.std()={distance.std():.2f}"
            )

    if distance.mean() > threshold and distance.mean() / distance.std() > 1:
        # probably the distribution of points is not flat
        print("distance.mean() > 1, probably the distribution of points is not flat")
        valid_plane = False
    return params, std_param, valid_plane


def pseudovoigt(x_axis, amp, cen, sig, ratio):
    """
    Pseudo Voigt line shape.

    :param x_axis: where to calculate the function
    :param amp: amplitude of the Pseudo Voigt
    :param cen: position of the center of the Pseudo Voigt
    :param sig: FWHM of the Pseudo Voigt
    :param ratio: ratio of the Gaussian line shape
    :return: the Pseudo Voigt line shape at x_axis
    """
    sigma_gaussian = sig / (2 * np.sqrt(2 * np.log(2)))
    scaling_gaussian = 1 / (
        sigma_gaussian * np.sqrt(2 * np.pi)
    )  # the Gaussian is normalized
    sigma_lorentzian = sig / 2
    scaling_lorentzian = 1  # the Lorentzian is normalized
    return amp * (
        ratio * gaussian(x_axis, scaling_gaussian, cen, scaling_gaussian)
        + (1 - ratio) * lorentzian(x_axis, scaling_lorentzian, cen, sigma_lorentzian)
    )


def ref_count(address):
    """
    Get the reference count using ctypes module.

    :param address: integer, the memory adress id
    :return: the number of references to the memory address
    """
    return ctypes.c_long.from_address(address).value


def remove_background(array, q_values, avg_background, avg_qvalues, method="normalize"):
    """
    Subtract the average 1D background to the 3D array using q values.

    :param array: the 3D array. It should be sparse for faster calculation.
    :param q_values: tuple of three 1D arrays (qx, qz, qy), q values for the 3D dataset
    :param avg_background: average background data
    :param avg_qvalues: q values for the 1D average background data
    :param method: 'subtract' or 'normalize'
    :return: the 3D background array
    """
    if array.ndim != 3:
        raise ValueError("data should be a 3D array")
    if (avg_background.ndim != 1) or (avg_qvalues.ndim != 1):
        raise ValueError("avg_background and distances should be 1D arrays")

    qx, qz, qy = q_values

    ind_z, ind_y, ind_x = np.nonzero(
        array
    )  # if data is sparse, a loop over these indices only will be fast

    if method == "subtract":
        avg_background[np.isnan(avg_background)] = 0
        interpolation = interp1d(
            avg_qvalues,
            avg_background,
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        for index in range(len(ind_z)):
            array[ind_z[index], ind_y[index], ind_x[index]] = array[
                ind_z[index], ind_y[index], ind_x[index]
            ] - interpolation(
                np.sqrt(
                    qx[ind_z[index]] ** 2
                    + qz[ind_y[index]] ** 2
                    + qy[ind_x[index]] ** 2
                )
            )
    elif method == "normalize":
        avg_background[np.isnan(avg_background)] = 1
        avg_background[avg_background < 1] = 1
        interpolation = interp1d(
            avg_qvalues,
            avg_background,
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        for index in range(len(ind_z)):
            array[ind_z[index], ind_y[index], ind_x[index]] = array[
                ind_z[index], ind_y[index], ind_x[index]
            ] / interpolation(
                np.sqrt(
                    qx[ind_z[index]] ** 2
                    + qz[ind_y[index]] ** 2
                    + qy[ind_x[index]] ** 2
                )
            )

    array[np.isnan(array)] = 0
    array[array < 0] = 0

    return array


def remove_nan(data, mask=None):
    """
    Remove nan values from data.

    Optionally, it can update a mask (masked data = 1 in the mask, 0 otherwise).

    :param data: numpy ndarray
    :param mask: if provided, numpy ndarray of the same shape as the data
    :return: the filtered data and (optionally) mask
    """
    if mask is not None:
        if mask.shape != data.shape:
            raise ValueError("data and mask should have the same shape")

        # check for Nan
        mask[np.isnan(data)] = 1
        mask[np.isnan(mask)] = 1
        # check for Inf
        mask[np.isinf(data)] = 1
        mask[np.isinf(mask)] = 1
        mask[np.nonzero(mask)] = 1

    data[np.isnan(data)] = 0
    data[np.isinf(data)] = 0
    return data, mask


def rgb2gray(rgb):
    """
    Convert a three layered RGB image in gray.

    :param rgb: the image in RGB
    :return: the image conveted to gray
    """
    if rgb.ndim != 3:
        raise ValueError("the input array should be 3d")
    return 0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]


def rotation_matrix_3d(axis_to_align, reference_axis):
    """
    Calculate the rotation matrix which aligns axis_to_align onto reference_axis in 3D.

    :param axis_to_align: the 3D vector to be aligned (e.g. vector q),
     expressed in an orthonormal frame x y z
    :param reference_axis: will align axis_to_align onto this 3D vector,
     expressed in an orthonormal frame  x y z
    :return: the rotation matrix as a np.array of shape (3, 3)
    """
    # check parameters
    valid.valid_container(
        axis_to_align,
        container_types=(list, tuple, np.ndarray),
        length=3,
        item_types=Real,
        name="axis_to_align",
    )
    valid.valid_container(
        reference_axis,
        container_types=(list, tuple, np.ndarray),
        length=3,
        item_types=Real,
        name="reference_axis",
    )

    # normalize the vectors
    axis_to_align = axis_to_align / np.linalg.norm(axis_to_align)
    reference_axis = reference_axis / np.linalg.norm(reference_axis)

    # calculate the skew-symmetric matrix
    v = np.cross(axis_to_align, reference_axis)
    skew_sym_matrix = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = (
        np.identity(3)
        + skew_sym_matrix
        + np.dot(skew_sym_matrix, skew_sym_matrix)
        / (1 + np.dot(axis_to_align, reference_axis))
    )

    return rotation_matrix.transpose()


def rotate_crystal(
    arrays,
    axis_to_align=None,
    reference_axis=None,
    voxel_size=None,
    fill_value=0,
    rotation_matrix=None,
    is_orthogonal=False,
    reciprocal_space=False,
    debugging=False,
    **kwargs,
):
    """
    Rotate arrays to align axis_to_align onto reference_axis.

    The pivot of the rotation is in the center of the arrays. axis_to_align and
    reference_axis should be in the order X Y Z, where Z is downstream, Y vertical
    and X outboard (CXI convention).

    :param arrays: tuple of 3D real arrays of the same shape.
    :param axis_to_align: the axis to be aligned (e.g. vector q),
     expressed in an orthonormal frame x y z
    :param reference_axis: will align axis_to_align onto this vector,
     expressed in an orthonormal frame  x y z
    :param voxel_size: tuple, voxel size of the 3D array in z, y, and x (CXI convention)
    :param fill_value: tuple of numeric values used in the RegularGridInterpolator
     for points outside of the interpolation domain. The length of the tuple should
     be equal to the number of input arrays.
    :param rotation_matrix: optional numpy ndarray of shape (3, 3),
     rotation matrix to apply to arrays. If it is provided, the parameters
     axis_to_align and reference_axis will be discarded.
    :param is_orthogonal: set to True is the frame is orthogonal, False otherwise
     (detector frame) Used for plot labels.
    :param reciprocal_space: True if the data is in reciprocal space, False otherwise.
     Used for plot labels.
    :param debugging: tuple of booleans of the same length as the number of
     input arrays, True to see plots before and after rotation
    :param kwargs:
     - 'title': tuple of strings, titles for the debugging plots, same length as the
       number of arrays
     - 'scale': tuple of strings (either 'linear' or 'log'), scale for the debugging
       plots, same length as the number of arrays
     - width_z: size of the area to plot in z (axis 0), centered on the middle of
       the initial array
     - width_y: size of the area to plot in y (axis 1), centered on the middle of
       the initial array
     - width_x: size of the area to plot in x (axis 2), centered on the middle of
       the initial array

    :return: a rotated array (if a single array was provided) or a tuple of rotated
     arrays (same length as the number of input arrays)
    """
    # check that arrays is a tuple of 3D arrays
    if isinstance(arrays, np.ndarray):
        arrays = (arrays,)
    valid.valid_container(
        arrays,
        container_types=(tuple, list),
        item_types=np.ndarray,
        min_length=1,
        name="arrays",
    )
    if any(array.ndim != 3 for array in arrays):
        raise ValueError("all arrays should be 3D ndarrays of the same shape")
    ref_shape = arrays[0].shape
    if any(array.shape != ref_shape for array in arrays):
        raise ValueError("all arrays should be 3D ndarrays of the same shape")
    nb_arrays = len(arrays)
    nbz, nby, nbx = ref_shape

    # check some parameters
    voxel_size = voxel_size or (1, 1, 1)
    if isinstance(voxel_size, Real):
        voxel_size = (voxel_size,) * 3
    valid.valid_container(
        voxel_size,
        container_types=(tuple, list),
        length=3,
        item_types=Real,
        name="postprocessing_utils.rotate_crystal",
        min_excluded=0,
    )
    if isinstance(fill_value, Real):
        fill_value = (fill_value,) * nb_arrays
    valid.valid_container(
        fill_value,
        container_types=(tuple, list, np.ndarray),
        length=nb_arrays,
        item_types=Real,
        name="fill_value",
    )
    if isinstance(debugging, bool):
        debugging = (debugging,) * nb_arrays
    valid.valid_container(
        debugging,
        container_types=(tuple, list),
        length=nb_arrays,
        item_types=bool,
        name="debugging",
    )
    if rotation_matrix is None:
        # 'axis_to_align' and 'reference_axis' need to be declared
        # in order to calculate the rotation matrix
        valid.valid_container(
            axis_to_align,
            container_types=(tuple, list, np.ndarray),
            length=3,
            item_types=Real,
            name="axis_to_align",
        )
        valid.valid_container(
            reference_axis,
            container_types=(tuple, list, np.ndarray),
            length=3,
            item_types=Real,
            name="reference_axis",
        )
    else:
        print(
            'The rotation matrix is provided, parameters "axis_to_align" and '
            '"reference_axis" will be discarded'
        )
        if not isinstance(rotation_matrix, np.ndarray):
            raise TypeError(
                "rotation_matrix should be a numpy ndarray,"
                f" got {type(rotation_matrix)}"
            )
        if rotation_matrix.shape != (3, 3):
            raise ValueError(
                "rotation_matrix should be of shape (3, 3),"
                f" got {rotation_matrix.shape}"
            )

    # check and load kwargs
    valid.valid_kwargs(
        kwargs=kwargs,
        allowed_kwargs={"title", "scale", "width_z", "width_y", "width_x"},
        name="Setup.orthogonalize",
    )
    title = kwargs.get("title", ("Object",) * nb_arrays)
    valid.valid_container(
        title,
        container_types=(tuple, list),
        length=nb_arrays,
        item_types=str,
        name="title",
    )
    scale = kwargs.get("scale", ("linear",) * nb_arrays)
    valid.valid_container(
        scale, container_types=(tuple, list), length=nb_arrays, name="scale"
    )
    if any(val not in {"log", "linear"} for val in scale):
        raise ValueError("scale should be either 'log' or 'linear'")

    width_z = kwargs.get("width_z")
    valid.valid_item(
        value=width_z,
        allowed_types=int,
        min_excluded=0,
        allow_none=True,
        name="width_z",
    )
    width_y = kwargs.get("width_y")
    valid.valid_item(
        value=width_y,
        allowed_types=int,
        min_excluded=0,
        allow_none=True,
        name="width_y",
    )
    width_x = kwargs.get("width_x")
    valid.valid_item(
        value=width_x,
        allowed_types=int,
        min_excluded=0,
        allow_none=True,
        name="width_x",
    )

    ################################################################################
    # calculate the rotation matrix which aligns axis_to_align onto reference_axis #
    ################################################################################
    if rotation_matrix is None:
        rotation_matrix = rotation_matrix_3d(axis_to_align, reference_axis)

    ##################################################
    # calculate the new indices after transformation #
    ##################################################
    old_z = np.arange(-nbz // 2, nbz // 2, 1) * voxel_size[0]
    old_y = np.arange(-nby // 2, nby // 2, 1) * voxel_size[1]
    old_x = np.arange(-nbx // 2, nbx // 2, 1) * voxel_size[2]

    myz, myy, myx = np.meshgrid(old_z, old_y, old_x, indexing="ij")

    new_x = (
        rotation_matrix[0, 0] * myx
        + rotation_matrix[0, 1] * myy
        + rotation_matrix[0, 2] * myz
    )
    new_y = (
        rotation_matrix[1, 0] * myx
        + rotation_matrix[1, 1] * myy
        + rotation_matrix[1, 2] * myz
    )
    new_z = (
        rotation_matrix[2, 0] * myx
        + rotation_matrix[2, 1] * myy
        + rotation_matrix[2, 2] * myz
    )
    del myx, myy, myz
    gc.collect()

    ######################
    # interpolate arrays #
    ######################
    output_arrays = []
    for idx, array in enumerate(arrays):
        # convert array to float, for integers the interpolation can lead to artefacts
        array = array.astype(float)

        # interpolate array onto the new positions
        rgi = RegularGridInterpolator(
            (old_z, old_y, old_x),
            array,
            method="linear",
            bounds_error=False,
            fill_value=fill_value[idx],
        )
        rotated_array = rgi(
            np.concatenate(
                (
                    new_z.reshape((1, new_z.size)),
                    new_y.reshape((1, new_z.size)),
                    new_x.reshape((1, new_z.size)),
                )
            ).transpose()
        )
        rotated_array = rotated_array.reshape((nbz, nby, nbx)).astype(array.dtype)
        output_arrays.append(rotated_array)

        if debugging[idx]:
            gu.multislices_plot(
                array,
                width_z=width_z,
                width_y=width_y,
                width_x=width_x,
                title=title[idx] + " before rotating",
                is_orthogonal=is_orthogonal,
                scale=scale[idx],
                reciprocal_space=reciprocal_space,
            )
            gu.multislices_plot(
                rotated_array,
                width_z=width_z,
                width_y=width_y,
                width_x=width_x,
                title=title[idx] + " after rotating",
                is_orthogonal=is_orthogonal,
                scale=scale[idx],
                reciprocal_space=reciprocal_space,
            )

    if nb_arrays == 1:
        output_arrays = output_arrays[0]  # return the array instead of the tuple
    return output_arrays


def rotate_vector(
    vectors, axis_to_align=None, reference_axis=None, rotation_matrix=None
):
    """
    Rotate vectors.

    It calculates the vector components (3D) in the basis where axis_to_align and
    reference_axis are aligned. axis_to_align and reference_axis should be in the
    order X Y Z, where Z is downstream, Y vertical and X outboard (CXI convention).

    :param vectors: the vectors to be rotated, tuple of three components
     (values or 1D-arrays) expressed in an orthonormal frame x y z
    :param axis_to_align: the axis of myobj (vector q), expressed in an orthonormal
     frame x y z
    :param reference_axis: will align axis_to_align onto this vector, expressed in an
     orthonormal frame x y z
    :param rotation_matrix: optional numpy ndarray of shape (3, 3), rotation matrix to
     apply to vectors. If it is provided, the parameters axis_to_align and
     reference_axis will be discarded.
    :return: tuple of three ndarrays in CXI convention z y x, each of shape
     (vectors[0].size, vectors[1].size, vectors[2].size). If a single vector is
     provided, returns a 1D array of size 3.
    """
    # check parameters
    if isinstance(vectors, np.ndarray):
        if vectors.ndim == 1:  # a single vecotr was provided
            vectors = tuple(vectors)
        else:
            raise ValueError("vectors should be a tuple of three values/arrays")
    valid.valid_container(
        vectors,
        container_types=(tuple, list),
        length=3,
        item_types=(np.ndarray, Real),
        name="vectors",
    )
    if rotation_matrix is None:
        # 'axis_to_align' and 'reference_axis' need to be declared in order to
        # calculate the rotation matrix
        valid.valid_container(
            axis_to_align,
            container_types=(tuple, list, np.ndarray),
            length=3,
            item_types=Real,
            name="axis_to_align",
        )
        valid.valid_container(
            reference_axis,
            container_types=(tuple, list, np.ndarray),
            length=3,
            item_types=Real,
            name="reference_axis",
        )
    else:
        print(
            "The rotation matrix is provided, "
            'parameters "axis_to_align" and "reference_axis" will be discarded'
        )
        if not isinstance(rotation_matrix, np.ndarray):
            raise TypeError(
                "rotation_matrix should be a numpy ndarray,"
                f" got {type(rotation_matrix)}"
            )
        if rotation_matrix.shape != (3, 3):
            raise ValueError(
                "rotation_matrix should be of shape (3, 3),"
                f" got {rotation_matrix.shape}"
            )

    ################################################################################
    # calculate the rotation matrix which aligns axis_to_align onto reference_axis #
    ################################################################################
    if rotation_matrix is None:
        rotation_matrix = rotation_matrix_3d(axis_to_align, reference_axis)

    # calculate the new vector components after transformation
    myz, myy, myx = np.meshgrid(vectors[2], vectors[1], vectors[0], indexing="ij")
    new_x = (
        rotation_matrix[0, 0] * myx
        + rotation_matrix[0, 1] * myy
        + rotation_matrix[0, 2] * myz
    )
    new_y = (
        rotation_matrix[1, 0] * myx
        + rotation_matrix[1, 1] * myy
        + rotation_matrix[1, 2] * myz
    )
    new_z = (
        rotation_matrix[2, 0] * myx
        + rotation_matrix[2, 1] * myy
        + rotation_matrix[2, 2] * myz
    )

    if (
        new_x.size == 1
    ):  # a single vector was given as input, return it in a friendly format
        return np.array([new_z[0, 0, 0], new_y[0, 0, 0], new_x[0, 0, 0]])

    return new_z, new_y, new_x


def skewed_gaussian(x_axis, amp, loc, sig, alpha):
    """
    Skewed Gaussian line shape.

    :param x_axis: where to calculate the function
    :param amp: the amplitude of the Gaussian
    :param loc: the location parameter
    :param sig: HWHM of the Gaussian
    :param alpha: the shape parameter
    :return: the skewed Gaussian line shape at x_axis
    """
    return (
        amp
        * np.exp(-((x_axis - loc) ** 2) / (2.0 * sig ** 2))
        * (1 + erf(alpha / np.sqrt(2) * (x_axis - loc) / sig))
    )


def sum_roi(array, roi, debugging=False):
    """
    Sum the array intensities in the defined region of interest.

    :param array: 2D or 3D array. If ndim=3, the region of interest is applied
     sequentially to each 2D frame, the iteration being peformed over the first axis.
    :param roi: [Vstart, Vstop, Hstart, Hstop] region of interest for the sum
    :param debugging: True to see plots
    :return: a number (if array.ndim=2) or a 1D array of length array.shape[0]
     (if array.ndim=3) of summed intensities
    """
    ndim = array.ndim
    if ndim == 2:
        nby, nbx = array.shape
    elif ndim == 3:
        _, nby, nbx = array.shape
    else:
        raise ValueError("array should be 2D or 3D")

    if not 0 <= roi[0] < roi[1] <= nby:
        raise ValueError("0 <= roi[0] < roi[1] <= nby   expected")
    if not 0 <= roi[2] < roi[3] <= nbx:
        raise ValueError("0 <= roi[2] < roi[3] <= nbx   expected")

    if ndim == 2:
        sum_array = array[roi[0] : roi[1], roi[2] : roi[3]].sum()
    else:  # ndim = 3
        sum_array = np.zeros(array.shape[0])
        for idx in range(array.shape[0]):
            sum_array[idx] = array[idx, roi[0] : roi[1], roi[2] : roi[3]].sum()
        array = array.sum(axis=0)

    if debugging:
        val = array.max()
        array[roi[0] : roi[1], roi[2] : roi[2] + 3] = val
        array[roi[0] : roi[1], roi[3] - 3 : roi[3]] = val
        array[roi[0] : roi[0] + 3, roi[2] : roi[3]] = val
        array[roi[1] - 3 : roi[1], roi[2] : roi[3]] = val
        gu.combined_plots(
            tuple_array=(array, sum_array),
            tuple_sum_frames=False,
            tuple_sum_axis=0,
            tuple_scale="log",
            tuple_title=("summed array", "ROI integrated intensity"),
            tuple_colorbar=True,
        )
    return sum_array
