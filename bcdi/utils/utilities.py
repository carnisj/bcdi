# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr
"""Functions related to data loading, encoding, fitting, data manipulation."""

import ctypes
import gc
import logging
import os
import shutil
from collections import OrderedDict
from functools import reduce
from logging import Logger
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy.stats import multivariate_normal

from bcdi.graph import graph_utils as gu
from bcdi.utils import validation as valid

module_logger = logging.getLogger(__name__)


def apply_logical_array(
    arrays: Union[Real, np.ndarray, Tuple[Union[Real, np.ndarray], ...]],
    frames_logical: Optional[np.ndarray],
) -> Union[Real, np.ndarray, Tuple[Union[Real, np.ndarray], ...]]:
    """
    Apply a logical array to a sequence of arrays.

    Assuming a 1D array, it will be cropped where frames_logical is 0.

    :param arrays: a list or tuple of numbers or 1D arrays
    :param frames_logical: array of length the number of measured frames.
     In case of cropping/padding the number of frames changes. A frame whose
     index is set to 1 means that it is used, 0 means not used, -1 means padded
     (added) frame
    :return: an array (if a single array was provided) or a tuple of cropped arrays
    """
    if not isinstance(arrays, (tuple, list)):
        arrays = (arrays,)
    if frames_logical is None:
        return arrays
    valid.valid_1d_array(
        frames_logical,
        allowed_types=Integral,
        allow_none=False,
        allowed_values=(-1, 0, 1),
        name="frames_logical",
    )

    # number of measured frames during the experiment
    # frames_logical[idx]=-1 means that a frame was added (padding) at index idx
    original_frames = frames_logical[frames_logical != -1]
    nb_original = len(original_frames)

    output = []
    for array in arrays:
        if isinstance(array, Real):
            output.append(array)
        else:
            valid.valid_ndarray(array, ndim=1, shape=(nb_original,))
            # padding occurs only at the edges of the dataset, so the original data is
            # contiguous, we can use array indexing directly
            output.append(array[original_frames != 0])

    if len(arrays) == 1:
        return np.asarray(output[0])  # return the array instead of the tuple
    return tuple(output)


def bin_data(array, binning, debugging=False, **kwargs):
    """
    Rebin a 1D, 2D or 3D array.

    If its dimensions are not a multiple of binning, the array will be cropped.
    Adapted from PyNX.

    :param array: the array to resize
    :param binning: the rebin factor - pixels will be summed by groups of
     binning (x binning (x binning)). This can also be a tuple/list of rebin values
     along each axis, e.g. binning=(4,1,2) for a 3D array
    :param debugging: boolean, True to see plots
    :param kwargs:

     - 'cmap': str, name of the colormap
     - 'logger': an optional logger

    :return: the binned array
    """
    cmap = kwargs.get("cmap", "turbo")
    logger = kwargs.get("logger", module_logger)
    valid.valid_ndarray(arrays=array, ndim=(1, 2, 3))
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
        logger.info(f"array shape after cropping but before binning: {array.shape}")
        logger.info(f"array shape after binning: {newarray.shape}")
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
            cmap=cmap,
        )
    return newarray


def bin_parameters(
    binning: int, nb_frames: int, params: List[Any], debugging: bool = True
) -> List[Any]:
    """
    Bin some parameters.

    It selects parameter values taking into account an eventual binning of the data.
    The use case is to bin diffractometer motor positions for a dataset binned along
    the rocking curve axis.

    :param binning: binning factor for the axis corresponding to the rocking curve
    :param nb_frames: number of frames of the rocking curve dimension
    :param params: list of parameters
    :param debugging: set to True to have printed parameters
    :return: list of parameters (same list length), taking into account binning
    """
    if binning == 1:  # nothing to do
        return params

    if debugging:
        print(params)

    nb_param = len(params)
    print(
        nb_param,
        "motor parameters modified to take into account "
        "binning of the rocking curve axis",
    )

    if (binning % 1) != 0:
        raise ValueError("Invalid binning value")
    for idx, param in enumerate(params):
        try:  # check if param has a length
            if len(params[idx]) != nb_frames:
                raise ValueError(
                    f"parameter {idx}: length {len(params[idx])} "
                    f"different from nb_frames {nb_frames}"
                )
        except TypeError:  # int or float
            params[idx] = np.repeat(param, nb_frames)
        temp = params[idx]
        params[idx] = temp[::binning]

    if debugging:
        print(params)

    return params


def cast(
    val: Union[float, List, np.ndarray], target_type: type = float, **kwargs
) -> Union[float, List, np.ndarray]:
    """
    Cast val to a number or an array of numbers of the target type.

    :param val: the value to be converted
    :param target_type: the type to convert to
    :param kwargs:
     - 'logger': an optional logger

    """
    logger = kwargs.get("logger", module_logger)
    if not isinstance(target_type, type):
        raise TypeError("target_type should be a type")
    if target_type not in [int, float]:
        raise ValueError(f"target_type should be 'int' or 'float', got {target_type}")
    try:
        if isinstance(val, np.ndarray):
            val = val.astype(target_type)
        elif isinstance(val, (list, tuple)):
            val = [cast(value, target_type=target_type) for value in val]
        else:
            val = target_type(val)
        return val
    except (TypeError, ValueError):
        logger.info(f"Cannot cast {val} to {target_type}")
        raise


def catch_error(exception):
    """
    Process exception in asynchronous multiprocessing.

    :param exception: the arisen exception
    """
    print(exception)


def convert_str_target(
    value: Any, target: str, conversion_table: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Convert strings from value to the desired target.

    :param value: an object containing strings to be converted
    :param target: the target string, which has to be present in the conversion table
    :param conversion_table: a dictionary for the conversion
    :return: the converted object
    """
    conversion_table = (
        conversion_table
        if conversion_table is not None
        else {"none": None, "true": True, "false": False}
    )
    target = target.lower()
    if target not in conversion_table:
        raise ValueError(
            f"invalid target {target}, valid targets: {list(conversion_table.keys())}"
        )
    if isinstance(value, str) and value.lower() == target:
        return conversion_table[target]
    if isinstance(value, (list, tuple)):
        new_value = list(value)
        for idx, val in enumerate(new_value):
            new_value[idx] = convert_str_target(val, target=target)
        return new_value
    if isinstance(value, dict):
        for key, item in value.items():
            value[key] = convert_str_target(item, target=target)
        return value

    return value


def crop_pad(
    array,
    output_shape,
    pad_value=0,
    pad_start=None,
    crop_center=None,
    debugging=False,
    **kwargs,
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
    :param kwargs:

     - 'cmap': str, name of the colormap
     - 'logger': an optional logger

    :return: myobj cropped or padded with zeros
    """
    logger = kwargs.get("logger", module_logger)
    valid.valid_ndarray(arrays=array, ndim=3)
    valid.valid_container(
        output_shape,
        container_types=(list, tuple, np.ndarray),
        item_types=int,
        length=3,
        min_excluded=0,
    )
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
        logger.info(f"array shape before crop/pad = {array.shape}")
        gu.multislices_plot(
            abs(array),
            sum_frames=True,
            scale="log",
            title="Before crop/pad",
            cmap=kwargs.get("cmap", "turbo"),
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
        logger.info(f"array shape after crop/pad = {newobj.shape}")
        gu.multislices_plot(
            abs(newobj),
            sum_frames=True,
            scale="log",
            title="After crop/pad",
            cmap=kwargs.get("cmap", "turbo"),
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
    valid.valid_ndarray(arrays=array, ndim=2)
    valid.valid_container(
        output_shape,
        container_types=(list, tuple, np.ndarray),
        item_types=int,
        length=2,
        min_excluded=0,
    )

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
    valid.valid_ndarray(arrays=array, ndim=1)
    valid.valid_item(output_length, allowed_types=int, min_excluded=0)

    nbx = array.shape[0]
    if pad_start is None:
        pad_start = (output_length - nbx) // 2

    if crop_center is None:
        crop_center = nbx // 2

    if output_length >= nbx:  # pad
        if not extrapolate:
            newobj = np.ones(output_length, dtype=array.dtype) * pad_value
            newobj[pad_start : pad_start + nbx] = array
        else:
            spacing = array[1] - array[0]
            pad_start = array[0] - ((output_length - nbx) // 2) * spacing
            newobj = pad_start + np.arange(output_length) * spacing
    else:  # crop
        if (crop_center - output_length // 2 < 0) or (
            crop_center + output_length // 2 > nbx
        ):
            raise ValueError("crop_center incompatible with output_length")
        newobj = array[
            crop_center
            - output_length // 2 : crop_center
            + output_length // 2
            + output_length % 2
        ]
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


def dos2unix(input_file: str, savedir: str) -> None:
    """
    Convert DOS linefeeds (crlf) to UNIX (lf).

    :param input_file: the original filename (absolute path)
    :param savedir: path of the directory where to save
    """
    filename = os.path.splitext(os.path.basename(input_file))[0]
    if not input_file.endswith(".txt"):
        raise ValueError("Expecting a text file as input, e.g. 'alias_dict.txt'")
    with open(input_file, "rb") as infile:
        content = infile.read()
    with open(os.path.join(savedir, f"{filename}_unix.txt"), "wb") as output:
        for row in content.splitlines():
            output.write(row + str.encode("\n"))


def find_crop_center(array_shape, crop_shape, pivot):
    """
    Find the position of the center of the cropping window.

    It finds the closest voxel to pivot which allows to crop an array of array_shape to
    crop_shape.

    :param array_shape: initial shape of the array
    :type array_shape: tuple
    :param crop_shape: final shape of the array
    :type crop_shape: tuple
    :param pivot: position on which the final region of interest dhould be centered
     (center of mass of the Bragg peak)
    :type pivot: tuple
    :return: the voxel position closest to pivot which allows cropping to the defined
     shape.
    """
    valid.valid_container(
        array_shape,
        container_types=(tuple, list, np.ndarray),
        min_length=1,
        item_types=int,
        name="array_shape",
    )
    ndim = len(array_shape)
    valid.valid_container(
        crop_shape,
        container_types=(tuple, list, np.ndarray),
        length=ndim,
        item_types=int,
        name="crop_shape",
    )
    valid.valid_container(
        pivot,
        container_types=(tuple, list, np.ndarray),
        length=ndim,
        item_types=int,
        name="pivot",
    )
    crop_center = np.empty(ndim)
    for idx, _ in enumerate(range(ndim)):
        if max(0, pivot[idx] - crop_shape[idx] // 2) == 0:
            # not enough range on this side of the com
            crop_center[idx] = crop_shape[idx] // 2
        else:
            if (
                min(array_shape[idx], pivot[idx] + crop_shape[idx] // 2)
                == array_shape[idx]
            ):
                # not enough range on this side of the com
                crop_center[idx] = array_shape[idx] - crop_shape[idx] // 2
            else:
                crop_center[idx] = pivot[idx]

    crop_center = list(map(int, crop_center))
    return crop_center


def find_file(filename: Optional[str], default_folder: str, **kwargs) -> str:
    """
    Locate a file.

    The filename can be either the name of the file (including the extension) or the
    full path to the file.

    :param filename: the name or full path to the file
    :param default_folder: it will look for the file in that folder if filename is not
     the full path.
    :param kwargs:

     - 'logger': an optional logger

    :return: str, the path to the file
    """
    logger = kwargs.get("logger", module_logger)
    if not isinstance(filename, str):
        raise TypeError("filename should be a string")

    if os.path.isfile(filename):
        # filename is already the full path to the file
        return filename
    logger.info(f"Could not find the file at: {filename}")

    if not isinstance(default_folder, str):
        raise TypeError("default_folder should be a string")
    if not default_folder.endswith("/"):
        default_folder += "/"

    if not os.path.isdir(default_folder):
        raise ValueError(f"The directory {default_folder} does not exist")
    full_name = default_folder + filename
    if not os.path.isfile(full_name):
        raise ValueError(f"Could not localize the file at {filename} or {full_name}")
    logger.info(f"File localized at: {full_name}")
    return full_name


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
    valid.valid_ndarray(
        arrays=(reference_array, original_array), ndim=1, fix_shape=False
    )

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
        amp = params[f"amp_{iterator}"].value
        cen = params[f"cen_{iterator}"].value
        sig = params[f"sig_{iterator}"].value
        return gaussian(x_axis=x_axis, amp=amp, cen=cen, sig=sig)
    if distribution == "skewed_gaussian":
        amp = params[f"amp_{iterator}"].value
        loc = params[f"loc_{iterator}"].value
        sig = params[f"sig_{iterator}"].value
        alpha = params[f"alpha_{iterator}"].value
        return skewed_gaussian(x_axis=x_axis, amp=amp, loc=loc, sig=sig, alpha=alpha)
    if distribution == "lorentzian":
        amp = params[f"amp_{iterator}"].value
        cen = params[f"cen_{iterator}"].value
        sig = params[f"sig_{iterator}"].value
        return lorentzian(x_axis=x_axis, amp=amp, cen=cen, sig=sig)
    if distribution == "pseudovoigt":
        amp = params[f"amp_{iterator}"].value
        cen = params[f"cen_{iterator}"].value
        sig = params[f"sig_{iterator}"].value
        ratio = params[f"ratio_{iterator}"].value
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
    return amp * np.exp(-((x_axis - cen) ** 2) / (2.0 * sig**2))


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
        covariance = np.diag(sigma**2)
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
        covariance = np.diag(sigma**2)
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


def generate_frames_logical(
    nb_images: int, frames_pattern: Optional[List[int]]
) -> np.ndarray:
    """
    Generate a logical array allowing ti discrad frames in the dataset.

    :param nb_images: the number of 2D images in te dataset.
    :param frames_pattern: user-provided list which can be:
     - a binary list of length nb_images
     - a list of the indices of frames to be skipped

    :return: a binary numpy array of length nb_images
    """
    valid.valid_item(nb_images, allowed_types=int, min_excluded=0, name="nb_images")
    if frames_pattern is None:
        return np.ones(nb_images, dtype=int)

    valid.valid_container(
        frames_pattern,
        container_types=list,
        max_length=nb_images,
        item_types=int,
        min_included=0,
        max_excluded=nb_images,
        allow_none=False,
        name="frames_pattern",
    )

    if len(frames_pattern) == nb_images:
        if all(val in {0, 1} for val in frames_pattern):
            return np.array(frames_pattern, dtype=int)
        raise ValueError(f"A binary list of lenght {nb_images} is expected")

    if len(set(frames_pattern)) != len(frames_pattern):
        if all(val in {0, 1} for val in frames_pattern):
            raise ValueError(
                "frame_patterns is a binary list of length "
                f"{len(frames_pattern)}, but there are {nb_images} images"
            )
        raise ValueError("Duplicated indices in frame_patterns")

    frames_logical = np.ones(nb_images, dtype=int)
    frames_logical[frames_pattern] = 0
    return frames_logical


def higher_primes(number, maxprime=13, required_dividers=(4,)):
    """
    Find the closest larger number that meets some condition.

    Find the closest integer >=n (or list/array of integers), for which the largest
    prime divider is <=maxprime, and has to include some dividers. The default values
    for maxprime is the largest integer accepted by the clFFT library for OpenCL GPU
    FFT. Adapted from PyNX.

    :param number: the integer number
    :param maxprime: the largest prime factor acceptable
    :param required_dividers: a list of required dividers for the returned integer.
    :return: the integer (or list/array of integers) fulfilling the requirements
    """
    if isinstance(number, (list, tuple, np.ndarray)):
        vn = []
        for i in number:
            limit = i
            if i <= 1 or maxprime > i:
                raise ValueError(f"Number is < {maxprime}")
            while (
                try_smaller_primes(
                    i, maxprime=maxprime, required_dividers=required_dividers
                )
                is False
            ):
                i = i + 1
                if i == limit:
                    return limit
            vn.append(i)
        if isinstance(number, np.ndarray):
            return np.array(vn)
        return vn
    limit = number
    if number <= 1 or maxprime > number:
        raise ValueError(f"Number is < {maxprime}")
    while (
        try_smaller_primes(
            number, maxprime=maxprime, required_dividers=required_dividers
        )
        is False
    ):
        number = number + 1
        if number == limit:
            return limit
    return number


def image_to_ndarray(filename, convert_grey=True):
    """
    Convert an image to a numpy array using pillow.

    Matplotlib only supports the PNG format.

    :param filename: absolute path of the image to open
    :param convert_grey: if True and the number of layers is 3, it will be converted
     to a single layer of grey
    :return:
    """
    im = Image.open(filename)

    array = np.asarray(im)
    if array.ndim == 3 and convert_grey:
        print("converting image to gray")
        array = rgb2gray(array)
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


def linecut(array, point, direction, direction_basis="voxel", voxel_size=1):
    """
    Calculate iteratively a linecut through an array without interpolation.

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
    valid.valid_ndarray(array, ndim=(2, 3))
    ndim = array.ndim
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


def load_background(background_file):
    """
    Load a background file.

    :param background_file: the path of the background file
    :return: a 2D background
    """
    if background_file:
        background, _ = load_file(background_file)
        valid.valid_ndarray(background, ndim=2)
    else:
        background = None
    return background


def load_file(
    file_path: str, fieldname: Optional[str] = None
) -> Tuple[np.ndarray, str]:
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
        with np.load(file_path) as npzfile:
            if fieldname is None:  # output of PyNX phasing or flatfield/background
                dataset = npzfile[list(npzfile.files)[0]]
            else:  # could be anything
                try:
                    dataset = npzfile[fieldname]
                except KeyError:
                    dataset = npzfile[list(npzfile.files)[0]]
    elif extension == ".npy":  # could be anything
        # no need to close the file for .npy extension, see np.load docstring
        dataset = np.load(file_path)
    elif extension == ".cxi":  # output of PyNX phasing
        with h5py.File(file_path, "r") as h5file:
            # group_key = list(h5file.keys())[1]
            # subgroup_key = list(h5file[group_key])
            # dataset = h5file['/'+group_key+'/'+subgroup_key[0]+'/data'].value
            dataset = h5file["/entry_1/data_1/data"][()]
    elif extension == ".h5":  # modes.h5
        with h5py.File(file_path, "r") as h5file:
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


def load_flatfield(flatfield_file):
    """
    Load a flatfield file.

    :param flatfield_file: the path of the flatfield file
    :return: a 2D flatfield
    """
    if flatfield_file and os.path.isfile(flatfield_file):
        flatfield, _ = load_file(flatfield_file)
        valid.valid_ndarray(flatfield, ndim=2)
    else:
        flatfield = None
    return flatfield


def load_hotpixels(hotpixels_file):
    """
    Load a hotpixels file.

    :param hotpixels_file: the path of the hotpixels file
    :return: a 2D array of hotpixels (1 for hotpixel, 0 for normal pixel)
    """
    if hotpixels_file:
        hotpixels, _ = load_file(hotpixels_file)
        valid.valid_ndarray(hotpixels, ndim=(2, 3))
        if hotpixels.ndim == 3:
            hotpixels = hotpixels.sum(axis=0)
        if (hotpixels == 0).sum() < hotpixels.size / 4:
            # masked pixels are more than 3/4 of the pixel number
            print("hotpixels values are probably 0 instead of 1, switching values")
            hotpixels[np.nonzero(hotpixels)] = -1
            hotpixels[hotpixels == 0] = 1
            hotpixels[hotpixels == -1] = 0

        hotpixels[np.nonzero(hotpixels)] = 1
    else:
        hotpixels = None
    return hotpixels


def lorentzian(x_axis, amp, cen, sig):
    """
    Lorentzian line shape.

    :param x_axis: where to calculate the function
    :param amp: the amplitude of the Lorentzian
    :param cen: the position of the center
    :param sig: HWHM of the Lorentzian
    :return: the Lorentzian line shape at x_axis
    """
    return amp / (sig * np.pi) / (1 + (x_axis - cen) ** 2 / (sig**2))


def make_support(
    arrays: Union[np.ndarray, Sequence[np.ndarray]], support_threshold: float
) -> Union[np.ndarray, Sequence[np.ndarray]]:
    """
    Create a support for each provided array, using a threshold on its modulus.

    :param arrays: a sequence of numpy ndarrays
    :param support_threshold: a float in [0, 1], normalized threshold that will be
     applied to the modulus of each array
    :return: a tuple of numpy ndarrays, supports corresponding to each input array
    """
    # check some parameters
    if isinstance(arrays, np.ndarray):
        arrays = (arrays,)
    valid.valid_container(
        arrays, container_types=(tuple, list), item_types=np.ndarray, name="arrays"
    )
    valid.valid_item(
        support_threshold,
        allowed_types=float,
        min_included=0,
        max_included=1,
        name="support_threshold",
    )

    # create the supports
    supports = []
    for _, array in enumerate(arrays):
        support = np.zeros(array.shape)
        support[abs(array) > support_threshold * abs(array)] = 1
        supports.append(support)

    if len(arrays) == 1:  # return an array to avoid having to unpack it every time
        return supports[0]
    return supports


def mean_filter(
    data,
    nb_neighbours,
    mask=None,
    target_val=0,
    extent=1,
    min_count=3,
    interpolate="mask_isolated",
    debugging=False,
    **kwargs,
):
    """
    Mask or apply a mean filter to data.

    The procedure is applied only if the empty pixel is surrounded by nb_neighbours or
    more pixels with at least min_count intensity per pixel.

    :param data: 2D or 3D array to be filtered
    :param nb_neighbours: minimum number of non-zero neighboring pixels for median
     filtering
    :param mask: mask array of the same shape as data
    :param target_val: value where to interpolate, allowed values are int>=0 or np.nan
    :param extent: in pixels, extent of the averaging window from the reference pixel
     (extent=1, 2, 3 ... corresponds to window width=3, 5, 7 ... )
    :param min_count: minimum intensity (inclusive) in the neighboring pixels to
     interpolate, the pixel will be masked otherwise.
    :param interpolate: based on 'nb_neighbours', if 'mask_isolated' will mask
     isolated pixels, if 'interp_isolated' will interpolate isolated pixels
    :param debugging: set to True to see plots
    :type debugging: bool
    :param kwargs:

     - 'cmap': str, name of the colormap

    :return: updated data and mask, number of pixels treated
    """
    cmap = kwargs.get("cmap", "turbo")
    valid.valid_ndarray(arrays=data, ndim=(2, 3))
    # check some mparameters
    if mask is None:
        mask = np.zeros(data.shape)
    valid.valid_ndarray(arrays=mask, shape=data.shape)

    if not np.isnan(target_val) and not isinstance(target_val, int):
        raise ValueError(
            "target_val should be nan or an integer, cannot assess float equality"
        )

    valid.valid_item(
        nb_neighbours, allowed_types=int, min_excluded=0, name="mean_filter"
    )
    valid.valid_item(extent, allowed_types=int, min_excluded=0, name="mean_filter")
    valid.valid_item(min_count, allowed_types=int, min_included=0, name="mean_filter")

    if interpolate not in {"mask_isolated", "interp_isolated"}:
        raise ValueError(
            f"invalid value '{interpolate}' for interpolate,"
            f" allowed are 'mask_isolated' and 'interp_isolated'"
        )
    if not isinstance(debugging, bool):
        raise TypeError(f"debugging should be a boolean, got {type(debugging)}")

    # find all voxels to be processed
    if target_val is np.nan:
        target_pixels = np.argwhere(np.isnan(data))
    else:
        target_pixels = np.argwhere(data == target_val)
    nb_pixels = 0

    if debugging:
        gu.combined_plots(
            tuple_array=(data, mask),
            tuple_sum_frames=(False, False),
            tuple_sum_axis=(0, 0),
            tuple_width_v=(None, None),
            tuple_width_h=(None, None),
            tuple_colorbar=(True, True),
            tuple_vmin=(-1, 0),
            tuple_vmax=(np.nan, 1),
            tuple_scale=("log", "linear"),
            tuple_title=("Data before filtering", "Mask before filtering"),
            reciprocal_space=True,
            cmap=cmap,
        )

    if data.ndim == 2:
        for indx in range(target_pixels.shape[0]):
            pixrow = target_pixels[indx, 0]
            pixcol = target_pixels[indx, 1]
            temp = data[
                pixrow - extent : pixrow + extent + 1,
                pixcol - extent : pixcol + extent + 1,
            ]
            temp = temp[np.logical_and(~np.isnan(temp), temp != target_val)]
            if (
                temp.size >= nb_neighbours and (temp > min_count).all()
            ):  # nb_neighbours is >= 1
                nb_pixels += 1
                if interpolate == "interp_isolated":
                    data[pixrow, pixcol] = temp.mean()
                    mask[pixrow, pixcol] = 0
                else:
                    mask[pixrow, pixcol] = 1
    else:  # 3D
        for indx in range(target_pixels.shape[0]):
            pix_z = target_pixels[indx, 0]
            pix_y = target_pixels[indx, 1]
            pix_x = target_pixels[indx, 2]
            temp = data[
                pix_z - extent : pix_z + extent + 1,
                pix_y - extent : pix_y + extent + 1,
                pix_x - extent : pix_x + extent + 1,
            ]
            temp = temp[np.logical_and(~np.isnan(temp), temp != target_val)]
            if (
                temp.size >= nb_neighbours and (temp > min_count).all()
            ):  # nb_neighbours is >= 1
                nb_pixels += 1
                if interpolate == "interp_isolated":
                    data[pix_z, pix_y, pix_x] = temp.mean()
                    mask[pix_z, pix_y, pix_x] = 0
                else:
                    mask[pix_z, pix_y, pix_x] = 1

    if debugging:
        gu.combined_plots(
            tuple_array=(data, mask),
            tuple_sum_frames=(True, True),
            tuple_sum_axis=(0, 0),
            tuple_width_v=(None, None),
            tuple_width_h=(None, None),
            tuple_colorbar=(True, True),
            tuple_vmin=(-1, 0),
            tuple_vmax=(np.nan, 1),
            tuple_scale=("log", "linear"),
            tuple_title=("Data after filtering", "Mask after filtering"),
            reciprocal_space=True,
            cmap=cmap,
        )

    return data, nb_pixels, mask


def move_log(result: Tuple[Path, Path, Optional[Logger]]):
    """
    Move log files to the desired location, after processing a file.

    It can be used as a standard function or as a callback for multiprocessing.

    :param result: the output of process_scan, containing the 2d data, 2d mask,
     counter for each frame; the file index; and an optional logger
    """
    logger = result[2] if result[2] is not None else module_logger
    filename = result[0].name
    shutil.move(result[0], result[1] / filename)
    logger.info(f"{filename.replace('.log', '')} processed")


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
    ndim = data.ndim
    if ndim == 1 and not isinstance(data[0], np.ndarray):  # single dataset
        data = data[np.newaxis, :]
        x_axis = x_axis[np.newaxis, :]
    if ndim not in [1, 2]:
        raise ValueError(f"data should be 1D or 2D, got {ndim}D")

    ndata = data.shape[0]
    resid = 0.0 * data[:]
    # make residual per data set
    for idx in range(ndata):
        resid[idx] = data[idx] - function_lmfit(
            params=params,
            iterator=idx,
            x_axis=x_axis[idx],
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


def pad_from_roi(
    arrays: Union[np.ndarray, Tuple[np.ndarray, ...]],
    roi: List[int],
    binning: Tuple[int, int],
    pad_value: Union[float, Tuple[float, ...]] = 0.0,
    **kwargs,
) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    """
    Pad a 3D stack of frames provided a region of interest.

    The stacking is assumed to be on the first axis.

    :param arrays: a 3D array or a sequence of 3D arrays of the same shape
    :param roi: the desired region of interest of the unbinned frame. For an array in
     arrays, the shape is (nz, ny, nx), and roi corresponds to [y0, y1, x0, x1]
    :param binning: tuple of two integers (binning along Y, binning along X)
    :param pad_value: number or tuple of nb_arrays numbers, will pad using this value
    :param kwargs:
     - 'logger': an optional logger

    :return: an array (if a single array was provided) or a tuple of arrays interpolated
     on an orthogonal grid (same length as the number of input arrays)
    """
    logger: Logger = kwargs.get("logger", module_logger)
    ####################
    # check parameters #
    ####################
    valid.valid_ndarray(arrays, ndim=3)
    nb_arrays = len(arrays)
    valid.valid_container(
        roi,
        container_types=(tuple, list, np.ndarray),
        item_types=int,
        length=4,
        name="roi",
    )
    valid.valid_container(
        binning,
        container_types=(tuple, list, np.ndarray),
        item_types=int,
        length=2,
        name="binning",
    )
    if isinstance(pad_value, float):
        pad_value = (pad_value,) * nb_arrays
    valid.valid_container(
        pad_value,
        container_types=(tuple, list, np.ndarray),
        item_types=Real,
        length=nb_arrays,
        name="pad_value",
    )

    ##############################################
    # calculate the starting indices for padding #
    ##############################################
    nbz, nby, nbx = arrays[0].shape
    output_shape = (
        nbz,
        int(np.rint((roi[1] - roi[0]) / binning[0])),
        int(np.rint((roi[3] - roi[2]) / binning[1])),
    )

    if output_shape[1] > nby or output_shape[2] > nbx:
        if roi[0] < 0:  # padding on the left
            starty = abs(roi[0] // binning[0])
            # loaded data will start at this index
        else:  # padding on the right
            starty = 0
        if roi[2] < 0:  # padding on the left
            startx = abs(roi[2] // binning[1])
            # loaded data will start at this index
        else:  # padding on the right
            startx = 0
        start = [int(val) for val in [0, starty, startx]]
        logger.info("Paddind the data to the shape defined by the ROI")

        ##############
        # pad arrays #
        ##############
        output_arrays = []
        for idx, array in enumerate(arrays):
            array = crop_pad(
                array=array,
                pad_value=pad_value[idx],
                pad_start=start,
                output_shape=output_shape,
            )
            output_arrays.append(array)

        if nb_arrays == 1:
            return np.asarray(output_arrays[0])  # return the array instead of the tuple
        return tuple(output_arrays)
    return arrays


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


def primes(number):
    """
    Return the prime decomposition of n as a list. Adapted from PyNX.

    :param number: the integer to be decomposed
    :return: the list of prime dividers of number
    """
    valid.valid_item(number, allowed_types=int, min_excluded=0, name="number")
    list_primes = [1]
    i = 2
    while i * i <= number:
        while number % i == 0:
            list_primes.append(i)
            number //= i
        i += 1
    if number > 1:
        list_primes.append(number)
    return list_primes


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


def remove_avg_background(
    array, q_values, avg_background, avg_qvalues, method="normalize"
):
    """
    Subtract the average 1D background to the 3D array using q values.

    :param array: the 3D array. It should be sparse for faster calculation.
    :param q_values: tuple of three 1D arrays (qx, qz, qy), q values for the 3D dataset
    :param avg_background: average background data
    :param avg_qvalues: q values for the 1D average background data
    :param method: 'subtract' or 'normalize'
    :return: the 3D background array
    """
    valid.valid_ndarray(array, ndim=3)
    valid.valid_ndarray(arrays=(avg_background, avg_qvalues), ndim=1)

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
        for index, item in enumerate(ind_z):
            array[item, ind_y[index], ind_x[index]] = array[
                item, ind_y[index], ind_x[index]
            ] - interpolation(
                np.sqrt(qx[item] ** 2 + qz[ind_y[index]] ** 2 + qy[ind_x[index]] ** 2)
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
        for index, item in enumerate(ind_z):
            array[item, ind_y[index], ind_x[index]] = array[
                item, ind_y[index], ind_x[index]
            ] / interpolation(
                np.sqrt(qx[item] ** 2 + qz[ind_y[index]] ** 2 + qy[ind_x[index]] ** 2)
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
    valid.valid_ndarray(data)
    if mask is not None:
        valid.valid_ndarray(mask, shape=data.shape)

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

     - 'cmap': str, name of the colormap
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
    if isinstance(arrays, np.ndarray):
        arrays = (arrays,)
    valid.valid_ndarray(arrays, ndim=3)
    nb_arrays = len(arrays)
    nbz, nby, nbx = arrays[0].shape

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
        allowed_kwargs={"cmap", "title", "scale", "width_z", "width_y", "width_x"},
        name="Setup.orthogonalize",
    )
    cmap = kwargs.get("cmap", "turbo")
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
                cmap=cmap,
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
                cmap=cmap,
            )

    if nb_arrays == 1:
        output_arrays = output_arrays[0]  # return the array instead of the tuple
    return output_arrays


def rotate_vector(
    vectors: Union[np.ndarray, Tuple[np.ndarray, ...]],
    axis_to_align: Optional[np.ndarray] = None,
    reference_axis: Optional[np.ndarray] = None,
    rotation_matrix: Optional[np.ndarray] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
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
        if vectors.ndim == 1:  # a single vector was provided
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
        * np.exp(-((x_axis - loc) ** 2) / (2.0 * sig**2))
        * (1 + erf(alpha / np.sqrt(2) * (x_axis - loc) / sig))
    )


def smaller_primes(number, maxprime=13, required_dividers=(4,)):
    """
    Find the closest smaller number that meets some condition.

    Find the closest integer <=n (or list/array of integers), for which the largest
    prime divider is <=maxprime, and has to include some dividers. The default values
    for maxprime is the largest integer accepted by the clFFT library for OpenCL GPU
    FFT. Adapted from PyNX.

    :param number: the integer number
    :param maxprime: the largest prime factor acceptable
    :param required_dividers: a list of required dividers for the returned integer.
    :return: the integer (or list/array of integers) fulfilling the requirements
    """
    if isinstance(number, (list, tuple, np.ndarray)):
        vn = []
        for i in number:
            if i <= 1 or maxprime > i:
                raise ValueError(f"Number is < {maxprime}")
            while (
                try_smaller_primes(
                    i, maxprime=maxprime, required_dividers=required_dividers
                )
                is False
            ):
                i = i - 1
                if i == 0:
                    return 0
            vn.append(i)
        if isinstance(number, np.ndarray):
            return np.array(vn)
        return vn
    if number <= 1 or maxprime > number:
        raise ValueError(f"Number is < {maxprime}")
    while (
        try_smaller_primes(
            number, maxprime=maxprime, required_dividers=required_dividers
        )
        is False
    ):
        number = number - 1
        if number == 0:
            return 0
    return number


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
    valid.valid_ndarray(array, ndim=(2, 3))
    ndim = array.ndim
    if ndim == 2:
        nby, nbx = array.shape
    else:  # 3D
        _, nby, nbx = array.shape

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


def try_smaller_primes(number, maxprime=13, required_dividers=(4,)):
    """
    Check if a number meets some condition.

    Check if the largest prime divider is <=maxprime, and optionally includes some
    dividers. Adapted from PyNX.

    :param number: the integer number for which the prime decomposition will be checked
    :param maxprime: the maximum acceptable prime number. This defaults to the
     largest integer accepted by the clFFT library for OpenCL GPU FFT.
    :param required_dividers: list of required dividers in the prime decomposition.
     If None, this check is skipped.
    :return: True if the conditions are met.
    """
    p = primes(number)
    if max(p) > maxprime:
        return False
    if required_dividers is not None:
        for k in required_dividers:
            if number % k != 0:
                return False
    return True


def unpack_array(
    array: Union[float, List[float], np.ndarray]
) -> Union[float, np.ndarray]:
    """Unpack an array or Sequence of length 1 into a single element."""
    if isinstance(array, (list, tuple, np.ndarray)) and len(array) == 1:
        return array[0]
    return np.asarray(array)


def update_frames_logical(
    frames_logical: np.ndarray, logical_subset: np.ndarray
) -> np.ndarray:
    """
    Update frames_logical with a logical array of smaller length.

    The number of non-zero elements of frames_logical should be equal to the length of
    the logical subset.
    """
    if len(np.where(frames_logical != 0)[0]) != len(logical_subset):
        raise ValueError(
            f"len(frames_logical != 0)={len(np.where(frames_logical != 0)[0])} "
            f"but len(logical_subset)={len(logical_subset)}"
        )
    counter = 0
    for idx, val in enumerate(frames_logical):
        if val != 0:
            frames_logical[idx] = logical_subset[counter]
            counter += 1
    return frames_logical


def upsample(array: Union[np.ndarray, List], factor: int = 2) -> np.ndarray:
    """
    Upsample an array.

    :param array: the numpy array to be upsampled
    :param factor: int, factor for the upsampling
    :return: the upsampled numpy array
    """
    # check parameters
    array = np.asarray(array)
    if array.dtype in ["int8", "int16", "int32", "int64"]:
        array = array.astype(float)

    valid.valid_item(factor, allowed_types=int, min_excluded=0, name="factor")

    # current points positions in each dimension
    old_positions = [np.arange(val) for val in array.shape]

    # calculate the new positions
    new_shape = [val * factor for val in array.shape]
    upsampled_positions = [
        np.linspace(0, val - 1, num=val * factor) for val in array.shape
    ]
    grid = np.meshgrid(*upsampled_positions, indexing="ij")
    new_grid = np.asarray(
        np.concatenate(
            [new_grid.reshape((1, new_grid.size)) for _, new_grid in enumerate(grid)]
        ).transpose()
    )
    # interpolate array #
    rgi = RegularGridInterpolator(
        old_positions,
        array,
        method="linear",
        bounds_error=False,
        fill_value=0,
    )
    return np.asarray(rgi(new_grid).reshape(new_shape).astype(array.dtype))


def wrap(obj, start_angle, range_angle):
    """
    Wrap obj between start_angle and (start_angle + range_angle).

    :param obj: number or array to be wrapped
    :param start_angle: start angle of the range
    :param range_angle: range
    :return: wrapped angle in [start_angle, start_angle+range[
    """
    return (obj - start_angle + range_angle) % range_angle + start_angle
