# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr
"""Functions related to visualization."""

from lmfit import minimize, Parameters
import numpy as np
from numbers import Real
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import matplotlib.colors as colors
from operator import itemgetter
import pathlib
from scipy.interpolate import griddata
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks
import sys
from typing import Dict, List, Optional, Tuple

from bcdi.graph.colormap import ColormapFactory
from bcdi.utils import utilities as util
from bcdi.utils import validation as valid

default_cmap = ColormapFactory().generate_cmap()


def close_event(event):
    """Handle closing events on plots."""
    print(event, "Click on the figure instead of closing it!")
    sys.exit()


def colorbar(mappable, scale="linear", numticks=10, label=None, pad=0.05):
    """
    Generate a colorbar whose height (or width) in sync with the master axes.

    :param mappable: the image where to put the colorbar
    :param scale: 'linear' or 'log', used for tick location
    :param numticks: number of ticks for the colorbar
    :param label: label for the colorbar
    :param pad: float (default 0.05). Fraction of original axes between colorbar and
     new image axes.
    :return: the colorbar instance
    """
    last_axes = plt.gca()
    try:
        ax = mappable.axes
    except AttributeError:  # QuadContourSet
        ax = mappable.ax
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=pad)
    cbar = fig.colorbar(mappable, cax=cax)
    if scale == "linear":
        cbar.locator = ticker.LinearLocator(numticks=numticks)
    elif scale == "log":
        cbar.locator = ticker.LogLocator(numticks=numticks)
    else:
        raise ValueError("Incorrect value for the scale parameter")
    if label is not None:
        cbar.ax.set_ylabel(label)
    cbar.update_ticks()
    plt.sca(last_axes)
    return cbar


def combined_plots(
    tuple_array,
    tuple_sum_frames,
    tuple_colorbar,
    tuple_title,
    tuple_scale,
    tuple_sum_axis=None,
    cmap=default_cmap,
    tick_direction="out",
    tick_width=1,
    tick_length=4,
    pixel_spacing=None,
    tuple_width_v=None,
    tuple_width_h=None,
    tuple_vmin=np.nan,
    tuple_vmax=np.nan,
    is_orthogonal=False,
    reciprocal_space=False,
    **kwargs,
):
    """
    Subplots of a 1D, 2D or 3D datasets using user-defined parameters.

    :param tuple_array: tuple of 1D, 2D or 3D arrays of real numbers
    :param tuple_sum_frames: boolean or tuple of boolean values. If True, will sum the
     data along sum_axis
    :param tuple_sum_axis: tuple of axis along which to sum or to take the middle slice
    :param tuple_width_v: int or tuple of user-defined zoom vertical width, should be
     smaller than the actual data size. Set it to None if you do not need it.
    :param tuple_width_h: int or tuple of user-defined zoom horizontal width, should be
     smaller than the actual data size. Set it to None if you do not need it.
    :param tuple_colorbar: boolean or tuple of boolean values. Set it to True in order
     to plot the colorbar
    :param tuple_vmin: float or tuple of lower boundaries for the colorbar,
     set to np.nan if you do not need it
    :param tuple_vmax: float or tuple of higher boundaries for the colorbar,
     set to np.nan if you do not need it
    :param tuple_title: string or tuple of strings, set to '' if you do not need it
    :param tuple_scale:  string ot tuple of strings with value 'linear' or 'log'
    :param cmap: colormap to be used
    :param tick_direction: 'out', 'in', 'inout'
    :param tick_width: width of tickes in plots
    :param tick_length: length of tickes in plots
    :param pixel_spacing: pixel_spacing = desired tick_spacing (in nm) / voxel_size
     of the reconstruction(in nm). It can be  a positive number or a tuple of
     array.ndim positive numbers
    :param is_orthogonal: set to True is the frame is orthogonal, False otherwise
     (detector frame) Used for plot labels.
    :param reciprocal_space: True if the data is in reciprocal space, False otherwise.
     Used for plot labels.
    :param kwargs:
     - 'xlabel' , label of the horizontal axis for plots: string or tuple of strings
     - 'ylabel' , label of the vertical axis for plots: string or tuple of strings
     - 'position' , tuple of subplot positions in the format 231 (2 rows, 3 columns,
       first subplot)
     - 'invert_y': boolean, True to invert the vertical axis of the plot.
       Will overwrite the default behavior.

    :return:  the figure instance
    """
    mpl.rcParams["axes.linewidth"] = tick_width  # set the linewidth globally
    ####################
    # check parameters #
    ####################
    tuple_sum_axis = tuple_sum_axis or 0
    invert_yaxis = False

    if isinstance(tuple_array, np.ndarray):
        tuple_array = (tuple_array,)
    valid.valid_ndarray(tuple_array, ndim=(1, 2, 3), fix_ndim=False)
    nb_subplots = len(tuple_array)

    if isinstance(tuple_sum_frames, bool):
        tuple_sum_frames = (tuple_sum_frames,) * nb_subplots
    valid.valid_container(
        obj=tuple_sum_frames,
        container_types=(tuple, list),
        length=nb_subplots,
        item_types=bool,
        name="graph_utils.combined_plots",
    )
    if isinstance(tuple_sum_axis, int):
        tuple_sum_axis = (tuple_sum_axis,) * nb_subplots
    valid.valid_container(
        obj=tuple_sum_axis,
        container_types=(tuple, list),
        length=nb_subplots,
        item_types=int,
        allow_none=True,
        min_included=0,
        name="graph_utils.combined_plots",
    )
    if any(sum_axis not in {0, 1, 2} for sum_axis in tuple_sum_axis):
        raise ValueError("sum_axis should be either 0, 1 or 2")

    if isinstance(tuple_width_v, int) or tuple_width_v is None:
        tuple_width_v = (tuple_width_v,) * nb_subplots
    valid.valid_container(
        obj=tuple_width_v,
        container_types=(tuple, list),
        length=nb_subplots,
        item_types=int,
        allow_none=True,
        min_excluded=0,
        name="graph_utils.combined_plots",
    )
    if isinstance(tuple_width_h, int) or tuple_width_h is None:
        tuple_width_h = (tuple_width_h,) * nb_subplots
    valid.valid_container(
        obj=tuple_width_h,
        container_types=(tuple, list),
        length=nb_subplots,
        item_types=int,
        allow_none=True,
        min_excluded=0,
        name="graph_utils.combined_plots",
    )
    if isinstance(tuple_colorbar, bool):
        tuple_colorbar = (tuple_colorbar,) * nb_subplots
    valid.valid_container(
        obj=tuple_colorbar,
        container_types=(tuple, list),
        length=nb_subplots,
        item_types=bool,
        name="graph_utils.combined_plots",
    )
    if isinstance(tuple_vmin, Real):
        tuple_vmin = (tuple_vmin,) * nb_subplots
    valid.valid_container(
        obj=tuple_vmin,
        container_types=(tuple, list),
        length=nb_subplots,
        item_types=Real,
        name="graph_utils.combined_plots",
    )
    if isinstance(tuple_vmax, Real):
        tuple_vmax = (tuple_vmax,) * nb_subplots
    valid.valid_container(
        obj=tuple_vmax,
        container_types=(tuple, list),
        length=nb_subplots,
        item_types=Real,
        name="graph_utils.combined_plots",
    )
    if any(
        vmin >= vmax
        for vmin, vmax in zip(tuple_vmin, tuple_vmax)
        if not np.isnan(vmin) and not np.isnan(vmax)
    ):
        raise ValueError("vmin should be strictly smaller than vmax")

    if isinstance(tuple_title, str):
        tuple_title = (tuple_title,) * nb_subplots
    valid.valid_container(
        obj=tuple_title,
        container_types=(tuple, list),
        length=nb_subplots,
        item_types=str,
        name="graph_utils.combined_plots",
    )
    if isinstance(tuple_scale, str):
        tuple_scale = (tuple_scale,) * nb_subplots
    valid.valid_container(
        obj=tuple_scale,
        container_types=(tuple, list),
        length=nb_subplots,
        item_types=str,
        name="graph_utils.combined_plots",
    )
    if any(scale not in {"linear", "log"} for scale in tuple_scale):
        raise ValueError('scale should be either "linear" or "log"')

    #########################
    # load and check kwargs #
    #########################
    valid.valid_kwargs(
        kwargs=kwargs,
        allowed_kwargs={"xlabel", "ylabel", "position", "invert_y"},
        name="graph_utils.combined_plots",
    )
    xlabel = kwargs.get("xlabel", "")
    ylabel = kwargs.get("ylabel", "")
    position = kwargs.get("position")
    invert_y = kwargs.get("invert_y", [None for _ in range(nb_subplots)])

    if isinstance(xlabel, str):
        xlabel = (xlabel,) * nb_subplots
    valid.valid_container(
        obj=xlabel,
        container_types=(tuple, list),
        length=nb_subplots,
        item_types=str,
        name="graph_utils.combined_plots",
    )
    if isinstance(ylabel, str):
        ylabel = (ylabel,) * nb_subplots
    valid.valid_container(
        obj=ylabel,
        container_types=(tuple, list),
        length=nb_subplots,
        item_types=str,
        name="graph_utils.combined_plots",
    )
    if position is None:
        nb_columns = nb_subplots // 2
        nb_rows = nb_subplots // nb_columns + nb_subplots % nb_columns
        position = [
            nb_rows * 100 + nb_columns * 10 + index
            for index in range(1, nb_subplots + 1)
        ]
    valid.valid_container(
        obj=position,
        container_types=(tuple, list),
        length=nb_subplots,
        item_types=int,
        name="graph_utils.combined_plots",
    )
    if isinstance(invert_y, bool):
        invert_y = (invert_y,) * nb_subplots
    valid.valid_container(
        obj=invert_y,
        container_types=(tuple, list),
        length=nb_subplots,
        item_types=bool,
        allow_none=True,
        name="graph_utils.combined_plots",
    )

    ##############################
    # plot subplots sequentially #
    ##############################
    plt.ion()
    fig = plt.figure(figsize=(12, 9))
    for idx in range(nb_subplots):

        axis = plt.subplot(position[idx])

        array = tuple_array[idx]
        sum_frames = tuple_sum_frames[idx]
        sum_axis = tuple_sum_axis[idx]
        width_v = tuple_width_v[idx]
        width_h = tuple_width_h[idx]
        plot_colorbar = tuple_colorbar[idx]
        vmin = tuple_vmin[idx]
        vmax = tuple_vmax[idx]
        title = tuple_title[idx]
        scale = tuple_scale[idx]

        nb_dim = array.ndim
        if nb_dim in {2, 3}:
            if isinstance(pixel_spacing, Real):
                pixel_spacing = (pixel_spacing,) * nb_dim
            valid.valid_container(
                obj=pixel_spacing,
                container_types=(tuple, list),
                length=nb_dim,
                item_types=Real,
                min_excluded=0,
                allow_none=True,
                name="graph_utils.combined_plots",
            )

        if nb_dim not in {1, 2, 3}:
            print("array ", idx, ": wrong number of dimensions")
            continue

        if nb_dim == 1:

            if np.isnan(vmin):
                tmp_array = np.copy(array).astype(float)
                tmp_array[np.isnan(array)] = np.inf
                tmp_array[
                    np.isinf(tmp_array)
                ] = np.inf  # set -inf to +inf to find the min
                vmin = tmp_array.min()
            if np.isnan(vmax):
                tmp_array = np.copy(array).astype(float)
                tmp_array[np.isnan(array)] = -1 * np.inf
                tmp_array[np.isinf(tmp_array)] = (
                    -1 * np.inf
                )  # set +inf to -inf to find the max
                vmax = tmp_array.max()
                if np.isclose(vmax, vmin):
                    vmax = vmin + 1

            axis.plot(array)
            axis.set_title(title)
            axis.set_ylim(vmin, vmax)
            axis.set_yscale(scale)
            axis.set_xlabel(xlabel[idx])
            axis.set_ylabel(ylabel[idx])

            continue

        if nb_dim == 3:  # 3D, needs to be reduced to 2D by slicing or projecting
            invert_yaxis = bool(is_orthogonal and sum_axis == 0)

            slice_names, ver_labels, hor_labels = define_labels(
                reciprocal_space=reciprocal_space,
                is_orthogonal=is_orthogonal,
                sum_frames=sum_frames,
            )
            nbz, nby, nbx = array.shape
            width_v = width_v or max(nbz, nby, nbx)
            width_h = width_h or max(nbz, nby, nbx)

            if sum_axis == 0:
                dim_v = nby
                dim_h = nbx
                if pixel_spacing is not None:
                    pixel_spacing = (
                        pixel_spacing[1],
                        pixel_spacing[2],
                    )  # vertical, horizontal
                if not sum_frames:
                    array = array[nbz // 2, :, :]
                else:
                    array = array.sum(axis=sum_axis)
                default_xlabel = hor_labels[0]
                default_ylabel = ver_labels[0]
            elif sum_axis == 1:
                dim_v = nbz
                dim_h = nbx
                if pixel_spacing is not None:
                    pixel_spacing = (
                        pixel_spacing[0],
                        pixel_spacing[2],
                    )  # vertical, horizontal
                if not sum_frames:
                    array = array[:, nby // 2, :]
                else:
                    array = array.sum(axis=sum_axis)
                default_xlabel = hor_labels[1]
                default_ylabel = ver_labels[1]
            else:  # sum_axis == 2:
                dim_v = nbz
                dim_h = nby
                if pixel_spacing is not None:
                    pixel_spacing = (
                        pixel_spacing[0],
                        pixel_spacing[1],
                    )  # vertical, horizontal
                if not sum_frames:
                    array = array[:, :, nbx // 2]
                else:
                    array = array.sum(axis=sum_axis)
                default_xlabel = hor_labels[2]
                default_ylabel = ver_labels[2]

            slice_name = slice_names[sum_axis]

        else:  # 2D
            nby, nbx = array.shape
            width_v = width_v or max(nby, nbx)
            width_h = width_h or max(nby, nbx)

            dim_v = nby
            dim_h = nbx
            slice_name = ""
            default_xlabel = ""
            default_ylabel = ""

        ############################
        # now array is 2D, plot it #
        ############################
        if invert_y[idx] is not None:  # overwrite invert_yaxis parameter
            invert_yaxis = invert_y[idx]

        width_v = min(width_v, dim_v)
        width_h = min(width_h, dim_h)
        array = array[
            int(np.rint(dim_v / 2 - width_v / 2)) : int(
                np.rint(dim_v / 2 - width_v / 2)
            )
            + width_v,
            int(np.rint(dim_h // 2 - width_h // 2)) : int(
                np.rint(dim_h // 2 - width_h // 2)
            )
            + width_h,
        ]

        if scale == "linear":
            if np.isnan(vmin):
                tmp_array = np.copy(array)
                tmp_array[np.isnan(array)] = np.inf
                tmp_array[
                    np.isinf(tmp_array)
                ] = np.inf  # set -inf to +inf to find the min
                vmin = tmp_array.min()
            if np.isnan(vmax):
                tmp_array = np.copy(array)
                tmp_array[np.isnan(array)] = -1 * np.inf
                tmp_array[np.isinf(tmp_array)] = (
                    -1 * np.inf
                )  # set +inf to -inf to find the max
                vmax = tmp_array.max()
                if np.isclose(vmax, vmin):
                    vmax = vmin + 1
            plot = axis.imshow(array, vmin=vmin, vmax=vmax, cmap=cmap)
        else:  # 'log'
            if np.isnan(vmin):
                tmp_array = np.copy(array)
                tmp_array[np.isnan(array)] = np.inf
                tmp_array[
                    np.isinf(tmp_array)
                ] = np.inf  # set -inf to +inf to find the min
                vmin = np.log10(abs(tmp_array).min())
                if np.isinf(vmin):
                    vmin = 0
            if np.isnan(vmax):
                tmp_array = np.copy(array)
                tmp_array[np.isnan(array)] = -1 * np.inf
                tmp_array[np.isinf(tmp_array)] = (
                    -1 * np.inf
                )  # set +inf to -inf to find the max
                vmax = np.log10(abs(tmp_array).max())
                if np.isclose(vmax, vmin):
                    vmax = vmin + 1
            plot = axis.imshow(np.log10(abs(array)), vmin=vmin, vmax=vmax, cmap=cmap)

        axis.set_title(title + slice_name)
        if len(xlabel[idx]) != 0:
            axis.set_xlabel(xlabel[idx])
        else:
            axis.set_xlabel(default_xlabel)
        if len(ylabel[idx]) != 0:
            axis.set_ylabel(ylabel[idx])
        else:
            axis.set_ylabel(default_ylabel)
        plt.axis("scaled")
        axis.tick_params(direction=tick_direction, length=tick_length, width=tick_width)
        if pixel_spacing is not None:
            axis.xaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing[1]))
            axis.yaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing[0]))
            axis.tick_params(labelbottom=False, labelleft=False, top=True, right=True)
        if invert_yaxis:  # Y is axis 0, need to be flipped
            axis.invert_yaxis()
        if plot_colorbar:
            cbar = colorbar(plot, numticks=5)
            cbar.ax.tick_params(length=tick_length, width=tick_width)

    plt.tight_layout()  # avoids the overlap of subplots with axes labels
    plt.pause(0.1)
    plt.ioff()

    return fig


def contour_slices(
    array,
    q_coordinates,
    sum_frames=False,
    slice_position=None,
    levels=150,
    width_z=None,
    width_y=None,
    width_x=None,
    plot_colorbar=False,
    cmap=default_cmap,
    title="",
    scale="linear",
    is_orthogonal=False,
    reciprocal_space=True,
):
    """
    Create a figure with three 2D contour plots from a 3D dataset.

    :param array: 3D array of real numbers
    :param q_coordinates: a tuple of (qx, qz, qy) 1D-coordinates corresponding to the
     (Z, Y, X) of the cxi convention
    :param sum_frames: if True, will sum the data along the 3rd axis
    :param slice_position: tuple of three integers where to slice the 3D array
    :param levels: int n, will use n data intervals and draw n+1 contour lines
    :param width_z: user-defined zoom width along axis 0 (rocking angle), should be
     smaller than the actual data size
    :param width_y: user-defined zoom width along axis 1 (vertical), should be smaller
     than the actual data size
    :param width_x: user-defined zoom width along axis 2 (horizontal), should be
     smaller than the actual data size
    :param plot_colorbar: set it to True in order to plot the colorbar
    :param cmap: colormap to be used
    :param title: string to include in the plot
    :param scale: 'linear' or 'log'
    :param is_orthogonal: set to True is the frame is orthogonal, False otherwise
     (detector frame) Used for plot labels.
    :param reciprocal_space: True if the data is in reciprocal space, False otherwise.
     Used for plot labels.
    :return: fig, (ax0, ax1, ax2, ax3), (plt0, plt1, plt2) instances
    """
    #########################
    # check some parameters #
    #########################
    valid.valid_ndarray(array, ndim=3)
    if scale not in {"linear", "log"}:
        raise ValueError('scale should be either "linear" or "log"')
    if any(len(qval) != shape for qval, shape in zip(q_coordinates, array.shape)):
        raise ValueError("Coordinates shape is not compatible with data shape")

    nbz, nby, nbx = array.shape
    qx, qz, qy = q_coordinates

    width_z = width_z or nbz
    width_y = width_y or nby
    width_x = width_x or nbx

    if not sum_frames:
        slice_position = slice_position or (int(nbz // 2), int(nby // 2), int(nbx // 2))
        valid.valid_container(
            obj=slice_position,
            container_types=(tuple, list),
            length=3,
            item_types=int,
            min_included=0,
            name="graph_utils.contour_slices",
        )

    #######################################
    # create the figure and plot subplots #
    #######################################
    slice_names, ver_labels, hor_labels = define_labels(
        reciprocal_space=reciprocal_space,
        is_orthogonal=is_orthogonal,
        sum_frames=sum_frames,
    )
    plt.ion()
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2, figsize=(12, 9))

    ##########
    # axis 0 #
    ##########
    temp_array = np.copy(array)
    if not sum_frames:
        temp_array = temp_array[slice_position[0], :, :]
    else:
        temp_array = temp_array.sum(axis=0)
    # now array is 2D
    temp_array = temp_array[
        int(np.rint(nby / 2 - min(width_y, nby) / 2)) : int(
            np.rint(nby / 2 - min(width_y, nby) / 2)
        )
        + min(width_y, nby),
        int(np.rint(nbx // 2 - min(width_x, nbx) // 2)) : int(
            np.rint(nbx // 2 - min(width_x, nbx) // 2)
        )
        + min(width_x, nbx),
    ]

    if scale == "linear":
        plt0 = ax0.contourf(qy, qz, temp_array, levels, cmap=cmap)
    else:  # 'log'
        plt0 = ax0.contourf(qy, qz, np.log10(abs(temp_array)), levels, cmap=cmap)

    ax0.set_aspect("equal")
    ax0.set_xlabel(hor_labels[0])
    ax0.set_ylabel(ver_labels[0])
    ax0.set_title(title + slice_names[0])
    if plot_colorbar:
        colorbar(plt0, numticks=5)

    ##########
    # axis 1 #
    ##########
    temp_array = np.copy(array)
    if not sum_frames:
        temp_array = temp_array[:, slice_position[1], :]
    else:
        temp_array = temp_array.sum(axis=1)
    # now array is 2D
    temp_array = temp_array[
        int(np.rint(nbz / 2 - min(width_z, nbz) / 2)) : int(
            np.rint(nbz / 2 - min(width_z, nbz) / 2)
        )
        + min(width_z, nbz),
        int(np.rint(nbx // 2 - min(width_x, nbx) // 2)) : int(
            np.rint(nbx // 2 - min(width_x, nbx) // 2)
        )
        + min(width_x, nbx),
    ]

    if scale == "linear":
        plt1 = ax1.contourf(qy, qx, temp_array, levels, cmap=cmap)
    else:  # 'log'
        plt1 = ax1.contourf(qy, qx, np.log10(abs(temp_array)), levels, cmap=cmap)

    ax1.set_aspect("equal")
    ax1.set_xlabel(hor_labels[1])
    ax1.set_ylabel(ver_labels[1])
    ax1.set_title(title + slice_names[1])
    if plot_colorbar:
        colorbar(plt1, numticks=5)

    ##########
    # axis 2 #
    ##########
    temp_array = np.copy(array)
    if not sum_frames:
        temp_array = temp_array[:, :, slice_position[2]]
    else:
        temp_array = temp_array.sum(axis=2)
    # now array is 2D
    temp_array = temp_array[
        int(np.rint(nbz / 2 - min(width_z, nbz) / 2)) : int(
            np.rint(nbz / 2 - min(width_z, nbz) / 2)
        )
        + min(width_z, nbz),
        int(np.rint(nby // 2 - min(width_y, nby) // 2)) : int(
            np.rint(nby // 2 - min(width_y, nby) // 2)
        )
        + min(width_y, nby),
    ]

    if scale == "linear":
        plt2 = ax2.contourf(qz, qx, temp_array, levels, cmap=cmap)
    else:  # 'log'
        plt2 = ax2.contourf(qz, qx, np.log10(abs(temp_array)), levels, cmap=cmap)

    ax2.set_aspect("equal")
    ax2.set_xlabel(hor_labels[2])
    ax2.set_ylabel(ver_labels[2])
    ax2.set_title(title + slice_names[2])
    if plot_colorbar:
        colorbar(plt2, numticks=5)

    ##########
    # axis 3 #
    ##########
    ax3.set_visible(False)

    plt.tight_layout()  # avoids the overlap of subplots with axes labels
    plt.pause(0.1)
    plt.ioff()
    return fig, (ax0, ax1, ax2, ax3), (plt0, plt1, plt2)


def contour_stereographic(
    euclidian_u,
    euclidian_v,
    color,
    radius_mean,
    planes=None,
    title="",
    plot_planes=True,
    contour_range=None,
    max_angle=95,
    cmap=default_cmap,
    uv_labels=("", ""),
    hide_axis=False,
    scale="linear",
    debugging=False,
):
    """
    Plot the stereographic projection with some cosmetics.

    :param euclidian_u: flattened array, normalized Euclidian metric coordinates
     (points can be not on a regular grid)
    :param euclidian_v: flattened array, normalized Euclidian metric coordinates
     (points can be not on a regular grid)
    :param color: flattened array, intensity of density kernel estimation at radius_mean
    :param radius_mean: radius of the sphere in reciprocal space from which the
     projection is done
    :param planes: dictionnary of crystallographic planes, e.g.
     {'111':angle_with_reflection}
    :param title: title for the stereographic plot
    :param plot_planes: if True, will draw circle corresponding to crystallographic
     planes in the pole figure
    :param contour_range: range for the plot contours
    :param max_angle: maximum angle in degrees of the stereographic projection
     (should be larger than 90)
    :param cmap: colormap to be used
    :param uv_labels: tuple of strings, labels for the u axis and the v axis,
     respectively
    :param hide_axis: hide the axis frame, ticks and ticks labels
    :param scale: 'linear' or 'log', scale for the colorbar of the plot
    :param debugging: True to see the scatter plot of euclidian coordinates
    :return: figure and axe instances
    """
    if scale not in {"linear", "log"}:
        raise ValueError('scale should be either "linear" or "log"')
    if contour_range is None:
        if scale == "linear":
            contour_range = range(0, 10001, 250)
        else:  # 'log'
            contour_range = np.logspace(0, 4, num=20, endpoint=True, base=10.0)

    if debugging:
        color2 = np.copy(color)
        color2 = color2 / abs(color2[~np.isnan(color2)]).max() * 10000
        _, ax0 = plt.subplots(nrows=1, ncols=1)
        plt0 = ax0.scatter(
            euclidian_u,
            euclidian_v,
            s=6,
            c=color2,
            cmap=default_cmap,
            norm=colors.LogNorm(
                vmin=max(color2[~np.isnan(color2)].min(), 1),
                vmax=color2[~np.isnan(color2)].max(),
            ),
        )
        circle = patches.Circle((0, 0), 90, color="k", fill=False, linewidth=1.5)
        ax0.add_artist(circle)
        ax0.axis("scaled")
        ax0.set_xlim(-max_angle, max_angle)
        ax0.set_ylim(-max_angle, max_angle)
        ax0.set_xlabel("u " + uv_labels[0])
        ax0.set_ylabel("v " + uv_labels[1])
        ax0.set_title(title)
        colorbar(plt0, scale="log", numticks=5)

    nb_points = 5 * max_angle + 1
    v_grid, u_grid = np.mgrid[
        -max_angle : max_angle : (nb_points * 1j),
        -max_angle : max_angle : (nb_points * 1j),
    ]
    # v_grid is changing along the vertical axis,
    # u_grid is changing along the horizontal axis
    intensity_grid = griddata(
        (euclidian_v, euclidian_u), color, (v_grid, u_grid), method="linear"
    )
    nan_indices = np.isnan(intensity_grid)
    # normalize the intensity for easier plotting
    intensity_grid = intensity_grid / abs(intensity_grid[~nan_indices]).max() * 10000

    #####################################
    # plot the stereographic projection #
    #####################################
    plt.ion()
    fig, ax0 = plt.subplots(
        nrows=1, ncols=1, figsize=(12, 9), facecolor="w", edgecolor="k"
    )
    if scale == "linear":
        plt0 = ax0.contourf(u_grid, v_grid, intensity_grid, contour_range, cmap=cmap)
        colorbar(plt0, scale="linear", numticks=5)
    else:  # log
        plt0 = ax0.contourf(
            u_grid,
            v_grid,
            intensity_grid,
            contour_range,
            cmap=cmap,
            norm=colors.LogNorm(
                vmin=max(intensity_grid[~nan_indices].min(), 1),
                vmax=intensity_grid[~nan_indices].max(),
            ),
        )
        colorbar(plt0, scale="log", numticks=5)
    ax0.axis("equal")

    # add the projection of the elevation angle, depending on the center of projection
    for ii in range(15, 90, 5):
        circle = patches.Circle(
            (0, 0),
            radius_mean
            * np.sin(ii * np.pi / 180)
            / (1 + np.cos(ii * np.pi / 180))
            * 90
            / radius_mean,
            color="grey",
            fill=False,
            linestyle="dotted",
            linewidth=0.5,
        )
        ax0.add_artist(circle)
    for ii in range(10, 90, 20):
        circle = patches.Circle(
            (0, 0),
            radius_mean
            * np.sin(ii * np.pi / 180)
            / (1 + np.cos(ii * np.pi / 180))
            * 90
            / radius_mean,
            color="grey",
            fill=False,
            linestyle="dotted",
            linewidth=1,
        )
        ax0.add_artist(circle)
    for ii in range(10, 95, 20):
        ax0.text(
            -radius_mean
            * np.sin(ii * np.pi / 180)
            / (1 + np.cos(ii * np.pi / 180))
            * 90
            / radius_mean,
            0,
            str(ii) + r"$^\circ$",
            fontsize=10,
            color="k",
        )
    circle = patches.Circle((0, 0), 90, color="k", fill=False, linewidth=1.5)
    ax0.add_artist(circle)

    # add azimutal lines every 5 and 45 degrees
    for ii in range(5, 365, 5):
        ax0.plot(
            [0, 90 * np.cos(ii * np.pi / 180)],
            [0, 90 * np.sin(ii * np.pi / 180)],
            color="grey",
            linestyle="dotted",
            linewidth=0.5,
        )
    for ii in range(0, 365, 20):
        ax0.plot(
            [0, 90 * np.cos(ii * np.pi / 180)],
            [0, 90 * np.sin(ii * np.pi / 180)],
            color="grey",
            linestyle="dotted",
            linewidth=1,
        )

    # draw circles corresponding to particular reflection
    if planes and plot_planes == 1:
        indx = 0
        for key, value in planes.items():
            circle = patches.Circle(
                (0, 0),
                radius_mean
                * np.sin(value * np.pi / 180)
                / (1 + np.cos(value * np.pi / 180))
                * 90
                / radius_mean,
                color="g",
                fill=False,
                linestyle="dotted",
                linewidth=1.5,
            )
            ax0.add_artist(circle)
            ax0.text(
                np.cos(indx * np.pi / 180)
                * radius_mean
                * np.sin(value * np.pi / 180)
                / (1 + np.cos(value * np.pi / 180))
                * 90
                / radius_mean,
                np.sin(indx * np.pi / 180)
                * radius_mean
                * np.sin(value * np.pi / 180)
                / (1 + np.cos(value * np.pi / 180))
                * 90
                / radius_mean,
                key,
                fontsize=10,
                color="k",
                fontweight="bold",
            )
            indx = indx + 6
            print(key + ": ", str("{:.2f}".format(value)))
        print("\n")
    ax0.set_xlabel("u " + uv_labels[0])
    ax0.set_ylabel("v " + uv_labels[1])
    if hide_axis:
        ax0.axis("off")
        ax0.set_title(title + "\nu horizontal, v vertical")
    else:
        ax0.set_title(title)
    ax0.axis("scaled")
    plt.pause(0.1)
    plt.ioff()
    return fig, ax0


def define_labels(reciprocal_space, is_orthogonal, sum_frames, labels=None):
    """
    Define default labels for plots.

    :param reciprocal_space: True if the data is in reciprocal space, False otherwise
    :param is_orthogonal: True is the frame is orthogonal, False otherwise
     (detector frame)
    :param sum_frames: True if the the data is summed along some axis
    :param labels: tuple of two strings (vertical label, horizontal label)
    :return: three tuples of three elements: slice_names, vertical labels,
     horizontal labels. The first element in the tuple corresponds to the first
     subplot and so on.
    """
    labels = labels or ("",) * 2

    if reciprocal_space:
        if is_orthogonal:
            if sum_frames:
                slice_names = (
                    " sum along Q$_x$",
                    " sum along Q$_z$",
                    " sum along Q$_y$",
                )
            else:
                slice_names = (" slice in Q$_x$", " slice in Q$_z$", " slice in Q$_y$")
            ver_labels = (
                labels[0] + r" Q$_z$",
                labels[0] + r" Q$_x$",
                labels[0] + r" Q$_x$",
            )
            hor_labels = (
                labels[1] + r" Q$_y$",
                labels[1] + r" Q$_y$",
                labels[1] + r" Q$_z$",
            )
        else:  # detector frame
            if sum_frames:
                slice_names = (" sum along Z", " sum along Y", " sum along X")
            else:
                slice_names = (" slice in Z", " slice in Y", " slice in X")
            ver_labels = (
                labels[0] + " Y",
                labels[0] + " rocking angle",
                labels[0] + " rocking angle",
            )
            hor_labels = (labels[1] + " X", labels[1] + " X", labels[1] + " Y")
    else:
        if is_orthogonal:
            if sum_frames:
                slice_names = (" sum along z", " sum along y", " sum along x")
            else:
                slice_names = (" slice in z", " slice in y", " slice in x")
            ver_labels = (labels[0] + " y", labels[0] + " z", labels[0] + " z")
            hor_labels = (labels[1] + " x", labels[1] + " x", labels[1] + " y")
        else:  # detector frame
            if sum_frames:
                slice_names = (" sum along Z", " sum along Y", " sum along X")
            else:
                slice_names = (" slice in Z", " slice in Y", " slice in X")
            ver_labels = (
                labels[0] + " Y",
                labels[0] + " rocking angle",
                labels[0] + " rocking angle",
            )
            hor_labels = (labels[1] + " X", labels[1] + " X", labels[1] + " Y")

    return slice_names, ver_labels, hor_labels


def fit_linecut(
    array: np.ndarray,
    indices: Optional[List[Tuple[int, ...]]] = None,
    fit_derivative: bool = False,
    support_threshold: float = 0.25,
    voxel_sizes: Optional[List[float]] = None,
    filename: Optional[str] = None,
) -> Dict:
    # check parameters
    valid.valid_ndarray(array, fix_shape=True, name="array")
    shape = array.shape
    ndim = len(shape)
    if indices is None:
        # default to the linecut through the center of the array in each dimension
        indices = []

        for idx, shp in enumerate(shape):
            default = [(val // 2, val // 2) for val in shape]
            default[idx] = (0, shp)
            indices.append(default)

    valid.valid_container(
        indices,
        container_types=list,
        item_types=list,
        length=ndim,
        name="indices",
    )
    for _, item in enumerate(indices):
        valid.valid_container(
            item,
            container_types=list,
            item_types=tuple,
            length=ndim,
            name="indices sublists",
        )
    if not isinstance(fit_derivative, bool):
        raise TypeError(f"fit_derivative should be a bool, got {type(fit_derivative)}")

    # generate a linecut for each dimension of the array
    result = {}
    for idx in range(ndim):
        result[f"dimension_{idx}"] = {}
        # generate the linecut
        cut = linecut(array=array, indices=indices[idx])
        result[f"dimension_{idx}"][f"linecut"] = np.vstack(
            (np.arange(indices[idx][idx][0], indices[idx][idx][1]), cut)
        )

        # optionally fit the derivative at the edge of the support
        if fit_derivative:
            support = np.zeros(shape)
            support[array > support_threshold] = 1
            support_cut = linecut(array=support, indices=indices[idx])

            peaks, metadata = find_peaks(
                abs(np.gradient(support_cut)), height=0.1, distance=10, width=1
            )
            dcut = abs(np.gradient(cut))

            # setup data and parameters for fitting
            combined_xaxis = []
            combined_data = []
            fit_params = Parameters()
            for peak_id, peak in enumerate(peaks):
                cropped_xaxis = np.arange(peak - 10, peak + 10)
                cropped_dcut = dcut[peak - 10 : peak + 10]
                result[f"dimension_{idx}"][f"derivative_{peak_id}"] = np.vstack(
                    (cropped_xaxis, cropped_dcut)
                )
                combined_xaxis.append(cropped_xaxis)
                combined_data.append(cropped_dcut)

                cen = peak
                fit_params.add("amp_%i" % (peak_id + 1), value=1, min=0.0, max=10)
                fit_params.add(
                    "cen_%i" % (peak_id + 1), value=cen, min=cen - 1, max=cen + 1
                )
                fit_params.add("sig_%i" % (peak_id + 1), value=2, min=0.1, max=10)

            # fit the data
            combined_xaxis = np.asarray(combined_xaxis)
            combined_data = np.asarray(combined_data)
            fit_result = minimize(
                util.objective_lmfit,
                fit_params,
                args=(
                    combined_xaxis,
                    combined_data,
                    "gaussian",
                ),
            )

            # generate fit curves
            for peak_id, peak in enumerate(peaks):
                interp_xaxis = util.upsample(combined_xaxis[peak_id], factor=4)
                # interp_xaxis = combined_xaxis[peak_id]
                y_fit = util.function_lmfit(
                    params=fit_result.params,
                    iterator=peak_id,
                    x_axis=interp_xaxis,
                    distribution="gaussian",
                )
                result[f"dimension_{idx}"][f"fit_{peak_id}"] = np.vstack(
                    (interp_xaxis, y_fit)
                )
                result[f"dimension_{idx}"][f"param_{peak_id}"] = {
                    "amp": fit_result.params[f"amp_{peak_id}"].value,
                    "sig": fit_result.params[f"sig_{peak_id}"].value,
                    "cen": fit_result.params[f"cen_{peak_id}"].value,
                }

    # plot the cut and optionally the fits
    plot_linecut(linecuts=result, filename=filename, voxel_sizes=voxel_sizes)
    return result


def imshow_plot(
    array,
    sum_frames=False,
    sum_axis=0,
    width_v=None,
    width_h=None,
    plot_colorbar=False,
    vmin=np.nan,
    vmax=np.nan,
    cmap=default_cmap,
    title="",
    labels=None,
    scale="linear",
    tick_direction="out",
    tick_width=1,
    tick_length=4,
    pixel_spacing=None,
    is_orthogonal=False,
    reciprocal_space=False,
    **kwargs,
):
    """
    2D imshow plot of a 2D or 3D dataset using user-defined parameters.

    :param array: 2D or 3D array of real numbers
    :param sum_frames: if True, will sum the data along sum_axis
    :param sum_axis: axis along which to sum
    :param width_v: user-defined zoom vertical width, should be smaller than the
     actual data size
    :param width_h: user-defined zoom horizontal width, should be smaller than the
     actual data size
    :param plot_colorbar: set it to True in order to plot the colorbar
    :param vmin: lower boundary for the colorbar
    :param vmax: higher boundary for the colorbar
    :param cmap: colormap to be used
    :param title: string to include in the plot
    :param labels: tuple of two strings (vertical label, horizontal label)
    :param scale: 'linear' or 'log'
    :param tick_direction: 'out', 'in', 'inout'
    :param tick_width: width of tickes in plots
    :param tick_length: length of tickes in plots
    :param pixel_spacing: pixel_spacing = desired tick_spacing (in nm) / voxel_size
     of the reconstruction(in nm). It can
     be  a positive number or a tuple of array.ndim positive numbers
    :param is_orthogonal: True is the array is in an orthogonal basis,
     False otherwise (detector frame). Used for plot labels.
    :param reciprocal_space: True if the data is in reciprocal space,
     False otherwise. Used for plot labels.
    :param kwargs:
     - 'invert_y': boolean, True to invert the vertical axis of the plot.
       Will overwrite the default behavior.

    :return:  fig, axis, plot instances
    """
    mpl.rcParams["axes.linewidth"] = tick_width  # set the linewidth globally
    #########################
    # check some parameters #
    #########################
    valid.valid_ndarray(array, ndim=(2, 3))
    if sum_axis not in {0, 1, 2}:
        raise ValueError("sum_axis should be either 0, 1 or 2")
    if not isinstance(sum_frames, bool):
        raise TypeError("sum_frames should be a boolean")
    if scale not in {"linear", "log"}:
        raise ValueError('scale should be either "linear" or "log"')
    if not np.isnan(vmin) and not np.isnan(vmax) and vmin >= vmax:
        raise ValueError("vmin should be strictly smaller than vmax")
    ###############
    # load kwargs #
    ###############
    valid.valid_kwargs(
        kwargs=kwargs, allowed_kwargs={"invert_y"}, name="graph_utils.imshow_plot"
    )
    invert_y = kwargs.get("invert_y")

    nb_dim = array.ndim

    if isinstance(pixel_spacing, Real):
        pixel_spacing = (pixel_spacing,) * nb_dim
        valid.valid_container(
            obj=pixel_spacing,
            container_types=(tuple, list),
            length=nb_dim,
            item_types=Real,
            min_excluded=0,
            allow_none=True,
            name="graph_utils.imshow_plot",
        )

    labels = labels or ("",) * 2
    valid.valid_container(
        obj=labels,
        container_types=(tuple, list),
        length=2,
        item_types=str,
        name="graph_utils.imshow_plot",
    )

    array = array.astype(float)
    plt.ion()

    if nb_dim == 3:
        invert_yaxis = bool(is_orthogonal)

        slice_names, ver_labels, hor_labels = define_labels(
            reciprocal_space=reciprocal_space,
            labels=labels,
            is_orthogonal=is_orthogonal,
            sum_frames=sum_frames,
        )

        nbz, nby, nbx = array.shape
        width_v = width_v or max(nbz, nby, nbx)
        width_h = width_h or max(nbz, nby, nbx)

        if sum_axis == 0:
            dim_v = nby
            dim_h = nbx
            if pixel_spacing is not None:
                pixel_spacing = (
                    pixel_spacing[1],
                    pixel_spacing[2],
                )  # vertical, horizontal
            if not sum_frames:
                array = array[nbz // 2, :, :]
            else:
                array = array.sum(axis=sum_axis)
        elif sum_axis == 1:
            dim_v = nbz
            dim_h = nbx
            if pixel_spacing is not None:
                pixel_spacing = (
                    pixel_spacing[0],
                    pixel_spacing[2],
                )  # vertical, horizontal
            if not sum_frames:
                array = array[:, nby // 2, :]
            else:
                array = array.sum(axis=sum_axis)
        else:  # 2
            dim_v = nbz
            dim_h = nby
            if pixel_spacing is not None:
                pixel_spacing = (
                    pixel_spacing[0],
                    pixel_spacing[1],
                )  # vertical, horizontal
            if not sum_frames:
                array = array[:, :, nbx // 2]
            else:
                array = array.sum(axis=sum_axis)

        slice_name = slice_names[sum_axis]
        ver_label = ver_labels[sum_axis]
        hor_label = hor_labels[sum_axis]

    else:  # array is 2D
        invert_yaxis = False
        nby, nbx = array.shape
        width_v = width_v or max(nby, nbx)
        width_h = width_h or max(nby, nbx)

        dim_v = nby
        dim_h = nbx
        slice_name, ver_label, hor_label = "", labels[0], labels[1]

    ############################
    # now array is 2D, plot it #
    ############################
    if invert_y is not None:  # overwrite invert_yaxis parameter
        invert_yaxis = invert_y

    fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(12, 9))
    width_v = min(width_v, dim_v)
    width_h = min(width_h, dim_h)
    array = array[
        int(np.rint(dim_v / 2 - width_v / 2)) : int(np.rint(dim_v / 2 - width_v / 2))
        + width_v,
        int(np.rint(dim_h // 2 - width_h // 2)) : int(
            np.rint(dim_h // 2 - width_h // 2)
        )
        + width_h,
    ]

    if scale == "linear":
        if np.isnan(vmin):
            tmp_array = np.copy(array)
            tmp_array[np.isnan(array)] = np.inf
            tmp_array[np.isinf(tmp_array)] = np.inf  # set -inf to +inf to find the min
            vmin = tmp_array.min()
        if np.isnan(vmax):
            tmp_array = np.copy(array)
            tmp_array[np.isnan(array)] = -1 * np.inf
            tmp_array[np.isinf(tmp_array)] = (
                -1 * np.inf
            )  # set +inf to -inf to find the max
            vmax = tmp_array.max()
        plot = axis.imshow(array, vmin=vmin, vmax=vmax, cmap=cmap)
    else:  # 'log'
        if np.isnan(vmin):
            tmp_array = np.copy(array)
            tmp_array[np.isnan(array)] = np.inf
            tmp_array[np.isinf(tmp_array)] = np.inf  # set -inf to +inf to find the min
            vmin = np.log10(abs(tmp_array).min())
            if np.isinf(vmin):
                vmin = 0
        if np.isnan(vmax):
            tmp_array = np.copy(array)
            tmp_array[np.isnan(array)] = -1 * np.inf
            tmp_array[np.isinf(tmp_array)] = (
                -1 * np.inf
            )  # set +inf to -inf to find the max
            vmax = np.log10(abs(tmp_array).max())
        plot = axis.imshow(np.log10(abs(array)), vmin=vmin, vmax=vmax, cmap=cmap)

    if invert_yaxis and sum_axis == 0:  # detector Y is axis 0, need to be flipped
        axis = plt.gca()
        axis.invert_yaxis()
    axis.set_xlabel(hor_label)
    axis.set_ylabel(ver_label)
    plt.title(title + slice_name)
    plt.axis("scaled")
    axis.tick_params(direction=tick_direction, length=tick_length, width=tick_width)
    if pixel_spacing is not None:
        axis.xaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing[1]))
        axis.yaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing[0]))
        axis.tick_params(labelbottom=False, labelleft=False, top=True, right=True)
    if plot_colorbar:
        cbar = colorbar(plot, numticks=5)
        cbar.ax.tick_params(length=tick_length, width=tick_width)
    plt.pause(0.1)
    plt.ioff()
    return fig, axis, plot


def linecut(
    array: np.ndarray,
    indices: List[Tuple[int, int]],
    interp_order: int = 3,
) -> np.ndarray:
    """
    Linecut through a 2D or 3D array.

    The user must input indices of the starting voxel and of the end voxel.

    :param array: a numpy array
    :param indices: list of tuples of (start, stop) indices, one tuple for each
     dimension of the array. e.g [(start0, stop0), (start1, stop1)] for a 2D array
    :param interp_order: order of the spline interpolation, default is 3.
     The order has to be in the range 0-5.
    :return: a 1D array interpolated between the start and stop indices
    """
    # check parameters
    valid.valid_ndarray(array)
    if not isinstance(indices, list):
        raise TypeError(f"'indices' should be a list, got {type(indices)}")
    for _, item in enumerate(indices):
        valid.valid_container(
            item,
            container_types=tuple,
            item_types=int,
            min_included=0,
            length=2,
            name="indices",
        )

    num_points = int(
        np.sqrt(sum((val[1] - val[0]) ** 2 for _, val in enumerate(indices)))
    )

    cut = map_coordinates(
        input=array,
        coordinates=np.vstack(
            (
                [
                    np.linspace(val[0], val[1], endpoint=False, num=num_points)
                    for _, val in enumerate(indices)
                ]
            )
        ),
        order=interp_order,
    )
    return np.asarray(cut)


def loop_thru_scan(
    key,
    array,
    figure,
    scale,
    dim,
    idx,
    savedir,
    cmap=default_cmap,
    vmin=None,
    vmax=None,
):
    """
    Update the plot while removing the parasitic diffraction intensity in 3D dataset.

    :param key: the keyboard key which was pressed
    :param array: the 3D data array
    :param figure: the figure instance
    :param scale: 'linear' or 'log'
    :param dim: the axis over which the loop is performed (axis 0, 1 or 2)
    :param idx: the frame index in the current axis
    :param savedir: path of the directory for saving images
    :param cmap: colormap to be used
    :param vmin: the lower boundary for the colorbar
    :param vmax: the higher boundary for the colorbar
    :return: updated controls
    """
    valid.valid_ndarray(array, ndim=3)

    nbz, nby, nbx = array.shape
    exit_flag = False
    if dim > 2:
        raise ValueError("dim should be 0, 1 or 2")

    vmin = vmin or array.min()
    vmax = vmax or array.max()

    axis = figure.gca()
    xmin, xmax = axis.get_xlim()
    ymin, ymax = axis.get_ylim()
    if key == "u":  # show next frame
        idx = idx + 1
        if dim == 0:
            if idx > nbz - 1:
                idx = 0
        elif dim == 1:
            if idx > nby - 1:
                idx = 0
        elif dim == 2 and idx > nbx - 1:
            idx = 0

    elif key == "d":  # show previous frame
        idx = idx - 1
        if dim == 0:
            if idx < 0:
                idx = nbz - 1
        elif dim == 1:
            if idx < 0:
                idx = nby - 1
        elif dim == 2 and idx < 0:
            idx = nbx - 1

    elif key == "right":  # increase colobar max
        if scale == "linear":
            vmax = vmax * 2
        else:
            vmax = vmax + 1

    elif key == "left":  # reduce colobar max
        if scale == "linear":
            vmax = vmax / 2
        else:
            vmax = vmax - 1
        vmax = max(vmax, 1)

    elif key == "p":  # plot full image
        if dim == 0:
            xmin, xmax = -0.5, nbx - 0.5
            ymin, ymax = nby - 0.5, -0.5  # pointing down
        elif dim == 1:
            xmin, xmax = -0.5, nbx - 0.5
            ymin, ymax = nbz - 0.5, -0.5  # pointing down
        elif dim == 2:
            xmin, xmax = -0.5, nby - 0.5
            ymin, ymax = nbz - 0.5, -0.5  # pointing down

    elif key == "q":
        exit_flag = True

    elif key == "r":
        filename = "frame" + str(idx) + "_dim" + str(dim) + ".png"
        plt.savefig(savedir + filename)

    # get the images on axis
    im = axis.images
    # get and remove the existing colorbar
    cb = im[0].colorbar  # there is only one axis in the list im
    cb.remove()

    axis.cla()
    if dim == 0:
        if scale == "linear":
            plot = axis.imshow(array[idx, :, :], vmin=vmin, vmax=vmax, cmap=cmap)
        else:  # 'log'
            plot = axis.imshow(
                np.log10(array[idx, :, :]), vmin=vmin, vmax=vmax, cmap=cmap
            )
        axis.set_title(
            "Frame "
            + str(idx + 1)
            + "/"
            + str(nbz)
            + "\nq quit ; u next frame ; d previous frame ; p unzoom\n"
            "right darker ; left brighter ; r save 2D frame"
        )
        colorbar(plot, numticks=5)
    elif dim == 1:
        if scale == "linear":
            plot = axis.imshow(array[:, idx, :], vmin=vmin, vmax=vmax, cmap=cmap)
        else:  # 'log'
            plot = axis.imshow(
                np.log10(array[:, idx, :]), vmin=vmin, vmax=vmax, cmap=cmap
            )
        axis.set_title(
            "Frame "
            + str(idx + 1)
            + "/"
            + str(nby)
            + "\nq quit ; u next frame ; d previous frame ; p unzoom\n"
            "right darker ; left brighter ; r save 2D frame"
        )
        colorbar(plot, numticks=5)
    elif dim == 2:
        if scale == "linear":
            plot = axis.imshow(array[:, :, idx], vmin=vmin, vmax=vmax, cmap=cmap)
        else:  # 'log'
            plot = axis.imshow(
                np.log10(array[:, :, idx]), vmin=vmin, vmax=vmax, cmap=cmap
            )
        axis.set_title(
            "Frame "
            + str(idx + 1)
            + "/"
            + str(nbx)
            + "\nq quit ; u next frame ; d previous frame ; p unzoom\n"
            "right darker ; left brighter ; r save 2D frame"
        )
        colorbar(plot, numticks=5)
    axis.set_xlim([xmin, xmax])
    axis.set_ylim([ymin, ymax])
    plt.draw()

    return vmax, idx, exit_flag


def multislices_plot(
    array,
    sum_frames=False,
    slice_position=None,
    width_z=None,
    width_y=None,
    width_x=None,
    plot_colorbar=False,
    cmap=default_cmap,
    title="",
    scale="linear",
    vmin=np.nan,
    vmax=np.nan,
    tick_direction="out",
    tick_width=1,
    tick_length=4,
    pixel_spacing=None,
    is_orthogonal=False,
    reciprocal_space=False,
    ipynb_layout=False,
    save_as: Optional[str] = None,
    **kwargs,
):
    """
    Create a figure with three 2D imshow plots from a 3D dataset.

    :param array: 3D array of real numbers
    :param sum_frames: if True, will sum the data along the 3rd axis
    :param slice_position: tuple of three integers where to slice the 3D array
    :param width_z: zoom width along axis 0 (rocking angle), should be smaller
     than the actual data size
    :param width_y: zoom width along axis 1 (vertical), should be smaller
     than the actual data size
    :param width_x: zoom width along axis 2 (horizontal), should be smaller
     than the actual data size
    :param plot_colorbar: set it to True in,der to plot the colorbar
    :param cmap: colormap to be used
    :param title: string to include in the plot
    :param scale: 'linear', 'log'
    :param tick_direction: 'out', 'in', 'inout'
    :param tick_width: width of tickes in plots
    :param tick_length: length of tickes in plots
    :param pixel_spacing: pixel_spacing=desired tick_spacing (in nm)/voxel_size of
     the reconstruction(in nm). It can be a positive number or a tuple of 3 positive
     numbers
    :param is_orthogonal: set to True is the frame is orthogonal, False otherwise
     (detector frame) Used for plot labels.
    :param reciprocal_space: True if the data is in reciprocal space,
     False otherwise. Used for plot labels.
    :param vmin: lower boundary for the colorbar. Float or tuple of 3 floats
    :param vmax: higher boundary for the colorbar. Float or tuple of 3 floats
    :param ipynb_layout: toggle for 3 plots in a row, cleaner in an Jupyter Notebook
    :param save_as: if string, saves figure at this path
    :param kwargs:
     - 'invert_y': boolean, True to invert the vertical axis of the plot.
       Will overwrite the default behavior.

    :return: fig, (ax0, ax1, ax2, ax3), (plt0, plt1, plt2) instances
    """
    mpl.rcParams["axes.linewidth"] = tick_width  # set the linewidth globally
    ###############
    # load kwargs #
    ###############
    valid.valid_kwargs(
        kwargs=kwargs, allowed_kwargs={"invert_y"}, name="graph_utils.multislices_plot"
    )
    invert_y = kwargs.get("invert_y")

    #########################
    # check some parameters #
    #########################
    if not isinstance(sum_frames, bool):
        raise TypeError("sum_frames should be a boolean")
    if scale not in {"linear", "log"}:
        raise ValueError('scale should be either "linear" or "log"')
    valid.valid_ndarray(array, ndim=3)
    nb_dim = array.ndim

    nbz, nby, nbx = array.shape

    if isinstance(vmin, Real):
        vmin = [vmin, vmin, vmin]
    valid.valid_container(
        obj=vmin,
        container_types=(tuple, list),
        length=3,
        item_types=Real,
        name="graph_utils.multislices_plot",
    )
    min_value = vmin

    if isinstance(vmax, Real):
        vmax = [vmax, vmax, vmax]
    valid.valid_container(
        obj=vmax,
        container_types=(tuple, list),
        length=3,
        item_types=Real,
        name="graph_utils.multislices_plot",
    )
    max_value = vmax
    if any(
        v_min >= v_max
        for v_min, v_max in zip(min_value, max_value)
        if not np.isnan(v_min) and not np.isnan(v_max)
    ):
        raise ValueError("vmin should be strictly smaller than vmax")

    if not sum_frames:
        slice_position = slice_position or (int(nbz // 2), int(nby // 2), int(nbx // 2))
        valid.valid_container(
            obj=slice_position,
            container_types=(tuple, list),
            length=3,
            item_types=int,
            min_included=0,
            name="graph_utils.multislices_plot",
        )

    if isinstance(pixel_spacing, Real):
        pixel_spacing = (pixel_spacing,) * nb_dim
        valid.valid_container(
            obj=pixel_spacing,
            container_types=(tuple, list),
            length=nb_dim,
            item_types=Real,
            min_excluded=0,
            allow_none=True,
            name="graph_utils.multislices_plot",
        )
    width_z = width_z or nbz
    width_y = width_y or nby
    width_x = width_x or nbx

    invert_yaxis = bool(is_orthogonal)
    if invert_y is not None:  # override the default behavior for invert_yaxis
        invert_yaxis = invert_y

    ####################################
    # create the labels and the figure #
    ####################################
    slice_names, ver_labels, hor_labels = define_labels(
        reciprocal_space=reciprocal_space,
        is_orthogonal=is_orthogonal,
        sum_frames=sum_frames,
    )

    plt.ion()
    if ipynb_layout:
        fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(15, 4.5))
        ax3 = None
    else:
        fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2, figsize=(12, 9))

    ##########
    # axis 0 #
    ##########
    temp_array = np.copy(array)
    if not sum_frames:
        temp_array = temp_array[slice_position[0], :, :]
    else:
        temp_array = temp_array.sum(axis=0)
    # now array is 2D
    temp_array = temp_array[
        int(np.rint(nby // 2 - min(width_y, nby) // 2)) : int(
            np.rint(nby // 2 - min(width_y, nby) // 2)
        )
        + min(width_y, nby),
        int(np.rint(nbx // 2 - min(width_x, nbx) // 2)) : int(
            np.rint(nbx // 2 - min(width_x, nbx) // 2)
        )
        + min(width_x, nbx),
    ]
    if scale == "linear":
        if np.isnan(min_value[0]):
            try:
                min_value[0] = temp_array[~np.isnan(temp_array)].min()
            except ValueError:
                min_value[0] = 0
        if np.isnan(max_value[0]):
            try:
                max_value[0] = temp_array[~np.isnan(temp_array)].max()
            except ValueError:
                max_value[0] = 1
        plt0 = ax0.imshow(temp_array, vmin=min_value[0], vmax=max_value[0], cmap=cmap)
    else:  # 'log'
        if np.isnan(min_value[0]):
            try:
                min_value[0] = np.log10(abs(temp_array[~np.isnan(temp_array)]).min())
            except ValueError:
                min_value[0] = 0
            if np.isinf(min_value[0]):
                min_value[0] = 0
        if np.isnan(max_value[0]):
            try:
                max_value[0] = np.log10(abs(temp_array[~np.isnan(temp_array)]).max())
            except ValueError:
                max_value[0] = 1
        plt0 = ax0.imshow(
            np.log10(abs(temp_array)), vmin=min_value[0], vmax=max_value[0], cmap=cmap
        )

    ax0.set_xlabel(hor_labels[0])
    ax0.set_ylabel(ver_labels[0])
    ax0.set_title(title + slice_names[0])
    if invert_yaxis:  # detector Y is axis 0, need to be flipped
        ax0.invert_yaxis()
    plt.axis("scaled")
    if plot_colorbar:
        cbar = colorbar(plt0, numticks=5)
        cbar.ax.tick_params(length=tick_length, width=tick_width)
    ax0.tick_params(direction=tick_direction, length=tick_length, width=tick_width)
    if pixel_spacing is not None:
        ax0.xaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing[2]))
        ax0.yaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing[1]))
        ax0.tick_params(labelbottom=False, labelleft=False, top=True, right=True)

    ##########
    # axis 1 #
    ##########
    temp_array = np.copy(array)
    if not sum_frames:
        temp_array = temp_array[:, slice_position[1], :]
    else:
        temp_array = temp_array.sum(axis=1)
    # now array is 2D
    temp_array = temp_array[
        int(np.rint(nbz // 2 - min(width_z, nbz) // 2)) : int(
            np.rint(nbz // 2 - min(width_z, nbz) // 2)
        )
        + min(width_z, nbz),
        int(np.rint(nbx // 2 - min(width_x, nbx) // 2)) : int(
            np.rint(nbx // 2 - min(width_x, nbx) // 2)
        )
        + min(width_x, nbx),
    ]
    if scale == "linear":
        if np.isnan(min_value[1]):
            try:
                min_value[1] = temp_array[~np.isnan(temp_array)].min()
            except ValueError:
                min_value[1] = 0
        if np.isnan(max_value[1]):
            try:
                max_value[1] = temp_array[~np.isnan(temp_array)].max()
            except ValueError:
                max_value[1] = 1
        plt1 = ax1.imshow(temp_array, vmin=min_value[1], vmax=max_value[1], cmap=cmap)
    else:  # 'log'
        if np.isnan(min_value[1]):
            try:
                min_value[1] = np.log10(abs(temp_array[~np.isnan(temp_array)]).min())
            except ValueError:
                min_value[1] = 0
            if np.isinf(min_value[1]):
                min_value[1] = 0
        if np.isnan(max_value[1]):
            try:
                max_value[1] = np.log10(abs(temp_array[~np.isnan(temp_array)]).max())
            except ValueError:
                max_value[1] = 1
        plt1 = ax1.imshow(
            np.log10(abs(temp_array)), vmin=min_value[1], vmax=max_value[1], cmap=cmap
        )

    ax1.set_xlabel(hor_labels[1])
    ax1.set_ylabel(ver_labels[1])
    ax1.set_title(title + slice_names[1])
    plt.axis("scaled")
    if plot_colorbar:
        cbar = colorbar(plt1, numticks=5)
        cbar.ax.tick_params(length=tick_length, width=tick_width)
    ax1.tick_params(direction=tick_direction, length=tick_length, width=tick_width)
    if pixel_spacing is not None:
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing[2]))
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing[0]))
        ax1.tick_params(labelbottom=False, labelleft=False, top=True, right=True)

    ##########
    # axis 2 #
    ##########
    temp_array = np.copy(array)
    if not sum_frames:
        temp_array = temp_array[:, :, slice_position[2]]
    else:
        temp_array = temp_array.sum(axis=2)
    # now array is 2D
    temp_array = temp_array[
        int(np.rint(nbz // 2 - min(width_z, nbz) // 2)) : int(
            np.rint(nbz // 2 - min(width_z, nbz) // 2)
        )
        + min(width_z, nbz),
        int(np.rint(nby // 2 - min(width_y, nby) // 2)) : int(
            np.rint(nby // 2 - min(width_y, nby) // 2)
        )
        + min(width_y, nby),
    ]
    if scale == "linear":
        if np.isnan(min_value[2]):
            try:
                min_value[2] = temp_array[~np.isnan(temp_array)].min()
            except ValueError:
                min_value[2] = 0
        if np.isnan(max_value[2]):
            try:
                max_value[2] = temp_array[~np.isnan(temp_array)].max()
            except ValueError:
                max_value[2] = 1
        plt2 = ax2.imshow(temp_array, vmin=min_value[2], vmax=max_value[2], cmap=cmap)
    else:  # 'log'
        if np.isnan(min_value[2]):
            try:
                min_value[2] = np.log10(abs(temp_array[~np.isnan(temp_array)]).min())
            except ValueError:
                min_value[2] = 0
            if np.isinf(min_value[2]):
                min_value[2] = 0
        if np.isnan(max_value[2]):
            try:
                max_value[2] = np.log10(abs(temp_array[~np.isnan(temp_array)]).max())
            except ValueError:
                max_value[2] = 1
        plt2 = ax2.imshow(
            np.log10(abs(temp_array)), vmin=min_value[2], vmax=max_value[2], cmap=cmap
        )

    ax2.set_xlabel(hor_labels[2])
    ax2.set_ylabel(ver_labels[2])
    ax2.set_title(title + slice_names[2])
    plt.axis("scaled")

    if plot_colorbar:
        cbar = colorbar(plt2, numticks=5)
        cbar.ax.tick_params(length=tick_length, width=tick_width)
    ax2.tick_params(direction=tick_direction, length=tick_length, width=tick_width)
    if pixel_spacing is not None:
        ax2.xaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing[1]))
        ax2.yaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing[0]))
        ax2.tick_params(labelbottom=False, labelleft=False, top=True, right=True)

    ##########
    # axis 3 #
    ##########
    if not ipynb_layout and ax3 is not None:
        # hide axis 3
        ax3.set_visible(False)

    plt.tight_layout()  # avoids the overlap of subplots with axes labels
    plt.pause(0.1)
    plt.ioff()

    if isinstance(save_as, str):
        pathlib.Path(save_as).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_as)

    if ipynb_layout:
        return fig, (ax0, ax1, ax2), (plt0, plt1, plt2)
    return fig, (ax0, ax1, ax2, ax3), (plt0, plt1, plt2)


def plot_linecut(linecuts: Dict, filename: Optional[str] = None) -> None:
    """
    Plot linecuts and optionally corresponding fits.

    linecuts = {
        'dimension_0': {
            'linecut': np.ndarray (2, M),
            'derivative_0': np.ndarray (2, N),
            'derivative_1': np.ndarray (2, O),
            'fit_0': np.ndarray (2, P),
            'fit_1': np.ndarray (2, P),
        },
        'dimension_1': {...}
        ...
    }

    :param linecuts: a dictionary containing cuts, with keys 'dim_0', 'dim_1', ...
    :param filename: str, the figure will be saved there
    """
    if filename is not None and not isinstance(filename, str):
        raise TypeError(f"filename should be a string, got {type(filename)}")
    if not isinstance(linecuts, dict):
        raise TypeError("expected a dictionary, got {type(linecuts)}")
    labels = {"dimension_0": "Z", "dimension_1": "Y", "dimension_0": "X"}
    # plot the linecuts
    fig, axes = plt.subplots(nrows=len(linecuts), ncols=1, figsize=(12, 9))
    for idx, key in enumerate(linecuts.keys()):
        axes[idx].plot(linecuts[key]["linecut"][0], linecuts[key]["linecut"][1], ".-b")
        axes[idx].set_xlabel(labels.get(key, key))
        axes[idx].set_ylabel("linecut")

    plt.tight_layout()  # avoids the overlap of subplots with axes labels
    plt.pause(0.1)
    plt.ioff()

    if filename:
        fig.savefig(filename)

    nb_deriv = (len(linecuts["dimension_0"]) - 1) // 2

    # plot the derivatives and the fits
    fig, axes = plt.subplots(nrows=len(linecuts), ncols=nb_deriv, figsize=(12, 9))
    for idx, key in enumerate(linecuts.keys()):
        for idy in range(nb_deriv):
            (line1,) = axes[idx][idy].plot(
                linecuts[key][f"derivative_{idy}"][0],
                linecuts[key][f"derivative_{idy}"][1],
                ".b",
                label="derivative",
            )
            (line2,) = axes[idx][idy].plot(
                linecuts[key][f"fit_{idy}"][0],
                linecuts[key][f"fit_{idy}"][1],
                "-r",
                label="fit",
            )
            axes[idx][idy].set_xlabel(labels.get(key, key))
            axes[idx][idy].legend(handles=[line1, line2])

    plt.tight_layout()  # avoids the overlap of subplots with axes labels
    plt.pause(0.1)
    plt.ioff()

    if filename:
        fig.savefig(filename)


def plot_3dmesh(
    vertices, faces, data_shape, title="Mesh - z axis flipped because of CXI convention"
):
    """
    Plot a 3D mesh defined by its vertices and faces.

    :param vertices: n*3 ndarray of n vertices defined by 3 positions
    :param faces: m*3 ndarray of m faces defined by 3 indices of vertices
    :param data_shape: tuple corresponding to the 3d data shape
    :param title: title for the plot
    :return: figure and axe instances
    """
    plt.ion()
    fig = plt.figure(figsize=(10, 10))
    ax0 = Axes3D(fig)
    mymesh = Poly3DCollection(vertices[faces])
    mymesh.set_edgecolor("k")
    ax0.add_collection3d(mymesh)
    ax0.set_xlim(0, data_shape[0])
    ax0.set_xlabel("Z")
    ax0.set_ylim(0, data_shape[1])
    ax0.set_ylabel("Y")
    ax0.set_zlim(0, data_shape[2])
    ax0.set_zlabel("X")
    plt.title(title)

    plt.pause(0.1)
    plt.ioff()
    return fig, ax0


def savefig(
    savedir,
    figure,
    axes,
    xlabels="",
    ylabels="",
    titles="",
    filename="",
    tick_direction="out",
    tick_width=2,
    tick_length=10,
    tick_labelsize=16,
    legend_labelsize=16,
    label_size=20,
    title_size=20,
    only_labels=False,
    **kwargs,
):
    """
    Save a template figures for publication, without and with labels.

    :param savedir: str, the directory where to save the figures
    :param figure: a matplotlib figure instance
    :param axes: a matplotlib axis or a tuple of axes
    :param xlabels: str, horizontal labels (one per axis)
    :param ylabels: str, vertical labels (one per axis)
    :param titles: str, title (one per axis)
    :param filename: name of the file for saving the figure
    :param tick_direction: 'in', 'out' or 'inout'
    :param tick_width: tick width in points
    :param tick_length: tick length in points
    :param tick_labelsize: label size in points of tick labels
    :param legend_labelsize: label size in points of the legend
    :param label_size: label size in points of axis labels
    :param title_size: label size in points of titles
    :param only_labels: bool, if True only the figure with all labels will be saved
    :param kwargs:
     - 'bottom', 'top', 'left', 'right': bool, whether to draw the respective ticks.
     - 'labelbottom', 'labeltop', 'labelleft', 'labelright': bool, whether to draw
       the respective tick labels.
     - 'legend': bool, wheter to show the legend or not
     - 'text': dict, a dictionnary of dictionnaries containing the parameters for
       matplotlib.pyplot.text function e.g. {0: {'x': 0.4, 'y': 0.4, 's': 'test',
       'fontsize': 12}, 1:{'x': 0.4, 'y': 0.5, 's': 'res', 'fontsize': 12}}

    """
    #########################
    # check and load kwargs #
    #########################
    valid.valid_kwargs(
        kwargs=kwargs,
        allowed_kwargs={
            "labelbottom",
            "labeltop",
            "labelleft",
            "labelright",
            "bottom",
            "top",
            "left",
            "right",
            "legend",
            "text",
        },
        name="kwargs",
    )
    labelbottom = kwargs.get("labelbottom", True)
    labeltop = kwargs.get("labeltop", False)
    labelleft = kwargs.get("labelleft", True)
    labelright = kwargs.get("labelright", False)
    bottom = kwargs.get("bottom", True)
    top = kwargs.get("top", False)
    left = kwargs.get("left", True)
    right = kwargs.get("right", False)
    legend = kwargs.get("legend", False)
    text = kwargs.get("text")

    ####################
    # check parameters #
    ####################
    fname = "savefig"
    valid.valid_container(savedir, container_types=str, min_length=1, name=fname)
    if not isinstance(figure, mpl.figure.Figure):
        raise TypeError("figure should be a matplotlib Figure")
    if isinstance(axes, mpl.axes.Axes):
        axes = (axes,)
    valid.valid_container(axes, container_types=(tuple, list), min_length=1, name=fname)
    for ax in axes:
        if not isinstance(ax, mpl.axes.Axes):
            raise TypeError("axes should be a tuple of matplotlib axes")
    nb_axes = len(axes)
    if isinstance(labelbottom, bool):
        labelbottom = (labelbottom,) * nb_axes
    valid.valid_container(
        labelbottom,
        container_types=(tuple, list),
        item_types=bool,
        length=nb_axes,
        name=fname,
    )
    if isinstance(labeltop, bool):
        labeltop = (labeltop,) * nb_axes
    valid.valid_container(
        labeltop,
        container_types=(tuple, list),
        item_types=bool,
        length=nb_axes,
        name=fname,
    )
    if isinstance(labelleft, bool):
        labelleft = (labelleft,) * nb_axes
    valid.valid_container(
        labelleft,
        container_types=(tuple, list),
        item_types=bool,
        length=nb_axes,
        name=fname,
    )
    if isinstance(labelright, bool):
        labelright = (labelright,) * nb_axes
    valid.valid_container(
        labelright,
        container_types=(tuple, list),
        item_types=bool,
        length=nb_axes,
        name=fname,
    )
    if isinstance(bottom, bool):
        bottom = (bottom,) * nb_axes
    valid.valid_container(
        bottom,
        container_types=(tuple, list),
        item_types=bool,
        length=nb_axes,
        name=fname,
    )
    if isinstance(top, bool):
        top = (top,) * nb_axes
    valid.valid_container(
        top, container_types=(tuple, list), item_types=bool, length=nb_axes, name=fname
    )
    if isinstance(left, bool):
        left = (left,) * nb_axes
    valid.valid_container(
        left, container_types=(tuple, list), item_types=bool, length=nb_axes, name=fname
    )
    if isinstance(right, bool):
        right = (right,) * nb_axes
    valid.valid_container(
        right,
        container_types=(tuple, list),
        item_types=bool,
        length=nb_axes,
        name=fname,
    )
    if isinstance(legend, bool):
        legend = (legend,) * nb_axes
    valid.valid_container(
        legend,
        container_types=(tuple, list),
        item_types=bool,
        length=nb_axes,
        name=fname,
    )
    if isinstance(xlabels, str):
        xlabels = (xlabels,) * nb_axes
    valid.valid_container(
        xlabels,
        container_types=(tuple, list),
        item_types=str,
        length=nb_axes,
        name=fname,
    )
    if isinstance(ylabels, str):
        ylabels = (ylabels,) * nb_axes
    valid.valid_container(
        ylabels,
        container_types=(tuple, list),
        item_types=str,
        length=nb_axes,
        name=fname,
    )
    if isinstance(titles, str):
        titles = (titles,) * nb_axes
    valid.valid_container(
        titles,
        container_types=(tuple, list),
        item_types=str,
        length=nb_axes,
        name=fname,
    )
    valid.valid_container(filename, container_types=str, name=fname)
    filename = filename.replace(
        ".png", ""
    )  # in case the user put the extension in the filename
    if tick_direction not in {"in", "out", "inout"}:
        raise ValueError(
            "Invalid value {tick_direction} for tick_direction,"
            " allowed are 'in', 'out', 'inout'"
        )
    valid.valid_item(tick_width, allowed_types=int, min_excluded=0, name=fname)
    valid.valid_item(tick_length, allowed_types=int, min_excluded=0, name=fname)
    valid.valid_item(tick_labelsize, allowed_types=int, min_excluded=0, name=fname)
    valid.valid_item(legend_labelsize, allowed_types=int, min_excluded=0, name=fname)
    valid.valid_item(label_size, allowed_types=int, min_excluded=0, name=fname)
    valid.valid_item(title_size, allowed_types=int, min_excluded=0, name=fname)
    valid.valid_container(
        text,
        container_types=dict,
        item_types=int,
        allow_none=True,
        min_length=1,
        min_included=0,
        name=fname,
    )
    valid.valid_item(only_labels, allowed_types=bool, name=fname)

    #########################
    # plot and save figures #
    #########################
    xlims = []
    ylims = []
    xlocs = []
    ylocs = []
    plt.ion()
    for idx, ax in enumerate(axes):
        ax.tick_params(
            labelbottom=False,
            labelleft=False,
            labelright=False,
            labeltop=False,
            bottom=bottom[idx],
            top=top[idx],
            left=left[idx],
            right=right[idx],
            direction=tick_direction,
            length=tick_length,
            width=tick_width,
            labelsize=tick_labelsize,
        )
        ax.set_xlabel(xlabels[idx], fontsize=label_size, visible=False)
        ax.set_ylabel(ylabels[idx], fontsize=label_size, visible=False)
        ax.set_title(titles[idx], fontsize=title_size, visible=False)
        ax.spines["right"].set_linewidth(tick_width)
        ax.spines["left"].set_linewidth(tick_width)
        ax.spines["top"].set_linewidth(tick_width)
        ax.spines["bottom"].set_linewidth(tick_width)
        try:  # Check if there is a colorbar
            cbar = ax.images[0].colorbar
            cbar.ax.tick_params(
                labelright=False,
                length=tick_length,
                width=tick_width,
                labelsize=tick_labelsize,
            )
            cbar.outline.set_linewidth(tick_width)
        except IndexError:
            cbar = None
        xlims.append(ax.get_xlim())
        ylims.append(ax.get_ylim())
        xlocs.append(ax.xaxis.get_ticklocs())
        ylocs.append(ax.yaxis.get_ticklocs())

    if not only_labels:
        figure.savefig(savedir + filename + ".png")

    for idx, ax in enumerate(axes):
        ax.tick_params(
            labelbottom=labelbottom[idx],
            labelleft=labelleft[idx],
            labelright=labelright[idx],
            labeltop=labeltop[idx],
            axis="both",
            which="major",
            labelsize=label_size,
        )
        ax.set_xlabel(xlabels[idx], fontsize=label_size, visible=True)
        ax.set_ylabel(ylabels[idx], fontsize=label_size, visible=True)
        ax.set_title(titles[idx], fontsize=title_size, visible=True)
        if legend[idx]:
            ax.legend(fontsize=legend_labelsize)
        ax.set_xticks(xlocs[idx])
        ax.set_yticks(ylocs[idx])
        ax.set_xlim(left=xlims[idx][0], right=xlims[idx][1])
        ax.set_ylim(bottom=ylims[idx][0], top=ylims[idx][1])

    if text is not None:
        for _, value in text.items():
            figure.text(**value)
    if cbar is not None:
        cbar.ax.tick_params(labelright=True)

    figure.tight_layout()
    figure.savefig(savedir + filename + "_labels.png")
    plt.ioff()


def save_to_vti(
    filename,
    voxel_size,
    tuple_array,
    tuple_fieldnames,
    origin=(0, 0, 0),
    amplitude_threshold=0.01,
):
    """
    Save arrays defined by their name in a single vti file.

    Paraview expects data in an orthonormal basis (x,y,z). For BCDI data in the .cxi
    convention (hence: z, y,x) it is necessary to flip the last axis. The data sent
    to Paraview will be in the orthonormal frame (z,y,-x), therefore Paraview_x is z
    (downstream), Paraview_y is y (vertical up), Paraview_z is -x (inboard) of the
    .cxi convention.

    :param filename: the file name of the vti file
    :param voxel_size: tuple (voxel_size_axis0, voxel_size_axis1, voxel_size_axis2)
    :param tuple_array: tuple of arrays of the same dimension
    :param tuple_fieldnames: tuple of strings for the field names, same number of
     elements as tuple_array
    :param origin: tuple of points for vtk SetOrigin()
    :param amplitude_threshold: lower threshold for saving the reconstruction
     modulus (save memory space)
    :return: nothing
    """
    import vtk
    from vtk.util import numpy_support

    #########################
    # check some parameters #
    #########################
    valid.valid_container(
        obj=voxel_size,
        container_types=(tuple, list),
        length=3,
        item_types=Real,
        min_excluded=0,
        name="voxel_size",
    )

    if isinstance(tuple_array, np.ndarray):
        tuple_array = (tuple_array,)
    valid.valid_ndarray(tuple_array, ndim=3)
    nb_arrays = len(tuple_array)
    nbz, nby, nbx = tuple_array[0].shape

    if isinstance(tuple_fieldnames, str):
        tuple_fieldnames = (tuple_fieldnames,)
    valid.valid_container(
        obj=tuple_fieldnames,
        container_types=(tuple, list),
        length=nb_arrays,
        item_types=str,
        name="tuple_fieldnames",
    )

    #############################
    # initialize the VTK object #
    #############################
    image_data = vtk.vtkImageData()
    image_data.SetOrigin(origin[0], origin[1], origin[2])
    image_data.SetSpacing(voxel_size[0], voxel_size[1], voxel_size[2])
    image_data.SetExtent(0, nbz - 1, 0, nby - 1, 0, nbx - 1)

    #######################################
    # check if one of the fields in 'amp' #
    #######################################
    # it will use the thresholded normalized 'amp' as support
    # when saving other fields, in order to save disk space
    try:
        index_first = tuple_fieldnames.index("amp")
        first_array = tuple_array[index_first]
        first_array = first_array / first_array.max()
        first_array[
            first_array < amplitude_threshold
        ] = 0  # theshold low amplitude values in order to save disk space
        is_amp = True
    except ValueError:
        print('"amp" not in fieldnames, will save arrays without thresholding')
        index_first = 0
        first_array = tuple_array[0]
        is_amp = False

    first_arr = np.transpose(np.flip(first_array, 2)).reshape(first_array.size)
    first_arr = numpy_support.numpy_to_vtk(first_arr)
    pd = image_data.GetPointData()
    pd.SetScalars(first_arr)
    pd.GetArray(0).SetName(tuple_fieldnames[index_first])
    counter = 1
    for idx in range(nb_arrays):
        if idx == index_first:
            continue
        temp_array = tuple_array[idx]
        if is_amp:
            temp_array[
                first_array == 0
            ] = 0  # use the thresholded amplitude as a support
            # in order to save disk space
        temp_array = np.transpose(np.flip(temp_array, 2)).reshape(temp_array.size)
        temp_array = numpy_support.numpy_to_vtk(temp_array)
        pd.AddArray(temp_array)
        pd.GetArray(counter).SetName(tuple_fieldnames[idx])
        pd.Update()
        counter = counter + 1

    # export data to file
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(image_data)
    writer.Write()


def scatter_plot(array, labels, markersize=4, markercolor="b", title=""):
    """
    2D or 3D Scatter plot of a 2D ndarray.

    :param array: 2D ndarray, the number of columns is the number of dimensions
     of the scatter plot (2 or 3)
    :param labels: tuple of string labels (length = number of columns in array)
    :param markersize: number corresponding to the marker size
    :param markercolor: string corresponding to the marker color
    :param title: string, title for the scatter plot
    :return: figure, axes instances
    """
    valid.valid_ndarray(array, ndim=2, fix_shape=False)
    ndim = array.shape[1]
    if isinstance(labels, tuple):
        if len(labels) != ndim:
            raise ValueError(
                "len(labels) is different from the number of columns in the array"
            )
    else:  # it is a string or a number
        labels = (labels,) * ndim

    plt.ion()
    fig = plt.figure()

    if ndim == 2:
        ax = plt.subplot(111)
        ax.scatter(array[:, 0], array[:, 1], s=markersize, color=markercolor)
        plt.title(title)
        ax.set_xlabel(
            labels[0]
        )  # first dimension is x for scatter plots, but z for NEXUS convention
        ax.set_ylabel(labels[1])
        plt.pause(0.1)
    elif ndim == 3:
        ax = plt.subplot(111, projection="3d")
        ax.scatter(
            array[:, 0], array[:, 1], array[:, 2], s=markersize, color=markercolor
        )
        plt.title(title)
        ax.set_xlabel(
            labels[0]
        )  # first dimension is x for scatter plots, but z for NEXUS convention
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
    else:
        raise ValueError("There should be 2 or 3 columns in the array")
    if ndim == 2:
        plt.axis("scaled")
    plt.pause(0.1)
    plt.ioff()
    return fig, ax


def scatter_plot_overlaid(arrays, markersizes, markercolors, labels, title=""):
    """
    Overlaid scatter plot of 2D ndarrays having the same number of columns.

    :param arrays: tuple of 2D ndarrays, the number of columns is the number
     of dimensions of the scatter plot (2 or 3)
    :param markersizes: tuple of numbers corresponding to the marker sizes
     (length = number of arrays)
    :param markercolors: tuple of strings corresponding to the marker color
     (length = number of arrays)
    :param labels: tuple of string labels (length = number of columns in arrays)
    :param title: string, title for the scatter plot
    :return: figure, axes instances
    """
    if isinstance(arrays, np.ndarray):
        fig, ax = scatter_plot(
            array=arrays,
            markersize=markersizes,
            markercolor=markercolors,
            labels=labels,
            title=title,
        )
        return fig, ax

    valid.valid_ndarray(arrays, ndim=2, fix_shape=False)

    ndim = arrays[0].shape[1]
    nb_arrays = len(arrays)

    if isinstance(labels, tuple):
        if len(labels) != ndim:
            raise ValueError(
                "len(labels) is different from the number of columns in the array"
            )
    else:  # it is a string or a number
        labels = (labels,) * ndim
    try:
        if len(markersizes) != nb_arrays:
            raise ValueError("len(markersizes) is different from the number of arrays")
    except TypeError:  # it is a number
        markersizes = (markersizes,) * nb_arrays
    if isinstance(markercolors, tuple):
        if len(markercolors) != nb_arrays:
            raise ValueError("len(markercolors) is different from the number of arrays")
    else:  # it is a string or a number
        markercolors = (markercolors,) * nb_arrays

    plt.ion()
    fig = plt.figure()
    if ndim == 2:
        ax = plt.subplot(111)
    elif ndim == 3:
        ax = plt.subplot(111, projection="3d")
    else:
        raise ValueError("There should be 2 or 3 columns in the array")

    for idx in range(nb_arrays):
        array = arrays[idx]
        if array.shape[1] != ndim:
            raise ValueError("All arrays should have the same number of columns")

        if ndim == 2:
            ax.scatter(
                array[:, 0], array[:, 1], s=markersizes[idx], color=markercolors[idx]
            )
        else:  # 3D
            ax.scatter(
                array[:, 0],
                array[:, 1],
                array[:, 2],
                s=markersizes[idx],
                color=markercolors[idx],
            )

    plt.title(title)
    if ndim == 2:
        ax.set_xlabel(
            labels[0]
        )  # first dimension is x for scatter plots, but z for NEXUS convention
        ax.set_ylabel(labels[1])
    else:
        ax.set_xlabel(
            labels[0]
        )  # first dimension is x for scatter plots, but z for NEXUS convention
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
    if ndim == 2:
        plt.axis("scaled")
    plt.pause(0.1)
    plt.ioff()
    return fig, ax


def scatter_stereographic(
    euclidian_u,
    euclidian_v,
    color,
    title="",
    max_angle=95,
    cmap=default_cmap,
    uv_labels=("", ""),
):
    """
    Plot the stereographic projection of the real scattered positions of data points.

    :param euclidian_u: flattened array, normalized Euclidian metric coordinates
     (points can be not on a regular grid)
    :param euclidian_v: flattened array, normalized Euclidian metric coordinates
     (points can be not on a regular grid)
    :param color: flattened array, intensity of density kernel estimation at radius_mean
    :param title: title for the stereographic plot
    :param max_angle: maximum angle in degrees of the stereographic projection
     (should be larger than 90)
    :param cmap: colormap to be used
    :param uv_labels: tuple of strings, labels for the u axis and the v axis,
     respectively
    :return: figure and axe instances
    """
    fig, ax0 = plt.subplots(nrows=1, ncols=1)
    plt0 = ax0.scatter(
        euclidian_u,
        euclidian_v,
        s=6,
        c=color,
        cmap=cmap,
        norm=colors.LogNorm(
            vmin=max(color[~np.isnan(color)].min(), 1),
            vmax=color[~np.isnan(color)].max(),
        ),
    )
    circle = patches.Circle((0, 0), 90, color="k", fill=False, linewidth=1.5)
    ax0.add_artist(circle)
    ax0.axis("scaled")
    ax0.set_xlim(-max_angle, max_angle)
    ax0.set_ylim(-max_angle, max_angle)
    ax0.set_xlabel("u " + uv_labels[0])
    ax0.set_ylabel("v " + uv_labels[1])
    ax0.set_title(title)
    colorbar(plt0, scale="log", numticks=5)
    plt.pause(0.1)
    plt.ioff()
    return fig, ax0


def update_aliens(
    key,
    pix,
    piy,
    original_data,
    original_mask,
    updated_data,
    updated_mask,
    figure,
    width,
    dim,
    idx,
    vmax,
    vmin=0,
    invert_yaxis=False,
):
    """
    Update the plot while removing the parasitic diffraction intensity in 3D dataset.

    :param key: the keyboard key which was pressed
    :param pix: the x value of the mouse pointer
    :param piy: the y value of the mouse pointer
    :param original_data: the 3D data array before masking aliens
    :param original_mask: the 3D mask array before masking aliens
    :param updated_data: the current 3D data array
    :param updated_mask: the current 3D mask array
    :param figure: the figure instance
    :param width: the half_width of the masking window
    :param dim: the axis currently under review (axis 0, 1 or 2)
    :param idx: the frame index in the current axis
    :param vmax: the higher boundary for the colorbar
    :param vmin: the lower boundary for the colorbar
    :param invert_yaxis: True to invert the y axis of imshow plots
    :return: updated data, mask and controls
    """
    # check some parameters
    valid.valid_ndarray(
        arrays=(original_data, updated_data, original_mask, updated_mask),
        ndim=3,
    )
    if dim not in {0, 1, 2}:
        raise ValueError("dim should be 0, 1 or 2")

    # process arrays
    nbz, nby, nbx = original_data.shape
    stop_masking = False
    if dim == 0:
        current_nby = nby
        current_nbx = nbx
    elif dim == 1:
        current_nby = nbz
        current_nbx = nbx
    else:  # dim = 2
        current_nby = nbz
        current_nbx = nby

    axs = figure.gca()
    xmin, xmax = axs.get_xlim()
    ymin, ymax = axs.get_ylim()
    if key == "u":  # show next frame
        idx = idx + 1
        if dim == 0:
            if idx > nbz - 1:
                idx = 0
        elif dim == 1:
            if idx > nby - 1:
                idx = 0
        else:  # dim=2
            if idx > nbx - 1:
                idx = 0

    elif key == "d":  # show previous frame
        idx = idx - 1
        if dim == 0:
            if idx < 0:
                idx = nbz - 1
        elif dim == 1:
            if idx < 0:
                idx = nby - 1
        else:  # dim=2
            if idx < 0:
                idx = nbx - 1

    elif key == "up":
        width = width + 1

    elif key == "down":
        width = width - 1
        width = max(width, 0)

    elif key == "right":  # increase colobar max
        vmax = vmax * 2

    elif key == "left":  # reduce colobar max
        vmax = vmax / 2
        vmax = max(vmax, 1)

    elif key == "m":  # mask intensities
        skip = False

        # check if the masking window fit in the data range
        # (vertical axis of the 2D plot)
        if (piy - width) < 0:
            starty = min(0, piy + width)
            if starty < 0:
                skip = True
        else:
            starty = piy - width
        if (piy + width) >= current_nby:
            stopy = max(current_nby, piy - width)
            if stopy > current_nby:
                skip = True
        else:
            stopy = piy + width + 1

        # check if the masking window fit in the data range
        # (horizontal axis of the 2D plot)
        if (pix - width) < 0:
            startx = min(0, pix + width)
            if startx < 0:
                skip = True
        else:
            startx = pix - width
        if (pix + width) >= current_nbx:
            stopx = max(current_nbx, pix - width)
            if stopx > current_nbx:
                skip = True
        else:
            stopx = pix + width + 1

        if not skip:
            if dim == 0:
                updated_data[idx, starty:stopy, startx:stopx] = 0
                updated_mask[idx, starty:stopy, startx:stopx] = 1
            elif dim == 1:
                updated_data[starty:stopy, idx, startx:stopx] = 0
                updated_mask[starty:stopy, idx, startx:stopx] = 1
            else:  # dim=2
                updated_data[starty:stopy, startx:stopx, idx] = 0
                updated_mask[starty:stopy, startx:stopx, idx] = 1

    elif key == "b":  # back to measured intensities
        skip = False

        # check if the masking window fit in the data range
        # (vertical axis of the 2D plot)
        if (piy - width) < 0:
            starty = min(0, piy + width)
            if starty < 0:
                skip = True
        else:
            starty = piy - width
        if (piy + width) >= current_nby:
            stopy = max(current_nby, piy - width)
            if stopy > current_nby:
                skip = True
        else:
            stopy = piy + width + 1

        # check if the masking window fit in the data range
        # (horizontal axis of the 2D plot)
        if (pix - width) < 0:
            startx = min(0, pix + width)
            if startx < 0:
                skip = True
        else:
            startx = pix - width
        if (pix + width) >= current_nbx:
            stopx = max(current_nbx, pix - width)
            if stopx > current_nbx:
                skip = True
        else:
            stopx = pix + width + 1

        if not skip:
            if dim == 0:
                updated_data[idx, starty:stopy, startx:stopx] = original_data[
                    idx, starty:stopy, startx:stopx
                ]
                updated_mask[idx, starty:stopy, startx:stopx] = original_mask[
                    idx, starty:stopy, startx:stopx
                ]

            elif dim == 1:
                updated_data[starty:stopy, idx, startx:stopx] = original_data[
                    starty:stopy, idx, startx:stopx
                ]
                updated_mask[starty:stopy, idx, startx:stopx] = original_mask[
                    starty:stopy, idx, startx:stopx
                ]
            else:  # dim=2
                updated_data[starty:stopy, startx:stopx, idx] = original_data[
                    starty:stopy, startx:stopx, idx
                ]
                updated_mask[starty:stopy, startx:stopx, idx] = original_mask[
                    starty:stopy, startx:stopx, idx
                ]

    elif key in {"p", "a"}:  # plot full image or restart masking
        if dim == 0:
            xmin, xmax = -0.5, nbx - 0.5
            if invert_yaxis:
                ymin, ymax = -0.5, nby - 0.5  # pointing up
            else:
                ymin, ymax = nby - 0.5, -0.5  # pointing down
        elif dim == 1:
            xmin, xmax = -0.5, nbx - 0.5
            ymin, ymax = nbz - 0.5, -0.5  # pointing down
        else:  # dim=2
            xmin, xmax = -0.5, nby - 0.5
            ymin, ymax = nbz - 0.5, -0.5  # pointing down
        if key == "a":  # restart masking
            updated_data[:] = original_data[:]
            updated_mask[:] = original_mask[:]

    elif key == "q":
        stop_masking = True

    else:
        return updated_data, updated_mask, width, vmax, idx, stop_masking

    axs.cla()
    if dim == 0:
        axs.imshow(updated_data[idx, :, :], vmin=vmin, vmax=vmax)
        axs.set_title(
            "XY - Frame " + str(idx + 1) + "/" + str(nbz) + "\n"
            "m mask ; b unmask ; q quit ; u next frame ; d previous frame\n"
            "up larger ; down smaller ; right darker ; left brighter"
        )
    elif dim == 1:
        axs.imshow(updated_data[:, idx, :], vmin=vmin, vmax=vmax)
        axs.set_title(
            "XZ - Frame " + str(idx + 1) + "/" + str(nby) + "\n"
            "m mask ; b unmask ; q quit ; u next frame ; d previous frame\n"
            "up larger ; down smaller ; right darker ; left brighter"
        )
    elif dim == 2:
        axs.imshow(updated_data[:, :, idx], vmin=vmin, vmax=vmax)
        axs.set_title(
            "YZ - Frame " + str(idx + 1) + "/" + str(nbx) + "\n"
            "m mask ; b unmask ; q quit ; u next frame ; d previous frame\n"
            "up larger ; down smaller ; right darker ; left brighter"
        )
    if invert_yaxis:
        axs.invert_yaxis()
    axs.set_xlim([xmin, xmax])
    axs.set_ylim([ymin, ymax])
    plt.draw()

    return updated_data, updated_mask, width, vmax, idx, stop_masking


def update_aliens_combined(
    key,
    pix,
    piy,
    original_data,
    original_mask,
    updated_data,
    updated_mask,
    axes,
    width,
    dim,
    frame_index,
    vmax,
    vmin=0,
    cmap=default_cmap,
    invert_yaxis=False,
):
    """
    Update the plot while removing the parasitic diffraction intensity in 3D dataset.

    :param key: the keyboard key which was pressed
    :param pix: the x value of the mouse pointer
    :param piy: the y value of the mouse pointer
    :param original_data: the 3D data array before masking aliens
    :param original_mask: the 3D mask array before masking aliens
    :param updated_data: the current 3D data array
    :param updated_mask: the current 3D mask array
    :param axes: tuple of the 4 axes instances in a plt.subplots(nrows=2, ncols=2)
    :param width: the half_width of the masking window
    :param dim: the axis currently under review (axis 0, 1 or 2)
    :param frame_index: list of 3 frame indices (one per axis)
    :param vmax: the higher boundary for the colorbar
    :param vmin: the lower boundary for the colorbar
    :param cmap: colormap to be used
    :param invert_yaxis: True to invert the y axis of imshow plots
    :return: updated data, mask (-1 filled, 0 non masked, 1 masked voxel) and controls
    """
    # check some parameters
    valid.valid_ndarray(
        arrays=(original_data, updated_data, original_mask, updated_mask),
        ndim=3,
    )
    if dim not in {0, 1, 2}:
        raise ValueError("dim should be 0, 1 or 2")

    # process arrays
    nbz, nby, nbx = original_data.shape
    stop_masking = False
    if dim == 0:
        current_nby = nby
        current_nbx = nbx
    elif dim == 1:
        current_nby = nbz
        current_nbx = nbx
    else:  # dim = 2
        current_nby = nbz
        current_nbx = nby

    xmin0, xmax0 = axes[0].get_xlim()
    ymin0, ymax0 = axes[0].get_ylim()
    xmin1, xmax1 = axes[1].get_xlim()
    ymin1, ymax1 = axes[1].get_ylim()
    xmin2, xmax2 = axes[2].get_xlim()
    ymin2, ymax2 = axes[2].get_ylim()

    if key == "u":  # show next frame
        if dim == 0:
            frame_index[0] = frame_index[0] + 1
            if frame_index[0] > nbz - 1:
                frame_index[0] = 0
        elif dim == 1:
            frame_index[1] = frame_index[1] + 1
            if frame_index[1] > nby - 1:
                frame_index[1] = 0
        else:  # dim=2
            frame_index[2] = frame_index[2] + 1
            if frame_index[2] > nbx - 1:
                frame_index[2] = 0

    elif key == "d":  # show previous frame
        if dim == 0:
            frame_index[0] = frame_index[0] - 1
            if frame_index[0] < 0:
                frame_index[0] = nbz - 1
        elif dim == 1:
            frame_index[1] = frame_index[1] - 1
            if frame_index[1] < 0:
                frame_index[1] = nby - 1
        else:  # dim=2
            frame_index[2] = frame_index[2] - 1
            if frame_index[2] < 0:
                frame_index[2] = nbx - 1

    elif key == "up":
        width = width + 1

    elif key == "down":
        width = width - 1
        width = max(width, 0)

    elif key == "right":  # increase colobar max
        vmax = vmax * 2

    elif key == "left":  # reduce colobar max
        vmax = vmax / 2
        vmax = max(vmax, 1)

    elif key == "m":  # mask intensities
        skip = False

        # check if the masking window fit in the data range
        # (vertical axis of the 2D plot)
        if (piy - width) < 0:
            starty = min(0, piy + width)
            if starty < 0:
                skip = True
        else:
            starty = piy - width
        if (piy + width) >= current_nby:
            stopy = max(current_nby, piy - width)
            if stopy > current_nby:
                skip = True
        else:
            stopy = piy + width + 1

        # check if the masking window fit in the data range
        # (horizontal axis of the 2D plot)
        if (pix - width) < 0:
            startx = min(0, pix + width)
            if startx < 0:
                skip = True
        else:
            startx = pix - width
        if (pix + width) >= current_nbx:
            stopx = max(current_nbx, pix - width)
            if stopx > current_nbx:
                skip = True
        else:
            stopx = pix + width + 1

        if not skip:
            if dim == 0:
                updated_data[frame_index[0], starty:stopy, startx:stopx] = 0
                updated_mask[frame_index[0], starty:stopy, startx:stopx] = 1
            elif dim == 1:
                updated_data[starty:stopy, frame_index[1], startx:stopx] = 0
                updated_mask[starty:stopy, frame_index[1], startx:stopx] = 1
            else:  # dim=2
                updated_data[starty:stopy, startx:stopx, frame_index[2]] = 0
                updated_mask[starty:stopy, startx:stopx, frame_index[2]] = 1

    elif key == "b":  # back to measured intensities
        skip = False

        # check if the masking window fit in the data range
        # (vertical axis of the 2D plot)
        if (piy - width) < 0:
            starty = min(0, piy + width)
            if starty < 0:
                skip = True
        else:
            starty = piy - width
        if (piy + width) >= current_nby:
            stopy = max(current_nby, piy - width)
            if stopy > current_nby:
                skip = True
        else:
            stopy = piy + width + 1

        # check if the masking window fit in the data range
        # (horizontal axis of the 2D plot)
        if (pix - width) < 0:
            startx = min(0, pix + width)
            if startx < 0:
                skip = True
        else:
            startx = pix - width
        if (pix + width) >= current_nbx:
            stopx = max(current_nbx, pix - width)
            if stopx > current_nbx:
                skip = True
        else:
            stopx = pix + width + 1

        if not skip:
            if dim == 0:
                updated_data[
                    frame_index[0], starty:stopy, startx:stopx
                ] = original_data[frame_index[0], starty:stopy, startx:stopx]
                updated_mask[
                    frame_index[0], starty:stopy, startx:stopx
                ] = original_mask[frame_index[0], starty:stopy, startx:stopx]
            elif dim == 1:
                updated_data[
                    starty:stopy, frame_index[1], startx:stopx
                ] = original_data[starty:stopy, frame_index[1], startx:stopx]
                updated_mask[
                    starty:stopy, frame_index[1], startx:stopx
                ] = original_mask[starty:stopy, frame_index[1], startx:stopx]
            else:  # dim=2
                updated_data[
                    starty:stopy, startx:stopx, frame_index[2]
                ] = original_data[starty:stopy, startx:stopx, frame_index[2]]
                updated_mask[
                    starty:stopy, startx:stopx, frame_index[2]
                ] = original_mask[starty:stopy, startx:stopx, frame_index[2]]

    elif key == "f":  # fill empty voxels
        skip = False

        # check if the masking window fit in the data range
        # (vertical axis of the 2D plot)
        if (piy - width) < 0:
            starty = min(0, piy + width)
            if starty < 0:
                skip = True
        else:
            starty = piy - width
        if (piy + width) >= current_nby:
            stopy = max(current_nby, piy - width)
            if stopy > current_nby:
                skip = True
        else:
            stopy = piy + width + 1

        # check if the masking window fit in the data range
        # (horizontal axis of the 2D plot)
        if (pix - width) < 0:
            startx = min(0, pix + width)
            if startx < 0:
                skip = True
        else:
            startx = pix - width
        if (pix + width) >= current_nbx:
            stopx = max(current_nbx, pix - width)
            if stopx > current_nbx:
                skip = True
        else:
            stopx = pix + width + 1

        if not skip:
            if dim == 0:
                updated_data[
                    frame_index[0], starty:stopy, startx:stopx
                ] = original_data.max()
                updated_mask[frame_index[0], starty:stopy, startx:stopx] = -1
            elif dim == 1:
                updated_data[
                    starty:stopy, frame_index[1], startx:stopx
                ] = original_data.max()
                updated_mask[starty:stopy, frame_index[1], startx:stopx] = -1
            else:  # dim=2
                updated_data[
                    starty:stopy, startx:stopx, frame_index[2]
                ] = original_data.max()
                updated_mask[starty:stopy, startx:stopx, frame_index[2]] = -1

    elif key in {"p", "a"}:  # plot full image or restart masking
        xmin0, xmax0 = -0.5, nbx - 0.5
        if invert_yaxis:
            ymin0, ymax0 = -0.5, nby - 0.5  # pointing up
        else:
            ymin0, ymax0 = nby - 0.5, -0.5  # pointing down
        xmin1, xmax1 = -0.5, nbx - 0.5
        ymin1, ymax1 = nbz - 0.5, -0.5  # pointing down
        xmin2, xmax2 = -0.5, nby - 0.5
        ymin2, ymax2 = nbz - 0.5, -0.5  # pointing down
        if key == "a":  # restart masking
            updated_data[:] = original_data[:]
            updated_mask[:] = original_mask[:]

    elif key == "q":
        stop_masking = True

    else:
        return updated_data, updated_mask, width, vmax, frame_index, stop_masking

    axes[0].cla()
    axes[1].cla()
    axes[2].cla()
    axes[0].imshow(updated_data[frame_index[0], :, :], vmin=vmin, vmax=vmax, cmap=cmap)
    axes[1].imshow(updated_data[:, frame_index[1], :], vmin=vmin, vmax=vmax, cmap=cmap)
    axes[2].imshow(updated_data[:, :, frame_index[2]], vmin=vmin, vmax=vmax, cmap=cmap)
    axes[0].set_title("XY - Frame " + str(frame_index[0] + 1) + "/" + str(nbz))
    axes[0].axis("scaled")
    if invert_yaxis:
        axes[0].invert_yaxis()
    axes[0].set_xlim([xmin0, xmax0])
    axes[0].set_ylim([ymin0, ymax0])
    axes[1].set_title("XZ - Frame " + str(frame_index[1] + 1) + "/" + str(nby))
    axes[1].axis("scaled")
    axes[1].set_xlim([xmin1, xmax1])
    axes[1].set_ylim([ymin1, ymax1])
    axes[2].set_title("YZ - Frame " + str(frame_index[2] + 1) + "/" + str(nbx))
    axes[2].axis("scaled")
    axes[2].set_xlim([xmin2, xmax2])
    axes[2].set_ylim([ymin2, ymax2])
    plt.draw()

    return updated_data, updated_mask, width, vmax, frame_index, stop_masking


def update_aliens_2d(
    key,
    pix,
    piy,
    original_data,
    original_mask,
    updated_data,
    updated_mask,
    figure,
    width,
    vmax,
    vmin=0,
    invert_yaxis=False,
):
    """
    Update the plot while removing the parasitic diffraction intensity in 2D dataset.

    :param key: the keyboard key which was pressed
    :param pix: the x value of the mouse pointer
    :param piy: the y value of the mouse pointer
    :param original_data: the 2D data array before masking aliens
    :param original_mask: the 2D mask array before masking aliens
    :param updated_data: the current 2D data array
    :param updated_mask: the current 2D mask array
    :param figure: the figure instance
    :param width: the half_width of the masking window
    :param vmax: the higher boundary for the colorbar
    :param vmin: the lower boundary for the colorbar
    :param invert_yaxis: True to invert the y axis of imshow plots
    :return: updated data, mask and controls
    """
    # check some parameters
    valid.valid_ndarray(
        arrays=(original_data, updated_data, original_mask, updated_mask),
        ndim=2,
    )

    # process arrays
    nby, nbx = original_data.shape
    stop_masking = False

    axs = figure.gca()
    xmin, xmax = axs.get_xlim()
    ymin, ymax = axs.get_ylim()

    if key == "up":
        width = width + 1

    elif key == "down":
        width = width - 1
        width = max(width, 0)

    elif key == "right":
        vmax = vmax * 2

    elif key == "left":
        vmax = vmax / 2
        vmax = max(vmax, 1)

    elif key == "m":
        skip = False

        # check if the masking window fit in the data range
        # (vertical axis of the 2D plot)
        if (piy - width) < 0:
            starty = min(0, piy + width)
            if starty < 0:
                skip = True
        else:
            starty = piy - width
        if (piy + width) >= nby:
            stopy = max(nby, piy - width)
            if stopy > nby:
                skip = True
        else:
            stopy = piy + width + 1

        # check if the masking window fit in the data range
        # (horizontal axis of the 2D plot)
        if (pix - width) < 0:
            startx = min(0, pix + width)
            if startx < 0:
                skip = True
        else:
            startx = pix - width
        if (pix + width) >= nbx:
            stopx = max(nbx, pix - width)
            if stopx > nbx:
                skip = True
        else:
            stopx = pix + width + 1

        if not skip:
            updated_data[starty:stopy, startx:stopx] = 0
            updated_mask[starty:stopy, startx:stopx] = 1

    elif key == "b":
        skip = False

        # check if the masking window fit in the data range
        # (vertical axis of the 2D plot)
        if (piy - width) < 0:
            starty = min(0, piy + width)
            if starty < 0:
                skip = True
        else:
            starty = piy - width
        if (piy + width) >= nby:
            stopy = max(nby, piy - width)
            if stopy > nby:
                skip = True
        else:
            stopy = piy + width + 1

        # check if the masking window fit in the data range
        # (horizontal axis of the 2D plot)
        if (pix - width) < 0:
            startx = min(0, pix + width)
            if startx < 0:
                skip = True
        else:
            startx = pix - width
        if (pix + width) >= nbx:
            stopx = max(nbx, pix - width)
            if stopx > nbx:
                skip = True
        else:
            stopx = pix + width + 1

        if not skip:
            updated_data[starty:stopy, startx:stopx] = original_data[
                starty:stopy, startx:stopx
            ]
            updated_mask[starty:stopy, startx:stopx] = original_mask[
                starty:stopy, startx:stopx
            ]

    elif key in {"p", "a"}:  # plot full image or restart masking
        xmin, xmax = -0.5, nbx - 0.5
        if invert_yaxis:
            ymin, ymax = -0.5, nby - 0.5  # pointing up
        else:
            ymin, ymax = nby - 0.5, -0.5  # pointing down
        if key == "a":  # restart masking
            updated_data[:] = original_data[:]
            updated_mask[:] = original_mask[:]

    elif key == "q":
        stop_masking = True

    else:
        return updated_data, updated_mask, width, vmax, stop_masking

    axs.cla()
    axs.imshow(updated_data, vmin=vmin, vmax=vmax)
    axs.set_title(
        "m mask ; b unmask ; q quit ; u next frame ; d previous frame\n"
        "up larger ; down smaller ; right darker ; left brighter"
    )
    if invert_yaxis:
        axs.invert_yaxis()
    axs.set_xlim([xmin, xmax])
    axs.set_ylim([ymin, ymax])
    plt.draw()

    return updated_data, updated_mask, width, vmax, stop_masking


def update_background(
    key, distances, data, figure, flag_pause, xy, scale="log", xlim=None, ylim=None
):
    """
    Define the background for a 1D reciprocal space dataset.

    :param key: the keyboard key which was pressed
    :param distances: x axis for data
    :param data: the 1D data before background subtraction
    :param figure: the figure instance
    :param flag_pause: set to 1 to stop registering vertices using mouse clicks
    :param xy: the list of vertices which defines a polygon to be masked
    :param scale: scale of data, 'linear' or 'log'
    :param xlim: x axis plot limits
    :param ylim: y axis plot limits
    :return: updated background and controls
    """
    valid.valid_ndarray(data, ndim=1)
    axs = figure.gca()
    if xlim is None:
        xmin, xmax = axs.get_xlim()
    else:
        xmin, xmax = xlim
    if ylim is None:
        ymin, ymax = axs.get_ylim()
    else:
        ymin, ymax = ylim

    stop_masking = False
    xy = sorted(xy, key=itemgetter(0))

    if key == "b":  # remove the last selected background point
        xy.pop()

    elif key == "a":  # restart background selection from the beginning
        xy = []
        print("restart background selection")

    elif key == "p":  # plot background
        pass

    elif key == "x":
        if not flag_pause:
            flag_pause = True
            print("pause for pan/zoom")
        else:
            flag_pause = False
            print("resume masking")

    elif key == "q":
        stop_masking = True

    else:
        return flag_pause, xy, stop_masking

    background = np.asarray(xy)
    axs.cla()
    if len(xy) != 0:
        if scale == "linear":
            axs.plot(distances, data, ".-r", background[:, 0], background[:, 1], "b")
        else:
            axs.plot(
                distances,
                np.log10(data),
                ".-r",
                background[:, 0],
                background[:, 1],
                "b",
            )  # background is in log scale directly
    else:  # restart background selection
        if scale == "linear":
            axs.plot(distances, data, ".-r")
        else:
            axs.plot(distances, np.log10(data), ".-r")
    axs.set_xlim([xmin, xmax])
    axs.set_ylim([ymin, ymax])
    axs.set_xlabel("q (1/nm)")
    axs.set_ylabel("Angular average (A.U.)")
    axs.set_title(
        "Click to select background points\nx to pause/resume for pan/zoom\n"
        "a restart ; p plot background ; q quit"
    )
    plt.draw()

    return flag_pause, xy, stop_masking


def update_mask(
    key,
    pix,
    piy,
    original_data,
    original_mask,
    updated_data,
    updated_mask,
    figure,
    flag_pause,
    points,
    xy,
    width,
    dim,
    vmax,
    vmin=0,
    masked_color=0.1,
    invert_yaxis=False,
):
    """
    Update the mask corresponding to parasitic intensities in a 3D dataset.

    The GUI contains one 2D projection of the mask, the projection axis is
    determined by the parameter "dim".

    :param key: the keyboard key which was pressed
    :param pix: the x value of the mouse pointer
    :param piy: the y value of the mouse pointer
    :param original_data: the 3D data array before masking
    :param original_mask: the 3D mask array before masking
    :param updated_data: the current 3D data array
    :param updated_mask: the temporary 2D mask array with updated points
    :param figure: the figure instance
    :param flag_pause: set to 1 to stop registering vertices using mouse clicks
    :param points: list of all point coordinates: points=np.stack((x, y), axis=0).T
     with x=x.flatten() , y = y.flatten() given x,y=np.meshgrid(np.arange(nx),
     np.arange(ny))
    :param xy: the list of vertices which defines a polygon to be masked
    :param width: the half_width of the masking window
    :param dim: the axis currently under review (axis 0, 1 or 2)
    :param vmax: the higher boundary for the colorbar
    :param vmin: the lower boundary for the colorbar
    :param masked_color: the value that detector gaps should have in plots
    :param invert_yaxis: True to invert the y axis of imshow plots
    :return: updated data, mask and controls
    """
    # check some parameters
    valid.valid_ndarray(
        arrays=(original_data, updated_data, original_mask),
        ndim=3,
    )
    valid.valid_ndarray(updated_mask, ndim=2)
    if dim not in {0, 1, 2}:
        raise ValueError("dim should be 0, 1 or 2")

    # process arrays
    nbz, nby, nbx = original_data.shape
    stop_masking = False
    if dim == 0:
        current_nby = nby
        current_nbx = nbx
    elif dim == 1:
        current_nby = nbz
        current_nbx = nbx
    else:  # dim = 2
        current_nby = nbz
        current_nbx = nby

    axs = figure.gca()
    xmin, xmax = axs.get_xlim()
    ymin, ymax = axs.get_ylim()

    if key == "up":
        width = width + 1

    elif key == "down":
        width = width - 1
        width = max(width, 0)

    elif key == "right":
        vmax = vmax + 1

    elif key == "left":
        vmax = vmax - 1
        vmax = max(vmax, 1)

    elif key == "m":
        skip = False

        # check if the masking window fit in the data range
        # (vertical axis of the 2D plot)
        if (piy - width) < 0:
            starty = min(0, piy + width)
            if starty < 0:
                skip = True
        else:
            starty = piy - width
        if (piy + width) >= current_nby:
            stopy = max(current_nby, piy - width)
            if stopy > current_nby:
                skip = True
        else:
            stopy = piy + width + 1

        # check if the masking window fit in the data range
        # (horizontal axis of the 2D plot)
        if (pix - width) < 0:
            startx = min(0, pix + width)
            if startx < 0:
                skip = True
        else:
            startx = pix - width
        if (pix + width) >= current_nbx:
            stopx = max(current_nbx, pix - width)
            if stopx > current_nbx:
                skip = True
        else:
            stopx = pix + width + 1

        if not skip:
            updated_mask[starty:stopy, startx:stopx] = 1

    elif key == "b":
        skip = False

        # check if the masking window fit in the data range
        # (vertical axis of the 2D plot)
        if (piy - width) < 0:
            starty = min(0, piy + width)
            if starty < 0:
                skip = True
        else:
            starty = piy - width
        if (piy + width) >= current_nby:
            stopy = max(current_nby, piy - width)
            if stopy > current_nby:
                skip = True
        else:
            stopy = piy + width + 1

        # check if the masking window fit in the data range
        # (horizontal axis of the 2D plot)
        if (pix - width) < 0:
            startx = min(0, pix + width)
            if startx < 0:
                skip = True
        else:
            startx = pix - width
        if (pix + width) >= current_nbx:
            stopx = max(current_nbx, pix - width)
            if stopx > current_nbx:
                skip = True
        else:
            stopx = pix + width + 1

        if not skip:
            updated_mask[starty:stopy, startx:stopx] = 0
            if dim == 0:
                updated_data[:, starty:stopy, startx:stopx] = original_data[
                    :, starty:stopy, startx:stopx
                ]
            elif dim == 1:
                updated_data[starty:stopy, :, startx:stopx] = original_data[
                    starty:stopy, :, startx:stopx
                ]
            else:  # dim=2
                updated_data[starty:stopy, startx:stopx, :] = original_data[
                    starty:stopy, startx:stopx, :
                ]

    elif key == "a":  # restart mask from beginning
        updated_data[:] = original_data[:]
        xy = []
        print("Restart masking...")
        if dim == 0:
            updated_data[original_mask == 1] = (
                masked_color / nbz
            )  # masked pixels plotted with the value of masked_pixel
            updated_mask = np.zeros((nby, nbx))
            xmin, xmax = -0.5, nbx - 0.5
            if invert_yaxis:
                ymin, ymax = -0.5, nby - 0.5  # pointing up
            else:
                ymin, ymax = nby - 0.5, -0.5  # pointing down
        elif dim == 1:
            updated_data[original_mask == 1] = (
                masked_color / nby
            )  # masked pixels plotted with the value of masked_pixel
            updated_mask = np.zeros((nbz, nbx))
            xmin, xmax = -0.5, nbx - 0.5
            ymin, ymax = nbz - 0.5, -0.5  # pointing down
        else:  # dim=2
            updated_data[original_mask == 1] = (
                masked_color / nbx
            )  # masked pixels plotted with the value of masked_pixel
            updated_mask = np.zeros((nbz, nby))
            xmin, xmax = -0.5, nby - 0.5
            ymin, ymax = nbz - 0.5, -0.5  # pointing down

    elif key == "p":  # plot full image
        if dim == 0:
            xmin, xmax = -0.5, nbx - 0.5
            if invert_yaxis:
                ymin, ymax = -0.5, nby - 0.5  # pointing up
            else:
                ymin, ymax = nby - 0.5, -0.5  # pointing down
        elif dim == 1:
            xmin, xmax = -0.5, nbx - 0.5
            ymin, ymax = nbz - 0.5, -0.5  # pointing down
        else:  # dim=2
            xmin, xmax = -0.5, nby - 0.5
            ymin, ymax = nbz - 0.5, -0.5  # pointing down
        if not flag_pause and len(xy) != 0:
            xy.append(xy[0])
            print(xy)
            if dim == 0:
                ind = Path(np.array(xy)).contains_points(points).reshape((nby, nbx))
            elif dim == 1:
                ind = Path(np.array(xy)).contains_points(points).reshape((nbz, nbx))
            else:  # dim=2
                ind = Path(np.array(xy)).contains_points(points).reshape((nbz, nby))
            updated_mask[ind] = 1
        xy = []  # allow to mask a different area

    elif key == "r":
        xy = []

    elif key == "x":
        if not flag_pause:
            flag_pause = True
            print("pause for pan/zoom")
        else:
            flag_pause = False
            print("resume masking")

    elif key == "q":
        stop_masking = True

    else:
        return updated_data, updated_mask, flag_pause, xy, width, vmax, stop_masking

    array = updated_data.sum(axis=dim)  # updated_data is not modified
    array[updated_mask == 1] = masked_color

    axs.cla()
    axs.imshow(np.log10(abs(array)), vmin=vmin, vmax=vmax)
    if invert_yaxis:
        axs.invert_yaxis()
    axs.set_xlim([xmin, xmax])
    axs.set_ylim([ymin, ymax])
    axs.set_title(
        "x to pause/resume masking for pan/zoom \n"
        "p plot mask ; a restart ; click to select vertices\n"
        "m mask ; b unmask ; q quit ; u next frame ; d previous frame\n"
        "up larger ; down smaller ; right darker ; left brighter"
    )
    plt.draw()

    return updated_data, updated_mask, flag_pause, xy, width, vmax, stop_masking


def update_mask_combined(
    key,
    pix,
    piy,
    original_data,
    original_mask,
    updated_data,
    updated_mask,
    axes,
    flag_pause,
    points,
    xy,
    width,
    dim,
    click_dim,
    info_text,
    vmax,
    vmin=0,
    cmap=default_cmap,
    invert_yaxis=False,
):
    """
    Update the mask to remove parasitic intensities in a 3D dataset.

    The GUI contains the three 2D projections of the mask, the axis being modified is
    determined by the parameter "dim".

    :param key: the keyboard key which was pressed
    :param pix: the x value of the mouse pointer
    :param piy: the y value of the mouse pointer
    :param original_data: the 3D data array before masking
    :param original_mask: the 3D mask array before masking
    :param updated_data: the current 3D data array
    :param updated_mask: the temporary 3D mask array with updated points
    :param axes: tuple of the 4 axes instances in a plt.subplots(nrows=2, ncols=2)
    :param flag_pause: set to 1 to stop registering vertices using mouse clicks
    :param points: list of all point coordinates: points=np.stack((x, y), axis=0).T
     with x=x.flatten() , y = y.flatten() given x,y=np.meshgrid(np.arange(nx),
     np.arange(ny))
    :param xy: the list of vertices which defines a polygon to be masked
    :param width: the half_width of the masking window
    :param dim: the axis currently under review (axis 0, 1 or 2)
    :param click_dim: the dimension (0, 1 or 2) where the selection of mask polygon
     vertices by clicking was performed
    :param info_text: text instance in the figure
    :param vmax: the higher boundary for the colorbar
    :param vmin: the lower boundary for the colorbar
    :param cmap: colormap to be used
    :param invert_yaxis: True to invert the y axis of imshow plots
    :return: updated data, mask (-1 filled, 0 non masked, 1 masked voxel) and controls
    """
    # check some parameters
    valid.valid_ndarray(
        arrays=(original_data, updated_data, original_mask, updated_mask),
        ndim=3,
    )
    if dim not in {0, 1, 2}:
        raise ValueError("dim should be 0, 1 or 2")

    # process arrays
    nbz, nby, nbx = original_data.shape
    stop_masking = False
    update_fig = False
    if dim == 0:
        current_nby = nby
        current_nbx = nbx
    elif dim == 1:
        current_nby = nbz
        current_nbx = nbx
    else:  # dim = 2
        current_nby = nbz
        current_nbx = nby

    xmin0, xmax0 = axes[0].get_xlim()
    ymin0, ymax0 = axes[0].get_ylim()
    xmin1, xmax1 = axes[1].get_xlim()
    ymin1, ymax1 = axes[1].get_ylim()
    xmin2, xmax2 = axes[2].get_xlim()
    ymin2, ymax2 = axes[2].get_ylim()

    if key == "up":
        width = width + 1

    elif key == "down":
        width = width - 1
        width = max(width, 0)

    elif key == "right":
        vmax = vmax + 1
        update_fig = True

    elif key == "left":
        vmax = vmax - 1
        vmax = max(vmax, 1)
        update_fig = True

    elif key == "m":
        skip = False
        update_fig = True

        # check if the masking window fit in the data range
        # (vertical axis of the 2D plot)
        if (piy - width) < 0:
            starty = min(0, piy + width)
            if starty < 0:
                skip = True
        else:
            starty = piy - width
        if (piy + width) >= current_nby:
            stopy = max(current_nby, piy - width)
            if stopy > current_nby:
                skip = True
        else:
            stopy = piy + width + 1

        # check if the masking window fit in the data range
        # (horizontal axis of the 2D plot)
        if (pix - width) < 0:
            startx = min(0, pix + width)
            if startx < 0:
                skip = True
        else:
            startx = pix - width
        if (pix + width) >= current_nbx:
            stopx = max(current_nbx, pix - width)
            if stopx > current_nbx:
                skip = True
        else:
            stopx = pix + width + 1

        if not skip:
            if dim == 0:
                updated_mask[:, starty:stopy, startx:stopx] = 1
            elif dim == 1:
                updated_mask[starty:stopy, :, startx:stopx] = 1
            else:  # dim=2
                updated_mask[starty:stopy, startx:stopx, :] = 1

    elif key == "b":
        skip = False
        update_fig = True

        # check if the masking window fit in the data range
        # (vertical axis of the 2D plot)
        if (piy - width) < 0:
            starty = min(0, piy + width)
            if starty < 0:
                skip = True
        else:
            starty = piy - width
        if (piy + width) >= current_nby:
            stopy = max(current_nby, piy - width)
            if stopy > current_nby:
                skip = True
        else:
            stopy = piy + width + 1

        # check if the masking window fit in the data range
        # (horizontal axis of the 2D plot)
        if (pix - width) < 0:
            startx = min(0, pix + width)
            if startx < 0:
                skip = True
        else:
            startx = pix - width
        if (pix + width) >= current_nbx:
            stopx = max(current_nbx, pix - width)
            if stopx > current_nbx:
                skip = True
        else:
            stopx = pix + width + 1

        if not skip:
            if dim == 0:
                updated_mask[:, starty:stopy, startx:stopx] = 0
                updated_data[:, starty:stopy, startx:stopx] = original_data[
                    :, starty:stopy, startx:stopx
                ]
            elif dim == 1:
                updated_mask[starty:stopy, :, startx:stopx] = 0
                updated_data[starty:stopy, :, startx:stopx] = original_data[
                    starty:stopy, :, startx:stopx
                ]
            else:  # dim=2
                updated_mask[starty:stopy, startx:stopx, :] = 0
                updated_data[starty:stopy, startx:stopx, :] = original_data[
                    starty:stopy, startx:stopx, :
                ]

    elif key == "f":  # fill with ones
        skip = False
        update_fig = True

        # check if the masking window fit in the data range
        # (vertical axis of the 2D plot)
        if (piy - width) < 0:
            starty = min(0, piy + width)
            if starty < 0:
                skip = True
        else:
            starty = piy - width
        if (piy + width) >= current_nby:
            stopy = max(current_nby, piy - width)
            if stopy > current_nby:
                skip = True
        else:
            stopy = piy + width + 1

        # check if the masking window fit in the data range
        # (horizontal axis of the 2D plot)
        if (pix - width) < 0:
            startx = min(0, pix + width)
            if startx < 0:
                skip = True
        else:
            startx = pix - width
        if (pix + width) >= current_nbx:
            stopx = max(current_nbx, pix - width)
            if stopx > current_nbx:
                skip = True
        else:
            stopx = pix + width + 1

        if not skip:
            if dim == 0:
                updated_mask[:, starty:stopy, startx:stopx] = -1
                updated_data[:, starty:stopy, startx:stopx] = original_data.max()
            elif dim == 1:
                updated_mask[starty:stopy, :, startx:stopx] = -1
                updated_data[starty:stopy, :, startx:stopx] = original_data.max()
            else:  # dim=2
                updated_mask[starty:stopy, startx:stopx, :] = -1
                updated_data[starty:stopy, startx:stopx, :] = original_data.max()

    elif key == "a":  # restart mask from beginning
        update_fig = True
        updated_data = np.copy(original_data)
        xy = []
        click_dim = None
        print("Restart masking...")
        xmin0, xmax0 = -0.5, nbx - 0.5
        if invert_yaxis:
            ymin0, ymax0 = -0.5, nby - 0.5  # pointing up
        else:
            ymin0, ymax0 = nby - 0.5, -0.5  # pointing down
        xmin1, xmax1 = -0.5, nbx - 0.5
        ymin1, ymax1 = nbz - 0.5, -0.5  # pointing down
        xmin2, xmax2 = -0.5, nby - 0.5
        ymin2, ymax2 = nbz - 0.5, -0.5  # pointing down

        updated_data[:] = original_data[:]
        updated_mask = np.zeros((nbz, nby, nbx))

    elif key == "p":  # plot full image
        update_fig = True
        xmin0, xmax0 = -0.5, nbx - 0.5
        if invert_yaxis:
            ymin0, ymax0 = -0.5, nby - 0.5  # pointing up
        else:
            ymin0, ymax0 = nby - 0.5, -0.5  # pointing down
        xmin1, xmax1 = -0.5, nbx - 0.5
        ymin1, ymax1 = nbz - 0.5, -0.5  # pointing down
        xmin2, xmax2 = -0.5, nby - 0.5
        ymin2, ymax2 = nbz - 0.5, -0.5  # pointing down
        if not flag_pause and len(xy) != 0:
            xy.append(xy[0])
            print(xy)
            if click_dim == 0:
                ind = Path(np.array(xy)).contains_points(points).reshape((nby, nbx))
                temp_mask = np.zeros((nby, nbx))
                temp_mask[ind] = 1
                updated_mask[
                    np.repeat(temp_mask[np.newaxis, :, :], repeats=nbz, axis=0) == 1
                ] = 1
            elif click_dim == 1:
                ind = Path(np.array(xy)).contains_points(points).reshape((nbz, nbx))
                temp_mask = np.zeros((nbz, nbx))
                temp_mask[ind] = 1
                updated_mask[
                    np.repeat(temp_mask[:, np.newaxis, :], repeats=nby, axis=1) == 1
                ] = 1
            else:  # dim=2
                ind = Path(np.array(xy)).contains_points(points).reshape((nbz, nby))
                temp_mask = np.zeros((nbz, nby))
                temp_mask[ind] = 1
                updated_mask[
                    np.repeat(temp_mask[:, :, np.newaxis], repeats=nbx, axis=2) == 1
                ] = 1
        xy = []  # allow to mask a different area
        click_dim = None

    elif key == "r":
        xy = []

    elif key == "x":
        if not flag_pause:
            flag_pause = True
            print("pause for pan/zoom")
        else:
            flag_pause = False
            print("resume masking")

    elif key == "q":
        stop_masking = True

    else:
        return (
            updated_data,
            updated_mask,
            flag_pause,
            xy,
            width,
            vmax,
            click_dim,
            stop_masking,
            info_text,
        )

    if update_fig:
        updated_data[original_mask == 1] = 0
        updated_data[updated_mask == 1] = 0

        axes[0].cla()
        axes[1].cla()
        axes[2].cla()
        axes[0].imshow(
            np.log10(updated_data.sum(axis=0)), vmin=vmin, vmax=vmax, cmap=cmap
        )
        axes[1].imshow(
            np.log10(updated_data.sum(axis=1)), vmin=vmin, vmax=vmax, cmap=cmap
        )
        axes[2].imshow(
            np.log10(updated_data.sum(axis=2)), vmin=vmin, vmax=vmax, cmap=cmap
        )
        axes[0].set_title("XY")
        axes[0].axis("scaled")
        if invert_yaxis:
            axes[0].invert_yaxis()
        axes[0].set_xlim([xmin0, xmax0])
        axes[0].set_ylim([ymin0, ymax0])
        axes[1].set_title("XZ")
        axes[1].axis("scaled")
        axes[1].set_xlim([xmin1, xmax1])
        axes[1].set_ylim([ymin1, ymax1])
        axes[2].set_title("YZ")
        axes[2].axis("scaled")
        axes[2].set_xlim([xmin2, xmax2])
        axes[2].set_ylim([ymin2, ymax2])
    fig = plt.gcf()
    info_text.remove()
    if flag_pause:
        info_text = fig.text(0.6, 0.05, "masking paused", size=16)
    else:
        info_text = fig.text(0.6, 0.05, "masking enabled", size=16)
    plt.draw()

    return (
        updated_data,
        updated_mask,
        flag_pause,
        xy,
        width,
        vmax,
        click_dim,
        stop_masking,
        info_text,
    )


def update_mask_2d(
    key,
    pix,
    piy,
    original_data,
    original_mask,
    updated_data,
    updated_mask,
    figure,
    flag_pause,
    points,
    xy,
    width,
    vmax,
    vmin=0,
    masked_color=0.1,
    invert_yaxis=False,
):
    """
    Update the mask corresponding to parasitic intensities in a 2D dataset.

    :param key: the keyboard key which was pressed
    :param pix: the x value of the mouse pointer
    :param piy: the y value of the mouse pointer
    :param original_data: the 2D data array before masking
    :param original_mask: the 2D mask array before masking
    :param updated_data: the current 2D data array
    :param updated_mask: the temporary 2D mask array with updated points
    :param figure: the figure instance
    :param flag_pause: set to 1 to stop registering vertices using mouse clicks
    :param points: list of all point coordinates: points=np.stack((x, y), axis=0).T
     with x=x.flatten() , y = y.flatten() given x,y=np.meshgrid(np.arange(nx),
     np.arange(ny))
    :param xy: the list of vertices which defines a polygon to be masked
    :param width: the half_width of the masking window
    :param vmax: the higher boundary for the colorbar
    :param vmin: the lower boundary for the colorbar
    :param masked_color: the value that detector gaps should have in plots
    :param invert_yaxis: True to invert the y axis of imshow plots
    :return: updated data, mask and controls
    """
    # check some parameters
    valid.valid_ndarray(
        arrays=(original_data, updated_data, original_mask, updated_mask),
        ndim=2,
    )
    # process arrays
    nby, nbx = original_data.shape
    stop_masking = False

    axs = figure.gca()
    xmin, xmax = axs.get_xlim()
    ymin, ymax = axs.get_ylim()

    if key == "up":
        width = width + 1

    elif key == "down":
        width = width - 1
        width = max(width, 0)

    elif key == "right":
        vmax = vmax + 1
        updated_data[updated_mask == 1] = masked_color

    elif key == "left":
        vmax = vmax - 1
        vmax = max(vmax, 1)
        updated_data[updated_mask == 1] = masked_color

    elif key == "m":
        skip = False

        # check if the masking window fit in the data range
        # (vertical axis of the 2D plot)
        if (piy - width) < 0:
            starty = min(0, piy + width)
            if starty < 0:
                skip = True
        else:
            starty = piy - width
        if (piy + width) >= nby:
            stopy = max(nby, piy - width)
            if stopy > nby:
                skip = True
        else:
            stopy = piy + width + 1

        # check if the masking window fit in the data range
        # (horizontal axis of the 2D plot)
        if (pix - width) < 0:
            startx = min(0, pix + width)
            if startx < 0:
                skip = True
        else:
            startx = pix - width
        if (pix + width) >= nbx:
            stopx = max(nbx, pix - width)
            if stopx > nbx:
                skip = True
        else:
            stopx = pix + width + 1

        if not skip:
            updated_mask[starty:stopy, startx:stopx] = 1
            updated_data[updated_mask == 1] = masked_color

    elif key == "b":
        skip = False

        # check if the masking window fit in the data range
        # (vertical axis of the 2D plot)
        if (piy - width) < 0:
            starty = min(0, piy + width)
            if starty < 0:
                skip = True
        else:
            starty = piy - width
        if (piy + width) >= nby:
            stopy = max(nby, piy - width)
            if stopy > nby:
                skip = True
        else:
            stopy = piy + width + 1

        # check if the masking window fit in the data range
        # (horizontal axis of the 2D plot)
        if (pix - width) < 0:
            startx = min(0, pix + width)
            if startx < 0:
                skip = True
        else:
            startx = pix - width
        if (pix + width) >= nbx:
            stopx = max(nbx, pix - width)
            if stopx > nbx:
                skip = True
        else:
            stopx = pix + width + 1

        if not skip:
            updated_mask[starty:stopy, startx:stopx] = 0
            updated_data[updated_mask == 1] = masked_color

    elif key == "a":  # restart mask from beginning
        updated_data = np.copy(original_data)
        xy = []
        print("restart masking")
        updated_data[
            original_mask == 1
        ] = masked_color  # masked pixels plotted with the value of masked_pixel
        updated_mask = np.zeros((nby, nbx))
        xmin, xmax = -0.5, nbx - 0.5
        if invert_yaxis:
            ymin, ymax = -0.5, nby - 0.5  # pointing up
        else:
            ymin, ymax = nby - 0.5, -0.5  # pointing down

    elif key == "p":  # plot full image
        xmin, xmax = -0.5, nbx - 0.5
        if invert_yaxis:
            ymin, ymax = -0.5, nby - 0.5  # pointing up
        else:
            ymin, ymax = nby - 0.5, -0.5  # pointing down
        if not flag_pause and len(xy) != 0:
            xy.append(xy[0])
            print(xy)
            ind = Path(np.array(xy)).contains_points(points).reshape((nby, nbx))
            updated_mask[ind] = 1

        updated_data[updated_mask == 1] = masked_color
        xy = []  # allow to mask a different area

    elif key == "r":
        xy = []

    elif key == "x":
        if not flag_pause:
            flag_pause = True
            print("pause for pan/zoom")
        else:
            flag_pause = False
            print("resume masking")

    elif key == "q":
        stop_masking = True

    else:
        return updated_data, updated_mask, flag_pause, xy, width, vmax, stop_masking

    axs.cla()
    axs.imshow(np.log10(abs(updated_data)), vmin=vmin, vmax=vmax)
    if invert_yaxis:
        axs.invert_yaxis()
    axs.set_xlim([xmin, xmax])
    axs.set_ylim([ymin, ymax])
    axs.set_title(
        "x to pause/resume masking for pan/zoom \n"
        "p plot mask ; a restart ; click to select vertices\n"
        "m mask ; b unmask ; q quit ; u next frame ; d previous frame\n"
        "up larger ; down smaller ; right darker ; left brighter"
    )
    plt.draw()

    return updated_data, updated_mask, flag_pause, xy, width, vmax, stop_masking
