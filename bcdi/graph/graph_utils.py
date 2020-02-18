# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patches
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap

# define a colormap
cdict = {'red':  ((0.0, 1.0, 1.0),
                  (0.11, 0.0, 0.0),
                  (0.36, 0.0, 0.0),
                  (0.62, 1.0, 1.0),
                  (0.87, 1.0, 1.0),
                  (1.0, 0.0, 0.0)),
         'green': ((0.0, 1.0, 1.0),
                   (0.11, 0.0, 0.0),
                   (0.36, 1.0, 1.0),
                   (0.62, 1.0, 1.0),
                   (0.87, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),
         'blue': ((0.0, 1.0, 1.0),
                  (0.11, 1.0, 1.0),
                  (0.36, 1.0, 1.0),
                  (0.62, 0.0, 0.0),
                  (0.87, 0.0, 0.0),
                  (1.0, 0.0, 0.0))}
my_cmap = LinearSegmentedColormap('my_colormap', cdict, 256)
my_cmap.set_bad(color='0.7')


class Colormap(object):
    """
    Class to define a colormap.
    """
    def __init__(self, bad_color='0.7', colormap='default'):
        """
        Initialize parameters of the experiment.

        :param colormap: a colormap string. Available choices at the moment: 'default'
        :param bad_color: a string which defines the grey level for nan pixels. example: '0.7'
        """
        if colormap == 'default':
            color_dict = {'red':  ((0.0, 1.0, 1.0),
                                   (0.11, 0.0, 0.0),
                                   (0.36, 0.0, 0.0),
                                   (0.62, 1.0, 1.0),
                                   (0.87, 1.0, 1.0),
                                   (1.0, 0.0, 0.0)),
                          'green': ((0.0, 1.0, 1.0),
                                    (0.11, 0.0, 0.0),
                                    (0.36, 1.0, 1.0),
                                    (0.62, 1.0, 1.0),
                                    (0.87, 0.0, 0.0),
                                    (1.0, 0.0, 0.0)),
                          'blue': ((0.0, 1.0, 1.0),
                                   (0.11, 1.0, 1.0),
                                   (0.36, 1.0, 1.0),
                                   (0.62, 0.0, 0.0),
                                   (0.87, 0.0, 0.0),
                                   (1.0, 0.0, 0.0))}
        else:
            raise ValueError('Only available colormaps: "default"')
        self.cdict = color_dict
        self.bad_color = bad_color
        self.cmap = LinearSegmentedColormap('my_colormap', color_dict, 256)
        self.cmap.set_bad(color=bad_color)


def combined_plots(tuple_array, tuple_sum_frames, tuple_width_v, tuple_width_h, tuple_colorbar, tuple_vmin,
                   tuple_vmax, tuple_title, tuple_scale, tuple_sum_axis=0, cmap=my_cmap, tick_direction='inout',
                   tick_width=1, tick_length=3, pixel_spacing=np.nan, is_orthogonal=False, reciprocal_space=False,
                   **kwargs):
    """
    Subplots of a 1D, 2D or 3D datasets using user-defined parameters.

    :param tuple_array: 2D or 3D array of real numbers
    :param tuple_sum_frames: boolean or tuple of boolean values. If True, will sum the data along sum_axis
    :param tuple_sum_axis: tuple of axis along which to sum or to take the middle slice
    :param tuple_width_v: int or tuple of user-defined zoom vertical width, should be smaller than the actual data
     size. Set it to np.nan if you do not need it.
    :param tuple_width_h: int or tuple of user-defined zoom horizontal width, should be smaller than the actual data
     size. Set it to np.nan if you do not need it.
    :param tuple_colorbar: boolean or tuple of boolean values. Set it to True in order to plot the colorbar
    :param tuple_vmin: float or tuple of lower boundaries for the colorbar, set to np.nan if you do not need it
    :param tuple_vmax: float or tuple of higher boundaries for the colorbar, set to np.nan if you do not need it
    :param tuple_title: string or tuple of strings, set to '' if you do not need it
    :param tuple_scale:  string ot tuple of strings with value 'linear' or 'log'
    :param cmap: colormap to be used
    :param tick_direction: 'out', 'in', 'inout'
    :param tick_width: width of tickes in plots
    :param tick_length: length of tickes in plots
    :param pixel_spacing: pixel_spacing = desired tick_spacing (in nm) / voxel_size of the reconstruction(in nm)
    :param is_orthogonal: set to True is the frame is orthogonal, False otherwise (detector frame)
    :param reciprocal_space: True if the data is in reciprocal space, False otherwise
    :param kwargs: optional 'xlabel' and 'ylabel', labels for plots
    :return:  the figure instance
    """
    if type(tuple_array) is not tuple:
        raise TypeError('Expected "tuple_array" to be a tuple')

    nb_subplots = len(tuple_array)

    if type(tuple_sum_frames) is not tuple:
        tuple_sum_frames = (tuple_sum_frames,) * nb_subplots
    if type(tuple_sum_axis) is not tuple:
        tuple_sum_axis = (tuple_sum_axis,) * nb_subplots
    if type(tuple_width_v) is not tuple:
        tuple_width_v = (tuple_width_v,) * nb_subplots
    if type(tuple_width_h) is not tuple:
        tuple_width_h = (tuple_width_h,) * nb_subplots
    if type(tuple_colorbar) is not tuple:
        tuple_colorbar = (tuple_colorbar,) * nb_subplots
    if type(tuple_vmin) is not tuple:
        tuple_vmin = (tuple_vmin,) * nb_subplots
    if type(tuple_vmax) is not tuple:
        tuple_vmax = (tuple_vmax,) * nb_subplots
    if type(tuple_title) is not tuple:
        tuple_title = (tuple_title,) * nb_subplots
    if type(tuple_scale) is not tuple:
        tuple_scale = (tuple_scale,) * nb_subplots

    for k in kwargs.keys():
        if k in ['xlabel']:
            xlabel = kwargs['xlabel']
        elif k in ['ylabel']:
            ylabel = kwargs['ylabel']
        else:
            raise Exception("unknown keyword argument given: allowed is"
                            "'xlabel' and 'ylabel'")
    try:
        xlabel
    except NameError:
        xlabel = ['']
        for idx in range(nb_subplots-1):
            xlabel.append('')

    try:
        ylabel
    except NameError:
        ylabel = ['']
        for idx in range(nb_subplots-1):
            ylabel.append('')

    nb_columns = nb_subplots // 2

    nb_raws = nb_subplots // nb_columns + nb_subplots % nb_columns

    plt.ion()
    plt.figure()
    for idx in range(nb_subplots):

        axis = plt.subplot(nb_raws, nb_columns, idx+1)

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

        if sum_frames:
            title = title + ' sum'

        nb_dim = array.ndim

        if nb_dim == 0 or nb_dim > 3:
            print('array ', idx, ': wrong dimension')
            continue

        elif nb_dim == 1:

            if np.isnan(vmin):
                tmp_array = np.copy(array).astype(float)
                tmp_array[np.isnan(array)] = np.inf
                tmp_array[np.isinf(tmp_array)] = np.inf  # set -inf to +inf to find the min
                vmin = tmp_array.min()
            if np.isnan(vmax):
                tmp_array = np.copy(array).astype(float)
                tmp_array[np.isnan(array)] = -1 * np.inf
                tmp_array[np.isinf(tmp_array)] = -1 * np.inf  # set +inf to -inf to find the max
                vmax = tmp_array.max()

            axis.plot(array)
            axis.set_title(title)
            axis.set_ylim(vmin, vmax)
            axis.set_yscale(scale)
            axis.set_xlabel(xlabel[idx])
            axis.set_ylabel(ylabel[idx])

            continue

        elif nb_dim == 3:  # 3D, needs to be reduced to 2D by slicing or projecting

            if reciprocal_space:
                if is_orthogonal:
                    slice_names = (' QyQz', ' QyQx', ' QzQx')
                else:  # detector frame
                    slice_names = (' XY', ' X_RockingAngle', ' Y_RockingAngle')
            else:
                if is_orthogonal:
                    slice_names = (' xy', ' xz', ' yz')
                else:  # detector frame
                    slice_names = (' XY', ' X_RockingAngle', ' Y_RockingAngle')

            nbz, nby, nbx = array.shape
            if np.isnan(width_v):
                width_v = max(nbz, nby, nbx)
            if np.isnan(width_h):
                width_h = max(nbz, nby, nbx)

            if sum_axis == 0:
                dim_v = nby
                dim_h = nbx
                if not sum_frames:
                    array = array[nbz // 2, :, :]
                else:
                    array = array.sum(axis=sum_axis)
            elif sum_axis == 1:
                dim_v = nbz
                dim_h = nbx
                if not sum_frames:
                    array = array[:, nby // 2, :]
                else:
                    array = array.sum(axis=sum_axis)
            elif sum_axis == 2:
                dim_v = nbz
                dim_h = nby
                if not sum_frames:
                    array = array[:, :, nbx // 2]
                else:
                    array = array.sum(axis=sum_axis)
            else:
                print('sum_axis should be only equal to 0, 1 or 2')
                return
            slice_name = slice_names[sum_axis]

        else:  # 2D
            nby, nbx = array.shape
            if np.isnan(width_v):
                width_v = max(nby, nbx)
            if np.isnan(width_h):
                width_h = max(nby, nbx)

            dim_v = nby
            dim_h = nbx
            slice_name = ''

        # now array is 2D
        width_v = min(width_v, dim_v)
        width_h = min(width_h, dim_h)
        array = array[int(np.rint(dim_v/2 - width_v/2)):int(np.rint(dim_v/2 - width_v/2)) + width_v,
                      int(np.rint(dim_h//2 - width_h//2)):int(np.rint(dim_h//2 - width_h//2)) + width_h]

        if scale == 'linear':
            if np.isnan(vmin):
                tmp_array = np.copy(array)
                tmp_array[np.isnan(array)] = np.inf
                tmp_array[np.isinf(tmp_array)] = np.inf  # set -inf to +inf to find the min
                vmin = tmp_array.min()
            if np.isnan(vmax):
                tmp_array = np.copy(array)
                tmp_array[np.isnan(array)] = -1 * np.inf
                tmp_array[np.isinf(tmp_array)] = -1 * np.inf  # set +inf to -inf to find the max
                vmax = tmp_array.max()

            plot = axis.imshow(array, vmin=vmin, vmax=vmax, cmap=cmap)
        elif scale == 'log':
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
                tmp_array[np.isinf(tmp_array)] = -1 * np.inf  # set +inf to -inf to find the max
                vmax = np.log10(abs(tmp_array).max())
            plot = axis.imshow(np.log10(abs(array)), vmin=vmin, vmax=vmax, cmap=cmap)
        else:
            raise ValueError('Wrong value for scale')

        axis.set_title(title + slice_name)
        axis.set_xlabel(xlabel[idx])
        axis.set_ylabel(ylabel[idx])
        plt.axis('scaled')
        if not np.isnan(pixel_spacing):
            axis.xaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
            axis.yaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
            axis.tick_params(labelbottom=False, labelleft=False, top=True, right=True, direction=tick_direction,
                             length=tick_length, width=tick_width)
        if plot_colorbar:
            plt.colorbar(plot, ax=axis)
        plt.tight_layout()

    plt.pause(0.5)
    plt.ioff()

    return plt.gcf()


def contour_slices(array, q_coordinates, sum_frames=False, levels=150, width_z=np.nan, width_y=np.nan, width_x=np.nan,
                   plot_colorbar=False, cmap=my_cmap, title='', scale='linear', is_orthogonal=False,
                   reciprocal_space=True):
    """
    Create a figure with three 2D contour plots from a 3D dataset.

    :param array: 3D array of real numbers
    :param q_coordinates: a tuple of (qx, qz, qy) 1D-coordinates corresponding to the (Z, Y, X) of the cxi convention
    :param sum_frames: if True, will sum the data along the 3rd axis
    :param levels: int n, will use n data intervals and draw n+1 contour lines
    :param width_z: user-defined zoom width along axis 0 (rocking angle), should be smaller than the actual data size
    :param width_y: user-defined zoom width along axis 1 (vertical), should be smaller than the actual data size
    :param width_x: user-defined zoom width along axis 2 (horizontal), should be smaller than the actual data size
    :param plot_colorbar: set it to True in order to plot the colorbar
    :param cmap: colormap to be used
    :param title: string to include in the plot
    :param scale: 'linear' or 'log'
    :param is_orthogonal: set to True is the frame is orthogonal, False otherwise (detector frame)
    :param reciprocal_space: True if the data is in reciprocal space, False otherwise
    :return: fig, (ax0, ax1, ax2, ax3), (plt0, plt1, plt2) instances
    """
    nb_dim = array.ndim
    plt.ion()
    if sum_frames:
        title = title + ' sum'

    if reciprocal_space:
        if is_orthogonal:
            slice_names = (' QyQz', ' QyQx', ' QzQx')
        else:  # detector frame
            slice_names = (' XY', ' X_RockingAngle', ' Y_RockingAngle')
    else:
        if is_orthogonal:
            slice_names = (' xy', ' xz', ' yz')
        else:  # detector frame
            slice_names = (' XY', ' X_RockingAngle', ' Y_RockingAngle')

    if nb_dim != 3:  # wrong array dimension
        print('multislices_plot() needs a 3D array')
        return
    else:
        nbz, nby, nbx = array.shape
        qx, qz, qy = q_coordinates
        if len(qx) != nbz or len(qz) != nby or len(qy) != nbx:
            print('Coordinates shape is not compatible with data shape')

        if np.isnan(width_z):
            width_z = nbz
        if np.isnan(width_y):
            width_y = nby
        if np.isnan(width_x):
            width_x = nbx

        fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2, figsize=(13, 9))

        # axis 0
        temp_array = np.copy(array)
        if not sum_frames:
            temp_array = temp_array[nbz // 2, :, :]
        else:
            temp_array = temp_array.sum(axis=0)
        # now array is 2D
        temp_array = temp_array[int(np.rint(nby / 2 - min(width_y, nby) / 2)):
                                int(np.rint(nby / 2 - min(width_y, nby) / 2)) + min(width_y, nby),
                                int(np.rint(nbx // 2 - min(width_x, nbx) // 2)):
                                int(np.rint(nbx // 2 - min(width_x, nbx) // 2)) + min(width_x, nbx)]

        if scale == 'linear':
            plt0 = ax0.contourf(qy, qz, temp_array, levels, cmap=cmap)
        elif scale == 'log':
            plt0 = ax0.contourf(qy, qz, np.log10(abs(temp_array)), levels, cmap=cmap)
        else:
            raise ValueError('Wrong value for scale')

        ax0.set_aspect("equal")
        ax0.set_title(title + slice_names[0])
        if plot_colorbar:
            plt.colorbar(plt0, ax=ax0)

        # axis 1
        temp_array = np.copy(array)
        if not sum_frames:
            temp_array = temp_array[:, nby // 2, :]
        else:
            temp_array = temp_array.sum(axis=1)
        # now array is 2D
        temp_array = temp_array[int(np.rint(nbz / 2 - min(width_z, nbz) / 2)):
                                int(np.rint(nbz / 2 - min(width_z, nbz) / 2)) + min(width_z, nbz),
                                int(np.rint(nbx // 2 - min(width_x, nbx) // 2)):
                                int(np.rint(nbx // 2 - min(width_x, nbx) // 2)) + min(width_x, nbx)]

        if scale == 'linear':
            plt1 = ax1.contourf(qy, qx, temp_array, levels, cmap=cmap)
        elif scale == 'log':
            plt1 = ax1.contourf(qy, qx, np.log10(abs(temp_array)), levels, cmap=cmap)
        else:
            raise ValueError('Wrong value for scale')

        ax1.set_aspect("equal")
        ax1.set_title(title + slice_names[1])
        if plot_colorbar:
            plt.colorbar(plt1, ax=ax1)

        # axis 2
        temp_array = np.copy(array)
        if not sum_frames:
            temp_array = temp_array[:, :, nbx // 2]
        else:
            temp_array = temp_array.sum(axis=2)
        # now array is 2D
        temp_array = temp_array[int(np.rint(nbz / 2 - min(width_z, nbz) / 2)):
                                int(np.rint(nbz / 2 - min(width_z, nbz) / 2)) + min(width_z, nbz),
                                int(np.rint(nby // 2 - min(width_y, nby) // 2)):
                                int(np.rint(nby // 2 - min(width_y, nby) // 2)) + min(width_y, nby)]

        if scale == 'linear':
            plt2 = ax2.contourf(qz, qx, temp_array, levels, cmap=cmap)
        elif scale == 'log':
            plt2 = ax2.contourf(qz, qx, np.log10(abs(temp_array)), levels, cmap=cmap)
        else:
            raise ValueError('Wrong value for scale')

        ax2.set_aspect("equal")
        ax2.set_title(title + slice_names[2])
        if plot_colorbar:
            plt.colorbar(plt2, ax=ax2)

        # axis 3
        ax3.set_visible(False)
    # plt.tight_layout()
    plt.pause(0.5)
    plt.ioff()
    return fig, (ax0, ax1, ax2, ax3), (plt0, plt1, plt2)


def imshow_plot(array, sum_frames=False, sum_axis=0, width_v=np.nan, width_h=np.nan, plot_colorbar=False,
                vmin=np.nan, vmax=np.nan, cmap=my_cmap, title='', scale='linear',
                tick_direction='inout', tick_width=1, tick_length=3, pixel_spacing=np.nan,
                is_orthogonal=False, reciprocal_space=False):
    """
    2D imshow plot of a 2D or 3D dataset using user-defined parameters.

    :param array: 2D or 3D array of real numbers
    :param sum_frames: if True, will sum the data along sum_axis
    :param sum_axis: axis along which to sum
    :param width_v: user-defined zoom vertical width, should be smaller than the actual data size
    :param width_h: user-defined zoom horizontal width, should be smaller than the actual data size
    :param plot_colorbar: set it to True in order to plot the colorbar
    :param vmin: lower boundary for the colorbar
    :param vmax: higher boundary for the colorbar
    :param cmap: colormap to be used
    :param title: string to include in the plot
    :param scale: 'linear' or 'log'
    :param tick_direction: 'out', 'in', 'inout'
    :param tick_width: width of tickes in plots
    :param tick_length: length of tickes in plots
    :param pixel_spacing: pixel_spacing = desired tick_spacing (in nm) / voxel_size of the reconstruction(in nm)
    :param is_orthogonal: set to True is the frame is orthogonal, False otherwise (detector frame)
    :param reciprocal_space: True if the data is in reciprocal space, False otherwise
    :return:  fig, axis, plot instances
    """
    nb_dim = array.ndim
    plt.ion()
    fig, axis = plt.subplots(1, 1)

    if nb_dim == 3:
        if sum_frames:
            title = title + ' sum'

        if reciprocal_space:
            if is_orthogonal:
                invert_yaxis = True
                slice_names = (' QyQz', ' QyQx', ' QzQx')
            else:  # detector frame
                invert_yaxis = False
                slice_names = (' XY', ' X_RockingAngle', ' Y_RockingAngle')
        else:
            if is_orthogonal:
                invert_yaxis = True
                slice_names = (' xy', ' xz', ' yz')
            else:  # detector frame
                invert_yaxis = False
                slice_names = (' XY', ' X_RockingAngle', ' Y_RockingAngle')

        nbz, nby, nbx = array.shape
        if np.isnan(width_v):
            width_v = max(nbz, nby, nbx)
        if np.isnan(width_h):
            width_h = max(nbz, nby, nbx)

        if sum_axis == 0:
            dim_v = nby
            dim_h = nbx
            if not sum_frames:
                array = array[nbz // 2, :, :]
            else:
                array = array.sum(axis=sum_axis)
        elif sum_axis == 1:
            dim_v = nbz
            dim_h = nbx
            if not sum_frames:
                array = array[:, nby // 2, :]
            else:
                array = array.sum(axis=sum_axis)
        elif sum_axis == 2:
            dim_v = nbz
            dim_h = nby
            if not sum_frames:
                array = array[:, :, nbx // 2]
            else:
                array = array.sum(axis=sum_axis)
        else:
            print('sum_axis should be only equal to 0, 1 or 2')
            return
        slice_name = slice_names[sum_axis]

    elif nb_dim == 2:
        invert_yaxis = False
        nby, nbx = array.shape
        if np.isnan(width_v):
            width_v = max(nby, nbx)
        if np.isnan(width_h):
            width_h = max(nby, nbx)

        dim_v = nby
        dim_h = nbx
        slice_name = ''

    else:  # wrong array dimension
        print('imshow_plot() needs a 2D or 3D array')
        return

    # now array is 2D
    width_v = min(width_v, dim_v)
    width_h = min(width_h, dim_h)
    array = array[int(np.rint(dim_v/2 - width_v/2)):int(np.rint(dim_v/2 - width_v/2)) + width_v,
                  int(np.rint(dim_h//2 - width_h//2)):int(np.rint(dim_h//2 - width_h//2)) + width_h]

    if scale == 'linear':
        if np.isnan(vmin):
            tmp_array = np.copy(array)
            tmp_array[np.isnan(array)] = np.inf
            tmp_array[np.isinf(tmp_array)] = np.inf  # set -inf to +inf to find the min
            vmin = tmp_array.min()
        if np.isnan(vmax):
            tmp_array = np.copy(array)
            tmp_array[np.isnan(array)] = -1 * np.inf
            tmp_array[np.isinf(tmp_array)] = -1 * np.inf  # set +inf to -inf to find the max
            vmax = tmp_array.max()
        plot = axis.imshow(array, vmin=vmin, vmax=vmax, cmap=cmap)
    elif scale == 'log':
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
            tmp_array[np.isinf(tmp_array)] = -1 * np.inf  # set +inf to -inf to find the max
            vmax = np.log10(abs(tmp_array).max())
        plot = axis.imshow(np.log10(abs(array)), vmin=vmin, vmax=vmax, cmap=cmap)
    else:
        raise ValueError('Wrong value for scale')

    if invert_yaxis and sum_axis == 0:  # Y is axis 0, need to be flipped
        ax = plt.gca()
        ax.invert_yaxis()
    plt.title(title + slice_name)
    plt.axis('scaled')
    if not np.isnan(pixel_spacing):
        axis.xaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
        axis.yaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
        axis.tick_params(labelbottom=False, labelleft=False, top=True, right=True, direction=tick_direction,
                         length=tick_length, width=tick_width)
    if plot_colorbar:
        plt.colorbar(plot, ax=axis)
    plt.tight_layout()
    plt.pause(0.5)
    plt.ioff()
    return fig, axis, plot


def loop_thru_scan(key, data, figure, scale, dim, idx, savedir, cmap=my_cmap, vmin=None, vmax=None):
    """
    Update the plot while removing the parasitic diffraction intensity in 3D dataset

    :param key: the keyboard key which was pressed
    :param data: the 3D data array
    :param figure: the figure instance
    :param scale: 'linear' or 'log'
    :param dim: the axis currently under review (axis 0, 1 or 2)
    :param idx: the frame index in the current axis
    :param savedir: path of the directory for saving images
    :param cmap: colormap to be used
    :param vmin: the lower boundary for the colorbar
    :param vmax: the higher boundary for the colorbar
    :return: updated controls
    """
    if data.ndim != 3:
        raise ValueError('data should be a 3D array')

    nbz, nby, nbx = data.shape
    exit_flag = False
    if dim > 2:
        raise ValueError('dim should be 0, 1 or 2')

    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()

    myaxs = figure.gca()
    xmin, xmax = myaxs.get_xlim()
    ymin, ymax = myaxs.get_ylim()
    if key == 'u':  # show next frame
        idx = idx + 1
        figure.clear()
        if dim == 0:
            if idx > nbz - 1:
                idx = 0
            if scale == 'linear':
                plt.imshow(data[idx, :, :], vmin=vmin, vmax=vmax, cmap=cmap)
            else:  # 'log'
                plt.imshow(np.log10(data[idx, :, :]), vmin=vmin, vmax=vmax, cmap=cmap)
            plt.title("Frame " + str(idx + 1) + "/" + str(nbz) + "\n"
                      "q quit ; u next frame ; d previous frame ; p unzoom\n"
                      "right darker ; left brighter ; r save 2D frame")
            plt.colorbar()
        elif dim == 1:
            if idx > nby - 1:
                idx = 0
            if scale == 'linear':
                plt.imshow(data[:, idx, :], vmin=vmin, vmax=vmax, cmap=cmap)
            else:  # 'log'
                plt.imshow(np.log10(data[:, idx, :]), vmin=vmin, vmax=vmax, cmap=cmap)
            plt.title("Frame " + str(idx + 1) + "/" + str(nby) + "\n"
                      "q quit ; u next frame ; d previous frame ; p unzoom\n"
                      "right darker ; left brighter ; r save 2D frame")
            plt.colorbar()
        elif dim == 2:
            if idx > nbx - 1:
                idx = 0
            if scale == 'linear':
                plt.imshow(data[:, :, idx], vmin=vmin, vmax=vmax, cmap=cmap)
            else:  # 'log'
                plt.imshow(np.log10(data[:, :, idx]), vmin=vmin, vmax=vmax, cmap=cmap)
            plt.title("Frame " + str(idx + 1) + "/" + str(nbx) + "\n"
                      "q quit ; u next frame ; d previous frame ; p unzoom\n"
                      "right darker ; left brighter ; r save 2D frame")
            plt.colorbar()
        myaxs = figure.gca()
        myaxs.set_xlim([xmin, xmax])
        myaxs.set_ylim([ymin, ymax])
        plt.draw()

    elif key == 'd':  # show previous frame
        idx = idx - 1
        figure.clear()
        if dim == 0:
            if idx < 0:
                idx = nbz - 1
            if scale == 'linear':
                plt.imshow(data[idx, :, :], vmin=vmin, vmax=vmax, cmap=cmap)
            else:  # 'log'
                plt.imshow(np.log10(data[idx, :, :]), vmin=vmin, vmax=vmax, cmap=cmap)
            plt.title("Frame " + str(idx + 1) + "/" + str(nbz) + "\n"
                      "q quit ; u next frame ; d previous frame ; p unzoom\n"
                      "right darker ; left brighter ; r save 2D frame")
            plt.colorbar()
        elif dim == 1:
            if idx < 0:
                idx = nby - 1
            if scale == 'linear':
                plt.imshow(data[:, idx, :], vmin=vmin, vmax=vmax, cmap=cmap)
            else:  # 'log'
                plt.imshow(np.log10(data[:, idx, :]), vmin=vmin, vmax=vmax, cmap=cmap)
            plt.title("Frame " + str(idx + 1) + "/" + str(nby) + "\n"
                      "q quit ; u next frame ; d previous frame ; p unzoom\n"
                      "right darker ; left brighter ; r save 2D frame")
            plt.colorbar()
        elif dim == 2:
            if idx < 0:
                idx = nbx - 1
            if scale == 'linear':
                plt.imshow(data[:, :, idx], vmin=vmin, vmax=vmax, cmap=cmap)
            else:  # 'log'
                plt.imshow(np.log10(data[:, :, idx]), vmin=vmin, vmax=vmax, cmap=cmap)
            plt.title("Frame " + str(idx + 1) + "/" + str(nbx) + "\n"
                      "q quit ; u next frame ; d previous frame ; p unzoom\n"
                      "right darker ; left brighter ; r save 2D frame")
            plt.colorbar()
        myaxs = figure.gca()
        myaxs.set_xlim([xmin, xmax])
        myaxs.set_ylim([ymin, ymax])
        plt.draw()

    elif key == 'right':  # increase colobar max
        vmax = vmax * 2
        figure.clear()
        if dim == 0:
            if scale == 'linear':
                plt.imshow(data[idx, :, :], vmin=vmin, vmax=vmax, cmap=cmap)
            else:  # 'log'
                plt.imshow(np.log10(data[idx, :, :]), vmin=vmin, vmax=vmax, cmap=cmap)
            plt.title("Frame " + str(idx + 1) + "/" + str(nbz) + "\n"
                      "q quit ; u next frame ; d previous frame ; p unzoom\n"
                      "right darker ; left brighter ; r save 2D frame")
            plt.colorbar()
        elif dim == 1:
            if scale == 'linear':
                plt.imshow(data[:, idx, :], vmin=vmin, vmax=vmax, cmap=cmap)
            else:  # 'log'
                plt.imshow(np.log10(data[:, idx, :]), vmin=vmin, vmax=vmax, cmap=cmap)
            plt.title("Frame " + str(idx + 1) + "/" + str(nby) + "\n"
                      "q quit ; u next frame ; d previous frame ; p unzoom\n"
                      "right darker ; left brighter ; r save 2D frame")
            plt.colorbar()
        elif dim == 2:
            if scale == 'linear':
                plt.imshow(data[:, :, idx], vmin=vmin, vmax=vmax, cmap=cmap)
            else:  # 'log'
                plt.imshow(np.log10(data[:, :, idx]), vmin=vmin, vmax=vmax, cmap=cmap)
            plt.title("Frame " + str(idx + 1) + "/" + str(nbx) + "\n"
                      "q quit ; u next frame ; d previous frame ; p unzoom\n"
                      "right darker ; left brighter ; r save 2D frame")
            plt.colorbar()
        myaxs = figure.gca()
        myaxs.set_xlim([xmin, xmax])
        myaxs.set_ylim([ymin, ymax])
        plt.draw()

    elif key == 'left':  # reduce colobar max
        vmax = vmax / 2
        if vmax < 1:
            vmax = 1
        figure.clear()
        if dim == 0:
            if scale == 'linear':
                plt.imshow(data[idx, :, :], vmin=vmin, vmax=vmax, cmap=cmap)
            else:  # 'log'
                plt.imshow(np.log10(data[idx, :, :]), vmin=vmin, vmax=vmax, cmap=cmap)
            plt.title("Frame " + str(idx + 1) + "/" + str(nbz) + "\n"
                      "q quit ; u next frame ; d previous frame ; p unzoom\n"
                      "right darker ; left brighter ; r save 2D frame")
            plt.colorbar()
        elif dim == 1:
            if scale == 'linear':
                plt.imshow(data[:, idx, :], vmin=vmin, vmax=vmax, cmap=cmap)
            else:  # 'log'
                plt.imshow(np.log10(data[:, idx, :]), vmin=vmin, vmax=vmax, cmap=cmap)
            plt.title("Frame " + str(idx + 1) + "/" + str(nby) + "\n"
                      "q quit ; u next frame ; d previous frame ; p unzoom\n"
                      "right darker ; left brighter ; r save 2D frame")
            plt.colorbar()
        elif dim == 2:
            if scale == 'linear':
                plt.imshow(data[:, :, idx], vmin=vmin, vmax=vmax, cmap=cmap)
            else:  # 'log'
                plt.imshow(np.log10(data[:, :, idx]), vmin=vmin, vmax=vmax, cmap=cmap)
            plt.title("Frame " + str(idx + 1) + "/" + str(nbx) + "\n"
                      "q quit ; u next frame ; d previous frame ; p unzoom\n"
                      "right darker ; left brighter ; r save 2D frame")
            plt.colorbar()
        myaxs = figure.gca()
        myaxs.set_xlim([xmin, xmax])
        myaxs.set_ylim([ymin, ymax])
        plt.draw()

    elif key == 'p':  # plot full image
        figure.clear()
        if dim == 0:
            if scale == 'linear':
                plt.imshow(data[idx, :, ], vmin=vmin, vmax=vmax, cmap=cmap)
            else:  # 'log'
                plt.imshow(np.log10(data[idx, :, ]), vmin=vmin, vmax=vmax, cmap=cmap)
            plt.title("Frame " + str(idx + 1) + "/" + str(nbz) + "\n"
                      "q quit ; u next frame ; d previous frame ; p unzoom\n"
                      "right darker ; left brighter ; r save 2D frame")
            plt.colorbar()
        elif dim == 1:
            if scale == 'linear':
                plt.imshow(data[:, idx, :], vmin=vmin, vmax=vmax, cmap=cmap)
            else:  # 'log'
                plt.imshow(np.log10(data[:, idx, :]), vmin=vmin, vmax=vmax, cmap=cmap)
            plt.title("Frame " + str(idx + 1) + "/" + str(nby) + "\n"
                      "q quit ; u next frame ; d previous frame ; p unzoom\n"
                      "right darker ; left brighter ; r save 2D frame")
            plt.colorbar()
        elif dim == 2:
            if scale == 'linear':
                plt.imshow(data[:, :, idx], vmin=vmin, vmax=vmax, cmap=cmap)
            else:  # 'log'
                plt.imshow(np.log10(data[:, :, idx]), vmin=vmin, vmax=vmax, cmap=cmap)
            plt.title("Frame " + str(idx + 1) + "/" + str(nbx) + "\n"
                      "q quit ; u next frame ; d previous frame ; p unzoom\n"
                      "right darker ; left brighter ; r save 2D frame")
            plt.colorbar()
        plt.draw()

    elif key == 'q':
        exit_flag = True

    elif key == 'r':
        filename = 'frame' + str(idx) + '_dim' + str(dim) + '.png'
        plt.savefig(savedir + filename)
    return vmax, idx, exit_flag


def multislices_plot(array, sum_frames=False, slice_position=None, width_z=None, width_y=None, width_x=None,
                     plot_colorbar=False, cmap=my_cmap, title='', scale='linear', vmin=np.nan, vmax=np.nan,
                     tick_direction='inout', tick_width=1, tick_length=3, pixel_spacing=None,
                     is_orthogonal=False, reciprocal_space=False):
    """
    Create a figure with three 2D imshow plots from a 3D dataset.

    :param array: 3D array of real numbers
    :param sum_frames: if True, will sum the data along the 3rd axis
    :param slice_position: tuple of three integers where to slice the 3D array
    :param width_z: zoom width along axis 0 (rocking angle), should be smaller than the actual data size
    :param width_y: zoom width along axis 1 (vertical), should be smaller than the actual data size
    :param width_x: zoom width along axis 2 (horizontal), should be smaller than the actual data size
    :param plot_colorbar: set it to True in order to plot the colorbar
    :param cmap: colormap to be used
    :param title: string to include in the plot
    :param scale: 'linear' or 'log'
    :param tick_direction: 'out', 'in', 'inout'
    :param tick_width: width of tickes in plots
    :param tick_length: length of tickes in plots
    :param pixel_spacing: pixel_spacing=desired tick_spacing (in nm)/voxel_size of the reconstruction(in nm)
    :param is_orthogonal: set to True is the frame is orthogonal, False otherwise (detector frame)
    :param reciprocal_space: True if the data is in reciprocal space, False otherwise. Used for plot titles.
    :param vmin: lower boundary for the colorbar. Float or tuple of 3 floats
    :param vmax: higher boundary for the colorbar. Float or tuple of 3 floats
    :return: fig, (ax0, ax1, ax2, ax3), (plt0, plt1, plt2) instances
    """
    nb_dim = array.ndim
    if nb_dim != 3:
        raise ValueError('multislices_plot() expects a 3D array')

    nbz, nby, nbx = array.shape

    try:
        if len(vmin) == 3:
            min_value = vmin
        else:
            raise ValueError('wrong shape for the parameter vmin')
    except TypeError:  # case len(vmin)=1
        min_value = [vmin, vmin, vmin]

    try:
        if len(vmax) == 3:
            max_value = vmax
        else:
            raise ValueError('wrong shape for the parameter vmax')
    except TypeError:  # case len(vmax)=1
        max_value = [vmax, vmax, vmax]

    if not sum_frames:
        if slice_position is None:
            slice_position = [nbz//2, nby//2, nbx//2]
        elif len(slice_position) != 3:
            raise ValueError('wrong shape for the parameter slice_position')
        else:
            slice_position = [int(position) for position in slice_position]

    plt.ion()
    if sum_frames:
        title = title + ' sum'
    if reciprocal_space:
        if is_orthogonal:
            invert_yaxis = True
            slice_names = (' QyQz', ' QyQx', ' QzQx')
        else:  # detector frame
            invert_yaxis = False
            slice_names = (' XY', ' X_RockingAngle', ' Y_RockingAngle')
    else:
        if is_orthogonal:
            invert_yaxis = True
            slice_names = (' xy', ' xz', ' yz')
        else:  # detector frame
            invert_yaxis = False
            slice_names = (' XY', ' X_RockingAngle', ' Y_RockingAngle')

    if width_z is None:
        width_z = nbz
    if width_y is None:
        width_y = nby
    if width_x is None:
        width_x = nbx

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2, figsize=(13, 9))

    # axis 0
    temp_array = np.copy(array)
    if not sum_frames:
        temp_array = temp_array[slice_position[0], :, :]
    else:
        temp_array = temp_array.sum(axis=0)
    # now array is 2D
    temp_array = temp_array[int(np.rint(nby // 2 - min(width_y, nby) // 2)):
                            int(np.rint(nby // 2 - min(width_y, nby) // 2)) + min(width_y, nby),
                            int(np.rint(nbx // 2 - min(width_x, nbx) // 2)):
                            int(np.rint(nbx // 2 - min(width_x, nbx) // 2)) + min(width_x, nbx)]

    if scale == 'linear':
        if np.isnan(min_value[0]):
            min_value[0] = temp_array[~np.isnan(temp_array)].min()
        if np.isnan(max_value[0]):
            max_value[0] = temp_array[~np.isnan(temp_array)].max()
        plt0 = ax0.imshow(temp_array, vmin=min_value[0], vmax=max_value[0], cmap=cmap)
    elif scale == 'log':
        if np.isnan(min_value[0]):
            min_value[0] = np.log10(abs(temp_array[~np.isnan(temp_array)]).min())
            if np.isinf(min_value[0]):
                min_value[0] = 0
        if np.isnan(max_value[0]):
            max_value[0] = np.log10(abs(temp_array[~np.isnan(temp_array)]).max())
        plt0 = ax0.imshow(np.log10(abs(temp_array)), vmin=min_value[0], vmax=max_value[0], cmap=cmap)
    else:
        raise ValueError('Wrong value for scale')

    ax0.set_title(title + slice_names[0])
    if invert_yaxis:  # Y is axis 0, need to be flipped
        ax0.invert_yaxis()
    plt.axis('scaled')
    if plot_colorbar:
        plt.colorbar(plt0, ax=ax0)
    if pixel_spacing is not None:
        ax0.xaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
        ax0.yaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
        ax0.tick_params(labelbottom=False, labelleft=False, top=True, right=True, direction=tick_direction,
                        length=tick_length, width=tick_width)

    # axis 1
    temp_array = np.copy(array)
    if not sum_frames:
        temp_array = temp_array[:, slice_position[1], :]
    else:
        temp_array = temp_array.sum(axis=1)
    # now array is 2D
    temp_array = temp_array[int(np.rint(nbz // 2 - min(width_z, nbz) // 2)):
                            int(np.rint(nbz // 2 - min(width_z, nbz) // 2)) + min(width_z, nbz),
                            int(np.rint(nbx // 2 - min(width_x, nbx) // 2)):
                            int(np.rint(nbx // 2 - min(width_x, nbx) // 2)) + min(width_x, nbx)]

    if scale == 'linear':
        if np.isnan(min_value[1]):
            min_value[1] = temp_array[~np.isnan(temp_array)].min()
        if np.isnan(max_value[1]):
            max_value[1] = temp_array[~np.isnan(temp_array)].max()
        plt1 = ax1.imshow(temp_array, vmin=min_value[1], vmax=max_value[1], cmap=cmap)
    elif scale == 'log':
        if np.isnan(min_value[1]):
            min_value[1] = np.log10(abs(temp_array[~np.isnan(temp_array)]).min())
            if np.isinf(min_value[1]):
                min_value[1] = 0
        if np.isnan(max_value[1]):
            max_value[1] = np.log10(abs(temp_array[~np.isnan(temp_array)]).max())
        plt1 = ax1.imshow(np.log10(abs(temp_array)), vmin=min_value[1], vmax=max_value[1], cmap=cmap)
    else:
        raise ValueError('Wrong value for scale')
    ax1.set_title(title + slice_names[1])
    plt.axis('scaled')
    if plot_colorbar:
        plt.colorbar(plt1, ax=ax1)
    if pixel_spacing is not None:
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
        ax1.tick_params(labelbottom=False, labelleft=False, top=True, right=True, direction=tick_direction,
                        length=tick_length, width=tick_width)

    # axis 2
    temp_array = np.copy(array)
    if not sum_frames:
        temp_array = temp_array[:, :, slice_position[2]]
    else:
        temp_array = temp_array.sum(axis=2)
    # now array is 2D
    temp_array = temp_array[int(np.rint(nbz // 2 - min(width_z, nbz) // 2)):
                            int(np.rint(nbz // 2 - min(width_z, nbz) // 2)) + min(width_z, nbz),
                            int(np.rint(nby // 2 - min(width_y, nby) // 2)):
                            int(np.rint(nby // 2 - min(width_y, nby) // 2)) + min(width_y, nby)]

    if scale == 'linear':
        if np.isnan(min_value[2]):
            min_value[2] = temp_array[~np.isnan(temp_array)].min()
        if np.isnan(max_value[2]):
            max_value[2] = temp_array[~np.isnan(temp_array)].max()
        plt2 = ax2.imshow(temp_array, vmin=min_value[2], vmax=max_value[2], cmap=cmap)
    elif scale == 'log':
        if np.isnan(min_value[2]):
            min_value[2] = np.log10(abs(temp_array[~np.isnan(temp_array)]).min())
            if np.isinf(min_value[2]):
                min_value[2] = 0
        if np.isnan(max_value[2]):
            max_value[2] = np.log10(abs(temp_array[~np.isnan(temp_array)]).max())
        plt2 = ax2.imshow(np.log10(abs(temp_array)), vmin=min_value[2], vmax=max_value[2], cmap=cmap)
    else:
        raise ValueError('Wrong value for scale')

    ax2.set_title(title + slice_names[2])
    plt.axis('scaled')

    if plot_colorbar:
        plt.colorbar(plt2, ax=ax2)
    if pixel_spacing is not None:
        ax2.xaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
        ax2.yaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
        ax2.tick_params(labelbottom=False, labelleft=False, top=True, right=True, direction=tick_direction,
                        length=tick_length, width=tick_width)

    # axis 3
    ax3.set_visible(False)

    plt.tight_layout()
    plt.pause(0.5)
    plt.ioff()
    return fig, (ax0, ax1, ax2, ax3), (plt0, plt1, plt2)


def plot_3dmesh(vertices, faces, data_shape, title='Mesh - z axis flipped because of CXI convention'):
    """
    Plot a 3D mesh defined by its vertices and faces.

    :param vertices: n*3 ndarray of n vertices defined by 3 positions
    :param faces: m*3 ndarray of m faces defined by 3 indices of vertices
    :param data_shape: tuple corresponding to the 3d data shape
    :param title: title for the plot
    :return: figure and axe instances
    """
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    plt.ion()
    fig = plt.figure(figsize=(10, 10))
    ax0 = Axes3D(fig)
    mymesh = Poly3DCollection(vertices[faces])
    mymesh.set_edgecolor('k')
    ax0.add_collection3d(mymesh)
    ax0.set_xlim(0, data_shape[0])
    ax0.set_xlabel('Z')
    ax0.set_ylim(0, data_shape[1])
    ax0.set_ylabel('Y')
    ax0.set_zlim(0, data_shape[2])
    ax0.set_zlabel('X')
    plt.title(title)

    plt.pause(0.5)
    plt.ioff()
    return fig, ax0


def plot_stereographic(euclidian_u, euclidian_v, color, radius_mean, planes={}, title="", plot_planes=True):
    """
    Plot the stereographic projection with some cosmetics.

    :param euclidian_u: normalized Euclidian metric coordinate
    :param euclidian_v: normalized Euclidian metric coordinate
    :param color: intensity of density kernel estimation at radius_mean
    :param radius_mean: radius of the sphere in reciprocal space from which the projection is done
    :param planes: dictionnary of crystallographic planes, e.g. {'111':angle_with_reflection}
    :param title: title for the stereographic plot
    :param plot_planes: if True, will draw circle corresponding to crystallographic planes in the pole figure
    :return: figure and axe instances
    """
    from scipy.interpolate import griddata

    u_grid, v_grid = np.mgrid[-91:91:183j, -91:91:183j]  # vertical, horizontal
    intensity_grid = griddata((euclidian_u, euclidian_v), color, (u_grid, v_grid), method='linear')
    intensity_grid = intensity_grid / intensity_grid[intensity_grid > 0].max() * 10000  # normalize for easier plotting

    # plot the stereographic projection
    plt.ion()
    fig, ax0 = plt.subplots(1, 1, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
    plt0 = ax0.contourf(u_grid, v_grid, abs(intensity_grid), range(100, 6100, 250), cmap='hsv')
    plt.colorbar(plt0, ax=ax0)
    ax0.axis('equal')
    ax0.axis('off')

    # add the projection of the elevation angle, depending on the center of projection
    for ii in range(15, 90, 5):
        circle =\
            patches.Circle((0, 0), radius_mean * np.sin(ii * np.pi / 180) /
                           (1 + np.cos(ii * np.pi / 180)) * 90 / radius_mean,
                           color='grey', fill=False, linestyle='dotted', linewidth=0.5)
        ax0.add_artist(circle)
    for ii in range(10, 90, 20):
        circle =\
            patches.Circle((0, 0), radius_mean * np.sin(ii * np.pi / 180) /
                           (1 + np.cos(ii * np.pi / 180)) * 90 / radius_mean,
                           color='grey', fill=False, linestyle='dotted', linewidth=1)
        ax0.add_artist(circle)
    for ii in range(10, 95, 20):
        ax0.text(-radius_mean * np.sin(ii * np.pi / 180) / (1 + np.cos(ii * np.pi / 180)) * 90 / radius_mean, 0,
                 str(ii) + '$^\circ$', fontsize=10, color='k')
    circle = patches.Circle((0, 0), 90, color='k', fill=False, linewidth=1.5)
    ax0.add_artist(circle)

    # add azimutal lines every 5 and 45 degrees
    for ii in range(5, 365, 5):
        ax0.plot([0, 90 * np.cos(ii * np.pi / 180)], [0, 90 * np.sin(ii * np.pi / 180)], color='grey',
                 linestyle='dotted', linewidth=0.5)
    for ii in range(0, 365, 20):
        ax0.plot([0, 90 * np.cos(ii * np.pi / 180)], [0, 90 * np.sin(ii * np.pi / 180)], color='grey',
                 linestyle='dotted', linewidth=1)

    # draw circles corresponding to particular reflection
    if plot_planes == 1 and len(planes) != 0:
        indx = 0
        for key, value in planes.items():
            circle = patches.Circle((0, 0), radius_mean * np.sin(value * np.pi / 180) /
                                    (1 + np.cos(value * np.pi / 180)) * 90 / radius_mean,
                                    color='g', fill=False, linestyle='dotted', linewidth=1.5)
            ax0.add_artist(circle)
            ax0.text(np.cos(indx * np.pi / 180) * radius_mean * np.sin(value * np.pi / 180) /
                     (1 + np.cos(value * np.pi / 180)) * 90 / radius_mean,
                     np.sin(indx * np.pi / 180) * radius_mean * np.sin(value * np.pi / 180) /
                     (1 + np.cos(value * np.pi / 180)) * 90 / radius_mean,
                     key, fontsize=10, color='k', fontweight='bold')
            indx = indx + 6
            print(key + ": ", str('{:.2f}'.format(value)))
        print('\n')
    ax0.set_title('Projection\nfrom ' + title)
    plt.pause(0.5)
    plt.ioff()
    return fig, ax0


def save_to_vti(filename, voxel_size, tuple_array, tuple_fieldnames, origin=(0, 0, 0), amplitude_threshold=0.01):
    """
    Save arrays defined by their name in a single vti file.

    :param filename: the file name of the vti file
    :param voxel_size: tuple (voxel_size_axis0, voxel_size_axis1, voxel_size_axis2)
    :param tuple_array: tuple of arrays of the same dimension
    :param tuple_fieldnames: tuple of name containing the same number of elements as tuple_array
    :param origin: tuple of points for vtk SetOrigin()
    :param amplitude_threshold: lower threshold for saving the reconstruction modulus (save memory space)
    :return: nothing
    """
    import vtk
    from vtk.util import numpy_support

    if type(tuple_fieldnames) is tuple:
        nb_fieldnames = len(tuple_fieldnames)
    elif type(tuple_fieldnames) is str:
        nb_fieldnames = 1
    else:
        raise TypeError('Invalid input for tuple_fieldnames')

    if type(tuple_array) is tuple:
        nb_arrays = len(tuple_array)
        nb_dim = tuple_array[0].ndim
        if nb_dim != 3:  # wrong array dimension
            raise ValueError('save_to_vti() needs a 3D array')
        nbz, nby, nbx = tuple_array[0].shape
    elif type(tuple_array) is np.ndarray:
        nb_arrays = 1
        nb_dim = tuple_array.ndim
        if nb_dim != 3:  # wrong array dimension
            raise ValueError('save_to_vti() needs a 3D array')
        nbz, nby, nbx = tuple_array.shape
    else:
        raise TypeError('Invalid input for tuple_array')

    if nb_arrays != nb_fieldnames:
        print('Different number of arrays and field names')
        return

    image_data = vtk.vtkImageData()
    image_data.SetOrigin(origin[0], origin[1], origin[2])
    image_data.SetSpacing(voxel_size[0], voxel_size[1], voxel_size[2])
    image_data.SetExtent(0, nbz - 1, 0, nby - 1, 0, nbx - 1)

    try:
        amp_index = tuple_fieldnames.index('amp')  # look for the substring 'amp'
        if nb_arrays > 1:
            amp_array = tuple_array[amp_index]
        else:
            amp_array = tuple_array
        amp_array = amp_array / amp_array.max()
        amp_array[amp_array < amplitude_threshold] = 0  # save disk space
        amp_array = np.transpose(np.flip(amp_array, 2)).reshape(amp_array.size)
        amp_array = numpy_support.numpy_to_vtk(amp_array)
        pd = image_data.GetPointData()
        pd.SetScalars(amp_array)
        pd.GetArray(0).SetName("amp")
        counter = 1
        if nb_arrays > 1:
            for idx in range(nb_arrays):
                if idx == amp_index:
                    continue
                temp_array = tuple_array[idx]
                temp_array[amp_array == 0] = 0
                temp_array = np.transpose(np.flip(temp_array, 2)).reshape(temp_array.size)
                temp_array = numpy_support.numpy_to_vtk(temp_array)
                pd.AddArray(temp_array)
                pd.GetArray(counter).SetName(tuple_fieldnames[idx])
                pd.Update()
                counter = counter + 1
    except ValueError:
        print('amp not in fieldnames, will save arrays without thresholding')
        if nb_arrays > 1:
            temp_array = tuple_array[0]
        else:
            temp_array = tuple_array
        temp_array = np.transpose(np.flip(temp_array, 2)).reshape(temp_array.size)
        temp_array = numpy_support.numpy_to_vtk(temp_array)
        pd = image_data.GetPointData()
        pd.SetScalars(temp_array)
        if nb_arrays > 1:
            pd.GetArray(0).SetName(tuple_fieldnames[0])
            for idx in range(1, nb_arrays):
                temp_array = tuple_array[idx]
                temp_array = np.transpose(np.flip(temp_array, 2)).reshape(temp_array.size)
                temp_array = numpy_support.numpy_to_vtk(temp_array)
                pd.AddArray(temp_array)
                pd.GetArray(idx).SetName(tuple_fieldnames[idx])
                pd.Update()
        else:
            pd.GetArray(0).SetName(tuple_fieldnames)

    # export data to file
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(image_data)
    writer.Write()
    return


def scatter_plot(array, labels, markersize=4, markercolor='b', title=''):
    """
    2D or 3D Scatter plot of a 2D ndarray.

    :param array: 2D ndarray, the number of columns is the number of dimensions of the scatter plot (2 or 3)
    :param labels: tuple of string labels (length = number of columns in array)
    :param markersize: number corresponding to the marker size
    :param markercolor: string corresponding to the marker color
    :param title: string, title for the scatter plot
    :return: figure, axes instances
    """
    if array.ndim != 2:
        raise ValueError('array should be 2D')
    ndim = array.shape[1]
    if not isinstance(labels, tuple):
        labels = (labels,) * ndim

    if len(labels) != ndim:
        raise ValueError('the number of labels is different from the number of columns in the array')
    plt.ion()
    fig = plt.figure()

    if ndim == 2:
        ax = plt.subplot(111)
        ax.scatter(array[:, 0], array[:, 1], s=markersize, color=markercolor)
        plt.title(title)
        ax.set_xlabel(labels[0])  # first dimension is x for scatter plots, but z for NEXUS convention
        ax.set_ylabel(labels[1])
        plt.pause(0.1)
    elif ndim == 3:
        ax = plt.subplot(111, projection='3d')
        ax.scatter(array[:, 0], array[:, 1], array[:, 2], s=markersize, color=markercolor)
        plt.title(title)
        ax.set_xlabel(labels[0])  # first dimension is x for scatter plots, but z for NEXUS convention
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
    else:
        raise ValueError('There should be 2 or 3 columns in the array')
    if ndim == 2:
        plt.axis('scaled')
    plt.pause(0.5)
    plt.ioff()
    return fig, ax


def scatter_plot_overlaid(arrays, markersizes, markercolors, labels, title=''):
    """
    Overlaid scatter plot of 2D ndarrays having the same number of columns.

    :param arrays: tuple of 2D ndarrays, the number of columns is the number of dimensions of the scatter plot (2 or 3)
    :param markersizes: tuple of numbers corresponding to the marker sizes (length = number of arrays)
    :param markercolors: tuple of strings corresponding to the marker color (length = number of arrays)
    :param labels: tuple of string labels (length = number of columns in arrays)
    :param title: string, title for the scatter plot
    :return: figure, axes instances
    """
    if not isinstance(arrays, tuple):
        fig, ax = scatter_plot(array=arrays, markersize=markersizes, markercolor=markercolors, labels=labels,
                               title=title)
        return fig, ax

    if arrays[0].ndim != 2:
        raise ValueError('arrays should be 2D')

    ndim = arrays[0].shape[1]
    nb_arrays = len(arrays)

    if not isinstance(labels, tuple):
        labels = (labels,) * ndim
    if not isinstance(markersizes, tuple):
        markersizes = (markersizes,) * ndim
    if not isinstance(markercolors, tuple):
        markercolors = (markercolors,) * ndim

    if len(labels) != ndim:
        raise ValueError('the number of labels is different from the number of columns in arrays')
    if len(markersizes) != nb_arrays:
        raise ValueError('the number of markersizes is different from the number of arrays')
    if len(markercolors) != nb_arrays:
        raise ValueError('the number of markercolors is different from the number of arrays')
    plt.ion()
    fig = plt.figure()
    if ndim == 2:
        ax = plt.subplot(111)
    elif ndim == 3:
        ax = plt.subplot(111, projection='3d')
    else:
        raise ValueError('There should be 2 or 3 columns in the array')

    for idx in range(nb_arrays):
        array = arrays[idx]
        if array.shape[1] != ndim:
            raise ValueError('All arrays should have the same number of columns')

        if ndim == 2:
            ax.scatter(array[:, 0], array[:, 1], s=markersizes[idx], color=markercolors[idx])
        else:  # 3D
            ax.scatter(array[:, 0], array[:, 1], array[:, 2], s=markersizes[idx], color=markercolors[idx])

    plt.title(title)
    if ndim == 2:
        ax.set_xlabel(labels[0])  # first dimension is x for scatter plots, but z for NEXUS convention
        ax.set_ylabel(labels[1])
    else:
        ax.set_xlabel(labels[0])  # first dimension is x for scatter plots, but z for NEXUS convention
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
    if ndim == 2:
        plt.axis('scaled')
    plt.pause(0.5)
    plt.ioff()
    return fig, ax


# if __name__ == "__main__":
#     datadir = 'D:/review paper/BCDI_isosurface/S2227/simu/crop300/test/'
#     strain = np.load(datadir +
#                      'S2227_ampphasestrain_1_gaussianthreshold_iso_0.68_avg1_apodize_crystal-frame.npz')['strain']
#     voxel_size = 4.0
#     tick_spacing = 50  # for plots, in nm
#     pixel_spacing = tick_spacing / voxel_size
#     tick_direction = 'inout'  # 'out', 'in', 'inout'
#     tick_length = 3  # 10  # in plots
#     tick_width = 1  # 2  # in plots
#     multislices_plot(strain, sum_frames=False, invert_yaxis=True, title='Orthogonal strain',
#                      vmin=-0.0002, vmax=0.0002, tick_direction=tick_direction,
#                      tick_width=tick_width, tick_length=tick_length, plot_colorbar=True,
#                      pixel_spacing=pixel_spacing)
#     plt.ioff()
#     plt.show()
