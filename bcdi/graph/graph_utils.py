# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#       authors:
#         Jerome Carnis, jerome.carnis@esrf.fr

import numpy as np
from matplotlib import pyplot as plt
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


def combined_plots(tuple_array, tuple_sum_frames, tuple_sum_axis, tuple_width_v, tuple_width_h, tuple_colorbar,
                   tuple_vmin, tuple_vmax, tuple_title, tuple_scale, cmap=my_cmap, tick_direction='inout',
                   tick_width=1, tick_length=3, pixel_spacing=np.nan, reciprocal_space=False, **kwargs):
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
    :param reciprocal_space: True if the data is in reciprocal space, False otherwise
    :param kwargs: optional 'xlabel' and 'ylabel', labels for plots
    :return:  the figure instance
    """
    if type(tuple_array) is not tuple:
        raise TypeError('Expected "tuple_array" to be a tuple')
    if type(tuple_sum_axis) is not tuple:
        raise TypeError('Expected "tuple_sum_axis" to be a tuple')

    nb_subplots = len(tuple_array)

    if type(tuple_sum_frames) is not tuple:
        tuple_sum_frames = (tuple_sum_frames,) * nb_subplots
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
                tmp_array = np.copy(array)
                tmp_array[np.isnan(array)] = np.inf
                tmp_array[np.isinf(tmp_array)] = np.inf  # set -inf to +inf to find the min
                vmin = tmp_array.min()
            if np.isnan(vmax):
                tmp_array = np.copy(array)
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
                slice_names = (' QyQz', ' QyQx', ' QzQx')
            else:
                slice_names = (' XY', ' XZ', ' YZ')

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
            axis.tick_params(labelbottom='off', labelleft='off', top='on', right='on', direction=tick_direction,
                             length=tick_length, width=tick_width)
        if plot_colorbar:
            plt.colorbar(plot, ax=axis)
        plt.tight_layout()

    plt.pause(0.5)
    plt.ioff()

    return plt.gcf()


def contour_slices(array, coordinates, sum_frames=False, levels=150, width_z=np.nan, width_y=np.nan, width_x=np.nan,
                   plot_colorbar=False, cmap=my_cmap, title='', scale='linear', reciprocal_space=True):
    """
    Create a figure with three 2D contour plots from a 3D dataset.

    :param array: 3D array of real numbers
    :param coordinates: a tuple of (qx, qz, qy) 1D-coordinates corresponding to the (Z, Y, X) of the cxi convention
    :param sum_frames: if True, will sum the data along the 3rd axis
    :param levels: int n, will use n data intervals and draw n+1 contour lines
    :param width_z: user-defined zoom width along axis 0 (rocking angle), should be smaller than the actual data size
    :param width_y: user-defined zoom width along axis 1 (vertical), should be smaller than the actual data size
    :param width_x: user-defined zoom width along axis 2 (horizontal), should be smaller than the actual data size
    :param plot_colorbar: set it to True in order to plot the colorbar
    :param cmap: colormap to be used
    :param title: string to include in the plot
    :param scale: 'linear' or 'log'
    :param reciprocal_space: True if the data is in reciprocal space, False otherwise
    :return: fig, (ax0, ax1, ax2, ax3), (plt0, plt1, plt2) instances
    """
    nb_dim = array.ndim
    plt.ion()
    if sum_frames:
        title = title + ' sum'

    if reciprocal_space:
        slice_names = (' QyQz', ' QyQx', ' QzQx')
    else:
        slice_names = (' XY', ' XZ', ' YZ')

    if nb_dim != 3:  # wrong array dimension
        print('multislices_plot() needs a 3D array')
        return
    else:
        nbz, nby, nbx = array.shape
        qx, qz, qy = coordinates
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

        plt.axis('scaled')
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
        plt.axis('scaled')
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

        plt.axis('scaled')
        ax2.set_title(title + slice_names[2])
        if plot_colorbar:
            plt.colorbar(plt2, ax=ax2)

        # axis 3
        ax3.set_visible(False)
    plt.tight_layout()
    plt.pause(0.5)
    plt.ioff()
    return fig, (ax0, ax1, ax2, ax3), (plt0, plt1, plt2)


def imshow_plot(array, sum_frames=False, sum_axis=0, width_v=np.nan, width_h=np.nan, plot_colorbar=False,
                vmin=np.nan, vmax=np.nan, cmap=my_cmap, title='', scale='linear',
                tick_direction='inout', tick_width=1, tick_length=3, pixel_spacing=np.nan, reciprocal_space=False):
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
            slice_names = (' QyQz', ' QyQx', ' QzQx')
        else:
            slice_names = (' XY', ' XZ', ' YZ')

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

    plt.title(title + slice_name)
    plt.axis('scaled')
    if not np.isnan(pixel_spacing):
        axis.xaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
        axis.yaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
        axis.tick_params(labelbottom='off', labelleft='off', top='on', right='on', direction=tick_direction,
                         length=tick_length, width=tick_width)
    if plot_colorbar:
        plt.colorbar(plot, ax=axis)
    plt.tight_layout()
    plt.pause(0.5)
    plt.ioff()
    return fig, axis, plot


def multislices_plot(array, sum_frames=False, width_z=np.nan, width_y=np.nan, width_x=np.nan, plot_colorbar=False,
                     cmap=my_cmap, title='', scale='linear', invert_yaxis=False, vmin=np.nan, vmax=np.nan,
                     tick_direction='inout', tick_width=1, tick_length=3, pixel_spacing=np.nan, reciprocal_space=False):
    """
    Create a figure with three 2D imshow plots from a 3D dataset.

    :param array: 3D array of real numbers
    :param sum_frames: if True, will sum the data along the 3rd axis
    :param width_z: zoom width along axis 0 (rocking angle), should be smaller than the actual data size
    :param width_y: zoom width along axis 1 (vertical), should be smaller than the actual data size
    :param width_x: zoom width along axis 2 (horizontal), should be smaller than the actual data size
    :param plot_colorbar: set it to True in order to plot the colorbar
    :param cmap: colormap to be used
    :param title: string to include in the plot
    :param scale: 'linear' or 'log'
    :param invert_yaxis: will invert the y axis in the XY slice
    :param tick_direction: 'out', 'in', 'inout'
    :param tick_width: width of tickes in plots
    :param tick_length: length of tickes in plots
    :param pixel_spacing: pixel_spacing=desired tick_spacing (in nm)/voxel_size of the reconstruction(in nm)
    :param reciprocal_space: True if the data is in reciprocal space, False otherwise
    :param vmin: lower boundary for the colorbar. Float or tuple of 3 floats
    :param vmax: higher boundary for the colorbar. Float or tuple of 3 floats
    :return: fig, (ax0, ax1, ax2, ax3), (plt0, plt1, plt2) instances
    """

    try:
        len_vmin = len(vmin)
        if len_vmin == 3:
            min_value = vmin
        else:
            raise ValueError('wrong shape for the parameter vmin')
    except TypeError:  # case len_vmin=1
        min_value = [vmin, vmin, vmin]

    try:
        len_vmax = len(vmax)
        if len_vmax == 3:
            max_value = vmax
        else:
            raise ValueError('wrong shape for the parameter vmax')
    except TypeError:  # case len_vmax=1
        max_value = [vmax, vmax, vmax]

    nb_dim = array.ndim
    plt.ion()
    if sum_frames:
        title = title + ' sum'
    if reciprocal_space:
        slice_names = (' QyQz', ' QyQx', ' QzQx')
    else:
        slice_names = (' XY', ' XZ', ' YZ')
    if nb_dim != 3:  # wrong array dimension
        raise ValueError('multislices_plot() needs a 3D array')
    else:
        nbz, nby, nbx = array.shape
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
        if invert_yaxis:
            ax0.invert_yaxis()
        plt.axis('scaled')
        if plot_colorbar:
            plt.colorbar(plt0, ax=ax0)
        if not np.isnan(pixel_spacing):
            ax0.xaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
            ax0.yaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
            ax0.tick_params(labelbottom='off', labelleft='off', top='on', right='on', direction=tick_direction,
                            length=tick_length, width=tick_width)

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
        if not np.isnan(pixel_spacing):
            ax1.xaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
            ax1.yaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
            ax1.tick_params(labelbottom='off', labelleft='off', top='on', right='on', direction=tick_direction,
                            length=tick_length, width=tick_width)

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
        if not np.isnan(pixel_spacing):
            ax2.xaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
            ax2.yaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
            ax2.tick_params(labelbottom='off', labelleft='off', top='on', right='on', direction=tick_direction,
                            length=tick_length, width=tick_width)

        # axis 3
        ax3.set_visible(False)
    plt.tight_layout()
    plt.pause(0.5)
    plt.ioff()
    return fig, (ax0, ax1, ax2, ax3), (plt0, plt1, plt2)

def multislices_plotv2(array, sum_frames=False, width_z=np.nan, width_y=np.nan, width_x=np.nan, plot_colorbar=False,
                     cmap=my_cmap, title='', scale='linear', invert_yaxis=False, vmin=np.nan, vmax=np.nan,
                     tick_direction='inout', tick_width=1, tick_length=3, pixel_spacing=np.nan, reciprocal_space=False):
    """
    Create a figure with three 2D imshow plots from a 3D dataset.

    :param array: 3D array of real numbers
    :param sum_frames: if True, will sum the data along the 3rd axis
    :param width_z: zoom width along axis 0 (rocking angle), should be smaller than the actual data size
    :param width_y: zoom width along axis 1 (vertical), should be smaller than the actual data size
    :param width_x: zoom width along axis 2 (horizontal), should be smaller than the actual data size
    :param plot_colorbar: set it to True in order to plot the colorbar
    :param cmap: colormap to be used
    :param title: string to include in the plot
    :param scale: 'linear' or 'log'
    :param invert_yaxis: will invert the y axis in the XY slice
    :param tick_direction: 'out', 'in', 'inout'
    :param tick_width: width of tickes in plots
    :param tick_length: length of tickes in plots
    :param pixel_spacing: pixel_spacing=desired tick_spacing (in nm)/voxel_size of the reconstruction(in nm)
    :param reciprocal_space: True if the data is in reciprocal space, False otherwise
    :param vmin: lower boundary for the colorbar. Float or tuple of 3 floats
    :param vmax: higher boundary for the colorbar. Float or tuple of 3 floats
    :return: fig, (ax0, ax1, ax2, ax3), (plt0, plt1, plt2) instances
    """

    try:
        len_vmin = len(vmin)
        if len_vmin == 3:
            min_value = vmin
        else:
            raise ValueError('wrong shape for the parameter vmin')
    except TypeError:  # case len_vmin=1
        min_value = [vmin, vmin, vmin]

    try:
        len_vmax = len(vmax)
        if len_vmax == 3:
            max_value = vmax
        else:
            raise ValueError('wrong shape for the parameter vmax')
    except TypeError:  # case len_vmax=1
        max_value = [vmax, vmax, vmax]

    nb_dim = array.ndim
    plt.ion()
    if sum_frames:
        title = title + ' sum'
    if reciprocal_space:
        slice_names = (' QyQz', ' QyQx', ' QzQx')
    else:
        slice_names = (' XY', ' XZ', ' YZ')
    if nb_dim != 3:  # wrong array dimension
        raise ValueError('multislices_plot() needs a 3D array')
    else:
        nbz, nby, nbx = array.shape
        if np.isnan(width_z):
            width_z = nbz
        if np.isnan(width_y):
            width_y = nby
        if np.isnan(width_x):
            width_x = nbx

        fig, ((ax0, ax1, ax2)) = plt.subplots(nrows=1, ncols=3, figsize=(15, 4.5))

        # axis 0
        temp_array = np.copy(array)
        if not sum_frames:
            temp_array = temp_array[nbz // 2, :, :]
        else:
            temp_array = temp_array.sum(axis=0)
        # now array is 2D
        """
        temp_array = temp_array[int(np.rint(nby / 2 - min(width_y, nby) / 2)):
                                int(np.rint(nby / 2 - min(width_y, nby) / 2)) + min(width_y, nby),
                                int(np.rint(nbx // 2 - min(width_x, nbx) // 2)):
                                int(np.rint(nbx // 2 - min(width_x, nbx) // 2)) + min(width_x, nbx)]
        """
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
        if invert_yaxis:
            ax0.invert_yaxis()
        plt.axis('scaled')
        if plot_colorbar:
            plt.colorbar(plt0, ax=ax0,fraction=0.04, pad= 0.1)
        if not np.isnan(pixel_spacing):
            ax0.xaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
            ax0.yaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
            ax0.tick_params(labelbottom='off', labelleft='off', top='on', right='on', direction=tick_direction,
                            length=tick_length, width=tick_width)

        # axis 1
        temp_array = np.copy(array)
        if not sum_frames:
            temp_array = temp_array[:, nby // 2, :]
        else:
            temp_array = temp_array.sum(axis=1)
        # now array is 2D
        """
        temp_array = temp_array[int(np.rint(nbz / 2 - min(width_z, nbz) / 2)):
                                int(np.rint(nbz / 2 - min(width_z, nbz) / 2)) + min(width_z, nbz),
                                int(np.rint(nbx // 2 - min(width_x, nbx) // 2)):
                                int(np.rint(nbx // 2 - min(width_x, nbx) // 2)) + min(width_x, nbx)]
		"""
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
            plt.colorbar(plt1, ax=ax1,fraction=0.04, pad= 0.1)
        if not np.isnan(pixel_spacing):
            ax1.xaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
            ax1.yaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
            ax1.tick_params(labelbottom='off', labelleft='off', top='on', right='on', direction=tick_direction,
                            length=tick_length, width=tick_width)

        # axis 2
        temp_array = np.copy(array)
        if not sum_frames:
            temp_array = temp_array[:, :, nbx // 2]
        else:
            temp_array = temp_array.sum(axis=2)
        # now array is 2D
        """
        temp_array = temp_array[int(np.rint(nbz / 2 - min(width_z, nbz) / 2)):
                                int(np.rint(nbz / 2 - min(width_z, nbz) / 2)) + min(width_z, nbz),
                                int(np.rint(nby // 2 - min(width_y, nby) // 2)):
                                int(np.rint(nby // 2 - min(width_y, nby) // 2)) + min(width_y, nby)]
        """
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
            plt.colorbar(plt2, ax=ax2,fraction=0.04, pad= 0.1)
        if not np.isnan(pixel_spacing):
            ax2.xaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
            ax2.yaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
            ax2.tick_params(labelbottom='off', labelleft='off', top='on', right='on', direction=tick_direction,
                            length=tick_length, width=tick_width)


    plt.tight_layout()
    plt.pause(0.5)
    plt.ioff()
    return fig, (ax0, ax1, ax2), (plt0, plt1, plt2)



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
    if type(tuple_array) is tuple:
        nb_arrays = len(tuple_array)
        nb_fieldnames = len(tuple_fieldnames)
        nb_dim = tuple_array[0].ndim
        if nb_dim != 3:  # wrong array dimension
            raise ValueError('save_to_vti() needs a 3D array')
        nbz, nby, nbx = tuple_array[0].shape
    else:  # a single numpy.ndarray
        nb_arrays = 1
        nb_fieldnames = 1
        nb_dim = tuple_array.ndim
        if nb_dim != 3:  # wrong array dimension
            raise ValueError('save_to_vti() needs a 3D array')
        nbz, nby, nbx = tuple_array.shape

    if nb_arrays != nb_fieldnames:
        print('Different number of arrays and field names')
        return

    image_data = vtk.vtkImageData()
    image_data.SetOrigin(origin[0], origin[1], origin[2])
    image_data.SetSpacing(voxel_size[0], voxel_size[1], voxel_size[2])
    image_data.SetExtent(0, nbz - 1, 0, nby - 1, 0, nbx - 1)

    try:
        amp_index = tuple_fieldnames.index('amp')
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
