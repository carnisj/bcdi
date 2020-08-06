# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
from operator import itemgetter

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


def colorbar(mappable, scale='linear', numticks=10, label=None):
    """
    Generate a colorbar whose height (or width) in sync with the master axes.

    :param mappable: the image where to put the colorbar
    :param scale: 'linear' or 'log', used for tick location
    :param numticks: number of ticks for the colorbar
    :param label: label for the colorbar
    :return: the colorbar instance
    """
    last_axes = plt.gca()
    try:
        ax = mappable.axes
    except AttributeError:  # QuadContourSet
        ax = mappable.ax
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    if scale == 'linear':
        cbar.locator = ticker.LinearLocator(numticks=numticks)
    elif scale == 'log':
        cbar.locator = ticker.LogLocator(numticks=numticks)
    else:
        raise ValueError('Incorrect value for the scale parameter')
    if label is not None:
        cbar.ax.set_ylabel(label)
    cbar.update_ticks()
    plt.sca(last_axes)
    return cbar


def combined_plots(tuple_array, tuple_sum_frames, tuple_colorbar, tuple_title, tuple_scale, tuple_sum_axis=0,
                   cmap=my_cmap, tick_direction='inout', tick_width=1, tick_length=3, pixel_spacing=np.nan,
                   tuple_width_v=None, tuple_width_h=None, tuple_vmin=np.nan, tuple_vmax=np.nan, is_orthogonal=False,
                   reciprocal_space=False, **kwargs):
    """
    Subplots of a 1D, 2D or 3D datasets using user-defined parameters.

    :param tuple_array: 2D or 3D array of real numbers
    :param tuple_sum_frames: boolean or tuple of boolean values. If True, will sum the data along sum_axis
    :param tuple_sum_axis: tuple of axis along which to sum or to take the middle slice
    :param tuple_width_v: int or tuple of user-defined zoom vertical width, should be smaller than the actual data
     size. Set it to None if you do not need it.
    :param tuple_width_h: int or tuple of user-defined zoom horizontal width, should be smaller than the actual data
     size. Set it to None if you do not need it.
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
    :param is_orthogonal: set to True is the frame is orthogonal, False otherwise (detector frame) Used for plot labels.
    :param reciprocal_space: True if the data is in reciprocal space, False otherwise. Used for plot labels.
    :param kwargs:
     - 'xlabel' , label of the horizontal axis for plots: string or tuple of strings
     - 'ylabel' , label of the vertical axis for plots: string or tuple of strings
     - 'position' , tuple of subplot positions in the format 231 (2 rows, 3 columns, first subplot)
    :return:  the figure instance
    """
    if type(tuple_array) is not tuple:
        raise TypeError('Expected "tuple_array" to be a tuple')

    nb_subplots = len(tuple_array)
    invert_yaxis = False

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
            if type(xlabel) is not tuple:
                xlabel = (xlabel,) * nb_subplots
        elif k in ['ylabel']:
            ylabel = kwargs['ylabel']
            if type(ylabel) is not tuple:
                ylabel = (ylabel,) * nb_subplots
        elif k in ['position']:
            position = kwargs['position']
            if type(position) is not tuple or len(position) != nb_subplots:
                raise ValueError('"position" should be a tuple of subplot positions')
        else:
            print(k)
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

    try:
        position
    except NameError:
        nb_columns = nb_subplots // 2
        nb_rows = nb_subplots // nb_columns + nb_subplots % nb_columns
        position = [nb_rows*100 + nb_columns*10 + index for index in range(1, nb_subplots+1)]

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
                    if sum_frames:
                        slice_names = (' sum along Q$_x$', ' sum along Q$_z$', ' sum along Q$_y$')
                    else:
                        slice_names = (' slice in Q$_x$', ' slice in Q$_z$', ' slice in Q$_y$')
                    ver_labels = ("Q$_z$", "Q$_x$", "Q$_x$")
                    hor_labels = ("Q$_y$", "Q$_y$", "Q$_z$")
                    if sum_axis == 0:
                        invert_yaxis = True
                    else:
                        invert_yaxis = False
                else:  # detector frame
                    if sum_frames:
                        slice_names = (' sum along Z', ' sum along Y', ' sum along X')
                    else:
                        slice_names = (' slice in Z', ' slice in Y', ' slice in X')
                    ver_labels = ('Y', 'rocking angle', 'rocking angle')
                    hor_labels = ('X', 'X', 'Y')
            else:
                if is_orthogonal:
                    if sum_frames:
                        slice_names = (' sum along z', ' sum along y', ' sum along x')
                    else:
                        slice_names = (' slice in z', ' slice in y', ' slice in x')
                    ver_labels = ('y', 'z', 'z')
                    hor_labels = ('x', 'x', 'y')
                    if sum_axis == 0:
                        invert_yaxis = True
                    else:
                        invert_yaxis = False
                else:  # detector frame
                    if sum_frames:
                        slice_names = (' sum along Z', ' sum along Y', ' sum along X')
                    else:
                        slice_names = (' slice in Z', ' slice in Y', ' slice in X')
                    ver_labels = ('Y', 'rocking angle', 'rocking angle')
                    hor_labels = ('X', 'X', 'Y')

            nbz, nby, nbx = array.shape
            if width_v is None:
                width_v = max(nbz, nby, nbx)
            if width_h is None:
                width_h = max(nbz, nby, nbx)

            if sum_axis == 0:
                dim_v = nby
                dim_h = nbx
                if not sum_frames:
                    array = array[nbz // 2, :, :]
                else:
                    array = array.sum(axis=sum_axis)
                default_xlabel = hor_labels[0]
                default_ylabel = ver_labels[0]
            elif sum_axis == 1:
                dim_v = nbz
                dim_h = nbx
                if not sum_frames:
                    array = array[:, nby // 2, :]
                else:
                    array = array.sum(axis=sum_axis)
                default_xlabel = hor_labels[1]
                default_ylabel = ver_labels[1]
            elif sum_axis == 2:
                dim_v = nbz
                dim_h = nby
                if not sum_frames:
                    array = array[:, :, nbx // 2]
                else:
                    array = array.sum(axis=sum_axis)
                default_xlabel = hor_labels[2]
                default_ylabel = ver_labels[2]
            else:
                print('sum_axis should be only equal to 0, 1 or 2')
                return
            slice_name = slice_names[sum_axis]

        else:  # 2D
            nby, nbx = array.shape
            if width_v is None:
                width_v = max(nby, nbx)
            if width_h is None:
                width_h = max(nby, nbx)

            dim_v = nby
            dim_h = nbx
            slice_name = ''
            default_xlabel = ''
            default_ylabel = ''

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
        if len(xlabel[idx]) != 0:
            axis.set_xlabel(xlabel[idx])
        else:
            axis.set_xlabel(default_xlabel)
        if len(ylabel[idx]) != 0:
            axis.set_ylabel(ylabel[idx])
        else:
            axis.set_ylabel(default_ylabel)
        plt.axis('scaled')
        if not np.isnan(pixel_spacing):
            axis.xaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
            axis.yaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
            axis.tick_params(labelbottom=False, labelleft=False, top=True, right=True, direction=tick_direction,
                             length=tick_length, width=tick_width)
        if invert_yaxis:  # Y is axis 0, need to be flipped
            axis.invert_yaxis()
        if plot_colorbar:
            colorbar(plot, numticks=5)

    plt.tight_layout()  # avoids the overlap of subplots with axes labels
    plt.pause(0.5)
    plt.ioff()

    return fig


def contour_slices(array, q_coordinates, sum_frames=False, slice_position=None, levels=150, width_z=None, width_y=None,
                   width_x=None, plot_colorbar=False, cmap=my_cmap, title='', scale='linear', is_orthogonal=False,
                   reciprocal_space=True):
    """
    Create a figure with three 2D contour plots from a 3D dataset.

    :param array: 3D array of real numbers
    :param q_coordinates: a tuple of (qx, qz, qy) 1D-coordinates corresponding to the (Z, Y, X) of the cxi convention
    :param sum_frames: if True, will sum the data along the 3rd axis
    :param slice_position: tuple of three integers where to slice the 3D array
    :param levels: int n, will use n data intervals and draw n+1 contour lines
    :param width_z: user-defined zoom width along axis 0 (rocking angle), should be smaller than the actual data size
    :param width_y: user-defined zoom width along axis 1 (vertical), should be smaller than the actual data size
    :param width_x: user-defined zoom width along axis 2 (horizontal), should be smaller than the actual data size
    :param plot_colorbar: set it to True in order to plot the colorbar
    :param cmap: colormap to be used
    :param title: string to include in the plot
    :param scale: 'linear' or 'log'
    :param is_orthogonal: set to True is the frame is orthogonal, False otherwise (detector frame) Used for plot labels.
    :param reciprocal_space: True if the data is in reciprocal space, False otherwise. Used for plot labels.
    :return: fig, (ax0, ax1, ax2, ax3), (plt0, plt1, plt2) instances
    """
    nb_dim = array.ndim
    plt.ion()

    if reciprocal_space:
        if is_orthogonal:
            if sum_frames:
                slice_names = (' sum along Q$_x$', ' sum along Q$_z$', ' sum along Q$_y$')
            else:
                slice_names = (' slice in Q$_x$', ' slice in Q$_z$', ' slice in Q$_y$')
            ver_labels = ("Q$_z$", "Q$_x$", "Q$_x$")
            hor_labels = ("Q$_y$", "Q$_y$", "Q$_z$")
        else:  # detector frame
            if sum_frames:
                slice_names = (' sum along Z', ' sum along Y', ' sum along X')
            else:
                slice_names = (' slice in Z', ' slice in Y', ' slice in X')
            ver_labels = ('Y', 'rocking angle', 'rocking angle')
            hor_labels = ('X', 'X', 'Y')
    else:
        if is_orthogonal:
            if sum_frames:
                slice_names = (' sum along z', ' sum along y', ' sum along x')
            else:
                slice_names = (' slice in z', ' slice in y', ' slice in x')
            ver_labels = ('y', 'z', 'z')
            hor_labels = ('x', 'x', 'y')
        else:  # detector frame
            if sum_frames:
                slice_names = (' sum along Z', ' sum along Y', ' sum along X')
            else:
                slice_names = (' slice in Z', ' slice in Y', ' slice in X')
            ver_labels = ('Y', 'rocking angle', 'rocking angle')
            hor_labels = ('X', 'X', 'Y')

    if nb_dim != 3:  # wrong array dimension
        print('multislices_plot() needs a 3D array')
        return
    else:
        nbz, nby, nbx = array.shape
        qx, qz, qy = q_coordinates
        if len(qx) != nbz or len(qz) != nby or len(qy) != nbx:
            print('Coordinates shape is not compatible with data shape')

        if width_z is None:
            width_z = nbz
        if width_y is None:
            width_y = nby
        if width_x is None:
            width_x = nbx

        if not sum_frames:
            if slice_position is None:
                slice_position = [nbz // 2, nby // 2, nbx // 2]
            elif len(slice_position) != 3:
                raise ValueError('wrong shape for the parameter slice_position')
            else:
                slice_position = [int(position) for position in slice_position]

        fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2, figsize=(12, 9))

        # axis 0
        temp_array = np.copy(array)
        if not sum_frames:
            temp_array = temp_array[slice_position[0], :, :]
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
        ax0.set_xlabel(hor_labels[0])
        ax0.set_ylabel(ver_labels[0])
        ax0.set_title(title + slice_names[0])
        if plot_colorbar:
            colorbar(plt0, numticks=5)

        # axis 1
        temp_array = np.copy(array)
        if not sum_frames:
            temp_array = temp_array[:, slice_position[1], :]
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
        ax1.set_xlabel(hor_labels[1])
        ax1.set_ylabel(ver_labels[1])
        ax1.set_title(title + slice_names[1])
        if plot_colorbar:
            colorbar(plt1, numticks=5)

        # axis 2
        temp_array = np.copy(array)
        if not sum_frames:
            temp_array = temp_array[:, :, slice_position[2]]
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
        ax2.set_xlabel(hor_labels[2])
        ax2.set_ylabel(ver_labels[2])
        ax2.set_title(title + slice_names[2])
        if plot_colorbar:
            colorbar(plt2, numticks=5)

        # axis 3
        ax3.set_visible(False)
    plt.tight_layout()  # avoids the overlap of subplots with axes labels
    plt.pause(0.5)
    plt.ioff()
    return fig, (ax0, ax1, ax2, ax3), (plt0, plt1, plt2)


def contour_stereographic(euclidian_u, euclidian_v, color, radius_mean, planes={}, title="", plot_planes=True,
                          contour_range=None, max_angle=95, cmap=my_cmap, uv_labels=('', ''), hide_axis=False,
                          scale='linear', debugging=False):
    """
    Plot the stereographic projection with some cosmetics.

    :param euclidian_u: flattened array, normalized Euclidian metric coordinates (points can be not on a regular grid)
    :param euclidian_v: flattened array, normalized Euclidian metric coordinates (points can be not on a regular grid)
    :param color: flattened array, intensity of density kernel estimation at radius_mean
    :param radius_mean: radius of the sphere in reciprocal space from which the projection is done
    :param planes: dictionnary of crystallographic planes, e.g. {'111':angle_with_reflection}
    :param title: title for the stereographic plot
    :param plot_planes: if True, will draw circle corresponding to crystallographic planes in the pole figure
    :param contour_range: range for the plot contours
    :param max_angle: maximum angle in degrees of the stereographic projection (should be larger than 90)
    :param cmap: colormap to be used
    :param uv_labels: tuple of strings, labels for the u axis and the v axis, respectively
    :param hide_axis: hide the axis frame, ticks and ticks labels
    :param scale: 'linear' or 'log', scale for the colorbar of the plot
    :param debugging: True to see the scatter plot of euclidian coordinates
    :return: figure and axe instances
    """
    from scipy.interpolate import griddata

    if debugging:
        _, ax0 = plt.subplots(nrows=1, ncols=1)
        plt0 = ax0.scatter(euclidian_u, euclidian_v, s=6, c=color, cmap=my_cmap,
                           norm=colors.LogNorm(vmin=max(color[~np.isnan(color)].min(), 1),
                                               vmax=color[~np.isnan(color)].max()))
        circle = patches.Circle((0, 0), 90, color='k', fill=False, linewidth=1.5)
        ax0.add_artist(circle)
        ax0.axis('scaled')
        ax0.set_xlim(-max_angle, max_angle)
        ax0.set_ylim(-max_angle, max_angle)
        ax0.set_xlabel('u ' + uv_labels[0])
        ax0.set_ylabel('v ' + uv_labels[1])
        ax0.set_title(title)
        colorbar(plt0, scale='log', numticks=5)

    nb_points = 5 * max_angle + 1
    v_grid, u_grid = np.mgrid[-max_angle:max_angle:(nb_points*1j), -max_angle:max_angle:(nb_points*1j)]
    # v_grid is changing along the vertical axis, u_grid is changing along the horizontal axis
    intensity_grid = griddata((euclidian_v, euclidian_u), color, (v_grid, u_grid), method='linear')
    nan_indices = np.isnan(intensity_grid)
    # normalize the intensity for easier plotting
    intensity_grid = intensity_grid / abs(intensity_grid[~nan_indices]).max() * 10000

    if contour_range is None:
        if scale == 'linear':
            contour_range = range(0, 10001, 250)
        elif scale == 'log':
            contour_range = np.logspace(0, 4, num=20, endpoint=True, base=10.0)
        else:
            raise ValueError('Incorrect value for scale parameter')

    # plot the stereographic projection
    plt.ion()
    fig, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(12, 9), facecolor='w', edgecolor='k')
    if scale == 'linear':
        plt0 = ax0.contourf(u_grid, v_grid, intensity_grid, contour_range, cmap=cmap)
        colorbar(plt0, scale='linear', numticks=5)
    else:  # log
        plt0 = ax0.contourf(u_grid, v_grid, intensity_grid, contour_range, cmap=cmap,
                            norm=colors.LogNorm(vmin=max(intensity_grid[~nan_indices].min(), 1),
                                                vmax=intensity_grid[~nan_indices].max()))
        colorbar(plt0, scale='log', numticks=5)
    ax0.axis('equal')

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
                 str(ii) + r'$^\circ$', fontsize=10, color='k')
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
    ax0.set_xlabel('u ' + uv_labels[0])
    ax0.set_ylabel('v ' + uv_labels[1])
    if hide_axis:
        ax0.axis('off')
        ax0.set_title(title + '\nu horizontal, v vertical')
    else:
        ax0.set_title(title)
    ax0.axis('scaled')
    plt.pause(0.5)
    plt.ioff()
    return fig, ax0


def imshow_plot(array, sum_frames=False, sum_axis=0, width_v=None, width_h=None, plot_colorbar=False,
                vmin=np.nan, vmax=np.nan, cmap=my_cmap, title='', labels=None, scale='linear',
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
    :param labels: tuple of two strings (vertical label, horizontal label)
    :param scale: 'linear' or 'log'
    :param tick_direction: 'out', 'in', 'inout'
    :param tick_width: width of tickes in plots
    :param tick_length: length of tickes in plots
    :param pixel_spacing: pixel_spacing = desired tick_spacing (in nm) / voxel_size of the reconstruction(in nm)
    :param is_orthogonal: set to True is the frame is orthogonal, False otherwise (detector frame) Used for plot labels.
    :param reciprocal_space: True if the data is in reciprocal space, False otherwise. Used for plot labels.
    :return:  fig, axis, plot instances
    """
    nb_dim = array.ndim
    array = array.astype(float)
    plt.ion()
    fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(12, 9))

    if labels is None:
        labels = ('', '')
    else:
        assert len(labels) == 2, 'labels should be a tuple of two strings (vertical label, horizontal label)'
    if nb_dim == 3:
        if reciprocal_space:
            if is_orthogonal:
                invert_yaxis = True
                if sum_frames:
                    slice_names = (' sum along Q$_x$', ' sum along Q$_z$', ' sum along Q$_y$')
                else:
                    slice_names = (' slice in Q$_x$', ' slice in Q$_z$', ' slice in Q$_y$')
                ver_labels = (labels[0] + r" Q$_z$", labels[0] + r" Q$_x$",
                              labels[0] + r" Q$_x$")
                hor_labels = (labels[1] + r" Q$_y$", labels[1] + r" Q$_y$",
                              labels[1] + r" Q$_z$")
            else:  # detector frame
                invert_yaxis = False
                if sum_frames:
                    slice_names = (' sum along Z', ' sum along Y', ' sum along X')
                else:
                    slice_names = (' slice in Z', ' slice in Y', ' slice in X')
                ver_labels = (labels[0] + ' Y', labels[0] + ' rocking angle', labels[0] + ' rocking angle')
                hor_labels = (labels[1] + ' X', labels[1] + ' X', labels[1] + ' Y')
        else:
            if is_orthogonal:
                invert_yaxis = True
                if sum_frames:
                    slice_names = (' sum along z', ' sum along y', ' sum along x')
                else:
                    slice_names = (' slice in z', ' slice in y', ' slice in x')
                ver_labels = (labels[0] + ' y', labels[0] + ' z', labels[0] + ' z')
                hor_labels = (labels[1] + ' x', labels[1] + ' x', labels[1] + ' y')
            else:  # detector frame
                invert_yaxis = False
                if sum_frames:
                    slice_names = (' sum along Z', ' sum along Y', ' sum along X')
                else:
                    slice_names = (' slice in Z', ' slice in Y', ' slice in X')
                ver_labels = (labels[0] + ' Y', labels[0] + ' rocking angle', labels[0] + ' rocking angle')
                hor_labels = (labels[1] + ' X', labels[1] + ' X', labels[1] + ' Y')

        nbz, nby, nbx = array.shape
        if width_v is None:
            width_v = max(nbz, nby, nbx)
        if width_h is None:
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
        ver_label = ver_labels[sum_axis]
        hor_label = hor_labels[sum_axis]

    elif nb_dim == 2:
        invert_yaxis = False
        nby, nbx = array.shape
        if width_v is None:
            width_v = max(nby, nbx)
        if width_h is None:
            width_h = max(nby, nbx)

        dim_v = nby
        dim_h = nbx
        slice_name, ver_label, hor_label = '', labels[0], labels[1]

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
        axis = plt.gca()
        axis.invert_yaxis()
    axis.set_xlabel(hor_label)
    axis.set_ylabel(ver_label)
    plt.title(title + slice_name)
    plt.axis('scaled')
    if not np.isnan(pixel_spacing):
        axis.xaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
        axis.yaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
        axis.tick_params(labelbottom=False, labelleft=False, top=True, right=True, direction=tick_direction,
                         length=tick_length, width=tick_width)
    if plot_colorbar:
        colorbar(plot, numticks=5)
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
    :param dim: the axis over which the loop is performed (axis 0, 1 or 2)
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

    axis = figure.gca()
    xmin, xmax = axis.get_xlim()
    ymin, ymax = axis.get_ylim()
    if key == 'u':  # show next frame
        idx = idx + 1
        if dim == 0:
            if idx > nbz - 1:
                idx = 0
        elif dim == 1:
            if idx > nby - 1:
                idx = 0
        elif dim == 2:
            if idx > nbx - 1:
                idx = 0

    elif key == 'd':  # show previous frame
        idx = idx - 1
        if dim == 0:
            if idx < 0:
                idx = nbz - 1
        elif dim == 1:
            if idx < 0:
                idx = nby - 1
        elif dim == 2:
            if idx < 0:
                idx = nbx - 1

    elif key == 'right':  # increase colobar max
        if scale == 'linear':
            vmax = vmax * 2
        else:
            vmax = vmax + 1

    elif key == 'left':  # reduce colobar max
        if scale == 'linear':
            vmax = vmax / 2
        else:
            vmax = vmax - 1
        if vmax < 1:
            vmax = 1

    elif key == 'p':  # plot full image
        if dim == 0:
            xmin, xmax = -0.5, nbx - 0.5
            ymin, ymax = nby - 0.5, -0.5  # pointing down
        elif dim == 1:
            xmin, xmax = -0.5, nbx - 0.5
            ymin, ymax = nbz - 0.5, -0.5  # pointing down
        elif dim == 2:
            xmin, xmax = -0.5, nby - 0.5
            ymin, ymax = nbz - 0.5, -0.5  # pointing down

    elif key == 'q':
        exit_flag = True

    elif key == 'r':
        filename = 'frame' + str(idx) + '_dim' + str(dim) + '.png'
        plt.savefig(savedir + filename)

    # get the images on axis
    im = axis.images
    # get and remove the existing colorbar
    cb = im[0].colorbar  # there is only one axis in the list im
    cb.remove()

    axis.cla()
    if dim == 0:
        if scale == 'linear':
            plot = axis.imshow(data[idx, :, :], vmin=vmin, vmax=vmax, cmap=cmap)
        else:  # 'log'
            plot = axis.imshow(np.log10(data[idx, :, :]), vmin=vmin, vmax=vmax, cmap=cmap)
        axis.set_title("Frame " + str(idx + 1) + "/" + str(nbz) +
                       "\nq quit ; u next frame ; d previous frame ; p unzoom\n"
                       "right darker ; left brighter ; r save 2D frame")
        colorbar(plot, numticks=5)
    elif dim == 1:
        if scale == 'linear':
            plot = axis.imshow(data[:, idx, :], vmin=vmin, vmax=vmax, cmap=cmap)
        else:  # 'log'
            plot = axis.imshow(np.log10(data[:, idx, :]), vmin=vmin, vmax=vmax, cmap=cmap)
        axis.set_title("Frame " + str(idx + 1) + "/" + str(nby) +
                       "\nq quit ; u next frame ; d previous frame ; p unzoom\n"
                       "right darker ; left brighter ; r save 2D frame")
        colorbar(plot, numticks=5)
    elif dim == 2:
        if scale == 'linear':
            plot = axis.imshow(data[:, :, idx], vmin=vmin, vmax=vmax, cmap=cmap)
        else:  # 'log'
            plot = axis.imshow(np.log10(data[:, :, idx]), vmin=vmin, vmax=vmax, cmap=cmap)
        axis.set_title("Frame " + str(idx + 1) + "/" + str(nbx) +
                       "\nq quit ; u next frame ; d previous frame ; p unzoom\n"
                       "right darker ; left brighter ; r save 2D frame")
        colorbar(plot, numticks=5)
    axis.set_xlim([xmin, xmax])
    axis.set_ylim([ymin, ymax])
    plt.draw()

    return vmax, idx, exit_flag


def mlab_contour3d(x, y, z, scalars, contours, extent, nb_labels, fig_size=(400, 350), azimut=150, elevation=70,
                   distance='auto', roll=0, vmin=None, vmax=None, opacity=1, color=None, colormap='jet', title='',
                   savedir=None):
    """
    3D scatter plot using mayavi. The frame convention is (x,y,z) right-handed.

    :param x: x position of voxels (numpy.mgrid)
    :param y: y position of voxels (numpy.mgrid)
    :param z: z position of voxels (numpy.mgrid)
    :param scalars: scalar field at each voxel.
    :param contours: integer/list specifying number/list of contours. Specifying a list of values will only give the
     requested contours asked for.
    :param extent:[xmin, xmax, ymin, ymax, zmin, zmax] Default is the x, y, z arrays extent.
     Use this to change the extent of the object created.
    :param nb_labels: the number of labels along each direction.
    :param fig_size: the size of the scene created, in pixels.
    :param azimut: the azimuthal angle (in degrees, 0-360), i.e. the angle subtended by the position vector on a sphere
     projected on to the x-y plane with the x-axis. It can be a list of angles for several plots.
    :param elevation: the zenith angle (in degrees, 0-180), i.e. the angle subtended by the position vector
     and the z-axis. It can be a list of angles for several plots.
    :param distance: a positive floating point number representing the distance from the focal point to place the
     camera. It can be a list of the same length as aimut. If ‘auto’ is passed, the distance is computed to have a best
      fit of objects in the frame.
    :param roll: absolute roll angle of the camera. It can be a list of angles for several plots.
    :param vmin: vmin is used to scale the colormap. If None, the min of the data will be used.
    :param vmax: vmax is used to scale the colormap. If None, the max of the data will be used.
    :param opacity:	the overall opacity of the vtk object. Must be a float. Default: 1.0.
    :param color: the color of the vtk object. Overides the colormap, if any, when specified. This is specified as a
     triplet of float ranging from 0 to 1, eg (1, 1, 1) for white.
    :param colormap: type of colormap to use.
    :param title: title to be included in the filename of the saved image. It can be a list of length len(azimut).
    :param savedir: path of the saving directory.
    :return: figure, axes and colorbar instances.
    """
    from mayavi import mlab

    # check input parameters
    if vmin is None:
        vmin = scalars.min()
    if vmax is None:
        vmax = scalars.max()

    assert len(fig_size) == 2, 'fig_size should be a tuple of 2 pixel numbers'

    if type(azimut) in [tuple, list]:  # several views to be plotted
        nb_plots = len(azimut)
        if type(elevation) in [int, float]:
            elevation = [elevation for _ in range(nb_plots)]
        else:
            assert len(elevation) == nb_plots, 'elevation should have the same number of elements as azimut'
        if type(roll) in [int, float]:
            roll = [roll for _ in range(nb_plots)]
        else:
            assert len(roll) == nb_plots, 'roll should have the same number of elements as azimut'
        if type(distance) in [int, float]:
            distance = [distance for _ in range(nb_plots)]
        else:
            assert len(distance) == nb_plots, 'distance should have the same number of elements as azimut'
        if type(title) is str:
            title = [title + '_' + str(idx) for idx in range(nb_plots)]
        else:
            assert len(title) == nb_plots, 'title should have the same number of elements as azimut'
    else:  # azimut is a number
        nb_plots = 1
        azimut = [azimut]
        assert type(elevation) in [int, float], 'elevation should be a number'
        elevation = [elevation]
        assert type(roll) in [int, float], 'roll should be a number'
        roll = [roll]
        assert type(title) is str, 'title should be a string'
        title = [title]

    # plot the contour3d figure
    fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=fig_size)
    if color is None:
        mlab.contour3d(x, y, z, scalars, contours=contours, opacity=opacity, vmin=vmin, vmax=vmax,  colormap=colormap)
    else:
        mlab.contour3d(x, y, z, scalars, contours=contours, opacity=opacity, vmin=vmin, vmax=vmax, color=color)

    # loop over the different views
    for idx in range(nb_plots):
        mlab.view(azimuth=azimut[idx], elevation=elevation[idx], distance=distance[idx])
        # azimut is the rotation around z axis of mayavi
        mlab.roll(roll[idx])
        ax = mlab.axes(extent=extent, line_width=2.0, nb_labels=nb_labels)
        cbar = mlab.colorbar(orientation='vertical')
        ax.label_text_property.opacity = 1.0
        ax.title_text_property.opacity = 1.0
        if savedir is not None:
            mlab.savefig(savedir + title[idx] + '_labels.png', figure=fig)
        cbar.visible = False
        ax.label_text_property.opacity = 0.0
        ax.title_text_property.opacity = 0.0
        if savedir is not None:
            mlab.savefig(savedir + title[idx] + '.png', figure=fig)
    return fig, ax, cbar


def mlab_points3d(x, y, z, scalars, extent, nb_labels, fig_size=(400, 350), azimut=150, elevation=70, distance='auto',
                  roll=0, mode='cube', vmin=None, vmax=None, opacity=1, colormap='jet', title='', savedir=None):
    """
    3D scatter plot using mayavi. The frame convention is (x,y,z) right-handed.

    :param x: x position of voxels (numpy.mgrid)
    :param y: y position of voxels (numpy.mgrid)
    :param z: z position of voxels (numpy.mgrid)
    :param scalars: scalar field at each voxel.
    :param extent:[xmin, xmax, ymin, ymax, zmin, zmax] Default is the x, y, z arrays extent.
     Use this to change the extent of the object created.
    :param nb_labels: the number of labels along each direction.
    :param fig_size: the size of the scene created, in pixels.
    :param azimut: the azimuthal angle (in degrees, 0-360), i.e. the angle subtended by the position vector on a sphere
     projected on to the x-y plane with the x-axis. It can be a list of angles for several plots.
    :param elevation: the zenith angle (in degrees, 0-180), i.e. the angle subtended by the position vector
     and the z-axis. It can be a list of angles for several plots.
    :param distance: a positive floating point number representing the distance from the focal point to place the
     camera. It can be a list of the same length as aimut. If ‘auto’ is passed, the distance is computed to have a best
      fit of objects in the frame.
    :param roll: absolute roll angle of the camera. It can be a list of angles for several plots.
    :param mode: the mode of the glyphs. Available modes are: ‘2darrow’, ‘2dcircle’, ‘2dcross’, ‘2ddash’, ‘2ddiamond’,
     ‘2dhooked_arrow’, ‘2dsquare’, ‘2dthick_arrow’, ‘2dthick_cross’, ‘2dtriangle’, ‘2dvertex’, ‘arrow’, ‘axes’, ‘cone’,
      ‘cube’, ‘cylinder’, ‘point’, ‘sphere’
    :param vmin: vmin is used to scale the colormap. If None, the min of the data will be used.
    :param vmax: vmax is used to scale the colormap. If None, the max of the data will be used.
    :param opacity:	the overall opacity of the vtk object. Must be a float. Default: 1.0
    :param colormap: type of colormap to use.
    :param title: title to be included in the filename of the saved image. It can be a list of length len(azimut).
    :param savedir: path of the saving directory.
    :return: figure, axes and colorbar instances
    """
    from mayavi import mlab

    # check input parameters
    if vmin is None:
        vmin = scalars.min()
    if vmax is None:
        vmax = scalars.max()

    assert len(fig_size) == 2, 'fig_size should be a tuple of 2 pixel numbers'

    if type(azimut) in [tuple, list]:  # several views to be plotted
        nb_plots = len(azimut)
        if type(elevation) in [int, float]:
            elevation = [elevation for _ in range(nb_plots)]
        else:
            assert len(elevation) == nb_plots, 'elevation should have the same number of elements as azimut'
        if type(roll) in [int, float]:
            roll = [roll for _ in range(nb_plots)]
        else:
            assert len(roll) == nb_plots, 'roll should have the same number of elements as azimut'
        if type(distance) in [int, float]:
            distance = [distance for _ in range(nb_plots)]
        else:
            assert len(distance) == nb_plots, 'distance should have the same number of elements as azimut'
        if type(title) is str:
            title = [title + '_' + str(idx) for idx in range(nb_plots)]
        else:
            assert len(title) == nb_plots, 'title should have the same number of elements as azimut'
    else:  # azimut is a number
        nb_plots = 1
        azimut = [azimut]
        assert type(elevation) in [int, float], 'elevation should be a number'
        elevation = [elevation]
        assert type(roll) in [int, float], 'roll should be a number'
        roll = [roll]
        assert type(title) is str, 'title should be a string'
        title = [title]

    # plot the points3d figure
    fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=fig_size)
    mlab.points3d(x, y, z, scalars, mode=mode, opacity=opacity, vmin=vmin, vmax=vmax,  colormap=colormap)

    # loop over the different views
    for idx in range(nb_plots):
        mlab.view(azimuth=azimut[idx], elevation=elevation[idx], distance=distance[idx])
        # azimut is the rotation around z axis of mayavi
        mlab.roll(roll[idx])
        ax = mlab.axes(extent=extent, line_width=2.0, nb_labels=nb_labels)
        cbar = mlab.colorbar(orientation='vertical')
        ax.label_text_property.opacity = 1.0
        ax.title_text_property.opacity = 1.0
        if savedir is not None:
            mlab.savefig(savedir + title[idx] + '_labels.png', figure=fig)
        cbar.visible = False
        ax.label_text_property.opacity = 0.0
        ax.title_text_property.opacity = 0.0
        if savedir is not None:
            mlab.savefig(savedir + title[idx] + '.png', figure=fig)
    return fig, ax, cbar


def multislices_plot(array, sum_frames=False, slice_position=None, width_z=None, width_y=None, width_x=None,
                     plot_colorbar=False, cmap=my_cmap, title='', scale='linear', vmin=np.nan, vmax=np.nan,
                     tick_direction='inout', tick_width=1, tick_length=3, pixel_spacing=None,
                     is_orthogonal=False, reciprocal_space=False, ipynb_layout=False):
    """
    Create a figure with three 2D imshow plots from a 3D dataset.

    :param array: 3D array of real numbers
    :param sum_frames: if True, will sum the data along the 3rd axis
    :param slice_position: tuple of three integers where to slice the 3D array
    :param width_z: zoom width along axis 0 (rocking angle), should be smaller than the actual data size
    :param width_y: zoom width along axis 1 (vertical), should be smaller than the actual data size
    :param width_x: zoom width along axis 2 (horizontal), should be smaller than the actual data size
    :param plot_colorbar: set it to True in,der to plot the colorbar
    :param cmap: colormap to be used
    :param title: string to include in the plot
    :param scale: 'linear', 'log'
    :param tick_direction: 'out', 'in', 'inout'
    :param tick_width: width of tickes in plots
    :param tick_length: length of tickes in plots
    :param pixel_spacing: pixel_spacing=desired tick_spacing (in nm)/voxel_size of the reconstruction(in nm)
    :param is_orthogonal: set to True is the frame is orthogonal, False otherwise (detector frame) Used for plot labels.
    :param reciprocal_space: True if the data is in reciprocal space, False otherwise. Used for plot labels.
    :param vmin: lower boundary for the colorbar. Float or tuple of 3 floats
    :param vmax: higher boundary for the colorbar. Float or tuple of 3 floats
    :param ipynb_layout: toggle for 3 plots in a row, cleaner in an Jupyter Notebook
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

    if reciprocal_space:
        if is_orthogonal:
            invert_yaxis = True
            if sum_frames:
                slice_names = (' sum along Q$_x$', ' sum along Q$_z$', ' sum along Q$_y$')
            else:
                slice_names = (' slice in Q$_x$', ' slice in Q$_z$', ' slice in Q$_y$')
            ver_labels = ("Q$_z$", "Q$_x$", "Q$_x$")
            hor_labels = ("Q$_y$", "Q$_y$", "Q$_z$")
        else:  # detector frame
            invert_yaxis = False
            if sum_frames:
                slice_names = (' sum along Z', ' sum along Y', ' sum along X')
            else:
                slice_names = (' slice in Z', ' slice in Y', ' slice in X')
            ver_labels = ('Y', 'rocking angle', 'rocking angle')
            hor_labels = ('X', 'X', 'Y')
    else:
        if is_orthogonal:
            invert_yaxis = True
            if sum_frames:
                slice_names = (' sum along z', ' sum along y', ' sum along x')
            else:
                slice_names = (' slice in z', ' slice in y', ' slice in x')
            ver_labels = ('y', 'z', 'z')
            hor_labels = ('x', 'x', 'y')
        else:  # detector frame
            invert_yaxis = False
            if sum_frames:
                slice_names = (' sum along Z', ' sum along Y', ' sum along X')
            else:
                slice_names = (' slice in Z', ' slice in Y', ' slice in X')
            ver_labels = ('Y', 'rocking angle', 'rocking angle')
            hor_labels = ('X', 'X', 'Y')

    if width_z is None:
        width_z = nbz
    if width_y is None:
        width_y = nby
    if width_x is None:
        width_x = nbx

    if ipynb_layout:
        fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(15, 4.5))
    else:
        fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2, figsize=(12, 9))

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
    ax0.set_xlabel(hor_labels[0])
    ax0.set_ylabel(ver_labels[0])
    ax0.set_title(title + slice_names[0])
    if invert_yaxis:  # Y is axis 0, need to be flipped
        ax0.invert_yaxis()
    plt.axis('scaled')
    if plot_colorbar:
        colorbar(plt0, numticks=5)
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
    ax1.set_xlabel(hor_labels[1])
    ax1.set_ylabel(ver_labels[1])
    ax1.set_title(title + slice_names[1])
    plt.axis('scaled')
    if plot_colorbar:
        colorbar(plt1, numticks=5)
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
    ax2.set_xlabel(hor_labels[2])
    ax2.set_ylabel(ver_labels[2])
    ax2.set_title(title + slice_names[2])
    plt.axis('scaled')

    if plot_colorbar:
        colorbar(plt2, numticks=5)
    if pixel_spacing is not None:
        ax2.xaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
        ax2.yaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
        ax2.tick_params(labelbottom=False, labelleft=False, top=True, right=True, direction=tick_direction,
                        length=tick_length, width=tick_width)
    if not ipynb_layout:
        # axis 3
        ax3.set_visible(False)

    plt.tight_layout()  # avoids the overlap of subplots with axes labels
    plt.pause(0.5)
    plt.ioff()
    
    if ipynb_layout:
        return fig, (ax0, ax1, ax2), (plt0, plt1, plt2)
    else:
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


def save_to_vti(filename, voxel_size, tuple_array, tuple_fieldnames, origin=(0, 0, 0), amplitude_threshold=0.01):
    """
    Save arrays defined by their name in a single vti file. Paraview expects data in an orthonormal basis (x,y,z). For
    BCDI data in the .cxi convention (hence: z,y,x) it is necessary to flip the last axis. The data sent to Paraview
    will be in the orthonormal frame (z,y,-x), therefore Paraview_x is z (downstream), Paraview_y is y (vertical up),
    Paraview_z is -x (inboard) of the .cxi convention.

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
        plt.pause(0.5)
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


def scatter_stereographic(euclidian_u, euclidian_v, color, title="", max_angle=95, cmap=my_cmap, uv_labels=('', '')):
    """
    Plot the stereographic projection of the real scattered positions of data points.

    :param euclidian_u: flattened array, normalized Euclidian metric coordinates (points can be not on a regular grid)
    :param euclidian_v: flattened array, normalized Euclidian metric coordinates (points can be not on a regular grid)
    :param color: flattened array, intensity of density kernel estimation at radius_mean
    :param title: title for the stereographic plot
    :param max_angle: maximum angle in degrees of the stereographic projection (should be larger than 90)
    :param cmap: colormap to be used
    :param uv_labels: tuple of strings, labels for the u axis and the v axis, respectively
    :return: figure and axe instances
    """
    fig, ax0 = plt.subplots(nrows=1, ncols=1)
    plt0 = ax0.scatter(euclidian_u, euclidian_v, s=6, c=color, cmap=cmap,
                       norm=colors.LogNorm(vmin=max(color[~np.isnan(color)].min(), 1),
                                           vmax=color[~np.isnan(color)].max()))
    circle = patches.Circle((0, 0), 90, color='k', fill=False, linewidth=1.5)
    ax0.add_artist(circle)
    ax0.axis('scaled')
    ax0.set_xlim(-max_angle, max_angle)
    ax0.set_ylim(-max_angle, max_angle)
    ax0.set_xlabel('u ' + uv_labels[0])
    ax0.set_ylabel('v ' + uv_labels[1])
    ax0.set_title(title)
    colorbar(plt0, scale='log', numticks=5)
    plt.pause(0.5)
    plt.ioff()
    return fig, ax0


def update_aliens(key, pix, piy, original_data, original_mask, updated_data, updated_mask, figure, width, dim, idx,
                  vmax, vmin=0, invert_yaxis=False):
    """
    Update the plot while removing the parasitic diffraction intensity in 3D dataset

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
    if original_data.ndim != 3 or updated_data.ndim != 3 or original_mask.ndim != 3 or updated_mask.ndim != 3:
        raise ValueError('original_data, original_mask, updated_data and updated_mask should be 3D arrays')

    nbz, nby, nbx = original_data.shape
    stop_masking = False
    if dim not in [0, 1, 2]:
        raise ValueError('dim should be 0, 1 or 2')

    axs = figure.gca()
    xmin, xmax = axs.get_xlim()
    ymin, ymax = axs.get_ylim()
    if key == 'u':  # show next frame
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

    elif key == 'd':  # show previous frame
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

    elif key == 'up':
        width = width + 1

    elif key == 'down':
        width = width - 1
        if width < 0:
            width = 0

    elif key == 'right':  # increase colobar max
        vmax = vmax * 2

    elif key == 'left':  # reduce colobar max
        vmax = vmax / 2
        if vmax < 1:
            vmax = 1

    elif key == 'm':  # mask intensities
        if (piy - width) < 0:
            starty = 0
        else:
            starty = piy - width
        if (pix - width) < 0:
            startx = 0
        else:
            startx = pix - width
        if dim == 0:
            updated_data[idx, starty:piy + width + 1, startx:pix + width + 1] = 0
            updated_mask[idx, starty:piy + width + 1, startx:pix + width + 1] = 1
        elif dim == 1:
            updated_data[starty:piy + width + 1, idx, startx:pix + width + 1] = 0
            updated_mask[starty:piy + width + 1, idx, startx:pix + width + 1] = 1
        else:  # dim=2
            updated_data[starty:piy + width + 1, startx:pix + width + 1, idx] = 0
            updated_mask[starty:piy + width + 1, startx:pix + width + 1, idx] = 1

    elif key == 'b':  # back to measured intensities
        if (piy - width) < 0:
            starty = 0
        else:
            starty = piy - width
        if (pix - width) < 0:
            startx = 0
        else:
            startx = pix - width
        if dim == 0:
            updated_data[idx, starty:piy + width + 1, startx:pix + width + 1] = \
                original_data[idx, starty:piy + width + 1, startx:pix + width + 1]
            updated_mask[idx, starty:piy + width + 1, startx:pix + width + 1] = \
                original_mask[idx, starty:piy + width + 1, startx:pix + width + 1]

        elif dim == 1:
            updated_data[starty:piy + width + 1, idx, startx:pix + width + 1] = \
                original_data[starty:piy + width + 1, idx, startx:pix + width + 1]
            updated_mask[starty:piy + width + 1, idx, startx:pix + width + 1] = \
                original_mask[starty:piy + width + 1, idx, startx:pix + width + 1]
        else:  # dim=2
            updated_data[starty:piy + width + 1, startx:pix + width + 1, idx] = \
                original_data[starty:piy + width + 1, startx:pix + width + 1, idx]
            updated_mask[starty:piy + width + 1, startx:pix + width + 1, idx] = \
                original_mask[starty:piy + width + 1, startx:pix + width + 1, idx]

    elif key == 'p' or key == 'a':  # plot full image or restart masking
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
        if key == 'a':  # restart masking
            updated_data[:] = original_data[:]
            updated_mask[:] = original_mask[:]

    elif key == 'q':
        stop_masking = True

    else:
        return updated_data, updated_mask, width, vmax, idx, stop_masking

    axs.cla()
    if dim == 0:
        axs.imshow(updated_data[idx, :, ], vmin=vmin, vmax=vmax)
        axs.set_title("XY - Frame " + str(idx + 1) + "/" + str(nbz) + "\n"
                      "m mask ; b unmask ; q quit ; u next frame ; d previous frame\n"
                      "up larger ; down smaller ; right darker ; left brighter")
    elif dim == 1:
        axs.imshow(updated_data[:, idx, :], vmin=vmin, vmax=vmax)
        axs.set_title("XZ - Frame " + str(idx + 1) + "/" + str(nby) + "\n"
                      "m mask ; b unmask ; q quit ; u next frame ; d previous frame\n"
                      "up larger ; down smaller ; right darker ; left brighter")
    elif dim == 2:
        axs.imshow(updated_data[:, :, idx], vmin=vmin, vmax=vmax)
        axs.set_title("YZ - Frame " + str(idx + 1) + "/" + str(nbx) + "\n"
                      "m mask ; b unmask ; q quit ; u next frame ; d previous frame\n"
                      "up larger ; down smaller ; right darker ; left brighter")
    if invert_yaxis:
        axs.invert_yaxis()
    axs.set_xlim([xmin, xmax])
    axs.set_ylim([ymin, ymax])
    plt.draw()

    return updated_data, updated_mask, width, vmax, idx, stop_masking


def update_aliens_combined(key, pix, piy, original_data, original_mask, updated_data, updated_mask, axes, width, dim,
                           frame_index, vmax, vmin=0, invert_yaxis=False):
    """
    Update the plot while removing the parasitic diffraction intensity in 3D dataset

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
    :param invert_yaxis: True to invert the y axis of imshow plots
    :return: updated data, mask and controls
    """
    if original_data.ndim != 3 or updated_data.ndim != 3 or original_mask.ndim != 3 or updated_mask.ndim != 3:
        raise ValueError('original_data, updated_data and updated_mask should be 3D arrays')

    nbz, nby, nbx = original_data.shape
    stop_masking = False
    if dim not in [0, 1, 2]:
        raise ValueError('dim should be 0, 1 or 2')

    xmin0, xmax0 = axes[0].get_xlim()
    ymin0, ymax0 = axes[0].get_ylim()
    xmin1, xmax1 = axes[1].get_xlim()
    ymin1, ymax1 = axes[1].get_ylim()
    xmin2, xmax2 = axes[2].get_xlim()
    ymin2, ymax2 = axes[2].get_ylim()

    if key == 'u':  # show next frame
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

    elif key == 'd':  # show previous frame
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

    elif key == 'up':
        width = width + 1

    elif key == 'down':
        width = width - 1
        if width < 0:
            width = 0

    elif key == 'right':  # increase colobar max
        vmax = vmax * 2

    elif key == 'left':  # reduce colobar max
        vmax = vmax / 2
        if vmax < 1:
            vmax = 1

    elif key == 'm':  # mask intensities
        if (piy - width) < 0:
            starty = 0
        else:
            starty = piy - width
        if (pix - width) < 0:
            startx = 0
        else:
            startx = pix - width
        if dim == 0:
            updated_data[frame_index[0], starty:piy + width + 1, startx:pix + width + 1] = 0
            updated_mask[frame_index[0], starty:piy + width + 1, startx:pix + width + 1] = 1
        elif dim == 1:
            updated_data[starty:piy + width + 1, frame_index[1], startx:pix + width + 1] = 0
            updated_mask[starty:piy + width + 1, frame_index[1], startx:pix + width + 1] = 1
        else:  # dim=2
            updated_data[starty:piy + width + 1, startx:pix + width + 1, frame_index[2]] = 0
            updated_mask[starty:piy + width + 1, startx:pix + width + 1, frame_index[2]] = 1

    elif key == 'b':  # back to measured intensities
        if (piy - width) < 0:
            starty = 0
        else:
            starty = piy - width
        if (pix - width) < 0:
            startx = 0
        else:
            startx = pix - width
        if dim == 0:
            updated_data[frame_index[0], starty:piy + width + 1, startx:pix + width + 1] = \
                original_data[frame_index[0], starty:piy + width + 1, startx:pix + width + 1]
            updated_mask[frame_index[0], starty:piy + width + 1, startx:pix + width + 1] = \
                original_mask[frame_index[0], starty:piy + width + 1, startx:pix + width + 1]
        elif dim == 1:
            updated_data[starty:piy + width + 1, frame_index[1], startx:pix + width + 1] = \
                original_data[starty:piy + width + 1, frame_index[1], startx:pix + width + 1]
            updated_mask[starty:piy + width + 1, frame_index[1], startx:pix + width + 1] = \
                original_mask[starty:piy + width + 1, frame_index[1], startx:pix + width + 1]
        else:  # dim=2
            updated_data[starty:piy + width + 1, startx:pix + width + 1, frame_index[2]] = \
                original_data[starty:piy + width + 1, startx:pix + width + 1, frame_index[2]]
            updated_mask[starty:piy + width + 1, startx:pix + width + 1, frame_index[2]] = \
                original_mask[starty:piy + width + 1, startx:pix + width + 1, frame_index[2]]

    elif key == 'p' or key == 'a':  # plot full image or restart masking
        xmin0, xmax0 = -0.5, nbx - 0.5
        if invert_yaxis:
            ymin0, ymax0 = -0.5, nby - 0.5  # pointing up
        else:
            ymin0, ymax0 = nby - 0.5, -0.5  # pointing down
        xmin1, xmax1 = -0.5, nbx - 0.5
        ymin1, ymax1 = nbz - 0.5, -0.5  # pointing down
        xmin2, xmax2 = -0.5, nby - 0.5
        ymin2, ymax2 = nbz - 0.5, -0.5  # pointing down
        if key == 'a':  # restart masking
            updated_data[:] = original_data[:]
            updated_mask[:] = original_mask[:]

    elif key == 'q':
        stop_masking = True

    else:
        return updated_data, updated_mask, width, vmax, frame_index, stop_masking

    axes[0].cla()
    axes[1].cla()
    axes[2].cla()
    axes[0].imshow(updated_data[frame_index[0], :, :], vmin=vmin, vmax=vmax)
    axes[1].imshow(updated_data[:, frame_index[1], :], vmin=vmin, vmax=vmax)
    axes[2].imshow(updated_data[:, :, frame_index[2]], vmin=vmin, vmax=vmax)
    axes[0].set_title("XY - Frame " + str(frame_index[0] + 1) + "/" + str(nbz))
    axes[0].axis('scaled')
    if invert_yaxis:
        axes[0].invert_yaxis()
    axes[0].set_xlim([xmin0, xmax0])
    axes[0].set_ylim([ymin0, ymax0])
    axes[1].set_title("XZ - Frame " + str(frame_index[1] + 1) + "/" + str(nby))
    axes[1].axis('scaled')
    axes[1].set_xlim([xmin1, xmax1])
    axes[1].set_ylim([ymin1, ymax1])
    axes[2].set_title("YZ - Frame " + str(frame_index[2] + 1) + "/" + str(nbx))
    axes[2].axis('scaled')
    axes[2].set_xlim([xmin2, xmax2])
    axes[2].set_ylim([ymin2, ymax2])
    plt.draw()

    return updated_data, updated_mask, width, vmax, frame_index, stop_masking


def update_aliens_2d(key, pix, piy, original_data, original_mask, updated_data, updated_mask, figure, width,
                     vmax, vmin=0, invert_yaxis=False):
    """
    Update the plot while removing the parasitic diffraction intensity in 2D dataset

    :param key: the keyboard key which was pressed
    :param pix: the x value of the mouse pointer
    :param piy: the y value of the mouse pointer
    :param original_data: the 2D data array before masking aliens
    :param original_mask: the 3D mask array before masking aliens
    :param updated_data: the current 2D data array
    :param updated_mask: the current 2D mask array
    :param figure: the figure instance
    :param width: the half_width of the masking window
    :param vmax: the higher boundary for the colorbar
    :param vmin: the lower boundary for the colorbar
    :param invert_yaxis: True to invert the y axis of imshow plots
    :return: updated data, mask and controls
    """
    if original_data.ndim != 2 or updated_data.ndim != 2 or original_mask.ndim != 2or updated_mask.ndim != 2:
        raise ValueError('original_data, updated_data and updated_mask should be 2D arrays')

    nby, nbx = original_data.shape
    stop_masking = False

    axs = figure.gca()
    xmin, xmax = axs.get_xlim()
    ymin, ymax = axs.get_ylim()

    if key == 'up':
        width = width + 1

    elif key == 'down':
        width = width - 1
        if width < 0:
            width = 0

    elif key == 'right':
        vmax = vmax * 2

    elif key == 'left':
        vmax = vmax / 2
        if vmax < 1:
            vmax = 1

    elif key == 'm':
        if (piy - width) < 0:
            starty = 0
        else:
            starty = piy - width
        if (pix - width) < 0:
            startx = 0
        else:
            startx = pix - width
        updated_data[starty:piy + width + 1, startx:pix + width + 1] = 0
        updated_mask[starty:piy + width + 1, startx:pix + width + 1] = 1

    elif key == 'b':
        if (piy - width) < 0:
            starty = 0
        else:
            starty = piy - width
        if (pix - width) < 0:
            startx = 0
        else:
            startx = pix - width
        updated_data[starty:piy + width + 1, startx:pix + width + 1] = \
            original_data[starty:piy + width + 1, startx:pix + width + 1]
        updated_mask[starty:piy + width + 1, startx:pix + width + 1] = \
            original_mask[starty:piy + width + 1, startx:pix + width + 1]

    elif key == 'p' or key == 'a':  # plot full image or restart masking
        xmin, xmax = -0.5, nbx - 0.5
        if invert_yaxis:
            ymin, ymax = -0.5, nby - 0.5  # pointing up
        else:
            ymin, ymax = nby - 0.5, -0.5  # pointing down
        if key == 'a':  # restart masking
            updated_data[:] = original_data[:]
            updated_mask[:] = original_mask[:]

    elif key == 'q':
        stop_masking = True

    else:
        return updated_data, updated_mask, width, vmax, stop_masking

    axs.cla()
    axs.imshow(updated_data, vmin=vmin, vmax=vmax)
    axs.set_title("m mask ; b unmask ; q quit ; u next frame ; d previous frame\n"
                  "up larger ; down smaller ; right darker ; left brighter")
    if invert_yaxis:
        axs.invert_yaxis()
    axs.set_xlim([xmin, xmax])
    axs.set_ylim([ymin, ymax])
    plt.draw()

    return updated_data, updated_mask, width, vmax, stop_masking


def update_background(key, distances, data, figure, flag_pause, xy, scale='log', xlim=None, ylim=None):
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
    if data.ndim != 1:
        raise ValueError('data is expected to be a 1D array')
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

    if key == 'b':  # remove the last selected background point
        xy.pop()

    elif key == 'a':  # restart background selection from the beginning
        xy = []
        print('restart background selection')

    elif key == 'p':  # plot background
        pass

    elif key == 'x':
        if not flag_pause:
            flag_pause = True
            print('pause for pan/zoom')
        else:
            flag_pause = False
            print('resume masking')

    elif key == 'q':
        stop_masking = True

    else:
        return flag_pause, xy, stop_masking

    background = np.asarray(xy)
    axs.cla()
    if len(xy) != 0:
        if scale == 'linear':
            axs.plot(distances, data, '.-r', background[:, 0], background[:, 1], 'b')
        else:
            axs.plot(distances, np.log10(data), '.-r',
                     background[:, 0], background[:, 1], 'b')  # background is in log scale directly
    else:  # restart background selection
        if scale == 'linear':
            axs.plot(distances, data, '.-r')
        else:
            axs.plot(distances, np.log10(data), '.-r')
    axs.set_xlim([xmin, xmax])
    axs.set_ylim([ymin, ymax])
    axs.set_xlabel('q (1/nm)')
    axs.set_ylabel('Angular average (A.U.)')
    axs.set_title("Click to select background points\nx to pause/resume for pan/zoom\n"
                  "a restart ; p plot background ; q quit")
    plt.draw()

    return flag_pause, xy, stop_masking


def update_mask(key, pix, piy, original_data, original_mask, updated_data, updated_mask, figure, flag_pause, points,
                xy, width, dim, vmax, vmin=0, masked_color=0.1, invert_yaxis=False):
    """
    Update the mask to remove parasitic diffraction intensity and hotpixels in 3D dataset.

    :param key: the keyboard key which was pressed
    :param pix: the x value of the mouse pointer
    :param piy: the y value of the mouse pointer
    :param original_data: the 3D data array before masking
    :param original_mask: the 3D mask array before masking
    :param updated_data: the current 3D data array
    :param updated_mask: the temporary 2D mask array with updated points
    :param figure: the figure instance
    :param flag_pause: set to 1 to stop registering vertices using mouse clicks
    :param points: list of all point coordinates: points=np.stack((x, y), axis=0).T with x=x.flatten() , y = y.flatten()
     given x,y=np.meshgrid(np.arange(nx), np.arange(ny))
    :param xy: the list of vertices which defines a polygon to be masked
    :param width: the half_width of the masking window
    :param dim: the axis currently under review (axis 0, 1 or 2)
    :param vmax: the higher boundary for the colorbar
    :param vmin: the lower boundary for the colorbar
    :param masked_color: the value that detector gaps should have in plots
    :param invert_yaxis: True to invert the y axis of imshow plots
    :return: updated data, mask and controls
    """
    if original_data.ndim != 3 or updated_data.ndim != 3 or original_mask.ndim != 3:
        raise ValueError('original_data, updated_data and original_mask should be 3D arrays')
    if updated_mask.ndim != 2:
        raise ValueError('updated_mask should be 2D arrays')

    nbz, nby, nbx = original_data.shape
    stop_masking = False
    if dim not in [0, 1, 2]:
        raise ValueError('dim should be 0, 1 or 2')

    axs = figure.gca()
    xmin, xmax = axs.get_xlim()
    ymin, ymax = axs.get_ylim()

    if key == 'up':
        width = width + 1

    elif key == 'down':
        width = width - 1
        if width < 0:
            width = 0

    elif key == 'right':
        vmax = vmax + 1

    elif key == 'left':
        vmax = vmax - 1
        if vmax < 1:
            vmax = 1

    elif key == 'm':
        if (piy - width) < 0:
            starty = 0
        else:
            starty = piy - width
        if (pix - width) < 0:
            startx = 0
        else:
            startx = pix - width
        updated_mask[starty:piy + width + 1, startx:pix + width + 1] = 1

    elif key == 'b':
        if (piy - width) < 0:
            starty = 0
        else:
            starty = piy - width
        if (pix - width) < 0:
            startx = 0
        else:
            startx = pix - width
        updated_mask[starty:piy + width + 1, startx:pix + width + 1] = 0
        if dim == 0:
            updated_data[:, starty:piy + width + 1, startx:pix + width + 1] = \
                original_data[:, starty:piy + width + 1, startx:pix + width + 1]
        elif dim == 1:
            updated_data[starty:piy + width + 1, :, startx:pix + width + 1] = \
                original_data[starty:piy + width + 1, :, startx:pix + width + 1]
        else:  # dim=2
            updated_data[starty:piy + width + 1, startx:pix + width + 1, :] = \
                original_data[starty:piy + width + 1, startx:pix + width + 1, :]

    elif key == 'a':  # restart mask from beginning
        updated_data[:] = original_data[:]
        xy = []
        print('Restart masking...')
        if dim == 0:
            updated_data[
                original_mask == 1] = masked_color / nbz  # masked pixels plotted with the value of masked_pixel
            updated_mask = np.zeros((nby, nbx))
            xmin, xmax = -0.5, nbx - 0.5
            if invert_yaxis:
                ymin, ymax = -0.5, nby - 0.5  # pointing up
            else:
                ymin, ymax = nby - 0.5, -0.5  # pointing down
        elif dim == 1:
            updated_data[
                original_mask == 1] = masked_color / nby  # masked pixels plotted with the value of masked_pixel
            updated_mask = np.zeros((nbz, nbx))
            xmin, xmax = -0.5, nbx - 0.5
            ymin, ymax = nbz - 0.5, -0.5  # pointing down
        else:  # dim=2
            updated_data[
                original_mask == 1] = masked_color / nbx  # masked pixels plotted with the value of masked_pixel
            updated_mask = np.zeros((nbz, nby))
            xmin, xmax = -0.5, nby - 0.5
            ymin, ymax = nbz - 0.5, -0.5  # pointing down

    elif key == 'p':  # plot full image
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

    elif key == 'r':
        xy = []

    elif key == 'x':
        if not flag_pause:
            flag_pause = True
            print('pause for pan/zoom')
        else:
            flag_pause = False
            print('resume masking')

    elif key == 'q':
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
    axs.set_title('x to pause/resume masking for pan/zoom \n'
                  'p plot mask ; a restart ; click to select vertices\n'
                  "m mask ; b unmask ; q quit ; u next frame ; d previous frame\n"
                  "up larger ; down smaller ; right darker ; left brighter")
    plt.draw()

    return updated_data, updated_mask, flag_pause, xy, width, vmax, stop_masking


def update_mask_combined(key, pix, piy, original_data, original_mask, updated_data, updated_mask, axes, flag_pause,
                         points, xy, width, dim, click_dim, info_text, vmax, vmin=0, invert_yaxis=False):
    """
    Update the mask to remove parasitic diffraction intensity and hotpixels in 3D dataset.

    :param key: the keyboard key which was pressed
    :param pix: the x value of the mouse pointer
    :param piy: the y value of the mouse pointer
    :param original_data: the 3D data array before masking
    :param original_mask: the 3D mask array before masking
    :param updated_data: the current 3D data array
    :param updated_mask: the temporary 3D mask array with updated points
    :param axes: tuple of the 4 axes instances in a plt.subplots(nrows=2, ncols=2)
    :param flag_pause: set to 1 to stop registering vertices using mouse clicks
    :param points: list of all point coordinates: points=np.stack((x, y), axis=0).T with x=x.flatten() , y = y.flatten()
     given x,y=np.meshgrid(np.arange(nx), np.arange(ny))
    :param xy: the list of vertices which defines a polygon to be masked
    :param width: the half_width of the masking window
    :param dim: the axis currently under review (axis 0, 1 or 2)
    :param click_dim: the dimension (0, 1 or 2) here the selection of mask polygon vertices by clicking was performed
    :param info_text: text instance in the figure
    :param vmax: the higher boundary for the colorbar
    :param vmin: the lower boundary for the colorbar
    :param invert_yaxis: True to invert the y axis of imshow plots
    :return: updated data, mask and controls
    """
    if original_data.ndim != 3 or updated_data.ndim != 3 or original_mask.ndim != 3 or updated_mask.ndim != 3:
        raise ValueError('original_data, updated_data and original_mask should be 3D arrays')

    nbz, nby, nbx = original_data.shape
    stop_masking = False
    if dim not in [0, 1, 2]:
        raise ValueError('dim should be 0, 1 or 2')

    xmin0, xmax0 = axes[0].get_xlim()
    ymin0, ymax0 = axes[0].get_ylim()
    xmin1, xmax1 = axes[1].get_xlim()
    ymin1, ymax1 = axes[1].get_ylim()
    xmin2, xmax2 = axes[2].get_xlim()
    ymin2, ymax2 = axes[2].get_ylim()

    if key == 'up':
        width = width + 1

    elif key == 'down':
        width = width - 1
        if width < 0:
            width = 0

    elif key == 'right':
        vmax = vmax + 1

    elif key == 'left':
        vmax = vmax - 1
        if vmax < 1:
            vmax = 1

    elif key == 'm':
        if (piy - width) < 0:
            starty = 0
        else:
            starty = piy - width
        if (pix - width) < 0:
            startx = 0
        else:
            startx = pix - width
        if dim == 0:
            updated_mask[:, starty:piy + width + 1, startx:pix + width + 1] = 1
        elif dim == 1:
            updated_mask[starty:piy + width + 1, :, startx:pix + width + 1] = 1
        else:  # dim=2
            updated_mask[starty:piy + width + 1, startx:pix + width + 1, :] = 1

    elif key == 'b':
        if (piy - width) < 0:
            starty = 0
        else:
            starty = piy - width
        if (pix - width) < 0:
            startx = 0
        else:
            startx = pix - width
        if dim == 0:
            updated_mask[:, starty:piy + width + 1, startx:pix + width + 1] = 0
            updated_data[:, starty:piy + width + 1, startx:pix + width + 1] =\
                original_data[:, starty:piy + width + 1, startx:pix + width + 1]
        elif dim == 1:
            updated_mask[starty:piy + width + 1, :, startx:pix + width + 1] = 0
            updated_data[starty:piy + width + 1, :, startx:pix + width + 1] = \
                original_data[starty:piy + width + 1, :, startx:pix + width + 1]
        else:  # dim=2
            updated_mask[starty:piy + width + 1, startx:pix + width + 1, :] = 0
            updated_data[starty:piy + width + 1, startx:pix + width + 1, :] = \
                original_data[starty:piy + width + 1, startx:pix + width + 1, :]

    elif key == 'a':  # restart mask from beginning
        updated_data = np.copy(original_data)
        xy = []
        click_dim = None
        print('Restart masking...')
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

    elif key == 'p':  # plot full image
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
                updated_mask[np.repeat(temp_mask[np.newaxis, :, :], repeats=nbz, axis=0) == 1] = 1
            elif click_dim == 1:
                ind = Path(np.array(xy)).contains_points(points).reshape((nbz, nbx))
                temp_mask = np.zeros((nbz, nbx))
                temp_mask[ind] = 1
                updated_mask[np.repeat(temp_mask[:, np.newaxis, :], repeats=nby, axis=1) == 1] = 1
            else:  # dim=2
                ind = Path(np.array(xy)).contains_points(points).reshape((nbz, nby))
                temp_mask = np.zeros((nbz, nby))
                temp_mask[ind] = 1
                updated_mask[np.repeat(temp_mask[:, :, np.newaxis], repeats=nbx, axis=2) == 1] = 1
        xy = []  # allow to mask a different area
        click_dim = None

    elif key == 'r':
        xy = []

    elif key == 'x':
        if not flag_pause:
            flag_pause = True
            print('pause for pan/zoom')
        else:
            flag_pause = False
            print('resume masking')

    elif key == 'q':
        stop_masking = True

    else:
        return updated_data, updated_mask, flag_pause, xy, width, vmax, click_dim, stop_masking

    updated_data[original_mask == 1] = 0
    updated_data[updated_mask == 1] = 0

    axes[0].cla()
    axes[1].cla()
    axes[2].cla()
    axes[0].imshow(np.log10(updated_data.sum(axis=0)), vmin=vmin, vmax=vmax)
    axes[1].imshow(np.log10(updated_data.sum(axis=1)), vmin=vmin, vmax=vmax)
    axes[2].imshow(np.log10(updated_data.sum(axis=2)), vmin=vmin, vmax=vmax)
    axes[0].set_title("XY")
    axes[0].axis('scaled')
    if invert_yaxis:
        axes[0].invert_yaxis()
    axes[0].set_xlim([xmin0, xmax0])
    axes[0].set_ylim([ymin0, ymax0])
    axes[1].set_title("XZ")
    axes[1].axis('scaled')
    axes[1].set_xlim([xmin1, xmax1])
    axes[1].set_ylim([ymin1, ymax1])
    axes[2].set_title("YZ")
    axes[2].axis('scaled')
    axes[2].set_xlim([xmin2, xmax2])
    axes[2].set_ylim([ymin2, ymax2])
    fig = plt.gcf()
    info_text.remove()
    if flag_pause:
        info_text = fig.text(0.6, 0.05, 'masking paused', size=16)
    else:
        info_text = fig.text(0.6, 0.05, 'masking enabled', size=16)
    plt.draw()

    return updated_data, updated_mask, flag_pause, xy, width, vmax, click_dim, stop_masking, info_text


def update_mask_2d(key, pix, piy, original_data, original_mask, updated_data, updated_mask, figure, flag_pause, points,
                   xy, width, vmax, vmin=0, masked_color=0.1, invert_yaxis=False):
    """
    Update the mask to remove parasitic diffraction intensity and hotpixels for 2d dataset.

    :param key: the keyboard key which was pressed
    :param pix: the x value of the mouse pointer
    :param piy: the y value of the mouse pointer
    :param original_data: the 2D data array before masking
    :param original_mask: the 2D mask array before masking
    :param updated_data: the current 2D data array
    :param updated_mask: the temporary 2D mask array with updated points
    :param figure: the figure instance
    :param flag_pause: set to 1 to stop registering vertices using mouse clicks
    :param points: list of all point coordinates: points=np.stack((x, y), axis=0).T with x=x.flatten() , y = y.flatten()
     given x,y=np.meshgrid(np.arange(nx), np.arange(ny))
    :param xy: the list of vertices which defines a polygon to be masked
    :param width: the half_width of the masking window
    :param vmax: the higher boundary for the colorbar
    :param vmin: the lower boundary for the colorbar
    :param masked_color: the value that detector gaps should have in plots
    :param invert_yaxis: True to invert the y axis of imshow plots
    :return: updated data, mask and controls
    """
    if original_data.ndim != 2 or updated_data.ndim != 2 or original_mask.ndim != 2 or updated_mask.ndim != 2:
        raise ValueError('original_data, updated_data, original_mask and updated_mask should be 2D arrays')

    nby, nbx = original_data.shape
    stop_masking = False

    axs = figure.gca()
    xmin, xmax = axs.get_xlim()
    ymin, ymax = axs.get_ylim()

    if key == 'up':
        width = width + 1

    elif key == 'down':
        width = width - 1
        if width < 0:
            width = 0

    elif key == 'right':
        vmax = vmax + 1
        updated_data[updated_mask == 1] = masked_color

    elif key == 'left':
        vmax = vmax - 1
        if vmax < 1:
            vmax = 1
        updated_data[updated_mask == 1] = masked_color

    elif key == 'm':
        if (piy - width) < 0:
            starty = 0
        else:
            starty = piy - width
        if (pix - width) < 0:
            startx = 0
        else:
            startx = pix - width
        updated_mask[starty:piy + width + 1, startx:pix + width + 1] = 1
        updated_data[updated_mask == 1] = masked_color

    elif key == 'b':
        if (piy - width) < 0:
            starty = 0
        else:
            starty = piy - width
        if (pix - width) < 0:
            startx = 0
        else:
            startx = pix - width
        updated_mask[starty:piy + width + 1, startx:pix + width + 1] = 0
        updated_data[updated_mask == 1] = masked_color

    elif key == 'a':  # restart mask from beginning
        updated_data = np.copy(original_data)
        xy = []
        print('restart masking')
        updated_data[
            original_mask == 1] = masked_color  # masked pixels plotted with the value of masked_pixel
        updated_mask = np.zeros((nby, nbx))
        xmin, xmax = -0.5, nbx - 0.5
        if invert_yaxis:
            ymin, ymax = -0.5, nby - 0.5  # pointing up
        else:
            ymin, ymax = nby - 0.5, -0.5  # pointing down

    elif key == 'p':  # plot full image
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

    elif key == 'r':
        xy = []

    elif key == 'x':
        if not flag_pause:
            flag_pause = True
            print('pause for pan/zoom')
        else:
            flag_pause = False
            print('resume masking')

    elif key == 'q':
        stop_masking = True

    else:
        return updated_data, updated_mask, flag_pause, xy, width, vmax, stop_masking

    axs.cla()
    axs.imshow(np.log10(abs(updated_data)), vmin=vmin, vmax=vmax)
    if invert_yaxis:
        axs.invert_yaxis()
    axs.set_xlim([xmin, xmax])
    axs.set_ylim([ymin, ymax])
    axs.set_title('x to pause/resume masking for pan/zoom \n'
                  'p plot mask ; a restart ; click to select vertices\n'
                  "m mask ; b unmask ; q quit ; u next frame ; d previous frame\n"
                  "up larger ; down smaller ; right darker ; left brighter")
    plt.draw()

    return updated_data, updated_mask, flag_pause, xy, width, vmax, stop_masking


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
