# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import h5py
import numpy as np
from numpy.fft import fftn, fftshift, ifftn, ifftshift
import matplotlib.pyplot as plt
import sys
sys.path.append('//win.desy.de/home/carnisj/My Documents/myscripts/bcdi/')
import bcdi.graph.graph_utils as gu
from bcdi.utils import image_registration as reg
from scipy.ndimage.measurements import center_of_mass
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import multivariate_normal
from scipy.stats import pearsonr
import gc
import os


def align_obj(avg_obj, ref_obj, obj, support_threshold=0.25, correlation_threshold=0.90, aligning_option='dft',
              width_z=np.nan, width_y=np.nan, width_x=np.nan, debugging=False):
    """
    Align two reconstructions by interpolating it based on COM offset, if their cross-correlation is larger than
    correlation_threshold.

    :param avg_obj: 3D array, average complex density
    :param ref_obj: 3D array, reference complex object
    :param obj: 3D array, complex density to average with
    :param support_threshold: for support definition
    :param correlation_threshold: minimum correlation between two dataset to average them
    :param aligning_option: 'com' for center of mass, 'dft' for dft registration and subpixel shift
    :param width_z: size of the area to plot in z (axis 0), centered on the middle of the initial array
    :param width_y: size of the area to plot in y (axis 1), centered on the middle of the initial array
    :param width_x: size of the area to plot in x (axis 2), centered on the middle of the initial array
    :param debugging: set to True to see plots
    :type debugging: bool
    :return: the average complex density
    """
    if obj.ndim != 3 or avg_obj.ndim != 3 or ref_obj.ndim != 3:
        raise ValueError('avg_obj, ref_obj and obj should be 3D arrays')
    if obj.shape != avg_obj.shape or obj.shape != ref_obj.shape:
        raise ValueError('avg_obj, ref_obj and obj must have the same shape\n'
                         'avg_obj is ', avg_obj.shape, ' - ref_obj is ', ref_obj.shape, ' - obj is ', obj.shape)

    nbz, nby, nbx = obj.shape
    avg_flag = 0
    if avg_obj.sum() == 0:
        avg_obj = ref_obj
        if debugging:
            gu.multislices_plot(abs(avg_obj), width_z=width_z, width_y=width_y, width_x=width_x,
                                sum_frames=True, invert_yaxis=True, title='Reference object')
    else:
        myref_support = np.zeros((nbz, nby, nbx))
        myref_support[abs(ref_obj) > support_threshold*abs(ref_obj).max()] = 1
        my_support = np.zeros((nbz, nby, nbx))
        my_support[abs(obj) > support_threshold * abs(obj).max()] = 1
        avg_piz, avg_piy, avg_pix = center_of_mass(abs(myref_support))
        piz, piy, pix = center_of_mass(abs(my_support))
        offset_z = avg_piz - piz
        offset_y = avg_piy - piy
        offset_x = avg_pix - pix
        print("center of mass offset with reference object: (", str('{:.2f}'.format(offset_z)), ',',
              str('{:.2f}'.format(offset_y)), ',', str('{:.2f}'.format(offset_x)), ') pixels')
        if aligning_option is 'com':
            # re-sample data on a new grid based on COM shift of support
            old_z = np.arange(-nbz // 2, nbz // 2)
            old_y = np.arange(-nby // 2, nby // 2)
            old_x = np.arange(-nbx // 2, nbx // 2)
            myz, myy, myx = np.meshgrid(old_z, old_y, old_x, indexing='ij')
            new_z = myz + offset_z
            new_y = myy + offset_y
            new_x = myx + offset_x
            del myx, myy, myz
            rgi = RegularGridInterpolator((old_z, old_y, old_x), obj, method='linear', bounds_error=False,
                                          fill_value=0)
            new_obj = rgi(np.concatenate((new_z.reshape((1, new_z.size)), new_y.reshape((1, new_z.size)),
                                          new_x.reshape((1, new_z.size)))).transpose())
            new_obj = new_obj.reshape((nbz, nby, nbx)).astype(obj.dtype)
        else:
            # dft registration and subpixel shift (see Matlab code)
            shiftz, shifty, shiftx = reg.getimageregistration(abs(ref_obj), abs(obj), precision=1000)
            new_obj = reg.subpixel_shift(obj, shiftz, shifty, shiftx)  # keep the complex output here
            print("Shift calculated from dft registration: (", str('{:.2f}'.format(shiftz)), ',',
                  str('{:.2f}'.format(shifty)), ',', str('{:.2f}'.format(shiftx)), ') pixels')

        new_obj = new_obj / abs(new_obj).max()  # renormalize

        correlation = pearsonr(np.ndarray.flatten(abs(ref_obj[np.nonzero(myref_support)])),
                               np.ndarray.flatten(abs(new_obj[np.nonzero(myref_support)])))[0]

        if correlation < correlation_threshold:
            print('pearson cross-correlation=', correlation, 'too low, skip this reconstruction')
        else:
            print('pearson-correlation=', correlation, ', average with this reconstruction')

            if debugging:

                myfig, _, _ = gu.multislices_plot(abs(new_obj), width_z=width_z, width_y=width_y, width_x=width_x,
                                                  sum_frames=True, invert_yaxis=True, title='Aligned object')
                myfig.text(0.60, 0.30, "pearson-correlation = " + str('{:.4f}'.format(correlation)), size=20)

            avg_obj = avg_obj + new_obj
            avg_flag = 1

        if debugging:
            gu.multislices_plot(abs(avg_obj), width_z=width_z, width_y=width_y, width_x=width_x,
                                sum_frames=True, invert_yaxis=True, title='New averaged object')

    return avg_obj, avg_flag


def apodize(amp, phase, initial_shape, window_type, debugging=False, **kwargs):
    """
    Apodize the complex array based on the window of the same shape.

    :param amp: 3D array, amplitude before apodization
    :param phase: 3D array, phase before apodization
    :param initial_shape: shape of the FFT used for phasing
    :param window_type: window filtering function, 'normal' or 'tukey' or 'blackman'
    :param debugging: set to True to see plots
    :type debugging: bool
    :param kwargs:
     - if 'normal': sigma and mu of the 3d multivariate normal distribution, tuples of 3 floats
     - if 'tukey': alpha (shape parameter) of the 3d Tukey window, tuple of 3 floats
    :return: filtered amplitude, phase of the same shape as myamp
    """
    if amp.ndim != 3 or phase.ndim != 3:
        raise ValueError('amp and phase should be 3D arrays')
    if amp.shape != phase.shape:
        raise ValueError('amp and phase must have the same shape\n'
                         'amp is ', amp.shape, ' while phase is ', phase.shape)

    for k in kwargs.keys():
        if k in ['sigma']:
            sigma = kwargs['sigma']
        elif k in ['mu']:
            mu = kwargs['mu']
        elif k in ['alpha']:
            alpha = kwargs['alpha']
        else:
            raise Exception("unknown keyword argument given: allowed is"
                            "'fix_bragg', 'fix_size', 'pad_size' and 'q_values'")

    nb_z, nb_y, nb_x = amp.shape
    nbz, nby, nbx = initial_shape
    myobj = crop_pad(amp * np.exp(1j * phase), (nbz, nby, nbx))
    del amp, phase
    gc.collect()
    if debugging:
        plt.figure()
        plt.imshow(abs(myobj[nbz // 2, :, :]))
        plt.pause(0.1)
    my_fft = fftshift(fftn(myobj))
    del myobj
    gc.collect()
    fftmax = abs(my_fft).max()
    print('Max FFT=', fftmax)
    if debugging:
        plt.figure()
        plt.imshow(np.log10(abs(my_fft[nbz // 2, :, :])), vmin=0, vmax=np.log10(fftmax))
        plt.colorbar()
        plt.pause(0.1)

    if window_type == 'normal':
        print('Apodization using a 3d multivariate normal window')
        try:
            sigma
        except NameError:  # sigma not declared
            sigma = np.array([0.3, 0.3, 0.3])
            print('defaulting sigma parameter')
        try:
            mu
        except NameError:  # mu not declared
            mu = np.array([0.0, 0.0, 0.0])
            print('defaulting mu parameter')

        grid_z, grid_y, grid_x = np.meshgrid(np.linspace(-1, 1, nbz), np.linspace(-1, 1, nby), np.linspace(-1, 1, nbx),
                                             indexing='ij')
        covariance = np.diag(sigma ** 2)
        window = multivariate_normal.pdf(np.column_stack([grid_z.flat, grid_y.flat, grid_x.flat]), mean=mu,
                                         cov=covariance)
        del grid_z, grid_y, grid_x
        gc.collect()
        window = window.reshape((nbz, nby, nbx))

    elif window_type == 'tukey':
        print('Apodization using a 3d Tukey window')
        try:
            alpha
        except NameError:  # alpha not declared
            alpha = np.array([0.5, 0.5, 0.5])
            print('defaulting alpha parameter')

        window = tukey_window(initial_shape, alpha=alpha)

    elif window_type == 'blackman':
        print('Apodization using a 3d Blackman window')
        window = blackman_window(initial_shape)

    else:
        raise ValueError('Invalid window type')

    my_fft = np.multiply(my_fft, window)
    del window
    gc.collect()
    my_fft = my_fft * fftmax / abs(my_fft).max()
    print('Max apodized FFT after normalization =', abs(my_fft).max())
    if debugging:
        plt.figure()
        plt.imshow(np.log10(abs(my_fft[nbz // 2, :, :])), vmin=0, vmax=np.log10(fftmax))
        plt.colorbar()
        plt.pause(0.1)
    myobj = ifftn(ifftshift(my_fft))
    del my_fft
    gc.collect()
    if debugging:
        plt.figure()
        plt.imshow(abs(myobj[nbz // 2, :, :]))
        plt.pause(0.1)
    myobj = crop_pad(myobj, (nb_z, nb_y, nb_x))  # return to the initial shape of myamp
    return abs(myobj), np.angle(myobj)


def blackman_window(shape):
    """
    Create a 3d Blackman window based on shape.

    :param shape: tuple, shape of the 3d window
    :return: the 3d Blackman window
    """
    nbz, nby, nbx = shape
    array_z = np.blackman(nbz)
    array_y = np.blackman(nby)
    array_x = np.blackman(nbx)
    blackman2 = np.ones((nbz, nby))
    blackman3 = np.ones((nbz, nby, nbx))
    for idz in range(nbz):
        blackman2[idz, :] = array_z[idz] * array_y
        for idy in range(nby):
            blackman3[idz, idy] = blackman2[idz, idy] * array_x
    return blackman3


def bragg_temperature(spacing, reflection, spacing_ref=None, temperature_ref=None, use_q=False, material=None):
    """
    Calculate the temperature from Bragg peak position.

    :param spacing: q or planar distance, in inverse angstroms or angstroms
    :param reflection: measured reflection, e.g. np.array([1, 1, 1])
    :param spacing_ref: reference spacing at known temperature (include substrate-induced strain)
    :param temperature_ref: in K, known temperature for the reference spacing
    :param use_q: set to True to use q, False to use planar distance
    :type use_q: bool
    :param material: at the moment only 'Pt'
    :return: calculated temperature
    """
    if material == 'Pt':
        # reference values for Pt: temperature in K, thermal expansion x 10^6 in 1/K, lattice parameter in angstroms
        expansion_data = np.array([[100, 6.77, 3.9173], [110, 7.10, 3.9176], [120, 7.37, 3.9179], [130, 7.59, 3.9182],
                                  [140, 7.78, 3.9185], [150, 7.93, 3.9188], [160, 8.07, 3.9191], [180, 8.29, 3.9198],
                                  [200, 8.46, 3.9204], [220, 8.59, 3.9211], [240, 8.70, 3.9218], [260, 8.80, 3.9224],
                                  [280, 8.89, 3.9231], [293.15, 8.93, 3.9236], [300, 8.95, 3.9238], [400, 9.25, 3.9274],
                                  [500, 9.48, 3.9311], [600, 9.71, 3.9349], [700, 9.94, 3.9387], [800, 10.19, 3.9427],
                                  [900, 10.47, 3.9468], [1000, 10.77, 3.9510], [1100, 11.10, 3.9553],
                                  [1200, 11.43, 3.9597]])
        if spacing_ref is None:
            print('Using the reference spacing of Platinum')
            spacing_ref = 3.9236 / np.linalg.norm(reflection)  # angstroms
        if temperature_ref is None:
            temperature_ref = 293.15  # K
    else:
        return 0
    if use_q:
        spacing = 2 * np.pi / spacing  # go back to distance
        spacing_ref = 2 * np.pi / spacing_ref  # go back to distance
    spacing = spacing * np.linalg.norm(reflection)  # go back to lattice constant
    spacing_ref = spacing_ref * np.linalg.norm(reflection)  # go back to lattice constant
    print('Reference spacing at', temperature_ref, 'K   =', str('{:.4f}'.format(spacing_ref)), 'angstroms')
    print('Spacing =', str('{:.4f}'.format(spacing)), 'angstroms using reflection', reflection)

    # fit the experimental spacing with non corrected platinum curve
    myfit = np.poly1d(np.polyfit(expansion_data[:, 2], expansion_data[:, 0], 3))
    print('Temperature without offset correction=', int(myfit(spacing) - 273.15), 'C')

    # find offset for platinum reference curve
    myfit = np.poly1d(np.polyfit(expansion_data[:, 0], expansion_data[:, 2], 3))
    spacing_offset = myfit(temperature_ref) - spacing_ref  # T in K, spacing in angstroms
    print('Spacing offset =', str('{:.4f}'.format(spacing_offset)), 'angstroms')

    # correct the platinum reference curve for the offset
    platinum_offset = np.copy(expansion_data)
    platinum_offset[:, 2] = platinum_offset[:, 2] - spacing_offset
    myfit = np.poly1d(np.polyfit(platinum_offset[:, 2], platinum_offset[:, 0], 3))
    mytemp = int(myfit(spacing) - 273.15)
    print('Temperature with offset correction=', mytemp, 'C')
    return mytemp


def bin_data(array, binning, debugging=False):
    """
    Bin the array using binning parameter.

    :param array: 2D or 3D array to be binned
    :param binning: tuple of binning factor, 1 for each dimension of array
    :param debugging: boolean, True to see plots
    :return: the binned array
    """
    if array.ndim != len(binning):
        raise ValueError('1 binning parameter expected for each dimension of array')
    if array.ndim == 2:
        nby, nbx = array.shape
        newarray = np.zeros((nby//binning[0], nbx//binning[1]), dtype=array.dtype)
        print('Initial array shape =', array.shape, 'Binned array shape =', newarray.shape)
        for idy in range(newarray.shape[0]):  # bin the vertical axis
            for idx in range(newarray.shape[1]):  # bin the horizontal axis
                newarray[idy, idx] = array[idy*binning[0]:(idy+1)*binning[0], idx*binning[1]:(idx+1)*binning[1]].sum()
    elif array.ndim == 3:
        nbz, nby, nbx = array.shape
        newarray = np.zeros((nbz//binning[0], nby//binning[1], nbx//binning[2]), dtype=array.dtype)
        print('Initial array shape =', array.shape)
        print('Binned array shape =', newarray.shape)
        for idz in range(newarray.shape[0]):  # bin axis 0
            for idy in range(newarray.shape[1]):  # bin the vertical axis
                for idx in range(newarray.shape[2]):  # bin the horizontal axis
                    newarray[idz, idy, idx] = array[idz*binning[0]:(idz+1)*binning[0],
                                                    idy*binning[1]:(idy+1)*binning[1],
                                                    idx*binning[2]:(idx+1)*binning[2]].sum()
    else:
        raise ValueError('The input array should be 2D or 3D')

    if debugging:
        gu.combined_plots(tuple_array=(array, newarray), tuple_sum_frames=False, tuple_sum_axis=(1, 1),
                          tuple_colorbar=True, tuple_width_v=np.nan, tuple_width_h=np.nan, tuple_vmin=0,
                          tuple_vmax=np.nan, tuple_title=('array', 'binned array'),
                          tuple_scale='log', reciprocal_space=True)
    return newarray


def calc_coordination(support, kernel=np.ones((3, 3, 3)), width_z=np.nan, width_y=np.nan, width_x=np.nan,
                      debugging=False):
    """
    Calculate the coordination number of voxels in a support (numbe of neighbours).

    :param support: 3D support array
    :param kernel: kernel used for convolution with the support
    :param width_z: size of the area to plot in z (axis 0), centered on the middle of the initial array
    :param width_y: size of the area to plot in y (axis 1), centered on the middle of the initial array
    :param width_x: size of the area to plot in x (axis 2), centered on the middle of the initial array
    :param debugging: set to True to see plots
    :type debugging: bool
    :return: the coordination matrix
    """
    from scipy.signal import convolve

    if support.ndim != 3:
        raise ValueError('Support should be a 3D array')

    mycoord = np.rint(convolve(support, kernel, mode='same'))
    mycoord = mycoord.astype(int)

    if debugging:
        gu.multislices_plot(mycoord, width_z=width_z, width_y=width_y, width_x=width_x,
                            invert_yaxis=True, vmin=0, title='Coordination matrix')
    return mycoord


def center_com(array, width_z=np.nan, width_y=np.nan, width_x=np.nan, debugging=False):
    """
    Center array based on center_of_mass(abs(array)) using pixel shift.

    :param array: 3D array to be centered based on the center of mass of abs(array)
    :param width_z: size of the area to plot in z (axis 0), centered on the middle of the initial array
    :param width_y: size of the area to plot in y (axis 1), centered on the middle of the initial array
    :param width_x: size of the area to plot in x (axis 2), centered on the middle of the initial array
    :param debugging: set to True to see plots
    :type debugging: bool
    :return: array centered by pixel shift
    """
    if array.ndim != 3:
        raise ValueError('array should be a 3D array')

    nbz, nby, nbx = array.shape

    if debugging:
        gu.multislices_plot(abs(array), width_z=width_z, width_y=width_y, width_x=width_x,
                            invert_yaxis=True, title='Before COM centering')

    piz, piy, pix = center_of_mass(abs(array))
    print("center of mass at (z, y, x): (", str('{:.2f}'.format(piz)), ',',
          str('{:.2f}'.format(piy)), ',', str('{:.2f}'.format(pix)), ')')
    offset_z = int(np.rint(nbz / 2.0 - piz))
    offset_y = int(np.rint(nby / 2.0 - piy))
    offset_x = int(np.rint(nbx / 2.0 - pix))
    print("center of mass offset: (", offset_z, ',', offset_y, ',', offset_x, ') pixels')
    array = np.roll(array, (offset_z, offset_y, offset_x), axis=(0, 1, 2))

    if debugging:
        gu.multislices_plot(abs(array), width_z=width_z, width_y=width_y, width_x=width_x,
                            invert_yaxis=True, title='After COM centering')
    return array


def center_max(array, width_z=np.nan, width_y=np.nan, width_x=np.nan, debugging=False):
    """
    Center array based on max(abs(array)) using pixel shift.

    :param array: 3D array to be centered based on max(abs(array))
    :param width_z: size of the area to plot in z (axis 0), centered on the middle of the initial array
    :param width_y: size of the area to plot in y (axis 1), centered on the middle of the initial array
    :param width_x: size of the area to plot in x (axis 2), centered on the middle of the initial array
    :param debugging: set to True to see plots
    :type debugging: bool
    :return: array centered by pixel shift
    """
    if array.ndim != 3:
        raise ValueError('array should be a 3D array')

    nbz, nby, nbx = array.shape

    if debugging:
        gu.multislices_plot(abs(array), width_z=width_z, width_y=width_y, width_x=width_x,
                            invert_yaxis=True, title='Before max centering')

    piz, piy, pix = np.unravel_index(abs(array).argmax(), array.shape)
    print("Max at (z, y, x): (", piz, ',', piy, ',', pix, ')')
    offset_z = int(np.rint(nbz / 2.0 - piz))
    offset_y = int(np.rint(nby / 2.0 - piy))
    offset_x = int(np.rint(nbx / 2.0 - pix))
    print("Max offset: (", offset_z, ',', offset_y, ',', offset_x, ') pixels')
    array = np.roll(array, (offset_z, offset_y, offset_x), axis=(0, 1, 2))

    if debugging:
        gu.multislices_plot(abs(array), width_z=width_z, width_y=width_y, width_x=width_x,
                            invert_yaxis=True, title='After max centering')
    return array


def crop_pad(array, output_shape, width_z=np.nan, width_y=np.nan, width_x=np.nan, debugging=False):
    """
    Crop or pad the 3D object depending on output_shape.

    :param array: 3D complex array to be padded
    :param output_shape: list of desired output shape [z, y, x]
    :param width_z: size of the area to plot in z (axis 0), centered on the middle of the initial array
    :param width_y: size of the area to plot in y (axis 1), centered on the middle of the initial array
    :param width_x: size of the area to plot in x (axis 2), centered on the middle of the initial array
    :param debugging: set to True to see plots
    :type debugging: bool
    :return: myobj cropped or padded with zeros
    """
    if array.ndim != 3:
        raise ValueError('array should be a 3D array')

    nbz, nby, nbx = array.shape
    newz, newy, newx = output_shape

    if debugging:
        gu.multislices_plot(abs(array), width_z=width_z, width_y=width_y, width_x=width_x,
                            invert_yaxis=True, title='Before crop/pad')
    # z
    if newz >= nbz:  # pad
        temp_z = np.zeros((output_shape[0], nby, nbx), dtype=array.dtype)
        temp_z[(newz - nbz) // 2:(newz + nbz) // 2, :, :] = array
    else:  # crop
        temp_z = array[(nbz - newz) // 2:(newz + nbz) // 2, :, :]
    # y
    if newy >= nby:  # pad
        temp_y = np.zeros((newz, newy, nbx), dtype=array.dtype)
        temp_y[:, (newy - nby) // 2:(newy + nby) // 2, :] = temp_z
    else:  # crop
        temp_y = temp_z[:, (nby - newy) // 2:(newy + nby) // 2, :]
    # x
    if newx >= nbx:  # pad
        newobj = np.zeros((newz, newy, newx), dtype=array.dtype)
        newobj[:, :, (newx - nbx) // 2:(newx + nbx) // 2] = temp_y
    else:  # crop
        newobj = temp_y[:, :, (nbx - newx) // 2:(newx + nbx) // 2]

    if debugging:
        gu.multislices_plot(abs(newobj), width_z=width_z, width_y=width_y, width_x=width_x,
                            invert_yaxis=True, title='After crop/pad')
    return newobj


def crop_pad_2d(array, output_shape, width_y=np.nan, width_x=np.nan, debugging=False):
    """
    Crop or pad the 2D object depending on output_shape.

    :param array: 2D complex array to be padded
    :param output_shape: list of desired output shape [y, x]
    :param width_y: size of the area to plot in y (axis 1), centered on the middle of the initial array
    :param width_x: size of the area to plot in x (axis 2), centered on the middle of the initial array
    :param debugging: set to True to see plots
    :type debugging: bool
    :return: myobj cropped or padded with zeros
    """
    if array.ndim != 2:
        raise ValueError('array should be a 2D array')

    nby, nbx = array.shape
    newy, newx = output_shape

    if debugging:
        gu.imshow_plot(abs(array), width_v=width_y, width_h=width_x, title='Before crop/pad')
    # y
    if newy >= nby:  # pad
        temp_y = np.zeros((output_shape[0], nbx), dtype=array.dtype)
        temp_y[(newy - nby) // 2:(newy + nby) // 2, :, :] = array
    else:  # crop
        temp_y = array[(nby - newy) // 2:(newy + nby) // 2, :, :]
    # x
    if newx >= nbx:  # pad
        newobj = np.zeros((newy, newx), dtype=array.dtype)
        newobj[:, (newx - nbx) // 2:(newx + nbx) // 2] = temp_y
    else:  # crop
        newobj = temp_y[:, (nbx - newx) // 2:(newx + nbx) // 2]

    if debugging:
        gu.imshow_plot(abs(array), width_v=width_y, width_h=width_x, title='After crop/pad')
    return newobj


def find_bulk(amp, support_threshold, method='threshold', width_z=np.nan, width_y=np.nan, width_x=np.nan,
              debugging=False):
    """
    Isolate the inner part of the crystal from the non-physical surface.

    :param amp: 3D array, reconstructed object amplitude
    :param support_threshold:  threshold for isosurface determination
    :param method: 'threshold' or 'defect'. If 'defect', removes layer by layer using the coordination number.
    :param width_z: size of the area to plot in z (axis 0), centered on the middle of the initial array
    :param width_y: size of the area to plot in y (axis 1), centered on the middle of the initial array
    :param width_x: size of the area to plot in x (axis 2), centered on the middle of the initial array
    :param debugging: set to True to see plots
    :type debugging: bool
    :return: the support corresponding to the bulk
    """
    if amp.ndim != 3:
        raise ValueError('amp should be a 3D array')

    nbz, nby, nbx = amp.shape
    mysupport = np.ones((nbz, nby, nbx))

    if method == 'threshold':
        mysupport[abs(amp) < support_threshold * abs(amp).max()] = 0
        mybulk = mysupport

    else:
        mysupport[abs(amp) < 0.01 * abs(amp).max()] = 0
        mykernel = np.ones((9, 9, 9))
        mycoordination_matrix = calc_coordination(mysupport, kernel=mykernel, debugging=False)
        outer = np.copy(mycoordination_matrix)
        outer[np.nonzero(outer)] = 1
        if mykernel.shape == np.ones((3, 3, 3)).shape:
            outer[mycoordination_matrix > 20] = 0  # remove the bulk
        elif mykernel.shape == np.ones((9, 9, 9)).shape:
            outer[mycoordination_matrix > 430] = 0  # remove the bulk, threshold=430 for kernel (9, 9, 9)
        else:
            raise ValueError('Kernel not yet implemented')

        outer[mycoordination_matrix == 0] = 1  # corresponds to outside of the crystal
        if debugging:
            gu.multislices_plot(outer, width_z=width_z, width_y=width_y, width_x=width_x,
                                invert_yaxis=True, vmin=0, vmax=1, title='Outer matrix')

        nb_voxels = 1  # initialize this counter which corresponds to the nb of voxels not included in outer
        idx = 0
        # is larger than mythreshold
        while nb_voxels > 0:  # nb of voxels not included in outer
            # first step: find the first underlayer
            mycoordination_matrix = calc_coordination(outer, kernel=mykernel, debugging=debugging)
            surface = np.copy(mycoordination_matrix)
            surface[np.nonzero(surface)] = 1
            surface[mycoordination_matrix > 450] = 0  # remove part from outer  420
            surface[mycoordination_matrix < 290] = 0  # remove part from bulk   290
            surface[0:5, :, :] = 0
            surface[:, 0:5, :] = 0
            surface[:, :, 0:5] = 0
            surface[nbz - 6:nbz, :, :] = 0
            surface[:, nby - 6:nby, :] = 0
            surface[:, :, nbx - 6:nbx] = 0
            if debugging:
                gu.multislices_plot(surface, width_z=width_z, width_y=width_y, width_x=width_x,
                                    invert_yaxis=True, vmin=0, vmax=1, title='Surface matrix')

            # second step: calculate the % of voxels from that layer whose amplitude is lower than support_threshold
            nb_voxels = surface[np.nonzero(surface)].sum()
            keep_voxels = surface[abs(amp) >= support_threshold * abs(amp).max()].sum()
            voxels_counter = keep_voxels / nb_voxels  # % of voxels whose amplitude is larger than support_threshold
            print('% of surface voxels above threshold = ', str('{:.2f}'.format(100 * voxels_counter)), '%')
            if voxels_counter < 0.90:  # surface reached only if 90% of voxels are above support_threshold
                outer[np.nonzero(surface)] = 1
                idx = idx + 1
            else:
                print('Surface of object reached after', idx, 'iterations')
                break
        mybulk = np.ones((nbz, nby, nbx)) - outer
    return mybulk


def find_datarange(array, plot_margin, amplitude_threshold=0.1, keep_size=False):
    """
    Find the meaningful range of the data, in order to reduce the memory consumption when manipulating the object. The
    range can be larger than the initial data size, which then will need to be padded.

    :param array: the complex 3D reconstruction
    :param plot_margin: user-defined margin to add to the minimum range of the data
    :param amplitude_threshold: threshold used to define a support from the amplitude
    :param keep_size: set to True in order to keep the dataset full size
    :return:
     - zrange: half size of the data range to use in the first axis (Z)
     - yrange: half size of the data range to use in the second axis (Y)
     - xrange: half size of the data range to use in the third axis (X)
    """
    nbz, nby, nbx = array.shape

    if keep_size:
        return nbz // 2, nby // 2, nbx // 2

    else:
        support = np.zeros((nbz, nby, nbx))
        support[abs(array) > amplitude_threshold * abs(array).max()] = 1

        z, y, x = np.meshgrid(np.arange(0, nbz, 1), np.arange(0, nby, 1), np.arange(0, nbx, 1),
                              indexing='ij')
        z = z * support
        min_z = min(int(np.min(z[np.nonzero(z)])), nbz - int(np.max(z[np.nonzero(z)])))

        y = y * support
        min_y = min(int(np.min(y[np.nonzero(y)])), nby - int(np.max(y[np.nonzero(y)])))

        x = x * support
        min_x = min(int(np.min(x[np.nonzero(x)])), nbx - int(np.max(x[np.nonzero(x)])))

        zrange = (nbz // 2 - min_z) + plot_margin[0]
        yrange = (nby // 2 - min_y) + plot_margin[1]
        xrange = (nbx // 2 - min_x) + plot_margin[2]

        return zrange, yrange, xrange


def flip_reconstruction(obj, debugging=False):
    """
    Calculate the conjugate object  giving the same diffracted intensity as 'obj'.

    :param obj: 3D reconstructed complex object
    :param debugging: set to True to see plots
    :type debugging: bool
    :return: the flipped complex object
    """
    if obj.ndim != 3:
        raise ValueError('obj should be a 3D array')

    flipped_obj = ifftn(ifftshift(np.conj(fftshift(fftn(obj)))))
    if debugging:
        gu.multislices_plot(abs(obj), vmin=0, sum_frames=False, invert_yaxis=True, plot_colorbar=True,
                            title='Initial object')
        gu.multislices_plot(abs(flipped_obj), vmin=0, sum_frames=False, invert_yaxis=True, plot_colorbar=True,
                            title='Flipped object')
    return flipped_obj


def get_opticalpath(support, direction, k, width_z=np.nan, width_y=np.nan, width_x=np.nan,
                    debugging=False):
    """
    Calculate the optical path for refraction/absorption corrections in the crystal. 'k' should be in the same basis
    (crystal or laboratory frame) as the data. For xrayutilities, the data is orthogonalized in crystal frame.

    :param support: 3D array, support used for defining the object
    :param direction: "in" or "out" , incident or diffracted wave
    :param k: vector for the incident or diffracted wave depending on direction (xrayutils_orthogonal=True case)
    :param width_z: size of the area to plot in z (axis 0), centered on the middle of the initial array
    :param width_y: size of the area to plot in y (axis 1), centered on the middle of the initial array
    :param width_x: size of the area to plot in x (axis 2), centered on the middle of the initial array
    :param debugging: set to True to see plots
    :type debugging: bool
    :return: the optical path, of the same shape as mysupport
    """
    if support.ndim != 3:
        raise ValueError('support should be a 3D array')

    nbz, nby, nbx = support.shape
    path = np.zeros((nbz, nby, nbx))
    if debugging:
        gu.multislices_plot(support, width_z=width_z, width_y=width_y, width_x=width_x, vmin=0, vmax=1,
                            sum_frames=False, invert_yaxis=True, title='Support for optical path')

    indices_support = np.nonzero(support)
    min_z = indices_support[0].min()
    max_z = indices_support[0].max() + 1  # last point not included in range()
    min_y = indices_support[1].min()
    max_y = indices_support[1].max() + 1  # last point not included in range()
    min_x = indices_support[2].min()
    max_x = indices_support[2].max() + 1  # last point not included in range()
    print("Support limits (start_z, stop_z, start_y, stop_y, start_x, stop_x):(",
          min_z, ',', max_z, ',', min_y, ',', max_y, ',', min_x, ',', max_x, ')')

    if direction == "in":
        k_norm = -1 * k / np.linalg.norm(k)  # we will work with -k_in
        if (k_norm == np.array([-1, 0, 0])).all():  # data orthogonalized in laboratory frame, k_in along axis 0
            for idz in range(min_z, max_z, 1):
                path[idz, :, :] = support[0:idz+1, :, :].sum(axis=0)  # include also the pixel
            path = np.multiply(path, support)

        else:  # data orthogonalized in crystal frame (xrayutilities), k_in is not along any array axis
            for idz in range(min_z, max_z, 1):
                for idy in range(min_y, max_y, 1):
                    for idx in range(min_x, max_x, 1):
                        if support[idz, idy, idx] == 1:
                            stop_flag = False
                            counter = 1
                            pixel = np.array([idz, idy, idx])  # pixel for which the optical path is calculated
                            while not stop_flag:
                                pixel = pixel + k_norm  # add unitary translation in -k_in direction
                                coords = np.rint(pixel)
                                stop_flag = True
                                if (min_z <= coords[0] <= max_z) and (min_y <= coords[1] <= max_y) and\
                                        (min_x <= coords[2] <= max_x):
                                    counter = counter + support[int(coords[0]), int(coords[1]), int(coords[2])]
                                    stop_flag = False
                            path[idz, idy, idx] = counter
                        else:  # point outside of the support, optical path = 0
                            path[idz, idy, idx] = 0

    if direction == "out":
        k_norm = k / np.linalg.norm(k)
        for idz in range(min_z, max_z, 1):
            for idy in range(min_y, max_y, 1):
                for idx in range(min_x, max_x, 1):
                    if support[idz, idy, idx] == 1:
                        stop_flag = False
                        counter = 1
                        pixel = np.array([idz, idy, idx])  # pixel for which the optical path is calculated
                        while not stop_flag:
                            pixel = pixel + k_norm  # add unitary translation in k_out direction
                            coords = np.rint(pixel)
                            stop_flag = True
                            if (min_z <= coords[0] <= max_z) and (min_y <= coords[1] <= max_y) and \
                                    (min_x <= coords[2] <= max_x):
                                counter = counter + support[int(coords[0]), int(coords[1]), int(coords[2])]
                                stop_flag = False
                        path[idz, idy, idx] = counter
                    else:  # point outside of the support, optical path = 0
                        path[idz, idy, idx] = 0

    if debugging:
        gu.multislices_plot(path, width_z=width_z, width_y=width_y, width_x=width_x,
                            invert_yaxis=True, title='Optical path ' + direction)
    return path


def get_strain(phase, planar_distance, voxel_size, reference_axis='y'):
    """
    Calculate the 3D strain array.

    :param phase: 3D phase array (do not forget the -1 sign if the phasing algorithm is python or matlab-based)
    :param planar_distance: the planar distance of the material corresponding to the measured Bragg peak
    :param voxel_size: the voxel size of the phase array in nm, should be isotropic
    :param reference_axis: the axis of the array along which q is aligned: 'x', 'y' or 'z' (CXI convention)
    :return: the strain 3D array
    """
    if phase.ndim != 3:
        raise ValueError('phase should be a 3D array')

    if reference_axis == "x":
        _, _, strain = np.gradient(planar_distance / (2 * np.pi) * phase,
                                   voxel_size)  # q is along x after rotating the crystal
    elif reference_axis == "y":
        _, strain, _ = np.gradient(planar_distance / (2 * np.pi) * phase,
                                   voxel_size)  # q is along y after rotating the crystal
    elif reference_axis == "z":
        strain, _, _ = np.gradient(planar_distance / (2 * np.pi) * phase,
                                   voxel_size)  # q is along y after rotating the crystal
    else:  # default is ref_axis_outplane = "y"
        raise ValueError("Wrong value for the reference axis, it should be 'x', 'y' or 'z'")
    return strain


def load_reconstruction(file_path):
    """
    Load the BCDI reconstruction.

    :param file_path: the path of the reconstruction to load. Format supported: .npy .npz .cxi .h5
    :return: the complex object and the extension of the file
    """
    _, extension = os.path.splitext(file_path)
    if extension == '.npz':
        npzfile = np.load(file_path)
        dataset = npzfile[list(npzfile.files)[0]]
    elif extension == '.npy':
        dataset = np.load(file_path)
    elif extension == '.cxi':
        h5file = h5py.File(file_path, 'r')
        group_key = list(h5file.keys())[1]
        subgroup_key = list(h5file[group_key])
        dataset = h5file['/'+group_key+'/'+subgroup_key[0]+'/data'].value
    elif extension == '.h5':  # modes.h5
        h5file = h5py.File(file_path, 'r')
        group_key = list(h5file.keys())[0]
        subgroup_key = list(h5file[group_key])
        dataset = h5file['/' + group_key + '/' + subgroup_key[0] + '/data'].value[0]
    else:
        raise ValueError("File format not supported: can load only '.npy', '.npz', '.cxi' or '.h5' files")
    return dataset, extension


def mean_filter(phase, support, half_width=0, width_z=np.nan, width_y=np.nan, width_x=np.nan,
                phase_range=np.pi, debugging=False):
    """
    Apply a mean filter to the phase (spatial average), taking care of the surface.

    :param phase: phase to be averaged
    :param support: support used for averaging
    :param half_width: half_width of the 2D square averaging window, 0 means no averaging, 1 is one pixel away...
    :param width_z: size of the area to plot in z (axis 0), centered on the middle of the initial array
    :param width_y: size of the area to plot in y (axis 1), centered on the middle of the initial array
    :param width_x: size of the area to plot in x (axis 2), centered on the middle of the initial array
    :param phase_range: range for plotting the phase, [-pi pi] by default
    :param debugging: set to True to see plots
    :type debugging: bool
    :return: averaged phase
    """
    if half_width != 0:
        if debugging:
            gu.multislices_plot(phase, width_z=width_z, width_y=width_y, width_x=width_x,
                                vmin=-phase_range, vmax=phase_range, invert_yaxis=True,
                                title='Phase before averaging', plot_colorbar=True)
            gu.multislices_plot(support, width_z=width_z, width_y=width_y, width_x=width_x,
                                vmin=0, vmax=1, invert_yaxis=True, title='Support for averaging')

        nonzero_pixels = np.argwhere(support != 0)
        new_values = np.zeros((nonzero_pixels.shape[0], 1), dtype=phase.dtype)
        counter = 0
        for indx in range(nonzero_pixels.shape[0]):
            piz = nonzero_pixels[indx, 0]
            piy = nonzero_pixels[indx, 1]
            pix = nonzero_pixels[indx, 2]
            tempo_support = support[piz-half_width:piz+half_width+1, piy-half_width:piy+half_width+1,
                                    pix-half_width:pix+half_width+1]
            nb_points = tempo_support.sum()
            temp_phase = phase[piz-half_width:piz+half_width+1, piy-half_width:piy+half_width+1,
                               pix-half_width:pix+half_width+1]
            if temp_phase.size != 0:
                value = temp_phase[np.nonzero(tempo_support)].sum()/nb_points
                new_values[indx] = value
            else:
                counter = counter + 1
        for indx in range(nonzero_pixels.shape[0]):
            phase[nonzero_pixels[indx, 0], nonzero_pixels[indx, 1], nonzero_pixels[indx, 2]] = new_values[indx]
        if debugging:
            gu.multislices_plot(phase, width_z=width_z, width_y=width_y, width_x=width_x,
                                vmin=-phase_range, vmax=phase_range, invert_yaxis=True,
                                title='Phase after averaging', plot_colorbar=True)
        if counter != 0:
            print("There were", counter, "voxels for which phase could not be averaged")
    return phase


def plane_angle(ref_plane, plane):
    """
    Calculate the angle between two crystallographic planes in cubic materials.

    :param ref_plane: measured reflection
    :param plane: plane for which angle should be calculated
    :return: the angle in degrees
    """
    if np.array_equal(ref_plane, plane):
        my_angle = 0.0
    else:
        my_angle = 180/np.pi*np.arccos(sum(np.multiply(ref_plane, plane)) /
                                       (np.linalg.norm(ref_plane)*np.linalg.norm(plane)))
    return my_angle


def regrid(array, voxel_zyx, voxel):
    """
    Interpolate real space data on a grid with cubic voxels.

    :param array: 3D array, the object to be interpolated
    :param voxel_zyx: tuple of actual voxel sizes in z, y, and x (CXI convention)
    :param voxel: desired voxel size for the interpolation
    :return: obj interpolated onto a grid with cubic voxels
    """
    from scipy.interpolate import RegularGridInterpolator

    if array.ndim != 3:
        raise ValueError('array should be a 3D array')

    nbz, nby, nbx = array.shape
    dz_realspace, dy_realspace, dx_realspace = voxel_zyx
    old_z = np.arange(-nbz // 2, nbz // 2, 1) * dz_realspace
    old_y = np.arange(-nby // 2, nby // 2, 1) * dy_realspace
    old_x = np.arange(-nbx // 2, nbx // 2, 1) * dx_realspace

    new_z, new_y, new_x = np.meshgrid(old_z * voxel / dz_realspace,
                                      old_y * voxel / dy_realspace,
                                      old_x * voxel / dx_realspace,
                                      indexing='ij')

    rgi = RegularGridInterpolator((old_z, old_y, old_x), array, method='linear', bounds_error=False, fill_value=0)

    new_array = rgi(np.concatenate((new_z.reshape((1, new_z.size)), new_y.reshape((1, new_y.size)),
                                   new_x.reshape((1, new_x.size)))).transpose())
    new_array = new_array.reshape((nbz, nby, nbx)).astype(array.dtype)
    return new_array


def remove_ramp(amp, phase, initial_shape, width_z=np.nan, width_y=np.nan, width_x=np.nan,
                amplitude_threshold=0.25, gradient_threshold=0.2, method='gradient', ups_factor=2, debugging=False):
    """
    Remove the linear trend in the ramp using its gradient and a threshold n 3D dataset.

    :param amp: 3D array, amplitude of the object
    :param phase: 3D array, phase of the object to be detrended
    :param initial_shape: shape of the FFT used for phasing
    :param width_z: size of the area to plot in z (axis 0), centered on the middle of the initial array
    :param width_y: size of the area to plot in y (axis 1), centered on the middle of the initial array
    :param width_x: size of the area to plot in x (axis 2), centered on the middle of the initial array
    :param amplitude_threshold: threshold used to define the support of the object from the amplitude
    :param gradient_threshold: higher threshold used to select valid voxels in the gradient array
    :param method: 'gradient' or 'upsampling'
    :param ups_factor: upsampling factor (the original shape will be multiplied by this value)
    :param debugging: set to True to see plots
    :type debugging: bool
    :return: normalized amplitude, detrended phase, ramp along z, ramp along y, ramp along x
    """
    if amp.ndim != 3 or phase.ndim != 3:
        raise ValueError('amp and phase should be 3D arrays')
    if amp.shape != phase.shape:
        raise ValueError('amp and phase must have the same shape\n'
                         'amp is ', amp.shape, ' while phase is ', phase.shape)

    if method == 'upsampling':
        nbz, nby, nbx = [mysize*ups_factor for mysize in initial_shape]
        nb_z, nb_y, nb_x = amp.shape
        myobj = crop_pad(amp * np.exp(1j * phase), (nbz, nby, nbx))
        if debugging:
            plt.figure()
            plt.imshow(np.log10(abs(myobj).sum(axis=0)))
            plt.title('np.log10(abs(myobj).sum(axis=0))')
            plt.pause(0.1)
        my_fft = fftshift(fftn(ifftshift(myobj)))
        del myobj, amp, phase
        gc.collect()
        if debugging:
            plt.figure()
            # plt.imshow(np.log10(abs(my_fft[nbz//2, :, :])))
            plt.imshow(np.log10(abs(my_fft).sum(axis=0)))
            plt.title('np.log10(abs(my_fft).sum(axis=0))')
            plt.pause(0.1)
        zcom, ycom, xcom = center_of_mass(abs(my_fft)**4)
        print('FFT shape for subpixel shift:', nbz, nby, nbx)
        print('COM before subpixel shift', zcom, ',', ycom, ',', xcom)
        shiftz = zcom - (nbz / 2)
        shifty = ycom - (nby / 2)
        shiftx = xcom - (nbx / 2)

        # phase shift in real space
        buf2ft = fftn(my_fft)  # in real space
        del my_fft
        gc.collect()
        if debugging:
            plt.figure()
            plt.imshow(abs(buf2ft).sum(axis=0))
            plt.title('abs(buf2ft).sum(axis=0)')
            plt.pause(0.1)

        z_axis = ifftshift(np.arange(-np.fix(nbz/2), np.ceil(nbz/2), 1))
        y_axis = ifftshift(np.arange(-np.fix(nby/2), np.ceil(nby/2), 1))
        x_axis = ifftshift(np.arange(-np.fix(nbx/2), np.ceil(nbx/2), 1))
        z_axis, y_axis, x_axis = np.meshgrid(z_axis, y_axis, x_axis, indexing='ij')
        greg = buf2ft * np.exp(1j * 2 * np.pi * (shiftz * z_axis / nbz + shifty * y_axis / nby + shiftx * x_axis / nbx))
        del buf2ft, z_axis, y_axis, x_axis
        gc.collect()
        if debugging:
            plt.figure()
            plt.imshow(abs(greg).sum(axis=0))
            plt.title('abs(greg).sum(axis=0)')
            plt.pause(0.1)

        my_fft = ifftn(greg)
        del greg
        gc.collect()
        # end of phase shift in real space

        if debugging:
            plt.figure()
            plt.imshow(np.log10(abs(my_fft).sum(axis=0)))
            plt.title('centered np.log10(abs(my_fft).sum(axis=0))')
            plt.pause(0.1)

        print('COM after subpixel shift', center_of_mass(abs(my_fft) ** 4))
        myobj = fftshift(ifftn(ifftshift(my_fft)))
        del my_fft
        gc.collect()
        if debugging:
            plt.figure()
            plt.imshow(abs(myobj).sum(axis=0))
            plt.title('centered abs(myobj).sum(axis=0)')
            plt.pause(0.1)

        myobj = crop_pad(myobj, (nb_z, nb_y, nb_x))  # return to the initial shape of myamp
        print('Upsampling: shift_z, shift_y, shift_x: (', str('{:.3f}'.format(shiftz)),
              str('{:.3f}'.format(shifty)), str('{:.3f}'.format(shiftx)), ') pixels')
        return abs(myobj)/abs(myobj).max(), np.angle(myobj)

    else:  # method='gradient'

        # define the support from the amplitude
        nbz, nby, nbx = amp.shape
        mysupport = np.zeros((nbz, nby, nbx))
        mysupport[amp > amplitude_threshold*abs(amp).max()] = 1

        # axis 0 (Z)
        mygradz, _, _ = np.gradient(phase, 1)

        mysupportz = np.zeros((nbz, nby, nbx))
        mysupportz[abs(mygradz) < gradient_threshold] = 1
        mysupportz = mysupportz * mysupport
        myrampz = mygradz[mysupportz == 1].mean()
        if debugging:
            gu.multislices_plot(mygradz, width_z=width_z, width_y=width_y, width_x=width_x,
                                invert_yaxis=True, vmin=-0.2, vmax=0.2, title='Phase gradient along Z')
            gu.multislices_plot(mysupportz, width_z=width_z, width_y=width_y, width_x=width_x,
                                invert_yaxis=True, vmin=0, vmax=1, title='Thresholded support along Z')
        del mysupportz, mygradz
        gc.collect()

        # axis 1 (Y)
        _, mygrady, _ = np.gradient(phase, 1)
        mysupporty = np.zeros((nbz, nby, nbx))
        mysupporty[abs(mygrady) < gradient_threshold] = 1
        mysupporty = mysupporty * mysupport
        myrampy = mygrady[mysupporty == 1].mean()
        if debugging:
            gu.multislices_plot(mygrady, width_z=width_z, width_y=width_y, width_x=width_x,
                                invert_yaxis=True, vmin=-0.2, vmax=0.2, title='Phase gradient along Y')
            gu.multislices_plot(mysupporty, width_z=width_z, width_y=width_y, width_x=width_x,
                                invert_yaxis=True, vmin=0, vmax=1, title='Thresholded support along Y')
        del mysupporty, mygrady
        gc.collect()

        # axis 2 (X)
        _, _, mygradx = np.gradient(phase, 1)
        mysupportx = np.zeros((nbz, nby, nbx))
        mysupportx[abs(mygradx) < gradient_threshold] = 1
        mysupportx = mysupportx * mysupport
        myrampx = mygradx[mysupportx == 1].mean()
        if debugging:
            gu.multislices_plot(mygradx, width_z=width_z, width_y=width_y, width_x=width_x,
                                invert_yaxis=True, vmin=-0.2, vmax=0.2, title='Phase gradient along X')
            gu.multislices_plot(mysupportx, width_z=width_z, width_y=width_y, width_x=width_x,
                                invert_yaxis=True, vmin=0, vmax=1, title='Thresholded support along X')
        del mysupportx, mygradx, mysupport
        gc.collect()

        myz, myy, myx = np.meshgrid(np.arange(0, nbz, 1), np.arange(0, nby, 1), np.arange(0, nbx, 1),
                                    indexing='ij')

        print('Gradient: Phase_ramp_z, Phase_ramp_y, Phase_ramp_x: (', str('{:.3f}'.format(myrampz)),
              str('{:.3f}'.format(myrampy)), str('{:.3f}'.format(myrampx)), ') rad')
        phase = phase - myz * myrampz - myy * myrampy - myx * myrampx
        return amp, phase, myrampz, myrampy, myrampx


def remove_ramp_2d(amp, phase, initial_shape, width_y=np.nan, width_x=np.nan, amplitude_threshold=0.25,
                   gradient_threshold=0.2, method='gradient', ups_factor=2, debugging=False):
    """
    Remove the linear trend in the ramp using its gradient and a threshold in 2D dataset.

    :param amp: 2D array, amplitude of the object
    :param phase: 2D array, phase of the object to be detrended
    :param initial_shape: shape of the FFT used for phasing
    :param width_y: size of the area to plot in y (vertical axis), centered on the middle of the initial array
    :param width_x: size of the area to plot in x (horizontal axis), centered on the middle of the initial array
    :param amplitude_threshold: threshold used to define the support of the object from the amplitude
    :param gradient_threshold: higher threshold used to select valid voxels in the gradient array
    :param method: 'gradient' or 'upsampling'
    :param ups_factor: upsampling factor (the original shape will be multiplied by this value)
    :param debugging: set to True to see plots
    :type debugging: bool
    :return: normalized amplitude, detrended phase, ramp along y, ramp along x
    """
    if amp.ndim != 2 or phase.ndim != 2:
        raise ValueError('amp and phase should be 2D arrays')
    if amp.shape != phase.shape:
        raise ValueError('amp and phase must have the same shape\n'
                         'amp is ', amp.shape, ' while phase is ', phase.shape)

    if method == 'upsampling':
        nby, nbx = [mysize * ups_factor for mysize in initial_shape]
        nb_y, nb_x = amp.shape
        myobj = crop_pad(amp * np.exp(1j * phase), (nby, nbx))
        if debugging:
            plt.figure()
            plt.imshow(np.log10(abs(myobj)))
            plt.title('np.log10(abs(myobj))')
            plt.pause(0.1)
        my_fft = fftshift(fftn(ifftshift(myobj)))
        del myobj, amp, phase
        gc.collect()
        if debugging:
            plt.figure()
            # plt.imshow(np.log10(abs(my_fft[nbz//2, :, :])))
            plt.imshow(np.log10(abs(my_fft)))
            plt.title('np.log10(abs(my_fft))')
            plt.pause(0.1)
        ycom, xcom = center_of_mass(abs(my_fft) ** 4)
        print('FFT shape for subpixel shift:', nby, nbx)
        print('COM before subpixel shift', ycom, ',', xcom)
        shifty = ycom - (nby / 2)
        shiftx = xcom - (nbx / 2)

        # phase shift in real space
        buf2ft = fftn(my_fft)  # in real space
        del my_fft
        gc.collect()
        if debugging:
            plt.figure()
            plt.imshow(abs(buf2ft))
            plt.title('abs(buf2ft)')
            plt.pause(0.1)

        y_axis = ifftshift(np.arange(-np.fix(nby / 2), np.ceil(nby / 2), 1))
        x_axis = ifftshift(np.arange(-np.fix(nbx / 2), np.ceil(nbx / 2), 1))
        y_axis, x_axis = np.meshgrid(y_axis, x_axis, indexing='ij')
        greg = buf2ft * np.exp(1j * 2 * np.pi * (shifty * y_axis / nby + shiftx * x_axis / nbx))
        del buf2ft, y_axis, x_axis
        gc.collect()
        if debugging:
            plt.figure()
            plt.imshow(abs(greg))
            plt.title('abs(greg)')
            plt.pause(0.1)

        my_fft = ifftn(greg)
        del greg
        gc.collect()
        # end of phase shift in real space

        if debugging:
            plt.figure()
            plt.imshow(np.log10(abs(my_fft)))
            plt.title('centered np.log10(abs(my_fft))')
            plt.pause(0.1)

        print('COM after subpixel shift', center_of_mass(abs(my_fft) ** 4))
        myobj = fftshift(ifftn(ifftshift(my_fft)))
        del my_fft
        gc.collect()
        if debugging:
            plt.figure()
            plt.imshow(abs(myobj))
            plt.title('centered abs(myobj)')
            plt.pause(0.1)

        myobj = crop_pad_2d(myobj, (nb_y, nb_x))  # return to the initial shape of myamp
        print('Upsampling: shift_y, shift_x: (', str('{:.3f}'.format(shifty)), str('{:.3f}'.format(shiftx)), ') pixels')
        return abs(myobj) / abs(myobj).max(), np.angle(myobj)

    else:  # method='gradient'

        # define the support from the amplitude
        nby, nbx = amp.shape
        mysupport = np.zeros((nby, nbx))
        mysupport[amp > amplitude_threshold * abs(amp).max()] = 1

        # axis 0 (Y)
        mygrady, _ = np.gradient(phase, 1)
        mysupporty = np.zeros((nby, nbx))
        mysupporty[abs(mygrady) < gradient_threshold] = 1
        mysupporty = mysupporty * mysupport
        myrampy = mygrady[mysupporty == 1].mean()
        if debugging:
            gu.imshow_plot(array=mygrady, width_v=width_y, width_h=width_x, vmin=-0.2, vmax=0.2,
                           title='Phase gradient along Y')
            gu.imshow_plot(array=mysupporty, width_v=width_y, width_h=width_x, vmin=0, vmax=1,
                           title='Thresholded support along Y')
        del mysupporty, mygrady
        gc.collect()

        # axis 1 (X)
        _, mygradx = np.gradient(phase, 1)
        mysupportx = np.zeros((nby, nbx))
        mysupportx[abs(mygradx) < gradient_threshold] = 1
        mysupportx = mysupportx * mysupport
        myrampx = mygradx[mysupportx == 1].mean()
        if debugging:
            gu.imshow_plot(array=mygradx, width_v=width_y, width_h=width_x, vmin=-0.2, vmax=0.2,
                           title='Phase gradient along X')
            gu.imshow_plot(array=mysupportx, width_v=width_y, width_h=width_x, vmin=0, vmax=1,
                           title='Thresholded support along X')
        del mysupportx, mygradx, mysupport
        gc.collect()

        myy, myx = np.meshgrid(np.arange(0, nby, 1), np.arange(0, nbx, 1), indexing='ij')

        print('Gradient: Phase_ramp_z, Phase_ramp_y, Phase_ramp_x: (', str('{:.3f}'.format(myrampy)),
              str('{:.3f}'.format(myrampx)), ') rad')
        phase = phase - myy * myrampy - myx * myrampx
        return amp, phase, myrampy, myrampx


def rotate_crystal(array, axis_to_align, reference_axis, width_z=np.nan, width_y=np.nan, width_x=np.nan,
                   debugging=False):
    """
    Rotate myobj to align axis_to_align onto reference_axis.
    axis_to_align and reference_axis should be in the order X Y Z, where Z is downstream, Y vertical and X outboard
    (CXI convention).

    :param array: 3D real array to be rotated
    :param axis_to_align: the axis of myobj (vector q) x y z
    :param reference_axis: will align axis_to_align onto this  x y z
    :param width_z: size of the area to plot in z (axis 0), centered on the middle of the initial array
    :param width_y: size of the area to plot in y (axis 1), centered on the middle of the initial array
    :param width_x: size of the area to plot in x (axis 2), centered on the middle of the initial array
    :param debugging: set to True to see plots before and after rotation
    :type debugging: bool
    :return: rotated myobj
    """
    if array.ndim != 3:
        raise ValueError('array should be 3D arrays')

    nbz, nby, nbx = array.shape
    if debugging:
        gu.multislices_plot(array, width_z=width_z, width_y=width_y, width_x=width_x,
                            invert_yaxis=True, title='Before rotating')

    v = np.cross(axis_to_align, reference_axis)
    skew_sym_matrix = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    my_rotation_matrix = np.identity(3) +\
        skew_sym_matrix + np.dot(skew_sym_matrix, skew_sym_matrix) / (1+np.dot(axis_to_align, reference_axis))
    transfer_matrix = my_rotation_matrix.transpose()
    old_z = np.arange(-nbz // 2, nbz // 2, 1)
    old_y = np.arange(-nby // 2, nby // 2, 1)
    old_x = np.arange(-nbx // 2, nbx // 2, 1)

    myz, myy, myx = np.meshgrid(old_z, old_y, old_x, indexing='ij')

    new_x = transfer_matrix[0, 0] * myx + transfer_matrix[0, 1] * myy + transfer_matrix[0, 2] * myz
    new_y = transfer_matrix[1, 0] * myx + transfer_matrix[1, 1] * myy + transfer_matrix[1, 2] * myz
    new_z = transfer_matrix[2, 0] * myx + transfer_matrix[2, 1] * myy + transfer_matrix[2, 2] * myz

    del myx, myy, myz
    rgi = RegularGridInterpolator((old_z, old_y, old_x), array, method='linear', bounds_error=False, fill_value=0)
    new_array = rgi(np.concatenate((new_z.reshape((1, new_z.size)), new_y.reshape((1, new_z.size)),
                                   new_x.reshape((1, new_z.size)))).transpose())
    new_array = new_array.reshape((nbz, nby, nbx)).astype(array.dtype)
    if debugging:
        gu.multislices_plot(new_array, width_z=width_z, width_y=width_y, width_x=width_x,
                            invert_yaxis=True, title='After rotating')
    return new_array


def rotate_vector(vector, axis_to_align, reference_axis):
    """
    Calculate vector components in the basis where axis_to_align and reference_axis are aligned.
    axis_to_align and reference_axis should be in the order X Y Z, where Z is downstream, Y vertical and X outboard
    (CXI convention).

    :param vector: the vector to be rotated  x y z
    :param axis_to_align: the axis of myobj (vector q) x y z
    :param reference_axis: will align axis_to_align onto this  x y z
    :return: rotated vector in CXI convention z y x
    """
    if vector.ndim != 1:
        raise ValueError('vector should be a 1D array')
    else:
        if len(vector) != 3:
            raise ValueError('vector should have 3 elements')

    v = np.cross(axis_to_align, reference_axis)
    skew_sym_matrix = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    my_rotation_matrix = np.identity(3) +\
        skew_sym_matrix + np.dot(skew_sym_matrix, skew_sym_matrix) / (1+np.dot(axis_to_align, reference_axis))
    transfer_matrix = my_rotation_matrix.transpose()

    new_x = transfer_matrix[0, 0] * vector[0] + transfer_matrix[0, 1] * vector[1] + transfer_matrix[0, 2] * vector[2]
    new_y = transfer_matrix[1, 0] * vector[0] + transfer_matrix[1, 1] * vector[1] + transfer_matrix[1, 2] * vector[2]
    new_z = transfer_matrix[2, 0] * vector[0] + transfer_matrix[2, 1] * vector[1] + transfer_matrix[2, 2] * vector[2]

    return np.array([new_z, new_y, new_x])


def sort_reconstruction(file_path, data_range, amplitude_threshold, sort_method='variance/mean'):
    """
    Sort out reconstructions based on the metric 'sort_method'.

    :param file_path: path of the reconstructions to sort out
    :param data_range: data will be cropped or padded to this range
    :param amplitude_threshold: threshold used to define a support from the amplitude
    :param sort_method: method for sorting the reconstructions: 'variance/mean', 'mean_amplitude', 'variance' or
     'volume'
    :return: a list of sorted indices in 'file_path', from the best object to the worst.
    """

    nbfiles = len(file_path)
    zrange, yrange, xrange = data_range

    quality_array = np.ones((nbfiles, 4))  # 1/mean_amp, variance(amp), variance(amp)/mean_amp, 1/volume
    for ii in range(nbfiles):
        obj, _ = load_reconstruction(file_path[ii])
        print('Opening ', file_path[ii])

        # use the range of interest defined above
        obj = crop_pad(obj, [2 * zrange, 2 * yrange, 2 * xrange], debugging=False)
        obj = abs(obj) / abs(obj).max()

        temp_support = np.zeros(obj.shape)
        temp_support[obj > amplitude_threshold] = 1  # only for plotting
        quality_array[ii, 0] = 1 / obj[obj > amplitude_threshold].mean()     # 1/mean(amp)
        quality_array[ii, 1] = np.var(obj[obj > amplitude_threshold])        # var(amp)
        quality_array[ii, 2] = quality_array[ii, 0] * quality_array[ii, 1]   # var(amp)/mean(amp) index of dispersion
        quality_array[ii, 3] = 1 / temp_support.sum()                        # 1/volume(support)
        del temp_support
        gc.collect()

        # order reconstructions by minimizing the quality factor
    if sort_method is 'mean_amplitude':    # sort by quality_array[:, 0] first
        sorted_obj = np.lexsort((quality_array[:, 3], quality_array[:, 2], quality_array[:, 1], quality_array[:, 0]))

    elif sort_method is 'variance':        # sort by quality_array[:, 1] first
        sorted_obj = np.lexsort((quality_array[:, 0], quality_array[:, 3], quality_array[:, 2], quality_array[:, 1]))

    elif sort_method is 'variance/mean':   # sort by quality_array[:, 2] first
        sorted_obj = np.lexsort((quality_array[:, 1], quality_array[:, 0], quality_array[:, 3], quality_array[:, 2]))

    elif sort_method is 'volume':          # sort by quality_array[:, 3] first
        sorted_obj = np.lexsort((quality_array[:, 2], quality_array[:, 1], quality_array[:, 0], quality_array[:, 3]))

    else:  # default case, use the index of dispersion
        sorted_obj = np.lexsort((quality_array[:, 1], quality_array[:, 0], quality_array[:, 3], quality_array[:, 2]))

    print('quality_array')
    print(quality_array)
    print("sorted list", sorted_obj)

    return sorted_obj


def tukey_window(shape, alpha=np.array([0.5, 0.5, 0.5])):
    """
    Create a 3d Tukey window based on shape and the shape parameter alpha.

    :param shape: tuple, shape of the 3d window
    :param alpha: shape parameter of the Tukey window, tuple or ndarray of 3 values
    :return: the 3d Tukey window
    """
    from scipy.signal.windows import tukey
    nbz, nby, nbx = shape
    array_z = tukey(nbz, alpha[0])
    array_y = tukey(nby, alpha[1])
    array_x = tukey(nbx, alpha[2])
    tukey2 = np.ones((nbz, nby))
    tukey3 = np.ones((nbz, nby, nbx))
    for idz in range(nbz):
        tukey2[idz, :] = array_z[idz] * array_y
        for idy in range(nby):
            tukey3[idz, idy] = tukey2[idz, idy] * array_x
    return tukey3


def wrap(phase):
    """
    Wrap the phase in [-pi pi] interval.

    :param phase: phase to wrap
    :return: phase wrapped in [-pi pi]
    """
    phase = (phase + np.pi) % (2 * np.pi) - np.pi
    return phase


# if __name__ == "__main__":
# #     # datadir = 'C:/Users/carnis/Work Folders/Documents/data/CH4760_Pt/S2191/pynxraw/'
# #     # data = np.load(datadir + 'S2191_pynx_270_432_400.npz')['data']
# #     # newdata = bin_data(data, (2, 1, 2), True)
# #
#     nbz, nby, nbx = (200, 200, 200)
#     w = tukey_window((nbz, nby, nbx), (0.2, 0.2, 0.2))
#     plt.figure()
#     plt.subplot(1, 3, 1)
#     plt.imshow(w[nbz//2, :, :])
#     plt.title('middle z')
#     plt.colorbar()
#     plt.subplot(1, 3, 2)
#     plt.imshow(w[0, :, :])
#     plt.title('first z')
#     plt.colorbar()
#     plt.subplot(1, 3, 3)
#     plt.imshow(w[1, :, :])
#     plt.title('second z')
#     plt.colorbar()
#     plt.figure()
#     plt.subplot(1, 3, 1)
#     plt.imshow(w[:, nby//2, :])
#     plt.colorbar()
#     plt.title('middle y')
#     plt.subplot(1, 3, 2)
#     plt.imshow(w[:, 0, :])
#     plt.title('first y')
#     plt.colorbar()
#     plt.subplot(1, 3, 3)
#     plt.imshow(w[:, 1, :])
#     plt.title('second y')
#     plt.colorbar()
#     plt.figure()
#     plt.subplot(1, 3, 1)
#     plt.imshow(w[:, :, nbx//2])
#     plt.title('middle x')
#     plt.colorbar()
#     plt.subplot(1, 3, 2)
#     plt.imshow(w[:, :, 0])
#     plt.title('first x')
#     plt.colorbar()
#     plt.subplot(1, 3, 3)
#     plt.imshow(w[:, :, 1])
#     plt.title('second x')
#     plt.colorbar()
#     plt.show()
