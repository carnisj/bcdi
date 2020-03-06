# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import os
import h5py
import numpy as np
from scipy.interpolate import interp1d





def find_nearest(original_array, array_values):
    """
    Find the indices where original_array is nearest to array_values.

    :param original_array: a 1D array where to look for the nearest values
    :param array_values: a number or a 1D array of numbers
    :return: index or indices from original_array nearest to values, of length len(array_values)
    """
    original_array, array_values = np.asarray(original_array), np.asarray(array_values)

    if original_array.ndim != 1:
        raise ValueError('original_array should be 1D')
    if array_values.ndim > 1:
        raise ValueError('array_values should be a number or a 1D array')
    if array_values.ndim == 0:
        nearest_index = (np.abs(original_array - array_values)).argmin()
    else:
        nb_values = len(array_values)
        nearest_index = np.zeros(nb_values, dtype=int)
        for idx in range(nb_values):
            nearest_index[idx] = (np.abs(original_array - array_values[idx])).argmin()

    return nearest_index


def gaussian(x_axis, amp, cen, sig):
    """
    Gaussian line shape.

    :param x_axis: where to calculate the function
    :param amp: the amplitude of the Gaussian
    :param cen: the position of the center
    :param sig: HWHM of the Gaussian
    :return: the Gaussian line shape at x_axis
    """
    return amp*np.exp(-(x_axis-cen)**2/(2.*sig**2))


def function_lmfit(params, iterator, x_axis, distribution):
    """
    Calculate distribution using by lmfit Parameters.

    :param params: a lmfit Parameters object
    :param iterator: the index of the relevant parameters
    :param x_axis: where to calculate the function
    :param distribution: the distribution to use
    :return: the gaussian function calculated at x_axis positions
    """
    if distribution == 'gaussian':
        amp = params['amp_%i' % (iterator+1)].value
        cen = params['cen_%i' % (iterator+1)].value
        sig = params['sig_%i' % (iterator+1)].value
        return gaussian(x_axis=x_axis, amp=amp, cen=cen, sig=sig)
    elif distribution == 'lorentzian':
        amp = params['amp_%i' % (iterator+1)].value
        cen = params['cen_%i' % (iterator+1)].value
        sig = params['sig_%i' % (iterator+1)].value
        return lorentzian(x_axis=x_axis, amp=amp, cen=cen, sig=sig)
    elif distribution == 'pseudovoigt':
        amp = params['amp_%i' % (iterator+1)].value
        cen = params['cen_%i' % (iterator+1)].value
        sig = params['sig_%i' % (iterator+1)].value
        ratio = params['ratio_%i' % (iterator+1)].value
        return pseudovoigt(x_axis, amp=amp, cen=cen, sig=sig, ratio=ratio)
    else:
        raise ValueError(distribution + ' not implemented')


def load_file(file_path, fieldname=None):
    """
    Load a file. In case of .cxi or .h5 file, it will use a default path to the data.
    'fieldname' is used only for .npz files.

    :param file_path: the path of the reconstruction to load. Format supported: .npy .npz .cxi .h5
    :param fieldname: the name of the field to be loaded
    :return: the loaded data and the extension of the file
    """
    _, extension = os.path.splitext(file_path)
    if extension == '.npz':  # could be anything
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
    elif extension == '.npy':  # could be anything
        dataset = np.load(file_path)
    elif extension == '.cxi':  # output of PyNX phasing
        h5file = h5py.File(file_path, 'r')
        group_key = list(h5file.keys())[1]
        subgroup_key = list(h5file[group_key])
        dataset = h5file['/'+group_key+'/'+subgroup_key[0]+'/data'].value
    elif extension == '.h5':  # modes.h5
        h5file = h5py.File(file_path, 'r')
        group_key = list(h5file.keys())[0]
        subgroup_key = list(h5file[group_key])
        dataset = h5file['/' + group_key + '/' + subgroup_key[0] + '/data'][0]  # select only first mode
    else:
        raise ValueError("File format not supported: can load only '.npy', '.npz', '.cxi' or '.h5' files")

    if fieldname == 'modulus':
        dataset = abs(dataset)
    elif fieldname == 'angle':
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
    return amp/(sig*np.pi)/(1+(x_axis-cen)**2/(sig**2))


def objective_lmfit(params, x_axis, data, distribution):
    """
    Calculate total residual for fits to several data sets held
    in a 2-D array, and modeled by gaussian functions.

    :param params: a lmfit Parameters object
    :param x_axis: where to calculate the gaussian distribution
    :param data: data to fit
    :param distribution: distribution to use for fitting
    :return: the residuals of the fit of data using the parameters
    """
    ndata, nx = data.shape
    resid = 0.0*data[:]
    # make residual per data set
    for idx in range(ndata):
        resid[idx, :] = data[idx, :] - function_lmfit(params=params, iterator=idx, x_axis=x_axis[idx, :],
                                                      distribution=distribution)
    # now flatten this to a 1D array, as minimize() needs
    return resid.flatten()


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
    sigma_gaussian = sig / (2*np.sqrt(2*np.log(2)))
    scaling_gaussian = 1 / (sigma_gaussian * np.sqrt(2*np.pi))  # the Gaussian is normalized
    sigma_lorentzian = sig / 2
    scaling_lorentzian = 1  # the Lorentzian is normalized
    return amp * (ratio * gaussian(x_axis, scaling_gaussian, cen, scaling_gaussian)
                  + (1-ratio) * lorentzian(x_axis, scaling_lorentzian, cen, sigma_lorentzian))


def remove_background(array, q_values, avg_background, avg_qvalues):
    """
    Subtract the averagae 1D background to the 3D array using q values.

    :param array: the 3D array. It should be sparse for faster calculation.
    :param q_values: tuple of three 1D arrays (qx, qz, qy), q values for the 3D dataset
    :param avg_background: average background data
    :param avg_qvalues: q values for the 1D average background data
    :return: the 3D background array
    """
    if array.ndim != 3:
        raise ValueError('data should be a 3D array')
    if (avg_background.ndim != 1) or (avg_qvalues.ndim != 1):
        raise ValueError('avg_background and distances should be 1D arrays')

    qx, qz, qy = q_values
    avg_background[np.isnan(avg_background)] = 0
    interpolation = interp1d(avg_qvalues, avg_background, kind='linear', bounds_error=False, fill_value=np.nan)

    ind_z, ind_y, ind_x = np.nonzero(array)  # if data is sparse, a loop over these indices only will be fast

    for index in range(len(ind_z)):
        array[ind_z[index], ind_y[index], ind_x[index]] =\
            array[ind_z[index], ind_y[index], ind_x[index]]\
            - interpolation(np.sqrt(qx[ind_z[index]] ** 2 + qz[ind_y[index]] ** 2 + qy[ind_x[index]] ** 2))

    array[np.isnan(array)] = 0
    array[array < 0] = 0

    return array
