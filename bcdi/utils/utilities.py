# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import os
import h5py
import numpy as np
from scipy.special import wofz


def find_nearest(original_array, array_values):
    """
    Find the indices where original_array is nearest to array_values.

    :param original_array: a 1D array where to look for the nearest values
    :param array_values: a 1D array of numbers
    :return: indices from original_array nearest to values, of length len(array_values)
    """
    original_array, array_values = np.asarray(original_array), np.asarray(array_values)

    if original_array.ndim != 1 or array_values.ndim != 1:
        raise ValueError('original_array and array_values are expected to be 1D arrays')
    nb_values = len(array_values)
    nearest_index = np.zeros(nb_values, dtype=int)
    for idx in range(nb_values):
        nearest_index[idx] = (np.abs(original_array - array_values[idx])).argmin()
    return nearest_index


def gaussian(x_axis, scaling, mu, sigma):
    """
    Gaussian line shape.

    :param x_axis: where to calculate the function
    :param scaling: the amplitude of the gaussian
    :param mu: the position of the center
    :param sigma: HWHM of the gaussian
    :return: the gaussian function
    """
    return scaling*np.exp(-(x_axis-mu)**2/(2.*sigma**2))


def function_lmfit(params, iterator, x_axis, distribution):
    """
    Calculate distribution using by lmfit Parameters.

    :param params: lmfit Parameters object
    :param iterator: the index of the relevant parameters
    :param x_axis: where to calculate the function
    :param distribution: the distribution to use
    :return: the gaussian function calculated at x_axis positions
    """
    if distribution == 'gaussian':
        scaling = params['amp_%i' % (iterator+1)].value
        mu = params['cen_%i' % (iterator+1)].value
        sigma = params['sig_%i' % (iterator+1)].value
        return gaussian(x_axis=x_axis, scaling=scaling, mu=mu, sigma=sigma)
    


def lorentzian(x_axis, scaling, mu, gamma):
    """
    Lorentzian line shape

    :param x_axis: where to calculate the function
    :param scaling: the amplitude of the gaussian
    :param mu: the position of the center
    :param gamma: HWHM of the lorentzian
    :return: the lorentzian function
    """
    return scaling/(gamma*np.pi)/((x_axis-mu)**2/(gamma**2))


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


def objective_lmfit(params, x_axis, data, distribution):
    """
    Calculate total residual for fits to several data sets held
    in a 2-D array, and modeled by gaussian functions.

    :param params: lmfit Parameters object
    :param x_axis: where to calculate the gaussian distribution
    :param data: data to fit
    :param distribution: distribution to use for fitting
    :return: the residuals of the fit of data using the parameters
    """
    ndata, nx = data.shape
    resid = 0.0*data[:]
    # make residual per data set
    for idx in range(ndata):
        if distribution == 'gaussian':
            resid[idx, :] = data[idx, :] - function_lmfit(params=params, iterator=idx, x_axis=x_axis[idx, :],
                                                          distribution=distribution)
    # now flatten this to a 1D array, as minimize() needs
    return resid.flatten()
