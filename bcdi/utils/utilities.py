# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import os
import h5py
import numpy as np
from scipy.special import erf
from scipy.interpolate import interp1d
import sys
sys.path.append('C:/Users/Jerome/Documents/myscripts/bcdi/')
sys.path.append('D:/myscripts/bcdi/')
import bcdi.graph.graph_utils as gu


def catch_error(exception):
    """
    Callback processing exception in asynchronous multiprocessing.

    :param exception: the arisen exception
    """
    print(exception)


def find_nearest(reference_array, test_values, width=None):
    """
    Find the indices where original_array is nearest to array_values.

    :param reference_array: a 1D array where to look for the nearest values
    :param test_values: a number or a 1D array of numbers to be tested
    :param width: if not None, it will look for the nearest element within the range [x-width/2, x+width/2[
    :return: index or indices from original_array nearest to values of length len(test_values). Returns the index -1
     if there is no nearest neighbour in the range defined by width.
    """
    original_array, test_values = np.asarray(reference_array), np.asarray(test_values)

    if original_array.ndim != 1:
        raise ValueError('original_array should be 1D')
    if test_values.ndim > 1:
        raise ValueError('array_values should be a number or a 1D array')
    if test_values.ndim == 0:
        nearest_index = (np.abs(original_array - test_values)).argmin()
        return nearest_index
    else:
        nb_values = len(test_values)
        nearest_index = np.zeros(nb_values, dtype=int)
        for idx in range(nb_values):
            nearest_index[idx] = (np.abs(original_array - test_values[idx])).argmin()
        if width is not None:
            for idx in range(nb_values):
                if (reference_array[nearest_index[idx]] >= test_values[idx] + width / 2)\
                        or (reference_array[nearest_index[idx]] < test_values[idx] - width / 2):
                    # no neighbour in the range defined by width
                    nearest_index[idx] = -1
    return nearest_index


def fit3d_poly1(x_axis, a, b, c, d):
    """
    Calculate the 1st order polynomial function on points in a 3D gridgiven the polynomial parameters.

    :param x_axis: (3xN) tuple or array of 3D coordinates
    :param a: offset
    :param b: 1st order parameter for the 1st coordinate
    :param c: 1st order parameter for the 2nd coordinate
    :param d: 1st order parameter for the 3rd coordinate
    :return: the 1st order polynomial calculated on x_axis
    """
    return a + b*x_axis[0] + c*x_axis[1] + d*x_axis[2]


def fit3d_poly2(x_axis, a, b, c, d, e, f, g):
    """
    Calculate the 2nd order polynomial function on points in a 3D gridgiven the polynomial parameters.

    :param x_axis: (3xN) tuple or array of 3D coordinates
    :param a: offset
    :param b: 1st order parameter for the 1st coordinate
    :param c: 1st order parameter for the 2nd coordinate
    :param d: 1st order parameter for the 3rd coordinate
    :param e: 2nd order parameter for the 1st coordinate
    :param f: 2nd order parameter for the 2nd coordinate
    :return: the 2nd order polynomial calculated on x_axis
    """
    return a + b*x_axis[0] + c*x_axis[1] + d*x_axis[2] + e*x_axis[0]**2 + f*x_axis[1]**2 + g*x_axis[2]**2


def fit3d_poly3(x_axis, a, b, c, d, e, f, g, h, i, j):
    """
    Calculate the 3rd order polynomial function on points in a 3D gridgiven the polynomial parameters.

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
    return a + b*x_axis[0] + c*x_axis[1] + d*x_axis[2] + e*x_axis[0]**2 + f*x_axis[1]**2 + g*x_axis[2]**2 +\
        h*x_axis[0]**3 + i*x_axis[1]**3 + j*x_axis[2]**3


def fit3d_poly4(x_axis, a, b, c, d, e, f, g, h, i, j, k, l, m):
    """
    Calculate the 4th order polynomial function on points in a 3D grid given the polynomial parameters.

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
    :param l: 4th order parameter for the 2nd coordinate
    :param m: 4th order parameter for the 3rd coordinate
    :return: the 4th order polynomial calculated on x_axis
    """
    return a + b*x_axis[0] + c*x_axis[1] + d*x_axis[2] + e*x_axis[0]**2 + f*x_axis[1]**2 + g*x_axis[2]**2 +\
        h*x_axis[0]**3 + i*x_axis[1]**3 + j*x_axis[2]**3 + k*x_axis[0]**4 + l*x_axis[1]**4 + m*x_axis[2]**4


def function_lmfit(params, x_axis, distribution, iterator=0):
    """
    Calculate distribution using by lmfit Parameters.

    :param params: a lmfit Parameters object
    :param x_axis: where to calculate the function
    :param distribution: the distribution to use
    :param iterator: the index of the relevant parameters
    :return: the gaussian function calculated at x_axis positions
    """
    if distribution == 'gaussian':
        amp = params['amp_%i' % (iterator+1)].value
        cen = params['cen_%i' % (iterator+1)].value
        sig = params['sig_%i' % (iterator+1)].value
        return gaussian(x_axis=x_axis, amp=amp, cen=cen, sig=sig)
    elif distribution == 'skewed_gaussian':
        amp = params['amp_%i' % (iterator+1)].value
        cen = params['cen_%i' % (iterator+1)].value
        sig = params['sig_%i' % (iterator+1)].value
        alpha = params['alpha_%i' % (iterator+1)].value
        return skewed_gaussian(x_axis=x_axis, amp=amp, cen=cen, sig=sig, alpha=alpha)
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
        # group_key = list(h5file.keys())[1]
        # subgroup_key = list(h5file[group_key])
        # dataset = h5file['/'+group_key+'/'+subgroup_key[0]+'/data'].value
        dataset = h5file['/entry_1/data_1/data'].value
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
    if len(data.shape) == 1:  # single dataset
        data = data[np.newaxis, :]
        x_axis = x_axis[np.newaxis, :]
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


def remove_background(array, q_values, avg_background, avg_qvalues, method='normalize'):
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
        raise ValueError('data should be a 3D array')
    if (avg_background.ndim != 1) or (avg_qvalues.ndim != 1):
        raise ValueError('avg_background and distances should be 1D arrays')

    qx, qz, qy = q_values

    ind_z, ind_y, ind_x = np.nonzero(array)  # if data is sparse, a loop over these indices only will be fast

    if method == 'subtract':
        avg_background[np.isnan(avg_background)] = 0
        interpolation = interp1d(avg_qvalues, avg_background, kind='linear', bounds_error=False, fill_value=np.nan)
        for index in range(len(ind_z)):
            array[ind_z[index], ind_y[index], ind_x[index]] =\
                array[ind_z[index], ind_y[index], ind_x[index]]\
                - interpolation(np.sqrt(qx[ind_z[index]] ** 2 + qz[ind_y[index]] ** 2 + qy[ind_x[index]] ** 2))
    elif method == 'normalize':
        avg_background[np.isnan(avg_background)] = 1
        avg_background[avg_background < 1] = 1
        interpolation = interp1d(avg_qvalues, avg_background, kind='linear', bounds_error=False, fill_value=np.nan)
        for index in range(len(ind_z)):
            array[ind_z[index], ind_y[index], ind_x[index]] =\
                array[ind_z[index], ind_y[index], ind_x[index]]\
                / interpolation(np.sqrt(qx[ind_z[index]] ** 2 + qz[ind_y[index]] ** 2 + qy[ind_x[index]] ** 2))

    array[np.isnan(array)] = 0
    array[array < 0] = 0

    return array


def skewed_gaussian(x_axis, amp, cen, sig, alpha):
    """
    Skewed Gaussian line shape.

    :param x_axis: where to calculate the function
    :param amp: the amplitude of the Gaussian
    :param cen: the position of the center
    :param sig: HWHM of the Gaussian
    :param alpha: skewness parameter
    :return: the skewed Gaussian line shape at x_axis
    """
    return amp*np.exp(-(x_axis-cen)**2/(2.*sig**2))*(1+erf(alpha/np.sqrt(2)*(x_axis-cen)/sig))


def sum_roi(array, roi, debugging=False):
    """
    Sum the array intensities in the defined region of interest.

    :param array: 2D or 3D array. If ndim=3, the region of interest is applied sequentially to each 2D
     frame, the iteration being peformed over the first axis.
    :param roi: [Vstart, Vstop, Hstart, Hstop] region of interest for the sum
    :param debugging: True to see plots
    :return: a number (if array.ndim=2) or a 1D array of length array.shape[0] (if array.ndim=3) of summed intensities
    """
    ndim = array.ndim
    if ndim == 2:
        nby, nbx = array.shape
    elif ndim == 3:
        nbz, nby, nbx = array.shape
    else:
        raise ValueError('array should be 2D or 3D')

    if not 0 <= roi[0] < roi[1] <= nby:
        raise ValueError('0 <= roi[0] < roi[1] <= nby   expected')
    if not 0 <= roi[2] < roi[3] <= nbx:
        raise ValueError('0 <= roi[2] < roi[3] <= nbx   expected')

    if ndim == 2:
        sum_array = array[roi[0]:roi[1], roi[2]:roi[3]].sum()
    else:  # ndim = 3
        sum_array = np.zeros(nbz)
        for idx in range(nbz):
            sum_array[idx] = array[idx, roi[0]:roi[1], roi[2]:roi[3]].sum()
        array = array.sum(axis=0)

    if debugging:
        val = array.max()
        array[roi[0]:roi[1], roi[2]:roi[2]+3] = val
        array[roi[0]:roi[1], roi[3]-3:roi[3]] = val
        array[roi[0]:roi[0]+3, roi[2]:roi[3]] = val
        array[roi[1]-3:roi[1], roi[2]:roi[3]] = val
        gu.combined_plots(tuple_array=(array, sum_array), tuple_sum_frames=False, tuple_sum_axis=0,
                          tuple_scale='log', tuple_title=('summed array', 'ROI integrated intensity'),
                          tuple_colorbar=True)
    return sum_array


# if __name__ == "__main__":
#     import numpy as np
#     ref_array = np.array([-0.048,1,2,3,4,5,6])
#     ind = find_nearest(reference_array=ref_array, test_values=np.array([0.2, 0.5]), width=0.5)
#     print(ind)
