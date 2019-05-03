# -*- coding: utf-8 -*-
# Calculate the diffraction pattern starting from a real object, adding Poisson noise
# the reconstruction should be orthogonalized, in the laboratory frame
import numpy as np
from numpy.random import poisson
from numpy.fft import fftn, fftshift
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import tkinter as tk
from tkinter import filedialog
from scipy.interpolate import RegularGridInterpolator
import sys
from scipy.signal import convolve
import vtk
from vtk.util import numpy_support
import gc
from scipy.ndimage import median_filter
import scipy.signal  # for medfilt2d
import os

scan = 2227  # spec scan number
datadir = "C:/Users/carnis/Work Folders/Documents/data/CH4760_Pt/S"+str(scan)+"/simu/crop400phase/new/"

sdd = 0.50678  # 1.0137  # sample to detector distance in m
en = 9000.0 - 6   # x-ray energy in eV, 6eV offset at ID01
voxel_size = 3  # in nm, voxel size of the reconstruction, should be eaqual in each direction
photon_threshold = 0  # 0.75
photon_number = 5e7  # total number of photons in the array, usually around 5e7
orthogonal_frame = False  # set to False to put the diffraction pattern in the detector frame
support_threshold = 0.24  # threshold for support determination
setup = "ID01"  # only "ID01"
rocking_angle = "outofplane"  # "outofplane" or "inplane"
outofplane_angle = 35.3240  # detector delta ID01
inplane_angle = -1.6029  # detector nu ID01
grazing_angle = 0  # in degrees, incident angle for in-plane rocking curves (eta ID01)
tilt_angle = 0.0102  # angular step size for rocking angle, eta ID01
pixel_size = 55e-6  # detector pixel size in m

set_gap = 0  # set to 1 if you want to use the detector gap in the simulation (updates the mask)
gap_width = 6  # number of pixels to mask
gap_pixel_start = 650

flat_phase = 0  # set to 1 to use a phase flat (0 everywhere)

include_noise = 0  # set to 1 to include poisson noise on the data, 0 otherwise

pad_size = [1000, 1000, 1000]  # will pad the array by this amount of zeroed pixels in z, y, x at both ends
# if only a number (e.g. 3), will pad to get three times the initial array size  # ! max size ~ [800, 800, 800]
crop_size = [400, 400, 400]  # will crop the array to this size

ref_axis_outplane = "y"  # "y"  # "z"  # q is supposed to be aligned along that axis before rotating back (nexus)
phase_range = np.pi  # for plots
strain_range = 0.001  # for plots
debug = 1  # 1 to see all plots
save_fig = True  # if True save figures
save_data = True  # if True save data as npz and VTK
comment = "_coord23_iso0.24"  # should start with _
if set_gap == 0:
    comment = comment + "_nogap"
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
#######################################


def primes(n):
    """ Returns the prime decomposition of n as a list
    """
    v = [1]
    assert (n > 0)
    i = 2
    while i * i <= n:
        while n % i == 0:
            v.append(i)
            n //= i
        i += 1
    if n > 1:
        v.append(n)
    return v


def try_smaller_primes(n, maxprime=13, required_dividers=(4,)):
    """
    Check if the largest prime divider is <=maxprime, and optionally includes some dividers.

    Args:
        n: the integer number for which the prime decomposition will be checked
        maxprime: the maximum acceptable prime number. This defaults to the largest integer accepted by the clFFT
        library for OpenCL GPU FFT.
        required_dividers: list of required dividers in the prime decomposition. If None, this check is skipped.
    Returns:
        True if the conditions are met.
    """
    p = primes(n)
    if max(p) > maxprime:
        return False
    if required_dividers is not None:
        for k in required_dividers:
            if n % k != 0:
                return False
    return True


def smaller_primes(n, maxprime=13, required_dividers=(4,)):
    """ Find the closest integer <=n (or list/array of integers), for which the largest prime divider is <=maxprime,
    and has to include some dividers.
    The default values for maxprime is the largest integer accepted by the clFFT library for OpenCL GPU FFT.

    Args:
        n: the integer number
        maxprime: the largest prime factor acceptable
        required_dividers: a list of required dividers for the returned integer.
    Returns:
        the integer (or list/array of integers) fulfilling the requirements
    """
    if (type(n) is list) or (type(n) is tuple) or (type(n) is np.ndarray):
        vn = []
        for i in n:
            assert (i > 1 and maxprime <= i)
            while try_smaller_primes(i, maxprime=maxprime, required_dividers=required_dividers) is False:
                i = i - 1
                if i == 0:
                    # TODO: should raise an exception
                    return 0
            vn.append(i)
        if type(n) is np.ndarray:
            return np.array(vn)
        return vn
    else:
        assert (n > 1 and maxprime <= n)
        while try_smaller_primes(n, maxprime=maxprime, required_dividers=required_dividers) is False:
            n = n - 1
            if n == 0:
                # TODO: should raise an exception
                return 0
        return n


def rotate_crystal(myobj, axis_to_align, reference_axis, debugging=0):
    """
    rotate myobj to align axis_to_align onto reference_axis
    :param myobj: 3d real array to be rotated
    :param axis_to_align: the axis of myobj (vector q) x y z
    :param reference_axis: will align axis_to_align onto this  x y z
    :param debugging: to plot myobj before and after rotation
    :return: rotated myobj
    """
    nbz, nby, nbx = myobj.shape
    if debugging == 1:
        plt.figure(figsize=(18, 15))
        plt.subplot(2, 2, 1)
        plt.imshow(myobj[:, :, nbx // 2],
                   vmin=0, vmax=1)
        plt.colorbar()
        plt.axis('scaled')
        plt.title("Middle slice in YZ before rotating")
        plt.subplot(2, 2, 2)
        plt.imshow(myobj[:, nby // 2, :],
                   vmin=0, vmax=1)
        plt.colorbar()
        plt.title("Middle slice in XZ before rotating")
        plt.axis('scaled')
        plt.subplot(2, 2, 3)
        plt.imshow(myobj[nbz // 2, :, :],
                   vmin=0, vmax=1)
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.title("Middle slice in XY before rotating")
        plt.axis('scaled')
        plt.pause(0.1)
    v = np.cross(axis_to_align, reference_axis)
    skew_sym_matrix = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    my_rotation_matrix = np.identity(3) + skew_sym_matrix + np.dot(skew_sym_matrix, skew_sym_matrix) /\
        (1+np.dot(axis_to_align, reference_axis))

    transfer_matrix = my_rotation_matrix.transpose()
    old_z = np.arange(-nbz // 2, nbz // 2, 1)
    old_y = np.arange(-nby // 2, nby // 2, 1)
    old_x = np.arange(-nbx // 2, nbx // 2, 1)

    myz, myy, myx = np.meshgrid(old_z, old_y, old_x, indexing='ij')

    # new_x = transfer_matrix[0, 0] * myx + transfer_matrix[0, 1] * myy + transfer_matrix[0, 2] * myz
    # new_y = transfer_matrix[1, 0] * myx + transfer_matrix[1, 1] * myy + transfer_matrix[1, 2] * myz
    # new_z = transfer_matrix[2, 0] * myx + transfer_matrix[2, 1] * myy + transfer_matrix[2, 2] * myz

    new_x = transfer_matrix[0, 0] * myx + transfer_matrix[0, 1] * myy + transfer_matrix[0, 2] * myz
    new_y = transfer_matrix[1, 0] * myx + transfer_matrix[1, 1] * myy + transfer_matrix[1, 2] * myz
    new_z = transfer_matrix[2, 0] * myx + transfer_matrix[2, 1] * myy + transfer_matrix[2, 2] * myz

    del myx, myy, myz
    rgi = RegularGridInterpolator((old_z, old_y, old_x), myobj, method='linear', bounds_error=False, fill_value=0)
    myobj = rgi(np.concatenate((new_z.reshape((1, new_z.size)), new_y.reshape((1, new_z.size)),
                                new_x.reshape((1, new_z.size)))).transpose())
    myobj = myobj.reshape((nbz, nby, nbx)).astype(myobj.dtype)
    if debugging == 1:
        plt.figure(figsize=(18, 15))
        plt.subplot(2, 2, 1)
        plt.imshow(myobj[:, :, nbx // 2],
                   vmin=0, vmax=1)
        plt.colorbar()
        plt.axis('scaled')
        plt.title("Middle slice in YZ after rotating")
        plt.subplot(2, 2, 2)
        plt.imshow(myobj[:, nby // 2, :],
                   vmin=0, vmax=1)
        plt.colorbar()
        plt.title("Middle slice in XZ after rotating")
        plt.axis('scaled')
        plt.subplot(2, 2, 3)
        plt.imshow(myobj[nbz // 2, :, :],
                   vmin=0, vmax=1)
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.title("Middle slice in XY after rotating")
        plt.axis('scaled')
        plt.pause(0.1)
    return myobj


def mask3d_maxipix(mydata, mymask, start_pixel, width_gap):
    mydata[:, :, start_pixel:start_pixel+width_gap] = 0
    mydata[:, start_pixel:start_pixel+width_gap, :] = 0

    mymask[:, :, start_pixel:start_pixel+width_gap] = 1
    mymask[:, start_pixel:start_pixel+width_gap, :] = 1
    return mydata, mymask


def detector_frame(myobj, energy, outofplane, inplane, tilt, myrocking_angle, mygrazing_angle, distance, pixel_x,
                   pixel_y, geometry, debugging=1):
    """
    interpolate orthogonal myobj back into the non-orthogonal detector frame
    :param myobj: real space object, in a non-orthogonal frame (output of phasing program)
    :param energy: in eV
    :param outofplane: in degrees
    :param inplane: in degrees  (also called inplane_angle depending on the diffractometer)
    :param tilt: angular step during the rocking curve, in degrees (ID01 geometry: eta)
    :param myrocking_angle: name of the angle which is tilted during the rocking curve
    :param mygrazing_angle: in degrees, incident angle for in-plane rocking curves (eta ID01, th 34ID, beta SIXS)
    :param distance: sample to detector distance, in meters
    :param pixel_x: horizontal pixel size, in meters
    :param pixel_y: vertical pixel size, in meters
    :param geometry: name of the setup 'ID01'or 'SIXS'
    :param debugging: to show plots before and after orthogonalization
    :return: object interpolated on an orthogonal grid
    """
    global nz, ny, nx, pad_size, crop_size  # the final detector resolution (in q) is defined by pad_size
    nbz, nby, nbx = pad_size
    numbz, numby, numbx = crop_size

    wavelength = 12.398 * 1e-7 / energy  # in m
    # TODO: check this when nx != ny != nz
    dqz = 2 * np.pi / (nz * voxel_size * 10)  # in inverse angstroms
    dqy = 2 * np.pi / (ny * voxel_size * 10)  # in inverse angstroms
    dqx = 2 * np.pi / (nx * voxel_size * 10)  # in inverse angstroms
    print('Original reciprocal space resolution (z, y, x): (', str('{:.5f}'.format(dqz)), 'A-1,',
          str('{:.5f}'.format(dqy)), 'A-1,', str('{:.5f}'.format(dqx)), 'A-1 )')
    voxelsize_z = 2 * np.pi / (nbz * dqz * 10)  # in nm
    voxelsize_y = 2 * np.pi / (nby * dqy * 10)  # in nm
    voxelsize_x = 2 * np.pi / (nbx * dqx * 10)  # in nm
    print('New voxel sizes (z, y, x) after padding: (', str('{:.2f}'.format(voxelsize_z)), 'nm,',
          str('{:.2f}'.format(voxelsize_y)), 'nm,', str('{:.2f}'.format(voxelsize_x)), 'nm )')
    dqz = 2 * np.pi / (nbz * voxelsize_z * 10)  # in inverse angstroms
    dqy = 2 * np.pi / (nby * voxelsize_y * 10)  # in inverse angstroms
    dqx = 2 * np.pi / (nbx * voxelsize_x * 10)  # in inverse angstroms
    print('New reciprocal space resolution (z, y, x) after padding: (', str('{:.5f}'.format(dqz)), 'A-1,',
          str('{:.5f}'.format(dqy)), 'A-1,', str('{:.5f}'.format(dqx)), 'A-1 )')
    voxelsizez_crop = 2 * np.pi / (numbz * dqz * 10)  # in nm
    voxelsizey_crop = 2 * np.pi / (numby * dqy * 10)  # in nm
    voxelsizex_crop = 2 * np.pi / (numbx * dqx * 10)  # in nm
    print('New voxel sizes (z, y, x) after cropping: (', str('{:.2f}'.format(voxelsizez_crop)), 'nm,',
          str('{:.2f}'.format(voxelsizey_crop)), 'nm,', str('{:.2f}'.format(voxelsizex_crop)), 'nm )')
    if debugging == 1:
        plt.figure(figsize=(18, 15))
        plt.subplot(2, 2, 1)
        plt.imshow(abs(myobj).sum(axis=2), cmap=my_cmap)
        plt.colorbar()
        plt.axis('scaled')
        plt.title('Sum(ortho_obj) in YZ')
        plt.subplot(2, 2, 2)
        plt.imshow(abs(myobj).sum(axis=1), cmap=my_cmap)
        plt.colorbar()
        plt.axis('scaled')
        plt.title('Sum(ortho_obj) in XZ')
        plt.subplot(2, 2, 3)
        plt.imshow(abs(myobj).sum(axis=0), cmap=my_cmap)
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.axis('scaled')
        plt.title('Sum(ortho_obj) in XY')
        plt.pause(0.1)

    myz, myy, myx = np.meshgrid(np.arange(0, nz, 1), np.arange(0, ny, 1), np.arange(0, nx, 1),
                                indexing='ij')
    _, _, _, ortho_matrix = update_coords(mygrid=(myz, myy, myx), wavelength=wavelength,
                                          outofplane=outofplane, inplane=inplane, tilt=tilt,
                                          myrocking_angle=myrocking_angle, mygrazing_angle=mygrazing_angle,
                                          distance=distance, pixel_x=pixel_x, pixel_y=pixel_y,
                                          geometry=geometry)
    del myz, myy, myx

    ############################
    # Vincent's method using inverse transformation
    ############################

    myz, myy, myx = np.meshgrid(np.arange(-nz//2, nz//2, 1),
                                np.arange(-ny//2, ny//2, 1),
                                np.arange(-nx//2, nx//2, 1), indexing='ij')

    new_x = ortho_matrix[0, 0] * myx + ortho_matrix[0, 1] * myy + ortho_matrix[0, 2] * myz
    new_y = ortho_matrix[1, 0] * myx + ortho_matrix[1, 1] * myy + ortho_matrix[1, 2] * myz
    new_z = ortho_matrix[2, 0] * myx + ortho_matrix[2, 1] * myy + ortho_matrix[2, 2] * myz
    del myx, myy, myz
    # la partie rgi est sure: c'est la taille de l'objet orthogonal de depart
    rgi = RegularGridInterpolator((np.arange(-nz//2, nz//2)*voxel_size*nbz/nz,
                                   np.arange(-ny//2, ny//2)*voxel_size*nby/ny,
                                   np.arange(-nx//2, nx//2)*voxel_size*nbx/nx),
                                  myobj, method='linear', bounds_error=False, fill_value=0)
    detector_obj = rgi(np.concatenate((new_z.reshape((1, new_z.size)), new_y.reshape((1, new_z.size)),
                                      new_x.reshape((1, new_z.size)))).transpose())
    detector_obj = detector_obj.reshape((nz, ny, nx)).astype(myobj.dtype)

    if debugging == 1:
        plt.figure(figsize=(18, 15))
        plt.subplot(2, 2, 1)
        plt.imshow(abs(detector_obj).sum(axis=2), cmap=my_cmap)
        plt.colorbar()
        plt.axis('scaled')
        plt.title('Sum(non_ortho_obj) in YZ')
        plt.subplot(2, 2, 2)
        plt.imshow(abs(detector_obj).sum(axis=1), cmap=my_cmap)
        plt.colorbar()
        plt.axis('scaled')
        plt.title('Sum(non_ortho_obj) in XZ')
        plt.subplot(2, 2, 3)
        plt.imshow(abs(detector_obj).sum(axis=0), cmap=my_cmap)
        plt.colorbar()
        plt.axis('scaled')
        plt.title('Sum(non_ortho_obj) in XY')
        plt.pause(0.1)

    # myz, myy, myx = np.meshgrid(np.arange(-nz//2, nz//2, 1)*voxel_size, np.arange(-ny//2, ny//2, 1)*voxel_size,
    #                             np.arange(-nx//2, nx//2, 1)*voxel_size, indexing='ij')
    # ortho_imatrix = np.linalg.inv(ortho_matrix)
    # new_x = ortho_imatrix[0, 0] * myx + ortho_imatrix[0, 1] * myy + ortho_imatrix[0, 2] * myz
    # new_y = ortho_imatrix[1, 0] * myx + ortho_imatrix[1, 1] * myy + ortho_imatrix[1, 2] * myz
    # new_z = ortho_imatrix[2, 0] * myx + ortho_imatrix[2, 1] * myy + ortho_imatrix[2, 2] * myz
    # del myx, myy, myz
    # rgi = RegularGridInterpolator((np.arange(-nz//2, nz//2), np.arange(-ny//2, ny//2),
    #                                np.arange(-nx//2, nx//2)), detector_obj, method='linear',
    #                               bounds_error=False, fill_value=0)
    # ortho_obj = rgi(np.concatenate((new_z.reshape((1, new_z.size)), new_y.reshape((1, new_z.size)),
    #                                 new_x.reshape((1, new_z.size)))).transpose())
    # ortho_obj = ortho_obj.reshape((nz, ny, nx)).astype(detector_obj.dtype)
    # if debugging == 1:
    #     plt.figure(figsize=(18, 15))
    #     plt.subplot(2, 2, 1)
    #     plt.imshow(abs(ortho_obj).sum(axis=2), cmap=my_cmap)
    #     plt.colorbar()
    #     plt.axis('scaled')
    #     plt.title('Sum(ortho_obj) in YZ')
    #     plt.subplot(2, 2, 2)
    #     plt.imshow(abs(ortho_obj).sum(axis=1), cmap=my_cmap)
    #     plt.colorbar()
    #     plt.axis('scaled')
    #     plt.title('Sum(ortho_obj) in XZ')
    #     plt.subplot(2, 2, 3)
    #     plt.imshow(abs(ortho_obj).sum(axis=0), cmap=my_cmap)
    #     plt.colorbar()
    #     plt.axis('scaled')
    #     plt.title('Sum(ortho_obj) in XY')
    #     plt.pause(0.1)
    return detector_obj, voxelsize_z, voxelsize_y, voxelsize_x


def update_coords(mygrid, wavelength, outofplane, inplane, tilt, myrocking_angle, mygrazing_angle, distance,
                  pixel_x, pixel_y, geometry):
    """
    calculate the pixel non-orthogonal coordinates in the orthogonal reference frame
    :param mygrid: grid corresponding to the real object size
    :param wavelength: in m
    :param outofplane: in degrees
    :param inplane: in degrees  (also called inplane_angle depending on the diffractometer)
    :param tilt: angular step during the rocking curve, in degrees
    :param myrocking_angle: name of the motor which is tilted during the rocking curve
    :param mygrazing_angle: in degrees, incident angle for in-plane rocking curves (eta ID01, th 34ID, beta SIXS)
    :param distance: sample to detector distance, in meters
    :param pixel_x: horizontal pixel size, in meters
    :param pixel_y: vertical pixel size, in meters
    :param geometry: name of the setup 'ID01'or 'SIXS'
    :return: coordinates of the non-orthogonal grid in the orthogonal reference grid
    """
    wavelength = wavelength * 1e9  # convert to nm
    distance = distance * 1e9  # convert to nm
    lambdaz = wavelength * distance
    pixel_x = pixel_x * 1e9  # convert to nm
    pixel_y = pixel_y * 1e9  # convert to nm
    mymatrix = np.zeros((3, 3))
    outofplane = np.radians(outofplane)
    inplane = np.radians(inplane)
    tilt = np.radians(tilt)
    mygrazing_angle = np.radians(mygrazing_angle)
    nbz, nby, nbx = mygrid[0].shape

    if geometry == 'ID01':
        print('using ESRF ID01 geometry')
        if myrocking_angle == "outofplane":
            print('rocking angle is eta')
            # rocking eta angle clockwise around x (phi does not matter, above eta)
            mymatrix[:, 0] = 2*np.pi*nbx / lambdaz * np.array([pixel_x*np.cos(inplane),
                                                               0,
                                                               pixel_x*np.sin(inplane)])
            mymatrix[:, 1] = 2*np.pi*nby / lambdaz * np.array([-pixel_y*np.sin(inplane)*np.sin(outofplane),
                                                              -pixel_y*np.cos(outofplane),
                                                              pixel_y*np.cos(inplane)*np.sin(outofplane)])
            mymatrix[:, 2] = 2*np.pi*nbz / lambdaz * np.array([0,
                                                               tilt*distance*(1-np.cos(inplane)*np.cos(outofplane)),
                                                               tilt*distance*np.sin(outofplane)])
        elif myrocking_angle == "inplane" and mygrazing_angle == 0:
            print('rocking angle is phi, eta=0')
            # rocking phi angle clockwise around y, assuming incident angle eta is zero (eta below phi)
            mymatrix[:, 0] = 2*np.pi*nbx / lambdaz * np.array([pixel_x*np.cos(inplane),
                                                               0,
                                                               pixel_x*np.sin(inplane)])
            mymatrix[:, 1] = 2*np.pi*nby / lambdaz * np.array([-pixel_y*np.sin(inplane)*np.sin(outofplane),
                                                               -pixel_y*np.cos(outofplane),
                                                               pixel_y*np.cos(inplane)*np.sin(outofplane)])
            mymatrix[:, 2] = 2*np.pi*nbz / lambdaz * np.array([-tilt*distance*(1-np.cos(inplane)*np.cos(outofplane)),
                                                               0,
                                                               tilt*distance*np.sin(inplane)*np.cos(outofplane)])
        elif myrocking_angle == "inplane" and mygrazing_angle != 0:
            print('rocking angle is phi, with eta non zero')
            # rocking phi angle clockwise around y, incident angle eta is non zero (eta below phi)
            mymatrix[:, 0] = 2*np.pi*nbx / lambdaz * np.array([pixel_x*np.cos(inplane),
                                                               0,
                                                               pixel_x*np.sin(inplane)])
            mymatrix[:, 1] = 2*np.pi*nby / lambdaz * np.array([-pixel_y*np.sin(inplane)*np.sin(outofplane),
                                                               -pixel_y*np.cos(outofplane),
                                                               pixel_y*np.cos(inplane)*np.sin(outofplane)])
            mymatrix[:, 2] = 2*np.pi*nbz / lambdaz * tilt * distance * \
                np.array([(np.sin(mygrazing_angle)*np.sin(outofplane) +
                          np.cos(mygrazing_angle)*(np.cos(inplane)*np.cos(outofplane)-1)),
                          np.sin(mygrazing_angle)*np.sin(inplane)*np.sin(outofplane),
                          np.cos(mygrazing_angle)*np.sin(inplane)*np.cos(outofplane)])
    transfer_matrix = 2*np.pi * np.linalg.inv(mymatrix).transpose()   # to go to orthogonal laboratory frame
    out_x = transfer_matrix[0, 0] * mygrid[2] + transfer_matrix[0, 1] * mygrid[1] + transfer_matrix[0, 2] * mygrid[0]
    out_y = transfer_matrix[1, 0] * mygrid[2] + transfer_matrix[1, 1] * mygrid[1] + transfer_matrix[1, 2] * mygrid[0]
    out_z = transfer_matrix[2, 0] * mygrid[2] + transfer_matrix[2, 1] * mygrid[1] + transfer_matrix[2, 2] * mygrid[0]

    return out_z, out_y, out_x, transfer_matrix


def wrap(myphase):
    """
    wrap the phase in [-pi pi] interval
    :param myphase:
    :return:
    """
    myphase = (myphase + np.pi) % (2 * np.pi) - np.pi
    return myphase


def plane_angle(ref_plane, plane):
    """
    Calculate the angle between two crystallographic planes in cubic materials
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


def calc_coordination(mysupport, debugging=0):
    nbz, nby, nbx = mysupport.shape

    mykernel = np.ones((3, 3, 3))
    mycoord = np.rint(convolve(mysupport, mykernel, mode='same'))
    mycoord = mycoord.astype(int)

    if debugging == 1:
        plt.figure(figsize=(18, 15))
        plt.subplot(2, 2, 1)
        plt.imshow(mycoord[:, :, nbx // 2])
        plt.colorbar()
        plt.axis('scaled')
        plt.title("Coordination matrix in middle slice in YZ")
        plt.subplot(2, 2, 2)
        plt.imshow(mycoord[:, nby // 2, :])
        plt.colorbar()
        plt.title("Coordination matrix in middle slice in XZ")
        plt.axis('scaled')
        plt.subplot(2, 2, 3)
        plt.imshow(mycoord[nbz // 2, :, :])
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.title("Coordination matrix in middle slice in XY")
        plt.axis('scaled')
        plt.pause(0.1)
    return mycoord


###########################################
plt.ion()
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(initialdir=datadir, filetypes=[("NPZ", "*.npz")])
npzfile = np.load(file_path)

amp = npzfile['amp']
nz, ny, nx = amp.shape
print("Initial data size: (", nz, ',', ny, ',', nx, ')')

plt.figure(figsize=(18, 15))
plt.subplot(2, 2, 1)
plt.imshow(amp[nz // 2 - 100:nz // 2 + 100, ny // 2 - 100:ny // 2 + 100, nx // 2], vmin=0, vmax=1, cmap=my_cmap)
plt.colorbar()
plt.axis('scaled')
plt.title("Amp at middle frame in YZ")
plt.subplot(2, 2, 2)
plt.imshow(amp[nz // 2 - 100:nz // 2 + 100, ny // 2, nx // 2 - 100:nx // 2 + 100], vmin=0, vmax=1, cmap=my_cmap)
plt.colorbar()
plt.title("Amp at middle frame in XZ")
plt.axis('scaled')
plt.subplot(2, 2, 3)
plt.imshow(amp[nz // 2, ny // 2 - 100:ny // 2 + 100, nx // 2 - 100:nx // 2 + 100], vmin=0, vmax=1, cmap=my_cmap)
plt.gca().invert_yaxis()
plt.colorbar()
plt.title("Amp at middle frame in XY")
plt.axis('scaled')
plt.pause(0.1)

#######################################
# q from direct calculation
wave = 12.398 * 1e-7 / en  # wavelength in m
kin = 2*np.pi/wave * np.array([1, 0, 0])  # z downstream, y vertical, x outboard
if setup == 'ID01':
    # nu is clockwise
    kout = 2 * np.pi / wave * np.array([np.cos(np.pi*inplane_angle/180)*np.cos(np.pi*outofplane_angle/180),  # z
                                       np.sin(np.pi*outofplane_angle/180),  # y
                                       -np.sin(np.pi*inplane_angle/180)*np.cos(np.pi*outofplane_angle/180)])  # x
else:
    print('Back rotation not yet implemented for other beamlines')
    sys.exit()
q = kout - kin
Qnorm = np.linalg.norm(q)
q = q / Qnorm
Qnorm = Qnorm * 1e-10  # switch to angstroms
planar_dist = 2*np.pi/Qnorm  # Qnorm should be in angstroms
print("Wavevector transfer [z, y, x]:", q*Qnorm)
print("Wavevector transfer: (angstroms)", str('{:.4f}'.format(Qnorm)))
print("Atomic plane distance: (angstroms)", str('{:.4f}'.format(planar_dist)), "angstroms")
planar_dist = planar_dist / 10  # switch to nm
########################################
support = np.ones((nz, ny, nx))

if flat_phase == 1:
    phase = np.zeros((nz, ny, nx))
else:
    # tentative new model
    # phase_offset = -3
    # phase = np.zeros((nz, ny, nx)) + phase_offset
    # z, y, x = np.meshgrid(np.arange(-nz // 2, nz // 2, 1), np.arange(-ny // 2+25, ny // 2+25, 1),
    #                       np.arange(-nx // 2, nx // 2, 1), indexing='ij')
    # distances = np.sqrt(z**2+y**2+x**2)  # distances in pixel from [0, 0, 0]
    # phase_start = 44
    # phase_middle = 45
    # phase_stop = 120  # stop of the phase variation in pixels from COM
    # coeff = 1.2
    # coeff2 = 26
    # step_r = 0.5
    # xaxis = np.arange(phase_middle, phase_stop, step_r)
    # yaxis = (np.exp((xaxis - phase_middle) / coeff2)) * coeff - \
    #         (np.exp((xaxis[0] - phase_middle) / coeff2)) * coeff + phase_offset
    # for idx in np.arange(phase_middle, phase_stop, step_r):
    #     phase[distances > idx] = (np.exp((idx-phase_middle)/coeff2)) * coeff - \
    #                              (np.exp((xaxis[0]-phase_middle)/coeff2)) * coeff + phase_offset
    # for idx in np.arange(phase_start, 0, -step_r):
    #     phase[distances < idx] = phase_offset - (idx - phase_start) * (yaxis[1]-yaxis[0])
    # plt.figure()
    # plt.plot(xaxis, yaxis)
    # plt.pause(0.1)

    # model for paper about artefacts in BCDI
    oscillation_period = 100  # in pixels
    z, y, x = np.meshgrid(np.cos(np.arange(-nz // 2, nz // 2, 1)*2*np.pi/oscillation_period),
                          np.cos(np.arange(-ny // 2, ny // 2, 1)*2*np.pi/oscillation_period),
                          np.cos(np.arange(-nx // 2, nx // 2, 1)*2*np.pi/oscillation_period), indexing='ij')
    phase = z + y + x

if debug:
    plt.figure(figsize=(18, 15))
    plt.subplot(2, 2, 1)
    plt.imshow(phase[nz // 2 - 100:nz // 2 + 100, ny // 2 - 100:ny // 2 + 100, nx // 2], vmin=-phase_range,
               vmax=phase_range, cmap=my_cmap)
    plt.colorbar()
    plt.axis('scaled')
    plt.title("Phase at middle frame in YZ before wrap")
    plt.subplot(2, 2, 2)
    plt.imshow(phase[nz // 2 - 100:nz // 2 + 100, ny // 2, nx // 2 - 100:nx // 2 + 100], vmin=-phase_range,
               vmax=phase_range, cmap=my_cmap)
    plt.colorbar()
    plt.title("Phase at middle frame in XZ before wrap")
    plt.axis('scaled')
    plt.subplot(2, 2, 3)
    plt.imshow(phase[nz // 2, ny // 2 - 100:ny // 2 + 100, nx // 2 - 100:nx // 2 + 100], vmin=-phase_range,
               vmax=phase_range, cmap=my_cmap)
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.title("Phase at middle frame in XY before wrap")
    plt.axis('scaled')
    plt.pause(0.1)

phase = wrap(phase)

support[abs(amp) < support_threshold * abs(amp).max()] = 0
del amp
phase[support == 0] = 0

if ref_axis_outplane == "x":
    _, _, strain = np.gradient(planar_dist / (2 * np.pi) * phase,
                               voxel_size)  # q is along x after rotating the crystal
elif ref_axis_outplane == "y":
    _, strain, _ = np.gradient(planar_dist / (2 * np.pi) * phase,
                               voxel_size)  # q is along y after rotating the crystal
elif ref_axis_outplane == "z":
    strain, _, _ = np.gradient(planar_dist / (2 * np.pi) * phase,
                               voxel_size)  # q is along y after rotating the crystal
else:  # default is ref_axis_outplane = "y"
    _, strain, _ = np.gradient(planar_dist / (2 * np.pi) * phase,
                               voxel_size)  # q is along y after rotating the crystal

# remove the outer layer of support for saving, because strain is undefined there
coordination_matrix = calc_coordination(support, debugging=0)
surface = np.copy(support)
surface[coordination_matrix > 23] = 0  # remove the bulk 22
bulk = support - surface
bulk[np.nonzero(bulk)] = 1

if debug == 1:
    plt.figure(figsize=(18, 15))
    plt.subplot(2, 3, 1)
    plt.imshow(surface[nz//2-100:nz//2+100, ny//2-100:ny//2+100, nx//2])
    plt.colorbar()
    plt.axis('scaled')
    plt.title("Surface matrix in middle slice in YZ")
    plt.subplot(2, 3, 2)
    plt.imshow(surface[nz//2-100:nz//2+100, ny//2, nx//2-100:nx//2+100])
    plt.colorbar()
    plt.title("Surface matrix in middle slice in XZ")
    plt.axis('scaled')
    plt.subplot(2, 3, 3)
    plt.imshow(surface[nz//2, ny//2-100:ny//2+100, nx//2-100:nx//2+100])
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.title("Surface matrix in middle slice in XY")
    plt.axis('scaled')

    surface = np.multiply(surface, strain)

    plt.subplot(2, 3, 4)
    plt.imshow(surface[nz//2-100:nz//2+100, ny//2-100:ny//2+100, nx//2], vmin=-strain_range,
               vmax=strain_range, cmap=my_cmap)
    plt.colorbar()
    plt.axis('scaled')
    plt.title("Surface strain in middle slice in YZ")
    plt.subplot(2, 3, 5)
    plt.imshow(surface[nz//2-100:nz//2+100, ny//2, nx//2-100:nx//2+100], vmin=-strain_range,
               vmax=strain_range, cmap=my_cmap)
    plt.colorbar()
    plt.title("Surface strain in middle slice in XZ")
    plt.axis('scaled')
    plt.subplot(2, 3, 6)
    plt.imshow(surface[nz//2, ny//2-100:ny//2+100, nx//2-100:nx//2+100], vmin=-strain_range,
               vmax=strain_range, cmap=my_cmap)
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.title("Surface strain in middle slice in XY")
    plt.axis('scaled')

    plt.figure(figsize=(18, 15))
    plt.subplot(2, 2, 1)
    plt.imshow(support.sum(axis=2), cmap=my_cmap)
    plt.colorbar()
    plt.axis('scaled')
    plt.title('Sum(orthogonal support) in YZ')
    plt.subplot(2, 2, 2)
    plt.imshow(support.sum(axis=1), cmap=my_cmap)
    plt.colorbar()
    plt.axis('scaled')
    plt.title('Sum(orthogonal support) in XZ')
    plt.subplot(2, 2, 3)
    plt.imshow(support.sum(axis=0), cmap=my_cmap)
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.axis('scaled')
    plt.title('Sum(orthogonal support) in XY')
    plt.pause(0.1)

    if flat_phase == 0:
        plt.figure(figsize=(18, 15))
        plt.subplot(2, 2, 1)
        plt.imshow(phase[nz//2-100:nz//2+100, ny//2-100:ny//2+100, nx//2], vmin=-phase_range,
                   vmax=phase_range, cmap=my_cmap)
        plt.colorbar()
        plt.axis('scaled')
        plt.title("Phase at middle frame in YZ")
        plt.subplot(2, 2, 2)
        plt.imshow(phase[nz//2-100:nz//2+100, ny//2, nx//2-100:nx//2+100], vmin=-phase_range,
                   vmax=phase_range, cmap=my_cmap)
        plt.colorbar()
        plt.title("Phase at middle frame in XZ")
        plt.axis('scaled')
        plt.subplot(2, 2, 3)
        plt.imshow(phase[nz//2, ny//2-100:ny//2+100, nx//2-100:nx//2+100], vmin=-phase_range,
                   vmax=phase_range, cmap=my_cmap)
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.title("Phase at middle frame in XY")
        plt.axis('scaled')
        plt.pause(0.1)

        strain[bulk == 0] = 0  # for easier visualization
        if save_fig == 1:
            plt.savefig(
                datadir + 'S' + str(scan) + '_phase_' + str('{:.0e}'.format(photon_number)) + comment + '.png')
        if save_data == 1:
            np.savez_compressed(datadir + 'S' + str(scan) + '_amp-phase-strain_SIMU' + comment,
                                amp=support, phase=phase, bulk=bulk, strain=strain)
            # save amp & phase to VTK
            # in VTK, x is downstream, y vertical, z inboard, thus need to flip the last axis
            temp_array = np.copy(support)
            temp_array = np.transpose(np.flip(temp_array, 2)).reshape(temp_array.size)
            temp_array = numpy_support.numpy_to_vtk(temp_array)

            image_data = vtk.vtkImageData()
            image_data.SetOrigin(0, 0, 0)
            image_data.SetSpacing(voxel_size, voxel_size, voxel_size)
            image_data.SetExtent(0, nz - 1, 0, ny - 1, 0, nx - 1)

            pd = image_data.GetPointData()
            pd.SetScalars(temp_array)
            pd.GetArray(0).SetName("amp")

            temp_array = np.copy(bulk)
            temp_array = np.transpose(np.flip(temp_array, 2)).reshape(temp_array.size)
            temp_array = numpy_support.numpy_to_vtk(temp_array)
            pd.AddArray(temp_array)
            pd.GetArray(1).SetName("amp_bulk")
            pd.Update()

            temp_array = np.copy(phase)
            temp_array = np.transpose(np.flip(temp_array, 2)).reshape(temp_array.size)
            temp_array = numpy_support.numpy_to_vtk(temp_array)
            pd.AddArray(temp_array)
            pd.GetArray(2).SetName("phase")
            pd.Update()

            temp_array = np.copy(strain)
            temp_array = np.transpose(np.flip(temp_array, 2)).reshape(temp_array.size)
            temp_array = numpy_support.numpy_to_vtk(temp_array)
            pd.AddArray(temp_array)
            pd.GetArray(3).SetName("strain")
            pd.Update()
            # export data to file
            writer = vtk.vtkXMLImageDataWriter()
            writer.SetFileName(os.path.join(datadir, "S" + str(scan) + "_amp-phase-strain_SIMU" + comment + ".vti"))
            writer.SetInputData(image_data)
            writer.Write()
            del temp_array, pd, writer, image_data

        plt.figure(figsize=(18, 15))
        plt.subplot(2, 2, 1)
        plt.imshow(strain[nz//2-100:nz//2+100, ny//2-100:ny//2+100, nx//2], vmin=-strain_range,
                   vmax=strain_range, cmap=my_cmap)
        plt.colorbar()
        plt.axis('scaled')
        plt.title("Strain at middle frame in YZ")
        plt.subplot(2, 2, 2)
        plt.imshow(strain[nz//2-100:nz//2+100, ny//2, nx//2-100:nx//2+100], vmin=-strain_range,
                   vmax=strain_range, cmap=my_cmap)
        plt.colorbar()
        plt.title("Strain at middle frame in XZ")
        plt.axis('scaled')
        plt.subplot(2, 2, 3)
        plt.imshow(strain[nz//2, ny//2-100:ny//2+100, nx//2-100:nx//2+100], vmin=-strain_range,
                   vmax=strain_range, cmap=my_cmap)
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.title("Strain at middle frame in XY")
        plt.axis('scaled')
        plt.pause(0.1)
        if save_fig == 1:
            plt.savefig(
                datadir + 'S' + str(scan) + '_strain_' + str('{:.0e}'.format(photon_number)) + comment + '.png')

del strain, bulk

if orthogonal_frame:
    obj = support * np.exp(1j * phase)
    del phase, support
    gc.collect()
    comment = comment + '_prtf'
    set_gap = 0  # gap is valid only in the detector frame
else:
    #############################################
    # rotate back object to have q in the same direction as in experiment
    #############################################
    if ref_axis_outplane == "x":
        myaxis = np.array([1, 0, 0])  # must be in [x, y, z] order
    elif ref_axis_outplane == "y":
        myaxis = np.array([0, 1, 0])  # must be in [x, y, z] order
    elif ref_axis_outplane == "z":
        myaxis = np.array([0, 0, 1])  # must be in [x, y, z] order
    else:
        ref_axis_outplane = "y"
        myaxis = np.array([0, 1, 0])  # must be in [x, y, z] order
    print('Q aligned along ', ref_axis_outplane, ":", myaxis)
    angle = plane_angle(np.array([q[2], q[1], q[0]])/np.linalg.norm(q), myaxis)
    print("Angle between q and", ref_axis_outplane, "=", angle, "deg")
    print("Angle with y in zy plane", np.arctan(q[0]/q[1])*180/np.pi, "deg")
    print("Angle with y in xy plane", np.arctan(-q[2]/q[1])*180/np.pi, "deg")
    print("Angle with z in xz plane", 180+np.arctan(q[2]/q[0])*180/np.pi, "deg")
    support = rotate_crystal(support, axis_to_align=myaxis,
                             reference_axis=np.array([q[2], q[1], q[0]]) / np.linalg.norm(q), debugging=1)
    phase = rotate_crystal(phase, axis_to_align=myaxis,
                           reference_axis=np.array([q[2], q[1], q[0]]) / np.linalg.norm(q), debugging=0)

    obj = support * np.exp(1j * phase)
    del phase, support
    gc.collect()

    #############################################
    # transform object back into detector frame
    #############################################

    obj, _, _, _ = detector_frame(myobj=obj, energy=en, outofplane=outofplane_angle,
                                  inplane=inplane_angle, tilt=tilt_angle, myrocking_angle=rocking_angle,
                                  mygrazing_angle=grazing_angle, distance=sdd, pixel_x=pixel_size,
                                  pixel_y=pixel_size, geometry=setup, debugging=1)
    if debug == 1:
        plt.figure(figsize=(18, 15))
        plt.subplot(2, 2, 1)
        plt.imshow(abs(obj.sum(axis=2)), cmap=my_cmap)
        plt.colorbar()
        plt.axis('scaled')
        plt.title('Sum(Non orthogonal support) in YZ')
        plt.subplot(2, 2, 2)
        plt.imshow(abs(obj.sum(axis=1)), cmap=my_cmap)
        plt.colorbar()
        plt.axis('scaled')
        plt.title('Sum(Non orthogonal support) in XZ')
        plt.subplot(2, 2, 3)
        plt.imshow(abs(obj.sum(axis=0)), cmap=my_cmap)
        plt.colorbar()
        plt.axis('scaled')
        plt.title('Sum(Non orthogonal support) in XY')

        plt.figure(figsize=(18, 15))
        plt.subplot(2, 2, 1)
        plt.imshow(np.angle(obj)[:, :, nx // 2], vmin=-phase_range, vmax=phase_range, cmap=my_cmap)
        plt.colorbar()
        plt.axis('scaled')
        plt.title("Phase at middle frame in YZ")
        plt.subplot(2, 2, 2)
        plt.imshow(np.angle(obj)[:, ny // 2, :], vmin=-phase_range, vmax=phase_range, cmap=my_cmap)
        plt.colorbar()
        plt.title("Phase at middle frame in XZ")
        plt.axis('scaled')
        plt.subplot(2, 2, 3)
        plt.imshow(np.angle(obj)[nz // 2, :, :], vmin=-phase_range, vmax=phase_range, cmap=my_cmap)
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.title("Phase at middle frame in XY")
        plt.axis('scaled')
        plt.pause(0.1)
    ##################################
    # uncomment this if you want to save the non-orthogonal support
    # in that case pad_size and crop_size should be identical
    ##################################
    # support = abs(obj)
    # support = support / support.max()
    # support[support < 0.05] = 0
    # support[np.nonzero(support)] = 1
    # np.savez_compressed(datadir + 'S' + str(scan) + 'support_nonortho400.npz', obj=support)

####################################
# pad array
####################################
nz1, ny1, nx1 = pad_size
if nz1 < nz or ny1 < ny or nx1 < nx:
    print('Pad size smaller than initial array size')
    sys.exit()

newobj = np.zeros((nz1, ny1, nx1), dtype=complex)
newobj[(nz1-nz)//2:(nz1+nz)//2, (ny1-ny)//2:(ny1+ny)//2, (nx1-nx)//2:(nx1+nx)//2] = obj
nz1, ny1, nx1 = newobj.shape
print("Padded data size: (", nz1, ',', ny1, ',', nx1, ')')
comment = comment + "_pad_" + str(nz1) + "," + str(ny1) + "," + str(nx1)

#############################################
# calculate the diffraction pattern and add detector gaps
#############################################
data = fftshift(abs(fftn(newobj))**2)
data = data / data.sum() * photon_number  # convert into photon number

#############################################
# apply photon threshold
#############################################
mask = np.zeros((nz1, ny1, nx1))
mask[data <= photon_threshold] = 1
data[data <= photon_threshold] = 0
fig = plt.figure(figsize=(18, 15))
plt.subplot(2, 2, 1)
plt.imshow(np.log10(abs(data[:, :, nx1 // 2])), cmap=my_cmap, vmin=-5)
plt.colorbar()
plt.xlabel(r"Q$_z$ ($1/\AA$)")
plt.ylabel(r"Q$_x$ ($1/\AA$)")
plt.axis('scaled')
plt.title('Middle frame in Qy after padding')
plt.subplot(2, 2, 2)
plt.imshow(np.log10(abs(data[:, ny1 // 2, :])), cmap=my_cmap, vmin=-5)
plt.xlabel(r"Q$_y$ ($1/\AA$)")
plt.ylabel(r"Q$_x$ ($1/\AA$)")
plt.colorbar()
plt.axis('scaled')
plt.title('Middle frame in Qz after padding')
plt.subplot(2, 2, 3)
plt.imshow(np.log10(abs(data[nz1 // 2, :, :])), cmap=my_cmap, vmin=-5)
plt.xlabel(r"Q$_y$ ($1/\AA$)")
plt.ylabel(r"Q$_z$ ($1/\AA$)")
plt.colorbar()
plt.axis('scaled')
plt.title('Middle frame in Qx after padding')
plt.pause(0.1)
if save_fig == 1:
    plt.savefig(datadir + 'S' + str(scan) + '_diff_float_' + str('{:.0e}'.format(photon_number))+comment + '_sum.png')
#############################################
# include noise
#############################################
if include_noise == 1:
    data = np.rint(poisson(data)).astype(int)
    comment = comment + "_noise"
else:
    data = np.rint(data).astype(int)

if set_gap == 1:
    data, mask = mask3d_maxipix(data, mask, start_pixel=gap_pixel_start, width_gap=gap_width)

fig = plt.figure(figsize=(18, 15))
plt.subplot(2, 2, 1)
plt.imshow(np.log10(abs(data[:, :, nx1 // 2])), cmap=my_cmap, vmin=-1)
plt.colorbar()
plt.xlabel(r"Q$_z$ ($1/\AA$)")
plt.ylabel(r"Q$_x$ ($1/\AA$)")
plt.axis('scaled')
plt.title('Middle frame in Qy after rounding')
plt.subplot(2, 2, 2)
plt.imshow(np.log10(abs(data[:, ny1 // 2, :])), cmap=my_cmap, vmin=-1)
plt.xlabel(r"Q$_y$ ($1/\AA$)")
plt.ylabel(r"Q$_x$ ($1/\AA$)")
plt.colorbar()
plt.axis('scaled')
plt.title('Middle frame in Qz after rounding')
plt.subplot(2, 2, 3)
plt.imshow(np.log10(abs(data[nz1 // 2, :, :])), cmap=my_cmap, vmin=-1)
plt.xlabel(r"Q$_y$ ($1/\AA$)")
plt.ylabel(r"Q$_z$ ($1/\AA$)")
plt.colorbar()
plt.axis('scaled')
plt.title('Middle frame in Qx after rounding')
plt.pause(0.1)

fig = plt.figure(figsize=(18, 15))
plt.subplot(2, 2, 1)
plt.imshow(np.log10(abs(data.sum(axis=2))), cmap=my_cmap, vmin=-1)
plt.colorbar()
plt.xlabel(r"Q$_z$ ($1/\AA$)")
plt.ylabel(r"Q$_x$ ($1/\AA$)")
plt.axis('scaled')
plt.title('masked intensity summed over Qy')
plt.subplot(2, 2, 2)
plt.imshow(np.log10(abs(data.sum(axis=1))), cmap=my_cmap, vmin=-1)
plt.xlabel(r"Q$_y$ ($1/\AA$)")
plt.ylabel(r"Q$_x$ ($1/\AA$)")
plt.colorbar()
plt.axis('scaled')
plt.title('masked intensity summed over Qz')
plt.subplot(2, 2, 3)
plt.imshow(np.log10(abs(data.sum(axis=0))), cmap=my_cmap, vmin=-1)
plt.xlabel(r"Q$_y$ ($1/\AA$)")
plt.ylabel(r"Q$_z$ ($1/\AA$)")
plt.colorbar()
plt.axis('scaled')
plt.title('masked intensity summed over Qx')
fig.text(0.60, 0.30, "Pad size =" + str(pad_size), size=20)
if save_fig == 1:
    plt.savefig(datadir + 'S' + str(scan) + '_diff_' + str('{:.0e}'.format(photon_number))+comment + '_sum.png')
###########################################
# crop arrays
###########################################
nz, ny, nx = data.shape
nz1, ny1, nx1 = crop_size
if nz < nz1 or ny < ny1 or nx < nx1:
    print('Crop size larger than initial array size')
    sys.exit()
data = data[(nz - nz1) // 2:(nz + nz1) // 2, (ny - ny1) // 2:(ny + ny1) // 2, (nx - nx1) // 2:(nx + nx1) // 2]
mask = mask[(nz - nz1) // 2:(nz + nz1) // 2, (ny - ny1) // 2:(ny + ny1) // 2, (nx - nx1) // 2:(nx + nx1) // 2]
nz, ny, nx = data.shape
nz1, ny1, nx1 = smaller_primes((nz, ny, nx), maxprime=7, required_dividers=(2,))
data = data[(nz - nz1) // 2:(nz + nz1) // 2, (ny - ny1) // 2:(ny + ny1) // 2, (nx - nx1) // 2:(nx + nx1) // 2]
mask = mask[(nz - nz1) // 2:(nz + nz1) // 2, (ny - ny1) // 2:(ny + ny1) // 2, (nx - nx1) // 2:(nx + nx1) // 2]
nz, ny, nx = data.shape
print("cropped FFT data size:", data.shape)
print("Total number of photons:", data.sum())
comment = comment + "_crop_" + str(nz) + "," + str(ny) + "," + str(nx)
###########################################
# save files
###########################################
if save_data == 1:
    np.savez_compressed(datadir + 'S' + str(scan) + '_diff_' + str('{:.0e}'.format(photon_number))+comment, data=data)
    np.savez_compressed(datadir + 'S' + str(scan) + '_mask_' + str('{:.0e}'.format(photon_number))+comment, mask=mask)

#############################################
# plot things
#############################################
plt.ioff()
if debug == 1:
    plt.figure(figsize=(18, 15))
    plt.subplot(2, 2, 1)
    plt.imshow(abs(mask.sum(axis=2)), cmap=my_cmap)
    plt.colorbar()
    plt.xlabel(r"Q$_z$ ($1/\AA$)")
    plt.ylabel(r"Q$_x$ ($1/\AA$)")
    plt.axis('scaled')
    plt.title('mask summed over Qy')
    plt.subplot(2, 2, 2)
    plt.imshow(abs(mask.sum(axis=1)), cmap=my_cmap)
    plt.xlabel(r"Q$_y$ ($1/\AA$)")
    plt.ylabel(r"Q$_x$ ($1/\AA$)")
    plt.colorbar()
    plt.axis('scaled')
    plt.title('mask summed over Qz')
    plt.subplot(2, 2, 3)
    plt.imshow(abs(mask.sum(axis=0)), cmap=my_cmap)
    plt.xlabel(r"Q$_y$ ($1/\AA$)")
    plt.ylabel(r"Q$_z$ ($1/\AA$)")
    plt.colorbar()
    plt.axis('scaled')
    plt.title('mask summed over Qx')

# data = data + 0.00001
fig = plt.figure(figsize=(18, 15))
plt.subplot(2, 2, 1)
plt.imshow(np.log10(abs(data[:, :, nx // 2])), cmap=my_cmap, vmin=-1)
plt.colorbar()
plt.xlabel(r"Q$_z$ ($1/\AA$)")
plt.ylabel(r"Q$_x$ ($1/\AA$)")
plt.axis('scaled')
plt.title('masked intensity at middle frame in Qy')
plt.subplot(2, 2, 2)
plt.imshow(np.log10(abs(data[:, ny // 2, :])), cmap=my_cmap, vmin=-1)
plt.xlabel(r"Q$_y$ ($1/\AA$)")
plt.ylabel(r"Q$_x$ ($1/\AA$)")
plt.colorbar()
plt.axis('scaled')
plt.title('masked intensity at middle frame in Qz')
plt.subplot(2, 2, 3)
plt.imshow(np.log10(abs(data[nz // 2, :, :])), cmap=my_cmap, vmin=-1)
plt.xlabel(r"Q$_y$ ($1/\AA$)")
plt.ylabel(r"Q$_z$ ($1/\AA$)")
plt.colorbar()
plt.axis('scaled')
plt.title('masked intensity at middle frame in Qx')
fig.text(0.60, 0.30, "Pad size =" + str(pad_size), size=20)
fig.text(0.60, 0.25, "Crop size =" + str(crop_size), size=20)
# fig.text(0.60, 0.20, "New tilt angle =" + str('{:.4f}'.format(tilt_crop)) + "deg", size=20)
# fig.text(0.60, 0.15, "New detector pixel size y =" + str('{:.2f}'.format(pixel_crop_y * 1e6)) + "um", size=20)
# fig.text(0.60, 0.10, "New detector pixel size x =" + str('{:.2f}'.format(pixel_crop_x * 1e6)) + "um", size=20)
if save_fig == 1:
    plt.savefig(datadir + 'S' + str(scan) + '_diff_' + str('{:.0e}'.format(photon_number))+comment + '_center.png')

fig = plt.figure(figsize=(18, 15))
plt.subplot(2, 2, 1)
plt.imshow(np.log10(abs(data.sum(axis=2))), cmap=my_cmap, vmin=-1)
plt.colorbar()
plt.xlabel(r"Q$_z$ ($1/\AA$)")
plt.ylabel(r"Q$_x$ ($1/\AA$)")
plt.axis('scaled')
plt.title('masked intensity summed over Qy')
plt.subplot(2, 2, 2)
plt.imshow(np.log10(abs(data.sum(axis=1))), cmap=my_cmap, vmin=-1)
plt.xlabel(r"Q$_y$ ($1/\AA$)")
plt.ylabel(r"Q$_x$ ($1/\AA$)")
plt.colorbar()
plt.axis('scaled')
plt.title('masked intensity summed over Qz')
plt.subplot(2, 2, 3)
plt.imshow(np.log10(abs(data.sum(axis=0))), cmap=my_cmap, vmin=-1)
plt.xlabel(r"Q$_y$ ($1/\AA$)")
plt.ylabel(r"Q$_z$ ($1/\AA$)")
plt.colorbar()
plt.axis('scaled')
plt.title('masked intensity summed over Qx')
fig.text(0.60, 0.30, "Pad size =" + str(pad_size), size=20)
fig.text(0.60, 0.25, "Crop size =" + str(crop_size), size=20)
if set_gap == 1:
    fig.text(0.60, 0.20, "Gap width =" + str(gap_width) + "pixels", size=20)
# fig.text(0.60, 0.20, "New tilt angle =" + str('{:.4f}'.format(tilt_crop)) + "deg", size=20)
# fig.text(0.60, 0.15, "New detector pixel size y =" + str('{:.2f}'.format(pixel_crop_y * 1e6)) + "um", size=20)
# fig.text(0.60, 0.10, "New detector pixel size x =" + str('{:.2f}'.format(pixel_crop_x * 1e6)) + "um", size=20)
if save_fig == 1:
    plt.savefig(datadir + 'S' + str(scan) + '_diff_' + str('{:.0e}'.format(photon_number))+comment + '_sum.png')
plt.show()
