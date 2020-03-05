# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np
from scipy.spatial.transform import Rotation
import sys
sys.path.append('C:/Users/Jerome/Documents/myscripts/bcdi/')
import bcdi.utils.utilities as util


def assign_peakshape(array_shape, lattice_list, peak_shape, pivot):
    """
    Assign the 3D peak_shape to lattice points.

    :param array_shape: shape of the output array
    :param lattice_list: list of points in pixels [[z1,y1,x1],[z2,y2,x2],...]
    :param peak_shape: the 3D kernel to apply at each lattice point
    :param pivot: position of the center of reciprocal space in pixels
    :return: a 3D array featuring the peak shape at each lattice point
    """
    array = np.zeros(array_shape)
    kernel_length = peak_shape.shape[0]
    # since we have a small list of peaks, do not use convolution (too slow) but for loop
    # 1 is related to indices for array, 2 is related to indices for peak_shape
    for [piz, piy, pix] in lattice_list:
        startz1, startz2 = max(0, int(piz - kernel_length // 2)), -min(0, int(piz - kernel_length // 2))
        stopz1, stopz2 = min(array_shape[0] - 1, int(piz + kernel_length // 2)), \
            kernel_length + min(0, int(array_shape[0] - 1 - (piz + kernel_length // 2)))
        starty1, starty2 = max(0, int(piy - kernel_length // 2)), -min(0, int(piy - kernel_length // 2))
        stopy1, stopy2 = min(array_shape[1] - 1, int(piy + kernel_length // 2)), \
            kernel_length + min(0, int(array_shape[1] - 1 - (piy + kernel_length // 2)))
        startx1, startx2 = max(0, int(pix - kernel_length // 2)), -min(0, int(pix - kernel_length // 2))
        stopx1, stopx2 = min(array_shape[2] - 1, int(pix + kernel_length // 2)), \
            kernel_length + min(0, int(array_shape[2] - 1 - (pix + kernel_length // 2)))
        array[startz1:stopz1 + 1, starty1:stopy1 + 1, startx1:stopx1 + 1] = \
            peak_shape[startz2:stopz2, starty2:stopy2, startx2:stopx2]

    # mask the region near the origin of the reciprocal space
    array[pivot[0] - kernel_length // 2:pivot[0] + kernel_length // 2 + 1,
          pivot[1] - kernel_length // 2:pivot[1] + kernel_length // 2 + 1,
          pivot[2] - kernel_length // 2:pivot[2] + kernel_length // 2 + 1] = 0

    return array


def bcc_lattice(q_values, unitcell_param, pivot, euler_angles=(0, 0, 0), verbose=False):
    """
    Calculate Bragg peaks positions using experimental parameters for a BCC unit cell.

    :param q_values: tuple of 1D arrays (qx, qz, qy), q_values range where to look for Bragg peaks
    :param unitcell_param: the unit cell parameter of the FCC lattice
    :param pivot:  tuple, the pivot point position in pixels for the rotation
    :param euler_angles: tuple of angles for rotating the unit cell around (qx, qz, qy)
    :param verbose: True to have printed comments
    :return: offsets after padding, the list of Bragg peaks positions in pixels, and the corresponding list of hlk.
    """
    lattice_list = []  # position of the pixels corresponding to hkl reflections
    peaks_list = []  # list of hkl fitting the data range

    recipr_param = 2*np.pi/unitcell_param  # reciprocal lattice is simple cubic of parameter 2*pi/unitcell_param
    if verbose:
        print('fcc unit cell of parameter a =', unitcell_param, 'nm')
        print('reciprocal unit cell of parameter 2*pi/a =', str('{:.4f}'.format(recipr_param)), '1/nm')

    qx = q_values[0]  # along z downstream in CXI convention
    qz = q_values[1]  # along y vertical up in CXI convention
    qy = q_values[2]  # along x outboard in CXI convention
    q_max = np.sqrt(abs(qx).max()**2+abs(qz).max()**2+abs(qy).max()**2)
    numz, numy, numx = len(qx), len(qz), len(qy)

    # calculate the maximum Miller indices which fit into q_max
    h_max = int(np.floor(q_max * unitcell_param / (2 * np.pi)))
    hkl = np.arange(start=-h_max, stop=h_max+1, step=1)

    # pad q arrays in order to find the position in pixels of each hkl within the array
    # otherwise it finds the first or last index but this can be far from the real peak position
    leftpad_z, leftpad_y, leftpad_x = numz, numy, numx  # offset of indices to the left
    pad_qx = qx[0] - leftpad_z * (qx[1] - qx[0]) + np.arange(3*numz) * (qx[1] - qx[0])
    pad_qz = qz[0] - leftpad_y * (qz[1] - qz[0]) + np.arange(3*numy) * (qz[1] - qz[0])
    pad_qy = qy[0] - leftpad_x * (qy[1] - qy[0]) + np.arange(3*numx) * (qy[1] - qy[0])

    for h in hkl:  # h downstream along qx
        for k in hkl:  # k outboard along qy
            for l in hkl:  # l vertical up along qz
                # simple cubic unit cell with two point basis (0,0,0), (0.5,0.5,0.5)
                struct_factor = np.real(1 + np.exp(1j*np.pi*(h+k+l)))
                if struct_factor != 0:  # find the position of the pixel nearest to q_bragg
                    pix_h = util.find_nearest(original_array=pad_qx, array_values=h * recipr_param)
                    pix_k = util.find_nearest(original_array=pad_qy, array_values=k * recipr_param)
                    pix_l = util.find_nearest(original_array=pad_qz, array_values=l * recipr_param)

                    lattice_list.append([pix_h, pix_l, pix_k])
                    peaks_list.append([h, l, k])

    lattice_pos, peaks = rotate_lattice(lattice_list=lattice_list, peaks_list=peaks_list,
                                        original_shape=(numz, numy, numx),
                                        pad_offset=(leftpad_z, leftpad_y, leftpad_x),
                                        pivot=pivot, euler_angles=euler_angles)

    return (leftpad_z, leftpad_y, leftpad_x), lattice_pos, peaks


def cubic_lattice(q_values, unitcell_param, pivot, euler_angles=(0, 0, 0), verbose=False):
    """
    Calculate Bragg peaks positions using experimental parameters for a simple cubic unit cell.

    :param q_values: tuple of 1D arrays (qx, qz, qy), q_values range where to look for Bragg peaks
    :param unitcell_param: the unit cell parameter of the FCC lattice
    :param pivot:  tuple, the pivot point position in pixels for the rotation
    :param euler_angles: tuple of angles for rotating the unit cell around (qx, qz, qy)
    :param verbose: True to have printed comments
    :return: offsets after padding, the list of Bragg peaks positions in pixels, and the corresponding list of hlk.
    """
    lattice_list = []  # position of the pixels corresponding to hkl reflections
    peaks_list = []  # list of hkl fitting the data range

    recipr_param = 2*np.pi/unitcell_param  # reciprocal lattice is simple cubic of parameter 2*pi/unitcell_param
    if verbose:
        print('fcc unit cell of parameter a =', unitcell_param, 'nm')
        print('reciprocal unit cell of parameter 2*pi/a =', str('{:.4f}'.format(recipr_param)), '1/nm')

    qx = q_values[0]  # along z downstream in CXI convention
    qz = q_values[1]  # along y vertical up in CXI convention
    qy = q_values[2]  # along x outboard in CXI convention
    q_max = np.sqrt(abs(qx).max()**2+abs(qz).max()**2+abs(qy).max()**2)
    numz, numy, numx = len(qx), len(qz), len(qy)

    # calculate the maximum Miller indices which fit into q_max
    h_max = int(np.floor(q_max * unitcell_param / (2 * np.pi)))
    hkl = np.arange(start=-h_max, stop=h_max+1, step=1)

    # pad q arrays in order to find the position in pixels of each hkl within the array
    # otherwise it finds the first or last index but this can be far from the real peak position
    leftpad_z, leftpad_y, leftpad_x = numz, numy, numx  # offset of indices to the left
    pad_qx = qx[0] - leftpad_z * (qx[1] - qx[0]) + np.arange(3*numz) * (qx[1] - qx[0])
    pad_qz = qz[0] - leftpad_y * (qz[1] - qz[0]) + np.arange(3*numy) * (qz[1] - qz[0])
    pad_qy = qy[0] - leftpad_x * (qy[1] - qy[0]) + np.arange(3*numx) * (qy[1] - qy[0])

    for h in hkl:  # h downstream along qx
        for k in hkl:  # k outboard along qy
            for l in hkl:  # l vertical up along qz
                # one atom basis (0,0,0): struct_factor = 1, all peaks are allowed
                pix_h = util.find_nearest(original_array=pad_qx, array_values=h * recipr_param)
                pix_k = util.find_nearest(original_array=pad_qy, array_values=k * recipr_param)
                pix_l = util.find_nearest(original_array=pad_qz, array_values=l * recipr_param)

                lattice_list.append([pix_h, pix_l, pix_k])
                peaks_list.append([h, l, k])

    lattice_pos, peaks = rotate_lattice(lattice_list=lattice_list, peaks_list=peaks_list,
                                        original_shape=(numz, numy, numx),
                                        pad_offset=(leftpad_z, leftpad_y, leftpad_x),
                                        pivot=pivot, euler_angles=euler_angles)

    return (leftpad_z, leftpad_y, leftpad_x), lattice_pos, peaks


def fcc_lattice(q_values, unitcell_param, pivot, euler_angles=(0, 0, 0), verbose=False):
    """
    Calculate Bragg peaks positions using experimental parameters for a FCC unit cell.

    :param q_values: tuple of 1D arrays (qx, qz, qy), q_values range where to look for Bragg peaks
    :param unitcell_param: the unit cell parameter of the FCC lattice
    :param pivot:  tuple, the pivot point position in pixels for the rotation
    :param euler_angles: tuple of angles for rotating the unit cell around (qx, qz, qy)
    :param verbose: True to have printed comments
    :return: offsets after padding, the list of Bragg peaks positions in pixels, and the corresponding list of hlk.
    """
    lattice_list = []  # position of the pixels corresponding to hkl reflections
    peaks_list = []  # list of hkl fitting the data range

    recipr_param = 2*np.pi/unitcell_param  # reciprocal lattice is simple cubic of parameter 2*pi/unitcell_param
    if verbose:
        print('fcc unit cell of parameter a =', unitcell_param, 'nm')
        print('reciprocal unit cell of parameter 2*pi/a =', str('{:.4f}'.format(recipr_param)), '1/nm')

    qx = q_values[0]  # along z downstream in CXI convention
    qz = q_values[1]  # along y vertical up in CXI convention
    qy = q_values[2]  # along x outboard in CXI convention
    q_max = np.sqrt(abs(qx).max()**2+abs(qz).max()**2+abs(qy).max()**2)
    numz, numy, numx = len(qx), len(qz), len(qy)

    # calculate the maximum Miller indices which fit into q_max
    h_max = int(np.floor(q_max * unitcell_param / (2 * np.pi)))
    hkl = np.arange(start=-h_max, stop=h_max+1, step=1)

    # pad q arrays in order to find the position in pixels of each hkl within the array
    # otherwise it finds the first or last index but this can be far from the real peak position
    leftpad_z, leftpad_y, leftpad_x = numz, numy, numx  # offset of indices to the left
    pad_qx = qx[0] - leftpad_z * (qx[1] - qx[0]) + np.arange(3*numz) * (qx[1] - qx[0])
    pad_qz = qz[0] - leftpad_y * (qz[1] - qz[0]) + np.arange(3*numy) * (qz[1] - qz[0])
    pad_qy = qy[0] - leftpad_x * (qy[1] - qy[0]) + np.arange(3*numx) * (qy[1] - qy[0])

    for h in hkl:  # h downstream along qx
        for k in hkl:  # k outboard along qy
            for l in hkl:  # l vertical up along qz
                # simple cubic unit cell with four point basis (0,0,0), (0.5,0.5,0), (0,0.5,0.5), (0.5,0,0.5)
                struct_factor = np.real(1 + np.exp(1j*np.pi*(h+k)) + np.exp(1j*np.pi*(h+l)) + np.exp(1j*np.pi*(k+l)))
                if struct_factor != 0:  # find the position of the pixel nearest to q_bragg
                    pix_h = util.find_nearest(original_array=pad_qx, array_values=h * recipr_param)
                    pix_k = util.find_nearest(original_array=pad_qy, array_values=k * recipr_param)
                    pix_l = util.find_nearest(original_array=pad_qz, array_values=l * recipr_param)

                    lattice_list.append([pix_h, pix_l, pix_k])
                    peaks_list.append([h, l, k])

    lattice_pos, peaks = rotate_lattice(lattice_list=lattice_list, peaks_list=peaks_list,
                                        original_shape=(numz, numy, numx),
                                        pad_offset=(leftpad_z, leftpad_y, leftpad_x),
                                        pivot=pivot, euler_angles=euler_angles)

    return (leftpad_z, leftpad_y, leftpad_x), lattice_pos, peaks


def lattice(energy, sdd, direct_beam, detector, unitcell, unitcell_param, euler_angles=(0, 0, 0)):
    """
    Calculate Bragg peaks positions using experimental parameters and unit cell.

    :param energy: X-ray energy in eV
    :param sdd: sample to detector distance in m
    :param direct_beam: direct beam position on the detector in pixels (vertical, horizontal)
    :param detector: the detector object: Class experiment_utils.Detector()
    :param unitcell: 'cubic', 'bcc' or 'fcc'
    :param unitcell_param: number or tuple for unit cell parameters
    :param euler_angles: tuple of angles for rotating the unit cell around (qx, qz, qy)
    :return: pivot position, q values, a list of pixels positions for each Bragg peak, Miller indices.
    """
    pixel_x = detector.pixelsize_x * 1e9  # convert to nm, pixel size in the horizontal direction
    pixel_y = detector.pixelsize_y * 1e9  # convert to nm, pixel size in the vertical direction

    # position of the direct beam in the detector frame
    directbeam_y = int((direct_beam[0] - detector.roi[0]) / detector.binning[1])  # detector Y vertical down
    directbeam_x = int((direct_beam[1] - detector.roi[2]) / detector.binning[2])  # horizontal X inboard

    wavelength = 12.398 * 1e2 / energy  # in nm, energy in eV
    distance = sdd * 1e9  # convert to nm
    lambdaz = wavelength * distance
    numz, numy, numx = np.floor((detector.roi[3] - detector.roi[2]) / detector.binning[2]),\
        np.floor((detector.roi[1] - detector.roi[0]) / detector.binning[1]), \
        np.floor((detector.roi[3] - detector.roi[2]) / detector.binning[2])
    # for P10 data the rotation is around y vertical, hence gridded data range & binning in z and x are identical

    ######################
    # calculate q values #
    ######################
    # calculate q spacing and q values using above voxel sizes
    dqx = 2 * np.pi / lambdaz * pixel_x  # in 1/nm, downstream
    dqz = 2 * np.pi / lambdaz * pixel_y  # in 1/nm, vertical up
    dqy = 2 * np.pi / lambdaz * pixel_x  # in 1/nm, outboard

    # calculation of q based on P10 geometry
    qx = np.arange(-directbeam_x, -directbeam_x + numz, 1) * dqx
    # downstream, same direction as detector X rotated by +90deg
    qz = np.arange(-(numy - directbeam_y), -(numy - directbeam_y) + numy, 1) * dqz  # vertical up opposite to detector Y
    qy = np.arange(-(numx - directbeam_x), -(numx - directbeam_x) + numx, 1) * dqy  # outboard opposite to detector X

    # calculate the position of the pivot point for the rotation in the laboratory frame
    pivot_z = int((direct_beam[1] - detector.roi[2]) / detector.binning[2])
    # 90 degrees conter-clockwise rotation of detector X around qz, downstream
    pivot_y = int(numy - directbeam_y)  # detector Y vertical down, opposite to qz vertical up
    pivot_x = int(numx - directbeam_x)  # detector X inboard at P10, opposite to qy outboard
    if unitcell == 'fcc':
        offset, lattice_pos, peaks = fcc_lattice(q_values=(qx, qz, qy), unitcell_param=unitcell_param,
                                                 pivot=(pivot_z, pivot_y, pivot_x), euler_angles=euler_angles)
    elif unitcell == 'bcc':
        offset, lattice_pos, peaks = bcc_lattice(q_values=(qx, qz, qy), unitcell_param=unitcell_param,
                                                 pivot=(pivot_z, pivot_y, pivot_x), euler_angles=euler_angles)
    elif unitcell == 'cubic':
        offset, lattice_pos, peaks = cubic_lattice(q_values=(qx, qz, qy), unitcell_param=unitcell_param,
                                                   pivot=(pivot_z, pivot_y, pivot_x), euler_angles=euler_angles)
    else:
        raise ValueError('Unit cell "' + unitcell + '" not yet implemented')

    return (pivot_z, pivot_y, pivot_x), offset, (qx, qz, qy), lattice_pos, peaks


def rotate_lattice(lattice_list, peaks_list, original_shape, pad_offset, pivot, euler_angles=(0, 0, 0)):
    """
    Rotate lattice points given Euler angles, the pivot position and an eventual offset of the origin.

    :param lattice_list: list of Bragg peaks positions in pixels to be rotated [[z1,y1,x1],[z2,y2,x2],...]
    :param peaks_list: corresponding list of [[h1,l1,k1],[h2,l2,k2]...]
    :param original_shape: shape of q values before padding
    :param pad_offset: index shift of the origin for the padded q values
    :param pivot:  tuple, the pivot point position in pixels for the rotation
    :param euler_angles: tuple of angles for rotating the unit cell around (qx, qz, qy)
    :return: list of Bragg peaks positions fitting into the range, and the corresponding list of hlk
    """
    lattice_pos = []  # position of the pixels corresponding to hkl reflections
    peaks = []  # list of hkl fitting the data range
    numz, numy, numx = original_shape
    pivot_z, pivot_y, pivot_x = pivot  # downstream, vertical up, outboard
    leftpad_z, leftpad_y, leftpad_x = pad_offset  # offset of the 0 index in padded q values: see fcc_lattice()

    # define the rotation using Euler angles with the direct beam as origin
    rotation = Rotation.from_euler('xzy', euler_angles, degrees=True)

    for idx, point in enumerate(lattice_list):
        pix_h, pix_l, pix_k = point
        # rotate the vector using Euler angles and the pivot point while compensating padding
        offset_h, offset_l, offset_k = pix_h-(pivot_z+leftpad_z), pix_l-(pivot_y+leftpad_y), \
            pix_k-(pivot_x+leftpad_x)

        rot_h, rot_k, rot_l = rotation.apply([offset_h, offset_k, offset_l])
        # coordinates order for Rotation(): [qx, qy, qz]

        # shift back the origin to (0, 0, 0)
        rot_h, rot_l, rot_k = np.rint(rot_h+pivot_z+leftpad_z).astype(int),\
            np.rint(rot_l+pivot_y+leftpad_y).astype(int),\
            np.rint(rot_k+pivot_x+leftpad_x).astype(int)

        # calculate indices in the original q values coordinates before padding
        rot_h, rot_l, rot_k = rot_h - leftpad_z, rot_l - leftpad_y, rot_k - leftpad_x

        # check if the rotated peak is in the non-padded data range
        if (0 <= rot_h < numz) and (0 <= rot_l < numy) and (0 <= rot_k < numx):
            # use here CXI convention: downstream, vertical up, outboard
            lattice_pos.append([rot_h, rot_l, rot_k])
            peaks.append(peaks_list[idx])

    return lattice_pos, peaks
