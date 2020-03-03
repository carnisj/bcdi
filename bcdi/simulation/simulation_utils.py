# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np
import sys
sys.path.append('C:/Users/Jerome/Documents/myscripts/bcdi/')
import bcdi.utils.utilities as util


def lattice(energy, sdd, direct_beam, detector, unitcell, unitcell_param):
    """


    :param energy:
    :param sdd:
    :param direct_beam:
    :param detector:
    :param unitcell:
    :param unitcell_param:
    :return:
    """
    pixel_x = detector.pixelsize_x * 1e9  # convert to nm, pixel size in the horizontal direction
    pixel_y = detector.pixelsize_y * 1e9  # convert to nm, pixel size in the vertical direction
    directbeam_y = int((direct_beam[0] - detector.roi[0]) / detector.binning[1])  # vertical
    directbeam_x = int((direct_beam[1] - detector.roi[2]) / detector.binning[2])  # horizontal
    roi = detector.roi  # before binning
    wavelength = 12.398 * 1e2 / energy  # in nm, energy in eV
    distance = sdd * 1e9  # convert to nm
    lambdaz = wavelength * distance
    numz, numy, numx = np.floor((roi[3] - roi[2]) / detector.binning[2]),\
        np.floor((roi[1] - roi[0]) / detector.binning[1]), \
        np.floor((roi[3] - roi[2]) / detector.binning[2])
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

    if unitcell == 'fcc':
        return fcc_lattice(q_values=(qx, qz, qy), unitcell_param=unitcell_param)


def fcc_lattice(q_values, unitcell_param):
    """


    :param q_values:
    :param unitcell_param:
    :return:
    """
    lattice_pos = []  # position of the pixels corresponding to hkl reflections
    peaks = []  # list of hkl fitting the data range
    print('fcc unit cell of parameter a =', unitcell_param, 'nm')
    recipr_param = 2*np.pi/unitcell_param  # reciprocal lattice is simple cubic of parameter 2*pi/unitcell_param
    print('reciprocal unit cell of parameter 2*pi/a =', recipr_param, '1/nm')
    qx = q_values[0]
    qz = q_values[1]
    qy = q_values[2]
    q_max = np.sqrt(abs(qx).max()**2+abs(qz).max()**2+abs(qy).max()**2)

    # calculate the maximum Miller indices which fit into q_max
    h_max = int(np.floor(q_max * unitcell_param / (2 * np.pi)))
    hkl = np.arange(start=-h_max, stop=h_max+1, step=1)
    for h in hkl:  # h downstream along qx
        for l in hkl:  # k outboard along qz
            for k in hkl:  # l vertical up along qy
                struct_factor = np.real(1 + np.exp(1j*np.pi*(h+k)) + np.exp(1j*np.pi*(h+l)) + np.exp(1j*np.pi*(k+l)))
                q_bragg = np.sqrt((h * recipr_param) ** 2 + (k * recipr_param) ** 2 + (l * recipr_param) ** 2)
                if (h == 0) and (k == 0) and (l == 0):
                    continue  # go to the next iteration of the loop, the code below is not evaluated
                if (struct_factor != 0) and (q_bragg < q_max):  # find the position of the pixel nearest to q_bragg
                    pix_h = util.find_nearest(original_array=qx, array_values=h * recipr_param)
                    pix_k = util.find_nearest(original_array=qy, array_values=k * recipr_param)
                    pix_l = util.find_nearest(original_array=qz, array_values=l * recipr_param)
                    lattice_pos.append([pix_h, pix_l, pix_k])  # CXI convention: downstream, vertical up, outboard
                    peaks.append([h, l, k])  # CXI convention: downstream, vertical up, outboard
    return lattice_pos, peaks
