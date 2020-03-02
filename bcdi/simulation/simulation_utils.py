# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('C:/Users/Jerome/Documents/myscripts/bcdi/')
import bcdi.graph.graph_utils as gu


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
    numz, numy, numx = (roi[3] - roi[2]) / detector.binning[2], \
                       (roi[1] - roi[0]) / detector.binning[1], \
                       (roi[3] - roi[2]) / detector.binning[2]  # for P10 data were the rotation is around y vertical

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
    qx = q_values[0]
    qz = q_values[1]
    qy = q_values[2]
    numz, numy, numx = qx.shape, qz.shape, qy.shape,
    q_max = np.sqrt(abs(qx).max()**2+abs(qz).max()**2+abs(qy).max()**2)

    # calculate the maximum Miller indices which fit into q_max
    h_max = int(np.floor(q_max * unitcell_param / (2 * np.pi)))
    print(h_max)
