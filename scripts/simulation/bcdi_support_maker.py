#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Steven Leake, leake@esrf.fr
"""example usage of supportMaker class."""

# cuboid
from bcdi.experiment.rotation_matrix import RotationMatrix as R
import scipy.fftpack as fft
import bcdi.simulation.supportMaker as sM
import numpy as np
import h5py as h5


shape = "cuboid"
if shape == "cuboid":
    planes, planesDist = sM.generatePlanesCuboid(800, 800, 100)

# tetrahedra
if shape == "tetrahedra":
    planes, planesDist = sM.generatePlanesTetrahedra(300)

# equilateral prism
if shape == "prism":
    planes, planesDist = sM.generatePlanesPrism(x, y)

# convert planes dist to nm
planesDist *= 1e-9

# define rotation of support
alpha, beta, gamma = 5, 45, 45
# define axis orientation and origin
origin, xaxis, yaxis, zaxis = [0, 0, 0], "x+", "y+", "z+"
Rx = R(xaxis, alpha).get_matrix()
Ry = R(yaxis, beta).get_matrix()
Rz = R(zaxis, gamma).get_matrix()

planes1 = sM.rot_planes(planes, Rz)
planes2 = sM.rot_planes(planes1, Ry)

# load an npz data array here
rawdata = np.zeros((64, 64, 64))

wavelength = 12.39842 / 10.2 * 1e-10
detector_distance = 2.9
detector_pixel_size = [10 * 55e-6, 3 * 55e-6]
ang_step = 0.004 * 2
braggAng = 9

supportMaker = sM.supportMaker(
    rawdata,
    wavelength,
    detector_distance,
    detector_pixel_size,
    ang_step,
    braggAng,
    planes2,
    planesDist,
    voxel_size=np.array([10, 10, 10]) * 1e-9,
)

support = supportMaker.get_support()

support[support >= 0.5] = 1
support[support < 0.5] = 0


# save to npz
np.savez("support.npz", support)

# save 2hdf5

with h5.File("support.h5", "a") as outf:
    outf["poly"] = support
    data1 = fft.fftn(support)
    # outf['poly_fft'] = abs(data1)
    outf["poly_fft_shift"] = abs(fft.fftshift(data1))
    # can be useufl to compare to rawdata
    outf["rawdata"] = rawdata


def makePoly_example():
    data = MakePoly((64, 64, 64), ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)))
    outf = h5.File("test.h5", "w")
    outf["poly"] = data

    data = MakePolyCen(
        (64, 64, 64), (10, 10, 10), ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))
    )
    data = MakePolyCen(
        (64, 64, 64),
        (10, 10, 10),
        ((10.1, 0, 0), (0, 10, 0), (0, 0, 10), (-10, 0, 0), (0, -10, 0), (0, 0, -10)),
    )
    outf["polycen"] = data
    data1 = fft.fftn(np.complex64(data))
    outf["poly_fft"] = abs(data1)
    outf["poly_fft_shift"] = abs(fft.fftshift(data1))
    outf.close()


makePoly_example()
