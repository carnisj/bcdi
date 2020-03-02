# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np
from numpy.random import poisson
from numpy.fft import fftn, fftshift
from matplotlib import pyplot as plt
from scipy.signal import convolve
import tkinter as tk
from tkinter import filedialog
from scipy.interpolate import RegularGridInterpolator
import gc
import os
import sys
sys.path.append('C:/Users/Jerome/Documents/myscripts/bcdi/')
import bcdi.graph.graph_utils as gu
import bcdi.experiment.experiment_utils as exp
import bcdi.postprocessing.postprocessing_utils as pu
import bcdi.preprocessing.preprocessing_utils as pru
import bcdi.simulation.simulation_utils as simu

helptext = """
Calculate the position of the Bragg peaks for a mesocrystal given the lattice type, the unit cell parameter
and beamline-related parameters.

Laboratory frame convention (CXI): z downstream, y vertical up, x outboard."""

savedir = "D:/data/P10_August2019/data/gold2_2_00515/simu/"
################
# sample setup #
################
unitcell = 'fcc'
unitcell_param = 23  # in nm, unit cell parameter
######################
# sample orientation #
######################
angles = [0, 0, 0]  # angle for the rotation around z, y and x respectively
#######################
# beamline parameters #
#######################
sdd = 4.95  # in m, sample to detector distance
energy = 8700  # in ev X-ray energy
##################
# detector setup #
##################
detector = "Eiger4M"  # "Eiger2M" or "Maxipix" or "Eiger4M"
direct_beam = (1349, 1321)  # tuple of int (vertical, horizontal): position of the direct beam in pixels
# this parameter is important for gridding the data onto the laboratory frame
roi_detector = [direct_beam[0] - 500, direct_beam[0] + 500, direct_beam[1] - 500, direct_beam[1] + 500]
# [Vstart, Vstop, Hstart, Hstop]
binning = [2, 2, 2]  # binning of the detector
###########
# options #
###########
debug = True  # True to see more plots
##################################
# end of user-defined parameters #
##################################

#######################
# Initialize detector #
#######################
detector = exp.Detector(name=detector, binning=binning, roi=roi_detector)

###################
# define colormap #
###################
bad_color = '1.0'  # white background
colormap = gu.Colormap(bad_color=bad_color)
my_cmap = colormap.cmap

#########################################
# create the lattice and the peak shape #
#########################################
lattice = simu.lattice(energy=energy, sdd=sdd, direct_beam=direct_beam, detector=detector, unitcell=unitcell,
                       unitcell_param=unitcell_param)
peak_shape = pu.gaussian_kernel(ndim=3, kernel_length=21, sigma=3, debugging=True)

######################
# rotate the lattice #
######################


##############################################
# convolute the lattice with a 3D peak shape #
##############################################
diffpattern = convolve(lattice, peak_shape, mode='same')
gu.multislices_plot(diffpattern, sum_frames=False, title='Simulated diffraction pattern', vmin=0, plot_colorbar=True,
                    cmap=my_cmap, is_orthogonal=True, reciprocal_space=True)

plt.show()
