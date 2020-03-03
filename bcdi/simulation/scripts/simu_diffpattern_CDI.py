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
import bcdi.experiment.experiment_utils as exp
import bcdi.postprocessing.postprocessing_utils as pu
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
unitcell_param = 22.4  # in nm, unit cell parameter
######################
# sample orientation #
######################
angles = [0, 30, 0]  # in degrees, rotation around z downstream, y vertical up and x outboard respectively
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
roi_detector = []  # [direct_beam[0] - 900, direct_beam[0] + 900, direct_beam[1] - 900, direct_beam[1] + 900]
# [Vstart, Vstop, Hstart, Hstop]
binning = [4, 4, 4]  # binning of the detector
###########
# options #
###########
kernel_length = 21  # width of the 3D gaussian window
debug = True  # True to see more plots
##################################
# end of user-defined parameters #
##################################

#######################
# Initialize detector #
#######################
detector = exp.Detector(name=detector, binning=binning, roi=roi_detector)

nbz, nby, nbx = int(np.floor((detector.roi[3] - detector.roi[2]) / detector.binning[2])), \
                   int(np.floor((detector.roi[1] - detector.roi[0]) / detector.binning[1])), \
                   int(np.floor((detector.roi[3] - detector.roi[2]) / detector.binning[2]))
# for P10 data the rotation is around y vertical, hence gridded data range & binning in z and x are identical

###################
# define colormap #
###################
bad_color = '1.0'  # white background
colormap = gu.Colormap(bad_color=bad_color)
my_cmap = colormap.cmap
plt.ion()

######################
# create the lattice #
######################
lattice, peaks = simu.lattice(energy=energy, sdd=sdd, direct_beam=direct_beam, detector=detector, unitcell=unitcell,
                              unitcell_param=unitcell_param, euler_angles=angles)
# peaks in the format [[h, l, k], ...]: CXI convention downstream , vertical up, outboard
for idx in range(len(peaks)):
    print('Miller indices:', peaks[idx], '    at pixels:', lattice[idx])
struct_array = np.zeros((nbz, nby, nbx))
for [piz, piy, pix] in lattice:
    struct_array[piz, piy, pix] = 1

##############################################
# convolute the lattice with a 3D peak shape #
##############################################
# since we have a small list of peaks, do not use convolution (too slow) but for loop
peak_shape = pu.gaussian_kernel(ndim=3, kernel_length=kernel_length, sigma=3, debugging=False)

for [piz, piy, pix] in lattice:
    startz = max(0, int(piz-kernel_length//2))
    stopz = min(nbz-1, int(piz+kernel_length//2))
    starty = max(0, int(piy-kernel_length//2))
    stopy = min(nby-1, int(piy+kernel_length//2))
    startx = max(0, int(pix-kernel_length//2))
    stopx = min(nbx-1, int(pix+kernel_length//2))
    struct_array[startz:stopz+1, starty:stopy+1, startx:stopx+1] = peak_shape

###############
# plot result #
###############
# direct beam position after binning
center_z = int((direct_beam[1] - detector.roi[2]) / detector.binning[2])  # horizontal downstream
# same orientation as detector X rotated by 90 deg at P10, along z (or qx)
directbeam_y = nby - int((direct_beam[0] - detector.roi[0]) / detector.binning[1])  # vertical
# detector Y along vertical down, opposite to y (and qz)
directbeam_x = nbx - int((direct_beam[1] - detector.roi[2]) / detector.binning[2])  # horizontal
# detector X inboard, opposite to x (and qy)

fig, _, _ = gu.multislices_plot(struct_array, sum_frames=False, title='Simulated diffraction pattern', vmin=0,
                                slice_position=[center_z, directbeam_y, directbeam_x], plot_colorbar=True,
                                cmap=my_cmap, is_orthogonal=True, reciprocal_space=True)
fig.text(0.60, 0.20, "Direct beam (Y,X) =" + str(directbeam_y) + "," + str(directbeam_x), size=20)
plt.pause(0.1)

fig, _, _ = gu.multislices_plot(struct_array, sum_frames=True, title='Simulated diffraction pattern', vmin=0,
                                plot_colorbar=True, cmap=my_cmap, is_orthogonal=True, reciprocal_space=True)
fig.text(0.60, 0.20, "Direct beam (Y,X) =" + str(directbeam_y) + "," + str(directbeam_x), size=20)
plt.pause(0.1)

plt.ioff()
plt.show()

