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
unitcell = 'cubic'
unitcell_param = 22.4  # in nm, unit cell parameter
######################
# sample orientation #
######################
angles = [-1, 18, 0]  # in degrees, rotation around qx downstream, qz vertical up and qy outboard respectively
#######################
# beamline parameters #
#######################
sdd = 4.95  # in m, sample to detector distance
energy = 8700  # in ev X-ray energy
##################
# detector setup #
##################
detector = "Eiger4M"  # "Eiger2M" or "Maxipix" or "Eiger4M"
direct_beam = (1195, 1187)  # tuple of int (vertical, horizontal): position of the direct beam in pixels
# this parameter is important for gridding the data onto the laboratory frame
roi_detector = [direct_beam[0] - 972, direct_beam[0] + 972, direct_beam[1] - 883, direct_beam[1] + 883]
# [Vstart, Vstop, Hstart, Hstop]
binning = [4, 4, 4]  # binning of the detector
###########
# options #
###########
kernel_length = 21  # width of the 3D gaussian window
debug = False  # True to see more plots
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
pivot, _, q_values, lattice, peaks = simu.lattice(energy=energy, sdd=sdd, direct_beam=direct_beam,
                                                       detector=detector, unitcell=unitcell,
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
maxpeak = peak_shape.max()

for [piz, piy, pix] in lattice:
    startz1, startz2 = max(0, int(piz-kernel_length//2)), -min(0, int(piz-kernel_length//2))
    stopz1, stopz2 = min(nbz-1, int(piz+kernel_length//2)), kernel_length + min(0, int(nbz-1 - (piz+kernel_length//2)))
    starty1, starty2 = max(0, int(piy-kernel_length//2)), -min(0, int(piy-kernel_length//2))
    stopy1, stopy2 = min(nby-1, int(piy+kernel_length//2)), kernel_length + min(0, int(nby-1 - (piy+kernel_length//2)))
    startx1, startx2 = max(0, int(pix-kernel_length//2)), -min(0, int(pix-kernel_length//2))
    stopx1, stopx2 = min(nbx-1, int(pix+kernel_length//2)), kernel_length + min(0, int(nbx-1 - (pix+kernel_length//2)))
    struct_array[startz1:stopz1+1, starty1:stopy1+1, startx1:stopx1+1] =\
        peak_shape[startz2:stopz2, starty2:stopy2, startx2:stopx2]

###############
# plot result #
###############
qx, qz, qy = q_values
# mark the direct beam position
struct_array[pivot[0]-kernel_length//2:pivot[0]+kernel_length//2+1,
             pivot[1]-kernel_length//2:pivot[1]+kernel_length//2+1,
             pivot[2]-kernel_length//2:pivot[2]+kernel_length//2+1] = 0
struct_array[pivot[0]-2:pivot[0]+3, pivot[1]-2:pivot[1]+3, pivot[2]-2:pivot[2]+3] = maxpeak

if debug:
    fig, _, _ = gu.multislices_plot(struct_array, sum_frames=False, title='Simulated diffraction pattern', vmin=0,
                                    vmax=maxpeak, slice_position=[pivot[0], pivot[1], pivot[2]],
                                    plot_colorbar=True, cmap=my_cmap, is_orthogonal=True, reciprocal_space=True)
    fig.text(0.60, 0.30, "Origin of reciprocal space (Qx,Qz,Qy) = " + str(pivot[0]) + "," + str(pivot[1]) + "," +
             str(pivot[2]), size=12)
    fig.text(0.60, 0.25, "Energy = " + str(energy/1000) + " keV", size=12)
    fig.text(0.60, 0.20, "SDD = " + str(sdd) + " m", size=12)
    fig.text(0.60, 0.15, unitcell + " unit cell of parameter = " + str(unitcell_param) + " nm", size=12)
    fig.text(0.60, 0.10, "Rotation of the unit cell in degrees (Qx, Qz, Qy) = " + str(angles[0]) + "," + str(angles[1])
             + "," + str(angles[2]), size=12)
    plt.pause(0.1)
    plt.savefig(savedir + 'central_slice_' + str(nbz) + '_' + str(nby) + '_' + str(nbx) + '_' + str(binning[0]) + '_' +
                str(binning[1]) + '_' + str(binning[2]) + '_rot_' + str(angles[0]) + '_' + str(angles[1]) + '_' +
                str(angles[2]) + '.png')

    fig, _, _ = gu.multislices_plot(struct_array, sum_frames=True, title='Simulated diffraction pattern', vmin=0,
                                    vmax=maxpeak, plot_colorbar=True, cmap=my_cmap, is_orthogonal=True,
                                    reciprocal_space=True)
    fig.text(0.60, 0.30, "Origin of reciprocal space  (Qx,Qz,Qy) = " + str(pivot[0]) + "," + str(pivot[1]) +
             "," + str(pivot[2]), size=12)
    fig.text(0.60, 0.25, "Energy = " + str(energy/1000) + " keV", size=12)
    fig.text(0.60, 0.20, "SDD = " + str(sdd) + " m", size=12)
    fig.text(0.60, 0.15, unitcell + " unit cell of parameter = " + str(unitcell_param) + " nm", size=12)
    fig.text(0.60, 0.10, "Rotation of the unit cell in degrees (Qx, Qz, Qy) = " + str(angles[0]) + "," + str(angles[1])
             + "," + str(angles[2]), size=12)
    plt.pause(0.1)
    plt.savefig(savedir + 'sum_' + str(nbz) + '_' + str(nby) + '_' + str(nbx) + '_' + str(binning[0]) + '_' +
                str(binning[1]) + '_' + str(binning[2]) + '_rot_' + str(angles[0]) + '_' + str(angles[1]) + '_' +
                str(angles[2]) + '.png')

fig, _, _ = gu.contour_slices(struct_array, (qx, qz, qy), sum_frames=False, title='Simulated diffraction pattern',
                              slice_position=[pivot[0], pivot[1], pivot[2]],
                              levels=np.linspace(0, struct_array.max(), 10, endpoint=False),
                              plot_colorbar=False, scale='linear', is_orthogonal=True, reciprocal_space=True)
fig.text(0.60, 0.25, "Energy = " + str(energy/1000) + " keV", size=12)
fig.text(0.60, 0.20, "SDD = " + str(sdd) + " m", size=12)
fig.text(0.60, 0.15, unitcell + " unit cell of parameter = " + str(unitcell_param) + " nm", size=12)
fig.text(0.60, 0.10, "Rotation of the unit cell in degrees (Qx, Qz, Qy) = " + str(angles[0]) + "," + str(angles[1])
         + "," + str(angles[2]), size=12)
plt.pause(0.1)
plt.savefig(savedir + 'q_central_slice_' + str(nbz) + '_' + str(nby) + '_' + str(nbx) + '_' + str(binning[0]) + '_' +
            str(binning[1]) + '_' + str(binning[2]) + '_rot_' + str(angles[0]) + '_' + str(angles[1]) + '_' +
            str(angles[2]) + '.png')

fig, _, _ = gu.contour_slices(struct_array, (qx, qz, qy), sum_frames=True, title='Simulated diffraction pattern',
                              levels=np.linspace(0, struct_array.max(), 10, endpoint=False),
                              plot_colorbar=False, scale='linear', is_orthogonal=True, reciprocal_space=True)
fig.text(0.60, 0.25, "Energy = " + str(energy / 1000) + " keV", size=12)
fig.text(0.60, 0.20, "SDD = " + str(sdd) + " m", size=12)
fig.text(0.60, 0.15, unitcell + " unit cell of parameter = " + str(unitcell_param) + " nm", size=12)
fig.text(0.60, 0.10, "Rotation of the unit cell in degrees (Qx, Qz, Qy) = " + str(angles[0]) + "," + str(angles[1])
         + "," + str(angles[2]), size=12)
plt.pause(0.1)
plt.savefig(
    savedir + 'q_sum_' + str(nbz) + '_' + str(nby) + '_' + str(nbx) + '_' + str(binning[0]) + '_' +
    str(binning[1]) + '_' + str(binning[2]) + '_rot_' + str(angles[0]) + '_' + str(angles[1]) + '_' +
    str(angles[2]) + '.png')

plt.ioff()
plt.show()
