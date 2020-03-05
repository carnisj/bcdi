# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import filedialog
import gc
import sys
sys.path.append('C:/Users/Jerome/Documents/myscripts/bcdi/')
import bcdi.graph.graph_utils as gu
import bcdi.experiment.experiment_utils as exp
import bcdi.utils.utilities as util
import bcdi.postprocessing.postprocessing_utils as pu
import bcdi.simulation.simulation_utils as simu

helptext = """
Calculate the position of the Bragg peaks for a mesocrystal given the lattice type, the unit cell parameter
and beamline-related parameters. Assign 3D Gaussians to each lattice point and rotates the unit cell in order to
maximize the cross-correlation of the simulated data with experimental data.

Laboratory frame convention (CXI): z downstream, y vertical up, x outboard."""

datadir = "D:/data/P10_August2019/data/gold2_2_00515/pynx/441_486_441_1_4_4_masked/"
savedir = "D:/data/P10_August2019/data/gold2_2_00515/simu/"
################
# sample setup #
################
unitcell = 'fcc'
unitcell_param = 22.4  # in nm, unit cell parameter
#########################
# unit cell orientation #
#########################
angles_ranges = [-5, 5, 15, 25, -5, 5]  # in degrees, ranges to span for the rotation around qx downstream,
# qz vertical up and qy outboard respectively: [start, stop, start, stop, start, stop]    stop is excluded
angular_step = 2  # in degrees
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
create_background = True  # True to create a 3D background
load_background = False  # True to load an existing 3D background

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
plt.ion()

###################################
# load experimental data and mask #
###################################
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(initialdir=datadir, title="Select the data to fit",
                                       filetypes=[("NPZ", "*.npz")])
data = np.load(file_path)['data']

try:
    file_path = filedialog.askopenfilename(initialdir=datadir, title="Select the mask",
                                           filetypes=[("NPZ", "*.npz")])
    mask = np.load(file_path)['mask']

    data[np.nonzero(mask)] = 0
    del mask
    gc.collect()
except FileNotFoundError:
    pass

try:
    file_path = filedialog.askopenfilename(initialdir=datadir, title="Select q values",
                                           filetypes=[("NPZ", "*.npz")])
    exp_qvalues = np.load(file_path)
    qvalues_flag = True
except FileNotFoundError:
    exp_qvalues = None
    qvalues_flag = False
    pass

##################################
# create the non rotated lattice #
##################################
# simu.rotate_lattice() needs that the origin of indices corresponds to the length of padded q values
pivot, offset, q_values, ref_lattice, ref_peaks = simu.lattice(energy=energy, sdd=sdd, direct_beam=direct_beam,
                                                               detector=detector, unitcell=unitcell,
                                                               unitcell_param=unitcell_param, euler_angles=[0, 0, 0],
                                                               offset_indices=True)
nbz, nby, nbx = len(q_values[0]), len(q_values[1]), len(q_values[2])

################################################
# remove background from the experimental data #
################################################
if create_background:
    file_path = filedialog.askopenfilename(initialdir=datadir, title="Select the 1D background file",
                                           filetypes=[("NPZ", "*.npz")])
    avg_background = np.load(file_path)['background']
    distances = np.load(file_path)['distances']

    if qvalues_flag:
        background = util.create_3d_background(q_values=(exp_qvalues['qx'], exp_qvalues['qz'], exp_qvalues['qy']),
                                              avg_background=avg_background, avg_qvalues=distances)
    else:
        print('Using calculated q values for background subtraction')
        background = util.create_3d_background(q_values=q_values, avg_background=avg_background, avg_qvalues=distances)

    gu.multislices_plot(background, sum_frames=True, title='Background subtracted data', vmin=0,
                        vmax=np.log10(background).max(), scale='log', plot_colorbar=True, cmap=my_cmap,
                        is_orthogonal=True, reciprocal_space=True)

    np.savez_compressed(savedir+'3D_background_'+str(nbz)+'_'+str(nby)+'_'+str(nbx)+'.npz', background=background)
    data = data - background
    data[np.isnan(data)] = 0
    data[data < 0] = 0

elif load_background:
    file_path = filedialog.askopenfilename(initialdir=datadir, title="Select the 3D background file",
                                           filetypes=[("NPZ", "*.npz")])
    background = np.load(file_path)['background']
    data = data - background
    data[np.isnan(data)] = 0
    data[data < 0] = 0

#########################
# define the peak shape #
#########################
peak_shape = pu.gaussian_kernel(ndim=3, kernel_length=kernel_length, sigma=3, debugging=False)

#####################################
# define the list of angles to test #
#####################################
angles_qx = np.arange(start=angles_ranges[0], stop=angles_ranges[1], step=angular_step)
angles_qz = np.arange(start=angles_ranges[2], stop=angles_ranges[3], step=angular_step)
angles_qy = np.arange(start=angles_ranges[4], stop=angles_ranges[5], step=angular_step)
print('Number of angles to test: ', len(angles_qx)*len(angles_qz)*len(angles_qy))

#############################
# loop over rotation angles #
#############################
corr = np.zeros((len(angles_qx), len(angles_qz), len(angles_qy)))
for idz, alpha in enumerate(angles_qx):
    for idy, beta in enumerate(angles_qz):
        for idx, gamma in enumerate(angles_qy):
            rot_lattice, _ = simu.rotate_lattice(lattice_list=ref_lattice, peaks_list=ref_peaks,
                                                 original_shape=(nbz, nby, nbx), pad_offset=offset, pivot=pivot,
                                                 euler_angles=(alpha, beta, gamma))
            # peaks in the format [[h, l, k], ...]: CXI convention downstream , vertical up, outboard

            # assign the peak shape to each lattice point
            struct_array = simu.assign_peakshape(array_shape=(nbz, nby, nbx), lattice_list=rot_lattice,
                                                 peak_shape=peak_shape, pivot=pivot)

            # calculate the correlation between experimental data and simulated data
            corr[idz, idy, idx] = np.multiply(struct_array, data).sum()
            # print(alpha, beta, gamma, corr[idz, idy, idx])

###############
# plot result #
###############
vmin = corr.min()
vmax = corr.max()
piz, piy, pix = np.unravel_index(abs(corr).argmax(), corr.shape)
alpha, beta, gamma = angles_qx[piz], angles_qz[piy], angles_qy[pix]
print('Maximum correlation for (angle_qx, angle_qz, angle_qy) =', alpha, beta, gamma)

fig, _, _ = gu.contour_slices(corr, (angles_qx, angles_qz, angles_qy), sum_frames=False,
                              title='Correlation', slice_position=[piz, piy, pix], plot_colorbar=True, cmap=my_cmap,
                              levels=np.linspace(vmin, vmax, 10, endpoint=False), is_orthogonal=True,
                              reciprocal_space=True)
fig.text(0.60, 0.25, "Kernel size = " + str(kernel_length) + " pixels", size=12)
plt.pause(0.1)
plt.savefig(savedir + 'cross_corr.png')

################################################
# rotate the lattice at calculated best values #
################################################
rot_lattice, _ = simu.rotate_lattice(lattice_list=ref_lattice, peaks_list=ref_peaks, original_shape=(nbz, nby, nbx),
                                     pad_offset=offset, pivot=pivot, euler_angles=(alpha, beta, gamma))
# peaks in the format [[h, l, k], ...]: CXI convention downstream , vertical up, outboard

# assign the peak shape to each lattice point
struct_array = simu.assign_peakshape(array_shape=(nbz, nby, nbx), lattice_list=rot_lattice,
                                     peak_shape=peak_shape, pivot=pivot)
if qvalues_flag:
    # gu.contour_slices(data, (exp_qvalues['qx'], exp_qvalues['qz'], exp_qvalues['qy']), sum_frames=True,
    #                   title='Experimental data', levels=np.linspace(0, np.log10(data).max(), 150, endpoint=False),
    #                   scale='log', plot_colorbar=True, is_orthogonal=True, reciprocal_space=True)
    gu.contour_slices(data, (exp_qvalues['qx'], exp_qvalues['qz'], exp_qvalues['qy']), sum_frames=True,
                      title='Experimental data', levels=np.linspace(0, 1, 10, endpoint=False),
                      scale='linear', plot_colorbar=True, is_orthogonal=True, reciprocal_space=True)
else:
    gu.multislices_plot(data, sum_frames=True, title='Experimental data', vmin=0, vmax=np.log10(data).max(),
                        scale='log', plot_colorbar=True, cmap=my_cmap, is_orthogonal=True, reciprocal_space=True)

fig, _, _ = gu.contour_slices(struct_array, q_values, sum_frames=True, title='Simulated diffraction pattern',
                              levels=np.linspace(0, struct_array.max(), 10, endpoint=False),
                              plot_colorbar=False, scale='linear', is_orthogonal=True, reciprocal_space=True)
fig.text(0.60, 0.25, "Energy = " + str(energy / 1000) + " keV", size=12)
fig.text(0.60, 0.20, "SDD = " + str(sdd) + " m", size=12)
fig.text(0.60, 0.15, unitcell + " unit cell of parameter = " + str(unitcell_param) + " nm", size=12)
fig.text(0.60, 0.10, "Rotation of the unit cell in degrees (Qx, Qz, Qy) = " + str(alpha) + "," +
         str(beta) + "," + str(gamma), size=12)
plt.pause(0.1)
plt.ioff()
plt.show()
