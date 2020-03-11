# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import peak_local_max
import tkinter as tk
from tkinter import filedialog
import gc
import time
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
maximize the cross-correlation of the simulated data with experimental data. The experimental data should be sparse 
(using a photon threshold), and Bragg peaks maximum must be clearly identifiable. 

Laboratory frame convention (CXI): z downstream, y vertical up, x outboard."""

datadir = "D:/data/P10_August2019/data/gold2_2_00515/pynx/441_486_441_1_4_4_masked/"
savedir = "D:/data/P10_August2019/data/gold2_2_00515/simu/"
comment = ''  # should start with _
################
# sample setup #
################
unitcell = 'bct'  # supported unit cells: 'cubic', 'bcc', 'fcc', 'bct'
unitcell_param = (15.2, 25.0)   # in nm, unit cell parameter.  # (15.84, 22.4)
# It can be a number or tuple of numbers depending on the unit cell.
unitcell_ranges = [14.8, 15.4, 24.6, 25.2]  # in nm, values of the unit cell parameters to test
# If the unit cell is cubic: [start, stop]. If the unit cell is bct: [start1, stop1, start2, stop2] etc...
unitcell_step = 6  # number of steps within unitcell_ranges
#########################
# unit cell orientation #
#########################
angles_ranges = [0, 5.25, 45+24, 45+28.25, -5, 0.25]  # [start, stop, start, stop, start, stop], in degrees
# ranges to span for the rotation around qx downstream, qz vertical up and qy outboard respectively (stop is excluded)
angular_step = 1  # in degrees
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
##########################
# peak detection options #
##########################
min_distance = 20  # minimum distance between Bragg peaks in pixels
peak_width = 0  # the total width will be (2*peak_width+1)
###########
# options #
###########
kernel_length = 21  # width of the 3D gaussian window
debug = False  # True to see more plots
correct_background = False  # True to create a 3D background
bckg_method = 'normalize'  # 'subtract' or 'normalize'

##################################
# end of user-defined parameters #
##################################
comment = comment + '_' + unitcell
if unitcell == 'bct':
    comment = comment + '_a=' + str(unitcell_param[0])+'_c=' + str(unitcell_param[1])
else:
    comment = comment + '_a=' + str(unitcell_param)

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
nz, ny, nx = data.shape
print('Sparsity of the data:', str('{:.2f}'.format((data == 0).sum()/(nz*ny*nx)*100)), '%')

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

##########################
# plot experimental data #
##########################
gu.multislices_plot(data, sum_frames=True, title='data', vmin=0, vmax=np.log10(data).max(), scale='log',
                    plot_colorbar=True, cmap=my_cmap, is_orthogonal=True, reciprocal_space=True)

if debug:
    if qvalues_flag:
        gu.contour_slices(data, q_coordinates=(exp_qvalues['qx'], exp_qvalues['qz'], exp_qvalues['qy']),
                          sum_frames=True, title='Experimental data',
                          levels=np.linspace(0, np.log10(data.max())+1, 10, endpoint=False),
                          scale='log', plot_colorbar=False, is_orthogonal=True, reciprocal_space=True)
    else:
        gu.contour_slices(data, q_coordinates=q_values, sum_frames=True,
                          title='Experimental data', levels=np.linspace(0, np.log10(data.max())+1, 10, endpoint=False),
                          scale='log', plot_colorbar=False, is_orthogonal=True, reciprocal_space=True)

################################################
# remove background from the experimental data #
################################################

if correct_background:
    file_path = filedialog.askopenfilename(initialdir=datadir, title="Select the 1D background file",
                                           filetypes=[("NPZ", "*.npz")])
    avg_background = np.load(file_path)['background']
    distances = np.load(file_path)['distances']

    if qvalues_flag:
        data = util.remove_background(array=data, avg_background=avg_background, avg_qvalues=distances,
                                      q_values=(exp_qvalues['qx'], exp_qvalues['qz'], exp_qvalues['qy']),
                                      method=bckg_method)
    else:
        print('Using calculated q values for background subtraction')
        data = util.remove_background(array=data, q_values=q_values, avg_background=avg_background,
                                      avg_qvalues=distances, method=bckg_method)

    np.savez_compressed(datadir+'data-background_'+str(nbz)+'_'+str(nby)+'_'+str(nbx)+'.npz', data=data)

    gu.multislices_plot(data, sum_frames=True, title='Background subtracted data', vmin=0,
                        vmax=np.log10(data).max(), scale='log', plot_colorbar=True, cmap=my_cmap,
                        is_orthogonal=True, reciprocal_space=True)

#############################################
# find Bragg peaks in the experimental data #
#############################################
density_map = np.copy(data)

# find peaks
local_maxi = peak_local_max(density_map, exclude_border=False, min_distance=min_distance, indices=True)
nb_peaks = local_maxi.shape[0]
print('Number of Bragg peaks isolated:', nb_peaks)
print('Bragg peaks positions:')
print(local_maxi)

density_map[:] = 0

for idx in range(nb_peaks):
    piz, piy, pix = local_maxi[idx]
    density_map[piz-peak_width:piz+peak_width+1, piy-peak_width:piy+peak_width+1, pix-peak_width:pix+peak_width+1] = 1

nonzero_indices = np.nonzero(density_map)
bragg_peaks = density_map[nonzero_indices]  # 1D array of length: nb_peaks*(2*peak_width+1)**3

if debug:
    gu.multislices_plot(density_map, sum_frames=True, title='Bragg peaks positions', slice_position=pivot, vmin=0,
                        vmax=1, scale='linear', cmap=my_cmap, is_orthogonal=True, reciprocal_space=True)
    plt.pause(0.1)

#########################
# define the peak shape #
#########################
peak_shape = pu.blackman_window(shape=(kernel_length, kernel_length, kernel_length), normalization=100)

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
start = time.time()
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
            corr[idz, idy, idx] = np.multiply(bragg_peaks, struct_array[nonzero_indices]).sum()
            # print(alpha, beta, gamma, corr[idz, idy, idx])
end = time.time()
print('Time ellapsed in the loop over angles (s)', int(end - start))

##########################################
# plot the correlation matrix at maximum #
##########################################
vmin = corr.min()
vmax = corr.max()
if vmax == vmin:
    print('The correlation map is flat: no maximum in this range of angles')
    sys.exit()

piz, piy, pix = np.unravel_index(abs(corr).argmax(), corr.shape)
alpha, beta, gamma = angles_qx[piz], angles_qz[piy], angles_qy[pix]
print('Maximum correlation for (angle_qx, angle_qz, angle_qy) =', alpha, beta, gamma)

if all([corr.shape[idx] > 1 for idx in range(corr.ndim)]):  # 3D
    fig, _, _ = gu.contour_slices(corr, (angles_qx, angles_qz, angles_qy), sum_frames=False,
                                  title='Correlation', slice_position=[piz, piy, pix], plot_colorbar=True, cmap=my_cmap,
                                  levels=np.linspace(vmin, vmax, 10, endpoint=False), is_orthogonal=True,
                                  reciprocal_space=True)
    fig.text(0.60, 0.25, "Kernel size = " + str(kernel_length) + " pixels", size=12)
else:
    # find which angle is 1D
    nonzero_dim = np.nonzero(np.asarray(corr.shape) != 1)[0]
    corr = np.squeeze(corr)
    labels = ['rotation around qx (deg)', 'rotation around qz (deg)', 'rotation around qy (deg)']
    if corr.ndim == 2:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        if (nonzero_dim[0] == 0) and (nonzero_dim[1] == 1):
            plt0 = ax.contourf(angles_qz, angles_qx, corr, np.linspace(vmin, vmax, 10, endpoint=False), cmap=my_cmap)
        elif (nonzero_dim[0] == 0) and (nonzero_dim[1] == 2):
            plt0 = ax.contourf(angles_qy, angles_qx, corr, np.linspace(vmin, vmax, 10, endpoint=False), cmap=my_cmap)
        else:
            plt0 = ax.contourf(angles_qy, angles_qz, corr, np.linspace(vmin, vmax, 10, endpoint=False), cmap=my_cmap)
        plt.colorbar(plt0, ax=ax)
        ax.set_ylabel(labels[nonzero_dim[0]])
        ax.set_xlabel(labels[nonzero_dim[1]])
        ax.set_title('Correlation')
    else:  # 1D
        fig = plt.figure()
        if nonzero_dim[0] == 0:
            plt.plot(angles_qx, corr, '.r')
        elif nonzero_dim[0] == 1:
            plt.plot(angles_qz, corr, '.r')
        elif nonzero_dim[0] == 2:
            plt.plot(angles_qy, corr, '.r')
        plt.xlabel(labels[nonzero_dim[0]])
        plt.ylabel('Correlation')
    plt.pause(0.1)
plt.savefig(savedir + 'cross_corr_' + str(nbz) + '_' + str(nby) + '_' + str(nbx) + '_' + str(binning[0]) + '_' +
            str(binning[1]) + '_' + str(binning[2]) + '_rot_' + str(alpha) + '_' + str(beta) + '_' + str(gamma) +
            comment + '.png')

################################################
# rotate the lattice at calculated best values #
################################################
rot_lattice, _ = simu.rotate_lattice(lattice_list=ref_lattice, peaks_list=ref_peaks, original_shape=(nbz, nby, nbx),
                                     pad_offset=offset, pivot=pivot, euler_angles=(alpha, beta, gamma))
# peaks in the format [[h, l, k], ...]: CXI convention downstream , vertical up, outboard

# assign the peak shape to each lattice point
struct_array = simu.assign_peakshape(array_shape=(nbz, nby, nbx), lattice_list=rot_lattice,
                                     peak_shape=peak_shape, pivot=pivot)

#######################################################
# plot the overlay of experimental and simulated data #
#######################################################
plot_max = 2*peak_shape.sum(axis=0).max()
density_map[np.nonzero(density_map)] = 10*plot_max
fig, _, _ = gu.multislices_plot(struct_array+density_map, sum_frames=True, title='Overlay',
                                vmin=0, vmax=plot_max, plot_colorbar=True, scale='linear',
                                is_orthogonal=True, reciprocal_space=True)
fig.text(0.55, 0.25, "Energy = " + str(energy / 1000) + " keV", size=12)
fig.text(0.55, 0.20, "SDD = " + str(sdd) + " m", size=12)
fig.text(0.55, 0.15, unitcell + " unit cell of parameter = " + str(unitcell_param) + " nm", size=12)
fig.text(0.55, 0.10, "Rotation of the unit cell in degrees (Qx, Qz, Qy) = " + str(alpha) + "," +
         str(beta) + "," + str(gamma), size=12)
plt.pause(0.1)
plt.savefig(
    savedir + 'q_sum_' + str(nbz) + '_' + str(nby) + '_' + str(nbx) + '_' + str(binning[0]) + '_' +
    str(binning[1]) + '_' + str(binning[2]) + '_rot_' + str(alpha) + '_' + str(beta) + '_' +
    str(gamma) + comment + '.png')

if debug:
    fig, _, _ = gu.multislices_plot(struct_array, sum_frames=True, title='Simulated diffraction pattern',
                                    vmin=0, vmax=plot_max, plot_colorbar=False, scale='linear',
                                    is_orthogonal=True, reciprocal_space=True)
    fig.text(0.55, 0.25, "Energy = " + str(energy / 1000) + " keV", size=12)
    fig.text(0.55, 0.20, "SDD = " + str(sdd) + " m", size=12)
    fig.text(0.55, 0.15, unitcell + " unit cell of parameter = " + str(unitcell_param) + " nm", size=12)
    fig.text(0.55, 0.10, "Rotation of the unit cell in degrees (Qx, Qz, Qy) = " + str(alpha) + "," +
             str(beta) + "," + str(gamma), size=12)
    plt.pause(0.1)

    fig, _, _ = gu.contour_slices(struct_array, q_coordinates=q_values, sum_frames=True,
                                  title='Simulated diffraction pattern', cmap=my_cmap,
                                  levels=np.linspace(struct_array.min()+plot_max/100, plot_max, 10, endpoint=False),
                                  plot_colorbar=True, scale='linear', is_orthogonal=True, reciprocal_space=True)
    fig.text(0.55, 0.25, "Energy = " + str(energy / 1000) + " keV", size=12)
    fig.text(0.55, 0.20, "SDD = " + str(sdd) + " m", size=12)
    fig.text(0.55, 0.15, unitcell + " unit cell of parameter = " + str(unitcell_param) + " nm", size=12)
    fig.text(0.55, 0.10, "Rotation of the unit cell in degrees (Qx, Qz, Qy) = " + str(alpha) + "," +
             str(beta) + "," + str(gamma), size=12)
    plt.pause(0.1)

#################################
# optimize unit cell parameters #
#################################
if unitcell == 'bct':
    a_values = np.linspace(start=unitcell_ranges[0], stop=unitcell_ranges[1], num=unitcell_step)
    c_values = np.linspace(start=unitcell_ranges[2], stop=unitcell_ranges[3], num=unitcell_step)
    param_range = np.concatenate((a_values, c_values)).reshape((2, unitcell_step))
    start = time.time()
    corr = np.zeros((unitcell_step, unitcell_step))
    for idy, a in enumerate(param_range[0]):
        for idx, c in enumerate(param_range[1]):
            _, _, _, rot_lattice, _ = simu.lattice(energy=energy, sdd=sdd, direct_beam=direct_beam,
                                                   detector=detector, unitcell=unitcell, unitcell_param=(a, c),
                                                   euler_angles=(alpha, beta, gamma), offset_indices=False)
            # peaks in the format [[h, l, k], ...]: CXI convention downstream , vertical up, outboard

            # assign the peak shape to each lattice point
            struct_array = simu.assign_peakshape(array_shape=(nbz, nby, nbx), lattice_list=rot_lattice,
                                                 peak_shape=peak_shape, pivot=pivot)

            # calculate the correlation between experimental data and simulated data
            corr[idy, idx] = np.multiply(bragg_peaks, struct_array[nonzero_indices]).sum()
    end = time.time()
    print('Time ellapsed in the loop over lattice parameters (s)', int(end - start))
else:
    param_range = np.linspace(start=unitcell_ranges[0], stop=unitcell_ranges[1], num=unitcell_step)
    start = time.time()
    corr = np.zeros(unitcell_step)
    for idx, a in enumerate(param_range[0]):
        _, _, _, rot_lattice, _ = simu.lattice(energy=energy, sdd=sdd, direct_beam=direct_beam,
                                               detector=detector, unitcell=unitcell, unitcell_param=a,
                                               euler_angles=(alpha, beta, gamma), offset_indices=False)
        # peaks in the format [[h, l, k], ...]: CXI convention downstream , vertical up, outboard

        # assign the peak shape to each lattice point
        struct_array = simu.assign_peakshape(array_shape=(nbz, nby, nbx), lattice_list=rot_lattice,
                                             peak_shape=peak_shape, pivot=pivot)

        # calculate the correlation between experimental data and simulated data
        corr[idx] = np.multiply(bragg_peaks, struct_array[nonzero_indices]).sum()
    end = time.time()
    print('Time ellapsed in the loop over the lattice parameter (s)', int(end - start))

##########################################
# plot the correlation matrix at maximum #
##########################################
vmin = corr.min()
vmax = corr.max()
if vmax == vmin:
    print('The correlation map is flat: no maximum in this range of unit cell lattice parameters')
    sys.exit()

if unitcell == 'bct':
    piy, pix = np.unravel_index(abs(corr).argmax(), corr.shape)
    best_param = param_range[0, piy], param_range[1, pix]
    print('Maximum correlation for (a, c) =', a, c)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    plt0 = ax.contourf(param_range[1], param_range[0], corr, np.linspace(vmin, vmax, 10, endpoint=False), cmap=my_cmap)
    plt.colorbar(plt0, ax=ax)
    ax.set_ylabel('a parameter (nm)')
    ax.set_xlabel('c parameter (nm)')
    ax.set_title('Correlation')
    plt.pause(0.1)
    plt.savefig(savedir + 'cross_corr_' + str(nbz) + '_' + str(nby) + '_' + str(nbx) + '_' + str(binning[0]) + '_' +
                str(binning[1]) + '_' + str(binning[2]) + '_param a=' + str(best_param[0]) + '_c=' + str(best_param[1])
                + comment + '.png')
else:  # 1D
    pix = np.unravel_index(abs(corr).argmax(), corr.shape)
    best_param = param_range[pix]
    fig = plt.figure()
    plt.plot(param_range, corr, '.r')
    plt.xlabel('a parameter (nm)')
    plt.ylabel('Correlation')
    plt.pause(0.1)
    plt.savefig(savedir + 'cross_corr_' + str(nbz) + '_' + str(nby) + '_' + str(nbx) + '_' + str(binning[0]) + '_' +
                str(binning[1]) + '_' + str(binning[2]) + '_param a=' + str(best_param) + comment + '.png')

###################################################
# calculate the lattice at calculated best values #
###################################################
_, _, _, rot_lattice, _ = simu.lattice(energy=energy, sdd=sdd, direct_beam=direct_beam,
                                       detector=detector, unitcell=unitcell, unitcell_param=best_param,
                                       euler_angles=(alpha, beta, gamma), offset_indices=False)
# peaks in the format [[h, l, k], ...]: CXI convention downstream , vertical up, outboard

# assign the peak shape to each lattice point
struct_array = simu.assign_peakshape(array_shape=(nbz, nby, nbx), lattice_list=rot_lattice,
                                     peak_shape=peak_shape, pivot=pivot)

#######################################################
# plot the overlay of experimental and simulated data #
#######################################################
if unitcell == 'bct':
    text = unitcell + " unit cell of parameter(s) = {:.2f} nm, {:.2f}".format(best_param[0], best_param[1]) + " nm"
else:
    text = unitcell + " unit cell of parameter(s) = " + str('{:.2f}'.format(best_param)) + " nm"

plot_max = 2*peak_shape.sum(axis=0).max()
density_map[np.nonzero(density_map)] = 10*plot_max
fig, _, _ = gu.multislices_plot(struct_array+density_map, sum_frames=True, title='Overlay',
                                vmin=0, vmax=plot_max, plot_colorbar=True, scale='linear',
                                is_orthogonal=True, reciprocal_space=True)
fig.text(0.55, 0.25, "Energy = " + str(energy / 1000) + " keV", size=12)
fig.text(0.55, 0.20, "SDD = " + str(sdd) + " m", size=12)
fig.text(0.55, 0.15, text, size=12)
fig.text(0.55, 0.10, "Rotation of the unit cell in degrees (Qx, Qz, Qy) = " + str(alpha) + "," +
         str(beta) + "," + str(gamma), size=12)
plt.pause(0.1)
plt.savefig(
    savedir + 'q_sum_' + str(nbz) + '_' + str(nby) + '_' + str(nbx) + '_' + str(binning[0]) + '_' +
    str(binning[1]) + '_' + str(binning[2]) + '_rot_' + str(alpha) + '_' + str(beta) + '_' +
    str(gamma) + comment + '.png')

if debug:
    fig, _, _ = gu.multislices_plot(struct_array, sum_frames=True, title='Simulated diffraction pattern',
                                    vmin=0, vmax=plot_max, plot_colorbar=False, scale='linear',
                                    is_orthogonal=True, reciprocal_space=True)
    fig.text(0.55, 0.25, "Energy = " + str(energy / 1000) + " keV", size=12)
    fig.text(0.55, 0.20, "SDD = " + str(sdd) + " m", size=12)
    fig.text(0.55, 0.15, text, size=12)
    fig.text(0.55, 0.10, "Rotation of the unit cell in degrees (Qx, Qz, Qy) = " + str(alpha) + "," +
             str(beta) + "," + str(gamma), size=12)
    plt.pause(0.1)

    fig, _, _ = gu.contour_slices(struct_array, q_coordinates=q_values, sum_frames=True,
                                  title='Simulated diffraction pattern', cmap=my_cmap,
                                  levels=np.linspace(struct_array.min()+plot_max/100, plot_max, 10, endpoint=False),
                                  plot_colorbar=True, scale='linear', is_orthogonal=True, reciprocal_space=True)
    fig.text(0.55, 0.25, "Energy = " + str(energy / 1000) + " keV", size=12)
    fig.text(0.55, 0.20, "SDD = " + str(sdd) + " m", size=12)
    fig.text(0.55, 0.15, text, size=12)
    fig.text(0.55, 0.10, "Rotation of the unit cell in degrees (Qx, Qz, Qy) = " + str(alpha) + "," +
             str(beta) + "," + str(gamma), size=12)
    plt.pause(0.1)

plt.ioff()
plt.show()
