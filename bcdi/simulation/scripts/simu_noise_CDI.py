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
import tkinter as tk
from tkinter import filedialog
from scipy.interpolate import RegularGridInterpolator
import gc
import os
import sys
sys.path.append('//win.desy.de/home/carnisj/My Documents/myscripts/bcdi/')
import bcdi.graph.graph_utils as gu
import bcdi.postprocessing.postprocessing_utils as pu
import bcdi.preprocessing.preprocessing_utils as pru

helptext = """
simu_noise.py

Using a support created from a reconstruction (real space), calculate the diffraction pattern depending on several 
parameters: detector size, detector distance, presence/width of a detector gap, Poisson noise, user-defined phase.

The provided reconstruction is expected to be orthogonalized, in the laboratory frame. """

scan = 2227  # spec scan number
datadir = "C:/Users/carnis/Work Folders/Documents/data/CH4760_Pt/S"+str(scan)+"/simu/crop400phase/new/"

sdd = 0.50678  # 1.0137  # sample to detector distance in m
en = 9000.0 - 6   # x-ray energy in eV, 6eV offset at ID01
voxel_size = 3  # in nm, voxel size of the reconstruction, should be eaqual in each direction
photon_threshold = 0  # 0.75
photon_number = 5e7  # total number of photons in the array, usually around 5e7
orthogonal_frame = False  # set to False to interpolate the diffraction pattern in the detector frame
support_threshold = 0.24  # threshold for support determination
setup = "ID01"  # only "ID01"
rocking_angle = "outofplane"  # "outofplane" or "inplane"
outofplane_angle = 35.3240  # detector delta ID01
inplane_angle = -1.6029  # detector nu ID01
grazing_angle = 0  # in degrees, incident angle for in-plane rocking curves (eta ID01)
tilt_angle = 0.0102  # angular step size for rocking angle, eta ID01
pixel_size = 55e-6  # detector pixel size in m

set_gap = False  # set to True if you want to use the detector gap in the simulation (updates the mask)
gap_width = 6  # number of pixels to mask
gap_pixel_start = 650

flat_phase = True  # set to True to use a phase flat (0 everywhere)

include_noise = False  # set to True to include poisson noise on the data

pad_size = [1000, 1000, 1000]  # will pad the array by this amount of zeroed pixels in z, y, x at both ends
# if only a number (e.g. 3), will pad to get three times the initial array size  # ! max size ~ [800, 800, 800]
crop_size = [400, 400, 400]  # will crop the array to this size

ref_axis_outplane = "y"  # "y"  # "z"  # q is supposed to be aligned along that axis before rotating back (nexus)
phase_range = np.pi  # for plots
strain_range = 0.001  # for plots
debug = True  # True to see all plots
save_fig = True  # if True save figures
save_data = True  # if True save data as npz and VTK
comment = "_coord23_iso0.24"  # should start with _
if not set_gap:
    comment = comment + "_nogap"
######################################


def mask3d_maxipix(mydata, mymask, start_pixel, width_gap):
    mydata[:, :, start_pixel:start_pixel+width_gap] = 0
    mydata[:, start_pixel:start_pixel+width_gap, :] = 0

    mymask[:, :, start_pixel:start_pixel+width_gap] = 1
    mymask[:, start_pixel:start_pixel+width_gap, :] = 1
    return mydata, mymask


def detector_frame(myobj, energy, outofplane, inplane, tilt, myrocking_angle, mygrazing_angle, distance, pixel_x,
                   pixel_y, geometry, debugging=True):
    """
    Interpolate orthogonal myobj back into the non-orthogonal detector frame

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
    if debugging:
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

    if debugging:
        gu.multislices_plot(abs(detector_obj), sum_frames=True, invert_yaxis=True, cmap=my_cmap,
                            title='Object interpolated in detector frame')

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
    # if debugging:
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


###################
# define colormap #
###################
bad_color = '1.0'  # white background
colormap = gu.Colormap(bad_color=bad_color)
my_cmap = colormap.cmap

#########################
# load a reconstruction #
#########################
plt.ion()
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(initialdir=datadir, filetypes=[("NPZ", "*.npz")])
npzfile = np.load(file_path)

amp = npzfile['amp']
nz, ny, nx = amp.shape
print("Initial data size: (", nz, ',', ny, ',', nx, ')')

gu.multislices_plot(amp, sum_frames=False, plot_colorbar=False, width_z=200, width_y=200, width_x=200,
                    vmin=0, vmax=1, invert_yaxis=True, cmap=my_cmap, title='Amp')

##########################################################
# calculate q for later regridding in the detector frame #
##########################################################
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

#########################################
# define the support and impose a phase #
#########################################
support = np.ones((nz, ny, nx))

if flat_phase:
    phase = np.zeros((nz, ny, nx))
else:
    # model for paper about artefacts in BCDI
    oscillation_period = 100  # in pixels
    z, y, x = np.meshgrid(np.cos(np.arange(-nz // 2, nz // 2, 1)*2*np.pi/oscillation_period),
                          np.cos(np.arange(-ny // 2, ny // 2, 1)*2*np.pi/oscillation_period),
                          np.cos(np.arange(-nx // 2, nx // 2, 1)*2*np.pi/oscillation_period), indexing='ij')
    phase = z + y + x

if debug:
    gu.multislices_plot(phase, sum_frames=False, plot_colorbar=True, width_z=200, width_y=200, width_x=200,
                        vmin=-phase_range, vmax=phase_range,
                        invert_yaxis=True, cmap=my_cmap, title='Phase before wrapping')

phase = pru.wrap(phase, start_angle=-np.pi, range_angle=2*np.pi)

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
coordination_matrix = pu.calc_coordination(support, debugging=debug)
surface = np.copy(support)
surface[coordination_matrix > 23] = 0  # remove the bulk 22
bulk = support - surface
bulk[np.nonzero(bulk)] = 1

if debug:
    gu.multislices_plot(surface, sum_frames=False, plot_colorbar=False, width_z=200, width_y=200, width_x=200,
                        vmin=0, vmax=1, invert_yaxis=True, cmap=my_cmap, title='surface')

    surface = np.multiply(surface, strain)

    gu.multislices_plot(surface, sum_frames=False, plot_colorbar=True, width_z=200, width_y=200, width_x=200,
                        vmin=-strain_range, vmax=strain_range, invert_yaxis=True, cmap=my_cmap, title='surface strain')

    gu.multislices_plot(support, sum_frames=True, plot_colorbar=False, width_z=200, width_y=200, width_x=200,
                        invert_yaxis=True, cmap=my_cmap, title='Orthogonal support')

    if not flat_phase:
        gu.multislices_plot(phase, sum_frames=False, plot_colorbar=True, width_z=200, width_y=200, width_x=200,
                            vmin=-phase_range, vmax=phase_range, invert_yaxis=True, cmap=my_cmap,
                            title='Orthogonal phase')

        strain[bulk == 0] = 0  # for easier visualization
        if save_fig:
            plt.savefig(
                datadir + 'S' + str(scan) + '_phase_' + str('{:.0e}'.format(photon_number)) + comment + '.png')
        if save_data:
            np.savez_compressed(datadir + 'S' + str(scan) + '_amp-phase-strain_SIMU' + comment,
                                amp=support, phase=phase, bulk=bulk, strain=strain)

            # save amp & phase to VTK
            # in VTK, x is downstream, y vertical, z inboard, thus need to flip the last axis
            gu.save_to_vti(filename=os.path.join(datadir, "S" + str(scan) +
                                                 "_amp-phase-strain_SIMU" + comment + ".vti"),
                           voxel_size=(voxel_size, voxel_size, voxel_size), tuple_array=(support, bulk, phase, strain),
                           tuple_fieldnames=('amp', 'bulk', 'phase', 'strain'), amplitude_threshold=0.01)

        gu.multislices_plot(strain, sum_frames=False, plot_colorbar=True, width_z=200, width_y=200, width_x=200,
                            vmin=-strain_range, vmax=strain_range, invert_yaxis=True, cmap=my_cmap,
                            title='strain')
        if save_fig:
            plt.savefig(
                datadir + 'S' + str(scan) + '_strain_' + str('{:.0e}'.format(photon_number)) + comment + '.png')

del strain, bulk

#####################################################################################
# keep the orthogonal object or interpolate it in the non-orthogonal detector frame #
#####################################################################################
if orthogonal_frame:
    obj = support * np.exp(1j * phase)
    del phase, support
    gc.collect()
    comment = comment + '_prtf'
    set_gap = 0  # gap is valid only in the detector frame
else:
    ######################################################################
    # rotate the object to have q in the same direction as in experiment #
    ######################################################################
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
    angle = pru.plane_angle(np.array([q[2], q[1], q[0]])/np.linalg.norm(q), myaxis)
    print("Angle between q and", ref_axis_outplane, "=", angle, "deg")
    print("Angle with y in zy plane", np.arctan(q[0]/q[1])*180/np.pi, "deg")
    print("Angle with y in xy plane", np.arctan(-q[2]/q[1])*180/np.pi, "deg")
    print("Angle with z in xz plane", 180+np.arctan(q[2]/q[0])*180/np.pi, "deg")
    support = pu.rotate_crystal(support, axis_to_align=myaxis,
                                reference_axis=np.array([q[2], q[1], q[0]]) / np.linalg.norm(q), debugging=False)
    phase = pu.rotate_crystal(phase, axis_to_align=myaxis,
                              reference_axis=np.array([q[2], q[1], q[0]]) / np.linalg.norm(q), debugging=True)

    obj = support * np.exp(1j * phase)
    del phase, support
    gc.collect()

    #############################################
    # transform object back into detector frame #
    #############################################

    obj, _, _, _ = detector_frame(myobj=obj, energy=en, outofplane=outofplane_angle,
                                  inplane=inplane_angle, tilt=tilt_angle, myrocking_angle=rocking_angle,
                                  mygrazing_angle=grazing_angle, distance=sdd, pixel_x=pixel_size,
                                  pixel_y=pixel_size, geometry=setup, debugging=True)
    if debug:

        gu.multislices_plot(abs(obj), sum_frames=True, invert_yaxis=True, cmap=my_cmap,
                            title='Support in detector frame')

        gu.multislices_plot(np.angle(obj), sum_frames=False, plot_colorbar=True,
                            vmin=-phase_range, vmax=phase_range, invert_yaxis=True, cmap=my_cmap,
                            title='Phase in detector frame')

    #################################################################
    # uncomment this if you want to save the non-orthogonal support #
    # in that case pad_size and crop_size should be identical       #
    #################################################################
    # support = abs(obj)
    # support = support / support.max()
    # support[support < 0.05] = 0
    # support[np.nonzero(support)] = 1
    # np.savez_compressed(datadir + 'S' + str(scan) + 'support_nonortho400.npz', obj=support)

#################
# pad the array #
#################
nz1, ny1, nx1 = pad_size
if nz1 < nz or ny1 < ny or nx1 < nx:
    print('Pad size smaller than initial array size')
    sys.exit()

newobj = np.zeros((nz1, ny1, nx1), dtype=complex)
newobj[(nz1-nz)//2:(nz1+nz)//2, (ny1-ny)//2:(ny1+ny)//2, (nx1-nx)//2:(nx1+nx)//2] = obj
nz1, ny1, nx1 = newobj.shape
print("Padded data size: (", nz1, ',', ny1, ',', nx1, ')')
comment = comment + "_pad_" + str(nz1) + "," + str(ny1) + "," + str(nx1)

###########################################################
# calculate the diffraction pattern and add detector gaps #
###########################################################
data = fftshift(abs(fftn(newobj))**2)
data = data / data.sum() * photon_number  # convert into photon number

##############################
# apply the photon threshold #
##############################
mask = np.zeros((nz1, ny1, nx1))
mask[data <= photon_threshold] = 1
data[data <= photon_threshold] = 0

gu.multislices_plot(data, sum_frames=False,  scale='log', plot_colorbar=True, vmin=-5, invert_yaxis=False,
                    cmap=my_cmap, reciprocal_space=True, is_orthogonal=False, title='After padding')
if save_fig:
    plt.savefig(datadir + 'S' + str(scan) + '_diff_float_' + str('{:.0e}'.format(photon_number))+comment + '_sum.png')

#########################
# include Poisson noise #
#########################
if include_noise:
    data = np.rint(poisson(data)).astype(int)
    comment = comment + "_noise"
else:
    data = np.rint(data).astype(int)

if set_gap:
    data, mask = mask3d_maxipix(data, mask, start_pixel=gap_pixel_start, width_gap=gap_width)

gu.multislices_plot(data, sum_frames=False,  scale='log', plot_colorbar=True, vmin=-1, invert_yaxis=False,
                    cmap=my_cmap, reciprocal_space=True, is_orthogonal=False, title='After rounding')

myfig, _, _ = gu.multislices_plot(data, sum_frames=True,  scale='log', plot_colorbar=True, vmin=-1, invert_yaxis=False,
                                  cmap=my_cmap, reciprocal_space=True, is_orthogonal=False, title='Masked intensity')
myfig.text(0.60, 0.30, "Pad size =" + str(pad_size), size=20)
if save_fig:
    myfig.savefig(datadir + 'S' + str(scan) + '_diff_' + str('{:.0e}'.format(photon_number))+comment + '_sum.png')

###############
# crop arrays #
###############
nz, ny, nx = data.shape
nz1, ny1, nx1 = crop_size
if nz < nz1 or ny < ny1 or nx < nx1:
    print('Crop size larger than initial array size')
    sys.exit()
data = data[(nz - nz1) // 2:(nz + nz1) // 2, (ny - ny1) // 2:(ny + ny1) // 2, (nx - nx1) // 2:(nx + nx1) // 2]
mask = mask[(nz - nz1) // 2:(nz + nz1) // 2, (ny - ny1) // 2:(ny + ny1) // 2, (nx - nx1) // 2:(nx + nx1) // 2]
nz, ny, nx = data.shape
nz1, ny1, nx1 = pru.smaller_primes((nz, ny, nx), maxprime=7, required_dividers=(2,))
data = data[(nz - nz1) // 2:(nz + nz1) // 2, (ny - ny1) // 2:(ny + ny1) // 2, (nx - nx1) // 2:(nx + nx1) // 2]
mask = mask[(nz - nz1) // 2:(nz + nz1) // 2, (ny - ny1) // 2:(ny + ny1) // 2, (nx - nx1) // 2:(nx + nx1) // 2]
nz, ny, nx = data.shape
print("cropped FFT data size:", data.shape)
print("Total number of photons:", data.sum())
comment = comment + "_crop_" + str(nz) + "," + str(ny) + "," + str(nx)

##############
# save files #
##############
if save_data:
    np.savez_compressed(datadir + 'S' + str(scan) + '_diff_' + str('{:.0e}'.format(photon_number))+comment, data=data)
    np.savez_compressed(datadir + 'S' + str(scan) + '_mask_' + str('{:.0e}'.format(photon_number))+comment, mask=mask)

#####################################
# plot mask and diffraction pattern #
#####################################
plt.ioff()
if debug:
    gu.multislices_plot(mask, sum_frames=True, scale='linear', plot_colorbar=False, invert_yaxis=False,
                        cmap=my_cmap, reciprocal_space=True, is_orthogonal=False, title='Mask')

myfig, _, _ = gu.multislices_plot(data, sum_frames=False,  scale='log', plot_colorbar=True, vmin=-1, invert_yaxis=False,
                                  cmap=my_cmap, reciprocal_space=True, is_orthogonal=False, title='Masked intensity')
myfig.text(0.60, 0.30, "Pad size =" + str(pad_size), size=20)
myfig.text(0.60, 0.25, "Crop size =" + str(crop_size), size=20)
# myfig.text(0.60, 0.20, "New tilt angle =" + str('{:.4f}'.format(tilt_crop)) + "deg", size=20)
# myfig.text(0.60, 0.15, "New detector pixel size y =" + str('{:.2f}'.format(pixel_crop_y * 1e6)) + "um", size=20)
# myfig.text(0.60, 0.10, "New detector pixel size x =" + str('{:.2f}'.format(pixel_crop_x * 1e6)) + "um", size=20)
if save_fig:
    myfig.savefig(datadir + 'S' + str(scan) + '_diff_' + str('{:.0e}'.format(photon_number))+comment + '_center.png')

myfig, _, _ = gu.multislices_plot(data, sum_frames=True,  scale='log', plot_colorbar=True, vmin=-1, invert_yaxis=False,
                                  cmap=my_cmap, reciprocal_space=True, is_orthogonal=False, title='Masked intensity')
myfig.text(0.60, 0.30, "Pad size =" + str(pad_size), size=20)
myfig.text(0.60, 0.25, "Crop size =" + str(crop_size), size=20)
if set_gap:
    myfig.text(0.60, 0.20, "Gap width =" + str(gap_width) + "pixels", size=20)
# myfig.text(0.60, 0.20, "New tilt angle =" + str('{:.4f}'.format(tilt_crop)) + "deg", size=20)
# myfig.text(0.60, 0.15, "New detector pixel size y =" + str('{:.2f}'.format(pixel_crop_y * 1e6)) + "um", size=20)
# myfig.text(0.60, 0.10, "New detector pixel size x =" + str('{:.2f}'.format(pixel_crop_x * 1e6)) + "um", size=20)
if save_fig:
    myfig.savefig(datadir + 'S' + str(scan) + '_diff_' + str('{:.0e}'.format(photon_number))+comment + '_sum.png')
plt.show()
