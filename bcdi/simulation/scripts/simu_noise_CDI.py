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
datadir = "D:/data/BCDI_isosurface/S"+str(scan)+"/test/"
# "C:/Users/carnis/Work Folders/Documents/data/CH4760_Pt/S"+str(scan)+"/simu/crop400phase/new/"

original_sdd = 0.50678  # 1.0137  # in m, sample to detector distance of the provided reconstruction
simulated_sdd = 0.50678  # in m, sample to detector distance for the simulated diffraction pattern
en = 9000.0 - 6   # x-ray energy in eV, 6eV offset at ID01
voxel_size = 3  # in nm, voxel size of the reconstruction, should be eaqual in each direction
photon_threshold = 0  # 0.75
photon_number = 5e7  # total number of photons in the array, usually around 5e7
orthogonal_frame = False  # set to False to interpolate the diffraction pattern in the detector frame
rotate_crystal = True  # if True, the crystal will be rotated as it was during the experiment
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

original_size = [400, 400, 400]  # size of the FFT array before binning. It will be modify to take into account binning
# during phasing automatically. Leave it to () if the shape did not change.
binning = (1, 1, 1)  # binning factor during phasing
pad_size = [500, 500, 500]  # will pad the array by this amount of zeroed pixels in z, y, x at both ends
# if only a number (e.g. 3), will pad to get three times the initial array size  # ! max size ~ [800, 800, 800]
crop_size = [300, 300, 300]  # will crop the array to this size

ref_axis_outplane = "y"  # "y"  # "z"  # q is supposed to be aligned along that axis before rotating back (nexus)
phase_range = np.pi  # for plots
strain_range = 0.001  # for plots
debug = False  # True to see all plots
save_fig = True  # if True save figures
save_data = True  # if True save data as npz and VTK
comment = ""  # should start with _
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
                   pixel_y, geometry, voxelsize, debugging=True, **kwargs):
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
    :param voxelsize: voxel size of the original object
    :param debugging: to show plots before and after orthogonalization
    :param kwargs:
     - 'title': title for the debugging plots
    :return: object interpolated on an orthogonal grid
    """
    global nz, ny, nx
    for k in kwargs.keys():
        if k in ['title']:
            title = kwargs['title']
        else:
            raise Exception("unknown keyword argument given: allowed is 'title'")
    try:
        title
    except NameError:  # title not declared
        title = 'Object'

    wavelength = 12.398 * 1e-7 / energy  # in m

    if debugging:
        gu.multislices_plot(abs(myobj), sum_frames=True, plot_colorbar=False,
                            invert_yaxis=True, cmap=my_cmap, title=title+' before interpolation\n')

    myz, myy, myx = np.meshgrid(np.arange(0, nz, 1), np.arange(0, ny, 1), np.arange(0, nx, 1),
                                indexing='ij')
    _, _, _, ortho_matrix = update_coords(mygrid=(myz, myy, myx), wavelength=wavelength,
                                          outofplane=outofplane, inplane=inplane, tilt=tilt,
                                          myrocking_angle=myrocking_angle, mygrazing_angle=mygrazing_angle,
                                          distance=distance, pixel_x=pixel_x, pixel_y=pixel_y,
                                          geometry=geometry)
    del myz, myy, myx

    ################################################
    # interpolate the data into the detector frame #
    ################################################
    myz, myy, myx = np.meshgrid(np.arange(-nz//2, nz//2, 1),
                                np.arange(-ny//2, ny//2, 1),
                                np.arange(-nx//2, nx//2, 1), indexing='ij')

    new_x = ortho_matrix[0, 0] * myx + ortho_matrix[0, 1] * myy + ortho_matrix[0, 2] * myz
    new_y = ortho_matrix[1, 0] * myx + ortho_matrix[1, 1] * myy + ortho_matrix[1, 2] * myz
    new_z = ortho_matrix[2, 0] * myx + ortho_matrix[2, 1] * myy + ortho_matrix[2, 2] * myz
    del myx, myy, myz
    # la partie rgi est sure: c'est la taille de l'objet orthogonal de depart
    rgi = RegularGridInterpolator((np.arange(-nz//2, nz//2)*voxelsize,
                                   np.arange(-ny//2, ny//2)*voxelsize,
                                   np.arange(-nx//2, nx//2)*voxelsize),
                                  myobj, method='linear', bounds_error=False, fill_value=0)
    detector_obj = rgi(np.concatenate((new_z.reshape((1, new_z.size)), new_y.reshape((1, new_z.size)),
                                      new_x.reshape((1, new_z.size)))).transpose())
    detector_obj = detector_obj.reshape((nz, ny, nx)).astype(myobj.dtype)

    if debugging:
        gu.multislices_plot(abs(detector_obj), sum_frames=True, invert_yaxis=True, cmap=my_cmap,
                            title=title+' interpolated in detector frame\n')
        
    return detector_obj


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
gu.multislices_plot(amp, sum_frames=False, plot_colorbar=False, vmin=0, vmax=1, invert_yaxis=True, cmap=my_cmap,
                    title='Input amplitude')

#################################
# pad data to the original size #
#################################
print("Initial data size:", amp.shape)
if len(original_size) == 0:
    original_size = amp.shape
print("FFT size before accounting for binning", original_size)
original_size = tuple([original_size[index] // binning[index] for index in range(len(binning))])
print("Binning used during phasing:", binning)
print("Padding back to original FFT size", original_size, '\n')
amp = pu.crop_pad(array=amp, output_shape=original_size)
nz, ny, nx = amp.shape

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
print("Interplanar distance: (angstroms)", str('{:.4f}'.format(planar_dist)), "angstroms")
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

if debug and not flat_phase:
    gu.multislices_plot(phase, sum_frames=False, plot_colorbar=True, width_z=200, width_y=200, width_x=200,
                        vmin=-phase_range, vmax=phase_range,
                        invert_yaxis=True, cmap=my_cmap, title='Phase before wrapping\n')

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

if debug and not flat_phase:
    gu.multislices_plot(surface, sum_frames=False, plot_colorbar=False, width_z=200, width_y=200, width_x=200,
                        vmin=0, vmax=1, invert_yaxis=True, cmap=my_cmap, title='surface')

    surface = np.multiply(surface, strain)

    gu.multislices_plot(surface, sum_frames=False, plot_colorbar=True, width_z=200, width_y=200, width_x=200,
                        vmin=-strain_range, vmax=strain_range, invert_yaxis=True, cmap=my_cmap, title='surface strain')

    gu.multislices_plot(support, sum_frames=True, plot_colorbar=False, invert_yaxis=True, cmap=my_cmap,
                        title='Orthogonal support\n')

    gu.multislices_plot(phase, sum_frames=False, plot_colorbar=True, width_z=200, width_y=200, width_x=200,
                        vmin=-phase_range, vmax=phase_range, invert_yaxis=True, cmap=my_cmap,
                        title='Orthogonal phase')

    strain[bulk == 0] = 0  # for easier visualization
    if save_fig:
        plt.savefig(datadir + 'S' + str(scan) + '_phase_' + str('{:.0e}'.format(photon_number)) + comment + '.png')
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
        plt.savefig(datadir + 'S' + str(scan) + '_strain_' + str('{:.0e}'.format(photon_number)) + comment + '.png')

del strain, bulk

##############################################################################
# rotate the object to have q in the same direction as during the experiment #
##############################################################################
if rotate_crystal:
    print('\nRotating the crystal to match experimental conditions')
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
    angle = pu.plane_angle(np.array([q[2], q[1], q[0]]) / np.linalg.norm(q), myaxis)
    print("Angle between q and", ref_axis_outplane, "=", angle, "deg")
    print("Angle with y in zy plane", np.arctan(q[0] / q[1]) * 180 / np.pi, "deg")
    print("Angle with y in xy plane", np.arctan(-q[2] / q[1]) * 180 / np.pi, "deg")
    print("Angle with z in xz plane", 180 + np.arctan(q[2] / q[0]) * 180 / np.pi, "deg")
    support = pu.rotate_crystal(support, axis_to_align=myaxis,
                                reference_axis=np.array([q[2], q[1], q[0]]) / np.linalg.norm(q), debugging=True)
    phase = pu.rotate_crystal(phase, axis_to_align=myaxis,
                              reference_axis=np.array([q[2], q[1], q[0]]) / np.linalg.norm(q), debugging=False)

original_obj = support * np.exp(1j * phase)
del phase, support
gc.collect()

##################################################################################################
# compensate padding in order to keep reciprocal space resolution (detector pixel size) constant #
##################################################################################################
gc.collect()
comment = comment + '_prtf'
set_gap = 0  # gap is valid only in the detector frame
print('\nOriginal voxel size', voxel_size, 'nm')
dqz = 2 * np.pi / (nz * voxel_size * 10)  # in inverse angstroms
dqy = 2 * np.pi / (ny * voxel_size * 10)  # in inverse angstroms
dqx = 2 * np.pi / (nx * voxel_size * 10)  # in inverse angstroms
print('Original reciprocal space resolution (z, y, x): (', str('{:.5f}'.format(dqz)), 'A-1,',
      str('{:.5f}'.format(dqy)), 'A-1,', str('{:.5f}'.format(dqx)), 'A-1 )')
print('Original q range (z, y, x): (', str('{:.5f}'.format(dqz*nz)), 'A-1,',
      str('{:.5f}'.format(dqy*ny)), 'A-1,', str('{:.5f}'.format(dqx*nx)), 'A-1 )\n')

dqz_pad = 2 * np.pi / (pad_size[0] * voxel_size * 10)  # in inverse angstroms
dqy_pad = 2 * np.pi / (pad_size[1] * voxel_size * 10)  # in inverse angstroms
dqx_pad = 2 * np.pi / (pad_size[2] * voxel_size * 10)  # in inverse angstroms
print('New reciprocal space resolution (z, y, x) after padding: (', str('{:.5f}'.format(dqz_pad)), 'A-1,',
      str('{:.5f}'.format(dqy_pad)), 'A-1,', str('{:.5f}'.format(dqx_pad)), 'A-1 )')
print('New q range after padding (z, y, x): (', str('{:.5f}'.format(dqz_pad*pad_size[0])), 'A-1,',
      str('{:.5f}'.format(dqy_pad*pad_size[1])), 'A-1,', str('{:.5f}'.format(dqx_pad*pad_size[2])), 'A-1 )\n')

voxelsize_z = 2 * np.pi / (pad_size[0] * dqz_pad * 10)  # in nm
voxelsize_y = 2 * np.pi / (pad_size[1] * dqy_pad * 10)  # in nm
voxelsize_x = 2 * np.pi / (pad_size[2] * dqx_pad * 10)  # in nm
print('New voxel sizes (z, y, x) after padding: (', str('{:.2f}'.format(voxelsize_z)), 'nm,',
      str('{:.2f}'.format(voxelsize_y)), 'nm,', str('{:.2f}'.format(voxelsize_x)), 'nm )')
print('Padding has no effect on real-space voxel size.\n')

print('Interpolating the object to keep the q resolution constant (i.e. the detector pixel size constant).')
print('Multiplication factor for the voxel size:  pad_size/original_size')

###########################################################################################
# interpolate the object in order to keep the q resolution (detector pixel size) constant #
###########################################################################################
newz, newy, newx = np.meshgrid(np.arange(-nz//2, nz//2, 1)*voxel_size,
                               np.arange(-ny//2, ny//2, 1)*voxel_size,
                               np.arange(-nx//2, nx//2, 1)*voxel_size, indexing='ij')

print('Voxel size for keeping pixel detector size constant', voxel_size*pad_size[0]/nz, 'nm\n')

rgi = RegularGridInterpolator((np.arange(-nz//2, nz//2)*voxel_size*pad_size[0]/nz,
                               np.arange(-ny//2, ny//2)*voxel_size*pad_size[1]/ny,
                               np.arange(-nx//2, nx//2)*voxel_size*pad_size[2]/nx),
                              original_obj, method='linear', bounds_error=False, fill_value=0)

obj = rgi(np.concatenate((newz.reshape((1, newz.size)), newy.reshape((1, newz.size)),
                          newx.reshape((1, newz.size)))).transpose())
obj = obj.reshape((nz, ny, nx)).astype(original_obj.dtype)

if debug:
    gu.multislices_plot(abs(obj), sum_frames=True, invert_yaxis=True, cmap=my_cmap,
                        title='Orthogonal support interpolated for padding compensation\n')
    if orthogonal_frame:
        data = fftshift(abs(fftn(original_obj)) ** 2)
        data = data / data.sum() * photon_number  # convert into photon number
        gu.multislices_plot(data, sum_frames=False, scale='log', plot_colorbar=True, vmin=-5, invert_yaxis=False,
                            cmap=my_cmap, reciprocal_space=True, is_orthogonal=False, title='FFT before padding\n')
else:
    del original_obj
    gc.collect()

###################################################
# interpolate the object back into detector frame #
###################################################
if not orthogonal_frame:
    if debug:
        original_obj = detector_frame(myobj=original_obj, energy=en, outofplane=outofplane_angle, inplane=inplane_angle,
                                      tilt=tilt_angle, myrocking_angle=rocking_angle, mygrazing_angle=grazing_angle,
                                      distance=original_sdd, pixel_x=pixel_size, pixel_y=pixel_size, geometry=setup,
                                      voxelsize=voxel_size, debugging=debug, title='Original object')
        data = fftshift(abs(fftn(original_obj)) ** 2)
        gu.multislices_plot(data, sum_frames=False, scale='log', plot_colorbar=True, vmin=-5, invert_yaxis=False,
                            cmap=my_cmap, reciprocal_space=True, is_orthogonal=False, title='FFT before padding\n')
        del original_obj, data
        gc.collect()

    obj = detector_frame(myobj=obj, energy=en, outofplane=outofplane_angle, inplane=inplane_angle, tilt=tilt_angle,
                         myrocking_angle=rocking_angle, mygrazing_angle=grazing_angle, distance=original_sdd,
                         pixel_x=pixel_size, pixel_y=pixel_size, geometry=setup, voxelsize=voxel_size, debugging=True,
                         title='Rescaled object')

    #################################################################
    # uncomment this if you want to save the non-orthogonal support #
    # in that case pad_size and crop_size should be identical       #
    #################################################################
    # support = abs(obj)
    # support = support / support.max()
    # support[support < 0.05] = 0
    # support[np.nonzero(support)] = 1
    # np.savez_compressed(datadir + 'S' + str(scan) + 'support_nonortho400.npz', obj=support)

##############################################################
# pad the array (after interpolation because of memory cost) #
##############################################################
nz_pad, ny_pad, nx_pad = pad_size
if nz_pad < nz or ny_pad < ny or nx_pad < nx:
    print('Pad size smaller than initial array size')
    sys.exit()

newobj = pu.crop_pad(obj, pad_size)

nz, ny, nx = newobj.shape
print("Padded data size: (", nz, ',', ny, ',', nx, ')')
comment = comment + "_pad_" + str(nz) + "," + str(ny) + "," + str(nx)
del obj
gc.collect()

#####################################
# calculate the diffraction pattern #
#####################################
data = fftshift(abs(fftn(newobj))**2)
gu.multislices_plot(data, sum_frames=False,  scale='log', plot_colorbar=True, vmin=-5, invert_yaxis=False,
                    cmap=my_cmap, reciprocal_space=True, is_orthogonal=False,
                    title='FFT for initial detector distance\n')
del newobj
gc.collect()

#################################################################################
# interpolate the diffraction pattern to accomodate change in detector distance #
#################################################################################
comment = comment + '_sdd_' + str(simulated_sdd)
print('\nCurrent detector pixel size', pixel_size, 'm')
print('New detector pixel size to compensate the change in detector distance',
      str('{:.5f}'.format(pixel_size * original_sdd / simulated_sdd)), 'm')
# if the detector is 2 times farther away, the pixel size is two times smaller (2 times better sampling)
# the 3D dataset is a stack along the first axis of 2D detector images

print('Reciprocal space resolution before detector distance change (z, y, x): (', str('{:.5f}'.format(dqz)), 'A-1,',
      str('{:.5f}'.format(dqy)), 'A-1,', str('{:.5f}'.format(dqx)), 'A-1 )')
print('q range before detector distance change (z, y, x): (', str('{:.5f}'.format(dqz*nz)), 'A-1,',
      str('{:.5f}'.format(dqy*ny)), 'A-1,', str('{:.5f}'.format(dqx*nx)), 'A-1 )')
voxelsize_z = 2 * np.pi / (nz * dqz * 10)  # in nm
voxelsize_y = 2 * np.pi / (ny * dqy * 10)  # in nm
voxelsize_x = 2 * np.pi / (nx * dqx * 10)  # in nm
print('Voxel sizes before detector distance change (z, y, x): (', str('{:.2f}'.format(voxelsize_z)), 'nm,',
      str('{:.2f}'.format(voxelsize_y)), 'nm,', str('{:.2f}'.format(voxelsize_x)), 'nm)\n')

dqz_simu, dqy_simu, dqx_simu = dqz*original_sdd/simulated_sdd,\
                               dqy*original_sdd/simulated_sdd,\
                               dqx*original_sdd/simulated_sdd

if original_sdd != simulated_sdd:
    print('Reciprocal space resolution after detector distance change (z, y, x): (', str('{:.5f}'.format(dqz_simu)),
          'A-1,', str('{:.5f}'.format(dqy_simu)), 'A-1,', str('{:.5f}'.format(dqx_simu)), 'A-1 )')
    print('q range after detector distance change (z, y, x): (', str('{:.5f}'.format(dqz_simu*nz)), 'A-1,',
          str('{:.5f}'.format(dqy_simu*ny)), 'A-1,', str('{:.5f}'.format(dqx_simu*nx)), 'A-1 )')
    voxelsize_z = 2 * np.pi / (nz * dqz_simu * 10)  # in nm
    voxelsize_y = 2 * np.pi / (ny * dqy_simu * 10)  # in nm
    voxelsize_x = 2 * np.pi / (nx * dqx_simu * 10)  # in nm
    print('Voxel sizes after detector distance change (z, y, x): (', str('{:.2f}'.format(voxelsize_z)), 'nm,',
          str('{:.2f}'.format(voxelsize_y)), 'nm,', str('{:.2f}'.format(voxelsize_x)), 'nm)\n')

    newz, newy, newx = np.meshgrid(np.arange(-nz//2, nz//2, 1)*dqz,
                                   np.arange(-ny//2, ny//2, 1)*dqy,
                                   np.arange(-nx//2, nx//2, 1)*dqx, indexing='ij')

    rgi = RegularGridInterpolator((np.arange(-nz//2, nz//2)*dqz*simulated_sdd/original_sdd,
                                   np.arange(-ny//2, ny//2)*dqy*simulated_sdd/original_sdd,
                                   np.arange(-nx//2, nx//2)*dqx*simulated_sdd/original_sdd),
                                  data, method='linear', bounds_error=False, fill_value=0)

    simu_data = rgi(np.concatenate((newz.reshape((1, newz.size)), newy.reshape((1, newz.size)),
                                   newx.reshape((1, newz.size)))).transpose())
    simu_data = simu_data.reshape((nz, ny, nx)).astype(data.dtype)
    gu.multislices_plot(simu_data, sum_frames=False,  scale='log', plot_colorbar=True, vmin=-5, invert_yaxis=False,
                        cmap=my_cmap, reciprocal_space=True, is_orthogonal=False,
                        title='FFT for simulated detector distance\n')
else:
    simu_data = data
del data
gc.collect()
#######################################################
# convert into photons and apply the photon threshold #
#######################################################
simu_data = simu_data / simu_data.sum() * photon_number  # convert into photon number

mask = np.zeros((nz, ny, nx))
mask[simu_data <= photon_threshold] = 1
simu_data[simu_data <= photon_threshold] = 0

gu.multislices_plot(simu_data, sum_frames=False,  scale='log', plot_colorbar=True, vmin=-5, invert_yaxis=False,
                    cmap=my_cmap, reciprocal_space=True, is_orthogonal=False, title='FFT converted into photons\n')
if save_fig:
    plt.savefig(datadir + 'S' + str(scan) + '_diff_float_' + str('{:.0e}'.format(photon_number))+comment + '_sum.png')

#########################
# include Poisson noise #
#########################
if include_noise:
    simu_data = np.rint(poisson(simu_data)).astype(int)
    comment = comment + "_noise"
else:
    simu_data = np.rint(simu_data).astype(int)

#####################
# add detector gaps #
#####################
if set_gap:
    simu_data, mask = mask3d_maxipix(simu_data, mask, start_pixel=gap_pixel_start, width_gap=gap_width)

gu.multislices_plot(simu_data, sum_frames=False,  scale='log', plot_colorbar=True, vmin=-1, invert_yaxis=False,
                    cmap=my_cmap, reciprocal_space=True, is_orthogonal=False, title='After rounding')

myfig, _, _ = gu.multislices_plot(simu_data, sum_frames=True,  scale='log', plot_colorbar=True, vmin=-1,
                                  invert_yaxis=False, cmap=my_cmap, reciprocal_space=True, is_orthogonal=False,
                                  title='Masked intensity')
myfig.text(0.60, 0.30, "Pad size =" + str(pad_size), size=20)
if save_fig:
    myfig.savefig(datadir + 'S' + str(scan) + '_diff_' + str('{:.0e}'.format(photon_number))+comment + '_sum.png')

#################################################
# crop arrays to obtain the final detector size #
#################################################
voxelsizez_crop = 2 * np.pi / (crop_size[0] * dqz_simu * 10)  # in nm
voxelsizey_crop = 2 * np.pi / (crop_size[1] * dqy_simu * 10)  # in nm
voxelsizex_crop = 2 * np.pi / (crop_size[2] * dqx_simu * 10)  # in nm
print('Real-space voxel sizes (z, y, x) after cropping: (', str('{:.2f}'.format(voxelsizez_crop)), 'nm,',
      str('{:.2f}'.format(voxelsizey_crop)), 'nm,', str('{:.2f}'.format(voxelsizex_crop)), 'nm )')

nz, ny, nx = simu_data.shape
nz_crop, ny_crop, nx_crop = crop_size
if nz < nz_crop or ny < ny_crop or nx < nx_crop:
    print('Crop size larger than initial array size')
    sys.exit()

simu_data = pu.crop_pad(simu_data, crop_size)
mask = pu.crop_pad(mask, crop_size)

##########################################################
# crop arrays to fulfill FFT requirements during phasing #
##########################################################
nz, ny, nx = simu_data.shape
nz_crop, ny_crop, nx_crop = pru.smaller_primes((nz, ny, nx), maxprime=7, required_dividers=(2,))

simu_data = pu.crop_pad(simu_data, (nz_crop, ny_crop, nx_crop))
mask = pu.crop_pad(mask, (nz_crop, ny_crop, nx_crop))

nz, ny, nx = simu_data.shape
print("cropped FFT data size:", simu_data.shape)
print("Total number of photons:", simu_data.sum())
comment = comment + "_crop_" + str(nz) + "," + str(ny) + "," + str(nx)

##############
# save files #
##############
if save_data:
    np.savez_compressed(datadir+'S'+str(scan)+'_diff_' + str('{:.0e}'.format(photon_number))+comment, data=simu_data)
    np.savez_compressed(datadir+'S'+str(scan)+'_mask_' + str('{:.0e}'.format(photon_number))+comment, mask=mask)

#####################################
# plot mask and diffraction pattern #
#####################################
plt.ioff()
if debug:
    gu.multislices_plot(mask, sum_frames=True, scale='linear', plot_colorbar=False, invert_yaxis=False,
                        cmap=my_cmap, reciprocal_space=True, is_orthogonal=False, title='Mask')

myfig, _, _ = gu.multislices_plot(simu_data, sum_frames=False,  scale='log', plot_colorbar=True, vmin=-1,
                                  invert_yaxis=False, cmap=my_cmap, reciprocal_space=True, is_orthogonal=False,
                                  title='Masked intensity')
myfig.text(0.60, 0.30, "Pad size =" + str(pad_size), size=20)
myfig.text(0.60, 0.25, "Crop size =" + str(crop_size), size=20)
myfig.text(0.60, 0.20, "Detector distance =" + str(simulated_sdd), size=20)
if set_gap:
    myfig.text(0.60, 0.15, "Gap width =" + str(gap_width) + "pixels", size=20)
if save_fig:
    myfig.savefig(datadir + 'S' + str(scan) + '_diff_' + str('{:.0e}'.format(photon_number))+comment + '_center.png')

myfig, _, _ = gu.multislices_plot(simu_data, sum_frames=True,  scale='log', plot_colorbar=True, vmin=-1,
                                  invert_yaxis=False, cmap=my_cmap, reciprocal_space=True, is_orthogonal=False,
                                  title='Masked intensity')
myfig.text(0.60, 0.30, "Pad size =" + str(pad_size), size=20)
myfig.text(0.60, 0.25, "Crop size =" + str(crop_size), size=20)
myfig.text(0.60, 0.20, "Detector distance =" + str(simulated_sdd), size=20)
if set_gap:
    myfig.text(0.60, 0.15, "Gap width =" + str(gap_width) + "pixels", size=20)
if save_fig:
    myfig.savefig(datadir + 'S' + str(scan) + '_diff_' + str('{:.0e}'.format(photon_number))+comment + '_sum.png')
plt.show()
