# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.measurements import center_of_mass
import os
import tkinter as tk
from tkinter import filedialog
import gc
import sys
sys.path.append('//win.desy.de/home/carnisj/My Documents/myscripts/bcdi/')
import bcdi.graph.graph_utils as gu
import bcdi.experiment.experiment_utils as exp
import bcdi.postprocessing.postprocessing_utils as pu
import bcdi.preprocessing.preprocessing_utils as pru

helptext = """
strain.py: calculate the strain component from experimental geometry

Input: complex amplitude array from a phasing program

Output: data on orthogonal frame (laboratory or crystal frame), amp_disp_strain array
        disp array should be divided by q to get the displacement (disp = -1*phase here)

Laboratory frame: z downstream, y vertical, x outboard (CXI convention)

Crystal reciprocal frame: qx downstream, qz vertical, qy outboard

Detector convention: when out_of_plane angle=0   Y=-y , when in_plane angle=0   X=x

In arrays, when plotting the first parameter is the row (vertical axis), 
and the second the column (horizontal axis).

Therefore the data structure is data[qx, qz, qy] for reciprocal space, 
or data[z, y, x] for real space
"""

scan = 1038  # spec scan number

datadir = 'D:/data/HC3207/SN' + str(scan) + "/pynxraw/"

sort_method = 'variance/mean'  # 'mean_amplitude' or 'variance' or 'variance/mean' or 'volume', metric for averaging
correlation_threshold = 0.90

#########################################################
# parameters relative to the FFT window and voxel sizes #
#########################################################
original_size = [100, 400, 512]  # size of the FFT array before binning. It will be modify to take into account binning
# during phasing automatically. Leave it to () if the shape did not change.
binning = (1, 1, 1)  # binning factor during phasing
output_size = (100, 100, 100)  # (z, y, x) Fix the size of the output array, leave it as () otherwise
keep_size = False  # set to True to keep the initial array size for orthogonalization (slower)
fix_voxel = 6.0  # in nm, put np.nan to use the default voxel size (mean of the voxel sizes in 3 directions)
plot_margin = (60, 30, 30)  # (z, y, x) margin outside the support in each direction, can be negative
# useful to avoid cutting the object during the orthogonalization
############################################
# parameters related to strain calculation #
############################################
isosurface_strain = 0.43  # threshold use for removing the outer layer (strain is undefined at the exact surface voxel)
isosurface_method = 'threshold'  # 'threshold' or 'defect'
phase_offset = 0  # manual offset to add to the phase, should be 0 in most cases
centering_method = 'max_com'  # 'com' (center of mass), 'max', 'max_com' (max then com), 'do_nothing'
# TODO: where is q for energy scans? Should we just rotate the reconstruction to have q along one axis,
#  instead of using sample offsets?
comment = "_unwrap_" + isosurface_method + "_iso_" + str(isosurface_strain)  # should start with _
#################################
# define the experimental setup #
#################################
beamline = "ID01"  # name of the beamline, used for data loading and normalization by monitor
# supported beamlines: 'ID01', 'SIXS_2018', 'SIXS_2019', 'CRISTAL', 'P10'
rocking_angle = "outofplane"  # "outofplane" or "inplane", does not matter for energy scan
#  "inplane" e.g. phi @ ID01, mu @ SIXS "outofplane" e.g. eta @ ID01
sdd = 0.86180  # sample to detector distance in m
pixel_size = 55e-6  # detector pixel size in m
energy = 8994  # x-ray energy in eV, 6eV offset at ID01
beam_direction = np.array([1, 0, 0])  # beam along z
outofplane_angle = 35.0502  # detector delta ID01, delta SIXS, gamma 34ID
inplane_angle = 3.5646  # detector nu ID01, gamma SIXS, tth 34ID
grazing_angle = 0  # in degrees, incident angle for in-plane rocking curves (eta ID01, th 34ID, beta SIXS)
tilt_angle = 0.010  # angular step size for rocking angle, eta ID01, mu SIXS, does not matter for energy scan
correct_refraction = False  # True for correcting the phase shift due to refraction
correct_absorption = False  # True for correcting the amplitude for absorption
dispersion = 4.1184E-05  # delta
# Pt:  3.2880E-05 @ 9994eV, 4.1184E-05 @ 8994keV, 5.2647E-05 @ 7994keV, 4.6353E-05 @ 8500eV / Ge 1.4718E-05 @ 8keV
absorption = 3.4298E-06  # beta
# Pt:  2.3486E-06 @ 9994eV, 3.4298E-06 @ 8994keV, 5.2245E-06 @ 7994keV, 4.1969E-06 @ 8500eV
threshold_refraction = 0.05  # threshold used to calculate the optical path
# the threshold for refraction/absorption corrections should be low, to correct for an object larger than the real one,
# otherwise it messes up the phase
##########################################
# parameteres for temperature estimation #
##########################################
get_temperature = False
reflection = np.array([1, 1, 1])  # measured reflection, use for estimating the temperature
reference_spacing = None  # for calibrating the thermal expansion, if None it is fixed to 3.9236/norm(reflection) Pt
reference_temperature = None  # used to calibrate the thermal expansion, if None it is fixed to 293.15K (RT)
###########
# options #
###########
simu_flag = False  # set to True if it is simulation, the parameter invert_phase will be set to 0
invert_phase = True  # True for the displacement to have the right sign (FFT convention), False only for simulations
flip_reconstruction = False  # True if you want to get the conjugate object
phase_ramp_removal = 'gradient'  # 'gradient' or 'upsampling'
threshold_gradient = 0.3  # upper threshold of the gradient of the phase, use for ramp removal
xrayutils_ortho = False  # True if the data is already orthogonalized
save_raw = False  # True to save the amp-phase.vti before orthogonalization
save_support = False  # True to save the non-orthogonal support for later phase retrieval
save_labframe = False  # True to save the data in the laboratory frame (before rotations)
save = True  # True to save amp.npz, phase.npz, strain.npz and vtk files
debug = False  # set to True to show all plots for debugging
roll_modes = (0, 1, 0)  # correct a roll of few pixels after the decomposition into modes in PyNX. axis=(0, 1, 2)
############################################
# setup for phase averaging or apodization #
############################################
hwidth = 0  # (width-1)/2 of the averaging window for the phase, 0 means no averaging
apodize_flag = False  # True to multiply the diffraction pattern by a filtering window
apodize_window = 'blackman'  # filtering window, multivariate 'normal' or 'tukey' or 'blackman'
mu = np.array([0.0, 0.0, 0.0])  # mu of the gaussian window
sigma = np.array([0.30, 0.30, 0.30])  # sigma of the gaussian window
alpha = np.array([1.0, 1.0, 1.0])  # shape parameter of the tukey window
############################################
# parameters related to data visualization #
############################################
align_crystal = True  # if True rotates the crystal to align it along q
ref_axis_outplane = "y"  # "y"  # "z"  # q will be aligned along that axis
align_inplane = False  # if True rotates afterwards the crystal inplane to align it along z for easier slicing
ref_axis_inplane = "x"  # "x"  # will align inplane_normal to that axis
inplane_normal = np.array([1, 0, -0.08])  # facet normal to align with ref_axis_inplane (y should be 0)
strain_range = 0.003  # for plots
phase_range = np.pi  # for plots
grey_background = True  # True to set the background to grey in phase and strain plots
tick_spacing = 50  # for plots, in nm
tick_direction = 'inout'  # 'out', 'in', 'inout'
tick_length = 3  # 10  # in plots
tick_width = 1  # 2  # in plots
##################################
# end of user-defined parameters #
##################################

####################
# Check parameters #
####################
if simu_flag:
    invert_phase = False
    correct_absorption = 0
    correct_refraction = 0
if invert_phase:
    phase_fieldname = 'disp'
else:
    phase_fieldname = 'phase'

###################
# define colormap #
###################
if grey_background:
    bad_color = '0.7'
else:
    bad_color = '1.0'  # white background
colormap = gu.Colormap(bad_color=bad_color)
my_cmap = colormap.cmap

####################################
# define the experimental geometry #
####################################
pixel_size = pixel_size * binning[1]
tilt_angle = tilt_angle * binning[0]
original_size = tuple([original_size[index] // binning[index] for index in range(len(binning))])


if binning[1] != binning[2]:
    print('Binning size different for each detector direction - not yet implemented')
    sys.exit()

setup = exp.SetupPostprocessing(beamline=beamline, energy=energy, outofplane_angle=outofplane_angle,
                                inplane_angle=inplane_angle, tilt_angle=tilt_angle, rocking_angle=rocking_angle,
                                grazing_angle=grazing_angle, distance=sdd, pixel_x=pixel_size, pixel_y=pixel_size)

################
# preload data #
################
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilenames(initialdir=datadir,
                                        filetypes=[("NPZ", "*.npz"),
                                                   ("NPY", "*.npy"), ("CXI", "*.cxi"), ("HDF5", "*.h5")])
nbfiles = len(file_path)
plt.ion()

obj, extension = pu.load_reconstruction(file_path[0])

if extension == '.h5':
    comment = comment + '_mode'

if len(original_size) != 0:
    print("Original FFT window size: ", original_size)
    print("Padding back to original FFT size")
    obj = pu.crop_pad(array=obj, output_shape=original_size)
else:
    original_size = obj.shape
nz, ny, nx = obj.shape
print("Initial data size: (", nz, ',', ny, ',', nx, ')')

###########################################################################
# define range for orthogonalization and plotting - speed up calculations #
###########################################################################
zrange, yrange, xrange =\
    pu.find_datarange(array=obj, plot_margin=plot_margin, amplitude_threshold=0.1, keep_size=keep_size)

numz = zrange * 2
numy = yrange * 2
numx = xrange * 2
print("Data shape used for orthogonalization and plotting: (", numz, ',', numy, ',', numx, ')')

####################################################################################
# find the best reconstruction from the list, based on mean amplitude and variance #
####################################################################################
if nbfiles > 1:
    print('\nTrying to find the best reconstruction')
    print('Sorting by ', sort_method)
    sorted_obj = pu.sort_reconstruction(file_path=file_path, amplitude_threshold=isosurface_strain,
                                        data_range=(zrange, yrange, xrange), sort_method='sort_method')
else:
    sorted_obj = [0]

#######################################
# load reconstructions and average it #
#######################################
avg_obj = np.zeros((numz, numy, numx))
ref_obj = np.zeros((numz, numy, numx))
avg_counter = 1
print('\nAveraging using', nbfiles, 'candidate reconstructions')
for ii in sorted_obj:
    obj, extension = pu.load_reconstruction(file_path[ii])
    print('\nOpening ', file_path[ii])

    if flip_reconstruction:
        obj = pu.flip_reconstruction(obj, debugging=True)

    if extension == '.h5':
        centering_method = 'do_nothing'  # do not center, data is already cropped just on support for mode decomposition
        # coorect a roll after the decomposition into modes in PyNX
        obj = np.roll(obj, roll_modes, axis=(0, 1, 2))

    # use the range of interest defined above
    obj = pu.crop_pad(obj, [2 * zrange, 2 * yrange, 2 * xrange], debugging=False)

    # align with average reconstruction
    if avg_obj.sum() == 0:  # the fist array loaded will serve as reference object
        print('This reconstruction will serve as reference object.')
        ref_obj = obj
        avg_obj = obj
    else:
        avg_obj, flag_avg = pu.align_obj(avg_obj=avg_obj, ref_obj=ref_obj, obj=obj, support_threshold=0.25,
                                         correlation_threshold=0.90, aligning_option='dft')
        avg_counter = avg_counter + flag_avg

avg_obj = avg_obj / avg_counter
print('\nAverage performed over ', avg_counter, 'reconstructions\n')
del obj, ref_obj
gc.collect()

################
# unwrap phase #
################
phase, extent_phase = pu.unwrap(avg_obj, support_threshold=0.05, debugging=debug)

print('Extent of the phase over an extended support (ceil(phase range))~ ', int(extent_phase), '(rad)')
phase = pru.wrap(phase, start_angle=-extent_phase/2, range_angle=extent_phase)
if debug:
    gu.multislices_plot(phase, width_z=2*zrange, width_y=2*yrange, width_x=2*xrange,
                        invert_yaxis=False, plot_colorbar=True, title='Phase after unwrap + wrap')
#############################################
# phase ramp removal before phase filtering #
#############################################
amp, phase, rampz, rampy, rampx = pu.remove_ramp(amp=abs(avg_obj), phase=phase, initial_shape=original_size,
                                                 method=phase_ramp_removal, amplitude_threshold=isosurface_strain,
                                                 gradient_threshold=threshold_gradient)
del avg_obj
gc.collect()

if debug:
    gu.multislices_plot(phase, width_z=2*zrange, width_y=2*yrange, width_x=2*xrange,
                        invert_yaxis=False, plot_colorbar=True, title='Phase after ramp removal')

#######################################
# phase offset removal (at COM value) #
#######################################
support = np.zeros(amp.shape)
support[amp > isosurface_strain*amp.max()] = 1
zcom, ycom, xcom = center_of_mass(support)
print("COM at (z, y, x): (", str('{:.2f}'.format(zcom)), ',', str('{:.2f}'.format(ycom)), ',',
      str('{:.2f}'.format(xcom)), ')')
print("Phase offset at COM(amp) of:", str('{:.2f}'.format(phase[int(zcom), int(ycom), int(xcom)])), "rad")

phase = phase - phase[int(zcom), int(ycom), int(xcom)]

phase = pru.wrap(obj=phase, start_angle=-extent_phase/2, range_angle=extent_phase)

if debug:
    gu.multislices_plot(phase, width_z=2*zrange, width_y=2*yrange, width_x=2*xrange,
                        invert_yaxis=False, plot_colorbar=True, title='Phase after offset removal')

print("Mean phase:", phase[support == 1].mean(), "rad")
phase = phase - phase[support == 1].mean() + phase_offset
del support, zcom, ycom, xcom
gc.collect()

phase = pru.wrap(obj=phase, start_angle=-extent_phase/2, range_angle=extent_phase)
if debug:
    gu.multislices_plot(phase, width_z=2*zrange, width_y=2*yrange, width_x=2*xrange,
                        invert_yaxis=False, plot_colorbar=True, title='Phase after mean removal')

##############################################################################
# average the phase over a window or apodize to reduce noise in strain plots #
##############################################################################
if hwidth != 0:
    bulk = pu.find_bulk(amp=amp, support_threshold=isosurface_strain, method=isosurface_method)
    # the phase should be averaged only in the support defined by the isosurface
    phase = pu.mean_filter(phase=phase, support=bulk, half_width=hwidth)
    del bulk
    gc.collect()

comment = comment + "_avg" + str(2*hwidth+1)

gridz, gridy, gridx = np.meshgrid(np.arange(0, numz, 1), np.arange(0, numy, 1), np.arange(0, numx, 1), indexing='ij')

phase = phase + gridz * rampz + gridy * rampy + gridx * rampx  # put back the phase ramp otherwise the diffraction
# pattern will be shifted and the prtf messed up

if apodize_flag:
    amp, phase = pu.apodize(amp=amp, phase=phase, initial_shape=original_size, window_type=apodize_window,
                            sigma=sigma, mu=mu, alpha=alpha, debugging=True)
    comment = comment + '_apodize_' + apodize_window

####################################################################################################################
# save the phase with the ramp for PRTF calculations, otherwise the object will be misaligned with the measurement #
####################################################################################################################
np.savez_compressed(datadir + 'S' + str(scan) + '_avg_obj_prtf' + comment, obj=amp * np.exp(1j * phase))

####################################################
# remove again phase ramp before orthogonalization #
####################################################
phase = phase - gridz * rampz - gridy * rampy - gridx * rampx

avg_obj = amp * np.exp(1j * phase)  # here the phase is again wrapped in [-pi pi[

del amp, phase, gridz, gridy, gridx, rampz, rampy, rampx
gc.collect()

######################
# centering of array #
######################
if centering_method is 'max':
    avg_obj = pu.center_max(avg_obj)
    # shift based on max value, required if it spans across the edge of the array before COM
elif centering_method is 'com':
    avg_obj = pu.center_com(avg_obj)
elif centering_method is 'max_com':
    avg_obj = pu.center_max(avg_obj)
    avg_obj = pu.center_com(avg_obj)

#########################################
#  plot amp & phase, save support & vti #
#########################################
gu.multislices_plot(abs(avg_obj), width_z=2*zrange, width_y=2*yrange, width_x=2*xrange,
                    sum_frames=False, invert_yaxis=False, plot_colorbar=True, vmin=0, vmax=abs(avg_obj).max(),
                    title='Amp before orthogonalization')
if debug:
    phase, _ = pu.unwrap(avg_obj, support_threshold=0.05, debugging=True)
    gu.multislices_plot(phase, width_z=2*zrange, width_y=2*yrange, width_x=2*xrange,
                        sum_frames=False, invert_yaxis=False, plot_colorbar=True,
                        title='Unwrapped phase before orthogonalization')
    del phase
    gc.collect()

if save_support:  # to be used as starting support in phasing, hence still in the detector frame
    support = np.zeros((numz, numy, numx))
    support[abs(avg_obj)/abs(avg_obj).max() > 0.01] = 1
    # low threshold because support will be cropped by shrinkwrap during phasing
    np.savez_compressed(datadir + 'S' + str(scan) + '_support' + comment, obj=support)
    del support
    gc.collect()

if save_raw:
    np.savez_compressed(datadir + 'S' + str(scan) + '_raw_amp-phase' + comment,
                        amp=abs(avg_obj), phase=np.angle(avg_obj))

    voxel_z = setup.wavelength / (original_size[0] * abs(tilt_angle) * np.pi / 180) * 1e9  # in nm
    voxel_y = setup.wavelength * sdd / (original_size[1] * pixel_size) * 1e9  # in nm
    voxel_x = setup.wavelength * sdd / (original_size[2] * pixel_size) * 1e9  # in nm

    # save raw amp & phase to VTK
    # in VTK, x is downstream, y vertical, z inboard, thus need to flip the last axis
    gu.save_to_vti(filename=os.path.join(datadir, "S" + str(scan) + "_raw_amp-phase" + comment + ".vti"),
                   voxel_size=(voxel_z, voxel_y, voxel_x), tuple_array=(abs(avg_obj), np.angle(avg_obj)),
                   tuple_fieldnames=('amp', 'phase'), amplitude_threshold=0.01)

#######################
#  orthogonalize data #
#######################
print('\nShape before orthogonalization', avg_obj.shape)
if not xrayutils_ortho:
    obj_ortho, voxel_size = setup.orthogonalize(obj=avg_obj, initial_shape=original_size, voxel_size=fix_voxel)
    print("VTK spacing :", str('{:.2f}'.format(voxel_size)), "nm")
    if True:
        phase, _ = pu.unwrap(obj_ortho, support_threshold=0.05, debugging=True)
        gu.multislices_plot(abs(obj_ortho), width_z=2 * zrange, width_y=2 * yrange, width_x=2 * xrange,
                            sum_frames=False, invert_yaxis=True, plot_colorbar=True, vmin=0, vmax=abs(obj_ortho).max(),
                            title='Amp after orthogonalization')
        gu.multislices_plot(phase, width_z=2 * zrange, width_y=2 * yrange, width_x=2 * xrange,
                            sum_frames=False, invert_yaxis=True, plot_colorbar=True,
                            title='Unwrapped phase after orthogonalization')
        del phase
        gc.collect()

else:  # data already orthogonalized using xrayutilities, will be in crystal frame
    obj_ortho = avg_obj
    try:
        print("Select the file containing QxQzQy")
        file_path = filedialog.askopenfilename(title="Select the file containing QxQzQy",
                                               initialdir=datadir, filetypes=[("NPZ", "*.npz")])
        npzfile = np.load(file_path)
        qx = npzfile['qx']
        qy = npzfile['qy']
        qz = npzfile['qz']
    except FileNotFoundError:
        print('Voxel size unknown')
        sys.exit()
    dy_real = 2 * np.pi / abs(qz.max() - qz.min()) / 10  # in nm qz=y in nexus convention
    dx_real = 2 * np.pi / abs(qy.max() - qy.min()) / 10  # in nm qy=x in nexus convention
    dz_real = 2 * np.pi / abs(qx.max() - qx.min()) / 10  # in nm qx=z in nexus convention
    if fix_voxel != 0:
        voxel_size = fix_voxel
    else:
        voxel_size = np.mean([dz_real, dy_real, dx_real])  # in nm
    print('real space pixel size: ', str('{:.2f}'.format(voxel_size)), 'nm')
    print('Use the same voxel size in each dimensions: interpolating...\n\n')
    obj_ortho = pu.regrid(obj_ortho, (dz_real, dy_real, dx_real), voxel_size)
del avg_obj
gc.collect()

##################################################
# calculate q, kin , kout from angles and energy #
##################################################
if ref_axis_outplane == "x":
    myaxis = np.array([1, 0, 0])  # must be in [x, y, z] order
elif ref_axis_outplane == "y":
    myaxis = np.array([0, 1, 0])  # must be in [x, y, z] order
elif ref_axis_outplane == "z":
    myaxis = np.array([0, 0, 1])  # must be in [x, y, z] order
else:
    ref_axis_outplane = "y"
    myaxis = np.array([0, 1, 0])  # must be in [x, y, z] order

kin = 2*np.pi/setup.wavelength * beam_direction  # in laboratory frame z downstream, y vertical, x outboard
kout = setup.exit_wavevector()  # in laboratory frame z downstream, y vertical, x outboard

q = kout - kin
Qnorm = np.linalg.norm(q)
q = q / Qnorm
angle = pu.plane_angle(ref_plane=np.array([q[2], q[1], q[0]]), plane=myaxis)
print("Angle between q and", ref_axis_outplane, "=", angle, "deg")
print("Angle with y in zy plane", np.arctan(q[0]/q[1])*180/np.pi, "deg")
print("Angle with y in xy plane", np.arctan(-q[2]/q[1])*180/np.pi, "deg")
print("Angle with z in xz plane", 180+np.arctan(q[2]/q[0])*180/np.pi, "deg")

if xrayutils_ortho:  # transform kin and kout back into the crystal frame (xrayutilities output in crystal frame)
    kin = pu.rotate_vector(vector=np.array([kin[2], kin[1], kin[0]]), axis_to_align=myaxis,
                           reference_axis=np.array([q[2], q[1], q[0]]))
    kout = pu.rotate_vector(vector=np.array([kout[2], kout[1], kout[0]]), axis_to_align=myaxis,
                            reference_axis=np.array([q[2], q[1], q[0]]))

Qnorm = Qnorm * 1e-10  # switch to angstroms
planar_dist = 2*np.pi/Qnorm  # Qnorm should be in angstroms
print("Wavevector transfer [z, y, x]:", q*Qnorm)
print("Wavevector transfer: (angstroms)", str('{:.4f}'.format(Qnorm)))
print("Atomic plane distance: (angstroms)", str('{:.4f}'.format(planar_dist)), "angstroms")

if get_temperature:
    temperature = pu.bragg_temperature(spacing=planar_dist, reflection=reflection, spacing_ref=reference_spacing,
                                       temperature_ref=reference_temperature, use_q=0, material="Pt")
planar_dist = planar_dist / 10  # switch to nm

######################
# centering of array #
######################
obj_ortho = pu.center_com(obj_ortho)
amp = abs(obj_ortho)
phase, extent_phase = pu.unwrap(obj_ortho, support_threshold=0.05, debugging=debug)
del obj_ortho
gc.collect()

if debug:
    gu.multislices_plot(amp, width_z=2 * zrange, width_y=2 * yrange, width_x=2 * xrange,
                        sum_frames=False, invert_yaxis=True, plot_colorbar=True, vmin=0, vmax=amp.max(),
                        title='Amp before absorption correction')
    gu.multislices_plot(phase, width_z=2 * zrange, width_y=2 * yrange, width_x=2 * xrange,
                        sum_frames=False, invert_yaxis=True, plot_colorbar=True,
                        title='Unwrapped phase before refraction correction')

#############################################
# invert phase: -1*phase = displacement * q #
#############################################
if invert_phase:
    phase = -1 * phase

########################################
# refraction and absorption correction #
########################################
if correct_refraction or correct_absorption:
    bulk = pu.find_bulk(amp=amp, support_threshold=threshold_refraction, method='threshold')

    # calculate the optical path of the incoming wavevector: it may be aligned with orthogonalized axis 0
    path_in = pu.get_opticalpath(support=bulk, direction="in", k=kin, debugging=True)

    # calculate the optical path of the outgoing wavevector: it is not aligned with any orthogonalized axis
    path_out = pu.get_opticalpath(support=bulk, direction="out", k=kout, debugging=True)

    del bulk
    gc.collect()

    optical_path = voxel_size * (path_in + path_out)  # in nm, in the laboratory frame
    del path_in, path_out
    gc.collect()

    if correct_refraction:
        phase_correction = 2 * np.pi / (1e9 * setup.wavelength) * dispersion * optical_path
        phase = phase + phase_correction

        gu.multislices_plot(phase_correction, width_z=2 * zrange, width_y=2 * yrange, width_x=2 * xrange,
                            sum_frames=False, invert_yaxis=True, plot_colorbar=True, vmin=0, vmax=np.pi/2,
                            title='Refraction correction')

    if correct_absorption:
        amp_correction = np.exp(2 * np.pi / (1e9 * setup.wavelength) * absorption * optical_path)
        amp = amp * amp_correction

        gu.multislices_plot(amp_correction, width_z=2 * zrange, width_y=2 * yrange, width_x=2 * xrange,
                            sum_frames=False, invert_yaxis=True, plot_colorbar=True, vmin=1, vmax=1.1,
                            title='Absorption correction')

    del optical_path
    gc.collect()

##############################################
# phase ramp and offset removal (mean value) #
##############################################
amp, phase, _, _, _ = pu.remove_ramp(amp=amp, phase=phase, initial_shape=original_size, method=phase_ramp_removal,
                                     amplitude_threshold=isosurface_strain, gradient_threshold=threshold_gradient)

support = np.zeros(amp.shape)
support[amp > isosurface_strain*amp.max()] = 1  # better to use the support here in case of defects (impact on the mean)
phase = phase - phase[support == 1].mean()
del support
gc.collect()

phase = pru.wrap(obj=phase, start_angle=-extent_phase/2, range_angle=extent_phase)
if True:
    gu.multislices_plot(phase, width_z=2 * zrange, width_y=2 * yrange, width_x=2 * xrange,
                        sum_frames=False, invert_yaxis=True, plot_colorbar=True,
                        title='Orthogonal phase after mean removal')

################################
# save to VTK before rotations #
################################
if save_labframe:
    if invert_phase:
        np.savez_compressed(datadir + 'S' + str(scan) + "_amp" + phase_fieldname + comment + '_LAB',
                            amp=amp, displacement=phase)
    else:
        np.savez_compressed(datadir + 'S' + str(scan) + "_amp" + phase_fieldname + comment + '_LAB',
                            amp=amp, phase=phase)

    print("VTK spacing :", str('{:.2f}'.format(voxel_size)), "nm")
    # save amp & phase to VTK before rotation in crystal frame
    # in VTK, x is downstream, y vertical, z inboard, thus need to flip the last axis
    gu.save_to_vti(filename=os.path.join(datadir, "S"+str(scan)+"_amp-"+phase_fieldname+"_LAB"+comment+".vti"),
                   voxel_size=(voxel_size, voxel_size, voxel_size), tuple_array=(amp, phase),
                   tuple_fieldnames=('amp', phase_fieldname), amplitude_threshold=0.01)
    
############################################################################
# put back the crystal in its frame, by aligning q onto the reference axis #
############################################################################
if not xrayutils_ortho:
    print('Aligning Q along ', ref_axis_outplane, ":", myaxis)
    amp = pu.rotate_crystal(array=amp, axis_to_align=np.array([q[2], q[1], q[0]])/np.linalg.norm(q),
                            reference_axis=myaxis, debugging=True)
    phase = pu.rotate_crystal(array=phase, axis_to_align=np.array([q[2], q[1], q[0]])/np.linalg.norm(q),
                              reference_axis=myaxis, debugging=False)

################################################################
# calculate the strain depending on which axis q is aligned on #
################################################################
strain = pu.get_strain(phase=phase, planar_distance=planar_dist, voxel_size=voxel_size,
                       reference_axis=ref_axis_outplane)

# old method of A. Ulvestad
# gradz, grady, gradx = np.gradient(planar_dist/(2*np.pi)*phase, voxel_size)  # planar_dist, voxel_size in nm
# strain = q[0]*gradz + q[1]*grady + q[2]*gradx  # q is normalized

################################################################
# rotates the crystal inplane for easier slicing of the result #
################################################################
if not xrayutils_ortho:
    if align_inplane:
        align_crystal = True
        if ref_axis_inplane == "x":
            myaxis_inplane = np.array([1, 0, 0])  # must be in [x, y, z] order
        elif ref_axis_inplane == "z":
            myaxis_inplane = np.array([0, 0, 1])  # must be in [x, y, z] order
        else:
            ref_axis_inplane = "z"
            myaxis_inplane = np.array([0, 0, 1])  # must be in [x, y, z] order
        amp = pu.rotate_crystal(array=amp, axis_to_align=inplane_normal/np.linalg.norm(inplane_normal),
                                reference_axis=myaxis_inplane, debugging=True)
        phase = pu.rotate_crystal(array=phase, axis_to_align=inplane_normal/np.linalg.norm(inplane_normal),
                                  reference_axis=myaxis_inplane, debugging=False)
        strain = pu.rotate_crystal(array=strain, axis_to_align=inplane_normal/np.linalg.norm(inplane_normal),
                                   reference_axis=myaxis_inplane, debugging=False)

    if align_crystal:
        comment = comment + '_crystal-frame'
    else:
        comment = comment + '_lab-frame'
        print('Rotating back the crystal in laboratory frame')
        amp = pu.rotate_crystal(array=amp, axis_to_align=myaxis,
                                reference_axis=np.array([q[2], q[1], q[0]])/np.linalg.norm(q), debugging=True)
        phase = pu.rotate_crystal(array=phase, axis_to_align=myaxis,
                                  reference_axis=np.array([q[2], q[1], q[0]])/np.linalg.norm(q), debugging=False)
        strain = pu.rotate_crystal(array=strain, axis_to_align=myaxis,
                                   reference_axis=np.array([q[2], q[1], q[0]])/np.linalg.norm(q), debugging=False)

    print('Voxel size: ', str('{:.2f}'.format(voxel_size)), "nm")

##############################################
# pad array to fit the output_size parameter #
##############################################
if not output_size:  # output_size not defined, default to actual size
    pass
else:
    amp = pu.crop_pad(array=amp, output_shape=output_size)
    phase = pu.crop_pad(array=phase, output_shape=output_size)
    strain = pu.crop_pad(array=strain, output_shape=output_size)
numz, numy, numx = amp.shape
print("Final data shape:", numz, numy, numx)

##############################################################################################
# save result to vtk (result in the laboratory frame or rotated result in the crystal frame) #
##############################################################################################
bulk = pu.find_bulk(amp=amp, support_threshold=isosurface_strain, method=isosurface_method)
if save:
    if invert_phase:
        np.savez_compressed(datadir + 'S' + str(scan) + "_amp" + phase_fieldname + "strain" + comment,
                            amp=amp, displacement=phase, bulk=bulk, strain=strain)
    else:
        np.savez_compressed(datadir + 'S' + str(scan) + "_amp" + phase_fieldname + "strain" + comment,
                            amp=amp, phase=phase, bulk=bulk, strain=strain)

    # save amp & phase to VTK
    # in VTK, x is downstream, y vertical, z inboard, thus need to flip the last axis
    gu.save_to_vti(filename=os.path.join(datadir, "S"+str(scan)+"_amp-"+phase_fieldname+"-strain"+comment+".vti"),
                   voxel_size=(voxel_size, voxel_size, voxel_size), tuple_array=(amp, bulk, phase, strain),
                   tuple_fieldnames=('amp', 'bulk', phase_fieldname, 'strain'), amplitude_threshold=0.01)


########################
# calculate the volume #
########################
amp = amp / amp.max()
temp_amp = np.copy(amp)
temp_amp[amp < isosurface_strain] = 0
temp_amp[np.nonzero(temp_amp)] = 1
volume = temp_amp.sum()*voxel_size**3  # in nm3
del temp_amp
gc.collect()

#######################
# plot phase & strain #
#######################
pixel_spacing = tick_spacing / voxel_size

print('Phase extent before and after thresholding:', phase.max()-phase.min(),
      phase[np.nonzero(bulk)].max()-phase[np.nonzero(bulk)].min())
piz, piy, pix = np.unravel_index(phase.argmax(), phase.shape)
print('phase.max() = ', phase[np.nonzero(bulk)].max(), ', at coordinates ', piz, piy, pix)
strain[bulk == 0] = np.nan
phase[bulk == 0] = np.nan
if True:
    gu.combined_plots((phase[piz, :, :], phase[:, piy, :], phase[:, :, pix]), tuple_sum_frames=False, tuple_sum_axis=0,
                      tuple_width_v=np.nan, tuple_width_h=np.nan, tuple_colorbar=True, tuple_vmin=np.nan,
                      tuple_vmax=np.nan, tuple_title=('phase at max in xy', 'phase at max in xz', 'phase at max in yz'),
                      tuple_scale='linear', cmap=my_cmap, pixel_spacing=pixel_spacing, is_orthogonal=True,
                      reciprocal_space=False)

# bulk support
fig, _, _ = gu.multislices_plot(bulk, sum_frames=False, invert_yaxis=True, title='Orthogonal bulk', vmin=0, vmax=1,
                                tick_direction=tick_direction, tick_width=tick_width, tick_length=tick_length,
                                pixel_spacing=pixel_spacing, is_orthogonal=True, reciprocal_space=False)
fig.text(0.60, 0.45, "Scan " + str(scan), size=20)
fig.text(0.60, 0.40, "Bulk - isosurface=" + str('{:.2f}'.format(isosurface_strain)), size=20)
fig.text(0.60, 0.35, "Ticks spacing=" + str(tick_spacing) + "nm", size=20)
plt.pause(0.1)
if save:
    plt.savefig(
        datadir + 'S' + str(scan) + '_bulk' + comment + '.png')

# amplitude
fig, _, _ = gu.multislices_plot(amp, sum_frames=False, invert_yaxis=True, title='Normalized orthogonal amp', vmin=0,
                                vmax=1, tick_direction=tick_direction, tick_width=tick_width, tick_length=tick_length,
                                pixel_spacing=pixel_spacing, plot_colorbar=True, is_orthogonal=True,
                                reciprocal_space=False)
fig.text(0.60, 0.45, "Scan " + str(scan), size=20)
fig.text(0.60, 0.40, "Voxel size=" + str('{:.2f}'.format(voxel_size)) + "nm", size=20)
fig.text(0.60, 0.35, "Ticks spacing=" + str(tick_spacing) + "nm", size=20)
fig.text(0.60, 0.30, "Volume=" + str(int(volume)) + "nm3", size=20)
fig.text(0.60, 0.25, "Sorted by " + sort_method, size=20)
fig.text(0.60, 0.20, 'correlation threshold=' + str(correlation_threshold), size=20)
fig.text(0.60, 0.15, "Average over " + str(avg_counter) + " reconstruction(s)", size=20)
fig.text(0.60, 0.10, "Planar distance=" + str('{:.5f}'.format(planar_dist)) + "nm", size=20)
if get_temperature:
    fig.text(0.60, 0.05, "Estimated T=" + str(temperature) + "C", size=20)
if save:
    plt.savefig(datadir + 'amp_S' + str(scan) + comment + '.png')

# phase
fig, _, _ = gu.multislices_plot(phase, sum_frames=False, invert_yaxis=True, title='Orthogonal displacement',
                                vmin=-phase_range, vmax=phase_range, tick_direction=tick_direction, cmap=my_cmap,
                                tick_width=tick_width, tick_length=tick_length, pixel_spacing=pixel_spacing,
                                plot_colorbar=True, is_orthogonal=True, reciprocal_space=False)
fig.text(0.60, 0.30, "Scan " + str(scan), size=20)
fig.text(0.60, 0.25, "Voxel size=" + str('{:.2f}'.format(voxel_size)) + "nm", size=20)
fig.text(0.60, 0.20, "Ticks spacing=" + str(tick_spacing) + "nm", size=20)
fig.text(0.60, 0.15, "Average over " + str(avg_counter) + " reconstruction(s)", size=20)
if hwidth > 0:
    fig.text(0.60, 0.10, "Averaging over " + str(2*hwidth+1) + " pixels", size=20)
else:
    fig.text(0.60, 0.10, "No phase averaging", size=20)
if save:
    plt.savefig(datadir + 'displacement_S' + str(scan) + comment + '.png')

# strain
fig, _, _ = gu.multislices_plot(strain, sum_frames=False, invert_yaxis=True, title='Orthogonal strain',
                                vmin=-strain_range, vmax=strain_range, tick_direction=tick_direction,
                                tick_width=tick_width, tick_length=tick_length, plot_colorbar=True, cmap=my_cmap,
                                pixel_spacing=pixel_spacing, is_orthogonal=True, reciprocal_space=False)
fig.text(0.60, 0.30, "Scan " + str(scan), size=20)
fig.text(0.60, 0.25, "Voxel size=" + str('{:.2f}'.format(voxel_size)) + "nm", size=20)
fig.text(0.60, 0.20, "Ticks spacing=" + str(tick_spacing) + "nm", size=20)
fig.text(0.60, 0.15, "Average over " + str(avg_counter) + " reconstruction(s)", size=20)
if hwidth > 0:
    fig.text(0.60, 0.10, "Averaging over " + str(2*hwidth+1) + " pixels", size=20)
else:
    fig.text(0.60, 0.10, "No phase averaging", size=20)
if save:
    plt.savefig(datadir + 'strain_S' + str(scan) + comment + '.png')


print('End of script')
plt.ioff()
plt.show()
