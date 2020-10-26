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
sys.path.append('D:/myscripts/bcdi/')
import bcdi.graph.graph_utils as gu
import bcdi.experiment.experiment_utils as exp
import bcdi.postprocessing.postprocessing_utils as pu
import bcdi.preprocessing.preprocessing_utils as pru
import bcdi.simulation.simulation_utils as simu
import bcdi.utils.utilities as util

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

scan = 2290  # spec scan number

datadir = "D:/data/Pt THH ex-situ/Data/CH4760/S" + str(scan) + "/pynxraw/gap_interp/"

sort_method = 'variance/mean'  # 'mean_amplitude' or 'variance' or 'variance/mean' or 'volume', metric for averaging
correlation_threshold = 0.90
#########################################################
# parameters relative to the FFT window and voxel sizes #
#########################################################
original_size = [240, 450, 432]  # size of the FFT array before binning. It will be modify to take into account binning
# during phasing automatically. Leave it to () if the shape did not change.
binning = (1, 1, 1)  # binning factor applied during phasing
output_size = (200, 200, 200)  # (z, y, x) Fix the size of the output array, leave it as () otherwise
keep_size = False  # True to keep the initial array size for orthogonalization (slower), it will be cropped otherwise
fix_voxel = 3  # voxel size in nm for the interpolation during the geometrical transformation
# put np.nan to use the default voxel size (mean of the voxel sizes in 3 directions)
plot_margin = (60, 60, 60)  # (z, y, x) margin in pixel to leave outside the support in each direction when cropping,
# it can be negative. It is useful in order to avoid cutting the object during the orthogonalization.
#############################################################
# parameters related to displacement and strain calculation #
#############################################################
isosurface_strain = 0.48  # threshold use for removing the outer layer (strain is undefined at the exact surface voxel)
strain_method = 'defect'  # 'default' or 'defect'. If 'defect', will offset the phase in a loop and keep the smallest
# magnitude value for the strain. See: F. Hofmann et al. PhysRevMaterials 4, 013801 (2020)
phase_offset = 0  # manual offset to add to the phase, should be 0 in most cases
offset_origin = []  # the phase at this pixels will be set to phase_offset, leave it as [] to use offset_method instead
offset_method = 'mean'  # 'COM' or 'mean', method for removing the offset in the phase
centering_method = 'max_com'  # 'com' (center of mass), 'max', 'max_com' (max then com), 'do_nothing'
# TODO: where is q for energy scans? Should we just rotate the reconstruction to have q along one axis,
#  instead of using sample offsets?
comment = '_defect_gap_iso' + str(isosurface_strain)  # should start with _
#################################
# define the experimental setup #
#################################
beamline = "ID01"  # name of the beamline, used for data loading and normalization by monitor and orthogonalisation
# supported beamlines: 'ID01', 'SIXS_2018', 'SIXS_2019', 'CRISTAL', 'P10', '34ID'
rocking_angle = "outofplane"  # "outofplane" or "inplane", does not matter for energy scan
#  "inplane" e.g. phi @ ID01, mu @ SIXS "outofplane" e.g. eta @ ID01
sdd = 0.50678  # 1.26  # sample to detector distance in m
pixel_size = 55e-6  # detector pixel size in m, taking into account an eventual binning during preprocessing
energy = 9000  # x-ray energy in eV, 6eV offset at ID01
beam_direction = np.array([1, 0, 0])  # incident beam along z
outofplane_angle = 35.2694  # detector delta ID01, delta SIXS, gamma 34ID
inplane_angle = -2.5110  # detector nu ID01, gamma SIXS, tth 34ID
grazing_angle = 0  # in degrees, incident angle for in-plane rocking curves (eta ID01, th 34ID, beta SIXS)
tilt_angle = 0.01016  # angular step size for rocking angle, eta ID01, mu SIXS, does not matter for energy scan
correct_refraction = False  # True for correcting the phase shift due to refraction
correct_absorption = False  # True for correcting the amplitude for absorption
optical_path_method = 'threshold'  # 'threshold' or 'defect', if 'threshold' it uses isosurface_strain to define the
# support  for the optical path calculation, if 'defect' (holes) it tries to remove only outer layers even if
# the amplitude is lower than isosurface_strain inside the crystal
dispersion = 4.1184E-05  # delta
# Pt:  3.0761E-05 @ 10300eV
# 3.2880E-05 @ 9994eV, 4.1184E-05 @ 8994eV, 5.2647E-05 @ 7994eV, 4.6353E-05 @ 8500eV / Ge 1.4718E-05 @ 8keV
absorption = 3.4298E-06  # beta
# Pt:  2.0982E-06 @ 10300eV
# 2.3486E-06 @ 9994eV, 3.4298E-06 @ 8994eV, 5.2245E-06 @ 7994eV, 4.1969E-06 @ 8500eV
threshold_unwrap_refraction = 0.05  # threshold used to calculate the optical path
# the threshold for refraction/absorption corrections should be low, to correct for an object larger than the real one,
# otherwise it messes up the phase
###########
# options #
###########
simu_flag = False  # set to True if it is simulation, the parameter invert_phase will be set to 0
invert_phase = True  # True for the displacement to have the right sign (FFT convention), False only for simulations
flip_reconstruction = False  # True if you want to get the conjugate object
phase_ramp_removal = 'gradient'  # 'gradient'  # 'gradient' or 'upsampling', 'gradient' is much faster
threshold_gradient = 0.1  # upper threshold of the gradient of the phase, use for ramp removal
is_orthogonal = False  # True if the data is already orthogonalized
save_raw = False  # True to save the amp-phase.vti before orthogonalization
save_support = False  # True to save the non-orthogonal support for later phase retrieval
save_labframe = False  # True to save the data in the laboratory frame (before rotations)
save = True  # True to save amp.npz, phase.npz, strain.npz and vtk files
debug = False  # set to True to show all plots for debugging
roll_modes = (0, 0, 0)   # axis=(0, 1, 2), correct a roll of few pixels after the decomposition into modes in PyNX
############################################
# parameters related to data visualization #
############################################
align_q = True  # if True rotates the crystal to align q it along one axis of the array
ref_axis_q = "y"  # q will be aligned along that axis
align_axis = False  # if True rotates the crystal to align axis_to_align along ref_axis
ref_axis = "y"  # will align axis_to_align to that axis
axis_to_align = np.array([-0.011662456997498807, 0.957321364700986, -0.28879022106682123])
# axis to align with ref_axis in the order x y z (axis 2, axis 1, axis 0)
strain_range = 0.002  # for plots
phase_range = np.pi  # for plots
grey_background = True  # True to set the background to grey in phase and strain plots
tick_spacing = 50  # for plots, in nm
tick_direction = 'inout'  # 'out', 'in', 'inout'
tick_length = 3  # 10  # in plots
tick_width = 1  # 2  # in plots
##########################################
# parameteres for temperature estimation #
##########################################
get_temperature = False  # only available for platinum at the moment
reflection = np.array([1, 1, 1])  # measured reflection, use for estimating the temperature
reference_spacing = None  # for calibrating the thermal expansion, if None it is fixed to 3.9236/norm(reflection) Pt
reference_temperature = None  # used to calibrate the thermal expansion, if None it is fixed to 293.15K (RT)
##########################################################
# parameters for averaging several reconstructed objects #
##########################################################
avg_method = 'reciprocal_space'  # 'real_space' or 'reciprocal_space'
avg_threshold = 0.90  # minimum correlation within reconstructed object for averaging
############################################
# setup for phase averaging or apodization #
############################################
hwidth = 0  # (width-1)/2 of the averaging window for the phase, 0 means no phase averaging
apodize_flag = False  # True to multiply the diffraction pattern by a filtering window
apodize_window = 'blackman'  # filtering window, multivariate 'normal' or 'tukey' or 'blackman'
mu = np.array([0.0, 0.0, 0.0])  # mu of the gaussian window
sigma = np.array([0.30, 0.30, 0.30])  # sigma of the gaussian window
alpha = np.array([1.0, 1.0, 1.0])  # shape parameter of the tukey window
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

obj, extension = util.load_file(file_path[0])
# obj[0:40,:,:] = 0
# obj[65:,:,:] = 0
if extension == '.h5':
    comment = comment + '_mode'

nz, ny, nx = obj.shape
print("Initial data size: (", nz, ',', ny, ',', nx, ')')
if len(original_size) == 0:
    original_size = obj.shape
print("FFT size before accounting for binning", original_size)
original_size = tuple([original_size[index] // binning[index] for index in range(len(binning))])
print("Binning used during phasing:", binning)
print("Padding back to original FFT size", original_size)
obj = pu.crop_pad(array=obj, output_shape=original_size)
nz, ny, nx = obj.shape

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
                                        data_range=(zrange, yrange, xrange), sort_method='variance/mean')
else:
    sorted_obj = [0]

#######################################
# load reconstructions and average it #
#######################################
avg_obj = np.zeros((numz, numy, numx))
ref_obj = np.zeros((numz, numy, numx))
avg_counter = 1
print('\nAveraging using', nbfiles, 'candidate reconstructions')
for counter, value in enumerate(sorted_obj):
    obj, extension = util.load_file(file_path[value])
    print('\nOpening ', file_path[value])

    if flip_reconstruction:
        obj = pu.flip_reconstruction(obj, debugging=True)

    if extension == '.h5':
        centering_method = 'do_nothing'  # do not center, data is already cropped just on support for mode decomposition
        # correct a roll after the decomposition into modes in PyNX
        obj = np.roll(obj, roll_modes, axis=(0, 1, 2))
        fig, _, _ = gu.multislices_plot(abs(obj), sum_frames=True, plot_colorbar=True, title='1st mode after centering')
        fig.waitforbuttonpress()
        plt.close(fig)
    # use the range of interest defined above
    obj = pu.crop_pad(obj, [2 * zrange, 2 * yrange, 2 * xrange], debugging=False)

    # align with average reconstruction
    if counter == 0:  # the fist array loaded will serve as reference object
        print('This reconstruction will serve as reference object.')
        ref_obj = obj

    avg_obj, flag_avg = pu.average_obj(avg_obj=avg_obj, ref_obj=ref_obj, obj=obj, support_threshold=0.25,
                                       correlation_threshold=avg_threshold, aligning_option='dft',
                                       method=avg_method, debugging=True)
    avg_counter = avg_counter + flag_avg

avg_obj = avg_obj / avg_counter
print('\nAverage performed over ', avg_counter, 'reconstructions\n')
del obj, ref_obj
gc.collect()

##################################################
# calculate q, kin , kout from angles and energy #
##################################################
if ref_axis_q == "x":
    myaxis = np.array([1, 0, 0])  # must be in [x, y, z] order
elif ref_axis_q == "y":
    myaxis = np.array([0, 1, 0])  # must be in [x, y, z] order
elif ref_axis_q == "z":
    myaxis = np.array([0, 0, 1])  # must be in [x, y, z] order
else:
    ref_axis_q = "y"
    myaxis = np.array([0, 1, 0])  # must be in [x, y, z] order

kin = 2*np.pi/setup.wavelength * beam_direction  # in laboratory frame z downstream, y vertical, x outboard
kout = setup.exit_wavevector()  # in laboratory frame z downstream, y vertical, x outboard

q = kout - kin
Qnorm = np.linalg.norm(q)
q = q / Qnorm
angle = simu.angle_vectors(ref_vector=np.array([q[2], q[1], q[0]]), test_vector=myaxis)
print("Angle between q and", ref_axis_q, "=", angle, "deg")
print("Angle with y in zy plane", np.arctan(q[0]/q[1])*180/np.pi, "deg")
print("Angle with y in xy plane", np.arctan(-q[2]/q[1])*180/np.pi, "deg")
print("Angle with z in xz plane", 180+np.arctan(q[2]/q[0])*180/np.pi, "deg")
Qnorm = Qnorm * 1e-10  # switch to angstroms
planar_dist = 2*np.pi/Qnorm  # Qnorm should be in angstroms
print("Normalized wavevector transfer [z, y, x]:", q)
print("Wavevector transfer: (angstroms)", str('{:.4f}'.format(Qnorm)))
print("Atomic plane distance: (angstroms)", str('{:.4f}'.format(planar_dist)), "angstroms")

temperature = None
if get_temperature:
    temperature = pu.bragg_temperature(spacing=planar_dist, reflection=reflection, spacing_ref=reference_spacing,
                                       temperature_ref=reference_temperature, use_q=False, material="Pt")
planar_dist = planar_dist / 10  # switch to nm

if is_orthogonal:  # transform kin and kout back into the crystal frame (xrayutilities output in crystal frame)
    # kin and kout are used for the estimation of the optical path (refraction correction)
    kin = pu.rotate_vector(vector=np.array([kin[2], kin[1], kin[0]]), axis_to_align=myaxis,
                           reference_axis=np.array([q[2], q[1], q[0]]))
    kout = pu.rotate_vector(vector=np.array([kout[2], kout[1], kout[0]]), axis_to_align=myaxis,
                            reference_axis=np.array([q[2], q[1], q[0]]))
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
    amp = abs(obj_ortho)
    phase = np.angle(obj_ortho)
    
else:  # transform back q in the detector frame, we need it to align q along one the array axis and calculate the strain
    q_lab = setup.orthogonalize_vector(vector=q, array_shape=original_size, tilt_angle=tilt_angle, pixel_x=pixel_size,
                                       pixel_y=pixel_size)
    print('\nq_lab = ', q_lab)
    print('Aligning q_lab along ', ref_axis_q, ":", myaxis)
    amp = pu.rotate_crystal(array=abs(avg_obj),
                            axis_to_align=np.array([q_lab[2], q_lab[1], q_lab[0]]) / np.linalg.norm(q_lab),
                            reference_axis=myaxis, debugging=True)
    phase = pu.rotate_crystal(array=np.angle(avg_obj),
                              axis_to_align=np.array([q_lab[2], q_lab[1], q_lab[0]]) / np.linalg.norm(q_lab),
                              reference_axis=myaxis, debugging=False)

###################################
# phase ramp removal (upsampling) #
###################################
if strain_method == 'defect':
    amp, phase, _, _, _ = pu.remove_ramp(amp=amp, phase=phase, initial_shape=original_size, method='upsampling',
                                         amplitude_threshold=isosurface_strain, gradient_threshold=threshold_gradient,
                                         debugging=debug)
else:
    amp, phase, _, _, _ = pu.remove_ramp(amp=amp, phase=phase, initial_shape=original_size, method=phase_ramp_removal,
                                         amplitude_threshold=isosurface_strain, gradient_threshold=threshold_gradient,
                                         debugging=debug)

################################################################
# calculate the strain depending on which axis q is aligned on #
################################################################
voxel_sizes = setup.voxel_sizes_detector(array_shape=original_size, tilt_angle=tilt_angle, pixel_x=pixel_size,
                                         pixel_y=pixel_size, debug=True)

strain = pu.get_strain(phase=phase, planar_distance=planar_dist, voxel_size=voxel_sizes,
                       reference_axis=ref_axis_q)

fig, _, _ = gu.multislices_plot(strain, sum_frames=False, title='Strain in detector frame',
                                vmin=-strain_range, vmax=strain_range, plot_colorbar=True, cmap=my_cmap,
                                is_orthogonal=False, reciprocal_space=False)

print('End of script')
plt.ioff()
plt.show()
