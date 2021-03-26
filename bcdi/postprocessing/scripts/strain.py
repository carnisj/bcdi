# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr


from functools import reduce
import gc
try:
    import hdf5plugin  # for P10, should be imported before h5py or PyTables
except ModuleNotFoundError:
    pass
from matplotlib import pyplot as plt
from numbers import Real
import numpy as np
import os
import tkinter as tk
import sys
from tkinter import filedialog
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

scan = 74  # spec scan number
root_folder = "D:/data/CRISTAL_March2021/"  # folder of the experiment, where all scans are stored
save_dir = None  # images will be saved here, leave it to None otherwise (default to data directory's parent)
sample_name = "S"  # "S"  # string in front of the scan number in the folder name.
comment = ''  # comment in filenames, should start with _
#########################################################
# parameters used when averaging several reconstruction #
#########################################################
sort_method = 'variance/mean'  # 'mean_amplitude' or 'variance' or 'variance/mean' or 'volume', metric for averaging
correlation_threshold = 0.90
#########################################################
# parameters relative to the FFT window and voxel sizes #
#########################################################
original_size = [256, 256, 360]  # size of the FFT array before binning. It will be modify to take into account binning
# during phasing automatically. Leave it to () if the shape did not change.
phasing_binning = (1, 1, 1)  # binning factor applied during phase retrieval
preprocessing_binning = (1, 1, 1)  # binning factors in each dimension used in preprocessing (not phase retrieval)
output_size = (150, 150, 150)  # (z, y, x) Fix the size of the output array, leave it as () otherwise
keep_size = False  # True to keep the initial array size for orthogonalization (slower), it will be cropped otherwise
fix_voxel = None # voxel size in nm for the interpolation during the geometrical transformation. If a single value is
# provided, the voxel size will be identical is all 3 directions. Set it to None to use the default voxel size
# (calculated from q values, it will be different in each dimension).
plot_margin = (60, 60, 60)  # (z, y, x) margin in pixel to leave outside the support in each direction when cropping,
# it can be negative. It is useful in order to avoid cutting the object during the orthogonalization.
#############################################################
# parameters related to displacement and strain calculation #
#############################################################
data_frame = 'detector'  # 'crystal' if the data was interpolated into the crystal frame using (xrayutilities) or
# (transformation matrix + align_q=True)
# 'laboratory' if the data was interpolated into the laboratory frame using the transformation matrix (align_q = False)
# 'detector' if the data is still in the detector frame
isosurface_strain = 0.2  # threshold use for removing the outer layer (strain is undefined at the exact surface voxel)
strain_method = 'default'  # 'default' or 'defect'. If 'defect', will offset the phase in a loop and keep the smallest
# magnitude value for the strain. See: F. Hofmann et al. PhysRevMaterials 4, 013801 (2020)
phase_offset = 0  # manual offset to add to the phase, should be 0 in most cases
offset_origin = None  # the phase at this voxel will be set to phase_offset, None otherwise
offset_method = 'mean'  # 'COM' or 'mean', method for removing the offset in the phase
centering_method = 'max_com'  # 'com' (center of mass), 'max', 'max_com' (max then com), 'do_nothing'
# TODO: where is q for energy scans? Should we just rotate the reconstruction to have q along one axis,
#  instead of using sample offsets?
######################################
# define beamline related parameters #
######################################
beamline = "CRISTAL"  # name of the beamline, used for data loading and normalization by monitor and orthogonalisation
# supported beamlines: 'ID01', 'SIXS_2018', 'SIXS_2019', 'CRISTAL', 'P10', '34ID'
actuators = {'rocking_angle': 'actuator_1_3'}
# Optional dictionary that can be used to define the entries corresponding to actuators in data files
# (useful at CRISTAL where the location of data keeps changing)
# e.g.  {'rocking_angle': 'actuator_1_3', 'detector': 'data_04', 'monitor': 'data_05'}
rocking_angle = "inplane"  # # "outofplane" for a sample rotation around x outboard, "inplane" for a sample rotation
# around y vertical up, does not matter for energy scan
#  "inplane" e.g. phi @ ID01, mu @ SIXS "outofplane" e.g. eta @ ID01
sdd = 0.914  # 1.26  # sample to detector distance in m
energy = 8500  # x-ray energy in eV, 6eV offset at ID01
beam_direction = np.array([1, 0, 0])  # incident beam along z, in the frame (z downstream, y vertical up, x outboard)
outofplane_angle = 20.8447  # detector angle in deg (rotation around x outboard): delta ID01, delta SIXS, gamma 34ID
# this is the true angle, corrected for the direct beam position
inplane_angle = 39.1953  # detector angle in deg(rotation around y vertical up): nu ID01, gamma SIXS, tth 34ID
# this is the true angle, corrected for the direct beam position
tilt_angle = 0.00469  # angular step size for rocking angle, eta ID01, mu SIXS, does not matter for energy scan
sample_offsets = (0, 0, 0)  # tuple of offsets in degrees of the sample around (downstream, vertical up, outboard)
# the sample offsets will be subtracted to the motor values
specfile_name = None  # root_folder + 'alias_dict_2021.txt'
# template for ID01: name of the spec file without '.spec'
# template for SIXS_2018: full path of the alias dictionnary, typically root_folder + 'alias_dict_2019.txt'
# template for all other beamlines: ''
###############################
# detector related parameters #
###############################
detector = "Maxipix"    # "Eiger2M", "Maxipix", "Eiger4M", "Merlin" or "Timepix"
nb_pixel_x = None  # fix to declare a known detector but with less pixels (e.g. one tile HS), leave None otherwise
nb_pixel_y = None  # fix to declare a known detector but with less pixels (e.g. one tile HS), leave None otherwise
template_imagefile = 'mgphi-2021_%04d.nxs'
# template for ID01: 'data_mpx4_%05d.edf.gz' or 'align_eiger2M_%05d.edf.gz'
# template for SIXS_2018: 'align.spec_ascan_mu_%05d.nxs'
# template for SIXS_2019: 'spare_ascan_mu_%05d.nxs'
# template for Cristal: 'S%d.nxs'
# template for P10: '_master.h5'
# template for NANOMAX: '%06d.h5'
# template for 34ID: 'Sample%dC_ES_data_51_256_256.npz'
###################################################
# parameters related to the refraction correction #
###################################################
correct_refraction = False  # True for correcting the phase shift due to refraction
optical_path_method = 'threshold'  # 'threshold' or 'defect', if 'threshold' it uses isosurface_strain to define the
# support  for the optical path calculation, if 'defect' (holes) it tries to remove only outer layers even if
# the amplitude is lower than isosurface_strain inside the crystal
dispersion = 4.6353E-05  # delta
# Pt:  3.0761E-05 @ 10300eV, 5.0328E-05 @ 8170eV
# 3.2880E-05 @ 9994eV, 4.1184E-05 @ 8994eV, 5.2647E-05 @ 7994eV, 4.6353E-05 @ 8500eV / Ge 1.4718E-05 @ 8keV
absorption = 4.1969E-06  # beta
# Pt:  2.0982E-06 @ 10300eV, 4.8341E-06 @ 8170eV
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
threshold_gradient = 1.0  # upper threshold of the gradient of the phase, use for ramp removal
save_raw = False  # True to save the amp-phase.vti before orthogonalization
save_support = False  # True to save the non-orthogonal support for later phase retrieval
save_labframe = False  # True to save the data in the laboratory frame (before rotations)
save = True  # True to save amp.npz, phase.npz, strain.npz and vtk files
debug = False  # set to True to show all plots for debugging
roll_modes = (0, 0, 0)   # axis=(0, 1, 2), correct a roll of few pixels after the decomposition into modes in PyNX
############################################
# parameters related to data visualization #
############################################
align_q = False  # if True rotates the crystal to align q it along one axis of the array
ref_axis_q = "x"  # q will be aligned along that axis
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

if fix_voxel:
    if isinstance(fix_voxel, Real):
        fix_voxel = (fix_voxel, fix_voxel, fix_voxel)
    assert isinstance(fix_voxel, (tuple, list)) and all(val > 0 for val in fix_voxel),\
        'fix_voxel should be a positive number or a tuple/list of three positive numbers'

if data_frame not in {'detector', 'crystal', 'laboratory'}:
    raise ValueError('Uncorrect setting for "data_frame" parameter')
elif data_frame == 'detector':
    is_orthogonal = False
else:
    is_orthogonal = True

comment = comment + '_' + str(isosurface_strain)

###################
# define colormap #
###################
if grey_background:
    bad_color = '0.7'
else:
    bad_color = '1.0'  # white background
colormap = gu.Colormap(bad_color=bad_color)
my_cmap = colormap.cmap

#######################
# Initialize detector #
#######################
kwargs = dict()  # create dictionnary
kwargs['preprocessing_binning'] = preprocessing_binning
if nb_pixel_x:
    kwargs['nb_pixel_x'] = nb_pixel_x  # fix to declare a known detector but with less pixels (e.g. one tile HS)
if nb_pixel_y:
    kwargs['nb_pixel_y'] = nb_pixel_y  # fix to declare a known detector but with less pixels (e.g. one tile HS)

detector = exp.Detector(name=detector, template_imagefile=template_imagefile, binning=phasing_binning, **kwargs)

####################################
# define the experimental geometry #
####################################
# correct the tilt_angle for binning
tilt_angle = tilt_angle * preprocessing_binning[0] * phasing_binning[0]
setup = exp.Setup(beamline=beamline, energy=energy, outofplane_angle=outofplane_angle, inplane_angle=inplane_angle,
                  tilt_angle=tilt_angle, rocking_angle=rocking_angle, distance=sdd, pixel_x=detector.pixelsize_x,
                  pixel_y=detector.pixelsize_y, sample_offsets=sample_offsets, actuators=actuators)

########################################
# Initialize the paths and the logfile #
########################################
setup.init_paths(detector=detector, sample_name=sample_name, scan_number=scan, root_folder=root_folder,
                 save_dir=save_dir, specfile_name=specfile_name, template_imagefile=template_imagefile,
                 create_savedir=True)

logfile = pru.create_logfile(setup=setup, detector=detector, scan_number=scan, root_folder=root_folder,
                             filename=detector.specfile)

############################################################################################################
# get the motor position of goniometer circles which are below the rocking angle (e.g., chi for eta/omega) #
############################################################################################################
_, setup.grazing_angle, _, _ = pru.goniometer_values(logfile=logfile, scan_number=scan, setup=setup)

###################
# print instances #
###################
print(f'{"#"*(5+len(str(scan)))}\nScan {scan}\n{"#"*(5+len(str(scan)))}')
print('\n##############\nSetup instance\n##############')
print(setup)
print('\n#################\nDetector instance\n#################')
print(detector)

################
# preload data #
################
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilenames(initialdir=detector.scandir,
                                        filetypes=[("NPZ", "*.npz"), ("NPY", "*.npy"),
                                                   ("CXI", "*.cxi"), ("HDF5", "*.h5")])
nbfiles = len(file_path)
plt.ion()

obj, extension = util.load_file(file_path[0])
if extension == '.h5':
    comment = comment + '_mode'

print('\n###############\nProcessing data\n###############')
nz, ny, nx = obj.shape
print("Initial data size: (", nz, ',', ny, ',', nx, ')')
if len(original_size) == 0:
    original_size = obj.shape
print("FFT size before accounting for phasing_binning", original_size)
original_size = tuple([original_size[index] // phasing_binning[index] for index in range(len(phasing_binning))])
print("Binning used during phasing:", detector.binning)
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
                                       method=avg_method, reciprocal_space=False, is_orthogonal=is_orthogonal,
                                       debugging=True)
    avg_counter = avg_counter + flag_avg

avg_obj = avg_obj / avg_counter
print('\nAverage performed over ', avg_counter, 'reconstructions\n')
del obj, ref_obj
gc.collect()

################
# unwrap phase #
################
phase, extent_phase = pu.unwrap(avg_obj, support_threshold=threshold_unwrap_refraction, debugging=True,
                                reciprocal_space=False, is_orthogonal=is_orthogonal)

print('Extent of the phase over an extended support (ceil(phase range)) ~ ', int(extent_phase), '(rad)')
phase = pru.wrap(phase, start_angle=-extent_phase/2, range_angle=extent_phase)
if debug:
    gu.multislices_plot(phase, width_z=2*zrange, width_y=2*yrange, width_x=2*xrange, plot_colorbar=True,
                        title='Phase after unwrap + wrap', reciprocal_space=False, is_orthogonal=is_orthogonal)

#############################################
# phase ramp removal before phase filtering #
#############################################
amp, phase, rampz, rampy, rampx = pu.remove_ramp(amp=abs(avg_obj), phase=phase, initial_shape=original_size,
                                                 method='gradient', amplitude_threshold=isosurface_strain,
                                                 gradient_threshold=threshold_gradient)
del avg_obj
gc.collect()

if debug:
    gu.multislices_plot(phase, width_z=2*zrange, width_y=2*yrange, width_x=2*xrange, plot_colorbar=True,
                        title='Phase after ramp removal', reciprocal_space=False, is_orthogonal=is_orthogonal)

########################
# phase offset removal #
########################
support = np.zeros(amp.shape)
support[amp > isosurface_strain*amp.max()] = 1
phase = pu.remove_offset(array=phase, support=support, offset_method=offset_method, user_offset=phase_offset,
                         offset_origin=offset_origin, title='Phase', debugging=debug)
del support
gc.collect()

phase = pru.wrap(obj=phase, start_angle=-extent_phase / 2, range_angle=extent_phase)

##############################################################################
# average the phase over a window or apodize to reduce noise in strain plots #
##############################################################################
if hwidth != 0:
    bulk = pu.find_bulk(amp=amp, support_threshold=isosurface_strain, method='threshold')
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
                            sigma=sigma, mu=mu, alpha=alpha, is_orthogonal=is_orthogonal, debugging=True)
    comment = comment + '_apodize_' + apodize_window

####################################################################################################################
# save the phase with the ramp for PRTF calculations, otherwise the object will be misaligned with the measurement #
####################################################################################################################
np.savez_compressed(detector.savedir + 'S' + str(scan) + '_avg_obj_prtf' + comment, obj=amp * np.exp(1j * phase))

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
if centering_method == 'max':
    avg_obj = pu.center_max(avg_obj)
    # shift based on max value, required if it spans across the edge of the array before COM
elif centering_method == 'com':
    avg_obj = pu.center_com(avg_obj)
elif centering_method == 'max_com':
    avg_obj = pu.center_max(avg_obj)
    avg_obj = pu.center_com(avg_obj)

#######################
#  save support & vti #
#######################
if save_support:  # to be used as starting support in phasing, hence still in the detector frame
    support = np.zeros((numz, numy, numx))
    support[abs(avg_obj)/abs(avg_obj).max() > 0.01] = 1
    # low threshold because support will be cropped by shrinkwrap during phasing
    np.savez_compressed(detector.savedir + 'S' + str(scan) + '_support' + comment, obj=support)
    del support
    gc.collect()

if save_raw:
    np.savez_compressed(detector.savedir + 'S' + str(scan) + '_raw_amp-phase' + comment,
                        amp=abs(avg_obj), phase=np.angle(avg_obj))

    # voxel sizes in the detector frame
    voxel_z, voxel_y, voxel_x = setup.voxel_sizes_detector(array_shape=original_size, tilt_angle=tilt_angle,
                                                           pixel_x=detector.pixelsize_x, pixel_y=detector.pixelsize_y,
                                                           verbose=True)
    # save raw amp & phase to VTK
    # in VTK, x is downstream, y vertical, z inboard, thus need to flip the last axis
    gu.save_to_vti(filename=os.path.join(detector.savedir, "S" + str(scan) + "_raw_amp-phase" + comment + ".vti"),
                   voxel_size=(voxel_z, voxel_y, voxel_x), tuple_array=(abs(avg_obj), np.angle(avg_obj)),
                   tuple_fieldnames=('amp', 'phase'), amplitude_threshold=0.01)

#######################
#  orthogonalize data #
#######################
print('\nShape before orthogonalization', avg_obj.shape)
if data_frame == 'detector':
    gu.multislices_plot(abs(avg_obj), width_z=2 * zrange, width_y=2 * yrange, width_x=2 * xrange,
                        sum_frames=False, plot_colorbar=True, vmin=0, vmax=abs(avg_obj).max(),
                        title='Amp before orthogonalization', reciprocal_space=False, is_orthogonal=False)
    if debug:
        phase, _ = pu.unwrap(avg_obj, support_threshold=threshold_unwrap_refraction, debugging=True,
                             reciprocal_space=False, is_orthogonal=False)
        gu.multislices_plot(phase, width_z=2 * zrange, width_y=2 * yrange, width_x=2 * xrange,
                            sum_frames=False, plot_colorbar=True, reciprocal_space=False, is_orthogonal=False,
                            title='Unwrapped phase before orthogonalization')
        del phase
        gc.collect()

    obj_ortho, voxel_size = setup.orthogonalize(obj=avg_obj, initial_shape=original_size, voxel_size=fix_voxel)
    print(f"VTK spacing : {voxel_size} (nm)")

    gu.multislices_plot(abs(obj_ortho), width_z=2 * zrange, width_y=2 * yrange, width_x=2 * xrange,
                        sum_frames=False, plot_colorbar=True, vmin=0, vmax=abs(obj_ortho).max(),
                        title='Amp after orthogonalization', reciprocal_space=False, is_orthogonal=True)

else:  # data already orthogonalized using xrayutilities or the linearized transformation matrix
    obj_ortho = avg_obj
    try:
        print("Select the file containing QxQzQy")
        file_path = filedialog.askopenfilename(title="Select the file containing QxQzQy",
                                               initialdir=detector.savedir, filetypes=[("NPZ", "*.npz")])
        npzfile = np.load(file_path)
        qx = npzfile['qx']
        qy = npzfile['qy']
        qz = npzfile['qz']
    except FileNotFoundError:
        raise FileNotFoundError('Voxel sizes not provided')
    dy_real = 2 * np.pi / abs(qz.max() - qz.min()) / 10  # in nm qz=y in nexus convention
    dx_real = 2 * np.pi / abs(qy.max() - qy.min()) / 10  # in nm qy=x in nexus convention
    dz_real = 2 * np.pi / abs(qx.max() - qx.min()) / 10  # in nm qx=z in nexus convention
    print(f'direct space voxel size from q values: {dz_real:.2f}nm, {dy_real:.2f}nm, {dx_real:.2f}nm')
    if fix_voxel:
        voxel_size = fix_voxel
        print(f'Direct space pixel size for the interpolation: {voxel_size} (nm)')
        print('Interpolating...\n')
        obj_ortho = pu.regrid(array=obj_ortho, old_voxelsize=(dz_real, dy_real, dx_real), new_voxelsize=voxel_size)
    else:
        # no need to interpolate
        voxel_size = dz_real, dy_real, dx_real  # in nm
del avg_obj
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

kin = 2*np.pi/setup.wavelength * setup.beam_direction  # in laboratory frame z downstream, y vertical, x outboard
kout = setup.exit_wavevector  # in laboratory frame z downstream, y vertical, x outboard

q = kout - kin  # in laboratory frame z downstream, y vertical, x outboard
qnorm = np.linalg.norm(q)
q = q / qnorm
angle = simu.angle_vectors(ref_vector=np.array([q[2], q[1], q[0]]), test_vector=myaxis)
print(f"\nAngle between q and {ref_axis_q} = {angle:.2f} deg")
if debug:
    print(f"Angle with y in zy plane = {np.arctan(q[0]/q[1])*180/np.pi:.2f} deg")
    print(f"Angle with y in xy plane = {np.arctan(-q[2]/q[1])*180/np.pi:.2f} deg")
    print(f"Angle with z in xz plane = {180+np.arctan(q[2]/q[0])*180/np.pi:.2f} deg\n")

qnorm = qnorm * 1e-10  # switch to angstroms
planar_dist = 2*np.pi/qnorm  # qnorm should be in angstroms
print(f"Normalized wavevector transfer (z*, y*, x*): {q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}")
print("Wavevector transfer: (1/A)", str('{:.4f}'.format(qnorm)))
print("Atomic plane distance: (A)", str('{:.4f}'.format(planar_dist)), "angstroms")

if get_temperature:
    temperature = pu.bragg_temperature(spacing=planar_dist, reflection=reflection, spacing_ref=reference_spacing,
                                       temperature_ref=reference_temperature, use_q=False, material="Pt")
else:
    temperature = 20  # C

planar_dist = planar_dist / 10  # switch to nm

######################
# centering of array #
######################
obj_ortho = pu.center_com(obj_ortho)
amp = abs(obj_ortho)
phase, extent_phase = pu.unwrap(obj_ortho, support_threshold=threshold_unwrap_refraction, debugging=debug,
                                reciprocal_space=False, is_orthogonal=True)
del obj_ortho
gc.collect()

if debug:
    gu.multislices_plot(amp, width_z=2 * zrange, width_y=2 * yrange, width_x=2 * xrange,
                        sum_frames=False, plot_colorbar=True, vmin=0, vmax=amp.max(),
                        title='Amp before absorption correction', reciprocal_space=False, is_orthogonal=True)
    gu.multislices_plot(phase, width_z=2 * zrange, width_y=2 * yrange, width_x=2 * xrange,
                        sum_frames=False, plot_colorbar=True, title='Unwrapped phase before refraction correction',
                        reciprocal_space=False, is_orthogonal=True)

#############################################
# invert phase: -1*phase = displacement * q #
#############################################
if invert_phase:
    phase = -1 * phase

########################################
# refraction and absorption correction #
########################################
if correct_refraction:  # or correct_absorption:
    bulk = pu.find_bulk(amp=amp, support_threshold=threshold_unwrap_refraction, method=optical_path_method,
                        debugging=debug)

    # kin and kout were calculated in the laboratory frame. If the crystal is in its frame, we need to transform kin
    # and kout back into the crystal frame (xrayutilities output is in crystal frame)
    if data_frame == 'crystal':
        kin = pu.rotate_vector(vector=np.array([kin[2], kin[1], kin[0]]), axis_to_align=myaxis,
                               reference_axis=np.array([q[2], q[1], q[0]]))
        kout = pu.rotate_vector(vector=np.array([kout[2], kout[1], kout[0]]), axis_to_align=myaxis,
                                reference_axis=np.array([q[2], q[1], q[0]]))

    # calculate the optical path of the incoming wavevector
    path_in = pu.get_opticalpath(support=bulk, direction="in", k=kin, debugging=debug)  # path_in already in nm

    # calculate the optical path of the outgoing wavevector
    path_out = pu.get_opticalpath(support=bulk, direction="out", k=kout, debugging=debug)  # path_our already in nm

    optical_path = path_in + path_out
    del path_in, path_out
    gc.collect()

    if correct_refraction:
        phase_correction = 2 * np.pi / (1e9 * setup.wavelength) * dispersion * optical_path
        phase = phase + phase_correction

        gu.multislices_plot(np.multiply(phase_correction, bulk), width_z=2 * zrange, width_y=2 * yrange,
                            width_x=2 * xrange, sum_frames=False, plot_colorbar=True, vmin=0, vmax=np.pi/2,
                            title='Refraction correction on the support', is_orthogonal=True, reciprocal_space=False)

    if False:  # correct_absorption:
        # TODO: it is correct to compensate also the X-ray absorption in the reconstructed modulus?
        amp_correction = np.exp(2 * np.pi / (1e9 * setup.wavelength) * absorption * optical_path)
        amp = amp * amp_correction

        gu.multislices_plot(np.multiply(amp_correction, bulk), width_z=2 * zrange, width_y=2 * yrange,
                            width_x=2 * xrange, sum_frames=False, plot_colorbar=True, vmin=1, vmax=1.1,
                            title='Absorption correction on the support', is_orthogonal=True, reciprocal_space=False)

    del bulk, optical_path
    gc.collect()

##############################################
# phase ramp and offset removal (mean value) #
##############################################
amp, phase, _, _, _ = pu.remove_ramp(amp=amp, phase=phase, initial_shape=original_size, method=phase_ramp_removal,
                                     amplitude_threshold=isosurface_strain, gradient_threshold=threshold_gradient,
                                     debugging=debug)

########################
# phase offset removal #
########################
support = np.zeros(amp.shape)
support[amp > isosurface_strain*amp.max()] = 1
phase = pu.remove_offset(array=phase, support=support, offset_method=offset_method, user_offset=phase_offset,
                         offset_origin=offset_origin, title='Orthogonal phase', debugging=debug,
                         reciprocal_space=False, is_orthogonal=True)
del support
gc.collect()

phase = pru.wrap(obj=phase, start_angle=-extent_phase / 2, range_angle=extent_phase)

################################
# save to VTK before rotations #
################################
if save_labframe:
    if invert_phase:
        np.savez_compressed(detector.savedir + 'S' + str(scan) + "_amp" + phase_fieldname + comment + '_LAB',
                            amp=amp, displacement=phase)
    else:
        np.savez_compressed(detector.savedir + 'S' + str(scan) + "_amp" + phase_fieldname + comment + '_LAB',
                            amp=amp, phase=phase)

    print(f'VTK spacing : {voxel_size} (nm)')
    # save amp & phase to VTK before rotation in crystal frame
    # in VTK, x is downstream, y vertical, z inboard, thus need to flip the last axis
    gu.save_to_vti(filename=os.path.join(detector.savedir, "S"+str(scan)+"_amp-"+phase_fieldname+"_LAB"+comment+".vti"),
                   voxel_size=voxel_size, tuple_array=(amp, phase), tuple_fieldnames=('amp', phase_fieldname),
                   amplitude_threshold=0.01)
    
############################################################################
# put back the crystal in its frame, by aligning q onto the reference axis #
############################################################################
if data_frame != 'crystal':
    print('\nAligning Q along ', ref_axis_q, ":", myaxis)
    amp = pu.rotate_crystal(array=amp, axis_to_align=np.array([q[2], q[1], q[0]])/np.linalg.norm(q),
                            reference_axis=myaxis, voxel_size=voxel_size, debugging=True,
                            is_orthogonal=True, reciprocal_space=False)
    phase = pu.rotate_crystal(array=phase, axis_to_align=np.array([q[2], q[1], q[0]])/np.linalg.norm(q),
                              reference_axis=myaxis, voxel_size=voxel_size, debugging=False,
                              is_orthogonal=True, reciprocal_space=False)

################################################################
# calculate the strain depending on which axis q is aligned on #
################################################################
strain = pu.get_strain(phase=phase, planar_distance=planar_dist, voxel_size=voxel_size, reference_axis=ref_axis_q,
                       extent_phase=extent_phase, method=strain_method, debugging=debug)

#######################################
# optionally rotates back the crystal #
#######################################
if data_frame != 'crystal':
    if align_q:
        comment = comment + '_crystal-frame'
    else:
        comment = comment + '_lab-frame'
        print('Rotating back the crystal in laboratory frame')
        amp = pu.rotate_crystal(array=amp, axis_to_align=myaxis, voxel_size=voxel_size,
                                reference_axis=np.array([q[2], q[1], q[0]])/np.linalg.norm(q), debugging=True,
                                is_orthogonal=True, reciprocal_space=False)
        phase = pu.rotate_crystal(array=phase, axis_to_align=myaxis, voxel_size=voxel_size,
                                  reference_axis=np.array([q[2], q[1], q[0]])/np.linalg.norm(q), debugging=False,
                                  is_orthogonal=True, reciprocal_space=False)
        strain = pu.rotate_crystal(array=strain, axis_to_align=myaxis, voxel_size=voxel_size,
                                   reference_axis=np.array([q[2], q[1], q[0]])/np.linalg.norm(q), debugging=False,
                                   is_orthogonal=True, reciprocal_space=False)

############################################################################
# rotates the crystal for example inplane for easier slicing of the result #
############################################################################
if align_axis:
    if ref_axis == "x":
        myaxis_inplane = np.array([1, 0, 0])  # must be in [x, y, z] order
    elif ref_axis == "y":
        myaxis_inplane = np.array([0, 1, 0])  # must be in [x, y, z] order
    else:  # ref_axis = "z"
        myaxis_inplane = np.array([0, 0, 1])  # must be in [x, y, z] order
    amp = pu.rotate_crystal(array=amp, axis_to_align=axis_to_align/np.linalg.norm(axis_to_align),
                            reference_axis=myaxis_inplane, voxel_size=voxel_size, debugging=True,
                            is_orthogonal=True, reciprocal_space=False)
    phase = pu.rotate_crystal(array=phase, axis_to_align=axis_to_align/np.linalg.norm(axis_to_align),
                              reference_axis=myaxis_inplane, voxel_size=voxel_size, debugging=False,
                              is_orthogonal=True, reciprocal_space=False)
    strain = pu.rotate_crystal(array=strain, axis_to_align=axis_to_align/np.linalg.norm(axis_to_align),
                               reference_axis=myaxis_inplane, voxel_size=voxel_size, debugging=False,
                               is_orthogonal=True, reciprocal_space=False)

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
print(f'Voxel size: ({voxel_size[0]:.2f}, {voxel_size[1]:.2f}, {voxel_size[2]:.2f}) (nm)')
bulk = pu.find_bulk(amp=amp, support_threshold=isosurface_strain, method='threshold')
if save:
    if invert_phase:
        np.savez_compressed(detector.savedir + 'S' + str(scan) + "_amp" + phase_fieldname + "strain" + comment,
                            amp=amp, displacement=phase, bulk=bulk, strain=strain)
    else:
        np.savez_compressed(detector.savedir + 'S' + str(scan) + "_amp" + phase_fieldname + "strain" + comment,
                            amp=amp, phase=phase, bulk=bulk, strain=strain)

    # save amp & phase to VTK
    # in VTK, x is downstream, y vertical, z inboard, thus need to flip the last axis
    gu.save_to_vti(filename=os.path.join(detector.savedir,
                                         "S"+str(scan)+"_amp-"+phase_fieldname+"-strain"+comment+".vti"),
                   voxel_size=voxel_size, tuple_array=(amp, bulk, phase, strain),
                   tuple_fieldnames=('amp', 'bulk', phase_fieldname, 'strain'), amplitude_threshold=0.01)


########################
# calculate the volume #
########################
amp = amp / amp.max()
temp_amp = np.copy(amp)
temp_amp[amp < isosurface_strain] = 0
temp_amp[np.nonzero(temp_amp)] = 1
volume = temp_amp.sum()*reduce(lambda x, y: x*y, voxel_size)  # in nm3
del temp_amp
gc.collect()

#######################
# plot phase & strain #
#######################
pixel_spacing = [tick_spacing / vox for vox in voxel_size]
print(f'Phase extent before and after thresholding: {phase.max()-phase.min():.2f},'
      f'{phase[np.nonzero(bulk)].max()-phase[np.nonzero(bulk)].min():.2f}')
piz, piy, pix = np.unravel_index(phase.argmax(), phase.shape)
print(f'phase.max() = {phase[np.nonzero(bulk)].max():.2f} at voxel ({piz}, {piy}, {pix})')
strain[bulk == 0] = np.nan
phase[bulk == 0] = np.nan

# plot the slice at the maximum phase
gu.combined_plots((phase[piz, :, :], phase[:, piy, :], phase[:, :, pix]), tuple_sum_frames=False, tuple_sum_axis=0,
                  tuple_width_v=None, tuple_width_h=None, tuple_colorbar=True, tuple_vmin=np.nan,
                  tuple_vmax=np.nan, tuple_title=('phase at max in xy', 'phase at max in xz', 'phase at max in yz'),
                  tuple_scale='linear', cmap=my_cmap, is_orthogonal=True, reciprocal_space=False)

# bulk support
fig, _, _ = gu.multislices_plot(bulk, sum_frames=False, title='Orthogonal bulk', vmin=0, vmax=1,
                                is_orthogonal=True, reciprocal_space=False)
fig.text(0.60, 0.45, "Scan " + str(scan), size=20)
fig.text(0.60, 0.40, "Bulk - isosurface=" + str('{:.2f}'.format(isosurface_strain)), size=20)
plt.pause(0.1)
if save:
    plt.savefig(detector.savedir + 'S' + str(scan) + '_bulk' + comment + '.png')

# amplitude
fig, _, _ = gu.multislices_plot(amp, sum_frames=False, title='Normalized orthogonal amp', vmin=0,
                                vmax=1, tick_direction=tick_direction, tick_width=tick_width, tick_length=tick_length,
                                pixel_spacing=pixel_spacing, plot_colorbar=True, is_orthogonal=True,
                                reciprocal_space=False)
fig.text(0.60, 0.45, f'Scan {scan}', size=20)
fig.text(0.60, 0.40, f'Voxel size=({voxel_size[0]:.1f}, {voxel_size[1]:.1f}, {voxel_size[2]:.1f}) (nm)', size=20)
fig.text(0.60, 0.35, f'Ticks spacing={tick_spacing} nm', size=20)
fig.text(0.60, 0.30, f'Volume={int(volume)} nm3', size=20)
fig.text(0.60, 0.25, 'Sorted by ' + sort_method, size=20)
fig.text(0.60, 0.20, f'correlation threshold={correlation_threshold}', size=20)
fig.text(0.60, 0.15, f'average over {avg_counter} reconstruction(s)', size=20)
fig.text(0.60, 0.10, f'Planar distance={planar_dist:.5f} nm', size=20)
if get_temperature:
    fig.text(0.60, 0.05, f'Estimated T={temperature} C', size=20)
if save:
    plt.savefig(detector.savedir + f'S{scan}_amp' + comment + '.png')

# amplitude histogram
fig, ax = plt.subplots(1, 1)
ax.hist(amp[amp > 0.05*amp.max()].flatten(), bins=250)
ax.set_ylim(bottom=1)
ax.tick_params(labelbottom=True, labelleft=True, direction='out', length=tick_length, width=tick_width)
ax.spines['right'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
fig.savefig(detector.savedir + f'S{scan}_histo_amp' + comment + '.png')

# phase
fig, _, _ = gu.multislices_plot(phase, sum_frames=False, title='Orthogonal displacement',
                                vmin=-phase_range, vmax=phase_range, tick_direction=tick_direction, cmap=my_cmap,
                                tick_width=tick_width, tick_length=tick_length, pixel_spacing=pixel_spacing,
                                plot_colorbar=True, is_orthogonal=True, reciprocal_space=False)
fig.text(0.60, 0.30, f'Scan {scan}', size=20)
fig.text(0.60, 0.25, f'Voxel size=({voxel_size[0]:.1f}, {voxel_size[1]:.1f}, {voxel_size[2]:.1f}) (nm)', size=20)
fig.text(0.60, 0.20, f'Ticks spacing={tick_spacing} nm', size=20)
fig.text(0.60, 0.15, f'average over {avg_counter} reconstruction(s)', size=20)
if hwidth > 0:
    fig.text(0.60, 0.10, f'Averaging over {2*hwidth+1} pixels', size=20)
else:
    fig.text(0.60, 0.10, "No phase averaging", size=20)
if save:
    plt.savefig(detector.savedir + f'S{scan}_displacement' + comment + '.png')

# strain
fig, _, _ = gu.multislices_plot(strain, sum_frames=False, title='Orthogonal strain',
                                vmin=-strain_range, vmax=strain_range, tick_direction=tick_direction,
                                tick_width=tick_width, tick_length=tick_length, plot_colorbar=True, cmap=my_cmap,
                                pixel_spacing=pixel_spacing, is_orthogonal=True, reciprocal_space=False)
fig.text(0.60, 0.30, f'Scan {scan}', size=20)
fig.text(0.60, 0.25, f'Voxel size=({voxel_size[0]:.1f}, {voxel_size[1]:.1f}, {voxel_size[2]:.1f}) (nm)', size=20)
fig.text(0.60, 0.20, f'Ticks spacing={tick_spacing} nm', size=20)
fig.text(0.60, 0.15, f'average over {avg_counter} reconstruction(s)', size=20)
if hwidth > 0:
    fig.text(0.60, 0.10, f'Averaging over {2*hwidth+1} pixels', size=20)
else:
    fig.text(0.60, 0.10, "No phase averaging", size=20)
if save:
    plt.savefig(detector.savedir + f'S{scan}_strain' + comment + '.png')


print('End of script')
plt.ioff()
plt.show()
