#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

from collections.abc import Sequence
from datetime import datetime
from functools import reduce
import gc

try:
    import hdf5plugin  # for P10, should be imported before h5py or PyTables
except ModuleNotFoundError:
    pass
import h5py
from matplotlib import pyplot as plt
from numbers import Real
import numpy as np
import os
import pprint
import tkinter as tk
from tkinter import filedialog
import bcdi.graph.graph_utils as gu
import bcdi.experiment.experiment_utils as exp
import bcdi.postprocessing.postprocessing_utils as pu
import bcdi.preprocessing.preprocessing_utils as pru
import bcdi.simulation.simulation_utils as simu
import bcdi.utils.utilities as util
import bcdi.utils.validation as valid

helptext = """
Interpolate the output of the phase retrieval into an orthonormal frame,
and calculate the strain component along the direction of the experimental diffusion 
vector q.

Input: complex amplitude array, output from a phase retrieval program.
Output: data in an orthonormal frame (laboratory or crystal frame), amp_disp_strain
array.The disp array should be divided by q to get the displacement (disp = -1*phase
here).

Laboratory frame: z downstream, y vertical, x outboard (CXI convention)
Crystal reciprocal frame: qx downstream, qz vertical, qy outboard
Detector convention: when out_of_plane angle=0   Y=-y , when in_plane angle=0   X=x

In arrays, when plotting the first parameter is the row (vertical axis), and the
second the column (horizontal axis). Therefore the data structure is data[qx, qz,
qy] for reciprocal space, or data[z, y, x] for real space
"""

scan = 76  # spec scan number
root_folder = "C:/Users/Jerome/Documents/data/debug/data/"
# folder of the experiment, where all scans are stored
save_dir = None
# images will be saved here,
# leave it to None otherwise (default to data directory's parent)
sample_name = "S"  # "S"  # string in front of the scan number in the folder name.
comment = ""  # comment in filenames, should start with _
#########################################################
# parameters used when averaging several reconstruction #
#########################################################
sort_method = "variance/mean"
# 'mean_amplitude' or 'variance' or 'variance/mean' or 'volume', metric for averaging
correlation_threshold = 0.90
#########################################################
# parameters relative to the FFT window and voxel sizes #
#########################################################
original_size = [
    252,
    294,
    360,
]  # size of the FFT array before binning.
# It will be modify to take into account binning
# during phasing automatically. Leave it to () if the shape did not change.
phasing_binning = (1, 1, 1)  # binning factor applied during phase retrieval
preprocessing_binning = (
    1,
    1,
    1,
)  # binning factors in each dimension used in preprocessing (not phase retrieval)
output_size = (
    100,
    100,
    100,
)  # (z, y, x) Fix the size of the output array, leave None to use the object size
keep_size = False  # True to keep the initial array size for orthogonalization (slower)
# it will be cropped otherwise
fix_voxel = 10  # voxel size in nm for the interpolation during the geometrical
# transformation. If a single value is provided, the voxel size will be identical is
# all 3 directions. Set it to None to use the default voxel size
# (calculated from q values, it will be different in each dimension).
#############################################################
# parameters related to displacement and strain calculation #
#############################################################
data_frame = "detector"
# 'crystal' if the data was interpolated into the crystal frame using (xrayutilities) or
# (transformation matrix + align_q=True)
# 'laboratory' if the data was interpolated into the laboratory frame using
# the transformation matrix (align_q = False)
# 'detector' if the data is still in the detector frame
ref_axis_q = (
    "y"  # axis along which q will be aligned (data_frame= 'detector' or 'laboratory')
)
# or is already aligned (data_frame='crystal')
save_frame = "laboratory"  # 'crystal', 'laboratory' or 'lab_flat_sample'
# 'crystal' to save the data with q aligned along ref_axis_q
# 'laboratory' to save the data in the laboratory frame (experimental geometry)
# 'lab_flat_sample' to save the data in the laboratory frame,
# with all sample angles rotated back to 0. The rotations for 'laboratory' and
# 'lab_flat_sample' are realized after the strain calculation
# (which is done in the crystal frame along ref_axis_q)
isosurface_strain = 0.2  # threshold use for removing the outer layer
# (strain is undefined at the exact surface voxel)
strain_method = "default"  # 'default' or 'defect'.
# If 'defect', will offset the phase in a loop and keep the smallest
# magnitude value for the strain.
# See: F. Hofmann et al. PhysRevMaterials 4, 013801 (2020)
phase_offset = 0  # manual offset to add to the phase, should be 0 in most cases
phase_offset_origin = (
    None  # the phase at this voxel will be set to phase_offset, None otherwise
)
offset_method = "mean"  # 'COM' or 'mean', method for removing the offset in the phase
centering_method = (
    "max_com"  # 'com' (center of mass), 'max', 'max_com' (max then com), 'do_nothing'
)
# TODO: where is q for energy scans?
#  Should we just rotate the reconstruction to have q along one axis,
#  instead of using sample offsets?
######################################
# define beamline related parameters #
######################################
beamline = "CRISTAL"  # name of the beamline, used for data loading and normalization
# by monitor and orthogonalisation
# supported beamlines: 'ID01', 'SIXS_2018', 'SIXS_2019', 'CRISTAL', 'P10', '34ID'
actuators = {"rocking_angle": "actuator_1_3"}
# Optional dictionary that can be used to define the entries
# corresponding to actuators in data files
# (useful at CRISTAL where the location of data keeps changing)
# e.g.  {'rocking_angle': 'actuator_1_3', 'detector': 'data_04', 'monitor': 'data_05'}
rocking_angle = "inplane"  # "outofplane" for a sample rotation around x outboard,
# "inplane" for a sample rotation
# around y vertical up, does not matter for energy scan
#  "inplane" e.g. phi @ ID01, mu @ SIXS "outofplane" e.g. eta @ ID01
sdd = 0.914  # 1.26  # sample to detector distance in m
energy = 8530.0  # x-ray energy in eV, 6eV offset at ID01
beam_direction = np.array(
    [1, 0, 0]
)  # incident beam along z, in the frame (z downstream, y vertical up, x outboard)
outofplane_angle = 21.4791  # detector angle in deg (rotation around x outboard):
# delta ID01, delta SIXS, gamma 34ID
# this is the true angle, corrected for the direct beam position
inplane_angle = 39.1504  # detector angle in deg(rotation around y vertical up):
# nu ID01, gamma SIXS, tth 34ID
# this is the true angle, corrected for the direct beam position
tilt_angle = (
    1.2 / 256.0
)  # angular step size for rocking angle, eta ID01, mu SIXS,
# does not matter for energy scan
sample_offsets = None
# tuple of offsets in degrees of the sample for each sample circle (outer first).
# the sample offsets will be subtracted to the motor values. Leave None if no offset.
specfile_name = None  # root_folder + 'alias_dict_2021.txt'
# template for ID01: name of the spec file without '.spec'
# template for SIXS_2018: full path of the alias dictionnary,
# typically root_folder + 'alias_dict_2019.txt'
# template for all other beamlines: ''
##########################
# setup for custom scans #
##########################
custom_scan = False  # set it to True for a stack of images acquired without scan,
# e.g. with ct in a macro, or when
# there is no spec/log file available, or for 34ID
custom_motors = {
    "delta": inplane_angle,
    "gamma": outofplane_angle,
    "theta": 1.0540277,
    "phi": -4.86,
}
###############################
# detector related parameters #
###############################
detector = "Maxipix"  # "Eiger2M", "Maxipix", "Eiger4M", "Merlin", "Timepix" or "Dummy"
nb_pixel_x = None  # fix to declare a known detector but with less pixels
# (e.g. one tile HS), leave None otherwise
nb_pixel_y = None  # fix to declare a known detector but with less pixels
# (e.g. one tile HS), leave None otherwise
pixel_size = None
# use this to declare the pixel size of the "Dummy" detector if different from 55e-6
template_imagefile = "mgtx2-mgty2-mgphi-2021-03-25_14-35-59_%04d.nxs"
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
optical_path_method = "threshold"
# 'threshold' or 'defect', if 'threshold' it uses isosurface_strain to define the
# support  for the optical path calculation, if 'defect' (holes) it tries to remove
# only outer layers even if
# the amplitude is lower than isosurface_strain inside the crystal
dispersion = 5.0328e-05  # delta
# Pt:  3.0761E-05 @ 10300eV, 5.0328E-05 @ 8170eV
# 3.2880E-05 @ 9994eV, 4.1184E-05 @ 8994eV, 5.2647E-05 @ 7994eV,
# 4.6353E-05 @ 8500eV / Ge 1.4718E-05 @ 8keV
absorption = 4.1969e-06  # beta
# Pt:  2.0982E-06 @ 10300eV, 4.8341E-06 @ 8170eV
# 2.3486E-06 @ 9994eV, 3.4298E-06 @ 8994eV, 5.2245E-06 @ 7994eV, 4.1969E-06 @ 8500eV
threshold_unwrap_refraction = 0.05  # threshold used to calculate the optical path
# the threshold for refraction/absorption corrections should be low,
# to correct for an object larger than the real one,
# otherwise it messes up the phase
###########
# options #
###########
simu_flag = False  # set to True if it is simulation,
# the parameter invert_phase will be set to 0
invert_phase = True  # True for the displacement to have the right sign
# (FFT convention), False only for simulations
flip_reconstruction = False  # True if you want to get the conjugate object
phase_ramp_removal = (
    "gradient"  # 'gradient'  # 'gradient' or 'upsampling', 'gradient' is much faster
)
threshold_gradient = (
    1.0  # upper threshold of the gradient of the phase, use for ramp removal
)
save_raw = False  # True to save the amp-phase.vti before orthogonalization
save_support = (
    False  # True to save the non-orthogonal support for later phase retrieval
)
save = True  # True to save amp.npz, phase.npz, strain.npz and vtk files
debug = False  # set to True to show all plots for debugging
roll_modes = (
    0,
    0,
    0,
)  # axis=(0, 1, 2), correct a roll of few pixels
# after the decomposition into modes in PyNX
############################################
# parameters related to data visualization #
############################################
align_axis = False  # for visualization, if True rotates the crystal to align
# axis_to_align along ref_axis after the
# calculation of the strain
ref_axis = "y"  # will align axis_to_align to that axis
axis_to_align = np.array(
    [-0.011662456997498807, 0.957321364700986, -0.28879022106682123]
)
# axis to align with ref_axis in the order x y z (axis 2, axis 1, axis 0)
strain_range = 0.003  # for plots
phase_range = np.pi / 2  # for plots
grey_background = True  # True to set the background to grey in phase and strain plots
tick_spacing = 50  # for plots, in nm
tick_direction = "inout"  # 'out', 'in', 'inout'
tick_length = 3  # 10  # in plots
tick_width = 1  # 2  # in plots
##########################################
# parameteres for temperature estimation #
##########################################
get_temperature = False  # only available for platinum at the moment
reflection = np.array(
    [1, 1, 1]
)  # measured reflection, use for estimating the temperature
reference_spacing = None  # for calibrating the thermal expansion,
# if None it is fixed to 3.9236/norm(reflection) Pt
reference_temperature = (
    None  # used to calibrate the thermal expansion, if None it is fixed to 293.15K (RT)
)
##########################################################
# parameters for averaging several reconstructed objects #
##########################################################
avg_method = "reciprocal_space"  # 'real_space' or 'reciprocal_space'
avg_threshold = 0.90  # minimum correlation within reconstructed object for averaging
############################################
# setup for phase averaging or apodization #
############################################
hwidth = (
    0  # (width-1)/2 of the averaging window for the phase, 0 means no phase averaging
)
apodize_flag = False  # True to multiply the diffraction pattern by a filtering window
apodize_window = (
    "blackman"  # filtering window, multivariate 'normal' or 'tukey' or 'blackman'
)
mu = np.array([0.0, 0.0, 0.0])  # mu of the gaussian window
sigma = np.array([0.30, 0.30, 0.30])  # sigma of the gaussian window
alpha = np.array([1.0, 1.0, 1.0])  # shape parameter of the tukey window
##################################
# end of user-defined parameters #
##################################

####################
# Check parameters #
####################
valid_name = "bcdi_strain"
if simu_flag:
    invert_phase = False
    correct_absorption = 0
    correct_refraction = 0

if invert_phase:
    phase_fieldname = "disp"
else:
    phase_fieldname = "phase"

if fix_voxel:
    if isinstance(fix_voxel, Real):
        fix_voxel = (fix_voxel, fix_voxel, fix_voxel)
    if not isinstance(fix_voxel, Sequence):
        raise TypeError("fix_voxel should be a sequence of three positive numbers")
    if any(val <= 0 for val in fix_voxel):
        raise ValueError(
            "fix_voxel should be a positive number or "
            "a sequence of three positive numbers"
        )

if actuators is not None and not isinstance(actuators, dict):
    raise TypeError("actuators should be a dictionnary of actuator fieldnames")

if data_frame not in {"detector", "crystal", "laboratory"}:
    raise ValueError('Uncorrect setting for "data_frame" parameter')
if data_frame == "detector":
    is_orthogonal = False
else:
    is_orthogonal = True

if ref_axis_q not in {"x", "y", "z"}:
    raise ValueError("ref_axis_q should be either 'x', 'y', 'z'")

if ref_axis not in {"x", "y", "z"}:
    raise ValueError("ref_axis should be either 'x', 'y', 'z'")

if save_frame not in {"crystal", "laboratory", "lab_flat_sample"}:
    raise ValueError(
        "save_frame should be either 'crystal', 'laboratory' or 'lab_flat_sample'"
    )

if data_frame == "crystal" and save_frame != "crystal":
    print(
        "data already in the crystal frame before phase retrieval,"
        " it is impossible to come back to the laboratory "
        "frame, parameter 'save_frame' defaulted to 'crystal'"
    )
    save_frame = "crystal"

if isinstance(output_size, Real):
    output_size = (output_size,) * 3
valid.valid_container(
    output_size,
    container_types=(tuple, list, np.ndarray),
    length=3,
    allow_none=True,
    item_types=int,
    name=valid_name,
)
axis_to_array_xyz = {
    "x": np.array([1, 0, 0]),
    "y": np.array([0, 1, 0]),
    "z": np.array([0, 0, 1]),
}  # in xyz order

if isinstance(save_dir, str) and not save_dir.endswith("/"):
    save_dir += "/"

if len(comment) != 0 and not comment.startswith("_"):
    comment = "_" + comment

##################################################
# parameters that will be saved with the results #
##################################################
params = {
    "isosurface_threshold": isosurface_strain,
    "strain_method": strain_method,
    "phase_offset": phase_offset,
    "phase_offset_origin": phase_offset_origin,
    "centering_method": centering_method,
    "data_frame": data_frame,
    "ref_axis_q": ref_axis_q,
    "save_frame": save_frame,
    "fix_voxel": fix_voxel,
    "original_size": original_size,
    "sample": f"{sample_name}+{scan}",
    "correct_refraction": correct_refraction,
    "optical_path_method": optical_path_method,
    "dispersion": dispersion,
    "time": f"{datetime.now()}",
    "threshold_unwrap_refraction": threshold_unwrap_refraction,
    "invert_phase": invert_phase,
    "phase_ramp_removal": phase_ramp_removal,
    "threshold_gradient": threshold_gradient,
    "tick_spacing_nm": tick_spacing,
    "hwidth": hwidth,
    "apodize_flag": apodize_flag,
    "apodize_window": apodize_window,
    "apod_mu": mu,
    "apod_sigma": sigma,
    "apod_alpha": alpha,
}
pretty = pprint.PrettyPrinter(indent=4)

###################
# define colormap #
###################
if grey_background:
    bad_color = "0.7"
else:
    bad_color = "1.0"  # white background
colormap = gu.Colormap(bad_color=bad_color)
my_cmap = colormap.cmap

#######################
# Initialize detector #
#######################
kwargs = {
    "preprocessing_binning": preprocessing_binning,
    "nb_pixel_x": nb_pixel_x,  # fix to declare a known detector but with less pixels
    # (e.g. one tile HS)
    "nb_pixel_y": nb_pixel_y,  # fix to declare a known detector but with less pixels
    # (e.g. one tile HS)
    "pixel_size": pixel_size,  # to declare the pixel size of the "Dummy" detector
}

detector = exp.Detector(
    name=detector,
    template_imagefile=template_imagefile,
    binning=phasing_binning,
    **kwargs,
)

####################################
# define the experimental geometry #
####################################
# correct the tilt_angle for binning
tilt_angle = tilt_angle * preprocessing_binning[0] * phasing_binning[0]
setup = exp.Setup(
    beamline=beamline,
    detector=detector,
    energy=energy,
    outofplane_angle=outofplane_angle,
    inplane_angle=inplane_angle,
    tilt_angle=tilt_angle,
    rocking_angle=rocking_angle,
    distance=sdd,
    sample_offsets=sample_offsets,
    actuators=actuators,
    custom_scan=custom_scan,
    custom_motors=custom_motors,
)

########################################
# Initialize the paths and the logfile #
########################################
setup.init_paths(
    sample_name=sample_name,
    scan_number=scan,
    root_folder=root_folder,
    save_dir=save_dir,
    specfile_name=specfile_name,
    template_imagefile=template_imagefile,
    create_savedir=True,
)

logfile = setup.create_logfile(
    scan_number=scan, root_folder=root_folder, filename=detector.specfile
)

#########################################################
# get the motor position of goniometer circles which    #
# are below the rocking angle (e.g., chi for eta/omega) #
#########################################################
_, setup.grazing_angle, _, _ = setup.diffractometer.goniometer_values(
    logfile=logfile, scan_number=scan, setup=setup
)

###################
# print instances #
###################
print(f'{"#"*(5+len(str(scan)))}\nScan {scan}\n{"#"*(5+len(str(scan)))}')
print("\n##############\nSetup instance\n##############")
pretty.pprint(setup.params)
print("\n#################\nDetector instance\n#################")
pretty.pprint(detector.params)

################
# preload data #
################
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilenames(
    initialdir=detector.scandir,
    filetypes=[("NPZ", "*.npz"), ("NPY", "*.npy"), ("CXI", "*.cxi"), ("HDF5", "*.h5")],
)
nbfiles = len(file_path)
plt.ion()

obj, extension = util.load_file(file_path[0])
if extension == ".h5":
    comment = comment + "_mode"

print("\n###############\nProcessing data\n###############")
nz, ny, nx = obj.shape
print("Initial data size: (", nz, ",", ny, ",", nx, ")")
if len(original_size) == 0:
    original_size = obj.shape
print("FFT size before accounting for phasing_binning", original_size)
original_size = tuple(
    [
        original_size[index] // phasing_binning[index]
        for index in range(len(phasing_binning))
    ]
)
print("Binning used during phasing:", detector.binning)
print("Padding back to original FFT size", original_size)
obj = util.crop_pad(array=obj, output_shape=original_size)
nz, ny, nx = obj.shape

###########################################################################
# define range for orthogonalization and plotting - speed up calculations #
###########################################################################
zrange, yrange, xrange = pu.find_datarange(
    array=obj, amplitude_threshold=0.05, keep_size=keep_size
)

numz = zrange * 2
numy = yrange * 2
numx = xrange * 2
print(f"Data shape used for orthogonalization and plotting: ({numz}, {numy}, {numx})")

####################################################################################
# find the best reconstruction from the list, based on mean amplitude and variance #
####################################################################################
if nbfiles > 1:
    print("\nTrying to find the best reconstruction\nSorting by ", sort_method)
    sorted_obj = pu.sort_reconstruction(
        file_path=file_path,
        amplitude_threshold=isosurface_strain,
        data_range=(zrange, yrange, xrange),
        sort_method="variance/mean",
    )
else:
    sorted_obj = [0]

#######################################
# load reconstructions and average it #
#######################################
avg_obj = np.zeros((numz, numy, numx))
ref_obj = np.zeros((numz, numy, numx))
avg_counter = 1
print("\nAveraging using", nbfiles, "candidate reconstructions")
for counter, value in enumerate(sorted_obj):
    obj, extension = util.load_file(file_path[value])
    print("\nOpening ", file_path[value])
    params[f"from_file_{counter}"] = file_path[value]

    if flip_reconstruction:
        obj = pu.flip_reconstruction(obj, debugging=True)

    if extension == ".h5":
        centering_method = "do_nothing"  # do not center, data is already cropped
        # just on support for mode decomposition
        # correct a roll after the decomposition into modes in PyNX
        obj = np.roll(obj, roll_modes, axis=(0, 1, 2))
        fig, _, _ = gu.multislices_plot(
            abs(obj),
            sum_frames=True,
            plot_colorbar=True,
            title="1st mode after centering",
        )
        fig.waitforbuttonpress()
        plt.close(fig)
    # use the range of interest defined above
    obj = util.crop_pad(obj, [2 * zrange, 2 * yrange, 2 * xrange], debugging=False)

    # align with average reconstruction
    if counter == 0:  # the fist array loaded will serve as reference object
        print("This reconstruction will be used as reference.")
        ref_obj = obj

    avg_obj, flag_avg = pu.average_obj(
        avg_obj=avg_obj,
        ref_obj=ref_obj,
        obj=obj,
        support_threshold=0.25,
        correlation_threshold=avg_threshold,
        aligning_option="dft",
        method=avg_method,
        reciprocal_space=False,
        is_orthogonal=is_orthogonal,
        debugging=debug,
    )
    avg_counter = avg_counter + flag_avg

avg_obj = avg_obj / avg_counter
if avg_counter > 1:
    print("\nAverage performed over ", avg_counter, "reconstructions\n")
del obj, ref_obj
gc.collect()

################
# unwrap phase #
################
phase, extent_phase = pu.unwrap(
    avg_obj,
    support_threshold=threshold_unwrap_refraction,
    debugging=debug,
    reciprocal_space=False,
    is_orthogonal=is_orthogonal,
)

print(
    "Extent of the phase over an extended support (ceil(phase range)) ~ ",
    int(extent_phase),
    "(rad)",
)
phase = pru.wrap(phase, start_angle=-extent_phase / 2, range_angle=extent_phase)
if debug:
    gu.multislices_plot(
        phase,
        width_z=2 * zrange,
        width_y=2 * yrange,
        width_x=2 * xrange,
        plot_colorbar=True,
        title="Phase after unwrap + wrap",
        reciprocal_space=False,
        is_orthogonal=is_orthogonal,
    )

#############################################
# phase ramp removal before phase filtering #
#############################################
amp, phase, rampz, rampy, rampx = pu.remove_ramp(
    amp=abs(avg_obj),
    phase=phase,
    initial_shape=original_size,
    method="gradient",
    amplitude_threshold=isosurface_strain,
    gradient_threshold=threshold_gradient,
)
del avg_obj
gc.collect()

if debug:
    gu.multislices_plot(
        phase,
        width_z=2 * zrange,
        width_y=2 * yrange,
        width_x=2 * xrange,
        plot_colorbar=True,
        title="Phase after ramp removal",
        reciprocal_space=False,
        is_orthogonal=is_orthogonal,
    )

########################
# phase offset removal #
########################
support = np.zeros(amp.shape)
support[amp > isosurface_strain * amp.max()] = 1
phase = pu.remove_offset(
    array=phase,
    support=support,
    offset_method=offset_method,
    user_offset=phase_offset,
    offset_origin=phase_offset_origin,
    title="Phase",
    debugging=debug,
)
del support
gc.collect()

phase = pru.wrap(obj=phase, start_angle=-extent_phase / 2, range_angle=extent_phase)

##############################################################################
# average the phase over a window or apodize to reduce noise in strain plots #
##############################################################################
if hwidth != 0:
    bulk = pu.find_bulk(
        amp=amp, support_threshold=isosurface_strain, method="threshold"
    )
    # the phase should be averaged only in the support defined by the isosurface
    phase = pu.mean_filter(array=phase, support=bulk, half_width=hwidth)
    del bulk
    gc.collect()

if hwidth != 0:
    comment = comment + "_avg" + str(2 * hwidth + 1)

gridz, gridy, gridx = np.meshgrid(
    np.arange(0, numz, 1), np.arange(0, numy, 1), np.arange(0, numx, 1), indexing="ij"
)

phase = (
    phase + gridz * rampz + gridy * rampy + gridx * rampx
)  # put back the phase ramp otherwise the diffraction
# pattern will be shifted and the prtf messed up

if apodize_flag:
    amp, phase = pu.apodize(
        amp=amp,
        phase=phase,
        initial_shape=original_size,
        window_type=apodize_window,
        sigma=sigma,
        mu=mu,
        alpha=alpha,
        is_orthogonal=is_orthogonal,
        debugging=True,
    )
    comment = comment + "_apodize_" + apodize_window

################################################################
# save the phase with the ramp for PRTF calculations,          #
# otherwise the object will be misaligned with the measurement #
################################################################
np.savez_compressed(
    detector.savedir + "S" + str(scan) + "_avg_obj_prtf" + comment,
    obj=amp * np.exp(1j * phase),
)

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
if centering_method == "max":
    avg_obj = pu.center_max(avg_obj)
    # shift based on max value,
    # required if it spans across the edge of the array before COM
elif centering_method == "com":
    avg_obj = pu.center_com(avg_obj)
elif centering_method == "max_com":
    avg_obj = pu.center_max(avg_obj)
    avg_obj = pu.center_com(avg_obj)

#######################
#  save support & vti #
#######################
if (
    save_support
):  # to be used as starting support in phasing, hence still in the detector frame
    support = np.zeros((numz, numy, numx))
    support[abs(avg_obj) / abs(avg_obj).max() > 0.01] = 1
    # low threshold because support will be cropped by shrinkwrap during phasing
    np.savez_compressed(
        detector.savedir + "S" + str(scan) + "_support" + comment, obj=support
    )
    del support
    gc.collect()

if save_raw:
    np.savez_compressed(
        detector.savedir + "S" + str(scan) + "_raw_amp-phase" + comment,
        amp=abs(avg_obj),
        phase=np.angle(avg_obj),
    )

    # voxel sizes in the detector frame
    voxel_z, voxel_y, voxel_x = setup.voxel_sizes_detector(
        array_shape=original_size,
        tilt_angle=tilt_angle,
        pixel_x=detector.pixelsize_x,
        pixel_y=detector.pixelsize_y,
        verbose=True,
    )
    # save raw amp & phase to VTK
    # in VTK, x is downstream, y vertical, z inboard, thus need to flip the last axis
    gu.save_to_vti(
        filename=os.path.join(
            detector.savedir, "S" + str(scan) + "_raw_amp-phase" + comment + ".vti"
        ),
        voxel_size=(voxel_z, voxel_y, voxel_x),
        tuple_array=(abs(avg_obj), np.angle(avg_obj)),
        tuple_fieldnames=("amp", "phase"),
        amplitude_threshold=0.01,
    )

#########################################################
# calculate q of the Bragg peak in the laboratory frame #
#########################################################
q_lab = (
    setup.q_laboratory
)  # (1/A), in the laboratory frame z downstream, y vertical, x outboard
qnorm = np.linalg.norm(q_lab)
q_lab = q_lab / qnorm

angle = simu.angle_vectors(
    ref_vector=[q_lab[2], q_lab[1], q_lab[0]], test_vector=axis_to_array_xyz[ref_axis_q]
)
print(
    f"\nNormalized diffusion vector in the laboratory frame (z*, y*, x*): "
    f"({q_lab[0]:.4f} 1/A, {q_lab[1]:.4f} 1/A, {q_lab[2]:.4f} 1/A)"
)

planar_dist = 2 * np.pi / qnorm  # qnorm should be in angstroms
print(f"Wavevector transfer: {qnorm:.4f} 1/A")
print(f"Atomic planar distance: {planar_dist:.4f} A")
print(f"\nAngle between q_lab and {ref_axis_q} = {angle:.2f} deg")
if debug:
    print(
        f"Angle with y in zy plane = {np.arctan(q_lab[0]/q_lab[1])*180/np.pi:.2f} deg"
    )
    print(
        f"Angle with y in xy plane = {np.arctan(-q_lab[2]/q_lab[1])*180/np.pi:.2f} deg"
    )
    print(
        f"Angle with z in xz plane = {180+np.arctan(q_lab[2]/q_lab[0])*180/np.pi:.2f} "
        "deg\n"
    )

planar_dist = planar_dist / 10  # switch to nm

#######################
#  orthogonalize data #
#######################
print("\nShape before orthogonalization", avg_obj.shape)
if data_frame == "detector":
    if debug:
        phase, _ = pu.unwrap(
            avg_obj,
            support_threshold=threshold_unwrap_refraction,
            debugging=True,
            reciprocal_space=False,
            is_orthogonal=False,
        )
        gu.multislices_plot(
            phase,
            width_z=2 * zrange,
            width_y=2 * yrange,
            width_x=2 * xrange,
            sum_frames=False,
            plot_colorbar=True,
            reciprocal_space=False,
            is_orthogonal=False,
            title="unwrapped phase before orthogonalization",
        )
        del phase
        gc.collect()

    obj_ortho, voxel_size = setup.ortho_directspace(
        arrays=avg_obj,
        q_com=np.array([q_lab[2], q_lab[1], q_lab[0]]),
        initial_shape=original_size,
        voxel_size=fix_voxel,
        reference_axis=axis_to_array_xyz[ref_axis_q],
        fill_value=0,
        debugging=True,
        title="amplitude",
    )

else:  # data already orthogonalized using xrayutilities
    # or the linearized transformation matrix
    obj_ortho = avg_obj
    try:
        print("Select the file containing QxQzQy")
        file_path = filedialog.askopenfilename(
            title="Select the file containing QxQzQy",
            initialdir=detector.savedir,
            filetypes=[("NPZ", "*.npz")],
        )
        npzfile = np.load(file_path)
        qx = npzfile["qx"]
        qy = npzfile["qy"]
        qz = npzfile["qz"]
    except FileNotFoundError:
        raise FileNotFoundError(
            "q values not provided, the voxel size cannot be calculated"
        )
    dy_real = (
        2 * np.pi / abs(qz.max() - qz.min()) / 10
    )  # in nm qz=y in nexus convention
    dx_real = (
        2 * np.pi / abs(qy.max() - qy.min()) / 10
    )  # in nm qy=x in nexus convention
    dz_real = (
        2 * np.pi / abs(qx.max() - qx.min()) / 10
    )  # in nm qx=z in nexus convention
    print(
        f"direct space voxel size from q values: ({dz_real:.2f} nm,"
        f" {dy_real:.2f} nm, {dx_real:.2f} nm)"
    )
    if fix_voxel:
        voxel_size = fix_voxel
        print(f"Direct space pixel size for the interpolation: {voxel_size} (nm)")
        print("Interpolating...\n")
        obj_ortho = pu.regrid(
            array=obj_ortho,
            old_voxelsize=(dz_real, dy_real, dx_real),
            new_voxelsize=voxel_size,
        )
    else:
        # no need to interpolate
        voxel_size = dz_real, dy_real, dx_real  # in nm

    if (
        data_frame == "laboratory"
    ):  # the object must be rotated into the crystal frame
        # before the strain calculation
        print("Rotating the object in the crystal frame for the strain calculation")

        amp, phase = util.rotate_crystal(
            arrays=(abs(obj_ortho), np.angle(obj_ortho)),
            is_orthogonal=True,
            reciprocal_space=False,
            voxel_size=voxel_size,
            debugging=(True, False),
            axis_to_align=q_lab[::-1],
            reference_axis=axis_to_array_xyz[ref_axis_q],
            title=("amp", "phase"),
        )

        obj_ortho = amp * np.exp(
            1j * phase
        )  # here the phase is again wrapped in [-pi pi[
        del amp, phase

del avg_obj
gc.collect()

######################################################
# center the object (centering based on the modulus) #
######################################################
print("\nCentering the crystal")
obj_ortho = pu.center_com(obj_ortho)

####################
# Phase unwrapping #
####################
print("\nPhase unwrapping")
phase, extent_phase = pu.unwrap(
    obj_ortho,
    support_threshold=threshold_unwrap_refraction,
    debugging=True,
    reciprocal_space=False,
    is_orthogonal=True,
)
amp = abs(obj_ortho)
del obj_ortho
gc.collect()

#############################################
# invert phase: -1*phase = displacement * q #
#############################################
if invert_phase:
    phase = -1 * phase

########################################
# refraction and absorption correction #
########################################
if correct_refraction:  # or correct_absorption:
    bulk = pu.find_bulk(
        amp=amp,
        support_threshold=threshold_unwrap_refraction,
        method=optical_path_method,
        debugging=debug,
    )

    kin = setup.incident_wavevector
    kout = setup.exit_wavevector
    # kin and kout were calculated in the laboratory frame,
    # but after the geometric transformation of the crystal, this
    # latter is always in the crystal frame (for simpler strain calculation).
    # We need to transform kin and kout back
    # into the crystal frame (also, xrayutilities output is in crystal frame)
    kin = util.rotate_vector(
        vectors=[kin[2], kin[1], kin[0]],
        axis_to_align=axis_to_array_xyz[ref_axis_q],
        reference_axis=[q_lab[2], q_lab[1], q_lab[0]],
    )
    kout = util.rotate_vector(
        vectors=[kout[2], kout[1], kout[0]],
        axis_to_align=axis_to_array_xyz[ref_axis_q],
        reference_axis=[q_lab[2], q_lab[1], q_lab[0]],
    )

    # calculate the optical path of the incoming wavevector
    path_in = pu.get_opticalpath(
        support=bulk, direction="in", k=kin, debugging=debug
    )  # path_in already in nm

    # calculate the optical path of the outgoing wavevector
    path_out = pu.get_opticalpath(
        support=bulk, direction="out", k=kout, debugging=debug
    )  # path_our already in nm

    optical_path = path_in + path_out
    del path_in, path_out
    gc.collect()

    if correct_refraction:
        phase_correction = (
            2 * np.pi / (1e9 * setup.wavelength) * dispersion * optical_path
        )
        phase = phase + phase_correction

        gu.multislices_plot(
            np.multiply(phase_correction, bulk),
            width_z=2 * zrange,
            width_y=2 * yrange,
            width_x=2 * xrange,
            sum_frames=False,
            plot_colorbar=True,
            vmin=0,
            vmax=np.nan,
            title="Refraction correction on the support",
            is_orthogonal=True,
            reciprocal_space=False,
        )
    correct_absorption = False
    if correct_absorption:
        # TODO: it is correct to compensate also
        #  the X-ray absorption in the reconstructed modulus?
        amp_correction = np.exp(
            2 * np.pi / (1e9 * setup.wavelength) * absorption * optical_path
        )
        amp = amp * amp_correction

        gu.multislices_plot(
            np.multiply(amp_correction, bulk),
            width_z=2 * zrange,
            width_y=2 * yrange,
            width_x=2 * xrange,
            sum_frames=False,
            plot_colorbar=True,
            vmin=1,
            vmax=1.1,
            title="Absorption correction on the support",
            is_orthogonal=True,
            reciprocal_space=False,
        )

    del bulk, optical_path
    gc.collect()

##############################################
# phase ramp and offset removal (mean value) #
##############################################
print("\nPhase ramp removal")
amp, phase, _, _, _ = pu.remove_ramp(
    amp=amp,
    phase=phase,
    initial_shape=original_size,
    method=phase_ramp_removal,
    amplitude_threshold=isosurface_strain,
    gradient_threshold=threshold_gradient,
    debugging=debug,
)

########################
# phase offset removal #
########################
print("\nPhase offset removal")
support = np.zeros(amp.shape)
support[amp > isosurface_strain * amp.max()] = 1
phase = pu.remove_offset(
    array=phase,
    support=support,
    offset_method=offset_method,
    user_offset=phase_offset,
    offset_origin=phase_offset_origin,
    title="Orthogonal phase",
    debugging=debug,
    reciprocal_space=False,
    is_orthogonal=True,
)
del support
gc.collect()
# Wrap the phase around 0 (no more offset)
phase = pru.wrap(obj=phase, start_angle=-extent_phase / 2, range_angle=extent_phase)

################################################################
# calculate the strain depending on which axis q is aligned on #
################################################################
print(f"\nCalculation of the strain along {ref_axis_q}")
strain = pu.get_strain(
    phase=phase,
    planar_distance=planar_dist,
    voxel_size=voxel_size,
    reference_axis=ref_axis_q,
    extent_phase=extent_phase,
    method=strain_method,
    debugging=debug,
)

################################################
# optionally rotates back the crystal into the #
# laboratory frame (for debugging purpose)     #
################################################
q_final = None
if save_frame in {"laboratory", "lab_flat_sample"}:
    comment = comment + "_labframe"
    print("\nRotating back the crystal in laboratory frame")
    amp, phase, strain = util.rotate_crystal(
        arrays=(amp, phase, strain),
        axis_to_align=axis_to_array_xyz[ref_axis_q],
        voxel_size=voxel_size,
        is_orthogonal=True,
        reciprocal_space=False,
        reference_axis=[q_lab[2], q_lab[1], q_lab[0]],
        debugging=(True, False, False),
        title=("amp", "phase", "strain"),
    )
    # q_lab is already in the laboratory frame
    q_final = q_lab

if save_frame == "lab_flat_sample":
    comment = comment + "_flat"
    print("\nSending sample stage circles to 0")
    sample_angles = setup.diffractometer.goniometer_values(
        logfile=logfile, scan_number=scan, setup=setup, stage_name="sample"
    )
    (amp, phase, strain), q_final = setup.diffractometer.flatten_sample(
        arrays=(amp, phase, strain),
        voxel_size=voxel_size,
        angles=sample_angles,
        q_com=q_lab[::-1],  # q_com needs to be in xyz order
        is_orthogonal=True,
        reciprocal_space=False,
        rocking_angle=rocking_angle,
        debugging=(True, False, False),
        title=("amp", "phase", "strain"),
    )
if save_frame == "crystal":
    # rotate also q_lab to have it along ref_axis_q,
    # as a cross-checkm, vectors needs to be in xyz order
    comment = comment + "_crystalframe"
    q_final = util.rotate_vector(
        vectors=q_lab[::-1],
        axis_to_align=axis_to_array_xyz[ref_axis_q],
        reference_axis=q_lab[::-1],
    )

###############################################
# rotates the crystal e.g. for easier slicing #
# of the result along a particular direction  #
###############################################
# typically this is an inplane rotation, q should stay aligned with the axis
# along which the strain was calculated
if align_axis:
    print("\nRotating arrays for visualization")
    amp, phase, strain = util.rotate_crystal(
        arrays=(amp, phase, strain),
        reference_axis=axis_to_array_xyz[ref_axis],
        axis_to_align=axis_to_align,
        voxel_size=voxel_size,
        debugging=(True, False, False),
        is_orthogonal=True,
        reciprocal_space=False,
        title=("amp", "phase", "strain"),
    )
    # rotate q accordingly, vectors needs to be in xyz order
    q_final = util.rotate_vector(
        vectors=q_final[::-1],
        axis_to_align=axis_to_array_xyz[ref_axis],
        reference_axis=axis_to_align,
    )

print(f"\nq_final = ({q_final[0]:.4f} 1/A, {q_final[1]:.4f} 1/A, {q_final[2]:.4f} 1/A)")

##############################################
# pad array to fit the output_size parameter #
##############################################
if output_size is not None:
    amp = util.crop_pad(array=amp, output_shape=output_size)
    phase = util.crop_pad(array=phase, output_shape=output_size)
    strain = util.crop_pad(array=strain, output_shape=output_size)
print(f"\nFinal data shape: {amp.shape}")

######################
# save result to vtk #
######################
print(
    f"\nVoxel size: ({voxel_size[0]:.2f} nm, {voxel_size[1]:.2f} nm,"
    f" {voxel_size[2]:.2f} nm)"
)
bulk = pu.find_bulk(amp=amp, support_threshold=isosurface_strain, method="threshold")
if save:
    params["comment"] = comment
    np.savez_compressed(
        f"{detector.savedir}S{scan}_amp{phase_fieldname}strain{comment}",
        amp=amp,
        phase=phase,
        bulk=bulk,
        strain=strain,
        q_com=q_final,
        voxel_sizes=voxel_size,
        detector=detector.params,
        setup=setup.params,
        params=params,
    )

    # save results in hdf5 file
    with h5py.File(
        f"{detector.savedir}S{scan}_amp{phase_fieldname}strain{comment}.h5", "w"
    ) as hf:
        out = hf.create_group("output")
        par = hf.create_group("params")
        out.create_dataset("amp", data=amp)
        out.create_dataset("bulk", data=bulk)
        out.create_dataset("phase", data=phase)
        out.create_dataset("strain", data=strain)
        out.create_dataset("q_com", data=q_final)
        out.create_dataset("voxel_sizes", data=voxel_size)
        par.create_dataset("detector", data=str(detector.params))
        par.create_dataset("setup", data=str(setup.params))
        par.create_dataset("parameters", data=str(params))

    # save amp & phase to VTK
    # in VTK, x is downstream, y vertical, z inboard, thus need to flip the last axis
    gu.save_to_vti(
        filename=os.path.join(
            detector.savedir,
            "S" + str(scan) + "_amp-" + phase_fieldname + "-strain" + comment + ".vti",
        ),
        voxel_size=voxel_size,
        tuple_array=(amp, bulk, phase, strain),
        tuple_fieldnames=("amp", "bulk", phase_fieldname, "strain"),
        amplitude_threshold=0.01,
    )


######################################
# estimate the volume of the crystal #
######################################
amp = amp / amp.max()
temp_amp = np.copy(amp)
temp_amp[amp < isosurface_strain] = 0
temp_amp[np.nonzero(temp_amp)] = 1
volume = temp_amp.sum() * reduce(lambda x, y: x * y, voxel_size)  # in nm3
del temp_amp
gc.collect()

##############################
# plot slices of the results #
##############################
pixel_spacing = [tick_spacing / vox for vox in voxel_size]
print(
    "\nPhase extent without / with thresholding the modulus "
    f"(threshold={isosurface_strain}): {phase.max()-phase.min():.2f} rad, "
    f"{phase[np.nonzero(bulk)].max()-phase[np.nonzero(bulk)].min():.2f} rad"
)
piz, piy, pix = np.unravel_index(phase.argmax(), phase.shape)
print(
    f"phase.max() = {phase[np.nonzero(bulk)].max():.2f} at voxel ({piz}, {piy}, {pix})"
)
strain[bulk == 0] = np.nan
phase[bulk == 0] = np.nan

# plot the slice at the maximum phase
gu.combined_plots(
    (phase[piz, :, :], phase[:, piy, :], phase[:, :, pix]),
    tuple_sum_frames=False,
    tuple_sum_axis=0,
    tuple_width_v=None,
    tuple_width_h=None,
    tuple_colorbar=True,
    tuple_vmin=np.nan,
    tuple_vmax=np.nan,
    tuple_title=("phase at max in xy", "phase at max in xz", "phase at max in yz"),
    tuple_scale="linear",
    cmap=my_cmap,
    is_orthogonal=True,
    reciprocal_space=False,
)

# bulk support
fig, _, _ = gu.multislices_plot(
    bulk,
    sum_frames=False,
    title="Orthogonal bulk",
    vmin=0,
    vmax=1,
    is_orthogonal=True,
    reciprocal_space=False,
)
fig.text(0.60, 0.45, "Scan " + str(scan), size=20)
fig.text(
    0.60, 0.40, "Bulk - isosurface=" + str("{:.2f}".format(isosurface_strain)), size=20
)
plt.pause(0.1)
if save:
    plt.savefig(detector.savedir + "S" + str(scan) + "_bulk" + comment + ".png")

# amplitude
fig, _, _ = gu.multislices_plot(
    amp,
    sum_frames=False,
    title="Normalized orthogonal amp",
    vmin=0,
    vmax=1,
    tick_direction=tick_direction,
    tick_width=tick_width,
    tick_length=tick_length,
    pixel_spacing=pixel_spacing,
    plot_colorbar=True,
    is_orthogonal=True,
    reciprocal_space=False,
)
fig.text(0.60, 0.45, f"Scan {scan}", size=20)
fig.text(
    0.60,
    0.40,
    f"Voxel size=({voxel_size[0]:.1f}, {voxel_size[1]:.1f}, {voxel_size[2]:.1f}) (nm)",
    size=20,
)
fig.text(0.60, 0.35, f"Ticks spacing={tick_spacing} nm", size=20)
fig.text(0.60, 0.30, f"Volume={int(volume)} nm3", size=20)
fig.text(0.60, 0.25, "Sorted by " + sort_method, size=20)
fig.text(0.60, 0.20, f"correlation threshold={correlation_threshold}", size=20)
fig.text(0.60, 0.15, f"average over {avg_counter} reconstruction(s)", size=20)
fig.text(0.60, 0.10, f"Planar distance={planar_dist:.5f} nm", size=20)
if get_temperature:
    temperature = pu.bragg_temperature(
        spacing=planar_dist * 10,
        reflection=reflection,
        spacing_ref=reference_spacing,
        temperature_ref=reference_temperature,
        use_q=False,
        material="Pt",
    )
    fig.text(0.60, 0.05, f"Estimated T={temperature} C", size=20)
if save:
    plt.savefig(detector.savedir + f"S{scan}_amp" + comment + ".png")

# amplitude histogram
fig, ax = plt.subplots(1, 1)
ax.hist(amp[amp > 0.05 * amp.max()].flatten(), bins=250)
ax.set_ylim(bottom=1)
ax.tick_params(
    labelbottom=True,
    labelleft=True,
    direction="out",
    length=tick_length,
    width=tick_width,
)
ax.spines["right"].set_linewidth(1.5)
ax.spines["left"].set_linewidth(1.5)
ax.spines["top"].set_linewidth(1.5)
ax.spines["bottom"].set_linewidth(1.5)
fig.savefig(detector.savedir + f"S{scan}_histo_amp" + comment + ".png")

# phase
fig, _, _ = gu.multislices_plot(
    phase,
    sum_frames=False,
    title="Orthogonal displacement",
    vmin=-phase_range,
    vmax=phase_range,
    tick_direction=tick_direction,
    cmap=my_cmap,
    tick_width=tick_width,
    tick_length=tick_length,
    pixel_spacing=pixel_spacing,
    plot_colorbar=True,
    is_orthogonal=True,
    reciprocal_space=False,
)
fig.text(0.60, 0.30, f"Scan {scan}", size=20)
fig.text(
    0.60,
    0.25,
    f"Voxel size=({voxel_size[0]:.1f}, {voxel_size[1]:.1f}, {voxel_size[2]:.1f}) (nm)",
    size=20,
)
fig.text(0.60, 0.20, f"Ticks spacing={tick_spacing} nm", size=20)
fig.text(0.60, 0.15, f"average over {avg_counter} reconstruction(s)", size=20)
if hwidth > 0:
    fig.text(0.60, 0.10, f"Averaging over {2*hwidth+1} pixels", size=20)
else:
    fig.text(0.60, 0.10, "No phase averaging", size=20)
if save:
    plt.savefig(detector.savedir + f"S{scan}_displacement" + comment + ".png")

# strain
fig, _, _ = gu.multislices_plot(
    strain,
    sum_frames=False,
    title="Orthogonal strain",
    vmin=-strain_range,
    vmax=strain_range,
    tick_direction=tick_direction,
    tick_width=tick_width,
    tick_length=tick_length,
    plot_colorbar=True,
    cmap=my_cmap,
    pixel_spacing=pixel_spacing,
    is_orthogonal=True,
    reciprocal_space=False,
)
fig.text(0.60, 0.30, f"Scan {scan}", size=20)
fig.text(
    0.60,
    0.25,
    f"Voxel size=({voxel_size[0]:.1f}, {voxel_size[1]:.1f}, {voxel_size[2]:.1f}) (nm)",
    size=20,
)
fig.text(0.60, 0.20, f"Ticks spacing={tick_spacing} nm", size=20)
fig.text(0.60, 0.15, f"average over {avg_counter} reconstruction(s)", size=20)
if hwidth > 0:
    fig.text(0.60, 0.10, f"Averaging over {2*hwidth+1} pixels", size=20)
else:
    fig.text(0.60, 0.10, "No phase averaging", size=20)
if save:
    plt.savefig(detector.savedir + f"S{scan}_strain" + comment + ".png")


print("\nEnd of script")
plt.ioff()
plt.show()
