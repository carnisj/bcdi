#!/usr/bin/env python3

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import sys
import tkinter as tk
from tkinter import filedialog

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

import bcdi.postprocessing.postprocessing_utils as pu
import bcdi.preprocessing.bcdi_utils as bu
import bcdi.utils.utilities as util
from bcdi.experiment.setup import Setup

helptext = """
Calculate exact inplane and out-of-plane detector angles from the direct beam and
Bragg peak positions, based on the beamline geometry. For Pt samples it gives also an
estimation of the temperature based on the thermal expansion.

Supported beamlines: ESRF ID01, PETRAIII P10, SOLEIL SIXS, SOLEIL CRISTAL.

Input: direct beam and Bragg peak position, sample to detector distance, energy.
Output: corrected inplane, out-of-plane detector angles for the Bragg peak.
"""
scan = 622
root_folder = "C:/Users/Jerome/Documents/data/dataset_34ID/IzrO/"
data_dir = "C:/Users/Jerome/Documents/data/dataset_34ID/IzrO/S622/data/"
# leave None to use the beamline default. It will look for the data at this location
sample_name = "S"
filtered_data = False  # set to True if the data is already a 3D array, False otherwise
# Should be the same shape as in specfile
peak_method = "maxcom"  # Bragg peak determination: 'max', 'com' or 'maxcom'.
normalize_flux = "monitor"
# 'monitor' to normalize the intensity by the default monitor values,
# 'skip' to do nothing
debug = False  # True to see more plots
######################################
# define beamline related parameters #
######################################
beamline = "34ID"
# name of the beamline, used for data loading and normalization by monitor
# supported beamlines: 'ID01', 'SIXS_2018', 'SIXS_2019', 'CRISTAL', 'P10'
actuators = None
# Optional dictionary that can be used to define the entries corresponding to
# actuators in data files
# (useful at CRISTAL where the location of data keeps changing)
# e.g.  {'rocking_angle': 'actuator_1_3', 'detector': 'data_04', 'monitor': 'data_05'}
is_series = False  # specific to series measurement at P10
custom_scan = False  # True for a stack of images acquired without scan,
# e.g. with ct in a macro (no info in spec file)
custom_images = None
# list of image numbers for the custom_scan
custom_monitor = None
# monitor values for normalization for the custom_scan
custom_motors = None
# ID01: eta, phi, nu, delta
# CRISTAL: mgomega, gamma, delta
# P10: om, phi, chi, mu, gamma, delta
# SIXS: beta, mu, gamma, delta
rocking_angle = "inplane"  # "outofplane" or "inplane"
specfile_name = "Dmitry1120c.spec"
# template for ID01 and 34ID: name of the spec file if it is at the default location
# (in root_folder) or full path to the spec file
# template for SIXS_2018: full path of the alias dictionnary 'alias_dict.txt',
# typically: root_folder + 'alias_dict.txt'
# template for all other beamlines: ''
#############################################################
# define detector related parameters and region of interest #
#############################################################
detector = "Timepix"  # detector name
x_bragg = None  # horizontal pixel number of the Bragg peak,
# can be used for the definition of the ROI
y_bragg = None  # vertical pixel number of the Bragg peak,
# can be used for the definition of the ROI
roi_detector = None  # [y_bragg - 400, y_bragg + 400, x_bragg - 400, x_bragg + 400]  #
# leave it as None to use the full detector.
# Use with center_fft='do_nothing' if you want this exact size.
high_threshold = 500000  # everything above will be considered as hotpixel
hotpixels_file = None
# non empty file path or None
flatfield_file = None
# non empty file path or None
template_imagefile = "Dmitry1120c_S0622_%05d.tif"
# template for ID01: 'data_mpx4_%05d.edf.gz' or 'align_eiger2M_%05d.edf.gz'
# template for SIXS_2018: 'align.spec_ascan_mu_%05d.nxs'
# template for SIXS_2019: 'spare_ascan_mu_%05d.nxs'
# template for Cristal: 'S%d.nxs'
# template for P10: '_master.h5'
# template for NANOMAX: '%06d.h5'
# template for 34ID: "Dmitry1120c_S0622_%05d.tif"
###################################
# define setup related parameters #
###################################
beam_direction = (1, 0, 0)  # beam along z
sample_offsets = None
# tuple of offsets in degrees of the sample around (downstream, vertical up, outboard)
# convention: the sample offsets will be subtracted to the motor values
directbeam_x = 0  # x horizontal,  cch2 in xrayutilities
directbeam_y = 0  # y vertical,  cch1 in xrayutilities
direct_inplane = 0.0  # outer angle in xrayutilities
direct_outofplane = 0.0
sdd = 0.5002125  # sample to detector distance in m
energy = 9000  # in eV, offset of 6eV at ID01
################################################
# parameters related to temperature estimation #
################################################
get_temperature = False  # True to estimate the temperature using the reference
# spacing of the material. Only for Pt.
reflection = np.array([1, 1, 1])
# measured reflection, use for estimating the temperature
reference_spacing = None  # for calibrating the thermal expansion,
# if None it is fixed to Pt 3.9236/norm(reflection)
reference_temperature = None
# used to calibrate the thermal expansion, if None it is fixed to 293.15K (RT)
##########################
# end of user parameters #
##########################

plt.ion()

####################
# Initialize setup #
####################
setup = Setup(
    beamline_name=beamline,
    energy=energy,
    rocking_angle=rocking_angle,
    distance=sdd,
    beam_direction=beam_direction,
    custom_scan=custom_scan,
    custom_images=custom_images,
    custom_monitor=custom_monitor,
    custom_motors=custom_motors,
    sample_offsets=sample_offsets,
    actuators=actuators,
    is_series=is_series,
    detector_name=detector,
    template_imagefile=template_imagefile,
    roi=roi_detector,
)

########################################
# Initialize the paths and the logfile #
########################################
# initialize the paths
setup.init_paths(
    sample_name=sample_name,
    scan_number=scan,
    root_folder=root_folder,
    save_dir=None,
    specfile_name=specfile_name,
    template_imagefile=template_imagefile,
    data_dir=data_dir,
)

setup.create_logfile(
    scan_number=scan, root_folder=root_folder, filename=setup.detector.specfile
)

#################
# load the data #
#################
flatfield = util.load_flatfield(flatfield_file)
hotpix_array = util.load_hotpixels(hotpixels_file)

if not filtered_data:
    data, _, monitor, frames_logical = setup.loader.load_check_dataset(
        scan_number=scan,
        setup=setup,
        flatfield=flatfield,
        hotpixels=hotpix_array,
        normalize=normalize_flux,
        debugging=debug,
    )
else:
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        initialdir=setup.detector.scandir + "pynxraw/",
        title="Select 3D data",
        filetypes=[("NPZ", "*.npz")],
    )
    data = np.load(file_path)["data"]
    data = data[
        setup.detector.roi[0] : setup.detector.roi[1],
        setup.detector.roi[2] : setup.detector.roi[3],
    ]
    frames_logical = np.ones(data.shape[0]).astype(
        int
    )  # use all frames from the filtered data
numz, numy, numx = data.shape
print("Shape of dataset: ", numz, numy, numx)

##############################################
# apply photon threshold to remove hotpixels #
##############################################
if high_threshold != 0:
    nb_thresholded = (data > high_threshold).sum()
    data[data > high_threshold] = 0
    print(f"Applying photon threshold, {nb_thresholded} high intensity pixels masked")

###############################
# load releavant motor values #
###############################
(
    tilt_values,
    setup.grazing_angle,
    setup.inplane_angle,
    setup.outofplane_angle,
) = setup.read_logfile(scan_number=scan)

nb_frames = len(tilt_values)
if numz != nb_frames:
    print("The loaded data has not the same shape as the raw data")
    sys.exit()

#######################
# Find the Bragg peak #
#######################
z0, y0, x0 = bu.find_bragg(data, peak_method=peak_method)

######################################################
# calculate rocking curve and fit it to get the FWHM #
######################################################
rocking_curve = np.zeros(nb_frames)
if filtered_data == 0:  # take a small ROI to avoid parasitic peaks
    for idx in range(nb_frames):
        rocking_curve[idx] = data[idx, y0 - 20 : y0 + 20, x0 - 20 : x0 + 20].sum()
    plot_title = "Rocking curve for a 40x40 pixels ROI"
else:  # take the whole detector
    for idx in range(nb_frames):
        rocking_curve[idx] = data[idx, :, :].sum()
    plot_title = "Rocking curve (full detector)"
z0 = np.unravel_index(rocking_curve.argmax(), rocking_curve.shape)[0]

interpolation = interp1d(tilt_values, rocking_curve, kind="cubic")
interp_points = 5 * nb_frames
interp_tilt = np.linspace(tilt_values.min(), tilt_values.max(), interp_points)
interp_curve = interpolation(interp_tilt)
interp_fwhm = (
    len(np.argwhere(interp_curve >= interp_curve.max() / 2))
    * (tilt_values.max() - tilt_values.min())
    / (interp_points - 1)
)
print(f"FWHM by interpolation {interp_fwhm:.3f} deg")

fig, (ax0, ax1) = plt.subplots(2, 1, sharex="col", figsize=(10, 5))
ax0.plot(tilt_values, rocking_curve, ".")
ax0.plot(interp_tilt, interp_curve)
ax0.set_ylabel("Integrated intensity")
ax0.legend(("data", "interpolation"))
ax0.set_title(plot_title)
ax1.plot(tilt_values, np.log10(rocking_curve), ".")
ax1.plot(interp_tilt, np.log10(interp_curve))
ax1.set_xlabel("Rocking angle (deg)")
ax1.set_ylabel("Log(integrated intensity)")
ax0.legend(("data", "interpolation"))
plt.pause(0.1)

##############################
# Calculate corrected angles #
##############################
bragg_x = setup.detector.roi[2] + x0  # convert it in full detector pixel
bragg_y = setup.detector.roi[0] + y0  # convert it in full detector pixel

x_direct_0 = directbeam_x + setup.inplane_coeff * (
    direct_inplane * np.pi / 180 * sdd / setup.detector.pixelsize_x
)  # inplane_coeff is +1 or -1
y_direct_0 = (
    directbeam_y
    - setup.outofplane_coeff
    * direct_outofplane
    * np.pi
    / 180
    * sdd
    / setup.detector.pixelsize_y
)  # outofplane_coeff is +1 or -1

print(
    f"\nDirect beam at (gam={direct_inplane}, "
    f"del={direct_outofplane}) (X, Y): {directbeam_x}, {directbeam_y}"
)
print(f"Direct beam at (gam=0, del=0) (X, Y): ({x_direct_0:.2f}, {y_direct_0:.2f})")
print(
    f"\nBragg peak at (gam={setup.inplane_angle}, "
    f"del={setup.outofplane_angle}) (X, Y): ({bragg_x:.2f}, {bragg_y:.2f})"
)

bragg_inplane = setup.inplane_angle + setup.inplane_coeff * (
    setup.detector.pixelsize_x * (bragg_x - x_direct_0) / sdd * 180 / np.pi
)  # inplane_coeff is +1 or -1
bragg_outofplane = (
    setup.outofplane_angle
    - setup.outofplane_coeff
    * setup.detector.pixelsize_y
    * (bragg_y - y_direct_0)
    / sdd
    * 180
    / np.pi
)  # outofplane_coeff is +1 or -1

print(
    f"\nBragg angles before correction (gam, del): ({setup.inplane_angle:.4f}, "
    f"{setup.outofplane_angle:.4f})"
)
print(
    f"Bragg angles after correction (gam, del): ({bragg_inplane:.4f}, "
    f"{bragg_outofplane:.4f})"
)

# update setup with the corrected detector angles
setup.inplane_angle = bragg_inplane
setup.outofplane_angle = bragg_outofplane

print(f"\nGrazing angle(s) = {setup.grazing_angle} deg")
print(f"Rocking step = {setup.tilt_angle:.5f} deg")

####################################
# wavevector transfer calculations #
####################################
kin = (
    2 * np.pi / setup.wavelength * np.asarray(beam_direction)
)  # in lab frame z downstream, y vertical, x outboard
kout = setup.exit_wavevector  # in lab.frame z downstream, y vertical, x outboard
q = (kout - kin) / 1e10  # convert from 1/m to 1/angstrom
qnorm = np.linalg.norm(q)
dist_plane = 2 * np.pi / qnorm
print(f"\nWavevector transfer of Bragg peak: {q}, Qnorm={qnorm:.4f}")
print(f"Interplanar distance: {dist_plane:.6f} angstroms")

if get_temperature:
    print("\nEstimating the temperature:")
    temperature = pu.bragg_temperature(
        spacing=dist_plane,
        reflection=reflection,
        spacing_ref=reference_spacing,
        temperature_ref=reference_temperature,
        use_q=False,
        material="Pt",
    )

#########################
# calculate voxel sizes #
#########################
#  update the detector angles in setup
setup.inplane_angle = bragg_inplane
setup.outofplane_angle = bragg_outofplane
dz_realspace, dy_realspace, dx_realspace = setup.voxel_sizes(
    (nb_frames, numy, numx),
    tilt_angle=setup.tilt_angle,
    pixel_x=setup.detector.pixelsize_x,
    pixel_y=setup.detector.pixelsize_y,
    verbose=True,
)

#################################
# plot image at Bragg condition #
#################################
plt.figure()
plt.imshow(np.log10(abs(data[int(round(z0)), :, :])), vmin=0, vmax=5)
plt.title(f"Central slice at frame {int(np.rint(z0))}")
plt.colorbar()
plt.ioff()
plt.show()
