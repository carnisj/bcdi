# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

try:
    import hdf5plugin  # for P10, should be imported before h5py or PyTables
except ModuleNotFoundError:
    pass
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import tkinter as tk
from tkinter import filedialog
import sys
sys.path.append('//win.desy.de/home/carnisj/My Documents/myscripts/bcdi/')
import bcdi.postprocessing.postprocessing_utils as pu
import bcdi.preprocessing.preprocessing_utils as pru
import bcdi.experiment.experiment_utils as exp

helptext = """
Calculate exact inplane and out-of-plane detector angles from the direct beam and Bragg peak positions,
based on the beamline geometry.

Supported beamlines: ESRF ID01, PETRAIII P10, SOLEIL SIXS, SOLEIL CRISTAL.

For Pt samples it gives also an estimation of the temperature based on the thermal expansion.

Input: direct beam and Bragg peak position, sample to detector distance, energy
Output: corrected inplane, out-of-plane detector angles for the Bragg peak.
"""
scan = 11
root_folder = "D:/data/Pt THH ex-situ/Data/CH4760/"
sample_name = "S"
filtered_data = False  # set to True if the data is already a 3D array, False otherwise
# Should be the same shape as in specfile
peak_method = 'maxcom'  # Bragg peak determination: 'max', 'com' or 'maxcom'.
normalize_flux = 'monitor'  # 'monitor' to normalize the intensity by the default monitor values, 'skip' to do nothing
debug = False  # True to see more plots
######################################
# define beamline related parameters #
######################################
beamline = 'ID01'  # name of the beamline, used for data loading and normalization by monitor
# supported beamlines: 'ID01', 'SIXS_2018', 'SIXS_2019', 'CRISTAL', 'P10'
is_series = False  # specific to series measurement at P10

custom_scan = False  # True for a stack of images acquired without scan, e.g. with ct in a macro (no info in spec file)
custom_images = np.arange(11353, 11453, 1)  # list of image numbers for the custom_scan
custom_monitor = np.ones(len(custom_images))  # monitor values for normalization for the custom_scan
custom_motors = {"eta": np.linspace(16.989, 18.989, num=100, endpoint=False), "phi": 0, "nu": -0.75, "delta": 36.65}
# ID01: eta, phi, nu, delta
# CRISTAL: mgomega, gamma, delta
# P10: om, phi, chi, mu, gamma, delta
# SIXS: beta, mu, gamma, delta

rocking_angle = "outofplane"  # "outofplane" or "inplane"
specfile_name = 'l5'
# template for ID01: name of the spec file without '.spec'
# template for SIXS_2018: full path of the alias dictionnary 'alias_dict.txt', typically: root_folder + 'alias_dict.txt'
# template for all other beamlines: ''
#############################################################
# define detector related parameters and region of interest #
#############################################################
detector = "Maxipix"    # "Eiger2M" or "Maxipix" or "Eiger4M"
x_bragg = None  # horizontal pixel number of the Bragg peak, can be used for the definition of the ROI
y_bragg = None  # vertical pixel number of the Bragg peak, can be used for the definition of the ROI
roi_detector = None  # [y_bragg-290, y_bragg+290, x_bragg-290, x_bragg+290]
# [y_bragg - 290, y_bragg + 350, x_bragg - 350, x_bragg + 350]  # Ar  # HC3207  x_bragg = 430
# leave it as None to use the full detector. Use with center_fft='do_nothing' if you want this exact size.
high_threshold = 1000000  # everything above will be considered as hotpixel
hotpixels_file = None  # root_folder + 'hotpixels_HS4670.npz'  # non empty file path or None
flatfield_file = root_folder + "flatfield_maxipix_8kev.npz"  # non empty file path or None
template_imagefile = 'data_mpx4_%05d.edf.gz'
# template for ID01: 'data_mpx4_%05d.edf.gz' or 'align_eiger2M_%05d.edf.gz'
# template for SIXS_2018: 'align.spec_ascan_mu_%05d.nxs'
# template for SIXS_2019: 'spare_ascan_mu_%05d.nxs'
# template for Cristal: 'S%d.nxs'
# template for P10: '_master.h5'
# template for NANOMAX: '%06d.h5'
# template for 34ID: 'Sample%dC_ES_data_51_256_256.npz'
###################################
# define setup related parameters #
###################################
beam_direction = (1, 0, 0)  # beam along z
sample_offsets = (0, 0, 0)  # tuple of offsets in degrees of the sample around (downstream, vertical up, outboard)
# convention: the sample offsets will be subtracted to the motor values
directbeam_x = 154  # x horizontal,  cch2 in xrayutilities
directbeam_y = 208  # y vertical,  cch1 in xrayutilities
direct_inplane = 0.0  # outer angle in xrayutilities
direct_outofplane = 0.0
sdd = 0.50678  # sample to detector distance in m
energy = 9000  # in eV, offset of 6eV at ID01
################################################
# parameters related to temperature estimation #
################################################
get_temperature = False  # True to estimate the temperature using the reference spacing of the material. Only for Pt.
reflection = np.array([1, 1, 1])  # measured reflection, use for estimating the temperature
reference_spacing = None  # for calibrating the thermal expansion, if None it is fixed to Pt 3.9236/norm(reflection)
reference_temperature = None  # used to calibrate the thermal expansion, if None it is fixed to 293.15K (RT)
##########################################################
# end of user parameters
##########################################################

plt.ion()
#######################
# Initialize detector #
#######################
detector = exp.Detector(name=detector, template_imagefile=template_imagefile, roi=roi_detector,
                        is_series=is_series)

####################
# Initialize setup #
####################
setup = exp.Setup(beamline=beamline, energy=energy, rocking_angle=rocking_angle, distance=sdd,
                  beam_direction=beam_direction, custom_scan=custom_scan, custom_images=custom_images,
                  custom_monitor=custom_monitor, custom_motors=custom_motors, pixel_x=detector.pixelsize_x,
                  pixel_y=detector.pixelsize_y, sample_offsets=sample_offsets)

########################################
# Initialize the paths and the logfile #
########################################
# initialize the paths
setup.init_paths(detector=detector, sample_name=sample_name, scan_number=scan, root_folder=root_folder, save_dir=None,
                 create_savedir=False, specfile_name=specfile_name, template_imagefile=template_imagefile, verbose=True)

logfile = pru.create_logfile(setup=setup, detector=detector, scan_number=scan, root_folder=root_folder,
                             filename=detector.specfile)

#################
# load the data #
#################
flatfield = pru.load_flatfield(flatfield_file)
hotpix_array = pru.load_hotpixels(hotpixels_file)

if not filtered_data:
    data, _, monitor, frames_logical = pru.load_data(logfile=logfile, scan_number=scan, detector=detector,
                                                     setup=setup, flatfield=flatfield, hotpixels=hotpix_array,
                                                     normalize=normalize_flux, debugging=debug)
    if normalize_flux == 'skip':
        print('Skip intensity normalization')
    else:
        print('Intensity normalization using ' + normalize_flux)
        data, monitor = pru.normalize_dataset(array=data, raw_monitor=monitor, frames_logical=frames_logical,
                                              norm_to_min=True, debugging=debug)
else:
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(initialdir=detector.scandir + "pynxraw/",
                                           title="Select 3D data", filetypes=[("NPZ", "*.npz")])
    data = np.load(file_path)['data']
    data = data[detector.roi[0]:detector.roi[1], detector.roi[2]:detector.roi[3]]
    frames_logical = np.ones(data.shape[0]).astype(int)  # use all frames from the filtered data
numz, numy, numx = data.shape
print("Shape of dataset: ", numz, numy, numx)

##############################################
# apply photon threshold to remove hotpixels #
##############################################
if high_threshold != 0:
    nb_thresholded = (data > high_threshold).sum()
    data[data > high_threshold] = 0
    print(f'Applying photon threshold, {nb_thresholded} high intensity pixels masked')

###############################
# load releavant motor values #
###############################
tilt_values, setup.grazing_angle, setup.inplane_angle, setup.outofplane_angle = \
    pru.goniometer_values(logfile=logfile, scan_number=scan, setup=setup, frames_logical=frames_logical)
setup.tilt_angle = tilt_values[1] - tilt_values[0]

nb_frames = len(tilt_values)
if numz != nb_frames:
    print('The loaded data has not the same shape as the raw data')
    sys.exit()

#######################
# Find the Bragg peak #
#######################
z0, y0, x0 = pru.find_bragg(data, peak_method=peak_method)
z0 = np.rint(z0).astype(int)
y0 = np.rint(y0).astype(int)
x0 = np.rint(x0).astype(int)

print(f'Bragg peak at (z, y, x): {z0}, {y0}, {x0}')
print(f'Bragg peak (full detector) at (z, y, x): {z0}, {y0+detector.roi[0]}, {x0+detector.roi[2]}')

######################################################
# calculate rocking curve and fit it to get the FWHM #
######################################################
rocking_curve = np.zeros(nb_frames)
if filtered_data == 0:  # take a small ROI to avoid parasitic peaks
    for idx in range(nb_frames):
        rocking_curve[idx] = data[idx, y0 - 20:y0 + 20, x0 - 20:x0 + 20].sum()
    plot_title = "Rocking curve for a 40x40 pixels ROI"
else:  # take the whole detector
    for idx in range(nb_frames):
        rocking_curve[idx] = data[idx, :, :].sum()
    plot_title = "Rocking curve (full detector)"
z0 = np.unravel_index(rocking_curve.argmax(), rocking_curve.shape)[0]

interpolation = interp1d(tilt_values, rocking_curve, kind='cubic')
interp_points = 5*nb_frames
interp_tilt = np.linspace(tilt_values.min(), tilt_values.max(), interp_points)
interp_curve = interpolation(interp_tilt)
interp_fwhm = len(np.argwhere(interp_curve >= interp_curve.max()/2)) * \
              (tilt_values.max()-tilt_values.min())/(interp_points-1)
print('FWHM by interpolation', str('{:.3f}'.format(interp_fwhm)), 'deg')

fig, (ax0, ax1) = plt.subplots(2, 1, sharex='col', figsize=(10, 5))
ax0.plot(tilt_values, rocking_curve, '.')
ax0.plot(interp_tilt, interp_curve)
ax0.set_ylabel('Integrated intensity')
ax0.legend(('data', 'interpolation'))
ax0.set_title(plot_title)
ax1.plot(tilt_values, np.log10(rocking_curve), '.')
ax1.plot(interp_tilt, np.log10(interp_curve))
ax1.set_xlabel('Rocking angle (deg)')
ax1.set_ylabel('Log(integrated intensity)')
ax0.legend(('data', 'interpolation'))
plt.pause(0.1)

##############################
# Calculate corrected angles #
##############################
bragg_x = detector.roi[2] + x0  # convert it in full detector pixel
bragg_y = detector.roi[0] + y0  # convert it in full detector pixel

x_direct_0 = directbeam_x + setup.inplane_coeff *\
             (direct_inplane*np.pi/180*sdd/detector.pixelsize_x)  # inplane_coeff is +1 or -1
y_direct_0 = directbeam_y - setup.outofplane_coeff *\
             direct_outofplane*np.pi/180*sdd/detector.pixelsize_y   # outofplane_coeff is +1 or -1

print(f'\nDirect beam at (gam={direct_inplane}, del={direct_outofplane}) (X, Y): {directbeam_x}, {directbeam_y}')
print(f'Direct beam at (gam=0, del=0) (X, Y): ({x_direct_0:.2f}, {y_direct_0:.2f})')
print(f'\nBragg peak at (gam={setup.inplane_angle}, del={setup.outofplane_angle}) (X, Y): ({bragg_x:.2f}, {bragg_y:.2f})')

bragg_inplane = setup.inplane_angle + setup.inplane_coeff *\
                (detector.pixelsize_x*(bragg_x-x_direct_0)/sdd*180/np.pi)  # inplane_coeff is +1 or -1
bragg_outofplane = setup.outofplane_angle - setup.outofplane_coeff *\
                   detector.pixelsize_y*(bragg_y-y_direct_0)/sdd*180/np.pi   # outofplane_coeff is +1 or -1

print(f'\nBragg angles before correction (gam, del): ({setup.inplane_angle:.4f}, {setup.outofplane_angle:.4f})')
print(f'Bragg angles after correction (gam, del): ({bragg_inplane:.4f}, {bragg_outofplane:.4f})')

# update setup with the corrected detector angles
setup.inplane_angle = bragg_inplane
setup.outofplane_angle = bragg_outofplane

print(f'\nGrazing angle(s) = {setup.grazing_angle} deg')
print(f'Rocking step = {setup.tilt_angle:.5f} deg')

####################################
# wavevector transfer calculations #
####################################
kin = 2*np.pi/setup.wavelength * np.asarray(beam_direction)  # in lab frame z downstream, y vertical, x outboard
kout = setup.exit_wavevector  # in lab.frame z downstream, y vertical, x outboard
q = (kout - kin) / 1e10  # convert from 1/m to 1/angstrom
qnorm = np.linalg.norm(q)
dist_plane = 2 * np.pi / qnorm
print(f'\nWavevector transfer of Bragg peak: {q}, Qnorm={qnorm:.4f}')
print(f'Interplanar distance: {dist_plane:.6f} angstroms')

if get_temperature:
    print('\nEstimating the temperature:')
    temperature = pu.bragg_temperature(spacing=dist_plane, reflection=reflection, spacing_ref=reference_spacing,
                                       temperature_ref=reference_temperature, use_q=False, material="Pt")

#########################
# calculate voxel sizes #
#########################
#  update the detector angles in setup
setup.inplane_angle = bragg_inplane
setup.outofplane_angle = bragg_outofplane
dz_realspace, dy_realspace, dx_realspace = setup.voxel_sizes((nb_frames, numy, numx), tilt_angle=setup.tilt_angle,
                                                             pixel_x=detector.pixelsize_x, pixel_y=detector.pixelsize_y,
                                                             verbose=True)

#################################
# plot image at Bragg condition #
#################################
plt.figure()
plt.imshow(np.log10(abs(data[int(round(z0)), :, :])), vmin=0, vmax=5)
plt.title(f'Central slice at frame {int(np.rint(z0))}')
plt.colorbar()
plt.ioff()
plt.show()
