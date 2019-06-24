# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#       authors:
#         Jerome Carnis, jerome.carnis@esrf.fr

import hdf5plugin  # for P10, should be imported before h5py or PyTables
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import tkinter as tk
from tkinter import filedialog
import sys
sys.path.append('C:\\Users\\carnis\\Work Folders\\Documents\\myscripts\\bcdi\\')
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
scan = 356
root_folder = "C:/Users/carnis/Work Folders/Documents/data/P10_2018/"
sample_name = "dewet5"
filtered_data = False  # set to True if the data is already a 3D array, False otherwise
# Should be the same shape as in specfile
peak_method = 'maxcom'  # Bragg peak determination: 'max', 'com' or 'maxcom'.
######################################
# define beamline related parameters #
######################################
beamline = 'P10'  # name of the beamline, used for data loading and normalization by monitor
# supported beamlines: 'ID01', 'SIXS_2018', 'SIXS_2019', 'CRISTAL', 'P10'
rocking_angle = "outofplane"  # "outofplane" or "inplane"
specfile_name = sample_name + '_%05d'
# .spec for ID01, .fio for P10, alias_dict.txt for SIXS, not used for CRISTAL
# template for ID01: name of the spec file without '.spec'
# template for SIXS: full path of the alias dictionnary 'alias_dict.txt', typically: root_folder + 'alias_dict.txt'
# template for P10: sample_name + '_%05d'
# template for CRISTAL: ''
#############################################################
# define detector related parameters and region of interest #
#############################################################
detector = "Eiger4M"    # "Eiger2M" or "Maxipix" or "Eiger4M"
x_bragg = 1393  # horizontal pixel number of the Bragg peak
# roi_detector = [1202, 1610, x_bragg - 256, x_bragg + 256]  # HC3207  x_bragg = 430
roi_detector = []
# leave it as [] to use the full detector. Use with center_fft='do_nothing' if you want this exact size.
photon_threshold = 0  # data[data <= photon_threshold] = 0
hotpixels_file = ''  # root_folder + 'hotpixels.npz'  #
flatfield_file = ''  # root_folder + "flatfield_8.5kev.npz"  #
template_imagefile = '_data_%06d.h5'
# template for ID01: 'data_mpx4_%05d.edf.gz' or 'align_eiger2M_%05d.edf.gz'
# template for SIXS_2018: 'align.spec_ascan_mu_%05d.nxs'
# template for SIXS_2019: 'spare_ascan_mu_%05d.nxs'
# template for Cristal: 'S%d.nxs'
# template for P10: '_data_%06d.h5'
###################################
# define setup related parameters #
###################################
reflection = np.array([1, 1, 1])  # measured reflection, use for estimating the temperature
reference_spacing = None  # for calibrating the thermal expansion, if None it is fixed to Pt 3.9236/norm(reflection)
reference_temperature = None  # used to calibrate the thermal expansion, if None it is fixed to 293.15K (RT)
beam_direction = (1, 0, 0)  # beam along z
directbeam_x = 476  # x horizontal,  cch2 in xrayutilities
directbeam_y = 1375  # y vertical,  cch1 in xrayutilities
direct_inplane = -2.0  # outer angle in xrayutilities
direct_outofplane = 0.8
sdd = 1.839  # sample to detector distance in m
energy = 10300  # in eV, offset of 6eV at ID01
##########################################################
# end of user parameters
##########################################################

plt.ion()
#######################
# Initialize detector #
#######################
detector = exp.Detector(name=detector, datadir='', template_imagefile=template_imagefile, roi=roi_detector)

####################
# Initialize setup #
####################
setup_pre = exp.SetupPreprocessing(beamline=beamline, rocking_angle=rocking_angle, distance=sdd, energy=energy,
                                   beam_direction=beam_direction)

if setup_pre.beamline != 'P10':
    homedir = root_folder + sample_name + str(scan) + '/'
    detector.datadir = homedir + "data/"
else:
    specfile_name = specfile_name % scan
    homedir = root_folder + specfile_name + '/'
    detector.datadir = homedir + 'e4m/'
    template_imagefile = specfile_name + template_imagefile
    detector.template_imagefile = template_imagefile

print('\nScan', scan)
print('Setup: ', setup_pre.beamline)
print('Detector: ', detector.name)
print('Pixel Size: ', detector.pixelsize, 'm')
print('Scan type: ', setup_pre.rocking_angle)
print('Sample to detector distance: ', setup_pre.distance, 'm')
print('Energy:', setup_pre.energy, 'ev')
##############
# load files #
##############
flatfield = pru.load_flatfield(flatfield_file)
hotpix_array = pru.load_hotpixels(hotpixels_file)

logfile = pru.create_logfile(beamline=setup_pre.beamline, detector=detector, scan_number=scan,
                             root_folder=root_folder, filename=specfile_name)

if filtered_data == 0:
    _, data, _, mask, _, frames_logical, monitor = \
        pru.gridmap(logfile=logfile, scan_number=scan, detector=detector, setup=setup_pre,
                    flatfield=flatfield, hotpixels=hotpix_array, hxrd=None, follow_bragg=False,
                    debugging=False, orthogonalize=False)
else:
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(initialdir=homedir + "pynxraw/",
                                           title="Select 3D data", filetypes=[("NPZ", "*.npz")])
    data = np.load(file_path)['data']
    data = data[detector.roi[0]:detector.roi[1], detector.roi[2]:detector.roi[3]]
    frames_logical = np.ones(data.shape[0])  # use all frames from the filtered data
numz, numy, numx = data.shape
print("Shape of dataset: ", numz, numy, numx)

##############################
# find motors values in .fio #
##############################
tilt, grazing, inplane, outofplane = pru.motor_values(frames_logical, logfile, scan, setup_pre, follow_bragg=False)

setup_post = exp.SetupPostprocessing(beamline=setup_pre.beamline, energy=setup_pre.energy,
                                     outofplane_angle=outofplane, inplane_angle=inplane, tilt_angle=tilt,
                                     rocking_angle=setup_pre.rocking_angle, grazing_angle=grazing,
                                     distance=setup_pre.distance, pixel_x=detector.pixelsize,
                                     pixel_y=detector.pixelsize)

nb_frames = len(tilt)
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

print("Bragg peak at (z, y, x): ", z0, y0, x0)
print("Bragg peak (full detector) at (z, y, x): ", z0, y0+detector.roi[0], x0+detector.roi[2])

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


interpolation = interp1d(tilt, rocking_curve, kind='cubic')
interp_points = 5*nb_frames
interp_tilt = np.linspace(tilt.min(), tilt.max(), interp_points)
interp_curve = interpolation(interp_tilt)
interp_fwhm = len(np.argwhere(interp_curve >= interp_curve.max()/2)) * \
              (tilt.max()-tilt.min())/(interp_points-1)
print('FWHM by interpolation', str('{:.3f}'.format(interp_fwhm)), 'deg')

fig, (ax0, ax1) = plt.subplots(2, 1, sharex='col', figsize=(10, 5))
ax0.plot(tilt, rocking_curve, '.')
ax0.plot(interp_tilt, interp_curve)
ax0.set_ylabel('Integrated intensity')
ax0.legend(('data', 'interpolation'))
ax0.set_title(plot_title)
ax1.plot(tilt, np.log10(rocking_curve), '.')
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

x_direct_0 = directbeam_x + setup_post.rotation_direction() *\
             (direct_inplane*np.pi/180*sdd/detector.pixelsize)  # rotation_direction is +1 or -1
y_direct_0 = directbeam_y - direct_outofplane*np.pi/180*sdd/detector.pixelsize   # outofplane is always clockwise

print("\nDirect beam at (gam=", str(direct_inplane), "del=", str(direct_outofplane),
      ") = (X, Y): ", directbeam_x, directbeam_y)
print("Direct beam at (gam= 0, del= 0) = (X, Y): ", str('{:.2f}'.format(x_direct_0)), str('{:.2f}'.format(y_direct_0)))
print("Bragg peak at (gam=", str(inplane), "del=", str(outofplane), ") = (X, Y): ",
      str('{:.2f}'.format(bragg_x)), str('{:.2f}'.format(bragg_y)))

bragg_inplane = inplane + setup_post.rotation_direction() *\
                (detector.pixelsize*(bragg_x-x_direct_0)/sdd*180/np.pi)  # rotation_direction is +1 or -1
bragg_outofplane = outofplane - detector.pixelsize*(bragg_y-y_direct_0)/sdd*180/np.pi   # outofplane is always clockwise

print("\nBragg angles before correction = (gam, del): ", str('{:.4f}'.format(inplane)),
      str('{:.4f}'.format(outofplane)))
print("\nBragg angles after correction = (gam, del): ", str('{:.4f}'.format(bragg_inplane)),
      str('{:.4f}'.format(bragg_outofplane)))

d_rocking_angle = tilt[1] - tilt[0]

print("\nGrazing angle=", str('{:.4f}'.format(grazing)), 'deg')

print("\nRocking step=", str('{:.4f}'.format(d_rocking_angle)), 'deg')

####################################
# wavevector transfer calculations #
####################################
kin = 2*np.pi/setup_post.wavelength * np.asarray(beam_direction)  # in lab frame z downstream, y vertical, x outboard
kout = setup_post.exit_wavevector()  # in lab.frame z downstream, y vertical, x outboard
q = (kout - kin) / 1e10  # convert from 1/m to 1/angstrom
Qnorm = np.linalg.norm(q)
dist_plane = 2 * np.pi / Qnorm
print("\nWavevector transfer of Bragg peak: ", q, str('{:.4f}'.format(Qnorm)))
print("Interplanar distance: ", str('{:.4f}'.format(dist_plane)), "angstroms")
temperature = pu.bragg_temperature(spacing=dist_plane, reflection=reflection, spacing_ref=reference_spacing,
                                   temperature_ref=reference_temperature, use_q=0, material="Pt")

#########################
# calculate voxel sizes #
#########################
dz_realspace = setup_post.wavelength * 1e9 / (nb_frames * d_rocking_angle * np.pi / 180)  # in nm
dy_realspace = setup_post.wavelength * 1e9 * sdd / (numy * detector.pixelsize)  # in nm
dx_realspace = setup_post.wavelength * 1e9 * sdd / (numx * detector.pixelsize)  # in nm
print('Real space voxel size (z, y, x): ', str('{:.2f}'.format(dz_realspace)), 'nm',
      str('{:.2f}'.format(dy_realspace)), 'nm', str('{:.2f}'.format(dx_realspace)), 'nm')

#################################
# plot image at Bragg condition #
#################################
plt.figure()
plt.imshow(np.log10(abs(data[int(round(z0)), :, :])), vmin=0, vmax=5)
plt.title('Central slice at frame '+str(int(np.rint(z0))))
plt.colorbar()
plt.ioff()
plt.show()
