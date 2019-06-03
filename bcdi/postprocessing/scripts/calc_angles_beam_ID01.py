# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#       authors:
#         Jerome Carnis, jerome.carnis@esrf.fr

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
Calculate delta and gamma angles from the direct beam and Bragg peak positions, based on ESRF ID01 geometry.

Input: direct beam and Bragg peak position, sample to detector distance, energy
Output: corrected gamma, delta of Bragg peak
"""
scan = 107
root_folder = "C:\\Users\\carnis\\Work Folders\\Documents\\data\\CH4760_Pt\\"
sample_name = "S"
filtered_data = False  # set to True if the data is already a 3D array, False otherwise
# Should be the same shape as in specfile
peak_method = 'maxcom'  # Bragg peak determination: 'max', 'com' or 'maxcom'.
######################################
# define beamline related parameters #
######################################
beamline = 'ID01'  # 'ID01' or 'SIXS' or 'CRISTAL' or 'P10', used for data loading and normalization by monitor
rocking_angle = "outofplane"  # "outofplane", "inplane"
specfile_name = 'align'  # name of the spec file without '.spec'
#############################################################
# define detector related parameters and region of interest #
#############################################################
detector = "Maxipix"    # "Eiger2M" or "Maxipix" or "Eiger4M"
x_bragg = 1409  # horizontal pixel number of the Bragg peak
# roi_detector = [1202, 1610, x_bragg - 256, x_bragg + 256]  # HC3207  x_bragg = 430
roi_detector = []
# leave it as [] to use the full detector. Use with center_fft='do_nothing' if you want this exact size.
photon_threshold = 0  # data[data <= photon_threshold] = 0
hotpixels_file = ''  # root_folder + 'hotpixels.npz'  #
flatfield_file = ''  # root_folder + "flatfield_eiger.npz"  #
template_imagefile = 'data_mpx4_%05d.edf.gz'  # 'align_eiger2M_%05d.edf.gz'
###################################
# define setup related parameters #
###################################
reflection = np.array([1, 1, 1])  # measured reflection, use for estimating the temperature
reference_spacing = None  # for calibrating the thermal expansion, if None it is fixed to Pt 3.9236/norm(reflection)
reference_temperature = None  # used to calibrate the thermal expansion, if None it is fixed to 293.15K (RT)
directbeam_x = 50.40  # x horizontal,  cch2 in xrayutilities
directbeam_y = 451.02  # y vertical,  cch1 in xrayutilities
direct_inplane = -0.124  # outer angle in xrayutilities
direct_outofplane = -0.052
sdd = 0.9207  # sample to detector distance in m
energy = 7994  # in eV, offset of 6eV at ID01
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
setup = exp.SetupPreprocessing(beamline=beamline, energy=energy, rocking_angle=rocking_angle, distance=sdd)

homedir = root_folder + sample_name + str(scan) + '/'
detector.datadir = homedir + "data/"

print('\nScan', scan)
print('Setup: ', setup.beamline)
print('Detector: ', detector.name)
print('Pixel Size: ', detector.pixelsize)
print('Scan type: ', setup.rocking_angle)

##############
# load files #
##############
flatfield = pru.load_flatfield(flatfield_file)
hotpix_array = pru.load_hotpixels(hotpixels_file)

logfile = pru.create_logfile(setup=setup, detector=detector, scan_number=scan,
                             root_folder=root_folder, filename='')

if filtered_data == 0:
    _, data, _, mask, _, frames_logical, monitor = \
        pru.gridmap(logfile=logfile, scan_number=scan, detector=detector, setup=setup,
                    flatfield=flatfield, hotpixels=hotpix_array, hxrd=None, follow_bragg=False,
                    debugging=False, orthogonalize=False)
else:
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(initialdir=homedir + "pynxraw/",
                                           title="Select 3D data", filetypes=[("NPZ", "*.npz")])
    data = np.load(file_path)['data']
    data = data[roi_detector[0]:roi_detector[1], roi_detector[2]:roi_detector[3]]
    frames_logical = np.ones(data.shape[0])  # use all frames from the filtered data
nz, ny, nx = data.shape
print("Shape of dataset: ", nz, ny, nx)

##############################
# find motors values in .fio #
##############################
if rocking_angle == 'outofplane':  # eta rocking curve
    tilt, _, _, inplane, outofplane, _, _ = \
        pru.motor_positions_id01(frames_logical, logfile, scan, setup, follow_bragg=False)
elif rocking_angle == 'inplane':  # phi rocking curve
    _, _, tilt, inplane, outofplane, _, _ = \
        pru.motor_positions_id01(frames_logical, logfile, scan, setup, follow_bragg=False)
else:
    print('Wrong value for "rocking_angle" parameter')
    sys.exit()

nb_frames = len(tilt)
if nz != nb_frames:
    print('The loaded data has not the same shape as the raw data')
    sys.exit()

#######################
# Find the Bragg peak #
#######################
z0, y0, x0 = pru.find_bragg(data, peak_method=peak_method)

print("Bragg peak at (z, y, x): ", np.rint(z0).astype(int), np.rint(y0).astype(int), np.rint(x0).astype(int))
print("Bragg peak (full detector) at (z, y, x): ", np.rint(z0).astype(int), np.rint(y0+roi_detector[0]).astype(int),
      np.rint(x0+roi_detector[2]).astype(int))


######################################################
# calculate rocking curve and fit it to get the FWHM #
######################################################
rocking_curve = np.zeros(nz)
if filtered_data == 0:  # take a small ROI to avoid parasitic peaks
    for idx in range(nz):
        rocking_curve[idx] = data[idx, y0 - 20:y0 + 20, x0 - 20:x0 + 20].sum()
    plot_title = "Rocking curve for a 40x40 pixels ROI"
else:  # take the whole detector
    for idx in range(nz):
        rocking_curve[idx] = data[idx, :, :].sum()
    plot_title = "Rocking curve (full detector"
z0 = np.unravel_index(rocking_curve.argmax(), rocking_curve.shape)[0]


interpolation = interp1d(tilt, rocking_curve, kind='cubic')
interp_points = 5*nz
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
bragg_x = roi_detector[2] + x0  # convert it in full detector pixel
bragg_y = roi_detector[0] + y0  # convert it in full detector pixel

x_direct_0 = directbeam_x - direct_inplane*np.pi/180*sdd/detector.pixel_size  # nu is clockwise
y_direct_0 = directbeam_y - direct_outofplane*np.pi/180*sdd/detector.pixel_size   # delta is clockwise

print("\nDirect beam at (gam=", str(direct_inplane), "del=", str(direct_outofplane),
      ") = (X, Y): ", directbeam_x, directbeam_y)
print("Direct beam at (gam= 0, del= 0) = (X, Y): ", str('{:.2f}'.format(x_direct_0)), str('{:.2f}'.format(y_direct_0)))
print("Bragg peak at (gam=", str(inplane), "del=", str(outofplane), ") = (X, Y): ",
      str('{:.2f}'.format(bragg_x)), str('{:.2f}'.format(bragg_y)))

bragg_inplane = inplane - detector.pixel_size*(bragg_x-x_direct_0)/sdd*180/np.pi  # nu is clockwise
bragg_outofplane = outofplane - detector.pixel_size*(bragg_y-y_direct_0)/sdd*180/np.pi

print("\nBragg angles before correction = (gam, del): ", str('{:.4f}'.format(inplane)),
      str('{:.4f}'.format(outofplane)))
print("Bragg angles after correction = (gam, del): ", str('{:.4f}'.format(bragg_inplane)),
      str('{:.4f}'.format(bragg_outofplane)))

d_rocking_angle = rocking_angle[1] - rocking_angle[0]

print("\nRocking step=", str('{:.4f}'.format(d_rocking_angle)), 'deg')

####################################
# wavevector transfer calculations #
####################################
wavelength = 12.398*1000/energy  # in angstroms
q1 = (np.cos(np.radians(bragg_inplane))*np.cos(np.radians(bragg_outofplane)))-1  # z downstream
q2 = np.sin(np.radians(bragg_outofplane))  # y vertical
q3 = -1*np.sin(np.radians(bragg_inplane))*np.cos(np.radians(bragg_outofplane))  # x outboard
q = 2*np.pi/wavelength*np.array([q1, q2, q3])
Qnorm = np.linalg.norm(q)
dist_plane = 2 * np.pi / Qnorm
print("\nWavevector transfer of Bragg peak: ", q, str('{:.4f}'.format(Qnorm)))
print("Interplanar distance: ", str('{:.4f}'.format(dist_plane)), "angstroms")
temperature = pu.bragg_temperature(spacing=dist_plane, reflection=reflection, spacing_ref=reference_spacing,
                                   temperature_ref=reference_temperature, use_q=0, material="Pt")

#########################
# calculate voxel sizes #
#########################
dz_realspace = wavelength / 10 / (nz * d_rocking_angle * np.pi / 180)  # in nm
dy_realspace = wavelength / 10 * sdd / (ny * detector.pixel_size)  # in nm
dx_realspace = wavelength / 10 * sdd / (nx * detector.pixel_size)  # in nm
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
