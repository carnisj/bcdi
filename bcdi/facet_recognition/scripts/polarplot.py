# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import hdf5plugin  # for P10, should be imported before h5py or PyTables
import pathlib
import numpy as np
import xrayutilities as xu
import scipy.signal  # for medfilt2d
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import sys
import tkinter as tk
from tkinter import filedialog
from numpy.fft import fftn, fftshift
import gc
sys.path.append('//win.desy.de/home/carnisj/My Documents/myscripts/bcdi/')
import bcdi.graph.graph_utils as gu
import bcdi.experiment.experiment_utils as exp
import bcdi.facet_recognition.facet_utils as fu
import bcdi.postprocessing.postprocessing_utils as pu
import bcdi.preprocessing.preprocessing_utils as pru


helptext = """
xrutils_polarplot.py
Stereographic projection of diffraction pattern, based on ESRF/ID01 geometry
Before interpolation lower z pixel is higher qz
After interpolation higher z pixel is higher qz
For x and y, higher pixel means higher q
In arrays, when plotting the first parameter is the row (vertical axis) and the second the column (horizontal axis)
Therefore the data structure is data[qx, qz, qy]
Hence the gridder is mygridder(myqx, myqz, myqy, rawdata)
And qx, qz, qy = mygridder.xaxis, mygridder.yaxis, mygridder.zaxis
"""

scan = 1    # spec scan number
root_folder = "D:/data/PtRh/"
sample_name = "S"  # "S"  #
comment = ""
reflection = np.array([1, 1, 1])  # np.array([0, 0, 2])  #   # reflection measured
filtered_data = True  # set to True if the data is already a 3D array, False otherwise
is_orthogonal = False  # True is the filtered_data is already orthogonalized, q values need to be provided
# Should be the same shape as in specfile, before orthogonalization
radius_mean = 0.04  # q from Bragg peak
dr = 0.002        # delta_q
offset_eta = 0  # positive make diff pattern rotate counter-clockwise (eta rotation around Qy)
# will shift peaks rightwards in the pole figure
offset_phi = 0     # positive make diff pattern rotate clockwise (phi rotation around Qz)
# will rotate peaks counterclockwise in the pole figure
offset_chi = 0  # positive make diff pattern rotate clockwise (chi rotation around Qx)
# will shift peaks upwards in the pole figure
range_min = 0  # low limit for the colorbar in polar plots, every below will be set to nan
range_max = 4800  # high limit for the colorbar in polar plots
range_step = 1000  # step for color change in polar plots
###################################################################################################
# parameters for plotting the stereographic projection starting from the phased real space object #
###################################################################################################
reconstructed_data = False  # set it to True if the data is a BCDI reconstruction (real space)
# the reconstruction should be in the crystal orthogonal frame
threshold_amp = 0.36  # threshold for support determination from amplitude, if reconstructed_data=1
use_phase = False  # set to False to use only a support, True to use the compex amplitude
voxel_size = 5  # in nm, voxel size of the CDI reconstruction, should be equal in all directions.  Put 0 if unknown
photon_nb = 5e7  # total number of photons in the diffraction pattern calculated from CDI reconstruction
pad_size = 3  # int >= 1, will pad to get this number times the initial array size  (avoid aliasing)
###################
# various options #
###################
flag_medianfilter = False  # set to True for applying med2filter [3,3]
flag_plotplanes = True  # if True, plot red dotted circle with plane index
flag_plottext = False  # if True, will plot plane indices and angles in the figure
photon_threshold = 1  # photon threshold in detector counts
normalize_flux = True  # will normalize the intensity by the default monitor.
debug = False  # True to show more plots, False otherwise
qz_offset = 0  # offset of the projection plane in the vertical direction (0 = equatorial plane)
######################################
# define beamline related parameters #
######################################
beamline = 'ID01'  # name of the beamline, used for data loading and normalization by monitor
# supported beamlines: 'ID01', 'SIXS_2018', 'SIXS_2019', 'CRISTAL', 'P10'

custom_scan = True  # True for a stack of images acquired without scan, e.g. with ct in a macro (no info in spec file)
custom_images = np.arange(11353, 11453, 1)  # list of image numbers for the custom_scan
custom_monitor = np.ones(len(custom_images))  # monitor values for normalization for the custom_scan
custom_motors = {"eta": np.linspace(16.989, 18.989, num=100, endpoint=False), "phi": 0, "nu": -0.75, "delta": 35.978}
# ID01: eta, phi, nu, delta
# CRISTAL: mgomega, gamma, delta
# P10: om, phi, chi, mu, gamma, delta
# SIXS: beta, mu, gamma, delta

rocking_angle = "outofplane"  # "outofplane" or "inplane" or "energy"
follow_bragg = False  # only for energy scans, set to True if the detector was also scanned to follow the Bragg peak
specfile_name = ''
# .spec for ID01, .fio for P10, alias_dict.txt for SIXS_2018, not used for CRISTAL and SIXS_2019
# template for ID01: name of the spec file without '.spec'
# template for SIXS_2018: full path of the alias dictionnary, typically root_folder + 'alias_dict_2019.txt'
# template for SIXS_2019: ''
# template for P10: sample_name + '_%05d'
# template for CRISTAL: ''
#############################################################
# define detector related parameters and region of interest #
#############################################################
detector = "Eiger2M"    # "Eiger2M" or "Maxipix" or "Eiger4M"
x_bragg = 451  # horizontal pixel number of the Bragg peak
y_bragg = 1450  # vertical pixel number of the Bragg peak
# roi_detector = [1202, 1610, x_bragg - 256, x_bragg + 256]  # HC3207  x_bragg = 430
roi_detector = [y_bragg - 290, y_bragg + 350, x_bragg - 350, x_bragg + 350]  # Ar
# roi_detector = [552, 1064, x_bragg - 240, x_bragg + 240]  # P10 2018
# leave it as [] to use the full detector. Use with center_fft='do_nothing' if you want this exact size.
photon_threshold = 0  # data[data <= photon_threshold] = 0
hotpixels_file = ''  # root_folder + 'hotpixels.npz'  #
flatfield_file = root_folder + "flatfield_eiger.npz"  #
template_imagefile = 'BCDI_eiger2M_%05d.edf'
# template for ID01: 'data_mpx4_%05d.edf.gz' or 'align_eiger2M_%05d.edf.gz'
# template for SIXS_2018: 'align.spec_ascan_mu_%05d.nxs'
# template for SIXS_2019: 'spare_ascan_mu_%05d.nxs'
# template for Cristal: 'S%d.nxs'
# template for P10: '_data_%06d.h5'
###################################################################
# define parameters for xrayutilities, used for orthogonalization #
###################################################################
# xrayutilities uses the xyz crystal frame: for incident angle = 0, x is downstream, y outboard, and z vertical up
sdd = 0.865  # sample to detector distance in m, not important if you use raw data
energy = 9000  # x-ray energy in eV, not important if you use raw data
beam_direction = (1, 0, 0)  # beam along z
sample_inplane = (1, 0, 0)  # sample inplane reference direction along the beam at 0 angles
sample_outofplane = (0, 0, 1)  # surface normal of the sample at 0 angles
offset_inplane = -0.5  # outer detector angle offset, not important if you use raw data
cch1 = 1273.5  # cch1 parameter from xrayutilities 2D detector calibration, detector roi is taken into account below
cch2 = 390.8  # cch2 parameter from xrayutilities 2D detector calibration, detector roi is taken into account below
detrot = 0  # detrot parameter from xrayutilities 2D detector calibration
tiltazimuth = 0  # tiltazimuth parameter from xrayutilities 2D detector calibration
tilt = 0  # tilt parameter from xrayutilities 2D detector calibration
##################################################################################################
# calculate theoretical angles between the measured reflection and other planes - only for cubic #
##################################################################################################
planes = dict()  # create dictionnary
planes['1 -1 1'] = fu.plane_angle_cubic(reflection, np.array([1, -1, 1]))
planes['1 0 0'] = fu.plane_angle_cubic(reflection, np.array([1, 0, 0]))
###################
# define colormap #
###################
bad_color = '1.0'  # white background
colormap = gu.Colormap(bad_color=bad_color)
my_cmap = colormap.cmap
##################################
# end of user-defined parameters #
##################################

#######################
# Initialize detector #
#######################
detector = exp.Detector(name=detector, datadir='', template_imagefile=template_imagefile, roi=roi_detector)

####################
# Initialize setup #
####################
setup = exp.SetupPreprocessing(beamline=beamline, energy=energy, rocking_angle=rocking_angle, distance=sdd,
                               beam_direction=beam_direction, sample_inplane=sample_inplane,
                               sample_outofplane=sample_outofplane, sample_offsets=(offset_chi, offset_phi, offset_eta),
                               offset_inplane=offset_inplane, custom_scan=custom_scan, custom_images=custom_images,
                               custom_monitor=custom_monitor, custom_motors=custom_motors, filtered_data=filtered_data,
                               is_orthogonal=is_orthogonal)

#############################################
# Initialize geometry for orthogonalization #
#############################################
qconv, offsets = pru.init_qconversion(setup)
detector.offsets = offsets
hxrd = xu.experiment.HXRD(sample_inplane, sample_outofplane, qconv=qconv)  # x downstream, y outboard, z vertical
# first two arguments in HXRD are the inplane reference direction along the beam and surface normal of the sample
cch1 = cch1 - detector.roi[0]  # take into account the roi if the image is cropped
cch2 = cch2 - detector.roi[2]  # take into account the roi if the image is cropped
hxrd.Ang2Q.init_area('z-', 'y+', cch1=cch1, cch2=cch2, Nch1=detector.roi[1] - detector.roi[0],
                     Nch2=detector.roi[3] - detector.roi[2], pwidth1=detector.pixelsize_y,
                     pwidth2=detector.pixelsize_x, distance=sdd, detrot=detrot, tiltazimuth=tiltazimuth, tilt=tilt)
# first two arguments in init_area are the direction of the detector, checked for ID01 and SIXS

#############
# load data #
#############
plt.ion()
root = tk.Tk()
root.withdraw()
if setup.beamline != 'P10':
    homedir = root_folder + sample_name + str(scan) + '/'
    detector.datadir = homedir + "data/"
else:
    specfile_name = specfile_name % scan
    homedir = root_folder + specfile_name + '/'
    detector.datadir = homedir + 'e4m/'
    template_imagefile = specfile_name + template_imagefile
    detector.template_imagefile = template_imagefile

detector.savedir = homedir

if not reconstructed_data:
    flatfield = pru.load_flatfield(flatfield_file)
    hotpix_array = pru.load_hotpixels(hotpixels_file)
    logfile = pru.create_logfile(setup=setup, detector=detector, scan_number=scan,
                                 root_folder=root_folder, filename=specfile_name)

    q_values, _, data, _, _, _, _ = \
        pru.gridmap(logfile=logfile, scan_number=scan, detector=detector, setup=setup,
                    flatfield=flatfield, hotpixels=hotpix_array, hxrd=hxrd, follow_bragg=follow_bragg,
                    normalize=normalize_flux, debugging=debug, orthogonalize=True)
    qx = q_values[0]  # axis=0, z downstream, qx in reciprocal space
    qz = q_values[1]  # axis=1, y vertical, qz in reciprocal space
    qy = q_values[2]  # axis=2, x outboard, qy in reciprocal space
else:
    comment = comment + "_CDI"
    file_path = filedialog.askopenfilename(initialdir=root_folder + "pynxraw/",
                                           title="Select 3D data", filetypes=[("NPZ", "*.npz")])
    amp = np.load(file_path)['amp']
    amp = amp / abs(amp).max()  # normalize amp
    nz, ny, nx = amp.shape  # nexus convention
    print('CDI data shape', amp.shape)
    nz1, ny1, nx1 = [value * pad_size for value in amp.shape]

    if use_phase:  # calculate the complex amplitude
        comment = comment + "_complex"
        phase = np.load(file_path)['phase']
        amp = amp * np.exp(1j * phase)  # amp is the complex amplitude
        del phase
        gc.collect()
    else:
        comment = comment + "_support"
        amp[amp > threshold_amp] = 1  # amp is a binary support

    # pad array to avoid aliasing
    amp = pu.crop_pad(amp, (nz1, ny1, nx1))

    # calculate the diffraction intensity
    data = fftshift(abs(fftn(amp)) ** 2)
    del amp
    gc.collect()

    voxel_size = voxel_size * 10  # conversion in angstroms
    if voxel_size <= 0:
        print('Using arbitraty voxel size of 1 nm')
        voxel_size = 10  # angstroms
    dqx = 2 * np.pi / (voxel_size * nz1)
    dqy = 2 * np.pi / (voxel_size * nx1)
    dqz = 2 * np.pi / (voxel_size * ny1)
    print('dqx', str('{:.5f}'.format(dqx)), 'dqy', str('{:.5f}'.format(dqy)), 'dqz', str('{:.5f}'.format(dqz)))

    data = data / abs(data).sum() * photon_nb  # convert into photon number
    # create qx, qy, qz vectors
    nz, ny, nx = data.shape
    qx = np.arange(-nz//2, nz//2) * dqx
    qy = np.arange(-nx//2, nx//2) * dqy
    qz = np.arange(-ny//2, ny//2) * dqz

nz, ny, nx = data.shape  # nexus convention
if flag_medianfilter:  # apply some noise filtering
    for idx in range(nz):
        data[idx, :, :] = scipy.signal.medfilt2d(data[idx, :, :], [3, 3])

###################################
# define the center of the sphere #
###################################
intensity = np.copy(data)
intensity[intensity <= photon_threshold] = 0   # photon threshold

qzCOM = 1/intensity.sum()*(qz*intensity.sum(axis=0).sum(axis=1)).sum()  # COM in qz
qyCOM = 1/intensity.sum()*(qy*intensity.sum(axis=0).sum(axis=0)).sum()  # COM in qy
qxCOM = 1/intensity.sum()*(qx*intensity.sum(axis=1).sum(axis=1)).sum()  # COM in qx
print("Center of mass [qx, qy, qz]: [",
      str('{:.2f}'.format(qxCOM)), str('{:.2f}'.format(qyCOM)), str('{:.2f}'.format(qzCOM)), ']')
del intensity
gc.collect()
##########################
# select the half sphere #
##########################
# take only the upper part of the sphere
intensity_top = data[:, np.where(qz > (qzCOM+qz_offset))[0].min():np.where(qz > (qzCOM+qz_offset))[0].max(), :]
qz_top = qz[np.where(qz > (qzCOM+qz_offset))[0].min():np.where(qz > (qzCOM+qz_offset))[0].max()]-qz_offset

# take only the lower part of the sphere
intensity_bottom = data[:, np.where(qz < (qzCOM+qz_offset))[0].min():np.where(qz < (qzCOM+qz_offset))[0].max(), :]
qz_bottom = qz[np.where(qz < (qzCOM+qz_offset))[0].min():np.where(qz < (qzCOM+qz_offset))[0].max()]-qz_offset

################################################
# create a 3D array of distances in q from COM #
################################################
qx1 = qx[:, np.newaxis, np.newaxis]  # broadcast array
qy1 = qy[np.newaxis, np.newaxis, :]  # broadcast array
qz1_top = qz_top[np.newaxis, :, np.newaxis]   # broadcast array
qz1_bottom = qz_bottom[np.newaxis, :, np.newaxis]   # broadcast array
distances_top = np.sqrt((qx1 - qxCOM)**2 + (qy1 - qyCOM)**2 + (qz1_top - (qzCOM+qz_offset))**2)
distances_bottom = np.sqrt((qx1 - qxCOM)**2 + (qy1 - qyCOM)**2 + (qz1_bottom - (qzCOM+qz_offset))**2)

######################################
# define matrix of radii radius_mean #
######################################
mask_top = np.logical_and((distances_top < (radius_mean+dr)), (distances_top > (radius_mean-dr)))
mask_bottom = np.logical_and((distances_bottom < (radius_mean+dr)), (distances_bottom > (radius_mean-dr)))

################
# plot 2D maps #
################
fig, ax = plt.subplots(figsize=(20, 15), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(2, 2, 1)
plt.contourf(qz, qx, xu.maplog(data.sum(axis=2)), 150, cmap=my_cmap)
plt.plot([min(qz), max(qz)], [qxCOM, qxCOM], color='k', linestyle='-', linewidth=2)
plt.plot([qzCOM, qzCOM], [min(qx), max(qx)], color='k', linestyle='-', linewidth=2)
circle = plt.Circle((qzCOM, qxCOM), radius_mean + dr, color='0', fill=False, linestyle='dotted')
fig.gca().add_artist(circle)
circle = plt.Circle((qzCOM, qxCOM), radius_mean - dr, color='0', fill=False, linestyle='dotted')
fig.gca().add_artist(circle)
plt.colorbar()
plt.xlabel(r"Q$_z$ ($1/\AA$)")
plt.ylabel(r"Q$_x$ ($1/\AA$)")
plt.axis('scaled')
plt.title('Sum(I) over Qy')
plt.subplot(2, 2, 2)
plt.contourf(qy, qx, xu.maplog(data.sum(axis=1)), 150, cmap=my_cmap)
plt.plot([min(qy), max(qy)], [qxCOM, qxCOM], color='k', linestyle='-', linewidth=2)
plt.plot([qyCOM, qyCOM], [min(qx), max(qx)], color='k', linestyle='-', linewidth=2)
circle = plt.Circle((qyCOM, qxCOM), radius_mean + dr, color='0', fill=False, linestyle='dotted')
fig.gca().add_artist(circle)
circle = plt.Circle((qyCOM, qxCOM), radius_mean - dr, color='0', fill=False, linestyle='dotted')
fig.gca().add_artist(circle)
plt.colorbar()
plt.xlabel(r"Q$_y$ ($1/\AA$)")
plt.ylabel(r"Q$_x$ ($1/\AA$)")
plt.axis('scaled')
plt.title('Sum(I) over Qz')
plt.subplot(2, 2, 3)
plt.contourf(qy, qz, xu.maplog(data.sum(axis=0)), 150, cmap=my_cmap)
plt.plot([qyCOM, qyCOM], [min(qz), max(qz)], color='k', linestyle='-', linewidth=2)
plt.plot([min(qy), max(qy)], [qzCOM, qzCOM], color='k', linestyle='-', linewidth=2)
circle = plt.Circle((qyCOM, qzCOM), radius_mean + dr, color='0', fill=False, linestyle='dotted')
fig.gca().add_artist(circle)
circle = plt.Circle((qyCOM, qzCOM), radius_mean - dr, color='0', fill=False, linestyle='dotted')
fig.gca().add_artist(circle)
plt.colorbar()
plt.xlabel(r"Q$_y$ ($1/\AA$)")
plt.ylabel(r"Q$_z$ ($1/\AA$)")
plt.axis('scaled')
plt.title('Sum(I) over Qx')
fig.text(0.60, 0.30, "Scan " + str(scan), size=20)
if reconstructed_data == 0:
    fig.text(0.60, 0.25, "offset_eta=" + str(offset_eta), size=20)
    fig.text(0.60, 0.20, "offset_phi=" + str(offset_phi), size=20)
    fig.text(0.60, 0.15, "offset_chi=" + str(offset_chi), size=20)
plt.pause(0.1)
plt.savefig(homedir + 'diffpattern' + comment + 'S' + str(scan) + '_q=' + str(radius_mean) + '.png')
####################################################################
#  plot upper and lower part of intensity with intersecting sphere #
####################################################################
if debug:
    fig, ax = plt.subplots(figsize=(20, 15), dpi=80, facecolor='w', edgecolor='k')
    plt.subplot(2, 3, 1)
    plt.contourf(qz_top, qx, xu.maplog(intensity_top.sum(axis=2), 6, 1), 75, cmap=my_cmap)
    plt.plot([qzCOM, max(qz)], [qxCOM, qxCOM], color='k', linestyle='-', linewidth=2)
    plt.plot([qzCOM, qzCOM], [min(qx), max(qx)], color='k', linestyle='-', linewidth=2)
    circle = plt.Circle((qzCOM, qxCOM), radius_mean + dr, color='0', fill=False, linestyle='dotted')
    fig.gca().add_artist(circle)
    circle = plt.Circle((qzCOM, qxCOM), radius_mean - dr, color='0', fill=False, linestyle='dotted')
    fig.gca().add_artist(circle)
    plt.xlabel(r"Q$_z$ ($1/\AA$)")
    plt.ylabel(r"Q$_x$ ($1/\AA$)")
    plt.title('Top\nSum(I) over Q$_y$')
    plt.axis('scaled')
    plt.subplot(2, 3, 2)
    plt.contourf(qy, qx, xu.maplog(intensity_top.sum(axis=1), 6, 1), 75, cmap=my_cmap)
    plt.plot([min(qy), max(qy)], [qxCOM, qxCOM], color='k', linestyle='-', linewidth=2)
    plt.plot([qyCOM, qyCOM], [min(qx), max(qx)], color='k', linestyle='-', linewidth=2)
    circle = plt.Circle((qyCOM, qxCOM), radius_mean + dr, color='0', fill=False, linestyle='dotted')
    fig.gca().add_artist(circle)
    circle = plt.Circle((qyCOM, qxCOM), radius_mean - dr, color='0', fill=False, linestyle='dotted')
    fig.gca().add_artist(circle)
    plt.xlabel(r"Q$_y$ ($1/\AA$)")
    plt.ylabel(r"Q$_x$ ($1/\AA$)")
    plt.title('Top\nSum(I) over Q$_z$>Q$_z$COM')
    plt.axis('scaled')
    plt.subplot(2, 3, 3)
    plt.contourf(qy, qz_top, xu.maplog(intensity_top.sum(axis=0), 6, 1), 75, cmap=my_cmap)
    plt.plot([qyCOM, qyCOM], [qzCOM, max(qz)], color='k', linestyle='-', linewidth=2)
    plt.plot([min(qy), max(qy)], [qzCOM, qzCOM], color='k', linestyle='-', linewidth=2)
    circle = plt.Circle((qyCOM, qzCOM), radius_mean + dr, color='0', fill=False, linestyle='dotted')
    fig.gca().add_artist(circle)
    circle = plt.Circle((qyCOM, qzCOM), radius_mean - dr, color='0', fill=False, linestyle='dotted')
    fig.gca().add_artist(circle)
    plt.xlabel(r"Q$_y$ ($1/\AA$)")
    plt.ylabel(r"Q$_z$ ($1/\AA$)")
    plt.title('Top\nSum(I) over Q$_x$')
    plt.axis('scaled')
    plt.subplot(2, 3, 4)
    plt.contourf(qz_bottom, qx, xu.maplog(intensity_bottom.sum(axis=2), 6, 1), 75, cmap=my_cmap)
    plt.plot([min(qz), qzCOM], [qxCOM, qxCOM], color='k', linestyle='-', linewidth=2)
    plt.plot([qzCOM, qzCOM], [min(qx), max(qx)], color='k', linestyle='-', linewidth=2)
    circle = plt.Circle((qzCOM, qxCOM), radius_mean + dr, color='0', fill=False, linestyle='dotted')
    fig.gca().add_artist(circle)
    circle = plt.Circle((qzCOM, qxCOM), radius_mean - dr, color='0', fill=False, linestyle='dotted')
    fig.gca().add_artist(circle)
    plt.xlabel(r"Q$_z$ ($1/\AA$)")
    plt.ylabel(r"Q$_x$ ($1/\AA$)")
    plt.title('Bottom\nSum(I) over Q$_y$')
    plt.axis('scaled')
    plt.subplot(2, 3, 5)
    plt.contourf(qy, qx, xu.maplog(intensity_bottom.sum(axis=1), 6, 1), 75, cmap=my_cmap)
    plt.plot([min(qy), max(qy)], [qxCOM, qxCOM], color='k', linestyle='-', linewidth=2)
    plt.plot([qyCOM, qyCOM], [min(qx), max(qx)], color='k', linestyle='-', linewidth=2)
    circle = plt.Circle((qyCOM, qxCOM), radius_mean + dr, color='0', fill=False, linestyle='dotted')
    fig.gca().add_artist(circle)
    circle = plt.Circle((qyCOM, qxCOM), radius_mean - dr, color='0', fill=False, linestyle='dotted')
    fig.gca().add_artist(circle)
    plt.xlabel(r"Q$_y$ ($1/\AA$)")
    plt.ylabel(r"Q$_x$ ($1/\AA$)")
    plt.title('Bottom\nSum(I) over Q$_z$<Q$_z$COM')
    plt.axis('scaled')
    plt.subplot(2, 3, 6)
    plt.contourf(qy, qz_bottom, xu.maplog(intensity_bottom.sum(axis=0), 6, 1), 75, cmap=my_cmap)
    plt.plot([qyCOM, qyCOM], [min(qz), qzCOM], color='k', linestyle='-', linewidth=2)
    plt.plot([min(qy), max(qy)], [qzCOM, qzCOM], color='k', linestyle='-', linewidth=2)
    circle = plt.Circle((qyCOM, qzCOM), radius_mean + dr, color='0', fill=False, linestyle='dotted')
    fig.gca().add_artist(circle)
    circle = plt.Circle((qyCOM, qzCOM), radius_mean - dr, color='0', fill=False, linestyle='dotted')
    fig.gca().add_artist(circle)
    plt.xlabel(r"Q$_y$ ($1/\AA$)")
    plt.ylabel(r"Q$_z$ ($1/\AA$)")
    plt.title('Bottom\nSum(I) over Q$_x$')
    plt.axis('scaled')
    plt.pause(0.1)

##############
# apply mask #
##############
I_masked_top = intensity_top*mask_top
I_masked_bottom = intensity_bottom*mask_bottom
if debug:
    fig, ax = plt.subplots(figsize=(20, 15), dpi=80, facecolor='w', edgecolor='k')
    plt.subplot(2, 3, 1)
    plt.contourf(qz_top, qx, xu.maplog(I_masked_top.sum(axis=2), 5, 1), 75, cmap=my_cmap)
    circle = plt.Circle((qzCOM, qxCOM), radius_mean + dr, color='0', fill=False, linestyle='dotted')
    fig.gca().add_artist(circle)
    circle = plt.Circle((qzCOM, qxCOM), radius_mean - dr, color='0', fill=False, linestyle='dotted')
    fig.gca().add_artist(circle)
    plt.xlabel(r"Q$_z$ ($1/\AA$)")
    plt.ylabel(r"Q$_x$ ($1/\AA$)")
    plt.title('Top\nSum(I*mask) over Q$_y$')
    plt.axis('scaled')
    plt.subplot(2, 3, 2)
    plt.contourf(qy, qx, xu.maplog(I_masked_top.sum(axis=1), 5, 1), 75, cmap=my_cmap)
    circle = plt.Circle((qyCOM, qxCOM), radius_mean + dr, color='0', fill=False, linestyle='dotted')
    fig.gca().add_artist(circle)
    circle = plt.Circle((qyCOM, qxCOM), radius_mean - dr, color='0', fill=False, linestyle='dotted')
    fig.gca().add_artist(circle)
    plt.xlabel(r"Q$_y$ ($1/\AA$)")
    plt.ylabel(r"Q$_x$ ($1/\AA$)")
    plt.title('Top\nSum(I*mask) over Q$_z$>Q$_z$COM')
    plt.axis('scaled')
    plt.subplot(2, 3, 3)
    plt.contourf(qy, qz_top, xu.maplog(I_masked_top.sum(axis=0), 5, 1), 75, cmap=my_cmap)
    circle = plt.Circle((qyCOM, qzCOM), radius_mean + dr, color='0', fill=False, linestyle='dotted')
    fig.gca().add_artist(circle)
    circle = plt.Circle((qyCOM, qzCOM), radius_mean - dr, color='0', fill=False, linestyle='dotted')
    fig.gca().add_artist(circle)
    plt.xlabel(r"Q$_y$ ($1/\AA$)")
    plt.ylabel(r"Q$_z$ ($1/\AA$)")
    plt.title('Top\nSum(I*mask) over Q$_x$')
    plt.axis('scaled')
    plt.subplot(2, 3, 4)
    plt.contourf(qz_bottom, qx, xu.maplog(I_masked_bottom.sum(axis=2), 5, 1), 75, cmap=my_cmap)
    circle = plt.Circle((qzCOM, qxCOM), radius_mean + dr, color='0', fill=False, linestyle='dotted')
    fig.gca().add_artist(circle)
    circle = plt.Circle((qzCOM, qxCOM), radius_mean - dr, color='0', fill=False, linestyle='dotted')
    fig.gca().add_artist(circle)
    plt.xlabel(r"Q$_z$ ($1/\AA$)")
    plt.ylabel(r"Q$_x$ ($1/\AA$)")
    plt.title('Bottom\nSum(I*mask) over Q$_y$')
    plt.axis('scaled')
    plt.subplot(2, 3, 5)
    plt.contourf(qy, qx, xu.maplog(I_masked_bottom.sum(axis=1), 5, 1), 75, cmap=my_cmap)
    circle = plt.Circle((qyCOM, qxCOM), radius_mean + dr, color='0', fill=False, linestyle='dotted')
    fig.gca().add_artist(circle)
    circle = plt.Circle((qyCOM, qxCOM), radius_mean - dr, color='0', fill=False, linestyle='dotted')
    fig.gca().add_artist(circle)
    plt.xlabel(r"Q$_y$ ($1/\AA$)")
    plt.ylabel(r"Q$_x$ ($1/\AA$)")
    plt.title('Bottom\nSum(I*mask) over Q$_z$<Q$_z$COM')
    plt.axis('scaled')
    plt.subplot(2, 3, 6)
    plt.contourf(qy, qz_bottom, xu.maplog(I_masked_bottom.sum(axis=0), 5, 1), 75, cmap=my_cmap)
    circle = plt.Circle((qyCOM, qzCOM), radius_mean + dr, color='0', fill=False, linestyle='dotted')
    fig.gca().add_artist(circle)
    circle = plt.Circle((qyCOM, qzCOM), radius_mean - dr, color='0', fill=False, linestyle='dotted')
    fig.gca().add_artist(circle)
    plt.xlabel(r"Q$_y$ ($1/\AA$)")
    plt.ylabel(r"Q$_z$ ($1/\AA$)")
    plt.title('Bottom\nSum(I*mask) over Q$_x$')
    plt.axis('scaled')
    plt.pause(0.1)

###############################################
# calculation of Euclidian metric coordinates #
###############################################
qx1_top = qx1*np.ones(intensity_top.shape)
qy1_top = qy1*np.ones(intensity_top.shape)
qx1_bottom = qx1*np.ones(intensity_bottom.shape)
qy1_bottom = qy1*np.ones(intensity_bottom.shape)
u_temp_top = (qx1_top - qxCOM)*radius_mean/(radius_mean+(qz1_top - qzCOM))  # projection from South pole
v_temp_top = (qy1_top - qyCOM)*radius_mean/(radius_mean+(qz1_top - qzCOM))  # projection from South pole
u_temp_bottom = (qx1_bottom - qxCOM)*radius_mean/(radius_mean+(qzCOM-qz1_bottom))  # projection from North pole
v_temp_bottom = (qy1_bottom - qyCOM)*radius_mean/(radius_mean+(qzCOM-qz1_bottom))  # projection from North pole
u_top = u_temp_top[mask_top]/radius_mean*90    # rescaling from radius_mean to 90
v_top = v_temp_top[mask_top]/radius_mean*90    # rescaling from radius_mean to 90
u_bottom = u_temp_bottom[mask_bottom]/radius_mean*90    # rescaling from radius_mean to 90
v_bottom = v_temp_bottom[mask_bottom]/radius_mean*90    # rescaling from radius_mean to 90

int_temp_top = I_masked_top[mask_top]
int_temp_bottom = I_masked_bottom[mask_bottom]
u_grid_top, v_grid_top = np.mgrid[-91:91:365j, -91:91:365j]
u_grid_bottom, v_grid_bottom = np.mgrid[-91:91:365j, -91:91:365j]
int_grid_top = griddata((u_top, v_top), int_temp_top, (u_grid_top, v_grid_top), method='linear')
int_grid_bottom = griddata((u_bottom, v_bottom), int_temp_bottom, (u_grid_bottom, v_grid_bottom), method='linear')
int_grid_top = int_grid_top / int_grid_top[int_grid_top > 0].max() * 10000  # normalize for easier plotting
int_grid_bottom = int_grid_bottom / int_grid_bottom[int_grid_bottom > 0].max() * 10000  # normalize for easier plotting
int_grid_top[np.isnan(int_grid_top)] = 0
int_grid_bottom[np.isnan(int_grid_bottom)] = 0

#########################################
# create top projection from South pole #
#########################################
# plot the stereographic projection
myfig, myax0 = plt.subplots(1, 1, figsize=(15, 10), facecolor='w', edgecolor='k')
# plot top part (projection from South pole on equator)
int_grid_top[int_grid_top < 10] = -1
plt0 = myax0.contourf(u_grid_top, v_grid_top, int_grid_top, range(range_min, range_max, range_step),
                      cmap=my_cmap)
plt.colorbar(plt0, ax=myax0)
myax0.axis('equal')
myax0.axis('off')

# # add the projection of the elevation angle, depending on the center of projection
for ii in range(15, 90, 5):
    circle = plt.Circle((0, 0),
                        radius_mean * np.sin(ii * np.pi / 180) / (1 + np.cos(ii * np.pi / 180)) * 90 / radius_mean,
                        color='grey', fill=False, linestyle='dotted', linewidth=0.2)
    myax0.add_artist(circle)
for ii in range(10, 90, 20):
    circle = plt.Circle((0, 0),
                        radius_mean * np.sin(ii * np.pi / 180) / (1 + np.cos(ii * np.pi / 180)) * 90 / radius_mean,
                        color='grey', fill=False, linestyle='dotted', linewidth=0.5)
    myax0.add_artist(circle)

if flag_plottext:
    for ii in range(10, 95, 20):
        myax0.text(-radius_mean * np.sin(ii * np.pi / 180) / (1 + np.cos(ii * np.pi / 180)) * 90 / radius_mean, 0,
                   str(ii) + '$^\circ$', fontsize=18, color='k', fontweight='bold')
circle = plt.Circle((0, 0), 90, color='k', fill=False, linewidth=1)
myax0.add_artist(circle)

# add azimutal lines every 5 and 45 degrees
for ii in range(5, 365, 5):
    myax0.plot([0, 90 * np.cos(ii * np.pi / 180)], [0, 90 * np.sin(ii * np.pi / 180)], color='grey',
               linestyle='dotted', linewidth=0.2)
for ii in range(0, 365, 20):
    myax0.plot([0, 90 * np.cos(ii * np.pi / 180)], [0, 90 * np.sin(ii * np.pi / 180)], color='grey',
               linestyle='dotted', linewidth=0.5)

# draw circles corresponding to particular reflection
if flag_plotplanes:
    indx = 5
    for key, value in planes.items():
        circle = plt.Circle((0, 0), radius_mean * np.sin(value * np.pi / 180) /
                            (1 + np.cos(value * np.pi / 180)) * 90 / radius_mean,
                            color='r', fill=False, linestyle='dotted', linewidth=2)
        myax0.add_artist(circle)
        if flag_plottext:
            myax0.text(np.cos(indx * np.pi / 180) * radius_mean * np.sin(value * np.pi / 180) /
                       (1 + np.cos(value * np.pi / 180)) * 90 / radius_mean,
                       np.sin(indx * np.pi / 180) * radius_mean * np.sin(value * np.pi / 180) /
                       (1 + np.cos(value * np.pi / 180)) * 90 / radius_mean,
                       key, fontsize=20, color='k', fontweight='bold')
            indx = indx + 5
        print(key + ": ", str('{:.2f}'.format(value)))
myax0.set_title('Top projection\nfrom South pole S' + str(scan)+'\n')
if reconstructed_data == 0:
    myfig.text(0.2, 0.05, "q=" + str(radius_mean) +
               " dq=" + str(dr) + " offset_eta=" + str(offset_eta) + " offset_phi=" + str(offset_phi) +
               " offset_chi=" + str(offset_chi), size=20)

else:
    myfig.text(0.4, 0.8, "q=" + str(radius_mean) + " dq=" + str(dr), size=20)
plt.pause(0.1)
plt.savefig(homedir + 'South pole' + comment + '_S' + str(scan) + '.png')
############################################
# create bottom projection from North pole #
############################################
myfig, myax1 = plt.subplots(1, 1, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
int_grid_bottom[int_grid_bottom < 10] = -1
plt1 = myax1.contourf(u_grid_bottom, v_grid_bottom, int_grid_bottom, range(range_min, range_max, range_step),
                      cmap=my_cmap)
plt.colorbar(plt1, ax=myax1)
myax1.axis('equal')
myax1.axis('off')

# # add the projection of the elevation angle, depending on the center of projection
for ii in range(15, 90, 5):
    circle = plt.Circle((0, 0),
                        radius_mean * np.sin(ii * np.pi / 180) / (1 + np.cos(ii * np.pi / 180)) * 90 / radius_mean,
                        color='grey', fill=False, linestyle='dotted', linewidth=0.2)
    myax1.add_artist(circle)
for ii in range(10, 90, 20):
    circle = plt.Circle((0, 0),
                        radius_mean * np.sin(ii * np.pi / 180) / (1 + np.cos(ii * np.pi / 180)) * 90 / radius_mean,
                        color='grey', fill=False, linestyle='dotted', linewidth=0.5)
    myax1.add_artist(circle)
if flag_plottext:
    for ii in range(10, 95, 20):
        myax1.text(-radius_mean * np.sin(ii * np.pi / 180) / (1 + np.cos(ii * np.pi / 180)) * 90 / radius_mean, 0,
                   str(ii) + '$^\circ$', fontsize=18, color='k', fontweight='bold')
circle = plt.Circle((0, 0), 90, color='k', fill=False, linewidth=1)
myax1.add_artist(circle)

# add azimutal lines every 5 and 45 degrees
for ii in range(5, 365, 5):
    myax1.plot([0, 90 * np.cos(ii * np.pi / 180)], [0, 90 * np.sin(ii * np.pi / 180)], color='grey',
               linestyle='dotted', linewidth=0.2)
for ii in range(0, 365, 20):
    myax1.plot([0, 90 * np.cos(ii * np.pi / 180)], [0, 90 * np.sin(ii * np.pi / 180)], color='grey',
               linestyle='dotted', linewidth=0.5)

# draw circles corresponding to particular reflection
if flag_plotplanes:
    indx = 0
    for key, value in planes.items():
        circle = plt.Circle((0, 0), radius_mean * np.sin(value * np.pi / 180) /
                            (1 + np.cos(value * np.pi / 180)) * 90 / radius_mean,
                            color='r', fill=False, linestyle='dotted', linewidth=2)
        myax1.add_artist(circle)
        if flag_plottext:
            myax1.text(np.cos(indx * np.pi / 180) * radius_mean * np.sin(value * np.pi / 180) /
                       (1 + np.cos(value * np.pi / 180)) * 90 / radius_mean,
                       np.sin(indx * np.pi / 180) * radius_mean * np.sin(value * np.pi / 180) /
                       (1 + np.cos(value * np.pi / 180)) * 90 / radius_mean,
                       key, fontsize=20, color='k', fontweight='bold')
            indx = indx + 5
        print(key + ": ", str('{:.2f}'.format(value)))
plt.title('Bottom projection\nfrom North pole S' + str(scan) + '\n')
# save figure
if reconstructed_data == 0:
    myfig.text(0.2, 0.05, "q=" + str(radius_mean) +
               " dq=" + str(dr) + " offset_eta=" + str(offset_eta) + " offset_phi=" + str(offset_phi) +
               " offset_chi=" + str(offset_chi), size=20)

else:
    myfig.text(0.4, 0.8, "q=" + str(radius_mean) + " dq=" + str(dr), size=20)
plt.pause(0.1)
plt.savefig(homedir + 'North pole' + comment + '_S' + str(scan) + '.png')

################################
# save grid points in txt file #
################################
fichier = open(homedir + 'Poles' + comment + '_S' + str(scan) + '.dat', "w")
# save metric coordinates in text file
for ii in range(len(u_grid_top)):
    for jj in range(len(v_grid_top)):
        fichier.write(str(u_grid_top[ii, 0]) + '\t' + str(v_grid_top[0, jj]) + '\t' +
                      str(int_grid_top[ii, jj]) + '\t' + str(u_grid_bottom[ii, 0]) + '\t' +
                      str(v_grid_bottom[0, jj]) + '\t' + str(int_grid_bottom[ii, jj]) + '\n')
fichier.close()
plt.ioff()
plt.show()
