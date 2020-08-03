# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import hdf5plugin  # for P10, should be imported before h5py or PyTables
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage.measurements import center_of_mass
import sys
sys.path.append('D:/myscripts/bcdi/')
import bcdi.experiment.experiment_utils as exp
import bcdi.preprocessing.preprocessing_utils as pru
import bcdi.graph.graph_utils as gu

helptext = """
Open a series of rocking curve data and track the position of the Bragg peak over the series.

Supported beamlines: ESRF ID01, PETRAIII P10, SOLEIL SIXS, SOLEIL CRISTAL.
"""

scans = np.arange(1009, 1075+1, step=3)  # list or array of scan numbers
scans = np.concatenate((scans, np.arange(1081, 1135+1, 3)))
scans = np.concatenate((scans, np.arange(1198, 1231+1, 3)))
scans = np.concatenate((scans, np.arange(1236, 1395+1, 3)))
root_folder = "D:/data/P10_OER/data/"
sample_name = "dewet2_2"  # list of sample names. If only one name is indicated,
# it will be repeated to match the number of scans
savedir = "D:/data/P10_OER/analysis/candidate_11/"
# images will be saved here, leave it to '' otherwise (default to root_folder)
x_axis = [0.8, 0.8, 0.9, 0.9, 1.0, 1.0, 1.1, 1.1, 1.2, 1.2, 1.1, 1.1, 1.0, 1.0, 0.9, 0.9, 0.8, 0.8, 0.7, 0.7, 0.6, 0.6,
          0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.3, 0.3, 0.2, 0.2, 0.1, 0.1, 0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4,
          0.5, 0.5, 0.6, 0.6, 0.7, 0.7, 0.8, 0.8, 0.9, 0.9, 0.9, 0.9, 1.0, 1.0, 1.1, 1.1, 1.2, 1.2, 1.1, 1.1, 1.0, 1.0,
          0.9, 0.9, 0.8, 0.8, 0.7, 0.7, 0.6, 0.6, 0.5, 0.5, 0.4, 0.4, 0.3, 0.3, 0.2, 0.2, 0.1, 0.1, 0.0, 0.0, 0.1, 0.1,
          0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5, 0.6, 0.6, 0.7, 0.7, 0.8, 0.8, 0.9, 0.9, 1.0, 1.0, 1.1, 1.1]
# values against which the Bragg peak center of mass evolution will be plotted, leave [] otherwise
x_label = 'voltage (V)'  # label for the X axis in plots, leave '' otherwise
comment = '_small_RC'  # comment for the saving filename, should start with _
debug = False  # set to True to see plots
###############################
# beamline related parameters #
###############################
beamline = 'P10'  # name of the beamline, used for data loading and normalization by monitor
# supported beamlines: 'ID01', 'SIXS_2018', 'SIXS_2019', 'CRISTAL', 'P10'

custom_scan = False  # True for a stack of images acquired without scan, e.g. with ct in a macro (no info in spec file)
custom_images = np.arange(11353, 11453, 1)  # list of image numbers for the custom_scan
custom_monitor = np.ones(len(custom_images))  # monitor values for normalization for the custom_scan
custom_motors = {"eta": np.linspace(16.989, 18.989, num=100, endpoint=False), "phi": 0, "nu": -0.75, "delta": 36.65}
# ID01: eta, phi, nu, delta
# CRISTAL: mgomega, gamma, delta
# P10: om, phi, chi, mu, gamma, delta
# SIXS: beta, mu, gamma, delta

rocking_angle = "outofplane"  # "outofplane" or "inplane"
is_series = False  # specific to series measurement at P10
specfile_name = ''
# .spec for ID01, .fio for P10, alias_dict.txt for SIXS_2018, not used for CRISTAL and SIXS_2019
# template for ID01: name of the spec file without '.spec'
# template for SIXS_2018: full path of the alias dictionnary 'alias_dict.txt', typically: root_folder + 'alias_dict.txt'
# template for SIXS_2019: ''
# template for P10: ''
# template for CRISTAL: ''
###############################
# detector related parameters #
###############################
detector = "Eiger4M"    # "Eiger2M" or "Maxipix" or "Eiger4M"
roi_detector = []  # [Vstart, Vstop, Hstart, Hstop]
# leave it as [] to use the full detector. Use with center_fft='skip' if you want this exact size.
peak_method = 'max'  # Bragg peak determination: 'max', 'com' or 'maxcom'.
hotpixels_file = ''  # root_folder + 'hotpixels.npz'  #
flatfield_file = ''  # root_folder + "flatfield_8.5kev.npz"  #
template_imagefile = '_master.h5'
# template for ID01: 'data_mpx4_%05d.edf.gz' or 'align_eiger2M_%05d.edf.gz'
# template for SIXS_2018: 'align.spec_ascan_mu_%05d.nxs'
# template for SIXS_2019: 'spare_ascan_mu_%05d.nxs'
# template for Cristal: 'S%d.nxs'
# template for P10: '_master.h5'
####################################
# q calculation related parameters #
####################################
convert_to_q = True  # True to convert from pixels to q values using parameters below
beam_direction = (1, 0, 0)  # beam along z
directbeam_x = 476  # x horizontal,  cch2 in xrayutilities
directbeam_y = 1374  # y vertical,  cch1 in xrayutilities
direct_inplane = -2.0  # outer angle in xrayutilities
direct_outofplane = 0.8
sdd = 1.83  # sample to detector distance in m
energy = 10300  # in eV, offset of 6eV at ID01
##################################
# end of user-defined parameters #
##################################

########################################
# check and initialize some parameters #
########################################
print('\n{:d} scans:'.format(len(scans)), scans)
print('\n {:d} x_axis values provided:'.format(len(x_axis)))
if len(x_axis) == 0:
    x_axis = np.arange(len(scans))
assert len(x_axis) == len(scans), 'the length of x_axis should be equal to the number of scans'

if type(sample_name) is list:
    if len(sample_name) == 1:
        sample_name = [sample_name[0] for idx in range(len(scans))]
    assert len(sample_name) == len(scans), 'sample_name and scan_list should have the same length'
elif type(sample_name) is str:
    sample_name = [sample_name for idx in range(len(scans))]
else:
    print('sample_name should be either a string or a list of strings')
    sys.exit()

int_sum = []
int_max = []
xcom = []
ycom = []
zcom = []
tilt_com = []
q_com = []

#################################
# Initialize detector and setup #
#################################
kwargs = dict()  # create dictionnary
kwargs['is_series'] = is_series
detector = exp.Detector(name=detector, datadir='', template_imagefile=template_imagefile, roi=roi_detector, **kwargs)

setup_pre = exp.SetupPreprocessing(beamline=beamline, rocking_angle=rocking_angle, custom_scan=custom_scan,
                                   custom_images=custom_images, custom_monitor=custom_monitor,
                                   custom_motors=custom_motors)

flatfield = pru.load_flatfield(flatfield_file)
hotpix_array = pru.load_hotpixels(hotpixels_file)

print('Setup: ', setup_pre.beamline)
print('Detector: ', detector.name)
print('Pixel number (VxH): ', detector.nb_pixel_y, detector.nb_pixel_x)
print('Detector ROI:', roi_detector)
print('Horizontal pixel size with binning: ', detector.pixelsize_x, 'm')
print('Vertical pixel size with binning: ', detector.pixelsize_y, 'm')
print('Scan type: ', setup_pre.rocking_angle)

if savedir == '':
    savedir = root_folder
detector.savedir = savedir
print('savedir: ', detector.savedir)

###############################################
# load recursively the scans and update lists #
###############################################
for scan_nb in range(len(scans)):

    if setup_pre.beamline != 'P10':
        homedir = root_folder + sample_name[scan_nb] + str(scans[scan_nb]) + '/'
        detector.datadir = homedir + "data/"
        specfile = specfile_name
    else:
        specfile = sample_name[scan_nb] + '_{:05d}'.format(scans[scan_nb])
        homedir = root_folder + specfile + '/'
        detector.datadir = homedir + 'e4m/'
        imagefile = specfile + template_imagefile
        detector.template_imagefile = imagefile

    logfile = pru.create_logfile(setup=setup_pre, detector=detector, scan_number=scans[scan_nb],
                                 root_folder=root_folder, filename=specfile)

    print('\nScan', scans[scan_nb])
    print('Specfile: ', specfile)

    data, mask, frames_logical, monitor = pru.load_bcdi_data(logfile=logfile, scan_number=scans[scan_nb],
                                                             detector=detector, setup=setup_pre,
                                                             flatfield=flatfield, hotpixels=hotpix_array,
                                                             normalize=True, debugging=debug)

    tilt, grazing, inplane, outofplane = pru.motor_values(frames_logical=frames_logical, logfile=logfile,
                                                          scan_number=scan_nb, setup=setup_pre)

    piz, piy, pix = center_of_mass(data)
    zcom.append(piz)
    ycom.append(piy)
    xcom.append(pix)
    int_sum.append(data.sum())
    int_max.append(data.max())

    interp_tilt = interp1d(np.arange(data.shape[0]), tilt, kind='linear')
    tilt_com.append(interp_tilt(piz))
    # tilt_com.append(tilt[int(piz)])
    if scan_nb == 0 or scan_nb == len(scans) - 1:
        gu.multislices_plot(data, sum_frames=True, scale='log', reciprocal_space=True, is_orthogonal=False,
                            title='scan {:d}'.format(scans[scan_nb]))

    ##############################
    # convert pixels to q values #
    ##############################
    if convert_to_q:
        setup_post = exp.SetupPostprocessing(beamline=setup_pre.beamline, energy=setup_pre.energy,
                                             outofplane_angle=outofplane, inplane_angle=inplane, tilt_angle=tilt,
                                             rocking_angle=setup_pre.rocking_angle, grazing_angle=grazing,
                                             distance=setup_pre.distance, pixel_x=detector.pixelsize_x,
                                             pixel_y=detector.pixelsize_y)
        bragg_x = detector.roi[2] + pix  # convert it in full detector pixel
        bragg_y = detector.roi[0] + piy  # convert it in full detector pixel

        x_direct_0 = directbeam_x + setup_post.rotation_direction() * \
            (direct_inplane * np.pi / 180 * sdd / detector.pixelsize_x)  # rotation_direction is +1 or -1
        y_direct_0 = directbeam_y - direct_outofplane * np.pi / 180 * sdd / detector.pixelsize_y
        # outofplane is always clockwise

        bragg_inplane = inplane + setup_post.rotation_direction() * \
            (detector.pixelsize_x * (bragg_x - x_direct_0) / sdd * 180 / np.pi)  # rotation_direction is +1 or -1
        bragg_outofplane = outofplane - detector.pixelsize_y * (bragg_y - y_direct_0) / sdd * 180 / np.pi
        # outofplane always clockwise
        print("Bragg angles before correction = (gam, del): ", str('{:.4f}'.format(inplane)),
              str('{:.4f}'.format(outofplane)))
        print("Bragg angles after correction = (gam, del): ", str('{:.4f}'.format(bragg_inplane)),
              str('{:.4f}'.format(bragg_outofplane)))

        # update setup_post with the corrected detector angles
        setup_post.inplane_angle = bragg_inplane
        setup_post.outofplane_angle = bragg_outofplane

        ##############################################################
        # wavevector transfer calculations (in the laboratory frame) #
        ##############################################################
        kin = 2 * np.pi / setup_post.wavelength * np.asarray(beam_direction)
        # in lab frame z downstream, y vertical, x outboard
        kout = setup_post.exit_wavevector()  # in lab.frame z downstream, y vertical, x outboard
        q = (kout - kin) / 1e10  # convert from 1/m to 1/angstrom
        q_com.append(np.linalg.norm(q))
        print("Wavevector transfer of Bragg peak: ", q, str('{:.4f}'.format(np.linalg.norm(q))))

##########################################################
# plot the evolution of the center of mass and intensity #
##########################################################
plt.ion()
fig, ((ax0, ax1, ax2), (ax3, ax4, ax5)) = plt.subplots(nrows=2, ncols=3, figsize=(12, 9))
ax0.plot(scans, x_axis, '-o')
ax0.set_xlabel('Scan number')
ax0.set_ylabel(x_label)
ax1.plot(x_axis, int_sum, '-o')
ax1.set_xlabel(x_label)
ax1.set_ylabel('Integrated intensity')
ax2.plot(x_axis, int_max, '-o')
ax2.set_xlabel(x_label)
ax2.set_ylabel('Maximum intensity')
ax3.plot(x_axis, xcom, '-o')
ax3.set_xlabel(x_label)
ax3.set_ylabel('xcom (pixels)')
ax4.plot(x_axis, ycom, '-o')
ax4.set_xlabel(x_label)
ax4.set_ylabel('ycom (pixels)')
ax5.plot(x_axis, zcom, '-o')
ax5.set_xlabel(x_label)
ax5.set_ylabel('zcom (pixels)')
plt.tight_layout()
plt.pause(0.1)
fig.savefig(savedir + 'summary' + comment + '.png')

fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
ax0.plot(scans, tilt_com, '-o')
ax0.set_xlabel('Scan number')
ax0.set_ylabel('Bragg angle (deg)')
ax1.plot(x_axis, tilt_com, '-o')
ax1.set_xlabel(x_label)
ax1.set_ylabel('Bragg angle (deg)')
plt.tight_layout()
plt.pause(0.1)
fig.savefig(savedir + 'Bragg angle' + comment + '.png')

if convert_to_q:
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
    ax0.plot(scans, q_com, '-o')
    ax0.set_xlabel('Scan number')
    ax0.set_ylabel('q (1/A)')
    ax1.plot(x_axis, q_com, '-o')
    ax1.set_xlabel(x_label)
    ax1.set_ylabel('q (1/A)')
    plt.tight_layout()
    plt.pause(0.1)
    fig.savefig(savedir + 'diffusion vector' + comment + '.png')

plt.ioff()
plt.show()
