# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import hdf5plugin  # for P10, should be imported before h5py or PyTables
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import sys
sys.path.append('D:/myscripts/bcdi/')
import bcdi.experiment.experiment_utils as exp
import bcdi.preprocessing.preprocessing_utils as pru
import bcdi.graph.graph_utils as gu

helptext = """
Open a rocking curve data, plot the mask, the monitor and the stack along the first axis.

It is usefull when you want to localize the Bragg peak for ROI determination.

Supported beamlines: ESRF ID01, PETRAIII P10, SOLEIL SIXS, SOLEIL CRISTAL.
"""

scan = 11
root_folder = "D:/data/Pt THH ex-situ/Data/CH4760/"
sample_name = "S"  # string in front of the scan number in the folder name
savedir = None  # images will be saved here, leave it to None otherwise (default to data directory's parent)
save_mask = False  # set to True to save the mask
debug = False  # True to see more plots
###############################
# beamline related parameters #
###############################
beamline = 'ID01'  # name of the beamline, used for data loading and normalization by monitor
# supported beamlines: 'ID01', 'SIXS_2018', 'SIXS_2019', 'CRISTAL', 'P10', 'NANOMAX'

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
specfile_name = 'l5'
# .spec for ID01, .fio for P10, alias_dict.txt for SIXS_2018, not used for CRISTAL and SIXS_2019
# template for ID01: name of the spec file without '.spec'
# template for SIXS_2018: full path of the alias dictionnary 'alias_dict.txt', typically: root_folder + 'alias_dict.txt'
# template for SIXS_2019: ''
# template for P10: ''
# template for NANOMAX: ''
# template for CRISTAL: ''
###############################
# detector related parameters #
###############################
detector = "Maxipix"    # "Eiger2M" or "Maxipix" or "Eiger4M" or 'Merlin'
bragg_position = []  # Bragg peak position [vertical, horizontal], leave it as [] if there is a single peak
peak_method = 'maxcom'  # Bragg peak determination: 'max', 'com' or 'maxcom'.
high_threshold = 150000  # everything above will be considered as hotpixel
hotpixels_file = ''  # root_folder + 'merlin_mask_190222_14keV.h5'  #
flatfield_file = root_folder + "flatfield_maxipix_8kev.npz"  #
template_imagefile = 'data_mpx4_%05d.edf.gz'
# template for ID01: 'data_mpx4_%05d.edf.gz' or 'align_eiger2M_%05d.edf.gz'
# template for SIXS_2018: 'align.spec_ascan_mu_%05d.nxs'
# template for SIXS_2019: 'spare_ascan_mu_%05d.nxs'
# template for Cristal: 'S%d.nxs'
# template for P10: '_master.h5'
# template for NANOMAX: '%06d.h5'
######################
# setup for the plot #
######################
vmin = 0  # min of the colorbar (log scale)
vmax = 6  # max of the colorbar (log scale)
low_threshold = 1  # everthing <= 1 will be set to 0 in the plot
width = [50, 50]  # [vertical, horizontal]
# half width in pixels of the region of interest centered on the peak for the plot
##################################
# end of user-defined parameters #
##################################

###################
# define colormap #
###################
bad_color = '1.0'  # white background
colormap = gu.Colormap(bad_color=bad_color)
my_cmap = colormap.cmap
plt.ion()


#################################################
# Initialize detector, setup, paths and logfile #
#################################################
detector = exp.Detector(name=detector, datadir='', template_imagefile=template_imagefile, is_series=is_series)

setup = exp.SetupPreprocessing(beamline=beamline, rocking_angle=rocking_angle, custom_scan=custom_scan,
                               custom_images=custom_images, custom_monitor=custom_monitor, custom_motors=custom_motors)

if setup.beamline == 'P10':
    specfile_name = sample_name + '_{:05d}'.format(scan)
    homedir = root_folder + specfile_name + '/'
    detector.datadir = homedir + 'e4m/'
    template_imagefile = specfile_name + template_imagefile
    detector.template_imagefile = template_imagefile
elif setup.beamline == 'SIXS_2018' or setup.beamline == 'SIXS_2019':
    homedir = root_folder 
    detector.datadir = homedir + "align/"
elif setup.beamline == 'NANOMAX':
    homedir = root_folder + sample_name + '{:06d}'.format(scan) + '/'
    detector.datadir = homedir + 'data/'
else:
    homedir = root_folder + sample_name + str(scan) + '/'
    detector.datadir = homedir + "data/"

savedir = savedir or os.path.abspath(os.path.join(detector.datadir, os.pardir)) + '/'

detector.savedir = savedir
print('savedir: ', savedir)

flatfield = pru.load_flatfield(flatfield_file)
hotpix_array = pru.load_hotpixels(hotpixels_file)

logfile = pru.create_logfile(setup=setup, detector=detector, scan_number=scan, root_folder=root_folder,
                             filename=specfile_name)

data, mask, monitor, frames_logical = pru.load_data(logfile=logfile, scan_number=scan, detector=detector,
                                                    setup=setup, flatfield=flatfield, hotpixels=hotpix_array,
                                                    debugging=debug)

numz, numy, numx = data.shape
print('Data shape: ', numz, numy, numx)

##########################
# apply photon threshold #
##########################
if high_threshold != 0:
    nb_thresholded = (data > high_threshold).sum()
    mask[data > high_threshold] = 1
    data[data > high_threshold] = 0
    print("Applying photon threshold, {:d} high intensity pixels masked".format(nb_thresholded))

######################################################
# calculate rocking curve and fit it to get the FWHM #
######################################################
if data.ndim == 3:
    tilt, _, _, _ = pru.motor_values(frames_logical=frames_logical, logfile=logfile, scan_number=scan, setup=setup,
                                     follow_bragg=False)
    rocking_curve = np.zeros(numz)

    z0, y0, x0 = pru.find_bragg(data, peak_method=peak_method)
    z0 = np.rint(z0).astype(int)
    y0 = np.rint(y0).astype(int)
    x0 = np.rint(x0).astype(int)

    if len(bragg_position) == 0:  # Bragg peak position not defined by te user, find the max
        bragg_position.append(y0)
        bragg_position.append(x0)

    print("Bragg peak (full detector) at (z, y, x): ", z0, y0, x0)

    for idx in range(numz):
        rocking_curve[idx] = data[idx, bragg_position[0] - 50:bragg_position[0] + 50,
                                  bragg_position[1] - 50:bragg_position[1] + 50].sum()
    plot_title = "Rocking curve for a ROI centered on (y, x): " + str(bragg_position[0]) + ',' + str(bragg_position[1])

    z0 = np.unravel_index(rocking_curve.argmax(), rocking_curve.shape)[0]

    interpolation = interp1d(tilt, rocking_curve, kind='cubic')
    interp_points = 5 * numz
    interp_tilt = np.linspace(tilt.min(), tilt.max(), interp_points)
    interp_curve = interpolation(interp_tilt)
    interp_fwhm = len(np.argwhere(interp_curve >= interp_curve.max() / 2)) * \
                     (tilt.max() - tilt.min()) / (interp_points - 1)
    print('FWHM by interpolation', str('{:.3f}'.format(interp_fwhm)), 'deg')

    _, (ax0, ax1) = plt.subplots(2, 1, sharex='col', figsize=(10, 5))
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

    # apply low threshold
    data[data <= low_threshold] = 0
    # data = data[data.shape[0]//2, :, :]  # select the first frame e.g. for detector mesh scan
    data = data.sum(axis=0)  # concatenate along the axis of the rocking curve
    title = f'data.sum(axis=0)   peak method={peak_method}\n'
else:  # 2D
    _, y0, x0 = pru.find_bragg(data, peak_method=peak_method)
    # apply low threshold
    data[data <= low_threshold] = 0
    title = f'peak method={peak_method}\n'

############################################
# plot mask, monitor and concatenated data #
############################################
if save_mask:
    np.savez_compressed(detector.savedir + 'hotpixels.npz', mask=mask)

gu.combined_plots(tuple_array=(monitor, mask), tuple_sum_frames=False, tuple_sum_axis=(0, 0),
                  tuple_width_v=None, tuple_width_h=None, tuple_colorbar=(True, False), tuple_vmin=np.nan,
                  tuple_vmax=np.nan, tuple_title=('monitor', 'mask'), tuple_scale='linear', cmap=my_cmap,
                  ylabel=('Counts (a.u.)', ''))

max_y, max_x = np.unravel_index(abs(data).argmax(), data.shape)
print("Max at (y, x): ", max_y, max_x, ' Max = ', int(data[max_y, max_x]))

# check the width for plotting the region of interest centered on the peak
width[0] = min(width[0], y0, numy-y0)
width[1] = min(width[1], x0, numx-x0)

# plot the region of interest centered on the peak
# extent (left, right, bottom, top)
fig, ax = plt.subplots(nrows=1, ncols=1)
plot = ax.imshow(np.log10(data[y0-width[0]:y0+width[0], x0-width[1]:x0+width[1]]), vmin=vmin, vmax=vmax, cmap=my_cmap,
                 extent=[x0-width[1], x0+width[1], y0+width[0], y0-width[0]])
ax.set_title(f'{title} Peak at (y, x): ({y0},{x0})   Peak value = {int(data[y0, x0])}')
if beamline == 'NANOMAX':
    ax.invert_yaxis()  # the detector is mounted upside-down on the robot arm at Nanomax
gu.colorbar(plot)
fig.savefig(detector.savedir + 'sum_S' + str(scan) + '.png')
plt.show()
