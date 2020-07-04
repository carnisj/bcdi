# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

helptext = """
Open mesh scans and plot interactively the integrated intensity vs. motor positions for a user-defined
region of interest.
"""

import hdf5plugin  # for lz4 filter
import h5py
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import os
import sys
sys.path.append('D:/myscripts/bcdi/')
import bcdi.graph.graph_utils as gu
import bcdi.experiment.experiment_utils as exp
import bcdi.preprocessing.preprocessing_utils as pru
import bcdi.postprocessing.postprocessing_utils as pu

scan = 38  # scan number as it appears in the folder name
sample_name = "p15"  # without _ at the end
root_folder = "D:/data/P10_isosurface/data/"
savedir = ''  # images will be saved here, leave it to '' otherwise (default to data directory's parent)
normalize_flux = False  # will normalize the intensity by the default monitor
###########################
# mesh related parameters #
###########################
fast_motor = 'hpy'  # fast scanning motor for the mesh
nb_fast = 41  # number of steps for the fast scanning motor
slow_motor = 'hpx'  # slow scanning motor for the mesh
nb_slow = 9  # number of steps for the slow scanning motor
###########################
# plot related parameters #
###########################
background_plot = '0.5'  # in level of grey in [0,1], 0 being dark. For visual comfort
fast_axis = 'horizontal'  # 'vertical' to plot the fast scanning motor vertically, 'horizontal' otherwise
invert_xaxis = False  # True to inverse the horizontal axis
invert_yaxis = False  # True to inverse the horizontal axis
###############################
# beamline related parameters #
###############################
beamline = 'P10'  # name of the beamlisne, used for data loading and normalization by monitor
# supported beamlines: 'P10' only for now, see preprocessing_utils.get_motor_pos()
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
binning = [4, 4]  # binning (detector vertical axis, detector horizontal axis) applied during data loading
template_imagefile = '_master.h5'
# template for ID01: 'data_mpx4_%05d.edf.gz' or 'align_eiger2M_%05d.edf.gz'
# template for SIXS_2018: 'align.spec_ascan_mu_%05d.nxs'
# template for SIXS_2019: 'spare_ascan_mu_%05d.nxs'
# template for Cristal: 'S%d.nxs'
# template for P10: '_master.h5'
##########################
# end of user parameters #
##########################


def onselect(click, release):
    """
    Process mouse click and release events in the interactive plot

    :param click: position of the mouse click event
    :param release: position of the mouse release event
    """
    global ax1, data, nb_slow, nb_fast, my_cmap, min_fast, min_slow, max_fast, max_slow, fast_motor
    global slow_motor, ny, nx, invert_xaxis, invert_yaxis

    y_start, y_stop, x_start, x_stop = int(click.ydata), int(release.ydata), int(click.xdata), int(release.xdata)

    ax1.cla()
    if fast_axis == 'vertical':
        ax1.imshow(np.log10(data[:, y_start:y_stop, x_start:x_stop].sum(axis=(1, 2)).reshape((nb_fast, nb_slow))),
                   cmap=my_cmap, extent=[min_slow, max_slow, max_fast, min_fast])  # extent (left, right, bottom, top)
        ax1.set_xlabel(slow_motor)
        ax1.set_ylabel(fast_motor)
    else:
        ax1.imshow(np.log10(data[:, y_start:y_stop, x_start:x_stop].sum(axis=(1, 2)).reshape((nb_slow, nb_fast))),
                   cmap=my_cmap, extent=[min_fast, max_fast, max_slow, min_slow])  # extent (left, right, bottom, top)
        ax1.set_xlabel(fast_motor)
        ax1.set_ylabel(slow_motor)
    if invert_xaxis:
        ax1.invert_xaxis()
    if invert_yaxis:
        ax1.invert_yaxis()
    ax1.axis('scaled')
    ax1.set_title("integrated intensity in the ROI")
    plt.draw()


def press_key(event):
    """
    Process key press events in the interactive plot

    :param event: button press event
    """
    global sumdata, max_colorbar, ax0

    if event.key == 'right':
        max_colorbar = max_colorbar + 1
    elif event.key == 'left':
        max_colorbar = max_colorbar - 1
        if max_colorbar < 1:
            max_colorbar = 1

    ax0.cla()
    ax0.imshow(np.log10(sumdata), vmin=0, vmax=max_colorbar)
    ax0.set_title("detector plane (sum)")
    ax0.axis('scaled')
    plt.draw()


#########################
# check some parameters #
#########################
assert fast_axis in ['vertical', 'horizontal'], print('fast_axis parameter value not supported')

###################
# define colormap #
###################
bad_color = '1.0'  # white background
colormap = gu.Colormap(bad_color=bad_color)
my_cmap = colormap.cmap
plt.ion()

#################################################
# initialize detector, setup, paths and logfile #
#################################################
kwargs = dict()  # create dictionnary
kwargs['is_series'] = is_series
detector = exp.Detector(name=detector, datadir='', template_imagefile=template_imagefile,
                        binning=[1, binning[0], binning[1]], **kwargs)

setup = exp.SetupPreprocessing(beamline=beamline)

if setup.beamline == 'P10':
    specfile_name = sample_name + '_{:05d}'.format(scan)
    homedir = root_folder + specfile_name + '/'
    detector.datadir = homedir + 'e4m/'
    template_imagefile = specfile_name + template_imagefile
    detector.template_imagefile = template_imagefile
elif setup.beamline == 'SIXS_2018' or setup.beamline == 'SIXS_2019':
    homedir = root_folder
    detector.datadir = homedir + "align/"
else:
    homedir = root_folder + sample_name + str(scan) + '/'
    detector.datadir = homedir + "data/"

if savedir == '':
    savedir = os.path.abspath(os.path.join(detector.datadir, os.pardir)) + '/'

detector.savedir = savedir
print('savedir: ', savedir)

logfile = pru.create_logfile(setup=setup, detector=detector, scan_number=scan, root_folder=root_folder,
                             filename=specfile_name)

#############
# load data #
#############
data, mask, monitor, frames_logical = pru.load_data(logfile=logfile, scan_number=scan, detector=detector,
                                                    setup=setup, bin_during_loading=True, debugging=False)
print('Data shape: ', data.shape)

###########################
# intensity normalization #
###########################
if normalize_flux:
    print('Intensity normalization using the default monitor')
    data, monitor = pru.normalize_dataset(array=data, raw_monitor=monitor, frames_logical=frames_logical,
                                          norm_to_min=True, savedir=detector.savedir, debugging=True)

############
# bin data #
############
# # bin data and mask in the detector plane if needed
# if (detector.binning[1] != 1) or (detector.binning[2] != 1):
#     print('Binning the data: detector vertical axis by', detector.binning[1],
#           ', detector horizontal axis by', detector.binning[2])
#     data = pu.bin_data(data, (1, detector.binning[1], detector.binning[2]), debugging=False)
#     mask = pu.bin_data(mask, (1, detector.binning[1], detector.binning[2]), debugging=False)
#     mask[np.nonzero(mask)] = 1

data[np.nonzero(mask)] = 0
nz, ny, nx = data.shape
# print('Binned data shape:', data.shape)

########################
# load motor positions #
########################
fast_positions = pru.get_motor_pos(logfile=logfile, scan_number=scan, setup=setup, motor_name=fast_motor)
slow_positions = pru.get_motor_pos(logfile=logfile, scan_number=scan, setup=setup, motor_name=slow_motor)

min_fast, max_fast = fast_positions[0], fast_positions[-1]
min_slow, max_slow = slow_positions[0], slow_positions[-1]

assert len(fast_positions) == nz, print('Number of fast scanning motor steps:', nb_fast,
                                        'incompatible with data shape:', nz)
assert len(slow_positions) == nz, print('Number of slow scanning motor steps:', nb_slow,
                                        'incompatible with data shape:', nz)

####################
# interactive plot #
####################
sumdata = data.sum(axis=0)
max_colorbar = 5
rectprops = dict(edgecolor='black', fill=False)  # rectangle properties
plt.ioff()

figure = plt.figure(figsize=(12, 9))
ax0 = figure.add_subplot(121)
ax1 = figure.add_subplot(122)
figure.canvas.mpl_disconnect(figure.canvas.manager.key_press_handler_id)
original_data = np.copy(data)
ax0.imshow(np.log10(sumdata), cmap=my_cmap, vmin=0, vmax=max_colorbar)
if fast_axis == 'vertical':
    ax1.imshow(np.log10(data.sum(axis=(1, 2)).reshape((nb_fast, nb_slow))),
               cmap=my_cmap, extent=[min_slow, max_slow, max_fast, min_fast])  # extent (left, right, bottom, top)
    ax1.set_xlabel(slow_motor)
    ax1.set_ylabel(fast_motor)
else:
    ax1.imshow(np.log10(data.sum(axis=(1, 2)).reshape((nb_slow, nb_fast))),
               cmap=my_cmap, extent=[min_fast, max_fast, max_slow, min_slow])  # extent (left, right, bottom, top)
    ax1.set_xlabel(fast_motor)
    ax1.set_ylabel(slow_motor)
if invert_xaxis:
    ax1.invert_xaxis()
if invert_yaxis:
    ax1.invert_yaxis()
ax0.axis('scaled')
ax1.axis('scaled')
ax0.set_title("sum of all images")
ax1.set_title("integrated intensity in the ROI")
plt.tight_layout()
plt.connect('key_press_event', press_key)
rectangle = RectangleSelector(ax0, onselect, drawtype='box', useblit=False, button=[1], interactive=True,
                              rectprops=rectprops)  # don't use middle and right buttons
figure.set_facecolor(background_plot)
plt.show()
