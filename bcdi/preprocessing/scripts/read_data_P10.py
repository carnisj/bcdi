# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

helptext = """
Open images or series data at P10 beamline.
"""

import hdf5plugin  # for lz4 filter
import h5py
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('D:/myscripts/bcdi/')
import bcdi.graph.graph_utils as gu
import bcdi.experiment.experiment_utils as exp
import bcdi.preprocessing.preprocessing_utils as pru

scan = 0  # scan number as it appears in the folder name
sample_name = "gold_2_2_2"  # without _ at the end
rootdir = "D:/data/P10_August2019/data/"
image_nb = 1  # np.arange(1, 381+1)
# list of file numbers, e.g. [1] for gold_2_2_2_00022_data_000001.h5
detector = "Eiger4M"    # "Eiger2M" or "Maxipix" or "Eiger4M"
high_threshold = 9  # data points where log10(data) > high_threshold will be masked
# if data is a series, the condition becomes log10(data.sum(axis=0)) > high_threshold
savedir = ''  # images will be saved here, leave it to '' otherwise (default to data directory's parent)
is_scan = False  # set to True is the measurement is a scan or a time series, False for a single image
compare_ends = True  # set to True to plot the difference between the last frame and the first frame
save_mask = False  # True to save the mask as 'hotpixels.npz'
##########################
# end of user parameters #
##########################

#######################
# Initialize detector #
#######################
detector = exp.Detector(name=detector)

####################
# Initialize paths #
####################
if type(image_nb) == int:
    image_nb = [image_nb]

nb_files = len(image_nb)

if is_scan:  # scan or time series
    datadir = rootdir + sample_name + '_' + str('{:05d}'.format(scan)) + '/e4m/'
else:  # single image
    datadir = rootdir + sample_name + '/e4m/'
    compare_ends = False

if savedir == '':
    savedir = os.path.abspath(os.path.join(datadir, os.pardir)) + '/'

#############
# Load data #
#############
sumdata = np.zeros((detector.nb_pixel_y, detector.nb_pixel_x))
mask = np.zeros((detector.nb_pixel_y, detector.nb_pixel_x))

for idx in range(nb_files):
    if is_scan:
        ccdfiletmp = os.path.join(datadir, sample_name + '_' + str('{:05d}'.format(scan)) +
                                  "_data_" + str('{:06d}'.format(image_nb[idx]))+".h5")
    else:
        ccdfiletmp = os.path.join(datadir, sample_name + '_take_' + str('{:05d}'.format(scan)) +
                                  "_data_" + str('{:06d}'.format(image_nb[idx]))+".h5")

    h5file = h5py.File(ccdfiletmp, 'r')
    data = h5file['entry']['data']['data'][:]
    nbz, nby, nbx = data.shape

    if compare_ends and nb_files == 1:
        data_start, _ = pru.mask_eiger4m(data=data[0, :, :], mask=mask)
        data_start[np.log10(data_start) > high_threshold] = 0
        data_start = data_start.astype(float)
        data_stop, _ = pru.mask_eiger4m(data=data[-1, :, :], mask=mask)
        data_stop[np.log10(data_stop) > high_threshold] = 0
        data_stop = data_stop.astype(float)

        fig, _, _ = gu.imshow_plot(data_stop - data_start, plot_colorbar=True, scale='log',
                                   title='difference between the last frame and the first frame of the series')

    data = data.sum(axis=0)  # data becomes 2D
    mask[np.log10(data) > high_threshold] = 1
    data[mask == 1] = 0
    sumdata = sumdata + data
    sys.stdout.write('\rLoading file {:d}'.format(idx+1) + ' / {:d}'.format(nb_files))
    sys.stdout.flush()

print('')
if is_scan:
    if nb_files > 1:
        plot_title = 'masked data - sum of ' + str(nb_files) + ' points with {:d} frames each'.format(nbz)
    else:
        plot_title = 'masked data - sum of ' + str(nbz) + ' frames'
    filename = 'S' + str(scan) + '_scan.png'
else:  # single image
    plot_title = 'masked data'
    filename = 'S' + str(scan) + '_image_' + str(image_nb[idx]) + '.png'

sumdata, mask = pru.mask_eiger4m(data=sumdata, mask=mask)
if save_mask:
    fig, _, _ = gu.imshow_plot(mask, plot_colorbar=False, title='mask')
    np.savez_compressed(savedir+'hotpixels.npz', mask=mask)
    fig.savefig(savedir + 'mask.png')

y0, x0 = np.unravel_index(abs(sumdata).argmax(), sumdata.shape)
print("Max at (y, x): ", y0, x0, ' Max = ', int(sumdata[y0, x0]))

fig, _, _ = gu.imshow_plot(sumdata, plot_colorbar=True, title=plot_title, vmin=0, scale='log')
np.savez_compressed(savedir + 'hotpixels.npz', mask=mask)
fig.savefig(savedir + filename)
plt.show()
