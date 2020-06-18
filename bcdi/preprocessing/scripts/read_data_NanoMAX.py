# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr
# 	  Adapted by Dmitry Dzhigaev    ddzhigaev@gmail.com

helptext = """
Open scans or series data at NanoMAX beamline.
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import os
import sys
# Change here for another processing location 
sys.path.append('/mxn/visitors/dzhigd/NanoMAX_beamtimes/python_scripts/bcdi/')
import bcdi.graph.graph_utils as gu # graphics
import bcdi.experiment.experiment_utils as exp # experimental parameters
import bcdi.preprocessing.preprocessing_utils as pru # preprocessing

beamtime_id = "20170093/2018032108/"
scan = 1  # scan number as it appears in the folder name
sample_name = "sample"  # without _ at the end
# rootdir = ":/data/P10_August2019/data/" : offline processing
rootdir = "data/visitors/nanomax/"+beamtime_id+"raw/"

detector = "merlin"    # "merlin" or "Eiger4M" or "pil100k"
# ROI [frameStart, frameEnd, roiYstart, roiYend, roiXstart, roiXend]
roi = []  # plot the integrated intensity in this region of interest. Leave it to [] to use the full detector
# [Vstart, Vstop, Hstart, Hstop]
high_threshold = 9  # data points where log10(data) > high_threshold will be masked
# if data is a series, the condition becomes log10(data.sum(axis=0)) > high_threshold
savedir = '/mxn/visitors/dzhigd/NanoMAX_beamtimes/24062020/data_processing/'+str('{:04d}'.format(scan))  # images will be saved here, leave it to '' otherwise (default to data directory's parent)
is_scan = True  # set to True is the measurement is a scan or a time series, False for a single image
compare_ends = True  # set to True to plot the difference between the last frame and the first frame
save_mask = False  # True to save the mask as 'hotpixels.npz'
##########################
# end of user parameters #
##########################

#######################
# Initialize detector #
#######################
detector = exp.Detector(name=detector)
nb_pixel_y, nb_pixel_x = detector.nb_pixel_y, detector.nb_pixel_x

####################
# Initialize paths #
####################
datadir = rootdir + sample_name

if savedir == '':
    savedir = os.path.abspath(os.path.join(datadir, os.pardir)) + '/'

#############
# Load data #
#############
plt.ion() # interactive graphics
mask = np.zeros((nb_pixel_y, nb_pixel_x))

ccdfiletmp = os.path.join(datadir,'scan_' + str('{:04d}'.format(scan)) + '_' + detector + '0000.hdf5'
h5file = h5py.File(ccdfiletmp, 'r')
    
data = h5file['entry']['measurement']['Merlin']['data']
nbz, nby, nbx = data.shape

if not roi:
    data = np.array(data) # the whole array is loaded here
else
    data = np.array(data[roi[0]:roi[1],roi[2]:roi[3],roi[4]:roi[5]])

mask[np.log10(data) > high_threshold] = 1
data[mask == 1] = 0

print('')
if is_scan: 
    plot_title = 'masked data'
    filename = 'S' + str(scan) + '_scan.npz'

#sumdata, mask = pru.mask_merlin(data=sumdata, mask=mask) NOT IMPLEMENTED
if save_mask:
    fig, _, _ = gu.imshow_plot(mask, plot_colorbar=False, title='mask')
    np.savez_compressed(savedir+'hotpixels.npz', mask=mask)
    fig.savefig(savedir + 'mask.png')

np.savez_compressed(savedir + filename, data=data)

