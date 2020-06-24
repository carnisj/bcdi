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

import hdf5plugin # Very important for compression: bitshuffle, lz4
import h5py
import numpy as np
import matplotlib
#matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import os
import sys
# Change here for another processing location 
sys.path.append('/mxn/visitors/dzhigd/NanoMAX_beamtimes/python_scripts/bcdi/')
#import bcdi.graph.graph_utils as gu # graphics
#import bcdi.experiment.experiment_utils as exp # experimental parameters
#import bcdi.preprocessing.preprocessing_utils as pru # preprocessing

# Routing to the data file
beamtime_id = "20200587/2020062408/"
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
savedir = '/mxn/visitors/dzhigd/NanoMAX_beamtimes/24062020/data_processing/'+str('{:06d}'.format(scan))  # images will be saved here, leave it to '' otherwise (default to data directory's parent)
is_scan = True  # set to True is the measurement is a scan or a time series, False for a single image
calculate_degradation = True  # set to True to plot the difference between the last frame and the first frame
save_mask = False  # True to save the mask as 'hotpixels.npz'
##########################
# end of user parameters #
##########################

####################
# Initialize paths #
####################
datadir = rootdir + sample_name

if savedir == '':
    savedir = os.path.abspath(os.path.join(datadir, os.pardir)) + '/'

#############
# Load data #
#############
nb_pixel_y = 515
nb_pixel_x = 515

plt.ion() # interactive graphics
mask = np.zeros((nb_pixel_y, nb_pixel_x))

ccdfiletmp = os.path.join(datadir,'scan_' + str('{:06d}'.format(scan)) + '_' + detector + '.hdf5'
h5file = h5py.File(ccdfiletmp, 'r')
    
data = h5file['entry']['measurement']['Merlin']['data']
nbz, nby, nbx = data.shape

if not roi:
    data = np.array(data) # the whole array is loaded here
else
    data = np.array(data[roi[0]:roi[1],roi[2]:roi[3],roi[4]:roi[5]])

mask[np.log10(data) > high_threshold] = 1
data[mask == 1] = 0

plt.figure(1)
plt.imshow(data[1,:,:])
plt.show()

filename = 'S' + str(scan) + '_scan.npz'
np.savez_compressed(savedir + filename, data=data)

if calculate_degradation:
	R = calculate_data_degradation(data)
	plt.figure(100)
	plt.plot(R)
	plt.show()

def crop_auto(data):
	sum_data = data.sum(2)
	[x,y] = ndimage.measurements.center_of_mass(sum_data)
	rows, cols = sum_data.shape
	w = np.floor(np.min([x,y,rows-x,cols-y]))
	if len(w) == 2:
    	w[0] = np.min([window[0],window[1]])
    	w[1] = np.min([window[0],window[1]])
	data_cropped = data[x-w[0]+1:x+w[0],y-w[1]+1:y+w[1]]
	return  data_cropped

def calculate_data_degradation(data):
	# R function is calculated for each frame with respect to the initial one
	A = data[1,:,:]
	A_mean = A-np.mean(A)
	R = np.zero(nbz-1)
	for ii in range(0,nbz):
		B =  data[ii,:,:]
		B_mean = B-np.mean(B)
		R[ii] = np.divide((np.multiply(A_mean,B_mean).sum(),np.sqrt((np.multiply(np.square(A_mean).sum(),np.square(B_mean).sum())))
		
	return R
