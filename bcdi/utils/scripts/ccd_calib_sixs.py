# -*- coding: utf-8 -*-
"""
Script to calibrate the Maxipix detector on SIXS beamline
command for the mesh detector is e.g.
SBS.mesh delta -0.40 0.70 12 gamma -0.40 0.70 12 1  (do not invert motors)
remove first image of ascan gamma
@author: CARNIS
"""
import xrayutilities as xu
import fabio
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.ndimage.measurements import center_of_mass
start_scan = 299
stop_scan = 314
rough_sdd = 1.25  # in m
window_hor = 2  # horizontal half-width of the window drawn around the max of each image
window_ver = 3  # vertical half-width of the window drawn around the max of each image
frames_to_exclude = [7, 24, 39, 56, 71, 88, 103, 120, 135, 152, 167, 184, 199, 216, 231, 248]
# frames in the gap etc..., leave it as [] otherwise
use_rawdata = 0  # 0 to draw a 3*3 square around the COM and mask the rest, 1 to use the raw data
specdir = "E:/backup_data/SIXS/exp/"
savedir = specdir + "S"+str(start_scan)+"det/"
datadir = specdir + "S"+str(start_scan)+"det/data/"
sys.path.append(specdir)
import nxsReady

hotpixels_file = specdir + "hotpixels.npz"
flatfield_file = specdir + "flatfield_8.5kev.npz"
spec_prefix = "align.spec"
ccdfiletmp = os.path.join(spec_prefix + "_ascan_gamma_%05d.nxs")  # template for the CCD file names
nav = [1, 1]  # reduce data: number of pixels to average in each detector direction
detector = 1  # 0 for eiger, 1 for maxipix
if detector == 0:  # eiger
    roi = [0, 2164, 0, 1030]
    pixelsize = 7.5e-05
elif detector == 1:  # maxipix
    roi = [0, 516, 0, 516]
    pixelsize = 5.5e-05
else:
    sys.exit("Incorrect value for 'detector' parameter")
##############################################################


def remove_hotpixels_eiger(mydata, hot_file):  # , mymask):
    f = fabio.open(hot_file)
    hotpixels = f.data
    mydata[hotpixels == -1] = 0
    # mymask[hotpixels == -1] = 1
    return mydata  # , mymask


def remove_hotpixels_maxipix(mydata, mymask, hot_file):
    """
    function to remove hot pixels from CCD frames
    """
    hotpixels = np.load(hot_file)
    npz_keys = hotpixels.keys()
    hotpixels = hotpixels[npz_keys[0]]
    mydata[hotpixels != 0] = 0
    mymask[hotpixels != 0] = 1
    return mydata, mymask


def mask_eiger(mydata):
    mydata[:, 255: 259] = 0
    mydata[:, 513: 517] = 0
    mydata[:, 771: 775] = 0
    mydata[:, 255: 259] = 0
    mydata[0: 257, 72: 80] = 0
    mydata[1650: 1905, 620: 628] = 0
    mydata[255: 259, :] = 0
    mydata[511: 552, :0] = 0
    mydata[804: 809, :] = 0
    mydata[1061: 1102, :] = 0
    mydata[1355: 1359, :] = 0
    mydata[1611: 1652, :] = 0
    mydata[1905: 1909, :] = 0
    return mydata


def mask_maxipix(mydata, mymask):
    mydata[:, 255:261] = 0
    mydata[255:261, :] = 0
    mydata[0:15, 0:150] = 0
    mydata[0:37, 0:40] = 0
    mydata[0:256, 0:14] = 0
    mydata[460:, 0:10] = 0

    mask[:, 255:261] = 1
    mask[255:261, :] = 1
    mask[0:15, 0:150] = 1
    mask[0:37, 0:40] = 1
    mask[0:256, 0:14] = 1
    mask[460:, 0:10] = 1
    return mydata, mymask


###########################################################################
scanlist = np.arange(start_scan, stop_scan + 1, 1)
if flatfield_file != "" and flatfield_file != '':
    flatfield = np.load(flatfield_file)
    npz_key = flatfield.keys()
    flatfield = flatfield[npz_key[0]]
else:
    flatfield = np.ones((roi[1] - roi[0], roi[3] - roi[2]))
mask = np.zeros((516, 516))

# load first scan to get the data size
dataset = nxsReady.DataSet(datadir + ccdfiletmp % start_scan, ccdfiletmp % start_scan, scan="SBS")
img_per_scan = dataset.mfilm[1:, :, :].shape[0]  # first image is repeated
nb_img = img_per_scan * len(scanlist)
raw_gamma = np.zeros(nb_img)
raw_delta = np.zeros(nb_img)

rawdata = np.zeros((nb_img, roi[1] - roi[0], roi[3] - roi[2]))
data = np.zeros((nb_img-len(frames_to_exclude), roi[1] - roi[0], roi[3] - roi[2]))
eta = np.zeros(nb_img-len(frames_to_exclude))
delta = np.zeros(nb_img-len(frames_to_exclude))
gamma = np.zeros(nb_img-len(frames_to_exclude))
sum_data = np.zeros((roi[1] - roi[0], roi[3] - roi[2]))

for index in range(len(scanlist)):
    scan = scanlist[index]
    dataset = nxsReady.DataSet(datadir + ccdfiletmp % scan, ccdfiletmp % scan, scan="SBS")
    rawdata[index*img_per_scan:(index+1)*img_per_scan, :, :] = dataset.mfilm[1:, :, :]  # first image is repeated
    raw_delta[index*img_per_scan:(index+1)*img_per_scan] = dataset.delta[1:]  # first image is repeated
    raw_gamma[index*img_per_scan:(index+1)*img_per_scan] = dataset.gamma[1:]  # first image is repeated

index_offset = 0
for index in range(nb_img):
    if detector == 1:
        rawdata[index, :, :], mask = remove_hotpixels_maxipix(rawdata[index, :, :], mask, hotpixels_file)
        rawdata[index, :, :], mask = mask_maxipix(rawdata[index, :, :], mask)
        flatfield[mask == 1] = 0
        rawdata[index, :, :] = rawdata[index, :, :] * flatfield
        piy, pix = np.unravel_index(rawdata[index, :, :].argmax(), rawdata[index, :, :].shape)

    sum_data = sum_data + rawdata[index, :, :]
    if index not in frames_to_exclude:
        if use_rawdata == 0:
            y0, x0 = center_of_mass(rawdata[index, :, :])
            # data[index - index_offset, int(np.rint(y0))-1:int(np.rint(y0))+2, int(np.rint(x0))-1:int(np.rint(x0))+2]\
            #     = 1000
            data[index - index_offset, piy - window_ver:piy + window_ver + 1, pix - window_hor:pix + window_hor + 1] = \
                rawdata[index, piy - window_ver:piy + window_ver + 1, pix - window_hor:pix + window_hor + 1]
        else:
            data[index - index_offset, :, :] = rawdata[index, :, :]
        delta[index - index_offset] = raw_delta[index]
        gamma[index - index_offset] = raw_gamma[index]
    else:
        index_offset = index_offset + 1
        print("Frame index", str(index), "excluded")
plt.ion()
plt.figure()
plt.imshow(np.log10(sum_data))
plt.title("Sum of all frames")
plt.savefig(savedir + 'sum.png')
plt.pause(0.1)

plt.figure()
plt.imshow(np.log10(data.sum(axis=0)))
plt.title("Sum of all frames: COM")
plt.savefig(savedir + 'COM.png')
plt.pause(0.1)
# call the fit for the detector parameters
# detector arm rotations and primary beam direction need to be given.
# in total 8 parameters are fitted, however the 4 misalignment parameters can
# be fixed they are the detector tilt azimuth, the detector tilt angle, the
# detector rotation around the primary beam and the outer angle offset
param, eps = xu.analysis.sample_align.area_detector_calib(
    gamma, delta, data, ['z-', 'y-'], 'x+', plot=True, start=(pixelsize, pixelsize, rough_sdd, 0, 0, 0, 0),
    fix=(True, True, False, False, False, False, False), plotlog=True, debug=False)
plt.show()