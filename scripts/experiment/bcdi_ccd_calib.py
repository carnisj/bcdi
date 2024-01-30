#!/usr/bin/env python3

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import os
import sys

import fabio
import matplotlib.pyplot as plt
import numpy as np
import silx.io
import xrayutilities as xu
from scipy.ndimage.measurements import center_of_mass

helptext = """
Area detector calibration, based on ESRF ID01 geometry.

The input should be a list of detector meshes.
Meshes at direct beam and Bragg angle can be combined.
The corresponding HKLs have to be provided.

Frames where the peak is truncated (in the gap...) can be excluded.
"""

scan_nb = [299, 47]  # [ , ] list of scans (ideally mesh at direct beam + Bragg peak)

en = 10000 - 6  # x-ray energy in eV, 6eV offset ar ID01
hkls = [
    (0, 0, 0),
    (1, 1, 1),
]  # list of tuples of hkls, for each scan. Put [(0,0,0)] for a mesh in direct beam.
material = xu.materials.Pt

rough_sdd = 0.61629  # in m
detector = 1  # 0 for eiger, 1 for maxipix
specdir = "C:/users/CARNIS/Work Folders/Documents/data/HC3796/PtGrowth/"
savedir = specdir
photon_threshold = 20
setup = "ID01"  # 'ID01' or 'SIXS'
frames_to_exclude = [
    [10, 11, 12, 13, 14],
    [3, 10, 17, 24, 31, 38, 45],
]  # [[list(3 + 7*np.linspace(0, 6, 7, dtype=int))],[]]
#  list of lists of frames to exclude, leave it as [[],[],...] otherwise,
#  there should be as many sublists as scans

use_rawdata = (
    1  # 0 to draw a 3*3 square around the COM and mask the rest, 1 to use the raw data
)
hotpixels_file = ""  # specdir + "hotpixels_HS4670.npz"
flatfield_file = specdir + "flatfield_maxipix_8kev.npz"
spec_prefix = "2018_10_31_073359PtGrowth"
nav = [1, 1]  # reduce data: number of pixels to average in each detector direction

if detector == 0:  # eiger
    roi = [0, 1614, 0, 1030]  # [0, 2164, 0, 1030]  # bottom tile is dead 1614
    counter = "ei2minr"
    pixelsize = 7.5e-05
elif detector == 1:  # maxipix
    roi = [0, 516, 0, 516]
    counter = "mpx4inr"
    pixelsize = 5.5e-05
else:
    sys.exit("Incorrect value for 'detector' parameter")
##############################################################


def remove_hotpixels(mydata, hotpixels):
    """Remove the hotpixels."""
    mydata[hotpixels == -1] = 0
    return mydata


def mask_eiger(mydata):
    """Mask the Eiger2M gaps."""
    mydata[:, 255:259] = 0
    mydata[:, 513:517] = 0
    mydata[:, 771:775] = 0
    mydata[:, 255:259] = 0
    mydata[0:257, 72:80] = 0
    mydata[1650:1905, 620:628] = 0
    mydata[255:259, :] = 0
    mydata[511:552, :0] = 0
    mydata[804:809, :] = 0
    mydata[1061:1102, :] = 0
    mydata[1355:1359, :] = 0
    mydata[1611:1652, :] = 0
    mydata[1905:1909, :] = 0
    return mydata


def mask_maxipix(mydata):
    """Mask the Maxipix gaps."""
    mydata[:, 255:261] = 0
    mydata[255:261, :] = 0
    return mydata


###########################################################################
plt.ion()
nb_scans = len(scan_nb)
print(nb_scans, " scans will be concatenated")
specfile = silx.io.open(specdir + spec_prefix + ".spec")
###########################################
# load spec file to get the number of points
###########################################
nb_points = 0
for idx in range(nb_scans):
    ccdn = specfile[
        "/" + str(scan_nb[idx]) + ".1/measurement/" + counter
    ].value  # ndarray if mesh
    print(
        "Scan",
        scan_nb[idx],
        " : ",
        len(ccdn),
        " frames, ",
        len(frames_to_exclude[idx]),
        " frames to exclude",
    )
    nb_points = nb_points + len(ccdn) - len(frames_to_exclude[idx])

##########################################
# initialize arrays
##########################################
hkl = []
eta = np.zeros(nb_points)
delta = np.zeros(nb_points)
nu = np.zeros(nb_points)
data = np.zeros((nb_points, roi[1] - roi[0], roi[3] - roi[2]))
sum_data = np.zeros((roi[1] - roi[0], roi[3] - roi[2]))
if flatfield_file != "":
    flatfield = np.load(flatfield_file)["flatfield"]
    if flatfield.shape[0] > 1614:
        flatfield = flatfield[0:1614, :]
else:
    flatfield = np.ones((roi[1] - roi[0], roi[3] - roi[2]))

if hotpixels_file != "":
    # f = fabio.open(hot_file)
    # hotpixels = f.data
    hotpix_array = np.load(hotpixels_file)["mask"]
    hotpix_array = hotpix_array.sum(axis=0)
    hotpix_array[hotpix_array != 0] = -1
else:
    hotpix_array = np.zeros((roi[1] - roi[0], roi[3] - roi[2]))
######################################################
# read images and angular positions from the data file
######################################################
# this might differ for data taken at different beamlines since
# they way how motor positions are stored is not always consistent
total_offset = 0
for idx in range(nb_scans):
    datadir = specdir + "S" + str(scan_nb[idx]) + "det/data/"
    raw_eta = specfile[
        "/" + str(scan_nb[idx]) + ".1/instrument/positioners/eta"
    ].value  # float
    raw_delta = specfile[
        "/" + str(scan_nb[idx]) + ".1/measurement/del"
    ].value  # ndarray if mesh
    raw_nu = specfile[
        "/" + str(scan_nb[idx]) + ".1/measurement/nu"
    ].value  # ndarray if mesh
    ccdn = specfile[
        "/" + str(scan_nb[idx]) + ".1/measurement/" + counter
    ].value  # ndarray if mesh
    ccdfiletmp = os.path.join(
        datadir, "data_mpx4_%05d.edf.gz"
    )  # template for the CCD file names
    index_offset = 0
    for index, item in enumerate(ccdn):
        i = int(item)
        e = fabio.open(ccdfiletmp % i)
        rawdata = e.data
        rawdata[rawdata <= photon_threshold] = 0
        if hotpixels_file != "":
            rawdata = remove_hotpixels(data, hotpix_array)
        if detector == 0:
            rawdata = mask_eiger(rawdata)
        elif detector == 1:
            rawdata = mask_maxipix(rawdata)
        rawdata = rawdata * flatfield
        sum_data = sum_data + rawdata

        if index not in frames_to_exclude[idx]:
            if use_rawdata == 0:
                y0, x0 = center_of_mass(rawdata)
                data[
                    total_offset + index - index_offset,
                    int(np.rint(y0)) - 1 : int(np.rint(y0)) + 2,
                    int(np.rint(x0)) - 1 : int(np.rint(x0)) + 2,
                ] = 1000
            else:
                data[total_offset + index - index_offset, :, :] = rawdata
            delta[total_offset + index - index_offset] = raw_delta[index]
            nu[total_offset + index - index_offset] = raw_nu[index]
            eta[total_offset + index - index_offset] = raw_eta
            hkl.append(hkls[idx])
        else:
            index_offset = index_offset + 1
            print("Frame index", str(index), "excluded")
    total_offset = len(ccdn) - index_offset
plt.figure()
plt.imshow(np.log10(data.sum(axis=0)))
plt.title("Sum of all frames: filtered")
plt.savefig(savedir + "S" + str(scan_nb) + "filtered.png")
plt.pause(0.1)

plt.figure()
plt.imshow(np.log10(sum_data))
plt.title("Sum of all raw images")
plt.savefig(savedir + "S" + str(scan_nb) + "raw.png")
plt.pause(0.1)

# call the fit for the detector parameters
# detector arm rotations and primary beam direction need to be given.
# in total 8 parameters are fitted, however the 4 misalignment parameters can
# be fixed they are the detector tilt azimuth, the detector tilt angle, the
# detector rotation around the primary beam and the outer angle offset

###############################################
# version for fitting mesh at direct beam only:
###############################################


################################################################
# version for fitting meshes in direct beam and Bragg condition:
################################################################
# parameters for area_detector_calib: (pwidth1,pwidth2,distance,tiltazimuth,
# tilt,detector_rotation,outerangle_offset,sampletilt,sampletiltazimuth,wavelength)
imgpbcnt = 0
for idx in range(len(eta)):
    if np.all(hkl[idx] != (0, 0, 0)):
        imgpbcnt += 1
if imgpbcnt == 0:
    print("Only data for calibration in direct beam")
    print("Using xu.analysis.sample_align.area_detector_calib()")
    start_variable = (pixelsize, pixelsize, rough_sdd, 0, 0, 0, 0)
    fix_variable = (True, True, False, False, False, False, False)
    param, eps = xu.analysis.sample_align.area_detector_calib(
        nu,
        delta,
        data,
        ["z-", "y-"],
        "x+",
        plot=True,
        start=start_variable,
        fix=fix_variable,
        plotlog=True,
        debug=False,
    )
else:
    print("Data at Bragg peak detected")
    print("Using xu.analysis.area_detector_calib_hkl()")
    wl = 12.398 / (en / 1000)  # wavelength in angstroms
    start_variable = (pixelsize, pixelsize, rough_sdd, 0, 0, 0, 0, 0, 0, wl)
    fix_variable = (True, True, False, False, False, False, False, False, False, False)
    wl = 12.398 / (en / 1000)  # wavelength in angstroms

    beam_direction = [1, 0, 0]  # beam along x
    qconv = xu.experiment.QConversion(
        ["y-"], ["z-", "y-"], r_i=beam_direction
    )  # for ID01
    # parameters for area_detector_calib_hkl: (pwidth1,pwidth2,distance,tiltazimuth,
    # tilt,detector_rotation,outerangle_offset,sampletilt,sampletiltazimuth,wavelength)
    hxrd = xu.HXRD([1, 0, 0], [0, 0, 1], wl=wl, qconv=qconv)
    param, eps = xu.analysis.area_detector_calib_hkl(
        eta,
        nu,
        delta,
        data,
        hkl,
        hxrd,
        material,
        ["z-", "y-"],
        "x+",
        start=start_variable,
        fix=fix_variable,
        plotlog=True,
        debug=False,
    )

plt.ioff()
plt.show()
