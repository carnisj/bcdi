#!/usr/bin/env python3

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import xrayutilities as xu
from scipy.ndimage.measurements import center_of_mass

helptext = """
Area detector calibration, based on SOLEIL CRISTAL geometry.

The input should be a list of detector meshes.
Meshes at direct beam and Bragg angle can be combined.
The corresponding HKLs have to be provided.

Frames where the peak is truncated (in the gap...) can be excluded.
"""

scan_nb = [60]  # [ , ] list of scans (ideally mesh at direct beam + Bragg peak)

en = 8300  # x-ray energy in eV, 6eV offset ar ID01
hkls = [
    (0, 0, 0)
]  # list of tuples of hkls, for each scan. Put [(0,0,0)] for a mesh in direct beam.
material = xu.materials.Pt

rough_sdd = 1.0  # in m
detector = 1  # 0 for eiger, 1 for maxipix
specdir = "C:/Users/carnis/Documents/cristal/data/"
savedir = specdir
photon_threshold = 20
setup = "CRISTAL"  #
frames_to_exclude = [
    [
        9,
        10,
        21,
        32,
        88,
        89,
        90,
        91,
        99,
        100,
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        108,
        109,
        110,
        11,
        112,
        113,
        114,
        115,
        116,
        117,
        118,
        119,
        120,
    ]
]  # [[list(3 + 7*np.linspace(0, 6, 7, dtype=int))],[]]
#  list of lists of frames to exclude, leave it as [[],[],...] otherwise,
#  there should be as many sublists as scans

use_rawdata = (
    1  # 0 to draw a 3*3 square around the COM and mask the rest, 1 to use the raw data
)
hotpixels_file = specdir + "hotpixels.npz"  # specdir + "hotpixels_HS4670.npz"
flatfield_file = ""  # specdir + "flatfield_maxipix_8kev.npz"
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
wl = 12.398 / en
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
###########################################
# load spec file to get the number of points
###########################################
nb_points = 0
for idx in range(nb_scans):
    h5file = h5py.File(
        specdir + "S" + str(scan_nb[idx]) + "det/data/S" + str(scan_nb[idx]) + ".nxs",
        "r",
    )
    ccdn = h5file["test_00" + str(scan_nb[idx])]["scan_data"]["data_06"][
        :
    ]  # ndarray if mesh
    print(
        "Scan",
        scan_nb[idx],
        " : ",
        ccdn.shape[0] * ccdn.shape[1],
        " frames, ",
        len(frames_to_exclude[idx]),
        " frames to exclude",
    )
    nb_points = nb_points + ccdn.shape[0] * ccdn.shape[1] - len(frames_to_exclude[idx])

##########################################
# initialize arrays
##########################################
hkl = []
mgomega = np.zeros(nb_points)
delta = np.zeros(nb_points)
gamma = np.zeros(nb_points)
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
    if len(hotpix_array.shape) == 3:
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
    h5file = h5py.File(datadir + "S" + str(scan_nb[idx]) + ".nxs", "r")

    raw_mgomega = (
        h5file["test_" + str(f"{scan_nb[idx]:04d}")]["CRISTAL"][
            "I06-C-C07-EX-MG_OMEGA"
        ]["positon_pre"][:]
        / 1e6
    )
    raw_delta = h5file["test_" + str(f"{scan_nb[idx]:04d}")]["scan_data"][
        "actuator_1_1"
    ][:]
    raw_gamma = h5file["test_" + str(f"{scan_nb[idx]:04d}")]["scan_data"][
        "actuator_2_1"
    ][:]
    ccdn = h5file["test_" + str(f"{scan_nb[idx]:04d}")]["scan_data"]["data_06"][
        :
    ]  # ndarray if mesh
    ccdn = np.reshape(
        ccdn, (ccdn.shape[0] * ccdn.shape[1], ccdn.shape[2], ccdn.shape[3])
    )
    raw_delta = np.reshape(raw_delta, raw_delta.shape[0] * raw_delta.shape[1])
    raw_gamma = np.repeat(raw_gamma, raw_delta.shape[0] / raw_gamma.shape[0], axis=0)
    index_offset = 0
    for index in range(ccdn.shape[0]):
        rawdata = ccdn[index]
        rawdata[rawdata <= photon_threshold] = 0
        if hotpixels_file != "":
            rawdata = remove_hotpixels(rawdata, hotpix_array)
        if detector == 0:
            rawdata = mask_eiger(rawdata)
        elif detector == 1:
            rawdata = mask_maxipix(rawdata)
        rawdata = rawdata * flatfield
        sum_data = sum_data + rawdata

        if index not in frames_to_exclude[idx]:
            print(total_offset + index - index_offset)
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
            gamma[total_offset + index - index_offset] = raw_gamma[index]
            mgomega[total_offset + index - index_offset] = raw_mgomega
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
for idx in range(len(mgomega)):
    if np.all(hkl[idx] != (0, 0, 0)):
        imgpbcnt += 1
if imgpbcnt == 0:
    print("Only data for calibration in direct beam")
    print("Using xu.analysis.sample_align.area_detector_calib()")
    start_variable = (pixelsize, pixelsize, rough_sdd, 0, 0, 0, 0)
    fix_variable = (True, True, False, False, False, False, False)
    param, eps = xu.analysis.sample_align.area_detector_calib(
        gamma,
        delta,
        data,
        ["z+", "y-"],
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
    start_variable = (pixelsize, pixelsize, rough_sdd, 0, 0, 0, 0, 0, 0, wl)
    fix_variable = (True, True, False, False, False, False, False, False, False, False)
    wl = 12.398 / (en / 1000)  # wavelength in angstroms
    beam_direction = [1, 0, 0]  # beam along x
    qconv = xu.experiment.QConversion(
        ["y-"], ["z+", "y-"], r_i=beam_direction
    )  # for ID01
    # parameters for area_detector_calib_hkl: (pwidth1,pwidth2,distance,tiltazimuth,
    # tilt,detector_rotation,outerangle_offset,sampletilt,sampletiltazimuth,wavelength)
    hxrd = xu.HXRD([1, 0, 0], [0, 0, 1], wl=wl, qconv=qconv)
    param, eps = xu.analysis.area_detector_calib_hkl(
        mgomega,
        gamma,
        delta,
        data,
        hkl,
        hxrd,
        material,
        ["z+", "y-"],
        "x+",
        start=start_variable,
        fix=fix_variable,
        plotlog=True,
        debug=False,
    )

plt.ioff()
plt.show()
