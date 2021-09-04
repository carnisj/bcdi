#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

try:
    import hdf5plugin  # for P10, should be imported before h5py or PyTables
except ModuleNotFoundError:
    pass

import fabio
import h5py
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import RegularGridInterpolator
from silx.io.specfile import SpecFile
import sys
import xrayutilities as xu

matplotlib.use("Qt5Agg")

helptext = """
Merge 3D reciprocal space map together

File structure should be (e.g. scan 583):
data in:        /specdir/S583/data/myscan.nxs
soecfile in:    /specdir/
output files will be saved in:   /specdir/S583/pynx/
"""


scans = [572, 936]  # list of spec scan numbers
###########################
detector = "Eiger2M"  # "Eiger2M" or "Maxipix" or "Eiger4M"
photon_threshold = (
    1  # for noise removal, in detector counts:   data[data <= photon_threshold] = 0
)
debug = 0  # 1 to show summed data and central frame before cropping
centering = (
    0  # Bragg peak determination, 0 for max, 1 for center of mass, 0 is better usually
)
specdir = "C:/Users/carnis/Work Folders/Documents/data/HC3207/ID01/"

setup = "ID01"  # 'ID01' or 'SIXS' or 'CRISTAL' or 'P10',
# used for data loading and normalization by monitor
rocking_angle = "outofplane"  # "outofplane" or "inplane" or "energy"
follow_delta = 0  # for energy_scan, set to 1 if the detector is also scanned
# to follow the Bragg peak (delta @ ID01)
output_size = [300, 500, 500]  # size for the interpolated summed data
comment = "_Scans" + str(scans)  # string, should start with "_"
hotpixels_file = ""  # specdir + 'hotpixels.npz'  #
flatfield_file = specdir + "flatfield_eiger.npz"  #
spec_name = "align2.spec"  # .spec , otherwise .fio for P10
nav = [1, 1]  # reduce data: number of pixels to average in each detector direction
if (
    detector == "Eiger2M"
):  # eiger.y_bragg = 1412  # y pixel of the Bragg peak, only used for Eiger ROI
    nb_pixel_x = 1030  # 1030
    nb_pixel_y = 2164  # 2164  # 1614 now since one quadrant is dead
    x_bragg = 405  # 895  # x pixel of the Bragg peak, only used for Eiger ROI
    # roi = [0, nb_pixel_y, 0, nb_pixel_x]  # 2164 x 1030
    roi = [775, 1710, 0, nb_pixel_x]  # HC3207
    # roi = [552, 803, x_bragg - 200, x_bragg + 200]  # 1060
    pixelsize = 7.5e-05
elif detector == "Maxipix":  # maxipix
    nb_pixel_x = 516  # 516
    nb_pixel_y = 516  # 516
    roi = [0, nb_pixel_y, 0, nb_pixel_x]
    pixelsize = 5.5e-05
elif detector == "Eiger4M":  # eiger 4M ()
    nb_pixel_x = 2070  # 2070
    nb_pixel_y = 2167  # 2167
    # y_bragg = 1412  # y pixel of the Bragg peak, only used for Eiger ROI
    x_bragg = 1373  # x pixel of the Bragg peak, only used for Eiger ROI
    # roi = [1102, 1610, 185, 685]
    roi = [1103, 1615, x_bragg - 256, x_bragg + 256]
    pixelsize = 7.5e-05
else:
    sys.exit("Incorrect value for 'detector' parameter")
nch1 = roi[1] - roi[0]  # 2164 Eiger, 516 Maxipix
nch2 = roi[3] - roi[2]  # 1030 Eiger, 516 Maxipix

######################################################################
# important only if you want to orthogonalize the data before phasing
######################################################################
sdd = 0.86180  # sample to detector distance in m, not important if you use raw data
energy = 9000 - 6  # x-ray energy in eV, not important if you use raw data
offset_inplane = (
    -2.5292
)  # outer detector angle offset, not important if you use raw data
sixs_beta = 0  # incident angle of diffractometer at SIXS
beam_direction = [1, 0, 0]  # beam along x
sample_inplane = [
    1,
    0,
    0,
]  # sample inplane reference direction along the beam at 0 angles
sample_outofplane = [0, 0, 1]  # surface normal of the sample at 0 angles
# geometry of diffractometer
if setup == "ID01":
    offsets = (0, 0, 0, offset_inplane, 0)  # eta chi phi nu del
    qconv = xu.experiment.QConversion(
        ["y-", "x+", "z-"], ["z-", "y-"], r_i=beam_direction
    )  # for ID01
    # 3S+2D goniometer (ID01 goniometer, sample: eta, chi, phi      detector: nu,del
    # the vector beam_direction is giving the direction of the primary beam
    # convention for coordinate system: x downstream; z upwards;
    # y to the "outside" (right-handed)
elif setup == "SIXS":
    offsets = (0, 0, 0, offset_inplane, 0)  # beta, mu, beta, gamma del
    sys.path.append(specdir)
    import nxsReady  # script to load a dictionnary for the HDF5 data file

    qconv = xu.experiment.QConversion(
        ["y-", "z+"], ["y-", "z+", "y-"], r_i=beam_direction
    )  # for SIXS
    # 2S+3D goniometer (SIXS goniometer, sample: beta, mu     detector: beta, gamma, del
    # beta is below both sample and detector circles
    # the vector is giving the direction of the primary beam
    # convention for coordinate system: x downstream; z upwards;
    # y to the "outside" (right-handed)
elif setup == "CRISTAL":
    # TODO: adapt to Cristal
    pass
elif setup == "P10":
    # TODO: adapt to P10
    pass
else:
    print("Setup of " + setup + " not supported!")
    sys.exit()

cch1 = 1272.57 - roi[0]  # 207.88 - roi[0]  #
cch2 = -16.47 - roi[2]  # 50.49 - roi[2]  #
hxrd = xu.experiment.HXRD(
    sample_inplane, sample_outofplane, qconv=qconv
)  # x downstream, y outboard, z vertical
# first two arguments in HXDD are the inplane reference direction along
# the beam and surface normal of the sample
hxrd.Ang2Q.init_area(
    "z-",
    "y+",
    cch1=cch1,
    cch2=-cch2,
    Nch1=nch1,
    Nch2=nch2,
    pwidth1=pixelsize,
    pwidth2=pixelsize,
    distance=sdd,
    detrot=-0.385,
    tiltazimuth=237.2,
    tilt=1.316,
)
# first two arguments in init_area are the direction of the detector,
# checked for ID01 and SIXS
##############################################################################
# parameters for plotting)
params = {
    "backend": "Qt5Agg",
    "axes.labelsize": 20,
    "font.size": 20,
    "legend.fontsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "text.usetex": False,
    "figure.figsize": (11, 9),
}
matplotlib.rcParams.update(params)
# define a colormap
cdict = {
    "red": (
        (0.0, 1.0, 1.0),
        (0.11, 0.0, 0.0),
        (0.36, 0.0, 0.0),
        (0.62, 1.0, 1.0),
        (0.87, 1.0, 1.0),
        (1.0, 0.0, 0.0),
    ),
    "green": (
        (0.0, 1.0, 1.0),
        (0.11, 0.0, 0.0),
        (0.36, 1.0, 1.0),
        (0.62, 1.0, 1.0),
        (0.87, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    ),
    "blue": (
        (0.0, 1.0, 1.0),
        (0.11, 1.0, 1.0),
        (0.36, 1.0, 1.0),
        (0.62, 0.0, 0.0),
        (0.87, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    ),
}
my_cmap = LinearSegmentedColormap("my_colormap", cdict, 256)


def remove_hotpixels(mydata, hotpixels, mymask):
    """Remove hot pixels from CCD frames."""
    mydata[hotpixels == -1] = 0
    mymask[hotpixels == -1] = 1
    return mydata, mymask


def check_pixels(mydata, mymask, var_threshold=5, debugging=0):
    """
    Check for hot pixels in the data.

    :param mydata: detector 3d data
    :param mymask: 2d mask
    :param var_threshold: pixels with 1/var > var_threshold*1/var.mean() will be masked
    :param debugging: to see plots before and after
    return mydata and mymask updated
    """
    numz, numy, numx = mydata.shape
    meandata = mydata.mean(axis=0)  # 2D
    vardata = 1 / mydata.var(axis=0)  # 2D

    if debugging == 1:
        plt.figure(figsize=(18, 15))
        plt.subplot(1, 2, 1)
        plt.imshow(meandata, vmin=0, vmax=1)
        plt.colorbar()
        plt.axis("scaled")
        plt.title("Mean of data along axis=0\nbefore masking")
        plt.subplot(1, 2, 2)
        plt.imshow(vardata, vmin=0)
        plt.colorbar()
        plt.title("1/variance of data along axis=0\nbefore masking")
        plt.axis("scaled")
        plt.pause(0.1)
    # TODO: check with RMS of amplitude
    var_mean = vardata[vardata != np.inf].mean()
    vardata[meandata == 0] = var_mean  # pixels were data=0 (hence 1/variance=inf)
    # are set to the mean of 1/var
    indices_badpixels = np.nonzero(
        vardata > var_mean * var_threshold
    )  # isolate constants pixels != 0 (1/variance=inf)
    mymask[indices_badpixels] = 1  # mymask is 2D
    for index in range(numz):
        tempdata = mydata[index, :, :]
        tempdata[indices_badpixels] = 0
        mydata[index, :, :] = tempdata

    if debugging == 1:
        meandata = mydata.mean(axis=0)
        vardata = 1 / mydata.var(axis=0)
        plt.figure(figsize=(18, 15))
        plt.subplot(1, 2, 1)
        plt.imshow(meandata, vmin=0, vmax=1)
        plt.colorbar()
        plt.axis("scaled")
        plt.title("Mean of data along axis=0\nafter masking")
        plt.subplot(1, 2, 2)
        plt.imshow(vardata, vmin=0)
        plt.colorbar()
        plt.title("Variance of data along axis=0\nafter masking")
        plt.axis("scaled")
        plt.pause(0.1)
    # print(str(indices_badpixels[0].shape[0]),
    # "badpixels with 1/var>", str(var_threshold),
    #       '*1/var.mean() were masked on a total of', str(numx*numy))
    print(
        str(indices_badpixels[0].shape[0]),
        "badpixels were masked on a total of",
        str(numx * numy),
    )
    return mydata, mymask


def mask_eiger(mydata, mymask):
    """Mask the Eiger2M frames."""
    mydata[:, 255:259] = 0
    mydata[:, 513:517] = 0
    mydata[:, 771:775] = 0
    mydata[0:257, 72:80] = 0
    mydata[255:259, :] = 0
    mydata[511:552, :0] = 0
    mydata[804:809, :] = 0
    mydata[1061:1102, :] = 0
    mydata[1355:1359, :] = 0
    mydata[1611:1652, :] = 0
    mydata[1905:1909, :] = 0
    mydata[1248:1290, 478] = 0
    mydata[1214:1298, 481] = 0
    mydata[1649:1910, 620:628] = 0

    mymask[:, 255:259] = 1
    mymask[:, 513:517] = 1
    mymask[:, 771:775] = 1
    mymask[0:257, 72:80] = 1
    mymask[255:259, :] = 1
    mymask[511:552, :] = 1
    mymask[804:809, :] = 1
    mymask[1061:1102, :] = 1
    mymask[1355:1359, :] = 1
    mymask[1611:1652, :] = 1
    mymask[1905:1909, :] = 1
    mymask[1248:1290, 478] = 1
    mymask[1214:1298, 481] = 1
    mymask[1649:1910, 620:628] = 1
    return mydata, mymask


def mask_maxipix(mydata, mymask):
    """Mask the Maxipix frames."""
    mydata[:, 255:261] = 0
    mydata[255:261, :] = 0

    mymask[:, 255:261] = 1
    mymask[255:261, :] = 1
    return mydata, mymask


def mask_eiger4m(mydata, mymask):
    """Mask the Eiger4M frames."""
    return mydata, mymask


def gridmap(
    specfile,
    scan_nb,
    mydetector,
    region=None,
    myflatfield=None,
    myhotpixels=None,
    reload=0,
    previous_data=None,
    previous_mask=None,
    mysetup="ID01",
    myrocking_angle="outofplane",
    follow_bragg=0,
    myenergy=None,
    myoffsets=(0, 0, 0, 0, 0),
):
    """
    Load the data, check for saturated pixels, interpolate on an orthogonal grid.

    :param specfile:
    :param scan_nb:
    :param mydetector: "Eiger4M" or "Eiger2M" or "Maxipix"
    :param region: roi on the detector
    :param myflatfield: 2D array, flatfield correction for the detector
    :param myhotpixels: 2D array, detector hotpixels to be masked
    :param reload: 1 when you reload the data
    :param previous_data: when you reload the data
    :param previous_mask: when you reload the data
    :param mysetup: 'ID01' or 'SIXS' or 'CRISTAL', different data loading method
    :param myrocking_angle: name of the motor which is tilted during the rocking curve
    :param follow_bragg: for energy_scan, set to 1 if the detector is scanned to
     follow the Bragg peak (delta @ ID01)
    :param myenergy: energy in eV of the experiment, in case it is not in the spec file
    :param myoffsets: sample and detector offsets for xrayutilities
    :return:
    """
    global sixs_beta, nb_pixel_x, nb_pixel_y
    if region is None:
        if mydetector == "Eiger2M":
            region = [0, nb_pixel_y, 0, nb_pixel_x]
        elif mydetector == "Maxipix":
            region = [0, nb_pixel_y, 0, nb_pixel_x]
        elif mydetector == "Eiger4M":
            region = [0, nb_pixel_y, 0, nb_pixel_x]
        else:
            region = [0, nb_pixel_y, 0, nb_pixel_x]
    if mysetup == "ID01":
        motor_names = specfile[str(scan_nb) + ".1"].motor_names  # positioners
        motor_positions = specfile[str(scan_nb) + ".1"].motor_positions  # positioners
        labels = specfile[str(scan_nb) + ".1"].labels  # motor scanned
        labels_data = specfile[str(scan_nb) + ".1"].data  # motor scanned
        chi = 0
        delta = motor_positions[motor_names.index("del")]
        nu = motor_positions[motor_names.index("nu")]
        if myrocking_angle == "outofplane":
            eta = labels_data[labels.index("eta"), :]
            phi = motor_positions[motor_names.index("phi")]
            myenergy = motor_positions[motor_names.index("nrj")]  # in kev
        elif myrocking_angle == "inplane":
            phi = labels_data[labels.index("phi"), :]
            eta = motor_positions[motor_names.index("eta")]
            myenergy = motor_positions[motor_names.index("nrj")]  # in kev
        elif myrocking_angle == "energy":
            myenergy = labels_data[labels.index("energy"), :]  # in kev
            if follow_bragg == 1:
                delta = labels_data[labels.index("del"), :]
                # TODO: understand why Qx is positive
            phi = motor_positions[motor_names.index("phi")]
            eta = motor_positions[motor_names.index("eta")]
        else:
            print("Error in rocking angle definition")
            sys.exit()
        myenergy = myenergy * 1000.0  # switch to eV
        if isinstance(myenergy, float) and myenergy < 0:
            print("Energy not correctly defined in spec file, default to 9keV")
            myenergy = 9000.0
    elif mysetup == "SIXS":
        mydataset = nxsReady.DataSet(
            datadir + ccdfiletmp % scan_nb, ccdfiletmp % scan_nb, scan="SBS"
        )
        img = mydataset.mfilm[1:, :, :]  # first frame is duplicated
        delta = mydataset.delta[1:].mean()  # not scanned
        gamma = mydataset.gamma[1:].mean()  # not scanned
        mu = mydataset.mu[1:]
    elif mysetup == "CRISTAL":
        omega = specfile["a"]["scan_data"]["actuator_1_1"][:] / 1e6
        delta = specfile["a/CRISTAL/I06-C-C07__EX__DIF-DELTA__#1/raw_value"][:]
        nu = specfile["a/CRISTAL/I06-C-C07__EX__DIF-GAMMA__#1/raw_value"][:]
        maxpix_img = specfile["a"]["scan_data"]["data_04"][:]
    elif mysetup == "P10":
        # TODO: find motors in specfile
        mu = []
        for index in range(25):  # header
            specfile.readline()
        for index, myline in enumerate(specfile, 0):
            myline = myline.strip()
            mycolumns = myline.split()
            if mycolumns[0] == "!":
                break
            mu.append(mycolumns[0])
        mu = np.asarray(mu, dtype=float)
        nb_img = len(mu)
        specfile.close()

    if reload == 0:
        if mydetector == "Eiger2M":
            counter = "ei2minr"
            mymask = np.zeros((nb_pixel_y, nb_pixel_x))
        elif mydetector == "Maxipix":
            counter = "mpx4inr"
            # counter = 'roi7'
            mymask = np.zeros((nb_pixel_y, nb_pixel_x))
        elif mydetector == "Eiger4M":
            mymask = np.zeros((nb_pixel_y, nb_pixel_x))
        else:
            counter = "mpx4inr"
            mymask = np.zeros((nb_pixel_y, nb_pixel_x))

        if myflatfield is None:
            myflatfield = np.ones(mymask.shape)
        if myhotpixels is None:
            myhotpixels = np.zeros(mymask.shape)
        if mysetup == "ID01":
            ccdn = labels_data[labels.index(counter), :]
            rawdata = np.zeros(
                (len(ccdn), region[1] - region[0], region[3] - region[2])
            )
            for index in range(len(ccdn)):
                i = int(ccdn[index])
                e = fabio.open(ccdfiletmp % i)
                ccdraw = e.data
                ccdraw, mymask = remove_hotpixels(ccdraw, myhotpixels, mymask)
                if mydetector == "Eiger2M":
                    ccdraw, mymask = mask_eiger(ccdraw, mymask)
                elif mydetector == "Maxipix":
                    ccdraw, mymask = mask_maxipix(ccdraw, mymask)
                ccdraw = myflatfield * ccdraw
                ccd = xu.blockAverage2D(ccdraw, nav[0], nav[1], roi=region)
                rawdata[int(i - ccdn[0]), :, :] = ccd
        elif mysetup == "SIXS":
            nb_img = img.shape[0]
            rawdata = np.zeros((nb_img, region[1] - region[0], region[3] - region[2]))
            for index in range(nb_img):
                ccdraw = img[index, :, :]  # first image is duplicated
                ccdraw, mymask = remove_hotpixels(ccdraw, myhotpixels, mymask)
                if mydetector == "Eiger2M":
                    ccdraw, mymask = mask_eiger(ccdraw, mymask)
                elif mydetector == "Maxipix":
                    ccdraw, mymask = mask_maxipix(ccdraw, mymask)
                ccdraw = myflatfield * ccdraw
                ccd = xu.blockAverage2D(ccdraw, nav[0], nav[1], roi=region)
                rawdata[index, :, :] = ccd
        elif mysetup == "CRISTAL":
            nb_img = omega.shape[0]
            rawdata = np.zeros((nb_img, region[1] - region[0], region[3] - region[2]))
            for index in range(nb_img):
                ccdraw = maxpix_img[index, :, :]
                ccdraw, mymask = remove_hotpixels(ccdraw, myhotpixels, mymask)
                ccdraw, mymask = mask_maxipix(ccdraw, mymask)
                ccdraw = myflatfield * ccdraw
                ccd = xu.blockAverage2D(ccdraw, nav[0], nav[1], roi=region)
                rawdata[index, :, :] = ccd
        elif mysetup == "P10":
            rawdata = np.zeros((nb_img, region[1] - region[0], region[3] - region[2]))
            for index in range(nb_img):
                h5file = h5py.File(ccdfiletmp % (index + 1), "r")
                try:
                    ccdraw = h5file["entry"]["data"]["data"][:].sum(axis=0)
                except OSError:
                    print("hdf5plugin is not installed")
                    sys.exit()
                ccdraw, mymask = remove_hotpixels(ccdraw, myhotpixels, mymask)
                ccdraw, mymask = mask_eiger4m(ccdraw, mymask)
                ccdraw = myflatfield * ccdraw
                ccd = xu.blockAverage2D(ccdraw, nav[0], nav[1], roi=region)
                rawdata[index, :, :] = ccd

        mymask = mymask[region[0] : region[1], region[2] : region[3]]
        rawdata, mymask = check_pixels(
            rawdata, mymask, var_threshold=5, debugging=0
        )  # additional check for hotpixels
        numz, numy, numx = rawdata.shape
        rawmask3d = np.repeat(mymask[np.newaxis, :, :], numz, axis=0)
        rawmask3d[np.isnan(rawdata)] = 1
        rawdata[np.isnan(rawdata)] = 0
    else:
        rawmask3d = previous_mask
        rawdata = previous_data
        numz, numy, numx = rawdata.shape
    # transform scan angles to reciprocal space coordinates for all detector pixels
    if mysetup == "ID01":
        myqx, myqy, myqz = hxrd.Ang2Q.area(
            eta, chi, phi, nu, delta, en=myenergy, delta=myoffsets
        )
        mygridder = xu.Gridder3D(numz, numy, numx)
        # convert mask to rectangular grid in reciprocal space
        mygridder(myqx, myqz, myqy, rawmask3d)
        mymask3d = np.copy(mygridder.data)
        # convert data to rectangular grid in reciprocal space
        mygridder(myqx, myqz, myqy, rawdata)
        return (
            mygridder.xaxis,
            mygridder.yaxis,
            mygridder.zaxis,
            rawdata,
            mygridder.data,
            rawmask3d,
            mymask3d,
        )

    if mysetup == "SIXS":
        if myenergy is None:
            print("Defaulting energy to 8.5keV")
            myenergy = 8500
        myqx, myqy, myqz = hxrd.Ang2Q.area(
            sixs_beta, mu, sixs_beta, gamma, delta, en=myenergy, delta=myoffsets
        )
        mygridder = xu.Gridder3D(numz, numy, numx)
        # convert mask to rectangular grid in reciprocal space
        mygridder(myqx, myqz, myqy, rawmask3d)
        mymask3d = np.copy(mygridder.data)
        # convert data to rectangular grid in reciprocal space
        mygridder(myqx, myqz, myqy, rawdata)
        return (
            mygridder.xaxis,
            mygridder.yaxis,
            mygridder.zaxis,
            rawdata,
            mygridder.data,
            rawmask3d,
            mymask3d,
        )

    if mysetup == "CRISTAL":
        # TODO: implement this for CRISTAL
        print("Gridder not yet implemented for CRISTAL setup")
        return 0, 0, 0, rawdata, 0, rawmask3d, 0
    if mysetup == "P10":
        # TODO: implement this for P10
        print("Gridder not yet implemented for P10 setup")
        return 0, 0, 0, rawdata, 0, rawmask3d, 0
    print("Wrong setup")
    sys.exit()


#########################################################
nb_scan = len(scans)
plt.ion()
if flatfield_file != "":
    flatfield = np.load(flatfield_file)
    npz_key = flatfield.keys()
    flatfield = flatfield[npz_key[0]]
    if flatfield.shape[0] > nb_pixel_y:
        flatfield = flatfield[0:nb_pixel_y, :]
else:
    flatfield = None
if hotpixels_file != "":
    hotpix_array = np.load(hotpixels_file)
    npz_key = hotpix_array.keys()
    hotpix_array = hotpix_array[npz_key[0]]
    if len(hotpix_array.shape) == 3:  # 3D array
        hotpix_array = hotpix_array.sum(axis=0)
    hotpix_array[hotpix_array != 0] = -1
    if flatfield.shape[0] > nb_pixel_y:
        flatfield = flatfield[0:nb_pixel_y, :]
else:
    hotpix_array = None


##############################################################
# find the q range for data interpolation
##############################################################
limit_q = []
for idx in range(nb_scan):
    datadir = specdir + "SN" + str(scans[idx]) + "/data/"
    ccdfiletmp = os.path.join(
        datadir, "align_eiger2M_%05d.edf.gz"
    )  # template for the CCD file names
    # ccdfiletmp = os.path.join("data_mpx4_%05d.edf.gz")
    # ID01 template for image name
    # ccdfiletmp = os.path.join(spec_name + "_ascan_mu_%05d.nxs")
    # SIXS template for image name
    # ccdfiletmp = os.path.join(datadir, "mgomega-2018_06_08_19-37-38_418.nxs")
    # Cristal template for image name
    # ccdfiletmp = os.path.join("Sample2371_ref_00079_data_%06d.h5")
    # P10 template for detector filenames
    if setup == "CRISTAL":
        spec_file = h5py.File(ccdfiletmp, "r")
    elif setup == "P10":
        spec_file = open(specdir + "S" + str(scans[idx]) + "/" + spec_name, "r")
    else:
        spec_file = SpecFile(specdir + spec_name)
    qx, qz, qy, _, _, _, _ = gridmap(
        specfile=spec_file,
        scan_nb=scans[idx],
        mydetector=detector,
        region=roi,
        myflatfield=flatfield,
        myhotpixels=hotpix_array,
        mysetup=setup,
        myrocking_angle=rocking_angle,
        follow_bragg=follow_delta,
        myenergy=energy,
        myoffsets=offsets,
    )
    if len(limit_q) > 0:
        limit_q[0] = min(limit_q[0], qx.min())
        limit_q[1] = max(limit_q[1], qx.max())
        limit_q[2] = min(limit_q[2], qz.min())
        limit_q[3] = max(limit_q[3], qz.max())
        limit_q[4] = min(limit_q[4], qy.min())
        limit_q[5] = max(limit_q[5], qy.max())
    else:
        limit_q.extend([qx.min(), qx.max(), qz.min(), qz.max(), qy.min(), qy.max()])

#############################################################################
# interpolate all data on a new grid based on q range
#############################################################################
sum_data = np.zeros((output_size[0], output_size[1], output_size[2]), dtype=float)

new_qx, new_qz, new_qy = np.meshgrid(
    np.linspace(limit_q[0], limit_q[1], num=output_size[0], endpoint=True),
    np.linspace(limit_q[2], limit_q[3], num=output_size[1], endpoint=True),
    np.linspace(limit_q[4], limit_q[5], num=output_size[2], endpoint=True),
    indexing="ij",
)
for idx in range(nb_scan):
    print("Scan ", scans[idx])
    datadir = specdir + "SN" + str(scans[idx]) + "/data/"
    ccdfiletmp = os.path.join(
        datadir, "align_eiger2M_%05d.edf.gz"
    )  # template for the CCD file names
    if setup == "CRISTAL":
        spec_file = h5py.File(ccdfiletmp, "r")
    elif setup == "P10":
        spec_file = open(specdir + "S" + str(scans[idx]) + "/" + spec_name, "r")
    else:
        spec_file = SpecFile(specdir + spec_name)
    qx, qz, qy, _, data, _, _ = gridmap(
        specfile=spec_file,
        scan_nb=scans[idx],
        mydetector=detector,
        region=roi,
        myflatfield=flatfield,
        myhotpixels=hotpix_array,
        mysetup=setup,
        myrocking_angle=rocking_angle,
        follow_bragg=follow_delta,
        myenergy=energy,
        myoffsets=offsets,
    )
    data[data < photon_threshold] = 0
    rgi = RegularGridInterpolator(
        (qx, qz, qy), data, method="linear", bounds_error=False, fill_value=0
    )
    new_data = rgi(
        np.concatenate(
            (
                new_qx.reshape((1, new_qx.size)),
                new_qz.reshape((1, new_qz.size)),
                new_qy.reshape((1, new_qy.size)),
            )
        ).transpose()
    )
    sum_data = sum_data + new_data.reshape(
        (output_size[0], output_size[1], output_size[2])
    ).astype(data.dtype)

    plt.figure(figsize=(18, 15))
    plt.subplot(2, 2, 1)
    plt.contourf(qx, qz, np.log10(abs(data.sum(axis=2))).T, 150, cmap=my_cmap, vmin=0)
    plt.colorbar(ticks=[1, 2, 3, 4, 5])
    plt.xlabel(r"Q$_x$ ($1/\AA$)")
    plt.ylabel(r"Q$_z$ ($1/\AA$)")
    plt.axis("scaled")
    plt.title("Sum(" + str(scans[idx]) + ") over Qy")
    plt.subplot(2, 2, 2)
    plt.contourf(qx, qy, np.log10(abs(data.sum(axis=1))).T, 150, cmap=my_cmap, vmin=0)
    plt.xlabel(r"Q$_x$ ($1/\AA$)")
    plt.ylabel(r"Q$_y$ ($1/\AA$)")
    plt.colorbar(ticks=[1, 2, 3, 4, 5])
    plt.axis("scaled")
    plt.title("Sum(" + str(scans[idx]) + ") over Qz")
    plt.subplot(2, 2, 3)
    plt.contourf(qy, qz, np.log10(abs(data.sum(axis=0))), 150, cmap=my_cmap, vmin=0)
    plt.xlabel(r"Q$_y$ ($1/\AA$)")
    plt.ylabel(r"Q$_z$ ($1/\AA$)")
    plt.colorbar(ticks=[1, 2, 3, 4, 5])
    plt.axis("scaled")
    plt.title("Sum(" + str(scans[idx]) + ") over Qx")
    plt.pause(0.1)

plt.figure()
plt.subplot(2, 2, 1)
plt.contourf(
    new_qx[:, 0, 0],
    new_qz[0, :, 0],
    np.log10(abs(sum_data.sum(axis=2))).T,
    150,
    cmap=my_cmap,
    vmin=0,
)
# plt.colorbar(ticks=[1, 2, 3, 4, 5])
plt.xlabel(r"Q$_x$ ($1/\AA$)")
plt.ylabel(r"Q$_z$ ($1/\AA$)")
plt.axis("scaled")
plt.title("Intensities summed over Qy")
plt.subplot(2, 2, 2)
plt.contourf(
    new_qx[:, 0, 0],
    new_qy[0, 0, :],
    np.log10(abs(sum_data.sum(axis=1))).T,
    150,
    cmap=my_cmap,
    vmin=0,
)
plt.xlabel(r"Q$_x$ ($1/\AA$)")
plt.ylabel(r"Q$_y$ ($1/\AA$)")
# plt.colorbar(ticks=[1, 2, 3, 4, 5])
plt.axis("scaled")
plt.title("Intensities summed over Qz")
plt.subplot(2, 2, 3)
plt.contourf(
    new_qy[0, 0, :],
    new_qz[0, :, 0],
    np.log10(abs(sum_data.sum(axis=0))),
    150,
    cmap=my_cmap,
    vmin=0,
    vmax=5.5,
)
plt.xlabel(r"Q$_y$ ($1/\AA$)")
plt.ylabel(r"Q$_z$ ($1/\AA$)")
# plt.colorbar(ticks=[1, 2, 3, 4, 5])
plt.axis("scaled")
plt.title("Intensities summed over Qx")
plt.pause(0.1)
plt.savefig(specdir + "RSM_" + comment + ".png")
plt.ioff()
plt.show()
