# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#       authors:
#         Jerome Carnis, jerome.carnis@esrf.fr

helptext = """
xrutils_polarplot.py
Stereographic projection of diffraction pattern, based on ESRF/ID01 geometry
Before interpolation lower z pixel is higher qz
After interpolation higher z pixel is higher qz
For x and y, higher pixel means higher q
In arrays, when plotting the first parameter is the row (vertical axis) and the second the column (horizontal axis)
Therefore the data structure is data[qx, qz, qy]
Hence the gridder is mygridder(myqx, myqz, myqy, rawdata)
And qx, qz, qy = mygridder.xaxis, mygridder.yaxis, mygridder.zaxis
@author: CARNIS
"""
import numpy as np
import xrayutilities as xu
import os
import matplotlib
import scipy.io  # for savemat
import scipy.signal  # for medfilt2d
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import sys
import fabio
from matplotlib.colors import LinearSegmentedColormap
from silx.io.specfile import SpecFile
import tkinter as tk
from tkinter import filedialog
from numpy.fft import fftn, fftshift

scan = 2606    # spec scan number
flag_medianfilter = 0  # set to 1 for applying med2filter [3,3]
comment = ""

filtered_data = 0  # set to 1 if the data is already a 3D array, 0 otherwise
# Should be the same shape as in specfile, before orthogonalization

reconstructed_data = 0  # set to 1 if the data is a BCDI reconstruction (real space), 0 otherwise
# the reconstruction should be in the crystal orthogonal frame
threshold_amp = 0.25  # threshold for support determination from amplitude, if reconstructed_data=1
use_phase = 0  # set to 0 to use only a support, 1 to use the compex amplitude
voxel_size = 5  # in nm, voxel size of the CDI reconstruction, should be equal in all directions.  Put 0 if unknown
photon_nb = 5e7  # total number of photons in the diffraction pattern calculated from CDI reconstruction
pad_size = 3  # int >= 1, will pad to get this number times the initial array size  (avoid aliasing)

flag_savedata = 0      # set to 1 to save data
flag_plotplanes = 1    # plot red dotted circle with plane index
debug = 1  # 1 to show more plots, 0 otherwise
sdd = 0.61681  # sample to detector distance in m
en = 9994     # x-ray energy in eV
offset_eta = 0  # positive make diff pattern rotate counter-clockwise (eta rotation around Qy)
# will shift peaks rightwards in the pole figure
offset_phi = -4     # positive make diff pattern rotate clockwise (phi rotation around Qz)
# will rotate peaks counterclockwise in the pole figure
offset_chi = 3  # positive make diff pattern rotate clockwise (chi rotation around Qx)
# will shift peaks upwards in the pole figure
offset = 2.9954   # outer detector angle offset (nu)
threshold = 1  # photon threshold in detector counts
radius_mean = 0.028  # q from Bragg peak
dr = 0.0005        # delta_q
reflection = np.array([1, 1, 1])  # np.array([0, 0, 2])  #   # reflection measured
# specdir = "C:/Users/carnis/Work Folders/Documents/data/CH4760_Pt/"
specdir = "C:/users/CARNIS/Work Folders/Documents/data/HC3796/OER/"
datadir = specdir + "S"+str(scan)+"/data/"
savedir = specdir + "S"+str(scan)+"/"
spec_prefix = "2018_11_01_022929_OER"  #
detector = 1    # 0 for eiger, 1 for maxipix
if detector == 0:  # eiger.y_bragg = 1412  # y pixel of the Bragg peak, only used for Eiger ROI
    x_bragg = 430  # x pixel of the Bragg peak, only used for Eiger ROI
    roi = [1102, 1610, x_bragg - 300, x_bragg + 301]
    pixelsize = 7.5e-05
    ccdfiletmp = os.path.join(datadir, "align_eiger2M_%05d.edf.gz")   # template for the CCD file names
elif detector == 1:  # maxipix
    roi = [0, 516, 0, 516]  # [261, 516, 261, 516]  # [0, 516, 0, 516]
    pixelsize = 5.5e-05
    ccdfiletmp = os.path.join(datadir, "data_mpx4_%05d.edf.gz")   # template for the CCD file names
else:
    sys.exit("Incorrect value for 'detector' parameter")
comment = ''
hotpixels_file = ""  # specdir + "hotpixels_HS4670.npz"  # specdir + "align_eiger2M_02694_limahotmask.edf"
flatfield_file = ""  # specdir + "flatfield_maxipix_8kev.npz"  # "flatfield_eiger.npz"
nch1 = roi[1] - roi[0]  # 2164 Eiger, 516 Maxipix
nch2 = roi[3] - roi[2]  # 1030 Eiger, 516 Maxipix
# geometry of diffractometer
qconv = xu.experiment.QConversion(['y-', 'x+', 'z-'], ['z-', 'y-'], [1, 0, 0])
# 3S+2D goniometer (simplified ID01 goniometer, sample: eta, chi, phi      detector: nu,del
# convention for coordinate system: x downstream; z upwards; y to the "outside" (righthanded)
cch1 = 180.66 - roi[0]  # direct_beam_y - roi[0]
cch2 = 995.44 - roi[2]  # direct_beam_x - roi[2]
hxrd = xu.experiment.HXRD([1, 0, 0], [0, 0, 1], en=en, qconv=qconv)
# detector should be calibrated for the same roi as defined above
hxrd.Ang2Q.init_area('z-', 'y+', cch1=cch1, cch2=cch2, Nch1=nch1, Nch2=nch2, pwidth1=pixelsize, pwidth2=pixelsize,
                     distance=sdd, detrot=-0.744, tiltazimuth=67.7, tilt=4.507)
nav = [1, 1]  # reduce data: number of pixels to average in each detector direction
##############################################################################
##############################################################################
# parameters for plotting
params = {'backend': 'Qt5Agg',
          'axes.labelsize': 20,
          'font.size': 20,
          'legend.fontsize': 20,
          'axes.titlesize': 20,
          'xtick.labelsize': 20,
          'ytick.labelsize': 20,
          'text.usetex': False,
          'figure.figsize': (11, 9)}
matplotlib.rcParams.update(params)
# define a colormap
cdict = {'red':  ((0.0, 1.0, 1.0),
                  (0.11, 0.0, 0.0),
                  (0.36, 0.0, 0.0),
                  (0.62, 1.0, 1.0),
                  (0.87, 1.0, 1.0),
                  (1.0, 0.0, 0.0)),
         'green': ((0.0, 1.0, 1.0),
                   (0.11, 0.0, 0.0),
                   (0.36, 1.0, 1.0),
                   (0.62, 1.0, 1.0),
                   (0.87, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),
         'blue': ((0.0, 1.0, 1.0),
                  (0.11, 1.0, 1.0),
                  (0.36, 1.0, 1.0),
                  (0.62, 0.0, 0.0),
                  (0.87, 0.0, 0.0),
                  (1.0, 0.0, 0.0))}
my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 256)


def remove_hotpixels(mydata, hotpixels, mymask):
    """
    function to remove hot pixels from CCD frames
    """
    mydata[hotpixels == -1] = 0
    mymask[hotpixels == -1] = 1
    return mydata, mymask


def check_pixels(mydata, mymask):
    """
    function to check for hot pixels in the data
    """
    numz, _, _ = mydata.shape
    sumdata = mydata.sum(axis=0)
    sumdata[sumdata > 1e6*numz/20] = 0
    print("Mask points number before hot pixels checking: ", int(round(mymask.sum())))
    mymask[sumdata > 1e6 * numz / 20] = 1
    print("Mask points number after hot pixels checking: ", int(round(mymask.sum())))
    for indx in range(numz):
        temp = mydata[indx, :, :]
        temp[mymask == 1] = 0
        mydata[indx, :, :] = temp
    return mydata, mymask


def mask_eiger(mydata, mymask):
    mydata[:, 255: 259] = 0
    mydata[:, 513: 517] = 0
    mydata[:, 771: 775] = 0
    mydata[0: 257, 72: 80] = 0
    mydata[255: 259, :] = 0
    mydata[511: 552, :0] = 0
    mydata[804: 809, :] = 0
    mydata[1061: 1102, :] = 0
    mydata[1355: 1359, :] = 0
    mydata[1611: 1652, :] = 0
    mydata[1905: 1909, :] = 0
    mydata[1248:1290, 478] = 0
    mydata[1214:1298, 481] = 0
    mydata[1649:1910, 620:628] = 0

    mymask[:, 255: 259] = 1
    mymask[:, 513: 517] = 1
    mymask[:, 771: 775] = 1
    mymask[0: 257, 72: 80] = 1
    mymask[255: 259, :] = 1
    mymask[511: 552, :] = 1
    mymask[804: 809, :] = 1
    mymask[1061: 1102, :] = 1
    mymask[1355: 1359, :] = 1
    mymask[1611: 1652, :] = 1
    mymask[1905: 1909, :] = 1
    mymask[1248:1290, 478] = 1
    mymask[1214:1298, 481] = 1
    mymask[1649:1910, 620:628] = 1
    return mydata, mymask


def mask_maxipix(mydata, mymask):
    mydata[:, 255:261] = 0
    mydata[255:261, :] = 0

    mymask[:, 255:261] = 1
    mymask[255:261, :] = 1
    return mydata, mymask


def gridmap(specfile, scan_nb, mydetector, region=None, myflatfield=None, myhotpixels=""):
    global offset, offset_chi, offset_eta, offset_phi, filtered_data, datadir
    if region is None:
        if mydetector == 0:
            region = [0, 2164, 0, 1030]
        elif mydetector == 1:
            region = [0, 516, 0, 516]
    if mydetector == 0:
        counter = 'ei2minr'
        mymask = np.zeros((2164, 1030))
    elif mydetector == 1:
        counter = 'mpx4inr'
        mymask = np.zeros((516, 516))
    else:
        sys.exit("Incorrect value for 'mydetector' parameter")
    if myhotpixels != "":
        # f = fabio.open(hot_file)
        # hotpixels = f.data
        hotpix_array = np.load(myhotpixels)['mask']
        hotpix_array = hotpix_array.sum(axis=0)
        hotpix_array[hotpix_array != 0] = -1
    if myflatfield is None:
        myflatfield = np.ones(mymask.shape)
    motor_names = specfile[str(scan_nb) + '.1'].motor_names  # positioners
    motor_positions = specfile[str(scan_nb) + '.1'].motor_positions  # positioners
    labels = specfile[str(scan_nb) + '.1'].labels  # motor scanned
    labels_data = specfile[str(scan_nb) + '.1'].data  # motor scanned
    chi = offset_chi
    phi = offset_phi
    delta = motor_positions[motor_names.index('del')]
    nu = motor_positions[motor_names.index('nu')]
    eta = labels_data[labels.index('eta'), :] + offset_eta
    ccdn = labels_data[labels.index(counter), :]
    if filtered_data == 0:
        rawdata = np.zeros((len(ccdn), region[1] - region[0], region[3] - region[2]))
        for index in range(len(ccdn)):
            i = int(ccdn[index])
            e = fabio.open(ccdfiletmp % i)
            ccdraw = e.data
            if myhotpixels != "":
                ccdraw, mymask = remove_hotpixels(ccdraw, hotpix_array, mymask)
            if mydetector == 0:
                ccdraw, mymask = mask_eiger(ccdraw, mymask)
            elif mydetector == 1:
                ccdraw, mymask = mask_maxipix(ccdraw, mymask)
            ccdraw = myflatfield * ccdraw
            ccd = xu.blockAverage2D(ccdraw, nav[0], nav[1], roi=region)
            rawdata[int(i - ccdn[0]), :, :] = ccd
    else:
        myfile_path = filedialog.askopenfilename(initialdir=savedir + "pynxraw/",
                                                 title="Select 3D data", filetypes=[("NPZ", "*.npz")])
        rawdata = np.load(myfile_path)['data']
        rawdata = rawdata[region[0]:region[1], region[2]:region[3]]
    mymask = mymask[region[0]:region[1], region[2]:region[3]]
    numz, numy, numx = rawdata.shape
    if numz != len(ccdn):
        print('Filtered data has not the same shape as raw data')
        sys.exit()
    rawmask3d = np.zeros((numz, region[1] - region[0], region[3] - region[2]))
    for indx in range(numz):
        rawmask3d[indx, :, :] = mymask
    # transform scan angles to reciprocal space coordinates for all detector pixels
    myqx, myqy, myqz = hxrd.Ang2Q.area(eta, chi, phi, nu, delta, delta=(0, 0, 0, offset, 0))
    mygridder = xu.Gridder3D(numz, numy, numx)
    # convert mask to rectangular grid in reciprocal space
    mygridder(myqx, myqz, myqy, rawmask3d)
    mymask3d = np.copy(mygridder.data)
    # convert data to rectangular grid in reciprocal space
    mygridder(myqx, myqz, myqy, rawdata)
    return mygridder.xaxis, mygridder.yaxis, mygridder.zaxis, rawdata, mygridder.data, rawmask3d, mymask3d


def plane_angle(ref_plane, plane):
    """
    Calculate the angle between two crystallographic planes in cubic materials
    :param ref_plane: measured reflection
    :param plane: plane for which angle should be calculated
    :return: the angle in degrees
    """
    if np.array_equal(ref_plane, plane):
        angle = 0.0
    else:
        angle = 180/np.pi*np.arccos(sum(np.multiply(ref_plane, plane)) /
                                    (np.linalg.norm(ref_plane)*np.linalg.norm(plane)))
    if angle > 90.0:
        angle = 180.0 - angle
    return angle


###################################################################################
plt.ion()
root = tk.Tk()
root.withdraw()

if reconstructed_data == 0:
    comment = comment + "_diffpattern"
    if flatfield_file != "":
        flatfield = np.load(flatfield_file)['flatfield']
    else:
        flatfield = None
    spec_file = SpecFile(specdir + spec_prefix + ".spec")
    qx, qz, qy, intensity, data, _, _ = gridmap(spec_file, scan, detector, roi, flatfield, hotpixels_file)
else:
    comment = comment + "_CDI"
    file_path = filedialog.askopenfilename(initialdir=savedir + "pynxraw/",
                                           title="Select 3D data", filetypes=[("NPZ", "*.npz")])
    amp = np.load(file_path)['amp']
    nz, ny, nx = amp.shape  # nexus convention
    print('CDI data shape', amp.shape)
    nz1, ny1, nx1 = [value * pad_size for value in amp.shape]
    if use_phase == 1:
        comment = comment + "_complex"
        phase = np.load(file_path)['phase']
        obj = amp * np.exp(1j * phase)
        newobj = np.zeros((nz1, ny1, nx1), dtype=complex)
    else:
        comment = comment + "_support"
        obj = np.zeros(amp.shape)
        obj[amp > threshold_amp] = 1  # obj is the support
        newobj = np.zeros((nz1, ny1, nx1), dtype=float)
    # pad array to avoid aliasing
    newobj[(nz1 - nz) // 2:(nz1 + nz) // 2, (ny1 - ny) // 2:(ny1 + ny) // 2, (nx1 - nx) // 2:(nx1 + nx) // 2] = obj
    # calculate the diffraction pattern from the support only
    data = fftshift(abs(fftn(newobj)) ** 2)
    voxel_size = voxel_size * 10  # conversion in angstroms
    if voxel_size <= 0:
        print('Using arbitraty voxel size of 1 nm')
        voxel_size = 10  # angstroms
    dqx = 2 * np.pi / (voxel_size * nz1)
    dqy = 2 * np.pi / (voxel_size * nx1)
    dqz = 2 * np.pi / (voxel_size * ny1)
    print('dqx', str('{:.5f}'.format(dqx)), 'dqy', str('{:.5f}'.format(dqy)), 'dqz', str('{:.5f}'.format(dqz)))

    # crop array to initial size
    # data = np.zeros(support.shape)
    # data = intensity[(nz1 - nz) // 2:(nz1 + nz) // 2, (ny1 - ny) // 2:(ny1 + ny) // 2, (nx1 - nx) // 2:(nx1 + nx) // 2]

    data = data / abs(data).sum() * photon_nb  # convert into photon number
    # create qx, qy, qz vectors
    nz, ny, nx = data.shape
    qx = np.arange(-nz//2, nz/2) * dqx
    qy = np.arange(-nx//2, nx//2) * dqy
    qz = np.arange(-ny//2, ny//2) * dqz

nz, ny, nx = data.shape  # nexus convention
if flag_medianfilter == 1:  # apply some noise filtering
    for idx in range(nz):
        data[idx, :, :] = scipy.signal.medfilt2d(data[idx, :, :], [3, 3])
if flag_savedata == 1:
    if reconstructed_data == 0:
        np.savez_compressed(savedir+'S'+str(scan)+'_stack', data=intensity)
    # save to .mat, x becomes z for Matlab phasing code
    scipy.io.savemat(savedir+'S'+str(scan)+'_stack.mat', {'data': np.moveaxis(intensity, [0, 1, 2], [-1, -3, -2])})
    np.savez_compressed(savedir+'S'+str(scan)+'_ortho_diffpattern', data=data)

##################################
# define the center of the sphere
##################################
intensity = data
intensity[intensity <= threshold] = 0   # photon threshold

qzCOM = 1/intensity.sum()*(qz*intensity.sum(axis=0).sum(axis=1)).sum()  # COM in qz
qyCOM = 1/intensity.sum()*(qy*intensity.sum(axis=0).sum(axis=0)).sum()  # COM in qy
qxCOM = 1/intensity.sum()*(qx*intensity.sum(axis=1).sum(axis=1)).sum()  # COM in qx
print("Center of mass [qx, qy, qz]: [",
      str('{:.2f}'.format(qxCOM)), str('{:.2f}'.format(qyCOM)), str('{:.2f}'.format(qzCOM)), ']')
###################################
# select the half sphere
###################################
qz_offset = -0.000
# take only the upper part of the sphere
intensity_top = data[:, np.where(qz > (qzCOM+qz_offset))[0].min():np.where(qz > (qzCOM+qz_offset))[0].max(), :]
qz_top = qz[np.where(qz > (qzCOM+qz_offset))[0].min():np.where(qz > (qzCOM+qz_offset))[0].max()]-qz_offset

# take only the lower part of the sphere
intensity_bottom = data[:, np.where(qz < (qzCOM+qz_offset))[0].min():np.where(qz < (qzCOM+qz_offset))[0].max(), :]
qz_bottom = qz[np.where(qz < (qzCOM+qz_offset))[0].min():np.where(qz < (qzCOM+qz_offset))[0].max()]-qz_offset

###################################
# create a 3D array of distances in q from COM
###################################
qx1 = qx[:, np.newaxis, np.newaxis]  # broadcast array
qy1 = qy[np.newaxis, np.newaxis, :]  # broadcast array
qz1_top = qz_top[np.newaxis, :, np.newaxis]   # broadcast array
qz1_bottom = qz_bottom[np.newaxis, :, np.newaxis]   # broadcast array
distances_top = np.sqrt((qx1 - qxCOM)**2 + (qy1 - qyCOM)**2 + (qz1_top - (qzCOM+qz_offset))**2)
distances_bottom = np.sqrt((qx1 - qxCOM)**2 + (qy1 - qyCOM)**2 + (qz1_bottom - (qzCOM+qz_offset))**2)
# The shape of volume_R is (qx1 qy1 qz1)

###################################
# define matrix of radii radius_mean
###################################
# mask_top = np.zeros(np.shape(intensity_top))
# mask_bottom = np.zeros(np.shape(intensity_bottom))
# mask_top[np.where((volume_R_top < (radius_mean+dr)) & (volume_R_top > (radius_mean-dr)))] = 1
mask_top = np.logical_and((distances_top < (radius_mean+dr)), (distances_top > (radius_mean-dr)))
mask_bottom = np.logical_and((distances_bottom < (radius_mean+dr)), (distances_bottom > (radius_mean-dr)))

############################
# plot 2D maps
############################
fig, ax = plt.subplots(num=1, figsize=(20, 15), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(2, 2, 1)
plt.contourf(qz, qx, xu.maplog(data.sum(axis=2)), 150, cmap=my_cmap)
plt.plot([min(qz), max(qz)], [qxCOM, qxCOM], color='k', linestyle='-', linewidth=2)
plt.plot([qzCOM, qzCOM], [min(qx), max(qx)], color='k', linestyle='-', linewidth=2)
circle = plt.Circle((qzCOM, qxCOM), radius_mean + dr, color='0', fill=False, linestyle='dotted')
fig.gca().add_artist(circle)
circle = plt.Circle((qzCOM, qxCOM), radius_mean - dr, color='0', fill=False, linestyle='dotted')
fig.gca().add_artist(circle)
plt.colorbar()
plt.xlabel(r"Q$_z$ ($1/\AA$)")
plt.ylabel(r"Q$_x$ ($1/\AA$)")
plt.axis('scaled')
plt.title('Sum(I) over Qy')
plt.subplot(2, 2, 2)
plt.contourf(qy, qx, xu.maplog(data.sum(axis=1)), 150, cmap=my_cmap)
plt.plot([min(qy), max(qy)], [qxCOM, qxCOM], color='k', linestyle='-', linewidth=2)
plt.plot([qyCOM, qyCOM], [min(qx), max(qx)], color='k', linestyle='-', linewidth=2)
circle = plt.Circle((qyCOM, qxCOM), radius_mean + dr, color='0', fill=False, linestyle='dotted')
fig.gca().add_artist(circle)
circle = plt.Circle((qyCOM, qxCOM), radius_mean - dr, color='0', fill=False, linestyle='dotted')
fig.gca().add_artist(circle)
plt.colorbar()
plt.xlabel(r"Q$_y$ ($1/\AA$)")
plt.ylabel(r"Q$_x$ ($1/\AA$)")
plt.axis('scaled')
plt.title('Sum(I) over Qz')
plt.subplot(2, 2, 3)
plt.contourf(qy, qz, xu.maplog(data.sum(axis=0)), 150, cmap=my_cmap)
plt.plot([qyCOM, qyCOM], [min(qz), max(qz)], color='k', linestyle='-', linewidth=2)
plt.plot([min(qy), max(qy)], [qzCOM, qzCOM], color='k', linestyle='-', linewidth=2)
circle = plt.Circle((qyCOM, qzCOM), radius_mean + dr, color='0', fill=False, linestyle='dotted')
fig.gca().add_artist(circle)
circle = plt.Circle((qyCOM, qzCOM), radius_mean - dr, color='0', fill=False, linestyle='dotted')
fig.gca().add_artist(circle)
plt.colorbar()
plt.xlabel(r"Q$_y$ ($1/\AA$)")
plt.ylabel(r"Q$_z$ ($1/\AA$)")
plt.axis('scaled')
plt.title('Sum(I) over Qx')
fig.text(0.60, 0.30, "Scan " + str(scan), size=20)
if reconstructed_data == 0:
    fig.text(0.60, 0.25, "offset_eta=" + str(offset_eta), size=20)
    fig.text(0.60, 0.20, "offset_phi=" + str(offset_phi), size=20)
    fig.text(0.60, 0.15, "offset_chi=" + str(offset_chi), size=20)
plt.pause(0.1)
plt.savefig(savedir + 'diffpattern' + comment + 'S' + str(scan) + '_q=' + str(radius_mean) + '.png')
####################################
#  plot upper and lower part of intensity with intersecting sphere
####################################
if debug == 1:
    fig, ax = plt.subplots(num=2, figsize=(20, 15), dpi=80, facecolor='w', edgecolor='k')
    plt.subplot(2, 3, 1)
    plt.contourf(qz_top, qx, xu.maplog(intensity_top.sum(axis=2), 6, 1), 75, cmap=my_cmap)
    plt.plot([qzCOM, max(qz)], [qxCOM, qxCOM], color='k', linestyle='-', linewidth=2)
    plt.plot([qzCOM, qzCOM], [min(qx), max(qx)], color='k', linestyle='-', linewidth=2)
    circle = plt.Circle((qzCOM, qxCOM), radius_mean + dr, color='0', fill=False, linestyle='dotted')
    fig.gca().add_artist(circle)
    circle = plt.Circle((qzCOM, qxCOM), radius_mean - dr, color='0', fill=False, linestyle='dotted')
    fig.gca().add_artist(circle)
    plt.xlabel(r"Q$_z$ ($1/\AA$)")
    plt.ylabel(r"Q$_x$ ($1/\AA$)")
    plt.title('Top\nSum(I) over Q$_y$')
    plt.axis('scaled')
    plt.subplot(2, 3, 2)
    plt.contourf(qy, qx, xu.maplog(intensity_top.sum(axis=1), 6, 1), 75, cmap=my_cmap)
    plt.plot([min(qy), max(qy)], [qxCOM, qxCOM], color='k', linestyle='-', linewidth=2)
    plt.plot([qyCOM, qyCOM], [min(qx), max(qx)], color='k', linestyle='-', linewidth=2)
    circle = plt.Circle((qyCOM, qxCOM), radius_mean + dr, color='0', fill=False, linestyle='dotted')
    fig.gca().add_artist(circle)
    circle = plt.Circle((qyCOM, qxCOM), radius_mean - dr, color='0', fill=False, linestyle='dotted')
    fig.gca().add_artist(circle)
    plt.xlabel(r"Q$_y$ ($1/\AA$)")
    plt.ylabel(r"Q$_x$ ($1/\AA$)")
    plt.title('Top\nSum(I) over Q$_z$>Q$_z$COM')
    plt.axis('scaled')
    plt.subplot(2, 3, 3)
    plt.contourf(qy, qz_top, xu.maplog(intensity_top.sum(axis=0), 6, 1), 75, cmap=my_cmap)
    plt.plot([qyCOM, qyCOM], [qzCOM, max(qz)], color='k', linestyle='-', linewidth=2)
    plt.plot([min(qy), max(qy)], [qzCOM, qzCOM], color='k', linestyle='-', linewidth=2)
    circle = plt.Circle((qyCOM, qzCOM), radius_mean + dr, color='0', fill=False, linestyle='dotted')
    fig.gca().add_artist(circle)
    circle = plt.Circle((qyCOM, qzCOM), radius_mean - dr, color='0', fill=False, linestyle='dotted')
    fig.gca().add_artist(circle)
    plt.xlabel(r"Q$_y$ ($1/\AA$)")
    plt.ylabel(r"Q$_z$ ($1/\AA$)")
    plt.title('Top\nSum(I) over Q$_x$')
    plt.axis('scaled')
    plt.subplot(2, 3, 4)
    plt.contourf(qz_bottom, qx, xu.maplog(intensity_bottom.sum(axis=2), 6, 1), 75, cmap=my_cmap)
    plt.plot([min(qz), qzCOM], [qxCOM, qxCOM], color='k', linestyle='-', linewidth=2)
    plt.plot([qzCOM, qzCOM], [min(qx), max(qx)], color='k', linestyle='-', linewidth=2)
    circle = plt.Circle((qzCOM, qxCOM), radius_mean + dr, color='0', fill=False, linestyle='dotted')
    fig.gca().add_artist(circle)
    circle = plt.Circle((qzCOM, qxCOM), radius_mean - dr, color='0', fill=False, linestyle='dotted')
    fig.gca().add_artist(circle)
    plt.xlabel(r"Q$_z$ ($1/\AA$)")
    plt.ylabel(r"Q$_x$ ($1/\AA$)")
    plt.title('Bottom\nSum(I) over Q$_y$')
    plt.axis('scaled')
    plt.subplot(2, 3, 5)
    plt.contourf(qy, qx, xu.maplog(intensity_bottom.sum(axis=1), 6, 1), 75, cmap=my_cmap)
    plt.plot([min(qy), max(qy)], [qxCOM, qxCOM], color='k', linestyle='-', linewidth=2)
    plt.plot([qyCOM, qyCOM], [min(qx), max(qx)], color='k', linestyle='-', linewidth=2)
    circle = plt.Circle((qyCOM, qxCOM), radius_mean + dr, color='0', fill=False, linestyle='dotted')
    fig.gca().add_artist(circle)
    circle = plt.Circle((qyCOM, qxCOM), radius_mean - dr, color='0', fill=False, linestyle='dotted')
    fig.gca().add_artist(circle)
    plt.xlabel(r"Q$_y$ ($1/\AA$)")
    plt.ylabel(r"Q$_x$ ($1/\AA$)")
    plt.title('Bottom\nSum(I) over Q$_z$<Q$_z$COM')
    plt.axis('scaled')
    plt.subplot(2, 3, 6)
    plt.contourf(qy, qz_bottom, xu.maplog(intensity_bottom.sum(axis=0), 6, 1), 75, cmap=my_cmap)
    plt.plot([qyCOM, qyCOM], [min(qz), qzCOM], color='k', linestyle='-', linewidth=2)
    plt.plot([min(qy), max(qy)], [qzCOM, qzCOM], color='k', linestyle='-', linewidth=2)
    circle = plt.Circle((qyCOM, qzCOM), radius_mean + dr, color='0', fill=False, linestyle='dotted')
    fig.gca().add_artist(circle)
    circle = plt.Circle((qyCOM, qzCOM), radius_mean - dr, color='0', fill=False, linestyle='dotted')
    fig.gca().add_artist(circle)
    plt.xlabel(r"Q$_y$ ($1/\AA$)")
    plt.ylabel(r"Q$_z$ ($1/\AA$)")
    plt.title('Bottom\nSum(I) over Q$_x$')
    plt.axis('scaled')
    plt.pause(0.1)

#############################################
# apply mask
#############################################
I_masked_top = intensity_top*mask_top
I_masked_bottom = intensity_bottom*mask_bottom
if debug == 1:
    fig, ax = plt.subplots(num=3, figsize=(20, 15), dpi=80, facecolor='w', edgecolor='k')
    plt.subplot(2, 3, 1)
    plt.contourf(qz_top, qx, xu.maplog(I_masked_top.sum(axis=2), 5, 1), 75, cmap=my_cmap)
    circle = plt.Circle((qzCOM, qxCOM), radius_mean + dr, color='0', fill=False, linestyle='dotted')
    fig.gca().add_artist(circle)
    circle = plt.Circle((qzCOM, qxCOM), radius_mean - dr, color='0', fill=False, linestyle='dotted')
    fig.gca().add_artist(circle)
    plt.xlabel(r"Q$_z$ ($1/\AA$)")
    plt.ylabel(r"Q$_x$ ($1/\AA$)")
    plt.title('Top\nSum(I*mask) over Q$_y$')
    plt.axis('scaled')
    plt.subplot(2, 3, 2)
    plt.contourf(qy, qx, xu.maplog(I_masked_top.sum(axis=1), 5, 1), 75, cmap=my_cmap)
    circle = plt.Circle((qyCOM, qxCOM), radius_mean + dr, color='0', fill=False, linestyle='dotted')
    fig.gca().add_artist(circle)
    circle = plt.Circle((qyCOM, qxCOM), radius_mean - dr, color='0', fill=False, linestyle='dotted')
    fig.gca().add_artist(circle)
    plt.xlabel(r"Q$_y$ ($1/\AA$)")
    plt.ylabel(r"Q$_x$ ($1/\AA$)")
    plt.title('Top\nSum(I*mask) over Q$_z$>Q$_z$COM')
    plt.axis('scaled')
    plt.subplot(2, 3, 3)
    plt.contourf(qy, qz_top, xu.maplog(I_masked_top.sum(axis=0), 5, 1), 75, cmap=my_cmap)
    circle = plt.Circle((qyCOM, qzCOM), radius_mean + dr, color='0', fill=False, linestyle='dotted')
    fig.gca().add_artist(circle)
    circle = plt.Circle((qyCOM, qzCOM), radius_mean - dr, color='0', fill=False, linestyle='dotted')
    fig.gca().add_artist(circle)
    plt.xlabel(r"Q$_y$ ($1/\AA$)")
    plt.ylabel(r"Q$_z$ ($1/\AA$)")
    plt.title('Top\nSum(I*mask) over Q$_x$')
    plt.axis('scaled')
    plt.subplot(2, 3, 4)
    plt.contourf(qz_bottom, qx, xu.maplog(I_masked_bottom.sum(axis=2), 5, 1), 75, cmap=my_cmap)
    circle = plt.Circle((qzCOM, qxCOM), radius_mean + dr, color='0', fill=False, linestyle='dotted')
    fig.gca().add_artist(circle)
    circle = plt.Circle((qzCOM, qxCOM), radius_mean - dr, color='0', fill=False, linestyle='dotted')
    fig.gca().add_artist(circle)
    plt.xlabel(r"Q$_z$ ($1/\AA$)")
    plt.ylabel(r"Q$_x$ ($1/\AA$)")
    plt.title('Bottom\nSum(I*mask) over Q$_y$')
    plt.axis('scaled')
    plt.subplot(2, 3, 5)
    plt.contourf(qy, qx, xu.maplog(I_masked_bottom.sum(axis=1), 5, 1), 75, cmap=my_cmap)
    circle = plt.Circle((qyCOM, qxCOM), radius_mean + dr, color='0', fill=False, linestyle='dotted')
    fig.gca().add_artist(circle)
    circle = plt.Circle((qyCOM, qxCOM), radius_mean - dr, color='0', fill=False, linestyle='dotted')
    fig.gca().add_artist(circle)
    plt.xlabel(r"Q$_y$ ($1/\AA$)")
    plt.ylabel(r"Q$_x$ ($1/\AA$)")
    plt.title('Bottom\nSum(I*mask) over Q$_z$<Q$_z$COM')
    plt.axis('scaled')
    plt.subplot(2, 3, 6)
    plt.contourf(qy, qz_bottom, xu.maplog(I_masked_bottom.sum(axis=0), 5, 1), 75, cmap=my_cmap)
    circle = plt.Circle((qyCOM, qzCOM), radius_mean + dr, color='0', fill=False, linestyle='dotted')
    fig.gca().add_artist(circle)
    circle = plt.Circle((qyCOM, qzCOM), radius_mean - dr, color='0', fill=False, linestyle='dotted')
    fig.gca().add_artist(circle)
    plt.xlabel(r"Q$_y$ ($1/\AA$)")
    plt.ylabel(r"Q$_z$ ($1/\AA$)")
    plt.title('Bottom\nSum(I*mask) over Q$_x$')
    plt.axis('scaled')
    plt.pause(0.1)

########################################
# calculation of Euclidian metric coordinates
########################################
qx1_top = qx1*np.ones(intensity_top.shape)
qy1_top = qy1*np.ones(intensity_top.shape)
qx1_bottom = qx1*np.ones(intensity_bottom.shape)
qy1_bottom = qy1*np.ones(intensity_bottom.shape)
u_temp_top = (qx1_top - qxCOM)*radius_mean/(radius_mean+(qz1_top - qzCOM))  # projection from South pole
v_temp_top = (qy1_top - qyCOM)*radius_mean/(radius_mean+(qz1_top - qzCOM))  # projection from South pole
u_temp_bottom = (qx1_bottom - qxCOM)*radius_mean/(radius_mean+(qzCOM-qz1_bottom))  # projection from North pole
v_temp_bottom = (qy1_bottom - qyCOM)*radius_mean/(radius_mean+(qzCOM-qz1_bottom))  # projection from North pole
u_top = u_temp_top[mask_top]/radius_mean*90    # rescaling from radius_mean to 90
v_top = v_temp_top[mask_top]/radius_mean*90    # rescaling from radius_mean to 90
u_bottom = u_temp_bottom[mask_bottom]/radius_mean*90    # rescaling from radius_mean to 90
v_bottom = v_temp_bottom[mask_bottom]/radius_mean*90    # rescaling from radius_mean to 90

int_temp_top = I_masked_top[mask_top]
int_temp_bottom = I_masked_bottom[mask_bottom]
u_grid_top, v_grid_top = np.mgrid[-91:91:365j, -91:91:365j]
u_grid_bottom, v_grid_bottom = np.mgrid[-91:91:365j, -91:91:365j]
int_grid_top = griddata((u_top, v_top), int_temp_top, (u_grid_top, v_grid_top), method='linear')
int_grid_bottom = griddata((u_bottom, v_bottom), int_temp_bottom, (u_grid_bottom, v_grid_bottom), method='linear')
int_grid_top = int_grid_top / int_grid_top[int_grid_top > 0].max() * 10000  # normalize for easier plotting
int_grid_bottom = int_grid_bottom / int_grid_bottom[int_grid_bottom > 0].max() * 10000  # normalize for easier plotting
int_grid_top[np.isnan(int_grid_top)] = 0
int_grid_bottom[np.isnan(int_grid_bottom)] = 0
########################################
# calculate theoretical angles between the measured reflection and other planes - only for cubic
########################################
planes = {}
# planes['1 0 0'] = plane_angle(reflection, np.array([1, 0, 0]))
# planes['-1 0 0'] = plane_angle(reflection, np.array([-1, 0, 0]))
# planes['1 1 0'] = plane_angle(reflection, np.array([1, 1, 0]))
# planes['1 -1 0'] = plane_angle(reflection, np.array([1, -1, 0]))
# planes['1 1 1'] = plane_angle(reflection, np.array([1, 1, 1]))
# planes['1 -1 1'] = plane_angle(reflection, np.array([1, -1, 1]))
# planes['1 -1 -1'] = plane_angle(reflection, np.array([1, -1, -1]))
planes['2 1 0'] = plane_angle(reflection, np.array([2, 1, 0]))
planes['2 -1 0'] = plane_angle(reflection, np.array([2, -1, 0]))
# planes['2 -1 1'] = plane_angle(reflection, np.array([2, -1, 1]))
# planes['3 0 1'] = plane_angle(reflection, np.array([3, 0, 1]))
# planes['3 -1 0'] = plane_angle(reflection, np.array([3, -1, 0]))
# planes['3 2 1'] = plane_angle(reflection, np.array([3, 2, 1]))
# planes['3 -2 -1'] = plane_angle(reflection, np.array([3, -2, -1]))
# planes['-3 0 -1'] = plane_angle(reflection, np.array([-3, 0, -1]))
# planes['4 0 -1'] = plane_angle(reflection, np.array([4, 0, -1]))
# planes['5 2 0'] = plane_angle(reflection, np.array([5, 2, 0]))
# planes['5 -2 0'] = plane_angle(reflection, np.array([5, -2, 0]))
# planes['5 2 1'] = plane_angle(reflection, np.array([5, 2, 1]))
# planes['5 -2 -1'] = plane_angle(reflection, np.array([5, -2, -1]))
# planes['-5 0 -2'] = plane_angle(reflection, np.array([-5, 0, -2]))
# planes['7 0 3'] = plane_angle(reflection, np.array([7, 0, 3]))
# planes['7 -3 0'] = plane_angle(reflection, np.array([-7, 0, 3]))
# planes['-7 0 -3'] = plane_angle(reflection, np.array([-7, 0, -3]))
# planes['1 3 6'] = plane_angle(reflection, np.array([1, 3, 6]))

########################################
# create top projection from South pole
########################################
# plot the stereographic projection
myfig, myax0 = plt.subplots(1, 1, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
# plot top part (projection from South pole on equator)
plt0 = myax0.contourf(u_grid_top, v_grid_top, abs(int_grid_top), range(100, 6100, 200), cmap='hsv')
# plt0 = myax0.contourf(u_grid_top, v_grid_top, np.log10(abs(int_grid_top)), np.linspace(2.0, 4.0, 9), cmap='hsv')
plt.colorbar(plt0, ax=myax0)
myax0.axis('equal')
myax0.axis('off')

# # add the projection of the elevation angle, depending on the center of projection
for ii in range(15, 90, 5):
    circle = plt.Circle((0, 0),
                        radius_mean * np.sin(ii * np.pi / 180) / (1 + np.cos(ii * np.pi / 180)) * 90 / radius_mean,
                        color='grey', fill=False, linestyle='dotted', linewidth=0.2)
    myax0.add_artist(circle)
for ii in range(10, 90, 20):
    circle = plt.Circle((0, 0),
                        radius_mean * np.sin(ii * np.pi / 180) / (1 + np.cos(ii * np.pi / 180)) * 90 / radius_mean,
                        color='grey', fill=False, linestyle='dotted', linewidth=0.5)
    myax0.add_artist(circle)
for ii in range(10, 95, 20):
    myax0.text(-radius_mean * np.sin(ii * np.pi / 180) / (1 + np.cos(ii * np.pi / 180)) * 90 / radius_mean, 0,
               str(ii) + '$^\circ$', fontsize=12, color='k', fontweight='bold')
circle = plt.Circle((0, 0), 90, color='k', fill=False, linewidth=1)
myax0.add_artist(circle)

# add azimutal lines every 5 and 45 degrees
for ii in range(5, 365, 5):
    myax0.plot([0, 90 * np.cos(ii * np.pi / 180)], [0, 90 * np.sin(ii * np.pi / 180)], color='grey',
               linestyle='dotted', linewidth=0.2)
for ii in range(0, 365, 20):
    myax0.plot([0, 90 * np.cos(ii * np.pi / 180)], [0, 90 * np.sin(ii * np.pi / 180)], color='grey',
               linestyle='dotted', linewidth=0.5)

# draw circles corresponding to particular reflection
if flag_plotplanes == 1:
    indx = 5
    for key, value in planes.items():
        circle = plt.Circle((0, 0), radius_mean * np.sin(value * np.pi / 180) /
                            (1 + np.cos(value * np.pi / 180)) * 90 / radius_mean,
                            color='r', fill=False, linestyle='dotted', linewidth=2)
        myax0.add_artist(circle)
        # myax0.text(np.cos(indx * np.pi / 180) * radius_mean * np.sin(value * np.pi / 180) /
        #            (1 + np.cos(value * np.pi / 180)) * 90 / radius_mean,
        #            np.sin(indx * np.pi / 180) * radius_mean * np.sin(value * np.pi / 180) /
        #            (1 + np.cos(value * np.pi / 180)) * 90 / radius_mean,
        #            key, fontsize=14, color='k', fontweight='bold')
        indx = indx + 5
        print(key + ": ", str('{:.2f}'.format(value)))
myax0.set_title('Top projection\nfrom South pole S' + str(scan)+'\n')
if reconstructed_data == 0:
    myfig.text(0.2, 0.05, "q=" + str(radius_mean) +
               " dq=" + str(dr) + " offset_eta=" + str(offset_eta) + " offset_phi=" + str(offset_phi) +
               " offset_chi=" + str(offset_chi), size=20)

else:
    myfig.text(0.4, 0.8, "q=" + str(radius_mean) + " dq=" + str(dr), size=20)
plt.pause(0.1)
plt.savefig(savedir + 'South pole' + comment + '_S' + str(scan) + '.png')
########################################
# create bottom projection from North pole
########################################
myfig, myax1 = plt.subplots(1, 1, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
plt1 = myax1.contourf(u_grid_bottom, v_grid_bottom, abs(int_grid_bottom), range(100, 6100, 200), cmap='hsv')
plt.colorbar(plt1, ax=myax1)
myax1.axis('equal')
myax1.axis('off')

# # add the projection of the elevation angle, depending on the center of projection
for ii in range(15, 90, 5):
    circle = plt.Circle((0, 0),
                        radius_mean * np.sin(ii * np.pi / 180) / (1 + np.cos(ii * np.pi / 180)) * 90 / radius_mean,
                        color='grey', fill=False, linestyle='dotted', linewidth=0.2)
    myax1.add_artist(circle)
for ii in range(10, 90, 20):
    circle = plt.Circle((0, 0),
                        radius_mean * np.sin(ii * np.pi / 180) / (1 + np.cos(ii * np.pi / 180)) * 90 / radius_mean,
                        color='grey', fill=False, linestyle='dotted', linewidth=0.5)
    myax1.add_artist(circle)
for ii in range(10, 95, 20):
    myax1.text(-radius_mean * np.sin(ii * np.pi / 180) / (1 + np.cos(ii * np.pi / 180)) * 90 / radius_mean, 0,
               str(ii) + '$^\circ$', fontsize=10, color='k', fontweight='bold')
circle = plt.Circle((0, 0), 90, color='k', fill=False, linewidth=1)
myax1.add_artist(circle)

# add azimutal lines every 5 and 45 degrees
for ii in range(5, 365, 5):
    myax1.plot([0, 90 * np.cos(ii * np.pi / 180)], [0, 90 * np.sin(ii * np.pi / 180)], color='grey',
               linestyle='dotted', linewidth=0.2)
for ii in range(0, 365, 20):
    myax1.plot([0, 90 * np.cos(ii * np.pi / 180)], [0, 90 * np.sin(ii * np.pi / 180)], color='grey',
               linestyle='dotted', linewidth=0.5)

# draw circles corresponding to particular reflection
if flag_plotplanes == 1:
    indx = 0
    for key, value in planes.items():
        circle = plt.Circle((0, 0), radius_mean * np.sin(value * np.pi / 180) /
                            (1 + np.cos(value * np.pi / 180)) * 90 / radius_mean,
                            color='r', fill=False, linestyle='dotted', linewidth=2)
        myax1.add_artist(circle)
        # myax1.text(np.cos(indx * np.pi / 180) * radius_mean * np.sin(value * np.pi / 180) /
        #            (1 + np.cos(value * np.pi / 180)) * 90 / radius_mean,
        #            np.sin(indx * np.pi / 180) * radius_mean * np.sin(value * np.pi / 180) /
        #            (1 + np.cos(value * np.pi / 180)) * 90 / radius_mean,
        #            key, fontsize=14, color='k', fontweight='bold')
        indx = indx + 5
        print(key + ": ", str('{:.2f}'.format(value)))
plt.title('Bottom projection\nfrom North pole S' + str(scan) + '\n')
# save figure
if reconstructed_data == 0:
    myfig.text(0.2, 0.05, "q=" + str(radius_mean) +
               " dq=" + str(dr) + " offset_eta=" + str(offset_eta) + " offset_phi=" + str(offset_phi) +
               " offset_chi=" + str(offset_chi), size=20)

else:
    myfig.text(0.4, 0.8, "q=" + str(radius_mean) + " dq=" + str(dr), size=20)
plt.pause(0.1)
plt.savefig(savedir + 'North pole' + comment + '_S' + str(scan) + '.png')

################################
# save grid points in txt file #
################################
fichier = open(savedir + 'Poles' + comment + '_S' + str(scan) + '.dat', "w")
# save metric coordinates in text file
for ii in range(len(u_grid_top)):
    for jj in range(len(v_grid_top)):
        fichier.write(str(u_grid_top[ii, 0]) + '\t' + str(v_grid_top[0, jj]) + '\t' +
                      str(int_grid_top[ii, jj]) + '\t' + str(u_grid_bottom[ii, 0]) + '\t' +
                      str(v_grid_bottom[0, jj]) + '\t' + str(int_grid_bottom[ii, jj]) + '\n')
fichier.close()
plt.ioff()
plt.show()