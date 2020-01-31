# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np
import matplotlib.ticker as ticker
import sys
from silx.io.specfile import SpecFile
import tkinter as tk
from tkinter import filedialog
import xrayutilities as xu
import fabio
import scipy.signal  # for medfilt2d
from mayavi import mlab
import os
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

helptext = """
# TODO Refactor
"""

scan = 2227    # spec scan number
flag_medianfilter = 0  # set to 1 for applying med2filter [3,3]
comment = ""

filtered_data = 0  # set to 1 if the data is already a 3D array, 0 otherwise
# Should be the same shape as in specfile, before orthogonalization
plot_2D = 1  # 1 to plot also 2D views
tick_direction = 'out'  # 'out', 'in', 'inout'
tick_length = 10  # in plots
tick_width = 2  # in plots
centering = "max"  # "max" or "com"
######################
fix_bragg = False  # True  # if COM centering does not work, set this to True and provide qzcom, qycom, qxcom below
qzCOM = 2.14875
qxCOM = 0.12175
qyCOM = -0.04365
######################
flag_savedata = 0      # set to 1 to save data
debug = 0  # 1 to show more plots, 0 otherwise
sdd = 0.50678  # sample to detector distance in m
en = 8994  # x-ray energy in eV
offset_eta = 0  # positive make diff pattern rotate counter-clockwise (eta rotation around Qy)
# will shift peaks rightwards in the pole figure
offset_phi = 0     # positive make diff pattern rotate clockwise (phi rotation around Qz)
# will rotate peaks counterclockwise in the pole figure
offset_chi = 0  # positive make diff pattern rotate clockwise (chi rotation around Qx)
# will shift peaks upwards in the pole figure
offset = -0.6358   # outer detector angle offset (nu)
threshold = 1  # photon threshold in detector counts
specdir = "C:/Users/carnis/Work Folders/Documents/data/CH4760_Pt/"
datadir = specdir + "S"+str(scan)+"/data/"
savedir = specdir + "S"+str(scan)+"/simu/Figures/Fig1_model/slices/"
spec_prefix = "alignment"
detector = "Maxipix"    # "Eiger2M" or "Maxipix"
if detector == "Eiger2M":  # eiger.y_bragg = 1412  # y pixel of the Bragg peak, only used for Eiger ROI
    x_bragg = 430  # x pixel of the Bragg peak, only used for Eiger ROI
    roi = [1102, 1610, x_bragg - 300, x_bragg + 301]
    pixelsize = 7.5e-05
    ccdfiletmp = os.path.join(datadir, "align_eiger2M_%05d.edf.gz")   # template for the CCD file names
elif detector == "Maxipix":  # maxipix
    roi = [0, 516, 0, 516]  # [261, 516, 261, 516]  # [0, 516, 0, 516]
    pixelsize = 5.5e-05
    ccdfiletmp = os.path.join(datadir, "data_mpx4_%05d.edf.gz")   # template for the CCD file names
else:
    sys.exit("Incorrect value for 'detector' parameter")
comment = ''
hotpixels_file = ""
flatfield_file = specdir + "flatfield_maxipix_8kev.npz"
nch1 = roi[1] - roi[0]  # 2164 Eiger, 516 Maxipix
nch2 = roi[3] - roi[2]  # 1030 Eiger, 516 Maxipix
# geometry of diffractometer
qconv = xu.experiment.QConversion(['y-', 'x+', 'z-'], ['z-', 'y-'], [1, 0, 0])
# 3S+2D goniometer (simplified ID01 goniometer, sample: eta, chi, phi      detector: nu,del
# convention for coordinate system: x downstream; z upwards; y to the "outside" (righthanded)
cch1 = 207.88 - roi[0]  # direct_beam_y - roi[0]
cch2 = 50.49 - roi[2]  # direct_beam_x - roi[2]
hxrd = xu.experiment.HXRD([1, 0, 0], [0, 0, 1], en=en, qconv=qconv)
# detector should be calibrated for the same roi as defined above
hxrd.Ang2Q.init_area('z-', 'y+', cch1=cch1, cch2=cch2, Nch1=nch1, Nch2=nch2, pwidth1=pixelsize, pwidth2=pixelsize,
                     distance=sdd, detrot=-0.436, tiltazimuth=273.2, tilt=3.940)
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
        if mydetector == "Eiger2M":
            region = [0, 2164, 0, 1030]
        elif mydetector == "Maxipix":
            region = [0, 516, 0, 516]
    if mydetector == "Eiger2M":
        counter = 'ei2minr'
        mymask = np.zeros((2164, 1030))
    elif mydetector == "Maxipix":
        counter = 'mpx4inr'
        mymask = np.zeros((516, 516))
    else:
        sys.exit("Incorrect value for 'mydetector' parameter")
    if myhotpixels != "":
        print("Loading hotpixels array")
        hotpix_array = np.load(myhotpixels)
        npz_key = hotpix_array.keys()
        hotpix_array = hotpix_array[npz_key[0]]
        if len(hotpix_array.shape) == 3:  # 3D array
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
            if mydetector == "Eiger2M":
                ccdraw, mymask = mask_eiger(ccdraw, mymask)
            elif mydetector == "Maxipix":
                ccdraw, mymask = mask_maxipix(ccdraw, mymask)
            ccdraw = myflatfield * ccdraw
            ccd = xu.blockAverage2D(ccdraw, nav[0], nav[1], roi=region)
            rawdata[int(i - ccdn[0]), :, :] = ccd
    else:
        print('Loading filtered data')
        myfile_path = filedialog.askopenfilename(initialdir=specdir,
                                                 title="Select 3D data", filetypes=[("NPZ", "*.npz")])
        rawdata = np.load(myfile_path)['data']
        rawdata = rawdata[region[0]:region[1], region[2]:region[3]]
    mymask = mymask[region[0]:region[1], region[2]:region[3]]
    numz, numy, numx = rawdata.shape
    if numz != len(ccdn):
        print('Filtered data has not the same shape as raw data')
        sys.exit()
    rawmask3d = np.zeros((numz, region[1] - region[0], region[3] - region[2]))
    for index in range(numz):
        rawmask3d[index, :, :] = mymask
    # transform scan angles to reciprocal space coordinates for all detector pixels
    myqx, myqy, myqz = hxrd.Ang2Q.area(eta, chi, phi, nu, delta, delta=(0, 0, 0, offset, 0))
    mygridder = xu.Gridder3D(numz, numy, numx)
    # convert mask to rectangular grid in reciprocal space
    mygridder(myqx, myqz, myqy, rawmask3d)
    mymask3d = np.copy(mygridder.data)
    # convert data to rectangular grid in reciprocal space
    mygridder(myqx, myqz, myqy, rawdata)
    return mygridder.xaxis, mygridder.yaxis, mygridder.zaxis, rawdata, mygridder.data, rawmask3d, mymask3d


###################################################################################
plt.ion()
root = tk.Tk()
root.withdraw()
plot_title = ['QzQx', 'QyQx', 'QyQz']
if flatfield_file != "":
    flatfield = np.load(flatfield_file)['flatfield']
else:
    flatfield = None
spec_file = SpecFile(specdir + spec_prefix + ".spec")
qx, qz, qy, intensity, data, _, _ = gridmap(spec_file, scan, detector, roi, flatfield, hotpixels_file)

nz, ny, nx = data.shape  # nexus convention
if flag_medianfilter == 1:  # apply some noise filtering
    for idx in range(nz):
        data[idx, :, :] = scipy.signal.medfilt2d(data[idx, :, :], [3, 3])
if flag_savedata == 1:
    np.savez_compressed(savedir+'S'+str(scan)+'_ortho_diffpattern', data=data)

##################################
# define the center of the sphere
##################################
intensity = data
intensity[intensity <= threshold] = 0   # photon threshold
if not fix_bragg:
    if centering == "com":
        qzCOM = 1/intensity.sum()*(qz*intensity.sum(axis=0).sum(axis=1)).sum()  # COM in qz
        qyCOM = 1/intensity.sum()*(qy*intensity.sum(axis=0).sum(axis=0)).sum()  # COM in qy
        qxCOM = 1/intensity.sum()*(qx*intensity.sum(axis=1).sum(axis=1)).sum()  # COM in qx
        print("Center of mass [qx, qy, qz]: [",
              str('{:.2f}'.format(qxCOM)), str('{:.2f}'.format(qyCOM)), str('{:.2f}'.format(qzCOM)), ']')
    else:  # "max"
        z0, y0, x0 = np.unravel_index(abs(data).argmax(), data.shape)  # Nexus convention
        qxCOM = qx[z0]
        qzCOM = qz[y0]
        qyCOM = qy[x0]
        print("Max [qx, qy, qz]: [",
              str('{:.2f}'.format(qxCOM)), str('{:.2f}'.format(qyCOM)), str('{:.2f}'.format(qzCOM)), ']')
else:
    print("User defined Bragg peak: [",
          str('{:.2f}'.format(qxCOM)), str('{:.2f}'.format(qyCOM)), str('{:.2f}'.format(qzCOM)), ']')

if plot_2D == 1:
    _, ax0 = plt.subplots(1, 1)
    plt.contourf(qz, qx, xu.maplog(intensity.sum(axis=2)), 150, cmap=my_cmap)
    ax0.tick_params(labelbottom='off', labelleft='off', bottom='off', top='off', left='off', right='off',
                    direction=tick_direction, length=tick_length, width=tick_width)
    plt.savefig(savedir + 'QzQx' + comment + '.png', bbox_inches="tight")
    ax0.tick_params(labelbottom='on', labelleft='on', bottom='on', top='off', left='on', right='off',
                    direction=tick_direction, length=tick_length, width=tick_width)
    plt.savefig(savedir + 'QzQx' + comment + '_label.png', bbox_inches="tight")

    _, ax0 = plt.subplots(1, 1)
    plt.contourf(qy, qx, xu.maplog(intensity.sum(axis=1)), 150, cmap=my_cmap)
    ax0.tick_params(labelbottom='off', labelleft='off', bottom='off', top='off', left='off', right='off',
                    direction=tick_direction, length=tick_length, width=tick_width)
    plt.savefig(savedir + 'QyQx' + comment + '.png', bbox_inches="tight")
    ax0.tick_params(labelbottom='on', labelleft='on', bottom='on', top='off', left='on', right='off',
                    direction=tick_direction, length=tick_length, width=tick_width)
    plt.savefig(savedir + 'QyQx' + comment + '_label.png', bbox_inches="tight")

    _, ax0 = plt.subplots(1, 1)
    plt.contourf(qy, qz, xu.maplog(intensity.sum(axis=0)), 150, cmap=my_cmap)
    ax0.tick_params(labelbottom='off', labelleft='off', bottom='off', top='off', left='off', right='off',
                    direction=tick_direction, length=tick_length, width=tick_width)
    plt.savefig(savedir + 'QyQz' + comment + '.png', bbox_inches="tight")
    ax0.tick_params(labelbottom='on', labelleft='on', bottom='on', top='off', left='on', right='off',
                    direction=tick_direction, length=tick_length, width=tick_width)
    plt.savefig(savedir + 'QyQz' + comment + '_label.png', bbox_inches="tight")

    fig, ax = plt.subplots(num=1, figsize=(20, 15), dpi=80, facecolor='w', edgecolor='k')
    plt.subplot(2, 2, 1)
    plt.contourf(qz, qx, xu.maplog(intensity.sum(axis=2)), 150, cmap=my_cmap)
    plt.plot([min(qz), max(qz)], [qxCOM, qxCOM], color='k', linestyle='-', linewidth=2)
    plt.plot([qzCOM, qzCOM], [min(qx), max(qx)], color='k', linestyle='-', linewidth=2)
    plt.colorbar()
    plt.xlabel(r"Q$_z$ ($1/\AA$)")
    plt.ylabel(r"Q$_x$ ($1/\AA$)")
    plt.axis('scaled')
    plt.title('Sum(I) over Qy')
    plt.subplot(2, 2, 2)
    plt.contourf(qy, qx, xu.maplog(intensity.sum(axis=1)), 150, cmap=my_cmap)
    plt.plot([min(qy), max(qy)], [qxCOM, qxCOM], color='k', linestyle='-', linewidth=2)
    plt.plot([qyCOM, qyCOM], [min(qx), max(qx)], color='k', linestyle='-', linewidth=2)
    plt.colorbar()
    plt.xlabel(r"Q$_y$ ($1/\AA$)")
    plt.ylabel(r"Q$_x$ ($1/\AA$)")
    plt.axis('scaled')
    plt.title('Sum(I) over Qz')
    plt.subplot(2, 2, 3)
    plt.contourf(qy, qz, xu.maplog(intensity.sum(axis=0)), 150, cmap=my_cmap)
    plt.plot([qyCOM, qyCOM], [min(qz), max(qz)], color='k', linestyle='-', linewidth=2)
    plt.plot([min(qy), max(qy)], [qzCOM, qzCOM], color='k', linestyle='-', linewidth=2)
    plt.colorbar()
    plt.xlabel(r"Q$_y$ ($1/\AA$)")
    plt.ylabel(r"Q$_z$ ($1/\AA$)")
    plt.axis('scaled')
    plt.title('Sum(I) over Qx')
    fig.tight_layout()
    plt.pause(0.1)

    fig = plt.figure(figsize=(18, 15))
    plt.subplot(2, 2, 1)
    plt.imshow(np.log10(intensity.sum(axis=2)), vmin=0, cmap=my_cmap)
    plt.title('sum(masked data)\n in ' + plot_title[0])
    plt.axis('scaled')
    plt.colorbar()
    plt.subplot(2, 2, 2)
    plt.imshow(np.log10(intensity.sum(axis=1)), vmin=0, cmap=my_cmap)
    plt.title('sum(masked data)\n in ' + plot_title[1])
    plt.axis('scaled')
    plt.colorbar()
    plt.subplot(2, 2, 3)
    plt.imshow(np.log10(intensity.sum(axis=0)), vmin=0, cmap=my_cmap)
    plt.title('sum(masked data)\n in ' + plot_title[2])
    plt.axis('scaled')
    plt.colorbar()
    fig.tight_layout()
    plt.pause(0.1)

# plot 3D map using mayavi mlab
intensity[intensity == 0] = np.nan
grid_qx, grid_qz, grid_qy = np.mgrid[qx.min():qx.max():1j * nz, qz.min():qz.max():1j * ny, qy.min():qy.max():1j*nx]
# in nexus convention, z is downstream, y vertical and x outboard
# but with Q, Qx is downstream, Qz vertical and Qy outboard
mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
mlab.contour3d(grid_qx[:, 165:435, 40:360], grid_qz[:, 165:435, 40:360],
               grid_qy[:, 165:435, 40:360], np.log10(intensity[:, 165:435, 40:360]),
               contours=20, opacity=0.5, colormap="jet")
mlab.colorbar(orientation="vertical", nb_labels=6)
mlab.outline(line_width=2.0)
mlab.axes(ranges=[-0.1, 0.05, 2.71, 2.83, 0.05, 0.19], nb_labels=3, xlabel='Qx', ylabel='Qz', zlabel='Qy')  #
mlab.show()
