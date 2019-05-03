# -*- coding: utf-8 -*-
"""
calc_angles_beam.py
calculate delta and nu angles from direct beam and Bragg peak positions
Based on ESRF/ID01 geometry
Input: direct beam and Bragg peak position, sample to detector distance
Output: energy, nu, delta of Bragg peak
@author: CARNIS
"""
import numpy as np
import xrayutilities as xu
from matplotlib import pyplot as plt
import os
import fabio
from scipy.ndimage.measurements import center_of_mass
from scipy.interpolate import interp1d
import sys
import silx.io
import tkinter as tk
from tkinter import filedialog
from lmfit.models import VoigtModel
# TODO: add the fitting for energy scan
scan = 107
filtered_data = 0  # set to 1 if the data is already a 3D array, 0 otherwise
# Should be the same shape as in specfile, before orthogonalization
reflection = np.array([1, 1, 1])  # measured reflection, use for estimating the temperature
reference_spacing = None  # for calibrating the thermal expansion, if None it is fixed to Pt 3.9236/norm(reflection)
reference_temperature = None  # used to calibrate the thermal expansion, if None it is fixed to 293.15K (RT)
direct_beam_x = 50.40  # x horizontal,  cch2 in xrayutilities
direct_beam_y = 451.02  # y vertical,  cch1 in xrayutilities
direct_nu = -0.124  # outer angle in xrayutilities
direct_del = -0.052
sdd = 0.9207  # sample to detector distance in m
energy = 7994  # in eV, offset of 6eV at ID01
detector = 1    # 0 for eiger, 1 for maxipix
specdir = "C:/Users/Jerome/Documents/data/MA3063/data/"
# specdir = "C:/Users/carnis/Work Folders/Documents/data/HC2277/"
# specdir = "C:/Users/carnis/Work Folders/Documents/data/IHR_11112018/"
# specdir = "C:/Users/carnis/Work Folders/Documents/data/HS4670/"
# specdir = "C:/Users/carnis/Work Folders/Documents/data/quasicrystal/"
# specdir = "C:/Users/carnis/Work Folders/Documents/data/CH5309/data/"
spec_prefix = "align"  # "alignment_freshstart"  #
datadir = specdir + "S" + str(scan)+"/data/"
centering = 0  # 0 max, 1 center of mass
hotpixels_file = ""  # specdir + "hotpixels.npz"
flatfield_file = ""  # specdir + "flatfield_maxipix_8kev.npz"  # "flatfield_eiger.npz"  #
if detector == 0:  # eiger
    nb_pixel_x = 1030  # 1030
    nb_pixel_y = 1614  # 2164  # 1614 now since one quadrant is dead
    x_bragg = 405  # 900  # 405
    # region = [552, 803, x_bragg - 200, x_bragg + 200]  # 1060
    # region = [1102, 1610, x_bragg - 300, x_bragg + 301]  # 2164 x 1030
    region = [0, nb_pixel_y, 0, nb_pixel_x]
    pixel_size = 7.5e-05
    counter = 'ei2minr'
    ccdfiletmp = os.path.join(datadir, "data_eiger2M_%05d.edf.gz")   # template for the CCD file names
elif detector == 1:  # maxipix
    nb_pixel_x = 516  # 516
    nb_pixel_y = 516  # 516
    region = [0, 516, 0, 516]  # 516 x 516
    pixel_size = 5.5e-05
    counter = 'mpx4inr'
    ccdfiletmp = os.path.join(datadir, "data_mpx4_%05d.edf.gz")   # template for the CCD file names5
else:
    sys.exit("Incorrect value for 'detector' parameter")
nav = [1, 1]  # reduce data: number of pixels to average in each detector direction
##########################################################
# end of user parameters
##########################################################


def check_pixels(mydata, mymask, var_threshold=5, debugging=0):
    """
    function to check for hot pixels in the data
    :param mydata: detector 3d data
    :param mymask: 2d mask
    :param var_threshold: pixels with 1/var > var_threshold*1/var.mean() will be masked
    :param debugging: to see plots before and after
    return mydata and mymask updated
    """
    numz, numy, numx = mydata.shape
    meandata = mydata.mean(axis=0)  # 2D
    vardata = 1/mydata.var(axis=0)  # 2D

    if debugging == 1:
        plt.figure(figsize=(18, 15))
        plt.subplot(1, 2, 1)
        plt.imshow(meandata, vmin=0, vmax=1)
        plt.colorbar()
        plt.axis('scaled')
        plt.title("Mean of data along axis=0\nbefore masking")
        plt.subplot(1, 2, 2)
        plt.imshow(vardata, vmin=0)
        plt.colorbar()
        plt.title("1/variance of data along axis=0\nbefore masking")
        plt.axis('scaled')
        plt.pause(0.1)
    # TODO: check with RMS of amplitude
    var_mean = vardata[vardata != np.inf].mean()
    vardata[meandata == 0] = var_mean  # pixels were data=0 (hence 1/variance=inf) are set to the mean of 1/var
    indices_badpixels = np.nonzero(vardata > var_mean * var_threshold)  # isolate constants pixels != 0 (1/variance=inf)
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
        plt.axis('scaled')
        plt.title("Mean of data along axis=0\nafter masking")
        plt.subplot(1, 2, 2)
        plt.imshow(vardata, vmin=0)
        plt.colorbar()
        plt.title("Variance of data along axis=0\nafter masking")
        plt.axis('scaled')
        plt.pause(0.1)
    # print(str(indices_badpixels[0].shape[0]), "badpixels with 1/var>", str(var_threshold),
    #       '*1/var.mean() were masked on a total of', str(numx*numy))
    print(str(indices_badpixels[0].shape[0]), "badpixels were masked on a total of", str(numx*numy))
    return mydata, mymask


def remove_hotpixels_maxipix(mydata, mymask, hot_file):  # , mymask):
    """
    function to remove hot pixels from CCD frames
    """
    if hot_file != "":
        hotpixels = np.load(hot_file)['data']
        mydata[hotpixels != 0] = 0
        mymask[hotpixels != 0] = 1
    return mydata, mymask


def mask_maxipix(mydata, mymask):
    mydata[:, 255:261] = 0
    mydata[255:261, :] = 0
    # mydata[0:50, 0:65] = 0
    mydata[303:307, 96] = 0
    mydata[304, 97] = 0

    mymask[:, 255:261] = 1
    mymask[255:261, :] = 1
    # mymask[0:50, 0:65] = 1
    mymask[303:307, 96] = 1
    mymask[304, 97] = 1
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


def bragg_temperature(spacing, my_reflection, spacing_ref=None, temperature_ref=None, use_q=0, material=None):
    """
    Calculate the temperature from Bragg peak position
    :param spacing: q or planar distance, in inverse angstroms or angstroms
    :param my_reflection: measured reflection, e.g. np.array([1, 1, 1])
    :param spacing_ref: reference spacing at known temperature (include substrate-induced strain)
    :param temperature_ref: in K, known temperature for the reference spacing
    :param use_q: use q (set to 1) or planar distance (set to 0)
    :param material: at the moment only 'Pt'
    :return: calculated temprature
    """
    if material == 'Pt':
        # reference values for Pt: temperature in K, thermal expansion x 10^6 in 1/K, lattice parameter in angstroms
        expansion_data = np.array([[100, 6.77, 3.9173], [110, 7.10, 3.9176], [120, 7.37, 3.9179], [130, 7.59, 3.9182],
                                  [140, 7.78, 3.9185], [150, 7.93, 3.9188], [160, 8.07, 3.9191], [180, 8.29, 3.9198],
                                  [200, 8.46, 3.9204], [220, 8.59, 3.9211], [240, 8.70, 3.9218], [260, 8.80, 3.9224],
                                  [280, 8.89, 3.9231], [293.15, 8.93, 3.9236], [300, 8.95, 3.9238], [400, 9.25, 3.9274],
                                  [500, 9.48, 3.9311], [600, 9.71, 3.9349], [700, 9.94, 3.9387], [800, 10.19, 3.9427],
                                  [900, 10.47, 3.9468], [1000, 10.77, 3.9510], [1100, 11.10, 3.9553],
                                  [1200, 11.43, 3.9597]])
        if spacing_ref is None:
            print('Using the reference spacing of Platinum')
            spacing_ref = 3.9236 / np.linalg.norm(my_reflection)  # angstroms
        if temperature_ref is None:
            temperature_ref = 293.15  # K
    else:
        return 0
    if use_q == 1:
        spacing = 2 * np.pi / spacing  # go back to distance
        spacing_ref = 2 * np.pi / spacing_ref  # go back to distance
    spacing = spacing * np.linalg.norm(my_reflection)  # go back to lattice constant
    spacing_ref = spacing_ref * np.linalg.norm(my_reflection)  # go back to lattice constant
    print('Reference spacing at', temperature_ref, 'K   =', str('{:.4f}'.format(spacing_ref)), 'angstroms')
    print('Spacing =', str('{:.4f}'.format(spacing)), 'angstroms using reflection', my_reflection)

    # fit the experimental spacing with non corrected platinum curve
    myfit = np.poly1d(np.polyfit(expansion_data[:, 2], expansion_data[:, 0], 3))
    print('Temperature without offset correction=', int(myfit(spacing) - 273.15), 'C')

    # find offset for platinum reference curve
    myfit = np.poly1d(np.polyfit(expansion_data[:, 0], expansion_data[:, 2], 3))
    spacing_offset = myfit(temperature_ref) - spacing_ref  # T in K, spacing in angstroms
    print('Spacing offset =', str('{:.4f}'.format(spacing_offset)), 'angstroms')

    # correct the platinum reference curve for the offset
    platinum_offset = np.copy(expansion_data)
    platinum_offset[:, 2] = platinum_offset[:, 2] - spacing_offset
    myfit = np.poly1d(np.polyfit(platinum_offset[:, 2], platinum_offset[:, 0], 3))
    mytemp = int(myfit(spacing) - 273.15)
    print('Temperature with offset correction=', mytemp, 'C')
    return mytemp


#############################################################
energy_flag = 0
no_voxel = 0
plt.ion()
print("Scan", scan)
if flatfield_file != "":
    flatfield = np.load(flatfield_file)['flatfield']
    if flatfield.shape[0] > nb_pixel_y:
        flatfield = flatfield[0:nb_pixel_y, :]
else:
    flatfield = None
# old method using xrayutilities
# specfile = xu.io.SPECFile(spec_prefix + '.spec', path=specdir)
# [eta, ccdn, delta, nu] = xu.io.getspec_scan(specfile, scan, 'eta', counter, 'del', 'nu')
specfile = silx.io.open(specdir + spec_prefix + ".spec")
delta = specfile['/' + str(scan) + '.1/instrument/positioners/del'].value  # float
nu = specfile['/' + str(scan) + '.1/instrument/positioners/nu'].value  # float
try:
    rocking_angle = specfile['/' + str(scan) + '.1/measurement/eta'].value  # ndarray if rocking curve
    title = ['Eta rocking curve', 'deg']
    print(title[0])
except KeyError:
    try:
        rocking_angle = specfile['/' + str(scan) + '.1/measurement/phi'].value  # ndarray if rocking curve
        title = ['Phi rocking curve', 'deg']
        print(title[0])
    except KeyError:
        rocking_angle = specfile['/' + str(scan) + '.1/measurement/energy'].value  # ndarray if energy scan
        rocking_angle = rocking_angle * 1000 - 6  # switch to eV, 6eV difference at ID01
        title = ['Energy scan', 'eV']
        energy_flag = 1
        print(title[0])
        try:
            out_of_plane = specfile['/' + str(scan) + '.1/measurement/del'].value  # ndarray if energy scan
            follow_bragg = 1
            print('Following the Bragg peak with delta angle')
        except KeyError:
            floow_Bragg = 0
ccdn = specfile['/' + str(scan) + '.1/measurement/' + counter].value  # ndarray if rocking curve
if filtered_data == 0:
    rawdata = np.zeros((len(ccdn), region[1] - region[0], region[3] - region[2]))
    for idx in range(len(ccdn)):
        i = int(ccdn[idx])
        e = fabio.open(ccdfiletmp % i)
        ccdraw = e.data
        if detector == 0:
            ccdraw, _ = mask_eiger(ccdraw, np.zeros((2164, 1030)))
        elif detector == 1:
            ccdraw, mask = remove_hotpixels_maxipix(ccdraw, np.zeros((516, 516)), hotpixels_file)
            ccdraw, _ = mask_maxipix(ccdraw, mask)
        if flatfield is not None:
            ccdraw = flatfield * ccdraw
        ccd = xu.blockAverage2D(ccdraw, nav[0], nav[1], roi=region)
        rawdata[int(i - ccdn[0]), :, :] = ccd
    rawdata, _ = check_pixels(rawdata, np.zeros(ccdraw.shape))
else:
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(initialdir=specdir + "S" + str(scan) + "pynxraw/",
                                           title="Select 3D data", filetypes=[("NPZ", "*.npz")])
    rawdata = np.load(file_path)['data']
    rawdata = rawdata[region[0]:region[1], region[2]:region[3]]
nz, ny, nx = rawdata.shape
if nz != len(ccdn):
    print('Filtered data has not the same shape as raw data')
    sys.exit()
print("Shape of dataset: ", nz, ny, nx)
if centering == 0:
    z0, y0, x0 = np.unravel_index(abs(rawdata).argmax(), rawdata.shape)
    print("Max at (z, y, x): ", z0, y0, x0, ' Max = ', int(rawdata[z0, y0, x0]))
    print("Image number for max:", int(ccdn[z0]))
elif centering == 1:
    z0, y0, x0 = center_of_mass(rawdata)
    print("Center of mass at (z, y, x): ", z0, y0, x0, ' COM = ', int(rawdata[int(z0), int(y0), int(x0)]))
    print("Image number for COM:", int(ccdn[z0]))
else:
    sys.exit("Incorrect value for 'centering' parameter")
if energy_flag == 1:
    energy = rocking_angle[int(z0)]
    print('Energy for the max / COM: ', energy, ' (offset of 6eV corrected)')
    root = tk.Tk()
    root.withdraw()
    try:
        file_path = filedialog.askopenfilename(title="Select the file containing QxQzQy",
                                               initialdir=datadir+'pynx/', filetypes=[("NPZ", "*.npz")])
        npzfile = np.load(file_path)
        qx = npzfile['qx']  # z in nexus convention
        qy = npzfile['qy']  # x in nexus convention
        qz = npzfile['qz']  # y in nexus convention
    except FileNotFoundError:
        print('Voxel size unknown')
        no_voxel = 1
    if follow_bragg == 1:
        delta = out_of_plane[int(z0)]
        print('Delta for the max / COM: ', delta)
# calculate rocking curve and fit it to get the FWHM
rocking_curve = np.zeros(nz)
if filtered_data == 0:  # take a small ROI to avoid parasitic peaks
    for idx in range(nz):
        rocking_curve[idx] = rawdata[idx, y0 - 20:y0 + 20, x0 - 20:x0 + 20].sum()
    plot_title = title[0] + " for a 40pixels x 40pixels ROI\ncentered on max"
else:  # take the whole detector
    for idx in range(nz):
        rocking_curve[idx] = rawdata[idx, :, :].sum()
    plot_title = title[0] + " (full detector)"
z0 = np.unravel_index(rocking_curve.argmax(), rocking_curve.shape)[0]

if energy_flag == 1:
    print('Need to implement interp1 for energy scans')
    # TODO: remove duplicated values in energy scans
else:
    interpolation = interp1d(rocking_angle, rocking_curve, kind='cubic')
    interp_points = 5*nz
    interp_rocking_angle = np.linspace(rocking_angle.min(), rocking_angle.max(), interp_points)
    interp_curve = interpolation(interp_rocking_angle)
    interp_fwhm = len(np.argwhere(interp_curve >= interp_curve.max()/2)) * \
                  (rocking_angle.max()-rocking_angle.min())/(interp_points-1)
    print('FWHM ' + title[0] + ' by interpolation', str('{:.3f}'.format(interp_fwhm)), title[1])

    intensity = rocking_curve[rocking_curve > rocking_curve.max()/100]
    rocking_angle_axis = rocking_angle[rocking_curve > rocking_curve.max()/100]
    model_Voigt = VoigtModel()
    params_Voigt = model_Voigt.guess(intensity, x=rocking_angle_axis)
    result_Voigt = model_Voigt.fit(intensity, params_Voigt, x=rocking_angle_axis)
    fwhm_Voigt = result_Voigt.params['fwhm'].value
    print('FWHM ' + title[0] + ' (Voigt fit)', str('{:.3f}'.format(fwhm_Voigt)), title[1])

    fig, (ax0, ax1) = plt.subplots(2, 1, sharex='col', figsize=(10, 5))
    ax0.plot(rocking_angle, rocking_curve, '.')
    ax0.plot(interp_rocking_angle, interp_curve)
    ax0.plot(rocking_angle_axis, result_Voigt.best_fit)
    ax0.set_ylabel('Integrated intensity')
    ax0.legend(('data', 'interpolation', 'Voigt'))
    ax0.set_title(plot_title)
    ax1.plot(rocking_angle, np.log10(rocking_curve), '.')
    ax1.plot(interp_rocking_angle, np.log10(interp_curve))
    ax1.plot(rocking_angle_axis, np.log10(result_Voigt.best_fit))
    ax1.set_xlabel(title[0] + title[1])
    ax1.set_ylabel('Log(integrated intensity)')
    ax0.legend(('data', 'interpolation', 'Voigt'))
    plt.pause(0.1)

# Find the Bragg peak
y0, x0 = center_of_mass(rawdata[z0, :, :])
print("Center of mass at (z, y, x): ", np.rint(z0).astype(int), np.rint(y0).astype(int), np.rint(x0).astype(int))
print("Center of mass (full detector) at (z, y, x): ", np.rint(z0).astype(int), np.rint(y0+region[0]).astype(int),
      np.rint(x0+region[2]).astype(int))
print("Image number for COM:", int(ccdn[z0]))

bragg_nu = nu
bragg_del = delta
bragg_x = region[2] + x0  # convert it in full detector pixel
bragg_y = region[0] + y0  # convert it in full detector pixel

x_direct_0 = direct_beam_x - direct_nu*np.pi/180*sdd/pixel_size  # nu is clockwise
y_direct_0 = direct_beam_y - direct_del*np.pi/180*sdd/pixel_size
print("Direct beam at (nu=", str(direct_nu), "del=", str(direct_del), ") = (X, Y): ", direct_beam_x, direct_beam_y)
print("Direct beam at (nu= 0, del= 0) = (X, Y): ", str('{:.2f}'.format(x_direct_0)), str('{:.2f}'.format(y_direct_0)))
print("Bragg peak at (nu=", str(bragg_nu), "del=", str(bragg_del), ") = (X, Y): ",
      str('{:.2f}'.format(bragg_x)), str('{:.2f}'.format(bragg_y)))
nu_bragg = bragg_nu - pixel_size*(bragg_x-x_direct_0)/sdd*180/np.pi  # nu is clockwise
del_bragg = bragg_del - pixel_size*(bragg_y-y_direct_0)/sdd*180/np.pi
print("Bragg angles before correction = (nu, del): ", str('{:.4f}'.format(bragg_nu)), str('{:.4f}'.format(bragg_del)))
print("Bragg angles after correction = (nu, del): ", str('{:.4f}'.format(nu_bragg)), str('{:.4f}'.format(del_bragg)))
d_rocking_angle = rocking_angle[1]-rocking_angle[0]
print("Rocking step=", str('{:.4f}'.format(d_rocking_angle)), title[1])

# wavevector transfer calculations
wavelength = 12.398*1000/energy  # in angstroms
q1 = (np.cos(np.radians(nu_bragg))*np.cos(np.radians(del_bragg)))-1  # z downstream
q2 = np.sin(np.radians(del_bragg))  # y vertical
q3 = -1*np.sin(np.radians(nu_bragg))*np.cos(np.radians(del_bragg))  # x outboard, positive if nu positive
q = 2*np.pi/wavelength*np.array([q1, q2, q3])
Qnorm = np.linalg.norm(q)
dist_plane = 2 * np.pi / Qnorm
print("Wavevector transfer of Bragg peak: ", q, str('{:.4f}'.format(Qnorm)))
print("Interplanar distance: ", str('{:.4f}'.format(dist_plane)), "angstroms")
temperature = bragg_temperature(dist_plane, reflection, spacing_ref=reference_spacing,
                                temperature_ref=reference_temperature, use_q=0, material="Pt")
# calculate voxel sizes
if energy_flag == 0:
    dz_realspace = wavelength / 10 / (nz * d_rocking_angle * np.pi / 180)  # in nm
    dy_realspace = wavelength / 10 * sdd / (ny * pixel_size)  # in nm
    dx_realspace = wavelength / 10 * sdd / (nx * pixel_size)  # in nm
    print('Real space voxel size (z, y, x): ', str('{:.2f}'.format(dz_realspace)), 'nm',
          str('{:.2f}'.format(dy_realspace)), 'nm', str('{:.2f}'.format(dx_realspace)), 'nm')
elif no_voxel == 0:
    dy_realspace = 2 * np.pi / abs(qz.max() - qz.min()) / 10  # in nm   qz=y in nexus convention
    dx_realspace = 2 * np.pi / abs(qy.max() - qy.min()) / 10  # in nm   qy=x in nexus convention
    dz_realspace = 2 * np.pi / abs(qx.max() - qx.min()) / 10  # in nm   qx=z in nexus convention
    print('\nReal space voxel size (z, y, x) using Q values of shape: (', len(qx), len(qz), len(qx), '):\n ',
          str('{:.2f}'.format(dz_realspace)), 'nm',
          str('{:.2f}'.format(dy_realspace)), 'nm', str('{:.2f}'.format(dx_realspace)), 'nm')

# plot image at Bragg condition
data = rawdata[int(round(z0)), :, :]
plt.figure()
plt.imshow(np.log10(abs(data)), vmin=0, vmax=5)
plt.title('Central slice at frame '+str(int(np.rint(z0))))
plt.colorbar()
plt.ioff()
plt.show()
#################################################################