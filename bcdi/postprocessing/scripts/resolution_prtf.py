# -*- coding: utf-8 -*-
"""
Calculate the resolution of a CDI reconstruction using the phase retrieval transfer function (PRTF)
Can load several reconstructions (given that the 3D array shape and the voxel size are identical) and calculate
the PRTF of the ensemble.

The measured diffraction pattern and reconstructions can be either in the detector frame
or already orthogonalized (_LAB.npz, before rotations).

For the laboratory frame, the CXI convention is used: z downstream, y vertical, x outboard
For q, the usual convention is used: qx downstream, qz vertical, qy outboard

created on 18/02/2019
@author: Carnis Jerome @ IM2NP / ESRF ID01
"""
import numpy as np
from numpy.fft import fftn, fftshift, ifftn, ifftshift
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage.measurements import center_of_mass
import tkinter as tk
from tkinter import filedialog
import gc
from scipy.interpolate import RegularGridInterpolator
import h5py
import xrayutilities as xu
from silx.io.specfile import SpecFile
from scipy.interpolate import interp1d
import sys
sys.path.append('C:/Users/carnis/Work Folders/Documents/myscripts/utils/')
import image_registration as reg
scan = 2191  # spec scan number
datadir = "C:/Users/carnis/Work Folders/Documents/data/CH4760_Pt/S"+str(scan)+"/pynxraw/no_apodization/avg1/"
savedir = "C:/Users/carnis/Work Folders/Documents/data/CH4760_Pt/S"+str(scan)+"/pynxraw/no_apodization/test/"  # no_apodization"  # apodize_during_phasing # apodize_postprocessing
comment = "_test"  # should start with _

# ################ geometry ###################
sdd = 0.50678  # 1.0137  # sample to detector distance in m
en = 9000.0 - 6   # x-ray energy in eV, 6eV offset at ID01
setup = "ID01"  # only "ID01"
rocking_angle = "outofplane"  # "outofplane" or "inplane"
outofplane_angle = 35.3440  # detector delta ID01
inplane_angle = -0.9265  # detector nu ID01
grazing_angle = 0  # in degrees, incident angle for in-plane rocking curves (eta ID01)
tilt_angle = 0.010155  # angular step size for rocking angle, eta ID01
pixel_size = 55e-6  # detector pixel size in m
# ############### Q conversion ##############
beam_direction = [1, 0, 0]  # beam along x
sample_inplane = [1, 0, 0]  # sample inplane reference direction along the beam at 0 angles
sample_outofplane = [0, 0, 1]  # surface normal of the sample at 0 angles
if setup == 'ID01':
    offsets = (0, 0, 0, 0, 0)  # eta chi phi nu del
    qconv = xu.experiment.QConversion(['y-', 'x+', 'z-'], ['z-', 'y-'], r_i=beam_direction)  # for ID01
    # 2S+2D goniometer (ID01 goniometer, sample: eta, chi, phi      detector: nu,del
    # the vector beam_direction is giving the direction of the primary beam
    # convention for coordinate system: x downstream; z upwards; y to the "outside" (right-handed)
    hxrd = xu.experiment.HXRD(sample_inplane, sample_outofplane, qconv=qconv)  # x downstream, y outboard, z vertical
    # first two arguments in HXDD are the inplane reference direction along the beam and surface normal of the sample

# ############ options ######################
modes = False  # set to True when the solution is the first mode - then the intensity needs to be normalized
simulation = False  # True is this is simulated data, will not load the specfile
if simulation:
    tilt_bragg = 17.1177  # value of the tilt angle at Bragg peak (eta at ID01), only needed for simulations
debug = False  # True to show more plots
save = True  # True to save the prtf figure
ortho_frame = False  # True to work in the laboratory frame (orthogonal). The diffraction pattern will be interpolated
#  onto an orthogonal grid using xrayutilities
#############################################
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
my_cmap = LinearSegmentedColormap('my_colormap', cdict, 256)
figure_size = (13, 9)  # in inches
#############################################################


def align_diffpattern(ref_pattern, mypattern, myprecision=10, debugging=0):
    """
    Align diffraction pattern, not very satisfying at that time.
    :param ref_pattern:
    :param mypattern:
    :param myprecision:
    :param debugging:
    :return:
    """
    nbz, nby, nbx = ref_pattern.shape
    if debugging:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(np.log10(abs(ref_pattern).sum(axis=0)))
        plt.title('np.log10(abs(ref_pattern).sum(axis=0))')
        plt.subplot(1, 2, 2)
        plt.imshow(np.log10(abs(mypattern).sum(axis=0)))
        plt.title('np.log10(abs(mypattern).sum(axis=0))')
        plt.pause(0.1)
    myshiftz, myshifty, myshiftx = reg.getimageregistration(mypattern, ref_pattern, precision=myprecision)
    zcom, ycom, xcom = center_of_mass(abs(mypattern)**4)
    print('COM before subpixel shift', zcom, ',', ycom, ',', xcom)

    # phase shift in real space
    buf2ft = fftn(mypattern)  # in real space
    del mypattern
    gc.collect()

    z_axis = ifftshift(np.arange(-np.fix(nbz/2), np.ceil(nbz/2), 1))
    y_axis = ifftshift(np.arange(-np.fix(nby/2), np.ceil(nby/2), 1))
    x_axis = ifftshift(np.arange(-np.fix(nbx/2), np.ceil(nbx/2), 1))
    z_axis, y_axis, x_axis = np.meshgrid(z_axis, y_axis, x_axis, indexing='ij')
    greg = buf2ft * np.exp(1j * 2 * np.pi * (myshiftz * z_axis / nbz + myshifty * y_axis / nby + myshiftx * x_axis / nbx))
    del buf2ft, z_axis, y_axis, x_axis
    gc.collect()

    mypattern = abs(ifftn(greg))  # the intensity is a real number
    del greg
    gc.collect()
    # end of phase shift in real space

    if debugging:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(np.log10(abs(ref_pattern).sum(axis=0)))
        plt.title('np.log10(abs(ref_pattern).sum(axis=0))')
        plt.subplot(1, 2, 2)
        plt.imshow(np.log10(abs(mypattern).sum(axis=0)))
        plt.title('centered np.log10(abs(my_fft)).sum(axis=0)')
        plt.pause(0.1)

    print('COM after subpixel shift', center_of_mass(abs(mypattern) ** 4))
    return mypattern


def crop_pad(myobj, myshape, debugging=0):
    """
    will crop or pad my obj depending on myshape
    :param myobj: 3d complex array to be padded
    :param myshape: list of desired output shape [z, y, x]
    :param debugging: to plot myobj before and after rotation
    :return: myobj padded with zeros
    """
    nbz, nby, nbx = myobj.shape
    newz, newy, newx = myshape
    if debugging == 1:
        plt.figure(figsize=figure_size)
        plt.subplot(2, 2, 1)
        plt.imshow(abs(myobj)[:, :, nbx // 2], vmin=0, vmax=1)
        plt.colorbar()
        plt.axis('scaled')
        plt.title("Middle slice in YZ before padding")
        plt.subplot(2, 2, 2)
        plt.imshow(abs(myobj)[:, nby // 2, :], vmin=0, vmax=1)
        plt.colorbar()
        plt.title("Middle slice in XZ before padding")
        plt.axis('scaled')
        plt.subplot(2, 2, 3)
        plt.imshow(abs(myobj)[nbz // 2, :, :], vmin=0, vmax=1)
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.title("Middle slice in XY before padding")
        plt.axis('scaled')
        plt.pause(0.1)
    # z
    if newz >= nbz:  # pad
        temp_z = np.zeros((myshape[0], nby, nbx), dtype=myobj.dtype)
        temp_z[(newz - nbz) // 2:(newz + nbz) // 2, :, :] = myobj
    else:  # crop
        temp_z = myobj[(nbz - newz) // 2:(newz + nbz) // 2, :, :]
    # y
    if newy >= nby:  # pad
        temp_y = np.zeros((newz, newy, nbx), dtype=myobj.dtype)
        temp_y[:, (newy - nby) // 2:(newy + nby) // 2, :] = temp_z
    else:  # crop
        temp_y = temp_z[:, (nby - newy) // 2:(newy + nby) // 2, :]
    # x
    if newx >= nbx:  # pad
        newobj = np.zeros((newz, newy, newx), dtype=myobj.dtype)
        newobj[:, :, (newx - nbx) // 2:(newx + nbx) // 2] = temp_y
    else:  # crop
        newobj = temp_y[:, :, (nbx - newx) // 2:(newx + nbx) // 2]

    if debugging == 1:
        plt.figure(figsize=figure_size)
        plt.subplot(2, 2, 1)
        plt.imshow(abs(newobj)[:, :, newx // 2], vmin=0, vmax=1)
        plt.colorbar()
        plt.axis('scaled')
        plt.title("Middle slice in YZ after padding")
        plt.subplot(2, 2, 2)
        plt.imshow(abs(newobj)[:, newy // 2, :], vmin=0, vmax=1)
        plt.colorbar()
        plt.title("Middle slice in XZ after padding")
        plt.axis('scaled')
        plt.subplot(2, 2, 3)
        plt.imshow(abs(newobj)[newz // 2, :, :], vmin=0, vmax=1)
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.title("Middle slice in XY after padding")
        plt.axis('scaled')
        plt.pause(0.1)
    return newobj


def q_values(my_shape, my_distance, my_nrj, mydetector_angles, my_tilt, myrocking_angle, mygrazing_angle, mypixel_size):
    """
    calculate q-values of diffracion pattern voxels in the detector frame
    :param my_shape: shape of the diffraction pattern
    :param my_distance: sample to detector distance in m
    :param my_nrj: energy in ev
    :param mydetector_angles: tuple of (outofplane_angle, inplane_angle)
    :param my_tilt: angular step during the rocking curve, in degrees
    :param myrocking_angle: name of the motor which is tilted during the rocking curve
    :param mygrazing_angle: in degrees, incident angle for in-plane rocking curves (eta ID01, th 34ID, beta SIXS)
    :param mypixel_size: in meters
    :return: the corresponding q-values as a 3d array of the same shape
    """
    nbz, nby, nbx = my_shape
    wavelength = 12.398 * 1e-7 / my_nrj  # in m

    transfer_matrix = reciprocal_basis(wavelength=wavelength, outofplane=mydetector_angles[0],
                                       inplane=mydetector_angles[1], tilt=my_tilt, geometry='ID01',
                                       myrocking_angle=myrocking_angle, mygrazing_angle=mygrazing_angle,
                                       distance=my_distance, pixel_x=mypixel_size, pixel_y=mypixel_size)
    myz, myy, myx = np.meshgrid(np.arange(0, nbz, 1), np.arange(0, nby, 1), np.arange(0, nbx, 1), indexing='ij')
    out_x = transfer_matrix[0, 0] * myx + transfer_matrix[0, 1] * myy + transfer_matrix[0, 2] * myz
    out_y = transfer_matrix[1, 0] * myx + transfer_matrix[1, 1] * myy + transfer_matrix[1, 2] * myz
    out_z = transfer_matrix[2, 0] * myx + transfer_matrix[2, 1] * myy + transfer_matrix[2, 2] * myz
    return out_x, out_y, out_z


def reciprocal_basis(wavelength, outofplane, inplane, tilt, myrocking_angle, mygrazing_angle, distance,
                     pixel_x, pixel_y, geometry):
    """
    calculate the basis matrix in reciprocal space
    :param wavelength: in m
    :param outofplane: in degrees
    :param inplane: in degrees  (also called inplane_angle depending on the diffractometer)
    :param tilt: angular step during the rocking curve, in degrees
    :param myrocking_angle: name of the motor which is tilted during the rocking curve
    :param mygrazing_angle: in degrees, incident angle for in-plane rocking curves (eta ID01, th 34ID, beta SIXS)
    :param distance: sample to detector distance, in meters
    :param pixel_x: horizontal pixel size, in meters
    :param pixel_y: vertical pixel size, in meters
    :param geometry: name of the setup 'ID01'or 'SIXS' or '34ID' or 'P10' or 'CRISTAL'
    :return: basis matrix 3x3
    """
    wavelength = wavelength * 1e9  # convert to nm
    distance = distance * 1e9  # convert to nm
    lambdaz = wavelength * distance
    pixel_x = pixel_x * 1e9  # convert to nm
    pixel_y = pixel_y * 1e9  # convert to nm
    mymatrix = np.zeros((3, 3))
    outofplane = np.radians(outofplane)
    inplane = np.radians(inplane)
    tilt = np.radians(tilt)
    mygrazing_angle = np.radians(mygrazing_angle)

    if geometry == 'ID01':
        print('using ESRF ID01 PSIC geometry')
        if myrocking_angle == "outofplane":
            print('rocking angle is eta')
            # rocking eta angle clockwise around x (phi does not matter, above eta)
            mymatrix[:, 0] = 2*np.pi / lambdaz * np.array([pixel_x*np.cos(inplane),
                                                           0,
                                                           pixel_x*np.sin(inplane)])
            mymatrix[:, 1] = 2*np.pi / lambdaz * np.array([-pixel_y*np.sin(inplane)*np.sin(outofplane),
                                                           -pixel_y*np.cos(outofplane),
                                                           pixel_y*np.cos(inplane)*np.sin(outofplane)])
            mymatrix[:, 2] = 2*np.pi / lambdaz * np.array([0,
                                                           tilt*distance*(1-np.cos(inplane)*np.cos(outofplane)),
                                                           tilt*distance*np.sin(outofplane)])
        elif myrocking_angle == "inplane" and mygrazing_angle == 0:
            print('rocking angle is phi, eta=0')
            # rocking phi angle clockwise around y, assuming incident angle eta is zero (eta below phi)
            mymatrix[:, 0] = 2*np.pi / lambdaz * np.array([pixel_x*np.cos(inplane),
                                                           0,
                                                           pixel_x*np.sin(inplane)])
            mymatrix[:, 1] = 2*np.pi / lambdaz * np.array([-pixel_y*np.sin(inplane)*np.sin(outofplane),
                                                           -pixel_y*np.cos(outofplane),
                                                           pixel_y*np.cos(inplane)*np.sin(outofplane)])
            mymatrix[:, 2] = 2*np.pi / lambdaz * np.array([-tilt*distance*(1-np.cos(inplane)*np.cos(outofplane)),
                                                           0,
                                                           tilt*distance*np.sin(inplane)*np.cos(outofplane)])
        elif myrocking_angle == "inplane" and mygrazing_angle != 0:
            print('rocking angle is phi, with eta non zero')
            # rocking phi angle clockwise around y, incident angle eta is non zero (eta below phi)
            mymatrix[:, 0] = 2*np.pi / lambdaz * np.array([pixel_x*np.cos(inplane),
                                                           0,
                                                           pixel_x*np.sin(inplane)])
            mymatrix[:, 1] = 2*np.pi / lambdaz * np.array([-pixel_y*np.sin(inplane)*np.sin(outofplane),
                                                           -pixel_y*np.cos(outofplane),
                                                           pixel_y*np.cos(inplane)*np.sin(outofplane)])
            mymatrix[:, 2] = 2*np.pi / lambdaz * tilt * distance * \
                             np.array([(np.sin(mygrazing_angle)*np.sin(outofplane) +
                                      np.cos(mygrazing_angle)*(np.cos(inplane)*np.cos(outofplane)-1)),
                                      np.sin(mygrazing_angle)*np.sin(inplane)*np.sin(outofplane),
                                      np.cos(mygrazing_angle)*np.sin(inplane)*np.cos(outofplane)])

    return mymatrix


######################################
plt.ion()
root = tk.Tk()
root.withdraw()

######################################
# load experimental data and mask
######################################
file_path = filedialog.askopenfilename(initialdir=datadir, title="Select diffraction pattern",
                                       filetypes=[("NPZ", "*.npz")])
npzfile = np.load(file_path)
diff_pattern = npzfile['data']
diff_pattern = diff_pattern.astype(float)


numz, numy, numx = diff_pattern.shape
print('Measured data shape =', numz, numy, numx)

max_fft = np.sqrt(diff_pattern).max()
print('Max(measured amplitude)=', max_fft)

file_path = filedialog.askopenfilename(initialdir=datadir, title="Select mask", filetypes=[("NPZ", "*.npz")])
npzfile = np.load(file_path)
mask = npzfile['mask']
diff_pattern[np.nonzero(mask)] = 0

z0, y0, x0 = center_of_mass(diff_pattern)
print("COM of measured pattern after masking: ", z0, y0, x0)
z0, y0, x0 = [int(z0), int(y0), int(x0)]
print('Number of unmasked photons =', diff_pattern.sum())

plt.figure()
plt.imshow(np.log10(np.sqrt(diff_pattern).sum(axis=0)), cmap=my_cmap, vmin=0, vmax=3.5)
plt.title('abs(diffraction amplitude).sum(axis=0)')
plt.colorbar()
plt.pause(0.1)

#######################################
# calculate q matrix respective to the COM
#######################################
# TODO: check the function q-values based on Vincent's calculations
# qx, qz, qy = q_values(my_shape=diff_pattern.shape, my_distance=sdd, my_nrj=en,
#                       mydetector_angles=(outofplane_angle, inplane_angle), my_tilt=tilt_angle,
#                       myrocking_angle=rocking_angle, mygrazing_angle=grazing_angle, mypixel_size=pixel_size)


hxrd.Ang2Q.init_area('z-', 'y+', cch1=int(y0), cch2=int(x0), Nch1=numy, Nch2=numx, pwidth1=55e-6, pwidth2=55e-6,
                     distance=sdd)
# first two arguments in init_area are the direction of the detector
if simulation:
    eta = tilt_bragg + tilt_angle * (np.arange(0, numz, 1) - int(z0))
else:
    file_path = filedialog.askopenfilename(initialdir=datadir, title="Select spec file", filetypes=[("SPEC", "*.spec")])
    spec_file = SpecFile(file_path)
    labels = spec_file[str(scan) + '.1'].labels  # motor scanned
    labels_data = spec_file[str(scan) + '.1'].data  # motor scanned
    eta = labels_data[labels.index('eta'), :]
    if eta.size < numz:  # data has been padded, we suppose it is centered in z dimension
        pad_low = int((numz - eta.size + ((numz - eta.size) % 2)) / 2)
        pad_high = int((numz - eta.size + 1) / 2 - ((numz - eta.size) % 2))
        eta = np.concatenate((eta[0] + np.arange(-pad_low, 0, 1) * tilt_angle,
                              eta,
                              eta[-1] + np.arange(1, pad_high+1, 1) * tilt_angle), axis=0)


myqx, myqy, myqz = hxrd.Ang2Q.area(eta, 0, 0, inplane_angle, outofplane_angle, delta=(0, 0, 0, 0, 0))
if debug:
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(myqz[0, :, :])
    plt.colorbar()
    plt.title('qz first frame')
    plt.subplot(1, 3, 2)
    plt.imshow(myqy[0, :, :])
    plt.colorbar()
    plt.title('qy first frame')
    plt.subplot(1, 3, 3)
    plt.imshow(myqx[:, :, 0])
    plt.colorbar()
    plt.title('qx first frame')
    plt.pause(0.1)
if ortho_frame:
    mygridder = xu.Gridder3D(numz, numy, numx)
    mygridder(myqx, myqz, myqy, diff_pattern)
    diff_pattern = mygridder.data
    myqx, myqz, myqy = np.meshgrid(mygridder.xaxis, mygridder.yaxis, mygridder.zaxis, indexing='ij')
    z0, y0, x0 = center_of_mass(diff_pattern)
    print("Center of mass after orthogonalization at (z, y, x): ", z0, y0, x0)
    z0, y0, x0 = [int(z0), int(y0), int(x0)]
    max_fft = np.sqrt(diff_pattern).max()
    print('Max(measured amplitude) after orthogonalization=', max_fft)

qxCOM = myqx[z0, y0, x0]
qyCOM = myqy[z0, y0, x0]
qzCOM = myqz[z0, y0, x0]
print('COM[qx, qy, qz] = ', qxCOM, qyCOM, qzCOM)
distances_q = np.sqrt((myqx - qxCOM)**2 + (myqy - qyCOM)**2 + (myqz - qzCOM)**2)
del myqx, myqy, myqz
gc.collect()
# if reconstructions are centered and of same shape q values will be identical

if debug:
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(distances_q[numz//2, :, :])
    plt.colorbar()
    plt.title('distances_q at numz/2')
    plt.subplot(1, 3, 2)
    plt.imshow(distances_q[:, numy//2, :])
    plt.colorbar()
    plt.title('distances_q at numy/2')
    plt.subplot(1, 3, 3)
    plt.imshow(distances_q[:, :, numx//2])
    plt.colorbar()
    plt.title('distances_q at numx/2')
    plt.pause(0.1)

######################################
# load and average reconstructed objects
######################################
file_path = filedialog.askopenfilenames(initialdir=datadir,  title="Select reconstructions (prtf)",
                                        filetypes=[("NPZ", "*.npz"), ("NPY", "*.npy"),
                                                   ("CXI", "*.cxi"), ("HDF5", "*.h5")])
summed_fft = np.zeros(diff_pattern.shape)
nbfiles = len(file_path)
if file_path[0].lower().endswith('.npz'):
    ext = '.npz'
elif file_path[0].lower().endswith('.npy'):
    ext = '.npy'
elif file_path[0].lower().endswith('.cxi'):
    ext = '.cxi'
elif file_path[0].lower().endswith('.h5'):
    ext = '.h5'
    comment = comment + "_1stmode"
else:
    sys.exit('wrong file format')

for ii in range(nbfiles):
    if ext == '.npz':
        npzfile = np.load(file_path[ii])
        if ortho_frame:
            amp = npzfile['amp']
            phase = npzfile['phase']
            obj = amp * np.exp(1j * phase)
            del amp, phase
            gc.collect()
        else:
            obj = npzfile['obj']
    elif ext == '.npy':
        obj = np.load(file_path[ii])
    elif ext == '.cxi':
        h5file = h5py.File(file_path[ii], 'r')
        group_key = list(h5file.keys())[1]
        subgroup_key = list(h5file[group_key])
        obj = h5file['/' + group_key + '/' + subgroup_key[0] + '/data'].value
    elif file_path[0].lower().endswith('.h5'):  # modes.h5
        ext = '.h5'
        h5file = h5py.File(file_path[0], 'r')
        group_key = list(h5file.keys())[0]
        subgroup_key = list(h5file[group_key])
        obj = h5file['/' + group_key + '/' + subgroup_key[0] + '/data'].value[0]
    else:
        sys.exit('Wrong file format')

    print('Opening ', file_path[ii])
    # check if the shape is the same as the measured diffraction pattern
    if obj.shape != diff_pattern.shape:
        print('Reconstructed object shape different from the experimental diffraction pattern: crop/pad')
        obj = crop_pad(myobj=obj, myshape=diff_pattern.shape, debugging=0)

    # calculate the retrieved diffraction amplitude
    my_fft = fftshift(fftn(obj)) / (np.sqrt(numz)*np.sqrt(numy)*np.sqrt(numx))  # complex amplitude
    del obj
    gc.collect()
    print('Max(retrieved amplitude) =', abs(my_fft).max())
    # if modes:  # if this is the first mode, intensity should be normalized to the measured diffraction pattern
    #     max_fft = 545
    #     my_fft = my_fft * max_fft / abs(my_fft).max()
    #     print('Max(retrieved amplitude) after modes normalization =', abs(my_fft).max())  # needed for modes

    plt.figure()
    plt.imshow(np.log10(abs(my_fft).sum(axis=0)), cmap=my_cmap, vmin=0, vmax=3.5)
    plt.title('abs(retrieved amplitude).sum(axis=0)')
    plt.colorbar()
    plt.pause(0.1)

    # sum to the average retrieved diffraction pattern
    summed_fft = summed_fft + my_fft
    del my_fft
    gc.collect()
summed_fft = summed_fft / nbfiles  # normalize for the number of reconstructions
summed_fft[np.nonzero(mask)] = 0  # do not take mask voxels into account
print('Max(average retrieved amplitude) =', abs(summed_fft).max())
print('COM of the average retrieved diffraction pattern after masking: ', center_of_mass(abs(summed_fft)))
del mask
gc.collect()
############################
# if working is the orthogonal frame
# need to align reconstructions with the diffraction pattern because of phase ramp removal
# TODO: make this work, not possible to align correctly diff_pattern and  FFT(recontruction) at the moment
############################
if ortho_frame:
    print('image registration before subpixel shift', reg.getimageregistration(diff_pattern, summed_fft, precision=100))

    summed_fft = align_diffpattern(diff_pattern, summed_fft, myprecision=100, debugging=1)

    print('image registration after subpixel shift', reg.getimageregistration(diff_pattern, summed_fft, precision=100))
    # shiftz, shifty, shiftx = reg.getimageregistration(diff_pattern, summed_fft, precision=100)
    # shiftz, shifty, shiftx = int(np.rint(shiftz)), int(np.rint(shifty)), int(np.rint(shiftx))
    # summed_fft = np.roll(summed_fft, (shiftz, shifty, shiftx), axis=(0, 1, 2))
    # # old_z = np.arange(-numz // 2, numz // 2)
    # # old_y = np.arange(-numy // 2, numy // 2)
    # # old_x = np.arange(-numx // 2, numx // 2)
    # # myz, myy, myx = np.meshgrid(old_z, old_y, old_x, indexing='ij')
    # # new_z = myz - shiftz
    # # new_y = myy - shifty
    # # new_x = myx - shiftx
    # # del myx, myy, myz
    # # rgi = RegularGridInterpolator((old_z, old_y, old_x), summed_fft, method='linear', bounds_error=False,
    # #                               fill_value=0)
    # # summed_fft = rgi(np.concatenate((new_z.reshape((1, new_z.size)), new_y.reshape((1, new_z.size)),
    # #                                 new_x.reshape((1, new_z.size)))).transpose())
    # # summed_fft = summed_fft.reshape((numz, numy, numx)).astype(diff_pattern.dtype)
    # print('image registration after RegularGridInterpolator', reg.getimageregistration(diff_pattern, summed_fft, precision=100))
    # #
    print('Max(retrieved amplitude) after alignment =', abs(summed_fft).max())

plt.figure()
plt.subplot(2, 3, 1)
plt.imshow(np.log10(abs(diff_pattern[numz//2, :, :])))
plt.title('diff_pattern @ numz/2')
plt.subplot(2, 3, 2)
plt.imshow(np.log10(abs(diff_pattern[:, numy//2, :])))
plt.title('diff_pattern @ numy/2')
plt.subplot(2, 3, 3)
plt.imshow(np.log10(abs(diff_pattern[:, :, numx//2])))
plt.title('diff_pattern @ numx/2')
plt.subplot(2, 3, 4)
plt.imshow(np.log10(abs(summed_fft[numz//2, :, :])), vmin=-1)
plt.title('np.log10(abs(summed_fft).sum(axis=0))')
plt.subplot(2, 3, 5)
plt.imshow(np.log10(abs(summed_fft[:, numy//2, :])), vmin=-1)
plt.title('summed_fft @ numy/2')
plt.subplot(2, 3, 6)
plt.imshow(np.log10(abs(summed_fft[:, :, numx//2])), vmin=-1)
plt.title('summed_fft @ numx/2')
plt.pause(0.1)
############################
# calculate retrieved / measured
############################
diff_pattern[diff_pattern == 0] = np.nan  # discard zero valued pixels
prtf_matrix = abs(summed_fft) / np.sqrt(diff_pattern)
plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(prtf_matrix[numz // 2, :, :], cmap=my_cmap, vmin=0, vmax=1.1)
plt.title('prtf_matrix @ numz/2')
plt.subplot(1, 3, 2)
plt.imshow(prtf_matrix[:, numy // 2, :], cmap=my_cmap, vmin=0, vmax=1.1)
plt.title('prtf_matrix @ numy/2')
plt.subplot(1, 3, 3)
plt.imshow(prtf_matrix[:, :, numx // 2], cmap=my_cmap, vmin=0, vmax=1.1)
plt.title('prtf_matrix @ numx/2')
plt.colorbar()
plt.pause(0.1)

############################
# average over spherical shells
############################
print('Distance max:', distances_q.max(), ' (1/A) at: ', np.unravel_index(abs(distances_q).argmax(), distances_q.shape))
nb_bins = numz // 3
prtf_avg = np.zeros(nb_bins)
dq = distances_q.max() / nb_bins  # in 1/A
q_axis = np.linspace(0, distances_q.max(), endpoint=True, num=nb_bins+1)  # in 1/A

for index in range(nb_bins):
    logical_array = np.logical_and((distances_q < q_axis[index+1]), (distances_q >= q_axis[index]))
    temp = prtf_matrix[logical_array]
    prtf_avg[index] = temp[~np.isnan(temp)].mean()
q_axis = q_axis[:-1]

if modes:
    print('Normalizing the PRTF to 1 ...')
    prtf_avg = prtf_avg / prtf_avg[~np.isnan(prtf_avg)].max()  # normalize to 1

############################
# plot and save the PRTF
############################
defined_q = 10 * q_axis[~np.isnan(prtf_avg)]

# create a new variable 'arc_length' to predict q and prtf parametrically (because prtf is not monotonic)
arc_length = np.concatenate((np.zeros(1),
                             np.cumsum(np.diff(prtf_avg[~np.isnan(prtf_avg)])**2 + np.diff(defined_q)**2)),
                            axis=0)  # cumulative linear arc length, used as the parameter
arc_length_interp = np.linspace(0, arc_length[-1], 10000)
fit_prtf = interp1d(arc_length, prtf_avg[~np.isnan(prtf_avg)], kind='linear')
prtf_interp = fit_prtf(arc_length_interp)
idx_resolution = [i for i, x in enumerate(prtf_interp) if x < 1/np.e]  # indices where prtf < 1/e

fit_q = interp1d(arc_length, defined_q, kind='linear')
q_interp = fit_q(arc_length_interp)

plt.figure()
plt.plot(prtf_avg[~np.isnan(prtf_avg)], defined_q, 'o', prtf_interp, q_interp, '.r')
plt.xlabel('PRTF')
plt.ylabel('q (1/nm)')

try:
    q_resolution = q_interp[min(idx_resolution)]
except ValueError:
    print('Resolution limited by the 1 photon counts only (min(prtf)>1/e)')
    print('min(PRTF) = ', prtf_avg[~np.isnan(prtf_avg)].min())
    q_resolution = 10 * q_axis[len(prtf_avg[~np.isnan(prtf_avg)])-1]
print('q resolution =', str('{:.5f}'.format(q_resolution)), ' (1/nm)')
print('resolution d= ' + str('{:.3f}'.format(2*np.pi / q_resolution)) + 'nm')

fig = plt.figure()
plt.plot(defined_q, prtf_avg[~np.isnan(prtf_avg)], 'o')  # q_axis in 1/nm
plt.title('PRTF')
plt.xlabel('q (1/nm)')
plt.ylim(0, 1.1)
fig.text(0.15, 0.25, "Scan " + str(scan) + comment, size=14)
fig.text(0.15, 0.20, "q at PRTF=1/e: " + str('{:.5f}'.format(q_resolution)) + '(1/nm)', size=14)
fig.text(0.15, 0.15, "resolution d= " + str('{:.3f}'.format(2*np.pi / q_resolution)) + 'nm', size=14)
if save:
    plt.savefig(savedir + 'S' + str(scan) + '_prtf' + comment + '.png')
plt.ioff()
plt.show()
