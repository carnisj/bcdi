# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#       authors:
#         Jerome Carnis, jerome.carnis@esrf.fr

import numpy as np
from numpy.fft import fftn, fftshift, ifftn, ifftshift
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage.measurements import center_of_mass
import tkinter as tk
from tkinter import filedialog
from scipy.interpolate import RegularGridInterpolator
import h5py
import xrayutilities as xu
from silx.io.specfile import SpecFile
from scipy.interpolate import interp1d
import gc
import sys
sys.path.append('C:\\Users\\carnis\\Work Folders\\Documents\\myscripts\\bcdi\\')
import bcdi.graph.graph_utils as gu
import bcdi.experiment.experiment_utils as exp
import bcdi.preprocessing.preprocessing_utils as pru
import bcdi.postprocessing.postprocessing_utils as pu

helptext = """
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

scan = 2191
datadir = "C:/Users/carnis/Work Folders/Documents/data/CH4760_Pt/S"+str(scan)+"/pynxraw/"
savedir = datadir
comment = "_test"  # should start with _

############################
# beamline parameters #
############################
beamline = 'ID01'  # 'ID01' or 'SIXS' or 'CRISTAL' or 'P10', used for data loading and normalization by monitor
rocking_angle = "outofplane"  # "outofplane" or "inplane"
outofplane_angle = 35.344  # detector out-of-plane angle
inplane_angle = -0.9265  # detector inplane angle
grazing_angle = 0  # in degrees, incident angle for inplane rocking curves
tilt_angle = 0.01015  # angular step size for the rocking angle
detector = "Maxipix"    # "Eiger2M" or "Maxipix" or "Eiger4M"
################################################################################
# parameters for calculating q values #
################################################################################
sdd = 0.50678  # sample to detector distance in m
energy = 9000 - 6   # x-ray energy in eV, 6eV offset at ID01
beam_direction = (1, 0, 0)  # beam along x
sample_inplane = (1, 0, 0)  # sample inplane reference direction along the beam at 0 angles
sample_outofplane = (0, 0, 1)  # surface normal of the sample at 0 angles

if beamline == 'ID01':
    offsets = (0, 0, 0, 0, 0)  # eta chi phi nu del
    qconv = xu.experiment.QConversion(['y-', 'x+', 'z-'], ['z-', 'y-'], r_i=beam_direction)  # for ID01
    # 2S+2D goniometer (ID01 goniometer, sample: eta, chi, phi      detector: nu,del
    # the vector beam_direction is giving the direction of the primary beam
    # convention for coordinate system: x downstream; z upwards; y to the "outside" (right-handed)
    hxrd = xu.experiment.HXRD(sample_inplane, sample_outofplane, qconv=qconv)  # x downstream, y outboard, z vertical
    # first two arguments in HXDD are the inplane reference direction along the beam and surface normal of the sample

###########
# options #
###########
modes = False  # set to True when the solution is the first mode - then the intensity needs to be normalized
simulation = False  # True is this is simulated data, will not load the specfile
if simulation:
    tilt_bragg = 17.1177  # value of the tilt angle at Bragg peak (eta at ID01), only needed for simulations
debug = True  # True to show more plots
save = True  # True to save the prtf figure
##########################
# end of user parameters #
##########################

#######################
# Initialize detector #
#######################
detector = exp.Detector(name=detector)

####################
# Initialize setup #
####################
setup = exp.SetupPreprocessing(beamline=beamline, energy=energy, rocking_angle=rocking_angle, distance=sdd,
                               grazing_angle=grazing_angle, beam_direction=beam_direction,
                               sample_inplane=sample_inplane, sample_outofplane=sample_outofplane)

###################
# define colormap #
###################
colormap = gu.Colormap()
my_cmap = colormap.cmap

###################################
# load experimental data and mask #
###################################
plt.ion()
root = tk.Tk()
root.withdraw()
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

################################################
# calculate the q matrix respective to the COM #
################################################
hxrd.Ang2Q.init_area('z-', 'y+', cch1=int(y0), cch2=int(x0), Nch1=numy, Nch2=numx, pwidth1=detector.pixelsize,
                     pwidth2=detector.pixelsize, distance=setup.distance)
# first two arguments in init_area are the direction of the detector

if simulation:  # need to give the value of eta angle at the Bragg peak
    eta = tilt_bragg + tilt_angle * (np.arange(0, numz, 1) - int(z0))
else:  # find the value of the incident angle at the Bragg peak in the log file
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
    gu.combined_plots(tuple_array=(myqz, myqy, myqx), tuple_sum_frames=False, tuple_sum_axis=(0, 1, 2),
                      tuple_width_v=np.nan, tuple_width_h=np.nan, tuple_colorbar=True, tuple_vmin=np.nan,
                      tuple_vmax=np.nan, tuple_title=('qz', 'qy', 'qx'), tuple_scale='linear')

qxCOM = myqx[z0, y0, x0]
qyCOM = myqy[z0, y0, x0]
qzCOM = myqz[z0, y0, x0]
print('COM[qx, qy, qz] = ', qxCOM, qyCOM, qzCOM)
distances_q = np.sqrt((myqx - qxCOM)**2 + (myqy - qyCOM)**2 + (myqz - qzCOM)**2)  # if reconstructions are centered
#  and of the same shape q values will be identical

del myqx, myqy, myqz
gc.collect()

if debug:
    gu.multislices_plot(distances_q, sum_frames=False, plot_colorbar=True, cmap=my_cmap,
                        title='distances_q', scale='linear', invert_yaxis=False, vmin=np.nan, vmax=np.nan,
                        reciprocal_space=True)

#############################
# load reconstructed object #
#############################
file_path = filedialog.askopenfilename(initialdir=datadir,  title="Select reconstructions (prtf)",
                                       filetypes=[("NPZ", "*.npz"), ("NPY", "*.npy"),
                                                  ("CXI", "*.cxi"), ("HDF5", "*.h5")])

obj, extension = pu.load_reconstruction(file_path)
print('Opening ', file_path)

if extension == '.h5':
    comment = comment + '_mode'

# check if the shape is the same as the measured diffraction pattern
if obj.shape != diff_pattern.shape:
    print('Reconstructed object shape different from the experimental diffraction pattern: crop/pad')
    obj = pu.crop_pad(array=obj, output_shape=diff_pattern.shape, debugging=False)

# calculate the retrieved diffraction amplitude
my_fft = fftshift(fftn(obj)) / (np.sqrt(numz)*np.sqrt(numy)*np.sqrt(numx))  # complex amplitude
del obj
gc.collect()
print('Max(retrieved amplitude) =', abs(my_fft).max())

if modes:  # if this is the first mode, intensity should be normalized to the measured diffraction pattern
    max_fft = np.sqrt(diff_pattern).max()
    my_fft = my_fft * max_fft / abs(my_fft).max()
    print('Max(retrieved amplitude) after modes normalization =', abs(my_fft).max())  # needed for modes

plt.figure()
plt.imshow(np.log10(abs(my_fft).sum(axis=0)), cmap=my_cmap, vmin=0, vmax=3.5)
plt.title('abs(retrieved amplitude).sum(axis=0)')
plt.colorbar()
plt.pause(0.1)

my_fft[np.nonzero(mask)] = 0  # do not take mask voxels into account
print('Max(retrieved amplitude) =', abs(my_fft).max())
print('COM of the retrieved diffraction pattern after masking: ', center_of_mass(abs(my_fft)))
del mask
gc.collect()

gu.combined_plots(tuple_array=(diff_pattern, diff_pattern, diff_pattern, my_fft, my_fft, my_fft),
                  tuple_sum_frames=False, tuple_sum_axis=(0, 1, 2, 0, 1, 2), tuple_width_v=np.nan, tuple_width_h=np.nan,
                  tuple_colorbar=False, tuple_vmin=(np.nan, np.nan, np.nan, -1, -1, -1), tuple_vmax=np.nan,
                  tuple_title=('diff_pattern', 'diff_pattern', 'diff_pattern', 'my_fft', 'my_fft', 'my_fft'),
                  tuple_scale='log')

#########################
# calculate the 3D PRTF #
#########################
diff_pattern[diff_pattern == 0] = np.nan  # discard zero valued pixels
prtf_matrix = abs(my_fft) / np.sqrt(diff_pattern)

gu.multislices_plot(prtf_matrix, sum_frames=False, plot_colorbar=True, cmap=my_cmap,
                    title='prtf_matrix', scale='linear', invert_yaxis=False, vmin=0, vmax=1.1,
                    reciprocal_space=True)

#################################
# average over spherical shells #
#################################
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

#############################
# plot and save the 1D PRTF #
#############################
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
