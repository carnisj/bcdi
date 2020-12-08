# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import hdf5plugin  # for P10, should be imported before h5py or PyTables
import numpy as np
from numpy.fft import fftn, fftshift
from matplotlib import pyplot as plt
from scipy.ndimage.measurements import center_of_mass
import tkinter as tk
from tkinter import filedialog
import xrayutilities as xu
from scipy.interpolate import interp1d
import gc
import sys
import os
sys.path.append('D:/myscripts/bcdi/')
sys.path.append('C:/Users/Jerome/Documents/myscripts/bcdi/')
import bcdi.graph.graph_utils as gu
import bcdi.experiment.experiment_utils as exp
import bcdi.preprocessing.preprocessing_utils as pru
import bcdi.postprocessing.postprocessing_utils as pu
import bcdi.utils.utilities as util

helptext = """
Calculate the resolution of a BCDI reconstruction using the phase retrieval transfer function (PRTF).

The measured diffraction pattern and reconstructions should be in the detector frame, before
phase ramp removal and centering.

For the laboratory frame, the CXI convention is used: z downstream, y vertical, x outboard
For q, the usual convention is used: qx downstream, qz vertical, qy outboard

Supported beamline: ESRF ID01, PETRAIII P10, SOLEIL SIXS, SOLEIL CRISTAL

Path structure: 
    specfile in /root_folder/
    data in /root_folder/S2191/data/
"""

scan = 54
sample_name = "p21"  # "SN"  #
root_folder = "D:/data/P10_isosurface/data/"  # location of the .spec or log file
savedir = ""  # PRTF will be saved here, leave it to '' otherwise
comment = ""  # should start with _
crop_roi = []  # ROI used if 'center_auto' was True in PyNX, leave [] otherwise
# in the.cxi file, it is the parameter 'entry_1/image_1/process_1/configuration/roi_final'
align_pattern = False  # if True, will align the retrieved diffraction amplitude with the measured one
flag_interact = True  # True to calculate interactively the PRTF along particular directions of reciprocal space
############################
# beamline parameters #
############################
beamline = 'P10'  # name of the beamline, used for data loading and normalization by monitor
# supported beamlines: 'ID01', 'SIXS_2018', 'SIXS_2019', 'CRISTAL', 'P10'
is_series = False  # specific to series measurement at P10
rocking_angle = "outofplane"  # "outofplane" or "inplane"
follow_bragg = False  # only for energy scans, set to True if the detector was also scanned to follow the Bragg peak
specfile_name = ''
# .spec for ID01, .fio for P10, alias_dict.txt for SIXS_2018, not used for CRISTAL and SIXS_2019
# template for ID01: name of the spec file without '.spec'
# template for SIXS_2018: full path of the alias dictionnary 'alias_dict.txt', typically: root_folder + 'alias_dict.txt'
# template for SIXS_2019: ''
# template for P10: ''
# template for CRISTAL: ''
######################################
# define detector related parameters #
######################################
detector = "Eiger4M"    # "Eiger2M" or "Maxipix" or "Eiger4M"
template_imagefile = '_master.h5'
# template for ID01: 'data_mpx4_%05d.edf.gz' or 'align_eiger2M_%05d.edf.gz'
# template for SIXS_2018: 'align.spec_ascan_mu_%05d.nxs'
# template for SIXS_2019: 'spare_ascan_mu_%05d.nxs'
# template for Cristal: 'S%d.nxs'
# template for P10: '_master.h5'
################################################################################
# parameters for calculating q values #
################################################################################
sdd = 1.83  # sample to detector distance in m
energy = 8820   # x-ray energy in eV, 6eV offset at ID01
beam_direction = (1, 0, 0)  # beam along x
sample_inplane = (1, 0, 0)  # sample inplane reference direction along the beam at 0 angles
sample_outofplane = (0, 0, 1)  # surface normal of the sample at 0 angles
pre_binning = (1, 3, 3)  # binning factor applied during preprocessing: rocking curve axis, detector vertical and
# horizontal axis. This is necessary to calculate correctly q values.
phasing_binning = (1, 1, 1)  # binning factor applied during phasing: rocking curve axis, detector vertical and
# horizontal axis.
# If the reconstructed object was further cropped after phasing, it will be automatically padded back to the FFT window
# shape used during phasing (after binning) before calculating the Fourier transform.
###############################
# only needed for simulations #
###############################
simulation = False  # True is this is simulated data, will not load the specfile
bragg_angle_simu = 17.1177  # value of the incident angle at Bragg peak (eta at ID01)
outofplane_simu = 35.3240  # detector delta @ ID01
inplane_simu = -1.6029  # detector nu @ ID01
tilt_simu = 0.0102  # angular step size for rocking angle, eta @ ID01
###########
# options #
###########
normalize_prtf = True  # set to True when the solution is the first mode - then the intensity needs to be normalized
debug = False  # True to show more plots
save = True  # True to save the prtf figure
background_plot = '0.5'  # in level of grey in [0,1], 0 being dark. For visual comfort during masking
##########################
# end of user parameters #
##########################


def on_click(event):
    """
    Function to interact with a plot, return the position of clicked pixel. If flag_pause==1 or
    if the mouse is out of plot axes, it will not register the click

    :param event: mouse click event
    """
    global ax0, ax1, ax2, ax3, prtf_matrix, z0, y0, x0, endpoint, distances_q, plt0, plt1, plt2, cut
    if event.inaxes == ax0:
        endpoint[2], endpoint[1] = int(np.rint(event.xdata)), int(np.rint(event.ydata))
        inaxes = True
    elif event.inaxes == ax1:
        endpoint[2], endpoint[0] = int(np.rint(event.xdata)), int(np.rint(event.ydata))
        inaxes = True
    elif event.inaxes == ax2:
        endpoint[1], endpoint[0] = int(np.rint(event.xdata)), int(np.rint(event.ydata))
        inaxes = True
    else:
        inaxes = False
    print(endpoint)

    if inaxes:
        cut = gu.linecut(prtf_matrix, start_indices=(z0, y0, x0), stop_indices=endpoint, interp_order=1,
                         debugging=False)
        plt0.remove()
        plt1.remove()
        plt2.remove()

        plt0, = ax0.plot([x0, endpoint[2]], [y0, endpoint[1]], 'ro-')  # sum axis 0
        plt1, = ax1.plot([x0, endpoint[2]], [z0, endpoint[0]], 'ro-')  # sum axis 1
        plt2, = ax2.plot([y0, endpoint[1]], [z0, endpoint[0]], 'ro-')  # sum axis 2
        ax3.cla()
        ax3.plot(cut)
        ax3.axis('auto')
        ax3.set_xlabel('q (1/nm)')
        ax3.set_ylabel('PRTF')
        plt.tight_layout()
        plt.draw()


def press_key(event):
    """
    Interact with the PRTF plot.

    :param event: button press event
    """
    global detector, endpoint, fig_prtf
    try:
        close_fig = False
        if event.inaxes:
            if event.key == 's':
                fig_prtf.savefig(detector.savedir+f'PRTF_endpoint={endpoint}.png')
            elif event.key == 'q':
                close_fig = True
        if close_fig:
            plt.close(close_fig)
    except AttributeError:  # mouse pointer out of axes
        pass


#################################################
# Initialize paths, detector, setup and logfile #
#################################################
kwargs = dict()  # create dictionnary
try:
    kwargs['is_series'] = is_series
except NameError:  # is_series not declared
    pass

detector = exp.Detector(name=detector, datadir='', template_imagefile=template_imagefile, binning=pre_binning, **kwargs)

setup = exp.SetupPreprocessing(beamline=beamline, rocking_angle=rocking_angle, distance=sdd, energy=energy,
                               beam_direction=beam_direction, sample_inplane=sample_inplane,
                               sample_outofplane=sample_outofplane,
                               offset_inplane=0)  # no need to worry about offsets, work relatively to the Bragg peak

print('\nScan', scan)
print('Setup: ', setup.beamline)
print('Detector: ', detector.name)
print('Pixel sizes after pre_binning (vertical, horizontal): ', detector.pixelsize_y, detector.pixelsize_x, '(m)')
print('Scan type: ', setup.rocking_angle)
print('Sample to detector distance: ', setup.distance, 'm')
print('Energy:', setup.energy, 'ev')

if simulation:
    detector.datadir = root_folder
    detector.savedir = root_folder
else:
    if setup.beamline == 'P10':
        specfile = sample_name + '_{:05d}'.format(scan)
        homedir = root_folder + specfile + '/'
        detector.datadir = homedir + 'e4m/'
        imagefile = specfile + template_imagefile
        detector.template_imagefile = imagefile
    elif setup.beamline == 'NANOMAX':
        homedir = root_folder + sample_name + '{:06d}'.format(scan) + '/'
        detector.datadir = homedir + 'data/'
        specfile = specfile_name
    else:
        homedir = root_folder + sample_name + str(scan) + '/'
        detector.datadir = homedir + "data/"
        specfile = specfile_name

    if savedir == '':
        detector.savedir = os.path.abspath(os.path.join(detector.datadir, os.pardir)) + '/'
    else:
        detector.savedir = savedir
    print('Datadir:', detector.datadir)
    print('Savedir:', detector.savedir)
    logfile = pru.create_logfile(setup=setup, detector=detector, scan_number=scan, root_folder=root_folder,
                                 filename=specfile)

#############################################
# Initialize geometry for orthogonalization #
#############################################
qconv, offsets = pru.init_qconversion(setup)
detector.offsets = offsets
hxrd = xu.experiment.HXRD(sample_inplane, sample_outofplane, qconv=qconv)  # x downstream, y outboard, z vertical
# first two arguments in HXRD are the inplane reference direction along the beam and surface normal of the sample

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
file_path = filedialog.askopenfilename(initialdir=detector.savedir, title="Select diffraction pattern",
                                       filetypes=[("NPZ", "*.npz"), ("NPY", "*.npy")])
diff_pattern, _ = util.load_file(file_path)
diff_pattern = diff_pattern.astype(float)

file_path = filedialog.askopenfilename(initialdir=detector.savedir, title="Select mask",
                                       filetypes=[("NPZ", "*.npz"), ("NPY", "*.npy")])
mask, _ = util.load_file(file_path)

# crop the diffraction pattern and the mask to compensate the "auto_center_resize" option used in PyNX.
# The shape will be equal to 'roi_final' parameter of the .cxi file
if len(crop_roi) == 6:
    diff_pattern = diff_pattern[crop_roi[0]:crop_roi[1], crop_roi[2]:crop_roi[3], crop_roi[4]:crop_roi[5]]
    mask = mask[crop_roi[0]:crop_roi[1], crop_roi[2]:crop_roi[3], crop_roi[4]:crop_roi[5]]
elif len(crop_roi) != 0:
    print('Crop_roi should be a list of 6 integers or a blank list!')
    sys.exit()

# bin the diffraction pattern and the mask to compensate the "rebin" option used in PyNX.
# update also the detector pixel sizes to take into account the binning
detector.pixelsize_y = detector.pixelsize_y * phasing_binning[1]
detector.pixelsize_x = detector.pixelsize_x * phasing_binning[2]
final_binning = np.multiply(pre_binning, phasing_binning)
detector.binning = final_binning
print('Pixel sizes after phasing_binning (vertical, horizontal): ', detector.pixelsize_y, detector.pixelsize_x, '(m)')
diff_pattern = pu.bin_data(array=diff_pattern, binning=phasing_binning, debugging=False)
mask = pu.bin_data(array=mask, binning=phasing_binning, debugging=False)

numz, numy, numx = diff_pattern.shape  # this shape will be used for the calculation of q values
print('\nMeasured data shape =', numz, numy, numx, ' Max(measured amplitude)=', np.sqrt(diff_pattern).max())
diff_pattern[np.nonzero(mask)] = 0

z0, y0, x0 = center_of_mass(diff_pattern)
z0, y0, x0 = [int(z0), int(y0), int(x0)]
print("COM of measured pattern after masking: ", z0, y0, x0, ' Number of unmasked photons =', diff_pattern.sum())

fig, _, _ = gu.multislices_plot(np.sqrt(diff_pattern), sum_frames=False, title='3D diffraction amplitude', vmin=0,
                                vmax=3.5, is_orthogonal=False, reciprocal_space=True, slice_position=[z0, y0, x0],
                                scale='log', plot_colorbar=True)

plt.figure()
plt.imshow(np.log10(np.sqrt(diff_pattern).sum(axis=0)), cmap=my_cmap, vmin=0, vmax=3.5)
plt.title('abs(diffraction amplitude).sum(axis=0)')
plt.colorbar()
plt.pause(0.1)

################################################
# calculate the q matrix respective to the COM #
################################################
hxrd.Ang2Q.init_area('z-', 'y+', cch1=int(y0), cch2=int(x0), Nch1=numy, Nch2=numx,
                     pwidth1=detector.pixelsize_y, pwidth2=detector.pixelsize_x, distance=setup.distance)
# first two arguments in init_area are the direction of the detector
if simulation:
    eta = bragg_angle_simu + tilt_simu * (np.arange(0, numz, 1) - int(z0))
    qx, qy, qz = hxrd.Ang2Q.area(eta, 0, 0, inplane_simu, outofplane_simu, delta=(0, 0, 0, 0, 0))
else:
    qx, qz, qy, _ = pru.regrid(logfile=logfile, nb_frames=numz, scan_number=scan, detector=detector,
                               setup=setup, hxrd=hxrd, follow_bragg=follow_bragg)

if debug:
    gu.combined_plots(tuple_array=(qz, qy, qx), tuple_sum_frames=False, tuple_sum_axis=(0, 1, 2),
                      tuple_width_v=None, tuple_width_h=None, tuple_colorbar=True, tuple_vmin=np.nan,
                      tuple_vmax=np.nan, tuple_title=('qz', 'qy', 'qx'), tuple_scale='linear')

qxCOM = qx[z0, y0, x0]
qyCOM = qy[z0, y0, x0]
qzCOM = qz[z0, y0, x0]
print('COM[qx, qy, qz] = ', qxCOM, qyCOM, qzCOM)
distances_q = np.sqrt((qx - qxCOM)**2 + (qy - qyCOM)**2 + (qz - qzCOM)**2)  # if reconstructions are centered
#  and of the same shape q values will be identical
del qx, qy, qz
gc.collect()

if distances_q.shape != diff_pattern.shape:
    print('\nThe shape of q values and the shape of the diffraction pattern are different: check binning parameters!')
    sys.exit()

if debug:
    gu.multislices_plot(distances_q, sum_frames=False, plot_colorbar=True, cmap=my_cmap,
                        title='distances_q', scale='linear', vmin=np.nan, vmax=np.nan,
                        reciprocal_space=True)

#############################
# load reconstructed object #
#############################
file_path = filedialog.askopenfilename(initialdir=detector.savedir, title="Select reconstructions (prtf)",
                                       filetypes=[("NPZ", "*.npz"), ("NPY", "*.npy"),
                                                  ("CXI", "*.cxi"), ("HDF5", "*.h5")])

obj, extension = util.load_file(file_path)
print('Opening ', file_path)

if extension == '.h5':
    comment = comment + '_mode'

# check if the shape of the real space object is the same as the measured diffraction pattern
# the real space object may have been further cropped to a tight support, to save memory space.
if obj.shape != diff_pattern.shape:
    print('Reconstructed object shape = ', obj.shape, 'different from the experimental diffraction pattern: crop/pad')
    obj = pu.crop_pad(array=obj, output_shape=diff_pattern.shape, debugging=False)

# calculate the retrieved diffraction amplitude
phased_fft = fftshift(fftn(obj)) / (np.sqrt(numz)*np.sqrt(numy)*np.sqrt(numx))  # complex amplitude
del obj
gc.collect()

if debug:
    plt.figure()
    plt.imshow(np.log10(abs(phased_fft).sum(axis=0)), cmap=my_cmap, vmin=0, vmax=3.5)
    plt.colorbar()
    plt.title('abs(retrieved amplitude).sum(axis=0) before alignment')
    plt.pause(0.1)

if align_pattern:
    # align the reconstruction with the initial diffraction data
    phased_fft, _ = pru.align_diffpattern(reference_data=diff_pattern, data=phased_fft, method='registration',
                                          combining_method='subpixel')

plt.figure()
plt.imshow(np.log10(abs(phased_fft).sum(axis=0)), cmap=my_cmap, vmin=0, vmax=3.5)
plt.title('abs(retrieved amplitude).sum(axis=0)')
plt.colorbar()
plt.pause(0.1)

phased_fft[np.nonzero(mask)] = 0  # do not take mask voxels into account
print('Max(retrieved amplitude) =', abs(phased_fft).max())
print('COM of the retrieved diffraction pattern after masking: ', center_of_mass(abs(phased_fft)))
del mask
gc.collect()

gu.combined_plots(tuple_array=(diff_pattern, phased_fft), tuple_sum_frames=False, tuple_sum_axis=(0, 0),
                  tuple_width_v=None, tuple_width_h=None, tuple_colorbar=False, tuple_vmin=(-1, -1),
                  tuple_vmax=np.nan, tuple_title=('measurement', 'phased_fft'), tuple_scale='log')

#########################
# calculate the 3D PRTF #
#########################
diff_pattern[diff_pattern == 0] = np.nan  # discard zero valued pixels
prtf_matrix = abs(phased_fft) / np.sqrt(diff_pattern)
# np.savez_compressed(detector.savedir+'prtf_3d.npz', prtf=prtf_matrix)
if normalize_prtf:
    print('Normalizing the PRTF to 1 at the center of mass ...')
    prtf_matrix = prtf_matrix / prtf_matrix[z0, y0, x0]

gu.multislices_plot(prtf_matrix, sum_frames=False, plot_colorbar=True, cmap=my_cmap,
                    title='prtf_matrix', scale='linear', vmin=0,
                    reciprocal_space=True)

#######################################################################
# interactive interface to check the PRTF along particular directions #
#######################################################################
if flag_interact:
    plt.ioff()
    max_colorbar = 5
    endpoint = [0, 0, 0]
    cut = gu.linecut(prtf_matrix, start_indices=(z0, y0, x0), stop_indices=endpoint, interp_order=1,
                     debugging=False)
    diff_pattern[np.isnan(diff_pattern)] = 0  # discard nans
    fig_prtf, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))
    fig_prtf.canvas.mpl_disconnect(fig_prtf.canvas.manager.key_press_handler_id)
    ax0.imshow(np.log10(diff_pattern.sum(axis=0)), vmin=0, vmax=max_colorbar, cmap=my_cmap)
    ax1.imshow(np.log10(diff_pattern.sum(axis=1)), vmin=0, vmax=max_colorbar, cmap=my_cmap)
    ax2.imshow(np.log10(diff_pattern.sum(axis=2)), vmin=0, vmax=max_colorbar, cmap=my_cmap)
    ax3.plot(cut)
    plt0, = ax0.plot([x0, endpoint[2]], [y0, endpoint[1]], 'ro-')  # sum axis 0
    plt1, = ax1.plot([x0, endpoint[2]], [z0, endpoint[0]], 'ro-')  # sum axis 1
    plt2, = ax2.plot([y0, endpoint[1]], [z0, endpoint[0]], 'ro-')  # sum axis 2
    ax0.axis('scaled')
    ax1.axis('scaled')
    ax2.axis('scaled')
    ax3.axis('auto')
    ax0.set_title("horizontal=X  vertical=Y")
    ax1.set_title("horizontal=X  vertical=rocking curve")
    ax2.set_title("horizontal=Y  vertical=rocking curve")
    ax3.set_xlabel('q (1/nm)')
    ax3.set_ylabel('PRTF')
    fig_prtf.text(0.01, 0.8, "click to select\nthe endpoint", size=10)
    fig_prtf.text(0.01, 0.7, "q to quit\ns to save", size=10)
    plt.tight_layout()
    plt.connect('key_press_event', press_key)
    plt.connect('button_press_event', on_click)
    fig_prtf.set_facecolor(background_plot)
    plt.show()

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

fig, ax = plt.subplots(1, 1)
ax.plot(defined_q, prtf_avg[~np.isnan(prtf_avg)], 'or')  # q_axis in 1/nm

ax.plot([defined_q.min(), defined_q.max()], [1/np.e, 1/np.e], 'k.', lw=1)
ax.set_xlim(defined_q.min(), defined_q.max())
ax.set_ylim(0, 1.1)
ax.spines['right'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.tick_params(labelbottom=False, labelleft=False)
if save:
    fig.savefig(detector.savedir + 'S' + str(scan) + '_prtf' + comment + '.png')
ax.set_title('PRTF')
ax.set_xlabel('q (1/nm)')
ax.tick_params(labelbottom=True, labelleft=True)
fig.text(0.15, 0.25, "Scan " + str(scan) + comment, size=14)
fig.text(0.15, 0.20, "q at PRTF=1/e: " + str('{:.5f}'.format(q_resolution)) + '(1/nm)', size=14)
fig.text(0.15, 0.15, "resolution d= " + str('{:.3f}'.format(2*np.pi / q_resolution)) + 'nm', size=14)
if save:
    fig.savefig(detector.savedir + 'S' + str(scan) + '_prtf_comments' + comment + '.png')
plt.ioff()
plt.show()
