#!/usr/bin/env python3

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import gc
import sys
import tkinter as tk
from tkinter import filedialog

import numpy as np
import xrayutilities as xu
from matplotlib import pyplot as plt
from numpy.fft import fftn, fftshift
from scipy.interpolate import interp1d
from scipy.ndimage.measurements import center_of_mass

import bcdi.graph.graph_utils as gu
import bcdi.utils.image_registration as reg
import bcdi.utils.utilities as util
from bcdi.experiment.setup import Setup
from bcdi.graph.colormap import ColormapFactory

helptext = """
Calculate the resolution of a 2D BCDI reconstruction using the phase retrieval
transfer function (PRTF). The measured diffraction pattern and reconstructions should
be in the detector frame, before phase ramp removal and centering. An optional mask
can be provided.

For the laboratory frame, the CXI convention is used: z downstream, y vertical,
x outboard. For q, the usual convention is used: qx downstream, qz vertical, qy outboard

Supported beamline: ESRF ID01, PETRAIII P10, SOLEIL SIXS, SOLEIL CRISTAL

Path structure:
    specfile in /root_folder/
    data in /root_folder/S2191/data/
"""

scan = 279
root_folder = "D:/data/DATA_exp/"
# folder of the experiment, where all scans are stored
save_dir = None  # PRTF will be saved here, leave None otherwise
sample_name = "S"  # "SN"  #
comment = ""  # should start with _
crop_roi = [
    3,
    255,
    3,
    387,
]  # ROI used if 'center_auto' was True in PyNX, leave [] otherwise
# in the.cxi file,
# it is the parameter 'entry_1/image_1/process_1/configuration/roi_final'
align_pattern = False
# if True, will align the retrieved diffraction amplitude with the measured one
slicing_axis = 1  # 0 for first axis, 1 for second, 2 for third
#######################
# beamline parameters #
#######################
beamline = (
    "ID01"  # name of the beamline, used for data loading and normalization by monitor
)
# supported beamlines: 'ID01', 'SIXS_2018', 'SIXS_2019', 'CRISTAL', 'P10'
actuators = {}
# Optional dictionary that can be used to define the entries corresponding to
# actuators in data files
# (useful at CRISTAL where the location of data keeps changing)
# e.g.  {'rocking_angle': 'actuator_1_3', 'detector': 'data_04', 'monitor': 'data_05'}
is_series = False  # specific to series measurement at P10
rocking_angle = "outofplane"  # "outofplane" or "inplane"
specfile_name = "alignment"
# .spec for ID01, .fio for P10, alias_dict.txt for SIXS_2018,
# not used for CRISTAL and SIXS_2019
# template for ID01: name of the spec file without '.spec'
# template for SIXS_2018: full path of the alias dictionnary 'alias_dict.txt',
# typically: root_folder + 'alias_dict.txt'
# template for all other beamlines: ''
#############################################################
# define detector related parameters and region of interest #
#############################################################
detector = "Maxipix"  # "Eiger2M" or "Maxipix" or "Eiger4M"
template_imagefile = "alignment_12_%04d.edf.gz"
# template for ID01: 'data_mpx4_%05d.edf.gz' or 'align_eiger2M_%05d.edf.gz'
# template for SIXS_2018: 'align.spec_ascan_mu_%05d.nxs'
# template for SIXS_2019: 'spare_ascan_mu_%05d.nxs'
# template for Cristal: 'S%d.nxs'
# template for P10: '_master.h5'
# template for NANOMAX: '%06d.h5'
# template for 34ID: 'Sample%dC_ES_data_51_256_256.npz'
#######################################
# parameters for calculating q values #
#######################################
sdd = 1.3  # sample to detector distance in m
energy = 9000  # x-ray energy in eV, 6eV offset at ID01
beam_direction = (1, 0, 0)  # beam along x
sample_inplane = (
    1,
    0,
    0,
)  # sample inplane reference direction along the beam at 0 angles
sample_outofplane = (0, 0, 1)  # surface normal of the sample at 0 angles
pre_binning = (
    1,
    3,
    1,
)  # binning factor applied during preprocessing: rocking curve axis,
# detector vertical and horizontal axis. This is necessary to calculate correctly q
# values. Use (1, binning_Y, binning_X) for 2D data.
phasing_binning = (
    1,
    2,
    2,
)  # binning factor applied during phasing: rocking curve axis, detector vertical and
# horizontal axis. Use (1, binning_Y, binning_X) for 2D data.
# If the reconstructed object was further cropped after phasing,
# it will be automatically padded back to the FFT window
# shape used during phasing (after binning) before calculating the Fourier transform.
sample_offsets = (
    0,
    0,
    0,
)  # tuple of offsets in degrees of the sample for each sample circle (outer first).
# the sample offsets will be subtracted to the motor values. Leave None if no offset.
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
normalize_prtf = True  # set to True when the solution is the first mode
# then the intensity needs to be normalized
debug = False  # True to show more plots
##########################
# end of user parameters #
##########################

####################
# Initialize setup #
####################
setup = Setup(
    beamline_name=beamline,
    energy=energy,
    rocking_angle=rocking_angle,
    distance=sdd,
    beam_direction=beam_direction,
    sample_inplane=sample_inplane,
    sample_outofplane=sample_outofplane,
    sample_offsets=sample_offsets,
    actuators=actuators,
    is_series=is_series,
    detector_name=detector,
    template_imagefile=template_imagefile,
    binning=(1, 1, 1),
    preprocessing_binning=pre_binning,
)

########################################
# Initialize the paths and the logfile #
########################################
setup.init_paths(
    sample_name=sample_name,
    scan_number=scan,
    root_folder=root_folder,
    save_dir=save_dir,
    specfile_name=specfile_name,
    template_imagefile=template_imagefile,
)

setup.create_logfile(
    scan_number=scan, root_folder=root_folder, filename=setup.detector.specfile
)

###################
# print instances #
###################
print(f'{"#"*(5+len(str(scan)))}\nScan {scan}\n{"#"*(5+len(str(scan)))}')
print("\n##############\nSetup instance\n##############")
print(setup)
print("\n#################\nDetector instance\n#################")
print(setup.detector)

#############################################
# Initialize geometry for orthogonalization #
#############################################
qconv, offsets = setup.init_qconversion()
setup.detector.offsets = offsets
hxrd = xu.experiment.HXRD(sample_inplane, sample_outofplane, qconv=qconv)
# x downstream, y outboard, z vertical
# first two arguments in HXRD are the inplane reference direction
# along the beam and surface normal of the sample

###################
# define colormap #
###################
my_cmap = ColormapFactory().cmap

##########################################################
# load experimental data, extracted 2D slice and 2D mask #
##########################################################
plt.ion()
root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename(
    initialdir=setup.detector.savedir,
    title="Select 2D slice",
    filetypes=[("NPZ", "*.npz"), ("NPY", "*.npy")],
)
slice_2D, _ = util.load_file(file_path)
slice_2D = slice_2D.astype(float)

file_path = filedialog.askopenfilename(
    initialdir=setup.detector.savedir,
    title="Select 2D mask",
    filetypes=[("NPZ", "*.npz"), ("NPY", "*.npy")],
)
try:
    mask_2D, _ = util.load_file(file_path)
except ValueError:
    mask_2D = np.zeros(slice_2D.shape, dtype=int)

###########################################################
# crop the diffraction pattern and the mask to compensate #
# the "auto_center_resize" option used in PyNX            #
###########################################################
# The shape will be equal to 'roi_final' parameter of the .cxi file
if len(crop_roi) == 4:
    slice_2D = slice_2D[crop_roi[0] : crop_roi[1], crop_roi[2] : crop_roi[3]]
    mask_2D = mask_2D[crop_roi[0] : crop_roi[1], crop_roi[2] : crop_roi[3]]
elif len(crop_roi) != 0:
    print("Crop_roi should be a list of 6 integers or a blank list!")
    sys.exit()

###############################################
# bin the diffraction pattern and the mask to #
# compensate the "rebin" option used in PyNX  #
###############################################
# update also the detector pixel sizes to take into account the binning
setup.detector.binning = phasing_binning
print(
    "Pixel sizes after phasing_binning (vertical, horizontal): ",
    setup.detector.pixelsize_y,
    setup.detector.pixelsize_x,
    "(m)",
)
slice_2D = util.bin_data(
    array=slice_2D, binning=(phasing_binning[1], phasing_binning[2]), debugging=False
)
mask_2D = util.bin_data(
    array=mask_2D, binning=(phasing_binning[1], phasing_binning[2]), debugging=False
)

slice_2D[np.nonzero(mask_2D)] = 0

plt.figure()
plt.imshow(np.log10(np.sqrt(slice_2D)), cmap=my_cmap, vmin=0, vmax=3.5)
plt.title("2D diffraction amplitude")
plt.colorbar()
plt.pause(0.1)

##########################################################
# load the 3D dataset in order to calculate the q values #
##########################################################
file_path = filedialog.askopenfilename(
    initialdir=setup.detector.savedir,
    title="Select the 3D diffraction pattern",
    filetypes=[("NPZ", "*.npz"), ("NPY", "*.npy")],
)
diff_pattern, _ = util.load_file(file_path)
diff_pattern = diff_pattern.astype(float)

# crop the diffraction pattern to compensate
# the "auto_center_resize" option used in PyNX.
# The shape will be equal to 'roi_final' parameter of the .cxi file
diff_pattern = diff_pattern[:, crop_roi[0] : crop_roi[1], crop_roi[2] : crop_roi[3]]

# bin the diffraction pattern to compensate the "rebin" option used in PyNX.
# the detector pixel sizes where already updated above.
diff_pattern = util.bin_data(
    array=diff_pattern, binning=phasing_binning, debugging=False
)

numz, numy, numx = diff_pattern.shape
print(
    "\nMeasured data shape =",
    numz,
    numy,
    numx,
    " Max(measured amplitude)=",
    np.sqrt(diff_pattern).max(),
)
z0, y0, x0 = center_of_mass(diff_pattern)
print(f"COM of measured pattern after masking: {z0:.2f}, {y0:.2f}, {x0:.2f}")
# refine the COM in a small ROI centered on the approximate COM, to avoid detector gaps
fine_com = center_of_mass(
    diff_pattern[
        int(z0) - 20 : int(z0) + 21,
        int(y0) - 20 : int(y0) + 21,
        int(x0) - 20 : int(x0) + 21,
    ]
)
z0, y0, x0 = [
    int(np.rint(z0 - 20 + fine_com[0])),
    int(np.rint(y0 - 20 + fine_com[1])),
    int(np.rint(x0 - 20 + fine_com[2])),
]
print(
    f"refined COM: {z0}, {y0}, {x0}, "
    f"Number of unmasked photons = {diff_pattern.sum():.0f}\n"
)

fig, _, _ = gu.multislices_plot(
    np.sqrt(diff_pattern),
    sum_frames=False,
    title="3D diffraction amplitude",
    vmin=0,
    vmax=3.5,
    is_orthogonal=False,
    reciprocal_space=True,
    slice_position=[z0, y0, x0],
    scale="log",
    plot_colorbar=True,
)

################################################
# calculate the q matrix respective to the COM #
################################################
hxrd.Ang2Q.init_area(
    "z-",
    "y+",
    cch1=int(y0),
    cch2=int(x0),
    Nch1=numy,
    Nch2=numx,
    pwidth1=setup.detector.pixelsize_y,
    pwidth2=setup.detector.pixelsize_x,
    distance=setup.distance,
)
# first two arguments in init_area are the direction of the detector
if simulation:
    eta = bragg_angle_simu + tilt_simu * (np.arange(0, numz, 1) - int(z0))
    qx, qy, qz = hxrd.Ang2Q.area(
        eta, 0, 0, inplane_simu, outofplane_simu, delta=(0, 0, 0, 0, 0)
    )
else:
    qx, qz, qy, _ = setup.calc_qvalues_xrutils(
        hxrd=hxrd,
        nb_frames=numz,
        scan_number=scan,
    )

if debug:
    gu.combined_plots(
        tuple_array=(qz, qy, qx),
        tuple_sum_frames=False,
        tuple_sum_axis=(0, 1, 2),
        tuple_width_v=None,
        tuple_width_h=None,
        tuple_colorbar=True,
        tuple_vmin=np.nan,
        tuple_vmax=np.nan,
        tuple_title=("qz", "qy", "qx"),
        tuple_scale="linear",
    )

qxCOM = qx[z0, y0, x0]
qyCOM = qy[z0, y0, x0]
qzCOM = qz[z0, y0, x0]
print(f"COM[qx, qz, qy] = {qxCOM:.2f}, {qzCOM:.2f}, {qyCOM:.2f}")
distances_q = np.sqrt(
    (qx - qxCOM) ** 2 + (qy - qyCOM) ** 2 + (qz - qzCOM) ** 2
)  # if reconstructions are centered
#  and of the same shape q values will be identical
del qx, qy, qz
gc.collect()

if distances_q.shape != diff_pattern.shape:
    print(
        "\nThe shape of q values and the shape of the diffraction pattern "
        "are different: check binning parameter"
    )
    sys.exit()

if debug:
    gu.multislices_plot(
        distances_q,
        sum_frames=False,
        plot_colorbar=True,
        cmap=my_cmap,
        title="distances_q",
        scale="linear",
        vmin=np.nan,
        vmax=np.nan,
        reciprocal_space=True,
    )

################################
# select the relevant 2D slice #
################################
if slicing_axis == 0:
    distances_q = distances_q[z0, :, :]  # take only the slice at the COM in y0
elif slicing_axis == 1:
    distances_q = distances_q[:, y0, :]  # take only the slice at the COM in y0
elif slicing_axis == 2:
    distances_q = distances_q[:, :, x0]  # take only the slice at the COM in y0
else:
    print('Invalid value for "slicing_axis" parameter')
    sys.exit()

if distances_q.shape != slice_2D.shape:
    print(
        "\nThe shape of 2D q values and the shape of "
        "the 2D diffraction pattern are different!"
    )
    sys.exit()

plt.figure()
plt.imshow(distances_q, cmap=my_cmap)
plt.title("2D distances_q")
plt.colorbar()
plt.pause(0.1)

#############################
# load reconstructed object #
#############################
file_path = filedialog.askopenfilename(
    initialdir=setup.detector.savedir,
    title="Select a 2D reconstruction (prtf)",
    filetypes=[("NPZ", "*.npz"), ("NPY", "*.npy"), ("CXI", "*.cxi"), ("HDF5", "*.h5")],
)
obj, extension = util.load_file(file_path)
print("Opening ", file_path)

if extension == ".h5":
    comment = comment + "_mode"

# check if the shape of the real space object is the same as
# the measured 2D diffraction pattern. The real space object may have been further
# cropped to a tight support, to save memory space.
if obj.shape != slice_2D.shape:
    print(
        f"Reconstructed object shape = {obj.shape},"
        " different from the 2D diffraction slice: crop/pad"
    )
    obj = util.crop_pad_2d(array=obj, output_shape=slice_2D.shape, debugging=False)

plt.figure()
plt.imshow(abs(obj), vmin=0, cmap=my_cmap)
plt.colorbar()
plt.title("abs(reconstructed object")
plt.pause(0.1)

# calculate the retrieved diffraction amplitude
numy, numx = slice_2D.shape
phased_fft = fftshift(fftn(obj)) / (np.sqrt(numy) * np.sqrt(numx))  # complex amplitude
del obj
gc.collect()

if debug:
    plt.figure()
    plt.imshow(np.log10(abs(phased_fft)), vmin=0, vmax=3.5, cmap=my_cmap)
    plt.colorbar()
    plt.title("abs(retrieved amplitude) before alignement")
    plt.pause(0.1)

if align_pattern:
    # align the reconstruction with the initial diffraction data
    phased_fft, _ = reg.align_diffpattern(
        reference_data=slice_2D,
        data=phased_fft,
        interpolation_method="subpixel",
    )

plt.figure()
plt.imshow(np.log10(abs(phased_fft)), cmap=my_cmap, vmin=0, vmax=3.5)
plt.title("abs(retrieved amplitude)")
plt.colorbar()
plt.pause(0.1)

phased_fft[np.nonzero(mask_2D)] = 0  # do not take mask voxels into account
print(f"Max(retrieved amplitude) = {abs(phased_fft).max():.1f}")
phased_com_y, phased_com_x = center_of_mass(abs(phased_fft))
print(
    f"COM of the retrieved diffraction pattern after masking: {phased_com_y:.2f},"
    f" {phased_com_x:.2f}\n"
)
del mask_2D
gc.collect()

gu.combined_plots(
    tuple_array=(slice_2D, phased_fft),
    tuple_sum_frames=False,
    tuple_sum_axis=(0, 0),
    tuple_width_v=None,
    tuple_width_h=None,
    tuple_colorbar=False,
    tuple_vmin=(-1, -1),
    tuple_vmax=np.nan,
    tuple_title=("measurement", "phased_fft"),
    tuple_scale="log",
)

#########################
# calculate the 2D PRTF #
#########################
slice_2D[slice_2D == 0] = np.nan  # discard zero valued pixels
prtf_matrix = abs(phased_fft) / np.sqrt(slice_2D)
plt.figure()
plt.imshow(prtf_matrix, cmap=my_cmap, vmin=0, vmax=1.1)
plt.title("prtf_matrix")
plt.colorbar()
plt.pause(0.1)

#######################
# average over shells #
#######################
print(
    f"Distance max: {distances_q.max():.6f}  (1/A) "
    f"at: {np.unravel_index(abs(distances_q).argmax(), distances_q.shape)}"
)
nb_bins = numy // 3
prtf_avg = np.zeros(nb_bins)
dq = distances_q.max() / nb_bins  # in 1/A
q_axis = np.linspace(0, distances_q.max(), endpoint=True, num=nb_bins + 1)  # in 1/A

for index in range(nb_bins):
    logical_array = np.logical_and(
        (distances_q < q_axis[index + 1]), (distances_q >= q_axis[index])
    )
    temp = prtf_matrix[logical_array]
    prtf_avg[index] = temp[~np.isnan(temp)].mean()
q_axis = q_axis[:-1]

if normalize_prtf:
    print("Normalizing the PRTF to 1 ...")
    prtf_avg = prtf_avg / prtf_avg[~np.isnan(prtf_avg)].max()  # normalize to 1

#############################
# plot and save the 1D PRTF #
#############################
defined_q = 10 * q_axis[~np.isnan(prtf_avg)]  # switch to 1/nm

# create a new variable 'arc_length' to predict q and prtf parametrically
# (because prtf is not monotonic)
arc_length = np.concatenate(
    (
        np.zeros(1),
        np.cumsum(
            np.diff(prtf_avg[~np.isnan(prtf_avg)]) ** 2 + np.diff(defined_q) ** 2
        ),
    ),
    axis=0,
)  # cumulative linear arc length, used as the parameter

fit_prtf = interp1d(prtf_avg[~np.isnan(prtf_avg)], arc_length, kind="linear")
try:
    arc_length_res = fit_prtf(1 / np.e)
    fit_q = interp1d(arc_length, defined_q, kind="linear")
    q_resolution = fit_q(arc_length_res)
except ValueError:
    if (prtf_avg[~np.isnan(prtf_avg)] > 1 / np.e).all():
        print("Resolution limited by the 1 photon counts only (min(prtf)>1/e)")
        print(f"min(PRTF) = {prtf_avg[~np.isnan(prtf_avg)].min()}")
        q_resolution = defined_q.max()
    else:  # PRTF always below 1/e
        print("PRTF < 1/e for all q values, problem of normalization")
        q_resolution = np.nan

print(f"q resolution = {q_resolution:.5f} (1/nm)")
print(f"resolution d = {2*np.pi / q_resolution:.1f} nm")

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 9))
ax.plot(defined_q, prtf_avg[~np.isnan(prtf_avg)], "or")  # q_axis in 1/nm
ax.axhline(
    y=1 / np.e, linestyle="dashed", color="k", linewidth=1
)  # horizontal line at PRTF=1/e
ax.set_xlim(defined_q.min(), defined_q.max())
ax.set_ylim(0, 1.1)

gu.savefig(
    savedir=setup.detector.savedir,
    figure=fig,
    axes=ax,
    tick_width=2,
    tick_length=10,
    tick_labelsize=14,
    label_size=16,
    xlabels="q (1/nm)",
    ylabels="PRTF",
    filename=f"S{scan}_prtf" + comment,
    text={
        0: {"x": 0.15, "y": 0.30, "s": "Scan " + str(scan) + comment, "fontsize": 16},
        1: {
            "x": 0.15,
            "y": 0.25,
            "s": f"q at PRTF=1/e: {q_resolution:.5f} (1/nm)",
            "fontsize": 16,
        },
        2: {
            "x": 0.15,
            "y": 0.20,
            "s": f"resolution d = {2*np.pi / q_resolution:.3f} nm",
            "fontsize": 16,
        },
    },
)

plt.ioff()
plt.show()
