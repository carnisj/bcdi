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
import bcdi.graph.linecut as lc
import bcdi.utils.image_registration as reg
import bcdi.utils.utilities as util
import bcdi.utils.validation as valid
from bcdi.experiment.setup import Setup
from bcdi.graph.colormap import ColormapFactory

helptext = """
Calculate the resolution of a BCDI reconstruction using the phase retrieval transfer
function (PRTF). The measured diffraction pattern and reconstructions should be in
the detector frame, before phase ramp removal and centering. An optional mask can be
provided.

For the laboratory frame, the CXI convention is used: z downstream, y vertical,
x outboard. For q, the usual convention is used: qx downstream, qz vertical, qy outboard

Supported beamline: ESRF ID01, PETRAIII P10, SOLEIL SIXS, SOLEIL CRISTAL

Path structure:
    specfile in /root_folder/
    data in /root_folder/S2191/data/
"""

scan = 74
sample_name = "S"  # "SN"  #
root_folder = "D:/data/CRISTAL_March2021/"
# folder of the experiment, where all scans are stored
save_dir = None  # PRTF will be saved here, leave None otherwise
comment = ""  # should start with _
crop_roi = None
# list of 6 integers, ROI used if 'center_auto' was True in PyNX, leave None otherwise
# in the.cxi file,
# it is the parameter 'entry_1/image_1/process_1/configuration/roi_final'
align_pattern = False
# if True, will align the retrieved diffraction amplitude with the measured one
flag_interact = False  # True to calculate interactively the PRTF along particular
# directions of reciprocal space
#######################
# beamline parameters #
#######################
beamline = "CRISTAL"  # name of the beamline, used for data loading
# and normalization by monitor
# supported beamlines: 'ID01', 'SIXS_2018', 'SIXS_2019', 'CRISTAL', 'P10'
actuators = {"rocking_angle": "actuator_1_1"}
# Optional dictionary that can be used to define the entries
# corresponding to actuators in data files
# (useful at CRISTAL where the location of data keeps changing)
# e.g.  {'rocking_angle': 'actuator_1_3', 'detector': 'data_04', 'monitor': 'data_05'}
is_series = True  # specific to series measurement at P10
rocking_angle = "inplane"  # "outofplane" or "inplane"
specfile_name = ""
# template for ID01: name of the spec file without '.spec'
# template for SIXS_2018: full path of the alias dictionnary 'alias_dict.txt',
# typically: root_folder + 'alias_dict.txt'
# template for all other beamlines: ''
######################################
# define detector related parameters #
######################################
detector = "Maxipix"  # "Eiger2M" or "Maxipix" or "Eiger4M"
template_imagefile = "mgphi-2021_%04d.nxs"
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
sdd = 0.914  # sample to detector distance in m
energy = 8500  # x-ray energy in eV, 6eV offset at ID01
beam_direction = (1, 0, 0)  # beam along x
sample_inplane = (
    1,
    0,
    0,
)  # sample inplane reference direction along the beam at 0 angles
sample_outofplane = (0, 0, 1)  # surface normal of the sample at 0 angles
pre_binning = (
    1,
    1,
    1,
)  # binning factor applied during preprocessing: rocking curve axis,
# detector vertical and horizontal axis. This is necessary to calculate correctly q
# values.
phasing_binning = (
    1,
    1,
    1,
)  # binning factor applied during phasing: rocking curve axis, detector vertical and
# horizontal axis.
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
interpolate_nans = False  # if True, interpolate nans in the PRTF
# before the interactive interface. Time consuming
debug = False  # True to show more plots
background_plot = "0.5"  # in level of grey in [0,1], 0 being dark.
# For visual comfort when using the GUI
##########################
# end of user parameters #
##########################


def on_click(event):
    """
    Function to interact with a plot, return the position of clicked pixel.
    If flag_pause==1 or if the mouse is out of plot axes, it will not register the click

    :param event: mouse click event
    """
    global linecut_prtf, z0, y0, x0, endpoint, distances_q
    global fig_prtf, ax0, ax1, ax2, ax3, plt0, plt1, plt2, cut, res_text
    if event.inaxes == ax0:  # hor=X, ver=Y
        endpoint[2], endpoint[1] = int(np.rint(event.xdata)), int(np.rint(event.ydata))
        update_cut = True
    elif event.inaxes == ax1:  # hor=X, ver=rocking curve
        endpoint[2], endpoint[0] = int(np.rint(event.xdata)), int(np.rint(event.ydata))
        update_cut = True
    elif event.inaxes == ax2:  # hor=Y, ver=rocking curve
        endpoint[1], endpoint[0] = int(np.rint(event.xdata)), int(np.rint(event.ydata))
        update_cut = True
    elif event.inaxes == ax3:  # print the resolution at the mouse position
        res_text.remove()
        res_text = fig_prtf.text(
            0.55, 0.25, f"Resolution={2*np.pi/(10*event.xdata):.1f} nm", size=10
        )
        plt.draw()
        update_cut = False
    else:
        update_cut = False

    if update_cut:
        res_text.remove()
        res_text = fig_prtf.text(0.55, 0.25, "", size=10)
        print(f"endpoint = {endpoint}")
        cut = lc.linecut(
            linecut_prtf,
            indices=list(zip(starting_point, endpoint)),
            interp_order=1,
        )
        plt0.remove()
        plt1.remove()
        plt2.remove()

        (plt0,) = ax0.plot([x0, endpoint[2]], [y0, endpoint[1]], "ro-")  # sum axis 0
        (plt1,) = ax1.plot([x0, endpoint[2]], [z0, endpoint[0]], "ro-")  # sum axis 1
        (plt2,) = ax2.plot([y0, endpoint[1]], [z0, endpoint[0]], "ro-")  # sum axis 2

        ax3.cla()
        ax3.plot(
            np.linspace(
                distances_q[z0, y0, x0],
                distances_q[endpoint[0], endpoint[1], endpoint[2]],
                num=len(cut),
            ),
            cut,
            "-or",
            markersize=3,
        )
        ax3.axhline(y=1 / np.e, linestyle="dashed", color="k", linewidth=1)
        ax3.set_xlabel("q (1/A)")
        ax3.set_ylabel("PRTF")
        ax3.axis("auto")
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
            if event.key == "s":
                fig_prtf.savefig(detector.savedir + f"PRTF_endpoint={endpoint}.png")
            elif event.key == "q":
                close_fig = True
        if close_fig:
            plt.close("all")
    except AttributeError:  # mouse pointer out of axes
        pass


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

########################
# Initialize the paths #
########################
setup.init_paths(
    sample_name=sample_name,
    scan_number=scan,
    root_folder=root_folder,
    save_dir=save_dir,
    specfile_name=specfile_name,
    template_imagefile=template_imagefile,
)

###################
# print instances #
###################
print(f'{"#"*(5+len(str(scan)))}\nScan {scan}\n{"#"*(5+len(str(scan)))}')
print("\n##############\nSetup instance\n##############")
print(setup)
print("\n#################\nDetector instance\n#################")
print(setup.detector)

##########################
# Initialize the logfile #
##########################
if simulation:
    setup.detector.datadir, setup.detector.datadir, setup.detector.savedir = (
        root_folder,
    ) * 3

setup.create_logfile(
    scan_number=scan, root_folder=root_folder, filename=setup.detector.specfile
)

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

###################################
# load experimental data and mask #
###################################
plt.ion()
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    initialdir=setup.detector.scandir,
    title="Select diffraction pattern",
    filetypes=[("NPZ", "*.npz"), ("NPY", "*.npy")],
)
diff_pattern, _ = util.load_file(file_path)
diff_pattern = diff_pattern.astype(float)

file_path = filedialog.askopenfilename(
    initialdir=setup.detector.scandir,
    title="Select mask",
    filetypes=[("NPZ", "*.npz"), ("NPY", "*.npy")],
)
try:
    mask, _ = util.load_file(file_path)
except ValueError:
    mask = np.zeros(diff_pattern.shape, dtype=int)

###########################################################
# crop the diffraction pattern and the mask to compensate #
# the "auto_center_resize" option used in PyNX            #
###########################################################
# The shape will be equal to 'roi_final' parameter of the .cxi file
valid.valid_container(
    obj=crop_roi,
    container_types=(list, tuple),
    length=6,
    item_types=int,
    allow_none=True,
    name="prtf_bcdi.py",
)
if crop_roi is not None:
    diff_pattern = diff_pattern[
        crop_roi[0] : crop_roi[1], crop_roi[2] : crop_roi[3], crop_roi[4] : crop_roi[5]
    ]
    mask = mask[
        crop_roi[0] : crop_roi[1], crop_roi[2] : crop_roi[3], crop_roi[4] : crop_roi[5]
    ]

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
diff_pattern = util.bin_data(
    array=diff_pattern, binning=phasing_binning, debugging=False
)
mask = util.bin_data(array=mask, binning=phasing_binning, debugging=False)

(
    numz,
    numy,
    numx,
) = diff_pattern.shape
# this shape will be used for the calculation of q values
print(
    f"\nMeasured data shape = {numz}, {numy}, {numx},"
    f" Max(measured amplitude)={np.sqrt(diff_pattern).max():.1f}"
)
diff_pattern[np.nonzero(mask)] = 0

######################################################
# find the center of mass of the diffraction pattern #
######################################################
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

plt.figure()
plt.imshow(np.log10(np.sqrt(diff_pattern).sum(axis=0)), cmap=my_cmap, vmin=0, vmax=3.5)
plt.title("abs(diffraction amplitude).sum(axis=0)")
plt.colorbar()
plt.pause(0.1)

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
        "\nThe shape of q values and the shape of the diffraction pattern"
        " are different: check binning parameters!"
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

#############################
# load reconstructed object #
#############################
file_path = filedialog.askopenfilename(
    initialdir=setup.detector.savedir,
    title="Select reconstructions (prtf)",
    filetypes=[("NPZ", "*.npz"), ("NPY", "*.npy"), ("CXI", "*.cxi"), ("HDF5", "*.h5")],
)

obj, extension = util.load_file(file_path)
print("Opening ", file_path)

if extension == ".h5":
    if comment:
        comment += "_mode"
    else:
        comment = "mode"

# check if the shape of the real space object is the same as
# the measured diffraction pattern. The real space object may have been further
# cropped to a tight support, to save memory space.
if obj.shape != diff_pattern.shape:
    print(
        "Reconstructed object shape = ",
        obj.shape,
        "different from the experimental diffraction pattern: crop/pad",
    )
    obj = util.crop_pad(array=obj, output_shape=diff_pattern.shape, debugging=False)

# calculate the retrieved diffraction amplitude
phased_fft = fftshift(fftn(obj)) / (
    np.sqrt(numz) * np.sqrt(numy) * np.sqrt(numx)
)  # complex amplitude
del obj
gc.collect()

if debug:
    plt.figure()
    plt.imshow(np.log10(abs(phased_fft).sum(axis=0)), cmap=my_cmap, vmin=0, vmax=3.5)
    plt.colorbar()
    plt.title("abs(retrieved amplitude).sum(axis=0) before alignment")
    plt.pause(0.1)

if align_pattern:
    # align the reconstruction with the initial diffraction data
    phased_fft, _ = reg.align_diffpattern(
        reference_data=diff_pattern,
        data=phased_fft,
        interpolation_method="subpixel",
    )

phased_fft[np.nonzero(mask)] = 0  # do not take mask voxels into account
print(f"Max(retrieved amplitude) = {abs(phased_fft).max():.1f}")
phased_com_z, phased_com_y, phased_com_x = center_of_mass(abs(phased_fft))
print(
    f"COM of the retrieved diffraction pattern after masking: {phased_com_z:.2f},"
    f" {phased_com_y:.2f}, {phased_com_x:.2f}\n"
)
del mask
gc.collect()

if normalize_prtf:
    print(
        "Normalizing the phased data to the sqrt of the measurement"
        " at the center of mass of the diffraction pattern ..."
    )
    norm_factor = (
        np.sqrt(diff_pattern[z0 - 3 : z0 + 4, y0 - 3 : y0 + 4, x0 - 3 : x0 + 4]).sum()
        / abs(phased_fft[z0 - 3 : z0 + 4, y0 - 3 : y0 + 4, x0 - 3 : x0 + 4]).sum()
    )
    print(f"Normalization factor = {norm_factor:.4f}")
    phased_fft = phased_fft * norm_factor

plt.figure()
plt.imshow(np.log10(abs(phased_fft).sum(axis=0)), cmap=my_cmap, vmin=0, vmax=3.5)
plt.title("abs(retrieved amplitude).sum(axis=0)")
plt.colorbar()
plt.pause(0.1)

gu.combined_plots(
    tuple_array=(np.sqrt(diff_pattern), phased_fft),
    tuple_sum_frames=False,
    tuple_sum_axis=(0, 0),
    tuple_width_v=None,
    tuple_width_h=None,
    tuple_colorbar=False,
    tuple_vmin=(-1, -1),
    tuple_vmax=np.nan,
    tuple_title=("sqrt(measurement)", "phased_fft"),
    tuple_scale="log",
)

#########################
# calculate the 3D PRTF #
#########################
diff_pattern[diff_pattern == 0] = np.nan  # discard zero valued pixels
prtf_matrix = abs(phased_fft) / np.sqrt(diff_pattern)

gu.multislices_plot(
    prtf_matrix,
    sum_frames=False,
    plot_colorbar=True,
    cmap=my_cmap,
    title="prtf_matrix",
    scale="linear",
    vmin=0,
    reciprocal_space=True,
)

#######################################################################
# interactive interface to check the PRTF along particular directions #
#######################################################################
if flag_interact:
    if interpolate_nans:  # filter out the nan values
        linecut_prtf = np.copy(prtf_matrix)
        print("\nInterpolating the 3D PRTF on nan values, it will take some time ...")
        print(f"nb_nans before interpolation = {np.isnan(linecut_prtf).sum()}")
        linecut_prtf, nb_filtered, _ = util.mean_filter(
            data=linecut_prtf,
            nb_neighbours=1,
            interpolate="interp_isolated",
            min_count=0,
            extent=1,
            target_val=np.nan,
            debugging=debug,
        )
        print(f"nb_nans after = {np.isnan(linecut_prtf).sum()}")
        gu.multislices_plot(
            linecut_prtf,
            sum_frames=False,
            plot_colorbar=True,
            cmap=my_cmap,
            title="prtf_matrix after interpolation",
            scale="linear",
            vmin=0,
            reciprocal_space=True,
        )
        np.savez_compressed(
            setup.detector.savedir + "linecut_prtf.npz", data=linecut_prtf
        )
    else:
        linecut_prtf = prtf_matrix

    plt.ioff()
    max_colorbar = 5
    starting_point = [z0, y0, x0]
    endpoint = [0, 0, 0]
    cut = lc.linecut(
        prtf_matrix,
        indices=list(zip(starting_point, endpoint)),
        interp_order=1,
    )
    diff_pattern[np.isnan(diff_pattern)] = 0  # discard nans
    fig_prtf, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))
    fig_prtf.canvas.mpl_disconnect(fig_prtf.canvas.manager.key_press_handler_id)
    ax0.imshow(
        np.log10(diff_pattern.sum(axis=0)), vmin=0, vmax=max_colorbar, cmap=my_cmap
    )
    ax1.imshow(
        np.log10(diff_pattern.sum(axis=1)), vmin=0, vmax=max_colorbar, cmap=my_cmap
    )
    ax2.imshow(
        np.log10(diff_pattern.sum(axis=2)), vmin=0, vmax=max_colorbar, cmap=my_cmap
    )
    ax3.plot(
        np.linspace(
            distances_q[z0, y0, x0],
            distances_q[endpoint[0], endpoint[1], endpoint[2]],
            num=len(cut),
        ),
        cut,
        "-or",
        markersize=3,
    )
    ax3.axhline(
        y=1 / np.e, linestyle="dashed", color="k", linewidth=1
    )  # horizontal line at 1/e
    (plt0,) = ax0.plot([x0, endpoint[2]], [y0, endpoint[1]], "ro-")  # sum axis 0
    (plt1,) = ax1.plot([x0, endpoint[2]], [z0, endpoint[0]], "ro-")  # sum axis 1
    (plt2,) = ax2.plot([y0, endpoint[1]], [z0, endpoint[0]], "ro-")  # sum axis 2
    ax0.axis("scaled")
    ax1.axis("scaled")
    ax2.axis("scaled")
    ax3.axis("auto")
    ax0.set_title("horizontal=X  vertical=Y")
    ax1.set_title("horizontal=X  vertical=rocking curve")
    ax2.set_title("horizontal=Y  vertical=rocking curve")
    ax3.set_xlabel("q (1/A)")
    ax3.set_ylabel("PRTF")
    res_text = fig_prtf.text(0.55, 0.25, "", size=10)
    fig_prtf.text(0.55, 0.15, "click to read the resolution", size=10)
    fig_prtf.text(0.01, 0.8, "click to select\nthe endpoint", size=10)
    fig_prtf.text(0.01, 0.7, "q to quit\ns to save", size=10)
    plt.tight_layout()
    plt.connect("key_press_event", press_key)
    plt.connect("button_press_event", on_click)
    fig_prtf.set_facecolor(background_plot)
    plt.show()

#################################
# average over spherical shells #
#################################
print(
    f"\nDistance max: {distances_q.max():.2f}(1/A) at:"
    f" {np.unravel_index(abs(distances_q).argmax(), distances_q.shape)}"
)
nb_bins = numz // 3
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
