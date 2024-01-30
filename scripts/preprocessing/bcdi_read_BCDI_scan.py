#!/usr/bin/env python3

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

import bcdi.graph.graph_utils as gu
import bcdi.preprocessing.bcdi_utils as bu
import bcdi.utils.utilities as util
from bcdi.experiment.setup import Setup
from bcdi.graph.colormap import ColormapFactory

helptext = """
Open a rocking curve data, plot the mask, the monitor and the stack along the first
axis.

It is usefull when you want to localize the Bragg peak for ROI determination.

Supported beamlines: ESRF ID01, PETRAIII P10, SOLEIL SIXS, SOLEIL CRISTAL.
"""

scan = 128
root_folder = "D:/data/P10_2nd_test_isosurface_Dec2020/data_nanolab/"
data_dir = None
# leave None to use the beamline default. It will look for the data at this location
sample_name = "PtNP1"  # string in front of the scan number in the folder name
save_dir = root_folder + "dataset_1_newpsf/test/"
# images will be saved here, leave it to None otherwise
# (default to data directory's parent)
save_mask = False  # set to True to save the mask
debug = True  # True to see more plots
binning = (1, 1, 1)  # binning to apply to the data
# (stacking dimension, detector vertical axis, detector horizontal axis)
###############################
# beamline related parameters #
###############################
beamline = "P10"
# name of the beamline, used for data loading and normalization by monitor
# supported beamlines: 'ID01', 'SIXS_2018', 'SIXS_2019', 'CRISTAL', 'P10', 'NANOMAX'
actuators = None  # {'rocking_angle': 'actuator_1_3'}
# Optional dictionary that can be used to define the entries corresponding to
# actuators in data files
# (useful at CRISTAL where the location of data keeps changing)
# e.g.  {'rocking_angle': 'actuator_1_3', 'detector': 'data_04', 'monitor': 'data_05'}
custom_scan = False  # True for a stack of images acquired without scan,
# e.g. with ct in a macro (no info in spec file)
custom_images = np.arange(11353, 11453, 1)  # list of image numbers for the custom_scan
custom_monitor = np.ones(len(custom_images))
# monitor values for normalization for the custom_scan
custom_motors = {
    "eta": np.linspace(16.989, 18.989, num=100, endpoint=False),
    "phi": 0,
    "nu": -0.75,
    "delta": 36.65,
}
# ID01: eta, phi, nu, delta
# CRISTAL: mgomega, gamma, delta
# P10: om, phi, chi, mu, gamma, delta
# SIXS: beta, mu, gamma, delta

rocking_angle = "outofplane"  # "outofplane" or "inplane"
is_series = True  # specific to series measurement at P10
specfile_name = ""
# .spec for ID01, .fio for P10, alias_dict.txt for SIXS_2018,
# not used for CRISTAL and SIXS_2019
# template for ID01: name of the spec file without '.spec'
# template for SIXS_2018: full path of the alias dictionnary 'alias_dict.txt',
# typically: root_folder + 'alias_dict.txt'
# template for all other beamlines: ''
###############################
# detector related parameters #
###############################
detector = "Eiger4M"  # "Eiger2M" or "Maxipix" or "Eiger4M" or 'Merlin'
x_bragg = 1355  # horizontal pixel number of the Bragg peak,
# leave None for automatic detection (using the max)
y_bragg = 796  # vertical pixel number of the Bragg peak,
# leave None for automatic detection (using the max)
roi_detector = [y_bragg - 200, y_bragg + 200, x_bragg - 200, x_bragg + 200]
# roi_detector = [y_bragg - 168, y_bragg + 168, x_bragg - 140, x_bragg + 140]  # CH5309
# roi_detector = [552, 1064, x_bragg - 240, x_bragg + 240]  # P10 2018
# roi_detector = [y_bragg - 290, y_bragg + 350, x_bragg - 350, x_bragg + 350]  # PtRh Ar
# [Vstart, Vstop, Hstart, Hstop]
# leave None to use the full detector. Use with center_fft='skip'
# if you want this exact size.
peak_method = "max"  # Bragg peak determination: 'max', 'com' or 'maxcom'.
normalize = "monitor"
# 'monitor' to return the default monitor values, 'skip' to do nothing
high_threshold = 500000  # everything above will be considered as hotpixel
hotpixels_file = ""  # root_folder + 'hotpixels_cristal.npz'
flatfield_file = ""  # root_folder + "flatfield_maxipix_8kev.npz"  #
template_imagefile = "_master.h5"
# template for ID01: 'data_mpx4_%05d.edf.gz' or 'align_eiger2M_%05d.edf.gz'
# template for SIXS_2018: 'align.spec_ascan_mu_%05d.nxs'
# template for SIXS_2019: 'spare_ascan_mu_%05d.nxs'
# template for Cristal: 'S%d.nxs'
# template for P10: '_master.h5'
# template for NANOMAX: '%06d.h5'
######################
# setup for the plot #
######################
vmin = 0  # min of the colorbar (log scale)
vmax = 6  # max of the colorbar (log scale)
low_threshold = 1  # everthing <= 1 will be set to 0 in the plot
width = None  # [50, 50]  # [vertical, horizontal], leave None for default
# half width in pixels of the region of interest centered on the peak for the plot
##################################
# end of user-defined parameters #
##################################

###################
# define colormap #
###################
bad_color = "1.0"  # white background
my_cmap = ColormapFactory(bad_color=bad_color).cmap
plt.ion()

########################################
# initialize and check some parameters #
########################################
save_dirname = "pynxraw"
flatfield = util.load_flatfield(flatfield_file)
hotpix_array = util.load_hotpixels(hotpixels_file)
if normalize not in {"skip", "monitor"}:
    raise ValueError(
        f"Invalid setting {normalize} for normalize,"
        " allowed values are 'skip' and 'monitor'"
    )

####################
# Initialize setup #
####################
setup = Setup(
    beamline_name=beamline,
    rocking_angle=rocking_angle,
    custom_scan=custom_scan,
    custom_images=custom_images,
    custom_monitor=custom_monitor,
    custom_motors=custom_motors,
    actuators=actuators,
    is_series=is_series,
    detector_name=detector,
    template_imagefile=template_imagefile,
    roi=roi_detector,
    binning=binning,
)

########################################
# print the current setup and detector #
########################################
print("\n##############\nSetup instance\n##############")
print(setup)
print("\n#################\nDetector instance\n#################")
print(setup.detector)

########################
# initialize the paths #
########################
setup.init_paths(
    sample_name=sample_name,
    scan_number=scan,
    root_folder=root_folder,
    save_dir=save_dir,
    save_dirname=save_dirname,
    specfile_name=specfile_name,
    template_imagefile=template_imagefile,
    data_dir=data_dir,
)

setup.create_logfile(
    scan_number=scan, root_folder=root_folder, filename=setup.detector.specfile
)


#################
# load the data #
#################
data, mask, monitor, frames_logical = setup.loader.load_check_dataset(
    scan_number=scan,
    setup=setup,
    flatfield=flatfield,
    hotpixels=hotpix_array,
    normalize=normalize,
    debugging=debug,
)

numz, numy, numx = data.shape
print(f"Data shape: ({numz}, {numy}, {numx})")

##########################
# apply photon threshold #
##########################
if high_threshold != 0:
    nb_thresholded = (data > high_threshold).sum()
    mask[data > high_threshold] = 1
    data[data > high_threshold] = 0
    print(f"Applying photon threshold, {nb_thresholded} high intensity pixels masked")

######################################################
# calculate rocking curve and fit it to get the FWHM #
######################################################
if data.ndim == 3:
    tilt, _, _, _ = setup.read_logfile(scan_number=scan)
    rocking_curve = np.zeros(numz)

    z0, y0, x0 = bu.find_bragg(data, peak_method=peak_method)

    if x_bragg is None:  # Bragg peak position not defined by the user, use the max
        x_bragg = x0
    else:  # calculate the new position with binning and cropping
        x_bragg = int(
            (x_bragg - setup.detector.roi[2])
            / (setup.detector.preprocessing_binning[2] * setup.detector.binning[2])
        )
    if y_bragg is None:  # Bragg peak position not defined by the user, use the max
        y_bragg = y0
    else:  # calculate the new position with binning and cropping
        y_bragg = int(
            (y_bragg - setup.detector.roi[0])
            / (setup.detector.preprocessing_binning[1] * setup.detector.binning[1])
        )

    peak_int = int(data[z0, y0, x0])
    print(
        "Bragg peak (indices in the eventually binned ROI) at (z, y, x):"
        f" {z0}, {y0}, {x0}, intensity = {peak_int}"
    )

    for idx in range(numz):
        rocking_curve[idx] = data[
            idx, y_bragg - 50 : y_bragg + 50, x_bragg - 50 : x_bragg + 50
        ].sum()
    plot_title = f"Rocking curve for a ROI centered on (y, x): ({y_bragg}, {x_bragg})"

    z0 = np.unravel_index(rocking_curve.argmax(), rocking_curve.shape)[0]

    interpolation = interp1d(tilt, rocking_curve, kind="cubic")
    interp_points = 5 * numz
    interp_tilt = np.linspace(tilt.min(), tilt.max(), interp_points)
    interp_curve = interpolation(interp_tilt)
    interp_fwhm = (
        len(np.argwhere(interp_curve >= interp_curve.max() / 2))
        * (tilt.max() - tilt.min())
        / (interp_points - 1)
    )
    print(f"FWHM by interpolation = {interp_fwhm:.3f} deg")

    _, (ax0, ax1) = plt.subplots(2, 1, sharex="col", figsize=(10, 5))
    ax0.plot(tilt, rocking_curve, ".")
    ax0.plot(interp_tilt, interp_curve)
    ax0.set_ylabel("Integrated intensity")
    ax0.legend(("data", "interpolation"))
    ax0.set_title(plot_title)
    ax1.plot(tilt, np.log10(rocking_curve), ".")
    ax1.plot(interp_tilt, np.log10(interp_curve))
    ax1.set_xlabel("Rocking angle (deg)")
    ax1.set_ylabel("Log(integrated intensity)")
    ax0.legend(("data", "interpolation"))
    plt.pause(0.1)

    # apply low threshold
    data[data <= low_threshold] = 0
    # data = data[data.shape[0]//2, :, :]
    # select the first frame e.g. for detector mesh scan
    data = data.sum(axis=0)  # concatenate along the axis of the rocking curve
    title = f"data.sum(axis=0)   peak method={peak_method}\n"
else:  # 2D
    y0, x0 = bu.find_bragg(data, peak_method=peak_method)
    peak_int = int(data[y0, x0])
    print(
        f"Bragg peak (indices in the eventually binned ROI) at (y, x): {y0}, {x0},"
        f" intensity = {peak_int}"
    )
    # apply low threshold
    data[data <= low_threshold] = 0
    title = f"peak method={peak_method}\n"

######################################################################################
# cehck the width parameter for plotting the region of interest centered on the peak #
######################################################################################
if width is None:
    width = [y0, numy - y0, x0, numx - x0]  # plot the full range
else:
    width = [
        min(width[0], y0, numy - y0),
        min(width[0], y0, numy - y0),
        min(width[1], x0, numx - x0),
        min(width[1], x0, numx - x0),
    ]
print(f"width for plotting: {width}")

############################################
# plot mask, monitor and concatenated data #
############################################
if save_mask:
    np.savez_compressed(setup.detector.savedir + "hotpixels.npz", mask=mask)

gu.combined_plots(
    tuple_array=(monitor, mask),
    tuple_sum_frames=False,
    tuple_sum_axis=(0, 0),
    tuple_width_v=None,
    tuple_width_h=None,
    tuple_colorbar=(True, False),
    tuple_vmin=np.nan,
    tuple_vmax=np.nan,
    tuple_title=("monitor", "mask"),
    tuple_scale="linear",
    cmap=my_cmap,
    ylabel=("Counts (a.u.)", ""),
)

max_y, max_x = np.unravel_index(abs(data).argmax(), data.shape)
print(
    f"Max of the concatenated data along axis 0 at (y, x): ({max_y}, {max_x})  "
    f"Max = {int(data[max_y, max_x])}"
)

# plot the region of interest centered on the peak
# extent (left, right, bottom, top)
fig, ax = plt.subplots(nrows=1, ncols=1)
plot = ax.imshow(
    np.log10(data[y0 - width[0] : y0 + width[1], x0 - width[2] : x0 + width[3]]),
    vmin=vmin,
    vmax=vmax,
    cmap=my_cmap,
    extent=[
        x0 - width[2] - 0.5,
        x0 + width[3] - 0.5,
        y0 + width[1] - 0.5,
        y0 - width[0] - 0.5,
    ],
)
ax.set_title(f"{title} Peak at (y, x): ({y0},{x0})   Bragg peak value = {peak_int}")
gu.colorbar(plot)
fig.savefig(setup.detector.savedir + f"sum_S{scan}.png")
plt.show()
