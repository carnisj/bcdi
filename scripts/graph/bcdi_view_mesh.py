#!/usr/bin/env python3

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector

from bcdi.experiment.setup import Setup
from bcdi.graph.colormap import ColormapFactory

matplotlib.use("Qt5Agg")

helptext = """
Open mesh scans and plot interactively the integrated intensity vs. motor positions for
a user-defined region of interest.
"""

scan = 26  # scan number as it appears in the folder name
sample_name = "B10_syn_S5"  # without _ at the end
root_folder = "D:/data/P10_Longfei/"
savedir = ""  # images will be saved here, leave it to ''
# otherwise (default to data directory's parent)
crop_roi = [
    550,
    1050,
    0,
    2070,
]  # only this region of interest of the detector will be considered (unbinned indices).
# Leave [] to use the full detector. [ystart, ystop, xstart, xstop]
sum_roi = [550, 1050, 0, 2070]  # region of interest for integrating the intensity.
# [ystart, ystop, xstart, xstop], in the unbinned detector indices.
# Leave it to [] to use the full detector
normalize_flux = False  # will normalize the intensity by the default monitor
threshold = 2  # data <= threshold will be set to 0
###########################
# mesh related parameters #
###########################
fast_motor = "hpx"  # fast scanning motor for the mesh
nb_fast = 51  # number of steps for the fast scanning motor
slow_motor = "hpy"  # slow scanning motor for the mesh
nb_slow = 51  # number of steps for the slow scanning motor
###########################
# plot related parameters #
###########################
background_plot = "0.7"  # in level of grey in [0,1], 0 being dark. For visual comfort
fast_axis = "horizontal"  # 'vertical' to plot the fast scanning motor vertically,
# 'horizontal' otherwise
invert_xaxis = False  # True to inverse the horizontal axis
invert_yaxis = True  # True to inverse the vertical axis
###############################
# beamline related parameters #
###############################
beamline = "P10"
# name of the beamlisne, used for data loading and normalization by monitor
# supported beamlines: 'P10' only for now
is_series = False  # specific to series measurement at P10
specfile_name = ""
# .spec for ID01, .fio for P10, alias_dict.txt for SIXS_2018,
# not used for CRISTAL and SIXS_2019
# template for ID01: name of the spec file without '.spec'
# template for SIXS_2018: full path of the alias dictionnary 'alias_dict.txt',
# typically: root_folder + 'alias_dict.txt'
# template for SIXS_2019: ''
# template for P10: ''
# template for CRISTAL: ''
###############################
# detector related parameters #
###############################
detector = "Eiger4M"  # "Eiger2M" or "Maxipix" or "Eiger4M"
binning = [
    4,
    4,
]  # binning (detector vertical axis, detector horizontal axis)
# applied during data loading
template_imagefile = "_master.h5"
# template for ID01: 'data_mpx4_%05d.edf.gz' or 'align_eiger2M_%05d.edf.gz'
# template for SIXS_2018: 'align.spec_ascan_mu_%05d.nxs'
# template for SIXS_2019: 'spare_ascan_mu_%05d.nxs'
# template for Cristal: 'S%d.nxs'
# template for P10: '_master.h5'
##########################
# end of user parameters #
##########################


def onclick(click_event):
    """
    Process mouse click events in the interactive line plot

    :param click_event: mouse click event
    """
    global fast_motor, slow_motor, ax1, motor_text, figure

    if click_event.inaxes == ax1:  # click in the 2D scanning map
        motor_text.remove()
        if fast_axis == "horizontal":
            motor_text = figure.text(
                0.40,
                0.95,
                f"{fast_motor} = {click_event.xdata:.2f}, "
                f"{slow_motor} = {click_event.ydata:.2f}",
                size=12,
            )
        else:
            motor_text = figure.text(
                0.40,
                0.95,
                f"{fast_motor} = {click_event.ydata:.2f}, "
                f"{slow_motor} = {click_event.xdata:.2f}",
                size=12,
            )
        plt.draw()


def onselect(click, release):
    """
    Process mouse click and release events in the interactive plot

    :param click: position of the mouse click event
    :param release: position of the mouse release event
    """
    global ax1, data, nb_slow, nb_fast, my_cmap, min_fast, min_slow, max_fast, max_slow
    global fast_motor, binning, rectangle
    global slow_motor, ny, nx, invert_xaxis, invert_yaxis, motor_text, sum_int, figure

    y_start, y_stop, x_start, x_stop = (
        int(click.ydata),
        int(release.ydata),
        int(click.xdata),
        int(release.xdata),
    )

    rectangle.extents = (
        x_start,
        x_stop,
        y_start,
        y_stop,
    )  # in the unbinned full detector pixel coordinates,
    # extents (xmin, xmax, ymin, ymax)

    # remove the offset due to crop_roi
    y_start, y_stop = y_start - crop_roi[0], y_stop - crop_roi[0]
    x_start, x_stop = x_start - crop_roi[2], x_stop - crop_roi[2]

    # correct for data binning
    y_start, y_stop = y_start // binning[0], y_stop // binning[0]
    x_start, x_stop = x_start // binning[1], x_stop // binning[1]

    ax1.cla()
    if fast_axis == "vertical":
        sum_int = (
            data[:, y_start:y_stop, x_start:x_stop]
            .sum(axis=(1, 2))
            .reshape((nb_fast, nb_slow))
        )
        # extent (left, right, bottom, top)
        ax1.imshow(
            np.log10(sum_int),
            cmap=my_cmap,
            extent=[min_slow, max_slow, max_fast, min_fast],
        )
        ax1.set_xlabel(slow_motor)
        ax1.set_ylabel(fast_motor)
    else:
        sum_int = (
            data[:, y_start:y_stop, x_start:x_stop]
            .sum(axis=(1, 2))
            .reshape((nb_slow, nb_fast))
        )
        # extent (left, right, bottom, top)
        ax1.imshow(
            np.log10(sum_int),
            cmap=my_cmap,
            extent=[min_fast, max_fast, max_slow, min_slow],
        )
        ax1.set_xlabel(fast_motor)
        ax1.set_ylabel(slow_motor)
    if invert_xaxis:
        ax1.invert_xaxis()
    if invert_yaxis:
        ax1.invert_yaxis()
    motor_text.remove()
    motor_text = figure.text(0.40, 0.95, "", size=12)
    ax1.axis("scaled")
    ax1.set_title("integrated intensity in the ROI")
    plt.draw()


def press_key(event):
    """
    Process key press events in the interactive plot

    :param event: button press event
    """
    global sumdata, max_colorbar, ax0, my_cmap, figure, rectangle, onselect, rectprops

    if event.key == "right":
        max_colorbar = max_colorbar + 1
    elif event.key == "left":
        max_colorbar = max_colorbar - 1
        max_colorbar = max(max_colorbar, 1)
    extents = rectangle.extents
    xmin0, xmax0 = ax0.get_xlim()
    ymin0, ymax0 = ax0.get_ylim()

    ax0.cla()
    ax0.imshow(
        np.log10(sumdata),
        vmin=0,
        vmax=max_colorbar,
        cmap=my_cmap,
        extent=[crop_roi[2], crop_roi[3], crop_roi[1], crop_roi[0]],
    )  # unbinned pixel coordinates
    # extent (left, right, bottom, top)
    ax0.set_title("detector plane (sum)")
    ax0.axis("scaled")
    ax0.set_xlim(xmin0, xmax0)
    ax0.set_ylim(ymin0, ymax0)
    plt.draw()
    rectangle = RectangleSelector(
        ax0,
        onselect,
        drawtype="box",
        useblit=False,
        button=[1],
        interactive=True,
        rectprops=rectprops,
    )  # don't use middle and right buttons
    rectangle.to_draw.set_visible(True)
    figure.canvas.draw()
    rectangle.extents = extents


###################
# define colormap #
###################
bad_color = "1.0"  # white background
my_cmap = ColormapFactory(bad_color=bad_color).cmap
plt.ion()

########################
# initialize the setup #
########################
setup = Setup(
    beamline_name=beamline,
    is_series=is_series,
    detector_name=detector,
    datadir="",
    template_imagefile=template_imagefile,
    sum_roi=sum_roi,
    binning=[1, binning[0], binning[1]],
)

crop_roi = crop_roi or (0, setup.detector.nb_pixel_y, 0, setup.detector.nb_pixel_x)
setup.detector.roi = crop_roi

if setup.beamline == "P10":
    specfile_name = sample_name + f"_{scan:05d}"
    homedir = root_folder + specfile_name + "/"
    setup.detector.datadir = homedir + "e4m/"
    template_imagefile = specfile_name + template_imagefile
    setup.detector.template_imagefile = template_imagefile
elif setup.beamline in {"SIXS_2018", "SIXS_2019"}:
    homedir = root_folder
    setup.detector.datadir = homedir + "align/"
else:
    homedir = root_folder + sample_name + str(scan) + "/"
    setup.detector.datadir = homedir + "data/"

if savedir == "":
    savedir = os.path.abspath(os.path.join(setup.detector.datadir, os.pardir)) + "/"

setup.detector.savedir = savedir
print("savedir: ", savedir)

setup.create_logfile(scan_number=scan, root_folder=root_folder, filename=specfile_name)

#########################
# check some parameters #
#########################
if fast_axis not in {"vertical", "horizontal"}:
    raise ValueError("fast_axis parameter value not supported")
if len(sum_roi) == 0:
    sum_roi = [0, setup.detector.nb_pixel_y, 0, setup.detector.nb_pixel_x]

print(f"sum_roi before binning and offset correction = {sum_roi}")

# correct the offset due to crop_roi and take into account data binning
sum_roi = (
    (sum_roi[0] - crop_roi[0]) // binning[0],
    (sum_roi[1] - crop_roi[0]) // binning[0],
    (sum_roi[2] - crop_roi[2]) // binning[1],
    (sum_roi[3] - crop_roi[2]) // binning[1],
)
print(f"sum_roi after binning and offset correction = {sum_roi}")

if not (
    sum_roi[0] >= 0
    and sum_roi[1] <= (crop_roi[1] - crop_roi[0]) // binning[0]
    and sum_roi[2] >= 0
    and sum_roi[3] <= (crop_roi[3] - crop_roi[2]) // binning[1]
):
    raise ValueError("sum_roi setting does not match the binned detector size")

#############
# load data #
#############
data, mask, monitor, frames_logical = setup.loader.load_check_dataset(
    scan_number=scan,
    setup=setup,
    bin_during_loading=True,
    debugging=False,
)
nz, ny, nx = data.shape
print(f"Data shape: ({nz}, {ny}, {nx})")
data[np.nonzero(mask)] = 0

#######################
# intensity threshold #
#######################
data[data <= threshold] = 0

########################
# load motor positions #
########################
fast_positions = setup.loader.read_device(
    setup=setup, device_name=fast_motor, scan_number=scan
)
slow_positions = setup.loader.read_device(
    setup=setup, device_name=slow_motor, scan_number=scan
)

min_fast, max_fast = fast_positions[0], fast_positions[-1]
min_slow, max_slow = slow_positions[0], slow_positions[-1]

if len(fast_positions) != nz:
    raise ValueError(
        f"Number of fast scanning motor steps: {nb_fast}"
        f" incompatible with data shape: {nz}"
    )
if len(slow_positions) != nz:
    raise ValueError(
        f"Number of slow scanning motor steps: {nb_slow}"
        f" incompatible with data shape: {nz}"
    )

####################
# interactive plot #
####################
sumdata = data.sum(axis=0)
max_colorbar = 5
rectprops = dict(edgecolor="black", fill=False)  # rectangle properties
plt.ioff()

figure = plt.figure()  # figsize=(12, 9))
ax0 = figure.add_subplot(121)
ax1 = figure.add_subplot(122)
figure.canvas.mpl_disconnect(figure.canvas.manager.key_press_handler_id)
original_data = np.copy(data)
ax0.imshow(
    np.log10(sumdata),
    cmap=my_cmap,
    vmin=0,
    vmax=max_colorbar,
    extent=[crop_roi[2], crop_roi[3], crop_roi[1], crop_roi[0]],
)  # unbinned pixel coordinates
# extent (left, right, bottom, top)

if fast_axis == "vertical":
    sum_int = (
        data[:, sum_roi[0] : sum_roi[1], sum_roi[2] : sum_roi[3]]
        .sum(axis=(1, 2))
        .reshape((nb_fast, nb_slow))
    )
    # extent (left, right, bottom, top)
    ax1.imshow(
        np.log10(sum_int), cmap=my_cmap, extent=[min_slow, max_slow, max_fast, min_fast]
    )
    ax1.set_xlabel(slow_motor)
    ax1.set_ylabel(fast_motor)
else:
    sum_int = (
        data[:, sum_roi[0] : sum_roi[1], sum_roi[2] : sum_roi[3]]
        .sum(axis=(1, 2))
        .reshape((nb_slow, nb_fast))
    )
    # extent (left, right, bottom, top)
    ax1.imshow(
        np.log10(sum_int), cmap=my_cmap, extent=[min_fast, max_fast, max_slow, min_slow]
    )
    ax1.set_xlabel(fast_motor)
    ax1.set_ylabel(slow_motor)
if invert_xaxis:
    ax1.invert_xaxis()
if invert_yaxis:
    ax1.invert_yaxis()
ax0.axis("scaled")
ax1.axis("scaled")
ax0.set_title("sum of all images")
ax1.set_title("integrated intensity in the ROI")
motor_text = figure.text(0.40, 0.95, "", size=12)
plt.tight_layout()
plt.connect("key_press_event", press_key)
plt.connect("button_press_event", onclick)
rectangle = RectangleSelector(
    ax0,
    onselect,
    drawtype="box",
    useblit=False,
    button=[1],
    interactive=True,
    rectprops=rectprops,
)  # don't use middle and right buttons
rectangle.to_draw.set_visible(True)
figure.canvas.draw()
rectangle.extents = (
    sum_roi[2] * binning[1] + crop_roi[2],
    sum_roi[3] * binning[1] + crop_roi[2],
    sum_roi[0] * binning[0] + crop_roi[0],
    sum_roi[1] * binning[0] + crop_roi[0],
)
# in the unbinned full detector pixel coordinates, extents (xmin, xmax, ymin, ymax)
figure.set_facecolor(background_plot)
plt.show()
