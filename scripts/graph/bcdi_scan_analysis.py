#!/usr/bin/env python3

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.widgets import RectangleSelector

import bcdi.utils.utilities as util
from bcdi.experiment.setup import Setup
from bcdi.graph.colormap import ColormapFactory

matplotlib.use("Qt5Agg")

helptext = """
Open 1Dscans and plot interactively the integrated intensity vs. motor positions in a
user-defined region of interest.
"""

scan = 7  # scan number as it appears in the folder name
sample_name = "p15_2"  # without _ at the end
root_folder = "D:/data/P10_isosurface/data/"
savedir = ""  # images will be saved here,
# leave it to '' otherwise (default to data directory's parent)
sum_roi = [
    100,
    300,
    300,
    500,
]  # integrate the intensity in this ROI (in units of the binned detector pixels).
# [ystart, ystop, xstart, xstop]
# Leave it to [] to use the full detector
motor_name = "hpx"  # scanned motor name
normalize_flux = False  # will normalize the intensity by the default monitor
###########################
# plot related parameters #
###########################
background_plot = "0.7"  # in level of grey in [0,1], 0 being dark. For visual comfort
scale = "linear"  # scale for 1D plots, 'linear' or 'log'
invert_xaxis = False  # True to inverse the horizontal axis
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
]
# binning (detector vertical axis, detector horizontal axis) applied during data loading
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
    global sum_roi, vline, ax1, ax2, index_peak, motor_positions, data, my_cmap, sum_int
    global figure, scale, motor_text, max_text

    if click_event.inaxes == ax1:  # click in the line plot
        index_peak = util.find_nearest(motor_positions, click_event.xdata)
        vline.remove()
        if scale == "linear":
            vline = ax1.vlines(
                x=motor_positions[index_peak],
                ymin=sum_int.min(),
                ymax=sum_int[index_peak],
                colors="r",
                linestyle="dotted",
            )
        else:  # 'log'
            vline = ax1.vlines(
                x=motor_positions[index_peak],
                ymin=np.log10(sum_int.min()),
                ymax=np.log10(sum_int[index_peak]),
                colors="r",
                linestyle="dotted",
            )
        ax2.cla()
        ax2.imshow(
            np.log10(
                data[index_peak, sum_roi[0] : sum_roi[1], sum_roi[2] : sum_roi[3]]
            ),
            cmap=my_cmap,
            vmin=0,
        )
        ax2.axis("scaled")
        ax2.set_title("ROI at line")
        motor_text.remove()
        motor_text = figure.text(
            0.70,
            0.75,
            motor_name + f" = {motor_positions[index_peak]:.2f}",
            size=10,
        )
        max_text.remove()
        max_text = figure.text(
            0.70,
            0.70,
            "ROI max at line = "
            f"{data[index_peak, sum_roi[0]:sum_roi[1], sum_roi[2]:sum_roi[3]].max():d}",
            size=10,
        )
        plt.draw()


def onselect(click, release):
    """
    Process mouse click and release events in the interactive plot

    :param click: position of the mouse click event
    :param release: position of the mouse release event
    """
    global ax1, ax2, data, my_cmap, motor_name, motor_positions, scale, invert_xaxis
    global sum_roi, roi_text, max_text, figure, index_peak, sum_int, vline, nz

    sum_roi = int(click.ydata), int(release.ydata), int(click.xdata), int(release.xdata)
    sum_int = data[:, sum_roi[0] : sum_roi[1], sum_roi[2] : sum_roi[3]].sum(axis=(1, 2))
    index_peak = np.unravel_index(sum_int.argmax(), nz)[0]

    ax1.cla()
    if scale == "linear":
        ax1.plot(motor_positions, sum_int, marker=".")
        ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1e"))
        vline = ax1.vlines(
            x=motor_positions[index_peak],
            ymin=sum_int.min(),
            ymax=sum_int[index_peak],
            colors="r",
            linestyle="dotted",
        )
    else:  # 'log'
        ax1.plot(motor_positions, np.log10(sum_int), marker=".")
        vline = ax1.vlines(
            x=motor_positions[index_peak],
            ymin=np.log10(sum_int.min()),
            ymax=np.log10(sum_int[index_peak]),
            colors="r",
            linestyle="dotted",
        )

    ax1.set_xlabel(motor_name)
    ax1.set_ylabel("integrated intensity")
    if invert_xaxis:
        ax1.invert_xaxis()
    ax1.set_aspect("auto", adjustable="datalim", anchor="S", share=False)
    ax2.cla()
    ax2.imshow(
        np.log10(data[index_peak, sum_roi[0] : sum_roi[1], sum_roi[2] : sum_roi[3]]),
        cmap=my_cmap,
        vmin=0,
    )
    ax2.axis("scaled")
    ax2.set_title("ROI at line")
    roi_text.remove()
    roi_text = figure.text(
        0.70,
        0.80,
        "unbinned ROI [y0 y1 x0 x1]\n"
        f"[{sum_roi[0] * binning[0]:d}, "
        f"{sum_roi[1] * binning[0]:d}, "
        f"{sum_roi[2] * binning[1]:d}, "
        f"{sum_roi[3] * binning[1]:d}]",
        size=10,
    )
    max_text.remove()
    max_text = figure.text(
        0.70,
        0.70,
        "ROI max at line = "
        + f"{data[index_peak, sum_roi[0]:sum_roi[1], sum_roi[2]:sum_roi[3]].max():d}",
        size=10,
    )
    plt.draw()


def press_key(event):
    """
    Process key press events in the interactive plot

    :param event: button press event
    """
    global sumdata, max_colorbar, ax0, my_cmap

    if event.key == "right":
        max_colorbar = max_colorbar + 1
    elif event.key == "left":
        max_colorbar = max_colorbar - 1
        max_colorbar = max(max_colorbar, 1)

    ax0.cla()
    ax0.imshow(np.log10(sumdata), vmin=0, vmax=max_colorbar, cmap=my_cmap)
    ax0.set_title("detector plane (sum)")
    ax0.axis("scaled")
    plt.draw()


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
if len(sum_roi) == 0:
    sum_roi = [0, setup.detector.nb_pixel_y, 0, setup.detector.nb_pixel_x]

if not (
    sum_roi[0] >= 0
    and sum_roi[1] <= setup.detector.nb_pixel_y // binning[0]
    and sum_roi[2] >= 0
    and sum_roi[3] <= setup.detector.nb_pixel_x // binning[1]
):
    raise ValueError("sum_roi setting does not match the binned detector size")

if scale not in {"linear", "log"}:
    raise ValueError("Incorrect value for scale parameter")

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
print("Data shape: ", nz, ny, nx)
data[np.nonzero(mask)] = 0

########################
# load motor positions #
########################
motor_positions = setup.loader.read_device(
    setup=setup, device_name=motor_name, scan_number=scan
)

min_fast, max_fast = motor_positions[0], motor_positions[-1]

if len(motor_positions) != nz:
    raise ValueError(
        f"Number of fast scanning motor steps: {len(motor_positions)}"
        f" incompatible with data shape: {nz}"
    )

####################
# interactive plot #
####################
sumdata = data.sum(axis=0)
max_colorbar = 5
rectprops = dict(edgecolor="black", fill=False)  # rectangle properties
plt.ioff()

figure = plt.figure()
ax0 = figure.add_subplot(231)
ax1 = figure.add_subplot(212)
ax2 = figure.add_subplot(232)
figure.canvas.mpl_disconnect(figure.canvas.manager.key_press_handler_id)
original_data = np.copy(data)
ax0.imshow(np.log10(sumdata), cmap=my_cmap, vmin=0, vmax=max_colorbar)
ax0.axis("scaled")
ax0.set_title("sum of all images")
sum_int = data[:, sum_roi[0] : sum_roi[1], sum_roi[2] : sum_roi[3]].sum(axis=(1, 2))
index_peak = np.unravel_index(sum_int.argmax(), nz)[0]
if scale == "linear":
    ax1.plot(motor_positions, sum_int, marker=".")
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1e"))
    vline = ax1.vlines(
        x=motor_positions[index_peak],
        ymin=sum_int.min(),
        ymax=sum_int[index_peak],
        colors="r",
        linestyle="dotted",
    )
else:  # 'log'
    ax1.plot(motor_positions, np.log10(sum_int), marker=".")
    vline = ax1.vlines(
        x=motor_positions[index_peak],
        ymin=np.log10(sum_int.min()),
        ymax=np.log10(sum_int[index_peak]),
        colors="r",
        linestyle="dotted",
    )
ax1.set_xlabel(motor_name)
ax1.set_ylabel("integrated intensity")
if invert_xaxis:
    ax1.invert_xaxis()
ax1.set_aspect("auto", adjustable="datalim", anchor="S", share=False)
ax2.imshow(
    np.log10(data[index_peak, sum_roi[0] : sum_roi[1], sum_roi[2] : sum_roi[3]]),
    cmap=my_cmap,
    vmin=0,
)
ax2.axis("scaled")
ax2.set_title("ROI at line")
roi_text = figure.text(
    0.70,
    0.80,
    "unbinned ROI [y0 y1 x0 x1]\n"
    f"[{sum_roi[0] * binning[0]:d}, "
    f"{sum_roi[1] * binning[0]:d}, "
    f"{sum_roi[2] * binning[1]:d}, "
    f"{sum_roi[3] * binning[1]:d}]",
    size=10,
)
motor_text = figure.text(
    0.70, 0.75, f"{motor_name} = {motor_positions[index_peak]:.2f}", size=10
)
max_text = figure.text(
    0.70,
    0.70,
    "ROI max at line = "
    + f"{data[index_peak, sum_roi[0]:sum_roi[1], sum_roi[2]:sum_roi[3]].max():d}",
    size=10,
)
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
    sum_roi[2],
    sum_roi[3],
    sum_roi[0],
    sum_roi[1],
)  # extents (xmin, xmax, ymin, ymax)
figure.set_facecolor(background_plot)
plt.show()
