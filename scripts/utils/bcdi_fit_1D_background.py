#!/usr/bin/env python3

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

import bcdi.graph.graph_utils as gu
import bcdi.utils.utilities as util

plt.switch_backend(
    "Qt5Agg"
)  # "Qt5Agg" or "Qt4Agg" depending on the version of Qt installer, bug with Tk

helptext = """
Determination of the background in a reciprocal space linecut using an interactive
interface. The background-subtracted data is saved in a different .npz file with the
original field names.
"""

datadir = "D:/data/P10_March2020_CDI/test_april/data/align_06_00248/pynx_not_masked/"
method = "manual"  # method for background determination: only 'manual' for now
xlim = None  # limits used for the horizontal axis of plots, leave None otherwise
ylim = None  # limits used for the vertical axis of plots, leave None otherwise
include_origin = True  # if True, will include the first data point in the background
scale = "log"  # scale for plots
field_names = ["distances", "average"]  # names of the fields in the file
##################################
# end of user-defined parameters #
##################################


def on_click(event):
    """
    Function to interact with a plot, return the position of clicked pixel. If
    flag_pause==1 or if the mouse is out of plot axes, it will not register the click

    :param event: mouse click event
    :return: updated list of vertices which defines a polygon to be masked
    """
    global xy, flag_pause
    if not event.inaxes:
        return
    if not flag_pause:
        xy.append([event.xdata, event.ydata])
    return


def press_key(event):
    """
    Interact with a plot for masking parasitic diffraction intensity or detector gaps

    :param event: button press event
    :return: updated data, mask and controls
    """
    global distances, data, flag_pause, xy, fig_back, xlim, ylim

    try:
        flag_pause, xy, stop_masking = gu.update_background(
            key=event.key,
            distances=distances,
            data=data,
            figure=fig_back,
            flag_pause=flag_pause,
            xy=xy,
            xlim=xlim,
            ylim=ylim,
        )
        if stop_masking:
            plt.close(fig_back)

    except AttributeError:  # mouse pointer out of axes
        pass


##############################
# load reciprocal space data #
##############################
plt.ion()
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    initialdir=datadir,
    title="Select the reciprocal space data",
    filetypes=[("NPZ", "*.npz")],
)
npzfile = np.load(file_path)
distances = npzfile[field_names[0]]
average = npzfile[field_names[1]]

#############
# plot data #
#############
# prepare for masking arrays - 'conventional' arrays won't do it
y_values = np.ma.array(average)
# mask values below a certain threshold
y_values_masked = np.ma.masked_where(np.isnan(y_values), y_values)

######################
# fit the background #
######################
plt.ioff()
xy = []  # list of points defining the background curve
if include_origin and scale == "linear":
    xy.append([0, y_values_masked[0]])
elif include_origin and scale == "log":
    xy.append([0, np.log10(y_values_masked[0])])

flag_pause = False  # press x to pause for pan/zoom
data = np.copy(y_values_masked)
fig_back, _ = plt.subplots(1, 1)
fig_back.canvas.mpl_disconnect(fig_back.canvas.manager.key_press_handler_id)
if scale == "linear":
    plt.plot(distances, data, ".-r")
else:
    plt.plot(distances, np.log10(data), ".-r")
plt.xlabel("q (1/nm)")
plt.ylabel("Angular average (A.U.)")
plt.title(
    "Click to select background points\nx to pause/resume for pan/zoom\n"
    "a restart ; p plot background ; q quit"
)
if xlim is not None:
    plt.xlim(xlim[0], xlim[1])
if ylim is not None:
    plt.ylim(ylim[0], ylim[1])
plt.connect("key_press_event", press_key)
plt.connect("button_press_event", on_click)
plt.show()

#########################################################
# fit background and interpolate it to mach data points #
#########################################################
xy_array = np.asarray(xy)
indices = util.find_nearest(distances, xy_array[:, 0])
if scale == "linear":
    interpolation = interp1d(
        distances[indices],
        data[indices],
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    background = interpolation(distances)
    background[np.isnan(background)] = 0
    data_back = data - background
    data_back[data_back <= 0] = 0
else:  # fit direcly log values, less artefactsdistances.max
    interpolation = interp1d(
        distances[indices],
        np.log10(data[indices]),
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    background = interpolation(distances)
    background = 10**background
    background[np.isnan(background)] = 0
    data_back = data - background
    data_back[data_back <= 1] = 1  # will appear as 0 in log plot

###################################
# save background subtracted data #
###################################
np.savez_compressed(
    datadir + "q+angular_avg_back.npz",
    distances=distances,
    average=data_back,
    background=background,
)

###################################
# plot background subtracted data #
###################################
xmin = distances[np.unravel_index(distances[~np.isnan(data)].argmin(), distances.shape)]
xmax = distances[np.unravel_index(distances[~np.isnan(data)].argmax(), distances.shape)]
ymin = 0

fig, (ax0, ax1) = plt.subplots(2, 1)
if scale == "linear":
    ymax = data[~np.isnan(data)].max()
    ax0.plot(distances, data, "r", distances, background, "b")
    ax1.plot(distances, data_back, "r")
else:
    ymax = np.log10(data[~np.isnan(data)].max())
    ax0.plot(distances, np.log10(data), "r", distances, np.log10(background), "b")
    ax1.plot(distances, np.log10(data_back))

ax0.legend(["data", "background"])
ax0.set_xlim([xmin, xmax])
ax0.set_ylim([ymin, ymax])
ax1.legend(["data-background"])
ax1.set_xlim([xmin, xmax])
ax1.set_ylim([ymin, ymax])
fig.savefig(datadir + "q+angular_avg_back.png")

plt.ioff()
plt.show()
