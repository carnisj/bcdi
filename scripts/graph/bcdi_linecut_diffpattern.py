#!/usr/bin/env python3

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import gc
import tkinter as tk
from tkinter import filedialog

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.measurements import center_of_mass

import bcdi.graph.linecut as lc
import bcdi.utils.utilities as util
import bcdi.utils.validation as valid
from bcdi.graph.colormap import ColormapFactory

helptext = """
Graphical user interface for plotting linecuts along particular direction of a 3D array.

For the laboratory frame, the CXI convention is used: z downstream, y vertical,
x outboard. For q, the usual convention is used: qx downstream, qz vertical, qy outboard
"""

scan = 54
sample_name = "p21"
datadir = f"D:/data/P10_1st_test_isosurface/data/{sample_name}_{scan:05d}/pynx/"
# data directory
savedir = datadir + "linecut/"  # if None, it will default to the data directory
load_qvalues = True  # True to load the q values (a NPZ file with the fields
# 'qx', 'qy', 'qz', each one containing
# a 1D or 3D array)
load_mask = False  # True to load a mask (same shape than the diffraction pattern)
#######################################
# parameters related to visualization #
#######################################
starting_point = None  # list of three indices (integers) for the starting point
# of the linecut. Leave None for default
endpoint = None  # list of three indices (integers) for the endpoint point
# of the linecut. Leave None for default
threshold = 0  # every voxel <= threshold will be set to 0
vmin = 0  # vmin for the plots, None for default
vmax = 5  # vmax for the plots, should be larger than vmin, None for default
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
    global diff_pattern, starting_point, endpoint, distance, fig_diff, ax0, ax1, ax2
    global q_text, load_qvalues, vmin, vmax, qx, qy, qz, ax3, plt0, plt1, plt2, cut
    global plt0_start, plt1_start, plt2_start, plt0_stop, plt1_stop, plt2_stop

    if event.inaxes == ax3:  # print the distance value at the mouse position
        q_text.remove()
        if load_qvalues:
            q_text = fig_diff.text(
                0.55, 0.43, f"distance = {event.xdata:.3f} (1/A)", size=10
            )
        else:
            q_text = fig_diff.text(0.55, 0.43, f"distance = {event.xdata:.1f}", size=10)
        plt.draw()
        update_cut = False
    else:
        if event.button == 1:  # left button
            if event.inaxes == ax0:  # hor=X, ver=Y
                starting_point[2], starting_point[1] = (
                    int(np.rint(event.xdata)),
                    int(np.rint(event.ydata)),
                )
                update_cut = True
            elif event.inaxes == ax1:  # hor=X, ver=rocking curve
                starting_point[2], starting_point[0] = (
                    int(np.rint(event.xdata)),
                    int(np.rint(event.ydata)),
                )
                update_cut = True
            elif event.inaxes == ax2:  # hor=Y, ver=rocking curve
                starting_point[1], starting_point[0] = (
                    int(np.rint(event.xdata)),
                    int(np.rint(event.ydata)),
                )
                update_cut = True
            else:
                update_cut = False
        elif event.button == 3:  # right button
            if event.inaxes == ax0:  # hor=X, ver=Y
                endpoint[2], endpoint[1] = (
                    int(np.rint(event.xdata)),
                    int(np.rint(event.ydata)),
                )
                update_cut = True
            elif event.inaxes == ax1:  # hor=X, ver=rocking curve
                endpoint[2], endpoint[0] = (
                    int(np.rint(event.xdata)),
                    int(np.rint(event.ydata)),
                )
                update_cut = True
            elif event.inaxes == ax2:  # hor=Y, ver=rocking curve
                endpoint[1], endpoint[0] = (
                    int(np.rint(event.xdata)),
                    int(np.rint(event.ydata)),
                )
                update_cut = True
            else:
                update_cut = False
        else:
            update_cut = False

    if update_cut:
        q_text.remove()
        q_text = fig_diff.text(0.55, 0.25, "", size=10)
        print(f"starting_point = {starting_point}, endpoint = {endpoint}")
        cut = lc.linecut(
            diff_pattern,
            indices=list(zip(starting_point, endpoint)),
            interp_order=1,
        )
        if qx.ndim == 1:
            d_q = np.sqrt(
                (qx[endpoint[0]] - qx[starting_point[0]]) ** 2
                + (qz[endpoint[1]] - qz[starting_point[1]]) ** 2
                + (qy[endpoint[2]] - qy[starting_point[2]]) ** 2
            )
        else:
            d_q = np.sqrt(
                (
                    qx[endpoint[0], endpoint[1], endpoint[2]]
                    - qx[starting_point[0], starting_point[1], starting_point[2]]
                )
                ** 2
                + (
                    qz[endpoint[0], endpoint[1], endpoint[2]]
                    - qz[starting_point[0], starting_point[1], starting_point[2]]
                )
                ** 2
                + (
                    qy[endpoint[0], endpoint[1], endpoint[2]]
                    - qy[starting_point[0], starting_point[1], starting_point[2]]
                )
                ** 2
            )
        distance = np.linspace(0, d_q, num=len(cut))
        plt0.remove()
        plt1.remove()
        plt2.remove()
        (plt0,) = ax0.plot(
            [starting_point[2], endpoint[2]], [starting_point[1], endpoint[1]], "r-"
        )  # sum axis 0
        (plt1,) = ax1.plot(
            [starting_point[2], endpoint[2]], [starting_point[0], endpoint[0]], "r-"
        )  # sum axis 1
        (plt2,) = ax2.plot(
            [starting_point[1], endpoint[1]], [starting_point[0], endpoint[0]], "r-"
        )  # sum axis 2

        plt0_start.remove()
        plt1_start.remove()
        plt2_start.remove()
        plt0_stop.remove()
        plt1_stop.remove()
        plt2_stop.remove()
        (plt0_start,) = ax0.plot(
            [starting_point[2]], [starting_point[1]], "bo"
        )  # sum axis 0
        (plt0_stop,) = ax0.plot([endpoint[2]], [endpoint[1]], "ro")  # sum axis 0
        (plt1_start,) = ax1.plot(
            [starting_point[2]], [starting_point[0]], "bo"
        )  # sum axis 1
        (plt1_stop,) = ax1.plot([endpoint[2]], [endpoint[0]], "ro")  # sum axis 1
        (plt2_start,) = ax2.plot(
            [starting_point[1]], [starting_point[0]], "bo"
        )  # sum axis 2
        (plt2_stop,) = ax2.plot([endpoint[1]], [endpoint[0]], "ro")  # sum axis 2

        ax3.cla()
        ax3.plot(distance, np.log10(cut), "-or", markersize=3)
        if load_qvalues:
            ax3.set_xlabel("distance along the linecut (1/A)")
        else:
            ax3.set_xlabel("distance along the linecut (pixels)")
        ax3.set_ylabel("Int (A.U.)")
        ax3.axis("auto")
        ax3.set_ylim(bottom=-2)
        plt.tight_layout()
        plt.draw()


def press_key(event):
    """
    Interact with the PRTF plot.

    :param event: button press event
    """
    global savedir, starting_point, endpoint, fig_diff, cut, distance, sample_name, scan
    try:
        close_fig = False
        if event.inaxes:
            if event.key == "s":
                template = (
                    savedir + f"{sample_name}_{scan}_"
                    f"linecut_start={starting_point}_stop={endpoint}"
                )
                fig_diff.savefig(template + ".png")
                np.savez_compressed(template + ".npz", linecut=cut, distance=distance)
            elif event.key == "q":
                close_fig = True
        if close_fig:
            plt.close("all")
    except AttributeError:  # mouse pointer out of axes
        pass


#########################
# check some parameters #
#########################
if (vmin and vmax) and (vmax <= vmin):
    raise ValueError("vmax should be larger than vmin")

savedir = savedir or datadir

valid.valid_container(
    obj=starting_point,
    container_types=list,
    allow_none=True,
    item_types=int,
    name="linecut_diffpattern.py",
)
valid.valid_container(
    obj=endpoint,
    container_types=list,
    allow_none=True,
    item_types=int,
    name="linecut_diffpattern.py",
)

###################
# define colormap #
###################
my_cmap = ColormapFactory().cmap

#################
# load the data #
#################
plt.ion()
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    initialdir=datadir,
    title="Select the diffraction pattern",
    filetypes=[("NPZ", "*.npz"), ("NPY", "*.npy")],
)
diff_pattern, _ = util.load_file(file_path)
diff_pattern = diff_pattern.astype(float)
diff_pattern[np.isnan(diff_pattern)] = 0  # discard nans
diff_pattern[diff_pattern <= threshold] = 0  # apply the intensity threshold
if diff_pattern.ndim != 3:
    raise ValueError("the diffraction pattern should be a 3D array")
nz, ny, nx = diff_pattern.shape

############################
# load the mask (optional) #
############################
if load_mask:
    file_path = filedialog.askopenfilename(
        initialdir=datadir,
        title="Select mask (optional)",
        filetypes=[("NPZ", "*.npz"), ("NPY", "*.npy")],
    )
    if not file_path:
        raise ValueError(
            'a mask should be provided if the parameter "load_mask" is set to True'
        )
    mask, _ = util.load_file(file_path)
    if mask.shape != diff_pattern.shape:
        raise ValueError(
            "the mask should have the same shape as the diffraction " "pattern"
        )
    diff_pattern[np.nonzero(mask)] = 0
    del mask
    gc.collect()

############################
# load q values (optional) #
############################
if load_qvalues:
    file_path = filedialog.askopenfilename(
        initialdir=datadir, title="Select q values", filetypes=[("NPZ", "*.npz")]
    )
    if not file_path:
        raise ValueError(
            'q values should be provided if the parameter "load_qvalues" is set to True'
        )
    q_values = np.load(file_path)
    qx = q_values["qx"]
    qz = q_values["qz"]
    qy = q_values["qy"]
else:
    qx = np.arange(nz)
    qz = np.arange(ny)
    qy = np.arange(nx)

#####################################################
# get the center of mass of the diffraction pattern #
#####################################################
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

##################################################
# calculate the array of distances from the peak #
##################################################
if qx.ndim == 1:
    qxCOM = qx[z0]
    qzCOM = qz[y0]
    qyCOM = qy[x0]
elif qx.ndim == 3:
    qxCOM = qx[z0, y0, x0]
    qyCOM = qy[z0, y0, x0]
    qzCOM = qz[z0, y0, x0]
else:
    raise ValueError("q components should be 1D or 3D arrays")

if load_qvalues:
    print(f"COM[qx, qz, qy] = {qxCOM:.3f} 1/A, {qzCOM:.3f} 1/A, {qyCOM:.3f} 1/A")
else:
    print(f"COM[qx, qz, qy] = {qxCOM:.3f}, {qzCOM:.3f}, {qyCOM:.3f}")

####################
# interactive plot #
####################
plt.ioff()
starting_point = starting_point or [nz // 2, 0, nx // 2]
endpoint = endpoint or [nz // 2, ny - 1, nx // 2]
cut = lc.linecut(
    diff_pattern,
    indices=list(zip(starting_point, endpoint)),
    interp_order=1,
)
if qx.ndim == 1:
    dq = np.sqrt(
        (qx[endpoint[0]] - qx[starting_point[0]]) ** 2
        + (qz[endpoint[1]] - qz[starting_point[1]]) ** 2
        + (qy[endpoint[2]] - qy[starting_point[2]]) ** 2
    )
else:
    dq = np.sqrt(
        (
            qx[endpoint[0], endpoint[1], endpoint[2]]
            - qx[starting_point[0], starting_point[1], starting_point[2]]
        )
        ** 2
        + (
            qz[endpoint[0], endpoint[1], endpoint[2]]
            - qz[starting_point[0], starting_point[1], starting_point[2]]
        )
        ** 2
        + (
            qy[endpoint[0], endpoint[1], endpoint[2]]
            - qy[starting_point[0], starting_point[1], starting_point[2]]
        )
        ** 2
    )
distance = np.linspace(0, dq, num=len(cut))

fig_diff, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))
fig_diff.canvas.mpl_disconnect(fig_diff.canvas.manager.key_press_handler_id)
ax0.imshow(np.log10(diff_pattern.sum(axis=0)), vmin=vmin, vmax=vmax, cmap=my_cmap)
ax1.imshow(np.log10(diff_pattern.sum(axis=1)), vmin=vmin, vmax=vmax, cmap=my_cmap)
ax2.imshow(np.log10(diff_pattern.sum(axis=2)), vmin=vmin, vmax=vmax, cmap=my_cmap)
ax3.plot(distance, np.log10(cut), "-or", markersize=3)
(plt0,) = ax0.plot(
    [starting_point[2], endpoint[2]], [starting_point[1], endpoint[1]], "r-"
)  # sum axis 0
(plt1,) = ax1.plot(
    [starting_point[2], endpoint[2]], [starting_point[0], endpoint[0]], "r-"
)  # sum axis 1
(plt2,) = ax2.plot(
    [starting_point[1], endpoint[1]], [starting_point[0], endpoint[0]], "r-"
)  # sum axis 2
(plt0_start,) = ax0.plot([starting_point[2]], [starting_point[1]], "bo")  # sum axis 0
(plt0_stop,) = ax0.plot([endpoint[2]], [endpoint[1]], "ro")  # sum axis 0
(plt1_start,) = ax1.plot([starting_point[2]], [starting_point[0]], "bo")  # sum axis 1
(plt1_stop,) = ax1.plot([endpoint[2]], [endpoint[0]], "ro")  # sum axis 1
(plt2_start,) = ax2.plot([starting_point[1]], [starting_point[0]], "bo")  # sum axis 2
(plt2_stop,) = ax2.plot([endpoint[1]], [endpoint[0]], "ro")  # sum axis 2
ax0.axis("scaled")
ax1.axis("scaled")
ax2.axis("scaled")
ax3.axis("auto")
ax3.set_ylim(bottom=-2)
if load_qvalues:
    ax0.set_title("horizontal=qy  vertical=qz")
    ax1.set_title("horizontal=qy  vertical=qx")
    ax2.set_title("horizontal=qz  vertical=qx")
    ax3.set_xlabel("distance along the linecut (1/A)")
else:
    ax0.set_title("horizontal=X  vertical=Y")
    ax1.set_title("horizontal=X  vertical=rocking curve")
    ax2.set_title("horizontal=Y  vertical=rocking curve")
    ax3.set_xlabel("distance along the linecut (pixels)")

ax3.set_ylabel("Int (A.U.)")
q_text = fig_diff.text(0.55, 0.43, "", size=10)
fig_diff.text(0.01, 0.9, "left click to select\nthe starting point", size=10)
fig_diff.text(0.01, 0.8, "right click to select\nthe endpoint", size=10)
fig_diff.text(0.01, 0.7, "q to quit\ns to save", size=10)
fig_diff.text(0.85, 0.40, "click to read\nthe distance", size=10)
plt.tight_layout()
plt.connect("key_press_event", press_key)
plt.connect("button_press_event", on_click)
fig_diff.set_facecolor(background_plot)
plt.show()
