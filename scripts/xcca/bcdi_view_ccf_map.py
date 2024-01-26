#!/usr/bin/env python3

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import os
import sys
import tkinter as tk
from tkinter import filedialog

import numpy as np
from matplotlib import pyplot as plt

import bcdi.utils.utilities as util
from bcdi.graph.colormap import ColormapFactory

helptext = """
Load the output file of xcca_3D_map_polar.py or xcca_3D_map_rect.py and plot the 2D
cross-correlation map. When clicking on the 2D map, the 1D cross-correlation at the
clicked q value is plotted.

Input: a NPZ file with the fields 'angles', 'q_range', 'ccf', 'points':
    - 'angles': angle values between [0, 180] where the cross-correlation function
      was calculated
    - 'q_range': q values where the cross-correlation CCF(q,q) was calculated
    - 'ccf': cross-correlation function values at these angles and q_values (2D array)
    - 'points': number of points contributing to the cross-correlation function at
      these angles and q values (2D array)
"""

datadir = "D:/data/P10_August2019_CDI/data/gold_2_2_2_00022/pynx/1_4_4_fullrange_xcca/"
savedir = "D:/data/P10_August2019_CDI/data/gold_2_2_2_00022/pynx/1_4_4_fullrange_xcca/"
scale = "log"  # 'linear' or 'log', scale for the 2D map
comment = ""  # should start with _
###########################
# plot related parameters #
###########################
background_plot = "0.7"  # in level of grey in [0,1], 0 being dark. For visual comfort
##########################
# end of user parameters #
##########################


def onclick(click_event):
    """
    Process mouse click events in the 2D cross-correlation map

    :param click_event: mouse click event
    """
    global angles, q_range, ccf, current_q, ax0, ax1, my_cmap, ymin, ymax

    if click_event.inaxes == ax0:  # click in the 2D cross-correlation map
        current_q = util.find_nearest(
            reference_array=q_range, test_values=click_event.ydata
        )
        ymin = ccf[current_q, indices].min()
        ymax = 1.2 * ccf[current_q, indices].max()
        ax1.cla()
        ax1.plot(
            angles,
            ccf[current_q, :],
            linestyle="None",
            marker=".",
            markerfacecolor="blue",
        )
        ax1.set_xlim(0, 180)
        ax1.set_ylim(ymin, ymax)
        ax1.set_xlabel("Angle (deg)")
        ax1.set_ylabel("Cross-correlation (A.U.)")
        ax1.set_xticks(np.arange(0, 181, 30))
        ax1.set_title(f"Cross-correlation at q={q_range[current_q]:.3f}")
        plt.draw()


def press_key(event):
    """
    Process key press events in the interactive plots

    :param event: button press event
    """
    global angles, q_range, ccf, current_q, ax0, ax1, my_cmap, ymin, ymax, min_colorbar
    global max_colorbar, scale

    if event.inaxes == ax0:
        if event.key == "right":
            if scale == "linear":
                max_colorbar = max_colorbar * 1.5
            else:  # 'log'
                max_colorbar = max_colorbar + 0.5
        elif event.key == "left":
            if scale == "linear":
                max_colorbar = max_colorbar / 1.5
                if max_colorbar <= min_colorbar:
                    max_colorbar = max_colorbar * 1.5
            else:  # 'log'
                max_colorbar = max_colorbar - 0.5
                if max_colorbar <= min_colorbar:
                    max_colorbar = max_colorbar + 0.5
        ax0.cla()
        if scale == "linear":
            ax0.imshow(
                ccf,
                cmap=my_cmap,
                vmin=min_colorbar,
                vmax=max_colorbar,
                extent=[0, 180, q_range[-1] + dq / 2, q_range[0] - dq / 2],
            )  # extent (left, right, bottom, top)
        else:  # 'log'
            ax0.imshow(
                np.log10(ccf),
                cmap=my_cmap,
                vmin=min_colorbar,
                vmax=max_colorbar,
                extent=[0, 180, q_range[-1] + dq / 2, q_range[0] - dq / 2],
            )  # extent (left, right, bottom, top)
        ax0.set_xlabel("Angle (deg)")
        ax0.set_ylabel("q (nm$^{-1}$)")
        ax0.set_xticks(np.arange(0, 181, 30))
        ax0.set_yticks(q_range)
        ax0.set_aspect("auto")
        ax0.set_title(f"CCF from q={q_range[0]:.3f} to q={q_range[-1]:.3f}")
        plt.draw()

    if event.inaxes == ax1:
        if event.key == "right":
            ymax = ymax * 1.5
        elif event.key == "left":
            ymax = ymax / 1.5
            if ymax <= ymin:
                ymax = ymax * 1.5
        ax1.cla()
        ax1.plot(
            angles,
            ccf[current_q, :],
            linestyle="None",
            marker=".",
            markerfacecolor="blue",
        )
        ax1.set_xlim(0, 180)
        ax1.set_ylim(ymin, ymax)
        ax1.set_xlabel("Angle (deg)")
        ax1.set_ylabel("Cross-correlation (A.U.)")
        ax1.set_xticks(np.arange(0, 181, 30))
        ax1.set_title(f"Cross-correlation at q={q_range[current_q]:.3f}")
        plt.draw()


###################
# define colormap #
###################
bad_color = "1.0"  # white background
my_cmap = ColormapFactory(bad_color=bad_color).cmap
plt.ion()

###################################
# load the cross-correlation data #
###################################
plt.ion()
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    initialdir=datadir, title="Select the CCF file", filetypes=[("NPZ", "*.npz")]
)
filename = os.path.splitext(os.path.basename(file_path))[
    0
]  # the extension .npz is removed
npzfile = np.load(file_path)
try:
    angles = npzfile["angles"]
    q_range = npzfile["q_range"]
    ccf = npzfile["ccf"]
    points = npzfile["points"]
except KeyError:
    print("Keys in the NPZ file:", list(npzfile.keys()))
    sys.exit()

#############################################################
# offset the cross-correlation if there are negative values #
#############################################################
for idx in range(ccf.shape[0]):  # loop over the q values
    ccf[idx, :] = ccf[idx, :] - min(0, ccf[idx, :].min())

#######################################
# plot the cross-correlation function #
#######################################
indices = np.argwhere(np.logical_and((angles >= 20), (angles <= 160)))[:, 0]
current_q = 0  # index of the q for the lineplot
ymin = ccf[current_q, indices].min()  # used for the lineplot
ymax = 1.2 * ccf[current_q, indices].max()  # used for the lineplot
if scale == "linear":
    min_colorbar = ccf[:, indices].min()  # used for the 2D map
    max_colorbar = 1.2 * ccf[:, indices].max()  # used for the 2D map
else:  # 'log'
    min_colorbar = np.log10(ccf[:, indices].min())  # used for the 2D map
    max_colorbar = np.log10(ccf[:, indices].max()) + 0.5  # used for the 2D map
dq = q_range[1] - q_range[0]
plt.ioff()

figure = plt.figure()
ax0 = figure.add_subplot(121)
ax1 = figure.add_subplot(122)
figure.canvas.mpl_disconnect(figure.canvas.manager.key_press_handler_id)
if scale == "linear":
    ax0.imshow(
        ccf,
        cmap=my_cmap,
        vmin=min_colorbar,
        vmax=max_colorbar,
        extent=[0, 180, q_range[-1] + dq / 2, q_range[0] - dq / 2],
    )  # extent (left, right, bottom, top)
else:  # 'log'
    ax0.imshow(
        np.log10(ccf),
        cmap=my_cmap,
        vmin=min_colorbar,
        vmax=max_colorbar,
        extent=[0, 180, q_range[-1] + dq / 2, q_range[0] - dq / 2],
    )  # extent (left, right, bottom, top)
ax0.set_xlabel("Angle (deg)")
ax0.set_ylabel("q (nm$^{-1}$)")
ax0.set_xticks(np.arange(0, 181, 30))
ax0.set_yticks(q_range)
ax0.set_aspect("auto")
ax0.set_title(f"CCF from q={q_range[0]:.3f} to q={q_range[-1]:.3f}")

ax1.plot(
    angles, ccf[current_q, :], linestyle="None", marker=".", markerfacecolor="blue"
)
ax1.set_xlim(0, 180)
ax1.set_ylim(ccf[current_q, :].min(), ymax)
ax1.set_xlabel("Angle (deg)")
ax1.set_ylabel("Cross-correlation (A.U.)")
ax1.set_xticks(np.arange(0, 181, 30))
ax1.set_title(f"Cross-correlation at q={q_range[current_q]:.3f}")

plt.tight_layout()
figure.set_facecolor(background_plot)
plt.connect("key_press_event", press_key)
plt.connect("button_press_event", onclick)
plt.show()
