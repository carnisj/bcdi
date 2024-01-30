#!/usr/bin/env python3

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import os

import fabio
import matplotlib.pyplot as plt
import numpy as np
import xrayutilities as xu

helptext = """
Calculate the cross-correlation of 2D detector images in live.
"""

start_image = 41233  # starting image number
detector = 1  # 0 for eiger, 1 for maxipix
exposure_time = 0.5
datadir = "/data/visitor/ch5309/id01/images/align/"
savedir = "/data/visitor/ch5309/id01/analysis/"
ccdfiletmp = os.path.join(datadir, "data_mpx4_%05d.edf.gz")
# template for the CCD file names
nav = [1, 1]  # reduce data: number of pixels to average in each detector direction
if detector == 0:
    roi = [1320, 1470, 810, 920]
else:
    roi = [0, 516, 0, 516]
stop_flag = 0  # flag to exit the while loop
stable_sample = 1  # 0 if a lot of change expected, 1 if very stable
##############################################################################


def load_file(start_img, current_img, mydetector, region=None):
    """Load images at ID01 and stack them."""
    # global index   # for testing, to simulate ongoing measurement
    if region is None:
        if mydetector == 0:
            region = [0, 2164, 0, 1030]
        elif mydetector == 1:
            region = [0, 516, 0, 516]
    frame_nb = current_img - start_img
    rawdata = np.zeros((frame_nb, region[1] - region[0], region[3] - region[2]))
    for idx in range(frame_nb):
        i = start_img + idx
        e = fabio.open(ccdfiletmp % i)
        ccdraw = e.data
        ccd = xu.blockAverage2D(ccdraw, nav[0], nav[1], roi=region)
        rawdata[idx, :, :] = ccd
    return rawdata


def calc_corr(array, previous_array, is_stable=0):
    """Calculate the cross correlation among a stack of images."""
    if previous_array is None:
        previous_array = np.array([])
    starting_index = previous_array.shape[0]
    nb_frames = array.shape[0]
    corr = np.zeros((nb_frames, nb_frames))
    corr[0:starting_index, 0:starting_index] = previous_array
    for idx in range(starting_index, nb_frames, 1):
        for idy in range(nb_frames):
            if is_stable == 0:
                corr[idx, idy] = (np.multiply(array[idx], array[idy]).sum()) / (
                    np.sqrt(np.square(array[idx]).sum() * np.square(array[idy]).sum())
                )
            else:
                corr[idx, idy] = 1 - (np.multiply(array[idx], array[idy]).sum()) / (
                    np.sqrt(np.square(array[idx]).sum() * np.square(array[idy]).sum())
                )
        corr[:, idx] = corr[idx, :]
        # since plot is symmetric, do not need to compute this again
    return corr


def press_key(event):
    """Process press_key events to exit a GUI."""
    global stop_flag
    try:
        key = event.key
    except AttributeError:  # mouse pointer out of axes
        return
    if key == "q":
        stop_flag = 1


##############################################################################
plt.ion()
fig, ax = plt.subplots(1, 1)
plt.connect("key_press_event", press_key)
index = 1
previous = np.array([])
stop_counter = 0
stop_image_old = start_image
while stop_flag != 1:
    # get the range of the dataset
    no_error = 1
    index = 0
    while no_error == 1:
        img_nb = start_image + index
        fname = ccdfiletmp % img_nb
        if os.path.isfile(fname):
            index = index + 1
        else:
            no_error = 0
    stop_image_new = start_image + index
    if stop_image_new == stop_image_old:
        stop_counter = stop_counter + 1
        plt.pause(1)
        continue
    data = load_file(start_image, stop_image_new, detector, roi)
    stop_counter = 0
    cross_corr = calc_corr(data, previous, is_stable=stable_sample)
    if stable_sample == 0:
        plt.title(
            "Running, iteration: "
            + str(index)
            + "\n Press q to stop (mouse on the plot)"
        )
        index = index + 1
        plt.imshow(cross_corr, vmin=0, vmax=1)
    else:
        plt.title(
            "Running, iteration: "
            + str(index)
            + "\n Press q to stop (mouse on the plot)"
        )
        index = index + 1
        plt.imshow(np.log10(abs(cross_corr)), vmin=-4, vmax=0)
    stop_image_old = stop_image_new
    plt.gca().invert_yaxis()
    plt.draw()
    previous = cross_corr
    del data, cross_corr
    plt.pause(exposure_time)

plt.ioff()
plt.title("Stopped at iteration: " + str(index))
plt.colorbar()
plt.savefig(savedir + "crosscorr.png")
plt.show()
