#!/usr/bin/env python3

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import datetime
import multiprocessing as mp
import os
import sys
import time

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

import bcdi.graph.graph_utils as gu
import bcdi.utils.utilities as util
from bcdi.experiment.detector import create_detector
from bcdi.graph.colormap import ColormapFactory

matplotlib.use("Qt5Agg")

helptext = """
Open images or series data at P10 beamline.
"""

scan_nb = 76  # scan number as it appears in the folder name
sample_name = "B15_syn_S1_2"  # without _ at the end
root_directory = "C:/Users/Jerome/Documents/data/dataset_P10/"
# parent directory of the scan
file_list = np.arange(1, 150 + 1)
# list of file numbers, e.g. [1] for gold_2_2_2_00022_data_000001.h5
detector_name = "Eiger4M"  # "Eiger2M" or "Maxipix" or "Eiger4M"
counter_roi = []  # plot the integrated intensity in this region of interest.
# Leave it to [] to use the full detector [Vstart, Vstop, Hstart, Hstop]
# if data is a series, the condition becomes
# log10(data.sum(axis=0)) > high_threshold * nb_frames
save_directory = root_directory + "test/"
# images will be saved here, leave it to None otherwise (default to the scan directory)
is_scan = True  # set to True is the measurement is a scan or a time series,
# False for a single image
compare_ends = False  # set to True to plot the difference between the last frame
# and the first frame
save_mask = False  # True to save the mask as 'hotpixels.npz'
save_to_mat = False  # True to save the 2D summed data to a .mat file
multiprocessing = True  # True to use multiprocessing
#######################################
# parameters related to visualization #
#######################################
grey_background = True  # if True, nans will be set to grey in imshow plots
photon_threshold = 0  # everything below this threshold will be set to 0
vmin = 0  # vmin for the plots, None for default
vmax = 5  # vmax for the plots, should be larger than vmin, None for default
##########################
# end of user parameters #
##########################

##############################################
# create the dictionnary of input parameters #
##############################################
params = {
    "scan": scan_nb,
    "sample_name": sample_name,
    "rootdir": root_directory,
    "file_list": file_list,
    "detector": detector_name,
    "counter_roi": counter_roi,
    "savedir": save_directory,
    "is_scan": is_scan,
    "compare_ends": compare_ends,
    "save_mask": save_mask,
    "threshold": photon_threshold,
    "cb_min": vmin,
    "cb_max": vmax,
    "multiprocessing": multiprocessing,
    "grey_bckg": grey_background,
}

#########################
# check some parameters #
#########################
if (vmin and vmax) and vmax <= vmin:
    raise ValueError("vmax should be larger than vmin")


def load_p10_file(my_detector, my_file, file_index, roi, threshold):
    """
    Load a P10 data file, mask the fetector gaps and eventually concatenate the series.

    :param my_detector: instance of the class experiment_utils.Detector()
    :param my_file: file name of the data to load
    :param file_index: index of the data file in the total file list, used to sort
     frames afterwards
    :param roi: region of interest used to calculate the counter
     (integrated intensity in the ROI)
    :param threshold: threshold applied to each frame,
     intensities <= threshold are set to 0
    :return: the 2D data, 2D mask, counter and file index
    """
    roi_sum = []
    file = h5py.File(my_file, "r")
    dataset = file["entry"]["data"]["data"][:]
    mask_2d = np.zeros((dataset.shape[1], dataset.shape[2]))
    dataset[dataset <= threshold] = 0
    for frame in range(dataset.shape[0]):
        roi_sum.append(dataset[frame, roi[0] : roi[1], roi[2] : roi[3]].sum())
    nb_frames = dataset.shape[0]  # collect the number of frames in the eventual series
    dataset, mask_2d = my_detector.mask_detector(
        data=dataset.sum(axis=0), mask=mask_2d, nb_frames=nb_frames
    )
    return dataset, mask_2d, [roi_sum, file_index]


def main(parameters):
    """
    Protection for multiprocessing.

    :param parameters: dictionnary containing input parameters
    """

    def collect_result(result):
        """
        Callback processing the result after asynchronous multiprocessing.

        Update the global arrays.

        :param result: the output of load_p10_file, containing the 2d data, 2d mask,
         counter for each frame, and the file index
        """
        nonlocal sumdata, mask, counter, nb_files, current_point
        # result is a tuple: data, mask, counter, file_index
        current_point += 1
        sumdata = sumdata + result[0]
        mask[np.nonzero(result[1])] = 1
        counter.append(result[2])

        sys.stdout.write(f"\rFile {current_point:d} / {nb_files:d}")
        sys.stdout.flush()

    ######################################
    # load the dictionnary of parameters #
    ######################################
    scan = parameters["scan"]
    samplename = parameters["sample_name"]
    rootdir = parameters["rootdir"]
    image_nb = parameters["file_list"]
    counterroi = parameters["counter_roi"]
    savedir = parameters["savedir"]
    load_scan = parameters["is_scan"]
    compare_end = parameters["compare_ends"]
    savemask = parameters["save_mask"]
    multiproc = parameters["multiprocessing"]
    threshold = parameters["threshold"]
    cb_min = parameters["cb_min"]
    cb_max = parameters["cb_max"]
    grey_bckg = parameters["grey_bckg"]
    ###################
    # define colormap #
    ###################
    if grey_bckg:
        bad_color = "0.7"
    else:
        bad_color = "1.0"  # white background
    my_cmap = ColormapFactory(bad_color=bad_color).generate_cmap().cmap
    #######################
    # Initialize detector #
    #######################
    detector = create_detector(name=parameters["detector"])
    nb_pixel_y, nb_pixel_x = detector.nb_pixel_y, detector.nb_pixel_x
    sumdata = np.zeros((nb_pixel_y, nb_pixel_x))
    mask = np.zeros((nb_pixel_y, nb_pixel_x))
    counter = []

    ####################
    # Initialize paths #
    ####################
    if isinstance(image_nb, int):
        image_nb = [image_nb]
    if len(counterroi) == 0:
        counterroi = [0, nb_pixel_y, 0, nb_pixel_x]

    if not (
        counterroi[0] >= 0
        and counterroi[1] <= nb_pixel_y
        and counterroi[2] >= 0
        and counterroi[3] <= nb_pixel_x
    ):
        raise ValueError("counter_roi setting does not match the detector size")

    nb_files = len(image_nb)
    if nb_files == 1:
        multiproc = False

    if load_scan:  # scan or time series
        detector.datadir = rootdir + samplename + "_" + str(f"{scan:05d}") + "/e4m/"
        template_file = (
            detector.datadir + samplename + "_" + str(f"{scan:05d}") + "_data_"
        )
    else:  # single image
        detector.datadir = rootdir + samplename + "/e4m/"
        template_file = (
            detector.datadir + samplename + "_take_" + str(f"{scan:05d}") + "_data_"
        )
        compare_end = False

    detector.savedir = (
        savedir or os.path.abspath(os.path.join(detector.datadir, os.pardir)) + "/"
    )
    print(f"datadir: {detector.datadir}")
    print(f"savedir: {detector.savedir}")

    #############
    # Load data #
    #############
    plt.ion()
    filenames = [template_file + f"{image_nb[idx]:06d}.h5" for idx in range(nb_files)]
    roi_counter = None
    current_point = 0
    start = time.time()

    if multiproc:
        print("\nNumber of processors used: ", min(mp.cpu_count(), len(filenames)))
        mp.freeze_support()
        pool = mp.Pool(
            processes=min(mp.cpu_count(), len(filenames))
        )  # use this number of processes

        for file in range(nb_files):
            pool.apply_async(
                load_p10_file,
                args=(detector, filenames[file], file, counterroi, threshold),
                callback=collect_result,
                error_callback=util.catch_error,
            )

        pool.close()
        pool.join()  # postpones the execution of next line of code
        # until all processes in the queue are done.

        # sort out counter values
        # (we are using asynchronous multiprocessing, order is not preserved)
        roi_counter = sorted(counter, key=lambda x: x[1])

    else:
        for idx in range(nb_files):
            sys.stdout.write(f"\rLoading file {idx + 1:d}" + f" / {nb_files:d}")
            sys.stdout.flush()
            h5file = h5py.File(filenames[idx], "r")
            data = h5file["entry"]["data"]["data"][:]
            data[data <= threshold] = 0
            nbz, _, _ = data.shape
            for index in range(nbz):
                counter.append(
                    data[
                        index,
                        counterroi[0] : counterroi[1],
                        counterroi[2] : counterroi[3],
                    ].sum()
                )

            if compare_end and nb_files == 1:
                data_start, _ = detector.mask_detector(data=data[0, :, :], mask=mask)
                data_start = data_start.astype(float)
                data_stop, _ = detector.mask_detector(data=data[-1, :, :], mask=mask)
                data_stop = data_stop.astype(float)

                fig, _, _ = gu.imshow_plot(
                    data_stop - data_start,
                    plot_colorbar=True,
                    scale="log",
                    title="""difference between the last frame and
                    the first frame of the series""",
                )
            nb_frames = data.shape[
                0
            ]  # collect the number of frames in the eventual series
            data, mask = detector.mask_detector(
                data=data.sum(axis=0), mask=mask, nb_frames=nb_frames
            )
            sumdata = sumdata + data
            roi_counter = [[counter, idx]]

    end = time.time()
    print(
        "\nTime ellapsed for loading data:",
        str(datetime.timedelta(seconds=int(end - start))),
    )

    frame_per_series = int(len(counter) / nb_files)

    print("")
    if load_scan:
        if nb_files > 1:
            plot_title = (
                "masked data - sum of "
                + str(nb_files)
                + f" points with {frame_per_series:d} frames each"
            )
        else:
            plot_title = "masked data - sum of " + str(frame_per_series) + " frames"
        filename = "S" + str(scan) + "_scan.png"
    else:  # single image
        plot_title = "masked data"
        filename = "S" + str(scan) + "_image_" + str(image_nb[0]) + ".png"

    if savemask:
        fig, _, _ = gu.imshow_plot(mask, plot_colorbar=False, title="mask")
        np.savez_compressed(detector.savedir + "hotpixels.npz", mask=mask)
        fig.savefig(detector.savedir + "mask.png")

    y0, x0 = np.unravel_index(abs(sumdata).argmax(), sumdata.shape)
    print("Max at (y, x): ", y0, x0, " Max = ", int(sumdata[y0, x0]))

    np.savez_compressed(
        detector.savedir + f"{sample_name}_{scan_nb:05d}_sumdata.npz", data=sumdata
    )
    if save_to_mat:
        savemat(
            detector.savedir + f"{sample_name}_{scan_nb:05d}_sumdata.mat",
            {"data": sumdata},
        )

    if len(roi_counter[0][0]) > 1:
        # roi_counter[0][0] is the list of counter intensities in a series
        int_roi = [
            val[0][idx] for val in roi_counter for idx in range(frame_per_series)
        ]
        plt.figure()
        plt.plot(np.asarray(int_roi))
        plt.title("Integrated intensity in counter_roi")
        plt.pause(0.1)

    cb_min = cb_min or sumdata.min()
    cb_max = cb_max or sumdata.max()

    fig, _, _ = gu.imshow_plot(
        sumdata,
        plot_colorbar=True,
        title=plot_title,
        vmin=cb_min,
        vmax=cb_max,
        scale="log",
        cmap=my_cmap,
    )
    np.savez_compressed(detector.savedir + "hotpixels.npz", mask=mask)
    fig.savefig(detector.savedir + filename)
    plt.show()


if __name__ == "__main__":
    main(parameters=params)
