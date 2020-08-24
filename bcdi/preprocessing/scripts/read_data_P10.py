# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import hdf5plugin  # for lz4 filter
import h5py
import time
import datetime
import numpy as np
import multiprocessing as mp
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('D:/myscripts/bcdi/')
import bcdi.graph.graph_utils as gu
import bcdi.utils.utilities as util
import bcdi.experiment.experiment_utils as exp
import bcdi.preprocessing.preprocessing_utils as pru

helptext = """
Open images or series data at P10 beamline.
"""

scan_nb = 22  # scan number as it appears in the folder name
sample_name = "gold_2_2_2"  # without _ at the end
root_directory = "D:/data/P10_August2019_CDI/data/"
file_list = 1  # np.arange(1, 381+1)
# list of file numbers, e.g. [1] for gold_2_2_2_00022_data_000001.h5
detector_name = "Eiger4M"    # "Eiger2M" or "Maxipix" or "Eiger4M"
counter_roi = []  # plot the integrated intensity in this region of interest. Leave it to [] to use the full detector
# [Vstart, Vstop, Hstart, Hstop]
high_threshold = 4000000000  # data points where log10(data) > high_threshold will be masked
# if data is a series, the condition becomes log10(data.sum(axis=0)) > high_threshold * nb_frames
save_directory = ''  # images will be saved here, leave it to '' otherwise (default to data directory's parent)
is_scan = True  # set to True is the measurement is a scan or a time series, False for a single image
compare_ends = False  # set to True to plot the difference between the last frame and the first frame
save_mask = False  # True to save the mask as 'hotpixels.npz'
multiprocessing = True  # True to use multiprocessing
##########################
# end of user parameters #
##########################

##############################################
# create the dictionnary of input parameters #
##############################################
params = {'scan': scan_nb, 'sample_name': sample_name, 'rootdir': root_directory, 'file_list': file_list,
          'detector': detector_name, 'counter_roi': counter_roi, 'high_threshold': high_threshold,
          'savedir': save_directory, 'is_scan': is_scan, 'compare_ends': compare_ends, 'save_mask': save_mask,
          'multiprocessing': multiprocessing}


def load_p10_file(filname, fil_idx, roi, threshold):
    roi_sum = []
    file = h5py.File(filname, 'r')
    dataset = file['entry']['data']['data'][:]
    mask_2d = np.zeros((dataset.shape[1], dataset.shape[2]))
    [roi_sum.append(dataset[frame, roi[0]:roi[1], roi[2]:roi[3]].sum())
     for frame in range(dataset.shape[0])]
    nb_img = dataset.shape[0]
    dataset = dataset.sum(axis=0)  # data becomes 2D
    mask_2d[np.log10(dataset) > nb_img*threshold] = 1  # here we threshold the sum of nb_img images
    dataset[mask_2d == 1] = 0
    return dataset, mask_2d, [roi_sum, fil_idx]


def main(parameters):
    """
    Protection for multiprocessing.

    :param parameters: dictionnary containing input parameters
    """
    global sumdata, mask, counter

    def collect_result(result):
        """
        Callback processing the result after asynchronous multiprocessing. Update the global arrays.

        :param result: the output of load_p10_file, containing the 2d data, 2d mask, counter for each frame, and the
         file index
        """
        global sumdata, mask, counter
        # result is a tuple: data, mask, counter, file_index
        sumdata = sumdata + result[0]
        mask[np.nonzero(result[1])] = 1
        counter.append(result[2])

        sys.stdout.write('\rFile {:d}'.format(result[2][1]))
        sys.stdout.flush()

    ######################################
    # load the dictionnary of parameters #
    ######################################
    scan = parameters['scan']
    samplename = parameters['sample_name']
    rootdir = parameters['rootdir']
    image_nb = parameters['file_list']
    counterroi = parameters['counter_roi']
    threshold = parameters['high_threshold']
    savedir = parameters['savedir']
    load_scan = parameters['is_scan']
    compare_end = parameters['compare_ends']
    savemask = parameters['save_mask']
    multiproc = parameters['multiprocessing']

    #######################
    # Initialize detector #
    #######################
    detector = exp.Detector(name=parameters['detector'])
    nb_pixel_y, nb_pixel_x = detector.nb_pixel_y, detector.nb_pixel_x
    sumdata = np.zeros((nb_pixel_y, nb_pixel_x))
    mask = np.zeros((nb_pixel_y, nb_pixel_x))
    counter = []

    ####################
    # Initialize paths #
    ####################
    if type(image_nb) == int:
        image_nb = [image_nb]
    if len(counterroi) == 0:
        counterroi = [0, nb_pixel_y, 0, nb_pixel_x]

    assert (counterroi[0] >= 0
            and counterroi[1] <= nb_pixel_y
            and counterroi[2] >= 0
            and counterroi[3] <= nb_pixel_x), 'counter_roi setting does not match the detector size'
    nb_files = len(image_nb)
    if nb_files == 1:
        multiproc = False

    if load_scan:  # scan or time series
        datadir = rootdir + samplename + '_' + str('{:05d}'.format(scan)) + '/e4m/'
        template_file = datadir + samplename + '_' + str('{:05d}'.format(scan)) + "_data_"
    else:  # single image
        datadir = rootdir + samplename + '/e4m/'
        template_file = datadir + samplename + '_take_' + str('{:05d}'.format(scan)) + "_data_"
        compare_end = False

    if savedir == '':
        savedir = os.path.abspath(os.path.join(datadir, os.pardir)) + '/'

    #############
    # Load data #
    #############
    plt.ion()
    filenames = [template_file + '{:06d}.h5'.format(image_nb[idx]) for idx in range(nb_files)]
    start = time.time()

    if multiproc:
        print("\nNumber of processors used: ", min(mp.cpu_count(), len(filenames)))
        mp.freeze_support()
        pool = mp.Pool(processes=min(mp.cpu_count(), len(filenames)))  # use this number of processes

        for file in range(nb_files):
            pool.apply_async(load_p10_file, args=(filenames[file], file, counterroi, threshold),
                             callback=collect_result, error_callback=util.catch_error)

        pool.close()
        pool.join()  # postpones the execution of next line of code until all processes in the queue are done.

        # sort out counter values (we are using asynchronous multiprocessing, order is not preserved)
        roi_counter = sorted(counter, key=lambda x: x[1])

    else:
        for idx in range(nb_files):
            sys.stdout.write('\rLoading file {:d}'.format(idx + 1) + ' / {:d}'.format(nb_files))
            sys.stdout.flush()
            h5file = h5py.File(filenames[idx], 'r')
            data = h5file['entry']['data']['data'][:]
            nbz, nby, nbx = data.shape
            [counter.append(data[index, counterroi[0]:counterroi[1], counterroi[2]:counterroi[3]].sum())
                for index in range(nbz)]
            if compare_end and nb_files == 1:
                data_start, _ = pru.mask_eiger4m(data=data[0, :, :], mask=mask)
                data_start[np.log10(data_start) > threshold] = 0  # here we threshold a single image
                data_start = data_start.astype(float)
                data_stop, _ = pru.mask_eiger4m(data=data[-1, :, :], mask=mask)
                data_stop[np.log10(data_stop) > threshold] = 0  # here we threshold a single image
                data_stop = data_stop.astype(float)

                fig, _, _ = gu.imshow_plot(data_stop - data_start, plot_colorbar=True, scale='log',
                                           title='difference between the last frame and the first frame of the series')
            nb_frames = data.shape[0]
            data = data.sum(axis=0)  # data becomes 2D
            mask[np.log10(data) > nb_frames*threshold] = 1  # here we threshold the sum of nb_frames images
            data[mask == 1] = 0
            sumdata = sumdata + data
            roi_counter = [[counter, idx]]

    end = time.time()
    print('\nTime ellapsed for loading data:', str(datetime.timedelta(seconds=int(end - start))))

    frame_per_series = int(len(counter) / nb_files)

    print('')
    if load_scan:
        if nb_files > 1:
            plot_title = 'masked data - sum of ' + str(nb_files)\
                         + ' points with {:d} frames each'.format(frame_per_series)
        else:
            plot_title = 'masked data - sum of ' + str(frame_per_series) + ' frames'
        filename = 'S' + str(scan) + '_scan.png'
    else:  # single image
        plot_title = 'masked data'
        filename = 'S' + str(scan) + '_image_' + str(image_nb[0]) + '.png'

    sumdata, mask = pru.mask_eiger4m(data=sumdata, mask=mask)
    if savemask:
        fig, _, _ = gu.imshow_plot(mask, plot_colorbar=False, title='mask')
        np.savez_compressed(savedir+'hotpixels.npz', mask=mask)
        fig.savefig(savedir + 'mask.png')

    y0, x0 = np.unravel_index(abs(sumdata).argmax(), sumdata.shape)
    print("Max at (y, x): ", y0, x0, ' Max = ', int(sumdata[y0, x0]))

    if len(roi_counter[0][0]) > 1:  # roi_counter[0][0] is the list of counter intensities in a series
        int_roi = []
        [int_roi.append(val[0][idx]) for val in roi_counter for idx in range(frame_per_series)]
        plt.figure()
        plt.plot(np.asarray(int_roi))
        plt.title('Integrated intensity in counter_roi')
        plt.pause(0.1)

    fig, _, _ = gu.imshow_plot(sumdata, plot_colorbar=True, title=plot_title, vmin=0, scale='log')
    np.savez_compressed(savedir + 'hotpixels.npz', mask=mask)
    fig.savefig(savedir + filename)
    plt.show()


if __name__ == "__main__":
    main(parameters=params)
