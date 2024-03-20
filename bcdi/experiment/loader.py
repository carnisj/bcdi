# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr
#         Clement Atlan, c.atlan@outlook.com

"""
Implementation of beamline-dependent data loading classes.

The class methods manage the initialization of the file system and loading of data and
motor positions. Generic method are implemented in the abstract base class Loader, and
beamline-dependent methods need to be implemented in each child class (they are
decorated by @abstractmethod in the base class; they are written in italic in the
following diagram).

.. mermaid::
  :align: center

  classDiagram
    class Loader{
      <<abstract>>
      +name
      +sample_offsets
      create_logfile()*
      init_paths()*
      load_data()*
      motor_positions()*
      read_device()*
      read_monitor()*
      init_data_mask()
      init_monitor()
      load_check_dataset()

  }
    ABC <|-- Loader

API Reference
-------------

"""

from __future__ import annotations

import logging
import os
import re
import tkinter as tk
from abc import ABC, abstractmethod
from numbers import Integral
from tkinter import filedialog
from typing import TYPE_CHECKING, Any

import fabio
import h5py
import matplotlib.pyplot as plt
import numpy as np
from silx.io.specfile import SpecFile

import bcdi.utils.format as fmt
from bcdi.graph import graph_utils as gu
from bcdi.utils import utilities as util
from bcdi.utils import validation as valid
from bcdi.utils.io_helper import ContextFile, safeload

if TYPE_CHECKING:
    from bcdi.experiment.setup import Setup

module_logger = logging.getLogger(__name__)


def create_loader(name, sample_offsets, **kwargs):
    """
    Create the instance of the loader.

    :param name: str, name of the beamline
    :param sample_offsets: list or tuple of angles in degrees, corresponding to
     the offsets of each of the sample circles (the offset for the most outer circle
     should be at index 0). The number of circles is beamline dependent. Convention:
     the sample offsets will be subtracted to measurement the motor values.
    :param kwargs:

     - 'logger': an optional logger

    :return: the corresponding beamline instance
    """
    if name == "ID01":
        return LoaderID01(name=name, sample_offsets=sample_offsets, **kwargs)
    if name == "BM02":
        return LoaderBM02(name=name, sample_offsets=sample_offsets, **kwargs)
    if name == "ID27":
        return LoaderID27(name=name, sample_offsets=sample_offsets, **kwargs)
    if name == "ID01BLISS":
        return LoaderID01BLISS(name=name, sample_offsets=sample_offsets, **kwargs)
    if name in {"SIXS_2018", "SIXS_2019"}:
        return LoaderSIXS(name=name, sample_offsets=sample_offsets, **kwargs)
    if name == "34ID":
        return Loader34ID(name=name, sample_offsets=sample_offsets, **kwargs)
    if name == "P10":
        return LoaderP10(name=name, sample_offsets=sample_offsets, **kwargs)
    if name == "P10_SAXS":
        return LoaderP10SAXS(name=name, sample_offsets=sample_offsets, **kwargs)
    if name == "CRISTAL":
        return LoaderCRISTAL(name=name, sample_offsets=sample_offsets, **kwargs)
    if name == "NANOMAX":
        return LoaderNANOMAX(name=name, sample_offsets=sample_offsets, **kwargs)
    raise ValueError(f"Loader {name} not supported")


def check_empty_frames(data, mask=None, monitor=None, frames_logical=None, **kwargs):
    """
    Check if there is intensity for all frames.

    In case of beam dump, some frames may be empty. The data and optional mask will be
    cropped to remove those empty frames.

    :param data: a numpy 3D array
    :param mask: a numpy 3D array of 0 (pixel not masked) and 1 (masked pixel),
     same shape as data
    :param monitor: a numpy 1D array of shape equal to data.shape[0]
    :param frames_logical: 1D array of length equal to the number of measured frames.
     In case of cropping the length of the stack of frames changes. A frame whose
     index is set to 1 means that it is used, 0 means not used.
    :param kwargs: an optional logger
    :return:
     - cropped data as a numpy 3D array
     - cropped mask as a numpy 3D array
     - cropped monitor as a numpy 1D array
     - updated frames_logical

    """
    logger = kwargs.get("logger", module_logger)
    valid.valid_ndarray(arrays=data, ndim=3)
    if mask is not None:
        valid.valid_ndarray(arrays=mask, shape=data.shape)
    if monitor is not None:
        if not isinstance(monitor, np.ndarray):
            raise TypeError("monitor should be a numpy array")
        if monitor.ndim != 1 or len(monitor) != data.shape[0]:
            raise ValueError("monitor be a 1D array of length data.shae[0]")

    if frames_logical is None:
        frames_logical = np.ones(data.shape[0], dtype=int)
    valid.valid_1d_array(
        frames_logical,
        allowed_types=Integral,
        allow_none=False,
        allowed_values=(0, 1),
        name="frames_logical",
    )

    # check if there are empty frames
    is_intensity = np.zeros(data.shape[0])
    is_intensity[np.argwhere(data.sum(axis=(1, 2)))] = 1
    if is_intensity.sum() != data.shape[0]:
        logger.info("Empty frame detected, cropping the data")

    # remove empty frames from the data, update the mask, monitor and frames_logical
    data = data[np.nonzero(is_intensity)]
    mask = mask[np.nonzero(is_intensity)]
    monitor = monitor[np.nonzero(is_intensity)]
    frames_logical = util.update_frames_logical(
        frames_logical=frames_logical, logical_subset=is_intensity
    )
    return data, mask, monitor, frames_logical


def check_pixels(data, mask, debugging=False, **kwargs):
    """
    Check for hot pixels in the data using the mean value and the variance.

    :param data: 3D diffraction data
    :param mask: 2D or 3D mask. Mask will summed along the first axis if a 3D array.
    :param debugging: set to True to see plots
    :type debugging: bool
    :param kwargs: an optional logger
    :return: the filtered 3D data and the updated 2D mask.
    """
    logger = kwargs.get("logger", module_logger)
    valid.valid_ndarray(arrays=data, ndim=3)
    valid.valid_ndarray(arrays=mask, ndim=(2, 3))
    nbz, nby, nbx = data.shape

    if mask.ndim == 3:  # 3D array
        logger.info("Mask is a 3D array, summing it along axis 0")
        mask = mask.sum(axis=0)
        mask[np.nonzero(mask)] = 1
    valid.valid_ndarray(arrays=mask, shape=(nby, nbx))

    logger.info(
        "number of masked pixels due to detector gaps = "
        f"{int(mask.sum())} on a total of {nbx*nby}"
    )
    meandata = data.mean(axis=0)  # 2D
    vardata = 1 / data.var(axis=0)  # 2D
    var_mean = vardata[vardata != np.inf].mean()
    vardata[meandata == 0] = var_mean
    # pixels were data=0 (i.e. 1/variance=inf) are set to the mean of  1/var:
    # we do not want to mask pixels where there was no intensity during the scan

    if debugging:
        gu.combined_plots(
            tuple_array=(mask, meandata, vardata),
            tuple_sum_frames=False,
            tuple_sum_axis=0,
            tuple_width_v=None,
            tuple_width_h=None,
            tuple_colorbar=True,
            tuple_vmin=0,
            tuple_vmax=(1, 1, np.nan),
            tuple_scale=("linear", "linear", "linear"),
            tuple_title=(
                "Input mask",
                "check_pixels()\nmean(data) before masking",
                "check_pixels()\n1/var(data) before masking",
            ),
            reciprocal_space=True,
            position=(131, 132, 133),
        )

    # calculate the mean and variance of a single photon event along the rocking curve
    min_count = 0.99  # pixels with only 1 photon count along the rocking curve,
    # use the value 0.99 to be inclusive
    mean_singlephoton = min_count / nbz
    var_singlephoton = (
        ((nbz - 1) * mean_singlephoton**2 + (min_count - mean_singlephoton) ** 2)
        * 1
        / nbz
    )
    logger.info(f"var_mean={var_mean:.2f}, 1/var_threshold={1 / var_singlephoton:.2f}")

    # mask hotpixels with zero variance
    temp_mask = np.zeros((nby, nbx))
    temp_mask[vardata == np.inf] = 1
    # this includes only hotpixels since zero intensity pixels were set to var_mean
    mask[np.nonzero(temp_mask)] = 1  # update the mask with zero variance hotpixels
    vardata[vardata == np.inf] = 0  # update the array
    logger.info(f"number of zero variance hotpixels = {int(temp_mask.sum()):d}")

    # filter out pixels which have a variance smaller that the threshold
    # (note that  vardata = 1/data.var())
    indices_badpixels = np.nonzero(vardata > 1 / var_singlephoton)
    mask[indices_badpixels] = 1  # mask is 2D
    logger.info(
        f"number of pixels with too low variance = {indices_badpixels[0].shape[0]:d}"
    )

    # update the data array
    indices_badpixels = np.nonzero(mask)  # update indices
    for index in range(nbz):
        tempdata = data[index, :, :]
        tempdata[indices_badpixels] = (
            0  # numpy array is mutable hence data will be modified
        )

    if debugging:
        meandata = data.mean(axis=0)
        vardata = 1 / data.var(axis=0)
        vardata[meandata == 0] = var_mean  # 0 intensity pixels, not masked
        gu.combined_plots(
            tuple_array=(mask, meandata, vardata),
            tuple_sum_frames=False,
            tuple_sum_axis=0,
            tuple_width_v=None,
            tuple_width_h=None,
            tuple_colorbar=True,
            tuple_vmin=0,
            tuple_vmax=(1, 1, np.nan),
            tuple_scale="linear",
            tuple_title=(
                "Output mask",
                "check_pixels()\nmean(data) after masking",
                "check_pixels()\n1/var(data) after masking",
            ),
            reciprocal_space=True,
            position=(131, 132, 133),
        )
    return data, mask


def load_filtered_data(
    detector, frames_pattern: list[int] | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a filtered dataset and the corresponding mask.

    In that case

    :param detector: an instance of the class Detector
    :param frames_pattern: user-provided list which can be:
     - a binary list of length nb_images
     - a list of the indices of frames to be skipped

    :return: the data and the mask array
    """
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        initialdir=detector.datadir,
        title="Select data file",
        filetypes=[("NPZ", "*.npz")],
    )
    with np.load(file_path) as npzfile:
        data = npzfile[list(npzfile.files)[0]]

    file_path = filedialog.askopenfilename(
        initialdir=detector.datadir,
        title="Select mask file",
        filetypes=[("NPZ", "*.npz")],
    )
    with np.load(file_path) as npzfile:
        mask = npzfile[list(npzfile.files)[0]]

    valid.valid_container(
        frames_pattern,
        container_types=list,
        item_types=int,
        min_included=0,
        max_included=1,
        allow_none=True,
        name="frames_pattern",
    )
    nb_images = data.shape[0]

    if frames_pattern is not None and len(frames_pattern) != nb_images:
        # it means that frames_pattern is a list of indices of skipped frames
        nb_images += len(frames_pattern)

    frames_logical = util.generate_frames_logical(
        nb_images=nb_images, frames_pattern=frames_pattern
    )
    monitor = np.ones(len(frames_logical))
    return data, mask, monitor, frames_logical


def load_frame(
    frame,
    mask2d,
    monitor,
    frames_per_point,
    detector,
    loading_roi,
    flatfield=None,
    background=None,
    hotpixels=None,
    normalize="skip",
    bin_during_loading=False,
    debugging=False,
):
    """
    Load a frame and apply correction to it.

    :param frame: the frame to be loaded
    :param mask2d: a numpy array of the same shape as frame
    :param monitor: the volue of the intensity monitor for this frame
    :param frames_per_point: number of images summed to yield the 2D data
     (e.g. in a series measurement), used when defining the threshold for hot pixels
    :param detector: an instance of the class Detector
    :param loading_roi: user-defined region of interest, it may be larger than the
     physical size of the detector
    :param flatfield: the 2D flatfield array
    :param hotpixels: the 2D hotpixels array
    :param background: the 2D background array to subtract to the data
    :param normalize: 'monitor' to return the default monitor values, 'sum_roi' to
     return a monitor based on the integrated intensity in the region of interest
     defined by detector.sum_roi, 'skip' to do nothing
    :param bin_during_loading: if True, the data will be binned in the detector
     frame while loading. It saves a lot of memory space for large 2D detectors.
    :param debugging: set to True to see plots
    :return:
    """
    frame, mask2d = detector.mask_detector(
        frame,
        mask2d,
        nb_frames=frames_per_point,
        flatfield=flatfield,
        background=background,
        hotpixels=hotpixels,
    )

    if normalize == "sum_roi":
        monitor = util.sum_roi(array=frame, roi=detector.sum_roi)

    frame = frame[loading_roi[0] : loading_roi[1], loading_roi[2] : loading_roi[3]]

    if bin_during_loading:
        frame = util.bin_data(
            frame,
            (detector.binning[1], detector.binning[2]),
            debugging=debugging,
        )

    return frame, mask2d, monitor


def normalize_dataset(
    array, monitor, savedir=None, norm_to_min=True, debugging=False, **kwargs
):
    """
    Normalize array using the monitor values.

    :param array: the 3D array to be normalized
    :param monitor: the monitor values
    :param savedir: path where to save the debugging figure
    :param norm_to_min: bool, True to normalize to min(monitor) instead of max(monitor),
     avoid multiplying the noise
    :param debugging: bool, True to see plots
    :param kwargs: an optional logger
    :return:

     - normalized dataset
     - updated monitor
     - a title for plotting

    """
    logger = kwargs.get("logger", module_logger)
    valid.valid_ndarray(arrays=array, ndim=3)

    if monitor is None or len(monitor) == 0:
        logger.info("No monitor, skipping intensity normalization.")
        return array, np.ones(array.shape[0])

    ndim = array.ndim
    nbz, nby, nbx = array.shape
    original_max = None
    original_data = None

    if ndim != 3:
        raise ValueError("Array should be 3D")

    if debugging:
        original_data = np.copy(array)
        original_max = original_data.max()
        original_data[original_data < 5] = 0  # remove the background
        original_data = original_data.sum(
            axis=1
        )  # the first axis is the normalization axis

    logger.info(
        "Monitor min, max, mean: "
        f"{monitor.min():.1f}, {monitor.max():.1f}, {monitor.mean():.1f}"
    )

    if norm_to_min:
        logger.info("Data normalization by monitor.min()/monitor")
        monitor = monitor.min() / monitor  # will divide higher intensities
    else:  # norm to max
        logger.info("Data normalization by monitor.max()/monitor")
        monitor = monitor.max() / monitor  # will multiply lower intensities

    nbz = array.shape[0]
    if len(monitor) != nbz:
        raise ValueError(
            "The frame number and the monitor data length are different:",
            f"got {nbz} frames but {len(monitor)} monitor values",
        )

    for idx in range(nbz):
        array[idx, :, :] = array[idx, :, :] * monitor[idx]

    if debugging:
        norm_data = np.copy(array)
        # rescale norm_data to original_data for easier comparison
        norm_data = norm_data * original_max / norm_data.max()
        norm_data[norm_data < 5] = 0  # remove the background
        norm_data = norm_data.sum(axis=1)  # the first axis is the normalization axis
        fig = gu.combined_plots(
            tuple_array=(monitor, original_data, norm_data),
            tuple_sum_frames=False,
            tuple_colorbar=False,
            tuple_vmin=(np.nan, 0, 0),
            tuple_vmax=np.nan,
            tuple_title=(
                "monitor.min() / monitor",
                "Before norm (thres. 5)",
                "After norm (thres. 5)",
            ),
            tuple_scale=("linear", "log", "log"),
            xlabel=("Frame number", "Detector X", "Detector X"),
            is_orthogonal=False,
            ylabel=("Counts (a.u.)", "Frame number", "Frame number"),
            position=(211, 223, 224),
            reciprocal_space=True,
        )
        if savedir is not None:
            fig.savefig(savedir + f"monitor_{nbz}_{nby}_{nbx}.png")
        plt.close(fig)

    return array, monitor


def select_frames(data: np.ndarray, frames_logical: np.ndarray) -> np.ndarray:
    """
    Select frames, update the monitor and create a logical array.

    Override this method in the child classes of you want to implement a particular
    behavior, for example if two frames were taken at a same motor position and you
    want to delete one or average them...

    :param data: a 3D data array
    :param frames_logical: 1D array of length equal to the number of measured frames.
     In case of cropping the length of the stack of frames changes. A frame whose
     index is set to 1 means that it is used, 0 means not used.
    :return:
     - the updated 3D data, eventually cropped along the first axis
     - a 1D array of length the original number of 2D frames, 0 if a frame was
       removed, 1 if it wasn't. It can be used later to crop goniometer motor values
       accordingly.

    """
    valid.valid_1d_array(
        frames_logical,
        length=data.shape[0],
        allow_none=False,
        allowed_types=Integral,
        allowed_values=(0, 1),
        name="frames_logical",
    )
    return np.asarray(data[frames_logical != 0])


class Loader(ABC):
    """
    Base class for data loading.

    The frame used is the laboratory frame with the CXI convention (z downstream,
    y vertical up, x outboard).

    :param name: name of the beamline
    :param sample_offsets: list or tuple of angles in degrees, corresponding to
     the offsets of each of the sample circles (the offset for the most outer circle
     should be at index 0). The number of circles is beamline dependent. Convention:
     the sample offsets will be subtracted to measurement the motor values.
    :param kwargs:

     - 'logger': an optional logger

    """

    def __init__(self, name: str, sample_offsets: tuple[float, ...], **kwargs) -> None:
        self.logger = kwargs.get("logger", module_logger)
        self.name = name
        self.sample_offsets = sample_offsets

    @abstractmethod
    def create_logfile(
        self,
        datadir: str,
        name: str,
        root_folder: str,
        scan_number: int,
        filename: str | None = None,
        template_imagefile: str | None = None,
    ):
        """
        Create the logfile, which can be a log/spec file or the data itself.

        The nature of this file is beamline dependent.

        :param datadir: str, the data directory
        :param name: str, the name of the beamline, e.g. 'SIXS_2019'
        :param root_folder: str, the root directory of the experiment
        :param scan_number: the scan number to load
        :param filename: str, absolute path to the spec/fio/alias file when it exists
        :param template_imagefile: str, template for the data file name
        :return: an instance of a context manager ContextFile
        """

    def init_data_mask(
        self,
        detector,
        setup,
        normalize,
        nb_frames,
        bin_during_loading,
        **kwargs,
    ):
        """
        Initialize data, mask and region of interest for loading a dataset.

        :param detector: an instance of the class Detector
        :param setup: an instance of the class Setup
        :param normalize: 'monitor' to return the default monitor values, 'sum_roi' to
         return a monitor based on the integrated intensity in the region of interest
         defined by detector.sum_roi, 'skip' to do nothing
        :param nb_frames: number of data points (not including series at each point)
        :param bin_during_loading: if True, the data will be binned in the detector
         frame while loading. It saves a lot of memory space for large 2D detectors.
        :param kwargs:

         - 'scan_number': int, the scan number to load

        :return:

         - the empty 3D data array
         - the 2D mask array initialized with 0 values
         - the initialized monitor as a 1D array
         - the region of interest use for loading the data, as a list of 4 integers

        """
        # define the loading ROI, the user-defined ROI may be larger than the physical
        # detector size
        if (
            detector.roi[0] < 0
            or detector.roi[1] > detector.unbinned_pixel_number[0]
            or detector.roi[2] < 0
            or detector.roi[3] > detector.unbinned_pixel_number[1]
        ):
            self.logger.info(
                "Data shape is limited by detector size,"
                " loaded data will be smaller than as defined by the ROI."
            )
        loading_roi = [
            max(0, detector.roi[0]),
            min(detector.unbinned_pixel_number[0], detector.roi[1]),
            max(0, detector.roi[2]),
            min(detector.unbinned_pixel_number[1], detector.roi[3]),
        ]

        # initialize the data array, the mask is binned afterwards in load_check_dataset
        if bin_during_loading:
            self.logger.info(
                f"Binning the data: detector vertical axis by {detector.binning[1]}, "
                f"detector horizontal axis by {detector.binning[2]}"
            )

            data = np.empty(
                (
                    nb_frames,
                    (loading_roi[1] - loading_roi[0]) // detector.binning[1],
                    (loading_roi[3] - loading_roi[2]) // detector.binning[2],
                ),
                dtype=float,
            )
        else:
            data = np.empty(
                (
                    nb_frames,
                    loading_roi[1] - loading_roi[0],
                    loading_roi[3] - loading_roi[2],
                ),
                dtype=float,
            )

        # initialize the monitor
        monitor = self.init_monitor(
            normalize=normalize,
            nb_frames=nb_frames,
            setup=setup,
            **kwargs,
        )

        return (
            data,
            np.zeros(detector.unbinned_pixel_number),
            monitor,
            loading_roi,
        )

    def init_monitor(self, normalize, nb_frames, setup, **kwargs):
        """
        Initialize the monitor for normalization.

        :param normalize: 'monitor' to return the default monitor values, 'sum_roi' to
         return a monitor based on the integrated intensity in the region of interest
         defined by detector.sum_roi, 'skip' to do nothing
        :param nb_frames: number of data points (not including series at each point)
        :param setup: an instance of the class Setup
        :param kwargs:

         - 'scan_number': int, the scan number to load

        :return: the initialized monitor as a 1D array
        """
        monitor = None
        if normalize == "sum_roi":
            monitor = np.zeros(nb_frames)
        elif normalize == "monitor":
            if setup.custom_scan:
                monitor = setup.custom_monitor
            else:
                monitor = self.read_monitor(setup=setup, **kwargs)
        if monitor is None or len(monitor) == 0:
            monitor = np.ones(nb_frames)
            self.logger.info("Skipping intensity normalization.")
        return monitor

    @staticmethod
    @abstractmethod
    def init_paths(root_folder, sample_name, scan_number, template_imagefile, **kwargs):
        """
        Initialize paths used for data processing and logging.

        :param root_folder: folder of the experiment, where all scans are stored
        :param sample_name: string in front of the scan number in the data folder
         name.
        :param scan_number: int, the scan number
        :param template_imagefile: beamline-dependent template for the data files:

         - ID01: 'data_mpx4_%05d.edf.gz' or 'align_eiger2M_%05d.edf.gz'
         - SIXS_2018: 'align.spec_ascan_mu_%05d.nxs'
         - SIXS_2019: 'spare_ascan_mu_%05d.nxs'
         - Cristal: 'S%d.nxs'
         - P10: '_master.h5'
         - NANOMAX: '%06d.h5'
         - 34ID: 'Sample%dC_ES_data_51_256_256.npz'

        :param kwargs: dictionnary of the setup parameters including the following keys:

         - 'specfile_name': beamline-dependent string:

           - ID01: name of the spec file without '.spec'
           - SIXS_2018 and SIXS_2019: None or full path of the alias dictionnary (e.g.
             root_folder+'alias_dict_2019.txt')
           - empty string for all other beamlines

        :return: a tuple of strings:

         - homedir: the path of the scan folder
         - default_dirname: the name of the folder containing images / raw data
         - specfile: the name of the specfile if it exists
         - template_imagefile: the template for data/image file names

        """

    def load_check_dataset(
        self,
        scan_number,
        setup,
        frames_pattern=None,
        flatfield=None,
        hotpixels=None,
        background=None,
        normalize="skip",
        bin_during_loading=False,
        debugging=False,
    ):
        """
        Load data, apply filters and concatenate it for phasing.

        :param scan_number: the scan number to load
        :param setup: an instance of the class Setup
        :param frames_pattern: user-provided list which can be:
         - a binary list of length nb_images
         - a list of the indices of frames to be skipped

        :param flatfield: the 2D flatfield array
        :param hotpixels: the 2D hotpixels array. 1 for a hotpixel, 0 for normal pixels.
        :param background: the 2D background array to subtract to the data
        :param normalize: 'monitor' to return the default monitor values, 'sum_roi' to
         return a monitor based on the integrated intensity in the region of interest
         defined by detector.sum_roi, 'skip' to do nothing
        :param bin_during_loading: if True, the data will be binned in the detector
         frame while loading. It saves a lot of memory for large detectors.
        :param debugging: set to True to see plots
        :return:

         - the 3D data array in the detector frame
         - the 3D mask array
         - the monitor values for normalization
         - frames_logical: 1D array of length equal to the number of measured frames.
           In case of cropping the length of the stack of frames changes. A frame whose
           index is set to 1 means that it is used, 0 means not used.

        """
        self.logger.info(
            "User-defined ROI size (VxH): "
            f"({setup.detector.roi[1] - setup.detector.roi[0]}, "
            f"{setup.detector.roi[3] - setup.detector.roi[2]})"
        )
        self.logger.info(
            "Detector physical size without binning (VxH): "
            f"({setup.detector.unbinned_pixel_number[0]}, "
            f"{setup.detector.unbinned_pixel_number[1]})"
        )
        self.logger.info(
            "Detector size with binning (VxH): "
            f"({setup.detector.unbinned_pixel_number[0] // setup.detector.binning[1]}, "
            f"{setup.detector.unbinned_pixel_number[1] // setup.detector.binning[2]})"
        )

        if setup.filtered_data:
            data, mask3d, monitor, frames_logical = load_filtered_data(
                detector=setup.detector, frames_pattern=frames_pattern
            )
        else:
            data, mask2d, monitor, loading_roi = self.load_data(
                setup=setup,
                scan_number=scan_number,
                detector=setup.detector,
                flatfield=flatfield,
                hotpixels=hotpixels,
                background=background,
                normalize=normalize,
                bin_during_loading=bin_during_loading,
                debugging=debugging,
            )

            ###################
            # update the mask #
            ###################
            mask2d = mask2d[
                loading_roi[0] : loading_roi[1], loading_roi[2] : loading_roi[3]
            ]
            if bin_during_loading:
                mask2d = util.bin_data(
                    mask2d,
                    (setup.detector.binning[1], setup.detector.binning[2]),
                    debugging=debugging,
                )
            mask2d[np.nonzero(mask2d)] = 1

            #################
            # select frames #
            #################
            frames_logical = util.generate_frames_logical(
                nb_images=data.shape[0], frames_pattern=frames_pattern
            )
            data = select_frames(data=data, frames_logical=frames_logical)

            #################################
            # crop the monitor if necessary #
            #################################
            monitor = util.apply_logical_array(
                arrays=monitor, frames_logical=frames_logical
            )

            ########################################
            # check for abnormally behaving pixels #
            ########################################
            data, mask2d = check_pixels(
                data=data, mask=mask2d, debugging=debugging, logger=self.logger
            )
            mask3d = np.repeat(mask2d[np.newaxis, :, :], data.shape[0], axis=0)
            mask3d[np.isnan(data)] = 1
            data[np.isnan(data)] = 0

            ####################################
            # check for empty frames (no beam) #
            ####################################
            data, mask3d, monitor, frames_logical = check_empty_frames(
                data=data,
                mask=mask3d,
                monitor=monitor,
                frames_logical=frames_logical,
                logger=self.logger,
            )

            ###########################
            # intensity normalization #
            ###########################
            if normalize == "skip":
                self.logger.info("Skip intensity normalization")
            else:
                self.logger.info(f"Intensity normalization using {normalize}")
                data, monitor = normalize_dataset(
                    array=data,
                    monitor=monitor,
                    norm_to_min=True,
                    savedir=setup.detector.savedir,
                    debugging=debugging,
                    logger=self.logger,
                )

            ##########################################################################
            # check for negative pixels, it can happen when subtracting a background #
            ##########################################################################
            self.logger.info(f"{(data < 0).sum()} negative data points masked")
            mask3d[data < 0] = 1
            data[data < 0] = 0

        return data, mask3d, monitor, frames_logical

    @abstractmethod
    def load_data(
        self,
        setup,
        flatfield=None,
        hotpixels=None,
        background=None,
        normalize="skip",
        bin_during_loading=False,
        debugging=False,
        **kwargs,
    ):
        """
        Load data including detector/background corrections.

        :param setup: an instance of the class Setup
        :param flatfield: the 2D flatfield array
        :param hotpixels: the 2D hotpixels array
        :param background: the 2D background array to subtract to the data
        :param normalize: 'monitor' to return the default monitor values, 'sum_roi' to
         return a monitor based on the integrated intensity in the region of interest
         defined by detector.sum_roi, 'skip' to do nothing
        :param bin_during_loading: if True, the data will be binned in the detector
         frame while loading. It saves a lot of memory space for large 2D detectors.
        :param debugging: set to True to see plots

        :param kwargs: beamline_specific parameters, which may include part of the
         totality of the following keys:

          - 'scan_number': the scan number to load (e.g. for ID01)

        :return: in this order

         - the 3D data array in the detector frame
         - the 2D mask array
         - the monitor values for normalization as a 1D array of length data.shape[0]
         - frames_logical as a 1D array of length the original number of 2D frames, 0 if
           a frame was removed, 1 if it wasn't. It can be used later to crop goniometer
           motor values accordingly.

        """

    @abstractmethod
    def motor_positions(self, setup, **kwargs):
        """
        Retrieve motor positions.

        This method is beamline dependent. It must be implemented in the child classes.

        :param setup: an instance of the class Setup
        :param kwargs: beamline_specific parameters, see the documentation for the
         child class.
        :return: the diffractometer motors positions for the particular setup. The
         energy (1D array or number) and the sample to detector distance are expected to
         be the last elements of the tuple in this order.
        """

    @abstractmethod
    def read_device(self, setup, device_name: str, **kwargs):
        """
        Extract the scanned device positions/values.

        :param setup: an instance of the class Setup
        :param device_name: name of the scanned device
        :param kwargs: beamline_specific parameters, which may include part of the
         totality of the following keys:

          - 'scan_number': int, number of the scan (e.g. for ID01)

        :return: the positions/values of the device as a numpy 1D array
        """

    @abstractmethod
    def read_monitor(self, setup: Setup, **kwargs) -> np.ndarray:
        """
        Load the default monitor for intensity normalization of the considered beamline.

        :param setup: an instance of the class Setup
        :param kwargs: beamline_specific parameter

          - 'scan_number': int, number of the scan (e.g. for ID01)

        :return: the default monitor values
        """

    def __repr__(self):
        """Representation string of the Loader instance."""
        return fmt.create_repr(self, Loader)


class LoaderID01(Loader):
    """Loader for ESRF ID01 beamline before the deployement of BLISS."""

    motor_table = {
        "old_names": {
            "mu": "Mu",
            "eta": "Eta",
            "phi": "Phi",
            "nu": "Nu",
            "delta": "Delta",
            "energy": "Energy",
        },
        "new_names": {
            "mu": "mu",
            "eta": "eta",
            "phi": "phi",
            "nu": "nu",
            "delta": "del",
            "energy": "nrj",
        },
    }

    def create_logfile(
        self,
        datadir: str,
        name: str,
        root_folder: str,
        scan_number: int,
        filename: str | None = None,
        template_imagefile: str | None = None,
    ) -> ContextFile:
        """
        Create the logfile, which is the spec file for ID01.

        :param datadir: str, the data directory
        :param name: str, the name of the beamline, e.g. 'SIXS_2019'
        :param root_folder: str, the root directory of the experiment, where is e.g. the
           specfile file.
        :param scan_number: the scan number to load
        :param filename: str, name of the spec file or full path of the spec file
        :param template_imagefile: str, template for the data file name
        :return: an instance of a context manager ContextFile
        """
        valid.valid_container(
            root_folder,
            container_types=str,
            min_length=1,
            name="root_folder",
        )
        if not os.path.isdir(root_folder):
            raise ValueError(f"The directory {root_folder} does not exist")
        valid.valid_container(
            filename,
            container_types=str,
            min_length=1,
            name="filename",
        )
        valid.valid_item(
            scan_number, allowed_types=int, min_included=1, name="scan_number"
        )
        path = util.find_file(
            filename=filename, default_folder=root_folder, logger=self.logger
        )
        return ContextFile(filename=path, open_func=SpecFile, scan_number=scan_number)

    @staticmethod
    def init_paths(
        root_folder: str,
        sample_name: str,
        scan_number: int,
        template_imagefile: str,
        **kwargs,
    ) -> tuple[str, str, str | None, str]:
        """
        Initialize paths used for data processing and logging at ID01.

        :param root_folder: folder of the experiment, where all scans are stored
        :param sample_name: string in front of the scan number in the data folder
         name.
        :param scan_number: int, the scan number
        :param template_imagefile: template for the data files, e.g.
         'data_mpx4_%05d.edf.gz' or 'align_eiger2M_%05d.edf.gz'
        :param kwargs:
         - 'specfile_name': name of the spec file without '.spec'

        :return: a tuple of strings:

         - homedir: the path of the scan folder
         - default_dirname: the name of the folder containing images / raw data
         - specfile: the name of the specfile if it exists
         - template_imagefile: the template for data/image file names

        """
        specfile_name = kwargs.get("specfile_name")

        homedir = root_folder + sample_name + str(scan_number) + "/"
        default_dirname = "data/"
        return homedir, default_dirname, specfile_name, template_imagefile

    @safeload
    def load_data(
        self,
        setup: Setup,
        flatfield: np.ndarray | None = None,
        hotpixels: np.ndarray | None = None,
        background: np.ndarray | None = None,
        normalize: str = "skip",
        bin_during_loading: bool = False,
        debugging: bool = False,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """
        Load ID01 data, apply filters and concatenate it for phasing.

        :param setup: an instance of the class Setup
        :param flatfield: the 2D flatfield array
        :param hotpixels: the 2D hotpixels array
        :param background: the 2D background array to subtract to the data
        :param normalize: 'monitor' to return the default monitor values, 'sum_roi' to
         return a monitor based on the integrated intensity in the region of interest
         defined by detector.sum_roi, 'skip' to do nothing
        :param bin_during_loading: if True, the data will be binned in the detector
         frame while loading. It saves a lot of memory space for large 2D detectors.
        :param debugging: set to True to see plots
        :return:
         - the 3D data array in the detector frame
         - the 2D mask array
         - the monitor values for normalization
         - the detector region of interest used for loading the data

        """
        file = kwargs.get("file")  # this kwarg is provided by @safeload
        if file is None:
            raise ValueError("file should be the opened file, not None")
        if setup.logfile is None:
            raise ValueError("logfile undefined")
        scan_number = setup.logfile.scan_number
        if not isinstance(scan_number, int):
            raise TypeError(
                "scan_number should be an integer, got " f"{type(scan_number)}"
            )
        if setup.detector.template_imagefile is None:
            raise ValueError("'template_imagefile' must be defined to load the images.")
        ccdfiletmp = os.path.join(
            setup.detector.datadir, setup.detector.template_imagefile
        )
        data_stack = None
        if not setup.custom_scan:
            # create the template for the image files
            labels = file[str(scan_number) + ".1"].labels  # motor scanned
            labels_data = file[str(scan_number) + ".1"].data  # motor scanned

            # find the number of images
            try:
                ccdn = labels_data[labels.index(setup.detector.counter("ID01")), :]
            except ValueError:
                try:
                    self.logger.info(
                        f"{setup.detector.counter('ID01')} not in the list, "
                        "trying 'ccd_n'",
                    )
                    ccdn = labels_data[labels.index("ccd_n"), :]
                except ValueError:
                    raise ValueError(
                        "ccd_n not in the list, the detector name may be wrong",
                    )
            nb_img = len(ccdn)
        else:
            ccdn = None  # not used for custom scans
            # create the template for the image files
            if len(setup.custom_images) == 0:
                raise ValueError("No image number provided in 'custom_images'")

            if len(setup.custom_images) > 1:
                nb_img = len(setup.custom_images)
            else:  # the data is stacked into a single file
                with np.load(ccdfiletmp % setup.custom_images[0]) as npzfile:
                    data_stack = npzfile[list(npzfile.files)[0]]
                nb_img = data_stack.shape[0]

        data, mask2d, monitor, loading_roi = self.init_data_mask(
            detector=setup.detector,
            setup=setup,
            normalize=normalize,
            nb_frames=nb_img,
            bin_during_loading=bin_during_loading,
            scan_number=scan_number,
        )

        # loop over frames, mask the detector and normalize / bin
        for idx in range(nb_img):
            if data_stack is not None:
                # custom scan with a stacked data loaded
                ccdraw = data_stack[idx, :, :]
            else:
                if setup.custom_scan:
                    # custom scan with one file per frame
                    i = int(setup.custom_images[idx])
                else:
                    i = int(ccdn[idx])
                with fabio.open(ccdfiletmp % i) as e:
                    ccdraw = e.data

            data[idx, :, :], mask2d, monitor[idx] = load_frame(
                frame=ccdraw,
                mask2d=mask2d,
                monitor=monitor[idx],
                frames_per_point=1,
                detector=setup.detector,
                loading_roi=loading_roi,
                flatfield=flatfield,
                background=background,
                hotpixels=hotpixels,
                normalize=normalize,
                bin_during_loading=bin_during_loading,
                debugging=debugging,
            )
        return data, mask2d, monitor, loading_roi

    @safeload
    def motor_positions(
        self, setup: Setup, **kwargs
    ) -> tuple[float | list | np.ndarray, ...]:
        """
        Load the scan data and extract motor positions.

        Stages names for data previous to ?2017? start with a capital letter.

        :param setup: an instance of the class Setup
        :return: (mu, eta, phi, nu, delta, energy) values
        """
        # load and check kwargs
        file = kwargs.get("file")  # this kwarg is provided by @safeload
        if file is None:
            raise ValueError("file should be the opened file, not None")
        if setup.logfile is None:
            raise ValueError("logfile undefined")
        scan_number = setup.logfile.scan_number
        if not isinstance(scan_number, int):
            raise TypeError(
                "scan_number should be an integer, got " f"{type(scan_number)}"
            )

        old_names = False
        if not setup.custom_scan:
            motor_names = file[str(scan_number) + ".1"].motor_names
            # positioners
            motor_values = file[str(scan_number) + ".1"].motor_positions
            # positioners
            labels = file[str(scan_number) + ".1"].labels  # motor scanned
            labels_data = file[str(scan_number) + ".1"].data  # motor scanned

            try:
                _ = motor_values[motor_names.index("nu")]  # positioner
            except ValueError:
                self.logger.info("'nu' not in the list, trying 'Nu'")
                _ = motor_values[motor_names.index("Nu")]  # positioner
                self.logger.info("Defaulting to old ID01 motor names")
                old_names = True

            if old_names:
                motor_table = self.motor_table["old_names"]
            else:
                motor_table = self.motor_table["new_names"]

            if motor_table["mu"] in labels:
                mu = labels_data[labels.index(motor_table["mu"]), :]  # scanned
            else:
                mu = motor_values[motor_names.index(motor_table["mu"])]  # positioner

            if motor_table["eta"] in labels:
                eta = labels_data[labels.index(motor_table["eta"]), :]  # scanned
            else:
                eta = motor_values[motor_names.index(motor_table["eta"])]  # positioner

            if motor_table["phi"] in labels:
                phi = labels_data[labels.index(motor_table["phi"]), :]  # scanned
            else:
                phi = motor_values[motor_names.index(motor_table["phi"])]  # positioner

            if motor_table["delta"] in labels:
                delta = labels_data[labels.index(motor_table["delta"]), :]  # scanned
            else:  # positioner
                delta = motor_values[motor_names.index(motor_table["delta"])]

            if motor_table["nu"] in labels:
                nu = labels_data[labels.index(motor_table["nu"]), :]  # scanned
            else:  # positioner
                nu = motor_values[motor_names.index(motor_table["nu"])]

            if motor_table["energy"] in labels:
                energy = labels_data[labels.index(motor_table["energy"]), :]
                # energy scanned, override the user-defined energy
            else:  # positioner
                energy = motor_values[motor_names.index(motor_table["energy"])]

            energy = energy * 1000.0  # switch to eV
            # remove user-defined sample offsets (sample: mu, eta, phi)
            mu = mu - self.sample_offsets[0]
            eta = eta - self.sample_offsets[1]
            phi = phi - self.sample_offsets[2]

        else:  # manually defined custom scan
            mu = setup.custom_motors["mu"]
            eta = setup.custom_motors["eta"]
            phi = setup.custom_motors["phi"]
            delta = setup.custom_motors["delta"]
            nu = setup.custom_motors["nu"]
            energy = setup.energy

        detector_distance = (
            self.retrieve_distance(
                filename=setup.detector.specfile, default_folder=setup.detector.rootdir
            )
            or setup.distance
        )
        return mu, eta, phi, nu, delta, energy, detector_distance

    @safeload
    def read_device(self, setup: Setup, device_name: str, **kwargs) -> np.ndarray:
        """
        Extract the scanned device positions/values at ID01 beamline.

        :param setup: an instance of the class Setup
        :param device_name: name of the scanned device
        :return: the positions/values of the device as a numpy 1D array
        """
        file = kwargs.get("file")  # this kwarg is provided by @safeload_static
        if file is None:
            raise ValueError("file should be the opened file, not None")
        if setup.logfile is None:
            raise ValueError("logfile undefined")
        scan_number = setup.logfile.scan_number
        if not isinstance(scan_number, int):
            raise TypeError(
                "scan_number should be an integer, got " f"{type(scan_number)}"
            )

        labels = file[str(scan_number) + ".1"].labels  # motor scanned
        labels_data = file[str(scan_number) + ".1"].data  # motor scanned
        self.logger.info(f"Trying to load values for {device_name}...")
        try:
            device_values = list(labels_data[labels.index(device_name), :])
            self.logger.info(f"{device_name} found!")
        except ValueError:  # device not in the list
            self.logger.info(f"no device {device_name} in the logfile")
            device_values = []
        return np.asarray(device_values)

    def read_monitor(self, setup: Setup, **kwargs: dict[str, Any]) -> np.ndarray:
        """
        Load the default monitor for a dataset measured at ID01.

        :param setup: an instance of the class Setup
        :return: the default monitor values
        """
        if setup.logfile is None:
            raise ValueError("logfile undefined")
        scan_number = setup.logfile.scan_number
        if not isinstance(scan_number, int):
            raise TypeError(
                "scan_number should be an integer, got " f"{type(scan_number)}"
            )
        if setup.actuators is not None:
            monitor_name = setup.actuators.get("monitor", "exp1")
        else:
            monitor_name = "exp1"
        monitor: np.ndarray = self.read_device(
            setup=setup, scan_number=scan_number, device_name=monitor_name
        )
        return monitor

    def retrieve_distance(self, filename: str, default_folder: str) -> float | None:
        """
        Load the spec file and retrieve the detector distance if it has been calibrated.

        :param filename: name of the spec file
        :param default_folder: folder where the spec file is expected
        :return: the detector distance in meters or None
        """
        distance = None
        found_distance = 0
        if not filename.endswith(".spec"):
            raise ValueError(f"Expecting a spec file, got {filename}")
        with open(
            util.find_file(
                filename=filename,
                default_folder=default_folder,
                logger=self.logger,
            ),
        ) as file:
            lines = file.readlines()
            for line in lines:
                if line.startswith("#UDETCALIB"):
                    words = line.split(",")
                    for word in words:
                        if word.startswith("det_distance_COM"):
                            distance = float(word[17:])
                            found_distance += 1

        if found_distance > 1:
            self.logger.info(
                "multiple dectector distances found in the spec file, using"
                f"{distance} m."
            )
        return distance


class LoaderBM02(Loader):
    """Loader for ESRF BM02 beamline before the deployement of BLISS."""

    def create_logfile(
        self,
        datadir: str,
        name: str,
        root_folder: str,
        scan_number: int,
        filename: str | None = None,
        template_imagefile: str | None = None,
    ) -> ContextFile:
        """
        Create the logfile, which is the spec file for BM02.

        :param datadir: str, the data directory
        :param name: str, the name of the beamline, e.g. 'BM02'
        :param root_folder: str, the root directory of the experiment, where is e.g. the
           specfile file.
        :param scan_number: the scan number to load
        :param filename: str, name of the spec file or full path of the spec file
        :param template_imagefile: str, template for the data file name
        :return: an instance of a context manager ContextFile
        """
        valid.valid_container(
            root_folder,
            container_types=str,
            min_length=1,
            name="root_folder",
        )
        if not os.path.isdir(root_folder):
            raise ValueError(f"The directory {root_folder} does not exist")
        valid.valid_container(
            filename,
            container_types=str,
            min_length=1,
            name="filename",
        )
        valid.valid_item(
            scan_number, allowed_types=int, min_included=1, name="scan_number"
        )
        path = util.find_file(
            filename=filename, default_folder=root_folder, logger=self.logger
        )
        return ContextFile(filename=path, open_func=SpecFile, scan_number=scan_number)

    @staticmethod
    def init_paths(
        root_folder: str,
        sample_name: str,
        scan_number: int,
        template_imagefile: str,
        **kwargs,
    ) -> tuple[str, str, str | None, str]:
        """
        Initialize paths used for data processing and logging at BM02.

        :param root_folder: folder of the experiment, where all scans are stored
        :param sample_name: string in front of the scan number in the data folder
         name.
        :param scan_number: int, the scan number
        :param template_imagefile: template for the data files, e.g. 'PtYSZ_%04d.edf'.
        :param kwargs:
         - 'specfile_name': name of the spec file without '.spec'

        :return: a tuple of strings:

         - homedir: the path of the scan folder
         - default_dirname: the name of the folder containing images / raw data
         - specfile: the name of the specfile if it exists
         - template_imagefile: the template for data/image file names

        """
        specfile_name = kwargs.get("specfile_name")

        homedir = root_folder + sample_name + str(scan_number) + "/"
        default_dirname = "data/"
        return homedir, default_dirname, specfile_name, template_imagefile

    @safeload
    def load_data(
        self,
        setup: Setup,
        flatfield: np.ndarray | None = None,
        hotpixels: np.ndarray | None = None,
        background: np.ndarray | None = None,
        normalize: str = "skip",
        bin_during_loading: bool = False,
        debugging: bool = False,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """
        Load BM02 data, apply filters and concatenate it for phasing.

        :param setup: an instance of the class Setup
        :param flatfield: the 2D flatfield array
        :param hotpixels: the 2D hotpixels array
        :param background: the 2D background array to subtract to the data
        :param normalize: 'monitor' to return the default monitor values, 'sum_roi' to
         return a monitor based on the integrated intensity in the region of interest
         defined by detector.sum_roi, 'skip' to do nothing
        :param bin_during_loading: if True, the data will be binned in the detector
         frame while loading. It saves a lot of memory space for large 2D detectors.
        :param debugging: set to True to see plots
        :return:
         - the 3D data array in the detector frame
         - the 2D mask array
         - the monitor values for normalization
         - the detector region of interest used for loading the data

        """
        file = kwargs.get("file")  # this kwarg is provided by @safeload
        if file is None:
            raise ValueError("file should be the opened file, not None")
        if setup.logfile is None:
            raise ValueError("logfile undefined")
        scan_number = setup.logfile.scan_number
        if not isinstance(scan_number, int):
            raise TypeError(
                "scan_number should be an integer, got " f"{type(scan_number)}"
            )
        if setup.detector.template_imagefile is None:
            raise ValueError("'template_imagefile' must be defined to load the images.")
        ccdfiletmp = os.path.join(
            setup.detector.datadir, setup.detector.template_imagefile
        )
        data_stack = None
        if not setup.custom_scan:
            # create the template for the image files
            labels = file[str(scan_number) + ".1"].labels  # motor scanned
            labels_data = file[str(scan_number) + ".1"].data  # motor scanned

            # find the number of images
            try:
                ccdn = labels_data[labels.index(setup.detector.counter("BM02")), :]
            except ValueError:
                raise ValueError(
                    "img not in the list, the detector name may be wrong",
                )
            nb_img = len(ccdn)
        else:
            ccdn = None  # not used for custom scans
            # create the template for the image files
            if len(setup.custom_images) == 0:
                raise ValueError("No image number provided in 'custom_images'")

            if len(setup.custom_images) > 1:
                nb_img = len(setup.custom_images)
            else:  # the data is stacked into a single file
                with np.load(ccdfiletmp % setup.custom_images[0]) as npzfile:
                    data_stack = npzfile[list(npzfile.files)[0]]
                nb_img = data_stack.shape[0]

        data, mask2d, monitor, loading_roi = self.init_data_mask(
            detector=setup.detector,
            setup=setup,
            normalize=normalize,
            nb_frames=nb_img,
            bin_during_loading=bin_during_loading,
            scan_number=scan_number,
        )

        # loop over frames, mask the detector and normalize / bin
        for idx in range(nb_img):
            if data_stack is not None:
                # custom scan with a stacked data loaded
                ccdraw = data_stack[idx, :, :]
            else:
                if setup.custom_scan:
                    # custom scan with one file per frame
                    i = int(setup.custom_images[idx])
                else:
                    i = int(ccdn[idx])
                with fabio.open(ccdfiletmp % i) as e:
                    ccdraw = e.data

            data[idx, :, :], mask2d, monitor[idx] = load_frame(
                frame=ccdraw,
                mask2d=mask2d,
                monitor=monitor[idx],
                frames_per_point=1,
                detector=setup.detector,
                loading_roi=loading_roi,
                flatfield=flatfield,
                background=background,
                hotpixels=hotpixels,
                normalize=normalize,
                bin_during_loading=bin_during_loading,
                debugging=debugging,
            )
        return data, mask2d, monitor, loading_roi

    @safeload
    def motor_positions(
        self, setup: Setup, **kwargs
    ) -> tuple[float | list | np.ndarray, ...]:
        """
        Load the scan data and extract motor positions.

        :param setup: an instance of the class Setup
        :return: (mu, th, chi, phi, nu, tth, energy) values
        """
        # load and check kwargs
        file = kwargs.get("file")  # this kwarg is provided by @safeload
        if file is None:
            raise ValueError("file should be the opened file, not None")
        if setup.logfile is None:
            raise ValueError("logfile undefined")
        scan_number = setup.logfile.scan_number
        if not isinstance(scan_number, int):
            raise TypeError(
                "scan_number should be an integer, got " f"{type(scan_number)}"
            )

        if not setup.custom_scan:
            motor_names = file[str(scan_number) + ".1"].motor_names
            # positioners
            motor_values = file[str(scan_number) + ".1"].motor_positions
            # positioners
            labels = file[str(scan_number) + ".1"].labels  # motor scanned
            labels_data = file[str(scan_number) + ".1"].data  # motor scanned

            try:
                _ = motor_values[motor_names.index("nu")]  # positioner
            except ValueError:
                self.logger.info("'nu' not in the list, trying 'Nu'")
                _ = motor_values[motor_names.index("Nu")]  # positioner
                self.logger.info("Defaulting to old ID01 motor names")

            if "mu" in labels:
                mu = labels_data[labels.index("mu"), :]  # scanned
            else:
                mu = motor_values[motor_names.index("mu")]  # positioner

            if "THETA" in labels:
                th = labels_data[labels.index("THETA"), :]  # scanned
            else:
                th = motor_values[motor_names.index("THETA")]  # positioner

            if "CHI" in labels:
                chi = labels_data[labels.index("CHI"), :]  # scanned
            else:
                chi = motor_values[motor_names.index("CHI")]  # positioner

            if "PHI" in labels:
                phi = labels_data[labels.index("PHI"), :]  # scanned
            else:
                phi = motor_values[motor_names.index("PHI")]  # positioner

            if "2THETA" in labels:
                tth = labels_data[labels.index("2THETA"), :]  # scanned
            else:  # positioner
                tth = motor_values[motor_names.index("2THETA")]

            if "nu" in labels:
                nu = labels_data[labels.index("nu"), :]  # scanned
            else:  # positioner
                nu = motor_values[motor_names.index("nu")]

            if "Emono" in labels:
                energy = labels_data[labels.index("Emono"), :]
                # energy scanned, override the user-defined energy
            else:  # positioner
                energy = motor_values[motor_names.index("Emono")]

            energy = energy * 1000.0  # switch to eV
            mu = mu - self.sample_offsets[0]
            th = th - self.sample_offsets[1]
            chi = chi - self.sample_offsets[2]
            phi = phi - self.sample_offsets[3]

        else:  # manually defined custom scan
            try:
                mu = setup.custom_motors["mu"]
                th = setup.custom_motors["th"]
                chi = setup.custom_motors["chi"]
                phi = setup.custom_motors["phi"]
                tth = setup.custom_motors["tth"]
                nu = setup.custom_motors["nu"]
            except KeyError:
                self.logger.error(
                    "Expected keys: 'mu', 'th', 'chi', 'phi', 'tth', 'nu'"
                )
                raise
            energy = setup.energy

        return mu, th, chi, phi, nu, tth, energy, setup.distance

    @safeload
    def read_device(self, setup: Setup, device_name: str, **kwargs) -> np.ndarray:
        """
        Extract the scanned device positions/values at BM02 beamline.

        :param setup: an instance of the class Setup
        :param device_name: name of the scanned device
        :return: the positions/values of the device as a numpy 1D array
        """
        file = kwargs.get("file")  # this kwarg is provided by @safeload_static
        if file is None:
            raise ValueError("file should be the opened file, not None")
        if setup.logfile is None:
            raise ValueError("logfile undefined")
        scan_number = setup.logfile.scan_number
        if not isinstance(scan_number, int):
            raise TypeError(
                "scan_number should be an integer, got " f"{type(scan_number)}"
            )

        labels = file[str(scan_number) + ".1"].labels  # motor scanned
        labels_data = file[str(scan_number) + ".1"].data  # motor scanned
        self.logger.info(f"Trying to load values for {device_name}...")
        try:
            device_values = list(labels_data[labels.index(device_name), :])
            self.logger.info(f"{device_name} found!")
        except ValueError:  # device not in the list
            self.logger.info(f"no device {device_name} in the logfile")
            device_values = []
        return np.asarray(device_values)

    def read_monitor(self, setup: Setup, **kwargs: dict[str, Any]) -> np.ndarray:
        """
        Load the default monitor for a dataset measured at BM02.

        :param setup: an instance of the class Setup
        :return: the default monitor values
        """
        if setup.logfile is None:
            raise ValueError("logfile undefined")
        scan_number = setup.logfile.scan_number
        if not isinstance(scan_number, int):
            raise TypeError(
                "scan_number should be an integer, got " f"{type(scan_number)}"
            )
        if setup.actuators is not None:
            monitor_name = setup.actuators.get("monitor", "d0_cps")
        else:
            monitor_name = "d0_cps"
        monitor: np.ndarray = self.read_device(
            setup=setup, scan_number=scan_number, device_name=monitor_name
        )
        return monitor


class LoaderID01BLISS(Loader):
    """Loader for ESRF ID01 beamline after the deployement of BLISS."""

    def create_logfile(
        self,
        datadir: str,
        name: str,
        root_folder: str,
        scan_number: int,
        filename: str | None = None,
        template_imagefile: str | None = None,
    ) -> ContextFile:
        """
        Create the logfile, which is the h5 file for ID01BLISS.

        :param datadir: str, the data directory
        :param name: str, the name of the beamline, e.g. 'ID01BLISS'
        :param root_folder: str, the root directory of the experiment
        :param scan_number: the scan number to load
        :param filename: not used at ID01BLISS
        :param template_imagefile: str, template for data file name,
         e.g. 'ihhc3715_sample5.h5'
        :return: an instance of a context manager ContextFile
        """
        valid.valid_container(datadir, container_types=str, name="datadir")
        if not os.path.isdir(datadir):
            raise ValueError(f"The directory {datadir} does not exist")
        valid.valid_container(
            template_imagefile, container_types=str, name="template_imagefile"
        )
        valid.valid_item(
            scan_number, allowed_types=int, min_included=0, name="scan_number"
        )
        filename = util.find_file(
            filename=template_imagefile, default_folder=datadir, logger=self.logger
        )
        return ContextFile(
            filename=filename, open_func=h5py.File, scan_number=scan_number
        )

    @staticmethod
    def init_paths(root_folder, sample_name, scan_number, template_imagefile, **kwargs):
        """
        Initialize paths used for data processing and logging at ID01 BLISS.

        :param root_folder: folder of the experiment, where all scans are stored
        :param sample_name: string in front of the scan number in the data folder
         name.
        :param scan_number: int, the scan number
        :param template_imagefile: template for the data files, e.g. 'S%d.h5'
        :return: a tuple of strings:

         - homedir: the path of the scan folder
         - default_dirname: the name of the folder containing images / raw data
         - specfile: not used at ID01BLISS
         - template_imagefile: the template for data/image file names

        """
        homedir = root_folder
        default_dirname = ""
        return homedir, default_dirname, None, template_imagefile

    @safeload
    def load_data(
        self,
        setup: Setup,
        flatfield: np.ndarray | None = None,
        hotpixels: np.ndarray | None = None,
        background: np.ndarray | None = None,
        normalize: str = "skip",
        bin_during_loading: bool = False,
        debugging: bool = False,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """
        Load ID01 BLISS data, apply filters and concatenate it for phasing.

        :param setup: an instance of the class Setup
        :param flatfield: the 2D flatfield array
        :param hotpixels: the 2D hotpixels array
        :param background: the 2D background array to subtract to the data
        :param normalize: 'monitor' to return the default monitor values, 'sum_roi' to
         return a monitor based on the integrated intensity in the region of interest
         defined by detector.sum_roi, 'skip' to do nothing
        :param bin_during_loading: if True, the data will be binned in the detector
         frame while loading. It saves a lot of memory space for large 2D detectors.
        :param debugging: set to True to see plots
        :return:
         - the 3D data array in the detector frame
         - the 2D mask array
         - the monitor values for normalization
         - the detector region of interest used for loading the data

        """
        file = kwargs.get("file")  # this kwarg is provided by @safeload
        if file is None:
            raise ValueError("file should be the opened file, not None")
        if setup.logfile is None:
            raise ValueError("logfile undefined")
        scan_number = setup.logfile.scan_number
        if scan_number is None:
            raise ValueError("'scan_number' parameter required")

        sample_name = setup.detector.sample_name
        if sample_name is None:
            raise ValueError("'sample_name' parameter required")

        key_path = (
            (f"{sample_name}_" if sample_name else "")
            + str(scan_number)
            + ".1/measurement/"
        )
        if setup.detector_name == "Maxipix":
            try:
                raw_data = file[key_path + "mpx1x4"]
            except KeyError:
                self.logger.info("Looking for 'mpxgaas' key")
                try:
                    raw_data = file[key_path + "mpxgaas"]
                except KeyError:
                    raise KeyError("No detector key found")
        elif setup.detector_name == "Eiger2M":
            raw_data = file[key_path + "eiger2M"]
        else:
            raise NotImplementedError(
                f"Unknown detector '{setup.detector_name}' for beamline ID01BLISS"
            )

        # find the number of images
        nb_img = raw_data.shape[0]

        data, mask2d, monitor, loading_roi = self.init_data_mask(
            detector=setup.detector,
            setup=setup,
            normalize=normalize,
            nb_frames=nb_img,
            bin_during_loading=bin_during_loading,
            scan_number=scan_number,
        )

        # loop over frames, mask the detector and normalize / bin
        for idx in range(nb_img):
            data[idx, :, :], mask2d, monitor[idx] = load_frame(
                frame=raw_data[idx, :, :],
                mask2d=mask2d,
                monitor=monitor[idx],
                frames_per_point=1,
                detector=setup.detector,
                loading_roi=loading_roi,
                flatfield=flatfield,
                background=background,
                hotpixels=hotpixels,
                normalize=normalize,
                bin_during_loading=bin_during_loading,
                debugging=debugging,
            )
        return data, mask2d, monitor, loading_roi

    @safeload
    def motor_positions(
        self, setup: Setup, **kwargs
    ) -> tuple[float | list | np.ndarray, ...]:
        """
        Load the scan data and extract motor positions.

        :param setup: an instance of the class Setup
        :return: (mu, eta, phi, nu, delta, energy) values
        """
        # load and check kwargs
        file = kwargs.get("file")  # this kwarg is provided by @safeload
        if file is None:
            raise ValueError("file should be the opened file, not None")
        if setup.logfile is None:
            raise ValueError("logfile undefined")
        scan_number = setup.logfile.scan_number
        if scan_number is None:
            raise ValueError("'scan_number' parameter required")

        sample_name = setup.detector.sample_name
        if sample_name is None:
            raise ValueError("'sample_name' parameter required")

        # load positioners
        positioners = file[
            (f"{sample_name}_" if sample_name else "")
            + str(scan_number)
            + ".1/instrument/positioners"
        ]
        if not setup.custom_scan:
            try:
                mu = util.cast(positioners["mu"][()], target_type=float)
            except KeyError:
                self.logger.info(
                    "mu not found in the logfile, use the default value of 0."
                )
                mu = 0.0

            nu = util.cast(positioners["nu"][()], target_type=float)
            delta = util.cast(positioners["delta"][()], target_type=float)
            eta = util.cast(positioners["eta"][()], target_type=float)
            phi = util.cast(positioners["phi"][()], target_type=float)

            # for now, return the setup.energy
            energy = setup.energy

        else:  # manually defined custom scan
            mu = setup.custom_motors["mu"]
            eta = setup.custom_motors["eta"]
            phi = setup.custom_motors["phi"]
            delta = setup.custom_motors["delta"]
            nu = setup.custom_motors["nu"]
            energy = setup.energy

        # detector_distance = self.retrieve_distance(setup=setup) or setup.distance
        detector_distance = setup.distance
        return mu, eta, phi, nu, delta, energy, detector_distance

    @safeload
    def read_device(self, setup: Setup, device_name: str, **kwargs) -> np.ndarray:
        """
        Extract the scanned device values at ID01 BLISS beamline.

        :param setup: an instance of the class Setup
        :param device_name: name of the scanned device
        :return: the positions/values of the device as a numpy 1D array
        """
        file = kwargs.get("file")  # this kwarg is provided by @safeload_static
        if file is None:
            raise ValueError("file should be the opened file, not None")
        if setup.logfile is None:
            raise ValueError("logfile undefined")
        scan_number = setup.logfile.scan_number
        if scan_number is None:
            raise ValueError("'scan_number' parameter required")

        # load positioners
        positioners = file[
            (f"{setup.detector.sample_name}_" if setup.detector.sample_name else "")
            + str(scan_number)
            + ".1/measurement"
        ]
        try:
            device_values = util.cast(positioners[device_name][()], target_type=float)
        except KeyError:
            self.logger.info(f"No device {device_name} found in the logfile")
            device_values = []
        return np.asarray(device_values)

    def read_monitor(self, setup: Setup, **kwargs: dict[str, Any]) -> np.ndarray:
        """
        Load the default monitor for a dataset measured at ID01 BLISS.

        :param setup: an instance of the class Setup
        :return: the default monitor values
        """
        if setup.logfile is None:
            raise ValueError("logfile undefined")
        scan_number = setup.logfile.scan_number
        if scan_number is None:
            raise ValueError("'scan_number' parameter required")
        if setup.actuators is not None:
            monitor_name = setup.actuators.get("monitor", "imon")
        else:
            monitor_name = "imon"
        monitor: np.ndarray = self.read_device(
            setup=setup, scan_number=scan_number, device_name=monitor_name
        )
        return monitor


class LoaderID27(Loader):
    """Loader for ESRF ID27 beamline."""

    def create_logfile(
        self,
        datadir: str,
        name: str,
        root_folder: str,
        scan_number: int,
        filename: str | None = None,
        template_imagefile: str | None = None,
    ) -> ContextFile:
        """
        Create the logfile, which is the h5 file for ID27.

        :param datadir: str, the data directory
        :param name: str, the name of the beamline, e.g. 'ID27'
        :param root_folder: str, the root directory of the experiment
        :param scan_number: the scan number to load
        :param filename: str, absolute path to the spec/fio/alias file when it exists
        :param template_imagefile: str, template for data file name,
         e.g. 'Ptx7_0007.h5'
        :return: an instance of a context manager ContextFile
        """
        valid.valid_container(datadir, container_types=str, name="datadir")
        if not os.path.isdir(datadir):
            raise ValueError(f"The directory {datadir} does not exist")
        valid.valid_container(
            template_imagefile, container_types=str, name="template_imagefile"
        )
        valid.valid_item(
            scan_number, allowed_types=int, min_included=0, name="scan_number"
        )
        filename = util.find_file(
            filename=template_imagefile, default_folder=datadir, logger=self.logger
        )
        return ContextFile(
            filename=filename, open_func=h5py.File, scan_number=scan_number
        )

    @staticmethod
    def init_paths(root_folder, sample_name, scan_number, template_imagefile, **kwargs):
        """
        Initialize paths used for data processing and logging at ID27.

        :param root_folder: folder of the experiment, where all scans are stored
        :param sample_name: string in front of the scan number in the data folder
         name.
        :param scan_number: int, the scan number
        :param template_imagefile: template for the data files, e.g. 'S%d.h5'
        :return: a tuple of strings:

         - homedir: the path of the scan folder
         - default_dirname: the name of the folder containing images / raw data
         - specfile: not used at ID27
         - template_imagefile: the template for data/image file names

        """
        homedir = root_folder
        default_dirname = ""
        return homedir, default_dirname, None, template_imagefile

    @safeload
    def load_data(
        self,
        setup: Setup,
        flatfield: np.ndarray | None = None,
        hotpixels: np.ndarray | None = None,
        background: np.ndarray | None = None,
        normalize: str = "skip",
        bin_during_loading: bool = False,
        debugging: bool = False,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """
        Load ID27 data, apply filters and concatenate it for phasing.

        :param setup: an instance of the class Setup
        :param flatfield: the 2D flatfield array
        :param hotpixels: the 2D hotpixels array
        :param background: the 2D background array to subtract to the data
        :param normalize: 'monitor' to return the default monitor values, 'sum_roi' to
         return a monitor based on the integrated intensity in the region of interest
         defined by detector.sum_roi, 'skip' to do nothing
        :param bin_during_loading: if True, the data will be binned in the detector
         frame while loading. It saves a lot of memory space for large 2D detectors.
        :param debugging: set to True to see plots
        :return:
         - the 3D data array in the detector frame
         - the 2D mask array
         - the monitor values for normalization
         - the detector region of interest used for loading the data

        """
        file = kwargs.get("file")  # this kwarg is provided by @safeload
        if file is None:
            raise ValueError("file should be the opened file, not None")
        if setup.logfile is None:
            raise ValueError("logfile undefined")
        scan_number = setup.logfile.scan_number
        if scan_number is None:
            raise ValueError("'scan_number' parameter required")

        sample_name = setup.detector.sample_name
        if sample_name is None:
            raise ValueError("'sample_name' parameter required")

        key_path = str(scan_number) + ".1/measurement/"
        raw_data = file[key_path + "eiger"]  # Dataset, does not actually the data

        # find the number of images
        nb_img = raw_data.shape[0]

        data, mask2d, monitor, loading_roi = self.init_data_mask(
            detector=setup.detector,
            setup=setup,
            normalize=normalize,
            nb_frames=nb_img,
            bin_during_loading=bin_during_loading,
            scan_number=scan_number,
        )

        # loop over frames, mask the detector and normalize / bin
        for idx in range(nb_img):
            data[idx, :, :], mask2d, monitor[idx] = load_frame(
                frame=raw_data[idx, :, :],
                mask2d=mask2d,
                monitor=monitor[idx],
                frames_per_point=1,
                detector=setup.detector,
                loading_roi=loading_roi,
                flatfield=flatfield,
                background=background,
                hotpixels=hotpixels,
                normalize=normalize,
                bin_during_loading=bin_during_loading,
                debugging=debugging,
            )
        return data, mask2d, monitor, loading_roi

    @safeload
    def motor_positions(
        self, setup: Setup, **kwargs
    ) -> tuple[float | list | np.ndarray, ...]:
        """
        Load the scan data and extract motor positions.

        :param setup: an instance of the class Setup
        :return: (nath, energy) values
        """
        # load and check kwargs
        file = kwargs.get("file")  # this kwarg is provided by @safeload
        if file is None:
            raise ValueError("file should be the opened file, not None")
        if setup.logfile is None:
            raise ValueError("logfile undefined")
        scan_number = setup.logfile.scan_number
        if scan_number is None:
            raise ValueError("'scan_number' parameter required")

        sample_name = setup.detector.sample_name
        if sample_name is None:
            raise ValueError("'sample_name' parameter required")

        # load positioners
        positioners = file[str(scan_number) + ".1/instrument/positioners"]
        if not setup.custom_scan:
            nath = util.cast(positioners["nath"][()], target_type=float)
            eigx = util.cast(positioners["eigx"][()], target_type=float)
            eigy = util.cast(positioners["eigy"][()], target_type=float)
            eigz = util.cast(positioners["eigz"][()], target_type=float)
            # for now, return the setup.energy
            energy = setup.energy

        else:  # manually defined custom scan
            nath = setup.custom_motors["nath"]
            eigx = setup.custom_motors["eigx"]
            eigy = setup.custom_motors["eigy"]
            eigz = setup.custom_motors["eigz"]
            energy = setup.energy

        # detector_distance = self.retrieve_distance(setup=setup) or setup.distance
        detector_distance = setup.distance
        return nath, eigx, eigz, eigy, energy, detector_distance

    @safeload
    def read_device(self, setup: Setup, device_name: str, **kwargs) -> np.ndarray:
        """
        Extract the scanned device values at ID27 beamline.

        :param setup: an instance of the class Setup
        :param device_name: name of the scanned device
        :return: the positions/values of the device as a numpy 1D array
        """
        file = kwargs.get("file")  # this kwarg is provided by @safeload_static
        if file is None:
            raise ValueError("file should be the opened file, not None")
        if setup.logfile is None:
            raise ValueError("logfile undefined")
        scan_number = setup.logfile.scan_number
        if scan_number is None:
            raise ValueError("'scan_number' parameter required")

        # load positioners
        positioners = file[str(scan_number) + ".1/measurement"]
        try:
            device_values = util.cast(positioners[device_name][()], target_type=float)
        except KeyError:
            self.logger.info(f"No device {device_name} found in the logfile")
            device_values = []
        return np.asarray(device_values)

    def read_monitor(self, setup: Setup, **kwargs: dict[str, Any]) -> np.ndarray:
        """
        Load the default monitor for a dataset measured at ID27.

        :param setup: an instance of the class Setup
        :return: the default monitor values
        """
        if setup.logfile is None:
            raise ValueError("logfile undefined")
        scan_number = setup.logfile.scan_number
        monitor_name: str | None = None
        if scan_number is None:
            raise ValueError("'scan_number' parameter required")
        if setup.actuators is not None:
            monitor_name = setup.actuators.get("monitor")
        if monitor_name is None:
            self.logger.info(
                "'monitor_name' is None, no default monitor defined for ID27."
            )
            return np.asarray([])
        monitor: np.ndarray = self.read_device(
            setup=setup, scan_number=scan_number, device_name=monitor_name
        )
        return monitor


class LoaderSIXS(Loader):
    """Loader for SOLEIL SIXS beamline."""

    def create_logfile(
        self,
        datadir: str,
        name: str,
        root_folder: str,
        scan_number: int,
        filename: str | None = None,
        template_imagefile: str | None = None,
    ) -> ContextFile:
        """
        Create the logfile, which is the data itself for SIXS.

        :param datadir: str, the data directory
        :param name: str, the name of the beamline, e.g. 'SIXS_2019'
        :param root_folder: str, the root directory of the experiment
        :param scan_number: the scan number to load
        :param filename: str, absolute path of 'alias_dict.txt'
        :param template_imagefile: str, template for data file name:

          - SIXS_2018: 'align.spec_ascan_mu_%05d.nxs'
          - SIXS_2019: 'spare_ascan_mu_%05d.nxs'

        :return: an instance of a context manager ContextFile
        """
        if datadir is None:
            raise ValueError("'datadir' parameter required for SIXS")
        if not os.path.isdir(datadir):
            raise ValueError(f"The directory {datadir} does not exist")
        if template_imagefile is None or not isinstance(template_imagefile, str):
            raise TypeError("'template_imagefile' should be a string")
        if filename is None or not isinstance(filename, str):
            raise TypeError("'filename' should be a string")
        valid.valid_item(
            scan_number, allowed_types=int, min_included=0, name="scan_number"
        )

        shortname = template_imagefile % scan_number
        if name == "SIXS_2018":
            # no specfile, load directly the dataset
            import bcdi.preprocessing.nxsReady as nxsReady

            return ContextFile(
                filename=filename,
                open_func=nxsReady.DataSet,
                longname=datadir + shortname,
                shortname=shortname,
                scan_number=scan_number,
            )
        if name == "SIXS_2019":
            # no specfile, load directly the dataset
            import bcdi.preprocessing.ReadNxs3 as ReadNxs3

            return ContextFile(
                filename=filename,
                open_func=ReadNxs3.DataSet,
                shortname=shortname,
                directory=datadir,
                scan_number=scan_number,
            )
        raise NotImplementedError(f"{name} is not implemented")

    @staticmethod
    def init_paths(root_folder, sample_name, scan_number, template_imagefile, **kwargs):
        """
        Initialize paths used for data processing and logging at SIXS.

        :param root_folder: folder of the experiment, where all scans are stored
        :param sample_name: string in front of the scan number in the data folder
         name.
        :param scan_number: int, the scan number
        :param template_imagefile: template for the data files, e.g.
         'align.spec_ascan_mu_%05d.nxs' (SIXS_2018), 'spare_ascan_mu_%05d.nxs'
         (SIXS_2019).
        :param kwargs:
         - 'specfile_name': None or full path of the alias dictionnary (e.g.
           root_folder+'alias_dict_2019.txt')

        :return: a tuple of strings:

         - homedir: the path of the scan folder
         - default_dirname: the name of the folder containing images / raw data
         - specfile: the name of the specfile if it exists
         - template_imagefile: the template for data/image file names

        """
        specfile_name = kwargs.get("specfile_name")

        homedir = root_folder + sample_name + str(scan_number) + "/"
        default_dirname = "data/"

        if specfile_name is None:
            # default to the alias dictionnary located within the package
            specfile = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    os.pardir,
                    "preprocessing/alias_dict_2021.txt",
                )
            )
        else:
            specfile = specfile_name

        return homedir, default_dirname, specfile, template_imagefile

    @safeload
    def load_data(
        self,
        setup: Setup,
        flatfield: np.ndarray | None = None,
        hotpixels: np.ndarray | None = None,
        background: np.ndarray | None = None,
        normalize: str = "skip",
        bin_during_loading: bool = False,
        debugging: bool = False,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """
        Load data, apply filters and concatenate it for phasing at SIXS.

        :param setup: an instance of the class Setup
        :param flatfield: the 2D flatfield array
        :param hotpixels: the 2D hotpixels array
        :param background: the 2D background array to subtract to the data
        :param normalize: 'monitor' to return the default monitor values, 'sum_roi' to
         return a monitor based on the integrated intensity in the region of interest
         defined by detector.sum_roi, 'skip' to do nothing
        :param bin_during_loading: if True, the data will be binned in the detector
         frame while loading. It saves a lot of memory space for large 2D detectors.
        :param debugging: set to True to see plots
        :return:

         - the 3D data array in the detector frame
         - the 2D mask array
         - the monitor values for normalization
         - the detector region of interest used for loading the data

        """
        file = kwargs.get("file")  # this kwarg is provided by @safeload
        if file is None:
            raise ValueError("file should be the opened file, not None")

        # load the data
        if setup.custom_scan:
            raise NotImplementedError("custom scan not implemented for SixS")
        if setup.detector.name == "Merlin":
            tmp_data = file.merlin[:]
        elif setup.detector.name == "MerlinSixS":
            tmp_data = file.merlin[:]
        else:  # Maxipix
            if setup.beamline == "SIXS_2018":  # type: ignore
                tmp_data = file.mfilm[:]
            else:
                try:
                    tmp_data = file.mpx_image[:]
                except AttributeError:
                    try:
                        tmp_data = file.maxpix[:]
                    except AttributeError:
                        # the alias dictionnary was probably not provided
                        tmp_data = file.image[:]

        # find the number of images
        nb_img = tmp_data.shape[0]

        # initialize arrays and loading ROI
        data, mask2d, monitor, loading_roi = self.init_data_mask(
            detector=setup.detector,
            setup=setup,
            normalize=normalize,
            nb_frames=nb_img,
            bin_during_loading=bin_during_loading,
        )

        # loop over frames, mask the detector and normalize / bin
        for idx in range(nb_img):
            data[idx, :, :], mask2d, monitor[idx] = load_frame(
                frame=tmp_data[idx, :, :],
                mask2d=mask2d,
                monitor=monitor[idx],
                frames_per_point=1,
                detector=setup.detector,
                loading_roi=loading_roi,
                flatfield=flatfield,
                background=background,
                hotpixels=hotpixels,
                normalize=normalize,
                bin_during_loading=bin_during_loading,
                debugging=debugging,
            )
        return data, mask2d, monitor, loading_roi

    @safeload
    def motor_positions(
        self, setup: Setup, **kwargs
    ) -> tuple[float | list | np.ndarray, ...]:
        """
        Load the scan data and extract motor positions at SIXS.

        :param setup: an instance of the class Setup
        :return: (beta, mu, gamma, delta, energy) values
        """
        file = kwargs.get("file")  # this kwarg is provided by @safeload
        if file is None:
            raise ValueError("file should be the opened file, not None")

        if not setup.custom_scan:
            mu = file.mu[:]  # scanned
            delta = file.delta[0]  # not scanned
            gamma = file.gamma[0]  # not scanned
            try:
                beta = file.basepitch[0]  # not scanned
            except AttributeError:  # data recorder changed after 11/03/2019
                try:
                    beta = file.beta[0]  # not scanned
                except AttributeError:
                    # the alias dictionnary was probably not provided
                    beta = 0

            # remove user-defined sample offsets (sample: beta, mu)
            beta = beta - self.sample_offsets[0]
            mu = mu - self.sample_offsets[1]

        else:  # manually defined custom scan
            beta = setup.custom_motors["beta"]
            delta = setup.custom_motors["delta"]
            gamma = setup.custom_motors["gamma"]
            mu = setup.custom_motors["mu"]
        return beta, mu, gamma, delta, setup.energy, setup.distance

    @safeload
    def read_device(self, setup: Setup, device_name: str, **kwargs) -> np.ndarray:
        """
        Extract the scanned device positions/values at SIXS beamline.

        :param setup: an instance of the class Setup
        :param device_name: name of the scanned device
        :return: the positions/values of the device as a numpy 1D array
        """
        file = kwargs.get("file")  # this kwarg is provided by @safeload_static
        if file is None:
            raise ValueError("file should be the opened file, not None")

        self.logger.info(f"Trying to load values for {device_name}...")
        try:
            device_values = getattr(file, device_name)
            self.logger.info(f"{device_name} found!")
        except AttributeError:
            self.logger.info(f"No device {device_name} in the logfile")
            device_values = []
        return np.asarray(device_values)

    def read_monitor(self, setup: Setup, **kwargs: dict[str, Any]) -> np.ndarray:
        """
        Load the default monitor for a dataset measured at SIXS.

        :param setup: an instance of the class Setup
        :return: the default monitor values
        """
        if setup.beamline == "SIXS_2018":  # type: ignore
            monitor: np.ndarray = self.read_device(setup=setup, device_name="imon1")
        else:  # "SIXS_2019"
            monitor = self.read_device(setup=setup, device_name="imon0")
        if len(monitor) == 0:
            # the alias dictionnary was probably not provided
            monitor = self.read_device(setup=setup, device_name="intensity")
        return monitor


class Loader34ID(Loader):
    """Loader for APS 34ID-C beamline."""

    motor_table = {
        "theta": "Theta",
        "chi": "Chi",
        "phi": "Phi",
        "gamma": "Gamma",
        "delta": "Delta",
        "energy": "Energy",
        "detector_distance": "camdist",
    }

    def create_logfile(
        self,
        datadir: str,
        name: str,
        root_folder: str,
        scan_number: int,
        filename: str | None = None,
        template_imagefile: str | None = None,
    ) -> ContextFile:
        """
        Create the logfile, which is the spec file for 34ID-C.

        :param datadir: str, the data directory
        :param name: str, the name of the beamline, e.g. '34ID'
        :param root_folder: str, the root directory of the experiment
        :param scan_number: the scan number to load
        :param filename: str, absolute path to the spec/fio/alias file when it exists
        :param template_imagefile: str, template for the data file name
        :return: an instance of a context manager ContextFile
        """
        valid.valid_container(
            root_folder,
            container_types=str,
            min_length=1,
            name="root_folder",
        )
        if not os.path.isdir(root_folder):
            raise ValueError(f"The directory {root_folder} does not exist")
        valid.valid_container(
            filename,
            container_types=str,
            min_length=1,
            name="filename",
        )
        valid.valid_item(
            scan_number, allowed_types=int, min_included=1, name="scan_number"
        )
        path = util.find_file(
            filename=filename, default_folder=root_folder, logger=self.logger
        )
        return ContextFile(filename=path, open_func=SpecFile, scan_number=scan_number)

    @staticmethod
    def init_paths(root_folder, sample_name, scan_number, template_imagefile, **kwargs):
        """
        Initialize paths used for data processing and logging at 34ID-C.

        :param root_folder: folder of the experiment, where all scans are stored
        :param sample_name: string in front of the scan number in the data folder
         name.
        :param scan_number: int, the scan number
        :param template_imagefile: template for the data files, e.g.
         'Sample%dC_ES_data_51_256_256.npz'.
        :param kwargs:
         - 'specfile_name': name of the spec file without '.spec'

        :return: a tuple of strings:

         - homedir: the path of the scan folder
         - default_dirname: the name of the folder containing images / raw data
         - specfile: the name of the specfile if it exists
         - template_imagefile: the template for data/image file names

        """
        specfile_name = kwargs.get("specfile_name")

        homedir = root_folder + sample_name + str(scan_number) + "/"
        default_dirname = "data/"
        return homedir, default_dirname, specfile_name, template_imagefile

    @safeload
    def load_data(
        self,
        setup: Setup,
        flatfield: np.ndarray | None = None,
        hotpixels: np.ndarray | None = None,
        background: np.ndarray | None = None,
        normalize: str = "skip",
        bin_during_loading: bool = False,
        debugging: bool = False,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """
        Load 34ID-C data including detector/background corrections.

        :param setup: an instance of the class Setup
        :param flatfield: the 2D flatfield array
        :param hotpixels: the 2D hotpixels array
        :param background: the 2D background array to subtract to the data
        :param normalize: 'monitor' to return the default monitor values, 'sum_roi' to
         return a monitor based on the integrated intensity in the region of interest
         defined by detector.sum_roi, 'skip' to do nothing
        :param bin_during_loading: if True, the data will be binned in the detector
         frame while loading. It saves a lot of memory space for large 2D detectors.
        :param debugging: set to True to see plots
        """
        file = kwargs.get("file")  # this kwarg is provided by @safeload
        if file is None:
            raise ValueError("file should be the opened file, not None")
        if setup.logfile is None:
            raise ValueError("logfile undefined")
        scan_number = setup.logfile.scan_number
        if scan_number is None:
            raise ValueError("'scan_number' parameter required")
        if setup.detector.template_imagefile is None:
            raise ValueError("'template_imagefile' must be defined to load the images.")
        ccdfiletmp = os.path.join(
            setup.detector.datadir, setup.detector.template_imagefile
        )
        data_stack = None
        if not setup.custom_scan:
            # create the template for the image files
            labels = file[str(scan_number) + ".1"].labels  # motor scanned
            labels_data = file[str(scan_number) + ".1"].data  # motor scanned

            # find the number of images
            try:
                nb_img = len(labels_data[labels.index("Monitor"), :])
            except ValueError:
                try:
                    self.logger.info("'Monitor' not in the list, trying 'Detector'")
                    nb_img = len(labels_data[labels.index("Detector"), :])
                except ValueError:
                    raise ValueError(
                        "'Detector' not in the list, can't retrieve "
                        "the number of frames",
                    )
        else:
            # create the template for the image files
            if len(setup.custom_images) == 0:
                raise ValueError("No image number provided in 'custom_images'")

            if len(setup.custom_images) > 1:
                nb_img = len(setup.custom_images)
            else:  # the data is stacked into a single file
                with np.load(ccdfiletmp % setup.custom_images[0]) as npzfile:
                    data_stack = npzfile[list(npzfile.files)[0]]
                nb_img = data_stack.shape[0]

        data, mask2d, monitor, loading_roi = self.init_data_mask(
            detector=setup.detector,
            setup=setup,
            normalize=normalize,
            nb_frames=nb_img,
            bin_during_loading=bin_during_loading,
            scan_number=scan_number,
        )

        # loop over frames, mask the detector and normalize / bin
        for idx in range(nb_img):
            if data_stack is not None:
                # custom scan with a stacked data loaded
                ccdraw = data_stack[idx, :, :]
            else:
                if setup.custom_scan:
                    # custom scan with one file per frame
                    i = int(setup.custom_images[idx])
                else:
                    i = idx
                try:
                    ccdraw = util.image_to_ndarray(
                        filename=ccdfiletmp % i,
                        convert_grey=True,
                    )
                except TypeError:
                    raise ValueError(
                        "Error in string formatting of the image filename, "
                        "check the value of 'template_imagefile'"
                    )

            data[idx, :, :], mask2d, monitor[idx] = load_frame(
                frame=ccdraw,
                mask2d=mask2d,
                monitor=monitor[idx],
                frames_per_point=1,
                detector=setup.detector,
                loading_roi=loading_roi,
                flatfield=flatfield,
                background=background,
                hotpixels=hotpixels,
                normalize=normalize,
                bin_during_loading=bin_during_loading,
                debugging=debugging,
            )
        return data, mask2d, monitor, loading_roi

    @safeload
    def motor_positions(
        self, setup: Setup, **kwargs
    ) -> tuple[float | list | np.ndarray, ...]:
        """
        Load the scan data and extract motor positions.

        :param setup: an instance of the class Setup
        :return: (theta, phi, delta, gamma, energy) values
        """
        # load and check kwargs
        file = kwargs.get("file")  # this kwarg is provided by @safeload
        if file is None:
            raise ValueError("file should be the opened file, not None")
        if setup.logfile is None:
            raise ValueError("logfile undefined")
        scan_number = setup.logfile.scan_number

        if not setup.custom_scan:
            motor_names = file[str(scan_number) + ".1"].motor_names
            # positioners
            motor_values = file[str(scan_number) + ".1"].motor_positions
            # positioners
            labels = file[str(scan_number) + ".1"].labels  # motor scanned
            labels_data = file[str(scan_number) + ".1"].data  # motor scanned

            if self.motor_table["theta"] in labels:  # scanned
                theta = labels_data[labels.index(self.motor_table["theta"]), :]
            else:  # positioner
                theta = motor_values[motor_names.index(self.motor_table["theta"])]

            if self.motor_table["chi"] in labels:  # scanned
                chi = labels_data[labels.index(self.motor_table["chi"]), :]
            else:  # positioner
                chi = motor_values[motor_names.index(self.motor_table["chi"])]

            if self.motor_table["phi"] in labels:  # scanned
                phi = labels_data[labels.index(self.motor_table["phi"]), :]
            else:  # positioner
                phi = motor_values[motor_names.index(self.motor_table["phi"])]

            if self.motor_table["delta"] in labels:  # scanned
                delta = labels_data[labels.index(self.motor_table["delta"]), :]
            else:  # positioner
                delta = motor_values[motor_names.index(self.motor_table["delta"])]

            if self.motor_table["gamma"] in labels:  # scanned
                gamma = labels_data[labels.index(self.motor_table["gamma"]), :]
            else:  # positioner
                gamma = motor_values[motor_names.index(self.motor_table["gamma"])]

            if self.motor_table["energy"] in labels:  # scanned
                energy = labels_data[labels.index(self.motor_table["energy"]), :]
                # energy scanned, override the user-defined energy
            else:  # positioner
                energy = motor_values[motor_names.index(self.motor_table["energy"])]

            energy = energy * 1000.0  # switch to eV
            detector_distance = (
                motor_values[motor_names.index(self.motor_table["detector_distance"])]
                / 1000
            )  # convert to m

            # remove user-defined sample offsets (sample: mu, eta, phi)
            theta = theta - self.sample_offsets[0]
            chi = chi - self.sample_offsets[1]
            phi = phi - self.sample_offsets[2]

        else:  # manually defined custom scan
            theta = setup.custom_motors["theta"]
            chi = setup.custom_motors["chi"]
            phi = setup.custom_motors["phi"]
            gamma = setup.custom_motors["gamma"]
            delta = setup.custom_motors["delta"]
            detector_distance = setup.distance
            energy = setup.energy

        return theta, chi, phi, delta, gamma, energy, detector_distance

    @safeload
    def read_device(self, setup: Setup, device_name: str, **kwargs) -> np.ndarray:
        """
        Extract the scanned device positions/values at 34ID-C beamline.

        :param setup: an instance of the class Setup
        :param device_name: name of the scanned device
        :return: the positions/values of the device as a numpy 1D array
        """
        file = kwargs.get("file")  # this kwarg is provided by @safeload_static
        if file is None:
            raise ValueError("file should be the opened file, not None")
        if setup.logfile is None:
            raise ValueError("logfile undefined")
        scan_number = setup.logfile.scan_number
        if scan_number is None:
            raise ValueError("'scan_number' parameter required")

        labels = file[str(scan_number) + ".1"].labels  # motor scanned
        labels_data = file[str(scan_number) + ".1"].data  # motor scanned
        self.logger.info(f"Trying to load values for {device_name}...")
        try:
            device_values = list(labels_data[labels.index(device_name), :])
            self.logger.info(f"{device_name} found!")
        except ValueError:  # device not in the list
            self.logger.info(f"no device {device_name} in the logfile")
            device_values = []
        return np.asarray(device_values)

    def read_monitor(self, setup: Setup, **kwargs: dict[str, Any]) -> np.ndarray:
        """
        Load the default monitor for a dataset measured at 34ID-C.

        :param setup: an instance of the class Setup
        :param kwargs:
         - 'scan_number': int, the scan number to load

        :return: the default monitor values
        """
        scan_number = kwargs.get("scan_number")
        if scan_number is None:
            raise ValueError("'scan_number' parameter required")
        if setup.actuators is not None:
            monitor_name = setup.actuators.get("monitor", "Monitor")
        else:
            monitor_name = "Monitor"
        monitor: np.ndarray = self.read_device(
            setup=setup, scan_number=scan_number, device_name=monitor_name
        )
        return monitor


class LoaderP10(Loader):
    """Loader for PETRAIII P10 beamline."""

    def create_logfile(
        self,
        datadir: str,
        name: str,
        root_folder: str,
        scan_number: int,
        filename: str | None = None,
        template_imagefile: str | None = None,
    ) -> ContextFile:
        """
        Create the logfile, which is the .fio file for P10.

        :param datadir: str, the data directory
        :param name: str, the name of the beamline, e.g. 'P10'
        :param root_folder: str, the root directory of the experiment
        :param scan_number: the scan number to load
        :param filename: str, name of the .fio file or full path of the .fio file
        :param template_imagefile: str, template for the data file name
        :return: an instance of a context manager ContextFile
        """
        valid.valid_container(
            root_folder,
            container_types=str,
            min_length=1,
            name="root_folder",
        )
        if not os.path.isdir(root_folder):
            raise ValueError(f"The directory {root_folder} does not exist")
        valid.valid_container(
            filename,
            container_types=str,
            min_length=1,
            name="filename",
        )
        valid.valid_item(
            scan_number, allowed_types=int, min_included=1, name="scan_number"
        )
        if filename is None or not isinstance(filename, str):
            raise TypeError("'filename' should be a string")
        if os.path.isfile(filename):
            # filename is already the full path to the .fio file
            return ContextFile(
                filename=filename,
                open_func=open,
                scan_number=scan_number,
                mode="r",
                encoding="utf-8",
            )

        self.logger.info(f"Could not find the fio file at: {filename}")

        # try the default path to the .fio file
        path = root_folder + filename + "/" + filename + ".fio"
        self.logger.info(f"Trying to load the fio file at: {path}")
        if not os.path.isfile(path):
            raise ValueError(f"Could not find the fio file at: {path}")
        return ContextFile(filename=path, open_func=open, mode="r", encoding="utf-8")

    @staticmethod
    def init_paths(root_folder, sample_name, scan_number, template_imagefile, **kwargs):
        """
        Initialize paths used for data processing and logging at P10.

        :param root_folder: folder of the experiment, where all scans are stored
        :param sample_name: string in front of the scan number in the data folder
         name.
        :param scan_number: int, the scan number
        :param template_imagefile: template for the data files, e.g. '_master.h5'
        :param kwargs:
         - 'specfile_name': optional, full path of the .fio file

        :return: a tuple of strings:

         - homedir: the path of the scan folder
         - default_dirname: the name of the folder containing images / raw data
         - specfile: the name of the specfile if it exists
         - template_imagefile: the template for data/image file names

        """
        specfile = kwargs.get("specfile_name")
        default_specfile = f"{sample_name}_{scan_number:05d}"
        if specfile is None or not os.path.isfile(specfile):
            # default to the usual position of .fio at P10
            specfile = default_specfile

        homedir = root_folder + default_specfile + "/"
        default_dirname = "e4m/"

        if template_imagefile is not None:
            template_imagefile = default_specfile + template_imagefile
        return homedir, default_dirname, specfile, template_imagefile

    def load_data(
        self,
        setup: Setup,
        flatfield: np.ndarray | None = None,
        hotpixels: np.ndarray | None = None,
        background: np.ndarray | None = None,
        normalize: str = "skip",
        bin_during_loading: bool = False,
        debugging: bool = False,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """
        Load P10 data, apply filters and concatenate it for phasing.

        :param setup: an instance of the class Setup
        :param flatfield: the 2D flatfield array
        :param hotpixels: the 2D hotpixels array
        :param background: the 2D background array to subtract to the data
        :param normalize: 'monitor' to return the default monitor values, 'sum_roi' to
         return a monitor based on the integrated intensity in the region of interest
         defined by detector.sum_roi, 'skip to do nothing'
        :param bin_during_loading: if True, the data will be binned in the detector
         frame while loading. It saves a lot of memory space for large 2D detectors.
        :param debugging: set to True to see plots
        :return:

         - the 3D data array in the detector frame
         - the 2D mask array
         - the monitor values for normalization
         - the detector region of interest used for loading the data

        """
        if setup.detector.template_imagefile is None:
            raise ValueError("'template_imagefile' must be defined to load the images.")
        # template for the master file
        ccdfiletmp = os.path.join(
            setup.detector.datadir, setup.detector.template_imagefile
        )
        is_series = setup.is_series
        if not setup.custom_scan:
            with h5py.File(ccdfiletmp, "r") as h5file:
                # find the number of images
                # (i.e. points, not including series at each point)
                if is_series:
                    nb_img = len(list(h5file["entry/data"]))
                else:
                    idx = 0
                    nb_img = 0
                    while True:
                        data_path = f"data_{idx + 1:06d}"
                        try:
                            nb_img += len(h5file["entry"]["data"][data_path])
                            idx += 1
                        except KeyError:
                            break
            self.logger.info(f"Number of points: {nb_img}")
        else:
            # create the template for the image files
            if len(setup.custom_images) > 0:
                nb_img = len(setup.custom_images)
            else:
                raise ValueError("No image number provided in 'custom_images'")

        # initialize arrays and loading ROI
        data, mask2d, monitor, loading_roi = self.init_data_mask(
            detector=setup.detector,
            setup=setup,
            normalize=normalize,
            nb_frames=nb_img,
            bin_during_loading=bin_during_loading,
        )

        # loop over frames, mask the detector and normalize / bin
        start_index = 0  # offset when not is_series
        for point_idx in range(nb_img):
            idx = 0
            series_data = []
            series_monitor = []
            if setup.custom_scan:
                # custom scan with one file per frame/series of frame,
                # no master file in this case, load directly data files.
                i = int(setup.custom_images[idx])
                ccdfiletmp = (
                    setup.detector.rootdir
                    + setup.detector.sample_name
                    + f"_{i:05d}/e4m/"
                    + setup.detector.sample_name
                    + f"_{i:05d}"
                    + setup.detector.template_file
                )
                data_path = "data_000001"
            else:
                # normal scan, ccdfiletmp points to the master .h5 file
                data_path = f"data_{point_idx + 1:06d}"

            with h5py.File(ccdfiletmp, "r") as h5file:
                while True:
                    try:
                        try:
                            tmp_data = h5file["entry"]["data"][data_path][idx]
                        except OSError:
                            raise OSError("hdf5plugin is not installed")

                        # a single frame from the (eventual) series is loaded
                        ccdraw, mask2d, temp_mon = load_frame(
                            frame=tmp_data,
                            mask2d=mask2d,
                            monitor=monitor[idx],
                            frames_per_point=1,
                            detector=setup.detector,
                            loading_roi=loading_roi,
                            flatfield=flatfield,
                            background=background,
                            hotpixels=hotpixels,
                            normalize=normalize,
                            bin_during_loading=bin_during_loading,
                            debugging=debugging,
                        )
                        series_data.append(ccdraw)
                        series_monitor.append(temp_mon)

                        idx = idx + 1
                    except IndexError:  # reached the end of the series
                        break
                    except ValueError:  # something went wrong
                        break

            if len(series_data) == 0:
                raise ValueError(
                    f"Check the parameter 'is_series', current value {is_series}"
                )
            if is_series:
                data[point_idx, :, :] = np.asarray(series_data).sum(axis=0)
                if normalize == "sum_roi":
                    monitor[point_idx] = np.asarray(series_monitor).sum()
            else:
                tempdata_length = len(series_data)
                data[start_index : start_index + tempdata_length, :, :] = np.asarray(
                    series_data
                )

                if normalize == "sum_roi":
                    monitor[start_index : start_index + tempdata_length] = np.asarray(
                        series_monitor
                    )
                start_index += tempdata_length
                if start_index == nb_img:
                    break
        return data, mask2d, monitor, loading_roi

    @safeload
    def motor_positions(
        self, setup: Setup, **kwargs
    ) -> tuple[float | list | np.ndarray, ...]:
        """
        Load the .fio file from the scan and extract motor positions.

        :param setup: an instance of the class Setup
        :return: (om, phi, chi, mu, gamma, delta, energy) values
        """
        # load and check kwargs
        file = kwargs.get("file")  # this kwarg is provided by @safeload
        if file is None:
            raise ValueError("file should be the opened file, not None")

        if not setup.custom_scan:
            index_om = None
            index_phi = None
            rocking_positions: list[float] = []
            om: float | np.ndarray | None = None
            phi: float | np.ndarray | None = None
            chi = None
            mu = None
            gamma = None
            delta = None
            energy = None

            lines = file.readlines()
            for line in lines:
                this_line = line.strip()
                words = this_line.split()

                if (
                    "Col" in words and "om" in words
                ):  # om scanned, template = ' Col 0 om DOUBLE\n'
                    index_om = int(words[1]) - 1  # python index starts at 0
                if (
                    "om" in words and "=" in words and setup.rocking_angle == "inplane"
                ):  # om is a positioner
                    om = float(words[2])

                if (
                    "Col" in words and "phi" in words
                ):  # phi scanned, template = ' Col 0 phi DOUBLE\n'
                    index_phi = int(words[1]) - 1  # python index starts at 0
                if (
                    "phi" in words
                    and "=" in words
                    and setup.rocking_angle == "outofplane"
                ):  # phi is a positioner
                    phi = float(words[2])

                if (
                    "chi" in words and "=" in words
                ):  # template for positioners: 'chi = 90.0\n'
                    chi = float(words[2])
                if (
                    "del" in words and "=" in words
                ):  # template for positioners: 'del = 30.05\n'
                    delta = float(words[2])
                if (
                    "gam" in words and "=" in words
                ):  # template for positioners: 'gam = 4.05\n'
                    gamma = float(words[2])
                if (
                    "mu" in words and "=" in words
                ):  # template for positioners: 'mu = 0.0\n'
                    mu = float(words[2])
                if (
                    "fmbenergy" in words and "=" in words
                ):  # template for positioners: 'mu = 0.0\n'
                    energy = float(words[2])

                if index_om is not None and valid.is_float(words[0]):
                    if index_phi is not None:
                        raise NotImplementedError(
                            "d2scan with om and phi not supported"
                        )
                    # reading data and index_om is defined (outofplane case)
                    rocking_positions.append(float(words[index_om]))
                if index_phi is not None and valid.is_float(words[0]):
                    if index_om is not None:
                        raise NotImplementedError(
                            "d2scan with om and phi not supported"
                        )
                    # reading data and index_phi is defined (inplane case)
                    rocking_positions.append(float(words[index_phi]))

            if setup.rocking_angle == "outofplane":
                om = np.asarray(rocking_positions, dtype=float)
            else:  # phi
                phi = np.asarray(rocking_positions, dtype=float)

            # remove user-defined sample offsets (sample: mu, om, chi, phi)
            if mu is None:
                raise ValueError("Problem reading the fio file, mu is None")
            mu = mu - self.sample_offsets[0]
            if om is None:
                raise ValueError("Problem reading the fio file, om is None")
            om = om - self.sample_offsets[1]
            if chi is None:
                raise ValueError("Problem reading the fio file, chi is None")
            chi = chi - self.sample_offsets[2]
            if phi is None:
                raise ValueError("Problem reading the fio file, phi is None")
            phi = phi - self.sample_offsets[3]
        else:  # manually defined custom scan
            om = setup.custom_motors["om"]
            chi = setup.custom_motors["chi"]
            phi = setup.custom_motors["phi"]
            delta = setup.custom_motors["delta"]
            gamma = setup.custom_motors["gamma"]
            mu = setup.custom_motors["mu"]
            energy = setup.energy

        if (
            mu is None
            or om is None
            or chi is None
            or phi is None
            or gamma is None
            or delta is None
            or energy is None
        ):
            # mypy does not understand 'any(val is None for val in ...)'
            raise ValueError("Problem loading P10 motor positions (None)")
        return mu, om, chi, phi, gamma, delta, energy, setup.distance

    @safeload
    def read_device(self, setup: Setup, device_name: str, **kwargs) -> np.ndarray:
        """
        Extract the scanned device positions/values at P10 beamline.

        :param setup: an instance of the class Setup
        :param device_name: name of the scanned device
        :return: the positions/values of the device as a numpy 1D array
        """
        file = kwargs.get("file")  # this kwarg is provided by @safeload_static
        if file is None:
            raise ValueError("file should be the opened file, not None")

        device_values = []
        index_device = None  # index of the column corresponding to the device in .fio
        self.logger.info(f"Trying to load values for {device_name}...")

        lines = file.readlines()
        for line in lines:
            this_line = line.strip()
            words = this_line.split()

            if "Col" in words and device_name in words:
                # device_name scanned, template = ' Col 0 motor_name DOUBLE\n'
                index_device = int(words[1]) - 1  # python index starts at 0

            if index_device is not None and valid.is_float(words[0]):
                # we are reading data and index_motor is defined
                device_values.append(float(words[index_device]))

        if index_device is None:
            self.logger.info(f"no device {device_name} in the logfile")
        else:
            self.logger.info(f"{device_name} found!")
        return np.asarray(device_values)

    def read_monitor(self, setup: Setup, **kwargs: dict[str, Any]) -> np.ndarray:
        """
        Load the default monitor for a dataset measured at P10.

        :param setup: an instance of the class Setup
        :return: the default monitor values
        """
        monitor: np.ndarray = self.read_device(setup=setup, device_name="ipetra")
        if len(monitor) == 0:
            monitor = self.read_device(setup=setup, device_name="curpetra")
        return monitor


class LoaderP10SAXS(LoaderP10):
    """Loader for PETRAIII P10 SAXS beamline."""

    @safeload
    def motor_positions(
        self, setup: Setup, **kwargs
    ) -> tuple[float | list | np.ndarray | Any, ...]:
        """
        Load the .fio file from the scan and extract motor positions.

        The detector positions are returned in the laboratory frame z, y, x

        :param setup: an instance of the class Setup
        :return: (phi, det_z, det_y, det_x, energy, detector_distance) values
        """
        # load and check kwargs
        file = kwargs.get("file")  # this kwarg is provided by @safeload
        if file is None:
            raise ValueError("file should be the opened file, not None")

        if setup.rocking_angle != "inplane":
            raise ValueError('Wrong value for "rocking_angle" parameter')

        if not setup.custom_scan:
            index_phi = None
            detx = None
            dety2 = None
            detz = None
            positions: list[float] = []

            lines = file.readlines()
            for line in lines:
                this_line = line.strip()
                words = this_line.split()

                if (
                    "detx" in words and "=" in words
                ):  # template for positioners: 'detx = -0.2\n'
                    detx = float(words[2])
                if (
                    "dety2" in words and "=" in words
                ):  # template for positioners: 'dety2 = 320.0\n'
                    dety2 = float(words[2])
                if (
                    "detz" in words and "=" in words
                ):  # template for positioners: 'detz = 28.5\n'
                    detz = float(words[2])

                if "Col" in words and ("sprz" in words or "hprz" in words):
                    # sprz or hprz (SAXS) scanned
                    # template = ' Col 0 sprz DOUBLE\n'
                    index_phi = int(words[1]) - 1  # python index starts at 0
                    self.logger.info(f"{words}, Index Phi= {index_phi}")
                if index_phi is not None and valid.is_float(words[0]):
                    # we are reading data and index_phi is defined
                    positions.append(float(words[index_phi]))

            phi = np.asarray(positions, dtype=float)
        else:
            phi = setup.custom_motors["phi"]
            detx = setup.custom_motors.get("detx", 0)
            detz = setup.custom_motors.get("detz", 0)
            dety2 = setup.custom_motors.get("dety2", 0)
        return phi, detx, detz, dety2, setup.energy, setup.distance


class LoaderCRISTAL(Loader):
    """Loader for SOLEIL CRISTAL beamline."""

    def create_logfile(
        self,
        datadir: str,
        name: str,
        root_folder: str,
        scan_number: int,
        filename: str | None = None,
        template_imagefile: str | None = None,
    ) -> ContextFile:
        """
        Create the logfile, which is the data itself for CRISTAL.

        :param datadir: str, the data directory
        :param name: str, the name of the beamline, e.g. 'CRISTAL'
        :param root_folder: str, the root directory of the experiment
        :param scan_number: the scan number to load
        :param filename: str, absolute path to the spec/fio/alias file when it exists
        :param template_imagefile: str, template for data file name, e.g. 'S%d.nxs'
        :return: an instance of a context manager ContextFile
        """
        valid.valid_container(datadir, container_types=str, name="datadir")
        if not os.path.isdir(datadir):
            raise ValueError(f"The directory {datadir} does not exist")
        if template_imagefile is None or not isinstance(template_imagefile, str):
            raise TypeError("'template_imagefile' should be a string")
        valid.valid_item(
            scan_number, allowed_types=int, min_included=0, name="scan_number"
        )

        # no specfile, load directly the dataset
        try:
            filename = os.path.join(datadir + template_imagefile % scan_number)
        except TypeError:
            self.logger.error("Formatting issue for 'template_imagefile % scan_number'")
            raise ValueError("Formatting issue for 'template_imagefile % scan_number'")
        return ContextFile(
            filename=filename, open_func=h5py.File, scan_number=scan_number
        )

    @safeload
    def cristal_load_motor(
        self,
        setup: Setup,
        root: str,
        actuator_name: str,
        field_name: str,
        **kwargs,
    ) -> float | list[float] | np.ndarray:
        """
        Try to load the dataset at the defined entry and returns it.

        Patterns keep changing at CRISTAL.

        :param setup: an instance of the class Setup
        :param root: string, path of the data up to the last subfolder
         (not included). This part is expected to not change over time
        :param actuator_name: string, name of the actuator
         (e.g. 'I06-C-C07-EX-DIF-KPHI'). Lowercase and uppercase will be tested when
         trying to load the data.
        :param field_name: name of the field under the actuator name (e.g. 'position')
        :return: the dataset if found or 0
        """
        # load and check kwargs
        file = kwargs.get("file")  # this kwarg is provided by @safeload
        if file is None:
            raise ValueError("file should be the opened file, not None")

        # check input arguments
        valid.valid_container(
            root, container_types=str, min_length=1, name="cristal_load_motor"
        )
        if not root.startswith("/"):
            root = "/" + root
        valid.valid_container(
            actuator_name, container_types=str, min_length=1, name="cristal_load_motor"
        )

        # check if there is an entry for the actuator
        if actuator_name not in file[root].keys():
            actuator_name = actuator_name.lower()
            if actuator_name not in file[root].keys():
                actuator_name = actuator_name.upper()
                if actuator_name not in file[root].keys():
                    self.logger.info(
                        f"Could not find the entry for the actuator'{actuator_name}': "
                        f"list of available actuators {list(file[root].keys())}. "
                        f"Defaulting '{actuator_name}' to 0 deg"
                    )
                    return 0

        # check if the field is a valid entry for the actuator
        try:
            dataset = file[root + "/" + actuator_name + "/" + field_name][:]
        except KeyError:  # try lowercase
            try:
                dataset = file[root + "/" + actuator_name + "/" + field_name.lower()][:]
            except KeyError:  # try uppercase
                try:
                    dataset = file[
                        root + "/" + actuator_name + "/" + field_name.upper()
                    ][:]
                except KeyError:  # nothing else that we can do
                    self.logger.info(
                        f"Could not find the field '{field_name}'"
                        f" in the actuator'{actuator_name}': list of available fields "
                        f"{list(file[root + '/' + actuator_name].keys())}. "
                        f"Defaulting '{actuator_name}' to 0 deg"
                    )
                    return 0
        return util.unpack_array(dataset)

    @safeload
    def find_detector(
        self,
        setup: Setup,
        root: str,
        data_path: str = "scan_data",
        pattern: str = "^data_[0-9][0-9]$",
        **kwargs,
    ):
        """
        Look for the entry corresponding to the detector data in CRISTAL dataset.

        :param setup: an instance of the class Setup
        :param root: root folder name in the data file
        :param data_path: string, name of the subfolder when the scan data is located
        :param pattern: string, pattern corresponding to the entries where the detector
         data could be located
        :return: numpy array of the shape of the detector dataset
        """
        file = kwargs.get("file")  # this kwarg is provided by @safeload
        if file is None:
            raise ValueError("file should be the opened file, not None")

        # check input arguments
        valid.valid_container(
            root, container_types=str, min_length=1, name="cristal_find_data"
        )
        if not root.startswith("/"):
            root = "/" + root

        valid.valid_container(
            data_path, container_types=str, min_length=1, name="cristal_find_data"
        )
        if not data_path.startswith("/"):
            data_path = "/" + data_path
        valid.valid_container(
            pattern, container_types=str, min_length=1, name="cristal_find_data"
        )

        if isinstance(setup.actuators, dict) and "detector" in setup.actuators:
            return file[root + data_path + "/" + setup.actuators["detector"]][:]

        # loop over the available keys at the defined path in the file
        # and check the shape of the corresponding dataset
        nb_pix_ver, nb_pix_hor = (setup.detector.nb_pixel_y, setup.detector.nb_pixel_x)
        for key in list(file[root + data_path]):
            if bool(re.match(pattern, key)):
                obj_shape = file[root + data_path + "/" + key][:].shape
                if nb_pix_ver in obj_shape and nb_pix_hor in obj_shape:
                    # found the key corresponding to the detector
                    self.logger.info(
                        f"subdirectory '{key}' contains the detector images, "
                        f"shape={obj_shape}"
                    )
                    return file[root + data_path + "/" + key][:]
        raise ValueError(
            f"Could not find detector data using data_path={data_path} "
            f"and pattern={pattern}"
        )

    @staticmethod
    def init_paths(root_folder, sample_name, scan_number, template_imagefile, **kwargs):
        """
        Initialize paths used for data processing and logging at CRISTAL.

        :param root_folder: folder of the experiment, where all scans are stored
        :param sample_name: string in front of the scan number in the data folder
         name.
        :param scan_number: int, the scan number
        :param template_imagefile: template for the data files, e.g. 'S%d.nxs'
        :return: a tuple of strings:

         - homedir: the path of the scan folder
         - default_dirname: the name of the folder containing images / raw data
         - specfile: not used at CRISTAL
         - template_imagefile: the template for data/image file names

        """
        homedir = root_folder + sample_name + str(scan_number) + "/"
        default_dirname = "data/"
        return homedir, default_dirname, None, template_imagefile

    @safeload
    def load_data(
        self,
        setup: Setup,
        flatfield: np.ndarray | None = None,
        hotpixels: np.ndarray | None = None,
        background: np.ndarray | None = None,
        normalize: str = "skip",
        bin_during_loading: bool = False,
        debugging: bool = False,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """
        Load CRISTAL data including detector/background corrections.

        It will look for the correct entry 'detector' in the dictionary 'actuators',
        and look for a dataset with compatible shape otherwise.

        :param setup: an instance of the class Setup
        :param flatfield: the 2D flatfield array
        :param hotpixels: the 2D hotpixels array
        :param background: the 2D background array to subtract to the data
        :param normalize: 'monitor' to return the default monitor values, 'sum_roi' to
         return a monitor based on the integrated intensity in the region of interest
         defined by detector.sum_roi, 'skip' to do nothing
        :param bin_during_loading: if True, the data will be binned in the detector
         frame while loading. It saves a lot of memory space for large 2D detectors.
        :param debugging: set to True to see plots
        :return:
         - the 3D data array in the detector frame
         - the 2D mask array
         - the monitor values for normalization
         - the detector region of interest used for loading the data

        """
        file = kwargs.get("file")  # this kwarg is provided by @safeload
        if file is None:
            raise ValueError("file should be the opened file, not None")

        if setup.actuators is None:
            raise ValueError("'actuators' parameter required")

        # look for the detector entry (keep changing at CRISTAL)
        if setup.custom_scan:
            raise NotImplementedError("custom scan not implemented for CRISTAL")
        group_key = list(file.keys())[0]
        tmp_data = self.find_detector(setup=setup, root=group_key)

        # find the number of images
        nb_img = tmp_data.shape[0]

        data, mask2d, monitor, loading_roi = self.init_data_mask(
            detector=setup.detector,
            setup=setup,
            normalize=normalize,
            nb_frames=nb_img,
            bin_during_loading=bin_during_loading,
        )

        # loop over frames, mask the detector and normalize / bin
        for idx in range(nb_img):
            data[idx, :, :], mask2d, monitor[idx] = load_frame(
                frame=tmp_data[idx, :, :],
                mask2d=mask2d,
                monitor=monitor[idx],
                frames_per_point=1,
                detector=setup.detector,
                loading_roi=loading_roi,
                flatfield=flatfield,
                background=background,
                hotpixels=hotpixels,
                normalize=normalize,
                bin_during_loading=bin_during_loading,
                debugging=debugging,
            )
        return data, mask2d, monitor, loading_roi

    def motor_positions(
        self, setup: Setup, **kwargs
    ) -> tuple[float | list | np.ndarray, ...]:
        """
        Load the scan data and extract motor positions.

        It will look for the correct entry 'rocking_angle' in the dictionary
        Setup.actuators, and use the default entry otherwise.

        :param setup: an instance of the class Setup
        :return: (mgomega, mgphi, gamma, delta, energy) values
        """
        if not setup.custom_scan:
            if setup.logfile is None:
                raise ValueError("logfile undefined")
            with setup.logfile as file:
                group_key = list(file.keys())[0]

            if setup.rocking_angle != "energy":
                delta = self.cristal_load_motor(
                    setup=setup,
                    root="/" + group_key + "/CRISTAL/Diffractometer/",
                    actuator_name="I06-C-C07-EX-DIF-DELTA",
                    field_name="position",
                )
                gamma = self.cristal_load_motor(
                    setup=setup,
                    root="/" + group_key + "/CRISTAL/Diffractometer/",
                    actuator_name="I06-C-C07-EX-DIF-GAMMA",
                    field_name="position",
                )
                energy = (
                    self.cristal_load_motor(
                        setup=setup,
                        root="/" + group_key + "/CRISTAL/",
                        actuator_name="Monochromator",
                        field_name="energy",
                    )
                    * 1000
                )  # in eV
                if (
                    setup.energy is not None
                    and isinstance(energy, (int, float))
                    and abs(energy - setup.energy) > 1
                ):
                    # difference larger than 1 eV
                    self.logger.warning(
                        f"user-defined energy = {setup.energy:.1f} eV different from "
                        f"the energy in the datafile = {energy:.1f} eV"
                    )

                scanned_motor = self.cristal_load_motor(
                    setup=setup,
                    root="/" + group_key,
                    actuator_name="scan_data",
                    field_name=setup.actuators.get("rocking_angle", "actuator_1_1"),
                )

                if setup.rocking_angle == "outofplane":
                    mgomega = scanned_motor  # mgomega is scanned
                    mgphi = self.cristal_load_motor(
                        setup=setup,
                        root="/" + group_key + "/CRISTAL/",
                        actuator_name="i06-c-c07-ex-mg_phi",
                        field_name="position",
                    )
                else:  # "inplane"
                    mgphi = scanned_motor  # mgphi is scanned
                    mgomega = self.cristal_load_motor(
                        setup=setup,
                        root="/" + group_key + "/CRISTAL/",
                        actuator_name="i06-c-c07-ex-mg_omega",
                        field_name="position",
                    )
            else:  # energy scan
                delta = self.cristal_load_motor(
                    setup=setup,
                    root="/" + group_key,
                    actuator_name="scan_data",
                    field_name=setup.actuators.get("delta", "actuator_1_2"),
                )
                energy = (
                    self.cristal_load_motor(
                        setup=setup,
                        root="/" + group_key,
                        actuator_name="scan_data",
                        field_name=setup.actuators.get("rocking_angle", "actuator_1_3"),
                    )
                    * 1000
                )  # switch to eV
                mgomega = self.cristal_load_motor(
                    setup=setup,
                    root="/" + group_key + "/CRISTAL/",
                    actuator_name="i06-c-c07-ex-mg_omega",
                    field_name="position",
                )
                mgphi = self.cristal_load_motor(
                    setup=setup,
                    root="/" + group_key + "/CRISTAL/",
                    actuator_name="i06-c-c07-ex-mg_phi",
                    field_name="position",
                )
                gamma = self.cristal_load_motor(
                    setup=setup,
                    root="/" + group_key + "/CRISTAL/Diffractometer/",
                    actuator_name="I06-C-C07-EX-DIF-GAMMA",
                    field_name="position",
                )

            # remove user-defined sample offsets (sample: mgomega, mgphi)
            mgomega = mgomega - self.sample_offsets[0]
            mgphi = mgphi - self.sample_offsets[1]

        else:  # manually defined custom scan
            mgomega = setup.custom_motors["mgomega"]
            delta = setup.custom_motors["delta"]
            gamma = setup.custom_motors["gamma"]
            mgphi = setup.custom_motors.get("mgphi", 0)
            energy = setup.energy

        # check if mgomega needs to be divided by 1e6
        # (data taken before the implementation of the correction)
        if isinstance(mgomega, float) and abs(mgomega) > 360:
            mgomega = mgomega / 1e6
        elif isinstance(mgomega, (tuple, list, np.ndarray)) and any(
            abs(val) > 360 for val in mgomega
        ):
            mgomega = np.asarray(mgomega) / 1e6

        return mgomega, mgphi, gamma, delta, energy, setup.distance

    @safeload
    def read_device(self, setup: Setup, device_name: str, **kwargs) -> np.ndarray:
        """
        Extract the scanned device positions/values at CRISTAL beamline.

        :param setup: an instance of the class Setup
        :param device_name: name of the scanned device
        :return: the positions/values of the device as a numpy 1D array
        """
        file = kwargs.get("file")  # this kwarg is provided by @safeload_static
        if file is None:
            raise ValueError("file should be the opened file, not None")

        group_key = list(file.keys())[0]
        self.logger.info(f"Trying to load values for {device_name}...")
        try:
            device_values = file["/" + group_key + "/scan_data/" + device_name][:]
            self.logger.info(f"{device_name} found!")
        except KeyError:
            self.logger.info(f"no device {device_name} in the logfile")
            device_values = []
        return np.asarray(device_values)

    def read_monitor(self, setup: Setup, **kwargs: dict[str, Any]) -> np.ndarray:
        """
        Load the default monitor for a dataset measured at CRISTAL.

        :param setup: an instance of the class Setup
        :return: the default monitor values
        """
        if setup.actuators is not None:
            monitor_name = setup.actuators.get("monitor", "data_04")
        else:
            monitor_name = "data_04"
        monitor: np.ndarray = self.read_device(setup=setup, device_name=monitor_name)
        return monitor


class LoaderNANOMAX(Loader):
    """Loader for MAX IV NANOMAX beamline."""

    def create_logfile(
        self,
        datadir: str,
        name: str,
        root_folder: str,
        scan_number: int,
        filename: str | None = None,
        template_imagefile: str | None = None,
    ) -> ContextFile:
        """
        Create the logfile, which is the data itself for Nanomax.

        :param datadir: str, the data directory
        :param name: str, the name of the beamline, e.g. 'Nanomax'
        :param root_folder: str, the root directory of the experiment
        :param scan_number: the scan number to load
        :param filename: str, absolute path to the spec/fio/alias file when it exists
        :param template_imagefile: str, template for data file name, e.g. '%06d.h5'
        :return: an instance of a context manager ContextFile
        """
        valid.valid_container(datadir, container_types=str, name="datadir")
        if not os.path.isdir(datadir):
            raise ValueError(f"The directory {datadir} does not exist")
        if template_imagefile is None or not isinstance(template_imagefile, str):
            raise TypeError("'template_imagefile' should be a string")
        valid.valid_item(
            scan_number, allowed_types=int, min_included=0, name="scan_number"
        )
        path = os.path.join(datadir + template_imagefile % scan_number)
        return ContextFile(filename=path, open_func=h5py.File, scan_number=scan_number)

    @staticmethod
    def init_paths(root_folder, sample_name, scan_number, template_imagefile, **kwargs):
        """
        Initialize paths used for data processing and logging at Nanomax.

        :param root_folder: folder of the experiment, where all scans are stored
        :param sample_name: string in front of the scan number in the data folder
         name.
        :param scan_number: int, the scan number
        :param template_imagefile: template for the data files, e.g. '%06d.h5'
        :return: a tuple of strings:

         - homedir: the path of the scan folder
         - default_dirname: the name of the folder containing images / raw data
         - specfile: the name of the specfile if it exists
         - template_imagefile: the template for data/image file names

        """
        homedir = root_folder + sample_name + f"{scan_number:06d}/"
        default_dirname = "data/"
        return homedir, default_dirname, None, template_imagefile

    @safeload
    def load_data(
        self,
        setup: Setup,
        flatfield: np.ndarray | None = None,
        hotpixels: np.ndarray | None = None,
        background: np.ndarray | None = None,
        normalize: str = "skip",
        bin_during_loading: bool = False,
        debugging: bool = False,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """
        Load NANOMAX data, apply filters and concatenate it for phasing.

        :param setup: an instance of the calss Setup
        :param flatfield: the 2D flatfield array
        :param hotpixels: the 2D hotpixels array
        :param background: the 2D background array to subtract to the data
        :param normalize: 'monitor' to return the default monitor values, 'sum_roi' to
         return a monitor based on the integrated intensity in the region of interest
         defined by detector.sum_roi, 'skip' to do nothing
        :param bin_during_loading: if True, the data will be binned in the detector
         frame while loading. It saves a lot of memory space for large 2D detectors.
        :param debugging: set to True to see plots
        :return:

         - the 3D data array in the detector frame
         - the 2D mask array
         - the monitor values for normalization
         - the detector region of interest used for loading the data

        """
        file = kwargs.get("file")  # this kwarg is provided by @safeload
        if file is None:
            raise ValueError("file should be the opened file, not None")

        if debugging:
            self.logger.info(
                str(file["entry"]["description"][()])[3:-2]
            )  # Reading only useful symbols

        if setup.custom_scan:
            raise NotImplementedError("custom scan not implemented for NANOMAX")
        group_key = list(file.keys())[0]  # currently 'entry'
        try:
            tmp_data = file["/" + group_key + "/measurement/merlin/frames"][:]
        except KeyError:
            tmp_data = file["/" + group_key + "measurement/Merlin/data"][()]

        # find the number of images
        nb_img = tmp_data.shape[0]

        data, mask2d, monitor, loading_roi = self.init_data_mask(
            detector=setup.detector,
            setup=setup,
            normalize=normalize,
            nb_frames=nb_img,
            bin_during_loading=bin_during_loading,
        )

        # loop over frames, mask the detector and normalize / bin
        for idx in range(nb_img):
            data[idx, :, :], mask2d, monitor[idx] = load_frame(
                frame=tmp_data[idx, :, :],
                mask2d=mask2d,
                monitor=monitor[idx],
                frames_per_point=1,
                detector=setup.detector,
                loading_roi=loading_roi,
                flatfield=flatfield,
                background=background,
                hotpixels=hotpixels,
                normalize=normalize,
                bin_during_loading=bin_during_loading,
                debugging=debugging,
            )
        return data, mask2d, monitor, loading_roi

    @safeload
    def motor_positions(
        self, setup: Setup, **kwargs
    ) -> tuple[float | list | np.ndarray, ...]:
        """
        Load the scan data and extract motor positions.

        :param setup: an instance of the class Setup
        :return: (theta, phi, gamma, delta, energy) values
        """
        file = kwargs.get("file")  # this kwarg is provided by @safeload
        if file is None:
            raise ValueError("file should be the opened file, not None")

        if not setup.custom_scan:
            # Detector positions
            group_key = list(file.keys())[0]  # currently 'entry'

            # positionners
            delta = util.unpack_array(file["/" + group_key + "/snapshot/delta"][:])
            gamma = util.unpack_array(file["/" + group_key + "/snapshot/gamma"][:])
            energy = util.unpack_array(file["/" + group_key + "/snapshot/energy"][:])

            if setup.rocking_angle == "inplane":
                try:
                    phi = util.unpack_array(
                        file["/" + group_key + "/measurement/gonphi"][:]
                    )
                except KeyError:
                    raise KeyError(
                        "phi not in measurement data,"
                        ' check the parameter "rocking_angle"'
                    )
                theta = util.unpack_array(
                    file["/" + group_key + "/snapshot/gontheta"][:]
                )
            else:
                try:
                    theta = util.unpack_array(
                        file["/" + group_key + "/measurement/gontheta"][:]
                    )
                except KeyError:
                    raise KeyError(
                        "theta not in measurement data,"
                        ' check the parameter "rocking_angle"'
                    )
                phi = util.unpack_array(file["/" + group_key + "/snapshot/gonphi"][:])

            # remove user-defined sample offsets (sample: theta, phi)
            theta = theta - self.sample_offsets[0]
            phi = phi - self.sample_offsets[1]

        else:  # manually defined custom scan
            theta = setup.custom_motors["theta"]
            phi = setup.custom_motors["phi"]
            delta = setup.custom_motors["delta"]
            gamma = setup.custom_motors["gamma"]
            energy = setup.energy

        return theta, phi, gamma, delta, energy, setup.distance

    @safeload
    def read_device(self, setup: Setup, device_name: str, **kwargs) -> np.ndarray:
        """
        Extract the scanned device positions/values at Nanomax beamline.

        :param setup: an instance of the class Setup
        :param device_name: name of the scanned device
        :return: the positions/values of the device as a numpy 1D array
        """
        file = kwargs.get("file")  # this kwarg is provided by @safeload_static
        if file is None:
            raise ValueError("file should be the opened file, not None")

        group_key = list(file.keys())[0]  # currently 'entry'
        self.logger.info(f"Trying to load values for {device_name}...")
        try:
            device_values = file["/" + group_key + "/measurement/" + device_name][:]
            self.logger.info(f"{device_name} found!")
        except KeyError:
            self.logger.info(f"No device {device_name} in the logfile")
            device_values = []
        return np.asarray(device_values)

    def read_monitor(self, setup: Setup, **kwargs: dict[str, Any]) -> np.ndarray:
        """
        Load the default monitor for a dataset measured at NANOMAX.

        :param setup: an instance of the class Setup
        :return: the default monitor values
        """
        monitor: np.ndarray = self.read_device(setup=setup, device_name="alba2")
        return monitor
