# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#   (c) 06/2021-present : DESY CFEL
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

from collections.abc import Sequence
from functools import reduce
import gc
import h5py
from math import isclose
from numbers import Number, Real
import numpy as np
import os
import pathlib
from scipy.interpolate import RegularGridInterpolator
import warnings
from ..graph import graph_utils as gu
from ..utils import utilities as util
from ..utils import validation as valid


class Detector:
    """
    Class to handle the configuration of the detector used for data acquisition.

    :param name: name of the detector in {'Maxipix', 'Timepix', 'Merlin', 'Eiger2M', 'Eiger4M'}
    :param datadir: directory where the data files are located
    :param savedir: directory where to save the results
    :param template_imagefile: beamline-dependent template for the data files

     - ID01: 'data_mpx4_%05d.edf.gz' or 'align_eiger2M_%05d.edf.gz'
     - SIXS_2018: 'align.spec_ascan_mu_%05d.nxs'
     - SIXS_2019: 'spare_ascan_mu_%05d.nxs'
     - Cristal: 'S%d.nxs'
     - P10: '_master.h5'
     - NANOMAX: '%06d.h5'
     - 34ID: 'Sample%dC_ES_data_51_256_256.npz'
    :param specfile: template for the log file or the data file depending on the beamline
    :param roi: region of interest of the detector used for analysis
    :param sum_roi: region of interest of the detector used for calculated an integrated intensity
    :param binning: binning factor of the 3D dataset
     (stacking dimension, detector vertical axis, detector horizontal axis)
    :param kwargs:

     - 'is_series': boolean, True is the measurement is a series at PETRAIII P10 beamline
     - 'nb_pixel_x' and 'nb_pixel_y': useful when part of the detector is broken (less pixels than expected)
     - 'preprocessing_binning': tuple of the three binning factors used in a previous preprocessing step
     - 'offsets': tuple or list, sample and detector offsets corresponding to the parameter delta
       in xrayutilities hxrd.Ang2Q.area method
     - 'linearity_func': function to apply to each pixel of the detector in order to compensate the deviation of the
       detector linearity for large intensities.
    """

    def __init__(
        self,
        name,
        rootdir=None,
        datadir=None,
        savedir=None,
        template_file=None,
        template_imagefile=None,
        specfile=None,
        sample_name=None,
        roi=None,
        sum_roi=None,
        binning=(1, 1, 1),
        **kwargs,
    ):
        # the detector name should be initialized first, other properties are depending on it
        self.name = name

        valid.valid_kwargs(
            kwargs=kwargs,
            allowed_kwargs={
                "is_series",
                "nb_pixel_x",
                "nb_pixel_y",
                "pixel_size",
                "preprocessing_binning",
                "offsets",
                "linearity_func",
            },
            name="Detector.__init__",
        )

        # load the kwargs
        self.is_series = kwargs.get("is_series", False)
        self.preprocessing_binning = kwargs.get("preprocessing_binning") or (1, 1, 1)
        self.nb_pixel_x = kwargs.get("nb_pixel_x")
        self.nb_pixel_y = kwargs.get("nb_pixel_y")
        self.custom_pixelsize = kwargs.get("pixel_size")
        self.offsets = kwargs.get("offsets")  # delegate the test to xrayutilities
        linearity_func = kwargs.get("linearity_func")
        if linearity_func is not None and not callable(linearity_func):
            raise TypeError(
                f"linearity_func should be a function, got {type(linearity_func)}"
            )
        self._linearity_func = linearity_func

        # load other positional arguments
        self.binning = binning
        self.roi = roi
        self.sum_roi = sum_roi
        # parameters related to data path
        self.rootdir = rootdir
        self.datadir = datadir
        self.savedir = savedir
        self.sample_name = sample_name
        self.template_file = template_file
        self.template_imagefile = template_imagefile
        self.specfile = specfile

    @property
    def binning(self):
        """
        Tuple of three positive integers corresponding to the binning of the data used in phase retrieval
         (stacking dimension, detector vertical axis, detector horizontal axis). To declare an additional binning factor
         due to a previous preprocessing step, use the kwarg 'preprocessing_binning' instead.
        """
        return self._binning

    @binning.setter
    def binning(self, value):
        valid.valid_container(
            value,
            container_types=(tuple, list),
            length=3,
            item_types=int,
            min_excluded=0,
            name="Detector.binning",
        )
        self._binning = value

    @property
    def counter(self):
        """
        Name of the counter for the image number.
        """
        counter_dict = {
            "Maxipix": "mpx4inr",
            "Eiger2M": "ei2minr",
            "Eiger4M": None,
            "Timepix": None,
            "Merlin": "alba2",
            "Dummy": None,
        }
        return counter_dict.get(self.name)

    @property
    def datadir(self):
        """
        Name of the data directory
        """
        return self._datadir

    @datadir.setter
    def datadir(self, value):
        valid.valid_container(
            value,
            container_types=str,
            min_length=1,
            allow_none=True,
            name="Detector.datadir",
        )
        self._datadir = value

    @property
    def is_series(self):
        """
        Boolean, True for a series measurement at PETRAIII P10
        """
        return self._is_series

    @is_series.setter
    def is_series(self, value):
        if not isinstance(value, bool):
            raise TypeError("is_series should be a boolean")
        self._is_series = value

    @property
    def name(self):
        """
        Name of the detector
        """
        return self._name

    @name.setter
    def name(self, value):
        valid_names = {"Maxipix", "Timepix", "Merlin", "Eiger2M", "Eiger4M", "Dummy"}
        if value not in valid_names:
            raise ValueError(f"Name should be in {valid_names}")
        self._name = value

    @property
    def nb_pixel_x(self):
        """
        Horizontal number of pixels of the detector, taking into account an eventual preprocessing binning.
        """
        return self._nb_pixel_x

    @nb_pixel_x.setter
    def nb_pixel_x(self, value):
        if value is None:
            value = self.pix_number[1]
        if not isinstance(value, int):
            raise TypeError("nb_pixel_x should be a positive integer")
        if value <= 0:
            raise ValueError("nb_pixel_x should be a positive integer")
        self._nb_pixel_x = value // self.preprocessing_binning[2]

    @property
    def nb_pixel_y(self):
        """
        Vertical number of pixels of the detector, taking into account an eventual preprocessing binning.
        """
        return self._nb_pixel_y

    @nb_pixel_y.setter
    def nb_pixel_y(self, value):
        if value is None:
            value = self.pix_number[0]
        if not isinstance(value, int):
            raise TypeError("nb_pixel_y should be a positive integer")
        if value <= 0:
            raise ValueError("nb_pixel_y should be a positive integer")
        self._nb_pixel_y = value // self.preprocessing_binning[1]

    @property
    def params(self):
        """
        Return a dictionnary with all parameters
        """
        return {
            "Class": self.__class__.__name__,
            "name": self.name,
            "unbinned_pixel_m": self.unbinned_pixel,
            "nb_pixel_x": self.nb_pixel_x,
            "nb_pixel_y": self.nb_pixel_y,
            "binning": self.binning,
            "roi": self.roi,
            "sum_roi": self.sum_roi,
            "preprocessing_binning": self.preprocessing_binning,
            "is_series": self.is_series,
            "rootdir": self.rootdir,
            "datadir": self.datadir,
            "scandir": self.scandir,
            "savedir": self.savedir,
            "sample_name": self.sample_name,
            "template_file": self.template_file,
            "template_imagefile": self.template_imagefile,
            "specfile": self.specfile,
        }

    @property
    def pixelsize_x(self):
        """
        Horizontal pixel size of the detector after taking into account binning.
        """
        return self.unbinned_pixel[1] * self.preprocessing_binning[2] * self.binning[2]

    @property
    def pixelsize_y(self):
        """
        Vertical pixel size of the detector after taking into account binning.
        """
        return self.unbinned_pixel[0] * self.preprocessing_binning[1] * self.binning[1]

    @property
    def pix_number(self):
        """
        Number of pixels (vertical, horizontal) of the unbinned detector.
        """
        if self.name in {"Maxipix", "Dummy"}:
            number = (516, 516)
        elif self.name == "Timepix":
            number = (256, 256)
        elif self.name == "Merlin":
            number = (515, 515)
        elif self.name == "Eiger2M":
            number = (2164, 1030)
        elif self.name == "Eiger4M":
            number = (2167, 2070)
        else:
            number = None
        return number

    @property
    def preprocessing_binning(self):
        """
        Tuple of three positive integers corresponding to the binning factor of the data used in a previous
         preprocessing step (stacking dimension, detector vertical axis, detector horizontal axis).
        """
        return self._preprocessing_binning

    @preprocessing_binning.setter
    def preprocessing_binning(self, value):
        valid.valid_container(
            value,
            container_types=(tuple, list),
            length=3,
            item_types=int,
            min_excluded=0,
            name="Detector.preprocessing_binning",
        )
        self._preprocessing_binning = value

    @property
    def roi(self):
        """
        Region of interest of the detector to be used [y_start, y_stop, x_start, x_stop]
        """
        return self._roi

    @roi.setter
    def roi(self, value):
        if not value:  # None or empty list/tuple
            value = [0, self.nb_pixel_y, 0, self.nb_pixel_x]
        valid.valid_container(
            value,
            container_types=(tuple, list),
            length=4,
            item_types=int,
            name="Detector.roi",
        )
        self._roi = value

    @property
    def rootdir(self):
        """
        Name of the root directory, which englobes all scans
        """
        return self._rootdir

    @rootdir.setter
    def rootdir(self, value):
        valid.valid_container(
            value,
            container_types=str,
            min_length=1,
            allow_none=True,
            name="Detector.rootdir",
        )
        self._rootdir = value

    @property
    def sample_name(self):
        """
        Name of the sample
        """
        return self._sample_name

    @sample_name.setter
    def sample_name(self, value):
        valid.valid_container(
            value,
            container_types=str,
            min_length=1,
            allow_none=True,
            name="Detector.sample_name",
        )
        self._sample_name = value

    @property
    def savedir(self):
        """
        Name of the saving directory
        """
        return self._savedir

    @savedir.setter
    def savedir(self, value):
        valid.valid_container(
            value,
            container_types=str,
            min_length=1,
            allow_none=True,
            name="Detector.savedir",
        )
        self._savedir = value

    @property
    def scandir(self):
        """
        Path of the scan, typically it is the parent folder of the data folder
        """
        if self.datadir:
            dir_path = os.path.abspath(os.path.join(self.datadir, os.pardir)) + "/"
            return dir_path.replace("\\", "/")

    @property
    def sum_roi(self):
        """
        Region of interest of the detector used for integrating the intensity [y_start, y_stop, x_start, x_stop]
        """
        return self._sum_roi

    @sum_roi.setter
    def sum_roi(self, value):
        if not value:  # None or empty list/tuple
            if not self.roi:
                value = [0, self.nb_pixel_y, 0, self.nb_pixel_x]
            else:
                value = self.roi
        valid.valid_container(
            value,
            container_types=(tuple, list),
            length=4,
            item_types=int,
            name="Detector.sum_roi",
        )
        self._sum_roi = value

    @property
    def template_file(self):
        """
        Template that can be used to generate template_imagefile.
        """
        return self._template_file

    @template_file.setter
    def template_file(self, value):
        valid.valid_container(
            value,
            container_types=str,
            min_length=0,
            allow_none=True,
            name="Detector.template_file",
        )
        self._template_file = value

    @property
    def template_imagefile(self):
        """
        Name of the data file.
        """
        return self._template_imagefile

    @template_imagefile.setter
    def template_imagefile(self, value):
        valid.valid_container(
            value,
            container_types=str,
            min_length=0,
            allow_none=True,
            name="Detector.imagefile",
        )
        self._template_imagefile = value

    @property
    def unbinned_pixel(self):
        """
        Pixel size (vertical, horizontal) of the unbinned detector in meters.
        """
        if self.name in {"Maxipix", "Timepix", "Merlin"}:
            pix = (55e-06, 55e-06)
        elif self.name in {"Eiger2M", "Eiger4M"}:
            pix = (75e-06, 75e-06)
        elif self.name == "Dummy":
            if self.custom_pixelsize is not None:
                valid.valid_item(
                    self.custom_pixelsize,
                    allowed_types=Real,
                    min_excluded=0,
                    name="custom_pixelsize",
                )
                pix = (self.custom_pixelsize, self.custom_pixelsize)
            else:
                pix = (55e-06, 55e-06)
                print(f"Defaulting the pixel size to {pix}")
        else:
            pix = None
        return pix

    def __repr__(self):
        """
        Representation string of the Detector instance.
        """
        return (
            f"{self.__class__.__name__}(name='{self.name}', unbinned_pixel={self.unbinned_pixel}, "
            f"nb_pixel_x={self.nb_pixel_x}, nb_pixel_y={self.nb_pixel_y}, binning={self.binning},\n"
            f"roi={self.roi}, sum_roi={self.sum_roi}, preprocessing_binning={self.preprocessing_binning}, "
            f"is_series={self.is_series}\nrootdir = {self.rootdir},\ndatadir = {self.datadir},\n"
            f"scandir = {self.scandir},\nsavedir = {self.savedir},\nsample_name = {self.sample_name},"
            f" template_file = {self.template_file}, template_imagefile = {self.template_imagefile},"
            f" specfile = {self.specfile},\n"
        )

    def mask_detector(
        self, data, mask, nb_img=1, flatfield=None, background=None, hotpixels=None
    ):
        """
        Mask data measured with a 2D detector (flatfield, background, hotpixels, gaps).

        :param data: the 2D data to mask
        :param mask: the 2D mask to be updated
        :param nb_img: number of images summed to yield the 2D data (e.g. in a series measurement)
        :param flatfield: the 2D flatfield array to be multiplied with the data
        :param background: a 2D array to be subtracted to the data
        :param hotpixels: a 2D array with hotpixels to be masked (1=hotpixel, 0=normal pixel)
        :return: the masked data and the updated mask
        """
        if not isinstance(data, np.ndarray) or not isinstance(mask, np.ndarray):
            raise TypeError("data and mask should be numpy arrays")
        if data.ndim != 2 or mask.ndim != 2:
            raise ValueError("data and mask should be 2D arrays")

        if data.shape != mask.shape:
            raise ValueError(
                "data and mask must have the same shape\n data is ",
                data.shape,
                " while mask is ",
                mask.shape,
            )

        # linearity correction
        if self._linearity_func is not None:
            data = data.astype(float)
            nby, nbx = data.shape
            data = self._linearity_func(data.flatten()).reshape((nby, nbx))

        # flatfield correction
        if flatfield is not None:
            if flatfield.shape != data.shape:
                raise ValueError(
                    "flatfield and data must have the same shape\n data is ",
                    flatfield.shape,
                    " while data is ",
                    data.shape,
                )
            data = np.multiply(flatfield, data)

        # remove the background
        if background is not None:
            if background.shape != data.shape:
                raise ValueError(
                    "background and data must have the same shape\n data is ",
                    background.shape,
                    " while data is ",
                    data.shape,
                )
            data = data - background

        # mask hotpixels
        if hotpixels is not None:
            if hotpixels.shape != data.shape:
                raise ValueError(
                    "hotpixels and data must have the same shape\n data is ",
                    hotpixels.shape,
                    " while data is ",
                    data.shape,
                )
            data[hotpixels == 1] = 0
            mask[hotpixels == 1] = 1

        if self.name == "Eiger2M":
            data[:, 255:259] = 0
            data[:, 513:517] = 0
            data[:, 771:775] = 0
            data[0:257, 72:80] = 0
            data[255:259, :] = 0
            data[511:552, :0] = 0
            data[804:809, :] = 0
            data[1061:1102, :] = 0
            data[1355:1359, :] = 0
            data[1611:1652, :] = 0
            data[1905:1909, :] = 0
            data[1248:1290, 478] = 0
            data[1214:1298, 481] = 0
            data[1649:1910, 620:628] = 0

            mask[:, 255:259] = 1
            mask[:, 513:517] = 1
            mask[:, 771:775] = 1
            mask[0:257, 72:80] = 1
            mask[255:259, :] = 1
            mask[511:552, :] = 1
            mask[804:809, :] = 1
            mask[1061:1102, :] = 1
            mask[1355:1359, :] = 1
            mask[1611:1652, :] = 1
            mask[1905:1909, :] = 1
            mask[1248:1290, 478] = 1
            mask[1214:1298, 481] = 1
            mask[1649:1910, 620:628] = 1

            # mask hot pixels
            mask[data > 1e6 * nb_img] = 1
            data[data > 1e6 * nb_img] = 0

        elif self.name == "Eiger4M":
            data[:, 0:1] = 0
            data[:, -1:] = 0
            data[0:1, :] = 0
            data[-1:, :] = 0
            data[:, 1029:1041] = 0
            data[513:552, :] = 0
            data[1064:1103, :] = 0
            data[1615:1654, :] = 0

            mask[:, 0:1] = 1
            mask[:, -1:] = 1
            mask[0:1, :] = 1
            mask[-1:, :] = 1
            mask[:, 1029:1041] = 1
            mask[513:552, :] = 1
            mask[1064:1103, :] = 1
            mask[1615:1654, :] = 1

            # mask hot pixels, 4000000000 for the Eiger4M
            mask[data > 4000000000 * nb_img] = 1
            data[data > 4000000000 * nb_img] = 0

        elif self.name == "Maxipix":
            data[:, 255:261] = 0
            data[255:261, :] = 0

            mask[:, 255:261] = 1
            mask[255:261, :] = 1

            # mask hot pixels
            mask[data > 1e6 * nb_img] = 1
            data[data > 1e6 * nb_img] = 0

        elif self.name == "Merlin":
            data[:, 255:260] = 0
            data[255:260, :] = 0

            mask[:, 255:260] = 1
            mask[255:260, :] = 1

            # mask hot pixels
            mask[data > 1e6 * nb_img] = 1
            data[data > 1e6 * nb_img] = 0

        elif self.name == "Timepix":
            pass  # no gaps

        return data, mask


class Diffractometer:
    """
    Base class for defining diffractometers. The frame used is the laboratory frame with the CXI convention
    (z downstream, y vertical up, x outboard).

    :param sample_offsets: list or tuple of three angles in degrees, corresponding to the offsets of each of the sample
     circles (the offset for the most outer circle should be at index 0).
     Convention: the sample offsets will be subtracted to measurement the motor values.
    :param sample_circles: list of sample circles from outer to inner (e.g. mu eta chi phi),
     expressed using a valid pattern within {'x+', 'x-', 'y+', 'y-', 'z+', 'z-'}. For example: ['y+' ,'x-', 'z-', 'y+']
    :param detector_circles: list of detector circles from outer to inner (e.g. gamma delta),
     expressed using a valid pattern within {'x+', 'x-', 'y+', 'y-', 'z+', 'z-'}. For example: ['y+', 'x-']
    """

    valid_circles = {
        "x+",
        "x-",
        "y+",
        "y-",
        "z+",
        "z-",
    }  # + counter-clockwise, - clockwise
    valid_names = {"sample": "_sample_circles", "detector": "_detector_circles"}

    def __init__(self, sample_offsets, sample_circles=(), detector_circles=()):
        self.sample_circles = sample_circles
        self.detector_circles = detector_circles
        self.sample_offsets = sample_offsets

    @property
    def detector_circles(self):
        """
        List of detector circles from outer to inner (e.g. gamma delta), expressed using a valid pattern within
        {'x+', 'x-', 'y+', 'y-', 'z+', 'z-'}. For example: ['y+' ,'x-', 'z-', 'y+']. Convention: CXI convetion
        (z downstream, y vertical up, x outboard), + for a counter-clockwise rotation, - for a clockwise rotation.
        """
        return self._detector_circles

    @detector_circles.setter
    def detector_circles(self, value):
        valid.valid_container(
            value,
            container_types=(tuple, list),
            min_length=0,
            item_types=str,
            name="Diffractometer.detector_circles",
        )
        if any(val not in self.valid_circles for val in value):
            raise ValueError(
                f"Invalid circle value encountered in detector_circles, valid are {self.valid_circles}"
            )
        self._detector_circles = list(value)

    @property
    def sample_circles(self):
        """
        List of sample circles from outer to inner (e.g. mu eta chi phi), expressed using a valid pattern within
        {'x+', 'x-', 'y+', 'y-', 'z+', 'z-'}. For example: ['y+' ,'x-', 'z-', 'y+']. Convention: CXI convetion
        (z downstream, y vertical up, x outboard), + for a counter-clockwise rotation, - for a clockwise rotation.
        """
        return self._sample_circles

    @sample_circles.setter
    def sample_circles(self, value):
        valid.valid_container(
            value,
            container_types=(tuple, list),
            min_length=0,
            item_types=str,
            name="Diffractometer.sample_circles",
        )
        if any(val not in self.valid_circles for val in value):
            raise ValueError(
                f"Invalid circle value encountered in sample_circles, valid are {self.valid_circles}"
            )
        self._sample_circles = list(value)

    @property
    def sample_offsets(self):
        """
        List or tuple of three angles in degrees, corresponding to the offsets of each of the sample circles
        (the offset for the most outer circle should be at index 0).
        Convention: the sample offsets will be subtracted to measurement the motor values.
        """
        return self._sample_offsets

    @sample_offsets.setter
    def sample_offsets(self, value):
        nb_circles = len(self.__getattribute__(self.valid_names["sample"]))
        if value is None:
            value = (0,) * nb_circles
        valid.valid_container(
            value,
            container_types=(tuple, list, np.ndarray),
            length=nb_circles,
            item_types=Real,
            name="Diffractometer.sample_offsets",
        )
        self._sample_offsets = value

    def add_circle(self, stage_name, index, circle):
        """
        Add a circle to the list of circles (the most outer circle should be at index 0).

        :param stage_name: supported stage name, 'sample' or 'detector'
        :param index: index where to put the circle in the list
        :param circle: valid circle in {'x+', 'x-', 'y+', 'y-', 'z+', 'z-'}.
         + for a counter-clockwise rotation, - for a clockwise rotation.
        """
        self.valid_name(stage_name)
        nb_circles = len(self.__getattribute__(self.valid_names[stage_name]))
        valid.valid_item(
            index,
            allowed_types=int,
            min_included=0,
            max_included=nb_circles,
            name="index",
        )
        if circle not in self.valid_circles:
            raise ValueError(
                f"{circle} is not in the list of valid circles:"
                f" {list(self.valid_circles)}"
            )
        self.__getattribute__(self.valid_names[stage_name]).insert(index, circle)

    def flatten_sample(
        self,
        arrays,
        voxel_size,
        angles,
        q_com,
        rocking_angle,
        central_angle=None,
        fill_value=0,
        is_orthogonal=True,
        reciprocal_space=False,
        debugging=False,
        **kwargs,
    ):
        """
        Rotate arrays such that all circles of the sample stage are at their zero position.

        :param arrays: tuple of 3D real arrays of the same shape.
        :param voxel_size: tuple, voxel size of the 3D array in z, y, and x (CXI convention)
        :param angles: tuple of angular values in degrees, one for each circle of the sample stage
        :param q_com: diffusion vector of the center of mass of the Bragg peak, expressed in an orthonormal frame x y z
        :param rocking_angle: angle which is tilted during the rocking curve in {'outofplane', 'inplane'}
        :param central_angle: if provided, angle to be used in the calculation of the rotation matrix for the rocking
         angle. If None, it will be defined as the angle value at the middle of the rocking curve.
        :param fill_value: tuple of numeric values used in the RegularGridInterpolator for points outside of the
         interpolation domain. The length of the tuple should be equal to the number of input arrays.
        :param is_orthogonal: set to True is the frame is orthogonal, False otherwise. Used for plot labels.
        :param reciprocal_space: True if the data is in reciprocal space, False otherwise. Used for plot labels.
        :param debugging: tuple of booleans of the same length as the number of input arrays, True to see plots before
         and after rotation
        :param kwargs:
         - 'title': tuple of strings, titles for the debugging plots, same length as the number of arrays
         - 'scale': tuple of strings (either 'linear' or 'log'), scale for the debugging plots, same length as the
           number of arrays
         - width_z: size of the area to plot in z (axis 0), centered on the middle of the initial array
         - width_y: size of the area to plot in y (axis 1), centered on the middle of the initial array
         - width_x: size of the area to plot in x (axis 2), centered on the middle of the initial array
        :return: a rotated array (if a single array was provided) or a tuple of rotated arrays (same length as the
         number of input arrays)
        """
        # check that arrays is a tuple of 3D arrays
        if isinstance(arrays, np.ndarray):
            arrays = (arrays,)
        valid.valid_container(
            arrays,
            container_types=(tuple, list),
            item_types=np.ndarray,
            min_length=1,
            name="arrays",
        )
        if any(array.ndim != 3 for array in arrays):
            raise ValueError("all arrays should be 3D ndarrays of the same shape")
        ref_shape = arrays[0].shape
        if any(array.shape != ref_shape for array in arrays):
            raise ValueError("all arrays should be 3D ndarrays of the same shape")

        # check few parameters, the rest will be validated in rotate_crystal
        valid.valid_container(
            q_com,
            container_types=(tuple, list, np.ndarray),
            length=3,
            item_types=Real,
            name="q_com",
        )
        if np.linalg.norm(q_com) == 0:
            raise ValueError("the norm of q_com is zero")
        nb_circles = len(self._sample_circles)
        valid.valid_container(
            angles, container_types=(tuple, list), length=nb_circles, name="angles"
        )
        valid.valid_item(
            central_angle, allowed_types=Real, allow_none=True, name="central_angle"
        )
        # find the index of the circle which corresponds to the rocking angle
        rocking_circle = self.get_rocking_circle(
            rocking_angle=rocking_angle, stage_name="sample", angles=angles
        )

        # get the relevant angle within the rocking circle. The reference point when orthogonalizing if the center of
        # the array, but we do not know to which angle it corresponds if the data was cropped.
        if central_angle is None:
            print(
                "central_angle=None, using the angle at half of the rocking curve for the calculation of the "
                f"rotation matrix"
            )
            nb_steps = len(angles[rocking_circle])
            central_angle = angles[rocking_circle][int(nb_steps // 2)]

        # use this angle in the calculation of the rotation matrix
        angles = list(angles)
        angles[rocking_circle] = central_angle
        print(
            f"sample stage circles: {self._sample_circles}\nsample stage angles:  {angles}"
        )

        # check that all angles are Real, not encapsulated in a list or an array
        for idx, angle in enumerate(angles):
            if not isinstance(angle, Real):  # list/tuple or ndarray, cannot be None
                if len(angle) != 1:
                    raise ValueError(
                        "A list of angles was provided instead of a single value"
                    )
                angles[idx] = angle[0]

        # calculate the rotation matrix
        rotation_matrix = self.rotation_matrix(stage_name="sample", angles=angles)

        # rotate the arrays
        rotated_arrays = util.rotate_crystal(
            arrays=arrays,
            rotation_matrix=rotation_matrix,
            voxel_size=voxel_size,
            fill_value=fill_value,
            debugging=debugging,
            is_orthogonal=is_orthogonal,
            reciprocal_space=reciprocal_space,
            **kwargs,
        )
        rotated_q = util.rotate_vector(
            vectors=q_com, rotation_matrix=np.linalg.inv(rotation_matrix)
        )
        return rotated_arrays, rotated_q

    def get_circles(self, stage_name):
        """
        Return the list of circles for the stage.

        :param stage_name: supported stage name, 'sample' or 'detector'
        """
        self.valid_name(stage_name)
        return self.__getattribute__(self.valid_names[stage_name])

    def get_rocking_circle(self, rocking_angle, stage_name, angles):
        """
        Find the index of the circle which corresponds to the rocking angle.

        :param rocking_angle: angle which is tilted during the rocking curve in {'outofplane', 'inplane'}
        :param stage_name: supported stage name, 'sample' or 'detector'
        :param angles: tuple of angular values in degrees, one for each circle of the sample stage
        :return: the index of the rocking circles in the list of angles
        """
        # check parameters
        if rocking_angle not in {"outofplane", "inplane"}:
            raise ValueError(
                f"Invalid value {rocking_angle} for rocking_angle,"
                f' should be either "inplane" or "outofplane"'
            )
        self.valid_name(stage_name)
        valid.valid_container(angles, container_types=(tuple, list), name="angles")
        nb_circles = len(angles)

        # find which angles were scanned
        candidate_circles = set()
        for idx in range(nb_circles):
            if not isinstance(angles[idx], Real) and len(angles[idx]) > 1:
                # not a number, hence a tuple/list/ndarray (cannot be None)
                candidate_circles.add(idx)

        # exclude arrays with identical values
        wrong_motors = []
        for idx in candidate_circles:
            if (
                angles[idx][1:] - angles[idx][:-1]
            ).mean() < 0.0001:  # motor not scanned, noise in the position readings
                wrong_motors.append(idx)
        candidate_circles.difference_update(wrong_motors)

        # check that there is only one candidate remaining
        if len(candidate_circles) > 1:
            raise ValueError("Several circles were identified as scanned motors")
        if len(candidate_circles) == 0:
            raise ValueError("No circle was identified as scanned motor")
        index_circle = next(iter(candidate_circles))

        # check that the rotation axis corresponds to the one definec by rocking_angle
        circles = self.__getattribute__(self.valid_names[stage_name])
        if rocking_angle == "inplane":
            if circles[index_circle][0] != "y":
                raise ValueError(
                    f"The identified circle '{circles[index_circle]}' is incompatible "
                    f"with the parameter '{rocking_angle}'"
                )
        else:  # 'outofplane'
            if circles[index_circle][0] != "x":
                raise ValueError(
                    f"The identified circle '{circles[index_circle]}' is incompatible "
                    f"with the parameter '{rocking_angle}'"
                )
        return index_circle

    def goniometer_values(self, **kwargs):
        """
        This method is beamline dependent and should be implemented in the child classes.

        :param kwargs: beamline_specific parameters
        :return: a list of motor positions
        """
        raise NotImplementedError(
            "This method is beamline specific and must be implemented in the child class"
        )

    def motor_positions(self, **kwargs):
        """
        This method is beamline dependent and should be implemented in the child classes.

        :param kwargs: beamline_specific parameters
        :return: the diffractometer motors positions for the particular setup.
        """
        raise NotImplementedError(
            "This method is beamline specific and must be implemented in the child class"
        )

    def remove_circle(self, stage_name, index):
        """
        Remove the circle at index from the list of sample circles.

        :param stage_name: supported stage name, 'sample' or 'detector'
        :param index: index of the circle to be removed from the list
        """
        if stage_name not in self.valid_names.keys():
            raise NotImplementedError(
                f"'{stage_name}' is not implemented,"
                f" available are {list(self.valid_names.keys())}"
            )
        nb_circles = len(self.__getattribute__(self.valid_names[stage_name]))
        if nb_circles > 0:
            valid.valid_item(
                index,
                allowed_types=int,
                min_included=0,
                max_included=nb_circles - 1,
                name="index",
            )
            del self.__getattribute__(self.valid_names[stage_name])[index]

    def rotation_matrix(self, stage_name, angles):
        """
        Calculate the 3x3 rotation matrix given by the list of angles corresponding to the stage circles.

        :param stage_name: supported stage name, 'sample' or 'detector'
        :param angles: list of angular values in degrees for the stage circles during the measurement
        :return: the rotation matrix as a numpy ndarray of shape (3, 3)
        """
        self.valid_name(stage_name)
        nb_circles = len(self.__getattribute__(self.valid_names[stage_name]))
        if isinstance(angles, Number):
            angles = (angles,)
        valid.valid_container(
            angles,
            container_types=(list, tuple, np.ndarray),
            length=nb_circles,
            item_types=Real,
            name="angles",
        )

        # create a list of rotation matrices corresponding to the circles, index 0 corresponds to the most outer circle
        rotation_matrices = [
            RotationMatrix(circle, angles[idx]).get_matrix()
            for idx, circle in enumerate(
                self.__getattribute__(self.valid_names[stage_name])
            )
        ]

        # calculate the total tranformation matrix by rotating back from outer circles to inner circles
        return np.array(reduce(lambda x, y: np.matmul(x, y), rotation_matrices))

    def valid_name(self, stage_name):
        """
        Check if the stage is defined.

        :param stage_name: supported stage name, 'sample' or 'detector'
        """
        if stage_name not in self.valid_names.keys():
            raise NotImplementedError(
                f"'{stage_name}' is not implemented,"
                f" available are {list(self.valid_names.keys())}"
            )


class Diffractometer34ID(Diffractometer):
    """
    34ID goniometer, 2S+2D (sample: theta (inplane), phi (out of plane)   /   detector: delta (inplane), gamma).
    The laboratory frame uses the CXI convention (z downstream, y vertical up, x outboard).
    """

    def __init__(self, sample_offsets):
        super().__init__(
            sample_circles=["y+", "x+"],
            detector_circles=["y+", "x-"],
            sample_offsets=sample_offsets,
        )

    def goniometer_values(self, setup, stage_name="bcdi", **kwargs):
        """
        Extract goniometer motor positions for a BCDI rocking scan.

        :param setup: the experimental setup: Class Setup
        :param stage_name: supported stage name, 'bcdi', 'sample' or 'detector'
        :return: a tuple of angular values in degrees, depending on stage_name:
         - 'bcdi': (rocking angular step, grazing incidence angles, inplane detector angle, outofplane detector angle).
           The grazing incidence angles are the positions of circles below the rocking circle.
         - 'sample': tuple of angular values for the sample circles, from the most outer to the most inner circle
         - 'detector': tuple of angular values for the detector circles, from the most outer to the most inner circle
        """
        # check some parameter
        if not isinstance(setup, Setup):
            raise TypeError("setup should be of type experiment.experiment_utils.Setup")
        if stage_name not in {"bcdi", "sample", "detector"}:
            raise ValueError(f"Invalid value {stage_name} for 'stage_name' parameter")

        # load the motor positions
        theta, phi, delta, gamma = self.motor_positions(setup=setup)

        # define the circles of interest for BCDI
        if setup.rocking_angle == "inplane":
            grazing = None  # phi is above theta at 34ID
            tilt, inplane, outofplane = (
                theta,
                delta,
                gamma,
            )  # theta is the rotation around the vertical axis
        elif setup.rocking_angle == "outofplane":
            grazing = (theta,)
            tilt, inplane, outofplane = (
                phi,
                delta,
                gamma,
            )  # phi is the incident angle at 34ID
        else:
            raise ValueError('Wrong value for "rocking_angle" parameter')

        # 34ID-C goniometer, 2S+2D (sample: theta (inplane), phi (out of plane)   detector: delta (inplane), gamma)
        sample_angles = (theta, phi)
        detector_angles = (delta, gamma)

        if stage_name == "sample":
            return sample_angles
        if stage_name == "detector":
            return detector_angles
        return tilt, grazing, inplane, outofplane

    def motor_positions(self, setup):
        """
        Load the scan data and extract motor positions.

        :param setup: the experimental setup: Class SetupPreprocessing()
        :return: (theta, phi, delta, gamma) motor positions
        """
        if not setup.custom_scan:
            raise NotImplementedError("Only custom_scan implemented for 34ID")
        theta = setup.custom_motors["theta"]
        phi = setup.custom_motors["phi"]
        gamma = setup.custom_motors["gamma"]
        delta = setup.custom_motors["delta"]

        return theta, phi, delta, gamma


class DiffractometerCRISTAL(Diffractometer):
    """
    CRISTAL goniometer, 2S+2D (sample: mgomega, mgphi / detector: gamma, delta).
    The laboratory frame uses the CXI convention (z downstream, y vertical up, x outboard).
    """

    def __init__(self, sample_offsets):
        super().__init__(
            sample_circles=["x-", "y+"],
            detector_circles=["y+", "x-"],
            sample_offsets=sample_offsets,
        )

    def goniometer_values(self, logfile, setup, stage_name="bcdi", **kwargs):
        """
        Extract goniometer motor positions for a BCDI rocking scan.

        :param logfile: file containing the information about the scan and image numbers (specfile, .fio...)
        :param setup: the experimental setup: Class Setup
        :param stage_name: supported stage name, 'bcdi', 'sample' or 'detector'
        :return: a tuple of angular values in degrees, depending on stage_name:
         - 'bcdi': (rocking angular step, grazing incidence angles, inplane detector angle, outofplane detector angle).
           The grazing incidence angles are the positions of circles below the rocking circle.
         - 'sample': tuple of angular values for the sample circles, from the most outer to the most inner circle
         - 'detector': tuple of angular values for the detector circles, from the most outer to the most inner circle
        """
        # check some parameter
        if not isinstance(setup, Setup):
            raise TypeError("setup should be of type experiment.experiment_utils.Setup")
        if stage_name not in {"bcdi", "sample", "detector"}:
            raise ValueError(f"Invalid value {stage_name} for 'stage_name' parameter")

        # load the motor positions
        mgomega, mgphi, gamma, delta, energy = self.motor_positions(logfile, setup)

        # define the circles of interest for BCDI
        if setup.rocking_angle == "outofplane":  # mgomega rocking curve
            grazing = None  # nothing below mgomega at CRISTAL
            tilt, inplane, outofplane = mgomega, gamma[0], delta[0]
        elif setup.rocking_angle == "inplane":  # phi rocking curve
            grazing = (mgomega[0],)
            tilt, inplane, outofplane = mgphi, gamma[0], delta[0]
        else:
            raise ValueError('Wrong value for "rocking_angle" parameter')

        # CRISTAL goniometer, 2S+2D (sample: mgomega, mgphi / detector: gamma, delta)
        sample_angles = (mgomega, mgphi)
        detector_angles = (gamma, delta)

        if stage_name == "sample":
            return sample_angles
        if stage_name == "detector":
            return detector_angles
        return tilt, grazing, inplane, outofplane

    def motor_positions(self, logfile, setup, **kwargs):
        """
        Load the scan data and extract motor positions. It will look for the correct entry 'rocking_angle' in the
         dictionary Setup.actuators, and use the default entry otherwise.

        :param logfile: h5py File object of CRISTAL .nxs scan file
        :param setup: the experimental setup: Class SetupPreprocessing()
        :param kwargs:
         - frames_logical: array of 0 (frame non used) or 1 (frame used) or -1 (padded frame). The initial length is
           equal to the number of measured frames. In case of data padding, the length changes.
        :return: (mgomega, mgphi, gamma, delta) motor positions
        """
        # check and load kwargs
        valid.valid_kwargs(
            kwargs=kwargs,
            allowed_kwargs={"follow_bragg", "frames_logical"},
            name="kwargs",
        )
        frames_logical = kwargs.get("frames_logical")
        if frames_logical is not None:
            if not isinstance(frames_logical, (list, np.ndarray)):
                raise TypeError("frames_logical should be a list/array")
            if not all(val in {-1, 0, 1} for val in frames_logical):
                raise ValueError(
                    "frames_logical should be a list of values in {-1, 0, 1}"
                )

        if not setup.custom_scan:
            group_key = list(logfile.keys())[0]
            energy = (
                self.cristal_load_motor(
                    datafile=logfile,
                    root="/" + group_key + "/CRISTAL/",
                    actuator_name="Monochromator",
                    field_name="energy",
                )
                * 1000
            )  # in eV
            if abs(energy - setup.energy) > 1:  # difference larger than 1 eV
                print(
                    f"\nWarning: user-defined energy = {setup.energy:.1f} eV different "
                    f"from the energy recorded in the datafile = {energy[0]:.1f} eV\n"
                )

            scanned_motor = self.cristal_load_motor(
                datafile=logfile,
                root="/" + group_key,
                actuator_name="scan_data",
                field_name=setup.actuators.get("rocking_angle", "actuator_1_1"),
            )
            if frames_logical is not None:
                scanned_motor = scanned_motor[
                    np.nonzero(frames_logical)
                ]  # exclude positions corresponding to empty frames

            if setup.rocking_angle == "outofplane":
                mgomega = scanned_motor  # mgomega is scanned
                mgphi = self.cristal_load_motor(
                    datafile=logfile,
                    root="/" + group_key + "/CRISTAL/",
                    actuator_name="i06-c-c07-ex-mg_phi",
                    field_name="position",
                )
                delta = self.cristal_load_motor(
                    datafile=logfile,
                    root="/" + group_key + "/CRISTAL/Diffractometer/",
                    actuator_name="I06-C-C07-EX-DIF-DELTA",
                    field_name="position",
                )
                gamma = self.cristal_load_motor(
                    datafile=logfile,
                    root="/" + group_key + "/CRISTAL/Diffractometer/",
                    actuator_name="I06-C-C07-EX-DIF-GAMMA",
                    field_name="position",
                )

            elif setup.rocking_angle == "inplane":
                mgphi = scanned_motor  # mgphi is scanned
                mgomega = self.cristal_load_motor(
                    datafile=logfile,
                    root="/" + group_key + "/CRISTAL/",
                    actuator_name="i06-c-c07-ex-mg_omega",
                    field_name="position",
                )
                delta = self.cristal_load_motor(
                    datafile=logfile,
                    root="/" + group_key + "/CRISTAL/Diffractometer/",
                    actuator_name="I06-C-C07-EX-DIF-DELTA",
                    field_name="position",
                )
                gamma = self.cristal_load_motor(
                    datafile=logfile,
                    root="/" + group_key + "/CRISTAL/Diffractometer/",
                    actuator_name="I06-C-C07-EX-DIF-GAMMA",
                    field_name="position",
                )
            else:
                raise ValueError('Wrong value for "rocking_angle" parameter')

            # remove user-defined sample offsets (sample: mgomega, mgphi)
            mgomega = mgomega - self.sample_offsets[0]
            mgphi = mgphi - self.sample_offsets[1]

        else:  # manually defined custom scan
            mgomega = setup.custom_motors["mgomega"]
            delta = setup.custom_motors["delta"]
            gamma = setup.custom_motors["gamma"]
            mgphi = setup.custom_motors.get("mgphi", None)
            energy = setup.custom_motors["energy", setup.energy]

        # check if mgomega needs to be divided by 1e6 (data taken before the implementation of the correction)
        if isinstance(mgomega, Real) and abs(mgomega) > 360:
            mgomega = mgomega / 1e6
        elif isinstance(mgomega, (tuple, list, np.ndarray)) and any(
            abs(val) > 360 for val in mgomega
        ):
            mgomega = mgomega / 1e6

        return mgomega, mgphi, gamma, delta, energy

    @staticmethod
    def cristal_load_motor(datafile, root, actuator_name, field_name):
        """
        Try to load the dataset at the defined entry and returns it. Patterns keep changing at CRISTAL.

        :param datafile: h5py File object of CRISTAL .nxs scan file
        :param root: string, path of the data up to the last subfolder (not included). This part is expected to
         not change over time
        :param actuator_name: string, name of the actuator (e.g. 'I06-C-C07-EX-DIF-KPHI'). Lowercase and uppercase will
         be tested when trying to load the data.
        :param field_name: name of the field under the actuator name (e.g. 'position')
        :return: the dataset if found or 0
        """
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
        if actuator_name not in datafile[root].keys():
            actuator_name = actuator_name.lower()
            if actuator_name not in datafile[root].keys():
                actuator_name = actuator_name.upper()
                if actuator_name not in datafile[root].keys():
                    print(
                        f"\nCould not find the entry for the actuator'{actuator_name}'"
                    )
                    print(
                        f"list of available actuators: {list(datafile[root].keys())}\n"
                    )
                    return 0

        # check if the field is a valid entry for the actuator
        try:
            dataset = datafile[root + "/" + actuator_name + "/" + field_name][:]
        except KeyError:  # try lowercase
            try:
                dataset = datafile[
                    root + "/" + actuator_name + "/" + field_name.lower()
                ][:]
            except KeyError:  # try uppercase
                try:
                    dataset = datafile[
                        root + "/" + actuator_name + "/" + field_name.upper()
                    ][:]
                except KeyError:  # nothing else that we can do
                    print(
                        f"\nCould not find the field '{field_name}' in the actuator'{actuator_name}'"
                    )
                    print(
                        f"list of available fields: {list(datafile[root + '/' + actuator_name].keys())}\n"
                    )
                    return 0
        return dataset


class DiffractometerID01(Diffractometer):
    """
    ID01 goniometer, 3S+2D (sample: mu, eta, phi / detector: nu,del).
    The laboratory frame uses the CXI convention (z downstream, y vertical up, x outboard).
    """

    def __init__(self, sample_offsets):
        super().__init__(
            sample_circles=["y-", "x-", "y-"],
            detector_circles=["y-", "x-"],
            sample_offsets=sample_offsets,
        )

    def goniometer_values(
        self, logfile, scan_number, setup, stage_name="bcdi", **kwargs
    ):
        """
        Extract goniometer motor positions for a BCDI rocking scan.

        :param logfile: file containing the information about the scan and image numbers (specfile, .fio...)
        :param scan_number: the scan number to load
        :param setup: the experimental setup: Class Setup
        :param stage_name: supported stage name, 'bcdi', 'sample' or 'detector'
        :param kwargs:
         - 'frames_logical': array of 0 (frame non used) or 1 (frame used) or -1 (padded frame). The initial length is
           equal to the number of measured frames. In case of data padding, the length changes.
         - 'follow_bragg': boolean, True for energy scans where the detector position is changed during the scan to
           follow the Bragg peak.
        :return: a tuple of angular values in degrees, depending on stage_name:
         - 'bcdi': (rocking angular step, grazing incidence angles, inplane detector angle, outofplane detector angle).
           The grazing incidence angles are the positions of circles below the rocking circle.
         - 'sample': tuple of angular values for the sample circles, from the most outer to the most inner circle
         - 'detector': tuple of angular values for the detector circles, from the most outer to the most inner circle
        """
        # load kwargs
        follow_bragg = kwargs.get("follow_bragg", False)
        frames_logical = kwargs.get("frames_logical")
        valid.valid_item(follow_bragg, allowed_types=bool, name="follow_bragg")
        if frames_logical is not None:
            if not isinstance(frames_logical, (list, np.ndarray)):
                raise TypeError("frames_logical should be a list/array")
            if not all(val in {-1, 0, 1} for val in frames_logical):
                raise ValueError(
                    "frames_logical should be a list of values in {-1, 0, 1}"
                )

        # check some parameter
        if not isinstance(setup, Setup):
            raise TypeError("setup should be of type experiment.experiment_utils.Setup")
        valid.valid_item(
            scan_number, allowed_types=int, min_excluded=0, name="scan_number"
        )
        if stage_name not in {"bcdi", "sample", "detector"}:
            raise ValueError(f"Invalid value {stage_name} for 'stage_name' parameter")

        # load motor positions
        mu, eta, phi, nu, delta, energy, frames_logical = self.motor_positions(
            logfile=logfile,
            scan_number=scan_number,
            setup=setup,
            frames_logical=frames_logical,
            follow_bragg=follow_bragg,
        )

        # define the circles of interest for BCDI
        if setup.rocking_angle == "outofplane":  # eta rocking curve
            grazing = (mu,)  # mu below eta but not used at ID01
            tilt, inplane, outofplane = eta, nu, delta
        elif setup.rocking_angle == "inplane":  # phi rocking curve
            grazing = (mu, eta)  # mu below eta but not used at ID01
            tilt, inplane, outofplane = phi, nu, delta
        else:
            raise ValueError('Wrong value for "rocking_angle" parameter')

        # ID01 goniometer, 3S+2D (sample: eta, chi, phi / detector: nu,del)
        sample_angles = (mu, eta, phi)
        detector_angles = (nu, delta)

        if stage_name == "sample":
            return sample_angles
        if stage_name == "detector":
            return detector_angles
        return tilt, grazing, inplane, outofplane

    def motor_positions(self, logfile, scan_number, setup, **kwargs):
        """
        Load the scan data and extract motor positions.

        :param logfile: Silx SpecFile object containing the information about the scan and image numbers
        :param scan_number: the scan number to load
        :param setup: the experimental setup: Class SetupPreprocessing()
        :param kwargs:
         - 'frames_logical': array of 0 (frame non used) or 1 (frame used) or -1 (padded frame). The initial length is
           equal to the number of measured frames. In case of data padding, the length changes.
         - 'follow_bragg': boolean, True for energy scans where the detector position is changed during the scan to
           follow the Bragg peak.
        :return: (mu, eta, phi, nu, delta, energy) motor positions
        """
        # check and load kwargs
        valid.valid_kwargs(
            kwargs=kwargs,
            allowed_kwargs={"follow_bragg", "frames_logical"},
            name="kwargs",
        )
        follow_bragg = kwargs.get("follow_bragg", False)
        frames_logical = kwargs.get("frames_logical")
        valid.valid_item(follow_bragg, allowed_types=bool, name="follow_bragg")
        if frames_logical is not None:
            if not isinstance(frames_logical, (list, np.ndarray)):
                raise TypeError("frames_logical should be a list/array")
            if not all(val in {-1, 0, 1} for val in frames_logical):
                raise ValueError(
                    "frames_logical should be a list of values in {-1, 0, 1}"
                )

        energy = setup.energy  # will be overridden if setup.rocking_angle is 'energy'
        old_names = False
        if not setup.custom_scan:
            motor_names = logfile[str(scan_number) + ".1"].motor_names  # positioners
            motor_positions = logfile[
                str(scan_number) + ".1"
            ].motor_positions  # positioners
            labels = logfile[str(scan_number) + ".1"].labels  # motor scanned
            labels_data = logfile[str(scan_number) + ".1"].data  # motor scanned

            try:
                nu = motor_positions[motor_names.index("nu")]  # positioner
            except ValueError:
                print("'nu' not in the list, trying 'Nu'")
                nu = motor_positions[motor_names.index("Nu")]  # positioner
                print("Defaulting to old ID01 motor names")
                old_names = True

            if not old_names:
                mu = motor_positions[motor_names.index("mu")]  # positioner
            else:
                mu = motor_positions[motor_names.index("Mu")]  # positioner

            if follow_bragg:
                if not old_names:
                    delta = list(labels_data[labels.index("del"), :])  # scanned
                else:
                    delta = list(labels_data[labels.index("Delta"), :])  # scanned
            else:
                if not old_names:
                    delta = motor_positions[motor_names.index("del")]  # positioner
                else:
                    delta = motor_positions[motor_names.index("Delta")]  # positioner

            if setup.rocking_angle == "outofplane":
                if not old_names:
                    eta = labels_data[labels.index("eta"), :]
                    phi = motor_positions[motor_names.index("phi")]
                else:
                    eta = labels_data[labels.index("Eta"), :]
                    phi = motor_positions[motor_names.index("Phi")]
            elif setup.rocking_angle == "inplane":
                if not old_names:
                    phi = labels_data[labels.index("phi"), :]
                    eta = motor_positions[motor_names.index("eta")]
                else:
                    phi = labels_data[labels.index("Phi"), :]
                    eta = motor_positions[motor_names.index("Eta")]
            elif setup.rocking_angle == "energy":
                raw_energy = list(
                    labels_data[labels.index("energy"), :]
                )  # in kev, scanned
                if not old_names:
                    phi = motor_positions[motor_names.index("phi")]  # positioner
                    eta = motor_positions[motor_names.index("eta")]  # positioner
                else:
                    phi = motor_positions[motor_names.index("Phi")]  # positioner
                    eta = motor_positions[motor_names.index("Eta")]  # positioner

                nb_overlap = 0
                energy = raw_energy[:]
                if frames_logical is None:
                    frames_logical = np.ones(len(energy))
                for idx in range(len(raw_energy) - 1):
                    if (
                        raw_energy[idx + 1] == raw_energy[idx]
                    ):  # duplicated energy when undulator gap is changed
                        frames_logical[idx + 1] = 0
                        energy.pop(idx - nb_overlap)
                        if follow_bragg:
                            delta.pop(idx - nb_overlap)
                        nb_overlap = nb_overlap + 1
                energy = np.array(energy) * 1000.0  # switch to eV

            else:
                raise ValueError(
                    "Invalid rocking angle ", setup.rocking_angle, "for ID01"
                )

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

        return mu, eta, phi, nu, delta, energy, frames_logical


class DiffractometerNANOMAX(Diffractometer):
    """
    NANOMAX goniometer, 2S+2D (sample: theta, phi / detector: gamma,delta).
    The laboratory frame uses the CXI convention (z downstream, y vertical up, x outboard).
    """

    def __init__(self, sample_offsets):
        super().__init__(
            sample_circles=["x-", "y-"],
            detector_circles=["y-", "x-"],
            sample_offsets=sample_offsets,
        )

    def goniometer_values(self, logfile, setup, stage_name="bcdi", **kwargs):
        """
        Extract goniometer motor positions for a BCDI rocking scan.

        :param logfile: file containing the information about the scan and image numbers (specfile, .fio...)
        :param setup: the experimental setup: Class Setup
        :param stage_name: supported stage name, 'bcdi', 'sample' or 'detector'
        :return: a tuple of angular values in degrees, depending on stage_name:
         - 'bcdi': (rocking angular step, grazing incidence angles, inplane detector angle, outofplane detector angle).
           The grazing incidence angles are the positions of circles below the rocking circle.
         - 'sample': tuple of angular values for the sample circles, from the most outer to the most inner circle
         - 'detector': tuple of angular values for the detector circles, from the most outer to the most inner circle
        """
        # check some parameter
        if not isinstance(setup, Setup):
            raise TypeError("setup should be of type experiment.experiment_utils.Setup")
        if stage_name not in {"bcdi", "sample", "detector"}:
            raise ValueError(f"Invalid value {stage_name} for 'stage_name' parameter")

        # load the motor positions
        theta, phi, gamma, delta, energy, radius = self.motor_positions(
            logfile=logfile, setup=setup
        )

        # define the circles of interest for BCDI
        if setup.rocking_angle == "outofplane":  # theta rocking curve
            grazing = None  # nothing below theta at NANOMAX
            tilt, inplane, outofplane = theta, gamma, delta
        elif setup.rocking_angle == "inplane":  # phi rocking curve
            grazing = (theta,)
            tilt, inplane, outofplane = phi, gamma, delta
        else:
            raise ValueError('Wrong value for "rocking_angle" parameter')

        # NANOMAX goniometer, 2S+2D (sample: theta, phi / detector: gamma,delta)
        sample_angles = (theta, phi)
        detector_angles = (gamma, delta)

        if stage_name == "sample":
            return sample_angles
        if stage_name == "detector":
            return detector_angles
        return tilt, grazing, inplane, outofplane

    def motor_positions(self, logfile, setup):
        """
        Load the scan data and extract motor positions.

        :param logfile: Silx SpecFile object containing the information about the scan and image numbers
        :param setup: the experimental setup: Class SetupPreprocessing()
        :return: (theta, phi, gamma, delta, energy, radius) motor positions
        """
        if not setup.custom_scan:
            # Detector positions
            group_key = list(logfile.keys())[0]  # currently 'entry'

            # positionners
            delta = logfile["/" + group_key + "/snapshot/delta"][:]
            gamma = logfile["/" + group_key + "/snapshot/gamma"][:]
            radius = logfile["/" + group_key + "/snapshot/radius"][:]
            energy = logfile["/" + group_key + "/snapshot/energy"][:]

            if setup.rocking_angle == "inplane":
                try:
                    phi = logfile["/" + group_key + "/measurement/gonphi"][:]
                except KeyError:
                    raise KeyError(
                        'phi not in measurement data, check the parameter "rocking_angle"'
                    )
                theta = logfile["/" + group_key + "/snapshot/gontheta"][:]
            else:
                try:
                    theta = logfile["/" + group_key + "/measurement/gontheta"][:]
                except KeyError:
                    raise KeyError(
                        'theta not in measurement data, check the parameter "rocking_angle"'
                    )
                phi = logfile["/" + group_key + "/snapshot/gonphi"][:]

            # remove user-defined sample offsets (sample: theta, phi)
            theta = theta - self.sample_offsets[0]
            phi = phi - self.sample_offsets[1]

        else:  # manually defined custom scan
            theta = setup.custom_motors["theta"]
            phi = setup.custom_motors["phi"]
            delta = setup.custom_motors["delta"]
            gamma = setup.custom_motors["gamma"]
            radius = setup.custom_motors["radius"]
            energy = setup.custom_motors["energy"]

        return theta, phi, gamma, delta, energy, radius


class DiffractometerP10(Diffractometer):
    """
    P10 goniometer, 4S+2D (sample: mu, om, chi, phi / detector: gamma, delta).
    The laboratory frame uses the CXI convention (z downstream, y vertical up, x outboard).
    """

    def __init__(self, sample_offsets):
        super().__init__(
            sample_circles=["y+", "x-", "z+", "y-"],
            detector_circles=["y+", "x-"],
            sample_offsets=sample_offsets,
        )

    def goniometer_values(self, logfile, setup, stage_name="bcdi", **kwargs):
        """
        Extract goniometer motor positions for a BCDI rocking scan.

        :param logfile: file containing the information about the scan and image numbers (specfile, .fio...)
        :param setup: the experimental setup: Class Setup
        :param stage_name: supported stage name, 'bcdi', 'sample' or 'detector'
        :return: a tuple of angular values in degrees, depending on stage_name:
         - 'bcdi': (rocking angular step, grazing incidence angles, inplane detector angle, outofplane detector angle).
           The grazing incidence angles are the positions of circles below the rocking circle.
         - 'sample': tuple of angular values for the sample circles, from the most outer to the most inner circle
         - 'detector': tuple of angular values for the detector circles, from the most outer to the most inner circle
        """
        # check some parameter
        if not isinstance(setup, Setup):
            raise TypeError("setup should be of type experiment.experiment_utils.Setup")
        if stage_name not in {"bcdi", "sample", "detector"}:
            raise ValueError(f"Invalid value {stage_name} for 'stage_name' parameter")

        # load the motor positions
        mu, om, chi, phi, gamma, delta = self.motor_positions(
            logfile=logfile, setup=setup
        )

        # define the circles of interest for BCDI
        if setup.rocking_angle == "outofplane":  # om rocking curve
            grazing = (mu,)
            tilt, inplane, outofplane = om, gamma, delta
        elif setup.rocking_angle == "inplane":  # phi rocking curve
            grazing = (mu, om, chi)
            tilt, inplane, outofplane = phi, gamma, delta
        else:
            raise ValueError('Wrong value for "rocking_angle" parameter')

        # P10 goniometer, 4S+2D (sample: mu, omega, chi, phi / detector: gamma, delta)
        sample_angles = (mu, om, chi, phi)
        detector_angles = (gamma, delta)
        if stage_name == "sample":
            return sample_angles
        if stage_name == "detector":
            return detector_angles
        return tilt, grazing, inplane, outofplane

    def motor_positions(self, logfile, setup):
        """
        Load the .fio file from the scan and extract motor positions for P10 6-circle difractometer setup.

        :param logfile: path of the . fio file containing the information about the scan
        :param setup: the experimental setup: Class SetupPreprocessing()
        :return: (om, phi, chi, mu, gamma, delta) motor positions
        """
        if not setup.custom_scan:
            fio = open(logfile, "r")
            index_om = None
            index_phi = None
            om = []
            phi = []
            chi = None
            mu = None
            gamma = None
            delta = None

            fio_lines = fio.readlines()
            for line in fio_lines:
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

                if index_om is not None and util.is_numeric(words[0]):
                    # reading data and index_om is defined (outofplane case)
                    om.append(float(words[index_om]))
                if index_phi is not None and util.is_numeric(words[0]):
                    # reading data and index_phi is defined (inplane case)
                    phi.append(float(words[index_phi]))

            if setup.rocking_angle == "outofplane":
                om = np.asarray(om, dtype=float)
            else:  # phi
                phi = np.asarray(phi, dtype=float)

            fio.close()

            # remove user-defined sample offsets (sample: mu, om, chi, phi)
            mu = mu - self.sample_offsets[0]
            om = om - self.sample_offsets[1]
            chi = chi - self.sample_offsets[2]
            phi = phi - self.sample_offsets[3]

        else:  # manually defined custom scan
            om = setup.custom_motors["om"]
            chi = setup.custom_motors["chi"]
            phi = setup.custom_motors["phi"]
            delta = setup.custom_motors["delta"]
            gamma = setup.custom_motors["gamma"]
            mu = setup.custom_motors["mu"]
        return mu, om, chi, phi, gamma, delta


class DiffractometerSIXS(Diffractometer):
    """
    SIXS goniometer, 2S+3D (sample: beta, mu / detector: beta, gamma, del).
    The laboratory frame uses the CXI convention (z downstream, y vertical up, x outboard).
    """

    def __init__(self, sample_offsets):
        super().__init__(
            sample_circles=["x-", "y+"],
            detector_circles=["x-", "y+", "x-"],
            sample_offsets=sample_offsets,
        )

    def goniometer_values(self, logfile, setup, stage_name="bcdi", **kwargs):
        """
        Extract goniometer motor positions for a BCDI rocking scan.

        :param logfile: file containing the information about the scan and image numbers (specfile, .fio...)
        :param setup: the experimental setup: Class Setup
        :param stage_name: supported stage name, 'bcdi', 'sample' or 'detector'
        :param kwargs:
         - 'frames_logical': array of 0 (frame non used) or 1 (frame used) or -1 (padded frame). The initial length is
           equal to the number of measured frames. In case of data padding, the length changes.
        :return: a tuple of angular values in degrees, depending on stage_name:
         - 'bcdi': (rocking angular step, grazing incidence angles, inplane detector angle, outofplane detector angle).
           The grazing incidence angles are the positions of circles below the rocking circle.
         - 'sample': tuple of angular values for the sample circles, from the most outer to the most inner circle
         - 'detector': tuple of angular values for the detector circles, from the most outer to the most inner circle
        """
        # load kwargs
        frames_logical = kwargs.get("frames_logical")
        if frames_logical is not None:
            if not isinstance(frames_logical, (list, np.ndarray)):
                raise TypeError("frames_logical should be a list/array")
            if not all(val in {-1, 0, 1} for val in frames_logical):
                raise ValueError(
                    "frames_logical should be a list of values in {-1, 0, 1}"
                )

        # check some parameter
        if not isinstance(setup, Setup):
            raise TypeError("setup should be of type experiment.experiment_utils.Setup")
        if stage_name not in {"bcdi", "sample", "detector"}:
            raise ValueError(f"Invalid value {stage_name} for 'stage_name' parameter")

        # load the motor positions
        beta, mu, gamma, delta, frames_logical = self.motor_positions(
            logfile=logfile, setup=setup, frames_logical=frames_logical
        )
        # define the circles of interest for BCDI
        if setup.rocking_angle == "inplane":  # mu rocking curve
            grazing = (beta,)  # beta below the whole diffractomter at SIXS
            tilt, inplane, outofplane = mu, gamma, delta
        elif setup.rocking_angle == "outofplane":
            raise NotImplementedError(
                "outofplane rocking curve not implemented for SIXS"
            )
        else:
            raise ValueError("Out-of-plane rocking curve not implemented for SIXS")

        # SIXS goniometer, 2S+3D (sample: beta, mu / detector: beta, gamma, del)
        sample_angles = (beta, mu)
        detector_angles = (beta, gamma, delta)

        if stage_name == "sample":
            return sample_angles
        if stage_name == "detector":
            return detector_angles
        return tilt, grazing, inplane, outofplane

    def motor_positions(self, logfile, setup, **kwargs):
        """
        Load the scan data and extract motor positions.

        :param logfile: nxsReady Dataset object of SIXS .nxs scan file
        :param setup: the experimental setup: Class SetupPreprocessing()
        :param kwargs:
         - frames_logical: array of 0 (frame non used) or 1 (frame used) or -1 (padded frame). The initial length is
           equal to the number of measured frames. In case of data padding, the length changes.
        :return: (beta, mu, gamma, delta) motor positions and updated frames_logical
        """
        # check and load kwargs
        valid.valid_kwargs(
            kwargs=kwargs, allowed_kwargs={"frames_logical"}, name="kwargs"
        )
        frames_logical = kwargs.get("frames_logical")
        if frames_logical is not None:
            if not isinstance(frames_logical, (list, np.ndarray)):
                raise TypeError("frames_logical should be a list/array")
            if not all(val in {-1, 0, 1} for val in frames_logical):
                raise ValueError(
                    "frames_logical should be a list of values in {-1, 0, 1}"
                )

        if not setup.custom_scan:
            delta = logfile.delta[0]  # not scanned
            gamma = logfile.gamma[0]  # not scanned
            try:
                beta = logfile.basepitch[0]  # not scanned
            except AttributeError:  # data recorder changed after 11/03/2019
                try:
                    beta = logfile.beta[0]  # not scanned
                except AttributeError:  # the alias dictionnary was probably not provided
                    beta = 0

            temp_mu = logfile.mu[:]
            if frames_logical is None:
                frames_logical = np.ones(len(temp_mu))
            mu = np.zeros(
                (frames_logical != 0).sum()
            )  # first frame is duplicated for SIXS_2018
            nb_overlap = 0
            for idx in range(len(frames_logical)):
                if frames_logical[idx]:
                    mu[idx - nb_overlap] = temp_mu[idx]
                else:
                    nb_overlap = nb_overlap + 1

            # remove user-defined sample offsets (sample: beta, mu)
            beta = beta - self.sample_offsets[0]
            mu = mu - self.sample_offsets[1]

        else:  # manually defined custom scan
            beta = setup.custom_motors["beta"]
            delta = setup.custom_motors["delta"]
            gamma = setup.custom_motors["gamma"]
            mu = setup.custom_motors["mu"]
        return beta, mu, gamma, delta, frames_logical


class RotationMatrix:
    """
    Class defining a rotation matrix given the rotation axis and the angle.

    :param circle: circle in {'x+', 'x-', 'y+', 'y-', 'z+', 'z-'}. The letter represents the rotation axis.
     + for a counter-clockwise rotation, - for a clockwise rotation.
    :param angle: angular value in degrees to be used in the calculation of the rotation matrix
    """

    valid_circles = {
        "x+",
        "x-",
        "y+",
        "y-",
        "z+",
        "z-",
    }  # + counter-clockwise, - clockwise

    def __init__(self, circle, angle):
        self.angle = angle
        self.circle = circle

    @property
    def angle(self):
        """
        Angular value to be used in the calculation of the rotation matrix.
        """
        return self._angle

    @angle.setter
    def angle(self, value):
        valid.valid_item(value, allowed_types=Real, name="value")
        if np.isnan(value):
            raise ValueError("value is a nan")
        self._angle = value

    @property
    def circle(self):
        """
        Circle definition used for the calculation of the rotation matrix in {'x+', 'x-', 'y+', 'y-', 'z+', 'z-'}.
        + for a counter-clockwise rotation, - for a clockwise rotation.
        """
        return self._circle

    @circle.setter
    def circle(self, value):
        if value not in RotationMatrix.valid_circles:
            raise ValueError(
                f"{value} is not in the list of valid circles:"
                f" {list(RotationMatrix.valid_circles)}"
            )
        self._circle = value

    def get_matrix(self):
        """
        Calculate the rotation matric for a given circle and angle.

        :return: a numpy ndarray of shape (3, 3)
        """
        angle = self.angle * np.pi / 180  # convert from degrees to radians

        if self.circle[1] == "+":
            rot_dir = 1
        else:  # '-'
            rot_dir = -1

        if self.circle[0] == "x":
            matrix = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(angle), -rot_dir * np.sin(angle)],
                    [0, rot_dir * np.sin(angle), np.cos(angle)],
                ]
            )
        elif self.circle[0] == "y":
            matrix = np.array(
                [
                    [np.cos(angle), 0, rot_dir * np.sin(angle)],
                    [0, 1, 0],
                    [-rot_dir * np.sin(angle), 0, np.cos(angle)],
                ]
            )
        elif self.circle[0] == "z":
            matrix = np.array(
                [
                    [np.cos(angle), -rot_dir * np.sin(angle), 0],
                    [rot_dir * np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1],
                ]
            )
        else:
            raise ValueError(
                f"{self.circle} is not in the list of valid circles:"
                f" {list(RotationMatrix.valid_circles)}"
            )
        return matrix


class Setup:
    """
    Class for defining the experimental geometry.

    :param beamline: name of the beamline, among {'ID01','SIXS_2018','SIXS_2019','34ID','P10','CRISTAL','NANOMAX'}
    :param detector: an instance of the cass experiment_utils.Detector()
    :param beam_direction: direction of the incident X-ray beam in the frame (z downstream,y vertical up,x outboard)
    :param energy: energy setting of the beamline, in eV.
    :param distance: sample to detector distance, in m.
    :param outofplane_angle: vertical detector angle, in degrees.
    :param inplane_angle: horizontal detector angle, in degrees.
    :param tilt_angle: angular step of the rocking curve, in degrees.
    :param rocking_angle: angle which is tilted during the rocking curve in {'outofplane', 'inplane', 'energy'}
    :param grazing_angle: motor positions for the goniometer circles below the rocking angle. It should be a
     list/tuple of lenght 1 for out-of-plane rocking curves (the chi motor value) and length 2 for inplane rocking
     curves (the chi and omega/om/eta motor values).
    :param kwargs:
     - 'direct_beam': tuple of two real numbers indicating the position of the direct beam in pixels at zero
       detector angles.
     - 'filtered_data': boolean, True if the data and the mask to be loaded were already preprocessed.
     - 'custom_scan': boolean, True is the scan does not follow the beamline's usual directory format.
     - 'custom_images': list of images numbers when the scan does no follow the beamline's usual directory format.
     - 'custom_monitor': list of monitor values when the scan does no follow the beamline's usual directory format.
       The number of values should be equal to the number of elements in custom_images.
     - 'custom_motors': list of motor values when the scan does no follow the beamline's usual directory format.
     - 'sample_inplane': sample inplane reference direction along the beam at 0 angles in xrayutilities frame
       (x is downstream, y outboard, and z vertical up at zero incident angle).
     - 'sample_outofplane': surface normal of the sample at 0 angles in xrayutilities frame
       (x is downstream, y outboard, and z vertical up at zero incident angle).
     - 'sample_offsets': list or tuple of three angles in degrees, corresponding to the offsets of each of the sample
       circles (the offset for the most outer circle should be at index 0).
       Convention: the sample offsets will be subtracted to measurement the motor values.
     - 'offset_inplane': inplane offset of the detector defined as the outer angle in xrayutilities area detector
       calibration.
     - 'actuators': optional dictionary that can be used to define the entries corresponding to actuators in data files
       (useful at CRISTAL where the location of data keeps changing)
    """

    def __init__(
        self,
        beamline,
        detector=Detector("Dummy"),
        beam_direction=(1, 0, 0),
        energy=None,
        distance=None,
        outofplane_angle=None,
        inplane_angle=None,
        tilt_angle=None,
        rocking_angle=None,
        grazing_angle=None,
        **kwargs,
    ):

        valid.valid_kwargs(
            kwargs=kwargs,
            allowed_kwargs={
                "direct_beam",
                "filtered_data",
                "custom_scan",
                "custom_images",
                "custom_monitor",
                "custom_motors",
                "sample_inplane",
                "sample_outofplane",
                "sample_offsets",
                "offset_inplane",
                "actuators",
            },
            name="Setup.__init__",
        )

        # kwargs for preprocessing forward CDI data
        self.direct_beam = kwargs.get("direct_beam")
        # kwargs for loading and preprocessing data
        sample_offsets = kwargs.get("sample_offsets")
        self.filtered_data = kwargs.get("filtered_data", False)  # boolean
        self.custom_scan = kwargs.get("custom_scan", False)  # boolean
        self.custom_images = kwargs.get("custom_images")  # list or tuple
        self.custom_monitor = kwargs.get("custom_monitor")  # list or tuple
        self.custom_motors = kwargs.get("custom_motors")  # dictionnary
        self.actuators = kwargs.get("actuators", {})  # list or tuple
        # kwargs for xrayutilities, delegate the test on their values to xrayutilities
        self.sample_inplane = kwargs.get("sample_inplane", (1, 0, 0))
        self.sample_outofplane = kwargs.get("sample_outofplane", (0, 0, 1))
        self.offset_inplane = kwargs.get("offset_inplane", 0)

        # load positional arguments corresponding to instance properties
        self.beamline = beamline
        self.detector = detector
        self.beam_direction = beam_direction
        self.energy = energy
        self.distance = distance
        self.outofplane_angle = outofplane_angle
        self.inplane_angle = inplane_angle
        self.tilt_angle = tilt_angle
        self.rocking_angle = rocking_angle
        self.grazing_angle = grazing_angle

        # create the Diffractometer instance
        self._diffractometer = self.create_diffractometer(sample_offsets)

    @property
    def actuators(self):
        """
        Optional dictionary that can be used to define the entries corresponding to actuators in data files
        (useful at CRISTAL where the location of data keeps changing)
        """
        return self._actuators

    @actuators.setter
    def actuators(self, value):
        if value is None:
            value = {}
        valid.valid_container(
            value, container_types=dict, item_types=str, name="Setup.actuators"
        )
        self._actuators = value

    @property
    def beam_direction(self):
        """
        Direction of the incident X-ray beam in the frame (z downstream, y vertical up, x outboard).
        """
        return self._beam_direction

    @beam_direction.setter
    def beam_direction(self, value):
        valid.valid_container(
            value,
            container_types=(tuple, list),
            length=3,
            item_types=Real,
            name="Setup.beam_direction",
        )
        if np.linalg.norm(value) == 0:
            raise ValueError(
                "At least of component of beam_direction should be non null."
            )
        self._beam_direction = value / np.linalg.norm(value)

    @property
    def beam_direction_xrutils(self):
        """
        Direction of the incident X-ray beam in the frame of xrayutilities (x downstream, y outboard, z vertical up).
        """
        u, v, w = self._beam_direction  # (u downstream, v vertical up, w outboard)
        return u, w, v

    @property
    def beamline(self):
        """
        Name of the beamline.
        """
        return self._beamline

    @beamline.setter
    def beamline(self, value):
        if value not in {
            "ID01",
            "SIXS_2018",
            "SIXS_2019",
            "34ID",
            "P10",
            "CRISTAL",
            "NANOMAX",
        }:
            raise ValueError(f"Beamline {value} not supported")
        self._beamline = value

    @property
    def custom_images(self):
        """
        List of images numbers when the scan does no follow the beamline's usual directory format.
        """
        return self._custom_images

    @custom_images.setter
    def custom_images(self, value):
        if not self._custom_scan:
            self._custom_images = None
        else:
            valid.valid_container(
                value,
                container_types=(tuple, list),
                min_length=1,
                item_types=int,
                allow_none=True,
                name="Setup.custom_images",
            )
            self._custom_images = value

    @property
    def custom_monitor(self):
        """
        List of monitor values when the scan does no follow the beamline's usual directory format. The number of values
         should be equal to the number of elements in custom_images.
        """
        return self._custom_monitor

    @custom_monitor.setter
    def custom_monitor(self, value):
        if not self._custom_scan or not self._custom_images:
            self._custom_monitor = None
        else:
            if value is None:
                value = np.ones(len(self._custom_images))
            valid.valid_container(
                value,
                container_types=(tuple, list, np.ndarray),
                length=len(self._custom_images),
                item_types=Real,
                name="Setup.custom_monitor",
            )
            self._custom_monitor = value

    @property
    def custom_motors(self):
        """
        List of motor values when the scan does no follow the beamline's usual directory format.
        """
        return self._custom_motors

    @custom_motors.setter
    def custom_motors(self, value):
        if not self._custom_scan:
            self._custom_motors = None
        else:
            if not isinstance(value, dict):
                raise TypeError(
                    'custom_motors should be a dictionnary of "motor_name": motor_positions pairs'
                )
            self._custom_motors = value

    @property
    def custom_scan(self):
        """
        Boolean, True is the scan does not follow the beamline's usual directory format.
        """
        return self._custom_scan

    @custom_scan.setter
    def custom_scan(self, value):
        if not isinstance(value, bool):
            raise TypeError("custom_scan should be a boolean")
        self._custom_scan = value

    @property
    def detector(self):
        """
        Detector instance
        """
        return self._detector

    @detector.setter
    def detector(self, value):
        if not isinstance(value, Detector):
            raise TypeError("value should be an instance of Detector")
        self._detector = value

    @property
    def detector_hor(self):
        """
        Defines the horizontal detector orientation for xrayutilities depending on the beamline.
         The frame convention of xrayutilities is the following: x downstream, y outboard, z vertical up.
        """
        if self.beamline in {"ID01", "SIXS_2018", "SIXS_2019", "CRISTAL", "NANOMAX"}:
            # we look at the detector from downstream, detector X along the outboard direction
            return "y+"
        # we look at the detector from upstream, detector X opposite to the outboard direction
        return "y-"

    @property
    def detector_ver(self):
        """
        Defines the vertical detector orientation for xrayutilities depending on the beamline.
         The frame convention of xrayutilities is the following: x downstream, y outboard, z vertical up.
        """
        if self.beamline in {
            "ID01",
            "SIXS_2018",
            "SIXS_2019",
            "CRISTAL",
            "NANOMAX",
            "P10",
            "34ID",
        }:
            # origin is at the top, detector Y along vertical down
            return "z-"
        return "z+"

    @property
    def diffractometer(self):
        """
        Return the diffractometer instance.
        """
        return self._diffractometer

    @property
    def direct_beam(self):
        """
        Tuple of two real numbers indicating the position of the direct beam in pixels at zero detector angles.
        """
        return self._direct_beam

    @direct_beam.setter
    def direct_beam(self, value):
        if value is not None:
            valid.valid_container(
                value,
                container_types=(tuple, list),
                length=2,
                item_types=Real,
                name="Setup.direct_beam",
            )
        self._direct_beam = value

    @property
    def distance(self):
        """
        Sample to detector distance, in m
        """
        return self._distance

    @distance.setter
    def distance(self, value):
        if value is None:
            self._distance = value
        elif not isinstance(value, Real):
            raise TypeError("distance should be a number in m")
        elif value <= 0:
            raise ValueError("distance should be a strictly positive number in m")
        else:
            self._distance = value

    @property
    def energy(self):
        """
        Energy setting of the beamline, in eV.
        """
        return self._energy

    @energy.setter
    def energy(self, value):
        if value is None:
            self._energy = value
        elif isinstance(value, Real):
            if value <= 0:
                raise ValueError("energy should be strictly positive, in eV")
            self._energy = value
        elif isinstance(value, (list, tuple, np.ndarray)):
            if len(value) == 0:
                raise ValueError(
                    "energy should be a number or a non-empty list of numbers in eV"
                )
            if any(val <= 0 for val in value):
                raise ValueError("energy should be strictly positive, in eV")
            self._energy = value
        else:
            raise TypeError("energy should be a number or a list of numbers, in eV")

    @property
    def exit_wavevector(self):
        """
        Calculate the exit wavevector kout depending on the setup parameters, in the laboratory frame (z downstream,
         y vertical, x outboard). The unit is 1/m

        :return: kout vector
        """
        kout = np.zeros(3)

        if self.beamline in {"SIXS_2018", "SIXS_2019"}:
            # gamma is anti-clockwise
            kout = (
                2
                * np.pi
                / self.wavelength
                * np.array(
                    [
                        np.cos(np.pi * self.inplane_angle / 180)
                        * np.cos(np.pi * self.outofplane_angle / 180),  # z
                        np.sin(np.pi * self.outofplane_angle / 180),  # y
                        np.sin(np.pi * self.inplane_angle / 180)
                        * np.cos(np.pi * self.outofplane_angle / 180),
                    ]
                )
            )  # x
        elif self.beamline == "ID01":
            # nu is clockwise
            kout = (
                2
                * np.pi
                / self.wavelength
                * np.array(
                    [
                        np.cos(np.pi * self.inplane_angle / 180)
                        * np.cos(np.pi * self.outofplane_angle / 180),  # z
                        np.sin(np.pi * self.outofplane_angle / 180),  # y
                        -np.sin(np.pi * self.inplane_angle / 180)
                        * np.cos(np.pi * self.outofplane_angle / 180),
                    ]
                )
            )  # x
        elif self.beamline == "34ID":
            # gamma is anti-clockwise
            kout = (
                2
                * np.pi
                / self.wavelength
                * np.array(
                    [
                        np.cos(np.pi * self.inplane_angle / 180)
                        * np.cos(np.pi * self.outofplane_angle / 180),  # z
                        np.sin(np.pi * self.outofplane_angle / 180),  # y
                        np.sin(np.pi * self.inplane_angle / 180)
                        * np.cos(np.pi * self.outofplane_angle / 180),
                    ]
                )
            )  # x
        elif self.beamline == "NANOMAX":
            # gamma is clockwise
            kout = (
                2
                * np.pi
                / self.wavelength
                * np.array(
                    [
                        np.cos(np.pi * self.inplane_angle / 180)
                        * np.cos(np.pi * self.outofplane_angle / 180),  # z
                        np.sin(np.pi * self.outofplane_angle / 180),  # y
                        -np.sin(np.pi * self.inplane_angle / 180)
                        * np.cos(np.pi * self.outofplane_angle / 180),
                    ]
                )
            )  # x
        elif self.beamline == "P10":
            # gamma is anti-clockwise
            kout = (
                2
                * np.pi
                / self.wavelength
                * np.array(
                    [
                        np.cos(np.pi * self.inplane_angle / 180)
                        * np.cos(np.pi * self.outofplane_angle / 180),  # z
                        np.sin(np.pi * self.outofplane_angle / 180),  # y
                        np.sin(np.pi * self.inplane_angle / 180)
                        * np.cos(np.pi * self.outofplane_angle / 180),
                    ]
                )
            )  # x
        elif self.beamline == "CRISTAL":
            # gamma is anti-clockwise
            kout = (
                2
                * np.pi
                / self.wavelength
                * np.array(
                    [
                        np.cos(np.pi * self.inplane_angle / 180)
                        * np.cos(np.pi * self.outofplane_angle / 180),  # z
                        np.sin(np.pi * self.outofplane_angle / 180),  # y
                        np.sin(np.pi * self.inplane_angle / 180)
                        * np.cos(np.pi * self.outofplane_angle / 180),
                    ]
                )
            )  # x

        return kout

    @property
    def filtered_data(self):
        """
        Boolean, True if the data and the mask to be loaded were already preprocessed.
        """
        return self._filtered_data

    @filtered_data.setter
    def filtered_data(self, value):
        if not isinstance(value, bool):
            raise TypeError("filtered_data should be a boolean")
        self._filtered_data = value

    @property
    def grazing_angle(self):
        """
        Motor positions for the goniometer circles below the rocking angle. It should be a list/tuple of lenght 1 for
         out-of-plane rocking curves (the motor value for mu if it exists) and length 2 for inplane rocking curves
         (mu and omega/om/eta motor values).
        """
        return self._grazing_angle

    @grazing_angle.setter
    def grazing_angle(self, value):
        if self.rocking_angle == "outofplane":
            # only the mu angle (rotation around the vertical axis, below the rocking angle omega/om/eta) is needed
            # mu is set to 0 if it does not exist
            valid.valid_container(
                value,
                container_types=(tuple, list),
                item_types=Real,
                allow_none=True,
                name="Setup.grazing_angle",
            )
            self._grazing_angle = value
        elif self.rocking_angle == "inplane":
            # one or more values needed, for example: mu angle, the omega/om/eta angle, the chi angle
            # (rotations respectively around the vertical axis, outboard and downstream, below the rocking angle phi)
            valid.valid_container(
                value,
                container_types=(tuple, list),
                item_types=Real,
                allow_none=True,
                name="Setup.grazing_angle",
            )
            self._grazing_angle = value
        else:  # self.rocking_angle == 'energy'
            # there is no sample rocking for energy scans, hence the grazing angle value do not matter
            self._grazing_angle = None

    @property
    def incident_wavevector(self):
        """
        Calculate the incident wavevector kin depending on the setup parameters, in the laboratory frame (z downstream,
         y vertical, x outboard). The unit is 1/m.

        :return: kin vector
        """
        return 2 * np.pi / self.wavelength * self.beam_direction

    @property
    def inplane_angle(self):
        """
        Horizontal detector angle, in degrees.
        """
        return self._inplane_angle

    @inplane_angle.setter
    def inplane_angle(self, value):
        if not isinstance(value, Real) and value is not None:
            raise TypeError("inplane_angle should be a number in degrees")
        self._inplane_angle = value

    @property
    def inplane_coeff(self):
        """
        Define a coefficient +/- 1 depending on the detector inplane rotation direction and the detector inplane
         orientation. The frame convention is the one of xrayutilities: x downstream, y outboard, z vertical up.
         See postprocessing/scripts/correct_angles_detector.py for an example.

        :return: +1 or -1
        """
        coeff_inplane = 0

        if self.detector_hor == "y+":
            hor_coeff = 1
        else:  # 'y-'
            hor_coeff = -1

        if self.beamline in {"SIXS_2018", "SIXS_2019"}:
            # gamma is anti-clockwise, we see the detector from downstream
            coeff_inplane = 1 * hor_coeff
        elif self.beamline == "ID01":
            # nu is clockwise, we see the detector from downstream
            coeff_inplane = -1 * hor_coeff
        elif self.beamline == "34ID":
            # delta is anti-clockwise, we see the detector from the front
            coeff_inplane = 1 * hor_coeff
        elif self.beamline == "P10":
            # gamma is anti-clockwise, we see the detector from the front
            coeff_inplane = 1 * hor_coeff
        elif self.beamline == "CRISTAL":
            # gamma is anti-clockwise, we see the detector from downstream
            coeff_inplane = 1 * hor_coeff
        elif self.beamline == "NANOMAX":
            # gamma is clockwise, we see the detector from downstream
            coeff_inplane = -1 * hor_coeff

        return coeff_inplane

    @property
    def outofplane_angle(self):
        """
        Vertical detector angle, in degrees.
        """
        return self._outofplane_angle

    @outofplane_angle.setter
    def outofplane_angle(self, value):
        if not isinstance(value, Real) and value is not None:
            raise TypeError("outofplane_angle should be a number in degrees")
        self._outofplane_angle = value

    @property
    def outofplane_coeff(self):
        """
        Define a coefficient +/- 1 depending on the detector out of plane rotation direction and the detector out of
         plane orientation. The frame convention is the one of xrayutilities: x downstream, y outboard, z vertical up.
         See postprocessing/scripts/correct_angles_detector.py for an example.

        :return: +1 or -1
        """
        if self.detector_ver == "z+":  # origin of pixels at the bottom
            ver_coeff = 1
        else:  # 'z-'  origin of pixels at the top
            ver_coeff = -1
        # the out of plane detector rotation is clockwise for all beamlines
        coeff_outofplane = -1 * ver_coeff
        return coeff_outofplane

    @property
    def params(self):
        """
        Return a dictionnary with all parameters
        """
        return {
            "Class": self.__class__.__name__,
            "beamline": self.beamline,
            "detector": self.detector.name,
            "pixel_size_m": self.detector.unbinned_pixel,
            "beam_direction": self.beam_direction,
            "energy_eV": self.energy,
            "distance_m": self.distance,
            "rocking_angle": self.rocking_angle,
            "outofplane_detector_angle_deg": self.outofplane_angle,
            "inplane_detector_angle_deg": self.inplane_angle,
            "tilt_angle_deg": self.tilt_angle,
            "grazing_angles_deg": self.grazing_angle,
            "sample_offsets_deg": self.diffractometer.sample_offsets,
            "direct_beam_pixel": self.direct_beam,
            "filtered_data": self.filtered_data,
            "custom_scan": self.custom_scan,
            "custom_images": self.custom_images,
            "actuators": self.actuators,
            "custom_monitor": self.custom_monitor,
            "custom_motors": self.custom_motors,
            "sample_inplane": self.sample_inplane,
            "sample_outofplane": self.sample_outofplane,
            "offset_inplane_deg": self.offset_inplane,
        }

    @property
    def q_laboratory(self):
        """
        Calculate the diffusion vector in the laboratory frame (z downstream, y vertical up, x outboard). The unit is
        1/A.

        :return: a tuple of three vectors components.
        """
        return (self.exit_wavevector - self.incident_wavevector) * 1e-10

    @property
    def rocking_angle(self):
        """
        Angle which is tilted during the rocking curve in {'outofplane', 'inplane'}
        """
        return self._rocking_angle

    @rocking_angle.setter
    def rocking_angle(self, value):
        if value is None:
            self._rocking_angle = value
        elif not isinstance(value, str):
            raise TypeError("rocking_angle should be a str")
        elif value not in {"outofplane", "inplane", "energy"}:
            raise ValueError(
                'rocking_angle can take only the value "outofplane", "inplane" or "energy"'
            )
        else:
            self._rocking_angle = value

    @property
    def tilt_angle(self):
        """
        Angular step of the rocking curve, in degrees.
        """
        return self._tilt_angle

    @tilt_angle.setter
    def tilt_angle(self, value):
        if not isinstance(value, Real) and value is not None:
            raise TypeError("tilt_angle should be a number in degrees")
        self._tilt_angle = value

    @property
    def wavelength(self):
        """
        Wavelength in meters.
        """
        if self.energy:
            return 12.398 * 1e-7 / self.energy  # in m

    def __repr__(self):
        """
        Representation string of the Setup instance.
        """
        return (
            f"{self.__class__.__name__}(beamline='{self.beamline}', detector='{self.detector.name}',"
            f" beam_direction={self.beam_direction}, "
            f"energy={self.energy}, distance={self.distance}, outofplane_angle={self.outofplane_angle},\n"
            f"inplane_angle={self.inplane_angle}, tilt_angle={self.tilt_angle}, "
            f"rocking_angle='{self.rocking_angle}', grazing_angle={self.grazing_angle},\n"
            f"pixel_size={self.detector.unbinned_pixel}, direct_beam={self.direct_beam}, "
            f"sample_offsets={self.diffractometer.sample_offsets}, "
            f"filtered_data={self.filtered_data}, custom_scan={self.custom_scan},\n"
            f"custom_images={self.custom_images},\ncustom_monitor={self.custom_monitor},\n"
            f"custom_motors={self.custom_motors},\n"
            f"sample_inplane={self.sample_inplane}, sample_outofplane={self.sample_outofplane}, "
            f"offset_inplane={self.offset_inplane})"
        )

    def create_logfile(self, scan_number, root_folder, filename):
        """
        Create the logfile used in gridmap().

        :param scan_number: the scan number to load
        :param root_folder: the root directory of the experiment, where is the specfile/.fio file
        :param filename: the file name to load, or the path of 'alias_dict.txt' for SIXS
        :return: logfile
        """
        logfile = None

        if self.beamline == "CRISTAL":  # no specfile, load directly the dataset
            ccdfiletmp = os.path.join(
                self.detector.datadir + self.detector.template_imagefile % scan_number
            )
            logfile = h5py.File(ccdfiletmp, "r")

        elif self.beamline == "P10":  # load .fio file
            logfile = root_folder + filename + "/" + filename + ".fio"

        elif self.beamline == "SIXS_2018":  # no specfile, load directly the dataset
            import bcdi.preprocessing.nxsReady as nxsReady

            logfile = nxsReady.DataSet(
                longname=self.detector.datadir
                + self.detector.template_imagefile % scan_number,
                shortname=self.detector.template_imagefile % scan_number,
                alias_dict=filename,
                scan="SBS",
            )
        elif self.beamline == "SIXS_2019":  # no specfile, load directly the dataset
            import bcdi.preprocessing.ReadNxs3 as ReadNxs3

            logfile = ReadNxs3.DataSet(
                directory=self.detector.datadir,
                filename=self.detector.template_imagefile % scan_number,
                alias_dict=filename,
            )

        elif self.beamline == "ID01":  # load spec file
            from silx.io.specfile import SpecFile

            logfile = SpecFile(root_folder + filename + ".spec")

        elif self.beamline == "NANOMAX":
            ccdfiletmp = os.path.join(
                self.detector.datadir + self.detector.template_imagefile % scan_number
            )
            logfile = h5py.File(ccdfiletmp, "r")

        return logfile

    def create_diffractometer(self, sample_offsets):
        """
        Create a Diffractometer instance depending on the beamline.
        """
        if self.beamline == "ID01":
            return DiffractometerID01(sample_offsets)
        if self.beamline in {"SIXS_2018", "SIXS_2019"}:
            return DiffractometerSIXS(sample_offsets)
        if self.beamline == "34ID":
            return Diffractometer34ID(sample_offsets)
        if self.beamline == "P10":
            return DiffractometerP10(sample_offsets)
        if self.beamline == "CRISTAL":
            return DiffractometerCRISTAL(sample_offsets)
        if self.beamline == "NANOMAX":
            return DiffractometerNANOMAX(sample_offsets)

    def detector_frame(
        self,
        obj,
        voxel_size,
        width_z=None,
        width_y=None,
        width_x=None,
        debugging=False,
        **kwargs,
    ):
        """
        Interpolate the orthogonal object back into the non-orthogonal detector frame

        :param obj: real space object, in the orthogonal laboratory frame
        :param voxel_size: voxel size of the original object, number of list/tuple of three numbers
        :param width_z: size of the area to plot in z (axis 0), centered on the middle of the initial array
        :param width_y: size of the area to plot in y (axis 1), centered on the middle of the initial array
        :param width_x: size of the area to plot in x (axis 2), centered on the middle of the initial array
        :param debugging: True to show plots before and after interpolation
        :param kwargs:
         - 'title': title for the debugging plots
        :return: object interpolated on an orthogonal grid
        """
        valid.valid_kwargs(
            kwargs=kwargs, allowed_kwargs={"title"}, name="Setup.detector_frame"
        )
        title = kwargs.get("title", "Object")

        if isinstance(voxel_size, Real):
            voxel_size = (voxel_size, voxel_size, voxel_size)
        valid.valid_container(
            obj=voxel_size,
            container_types=(tuple, list),
            length=3,
            item_types=Real,
            min_excluded=0,
            name="Setup.detector_frame",
        )

        nbz, nby, nbx = obj.shape

        if debugging:
            gu.multislices_plot(
                abs(obj),
                sum_frames=True,
                width_z=width_z,
                width_y=width_y,
                width_x=width_x,
                title=title + " before interpolation\n",
            )

        ortho_matrix = self.transformation_matrix(
            array_shape=(nbz, nby, nbx),
            tilt_angle=self.tilt_angle,
            pixel_x=self.detector.unbinned_pixel[1],
            pixel_y=self.detector.unbinned_pixel[0],
        )

        ################################################
        # interpolate the data into the detector frame #
        ################################################
        myz, myy, myx = np.meshgrid(
            np.arange(-nbz // 2, nbz // 2, 1),
            np.arange(-nby // 2, nby // 2, 1),
            np.arange(-nbx // 2, nbx // 2, 1),
            indexing="ij",
        )

        new_x = (
            ortho_matrix[0, 0] * myx
            + ortho_matrix[0, 1] * myy
            + ortho_matrix[0, 2] * myz
        )
        new_y = (
            ortho_matrix[1, 0] * myx
            + ortho_matrix[1, 1] * myy
            + ortho_matrix[1, 2] * myz
        )
        new_z = (
            ortho_matrix[2, 0] * myx
            + ortho_matrix[2, 1] * myy
            + ortho_matrix[2, 2] * myz
        )
        del myx, myy, myz
        # la partie rgi est sure: c'est la taille de l'objet orthogonal de depart
        rgi = RegularGridInterpolator(
            (
                np.arange(-nbz // 2, nbz // 2) * voxel_size[0],
                np.arange(-nby // 2, nby // 2) * voxel_size[1],
                np.arange(-nbx // 2, nbx // 2) * voxel_size[2],
            ),
            obj,
            method="linear",
            bounds_error=False,
            fill_value=0,
        )
        detector_obj = rgi(
            np.concatenate(
                (
                    new_z.reshape((1, new_z.size)),
                    new_y.reshape((1, new_z.size)),
                    new_x.reshape((1, new_z.size)),
                )
            ).transpose()
        )
        detector_obj = detector_obj.reshape((nbz, nby, nbx)).astype(obj.dtype)

        if debugging:
            gu.multislices_plot(
                abs(detector_obj),
                sum_frames=True,
                width_z=width_z,
                width_y=width_y,
                width_x=width_x,
                title=title + " interpolated in detector frame\n",
            )

        return detector_obj

    def init_paths(
        self,
        sample_name,
        scan_number,
        root_folder,
        save_dir,
        specfile_name,
        template_imagefile,
        data_dirname=None,
        save_dirname="result",
        create_savedir=False,
        verbose=False,
    ):
        """
        Update the detector instance with initialized paths and template for filenames depending on the beamline

        :param sample_name: string in front of the scan number in the data folder name.
        :param scan_number: the scan number
        :param root_folder: folder of the experiment, where all scans are stored
        :param save_dir: path of the directory where to save the analysis results, can be None
        :param specfile_name: beamline-dependent string
         - ID01: name of the spec file without '.spec'
         - SIXS_2018 and SIXS_2019: None or full path of the alias dictionnary (e.g. root_folder+'alias_dict_2019.txt')
         - empty string for all other beamlines
        :param template_imagefile: beamline-dependent template for the data files
         - ID01: 'data_mpx4_%05d.edf.gz' or 'align_eiger2M_%05d.edf.gz'
         - SIXS_2018: 'align.spec_ascan_mu_%05d.nxs'
         - SIXS_2019: 'spare_ascan_mu_%05d.nxs'
         - Cristal: 'S%d.nxs'
         - P10: '_master.h5'
         - NANOMAX: '%06d.h5'
         - 34ID: 'Sample%dC_ES_data_51_256_256.npz'
        :param data_dirname: name of the data folder, if None it will use the beamline default, if it is an empty
         string, it will look for the data directly into the scan folder (no subfolder)
        :param save_dirname: name of the saving folder, by default 'save_dir/result/' will be created
        :param create_savedir: boolean, True to create the saving folder if it does not exist
        :param verbose: True to print the paths
        """
        if not isinstance(scan_number, int):
            raise TypeError("scan_number should be an integer")

        if not isinstance(sample_name, str):
            raise TypeError("sample_name should be a string")

        # check that the provided folder names are not an empty string
        valid.valid_container(
            save_dirname, container_types=str, min_length=1, name="Setup.init_paths"
        )
        valid.valid_container(
            data_dirname,
            container_types=str,
            min_length=0,
            allow_none=True,
            name="Setup.init_paths",
        )
        (
            self.detector.rootdir,
            self.detector.sample_name,
            self.detector.template_file,
        ) = (root_folder, sample_name, template_imagefile)

        if self.beamline == "P10":
            specfile = sample_name + "_{:05d}".format(scan_number)
            homedir = root_folder + specfile + "/"
            default_dirname = "e4m/"
            template_imagefile = specfile + template_imagefile
        elif self.beamline == "NANOMAX":
            homedir = root_folder + sample_name + "{:06d}".format(scan_number) + "/"
            default_dirname = "data/"
            specfile = specfile_name
        elif self.beamline in {"SIXS_2018", "SIXS_2019"}:
            homedir = root_folder + sample_name + str(scan_number) + "/"
            default_dirname = "data/"
            if (
                specfile_name is None
            ):  # default to the alias dictionnary located within the package
                specfile_name = os.path.abspath(
                    os.path.join(
                        os.path.dirname(__file__),
                        os.pardir,
                        "preprocessing/alias_dict_2021.txt",
                    )
                )
            specfile = specfile_name
        else:
            homedir = root_folder + sample_name + str(scan_number) + "/"
            default_dirname = "data/"
            specfile = specfile_name

        if data_dirname is not None:
            if len(data_dirname) == 0:  # no subfolder
                datadir = homedir
            else:
                datadir = homedir + data_dirname
        else:
            datadir = homedir + default_dirname

        if save_dir:
            savedir = save_dir
        else:
            savedir = homedir + save_dirname + "/"

        if not savedir.endswith("/"):
            savedir += "/"
        if not datadir.endswith("/"):
            datadir += "/"

        (
            self.detector.savedir,
            self.detector.datadir,
            self.detector.specfile,
            self.detector.template_imagefile,
        ) = (savedir, datadir, specfile, template_imagefile)

        if create_savedir:
            pathlib.Path(self.detector.savedir).mkdir(parents=True, exist_ok=True)

        if verbose:
            if not self.custom_scan:
                print(
                    f"datadir = '{datadir}'\nsavedir = '{savedir}'\ntemplate_imagefile = '{template_imagefile}'\n"
                )
            else:
                print(
                    f"rootdir = '{root_folder}'\nsavedir = '{savedir}'\nsample_name = '{self.detector.sample_name}'\n"
                    f"template_imagefile = '{self.detector.template_file}'\n"
                )

    def ortho_directspace(
        self,
        arrays,
        q_com,
        initial_shape=None,
        voxel_size=None,
        fill_value=0,
        reference_axis=(0, 1, 0),
        verbose=True,
        debugging=False,
        **kwargs,
    ):
        """
        Interpolate arrays (direct space output of the phase retrieval) in the orthogonal reference frame where q_com
        is aligned onto the array axis reference_axis.

        :param arrays: tuple of 3D arrays of the same shape (output of the phase retrieval), in the detector frame
        :param q_com: tuple of 3 vector components for the q values of the center of mass of the Bragg peak,
         expressed in an orthonormal frame x y z
        :param initial_shape: shape of the FFT used for phasing
        :param voxel_size: number or list of three user-defined voxel sizes for the interpolation, in nm.
         If a single number is provided, the voxel size will be identical in all directions.
        :param fill_value: tuple of real numbers, fill_value parameter for the RegularGridInterpolator, same length as
         the number of arrays
        :param reference_axis: 3D vector along which q will be aligned, expressed in an orthonormal frame x y z
        :param verbose: True to have printed comments
        :param debugging: tuple of booleans of the same length as the number of input arrays, True to show plots before
         and after interpolation
        :param kwargs:
         - 'title': tuple of strings, titles for the debugging plots, same length as the number of arrays
         - width_z: size of the area to plot in z (axis 0), centered on the middle of the initial array
         - width_y: size of the area to plot in y (axis 1), centered on the middle of the initial array
         - width_x: size of the area to plot in x (axis 2), centered on the middle of the initial array
        :return:
         - an array (if a single array was provided) or a tuple of arrays interpolated on an orthogonal grid
           (same length as the number of input arrays)
         - a tuple of 3 voxels size for the interpolated arrays
        """
        #############################################
        # check that arrays is a tuple of 3D arrays #
        #############################################
        if isinstance(arrays, np.ndarray):
            arrays = (arrays,)
        valid.valid_container(
            arrays,
            container_types=(tuple, list),
            item_types=np.ndarray,
            min_length=1,
            name="arrays",
        )
        if any(array.ndim != 3 for array in arrays):
            raise ValueError("all arrays should be 3D ndarrays of the same shape")
        ref_shape = arrays[0].shape
        if any(array.shape != ref_shape for array in arrays):
            raise ValueError("all arrays should be 3D ndarrays of the same shape")
        nb_arrays = len(arrays)
        input_shape = arrays[
            0
        ].shape  # could be smaller than the shape used in phase retrieval,
        # if the object was cropped around the support

        #########################
        # check and load kwargs #
        #########################
        valid.valid_kwargs(
            kwargs=kwargs,
            allowed_kwargs={"title", "width_z", "width_y", "width_x"},
            name="kwargs",
        )
        title = kwargs.get("title", ("Object",) * nb_arrays)
        if isinstance(title, str):
            title = (title,) * nb_arrays
        valid.valid_container(
            title,
            container_types=(tuple, list),
            length=nb_arrays,
            item_types=str,
            name="title",
        )
        width_z = kwargs.get("width_z")
        valid.valid_item(
            value=width_z,
            allowed_types=int,
            min_excluded=0,
            allow_none=True,
            name="width_z",
        )
        width_y = kwargs.get("width_y")
        valid.valid_item(
            value=width_y,
            allowed_types=int,
            min_excluded=0,
            allow_none=True,
            name="width_y",
        )
        width_x = kwargs.get("width_x")
        valid.valid_item(
            value=width_x,
            allowed_types=int,
            min_excluded=0,
            allow_none=True,
            name="width_x",
        )

        #########################
        # check some parameters #
        #########################
        valid.valid_container(
            q_com,
            container_types=(tuple, list, np.ndarray),
            length=3,
            item_types=Real,
            name="q_com",
        )
        if np.linalg.norm(q_com) == 0:
            raise ValueError("q_com should be a non zero vector")

        if isinstance(fill_value, Real):
            fill_value = (fill_value,) * nb_arrays
        valid.valid_container(
            fill_value,
            container_types=(tuple, list, np.ndarray),
            length=nb_arrays,
            item_types=Real,
            name="fill_value",
        )
        if isinstance(debugging, bool):
            debugging = (debugging,) * nb_arrays
        valid.valid_container(
            debugging,
            container_types=(tuple, list),
            length=nb_arrays,
            item_types=bool,
            name="debugging",
        )
        valid.valid_container(
            q_com,
            container_types=(tuple, list, np.ndarray),
            length=3,
            item_types=Real,
            name="q_com",
        )
        q_com = np.array(q_com)
        valid.valid_container(
            reference_axis,
            container_types=(tuple, list, np.ndarray),
            length=3,
            item_types=Real,
            name="reference_axis",
        )
        reference_axis = np.array(reference_axis)
        if not any(
            (reference_axis == val).all()
            for val in (np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]))
        ):
            raise NotImplementedError(
                "strain calculation along directions other than array axes is not implemented"
            )

        if not initial_shape:
            initial_shape = input_shape
        else:
            valid.valid_container(
                initial_shape,
                container_types=(tuple, list),
                length=3,
                item_types=int,
                min_excluded=0,
                name="Setup.orthogonalize",
            )

        ######################################################################################################
        # calculate the direct space voxel sizes in nm based on the FFT window shape used in phase retrieval #
        ######################################################################################################
        dz_realspace, dy_realspace, dx_realspace = self.voxel_sizes(
            initial_shape,
            tilt_angle=abs(self.tilt_angle),
            pixel_x=self.detector.unbinned_pixel[1],
            pixel_y=self.detector.unbinned_pixel[0],
        )
        if verbose:
            print(
                "Sampling in the laboratory frame (z, y, x): ",
                f"({dz_realspace:.2f} nm, {dy_realspace:.2f} nm, {dx_realspace:.2f} nm)",
            )

        if input_shape != initial_shape:
            # recalculate the tilt and pixel sizes to accomodate a shape change
            tilt = self.tilt_angle * initial_shape[0] / input_shape[0]
            pixel_y = (
                self.detector.unbinned_pixel[0] * initial_shape[1] / input_shape[1]
            )
            pixel_x = (
                self.detector.unbinned_pixel[1] * initial_shape[2] / input_shape[2]
            )
            if verbose:
                print(
                    "Tilt, pixel_y, pixel_x based on the shape of the cropped array:",
                    f"({tilt:.4f} deg, {pixel_y * 1e6:.2f} um, {pixel_x * 1e6:.2f} um)",
                )

            # sanity check, the direct space voxel sizes calculated below should be equal to the original ones
            dz_realspace, dy_realspace, dx_realspace = self.voxel_sizes(
                input_shape, tilt_angle=abs(tilt), pixel_x=pixel_x, pixel_y=pixel_y
            )
            if verbose:
                print(
                    "Sanity check, recalculated direct space voxel sizes (z, y, x): ",
                    f"({dz_realspace:.2f} nm, {dy_realspace:.2f} nm, {dx_realspace:.2f} nm)",
                )
        else:
            tilt = self.tilt_angle
            pixel_y = self.detector.unbinned_pixel[0]
            pixel_x = self.detector.unbinned_pixel[1]

        if not voxel_size:
            voxel_size = dz_realspace, dy_realspace, dx_realspace  # in nm
        else:
            if isinstance(voxel_size, Real):
                voxel_size = (voxel_size, voxel_size, voxel_size)
            if not isinstance(voxel_size, Sequence):
                raise TypeError(
                    "voxel size should be a sequence of three positive numbers in nm"
                )
            if len(voxel_size) != 3 or any(val <= 0 for val in voxel_size):
                raise ValueError(
                    "voxel_size should be a sequence of three positive numbers in nm"
                )

        ######################################################################
        # calculate the transformation matrix based on the beamline geometry #
        ######################################################################
        transfer_matrix = self.transformation_matrix(
            array_shape=input_shape,
            tilt_angle=tilt,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            verbose=verbose,
        )

        ################################################################################
        # calculate the rotation matrix from the crystal frame to the laboratory frame #
        ################################################################################
        # (inverse rotation to have reference_axis along q)
        rotation_matrix = util.rotation_matrix_3d(
            axis_to_align=reference_axis, reference_axis=q_com / np.linalg.norm(q_com)
        )
        # rotation_matrix = np.identity(3)
        ####################################################################################
        # calculate the full transfer matrix including the rotation into the crystal frame #
        ####################################################################################
        transfer_matrix = np.matmul(rotation_matrix, transfer_matrix)
        # transfer_matrix is the transformation matrix of the direct space coordinates
        # the spacing in the crystal frame is therefore given by the rows of the matrix
        d_along_x = np.linalg.norm(transfer_matrix[0, :])  # along x outboard
        d_along_y = np.linalg.norm(transfer_matrix[1, :])  # along y vertical up
        d_along_z = np.linalg.norm(transfer_matrix[2, :])  # along z downstream

        ############################################################################################
        # find the shape of the output array that fits the extent of the data after transformation #
        ############################################################################################

        # calculate the voxel coordinates of the data points in the laboratory frame
        myz, myy, myx = np.meshgrid(
            np.arange(-input_shape[0] // 2, input_shape[0] // 2, 1),
            np.arange(-input_shape[1] // 2, input_shape[1] // 2, 1),
            np.arange(-input_shape[2] // 2, input_shape[2] // 2, 1),
            indexing="ij",
        )

        pos_along_x = (
            transfer_matrix[0, 0] * myx
            + transfer_matrix[0, 1] * myy
            + transfer_matrix[0, 2] * myz
        )
        pos_along_y = (
            transfer_matrix[1, 0] * myx
            + transfer_matrix[1, 1] * myy
            + transfer_matrix[1, 2] * myz
        )
        pos_along_z = (
            transfer_matrix[2, 0] * myx
            + transfer_matrix[2, 1] * myy
            + transfer_matrix[2, 2] * myz
        )

        if verbose:
            print(
                "\nCalculating the shape of the output array fitting the data extent after transformation:"
                f"\nSampling in the crystal frame (axis 0, axis 1, axis 2):    "
                f"({d_along_z:.2f} nm, {d_along_y:.2f} nm, {d_along_x:.2f} nm)"
            )
        # these positions are not equally spaced, we just extract the data extent from them
        nx_output = int(np.rint((pos_along_x.max() - pos_along_x.min()) / d_along_x))
        ny_output = int(np.rint((pos_along_y.max() - pos_along_y.min()) / d_along_y))
        nz_output = int(np.rint((pos_along_z.max() - pos_along_z.min()) / d_along_z))

        # add some margin to the output shape for easier visualization
        nx_output += 10
        ny_output += 10
        nz_output += 10
        del pos_along_x, pos_along_y, pos_along_z
        gc.collect()

        #########################################
        # calculate the interpolation positions #
        #########################################
        myz, myy, myx = np.meshgrid(
            np.arange(-nz_output // 2, nz_output // 2, 1) * voxel_size[0],
            np.arange(-ny_output // 2, ny_output // 2, 1) * voxel_size[1],
            np.arange(-nx_output // 2, nx_output // 2, 1) * voxel_size[2],
            indexing="ij",
        )

        # ortho_matrix is the transformation matrix from the detector coordinates to the laboratory frame
        # in RGI, we want to calculate the coordinates that would have a grid of the laboratory frame expressed in the
        # detector frame, i.e. one has to inverse the transformation matrix.
        transfer_imatrix = np.linalg.inv(transfer_matrix)
        new_x = (
            transfer_imatrix[0, 0] * myx
            + transfer_imatrix[0, 1] * myy
            + transfer_imatrix[0, 2] * myz
        )
        new_y = (
            transfer_imatrix[1, 0] * myx
            + transfer_imatrix[1, 1] * myy
            + transfer_imatrix[1, 2] * myz
        )
        new_z = (
            transfer_imatrix[2, 0] * myx
            + transfer_imatrix[2, 1] * myy
            + transfer_imatrix[2, 2] * myz
        )
        del myx, myy, myz
        gc.collect()

        ######################
        # interpolate arrays #
        ######################
        output_arrays = []
        for idx, array in enumerate(arrays):
            rgi = RegularGridInterpolator(
                (
                    np.arange(-input_shape[0] // 2, input_shape[0] // 2, 1),
                    np.arange(-input_shape[1] // 2, input_shape[1] // 2, 1),
                    np.arange(-input_shape[2] // 2, input_shape[2] // 2, 1),
                ),
                array,
                method="linear",
                bounds_error=False,
                fill_value=fill_value[idx],
            )
            ortho_array = rgi(
                np.concatenate(
                    (
                        new_z.reshape((1, new_z.size)),
                        new_y.reshape((1, new_z.size)),
                        new_x.reshape((1, new_z.size)),
                    )
                ).transpose()
            )
            ortho_array = ortho_array.reshape((nz_output, ny_output, nx_output)).astype(
                array.dtype
            )
            output_arrays.append(ortho_array)

            if debugging[idx]:
                gu.multislices_plot(
                    abs(array),
                    sum_frames=False,
                    width_z=width_z,
                    width_y=width_y,
                    width_x=width_x,
                    reciprocal_space=False,
                    is_orthogonal=False,
                    scale="linear",
                    title=title[idx] + " in detector frame",
                )

                gu.multislices_plot(
                    abs(ortho_array),
                    sum_frames=False,
                    width_z=width_z,
                    width_y=width_y,
                    width_x=width_x,
                    reciprocal_space=False,
                    is_orthogonal=True,
                    scale="linear",
                    title=title[idx] + " in crystal frame",
                )

        if nb_arrays == 1:
            output_arrays = output_arrays[0]  # return the array instead of the tuple
        return output_arrays, voxel_size

    def ortho_reciprocal(
        self,
        arrays,
        fill_value=0,
        align_q=False,
        reference_axis=(0, 1, 0),
        verbose=True,
        debugging=False,
        **kwargs,
    ):
        """
        Interpolate arrays in the orthogonal laboratory frame (z/qx downstream, y/qz vertical up, x/qy outboard)
        or crystal frame (q aligned along one array axis). The ouput shape will be increased in order to keep the same
        range in q in each direction. The sampling in q is defined as the norm of the rows of the transformation matrix.

        :param arrays: tuple of 3D arrays of the same shape (e.g.: reciprocal space diffraction pattern and mask),
         in the detector frame
        :param fill_value: tuple of real numbers, fill_value parameter for the RegularGridInterpolator, same length as
         the number of arrays
        :param align_q: boolean, if True the data will be rotated such that q is along reference_axis, and q values
         will be calculated in the pseudo crystal frame.
        :param reference_axis: 3D vector along which q will be aligned, expressed in an orthonormal frame x y z
        :param verbose: True to have printed comments
        :param debugging: tuple of booleans of the same length as the number of input arrays, True to show plots before
         and after interpolation
        :param kwargs:
         - 'title': tuple of strings, titles for the debugging plots, same length as the number of arrays
         - 'scale': tuple of strings (either 'linear' or 'log'), scale for the debugging plots, same length as the
           number of arrays
         - width_z: size of the area to plot in z (axis 0), centered on the middle of the initial array
         - width_y: size of the area to plot in y (axis 1), centered on the middle of the initial array
         - width_x: size of the area to plot in x (axis 2), centered on the middle of the initial array
        :return:
         - an array (if a single array was provided) or a tuple of arrays interpolated on an orthogonal grid
           (same length as the number of input arrays)
         - a tuple of three 1D vectors of q values (qx, qz, qy)
        """
        #############################################
        # check that arrays is a tuple of 3D arrays #
        #############################################
        if isinstance(arrays, np.ndarray):
            arrays = (arrays,)
        valid.valid_container(
            arrays,
            container_types=(tuple, list),
            item_types=np.ndarray,
            min_length=1,
            name="arrays",
        )
        if any(array.ndim != 3 for array in arrays):
            raise ValueError("all arrays should be 3D ndarrays of the same shape")
        ref_shape = arrays[0].shape
        if any(array.shape != ref_shape for array in arrays):
            raise ValueError("all arrays should be 3D ndarrays of the same shape")
        nb_arrays = len(arrays)
        nbz, nby, nbx = ref_shape

        #########################
        # check and load kwargs #
        #########################
        valid.valid_kwargs(
            kwargs=kwargs,
            allowed_kwargs={"title", "scale", "width_z", "width_y", "width_x"},
            name="kwargs",
        )
        title = kwargs.get("title", ("Object",) * nb_arrays)
        if isinstance(title, str):
            title = (title,) * nb_arrays
        valid.valid_container(
            title,
            container_types=(tuple, list),
            length=nb_arrays,
            item_types=str,
            name="title",
        )
        scale = kwargs.get("scale", ("log",) * nb_arrays)
        if isinstance(scale, str):
            scale = (scale,) * nb_arrays
        valid.valid_container(
            scale, container_types=(tuple, list), length=nb_arrays, name="scale"
        )
        if any(val not in {"log", "linear"} for val in scale):
            raise ValueError("scale should be either 'log' or 'linear'")

        width_z = kwargs.get("width_z")
        valid.valid_item(
            value=width_z,
            allowed_types=int,
            min_excluded=0,
            allow_none=True,
            name="width_z",
        )
        width_y = kwargs.get("width_y")
        valid.valid_item(
            value=width_y,
            allowed_types=int,
            min_excluded=0,
            allow_none=True,
            name="width_y",
        )
        width_x = kwargs.get("width_x")
        valid.valid_item(
            value=width_x,
            allowed_types=int,
            min_excluded=0,
            allow_none=True,
            name="width_x",
        )

        #########################
        # check some parameters #
        #########################
        if isinstance(fill_value, Real):
            fill_value = (fill_value,) * nb_arrays
        valid.valid_container(
            fill_value,
            container_types=(tuple, list, np.ndarray),
            length=nb_arrays,
            item_types=Real,
            name="fill_value",
        )
        if isinstance(debugging, bool):
            debugging = (debugging,) * nb_arrays
        valid.valid_container(
            debugging,
            container_types=(tuple, list),
            length=nb_arrays,
            item_types=bool,
            name="debugging",
        )
        valid.valid_item(align_q, allowed_types=bool, name="align_q")
        valid.valid_container(
            reference_axis,
            container_types=(tuple, list, np.ndarray),
            length=3,
            item_types=Real,
            name="reference_axis",
        )
        reference_axis = np.array(reference_axis)

        ##########################################################
        # calculate the transformation matrix (the unit is 1/nm) #
        ##########################################################
        transfer_matrix, q_offset = self.transformation_matrix(
            array_shape=ref_shape,
            tilt_angle=self.tilt_angle,
            direct_space=False,
            verbose=verbose,
            pixel_x=self.detector.unbinned_pixel[1],
            pixel_y=self.detector.unbinned_pixel[0],
        )

        # the voxel size in q in the laboratory frame is given by the rows of the transformation matrix
        # (the unit is 1/nm)
        dq_along_x = np.linalg.norm(transfer_matrix[0, :])  # along x outboard
        dq_along_y = np.linalg.norm(transfer_matrix[1, :])  # along y vertical up
        dq_along_z = np.linalg.norm(transfer_matrix[2, :])  # along z downstream

        ############################################################################################
        # find the shape of the output array that fits the extent of the data after transformation #
        ############################################################################################

        # calculate the q coordinates of the data points in the laboratory frame
        myz, myy, myx = np.meshgrid(
            np.arange(-nbz // 2, nbz // 2, 1),
            np.arange(-nby // 2, nby // 2, 1),
            np.arange(-nbx // 2, nbx // 2, 1),
            indexing="ij",
        )

        q_along_x = (
            transfer_matrix[0, 0] * myx
            + transfer_matrix[0, 1] * myy
            + transfer_matrix[0, 2] * myz
        )
        q_along_y = (
            transfer_matrix[1, 0] * myx
            + transfer_matrix[1, 1] * myy
            + transfer_matrix[1, 2] * myz
        )
        q_along_z = (
            transfer_matrix[2, 0] * myx
            + transfer_matrix[2, 1] * myy
            + transfer_matrix[2, 2] * myz
        )
        if verbose:
            print(
                "\nInterpolating:"
                f"\nSampling in q in the laboratory frame (z*, y*, x*):    "
                f"({dq_along_z:.5f} 1/nm, {dq_along_y:.5f} 1/nm, {dq_along_x:.5f} 1/nm)"
            )
        # these q values are not equally spaced, we just extract the q extent from them
        nx_output = int(np.rint((q_along_x.max() - q_along_x.min()) / dq_along_x))
        ny_output = int(np.rint((q_along_y.max() - q_along_y.min()) / dq_along_y))
        nz_output = int(np.rint((q_along_z.max() - q_along_z.min()) / dq_along_z))

        if align_q:
            #######################################################################
            # find the shape of the output array that fits the extent of the data #
            # after rotating further these q values in a pseudo crystal frame     #
            #######################################################################
            # the center of mass of the diffraction should be in the center of the array!
            # TODO: implement any offset of the center of mass
            q_along_z_com = (
                q_along_z[nbz // 2, nby // 2, nbx // 2] + q_offset[2]
            )  # q_offset in the order xyz
            q_along_y_com = q_along_y[nbz // 2, nby // 2, nbx // 2] + q_offset[1]
            q_along_x_com = q_along_x[nbz // 2, nby // 2, nbx // 2] + q_offset[0]
            qnorm = np.linalg.norm(
                np.array([q_along_x_com, q_along_y_com, q_along_z_com])
            )  # in 1/A
            if verbose:
                print(f"\nAligning Q along {reference_axis} (x,y,z)")

            # calculate the rotation matrix from the crystal frame to the laboratory frame
            # (inverse rotation to have reference_axis along q)
            rotation_matrix = util.rotation_matrix_3d(
                axis_to_align=reference_axis,
                reference_axis=np.array([q_along_x_com, q_along_y_com, q_along_z_com])
                / qnorm,
            )

            # calculate the full transfer matrix including the rotation into the crystal frame
            transfer_matrix = np.matmul(rotation_matrix, transfer_matrix)

            # the voxel size in q in the laboratory frame is given by the rows of the transformation matrix
            # (the unit is 1/nm)
            dq_along_x = np.linalg.norm(transfer_matrix[0, :])  # along x outboard
            dq_along_y = np.linalg.norm(transfer_matrix[1, :])  # along y vertical up
            dq_along_z = np.linalg.norm(transfer_matrix[2, :])  # along z downstream

            # calculate the new offset in the crystal frame (inverse rotation to have qz along q)
            offset_crystal = util.rotate_vector(
                vectors=q_offset,
                axis_to_align=reference_axis,
                reference_axis=np.array([q_along_x_com, q_along_y_com, q_along_z_com])
                / qnorm,
            )
            q_offset = offset_crystal[::-1]  # offset_crystal is in the order z, y, x

            # calculate the q coordinates of the data points in the crystal frame
            q_along_x = (
                transfer_matrix[0, 0] * myx
                + transfer_matrix[0, 1] * myy
                + transfer_matrix[0, 2] * myz
            )
            q_along_y = (
                transfer_matrix[1, 0] * myx
                + transfer_matrix[1, 1] * myy
                + transfer_matrix[1, 2] * myz
            )
            q_along_z = (
                transfer_matrix[2, 0] * myx
                + transfer_matrix[2, 1] * myy
                + transfer_matrix[2, 2] * myz
            )

            # these q values are not equally spaced, we just extract the q extent from them
            nx_output = int(np.rint((q_along_x.max() - q_along_x.min()) / dq_along_x))
            ny_output = int(np.rint((q_along_y.max() - q_along_y.min()) / dq_along_y))
            nz_output = int(np.rint((q_along_z.max() - q_along_z.min()) / dq_along_z))

            if verbose:
                print(
                    f"\nSampling in q in the crystal frame (axis 0, axis 1, axis 2):    "
                    f"({dq_along_z:.5f} 1/nm, {dq_along_y:.5f} 1/nm, {dq_along_x:.5f} 1/nm)"
                )

        del q_along_x, q_along_y, q_along_z, myx, myy, myz
        gc.collect()

        ##########################################################
        # crop the output shape in order to fit FFT requirements #
        ##########################################################
        nz_output, ny_output, nx_output = smaller_primes(
            (nz_output, ny_output, nx_output), maxprime=7, required_dividers=(2,)
        )
        if verbose:
            print(
                f"\nInitial shape = ({nbz},{nby},{nbx})\nOutput shape  = ({nz_output},{ny_output},{nx_output})"
                f" (satisfying FFT shape requirements)"
            )

        #####################################################################################################
        # define the interpolation qx qz qy 1D vectors in 1/nm, the reference being the center of the array #
        #####################################################################################################
        # the usual frame is used for q values: qx downstream, qz vertical up, qy outboard
        # this assumes that the center of mass of the diffraction pattern was at the center of the array
        # TODO : correct this if the diffraction pattern is not centered
        qx = (
            np.arange(-nz_output // 2, nz_output // 2, 1) * dq_along_z
        )  # along z downstream
        qz = (
            np.arange(-ny_output // 2, ny_output // 2, 1) * dq_along_y
        )  # along y vertical up
        qy = (
            np.arange(-nx_output // 2, nx_output // 2, 1) * dq_along_x
        )  # along x outboard

        myz, myy, myx = np.meshgrid(qx, qz, qy, indexing="ij")

        # transfer_matrix is the transformation matrix from the detector coordinates to the laboratory/crystal frame
        # in RGI, we want to calculate the coordinates that would have a grid of the laboratory/crystal frame expressed
        # in the detector frame, i.e. one has to inverse the transformation matrix.
        transfer_imatrix = np.linalg.inv(transfer_matrix)
        new_x = (
            transfer_imatrix[0, 0] * myx
            + transfer_imatrix[0, 1] * myy
            + transfer_imatrix[0, 2] * myz
        )
        new_y = (
            transfer_imatrix[1, 0] * myx
            + transfer_imatrix[1, 1] * myy
            + transfer_imatrix[1, 2] * myz
        )
        new_z = (
            transfer_imatrix[2, 0] * myx
            + transfer_imatrix[2, 1] * myy
            + transfer_imatrix[2, 2] * myz
        )
        del myx, myy, myz
        gc.collect()

        ######################
        # interpolate arrays #
        ######################
        output_arrays = []
        for idx, array in enumerate(arrays):
            # convert array type to float, for integers the interpolation can lead to artefacts
            array = array.astype(float)
            rgi = RegularGridInterpolator(
                (
                    np.arange(-nbz // 2, nbz // 2),
                    np.arange(-nby // 2, nby // 2),
                    np.arange(-nbx // 2, nbx // 2),
                ),
                array,
                method="linear",
                bounds_error=False,
                fill_value=fill_value[idx],
            )
            ortho_array = rgi(
                np.concatenate(
                    (
                        new_z.reshape((1, new_z.size)),
                        new_y.reshape((1, new_z.size)),
                        new_x.reshape((1, new_z.size)),
                    )
                ).transpose()
            )
            ortho_array = ortho_array.reshape((nz_output, ny_output, nx_output)).astype(
                array.dtype
            )
            output_arrays.append(ortho_array)

            if debugging[idx]:
                gu.multislices_plot(
                    abs(array),
                    sum_frames=True,
                    scale=scale,
                    plot_colorbar=True,
                    width_z=width_z,
                    width_y=width_y,
                    width_x=width_x,
                    is_orthogonal=False,
                    reciprocal_space=True,
                    vmin=0,
                    title=title[idx] + " in detector frame",
                )
                gu.multislices_plot(
                    abs(ortho_array),
                    sum_frames=True,
                    scale=scale[idx],
                    plot_colorbar=True,
                    width_z=width_z,
                    width_y=width_y,
                    width_x=width_x,
                    is_orthogonal=True,
                    reciprocal_space=True,
                    vmin=0,
                    title=title[idx] + " in the orthogonal frame",
                )

        # add the offset due to the detector angles to qx qz qy vectors, convert them to 1/A
        # the offset components are in the order (x/qy, y/qz, z/qx)
        qx = (qx + q_offset[2]) / 10  # along z downstream
        qz = (qz + q_offset[1]) / 10  # along y vertical up
        qy = (qy + q_offset[0]) / 10  # along x outboard

        if nb_arrays == 1:
            output_arrays = output_arrays[0]  # return the array instead of the tuple
        return output_arrays, (qx, qz, qy)

    def orthogonalize_vector(
        self, vector, array_shape, tilt_angle, pixel_x, pixel_y, verbose=False
    ):
        """
        Calculate the coordinates of the vector in the laboratory frame.

        :param vector: tuple of 3 coordinates, vector to be transformed in the detector frame
        :param array_shape: shape of the 3D array to orthogonalize
        :param tilt_angle: angular step during the rocking curve, in degrees
        :param pixel_x: horizontal pixel size, in meters
        :param pixel_y: vertical pixel size, in meters
        :param verbose: True to have printed comments
        :return: tuple of 3 numbers, the coordinates of the vector expressed in the laboratory frame
        """
        valid.valid_container(
            array_shape,
            container_types=(tuple, list),
            length=3,
            item_types=int,
            min_excluded=0,
            name="Setup.orthogonalize_vector",
        )

        ortho_matrix = self.transformation_matrix(
            array_shape=array_shape,
            tilt_angle=tilt_angle,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            verbose=verbose,
        )
        # ortho_matrix is the transformation matrix from the detector coordinates to the laboratory frame
        # Here, we want to calculate the coordinates that would have a vector of the laboratory frame expressed in the
        # detector frame, i.e. one has to inverse the transformation matrix.
        ortho_imatrix = np.linalg.inv(ortho_matrix)
        new_x = (
            ortho_imatrix[0, 0] * vector[2]
            + ortho_imatrix[0, 1] * vector[1]
            + ortho_imatrix[0, 2] * vector[0]
        )
        new_y = (
            ortho_imatrix[1, 0] * vector[2]
            + ortho_imatrix[1, 1] * vector[1]
            + ortho_imatrix[1, 2] * vector[0]
        )
        new_z = (
            ortho_imatrix[2, 0] * vector[2]
            + ortho_imatrix[2, 1] * vector[1]
            + ortho_imatrix[2, 2] * vector[0]
        )
        return new_z, new_y, new_x

    def transformation_matrix(
        self, array_shape, tilt_angle, pixel_x, pixel_y, direct_space=True, verbose=True
    ):
        """
        Calculate the transformation matrix from the detector frame to the laboratory frame. For direct space, the
        length scale is in nm, for reciprocal space, it is in 1/nm.

        :param array_shape: shape of the 3D array to orthogonalize
        :param tilt_angle: angular step during the rocking curve, in degrees
        :param pixel_x: horizontal pixel size, in meters
        :param pixel_y: vertical pixel size, in meters
        :param direct_space: True in order to return the transformation matrix in direct space
        :param verbose: True to have printed comments
        :return:
         - the transformation matrix from the detector frame to the laboratory frame
         - the q offset (3D vector) if direct_space is False.
        """
        if verbose:
            print(
                f"\nout-of plane detector angle={self.outofplane_angle:.3f} deg,"
                f" inplane_angle={self.inplane_angle:.3f} deg"
            )
        wavelength = self.wavelength * 1e9  # convert to nm
        distance = self.distance * 1e9  # convert to nm
        pixel_x = pixel_x * 1e9  # convert to nm
        pixel_y = pixel_y * 1e9  # convert to nm
        outofplane = np.radians(self.outofplane_angle)
        inplane = np.radians(self.inplane_angle)
        if self.grazing_angle is not None:
            grazing_angle = [np.radians(val) for val in self.grazing_angle]
        else:
            grazing_angle = None
        lambdaz = wavelength * distance
        mymatrix = np.zeros((3, 3))
        tilt = np.radians(tilt_angle)
        q_offset = np.zeros(3)
        nbz, nby, nbx = array_shape
        if self.detector_hor == "y-":  # inboard,  as it should be in the CXI convention
            hor_coeff = 1
        else:  # 'y+', outboard,  opposite to what it should be in the CXI convention
            hor_coeff = -1
        if (
            self.detector_ver == "z-"
        ):  # vertical down,  as it should be in the CXI convention
            ver_coeff = 1
        else:  # 'z+', vertical up,  opposite to what it should be in the CXI convention
            ver_coeff = -1

        if self.beamline == "ID01":
            if verbose:
                print("using ESRF ID01 PSIC geometry")
            if not isclose(grazing_angle[0], 0, rel_tol=1e-09, abs_tol=1e-09):
                raise NotImplementedError(
                    "Non-zero mu not implemented for the transformation matrices at ID01"
                )

            if self.rocking_angle == "outofplane":
                if verbose:
                    print(
                        f"rocking angle is eta, mu={grazing_angle[0]*180/np.pi:.3f} deg"
                    )
                # rocking eta angle clockwise around x (phi does not matter, above eta)
                mymatrix[:, 0] = (
                    2
                    * np.pi
                    / lambdaz
                    * hor_coeff
                    * np.array(
                        [-pixel_x * np.cos(inplane), 0, -pixel_x * np.sin(inplane)]
                    )
                )
                mymatrix[:, 1] = (
                    2
                    * np.pi
                    / lambdaz
                    * ver_coeff
                    * np.array(
                        [
                            -pixel_y * np.sin(inplane) * np.sin(outofplane),
                            -pixel_y * np.cos(outofplane),
                            pixel_y * np.cos(inplane) * np.sin(outofplane),
                        ]
                    )
                )
                mymatrix[:, 2] = (
                    2
                    * np.pi
                    / lambdaz
                    * np.array(
                        [
                            0,
                            tilt
                            * distance
                            * (1 - np.cos(inplane) * np.cos(outofplane)),
                            tilt * distance * np.sin(outofplane),
                        ]
                    )
                )
                q_offset[0] = (
                    -2
                    * np.pi
                    / lambdaz
                    * distance
                    * np.cos(outofplane)
                    * np.sin(inplane)
                )
                q_offset[1] = 2 * np.pi / lambdaz * distance * np.sin(outofplane)
                q_offset[2] = (
                    2
                    * np.pi
                    / lambdaz
                    * distance
                    * (np.cos(inplane) * np.cos(outofplane) - 1)
                )

            elif self.rocking_angle == "inplane":
                if verbose:
                    print(
                        f"rocking angle is phi,"
                        f" mu={grazing_angle[0]*180/np.pi:.3f} deg,"
                        f" eta={grazing_angle[1]*180/np.pi:.3f}deg"
                    )

                # rocking phi angle clockwise around y, incident angle eta is non zero (eta below phi)
                mymatrix[:, 0] = (
                    2
                    * np.pi
                    / lambdaz
                    * hor_coeff
                    * np.array(
                        [-pixel_x * np.cos(inplane), 0, -pixel_x * np.sin(inplane)]
                    )
                )
                mymatrix[:, 1] = (
                    2
                    * np.pi
                    / lambdaz
                    * ver_coeff
                    * np.array(
                        [
                            -pixel_y * np.sin(inplane) * np.sin(outofplane),
                            -pixel_y * np.cos(outofplane),
                            pixel_y * np.cos(inplane) * np.sin(outofplane),
                        ]
                    )
                )
                mymatrix[:, 2] = (
                    2
                    * np.pi
                    / lambdaz
                    * tilt
                    * distance
                    * np.array(
                        [
                            (
                                np.sin(grazing_angle[1]) * np.sin(outofplane)
                                + np.cos(grazing_angle[1])
                                * (np.cos(inplane) * np.cos(outofplane) - 1)
                            ),
                            np.sin(grazing_angle[1])
                            * np.sin(inplane)
                            * np.cos(outofplane),
                            np.cos(grazing_angle[1])
                            * np.sin(inplane)
                            * np.cos(outofplane),
                        ]
                    )
                )
                q_offset[0] = (
                    -2
                    * np.pi
                    / lambdaz
                    * distance
                    * np.cos(outofplane)
                    * np.sin(inplane)
                )
                q_offset[1] = 2 * np.pi / lambdaz * distance * np.sin(outofplane)
                q_offset[2] = (
                    2
                    * np.pi
                    / lambdaz
                    * distance
                    * (np.cos(inplane) * np.cos(outofplane) - 1)
                )

        if self.beamline == "P10":
            if verbose:
                print("using PETRAIII P10 geometry")
            if self.rocking_angle == "outofplane":
                if verbose:
                    print(
                        f"rocking angle is om, mu={grazing_angle[0]*180/np.pi:.3f} deg"
                    )
                # rocking omega angle clockwise around x at mu=0, chi potentially non zero (chi below omega)
                # (phi does not matter, above eta)
                mymatrix[:, 0] = (
                    2
                    * np.pi
                    / lambdaz
                    * pixel_x
                    * hor_coeff
                    * np.array([-np.cos(inplane), 0, np.sin(inplane)])
                )
                mymatrix[:, 1] = (
                    2
                    * np.pi
                    / lambdaz
                    * pixel_y
                    * ver_coeff
                    * np.array(
                        [
                            np.sin(inplane) * np.sin(outofplane),
                            -np.cos(outofplane),
                            np.cos(inplane) * np.sin(outofplane),
                        ]
                    )
                )
                mymatrix[:, 2] = (
                    2
                    * np.pi
                    / lambdaz
                    * tilt
                    * distance
                    * np.array(
                        [
                            np.sin(grazing_angle[0]) * np.sin(outofplane),
                            np.cos(grazing_angle[0])
                            * (1 - np.cos(inplane) * np.cos(outofplane))
                            - np.sin(grazing_angle[0])
                            * np.cos(outofplane)
                            * np.sin(inplane),
                            np.sin(outofplane) * np.cos(grazing_angle[0]),
                        ]
                    )
                )
                q_offset[0] = (
                    2
                    * np.pi
                    / lambdaz
                    * distance
                    * np.cos(outofplane)
                    * np.sin(inplane)
                )
                q_offset[1] = 2 * np.pi / lambdaz * distance * np.sin(outofplane)
                q_offset[2] = (
                    2
                    * np.pi
                    / lambdaz
                    * distance
                    * (np.cos(inplane) * np.cos(outofplane) - 1)
                )

            elif self.rocking_angle == "inplane":
                if not isclose(grazing_angle[0], 0, rel_tol=1e-09, abs_tol=1e-09):
                    raise NotImplementedError(
                        "Non-zero mu not implemented for inplane rocking curve at P10"
                    )
                if verbose:
                    print(
                        f"rocking angle is phi,"
                        f" mu={grazing_angle[0]*180/np.pi:.3f} deg,"
                        f" om={grazing_angle[1]*180/np.pi:.3f} deg,"
                        f" chi={grazing_angle[2]*180/np.pi:.3f} deg"
                    )

                # rocking phi angle clockwise around y, omega and chi potentially non zero (chi below omega below phi)
                mymatrix[:, 0] = (
                    2
                    * np.pi
                    / lambdaz
                    * pixel_x
                    * hor_coeff
                    * np.array([-np.cos(inplane), 0, np.sin(inplane)])
                )
                mymatrix[:, 1] = (
                    2
                    * np.pi
                    / lambdaz
                    * pixel_y
                    * ver_coeff
                    * np.array(
                        [
                            np.sin(inplane) * np.sin(outofplane),
                            -np.cos(outofplane),
                            np.cos(inplane) * np.sin(outofplane),
                        ]
                    )
                )
                mymatrix[:, 2] = (
                    2
                    * np.pi
                    / lambdaz
                    * tilt
                    * distance
                    * np.array(
                        [
                            (
                                np.sin(grazing_angle[1])
                                * np.cos(grazing_angle[2])
                                * np.sin(outofplane)
                                + np.cos(grazing_angle[1])
                                * np.cos(grazing_angle[2])
                                * (np.cos(inplane) * np.cos(outofplane) - 1)
                            ),
                            (
                                -np.sin(grazing_angle[1])
                                * np.cos(grazing_angle[2])
                                * np.sin(inplane)
                                * np.cos(outofplane)
                                + np.sin(grazing_angle[2])
                                * (np.cos(inplane) * np.cos(outofplane) - 1)
                            ),
                            (
                                -np.cos(grazing_angle[1])
                                * np.cos(grazing_angle[2])
                                * np.sin(inplane)
                                * np.cos(outofplane)
                                - np.sin(grazing_angle[2]) * np.sin(outofplane)
                            ),
                        ]
                    )
                )
                q_offset[0] = (
                    2
                    * np.pi
                    / lambdaz
                    * distance
                    * np.cos(outofplane)
                    * np.sin(inplane)
                )
                q_offset[1] = 2 * np.pi / lambdaz * distance * np.sin(outofplane)
                q_offset[2] = (
                    2
                    * np.pi
                    / lambdaz
                    * distance
                    * (np.cos(inplane) * np.cos(outofplane) - 1)
                )

        if self.beamline == "NANOMAX":
            if verbose:
                print("using NANOMAX geometry")

            if self.rocking_angle == "outofplane":
                if grazing_angle is not None:
                    raise NotImplementedError(
                        "Circle below theta not implemented for NANOMAX"
                    )
                if verbose:
                    print("rocking angle is theta")
                # rocking theta angle clockwise around x (phi does not matter, above eta)
                mymatrix[:, 0] = (
                    2
                    * np.pi
                    / lambdaz
                    * pixel_x
                    * hor_coeff
                    * np.array([-np.cos(inplane), 0, -np.sin(inplane)])
                )
                mymatrix[:, 1] = (
                    2
                    * np.pi
                    / lambdaz
                    * pixel_y
                    * ver_coeff
                    * np.array(
                        [
                            -np.sin(inplane) * np.sin(outofplane),
                            -np.cos(outofplane),
                            np.cos(inplane) * np.sin(outofplane),
                        ]
                    )
                )
                mymatrix[:, 2] = (
                    2
                    * np.pi
                    / lambdaz
                    * tilt
                    * distance
                    * np.array(
                        [
                            0,
                            1 - np.cos(inplane) * np.cos(outofplane),
                            np.sin(outofplane),
                        ]
                    )
                )
                q_offset[0] = (
                    -2
                    * np.pi
                    / lambdaz
                    * distance
                    * np.cos(outofplane)
                    * np.sin(inplane)
                )
                q_offset[1] = 2 * np.pi / lambdaz * distance * np.sin(outofplane)
                q_offset[2] = (
                    2
                    * np.pi
                    / lambdaz
                    * distance
                    * (np.cos(inplane) * np.cos(outofplane) - 1)
                )

            elif self.rocking_angle == "inplane":
                if verbose:
                    print(
                        f"rocking angle is phi, theta={grazing_angle[0]*180/np.pi:.3f} deg"
                    )
                # rocking phi angle clockwise around y, incident angle theta is non zero (theta below phi)
                mymatrix[:, 0] = (
                    2
                    * np.pi
                    / lambdaz
                    * pixel_x
                    * hor_coeff
                    * np.array([-np.cos(inplane), 0, -np.sin(inplane)])
                )
                mymatrix[:, 1] = (
                    2
                    * np.pi
                    / lambdaz
                    * pixel_y
                    * ver_coeff
                    * np.array(
                        [
                            -np.sin(inplane) * np.sin(outofplane),
                            -np.cos(outofplane),
                            np.cos(inplane) * np.sin(outofplane),
                        ]
                    )
                )
                mymatrix[:, 2] = (
                    2
                    * np.pi
                    / lambdaz
                    * tilt
                    * distance
                    * np.array(
                        [
                            (
                                np.sin(grazing_angle[0]) * np.sin(outofplane)
                                + np.cos(grazing_angle[0])
                                * (np.cos(inplane) * np.cos(outofplane) - 1)
                            ),
                            np.sin(grazing_angle[0])
                            * np.sin(inplane)
                            * np.cos(outofplane),
                            np.cos(grazing_angle[0])
                            * np.sin(inplane)
                            * np.cos(outofplane),
                        ]
                    )
                )
                q_offset[0] = (
                    -2
                    * np.pi
                    / lambdaz
                    * distance
                    * np.cos(outofplane)
                    * np.sin(inplane)
                )
                q_offset[1] = 2 * np.pi / lambdaz * distance * np.sin(outofplane)
                q_offset[2] = (
                    2
                    * np.pi
                    / lambdaz
                    * distance
                    * (np.cos(inplane) * np.cos(outofplane) - 1)
                )

        if self.beamline == "34ID":
            if verbose:
                print("using APS 34ID geometry")
            if self.rocking_angle == "inplane":
                if grazing_angle is not None:
                    raise NotImplementedError(
                        "Circle blow theta not implemented for 34ID-C"
                    )
                if verbose:
                    print("rocking angle is theta, no grazing angle (phi above theta)")
                # rocking theta angle anti-clockwise around y
                mymatrix[:, 0] = (
                    2
                    * np.pi
                    / lambdaz
                    * pixel_x
                    * hor_coeff
                    * np.array([-np.cos(inplane), 0, np.sin(inplane)])
                )
                mymatrix[:, 1] = (
                    2
                    * np.pi
                    / lambdaz
                    * pixel_y
                    * ver_coeff
                    * np.array(
                        [
                            np.sin(inplane) * np.sin(outofplane),
                            -np.cos(outofplane),
                            np.cos(inplane) * np.sin(outofplane),
                        ]
                    )
                )
                mymatrix[:, 2] = (
                    2
                    * np.pi
                    / lambdaz
                    * tilt
                    * distance
                    * np.array(
                        [
                            1 - np.cos(inplane) * np.cos(outofplane),
                            0,
                            np.sin(inplane) * np.cos(outofplane),
                        ]
                    )
                )
                q_offset[0] = (
                    2
                    * np.pi
                    / lambdaz
                    * distance
                    * np.cos(outofplane)
                    * np.sin(inplane)
                )
                q_offset[1] = 2 * np.pi / lambdaz * distance * np.sin(outofplane)
                q_offset[2] = (
                    2
                    * np.pi
                    / lambdaz
                    * distance
                    * (np.cos(inplane) * np.cos(outofplane) - 1)
                )

            elif self.rocking_angle == "outofplane":
                if verbose:
                    print(
                        f"rocking angle is phi, theta={grazing_angle[0] * 180 / np.pi:.3f} deg"
                    )
                # rocking phi angle anti-clockwise around x
                mymatrix[:, 0] = (
                    2
                    * np.pi
                    / lambdaz
                    * pixel_x
                    * hor_coeff
                    * np.array([-np.cos(inplane), 0, np.sin(inplane)])
                )
                mymatrix[:, 1] = (
                    2
                    * np.pi
                    / lambdaz
                    * pixel_y
                    * ver_coeff
                    * np.array(
                        [
                            np.sin(inplane) * np.sin(outofplane),
                            -np.cos(outofplane),
                            np.cos(inplane) * np.sin(outofplane),
                        ]
                    )
                )
                mymatrix[:, 2] = (
                    2
                    * np.pi
                    / lambdaz
                    * tilt
                    * distance
                    * np.array(
                        [
                            -np.sin(grazing_angle[0]) * np.sin(outofplane),
                            np.cos(grazing_angle[0])
                            * (np.cos(inplane) * np.cos(outofplane) - 1),
                            -np.cos(grazing_angle[0]) * np.sin(outofplane),
                        ]
                    )
                )
                q_offset[0] = (
                    2
                    * np.pi
                    / lambdaz
                    * distance
                    * np.cos(outofplane)
                    * np.sin(inplane)
                )
                q_offset[1] = 2 * np.pi / lambdaz * distance * np.sin(outofplane)
                q_offset[2] = (
                    2
                    * np.pi
                    / lambdaz
                    * distance
                    * (np.cos(inplane) * np.cos(outofplane) - 1)
                )

        if self.beamline in {"SIXS_2018", "SIXS_2019"}:
            if verbose:
                print("using SIXS geometry")

            if self.rocking_angle == "inplane":
                if verbose:
                    print(
                        f"rocking angle is mu, beta={grazing_angle[0] * 180 / np.pi:.3f} deg"
                    )

                # rocking mu angle anti-clockwise around y
                mymatrix[:, 0] = (
                    2
                    * np.pi
                    / lambdaz
                    * pixel_x
                    * hor_coeff
                    * np.array(
                        [
                            -np.cos(inplane),
                            np.sin(grazing_angle[0]) * np.sin(inplane),
                            np.cos(grazing_angle[0]) * np.sin(inplane),
                        ]
                    )
                )
                mymatrix[:, 1] = (
                    2
                    * np.pi
                    / lambdaz
                    * pixel_y
                    * ver_coeff
                    * np.array(
                        [
                            np.sin(inplane) * np.sin(outofplane),
                            (
                                np.sin(grazing_angle[0])
                                * np.cos(inplane)
                                * np.sin(outofplane)
                                - np.cos(grazing_angle[0]) * np.cos(outofplane)
                            ),
                            (
                                np.cos(grazing_angle[0])
                                * np.cos(inplane)
                                * np.sin(outofplane)
                                + np.sin(grazing_angle[0]) * np.cos(outofplane)
                            ),
                        ]
                    )
                )
                mymatrix[:, 2] = (
                    2
                    * np.pi
                    / lambdaz
                    * tilt
                    * distance
                    * np.array(
                        [
                            np.cos(grazing_angle[0])
                            - np.cos(inplane) * np.cos(outofplane),
                            np.sin(grazing_angle[0])
                            * np.sin(inplane)
                            * np.cos(outofplane),
                            np.cos(grazing_angle[0])
                            * np.sin(inplane)
                            * np.cos(outofplane),
                        ]
                    )
                )
                q_offset[0] = (
                    2
                    * np.pi
                    / lambdaz
                    * distance
                    * np.cos(outofplane)
                    * np.sin(inplane)
                )
                q_offset[1] = (
                    2
                    * np.pi
                    / lambdaz
                    * distance
                    * (
                        np.cos(grazing_angle[0]) * np.sin(outofplane)
                        + np.sin(grazing_angle[0])
                        * np.cos(inplane)
                        * np.cos(outofplane)
                    )
                )
                q_offset[2] = (
                    2
                    * np.pi
                    / lambdaz
                    * distance
                    * (
                        np.cos(grazing_angle[0]) * np.cos(inplane) * np.cos(outofplane)
                        - np.sin(grazing_angle[0]) * np.sin(outofplane)
                        - 1
                    )
                )
            else:
                raise NotImplementedError(
                    "out of plane rocking curve not implemented for SIXS"
                )

        if self.beamline == "CRISTAL":
            if verbose:
                print("using CRISTAL geometry")

            if self.rocking_angle == "outofplane":
                if grazing_angle is not None:
                    raise NotImplementedError(
                        "Circle below mgomega not implemented for CRISTAL"
                    )
                if verbose:
                    print("rocking angle is mgomega")
                # rocking mgomega angle clockwise around x
                mymatrix[:, 0] = (
                    2
                    * np.pi
                    / lambdaz
                    * pixel_x
                    * hor_coeff
                    * np.array([-np.cos(inplane), 0, np.sin(inplane)])
                )
                mymatrix[:, 1] = (
                    2
                    * np.pi
                    / lambdaz
                    * pixel_y
                    * ver_coeff
                    * np.array(
                        [
                            np.sin(inplane) * np.sin(outofplane),
                            -np.cos(outofplane),
                            np.cos(inplane) * np.sin(outofplane),
                        ]
                    )
                )
                mymatrix[:, 2] = (
                    2
                    * np.pi
                    / lambdaz
                    * tilt
                    * distance
                    * np.array(
                        [
                            0,
                            1 - np.cos(inplane) * np.cos(outofplane),
                            np.sin(outofplane),
                        ]
                    )
                )
                q_offset[0] = (
                    2
                    * np.pi
                    / lambdaz
                    * distance
                    * np.cos(outofplane)
                    * np.sin(inplane)
                )
                q_offset[1] = 2 * np.pi / lambdaz * distance * np.sin(outofplane)
                q_offset[2] = (
                    2
                    * np.pi
                    / lambdaz
                    * distance
                    * (np.cos(inplane) * np.cos(outofplane) - 1)
                )

            elif self.rocking_angle == "inplane":
                if verbose:
                    print(
                        f"rocking angle is phi, mgomega={grazing_angle[0]*180/np.pi:.3f} deg"
                    )
                # rocking phi angle anti-clockwise around y, incident angle mgomega is non zero (mgomega below phi)
                mymatrix[:, 0] = (
                    2
                    * np.pi
                    / lambdaz
                    * pixel_x
                    * hor_coeff
                    * np.array([-np.cos(inplane), 0, np.sin(inplane)])
                )
                mymatrix[:, 1] = (
                    2
                    * np.pi
                    / lambdaz
                    * pixel_y
                    * ver_coeff
                    * np.array(
                        [
                            np.sin(inplane) * np.sin(outofplane),
                            -np.cos(outofplane),
                            np.cos(inplane) * np.sin(outofplane),
                        ]
                    )
                )
                mymatrix[:, 2] = (
                    2
                    * np.pi
                    / lambdaz
                    * tilt
                    * distance
                    * np.array(
                        [
                            (
                                -np.sin(grazing_angle[0]) * np.sin(outofplane)
                                - np.cos(grazing_angle[0])
                                * (np.cos(inplane) * np.cos(outofplane) - 1)
                            ),
                            np.sin(grazing_angle[0])
                            * np.sin(inplane)
                            * np.cos(outofplane),
                            np.cos(grazing_angle[0])
                            * np.sin(inplane)
                            * np.cos(outofplane),
                        ]
                    )
                )
                q_offset[0] = (
                    2
                    * np.pi
                    / lambdaz
                    * distance
                    * np.cos(outofplane)
                    * np.sin(inplane)
                )
                q_offset[1] = 2 * np.pi / lambdaz * distance * np.sin(outofplane)
                q_offset[2] = (
                    2
                    * np.pi
                    / lambdaz
                    * distance
                    * (np.cos(inplane) * np.cos(outofplane) - 1)
                )

        if direct_space:  # length scale in nm
            # for a discrete FT, the dimensions of the basis vectors after the transformation are related to the total
            # domain size
            mymatrix[:, 0] = nbx * mymatrix[:, 0]
            mymatrix[:, 1] = nby * mymatrix[:, 1]
            mymatrix[:, 2] = nbz * mymatrix[:, 2]
            return 2 * np.pi * np.linalg.inv(mymatrix).transpose()
        # reciprocal length scale in  1/nm
        return mymatrix, q_offset

    def voxel_sizes(self, array_shape, tilt_angle, pixel_x, pixel_y, verbose=False):
        """
        Calculate the direct space voxel sizes in the laboratory frame (z downstream, y vertical up, x outboard).

        :param array_shape: shape of the 3D array to orthogonalize
        :param tilt_angle: angular step during the rocking curve, in degrees
        :param pixel_x: horizontal pixel size, in meters
        :param pixel_y: vertical pixel size, in meters
        :param verbose: True to have printed comments
        :return: the direct space voxel sizes in nm, in the laboratory frame (voxel_z, voxel_y, voxel_x)
        """
        valid.valid_container(
            array_shape,
            container_types=(tuple, list),
            length=3,
            item_types=int,
            min_excluded=0,
            name="Setup.voxel_sizes",
        )

        transfer_matrix = self.transformation_matrix(
            array_shape=array_shape,
            tilt_angle=tilt_angle,
            direct_space=True,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            verbose=verbose,
        )
        # transfer_matrix is the transformation matrix of the direct space coordinates (its columns are the
        # non-orthogonal basis vectors reciprocal to the detector frame)
        # the spacing in the laboratory frame is therefore given by the rows of the matrix
        dx = np.linalg.norm(transfer_matrix[0, :])  # along x outboard
        dy = np.linalg.norm(transfer_matrix[1, :])  # along y vertical up
        dz = np.linalg.norm(transfer_matrix[2, :])  # along z downstream

        if verbose:
            print(
                f"Direct space voxel size (z, y, x) = ({dz:.2f}, {dy:.2f}, {dx:.2f}) (nm)"
            )
        return dz, dy, dx

    def voxel_sizes_detector(
        self, array_shape, tilt_angle, pixel_x, pixel_y, verbose=False
    ):
        """
        Calculate the direct space voxel sizes in the detector frame
         (z rocking angle, y detector vertical axis, x detector horizontal axis).

        :param array_shape: shape of the 3D array used in phase retrieval
        :param tilt_angle: angular step during the rocking curve, in degrees
        :param pixel_x: horizontal pixel size, in meters
        :param pixel_y: vertical pixel size, in meters
        :param verbose: True to have printed comments
        :return: the direct space voxel sizes in nm, in the detector frame (voxel_z, voxel_y, voxel_x)
        """
        voxel_z = (
            self.wavelength / (array_shape[0] * abs(tilt_angle) * np.pi / 180) * 1e9
        )  # in nm
        voxel_y = (
            self.wavelength * self.distance / (array_shape[1] * pixel_y) * 1e9
        )  # in nm
        voxel_x = (
            self.wavelength * self.distance / (array_shape[2] * pixel_x) * 1e9
        )  # in nm
        if verbose:
            print(
                "voxelsize_z, voxelsize_y, voxelsize_x="
                "({0:.2f}, {1:.2f}, {2:.2f}) (1/nm)".format(voxel_z, voxel_y, voxel_x)
            )
        return voxel_z, voxel_y, voxel_x


def higher_primes(number, maxprime=13, required_dividers=(4,)):
    """
    Find the closest integer >=n (or list/array of integers), for which the largest prime divider is <=maxprime,
    and has to include some dividers. The default values for maxprime is the largest integer accepted
    by the clFFT library for OpenCL GPU FFT. Adapted from PyNX.

    :param number: the integer number
    :param maxprime: the largest prime factor acceptable
    :param required_dividers: a list of required dividers for the returned integer.
    :return: the integer (or list/array of integers) fulfilling the requirements
    """
    if isinstance(number, (list, tuple, np.ndarray)):
        vn = []
        for i in number:
            limit = i
            if i <= 1 or maxprime > i:
                raise ValueError(f"Number is < {maxprime}")
            while (
                try_smaller_primes(
                    i, maxprime=maxprime, required_dividers=required_dividers
                )
                is False
            ):
                i = i + 1
                if i == limit:
                    return limit
            vn.append(i)
        if isinstance(number, np.ndarray):
            return np.array(vn)
        return vn
    limit = number
    if number <= 1 or maxprime > number:
        raise ValueError(f"Number is < {maxprime}")
    while (
        try_smaller_primes(
            number, maxprime=maxprime, required_dividers=required_dividers
        )
        is False
    ):
        number = number + 1
        if number == limit:
            return limit
    return number


def primes(number):
    """
    Returns the prime decomposition of n as a list. Adapted from PyNX.

    :param number: the integer to be decomposed
    :return: the list of prime dividers of number
    """
    valid.valid_item(
        number, allowed_types=int, min_excluded=0, name="preprocessing_utils.primes"
    )
    list_primes = [1]
    i = 2
    while i * i <= number:
        while number % i == 0:
            list_primes.append(i)
            number //= i
        i += 1
    if number > 1:
        list_primes.append(number)
    return list_primes


def smaller_primes(number, maxprime=13, required_dividers=(4,)):
    """
    Find the closest integer <=n (or list/array of integers), for which the largest prime divider is <=maxprime,
    and has to include some dividers. The default values for maxprime is the largest integer accepted
    by the clFFT library for OpenCL GPU FFT. Adapted from PyNX.

    :param number: the integer number
    :param maxprime: the largest prime factor acceptable
    :param required_dividers: a list of required dividers for the returned integer.
    :return: the integer (or list/array of integers) fulfilling the requirements
    """
    if isinstance(number, (list, tuple, np.ndarray)):
        vn = []
        for i in number:
            if i <= 1 or maxprime > i:
                raise ValueError(f"Number is < {maxprime}")
            while (
                try_smaller_primes(
                    i, maxprime=maxprime, required_dividers=required_dividers
                )
                is False
            ):
                i = i - 1
                if i == 0:
                    return 0
            vn.append(i)
        if isinstance(number, np.ndarray):
            return np.array(vn)
        return vn
    if number <= 1 or maxprime > number:
        raise ValueError(f"Number is < {maxprime}")
    while (
        try_smaller_primes(
            number, maxprime=maxprime, required_dividers=required_dividers
        )
        is False
    ):
        number = number - 1
        if number == 0:
            return 0
    return number


def try_smaller_primes(number, maxprime=13, required_dividers=(4,)):
    """
    Check if the largest prime divider is <=maxprime, and optionally includes some dividers. Adapted from PyNX.

    :param number: the integer number for which the prime decomposition will be checked
    :param maxprime: the maximum acceptable prime number. This defaults to the largest integer accepted by the clFFT
        library for OpenCL GPU FFT.
    :param required_dividers: list of required dividers in the prime decomposition. If None, this check is skipped.
    :return: True if the conditions are met.
    """
    p = primes(number)
    if max(p) > maxprime:
        return False
    if required_dividers is not None:
        for k in required_dividers:
            if number % k != 0:
                return False
    return True


class SetupPostprocessing:
    """
    Class to handle the experimental geometry for postprocessing.
    """

    def __init__(
        self,
        beamline,
        energy,
        outofplane_angle,
        inplane_angle,
        tilt_angle,
        rocking_angle,
        distance,
        grazing_angle=0,
        pixel_x=55e-6,
        pixel_y=55e-6,
    ):
        """
        Initialize parameters of the experiment.

        :param beamline: name of the beamline: 'ID01', 'SIXS_2018', 'SIXS_2019', '34ID', 'P10', 'CRISTAL', 'NANOMAX'
        :param energy: X-ray energy in eV
        :param outofplane_angle: out of plane angle of the detector in degrees
        :param inplane_angle: inplane angle of the detector in degrees
        :param tilt_angle: angular step of the sample during the rocking curve, in degrees
        :param rocking_angle: name of the angle which is tilted during the rocking curve, 'outofplane' or 'inplane'
        :param distance: sample to detector distance in meters
        :param grazing_angle: grazing angle for in-plane rocking curves (eta ID01, th 34ID, beta SIXS), in degrees
        :param pixel_x: horizontal pixel size, in meters
        :param pixel_y: vertical pixel size, in meters
        """
        warnings.warn("deprecated, use the class Setup instead", DeprecationWarning)
        self.beamline = beamline  # string
        self.energy = energy  # in eV
        self.wavelength = 12.398 * 1e-7 / energy  # in m
        self.outofplane_angle = outofplane_angle  # in degrees
        self.inplane_angle = inplane_angle  # in degrees
        self.tilt_angle = tilt_angle  # in degrees
        self.rocking_angle = rocking_angle  # string
        self.grazing_angle = grazing_angle  # in degrees
        self.distance = distance  # in meters
        self.pixel_x = pixel_x  # in meters
        self.pixel_y = pixel_y  # in meters

        #############################################################
        # detector orientation convention depending on the beamline #
        #############################################################
        # the frame convention is the one of xrayutilities: x downstream, y outboard, z vertical up

        # horizontal axis:
        if beamline in {"ID01", "SIXS_2018", "SIXS_2019", "CRISTAL", "NANOMAX"}:
            # we look at the detector from downstream, detector X along the outboard direction
            self.detector_hor = "y+"
        else:  # 'P10', '34ID'
            # we look at the detector from upstream, detector X opposite to the outboard direction
            self.detector_hor = "y-"

        # vertical axis:
        # origin is at the top, detector Y along vertical down
        self.detector_ver = "z-"

    def __repr__(self):
        """
        :return: a nicely formatted representation string
        """
        return (
            f"{self.__class__.__name__}: beamline={self.beamline}, energy={self.energy}eV,"
            f" sample to detector distance={self.distance}m, pixel size (VxH)=({self.pixel_y},{self.pixel_x})"
        )

    def detector_frame(
        self,
        obj,
        voxelsize,
        width_z=None,
        width_y=None,
        width_x=None,
        debugging=False,
        **kwargs,
    ):
        """
        Interpolate the orthogonal object back into the non-orthogonal detector frame

        :param obj: real space object, in the orthogonal laboratory frame
        :param voxelsize: voxel size of the original object
        :param width_z: size of the area to plot in z (axis 0), centered on the middle of the initial array
        :param width_y: size of the area to plot in y (axis 1), centered on the middle of the initial array
        :param width_x: size of the area to plot in x (axis 2), centered on the middle of the initial array
        :param debugging: True to show plots before and after interpolation
        :param kwargs:
         - 'title': title for the debugging plots
        :return: object interpolated on an orthogonal grid
        """
        title = kwargs.get("title", "Object")

        for k in kwargs:
            if k not in {"title"}:
                raise Exception("unknown keyword argument given:", k)

        nbz, nby, nbx = obj.shape

        if debugging:
            gu.multislices_plot(
                abs(obj),
                sum_frames=True,
                width_z=width_z,
                width_y=width_y,
                width_x=width_x,
                title=title + " before interpolation\n",
            )

        ortho_matrix = self.update_coords(
            array_shape=(nbz, nby, nbx),
            tilt_angle=self.tilt_angle,
            pixel_x=self.pixel_x,
            pixel_y=self.pixel_y,
        )

        ################################################
        # interpolate the data into the detector frame #
        ################################################
        myz, myy, myx = np.meshgrid(
            np.arange(-nbz // 2, nbz // 2, 1),
            np.arange(-nby // 2, nby // 2, 1),
            np.arange(-nbx // 2, nbx // 2, 1),
            indexing="ij",
        )

        new_x = (
            ortho_matrix[0, 0] * myx
            + ortho_matrix[0, 1] * myy
            + ortho_matrix[0, 2] * myz
        )
        new_y = (
            ortho_matrix[1, 0] * myx
            + ortho_matrix[1, 1] * myy
            + ortho_matrix[1, 2] * myz
        )
        new_z = (
            ortho_matrix[2, 0] * myx
            + ortho_matrix[2, 1] * myy
            + ortho_matrix[2, 2] * myz
        )
        del myx, myy, myz
        # la partie rgi est sure: c'est la taille de l'objet orthogonal de depart
        rgi = RegularGridInterpolator(
            (
                np.arange(-nbz // 2, nbz // 2) * voxelsize,
                np.arange(-nby // 2, nby // 2) * voxelsize,
                np.arange(-nbx // 2, nbx // 2) * voxelsize,
            ),
            obj,
            method="linear",
            bounds_error=False,
            fill_value=0,
        )
        detector_obj = rgi(
            np.concatenate(
                (
                    new_z.reshape((1, new_z.size)),
                    new_y.reshape((1, new_z.size)),
                    new_x.reshape((1, new_z.size)),
                )
            ).transpose()
        )
        detector_obj = detector_obj.reshape((nbz, nby, nbx)).astype(obj.dtype)

        if debugging:
            gu.multislices_plot(
                abs(detector_obj),
                sum_frames=True,
                width_z=width_z,
                width_y=width_y,
                width_x=width_x,
                title=title + " interpolated in detector frame\n",
            )

        return detector_obj

    def exit_wavevector(self):
        """
        Calculate the exit wavevector kout depending on the setup parameters, in laboratory frame (z downstream,
         y vertical, x outboard).

        :return: kout vector
        """
        if self.beamline in {"SIXS_2018", "SIXS_2019"}:
            # gamma is anti-clockwise
            kout = (
                2
                * np.pi
                / self.wavelength
                * np.array(
                    [
                        np.cos(np.pi * self.inplane_angle / 180)
                        * np.cos(np.pi * self.outofplane_angle / 180),  # z
                        np.sin(np.pi * self.outofplane_angle / 180),  # y
                        np.sin(np.pi * self.inplane_angle / 180)
                        * np.cos(np.pi * self.outofplane_angle / 180),
                    ]
                )
            )  # x
        elif self.beamline == "ID01":
            # nu is clockwise
            kout = (
                2
                * np.pi
                / self.wavelength
                * np.array(
                    [
                        np.cos(np.pi * self.inplane_angle / 180)
                        * np.cos(np.pi * self.outofplane_angle / 180),  # z
                        np.sin(np.pi * self.outofplane_angle / 180),  # y
                        -np.sin(np.pi * self.inplane_angle / 180)
                        * np.cos(np.pi * self.outofplane_angle / 180),
                    ]
                )
            )  # x
        elif self.beamline == "34ID":
            # gamma is anti-clockwise
            kout = (
                2
                * np.pi
                / self.wavelength
                * np.array(
                    [
                        np.cos(np.pi * self.inplane_angle / 180)
                        * np.cos(np.pi * self.outofplane_angle / 180),  # z
                        np.sin(np.pi * self.outofplane_angle / 180),  # y
                        np.sin(np.pi * self.inplane_angle / 180)
                        * np.cos(np.pi * self.outofplane_angle / 180),
                    ]
                )
            )  # x
        elif self.beamline == "NANOMAX":
            # gamma is clockwise
            kout = (
                2
                * np.pi
                / self.wavelength
                * np.array(
                    [
                        np.cos(np.pi * self.inplane_angle / 180)
                        * np.cos(np.pi * self.outofplane_angle / 180),  # z
                        np.sin(np.pi * self.outofplane_angle / 180),  # y
                        -np.sin(np.pi * self.inplane_angle / 180)
                        * np.cos(np.pi * self.outofplane_angle / 180),
                    ]
                )
            )  # x
        elif self.beamline == "P10":
            # gamma is anti-clockwise
            kout = (
                2
                * np.pi
                / self.wavelength
                * np.array(
                    [
                        np.cos(np.pi * self.inplane_angle / 180)
                        * np.cos(np.pi * self.outofplane_angle / 180),  # z
                        np.sin(np.pi * self.outofplane_angle / 180),  # y
                        np.sin(np.pi * self.inplane_angle / 180)
                        * np.cos(np.pi * self.outofplane_angle / 180),
                    ]
                )
            )  # x
        elif self.beamline == "CRISTAL":
            # gamma is anti-clockwise
            kout = (
                2
                * np.pi
                / self.wavelength
                * np.array(
                    [
                        np.cos(np.pi * self.inplane_angle / 180)
                        * np.cos(np.pi * self.outofplane_angle / 180),  # z
                        np.sin(np.pi * self.outofplane_angle / 180),  # y
                        np.sin(np.pi * self.inplane_angle / 180)
                        * np.cos(np.pi * self.outofplane_angle / 180),
                    ]
                )
            )  # x
        else:
            raise ValueError("setup parameter: ", self.beamline, "not defined")
        return kout

    def inplane_coeff(self):
        """
        Define a coefficient +/- 1 depending on the detector inplane rotation direction and the detector inplane
         orientation. The frame convention is the one of xrayutilities: x downstream, y outboard, z vertical up.
         See postprocessing/scripts/correct_angles_detector.py for an example.

        :return: +1 or -1
        """
        if self.detector_hor == "y+":
            hor_coeff = 1
        else:  # 'y-'
            hor_coeff = -1

        if self.beamline in {"SIXS_2018", "SIXS_2019"}:
            # gamma is anti-clockwise, we see the detector from downstream
            coeff_inplane = 1 * hor_coeff
        elif self.beamline == "ID01":
            # nu is clockwise, we see the detector from downstream
            coeff_inplane = -1 * hor_coeff
        elif self.beamline == "34ID":
            # delta is anti-clockwise, we see the detector from the front
            coeff_inplane = 1 * hor_coeff
        elif self.beamline == "P10":
            # gamma is anti-clockwise, we see the detector from the front
            coeff_inplane = 1 * hor_coeff
        elif self.beamline == "CRISTAL":
            # gamma is anti-clockwise, we see the detector from downstream
            coeff_inplane = 1 * hor_coeff
        elif self.beamline == "NANOMAX":
            # gamma is clockwise, we see the detector from downstream
            coeff_inplane = -1 * hor_coeff
        else:
            raise ValueError("setup parameter: ", self.beamline, "not defined")
        return coeff_inplane

    def orthogonalize(
        self,
        obj,
        initial_shape=(),
        voxel_size=np.nan,
        width_z=None,
        width_y=None,
        width_x=None,
        verbose=True,
        debugging=False,
        **kwargs,
    ):
        """
        Interpolate obj on the orthogonal reference frame defined by the setup.

        :param obj: real space object, in a non-orthogonal frame (output of phasing program)
        :param initial_shape: shape of the FFT used for phasing
        :param voxel_size: user-defined voxel size, in nm
        :param width_z: size of the area to plot in z (axis 0), centered on the middle of the initial array
        :param width_y: size of the area to plot in y (axis 1), centered on the middle of the initial array
        :param width_x: size of the area to plot in x (axis 2), centered on the middle of the initial array
        :param verbose: True to have printed comments
        :param debugging: True to show plots before and after interpolation
        :param kwargs:
         - 'title': title for the debugging plots
        :return: object interpolated on an orthogonal grid
        """
        title = kwargs.get("title", "Object")

        for k in kwargs:
            if k not in {"title"}:
                raise Exception("unknown keyword argument given:", k)

        if len(initial_shape) == 0:
            initial_shape = obj.shape

        if debugging:
            gu.multislices_plot(
                abs(obj),
                sum_frames=True,
                width_z=width_z,
                width_y=width_y,
                width_x=width_x,
                title=title + " in detector frame",
            )

        # estimate the direct space voxel sizes in nm based on the FFT window shape used in phase retrieval
        dz_realspace, dy_realspace, dx_realspace = self.voxel_sizes(
            initial_shape,
            tilt_angle=abs(self.tilt_angle),
            pixel_x=self.pixel_x,
            pixel_y=self.pixel_y,
        )

        if verbose:
            print(
                "Direct space voxel sizes (z, y, x) based on initial FFT shape: (",
                str("{:.2f}".format(dz_realspace)),
                "nm,",
                str("{:.2f}".format(dy_realspace)),
                "nm,",
                str("{:.2f}".format(dx_realspace)),
                "nm )",
            )

        (
            nbz,
            nby,
            nbx,
        ) = obj.shape  # could be smaller if the object was cropped around the support
        if (
            nbz != initial_shape[0]
            or nby != initial_shape[1]
            or nbx != initial_shape[2]
        ):
            # recalculate the tilt and pixel sizes to accomodate a shape change
            tilt = self.tilt_angle * initial_shape[0] / nbz
            pixel_y = self.pixel_y * initial_shape[1] / nby
            pixel_x = self.pixel_x * initial_shape[2] / nbx
            if verbose:
                print(
                    "Tilt, pixel_y, pixel_x based on cropped array shape: (",
                    str("{:.4f}".format(tilt)),
                    "deg,",
                    str("{:.2f}".format(pixel_y * 1e6)),
                    "um,",
                    str("{:.2f}".format(pixel_x * 1e6)),
                    "um)",
                )

            # sanity check, the direct space voxel sizes calculated below should be equal to the original ones
            dz_realspace, dy_realspace, dx_realspace = self.voxel_sizes(
                (nbz, nby, nbx), tilt_angle=abs(tilt), pixel_x=pixel_x, pixel_y=pixel_y
            )
            if verbose:
                print(
                    "Sanity check, recalculated direct space voxel sizes: (",
                    str("{:.2f}".format(dz_realspace)),
                    " nm,",
                    str("{:.2f}".format(dy_realspace)),
                    "nm,",
                    str("{:.2f}".format(dx_realspace)),
                    "nm )",
                )
        else:
            tilt = self.tilt_angle
            pixel_y = self.pixel_y
            pixel_x = self.pixel_x

        if np.isnan(voxel_size):
            voxel = np.mean([dz_realspace, dy_realspace, dx_realspace])  # in nm
        else:
            voxel = voxel_size

        ortho_matrix = self.update_coords(
            array_shape=(nbz, nby, nbx),
            tilt_angle=tilt,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            verbose=verbose,
        )

        ###############################################################
        # Vincent Favre-Nicolin's method using inverse transformation #
        ###############################################################
        myz, myy, myx = np.meshgrid(
            np.arange(-nbz // 2, nbz // 2, 1) * voxel,
            np.arange(-nby // 2, nby // 2, 1) * voxel,
            np.arange(-nbx // 2, nbx // 2, 1) * voxel,
            indexing="ij",
        )

        # ortho_matrix is the transformation matrix from the detector coordinates to the laboratory frame
        # in RGI, we want to calculate the coordinates that would have a grid of the laboratory frame expressed in the
        # detector frame, i.e. one has to inverse the transformation matrix.
        ortho_imatrix = np.linalg.inv(ortho_matrix)
        new_x = (
            ortho_imatrix[0, 0] * myx
            + ortho_imatrix[0, 1] * myy
            + ortho_imatrix[0, 2] * myz
        )
        new_y = (
            ortho_imatrix[1, 0] * myx
            + ortho_imatrix[1, 1] * myy
            + ortho_imatrix[1, 2] * myz
        )
        new_z = (
            ortho_imatrix[2, 0] * myx
            + ortho_imatrix[2, 1] * myy
            + ortho_imatrix[2, 2] * myz
        )
        del myx, myy, myz
        gc.collect()

        rgi = RegularGridInterpolator(
            (
                np.arange(-nbz // 2, nbz // 2),
                np.arange(-nby // 2, nby // 2),
                np.arange(-nbx // 2, nbx // 2),
            ),
            obj,
            method="linear",
            bounds_error=False,
            fill_value=0,
        )
        ortho_obj = rgi(
            np.concatenate(
                (
                    new_z.reshape((1, new_z.size)),
                    new_y.reshape((1, new_z.size)),
                    new_x.reshape((1, new_z.size)),
                )
            ).transpose()
        )
        ortho_obj = ortho_obj.reshape((nbz, nby, nbx)).astype(obj.dtype)

        if debugging:
            gu.multislices_plot(
                abs(ortho_obj),
                sum_frames=True,
                width_z=width_z,
                width_y=width_y,
                width_x=width_x,
                title=title + " in the orthogonal laboratory frame",
            )
        return ortho_obj, voxel

    def orthogonalize_vector(
        self, vector, array_shape, tilt_angle, pixel_x, pixel_y, verbose=False
    ):
        """
        Calculate the direct space voxel sizes in the laboratory frame (z downstream, y vertical up, x outboard).

        :param vector: tuple of 3 coordinates, vector to be transformed in the detector frame
        :param array_shape: shape of the 3D array to orthogonalize
        :param tilt_angle: angular step during the rocking curve, in degrees
        :param pixel_x: horizontal pixel size, in meters
        :param pixel_y: vertical pixel size, in meters
        :param verbose: True to have printed comments
        :return: the direct space voxel sizes in nm, in the laboratory frame (voxel_z, voxel_y, voxel_x)
        """
        ortho_matrix = self.update_coords(
            array_shape=array_shape,
            tilt_angle=tilt_angle,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            verbose=verbose,
        )
        # ortho_matrix is the transformation matrix from the detector coordinates to the laboratory frame
        # Here, we want to calculate the coordinates that would have a vector of the laboratory frame expressed in the
        # detector frame, i.e. one has to inverse the transformation matrix.
        ortho_imatrix = np.linalg.inv(ortho_matrix)
        new_x = (
            ortho_imatrix[0, 0] * vector[2]
            + ortho_imatrix[0, 1] * vector[1]
            + ortho_imatrix[0, 2] * vector[0]
        )
        new_y = (
            ortho_imatrix[1, 0] * vector[2]
            + ortho_imatrix[1, 1] * vector[1]
            + ortho_imatrix[1, 2] * vector[0]
        )
        new_z = (
            ortho_imatrix[2, 0] * vector[2]
            + ortho_imatrix[2, 1] * vector[1]
            + ortho_imatrix[2, 2] * vector[0]
        )
        return new_z, new_y, new_x

    def outofplane_coeff(self):
        """
        Define a coefficient +/- 1 depending on the detector out of plane rotation direction and the detector out of
         plane orientation. The frame convention is the one of xrayutilities: x downstream, y outboard, z vertical up.
         See postprocessing/scripts/correct_angles_detector.py for an example.

        :return: +1 or -1
        """
        if self.detector_ver == "z+":  # origin of pixels at the bottom
            ver_coeff = 1
        else:  # 'z-'  origin of pixels at the top
            ver_coeff = -1

        # the out of plane detector rotation is clockwise for all beamlines
        coeff_outofplane = -1 * ver_coeff

        return coeff_outofplane

    def update_coords(self, array_shape, tilt_angle, pixel_x, pixel_y, verbose=True):
        """
        Calculate the pixel non-orthogonal coordinates in the orthogonal reference frame.

        :param array_shape: shape of the 3D array to orthogonalize
        :param tilt_angle: angular step during the rocking curve, in degrees
        :param pixel_x: horizontal pixel size, in meters
        :param pixel_y: vertical pixel size, in meters
        :param verbose: True to have printed comments
        :return: the transfer matrix from the detector frame to the laboratory frame
        """
        wavelength = self.wavelength * 1e9  # convert to nm
        distance = self.distance * 1e9  # convert to nm
        pixel_x = pixel_x * 1e9  # convert to nm
        pixel_y = pixel_y * 1e9  # convert to nm
        outofplane = np.radians(self.outofplane_angle)
        inplane = np.radians(self.inplane_angle)
        mygrazing_angle = np.radians(self.grazing_angle)
        lambdaz = wavelength * distance
        mymatrix = np.zeros((3, 3))
        tilt = np.radians(tilt_angle)

        nbz, nby, nbx = array_shape

        if self.beamline == "ID01":
            if verbose:
                print("using ESRF ID01 PSIC geometry")
            if self.rocking_angle == "outofplane":
                if verbose:
                    print("rocking angle is eta")
                # rocking eta angle clockwise around x (phi does not matter, above eta)
                mymatrix[:, 0] = (
                    2
                    * np.pi
                    * nbx
                    / lambdaz
                    * np.array(
                        [pixel_x * np.cos(inplane), 0, pixel_x * np.sin(inplane)]
                    )
                )
                mymatrix[:, 1] = (
                    2
                    * np.pi
                    * nby
                    / lambdaz
                    * np.array(
                        [
                            -pixel_y * np.sin(inplane) * np.sin(outofplane),
                            -pixel_y * np.cos(outofplane),
                            pixel_y * np.cos(inplane) * np.sin(outofplane),
                        ]
                    )
                )
                mymatrix[:, 2] = (
                    2
                    * np.pi
                    * nbz
                    / lambdaz
                    * np.array(
                        [
                            0,
                            tilt
                            * distance
                            * (1 - np.cos(inplane) * np.cos(outofplane)),
                            tilt * distance * np.sin(outofplane),
                        ]
                    )
                )
            elif self.rocking_angle == "inplane" and mygrazing_angle == 0:
                if verbose:
                    print("rocking angle is phi, eta=0")
                # rocking phi angle clockwise around y, assuming incident angle eta is zero (eta below phi)
                mymatrix[:, 0] = (
                    2
                    * np.pi
                    * nbx
                    / lambdaz
                    * np.array(
                        [pixel_x * np.cos(inplane), 0, pixel_x * np.sin(inplane)]
                    )
                )
                mymatrix[:, 1] = (
                    2
                    * np.pi
                    * nby
                    / lambdaz
                    * np.array(
                        [
                            -pixel_y * np.sin(inplane) * np.sin(outofplane),
                            -pixel_y * np.cos(outofplane),
                            pixel_y * np.cos(inplane) * np.sin(outofplane),
                        ]
                    )
                )
                mymatrix[:, 2] = (
                    2
                    * np.pi
                    * nbz
                    / lambdaz
                    * np.array(
                        [
                            -tilt
                            * distance
                            * (1 - np.cos(inplane) * np.cos(outofplane)),
                            0,
                            tilt * distance * np.sin(inplane) * np.cos(outofplane),
                        ]
                    )
                )
            elif self.rocking_angle == "inplane" and mygrazing_angle != 0:
                if verbose:
                    print("rocking angle is phi, with eta non zero")
                # rocking phi angle clockwise around y, incident angle eta is non zero (eta below phi)
                mymatrix[:, 0] = (
                    2
                    * np.pi
                    * nbx
                    / lambdaz
                    * np.array(
                        [pixel_x * np.cos(inplane), 0, pixel_x * np.sin(inplane)]
                    )
                )
                mymatrix[:, 1] = (
                    2
                    * np.pi
                    * nby
                    / lambdaz
                    * np.array(
                        [
                            -pixel_y * np.sin(inplane) * np.sin(outofplane),
                            -pixel_y * np.cos(outofplane),
                            pixel_y * np.cos(inplane) * np.sin(outofplane),
                        ]
                    )
                )
                mymatrix[:, 2] = (
                    2
                    * np.pi
                    * nbz
                    / lambdaz
                    * tilt
                    * distance
                    * np.array(
                        [
                            (
                                np.sin(mygrazing_angle) * np.sin(outofplane)
                                + np.cos(mygrazing_angle)
                                * (np.cos(inplane) * np.cos(outofplane) - 1)
                            ),
                            np.sin(mygrazing_angle)
                            * np.sin(inplane)
                            * np.cos(outofplane),
                            np.cos(mygrazing_angle)
                            * np.sin(inplane)
                            * np.cos(outofplane),
                        ]
                    )
                )
        if self.beamline == "P10":
            if verbose:
                print("using PETRAIII P10 geometry")
            if self.rocking_angle == "outofplane":
                if verbose:
                    print("rocking angle is omega")
                # rocking omega angle clockwise around x at mu=0 (phi does not matter, above eta)
                mymatrix[:, 0] = (
                    2
                    * np.pi
                    * nbx
                    / lambdaz
                    * np.array(
                        [-pixel_x * np.cos(inplane), 0, pixel_x * np.sin(inplane)]
                    )
                )
                mymatrix[:, 1] = (
                    2
                    * np.pi
                    * nby
                    / lambdaz
                    * np.array(
                        [
                            pixel_y * np.sin(inplane) * np.sin(outofplane),
                            -pixel_y * np.cos(outofplane),
                            pixel_y * np.cos(inplane) * np.sin(outofplane),
                        ]
                    )
                )
                mymatrix[:, 2] = (
                    2
                    * np.pi
                    * nbz
                    / lambdaz
                    * np.array(
                        [
                            0,
                            tilt
                            * distance
                            * (1 - np.cos(inplane) * np.cos(outofplane)),
                            tilt * distance * np.sin(outofplane),
                        ]
                    )
                )
            elif self.rocking_angle == "inplane" and mygrazing_angle == 0:
                if verbose:
                    print("rocking angle is phi, omega=0")
                # rocking phi angle clockwise around y, incident angle omega is zero (omega below phi)
                mymatrix[:, 0] = (
                    2
                    * np.pi
                    * nbx
                    / lambdaz
                    * np.array(
                        [-pixel_x * np.cos(inplane), 0, pixel_x * np.sin(inplane)]
                    )
                )
                mymatrix[:, 1] = (
                    2
                    * np.pi
                    * nby
                    / lambdaz
                    * np.array(
                        [
                            pixel_y * np.sin(inplane) * np.sin(outofplane),
                            -pixel_y * np.cos(outofplane),
                            pixel_y * np.cos(inplane) * np.sin(outofplane),
                        ]
                    )
                )
                mymatrix[:, 2] = (
                    2
                    * np.pi
                    * nbz
                    / lambdaz
                    * np.array(
                        [
                            tilt
                            * distance
                            * (np.cos(inplane) * np.cos(outofplane) - 1),
                            0,
                            -tilt * distance * np.sin(inplane) * np.cos(outofplane),
                        ]
                    )
                )

            elif self.rocking_angle == "inplane" and mygrazing_angle != 0:
                if verbose:
                    print("rocking angle is phi, with omega non zero")
                # rocking phi angle clockwise around y, incident angle omega is non zero (omega below phi)
                mymatrix[:, 0] = (
                    2
                    * np.pi
                    * nbx
                    / lambdaz
                    * np.array(
                        [-pixel_x * np.cos(inplane), 0, pixel_x * np.sin(inplane)]
                    )
                )
                mymatrix[:, 1] = (
                    2
                    * np.pi
                    * nby
                    / lambdaz
                    * np.array(
                        [
                            pixel_y * np.sin(inplane) * np.sin(outofplane),
                            -pixel_y * np.cos(outofplane),
                            pixel_y * np.cos(inplane) * np.sin(outofplane),
                        ]
                    )
                )
                mymatrix[:, 2] = (
                    2
                    * np.pi
                    * nbz
                    / lambdaz
                    * tilt
                    * distance
                    * np.array(
                        [
                            (
                                np.sin(mygrazing_angle) * np.sin(outofplane)
                                + np.cos(mygrazing_angle)
                                * (np.cos(inplane) * np.cos(outofplane) - 1)
                            ),
                            -np.sin(mygrazing_angle)
                            * np.sin(inplane)
                            * np.cos(outofplane),
                            -np.cos(mygrazing_angle)
                            * np.sin(inplane)
                            * np.cos(outofplane),
                        ]
                    )
                )

        if self.beamline == "NANOMAX":
            if verbose:
                print("using NANOMAX geometry")
            if self.rocking_angle == "outofplane":
                if verbose:
                    print("rocking angle is theta")
                # rocking eta angle clockwise around x (phi does not matter, above eta)
                mymatrix[:, 0] = (
                    2
                    * np.pi
                    * nbx
                    / lambdaz
                    * np.array(
                        [pixel_x * np.cos(inplane), 0, pixel_x * np.sin(inplane)]
                    )
                )
                mymatrix[:, 1] = (
                    2
                    * np.pi
                    * nby
                    / lambdaz
                    * np.array(
                        [
                            pixel_y * np.sin(inplane) * np.sin(outofplane),
                            pixel_y * np.cos(outofplane),
                            -pixel_y * np.cos(inplane) * np.sin(outofplane),
                        ]
                    )
                )
                mymatrix[:, 2] = (
                    2
                    * np.pi
                    * nbz
                    / lambdaz
                    * np.array(
                        [
                            0,
                            tilt
                            * distance
                            * (1 - np.cos(inplane) * np.cos(outofplane)),
                            tilt * distance * np.sin(outofplane),
                        ]
                    )
                )
            elif self.rocking_angle == "inplane" and mygrazing_angle == 0:
                if verbose:
                    print("rocking angle is phi, theta=0")
                # rocking phi angle clockwise around y, assuming incident angle eta is zero (eta below phi)
                mymatrix[:, 0] = (
                    2
                    * np.pi
                    * nbx
                    / lambdaz
                    * np.array(
                        [pixel_x * np.cos(inplane), 0, pixel_x * np.sin(inplane)]
                    )
                )
                mymatrix[:, 1] = (
                    2
                    * np.pi
                    * nby
                    / lambdaz
                    * np.array(
                        [
                            pixel_y * np.sin(inplane) * np.sin(outofplane),
                            pixel_y * np.cos(outofplane),
                            -pixel_y * np.cos(inplane) * np.sin(outofplane),
                        ]
                    )
                )
                mymatrix[:, 2] = (
                    2
                    * np.pi
                    * nbz
                    / lambdaz
                    * np.array(
                        [
                            -tilt
                            * distance
                            * (1 - np.cos(inplane) * np.cos(outofplane)),
                            0,
                            tilt * distance * np.sin(inplane) * np.cos(outofplane),
                        ]
                    )
                )
            elif self.rocking_angle == "inplane" and mygrazing_angle != 0:
                if verbose:
                    print("rocking angle is phi, with theta non zero")
                # rocking phi angle clockwise around y, incident angle eta is non zero (eta below phi)
                mymatrix[:, 0] = (
                    2
                    * np.pi
                    * nbx
                    / lambdaz
                    * np.array(
                        [pixel_x * np.cos(inplane), 0, pixel_x * np.sin(inplane)]
                    )
                )
                mymatrix[:, 1] = (
                    2
                    * np.pi
                    * nby
                    / lambdaz
                    * np.array(
                        [
                            pixel_y * np.sin(inplane) * np.sin(outofplane),
                            pixel_y * np.cos(outofplane),
                            -pixel_y * np.cos(inplane) * np.sin(outofplane),
                        ]
                    )
                )
                mymatrix[:, 2] = (
                    2
                    * np.pi
                    * nbz
                    / lambdaz
                    * tilt
                    * distance
                    * np.array(
                        [
                            (
                                np.sin(mygrazing_angle) * np.sin(outofplane)
                                + np.cos(mygrazing_angle)
                                * (np.cos(inplane) * np.cos(outofplane) - 1)
                            ),
                            np.sin(mygrazing_angle)
                            * np.sin(inplane)
                            * np.cos(outofplane),
                            np.cos(mygrazing_angle)
                            * np.sin(inplane)
                            * np.cos(outofplane),
                        ]
                    )
                )

        if self.beamline == "34ID":
            if verbose:
                print("using APS 34ID geometry")
            if self.rocking_angle == "outofplane":
                if verbose:
                    print("rocking angle is phi")
                # rocking phi angle anti-clockwise around x (theta does not matter, above phi)
                mymatrix[:, 0] = (
                    2
                    * np.pi
                    * nbx
                    / lambdaz
                    * np.array(
                        [pixel_x * np.cos(inplane), 0, -pixel_x * np.sin(inplane)]
                    )
                )
                mymatrix[:, 1] = (
                    2
                    * np.pi
                    * nby
                    / lambdaz
                    * np.array(
                        [
                            pixel_y * np.sin(inplane) * np.sin(outofplane),
                            -pixel_y * np.cos(outofplane),
                            pixel_y * np.cos(inplane) * np.sin(outofplane),
                        ]
                    )
                )
                mymatrix[:, 2] = (
                    2
                    * np.pi
                    * nbz
                    / lambdaz
                    * np.array(
                        [
                            0,
                            -tilt
                            * distance
                            * (1 - np.cos(inplane) * np.cos(outofplane)),
                            -tilt * distance * np.sin(outofplane),
                        ]
                    )
                )

            elif self.rocking_angle == "inplane" and mygrazing_angle != 0:
                if verbose:
                    print("rocking angle is theta, with phi non zero")
                # rocking theta angle anti-clockwise around y, incident angle is non zero (theta is above phi)
                mymatrix[:, 0] = (
                    2
                    * np.pi
                    * nbx
                    / lambdaz
                    * np.array(
                        [pixel_x * np.cos(inplane), 0, -pixel_x * np.sin(inplane)]
                    )
                )
                mymatrix[:, 1] = (
                    2
                    * np.pi
                    * nby
                    / lambdaz
                    * np.array(
                        [
                            pixel_y * np.sin(inplane) * np.sin(outofplane),
                            -pixel_y * np.cos(outofplane),
                            pixel_y * np.cos(inplane) * np.sin(outofplane),
                        ]
                    )
                )
                mymatrix[:, 2] = (
                    2
                    * np.pi
                    * nbz
                    / lambdaz
                    * tilt
                    * distance
                    * np.array(
                        [
                            (
                                np.sin(mygrazing_angle) * np.sin(outofplane)
                                + np.cos(mygrazing_angle)
                                * (1 - np.cos(inplane) * np.cos(outofplane))
                            ),
                            -np.sin(mygrazing_angle)
                            * np.sin(inplane)
                            * np.cos(outofplane),
                            np.cos(mygrazing_angle)
                            * np.sin(inplane)
                            * np.cos(outofplane),
                        ]
                    )
                )

            elif self.rocking_angle == "inplane" and mygrazing_angle == 0:
                if verbose:
                    print("rocking angle is theta, phi=0")
                # rocking theta angle anti-clockwise around y, assuming incident angle is zero (theta is above phi)
                mymatrix[:, 0] = (
                    2
                    * np.pi
                    * nbx
                    / lambdaz
                    * np.array(
                        [pixel_x * np.cos(inplane), 0, -pixel_x * np.sin(inplane)]
                    )
                )
                mymatrix[:, 1] = (
                    2
                    * np.pi
                    * nby
                    / lambdaz
                    * np.array(
                        [
                            pixel_y * np.sin(inplane) * np.sin(outofplane),
                            -pixel_y * np.cos(outofplane),
                            pixel_y * np.cos(inplane) * np.sin(outofplane),
                        ]
                    )
                )
                mymatrix[:, 2] = (
                    2
                    * np.pi
                    * nbz
                    / lambdaz
                    * np.array(
                        [
                            tilt
                            * distance
                            * (1 - np.cos(inplane) * np.cos(outofplane)),
                            0,
                            tilt * distance * np.sin(inplane) * np.cos(outofplane),
                        ]
                    )
                )
        if self.beamline in {"SIXS_2018", "SIXS_2019"}:
            if verbose:
                print("using SIXS geometry")
            if self.rocking_angle == "inplane" and mygrazing_angle != 0:
                if verbose:
                    print("rocking angle is mu, with beta non zero")
                # rocking mu angle anti-clockwise around y
                mymatrix[:, 0] = (
                    2
                    * np.pi
                    * nbx
                    / lambdaz
                    * pixel_x
                    * np.array(
                        [
                            np.cos(inplane),
                            -np.sin(mygrazing_angle) * np.sin(inplane),
                            -np.cos(mygrazing_angle) * np.sin(inplane),
                        ]
                    )
                )
                mymatrix[:, 1] = (
                    2
                    * np.pi
                    * nby
                    / lambdaz
                    * pixel_y
                    * np.array(
                        [
                            np.sin(inplane) * np.sin(outofplane),
                            (
                                np.sin(mygrazing_angle)
                                * np.cos(inplane)
                                * np.sin(outofplane)
                                - np.cos(mygrazing_angle) * np.cos(outofplane)
                            ),
                            (
                                np.cos(mygrazing_angle)
                                * np.cos(inplane)
                                * np.sin(outofplane)
                                + np.sin(mygrazing_angle) * np.cos(outofplane)
                            ),
                        ]
                    )
                )
                mymatrix[:, 2] = (
                    2
                    * np.pi
                    * nbz
                    / lambdaz
                    * tilt
                    * distance
                    * np.array(
                        [
                            np.cos(mygrazing_angle)
                            - np.cos(inplane) * np.cos(outofplane),
                            np.sin(mygrazing_angle)
                            * np.sin(inplane)
                            * np.cos(outofplane),
                            np.cos(mygrazing_angle)
                            * np.sin(inplane)
                            * np.cos(outofplane),
                        ]
                    )
                )

            elif self.rocking_angle == "inplane" and mygrazing_angle == 0:
                if verbose:
                    print("rocking angle is mu, beta=0")
                # rocking th angle anti-clockwise around y, assuming incident angle is zero (th above tilt)
                mymatrix[:, 0] = (
                    2
                    * np.pi
                    * nbx
                    / lambdaz
                    * np.array(
                        [pixel_x * np.cos(inplane), 0, -pixel_x * np.sin(inplane)]
                    )
                )
                mymatrix[:, 1] = (
                    2
                    * np.pi
                    * nby
                    / lambdaz
                    * np.array(
                        [
                            pixel_y * np.sin(inplane) * np.sin(outofplane),
                            -pixel_y * np.cos(outofplane),
                            pixel_y * np.cos(inplane) * np.sin(outofplane),
                        ]
                    )
                )
                mymatrix[:, 2] = (
                    2
                    * np.pi
                    * nbz
                    / lambdaz
                    * np.array(
                        [
                            tilt
                            * distance
                            * (1 - np.cos(inplane) * np.cos(outofplane)),
                            0,
                            tilt * distance * np.sin(inplane) * np.cos(outofplane),
                        ]
                    )
                )
        if self.beamline == "CRISTAL":
            if verbose:
                print("using CRISTAL geometry")
            if self.rocking_angle == "outofplane":
                if verbose:
                    print("rocking angle is komega")
                # rocking tilt angle clockwise around x
                mymatrix[:, 0] = (
                    2
                    * np.pi
                    * nbx
                    / lambdaz
                    * np.array(
                        [pixel_x * np.cos(inplane), 0, -pixel_x * np.sin(inplane)]
                    )
                )
                mymatrix[:, 1] = (
                    2
                    * np.pi
                    * nby
                    / lambdaz
                    * np.array(
                        [
                            pixel_y * np.sin(inplane) * np.sin(outofplane),
                            -pixel_y * np.cos(outofplane),
                            pixel_y * np.cos(inplane) * np.sin(outofplane),
                        ]
                    )
                )
                mymatrix[:, 2] = (
                    2
                    * np.pi
                    * nbz
                    / lambdaz
                    * np.array(
                        [
                            0,
                            tilt
                            * distance
                            * (1 - np.cos(inplane) * np.cos(outofplane)),
                            tilt * distance * np.sin(outofplane),
                        ]
                    )
                )

        transfer_matrix = 2 * np.pi * np.linalg.inv(mymatrix).transpose()
        return transfer_matrix

    def voxel_sizes(self, array_shape, tilt_angle, pixel_x, pixel_y, verbose=False):
        """
        Calculate the direct space voxel sizes in the laboratory frame (z downstream, y vertical up, x outboard).

        :param array_shape: shape of the 3D array to orthogonalize
        :param tilt_angle: angular step during the rocking curve, in degrees
        :param pixel_x: horizontal pixel size, in meters
        :param pixel_y: vertical pixel size, in meters
        :param verbose: True to have printed comments
        :return: the direct space voxel sizes in nm, in the laboratory frame (voxel_z, voxel_y, voxel_x)
        """
        transfer_matrix = self.update_coords(
            array_shape=array_shape,
            tilt_angle=tilt_angle,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            verbose=verbose,
        )
        rec_matrix = 2 * np.pi * np.linalg.inv(transfer_matrix).transpose()
        qx_range = np.linalg.norm(rec_matrix[0, :])
        qy_range = np.linalg.norm(rec_matrix[1, :])
        qz_range = np.linalg.norm(rec_matrix[2, :])
        if verbose:
            print(
                "q_range_z, q_range_y, q_range_x=({0:.5f}, {1:.5f}, {2:.5f}) (1/nm)".format(
                    qz_range, qy_range, qx_range
                )
            )
            print(
                "voxelsize_z, voxelsize_y, voxelsize_x="
                "({0:.2f}, {1:.2f}, {2:.2f}) (1/nm)".format(
                    2 * np.pi / qz_range, 2 * np.pi / qy_range, 2 * np.pi / qx_range
                )
            )
        return 2 * np.pi / qz_range, 2 * np.pi / qy_range, 2 * np.pi / qx_range

    def voxel_sizes_detector(
        self, array_shape, tilt_angle, pixel_x, pixel_y, verbose=False
    ):
        """
        Calculate the direct space voxel sizes in the detector frame
         (z rocking angle, y detector vertical axis, x detector horizontal axis).

        :param array_shape: shape of the 3D array used in phase retrieval
        :param tilt_angle: angular step during the rocking curve, in degrees
        :param pixel_x: horizontal pixel size, in meters
        :param pixel_y: vertical pixel size, in meters
        :param verbose: True to have printed comments
        :return: the direct space voxel sizes in nm, in the detector frame (voxel_z, voxel_y, voxel_x)
        """
        voxel_z = (
            self.wavelength / (array_shape[0] * abs(tilt_angle) * np.pi / 180) * 1e9
        )  # in nm
        voxel_y = (
            self.wavelength * self.distance / (array_shape[1] * pixel_y) * 1e9
        )  # in nm
        voxel_x = (
            self.wavelength * self.distance / (array_shape[2] * pixel_x) * 1e9
        )  # in nm
        if verbose:
            print(
                "voxelsize_z, voxelsize_y, voxelsize_x="
                "({0:.2f}, {1:.2f}, {2:.2f}) (1/nm)".format(voxel_z, voxel_y, voxel_x)
            )
        return voxel_z, voxel_y, voxel_x
