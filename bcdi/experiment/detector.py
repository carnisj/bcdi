# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

"""
Implementation of the detector classes.

These classes handle the detector-dependent paths and data filenames, on top of the
detector-dependent properties (e.g., number and size of the pixels, gaps between
tiles...). Generic methods are implemented in the abstract base class Detector, and
detector-dependent properties need to be implemented in each child class (they are
decoracted by @abstractmethod in the base class, they are written in italic in the
following diagram).

.. mermaid::
  :align: center

  classDiagram
    class Detector{
      <<abstract>>
      +str name
      +tuple binning
      +str datadir
      +int nb_pixel_x
      +int nb_pixel_y
      +float pixelsize_x
      +float pixelsize_y
      +dict params
      +tuple preprocessing_binning
      +tuple roi
      +str rootdir
      +str sample_name
      +str savedir
      +str scandir
      +tuple sum_roi
      +str template_file
      +str template_imagefile
      unbinned_pixel_number()*
      unbinned_pixel_size()*
      _background_subtraction()
      _flatfield_correction()
      _hotpixels_correction()
      _linearity_correction()
      _mask_gaps()
      _saturation_correction()
      counter()
      linearity_func()
      mask_detector()
  }
    ABC <|-- Detector

API Reference
-------------

"""

import logging
import os
import pathlib
from abc import ABC, abstractmethod
from functools import partial
from numbers import Integral, Real

import numpy as np

import bcdi.utils.format as fmt
from bcdi.utils import validation as valid

module_logger = logging.getLogger(__name__)


def create_detector(name, **kwargs):
    """
    Create a Detector instance depending on the detector.

    :param name: str, name of the detector
    :return:  the corresponding detector instance
    """
    if name == "Maxipix":
        return Maxipix(name=name, **kwargs)
    if name == "Eiger2M":
        return Eiger2M(name=name, **kwargs)
    if name == "Eiger4M":
        return Eiger4M(name=name, **kwargs)
    if name == "Eiger9M":
        return Eiger9M(name=name, **kwargs)
    if name == "Lambda":
        return Lambda(name=name, **kwargs)
    if name == "Timepix":
        return Timepix(name=name, **kwargs)
    if name == "Merlin":
        return Merlin(name=name, **kwargs)
    if name == "MerlinSixS":
        return MerlinSixS(name=name, **kwargs)
    if name == "Dummy":
        return Dummy(name=name, **kwargs)
    raise NotImplementedError(f"No implementation for the {name} detector")


class Detector(ABC):
    """
    Class to handle the configuration of the detector used for data acquisition.

    :param name: name of the detector in {'Maxipix', 'Timepix', 'Merlin',
     'MerlinSixS', 'Eiger2M', 'Eiger4M', 'Dummy'}
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

    :param specfile: template for the log file or the data file depending on the
     beamline
    :param roi: region of interest of the detector used for analysis
    :param sum_roi: region of interest of the detector used for calculated an
     integrated intensity
    :param binning: binning factor of the 3D dataset
     (stacking dimension, detector vertical axis, detector horizontal axis)
    :param preprocessing_binning: tuple of the three binning factors used in a previous
       preprocessing step
    :param offsets: tuple or list, sample and detector offsets corresponding to the
       parameter delta in xrayutilities hxrd.Ang2Q.area method
    :param linearity_func: function to apply to each pixel of the detector in order to
       compensate the deviation of the detector linearity for large intensities.
    :param kwargs:
     - 'logger': an optional logger

    """

    def __init__(
        self,
        name,
        rootdir=None,
        datadir=None,
        savedir=None,
        template_imagefile=None,
        specfile=None,
        sample_name=None,
        roi=None,
        sum_roi=None,
        binning=(1, 1, 1),
        preprocessing_binning=(1, 1, 1),
        offsets=None,
        linearity_func=None,
        **kwargs,
    ):
        # the detector name should be initialized first,
        # other properties depend on it
        self._name = name

        # load the kwargs
        self.logger = kwargs.get("logger", module_logger)
        self.preprocessing_binning = preprocessing_binning
        self.offsets = offsets  # delegate the test to xrayutilities
        self.linearity_func = linearity_func

        # load other positional arguments
        self.binning = binning
        self.roi = roi
        self.sum_roi = sum_roi
        # parameters related to data path
        self.rootdir = rootdir
        self.datadir = datadir
        self.savedir = savedir
        self.sample_name = sample_name
        self.template_imagefile = template_imagefile
        self.specfile = specfile

        # dictionary of keys: beamline_name and values: counter name for the image
        # number in the log file.
        self._counter_table = {}

        # initialize the threshold for saturation, can be overriden in child classes
        self.saturation_threshold = None

        # property used to track the binning factor throughout data processing
        # the starting point is preprocessing_binning, which is the state of the data
        # when loaded
        self.current_binning = list(self.preprocessing_binning)

    @property
    def binning(self):
        """
        Binning factor of the dataset.

        Tuple of three positive integers corresponding to the binning of the data used
        in phase retrieval (stacking dimension, detector vertical axis, detector
        horizontal axis). To declare an additional binning factor due to a previous
        preprocessing step, use the kwarg 'preprocessing_binning' instead.
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

    def counter(self, beamline):
        """
        Name of the counter in the log file for the image number.

        :param beamline: str, name of the beamline
        """
        if not isinstance(beamline, str):
            raise TypeError("beamline should be a string")
        return self._counter_table.get(beamline)

    @property
    def current_binning(self):
        """
        Display the current binning factor of the dataset.

        Tuple of three positive integers corresponding to the current binning of the
        data in the processing pipeline.
        """
        return self._current_binning

    @current_binning.setter
    def current_binning(self, value):
        valid.valid_container(
            value,
            container_types=list,
            length=3,
            item_types=int,
            min_excluded=0,
            name="Detector.current_binning",
        )
        self._current_binning = value

    @property
    def datadir(self):
        """Name of the data directory."""
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
        if value is not None and not os.path.isdir(value):
            raise ValueError(f"The directory '{value}' does not exist")
        self._datadir = value

    @property
    def linearity_func(self):
        """Correction of the non-linearity of the detector with high incoming flux."""
        return self._linearity_func

    @linearity_func.setter
    def linearity_func(self, value):
        def poly4(array_1d, coeffs):
            """
            Define a 4th order polynomial and apply it on a 1D array.

            :param array_1d: a numpy 1D array
            :param coeffs: sequence of 5 Real numbers, the coefficients [a, b, c, d, e]
             of the polynomial ax^4 + bx^3 + cx^2 + dx + e
            :return: the updated 1D array
            """
            return (
                coeffs[0] * array_1d**4
                + coeffs[1] * array_1d**3
                + coeffs[2] * array_1d**2
                + coeffs[3] * array_1d
                + coeffs[4]
            )

        if value is None:
            self._linearity_func = None
            return

        valid.valid_container(
            value,
            container_types=(tuple, list, np.ndarray),
            length=5,
            item_types=Real,
            name="linearity_func",
        )
        self._linearity_func = partial(poly4, coeffs=value)

    @property
    def name(self):
        """Name of the detector."""
        return self._name

    @property
    def nb_pixel_x(self):
        """
        Horizontal number of pixels of the detector.

        It takes into account an eventual preprocessing binning (useful when
        reloading a already preprocessed file).
        """
        return self.unbinned_pixel_number[1] // self.preprocessing_binning[2]

    @property
    def nb_pixel_y(self):
        """
        Vertical number of pixels of the detector.

        It takes into account an eventual preprocessing binning (useful when
        reloading a already preprocessed file).
        """
        return self.unbinned_pixel_number[0] // self.preprocessing_binning[1]

    @property
    def params(self):
        """Return a dictionnary with all parameters."""
        return {
            "Class": self.__class__.__name__,
            "name": self.name,
            "unbinned_pixel_size_m": self.unbinned_pixel_size,
            "nb_pixel_x": self.nb_pixel_x,
            "nb_pixel_y": self.nb_pixel_y,
            "binning": self.binning,
            "roi": self.roi,
            "sum_roi": self.sum_roi,
            "preprocessing_binning": self.preprocessing_binning,
            "rootdir": self.rootdir,
            "datadir": self.datadir,
            "scandir": self.scandir,
            "savedir": self.savedir,
            "sample_name": self.sample_name,
            "template_imagefile": self.template_imagefile,
            "specfile": self.specfile,
        }

    @property
    def pixelsize_x(self):
        """Horizontal pixel size of the detector after taking into account binning."""
        return (
            self.unbinned_pixel_size[1]
            * self.preprocessing_binning[2]
            * self.binning[2]
        )

    @property
    def pixelsize_y(self):
        """Vertical pixel size of the detector after taking into account binning."""
        return (
            self.unbinned_pixel_size[0]
            * self.preprocessing_binning[1]
            * self.binning[1]
        )

    @property
    def preprocessing_binning(self):
        """
        Preprocessing binning factor of the data.

        Tuple of three positive integers corresponding to the binning factor of the
        data used in a previous preprocessing step (stacking dimension, detector
        vertical axis, detector horizontal axis).
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
        Region of interest of the detector to be used.

        Convention: [y_start, y_stop, x_start, x_stop]
        """
        return self._roi

    @roi.setter
    def roi(self, value):
        if not value:  # None or empty list/tuple
            value = [0, self.unbinned_pixel_number[0], 0, self.unbinned_pixel_number[1]]
        valid.valid_container(
            value,
            container_types=(tuple, list, np.ndarray),
            length=4,
            item_types=int,
            name="Detector.roi",
        )
        if value[1] <= value[0] or value[3] <= value[2]:
            raise ValueError("roi coordinates should be increasing in x and y")
        self._roi = value

    @property
    def rootdir(self):
        """Name of the root directory, which englobes all scans."""
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
        if value is not None and not os.path.isdir(value):
            raise ValueError(f"The directory {value} does not exist")
        self._rootdir = value

    @property
    def sample_name(self):
        """Name of the sample."""
        return self._sample_name

    @sample_name.setter
    def sample_name(self, value):
        valid.valid_container(
            value,
            container_types=str,
            allow_none=True,
            name="Detector.sample_name",
        )
        self._sample_name = value

    @property
    def savedir(self):
        """Name of the saving directory."""
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
        if value is not None:
            pathlib.Path(value).mkdir(parents=True, exist_ok=True)
        self._savedir = value

    @property
    def scandir(self):
        """Path of the scan, typically it is the parent folder of the data folder."""
        if self.datadir:
            dir_path = os.path.abspath(os.path.join(self.datadir, os.pardir)) + "/"
            return dir_path.replace("\\", "/")
        return None

    @property
    def sum_roi(self):
        """
        Region of interest of the detector used for integrating the intensity.

        Convention: [y_start, y_stop, x_start, x_stop]
        """
        return self._sum_roi

    @sum_roi.setter
    def sum_roi(self, value):
        if not value:  # None or empty list/tuple
            value = self.roi
        valid.valid_container(
            value,
            container_types=(tuple, list),
            length=4,
            item_types=int,
            name="Detector.sum_roi",
        )
        if value[1] <= value[0] or value[3] <= value[2]:
            raise ValueError("roi coordinates should be increasing in x and y")
        self._sum_roi = value

    @property
    def template_imagefile(self):
        """Name of the data file."""
        return self._template_imagefile

    @template_imagefile.setter
    def template_imagefile(self, value):
        valid.valid_container(
            value,
            container_types=str,
            min_length=0,
            allow_none=True,
            name="template_imagefile",
        )
        self._template_imagefile = value

    @property
    @abstractmethod
    def unbinned_pixel_number(self):
        """
        Define the number of pixels of the unbinned detector.

        Convention: (vertical, horizontal)
        """
        return 1, 1

    @property
    @abstractmethod
    def unbinned_pixel_size(self):
        """Pixel size (vertical, horizontal) of the unbinned detector in meters."""
        return 1, 1

    def __repr__(self):
        """Representation string of the Detector instance."""
        return fmt.create_repr(self, Detector)

    @staticmethod
    def _background_subtraction(data, background):
        """
        Apply background subtraction to the data.

        :param data: a 2D numpy ndarray
        :param background: None or a 2D numpy array
        :return: the corrected data array
        """
        if background is not None:
            valid.valid_ndarray((data, background), ndim=2)
            return data - background
        return data

    @staticmethod
    def _flatfield_correction(data, flatfield):
        """
        Apply flatfield correction to the data (multiplication).

        :param data: a 2D numpy ndarray
        :param flatfield: None or a 2D numpy array
        :return: the corrected data array
        """
        if flatfield is not None:
            valid.valid_ndarray((data, flatfield), ndim=2)
            return np.multiply(flatfield, data)
        return data

    @staticmethod
    def _hotpixels_correction(data, mask, hotpixels):
        """
        Apply hotpixels correction to the data and update the mask.

        :param data: a 2D numpy ndarray
        :param hotpixels: None or a 2D numpy array, 1 if the pixel needs to be masked,
         0 otherwise
        :return: the corrected data array
        """
        if hotpixels is not None:
            valid.valid_ndarray((data, mask, hotpixels), ndim=2)
            if ((hotpixels == 0).sum() + (hotpixels == 1).sum()) != hotpixels.size:
                raise ValueError("hotpixels should be an array of 0 and 1")

            data[hotpixels == 1] = 0
            mask[hotpixels == 1] = 1

        return data, mask

    def _linearity_correction(self, data):
        """
        Apply a correction to data if the detector response is not linear.

        :param data: a 2D numpy array
        :return: the corrected data array
        """
        if self.linearity_func is not None and callable(self.linearity_func):
            valid.valid_ndarray(data, ndim=2)
            data = data.astype(float)
            nby, nbx = data.shape
            return self.linearity_func(array_1d=data.flatten()).reshape((nby, nbx))
        return data

    def mask_detector(
        self, data, mask, nb_frames=1, flatfield=None, background=None, hotpixels=None
    ):
        """
        Mask data measured with a 2D detector.

        It can apply flatfield correction, background subtraction, masking of hotpixels
        and detector gaps.

        :param data: the 2D data to mask
        :param mask: the 2D mask to be updated
        :param nb_frames: number of frames summed to yield the 2D data
         (e.g. in a series measurement), used when defining the threshold for hot pixels
        :param flatfield: the 2D flatfield array to be multiplied with the data
        :param background: a 2D array to be subtracted to the data
        :param hotpixels: a 2D array with hotpixels to be masked
         (1=hotpixel, 0=normal pixel)
        :return: the masked data and the updated mask
        """
        valid.valid_ndarray((data, mask), ndim=2)

        # linearity correction, returns the data itself if linearity_func is None
        data = self._linearity_correction(data)

        # flatfield correction
        data = self._flatfield_correction(data=data, flatfield=flatfield)

        # remove the background
        data = self._background_subtraction(data=data, background=background)

        # mask hotpixels
        data, mask = self._hotpixels_correction(
            data=data, mask=mask, hotpixels=hotpixels
        )
        # mask detector gaps
        data, mask = self._mask_gaps(data, mask)

        # remove saturated pixels
        data, mask = self._saturation_correction(data, mask, nb_frames=nb_frames)

        return data, mask

    def _mask_gaps(self, data, mask):
        """
        Mask the gaps between sensors in the detector.

        :param data: a 2D numpy array
        :param mask: a 2D numpy array of the same shape as data
        :return:

         - the masked data
         - the updated mask

        """
        valid.valid_ndarray(
            (data, mask), ndim=2, shape=self.unbinned_pixel_number, fix_shape=True
        )
        return data, mask

    def _saturation_correction(self, data, mask, nb_frames):
        """
        Mask pixels above a certain threshold.

        This is detector dependent. If a 2D frames was obtained by summing a series of
        frames (e.g. series measurement at P10), the threshold is multiplied
        accordingly.

        :param data: a 2D numpy array
        :param mask: a 2D numpy array of the same shape as data
        :param nb_frames: int, number of frames concatenated to obtain the 2D data array
        :return:

         - the masked data
         - the updated mask

        """
        if self.saturation_threshold is not None:
            valid.valid_ndarray((data, mask), ndim=2)

            valid.valid_item(
                nb_frames, allowed_types=int, min_excluded=0, name="nb_frames"
            )
            mask[data > self.saturation_threshold * nb_frames] = 1
            data[data > self.saturation_threshold * nb_frames] = 0
        return data, mask


class Maxipix(Detector):
    """Implementation of the Maxipix detector."""

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)
        self._counter_table = {"ID01": "mpx4inr"}  # useful if the same type of detector
        # is used at several beamlines
        self.saturation_threshold = 1e6

    def _mask_gaps(self, data, mask):
        """
        Mask the gaps between sensors in the detector.

        :param data: a 2D numpy array
        :param mask: a 2D numpy array of the same shape as data
        :return:

         - the masked data
         - the updated mask

        """
        valid.valid_ndarray(
            (data, mask), ndim=2, shape=self.unbinned_pixel_number, fix_shape=True
        )

        data[:, 255:261] = 0
        data[255:261, :] = 0

        mask[:, 255:261] = 1
        mask[255:261, :] = 1
        return data, mask

    @property
    def unbinned_pixel_number(self):
        """
        Define the number of pixels of the unbinned detector.

        Convention: (vertical, horizontal)
        """
        return 516, 516

    @property
    def unbinned_pixel_size(self):
        """Pixel size (vertical, horizontal) of the unbinned detector in meters."""
        return 55e-06, 55e-06


class Eiger2M(Detector):
    """Implementation of the Eiger2M detector."""

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)
        self._counter_table = {"ID01": "ei2minr"}  # useful if the same type of detector
        # is used at several beamlines
        self.saturation_threshold = 1e7

    def _mask_gaps(self, data, mask):
        """
        Mask the gaps between sensors in the detector.

        :param data: a 2D numpy array
        :param mask: a 2D numpy array of the same shape as data
        :return:

         - the masked data
         - the updated mask

        """
        valid.valid_ndarray(
            (data, mask), ndim=2, shape=self.unbinned_pixel_number, fix_shape=True
        )

        data[:, 255:259] = 0
        data[:, 513:517] = 0
        data[:, 771:775] = 0
        data[0:257, 72:80] = 0
        data[255:259, :] = 0
        data[511:552, :] = 0
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
        return data, mask

    @property
    def unbinned_pixel_number(self):
        """
        Define the number of pixels of the unbinned detector.

        Convention: (vertical, horizontal)
        """
        return 2164, 1030

    @property
    def unbinned_pixel_size(self):
        """Pixel size (vertical, horizontal) of the unbinned detector in meters."""
        return 75e-06, 75e-06


class Eiger4M(Detector):
    """Implementation of the Eiger4M detector."""

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)
        self.saturation_threshold = 4000000000

    def _mask_gaps(self, data, mask):
        """
        Mask the gaps between sensors in the detector.

        :param data: a 2D numpy array
        :param mask: a 2D numpy array of the same shape as data
        :return:

         - the masked data
         - the updated mask

        """
        valid.valid_ndarray(
            (data, mask), ndim=2, shape=self.unbinned_pixel_number, fix_shape=True
        )

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
        return data, mask

    @property
    def unbinned_pixel_number(self):
        """
        Define the number of pixels of the unbinned detector.

        Convention: (vertical, horizontal)
        """
        return 2167, 2070

    @property
    def unbinned_pixel_size(self):
        """Pixel size (vertical, horizontal) of the unbinned detector in meters."""
        return 75e-06, 75e-06


class Eiger9M(Detector):
    """Implementation of the Eiger9M detector."""

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)
        self.saturation_threshold = 4000000000

    def _mask_gaps(self, data, mask):
        """
        Mask the gaps between sensors in the detector.

        :param data: a 2D numpy array
        :param mask: a 2D numpy array of the same shape as data
        :return:

         - the masked data
         - the updated mask

        """
        valid.valid_ndarray(
            (data, mask), ndim=2, shape=self.unbinned_pixel_number, fix_shape=True
        )
        data[:, 0:1] = 0
        data[:, -1:] = 0
        data[0:1, :] = 0
        data[-1:, :] = 0
        data[:, 513:515] = 0
        data[:, 1028:1040] = 0
        data[:, 1553:1555] = 0
        data[:, 2068:2080] = 0
        data[:, 2593:2595] = 0
        data[512:550, :] = 0
        data[1062:1100, :] = 0
        data[1612:1650, :] = 0
        data[2162:2200, :] = 0
        data[2712:2750, :] = 0

        mask[:, 0:1] = 1
        mask[:, -1:] = 1
        mask[0:1, :] = 1
        mask[-1:, :] = 1
        mask[:, 513:515] = 1
        mask[:, 1028:1040] = 1
        mask[:, 1553:1555] = 1
        mask[:, 2068:2080] = 1
        mask[:, 2593:2595] = 1
        mask[512:550, :] = 1
        mask[1062:1100, :] = 1
        mask[1612:1650, :] = 1
        mask[2162:2200, :] = 1
        mask[2712:2750, :] = 1
        return data, mask

    @property
    def unbinned_pixel_number(self):
        """
        Define the number of pixels of the unbinned detector.

        Convention: (vertical, horizontal)
        """
        return 3262, 3108

    @property
    def unbinned_pixel_size(self):
        """Pixel size (vertical, horizontal) of the unbinned detector in meters."""
        return 75e-06, 75e-06


class Timepix(Detector):
    """Implementation of the Timepix detector."""

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

    @property
    def unbinned_pixel_number(self):
        """
        Define the number of pixels of the unbinned detector.

        Convention: (vertical, horizontal)
        """
        return 256, 256

    @property
    def unbinned_pixel_size(self):
        """Pixel size (vertical, horizontal) of the unbinned detector in meters."""
        return 55e-06, 55e-06


class Merlin(Detector):
    """Implementation of the Merlin detector."""

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)
        self.saturation_threshold = 1e6

    def _mask_gaps(self, data, mask):
        """
        Mask the gaps between sensors in the detector.

        :param data: a 2D numpy array
        :param mask: a 2D numpy array of the same shape as data
        :return:

         - the masked data
         - the updated mask

        """
        valid.valid_ndarray(
            (data, mask), ndim=2, shape=self.unbinned_pixel_number, fix_shape=True
        )

        data[:, 255:260] = 0
        data[255:260, :] = 0

        mask[:, 255:260] = 1
        mask[255:260, :] = 1
        return data, mask

    @property
    def unbinned_pixel_number(self):
        """
        Define the number of pixels of the unbinned detector.

        Convention: (vertical, horizontal)
        """
        return 515, 515

    @property
    def unbinned_pixel_size(self):
        """Pixel size (vertical, horizontal) of the unbinned detector in meters."""
        return 55e-06, 55e-06


class MerlinSixS(Detector):
    """Implementation of the Merlin detector for SixS."""

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)
        self.saturation_threshold = 1e6

    def _mask_gaps(self, data, mask):
        """
        Mask the gaps between sensors in the detector.

        :param data: a 2D numpy array
        :param mask: a 2D numpy array of the same shape as data
        :return:

         - the masked data
         - the updated mask

        """
        valid.valid_ndarray(
            (data, mask), ndim=2, shape=self.unbinned_pixel_number, fix_shape=True
        )

        data[:, 254:257] = 0
        data[254:257, :] = 0

        mask[:, 254:257] = 1
        mask[254:257, :] = 1
        return data, mask

    @property
    def unbinned_pixel_number(self):
        """
        Define the number of pixels of the unbinned detector.

        Convention: (vertical, horizontal)
        """
        return 512, 512

    @property
    def unbinned_pixel_size(self):
        """Pixel size (vertical, horizontal) of the unbinned detector in meters."""
        return 55e-06, 55e-06


class Dummy(Detector):
    """
    Implementation of the Dummy detector.

    :param kwargs:
     - 'custom_pixelnumber': (V, H) number of pixels of the unbinned dummy detector, as
       a tuple of two positive integers.
     - 'custom_pixelsize': float, pixel size of the dummy detector in m.

    """

    def __init__(self, name, **kwargs):
        self.custom_pixelsize = kwargs.get("custom_pixelsize")
        valid.valid_item(
            self.custom_pixelsize,
            allowed_types=Real,
            min_excluded=0,
            allow_none=True,
            name="custom_pixelsize",
        )
        self.custom_pixelnumber = kwargs.get("custom_pixelnumber")
        if isinstance(self.custom_pixelnumber, np.ndarray):
            self.custom_pixelnumber = list(self.custom_pixelnumber)
        valid.valid_container(
            self.custom_pixelnumber,
            container_types=(list, tuple),
            length=2,
            item_types=Integral,
            min_excluded=0,
            allow_none=True,
            name="custom_pixelnumber",
        )
        super().__init__(name=name, **kwargs)

    @property
    def unbinned_pixel_number(self):
        """
        Define the number of pixels of the unbinned detector.

        Convention: (vertical, horizontal)
        """
        if self.custom_pixelnumber is not None and all(
            val is not None for val in self.custom_pixelnumber
        ):
            return self.custom_pixelnumber
        self.logger.info(f"Defaulting the pixel number to {516, 516}")
        return 516, 516

    @property
    def unbinned_pixel_size(self):
        """Pixel size (vertical, horizontal) of the unbinned detector in meters."""
        if self.custom_pixelsize is not None:
            return self.custom_pixelsize, self.custom_pixelsize
        self.logger.info(f"Defaulting the pixel size to {55e-06, 55e-06}")
        return 55e-06, 55e-06


class Lambda(Detector):
    """Implementation of the Lambda detector."""

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)
        self._counter_table = {"BM02": "img"}
        # useful if the same type of detector is used at several beamlines
        self.saturation_threshold = 1.5e6

    @property
    def unbinned_pixel_number(self):
        """
        Define the number of pixels of the unbinned detector.

        Convention: (vertical, horizontal)
        """
        return 516, 516

    @property
    def unbinned_pixel_size(self):
        """Pixel size (vertical, horizontal) of the unbinned detector in meters."""
        return 55e-06, 55e-06
