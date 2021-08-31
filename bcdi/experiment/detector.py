# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#   (c) 06/2021-present : DESY CFEL
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

"""
Detector classes.

These classes handles the detector config used for data acquisition. The available
detectors are:

- Maxipix
- Eiger2M
- Eiger4M
- Timepix
- Merlin
- Dummy

"""

import numpy as np
from numbers import Real
import os

from bcdi.utils import validation as valid


def create_detector(name, **kwargs):
    """
    Create a Detector instance depending on the detector.

    :param name: str, name of the detector
    :return:  the corresponding diffractometer instance
    """
    if name == "Maxipix":
        return Maxipix(name=name, **kwargs)
    if name == "Eiger2M":
        return Eiger2M(name=name, **kwargs)
    if name == "Eiger4M":
        return Eiger4M(name=name, **kwargs)
    if name == "Timepix":
        return Timepix(name=name, **kwargs)
    if name == "Merlin":
        return Merlin(name=name, **kwargs)
    if name == "Dummy":
        return Dummy(name=name, **kwargs)
    raise NotImplementedError(
        f"No implementation for the {name} detector"
    )


class Detector:
    """
    Class to handle the configuration of the detector used for data acquisition.

    :param name: name of the detector in {'Maxipix', 'Timepix', 'Merlin', 'Eiger2M',
     'Eiger4M', 'Dummy'}
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
    :param kwargs:

     - 'nb_pixel_x' and 'nb_pixel_y': useful when part of the detector is broken
       (less pixels than expected)
     - 'preprocessing_binning': tuple of the three binning factors used in a previous
       preprocessing step
     - 'offsets': tuple or list, sample and detector offsets corresponding to the
       parameter delta in xrayutilities hxrd.Ang2Q.area method
     - 'linearity_func': function to apply to each pixel of the detector in order to
       compensate the deviation of the detector linearity for large intensities.

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
        # the detector name should be initialized first,
        # other properties are depending on it
        self.name = name

        valid.valid_kwargs(
            kwargs=kwargs,
            allowed_kwargs={
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

    @property
    def counter(self):
        """Name of the counter for the image number."""
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
        self._datadir = value

    @property
    def name(self):
        """Name of the detector."""
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
        Horizontal number of pixels of the detector.

        It takes into account an eventual preprocessing binning.
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
        Vertical number of pixels of the detector.

        It takes into account an eventual preprocessing binning.
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
        """Return a dictionnary with all parameters."""
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
        """Horizontal pixel size of the detector after taking into account binning."""
        return self.unbinned_pixel[1] * self.preprocessing_binning[2] * self.binning[2]

    @property
    def pixelsize_y(self):
        """Vertical pixel size of the detector after taking into account binning."""
        return self.unbinned_pixel[0] * self.preprocessing_binning[1] * self.binning[1]

    @property
    def pix_number(self):
        """
        Define the number of pixels of the unbinned detector.

        Convention: (vertical, horizontal)
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
            min_length=1,
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
        self._savedir = value

    @property
    def scandir(self):
        """Path of the scan, typically it is the parent folder of the data folder."""
        if self.datadir:
            dir_path = os.path.abspath(os.path.join(self.datadir, os.pardir)) + "/"
            return dir_path.replace("\\", "/")

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
        """Template that can be used to generate template_imagefile."""
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
        """Name of the data file."""
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
        """Pixel size (vertical, horizontal) of the unbinned detector in meters."""
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
        """Representation string of the Detector instance."""
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"unbinned_pixel={self.unbinned_pixel}, "
            f"nb_pixel_x={self.nb_pixel_x}, "
            f"nb_pixel_y={self.nb_pixel_y}, "
            f"binning={self.binning},\n"
            f"roi={self.roi}, "
            f"sum_roi={self.sum_roi}, "
            f"preprocessing_binning={self.preprocessing_binning}, "
            f"rootdir = {self.rootdir},\n"
            f"datadir = {self.datadir},\n"
            f"scandir = {self.scandir},\n"
            f"savedir = {self.savedir},\n"
            f"sample_name = {self.sample_name},"
            f" template_file = {self.template_file}, "
            f"template_imagefile = {self.template_imagefile},"
            f" specfile = {self.specfile},\n"
        )

    def mask_detector(
        self, data, mask, nb_img=1, flatfield=None, background=None, hotpixels=None
    ):
        """
        Mask data measured with a 2D detector.

        It can apply flatfield correction, background subtraction, masking of hotpixels
        and detector gaps.

        :param data: the 2D data to mask
        :param mask: the 2D mask to be updated
        :param nb_img: number of images summed to yield the 2D data
         (e.g. in a series measurement), used when defining the threshold for hot pixels
        :param flatfield: the 2D flatfield array to be multiplied with the data
        :param background: a 2D array to be subtracted to the data
        :param hotpixels: a 2D array with hotpixels to be masked
         (1=hotpixel, 0=normal pixel)
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
