# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import gc
from math import isclose
from numbers import Real
import numpy as np
import os
import pathlib
from scipy.interpolate import RegularGridInterpolator
import sys
import warnings
sys.path.append('D:/myscripts/bcdi/')
import bcdi.graph.graph_utils as gu
import bcdi.utils.validation as valid


class Detector(object):
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
    """
    def __init__(self, name, rootdir=None, datadir=None, savedir=None, template_file=None, template_imagefile=None,
                 specfile=None, sample_name=None, roi=None, sum_roi=None, binning=(1, 1, 1), **kwargs):
        # the detector name should be initialized first, other properties are depending on it
        self.name = name

        valid.valid_kwargs(kwargs=kwargs,
                           allowed_kwargs={'is_series', 'nb_pixel_x', 'nb_pixel_y', 'preprocessing_binning', 'offsets'},
                           name='Detector.__init__')

        # load the kwargs
        self.is_series = kwargs.get('is_series', False)
        self.preprocessing_binning = kwargs.get('preprocessing_binning', None) or (1, 1, 1)
        self.nb_pixel_x = kwargs.get('nb_pixel_x', None)
        self.nb_pixel_y = kwargs.get('nb_pixel_y', None)
        self.offsets = kwargs.get('offsets', None)  # delegate the test to xrayutilities

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
        valid.valid_container(value, container_types=(tuple, list), length=3, item_types=int, min_excluded=0,
                              name='Detector.binning')
        self._binning = value

    @property
    def counter(self):
        """
        Name of the counter for the image number.
        """
        counter_dict = {'Maxipix': 'mpx4inr', 'Eiger2M': 'ei2minr', 'Eiger4M': None, 'Timepix': None, 'Merlin': 'alba2'}
        return counter_dict.get(self.name, None)

    @ property
    def datadir(self):
        """
        Name of the data directory
        """
        return self._datadir

    @datadir.setter
    def datadir(self, value):
        valid.valid_container(value, container_types=str, min_length=1, allow_none=True, name='Detector.datadir')
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
            raise TypeError('is_series should be a boolean')
        else:
            self._is_series = value

    @property
    def name(self):
        """
        Name of the detector: 'Maxipix', 'Timepix', 'Merlin', 'Eiger2M', 'Eiger4M'
        """
        return self._name

    @name.setter
    def name(self, value):
        if value not in {'Maxipix', 'Timepix', 'Merlin', 'Eiger2M', 'Eiger4M'}:
            raise ValueError("Name should be in {'Maxipix', 'Timepix', 'Merlin', 'Eiger2M', 'Eiger4M'}")
        else:
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
            raise TypeError('nb_pixel_x should be a positive integer')
        elif value <= 0:
            raise ValueError('nb_pixel_x should be a positive integer')
        else:
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
            raise TypeError('nb_pixel_y should be a positive integer')
        elif value <= 0:
            raise ValueError('nb_pixel_y should be a positive integer')
        else:
            self._nb_pixel_y = value // self.preprocessing_binning[1]

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
        if self.name == 'Maxipix':
            number = (516, 516)
        elif self.name == 'Timepix':
            number = (256, 256)
        elif self.name == 'Merlin':
            number = (515, 515)
        elif self.name == 'Eiger2M':
            number = (2164, 1030)
        elif self.name == 'Eiger4M':
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
        valid.valid_container(value, container_types=(tuple, list), length=3, item_types=int, min_excluded=0,
                              name='Detector.preprocessing_binning')
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
        valid.valid_container(value, container_types=(tuple, list), length=4, item_types=int, name='Detector.roi')
        self._roi = value

    @property
    def rootdir(self):
        """
        Name of the root directory, which englobes all scans
        """
        return self._rootdir

    @rootdir.setter
    def rootdir(self, value):
        valid.valid_container(value, container_types=str, min_length=1, allow_none=True, name='Detector.rootdir')
        self._rootdir = value

    @property
    def sample_name(self):
        """
        Name of the sample
        """
        return self._sample_name

    @sample_name.setter
    def sample_name(self, value):
        valid.valid_container(value, container_types=str, min_length=1, allow_none=True, name='Detector.sample_name')
        self._sample_name = value

    @property
    def savedir(self):
        """
        Name of the saving directory
        """
        return self._savedir

    @savedir.setter
    def savedir(self, value):
        valid.valid_container(value, container_types=str, min_length=1, allow_none=True, name='Detector.savedir')
        self._savedir = value

    @property
    def scandir(self):
        """
        Path of the scan, typically it is the parent folder of the data folder
        """
        if self.datadir:
            dir_path = os.path.abspath(os.path.join(self.datadir, os.pardir)) + '/'
            return dir_path.replace('\\', '/')

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
        valid.valid_container(value, container_types=(tuple, list), length=4, item_types=int, name='Detector.sum_roi')
        self._sum_roi = value

    @property
    def template_file(self):
        """
        Template that can be used to generate template_imagefile.
        """
        return self._template_file

    @template_file.setter
    def template_file(self, value):
        valid.valid_container(value, container_types=str, min_length=1, allow_none=True, name='Detector.template_file')
        self._template_file = value

    @property
    def template_imagefile(self):
        """
        Name of the data file.
        """
        return self._template_imagefile

    @template_imagefile.setter
    def template_imagefile(self, value):
        valid.valid_container(value, container_types=str, min_length=1, allow_none=True, name='Detector.imagefile')
        self._template_imagefile = value

    @property
    def unbinned_pixel(self):
        """
        Pixel size (vertical, horizontal) of the unbinned detector in meters.
        """
        if self.name in {'Maxipix', 'Timepix', 'Merlin'}:
            pix = (55e-06, 55e-06)
        elif self.name in {'Eiger2M', 'Eiger4M'}:
            pix = (75e-06, 75e-06)
        else:
            pix = None
        return pix

    def __repr__(self):
        """
        Representation string of the Detector instance.
        """
        return (f"{self.__class__.__name__}(name='{self.name}', unbinned_pixel={self.unbinned_pixel}, "
                f"nb_pixel_x={self.nb_pixel_x}, nb_pixel_y={self.nb_pixel_y}, binning={self.binning},\n"
                f"roi={self.roi}, sum_roi={self.sum_roi}, preprocessing_binning={self.preprocessing_binning}, "
                f"is_series={self.is_series}\nrootdir = {self.rootdir},\ndatadir = {self.datadir},\n"
                f"scandir = {self.scandir},\nsavedir = {self.savedir},\nsample_name = {self.sample_name},"
                f" template_file = {self.template_file}, template_imagefile = {self.template_imagefile},"
                f" specfile = {self.specfile},\n")

    def mask_detector(self, data, mask, nb_img=1, flatfield=None, background=None, hotpixels=None):
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

        assert isinstance(data, np.ndarray) and isinstance(mask, np.ndarray), 'data and mask should be numpy arrays'
        if data.ndim != 2 or mask.ndim != 2:
            raise ValueError('data and mask should be 2D arrays')

        if data.shape != mask.shape:
            raise ValueError('data and mask must have the same shape\n data is ', data.shape,
                             ' while mask is ', mask.shape)

        # flatfield correction
        if flatfield is not None:
            if flatfield.shape != data.shape:
                raise ValueError('flatfield and data must have the same shape\n data is ', flatfield.shape,
                                 ' while data is ', data.shape)
            data = np.multiply(flatfield, data)

        # remove the background
        if background is not None:
            if background.shape != data.shape:
                raise ValueError('background and data must have the same shape\n data is ', background.shape,
                                 ' while data is ', data.shape)
            data = data - background

        # mask hotpixels
        if hotpixels is not None:
            if hotpixels.shape != data.shape:
                raise ValueError('hotpixels and data must have the same shape\n data is ', hotpixels.shape,
                                 ' while data is ', data.shape)
            data[hotpixels == 1] = 0
            mask[hotpixels == 1] = 1

        if self.name == 'Eiger2M':
            data[:, 255: 259] = 0
            data[:, 513: 517] = 0
            data[:, 771: 775] = 0
            data[0: 257, 72: 80] = 0
            data[255: 259, :] = 0
            data[511: 552, :0] = 0
            data[804: 809, :] = 0
            data[1061: 1102, :] = 0
            data[1355: 1359, :] = 0
            data[1611: 1652, :] = 0
            data[1905: 1909, :] = 0
            data[1248:1290, 478] = 0
            data[1214:1298, 481] = 0
            data[1649:1910, 620:628] = 0

            mask[:, 255: 259] = 1
            mask[:, 513: 517] = 1
            mask[:, 771: 775] = 1
            mask[0: 257, 72: 80] = 1
            mask[255: 259, :] = 1
            mask[511: 552, :] = 1
            mask[804: 809, :] = 1
            mask[1061: 1102, :] = 1
            mask[1355: 1359, :] = 1
            mask[1611: 1652, :] = 1
            mask[1905: 1909, :] = 1
            mask[1248:1290, 478] = 1
            mask[1214:1298, 481] = 1
            mask[1649:1910, 620:628] = 1

            # mask hot pixels
            mask[data > 1e6 * nb_img] = 1
            data[data > 1e6 * nb_img] = 0

        elif self.name == 'Eiger4M':
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

        elif self.name == 'Maxipix':
            data[:, 255:261] = 0
            data[255:261, :] = 0

            mask[:, 255:261] = 1
            mask[255:261, :] = 1

            # mask hot pixels
            mask[data > 1e6 * nb_img] = 1
            data[data > 1e6 * nb_img] = 0

        elif self.name == 'Merlin':
            data[:, 255:260] = 0
            data[255:260, :] = 0

            mask[:, 255:260] = 1
            mask[255:260, :] = 1

            # mask hot pixels
            mask[data > 1e6 * nb_img] = 1
            data[data > 1e6 * nb_img] = 0

        elif self.name == 'Timepix':
            pass  # no gaps

        else:
            raise NotImplementedError('Detector not implemented')

        return data, mask


class Setup(object):
    """
    Class for defining the experimental geometry.

    :param beamline: name of the beamline, among {'ID01','SIXS_2018','SIXS_2019','34ID','P10','CRISTAL','NANOMAX'}
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
    :param pixel_x: detector horizontal pixel size, in meters.
    :param pixel_y: detector vertical pixel size, in meters.
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
     - 'sample_offsets': list or tuple of three angles in degrees, corresponding to the offsets of the sample
       goniometers around (downstream, vertical up, outboard). Convention: the sample offsets will be subtracted to
       the motor values.
     - 'offset_inplane': inplane offset of the detector defined as the outer angle in xrayutilities area detector
       calibration.
    """
    def __init__(self, beamline, beam_direction=(1, 0, 0), energy=None, distance=None, outofplane_angle=None,
                 inplane_angle=None, tilt_angle=None, rocking_angle=None, grazing_angle=None, pixel_x=None,
                 pixel_y=None, **kwargs):

        valid.valid_kwargs(kwargs=kwargs,
                           allowed_kwargs={'direct_beam', 'filtered_data', 'custom_scan', 'custom_images',
                                           'custom_monitor', 'custom_motors', 'sample_inplane', 'sample_outofplane',
                                           'sample_offsets', 'offset_inplane'},
                           name='Setup.__init__')

        # kwargs for preprocessing forward CDI data
        self.direct_beam = kwargs.get('direct_beam', None)
        # kwargs for loading and preprocessing data
        self.sample_offsets = kwargs.get('sample_offsets', (0, 0, 0))
        self.filtered_data = kwargs.get('filtered_data', False)  # boolean
        self.custom_scan = kwargs.get('custom_scan', False)  # boolean
        self.custom_images = kwargs.get('custom_images', None)  # list or tuple
        self.custom_monitor = kwargs.get('custom_monitor', None)  # list or tuple
        self.custom_motors = kwargs.get('custom_motors', None)  # dictionnary
        # kwargs for xrayutilities, delegate the test on their values to xrayutilities
        self.sample_inplane = kwargs.get('sample_inplane', (1, 0, 0))
        self.sample_outofplane = kwargs.get('sample_outofplane', (0, 0, 1))
        self.offset_inplane = kwargs.get('offset_inplane', 0)

        # load positional arguments corresponding to instance properties
        self.beamline = beamline
        self.beam_direction = beam_direction
        self.energy = energy
        self.distance = distance
        self.outofplane_angle = outofplane_angle
        self.inplane_angle = inplane_angle
        self.tilt_angle = tilt_angle
        self.rocking_angle = rocking_angle
        self.grazing_angle = grazing_angle
        self.pixel_x = pixel_x
        self.pixel_y = pixel_y

    @property
    def beam_direction(self):
        """
        Direction of the incident X-ray beam in the frame (z downstream, y vertical up, x outboard).
        """
        return self._beam_direction

    @beam_direction.setter
    def beam_direction(self, value):
        valid.valid_container(value, container_types=(tuple, list), length=3, item_types=Real,
                              name='Setup.beam_direction')
        if np.linalg.norm(value) == 0:
            raise ValueError('At least of component of beam_direction should be non null.')
        else:
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
        if value not in {'ID01', 'SIXS_2018', 'SIXS_2019', '34ID', 'P10', 'CRISTAL', 'NANOMAX'}:
            raise ValueError(f'Beamline {value} not supported')
        else:
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
            valid.valid_container(value, container_types=(tuple, list), min_length=1, item_types=int,
                                  name='Setup.custom_images')
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
        if not self._custom_scan:
            self._custom_monitor = None
        else:
            if value is None:
                value = np.ones(len(self._custom_images))
            valid.valid_container(value, container_types=(tuple, list, np.ndarray), length=len(self._custom_images),
                                  item_types=Real, name='Setup.custom_monitor')
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
                raise TypeError('custom_monitor should be a dictionnary of "motor_name": motor_positions pairs')
            else:
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
            raise TypeError('custom_scan should be a boolean')
        else:
            self._custom_scan = value

    @property
    def detector_hor(self):
        """
        Defines the horizontal detector orientation for xrayutilities depending on the beamline.
         The frame convention of xrayutilities is the following: x downstream, y outboard, z vertical up.
        """
        if self.beamline in {'ID01', 'SIXS_2018', 'SIXS_2019', 'CRISTAL', 'NANOMAX'}:
            # we look at the detector from downstream, detector X along the outboard direction
            return 'y+'
        else:  # 'P10', '34ID'
            # we look at the detector from upstream, detector X opposite to the outboard direction
            return 'y-'

    @property
    def detector_ver(self):
        """
        Defines the vertical detector orientation for xrayutilities depending on the beamline.
         The frame convention of xrayutilities is the following: x downstream, y outboard, z vertical up.
        """
        if self.beamline in {'ID01', 'SIXS_2018', 'SIXS_2019', 'CRISTAL', 'NANOMAX', 'P10', '34ID'}:
            # origin is at the top, detector Y along vertical down
            return 'z-'
        else:
            return 'z+'

    @property
    def direct_beam(self):
        """
        Tuple of two real numbers indicating the position of the direct beam in pixels at zero detector angles.
        """
        return self._direct_beam

    @direct_beam.setter
    def direct_beam(self, value):
        if value is not None:
            valid.valid_container(value, container_types=(tuple, list), length=2, item_types=Real,
                                  name='Setup.direct_beam')
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
            raise TypeError('distance should be a number in m')
        elif value <= 0:
            raise ValueError('distance should be a strictly positive number in m')
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
        elif not isinstance(value, Real):
            raise TypeError('energy should be a number in eV')
        elif value <= 0:
            raise ValueError('energy should be a strictly positive number in eV')
        else:
            self._energy = value

    @property
    def exit_wavevector(self):
        """
        Calculate the exit wavevector kout depending on the setup parameters, in the laboratory frame (z downstream,
         y vertical, x outboard).

        :return: kout vector
        """
        if self.beamline == 'SIXS_2018' or self.beamline == 'SIXS_2019':
            # gamma is anti-clockwise
            kout = 2 * np.pi / self.wavelength * np.array(
                [np.cos(np.pi * self.inplane_angle / 180) * np.cos(np.pi * self.outofplane_angle / 180),  # z
                 np.sin(np.pi * self.outofplane_angle / 180),  # y
                 np.sin(np.pi * self.inplane_angle / 180) * np.cos(np.pi * self.outofplane_angle / 180)])  # x
        elif self.beamline == 'ID01':
            # nu is clockwise
            kout = 2 * np.pi / self.wavelength * np.array(
                [np.cos(np.pi * self.inplane_angle / 180) * np.cos(np.pi * self.outofplane_angle / 180),  # z
                 np.sin(np.pi * self.outofplane_angle / 180),  # y
                 -np.sin(np.pi * self.inplane_angle / 180) * np.cos(np.pi * self.outofplane_angle / 180)])  # x
        elif self.beamline == '34ID':
            # gamma is anti-clockwise
            kout = 2 * np.pi / self.wavelength * np.array(
                [np.cos(np.pi * self.inplane_angle / 180) * np.cos(np.pi * self.outofplane_angle / 180),  # z
                 np.sin(np.pi * self.outofplane_angle / 180),  # y
                 np.sin(np.pi * self.inplane_angle / 180) * np.cos(np.pi * self.outofplane_angle / 180)])  # x
        elif self.beamline == 'NANOMAX':
            # gamma is clockwise
            kout = 2 * np.pi / self.wavelength * np.array(
                [np.cos(np.pi * self.inplane_angle / 180) * np.cos(np.pi * self.outofplane_angle / 180),  # z
                 np.sin(np.pi * self.outofplane_angle / 180),  # y
                 -np.sin(np.pi * self.inplane_angle / 180) * np.cos(np.pi * self.outofplane_angle / 180)])  # x
        elif self.beamline == 'P10':
            # gamma is anti-clockwise
            kout = 2 * np.pi / self.wavelength * np.array(
                [np.cos(np.pi * self.inplane_angle / 180) * np.cos(np.pi * self.outofplane_angle / 180),  # z
                 np.sin(np.pi * self.outofplane_angle / 180),  # y
                 np.sin(np.pi * self.inplane_angle / 180) * np.cos(np.pi * self.outofplane_angle / 180)])  # x
        elif self.beamline == 'CRISTAL':
            # gamma is anti-clockwise
            kout = 2 * np.pi / self.wavelength * np.array(
                [np.cos(np.pi * self.inplane_angle / 180) * np.cos(np.pi * self.outofplane_angle / 180),  # z
                 np.sin(np.pi * self.outofplane_angle / 180),  # y
                 np.sin(np.pi * self.inplane_angle / 180) * np.cos(np.pi * self.outofplane_angle / 180)])  # x
        else:
            raise ValueError('setup parameter: ', self.beamline, 'not defined')
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
            raise TypeError('filtered_data should be a boolean')
        else:
            self._filtered_data = value

    @property
    def grazing_angle(self):
        """
        Motor positions for the goniometer circles below the rocking angle. It should be a list/tuple of lenght 1 for
         out-of-plane rocking curves (the chi motor value) and length 2 for inplane rocking curves
         (the chi and omega/om/eta motor values).
        """
        return self._grazing_angle

    @grazing_angle.setter
    def grazing_angle(self, value):
        if self.rocking_angle == 'outofplane':
            # only the chi angle (rotation around z, below the rocking angle omega/om/eta) is needed
            valid.valid_container(value, container_types=(tuple, list), length=1, item_types=Real, allow_none=True,
                                  name='Setup.grazing_angle')
            self._grazing_angle = value
        elif self.rocking_angle == 'inplane':
            # two values needed: the chi angle and the omega/om/eta angle (rotations respectively around z and x,
            # below the rocking angle phi)
            valid.valid_container(value, container_types=(tuple, list), length=2, item_types=Real, allow_none=True,
                                  name='Setup.grazing_angle')
            self._grazing_angle = value
        else:  # self.rocking_angle == 'energy'
            # there is no sample rocking for energy scans, hence the grazing angle value do not matter
            self._grazing_angle = None

    @property
    def inplane_angle(self):
        """
        Horizontal detector angle, in degrees.
        """
        return self._inplane_angle

    @inplane_angle.setter
    def inplane_angle(self, value):
        if not isinstance(value, Real) and value is not None:
            raise TypeError('inplane_angle should be a number in degrees')
        else:
            self._inplane_angle = value

    @property
    def inplane_coeff(self):
        """
        Define a coefficient +/- 1 depending on the detector inplane rotation direction and the detector inplane
         orientation. The frame convention is the one of xrayutilities: x downstream, y outboard, z vertical up.
         See postprocessing/scripts/correct_angles_detector.py for an example.

        :return: +1 or -1
        """
        if self.detector_hor == 'y+':
            hor_coeff = 1
        else:  # 'y-'
            hor_coeff = -1

        if self.beamline == 'SIXS_2018' or self.beamline == 'SIXS_2019':
            # gamma is anti-clockwise, we see the detector from downstream
            coeff_inplane = 1 * hor_coeff
        elif self.beamline == 'ID01':
            # nu is clockwise, we see the detector from downstream
            coeff_inplane = -1 * hor_coeff
        elif self.beamline == '34ID':
            # delta is anti-clockwise, we see the detector from the front
            coeff_inplane = 1 * hor_coeff
        elif self.beamline == 'P10':
            # gamma is anti-clockwise, we see the detector from the front
            coeff_inplane = 1 * hor_coeff
        elif self.beamline == 'CRISTAL':
            # gamma is anti-clockwise, we see the detector from downstream
            coeff_inplane = 1 * hor_coeff
        elif self.beamline == 'NANOMAX':
            # gamma is clockwise, we see the detector from downstream
            coeff_inplane = -1 * hor_coeff
        else:
            raise ValueError('setup parameter: ', self.beamline, 'not defined')
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
            raise TypeError('outofplane_angle should be a number in degrees')
        else:
            self._outofplane_angle = value

    @property
    def outofplane_coeff(self):
        """
        Define a coefficient +/- 1 depending on the detector out of plane rotation direction and the detector out of
         plane orientation. The frame convention is the one of xrayutilities: x downstream, y outboard, z vertical up.
         See postprocessing/scripts/correct_angles_detector.py for an example.

        :return: +1 or -1
        """
        if self.detector_ver == 'z+':  # origin of pixels at the bottom
            ver_coeff = 1
        else:  # 'z-'  origin of pixels at the top
            ver_coeff = -1
        # the out of plane detector rotation is clockwise for all beamlines
        coeff_outofplane = -1 * ver_coeff
        return coeff_outofplane

    @property
    def pixel_x(self):
        """
        Detector horizontal pixel size, in meters.
        """
        return self._pixel_x

    @pixel_x.setter
    def pixel_x(self, value):
        if value is None:
            self._pixel_x = value
        elif not isinstance(value, Real):
            raise TypeError('pixel_x should be a number in m')
        elif value <= 0:
            raise ValueError('pixel_x should be a strictly positive number in m')
        else:
            self._pixel_x = value

    @property
    def pixel_y(self):
        """
        Detector vertical pixel size, in meters.
        """
        return self._pixel_y

    @pixel_y.setter
    def pixel_y(self, value):
        if value is None:
            self._pixel_y = value
        elif not isinstance(value, Real):
            raise TypeError('pixel_y should be a number in m')
        elif value <= 0:
            raise ValueError('pixel_y should be a strictly positive number in m')
        else:
            self._pixel_y = value

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
            raise TypeError('rocking_angle should be a str')
        elif value not in {'outofplane', 'inplane', 'energy'}:
            raise ValueError('rocking_angle can take only the value "outofplane", "inplane" or "energy"')
        else:
            self._rocking_angle = value

    @property
    def sample_offsets(self):
        """
        List or tuple of three angles in degrees, corresponding to the offsets of the sample goniometers around
        (downstream, vertical up, outboard). Convention: the sample offsets will be subtracted to the motor values.
        """
        return self._sample_offsets

    @sample_offsets.setter
    def sample_offsets(self, value):
        valid.valid_container(value, container_types=(tuple, list), length=3, item_types=Real,
                              name='Setup.sample_offsets')
        self._sample_offsets = value

    @property
    def tilt_angle(self):
        """
        Angular step of the rocking curve, in degrees.
        """
        return self._tilt_angle

    @tilt_angle.setter
    def tilt_angle(self, value):
        if not isinstance(value, Real) and value is not None:
            raise TypeError('tilt_angle should be a number in degrees')
        else:
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
        return (f"{self.__class__.__name__}(beamline='{self.beamline}', beam_direction={self.beam_direction}, "
                f"energy={self.energy}, distance={self.distance}, outofplane_angle={self.outofplane_angle},\n"
                f"inplane_angle={self.inplane_angle}, tilt_angle={self.tilt_angle}, "
                f"rocking_angle='{self.rocking_angle}', grazing_angle={self.grazing_angle}, pixel_x={self.pixel_x},\n"
                f"pixel_y={self.pixel_y}, direct_beam={self.direct_beam}, sample_offsets={self.sample_offsets}, "
                f"filtered_data={self.filtered_data}, custom_scan={self.custom_scan},\n"
                f"custom_images={self.custom_images},\ncustom_monitor={self.custom_monitor},\n"
                f"custom_motors={self.custom_motors},\n"
                f"sample_inplane={self.sample_inplane}, sample_outofplane={self.sample_outofplane}, "
                f"offset_inplane={self.offset_inplane})")

    def detector_frame(self, obj, voxel_size, width_z=None, width_y=None, width_x=None,
                       debugging=False, **kwargs):
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
        valid.valid_kwargs(kwargs=kwargs, allowed_kwargs={'title'}, name='Setup.detector_frame')
        title = kwargs.get('title', 'Object')

        if isinstance(voxel_size, Real):
            voxel_size = (voxel_size, voxel_size, voxel_size)
        valid.valid_container(obj=voxel_size, container_types=(tuple, list), length=3, item_types=Real,
                              min_excluded=0, name='Setup.detector_frame')

        nbz, nby, nbx = obj.shape

        if debugging:
            gu.multislices_plot(abs(obj), sum_frames=True, width_z=width_z, width_y=width_y, width_x=width_x,
                                title=title + ' before interpolation\n')

        ortho_matrix = self.transformation_matrix(array_shape=(nbz, nby, nbx), tilt_angle=self.tilt_angle,
                                                  pixel_x=self.pixel_x, pixel_y=self.pixel_y)

        ################################################
        # interpolate the data into the detector frame #
        ################################################
        myz, myy, myx = np.meshgrid(np.arange(-nbz // 2, nbz // 2, 1),
                                    np.arange(-nby // 2, nby // 2, 1),
                                    np.arange(-nbx // 2, nbx // 2, 1), indexing='ij')

        new_x = ortho_matrix[0, 0] * myx + ortho_matrix[0, 1] * myy + ortho_matrix[0, 2] * myz
        new_y = ortho_matrix[1, 0] * myx + ortho_matrix[1, 1] * myy + ortho_matrix[1, 2] * myz
        new_z = ortho_matrix[2, 0] * myx + ortho_matrix[2, 1] * myy + ortho_matrix[2, 2] * myz
        del myx, myy, myz
        # la partie rgi est sure: c'est la taille de l'objet orthogonal de depart
        rgi = RegularGridInterpolator((np.arange(-nbz // 2, nbz // 2) * voxel_size[0],
                                       np.arange(-nby // 2, nby // 2) * voxel_size[1],
                                       np.arange(-nbx // 2, nbx // 2) * voxel_size[2]),
                                      obj, method='linear', bounds_error=False, fill_value=0)
        detector_obj = rgi(np.concatenate((new_z.reshape((1, new_z.size)), new_y.reshape((1, new_z.size)),
                                           new_x.reshape((1, new_z.size)))).transpose())
        detector_obj = detector_obj.reshape((nbz, nby, nbx)).astype(obj.dtype)

        if debugging:
            gu.multislices_plot(abs(detector_obj), sum_frames=True, width_z=width_z, width_y=width_y, width_x=width_x,
                                title=title + ' interpolated in detector frame\n')

        return detector_obj

    def init_paths(self, detector, sample_name, scan_number, root_folder, save_dir, specfile_name, template_imagefile,
                   save_dirname='result', create_savedir=False, verbose=False):
        """
        Update the detector instance with initialized paths and template for filenames depending on the beamline

        :param detector: instance of the Class Detector
        :param sample_name: string in front of the scan number in the data folder name.
        :param scan_number: the scan number
        :param root_folder: folder of the experiment, where all scans are stored
        :param save_dir: path of the directory where to save the analysis results, can be None
        :param specfile_name: beamline-dependent string
         - ID01: name of the spec file without '.spec'
         - SIXS_2018: full path of the alias dictionnary, typically root_folder + 'alias_dict_2019.txt'
         - empty string for all other beamlines
        :param template_imagefile: beamline-dependent template for the data files
         - ID01: 'data_mpx4_%05d.edf.gz' or 'align_eiger2M_%05d.edf.gz'
         - SIXS_2018: 'align.spec_ascan_mu_%05d.nxs'
         - SIXS_2019: 'spare_ascan_mu_%05d.nxs'
         - Cristal: 'S%d.nxs'
         - P10: '_master.h5'
         - NANOMAX: '%06d.h5'
         - 34ID: 'Sample%dC_ES_data_51_256_256.npz'
        :param save_dirname: name of the saving folder, by default 'save_dir/result/' will be created
        :param create_savedir: boolean, True to create the saving folder if it does not exist
        :param verbose: True to print the paths
        """
        if not isinstance(detector, Detector):
            raise TypeError('detector should be an instance of the Class Detector')

        if not isinstance(scan_number, int):
            raise TypeError('scan_number should be an integer')

        if not isinstance(sample_name, str):
            raise TypeError('sample_name should be a string')
        # check that the name is not an empty string
        valid.valid_container(save_dirname, container_types=str, min_length=1, name='Setup.init_paths')
        detector.rootdir, detector.sample_name, detector.template_file = root_folder, sample_name, template_imagefile

        if self.beamline == 'P10':
            specfile = sample_name + '_{:05d}'.format(scan_number)
            homedir = root_folder + specfile + '/'
            datadir = homedir + 'e4m/'
            template_imagefile = specfile + template_imagefile
            scan_template = sample_name + '_{:05d}'.format(scan_number) + '/'  # used to create the folder
        elif self.beamline == 'NANOMAX':
            homedir = root_folder + sample_name + '{:06d}'.format(scan_number) + '/'
            datadir = homedir + 'data/'
            specfile = specfile_name
            scan_template = sample_name + '_{:06d}'.format(scan_number) + '/'  # used to create the folder
        else:
            homedir = root_folder + sample_name + str(scan_number) + '/'
            datadir = homedir + "data/"
            specfile = specfile_name
            scan_template = sample_name + '_' + str(scan_number) + '/'  # used to create the folder
        if save_dir:
            savedir = save_dir + scan_template + save_dirname + '/'
        else:
            savedir = homedir + save_dirname + '/'
        detector.savedir, detector.datadir, detector.specfile, detector.template_imagefile = \
            savedir, datadir, specfile, template_imagefile
        if create_savedir:
            pathlib.Path(detector.savedir).mkdir(parents=True, exist_ok=True)
        if verbose:
            if not self.custom_scan:
                print(f"datadir = '{datadir}'\nsavedir = '{savedir}'\ntemplate_imagefile = '{template_imagefile}'\n")
            else:
                print(f"rootdir = '{root_folder}'\nsavedir = '{savedir}'\nsample_name = '{detector.sample_name}'\n"
                      f"template_imagefile = '{detector.template_file}'\n")

    def orthogonalize(self, obj, initial_shape=None, voxel_size=None, width_z=None, width_y=None,
                      width_x=None, verbose=True, debugging=False, **kwargs):
        """
        Interpolate obj on the orthogonal reference frame defined by the setup.

        :param obj: real space object, in a non-orthogonal frame (output of phasing program)
        :param initial_shape: shape of the FFT used for phasing
        :param voxel_size: number or list of three user-defined voxel sizes for the interpolation, in nm.
         If a single number is provided, the voxel size will be identical in all directions.
        :param width_z: size of the area to plot in z (axis 0), centered on the middle of the initial array
        :param width_y: size of the area to plot in y (axis 1), centered on the middle of the initial array
        :param width_x: size of the area to plot in x (axis 2), centered on the middle of the initial array
        :param verbose: True to have printed comments
        :param debugging: True to show plots before and after interpolation
        :param kwargs:
         - 'title': title for the debugging plots
        :return: object interpolated on an orthogonal grid
        """
        valid.valid_kwargs(kwargs=kwargs, allowed_kwargs={'title'}, name='Setup.orthogonalize')
        title = kwargs.get('title', 'Object')

        if not initial_shape:
            initial_shape = obj.shape
        else:
            valid.valid_container(initial_shape, container_types=(tuple, list), length=3, item_types=int,
                                  min_excluded=0, name='Setup.orthogonalize')

        if debugging:
            gu.multislices_plot(abs(obj), sum_frames=True, width_z=width_z, width_y=width_y, width_x=width_x,
                                title=title+' in detector frame')

        # estimate the direct space voxel sizes in nm based on the FFT window shape used in phase retrieval
        dz_realspace, dy_realspace, dx_realspace = self.voxel_sizes(initial_shape, tilt_angle=abs(self.tilt_angle),
                                                                    pixel_x=self.pixel_x, pixel_y=self.pixel_y)

        if verbose:
            print('Direct space voxel sizes (z, y, x) based on initial FFT shape: (',
                  str('{:.2f}'.format(dz_realspace)), 'nm,',
                  str('{:.2f}'.format(dy_realspace)), 'nm,',
                  str('{:.2f}'.format(dx_realspace)), 'nm )')

        nbz, nby, nbx = obj.shape  # could be smaller if the object was cropped around the support
        if nbz != initial_shape[0] or nby != initial_shape[1] or nbx != initial_shape[2]:
            # recalculate the tilt and pixel sizes to accomodate a shape change
            tilt = self.tilt_angle * initial_shape[0] / nbz
            pixel_y = self.pixel_y * initial_shape[1] / nby
            pixel_x = self.pixel_x * initial_shape[2] / nbx
            if verbose:
                print('Tilt, pixel_y, pixel_x based on cropped array shape: (',
                      str('{:.4f}'.format(tilt)), 'deg,',
                      str('{:.2f}'.format(pixel_y * 1e6)), 'um,',
                      str('{:.2f}'.format(pixel_x * 1e6)), 'um)')

            # sanity check, the direct space voxel sizes calculated below should be equal to the original ones
            dz_realspace, dy_realspace, dx_realspace = self.voxel_sizes((nbz, nby, nbx), tilt_angle=abs(tilt),
                                                                        pixel_x=pixel_x, pixel_y=pixel_y)
            if verbose:
                print('Sanity check, recalculated direct space voxel sizes: (',
                      str('{:.2f}'.format(dz_realspace)), 'nm,',
                      str('{:.2f}'.format(dy_realspace)), 'nm,',
                      str('{:.2f}'.format(dx_realspace)), 'nm )')
        else:
            tilt = self.tilt_angle
            pixel_y = self.pixel_y
            pixel_x = self.pixel_x

        if not voxel_size:
            voxel_size = dz_realspace, dy_realspace, dx_realspace  # in nm
        else:
            if isinstance(voxel_size, Real):
                voxel_size = (voxel_size, voxel_size, voxel_size)
            assert isinstance(voxel_size, (tuple, list)) and len(voxel_size) == 3 and\
                all(val > 0 for val in voxel_size), 'voxel_size should be a list/tuple of three positive numbers in nm'

        ortho_matrix = self.transformation_matrix(array_shape=(nbz, nby, nbx), tilt_angle=tilt,
                                                  pixel_x=pixel_x, pixel_y=pixel_y, verbose=verbose)

        # this assumes that the direct beam was at the center of the array
        # TODO : correct this if the position of the direct beam is provided
        myz, myy, myx = np.meshgrid(np.arange(-nbz // 2, nbz // 2, 1) * voxel_size[0],
                                    np.arange(-nby // 2, nby // 2, 1) * voxel_size[1],
                                    np.arange(-nbx // 2, nbx // 2, 1) * voxel_size[2], indexing='ij')

        # ortho_matrix is the transformation matrix from the detector coordinates to the laboratory frame
        # in RGI, we want to calculate the coordinates that would have a grid of the laboratory frame expressed in the
        # detector frame, i.e. one has to inverse the transformation matrix.
        ortho_imatrix = np.linalg.inv(ortho_matrix)
        new_x = ortho_imatrix[0, 0] * myx + ortho_imatrix[0, 1] * myy + ortho_imatrix[0, 2] * myz
        new_y = ortho_imatrix[1, 0] * myx + ortho_imatrix[1, 1] * myy + ortho_imatrix[1, 2] * myz
        new_z = ortho_imatrix[2, 0] * myx + ortho_imatrix[2, 1] * myy + ortho_imatrix[2, 2] * myz
        del myx, myy, myz
        gc.collect()

        rgi = RegularGridInterpolator((np.arange(-nbz // 2, nbz // 2), np.arange(-nby // 2, nby // 2),
                                       np.arange(-nbx // 2, nbx // 2)), obj, method='linear',
                                      bounds_error=False, fill_value=0)
        ortho_obj = rgi(np.concatenate((new_z.reshape((1, new_z.size)), new_y.reshape((1, new_z.size)),
                                        new_x.reshape((1, new_z.size)))).transpose())
        ortho_obj = ortho_obj.reshape((nbz, nby, nbx)).astype(obj.dtype)

        if debugging:
            gu.multislices_plot(abs(ortho_obj), sum_frames=True, width_z=width_z, width_y=width_y, width_x=width_x,
                                title=title+' in the orthogonal laboratory frame')
        return ortho_obj, voxel_size

    def ortho_reciprocal(self, obj, method_shape='fix_sampling', verbose=True, debugging=False, **kwargs):
        """
        Interpolate obj in the orthogonal laboratory frame (z/qx downstream, y/qz vertical up, x/qy outboard).

        :param obj: reciprocal space diffraction pattern, in the detector frame
        :param method_shape: if 'fix_shape', the output array will have the same shape as the input array.
         If 'fix_sampling', the ouput shape will be increased in order to keep the sampling in q in each direction.
        :param verbose: True to have printed comments
        :param debugging: True to show plots before and after interpolation
        :param kwargs:
         - 'title': title for the debugging plots
         - 'scale': 'linear' or 'log', scale for the debugging plots
         - width_z: size of the area to plot in z (axis 0), centered on the middle of the initial array
         - width_y: size of the area to plot in y (axis 1), centered on the middle of the initial array
         - width_x: size of the area to plot in x (axis 2), centered on the middle of the initial array
        :return: object interpolated on an orthogonal grid, q values as a tuple of three 1D vectors (qx, qz, qy)
        """
        # check and load kwargs
        valid.valid_kwargs(kwargs=kwargs, allowed_kwargs={'title', 'scale', 'width_z', 'width_y', 'width_x'},
                           name='Setup.orthogonalize')
        title = kwargs.get('title', 'Object')
        valid.valid_item(value=title, allowed_types=str, name='Setup.ortho_reciprocal')
        scale = kwargs.get('scale', 'log')
        valid.valid_item(value=scale, allowed_types=str, name='Setup.ortho_reciprocal')
        width_z = kwargs.get('width_z', None)
        valid.valid_item(value=width_z, allowed_types=int, min_excluded=0, allow_none=True,
                         name='Setup.ortho_reciprocal')
        width_y = kwargs.get('width_y', None)
        valid.valid_item(value=width_y, allowed_types=int, min_excluded=0, allow_none=True,
                         name='Setup.ortho_reciprocal')
        width_x = kwargs.get('width_x', None)
        valid.valid_item(value=width_x, allowed_types=int, min_excluded=0, allow_none=True,
                         name='Setup.ortho_reciprocal')

        # check some parameters
        if method_shape not in {'fix_sampling', 'fix_shape'}:
            raise ValueError('method_shape should be either "fix_sampling" or "fix_shape"')

        if not isinstance(obj, np.ndarray) or obj.ndim != 3:
            raise ValueError('obj should be a 3D numpy array')

        # plot the original data
        if debugging:
            gu.multislices_plot(abs(obj), sum_frames=True, scale=scale, plot_colorbar=True, width_z=width_z,
                                width_y=width_y, width_x=width_x, is_orthogonal=False, reciprocal_space=True, vmin=0,
                                title=title+' in detector frame')

        # calculate the transformation matrix (the unit is 1/nm)
        transfer_matrix, q_offset = self.transformation_matrix(array_shape=obj.shape, tilt_angle=self.tilt_angle,
                                                               direct_space=False, pixel_x=self.pixel_x,
                                                               pixel_y=self.pixel_y, verbose=verbose)

        # the voxel size in q in given by the rows of the transformation matrix (the unit is 1/nm)
        dq_along_x = np.linalg.norm(transfer_matrix[0, :])  # along x outboard
        dq_along_y = np.linalg.norm(transfer_matrix[1, :])  # along y vertical up
        dq_along_z = np.linalg.norm(transfer_matrix[2, :])  # along z downstream

        # calculate the shape of the output array
        if method_shape == 'fix_shape':
            nbz, nby, nbx = obj.shape
        else:  # 'fix_sampling'
            # calculate the direct space voxel sizes considering the current shape
            dz_realspace, dy_realspace, dx_realspace = self.voxel_sizes(obj.shape, tilt_angle=abs(self.tilt_angle),
                                                                        pixel_x=self.pixel_x, pixel_y=self.pixel_y)

            # calculate the new sizes nz, ny, nx such that the direct space voxel size is not affected by cropping
            # the diffraction pattern during the interpolation
            nbz = int(np.rint(2 * np.pi / (dz_realspace * dq_along_z)))
            nby = int(np.rint(2 * np.pi / (dy_realspace * dq_along_y)))
            nbx = int(np.rint(2 * np.pi / (dx_realspace * dq_along_x)))

            raise NotImplementedError('need to calculate the shape when keeping the sampling constant')

        # this assumes that the direct beam was at the center of the array
        # TODO : correct this if the position of the direct beam is provided

        # calculate qx qz qy vectors in 1/nm, the reference being the center of the array
        # the usual frame is used for q values: qx downstream, qz vertical up, qy outboard
        qx = np.arange(-nbz // 2, nbz // 2, 1) * dq_along_z  # along z downstream
        qz = np.arange(-nby // 2, nby // 2, 1) * dq_along_y  # along y vertical up
        qy = np.arange(-nbx // 2, nbx // 2, 1) * dq_along_x  # along x outboard

        myz, myy, myx = np.meshgrid(qx, qz, qy, indexing='ij')

        # ortho_matrix is the transformation matrix from the detector coordinates to the laboratory frame
        # in RGI, we want to calculate the coordinates that would have a grid of the laboratory frame expressed in
        # the detector frame, i.e. one has to inverse the transformation matrix.
        transfer_imatrix = np.linalg.inv(transfer_matrix)
        new_x = transfer_imatrix[0, 0] * myx + transfer_imatrix[0, 1] * myy + transfer_imatrix[0, 2] * myz
        new_y = transfer_imatrix[1, 0] * myx + transfer_imatrix[1, 1] * myy + transfer_imatrix[1, 2] * myz
        new_z = transfer_imatrix[2, 0] * myx + transfer_imatrix[2, 1] * myy + transfer_imatrix[2, 2] * myz
        del myx, myy, myz
        gc.collect()

        rgi = RegularGridInterpolator((np.arange(-nbz // 2, nbz // 2), np.arange(-nby // 2, nby // 2),
                                       np.arange(-nbx // 2, nbx // 2)), obj, method='linear',
                                      bounds_error=False, fill_value=0)
        ortho_obj = rgi(np.concatenate((new_z.reshape((1, new_z.size)), new_y.reshape((1, new_z.size)),
                                        new_x.reshape((1, new_z.size)))).transpose())
        ortho_obj = ortho_obj.reshape((len(qx), len(qz), len(qy))).astype(obj.dtype)

        # add the offset due to the detector angles to qx qz qy vectors, convert them to 1/A
        # the offset components are in the order (x/qy, y/qz, z/qx)
        qx = (qx + q_offset[2]) / 10  # along z downstream
        qz = (qz + q_offset[1]) / 10  # along y vertical up
        qy = (qy + q_offset[0]) / 10  # along x outboard

        if debugging:
            gu.multislices_plot(abs(ortho_obj), sum_frames=True, scale=scale, plot_colorbar=True, width_z=width_z,
                                width_y=width_y, width_x=width_x, is_orthogonal=True, reciprocal_space=True, vmin=0,
                                title=title+' in the orthogonal laboratory frame')
        return ortho_obj, (qx, qz, qy)

    def orthogonalize_vector(self, vector, array_shape, tilt_angle, pixel_x, pixel_y, verbose=False):
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
        valid.valid_container(array_shape, container_types=(tuple, list), length=3, item_types=int,
                              min_excluded=0, name='Setup.orthogonalize_vector')

        ortho_matrix = self.transformation_matrix(array_shape=array_shape, tilt_angle=tilt_angle,
                                                  pixel_x=pixel_x, pixel_y=pixel_y, verbose=verbose)
        # ortho_matrix is the transformation matrix from the detector coordinates to the laboratory frame
        # Here, we want to calculate the coordinates that would have a vector of the laboratory frame expressed in the
        # detector frame, i.e. one has to inverse the transformation matrix.
        ortho_imatrix = np.linalg.inv(ortho_matrix)
        new_x = ortho_imatrix[0, 0] * vector[2] + ortho_imatrix[0, 1] * vector[1] + ortho_imatrix[0, 2] * vector[0]
        new_y = ortho_imatrix[1, 0] * vector[2] + ortho_imatrix[1, 1] * vector[1] + ortho_imatrix[1, 2] * vector[0]
        new_z = ortho_imatrix[2, 0] * vector[2] + ortho_imatrix[2, 1] * vector[1] + ortho_imatrix[2, 2] * vector[0]
        return new_z, new_y, new_x

    def transformation_matrix(self, array_shape, tilt_angle, pixel_x, pixel_y, direct_space=True, verbose=True):
        """
        Calculate the pixel non-orthogonal coordinates in the orthogonal reference frame.

        :param array_shape: shape of the 3D array to orthogonalize
        :param tilt_angle: angular step during the rocking curve, in degrees
        :param pixel_x: horizontal pixel size, in meters
        :param pixel_y: vertical pixel size, in meters
        :param direct_space: True in order to return the transformation matrix in direct space
        :param verbose: True to have printed comments
        :return: the transformation matrix from the detector frame to the laboratory frame, and the q offset
         (3D vector) if direct_space is False. For direct space, the length scale is in nm, for reciprocal space,
         it is in 1/nm.
        """
        if verbose:
            print(f'out-of plane detector angle={self.outofplane_angle:.3f} deg,'
                  f' inplane_angle={self.inplane_angle:.3f} deg')
        wavelength = self.wavelength * 1e9  # convert to nm
        distance = self.distance * 1e9  # convert to nm
        pixel_x = pixel_x * 1e9  # convert to nm
        pixel_y = pixel_y * 1e9  # convert to nm
        outofplane = np.radians(self.outofplane_angle)
        inplane = np.radians(self.inplane_angle)
        grazing_angle = [np.radians(val) for val in self.grazing_angle]
        lambdaz = wavelength * distance
        mymatrix = np.zeros((3, 3))
        tilt = np.radians(tilt_angle)
        q_offset = np.zeros(3)  # TODO: calculate the q offset for all geometries
        nbz, nby, nbx = array_shape

        if self.beamline == 'ID01':
            if verbose:
                print('using ESRF ID01 PSIC geometry')
            if not isclose(grazing_angle[0], 0, rel_tol=1e-09, abs_tol=1e-09):
                raise NotImplementedError('Non-zero chi is not implemented for ID01')
            if self.rocking_angle == "outofplane" and isclose(grazing_angle[0], 0, rel_tol=1e-09, abs_tol=1e-09):
                if verbose:
                    print('rocking angle is eta')
                # rocking eta angle clockwise around x (phi does not matter, above eta)
                mymatrix[:, 0] = 2 * np.pi / lambdaz *\
                    np.array([pixel_x * np.cos(inplane),
                              0,
                              pixel_x * np.sin(inplane)])
                mymatrix[:, 1] = 2 * np.pi / lambdaz *\
                    np.array([-pixel_y * np.sin(inplane) * np.sin(outofplane),
                              -pixel_y * np.cos(outofplane),
                              pixel_y * np.cos(inplane) * np.sin(outofplane)])
                mymatrix[:, 2] = 2 * np.pi / lambdaz *\
                    np.array([0,
                              tilt * distance * (1 - np.cos(inplane) * np.cos(outofplane)),
                              tilt * distance * np.sin(outofplane)])
                q_offset[0] = -2 * np.pi / lambdaz * distance * np.cos(outofplane) * np.sin(inplane)
                q_offset[1] = 2 * np.pi / lambdaz * distance * np.sin(outofplane)
                q_offset[2] = 2 * np.pi / lambdaz * distance * (np.cos(inplane) * np.cos(outofplane) - 1)

            elif self.rocking_angle == "inplane":
                if verbose:
                    print(f'rocking angle is phi, eta={grazing_angle[1]*180/np.pi:.3f} deg')
                # rocking phi angle clockwise around y, incident angle eta is non zero (eta below phi)
                mymatrix[:, 0] = 2 * np.pi / lambdaz *\
                    np.array([pixel_x * np.cos(inplane),
                              0,
                              pixel_x * np.sin(inplane)])
                mymatrix[:, 1] = 2 * np.pi / lambdaz *\
                    np.array([-pixel_y * np.sin(inplane) * np.sin(outofplane),
                              -pixel_y * np.cos(outofplane),
                              pixel_y * np.cos(inplane) * np.sin(outofplane)])
                mymatrix[:, 2] = 2 * np.pi / lambdaz * tilt * distance * \
                    np.array([(np.sin(grazing_angle[1]) * np.sin(outofplane) +
                             np.cos(grazing_angle[1]) * (np.cos(inplane) * np.cos(outofplane) - 1)),
                             np.sin(grazing_angle[1]) * np.sin(inplane) * np.cos(outofplane),
                             np.cos(grazing_angle[1]) * np.sin(inplane) * np.cos(outofplane)])
                q_offset[0] = -2 * np.pi / lambdaz * distance * np.cos(outofplane) * np.sin(inplane)
                q_offset[1] = 2 * np.pi / lambdaz * distance * np.sin(outofplane)
                q_offset[2] = 2 * np.pi / lambdaz * distance * (np.cos(inplane) * np.cos(outofplane) - 1)

        if self.beamline == 'P10':
            if verbose:
                print('using PETRAIII P10 geometry')
            if self.rocking_angle == "outofplane":
                if verbose:
                    print(f'rocking angle is omega, chi={grazing_angle[0]*180/np.pi:.3f} deg')
                # rocking omega angle clockwise around x at mu=0, chi potentially non zero (chi below omega)
                # (phi does not matter, above eta)
                mymatrix[:, 0] = 2 * np.pi / lambdaz *\
                    np.array([-pixel_x * np.cos(inplane),
                              0,
                              pixel_x * np.sin(inplane)])
                mymatrix[:, 1] = 2 * np.pi / lambdaz *\
                    np.array([pixel_y * np.sin(inplane) * np.sin(outofplane),
                              -pixel_y * np.cos(outofplane),
                              pixel_y * np.cos(inplane) * np.sin(outofplane)])
                mymatrix[:, 2] = 2 * np.pi / lambdaz *\
                    np.array([tilt * distance * np.sin(grazing_angle[0]) * (np.cos(inplane) * np.cos(outofplane) - 1),
                              tilt * distance * np.cos(grazing_angle[0]) * (1 - np.cos(inplane) * np.cos(outofplane)),
                              tilt * distance * (np.sin(outofplane) * np.cos(grazing_angle[0]) -
                                                 np.cos(outofplane) * np.sin(inplane) * np.sin(grazing_angle[0]))])
                q_offset[0] = 2 * np.pi / lambdaz * distance * np.cos(outofplane) * np.sin(inplane)
                q_offset[1] = 2 * np.pi / lambdaz * distance * np.sin(outofplane)
                q_offset[2] = 2 * np.pi / lambdaz * distance * (np.cos(inplane) * np.cos(outofplane) - 1)

            elif self.rocking_angle == "inplane":
                if verbose:
                    print(f'rocking angle is phi, omega={grazing_angle[1]*180/np.pi:.3f} deg,'
                          f' chi={grazing_angle[0]*180/np.pi:.3f} deg')

                # rocking phi angle clockwise around y, omega and chi potentially non zero (chi below omega below phi)
                mymatrix[:, 0] = 2 * np.pi / lambdaz *\
                    np.array([-pixel_x * np.cos(inplane),
                              0,
                              pixel_x * np.sin(inplane)])
                mymatrix[:, 1] = 2 * np.pi / lambdaz *\
                    np.array([pixel_y * np.sin(inplane) * np.sin(outofplane),
                              -pixel_y * np.cos(outofplane),
                              pixel_y * np.cos(inplane) * np.sin(outofplane)])
                mymatrix[:, 2] = 2 * np.pi / lambdaz * tilt * distance * \
                    np.array([(np.sin(grazing_angle[1]) * np.sin(outofplane) +
                              np.cos(grazing_angle[0])*np.cos(grazing_angle[1])*(np.cos(inplane)*np.cos(outofplane)-1)),
                              (-np.sin(grazing_angle[1]) * np.sin(inplane) * np.cos(outofplane) +
                              np.sin(grazing_angle[0])*np.cos(grazing_angle[1])*(np.cos(inplane)*np.cos(outofplane)-1)),
                              (-np.cos(grazing_angle[0])*np.cos(grazing_angle[1])*np.sin(inplane)*np.cos(outofplane) -
                               np.sin(grazing_angle[0])*np.cos(grazing_angle[1])*np.sin(outofplane))])
                q_offset[0] = 2 * np.pi / lambdaz * distance * np.cos(outofplane) * np.sin(inplane)
                q_offset[1] = 2 * np.pi / lambdaz * distance * np.sin(outofplane)
                q_offset[2] = 2 * np.pi / lambdaz * distance * (np.cos(inplane) * np.cos(outofplane) - 1)

        if self.beamline == 'NANOMAX':
            if verbose:
                print('using NANOMAX geometry')
            if not isclose(grazing_angle[0], 0, rel_tol=1e-09, abs_tol=1e-09):
                raise NotImplementedError('Non-zero chi is not implemented for NANOMAX')
            if self.rocking_angle == "outofplane" and isclose(grazing_angle[0], 0, rel_tol=1e-09, abs_tol=1e-09):
                if verbose:
                    print('rocking angle is theta')
                # rocking theta angle clockwise around x (phi does not matter, above eta)
                mymatrix[:, 0] = 2 * np.pi / lambdaz *\
                    np.array([pixel_x * np.cos(inplane),
                              0,
                              pixel_x * np.sin(inplane)])
                mymatrix[:, 1] = 2 * np.pi / lambdaz *\
                    np.array([-pixel_y * np.sin(inplane) * np.sin(outofplane),
                              -pixel_y * np.cos(outofplane),
                              pixel_y * np.cos(inplane) * np.sin(outofplane)])
                mymatrix[:, 2] = 2 * np.pi / lambdaz *\
                    np.array([0,
                              tilt * distance * (1 - np.cos(inplane) * np.cos(outofplane)),
                              tilt * distance * np.sin(outofplane)])
                q_offset[0] = -2 * np.pi / lambdaz * distance * np.cos(outofplane) * np.sin(inplane)
                q_offset[1] = 2 * np.pi / lambdaz * distance * np.sin(outofplane)
                q_offset[2] = 2 * np.pi / lambdaz * distance * (np.cos(inplane) * np.cos(outofplane) - 1)

            elif self.rocking_angle == "inplane":
                if verbose:
                    print(f'rocking angle is phi, theta={grazing_angle[1]*180/np.pi:.3f} deg')
                # rocking phi angle clockwise around y, incident angle theta is non zero (theta below phi)
                mymatrix[:, 0] = 2 * np.pi / lambdaz *\
                    np.array([pixel_x * np.cos(inplane),
                              0,
                              pixel_x * np.sin(inplane)])
                mymatrix[:, 1] = 2 * np.pi / lambdaz *\
                    np.array([-pixel_y * np.sin(inplane) * np.sin(outofplane),
                              -pixel_y * np.cos(outofplane),
                              pixel_y * np.cos(inplane) * np.sin(outofplane)])
                mymatrix[:, 2] = 2 * np.pi / lambdaz * tilt * distance * \
                    np.array([(np.sin(grazing_angle[1]) * np.sin(outofplane) +
                               np.cos(grazing_angle[1]) * (np.cos(inplane) * np.cos(outofplane) - 1)),
                              np.sin(grazing_angle[1]) * np.sin(inplane) * np.cos(outofplane),
                              np.cos(grazing_angle[1]) * np.sin(inplane) * np.cos(outofplane)])
                q_offset[0] = -2 * np.pi / lambdaz * distance * np.cos(outofplane) * np.sin(inplane)
                q_offset[1] = 2 * np.pi / lambdaz * distance * np.sin(outofplane)
                q_offset[2] = 2 * np.pi / lambdaz * distance * (np.cos(inplane) * np.cos(outofplane) - 1)

        if self.beamline == '34ID':
            if verbose:
                print('using APS 34ID geometry')
            if self.rocking_angle == "outofplane":
                if verbose:
                    print(f'rocking angle is phi, chi={grazing_angle[0] * 180 / np.pi:.3f} deg')
                # rocking phi angle anti-clockwise around x (theta does not matter, above phi)
                mymatrix[:, 0] = 2 * np.pi / lambdaz *\
                    np.array([-pixel_x * np.cos(inplane),
                              0,
                              pixel_x * np.sin(inplane)])
                mymatrix[:, 1] = 2 * np.pi / lambdaz *\
                    np.array([pixel_y * np.sin(inplane) * np.sin(outofplane),
                              -pixel_y * np.cos(outofplane),
                              pixel_y * np.cos(inplane) * np.sin(outofplane)])
                mymatrix[:, 2] = 2 * np.pi / lambdaz *\
                    np.array([tilt * distance * np.sin(grazing_angle[0]) * (np.cos(inplane) * np.cos(outofplane) - 1),
                              tilt * distance * np.cos(grazing_angle[0]) * (np.cos(inplane) * np.cos(outofplane) - 1),
                              -tilt * distance * (np.sin(outofplane) * np.cos(grazing_angle[0]) +
                                                  np.cos(outofplane) * np.sin(inplane) * np.sin(grazing_angle[0]))])
                q_offset[0] = 2 * np.pi / lambdaz * distance * np.cos(outofplane) * np.sin(inplane)
                q_offset[1] = 2 * np.pi / lambdaz * distance * np.sin(outofplane)
                q_offset[2] = 2 * np.pi / lambdaz * distance * (np.cos(inplane) * np.cos(outofplane) - 1)

            elif self.rocking_angle == "inplane":
                if verbose:
                    print(f'rocking angle is theta, phi={grazing_angle[1]*180/np.pi:.3f} deg,'
                          f' chi={grazing_angle[0]*180/np.pi:.3f} deg')
                # rocking theta angle anti-clockwise around y, incident angle is non zero (theta is above phi)
                mymatrix[:, 0] = 2 * np.pi / lambdaz * \
                    np.array([-pixel_x * np.cos(inplane),
                              0,
                              pixel_x * np.sin(inplane)])
                mymatrix[:, 1] = 2 * np.pi / lambdaz * \
                    np.array([pixel_y * np.sin(inplane) * np.sin(outofplane),
                              -pixel_y * np.cos(outofplane),
                              pixel_y * np.cos(inplane) * np.sin(outofplane)])
                mymatrix[:, 2] = 2 * np.pi / lambdaz * tilt * distance * \
                    np.array([(np.sin(grazing_angle[1]) * np.sin(outofplane) -
                              np.cos(grazing_angle[0])*np.cos(grazing_angle[1])*(np.cos(inplane)*np.cos(outofplane)-1)),
                              (-np.sin(grazing_angle[1]) * np.sin(inplane) * np.cos(outofplane) +
                              np.sin(grazing_angle[0])*np.cos(grazing_angle[1])*(np.cos(inplane)*np.cos(outofplane)-1)),
                              (np.cos(grazing_angle[0])*np.cos(grazing_angle[1])*np.sin(inplane)*np.cos(outofplane) -
                               np.sin(grazing_angle[0])*np.cos(grazing_angle[1])*np.sin(outofplane))])
                q_offset[0] = 2 * np.pi / lambdaz * distance * np.cos(outofplane) * np.sin(inplane)
                q_offset[1] = 2 * np.pi / lambdaz * distance * np.sin(outofplane)
                q_offset[2] = 2 * np.pi / lambdaz * distance * (np.cos(inplane) * np.cos(outofplane) - 1)

        if self.beamline == 'SIXS_2018' or self.beamline == 'SIXS_2019':
            if verbose:
                print('using SIXS geometry')
            if not isclose(grazing_angle[0], 0, rel_tol=1e-09, abs_tol=1e-09):
                raise NotImplementedError('Non-zero chi is not implemented for SIXS')
            if self.rocking_angle == "inplane":
                if verbose:
                    print(f'rocking angle is mu, beta={grazing_angle[1] * 180 / np.pi:.3f} deg')

                # rocking mu angle anti-clockwise around y
                mymatrix[:, 0] = 2 * np.pi / lambdaz * pixel_x *\
                    np.array([np.cos(inplane),
                              -np.sin(grazing_angle[1]) * np.sin(inplane),
                              -np.cos(grazing_angle[1]) * np.sin(inplane)])
                mymatrix[:, 1] = 2 * np.pi / lambdaz * pixel_y *\
                    np.array([np.sin(inplane) * np.sin(outofplane),
                              (np.sin(grazing_angle[1]) * np.cos(inplane) * np.sin(outofplane)
                               - np.cos(grazing_angle[1]) * np.cos(outofplane)),
                              (np.cos(grazing_angle[1]) * np.cos(inplane) * np.sin(outofplane)
                               + np.sin(grazing_angle[1]) * np.cos(outofplane))])
                mymatrix[:, 2] = 2 * np.pi / lambdaz * tilt * distance *\
                    np.array([np.cos(grazing_angle[1]) - np.cos(inplane) * np.cos(outofplane),
                              np.sin(grazing_angle[1]) * np.sin(inplane) * np.cos(outofplane),
                              np.cos(grazing_angle[1]) * np.sin(inplane) * np.cos(outofplane)])
                q_offset[0] = 2 * np.pi / lambdaz * distance * np.cos(outofplane) * np.sin(inplane)
                q_offset[1] = 2 * np.pi / lambdaz * distance *\
                    (np.cos(grazing_angle[1]) * np.sin(outofplane) +
                     np.sin(grazing_angle[1]) * np.cos(inplane) * np.cos(outofplane))
                q_offset[2] = 2 * np.pi / lambdaz * distance *\
                    (np.cos(grazing_angle[1]) * np.cos(inplane) * np.cos(outofplane) -
                     np.sin(grazing_angle[1]) * np.sin(outofplane) - 1)
            else:
                raise NotImplementedError('out of plane rocking curve not implemented for SIXS')

        if self.beamline == 'CRISTAL':
            if verbose:
                print('using CRISTAL geometry')
            if not isclose(grazing_angle[0], 0, rel_tol=1e-09, abs_tol=1e-09):
                raise NotImplementedError('Non-zero chi is not implemented for CRISTAL')
            if self.rocking_angle == "outofplane":
                if verbose:
                    print('rocking angle is mgomega')
                # rocking mgomega angle clockwise around x
                mymatrix[:, 0] = 2 * np.pi / lambdaz *\
                    np.array([pixel_x * np.cos(inplane),
                              0,
                              -pixel_x * np.sin(inplane)])
                mymatrix[:, 1] = 2 * np.pi / lambdaz *\
                    np.array([pixel_y * np.sin(inplane) * np.sin(outofplane),
                              -pixel_y * np.cos(outofplane),
                              pixel_y * np.cos(inplane) * np.sin(outofplane)])
                mymatrix[:, 2] = 2 * np.pi / lambdaz *\
                    np.array([0,
                              tilt * distance * (1 - np.cos(inplane) * np.cos(outofplane)),
                              tilt * distance * np.sin(outofplane)])
                q_offset[0] = 2 * np.pi / lambdaz * distance * np.cos(outofplane) * np.sin(inplane)
                q_offset[1] = 2 * np.pi / lambdaz * distance * np.sin(outofplane)
                q_offset[2] = 2 * np.pi / lambdaz * distance * (np.cos(inplane) * np.cos(outofplane) - 1)
            elif self.rocking_angle == "inplane":
                if verbose:
                    print(f'rocking angle is phi, mgomega={grazing_angle[1]*180/np.pi:.3f} deg')
                # rocking phi angle anti-clockwise around y, incident angle mgomega is non zero (mgomega below phi)
                mymatrix[:, 0] = 2 * np.pi / lambdaz *\
                    np.array([pixel_x * np.cos(inplane),
                              0,
                              -pixel_x * np.sin(inplane)])
                mymatrix[:, 1] = 2 * np.pi / lambdaz *\
                    np.array([pixel_y * np.sin(inplane) * np.sin(outofplane),
                              -pixel_y * np.cos(outofplane),
                              pixel_y * np.cos(inplane) * np.sin(outofplane)])
                mymatrix[:, 2] = 2 * np.pi / lambdaz * tilt * distance * \
                    np.array([(-np.sin(grazing_angle[1]) * np.sin(outofplane) -
                               np.cos(grazing_angle[1]) * (np.cos(inplane) * np.cos(outofplane) - 1)),
                              np.sin(grazing_angle[1]) * np.sin(inplane) * np.cos(outofplane),
                              np.cos(grazing_angle[1]) * np.sin(inplane) * np.cos(outofplane)])
                q_offset[0] = 2 * np.pi / lambdaz * distance * np.cos(outofplane) * np.sin(inplane)
                q_offset[1] = 2 * np.pi / lambdaz * distance * np.sin(outofplane)
                q_offset[2] = 2 * np.pi / lambdaz * distance * (np.cos(inplane) * np.cos(outofplane) - 1)

        if direct_space:  # length scale in nm
            # for a discrete FT, the dimensions of the basis vectors after the transformation are related to the total
            # domain size
            mymatrix[:, 0] = nbx * mymatrix[:, 0]
            mymatrix[:, 1] = nby * mymatrix[:, 1]
            mymatrix[:, 2] = nbz * mymatrix[:, 2]
            return 2 * np.pi * np.linalg.inv(mymatrix).transpose()
        else:
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
        valid.valid_container(array_shape, container_types=(tuple, list), length=3, item_types=int,
                              min_excluded=0, name='Setup.voxel_sizes')

        transfer_matrix = self.transformation_matrix(array_shape=array_shape, tilt_angle=tilt_angle,
                                                     direct_space=True, pixel_x=pixel_x, pixel_y=pixel_y,
                                                     verbose=verbose)
        # transfer_matrix is the transformation matrix of the direct space coordinates (its columns are the
        # non-orthogonal basis vectors reciprocal to the detector frame)
        # the spacing in the laboratory frame is therefore given by the rows of the matrix
        dx = np.linalg.norm(transfer_matrix[0, :])  # along x outboard
        dy = np.linalg.norm(transfer_matrix[1, :])  # along y vertical up
        dz = np.linalg.norm(transfer_matrix[2, :])  # along z downstream

        if verbose:
            print(f'Direct space voxel size (z, y, x) = ({dz:.2f}, {dy:.2f}, {dx:.2f}) (nm)')
        return dz, dy, dx

    def voxel_sizes_detector(self, array_shape, tilt_angle, pixel_x, pixel_y, verbose=False):
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
        voxel_z = self.wavelength / (array_shape[0] * abs(tilt_angle) * np.pi / 180) * 1e9  # in nm
        voxel_y = self.wavelength * self.distance / (array_shape[1] * pixel_y) * 1e9  # in nm
        voxel_x = self.wavelength * self.distance / (array_shape[2] * pixel_x) * 1e9  # in nm
        if verbose:
            print('voxelsize_z, voxelsize_y, voxelsize_x='
                  '({0:.2f}, {1:.2f}, {2:.2f}) (1/nm)'.format(voxel_z, voxel_y, voxel_x))
        return voxel_z, voxel_y, voxel_x


class SetupPostprocessing(object):
    """
    Class to handle the experimental geometry for postprocessing.
    """
    def __init__(self, beamline, energy, outofplane_angle, inplane_angle, tilt_angle, rocking_angle, distance,
                 grazing_angle=0, pixel_x=55e-6, pixel_y=55e-6):
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
        if beamline in {'ID01', 'SIXS_2018', 'SIXS_2019', 'CRISTAL', 'NANOMAX'}:
            # we look at the detector from downstream, detector X along the outboard direction
            self.detector_hor = 'y+'
        else:  # 'P10', '34ID'
            # we look at the detector from upstream, detector X opposite to the outboard direction
            self.detector_hor = 'y-'

        # vertical axis:
        # origin is at the top, detector Y along vertical down
        self.detector_ver = 'z-'

    def __repr__(self):
        """
        :return: a nicely formatted representation string
        """
        return f"{self.__class__.__name__}: beamline={self.beamline}, energy={self.energy}eV," \
               f" sample to detector distance={self.distance}m, pixel size (VxH)=({self.pixel_y},{self.pixel_x})"

    def detector_frame(self, obj, voxelsize, width_z=None, width_y=None, width_x=None,
                       debugging=False, **kwargs):
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
        title = kwargs.get('title', 'Object')

        for k in kwargs.keys():
            if k not in {'title'}:
                raise Exception("unknown keyword argument given:", k)

        nbz, nby, nbx = obj.shape

        if debugging:
            gu.multislices_plot(abs(obj), sum_frames=True, width_z=width_z, width_y=width_y, width_x=width_x,
                                title=title + ' before interpolation\n')

        ortho_matrix = self.update_coords(array_shape=(nbz, nby, nbx), tilt_angle=self.tilt_angle,
                                          pixel_x=self.pixel_x, pixel_y=self.pixel_y)

        ################################################
        # interpolate the data into the detector frame #
        ################################################
        myz, myy, myx = np.meshgrid(np.arange(-nbz // 2, nbz // 2, 1),
                                    np.arange(-nby // 2, nby // 2, 1),
                                    np.arange(-nbx // 2, nbx // 2, 1), indexing='ij')

        new_x = ortho_matrix[0, 0] * myx + ortho_matrix[0, 1] * myy + ortho_matrix[0, 2] * myz
        new_y = ortho_matrix[1, 0] * myx + ortho_matrix[1, 1] * myy + ortho_matrix[1, 2] * myz
        new_z = ortho_matrix[2, 0] * myx + ortho_matrix[2, 1] * myy + ortho_matrix[2, 2] * myz
        del myx, myy, myz
        # la partie rgi est sure: c'est la taille de l'objet orthogonal de depart
        rgi = RegularGridInterpolator((np.arange(-nbz // 2, nbz // 2) * voxelsize,
                                       np.arange(-nby // 2, nby // 2) * voxelsize,
                                       np.arange(-nbx // 2, nbx // 2) * voxelsize),
                                      obj, method='linear', bounds_error=False, fill_value=0)
        detector_obj = rgi(np.concatenate((new_z.reshape((1, new_z.size)), new_y.reshape((1, new_z.size)),
                                           new_x.reshape((1, new_z.size)))).transpose())
        detector_obj = detector_obj.reshape((nbz, nby, nbx)).astype(obj.dtype)

        if debugging:
            gu.multislices_plot(abs(detector_obj), sum_frames=True, width_z=width_z, width_y=width_y, width_x=width_x,
                                title=title + ' interpolated in detector frame\n')

        return detector_obj

    def exit_wavevector(self):
        """
        Calculate the exit wavevector kout depending on the setup parameters, in laboratory frame (z downstream,
         y vertical, x outboard).

        :return: kout vector
        """
        if self.beamline == 'SIXS_2018' or self.beamline == 'SIXS_2019':
            # gamma is anti-clockwise
            kout = 2 * np.pi / self.wavelength * np.array(
                [np.cos(np.pi * self.inplane_angle / 180) * np.cos(np.pi * self.outofplane_angle / 180),  # z
                 np.sin(np.pi * self.outofplane_angle / 180),  # y
                 np.sin(np.pi * self.inplane_angle / 180) * np.cos(np.pi * self.outofplane_angle / 180)])  # x
        elif self.beamline == 'ID01':
            # nu is clockwise
            kout = 2 * np.pi / self.wavelength * np.array(
                [np.cos(np.pi * self.inplane_angle / 180) * np.cos(np.pi * self.outofplane_angle / 180),  # z
                 np.sin(np.pi * self.outofplane_angle / 180),  # y
                 -np.sin(np.pi * self.inplane_angle / 180) * np.cos(np.pi * self.outofplane_angle / 180)])  # x
        elif self.beamline == '34ID':
            # gamma is anti-clockwise
            kout = 2 * np.pi / self.wavelength * np.array(
                [np.cos(np.pi * self.inplane_angle / 180) * np.cos(np.pi * self.outofplane_angle / 180),  # z
                 np.sin(np.pi * self.outofplane_angle / 180),  # y
                 np.sin(np.pi * self.inplane_angle / 180) * np.cos(np.pi * self.outofplane_angle / 180)])  # x
        elif self.beamline == 'NANOMAX':
            # gamma is clockwise
            kout = 2 * np.pi / self.wavelength * np.array(
                [np.cos(np.pi * self.inplane_angle / 180) * np.cos(np.pi * self.outofplane_angle / 180),  # z
                 np.sin(np.pi * self.outofplane_angle / 180),  # y
                 -np.sin(np.pi * self.inplane_angle / 180) * np.cos(np.pi * self.outofplane_angle / 180)])  # x
        elif self.beamline == 'P10':
            # gamma is anti-clockwise
            kout = 2 * np.pi / self.wavelength * np.array(
                [np.cos(np.pi * self.inplane_angle / 180) * np.cos(np.pi * self.outofplane_angle / 180),  # z
                 np.sin(np.pi * self.outofplane_angle / 180),  # y
                 np.sin(np.pi * self.inplane_angle / 180) * np.cos(np.pi * self.outofplane_angle / 180)])  # x
        elif self.beamline == 'CRISTAL':
            # gamma is anti-clockwise
            kout = 2 * np.pi / self.wavelength * np.array(
                [np.cos(np.pi * self.inplane_angle / 180) * np.cos(np.pi * self.outofplane_angle / 180),  # z
                 np.sin(np.pi * self.outofplane_angle / 180),  # y
                 np.sin(np.pi * self.inplane_angle / 180) * np.cos(np.pi * self.outofplane_angle / 180)])  # x
        else:
            raise ValueError('setup parameter: ', self.beamline, 'not defined')
        return kout

    def inplane_coeff(self):
        """
        Define a coefficient +/- 1 depending on the detector inplane rotation direction and the detector inplane
         orientation. The frame convention is the one of xrayutilities: x downstream, y outboard, z vertical up.
         See postprocessing/scripts/correct_angles_detector.py for an example.

        :return: +1 or -1
        """
        if self.detector_hor == 'y+':
            hor_coeff = 1
        else:  # 'y-'
            hor_coeff = -1

        if self.beamline == 'SIXS_2018' or self.beamline == 'SIXS_2019':
            # gamma is anti-clockwise, we see the detector from downstream
            coeff_inplane = 1 * hor_coeff
        elif self.beamline == 'ID01':
            # nu is clockwise, we see the detector from downstream
            coeff_inplane = -1 * hor_coeff
        elif self.beamline == '34ID':
            # delta is anti-clockwise, we see the detector from the front
            coeff_inplane = 1 * hor_coeff
        elif self.beamline == 'P10':
            # gamma is anti-clockwise, we see the detector from the front
            coeff_inplane = 1 * hor_coeff
        elif self.beamline == 'CRISTAL':
            # gamma is anti-clockwise, we see the detector from downstream
            coeff_inplane = 1 * hor_coeff
        elif self.beamline == 'NANOMAX':
            # gamma is clockwise, we see the detector from downstream
            coeff_inplane = -1 * hor_coeff
        else:
            raise ValueError('setup parameter: ', self.beamline, 'not defined')
        return coeff_inplane

    def orthogonalize(self, obj, initial_shape=(), voxel_size=np.nan, width_z=None, width_y=None,
                      width_x=None, verbose=True, debugging=False, **kwargs):
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
        title = kwargs.get('title', 'Object')

        for k in kwargs.keys():
            if k not in {'title'}:
                raise Exception("unknown keyword argument given:", k)

        if len(initial_shape) == 0:
            initial_shape = obj.shape

        if debugging:
            gu.multislices_plot(abs(obj), sum_frames=True, width_z=width_z, width_y=width_y, width_x=width_x,
                                title=title+' in detector frame')

        # estimate the direct space voxel sizes in nm based on the FFT window shape used in phase retrieval
        dz_realspace, dy_realspace, dx_realspace = self.voxel_sizes(initial_shape, tilt_angle=abs(self.tilt_angle),
                                                                    pixel_x=self.pixel_x, pixel_y=self.pixel_y)

        if verbose:
            print('Direct space voxel sizes (z, y, x) based on initial FFT shape: (',
                  str('{:.2f}'.format(dz_realspace)), 'nm,',
                  str('{:.2f}'.format(dy_realspace)), 'nm,',
                  str('{:.2f}'.format(dx_realspace)), 'nm )')

        nbz, nby, nbx = obj.shape  # could be smaller if the object was cropped around the support
        if nbz != initial_shape[0] or nby != initial_shape[1] or nbx != initial_shape[2]:
            # recalculate the tilt and pixel sizes to accomodate a shape change
            tilt = self.tilt_angle * initial_shape[0] / nbz
            pixel_y = self.pixel_y * initial_shape[1] / nby
            pixel_x = self.pixel_x * initial_shape[2] / nbx
            if verbose:
                print('Tilt, pixel_y, pixel_x based on cropped array shape: (',
                      str('{:.4f}'.format(tilt)), 'deg,',
                      str('{:.2f}'.format(pixel_y * 1e6)), 'um,',
                      str('{:.2f}'.format(pixel_x * 1e6)), 'um)')

            # sanity check, the direct space voxel sizes calculated below should be equal to the original ones
            dz_realspace, dy_realspace, dx_realspace = self.voxel_sizes((nbz, nby, nbx),
                                                                        tilt_angle=abs(tilt),
                                                                        pixel_x=pixel_x, pixel_y=pixel_y)
            if verbose:
                print('Sanity check, recalculated direct space voxel sizes: (',
                      str('{:.2f}'.format(dz_realspace)), ' nm,',
                      str('{:.2f}'.format(dy_realspace)), 'nm,',
                      str('{:.2f}'.format(dx_realspace)), 'nm )')
        else:
            tilt = self.tilt_angle
            pixel_y = self.pixel_y
            pixel_x = self.pixel_x

        if np.isnan(voxel_size):
            voxel = np.mean([dz_realspace, dy_realspace, dx_realspace])  # in nm
        else:
            voxel = voxel_size

        ortho_matrix = self.update_coords(array_shape=(nbz, nby, nbx), tilt_angle=tilt,
                                          pixel_x=pixel_x, pixel_y=pixel_y, verbose=verbose)

        ###############################################################
        # Vincent Favre-Nicolin's method using inverse transformation #
        ###############################################################
        myz, myy, myx = np.meshgrid(np.arange(-nbz // 2, nbz // 2, 1) * voxel,
                                    np.arange(-nby // 2, nby // 2, 1) * voxel,
                                    np.arange(-nbx // 2, nbx // 2, 1) * voxel, indexing='ij')

        # ortho_matrix is the transformation matrix from the detector coordinates to the laboratory frame
        # in RGI, we want to calculate the coordinates that would have a grid of the laboratory frame expressed in the
        # detector frame, i.e. one has to inverse the transformation matrix.
        ortho_imatrix = np.linalg.inv(ortho_matrix)
        new_x = ortho_imatrix[0, 0] * myx + ortho_imatrix[0, 1] * myy + ortho_imatrix[0, 2] * myz
        new_y = ortho_imatrix[1, 0] * myx + ortho_imatrix[1, 1] * myy + ortho_imatrix[1, 2] * myz
        new_z = ortho_imatrix[2, 0] * myx + ortho_imatrix[2, 1] * myy + ortho_imatrix[2, 2] * myz
        del myx, myy, myz
        gc.collect()

        rgi = RegularGridInterpolator((np.arange(-nbz // 2, nbz // 2), np.arange(-nby // 2, nby // 2),
                                       np.arange(-nbx // 2, nbx // 2)), obj, method='linear',
                                      bounds_error=False, fill_value=0)
        ortho_obj = rgi(np.concatenate((new_z.reshape((1, new_z.size)), new_y.reshape((1, new_z.size)),
                                        new_x.reshape((1, new_z.size)))).transpose())
        ortho_obj = ortho_obj.reshape((nbz, nby, nbx)).astype(obj.dtype)

        if debugging:
            gu.multislices_plot(abs(ortho_obj), sum_frames=True, width_z=width_z, width_y=width_y, width_x=width_x,
                                title=title+' in the orthogonal laboratory frame')
        return ortho_obj, voxel

    def orthogonalize_vector(self, vector, array_shape, tilt_angle, pixel_x, pixel_y, verbose=False):
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
        ortho_matrix = self.update_coords(array_shape=array_shape, tilt_angle=tilt_angle,
                                          pixel_x=pixel_x, pixel_y=pixel_y, verbose=verbose)
        # ortho_matrix is the transformation matrix from the detector coordinates to the laboratory frame
        # Here, we want to calculate the coordinates that would have a vector of the laboratory frame expressed in the
        # detector frame, i.e. one has to inverse the transformation matrix.
        ortho_imatrix = np.linalg.inv(ortho_matrix)
        new_x = ortho_imatrix[0, 0] * vector[2] + ortho_imatrix[0, 1] * vector[1] + ortho_imatrix[0, 2] * vector[0]
        new_y = ortho_imatrix[1, 0] * vector[2] + ortho_imatrix[1, 1] * vector[1] + ortho_imatrix[1, 2] * vector[0]
        new_z = ortho_imatrix[2, 0] * vector[2] + ortho_imatrix[2, 1] * vector[1] + ortho_imatrix[2, 2] * vector[0]
        return new_z, new_y, new_x

    def outofplane_coeff(self):
        """
        Define a coefficient +/- 1 depending on the detector out of plane rotation direction and the detector out of
         plane orientation. The frame convention is the one of xrayutilities: x downstream, y outboard, z vertical up.
         See postprocessing/scripts/correct_angles_detector.py for an example.

        :return: +1 or -1
        """
        if self.detector_ver == 'z+':  # origin of pixels at the bottom
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

        if self.beamline == 'ID01':
            if verbose:
                print('using ESRF ID01 PSIC geometry')
            if self.rocking_angle == "outofplane":
                if verbose:
                    print('rocking angle is eta')
                # rocking eta angle clockwise around x (phi does not matter, above eta)
                mymatrix[:, 0] = 2 * np.pi * nbx / lambdaz * np.array([pixel_x * np.cos(inplane),
                                                                       0,
                                                                       pixel_x * np.sin(inplane)])
                mymatrix[:, 1] = 2 * np.pi * nby / lambdaz * np.array([-pixel_y * np.sin(inplane) * np.sin(outofplane),
                                                                       -pixel_y * np.cos(outofplane),
                                                                       pixel_y * np.cos(inplane) * np.sin(outofplane)])
                mymatrix[:, 2] = 2 * np.pi * nbz / lambdaz * np.array([0,
                                                                       tilt * distance * (1 - np.cos(inplane) * np.cos(
                                                                           outofplane)),
                                                                       tilt * distance * np.sin(outofplane)])
            elif self.rocking_angle == "inplane" and mygrazing_angle == 0:
                if verbose:
                    print('rocking angle is phi, eta=0')
                # rocking phi angle clockwise around y, assuming incident angle eta is zero (eta below phi)
                mymatrix[:, 0] = 2 * np.pi * nbx / lambdaz * np.array([pixel_x * np.cos(inplane),
                                                                       0,
                                                                       pixel_x * np.sin(inplane)])
                mymatrix[:, 1] = 2 * np.pi * nby / lambdaz * np.array([-pixel_y * np.sin(inplane) * np.sin(outofplane),
                                                                       -pixel_y * np.cos(outofplane),
                                                                       pixel_y * np.cos(inplane) * np.sin(outofplane)])
                mymatrix[:, 2] = 2 * np.pi * nbz / lambdaz * np.array(
                    [-tilt * distance * (1 - np.cos(inplane) * np.cos(outofplane)),
                     0,
                     tilt * distance * np.sin(inplane) * np.cos(outofplane)])
            elif self.rocking_angle == "inplane" and mygrazing_angle != 0:
                if verbose:
                    print('rocking angle is phi, with eta non zero')
                # rocking phi angle clockwise around y, incident angle eta is non zero (eta below phi)
                mymatrix[:, 0] = 2 * np.pi * nbx / lambdaz * np.array([pixel_x * np.cos(inplane),
                                                                       0,
                                                                       pixel_x * np.sin(inplane)])
                mymatrix[:, 1] = 2 * np.pi * nby / lambdaz * np.array([-pixel_y * np.sin(inplane) * np.sin(outofplane),
                                                                       -pixel_y * np.cos(outofplane),
                                                                       pixel_y * np.cos(inplane) * np.sin(outofplane)])
                mymatrix[:, 2] = 2 * np.pi * nbz / lambdaz * tilt * distance * \
                    np.array([(np.sin(mygrazing_angle) * np.sin(outofplane) +
                             np.cos(mygrazing_angle) * (np.cos(inplane) * np.cos(outofplane) - 1)),
                             np.sin(mygrazing_angle) * np.sin(inplane) * np.cos(outofplane),
                             np.cos(mygrazing_angle) * np.sin(inplane) * np.cos(outofplane)])
        if self.beamline == 'P10':
            if verbose:
                print('using PETRAIII P10 geometry')
            if self.rocking_angle == "outofplane":
                if verbose:
                    print('rocking angle is omega')
                # rocking omega angle clockwise around x at mu=0 (phi does not matter, above eta)
                mymatrix[:, 0] = 2 * np.pi * nbx / lambdaz * np.array([-pixel_x * np.cos(inplane),
                                                                       0,
                                                                       pixel_x * np.sin(inplane)])
                mymatrix[:, 1] = 2 * np.pi * nby / lambdaz * np.array([pixel_y * np.sin(inplane) * np.sin(outofplane),
                                                                       -pixel_y * np.cos(outofplane),
                                                                       pixel_y * np.cos(inplane) * np.sin(outofplane)])
                mymatrix[:, 2] = 2 * np.pi * nbz / lambdaz * np.array([0,
                                                                       tilt * distance * (1 - np.cos(inplane) * np.cos(
                                                                           outofplane)),
                                                                       tilt * distance * np.sin(outofplane)])
            elif self.rocking_angle == "inplane" and mygrazing_angle == 0:
                if verbose:
                    print('rocking angle is phi, omega=0')
                # rocking phi angle clockwise around y, incident angle omega is zero (omega below phi)
                mymatrix[:, 0] = 2 * np.pi * nbx / lambdaz * np.array([-pixel_x * np.cos(inplane),
                                                                       0,
                                                                       pixel_x * np.sin(inplane)])
                mymatrix[:, 1] = 2 * np.pi * nby / lambdaz * np.array([pixel_y * np.sin(inplane) * np.sin(outofplane),
                                                                       -pixel_y * np.cos(outofplane),
                                                                       pixel_y * np.cos(inplane) * np.sin(outofplane)])
                mymatrix[:, 2] = 2 * np.pi * nbz / lambdaz * np.array(
                    [tilt * distance * (np.cos(inplane) * np.cos(outofplane) - 1),
                     0,
                     - tilt * distance * np.sin(inplane) * np.cos(outofplane)])

            elif self.rocking_angle == "inplane" and mygrazing_angle != 0:
                if verbose:
                    print('rocking angle is phi, with omega non zero')
                # rocking phi angle clockwise around y, incident angle omega is non zero (omega below phi)
                mymatrix[:, 0] = 2 * np.pi * nbx / lambdaz * np.array([-pixel_x * np.cos(inplane),
                                                                       0,
                                                                       pixel_x * np.sin(inplane)])
                mymatrix[:, 1] = 2 * np.pi * nby / lambdaz * np.array([pixel_y * np.sin(inplane) * np.sin(outofplane),
                                                                       -pixel_y * np.cos(outofplane),
                                                                       pixel_y * np.cos(inplane) * np.sin(outofplane)])
                mymatrix[:, 2] = 2 * np.pi * nbz / lambdaz * tilt * distance * \
                    np.array([(np.sin(mygrazing_angle) * np.sin(outofplane) +
                             np.cos(mygrazing_angle) * (np.cos(inplane) * np.cos(outofplane) - 1)),
                             - np.sin(mygrazing_angle) * np.sin(inplane) * np.cos(outofplane),
                             - np.cos(mygrazing_angle) * np.sin(inplane) * np.cos(outofplane)])

        if self.beamline == 'NANOMAX':
            if verbose:
                print('using NANOMAX geometry')
            if self.rocking_angle == "outofplane":
                if verbose:
                    print('rocking angle is theta')
                # rocking eta angle clockwise around x (phi does not matter, above eta)
                mymatrix[:, 0] = 2 * np.pi * nbx / lambdaz * np.array([pixel_x * np.cos(inplane),
                                                                       0,
                                                                       pixel_x * np.sin(inplane)])
                mymatrix[:, 1] = 2 * np.pi * nby / lambdaz * np.array([pixel_y * np.sin(inplane) * np.sin(outofplane),
                                                                       pixel_y * np.cos(outofplane),
                                                                       -pixel_y * np.cos(inplane) * np.sin(outofplane)])
                mymatrix[:, 2] = 2 * np.pi * nbz / lambdaz * np.array([0,
                                                                       tilt * distance * (1 - np.cos(inplane) * np.cos(
                                                                           outofplane)),
                                                                       tilt * distance * np.sin(outofplane)])
            elif self.rocking_angle == "inplane" and mygrazing_angle == 0:
                if verbose:
                    print('rocking angle is phi, theta=0')
                # rocking phi angle clockwise around y, assuming incident angle eta is zero (eta below phi)
                mymatrix[:, 0] = 2 * np.pi * nbx / lambdaz * np.array([pixel_x * np.cos(inplane),
                                                                       0,
                                                                       pixel_x * np.sin(inplane)])
                mymatrix[:, 1] = 2 * np.pi * nby / lambdaz * np.array([pixel_y * np.sin(inplane) * np.sin(outofplane),
                                                                       pixel_y * np.cos(outofplane),
                                                                       -pixel_y * np.cos(inplane) * np.sin(outofplane)])
                mymatrix[:, 2] = 2 * np.pi * nbz / lambdaz * np.array(
                    [-tilt * distance * (1 - np.cos(inplane) * np.cos(outofplane)),
                     0,
                     tilt * distance * np.sin(inplane) * np.cos(outofplane)])
            elif self.rocking_angle == "inplane" and mygrazing_angle != 0:
                if verbose:
                    print('rocking angle is phi, with theta non zero')
                # rocking phi angle clockwise around y, incident angle eta is non zero (eta below phi)
                mymatrix[:, 0] = 2 * np.pi * nbx / lambdaz * np.array([pixel_x * np.cos(inplane),
                                                                       0,
                                                                       pixel_x * np.sin(inplane)])
                mymatrix[:, 1] = 2 * np.pi * nby / lambdaz * np.array([pixel_y * np.sin(inplane) * np.sin(outofplane),
                                                                       pixel_y * np.cos(outofplane),
                                                                       -pixel_y * np.cos(inplane) * np.sin(outofplane)])
                mymatrix[:, 2] = 2 * np.pi * nbz / lambdaz * tilt * distance * \
                    np.array([(np.sin(mygrazing_angle) * np.sin(outofplane) +
                               np.cos(mygrazing_angle) * (np.cos(inplane) * np.cos(outofplane) - 1)),
                              np.sin(mygrazing_angle) * np.sin(inplane) * np.cos(outofplane),
                              np.cos(mygrazing_angle) * np.sin(inplane) * np.cos(outofplane)])

        if self.beamline == '34ID':
            if verbose:
                print('using APS 34ID geometry')
            if self.rocking_angle == "outofplane":
                if verbose:
                    print('rocking angle is phi')
                # rocking phi angle anti-clockwise around x (theta does not matter, above phi)
                mymatrix[:, 0] = 2 * np.pi * nbx / lambdaz * np.array([pixel_x * np.cos(inplane),
                                                                       0,
                                                                       -pixel_x * np.sin(inplane)])
                mymatrix[:, 1] = 2 * np.pi * nby / lambdaz * np.array([pixel_y * np.sin(inplane) * np.sin(outofplane),
                                                                       -pixel_y * np.cos(outofplane),
                                                                       pixel_y * np.cos(inplane) * np.sin(outofplane)])
                mymatrix[:, 2] = 2 * np.pi * nbz / lambdaz * np.array([0,
                                                                       -tilt * distance * (1 - np.cos(inplane) * np.cos(
                                                                           outofplane)),
                                                                       -tilt * distance * np.sin(outofplane)])

            elif self.rocking_angle == "inplane" and mygrazing_angle != 0:
                if verbose:
                    print('rocking angle is theta, with phi non zero')
                # rocking theta angle anti-clockwise around y, incident angle is non zero (theta is above phi)
                mymatrix[:, 0] = 2 * np.pi * nbx / lambdaz * np.array([pixel_x * np.cos(inplane),
                                                                       0,
                                                                       -pixel_x * np.sin(inplane)])
                mymatrix[:, 1] = 2 * np.pi * nby / lambdaz * np.array([pixel_y * np.sin(inplane) * np.sin(outofplane),
                                                                       -pixel_y * np.cos(outofplane),
                                                                       pixel_y * np.cos(inplane) * np.sin(outofplane)])
                mymatrix[:, 2] = 2 * np.pi * nbz / lambdaz * tilt * distance * \
                    np.array([(np.sin(mygrazing_angle) * np.sin(outofplane) +
                              np.cos(mygrazing_angle) * (1 - np.cos(inplane) * np.cos(outofplane))),
                              -np.sin(mygrazing_angle) * np.sin(inplane) * np.cos(outofplane),
                              np.cos(mygrazing_angle) * np.sin(inplane) * np.cos(outofplane)])

            elif self.rocking_angle == "inplane" and mygrazing_angle == 0:
                if verbose:
                    print('rocking angle is theta, phi=0')
                # rocking theta angle anti-clockwise around y, assuming incident angle is zero (theta is above phi)
                mymatrix[:, 0] = 2 * np.pi * nbx / lambdaz * np.array([pixel_x * np.cos(inplane),
                                                                       0,
                                                                       -pixel_x * np.sin(inplane)])
                mymatrix[:, 1] = 2 * np.pi * nby / lambdaz * np.array([pixel_y * np.sin(inplane) * np.sin(outofplane),
                                                                       -pixel_y * np.cos(outofplane),
                                                                       pixel_y * np.cos(inplane) * np.sin(outofplane)])
                mymatrix[:, 2] = 2 * np.pi * nbz / lambdaz * np.array(
                    [tilt * distance * (1 - np.cos(inplane) * np.cos(outofplane)),
                     0,
                     tilt * distance * np.sin(inplane) * np.cos(outofplane)])
        if self.beamline == 'SIXS_2018' or self.beamline == 'SIXS_2019':
            if verbose:
                print('using SIXS geometry')
            if self.rocking_angle == "inplane" and mygrazing_angle != 0:
                if verbose:
                    print('rocking angle is mu, with beta non zero')
                # rocking mu angle anti-clockwise around y
                mymatrix[:, 0] = 2 * np.pi * nbx / lambdaz * pixel_x * np.array([np.cos(inplane),
                                                                                 -np.sin(mygrazing_angle) * np.sin(
                                                                                     inplane),
                                                                                 -np.cos(mygrazing_angle) * np.sin(
                                                                                     inplane)])
                mymatrix[:, 1] = 2 * np.pi * nby / \
                    lambdaz * pixel_y * np.array([np.sin(inplane) * np.sin(outofplane),
                                                  (np.sin(mygrazing_angle) * np.cos(inplane) * np.sin(outofplane)
                                                   - np.cos(mygrazing_angle) * np.cos(outofplane)),
                                                  (np.cos(mygrazing_angle) * np.cos(inplane) * np.sin(outofplane)
                                                   + np.sin(mygrazing_angle) * np.cos(outofplane))])
                mymatrix[:, 2] = 2 * np.pi * nbz / lambdaz * tilt * distance \
                    * np.array([np.cos(mygrazing_angle) - np.cos(inplane) * np.cos(outofplane),
                                np.sin(mygrazing_angle) * np.sin(inplane) * np.cos(outofplane),
                                np.cos(mygrazing_angle) * np.sin(inplane) * np.cos(outofplane)])

            elif self.rocking_angle == "inplane" and mygrazing_angle == 0:
                if verbose:
                    print('rocking angle is mu, beta=0')
                # rocking th angle anti-clockwise around y, assuming incident angle is zero (th above tilt)
                mymatrix[:, 0] = 2 * np.pi * nbx / lambdaz * np.array([pixel_x * np.cos(inplane),
                                                                       0,
                                                                       -pixel_x * np.sin(inplane)])
                mymatrix[:, 1] = 2 * np.pi * nby / lambdaz * np.array([pixel_y * np.sin(inplane) * np.sin(outofplane),
                                                                       -pixel_y * np.cos(outofplane),
                                                                       pixel_y * np.cos(inplane) * np.sin(outofplane)])
                mymatrix[:, 2] = 2 * np.pi * nbz / lambdaz * np.array(
                    [tilt * distance * (1 - np.cos(inplane) * np.cos(outofplane)),
                     0,
                     tilt * distance * np.sin(inplane) * np.cos(outofplane)])
        if self.beamline == 'CRISTAL':
            if verbose:
                print('using CRISTAL geometry')
            if self.rocking_angle == "outofplane":
                if verbose:
                    print('rocking angle is komega')
                # rocking tilt angle clockwise around x
                mymatrix[:, 0] = 2 * np.pi * nbx / lambdaz * np.array([pixel_x * np.cos(inplane),
                                                                       0,
                                                                       -pixel_x * np.sin(inplane)])
                mymatrix[:, 1] = 2 * np.pi * nby / lambdaz * np.array([pixel_y * np.sin(inplane) * np.sin(outofplane),
                                                                       -pixel_y * np.cos(outofplane),
                                                                       pixel_y * np.cos(inplane) * np.sin(outofplane)])
                mymatrix[:, 2] = 2 * np.pi * nbz / lambdaz * np.array([0,
                                                                       tilt * distance * (1 - np.cos(inplane) * np.cos(
                                                                           outofplane)),
                                                                       tilt * distance * np.sin(outofplane)])

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
        transfer_matrix = self.update_coords(array_shape=array_shape, tilt_angle=tilt_angle,
                                             pixel_x=pixel_x, pixel_y=pixel_y, verbose=verbose)
        rec_matrix = 2 * np.pi * np.linalg.inv(transfer_matrix).transpose()
        qx_range = np.linalg.norm(rec_matrix[0, :])
        qy_range = np.linalg.norm(rec_matrix[1, :])
        qz_range = np.linalg.norm(rec_matrix[2, :])
        if verbose:
            print('q_range_z, q_range_y, q_range_x=({0:.5f}, {1:.5f}, {2:.5f}) (1/nm)'.format(qz_range, qy_range,
                                                                                              qx_range))
            print('voxelsize_z, voxelsize_y, voxelsize_x='
                  '({0:.2f}, {1:.2f}, {2:.2f}) (1/nm)'.format(2 * np.pi / qz_range, 2 * np.pi / qy_range,
                                                              2 * np.pi / qx_range))
        return 2 * np.pi / qz_range, 2 * np.pi / qy_range, 2 * np.pi / qx_range

    def voxel_sizes_detector(self, array_shape, tilt_angle, pixel_x, pixel_y, verbose=False):
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
        voxel_z = self.wavelength / (array_shape[0] * abs(tilt_angle) * np.pi / 180) * 1e9  # in nm
        voxel_y = self.wavelength * self.distance / (array_shape[1] * pixel_y) * 1e9  # in nm
        voxel_x = self.wavelength * self.distance / (array_shape[2] * pixel_x) * 1e9  # in nm
        if verbose:
            print('voxelsize_z, voxelsize_y, voxelsize_x='
                  '({0:.2f}, {1:.2f}, {2:.2f}) (1/nm)'.format(voxel_z, voxel_y, voxel_x))
        return voxel_z, voxel_y, voxel_x


class SetupPreprocessing(object):
    """
    Class to handle the experimental geometry for preprocessing.
    """
    def __init__(self, beamline, rocking_angle=None, distance=1, energy=8000, direct_beam=(0, 0),
                 beam_direction=(1, 0, 0), sample_inplane=(1, 0, 0), sample_outofplane=(0, 0, 1),
                 sample_offsets=(0, 0, 0), offset_inplane=0, **kwargs):
        """
        Initialize parameters of the experiment.

        :param beamline: name of the beamline: 'ID01', 'SIXS_2018', 'SIXS_2019', '34ID', 'P10', 'CRISTAL'
        :param rocking_angle: angle which is tilted during the scan. 'outofplane', 'inplane', or 'energy'
        :param distance: sample to detector distance in meters, default=1m
        :param energy: X-ray energy in eV, default=8000eV
        :param direct_beam: tuple describing the position of the direct beam in pixels (vertical, horizontal)
        :param beam_direction: x-ray beam direction
        :param sample_inplane: sample inplane reference direction along the beam at 0 angles
        :param sample_outofplane: surface normal of the sample at 0 angles
        :param sample_offsets: tuple of offsets in degree of the sample around z (downstream), y (vertical up) and x
         (outboard). This corresponds to (chi, phi, incident angle) in a standard diffractometer.
        :param offset_inplane: outer angle offset as defined by xrayutilities detector calibration
        :param kwargs:
         - 'filtered_data' = True when the data is a 3D npy/npz array already cleaned up
         - 'is_orthogonal' = True if 'filtered_data' is already orthogonalized
         - 'custom_scan' = True for a stack of images acquired without scan, (no motor data in the spec file)
         - 'custom_images' = list of image numbers for the custom_scan
         - 'custom_monitor' = list of monitor values for normalization for the custom_scan
         - 'custom_motors' = dictionnary of motors values during the scan
        """
        warnings.warn("deprecated, use the class Setup instead", DeprecationWarning)
        for k in kwargs.keys():
            if k not in {'filtered_data', 'is_orthogonal', 'custom_scan', 'custom_images', 'custom_monitor',
                         'custom_motors'}:
                raise Exception("unknown keyword argument given:", k)

        self.beamline = beamline  # string
        self.filtered_data = kwargs.get('filtered_data', False)  # boolean
        self.is_orthogonal = kwargs.get('is_orthogonal', False)  # boolean
        self.custom_scan = kwargs.get('custom_scan', False)  # boolean
        self.custom_images = kwargs.get('custom_images', [])  # list
        self.custom_monitor = kwargs.get('custom_monitor', [])  # list
        self.custom_motors = kwargs.get('custom_motors', {})  # dictionnary
        self.energy = energy  # in eV
        self.wavelength = 12.398 * 1e-7 / energy  # in m
        self.rocking_angle = rocking_angle  # string
        self.distance = distance  # in meters
        self.direct_beam = direct_beam  # in pixels (vertical, horizontal)
        self.beam_direction = beam_direction  # tuple
        self.sample_inplane = sample_inplane  # tuple
        self.sample_outofplane = sample_outofplane  # tuple
        self.sample_offsets = sample_offsets  # tuple
        self.offset_inplane = offset_inplane  # in degrees

        #############################################################
        # detector orientation convention depending on the beamline #
        #############################################################
        # the frame convention is the one of xrayutilities: x downstream, y outboard, z vertical up

        # horizontal axis:
        if beamline in {'ID01', 'SIXS_2018', 'SIXS_2019', 'CRISTAL', 'NANOMAX'}:
            # we look at the detector from downstream, detector X along the outboard direction
            self.detector_hor = 'y+'
        else:  # 'P10', '34ID'
            # we look at the detector from upstream, detector X opposite to the outboard direction
            self.detector_hor = 'y-'

        # vertical axis:
        # origin is at the top, detector Y along vertical down
        self.detector_ver = 'z-'

    def __repr__(self):
        """
        :return: a nicely formatted representation string
        """
        return f"{self.__class__.__name__}: beamline={self.beamline}, energy={self.energy}eV," \
               f" sample to detector distance={self.distance}m"


if __name__ == "__main__":
    print(help(Detector))
