# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#   (c) 06/2021-present : DESY CFEL
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

"""Setup class that defines the experimental geometry."""
from collections.abc import Sequence
import gc
import h5py
from math import isclose
from numbers import Number, Real
import numpy as np
import os
import pathlib
from scipy.interpolate import RegularGridInterpolator

from ..graph import graph_utils as gu
from ..utils import utilities as util
from ..utils import validation as valid
from .diffractometer import (
    Diffractometer34ID,
    DiffractometerCRISTAL,
    DiffractometerID01,
    DiffractometerNANOMAX,
    DiffractometerP10,
    DiffractometerSIXS,
)
from .beamline import create_beamline
from .detector import Detector


class Setup:
    """
    Class for defining the experimental geometry.

    :param beamline: name of the beamline, among {'ID01','SIXS_2018','SIXS_2019',
     '34ID','P10','CRISTAL','NANOMAX'}
    :param detector: an instance of the cass experiment_utils.Detector()
    :param beam_direction: direction of the incident X-ray beam in the frame
     (z downstream,y vertical up,x outboard)
    :param energy: energy setting of the beamline, in eV.
    :param distance: sample to detector distance, in m.
    :param outofplane_angle: vertical detector angle, in degrees.
    :param inplane_angle: horizontal detector angle, in degrees.
    :param tilt_angle: angular step of the rocking curve, in degrees.
    :param rocking_angle: angle which is tilted during the rocking curve in
     {'outofplane', 'inplane', 'energy'}
    :param grazing_angle: motor positions for the goniometer circles below the
     rocking angle. It should be a list/tuple of lenght 1 for out-of-plane rocking
     curves (the chi motor value) and length 2 for inplane rocking
     curves (the chi and omega/om/eta motor values).
    :param kwargs:

     - 'direct_beam': tuple of two real numbers indicating the position of the direct
       beam in pixels at zero detector angles.
     - 'filtered_data': boolean, True if the data and the mask to be loaded were
       already preprocessed.
     - 'custom_scan': boolean, True is the scan does not follow the beamline's usual
       directory format.
     - 'custom_images': list of images numbers when the scan does no follow
       the beamline's usual directory format.
     - 'custom_monitor': list of monitor values when the scan does no follow
       the beamline's usual directory format. The number of values should be equal
       to the number of elements in custom_images.
     - 'custom_motors': list of motor values when the scan does no follow
       the beamline's usual directory format.
     - 'sample_inplane': sample inplane reference direction along the beam at
       0 angles in xrayutilities frame (x is downstream, y outboard, and z vertical
       up at zero incident angle).
     - 'sample_outofplane': surface normal of the sample at 0 angles in xrayutilities
       frame (x is downstream, y outboard, and z vertical up at zero incident angle).
     - 'sample_offsets': list or tuple of three angles in degrees, corresponding to
       the offsets of each of the sample circles (the offset for the most outer
       circle should be at index 0). Convention: the sample offsets will be
       subtracted to measurement the motor values.
     - 'offset_inplane': inplane offset of the detector defined as the outer angle
       in xrayutilities area detector calibration.
     - 'actuators': optional dictionary that can be used to define the entries
       corresponding to actuators in data files (useful at CRISTAL where the location
       of data keeps changing)

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
        Define motors names in the data file.

        This optional dictionary  can be used to define the entries corresponding to
        actuators in data files (useful at CRISTAL where the location of data keeps
        changing)
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
        Direction of the incident X-ray beam.

        Frame convention: (z downstream, y vertical up, x outboard).
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
        Direction of the incident X-ray beam.

        xrayutilities frame convention: (x downstream, y outboard, z vertical up).
        """
        u, v, w = self._beam_direction  # (u downstream, v vertical up, w outboard)
        return u, w, v

    @property
    def beamline(self):
        """Instance of the beamline."""
        return self._beamline.name

    @beamline.setter
    def beamline(self, name):
        self._beamline = create_beamline(name=name)

    @property
    def custom_images(self):
        """
        List of images numbers.

        It can be used for scans which don't follow the beamline's usual directory
        format (e.g. manual scan).
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
        List of monitor values.

        It can be used for scans which don't follow the beamline's usual directory
        format (e.g. manual scan). The number of values should be equal to the number of
        elements in custom_images.
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
        List of motor values.

        It can be used for scans which don't follow the beamline's usual directory
        format (e.g. manual scan).
        """
        return self._custom_motors

    @custom_motors.setter
    def custom_motors(self, value):
        if not self._custom_scan:
            self._custom_motors = None
        else:
            if not isinstance(value, dict):
                raise TypeError(
                    'custom_motors should be a dictionnary of "motor_name": '
                    "motor_positions pairs"
                )
            self._custom_motors = value

    @property
    def custom_scan(self):
        """
        Define is a scan follows the standard directory format or not.

        Boolean, True is the scan does not follow the beamline's usual directory
        format.
        """
        return self._custom_scan

    @custom_scan.setter
    def custom_scan(self, value):
        if not isinstance(value, bool):
            raise TypeError("custom_scan should be a boolean")
        self._custom_scan = value

    @property
    def detector(self):
        """Define a valid Detector instance."""
        return self._detector

    @detector.setter
    def detector(self, value):
        if not isinstance(value, Detector):
            raise TypeError("value should be an instance of Detector")
        self._detector = value

    @property
    def detector_hor(self):
        """
        Expose the detector_hor beamline property to the outer world.

        :return: +/-1 depending on the detector horizontal orientation
        """
        return self._beamline.detector_orientation[self._beamline.detector_hor]

    @property
    def detector_ver(self):
        """
        Expose the detector_ver beamline property to the outer world.

        :return: +/-1 depending on the detector vertical orientation
        """
        return self._beamline.detector_orientation[self._beamline.detector_ver]

    @property
    def diffractometer(self):
        """Return the diffractometer instance."""
        return self._diffractometer

    @property
    def direct_beam(self):
        """
        Direct beam position in pixels.

        Tuple of two real numbers indicating the position of the direct beam in pixels
        at zero detector angles.
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
        """Sample to detector distance, in m."""
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
        """Energy setting of the beamline, in eV."""
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
        Calculate the exit wavevector kout.

        It uses the setup instance parameters. kout is expressed in 1/m in the
        laboratory frame (z downstream, y vertical, x outboard).

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
        Define if data was already preprocessed.

        Boolean, True if the data and the mask to be loaded were already
        preprocessed.
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
        Motor positions for the goniometer circles below the rocking angle.

        It should be a list/tuple of lenght 1 for out-of-plane rocking curves (the
        motor value for mu if it exists) and length 2 for inplane rocking curves (
        e.g. mu and omega/om/eta motor values).
        """
        return self._grazing_angle

    @grazing_angle.setter
    def grazing_angle(self, value):
        if self.rocking_angle == "outofplane":
            # only the mu angle (rotation around the vertical axis,
            # below the rocking angle omega/om/eta) is needed
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
            # one or more values needed, for example: mu angle,
            # the omega/om/eta angle, the chi angle
            # (rotations respectively around the vertical axis,
            # outboard and downstream, below the rocking angle phi)
            valid.valid_container(
                value,
                container_types=(tuple, list),
                item_types=Real,
                allow_none=True,
                name="Setup.grazing_angle",
            )
            self._grazing_angle = value
        else:  # self.rocking_angle == 'energy'
            # there is no sample rocking for energy scans,
            # hence the grazing angle value do not matter
            self._grazing_angle = None

    @property
    def incident_wavevector(self):
        """
        Calculate the incident wavevector kout.

        It uses the setup instance parameters. kin is expressed in 1/m in the
        laboratory frame (z downstream, y vertical, x outboard).

        :return: kin vector
        """
        return 2 * np.pi / self.wavelength * self.beam_direction

    @property
    def inplane_angle(self):
        """Horizontal detector angle, in degrees."""
        return self._inplane_angle

    @inplane_angle.setter
    def inplane_angle(self, value):
        if not isinstance(value, Real) and value is not None:
            raise TypeError("inplane_angle should be a number in degrees")
        self._inplane_angle = value

    @property
    def inplane_coeff(self):
        """
        Expose the inplane_coeff beamline property to the outer world.

        Return a coefficient +/- 1 depending on the detector inplane rotation direction
        and the detector inplane orientation.

        :return: +1 or -1
        """
        return self._beamline.inplane_coeff

    @property
    def outofplane_angle(self):
        """Vertical detector angle, in degrees."""
        return self._outofplane_angle

    @outofplane_angle.setter
    def outofplane_angle(self, value):
        if not isinstance(value, Real) and value is not None:
            raise TypeError("outofplane_angle should be a number in degrees")
        self._outofplane_angle = value

    @property
    def outofplane_coeff(self):
        """
        Expose the outofplane_coeff beamline property to the outer world.

        Return a coefficient +/- 1 depending on the detector out of plane rotation
        direction and the detector out of plane orientation.

        :return: +1 or -1
        """
        return self._beamline.outofplane_coeff

    @property
    def params(self):
        """Return a dictionnary with all parameters."""
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
        Calculate the diffusion vector in the laboratory frame.

        Frame convention: (z downstream, y vertical up, x outboard). The unit is 1/A.

        :return: a tuple of three vectors components.
        """
        return (self.exit_wavevector - self.incident_wavevector) * 1e-10

    @property
    def rocking_angle(self):
        """
        Angle which is tilted during the rocking curve.

        Valid values: {'outofplane', 'inplane'}
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
                'rocking_angle can take only the value "outofplane", '
                '"inplane" or "energy"'
            )
        else:
            self._rocking_angle = value

    @property
    def tilt_angle(self):
        """Angular step of the rocking curve, in degrees."""
        return self._tilt_angle

    @tilt_angle.setter
    def tilt_angle(self, value):
        if not isinstance(value, Real) and value is not None:
            raise TypeError("tilt_angle should be a number in degrees")
        self._tilt_angle = value

    @property
    def wavelength(self):
        """Wavelength in meters."""
        if self.energy:
            return 12.398 * 1e-7 / self.energy  # in m

    def __repr__(self):
        """Representation string of the Setup instance."""
        return (
            f"{self.__class__.__name__}(beamline='{self.beamline}', "
            f"detector='{self.detector.name}',"
            f" beam_direction={self.beam_direction}, "
            f"energy={self.energy}, distance={self.distance}, "
            f"outofplane_angle={self.outofplane_angle},\n"
            f"inplane_angle={self.inplane_angle}, "
            f"tilt_angle={self.tilt_angle}, "
            f"rocking_angle='{self.rocking_angle}', "
            f"grazing_angle={self.grazing_angle},\n"
            f"pixel_size={self.detector.unbinned_pixel}, "
            f"direct_beam={self.direct_beam}, "
            f"sample_offsets={self.diffractometer.sample_offsets}, "
            f"filtered_data={self.filtered_data}, "
            f"custom_scan={self.custom_scan},\n"
            f"custom_images={self.custom_images},\n"
            f"custom_monitor={self.custom_monitor},\n"
            f"custom_motors={self.custom_motors},\n"
            f"sample_inplane={self.sample_inplane}, "
            f"sample_outofplane={self.sample_outofplane}, "
            f"offset_inplane={self.offset_inplane})"
        )

    def create_logfile(self, scan_number, root_folder, filename):
        """
        Create the logfile used in gridmap().

        :param scan_number: the scan number to load
        :param root_folder: the root directory of the experiment, where is the
         specfile/.fio file
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
        """Create a Diffractometer instance depending on the beamline."""
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
        Interpolate the orthogonal object back into the non-orthogonal detector frame.

        :param obj: real space object, in the orthogonal laboratory frame
        :param voxel_size: voxel size of the original object, number of list/tuple of
         three numbers
        :param width_z: size of the area to plot in z (axis 0), centered on the middle
         of the initial array
        :param width_y: size of the area to plot in y (axis 1), centered on the middle
         of the initial array
        :param width_x: size of the area to plot in x (axis 2), centered on the middle
         of the initial array
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
        Init paths used for data processing and logging.

        Update the detector instance with initialized paths and template for filenames
        depending on the beamline.

        :param sample_name: string in front of the scan number in the data folder name.
        :param scan_number: the scan number
        :param root_folder: folder of the experiment, where all scans are stored
        :param save_dir: path of the directory where to save the analysis results,
         can be None
        :param specfile_name: beamline-dependent string

         - ID01: name of the spec file without '.spec'
         - SIXS_2018 and SIXS_2019: None or full path of the alias dictionnary (e.g.
           root_folder+'alias_dict_2019.txt')
         - empty string for all other beamlines

        :param template_imagefile: beamline-dependent template for the data files

         - ID01: 'data_mpx4_%05d.edf.gz' or 'align_eiger2M_%05d.edf.gz'
         - SIXS_2018: 'align.spec_ascan_mu_%05d.nxs'
         - SIXS_2019: 'spare_ascan_mu_%05d.nxs'
         - Cristal: 'S%d.nxs'
         - P10: '_master.h5'
         - NANOMAX: '%06d.h5'
         - 34ID: 'Sample%dC_ES_data_51_256_256.npz'

        :param data_dirname: name of the data folder, if None it will use the beamline
         default, if it is an empty string, it will look for the data directly into
         the scan folder (no subfolder)
        :param save_dirname: name of the saving folder, by default 'save_dir/result/'
         will be created
        :param create_savedir: boolean, True to create the saving folder if it does
         not exist
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
                    f"datadir = '{datadir}'\nsavedir = '{savedir}'\n"
                    f"template_imagefile = '{template_imagefile}'\n"
                )
            else:
                print(
                    f"rootdir = '{root_folder}'\nsavedir = '{savedir}'\n"
                    f"sample_name = '{self.detector.sample_name}'\n"
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
        Geometrical transformation in direct space.

        Interpolate arrays (direct space output of the phase retrieval) in the
        orthogonal reference frame where q_com is aligned onto the array axis
        reference_axis.

        :param arrays: tuple of 3D arrays of the same shape (output of the phase
         retrieval), in the detector frame
        :param q_com: tuple of 3 vector components for the q values of the center
         of mass of the Bragg peak, expressed in an orthonormal frame x y z
        :param initial_shape: shape of the FFT used for phasing
        :param voxel_size: number or list of three user-defined voxel sizes for
         the interpolation, in nm. If a single number is provided, the voxel size
         will be identical in all directions.
        :param fill_value: tuple of real numbers, fill_value parameter for the
         RegularGridInterpolator, same length as the number of arrays
        :param reference_axis: 3D vector along which q will be aligned, expressed in
         an orthonormal frame x y z
        :param verbose: True to have printed comments
        :param debugging: tuple of booleans of the same length as the number of
         input arrays, True to show plots before and after interpolation
        :param kwargs:

         - 'title': tuple of strings, titles for the debugging plots, same length as
           the number of arrays
         - width_z: size of the area to plot in z (axis 0), centered on the middle of
           the initial array
         - width_y: size of the area to plot in y (axis 1), centered on the middle of
           the initial array
         - width_x: size of the area to plot in x (axis 2), centered on the middle of
           the initial array

        :return:

         - an array (if a single array was provided) or a tuple of arrays interpolated
           on an orthogonal grid (same length as the number of input arrays)
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
                "strain calculation along directions "
                "other than array axes is not implemented"
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

        #########################################################
        # calculate the direct space voxel sizes in nm          #
        # based on the FFT window shape used in phase retrieval #
        #########################################################
        dz_realspace, dy_realspace, dx_realspace = self.voxel_sizes(
            initial_shape,
            tilt_angle=abs(self.tilt_angle),
            pixel_x=self.detector.unbinned_pixel[1],
            pixel_y=self.detector.unbinned_pixel[0],
        )
        if verbose:
            print(
                "Sampling in the laboratory frame (z, y, x): ",
                f"({dz_realspace:.2f} nm,"
                f" {dy_realspace:.2f} nm,"
                f" {dx_realspace:.2f} nm)",
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
                    f"({tilt:.4f} deg,"
                    f" {pixel_y * 1e6:.2f} um,"
                    f" {pixel_x * 1e6:.2f} um)",
                )

            # sanity check, the direct space voxel sizes
            # calculated below should be equal to the original ones
            dz_realspace, dy_realspace, dx_realspace = self.voxel_sizes(
                input_shape, tilt_angle=abs(tilt), pixel_x=pixel_x, pixel_y=pixel_y
            )
            if verbose:
                print(
                    "Sanity check, recalculated direct space voxel sizes (z, y, x): ",
                    f"({dz_realspace:.2f} nm,"
                    f" {dy_realspace:.2f} nm,"
                    f" {dx_realspace:.2f} nm)",
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
        ################################################
        # calculate the full transfer matrix including #
        # the rotation into the crystal frame          #
        ################################################
        transfer_matrix = np.matmul(rotation_matrix, transfer_matrix)
        # transfer_matrix is the transformation matrix of the direct space coordinates
        # the spacing in the crystal frame is therefore given by the rows of the matrix
        d_along_x = np.linalg.norm(transfer_matrix[0, :])  # along x outboard
        d_along_y = np.linalg.norm(transfer_matrix[1, :])  # along y vertical up
        d_along_z = np.linalg.norm(transfer_matrix[2, :])  # along z downstream

        ################################################
        # find the shape of the output array that fits #
        # the extent of the data after transformation  #
        ################################################

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
                "\nCalculating the shape of the output array "
                "fitting the data extent after transformation:"
                f"\nSampling in the crystal frame (axis 0, axis 1, axis 2):    "
                f"({d_along_z:.2f} nm,"
                f" {d_along_y:.2f} nm,"
                f" {d_along_x:.2f} nm)"
            )
        # these positions are not equally spaced,
        # we just extract the data extent from them
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

        # ortho_matrix is the transformation matrix from the detector
        # coordinates to the laboratory frame
        # in RGI, we want to calculate the coordinates that would have
        # a grid of the laboratory frame expressed in the
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
        Geometrical transformation in reciprocal (Fourier) space.

        Interpolate arrays in the orthogonal laboratory frame (z/qx downstream,
        y/qz vertical up, x/qy outboard) or crystal frame (q aligned along one array
        axis). The ouput shape will be increased in order to keep the same range in q
        in each direction. The sampling in q is defined as the norm of the rows of the
        transformation matrix.

        :param arrays: tuple of 3D arrays of the same shape (e.g.: reciprocal space
         diffraction pattern and mask), in the detector frame
        :param fill_value: tuple of real numbers, fill_value parameter for the
         RegularGridInterpolator, same length as the number of arrays
        :param align_q: boolean, if True the data will be rotated such that q is along
         reference_axis, and q values will be calculated in the pseudo crystal frame.
        :param reference_axis: 3D vector along which q will be aligned, expressed in
         an orthonormal frame x y z
        :param verbose: True to have printed comments
        :param debugging: tuple of booleans of the same length as the number of
         input arrays, True to show plots before and after interpolation
        :param kwargs:

         - 'title': tuple of strings, titles for the debugging plots, same length as
           the number of arrays
         - 'scale': tuple of strings (either 'linear' or 'log'), scale for
           the debugging plots, same length as the number of arrays
         - width_z: size of the area to plot in z (axis 0), centered on
           the middle of the initial array
         - width_y: size of the area to plot in y (axis 1), centered on
           the middle of the initial array
         - width_x: size of the area to plot in x (axis 2), centered on
           the middle of the initial array

        :return:

         - an array (if a single array was provided) or a tuple of arrays interpolated
           on an orthogonal grid (same length as the number of input arrays)
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

        # the voxel size in q in the laboratory frame
        # is given by the rows of the transformation matrix
        # (the unit is 1/nm)
        dq_along_x = np.linalg.norm(transfer_matrix[0, :])  # along x outboard
        dq_along_y = np.linalg.norm(transfer_matrix[1, :])  # along y vertical up
        dq_along_z = np.linalg.norm(transfer_matrix[2, :])  # along z downstream

        ################################################
        # find the shape of the output array that fits #
        # the extent of the data after transformation  #
        ################################################

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
            # the center of mass of the diffraction
            # should be in the center of the array!
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

            # calculate the rotation matrix from the crystal frame
            # to the laboratory frame
            # (inverse rotation to have reference_axis along q)
            rotation_matrix = util.rotation_matrix_3d(
                axis_to_align=reference_axis,
                reference_axis=np.array([q_along_x_com, q_along_y_com, q_along_z_com])
                / qnorm,
            )

            # calculate the full transfer matrix
            # including the rotation into the crystal frame
            transfer_matrix = np.matmul(rotation_matrix, transfer_matrix)

            # the voxel size in q in the laboratory frame
            # is given by the rows of the transformation matrix
            # (the unit is 1/nm)
            dq_along_x = np.linalg.norm(transfer_matrix[0, :])  # along x outboard
            dq_along_y = np.linalg.norm(transfer_matrix[1, :])  # along y vertical up
            dq_along_z = np.linalg.norm(transfer_matrix[2, :])  # along z downstream

            # calculate the new offset in the crystal frame
            # (inverse rotation to have qz along q)
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

            # these q values are not equally spaced,
            # we just extract the q extent from them
            nx_output = int(np.rint((q_along_x.max() - q_along_x.min()) / dq_along_x))
            ny_output = int(np.rint((q_along_y.max() - q_along_y.min()) / dq_along_y))
            nz_output = int(np.rint((q_along_z.max() - q_along_z.min()) / dq_along_z))

            if verbose:
                print(
                    f"\nSampling in q in the crystal frame (axis 0, axis 1, axis 2):  "
                    f"({dq_along_z:.5f} 1/nm,"
                    f" {dq_along_y:.5f} 1/nm,"
                    f" {dq_along_x:.5f} 1/nm)"
                )

        del q_along_x, q_along_y, q_along_z, myx, myy, myz
        gc.collect()

        ##########################################################
        # crop the output shape in order to fit FFT requirements #
        ##########################################################
        nz_output, ny_output, nx_output = util.smaller_primes(
            (nz_output, ny_output, nx_output), maxprime=7, required_dividers=(2,)
        )
        if verbose:
            print(
                f"\nInitial shape = ({nbz},{nby},{nbx})\n"
                f"Output shape  = ({nz_output},{ny_output},{nx_output})"
                f" (satisfying FFT shape requirements)"
            )

        ########################################################
        # define the interpolation qx qz qy 1D vectors in 1/nm #
        # the reference being the center of the array          #
        ########################################################
        # the usual frame is used for q values:
        # qx downstream, qz vertical up, qy outboard
        # this assumes that the center of mass of the diffraction pattern
        # was at the center of the array
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

        # transfer_matrix is the transformation matrix from
        # the detector coordinates to the laboratory/crystal frame
        # in RGI, we want to calculate the coordinates that would have a grid
        # of the laboratory/crystal frame expressed
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
            # convert array type to float,
            # for integers the interpolation can lead to artefacts
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

        # add the offset due to the detector angles
        # to qx qz qy vectors, convert them to 1/A
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

        :param vector: tuple of 3 coordinates, vector to be transformed in the detector
         frame
        :param array_shape: shape of the 3D array to orthogonalize
        :param tilt_angle: angular step during the rocking curve, in degrees
        :param pixel_x: horizontal pixel size, in meters
        :param pixel_y: vertical pixel size, in meters
        :param verbose: True to have printed comments
        :return: tuple of 3 numbers, the coordinates of the vector expressed in the
         laboratory frame
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
        # ortho_matrix is the transformation matrix
        # from the detector coordinates to the laboratory frame
        # Here, we want to calculate the coordinates that would have
        # a vector of the laboratory frame expressed in the
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
        Calculate the transformation matrix from detector frame to laboratory frame.

        For the transformation in direct space, the length scale is in nm,
        for the transformation in reciprocal space, it is in 1/nm.

        :param array_shape: shape of the 3D array to orthogonalize
        :param tilt_angle: angular step during the rocking curve, in degrees
        :param pixel_x: horizontal pixel size, in meters
        :param pixel_y: vertical pixel size, in meters
        :param direct_space: True in order to return the transformation matrix in
         direct space
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

        if self.beamline == "ID01":
            if verbose:
                print("using ESRF ID01 PSIC geometry")
            if not isclose(grazing_angle[0], 0, rel_tol=1e-09, abs_tol=1e-09):
                raise NotImplementedError(
                    "Non-zero mu not implemented "
                    "for the transformation matrices at ID01"
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
                    * self.detector_hor
                    * np.array(
                        [-pixel_x * np.cos(inplane), 0, -pixel_x * np.sin(inplane)]
                    )
                )
                mymatrix[:, 1] = (
                    2
                    * np.pi
                    / lambdaz
                    * self.detector_ver
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

                # rocking phi angle clockwise around y,
                # incident angle eta is non zero (eta below phi)
                mymatrix[:, 0] = (
                    2
                    * np.pi
                    / lambdaz
                    * self.detector_hor
                    * np.array(
                        [-pixel_x * np.cos(inplane), 0, -pixel_x * np.sin(inplane)]
                    )
                )
                mymatrix[:, 1] = (
                    2
                    * np.pi
                    / lambdaz
                    * self.detector_ver
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
                # rocking omega angle clockwise around x at mu=0,
                # chi potentially non zero (chi below omega)
                # (phi does not matter, above eta)
                mymatrix[:, 0] = (
                    2
                    * np.pi
                    / lambdaz
                    * pixel_x
                    * self.detector_hor
                    * np.array([-np.cos(inplane), 0, np.sin(inplane)])
                )
                mymatrix[:, 1] = (
                    2
                    * np.pi
                    / lambdaz
                    * pixel_y
                    * self.detector_ver
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

                # rocking phi angle clockwise around y,
                # omega and chi potentially non zero (chi below omega below phi)
                mymatrix[:, 0] = (
                    2
                    * np.pi
                    / lambdaz
                    * pixel_x
                    * self.detector_hor
                    * np.array([-np.cos(inplane), 0, np.sin(inplane)])
                )
                mymatrix[:, 1] = (
                    2
                    * np.pi
                    / lambdaz
                    * pixel_y
                    * self.detector_ver
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
                # rocking theta angle clockwise around x
                # (phi does not matter, above eta)
                mymatrix[:, 0] = (
                    2
                    * np.pi
                    / lambdaz
                    * pixel_x
                    * self.detector_hor
                    * np.array([-np.cos(inplane), 0, -np.sin(inplane)])
                )
                mymatrix[:, 1] = (
                    2
                    * np.pi
                    / lambdaz
                    * pixel_y
                    * self.detector_ver
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
                        "rocking angle is phi,"
                        f" theta={grazing_angle[0]*180/np.pi:.3f} deg"
                    )
                # rocking phi angle clockwise around y,
                # incident angle theta is non zero (theta below phi)
                mymatrix[:, 0] = (
                    2
                    * np.pi
                    / lambdaz
                    * pixel_x
                    * self.detector_hor
                    * np.array([-np.cos(inplane), 0, -np.sin(inplane)])
                )
                mymatrix[:, 1] = (
                    2
                    * np.pi
                    / lambdaz
                    * pixel_y
                    * self.detector_ver
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
                    * self.detector_hor
                    * np.array([-np.cos(inplane), 0, np.sin(inplane)])
                )
                mymatrix[:, 1] = (
                    2
                    * np.pi
                    / lambdaz
                    * pixel_y
                    * self.detector_ver
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
                        "rocking angle is phi,"
                        f" theta={grazing_angle[0] * 180 / np.pi:.3f} deg"
                    )
                # rocking phi angle anti-clockwise around x
                mymatrix[:, 0] = (
                    2
                    * np.pi
                    / lambdaz
                    * pixel_x
                    * self.detector_hor
                    * np.array([-np.cos(inplane), 0, np.sin(inplane)])
                )
                mymatrix[:, 1] = (
                    2
                    * np.pi
                    / lambdaz
                    * pixel_y
                    * self.detector_ver
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
                        "rocking angle is mu,"
                        f" beta={grazing_angle[0] * 180 / np.pi:.3f} deg"
                    )

                # rocking mu angle anti-clockwise around y
                mymatrix[:, 0] = (
                    2
                    * np.pi
                    / lambdaz
                    * pixel_x
                    * self.detector_hor
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
                    * self.detector_ver
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
                    * self.detector_hor
                    * np.array([-np.cos(inplane), 0, np.sin(inplane)])
                )
                mymatrix[:, 1] = (
                    2
                    * np.pi
                    / lambdaz
                    * pixel_y
                    * self.detector_ver
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
                        "rocking angle is phi,"
                        f" mgomega={grazing_angle[0]*180/np.pi:.3f} deg"
                    )
                # rocking phi angle anti-clockwise around y,
                # incident angle mgomega is non zero (mgomega below phi)
                mymatrix[:, 0] = (
                    2
                    * np.pi
                    / lambdaz
                    * pixel_x
                    * self.detector_hor
                    * np.array([-np.cos(inplane), 0, np.sin(inplane)])
                )
                mymatrix[:, 1] = (
                    2
                    * np.pi
                    / lambdaz
                    * pixel_y
                    * self.detector_ver
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
            # for a discrete FT, the dimensions of the basis vectors
            # after the transformation are related to the total
            # domain size
            mymatrix[:, 0] = nbx * mymatrix[:, 0]
            mymatrix[:, 1] = nby * mymatrix[:, 1]
            mymatrix[:, 2] = nbz * mymatrix[:, 2]
            return 2 * np.pi * np.linalg.inv(mymatrix).transpose()
        # reciprocal length scale in  1/nm
        return mymatrix, q_offset

    def voxel_sizes(self, array_shape, tilt_angle, pixel_x, pixel_y, verbose=False):
        """
        Calculate the direct space voxel sizes in the laboratory frame.

        Frame convention: (z downstream, y vertical up, x outboard).

        :param array_shape: shape of the 3D array to orthogonalize
        :param tilt_angle: angular step during the rocking curve, in degrees
        :param pixel_x: horizontal pixel size, in meters
        :param pixel_y: vertical pixel size, in meters
        :param verbose: True to have printed comments
        :return: the direct space voxel sizes in nm, in the laboratory frame
         (voxel_z, voxel_y, voxel_x)
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
        # transfer_matrix is the transformation matrix
        # of the direct space coordinates (its columns are the
        # non-orthogonal basis vectors reciprocal to the detector frame)
        # the spacing in the laboratory frame is therefore
        # given by the rows of the matrix
        dx = np.linalg.norm(transfer_matrix[0, :])  # along x outboard
        dy = np.linalg.norm(transfer_matrix[1, :])  # along y vertical up
        dz = np.linalg.norm(transfer_matrix[2, :])  # along z downstream

        if verbose:
            print(
                "Direct space voxel size (z, y, x) = "
                f"({dz:.2f}, {dy:.2f}, {dx:.2f}) (nm)"
            )
        return dz, dy, dx

    def voxel_sizes_detector(
        self, array_shape, tilt_angle, pixel_x, pixel_y, verbose=False
    ):
        """
        Calculate the direct space voxel sizes in the detector frame.

        Frame convention: (z rocking angle, y detector vertical axis, x detector
        horizontal axis).

        :param array_shape: shape of the 3D array used in phase retrieval
        :param tilt_angle: angular step during the rocking curve, in degrees
        :param pixel_x: horizontal pixel size, in meters
        :param pixel_y: vertical pixel size, in meters
        :param verbose: True to have printed comments
        :return: the direct space voxel sizes in nm, in the detector frame
         (voxel_z, voxel_y, voxel_x)
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
