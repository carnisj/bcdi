# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#   (c) 06/2021-present : DESY CFEL
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

"""
Setup class that defines the experimental geometry.

You can think of it as the public interface for the Beamline and Diffractometer child
classes. A script would call a method from Setup, which would then retrieve the required
beamline-dependent information from the child classes.
"""
from collections.abc import Sequence
import datetime
import multiprocessing as mp
from numbers import Integral, Real
import numpy as np
from scipy.interpolate import griddata, RegularGridInterpolator
import sys
import time
from typing import List, Optional, Tuple, Union
from ..graph import graph_utils as gu
from ..utils import utilities as util
from ..utils import validation as valid
from .diffractometer import create_diffractometer
from .beamline import create_beamline
from .detector import create_detector, Detector


class Setup:
    """
    Class for defining the experimental geometry.

    :param beamline: str, name of the beamline
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

    labframe_to_xrayutil = {
        "x+": "y+",
        "x-": "y-",
        "y+": "z+",
        "y-": "z-",
        "z+": "x+",
        "z-": "x-",
    }  # conversion table from the laboratory frame (CXI convention)
    # (z downstream, y vertical up, x outboard) to the frame of xrayutilities
    # (x downstream, y outboard, z vertical up)

    def __init__(
        self,
        beamline,
        detector=None,
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
                "is_series",
            },
            name="Setup.__init__",
        )

        # kwargs for preprocessing forward CDI data
        self.direct_beam = kwargs.get("direct_beam")
        # kwargs for loading and preprocessing data
        sample_offsets = kwargs.get("sample_offsets")  # sequence
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
        # kwargs for series (several frames per point) at P10
        self.is_series = kwargs.get("is_series", False)  # boolean
        # load positional arguments corresponding to instance properties
        self.beamline = beamline
        # create the Diffractometer instance
        self._diffractometer = create_diffractometer(
            beamline=self.beamline,
            sample_offsets=sample_offsets,
        )
        self.detector = detector or create_detector("Dummy")
        self.beam_direction = beam_direction
        self.energy = energy
        self.distance = distance
        self.outofplane_angle = outofplane_angle
        self.inplane_angle = inplane_angle
        self.tilt_angle = tilt_angle
        self.rocking_angle = rocking_angle
        self.grazing_angle = grazing_angle

        # initialize other attributes
        self.logfile = None

    @property
    def actuators(self):
        """
        Define motors names in the data file.

        This optional dictionary can be used to define the entries corresponding to
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
            container_types=(tuple, list, np.ndarray),
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
        Direction of the incident X-ray beam in xrayutilities frame.

        xrayutilities frame convention: (x downstream, y outboard, z vertical up).
        """
        u, v, w = self._beam_direction  # (u downstream, v vertical up, w outboard)
        return u, w, v

    @property
    def beamline(self):
        """Create a beamline instance, but return the name only."""
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
            if isinstance(value, np.ndarray):
                value = list(value)
            valid.valid_container(
                value,
                container_types=(tuple, list),
                min_length=1,
                item_types=Integral,
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
    def detector_hor_xrutil(self):
        """
        Convert the detector horizontal orientation to xrayutilities frame.

        The laboratory frame convention is (z downstream, y vertical, x outboard).
        The frame convention of xrayutilities is (x downstream, y outboard,
        z vertical up).

        :return: "x+" or "x-" depending on the detector horizontal orientation
        """
        return self.labframe_to_xrayutil[self._beamline.detector_hor]

    @property
    def detector_ver_xrutil(self):
        """
        Convert the detector vertical orientation to xrayutilities frame.

        The laboratory frame convention is (z downstream, y vertical, x outboard).
        The frame convention of xrayutilities is (x downstream, y outboard,
        z vertical up).

        :return: "z+" or "z-" depending on the detector vertical orientation
        """
        return self.labframe_to_xrayutil[self._beamline.detector_ver]

    @property
    def diffractometer(self):
        """Public interface to access the diffractometer instance."""
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
        return self._beamline.exit_wavevector(
            diffractometer=self.diffractometer,
            inplane_angle=self.inplane_angle,
            outofplane_angle=self.outofplane_angle,
            wavelength=self.wavelength,
        )

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
        valid.valid_container(
            value,
            container_types=(tuple, list),
            item_types=Real,
            allow_none=True,
            name="Setup.grazing_angle",
        )
        self._grazing_angle = value

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
        return self._beamline.inplane_coeff(self.diffractometer)

    @property
    def is_series(self):
        """Set to true for series measurement at P10 (several frames per point)."""
        return self._is_series

    @is_series.setter
    def is_series(self, val):
        if not isinstance(val, bool):
            raise TypeError(f"is_series should be a boolean, got {type(val)}")
        self._is_series = val

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
        return self._beamline.outofplane_coeff(self.diffractometer)

    @property
    def params(self):
        """Return a dictionnary with all parameters."""
        return {
            "Class": self.__class__.__name__,
            "beamline": self.beamline,
            "detector": self.detector.name,
            "pixel_size_m": self.detector.unbinned_pixel_size,
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
            "wavelength_m": self.wavelength,
            "is_series": self.is_series,
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
        return None

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
            f"pixel_size={self.detector.unbinned_pixel_size}, "
            f"direct_beam={self.direct_beam}, "
            f"sample_offsets={self.diffractometer.sample_offsets}, "
            f"filtered_data={self.filtered_data},\n"
            f"custom_scan={self.custom_scan}, "
            f"custom_images={self.custom_images},\n"
            f"custom_monitor={self.custom_monitor},\n"
            f"custom_motors={self.custom_motors},\n"
            f"sample_inplane={self.sample_inplane}, "
            f"sample_outofplane={self.sample_outofplane}, "
            f"offset_inplane={self.offset_inplane}, "
            f"is_series={self.is_series})"
        )

    def calc_qvalues_xrutils(self, logfile, hxrd, nb_frames, **kwargs):
        """
        Calculate the 3D q values of the BCDI scan using xrayutilities.

        :param logfile: the logfile created in Setup.create_logfile()
        :param hxrd: an initialized xrayutilities HXRD object used for the
         orthogonalization of the dataset
        :param nb_frames: length of axis 0 in the 3D dataset. If the data was cropped
         or padded, it may be different from the original length len(frames_logical)
        :param kwargs:

         - 'scan_number': the scan number to load
         - 'frames_logical': array of length the number of measured frames.
           In case of cropping/padding the number of frames changes. A frame whose
           index is set to 1 means that it is used, 0 means not used, -1 means padded
           (added) frame

        :return:
         - qx, qz, qy components for the dataset. xrayutilities uses the xyz crystal
           frame: for incident angle = 0, x is downstream, y outboard, and z vertical
           up. The output of hxrd.Ang2Q.area is qx, qy, qz is this order. If q values
           seem wrong, check if diffractometer angles have default values set at 0,
           otherwise use the parameter setup.diffractometer.sample_offsets to correct it
         - updated frames_logical

        """
        # check some parameters
        frames_logical = kwargs.get("frames_logical")
        valid.valid_1d_array(
            frames_logical,
            allow_none=True,
            allowed_types=Integral,
            allowed_values=(-1, 0, 1),
            name="frames_logical",
        )
        scan_number = kwargs.get("scan_number")
        if not isinstance(scan_number, int):
            raise TypeError(
                "scan_number should be an integer, got " f"{type(scan_number)}"
            )

        # process motor positions
        processed_positions = self._beamline.process_positions(
            setup=self,
            logfile=logfile,
            nb_frames=nb_frames,
            scan_number=scan_number,
            frames_logical=frames_logical,
        )

        # calculate q values
        qx, qy, qz = hxrd.Ang2Q.area(
            *processed_positions[:-1],
            en=processed_positions[-1],
            delta=self.detector.offsets,
        )
        print("Use the parameter 'sample_offsets' to correct diffractometer values.\n")
        return qx, qz, qy, frames_logical

    def check_setup(
        self,
        grazing_angle: Optional[Tuple[Real, ...]],
        inplane_angle: Real,
        outofplane_angle: Real,
        tilt_angle: np.ndarray,
        detector_distance: Real,
        energy: Real,
    ) -> None:
        """
        Check if the required parameters are correctly defined.

        This method is called in Diffractometer.goniometer_value, which is used only for
        the geometric transformation using the linearized transformation matrix. Hence,
        arrays for detector angles and the energy are not allowed.

        :param grazing_angle:
        :param inplane_angle: detector inplane angle in degrees
        :param outofplane_angle: detector out-of-plane angle in degrees
        :param tilt_angle: ndarray of shape (N,), values of the rocking angle
        :param detector_distance: sample to detector distance in meters
        :param energy: X-ray energy in eV
        """
        self.grazing_angle = grazing_angle

        self.energy = self.energy or energy
        if self.energy is None:
            raise ValueError("the X-ray energy is not defined")
        if not isinstance(self.energy, Real):
            raise TypeError("the X-ray energy should be fixed")

        self.distance = self.distance or detector_distance
        if self.distance is None:
            raise ValueError("the sample to detector distance is not defined")
        if not isinstance(self.distance, Real):
            raise TypeError("the sample to detector distance should be fixed")

        self.outofplane_angle = self.outofplane_angle or outofplane_angle
        if self.outofplane_angle is None:
            raise ValueError("the detector out-of-plane angle is not defined")

        self.inplane_angle = self.inplane_angle or inplane_angle
        if self.inplane_angle is None:
            raise ValueError("the detector in-plane angle is not defined")

        if tilt_angle is not None:
            tilt_angle = np.mean(
                np.asarray(tilt_angle)[1:] - np.asarray(tilt_angle)[0:-1]
            )
        self.tilt_angle = self.tilt_angle or tilt_angle
        if self.tilt_angle is None:
            raise ValueError("the tilt angle is not defined")
        if not isinstance(self.tilt_angle, Real):
            raise TypeError("the tilt angle should be a number")

    def create_logfile(self, scan_number, root_folder, filename):
        """
        Create the logfile, which can be a log/spec file or the data itself.

        The nature of this file is beamline dependent.

        :param scan_number: the scan number to load
        :param root_folder: the root directory of the experiment, where is the
         specfile/.fio file
        :param filename: the file name to load, or the absolute path of
         'alias_dict.txt' for SIXS
        :return: logfile
        """
        if self.custom_scan:
            logfile = None
        else:
            logfile = self._beamline.create_logfile(
                scan_number=scan_number,
                root_folder=root_folder,
                filename=filename,
                datadir=self.detector.datadir,
                template_imagefile=self.detector.template_imagefile,
                name=self.beamline,
            )
        self.logfile = logfile

        return logfile

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

        ortho_matrix, _ = self.transformation_bcdi(
            array_shape=(nbz, nby, nbx),
            tilt_angle=self.tilt_angle,
            pixel_x=self.detector.unbinned_pixel_size[1],
            pixel_y=self.detector.unbinned_pixel_size[0],
            direct_space=True,
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

    def grid_cylindrical(
        self,
        array,
        rotation_angle,
        direct_beam,
        interp_angle,
        interp_radius,
        fill_value=np.nan,
        comment="",
        multiprocessing=False,
    ):
        """
        Interpolate a tomographic dataset onto cartesian coordinates.

        The initial 3D array is in cylindrical coordinates. There is no benefit from
        multiprocessing, the data transfers are the limiting factor.

        :param array: 3D array of intensities measured in the detector frame
        :param rotation_angle: array, rotation angle values for the rocking scan
        :param direct_beam: position in pixels of the rotation pivot in the direction
         perpendicular to the rotation axis
        :param interp_angle: 2D array, polar angles for the interpolation in a plane
         perpendicular to the rotation axis
        :param interp_radius: 2D array, polar radii for the interpolation in a plane
         perpendicular to the rotation axis
        :param fill_value: real number (np.nan allowed), fill_value parameter for the
         RegularGridInterpolator
        :param comment: a comment to be printed
        :param multiprocessing: True to use multiprocessing
        :return: the 3D array interpolated onto the 3D cartesian grid
        """
        valid.valid_ndarray(arrays=array, ndim=3)

        def collect_result(result):
            """
            Process the result after asynchronous multiprocessing.

            This callback function updates global arrays.

            :param result: the output of interp_slice, containing the 2d interpolated
             slice and the slice index
            """
            nonlocal interp_array, number_y, slices_done
            slices_done = slices_done + 1
            # result is a tuple: data, mask, counter, file_index
            # stack the 2D interpolated frame along the rotation axis,
            # taking into account the flip of the detector Y axis (pointing down)
            # compare to the laboratory frame vertical axis (pointing up)
            interp_array[:, number_y - (result[1] + 1), :] = result[0]
            sys.stdout.write(
                "\r    gridding progress: {:d}%".format(
                    int(slices_done / number_y * 100)
                )
            )
            sys.stdout.flush()

        rotation_step = rotation_angle[1] - rotation_angle[0]
        if rotation_step < 0:
            # flip rotation_angle and the data accordingly, RegularGridInterpolator
            # takes only increasing position vectors
            rotation_angle = np.flip(rotation_angle)
            array = np.flip(array, axis=0)

        _, number_y, _ = array.shape
        _, numx = interp_angle.shape  # data shape is (numx, numx) by construction
        interp_array = np.zeros((numx, number_y, numx), dtype=array.dtype)
        slices_done = 0

        start = time.time()
        if multiprocessing:
            print(
                "\nGridding",
                comment,
                ", number of processors used: ",
                min(mp.cpu_count(), number_y),
            )
            mp.freeze_support()
            pool = mp.Pool(
                processes=min(mp.cpu_count(), number_y)
            )  # use this number of processesu

            for idx in range(number_y):
                pool.apply_async(
                    self.interp_2dslice,
                    args=(
                        array[:, idx, :],
                        idx,
                        rotation_angle,
                        direct_beam,
                        interp_angle,
                        interp_radius,
                        fill_value,
                    ),
                    callback=collect_result,
                    error_callback=util.catch_error,
                )
                # interp_2dslice must be a pickable object,
                # i.e. defined at the top level of the module

            pool.close()
            pool.join()
            # postpones the execution of next line of code until all processes
            # in the queue are done.

        else:  # no multiprocessing
            print("\nGridding", comment, ", no multiprocessing")
            for idx in range(
                number_y
            ):  # loop over 2D frames perpendicular to the rotation axis
                temp_array, _ = self.interp_2dslice(
                    array=array[:, idx, :],
                    slice_index=idx,
                    rotation_angle=rotation_angle,
                    direct_beam=direct_beam,
                    interp_angle=interp_angle,
                    interp_radius=interp_radius,
                    fill_value=fill_value,
                )

                # stack the 2D interpolated frame along the rotation axis,
                # taking into account the flip of the
                # detector Y axis (pointing down) compare to the laboratory frame
                # vertical axis (pointing up)
                interp_array[:, number_y - (idx + 1), :] = temp_array
                sys.stdout.write(
                    "\rGridding progress: {:d}%".format(int((idx + 1) / number_y * 100))
                )
                sys.stdout.flush()

        end = time.time()
        print(
            "\nTime ellapsed for gridding data:",
            str(datetime.timedelta(seconds=int(end - start))),
        )
        return interp_array

    def init_paths(
        self,
        sample_name,
        scan_number,
        root_folder,
        save_dir,
        specfile_name=None,
        template_imagefile=None,
        data_dir=None,
        save_dirname="result",
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
         - None for all other beamlines

        :param template_imagefile: beamline-dependent template for the data files

         - ID01: 'data_mpx4_%05d.edf.gz' or 'align_eiger2M_%05d.edf.gz'
         - SIXS_2018: 'align.spec_ascan_mu_%05d.nxs'
         - SIXS_2019: 'spare_ascan_mu_%05d.nxs'
         - Cristal: 'S%d.nxs'
         - P10: '_master.h5'
         - NANOMAX: '%06d.h5'
         - 34ID: 'Sample%dC_ES_data_51_256_256.npz'

        :param data_dir: if None it will use the beamline default, otherwise it will
         look for the data directly into that directory
        :param save_dirname: name of the saving folder, by default 'save_dir/result/'
         will be created
        """
        # check some parameters
        if not isinstance(scan_number, int):
            raise TypeError("scan_number should be an integer")

        if not isinstance(sample_name, str):
            raise TypeError("sample_name should be a string")

        valid.valid_container(
            specfile_name,
            container_types=str,
            min_length=1,
            allow_none=True,
            name="specfile_name",
        )
        valid.valid_container(
            template_imagefile,
            container_types=str,
            min_length=0,
            allow_none=True,
            name="template_imagefile",
        )

        # check that the provided folder names are not an empty string
        valid.valid_container(
            save_dirname, container_types=str, min_length=1, name="save_dirname"
        )
        valid.valid_container(
            data_dir,
            container_types=str,
            min_length=1,
            allow_none=True,
            name="data_dir",
        )

        #############################################
        # create beamline-dependent path parameters #
        #############################################
        (
            homedir,
            default_dirname,
            specfile,
            template_imagefile,
        ) = self._beamline.init_paths(
            root_folder=root_folder,
            sample_name=sample_name,
            scan_number=scan_number,
            template_imagefile=template_imagefile,
            specfile_name=specfile_name,
        )

        # define the data directory
        if data_dir is not None:
            datadir = data_dir
        else:
            datadir = homedir + default_dirname
        if not datadir.endswith("/"):
            datadir += "/"

        # define and create the saving directory
        if save_dir:
            savedir = save_dir
        else:
            savedir = homedir + save_dirname + "/"
        if not savedir.endswith("/"):
            savedir += "/"

        # update the detector instance
        (
            self.detector.rootdir,
            self.detector.savedir,
            self.detector.datadir,
            self.detector.sample_name,
            self.detector.specfile,
            self.detector.template_imagefile,
        ) = (root_folder, savedir, datadir, sample_name, specfile, template_imagefile)

    def init_qconversion(self):
        """
        Initialize the qconv object for xrayutilities depending on the setup parameters.

        The convention in xrayutilities is x downstream, z vertical up, y outboard.
        Note: the user-defined motor offsets are applied directly when reading motor
        positions, therefore do not need to be taken into account in xrayutilities apart
        from the detector inplane offset determined by the area detector calibration.

        :return: a tuple containing:

         - the qconv object for xrayutilities
         - a tuple of motor offsets used later for q calculation

        """
        return self._beamline.init_qconversion(
            conversion_table=self.labframe_to_xrayutil,
            beam_direction=self.beam_direction_xrutils,
            offset_inplane=self.offset_inplane,
            diffractometer=self.diffractometer,
        )

    @staticmethod
    def interp_2dslice(
        array,
        slice_index,
        rotation_angle,
        direct_beam,
        interp_angle,
        interp_radius,
        fill_value,
    ):
        """
        Interpolate a 2D slice from a tomographic dataset onto cartesian coordinates.

        The initial 3D array is in cylindrical coordinates.

        :param array: 3D array of intensities measured in the detector frame
        :param slice_index: the index along the rotation axis of the 2D slice in array
         to interpolate
        :param rotation_angle: array, rotation angle values for the rocking scan
        :param direct_beam: position in pixels of the rotation pivot in the direction
         perpendicular to the rotation axis
        :param interp_angle: 2D array, polar angles for the interpolation in a plane
         perpendicular to the rotation axis
        :param interp_radius: 2D array, polar radii for the interpolation in a plane
         perpendicular to the rotation axis
        :param fill_value: real number (np.nan allowed), fill_value parameter for the
         RegularGridInterpolator
        :return: the interpolated slice, the slice index
        """
        valid.valid_ndarray(arrays=array, ndim=2)
        # position of the experimental data points
        number_x = array.shape[1]
        rgi = RegularGridInterpolator(
            (
                rotation_angle * np.pi / 180,
                np.arange(-direct_beam, -direct_beam + number_x, 1),
            ),
            array,
            method="linear",
            bounds_error=False,
            fill_value=fill_value,
        )

        # interpolate the data onto the new points
        tmp_array = rgi(
            np.concatenate(
                (
                    interp_angle.reshape((1, interp_angle.size)),
                    interp_radius.reshape((1, interp_angle.size)),
                )
            ).transpose()
        )
        tmp_array = tmp_array.reshape(interp_angle.shape)

        return tmp_array, slice_index

    def ortho_cdi(
        self,
        arrays,
        cdi_angle,
        fill_value=0,
        correct_curvature=False,
        debugging=False,
    ):
        """
        Interpolate forward CDI data in the laboratory frame.

        :param arrays: tuple of 3D arrays of the same shape (e.g.: reciprocal space
         diffraction pattern and mask), in the detector frame
        :param cdi_angle: 1D array of measurement angles in degrees
        :param fill_value: tuple of real numbers (np.nan allowed), fill_value parameter
         for the RegularGridInterpolator, same length as the number of arrays
        :param correct_curvature: bool, True to take into account the curvature of the
         Ewald sphere (uses griddata, very slow)
        :param debugging: bool, True to see more plots
        :return:
         - an array (if a single array was provided) or a tuple of arrays interpolated
           on an orthogonal grid (same length as the number of input arrays)
         - a tuple of three 1D arrays for the q values (qx, qz, qy) where qx is
           downstream, qz is vertical up and qy is outboard.
         - a tuple of two integersfor the corrected position of the direct beam (V, H)

        """
        #########################
        # check some parameters #
        #########################
        valid.valid_ndarray(arrays, ndim=3)
        nb_arrays = len(arrays)
        valid.valid_item(
            correct_curvature, allowed_types=bool, name="correct_curvature"
        )
        valid.valid_item(debugging, allowed_types=bool, name="debugging")
        if isinstance(fill_value, Real):
            fill_value = (fill_value,) * nb_arrays
        valid.valid_container(
            fill_value,
            container_types=(tuple, list, np.ndarray),
            length=nb_arrays,
            item_types=Real,
            name="fill_value",
        )

        #####################################################
        # recalculate the direct beam position with binning #
        #####################################################
        directbeam_y = int(
            (self.direct_beam[0] - self.detector.roi[0]) / self.detector.binning[1]
        )
        # vertical
        directbeam_x = int(
            (self.direct_beam[1] - self.detector.roi[2]) / self.detector.binning[2]
        )
        # horizontal
        print(
            "\nDirect beam for the ROI and binning (y, x):", directbeam_y, directbeam_x
        )

        #######################################
        # interpolate the diffraction pattern #
        #######################################
        if correct_curvature:
            arrays, q_values = self.transformation_cdi_ewald(
                arrays=arrays,
                direct_beam=(directbeam_y, directbeam_x),
                cdi_angle=cdi_angle,
                fill_value=fill_value,
            )
        else:
            arrays, q_values = self.transformation_cdi(
                arrays=arrays,
                direct_beam=(directbeam_y, directbeam_x),
                cdi_angle=cdi_angle,
                fill_value=fill_value,
                debugging=debugging,
            )
        return arrays, q_values, (directbeam_y, directbeam_x)

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
         - a numpy array of shape (3, 3): transformation matrix from the detector
           frame to the laboratory/crystal frame

        """
        if isinstance(arrays, np.ndarray):
            arrays = (arrays,)
        valid.valid_ndarray(arrays, ndim=3)
        nb_arrays = len(arrays)
        input_shape = arrays[0].shape
        # could be smaller than the shape used in phase retrieval,
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
        tilt = (
            self.tilt_angle
            * self.detector.preprocessing_binning[0]
            * self.detector.binning[0]
        )

        dz_realspace, dy_realspace, dx_realspace = self.voxel_sizes(
            initial_shape,
            tilt_angle=abs(tilt),
            pixel_x=self.detector.unbinned_pixel_size[1],
            pixel_y=self.detector.unbinned_pixel_size[0],
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
            tilt *= initial_shape[0] / input_shape[0]
            pixel_y = (
                self.detector.unbinned_pixel_size[0] * initial_shape[1] / input_shape[1]
            )
            pixel_x = (
                self.detector.unbinned_pixel_size[1] * initial_shape[2] / input_shape[2]
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
            pixel_y = self.detector.unbinned_pixel_size[0]
            pixel_x = self.detector.unbinned_pixel_size[1]

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
        transfer_matrix, _ = self.transformation_bcdi(
            array_shape=input_shape,
            tilt_angle=tilt,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            direct_space=True,
            verbose=verbose,
        )

        ################################################################################
        # calculate the rotation matrix from the crystal frame to the laboratory frame #
        ################################################################################
        # (inverse rotation to have reference_axis along q)
        rotation_matrix = util.rotation_matrix_3d(
            axis_to_align=reference_axis, reference_axis=q_com / np.linalg.norm(q_com)
        )

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
        return output_arrays, voxel_size, transfer_matrix

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
         - a numpy array of shape (3, 3): transformation matrix from the detector
           frame to the laboratory/crystal frame

        """
        valid.valid_ndarray(arrays, ndim=3)
        nb_arrays = len(arrays)
        nbz, nby, nbx = arrays[0].shape

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
        transfer_matrix, q_offset = self.transformation_bcdi(
            array_shape=(nbz, nby, nbx),
            tilt_angle=self.tilt_angle * self.detector.preprocessing_binning[0],
            direct_space=False,
            verbose=verbose,
            pixel_x=self.detector.unbinned_pixel_size[1],
            pixel_y=self.detector.unbinned_pixel_size[0],
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
        qx = np.arange(-nz_output // 2, nz_output // 2, 1) * dq_along_z
        # along z downstream
        qz = np.arange(-ny_output // 2, ny_output // 2, 1) * dq_along_y
        # along y vertical up
        qy = np.arange(-nx_output // 2, nx_output // 2, 1) * dq_along_x
        # along x outboard

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
        return output_arrays, (qx, qz, qy), transfer_matrix

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
            name="array_shape",
        )

        ortho_matrix, _ = self.transformation_bcdi(
            array_shape=array_shape,
            tilt_angle=tilt_angle,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            direct_space=True,
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

    def read_logfile(self, **kwargs):
        """
        Extract values of interest for the geometric transformation from the logfile.

        This is the public interface of Diffractometer.goniometer_values

        :param kwargs: beamline_specific parameters

         - 'scan_number': int, the scan number to load

        :return: a tuple of angular values in degrees (rocking angular step, grazing
         incidence angles, inplane detector angle, outofplane detector angle). The
         grazing incidence angles are the positions of circles below the rocking circle.
        """
        scan_number = kwargs.get("scan_number")
        valid.valid_item(
            scan_number,
            allowed_types=int,
            allow_none=True,
            min_included=0,
            name="scan_number",
        )

        return self.diffractometer.goniometer_values(
            setup=self,
            scan_number=scan_number,
        )

    def transformation_bcdi(
        self, array_shape, tilt_angle, pixel_x, pixel_y, direct_space, verbose=True
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

        # convert lengths to nanometers and angles to radians
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
        tilt = np.radians(tilt_angle)

        ###########################################################
        # calculate the transformation matrix in reciprocal space #
        ###########################################################
        mymatrix, q_offset = self._beamline.transformation_matrix(
            wavelength=wavelength,
            distance=distance,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            inplane=inplane,
            outofplane=outofplane,
            grazing_angle=grazing_angle,
            rocking_angle=self.rocking_angle,
            tilt=tilt,
            verbose=verbose,
        )

        ###############################################################
        # (optional) convert the tranformation matrix to direct space #
        ###############################################################
        if direct_space:  # length scale in nm
            # for a discrete FT, the dimensions of the basis vectors
            # after the transformation are related to the total
            # domain size
            mymatrix[:, 0] = array_shape[2] * mymatrix[:, 0]
            mymatrix[:, 1] = array_shape[1] * mymatrix[:, 1]
            mymatrix[:, 2] = array_shape[0] * mymatrix[:, 2]
            return 2 * np.pi * np.linalg.inv(mymatrix).transpose(), None

        # reciprocal length scale in  1/nm
        return mymatrix, q_offset

    def transformation_cdi(self, arrays, direct_beam, cdi_angle, fill_value, debugging):
        """
        Calculate the transformation matrix from detector frame to laboratory frame.

        For the transformation in direct space, the length scale is in nm,
        for the transformation in reciprocal space, it is in 1/nm.

        :param arrays: tuple of 3D arrays of the same shape (e.g.: reciprocal space
         diffraction pattern and mask), in the detector frame
        :param direct_beam: tuple of 2 integers, position of the direction beam (V, H)
        :param cdi_angle: 1D array of measurement angles in degrees
        :param fill_value: tuple of real numbers (np.nan allowed), fill_value parameter
         for the RegularGridInterpolator, same length as the number of arrays
        :param debugging: bool, True to see more plots
        :return:

         - a tuple of arrays interpolated on an orthogonal grid (same length as the
           number of input arrays)
         - a tuple of three 1D arrays for the q values (qx, qz, qy) where qx is
           downstream, qz is vertical up and qy is outboard.

        """
        #########################
        # check some parameters #
        #########################
        valid.valid_ndarray(arrays, ndim=3)
        valid.valid_container(
            direct_beam,
            container_types=(tuple, list),
            length=2,
            item_types=int,
            name="direct_beam",
        )
        valid.valid_1d_array(
            cdi_angle,
            allowed_types=Real,
            allow_none=False,
            name="cdi_angle",
        )
        valid.valid_container(
            fill_value,
            container_types=(tuple, list),
            length=2,
            item_types=Real,
            name="fill_value",
        )
        valid.valid_item(debugging, allowed_types=bool, name="debugging")

        #########################
        # convert lengths to nm #
        #########################
        wavelength = self.wavelength * 1e9
        distance = self.distance * 1e9
        pixel_x = self.detector.pixelsize_x * 1e9
        # binned pixel size in the horizontal direction
        pixel_y = self.detector.pixelsize_y * 1e9
        # binned pixel size in the vertical direction
        lambdaz = wavelength * distance

        _, nby, nbx = arrays[0].shape
        directbeam_y, directbeam_x = direct_beam
        # calculate the number of voxels available to accomodate the gridded data
        # directbeam_x and directbeam_y already are already taking into account
        # the ROI and binning
        numx = 2 * max(directbeam_x, nbx - directbeam_x)
        # number of interpolated voxels in the plane perpendicular
        # to the rotation axis. It will accomodate the full data range.
        numy = nby  # no change of the voxel numbers along the rotation axis
        print("\nData shape after regridding:", numx, numy, numx)

        # update the direct beam position due to an eventual padding along X
        if nbx - directbeam_x < directbeam_x:
            pivot = directbeam_x
        else:  # padding to the left along x, need to correct the pivot position
            pivot = nbx - directbeam_x

        dqx = 2 * np.pi / lambdaz * pixel_x
        # in 1/nm, downstream, pixel_x is the binned pixel size
        dqz = 2 * np.pi / lambdaz * pixel_y
        # in 1/nm, vertical up, pixel_y is the binned pixel size
        dqy = 2 * np.pi / lambdaz * pixel_x
        # in 1/nm, outboard, pixel_x is the binned pixel size

        ##########################################
        # calculation of q based on P10 geometry #
        ##########################################
        qx = np.arange(-directbeam_x, -directbeam_x + numx, 1) * dqx
        # downstream, same direction as detector X rotated by +90deg
        qz = np.arange(directbeam_y - numy, directbeam_y, 1) * dqz
        # vertical up opposite to detector Y
        qy = np.arange(directbeam_x - numx, directbeam_x, 1) * dqy
        # outboard opposite to detector X
        print(
            "q spacing for the interpolation (z,y,x) = "
            f"({dqx:.6f}, {dqz:.6f},{dqy:.6f}) (1/nm)"
        )

        ##############################################################
        # loop over 2D slices perpendicular to the rotation axis     #
        # slower than doing a 3D interpolation but needs less memory #
        ##############################################################
        # find the corresponding polar coordinates of a cartesian 2D grid
        # perpendicular to the rotation axis
        interp_angle, interp_radius = self._beamline.cartesian2polar(
            nb_pixels=numx,
            pivot=pivot,
            offset_angle=cdi_angle.min(),
            debugging=debugging,
        )

        #################################################
        # Interpolate the data onto a cartesian 3D grid #
        #################################################
        output_arrays = []
        comment = ("data", "mask")
        for idx, array in enumerate(arrays):
            ortho_array = self.grid_cylindrical(
                array=array,
                rotation_angle=cdi_angle,
                direct_beam=directbeam_x,
                interp_angle=interp_angle,
                interp_radius=interp_radius,
                fill_value=fill_value[idx],
                comment=comment[idx],
            )
            output_arrays.append(ortho_array)

        return output_arrays, (qx, qz, qy)

    def transformation_cdi_ewald(self, arrays, direct_beam, cdi_angle, fill_value):
        """
        Interpolate forward CDI data considering the curvature of the Ewald sphere.

        :param arrays: tuple of 3D arrays of the same shape (e.g.: reciprocal space
         diffraction pattern and mask), in the detector frame
         :param direct_beam: tuple of 2 integers, position of the direction beam (V, H)
        :param cdi_angle: 1D array of measurement angles in degrees
        :param fill_value: tuple of real numbers (np.nan allowed), fill_value parameter
         for the RegularGridInterpolator, same length as the number of arrays
        :return:
        """
        nbz, nby, nbx = arrays[0].shape
        _, directbeam_x = direct_beam
        # calculate the number of voxels available to accomodate the gridded data
        # directbeam_x and directbeam_y already are already taking into account
        # the ROI and binning
        numx = 2 * max(directbeam_x, nbx - directbeam_x)
        # number of interpolated voxels in the plane perpendicular
        # to the rotation axis. It will accomodate the full data range.
        numy = nby  # no change of the voxel numbers along the rotation axis
        print("\nData shape after regridding:", numx, numy, numx)

        # calculate exact q values for each voxel of the 3D dataset
        old_qx, old_qz, old_qy = self._beamline.ewald_curvature_saxs(
            wavelength=self.wavelength * 1e9,
            pixelsize_x=self.detector.pixelsize_x * 1e9,
            pixelsize_y=self.detector.pixelsize_y * 1e9,
            distance=self.distance * 1e9,
            array_shape=(nbz, nby, nbx),
            cdi_angle=cdi_angle,
            direct_beam=direct_beam,
        )

        # create the grid for interpolation
        qx = np.linspace(
            old_qx.min(), old_qx.max(), numx, endpoint=False
        )  # z downstream
        qz = np.linspace(
            old_qz.min(), old_qz.max(), numy, endpoint=False
        )  # y vertical up
        qy = np.linspace(old_qy.min(), old_qy.max(), numx, endpoint=False)  # x outboard

        new_qx, new_qz, new_qy = np.meshgrid(qx, qz, qy, indexing="ij")

        ###########################################################
        # interpolate the data onto the new points using griddata #
        # (the original grid is not regular, very slow)           #
        ###########################################################
        print("Interpolating the data using griddata, will take time...")
        output_arrays = []
        for idx, array in enumerate(arrays):
            # convert array type to float,
            # for integers the interpolation can lead to artefacts
            array = array.astype(float)
            ortho_array = griddata(
                np.array(
                    [
                        np.ndarray.flatten(old_qx),
                        np.ndarray.flatten(old_qz),
                        np.ndarray.flatten(old_qy),
                    ]
                ).T,
                np.ndarray.flatten(array),
                np.array(
                    [
                        np.ndarray.flatten(new_qx),
                        np.ndarray.flatten(new_qz),
                        np.ndarray.flatten(new_qy),
                    ]
                ).T,
                method="linear",
                fill_value=fill_value[idx],
            )
            ortho_array = ortho_array.reshape((numx, numy, numx))
            output_arrays.append(ortho_array)
        return output_arrays, (qx, qz, qy)

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
            name="array_shape",
        )

        transfer_matrix, _ = self.transformation_bcdi(
            array_shape=array_shape,
            tilt_angle=tilt_angle,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            direct_space=True,
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
