# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

"""
Setup class that defines the experimental geometry.

You can think of it as the public interface for the Beamline and Diffractometer child
classes. A script would call a method from Setup, which would then retrieve the required
beamline-dependent information from the child classes.
"""
import datetime
import logging
import multiprocessing as mp
import time
from numbers import Integral, Real
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.interpolate import RegularGridInterpolator, griddata

import bcdi.utils.format as fmt
from bcdi.experiment.beamline import create_beamline
from bcdi.experiment.beamline_factory import BeamlineGoniometer, BeamlineSaxs
from bcdi.experiment.detector import Detector, create_detector
from bcdi.graph import graph_utils as gu
from bcdi.utils import utilities as util
from bcdi.utils import validation as valid
from bcdi.utils.io_helper import ContextFile

module_logger = logging.getLogger(__name__)


def get_mean_tilt(
    angles: Optional[Union[float, int, np.ndarray, List[Union[float, int]]]]
) -> Optional[float]:
    """
    Calculate the mean tilt depending on the array of incident angles.

    E.g., for input angles of [0, 0.25, 0.5, 0.75], the mean tilt is 0.25.
    """
    if angles is None:
        return angles
    if isinstance(angles, list):
        angles = np.asarray(angles)
    if isinstance(angles, (float, int)) or (
        isinstance(angles, np.ndarray) and angles.size == 1
    ):
        return float(angles)
    if isinstance(angles, np.ndarray) and angles.size > 1:
        return float(np.mean(angles[1:] - angles[0:-1]))
    raise TypeError(f"tilt_angle should be a ndarray, got {type(angles)}")


class Setup:
    """
    Class for defining the experimental geometry.

    :param parameters: dictionary of parameters
    :param scan_index: index of the scan to analyze
    :param kwargs:
     - 'logger': an optional logger

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
        parameters: Dict[str, Any],
        scan_index: int = 0,
        **kwargs,
    ):
        self.logger = kwargs.get("logger", module_logger)
        self.parameters = parameters
        self.scan_index = scan_index

        # create the detector instance
        self.detector_name = self.parameters.get("detector", "Dummy")
        self.detector = create_detector(
            name=self.detector_name,
            roi=self.parameters.get("roi_detector"),
            sum_roi=self.parameters.get("sum_roi"),
            binning=self.parameters.get("phasing_binning", (1, 1, 1)),
            preprocessing_binning=self.parameters.get(
                "preprocessing_binning", (1, 1, 1)
            ),
            offsets=self.parameters.get("sample_offsets"),
            linearity_func=self.parameters.get("linearity_func"),
            logger=self.logger,
        )

        # create the beamline instance
        self.beamline_name = self.parameters["beamline"]
        self.beamline = create_beamline(
            name=self.beamline_name,
            sample_offsets=self.parameters.get("sample_offsets"),
            logger=self.logger,
        )

        # load positional arguments corresponding to instance properties
        self.beam_direction = self.parameters.get("beam_direction", (1, 0, 0))
        self.energy = self.parameters.get("energy")
        self.distance = self.parameters.get("detector_distance")
        self.outofplane_angle = self.parameters.get("outofplane_angle")
        self.inplane_angle = self.parameters.get("inplane_angle")
        self.tilt_angle = self.parameters.get("tilt_angle")
        self.rocking_angle = self.parameters.get("rocking_angle")
        self.grazing_angle = self.parameters.get("grazing_angle")

        # parameters for  loading and preprocessing data
        self.dirbeam_detector_angles = self.parameters.get("dirbeam_detector_angles")
        self.dirbeam_detector_position = self.parameters.get(
            "dirbeam_detector_position"
        )
        self.direct_beam = self.parameters.get("direct_beam")
        self.filtered_data = self.parameters.get("filtered_data", False)  # boolean
        self.custom_scan = self.parameters.get("custom_scan", False)  # boolean
        self.custom_images = self.parameters.get("custom_images")  # list or tuple
        self.custom_monitor = self.parameters.get("custom_monitor")  # list or tuple
        self.custom_motors = self.parameters.get("custom_motors")  # dictionnary
        self.actuators = self.parameters.get("actuators", {})  # list or tuple

        # parameters for xrayutilities
        self.sample_inplane = self.parameters.get("sample_inplane", (1, 0, 0))
        self.sample_outofplane = self.parameters.get("sample_outofplane", (0, 0, 1))
        self.offset_inplane = self.parameters.get("offset_inplane", 0)

        # parameters for series (several frames per point) at P10
        self.is_series = self.parameters.get("is_series", False)  # boolean

        # initialize other attributes
        self.logfile: Optional[ContextFile] = None
        self.detector_position: Optional[Tuple[Real, Real, Real]] = None
        self.tilt_angles: Optional[np.ndarray] = None
        self.frames_logical: Optional[np.ndarray] = None

        # initialize the paths and the logfile
        self.initialize_analysis()

    def initialize_analysis(self) -> None:
        """Initialize the paths, logfile and load motor positions."""
        self.init_paths(
            sample_name=self.parameters["sample_name"][self.scan_index],
            scan_number=self.scan_nb,
            data_dir=self.parameters["data_dir"][self.scan_index],
            root_folder=self.parameters["root_folder"],
            save_dir=self.parameters["save_dir"][self.scan_index],
            save_dirname=self.parameters["save_dirname"],
            specfile_name=self.parameters["specfile_name"][self.scan_index],
            template_imagefile=self.parameters["template_imagefile"][self.scan_index],
        )
        self.create_logfile(
            scan_number=self.scan_nb,
            root_folder=self.parameters["root_folder"],
            filename=self.detector.specfile,
        )
        self.read_logfile(scan_number=self.scan_nb)

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
    def beam_direction(self) -> np.ndarray:
        """
        Direction of the incident X-ray beam.

        Frame convention: (z downstream, y vertical up, x outboard).
        """
        return self._beam_direction

    @beam_direction.setter
    def beam_direction(self, value: List[float]) -> None:
        valid.valid_container(
            value,
            container_types=(tuple, list, np.ndarray),
            length=3,
            item_types=Real,
            name="Setup.beam_direction",
        )
        value_as_array = np.asarray(value)
        if np.linalg.norm(value_as_array) == 0:
            raise ValueError(
                "At least of component of beam_direction should be non null."
            )
        self._beam_direction = value_as_array / float(np.linalg.norm(value_as_array))

    @property
    def beam_direction_xrutils(self) -> np.ndarray:
        """
        Direction of the incident X-ray beam in xrayutilities frame.

        xrayutilities frame convention: (x downstream, y outboard, z vertical up).
        """
        u, v, w = self._beam_direction  # (u downstream, v vertical up, w outboard)
        return np.array([u, w, v])

    @property
    def name(self):
        """Name of the beamline."""
        return self.beamline.name

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
        return self.labframe_to_xrayutil[self.beamline.detector_hor]

    @property
    def detector_ver_xrutil(self):
        """
        Convert the detector vertical orientation to xrayutilities frame.

        The laboratory frame convention is (z downstream, y vertical, x outboard).
        The frame convention of xrayutilities is (x downstream, y outboard,
        z vertical up).

        :return: "z+" or "z-" depending on the detector vertical orientation
        """
        return self.labframe_to_xrayutil[self.beamline.detector_ver]

    @property
    def diffractometer(self):
        """Public interface to access the diffractometer instance."""
        return self.beamline.diffractometer

    @property
    def dirbeam_detector_angles(self):
        """
        Detector angles in degrees for the measurement of the direct beam.

        [outofplane, inplane]
        """
        return self._dirbeam_detector_angles

    @dirbeam_detector_angles.setter
    def dirbeam_detector_angles(self, value):
        if value is not None:
            valid.valid_container(
                value,
                container_types=(tuple, list),
                length=2,
                item_types=Real,
                allow_none=True,
                name="Setup.dirbeam_detector_angles",
            )
        self._dirbeam_detector_angles = value

    @property
    def dirbeam_detector_position(self):
        """
        Detector position for the measurement of the direct beam.

        [z, y, x] in the laboratory frame
        """
        return self._dirbeam_detector_position

    @dirbeam_detector_position.setter
    def dirbeam_detector_position(self, value):
        if value is not None:
            valid.valid_container(
                value,
                container_types=(tuple, list),
                length=3,
                item_types=Real,
                allow_none=True,
                name="Setup.dirbeam_detector_position",
            )
        self._dirbeam_detector_position = value

    @property
    def direct_beam(self):
        """
        Direct beam position in pixels.

        Tuple of two real numbers indicating the position of the direct beam in pixels.
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
    def exit_wavevector(self) -> np.ndarray:
        """
        Calculate the exit wavevector kout.

        It uses the setup instance parameters. kout is expressed in 1/m in the
        laboratory frame (z downstream, y vertical, x outboard).

        :return: kout vector
        """
        if self.inplane_angle is None or self.outofplane_angle is None:
            raise ValueError("detector angles are None")
        if self.wavelength is None:
            raise ValueError("wavelength is None")
        return self.beamline.exit_wavevector(  # type: ignore
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
    def frames_logical(self) -> Optional[np.ndarray]:
        """
        Specify invalid frames using a logical array.

        1D array of length equal to the number of measured frames.
        In case of cropping the length of the stack of frames changes. A frame whose
        index is set to 1 means that it is used, 0 means not used.
        """
        return self._frames_logical

    @frames_logical.setter
    def frames_logical(self, value: Optional[np.ndarray]) -> None:
        valid.valid_1d_array(
            value,
            allowed_types=Integral,
            allow_none=True,
            allowed_values=(-1, 0, 1),
            name="frames_logical",
        )
        self._frames_logical = value
        if self._frames_logical is not None:
            self.apply_frames_logical()

    @property
    def grazing_angle(self):
        """
        Motor positions for the goniometer circles below the rocking angle.

        None if there is no such circle.
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
    def incident_wavevector(self) -> np.ndarray:
        """
        Calculate the incident wavevector kout.

        It uses the setup instance parameters. kin is expressed in 1/m in the
        laboratory frame (z downstream, y vertical, x outboard).

        :return: kin vector
        """
        if self.wavelength is None:
            raise ValueError("wavelength is None")
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
        return self.beamline.inplane_coeff()

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
    def loader(self):
        """Public interface to access the beamline data loader instance."""
        return self.beamline.loader

    @property
    def outofplane_angle(self):
        """Vertical detector angle, in degrees."""
        return self._outofplane_angle

    @outofplane_angle.setter
    def outofplane_angle(self, value):
        if not isinstance(value, (Real, np.ndarray)) and value is not None:
            raise TypeError(
                "outofplane_angle should be a number in degrees "
                "or an array of numbers in degrees"
            )
        self._outofplane_angle = value

    @property
    def outofplane_coeff(self):
        """
        Expose the outofplane_coeff beamline property to the outer world.

        Return a coefficient +/- 1 depending on the detector out of plane rotation
        direction and the detector out of plane orientation.

        :return: +1 or -1
        """
        return self.beamline.outofplane_coeff()

    @property
    def params(self):
        """Return a dictionnary with all parameters."""
        return {
            "Class": self.__class__.__name__,
            "beamline": self.beamline.name,
            "detector": self.detector.name,
            "direct_beam": self.direct_beam,
            "dirbeam_detector_angles": self.dirbeam_detector_angles,
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
    def q_laboratory(self) -> Optional[np.ndarray]:
        """
        Calculate the diffusion vector in the laboratory frame.

        Frame convention: (z downstream, y vertical up, x outboard). The unit is 1/A.

        :return: ndarray of three vectors components.
        """
        if self.exit_wavevector.ndim > 1:  # energy scan
            return None
        q_laboratory = (self.exit_wavevector - self.incident_wavevector) * 1e-10
        if np.isclose(np.linalg.norm(q_laboratory), 0, atol=1e-15):
            raise ValueError("q_laboratory is null")
        return q_laboratory  # type: ignore

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
    def scan_nb(self) -> int:
        return int(self.parameters["scans"][self.scan_index])

    @property
    def tilt_angle(self):
        """Angular step of the rocking curve, in degrees."""
        return self._tilt_angle

    @tilt_angle.setter
    def tilt_angle(self, value: Union[float, int]) -> None:
        if not isinstance(value, (float, int)) and value is not None:
            raise TypeError("tilt_angle should be a number in degrees")
        self._tilt_angle = value

    @property
    def wavelength(self) -> Optional[float]:
        """Wavelength in meters."""
        if isinstance(self.energy, Real):
            return 12.398 * 1e-7 / float(self.energy)  # in m
        return None

    def __repr__(self):
        """Representation string of the Setup instance."""
        return fmt.create_repr(self, Setup)

    def calc_qvalues_xrutils(self, hxrd, nb_frames, **kwargs):
        """
        Calculate the 3D q values of the BCDI scan using xrayutilities.

        :param hxrd: an initialized xrayutilities HXRD object used for the
         orthogonalization of the dataset
        :param nb_frames: length of axis 0 in the 3D dataset. If the data was cropped
         or padded, it may be different from the original length len(frames_logical)
        :param kwargs:

         - 'scan_number': the scan number to load

        :return:
         - qx, qz, qy components for the dataset. xrayutilities uses the xyz crystal
           frame: for incident angle = 0, x is downstream, y outboard, and z vertical
           up. The output of hxrd.Ang2Q.area is qx, qy, qz is this order. If q values
           seem wrong, check if diffractometer angles have default values set at 0,
           otherwise use the parameter setup.diffractometer.sample_offsets to correct it
         - updated frames_logical

        """
        # check some parameters
        valid.valid_1d_array(
            self.frames_logical,
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
        processed_positions = self.beamline.process_positions(
            setup=self,
            nb_frames=nb_frames,
            scan_number=scan_number,
            frames_logical=self.frames_logical,
        )

        # calculate q values
        qx, qy, qz = hxrd.Ang2Q.area(
            *processed_positions[:-1],
            en=processed_positions[-1],
            delta=self.detector.offsets,
        )
        self.logger.info(
            "Use the parameter 'sample_offsets' to correct diffractometer values."
        )
        return qx, qz, qy, self.frames_logical

    def apply_frames_logical(self) -> None:
        """Crop setup attributes where data frames have been excluded."""
        if isinstance(self.energy, np.ndarray):
            self.energy = util.apply_logical_array(
                arrays=self.energy,
                frames_logical=self.frames_logical,
            )
            print("energy")
        if isinstance(self.outofplane_angle, np.ndarray):
            self.outofplane_angle = util.apply_logical_array(
                arrays=self.outofplane_angle,
                frames_logical=self.frames_logical,
            )
            print("outofplane_angle")
        if isinstance(self.tilt_angles, np.ndarray):
            self.tilt_angles = np.asarray(
                util.apply_logical_array(
                    arrays=self.tilt_angles,
                    frames_logical=self.frames_logical,
                )
            )

    def check_setup(
        self,
        grazing_angle: Optional[Tuple[Real, ...]],
        inplane_angle: Real,
        outofplane_angle: Union[Real, np.ndarray],
        tilt_angle: np.ndarray,
        detector_distance: Real,
        energy: Union[Real, np.ndarray],
    ) -> None:
        """
        Check if the required parameters are correctly defined.

        This method is called in Diffractometer.goniometer_value, which is used only for
        the geometric transformation using the linearized transformation matrix. Hence,
        arrays for detector angles and the energy are not allowed.

        :param grazing_angle: tuple of motor positions for the goniometer circles below
         the rocking angle. Leave None if there is no such circle.
        :param inplane_angle: detector inplane angle in degrees
        :param outofplane_angle: detector out-of-plane angle in degrees
        :param tilt_angle: ndarray of shape (N,), values of the rocking angle
        :param detector_distance: sample to detector distance in meters
        :param energy: X-ray energy in eV
        """
        self.grazing_angle = grazing_angle

        self.energy = self.energy or energy
        # the user-defined energy overrides the logged energy
        if self.energy is None:
            raise ValueError("the X-ray energy is not defined")

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

        self.tilt_angles = tilt_angle
        self.tilt_angle = self.tilt_angle or get_mean_tilt(tilt_angle)
        if self.tilt_angle is None:
            raise ValueError("the tilt angle is not defined")
        if not isinstance(self.tilt_angle, Real):
            raise TypeError("the tilt angle should be a number")

    def check_setup_cdi(
        self,
        grazing_angle: Optional[Tuple[Real, ...]],
        detector_position: Tuple[Real, Real, Real],
        tilt_angle: np.ndarray,
        detector_distance: Real,
        energy: Real,
    ) -> None:
        """
        Check if the required parameters are correctly defined.

        This method is called in Diffractometer.goniometer_value, which is used only for
        the geometric transformation using the linearized transformation matrix. Hence,
        arrays for detector angles and the energy are not allowed.

        :param grazing_angle: tuple of motor positions for the goniometer circles below
         the rocking angle. Leave None if there is no such circle.
        :param detector_position: detector positions [det_z, det_y, det_x]
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

        self.detector_position = detector_position
        if self.detector_position is None:
            raise ValueError("the detector position is not defined")

        self.tilt_angles = tilt_angle
        if tilt_angle is not None:
            tilt_angle = np.mean(
                np.asarray(tilt_angle)[1:] - np.asarray(tilt_angle)[0:-1]
            )
        self.tilt_angle = self.tilt_angle or tilt_angle
        if self.tilt_angle is None:
            raise ValueError("the tilt angle is not defined")
        if not isinstance(self.tilt_angle, Real):
            raise TypeError("the tilt angle should be a number")

    def correct_detector_angles(
        self,
        bragg_peak_position: Optional[List[int]],
        verbose: bool = True,
    ) -> None:
        """
        Correct the detector angles given the direct beam position.

        The detector angles for the direct beam measurement can be non-zero.

        :param bragg_peak_position: [vertical, horizontal] position of the Bragg peak
         in the unbinned, full detector
        :param verbose: True to self.logger.info more comments
        """
        # check parameters
        if self.direct_beam is None:
            self.logger.info(
                f"'direct_beam' is {self.direct_beam}, can't correct detector angles."
            )
            return
        if self.dirbeam_detector_angles is None:
            self.logger.info(
                f"'dirbeam_detector_angles' is {self.dirbeam_detector_angles}, "
                "can't correct detector angles."
            )
            return
        if any(
            val is None
            for val in {self.inplane_angle, self.outofplane_angle, self.distance}
        ):
            raise ValueError("call setup.read_logfile before calling this method")

        if bragg_peak_position is None:
            self.logger.info(
                "Bragg peak position not defined, can't correct detector angles"
            )
            return

        if len(bragg_peak_position) == 3:
            bragg_peak_position = bragg_peak_position[-2:]
        valid.valid_container(
            bragg_peak_position,
            container_types=(tuple, list, np.ndarray),
            item_types=Real,
            length=2,
            name="bragg_peak_position",
        )
        valid.valid_item(verbose, allowed_types=bool, name="verbose")

        if verbose:
            self.logger.info(
                f"Direct beam at (inplane {self.dirbeam_detector_angles[1]} deg, "
                f"out-of-plane {self.dirbeam_detector_angles[0]} deg)"
                f"(X, Y): {self.direct_beam[1]}, {self.direct_beam[0]}"
            )
            self.logger.info(
                f"Detector angles before correction: inplane {self.inplane_angle:.2f}"
                f" deg, outofplane {self.outofplane_angle:.2f} deg"
            )

        self.inplane_angle = (
            self.inplane_angle
            + self.inplane_coeff
            * (
                self.detector.unbinned_pixel_size[1]
                / self.distance
                * 180
                / np.pi
                * (bragg_peak_position[1] - self.direct_beam[1])
            )
            - self.dirbeam_detector_angles[1]
        )

        self.outofplane_angle = (
            self.outofplane_angle
            - self.outofplane_coeff
            * self.detector.unbinned_pixel_size[0]
            / self.distance
            * 180
            / np.pi
            * (bragg_peak_position[0] - self.direct_beam[0])
            - self.dirbeam_detector_angles[0]
        )

        if verbose:
            self.logger.info(
                f"Corrected detector angles: inplane {self.inplane_angle:.2f} deg, "
                f"outofplane {self.outofplane_angle:.2f} deg"
            )

    def correct_direct_beam(self) -> Optional[Tuple[Real, ...]]:
        """
        Calculate the direct beam position in pixels at zero detector angles.

        :return: a tuple representing the direct beam position at zero detector angles
        """
        if self.direct_beam is None:
            self.logger.info("direct beam position not defined")
            return None

        if self.dirbeam_detector_angles is None:
            self.logger.info(
                "detector angles for the direct beam measurement not defined"
            )
            return tuple(self.direct_beam)

        ver_direct = (
            self.direct_beam[0]
            - self.outofplane_coeff
            * self.dirbeam_detector_angles[0]
            * np.pi
            / 180
            * self.distance
            / self.detector.unbinned_pixel_size[0]
        )  # outofplane_coeff is +1 or -1

        hor_direct = self.direct_beam[1] + self.inplane_coeff * (
            self.dirbeam_detector_angles[1]
            * np.pi
            / 180
            * self.distance
            / self.detector.unbinned_pixel_size[1]
        )  # inplane_coeff is +1 or -1

        return ver_direct, hor_direct

    def create_logfile(self, scan_number: int, root_folder: str, filename: str) -> None:
        """
        Create the logfile, which can be a log/spec file or the data itself.

        The nature of this file is beamline dependent.

        :param scan_number: the scan number to load
        :param root_folder: the root directory of the experiment, where is the
         specfile/.fio file
        :param filename: the file name to load, or the absolute path of
         'alias_dict.txt' for SIXS
        """
        self.logfile = (
            None
            if self.custom_scan
            else self.loader.create_logfile(
                datadir=self.detector.datadir,
                name=self.beamline.name,
                scan_number=scan_number,
                root_folder=root_folder,
                filename=filename,
                template_imagefile=self.detector.template_imagefile,
            )
        )

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

    def ewald_curvature_saxs(
        self,
        array_shape,
        cdi_angle,
    ):
        """
        Calculate q values taking into account the curvature of Ewald sphere.

        Based on the CXI detector geometry convention: Laboratory frame: z downstream,
        y vertical up, x outboard. Detector axes: Y vertical and X horizontal.

        :param array_shape: tuple of three integers, shape of the dataset to be gridded
        :param cdi_angle: 1D array of measurement angles in degrees
        :return: qx, qz, qy values in the laboratory frame
         (downstream, vertical up, outboard). Each array has the shape: nb_pixel_x *
         nb_pixel_y * nb_angles
        """
        distance = self.distance * 1e9
        pixelsize_x = self.detector.pixelsize_x * 1e9
        pixelsize_y = self.detector.pixelsize_y * 1e9
        wavelength = self.wavelength * 1e9
        #########################
        # check some parameters #
        #########################
        valid.valid_container(
            array_shape,
            container_types=(tuple, list),
            length=3,
            item_types=int,
            min_excluded=1,
            name="array_shape",
        )
        valid.valid_1d_array(
            cdi_angle,
            allowed_types=Real,
            allow_none=False,
            name="cdi_angle",
        )

        if not np.array_equal(self.beam_direction, np.array([1, 0, 0])):
            raise NotImplementedError(
                "Only the geometry with the beam along downstream is implemented, "
                f"'beam_direction'={self.beam_direction}"
            )

        # initialize arrays for q values
        nbz, nby, nbx = array_shape
        q_downstream = np.empty((nbz, nby, nbx), dtype=float)
        q_vertical = np.empty((nbz, nby, nbx), dtype=float)
        q_outboard = np.empty((nbz, nby, nbx), dtype=float)

        #########################################################
        # calculate the index range relative to the direct beam #
        #########################################################
        offseted_direct_beam_y, offseted_direct_beam_x = self.get_offseted_beam(
            detector_offsets=self.get_detector_offset()
        )

        ##########################################################################
        # calculate q values of the detector frame for each angle and stack them #
        ##########################################################################
        for idx, item in enumerate(cdi_angle):
            angle = item * np.pi / 180

            # rotation matrix around y vertical up
            rotation_matrix = np.array(
                [
                    [
                        np.cos(angle),
                        0,
                        -self.beamline.orientation_lookup[
                            self.diffractometer.sample_circles[-1]
                        ]
                        * np.sin(angle),
                    ],
                    [0, 1, 0],
                    [
                        self.beamline.orientation_lookup[
                            self.diffractometer.sample_circles[-1]
                        ]
                        * np.sin(angle),
                        0,
                        np.cos(angle),
                    ],
                ]
            )

            # generate a grid with pixel indices relative to the direct beam
            myy, myx = np.meshgrid(
                np.linspace(
                    -offseted_direct_beam_y,
                    -offseted_direct_beam_y + nby,
                    num=nby,
                    endpoint=False,
                ),
                np.linspace(
                    -offseted_direct_beam_x,
                    -offseted_direct_beam_x + nbx,
                    num=nbx,
                    endpoint=False,
                ),
                indexing="ij",
            )

            # angle of the exit wavevector along the outboard direction
            two_theta = np.arctan(myx * pixelsize_x / distance)
            # angle of the exit wavevector along the vertical direction
            alpha_f = np.arctan(
                np.divide(
                    myy * pixelsize_y,
                    np.sqrt(distance**2 + np.power(myx * pixelsize_x, 2)),
                )
            )

            q_along_z = (
                2 * np.pi / wavelength * (np.cos(alpha_f) * np.cos(two_theta) - 1)
            )  # downstream
            q_along_x = (
                2 * np.pi / wavelength * (np.cos(alpha_f) * np.sin(two_theta))
            )  # outboard
            q_along_y = 2 * np.pi / wavelength * (np.sin(alpha_f))  # vertical up

            q_downstream[idx, :, :] = (
                rotation_matrix[0, 0] * q_along_x
                + rotation_matrix[0, 1] * q_along_y
                + rotation_matrix[0, 2] * q_along_z
            )
            q_vertical[idx, :, :] = (
                rotation_matrix[1, 0] * q_along_x
                + rotation_matrix[1, 1] * q_along_y
                + rotation_matrix[1, 2] * q_along_z
            )
            q_outboard[idx, :, :] = (
                rotation_matrix[2, 0] * q_along_x
                + rotation_matrix[2, 1] * q_along_y
                + rotation_matrix[2, 2] * q_along_z
            )
        center_slice = np.s_[nbz // 2, nby // 2, nbx // 2]
        q_center = np.sqrt(
            q_downstream[center_slice] ** 2
            + q_vertical[center_slice] ** 2
            + q_outboard[center_slice] ** 2
        )
        self.logger.info(f"q at the center of the ROI: {q_center:.2f} 1/nm")
        return (q_downstream, q_vertical, q_outboard), (
            offseted_direct_beam_y,
            offseted_direct_beam_x,
        )

    def get_detector_offset(self) -> Tuple[float, float]:
        """
        Calculate the offset in pixels in the detector frame.

        The offset is calculated by comparing the detector position for the direct beam
        measurement and the detector position during the BCDI data collection.

        :return: (offset_y, offset_x) in unbinned pixels in the detector frame.
        """
        if self.detector_position is None:
            raise ValueError("'detector_position' is None")
        delta = [
            val2 - val1
            for (val1, val2) in zip(
                self.dirbeam_detector_position, self.detector_position
            )
        ]  # [delta_z, delta_y, delta_x]
        if not np.isclose(delta[0], 0):
            self.logger.warning(
                "the detector moved along the beam, neglecting any detector tilt"
            )

        # convert to mm (factor 1000)
        delta_pixel_y = (
            delta[1]
            * self.detector.unbinned_pixel_size[0]
            * 1000
            * self.beamline.orientation_lookup[self.beamline.detector_ver]
            * self.beamline.orientation_lookup[self.diffractometer.detector_axes[1]]
        )
        # convert to mm (factor 1000)
        delta_pixel_x = (
            delta[2]
            * self.detector.unbinned_pixel_size[1]
            * 1000
            * self.beamline.orientation_lookup[self.beamline.detector_hor]
            * self.beamline.orientation_lookup[self.diffractometer.detector_axes[1]]
        )
        return delta_pixel_y, delta_pixel_x

    def get_offseted_beam(
        self, detector_offsets: Tuple[float, float] = (0, 0)
    ) -> Tuple[int, int]:
        """
        Calculate the position of the direct beam compared to the origin of the frame.

        It takes into account an eventual shift of the detector between the direct beam
        measurement and the detector position during the BCDI data collection, an
        eventual user-defined region of interest when loading the data, and binning.

        :param detector_offsets: a tuple of two floats indicating the offset in unbinned
         pixels due to the detector shift.
         Orientation convention: Y vertical down, X inboard
        :return: a tuple of int, position of the offseted direct beam compare to the
         origin of indices.
        """
        # vertical
        binning_y = self.detector.preprocessing_binning[1] * self.detector.binning[1]
        offseted_direct_beam_y = int(
            (self.direct_beam[0] - self.detector.roi[0] + detector_offsets[0])
            / binning_y
        )
        # horizontal
        binning_x = self.detector.preprocessing_binning[2] * self.detector.binning[2]
        offseted_direct_beam_x = int(
            (self.direct_beam[1] - self.detector.roi[2] + detector_offsets[1])
            / binning_x
        )
        self.logger.info(
            f"Direct beam (VxH) including detector shift {detector_offsets}, "
            f"region of interest {self.detector.roi} and "
            f"binning {binning_y, binning_x} : "
            f"({offseted_direct_beam_y},{offseted_direct_beam_x})"
        )
        return offseted_direct_beam_y, offseted_direct_beam_x

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
            self.logger.info(
                f"Gridding {comment}, number of processors used: "
                f"{min(mp.cpu_count(), number_y)}"
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
            self.logger.info(f"Gridding {comment}, no multiprocessing")
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

        end = time.time()
        self.logger.info(
            "Time ellapsed for gridding data: "
            f"{str(datetime.timedelta(seconds=int(end - start)))}"
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
            min_length=0,
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
        ) = self.beamline.loader.init_paths(
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
        if not isinstance(self.beamline, BeamlineGoniometer):
            raise TypeError(
                "init_qconversion supports only for beamlines with goniometer, "
                f"got {type(self.beamline)}"
            )
        return self.beamline.init_qconversion(
            conversion_table=self.labframe_to_xrayutil,
            beam_direction=self.beam_direction_xrutils,
            offset_inplane=self.offset_inplane,
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

        #######################################
        # interpolate the diffraction pattern #
        #######################################
        if correct_curvature:
            (
                arrays,
                q_values,
                offseted_direct_beam,
            ) = self.transformation_cdi_ewald(
                arrays=arrays,
                cdi_angle=cdi_angle,
                fill_value=fill_value,
            )
        elif self.beamline_name != "P10_SAXS":
            raise NotImplementedError("Method implemented only for P10 USAXS setup")
        else:
            arrays, q_values, offseted_direct_beam = self.transformation_cdi(
                arrays=arrays,
                cdi_angle=cdi_angle,
                fill_value=fill_value,
                debugging=debugging,
            )
        return arrays, q_values, offseted_direct_beam

    def ortho_directspace(
        self,
        arrays: Union[np.ndarray, Tuple[np.ndarray, ...]],
        q_bragg: np.ndarray,
        initial_shape: Optional[Tuple[int, int, int]] = None,
        voxel_size: Optional[Tuple[float, float, float]] = None,
        fill_value: Tuple[float, ...] = (0,),
        reference_axis: Union[np.ndarray, Tuple[int, int, int]] = (0, 1, 0),
        verbose: bool = True,
        debugging: bool = False,
        **kwargs,
    ) -> Tuple[
        Union[np.ndarray, List[np.ndarray]],
        Tuple[float, float, float],
        np.ndarray,
    ]:
        """
        Geometrical transformation in direct space, into the crystal frame.

        Interpolate arrays (direct space output of the phase retrieval) in the
        orthogonal reference frame where q_bragg is aligned onto the array axis
        reference_axis.

        :param arrays: tuple of 3D arrays of the same shape (output of the phase
         retrieval), in the detector frame
        :param q_bragg: array of 3 vector components for the q values of the center
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

         - 'cmap': str, name of the colormap
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
        current_shape = arrays[0].shape

        transfer_matrix, voxel_size = self.get_transfer_matrix_crystal_frame(
            current_shape=current_shape,  # type: ignore
            q_bragg=q_bragg,
            reference_axis=reference_axis,
            initial_shape=initial_shape,
            voxel_size=voxel_size,
            verbose=verbose,
        )
        output_arrays, voxel_size, transfer_matrix = self.interpolate_direct_space(
            arrays=arrays,
            current_shape=current_shape,
            transfer_matrix=transfer_matrix,
            voxel_size=voxel_size,
            fill_value=fill_value,
            verbose=verbose,
            debugging=debugging,
            **kwargs,
        )

        return output_arrays, voxel_size, transfer_matrix

    def ortho_directspace_labframe(
        self,
        arrays: Union[np.ndarray, Tuple[np.ndarray, ...]],
        initial_shape: Optional[Tuple[int, int, int]] = None,
        voxel_size: Optional[Tuple[float, float, float]] = None,
        fill_value: Tuple[float, ...] = (0,),
        verbose: bool = True,
        debugging: bool = False,
        **kwargs,
    ) -> Tuple[
        Union[np.ndarray, List[np.ndarray]],
        Tuple[float, float, float],
        np.ndarray,
    ]:
        """
        Geometrical transformation in direct space, into the laboratory frame.

        Interpolate arrays (direct space output of the phase retrieval) in the
        orthogonal laboratory frame.

        :param arrays: tuple of 3D arrays of the same shape (output of the phase
         retrieval), in the detector frame
        :param initial_shape: shape of the FFT used for phasing
        :param voxel_size: number or list of three user-defined voxel sizes for
         the interpolation, in nm. If a single number is provided, the voxel size
         will be identical in all directions.
        :param fill_value: tuple of real numbers, fill_value parameter for the
         RegularGridInterpolator, same length as the number of arrays
        :param verbose: True to have printed comments
        :param debugging: tuple of booleans of the same length as the number of
         input arrays, True to show plots before and after interpolation
        :param kwargs:

         - 'cmap': str, name of the colormap
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
        current_shape = arrays[0].shape

        transfer_matrix, voxel_size = self.get_transfer_matrix_labframe(
            current_shape=current_shape,  # type: ignore
            initial_shape=initial_shape,
            voxel_size=voxel_size,
            verbose=verbose,
        )
        output_arrays, voxel_size, transfer_matrix = self.interpolate_direct_space(
            arrays=arrays,
            current_shape=current_shape,
            transfer_matrix=transfer_matrix,
            voxel_size=voxel_size,
            fill_value=fill_value,
            verbose=verbose,
            debugging=debugging,
            **kwargs,
        )

        return output_arrays, voxel_size, transfer_matrix

    def get_transfer_matrix_labframe(
        self,
        current_shape: Tuple[int, int, int],
        initial_shape: Optional[Tuple[int, int, int]] = None,
        voxel_size: Optional[Tuple[float, float, float]] = None,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """
        Calculate the transformation matrix in direct space, in the laboratory frame.

        :param current_shape: shape of the output of phase retrieval. It could be
         smaller than the shape used in phase retrieval, if the object was cropped
         around the support.
        :param initial_shape: shape of the FFT used for phasing
        :param voxel_size: number or list of three user-defined voxel sizes for
         the interpolation, in nm. If a single number is provided, the voxel size
         will be identical in all directions.
        :param verbose: True to have printed comments
        :return:

         - the transformation matrix as a numpy array of shape (3, 3)
         - a tuple of 3 voxels size for the interpolated arrays

        """
        #########################
        # check some parameters #
        #########################
        if not initial_shape:
            initial_shape = current_shape
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
            pixel_x=self.detector.pixelsize_x,
            pixel_y=self.detector.pixelsize_y,
        )
        if verbose:
            self.logger.info(
                "Sampling in the laboratory frame (z, y, x): "
                f"({dz_realspace:.2f} nm,"
                f" {dy_realspace:.2f} nm,"
                f" {dx_realspace:.2f} nm)"
            )

        if current_shape != initial_shape:
            # recalculate the tilt and pixel sizes to accomodate a shape change
            tilt *= initial_shape[0] / current_shape[0]
            pixel_y = self.detector.pixelsize_y * initial_shape[1] / current_shape[1]
            pixel_x = self.detector.pixelsize_x * initial_shape[2] / current_shape[2]
            if verbose:
                self.logger.info(
                    "Tilt, pixel_y, pixel_x based on the shape of the cropped array: "
                    f"({tilt:.4f} deg, "
                    f"{pixel_y * 1e6:.2f} um, "
                    f"{pixel_x * 1e6:.2f} um)"
                )

            # sanity check, the direct space voxel sizes
            # calculated below should be equal to the original ones
            dz_realspace, dy_realspace, dx_realspace = self.voxel_sizes(
                current_shape, tilt_angle=abs(tilt), pixel_x=pixel_x, pixel_y=pixel_y
            )
            if verbose:
                self.logger.info(
                    "Sanity check, recalculated direct space voxel sizes (z, y, x): "
                    f"({dz_realspace:.2f} nm, "
                    f"{dy_realspace:.2f} nm, "
                    f"{dx_realspace:.2f} nm)"
                )
        else:
            pixel_y = self.detector.pixelsize_y
            pixel_x = self.detector.pixelsize_x

        if voxel_size is None:
            voxel_size = dz_realspace, dy_realspace, dx_realspace  # in nm
        else:
            if isinstance(voxel_size, Real):
                voxel_size = (voxel_size, voxel_size, voxel_size)
        valid.valid_container(
            voxel_size,
            container_types=tuple,
            min_excluded=0,
            length=3,
            name="voxel_size",
        )

        ######################################################################
        # calculate the transformation matrix based on the beamline geometry #
        ######################################################################
        transfer_matrix, _ = self.transformation_bcdi(
            array_shape=current_shape,
            tilt_angle=tilt,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            direct_space=True,
            verbose=verbose,
        )
        return transfer_matrix, voxel_size

    def get_transfer_matrix_crystal_frame(
        self,
        current_shape: Tuple[int, int, int],
        q_bragg: np.ndarray,
        reference_axis: Union[np.ndarray, Tuple[int, int, int]] = (0, 1, 0),
        initial_shape: Optional[Tuple[int, int, int]] = None,
        voxel_size: Optional[Tuple[float, float, float]] = None,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """
        Calculate the transformation matrix in direct space, in crystal frame.

        :param current_shape: shape of the output of phase retrieval. It could be
         smaller than the shape used in phase retrieval, if the object was cropped
         around the support.
        :param q_bragg: tuple of 3 vector components for the q values of the center
         of mass of the Bragg peak, expressed in an orthonormal frame x y z
        :param reference_axis: 3D vector along which q will be aligned, expressed in
         an orthonormal frame x y z
        :param initial_shape: shape of the FFT used for phasing
        :param voxel_size: number or list of three user-defined voxel sizes for
         the interpolation, in nm. If a single number is provided, the voxel size
         will be identical in all directions.
        :param verbose: True to have printed comments
        :return:

         - the transformation matrix as a numpy array of shape (3, 3)
         - a tuple of 3 voxels size for the interpolated arrays

        """
        #########################
        # check some parameters #
        #########################
        valid.valid_container(
            q_bragg,
            container_types=(tuple, list, np.ndarray),
            length=3,
            item_types=Real,
            name="q_bragg",
        )
        if np.linalg.norm(q_bragg) == 0:
            raise ValueError("q_bragg should be a non zero vector")
        valid.valid_container(
            reference_axis,
            container_types=(tuple, list, np.ndarray),
            length=3,
            item_types=Real,
            name="reference_axis",
        )
        if not any(
            (np.array(reference_axis) == val).all()
            for val in (np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]))
        ):
            raise NotImplementedError(
                "strain calculation along directions "
                "other than array axes is not implemented"
            )

        q_com = np.array(q_bragg)
        transfer_matrix, voxel_size = self.get_transfer_matrix_labframe(
            current_shape=current_shape,
            initial_shape=initial_shape,
            voxel_size=voxel_size,
            verbose=verbose,
        )
        ################################################################################
        # calculate the rotation matrix from the crystal frame to the laboratory frame #
        ################################################################################
        # (inverse rotation to have reference_axis along q)
        rotation_matrix = util.rotation_matrix_3d(
            axis_to_align=np.array(reference_axis),
            reference_axis=q_com / np.linalg.norm(q_com),
        )
        return np.matmul(rotation_matrix, transfer_matrix), voxel_size

    def interpolate_direct_space(
        self,
        arrays: Union[np.ndarray, Tuple[np.ndarray, ...]],
        current_shape: Tuple[int, ...],
        transfer_matrix: np.ndarray,
        voxel_size: Tuple[float, float, float],
        fill_value: Tuple[float, ...],
        verbose: bool,
        debugging: Union[bool, Tuple[bool, ...]],
        **kwargs,
    ) -> Tuple[
        Union[np.ndarray, List[np.ndarray]],
        Tuple[float, float, float],
        np.ndarray,
    ]:
        """
        Interpolate arrays using the transfer matrix.

        :param arrays: tuple of 3D arrays of the same shape (output of the phase
         retrieval), in the detector frame
        :param current_shape: shape of the output of phase retrieval. It could be
         smaller than the shape used in phase retrieval, if the object was cropped
         around the support.
        :param transfer_matrix: the transformation matrix, numpy array of shape (3, 3)
        :param voxel_size: number or list of three user-defined voxel sizes for
         the interpolation, in nm. If a single number is provided, the voxel size
         will be identical in all directions.
        :param fill_value: tuple of real numbers, fill_value parameter for the
         RegularGridInterpolator, same length as the number of arrays
        :param verbose: True to have printed comments
        :param debugging: tuple of booleans of the same length as the number of
         input arrays, True to show plots before and after interpolation
        :param kwargs:

         - 'cmap': str, name of the colormap
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
        #########################
        # check some parameters #
        #########################
        if isinstance(arrays, np.ndarray):
            arrays = (arrays,)
        valid.valid_ndarray(arrays, ndim=3)
        nb_arrays = len(arrays)
        if isinstance(debugging, bool):
            debugging = (debugging,) * nb_arrays
        if isinstance(fill_value, (float, int)):
            fill_value = (fill_value,) * nb_arrays
        if len(fill_value) == 1:
            fill_value *= nb_arrays
        valid.valid_container(
            fill_value,
            container_types=(tuple, list, np.ndarray),
            length=nb_arrays,
            item_types=Real,
            name="fill_value",
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
            np.arange(-current_shape[0] // 2, current_shape[0] // 2, 1),
            np.arange(-current_shape[1] // 2, current_shape[1] // 2, 1),
            np.arange(-current_shape[2] // 2, current_shape[2] // 2, 1),
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
            self.logger.info(
                "Calculating the shape of the output array "
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
                    np.arange(-current_shape[0] // 2, current_shape[0] // 2, 1),
                    np.arange(-current_shape[1] // 2, current_shape[1] // 2, 1),
                    np.arange(-current_shape[2] // 2, current_shape[2] // 2, 1),
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
                    width_z=kwargs.get("width_z"),
                    width_y=kwargs.get("width_y"),
                    width_x=kwargs.get("width_x"),
                    reciprocal_space=False,
                    is_orthogonal=False,
                    scale="linear",
                    title=title[idx] + " in detector frame",
                    cmap=kwargs.get("cmap", "turbo"),
                )

                gu.multislices_plot(
                    abs(ortho_array),
                    sum_frames=False,
                    width_z=kwargs.get("width_z"),
                    width_y=kwargs.get("width_y"),
                    width_x=kwargs.get("width_x"),
                    reciprocal_space=False,
                    is_orthogonal=True,
                    scale="linear",
                    title=title[idx] + " in crystal frame",
                    cmap=kwargs.get("cmap", "turbo"),
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
            pixel_x=self.detector.pixelsize_x,
            pixel_y=self.detector.pixelsize_y,
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
            self.logger.info(
                "Interpolating:"
                f"\n\tSampling in q in the laboratory frame (z*, y*, x*):    "
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
                self.logger.info(f"Aligning Q along {reference_axis} (x,y,z)")

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
                self.logger.info(
                    f"Sampling in q in the crystal frame (axis 0, axis 1, axis 2):  "
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
            self.logger.info(
                f"Initial shape = ({nbz},{nby},{nbx})\n"
                f"Output shape  = ({nz_output},{ny_output},{nx_output}) "
                f"(satisfying FFT shape requirements)"
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

        return self.beamline.goniometer_values(setup=self, scan_number=scan_number)

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
        if not isinstance(self.beamline, BeamlineGoniometer):
            raise TypeError(
                "transformation_bcdi supports only for beamlines with goniometer, "
                f"got {type(self.beamline)}"
            )
        if verbose:
            self.logger.info(
                f"out-of plane detector angle={self.outofplane_angle:.3f} deg, "
                f"inplane_angle={self.inplane_angle:.3f} deg"
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
        mymatrix, q_offset = self.beamline.transformation_matrix(
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

    def transformation_cdi(self, arrays, cdi_angle, fill_value, debugging):
        """
        Calculate the transformation matrix from detector frame to laboratory frame.

        For the transformation in direct space, the length scale is in nm,
        for the transformation in reciprocal space, it is in 1/nm.

        :param arrays: tuple of 3D arrays of the same shape (e.g.: reciprocal space
         diffraction pattern and mask), in the detector frame
        :param cdi_angle: 1D array of measurement angles in degrees
        :param fill_value: tuple of real numbers (np.nan allowed), fill_value parameter
         for the RegularGridInterpolator, same length as the number of arrays
        :param debugging: bool, True to see more plots
        :return:

         - a tuple of arrays interpolated on an orthogonal grid (same length as the
           number of input arrays)
         - a tuple of three 1D arrays for the q values (qx, qz, qy) where qx is
           downstream, qz is vertical up and qy is outboard.
         - a tuple of 2 floats: position of the direct beam after taking into accout the
           region of interest and binning.

        """
        if not isinstance(self.beamline, BeamlineSaxs):
            raise TypeError(
                "transformation_cdi supports only for SAXS beamlines, "
                f"got {type(self.beamline)}"
            )
        #########################
        # check some parameters #
        #########################
        valid.valid_ndarray(arrays, ndim=3)
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

        #########################################################
        # calculate the index range relative to the direct beam #
        #########################################################
        offseted_direct_beam_y, offseted_direct_beam_x = self.get_offseted_beam()

        ###########################################################################
        # calculate the number of voxels available to accomodate the gridded data #
        ###########################################################################
        numx = 2 * max(offseted_direct_beam_x, nbx - offseted_direct_beam_x)
        # number of interpolated voxels in the plane perpendicular
        # to the rotation axis. It will accomodate the full data range.
        numy = nby  # no change of the voxel numbers along the rotation axis
        self.logger.info(f"Data shape after regridding: ({numx},{numy},{numx})")

        # update the direct beam position due to an eventual padding along X
        if nbx - offseted_direct_beam_x < offseted_direct_beam_x:
            pivot = offseted_direct_beam_x
        else:  # padding to the left along x, need to correct the pivot position
            pivot = nbx - offseted_direct_beam_x

        dqx = 2 * np.pi / lambdaz * pixel_x
        # in 1/nm, downstream, pixel_x is the binned pixel size
        dqz = 2 * np.pi / lambdaz * pixel_y
        # in 1/nm, vertical up, pixel_y is the binned pixel size
        dqy = 2 * np.pi / lambdaz * pixel_x
        # in 1/nm, outboard, pixel_x is the binned pixel size

        ##########################################
        # calculation of q based on P10 geometry #
        ##########################################
        qx = np.arange(-offseted_direct_beam_x, -offseted_direct_beam_x + numx, 1) * dqx
        # downstream, same direction as detector X rotated by +90deg
        qz = np.arange(offseted_direct_beam_y - numy, offseted_direct_beam_y, 1) * dqz
        # vertical up opposite to detector Y
        qy = np.arange(offseted_direct_beam_x - numx, offseted_direct_beam_x, 1) * dqy
        # outboard opposite to detector X
        self.logger.info(
            "q spacing for the interpolation (z,y,x) = "
            f"({dqx:.6f}, {dqz:.6f}, {dqy:.6f}) (1/nm)"
        )

        ##############################################################
        # loop over 2D slices perpendicular to the rotation axis     #
        # slower than doing a 3D interpolation but needs less memory #
        ##############################################################
        # find the corresponding polar coordinates of a cartesian 2D grid
        # perpendicular to the rotation axis
        interp_angle, interp_radius = self.beamline.cartesian2polar(
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
                direct_beam=offseted_direct_beam_x,
                interp_angle=interp_angle,
                interp_radius=interp_radius,
                fill_value=fill_value[idx],
                comment=comment[idx],
            )
            output_arrays.append(ortho_array)

        return (
            output_arrays,
            (qx, qz, qy),
            (offseted_direct_beam_y, offseted_direct_beam_x),
        )

    def transformation_cdi_ewald(self, arrays, cdi_angle, fill_value):
        """
        Interpolate forward CDI data considering the curvature of the Ewald sphere.

        :param arrays: tuple of 3D arrays of the same shape (e.g.: reciprocal space
         diffraction pattern and mask), in the detector frame
        :param cdi_angle: 1D array of measurement angles in degrees
        :param fill_value: tuple of real numbers (np.nan allowed), fill_value parameter
         for the RegularGridInterpolator, same length as the number of arrays
        :return:
        """
        # calculate exact q values for each voxel of the 3D dataset
        (old_qx, old_qz, old_qy), offseted_direct_beam = self.ewald_curvature_saxs(
            array_shape=arrays[0].shape,
            cdi_angle=cdi_angle,
        )

        # calculate the number of voxels needed to accomodate the gridded data
        maxbins: List[int] = []
        for dim in (old_qx, old_qz, old_qy):
            maxstep = max(abs(np.diff(dim, axis=j)).max() for j in range(3))
            maxbins.append(int(abs(dim.max() - dim.min()) / maxstep))
        self.logger.info(
            f"Maximum number of bins based on the sampling in q: {maxbins}"
        )
        maxbins = util.smaller_primes(maxbins, maxprime=7, required_dividers=(2,))
        self.logger.info(
            f"Maximum number of bins based on the shape requirements for FFT: {maxbins}"
        )

        # create the grid for interpolation
        qx = np.linspace(old_qx.min(), old_qx.max(), maxbins[0], endpoint=False)
        # along z downstream
        qz = np.linspace(old_qz.min(), old_qz.max(), maxbins[1], endpoint=False)
        # along y vertical up
        qy = np.linspace(old_qy.min(), old_qy.max(), maxbins[2], endpoint=False)
        # along x outboard

        new_qx, new_qz, new_qy = np.meshgrid(qx, qz, qy, indexing="ij")

        ###########################################################
        # interpolate the data onto the new points using griddata #
        # (the original grid is not regular, very slow)           #
        ###########################################################
        self.logger.info("Interpolating the data using griddata, make some coffee...")
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
            ortho_array = ortho_array.reshape(maxbins)
            output_arrays.append(ortho_array)
        return output_arrays, (qx, qz, qy), offseted_direct_beam

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
            self.logger.info(
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
            self.logger.info(
                "(voxelsize_z, voxelsize_y, voxelsize_x) = "
                f"({voxel_z:.2f}, {voxel_y:.2f}, {voxel_x:.2f}) (1/nm)"
            )
        return voxel_z, voxel_y, voxel_x
