# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#   (c) 06/2021-present : DESY CFEL
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

"""
Beamline-related diffractometer classes.

These classes are not meant to be instantiated directly but via a Setup instance. The
methods in child classes have the same signature as in the base class. The available
diffractometers are:

- DiffractometerID01
- DiffractometerSIXS
- Diffractometer34ID
- DiffractometerP10
- DiffractometerCRISTAL
- DiffractometerNANOMAX

"""

from abc import ABC, abstractmethod
from functools import reduce
from numbers import Number, Real
import numpy as np

from bcdi.utils import utilities as util
from bcdi.utils import validation as valid
from .rotation_matrix import RotationMatrix


def create_diffractometer(beamline, sample_offsets):
    """
    Create a Diffractometer instance depending on the beamline.

    :param beamline: str, name of the beamline
    :param sample_offsets: list or tuple of angles in degrees, corresponding to
     the offsets of each of the sample circles (the offset for the most outer circle
     should be at index 0). The number of circles is beamline dependent. Convention:
     the sample offsets will be subtracted to measurement the motor values.
    :return:  the corresponding diffractometer instance
    """
    if beamline == "ID01":
        return DiffractometerID01(sample_offsets)
    if beamline in {"SIXS_2018", "SIXS_2019"}:
        return DiffractometerSIXS(sample_offsets)
    if beamline == "34ID":
        return Diffractometer34ID(sample_offsets)
    if beamline == "P10":
        return DiffractometerP10(sample_offsets)
    if beamline == "CRISTAL":
        return DiffractometerCRISTAL(sample_offsets)
    if beamline == "NANOMAX":
        return DiffractometerNANOMAX(sample_offsets)
    raise NotImplementedError(
        f"No diffractometer implemented for the beamline {beamline}"
    )


class Diffractometer(ABC):
    """
    Base class for defining diffractometers.

    The frame used is the laboratory frame with the CXI convention (z downstream,
    y vertical up, x outboard).

    :param sample_offsets: list or tuple of angles in degrees, corresponding to
     the offsets of each of the sample circles (the offset for the most outer circle
     should be at index 0). The number of circles is beamline dependent. Convention:
     the sample offsets will be subtracted to measurement the motor values.
    :param sample_circles: list of sample circles from outer to inner (e.g. mu eta
     chi phi), expressed using a valid pattern within {'x+', 'x-', 'y+', 'y-', 'z+',
     'z-'}. For example: ['y+' ,'x-', 'z-', 'y+']
    :param detector_circles: list of detector circles from outer to inner
     (e.g. gamma delta), expressed using a valid pattern within {'x+', 'x-', 'y+',
     'y-', 'z+', 'z-'}. For example: ['y+', 'x-']
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
        List of detector circles.

        The circles should be listed from outer to inner (e.g. gamma delta), expressed
        using a valid pattern within {'x+', 'x-', 'y+', 'y-', 'z+', 'z-'}. For
        example: ['y+' ,'x-', 'z-', 'y+']. Convention: CXI convention (z downstream,
        y vertical up, x outboard), + for a counter-clockwise rotation, - for a
        clockwise rotation.
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
                "Invalid circle value encountered in detector_circles,"
                f" valid are {self.valid_circles}"
            )
        self._detector_circles = list(value)

    @property
    def sample_circles(self):
        """
        List of sample circles.

        The sample circles should be listed from outer to inner (e.g. mu eta chi phi),
        expressed using a valid pattern within {'x+', 'x-', 'y+', 'y-', 'z+', 'z-'}. For
        example: ['y+' ,'x-', 'z-', 'y+']. Convention: CXI convention (z downstream,
        y vertical up, x outboard), + for a counter-clockwise rotation, - for a
        clockwise rotation.
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
                "Invalid circle value encountered in sample_circles,"
                f" valid are {self.valid_circles}"
            )
        self._sample_circles = list(value)

    @property
    def sample_offsets(self):
        """
        List or tuple of sample angular offsets in degrees.

        These angles correspond to the offsets of each f the sample circles (the
        offset for the most outer circle should be at index 0). Convention: the
        sample offsets will be subtracted to measurement the motor values.
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
        Add a circle to the list of circles.

        The most outer circle should be at index 0.

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
        Send all sample circles to zero degrees.

        Arrays are rotatedsuch that all circles of the sample stage are at their zero
        position.

        :param arrays: tuple of 3D real arrays of the same shape.
        :param voxel_size: tuple, voxel size of the 3D array in z, y, and x
         (CXI convention)
        :param angles: tuple of angular values in degrees, one for each circle
         of the sample stage
        :param q_com: diffusion vector of the center of mass of the Bragg peak,
         expressed in an orthonormal frame x y z
        :param rocking_angle: angle which is tilted during the rocking curve in
         {'outofplane', 'inplane'}
        :param central_angle: if provided, angle to be used in the calculation
         of the rotation matrix for the rocking angle. If None, it will be defined as
         the angle value at the middle of the rocking curve.
        :param fill_value: tuple of numeric values used in the RegularGridInterpolator
         for points outside of the interpolation domain. The length of the tuple
         should be equal to the number of input arrays.
        :param is_orthogonal: set to True is the frame is orthogonal, False otherwise.
         Used for plot labels.
        :param reciprocal_space: True if the data is in reciprocal space,
         False otherwise. Used for plot labels.
        :param debugging: tuple of booleans of the same length as the number
         of input arrays, True to see plots before and after rotation
        :param kwargs:

         - 'title': tuple of strings, titles for the debugging plots, same length as
           the number of arrays
         - 'scale': tuple of strings (either 'linear' or 'log'), scale for the
           debugging plots, same length as the number of arrays
         - width_z: size of the area to plot in z (axis 0), centered on the middle
           of the initial array
         - width_y: size of the area to plot in y (axis 1), centered on the middle
           of the initial array
         - width_x: size of the area to plot in x (axis 2), centered on the middle
           of the initial array

        :return: a rotated array (if a single array was provided) or a tuple of
         rotated arrays (same length as the number of input arrays)
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

        # get the relevant angle within the rocking circle.
        # The reference point when orthogonalizing if the center of the array,
        # but we do not know to which angle it corresponds if the data was cropped.
        if central_angle is None:
            print(
                "central_angle=None, using the angle at half of the rocking curve"
                " for the calculation of the rotation matrix"
            )
            nb_steps = len(angles[rocking_circle])
            central_angle = angles[rocking_circle][int(nb_steps // 2)]

        # use this angle in the calculation of the rotation matrix
        angles = list(angles)
        angles[rocking_circle] = central_angle
        print(
            f"sample stage circles: {self._sample_circles}\n"
            f"sample stage angles:  {angles}"
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

        :param rocking_angle: angle which is tilted during the rocking curve in
         {'outofplane', 'inplane'}
        :param stage_name: supported stage name, 'sample' or 'detector'
        :param angles: tuple of angular values in degrees, one for each circle
         of the sample stage
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

    @abstractmethod
    def goniometer_values(self, **kwargs):
        """
        Retrieve goniometer values.

        This method is beamline dependent. It must be implemented in the child classes.

        :param kwargs: beamline_specific parameters
        :return: a list of motor positions
        """

    @abstractmethod
    def motor_positions(self, **kwargs):
        """
        Retrieve motor positions.

        This method is beamline dependent. It must be implemented in the child classes.

        :param kwargs: beamline_specific parameters
        :return: the diffractometer motors positions for the particular setup.
        """

    @staticmethod
    @abstractmethod
    def read_device(**kwargs):
        """
        Extract the device positions/values during a scan.

        :param kwargs: beamline_specific parameters, which may include part of the
         totality of the following keys:

          - 'logfile': the logfile created in Setup.create_logfile()
          - 'scan_number': int, number of the scan
          - 'device_name': str, name of the device

        :return: the positions/values of the device as a numpy 1D array
        """

    @staticmethod
    @abstractmethod
    def read_monitor(**kwargs):
        """
        Load the default monitor for intensity normalization of the considered beamline.

        :param kwargs: beamline_specific parameters, which may include part of the
         totality of the following keys:

          - 'logfile': the logfile created in Setup.create_logfile()
          - 'scan_number': int, number of the scan
          - 'actuators': dictionary defining the entries corresponding to actuators
            in the data file (at CRISTAL the location of data keeps changing)
          - 'beamline': str, name of the beamline. E.g. "SIXS_2018"

        :return: the default monitor values
        """

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
        Calculate a 3D rotation matrix given rotation axes and angles.

        :param stage_name: supported stage name, 'sample' or 'detector'
        :param angles: list of angular values in degrees for the stage circles
         during the measurement
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

        # create a list of rotation matrices corresponding to the circles,
        # index 0 corresponds to the most outer circle
        rotation_matrices = [
            RotationMatrix(circle, angles[idx]).get_matrix()
            for idx, circle in enumerate(
                self.__getattribute__(self.valid_names[stage_name])
            )
        ]

        # calculate the total tranformation matrix by rotating back
        # from outer circles to inner circles
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
    Define 34ID goniometer: 2 sample circles + 2 detector circles.

    The laboratory frame uses the CXI convention (z downstream, y vertical up,
    x outboard).

    - sample: theta (inplane), phi (out of plane)
    - detector: delta (inplane), gamma).

    """

    def __init__(self, sample_offsets):
        super().__init__(
            sample_circles=["y+", "x+"],
            detector_circles=["y+", "x-"],
            sample_offsets=sample_offsets,
        )

    def goniometer_values(self, setup, stage_name="bcdi", **kwargs):
        """
        Retrieve goniometer motor positions for a BCDI rocking scan.

        :param setup: the experimental setup: Class Setup
        :param stage_name: supported stage name, 'bcdi', 'sample' or 'detector'
        :return: a tuple of angular values in degrees, depending on stage_name:

         - 'bcdi': (rocking angular step, grazing incidence angles, inplane detector
           angle, outofplane detector angle). The grazing incidence angles are the
           positions of circles below the rocking circle.
         - 'sample': tuple of angular values for the sample circles, from the most outer
           to the most inner circle
         - 'detector': tuple of angular values for the detector circles, from the most
           outer to the most inner circle

        """
        # check some parameter
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

        # 34ID-C goniometer, 2S+2D (sample: theta (inplane),
        # phi (out of plane)   detector: delta (inplane), gamma)
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

    @staticmethod
    def read_device(**kwargs):
        """Extract the device positions/values during the scan at 34ID-C beamline."""
        raise NotImplementedError("'read_device' not implemented for 34ID-C")

    @staticmethod
    def read_monitor(**kwargs):
        """Load the default monitor for a dataset measured at 34ID-C."""
        raise NotImplementedError("'read_monitor' not implemented for 34ID-C")


class DiffractometerCRISTAL(Diffractometer):
    """
    Define CRISTAL goniometer: 2 sample circles + 2 detector circles.

    The laboratory frame uses the CXI convention (z downstream, y vertical up,
    x outboard).

    - sample: mgomega, mgphi
    - detector: gamma, delta.

    """

    def __init__(self, sample_offsets):
        super().__init__(
            sample_circles=["x-", "y+"],
            detector_circles=["y+", "x-"],
            sample_offsets=sample_offsets,
        )

    def goniometer_values(self, logfile, setup, stage_name="bcdi", **kwargs):
        """
        Retrieve goniometer motor positions for a BCDI rocking scan.

        :param logfile: file containing the information about the scan and image
         numbers (specfile, .fio...)
        :param setup: the experimental setup: Class Setup
        :param stage_name: supported stage name, 'bcdi', 'sample' or 'detector'
        :return: a tuple of angular values in degrees, depending on stage_name:

         - 'bcdi': (rocking angular step, grazing incidence angles, inplane detector
           angle, outofplane detector angle). The grazing incidence angles are the
           positions of circles below the rocking circle.
         - 'sample': tuple of angular values for the sample circles, from the most
           outer to the most inner circle
         - 'detector': tuple of angular values for the detector circles, from the most
           outer to the most inner circle

        """
        # check some parameter
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
        Load the scan data and extract motor positions.

        It will look for the correct entry 'rocking_angle' in the dictionary
        Setup.actuators, and use the default entry otherwise.

        :param logfile: h5py File object of CRISTAL .nxs scan file
        :param setup: the experimental setup: Class SetupPreprocessing()
        :param kwargs:

         - frames_logical: array of 0 (frame non used) or 1 (frame used) or -1
           (padded frame). The initial length is equal to the number of measured
           frames. In case of data padding, the length changes.

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

        # check if mgomega needs to be divided by 1e6
        # (data taken before the implementation of the correction)
        if isinstance(mgomega, float) and abs(mgomega) > 360:
            mgomega = mgomega / 1e6
        elif isinstance(mgomega, (tuple, list, np.ndarray)) and any(
            abs(val) > 360 for val in mgomega
        ):
            mgomega = mgomega / 1e6

        return mgomega, mgphi, gamma, delta, energy

    @staticmethod
    def cristal_load_motor(datafile, root, actuator_name, field_name):
        """
        Try to load the dataset at the defined entry and returns it.

        Patterns keep changing at CRISTAL.

        :param datafile: h5py File object of CRISTAL .nxs scan file
        :param root: string, path of the data up to the last subfolder
         (not included). This part is expected to not change over time
        :param actuator_name: string, name of the actuator
         (e.g. 'I06-C-C07-EX-DIF-KPHI'). Lowercase and uppercase will be tested when
         trying to load the data.
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
                        f"\nCould not find the field '{field_name}'"
                        f" in the actuator'{actuator_name}'"
                    )
                    print(
                        "list of available fields:"
                        f" {list(datafile[root + '/' + actuator_name].keys())}\n"
                    )
                    return 0
        return dataset

    @staticmethod
    def read_device(logfile, device_name, **kwargs):
        """
        Extract the device positions/values during the scan at CRISTAL beamline.

        :param logfile: the logfile created in Setup.create_logfile()
        :param device_name: name of the device
        :return: the positions/values of the device as a numpy 1D array
        """
        group_key = list(logfile.keys())[0]
        try:
            device_values = logfile["/" + group_key + "/scan_data/" + device_name][:]
        except KeyError:
            print(f"No device {device_name} in the logfile, defaulting values to []")
            device_values = []
        return np.asarray(device_values)

    def read_monitor(self, logfile, actuators, **kwargs):
        """
        Load the default monitor for a dataset measured at CRISTAL.

        :param logfile: the logfile created in Setup.create_logfile()
        :param actuators: dictionary defining the entries corresponding to actuators
         in the data file (at CRISTAL the location of data keeps changing)
        :return: the default monitor values
        """
        monitor_name = actuators.get("monitor", "data_04")
        return self.read_device(logfile=logfile, device_name=monitor_name)


class DiffractometerID01(Diffractometer):
    """
    Define ID01 goniometer: 3 sample circles + 2 detector circles.

    The laboratory frame uses the CXI convention (z downstream, y vertical up,
    x outboard).

    - sample: mu, eta, phi
    - detector: nu,del.

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
        Retrieve goniometer motor positions for a BCDI rocking scan.

        :param logfile: file containing the information about the scan and image
         numbers (specfile, .fio...)
        :param scan_number: the scan number to load
        :param setup: the experimental setup: Class Setup
        :param stage_name: supported stage name, 'bcdi', 'sample' or 'detector'
        :param kwargs:

         - 'frames_logical': array of 0 (frame non used) or 1 (frame used) or -1
           (padded frame). The initial length is equal to the number of measured
           frames. In case of data padding, the length changes.
         - 'follow_bragg': boolean, True for energy scans where the detector position
           is changed during the scan to follow the Bragg peak.

        :return: a tuple of angular values in degrees, depending on stage_name:

         - 'bcdi': (rocking angular step, grazing incidence angles, inplane detector
           angle, outofplane detector angle). The grazing incidence angles are the
           positions of circles below the rocking circle.
         - 'sample': tuple of angular values for the sample circles, from the most
           outer to the most inner circle
         - 'detector': tuple of angular values for the detector circles, from the most
           outer to the most inner circle

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

        :param logfile: Silx SpecFile object containing the information about the scan
         and image numbers
        :param scan_number: the scan number to load
        :param setup: the experimental setup: Class SetupPreprocessing()
        :param kwargs:

         - 'frames_logical': array of 0 (frame non used) or 1 (frame used) or -1
           (padded frame). The initial length is equal to the number of measured
           frames. In case of data padding, the length changes.
         - 'follow_bragg': boolean, True for energy scans where the detector position
           is changed during the scan to follow the Bragg peak.

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

    @staticmethod
    def read_device(logfile, scan_number, device_name, **kwargs):
        """
        Extract the device positions/values during the scan at ID01 beamline.

        :param logfile: the logfile created in Setup.create_logfile()
        :param scan_number: number of the scan
        :param device_name: name of the device
        :return: the positions/values of the device as a numpy 1D array
        """
        labels = logfile[str(scan_number) + ".1"].labels  # motor scanned
        labels_data = logfile[str(scan_number) + ".1"].data  # motor scanned
        try:
            device_values = list(labels_data[labels.index(device_name), :])
        except ValueError:  # device not in the list
            print(f"No device {device_name} in the logfile, defaulting values to []")
            device_values = []
        return np.asarray(device_values)

    def read_monitor(self, logfile, scan_number, **kwargs):
        """
        Load the default monitor for a dataset measured at ID01.

        :param logfile: the logfile created in Setup.create_logfile()
        :param scan_number: int, the scan number to load
        :return: the default monitor values
        """
        monitor = self.read_device(
            logfile=logfile,
            scan_number=scan_number,
            device_name="mon2"
        )
        if len(monitor) == 0:
            monitor = self.read_device(
                logfile=logfile,
                scan_number=scan_number,
                device_name="exp1"  # exp1 for old data at ID01
            )
        return monitor


class DiffractometerNANOMAX(Diffractometer):
    """
    Define NANOMAX goniometer: 2 sample circles + 2 detector circles.

    The laboratory frame uses the CXI convention (z downstream, y vertical up,
    x outboard).

    - sample: theta, phi
    - detector: gamma,delta.

    """

    def __init__(self, sample_offsets):
        super().__init__(
            sample_circles=["x-", "y-"],
            detector_circles=["y-", "x-"],
            sample_offsets=sample_offsets,
        )

    def goniometer_values(self, logfile, setup, stage_name="bcdi", **kwargs):
        """
        Retrieve goniometer motor positions for a BCDI rocking scan.

        :param logfile: file containing the information about the scan and image
         numbers (specfile, .fio...)
        :param setup: the experimental setup: Class Setup
        :param stage_name: supported stage name, 'bcdi', 'sample' or 'detector'
        :return: a tuple of angular values in degrees, depending on stage_name:

         - 'bcdi': (rocking angular step, grazing incidence angles, inplane detector
           angle, outofplane detector angle). The grazing incidence angles are the
           positions of circles below the rocking circle.
         - 'sample': tuple of angular values for the sample circles, from the most
           outer to the most inner circle
         - 'detector': tuple of angular values for the detector circles, from the most
           outer to the most inner circle

        """
        # check some parameter
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

        :param logfile: Silx SpecFile object containing the information about the scan
         and image numbers
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
                        "phi not in measurement data,"
                        ' check the parameter "rocking_angle"'
                    )
                theta = logfile["/" + group_key + "/snapshot/gontheta"][:]
            else:
                try:
                    theta = logfile["/" + group_key + "/measurement/gontheta"][:]
                except KeyError:
                    raise KeyError(
                        "theta not in measurement data,"
                        ' check the parameter "rocking_angle"'
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

    @staticmethod
    def read_device(logfile, device_name, **kwargs):
        """
        Extract the device positions/values during the scan at Nanomax beamline.

        :param logfile: the logfile created in Setup.create_logfile()
        :param device_name: name of the device
        :return: the positions/values of the device as a numpy 1D array
        """
        group_key = list(logfile.keys())[0]  # currently 'entry'
        try:
            device_values = logfile["/" + group_key + "/measurement/" + device_name][:]
        except KeyError:
            print(f"No device {device_name} in the logfile, defaulting values to []")
            device_values = []
        return np.asarray(device_values)

    def read_monitor(self, logfile, **kwargs):
        """
        Load the default monitor for a dataset measured at NANOMAX.

        :param logfile: the logfile created in Setup.create_logfile()
        :return: the default monitor values
        """
        return self.read_device(logfile=logfile, device_name="alba2")


class DiffractometerP10(Diffractometer):
    """
    Define P10 goniometer: 4 sample circles + 2 detector circles.

    The laboratory frame uses the CXI convention (z downstream, y vertical up,
    x outboard).

    - sample: mu, om, chi, phi
    - detector: gamma, delta.

    """

    def __init__(self, sample_offsets):
        super().__init__(
            sample_circles=["y+", "x-", "z+", "y-"],
            detector_circles=["y+", "x-"],
            sample_offsets=sample_offsets,
        )

    def goniometer_values(self, logfile, setup, stage_name="bcdi", **kwargs):
        """
        Retrieve goniometer motor positions for a BCDI rocking scan.

        :param logfile: file containing the information about the scan and image
         numbers (specfile, .fio...)
        :param setup: the experimental setup: Class Setup
        :param stage_name: supported stage name, 'bcdi', 'sample' or 'detector'
        :return: a tuple of angular values in degrees, depending on stage_name:

         - 'bcdi': (rocking angular step, grazing incidence angles, inplane detector
           angle, outofplane detector angle). The grazing incidence angles are the
           positions of circles below the rocking circle.
         - 'sample': tuple of angular values for the sample circles, from the most
           outer to the most inner circle
         - 'detector': tuple of angular values for the detector circles, from the most
           outer to the most inner circle

        """
        # check some parameter
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
        Load the .fio file from the scan and extract motor positions.

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

    @staticmethod
    def read_device(logfile, device_name, **kwargs):
        """
        Extract the device positions/values during the scan at P10 beamline.

        :param logfile: the logfile created in Setup.create_logfile()
        :param device_name: name of the device
        :return: the positions/values of the device as a numpy 1D array
        """
        device_values = []
        index_device = None  # index of the column corresponding to the device in .fio
        with open(logfile, "r") as fio:
            fio_lines = fio.readlines()
            for line in fio_lines:
                this_line = line.strip()
                words = this_line.split()

                if "Col" in words and device_name in words:
                    # device_name scanned, template = ' Col 0 motor_name DOUBLE\n'
                    index_device = int(words[1]) - 1  # python index starts at 0

                if index_device is not None and util.is_numeric(words[0]):
                    # we are reading data and index_motor is defined
                    device_values.append(float(words[index_device]))

        if index_device is None:
            print(f"No device {device_name} in the logfile, defaulting values to []")
        return np.asarray(device_values)

    def read_monitor(self, logfile, **kwargs):
        """
        Load the default monitor for a dataset measured at P10.

        :param logfile: the logfile created in Setup.create_logfile()
        :return: the default monitor values
        """
        monitor = self.read_device(logfile=logfile, device_name="ipetra")
        if len(monitor) == 0:
            monitor = self.read_device(logfile=logfile, device_name="curpetra")
        return monitor


class DiffractometerSIXS(Diffractometer):
    """
    Define SIXS goniometer: 2 sample circles + 3 detector circles.

    The laboratory frame uses the CXI convention (z downstream, y vertical up,
    x outboard).

    - sample: beta, mu
    - detector: beta, gamma, del.

    """

    def __init__(self, sample_offsets):
        super().__init__(
            sample_circles=["x-", "y+"],
            detector_circles=["x-", "y+", "x-"],
            sample_offsets=sample_offsets,
        )

    def goniometer_values(self, logfile, setup, stage_name="bcdi", **kwargs):
        """
        Retrieve goniometer motor positions for a BCDI rocking scan.

        :param logfile: file containing the information about the scan and image
         numbers (specfile, .fio...)
        :param setup: the experimental setup: Class Setup
        :param stage_name: supported stage name, 'bcdi', 'sample' or 'detector'
        :param kwargs:

         - 'frames_logical': array of 0 (frame non used) or 1 (frame used) or -1
           (padded frame). The initial length is equal to the number of measured
           frames. In case of data padding, the length changes.

        :return: a tuple of angular values in degrees, depending on stage_name:

         - 'bcdi': (rocking angular step, grazing incidence angles, inplane detector
           angle, outofplane detector angle). The grazing incidence angles are the
           positions of circles below the rocking circle.
         - 'sample': tuple of angular values for the sample circles, from the most
           outer to the most inner circle
         - 'detector': tuple of angular values for the detector circles, from the most
           outer to the most inner circle

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

         - frames_logical: array of 0 (frame non used) or 1 (frame used) or -1
           (padded frame). The initial length is equal to the number of measured
           frames. In case of data padding, the length changes.

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
                except AttributeError:
                    # the alias dictionnary was probably not provided
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

    @staticmethod
    def read_device(logfile, device_name, **kwargs):
        """
        Extract the device positions/values during the scan at SIXS beamline.

        :param logfile: the logfile created in Setup.create_logfile()
        :param device_name: name of the device
        :return: the positions/values of the device as a numpy 1D array
        """
        try:
            device_values = getattr(logfile, device_name)
        except AttributeError:
            print(f"No device {device_name} in the logfile, defaulting values to []")
            device_values = []
        return np.asarray(device_values)

    def read_monitor(self, logfile, beamline, **kwargs):
        """
        Load the default monitor for a dataset measured at SIXS.

        :param logfile: the logfile created in Setup.create_logfile()
        :param beamline: str, name of the beamline. E.g. "SIXS_2018"
        :return: the default monitor values
        """
        if beamline == "SIXS_2018":
            return self.read_device(logfile=logfile, device_name="imon1")
        # SIXS_2019
        monitor = self.read_device(logfile=logfile, device_name="imon0")
        if len(monitor) == 0:  # the alias dictionnary was probably not provided
            monitor = self.read_device(logfile=logfile, device_name="intensity")
        return monitor
