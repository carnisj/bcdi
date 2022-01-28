# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

"""
Beamline-dependent diffractometer classes.

These classes follow the same structure as beamline classes. It would have been
possible to put all the beamline-dependent code in a single child class per beamline,
but the class would have been huge and more difficult to maintain. The class methods
manage the extraction of motors/counters position, data loading (which can be thought
just as another counter), data preprocessing (normalization by monitor, flatfield,
hotpixels removal, background subtraction), and the rotation of the sample so that all
sample circles are at 0 degrees. Generic method are implemented in the abstract base
class Diffractometer, and beamline-dependent methods need to be implemented in each
child class (they are decoracted by @abstractmethod in the base class, they are
indicated using @ in the following diagram). These classes are not meant to be
instantiated directly but via a Setup instance.

.. mermaid::
  :align: center

  classDiagram
    class Diffractometer{
      +tuple sample_offsets
      +tuple sample_circles
      +tuple detector_circles
      add_circle()
      get_circles()
      get_rocking_circle()
      remove_circle()
      rotation_matrix()
      select_frames()
      valid_name()
  }
    ABC <|-- Diffractometer

API Reference
-------------

"""

from abc import ABC
from functools import reduce
from numbers import Number, Real
import numpy as np
from typing import List

from bcdi.experiment.rotation_matrix import RotationMatrix
from bcdi.utils import validation as valid


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
    if beamline in {"ID01", "ID01BLISS"}:
        return DiffractometerID01(sample_offsets)
    if beamline in {"SIXS_2018", "SIXS_2019"}:
        return DiffractometerSIXS(sample_offsets)
    if beamline == "34ID":
        return Diffractometer34ID(sample_offsets)
    if beamline == "P10":
        return DiffractometerP10(sample_offsets)
    if beamline == "P10_SAXS":
        return DiffractometerP10SAXS()
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
    :param kwargs:
     - 'default_offsets': tuple, default sample offsets of the diffractometer. It needs
       to be implemented as a class attribute in the child class if necessary. See an
       example in DiffractometerP10

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

    def __init__(
        self,
        sample_offsets,
        sample_circles=(),
        detector_circles=(),
        **kwargs,
    ):
        self.sample_angles = None
        self.sample_circles = sample_circles
        self.detector_angles = None
        self.detector_circles = detector_circles
        if sample_offsets is None:
            sample_offsets = kwargs.get("default_offsets")
        self.sample_offsets = sample_offsets

    @property
    def detector_angles(self):
        """Tuple of goniometer angular values for the detector stages."""
        return self._detector_angles

    @detector_angles.setter
    def detector_angles(self, value):
        valid.valid_container(
            value,
            container_types=tuple,
            item_types=(Real, np.ndarray),
            allow_none=True,
            name="detector_angles",
        )
        self._detector_angles = value

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
    def sample_angles(self):
        """Tuple of goniometer angular values for the sample stages."""
        return self._sample_angles

    @sample_angles.setter
    def sample_angles(self, value):
        valid.valid_container(
            value,
            container_types=tuple,
            item_types=(Real, np.ndarray),
            allow_none=True,
            name="sample_angles",
        )
        self._sample_angles = value

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
        return np.array(reduce(np.matmul, rotation_matrices))

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


class DiffractometerCRISTAL(Diffractometer):
    """
    Define CRISTAL goniometer: 2 sample circles + 2 detector circles.

    The laboratory frame uses the CXI convention (z downstream, y vertical up,
    x outboard).

    - sample: mgomega, mgphi
    - detector: gamma, delta.

    """

    sample_rotations = ["x-", "y+"]
    detector_rotations = ["y+", "x-"]

    def __init__(self, sample_offsets):
        super().__init__(
            sample_circles=self.sample_rotations,
            detector_circles=self.detector_rotations,
            sample_offsets=sample_offsets,
        )


class DiffractometerID01(Diffractometer):
    """
    Define ID01 goniometer: 3 sample circles + 2 detector circles.

    The laboratory frame uses the CXI convention (z downstream, y vertical up,
    x outboard).

    - sample: mu, eta, phi
    - detector: nu,del.

    """

    sample_rotations = ["y-", "x-", "y-"]
    detector_rotations = ["y-", "x-"]

    def __init__(self, sample_offsets):
        super().__init__(
            sample_circles=self.sample_rotations,
            detector_circles=self.detector_rotations,
            sample_offsets=sample_offsets,
        )


class DiffractometerNANOMAX(Diffractometer):
    """
    Define NANOMAX goniometer: 2 sample circles + 2 detector circles.

    The laboratory frame uses the CXI convention (z downstream, y vertical up,
    x outboard).

    - sample: theta, phi
    - detector: gamma,delta.

    """

    sample_rotations = ["x-", "y-"]
    detector_rotations = ["y-", "x-"]

    def __init__(self, sample_offsets):
        super().__init__(
            sample_circles=self.sample_rotations,
            detector_circles=self.detector_rotations,
            sample_offsets=sample_offsets,
        )


class DiffractometerP10(Diffractometer):
    """
    Define P10 goniometer: 4 sample circles + 2 detector circles.

    The laboratory frame uses the CXI convention (z downstream, y vertical up,
    x outboard).

    - sample: mu, om, chi, phi
    - detector: gamma, delta.

    """

    sample_rotations = ["y+", "x-", "z+", "y-"]
    detector_rotations = ["y+", "x-"]
    default_offsets = (0, 0, 90, 0)

    def __init__(self, sample_offsets):
        super().__init__(
            sample_circles=self.sample_rotations,
            detector_circles=self.detector_rotations,
            sample_offsets=sample_offsets,
            default_offsets=self.default_offsets,
        )


class DiffractometerP10SAXS(Diffractometer):
    """
    Define P10 goniometer for the USAXS setup: 1 sample circle, no detector circle.

    The laboratory frame uses the CXI convention (z downstream, y vertical up,
    x outboard).

    - sample: phi (names hprz or sprz at the beamline)

    """

    sample_rotations = ["y+"]
    detector_rotations: List[str] = []
    default_offsets = (0,)

    def __init__(self):
        super().__init__(sample_offsets=self.default_offsets)


class DiffractometerSIXS(Diffractometer):
    """
    Define SIXS goniometer: 2 sample circles + 3 detector circles.

    The laboratory frame uses the CXI convention (z downstream, y vertical up,
    x outboard).

    - sample: beta, mu
    - detector: beta, gamma, del.

    """

    sample_rotations = ["x-", "y+"]
    detector_rotations = ["x-", "y+", "x-"]

    def __init__(self, sample_offsets):
        super().__init__(
            sample_circles=self.sample_rotations,
            detector_circles=self.detector_rotations,
            sample_offsets=sample_offsets,
        )


class Diffractometer34ID(Diffractometer):
    """
    Define 34ID goniometer: 3 sample circles + 2 detector circles.

    The laboratory frame uses the CXI convention (z downstream, y vertical up,
    x outboard).

    - sample: theta (inplane), chi, phi (out of plane)
    - detector: delta (inplane), gamma).

    """

    sample_rotations = ["y+", "z-", "y+"]
    detector_rotations = ["y+", "x-"]
    default_offsets = (0, 90, 0)

    def __init__(self, sample_offsets):
        super().__init__(
            sample_circles=self.sample_rotations,
            detector_circles=self.detector_rotations,
            sample_offsets=sample_offsets,
            default_offsets=self.default_offsets,
        )
