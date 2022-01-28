# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

"""
Diffractometer class.

This class holds the definition of the geometry for the sample and detector circles.
The beamline-specific geometry is defined via a named tuple.

.. mermaid::
  :align: center

  classDiagram
    class Diffractometer{
      +detector_circles
      +name
      +sample_circles
      +sample_offsets
      +tuple sample_offsets
      -_geometry
      add_circle()
      get_circles()
      get_rocking_circle()
      remove_circle()
      rotation_matrix()
      valid_name()
  }

API Reference
-------------

"""

from collections import namedtuple
from functools import reduce
from numbers import Number, Real
import numpy as np

from bcdi.experiment.rotation_matrix import RotationMatrix
from bcdi.utils import validation as valid

Geometry = namedtuple(
    "Geometry",
    ["sample_circles", "detector_circles", "default_offsets", "user_offsets"],
)
Geometry.__doc__ = """
Describe the geometry of the diffractometer.

:param sample_circles: list of sample circles from outer to inner (e.g. mu eta
 chi phi), expressed using a valid pattern within {'x+', 'x-', 'y+', 'y-', 'z+',
 'z-'}. For example: ['y+' ,'x-', 'z-', 'y+']
:param detector_circles: list of detector circles from outer to inner
 (e.g. gamma delta), expressed using a valid pattern within {'x+', 'x-', 'y+',
 'y-', 'z+', 'z-'}. For example: ['y+', 'x-']
:param default_offsets: tuple, default sample offsets of the diffractometer, same length
 as sample_circles.
:param user_offsets: tuple, user-defined sample offsets of the diffractometer, same
 length as sample_circles.
"""


def create_geometry(beamline, sample_offsets=None):
    """
    Create a Diffractometer instance depending on the beamline.

    :param beamline: str, name of the
    :param sample_offsets: list or tuple of angles in degrees, corresponding to
     the offsets of each of the sample circles (the offset for the most outer circle
     should be at index 0). The number of circles is beamline dependent. Convention:
     the sample offsets will be subtracted to measurement the motor values.
    :return:  the corresponding Geometry named tuple
    """
    if beamline in {"ID01", "ID01BLISS"}:
        return Geometry(
            sample_circles=["y-", "x-", "y-"],
            detector_circles=["y-", "x-"],
            default_offsets=(0, 0, 0),
            user_offsets=sample_offsets,
        )
    if beamline in {"SIXS_2018", "SIXS_2019"}:
        return Geometry(
            sample_circles=["x-", "y+"],
            detector_circles=["x-", "y+", "x-"],
            default_offsets=(0, 0),
            user_offsets=sample_offsets,
        )
    if beamline == "34ID":
        return Geometry(
            sample_circles=["y+", "z-", "y+"],
            detector_circles=["y+", "x-"],
            default_offsets=(0, 90, 0),
            user_offsets=sample_offsets,
        )
    if beamline == "P10":
        return Geometry(
            sample_circles=["y+", "x-", "z+", "y-"],
            detector_circles=["y+", "x-"],
            default_offsets=(0, 0, 90, 0),
            user_offsets=sample_offsets,
        )
    if beamline == "P10_SAXS":
        return Geometry(
            sample_circles=["y+"],
            detector_circles=[],
            default_offsets=(0,),
            user_offsets=sample_offsets,
        )
    if beamline == "CRISTAL":
        return Geometry(
            sample_circles=["x-", "y+"],
            detector_circles=["y+", "x-"],
            default_offsets=(0, 0),
            user_offsets=sample_offsets,
        )
    if beamline == "NANOMAX":
        return Geometry(
            sample_circles=["x-", "y-"],
            detector_circles=["y-", "x-"],
            default_offsets=(0, 0),
            user_offsets=sample_offsets,
        )
    raise NotImplementedError(
        f"No diffractometer implemented for the beamline {beamline}"
    )


class Diffractometer:
    """
    Base class for defining diffractometers.

    The frame used is the laboratory frame with the CXI convention (z downstream,
    y vertical up, x outboard).

    :param name: name of the beamline
    :param sample_offsets: list or tuple of angles in degrees, corresponding to
     the offsets of each of the sample circles (the offset for the most outer circle
     should be at index 0). The number of circles is beamline dependent. Convention:
     the sample offsets will be subtracted to measurement the motor values.

    """

    valid_circles = {
        "x+",
        "x-",
        "y+",
        "y-",
        "z+",
        "z-",
    }  # + counter-clockwise, - clockwise
    valid_names = {"sample": "sample_circles", "detector": "detector_circles"}

    def __init__(
        self,
        name,
        sample_offsets=None,
    ):
        self._geometry = create_geometry(beamline=name, sample_offsets=sample_offsets)
        self.name = name
        self.sample_circles = self._geometry.sample_circles
        self.detector_circles = self._geometry.detector_circles
        self.sample_offsets = (
            self._geometry.user_offsets
            if self._geometry.user_offsets is not None
            else self._geometry.default_offsets
        )

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
