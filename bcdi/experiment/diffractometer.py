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
      +tuple sample_offsets
      -_geometry
      add_circle()
      get_circles()
      get_rocking_circle()
      remove_circle()
      rotation_matrix()
      valid_name()
  }

'sample_offsets' is a list or tuple of angles in degrees, corresponding to the offsets
of each of the sample circles (the offset for the most outer circle should be at
index 0). The number of circles is beamline dependent, as indicated below with default
values in degrees:

- ID01: (mu=0, eta=0, phi=0,)
- SIXS: (beta=0, mu=0,)
- 34ID-C: (theta=0, chi=90, phi=0,)
- P10: (mu=0, om=0, chi=90, phi=0,)
- P10_SAXS: (phi=0,)
- CRISTAL: (mgomega=0, mgphi=0,)
- NANOMAX: (theta=0, phi=0,)

API Reference
-------------

"""

import logging
from abc import ABC
from collections import namedtuple
from functools import reduce
from numbers import Number, Real
from typing import List, Optional, Tuple, Union

import numpy as np

import bcdi.utils.format as fmt
from bcdi.constants import BEAMLINES_BCDI, BEAMLINES_SAXS
from bcdi.experiment.rotation_matrix import RotationMatrix
from bcdi.utils import validation as valid

module_logger = logging.getLogger(__name__)

Geometry = namedtuple(
    "Geometry",
    ["sample_circles", "detector_circles", "default_offsets", "user_offsets"],
)
Geometry.__doc__ = """
Describe the geometry of the diffractometer including detector circles.

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

Geometry_SAXS = namedtuple(
    "Geometry_SAXS",
    ["sample_circles", "detector_axes", "default_offsets", "user_offsets"],
)
Geometry.__doc__ = """
Describe the geometry of the diffractometer with a detector not on a circle.

The detector plane is always perpendicular to the direct beam, independently of its
position. The frame used is the laboratory frame with the CXI convention
(z downstream, y vertical up, x outboard).

:param sample_circles: list of sample circles from outer to inner (e.g. mu eta
 chi phi), expressed using a valid pattern within {'x+', 'x-', 'y+', 'y-', 'z+',
 'z-'}. For example: ['y+' ,'x-', 'z-', 'y+']
:param detector_axes: list of the translation direction of detector axes for a detector
 not sitting on a goninometer, expressed in the laboratory frame.
 For example: ['z+', 'y+', 'x-']
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
    if beamline == "BM02":
        return Geometry(
            sample_circles=["y+", "x-", "z+", "x-"],
            detector_circles=["y+", "x-"],
            default_offsets=(0, 0, 0, 0),
            user_offsets=sample_offsets,
        )
    if beamline == "ID27":
        return Geometry_SAXS(
            sample_circles=["y+"],
            detector_axes=["z+", "y+", "x+"],
            default_offsets=(0,),
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
        return Geometry_SAXS(
            sample_circles=["y+"],
            detector_axes=["z+", "y+", "x+"],  # check the translation directions!
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


class Diffractometer(ABC):
    """
    Base class for defining diffractometers.

    The frame used is the laboratory frame with the CXI convention (z downstream,
    y vertical up, x outboard).

    :param name: name of the beamline
    :param sample_offsets: list or tuple of angles in degrees, corresponding to
     the offsets of each of the sample circles (the offset for the most outer circle
     should be at index 0). The number of circles is beamline dependent. Convention:
     the sample offsets will be subtracted to measurement the motor values.
    :param kwargs:

     - 'logger': an optional logger

    """

    valid_circles = {
        "x+",
        "x-",
        "y+",
        "y-",
        "z+",
        "z-",
    }  # + counter-clockwise, - clockwise
    valid_names = {"sample": "sample_circles"}

    def __init__(self, name: str, sample_offsets=None, **kwargs) -> None:
        self.logger = kwargs.get("logger", module_logger)
        self._geometry = create_geometry(beamline=name, sample_offsets=sample_offsets)
        self.name = name
        self.sample_circles = self._geometry.sample_circles
        self.sample_offsets = (
            self._geometry.user_offsets
            if self._geometry.user_offsets is not None
            else self._geometry.default_offsets
        )

    @property
    def sample_circles(self) -> List[str]:
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
    def sample_circles(self, value: Union[Tuple[str, ...], List[str]]) -> None:
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
    def sample_offsets(self) -> Union[Tuple[Real, ...], List[Real]]:
        """
        List or tuple of sample angular offsets in degrees.

        These angles correspond to the offsets of each f the sample circles (the
        offset for the most outer circle should be at index 0). Convention: the
        sample offsets will be subtracted to measurement the motor values.
        """
        return self._sample_offsets

    @sample_offsets.setter
    def sample_offsets(self, value: Union[Tuple[Real, ...], List[Real]]) -> None:
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

    def add_circle(self, stage_name: str, index: int, circle: str) -> None:
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

    def get_circles(self, stage_name: str) -> List[str]:
        """
        Return the list of circles for the stage.

        :param stage_name: supported stage name, 'sample' or 'detector'
        """
        self.valid_name(stage_name)
        return list(self.__getattribute__(self.valid_names[stage_name]))

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

    def remove_circle(self, stage_name: str, index: int) -> None:
        """
        Remove the circle at index from the list of sample circles.

        :param stage_name: supported stage name, 'sample' or 'detector'
        :param index: index of the circle to be removed from the list
        """
        if stage_name not in self.valid_names:
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

    def __repr__(self) -> str:
        """Representation string of the Diffractometer instance."""
        return fmt.create_repr(self, Diffractometer)

    def rotation_matrix(self, stage_name: str, angles: List[Real]) -> np.ndarray:
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
            RotationMatrix(circle, angles[idx], logger=self.logger).get_matrix()
            for idx, circle in enumerate(
                self.__getattribute__(self.valid_names[stage_name])
            )
        ]

        # calculate the total tranformation matrix by rotating back
        # from outer circles to inner circles
        return np.array(reduce(np.matmul, rotation_matrices))

    def valid_name(self, stage_name: str) -> None:
        """
        Check if the stage is defined.

        :param stage_name: supported stage name, e.g. 'sample'
        """
        if stage_name not in self.valid_names:
            raise NotImplementedError(
                f"'{stage_name}' is not implemented,"
                f" available are {list(self.valid_names.keys())}"
            )


class FullDiffractometer(Diffractometer):
    """
    Class for defining diffractometers including detector circles.

    The frame used is the laboratory frame with the CXI convention (z downstream,
    y vertical up, x outboard).

    :param name: name of the beamline
    :param sample_offsets: list or tuple of angles in degrees, corresponding to
     the offsets of each of the sample circles (the offset for the most outer circle
     should be at index 0). The number of circles is beamline dependent. Convention:
     the sample offsets will be subtracted to measurement the motor values.
    :param kwargs:

     - 'logger': an optional logger

    """

    valid_names = {"sample": "sample_circles", "detector": "detector_circles"}

    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.detector_circles = self._geometry.detector_circles

    @property
    def detector_circles(self) -> List[str]:
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
    def detector_circles(self, value: Union[Tuple[str, ...], List[str]]) -> None:
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


class DiffractometerSAXS(Diffractometer):
    """
    Class for defining diffractometers where the detector is not on a circle.

    The detector plane is always perpendicular to the direct beam, independently of its
    position. The frame used is the laboratory frame with the CXI convention
    (z downstream, y vertical up, x outboard).

    :param name: name of the beamline
    :param sample_offsets: list or tuple of angles in degrees, corresponding to
     the offsets of each of the sample circles (the offset for the most outer circle
     should be at index 0). The number of circles is beamline dependent. Convention:
     the sample offsets will be subtracted to measurement the motor values.
    :param kwargs:

     - 'logger': an optional logger

    """

    valid_names = {"sample": "sample_circles", "detector": "detector_circles"}

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)
        self.detector_axes = self._geometry.detector_axes

    @property
    def detector_axes(self) -> List[str]:
        """
        List of detector axes.

        The axes are expressed in the order [z, y, x], the sign corresponding to the
        translation direction using a valid pattern within
        {'x+', 'x-', 'y+', 'y-', 'z+', 'z-'}. For example: ['z+' ,'y-', 'x-'].
        Convention: CXI convention (z downstream, y vertical up, x outboard).
        """
        return self._detector_axes

    @detector_axes.setter
    def detector_axes(self, value: Union[Tuple[str, ...], List[str]]) -> None:
        valid.valid_container(
            value,
            container_types=(tuple, list),
            length=3,
            item_types=str,
            name="Diffractometer.detector_circles",
        )
        if any(val not in self.valid_circles for val in value):
            raise ValueError(
                "Invalid circle value encountered in detector_axes,"
                f" valid are {self.valid_circles}"
            )
        self._detector_axes = list(value)


class DiffractometerFactory:
    """Create a diffractometer depending on the beamline name."""

    @staticmethod
    def create_diffractometer(
        name: str, sample_offsets: Optional[Tuple[float, ...]] = None, **kwargs
    ) -> Union[FullDiffractometer, DiffractometerSAXS]:
        """
        Create a diffractometer instance of the corresponding class.

        :param name: name of the beamline
        :param sample_offsets: list or tuple of angles in degrees, corresponding to
         the offsets of each of the sample circles (the offset for the most outer circle
         should be at index 0). The number of circles is beamline dependent. Convention:
         the sample offsets will be subtracted to measurement the motor values.
        :param kwargs:
         - 'logger': an optional logger

        :return: an instance of the corresponding class
        """
        if name in BEAMLINES_BCDI:
            return FullDiffractometer(
                name=name, sample_offsets=sample_offsets, **kwargs
            )
        if name in BEAMLINES_SAXS:
            return DiffractometerSAXS(
                name=name, sample_offsets=sample_offsets, **kwargs
            )
        raise NotImplementedError(f"Beamline {name} not supported")
