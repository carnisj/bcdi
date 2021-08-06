# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#   (c) 06/2021-present : DESY CFEL
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

"""RotationMatrix class."""
import numpy as np
from numbers import Number, Real

from bcdi.utils import validation as valid



class RotationMatrix:
    """
    Class defining a rotation matrix given the rotation axis and the angle.

    :param circle: circle in {'x+', 'x-', 'y+', 'y-', 'z+', 'z-'}. The letter
     represents the rotation axis. + for a counter-clockwise rotation, - for a
     clockwise rotation.
    :param angle: angular value in degrees to be used in the calculation of the
     rotation matrix
    """

    valid_circles = {
        "x+",
        "x-",
        "y+",
        "y-",
        "z+",
        "z-",
    }  # + counter-clockwise, - clockwise

    def __init__(self, circle, angle):
        self.angle = angle
        self.circle = circle

    @property
    def angle(self):
        """Angular value to be used in the calculation of the rotation matrix."""
        return self._angle

    @angle.setter
    def angle(self, value):
        valid.valid_item(value, allowed_types=Real, name="value")
        if np.isnan(value):
            raise ValueError("value is a nan")
        self._angle = value

    @property
    def circle(self):
        """
        Circle definition used for the calculation of the rotation matrix.

        Allowed values: {'x+', 'x-', 'y+', 'y-', 'z+', 'z-'}.
        + for a counter-clockwise rotation, - for a clockwise rotation.
        """
        return self._circle

    @circle.setter
    def circle(self, value):
        if value not in RotationMatrix.valid_circles:
            raise ValueError(
                f"{value} is not in the list of valid circles:"
                f" {list(RotationMatrix.valid_circles)}"
            )
        self._circle = value

    def get_matrix(self):
        """
        Calculate the rotation matric for a given circle and angle.

        :return: a numpy ndarray of shape (3, 3)
        """
        angle = self.angle * np.pi / 180  # convert from degrees to radians

        if self.circle[1] == "+":
            rot_dir = 1
        else:  # '-'
            rot_dir = -1

        if self.circle[0] == "x":
            matrix = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(angle), -rot_dir * np.sin(angle)],
                    [0, rot_dir * np.sin(angle), np.cos(angle)],
                ]
            )
        elif self.circle[0] == "y":
            matrix = np.array(
                [
                    [np.cos(angle), 0, rot_dir * np.sin(angle)],
                    [0, 1, 0],
                    [-rot_dir * np.sin(angle), 0, np.cos(angle)],
                ]
            )
        elif self.circle[0] == "z":
            matrix = np.array(
                [
                    [np.cos(angle), -rot_dir * np.sin(angle), 0],
                    [rot_dir * np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1],
                ]
            )
        else:
            raise ValueError(
                f"{self.circle} is not in the list of valid circles:"
                f" {list(RotationMatrix.valid_circles)}"
            )
        return matrix
