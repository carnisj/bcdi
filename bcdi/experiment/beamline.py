# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#   (c) 06/2021-present : DESY CFEL
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

"""Beamline-related classes."""
from abc import ABC, abstractmethod


def create_beamline(name):
    """
    Create the instance of the beamline.

    :param name: str, name of the beamline
    :return: the corresponding beamline instance
    """
    if name == "ID01":
        return BeamlineID01(name=name)
    elif name in {"SIXS_2018", "SIXS_2019"}:
        return BeamlineSIXS(name=name)
    elif name == "34ID":
        return Beamline34ID(name=name)
    elif name == "P10":
        return BeamlineP10(name=name)
    elif name == "CRISTAL":
        return BeamlineCRISTAL(name=name)
    elif name == "NANOMAX":
        return BeamlineNANOMAX(name=name)
    else:
        raise ValueError(f"Beamline {name} not supported")


class Beamline(ABC):
    """
    Base class for defining a beamline.
    """
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        """Name of the beamline."""
        return self._name

    @property
    @abstractmethod
    def detector_hor(self):
        """
        Horizontal detector orientation expressed in the frame of xrayutilities.

        This is beamline-dependent. The frame convention of xrayutilities is the
        following: x downstream, y outboard, z vertical up.
        """

    @property
    @abstractmethod
    def detector_ver(self):
        """
        Vertical detector orientation expressed in the frame of xrayutilities.

        This is beamline-dependent. The frame convention of xrayutilities is the
        following: x downstream, y outboard, z vertical up.
        """

    @property
    @abstractmethod
    def inplane_coeff(self):
        """
        Coefficient related to the detector inplane orientation.

        Define a coefficient +/- 1 depending on the detector inplane rotation direction
        and the detector inplane orientation. The frame convention is the one of
        xrayutilities: x downstream, y outboard, z vertical up. See
        scripts/postprocessing/correct_angles_detector.py for a use case.

        :return: +1 or -1
        """


class BeamlineCRISTAL(Beamline):
    """
    Definition of CRISTAL beamline.
    """
    def __init__(self, name):
        super().__init__(name=name)

    @property
    def detector_hor(self):
        # we look at the detector from downstream, detector X is along the outboard
        # direction
        return "y+"

    @property
    def detector_ver(self):
        # origin is at the top, detector Y along vertical down
        return "z-"

    @property
    def inplane_coeff(self):
        # gamma is anti-clockwise, we see the detector from downstream
        return -1 * self.detector_hor


class BeamlineID01(Beamline):
    """
    Definition of ID01 beamline.
    """
    def __init__(self, name):
        super().__init__(name=name)

    @property
    def detector_hor(self):
        # we look at the detector from downstream, detector X is along the outboard
        # direction
        return "y+"

    @property
    def detector_ver(self):
        # origin is at the top, detector Y along vertical down
        return "z-"

    @property
    def inplane_coeff(self):
        # nu is clockwise, we see the detector from downstream
        return 1 * self.detector_hor


class BeamlineNANOMAX(Beamline):
    """
    Definition of NANOMAX beamline.
    """
    def __init__(self, name):
        super().__init__(name=name)

    @property
    def detector_hor(self):
        # we look at the detector from downstream, detector X is along the outboard
        # direction
        return "y+"

    @property
    def detector_ver(self):
        # origin is at the top, detector Y along vertical down
        return "z-"

    @property
    def inplane_coeff(self):
        # gamma is clockwise, we see the detector from downstream
        return 1 * self.detector_hor


class BeamlineP10(Beamline):
    """
    Definition of P10 beamline.
    """
    def __init__(self, name):
        super().__init__(name=name)

    @property
    def detector_hor(self):
        # we look at the detector from upstream, detector X is opposite to the outboard
        # direction
        return "y-"

    @property
    def detector_ver(self):
        # origin is at the top, detector Y along vertical down
        return "z-"

    @property
    def inplane_coeff(self):
        # gamma is anti-clockwise, we see the detector from the front
        return -1 * self.detector_hor


class BeamlineSIXS(Beamline):
    """
    Definition of SIXS beamline.
    """
    def __init__(self, name):
        super().__init__(name=name)

    @property
    def detector_hor(self):
        # we look at the detector from downstream, detector X is along the outboard
        # direction
        return "y+"

    @property
    def detector_ver(self):
        # origin is at the top, detector Y along vertical down
        return "z-"

    @property
    def inplane_coeff(self):
        # gamma is anti-clockwise, we see the detector from downstream
        return -1 * self.detector_hor


class Beamline34ID(Beamline):
    """
    Definition of 34ID beamline.
    """
    def __init__(self, name):
        super().__init__(name=name)

    @property
    def detector_hor(self):
        # we look at the detector from upstream, detector X is opposite to the outboard
        # direction
        return "y-"

    @property
    def detector_ver(self):
        # origin is at the top, detector Y along vertical down
        return "z-"

    @property
    def inplane_coeff(self):
        # delta is anti-clockwise, we see the detector from the front
        return -1 * self.detector_hor
