# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#   (c) 06/2021-present : DESY CFEL
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

"""Beamline-related classes."""
from abc import ABC, abstractmethod
import numpy as np
import os
import h5py
from silx.io.specfile import SpecFile


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
    detector_orientation = {"y-": 1, "y+": -1, "z-": 1, "z+": -1}
    # "y-" detector horizontal axis inboard, as it should be in the CXI convention
    # "z-" detector vertical axis down, as it should be in the CXI convention

    def __init__(self, name):
        self._name = name

    @staticmethod
    @abstractmethod
    def create_logfile(params):
        """
        Create the logfile, which can be a log/spec file or the data itself.

        The nature of this file is beamline dependent.

        :param params: dictionnary of the setup parameters including the following keys:
          - 'scan_number': the scan number to load.
          - 'root_folder': the root directory of the experiment, where is e.g. the
            specfile/.fio file.
          - 'filename': the file name to load, or the path of 'alias_dict.txt' for SIXS.
          - 'datadir': the data directory
          - 'template_imagefile': the template for data/image file names

        :return: logfile
        """
        
    @property
    @abstractmethod
    def detector_hor(self):
        """
        Horizontal detector orientation expressed in the frame of xrayutilities.

        This is beamline-dependent. The frame convention of xrayutilities is the
        following: x downstream, y outboard, z vertical up.

        :return: "y+" or "y-"
        """

    @property
    @abstractmethod
    def detector_ver(self):
        """
        Vertical detector orientation expressed in the frame of xrayutilities.

        This is beamline-dependent. The frame convention of xrayutilities is the
        following: x downstream, y outboard, z vertical up.

        :return: "z+" or "z-"
        """

    @staticmethod
    @abstractmethod
    def exit_wavevector(params):
        """
        Calculate the exit wavevector kout.

        It uses the setup parameters. kout is expressed in 1/m in the
        laboratory frame (z downstream, y vertical, x outboard).

        :param params: dictionnary of the setup parameters including the following keys:
          - 'wavelength_m': X-ray wavelength in meters.
          - 'inplane_angle': horizontal detector angle, in degrees.
          - 'outofplane_angle': vertical detector angle, in degrees.
          
        :return: kout vector as a numpy array of shape (3)
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

    @property
    def name(self):
        """Name of the beamline."""
        return self._name
    
    @property
    @abstractmethod
    def outofplane_coeff(self):
        """
        Coefficient related to the detector vertical orientation.

        Define a coefficient +/- 1 depending on the detector out of plane rotation
        direction and the detector out of  plane orientation. The frame convention is
        the one of xrayutilities: x downstream, y outboard, z vertical up. See
        scripts/postprocessing/correct_angles_detector.py for a use case.

        :return: +1 or -1
        """


class BeamlineCRISTAL(Beamline):
    """
    Definition of CRISTAL beamline.
    """
    def __init__(self, name):
        super().__init__(name=name)

    @staticmethod
    def create_logfile(params):
        # no specfile, load directly the dataset
        ccdfiletmp = os.path.join(
            params["datadir"] + params["template_imagefile'"] % params["scan_number"]
        )
        return h5py.File(ccdfiletmp, "r")

    @property
    def detector_hor(self):
        # we look at the detector from downstream, detector X is along the outboard
        # direction
        return "y+"

    @property
    def detector_ver(self):
        # origin is at the top, detector Y along vertical down
        return "z-"

    @staticmethod
    def exit_wavevector(params):        
        # gamma is anti-clockwise
        kout = (2 * np.pi / params["wavelength_m"] * np.array(
            [
                np.cos(np.pi * params["inplane_angle"] / 180)
                * np.cos(np.pi * params["outofplane_angle"] / 180),  # z
                np.sin(np.pi * params["outofplane_angle"] / 180),  # y
                np.sin(np.pi * params["inplane_angle"] / 180)
                * np.cos(np.pi * params["outofplane_angle"] / 180),  # x
            ]
        )
        )
        return kout

    @property
    def inplane_coeff(self):
        # gamma is anti-clockwise, we see the detector from downstream
        return -1 * self.detector_orientation[self.detector_hor]

    @property
    def outofplane_coeff(self):
        # the out of plane detector rotation is clockwise
        return 1 * self.detector_orientation[self.detector_ver]


class BeamlineID01(Beamline):
    """
    Definition of ID01 beamline.
    """
    def __init__(self, name):
        super().__init__(name=name)

    @staticmethod
    def create_logfile(params):
        # load the spec file
        return SpecFile(params["root_folder"] + params["filename"] + ".spec")

    @property
    def detector_hor(self):
        # we look at the detector from downstream, detector X is along the outboard
        # direction
        return "y+"

    @property
    def detector_ver(self):
        # origin is at the top, detector Y along vertical down
        return "z-"

    @staticmethod
    def exit_wavevector(params):
        # nu is clockwise
        kout = (2 * np.pi / params["wavelength_m"] * np.array(
            [
                np.cos(np.pi * params["inplane_angle"] / 180)
                * np.cos(np.pi * params["outofplane_angle"] / 180),  # z
                np.sin(np.pi * params["outofplane_angle"] / 180),  # y
                -np.sin(np.pi * params["inplane_angle"] / 180)
                * np.cos(np.pi * params["outofplane_angle"] / 180),  # x
            ]
        )
        )
        return kout

    @property
    def inplane_coeff(self):
        # nu is clockwise, we see the detector from downstream
        return 1 * self.detector_orientation[self.detector_hor]

    @property
    def outofplane_coeff(self):
        # the out of plane detector rotation is clockwise
        return 1 * self.detector_orientation[self.detector_ver]


class BeamlineNANOMAX(Beamline):
    """
    Definition of NANOMAX beamline.
    """
    def __init__(self, name):
        super().__init__(name=name)

    @staticmethod
    def create_logfile(params):
        ccdfiletmp = os.path.join(
            params["datadir"] + params["template_imagefile"] % params["scan_number"]
        )
        return h5py.File(ccdfiletmp, "r")

    @property
    def detector_hor(self):
        # we look at the detector from downstream, detector X is along the outboard
        # direction
        return "y+"

    @property
    def detector_ver(self):
        # origin is at the top, detector Y along vertical down
        return "z-"

    @staticmethod
    def exit_wavevector(params):
        # gamma is clockwise
        kout = (2 * np.pi / params["wavelength_m"] * np.array(
            [
                np.cos(np.pi * params["inplane_angle"] / 180)
                * np.cos(np.pi * params["outofplane_angle"] / 180),  # z
                np.sin(np.pi * params["outofplane_angle"] / 180),  # y
                -np.sin(np.pi * params["inplane_angle"] / 180)
                * np.cos(np.pi * params["outofplane_angle"] / 180),  # x
            ]
        )
        )
        return kout

    @property
    def inplane_coeff(self):
        # gamma is clockwise, we see the detector from downstream
        return 1 * self.detector_orientation[self.detector_hor]

    @property
    def outofplane_coeff(self):
        # the out of plane detector rotation is clockwise
        return 1 * self.detector_orientation[self.detector_ver]


class BeamlineP10(Beamline):
    """
    Definition of P10 beamline.
    """
    def __init__(self, name):
        super().__init__(name=name)

    @staticmethod
    def create_logfile(params):
        # load .fio file
        return params["root_folder"] + params["filename"] + "/" + params["filename"] + \
               ".fio"

    @property
    def detector_hor(self):
        # we look at the detector from upstream, detector X is opposite to the outboard
        # direction
        return "y-"

    @property
    def detector_ver(self):
        # origin is at the top, detector Y along vertical down
        return "z-"

    @staticmethod
    def exit_wavevector(params):
        # gamma is anti-clockwise
        kout = (2 * np.pi / params["wavelength_m"] * np.array(
            [
                np.cos(np.pi * params["inplane_angle"] / 180)
                * np.cos(np.pi * params["outofplane_angle"] / 180),  # z
                np.sin(np.pi * params["outofplane_angle"] / 180),  # y
                np.sin(np.pi * params["inplane_angle"] / 180)
                * np.cos(np.pi * params["outofplane_angle"] / 180),  # x
            ]
        )
        )
        return kout

    @property
    def inplane_coeff(self):
        # gamma is anti-clockwise, we see the detector from the front
        return -1 * self.detector_orientation[self.detector_hor]

    @property
    def outofplane_coeff(self):
        # the out of plane detector rotation is clockwise
        return 1 * self.detector_orientation[self.detector_ver]


class BeamlineSIXS(Beamline):
    """
    Definition of SIXS beamline.
    """
    def __init__(self, name):
        super().__init__(name=name)

    def create_logfile(self, params):
        shortname = params["template_imagefile"] % params["scan_number"]
        if self.name == "SIXS_2018":
            # no specfile, load directly the dataset
            import bcdi.preprocessing.nxsReady as nxsReady
            return nxsReady.DataSet(
                longname=params["datadir"] + shortname,
                shortname=shortname,
                alias_dict=params["filename"],
                scan="SBS",
            )
        if self.name == "SIXS_2019":
            # no specfile, load directly the dataset
            import bcdi.preprocessing.ReadNxs3 as ReadNxs3
            return ReadNxs3.DataSet(
                directory=params["datadir"],
                filename=shortname,
                alias_dict=params["filename"],
            )

    @property
    def detector_hor(self):
        # we look at the detector from downstream, detector X is along the outboard
        # direction
        return "y+"

    @property
    def detector_ver(self):
        # origin is at the top, detector Y along vertical down
        return "z-"

    @staticmethod
    def exit_wavevector(params):
        # gamma is anti-clockwise
        kout = (2 * np.pi / params["wavelength_m"] * np.array(
            [
                np.cos(np.pi * params["inplane_angle"] / 180)
                * np.cos(np.pi * params["outofplane_angle"] / 180),  # z
                np.sin(np.pi * params["outofplane_angle"] / 180),  # y
                np.sin(np.pi * params["inplane_angle"] / 180)
                * np.cos(np.pi * params["outofplane_angle"] / 180),  # x
            ]
        )
        )
        return kout

    @property
    def inplane_coeff(self):
        # gamma is anti-clockwise, we see the detector from downstream
        return -1 * self.detector_orientation[self.detector_hor]

    @property
    def outofplane_coeff(self):
        # the out of plane detector rotation is clockwise
        return 1 * self.detector_orientation[self.detector_ver]


class Beamline34ID(Beamline):
    """
    Definition of 34ID beamline.
    """
    def __init__(self, name):
        super().__init__(name=name)

    @staticmethod
    def create_logfile(params):
        raise NotImplementedError("create_logfile method not implemented for 34ID")

    @property
    def detector_hor(self):
        # we look at the detector from upstream, detector X is opposite to the outboard
        # direction
        return "y-"

    @property
    def detector_ver(self):
        # origin is at the top, detector Y along vertical down
        return "z-"

    @staticmethod
    def exit_wavevector(params):
        # gamma is anti-clockwise
        kout = (2 * np.pi / params["wavelength_m"] * np.array(
            [
                np.cos(np.pi * params["inplane_angle"] / 180)
                * np.cos(np.pi * params["outofplane_angle"] / 180),  # z
                np.sin(np.pi * params["outofplane_angle"] / 180),  # y
                np.sin(np.pi * params["inplane_angle"] / 180)
                * np.cos(np.pi * params["outofplane_angle"] / 180),  # x
            ]
        )
        )
        return kout

    @property
    def inplane_coeff(self):
        # delta is anti-clockwise, we see the detector from the front
        return -1 * self.detector_orientation[self.detector_hor]

    @property
    def outofplane_coeff(self):
        # the out of plane detector rotation is clockwise
        return 1 * self.detector_orientation[self.detector_ver]
