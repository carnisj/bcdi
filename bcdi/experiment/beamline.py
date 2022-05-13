# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

"""
Implementation of beamline-related classes.

The class methods manage the calculations related to reciprocal or direct space
transformation (interpolation in an orthonormal grid). Generic method are implemented
in the abstract base class Beamline, and beamline-dependent methods need to be
implemented in each child class (they are decorated by @abstractmethod in the base
class; they are written in italic in the following diagram). These classes are not meant
to be instantiated directly but via a Setup instance.

.. mermaid::
  :align: center

  classDiagram
    class Beamline{
      <<abstract>>
      +diffractometer
      +loader
      +name
      +sample_angles
      +detector_angles
      detector_hor()*
      detector_ver()*
      goniometer_values()*
      process_positions()*
      transformation_matrix()*
      exit_wavevector()
      find_inplane()
      find_outofplane()
      flatten_sample()
      init_qconversion()
      inplane_coeff()
      outofplane_coeff()
      process_tilt()
  }
    ABC <|-- Beamline

API Reference
-------------

"""

import logging
from math import hypot, isclose
from numbers import Real

import numpy as np

from bcdi.experiment.beamline_factory import Beamline, BeamlineSaxs

from bcdi.graph import graph_utils as gu
from bcdi.utils import utilities as util
from bcdi.utils import validation as valid

module_logger = logging.getLogger(__name__)


def create_beamline(name, **kwargs):
    """
    Create the instance of the beamline.

    :param name: str, name of the beamline
    :param kwargs: optional beamline-dependent parameters
    :return: the corresponding beamline instance
    """
    if name in {"ID01", "ID01BLISS"}:
        return BeamlineID01(name=name, **kwargs)
    if name in {"SIXS_2018", "SIXS_2019"}:
        return BeamlineSIXS(name=name, **kwargs)
    if name == "34ID":
        return Beamline34ID(name=name, **kwargs)
    if name == "P10":
        return BeamlineP10(name=name, **kwargs)
    if name == "P10_SAXS":
        return BeamlineP10SAXS(name=name, **kwargs)
    if name == "CRISTAL":
        return BeamlineCRISTAL(name=name, **kwargs)
    if name == "NANOMAX":
        return BeamlineNANOMAX(name=name, **kwargs)
    raise ValueError(f"Beamline {name} not supported")


class BeamlineCRISTAL(Beamline):
    """
    Definition of SOLEIL CRISTAL beamline.

    :param name: name of the beamline
    """

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

    @property
    def detector_hor(self):
        """
        Horizontal detector orientation expressed in the laboratory frame.

        We look at the detector from downstream, detector X is along the outboard
        direction. The laboratory frame convention is (z downstream, y vertical,
        x outboard).
        """
        return "x+"

    @property
    def detector_ver(self):
        """
        Vertical detector orientation expressed in the laboratory frame.

        The origin is at the top, detector Y along vertical down. The laboratory
        frame convention is (z downstream, y vertical, x outboard).
        """
        return "y-"

    def goniometer_values(self, setup, **kwargs):
        """
        Retrieve goniometer motor positions for a BCDI rocking scan.

        :param setup: the experimental setup: Class Setup
        :return: a tuple of angular values in degrees (rocking angular step, grazing
         incidence angles, inplane detector angle, outofplane detector angle). The
         grazing incidence angles are the positions of circles below the rocking circle.
        """
        # load the motor positions
        (
            mgomega,
            mgphi,
            inplane_angle,
            outofplane_angle,
            energy,
            detector_distance,
        ) = self.loader.motor_positions(setup=setup)

        # define the circles of interest for BCDI
        if setup.rocking_angle == "outofplane":  # mgomega rocking curve
            grazing = None  # nothing below mgomega at CRISTAL
            tilt_angle = mgomega
        elif setup.rocking_angle == "inplane":  # phi rocking curve
            grazing = (mgomega,)
            tilt_angle = mgphi
        else:
            raise ValueError('Wrong value for "rocking_angle" parameter')

        setup.check_setup(
            grazing_angle=grazing,
            inplane_angle=inplane_angle,
            outofplane_angle=outofplane_angle,
            tilt_angle=tilt_angle,
            detector_distance=detector_distance,
            energy=energy,
        )

        # CRISTAL goniometer, 2S+2D (sample: mgomega, mgphi / detector: gamma, delta)
        self.sample_angles = (mgomega, mgphi)
        self.detector_angles = (inplane_angle, outofplane_angle)

        return tilt_angle, grazing, inplane_angle, outofplane_angle

    def process_positions(
        self,
        setup,
        nb_frames,
        scan_number,
        frames_logical=None,
    ):
        """
        Load and crop/pad motor positions depending on the number of frames at CRISTAL.

        The current number of frames may be different from the original number of frames
        if the data was cropped/padded, and motor values must be processed accordingly.

        :param setup: an instance of the class Setup
        :param nb_frames: the number of frames in the current dataset
        :param scan_number: the scan number to load
        :param frames_logical: array of length the number of measured frames.
         In case of cropping/padding the number of frames changes. A frame whose
         index is set to 1 means that it is used, 0 means not used, -1 means padded
         (added) frame
        :return: a tuple of 1D arrays (sample circles, detector circles, energy)
        """
        mgomega, mgphi, gamma, delta, energy, _ = super().process_positions(
            setup=setup,
            nb_frames=nb_frames,
            scan_number=scan_number,
            frames_logical=frames_logical,
        )

        # eventually crop/pad motor values if the provided dataset was further
        # cropped/padded
        if setup.rocking_angle == "outofplane":  # mgomega rocking curve
            nb_steps = len(mgomega)
            tilt_angle = (mgomega[1:] - mgomega[0:-1]).mean()
            mgomega = self.process_tilt(
                mgomega, nb_steps=nb_steps, nb_frames=nb_frames, angular_step=tilt_angle
            )
        elif setup.rocking_angle == "inplane":  # mgphi rocking curve
            self.logger.info("mgomega", mgomega)
            nb_steps = len(mgphi)
            tilt_angle = (mgphi[1:] - mgphi[0:-1]).mean()
            mgphi = self.process_tilt(
                mgphi, nb_steps=nb_steps, nb_frames=nb_frames, angular_step=tilt_angle
            )
        else:
            raise ValueError('Wrong value for "rocking_angle" parameter')

        return util.bin_parameters(
            binning=setup.detector.binning[0],
            nb_frames=nb_frames,
            params=[mgomega, mgphi, gamma, delta, energy],
        )

    def transformation_matrix(
        self,
        wavelength,
        distance,
        pixel_x,
        pixel_y,
        inplane,
        outofplane,
        grazing_angle,
        tilt,
        rocking_angle,
        verbose=True,
    ):
        """
        Calculate the transformation matrix from detector frame to laboratory frame.

        For the transformation in direct space, the length scale is in nm,
        for the transformation in reciprocal space, it is in 1/nm.

        :param wavelength: X-ray wavelength in nm
        :param distance: detector distance in nm
        :param pixel_x: horizontal detector pixel size in nm
        :param pixel_y: vertical detector pixel size in nm
        :param inplane: horizontal detector angle in radians
        :param outofplane: vertical detector angle in radians
        :param grazing_angle: angle or list of angles of the sample circles which are
         below the rotated circle
        :param tilt: angular step of the rocking curve in radians
        :param rocking_angle: "outofplane" or "inplane"
        :param verbose: True to have printed comments
        :return: a tuple of two numpy arrays

         - the transformation matrix from the detector frame to the
           laboratory frame in reciprocal space (reciprocal length scale in  1/nm), as a
           numpy array of shape (3,3)
         - the q offset (3D vector)

        """
        lambdaz = wavelength * distance
        mymatrix = np.zeros((3, 3))
        q_offset = np.zeros(3)

        if verbose:
            self.logger.info("using CRISTAL geometry")

        if rocking_angle == "outofplane":
            if grazing_angle is not None:
                raise NotImplementedError(
                    "Circle below mgomega not implemented for CRISTAL"
                )
            if verbose:
                self.logger.info("rocking angle is mgomega")
            # rocking mgomega angle clockwise around x
            mymatrix[:, 0] = (
                2
                * np.pi
                / lambdaz
                * pixel_x
                * self.orientation_lookup[self.detector_hor]
                * np.array([-np.cos(inplane), 0, np.sin(inplane)])
            )
            mymatrix[:, 1] = (
                2
                * np.pi
                / lambdaz
                * pixel_y
                * self.orientation_lookup[self.detector_ver]
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
                2 * np.pi / lambdaz * distance * np.cos(outofplane) * np.sin(inplane)
            )
            q_offset[1] = 2 * np.pi / lambdaz * distance * np.sin(outofplane)
            q_offset[2] = (
                2
                * np.pi
                / lambdaz
                * distance
                * (np.cos(inplane) * np.cos(outofplane) - 1)
            )

        elif rocking_angle == "inplane":
            if isinstance(grazing_angle, Real):
                grazing_angle = (grazing_angle,)
            valid.valid_container(
                grazing_angle,
                container_types=(tuple, list),
                item_types=Real,
                length=1,
                name="grazing_angle",
            )
            if verbose:
                self.logger.info(
                    "rocking angle is phi,"
                    f" mgomega={grazing_angle[0] * 180 / np.pi:.3f} deg"
                )
            # rocking phi angle anti-clockwise around y,
            # incident angle mgomega is non zero (mgomega below phi)
            mymatrix[:, 0] = (
                2
                * np.pi
                / lambdaz
                * pixel_x
                * self.orientation_lookup[self.detector_hor]
                * np.array([-np.cos(inplane), 0, np.sin(inplane)])
            )
            mymatrix[:, 1] = (
                2
                * np.pi
                / lambdaz
                * pixel_y
                * self.orientation_lookup[self.detector_ver]
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
                        np.sin(grazing_angle[0]) * np.sin(inplane) * np.cos(outofplane),
                        np.cos(grazing_angle[0]) * np.sin(inplane) * np.cos(outofplane),
                    ]
                )
            )
            q_offset[0] = (
                2 * np.pi / lambdaz * distance * np.cos(outofplane) * np.sin(inplane)
            )
            q_offset[1] = 2 * np.pi / lambdaz * distance * np.sin(outofplane)
            q_offset[2] = (
                2
                * np.pi
                / lambdaz
                * distance
                * (np.cos(inplane) * np.cos(outofplane) - 1)
            )

        else:
            raise NotImplementedError(f"rocking_angle={rocking_angle} not implemented")

        return mymatrix, q_offset


class BeamlineID01(Beamline):
    """
    Definition of ESRF ID01 beamline.

    :param name: name of the beamline
    """

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

    @property
    def detector_hor(self):
        """
        Horizontal detector orientation expressed in the laboratory frame.

        We look at the detector from downstream, detector X is along the outboard
        direction. The laboratory frame convention is (z downstream, y vertical,
        x outboard).
        """
        return "x+"

    @property
    def detector_ver(self):
        """
        Vertical detector orientation expressed in the laboratory frame.

        The origin is at the top, detector Y along vertical down. The laboratory frame
        convention is (z downstream, y vertical, x outboard).
        """
        return "y-"

    def goniometer_values(self, setup, **kwargs):
        """
        Retrieve goniometer motor positions for a BCDI rocking scan.

        :param setup: the experimental setup: Class Setup
        :param kwargs:
         - 'scan_number': the scan number to load

        :return: a tuple of angular values in degrees (rocking angular step, grazing
         incidence angles, inplane detector angle, outofplane detector angle). The
         grazing incidence angles are the positions of circles below the rocking circle.
        """
        # load kwargs
        scan_number = kwargs["scan_number"]

        # check some parameter
        valid.valid_item(
            scan_number, allowed_types=int, min_excluded=0, name="scan_number"
        )

        # load motor positions
        (
            mu,
            eta,
            phi,
            inplane_angle,
            outofplane_angle,
            energy,
            detector_distance,
        ) = self.loader.motor_positions(
            setup=setup,
            scan_number=scan_number,
        )

        # define the circles of interest for BCDI
        if setup.rocking_angle == "outofplane":  # eta rocking curve
            grazing = (mu,)  # mu below eta but not used at ID01
            tilt_angle = eta
        elif setup.rocking_angle == "inplane":  # phi rocking curve
            grazing = (mu, eta)  # mu below eta but not used at ID01
            tilt_angle = phi
        else:
            raise ValueError('Wrong value for "rocking_angle" parameter')

        setup.check_setup(
            grazing_angle=grazing,
            inplane_angle=inplane_angle,
            outofplane_angle=outofplane_angle,
            tilt_angle=tilt_angle,
            detector_distance=detector_distance,
            energy=energy,
        )

        # ID01 goniometer, 3S+2D (sample: eta, chi, phi / detector: nu,del)
        self.sample_angles = (mu, eta, phi)
        self.detector_angles = (inplane_angle, outofplane_angle)

        return tilt_angle, grazing, inplane_angle, outofplane_angle

    def process_positions(
        self,
        setup,
        nb_frames,
        scan_number,
        frames_logical=None,
    ):
        """
        Load and crop/pad motor positions depending on the number of frames at ID01.

        The current number of frames may be different from the original number of frames
        if the data was cropped/padded, and motor values must be processed accordingly.

        :param setup: an instance of the class Setup
        :param nb_frames: the number of frames in the current dataset
        :param scan_number: the scan number to load
        :param frames_logical: array of length the number of measured frames.
         In case of cropping/padding the number of frames changes. A frame whose
         index is set to 1 means that it is used, 0 means not used, -1 means padded
         (added) frame
        :return: a tuple of 1D arrays (sample circles, detector circles, energy)
        """
        mu, eta, phi, nu, delta, energy, _ = super().process_positions(
            setup=setup,
            nb_frames=nb_frames,
            scan_number=scan_number,
            frames_logical=frames_logical,
        )

        # eventually crop/pad motor values if the provided dataset was further
        # cropped/padded
        if setup.rocking_angle == "outofplane":  # eta rocking curve
            self.logger.info("phi", phi)
            nb_steps = len(eta)
            tilt_angle = (eta[1:] - eta[0:-1]).mean()
            eta = self.process_tilt(
                eta, nb_steps=nb_steps, nb_frames=nb_frames, angular_step=tilt_angle
            )
        elif setup.rocking_angle == "inplane":  # phi rocking curve
            self.logger.info("eta", eta)
            nb_steps = len(phi)
            tilt_angle = (phi[1:] - phi[0:-1]).mean()
            phi = self.process_tilt(
                phi, nb_steps=nb_steps, nb_frames=nb_frames, angular_step=tilt_angle
            )
        elif setup.rocking_angle == "energy":
            pass
        else:
            raise ValueError('Wrong value for "rocking_angle" parameter')

        return util.bin_parameters(
            binning=setup.detector.binning[0],
            nb_frames=nb_frames,
            params=[mu, eta, phi, nu, delta, energy],
        )

    def transformation_matrix(
        self,
        wavelength,
        distance,
        pixel_x,
        pixel_y,
        inplane,
        outofplane,
        grazing_angle,
        tilt,
        rocking_angle,
        verbose=True,
    ):
        """
        Calculate the transformation matrix from detector frame to laboratory frame.

        For the transformation in direct space, the length scale is in nm,
        for the transformation in reciprocal space, it is in 1/nm.

        :param wavelength: X-ray wavelength in nm
        :param distance: detector distance in nm
        :param pixel_x: horizontal detector pixel size in nm
        :param pixel_y: vertical detector pixel size in nm
        :param inplane: horizontal detector angle in radians
        :param outofplane: vertical detector angle in radians
        :param grazing_angle: angle or list of angles of the sample circles which are
         below the rotated circle
        :param tilt: angular step of the rocking curve in radians
        :param rocking_angle: "outofplane", "inplane" or "energy"
        :param verbose: True to have printed comments
        :return: a tuple of two numpy arrays

         - the transformation matrix from the detector frame to the
           laboratory frame in reciprocal space (reciprocal length scale in  1/nm), as a
           numpy array of shape (3,3)
         - the q offset (3D vector)

        """
        lambdaz = wavelength * distance
        mymatrix = np.zeros((3, 3))
        q_offset = np.zeros(3)

        if verbose:
            self.logger.info("using ESRF ID01 PSIC geometry")

        if isinstance(grazing_angle, Real):
            grazing_angle = (grazing_angle,)
        valid.valid_container(
            grazing_angle,
            container_types=(tuple, list),
            item_types=Real,
            min_length=1,
            name="grazing_angle",
        )
        if not isclose(grazing_angle[0], 0, rel_tol=1e-09, abs_tol=1e-09):
            raise NotImplementedError(
                "Non-zero mu not implemented " "for the transformation matrices at ID01"
            )

        if rocking_angle == "outofplane":
            if verbose:
                self.logger.info(
                    f"rocking angle is eta, mu={grazing_angle[0] * 180 / np.pi:.3f} deg"
                )
            # rocking eta angle clockwise around x (phi does not matter, above eta)
            mymatrix[:, 0] = (
                2
                * np.pi
                / lambdaz
                * self.orientation_lookup[self.detector_hor]
                * np.array([-pixel_x * np.cos(inplane), 0, -pixel_x * np.sin(inplane)])
            )
            mymatrix[:, 1] = (
                2
                * np.pi
                / lambdaz
                * self.orientation_lookup[self.detector_ver]
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
                        tilt * distance * (1 - np.cos(inplane) * np.cos(outofplane)),
                        tilt * distance * np.sin(outofplane),
                    ]
                )
            )
            q_offset[0] = (
                -2 * np.pi / lambdaz * distance * np.cos(outofplane) * np.sin(inplane)
            )
            q_offset[1] = 2 * np.pi / lambdaz * distance * np.sin(outofplane)
            q_offset[2] = (
                2
                * np.pi
                / lambdaz
                * distance
                * (np.cos(inplane) * np.cos(outofplane) - 1)
            )

        elif rocking_angle == "inplane":
            if len(grazing_angle) != 2:
                raise ValueError("grazing_angle should be of length 2")
            if verbose:
                self.logger.info(
                    f"rocking angle is phi,"
                    f" mu={grazing_angle[0] * 180 / np.pi:.3f} deg,"
                    f" eta={grazing_angle[1] * 180 / np.pi:.3f}deg"
                )

            # rocking phi angle clockwise around y,
            # incident angle eta is non zero (eta below phi)
            mymatrix[:, 0] = (
                2
                * np.pi
                / lambdaz
                * self.orientation_lookup[self.detector_hor]
                * np.array([-pixel_x * np.cos(inplane), 0, -pixel_x * np.sin(inplane)])
            )
            mymatrix[:, 1] = (
                2
                * np.pi
                / lambdaz
                * self.orientation_lookup[self.detector_ver]
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
                        np.sin(grazing_angle[1]) * np.sin(inplane) * np.cos(outofplane),
                        np.cos(grazing_angle[1]) * np.sin(inplane) * np.cos(outofplane),
                    ]
                )
            )
            q_offset[0] = (
                -2 * np.pi / lambdaz * distance * np.cos(outofplane) * np.sin(inplane)
            )
            q_offset[1] = 2 * np.pi / lambdaz * distance * np.sin(outofplane)
            q_offset[2] = (
                2
                * np.pi
                / lambdaz
                * distance
                * (np.cos(inplane) * np.cos(outofplane) - 1)
            )

        else:
            raise NotImplementedError(f"rocking_angle={rocking_angle} not implemented")

        return mymatrix, q_offset


class BeamlineNANOMAX(Beamline):
    """
    Definition of MAX IV NANOMAX beamline.

    :param name: name of the beamline
    """

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

    @property
    def detector_hor(self):
        """
        Horizontal detector orientation expressed in the laboratory frame.

        We look at the detector from downstream, detector X is along the outboard
        direction. The laboratory frame convention is (z downstream, y vertical,
        x outboard).
        """
        return "x+"

    @property
    def detector_ver(self):
        """
        Vertical detector orientation expressed in the laboratory frame.

        The origin is at the top, detector Y along vertical down. The laboratory frame
        convention is (z downstream, y vertical, x outboard).
        """
        return "y-"

    def goniometer_values(self, setup, **kwargs):
        """
        Retrieve goniometer motor positions for a BCDI rocking scan.

        :param setup: the experimental setup: Class Setup
        :return: a tuple of angular values in degrees (rocking angular step, grazing
         incidence angles, inplane detector angle, outofplane detector angle). The
         grazing incidence angles are the positions of circles below the rocking circle.
        """
        # load the motor positions
        (
            theta,
            phi,
            inplane_angle,
            outofplane_angle,
            energy,
            detector_distance,
        ) = self.loader.motor_positions(setup=setup)

        # define the circles of interest for BCDI
        if setup.rocking_angle == "outofplane":  # theta rocking curve
            grazing = None  # nothing below theta at NANOMAX
            tilt_angle = theta
        elif setup.rocking_angle == "inplane":  # phi rocking curve
            grazing = (theta,)
            tilt_angle = phi
        else:
            raise ValueError('Wrong value for "rocking_angle" parameter')

        setup.check_setup(
            grazing_angle=grazing,
            inplane_angle=inplane_angle,
            outofplane_angle=outofplane_angle,
            tilt_angle=tilt_angle,
            detector_distance=detector_distance,
            energy=energy,
        )

        # NANOMAX goniometer, 2S+2D (sample: theta, phi / detector: gamma,delta)
        self.sample_angles = (theta, phi)
        self.detector_angles = (inplane_angle, outofplane_angle)

        return tilt_angle, grazing, inplane_angle, outofplane_angle

    def process_positions(
        self,
        setup,
        nb_frames,
        scan_number,
        frames_logical=None,
    ):
        """
        Load and crop/pad motor positions depending on the number of frames at NANOMAX.

        The current number of frames may be different from the original number of frames
        if the data was cropped/padded, and motor values must be processed accordingly.

        :param setup: an instance of the class Setup
        :param nb_frames: the number of frames in the current dataset
        :param scan_number: the scan number to load
        :param frames_logical: array of length the number of measured frames.
         In case of cropping/padding the number of frames changes. A frame whose
         index is set to 1 means that it is used, 0 means not used, -1 means padded
         (added) frame
        :return: a tuple of 1D arrays (sample circles, detector circles, energy)
        """
        theta, phi, gamma, delta, energy, _ = super().process_positions(
            setup=setup,
            nb_frames=nb_frames,
            scan_number=scan_number,
            frames_logical=frames_logical,
        )

        # eventually crop/pad motor values if the provided dataset was further
        # cropped/padded
        if setup.rocking_angle == "outofplane":  # theta rocking curve
            nb_steps = len(theta)
            tilt_angle = (theta[1:] - theta[0:-1]).mean()
            theta = self.process_tilt(
                theta, nb_steps=nb_steps, nb_frames=nb_frames, angular_step=tilt_angle
            )
        elif setup.rocking_angle == "inplane":  # phi rocking curve
            nb_steps = len(phi)
            tilt_angle = (phi[1:] - phi[0:-1]).mean()
            phi = self.process_tilt(
                phi, nb_steps=nb_steps, nb_frames=nb_frames, angular_step=tilt_angle
            )
        elif setup.rocking_angle == "energy":
            pass
        else:
            raise ValueError('Wrong value for "rocking_angle" parameter')

        return util.bin_parameters(
            binning=setup.detector.binning[0],
            nb_frames=nb_frames,
            params=[delta, gamma, phi, theta, energy],
        )

    def transformation_matrix(
        self,
        wavelength,
        distance,
        pixel_x,
        pixel_y,
        inplane,
        outofplane,
        grazing_angle,
        tilt,
        rocking_angle,
        verbose=True,
    ):
        """
        Calculate the transformation matrix from detector frame to laboratory frame.

        For the transformation in direct space, the length scale is in nm,
        for the transformation in reciprocal space, it is in 1/nm.

        :param wavelength: X-ray wavelength in nm
        :param distance: detector distance in nm
        :param pixel_x: horizontal detector pixel size in nm
        :param pixel_y: vertical detector pixel size in nm
        :param inplane: horizontal detector angle in radians
        :param outofplane: vertical detector angle in radians
        :param grazing_angle: angle or list of angles of the sample circles which are
         below the rotated circle
        :param tilt: angular step of the rocking curve in radians
        :param rocking_angle: "outofplane", "inplane" or "energy"
        :param verbose: True to have printed comments
        :return: a tuple of two numpy arrays

         - the transformation matrix from the detector frame to the
           laboratory frame in reciprocal space (reciprocal length scale in  1/nm), as a
           numpy array of shape (3,3)
         - the q offset (3D vector)

        """
        lambdaz = wavelength * distance
        mymatrix = np.zeros((3, 3))
        q_offset = np.zeros(3)

        if verbose:
            self.logger.info("using NANOMAX geometry")

        if rocking_angle == "outofplane":
            if grazing_angle is not None:
                raise NotImplementedError(
                    "Circle below theta not implemented for NANOMAX"
                )
            if verbose:
                self.logger.info("rocking angle is theta")
            # rocking theta angle clockwise around x
            # (phi does not matter, above eta)
            mymatrix[:, 0] = (
                2
                * np.pi
                / lambdaz
                * pixel_x
                * self.orientation_lookup[self.detector_hor]
                * np.array([-np.cos(inplane), 0, -np.sin(inplane)])
            )
            mymatrix[:, 1] = (
                2
                * np.pi
                / lambdaz
                * pixel_y
                * self.orientation_lookup[self.detector_ver]
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
                -2 * np.pi / lambdaz * distance * np.cos(outofplane) * np.sin(inplane)
            )
            q_offset[1] = 2 * np.pi / lambdaz * distance * np.sin(outofplane)
            q_offset[2] = (
                2
                * np.pi
                / lambdaz
                * distance
                * (np.cos(inplane) * np.cos(outofplane) - 1)
            )

        elif rocking_angle == "inplane":
            if isinstance(grazing_angle, Real):
                grazing_angle = (grazing_angle,)
            valid.valid_container(
                grazing_angle,
                container_types=(tuple, list),
                item_types=Real,
                length=1,
                name="grazing_angle",
            )
            if verbose:
                self.logger.info(
                    "rocking angle is phi,"
                    f" theta={grazing_angle[0] * 180 / np.pi:.3f} deg"
                )
            # rocking phi angle clockwise around y,
            # incident angle theta is non zero (theta below phi)
            mymatrix[:, 0] = (
                2
                * np.pi
                / lambdaz
                * pixel_x
                * self.orientation_lookup[self.detector_hor]
                * np.array([-np.cos(inplane), 0, -np.sin(inplane)])
            )
            mymatrix[:, 1] = (
                2
                * np.pi
                / lambdaz
                * pixel_y
                * self.orientation_lookup[self.detector_ver]
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
                        np.sin(grazing_angle[0]) * np.sin(inplane) * np.cos(outofplane),
                        np.cos(grazing_angle[0]) * np.sin(inplane) * np.cos(outofplane),
                    ]
                )
            )
            q_offset[0] = (
                -2 * np.pi / lambdaz * distance * np.cos(outofplane) * np.sin(inplane)
            )
            q_offset[1] = 2 * np.pi / lambdaz * distance * np.sin(outofplane)
            q_offset[2] = (
                2
                * np.pi
                / lambdaz
                * distance
                * (np.cos(inplane) * np.cos(outofplane) - 1)
            )

        else:
            raise NotImplementedError(f"rocking_angle={rocking_angle} not implemented")

        return mymatrix, q_offset


class BeamlineP10(Beamline):
    """
    Definition of PETRA III P10 beamline.

    :param name: name of the beamline
    """

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

    @property
    def detector_hor(self):
        """
        Horizontal detector orientation expressed in the laboratory frame.

        We look at the detector from upstream, detector X is opposite to the outboard
        direction. The laboratory frame convention is (z downstream, y vertical,
        x outboard).
        """
        return "x-"

    @property
    def detector_ver(self):
        """
        Vertical detector orientation expressed in the laboratory frame.

        The origin is at the top, detector Y along vertical down. The laboratory frame
        convention is (z downstream, y vertical, x outboard).
        """
        return "y-"

    def goniometer_values(self, setup, **kwargs):
        """
        Retrieve goniometer motor positions for a BCDI rocking scan.

        :param setup: the experimental setup: Class Setup
        :return: a tuple of angular values in degrees (rocking angular step, grazing
         incidence angles, inplane detector angle, outofplane detector angle). The
         grazing incidence angles are the positions of circles below the rocking circle.
        """
        # load the motor positions
        (
            mu,
            om,
            chi,
            phi,
            inplane_angle,
            outofplane_angle,
            energy,
            detector_distance,
        ) = self.loader.motor_positions(setup=setup)

        # define the circles of interest for BCDI
        if setup.rocking_angle == "outofplane":  # om rocking curve
            grazing = (mu,)
            tilt_angle = om
        elif setup.rocking_angle == "inplane":  # phi rocking curve
            grazing = (mu, om, chi)
            tilt_angle = phi
        else:
            raise ValueError('Wrong value for "rocking_angle" parameter')

        setup.check_setup(
            grazing_angle=grazing,
            inplane_angle=inplane_angle,
            outofplane_angle=outofplane_angle,
            tilt_angle=tilt_angle,
            detector_distance=detector_distance,
            energy=energy,
        )

        # P10 goniometer, 4S+2D (sample: mu, omega, chi, phi / detector: gamma, delta)
        self.sample_angles = (mu, om, chi, phi)
        self.detector_angles = (inplane_angle, outofplane_angle)

        return tilt_angle, grazing, inplane_angle, outofplane_angle

    def process_positions(
        self,
        setup,
        nb_frames,
        scan_number,
        frames_logical=None,
    ):
        """
        Load and crop/pad motor positions depending on the number of frames at P10.

        The current number of frames may be different from the original number of frames
        if the data was cropped/padded, and motor values must be processed accordingly.

        :param setup: an instance of the class Setup
        :param nb_frames: the number of frames in the current dataset
        :param scan_number: the scan number to load
        :param frames_logical: array of length the number of measured frames.
         In case of cropping/padding the number of frames changes. A frame whose
         index is set to 1 means that it is used, 0 means not used, -1 means padded
         (added) frame
        :return: a tuple of 1D arrays (sample circles, detector circles, energy)
        """
        mu, om, chi, phi, gamma, delta, energy, _ = super().process_positions(
            setup=setup,
            nb_frames=nb_frames,
            scan_number=scan_number,
            frames_logical=frames_logical,
        )

        # eventually crop/pad motor values if the provided dataset was further
        # cropped/padded
        self.logger.info("chi", chi)
        self.logger.info("mu", mu)
        if setup.rocking_angle == "outofplane":  # om rocking curve
            self.logger.info("phi", phi)
            nb_steps = len(om)
            tilt_angle = (om[1:] - om[0:-1]).mean()
            om = self.process_tilt(
                om, nb_steps=nb_steps, nb_frames=nb_frames, angular_step=tilt_angle
            )
        elif setup.rocking_angle == "inplane":  # phi rocking curve
            self.logger.info("om", om)
            nb_steps = len(phi)
            tilt_angle = (phi[1:] - phi[0:-1]).mean()
            phi = self.process_tilt(
                phi, nb_steps=nb_steps, nb_frames=nb_frames, angular_step=tilt_angle
            )
        else:
            raise ValueError('Wrong value for "rocking_angle" parameter')

        return util.bin_parameters(
            binning=setup.detector.binning[0],
            nb_frames=nb_frames,
            params=[mu, om, chi, phi, gamma, delta, energy],
        )

    def transformation_matrix(
        self,
        wavelength,
        distance,
        pixel_x,
        pixel_y,
        inplane,
        outofplane,
        grazing_angle,
        tilt,
        rocking_angle,
        verbose=True,
    ):
        """
        Calculate the transformation matrix from detector frame to laboratory frame.

        For the transformation in direct space, the length scale is in nm,
        for the transformation in reciprocal space, it is in 1/nm.

        :param wavelength: X-ray wavelength in nm
        :param distance: detector distance in nm
        :param pixel_x: horizontal detector pixel size in nm
        :param pixel_y: vertical detector pixel size in nm
        :param inplane: horizontal detector angle in radians
        :param outofplane: vertical detector angle in radians
        :param grazing_angle: angle or list of angles of the sample circles which are
         below the rotated circle
        :param tilt: angular step of the rocking curve in radians
        :param rocking_angle: "outofplane", "inplane" or "energy"
        :param verbose: True to have printed comments
        :return: a tuple of two numpy arrays

         - the transformation matrix from the detector frame to the
           laboratory frame in reciprocal space (reciprocal length scale in  1/nm), as a
           numpy array of shape (3,3)
         - the q offset (3D vector)

        """
        if self.name == "P10_SAXS":
            raise ValueError("Method invalid for P10_SAXS")

        lambdaz = wavelength * distance
        mymatrix = np.zeros((3, 3))
        q_offset = np.zeros(3)

        if verbose:
            self.logger.info("using PETRAIII P10 geometry")

        if isinstance(grazing_angle, Real):
            grazing_angle = (grazing_angle,)
        valid.valid_container(
            grazing_angle,
            container_types=(tuple, list),
            item_types=Real,
            min_length=1,
            name="grazing_angle",
        )

        if rocking_angle == "outofplane":
            if verbose:
                self.logger.info(
                    f"rocking angle is om, mu={grazing_angle[0] * 180 / np.pi:.3f} deg"
                )
            # rocking omega angle clockwise around x at mu=0,
            # chi potentially non zero (chi below omega)
            # (phi does not matter, above eta)
            mymatrix[:, 0] = (
                2
                * np.pi
                / lambdaz
                * pixel_x
                * self.orientation_lookup[self.detector_hor]
                * np.array([-np.cos(inplane), 0, np.sin(inplane)])
            )
            mymatrix[:, 1] = (
                2
                * np.pi
                / lambdaz
                * pixel_y
                * self.orientation_lookup[self.detector_ver]
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
                2 * np.pi / lambdaz * distance * np.cos(outofplane) * np.sin(inplane)
            )
            q_offset[1] = 2 * np.pi / lambdaz * distance * np.sin(outofplane)
            q_offset[2] = (
                2
                * np.pi
                / lambdaz
                * distance
                * (np.cos(inplane) * np.cos(outofplane) - 1)
            )

        elif rocking_angle == "inplane":
            if len(grazing_angle) != 3:
                raise ValueError("grazing_angle should be of length 3")
            if not isclose(grazing_angle[0], 0, rel_tol=1e-09, abs_tol=1e-09):
                raise NotImplementedError(
                    "Non-zero mu not implemented for inplane rocking curve at P10"
                )
            if verbose:
                self.logger.info(
                    f"rocking angle is phi,"
                    f" mu={grazing_angle[0] * 180 / np.pi:.3f} deg,"
                    f" om={grazing_angle[1] * 180 / np.pi:.3f} deg,"
                    f" chi={grazing_angle[2] * 180 / np.pi:.3f} deg"
                )

            # rocking phi angle clockwise around y,
            # omega and chi potentially non zero (chi below omega below phi)
            mymatrix[:, 0] = (
                2
                * np.pi
                / lambdaz
                * pixel_x
                * self.orientation_lookup[self.detector_hor]
                * np.array([-np.cos(inplane), 0, np.sin(inplane)])
            )
            mymatrix[:, 1] = (
                2
                * np.pi
                / lambdaz
                * pixel_y
                * self.orientation_lookup[self.detector_ver]
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
                2 * np.pi / lambdaz * distance * np.cos(outofplane) * np.sin(inplane)
            )
            q_offset[1] = 2 * np.pi / lambdaz * distance * np.sin(outofplane)
            q_offset[2] = (
                2
                * np.pi
                / lambdaz
                * distance
                * (np.cos(inplane) * np.cos(outofplane) - 1)
            )

        else:
            raise NotImplementedError(f"rocking_angle={rocking_angle} not implemented")

        return mymatrix, q_offset


class BeamlineP10SAXS(BeamlineP10):
    """
    Definition of PETRA III P10 beamline for the USAXS setup.

    :param name: name of the beamline
    """

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

    @staticmethod
    def cartesian2polar(nb_pixels, pivot, offset_angle, debugging=False):
        """
        Find the corresponding polar coordinates of a cartesian 2D grid.

        The grid is assumed perpendicular to the rotation axis.

        :param nb_pixels: number of pixels of the axis of the squared grid
        :param pivot: position in pixels of the origin of the polar coordinates system
        :param offset_angle: reference angle for the angle wrapping
        :param debugging: True to see more plots
        :return: the corresponding 1D array of angular coordinates, 1D array of radial
         coordinates
        """
        z_interp, x_interp = np.meshgrid(
            np.linspace(-pivot, -pivot + nb_pixels, num=nb_pixels, endpoint=False),
            np.linspace(pivot - nb_pixels, pivot, num=nb_pixels, endpoint=False),
            indexing="ij",
        )  # z_interp changes along rows, x_interp along columns
        # z_interp downstream, same direction as detector X rotated by +90deg
        # x_interp along outboard opposite to detector X

        # map these points to (cdi_angle, X), the measurement polar coordinates
        interp_angle = util.wrap(
            obj=np.arctan2(z_interp, -x_interp),
            start_angle=offset_angle * np.pi / 180,
            range_angle=np.pi,
        )  # in radians, located in the range [start_angle, start_angle+np.pi[

        sign_array = -1 * np.sign(np.cos(interp_angle)) * np.sign(x_interp)
        sign_array[x_interp == 0] = np.sign(z_interp[x_interp == 0]) * np.sign(
            interp_angle[x_interp == 0]
        )

        interp_radius = np.multiply(sign_array, hypot(x_interp, z_interp))

        if debugging:
            gu.imshow_plot(
                interp_angle * 180 / np.pi,
                plot_colorbar=True,
                scale="linear",
                labels=("Qx (z_interp)", "Qy (x_interp)"),
                title="calculated polar angle for the 2D grid",
            )

            gu.imshow_plot(
                sign_array,
                plot_colorbar=True,
                scale="linear",
                labels=("Qx (z_interp)", "Qy (x_interp)"),
                title="sign_array",
            )

            gu.imshow_plot(
                interp_radius,
                plot_colorbar=True,
                scale="linear",
                labels=("Qx (z_interp)", "Qy (x_interp)"),
                title="calculated polar radius for the 2D grid",
            )
        return interp_angle, interp_radius

    @staticmethod
    def ewald_curvature_saxs(
        wavelength,
        beam_direction,
        pixelsize_x,
        pixelsize_y,
        distance,
        array_shape,
        cdi_angle,
        direct_beam,
        anticlockwise=True,
    ):
        """
        Calculate q values taking into account the curvature of Ewald sphere.

        Based on the CXI detector geometry convention: Laboratory frame: z downstream,
        y vertical up, x outboard. Detector axes: Y vertical and X horizontal
        (detector Y is vertical down at out-of-plane angle=0, detector X is opposite
        to x at inplane angle=0)

        :param wavelength: X-ray wavelength in nm
        :param beam_direction: direction of the incident X-ray beam.
        :param distance: detector distance in nm
        :param pixelsize_x: horizontal binned detector pixel size in nm
        :param pixelsize_y: vertical binned detector pixel size in nm
        :param array_shape: tuple of three integers, shape of the dataset to be gridded
        :param cdi_angle: 1D array of measurement angles in degrees
        :param direct_beam: tuple of 2 integers, position of the direction beam (V, H)
        :param anticlockwise: True if the rotation is anticlockwise
        :return: qx, qz, qy values in the laboratory frame
         (downstream, vertical up, outboard). Each array has the shape: nb_pixel_x *
         nb_pixel_y * nb_angles
        """
        #########################
        # check some parameters #
        #########################
        valid.valid_container(
            direct_beam,
            container_types=(tuple, list),
            length=2,
            item_types=int,
            name="direct_beam",
        )
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
        valid.valid_item(anticlockwise, allowed_types=bool, name="anticlockwise")

        # calculate q values of the measurement
        nbz, nby, nbx = array_shape
        qz = np.empty((nbz, nby, nbx), dtype=float)
        qy = np.empty((nbz, nby, nbx), dtype=float)
        qx = np.empty((nbz, nby, nbx), dtype=float)

        # calculate q values of the detector frame
        # for each angular position and stack them
        for idx, item in enumerate(cdi_angle):
            angle = item * np.pi / 180
            if not anticlockwise:
                rotation_matrix = np.array(
                    [
                        [np.cos(angle), 0, -np.sin(angle)],
                        [0, 1, 0],
                        [np.sin(angle), 0, np.cos(angle)],
                    ]
                )
            else:
                rotation_matrix = np.array(
                    [
                        [np.cos(angle), 0, np.sin(angle)],
                        [0, 1, 0],
                        [-np.sin(angle), 0, np.cos(angle)],
                    ]
                )

            myy, myx = np.meshgrid(
                np.linspace(
                    -direct_beam[0], -direct_beam[0] + nby, num=nby, endpoint=False
                ),
                np.linspace(
                    -direct_beam[1], -direct_beam[1] + nbx, num=nbx, endpoint=False
                ),
                indexing="ij",
            )

            two_theta = np.arctan(myx * pixelsize_x / distance)
            alpha_f = np.arctan(
                np.divide(
                    myy * pixelsize_y,
                    np.sqrt(distance**2 + np.power(myx * pixelsize_x, 2)),
                )
            )

            qlab0 = (
                2
                * np.pi
                / wavelength
                * (np.cos(alpha_f) * np.cos(two_theta) - beam_direction[0])
            )
            # along z* downstream
            qlab1 = 2 * np.pi / wavelength * (np.sin(alpha_f) - beam_direction[1])
            # along y* vertical up
            qlab2 = (
                2
                * np.pi
                / wavelength
                * (np.cos(alpha_f) * np.sin(two_theta) - beam_direction[2])
            )
            # along x* outboard

            qx[idx, :, :] = (
                rotation_matrix[0, 0] * qlab0
                + rotation_matrix[0, 1] * qlab1
                + rotation_matrix[0, 2] * qlab2
            )
            qz[idx, :, :] = (
                rotation_matrix[1, 0] * qlab0
                + rotation_matrix[1, 1] * qlab1
                + rotation_matrix[1, 2] * qlab2
            )
            qy[idx, :, :] = (
                rotation_matrix[2, 0] * qlab0
                + rotation_matrix[2, 1] * qlab1
                + rotation_matrix[2, 2] * qlab2
            )

        return qx, qz, qy

    def goniometer_values(self, setup, **kwargs):
        """
        Retrieve goniometer motor positions for a CDI tomographic scan.

        :param setup: the experimental setup: Class Setup
        :return: a tuple of angular values in degrees (rocking angular step, grazing
         incidence angles, inplane detector angle, outofplane detector angle). The
         grazing incidence angles are the positions of circles below the rocking circle.
        """
        # load the motor positions
        phi, energy, detector_distance = self.loader.motor_positions(setup=setup)

        # define the circles of interest for CDI
        # no circle yet below phi at P10
        if setup.rocking_angle == "inplane":  # phi rocking curve
            grazing = (0,)
            tilt_angle = phi
        else:
            raise ValueError('Wrong value for "rocking_angle" parameter')

        setup.check_setup(
            grazing_angle=grazing,
            inplane_angle=0,
            outofplane_angle=0,
            tilt_angle=tilt_angle,
            detector_distance=detector_distance,
            energy=energy,
        )

        # P10 SAXS goniometer, 1S + 0D (sample: phi / detector: None)
        self.sample_angles = (phi,)
        self.detector_angles = (0, 0)

        return tilt_angle, grazing, 0, 0


class BeamlineSIXS(Beamline):
    """
    Definition of SOLEIL SIXS beamline.

    :param name: name of the beamline
    """

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

    @property
    def detector_hor(self):
        """
        Horizontal detector orientation expressed in the laboratory frame.

        We look at the detector from downstream, detector X is along the outboard
        direction. The laboratory frame convention is (z downstream, y vertical,
        x outboard).
        """
        return "x+"

    @property
    def detector_ver(self):
        """
        Vertical detector orientation expressed in the laboratory frame.

        The origin is at the top, detector Y along vertical down. The laboratory frame
        convention is (z downstream, y vertical, x outboard).
        """
        return "y-"

    def goniometer_values(self, setup, **kwargs):
        """
        Retrieve goniometer motor positions for a BCDI rocking scan at SIXS.

        :param setup: the experimental setup: Class Setup
        :return: a tuple of angular values in degrees (rocking angular step, grazing
         incidence angles, inplane detector angle, outofplane detector angle). The
         grazing incidence angles are the positions of circles below the rocking circle.
        """
        # load the motor positions
        (
            beta,
            mu,
            inplane_angle,
            outofplane_angle,
            energy,
            detector_distance,
        ) = self.loader.motor_positions(setup=setup)

        # define the circles of interest for BCDI
        if setup.rocking_angle == "inplane":  # mu rocking curve
            grazing = (beta,)  # beta below the whole diffractomter at SIXS
            tilt_angle = mu
        elif setup.rocking_angle == "outofplane":
            raise NotImplementedError(
                "outofplane rocking curve not implemented for SIXS"
            )
        else:
            raise ValueError("Out-of-plane rocking curve not implemented for SIXS")

        setup.check_setup(
            grazing_angle=grazing,
            inplane_angle=inplane_angle,
            outofplane_angle=outofplane_angle,
            tilt_angle=tilt_angle,
            detector_distance=detector_distance,
            energy=energy,
        )

        # SIXS goniometer, 2S+3D (sample: beta, mu / detector: beta, gamma, del)
        self.sample_angles = (beta, mu)
        self.detector_angles = (beta, inplane_angle, outofplane_angle)

        return tilt_angle, grazing, inplane_angle, outofplane_angle

    def process_positions(
        self,
        setup,
        nb_frames,
        scan_number,
        frames_logical=None,
    ):
        """
        Load and crop/pad motor positions depending on the number of frames at SIXS.

        The current number of frames may be different from the original number of frames
        if the data was cropped/padded, and motor values must be processed accordingly.

        :param setup: an instance of the class Setup
        :param nb_frames: the number of frames in the current dataset
        :param scan_number: the scan number to load
        :param frames_logical: array of length the number of measured frames.
         In case of cropping/padding the number of frames changes. A frame whose
         index is set to 1 means that it is used, 0 means not used, -1 means padded
         (added) frame
        :return: a tuple of 1D arrays (sample circles, detector circles, energy)
        """
        beta, mu, gamma, delta, energy, _ = super().process_positions(
            setup=setup,
            nb_frames=nb_frames,
            scan_number=scan_number,
            frames_logical=frames_logical,
        )

        # eventually crop/pad motor values if the provided dataset was further
        # cropped/padded
        self.logger.info("beta", beta)
        if setup.rocking_angle == "inplane":  # mu rocking curve
            nb_steps = len(mu)
            tilt_angle = (mu[1:] - mu[0:-1]).mean()
            mu = self.process_tilt(
                mu, nb_steps=nb_steps, nb_frames=nb_frames, angular_step=tilt_angle
            )
        else:
            raise ValueError("Out-of-plane rocking curve not implemented for SIXS")

        return util.bin_parameters(
            binning=setup.detector.binning[0],
            nb_frames=nb_frames,
            params=[beta, mu, beta, gamma, delta, energy],
        )

    def transformation_matrix(
        self,
        wavelength,
        distance,
        pixel_x,
        pixel_y,
        inplane,
        outofplane,
        grazing_angle,
        tilt,
        rocking_angle,
        verbose=True,
    ):
        """
        Calculate the transformation matrix from detector frame to laboratory frame.

        For the transformation in direct space, the length scale is in nm,
        for the transformation in reciprocal space, it is in 1/nm.

        :param wavelength: X-ray wavelength in nm
        :param distance: detector distance in nm
        :param pixel_x: horizontal detector pixel size in nm
        :param pixel_y: vertical detector pixel size in nm
        :param inplane: horizontal detector angle in radians
        :param outofplane: vertical detector angle in radians
        :param grazing_angle: angle or list of angles of the sample circles which are
         below the rotated circle
        :param tilt: angular step of the rocking curve in radians
        :param rocking_angle: "outofplane", "inplane" or "energy"
        :param verbose: True to have printed comments
        :return: a tuple of two numpy arrays

         - the transformation matrix from the detector frame to the
           laboratory frame in reciprocal space (reciprocal length scale in  1/nm), as a
           numpy array of shape (3,3)
         - the q offset (3D vector)

        """
        lambdaz = wavelength * distance
        mymatrix = np.zeros((3, 3))
        q_offset = np.zeros(3)

        if verbose:
            self.logger.info("using SIXS geometry")

        if isinstance(grazing_angle, Real):
            grazing_angle = (grazing_angle,)
        valid.valid_container(
            grazing_angle,
            container_types=(tuple, list),
            item_types=Real,
            length=1,
            name="grazing_angle",
        )

        if rocking_angle == "inplane":
            if verbose:
                self.logger.info(
                    "rocking angle is mu,"
                    f" beta={grazing_angle[0] * 180 / np.pi:.3f} deg"
                )

            # rocking mu angle anti-clockwise around y
            mymatrix[:, 0] = (
                2
                * np.pi
                / lambdaz
                * pixel_x
                * self.orientation_lookup[self.detector_hor]
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
                * self.orientation_lookup[self.detector_ver]
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
                        np.cos(grazing_angle[0]) - np.cos(inplane) * np.cos(outofplane),
                        np.sin(grazing_angle[0]) * np.sin(inplane) * np.cos(outofplane),
                        np.cos(grazing_angle[0]) * np.sin(inplane) * np.cos(outofplane),
                    ]
                )
            )
            q_offset[0] = (
                2 * np.pi / lambdaz * distance * np.cos(outofplane) * np.sin(inplane)
            )
            q_offset[1] = (
                2
                * np.pi
                / lambdaz
                * distance
                * (
                    np.cos(grazing_angle[0]) * np.sin(outofplane)
                    + np.sin(grazing_angle[0]) * np.cos(inplane) * np.cos(outofplane)
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
            raise NotImplementedError(f"rocking_angle={rocking_angle} not implemented")

        return mymatrix, q_offset


class Beamline34ID(Beamline):
    """
    Definition of APS 34ID-C beamline.

    :param name: name of the beamline
    """

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

    @property
    def detector_hor(self):
        """
        Horizontal detector orientation expressed in the laboratory frame.

        We look at the detector from downstream, detector X is along the outboard
        direction. The laboratory frame convention is (z downstream, y vertical,
        x outboard).
        """
        return "x+"

    @property
    def detector_ver(self):
        """
        Vertical detector orientation expressed in the laboratory frame.

        The origin is at the top, detector Y along vertical down. The laboratory frame
        convention is (z downstream, y vertical, x outboard).
        """
        return "y-"

    def goniometer_values(self, setup, **kwargs):
        """
        Retrieve goniometer motor positions for a BCDI rocking scan.

        :param setup: the experimental setup: Class Setup
        :param kwargs:
         - 'scan_number': the scan number to load

        :return: a tuple of angular values in degrees (rocking angular step, grazing
         incidence angles, inplane detector angle, outofplane detector angle). The
         grazing incidence angles are the positions of circles below the rocking circle.
        """
        # load kwargs
        scan_number = kwargs["scan_number"]

        # check some parameter
        valid.valid_item(
            scan_number, allowed_types=int, min_excluded=0, name="scan_number"
        )

        # load the motor positions
        (
            theta,
            chi,
            phi,
            inplane_angle,
            outofplane_angle,
            energy,
            detector_distance,
        ) = self.loader.motor_positions(setup=setup, scan_number=scan_number)

        # define the circles of interest for BCDI
        if setup.rocking_angle == "inplane":
            # theta is the inplane rotation around the vertical axis at 34ID
            grazing = None  # theta (inplane) is below phi
            tilt_angle = theta
        elif setup.rocking_angle == "outofplane":
            # phi is the incident angle (out of plane rotation) at 34ID
            grazing = (theta, chi)
            tilt_angle = phi
        else:
            raise ValueError('Wrong value for "rocking_angle" parameter')

        setup.check_setup(
            grazing_angle=grazing,
            inplane_angle=inplane_angle,
            outofplane_angle=outofplane_angle,
            tilt_angle=tilt_angle,
            detector_distance=detector_distance,
            energy=energy,
        )

        # 34ID-C goniometer, 3S+2D (sample: theta (inplane), chi (close to 90 deg),
        # phi (out of plane)   detector: delta (inplane), gamma)
        self.sample_angles = (theta, chi, phi)
        self.detector_angles = (inplane_angle, outofplane_angle)

        return tilt_angle, grazing, inplane_angle, outofplane_angle

    def process_positions(
        self,
        setup,
        nb_frames,
        scan_number,
        frames_logical=None,
    ):
        """
        Load and crop/pad motor positions depending on the number of frames at 34ID-C.

        The current number of frames may be different from the original number of frames
        if the data was cropped/padded, and motor values must be processed accordingly.

        :param setup: an instance of the class Setup
        :param nb_frames: the number of frames in the current dataset
        :param scan_number: the scan number to load
        :param frames_logical: array of length the number of measured frames.
         In case of cropping/padding the number of frames changes. A frame whose
         index is set to 1 means that it is used, 0 means not used, -1 means padded
         (added) frame
        :return: a tuple of 1D arrays (sample circles, detector circles, energy)
        """
        theta, chi, phi, delta, gamma, energy, _ = super().process_positions(
            setup=setup,
            nb_frames=nb_frames,
            scan_number=scan_number,
            frames_logical=frames_logical,
        )

        # eventually crop/pad motor values if the provided dataset was further
        # cropped/padded
        if setup.rocking_angle == "outofplane":  # phi rocking curve
            nb_steps = len(phi)
            tilt_angle = (phi[1:] - phi[0:-1]).mean()
            phi = self.process_tilt(
                phi, nb_steps=nb_steps, nb_frames=nb_frames, angular_step=tilt_angle
            )
        elif setup.rocking_angle == "inplane":  # theta rocking curve
            nb_steps = len(theta)
            tilt_angle = (theta[1:] - theta[0:-1]).mean()
            theta = self.process_tilt(
                theta, nb_steps=nb_steps, nb_frames=nb_frames, angular_step=tilt_angle
            )
        elif setup.rocking_angle == "energy":
            pass
        else:
            raise ValueError('Wrong value for "rocking_angle" parameter')

        return util.bin_parameters(
            binning=setup.detector.binning[0],
            nb_frames=nb_frames,
            params=[theta, chi, phi, delta, gamma, energy],
        )

    def transformation_matrix(
        self,
        wavelength,
        distance,
        pixel_x,
        pixel_y,
        inplane,
        outofplane,
        grazing_angle,
        tilt,
        rocking_angle,
        verbose=True,
    ):
        """
        Calculate the transformation matrix from detector frame to laboratory frame.

        For the transformation in direct space, the length scale is in nm,
        for the transformation in reciprocal space, it is in 1/nm.

        :param wavelength: X-ray wavelength in nm
        :param distance: detector distance in nm
        :param pixel_x: horizontal detector pixel size in nm
        :param pixel_y: vertical detector pixel size in nm
        :param inplane: horizontal detector angle in radians
        :param outofplane: vertical detector angle in radians
        :param grazing_angle: angle or list of angles of the sample circles which are
         below the rotated circle
        :param tilt: angular step of the rocking curve in radians
        :param rocking_angle: "outofplane", "inplane" or "energy"
        :param verbose: True to have printed comments
        :return: a tuple of two numpy arrays

         - the transformation matrix from the detector frame to the
           laboratory frame in reciprocal space (reciprocal length scale in  1/nm), as a
           numpy array of shape (3,3)
         - the q offset (3D vector)

        """
        lambdaz = wavelength * distance
        mymatrix = np.zeros((3, 3))
        q_offset = np.zeros(3)

        if verbose:
            self.logger.info("using APS 34ID geometry")

        if rocking_angle == "inplane":
            if grazing_angle is not None:
                raise NotImplementedError(
                    "Circle below theta not implemented for 34ID-C"
                )
            if verbose:
                self.logger.info(
                    "rocking angle is theta, no grazing angle (chi, phi above theta)"
                )
            # rocking theta angle anti-clockwise around y
            mymatrix[:, 0] = (
                2
                * np.pi
                / lambdaz
                * pixel_x
                * self.orientation_lookup[self.detector_hor]
                * np.array([-np.cos(inplane), 0, np.sin(inplane)])
            )
            mymatrix[:, 1] = (
                2
                * np.pi
                / lambdaz
                * pixel_y
                * self.orientation_lookup[self.detector_ver]
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
                2 * np.pi / lambdaz * distance * np.cos(outofplane) * np.sin(inplane)
            )
            q_offset[1] = 2 * np.pi / lambdaz * distance * np.sin(outofplane)
            q_offset[2] = (
                2
                * np.pi
                / lambdaz
                * distance
                * (np.cos(inplane) * np.cos(outofplane) - 1)
            )

        elif rocking_angle == "outofplane":
            valid.valid_container(
                grazing_angle,
                container_types=(tuple, list),
                item_types=Real,
                length=2,
                name="grazing_angle",
            )
            if verbose:
                self.logger.info(
                    "rocking angle is phi,"
                    f" theta={grazing_angle[0] * 180 / np.pi:.3f} deg,"
                    f" chi={grazing_angle[1] * 180 / np.pi:.3f} deg"
                )
            # rocking phi angle anti-clockwise around x
            mymatrix[:, 0] = (
                2
                * np.pi
                / lambdaz
                * pixel_x
                * self.orientation_lookup[self.detector_hor]
                * np.array([-np.cos(inplane), 0, np.sin(inplane)])
            )
            mymatrix[:, 1] = (
                2
                * np.pi
                / lambdaz
                * pixel_y
                * self.orientation_lookup[self.detector_ver]
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
                            np.cos(grazing_angle[1])
                            * (1 - np.cos(inplane) * np.cos(outofplane))
                            - np.sin(grazing_angle[0])
                            * np.sin(grazing_angle[1])
                            * np.sin(outofplane)
                        ),
                        np.sin(grazing_angle[1])
                        * (
                            np.cos(grazing_angle[0])
                            * (np.cos(inplane) * np.cos(outofplane) - 1)
                            + np.sin(grazing_angle[0])
                            * np.sin(inplane)
                            * np.cos(outofplane)
                        ),
                        (
                            np.cos(grazing_angle[1])
                            * np.sin(inplane)
                            * np.cos(outofplane)
                            - np.sin(grazing_angle[1])
                            * np.cos(grazing_angle[0])
                            * np.sin(outofplane)
                        ),
                    ]
                )
            )
            q_offset[0] = (
                2 * np.pi / lambdaz * distance * np.sin(inplane) * np.cos(outofplane)
            )
            q_offset[1] = 2 * np.pi / lambdaz * distance * np.sin(outofplane)
            q_offset[2] = (
                2
                * np.pi
                / lambdaz
                * distance
                * (np.cos(inplane) * np.cos(outofplane) - 1)
            )

        else:
            raise NotImplementedError(f"rocking_angle={rocking_angle} not implemented")

        return mymatrix, q_offset
