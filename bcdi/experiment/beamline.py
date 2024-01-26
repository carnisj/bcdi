# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

"""
Implementation of beamline-specific classes.

API Reference
-------------

"""

import logging
from math import isclose
from numbers import Real
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np

from bcdi.experiment.beamline_factory import Beamline, BeamlineGoniometer, BeamlineSaxs
from bcdi.utils import utilities as util
from bcdi.utils import validation as valid

if TYPE_CHECKING:
    from bcdi.experiment.setup import Setup

module_logger = logging.getLogger(__name__)


def create_beamline(name, sample_offsets=None, **kwargs) -> Beamline:
    """
    Create the instance of the beamline.

    :param name: str, name of the beamline
    :param sample_offsets: list or tuple of angles in degrees, corresponding to
     the offsets of each of the sample circles (the offset for the most outer circle
     should be at index 0). The number of circles is beamline dependent. Convention:
     the sample offsets will be subtracted to measurement the motor values.
    :param kwargs:
     - optional beamline-dependent parameters
     - 'logger': an optional logger

    :return: the corresponding beamline instance
    """
    if name in {"ID01", "ID01BLISS"}:
        return BeamlineID01(name=name, sample_offsets=sample_offsets, **kwargs)
    if name == "BM02":
        return BeamlineBM02(name=name, sample_offsets=sample_offsets, **kwargs)
    if name == "ID27":
        return BeamlineID27(name=name, sample_offsets=sample_offsets, **kwargs)
    if name in {"SIXS_2018", "SIXS_2019"}:
        return BeamlineSIXS(name=name, sample_offsets=sample_offsets, **kwargs)
    if name == "34ID":
        return Beamline34ID(name=name, sample_offsets=sample_offsets, **kwargs)
    if name == "P10":
        return BeamlineP10(name=name, sample_offsets=sample_offsets, **kwargs)
    if name == "P10_SAXS":
        return BeamlineP10SAXS(name=name, sample_offsets=sample_offsets, **kwargs)
    if name == "CRISTAL":
        return BeamlineCRISTAL(name=name, sample_offsets=sample_offsets, **kwargs)
    if name == "NANOMAX":
        return BeamlineNANOMAX(name=name, sample_offsets=sample_offsets, **kwargs)
    raise ValueError(f"Beamline {name} not supported")


class BeamlineCRISTAL(BeamlineGoniometer):
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
        else:  # energy scan
            grazing = None
            tilt_angle = energy

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
                mgomega, nb_steps=nb_steps, nb_frames=nb_frames, step_size=tilt_angle
            )
        elif setup.rocking_angle == "inplane":  # mgphi rocking curve
            self.logger.info(f"mgomega {mgomega}")
            nb_steps = len(mgphi)
            tilt_angle = (mgphi[1:] - mgphi[0:-1]).mean()
            mgphi = self.process_tilt(
                mgphi, nb_steps=nb_steps, nb_frames=nb_frames, step_size=tilt_angle
            )
        elif setup.rocking_angle == "energy":
            nb_steps = len(energy)
            tilt_angle = (energy[1:] - energy[0:-1]).mean()
            energy = self.process_tilt(
                energy, nb_steps=nb_steps, nb_frames=nb_frames, step_size=tilt_angle
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


class BeamlineID01(BeamlineGoniometer):
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
        else:  # energy scan
            grazing = None
            tilt_angle = energy

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
            nb_steps = len(eta)
            tilt_angle = (eta[1:] - eta[0:-1]).mean()
            eta = self.process_tilt(
                eta, nb_steps=nb_steps, nb_frames=nb_frames, step_size=tilt_angle
            )
        elif setup.rocking_angle == "inplane":  # phi rocking curve
            nb_steps = len(phi)
            tilt_angle = (phi[1:] - phi[0:-1]).mean()
            phi = self.process_tilt(
                phi, nb_steps=nb_steps, nb_frames=nb_frames, step_size=tilt_angle
            )
        elif setup.rocking_angle == "energy":
            nb_steps = len(energy)
            tilt_angle = (energy[1:] - energy[0:-1]).mean()
            energy = self.process_tilt(
                energy, nb_steps=nb_steps, nb_frames=nb_frames, step_size=tilt_angle
            )
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
                    "rocking angle is phi,"
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


class BeamlineBM02(BeamlineGoniometer):
    """
    Definition of ESRF BM02 beamline.

    :param name: name of the beamline
    """

    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name=name, **kwargs)

    @property
    def detector_hor(self) -> str:
        """
        Horizontal detector orientation expressed in the laboratory frame.

        We look at the detector from upstream, detector X is along the inboard
        direction. The laboratory frame convention is (z downstream, y vertical,
        x outboard).
        """
        return "x-"

    @property
    def detector_ver(self) -> str:
        """
        Vertical detector orientation expressed in the laboratory frame.

        The origin is at the top, detector Y along vertical down. The laboratory frame
        convention is (z downstream, y vertical, x outboard).
        """
        return "y-"

    def goniometer_values(self, setup: "Setup", **kwargs) -> Tuple[Any, ...]:
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
            th,
            chi,
            phi,
            inplane_angle,
            outofplane_angle,
            energy,
            detector_distance,
        ) = self.loader.motor_positions(
            setup=setup,
            scan_number=scan_number,
        )

        grazing: Optional[Tuple[Real, ...]] = None

        # define the circles of interest in BCDI
        if setup.rocking_angle == "outofplane":  # th rocking curve
            grazing = (mu,)
            tilt_angle = th
        elif setup.rocking_angle == "inplane":  # phi rocking curve
            grazing = (mu, th, chi)
            tilt_angle = phi
        else:  # energy scan
            tilt_angle = energy

        setup.check_setup(
            grazing_angle=grazing,
            inplane_angle=inplane_angle,
            outofplane_angle=outofplane_angle,
            tilt_angle=tilt_angle,
            detector_distance=detector_distance,
            energy=energy,
        )

        # BM02 goniometer, 4S+2D (sample: mu, th, chi, phi / detector: nu, tth)
        self.sample_angles = (mu, th, chi, phi)
        self.detector_angles = (inplane_angle, outofplane_angle)

        return tilt_angle, grazing, inplane_angle, outofplane_angle

    def process_positions(
        self,
        setup: "Setup",
        nb_frames: int,
        scan_number: int,
        frames_logical: Optional[np.ndarray] = None,
    ) -> List[Any]:
        """
        Load and crop/pad motor positions depending on the number of frames at BM02.

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
        mu, th, chi, phi, nu, tth, energy, _ = super().process_positions(
            setup=setup,
            nb_frames=nb_frames,
            scan_number=scan_number,
            frames_logical=frames_logical,
        )

        # eventually crop/pad motor values if the provided dataset was further
        # cropped/padded
        if setup.rocking_angle == "outofplane":  # th rocking curve
            nb_steps = len(th)
            tilt_angle = (th[1:] - th[0:-1]).mean()
            th = self.process_tilt(
                th, nb_steps=nb_steps, nb_frames=nb_frames, step_size=tilt_angle
            )
        elif setup.rocking_angle == "inplane":  # phi rocking curve
            nb_steps = len(phi)
            tilt_angle = (phi[1:] - phi[0:-1]).mean()
            phi = self.process_tilt(
                phi, nb_steps=nb_steps, nb_frames=nb_frames, step_size=tilt_angle
            )
        elif setup.rocking_angle == "energy":
            nb_steps = len(energy)
            tilt_angle = (energy[1:] - energy[0:-1]).mean()
            energy = self.process_tilt(
                energy, nb_steps=nb_steps, nb_frames=nb_frames, step_size=tilt_angle
            )
        else:
            raise ValueError('Wrong value for "rocking_angle" parameter')

        return util.bin_parameters(
            binning=setup.detector.binning[0],
            nb_frames=nb_frames,
            params=[mu, th, chi, phi, nu, tth, energy],
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
            self.logger.info("using BM02 geometry")

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
                    f"rocking angle is th, mu={grazing_angle[0] * 180 / np.pi:.3f} deg"
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
        else:
            raise NotImplementedError(f"rocking_angle={rocking_angle} not implemented")

        return mymatrix, q_offset


class BeamlineNANOMAX(BeamlineGoniometer):
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
                theta, nb_steps=nb_steps, nb_frames=nb_frames, step_size=tilt_angle
            )
        elif setup.rocking_angle == "inplane":  # phi rocking curve
            nb_steps = len(phi)
            tilt_angle = (phi[1:] - phi[0:-1]).mean()
            phi = self.process_tilt(
                phi, nb_steps=nb_steps, nb_frames=nb_frames, step_size=tilt_angle
            )
        elif setup.rocking_angle == "energy":
            nb_steps = len(energy)
            tilt_angle = (energy[1:] - energy[0:-1]).mean()
            energy = self.process_tilt(
                energy, nb_steps=nb_steps, nb_frames=nb_frames, step_size=tilt_angle
            )
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


class BeamlineP10(BeamlineGoniometer):
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
        self.logger.info(f"chi {chi}")
        self.logger.info(f"mu {mu}")
        if setup.rocking_angle == "outofplane":  # om rocking curve
            self.logger.info(f"phi {phi}")
            nb_steps = len(om)
            tilt_angle = (om[1:] - om[0:-1]).mean()
            om = self.process_tilt(
                om, nb_steps=nb_steps, nb_frames=nb_frames, step_size=tilt_angle
            )
        elif setup.rocking_angle == "inplane":  # phi rocking curve
            self.logger.info(f"om {om}")
            nb_steps = len(phi)
            tilt_angle = (phi[1:] - phi[0:-1]).mean()
            phi = self.process_tilt(
                phi, nb_steps=nb_steps, nb_frames=nb_frames, step_size=tilt_angle
            )
        elif setup.rocking_angle == "energy":
            nb_steps = len(energy)
            tilt_angle = (energy[1:] - energy[0:-1]).mean()
            energy = self.process_tilt(
                energy, nb_steps=nb_steps, nb_frames=nb_frames, step_size=tilt_angle
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
            # omega and chi potentially non zero (om below chi below phi)
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


class BeamlineSIXS(BeamlineGoniometer):
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
        self.logger.info(f"beta {beta}")
        if setup.rocking_angle == "inplane":  # mu rocking curve
            nb_steps = len(mu)
            tilt_angle = (mu[1:] - mu[0:-1]).mean()
            mu = self.process_tilt(
                mu, nb_steps=nb_steps, nb_frames=nb_frames, step_size=tilt_angle
            )
        elif setup.rocking_angle == "energy":
            nb_steps = len(energy)
            tilt_angle = (energy[1:] - energy[0:-1]).mean()
            energy = self.process_tilt(
                energy, nb_steps=nb_steps, nb_frames=nb_frames, step_size=tilt_angle
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


class Beamline34ID(BeamlineGoniometer):
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
                phi, nb_steps=nb_steps, nb_frames=nb_frames, step_size=tilt_angle
            )
        elif setup.rocking_angle == "inplane":  # theta rocking curve
            nb_steps = len(theta)
            tilt_angle = (theta[1:] - theta[0:-1]).mean()
            theta = self.process_tilt(
                theta, nb_steps=nb_steps, nb_frames=nb_frames, step_size=tilt_angle
            )
        elif setup.rocking_angle == "energy":
            nb_steps = len(energy)
            tilt_angle = (energy[1:] - energy[0:-1]).mean()
            energy = self.process_tilt(
                energy, nb_steps=nb_steps, nb_frames=nb_frames, step_size=tilt_angle
            )
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


class BeamlineP10SAXS(BeamlineSaxs):
    """
    Definition of PETRA III P10 beamline for the USAXS setup.

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
        Retrieve goniometer motor positions for a CDI tomographic scan.

        :param setup: the experimental setup: Class Setup
        :return: a tuple of angular values in degrees (rocking angular step, grazing
         incidence angles, inplane detector angle, outofplane detector angle). The
         grazing incidence angles are the positions of circles below the rocking circle.
        """
        # load the motor positions
        (
            phi,
            det_z,
            det_y,
            det_x,
            energy,
            detector_distance,
        ) = self.loader.motor_positions(setup=setup)

        # define the circles of interest for CDI
        # no circle yet below phi at P10
        if setup.rocking_angle == "inplane":  # phi rocking curve
            grazing = (0,)
            tilt_angle = phi
        else:
            raise ValueError('Wrong value for "rocking_angle" parameter')

        setup.check_setup_cdi(
            grazing_angle=grazing,
            detector_position=(det_z, det_y, det_x),
            tilt_angle=tilt_angle,
            detector_distance=detector_distance,
            energy=energy,
        )

        # P10 SAXS goniometer, 1S + 0D (sample: phi / detector: None)
        self.sample_angles = (phi,)
        return tilt_angle, grazing, 0, 0

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
        # TODO adapt this to the USAXS setup  # skipcq: PYL-W0511
        mu, om, chi, phi, gamma, delta, energy, _ = super().process_positions(
            setup=setup,
            nb_frames=nb_frames,
            scan_number=scan_number,
            frames_logical=frames_logical,
        )

        # eventually crop/pad motor values if the provided dataset was further
        # cropped/padded
        self.logger.info(f"chi {chi}")
        self.logger.info(f"mu {mu}")
        if setup.rocking_angle == "outofplane":  # om rocking curve
            self.logger.info(f"phi {phi}")
            nb_steps = len(om)
            tilt_angle = (om[1:] - om[0:-1]).mean()
            om = self.process_tilt(
                om, nb_steps=nb_steps, nb_frames=nb_frames, step_size=tilt_angle
            )
        elif setup.rocking_angle == "inplane":  # phi rocking curve
            self.logger.info(f"om {om}")
            nb_steps = len(phi)
            tilt_angle = (phi[1:] - phi[0:-1]).mean()
            phi = self.process_tilt(
                phi, nb_steps=nb_steps, nb_frames=nb_frames, step_size=tilt_angle
            )
        elif setup.rocking_angle == "energy":
            nb_steps = len(energy)
            tilt_angle = (energy[1:] - energy[0:-1]).mean()
            energy = self.process_tilt(
                energy, nb_steps=nb_steps, nb_frames=nb_frames, step_size=tilt_angle
            )
        else:
            raise ValueError('Wrong value for "rocking_angle" parameter')

        return util.bin_parameters(
            binning=setup.detector.binning[0],
            nb_frames=nb_frames,
            params=[mu, om, chi, phi, gamma, delta, energy],
        )


class BeamlineID27(BeamlineSaxs):
    """
    Definition of ID27 beamline for the high-energy BCDI setup.

    The detector is not on a goniometer, its plane is always perpendiculat to the direct
    beam.

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
        Retrieve goniometer motor positions for a scan.

        :param setup: the experimental setup: Class Setup
        :return: a tuple of angular values in degrees (rocking angular step, grazing
         incidence angles, inplane detector angle, outofplane detector angle). The
         grazing incidence angles are the positions of circles below the rocking circle.
        """
        # load the motor positions
        (
            nath,
            det_z,
            det_y,
            det_x,
            energy,
            detector_distance,
        ) = self.loader.motor_positions(setup=setup)

        # define the circles of interest
        # no circle yet below nath at ID27
        if setup.rocking_angle == "inplane":  # nath rocking curve
            grazing = (0,)
            tilt_angle = nath
        else:
            raise ValueError('Wrong value for "rocking_angle" parameter')

        setup.check_setup_cdi(
            grazing_angle=grazing,
            detector_position=(det_z, det_y, det_x),
            tilt_angle=tilt_angle,
            detector_distance=detector_distance,
            energy=energy,
        )

        # ID27 SAXS goniometer, 1S + 0D (sample: nath / detector: None)
        self.sample_angles = (nath,)
        return tilt_angle, grazing

    def process_positions(
        self,
        setup,
        nb_frames,
        scan_number,
        frames_logical=None,
    ):
        """
        Load and crop/pad motor positions depending on the number of frames at ID27.

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
        nath, energy, _ = super().process_positions(
            setup=setup,
            nb_frames=nb_frames,
            scan_number=scan_number,
            frames_logical=frames_logical,
        )

        # eventually crop/pad motor values if the provided dataset was further
        # cropped/padded
        if setup.rocking_angle == "inplane":  # nath rocking curve
            nb_steps = len(nath)
            tilt_angle = (nath[1:] - nath[0:-1]).mean()
            nath = self.process_tilt(
                nath, nb_steps=nb_steps, nb_frames=nb_frames, step_size=tilt_angle
            )
        elif setup.rocking_angle == "energy":
            nb_steps = len(energy)
            tilt_angle = (energy[1:] - energy[0:-1]).mean()
            energy = self.process_tilt(
                energy, nb_steps=nb_steps, nb_frames=nb_frames, step_size=tilt_angle
            )
        else:
            raise ValueError('Wrong value for "rocking_angle" parameter')

        return util.bin_parameters(
            binning=setup.detector.binning[0],
            nb_frames=nb_frames,
            params=[nath, energy],
        )
