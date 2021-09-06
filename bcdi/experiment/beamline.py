# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#   (c) 06/2021-present : DESY CFEL
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

"""
Beamline-related classes.

These classes are not meant to be instantiated directly but via a Setup instance.
The available beamlines are:

- BeamlineID01
- BeamlineSIXS
- Beamline34ID
- BeamlineP10
- BeamlineCRISTAL
- BeamlineNANOMAX

"""
from abc import ABC, abstractmethod
import numpy as np
import os
import h5py
from math import isclose
from silx.io.specfile import SpecFile
import xrayutilities as xu
from ..utils import utilities as util


def create_beamline(name, **kwargs):
    """
    Create the instance of the beamline.

    :param name: str, name of the beamline
    :param kwargs: optional beamline-dependent parameters
    :return: the corresponding beamline instance
    """
    if name == "ID01":
        return BeamlineID01(name=name, **kwargs)
    if name in {"SIXS_2018", "SIXS_2019"}:
        return BeamlineSIXS(name=name, **kwargs)
    if name == "34ID":
        return Beamline34ID(name=name, **kwargs)
    if name == "P10":
        return BeamlineP10(name=name, **kwargs)
    if name == "CRISTAL":
        return BeamlineCRISTAL(name=name, **kwargs)
    if name == "NANOMAX":
        return BeamlineNANOMAX(name=name, **kwargs)
    raise ValueError(f"Beamline {name} not supported")


class Beamline(ABC):
    """
    Base class for defining a beamline.

    :param name: name of the beamline
    :param kwargs: optional beamline-dependent parameters
    """

    orientation_lookup = {"x-": 1, "x+": -1, "y-": 1, "y+": -1}  # lookup table for the
    # detector orientation and rotation direction, where axes are expressed in the
    # laboratory frame (z downstream, y vertical up, x outboard).
    # Expected detector orientation:
    # "x-" detector horizontal axis inboard, as it should be in the CXI convention
    # "y-" detector vertical axis down, as it should be in the CXI convention

    def __init__(self, name, **kwargs):
        self._name = name

    @staticmethod
    @abstractmethod
    def create_logfile(**kwargs):
        """
        Create the logfile, which can be a log/spec file or the data itself.

        The nature of this file is beamline dependent.

        :param kwargs: beamline_specific parameters, which may include part of the
         totality of the following keys:

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
        Horizontal detector orientation expressed in the laboratory frame.

        This is beamline-dependent. The laboratory frame convention is
        (z downstream, y vertical, x outboard).

        :return: "x+" or "x-"
        """

    @property
    @abstractmethod
    def detector_ver(self):
        """
        Vertical detector orientation expressed in the laboratory frame.

        This is beamline-dependent. The laboratory frame convention is
        (z downstream, y vertical, x outboard).

        :return: "y+" or "y-"
        """

    def exit_wavevector(
        self, diffractometer, wavelength, inplane_angle, outofplane_angle
    ):
        """
        Calculate the exit wavevector kout.

        It uses the setup parameters. kout is expressed in 1/m in the
        laboratory frame (z downstream, y vertical, x outboard).

        :param diffractometer: an instance of the class Diffractometer
        :param wavelength: float, X-ray wavelength in meters.
        :param inplane_angle: float, horizontal detector angle, in degrees.
        :param outofplane_angle: float, vertical detector angle, in degrees.
        :return: kout vector as a numpy array of shape (3)
        """
        # look for the index of the inplane detector circle
        index = self.find_inplane(diffractometer=diffractometer)

        factor = self.orientation_lookup[diffractometer.detector_circles[index]]

        kout = (
            2
            * np.pi
            / wavelength
            * np.array(
                [
                    np.cos(np.pi * inplane_angle / 180)
                    * np.cos(np.pi * outofplane_angle / 180),  # z
                    np.sin(np.pi * outofplane_angle / 180),  # y
                    -1
                    * factor
                    * np.sin(np.pi * inplane_angle / 180)
                    * np.cos(np.pi * outofplane_angle / 180),  # x
                ]
            )
        )
        return kout

    @staticmethod
    def find_inplane(diffractometer):
        """
        Find the index of the detector inplane circle.

        It looks for the index of the detector inplane rotation in the detector_circles
        property of the diffractometer ("y+" or "y-") . The coordinate
        convention is the laboratory  frame (z downstream, y vertical up, x outboard).

        :param: diffractometer: an instance of the class Diffractometer
        :return: int, the index. None if not found.
        """
        index = None
        for idx, val in enumerate(diffractometer.detector_circles):
            if val.startswith("y"):
                index = idx
        return index

    @staticmethod
    def find_outofplane(diffractometer):
        """
        Find the index of the detector out-of-plane circle.

        It looks for the index of the detector out-of-plane rotation in the
        detector_circles property of the diffractometer (typically "x-") . The
        coordinate convention is the laboratory  frame (z downstream, y vertical up,
        x outboard). This is useful only for SIXS where there are two out-of-plane
        detector rotations due to the beta circle. We need the index of the most inner
        circle, not beta.

        :param: diffractometer: an instance of the class Diffractometer
        :return: int, the index. None if not found.
        """
        index = None
        for idx, val in enumerate(diffractometer.detector_circles):
            if val.startswith("x"):
                index = idx
        return index

    @staticmethod
    @abstractmethod
    def init_paths(**kwargs):
        """
        Initialize paths used for data processing and logging.

        :param kwargs: dictionnary of the setup parameters including the following keys:

         - 'sample_name': string in front of the scan number in the data folder
           name.
         - 'scan_number': int, the scan number
         - 'root_folder': folder of the experiment, where all scans are stored
         - 'save_dir': path of the directory where to save the analysis results,
           can be None
         - 'specfile_name': beamline-dependent string:

           - ID01: name of the spec file without '.spec'
           - SIXS_2018 and SIXS_2019: None or full path of the alias dictionnary (e.g.
             root_folder+'alias_dict_2019.txt')
           - empty string for all other beamlines

         - 'template_imagefile': beamline-dependent template for the data files:

           - ID01: 'data_mpx4_%05d.edf.gz' or 'align_eiger2M_%05d.edf.gz'
           - SIXS_2018: 'align.spec_ascan_mu_%05d.nxs'
           - SIXS_2019: 'spare_ascan_mu_%05d.nxs'
           - Cristal: 'S%d.nxs'
           - P10: '_master.h5'
           - NANOMAX: '%06d.h5'
           - 34ID: 'Sample%dC_ES_data_51_256_256.npz'

        :return: a tuple of strings:

         - homedir: the path of the scan folder
         - default_dirname: the name of the folder containing images / raw data
         - specfile: the name of the specfile if it exists
         - template_imagefile: the template for data/image file names

        """

    def init_qconversion(
        self, conversion_table, beam_direction, offset_inplane, diffractometer
    ):
        """
        Initialize the qconv object for xrayutilities depending on the setup parameters.

        The convention in xrayutilities is x downstream, z vertical up, y outboard.
        Note: the user-defined motor offsets are applied directly when reading motor
        positions, therefore do not need to be taken into account in xrayutilities apart
        from the detector inplane offset determined by the area detector calibration.

        :param conversion_table: dictionary where keys are axes in the laboratory frame
         (z downstream, y vertical up, x outboard) and values are the corresponding
         axes in the frame of xrayutilities (x downstream, y outboard, z vertical up).
         E.g. {"x+": "y+", "x-": "y-", "y+": "z+", "y-": "z-", "z+": "x+", "z-": "x-"}
        :param beam_direction: direction of the incident X-ray beam in the frame of
         xrayutilities.
        :param offset_inplane: inplane offset of the detector defined as the outer angle
         in xrayutilities area detector calibration.
        :param diffractometer: instance of the class Diffractometer
        :return: a tuple containing:

         - the qconv object for xrayutilities
         - a tuple of motor offsets used later for q calculation

        """
        # look for the index of the inplane detector circle
        index = self.find_inplane(diffractometer=diffractometer)

        # convert axes from the laboratory frame to the frame of xrayutilies
        sample_circles = [
            conversion_table[val] for val in diffractometer.sample_circles
        ]
        detector_circles = [
            conversion_table[val] for val in diffractometer.detector_circles
        ]
        qconv = xu.experiment.QConversion(
            sample_circles, detector_circles, r_i=beam_direction
        )

        # create the tuple of offsets, all 0 except for the detector inplane circle
        if index is None:
            print("no detector inplane circle detected, discarding 'offset_inplane'")
            offsets = [0 for _ in range(len(sample_circles) + len(detector_circles))]
        else:
            offsets = [0 for _ in range(len(sample_circles) + index)]
            offsets.append(offset_inplane)
            for _ in range(len(detector_circles) - index - 1):
                offsets.append(0)

        return qconv, offsets

    def inplane_coeff(self, diffractometer):
        """
        Coefficient related to the detector inplane orientation.

        Define a coefficient +/- 1 depending on the detector inplane rotation direction
        (1 for clockwise, -1 for anti-clockwise) and the detector inplane orientation
        (1 for inboard, -1 for outboard).

        See scripts/postprocessing/correct_angles_detector.py for a use case.

        :param diffractometer: Diffractometer instance of the beamline.
        :return: +1 or -1
        """
        # look for the index of the inplane detector circle
        index = self.find_inplane(diffractometer=diffractometer)
        return (
            self.orientation_lookup[diffractometer.detector_circles[index]]
            * self.orientation_lookup[self.detector_hor]
        )

    @property
    def name(self):
        """Name of the beamline."""
        return self._name

    def outofplane_coeff(self, diffractometer):
        """
        Coefficient related to the detector vertical orientation.

        Define a coefficient +/- 1 depending on the detector out of plane rotation
        direction (1 for clockwise, -1 for anti-clockwise) and the detector out of
        plane orientation (1 for downward, -1 for upward).

        See scripts/postprocessing/correct_angles_detector.py for a use case.

        :param diffractometer: Diffractometer instance of the beamline.
        :return: +1 or -1
        """
        # look for the index of the out-of-plane detector circle
        index = self.find_outofplane(diffractometer=diffractometer)
        return (
            self.orientation_lookup[diffractometer.detector_circles[index]]
            * self.orientation_lookup[self.detector_ver]
        )

    @staticmethod
    @abstractmethod
    def process_positions(
        setup, logfile, nb_frames, scan_number, frames_logical=None, follow_bragg=False
    ):
        """
        Load and crop/pad motor positions depending on the current number of frames.

        The current number of frames may be different from the original number of frames
        if the data was cropped/padded, and motor values must be processed accordingly.

        :param setup: an instance of the class Setup
        :param logfile: the logfile created in Setup.create_logfile()
        :param nb_frames: the number of frames in the current dataset
        :param scan_number: the scan number to load
        :param frames_logical: array of length the number of measured frames.
         In case of cropping/padding the number of frames changes. A frame whose
         index is set to 1 means that it is used, 0 means not used, -1 means padded
         (added) frame
        :param follow_bragg: True when in energy scans the detector was also scanned
         to follow the Bragg peak

        :return: a tuple of 1D arrays (sample circles, detector circles, energy)
        """

    @abstractmethod
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

        :param wavelength: X-ray wasvelength in nm
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


class BeamlineCRISTAL(Beamline):
    """
    Definition of SOLEIL CRISTAL beamline.

    :param name: name of the beamline
    """

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

    @staticmethod
    def create_logfile(datadir, template_imagefile, scan_number, **kwargs):
        """
        Create the logfile, which is the data itself for CRISTAL.

        :param datadir: str, the data directory
        :param template_imagefile: str, template for data file name, e.g. 'S%d.nxs'
        :param scan_number: int, the scan number to load
        :return: logfile
        """
        if not all(isinstance(val, str) for val in {datadir, template_imagefile}):
            raise TypeError("datadir and template_imagefile should be strings")
        if not isinstance(scan_number, int):
            raise TypeError(
                "scan_number should be an integer, " f"got {type(scan_number)}"
            )
        # no specfile, load directly the dataset
        ccdfiletmp = os.path.join(datadir + template_imagefile % scan_number)
        return h5py.File(ccdfiletmp, "r")

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

    @staticmethod
    def init_paths(root_folder, sample_name, scan_number, template_imagefile, **kwargs):
        """
        Initialize paths used for data processing and logging at CRISTAL.

        :param root_folder: folder of the experiment, where all scans are stored
        :param sample_name: string in front of the scan number in the data folder
         name.
        :param scan_number: int, the scan number
        :param template_imagefile: template for the data files, e.g. 'S%d.nxs'
        :return: a tuple of strings:

         - homedir: the path of the scan folder
         - default_dirname: the name of the folder containing images / raw data
         - specfile: the name of the specfile if it exists
         - template_imagefile: the template for data/image file names

        """
        homedir = root_folder + sample_name + str(scan_number) + "/"
        default_dirname = "data/"
        return homedir, default_dirname, "", template_imagefile

    @staticmethod
    def process_positions(
        setup,
        logfile,
        nb_frames,
        scan_number,
        frames_logical=None,
        follow_bragg=False,
    ):
        """
        Load and crop/pad motor positions depending on the number of frames at CRISTAL.

        The current number of frames may be different from the original number of frames
        if the data was cropped/padded, and motor values must be processed accordingly.

        :param setup: an instance of the class Setup
        :param logfile: the logfile created in Setup.create_logfile()
        :param nb_frames: the number of frames in the current dataset
        :param scan_number: the scan number to load
        :param frames_logical: array of length the number of measured frames.
         In case of cropping/padding the number of frames changes. A frame whose
         index is set to 1 means that it is used, 0 means not used, -1 means padded
         (added) frame
        :param follow_bragg: True when in energy scans the detector was also scanned
         to follow the Bragg peak
        :return: a tuple of 1D arrays (sample circles, detector circles, energy)
        """
        mgomega, mgphi, gamma, delta, energy = setup.diffractometer.motor_positions(
            setup=setup,
            logfile=logfile,
            scan_number=scan_number,
            follow_bragg=follow_bragg,
        )
        # first, remove the motor positions corresponding to deleted frames during data
        # loading (frames_logical = 0)
        mgomega, mgphi, gamma, delta, energy = util.apply_logical_array(
            arrays=(mgomega, mgphi, gamma, delta, energy),
            frames_logical=frames_logical,
        )

        # then, eventually crop/pad motor values if the provided dataset was further
        # cropped/padded
        if setup.rocking_angle == "outofplane":  # mgomega rocking curve
            nb_steps = len(mgomega)
            tilt_angle = (mgomega[1:] - mgomega[0:-1]).mean()

            if nb_steps < nb_frames:
                # data has been padded, we suppose it is centered in z dimension
                pad_low = int((nb_frames - nb_steps + ((nb_frames - nb_steps) % 2)) / 2)
                pad_high = int(
                    (nb_frames - nb_steps + 1) / 2 - ((nb_frames - nb_steps) % 2)
                )
                mgomega = np.concatenate(
                    (
                        mgomega[0] + np.arange(-pad_low, 0, 1) * tilt_angle,
                        mgomega,
                        mgomega[-1] + np.arange(1, pad_high + 1, 1) * tilt_angle,
                    ),
                    axis=0,
                )
            if (
                nb_steps > nb_frames
            ):  # data has been cropped, we suppose it is centered in z dimension
                mgomega = mgomega[
                    (nb_steps - nb_frames) // 2 : (nb_steps + nb_frames) // 2
                ]

        elif setup.rocking_angle == "inplane":  # mgphi rocking curve
            print("mgomega", mgomega)
            nb_steps = len(mgphi)
            tilt_angle = (mgphi[1:] - mgphi[0:-1]).mean()

            if nb_steps < nb_frames:
                # data has been padded, we suppose it is centered in z dimension
                pad_low = int((nb_frames - nb_steps + ((nb_frames - nb_steps) % 2)) / 2)
                pad_high = int(
                    (nb_frames - nb_steps + 1) / 2 - ((nb_frames - nb_steps) % 2)
                )
                mgphi = np.concatenate(
                    (
                        mgphi[0] + np.arange(-pad_low, 0, 1) * tilt_angle,
                        mgphi,
                        mgphi[-1] + np.arange(1, pad_high + 1, 1) * tilt_angle,
                    ),
                    axis=0,
                )
            if nb_steps > nb_frames:
                # data has been cropped, we suppose it is centered in z dimension
                mgphi = mgphi[(nb_steps - nb_frames) // 2 : (nb_steps + nb_frames) // 2]

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

        :param wavelength: X-ray wasvelength in nm
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
            print("using CRISTAL geometry")

        if rocking_angle == "outofplane":
            if grazing_angle is not None:
                raise NotImplementedError(
                    "Circle below mgomega not implemented for CRISTAL"
                )
            if verbose:
                print("rocking angle is mgomega")
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
            if verbose:
                print(
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

        return mymatrix, q_offset


class BeamlineID01(Beamline):
    """
    Definition of ESRF ID01 beamline.

    :param name: name of the beamline
    """

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

    @staticmethod
    def create_logfile(root_folder, filename, **kwargs):
        """
        Create the logfile, which is the spec file for ID01.

        :param root_folder: str, the root directory of the experiment, where is e.g. the
         specfile file.
        :param filename: str, name of the spec file without '.spec'
        :return: logfile
        """
        if not all(isinstance(val, str) for val in {root_folder, filename}):
            raise ValueError("root_folder and filename should be strings")
        # load the spec file
        return SpecFile(root_folder + filename + ".spec")

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

    @staticmethod
    def init_paths(
        root_folder,
        sample_name,
        scan_number,
        specfile_name,
        template_imagefile,
        **kwargs,
    ):
        """
        Initialize paths used for data processing and logging at ID01.

        :param root_folder: folder of the experiment, where all scans are stored
        :param sample_name: string in front of the scan number in the data folder
         name.
        :param scan_number: int, the scan number
        :param specfile_name: name of the spec file without '.spec'
        :param template_imagefile: template for the data files, e.g.
         'data_mpx4_%05d.edf.gz' or 'align_eiger2M_%05d.edf.gz'
        :return: a tuple of strings:

         - homedir: the path of the scan folder
         - default_dirname: the name of the folder containing images / raw data
         - specfile: the name of the specfile if it exists
         - template_imagefile: the template for data/image file names

        """
        homedir = root_folder + sample_name + str(scan_number) + "/"
        default_dirname = "data/"
        return homedir, default_dirname, specfile_name, template_imagefile

    @staticmethod
    def process_positions(
        setup,
        logfile,
        nb_frames,
        scan_number,
        frames_logical=None,
        follow_bragg=False,
    ):
        """
        Load and crop/pad motor positions depending on the number of frames at ID01.

        The current number of frames may be different from the original number of frames
        if the data was cropped/padded, and motor values must be processed accordingly.

        :param setup: an instance of the class Setup
        :param logfile: the logfile created in Setup.create_logfile()
        :param nb_frames: the number of frames in the current dataset
        :param scan_number: the scan number to load
        :param frames_logical: array of length the number of measured frames.
         In case of cropping/padding the number of frames changes. A frame whose
         index is set to 1 means that it is used, 0 means not used, -1 means padded
         (added) frame
        :param follow_bragg: True when in energy scans the detector was also scanned
         to follow the Bragg peak
        :return: a tuple of 1D arrays (sample circles, detector circles, energy)
        """
        mu, eta, phi, nu, delta, energy = setup.diffractometer.motor_positions(
            setup=setup,
            logfile=logfile,
            scan_number=scan_number,
            follow_bragg=follow_bragg,
        )
        # first, remove the motor positions corresponding to deleted frames during data
        # loading (frames_logical = 0)
        mu, eta, phi, nu, delta, energy = util.apply_logical_array(
            arrays=(mu, eta, phi, nu, delta, energy),
            frames_logical=frames_logical,
        )

        # then, eventually crop/pad motor values if the provided dataset was further
        # cropped/padded
        if setup.rocking_angle == "outofplane":  # eta rocking curve
            print("phi", phi)
            nb_steps = len(eta)
            tilt_angle = (eta[1:] - eta[0:-1]).mean()

            if nb_steps < nb_frames:
                # data has been padded, we suppose it is centered in z dimension
                pad_low = int((nb_frames - nb_steps + ((nb_frames - nb_steps) % 2)) / 2)
                pad_high = int(
                    (nb_frames - nb_steps + 1) / 2 - ((nb_frames - nb_steps) % 2)
                )
                eta = np.concatenate(
                    (
                        eta[0] + np.arange(-pad_low, 0, 1) * tilt_angle,
                        eta,
                        eta[-1] + np.arange(1, pad_high + 1, 1) * tilt_angle,
                    ),
                    axis=0,
                )
            if (
                nb_steps > nb_frames
            ):  # data has been cropped, we suppose it is centered in z dimension
                eta = eta[(nb_steps - nb_frames) // 2 : (nb_steps + nb_frames) // 2]

        elif setup.rocking_angle == "inplane":  # phi rocking curve
            print("eta", eta)
            nb_steps = len(phi)
            tilt_angle = (phi[1:] - phi[0:-1]).mean()

            if (
                nb_steps < nb_frames
            ):  # data has been padded, we suppose it is centered in z dimension
                pad_low = int((nb_frames - nb_steps + ((nb_frames - nb_steps) % 2)) / 2)
                pad_high = int(
                    (nb_frames - nb_steps + 1) / 2 - ((nb_frames - nb_steps) % 2)
                )
                phi = np.concatenate(
                    (
                        phi[0] + np.arange(-pad_low, 0, 1) * tilt_angle,
                        phi,
                        phi[-1] + np.arange(1, pad_high + 1, 1) * tilt_angle,
                    ),
                    axis=0,
                )
            if nb_steps > nb_frames:
                # data has been cropped, we suppose it is centered in z dimension
                phi = phi[(nb_steps - nb_frames) // 2 : (nb_steps + nb_frames) // 2]

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

        :param wavelength: X-ray wasvelength in nm
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
            print("using ESRF ID01 PSIC geometry")

        if not isclose(grazing_angle[0], 0, rel_tol=1e-09, abs_tol=1e-09):
            raise NotImplementedError(
                "Non-zero mu not implemented " "for the transformation matrices at ID01"
            )

        if rocking_angle == "outofplane":
            if verbose:
                print(
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
            if verbose:
                print(
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

        return mymatrix, q_offset


class BeamlineNANOMAX(Beamline):
    """
    Definition of MAX IV NANOMAX beamline.

    :param name: name of the beamline
    """

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

    @staticmethod
    def create_logfile(datadir, template_imagefile, scan_number, **kwargs):
        """
        Create the logfile, which is the data itself for Nanomax.

        :param datadir: str, the data directory
        :param template_imagefile: str, template for data file name, e.g. '%06d.h5'
        :param scan_number: int, the scan number to load
        :return: logfile
        """
        if not all(isinstance(val, str) for val in {datadir, template_imagefile}):
            raise TypeError("datadir and template_imagefile should be strings")
        if not isinstance(scan_number, int):
            raise TypeError(
                "scan_number should be an integer, " f"got {type(scan_number)}"
            )
        ccdfiletmp = os.path.join(datadir + template_imagefile % scan_number)
        return h5py.File(ccdfiletmp, "r")

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

    @staticmethod
    def init_paths(root_folder, sample_name, scan_number, template_imagefile, **kwargs):
        """
        Initialize paths used for data processing and logging at Nanomax.

        :param root_folder: folder of the experiment, where all scans are stored
        :param sample_name: string in front of the scan number in the data folder
         name.
        :param scan_number: int, the scan number
        :param template_imagefile: template for the data files, e.g. '%06d.h5'
        :return: a tuple of strings:

         - homedir: the path of the scan folder
         - default_dirname: the name of the folder containing images / raw data
         - specfile: the name of the specfile if it exists
         - template_imagefile: the template for data/image file names

        """
        homedir = root_folder + sample_name + "{:06d}".format(scan_number) + "/"
        default_dirname = "data/"
        return homedir, default_dirname, "", template_imagefile

    @staticmethod
    def process_positions(
        setup,
        logfile,
        nb_frames,
        scan_number,
        frames_logical=None,
        follow_bragg=False,
    ):
        """
        Load and crop/pad motor positions depending on the number of frames at NANOMAX.

        The current number of frames may be different from the original number of frames
        if the data was cropped/padded, and motor values must be processed accordingly.

        :param setup: an instance of the class Setup
        :param logfile: the logfile created in Setup.create_logfile()
        :param nb_frames: the number of frames in the current dataset
        :param scan_number: the scan number to load
        :param frames_logical: array of length the number of measured frames.
         In case of cropping/padding the number of frames changes. A frame whose
         index is set to 1 means that it is used, 0 means not used, -1 means padded
         (added) frame
        :param follow_bragg: True when in energy scans the detector was also scanned
         to follow the Bragg peak
        :return: a tuple of 1D arrays (sample circles, detector circles, energy)
        """
        theta, phi, gamma, delta, energy = setup.diffractometer.motor_positions(
            setup=setup,
            logfile=logfile,
            scan_number=scan_number,
            follow_bragg=follow_bragg,
        )
        # first, remove the motor positions corresponding to deleted frames during data
        # loading (frames_logical = 0)
        theta, phi, gamma, delta, energy = util.apply_logical_array(
            arrays=(theta, phi, gamma, delta, energy),
            frames_logical=frames_logical,
        )

        # then, eventually crop/pad motor values if the provided dataset was further
        # cropped/padded
        if setup.rocking_angle == "outofplane":  # theta rocking curve
            nb_steps = len(theta)
            tilt_angle = (theta[1:] - theta[0:-1]).mean()

            if nb_steps < nb_frames:
                # data has been padded, we suppose it is centered in z dimension
                pad_low = int((nb_frames - nb_steps + ((nb_frames - nb_steps) % 2)) / 2)
                pad_high = int(
                    (nb_frames - nb_steps + 1) / 2 - ((nb_frames - nb_steps) % 2)
                )
                theta = np.concatenate(
                    (
                        theta[0] + np.arange(-pad_low, 0, 1) * tilt_angle,
                        theta,
                        theta[-1] + np.arange(1, pad_high + 1, 1) * tilt_angle,
                    ),
                    axis=0,
                )
            if (
                nb_steps > nb_frames
            ):  # data has been cropped, we suppose it is centered in z dimension
                theta = theta[(nb_steps - nb_frames) // 2 : (nb_steps + nb_frames) // 2]

        elif setup.rocking_angle == "inplane":  # phi rocking curve
            nb_steps = len(phi)
            tilt_angle = (phi[1:] - phi[0:-1]).mean()

            if nb_steps < nb_frames:
                # data has been padded, we suppose it is centered in z dimension
                pad_low = int((nb_frames - nb_steps + ((nb_frames - nb_steps) % 2)) / 2)
                pad_high = int(
                    (nb_frames - nb_steps + 1) / 2 - ((nb_frames - nb_steps) % 2)
                )
                phi = np.concatenate(
                    (
                        phi[0] + np.arange(-pad_low, 0, 1) * tilt_angle,
                        phi,
                        phi[-1] + np.arange(1, pad_high + 1, 1) * tilt_angle,
                    ),
                    axis=0,
                )
            if nb_steps > nb_frames:
                # data has been cropped, we suppose it is centered in z dimension
                phi = phi[(nb_steps - nb_frames) // 2 : (nb_steps + nb_frames) // 2]

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

        :param wavelength: X-ray wasvelength in nm
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
            print("using NANOMAX geometry")

        if rocking_angle == "outofplane":
            if grazing_angle is not None:
                raise NotImplementedError(
                    "Circle below theta not implemented for NANOMAX"
                )
            if verbose:
                print("rocking angle is theta")
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
            if verbose:
                print(
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

        return mymatrix, q_offset


class BeamlineP10(Beamline):
    """
    Definition of PETRA III P10 beamline.

    :param name: name of the beamline
    """

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

    @staticmethod
    def create_logfile(root_folder, filename, **kwargs):
        """
        Create the logfile, which is the .fio file for P10.

        :param root_folder: str, the root directory of the experiment, where the scan
         folders are located.
        :param filename: str, name of the .fio file (without ".fio")
        :return: logfile
        """
        if not all(isinstance(val, str) for val in {root_folder, filename}):
            raise TypeError("root_folder and filename should be strings")
        # load .fio file
        return root_folder + filename + "/" + filename + ".fio"

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

    @staticmethod
    def init_paths(root_folder, sample_name, scan_number, template_imagefile, **kwargs):
        """
        Initialize paths used for data processing and logging at P10.

        :param root_folder: folder of the experiment, where all scans are stored
        :param sample_name: string in front of the scan number in the data folder
         name.
        :param scan_number: int, the scan number
        :param template_imagefile: template for the data files, e.g. '_master.h5'
        :return: a tuple of strings:

         - homedir: the path of the scan folder
         - default_dirname: the name of the folder containing images / raw data
         - specfile: the name of the specfile if it exists
         - template_imagefile: the template for data/image file names

        """
        specfile = sample_name + "_{:05d}".format(scan_number)
        homedir = root_folder + specfile + "/"
        default_dirname = "e4m/"
        template_imagefile = specfile + template_imagefile
        return homedir, default_dirname, specfile, template_imagefile

    @staticmethod
    def process_positions(
        setup,
        logfile,
        nb_frames,
        scan_number,
        frames_logical=None,
        follow_bragg=False,
    ):
        """
        Load and crop/pad motor positions depending on the number of frames at P10.

        The current number of frames may be different from the original number of frames
        if the data was cropped/padded, and motor values must be processed accordingly.

        :param setup: an instance of the class Setup
        :param logfile: the logfile created in Setup.create_logfile()
        :param nb_frames: the number of frames in the current dataset
        :param scan_number: the scan number to load
        :param frames_logical: array of length the number of measured frames.
         In case of cropping/padding the number of frames changes. A frame whose
         index is set to 1 means that it is used, 0 means not used, -1 means padded
         (added) frame
        :param follow_bragg: True when in energy scans the detector was also scanned
         to follow the Bragg peak
        :return: a tuple of 1D arrays (sample circles, detector circles, energy)
        """
        mu, om, chi, phi, gamma, delta, energy = setup.diffractometer.motor_positions(
            setup=setup,
            logfile=logfile,
            scan_number=scan_number,
            follow_bragg=follow_bragg,
        )
        # first, remove the motor positions corresponding to deleted frames during data
        # loading (frames_logical = 0)
        mu, om, chi, phi, gamma, delta, energy = util.apply_logical_array(
            arrays=(mu, om, chi, phi, gamma, delta, energy),
            frames_logical=frames_logical,
        )

        # then, eventually crop/pad motor values if the provided dataset was further
        # cropped/padded
        print("chi", chi)
        print("mu", mu)
        if setup.rocking_angle == "outofplane":  # om rocking curve
            print("phi", phi)
            nb_steps = len(om)
            tilt_angle = (om[1:] - om[0:-1]).mean()

            if nb_steps < nb_frames:
                # data has been padded, we suppose it is centered in z dimension
                pad_low = int((nb_frames - nb_steps + ((nb_frames - nb_steps) % 2)) / 2)
                pad_high = int(
                    (nb_frames - nb_steps + 1) / 2 - ((nb_frames - nb_steps) % 2)
                )
                om = np.concatenate(
                    (
                        om[0] + np.arange(-pad_low, 0, 1) * tilt_angle,
                        om,
                        om[-1] + np.arange(1, pad_high + 1, 1) * tilt_angle,
                    ),
                    axis=0,
                )
            if nb_steps > nb_frames:
                # data has been cropped, we suppose it is centered in z dimension
                om = om[(nb_steps - nb_frames) // 2 : (nb_steps + nb_frames) // 2]

        elif setup.rocking_angle == "inplane":  # phi rocking curve
            print("om", om)
            nb_steps = len(phi)
            tilt_angle = (phi[1:] - phi[0:-1]).mean()

            if nb_steps < nb_frames:
                # data has been padded, we suppose it is centered in z dimension
                pad_low = int((nb_frames - nb_steps + ((nb_frames - nb_steps) % 2)) / 2)
                pad_high = int(
                    (nb_frames - nb_steps + 1) / 2 - ((nb_frames - nb_steps) % 2)
                )
                phi = np.concatenate(
                    (
                        phi[0] + np.arange(-pad_low, 0, 1) * tilt_angle,
                        phi,
                        phi[-1] + np.arange(1, pad_high + 1, 1) * tilt_angle,
                    ),
                    axis=0,
                )
            if nb_steps > nb_frames:
                # data has been cropped, we suppose it is centered in z dimension
                phi = phi[(nb_steps - nb_frames) // 2 : (nb_steps + nb_frames) // 2]

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

        :param wavelength: X-ray wasvelength in nm
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
            print("using PETRAIII P10 geometry")

        if rocking_angle == "outofplane":
            if verbose:
                print(
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
            if not isclose(grazing_angle[0], 0, rel_tol=1e-09, abs_tol=1e-09):
                raise NotImplementedError(
                    "Non-zero mu not implemented for inplane rocking curve at P10"
                )
            if verbose:
                print(
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

        return mymatrix, q_offset


class BeamlineSIXS(Beamline):
    """
    Definition of SOLEIL SIXS beamline.

    :param name: name of the beamline
    """

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

    def create_logfile(
        self, datadir, template_imagefile, scan_number, filename, **kwargs
    ):
        """
        Create the logfile, which is the data itself for SIXS.

        :param datadir: str, the data directory
        :param template_imagefile: str, template for data file name:

           - SIXS_2018: 'align.spec_ascan_mu_%05d.nxs'
           - SIXS_2019: 'spare_ascan_mu_%05d.nxs'

        :param scan_number: int, the scan number to load
        :param filename: str, absolute path of 'alias_dict.txt'
        :return: logfile
        """
        if not all(
            isinstance(val, str) for val in {datadir, template_imagefile, filename}
        ):
            raise TypeError("datadir and template_imagefile should be strings")
        if not isinstance(scan_number, int):
            raise TypeError(
                "scan_number should be an integer, " f"got {type(scan_number)}"
            )

        shortname = template_imagefile % scan_number
        if self.name == "SIXS_2018":
            # no specfile, load directly the dataset
            import bcdi.preprocessing.nxsReady as nxsReady

            return nxsReady.DataSet(
                longname=datadir + shortname,
                shortname=shortname,
                alias_dict=filename,
                scan="SBS",
            )
        if self.name == "SIXS_2019":
            # no specfile, load directly the dataset
            import bcdi.preprocessing.ReadNxs3 as ReadNxs3

            return ReadNxs3.DataSet(
                directory=datadir,
                filename=shortname,
                alias_dict=filename,
            )
        raise NotImplementedError(f"{self.name} is not implemented")

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

    @staticmethod
    def init_paths(
        root_folder,
        sample_name,
        scan_number,
        specfile_name,
        template_imagefile,
        **kwargs,
    ):
        """
        Initialize paths used for data processing and logging at SIXS.

        :param root_folder: folder of the experiment, where all scans are stored
        :param sample_name: string in front of the scan number in the data folder
         name.
        :param scan_number: int, the scan number
        :param specfile_name: None or full path of the alias dictionnary (e.g.
         root_folder+'alias_dict_2019.txt')
        :param template_imagefile: template for the data files, e.g.
         'align.spec_ascan_mu_%05d.nxs' (SIXS_2018), 'spare_ascan_mu_%05d.nxs'
         (SIXS_2019).
        :return: a tuple of strings:

         - homedir: the path of the scan folder
         - default_dirname: the name of the folder containing images / raw data
         - specfile: the name of the specfile if it exists
         - template_imagefile: the template for data/image file names

        """
        homedir = root_folder + sample_name + str(scan_number) + "/"
        default_dirname = "data/"

        if specfile_name is None:
            # default to the alias dictionnary located within the package
            specfile = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    os.pardir,
                    "preprocessing/alias_dict_2021.txt",
                )
            )
        else:
            specfile = specfile_name

        return homedir, default_dirname, specfile, template_imagefile

    @staticmethod
    def process_positions(
        setup,
        logfile,
        nb_frames,
        scan_number,
        frames_logical=None,
        follow_bragg=False,
    ):
        """
        Load and crop/pad motor positions depending on the number of frames at SIXS.

        The current number of frames may be different from the original number of frames
        if the data was cropped/padded, and motor values must be processed accordingly.

        :param setup: an instance of the class Setup
        :param logfile: the logfile created in Setup.create_logfile()
        :param nb_frames: the number of frames in the current dataset
        :param scan_number: the scan number to load
        :param frames_logical: array of length the number of measured frames.
         In case of cropping/padding the number of frames changes. A frame whose
         index is set to 1 means that it is used, 0 means not used, -1 means padded
         (added) frame
        :param follow_bragg: True when in energy scans the detector was also scanned
         to follow the Bragg peak
        :return: a tuple of 1D arrays (sample circles, detector circles, energy)
        """
        beta, mu, gamma, delta, energy = setup.diffractometer.motor_positions(
            setup=setup,
            logfile=logfile,
            scan_number=scan_number,
            follow_bragg=follow_bragg,
        )
        # first, remove the motor positions corresponding to deleted frames during data
        # loading (frames_logical = 0)
        beta, mu, gamma, delta, energy = util.apply_logical_array(
            arrays=(beta, mu, gamma, delta, energy),
            frames_logical=frames_logical,
        )

        # then, eventually crop/pad motor values if the provided dataset was further
        # cropped/padded
        print("beta", beta)
        if setup.rocking_angle == "inplane":  # mu rocking curve
            nb_steps = len(mu)
            tilt_angle = (mu[1:] - mu[0:-1]).mean()

            if nb_steps < nb_frames:
                # data has been padded, we suppose it is centered in z dimension
                pad_low = int((nb_frames - nb_steps + ((nb_frames - nb_steps) % 2)) / 2)
                pad_high = int(
                    (nb_frames - nb_steps + 1) / 2 - ((nb_frames - nb_steps) % 2)
                )
                mu = np.concatenate(
                    (
                        mu[0] + np.arange(-pad_low, 0, 1) * tilt_angle,
                        mu,
                        mu[-1] + np.arange(1, pad_high + 1, 1) * tilt_angle,
                    ),
                    axis=0,
                )
            if nb_steps > nb_frames:
                # data has been cropped, we suppose it is centered in z dimension
                mu = mu[(nb_steps - nb_frames) // 2 : (nb_steps + nb_frames) // 2]

        else:
            raise ValueError("Out-of-plane rocking curve not implemented for SIXS")

        return util.bin_parameters(
            binning=setup.detector.binning[0],
            nb_frames=nb_frames,
            params=[beta, mu, gamma, delta, energy],
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

        :param wavelength: X-ray wasvelength in nm
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
            print("using SIXS geometry")

        if rocking_angle == "inplane":
            if verbose:
                print(
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
            raise NotImplementedError(
                "out of plane rocking curve not implemented for SIXS"
            )

        return mymatrix, q_offset


class Beamline34ID(Beamline):
    """
    Definition of APS 34ID-C beamline.

    :param name: name of the beamline
    """

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

    @staticmethod
    def create_logfile(**kwargs):
        """Create logfile for 34ID-C."""
        raise NotImplementedError("create_logfile method not implemented for 34ID")

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

    @staticmethod
    def init_paths(root_folder, sample_name, scan_number, template_imagefile, **kwargs):
        """
        Initialize paths used for data processing and logging at 34ID-C.

        :param root_folder: folder of the experiment, where all scans are stored
        :param sample_name: string in front of the scan number in the data folder
         name.
        :param scan_number: int, the scan number
        :param template_imagefile: template for the data files, e.g.
         'Sample%dC_ES_data_51_256_256.npz'.
        :return: a tuple of strings:

         - homedir: the path of the scan folder
         - default_dirname: the name of the folder containing images / raw data
         - specfile: the name of the specfile if it exists
         - template_imagefile: the template for data/image file names

        """
        homedir = root_folder + sample_name + str(scan_number) + "/"
        default_dirname = "data/"
        return homedir, default_dirname, "", template_imagefile

    @staticmethod
    def process_positions(
        setup,
        logfile,
        nb_frames,
        scan_number,
        frames_logical=None,
        follow_bragg=False,
    ):
        """
        Load and crop/pad motor positions depending on the number of frames at 34ID-C.

        The current number of frames may be different from the original number of frames
        if the data was cropped/padded, and motor values must be processed accordingly.

        :param setup: an instance of the class Setup
        :param logfile: the logfile created in Setup.create_logfile()
        :param nb_frames: the number of frames in the current dataset
        :param scan_number: the scan number to load
        :param frames_logical: array of length the number of measured frames.
         In case of cropping/padding the number of frames changes. A frame whose
         index is set to 1 means that it is used, 0 means not used, -1 means padded
         (added) frame
        :param follow_bragg: True when in energy scans the detector was also scanned
         to follow the Bragg peak
        :return: a tuple of 1D arrays (sample circles, detector circles, energy)
        """
        theta, phi, delta, gamma, energy = setup.diffractometer.motor_positions(
            setup=setup,
            logfile=logfile,
            scan_number=scan_number,
            follow_bragg=follow_bragg,
        )
        # first, remove the motor positions corresponding to deleted frames during data
        # loading (frames_logical = 0)
        theta, phi, delta, gamma, energy = util.apply_logical_array(
            arrays=(theta, phi, delta, gamma, energy),
            frames_logical=frames_logical,
        )

        # then, eventually crop/pad motor values if the provided dataset was further
        # cropped/padded
        if setup.rocking_angle == "outofplane":  # phi rocking curve
            nb_steps = len(phi)
            tilt_angle = (phi[1:] - phi[0:-1]).mean()

            if nb_steps < nb_frames:
                # data has been padded, we suppose it is centered in z dimension
                pad_low = int((nb_frames - nb_steps + ((nb_frames - nb_steps) % 2)) / 2)
                pad_high = int(
                    (nb_frames - nb_steps + 1) / 2 - ((nb_frames - nb_steps) % 2)
                )
                phi = np.concatenate(
                    (
                        phi[0] + np.arange(-pad_low, 0, 1) * tilt_angle,
                        phi,
                        phi[-1] + np.arange(1, pad_high + 1, 1) * tilt_angle,
                    ),
                    axis=0,
                )
            if nb_steps > nb_frames:
                # data has been cropped, we suppose it is centered in z dimension
                phi = phi[(nb_steps - nb_frames) // 2 : (nb_steps + nb_frames) // 2]

        elif setup.rocking_angle == "inplane":  # theta rocking curve
            nb_steps = len(theta)
            tilt_angle = (theta[1:] - theta[0:-1]).mean()

            if nb_steps < nb_frames:
                # data has been padded, we suppose it is centered in z dimension
                pad_low = int((nb_frames - nb_steps + ((nb_frames - nb_steps) % 2)) / 2)
                pad_high = int(
                    (nb_frames - nb_steps + 1) / 2 - ((nb_frames - nb_steps) % 2)
                )
                theta = np.concatenate(
                    (
                        theta[0] + np.arange(-pad_low, 0, 1) * tilt_angle,
                        theta,
                        theta[-1] + np.arange(1, pad_high + 1, 1) * tilt_angle,
                    ),
                    axis=0,
                )
            if nb_steps > nb_frames:
                # data has been cropped, we suppose it is centered in z dimension
                theta = theta[(nb_steps - nb_frames) // 2 : (nb_steps + nb_frames) // 2]

        elif setup.rocking_angle == "energy":
            pass
        else:
            raise ValueError('Wrong value for "rocking_angle" parameter')

        return util.bin_parameters(
            binning=setup.detector.binning[0],
            nb_frames=nb_frames,
            params=[theta, phi, delta, gamma, energy],
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

        :param wavelength: X-ray wasvelength in nm
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
            print("using APS 34ID geometry")

        if rocking_angle == "inplane":
            if grazing_angle is not None:
                raise NotImplementedError(
                    "Circle blow theta not implemented for 34ID-C"
                )
            if verbose:
                print("rocking angle is theta, no grazing angle (phi above theta)")
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
            if verbose:
                print(
                    "rocking angle is phi,"
                    f" theta={grazing_angle[0] * 180 / np.pi:.3f} deg"
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
                        -np.sin(grazing_angle[0]) * np.sin(outofplane),
                        np.cos(grazing_angle[0])
                        * (np.cos(inplane) * np.cos(outofplane) - 1),
                        -np.cos(grazing_angle[0]) * np.sin(outofplane),
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

        return mymatrix, q_offset
