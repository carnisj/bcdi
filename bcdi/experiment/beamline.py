# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#   (c) 06/2021-present : DESY CFEL
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

"""
Implementation of beamline-related classes.

The class methods manage the initialization of the file system and the calculations
related to reciprocal or direct space transformation (interpolation in an orthonormal
grid). Generic method are implemented in the abstract base class Beamline, and
beamline-dependent methods need to be implemented in each child class (they are
decorated by @abstractmethod in the base class; they are indicated using @ in the
following diagram). These classes are not meant to be instantiated directly but via a
Setup instance.

.. mermaid::
  :align: center

  classDiagram
    class Beamline{
      +str name
      create_logfile(@)
      detector_hor(@)
      detector_ver(@)
      init_paths(@)
      process_positions(@)
      transformation_matrix(@)
      exit_wavevector()
      find_inplane()
      find_outofplane()
      init_qconversion()
      inplane_coeff()
      outofplane_coeff()
      process_tilt()
  }
    ABC <|-- Beamline

API Reference
-------------

"""
from abc import ABC, abstractmethod
import h5py
from math import isclose
import numpy as np
from numbers import Real
import os
from silx.io.specfile import SpecFile
import xrayutilities as xu

from bcdi.graph import graph_utils as gu
from bcdi.utils import utilities as util
from bcdi.utils import validation as valid


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
    if name == "P10_SAXS":
        return BeamlineP10SAXS(name=name, **kwargs)
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
          - 'name': str, the name of the beamline, e.g. 'SIXS_2019'

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
    def init_paths(root_folder, sample_name, scan_number, template_imagefile, **kwargs):
        """
        Initialize paths used for data processing and logging.

        :param root_folder: folder of the experiment, where all scans are stored
        :param sample_name: string in front of the scan number in the data folder
         name.
        :param scan_number: int, the scan number
        :param template_imagefile: beamline-dependent template for the data files:

         - ID01: 'data_mpx4_%05d.edf.gz' or 'align_eiger2M_%05d.edf.gz'
         - SIXS_2018: 'align.spec_ascan_mu_%05d.nxs'
         - SIXS_2019: 'spare_ascan_mu_%05d.nxs'
         - Cristal: 'S%d.nxs'
         - P10: '_master.h5'
         - NANOMAX: '%06d.h5'
         - 34ID: 'Sample%dC_ES_data_51_256_256.npz'

        :param kwargs: dictionnary of the setup parameters including the following keys:

         - 'specfile_name': beamline-dependent string:

           - ID01: name of the spec file without '.spec'
           - SIXS_2018 and SIXS_2019: None or full path of the alias dictionnary (e.g.
             root_folder+'alias_dict_2019.txt')
           - empty string for all other beamlines

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
    def process_positions(setup, logfile, nb_frames, scan_number, frames_logical=None):
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
        :return: a tuple of 1D arrays (sample circles, detector circles, energy)
        """
        motor_positions = setup.diffractometer.motor_positions(
            setup=setup,
            logfile=logfile,
            scan_number=scan_number,
        )
        # remove the motor positions corresponding to deleted frames during data
        # loading (frames_logical = 0)
        return util.apply_logical_array(
            arrays=motor_positions,
            frames_logical=frames_logical,
        )

    @staticmethod
    def process_tilt(array, nb_steps, nb_frames, angular_step):
        """
        Crop or pad array depending on how compare two numbers.

        Cropping or padding depends on the number of current frames compared to the
        number of motor steps. For padding it assumes that array is linear in the
        angular_step.

        :param array: a 1D numpy array of motor values
        :param nb_steps: int, the number of motor positions
        :param nb_frames: int, the number of frames
        :param angular_step: float, the angular tilt of the rocking curve
        :return: the cropped/padded array
        """
        # check parameters
        valid.valid_1d_array(array, length=nb_steps, allow_none=False, name="array")
        valid.valid_item(nb_steps, allowed_types=int, min_excluded=0, name="nb_steps")
        valid.valid_item(nb_frames, allowed_types=int, min_excluded=0, name="nb_frames")
        valid.valid_item(angular_step, allowed_types=Real, name="angular_step")

        if nb_steps < nb_frames:
            # data has been padded, we suppose it is centered in z dimension
            pad_low = int((nb_frames - nb_steps + ((nb_frames - nb_steps) % 2)) / 2)
            pad_high = int(
                (nb_frames - nb_steps + 1) / 2 - ((nb_frames - nb_steps) % 2)
            )
            array = np.concatenate(
                (
                    array[0] + np.arange(-pad_low, 0, 1) * angular_step,
                    array,
                    array[-1] + np.arange(1, pad_high + 1, 1) * angular_step,
                ),
                axis=0,
            )
        if nb_steps > nb_frames:
            # data has been cropped, we suppose it is centered in z dimension
            array = array[(nb_steps - nb_frames) // 2 : (nb_steps + nb_frames) // 2]
        return array

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


class BeamlineCRISTAL(Beamline):
    """
    Definition of SOLEIL CRISTAL beamline.

    :param name: name of the beamline
    """

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

    @staticmethod
    def create_logfile(**kwargs):
        """
        Create the logfile, which is the data itself for CRISTAL.

        :param kwargs:
         - 'datadir': str, the data directory
         - 'template_imagefile': str, template for data file name, e.g. 'S%d.nxs'
         - 'scan_number': int, the scan number to load

        :return: logfile
        """
        datadir = kwargs.get("datadir")
        template_imagefile = kwargs.get("template_imagefile")
        scan_number = kwargs.get("scan_number")

        if not os.path.isdir(datadir):
            raise ValueError(f"The directory {datadir} does not exist")
        valid.valid_container(
            template_imagefile, container_types=str, name="template_imagefile"
        )
        valid.valid_item(
            scan_number, allowed_types=int, min_included=0, name="scan_number"
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
         - specfile: not used at CRISTAL
         - template_imagefile: the template for data/image file names

        """
        homedir = root_folder + sample_name + str(scan_number) + "/"
        default_dirname = "data/"
        return homedir, default_dirname, None, template_imagefile

    def process_positions(
        self,
        setup,
        logfile,
        nb_frames,
        scan_number,
        frames_logical=None,
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
        :return: a tuple of 1D arrays (sample circles, detector circles, energy)
        """
        mgomega, mgphi, gamma, delta, energy = super().process_positions(
            setup=setup,
            logfile=logfile,
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
            print("mgomega", mgomega)
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

    @staticmethod
    def create_logfile(**kwargs):
        """
        Create the logfile, which is the spec file for ID01.

        :param kwargs:
         - 'root_folder': str, the root directory of the experiment, where is e.g. the
           specfile file.
         - 'filename': str, name of the spec file or full path of the spec file

        :return: logfile
        """
        root_folder = kwargs.get("root_folder")
        filename = kwargs.get("filename")

        valid.valid_container(
            filename,
            container_types=str,
            min_length=1,
            name="filename",
        )

        if os.path.isfile(filename):
            # filename is already the full path to the .spec file
            return SpecFile(filename)
        print(f"Could not find the spec file at {filename}")

        if not os.path.isdir(root_folder):
            raise ValueError(f"The directory {root_folder} does not exist")
        path = root_folder + filename
        print(f"Trying to load the spec file at {path}")
        return SpecFile(path)

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
        Initialize paths used for data processing and logging at ID01.

        :param root_folder: folder of the experiment, where all scans are stored
        :param sample_name: string in front of the scan number in the data folder
         name.
        :param scan_number: int, the scan number
        :param template_imagefile: template for the data files, e.g.
         'data_mpx4_%05d.edf.gz' or 'align_eiger2M_%05d.edf.gz'
        :param kwargs:
         - 'specfile_name': name of the spec file without '.spec'

        :return: a tuple of strings:

         - homedir: the path of the scan folder
         - default_dirname: the name of the folder containing images / raw data
         - specfile: the name of the specfile if it exists
         - template_imagefile: the template for data/image file names

        """
        specfile_name = kwargs.get("specfile_name")

        homedir = root_folder + sample_name + str(scan_number) + "/"
        default_dirname = "data/"
        return homedir, default_dirname, specfile_name, template_imagefile

    def process_positions(
        self,
        setup,
        logfile,
        nb_frames,
        scan_number,
        frames_logical=None,
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
        :return: a tuple of 1D arrays (sample circles, detector circles, energy)
        """
        mu, eta, phi, nu, delta, energy = super().process_positions(
            setup=setup,
            logfile=logfile,
            nb_frames=nb_frames,
            scan_number=scan_number,
            frames_logical=frames_logical,
        )

        # eventually crop/pad motor values if the provided dataset was further
        # cropped/padded
        if setup.rocking_angle == "outofplane":  # eta rocking curve
            print("phi", phi)
            nb_steps = len(eta)
            tilt_angle = (eta[1:] - eta[0:-1]).mean()
            eta = self.process_tilt(
                eta, nb_steps=nb_steps, nb_frames=nb_frames, angular_step=tilt_angle
            )
        elif setup.rocking_angle == "inplane":  # phi rocking curve
            print("eta", eta)
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
            print("using ESRF ID01 PSIC geometry")

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
            if len(grazing_angle) != 2:
                raise ValueError("grazing_angle should be of length 2")
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

    @staticmethod
    def create_logfile(**kwargs):
        """
        Create the logfile, which is the data itself for Nanomax.

        :param kwargs:
         - 'datadir': str, the data directory
         - 'template_imagefile': str, template for data file name, e.g. '%06d.h5'
         - 'scan_number': int, the scan number to load

        :return: logfile
        """
        datadir = kwargs.get("datadir")
        template_imagefile = kwargs.get("template_imagefile")
        scan_number = kwargs.get("scan_number")

        if not os.path.isdir(datadir):
            raise ValueError(f"The directory {datadir} does not exist")
        valid.valid_container(
            template_imagefile, container_types=str, name="template_imagefile"
        )
        valid.valid_item(
            scan_number, allowed_types=int, min_included=0, name="scan_number"
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
        return homedir, default_dirname, None, template_imagefile

    def process_positions(
        self,
        setup,
        logfile,
        nb_frames,
        scan_number,
        frames_logical=None,
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
        :return: a tuple of 1D arrays (sample circles, detector circles, energy)
        """
        theta, phi, gamma, delta, energy = super().process_positions(
            setup=setup,
            logfile=logfile,
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

    @staticmethod
    def create_logfile(**kwargs):
        """
        Create the logfile, which is the .fio file for P10.

        :param kwargs:
         - 'root_folder': str, the root directory of the experiment, where the scan
           folders are located.
         - 'filename': str, name of the .fio file or full path of the .fio file

        :return: logfile
        """
        root_folder = kwargs.get("root_folder")
        filename = kwargs.get("filename")

        valid.valid_container(
            filename,
            container_types=str,
            min_length=1,
            name="filename",
        )

        if os.path.isfile(filename):
            # filename is already the full path to the .fio file
            return filename
        print(f"Could not find the spec file at {filename}")

        if not os.path.isdir(root_folder):
            raise ValueError(f"The directory {root_folder} does not exist")

        # return the path to the .fio file
        path = root_folder + filename + "/" + filename + ".fio"
        print(f"Trying to load the fio file at {path}")
        return path

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
        :param kwargs:
         - 'specfile_name': optional, full path of the .fio file

        :return: a tuple of strings:

         - homedir: the path of the scan folder
         - default_dirname: the name of the folder containing images / raw data
         - specfile: the name of the specfile if it exists
         - template_imagefile: the template for data/image file names

        """
        specfile = kwargs.get("specfile_name")
        if specfile is None or not os.path.isfile(specfile):
            # default to the usual position of .fio at P10
            specfile = sample_name + "_{:05d}".format(scan_number)

        homedir = root_folder + specfile + "/"
        default_dirname = "e4m/"
        template_imagefile = specfile + template_imagefile
        return homedir, default_dirname, specfile, template_imagefile

    def process_positions(
        self,
        setup,
        logfile,
        nb_frames,
        scan_number,
        frames_logical=None,
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
        :return: a tuple of 1D arrays (sample circles, detector circles, energy)
        """
        mu, om, chi, phi, gamma, delta, energy = super().process_positions(
            setup=setup,
            logfile=logfile,
            nb_frames=nb_frames,
            scan_number=scan_number,
            frames_logical=frames_logical,
        )

        # eventually crop/pad motor values if the provided dataset was further
        # cropped/padded
        print("chi", chi)
        print("mu", mu)
        if setup.rocking_angle == "outofplane":  # om rocking curve
            print("phi", phi)
            nb_steps = len(om)
            tilt_angle = (om[1:] - om[0:-1]).mean()
            om = self.process_tilt(
                om, nb_steps=nb_steps, nb_frames=nb_frames, angular_step=tilt_angle
            )
        elif setup.rocking_angle == "inplane":  # phi rocking curve
            print("om", om)
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
            print("using PETRAIII P10 geometry")

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
            if len(grazing_angle) != 3:
                raise ValueError("grazing_angle should be of length 3")
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

        interp_radius = np.multiply(sign_array, np.sqrt(x_interp ** 2 + z_interp ** 2))

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
                    np.sqrt(distance ** 2 + np.power(myx * pixelsize_x, 2)),
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


class BeamlineSIXS(Beamline):
    """
    Definition of SOLEIL SIXS beamline.

    :param name: name of the beamline
    """

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

    @staticmethod
    def create_logfile(**kwargs):
        """
        Create the logfile, which is the data itself for SIXS.

        :param kwargs:
         - 'datadir': str, the data directory
         - 'template_imagefile': str, template for data file name:

           - SIXS_2018: 'align.spec_ascan_mu_%05d.nxs'
           - SIXS_2019: 'spare_ascan_mu_%05d.nxs'

         - 'scan_number': int, the scan number to load
         - 'filename': str, absolute path of 'alias_dict.txt'
         - 'name': str, the name of the beamline, e.g. 'SIXS_2019'

        :return: logfile
        """
        datadir = kwargs.get("datadir")
        template_imagefile = kwargs.get("template_imagefile")
        scan_number = kwargs.get("scan_number")
        filename = kwargs.get("filename")
        name = kwargs.get("name")

        if not os.path.isdir(datadir):
            raise ValueError(f"The directory {datadir} does not exist")
        valid.valid_container(
            template_imagefile, container_types=str, name="template_imagefile"
        )
        valid.valid_container(filename, container_types=str, name="filename")
        valid.valid_item(
            scan_number, allowed_types=int, min_included=0, name="scan_number"
        )

        shortname = template_imagefile % scan_number
        if name == "SIXS_2018":
            # no specfile, load directly the dataset
            import bcdi.preprocessing.nxsReady as nxsReady

            return nxsReady.DataSet(
                longname=datadir + shortname,
                shortname=shortname,
                alias_dict=filename,
                scan="SBS",
            )
        if name == "SIXS_2019":
            # no specfile, load directly the dataset
            import bcdi.preprocessing.ReadNxs3 as ReadNxs3

            return ReadNxs3.DataSet(
                directory=datadir,
                filename=shortname,
                alias_dict=filename,
            )
        raise NotImplementedError(f"{name} is not implemented")

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
        Initialize paths used for data processing and logging at SIXS.

        :param root_folder: folder of the experiment, where all scans are stored
        :param sample_name: string in front of the scan number in the data folder
         name.
        :param scan_number: int, the scan number
        :param template_imagefile: template for the data files, e.g.
         'align.spec_ascan_mu_%05d.nxs' (SIXS_2018), 'spare_ascan_mu_%05d.nxs'
         (SIXS_2019).
        :param kwargs:
         - 'specfile_name': None or full path of the alias dictionnary (e.g.
           root_folder+'alias_dict_2019.txt')

        :return: a tuple of strings:

         - homedir: the path of the scan folder
         - default_dirname: the name of the folder containing images / raw data
         - specfile: the name of the specfile if it exists
         - template_imagefile: the template for data/image file names

        """
        specfile_name = kwargs.get("specfile_name")

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

    def process_positions(
        self,
        setup,
        logfile,
        nb_frames,
        scan_number,
        frames_logical=None,
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
        :return: a tuple of 1D arrays (sample circles, detector circles, energy)
        """
        beta, mu, gamma, delta, energy = super().process_positions(
            setup=setup,
            logfile=logfile,
            nb_frames=nb_frames,
            scan_number=scan_number,
            frames_logical=frames_logical,
        )

        # eventually crop/pad motor values if the provided dataset was further
        # cropped/padded
        print("beta", beta)
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
            print("using SIXS geometry")

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
            raise NotImplementedError(f"rocking_angle={rocking_angle} not implemented")

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
        """
        Create the logfile, which is the spec file for 34ID-C.

        :param kwargs:
         - 'root_folder': str, the root directory of the experiment, where is e.g. the
           specfile file.
         - 'filename': str, name of the spec file or full path of the .spec file

        :return: logfile
        """
        root_folder = kwargs.get("root_folder")
        filename = kwargs.get("filename")

        valid.valid_container(
            filename,
            container_types=str,
            min_length=1,
            name="filename",
        )

        if os.path.isfile(filename):
            # filename is already the full path to the .spec file
            return SpecFile(filename)
        print(f"Could not find the spec file at {filename}")

        if not os.path.isdir(root_folder):
            raise ValueError(f"The directory {root_folder} does not exist")
        path = root_folder + filename
        print(f"Trying to load the spec file at {path}")
        return SpecFile(path)

    @property
    def detector_hor(self):
        """
        Horizontal detector orientation expressed in the laboratory frame.

        We look at the detector from upstream, detector X is opposite to the outboard
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
        Initialize paths used for data processing and logging at 34ID-C.

        :param root_folder: folder of the experiment, where all scans are stored
        :param sample_name: string in front of the scan number in the data folder
         name.
        :param scan_number: int, the scan number
        :param template_imagefile: template for the data files, e.g.
         'Sample%dC_ES_data_51_256_256.npz'.
        :param kwargs:
         - 'specfile_name': name of the spec file without '.spec'

        :return: a tuple of strings:

         - homedir: the path of the scan folder
         - default_dirname: the name of the folder containing images / raw data
         - specfile: the name of the specfile if it exists
         - template_imagefile: the template for data/image file names

        """
        specfile_name = kwargs.get("specfile_name")

        homedir = root_folder + sample_name + str(scan_number) + "/"
        default_dirname = "data/"
        return homedir, default_dirname, specfile_name, template_imagefile

    def process_positions(
        self,
        setup,
        logfile,
        nb_frames,
        scan_number,
        frames_logical=None,
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
        :return: a tuple of 1D arrays (sample circles, detector circles, energy)
        """
        theta, phi, delta, gamma, energy = super().process_positions(
            setup=setup,
            logfile=logfile,
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
            print("using APS 34ID geometry")

        if rocking_angle == "inplane":
            if grazing_angle is not None:
                raise NotImplementedError(
                    "Circle below theta not implemented for 34ID-C"
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

        else:
            raise NotImplementedError(f"rocking_angle={rocking_angle} not implemented")

        return mymatrix, q_offset
