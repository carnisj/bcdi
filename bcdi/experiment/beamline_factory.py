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
from abc import ABC, abstractmethod
from numbers import Real

import numpy as np
import xrayutilities as xu

from bcdi.experiment.diffractometer import Diffractometer
from bcdi.experiment.loader import create_loader
from bcdi.utils import utilities as util
from bcdi.utils import validation as valid

module_logger = logging.getLogger(__name__)


class Beamline(ABC):
    """
    Base class for defining a beamline.

    :param name: name of the beamline
    :param kwargs:

     - optional beamline-dependent parameters
     - 'logger': an optional logger

    """

    orientation_lookup = {"x-": 1, "x+": -1, "y-": 1, "y+": -1}  # lookup table for the
    # detector orientation and rotation direction, where axes are expressed in the
    # laboratory frame (z downstream, y vertical up, x outboard).
    # Expected detector orientation:
    # "x-" detector horizontal axis inboard, as it should be in the CXI convention
    # "y-" detector vertical axis down, as it should be in the CXI convention

    def __init__(self, name, **kwargs):
        self.logger = kwargs.get("logger", module_logger)
        self.name = name
        self.diffractometer = Diffractometer(name=name, **kwargs)
        loader_kwargs = {"logger": self.logger} if kwargs.get("logger") else {}
        self.loader = create_loader(
            name=name,
            sample_offsets=self.diffractometer.sample_offsets,
            **loader_kwargs,
        )
        self.sample_angles = None
        self.detector_angles = None

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

    def exit_wavevector(self, wavelength, inplane_angle, outofplane_angle):
        """
        Calculate the exit wavevector kout.

        It uses the setup parameters. kout is expressed in 1/m in the
        laboratory frame (z downstream, y vertical, x outboard).

        :param wavelength: float, X-ray wavelength in meters.
        :param inplane_angle: float, horizontal detector angle, in degrees.
        :param outofplane_angle: float, vertical detector angle, in degrees.
        :return: kout vector as a numpy array of shape (3)
        """
        # look for the index of the inplane detector circle
        index = self.find_inplane()

        factor = self.orientation_lookup[self.diffractometer.detector_circles[index]]

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

    def find_inplane(self):
        """
        Find the index of the detector inplane circle.

        It looks for the index of the detector inplane rotation in the detector_circles
        property of the diffractometer ("y+" or "y-") . The coordinate
        convention is the laboratory  frame (z downstream, y vertical up, x outboard).

        :return: int, the index. None if not found.
        """
        index = None
        for idx, val in enumerate(self.diffractometer.detector_circles):
            if val.startswith("y"):
                index = idx
        return index

    def find_outofplane(self):
        """
        Find the index of the detector out-of-plane circle.

        It looks for the index of the detector out-of-plane rotation in the
        detector_circles property of the diffractometer (typically "x-") . The
        coordinate convention is the laboratory  frame (z downstream, y vertical up,
        x outboard). This is useful only for SIXS where there are two out-of-plane
        detector rotations due to the beta circle. We need the index of the most inner
        circle, not beta.

        :return: int, the index. None if not found.
        """
        index = None
        for idx, val in enumerate(self.diffractometer.detector_circles):
            if val.startswith("x"):
                index = idx
        return index

    def flatten_sample(
        self,
        arrays,
        voxel_size,
        q_com,
        rocking_angle,
        central_angle=None,
        fill_value=0,
        is_orthogonal=True,
        reciprocal_space=False,
        debugging=False,
        **kwargs,
    ):
        """
        Send all sample circles to zero degrees.

        Arrays are rotatedsuch that all circles of the sample stage are at their zero
        position.

        :param arrays: tuple of 3D real arrays of the same shape.
        :param voxel_size: tuple, voxel size of the 3D array in z, y, and x
         (CXI convention)
        :param q_com: diffusion vector of the center of mass of the Bragg peak,
         expressed in an orthonormal frame x y z
        :param rocking_angle: angle which is tilted during the rocking curve in
         {'outofplane', 'inplane'}
        :param central_angle: if provided, angle to be used in the calculation
         of the rotation matrix for the rocking angle. If None, it will be defined as
         the angle value at the middle of the rocking curve.
        :param fill_value: tuple of numeric values used in the RegularGridInterpolator
         for points outside of the interpolation domain. The length of the tuple
         should be equal to the number of input arrays.
        :param is_orthogonal: set to True is the frame is orthogonal, False otherwise.
         Used for plot labels.
        :param reciprocal_space: True if the data is in reciprocal space,
         False otherwise. Used for plot labels.
        :param debugging: tuple of booleans of the same length as the number
         of input arrays, True to see plots before and after rotation
        :param kwargs:

         - 'cmap': str, name of the colormap
         - 'title': tuple of strings, titles for the debugging plots, same length as
           the number of arrays
         - 'scale': tuple of strings (either 'linear' or 'log'), scale for the
           debugging plots, same length as the number of arrays
         - width_z: size of the area to plot in z (axis 0), centered on the middle
           of the initial array
         - width_y: size of the area to plot in y (axis 1), centered on the middle
           of the initial array
         - width_x: size of the area to plot in x (axis 2), centered on the middle
           of the initial array

        :return: a rotated array (if a single array was provided) or a tuple of
         rotated arrays (same length as the number of input arrays)
        """
        valid.valid_ndarray(arrays, ndim=3)

        # check few parameters, the rest will be validated in rotate_crystal
        valid.valid_container(
            q_com,
            container_types=(tuple, list, np.ndarray),
            length=3,
            item_types=Real,
            name="q_com",
        )
        if np.linalg.norm(q_com) == 0:
            raise ValueError("the norm of q_com is zero")
        if self.sample_angles is None:
            raise ValueError(
                "call diffractometer.goniometer_values before calling this method"
            )
        valid.valid_item(
            central_angle, allowed_types=Real, allow_none=True, name="central_angle"
        )
        # find the index of the circle which corresponds to the rocking angle
        angles = self.sample_angles
        rocking_circle = self.diffractometer.get_rocking_circle(
            rocking_angle=rocking_angle, stage_name="sample", angles=angles
        )

        # get the relevant angle within the rocking circle.
        # The reference point when orthogonalizing if the center of the array,
        # but we do not know to which angle it corresponds if the data was cropped.
        if central_angle is None:
            self.logger.info(
                "central_angle=None, using the angle at half of the rocking curve"
                " for the calculation of the rotation matrix"
            )
            nb_steps = len(angles[rocking_circle])
            central_angle = angles[rocking_circle][int(nb_steps // 2)]

        # use this angle in the calculation of the rotation matrix
        angles = list(angles)
        angles[rocking_circle] = central_angle
        self.logger.info(
            f"sample stage circles: {self.diffractometer.sample_circles}\n"
            f"sample stage angles:  {angles}"
        )

        # check that all angles are Real, not encapsulated in a list or an array
        for idx, angle in enumerate(angles):
            if not isinstance(angle, Real):  # list/tuple or ndarray, cannot be None
                if len(angle) != 1:
                    raise ValueError(
                        "A list of angles was provided instead of a single value"
                    )
                angles[idx] = angle[0]

        # calculate the rotation matrix
        rotation_matrix = self.diffractometer.rotation_matrix(
            stage_name="sample", angles=angles
        )

        # rotate the arrays
        rotated_arrays = util.rotate_crystal(
            arrays=arrays,
            rotation_matrix=rotation_matrix,
            voxel_size=voxel_size,
            fill_value=fill_value,
            debugging=debugging,
            is_orthogonal=is_orthogonal,
            reciprocal_space=reciprocal_space,
            **kwargs,
        )
        rotated_q = util.rotate_vector(
            vectors=q_com, rotation_matrix=np.linalg.inv(rotation_matrix)
        )
        return rotated_arrays, rotated_q

    @abstractmethod
    def goniometer_values(self, setup, **kwargs):
        """
        Retrieve goniometer values.

        This method is beamline dependent. It must be implemented in the child classes.

        :param setup: the experimental setup: Class Setup
        :param kwargs: beamline_specific parameters
        :return: a tuple of angular values in degrees (rocking angular step, grazing
         incidence angles, inplane detector angle, outofplane detector angle). The
         grazing incidence angles are the positions of circles below the rocking circle.
        """

    def init_qconversion(self, conversion_table, beam_direction, offset_inplane):
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
        :return: a tuple containing:

         - the qconv object for xrayutilities
         - a tuple of motor offsets used later for q calculation

        """
        # look for the index of the inplane detector circle
        index = self.find_inplane()

        # convert axes from the laboratory frame to the frame of xrayutilies
        sample_circles = [
            conversion_table[val] for val in self.diffractometer.sample_circles
        ]
        detector_circles = [
            conversion_table[val] for val in self.diffractometer.detector_circles
        ]
        qconv = xu.experiment.QConversion(
            sample_circles, detector_circles, r_i=beam_direction
        )

        # create the tuple of offsets, all 0 except for the detector inplane circle
        if index is None:
            self.logger.info(
                "no detector inplane circle detected, discarding 'offset_inplane'"
            )
            offsets = [0 for _ in range(len(sample_circles) + len(detector_circles))]
        else:
            offsets = [0 for _ in range(len(sample_circles) + index)]
            offsets.append(offset_inplane)
            for _ in range(len(detector_circles) - index - 1):
                offsets.append(0)

        return qconv, offsets

    def inplane_coeff(self):
        """
        Coefficient related to the detector inplane orientation.

        Define a coefficient +/- 1 depending on the detector inplane rotation direction
        (1 for clockwise, -1 for anti-clockwise) and the detector inplane orientation
        (1 for inboard, -1 for outboard).

        See scripts/postprocessing/correct_angles_detector.py for a use case.

        :return: +1 or -1
        """
        # look for the index of the inplane detector circle
        index = self.find_inplane()
        return (
            self.orientation_lookup[self.diffractometer.detector_circles[index]]
            * self.orientation_lookup[self.detector_hor]
        )

    def outofplane_coeff(self):
        """
        Coefficient related to the detector vertical orientation.

        Define a coefficient +/- 1 depending on the detector out of plane rotation
        direction (1 for clockwise, -1 for anti-clockwise) and the detector out of
        plane orientation (1 for downward, -1 for upward).

        See scripts/postprocessing/correct_angles_detector.py for a use case.

        :return: +1 or -1
        """
        # look for the index of the out-of-plane detector circle
        index = self.find_outofplane()
        return (
            self.orientation_lookup[self.diffractometer.detector_circles[index]]
            * self.orientation_lookup[self.detector_ver]
        )

    @staticmethod
    @abstractmethod
    def process_positions(setup, nb_frames, scan_number, frames_logical=None):
        """
        Load and crop/pad motor positions depending on the current number of frames.

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
        motor_positions = setup.loader.motor_positions(
            setup=setup,
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

    def __repr__(self):
        """Representation string of the Beamline instance."""
        return util.create_repr(self, Beamline)

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


class BeamlineSaxs(Beamline):
    """
    Base class for defining a beamline in the SAXS geometry.

    The detector is fixed and its plane is always perpendicular to the direct beam,
    independently of its position.

    :param name: name of the beamline
    :param kwargs:

     - optional beamline-dependent parameters
     - 'logger': an optional logger

    """
