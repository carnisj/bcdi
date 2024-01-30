# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

"""
Implementation of beamline abstract classes.

The class methods manage the calculations related to reciprocal or direct space
transformation (interpolation in an orthonormal grid). Generic method are implemented
in the abstract base classes Beamline and BeamlineSaxs; beamline-dependent methods
need to be implemented in each child class (they are decorated by @abstractmethod in
the base class; they are written in italic in the following diagram). These classes are
not meant to be instantiated directly (although it is still possible) but via a Setup
instance.

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
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import numpy as np
import xrayutilities as xu

import bcdi.utils.format as fmt
from bcdi.experiment.diffractometer import DiffractometerFactory
from bcdi.experiment.loader import create_loader
from bcdi.graph import graph_utils as gu
from bcdi.utils import utilities as util
from bcdi.utils import validation as valid

if TYPE_CHECKING:
    from bcdi.experiment.setup import Setup

module_logger = logging.getLogger(__name__)


class Beamline(ABC):
    """
    Base class for defining a beamline.

    :param name: name of the beamline
    :param sample_offsets: list or tuple of angles in degrees, corresponding to
     the offsets of each of the sample circles (the offset for the most outer circle
     should be at index 0). The number of circles is beamline dependent. Convention:
     the sample offsets will be subtracted to measurement the motor values.
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

    def __init__(self, name, sample_offsets=None, **kwargs):
        self.logger = kwargs.get("logger", module_logger)
        self.name = name
        self.diffractometer = DiffractometerFactory().create_diffractometer(
            name=name, sample_offsets=sample_offsets, **kwargs
        )
        self.loader = create_loader(
            name=name,
            sample_offsets=self.diffractometer.sample_offsets,
            **kwargs,
        )
        self.sample_angles: Optional[Tuple[Union[float, List, np.ndarray]]] = None

    @property
    @abstractmethod
    def detector_hor(self):
        """
        Horizontal detector orientation expressed in the laboratory frame.

        The orientation is defined when the detector plane is perpendicular to the
        direct beam. This is beamline-dependent. The laboratory frame convention is
        (z downstream, y vertical, x outboard).

        :return: "x+" or "x-"
        """

    @property
    @abstractmethod
    def detector_ver(self):
        """
        Vertical detector orientation expressed in the laboratory frame.

        The orientation is defined when the detector plane is perpendicular to the
        direct beam. This is beamline-dependent. The laboratory frame convention is
        (z downstream, y vertical, x outboard).

        :return: "y+" or "y-"
        """

    def flatten_sample(
        self,
        arrays,
        voxel_size,
        q_bragg,
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

        Arrays are rotated such that all circles of the sample stage are at their zero
        position.

        :param arrays: tuple of 3D real arrays of the same shape.
        :param voxel_size: tuple, voxel size of the 3D array in z, y, and x
         (CXI convention)
        :param q_bragg: diffusion vector of the center of mass of the Bragg peak,
         expressed in an orthonormal frame x y z
        :param rocking_angle: angle which is tilted during the rocking curve in
         {'outofplane', 'inplane'}
        :param central_angle: if provided, angle to be used in the calculation
         of the rotation matrix for the rocking angle. If None, it will be defined as
         the angle value at the middle of the rocking curve.
        :param fill_value: tuple of numeric values used in the RegularGridInterpolator
         for points outside the interpolation domain. The length of the tuple
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
            q_bragg,
            container_types=(tuple, list, np.ndarray),
            length=3,
            item_types=Real,
            name="q_bragg",
        )
        if np.linalg.norm(q_bragg) == 0:
            raise ValueError("the norm of q_bragg is zero")
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
            vectors=q_bragg, rotation_matrix=np.linalg.inv(rotation_matrix)
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

    @abstractmethod
    def inplane_coeff(self):
        """
        Coefficient related to the detector inplane orientation.

        Define a coefficient +/- 1 depending on the detector inplane rotation direction
        (1 for clockwise, -1 for anti-clockwise) and the detector inplane orientation
        (1 for inboard, -1 for outboard).

        For a SAXS Beamline it does not matter, the fixed detector inplane angle is 0.

        :return: +1
        """
        raise NotImplementedError

    @abstractmethod
    def outofplane_coeff(self):
        """
        Coefficient related to the detector inplane orientation.

        Define a coefficient +/- 1 depending on the detector inplane rotation direction
        (1 for clockwise, -1 for anti-clockwise) and the detector inplane orientation
        (1 for inboard, -1 for outboard).

        For a SAXS Beamline it does not matter, the fixed detector inplane angle is 0.

        :return: +1
        """
        raise NotImplementedError

    @abstractmethod
    def process_positions(
        self,
        setup: "Setup",
        nb_frames: int,
        scan_number: int,
        frames_logical: Optional[np.ndarray] = None,
    ) -> Any:
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
    def process_tilt(array, nb_steps, nb_frames, step_size):
        """
        Crop or pad array depending on how compare two numbers.

        Cropping or padding depends on the number of current frames compared to the
        number of motor steps. For pading it assumes that array is linear in the
        angular_step.

        :param array: a 1D numpy array of motor values
        :param nb_steps: int, the number of motor positions
        :param nb_frames: int, the number of frames
        :param step_size: float, the angular tilt of the rocking curve
        :return: the cropped/padded array
        """
        # check parameters
        valid.valid_1d_array(array, length=nb_steps, allow_none=False, name="array")
        valid.valid_item(nb_steps, allowed_types=int, min_excluded=0, name="nb_steps")
        valid.valid_item(nb_frames, allowed_types=int, min_excluded=0, name="nb_frames")
        valid.valid_item(step_size, allowed_types=Real, name="step_size")

        if nb_steps < nb_frames:
            # data has been padded, we suppose it is centered in z dimension
            pad_low = int((nb_frames - nb_steps + ((nb_frames - nb_steps) % 2)) / 2)
            pad_high = int(
                (nb_frames - nb_steps + 1) / 2 - ((nb_frames - nb_steps) % 2)
            )
            array = np.concatenate(
                (
                    array[0] + np.arange(-pad_low, 0, 1) * step_size,
                    array,
                    array[-1] + np.arange(1, pad_high + 1, 1) * step_size,
                ),
                axis=0,
            )
        if nb_steps > nb_frames:
            # data has been cropped, we suppose it is centered in z dimension
            array = array[(nb_steps - nb_frames) // 2 : (nb_steps + nb_frames) // 2]
        return array

    def __repr__(self):
        """Representation string of the Beamline instance."""
        return fmt.create_repr(self, Beamline)

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


class BeamlineGoniometer(Beamline):
    """
    Base class for defining a beamline where the detector is on a goniometer.

    :param name: name of the beamline
    :param kwargs:

     - optional beamline-dependent parameters
     - 'logger': an optional logger

    """

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.detector_angles: Optional[Tuple[Union[float, List, np.ndarray]]] = None

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

    def exit_wavevector(
        self, wavelength: float, inplane_angle: float, outofplane_angle: float
    ) -> np.ndarray:
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

        factor: int = self.orientation_lookup[
            self.diffractometer.detector_circles[index]
        ]

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

    def find_inplane(self) -> int:
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
        if index is None:
            raise ValueError("detector inplane circle not found")
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

    def __repr__(self):
        """Representation string of the Beamline instance."""
        return fmt.create_repr(self, BeamlineGoniometer)

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
    """

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

        interp_radius = np.multiply(sign_array, np.hypot(x_interp, z_interp))

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

    def inplane_coeff(self):
        """
        Coefficient related to the detector inplane orientation.

        Define a coefficient +/- 1 depending on the detector inplane rotation direction
        (1 for clockwise, -1 for anti-clockwise) and the detector inplane orientation
        (1 for inboard, -1 for outboard).

        For a SAXS Beamline it does not matter, the fixed detector inplane angle is 0.

        :return: +1
        """
        return 1

    def outofplane_coeff(self):
        """
        Coefficient related to the detector inplane orientation.

        Define a coefficient +/- 1 depending on the detector inplane rotation direction
        (1 for clockwise, -1 for anti-clockwise) and the detector inplane orientation
        (1 for inboard, -1 for outboard).

        For a SAXS Beamline it does not matter, the fixed detector inplane angle is 0.

        :return: +1
        """
        return 1
