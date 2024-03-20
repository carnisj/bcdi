# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr
"""Implementation of postprocessing analysis classes."""

import logging
import os
import tkinter as tk
from abc import ABC, abstractmethod
from functools import reduce
from tkinter import filedialog
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import yaml
from matplotlib import pyplot as plt
from numpy.fft import fftn, fftshift

import bcdi.graph.graph_utils as gu
import bcdi.graph.linecut as linecut
import bcdi.postprocessing.postprocessing_utils as pu
import bcdi.preprocessing.bcdi_utils as bu
import bcdi.utils.image_registration as reg
import bcdi.utils.utilities as util
import bcdi.utils.validation as valid
from bcdi.constants import AXIS_TO_ARRAY
from bcdi.experiment.setup import Setup
from bcdi.utils.text import Comment

module_logger = logging.getLogger(__name__)


class Analysis(ABC):
    """Base class for the post-processing analysis workflow."""

    def __init__(
        self,
        scan_index: int,
        parameters: Dict[str, Any],
        setup: Setup,
        **kwargs,
    ) -> None:
        self.scan_index = scan_index
        self.parameters = parameters
        self.setup = setup
        self.logger = kwargs.get("logger", module_logger)

        self.file_path = self.get_reconstrutions_path()
        self.data = self.load_reconstruction_data(file_index=0)
        self.original_shape = self.get_shape_during_phasing(
            self.parameters.get("original_size")
        )
        self.extent_phase: Optional[float] = None
        self.q_values: Optional[List[np.ndarray]] = None
        self.voxel_sizes: Optional[List[float]] = None
        self.optimized_range = self.original_shape
        self.sorted_reconstructions_best_first = [0]
        self.nb_reconstructions = len(self.file_path)
        self.comment = self.initialize_comment()
        if os.path.splitext(self.file_path[0])[1] == ".h5":
            self.comment.concatenate("mode")

    @property
    def voxel_sizes(self):
        """List of three positive numbers, voxel size in nm in each dimension."""
        return self._voxel_sizes

    @voxel_sizes.setter
    def voxel_sizes(self, value):
        if value is None:
            self._voxel_sizes = value
        elif isinstance(value, (int, float)):
            self._voxel_sizes = [value, value, value]
        elif isinstance(value, (list, float)):
            valid.valid_container(
                value,
                container_types=(list, tuple),
                length=3,
                min_excluded=0,
                allow_none=False,
                name="voxel_sizes",
            )
            self._voxel_sizes = value
        else:
            raise TypeError("voxel_sizes should be a list of three positive values")

    @property
    def undefined_bragg_peak_but_retrievable(self) -> bool:
        return (
            self.parameters["bragg_peak"] is None
            and self.setup.detector.template_imagefile is not None
        )

    @property
    def detector_angles_correction_needed(self) -> bool:
        return self.parameters["data_frame"] == "detector" and (
            not self.parameters["outofplane_angle"]
            or not self.parameters["inplane_angle"]
        )

    @property
    def get_interplanar_distance(self) -> Optional[float]:
        """Calculate the interplanar distance in nm."""
        q_bragg = (
            self.get_q_bragg_laboratory_frame
            if self.get_q_bragg_laboratory_frame is not None
            else self.get_q_bragg_crystal_frame()
        )
        if q_bragg is None:
            return None
        return float(2 * np.pi / (10 * np.linalg.norm(q_bragg)))

    @property
    def get_normalized_q_bragg_laboratory_frame(self) -> Optional[np.ndarray]:
        if self.setup.q_laboratory is None:
            return None
        return self.setup.q_laboratory / float(np.linalg.norm(self.setup.q_laboratory))

    @property
    def get_norm_q_bragg(self) -> Optional[float]:
        q_bragg = (
            self.get_q_bragg_laboratory_frame
            if self.get_q_bragg_laboratory_frame is not None
            else self.get_q_bragg_crystal_frame()
        )
        if q_bragg is None:
            return None
        return float(np.linalg.norm(q_bragg))

    @property
    def get_q_bragg_laboratory_frame(self) -> Optional[np.ndarray]:
        return self.setup.q_laboratory

    @property
    def is_data_in_laboratory_frame(self):
        return self.parameters["data_frame"] == "laboratory"

    @property
    def scan_index(self) -> int:
        return self._scan_index

    @scan_index.setter
    def scan_index(self, val: int) -> None:
        if val < 0:
            raise ValueError("'scan_index' should be >= 0, " f"got {val}")
        self._scan_index = val

    def average_reconstructions(self) -> None:
        """Create an average reconstruction out of many, after proper alignment."""
        average_array = np.zeros(self.optimized_range)
        average_counter = 1
        self.logger.info(
            f"Averaging using {self.nb_reconstructions} " "candidate reconstructions"
        )
        for counter, value in enumerate(self.sorted_reconstructions_best_first):
            obj, extension = util.load_file(self.file_path[value])
            self.logger.info(f"Opening {self.file_path[value]}")
            self.parameters[f"from_file_{counter}"] = self.file_path[value]

            if self.parameters["flip_reconstruction"]:
                obj = pu.flip_reconstruction(
                    obj, debugging=True, cmap=self.parameters["colormap"].cmap
                )

            if extension == ".h5":  # data is already cropped by PyNX
                self.parameters["centering_method"]["direct_space"] = "skip"
                # correct a roll after the decomposition into modes in PyNX
                obj = np.roll(obj, self.parameters["roll_modes"], axis=(0, 1, 2))
                gu.multislices_plot(
                    abs(obj),
                    sum_frames=True,
                    plot_colorbar=True,
                    title="1st mode after centering",
                    cmap=self.parameters["colormap"].cmap,
                )

            # use the range of interest defined above
            obj = util.crop_pad(
                array=obj,
                output_shape=self.optimized_range,
                debugging=False,
                cmap=self.parameters["colormap"].cmap,
            )

            # align with average reconstruction
            if counter == 0:  # the fist array loaded will serve as reference object
                self.logger.info("This reconstruction will be used as reference.")
                self.data = obj

            average_array, flag_avg = reg.average_arrays(
                avg_obj=average_array,
                ref_obj=self.data,
                obj=obj,
                support_threshold=0.25,
                correlation_threshold=self.parameters["correlation_threshold"],
                aligning_option="dft",
                space=self.parameters["averaging_space"],
                reciprocal_space=False,
                is_orthogonal=self.parameters["is_orthogonal"],
                debugging=self.parameters["debug"],
                cmap=self.parameters["colormap"].cmap,
            )
            average_counter += flag_avg

        self.data = average_array / average_counter
        if average_counter > 1:
            self.logger.info(f"Average performed over {average_counter} arrays")

    def center_object_based_on_modulus(self, **kwargs):
        """
        Center the object based on one of the methods 'max', 'com' and 'max_com'.

        :param kwargs:

         - 'centering_method': centering method name among 'max', 'com' and 'max_com'

        """
        method = (
            kwargs.get("centering_method")
            or self.parameters["centering_method"]["direct_space"]
        )
        self.data = pu.center_object(method=method, obj=self.data)

    def crop_pad_data(self, value: Tuple[int, ...]) -> None:
        self.data = util.crop_pad(
            array=self.data,
            output_shape=value,
            cmap=self.parameters["colormap"].cmap,
        )

    def find_best_reconstruction(self) -> None:
        if self.nb_reconstructions > 1:
            self.logger.info(
                "Trying to find the best reconstruction\nSorting by "
                f"{self.parameters['sort_method']}"
            )
            self.sorted_reconstructions_best_first = pu.sort_reconstruction(
                file_path=self.file_path,
                amplitude_threshold=self.parameters["isosurface_strain"],
                data_range=self.optimized_range,
                sort_method=self.parameters["sort_method"],
            )
        else:
            self.sorted_reconstructions_best_first = [0]

    def find_data_range(
        self, amplitude_threshold: float = 0.1, plot_margin: Union[int, List[int]] = 10
    ) -> None:
        self.optimized_range = tuple(
            pu.find_datarange(
                array=self.data,
                amplitude_threshold=amplitude_threshold,
                plot_margin=plot_margin,
                keep_size=self.parameters["keep_size"],
            )
        )

        self.logger.info(
            "Data shape used for orthogonalization and plotting: "
            f"{self.optimized_range}"
        )

    def get_optical_path(self) -> np.ndarray:
        """Calculate the optical path through the crystal."""
        if self.get_normalized_q_bragg_laboratory_frame is None:
            raise ValueError("q_bragg_laboratory_frame is None")
        bulk = pu.find_bulk(
            amp=abs(self.data),
            support_threshold=self.parameters["threshold_unwrap_refraction"],
            method=self.parameters["optical_path_method"],
            debugging=self.parameters["debug"],
            cmap=self.parameters["colormap"].cmap,
        )

        kin = self.setup.incident_wavevector
        kout = self.setup.exit_wavevector
        # kin and kout were calculated in the laboratory frame,
        # but after the geometric transformation of the crystal, this
        # latter is always in the crystal frame (for simpler strain calculation).
        # We need to transform kin and kout back
        # into the crystal frame (also, xrayutilities output is in crystal frame)
        rotated_kin = util.rotate_vector(
            vectors=(kin[2], kin[1], kin[0]),
            axis_to_align=AXIS_TO_ARRAY[self.parameters["ref_axis_q"]],
            reference_axis=self.get_normalized_q_bragg_laboratory_frame[::-1],
        )
        if isinstance(rotated_kin, tuple):
            raise TypeError(f"rotated_kin should be a ndarray, got {type(rotated_kin)}")
        rotated_kout = util.rotate_vector(
            vectors=(kout[2], kout[1], kout[0]),
            axis_to_align=AXIS_TO_ARRAY[self.parameters["ref_axis_q"]],
            reference_axis=self.get_normalized_q_bragg_laboratory_frame[::-1],
        )
        if isinstance(rotated_kout, tuple):
            raise TypeError(
                f"rotated_kin should be a ndarray, got {type(rotated_kout)}"
            )
        # calculate the optical path of the incoming wavevector
        path_in = pu.get_opticalpath(
            support=bulk,
            direction="in",
            k=rotated_kin,
            debugging=self.parameters["debug"],
            cmap=self.parameters["colormap"].cmap,
        )  # path_in already in nm

        # calculate the optical path of the outgoing wavevector
        path_out = pu.get_opticalpath(
            support=bulk,
            direction="out",
            k=rotated_kout,
            debugging=self.parameters["debug"],
            cmap=self.parameters["colormap"].cmap,
        )  # path_our already in nm

        return np.asarray(path_in + path_out)

    def get_phase_manipulator(self):
        return PhaseManipulator(
            data=self.data,
            parameters=self.parameters,
            original_shape=self.original_shape,
            wavelength=self.setup.wavelength,
            savedir=self.setup.detector.savedir,
            logger=self.logger,
        )

    @abstractmethod
    def get_q_bragg_crystal_frame(self) -> Optional[np.ndarray]:
        """Return the q vector of the Bragg peak in the crystal frame."""
        raise NotImplementedError

    def get_reconstrutions_path(self) -> Tuple[Any, ...]:
        if self.parameters["reconstruction_files"][self.scan_index] is not None:
            file_path = self.parameters["reconstruction_files"][self.scan_index]
            if isinstance(file_path, str):
                file_path = (file_path,)
            if isinstance(file_path, list):
                file_path = tuple(file_path)
        else:
            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.askopenfilenames(
                initialdir=(
                    self.setup.detector.scandir
                    if self.parameters["data_dir"] is None
                    else self.setup.detector.datadir
                ),
                filetypes=[
                    ("HDF5", "*.h5"),
                    ("CXI", "*.cxi"),
                    ("NPZ", "*.npz"),
                    ("NPY", "*.npy"),
                ],
            )
        return tuple(file_path)

    def get_shape_during_phasing(
        self, user_defined_shape: Optional[List[int]]
    ) -> Tuple[int, ...]:
        input_shape = (
            user_defined_shape if user_defined_shape is not None else self.data.shape
        )
        self.logger.info(
            f"FFT size before accounting for phasing_binning: {input_shape}"
        )
        original_shape = tuple(
            int(input_shape[index] // binning)
            for index, binning in enumerate(self.setup.detector.binning)
        )
        self.logger.info(f"Binning used during phasing: {self.setup.detector.binning}")
        self.logger.info(f"Original data shape during phasing: {original_shape}")
        return original_shape

    def initialize_comment(self) -> Comment:
        return Comment(self.parameters.get("comment", ""))

    @abstractmethod
    def interpolate_into_crystal_frame(self):
        """
        Interpolate the direct space object into the crystal frame.

        The exact steps depend on which frame the data lies in.
        """

    def load_diffraction_data(
        self,
    ):
        return self.setup.loader.load_check_dataset(
            scan_number=self.parameters["scans"][self.scan_index],
            setup=self.setup,
            frames_pattern=self.parameters["frames_pattern"],
            bin_during_loading=False,
            flatfield=self.parameters["flatfield_file"],
            hotpixels=self.parameters["hotpixels_file"],
            background=self.parameters["background_file"],
            normalize=self.parameters["normalize_flux"],
        )

    def load_reconstruction_data(self, file_index: int) -> np.ndarray:
        return util.load_file(self.file_path[file_index])[0]

    def retrieve_bragg_peak(self) -> Dict[str, Any]:
        output = self.load_diffraction_data()

        return bu.find_bragg(
            array=output[0],
            binning=None,
            roi=self.setup.detector.roi,
            peak_method=self.parameters["centering_method"]["reciprocal_space"],
            tilt_values=self.setup.tilt_angles,
            savedir=self.setup.detector.savedir,
            user_defined_peak=self.parameters["bragg_peak"],
            logger=self.logger,
            plot_fit=True,
        )

    def save_modulus_phase(self, filename: str) -> None:
        np.savez_compressed(
            filename,
            amp=abs(self.data),
            phase=np.angle(self.data),
        )

    def save_support(self, filename: str, modulus_threshold: float = 0.1) -> None:
        support = np.zeros(self.data.shape)
        support[abs(self.data) / abs(self.data).max() > modulus_threshold] = 1
        np.savez_compressed(filename, obj=support)

    def save_to_vti(self, filename: str) -> None:
        voxel_z, voxel_y, voxel_x = self.setup.voxel_sizes_detector(
            array_shape=self.original_shape,
            tilt_angle=(
                self.parameters["tilt_angle"]
                * self.setup.detector.preprocessing_binning[0]
                * self.setup.detector.binning[0]
            ),
            pixel_x=self.setup.detector.pixelsize_x,
            pixel_y=self.setup.detector.pixelsize_y,
            verbose=True,
        )
        # save raw amp & phase to VTK (in VTK, x is downstream, y vertical, z inboard)
        gu.save_to_vti(
            filename=filename,
            voxel_size=(voxel_z, voxel_y, voxel_x),
            tuple_array=(abs(self.data), np.angle(self.data)),
            tuple_fieldnames=("amp", "phase"),
            amplitude_threshold=0.01,
        )

    def update_data(self, modulus: np.ndarray, phase: np.ndarray) -> None:
        # here the phase is again wrapped in [-pi pi[
        self.data = modulus * np.exp(1j * phase)

    def update_detector_angles(self, bragg_peak_position: List[int]) -> None:
        self.setup.correct_detector_angles(bragg_peak_position=bragg_peak_position)
        self.update_parameters(
            {
                "inplane_angle": self.setup.inplane_angle,
                "outofplane_angle": self.setup.outofplane_angle,
            }
        )

    def update_parameters(self, dictionary: Dict[str, Any]) -> None:
        self.parameters.update(dictionary)


class DetectorFrameLinearization(Analysis):
    """
    Analysis workflow for interpolation based on the linearization matrix.

    The data before interpolation is in the detector frame
    (axis 0 = rocking angle, axis 1 detector Y, axis 2 detector X).
    The data after interpolation is in the pseudo-crystal frame (only 'ref_axis_q'is
    an axis of the crystal frame, there is an unknown inplane rotation around it).
    """

    def get_q_bragg_crystal_frame(self) -> Optional[np.ndarray]:
        if self.get_q_bragg_laboratory_frame is None:
            return None
        return np.asarray(
            util.rotate_vector(
                vectors=self.get_q_bragg_laboratory_frame[::-1],
                axis_to_align=AXIS_TO_ARRAY[self.parameters["ref_axis_q"]],
                reference_axis=self.get_q_bragg_laboratory_frame[::-1],
            )
        )

    def interpolate_into_crystal_frame(self) -> None:
        """Interpolate the data in the pseudo crystal frame."""
        original_shape = self.original_shape
        if len(original_shape) != 3:
            raise NotImplementedError("Can only process 3D arrays")
        if self.get_normalized_q_bragg_laboratory_frame is None:
            raise ValueError("q_bragg_laboratory_frame is None")
        obj_ortho, voxel_sizes, transfer_matrix = self.setup.ortho_directspace(
            arrays=self.data,
            q_bragg=np.array(self.get_normalized_q_bragg_laboratory_frame[::-1]),
            initial_shape=original_shape,  # type: ignore
            voxel_size=self.parameters["fix_voxel"],
            reference_axis=AXIS_TO_ARRAY[self.parameters["ref_axis_q"]],
            fill_value=(0,),
            debugging=True,
            title="modulus",
            cmap=self.parameters["colormap"].cmap,
        )
        if not isinstance(obj_ortho, np.ndarray):
            raise TypeError(
                "Expecting a single interpolated array, " f"got {type(obj_ortho)}"
            )
        self.data = obj_ortho
        self.voxel_sizes = list(voxel_sizes)
        self.update_parameters(
            {"transformation_matrix": transfer_matrix, "is_orthogonal": True}
        )


class OrthogonalFrame(Analysis):
    """Analysis workflow for data already in an orthogonal frame."""

    @property
    def q_values(self) -> Optional[List[np.ndarray]]:
        """List of three 1D arrays [qx, qz, qy]."""
        return self._q_values

    @q_values.setter
    def q_values(self, value: Optional[Dict[str, np.ndarray]]) -> None:
        if value is None:
            self._q_values = None
        else:
            self._q_values = [value["qx"], value["qz"], value["qy"]]

    @property
    def user_defined_voxel_size(self):
        return self.parameters["fix_voxel"]

    def calculate_voxel_sizes(self) -> List[float]:
        """Calculate the direct space voxel sizes based on loaded q values."""
        self.q_values = self.load_q_values()
        if self.q_values is None:
            raise ValueError("q_values is None")
        qx = self.q_values[0]
        qz = self.q_values[1]
        qy = self.q_values[2]
        dy_real = (
            2 * np.pi / abs(qz.max() - qz.min()) / 10
        )  # in nm qz=y in nexus convention
        dx_real = (
            2 * np.pi / abs(qy.max() - qy.min()) / 10
        )  # in nm qy=x in nexus convention
        dz_real = (
            2 * np.pi / abs(qx.max() - qx.min()) / 10
        )  # in nm qx=z in nexus convention
        self.logger.info(
            f"direct space voxel size from q values: ({dz_real:.2f} nm,"
            f" {dy_real:.2f} nm, {dx_real:.2f} nm)"
        )
        return [dz_real, dy_real, dx_real]

    def get_q_bragg_crystal_frame(self) -> Optional[np.ndarray]:
        if (
            self.is_data_in_laboratory_frame
            and self.get_q_bragg_laboratory_frame is not None
        ):
            return np.asarray(
                util.rotate_vector(
                    vectors=self.get_q_bragg_laboratory_frame[::-1],
                    axis_to_align=AXIS_TO_ARRAY[self.parameters["ref_axis_q"]],
                    reference_axis=self.get_q_bragg_laboratory_frame[::-1],
                )
            )
        if self.q_values is None:
            self.q_values = self.load_q_values()

        # get the data padded to the phasing shape
        data = util.crop_pad(
            array=self.data,
            output_shape=self.original_shape,
            cmap=self.parameters["colormap"].cmap,
        )
        # go back to reciprocal space
        data = abs(fftshift(fftn(data)))
        # find the Bragg peak location in reciprocal space
        indices = np.asarray(
            bu.find_bragg(
                array=data,
                binning=None,
                roi=None,
                peak_method=self.parameters["centering_method"]["reciprocal_space"],
                tilt_values=None,
                savedir=self.setup.detector.savedir,
                user_defined_peak=self.parameters["bragg_peak"],
                logger=self.logger,
                plot_fit=False,
            )["bragg_peak"]
        )
        return np.array(
            [
                self.q_values[0][indices[0]],
                self.q_values[1][indices[1]],
                self.q_values[2][indices[2]],
            ]
        )

    def interpolate_into_crystal_frame(self) -> None:
        """Regrid and rotate the data if necessary."""
        self.update_parameters({"is_orthogonal": True})
        self.voxel_sizes = self.calculate_voxel_sizes()
        if self.user_defined_voxel_size:
            self.regrid(self.user_defined_voxel_size)
        if self.is_data_in_laboratory_frame:
            self.rotate_into_crystal_frame()

    def rotate_into_crystal_frame(self) -> None:
        if self.get_normalized_q_bragg_laboratory_frame is None:
            raise ValueError("q_bragg_laboratory_frame is None")
        self.logger.info(
            "Rotating the object in the crystal frame " "for the strain calculation"
        )
        amp, phase = util.rotate_crystal(
            arrays=(abs(self.data), np.angle(self.data)),
            is_orthogonal=True,
            reciprocal_space=False,
            voxel_size=self.voxel_sizes,
            debugging=(True, False),
            axis_to_align=self.get_normalized_q_bragg_laboratory_frame[::-1],
            reference_axis=AXIS_TO_ARRAY[self.parameters["ref_axis_q"]],
            title=("amp", "phase"),
            cmap=self.parameters["colormap"].cmap,
        )
        self.data = amp * np.exp(1j * phase)

    def load_q_values(self) -> Any:
        try:
            self.logger.info("Select the file containing QxQzQy")
            file_path = filedialog.askopenfilename(
                title="Select the file containing QxQzQy",
                initialdir=self.setup.detector.savedir,
                filetypes=[("NPZ", "*.npz")],
            )
            return np.load(file_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                "q values not provided, the voxel size cannot be calculated"
            )

    def regrid(self, new_voxelsizes: List[float]) -> None:
        """Regrid the data based on user-defined voxel sizes."""
        self.logger.info(
            f"Direct space pixel size for the interpolation: {new_voxelsizes} (nm)"
        )
        self.logger.info("Interpolating...\n")
        self.data = pu.regrid(
            array=self.data,
            old_voxelsize=self.voxel_sizes,
            new_voxelsize=new_voxelsizes,
        )
        self.voxel_sizes = new_voxelsizes


class InterpolatedCrystal:
    """Process the strain, modulus and phase for visualization."""

    def __init__(
        self,
        modulus: np.ndarray,
        phase: np.ndarray,
        strain: np.ndarray,
        planar_distance: float,
        parameters: Dict[str, Any],
        voxel_sizes: List[float],
        **kwargs,
    ):
        self.modulus = modulus
        self.phase = phase
        self.strain = strain
        self.planar_distance = planar_distance  # in nm
        self.parameters = parameters
        self.voxel_sizes = voxel_sizes
        self.logger = kwargs.get("logger", module_logger)

        self.q_bragg_in_saving_frame: Optional[np.ndarray] = None
        self.estimated_crystal_volume: Optional[int] = None

    @property
    def norm_of_q(self) -> float:
        """Calculate the norm of q in 1/A (planar distance in nm)."""
        return 2 * np.pi / (10 * self.planar_distance)

    def crop_pad_arrays(self):
        output_size = self.parameters.get("output_size")
        if output_size is not None:
            cmap = self.parameters["colormap"].cmap
            self.modulus, self.phase, self.strain = (
                util.crop_pad(array=array, output_shape=output_size, cmap=cmap)
                for array in [self.modulus, self.phase, self.strain]
            )

    def estimate_crystal_volume(self) -> None:
        support = np.copy(self.modulus / self.modulus.max())
        support[support < self.parameters["isosurface_strain"]] = 0
        support[np.nonzero(support)] = 1
        self.estimated_crystal_volume = support.sum() * reduce(
            lambda x, y: x * y, self.voxel_sizes
        )  # in nm3

    def find_phase_extent_within_crystal(self) -> float:
        support = self.get_bulk()
        return float(
            self.phase[np.nonzero(support)].max()
            - self.phase[np.nonzero(support)].min()
        )

    def find_max_phase(self, filename: str) -> None:
        piz, piy, pix = np.unravel_index(self.phase.argmax(), self.phase.shape)
        max_phase = self.phase[np.nonzero(self.get_bulk())].max()
        self.logger.info(
            f"phase.max() = {max_phase:.2f} " f"at voxel ({piz}, {piy}, {pix})"
        )
        # plot the slice at the maximum phase
        fig = gu.combined_plots(
            (self.phase[piz, :, :], self.phase[:, piy, :], self.phase[:, :, pix]),
            tuple_sum_frames=False,
            tuple_sum_axis=0,
            tuple_width_v=None,
            tuple_width_h=None,
            tuple_colorbar=True,
            tuple_vmin=np.nan,
            tuple_vmax=np.nan,
            tuple_title=(
                "phase at max in xy",
                "phase at max in xz",
                "phase at max in yz",
            ),
            tuple_scale="linear",
            cmap=self.parameters["colormap"].cmap,
            is_orthogonal=self.parameters["is_orthogonal"],
            reciprocal_space=False,
        )
        plt.pause(0.1)
        if self.parameters["save"]:
            fig.savefig(filename)
        plt.close(fig)

    def fit_linecuts_through_crystal_edges(self, filename: str) -> None:
        linecut.fit_linecut(
            array=self.modulus,
            fit_derivative=True,
            filename=filename,
            voxel_sizes=self.voxel_sizes,
            label="modulus",
            logger=self.logger,
        )

    def flatten_sample_circles(self, setup: "Setup") -> None:
        """
        Send all sample circles to zero degrees.

        Arrays are rotated such that all circles of the sample stage are at their zero
        position.
        """
        if setup.q_laboratory is None:
            raise ValueError("setup.q_laboratory is None")
        self.logger.info("Sending sample stage circles to 0")
        (
            self.modulus,
            self.phase,
            self.strain,
        ), self.q_bragg_in_saving_frame = setup.beamline.flatten_sample(
            arrays=(self.modulus, self.phase, self.strain),
            voxel_size=self.voxel_sizes,
            q_bragg=setup.q_laboratory / float(np.linalg.norm(setup.q_laboratory)),
            is_orthogonal=self.parameters["is_orthogonal"],
            reciprocal_space=False,
            rocking_angle=setup.rocking_angle,
            debugging=(True, False, False),
            title=("amp", "phase", "strain"),
            cmap=self.parameters["colormap"].cmap,
        )

    def get_bulk(self, method: str = "threshold") -> np.ndarray:
        """Calculate the support representing the crystal without the surface layer."""
        return pu.find_bulk(
            amp=self.modulus,
            support_threshold=self.parameters["isosurface_strain"],
            method=method,
            cmap=self.parameters["colormap"].cmap,
        )

    def normalize_modulus(self) -> None:
        self.modulus = self.modulus / self.modulus.max()

    def rescale_q(self) -> None:
        """Multiply the normalized diffusion vector by its original norm."""
        if self.q_bragg_in_saving_frame is not None:
            self.q_bragg_in_saving_frame = self.q_bragg_in_saving_frame * self.norm_of_q

    def rotate_crystal(
        self, axis_to_align: List[float], reference_axis: np.ndarray
    ) -> None:
        """Rotate the modulus, phase and strain along the reference axis."""
        self.modulus, self.phase, self.strain = util.rotate_crystal(
            arrays=(self.modulus, self.phase, self.strain),
            axis_to_align=axis_to_align,
            voxel_size=self.voxel_sizes,
            is_orthogonal=self.parameters["is_orthogonal"],
            reciprocal_space=False,
            reference_axis=reference_axis,
            debugging=(True, False, False),
            title=("modulus", "phase", "strain"),
            cmap=self.parameters["colormap"].cmap,
        )
        self.q_bragg_in_saving_frame = reference_axis

    def rotate_vector_to_saving_frame(
        self,
        vector: np.ndarray,
        axis_to_align: np.ndarray,
        reference_axis: np.ndarray,
    ) -> None:
        """Calculate the diffusion vector in the crystal frame."""
        self.q_bragg_in_saving_frame = np.asarray(
            util.rotate_vector(
                vectors=vector,
                axis_to_align=axis_to_align,
                reference_axis=reference_axis,
            )
        )

    def set_q_bragg_to_saving_frame(self, analysis: Analysis) -> None:
        if (
            analysis.is_data_in_laboratory_frame
            and analysis.get_normalized_q_bragg_laboratory_frame is not None
        ):
            self.rotate_vector_to_saving_frame(
                vector=analysis.get_normalized_q_bragg_laboratory_frame[::-1],
                axis_to_align=AXIS_TO_ARRAY[self.parameters["ref_axis_q"]],
                reference_axis=analysis.get_normalized_q_bragg_laboratory_frame[::-1],
            )
        else:
            q_bragg = analysis.get_q_bragg_crystal_frame()
            if q_bragg is not None:
                self.q_bragg_in_saving_frame = q_bragg / np.linalg.norm(q_bragg)
            else:
                raise ValueError("q_bragg_crystal_frame is undefined")
        self.rescale_q()

    def save_results_as_npz(self, filename: str, setup: "Setup") -> None:
        np.savez_compressed(
            filename,
            amp=self.modulus,
            phase=self.phase,
            bulk=self.get_bulk(),
            strain=self.strain,
            q_bragg=(
                self.q_bragg_in_saving_frame
                if self.q_bragg_in_saving_frame is not None
                else np.nan
            ),
            voxel_sizes=self.voxel_sizes,
            detector=str(yaml.dump(setup.detector.params)),
            setup=str(yaml.dump(setup.params)),
            params=str(yaml.dump(self.parameters)),
        )

    def save_results_as_h5(self, filename: str, setup: "Setup") -> None:
        with h5py.File(
            filename,
            "w",
        ) as hf:
            out = hf.create_group("output")
            par = hf.create_group("params")
            out.create_dataset("amp", data=self.modulus)
            out.create_dataset("bulk", data=self.get_bulk())
            out.create_dataset("phase", data=self.phase)
            out.create_dataset("strain", data=self.strain)
            out.create_dataset("q_bragg", data=self.q_bragg_in_saving_frame)
            out.create_dataset("voxel_sizes", data=self.voxel_sizes)
            par.create_dataset("detector", data=str(setup.detector.params))
            par.create_dataset("setup", data=str(setup.params))
            par.create_dataset("parameters", data=str(self.parameters))

    def save_results_as_vti(self, filename: str) -> None:
        gu.save_to_vti(
            filename=filename,
            voxel_size=self.voxel_sizes,
            tuple_array=(self.modulus, self.get_bulk(), self.phase, self.strain),
            tuple_fieldnames=(
                "amp",
                "bulk",
                "phase",
                "strain",
            ),
            amplitude_threshold=0.01,
        )

    def threshold_phase_strain(self):
        support = self.get_bulk()
        self.strain[support == 0] = np.nan
        self.phase[support == 0] = np.nan


class PhaseManipulator:
    """Process the phase of the data."""

    def __init__(
        self,
        data: np.ndarray,
        parameters: Dict[str, Any],
        original_shape: Tuple[int, ...],
        wavelength: float,
        save_directory: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.data = data
        self._phase, self._modulus = self.extract_phase_modulus()
        self.parameters = parameters
        self.original_shape = original_shape
        self.wavelength = wavelength
        self.save_directory = save_directory

        self._extent_phase: Optional[float] = None
        self._phase_ramp: Optional[List[float]] = None
        self.logger = kwargs.get("logger", module_logger)

    @property
    def extent_phase(self) -> float:
        return (
            self._extent_phase
            if self._extent_phase is not None
            else self.get_extent_phase()
        )

    @property
    def modulus(self) -> np.ndarray:
        return self._modulus

    @property
    def phase(self) -> np.ndarray:
        return self._phase

    @property
    def phase_ramp(self) -> Optional[List[float]]:
        return self._phase_ramp

    @property
    def save_directory(self) -> Optional[str]:
        return self._save_directory

    @save_directory.setter
    def save_directory(self, val: Optional[str]) -> None:
        if isinstance(val, str) and not val.endswith("/"):
            val += "/"
        self._save_directory = val

    def add_ramp(self, sign: int = +1) -> None:
        """Add a linear ramp to the phase."""
        if self.phase_ramp is None:
            raise ValueError("'phase_ramp' is None, can't add the phase ramp")
        gridz, gridy, gridx = np.meshgrid(
            np.arange(0, self.data.shape[0], 1),
            np.arange(0, self.data.shape[1], 1),
            np.arange(0, self.data.shape[2], 1),
            indexing="ij",
        )

        self._phase = (
            self.phase
            + gridz * sign * self.phase_ramp[0]
            + gridy * sign * self.phase_ramp[1]
            + gridx * sign * self.phase_ramp[2]
        )

    def apodize(self) -> None:
        """Apply a filtering window to the phase."""
        self._modulus, self._phase = pu.apodize(
            amp=self.modulus,
            phase=self.phase,
            initial_shape=self.original_shape,
            window_type=self.parameters["apodization_window"],
            sigma=self.parameters["apodization_sigma"],
            mu=self.parameters["apodization_mu"],
            alpha=self.parameters["apodization_alpha"],
            is_orthogonal=self.parameters["is_orthogonal"],
            debugging=True,
            cmap=self.parameters["colormap"].cmap,
            logger=self.logger,
        )

    def average_phase(self) -> None:
        """Apply an averaging window to the phase."""
        bulk = pu.find_bulk(
            amp=self.modulus,
            support_threshold=self.parameters["isosurface_strain"],
            method="threshold",
            cmap=self.parameters["colormap"].cmap,
            logger=self.logger,
        )
        # the phase should be averaged only in the support defined by the isosurface
        self._phase = pu.mean_filter(
            array=self.phase,
            support=bulk,
            half_width=self.parameters["half_width_avg_phase"],
            cmap=self.parameters["colormap"].cmap,
            logger=self.logger,
        )

    def calculate_strain(
        self, planar_distance: float, voxel_sizes: List[float]
    ) -> InterpolatedCrystal:
        self.logger.info(
            f"Calculation of the strain along {self.parameters['ref_axis_q']}"
        )
        strain = pu.get_strain(
            phase=self.phase,
            planar_distance=planar_distance,
            voxel_size=voxel_sizes,
            reference_axis=self.parameters["ref_axis_q"],
            extent_phase=self.extent_phase,
            method=self.parameters["strain_method"],
            debugging=self.parameters["debug"],
            cmap=self.parameters["colormap"].cmap,
        )
        return InterpolatedCrystal(
            modulus=self.modulus,
            phase=self.phase,
            strain=strain,
            planar_distance=planar_distance,
            parameters=self.parameters,
            voxel_sizes=voxel_sizes,
            logger=self.logger,
        )

    def center_phase(self) -> None:
        """Wrap the phase around its mean."""
        if self.extent_phase is None:
            raise ValueError("extent_phase is None")
        self._phase = util.wrap(
            self.phase,
            start_angle=-self.extent_phase / 2,
            range_angle=self.extent_phase,
        )

    def compensate_refraction(self, optical_path: np.ndarray) -> None:
        """Compensate the phase shift due to refraction through the crystal medium."""
        phase_correction = (
            2
            * np.pi
            / (1e9 * self.wavelength)
            * self.parameters["dispersion"]
            * optical_path
        )
        self._phase = self.phase + phase_correction

        gu.multislices_plot(
            np.multiply(phase_correction, self.modulus),
            sum_frames=False,
            plot_colorbar=True,
            vmin=0,
            vmax=np.nan,
            title="Refraction correction on the support",
            is_orthogonal=True,
            reciprocal_space=False,
            cmap=self.parameters["colormap"].cmap,
        )

    def extract_phase_modulus(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the phase and the modulus out of the data."""
        return np.angle(self.data), abs(self.data)

    def get_extent_phase(self) -> float:
        _, extent_phase = pu.unwrap(
            self.data,
            support_threshold=self.parameters["threshold_unwrap_refraction"],
            debugging=self.parameters["debug"],
            reciprocal_space=False,
            is_orthogonal=self.parameters["is_orthogonal"],
            cmap=self.parameters["colormap"].cmap,
        )
        return extent_phase

    def invert_phase(self) -> None:
        self._phase = -1 * self.phase

    def plot_phase(self, plot_title: str = "", save_plot: bool = False) -> None:
        fig, _, _ = gu.multislices_plot(
            self.phase,
            plot_colorbar=True,
            title=plot_title,
            reciprocal_space=False,
            is_orthogonal=self.parameters["is_orthogonal"],
            cmap=self.parameters["colormap"].cmap,
        )
        if save_plot and self.save_directory is not None:
            fig.savefig(self.save_directory + plot_title + ".png")

    def remove_offset(self) -> None:
        """Remove a phase offset to the phase."""
        support = np.zeros(self.data.shape)
        support[
            self.modulus > self.parameters["isosurface_strain"] * self.modulus.max()
        ] = 1
        self._phase = pu.remove_offset(
            array=self.phase,
            support=support,
            offset_method=self.parameters["offset_method"],
            phase_offset=self.parameters["phase_offset"],
            offset_origin=self.parameters["phase_offset_origin"],
            title="Phase",
            debugging=self.parameters["debug"],
            cmap=self.parameters["colormap"].cmap,
            reciprocal_space=False,
            is_orthogonal=self.parameters["is_orthogonal"],
        )

    def remove_ramp(self) -> None:
        """Remove the linear trend in the phase based on a modulus support."""
        self._modulus, self._phase, *self._phase_ramp = pu.remove_ramp(
            amp=self.modulus,
            phase=self.phase,
            initial_shape=self.original_shape,
            method="gradient",
            amplitude_threshold=self.parameters["isosurface_strain"],
            threshold_gradient=self.parameters["threshold_gradient"],
            cmap=self.parameters["colormap"].cmap,
            logger=self.logger,
        )

    def unwrap_phase(self) -> None:
        self._phase, self._extent_phase = pu.unwrap(
            self.data,
            support_threshold=self.parameters["threshold_unwrap_refraction"],
            debugging=self.parameters["debug"],
            reciprocal_space=False,
            is_orthogonal=self.parameters["is_orthogonal"],
            cmap=self.parameters["colormap"].cmap,
        )

        self.logger.info(
            "Extent of the phase over an extended support (ceil(phase range)) ~ "
            f"{int(self.extent_phase)} (rad)",
        )


def define_analysis_type(data_frame: str) -> str:
    """Define the correct analysis type depending on the parameters."""
    if data_frame == "detector":
        return "linearization"
    return "orthogonal"


def create_analysis(
    scan_index: int,
    parameters: Dict[str, Any],
    setup: Setup,
    **kwargs,
) -> Analysis:
    """Create the correct analysis class depending on the parameters."""
    name = define_analysis_type(
        data_frame=parameters["data_frame"],
    )
    if name == "linearization":
        return DetectorFrameLinearization(
            scan_index=scan_index,
            parameters=parameters,
            setup=setup,
            **kwargs,
        )
    if name == "orthogonal":
        return OrthogonalFrame(
            scan_index=scan_index,
            parameters=parameters,
            setup=setup,
            **kwargs,
        )
    raise ValueError(f"Analysis {name} not supported")
