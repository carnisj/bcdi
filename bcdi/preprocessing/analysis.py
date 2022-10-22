# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr
"""Implementation of preprocessing analysis classes."""

import logging
from operator import mul
import os
import tkinter as tk
from abc import ABC, abstractmethod
from functools import reduce
from tkinter import filedialog
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
from scipy.io import savemat
import yaml
from matplotlib import pyplot as plt
import xrayutilities as xu

import bcdi.graph.graph_utils as gu
import bcdi.graph.linecut as linecut
import bcdi.preprocessing.bcdi_utils as bu
import bcdi.utils.image_registration as reg
import bcdi.utils.utilities as util
import bcdi.utils.validation as valid
from bcdi.experiment.setup import Setup
from bcdi.utils.constants import AXIS_TO_ARRAY
from bcdi.utils.text import Comment

module_logger = logging.getLogger(__name__)


class Analysis(ABC):
    """Base class for the pre-processing analysis workflow."""

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

        self.data_loader = create_data_loader(
            scan_index=self.scan_index,
            parameters=self.parameters,
            setup=self.setup,
            logger=self.logger,
        )
        self.comment = self.initialize_comment()
        if parameters["normalize_flux"]:
            self.comment.concatenate("norm")
        self.q_bragg: Optional[np.ndarray] = None
        self.interpolation_needed = False
        self.starting_frame = [0, 0, 0]
        self.is_orthogonal = self.parameters["reload_orthogonal"]
        (
            self.data,
            self.mask,
            self.frames_logical,
            self.monitor,
        ) = self.data_loader.load_dataset()

    @property
    def detector_angles_correction_needed(self) -> bool:
        return self.parameters["rocking_angle"] != "energy" and (
            not self.parameters["outofplane_angle"]
            or not self.parameters["inplane_angle"]
        )

    @property
    def is_raw_data_available(self) -> bool:
        return not self.parameters["reload_orthogonal"]

    @property
    def planar_distance(self) -> Optional[float]:
        if self.q_norm is not None and self.q_norm != 0:
            return 2 * np.pi / self.q_norm
        return None

    @property
    def q_norm(self) -> Optional[float]:
        if self.q_bragg is not None:
            return np.linalg.norm(self.q_bragg)
        return None

    @property
    def scan_nb(self) -> int:
        return self.parameters["scans"][self.scan_index]

    @abstractmethod
    def calculate_q_bragg(self, **kwargs) -> None:
        raise NotImplementedError

    def center_fft(self) -> None:
        (
            self.data,
            self.mask,
            pad_width,
            self.data_loader.q_values,
            self.frames_logical,
        ) = bu.center_fft(
            data=self.data,
            mask=self.mask,
            detector=self.setup.detector,
            frames_logical=self.frames_logical,
            centering=self.parameters["centering_method"],
            fft_option=self.parameters["center_fft"],
            pad_size=self.parameters["pad_size"],
            fix_bragg=self.parameters["bragg_peak"],
            q_values=self.data_loader.q_values,
            logger=self.logger,
        )
        self.starting_frame = [pad_width[0], pad_width[2], pad_width[4]]
        # no need to check padded frames during masking
        self.logger.info(f"Pad width: {pad_width}")
        self.logger.info(f"Data size after cropping / padding: {self.data.shape}")

    def get_masked_data(self) -> np.ndarray:
        data = np.copy(self.data)
        data[self.mask == 1] = 0
        return data

    def interpolate_data(self) -> None:
        raise NotImplementedError

    def initialize_comment(self) -> Comment:
        return Comment(self.parameters.get("comment", ""))

    def mask_data(self) -> None:
        self.data[np.nonzero(self.mask)] = 0

    def mask_zero_events(self) -> None:
        nz, ny, nx = self.data.shape
        temp_mask = np.zeros((ny, nx))
        temp_mask[np.sum(self.data, axis=0) == 0] = 1
        self.mask[np.repeat(temp_mask[np.newaxis, :, :], repeats=nz, axis=0) == 1] = 1

    def retrieve_bragg_peak(self) -> Dict[str, Any]:
        return bu.find_bragg(
            array=self.data,
            binning=self.setup.detector.current_binning,
            roi=self.setup.detector.roi,
            peak_method=self.parameters["centering_method"]["reciprocal_space"],
            tilt_values=self.setup.tilt_angles,
            savedir=self.setup.detector.savedir,
            logger=self.logger,
            plot_fit=True,
        )

    def save_data(self, filename: str, save_to_mat: bool) -> None:
        np.savez_compressed(filename, data=self.data)
        if save_to_mat:
            if self.data.ndim != 3:
                raise ValueError(
                    f"Only 3D ndarray supported, data is {self.data.ndim}D"
                )
            # save to .mat, the new order is x y z
            # (outboard, vertical up, downstream)
            savemat(
                filename + ".mat",
                {"data": np.moveaxis(self.data, [0, 1, 2], [-1, -2, -3])},
            )

    def save_mask(self, filename: str) -> None:
        np.savez_compressed(filename, hotpixels=self.mask.astype(int))

    def save_to_vti(self, filename: Optional[str]) -> None:
        qx, qz, qy = self.data_loader.q_values
        # save diffraction pattern to vti

        nqx, nqz, nqy = self.data.shape
        # in nexus z downstream, y vertical / in q z vertical, x downstream
        self.logger.info(
            f"(dqx, dqy, dqz) = ({qx[1] - qx[0]:2f}, {qy[1] - qy[0]:2f}, "
            f"{qz[1] - qz[0]:2f})"
        )
        # in nexus z downstream, y vertical / in q z vertical, x downstream
        qx0 = qx.min()
        dqx = (qx.max() - qx0) / nqx
        qy0 = qy.min()
        dqy = (qy.max() - qy0) / nqy
        qz0 = qz.min()
        dqz = (qz.max() - qz0) / nqz

        gu.save_to_vti(
            filename=filename,
            voxel_size=(dqx, dqz, dqy),
            tuple_array=self.data,
            tuple_fieldnames="int",
            origin=(qx0, qz0, qy0),
            logger=self.logger,
        )

    def show_array(self, array: np.ndarray, title: str, **kwargs) -> Any:
        fig, _, _ = gu.multislices_plot(
            array,
            sum_frames=True,
            scale="log",
            plot_colorbar=True,
            vmin=0,
            title=title,
            is_orthogonal=self.is_orthogonal,
            reciprocal_space=True,
            cmap=self.parameters["colormap"].cmap,
            **kwargs,
        )
        return fig

    def show_mask(self, title: str, filename: Optional[str]) -> None:
        fig = self.show_array(array=self.mask, title=title)
        if filename is not None:
            fig.savefig(filename)
        plt.close(fig)

    def show_masked_data(self, title: str, filename: Optional[str]):
        fig = self.show_array(array=self.get_masked_data(), title=title)
        if filename is not None:
            fig.savefig(filename)
        plt.close(fig)

    def show_masked_data_at_max(self, title: str, filename: Optional[str]):
        data = self.get_masked_data()
        fig = self.show_array(
            array=data, title=title, slice=np.unravel_index(data.argmax(), data.shape)
        )
        if filename is not None:
            fig.savefig(filename)
        plt.close(fig)

    def update_detector_angles(self, bragg_peak_position: List[int]) -> None:
        self.setup.correct_detector_angles(bragg_peak_position=bragg_peak_position)

    def update_mask(self, mask_file: str) -> None:
        config_mask, _ = util.load_file(mask_file)
        valid.valid_ndarray(config_mask, shape=self.data.shape)
        config_mask[np.nonzero(config_mask)] = 1
        self.mask = np.multiply(self.mask, config_mask.astype(self.mask.dtype))

    def update_parameters(self, dictionary: Dict[str, Any]) -> None:
        self.parameters.update(dictionary)

    def _calculate_q_bragg_orthogonal(self) -> None:
        if (
            self.data_loader.q_values
        ):  # find the Bragg peak position from the interpolated data
            interpolated_metadata = bu.find_bragg(
                array=self.data,
                binning=None,
                roi=None,
                peak_method=self.parameters["centering_method"]["reciprocal_space"],
                tilt_values=None,
                logger=self.logger,
                plot_fit=False,
            )
            self.q_bragg = np.array(
                [
                    self.data_loader.q_values[0][
                        interpolated_metadata["bragg_peak"][0]
                    ],
                    self.data_loader.q_values[1][
                        interpolated_metadata["bragg_peak"][1]
                    ],
                    self.data_loader.q_values[2][
                        interpolated_metadata["bragg_peak"][2]
                    ],
                ]
            )


class DetectorFrameAnalysis(Analysis):
    """Analysis worklow in the detector frame."""

    def calculate_q_bragg(self, **kwargs) -> None:
        self.q_bragg = self.setup.q_laboratory

    def interpolate_data(self) -> None:
        super().interpolate_data()


class LinearizationAnalysis(Analysis):
    """Analysis worklow for data to be interpolated using the linerization matrix."""

    def __init__(
        self,
        scan_index: int,
        parameters: Dict[str, Any],
        setup: Setup,
        **kwargs,
    ) -> None:
        super().__init__(
            scan_index=scan_index, parameters=parameters, setup=setup, **kwargs
        )
        # load the goniometer positions needed in the calculation
        # of the transformation matrix
        self.setup.read_logfile(scan_number=self.scan_nb)
        self.comment.concatenate("ortho_lin")
        self.interpolation_needed = True

    def calculate_q_bragg(self, **kwargs) -> None:
        self.q_bragg = self.setup.q_laboratory

    def interpolate_data(self) -> None:
        # for q values, the frame used is
        # (qx downstream, qy outboard, qz vertical up)
        # for reference_axis, the frame is z downstream, y vertical up,
        # x outboard but the order must be x,y,z
        (
            self.data,
            self.mask,
            self.data_loader.q_values,
            transfer_matrix,
        ) = bu.grid_bcdi_labframe(
            data=self.data,
            mask=self.mask,
            detector=self.setup.detector,
            setup=self.setup,
            align_q=self.parameters["align_q"],
            reference_axis=AXIS_TO_ARRAY[self.parameters["ref_axis_q"]],
            debugging=self.parameters["debug"],
            fill_value=(0, self.parameters["fill_value_mask"]),
            cmap=self.parameters["colormap"].cmap,
            logger=self.logger,
        )
        self.parameters["transformation_matrix"] = transfer_matrix

        nz, ny, nx = self.data.shape
        self.logger.info(
            "Data size after interpolation into an orthonormal frame:"
            f"{nz}, {ny}, {nx}"
        )
        self.is_orthogonal = True


class OrthogonalFrameAnalysis(Analysis):
    """Analysis worklow for reloaded data interpolated in an orthogonal frame."""

    def calculate_q_bragg(self, **kwargs) -> None:
        self._calculate_q_bragg_orthogonal()

    def interpolate_data(self) -> None:
        super().interpolate_data()


class XrayUtilitiesAnalysis(Analysis):
    """Analysis worklow for data to be interpolated using the xrayutilities."""

    def __init__(
        self,
        scan_index: int,
        parameters: Dict[str, Any],
        setup: Setup,
        **kwargs,
    ) -> None:
        super().__init__(
            scan_index=scan_index, parameters=parameters, setup=setup, **kwargs
        )
        self.comment.concatenate("ortho_xrutils")
        self.interpolation_needed = True

    def calculate_q_bragg(self, **kwargs) -> None:
        self._calculate_q_bragg_orthogonal()

    def interpolate_data(self) -> None:
        qconv, offsets = self.setup.init_qconversion()
        self.setup.detector.offsets = offsets
        hxrd = xu.experiment.HXRD(
            self.parameters["sample_inplane"],
            self.parameters["sample_outofplane"],
            en=self.setup.energy,
            qconv=qconv,
        )
        # the first 2 arguments in HXRD are the inplane reference direction
        # along the beam and surface normal of the sample

        # Update the direct beam vertical position,
        # take into account the roi and binning
        cch1 = (self.parameters["cch1"] - self.setup.detector.roi[0]) / (
            self.setup.detector.preprocessing_binning[1]
            * self.setup.detector.binning[1]
        )
        # Update the direct beam horizontal position,
        # take into account the roi and binning
        cch2 = (self.parameters["cch2"] - self.setup.detector.roi[2]) / (
            self.setup.detector.preprocessing_binning[2]
            * self.setup.detector.binning[2]
        )
        # number of pixels after taking into account the roi and binning
        nch1 = (self.setup.detector.roi[1] - self.setup.detector.roi[0]) // (
            self.setup.detector.preprocessing_binning[1]
            * self.setup.detector.binning[1]
        ) + (self.setup.detector.roi[1] - self.setup.detector.roi[0]) % (
            self.setup.detector.preprocessing_binning[1]
            * self.setup.detector.binning[1]
        )
        nch2 = (self.setup.detector.roi[3] - self.setup.detector.roi[2]) // (
            self.setup.detector.preprocessing_binning[2]
            * self.setup.detector.binning[2]
        ) + (self.setup.detector.roi[3] - self.setup.detector.roi[2]) % (
            self.setup.detector.preprocessing_binning[2]
            * self.setup.detector.binning[2]
        )
        # detector init_area method, pixel sizes are the binned ones
        hxrd.Ang2Q.init_area(
            self.setup.detector_ver_xrutil,
            self.setup.detector_hor_xrutil,
            cch1=cch1,
            cch2=cch2,
            Nch1=nch1,
            Nch2=nch2,
            pwidth1=self.setup.detector.pixelsize_y,
            pwidth2=self.setup.detector.pixelsize_x,
            distance=self.setup.distance,
            detrot=self.parameters["detrot"],
            tiltazimuth=self.parameters["tiltazimuth"],
            tilt=self.parameters["tilt_detector"],
        )
        # the first two arguments in init_area are
        # the direction of the detector

        (
            self.data,
            self.mask,
            self.data_loader.q_values,
            self.frames_logical,
        ) = bu.grid_bcdi_xrayutil(
            data=self.data,
            mask=self.mask,
            scan_number=self.scan_nb,
            setup=self.setup,
            frames_logical=self.frames_logical,
            hxrd=hxrd,
            debugging=self.parameters["debug"],
            cmap=self.parameters["colormap"].cmap,
            logger=self.logger,
        )
        self.is_orthogonal = True


def define_analysis_type(
    reload_orthogonal: bool, use_rawdata: bool, interpolation_method: str
) -> str:
    """Define the correct analysis type depending on the parameters."""
    if reload_orthogonal:
        return "interpolated"
    if use_rawdata:
        return "detector_frame"
    return interpolation_method


def create_analysis(
    scan_index: int,
    parameters: Dict[str, Any],
    setup: Setup,
    **kwargs,
) -> Analysis:
    """Create the correct analysis class depending on the parameters."""
    name = define_analysis_type(
        reload_orthogonal=parameters["reload_orthogonal"],
        use_rawdata=parameters["use_rawdata"],
        interpolation_method=parameters["interpolation_method"],
    )
    if name == "detector_frame":
        return DetectorFrameAnalysis(
            scan_index=scan_index,
            parameters=parameters,
            setup=setup,
            **kwargs,
        )
    if name == "linearization":
        return LinearizationAnalysis(
            scan_index=scan_index,
            parameters=parameters,
            setup=setup,
            **kwargs,
        )
    if name == "xrayutilities":
        return XrayUtilitiesAnalysis(
            scan_index=scan_index,
            parameters=parameters,
            setup=setup,
            **kwargs,
        )
    if name == "interpolated":
        return OrthogonalFrameAnalysis(
            scan_index=scan_index,
            parameters=parameters,
            setup=setup,
            **kwargs,
        )
    raise ValueError(f"Analysis {name} not supported")


class PreprocessingLoader(ABC):
    def __init__(
        self, scan_index: int, parameters: Dict[str, Any], setup: "Setup", **kwargs
    ) -> None:
        self.scan_index = scan_index
        self.parameters = parameters
        self.setup = setup
        self.logger = kwargs.get("logger", module_logger)

        self.q_values: Optional[List[np.ndarray]] = None

    @property
    def scan_nb(self) -> int:
        return self.parameters["scans"][self.scan_index]

    @abstractmethod
    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError

    def reload(self) -> Tuple[np.ndarray, np.ndarray]:
        file_path = filedialog.askopenfilename(
            initialdir=self.setup.detector.scandir,
            title="Select data file",
            filetypes=[("NPZ", "*.npz")],
        )
        data = np.load(file_path)
        npz_key = data.files
        data = data[npz_key[0]]
        _, ny, nx = np.shape(data)

        # check that the ROI is correctly defined
        self.setup.detector.roi = self.parameters["roi_detector"] or [0, ny, 0, nx]
        self.logger.info(f"Detector ROI: {self.setup.detector.roi}")
        # update savedir to save the data in the same directory as the reloaded data
        if not self.parameters["save_dir"]:
            self.setup.detector.savedir = os.path.dirname(file_path) + "/"
            self.logger.info(f"Updated saving directory: {self.setup.detector.savedir}")

        file_path = filedialog.askopenfilename(
            initialdir=os.path.dirname(file_path) + "/",
            title="Select mask file",
            filetypes=[("NPZ", "*.npz")],
        )
        mask = np.load(file_path)
        npz_key = mask.files
        mask = mask[npz_key[0]]
        return data, mask


class FirstDataLoading(PreprocessingLoader):
    """Load a dataset for the first time, de facto in the detector frame."""

    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return bu.load_bcdi_data(
            scan_number=self.scan_nb,
            setup=self.setup,
            frames_pattern=self.parameters["frames_pattern"],
            bin_during_loading=self.parameters["bin_during_loading"],
            flatfield=util.load_flatfield(self.parameters["flatfield_file"]),
            hotpixels=util.load_hotpixels(self.parameters["hotpixels_file"]),
            background=util.load_background(self.parameters["background_file"]),
            normalize=self.parameters["normalize_flux"],
            debugging=self.parameters["debug"],
            photon_threshold=self.parameters["loading_threshold"],
            logger=self.logger,
        )


class ReloadingDetectorFrame(PreprocessingLoader):
    """Reload a dataset which is still in the detector frame."""

    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        data, mask = self.reload()
        return bu.reload_bcdi_data(
            data=data,
            mask=mask,
            scan_number=self.scan_nb,
            setup=self.setup,
            normalize=self.parameters["normalize_flux"],
            debugging=self.parameters["debug"],
            photon_threshold=self.parameters["loading_threshold"],
            logger=self.logger,
        )


class ReloadingOrthogonalFrame(PreprocessingLoader):
    """Reload a dataset already interpolated in an orthogonal frame."""

    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        data, mask = self.reload()
        q_values = self.load_q_values()

        self.parameters["normalize_flux"] = "skip"
        # we assume that normalization was already performed
        monitor = []  # we assume that normalization was already performed
        self.parameters["center_fft"] = "skip"
        # we assume that crop/pad/centering was already performed

        # bin data and mask if needed
        if any(val != 1 for val in self.setup.detector.binning):
            self.logger.info(
                f"Binning the reloaded orthogonal data by {self.setup.detector.binning}"
            )
            data = util.bin_data(
                data,
                binning=self.setup.detector.binning,
                debugging=False,
                cmap=self.parameters["colormap"].cmap,
                logger=self.logger,
            )
            mask = util.bin_data(
                mask,
                binning=self.setup.detector.binning,
                debugging=False,
                cmap=self.parameters["colormap"].cmap,
                logger=self.logger,
            )
            self.setup.detector.current_binning = list(
                map(
                    mul,
                    self.setup.detector.current_binning,
                    self.setup.detector.binning,
                )
            )
            mask[np.nonzero(mask)] = 1

            if q_values is not None:
                qx = q_values[0]
                qz = q_values[1]
                qy = q_values[2]
                numz, numy, numx = len(qx), len(qz), len(qy)
                qx = qx[
                    : numz
                    - (
                        numz % self.setup.detector.binning[0]
                    ) : self.setup.detector.binning[0]
                ]  # along z downstream
                qz = qz[
                    : numy
                    - (
                        numy % self.setup.detector.binning[1]
                    ) : self.setup.detector.binning[1]
                ]  # along y vertical
                qy = qy[
                    : numx
                    - (
                        numx % self.setup.detector.binning[2]
                    ) : self.setup.detector.binning[2]
                ]  # along x outboard
                self.q_values = [qx, qz, qy]
        return data, mask, np.ones(data.shape[0]), monitor

    def load_q_values(self) -> Optional[List[np.ndarray]]:
        try:
            file_path = filedialog.askopenfilename(
                initialdir=self.setup.detector.savedir,
                title="Select q values",
                filetypes=[("NPZ", "*.npz")],
            )
            reload_qvalues = np.load(file_path)
            return [
                reload_qvalues["qx"],
                reload_qvalues["qz"],
                reload_qvalues["qy"],
            ]
        except FileNotFoundError:
            return None


def create_data_loader(
    scan_index: int, parameters: Dict[str, Any], setup: "Setup", **kwargs
) -> PreprocessingLoader:
    if parameters["reload_previous"]:
        if parameters["reload_orthogonal"]:
            return ReloadingOrthogonalFrame(
                scan_index=scan_index, parameters=parameters, setup=setup, **kwargs
            )
        return ReloadingDetectorFrame(
            scan_index=scan_index, parameters=parameters, setup=setup, **kwargs
        )
    return FirstDataLoading(
        scan_index=scan_index, parameters=parameters, setup=setup, **kwargs
    )
