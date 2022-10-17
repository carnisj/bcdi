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
import yaml
from matplotlib import pyplot as plt

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

    @property
    def scan_nb(self) -> int:
        return self.parameters["scans"][self.scan_index]

    def initialize_comment(self) -> Comment:
        return Comment(self.parameters.get("comment", ""))


class DetectorFrameAnalysis(Analysis):
    """Analysis worklow in the detector frame."""


class LinearizationAnalysis(Analysis):
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


class XrayUtilitiesAnalysis(Analysis):
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


def define_analysis_type(use_rawdata: bool, interpolation_method: str) -> str:
    """Define the correct analysis type depending on the parameters."""
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
    if name == "orthogonal":
        return XrayUtilitiesAnalysis(
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


class ReloadingOrthogonal(PreprocessingLoader):
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
            return ReloadingOrthogonal(
                scan_index=scan_index, parameters=parameters, setup=setup, **kwargs
            )
        return ReloadingDetectorFrame(
            scan_index=scan_index, parameters=parameters, setup=setup, **kwargs
        )
    return FirstDataLoading(
        scan_index=scan_index, parameters=parameters, setup=setup, **kwargs
    )
