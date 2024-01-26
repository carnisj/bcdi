# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr
"""Implementation of preprocessing analysis classes."""
# mypy: ignore-errors
import logging
import os
from abc import ABC, abstractmethod
from operator import mul
from tkinter import filedialog
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import scipy.signal  # for medfilt2d
import xrayutilities as xu
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backend_bases import KeyEvent, MouseEvent
from matplotlib.figure import Figure
from matplotlib.text import Text
from scipy.io import savemat
from scipy.ndimage import center_of_mass

import bcdi.graph.graph_utils as gu
import bcdi.preprocessing.bcdi_utils as bu
import bcdi.utils.utilities as util
import bcdi.utils.validation as valid
from bcdi.constants import AXIS_TO_ARRAY
from bcdi.experiment.setup import Setup
from bcdi.utils.text import Comment

module_logger = logging.getLogger(__name__)


class InteractiveMasker:
    """Mask interactively aliens (slice by slice) and hotpixels in a 3D CDI dataset."""

    def __init__(
        self,
        data: np.ndarray,
        mask: np.ndarray,
        parameters: Dict[str, Any],
        starting_frame: List[int],
    ) -> None:
        self.original_data: np.ndarray = data
        self.original_mask: np.ndarray = mask
        self.parameters: Dict[str, Any] = parameters
        self.starting_frame: List[int] = starting_frame

        self._data: Optional[np.ndarray] = None
        self._mask: Optional[np.ndarray] = None
        self._updated_mask: Optional[np.ndarray] = None
        self.xy: List[List[int]] = []
        self.flag_pause: bool = False
        self.previous_axis: Optional[int] = None
        self.width: int = 5
        self.max_colorbar: int = 5
        self.fig_mask: Optional[Figure] = None
        self.ax0: Optional[Axes] = None
        self.ax1: Optional[Axes] = None
        self.ax2: Optional[Axes] = None
        self.ax3: Optional[Axes] = None
        self.info_text: Optional[Text] = None
        self.frame_index: Optional[List[int]] = None

    @property
    def mask(self) -> Optional[np.ndarray]:
        return self._mask

    def on_click(self, event: MouseEvent) -> None:
        """
        Interact with a plot, return the position of clicked pixel.

        If flag_pause==1 or if the mouse is out of plot axes, it will not register
        the click.

        :param event: mouse click event
        """
        if not event.inaxes:
            return
        if not self.flag_pause:
            if (self.previous_axis == event.inaxes) or (
                self.previous_axis is None
            ):  # collect points
                _x, _y = int(np.rint(event.xdata)), int(np.rint(event.ydata))
                self.xy.append([_x, _y])
                if self.previous_axis is None:
                    self.previous_axis = event.inaxes
            else:  # the click is not in the same subplot, restart collecting points
                print(
                    "Select mask polygon vertices within "
                    "the same subplot: restart masking..."
                )
                self.xy = []
                self.previous_axis = None

    def press_key_aliens(self, event: KeyEvent) -> None:
        """
        Interact with a plot for masking parasitic intensity or detector gaps.

        :param event: button press event
        """
        dim = self._get_dim_from_click_position(event=event)

        if dim in {0, 1, 2}:
            (
                self._data,
                self._mask,
                self.width,
                self.max_colorbar,
                self.frame_index,
                stop_masking,
            ) = gu.update_aliens_combined(
                key=event.key,
                pix=int(np.rint(event.xdata)),
                piy=int(np.rint(event.ydata)),
                original_data=self.original_data,
                original_mask=self.original_mask,
                updated_data=self._data,
                updated_mask=self._mask,
                axes=(self.ax0, self.ax1, self.ax2, self.ax3),
                width=self.width,
                dim=dim,
                frame_index=self.frame_index,
                vmin=0,
                vmax=self.max_colorbar,
                cmap=self.parameters["colormap"].cmap,
                invert_yaxis=not self.parameters["use_rawdata"],
            )
            if stop_masking:
                plt.close("all")

    def press_key_mask(self, event: KeyEvent) -> None:
        """
        Interact with a plot for masking parasitic intensity or detector gaps.

        :param event: button press event
        """
        nz, ny, nx = self.original_data.shape
        dim = self._get_dim_from_click_position(event=event)

        if dim in {0, 1, 2}:
            if self.previous_axis == self.ax0:
                click_dim = 0
                x, y = np.meshgrid(np.arange(nx), np.arange(ny))
                points = np.stack((x.flatten(), y.flatten()), axis=0).T
            elif self.previous_axis == self.ax1:
                click_dim = 1
                x, y = np.meshgrid(np.arange(nx), np.arange(nz))
                points = np.stack((x.flatten(), y.flatten()), axis=0).T
            elif self.previous_axis == self.ax2:
                click_dim = 2
                x, y = np.meshgrid(np.arange(ny), np.arange(nz))
                points = np.stack((x.flatten(), y.flatten()), axis=0).T
            else:
                click_dim = None
                points = None

            (
                self._data,
                self._updated_mask,
                self.flag_pause,
                self.xy,
                self.width,
                self.max_colorbar,
                click_dim,
                stop_masking,
                self.info_text,
            ) = gu.update_mask_combined(
                key=event.key,
                pix=int(np.rint(event.xdata)),
                piy=int(np.rint(event.ydata)),
                original_data=self.original_data,
                original_mask=self._mask,
                updated_data=self._data,
                updated_mask=self._updated_mask,
                axes=(self.ax0, self.ax1, self.ax2, self.ax3),
                flag_pause=self.flag_pause,
                points=points,
                xy=self.xy,
                width=self.width,
                dim=dim,
                click_dim=click_dim,
                info_text=self.info_text,
                vmin=0,
                vmax=self.max_colorbar,
                cmap=self.parameters["colormap"].cmap,
                invert_yaxis=not self.parameters["use_rawdata"],
            )
            if click_dim is None:
                self.previous_axis = None

            if stop_masking:
                plt.close("all")

    def interactive_masking_aliens(self) -> None:
        plt.ioff()
        self.width = 5
        self.max_colorbar = 5
        nz, ny, nx = self.original_data.shape

        self.fig_mask, ((self.ax0, self.ax1), (self.ax2, self.ax3)) = plt.subplots(
            nrows=2, ncols=2, figsize=(12, 6)
        )
        self.fig_mask.canvas.mpl_disconnect(
            self.fig_mask.canvas.manager.key_press_handler_id
        )
        self._data = np.copy(self.original_data)
        self._mask = np.copy(self.original_mask)
        self.frame_index = self.starting_frame
        self.ax0.imshow(
            self._data[self.frame_index[0], :, :],
            vmin=0,
            vmax=self.max_colorbar,
            cmap=self.parameters["colormap"].cmap,
        )
        self.ax1.imshow(
            self._data[:, self.frame_index[1], :],
            vmin=0,
            vmax=self.max_colorbar,
            cmap=self.parameters["colormap"].cmap,
        )
        self.ax2.imshow(
            self._data[:, :, self.frame_index[2]],
            vmin=0,
            vmax=self.max_colorbar,
            cmap=self.parameters["colormap"].cmap,
        )
        self.ax3.set_visible(False)
        self.ax0.axis("scaled")
        self.ax1.axis("scaled")
        self.ax2.axis("scaled")
        if not self.parameters["use_rawdata"]:
            self.ax0.invert_yaxis()  # detector Y is vertical down
        self.ax0.set_title(f"XY - Frame {self.frame_index[0] + 1} / {nz}")
        self.ax1.set_title(f"XZ - Frame {self.frame_index[1] + 1} / {ny}")
        self.ax2.set_title(f"YZ - Frame {self.frame_index[2] + 1} / {nx}")
        self.fig_mask.text(
            0.60,
            0.30,
            "m mask ; b unmask ; u next frame ; d previous frame",
            size=12,
        )
        self.fig_mask.text(
            0.60,
            0.25,
            "up larger ; down smaller ; right darker ; left brighter",
            size=12,
        )
        self.fig_mask.text(0.60, 0.20, "p plot full image ; q quit", size=12)
        plt.tight_layout()
        plt.connect("key_press_event", self.press_key_aliens)
        self.fig_mask.set_facecolor(self.parameters["background_plot"])
        plt.show()

        self._mask[np.nonzero(self._mask)] = 1

    def refine_mask(self) -> None:
        if self._mask is None:
            self._mask = np.copy(self.original_mask)
        plt.ioff()
        self.width = 0
        self.max_colorbar = 5
        self.previous_axis = None
        self.xy = []  # list of points for mask

        self.fig_mask, ((self.ax0, self.ax1), (self.ax2, self.ax3)) = plt.subplots(
            nrows=2, ncols=2, figsize=(12, 6)
        )
        self.fig_mask.canvas.mpl_disconnect(
            self.fig_mask.canvas.manager.key_press_handler_id
        )
        self._data = np.copy(self.original_data)
        self._updated_mask = np.zeros(self._data.shape)
        self._data[self._mask == 1] = 0  # will appear as grey in the log plot (nan)
        self.ax0.imshow(
            np.log10(abs(self._data).sum(axis=0)),
            vmin=0,
            vmax=self.max_colorbar,
            cmap=self.parameters["colormap"].cmap,
        )
        self.ax1.imshow(
            np.log10(abs(self._data).sum(axis=1)),
            vmin=0,
            vmax=self.max_colorbar,
            cmap=self.parameters["colormap"].cmap,
        )
        self.ax2.imshow(
            np.log10(abs(self._data).sum(axis=2)),
            vmin=0,
            vmax=self.max_colorbar,
            cmap=self.parameters["colormap"].cmap,
        )
        self.ax3.set_visible(False)
        self.ax0.axis("scaled")
        self.ax1.axis("scaled")
        self.ax2.axis("scaled")
        if not self.parameters["use_rawdata"]:
            self.ax0.invert_yaxis()  # detector Y is vertical down
        self.ax0.set_title("XY")
        self.ax1.set_title("XZ")
        self.ax2.set_title("YZ")
        self.fig_mask.text(
            0.60, 0.45, "click to select the vertices of a polygon mask", size=12
        )
        self.fig_mask.text(
            0.60, 0.40, "x to pause/resume polygon masking for pan/zoom", size=12
        )
        self.fig_mask.text(0.60, 0.35, "p plot mask ; r reset current points", size=12)
        self.fig_mask.text(
            0.60,
            0.30,
            "m square mask ; b unmask ; right darker ; left brighter",
            size=12,
        )
        self.fig_mask.text(
            0.60, 0.25, "up larger masking box ; down smaller masking box", size=12
        )
        self.fig_mask.text(0.60, 0.20, "a restart ; q quit", size=12)
        self.info_text = self.fig_mask.text(0.60, 0.05, "masking enabled", size=16)
        plt.tight_layout()
        plt.connect("key_press_event", self.press_key_mask)
        plt.connect("button_press_event", self.on_click)
        self.fig_mask.set_facecolor(self.parameters["background_plot"])
        plt.show()

        self._mask[np.nonzero(self._updated_mask)] = 1

    def _get_dim_from_click_position(self, event: KeyEvent) -> int:
        """Convert the mouse click position into a slice dimention in the 3D array."""
        try:
            if event.inaxes == self.ax0:
                dim = 0
            elif event.inaxes == self.ax1:
                dim = 1
            elif event.inaxes == self.ax2:
                dim = 2
            else:
                dim = -1
        except AttributeError:  # mouse pointer out of axes
            dim = -1
        return dim


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
        self.metadata: Optional[Dict[str, Any]] = None
        self.pad_width = [0, 0, 0, 0, 0, 0]
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
    def is_binning_rocking_axis_needed(self) -> bool:
        # data was already binned for reload_orthogonal
        return (
            self.setup.detector.binning[0] != 1
            and not self.parameters["reload_orthogonal"]
        )

    @property
    def is_filtering_needed(self) -> bool:
        return self.parameters.get("median_filter", "skip") in {
            "mask_isolated",
            "median",
            "interp_isolated",
        }

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
            return float(np.linalg.norm(self.q_bragg))
        return None

    @property
    def scan_nb(self) -> int:
        return int(self.parameters["scans"][self.scan_index])

    @property
    def starting_frame(self) -> List[int]:
        return [self.pad_width[0], self.pad_width[2], self.pad_width[4]]

    def apply_photon_threshold(self) -> None:
        threshold = self.parameters["photon_threshold"]
        if threshold != 0:
            self.mask[self.data < threshold] = 1
            self.data[self.data < threshold] = 0
            self.logger.info(f"Applying photon threshold < {threshold}")

    def bin_rocking_axis(self) -> None:
        self.data = util.bin_data(
            self.data,
            (self.setup.detector.binning[0], 1, 1),
            debugging=False,
            cmap=self.parameters["colormap"].cmap,
            logger=self.logger,
        )
        self.mask = util.bin_data(
            self.mask,
            (self.setup.detector.binning[0], 1, 1),
            debugging=False,
            cmap=self.parameters["colormap"].cmap,
            logger=self.logger,
        )
        self.mask[np.nonzero(self.mask)] = 1
        self.setup.detector.current_binning = list(
            map(
                mul,
                self.setup.detector.current_binning,
                (self.setup.detector.binning[0], 1, 1),
            )
        )
        self.logger.info(
            f"Data size after binning the stacking dimension: {self.data.shape}"
        )
        if self.is_orthogonal and self.data_loader.q_values is not None:
            numz = len(self.data_loader.q_values[0])
            self.data_loader.q_values[0] = self.data_loader.q_values[0][
                : numz
                - (numz % self.setup.detector.binning[0]) : self.setup.detector.binning[
                    0
                ]
            ]

    @abstractmethod
    def calculate_q_bragg(self, **kwargs) -> None:
        raise NotImplementedError

    def calculate_q_bragg_orthogonal(self) -> None:
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

    def cast_data_to_int(self):
        self.data = self.data.astype(int)

    def center_fft(self) -> None:
        (
            self.data,
            self.mask,
            self.pad_width,
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
            bragg_peak=self.parameters["bragg_peak"],
            q_values=self.data_loader.q_values,
            logger=self.logger,
        )
        # no need to check padded frames during masking
        self.logger.info(f"Pad width: {self.pad_width}")
        self.logger.info(f"Data size after cropping / padding: {self.data.shape}")

    def check_binning(self) -> None:
        expected_binning = list(
            map(
                mul,
                self.setup.detector.preprocessing_binning,
                self.setup.detector.binning,
            )
        )
        if any(
            val1 != val2
            for val1, val2 in zip(self.setup.detector.current_binning, expected_binning)
        ):
            raise ValueError(
                "Mismatch in binning, current_binning = "
                f"{self.setup.detector.current_binning}, "
                f"expected binning = {expected_binning}"
            )

    def contour_data(self, title: str, filename: Optional[str]):
        if self.data_loader.q_values is None:
            raise ValueError("q_values not defined")
        max_z = self.data.sum(axis=0).max()
        fig, _, _ = gu.contour_slices(
            self.data,
            self.data_loader.q_values,
            sum_frames=True,
            title=title,
            plot_colorbar=True,
            scale="log",
            is_orthogonal=True,
            levels=np.linspace(0, np.ceil(np.log10(max_z)), 150, endpoint=False),
            reciprocal_space=True,
            cmap=self.parameters["colormap"].cmap,
        )
        fig.savefig(filename)
        plt.close(fig)

    def crop_to_fft_compliant_shape(self) -> None:
        final_shape = util.smaller_primes(
            self.data.shape, maxprime=7, required_dividers=(2,)
        )
        com = tuple(map(lambda x: int(np.rint(x)), center_of_mass(self.data)))
        crop_center = util.find_crop_center(
            array_shape=self.data.shape, crop_shape=final_shape, pivot=com
        )
        self.data = util.crop_pad(
            self.data,
            output_shape=final_shape,
            crop_center=crop_center,
            cmap=self.parameters["colormap"].cmap,
            logger=self.logger,
        )
        self.mask = util.crop_pad(
            self.mask,
            output_shape=final_shape,
            crop_center=crop_center,
            cmap=self.parameters["colormap"].cmap,
            logger=self.logger,
        )
        self.logger.info(
            f"Data size after considering FFT shape requirements: {self.data.shape}"
        )

    def filter_data(self) -> None:
        if self.parameters["median_filter"] in {"mask_isolated", "interp_isolated"}:
            self.logger.info("Filtering isolated pixels")
            nb_pix = 0
            for idx in range(
                self.pad_width[0], self.data.shape[0] - self.pad_width[1]
            ):  # filter only frames whith data (not padded)
                (
                    self.data[idx, :, :],
                    processed_pix,
                    self.mask[idx, :, :],
                ) = util.mean_filter(
                    data=self.data[idx, :, :],
                    nb_neighbours=self.parameters["median_filter_order"],
                    mask=self.mask[idx, :, :],
                    interpolate=self.parameters["median_filter"],
                    min_count=3,
                    debugging=self.parameters["debug"],
                    cmap=self.parameters["colormap"].cmap,
                )
                nb_pix += processed_pix
            self.logger.info(f"Total number of filtered pixels: {nb_pix}")
        if self.parameters["median_filter"] == "median":  # apply median filter
            self.logger.info("Applying median filtering")
            for idx in range(self.pad_width[0], self.data.shape[0] - self.pad_width[1]):
                # filter only frames whith data (not padded)
                self.data[idx, :, :] = scipy.signal.medfilt2d(
                    self.data[idx, :, :], [3, 3]
                )

    def get_interactive_masker(self) -> InteractiveMasker:
        return InteractiveMasker(
            data=self.data,
            mask=self.mask,
            parameters=self.parameters,
            starting_frame=self.starting_frame,
        )

    def get_masked_data(self) -> np.ndarray:
        data = np.copy(self.data)
        data[self.mask == 1] = 0
        return data

    def interpolate_data(self) -> None:
        raise NotImplementedError

    def initialize_comment(self) -> Comment:
        return Comment(self.parameters.get("comment", ""))

    def apply_mask_to_data(self) -> None:
        self.data[np.nonzero(self.mask)] = 0

    def mask_zero_events(self) -> None:
        nz, ny, nx = self.data.shape
        temp_mask = np.zeros((ny, nx))
        temp_mask[np.sum(self.data, axis=0) == 0] = 1
        self.mask[np.repeat(temp_mask[np.newaxis, :, :], repeats=nz, axis=0) == 1] = 1

    def remove_nan(self) -> None:
        self.data, self.mask = util.remove_nan(data=self.data, mask=self.mask)
        self.apply_mask_to_data()

    def retrieve_bragg_peak(self) -> None:
        self.metadata = bu.find_bragg(
            array=self.data,
            binning=self.setup.detector.current_binning,
            roi=self.setup.detector.roi,
            peak_method=self.parameters["centering_method"]["reciprocal_space"],
            tilt_values=self.setup.tilt_angles,
            savedir=self.setup.detector.savedir,
            user_defined_peak=self.parameters["bragg_peak"],
            logger=self.logger,
            plot_fit=True,
        )
        self.update_parameters({"bragg_peak": self.metadata["bragg_peak"]})

    def save_data(self, filename: str) -> None:
        if self.parameters["save_to_npz"]:
            np.savez_compressed(filename, data=self.data)
        if self.parameters["save_to_mat"]:
            if self.data.ndim != 3:
                raise ValueError(
                    f"Only 3D ndarray supported, data is {self.data.ndim}D"
                )
            # save to .mat, the new order is x y z
            # (outboard, vertical up, downstream)
            savemat(
                filename + ".mat",
                {
                    "data": np.moveaxis(
                        self.data.astype(
                            int if self.parameters["save_as_int"] else np.float32
                        ),
                        [0, 1, 2],
                        [-1, -2, -3],
                    )
                },
            )

    def save_hotpixels(self, filename: str) -> None:
        np.savez_compressed(filename, hotpixels=self.mask.astype(int))

    def save_mask(self, filename: str) -> None:
        if self.parameters["save_to_npz"]:
            np.savez_compressed(filename, mask=self.mask.astype(int))
        if self.parameters["save_to_mat"]:
            if self.data.ndim != 3:
                raise ValueError(
                    f"Only 3D ndarray supported, data is {self.data.ndim}D"
                )
            # save to .mat, the new order is x y z
            # (outboard, vertical up, downstream)
            savemat(
                filename + ".mat",
                {
                    "mask": np.moveaxis(
                        self.mask.astype(np.int8),
                        [0, 1, 2],
                        [-1, -2, -3],
                    )
                },
            )

    def save_results_as_h5(self, filename: str) -> None:
        with h5py.File(filename, "w") as hf:
            out = hf.create_group("output")
            par = hf.create_group("params")
            self._create_h5_dataset(out, "data", data=self.data)
            self._create_h5_dataset(out, "mask", data=self.mask)

            if self.metadata is not None:
                self._create_h5_dataset(
                    out, "tilt_values", data=self.metadata["tilt_values"]
                )
                self._create_h5_dataset(
                    out, "rocking_curve", data=self.metadata["rocking_curve"]
                )
                self._create_h5_dataset(
                    out, "interp_tilt", data=self.metadata["interp_tilt_values"]
                )
                self._create_h5_dataset(
                    out, "interp_curve", data=self.metadata["interp_rocking_curve"]
                )
                self._create_h5_dataset(
                    out, "COM_rocking_curve", data=self.metadata["tilt_value_at_peak"]
                )
                self._create_h5_dataset(
                    out,
                    "detector_data_COM",
                    data=self.metadata["detector_data_at_peak"],
                )
                self._create_h5_dataset(
                    out, "interp_fwhm", data=self.metadata["interp_fwhm"]
                )
            self._create_h5_dataset(
                out, "bragg_peak", data=self.parameters["bragg_peak"]
            )
            self._create_h5_dataset(out, "q_bragg", data=self.q_bragg)
            self._create_h5_dataset(out, "qnorm", data=self.q_norm)
            self._create_h5_dataset(out, "planar_distance", data=self.planar_distance)
            if self.parameters["rocking_angle"] != "energy":
                self._create_h5_dataset(
                    out, "bragg_inplane", data=self.parameters["inplane_angle"]
                )
                self._create_h5_dataset(
                    out, "bragg_outofplane", data=self.parameters["outofplane_angle"]
                )
            self._create_h5_dataset(
                par, "detector", data=str(self.setup.detector.params)
            )
            self._create_h5_dataset(par, "setup", data=str(self.setup.params))
            self._create_h5_dataset(par, "parameters", data=str(self.parameters))

    def save_q_values(self, filename: str) -> None:
        if self.data_loader.q_values is None:
            raise ValueError("q_values not defined")
        qx, qz, qy = self.data_loader.q_values
        if self.parameters["save_to_npz"]:
            np.savez_compressed(
                filename,
                qx=qx,
                qz=qz,
                qy=qy,
            )
        if self.parameters["save_to_mat"]:
            savemat(self.setup.detector.savedir + f"S{self.scan_nb}_qx.mat", {"qx": qx})
            savemat(self.setup.detector.savedir + f"S{self.scan_nb}_qz.mat", {"qz": qz})
            savemat(self.setup.detector.savedir + f"S{self.scan_nb}_qy.mat", {"qy": qy})

    def save_to_vti(self, filename: Optional[str]) -> None:
        if self.data_loader.q_values is None:
            raise ValueError("q_values are undefined.")
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

    def set_binary_mask(self):
        self.mask[np.nonzero(self.mask)] = 1
        self.mask = self.mask.astype(int)

    def show_array(self, array: np.ndarray, title: str, **kwargs) -> Any:
        fig, _, _ = gu.multislices_plot(
            array,
            sum_frames=True,
            plot_colorbar=True,
            vmin=0,
            title=title,
            is_orthogonal=self.is_orthogonal,
            reciprocal_space=True,
            cmap=self.parameters["colormap"].cmap,
            **kwargs,
        )
        return fig

    def show_mask(self, title: str, filename: Optional[str], **kwargs) -> None:
        fig = self.show_array(array=self.mask, title=title, scale="linear", **kwargs)
        if filename is not None:
            fig.savefig(filename)
        plt.close(fig)

    def show_masked_data(self, title: str, filename: Optional[str]):
        fig = self.show_array(array=self.get_masked_data(), title=title, scale="log")
        if filename is not None:
            fig.savefig(filename)
        plt.close(fig)

    def show_masked_data_at_com(self, title: str, filename: Optional[str]):
        data = self.get_masked_data()
        z0, y0, x0 = center_of_mass(data)
        fig = self.show_array(
            array=data,
            title=title,
            slice_position=[int(z0), int(y0), int(x0)],
            scale="log",
        )
        if filename is not None:
            fig.savefig(filename)
        plt.close(fig)

    def show_masked_data_at_max(self, title: str, filename: Optional[str]):
        data = self.get_masked_data()
        fig = self.show_array(
            array=data,
            title=title,
            slice_position=[
                int(val) for val in np.unravel_index(data.argmax(), data.shape)
            ],
            scale="log",
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

    def _create_h5_dataset(self, group, dataset_name: str, data: Any) -> None:
        try:
            group.create_dataset(dataset_name, data=data)
        except TypeError:
            self.logger.warning(
                f"Cannot create dataset '{dataset_name}', data not available"
            )


class DetectorFrameAnalysis(Analysis):
    """Analysis worklow in the detector frame."""

    def calculate_q_bragg(self, **kwargs) -> None:
        self.q_bragg = self.setup.q_laboratory

    def interpolate_data(self) -> None:
        self.logger("No interpolation performed for the analysis in detector frame")


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
        if (
            self.parameters["reload_previous"]
            and self.parameters.get("frames_pattern") is None
        ):
            raise ValueError(
                "The parameters 'frames_pattern' is required to process "
                "correctly the motor positions"
            )

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
        self.calculate_q_bragg_orthogonal()

    def interpolate_data(self) -> None:
        self.logger("Reloaded data is already interpolated in an orthogonal frame.")


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
        if (
            self.parameters["reload_previous"]
            and self.parameters.get("frames_pattern") is None
        ):
            raise ValueError(
                "The parameters 'frames_pattern' is required to process "
                "correctly the motor positions"
            )

    def calculate_q_bragg(self, **kwargs) -> None:
        self.calculate_q_bragg_orthogonal()

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
        nch1, nch2 = self.data.shape[1:]
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
    """Base class for loading/reloading a dataset in preprocessing."""

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
        return int(self.parameters["scans"][self.scan_index])

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
            frames_pattern=self.parameters.get("frames_pattern"),
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

    def __init__(
        self, scan_index: int, parameters: Dict[str, Any], setup: "Setup", **kwargs
    ) -> None:
        super().__init__(
            scan_index=scan_index, parameters=parameters, setup=setup, **kwargs
        )
        self.parameters["normalize_flux"] = "skip"

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
            frames_pattern=self.parameters.get("frames_pattern"),
        )


class ReloadingOrthogonalFrame(PreprocessingLoader):
    """Reload a dataset already interpolated in an orthogonal frame."""

    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        data, mask = self.reload()
        q_values = self.load_q_values()

        self.parameters["normalize_flux"] = "skip"
        # we assume that normalization was already performed
        monitor = np.ones(data.shape[0])
        frames_logical = np.ones(data.shape[0], dtype=int)
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
        return data, mask, frames_logical, monitor

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
    """Instantiate a data loading class depending on analysis parameters."""
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
