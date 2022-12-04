# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, jerome.carnis@esrf.fr

"""Class for centering methods."""

import logging
from abc import ABC
from typing import Any, List, Optional, Tuple
import numpy as np
from scipy.ndimage import center_of_mass
from bcdi.utils import utilities as util
from bcdi.preprocessing import bcdi_utils as bu

module_logger = logging.getLogger(__name__)


class CenterFFT(ABC):
    """

    axis=0, z downstream, qx in reciprocal space
    axis=1, y vertical, qz in reciprocal space
    axis=2, x outboard, qy in reciprocal space
    """

    def __init__(
        self,
        binning: Tuple[int, int, int],
        preprocessing_binning: Tuple[int, int, int],
        roi: Tuple[int, int, int, int],
        center_position: Tuple[int, int, int],
        max_symmetrical_window: Tuple[int, int, int],
        fix_bragg: List[int],
        fft_option: str,
        pad_size: Optional[Tuple[int, int, int]],
        q_values: Optional[Any] = None,
        logger: logging.Logger = module_logger,
    ):
        self.binning = binning
        self.preprocessing_binning = preprocessing_binning
        self.roi = roi
        self.center_position = center_position
        self.max_symmetrical_window = max_symmetrical_window
        self.fix_bragg = fix_bragg
        self.fft_option = fft_option
        self.pad_size = pad_size
        self.q_values = q_values
        self.logger = logger

        self.data_shape: Optional[Tuple[int, int, int]] = None
        self.pad_width: Optional[np.ndarray] = None
        self.start_stop_indices: Optional[Tuple[int, int, int, int, int, int]] = None

    @property
    def pad_size(self):
        return self._pad_size

    @pad_size.setter
    def pad_size(self, value):
        if isinstance(value, (list, tuple)):
            if len(value) != 3:
                raise ValueError("pad_size should be a list of three elements")
            if value[0] != util.higher_primes(
                value[0], maxprime=7, required_dividers=(2,)
            ):
                raise ValueError(
                    f"pad_size[0]={value[0]} " f"does not meet FFT requirements"
                )
        elif value is not None:
            raise ValueError(
                "pad_size should be a None or list of three elements, " f"got {value}"
            )
        self._pad_size = value

    def center_fft(
        self,
        data: np.ndarray,
        mask: Optional[np.ndarray],
        frames_logical: Optional[np.ndarray],
    ):
        self.get_data_shape(data)
        data = self.crop_array(data)
        if mask is not None:
            mask = self.crop_array(mask)

        self.get_pad_width()
        data = bu.zero_pad(data, padding_width=self.pad_width, mask_flag=False)
        if mask is not None:
            bu.zero_pad(
                mask, padding_width=self.pad_width, mask_flag=True
            )  # mask padded pixels
        self.logger.info(f"FFT box (qx, qz, qy): {self.data_shape}")

        if frames_logical is not None:
            frames_logical = self.update_frames_logical(frames_logical)

        if self.q_values is not None:
            self.update_q_values()

        return data, mask, self.pad_width, frames_logical, self.q_values

    def crop_array(self, array: np.ndarray):
        if self.start_stop_indices is None:
            raise ValueError("start_stop_indices is None")
        return array[
            self.start_stop_indices[0] : self.start_stop_indices[1],
            self.start_stop_indices[2] : self.start_stop_indices[3],
            self.start_stop_indices[4] : self.start_stop_indices[5],
        ]

    def get_pad_width(self) -> None:
        raise NotImplementedError

    def get_data_shape(self, data: np.ndarray) -> None:
        self.data_shape = data.shape

    def get_start_stop_indices(self) -> Tuple[int, int, int, int, int, int]:
        raise NotImplementedError

    def update_frames_logical(
        self,
        frames_logical: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError

    def update_q_values(self) -> None:
        self.q_values[0] = self.q_values[0][
            self.start_stop_indices[0] : self.start_stop_indices[1]
        ]
        self.q_values[1] = self.q_values[1][
            self.start_stop_indices[2] : self.start_stop_indices[3]
        ]
        self.q_values[2] = self.q_values[2][
            self.start_stop_indices[4] : self.start_stop_indices[5]
        ]


class CenteringFactory:
    def __init__(
        self,
        data: np.ndarray,
        binning: Tuple[int, int, int],
        preprocessing_binning: Tuple[int, int, int],
        roi: Tuple[int, int, int, int],
        fix_bragg: List[int],
        fft_option: str = "crop_asymmetric_ZYX",
        pad_size: Optional[Tuple[int, int, int]] = None,
        centering_method: str = "max",
        logger: logging.Logger = module_logger,
        q_values: Optional[List[np.ndarray]] = None,
    ):
        self.binning = binning
        self.preprocessing_binning = preprocessing_binning
        self.roi = roi
        self.fix_bragg = fix_bragg
        self.fft_option = fft_option
        self.pad_size = pad_size
        self.logger = logger
        self.q_values = q_values

        self.center_position = self.find_center(data=data, method=centering_method)
        self.max_symmetrical_window = self.get_max_symmetrical_box(data=data)
        self.check_center_position()

    def get_max_symmetrical_box(self, data: np.ndarray) -> Tuple[int, int, int]:
        nbz, nby, nbx = np.shape(data)
        iz0, iy0, ix0 = self.center_position
        return (
            abs(2 * min(iz0, nbz - iz0)),
            2 * min(iy0, nby - iy0),
            abs(2 * min(ix0, nbx - ix0)),
        )

    def check_center_position(self) -> None:
        max_nz, max_ny, max_nx = self.max_symmetrical_window
        if self.fft_option != "skip":
            self.logger.info(
                f"Max symmetrical box (qx, qz, qy): ({max_nz, max_ny, max_nx})"
            )
        if any(val == 0 for val in (max_nz, max_ny, max_nx)):
            self.logger.info(
                "Empty images or presence of hotpixel at the border,"
                ' defaulting fft_option to "skip"!'
            )
            self.fft_option = "skip"

    def find_center(self, data: np.ndarray, method: str) -> Tuple[int, int, int]:
        if method == "max":
            z0, y0, x0 = np.unravel_index(abs(data).argmax(), data.shape)
            if self.q_values:
                self.logger.info(
                    f"Max at (qx, qz, qy): {self.q_values[0][z0]:.5f}, "
                    f"{self.q_values[1][y0]:.5f}, {self.q_values[2][x0]:.5f}"
                )
            else:
                self.logger.info(f"Max at pixel (Z, Y, X): ({z0, y0, x0})")
        elif method == "com":
            z0, y0, x0 = center_of_mass(data)
            if self.q_values:
                self.logger.info(
                    "Center of mass at (qx, qz, qy): "
                    f"{self.q_values[0][z0]:.5f}, {self.q_values[1][y0]:.5f}, "
                    f"{self.q_values[2][x0]:.5f}"
                )
            else:
                self.logger.info(f"Center of mass at pixel (Z, Y, X): ({z0, y0, x0})")
        else:  # 'max_com'
            position = list(np.unravel_index(abs(data).argmax(), data.shape))
            position[1:] = center_of_mass(data[position[0], :, :])
            z0, y0, x0 = tuple(map(lambda x: int(np.rint(x)), position))

        if self.fix_bragg:
            if len(self.fix_bragg) != 3:
                raise ValueError("fix_bragg should be a list of 3 integers")
            z0, y0, x0 = self.fix_bragg
            self.logger.info(
                "Peak intensity position defined by user on the full detector: "
                f"({z0, y0, x0})"
            )
            y0 = (y0 - self.roi[0]) / (self.preprocessing_binning[1] * self.binning[1])
            x0 = (x0 - self.roi[2]) / (self.preprocessing_binning[2] * self.binning[2])
            self.logger.info(
                "Peak intensity position with detector ROI and binning in the "
                f"detector plane: ({z0, y0, x0})"
            )
        return int(round(z0)), int(round(y0)), int(round(x0))

    def get_centering_instance(self) -> CenterFFT:
        if self.fft_option == "crop_sym_ZYX":
            centering_class = CenterFFTCropSymZYX
        elif self.fft_option == "crop_asym_ZYX":
            centering_class = CenterFFTCropAsymZYX
        elif self.fft_option == "pad_sym_Z_crop_sym_YX":
            centering_class = CenterFFTPadSymZCropSymYX
        elif self.fft_option == "pad_sym_Z_crop_asym_YX":
            centering_class = CenterFFTPadSymZCropAsymYX
        elif self.fft_option == "pad_asym_Z_crop_sym_YX":
            centering_class = CenterFFTPadAsymZCropSymYX
        elif self.fft_option == "pad_asym_Z_crop_asym_YX":
            centering_class = CenterFFTPadAsymZCropAsymYX
        elif self.fft_option == "pad_sym_Z":
            centering_class = CenterFFTPadSymZ
        elif self.fft_option == "pad_asym_Z":
            centering_class = CenterFFTPadAsymZ
        elif self.fft_option == "pad_sym_ZYX":
            centering_class = CenterFFTPadSymZYX
        elif self.fft_option == "pad_asym_ZYX":
            centering_class = CenterFFTPadAsymZYX
        elif self.fft_option == "skip":
            centering_class = SkipCentering
        else:
            raise ValueError(f"Incorrect value {self.fft_option} for 'fft_option'")

        return centering_class(
            binning=self.binning,
            preprocessing_binning=self.preprocessing_binning,
            roi=self.roi,
            center_position=self.center_position,
            max_symmetrical_window=self.max_symmetrical_window,
            fix_bragg=self.fix_bragg,
            fft_option=self.fft_option,
            pad_size=self.pad_size,
            logger=self.logger,
            q_values=self.q_values,
        )


class CenterFFTCropSymZYX(CenterFFT):
    def __init__(
        self,
        binning: Tuple[int, int, int],
        preprocessing_binning: Tuple[int, int, int],
        roi: Tuple[int, int, int, int],
        center_position: Tuple[int, int, int],
        max_symmetrical_window: Tuple[int, int, int],
        fix_bragg: List[int],
        fft_option: str,
        pad_size: Optional[Tuple[int, int, int]],
        q_values: Optional[List[np.ndarray]] = None,
        logger: logging.Logger = module_logger,
    ):
        super().__init__(
            binning=binning,
            preprocessing_binning=preprocessing_binning,
            roi=roi,
            center_position=center_position,
            max_symmetrical_window=max_symmetrical_window,
            fix_bragg=fix_bragg,
            fft_option=fft_option,
            pad_size=pad_size,
            q_values=q_values,
            logger=logger,
        )
        self.start_stop_indices = self.get_start_stop_indices()

    def get_pad_width(self) -> None:
        self.pad_width = np.zeros(6, dtype=int)

    def get_start_stop_indices(self) -> Tuple[int, int, int, int, int, int]:
        iz0, iy0, ix0 = self.center_position

        # crop rocking angle and detector, Bragg peak centered
        nz1, ny1, nx1 = util.smaller_primes(
            self.max_symmetrical_window, maxprime=7, required_dividers=(2,)
        )
        return (
            iz0 - nz1 // 2,
            iz0 + nz1 // 2,
            iy0 - ny1 // 2,
            iy0 + ny1 // 2,
            ix0 - nx1 // 2,
            ix0 + nx1 // 2,
        )

    def update_frames_logical(
        self,
        frames_logical: np.ndarray,
    ) -> np.ndarray:
        if self.data_shape is None:
            raise ValueError("data_shape is None")
        if self.start_stop_indices[0] > 0:  # if 0, the first frame is used
            frames_logical[0 : self.start_stop_indices[0]] = 0
        if self.start_stop_indices[1] < self.data_shape[0]:
            # if nbz, the last frame is used
            frames_logical[self.start_stop_indices[1] :] = 0
        return frames_logical


class CenterFFTCropAsymZYX(CenterFFT):
    def get_pad_width(self) -> None:
        self.pad_width = np.zeros(6, dtype=int)

    def get_start_stop_indices(self) -> Tuple[int, int, int, int, int, int]:
        nbz, nby, nbx = self.data_shape
        # crop rocking angle and detector without centering the Bragg peak
        nz1, ny1, nx1 = util.smaller_primes(
            (nbz, nby, nbx), maxprime=7, required_dividers=(2,)
        )
        return (
            nbz - nz1 // 2,
            nbz + nz1 // 2,
            nby - ny1 // 2,
            nby + ny1 // 2,
            nbx - nx1 // 2,
            nbx + nx1 // 2,
        )

    def update_frames_logical(
        self,
        frames_logical: np.ndarray,
    ) -> np.ndarray:
        if self.data_shape is None:
            raise ValueError("data_shape is None")
        if self.start_stop_indices[0] > 0:  # if 0, the first frame is used
            frames_logical[0 : self.start_stop_indices[0]] = 0
        if self.start_stop_indices[1] < self.data_shape[0]:
            # if nbz, the last frame is used
            frames_logical[self.start_stop_indices[1] :] = 0
        return frames_logical


class CenterFFTPadSymZCropSymYX(CenterFFT):
    def get_pad_width(self) -> None:
        if self.pad_size is None:
            self.pad_width = np.zeros(6, dtype=int)

        self.pad_width = np.array(
            [
                int(
                    min(
                        self.pad_size[0] / 2 - self.center_position[0],
                        self.pad_size[0] - self.data_shape[0],
                    )
                ),
                int(
                    min(
                        self.pad_size[0] / 2
                        - self.data_shape[0]
                        + self.center_position[0],
                        self.pad_size[0] - self.data_shape[0],
                    )
                ),
                0,
                0,
                0,
                0,
            ],
            dtype=int,
        )

    def get_start_stop_indices(self) -> Tuple[int, int, int, int, int, int]:
        _, iy0, ix0 = self.center_position

        ny1, nx1 = util.smaller_primes(
            self.max_symmetrical_window[1:], maxprime=7, required_dividers=(2,)
        )
        return (
            0,
            self.data_shape[0],
            iy0 - ny1 // 2,
            iy0 + ny1 // 2,
            ix0 - nx1 // 2,
            ix0 + nx1 // 2,
        )

    def update_frames_logical(
        self,
        frames_logical: np.ndarray,
    ) -> np.ndarray:
        temp_frames = -1 * np.ones(self.data_shape[0])
        temp_frames[
            self.pad_width[0] : self.pad_width[0] + self.data_shape[0]
        ] = frames_logical
        return temp_frames

    def update_q_values(self) -> None:
        dqx = self.q_values[0][1] - self.q_values[0][0]
        qx0 = self.q_values[0][0] - self.pad_width[0] * dqx
        self.q_values[0] = qx0 + np.arange(self.pad_size[0]) * dqx
        self.q_values[1] = self.q_values[1][
            self.start_stop_indices[2] : self.start_stop_indices[3]
        ]
        self.q_values[2] = self.q_values[2][
            self.start_stop_indices[4] : self.start_stop_indices[5]
        ]


class CenterFFTPadSymZCropAsymYX(CenterFFT):
    def get_pad_width(self) -> None:
        if self.pad_size is None:
            self.pad_width = np.zeros(6, dtype=int)

        self.pad_width = np.array(
            [
                int(
                    min(
                        self.pad_size[0] / 2 - self.center_position[0],
                        self.pad_size[0] - self.data_shape[0],
                    )
                ),
                int(
                    min(
                        self.pad_size[0] / 2
                        - self.data_shape[0]
                        + self.center_position[0],
                        self.pad_size[0] - self.data_shape[0],
                    )
                ),
                0,
                0,
                0,
                0,
            ],
            dtype=int,
        )

    def get_start_stop_indices(self) -> Tuple[int, int, int, int, int, int]:
        ny1, nx1 = util.smaller_primes(
            self.max_symmetrical_window[1:], maxprime=7, required_dividers=(2,)
        )
        return (
            0,
            self.data_shape[0],
            self.data_shape[1] - ny1 // 2,
            self.data_shape[1] + ny1 // 2,
            self.data_shape[2] - nx1 // 2,
            self.data_shape[2] + nx1 // 2,
        )

    def update_frames_logical(
        self,
        frames_logical: np.ndarray,
    ) -> np.ndarray:
        temp_frames = -1 * np.ones(self.data_shape[0])
        temp_frames[
            self.pad_width[0] : self.pad_width[0] + self.data_shape[0]
        ] = frames_logical
        return temp_frames

    def update_q_values(self) -> None:
        dqx = self.q_values[0][1] - self.q_values[0][0]
        qx0 = self.q_values[0][0] - self.pad_width[0] * dqx
        self.q_values[0] = qx0 + np.arange(self.pad_size[0]) * dqx
        self.q_values[1] = self.q_values[1][
            self.start_stop_indices[2] : self.start_stop_indices[3]
        ]
        self.q_values[2] = self.q_values[2][
            self.start_stop_indices[4] : self.start_stop_indices[5]
        ]


class CenterFFTPadAsymZCropSymYX(CenterFFT):
    def get_pad_width(self) -> None:
        # pad rocking angle without centering the Bragg peak
        nz1 = util.higher_primes(self.data_shape[0], maxprime=7, required_dividers=(2,))
        self.pad_width = np.array(
            [
                int((nz1 - self.data_shape[0] + ((nz1 - self.data_shape[0]) % 2)) / 2),
                int(
                    (nz1 - self.data_shape[0] + 1) / 2
                    - ((nz1 - self.data_shape[0]) % 2)
                ),
                0,
                0,
                0,
                0,
            ],
            dtype=int,
        )

    def get_start_stop_indices(self) -> Tuple[int, int, int, int, int, int]:
        # crop detector (Bragg peak centered)
        ny1, nx1 = util.smaller_primes(
            self.max_symmetrical_window[1:], maxprime=7, required_dividers=(2,)
        )
        _, iy0, ix0 = self.center_position
        return (
            0,
            self.data_shape[0],
            iy0 - ny1 // 2,
            iy0 + ny1 // 2,
            ix0 - nx1 // 2,
            ix0 + nx1 // 2,
        )

    def update_frames_logical(
        self,
        frames_logical: np.ndarray,
    ) -> np.ndarray:
        temp_frames = -1 * np.ones(self.data_shape[0])
        temp_frames[
            self.pad_width[0] : self.pad_width[0] + self.data_shape[0]
        ] = frames_logical
        return temp_frames

    def update_q_values(self) -> None:
        nz1 = util.higher_primes(self.data_shape[0], maxprime=7, required_dividers=(2,))
        dqx = self.q_values[0][1] - self.q_values[0][0]
        qx0 = self.q_values[0][0] - self.pad_width[0] * dqx
        self.q_values[0] = qx0 + np.arange(nz1) * dqx
        self.q_values[1] = self.q_values[1][
            self.start_stop_indices[2] : self.start_stop_indices[3]
        ]
        self.q_values[2] = self.q_values[2][
            self.start_stop_indices[4] : self.start_stop_indices[5]
        ]


class CenterFFTPadAsymZCropAsymYX(CenterFFT):
    def get_pad_width(self) -> None:
        nz1 = util.higher_primes(self.data_shape[0], maxprime=7, required_dividers=(2,))
        self.pad_width = np.array(
            [
                int((nz1 - self.data_shape[0] + ((nz1 - self.data_shape[0]) % 2)) / 2),
                int(
                    (nz1 - self.data_shape[0] + 1) / 2
                    - ((nz1 - self.data_shape[0]) % 2)
                ),
                0,
                0,
                0,
                0,
            ],
            dtype=int,
        )

    def get_start_stop_indices(self) -> Tuple[int, int, int, int, int, int]:
        # crop detector without centering the Bragg peak
        ny1, nx1 = util.smaller_primes(
            self.data_shape[1:], maxprime=7, required_dividers=(2,)
        )
        return (
            0,
            self.data_shape[0],
            self.data_shape[1] // 2 - ny1 // 2,
            self.data_shape[1] // 2 + ny1 // 2,
            self.data_shape[2] // 2 - nx1 // 2,
            self.data_shape[2] // 2 + nx1 // 2,
        )

    def update_frames_logical(
        self,
        frames_logical: np.ndarray,
    ) -> np.ndarray:
        temp_frames = -1 * np.ones(self.data_shape[0])
        temp_frames[
            self.pad_width[0] : self.pad_width[0] + self.data_shape[0]
        ] = frames_logical
        return temp_frames

    def update_q_values(self) -> None:
        nz1 = util.higher_primes(self.data_shape[0], maxprime=7, required_dividers=(2,))
        dqx = self.q_values[0][1] - self.q_values[0][0]
        qx0 = self.q_values[0][0] - self.pad_width[0] * dqx
        self.q_values[0] = qx0 + np.arange(nz1) * dqx
        self.q_values[1] = self.q_values[1][
            self.start_stop_indices[2] : self.start_stop_indices[3]
        ]
        self.q_values[2] = self.q_values[2][
            self.start_stop_indices[4] : self.start_stop_indices[5]
        ]


class CenterFFTPadSymZ(CenterFFT):
    def get_pad_width(self) -> None:
        if self.pad_size[0] != util.higher_primes(
            self.pad_size[0], maxprime=7, required_dividers=(2,)
        ):
            raise ValueError(self.pad_size[0], "does not meet FFT requirements")
        self.pad_width = np.array(
            [
                int(
                    min(
                        self.pad_size[0] / 2 - self.center_position[0],
                        self.pad_size[0] - self.data_shape[0],
                    )
                ),
                int(
                    min(
                        self.pad_size[0] / 2
                        - self.data_shape[0]
                        + self.center_position[0],
                        self.pad_size[0] - self.data_shape[0],
                    )
                ),
                0,
                0,
                0,
                0,
            ],
            dtype=int,
        )

    def get_start_stop_indices(self) -> Tuple[int, int, int, int, int, int]:
        return 0, self.data_shape[0], 0, self.data_shape[1], 0, self.data_shape[2]

    def update_frames_logical(
        self,
        frames_logical: np.ndarray,
    ) -> np.ndarray:
        temp_frames = -1 * np.ones(self.data_shape[0])
        temp_frames[
            self.pad_width[0] : self.pad_width[0] + self.data_shape[0]
        ] = frames_logical
        return temp_frames

    def update_q_values(self) -> None:
        dqx = self.q_values[0][1] - self.q_values[0][0]
        qx0 = self.q_values[0][0] - self.pad_width[0] * dqx
        self.q_values[0] = qx0 + np.arange(self.pad_size[0]) * dqx


class CenterFFTPadAsymZ(CenterFFT):
    def get_pad_width(self) -> None:
        # pad rocking angle without centering the Bragg peak, keep detector size
        nz1 = util.higher_primes(self.data_shape[0], maxprime=7, required_dividers=(2,))

        self.pad_width = np.array(
            [
                int((nz1 - self.data_shape[0] + ((nz1 - self.data_shape[0]) % 2)) / 2),
                int(
                    (nz1 - self.data_shape[0] + 1) / 2
                    - ((nz1 - self.data_shape[0]) % 2)
                ),
                0,
                0,
                0,
                0,
            ],
            dtype=int,
        )

    def get_start_stop_indices(self) -> Tuple[int, int, int, int, int, int]:
        return 0, self.data_shape[0], 0, self.data_shape[1], 0, self.data_shape[2]

    def update_frames_logical(
        self,
        frames_logical: np.ndarray,
    ) -> np.ndarray:
        temp_frames = -1 * np.ones(self.data_shape[0])
        temp_frames[
            self.pad_width[0] : self.pad_width[0] + self.data_shape[0]
        ] = frames_logical
        return temp_frames

    def update_q_values(self) -> None:
        nz1 = util.higher_primes(self.data_shape[0], maxprime=7, required_dividers=(2,))
        dqx = self.q_values[0][1] - self.q_values[0][0]
        qx0 = self.q_values[0][0] - self.pad_width[0] * dqx
        self.q_values[0] = qx0 + np.arange(nz1) * dqx


class CenterFFTPadSymZYX(CenterFFT):
    def get_pad_width(self) -> None:
        # pad both dimensions based on 'pad_size'
        self.logger.info(f"pad_size: {self.pad_size}")
        self.logger.info(
            "The 1st axis (stacking dimension) is padded before binning,"
            " detector plane after binning."
        )
        if self.pad_size[0] != util.higher_primes(
            self.pad_size[0], maxprime=7, required_dividers=(2,)
        ):
            raise ValueError(self.pad_size[0], "does not meet FFT requirements")
        if self.pad_size[1] != util.higher_primes(
            self.pad_size[1], maxprime=7, required_dividers=(2,)
        ):
            raise ValueError(self.pad_size[1], "does not meet FFT requirements")
        if self.pad_size[2] != util.higher_primes(
            self.pad_size[2], maxprime=7, required_dividers=(2,)
        ):
            raise ValueError(self.pad_size[2], "does not meet FFT requirements")
        nbz, nby, nbx = self.data_shape
        iz0, iy0, ix0 = self.center_position
        pad_width = [
            int(min(self.pad_size[0] / 2 - iz0, self.pad_size[0] - nbz)),
            int(min(self.pad_size[0] / 2 - nbz + iz0, self.pad_size[0] - nbz)),
            int(min(self.pad_size[1] / 2 - iy0, self.pad_size[1] - nby)),
            int(min(self.pad_size[1] / 2 - nby + iy0, self.pad_size[1] - nby)),
            int(min(self.pad_size[2] / 2 - ix0, self.pad_size[2] - nbx)),
            int(min(self.pad_size[2] / 2 - nbx + ix0, self.pad_size[2] - nbx)),
        ]
        # remove negative numbers
        self.pad_width = np.array(
            list((map(lambda value: max(value, 0), pad_width))), dtype=int
        )

    def get_start_stop_indices(self) -> Tuple[int, int, int, int, int, int]:
        return 0, self.data_shape[0], 0, self.data_shape[1], 0, self.data_shape[2]

    def update_frames_logical(
        self,
        frames_logical: np.ndarray,
    ) -> np.ndarray:
        temp_frames = -1 * np.ones(self.data_shape[0])
        temp_frames[
            self.pad_width[0] : self.pad_width[0] + self.data_shape
        ] = frames_logical
        return temp_frames

    def update_q_values(self) -> None:
        dqx = self.q_values[0][1] - self.q_values[0][0]
        dqz = self.q_values[1][1] - self.q_values[1][0]
        dqy = self.q_values[2][1] - self.q_values[2][0]
        qx0 = self.q_values[0][0] - self.pad_width[0] * dqx
        qz0 = self.q_values[1][0] - self.pad_width[1] * dqz
        qy0 = self.q_values[2][0] - self.pad_width[2] * dqy
        self.q_values[0] = qx0 + np.arange(self.pad_size[0]) * dqx
        self.q_values[1] = qz0 + np.arange(self.pad_size[1]) * dqz
        self.q_values[2] = qy0 + np.arange(self.pad_size[2]) * dqy


class CenterFFTPadAsymZYX(CenterFFT):
    def get_pad_width(self) -> None:
        nbz, nby, nbx = self.data_shape
        nz1, ny1, nx1 = [
            util.higher_primes(nbz, maxprime=7, required_dividers=(2,)),
            util.higher_primes(nby, maxprime=7, required_dividers=(2,)),
            util.higher_primes(nbx, maxprime=7, required_dividers=(2,)),
        ]

        self.pad_width = np.array(
            [
                int((nz1 - nbz + ((nz1 - nbz) % 2)) / 2),
                int((nz1 - nbz + 1) / 2 - ((nz1 - nbz) % 2)),
                int((ny1 - nby + ((ny1 - nby) % 2)) / 2),
                int((ny1 - nby + 1) / 2 - ((ny1 - nby) % 2)),
                int((nx1 - nbx + ((nx1 - nbx) % 2)) / 2),
                int((nx1 - nbx + 1) / 2 - ((nx1 - nbx) % 2)),
            ]
        )

    def get_start_stop_indices(self) -> Tuple[int, int, int, int, int, int]:
        return 0, self.data_shape[0], 0, self.data_shape[1], 0, self.data_shape[2]

    def update_frames_logical(
        self,
        frames_logical: np.ndarray,
    ) -> np.ndarray:
        temp_frames = -1 * np.ones(self.data_shape[0])
        temp_frames[
            self.pad_width[0] : self.pad_width[0] + self.data_shape[0]
        ] = frames_logical
        return temp_frames

    def update_q_values(self) -> None:
        nbz, nby, nbx = self.data_shape
        nz1, ny1, nx1 = [
            util.higher_primes(nbz, maxprime=7, required_dividers=(2,)),
            util.higher_primes(nby, maxprime=7, required_dividers=(2,)),
            util.higher_primes(nbx, maxprime=7, required_dividers=(2,)),
        ]
        dqx = self.q_values[0][1] - self.q_values[0][0]
        dqz = self.q_values[1][1] - self.q_values[1][0]
        dqy = self.q_values[2][1] - self.q_values[2][0]
        qx0 = self.q_values[0][0] - self.pad_width[0] * dqx
        qz0 = self.q_values[1][0] - self.pad_width[1] * dqz
        qy0 = self.q_values[2][0] - self.pad_width[2] * dqy
        self.q_values[0] = qx0 + np.arange(nz1) * dqx
        self.q_values[1] = qz0 + np.arange(ny1) * dqz
        self.q_values[2] = qy0 + np.arange(nx1) * dqy


class SkipCentering(CenterFFT):
    def get_pad_width(self) -> None:
        self.pad_width = np.zeros(6, dtype=int)

    def get_start_stop_indices(self) -> Tuple[int, int, int, int, int, int]:
        return 0, self.data_shape[0], 0, self.data_shape[1], 0, self.data_shape[2]

    def update_frames_logical(
        self,
        frames_logical: np.ndarray,
    ) -> np.ndarray:
        return frames_logical

    def update_q_values(self) -> None:
        return
