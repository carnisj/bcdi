# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, jerome.carnis@esrf.fr

"""Classes related to centering the diffraction pattern in preprocessing."""

import logging
from abc import ABC
from numbers import Real
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from scipy.ndimage import center_of_mass

from bcdi.graph import graph_utils as gu
from bcdi.utils import utilities as util
from bcdi.utils import validation as valid

module_logger = logging.getLogger(__name__)


def round_sequence_to_int(
    sequence: Union[Tuple[Any, ...], List[Any]]
) -> Tuple[int, ...]:
    """Round a sequence of numbers to integers."""
    if not isinstance(sequence, (tuple, list)):
        raise TypeError(f"Expected a list or tuple, got {type(sequence)}")
    if not all(isinstance(val, Real) for val in sequence):
        raise ValueError("Non-numeric type encountered")
    return tuple(map(lambda x: int(np.rint(x)), sequence))


def zero_pad(
    array: np.ndarray,
    padding_width: np.ndarray = np.zeros(6),
    mask_flag: bool = False,
    debugging: bool = False,
) -> np.ndarray:
    """
    Pad obj with zeros.

    :param array: 3D array to be padded
    :param padding_width: number of zero pixels to padd on each side
    :param mask_flag: set to True to pad with 1, False to pad with 0
    :type mask_flag: bool
    :param debugging: set to True to see plots
    :type debugging: bool
    :return: obj padded with zeros
    """
    valid.valid_ndarray(arrays=array, ndim=3)
    nbz, nby, nbx = array.shape
    if all(x == 0 for x in np.zeros(6)):
        return array

    if debugging:
        gu.multislices_plot(
            array=array,
            sum_frames=False,
            plot_colorbar=True,
            vmin=0,
            vmax=1,
            title="Array before padding",
        )

    if mask_flag:
        newobj = np.ones(
            (
                nbz + padding_width[0] + padding_width[1],
                nby + padding_width[2] + padding_width[3],
                nbx + padding_width[4] + padding_width[5],
            )
        )
    else:
        newobj = np.zeros(
            (
                nbz + padding_width[0] + padding_width[1],
                nby + padding_width[2] + padding_width[3],
                nbx + padding_width[4] + padding_width[5],
            )
        )

    newobj[
        padding_width[0] : padding_width[0] + nbz,
        padding_width[2] : padding_width[2] + nby,
        padding_width[4] : padding_width[4] + nbx,
    ] = array

    if debugging:
        gu.multislices_plot(
            array=newobj,
            sum_frames=False,
            plot_colorbar=True,
            vmin=0,
            vmax=1,
            title="Array after padding",
        )
    return newobj


class CenterFFT(ABC):
    """
    Base class, with methods needing to be overloaded depending on the centering method.

    Frame conventions:
     - axis=0, z downstream, qx in reciprocal space
     - axis=1, y vertical, qz in reciprocal space
     - axis=2, x outboard, qy in reciprocal space

    :params data_shape: shape of the 3D dataset to be centered
    :param binning: binning factor of data in each dimension
    :param preprocessing_binning: additional binning factor due to a previous
     preprocessing step
    :param roi: region of interest of the detector used to generate data.
     [y_start, y_stop, x_start, x_stop]
    :param center_position: position of the determined center
    :param max_symmetrical_window: width of the largest symmetrical window around the
     center_position
    :param bragg_peak: position in pixels of the Bragg peak [z_bragg, y_bragg, x_bragg]
    :param fft_option:
     - 'crop_sym_ZYX': crop the array for FFT requirements, Bragg peak centered
     - 'crop_asym_ZYX': crop the array for FFT requirements without centering the
       Brag peak
     - 'pad_sym_Z_crop_sym_YX': crop detector images (Bragg peak centered) and pad
       the rocking angle based on 'pad_size' (Bragg peak centered)
     - 'pad_sym_Z_crop_asym_YX': pad rocking angle based on 'pad_size'
       (Bragg peak centered) and crop detector (Bragg peak non-centered)
     - 'pad_asym_Z_crop_sym_YX': crop detector images (Bragg peak centered),
       pad the rocking angle without centering the Brag peak
     - 'pad_asym_Z_crop_asym_YX': pad rocking angle and crop detector without centering
       the Bragg peak
     - 'pad_sym_Z': keep detector size and pad/center the rocking angle based on
       'pad_size', Bragg peak centered
     - 'pad_asym_Z': keep detector size and pad the rocking angle without centering
       the Brag peak
     - 'pad_sym_ZYX': pad all dimensions based on 'pad_size', Brag peak centered
     - 'pad_asym_ZYX': pad all dimensions based on 'pad_size' without centering
       the Brag peak
     - 'skip': keep the full dataset

    :param pad_size: user defined output array size [nbz, nby, nbx]
    :param q_values: [qx, qz, qy], each component being a 1D array
    :param logger: a logger instance
    """

    def __init__(
        self,
        data_shape: Tuple[int, int, int],
        binning: Tuple[int, int, int],
        preprocessing_binning: Tuple[int, int, int],
        roi: Tuple[int, int, int, int],
        center_position: Tuple[int, ...],
        max_symmetrical_window: Tuple[int, int, int],
        bragg_peak: Optional[List[int]],
        fft_option: str,
        pad_size: Optional[Tuple[int, int, int]],
        q_values: Optional[Any] = None,
        logger: logging.Logger = module_logger,
    ):
        self.data_shape = data_shape
        self.binning = binning
        self.preprocessing_binning = preprocessing_binning
        self.roi = roi
        self.center_position = center_position
        self.max_symmetrical_window = max_symmetrical_window
        self.bragg_peak = bragg_peak
        self.fft_option = fft_option
        self.pad_size = pad_size
        self.q_values = q_values
        self.logger = logger

        self.pad_width = np.zeros(6, dtype=int)
        self.start_stop_indices = (0, data_shape[0], 0, data_shape[1], 0, data_shape[2])

    @property
    def data_shape(self) -> Tuple[int, int, int]:
        """Shape of the target array."""
        return self._data_shape

    @data_shape.setter
    def data_shape(self, value: Tuple[int, int, int]) -> None:
        if len(value) != 3:
            raise ValueError(f"Only 3D data supported, got {len(value)}D")
        self._data_shape = value

    @property
    def pad_size(self) -> Optional[Tuple[int, int, int]]:
        """User defined shape to which the data should be padded."""
        return self._pad_size

    @pad_size.setter
    def pad_size(self, value: Optional[Tuple[int, int, int]]) -> None:
        if isinstance(value, (list, tuple)):
            if len(value) != len(self.data_shape):
                raise ValueError(
                    f"pad_size should be a list of {len(self.data_shape)} elements"
                )
            if value[0] != util.higher_primes(
                value[0], maxprime=7, required_dividers=(2,)
            ):
                raise ValueError(
                    f"pad_size[0]={value[0]} does not meet FFT requirements"
                )
            if any(not isinstance(val, int) for val in value):
                raise TypeError("indices should be integers")
        elif value is not None:
            raise TypeError(
                f"pad_size should be None or a list of {len(self.data_shape)} elements,"
                f" got {value}"
            )
        self._pad_size = value

    @property
    def start_stop_indices(self) -> Tuple[int, int, int, int, int, int]:
        """Define the indices defining ranges used when cropping the array."""
        return self._start_stop_indices

    @start_stop_indices.setter
    def start_stop_indices(self, value: Tuple[int, int, int, int, int, int]) -> None:
        if not isinstance(value, (list, tuple)):
            raise TypeError("expecting a tuple, got " f"{type(value)}")
        if len(value) != 2 * len(self.data_shape):
            raise ValueError(
                f"expecting a tuple of {2 * len(self.data_shape)} integers, "
                f"got {len(value)} values"
            )
        if any(val < 0 for val in value):
            raise ValueError(f"start indices should be >=0, got {value}")
        if (
            value[1] > self.data_shape[0]
            or value[3] > self.data_shape[1]
            or value[5] > self.data_shape[2]
        ):
            raise ValueError(
                "stop indices should be smaller than the data shape, got " f"{value}"
            )
        if any(not isinstance(val, int) for val in value):
            raise TypeError("indices should be integers")
        self._start_stop_indices = value

    def center_fft(
        self,
        data: np.ndarray,
        mask: Optional[np.ndarray],
        frames_logical: Optional[np.ndarray],
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        np.ndarray,
        Optional[Any],
        Optional[np.ndarray],
    ]:
        """
        Center/pad arrays and modify accordingly related objects.

        :param data: a 3D array
        :param mask: an optional 3D binary array of the same shape as data
        :param frames_logical: array of initial length the number of measured frames.
         In case of padding the length changes. A frame whose index is set to 1 means
         that it is used, 0 means not used, -1 means padded (added) frame.
        :return: the updated data, mask, pad_width, q_values, frames_logical
        """
        self.set_start_stop_indices()
        data = self.crop_array(data)
        if mask is not None:
            mask = self.crop_array(mask)

        self.set_pad_width()
        data = zero_pad(data, padding_width=self.pad_width, mask_flag=False)
        if mask is not None:
            zero_pad(
                mask, padding_width=self.pad_width, mask_flag=True
            )  # mask padded pixels
        self.logger.info(f"FFT box (qx, qz, qy): {self.data_shape}")

        if frames_logical is not None:
            frames_logical = self.update_frames_logical(frames_logical)

        self.update_q_values()

        return data, mask, self.pad_width, self.q_values, frames_logical

    def crop_array(self, array: np.ndarray) -> np.ndarray:
        """Crop an array given stop-start indices."""
        return array[
            self.start_stop_indices[0] : self.start_stop_indices[1],
            self.start_stop_indices[2] : self.start_stop_indices[3],
            self.start_stop_indices[4] : self.start_stop_indices[5],
        ]

    def set_pad_width(self) -> None:
        """Calculate the pad_width parameter depending on the centering method."""
        raise NotImplementedError

    def set_start_stop_indices(self) -> None:
        """Calculate the start-stop indices used for cropping the data."""
        return  # start_stop_indices already set for the general case in __init__

    def update_frames_logical(
        self,
        frames_logical: np.ndarray,
    ) -> np.ndarray:
        """Update the logical array depending on the processing applied to the data."""
        raise NotImplementedError

    def update_q_values(self) -> None:
        """Update q values depending on the processing applied to the data."""
        if self.q_values is None:
            return
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
    """
    Factory class to instantiate the corrent centering child class.

    :param data: the 3D data array
    :param binning: binning factor of data in each dimension
    :param preprocessing_binning: additional binning factor due to a previous
     preprocessing step
    :param roi: region of interest of the detector used to generate data.
     [y_start, y_stop, x_start, x_stop]
    :param bragg_peak: position in pixels of the Bragg peak [z_bragg, y_bragg, x_bragg]
    :param fft_option:
     - 'crop_sym_ZYX': crop the array for FFT requirements, Bragg peak centered
     - 'crop_asym_ZYX': crop the array for FFT requirements without centering the
       Brag peak
     - 'pad_sym_Z_crop_sym_YX': crop detector images (Bragg peak centered) and pad
       the rocking angle based on 'pad_size' (Bragg peak centered)
     - 'pad_sym_Z_crop_asym_YX': pad rocking angle based on 'pad_size'
       (Bragg peak centered) and crop detector (Bragg peak non-centered)
     - 'pad_asym_Z_crop_sym_YX': crop detector images (Bragg peak centered),
       pad the rocking angle without centering the Brag peak
     - 'pad_asym_Z_crop_asym_YX': pad rocking angle and crop detector without centering
       the Bragg peak
     - 'pad_sym_Z': keep detector size and pad/center the rocking angle based on
       'pad_size', Bragg peak centered
     - 'pad_asym_Z': keep detector size and pad the rocking angle without centering
       the Brag peak
     - 'pad_sym_ZYX': pad all dimensions based on 'pad_size', Brag peak centered
     - 'pad_asym_ZYX': pad all dimensions based on 'pad_size' without centering
       the Brag peak
     - 'skip': keep the full dataset

    :param pad_size: user defined output array size [nbz, nby, nbx]
    :param centering_method: method used to determine the location of the Bragg peak:
     'max', 'com' (center of mass), or 'max_com' (max along the first axis, center of
     mass in the detector plane)
    :param q_values: [qx, qz, qy], each component being a 1D array
    :param logger: a logger instance
    """

    def __init__(
        self,
        data: np.ndarray,
        binning: Tuple[int, int, int],
        preprocessing_binning: Tuple[int, int, int],
        roi: Tuple[int, int, int, int],
        bragg_peak: Optional[List[int]],
        fft_option: str = "crop_asymmetric_ZYX",
        pad_size: Optional[Tuple[int, int, int]] = None,
        centering_method: str = "max",
        q_values: Optional[List[np.ndarray]] = None,
        logger: logging.Logger = module_logger,
    ):
        self.data_shape = data.shape
        self.binning = binning
        self.preprocessing_binning = preprocessing_binning
        self.roi = roi
        self.bragg_peak = bragg_peak
        self.fft_option = fft_option
        self.pad_size = pad_size
        self.q_values = q_values
        self.logger = logger

        self.center_position = self.find_center(data=data, method=centering_method)
        self.log_q_values_at_center(method=centering_method)
        self.max_symmetrical_window = self.get_max_symmetrical_box(data=data)
        self.check_center_position()

    @property
    def data_shape(self):
        """Store the shape of the 3D dataset."""
        return self._data_shape

    @data_shape.setter
    def data_shape(self, value):
        if not isinstance(value, tuple) or len(value) != 3:
            raise ValueError(f"Only 3D data is supported, got {len(value)}")
        self._data_shape = value

    def get_max_symmetrical_box(self, data: np.ndarray) -> Tuple[int, int, int]:
        """
        Calculate the largest symmetrical box around the center.

        :param data: the 3D intensity dataset
        :return: the width of the largest symmetrical box as a tuple of three positive
         integers
        """
        nbz, nby, nbx = np.shape(data)
        iz0, iy0, ix0 = self.center_position
        return (
            abs(2 * min(iz0, nbz - iz0)),
            2 * min(iy0, nby - iy0),
            abs(2 * min(ix0, nbx - ix0)),
        )

    def check_center_position(self) -> None:
        """Check if the found center position is not at the edge of the data array."""
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

    def find_center(self, data: np.ndarray, method: str) -> Tuple[int, ...]:
        """
        Find the center (ideally the Bragg peak) of the dataset.

        :param data: the 3D intensity dataset
        :param method: "max", "com", "max_com". It is overruled by bragg_peak if this
         parameter is not None.
        :return: the position of the found center as a tuple of three positive integers
        """
        if self.bragg_peak:
            if len(self.bragg_peak) != 3:
                raise ValueError("bragg_peak should be a list of 3 integers")
            self.logger.info(
                "Peak intensity position on the full detector provided: "
                f"({self.bragg_peak})"
            )
            y0 = (self.bragg_peak[1] - self.roi[0]) / (
                self.preprocessing_binning[1] * self.binning[1]
            )
            x0 = (self.bragg_peak[2] - self.roi[2]) / (
                self.preprocessing_binning[2] * self.binning[2]
            )
            return round_sequence_to_int((self.bragg_peak[0], y0, x0))

        if method == "max":
            return round_sequence_to_int(
                np.unravel_index(abs(data).argmax(), data.shape)
            )
        if method == "com":
            return round_sequence_to_int(center_of_mass(data))
        # 'max_com'
        position = list(np.unravel_index(abs(data).argmax(), data.shape))
        position[1:] = center_of_mass(data[position[0], :, :])
        return round_sequence_to_int(position)

    def get_centering_instance(self) -> CenterFFT:
        """Return the correct centering instance depending on the FFT option."""
        if self.fft_option == "crop_sym_ZYX":
            centering_class = CenterFFTCropSymZYX
        elif self.fft_option == "crop_asym_ZYX":
            centering_class = CenterFFTCropAsymZYX  # type: ignore
        elif self.fft_option == "pad_sym_Z_crop_sym_YX":
            centering_class = CenterFFTPadSymZCropSymYX  # type: ignore
        elif self.fft_option == "pad_sym_Z_crop_asym_YX":
            centering_class = CenterFFTPadSymZCropAsymYX  # type: ignore
        elif self.fft_option == "pad_asym_Z_crop_sym_YX":
            centering_class = CenterFFTPadAsymZCropSymYX  # type: ignore
        elif self.fft_option == "pad_asym_Z_crop_asym_YX":
            centering_class = CenterFFTPadAsymZCropAsymYX  # type: ignore
        elif self.fft_option == "pad_sym_Z":
            centering_class = CenterFFTPadSymZ  # type: ignore
        elif self.fft_option == "pad_asym_Z":
            centering_class = CenterFFTPadAsymZ  # type: ignore
        elif self.fft_option == "pad_sym_ZYX":
            centering_class = CenterFFTPadSymZYX  # type: ignore
        elif self.fft_option == "pad_asym_ZYX":
            centering_class = CenterFFTPadAsymZYX  # type: ignore
        elif self.fft_option == "skip":
            centering_class = SkipCentering  # type: ignore
        else:
            raise ValueError(f"Incorrect value {self.fft_option} for 'fft_option'")

        return centering_class(
            data_shape=self.data_shape,
            binning=self.binning,
            preprocessing_binning=self.preprocessing_binning,
            roi=self.roi,
            center_position=self.center_position,
            max_symmetrical_window=self.max_symmetrical_window,
            bragg_peak=self.bragg_peak,
            fft_option=self.fft_option,
            pad_size=self.pad_size,
            logger=self.logger,
            q_values=self.q_values,
        )

    def log_q_values_at_center(self, method: str) -> None:
        """Log some message about q values at the center found by the method."""
        z0, y0, x0 = self.center_position
        if self.bragg_peak is not None:
            self.logger.info(
                "Peak intensity position with detector ROI and binning in the "
                f"detector plane: ({z0, y0, x0})"
            )
            return

        if method == "max":
            text = "Max"
        elif method == "com":
            text = "Center of mass"
        else:
            text = "Max_com"

        if self.q_values is None:
            self.logger.info(f"{text} at pixel (Z, Y, X): ({z0, y0, x0})")
        else:
            self.logger.info(
                f"{text} at (qx, qz, qy): {self.q_values[0][z0]:.5f}, "
                f"{self.q_values[1][y0]:.5f}, {self.q_values[2][x0]:.5f}"
            )


class CenterFFTCropSymZYX(CenterFFT):
    """Crop the data so that the peak is centered in XYZ."""

    def set_pad_width(self) -> None:
        self.pad_width = np.zeros(6, dtype=int)

    def set_start_stop_indices(self) -> None:
        iz0, iy0, ix0 = self.center_position

        # crop rocking angle and detector, Bragg peak centered
        nz1, ny1, nx1 = util.smaller_primes(
            self.max_symmetrical_window, maxprime=7, required_dividers=(2,)
        )
        self.start_stop_indices = (
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
        if self.start_stop_indices[0] > 0:  # if 0, the first frame is used
            frames_logical[0 : self.start_stop_indices[0]] = 0
        if self.start_stop_indices[1] < self.data_shape[0]:
            # if nbz, the last frame is used
            frames_logical[self.start_stop_indices[1] :] = 0
        return frames_logical


class CenterFFTCropAsymZYX(CenterFFT):
    """Crop the data without constraint on the peak position."""

    def set_pad_width(self) -> None:
        self.pad_width = np.zeros(6, dtype=int)

    def set_start_stop_indices(self) -> None:
        nbz, nby, nbx = self.data_shape
        # crop rocking angle and detector without centering the Bragg peak
        nz1, ny1, nx1 = util.smaller_primes(
            (nbz, nby, nbx), maxprime=7, required_dividers=(2,)
        )
        self.start_stop_indices = (
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
        if self.start_stop_indices[0] > 0:  # if 0, the first frame is used
            frames_logical[0 : self.start_stop_indices[0]] = 0
        if self.start_stop_indices[1] < self.data_shape[0]:
            # if nbz, the last frame is used
            frames_logical[self.start_stop_indices[1] :] = 0
        return frames_logical


class CenterFFTPadSymZCropSymYX(CenterFFT):
    """
    Pad the data along Z, crop it in Y and X.

    The peak is centered in ZYX.
    """

    def set_pad_width(self) -> None:
        if self.pad_size is None:
            self.pad_width = np.zeros(6, dtype=int)
            return

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

    def set_start_stop_indices(self) -> None:
        _, iy0, ix0 = self.center_position

        ny1, nx1 = util.smaller_primes(
            self.max_symmetrical_window[1:], maxprime=7, required_dividers=(2,)
        )
        self.start_stop_indices = (
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
        temp_frames[self.pad_width[0] : self.pad_width[0] + self.data_shape[0]] = (
            frames_logical
        )
        return temp_frames

    def update_q_values(self) -> None:
        if self.q_values is None:
            return
        if self.pad_size is None:
            raise ValueError(
                "pad_size should be a sequence of three integers, got None"
            )
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
    """
    Pad the data along Z, crop it in Y and X.

    The peak is centered in Z only.
    """

    def set_pad_width(self) -> None:
        if self.pad_size is None:
            self.pad_width = np.zeros(6, dtype=int)
            return

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

    def set_start_stop_indices(self) -> None:
        ny1, nx1 = util.smaller_primes(
            self.max_symmetrical_window[1:], maxprime=7, required_dividers=(2,)
        )
        self.start_stop_indices = (
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
        temp_frames[self.pad_width[0] : self.pad_width[0] + self.data_shape[0]] = (
            frames_logical
        )
        return temp_frames

    def update_q_values(self) -> None:
        if self.q_values is None:
            return
        if self.pad_size is None:
            raise ValueError(
                "pad_size should be a sequence of three integers, got None"
            )
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
    """
    Pad the data along Z, crop it in Y and X.

    The peak is centered in Y and X only.
    """

    def set_pad_width(self) -> None:
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

    def set_start_stop_indices(self) -> None:
        # crop detector (Bragg peak centered)
        ny1, nx1 = util.smaller_primes(
            self.max_symmetrical_window[1:], maxprime=7, required_dividers=(2,)
        )
        _, iy0, ix0 = self.center_position
        self.start_stop_indices = (
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
        temp_frames[self.pad_width[0] : self.pad_width[0] + self.data_shape[0]] = (
            frames_logical
        )
        return temp_frames

    def update_q_values(self) -> None:
        if self.q_values is None:
            return
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
    """
    Pad the data along Z, crop it in Y and X.

    The peak is not centered.
    """

    def set_pad_width(self) -> None:
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

    def set_start_stop_indices(self) -> None:
        # crop detector without centering the Bragg peak
        ny1, nx1 = util.smaller_primes(
            self.data_shape[1:], maxprime=7, required_dividers=(2,)
        )
        self.start_stop_indices = (
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
        temp_frames[self.pad_width[0] : self.pad_width[0] + self.data_shape[0]] = (
            frames_logical
        )
        return temp_frames

    def update_q_values(self) -> None:
        if self.q_values is None:
            return
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
    """
    Pad the data along Z.

    The peak is centered in Z.
    """

    def set_pad_width(self) -> None:
        if self.pad_size is None:
            raise ValueError(
                "pad_size should be a sequence of three integers, got None"
            )
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

    def update_frames_logical(
        self,
        frames_logical: np.ndarray,
    ) -> np.ndarray:
        temp_frames = -1 * np.ones(self.data_shape[0])
        temp_frames[self.pad_width[0] : self.pad_width[0] + self.data_shape[0]] = (
            frames_logical
        )
        return temp_frames

    def update_q_values(self) -> None:
        if self.q_values is None:
            return
        if self.pad_size is None:
            raise ValueError(
                "pad_size should be a sequence of three integers, got None"
            )
        dqx = self.q_values[0][1] - self.q_values[0][0]
        qx0 = self.q_values[0][0] - self.pad_width[0] * dqx
        self.q_values[0] = qx0 + np.arange(self.pad_size[0]) * dqx


class CenterFFTPadAsymZ(CenterFFT):
    """
    Pad the data along Z.

    The peak is not centered.
    """

    def set_pad_width(self) -> None:
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

    def update_frames_logical(
        self,
        frames_logical: np.ndarray,
    ) -> np.ndarray:
        temp_frames = -1 * np.ones(self.data_shape[0])
        temp_frames[self.pad_width[0] : self.pad_width[0] + self.data_shape[0]] = (
            frames_logical
        )
        return temp_frames

    def update_q_values(self) -> None:
        if self.q_values is None:
            return
        nz1 = util.higher_primes(self.data_shape[0], maxprime=7, required_dividers=(2,))
        dqx = self.q_values[0][1] - self.q_values[0][0]
        qx0 = self.q_values[0][0] - self.pad_width[0] * dqx
        self.q_values[0] = qx0 + np.arange(nz1) * dqx


class CenterFFTPadSymZYX(CenterFFT):
    """
    Pad the data along Z, Y and X.

    The peak is centered.
    """

    def set_pad_width(self) -> None:
        if self.pad_size is None:
            raise ValueError(
                "pad_size should be a sequence of three integers, got None"
            )
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
            list(map(lambda value: max(value, 0), pad_width)), dtype=int
        )

    def update_frames_logical(
        self,
        frames_logical: np.ndarray,
    ) -> np.ndarray:
        temp_frames = -1 * np.ones(self.data_shape[0])
        temp_frames[self.pad_width[0] : self.pad_width[0] + self.data_shape] = (
            frames_logical
        )
        return temp_frames

    def update_q_values(self) -> None:
        if self.q_values is None:
            return
        if self.pad_size is None:
            raise ValueError(
                "pad_size should be a sequence of three integers, got None"
            )
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
    """
    Pad the data along Z, Y and X.

    The peak is not centered.
    """

    def set_pad_width(self) -> None:
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

    def update_frames_logical(
        self,
        frames_logical: np.ndarray,
    ) -> np.ndarray:
        temp_frames = -1 * np.ones(self.data_shape[0])
        temp_frames[self.pad_width[0] : self.pad_width[0] + self.data_shape[0]] = (
            frames_logical
        )
        return temp_frames

    def update_q_values(self) -> None:
        if self.q_values is None:
            return
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
    """Skip centering."""

    def set_pad_width(self) -> None:
        self.pad_width = np.zeros(6, dtype=int)

    def update_frames_logical(
        self,
        frames_logical: np.ndarray,
    ) -> np.ndarray:
        return frames_logical

    def update_q_values(self) -> None:
        return  # nothing to do in that case
