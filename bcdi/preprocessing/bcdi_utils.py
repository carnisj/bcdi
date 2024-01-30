# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, jerome.carnis@esrf.fr
"""Functions related to BCDI data preprocessing, before phase retrieval."""

import logging
import pathlib
from numbers import Real
from operator import mul
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xrayutilities as xu
from scipy.interpolate import interp1d
from scipy.ndimage import center_of_mass

from bcdi.experiment import loader
from bcdi.experiment.detector import Detector
from bcdi.graph import graph_utils as gu
from bcdi.preprocessing.center_fft import CenteringFactory
from bcdi.utils import utilities as util
from bcdi.utils import validation as valid

if TYPE_CHECKING:
    from bcdi.experiment.setup import Setup

module_logger = logging.getLogger(__name__)


class PeakFinder:
    """
    Find the Bragg peak and optionally fit the rocking curve and plot results.

    The data is expected to be stacked into a 3D array, the first axis corresponding to
    the rocking angle and axes 1 and 2 to the detector plane (vertical, horizontal).

    :param array: the detector data
    :param binning: the binning factor of array relative to the unbinned detector
    :param region_of_interest: the region of interest applied to build array out of the
     full detector
    :param peak_method: peak searching method, among "max", "com", "max_com".
    :param kwargs:
     - 'logger': an optional logger
     - "user_defined_peak": [z, y, x] Bragg peak position defined by the user

    """

    PEAK_METHODS = {"max", "com", "max_com", "user", "skip"}

    def __init__(
        self,
        array: np.ndarray,
        region_of_interest: Optional[List[int]] = None,
        binning: Optional[List[int]] = None,
        peak_method: str = "max_com",
        **kwargs,
    ):
        self.array = array
        self.region_of_interest = (
            [0, self.array.shape[1], 0, self.array.shape[2]]
            if region_of_interest is None
            else region_of_interest
        )
        self.binning = [1, 1, 1] if binning is None else binning
        self.peak_method = peak_method
        self.logger: logging.Logger = kwargs.get("logger", module_logger)

        self._peaks = self.find_peak(kwargs.get("user_defined_peak"))
        self._rocking_curve: Optional[np.ndarray] = None
        self._detector_data_at_peak: Optional[np.ndarray] = self.array[
            self._roi_center[0], :, :
        ]
        self._tilt_values: Optional[np.ndarray] = None
        self._interp_tilt_values: Optional[np.ndarray] = None
        self._interp_rocking_curve: Optional[np.ndarray] = None
        self._interp_fwhm: Optional[float] = None
        self._tilt_value_at_peak: Optional[float] = None

    @property
    def binning(self) -> List[int]:
        """Binning factor of the array pixels, one number per array axis."""
        return self._binning

    @binning.setter
    def binning(self, value: List[int]) -> None:
        valid.valid_container(
            value,
            container_types=(tuple, list, np.ndarray),
            item_types=int,
            length=self.array.ndim,
            allow_none=False,
            name="binning",
        )
        self._binning = value

    @property
    def bragg_peak(self) -> Tuple[int, int, int]:
        """Export the retrieved Bragg peak position."""
        return self.peaks[self.peak_method]

    @property
    def metadata(self) -> Dict[str, Any]:
        """Export the retrieved peaks and fitted rocking curve."""
        return {
            "bragg_peak": self.peaks[self.peak_method],
            "peaks": self.peaks,
            "rocking_curve": self._rocking_curve,
            "detector_data_at_peak": self._detector_data_at_peak,
            "tilt_values": self._tilt_values,
            "interp_tilt_values": self._interp_tilt_values,
            "interp_rocking_curve": self._interp_rocking_curve,
            "interp_fwhm": self._interp_fwhm,
            "tilt_value_at_peak": self._tilt_value_at_peak,
        }

    @property
    def peak_method(self) -> str:
        """Localize the peak using this method."""
        return self._peak_method

    @peak_method.setter
    def peak_method(self, value: str) -> None:
        if value not in self.PEAK_METHODS:
            raise ValueError(f"allowed peak methods {self.PEAK_METHODS}, got {value}")
        self._peak_method = value

    @property
    def peaks(self) -> Dict[str, Tuple[int, int, int]]:
        """Position of the detected peaks in the full, unbinned detector frame."""
        return self._peaks

    @property
    def region_of_interest(self) -> List[int]:
        """
        Region of interest used when loading the detector images.

        [y_start, y_stop, x_start, x_stop]
        """
        return self._region_of_interest

    @region_of_interest.setter
    def region_of_interest(self, value: List[int]) -> None:
        valid.valid_container(
            value,
            container_types=(tuple, list, np.ndarray),
            item_types=int,
            length=4,
            allow_none=True,
            name="region_of_interest",
        )
        self._region_of_interest = value

    def find_peak(
        self, user_defined_peak: Optional[Tuple[int, int, int]] = None
    ) -> Dict[str, Tuple[int, int, int]]:
        """
        Find the position of the Bragg peak using three different metrics.

        Peak-searching methods:
         - "max": maximum of the modulus
         - "com": center of mass of the modulus
         - "max_com": "max" along the first axis, "com" along the other axes

        :param user_defined_peak: [z, y, x] Bragg peak position defined by the user
        """
        index_max = np.unravel_index(abs(self.array).argmax(), self.array.shape)
        position_max = [int(val) for val in index_max]
        self.logger.info(
            f"Max at: {position_max}, value = {int(self.array[index_max])}"
        )

        position_com = center_of_mass(self.array)
        position_com = tuple(map(lambda x: int(np.rint(x)), position_com))
        self.logger.info(
            f"Center of mass at: {position_com}, "
            f"value = {int(self.array[position_com])}"
        )

        index_max_com = list(
            np.unravel_index(abs(self.array).argmax(), self.array.shape)
        )
        index_max_com[1:] = center_of_mass(self.array[index_max_com[0], :, :])
        position_max_com = tuple(map(lambda x: int(np.rint(x)), index_max_com))
        self.logger.info(
            f"MaxCom at (z, y, x): {position_max_com}, "
            f"value = {int(self.array[position_max_com])}"
        )
        if user_defined_peak is not None:
            return {
                "max": self.get_indices_full_detector(list(position_max)),
                "com": self.get_indices_full_detector(list(position_com)),
                "max_com": self.get_indices_full_detector(list(position_max_com)),
                "user": user_defined_peak,
            }
        return {
            "max": self.get_indices_full_detector(list(position_max)),
            "com": self.get_indices_full_detector(list(position_com)),
            "max_com": self.get_indices_full_detector(list(position_max_com)),
        }

    def get_indices_cropped_binned_detector(self, position: List[int]) -> List[int]:
        """Calculate the position in the cropped, binned detector frame."""
        cropped_position = self._offset(position, frame="region_of_interest")
        return self._bin(list(cropped_position))

    def get_indices_full_detector(self, position: List[int]) -> Tuple[int, int, int]:
        """Calculate the position in the unbinned, full detector frame."""
        return self._offset(self._unbin(position), frame="full_detector")

    def fit_rocking_curve(self, tilt_values: Optional[np.ndarray] = None):
        """
        Calculate and plot the rocking curve, optionally save the figure.

        :param tilt_values: values of the tilt angle during the rocking curve.
        """
        self._get_rocking_curve()
        self._fit_rocking_curve(tilt_values=tilt_values)

    def plot_peaks(self, savedir: Optional[str] = None) -> None:
        """
        Plot the detected peak position by several methods, optionally save the figure.

        It plots the peaks in the binned, cropped detector frame.

        Peak-searching methods:
         - "max": maximum of the modulus
         - "com": center of mass of the modulus
         - "max_com": "max" along the first axis, "com" along the other axes

        :param savedir: folder where to save the figure
        """
        plt.ion()

        methods = {
            "max": (
                self.get_indices_cropped_binned_detector(list(self.peaks["max"])),
                "k",
            ),
            "com": (
                self.get_indices_cropped_binned_detector(list(self.peaks["com"])),
                "g",
            ),
            "max_com": (
                self.get_indices_cropped_binned_detector(list(self.peaks["max_com"])),
                "b",
            ),
        }
        indices = {0: [2, 1], 1: [2, 0], 2: [1, 0]}
        fig, axes, _ = gu.multislices_plot(
            self.array,
            sum_frames=True,
            scale="log",
            plot_colorbar=True,
            vmin=0,
            vmax=6,
            title="data",
        )
        for ax, ind in indices.items():
            for method, values in methods.items():
                axes[ax].scatter(
                    values[0][ind[0]],
                    values[0][ind[1]],
                    color=values[1],
                    marker="1",
                    alpha=0.7,
                    linewidth=2,
                    label=method,
                )
            axes[ax].legend()
        plt.pause(0.1)
        if savedir is not None:
            path = pathlib.Path(savedir) / "centering_method.png"
            fig.savefig(path)
        plt.close(fig)
        plt.ioff()

    def plot_rocking_curve(self, savedir: Optional[str] = None) -> None:
        """
        Plot the rocking curve, optionally save the figure.

        :param savedir: folder where to save the figure
        """
        rocking_curve = self.metadata.get("rocking_curve")
        if rocking_curve is None:
            self.logger.info("'rocking_curve' is None, nothing to plot")
            return

        tilt_values = self.metadata.get("tilt_values")
        if tilt_values is None or len(tilt_values) != len(rocking_curve):
            tilt_values = np.arange(self.array.shape[0])
            x_label = "Frame number"
        else:
            x_label = "Rocking angle (deg)"

        interp_tilt = self.metadata.get("interp_tilt_values")
        interp_curve = self.metadata.get("interp_rocking_curve")

        plt.ion()
        fig, (ax0, ax1) = plt.subplots(2, 1, sharex="col", figsize=(10, 5))
        ax0.plot(tilt_values, rocking_curve, ".")
        if interp_tilt is not None and interp_curve is not None:
            ax0.plot(interp_tilt, interp_curve)
            legend = ["data", "interpolation"]
        else:
            legend = ["data"]
        ax0.axvline(tilt_values[self._roi_center[0]], color="r", alpha=0.7, linewidth=1)
        ax0.set_ylabel("Integrated intensity")
        ax0.legend(legend)
        ax0.set_title("Rocking curve")
        ax1.plot(tilt_values, np.log10(rocking_curve), ".")
        if interp_tilt is not None and interp_curve is not None:
            ax1.plot(interp_tilt, np.log10(interp_curve))
        ax1.axvline(tilt_values[self._roi_center[0]], color="r", alpha=0.7, linewidth=1)
        ax1.set_xlabel(x_label)
        ax1.set_ylabel("Log(integrated intensity)")
        ax1.legend(legend)
        plt.pause(0.1)
        if savedir is not None:
            path = pathlib.Path(savedir) / "rocking_curve.png"
            fig.savefig(path)
        plt.close(fig)

    @property
    def _roi_center(self) -> Tuple[int, int, int]:
        """Position of the Bragg peak in the cropped and binned detector frame."""
        bragg_peak = self.bragg_peak
        return (
            bragg_peak[0],
            (bragg_peak[1] - self.region_of_interest[0]) // self.binning[1],
            (bragg_peak[2] - self.region_of_interest[2]) // self.binning[2],
        )

    def _fit_rocking_curve(self, tilt_values) -> None:
        """Fit the rocking curve and optionally tilt values by cubic interpolation."""
        self._tilt_values = tilt_values
        self._tilt_value_at_peak = (
            tilt_values[self._roi_center[0]] if tilt_values is not None else None
        )
        rocking_curve = self.metadata.get("rocking_curve")
        if rocking_curve is None:
            self.logger.info("'rocking_curve' is None, nothing to fit")
            return
        x_axis = (
            tilt_values if tilt_values is not None else np.arange(len(rocking_curve))
        )
        if len(x_axis) != len(rocking_curve):
            self.logger.warning(
                "tilt_values and rocking curve don't have the same length (hint: did "
                "you reload cropped data?)"
            )
            return
        interpolation = interp1d(x_axis, rocking_curve, kind="cubic")
        interp_points = 5 * self.array.shape[0]
        interp_tilt = np.linspace(x_axis.min(), x_axis.max(), interp_points)
        interp_curve = interpolation(interp_tilt)
        interp_fwhm = (
            len(np.argwhere(interp_curve >= interp_curve.max() / 2))
            * (x_axis.max() - x_axis.min())
            / (interp_points - 1)
        )
        self.logger.info(f"FWHM by interpolation: {interp_fwhm:.3f} deg")
        self._interp_tilt_values = interp_tilt
        self._interp_rocking_curve = interp_curve
        self._interp_fwhm = interp_fwhm

    def _get_rocking_curve(
        self,
    ) -> None:
        """Integrate the intensity in the detector plane during a rocking curve."""
        bragg_peak = self.peaks[self.peak_method]
        if bragg_peak is None:
            raise ValueError(f"Bragg peak not detected with method {self.peak_method}")

        self._rocking_curve = self.array.sum(axis=(1, 2))

    def _offset(self, peak: List[int], frame: str) -> Tuple[int, int, int]:
        """
        Calculate the peak position with an offset.

        The offset is added or subtracted dependengin on the target detector frame (full
        frame or frame cropped to a certain region of interest).

        :param peak: position of the peak
        :param frame: "full_detector" to provide the peak position in the full detector,
         "region_of_interest" to provide the peak position in the cropped frame.
        """
        if frame not in {"full_detector", "region_of_interest"}:
            raise ValueError(
                "allowed values 'full_detector' and 'region_of_interest'"
                f"got '{frame}'"
            )
        sign = {"full_detector": 1, "region_of_interest": -1}
        return (
            peak[0],
            peak[1] + sign[frame] * self.region_of_interest[0],
            peak[2] + sign[frame] * self.region_of_interest[2],
        )

    def _bin(self, peak: List[int]) -> List[int]:
        """Calculate the peak position in the binned detector frame."""
        return [a // b for a, b in zip(peak, self.binning)]

    def _unbin(self, peak: List[int]) -> List[int]:
        """Calculate the peak position in the unbinned detector frame."""
        return [a * b for a, b in zip(peak, self.binning)]


def center_fft(
    data: np.ndarray,
    mask: np.ndarray,
    detector: Detector,
    frames_logical: np.ndarray,
    centering: str = "max",
    fft_option: str = "crop_asymmetric_ZYX",
    **kwargs,
):
    """
    Center and crop/pad the dataset depending on user parameters.

    :param data: the 3D data array
    :param mask: the 3D mask array
    :param detector: an instance of the class Detector
    :param frames_logical: array of initial length the number of measured frames.
     In case of padding the length changes. A frame whose index is set to 1 means
     that it is used, 0 means not used, -1 means padded (added) frame.
    :param centering: method used to determine the location of the Bragg peak: 'max',
     'com' (center of mass), or 'max_com' (max along the first axis, center of mass in
     the detector plane)
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

    :param kwargs:
     - 'fix_bragg': user-defined position in pixels of the Bragg peak
       [z_bragg, y_bragg, x_bragg]
     - 'pad_size': user defined output array size [nbz, nby, nbx]
     - 'q_values': [qx, qz, qy], each component being a 1D array
     - 'logger': an optional logger

    :return:
     - updated data, mask (and q_values if provided, [] otherwise)
     - pad_width = [z0, z1, y0, y1, x0, x1] number of pixels added at each end of the
       original data
     - updated frames_logical

    """
    logger = kwargs.get("logger", module_logger)

    valid.valid_ndarray(arrays=(data, mask), ndim=3)
    # check and load kwargs
    valid.valid_kwargs(
        kwargs=kwargs,
        allowed_kwargs={"bragg_peak", "logger", "pad_size", "q_values"},
    )

    centering_fft = CenteringFactory(
        data=data,
        binning=detector.binning,
        preprocessing_binning=detector.preprocessing_binning,
        roi=detector.roi,
        bragg_peak=kwargs.get("bragg_peak"),
        fft_option=fft_option,
        pad_size=kwargs.get("pad_size"),
        centering_method=centering,
        logger=logger,
        q_values=kwargs.get("q_values", []),
    ).get_centering_instance()

    return centering_fft.center_fft(
        data=data,
        mask=mask,
        frames_logical=frames_logical,
    )


def find_bragg(
    array: np.ndarray,
    binning: Optional[List[int]] = None,
    roi: Optional[List[int]] = None,
    peak_method: str = "max_com",
    tilt_values: Optional[np.ndarray] = None,
    savedir: Optional[str] = None,
    plot_fit: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    Find the Bragg peak position.

    Optionally, fit the rocking curve and plot results.

    :param array: the detector data
    :param binning: the binning factor of array relative to the unbinned detector
    :param roi: the region of interest applied to build array out of the
     full detector
    :param peak_method: peak searching method, among "max", "com", "max_com".
    :param tilt_values: the angular values of the motor during the rocking curve
    :param savedir: where to save the plots
    :param plot_fit: if True, will plot results and fit the rocking curve
    :param kwargs:
     - "logger": an optional logger
     - "user_defined_peak": [z, y, x] Bragg peak position defined by the user

    :return: the metadata with the results of the peak search and the fit.
    """
    valid.valid_kwargs(
        kwargs=kwargs,
        allowed_kwargs={"logger", "user_defined_peak"},
    )
    logger: logging.Logger = kwargs.get("logger", module_logger)
    peakfinder = PeakFinder(
        array=array,
        region_of_interest=roi,
        binning=binning,
        peak_method=peak_method,
        user_defined_peak=kwargs.get("user_defined_peak"),
        logger=logger,
    )

    if peakfinder.metadata["bragg_peak"] is None:
        raise ValueError("The position of the Bragg peak is undefined.")
    logger.info(
        "Bragg peak (full unbinned roi) at: " f"{peakfinder.metadata['bragg_peak']}"
    )

    if plot_fit:
        peakfinder.fit_rocking_curve(tilt_values=tilt_values)
        peakfinder.plot_peaks(savedir=savedir)
        peakfinder.plot_rocking_curve(savedir=savedir)
    return peakfinder.metadata


def grid_bcdi_labframe(
    data,
    mask,
    detector,
    setup,
    align_q=False,
    reference_axis=(0, 1, 0),
    debugging=False,
    **kwargs,
):
    """
    Interpolate BCDI reciprocal space data using a linearized transformation matrix.

    The resulting (qx, qy, qz) are in the laboratory frame (qx downstrean,
    qz vertical up, qy outboard).

    :param data: the 3D data, already binned in the detector frame
    :param mask: the corresponding 3D mask
    :param detector: an instance of the class Detector
    :param setup: instance of the Class experiment_utils.Setup()
    :param align_q: boolean, if True the data will be rotated such that q is along
     reference_axis, and q values will be calculated in the pseudo crystal frame.
    :param reference_axis: 3D vector along which q will be aligned, expressed in an
     orthonormal frame x y z
    :param debugging: set to True to see plots
    :param kwargs:

     - 'cmap': str, name of the colormap
     - 'fill_value': tuple of two real numbers, fill values to use for pixels outside
       of the interpolation range. The first value is for the data, the second for the
       mask. Default is (0, 0)
     - 'logger': an optional logger

    :return:

     - the data interpolated in the laboratory frame
     - the mask interpolated in the laboratory frame
     - a tuple of three 1D vectors of q values (qx, qz, qy)
     - a numpy array of shape (3, 3): transformation matrix from the detector
       frame to the laboratory/crystal frame

    """
    logger = kwargs.get("logger", module_logger)
    valid.valid_ndarray(arrays=(data, mask), ndim=3)
    # check and load kwargs
    valid.valid_kwargs(
        kwargs=kwargs,
        allowed_kwargs={"cmap", "fill_value", "logger", "reference_axis"},
    )
    cmap = kwargs.get("cmap", "turbo")
    fill_value = kwargs.get("fill_value", (0, 0))
    valid.valid_container(
        fill_value,
        container_types=(tuple, list, np.ndarray),
        length=2,
        item_types=Real,
        name="fill_value",
    )

    # check some parameters
    if setup.rocking_angle == "energy":
        raise NotImplementedError(
            "Geometric transformation not yet implemented for energy scans"
        )
    valid.valid_item(align_q, allowed_types=bool, name="align_q")
    valid.valid_container(
        reference_axis,
        container_types=(tuple, list, np.ndarray),
        length=3,
        item_types=Real,
        name="reference_axis",
    )
    reference_axis = np.array(reference_axis)

    # grid the data
    logger.info(
        "Gridding the data using the linearized matrix, "
        "the result will be in the laboratory frame"
    )
    string = "linmat_reciprocal_space_"
    (interp_data, interp_mask), q_values, transfer_matrix = setup.ortho_reciprocal(
        arrays=(data, mask),
        verbose=True,
        debugging=debugging,
        fill_value=fill_value,
        align_q=align_q,
        reference_axis=reference_axis,
        scale=("log", "linear"),
        title=("data", "mask"),
    )
    qx, qz, qy = q_values

    # check for Nan
    interp_mask[np.isnan(interp_data)] = 1
    interp_data[np.isnan(interp_data)] = 0
    interp_mask[np.isnan(interp_mask)] = 1
    # set the mask as an array of integers, 0 or 1
    interp_mask[np.nonzero(interp_mask)] = 1
    interp_mask = interp_mask.astype(int)

    # apply the mask to the data
    interp_data[np.nonzero(interp_mask)] = 0

    # save plots of the gridded data
    final_binning = (
        detector.preprocessing_binning[0] * detector.binning[0],
        detector.preprocessing_binning[1] * detector.binning[1],
        detector.preprocessing_binning[2] * detector.binning[2],
    )

    numz, numy, numx = interp_data.shape
    plot_comment = (
        f"_{numz}_{numy}_{numx}_"
        f"{final_binning[0]}_{final_binning[1]}_{final_binning[2]}.png"
    )

    max_z = interp_data.sum(axis=0).max()
    fig, _, _ = gu.contour_slices(
        interp_data,
        (qx, qz, qy),
        sum_frames=True,
        title="Regridded data",
        levels=np.linspace(0, np.ceil(np.log10(max_z)), 150, endpoint=True),
        plot_colorbar=True,
        scale="log",
        is_orthogonal=True,
        reciprocal_space=True,
        cmap=cmap,
    )
    fig.savefig(detector.savedir + string + "sum" + plot_comment)
    plt.close(fig)

    fig, _, _ = gu.contour_slices(
        interp_data,
        (qx, qz, qy),
        sum_frames=False,
        title="Regridded data",
        levels=np.linspace(0, np.ceil(np.log10(interp_data.max())), 150, endpoint=True),
        plot_colorbar=True,
        scale="log",
        is_orthogonal=True,
        reciprocal_space=True,
        cmap=cmap,
    )
    fig.savefig(detector.savedir + string + "central" + plot_comment)
    plt.close(fig)

    fig, _, _ = gu.multislices_plot(
        interp_data,
        sum_frames=True,
        scale="log",
        plot_colorbar=True,
        vmin=0,
        title="Regridded data",
        is_orthogonal=True,
        reciprocal_space=True,
        cmap=cmap,
    )
    fig.savefig(detector.savedir + string + "sum_pix" + plot_comment)
    plt.close(fig)

    fig, _, _ = gu.multislices_plot(
        interp_data,
        sum_frames=False,
        scale="log",
        plot_colorbar=True,
        vmin=0,
        title="Regridded data",
        is_orthogonal=True,
        reciprocal_space=True,
        cmap=cmap,
    )
    fig.savefig(detector.savedir + string + "central_pix" + plot_comment)
    plt.close(fig)
    if debugging:
        gu.multislices_plot(
            interp_mask,
            sum_frames=False,
            scale="linear",
            plot_colorbar=True,
            vmin=0,
            title="Regridded mask",
            is_orthogonal=True,
            reciprocal_space=True,
            cmap=cmap,
        )

    return interp_data, interp_mask, q_values, transfer_matrix


def grid_bcdi_xrayutil(
    data, mask, scan_number, setup, frames_logical, hxrd, debugging=False, **kwargs
):
    """
    Interpolate BCDI reciprocal space data using xrayutilities package.

    The resulting (qx, qy, qz) are in the crystal frame (qz vertical).

    :param data: the 3D data, already binned in the detector frame
    :param mask: the corresponding 3D mask
    :param scan_number: the scan number to load
    :param setup: instance of the Class experiment_utils.Setup()
    :param frames_logical: array of initial length the number of measured frames.
     In case of padding the length changes. A frame whose index is set to 1 means
     that it is used, 0 means not used, -1 means padded (added) frame.
    :param hxrd: an initialized xrayutilities HXRD object used for the orthogonalization
     of the dataset
    :param debugging: set to True to see plots
    :param kwargs:

     - 'cmap': str, name of the colormap
     - 'logger': an optional logger

    :return: the data and mask interpolated in the crystal frame, q values
     (downstream, vertical up, outboard). q values are in inverse angstroms.
    """
    logger = kwargs.get("logger", module_logger)
    valid.valid_ndarray(arrays=(data, mask), ndim=3)
    cmap = kwargs.get("cmap", "turbo")
    numz, numy, numx = data.shape
    logger.info(
        "Gridding the data using xrayutilities package, "
        "the result will be in the crystal frame"
    )
    string = "xrutil_reciprocal_space_"
    if setup.filtered_data:
        logger.info(
            "Trying to orthogonalize a filtered data, "
            "the corresponding detector ROI should be provided "
            "otherwise q values will be wrong."
        )
    qx, qz, qy, frames_logical = setup.calc_qvalues_xrutils(
        hxrd=hxrd,
        nb_frames=numz,
        scan_number=scan_number,
    )

    maxbins: List[int] = []
    for dim in (qx, qy, qz):
        maxstep = max(abs(np.diff(dim, axis=j)).max() for j in range(3))
        maxbins.append(int(abs(dim.max() - dim.min()) / maxstep))
    logger.info(f"Maximum number of bins based on the sampling in q: {maxbins}")
    maxbins = util.smaller_primes(maxbins, maxprime=7, required_dividers=(2,))
    logger.info(
        f"Maximum number of bins based on the shape requirements for FFT: {maxbins}"
    )
    # only rectangular cuboidal voxels are supported in xrayutilities FuzzyGridder3D
    gridder = xu.FuzzyGridder3D(*maxbins)
    #
    # define the width of data points (rectangular datapoints, xrayutilities use half
    # of these values but there are artefacts sometimes)
    wx = (qx.max() - qx.min()) / maxbins[0]
    wy = (qy.max() - qy.min()) / maxbins[1]
    wz = (qz.max() - qz.min()) / maxbins[2]
    # convert mask to rectangular grid in reciprocal space
    gridder(
        qx, qz, qy, mask, width=(wx, wz, wy)
    )  # qx downstream, qz vertical up, qy outboard
    interp_mask = np.copy(gridder.data)
    # convert data to rectangular grid in reciprocal space
    gridder(
        qx, qz, qy, data, width=(wx, wz, wy)
    )  # qx downstream, qz vertical up, qy outboard
    interp_data = gridder.data

    qx, qz, qy = [
        gridder.xaxis,
        gridder.yaxis,
        gridder.zaxis,
    ]  # downstream, vertical up, outboard
    # q values are 1D arrays

    # check for Nan
    interp_mask[np.isnan(interp_data)] = 1
    interp_data[np.isnan(interp_data)] = 0
    interp_mask[np.isnan(interp_mask)] = 1
    interp_mask = interp_mask.astype(int)

    # apply the mask to the data
    interp_data[np.nonzero(interp_mask)] = 0

    # plot the gridded data
    final_binning = (
        setup.detector.preprocessing_binning[0] * setup.detector.binning[0],
        setup.detector.preprocessing_binning[1] * setup.detector.binning[1],
        setup.detector.preprocessing_binning[2] * setup.detector.binning[2],
    )

    numz, numy, numx = interp_data.shape
    plot_comment = (
        f"_{numz}_{numy}_{numx}"
        f"_{final_binning[0]}_{final_binning[1]}_{final_binning[2]}.png"
    )

    max_z = interp_data.sum(axis=0).max()
    fig, _, _ = gu.contour_slices(
        interp_data,
        (qx, qz, qy),
        sum_frames=True,
        title="Regridded data",
        levels=np.linspace(0, np.ceil(np.log10(max_z)), 150, endpoint=True),
        plot_colorbar=True,
        scale="log",
        is_orthogonal=True,
        reciprocal_space=True,
        cmap=cmap,
    )
    fig.savefig(setup.detector.savedir + string + "sum" + plot_comment)
    plt.close(fig)

    fig, _, _ = gu.contour_slices(
        interp_data,
        (qx, qz, qy),
        sum_frames=False,
        title="Regridded data",
        levels=np.linspace(0, np.ceil(np.log10(interp_data.max())), 150, endpoint=True),
        plot_colorbar=True,
        scale="log",
        is_orthogonal=True,
        reciprocal_space=True,
        cmap=cmap,
    )
    fig.savefig(setup.detector.savedir + string + "central" + plot_comment)
    plt.close(fig)

    fig, _, _ = gu.multislices_plot(
        interp_data,
        sum_frames=True,
        scale="log",
        plot_colorbar=True,
        vmin=0,
        title="Regridded data",
        is_orthogonal=True,
        reciprocal_space=True,
        cmap=cmap,
    )
    fig.savefig(setup.detector.savedir + string + "sum_pix" + plot_comment)
    plt.close(fig)

    fig, _, _ = gu.multislices_plot(
        interp_data,
        sum_frames=False,
        scale="log",
        plot_colorbar=True,
        vmin=0,
        title="Regridded data",
        is_orthogonal=True,
        reciprocal_space=True,
        cmap=cmap,
    )
    fig.savefig(setup.detector.savedir + string + "central_pix" + plot_comment)
    plt.close(fig)
    if debugging:
        gu.multislices_plot(
            interp_mask,
            sum_frames=False,
            scale="linear",
            plot_colorbar=True,
            vmin=0,
            title="Regridded mask",
            is_orthogonal=True,
            reciprocal_space=True,
            cmap=cmap,
        )

    return interp_data, interp_mask, (qx, qz, qy), frames_logical


def load_bcdi_data(
    scan_number: int,
    setup: "Setup",
    bin_during_loading: bool = False,
    flatfield: Optional[np.ndarray] = None,
    hotpixels: Optional[np.ndarray] = None,
    background: Optional[np.ndarray] = None,
    normalize: str = "skip",
    debugging: bool = False,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Bragg CDI data, apply optional threshold, normalization and binning.

    :param scan_number: the scan number to load
    :param setup: an instance of the class Setup
    :param bin_during_loading: True to bin the data during loading (faster)
    :param flatfield: the 2D flatfield array
    :param hotpixels: the 2D hotpixels array. 1 for a hotpixel, 0 for normal pixels.
    :param background: the 2D background array to subtract to the data
    :param normalize: 'monitor' to return the default monitor values, 'sum_roi' to
     return a monitor based on the integrated intensity in the region of interest
     defined by detector.sum_roi, 'skip' to do nothing
    :param debugging:  set to True to see plots
    :param kwargs:

     - 'photon_threshold': float, photon threshold to apply before binning
     - 'frames_pattern': None or list of int.
       Use this if you need to remove some frames, and you know it in advance. You can
       provide a binary list of length the number of images in the dataset. If
       frames_pattern is 0 at index, the frame at data[index] will be skipped, if 1 the
       frame will be added to the stack. Or you can directly specify the indices of the
       frames to be skipped, e.g. [0, 127] to skip frames at indices 0 and 127.
     - 'logger': an optional logger

    :return:
     - the 3D data and mask arrays
     - frames_logical: array of initial length the number of measured frames.
       In case of padding the length changes. A frame whose index is set to 1 means
       that it is used, 0 means not used, -1 means padded (added) frame.
     - the monitor values used for the intensity normalization

    """
    logger = kwargs.get("logger", module_logger)

    # check and load kwargs
    valid.valid_kwargs(
        kwargs=kwargs,
        allowed_kwargs={"photon_threshold", "frames_pattern", "logger"},
    )
    photon_threshold = kwargs.get("photon_threshold", 0)
    valid.valid_item(
        photon_threshold,
        allowed_types=Real,
        min_included=0,
        name="photon_threshold",
    )

    rawdata, rawmask, monitor, frames_logical = setup.loader.load_check_dataset(
        scan_number=scan_number,
        setup=setup,
        frames_pattern=kwargs.get("frames_pattern"),
        bin_during_loading=bin_during_loading,
        flatfield=flatfield,
        hotpixels=hotpixels,
        background=background,
        normalize=normalize,
        debugging=debugging,
    )
    setup.frames_logical = frames_logical
    #####################################################
    # apply an optional photon threshold before binning #
    #####################################################
    if photon_threshold != 0:
        rawmask[rawdata < photon_threshold] = 1
        rawdata[rawdata < photon_threshold] = 0
        logger.info(f"Applying photon threshold before binning: < {photon_threshold}")

    ####################################################################################
    # bin data and mask in the detector plane if not already done during loading       #
    # binning in the stacking dimension is done at the very end of the data processing #
    ####################################################################################
    if not bin_during_loading and (
        (setup.detector.binning[1] != 1) or (setup.detector.binning[2] != 1)
    ):
        logger.info(
            f"Binning the data: detector vertical axis by {setup.detector.binning[1]}, "
            f"detector horizontal axis by {setup.detector.binning[2]}"
        )
        rawdata = util.bin_data(
            rawdata,
            (1, setup.detector.binning[1], setup.detector.binning[2]),
            debugging=False,
        )
        rawmask = util.bin_data(
            rawmask,
            (1, setup.detector.binning[1], setup.detector.binning[2]),
            debugging=False,
        )
        rawmask[np.nonzero(rawmask)] = 1

    # update the current binning factor
    setup.detector.current_binning = list(
        map(
            mul,
            setup.detector.current_binning,
            (1, setup.detector.binning[1], setup.detector.binning[2]),
        )
    )
    ################################################
    # pad the data to the shape defined by the ROI #
    ################################################
    rawdata, rawmask = util.pad_from_roi(
        arrays=(rawdata, rawmask),
        roi=setup.detector.roi,
        binning=setup.detector.binning[1:],
        pad_value=(0, 1),
    )

    return np.asarray(rawdata), np.asarray(rawmask), frames_logical, np.asarray(monitor)


def reload_bcdi_data(
    data: np.ndarray,
    mask: np.ndarray,
    scan_number: int,
    setup: "Setup",
    normalize: bool = False,
    debugging: bool = False,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reload BCDI data, apply optional threshold, normalization and binning.

    :param data: the 3D data array
    :param mask: the 3D mask array
    :param scan_number: the scan number to load
    :param setup: an instance of the class Setup
    :param normalize: set to "monitor" to normalize by the default monitor of
     the beamline, otherwise set to "skip"
    :param debugging:  set to True to see plots
    :parama kwargs:

     - 'frames_pattern': None or list of int.
       Use this if you need to remove some frames, and you know it in advance. You can
       provide a binary list of length the number of images in the dataset. If
       frames_pattern is 0 at index, the frame at data[index] will be skipped, if 1 the
       frame will be added to the stack. Or you can directly specify the indices of the
       frames to be skipped, e.g. [0, 127] to skip frames at indices 0 and 127.
     - 'photon_threshold' = float, photon threshold to apply before binning
     - 'logger': an optional logger

    :return:
     - the updated 3D data and mask arrays
     - the monitor values used for the intensity normalization

    """
    logger = kwargs.get("logger", module_logger)
    valid.valid_ndarray(arrays=(data, mask), ndim=3)
    # check and load kwargs
    valid.valid_kwargs(
        kwargs=kwargs,
        allowed_kwargs={"frames_pattern", "logger", "photon_threshold"},
    )
    photon_threshold = kwargs.get("photon_threshold", 0)
    valid.valid_item(
        photon_threshold,
        allowed_types=Real,
        min_included=0,
        name="photon_threshold",
    )

    nbz, nby, nbx = data.shape
    setup.frames_logical = util.generate_frames_logical(
        nb_images=nbz, frames_pattern=kwargs.get("frames_pattern")
    )

    logger.info(f"{(data < 0).sum()} negative data points masked")
    # can happen when subtracting a background
    mask[data < 0] = 1
    data[data < 0] = 0

    # normalize by the incident X-ray beam intensity
    if normalize == "skip":
        logger.info("Skip intensity normalization")
        monitor = np.ones(nbz)
    else:  # use the default monitor of the beamline
        monitor = setup.loader.read_monitor(
            scan_number=scan_number,
            setup=setup,
        )

        logger.info(f"Intensity normalization using {normalize}")
        data, monitor = loader.normalize_dataset(
            array=data,
            monitor=monitor,
            norm_to_min=True,
            savedir=setup.detector.savedir,
            debugging=True,
            logger=logger,
        )

    # pad the data to the shape defined by the ROI
    if (
        setup.detector.roi[1] - setup.detector.roi[0] > nby
        or setup.detector.roi[3] - setup.detector.roi[2] > nbx
    ):
        start = (np.nan, min(0, setup.detector.roi[0]), min(0, setup.detector.roi[2]))
        logger.info("Padding the data to the shape defined by the ROI")
        data = util.crop_pad(
            array=data,
            pad_start=start,
            output_shape=(
                data.shape[0],
                setup.detector.roi[1] - setup.detector.roi[0],
                setup.detector.roi[3] - setup.detector.roi[2],
            ),
        )
        mask = util.crop_pad(
            array=mask,
            pad_value=1,
            pad_start=start,
            output_shape=(
                mask.shape[0],
                setup.detector.roi[1] - setup.detector.roi[0],
                setup.detector.roi[3] - setup.detector.roi[2],
            ),
        )

    # apply optional photon threshold before binning
    if photon_threshold != 0:
        mask[data < photon_threshold] = 1
        data[data < photon_threshold] = 0
        logger.info(f"Applying photon threshold before binning: < {photon_threshold}")

    # bin data and mask in the detector plane if needed
    # binning in the stacking dimension is done at the very end of the data processing
    if (setup.detector.binning[1] != 1) or (setup.detector.binning[2] != 1):
        logger.info(
            "Binning the data: setup.detector vertical axis by "
            f"{setup.detector.binning[1]}, setup.detector horizontal axis by "
            f"{setup.detector.binning[2]}"
        )
        data = util.bin_data(
            data,
            (1, setup.detector.binning[1], setup.detector.binning[2]),
            debugging=debugging,
        )
        mask = util.bin_data(
            mask,
            (1, setup.detector.binning[1], setup.detector.binning[2]),
            debugging=debugging,
        )
        mask[np.nonzero(mask)] = 1
        setup.detector.current_binning = list(
            map(
                mul,
                setup.detector.current_binning,
                (1, setup.detector.binning[1], setup.detector.binning[2]),
            )
        )

    return data, mask, setup.frames_logical, monitor
