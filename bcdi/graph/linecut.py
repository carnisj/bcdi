# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr
"""Class and functions related to the generation of linecuts through an array."""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from lmfit import Parameters, minimize
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks

from bcdi.utils import utilities as util
from bcdi.utils import validation as valid

module_logger = logging.getLogger(__name__)


class LinecutGenerator:
    """
    Generate linecuts of an array, including fitting and plotting methods.

    :param array: a 2D or 3D array, typically the modulus of a real space object after
     phase retrieval
    :param indices: a list of ndim lists of ndim tuples (start, stop), ndim being the
     number of dimensions of the array (one list per linecut)
    :param fit_derivative: True to fit the gradient of the linecut with a gaussian line
     shape
    :param support_threshold: float, threshold used to define the support, for the
     determination of the location of crystal boundaries
    :param voxel_sizes: list of voxels sizes, in each dimension of the array
    :param filename: name for saving plots
    :param label: labels for the vertical axis of plots
    :param kwargs:

     - 'logger': an optional logger

    """

    def __init__(
        self,
        array: np.ndarray,
        indices: Optional[List[List[Tuple[int, int]]]] = None,
        filename: Optional[str] = None,
        fit_derivative: bool = False,
        voxel_sizes: Optional[List[float]] = None,
        support_threshold: float = 0.25,
        label: str = "linecut",
        **kwargs,
    ):
        self.array = array
        self.indices = indices
        self.filename = filename
        self.fit_derivative = fit_derivative
        self.voxel_sizes = voxel_sizes
        self.support_threshold = support_threshold
        self.user_label = label
        self.logger: logging.Logger = kwargs.get("logger", module_logger)
        self.result: Dict[str, Any] = {}

        self._current_axis: int = 0
        self._current_linecut: Optional[np.ndarray] = None
        self._peaks: Optional[np.ndarray] = None

    @property
    def filename(self):
        """File name for saving plots."""
        return self._filename

    @filename.setter
    def filename(self, value):
        if value is not None and not isinstance(value, str):
            raise TypeError(f"filename should be a string, got {type(value)}")
        self._filename = value

    @property
    def indices(self):
        """List of indices for the linecut."""
        return self._indices

    @indices.setter
    def indices(self, value):
        if value is None:
            # default to the linecut through the center of the array in each dimension
            value = []
            for idx, shp in enumerate(self.array.shape):
                default = [(val // 2, val // 2) for val in self.array.shape]
                default[idx] = (0, shp - 1)
                value.append(default)

        valid.valid_container(
            value,
            container_types=list,
            item_types=list,
            length=self.array.ndim,
            name="indices",
        )
        for _, item in enumerate(value):
            valid.valid_container(
                item,
                container_types=list,
                item_types=tuple,
                length=self.array.ndim,
                name="indices sublists",
            )
        self._indices = value

    @property
    def plot_labels(self):
        """Axis labels used in plots."""
        return {
            "dimension_0": f"Z ({self.unit})",
            "dimension_1": f"Y ({self.unit})",
            "dimension_2": f"X ({self.unit})",
        }

    @property
    def unit(self):
        """Define the unit length for the array pixel."""
        return "nm" if self.voxel_sizes is not None else "pixels"

    @property
    def voxel_sizes(self):
        """Voxel sizes of the array in each dimension."""
        return self._voxel_sizes

    @voxel_sizes.setter
    def voxel_sizes(self, value):
        valid.valid_container(
            value,
            container_types=(tuple, list),
            item_types=(float, int),
            length=self.array.ndim,
            allow_none=True,
            name="voxel_sizes",
        )
        self._voxel_sizes = value

    def fit_boundaries(self) -> None:
        """Fit the array with gaussians at the location of the detected boundaries."""
        if self._peaks is None or len(self._peaks) == 0:
            self.logger.info(
                f"No peak detected in the linecut of axis {self._current_axis}"
            )
            return
        if self._current_linecut is None:
            self.logger.info(f"No defined linecut for axis {self._current_axis}")
            return

        dcut = abs(np.gradient(self._current_linecut))

        # setup data and parameters for fitting
        for peak_id, peak in enumerate(self._peaks):
            index_start = max(0, peak - 10)
            index_stop = min(len(dcut), peak + 10)
            cropped_xaxis = np.arange(index_start, index_stop)
            cropped_dcut = dcut[index_start:index_stop]
            self.result[f"dimension_{self._current_axis}"][f"derivative_{peak_id}"] = (
                np.vstack((cropped_xaxis, cropped_dcut))
            )

            fit_params = Parameters()
            fit_params.add("amp_0", value=1, min=0.1, max=10)
            fit_params.add("cen_0", value=peak, min=peak - 3, max=peak + 3)
            fit_params.add("sig_0", value=2, min=0.1, max=10)

            # fit the data
            fit_result = minimize(
                util.objective_lmfit,
                fit_params,
                args=(
                    np.asarray(cropped_xaxis),
                    np.asarray(cropped_dcut),
                    "gaussian",
                ),
            )

            # generate fit curves
            interp_xaxis = util.upsample(cropped_xaxis, factor=4)
            y_fit = util.function_lmfit(
                params=fit_result.params,
                x_axis=interp_xaxis,
                distribution="gaussian",
            )
            self.result[f"dimension_{self._current_axis}"][f"fit_{peak_id}"] = (
                np.vstack((interp_xaxis, y_fit))
            )
            self.result[f"dimension_{self._current_axis}"][f"param_{peak_id}"] = {
                "amp": fit_result.params["amp_0"].value,
                "sig": fit_result.params["sig_0"].value,
                "cen": fit_result.params["cen_0"].value,
            }

    def find_boundaries(self) -> None:
        """
        Localize coarsely the position of the jumps in the modulus of the array.

        For a cristal reconstructed after phase retrieval, these position correspond to
        the cristal boundaries.
        """
        if self.fit_derivative:
            support = np.zeros(self.array.shape)
            support[self.array > self.support_threshold] = 1
            support_cut = linecut(
                array=support, indices=self.indices[self._current_axis]
            )

            self._peaks, _ = find_peaks(
                abs(np.gradient(support_cut)), height=0.1, distance=10, width=1
            )

    def generate_linecut(self) -> None:
        """Generate a linecut along a given axis."""
        self.result[f"dimension_{self._current_axis}"] = {}

        self._current_linecut = linecut(
            array=self.array, indices=self.indices[self._current_axis]
        )
        self.result[f"dimension_{self._current_axis}"]["linecut"] = np.vstack(
            (
                np.arange(
                    self.indices[self._current_axis][self._current_axis][0],
                    self.indices[self._current_axis][self._current_axis][1] + 1,
                ),
                self._current_linecut,
            )
        )

    def generate_linecuts(self):
        """Generate a linecut for each dimension of the array."""
        for axis in range(self.array.ndim):
            self._current_axis = axis
            self.generate_linecut()
            self.find_boundaries()
            self.fit_boundaries()

    def plot_fits(self) -> None:
        """
        Plot fits to the derivatives of the linecuts.

        Expected structure for linecuts::

            linecuts = {
                'dimension_0': {
                    'linecut': np.ndarray (2, M),
                    'derivative_0': np.ndarray (2, N),
                    'derivative_1': np.ndarray (2, O),
                    'fit_0': np.ndarray (2, P),
                    'fit_1': np.ndarray (2, P),
                    'param_0': {'amp': float, 'sig': float, 'cen': float},
                    'param_1': {'amp': float, 'sig': float, 'cen': float},
                },
                'dimension_1': {...}
                ...
            }
        """
        fig, axes = plt.subplots(nrows=self.array.ndim, ncols=2, figsize=(12, 9))
        try:
            for idx, key in enumerate(self.result.keys()):
                factor = self.voxel_sizes[idx] if self.voxel_sizes is not None else 1
                for subkey in self.result[key].keys():
                    if subkey.startswith("derivative"):
                        index = int(subkey[-1])
                        (line1,) = axes[idx][index].plot(
                            self.result[key][subkey][0] * factor,
                            self.result[key][subkey][1],
                            ".b",
                            label="derivative",
                        )
                        (line2,) = axes[idx][index].plot(
                            self.result[key][f"fit_{index}"][0] * factor,
                            self.result[key][f"fit_{index}"][1],
                            "-r",
                            label="gaussian fit",
                        )
                        axes[idx][index].set_xlabel(self.plot_labels.get(key, key))
                        axes[idx][index].legend(handles=[line1, line2])
                        fwhm = (
                            2
                            * np.sqrt(2 * np.log(2))
                            * factor
                            * self.result[key][f"param_{index}"]["sig"]
                        )

                        axes[idx][index].text(
                            x=0.05,
                            y=0.9,
                            s=f"FWHM={fwhm:.2f} {self.unit}",
                            transform=axes[idx][index].transAxes,
                        )

            plt.tight_layout()  # avoids the overlap of subplots with axes labels
            plt.pause(0.1)
            plt.ioff()
            if self.filename:
                base, _ = os.path.splitext(self.filename)
                fig.savefig(base + "_fits.png")

        except IndexError:  # fits not successfull
            self.logger.warning("Plot of fitted linecuts failed.")
            plt.close(fig)

    def plot_linecuts(self) -> None:
        """
        Plot the generated linecuts and optionally save the figure.

        Expected structure for linecuts::

            linecuts = {
                'dimension_0': {
                    'linecut': np.ndarray (2, M),
                    'derivative_0': np.ndarray (2, N),
                    'derivative_1': np.ndarray (2, O),
                    'fit_0': np.ndarray (2, P),
                    'fit_1': np.ndarray (2, P),
                    'param_0': {'amp': float, 'sig': float, 'cen': float},
                    'param_1': {'amp': float, 'sig': float, 'cen': float},
                },
                'dimension_1': {...}
                ...
            }
        """
        fig, axes = plt.subplots(nrows=self.array.ndim, ncols=1, figsize=(12, 9))
        for idx, key in enumerate(self.result.keys()):
            factor = self.voxel_sizes[idx] if self.voxel_sizes is not None else 1

            axes[idx].plot(
                self.result[key]["linecut"][0] * factor,
                self.result[key]["linecut"][1],
                ".-b",
            )
            axes[idx].set_xlabel(self.plot_labels.get(key, key))
            axes[idx].set_ylabel(self.user_label)

        plt.tight_layout()  # avoids the overlap of subplots with axes labels
        plt.pause(0.1)
        plt.ioff()
        if self.filename:
            fig.savefig(self.filename)


def fit_linecut(
    array: np.ndarray,
    indices: Optional[List[List[Tuple[int, int]]]] = None,
    fit_derivative: bool = False,
    support_threshold: float = 0.25,
    voxel_sizes: Optional[List[float]] = None,
    filename: Optional[str] = None,
    label: str = "linecut",
    **kwargs,
) -> Dict:
    """
    Perform a linecut on an array and optionally fit its gradient.

    :param array: a 2D or 3D array, typically the modulus of a real space object after
     phase retrieval
    :param indices: a list of ndim lists of ndim tuples (start, stop), ndim being the
     number of dimensions of the array (one list per linecut)
    :param fit_derivative: True to fit the gradient of the linecut with a gaussian line
     shape
    :param support_threshold: float, threshold used to define the support, for the
     determination of the location of crystal boundaries
    :param voxel_sizes: list of voxels sizes, in each dimension of the array
    :param filename: name for saving plots
    :param label: labels for the vertical axis of plots
    :param kwargs:

     - 'logger': an optional logger

    :return: a dictionary containing linecuts, fits and fitted parameters
    """
    # generate a linecut for each dimension of the array
    linecut_generator = LinecutGenerator(
        array=array,
        indices=indices,
        filename=filename,
        fit_derivative=fit_derivative,
        voxel_sizes=voxel_sizes,
        support_threshold=support_threshold,
        label=label,
        **kwargs,
    )

    linecut_generator.generate_linecuts()
    linecut_generator.plot_linecuts()
    linecut_generator.plot_fits()
    return linecut_generator.result


def linecut(
    array: np.ndarray,
    indices: List[Tuple[int, int]],
    interp_order: int = 1,
) -> np.ndarray:
    """
    Linecut through a 2D or 3D array.

    The user must input indices of the starting voxel and of the end voxel.

    :param array: a numpy array
    :param indices: list of tuples of (start, stop) indices, one tuple for each
     dimension of the array. e.g [(start0, stop0), (start1, stop1)] for a 2D array
    :param interp_order: order of the spline interpolation, default is 3.
     The order has to be in the range 0-5.
    :return: a 1D array interpolated between the start and stop indices
    """
    # check parameters
    valid.valid_ndarray(array)
    array = np.asarray(array)
    if array.dtype in ["int8", "int16", "int32", "int64"]:
        array = array.astype(float)

    if not isinstance(indices, list):
        raise TypeError(f"'indices' should be a list, got {type(indices)}")
    for _, item in enumerate(indices):
        valid.valid_container(
            item,
            container_types=tuple,
            item_types=int,
            min_included=0,
            length=2,
            name="indices",
        )
    valid.valid_item(
        interp_order, allowed_types=int, min_included=1, name="interp_order"
    )

    num_points = int(
        np.sqrt(sum((val[1] - val[0] + 1) ** 2 for _, val in enumerate(indices)))
    )

    cut = map_coordinates(
        input=array,
        coordinates=np.vstack(
            [
                np.linspace(val[0], val[1], endpoint=True, num=num_points)
                for _, val in enumerate(indices)
            ]
        ),
        order=interp_order,
    )
    return np.asarray(cut)
