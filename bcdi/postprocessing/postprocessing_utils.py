# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr
"""Functions related to data postprocessing after phase retrieval."""

import gc
import logging
from math import pi
from numbers import Number, Real
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import scipy
from numpy.fft import fftn, fftshift, ifftn, ifftshift
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import center_of_mass
from scipy.signal import convolve
from scipy.stats import multivariate_normal, norm
from skimage.restoration import unwrap_phase

from ..graph import graph_utils as gu
from ..utils import utilities as util
from ..utils import validation as valid

module_logger = logging.getLogger(__name__)


class ModulusBelowThreshold(Exception):
    """Custom exception when there are no voxels above a threshold in an array."""


def apodize(amp, phase, initial_shape, window_type, debugging=False, **kwargs):
    """
    Apodize the complex array based on the window of the same shape.

    :param amp: 3D array, amplitude before apodization
    :param phase: 3D array, phase before apodization
    :param initial_shape: shape of the FFT used for phasing
    :param window_type: window filtering function, 'normal' or 'tukey' or 'blackman'
    :param debugging: set to True to see plots
    :type debugging: bool
    :param kwargs:

     - 'cmap': str, name of the colormap
     - for the normal distribution: 'sigma' and 'mu' of the 3d multivariate normal
       distribution, tuples of 3 floats
     - for the Tuckey window: 'alpha' (shape parameter) of the 3d Tukey window,
       tuple of 3 floats
     - 'is_orthogonal': True if the data is in an orthonormal frame. Used for defining
       default plot labels.
     - 'logger': an optional logger

    :return: filtered amplitude, phase of the same shape as myamp
    """
    valid.valid_ndarray(arrays=(amp, phase), ndim=3)
    # check and load kwargs
    valid.valid_kwargs(
        kwargs=kwargs,
        allowed_kwargs={"logger", "cmap", "sigma", "mu", "alpha", "is_orthogonal"},
        name="postprocessing_utils.apodize",
    )
    logger = kwargs.get("logger", module_logger)
    sigma = kwargs.get("sigma")
    mu = kwargs.get("mu")
    alpha = kwargs.get("alpha")
    is_orthogonal = kwargs.get("is_orthogonal", False)
    cmap = kwargs.get("cmap", "turbo")
    # calculate the diffraction pattern of the reconstructed object
    nb_z, nb_y, nb_x = amp.shape
    nbz, nby, nbx = initial_shape
    myobj = util.crop_pad(amp * np.exp(1j * phase), (nbz, nby, nbx))
    del amp, phase
    gc.collect()
    if debugging:
        gu.multislices_plot(
            array=abs(myobj),
            sum_frames=False,
            plot_colorbar=True,
            title="modulus before apodization",
            reciprocal_space=False,
            is_orthogonal=is_orthogonal,
            scale="linear",
            cmap=cmap,
        )

    my_fft = fftshift(fftn(myobj))
    del myobj
    gc.collect()
    fftmax = abs(my_fft).max()
    if debugging:
        logger.info(f"Max FFT={fftmax}")
        gu.multislices_plot(
            array=abs(my_fft),
            sum_frames=False,
            plot_colorbar=True,
            title="diffraction amplitude before apodization",
            reciprocal_space=True,
            is_orthogonal=is_orthogonal,
            scale="log",
            cmap=cmap,
        )

    if window_type == "normal":
        logger.info("Apodization using a 3d multivariate normal window")
        sigma = sigma or np.array([0.3, 0.3, 0.3])
        mu = mu or np.array([0.0, 0.0, 0.0])

        grid_z, grid_y, grid_x = np.meshgrid(
            np.linspace(-1, 1, nbz),
            np.linspace(-1, 1, nby),
            np.linspace(-1, 1, nbx),
            indexing="ij",
        )
        covariance = np.diag(sigma**2)
        window = multivariate_normal.pdf(
            np.column_stack([grid_z.flat, grid_y.flat, grid_x.flat]),
            mean=mu,
            cov=covariance,
        )
        del grid_z, grid_y, grid_x
        gc.collect()
        window = window.reshape((nbz, nby, nbx))

    elif window_type == "tukey":
        logger.info("Apodization using a 3d Tukey window")
        alpha = alpha or np.array([0.5, 0.5, 0.5])
        window = tukey_window(initial_shape, alpha=alpha)

    elif window_type == "blackman":
        logger.info("Apodization using a 3d Blackman window")
        window = blackman_window(initial_shape)

    else:
        raise ValueError("Invalid window type")

    my_fft = np.multiply(my_fft, window)
    del window
    gc.collect()
    my_fft = my_fft * fftmax / abs(my_fft).max()
    if debugging:
        logger.info(f"Max apodized FFT after normalization = {abs(my_fft).max()}")
        gu.multislices_plot(
            array=abs(my_fft),
            sum_frames=False,
            plot_colorbar=True,
            title="diffraction amplitude after apodization",
            reciprocal_space=True,
            is_orthogonal=is_orthogonal,
            scale="log",
            cmap=cmap,
        )

    myobj = ifftn(ifftshift(my_fft))
    del my_fft
    gc.collect()
    if debugging:
        gu.multislices_plot(
            array=abs(myobj),
            sum_frames=False,
            plot_colorbar=True,
            title="modulus after apodization",
            reciprocal_space=False,
            is_orthogonal=is_orthogonal,
            scale="linear",
            cmap=cmap,
        )
    myobj = util.crop_pad(
        myobj, (nb_z, nb_y, nb_x)
    )  # return to the initial shape of myamp
    return abs(myobj), np.angle(myobj)


def blackman_window(shape, normalization=1):
    """
    Create a 3d Blackman window based on shape.

    :param shape: tuple, shape of the 3d window
    :param normalization: value of the integral of the backman window
    :return: the 3d Blackman window
    """
    nbz, nby, nbx = shape
    array_z = np.blackman(nbz)
    array_y = np.blackman(nby)
    array_x = np.blackman(nbx)
    blackman2 = np.ones((nbz, nby))
    blackman3 = np.ones((nbz, nby, nbx))
    for idz in range(nbz):
        blackman2[idz, :] = array_z[idz] * array_y
        for idy in range(nby):
            blackman3[idz, idy] = blackman2[idz, idy] * array_x
    blackman3 = blackman3 / blackman3.sum() * normalization
    return blackman3


def bragg_temperature(
    spacing,
    reflection,
    spacing_ref=None,
    temperature_ref=None,
    use_q=False,
    material=None,
):
    """
    Calculate the temperature from Bragg peak position.

    :param spacing: q or planar distance, in inverse angstroms or angstroms
    :param reflection: measured reflection, e.g. np.array([1, 1, 1])
    :param spacing_ref: reference spacing at known temperature
     (include substrate-induced strain)
    :param temperature_ref: in K, known temperature for the reference spacing
    :param use_q: set to True to use q, False to use planar distance
    :type use_q: bool
    :param material: at the moment only 'Pt'
    :return: calculated temperature
    """
    print("\n")
    if material == "Pt":
        # reference values for Pt: temperature in K, thermal expansion x 10^6 in 1/K,
        # lattice parameter in angstroms
        expansion_data = np.array(
            [
                [100, 6.77, 3.9173],
                [110, 7.10, 3.9176],
                [120, 7.37, 3.9179],
                [130, 7.59, 3.9182],
                [140, 7.78, 3.9185],
                [150, 7.93, 3.9188],
                [160, 8.07, 3.9191],
                [180, 8.29, 3.9198],
                [200, 8.46, 3.9204],
                [220, 8.59, 3.9211],
                [240, 8.70, 3.9218],
                [260, 8.80, 3.9224],
                [280, 8.89, 3.9231],
                [293.15, 8.93, 3.9236],
                [300, 8.95, 3.9238],
                [400, 9.25, 3.9274],
                [500, 9.48, 3.9311],
                [600, 9.71, 3.9349],
                [700, 9.94, 3.9387],
                [800, 10.19, 3.9427],
                [900, 10.47, 3.9468],
                [1000, 10.77, 3.9510],
                [1100, 11.10, 3.9553],
                [1200, 11.43, 3.9597],
            ]
        )
        if spacing_ref is None:
            print("Using the reference spacing of Platinum")
            spacing_ref = 3.9236 / np.linalg.norm(reflection)  # angstroms
        if temperature_ref is None:
            temperature_ref = 293.15  # K
    else:
        raise ValueError('Only "Pt" available for temperature estimation')
    if use_q:
        spacing = 2 * np.pi / spacing  # go back to distance
        spacing_ref = 2 * np.pi / spacing_ref  # go back to distance
    spacing = spacing * np.linalg.norm(reflection)  # go back to lattice constant
    spacing_ref = spacing_ref * np.linalg.norm(
        reflection
    )  # go back to lattice constant
    print(
        f"Reference spacing at {temperature_ref} K = {spacing_ref:.4f} angstroms",
    )
    print(f"Spacing = {spacing:.4f} angstroms using reflection {reflection}")

    # fit the experimental spacing with non-corrected platinum curve
    myfit = np.poly1d(np.polyfit(expansion_data[:, 2], expansion_data[:, 0], 3))
    print(
        f"Temperature without offset correction = {int(myfit(spacing) - 273.15)} degC"
    )

    # find offset for platinum reference curve
    myfit = np.poly1d(np.polyfit(expansion_data[:, 0], expansion_data[:, 2], 3))
    spacing_offset = (
        myfit(temperature_ref) - spacing_ref
    )  # T in K, spacing in angstroms
    print(f"Spacing offset = {spacing_offset:.4f} angstroms")

    # correct the platinum reference curve for the offset
    platinum_offset = np.copy(expansion_data)
    platinum_offset[:, 2] = platinum_offset[:, 2] - spacing_offset
    myfit = np.poly1d(np.polyfit(platinum_offset[:, 2], platinum_offset[:, 0], 3))
    mytemp = int(myfit(spacing) - 273.15)
    print("Temperature with offset correction=", mytemp, "C")
    return mytemp


def calc_coordination(
    support,
    kernel=np.ones((3, 3, 3)),
    width_z=None,
    width_y=None,
    width_x=None,
    debugging=False,
):
    """
    Calculate the coordination number of voxels in a support (numbe of neighbours).

    :param support: 3D support array
    :param kernel: kernel used for convolution with the support
    :param width_z: size of the area to plot in z (axis 0), centered on the middle
     of the initial array
    :param width_y: size of the area to plot in y (axis 1), centered on the middle
     of the initial array
    :param width_x: size of the area to plot in x (axis 2), centered on the middle
     of the initial array
    :param debugging: set to True to see plots
    :type debugging: bool
    :return: the coordination matrix
    """
    valid.valid_ndarray(arrays=support, ndim=3)

    mycoord = np.rint(convolve(support, kernel, mode="same"))
    mycoord = mycoord.astype(int)

    if debugging:
        gu.multislices_plot(
            support,
            width_z=width_z,
            width_y=width_y,
            width_x=width_x,
            vmin=0,
            is_orthogonal=True,
            reciprocal_space=False,
            title="Input support",
        )
        gu.multislices_plot(
            mycoord,
            plot_colorbar=True,
            width_z=width_z,
            width_y=width_y,
            width_x=width_x,
            vmin=0,
            is_orthogonal=True,
            reciprocal_space=False,
            title="Coordination matrix",
        )
    return mycoord


def center_com(array: np.ndarray, debugging: bool = False, **kwargs) -> np.ndarray:
    """
    Center array based on center_of_mass(abs(array)) using pixel shift.

    :param array: 3D array to be centered using the center of mass of abs(array)
    :param debugging: boolean, True to see plots
    :param kwargs:
     - width_z: size of the area to plot in z (axis 0), centered in the middle
       of the initial array
     - width_y: size of the area to plot in y (axis 1), centered in the middle
       of the initial array
     - width_x: size of the area to plot in x (axis 2), centered in the middle
       of the initial array

    :return: array centered by pixel shift
    """
    valid.valid_ndarray(arrays=array, ndim=3)
    #########################
    # check and load kwargs #
    #########################
    valid.valid_kwargs(
        kwargs=kwargs, allowed_kwargs={"width_z", "width_y", "width_x"}, name="kwargs"
    )
    width_z = kwargs.get("width_z")
    valid.valid_item(
        value=width_z,
        allowed_types=int,
        min_excluded=0,
        allow_none=True,
        name="width_z",
    )
    width_y = kwargs.get("width_y")
    valid.valid_item(
        value=width_y,
        allowed_types=int,
        min_excluded=0,
        allow_none=True,
        name="width_y",
    )
    width_x = kwargs.get("width_x")
    valid.valid_item(
        value=width_x,
        allowed_types=int,
        min_excluded=0,
        allow_none=True,
        name="width_x",
    )

    #########################################
    # find the offset of the center of mass #
    #########################################
    nbz, nby, nbx = array.shape
    piz, piy, pix = center_of_mass(abs(array))
    offset_z = int(np.rint(nbz / 2.0 - piz))
    offset_y = int(np.rint(nby / 2.0 - piy))
    offset_x = int(np.rint(nbx / 2.0 - pix))

    if debugging:
        gu.multislices_plot(
            abs(array),
            width_z=width_z,
            width_y=width_y,
            width_x=width_x,
            title="Before COM centering",
        )

        print(
            f"center of mass at (z, y, x): ({piz:.2f},{piy:.2f},{pix:.2f})",
        )
        print(
            f"center of mass offset: ({offset_z},{offset_y},{offset_x}) pixels",
        )

    #####################
    # center the object #
    #####################
    array = np.roll(array, (offset_z, offset_y, offset_x), axis=(0, 1, 2))

    if debugging:
        gu.multislices_plot(
            abs(array),
            width_z=width_z,
            width_y=width_y,
            width_x=width_x,
            title="After COM centering",
        )
    return array


def center_max(array: np.ndarray, debugging: bool = False, **kwargs) -> np.ndarray:
    """
    Center array based on max(abs(array)) using pixel shift.

    :param array: 3D array to be centered based on max(abs(array))
    :param debugging: boolean, True to see plots
    :param kwargs:
     - width_z: size of the area to plot in z (axis 0), centered in the middle
       of the initial array
     - width_y: size of the area to plot in y (axis 1), centered in the middle
       of the initial array
     - width_x: size of the area to plot in x (axis 2), centered in the middle
       of the initial array

    :return: array centered by pixel shift
    """
    valid.valid_ndarray(arrays=array, ndim=3)
    #########################
    # check and load kwargs #
    #########################
    valid.valid_kwargs(
        kwargs=kwargs, allowed_kwargs={"width_z", "width_y", "width_x"}, name="kwargs"
    )
    width_z = kwargs.get("width_z")
    valid.valid_item(
        value=width_z,
        allowed_types=int,
        min_excluded=0,
        allow_none=True,
        name="width_z",
    )
    width_y = kwargs.get("width_y")
    valid.valid_item(
        value=width_y,
        allowed_types=int,
        min_excluded=0,
        allow_none=True,
        name="width_y",
    )
    width_x = kwargs.get("width_x")
    valid.valid_item(
        value=width_x,
        allowed_types=int,
        min_excluded=0,
        allow_none=True,
        name="width_x",
    )

    ##################################################################
    # find the offset of the max relative to the center of the array #
    ##################################################################
    nbz, nby, nbx = array.shape
    piz, piy, pix = np.unravel_index(abs(array).argmax(), array.shape)
    offset_z = int(np.rint(nbz / 2.0 - piz))
    offset_y = int(np.rint(nby / 2.0 - piy))
    offset_x = int(np.rint(nbx / 2.0 - pix))

    if debugging:
        gu.multislices_plot(
            abs(array),
            width_z=width_z,
            width_y=width_y,
            width_x=width_x,
            title="Before max centering",
        )
        print("Max at (z, y, x): (", piz, ",", piy, ",", pix, ")")

        print("Max offset: (", offset_z, ",", offset_y, ",", offset_x, ") pixels")

    #####################
    # center the object #
    #####################
    array = np.roll(array, (offset_z, offset_y, offset_x), axis=(0, 1, 2))

    if debugging:
        gu.multislices_plot(
            abs(array),
            width_z=width_z,
            width_y=width_y,
            width_x=width_x,
            title="After max centering",
        )
    return array


def center_object(method: str, obj: np.ndarray) -> np.ndarray:
    """
    Center an object based on its modulus.

    :param method: method used for centering. 'com' (center of mass), 'max', 'max_com'
     (max then com), 'skip'.
    :param obj: a 3D numpy array
    :return: the centered array of the same shape as obj
    """
    valid.valid_ndarray(obj, ndim=3)
    if method == "max":
        return center_max(obj)
        # shift based on max value, use this if the object spans across the edge of the
        # window before using the center of mass
    if method == "com":
        return center_com(obj)
    if method == "max_com":
        return center_com(center_max(obj))
    return obj


def filter_3d(
    array, filter_name="gaussian_highpass", kernel_length=21, debugging=False, **kwargs
):
    """
    Apply a filter to the array by convoluting with a filtering kernel.

    :param array: 2D or 3D array to be filtered
    :param filter_name: name of the filter, 'gaussian', 'gaussian_highpass'
    :param kernel_length: length in pixels of the filtering kernel
    :param debugging: True to see a plot of the kernel
    :param kwargs:
     - 'sigma': sigma of the gaussian kernel

    """
    valid.valid_ndarray(arrays=array, ndim=(2, 3))
    # check and load kwargs
    valid.valid_kwargs(
        kwargs=kwargs, allowed_kwargs={"sigma"}, name="postprocessing_utils.filter_3d"
    )
    sigma = kwargs.get("sigma")

    if filter_name == "gaussian_highpass":
        sigma = sigma or 3
        kernel = gaussian_kernel(
            ndim=array.ndim,
            kernel_length=kernel_length,
            sigma=sigma,
            debugging=debugging,
        )
        return array - convolve(array, kernel, mode="same")
    if filter_name == "gaussian":
        sigma = sigma or 0.5
        kernel = gaussian_kernel(
            ndim=array.ndim,
            kernel_length=kernel_length,
            sigma=sigma,
            debugging=debugging,
        )
        return convolve(array, kernel, mode="same")
    raise ValueError("Only the gaussian_kernel is implemented up to now.")


def find_bulk(
    amp: np.ndarray,
    support_threshold: float,
    method: str = "threshold",
    width_z: Optional[int] = None,
    width_y: Optional[int] = None,
    width_x: Optional[int] = None,
    debugging: bool = False,
    **kwargs,
) -> np.ndarray:
    """
    Isolate the inner part of the crystal from the non-physical surface.

    :param amp: 3D array, reconstructed object amplitude
    :param support_threshold:  threshold for isosurface determination
    :param method: 'threshold' or 'defect'. If 'defect', removes layer by layer using
     the coordination number.
    :param width_z: size of the area to plot in z (axis 0), centered on the middle
     of the initial array
    :param width_y: size of the area to plot in y (axis 1), centered on the middle
     of the initial array
    :param width_x: size of the area to plot in x (axis 2), centered on the middle
     of the initial array
    :param debugging: set to True to see plots
    :type debugging: bool
    :param kwargs:

     - 'cmap': str, name of the colormap
     - 'logger': an optional logger

    :return: the support corresponding to the bulk
    """
    logger = kwargs.get("logger", module_logger)
    valid.valid_ndarray(arrays=amp, ndim=3)
    cmap = kwargs.get("cmap", "turbo")

    nbz, nby, nbx = amp.shape
    max_amp = abs(amp).max()
    support = np.ones((nbz, nby, nbx))

    if method == "threshold":
        support[abs(amp) < support_threshold * max_amp] = 0
    else:
        support[abs(amp) < 0.05 * max_amp] = 0  # predefine a larger support
        mykernel = np.ones((9, 9, 9))
        mycoordination_matrix = calc_coordination(
            support, kernel=mykernel, debugging=debugging
        )
        outer = np.copy(mycoordination_matrix)
        outer[np.nonzero(outer)] = 1
        if mykernel.shape == np.ones((9, 9, 9)).shape:
            outer[mycoordination_matrix > 300] = (
                0  # start with a larger object, the mean surface amplitude is ~ 5%
            )
        else:
            raise ValueError("Kernel not yet implemented")

        outer[mycoordination_matrix == 0] = 1  # corresponds to outside the crystal
        if debugging:
            gu.multislices_plot(
                outer,
                width_z=width_z,
                width_y=width_y,
                width_x=width_x,
                vmin=0,
                vmax=1,
                title="Outer matrix",
                cmap=cmap,
            )

        ############################################################################
        # remove layer by layer until the correct isosurface is reached on average #
        ############################################################################
        nb_voxels = 1  # initialize this counter which corresponds to
        # the number of voxels not included in outer
        idx = 0
        # is larger than mythreshold
        while nb_voxels > 0:  # nb of voxels not included in outer
            # first step: find the first underlayer
            mycoordination_matrix = calc_coordination(
                outer, kernel=mykernel, debugging=debugging
            )
            surface = np.copy(mycoordination_matrix)
            surface[np.nonzero(surface)] = 1
            surface[mycoordination_matrix > 389] = 0  # remove part from outer  389
            outer[mycoordination_matrix > 389] = (
                1  # include points left over by the coordination number selection
            )
            surface[mycoordination_matrix < 362] = 0  # remove part from bulk   311
            # below is to exclude from surface the frame outer part
            surface[0:5, :, :] = 0
            surface[:, 0:5, :] = 0
            surface[:, :, 0:5] = 0
            surface[nbz - 6 : nbz, :, :] = 0
            surface[:, nby - 6 : nby, :] = 0
            surface[:, :, nbx - 6 : nbx] = 0
            if debugging:
                gu.multislices_plot(
                    surface,
                    width_z=width_z,
                    width_y=width_y,
                    width_x=width_x,
                    vmin=0,
                    vmax=1,
                    title="Surface matrix",
                    cmap=cmap,
                )

            # second step: calculate the % of voxels from that layer whose amplitude
            # is lower than support_threshold
            nb_voxels = surface[np.nonzero(surface)].sum()
            keep_voxels = surface[abs(amp) >= support_threshold * max_amp].sum()
            voxels_counter = (
                keep_voxels / nb_voxels
            )  # % of voxels whose amplitude is larger than support_threshold
            mean_amp = np.mean(amp[np.nonzero(surface)].flatten()) / max_amp
            logger.info(
                f"number of surface voxels = {nb_voxels}, "
                f"% of surface voxels above threshold = {100 * voxels_counter:.2f} %, ",
                f"mean surface amplitude = {mean_amp}",
            )
            if mean_amp < support_threshold:
                outer[np.nonzero(surface)] = 1
                idx = idx + 1
            else:
                logger.info(f"Surface of object reached after {idx} iterations")
                break
        support_defect = np.ones((nbz, nby, nbx)) - outer
        support = np.ones((nbz, nby, nbx))
        support[abs(amp) < support_threshold * max_amp] = 0
        # add voxels detected by support_defect
        support[np.nonzero(support_defect)] = 1
    return support


def find_datarange(
    array: np.ndarray,
    plot_margin: Union[int, List[int]] = 10,
    amplitude_threshold: float = 0.1,
    keep_size: bool = False,
) -> List[int]:
    """
    Find the range where data is larger than a threshold.

    It finds the meaningful range of the array where it is larger than the threshold, in
    order to later crop the array to that shape and reduce the memory consumption in
    processing. The range can be larger than the initial data size, which then will need
    to be padded.

    :param array: a non-empty numpy array
    :param plot_margin: user-defined margin to add on each side of the thresholded array
    :param amplitude_threshold: threshold used to define a support from the amplitude
    :param keep_size: set to True in order to keep the dataset full size
    :return: a list of the ranges (centered in the middle of the array) to use in each
     dimension.
    """
    if array.ndim == 0:
        raise ValueError("Empty array provided.")
    if isinstance(plot_margin, int):
        plot_margin = [plot_margin] * array.ndim
    if not isinstance(plot_margin, list):
        raise TypeError(
            f"Expected 'plot_margin' to be a list, got {type(plot_margin)} "
        )
    if len(plot_margin) != array.ndim:
        raise ValueError(
            f"'plot_margin' should be of lenght {array.ndim}, got {len(plot_margin)}"
        )

    if keep_size:
        return [int(val) for val in array.shape]

    support = np.zeros(array.shape)
    support[abs(array) > amplitude_threshold * abs(array).max()] = 1
    non_zero_indices = np.nonzero(support)
    min_half_width_per_axis: List[int] = []
    try:
        for idx, nb_elements in enumerate(array.shape):
            min_half_width_per_axis.append(
                min(
                    min(non_zero_indices[idx]),
                    nb_elements - 1 - max(non_zero_indices[idx]),
                )
            )
    except ValueError as e:
        raise ModulusBelowThreshold(
            f"No voxel above the provided threshold {amplitude_threshold}"
        ) from e
    return [
        int(
            (nb_elements // 2 - min_half_width_per_axis[idx] + plot_margin[idx]) * 2
            + nb_elements % 2
        )
        for idx, nb_elements in enumerate(array.shape)
    ]


def flip_reconstruction(obj, debugging=False, **kwargs):
    """
    Calculate the conjugate object giving the same diffracted intensity as 'obj'.

    :param obj: 3D reconstructed complex object
    :param debugging: set to True to see plots
    :type debugging: bool
    :param kwargs:

     - 'cmap': str, name of the colormap

    :return: the flipped complex object
    """
    valid.valid_ndarray(arrays=obj, ndim=3)
    cmap = kwargs.get("cmap", "turbo")
    flipped_obj = ifftn(ifftshift(np.conj(fftshift(fftn(obj)))))
    if debugging:
        gu.multislices_plot(
            abs(obj),
            vmin=0,
            sum_frames=False,
            plot_colorbar=True,
            title="Initial object",
            cmap=cmap,
        )
        gu.multislices_plot(
            abs(flipped_obj),
            vmin=0,
            sum_frames=False,
            plot_colorbar=True,
            title="Flipped object",
            cmap=cmap,
        )
    return flipped_obj


def gaussian_kernel(ndim, kernel_length=21, sigma=3, debugging=False):
    """
    Generate 2D or 3D Gaussian kernels.

    :param ndim: number of dimensions of the kernel, 2 or 3
    :param kernel_length: length in pixels of the filtering kernel
    :param sigma: sigma of the gaussian pdf
    :param debugging: True to see plots
    :return: a 2D or 3D Gaussian kernel
    """
    if kernel_length % 2 == 0:
        raise ValueError("kernel_length should be an even number")
    half_range = kernel_length // 2
    kernel_1d = norm.pdf(np.arange(-half_range, half_range + 1, 1), 0, sigma)

    if ndim == 2:
        kernel = np.ones((kernel_length, kernel_length))
        for idy in range(kernel_length):
            kernel[idy, :] = kernel_1d[idy] * kernel_1d

        if debugging:
            plt.figure()
            plt.imshow(kernel)
            plt.colorbar()
            plt.title("Gaussian kernel")
            plt.pause(0.1)

    elif ndim == 3:
        kernel_2d = np.ones((kernel_length, kernel_length))
        kernel = np.ones((kernel_length, kernel_length, kernel_length))
        for idz in range(kernel_length):
            kernel_2d[idz, :] = kernel_1d[idz] * kernel_1d
            for idy in range(kernel_length):
                kernel[idz, idy] = kernel_2d[idz, idy] * kernel_1d

        if debugging:
            plt.figure()
            plt.imshow(kernel[half_range, :, :])
            plt.colorbar()
            plt.title("Central slice of the Gaussian kernel")
            plt.pause(0.1)
    else:
        raise ValueError("This function generates only 2D or 3D kernels")

    return kernel


def get_opticalpath(
    support: np.ndarray,
    direction: str,
    k: np.ndarray,
    voxel_size: Optional[Union[float, Tuple[float, float, float]]] = None,
    debugging: bool = False,
    **kwargs,
) -> np.ndarray:
    """
    Calculate the optical path for refraction/absorption corrections in the crystal.

    'k' should be in the same basis (crystal or laboratory frame) as the data. For
    xrayutilities, the data is orthogonalized in crystal frame.

    :param support: 3D array, support used for defining the object
    :param direction: "in" or "out" , incident or diffracted wave
    :param k: vector for the incident or diffracted wave depending on direction,
     expressed in an orthonormal frame (without taking in to account the different
     voxel size in each dimension)
    :param voxel_size: tuple, actual voxel size in z, y, and x (CXI convention)
    :param debugging: boolena, True to see plots
    :param kwargs:

     - 'cmap': str, name of the colormap
     - width_z: size of the area to plot in z (axis 0), centered on the middle
       of the initial array
     - width_y: size of the area to plot in y (axis 1), centered on the middle
       of the initial array
     - width_x: size of the area to plot in x (axis 2), centered on the middle
       of the initial array

    :return: the optical path in nm, of the same shape as mysupport
    """
    #########################
    # check and load kwargs #
    #########################
    valid.valid_kwargs(
        kwargs=kwargs,
        allowed_kwargs={"cmap", "width_z", "width_y", "width_x"},
        name="kwargs",
    )
    cmap = kwargs.get("cmap", "turbo")
    width_z = kwargs.get("width_z")
    valid.valid_item(
        value=width_z,
        allowed_types=int,
        min_excluded=0,
        allow_none=True,
        name="width_z",
    )
    width_y = kwargs.get("width_y")
    valid.valid_item(
        value=width_y,
        allowed_types=int,
        min_excluded=0,
        allow_none=True,
        name="width_y",
    )
    width_x = kwargs.get("width_x")
    valid.valid_item(
        value=width_x,
        allowed_types=int,
        min_excluded=0,
        allow_none=True,
        name="width_x",
    )

    #########################
    # check some parameters #
    #########################
    valid.valid_ndarray(arrays=support, ndim=3)

    voxel_size = voxel_size or (1, 1, 1)
    if isinstance(voxel_size, (float, int)):
        voxel_size = (voxel_size,) * 3
    valid.valid_container(
        voxel_size,
        container_types=(tuple, list),
        length=3,
        item_types=Real,
        min_excluded=0,
        name="voxel_size",
    )

    ############################################################
    # correct k for the different voxel size in each dimension #
    # (k is expressed in an orthonormal basis)                 #
    ############################################################
    rescaled_k = [k[i] * voxel_size[i] for i in range(3)]

    ###################################################################
    # find the extent of the object, to optimize the calculation time #
    ###################################################################
    nbz, nby, nbx = support.shape
    path = np.zeros((nbz, nby, nbx), dtype=float)
    indices_support = np.nonzero(support)
    min_z = indices_support[0].min()
    max_z = indices_support[0].max() + 1  # last point not included in range()
    min_y = indices_support[1].min()
    max_y = indices_support[1].max() + 1  # last point not included in range()
    min_x = indices_support[2].min()
    max_x = indices_support[2].max() + 1  # last point not included in range()

    #############################################
    # normalize k, now it is in units of voxels #
    #############################################
    if direction == "in":
        k_norm = (
            -1 / np.linalg.norm(rescaled_k) * np.asarray(rescaled_k)
        )  # we will work with -k_in
    else:  # "out"
        k_norm = 1 / np.linalg.norm(rescaled_k) * np.asarray(rescaled_k)

    #############################################
    # calculate the optical path for each voxel #
    #############################################
    for idz in range(min_z, max_z, 1):
        for idy in range(min_y, max_y, 1):
            for idx in range(min_x, max_x, 1):
                stop_flag = False
                counter = support[
                    idz, idy, idx
                ]  # include also the pixel if it belongs to the support
                pixel = np.array(
                    [idz, idy, idx]
                )  # pixel for which the optical path is calculated
                # beware, the support could be 0 at some voxel inside the object also,
                # but the loop should continue until it reaches the end of the box
                # (min_z, max_z, min_y, max_y, min_x, max_x)
                while not stop_flag:
                    pixel = pixel + k_norm  # add unitary translation in -k_in direction
                    coords = np.rint(pixel)
                    stop_flag = True
                    if (
                        (min_z <= coords[0] <= max_z)
                        and (min_y <= coords[1] <= max_y)
                        and (min_x <= coords[2] <= max_x)
                    ):
                        counter = (
                            counter
                            + support[int(coords[0]), int(coords[1]), int(coords[2])]
                        )
                        stop_flag = False

                # For each voxel, counter is the number of steps along the unitary
                # k vector where the support is non zero. Now we need to convert this
                # into nm using the voxel size, different in each dimension
                endpoint = (
                    np.array([idz, idy, idx]) + counter * k_norm
                )  # indices of the final voxel
                path[idz, idy, idx] = np.sqrt(
                    ((np.rint(endpoint[0]) - idz) * voxel_size[0]) ** 2
                    + ((np.rint(endpoint[1]) - idy) * voxel_size[1]) ** 2
                    + ((np.rint(endpoint[2]) - idx) * voxel_size[2]) ** 2
                )

    ##################
    # debugging plot #
    ##################
    if debugging:
        print(
            "Optical path calculation, support limits "
            "(start_z, stop_z, start_y, stop_y, start_x, stop_x):"
            f"{min_z}, {max_z}, {min_y}, {max_y}, {min_x}, {max_x}"
        )
        gu.multislices_plot(
            support,
            width_z=width_z,
            width_y=width_y,
            width_x=width_x,
            vmin=0,
            vmax=1,
            sum_frames=False,
            title="Support for optical path",
            is_orthogonal=True,
            reciprocal_space=False,
            cmap=cmap,
        )

    ###########################################
    # apply a mean filter to reduce artefacts #
    ###########################################
    # the path should be averaged only in the support defined by the isosurface
    path = mean_filter(
        array=path,
        support=support,
        half_width=1,
        title="Optical path",
        debugging=debugging,
        cmap=cmap,
    )

    return path


def get_strain(
    phase,
    planar_distance,
    voxel_size,
    reference_axis="y",
    extent_phase=2 * pi,
    method="default",
    debugging=False,
    **kwargs,
):
    """
    Calculate the 3D strain array.

    :param phase: 3D phase array (do not forget the -1 sign if the phasing algorithm
     is python or matlab-based)
    :param planar_distance: the planar distance of the material corresponding to
     the measured Bragg peak
    :param voxel_size: float or tuple of three floats, the voxel size of the
     phase array in nm
    :param reference_axis: the axis of the array along which q is aligned:
     'x', 'y' or 'z' (CXI convention)
    :param extent_phase: range for phase wrapping, specify it when the phase spans
     over more than 2*pi
    :param method: 'default' or 'defect'. If 'defect', will offset the phase
     in a loop and keep the smallest value for the strain (Felix Hofmann's method 2019).
    :param debugging: True to see plots
    :param kwargs:

     - 'cmap': str, name of the colormap

    :return: the strain 3D array
    """
    # check some parameters
    cmap = kwargs.get("cmap", "turbo")
    valid.valid_ndarray(arrays=phase, ndim=3)
    if reference_axis not in {"x", "y", "z"}:
        raise ValueError("The reference axis should be 'x', 'y' or 'z'")
    if isinstance(voxel_size, Number):
        voxel_size = (voxel_size,) * 3
    valid.valid_container(
        voxel_size,
        container_types=(tuple, list),
        length=3,
        item_types=Real,
        name="postprocessing_utils.get_strain",
        min_excluded=0,
    )

    strain = np.inf * np.ones(phase.shape)
    if method == "defect":
        offsets = 2 * np.pi / 10 * np.linspace(-10, 10, num=11)
        print(
            "Strain method = defect, the following phase offsets will be processed:",
            offsets,
        )
    else:  # 'default'
        offsets = (0,)

    for offset in offsets:
        # offset the phase
        if method == "defect":
            temp_phase = np.copy(phase)
            temp_phase = temp_phase + offset
            # wrap again the offseted phase
            temp_phase = util.wrap(
                obj=temp_phase, start_angle=-extent_phase / 2, range_angle=extent_phase
            )
        else:  # no need to copy the phase, offset = 0
            temp_phase = phase

        # calculate the strain for this offset
        if reference_axis == "x":
            _, _, temp_strain = np.gradient(
                planar_distance / (2 * np.pi) * temp_phase, voxel_size[2]
            )  # q is along x after rotating the crystal
        elif reference_axis == "y":
            _, temp_strain, _ = np.gradient(
                planar_distance / (2 * np.pi) * temp_phase, voxel_size[1]
            )  # q is along y after rotating the crystal
        else:  # "z"
            temp_strain, _, _ = np.gradient(
                planar_distance / (2 * np.pi) * temp_phase, voxel_size[0]
            )  # q is along z after rotating the crystal

        # update the strain values
        strain = np.where(abs(strain) < abs(temp_strain), strain, temp_strain)
        if debugging:
            gu.multislices_plot(
                temp_phase,
                sum_frames=False,
                title="Offseted phase",
                vmin=-np.pi,
                vmax=np.pi,
                plot_colorbar=True,
                is_orthogonal=True,
                reciprocal_space=False,
                cmap=cmap,
            )
            gu.multislices_plot(
                strain,
                sum_frames=False,
                title="strain",
                vmin=-0.002,
                vmax=0.002,
                plot_colorbar=True,
                is_orthogonal=True,
                reciprocal_space=False,
                cmap=cmap,
            )
    return strain


def match_data_range_for_interpolation(
    old_shape: Tuple[int, int, int],
    old_voxelsizes: Tuple[int, int, int],
    new_voxelsizes: Tuple[int, int, int],
) -> Tuple[int, int, int]:
    """
    Find the shape of the array that keeps the data span constant with new voxel sizes.

    :param old_shape: current shape of the 3D array
    :param old_voxelsizes: current voxel size of the 3D array in each dimension
    :param new_voxelsizes: voxel size of the 3D array in each dimension for the
     interpolation
    :return: the shape that keep the data span constant with the new voxel sizes.
    """
    nbz, nby, nbx = old_shape
    data_extent = (
        nbz * old_voxelsizes[0],
        nby * old_voxelsizes[1],
        nbx * old_voxelsizes[2],
    )
    new_extent = (
        nbz * new_voxelsizes[0],
        nby * new_voxelsizes[1],
        nbx * new_voxelsizes[2],
    )
    pad_size = [
        (data_extent[0] - new_extent[0]) // new_voxelsizes[0] + 1,
        (data_extent[1] - new_extent[1]) // new_voxelsizes[1] + 1,
        (data_extent[2] - new_extent[2]) // new_voxelsizes[2] + 1,
    ]
    pad_size = [int(val) if val >= 0 else 0 for val in pad_size]
    return nbz + pad_size[0], nby + pad_size[1], nbx + pad_size[2]


def mean_filter(
    array,
    support,
    half_width=0,
    width_z=None,
    width_y=None,
    width_x=None,
    vmin=np.nan,
    vmax=np.nan,
    title="Object",
    debugging=False,
    **kwargs,
):
    """
    Apply a mean filter to an object defined by a support.

    Only voxels belonging to the object are taken into account, taking care of the
    object's surface.

    :param array: 3D array to be averaged
    :param support: support used for averaging
    :param half_width: half_width of the 2D square averaging window,
     0 means no averaging, 1 is one pixel away...
    :param width_z: size of the area to plot in z (axis 0), centered on the middle
     of the initial array
    :param width_y: size of the area to plot in y (axis 1), centered on the middle
     of the initial array
    :param width_x: size of the area to plot in x (axis 2), centered on the middle
     of the initial array
    :param vmin: real number, lower boundary for the colorbar of the plots
    :param vmax: real number, higher boundary for the colorbar of the plots
    :param title: str, title for the plots
    :param debugging: bool, True to see plots
    :param kwargs:

     - 'cmap': str, name of the colormap
     - 'logger': an optional logger

    :return: averaged array of the same shape as the input array
    """
    #########################
    # check some parameters #
    #########################
    cmap = kwargs.get("cmap", "turbo")
    logger = kwargs.get("logger", module_logger)
    valid.valid_ndarray(arrays=(array, support), ndim=3)
    valid.valid_item(half_width, allowed_types=int, min_included=0, name="half_width")
    valid.valid_container(title, container_types=str, name="title")
    valid.valid_item(vmin, allowed_types=Real, name="vmin")
    valid.valid_item(vmax, allowed_types=Real, name="vmax")
    valid.valid_item(debugging, allowed_types=bool, name="debugging")
    valid.valid_item(
        value=width_z,
        allowed_types=int,
        min_excluded=0,
        allow_none=True,
        name="width_z",
    )
    valid.valid_item(
        value=width_y,
        allowed_types=int,
        min_excluded=0,
        allow_none=True,
        name="width_y",
    )
    valid.valid_item(
        value=width_x,
        allowed_types=int,
        min_excluded=0,
        allow_none=True,
        name="width_x",
    )

    #########################
    # apply the mean filter #
    #########################
    if half_width != 0:
        if debugging:
            gu.multislices_plot(
                array,
                width_z=width_z,
                width_y=width_y,
                width_x=width_x,
                vmin=vmin,
                vmax=vmax,
                title=title + " before averaging",
                plot_colorbar=True,
                cmap=cmap,
            )
            gu.multislices_plot(
                support,
                width_z=width_z,
                width_y=width_y,
                width_x=width_x,
                vmin=0,
                vmax=1,
                title="Support for averaging",
                cmap=cmap,
            )

        nonzero_pixels = np.argwhere(support != 0)
        new_values = np.zeros((nonzero_pixels.shape[0], 1), dtype=array.dtype)
        counter = 0
        for indx in range(nonzero_pixels.shape[0]):
            piz = nonzero_pixels[indx, 0]
            piy = nonzero_pixels[indx, 1]
            pix = nonzero_pixels[indx, 2]
            tempo_support = support[
                piz - half_width : piz + half_width + 1,
                piy - half_width : piy + half_width + 1,
                pix - half_width : pix + half_width + 1,
            ]
            nb_points = tempo_support.sum()
            temp_phase = array[
                piz - half_width : piz + half_width + 1,
                piy - half_width : piy + half_width + 1,
                pix - half_width : pix + half_width + 1,
            ]
            if temp_phase.size != 0:
                value = temp_phase[np.nonzero(tempo_support)].sum() / nb_points
                new_values[indx] = value
            else:
                counter = counter + 1
        for indx in range(nonzero_pixels.shape[0]):
            array[
                nonzero_pixels[indx, 0],
                nonzero_pixels[indx, 1],
                nonzero_pixels[indx, 2],
            ] = new_values[indx]
        if debugging:
            gu.multislices_plot(
                array,
                width_z=width_z,
                width_y=width_y,
                width_x=width_x,
                vmin=vmin,
                vmax=vmax,
                title=title + " after averaging",
                plot_colorbar=True,
                cmap=cmap,
            )
        if counter != 0:
            logger.info(
                f"There were {counter} voxels for which phase could not be averaged"
            )
    return array


def ortho_modes(array_stack, nb_mode=None, method="eig", verbose=False):
    """
    Decompose an object into a set of orthogonal modes.

    It finds modes from a N+1 dimensional array or a list/tuple of N-dimensional
    arrays. The decomposition is such that the total intensity (i.e. (abs(m)**2).sum(
    )) is conserved. Adapted from PyNX.

     :param array_stack: the stack of modes to orthogonalize along the first dimension.
     :param nb_mode: the maximum number of modes to be returned. If None,
      all are returned. This is useful if nb_mode is used, and only a partial list
      of modes is returned.
     :param method: either 'eig' to use eigenvalue decomposition or 'svd' to use
      singular value decomposition.
     :param verbose: set it to True to have more printed comments
     :return: an array (modes) with the same shape as given in input, but with
      orthogonal modes, i.e. (mo[i]*mo[j].conj()).sum()=0 for i!=j. The modes are
      sorted by decreasing norm. If nb_mode is not None, only modes up
      to nb_mode will be returned.
    """
    valid.valid_ndarray(arrays=array_stack, ndim=4)

    # array stack has the shape: (nb_arrays, L, M, N)
    nb_arrays = array_stack.shape[0]
    array_size = array_stack[0].size  # the size of individual arrays is L x M x N

    if method == "eig":
        my_matrix = np.array(
            [
                [np.vdot(array2, array1) for array1 in array_stack]
                for array2 in array_stack
            ]
        )
        # array of shape (nb_arrays,nb_arrays)
        eigenvalues, eigenvectors = np.linalg.eig(
            my_matrix
        )  # the number of eigenvalues is nb_arrays
    elif method == "svd":  # Singular value decomposition
        my_matrix = np.reshape(array_stack, (nb_arrays, array_size))
        eigenvectors, eigenvalues, _ = scipy.linalg.svd(
            my_matrix, full_matrices=False, compute_uv=True
        )
        # my_matrix = eigenvectors x S x Vh,
        # where S is a suitably shaped matrix of zeros with main diagonal s
        # The shapes are (M, K) for the eigenvectors and (K, N)
        # for the unitary matrix Vh where K = min(M, N)
        # Here, M is the number of reconstructions nb_arrays,
        # N is the size of a reconstruction array_size
    else:
        raise ValueError('Incorrect value for parameter "method"')

    sort_indices = (
        -eigenvalues
    ).argsort()  # returns the indices that would sort eigenvalues in descending order
    print("\neigenvalues", eigenvalues)
    eigenvectors = eigenvectors[
        :, sort_indices
    ]  # sort eigenvectors using sort_indices, same shape as my_matrix

    for idx in range(len(sort_indices)):
        if eigenvectors[abs(eigenvectors[:, idx]).argmax(), idx].real < 0:
            eigenvectors[:, idx] *= -1

    modes = np.array(
        [
            sum(array_stack[i] * eigenvectors[i, j] for i in range(nb_arrays))
            for j in range(nb_arrays)
        ]
    )
    # # the double nested comprehension list above is equivalent to the following code:
    # modes = np.zeros(array_stack.shape, dtype=complex)
    # for j in range(nb_arrays):
    #     temp = np.zeros(array_stack[0].shape, dtype=complex)
    #     for i in range(nb_arrays):
    #         temp += array_stack[i] * eigenvectors[i, j]
    #     modes[j] = temp

    if verbose:
        print("Orthonormal decomposition coefficients (rows)")
        print(
            np.array2string(
                (eigenvectors.transpose()),
                threshold=10,
                precision=3,
                floatmode="fixed",
                suppress_small=True,
            )
        )

    if nb_mode is not None:
        nb_mode = min(nb_arrays, nb_mode)
    else:
        nb_mode = nb_arrays

    weights = (
        np.array([(abs(modes[i]) ** 2).sum() for i in range(nb_arrays)])
        / (abs(modes) ** 2).sum()
    )

    return modes[:nb_mode], eigenvectors, weights


def regrid(array, old_voxelsize, new_voxelsize):
    """
    Interpolate real space data on a grid with a different voxel size.

    :param array: 3D array, the object to be interpolated
    :param old_voxelsize: tuple, actual voxel size in z, y, and x (CXI convention)
    :param new_voxelsize: tuple, desired voxel size for the interpolation in
     z, y, and x (CXI convention)
    :return: obj interpolated using the new voxel sizes
    """
    valid.valid_ndarray(arrays=array, ndim=3)

    if isinstance(old_voxelsize, Number):
        old_voxelsize = (old_voxelsize,) * 3
    valid.valid_container(
        old_voxelsize,
        container_types=(tuple, list),
        length=3,
        item_types=Real,
        name="postprocessing_utils.regrid",
        min_excluded=0,
    )

    if isinstance(new_voxelsize, Number):
        new_voxelsize = (new_voxelsize,) * 3
    valid.valid_container(
        new_voxelsize,
        container_types=(tuple, list),
        length=3,
        item_types=Real,
        name="postprocessing_utils.regrid",
        min_excluded=0,
    )

    nbz, nby, nbx = match_data_range_for_interpolation(
        old_shape=array.shape,
        old_voxelsizes=old_voxelsize,
        new_voxelsizes=new_voxelsize,
    )

    array = util.crop_pad(
        array=array,
        output_shape=(nbz, nby, nbx),
    )

    old_z = np.arange(-nbz // 2, nbz // 2, 1) * old_voxelsize[0]
    old_y = np.arange(-nby // 2, nby // 2, 1) * old_voxelsize[1]
    old_x = np.arange(-nbx // 2, nbx // 2, 1) * old_voxelsize[2]

    new_z, new_y, new_x = np.meshgrid(
        old_z * new_voxelsize[0] / old_voxelsize[0],
        old_y * new_voxelsize[1] / old_voxelsize[1],
        old_x * new_voxelsize[2] / old_voxelsize[2],
        indexing="ij",
    )

    rgi = RegularGridInterpolator(
        (old_z, old_y, old_x), array, method="linear", bounds_error=False, fill_value=0
    )

    new_array = rgi(
        np.concatenate(
            (
                new_z.reshape((1, new_z.size)),
                new_y.reshape((1, new_y.size)),
                new_x.reshape((1, new_x.size)),
            )
        ).transpose()
    )
    new_array = new_array.reshape((nbz, nby, nbx)).astype(array.dtype)
    return new_array


def remove_offset(
    array,
    support,
    offset_method="com",
    phase_offset=0,
    offset_origin=None,
    title="",
    debugging=False,
    **kwargs,
):
    """
    Remove the offset in a 3D array based on a 3D support.

    :param array: a 3D array
    :param support: A 3D support of the same shape as array, defining the object
    :param offset_method: 'com' or 'mean'. If 'com', the value of array at the center
     of mass of the support will be subtracted to the array. If 'mean', the mean
     value of array on the support will be subtracted to the array.
    :param phase_offset: value to add to the array
    :param offset_origin: If provided, the value of array at this voxel will be
     subtracted to the array.
    :param title: string, used in plot title
    :param debugging: True to see plots
    :param kwargs:

     - 'cmap': str, name of the colormap
     - 'reciprocal_space': True if the object is in reciprocal space
     - 'is_orthogonal': True if the data is in an orthonormal frame. Used for defining
       default plot labels.

    :return: the processed array
    """
    valid.valid_ndarray(arrays=(array, support), ndim=3)
    # check and load kwargs
    valid.valid_kwargs(
        kwargs=kwargs,
        allowed_kwargs={"cmap", "reciprocal_space", "is_orthogonal"},
        name="postprocessing_utils.average_obj",
    )
    cmap = kwargs.get("cmap", "turbo")
    valid.valid_container(
        offset_origin,
        container_types=(tuple, list, np.ndarray),
        item_types=int,
        allow_none=True,
        name="offset_origin",
    )
    reciprocal_space = kwargs.get("reciprocal_space", False)
    is_orthogonal = kwargs.get("is_orthogonal", False)

    if debugging:
        gu.multislices_plot(
            array,
            sum_frames=False,
            plot_colorbar=True,
            title=title + " before offset removal",
            reciprocal_space=reciprocal_space,
            is_orthogonal=is_orthogonal,
            cmap=cmap,
        )

    if phase_offset is None:
        phase_offset = 0

    if offset_origin is None:  # use offset_method to remove the offset
        if offset_method == "com":
            zcom, ycom, xcom = center_of_mass(support)
            zcom, ycom, xcom = (
                int(np.rint(zcom)),
                int(np.rint(ycom)),
                int(np.rint(xcom)),
            )
            print(f"\nCOM at pixels (z, y, x): {zcom, ycom, xcom}")
            print(
                f"Offset at COM(support) of: {array[zcom, ycom, xcom]:.2f} rad",
            )
            array = array - array[zcom, ycom, xcom] + phase_offset
        elif offset_method == "mean":
            array = array - array[support == 1].mean() + phase_offset
        else:
            raise ValueError('Invalid setting for parameter "offset_method"')
    else:
        print(
            "\nOrigin for offset removal at pixels (z, y, x): ",
            offset_origin[0],
            offset_origin[1],
            offset_origin[2],
        )
        print(
            "Offset of "
            f"{array[offset_origin[0], offset_origin[1], offset_origin[2]]:.2f} rad",
        )
        array = (
            array
            - array[offset_origin[0], offset_origin[1], offset_origin[2]]
            + phase_offset
        )

    if debugging:
        gu.multislices_plot(
            array,
            sum_frames=False,
            plot_colorbar=True,
            title=title + " after offset removal",
            reciprocal_space=reciprocal_space,
            is_orthogonal=is_orthogonal,
            cmap=cmap,
        )
    return array


def remove_ramp(
    amp,
    phase,
    initial_shape,
    width_z=None,
    width_y=None,
    width_x=None,
    amplitude_threshold=0.25,
    threshold_gradient=0.2,
    method="gradient",
    ups_factor=2,
    debugging=False,
    **kwargs,
):
    """
    Remove the linear trend in the ramp using its gradient and a threshold n 3D dataset.

    :param amp: 3D array, amplitude of the object
    :param phase: 3D array, phase of the object to be detrended
    :param initial_shape: shape of the FFT used for phasing
    :param width_z: size of the area to plot in z (axis 0), centered on the middle
     of the initial array
    :param width_y: size of the area to plot in y (axis 1), centered on the middle
     of the initial array
    :param width_x: size of the area to plot in x (axis 2), centered on the middle
     of the initial array
    :param amplitude_threshold: threshold used to define the support of the object
     from the amplitude
    :param threshold_gradient: higher threshold used to select valid voxels in
     the gradient array
    :param method: 'gradient' or 'upsampling'
    :param ups_factor: upsampling factor (the original shape will be multiplied by
     this value)
    :param debugging: set to True to see plots
    :type debugging: bool
    :param kwargs:

     - 'cmap': str, name of the colormap
     - 'logger': an optional logger

    :return: normalized amplitude, detrended phase, ramp along z, ramp along y,
     ramp along x
    """
    logger = kwargs.get("logger", module_logger)
    valid.valid_ndarray(arrays=(amp, phase), ndim=3)
    cmap = kwargs.get("cmap", "turbo")
    if method == "upsampling":
        nbz, nby, nbx = (mysize * ups_factor for mysize in initial_shape)
        nb_z, nb_y, nb_x = amp.shape
        myobj = util.crop_pad(amp * np.exp(1j * phase), (nbz, nby, nbx), cmap=cmap)
        if debugging:
            plt.figure()
            plt.imshow(np.log10(abs(myobj).sum(axis=0)))
            plt.title("np.log10(abs(myobj).sum(axis=0))")
            plt.pause(0.1)
        my_fft = fftshift(fftn(ifftshift(myobj)))
        del myobj, amp, phase
        gc.collect()
        if debugging:
            plt.figure()
            # plt.imshow(np.log10(abs(my_fft[nbz//2, :, :])))
            plt.imshow(np.log10(abs(my_fft).sum(axis=0)), cmap=cmap)
            plt.title("np.log10(abs(my_fft).sum(axis=0))")
            plt.pause(0.1)
        zcom, ycom, xcom = center_of_mass(abs(my_fft) ** 4)
        logger.info(f"FFT shape for subpixel shift: ({nbz}, {nby}, {nbx})")
        logger.info(f"COM before subpixel shift: ({zcom}, {ycom}, {xcom})")
        shiftz = zcom - (nbz / 2)
        shifty = ycom - (nby / 2)
        shiftx = xcom - (nbx / 2)

        # phase shift in real space
        buf2ft = fftn(my_fft)  # in real space
        del my_fft
        gc.collect()
        if debugging:
            plt.figure()
            plt.imshow(abs(buf2ft).sum(axis=0), cmap=cmap)
            plt.title("abs(buf2ft).sum(axis=0)")
            plt.pause(0.1)

        z_axis = ifftshift(np.arange(-np.fix(nbz / 2), np.ceil(nbz / 2), 1))
        y_axis = ifftshift(np.arange(-np.fix(nby / 2), np.ceil(nby / 2), 1))
        x_axis = ifftshift(np.arange(-np.fix(nbx / 2), np.ceil(nbx / 2), 1))
        z_axis, y_axis, x_axis = np.meshgrid(z_axis, y_axis, x_axis, indexing="ij")
        greg = buf2ft * np.exp(
            1j
            * 2
            * np.pi
            * (shiftz * z_axis / nbz + shifty * y_axis / nby + shiftx * x_axis / nbx)
        )
        del buf2ft, z_axis, y_axis, x_axis
        gc.collect()
        if debugging:
            plt.figure()
            plt.imshow(abs(greg).sum(axis=0), cmap=cmap)
            plt.title("abs(greg).sum(axis=0)")
            plt.pause(0.1)

        my_fft = ifftn(greg)
        del greg
        gc.collect()
        # end of phase shift in real space

        if debugging:
            plt.figure()
            plt.imshow(np.log10(abs(my_fft).sum(axis=0)), cmap=cmap)
            plt.title("centered np.log10(abs(my_fft).sum(axis=0))")
            plt.pause(0.1)

        logger.info(f"COM after subpixel shift: {center_of_mass(abs(my_fft) ** 4)}")
        myobj = fftshift(ifftn(ifftshift(my_fft)))
        del my_fft
        gc.collect()
        if debugging:
            plt.figure()
            plt.imshow(abs(myobj).sum(axis=0), cmap=cmap)
            plt.title("centered abs(myobj).sum(axis=0)")
            plt.pause(0.1)

        myobj = util.crop_pad(
            myobj, (nb_z, nb_y, nb_x)
        )  # return to the initial shape of myamp
        print(
            "Upsampling: shift_z, shift_y, shift_x: "
            f"({shiftz:.3f}, {shifty:.3f}, {shiftx:.3f}) pixels"
        )
        return abs(myobj) / abs(myobj).max(), np.angle(myobj), shiftz, shifty, shiftx

    # method='gradient'
    # define the support from the amplitude
    nbz, nby, nbx = amp.shape
    mysupport = np.zeros((nbz, nby, nbx))
    mysupport[amp > amplitude_threshold * abs(amp).max()] = 1

    # axis 0 (Z)
    mygradz, _, _ = np.gradient(phase, 1)

    mysupportz = np.zeros((nbz, nby, nbx))
    mysupportz[abs(mygradz) < threshold_gradient] = 1
    mysupportz = mysupportz * mysupport
    if mysupportz.sum() == 0:
        raise ValueError(
            "No voxel below the threshold, raise the parameter threshold_gradient"
        )
    myrampz = mygradz[mysupportz == 1].mean()
    if debugging:
        gu.multislices_plot(
            mygradz,
            plot_colorbar=True,
            width_z=width_z,
            width_y=width_y,
            width_x=width_x,
            vmin=-threshold_gradient,
            vmax=threshold_gradient,
            title="Phase gradient along Z",
            cmap=cmap,
        )
        gu.multislices_plot(
            mysupportz,
            width_z=width_z,
            width_y=width_y,
            width_x=width_x,
            vmin=0,
            vmax=1,
            title="Thresholded support along Z",
            cmap=cmap,
        )
    del mysupportz, mygradz
    gc.collect()

    # axis 1 (Y)
    _, mygrady, _ = np.gradient(phase, 1)
    mysupporty = np.zeros((nbz, nby, nbx))
    mysupporty[abs(mygrady) < threshold_gradient] = 1
    mysupporty = mysupporty * mysupport
    if mysupporty.sum() == 0:
        raise ValueError(
            "No voxel below the threshold, raise the parameter threshold_gradient"
        )
    myrampy = mygrady[mysupporty == 1].mean()
    if debugging:
        gu.multislices_plot(
            mygrady,
            plot_colorbar=True,
            width_z=width_z,
            width_y=width_y,
            width_x=width_x,
            vmin=-threshold_gradient,
            vmax=threshold_gradient,
            title="Phase gradient along Y",
            cmap=cmap,
        )
        gu.multislices_plot(
            mysupporty,
            width_z=width_z,
            width_y=width_y,
            width_x=width_x,
            vmin=0,
            vmax=1,
            title="Thresholded support along Y",
            cmap=cmap,
        )
    del mysupporty, mygrady
    gc.collect()

    # axis 2 (X)
    _, _, mygradx = np.gradient(phase, 1)
    mysupportx = np.zeros((nbz, nby, nbx))
    mysupportx[abs(mygradx) < threshold_gradient] = 1
    mysupportx = mysupportx * mysupport
    if mysupportx.sum() == 0:
        raise ValueError(
            "No voxel below the threshold, raise the parameter threshold_gradient"
        )
    myrampx = mygradx[mysupportx == 1].mean()
    if debugging:
        gu.multislices_plot(
            mygradx,
            plot_colorbar=True,
            width_z=width_z,
            width_y=width_y,
            width_x=width_x,
            vmin=-threshold_gradient,
            vmax=threshold_gradient,
            title="Phase gradient along X",
            cmap=cmap,
        )
        gu.multislices_plot(
            mysupportx,
            width_z=width_z,
            width_y=width_y,
            width_x=width_x,
            vmin=0,
            vmax=1,
            title="Thresholded support along X",
            cmap=cmap,
        )
    del mysupportx, mygradx, mysupport
    gc.collect()

    myz, myy, myx = np.meshgrid(
        np.arange(0, nbz, 1),
        np.arange(0, nby, 1),
        np.arange(0, nbx, 1),
        indexing="ij",
    )

    logger.info(
        "Gradient: phase_ramp_z, phase_ramp_y, phase_ramp_x: "
        f"({myrampz:.3f} rad, {myrampy:.3f} rad, {myrampx:.3f} rad)"
    )
    phase = phase - myz * myrampz - myy * myrampy - myx * myrampx
    return amp, phase, myrampz, myrampy, myrampx


def remove_ramp_2d(
    amp,
    phase,
    initial_shape,
    width_y=None,
    width_x=None,
    amplitude_threshold=0.25,
    threshold_gradient=0.2,
    method="gradient",
    ups_factor=2,
    debugging=False,
):
    """
    Remove the linear trend in the ramp using its gradient and a threshold.

    This function can be used for a 2D dataset.

    :param amp: 2D array, amplitude of the object
    :param phase: 2D array, phase of the object to be detrended
    :param initial_shape: shape of the FFT used for phasing
    :param width_y: size of the area to plot in y (vertical axis), centered on
     the middle of the initial array
    :param width_x: size of the area to plot in x (horizontal axis), centered on
     the middle of the initial array
    :param amplitude_threshold: threshold used to define the support of the object
     from the amplitude
    :param threshold_gradient: higher threshold used to select valid voxels in
     the gradient array
    :param method: 'gradient' or 'upsampling'
    :param ups_factor: upsampling factor (the original shape will be multiplied
     by this value)
    :param debugging: set to True to see plots
    :type debugging: bool
    :return: normalized amplitude, detrended phase, ramp along y, ramp along x
    """
    valid.valid_ndarray(arrays=(amp, phase), ndim=2)

    if method == "upsampling":
        nby, nbx = (mysize * ups_factor for mysize in initial_shape)
        nb_y, nb_x = amp.shape
        myobj = util.crop_pad(amp * np.exp(1j * phase), (nby, nbx))
        if debugging:
            plt.figure()
            plt.imshow(np.log10(abs(myobj)))
            plt.title("np.log10(abs(myobj))")
            plt.pause(0.1)
        my_fft = fftshift(fftn(ifftshift(myobj)))
        del myobj, amp, phase
        gc.collect()
        if debugging:
            plt.figure()
            # plt.imshow(np.log10(abs(my_fft[nbz//2, :, :])))
            plt.imshow(np.log10(abs(my_fft)))
            plt.title("np.log10(abs(my_fft))")
            plt.pause(0.1)
        ycom, xcom = center_of_mass(abs(my_fft) ** 4)
        print("FFT shape for subpixel shift:", nby, nbx)
        print("COM before subpixel shift", ycom, ",", xcom)
        shifty = ycom - (nby / 2)
        shiftx = xcom - (nbx / 2)

        # phase shift in real space
        buf2ft = fftn(my_fft)  # in real space
        del my_fft
        gc.collect()
        if debugging:
            plt.figure()
            plt.imshow(abs(buf2ft))
            plt.title("abs(buf2ft)")
            plt.pause(0.1)

        y_axis = ifftshift(np.arange(-np.fix(nby / 2), np.ceil(nby / 2), 1))
        x_axis = ifftshift(np.arange(-np.fix(nbx / 2), np.ceil(nbx / 2), 1))
        y_axis, x_axis = np.meshgrid(y_axis, x_axis, indexing="ij")
        greg = buf2ft * np.exp(
            1j * 2 * np.pi * (shifty * y_axis / nby + shiftx * x_axis / nbx)
        )
        del buf2ft, y_axis, x_axis
        gc.collect()
        if debugging:
            plt.figure()
            plt.imshow(abs(greg))
            plt.title("abs(greg)")
            plt.pause(0.1)

        my_fft = ifftn(greg)
        del greg
        gc.collect()
        # end of phase shift in real space

        if debugging:
            plt.figure()
            plt.imshow(np.log10(abs(my_fft)))
            plt.title("centered np.log10(abs(my_fft))")
            plt.pause(0.1)

        print("COM after subpixel shift", center_of_mass(abs(my_fft) ** 4))
        myobj = fftshift(ifftn(ifftshift(my_fft)))
        del my_fft
        gc.collect()
        if debugging:
            plt.figure()
            plt.imshow(abs(myobj))
            plt.title("centered abs(myobj)")
            plt.pause(0.1)

        myobj = util.crop_pad_2d(
            myobj, (nb_y, nb_x)
        )  # return to the initial shape of myamp
        print(f"Upsampling: (shift_y, shift_x) = ({shifty:.3f},{shiftx:.3f}) pixels")
        return abs(myobj) / abs(myobj).max(), np.angle(myobj)

    # method='gradient'
    # define the support from the amplitude
    nby, nbx = amp.shape
    mysupport = np.zeros((nby, nbx))
    mysupport[amp > amplitude_threshold * abs(amp).max()] = 1

    # axis 0 (Y)
    mygrady, _ = np.gradient(phase, 1)
    mysupporty = np.zeros((nby, nbx))
    mysupporty[abs(mygrady) < threshold_gradient] = 1
    mysupporty = mysupporty * mysupport
    myrampy = mygrady[mysupporty == 1].mean()
    if debugging:
        gu.imshow_plot(
            array=mygrady,
            width_v=width_y,
            width_h=width_x,
            vmin=-0.2,
            vmax=0.2,
            title="Phase gradient along Y",
        )
        gu.imshow_plot(
            array=mysupporty,
            width_v=width_y,
            width_h=width_x,
            vmin=0,
            vmax=1,
            title="Thresholded support along Y",
        )
    del mysupporty, mygrady
    gc.collect()

    # axis 1 (X)
    _, mygradx = np.gradient(phase, 1)
    mysupportx = np.zeros((nby, nbx))
    mysupportx[abs(mygradx) < threshold_gradient] = 1
    mysupportx = mysupportx * mysupport
    myrampx = mygradx[mysupportx == 1].mean()
    if debugging:
        gu.imshow_plot(
            array=mygradx,
            width_v=width_y,
            width_h=width_x,
            vmin=-0.2,
            vmax=0.2,
            title="Phase gradient along X",
        )
        gu.imshow_plot(
            array=mysupportx,
            width_v=width_y,
            width_h=width_x,
            vmin=0,
            vmax=1,
            title="Thresholded support along X",
        )
    del mysupportx, mygradx, mysupport
    gc.collect()

    myy, myx = np.meshgrid(np.arange(0, nby, 1), np.arange(0, nbx, 1), indexing="ij")

    print(f"Gradient: (Phase_ramp_y, Phase_ramp_x) = ({myrampy:.3f},{myrampx:.3f}) rad")
    phase = phase - myy * myrampy - myx * myrampx
    return amp, phase, myrampy, myrampx


def sort_reconstruction(
    file_path, data_range, amplitude_threshold, sort_method="variance/mean"
) -> List[int]:
    """
    Sort out reconstructions based on the metric 'sort_method'.

    :param file_path: path of the reconstructions to sort out
    :param data_range: data will be cropped or padded to this range
    :param amplitude_threshold: threshold used to define a support from the amplitude
    :param sort_method: method for sorting the reconstructions: 'variance/mean',
     'mean_amplitude', 'variance' or 'volume'
    :return: a list of sorted indices in 'file_path', from the best object to the worst.
    """
    nbfiles = len(file_path)
    zrange, yrange, xrange = data_range

    quality_array = np.ones(
        (nbfiles, 4)
    )  # 1/mean_amp, variance(amp), variance(amp)/mean_amp, 1/volume
    for ii in range(nbfiles):
        obj, _ = util.load_file(file_path[ii])
        print("Opening ", file_path[ii])

        # use the range of interest defined above
        obj = util.crop_pad(obj, [2 * zrange, 2 * yrange, 2 * xrange], debugging=False)
        obj = abs(obj) / abs(obj).max()

        temp_support = np.zeros(obj.shape)
        temp_support[obj > amplitude_threshold] = 1  # only for plotting
        quality_array[ii, 0] = 1 / obj[obj > amplitude_threshold].mean()  # 1/mean(amp)
        quality_array[ii, 1] = np.var(obj[obj > amplitude_threshold])  # var(amp)
        quality_array[ii, 2] = (
            quality_array[ii, 0] * quality_array[ii, 1]
        )  # var(amp)/mean(amp) index of dispersion
        quality_array[ii, 3] = 1 / temp_support.sum()  # 1/volume(support)
        del temp_support
        gc.collect()

        # order reconstructions by minimizing the quality factor
    if sort_method == "mean_amplitude":  # sort by quality_array[:, 0] first
        sorted_obj = np.lexsort(
            (
                quality_array[:, 3],
                quality_array[:, 2],
                quality_array[:, 1],
                quality_array[:, 0],
            )
        )

    elif sort_method == "variance":  # sort by quality_array[:, 1] first
        sorted_obj = np.lexsort(
            (
                quality_array[:, 0],
                quality_array[:, 3],
                quality_array[:, 2],
                quality_array[:, 1],
            )
        )

    elif sort_method == "variance/mean":  # sort by quality_array[:, 2] first
        sorted_obj = np.lexsort(
            (
                quality_array[:, 1],
                quality_array[:, 0],
                quality_array[:, 3],
                quality_array[:, 2],
            )
        )

    elif sort_method == "volume":  # sort by quality_array[:, 3] first
        sorted_obj = np.lexsort(
            (
                quality_array[:, 2],
                quality_array[:, 1],
                quality_array[:, 0],
                quality_array[:, 3],
            )
        )

    else:  # default case, use the index of dispersion
        sorted_obj = np.lexsort(
            (
                quality_array[:, 1],
                quality_array[:, 0],
                quality_array[:, 3],
                quality_array[:, 2],
            )
        )

    print("quality_array")
    print(quality_array)
    print("sorted list", sorted_obj)

    return [int(val) for val in sorted_obj]


def tukey_window(shape, alpha=np.array([0.5, 0.5, 0.5])):
    """
    Create a 3d Tukey window based on shape and the shape parameter alpha.

    :param shape: tuple, shape of the 3d window
    :param alpha: shape parameter of the Tukey window, tuple or ndarray of 3 values
    :return: the 3d Tukey window
    """
    from scipy.signal.windows import tukey

    nbz, nby, nbx = shape
    array_z = tukey(nbz, alpha[0])
    array_y = tukey(nby, alpha[1])
    array_x = tukey(nbx, alpha[2])
    tukey2 = np.ones((nbz, nby))
    tukey3 = np.ones((nbz, nby, nbx))
    for idz in range(nbz):
        tukey2[idz, :] = array_z[idz] * array_y
        for idy in range(nby):
            tukey3[idz, idy] = tukey2[idz, idy] * array_x
    return tukey3


def unwrap(
    obj: np.ndarray,
    support_threshold: float,
    seed: int = 0,
    debugging: bool = True,
    **kwargs,
) -> Tuple[np.ndarray, float]:
    """
    Unwrap the phase of a complex object.

    It is based on skimage.restoration.unwrap_phase. A mask can be applied by
    thresholding the modulus of the object.

    :param obj: array to be unwrapped
    :param support_threshold: relative threshold used to define a support from abs(obj)
    :param seed: int, random seed. Use always the same value if you want a
     deterministic behavior.
    :param debugging: set to True to see plots
    :param kwargs:

     - 'cmap': str, name of the colormap
     - 'reciprocal_space': True if the object is in reciprocal space
     - 'is_orthogonal': True if the data is in an orthonormal frame.
       Used for defining default plot labels.

    :return: unwrapped phase, unwrapped phase range
    """
    valid.valid_ndarray(arrays=obj)
    if support_threshold < 0 or support_threshold > 1:
        raise ValueError(
            "support_threshold is a relative threshold, expected value "
            "between 0 and 1"
        )
    # check and load kwargs
    valid.valid_kwargs(
        kwargs=kwargs,
        allowed_kwargs={"cmap", "reciprocal_space", "is_orthogonal"},
        name="postprocessing_utils.average_obj",
    )
    reciprocal_space = kwargs.get("reciprocal_space", False)
    is_orthogonal = kwargs.get("is_orthogonal", False)

    ndim = obj.ndim
    unwrap_support = np.ones(obj.shape, dtype=int)
    unwrap_support[abs(obj) > support_threshold * abs(obj).max()] = 0
    # 0 is a valid entry for ma.masked_array
    phase_wrapped: np.ndarray = ma.masked_array(np.angle(obj), mask=unwrap_support)

    if debugging and ndim == 3:
        gu.multislices_plot(
            phase_wrapped.data,
            plot_colorbar=True,
            title="Before unwrapping",
            reciprocal_space=reciprocal_space,
            is_orthogonal=is_orthogonal,
            cmap=kwargs.get("cmap", "turbo"),
        )

    phase_unwrapped = unwrap_phase(phase_wrapped, wrap_around=False, seed=seed).data
    phase_unwrapped[np.nonzero(unwrap_support)] = 0
    if debugging and ndim == 3:
        gu.multislices_plot(
            phase_unwrapped,
            plot_colorbar=True,
            title="After unwrapping",
            reciprocal_space=reciprocal_space,
            is_orthogonal=is_orthogonal,
            cmap=kwargs.get("cmap", "turbo"),
        )

    extent_phase = np.ceil(phase_unwrapped.max() - phase_unwrapped.min())
    return phase_unwrapped, extent_phase
