# getimageregistration, dftups, dft_registration and subpixel shift functions for
# aligning arrays: original code from Xianhui Xiao APS Sector 2
# Updated by Ross Harder
# Updated by Steven Leake 30/07/2014
# Changed variable names to make it clearer and put it in CXI convention (z y x)
# J.Carnis 27/04/2018
"""Functions related to the registration and alignement of two arrays."""

import logging
from numbers import Complex, Real
from typing import Sequence, Union

import numpy as np
from numpy.fft import fftn, fftshift, ifftn, ifftshift
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import pearsonr

from ..graph import graph_utils as gu
from ..utils import utilities as util
from ..utils import validation as valid

module_logger = logging.getLogger(__name__)


def align_arrays(
    reference_array,
    shifted_array,
    shift_method="modulus",
    interpolation_method="subpixel",
    support_threshold=None,
    precision=1000,
    verbose=True,
    debugging=False,
    **kwargs,
):
    """
    Align two arrays using dft registration and subpixel shift.

    The shift between arrays can be determined either using the modulus of the arrays
    or a support created from it using a threshold.

    :param reference_array: 3D array, reference object
    :param shifted_array: 3D array to be aligned, same shape as reference_obj
    :param shift_method: 'raw', 'modulus', 'support' or 'skip'. Object to use for the
     determination of the shift. If 'raw', it uses the raw, eventually complex array.
     if 'modulus', it uses the modulus of the array. If 'support', it uses a support
     created by threshold the modulus of the array.
    :param interpolation_method: 'subpixel' for interpolating using subpixel shift,
     'rgi' for interpolating using a RegularGridInterpolator, 'roll' to shift voxels
     by an integral number (the shifts are rounded to the nearest integer)
    :param support_threshold: all points where the normalized modulus is larger than
     this value will be set to 1 in the support.
    :param precision: precision for the DFT registration in 1/pixel
    :param verbose: boolean, True to print comments
    :param debugging: boolean, set to True to see plots
    :return:
     - the aligned array
     - the shift: tuple of floats

    """
    # check some parameters
    cmap = kwargs.get("cmap", "turbo")
    valid.valid_ndarray(
        arrays=(shifted_array, reference_array), ndim=(2, 3), fix_shape=False
    )
    if shift_method not in {"raw", "modulus", "support", "skip"}:
        raise ValueError("shift_method should be 'raw', 'modulus', 'support' or 'skip'")
    if interpolation_method not in {"subpixel", "rgi", "roll"}:
        raise ValueError("shift_method should be 'subpixel', 'rgi' or 'roll'")
    valid.valid_item(verbose, allowed_types=bool, name="verbose")
    valid.valid_item(debugging, allowed_types=bool, name="debugging")

    if shifted_array.shape != reference_array.shape:
        if verbose:
            print(
                "reference_obj and obj do not have the same shape\n",
                reference_array.shape,
                shifted_array.shape,
                "crop/pad obj",
            )
        shifted_array = util.crop_pad(
            array=shifted_array,
            output_shape=reference_array.shape,
            cmap=cmap,
        )

    if shift_method != "skip":
        ##############################################
        # calculate the shift between the two arrays #
        ##############################################
        shift = get_shift(
            reference_array=reference_array,
            shifted_array=shifted_array,
            shift_method=shift_method,
            support_threshold=support_threshold,
            precision=precision,
            verbose=verbose,
        )

        #####################
        # align shifted_obj #
        #####################
        aligned_array = shift_array(
            array=shifted_array,
            shift=shift,
            interpolation_method=interpolation_method,
        )

    else:  # 'skip'
        if verbose:
            print("Skipping alignment")
        aligned_array = shifted_array
        shift = (0,) * shifted_array.ndim

    #################
    # optional plot #
    #################
    if debugging:
        if reference_array.ndim == 3:
            gu.multislices_plot(
                abs(reference_array),
                sum_frames=True,
                title="Reference object",
                cmap=cmap,
            )
            gu.multislices_plot(
                abs(aligned_array),
                sum_frames=True,
                title="Aligned object",
                cmap=cmap,
            )
        else:  # 2D case
            gu.imshow_plot(
                abs(reference_array),
                title="Reference object",
                cmap=cmap,
            )
            gu.imshow_plot(
                abs(aligned_array),
                title="Aligned object",
                cmap=cmap,
            )

    return aligned_array, shift


def align_diffpattern(
    reference_data,
    data,
    mask=None,
    shift_method="raw",
    interpolation_method="roll",
    verbose=True,
    debugging=False,
    **kwargs,
):
    """
    Align two diffraction patterns.

    The alignement can be based either on the shift of the center of mass or on dft
    registration.

    :param reference_data: the first 3D or 2D diffraction intensity array which will
     serve as a reference.
    :param data: the 3D or 2D diffraction intensity array to align.
    :param mask: the 3D or 2D mask corresponding to data
    :param shift_method: 'raw', 'modulus', 'support' or 'skip'. Object to use for the
     determination of the shift. If 'raw', it uses the raw, eventually complex array.
     if 'modulus', it uses the modulus of the array. If 'support', it uses a support
     created by threshold the modulus of the array.
    :param interpolation_method: 'rgi' for RegularGridInterpolator or 'subpixel' for
     subpixel shift
    :param verbose: boolean, True to print comments
    :param debugging: boolean, set to True to see plots
    :return:
     - the shifted data
     - the shifted mask
     - if return_shift, returns a tuple containing the shifts

    """
    #########################
    # check some parameters #
    #########################
    valid.valid_ndarray(arrays=(reference_data, data), ndim=(2, 3), fix_shape=True)
    if mask is not None:
        valid.valid_ndarray(arrays=mask, shape=data.shape)
    if shift_method not in {"raw", "modulus", "support", "skip"}:
        raise ValueError("shift_method should be 'raw', 'modulus', 'support' or 'skip'")
    if interpolation_method not in {"subpixel", "rgi", "roll"}:
        raise ValueError("shift_method should be 'subpixel', 'rgi' or 'roll'")
    valid.valid_item(verbose, allowed_types=bool, name="verbose")
    valid.valid_item(debugging, allowed_types=bool, name="debugging")

    ##################
    # align the data #
    ##################
    data, shift = align_arrays(
        reference_array=reference_data,
        shifted_array=data,
        shift_method=shift_method,
        interpolation_method=interpolation_method,
        verbose=verbose,
        debugging=debugging,
        cmap=kwargs.get("cmap", "turbo"),
    )

    ##############################################
    # shift the optional mask by the same amount #
    ##############################################
    if mask is not None:
        mask = shift_array(
            array=mask,
            shift=shift,
            interpolation_method=interpolation_method,
        )

    ####################################
    # filter the data and mask for nan #
    ####################################
    data, mask = util.remove_nan(data=data, mask=mask)

    if mask is None:
        return data, shift

    return data, mask, shift


def average_arrays(
    avg_obj,
    ref_obj,
    obj,
    support_threshold=0.25,
    correlation_threshold=0.90,
    aligning_option="dft",
    space="reciprocal_space",
    debugging=False,
    **kwargs,
):
    """
    Average two reconstructions after aligning it.

    This function can be used to average a series of arrays within a loop. Alignment is
    performed using either DFT registration or the shift of the center of mass of the
    array. Averaging is processed only if their Pearson cross-correlation after
    alignment is larger than the correlation threshold.

    :param avg_obj: 3D array of complex numbers, current average
    :param ref_obj: 3D array of complex numbers, used as a reference for the alignment
    :param obj: 3D array of complex numbers, array to be aligned with the reference and
     to be added to avg_obj
    :param support_threshold: normalized threshold for the definition of the support. It
     is applied on the modulus of the array
    :param correlation_threshold: float in [0, 1], minimum correlation between two
     dataset to average them
    :param aligning_option: 'com' for center of mass, 'dft' for dft registration and
     subpixel shift
    :param space: 'direct_space' or 'reciprocal_space', in which space the average will
     be performed
    :param debugging: boolean, set to True to see plots
    :param kwargs:

     - 'cmap': str, name of the colormap
     - 'width_z': size of the area to plot in z (axis 0), centered on the middle of
       the initial array
     - 'width_y': size of the area to plot in y (axis 1), centered on the middle of
       the initial array
     - 'width_x': size of the area to plot in x (axis 2), centered on the middle of
       the initial array
     - 'reciprocal_space': True if the object is in reciprocal space, it is used only
       for defining labels in plots
     - 'is_orthogonal': True if the data is in an orthonormal frame. Used for defining
       default plot labels.

    :return: the average complex density
    """
    # check some parameters
    valid.valid_ndarray(arrays=(obj, avg_obj, ref_obj), ndim=3)
    if space not in {"direct_space", "reciprocal_space"}:
        raise ValueError("space should be 'direct_space' or 'reciprocal_space'")
    valid.valid_kwargs(
        kwargs=kwargs,
        allowed_kwargs={
            "cmap",
            "width_z",
            "width_y",
            "width_x",
            "reciprocal_space",
            "is_orthogonal",
        },
        name="postprocessing_utils.average_obj",
    )
    cmap = kwargs.get("cmap", "turbo")
    width_z = kwargs.get("width_z")
    width_y = kwargs.get("width_y")
    width_x = kwargs.get("width_x")
    reciprocal_space = kwargs.get("reciprocal_space", False)
    is_orthogonal = kwargs.get("is_orthogonal", False)

    avg_flag = 0

    #######################################################
    # first iteration of the loop, no running average yet #
    #######################################################
    if avg_obj.sum() == 0:
        avg_obj = ref_obj
        if debugging:
            gu.multislices_plot(
                abs(avg_obj),
                width_z=width_z,
                width_y=width_y,
                width_x=width_x,
                plot_colorbar=True,
                sum_frames=True,
                title="Reference object",
                reciprocal_space=reciprocal_space,
                is_orthogonal=is_orthogonal,
                cmap=cmap,
            )
        return avg_obj, avg_flag

    ###############################################
    # next iterations, update the running average #
    ###############################################

    # align obj
    new_obj, _ = align_arrays(
        reference_array=ref_obj,
        shifted_array=obj,
        shift_method="modulus",
        interpolation_method=aligning_option,
        support_threshold=support_threshold,
        precision=1000,
        verbose=True,
        debugging=debugging,
        cmap=cmap,
    )

    # renormalize new_obj
    new_obj = new_obj / abs(new_obj).max()

    # calculate the correlation between arrays and average them eventually
    correlation = pearsonr(
        np.ndarray.flatten(abs(ref_obj)), np.ndarray.flatten(abs(new_obj))
    )[0]
    if correlation < correlation_threshold:
        print(
            f"pearson cross-correlation = {correlation} too low, "
            "skip this reconstruction"
        )
    else:  # combine the arrays
        print(
            f"pearson-correlation = {correlation}, ",
            "average with this reconstruction",
        )

        if debugging:
            myfig, _, _ = gu.multislices_plot(
                abs(new_obj),
                width_z=width_z,
                width_y=width_y,
                width_x=width_x,
                sum_frames=True,
                plot_colorbar=True,
                title="Aligned object",
                reciprocal_space=reciprocal_space,
                is_orthogonal=is_orthogonal,
                cmap=cmap,
            )
            myfig.text(
                0.60,
                0.30,
                f"pearson-correlation = {correlation:.4f}",
                size=20,
            )

        # update the average either in direct space or in reciprocal space
        if space == "direct_space":
            avg_obj = avg_obj + new_obj
        else:  # "reciprocal_space":
            avg_obj = ifftn(fftn(avg_obj) + fftn(obj))
        avg_flag = 1

    if debugging:
        gu.multislices_plot(
            abs(avg_obj),
            plot_colorbar=True,
            width_z=width_z,
            width_y=width_y,
            width_x=width_x,
            sum_frames=True,
            title="New averaged object",
            reciprocal_space=reciprocal_space,
            is_orthogonal=is_orthogonal,
            cmap=cmap,
        )

    return avg_obj, avg_flag


def calc_new_positions(old_positions: list, shift: Sequence[float]) -> np.ndarray:
    """
    Transform old_positions depending on the shift.

    :param old_positions: list of 1D arrays corresponding to the position of the
     voxels in a regular grid before transformation, in each dimension. For example,
     if the array to be interpolated is 3D, old_positions will be a list of three 1D
     arrays [array0, array1, array2], where array0 describes the voxel positions along
     axis 0, array1 along axis 1 and array2 along axis 2. It does not work if the grid
     is not regular (each coordinate would need to be described by a 3D array instead).
    :param shift: a tuple of floats, corresponding to the shift in each dimension that
     need to be applied to array
    :return: the shifted positions where to interpolate, for the RegularGridInterpolator
    """
    # check parameters
    valid.valid_container(
        old_positions,
        container_types=(tuple, list, np.ndarray),
        item_types=np.ndarray,
        min_length=1,
        name="old_positions",
    )
    valid.valid_container(
        shift,
        container_types=(tuple, list),
        item_types=Real,
        length=len(old_positions),
        name="shift",
    )

    # calculate the new positions
    grids = np.meshgrid(*old_positions, indexing="ij")
    new_positions = [grid - shift[index] for index, grid in enumerate(grids)]
    return np.asarray(
        np.concatenate(
            [
                new_grid.reshape((1, new_grid.size))
                for _, new_grid in enumerate(new_positions)
            ]
        ).transpose()
    )


def dft_registration(buf1ft, buf2ft, ups_factor=100):
    """
    Efficient subpixel image registration by cross-correlation.

    This code gives the same precision as the FFT upsampled cross correlation in a
    small fraction of the computation time and with reduced memory requirements. It
    obtains an initial estimate of the cross-correlation peak by an FFT and then
    refines the shift estimation by upsampling the DFT only in a small neighborhood
    of that estimate by means of a matrix-multiply DFT. With this procedure all the
    image points are used to compute the upsampled cross-correlation.
    Manuel Guizar - Dec 13, 2007

    Portions of this code were taken from code written by Ann M. Kowalczyk
    and James R. Fienup. J.R. Fienup and A.M. Kowalczyk, "Phase retrieval for a
    complex-valued object by using a low-resolution image," J. Opt. Soc. Am. A 7,
    450-458 (1990).

    Citation for this algorithm:
    Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup, "Efficient
    subpixel image registration algorithms," Opt. Lett. 33, 156-158 (2008).

    :param buf1ft: Fourier transform of reference image, DC in (1,1) [DO NOT FFTSHIFT]
    :param buf2ft: Fourier transform of image to register, DC in (1,1) [DO NOT FFTSHIFT]
    :param ups_factor: upsampling factor (integer). Images will be registered to
     within 1/ups_factor of a pixel. For example ups_factor = 20 means the images
     will be registered within 1/20 of a pixel. (default = 1)
    :return:
     - output: [error,diff_phase,net_row_shift,net_col_shift]
     - error: translation invariant normalized RMS error between f and g
     - diff_phase: global phase difference between the two images (should be zero if
       images are non-negative).
     - row_shift, col_shift: pixel shifts between images

    """
    if ups_factor == 0:
        crosscorr_max = np.sum(buf1ft * np.conj(buf2ft))
        rfzero = np.sum(abs(buf1ft) ** 2) / buf1ft.size
        rgzero = np.sum(abs(buf2ft) ** 2) / buf2ft.size
        error = 1.0 - crosscorr_max * np.conj(crosscorr_max) / (rgzero * rfzero)
        error = np.sqrt(np.abs(error))
        diff_phase = np.arctan2(np.imag(crosscorr_max), np.real(crosscorr_max))
        return error, diff_phase

    # Whole-pixel shift - Compute cross-correlation by an IFFT and locate the
    # peak
    if ups_factor == 1:
        row_nb = buf1ft.shape[0]
        column_nb = buf1ft.shape[1]
        crosscorr = ifftn(buf1ft * np.conj(buf2ft))
        _, indices = index_max(crosscorr)
        row_max = indices[0]
        column_max = indices[1]
        crosscorr_max = crosscorr[row_max, column_max]
        rfzero = np.sum(np.abs(buf1ft) ** 2) / (row_nb * column_nb)
        rgzero = np.sum(np.abs(buf2ft) ** 2) / (row_nb * column_nb)
        error = 1.0 - crosscorr_max * np.conj(crosscorr_max) / (rgzero * rfzero)
        error = np.sqrt(np.abs(error))
        diff_phase = np.arctan2(np.imag(crosscorr_max), np.real(crosscorr_max))
        md2 = np.fix(row_nb / 2)
        nd2 = np.fix(column_nb / 2)
        if row_max > md2:
            row_shift = row_max - row_nb
        else:
            row_shift = row_max

        if column_max > nd2:
            col_shift = column_max - column_nb
        else:
            col_shift = column_max

        return error, diff_phase, row_shift, col_shift

    # ups_factor > 1
    # Partial-pixel shift
    # First upsample by a factor of 2 to obtain initial estimate
    # Embed Fourier data in a 2x larger array
    row_nb = buf1ft.shape[0]
    column_nb = buf1ft.shape[1]
    mlarge = row_nb * 2
    nlarge = column_nb * 2
    crosscorr = np.zeros([mlarge, nlarge], dtype=np.complex128)

    crosscorr[
        int(row_nb - np.fix(row_nb / 2)) : int(row_nb + 1 + np.fix((row_nb - 1) / 2)),
        int(column_nb - np.fix(column_nb / 2)) : int(
            column_nb + 1 + np.fix((column_nb - 1) / 2)
        ),
    ] = (fftshift(buf1ft) * np.conj(fftshift(buf2ft)))[:, :]

    # Compute cross-correlation and locate the peak
    crosscorr = ifftn(ifftshift(crosscorr))  # Calculate cross-correlation
    _, indices = index_max(np.abs(crosscorr))
    row_max = indices[0]
    column_max = indices[1]
    crosscorr_max = crosscorr[row_max, column_max]

    # Obtain shift in original pixel grid from the position of the
    # cross-correlation peak
    row_nb = crosscorr.shape[0]
    column_nb = crosscorr.shape[1]

    md2 = np.fix(row_nb / 2)
    nd2 = np.fix(column_nb / 2)
    if row_max > md2:
        row_shift = row_max - row_nb
    else:
        row_shift = row_max

    if column_max > nd2:
        col_shift = column_max - column_nb
    else:
        col_shift = column_max

    row_shift = row_shift / 2
    col_shift = col_shift / 2

    # If upsampling > 2, then refine estimate with matrix multiply DFT
    if ups_factor > 2:
        # DFT computation
        # Initial shift estimate in upsampled grid
        row_shift = 1.0 * np.round(row_shift * ups_factor) / ups_factor
        col_shift = 1.0 * np.round(col_shift * ups_factor) / ups_factor
        dftshift = np.fix(
            np.ceil(ups_factor * 1.5) / 2
        )  # Center of output array at dftshift+1
        # Matrix multiply DFT around the current shift estimate
        crosscorr = np.conj(
            dftups(
                buf2ft * np.conj(buf1ft),
                np.ceil(ups_factor * 1.5),
                np.ceil(ups_factor * 1.5),
                ups_factor,
                dftshift - row_shift * ups_factor,
                dftshift - col_shift * ups_factor,
            )
        ) / (md2 * nd2 * ups_factor**2)
        # Locate maximum and map back to original pixel grid
        _, indices = index_max(np.abs(crosscorr))
        row_max = indices[0]
        column_max = indices[1]

        crosscorr_max = crosscorr[row_max, column_max]
        rg00 = dftups(buf1ft * np.conj(buf1ft), 1, 1, ups_factor) / (
            md2 * nd2 * ups_factor**2
        )
        rf00 = dftups(buf2ft * np.conj(buf2ft), 1, 1, ups_factor) / (
            md2 * nd2 * ups_factor**2
        )
        row_max = row_max - dftshift
        column_max = column_max - dftshift
        row_shift = 1.0 * row_shift + 1.0 * row_max / ups_factor
        col_shift = 1.0 * col_shift + 1.0 * column_max / ups_factor

    # If upsampling = 2, no additional pixel shift refinement
    else:
        rg00 = np.sum(buf1ft * np.conj(buf1ft)) / row_nb / column_nb
        rf00 = np.sum(buf2ft * np.conj(buf2ft)) / row_nb / column_nb

    error = 1.0 - crosscorr_max * np.conj(crosscorr_max) / (rg00 * rf00)
    error = np.sqrt(np.abs(error))
    diff_phase = np.arctan2(np.imag(crosscorr_max), np.real(crosscorr_max))
    # If its only one row or column the shift along that dimension has no
    # effect. We set to zero.
    if md2 == 1:
        row_shift = 0

    if nd2 == 1:
        col_shift = 0
    return error, diff_phase, row_shift, col_shift


def dftups(
    array,
    output_row_nb,
    output_column_nb,
    ups_factor=1,
    row_offset=0,
    column_offset=0,
):
    """
    Upsampled DFT by matrix multiplies.

    It can compute an upsampled DFT in just a small region. Receives DC in upper left
    corner, image center must be in (1,1). Manuel Guizar - Dec 13, 2007
    Modified from dftups, by J.R. Fienup 7/31/06

    This code is intended to provide the same result as if the following
    operations were performed:

      - Embed the array "in" in an array that is usfac times larger in each
        dimension. ifftshift to bring the center of the image to (1,1).
      - Take the FFT of the larger array
      - Extract an [output_row_nb, output_column_nb] region of the result.
        Starting with the [row_offset+1 column_offset+1] element.

    It achieves this result by computing the DFT in the output array without the need
    to zero pad. Much faster and memory efficient than the zero-padded FFT approach
    if [nor noc] are much smaller than [nr*usfac nc*usfac]

    :param array: 2D input array
    :param output_row_nb: number of pixels in the output upsampled DFT, in units of
     upsampled pixels (default = size(in))
    :param output_column_nb: number of pixels in the output upsampled DFT, in units of
     upsampled pixels (default = size(in))
    :param ups_factor: upsampling factor (default ups_factor = 1)
    :param row_offset: row offset, allow to shift the output array to a region of
     interest on the DFT (default = 0)
    :param column_offset: column offset, allow to shift the output array to a region
     of interest on the DFT (default = 0)
    """
    input_row_nb = array.shape[0]
    input_column_nb = array.shape[1]

    # Compute kernels and obtain DFT by matrix products
    temp_column = np.zeros([input_column_nb, 1])
    temp_column[:, 0] = (
        (ifftshift(np.arange(input_column_nb))) - np.floor(1.0 * input_column_nb / 2)
    )[:]
    temp_row = np.zeros([1, int(output_column_nb)])
    temp_row[0, :] = (np.arange(output_column_nb) - column_offset)[:]
    kernel_column = np.exp(
        (-1j * 2 * np.pi / (input_column_nb * ups_factor))
        * np.dot(temp_column, temp_row)
    )

    temp_column = np.zeros([int(output_row_nb), 1])
    temp_column[:, 0] = (np.arange(output_row_nb) - row_offset)[:]
    temp_row = np.zeros([1, input_row_nb])
    temp_row[0, :] = (
        ifftshift(np.arange(input_row_nb)) - np.floor(1.0 * input_row_nb / 2)
    )[:]
    kernel_row = np.exp(
        (-1j * 2 * np.pi / (input_row_nb * ups_factor)) * np.dot(temp_column, temp_row)
    )

    return np.dot(np.dot(kernel_row, array), kernel_column)


def getimageregistration(array1, array2, precision=10):
    """
    Calculate the registration (shift) between two arrays.

    :param array1: the reference array
    :param array2: the array to register
    :param precision: subpixel precision of the registration. Images will be
     registered to within 1/precision of a pixel
    :return: the list of shifts that need to be applied to array2 in order to align it
     with array1 (no need to flip signs)
    """
    if array1.shape != array2.shape:
        raise ValueError("Arrays should have the same shape")
    # 3D arrays
    if len(array1.shape) == 3:
        abs_array1 = np.abs(array1)
        abs_array2 = np.abs(array2)
        # compress array (sum) in each dimension, i.e. a bunch of 2D arrays
        ft_array1_0 = fftn(
            fftshift(np.sum(abs_array1, 0))
        )  # need fftshift for wrap around
        ft_array2_0 = fftn(fftshift(np.sum(abs_array2, 0)))
        ft_array1_1 = fftn(fftshift(np.sum(abs_array1, 1)))
        ft_array2_1 = fftn(fftshift(np.sum(abs_array2, 1)))
        ft_array1_2 = fftn(fftshift(np.sum(abs_array1, 2)))
        ft_array2_2 = fftn(fftshift(np.sum(abs_array2, 2)))

        # calculate shift in each dimension, i.e. 2 estimates of shift
        result = dft_registration(ft_array1_2, ft_array2_2, ups_factor=precision)
        (
            shiftx1,
            shifty1,
        ) = result[2:4]
        result = dft_registration(ft_array1_1, ft_array2_1, ups_factor=precision)
        (
            shiftx2,
            shiftz1,
        ) = result[2:4]
        result = dft_registration(ft_array1_0, ft_array2_0, ups_factor=precision)
        (
            shifty2,
            shiftz2,
        ) = result[2:4]

        # average them
        xshift = (shiftx1 + shiftx2) / 2
        yshift = (shifty1 + shifty2) / 2
        zshift = (shiftz1 + shiftz2) / 2
        shift_list = xshift, yshift, zshift

    # 2D arrays
    elif len(array1.shape) == 2:
        ft_array1 = fftn(array1)
        ft_array2 = fftn(array2)
        result = dft_registration(ft_array1, ft_array2, ups_factor=precision)
        shift_list = tuple(result[2:])
    else:
        shift_list = None
    return shift_list


def get_shift(
    reference_array: np.ndarray,
    shifted_array: np.ndarray,
    shift_method: str = "modulus",
    precision: int = 1000,
    support_threshold: Union[None, float] = None,
    verbose: bool = True,
    **kwargs,
) -> Sequence[float]:
    """
    Calculate the shift between two arrays.

    The shift is calculated using dft registration. If a threshold for creating a
    support is not provided, DFT registration is performed on the arrays themselves.
    If a threshold is provided, shifts are calculated using the support created by
    thresholding the modulus of the arrays.

    :param reference_array: numpy ndarray
    :param shifted_array: numpy ndarray of the same shape as reference_array
    :param shift_method: 'raw', 'modulus' or 'support'. Object to use for the
     determination of the shift. If 'raw', it uses the raw, eventually complex array.
     if 'modulus', it uses the modulus of the array. If 'support', it uses a support
     created by threshold the modulus of the array.
    :param precision: precision for the DFT registration in 1/pixel
    :param support_threshold: optional normalized threshold in [0, 1]. If not None, it
     will be used to define a support. The center of mass will be calculated for that
     support instead of the modulus.
    :param verbose: True to print comment
    :param kwargs:
     - 'logger': an optional logger

    :return: list of shifts, of length equal to the number of dimensions of the arrays.
     These are shifts that need to be applied to shifted_array in order to align it
     with reference_array (no need to flip signs)
    """
    logger = kwargs.get("logger", module_logger)
    ##########################
    # check input parameters #
    ##########################
    valid.valid_ndarray(
        arrays=(reference_array, shifted_array),
        fix_shape=True,
        name="arrays",
    )
    if shift_method not in {"raw", "modulus", "support"}:
        raise ValueError("shift_method should be 'raw', 'modulus' or 'support'")
    valid.valid_item(
        precision,
        allowed_types=int,
        min_included=1,
        name="precision",
    )
    valid.valid_item(
        support_threshold,
        allowed_types=Real,
        min_included=0,
        max_included=1,
        allow_none=True,
        name="support_threshold",
    )
    valid.valid_item(verbose, allowed_types=bool, name="verbose")

    ##########################################################################
    # define the objects that will be used for the calculation of the shift  #
    ##########################################################################
    if shift_method == "raw":
        reference_obj = reference_array
        shifted_obj = shifted_array
    elif shift_method == "modulus":
        reference_obj = abs(reference_array)
        shifted_obj = abs(shifted_array)
    else:  # "support"
        if support_threshold is None:
            raise ValueError("support_threshold should be a float in [0, 1]")
        reference_obj, shifted_obj = util.make_support(
            arrays=(reference_array, shifted_array), support_threshold=support_threshold
        )

    ##############################################
    # calculate the shift between the two arrays #
    ##############################################
    shift = getimageregistration(reference_obj, shifted_obj, precision=precision)

    if verbose:
        logger.info(f"shifts with the reference object: {shift} pixels")
    return tuple(shift)


def index_max(mydata):
    """Look for the data max and location."""
    myamp = np.abs(mydata)
    myamp_max = myamp.max()
    idx = np.unravel_index(myamp.argmax(), mydata.shape)
    return myamp_max, idx


def index_max1(mydata):
    """Look for the data maximum locations."""
    return np.where(mydata == mydata.max())


def interp_rgi_translation(array: np.ndarray, shift: Sequence[float]) -> np.ndarray:
    """
    Interpolate the shifted array on new positions using a RegularGridInterpolator.

    :param array: a numpy array expressed in a regular grid
    :param shift: a tuple of floats, corresponding to the shift in each dimension that
     need to be applied to array
    :return: the shifted array
    """
    # check some parameters
    valid.valid_ndarray(array, name="array")
    valid.valid_container(
        shift, container_types=(tuple, list), item_types=float, name="shift"
    )

    # current points positions in each dimension
    old_positions = [np.arange(-val // 2, val // 2) for val in array.shape]

    # calculate the new positions
    new_positions = calc_new_positions(old_positions, shift)

    # interpolate array #
    rgi = RegularGridInterpolator(
        old_positions,
        array,
        method="linear",
        bounds_error=False,
        fill_value=0,
    )
    shifted_array = rgi(new_positions)

    return np.asarray(shifted_array.reshape(array.shape).astype(array.dtype))


def shift_array(
    array: np.ndarray, shift: Sequence[float], interpolation_method: str = "subpixel"
) -> np.ndarray:
    """
    Shift array using the defined method given the offsets.

    :param array: a numpy ndarray
    :param shift: tuple of floats, shifts of the array in each dimension
    :param interpolation_method: 'subpixel' for interpolating using subpixel shift,
     'rgi' for interpolating using a RegularGridInterpolator, 'roll' to shift voxels
     by an integral number (the shifts are rounded to the nearest integer)
     created by thresholding the modulus of the array.
    :return: the shifted array
    """
    #########################
    # check some parameters #
    #########################
    valid.valid_ndarray(array, ndim=(2, 3), name="array")
    valid.valid_container(
        shift, container_types=(tuple, list), item_types=Real, name="shifts"
    )
    if interpolation_method not in {"subpixel", "rgi", "roll"}:
        raise ValueError("shift_method should be 'subpixel', 'rgi' or 'roll'")

    ###################
    # shift the array #
    ###################
    if interpolation_method == "subpixel":
        # align obj using subpixel shift, keep the complex output
        shifted_array = subpixel_shift(array, *shift)

        ###################################################
        # convert shifted_array to the original data type #
        # subpixel_shift outputs a complex number         #
        ###################################################
        if not isinstance(array.flatten()[0], Complex):
            shifted_array = abs(shifted_array)

    elif interpolation_method == "rgi":
        # re-sample data on a new grid based on COM shift of support
        shifted_array = interp_rgi_translation(array=array, shift=shift)
    else:  # "roll"
        shifted_array = np.roll(
            array,
            shift=list(map(lambda x: int(np.rint(x)), shift)),
            axis=tuple(range(len(shift))),
        )
    return np.asarray(shifted_array)


def subpixel_shift(array, z_shift, y_shift, x_shift=None):
    """
    Shift array by the shift values.

    Adapted from the Matlab code of Jesse Clark.

    :param array: array to be shifted
    :param z_shift: shift in the first dimension
    :param y_shift: shift in the second dimension
    :param x_shift: shift in the third dimension
    :return: the shifted array
    """
    # check some parameters
    valid.valid_ndarray(array, ndim=(2, 3), name="array")
    valid.valid_item(z_shift, allowed_types=float, name="z_shift")
    valid.valid_item(y_shift, allowed_types=float, name="y_shift")
    valid.valid_item(x_shift, allowed_types=float, allow_none=True, name="x_shift")

    # shift the array
    ndim = len(array.shape)
    if ndim == 3:
        numz, numy, numx = array.shape
        buf2ft = fftn(array)
        temp_z = ifftshift(
            np.arange(-np.fix(numz / 2), np.ceil(numz / 2))
        )  # python does not include the end point
        temp_y = ifftshift(
            np.arange(-np.fix(numy / 2), np.ceil(numy / 2))
        )  # python does not include the end point
        temp_x = ifftshift(
            np.arange(-np.fix(numx / 2), np.ceil(numx / 2))
        )  # python does not include the end point
        myz, myy, myx = np.meshgrid(temp_z, temp_y, temp_x, indexing="ij")
        greg = buf2ft * np.exp(
            -1j
            * 2
            * np.pi
            * (z_shift * myz / numz + y_shift * myy / numy + x_shift * myx / numx)
        )
        shifted_array = ifftn(greg)
    else:  # 2D case
        buf2ft = fftn(array)
        numz, numy = array.shape
        temp_z = ifftshift(
            np.arange(-np.fix(numz / 2), np.ceil(numz / 2))
        )  # python does not include the end point
        temp_y = ifftshift(
            np.arange(-np.fix(numy / 2), np.ceil(numy / 2))
        )  # python does not include the end point
        myz, myy = np.meshgrid(temp_z, temp_y, indexing="ij")
        greg = buf2ft * np.exp(
            -1j * 2 * np.pi * (z_shift * myz / numz + y_shift * myy / numy)
        )
        shifted_array = ifftn(greg)
    return shifted_array
