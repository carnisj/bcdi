# Functions for aligning arrays
# Original code from Xianhui Xiao APS Sector 2
# Updated by Ross Harder
# Updated by Steven Leake 30/07/2014
# Changed variable names to make it clearer and put it in CXI convention (z y x)
# J.Carnis 27/04/2018
"""Functions related to the registration and alignement of two arrays."""

from collections.abc import Sequence
from numbers import Real
import numpy as np
from numpy.fft import fftn, fftshift, ifftn, ifftshift
from scipy.interpolate import RegularGridInterpolator
from typing import Union

from ..utils import utilities as util
from ..utils import validation as valid


def getimageregistration(array1, array2, precision=10):
    """
    Calculate the registration (shift) between two arrays.

    :param array1: the reference array
    :param array2: the array to register
    :param precision: subpixel precision of the registration. Images will be
     registered to within 1/precision of a pixel.
    :return: the list of shifts
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


def index_max(mydata):
    """Look for the data max and location."""
    myamp = np.abs(mydata)
    myamp_max = myamp.max()
    idx = np.unravel_index(myamp.argmax(), mydata.shape)
    return myamp_max, idx


def index_max1(mydata):
    """Look for the data maximum locations."""
    return np.where(mydata == mydata.max())


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
        crosscorr = (
            np.conj(
                dftups(
                    buf2ft * np.conj(buf1ft),
                    np.ceil(ups_factor * 1.5),
                    np.ceil(ups_factor * 1.5),
                    ups_factor,
                    dftshift - row_shift * ups_factor,
                    dftshift - col_shift * ups_factor,
                )
            )
            / (md2 * nd2 * ups_factor ** 2)
        )
        # Locate maximum and map back to original pixel grid
        _, indices = index_max(np.abs(crosscorr))
        row_max = indices[0]
        column_max = indices[1]

        crosscorr_max = crosscorr[row_max, column_max]
        rg00 = dftups(buf1ft * np.conj(buf1ft), 1, 1, ups_factor) / (
            md2 * nd2 * ups_factor ** 2
        )
        rf00 = dftups(buf2ft * np.conj(buf2ft), 1, 1, ups_factor) / (
            md2 * nd2 * ups_factor ** 2
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


def get_shift(
    reference_array: np.ndarray,
    shifted_array: np.ndarray,
    shift_method: str = "modulus",
    precision: int = 1000,
    support_threshold: Union[None, float] = None,
    verbose: bool = True,
) -> Sequence[float]:
    """
    Calculate the shift between two arrays.

    The shift is calculated using dft registration. If a threshold for creating a
    support is not provided, DFT registration is performed on the arrays themselves.
    If a threshold is provided, shifts are calculated using the support created by
    thresholding the modulus of the arrays.

    :param reference_array: numpy ndarray
    :param shifted_array: numpy ndarray of the same shape as reference_array
    :param shift_method: 'raw', 'modulus', 'support' or 'skip'. Object to use for the
     determination of the shift. If 'raw', it uses the raw, eventually complex array.
     if 'modulus', it uses the modulus of the array. If 'support', it uses a support
     created by threshold the modulus of the array.
    :param precision: precision for the DFT registration in 1/pixel
    :param support_threshold: optional normalized threshold in [0, 1]. If not None, it
     will be used to define a support. The center of mass will be calculated for that
     support instead of the modulus.
    :param verbose: True to print comment
    :return: list of shifts, of length equal to the number of dimensions of the arrays
    """
    ##########################
    # check input parameters #
    ##########################
    valid.valid_ndarray(
        arrays=(reference_array, shifted_array),
        fix_shape=True,
        name="get_shift_arrays_com",
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
        reference_obj, shifted_obj = util.make_support(
            arrays=(reference_array, shifted_array), support_threshold=support_threshold
        )

    ##############################################
    # calculate the shift between the two arrays #
    ##############################################
    shifts = getimageregistration(reference_obj, shifted_obj, precision=precision)

    if verbose:
        print(f"shifts with the reference object: {shifts} pixels")
    return shifts


def shift_array(
    array: np.ndarray, shifts: Sequence[float], interpolation_method: str = "subpixel"
) -> np.ndarray:
    """
    Shift array using the defined method given the offsets.

    :param array: a numpy ndarray
    :param shifts: tuple of floats, shifts of the array in each dimension
    :param interpolation_method: 'raw', 'modulus', 'support'. Object to use for the
     determination of the shift. If 'raw', it uses the raw, eventually complex array.
     if 'modulus', it uses the modulus of the array. If 'support', it uses a support
     created by thresholding the modulus of the array.
    :return: the shifted array
    """
    #########################
    # check some parameters #
    #########################
    valid.valid_ndarray(array, name="array")
    valid.valid_container(
        shifts, container_types=(tuple, list), item_types=Real, name="shifts"
    )

    ###################
    # shift the array #
    ###################
    if interpolation_method == "subpixel":
        # align obj using subpixel shift, keep the complex output
        shifted_array = subpixel_shift(array, *shifts)
    elif interpolation_method == "rgi":
        # re-sample data on a new grid based on COM shift of support
        nbz, nby, nbx = array.shape
        old_z = np.arange(-nbz // 2, nbz // 2)
        old_y = np.arange(-nby // 2, nby // 2)
        old_x = np.arange(-nbx // 2, nbx // 2)
        myz, myy, myx = np.meshgrid(old_z, old_y, old_x, indexing="ij")
        new_z = myz + shifts[0]
        new_y = myy + shifts[1]
        new_x = myx + shifts[2]
        del myx, myy, myz
        rgi = RegularGridInterpolator(
            (old_z, old_y, old_x),
            array,
            method="linear",
            bounds_error=False,
            fill_value=0,
        )
        shifted_array = rgi(
            np.concatenate(
                (
                    new_z.reshape((1, new_z.size)),
                    new_y.reshape((1, new_z.size)),
                    new_x.reshape((1, new_z.size)),
                )
            ).transpose()
        )
        shifted_array = shifted_array.reshape((nbz, nby, nbx)).astype(array.dtype)
    else:  # "roll"
        shifted_array = np.roll(array, shifts, axis=(0, 1, 2))

    return shifted_array


def subpixel_shift(array, z_shift, y_shift, x_shift=0):
    """
    Shift array by the shift values.

    Adapted from the Matlab code of Jesse Clark.

    :param array: array to be shifted
    :param z_shift: shift in the first dimension
    :param y_shift: shift in the second dimension
    :param x_shift: shift in the third dimension
    :return: the shifted array
    """
    if len(array.shape) == 3:
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
    else:
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
