# Functions for aligning arrays
# Original code from Xianhui Xiao APS Sector 2
# Updated by Ross Harder
# Updated by Steven Leake 30/07/2014
# Changed variable names to make it clearer and put it in CXI convention (z y x)
# J.Carnis 27/04/2018
import numpy as np
from numpy.fft import fftn, fftshift, ifftn, ifftshift
import gc


def getimageregistration(myarray1, myarray2, precision=10):
    assert (
        myarray1.shape == myarray2.shape
    ), "Arrays are different shape in registration"
    # 3D arrays
    if len(myarray1.shape) == 3:
        abs_array1 = np.abs(myarray1)
        abs_array2 = np.abs(myarray2)
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
    elif len(myarray1.shape) == 2:
        ft_array1 = fftn(myarray1)
        ft_array2 = fftn(myarray2)
        result = dft_registration(ft_array1, ft_array2, ups_factor=precision)
        shift_list = tuple(result[2:])
    else:
        shift_list = None
    return shift_list


def index_max(mydata):
    myamp = np.abs(mydata)
    myamp_max = myamp.max()
    idx = np.unravel_index(myamp.argmax(), mydata.shape)
    return myamp_max, idx


def index_max1(mydata):
    return np.where(mydata == mydata.max())


def dft_registration(buf1ft, buf2ft, ups_factor=100):
    """
    # function [output Greg] = dft_registration(buf1ft,buf2ft,usfac);
    # Efficient subpixel image registration by cross-correlation. This code
    # gives the same precision as the FFT upsampled cross correlation in a
    # small fraction of the computation time and with reduced memory
    # requirements. It obtains an initial estimate of the cross-correlation peak
    # by an FFT and then refines the shift estimation by upsampling the DFT
    # only in a small neighborhood of that estimate by means of a
    # matrix-multiply DFT. With this procedure all the image points are used to
    # compute the upsampled cross-correlation.
    # Manuel Guizar - Dec 13, 2007

    # Portions of this code were taken from code written by Ann M. Kowalczyk
    # and James R. Fienup.
    # J.R. Fienup and A.M. Kowalczyk, "Phase retrieval for a complex-valued
    # object by using a low-resolution image," J. Opt. Soc. Am. A 7, 450-458
    # (1990).

    # Citation for this algorithm:
    # Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
    # "Efficient subpixel image registration algorithms," Opt. Lett. 33,
    # 156-158 (2008).

    # Inputs
    # buf1ft    Fourier transform of reference image,
    #           DC in (1,1)   [DO NOT FFTSHIFT]
    # buf2ft    Fourier transform of image to register,
    #           DC in (1,1) [DO NOT FFTSHIFT]
    # ups_factor Upsampling factor (integer). Images will be registered to
    #           within 1/ups_factor of a pixel. For example ups_factor = 20 means
    #           the images will be registered within 1/20 of a pixel. (default = 1)

    # Outputs
    # output =  [error,diff_phase,net_row_shift,net_col_shift]
    # error     Translation invariant normalized RMS error between f and g
    # diff_phase     Global phase difference between the two images (should be
    #               zero if images are non-negative).
    # row_shift col_shift   Pixel shifts between images
    # Greg      (Optional) Fourier transform of registered version of buf2ft,
    #           the global phase difference is compensated for.
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
    myarray,
    output_row_nb,
    output_column_nb,
    ups_factor=1,
    row_offset=0,
    column_offset=0,
):
    """
    # function out=dftups(input,nor,noc,usfac,row_offset,column_offset);
    # Upsampled DFT by matrix multiplies, can compute an upsampled DFT in just
    # a small region.
    # ups_factor         Upsampling factor (default ups_factor = 1)
    # [output_row_nb,output_column_nb]     Number of pixels in the output upsampled
    #               DFT, in units of upsampled pixels (default = size(in))
    # row_offset, column_offset    Row and column offsets, allow to shift the output
    #               array to a region of interest on the DFT (default = 0)
    # Receives DC in upper left corner, image center must be in (1,1)
    # Manuel Guizar - Dec 13, 2007
    # Modified from dftups, by J.R. Fienup 7/31/06

    # This code is intended to provide the same result as if the following
    # operations were performed
    #   - Embed the array "in" in an array that is usfac times larger in each
    #     dimension. ifftshift to bring the center of the image to (1,1).
    #   - Take the FFT of the larger array
    #   - Extract an [output_row_nb, output_column_nb] region of the result.
    #     Starting with the [row_offset+1 column_offset+1] element.

    # It achieves this result by computing the DFT in the output array without
    # the need to zero pad. Much faster and memory efficient than the
    # zero-padded FFT approach if [nor noc] are much smaller than
    # [nr*usfac nc*usfac]
    """
    input_row_nb = myarray.shape[0]
    input_column_nb = myarray.shape[1]

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

    return np.dot(np.dot(kernel_row, myarray), kernel_column)


def subpixel_shift(myarray, z_shift, y_shift, x_shift=0):
    """
    subpixel_shift: shift my array by the shift values
    from Matlab code of Jesse Clark
    :param myarray: array to be shifted
    :param z_shift: shift in the first dimension
    :param y_shift: shift in the second dimension
    :param x_shift: shift in the third dimension
    :return: the shifted array
    """
    if len(myarray.shape) == 3:
        numz, numy, numx = myarray.shape
        buf2ft = fftn(myarray)
        del myarray
        gc.collect()
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
        del buf2ft, myz, myy, myx
        gc.collect()
        shifted_array = ifftn(greg)
    else:
        buf2ft = fftn(myarray)
        numz, numy = myarray.shape
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


# uncomment below to test the code

# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     from scipy.ndimage.interpolation import shift
#
#     img1 = np.zeros((64, 64, 64), dtype="Float64")
#     img1[32-10:32+10, 32-10:32+10, 32-10:32+10] = 1
#     img1 = img1[:, :, 32]
#     img2 = img1.copy()
#     img2 = shift(img2, (-5, 3))
#     shiftz, shifty = getimageregistration(img1, img2, precision=1000)
#     print(shiftz, shifty)
#     shifted_img2 = subpixel_shift(img2, shiftz, shifty)
#
#     fig = plt.figure()
#     plt.subplot(2, 3, 1)
#     plt.imshow(abs(img1))
#     plt.title('original image amp')
#     plt.subplot(2, 3, 2)
#     plt.imshow(abs(img2))
#     plt.title('shifted image amp')
#     plt.subplot(2, 3, 3)
#     plt.imshow(abs(shifted_img2))
#     plt.title('registered image amp')
#     plt.subplot(2, 3, 4)
#     plt.imshow(np.angle(img1))
#     plt.title('original image phase')
#     plt.subplot(2, 3, 5)
#     plt.imshow(np.angle(img2))
#     plt.title('shifted image phase')
#     plt.subplot(2, 3, 6)
#     plt.imshow(np.angle(shifted_img2))
#     plt.title('registered image phase')
#     plt.show()
