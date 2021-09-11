# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#   (c) 06/2021-present : DESY CFEL
#       authors:
#         Jerome Carnis, jerome.carnis@esrf.fr
"""Functions related to data loading and preprocessing, before phase retrieval."""

try:
    import hdf5plugin  # for P10, should be imported before h5py or PyTables
except ModuleNotFoundError:
    pass

import matplotlib.pyplot as plt
from numbers import Real, Integral
import numpy as np
from scipy.ndimage.measurements import center_of_mass
from scipy.interpolate import RegularGridInterpolator
import tkinter as tk
from tkinter import filedialog
import xrayutilities as xu
from ..graph import graph_utils as gu
from ..utils import image_registration as reg
from ..utils import utilities as util
from ..utils import validation as valid


def align_diffpattern(
    reference_data,
    data,
    mask=None,
    method="registration",
    combining_method="rgi",
    return_shift=False,
):
    """
    Align two diffraction patterns.

    The alignement can be based either on the shift of the center of mass or on dft
    registration.

    :param reference_data: the first 3D or 2D diffraction intensity array which will
     serve as a reference.
    :param data: the 3D or 2D diffraction intensity array to align.
    :param mask: the 3D or 2D mask corresponding to data
    :param method: 'center_of_mass' or 'registration'. For 'registration',
     see: Opt. Lett. 33, 156-158 (2008).
    :param combining_method: 'rgi' for RegularGridInterpolator or 'subpixel' for
     subpixel shift
    :param return_shift: if True, will return the shifts as a tuple
    :return:
     - the shifted data
     - the shifted mask
     - if return_shift, returns a tuple containing the shifts
    """
    #########################
    # check some parameters #
    #########################
    valid.valid_ndarray(arrays=(reference_data, data), ndim=(2, 3))
    if method not in {"center_of_mass", "registration"}:
        raise ValueError(f'Incorrect setting {method} for the parameter "method"')
    if combining_method not in {"rgi", "subpixel"}:
        raise ValueError(
            f'Incorrect setting {combining_method} for the parameter "combining_method"'
        )

    ######################
    # align the datasets #
    ######################

    ###########
    # 3D case #
    ###########
    if data.ndim == 3:
        nbz, nby, nbx = reference_data.shape
        if method == "registration":
            shiftz, shifty, shiftx = reg.getimageregistration(
                abs(reference_data), abs(data), precision=100
            )
        else:  # 'center_of_mass'
            ref_piz, ref_piy, ref_pix = center_of_mass(abs(reference_data))
            piz, piy, pix = center_of_mass(abs(data))
            shiftz = ref_piz - piz
            shifty = ref_piy - piy
            shiftx = ref_pix - pix
        print(f"Shifts (z, y, x) = ({shiftz:.2f}, {shifty:.2f}, {shiftx:.2f})")
        if all(val == 0 for val in (shiftz, shifty, shiftx)):
            if not return_shift:
                return data, mask
            return data, mask, (shiftz, shifty, shiftx)

        if combining_method == "rgi":
            # re-sample data on a new grid based on the shift
            old_z = np.arange(-nbz // 2, nbz // 2)
            old_y = np.arange(-nby // 2, nby // 2)
            old_x = np.arange(-nbx // 2, nbx // 2)
            myz, myy, myx = np.meshgrid(old_z, old_y, old_x, indexing="ij")
            new_z = myz - shiftz
            new_y = myy - shifty
            new_x = myx - shiftx
            del myx, myy, myz
            rgi = RegularGridInterpolator(
                (old_z, old_y, old_x),
                data,
                method="linear",
                bounds_error=False,
                fill_value=0,
            )
            data = rgi(
                np.concatenate(
                    (
                        new_z.reshape((1, new_z.size)),
                        new_y.reshape((1, new_z.size)),
                        new_x.reshape((1, new_z.size)),
                    )
                ).transpose()
            )
            data = data.reshape((nbz, nby, nbx)).astype(reference_data.dtype)
            if mask is not None:
                rgi = RegularGridInterpolator(
                    (old_z, old_y, old_x),
                    mask,
                    method="linear",
                    bounds_error=False,
                    fill_value=1,
                )  # fill_value=1: mask voxels where data is not defined
                mask = rgi(
                    np.concatenate(
                        (
                            new_z.reshape((1, new_z.size)),
                            new_y.reshape((1, new_z.size)),
                            new_x.reshape((1, new_z.size)),
                        )
                    ).transpose()
                )
                mask = mask.reshape((nbz, nby, nbx)).astype(data.dtype)

        else:  # 'subpixel'
            data = abs(
                reg.subpixel_shift(data, shiftz, shifty, shiftx)
            )  # data is a real number (intensity)
            if mask is not None:
                mask = abs(reg.subpixel_shift(mask, shiftz, shifty, shiftx))

        shift = shiftz, shifty, shiftx

    ###########
    # 2D case #
    ###########
    else:  # ndim = 2
        nby, nbx = reference_data.shape
        if method == "registration":
            shifty, shiftx = reg.getimageregistration(
                abs(reference_data), abs(data), precision=100
            )
        else:  # 'center_of_mass'
            ref_piy, ref_pix = center_of_mass(abs(reference_data))
            piy, pix = center_of_mass(abs(data))
            shifty = ref_piy - piy
            shiftx = ref_pix - pix
        print(f"Shifts (y, x) = ({shifty:.2f}, {shiftx:.2f})")
        if all(val == 0 for val in (shifty, shiftx)):
            if not return_shift:
                return data, mask
            return data, mask, (shifty, shiftx)

        if combining_method == "rgi":
            # re-sample data on a new grid based on the shift
            old_y = np.arange(-nby // 2, nby // 2)
            old_x = np.arange(-nbx // 2, nbx // 2)
            myy, myx = np.meshgrid(old_y, old_x, indexing="ij")
            new_y = myy - shifty
            new_x = myx - shiftx
            del myx, myy
            rgi = RegularGridInterpolator(
                (old_y, old_x), data, method="linear", bounds_error=False, fill_value=0
            )
            data = rgi(
                np.concatenate(
                    (new_y.reshape((1, new_y.size)), new_x.reshape((1, new_y.size)))
                ).transpose()
            )
            data = data.reshape((nby, nbx)).astype(reference_data.dtype)
            if mask is not None:
                rgi = RegularGridInterpolator(
                    (old_y, old_x),
                    mask,
                    method="linear",
                    bounds_error=False,
                    fill_value=1,
                )
                # fill_value=1: mask voxels where data is not defined
                mask = rgi(
                    np.concatenate(
                        (new_y.reshape((1, new_y.size)), new_x.reshape((1, new_y.size)))
                    ).transpose()
                )
                mask = mask.reshape((nby, nbx)).astype(data.dtype)
        else:  # 'subpixel'
            data = abs(
                reg.subpixel_shift(data, shifty, shiftx)
            )  # data is a real number (intensity)
            if mask is not None:
                mask = abs(reg.subpixel_shift(mask, shifty, shiftx))

        shift = shifty, shiftx

    ####################################
    # filter the data and mask for nan #
    ####################################
    data, mask = util.remove_nan(data=data, mask=mask)

    ###########################
    # return aligned datasets #
    ###########################
    if not return_shift:
        return data, mask
    return data, mask, shift


def center_fft(
    data,
    mask,
    detector,
    frames_logical,
    centering="max",
    fft_option="crop_asymmetric_ZYX",
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
    :param centering: centering option, 'max' or 'com'. It will be overridden if the
     kwarg 'fix_bragg' is provided.
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
     - 'skip': keep the full dataset or crop it to the size defined by fix_size

    :param kwargs:
     - 'fix_bragg' = user-defined position in pixels of the Bragg peak
       [z_bragg, y_bragg, x_bragg]
     - 'fix_size' = user defined output array size
       [zstart, zstop, ystart, ystop, xstart, xstop]
     - 'pad_size' = user defined output array size [nbz, nby, nbx]
     - 'q_values' = [qx, qz, qy], each component being a 1D array

    :return:
     - updated data, mask (and q_values if provided, [] otherwise)
     - pad_width = [z0, z1, y0, y1, x0, x1] number of pixels added at each end of the
       original data
     - updated frames_logical

    """
    valid.valid_ndarray(arrays=(data, mask), ndim=3)
    # check and load kwargs
    valid.valid_kwargs(
        kwargs=kwargs,
        allowed_kwargs={"fix_bragg", "fix_size", "pad_size", "q_values"},
        name="preprocessing_utils.center_fft",
    )
    fix_bragg = kwargs.get("fix_bragg", [])
    fix_size = kwargs.get("fix_size", [])
    pad_size = kwargs.get("pad_size", [])
    q_values = kwargs.get("q_values", [])

    if q_values:  # len(q_values) != 0
        qx = q_values[0]  # axis=0, z downstream, qx in reciprocal space
        qz = q_values[1]  # axis=1, y vertical, qz in reciprocal space
        qy = q_values[2]  # axis=2, x outboard, qy in reciprocal space
    else:
        qx = []
        qy = []
        qz = []

    if centering == "max":
        z0, y0, x0 = np.unravel_index(abs(data).argmax(), data.shape)
        if q_values:
            print(f"Max at (qx, qz, qy): {qx[z0]:.5f}, {qz[y0]:.5f}, {qy[x0]:.5f}")
        else:
            print("Max at pixel (Z, Y, X): ", z0, y0, x0)
    elif centering == "com":
        z0, y0, x0 = center_of_mass(data)
        if q_values:
            print(
                "Center of mass at (qx, qz, qy): "
                f"{qx[z0]:.5f}, {qz[y0]:.5f}, {qy[x0]:.5f}"
            )
        else:
            print("Center of mass at pixel (Z, Y, X): ", z0, y0, x0)
    else:
        raise ValueError("Incorrect value for 'centering' parameter")

    if len(fix_bragg) != 0:
        if len(fix_bragg) != 3:
            raise ValueError("fix_bragg should be a list of 3 integers")
        z0, y0, x0 = fix_bragg
        print(
            "Peak intensity position defined by user on the full detector: ", z0, y0, x0
        )
        y0 = (y0 - detector.roi[0]) / detector.binning[1]
        x0 = (x0 - detector.roi[2]) / detector.binning[2]
        print(
            "Peak intensity position with detector ROI and binning in detector plane: ",
            z0,
            y0,
            x0,
        )

    iz0, iy0, ix0 = int(round(z0)), int(round(y0)), int(round(x0))
    print(f"Data peak value = {data[iz0, iy0, ix0]:.1f}")

    # Max symmetrical box around center of mass
    nbz, nby, nbx = np.shape(data)
    max_nz = abs(2 * min(iz0, nbz - iz0))
    max_ny = 2 * min(iy0, nby - iy0)
    max_nx = abs(2 * min(ix0, nbx - ix0))
    if fft_option != "skip":
        print("Max symmetrical box (qx, qz, qy): ", max_nz, max_ny, max_nx)
    if any(val == 0 for val in (max_nz, max_ny, max_nx)):
        print(
            "Empty images or presence of hotpixel at the border,"
            ' defaulting fft_option to "skip"!'
        )
        fft_option = "skip"

    # Crop/pad data to fulfill FFT size and user requirements
    if fft_option == "crop_sym_ZYX":
        # crop rocking angle and detector, Bragg peak centered
        nz1, ny1, nx1 = util.smaller_primes(
            (max_nz, max_ny, max_nx), maxprime=7, required_dividers=(2,)
        )
        pad_width = np.zeros(6, dtype=int)

        data = data[
            iz0 - nz1 // 2 : iz0 + nz1 // 2,
            iy0 - ny1 // 2 : iy0 + ny1 // 2,
            ix0 - nx1 // 2 : ix0 + nx1 // 2,
        ]
        mask = mask[
            iz0 - nz1 // 2 : iz0 + nz1 // 2,
            iy0 - ny1 // 2 : iy0 + ny1 // 2,
            ix0 - nx1 // 2 : ix0 + nx1 // 2,
        ]
        print("FFT box (qx, qz, qy): ", data.shape)

        if (iz0 - nz1 // 2) > 0:  # if 0, the first frame is used
            frames_logical[0 : iz0 - nz1 // 2] = 0
        if (iz0 + nz1 // 2) < nbz:  # if nbz, the last frame is used
            frames_logical[iz0 + nz1 // 2 :] = 0

        if len(q_values) != 0:
            qx = qx[iz0 - nz1 // 2 : iz0 + nz1 // 2]
            qy = qy[ix0 - nx1 // 2 : ix0 + nx1 // 2]
            qz = qz[iy0 - ny1 // 2 : iy0 + ny1 // 2]

    elif fft_option == "crop_asym_ZYX":
        # crop rocking angle and detector without centering the Bragg peak
        nz1, ny1, nx1 = util.smaller_primes(
            (nbz, nby, nbx), maxprime=7, required_dividers=(2,)
        )
        pad_width = np.zeros(6, dtype=int)

        data = data[
            nbz // 2 - nz1 // 2 : nbz // 2 + nz1 // 2,
            nby // 2 - ny1 // 2 : nby // 2 + ny1 // 2,
            nbx // 2 - nx1 // 2 : nbx // 2 + nx1 // 2,
        ]
        mask = mask[
            nbz // 2 - nz1 // 2 : nbz // 2 + nz1 // 2,
            nby // 2 - ny1 // 2 : nby // 2 + ny1 // 2,
            nbx // 2 - nx1 // 2 : nbx // 2 + nx1 // 2,
        ]
        print("FFT box (qx, qz, qy): ", data.shape)

        if (nbz // 2 - nz1 // 2) > 0:  # if 0, the first frame is used
            frames_logical[0 : nbz // 2 - nz1 // 2] = 0
        if (nbz // 2 + nz1 // 2) < nbz:  # if nbz, the last frame is used
            frames_logical[nbz // 2 + nz1 // 2 :] = 0

        if len(q_values) != 0:
            qx = qx[nbz // 2 - nz1 // 2 : nbz // 2 + nz1 // 2]
            qy = qy[nbx // 2 - nx1 // 2 : nbx // 2 + nx1 // 2]
            qz = qz[nby // 2 - ny1 // 2 : nby // 2 + ny1 // 2]

    elif fft_option == "pad_sym_Z_crop_sym_YX":
        # pad rocking angle based on 'pad_size' (Bragg peak centered)
        # and crop detector (Bragg peak centered)
        if len(pad_size) != 3:
            raise ValueError("pad_size should be a list of three elements")
        if pad_size[0] != util.higher_primes(
            pad_size[0], maxprime=7, required_dividers=(2,)
        ):
            raise ValueError(pad_size[0], "does not meet FFT requirements")
        ny1, nx1 = util.smaller_primes(
            (max_ny, max_nx), maxprime=7, required_dividers=(2,)
        )

        data = data[:, iy0 - ny1 // 2 : iy0 + ny1 // 2, ix0 - nx1 // 2 : ix0 + nx1 // 2]
        mask = mask[:, iy0 - ny1 // 2 : iy0 + ny1 // 2, ix0 - nx1 // 2 : ix0 + nx1 // 2]
        pad_width = np.array(
            [
                int(min(pad_size[0] / 2 - iz0, pad_size[0] - nbz)),
                int(min(pad_size[0] / 2 - nbz + iz0, pad_size[0] - nbz)),
                0,
                0,
                0,
                0,
            ],
            dtype=int,
        )
        data = zero_pad(data, padding_width=pad_width, mask_flag=False)
        mask = zero_pad(
            mask, padding_width=pad_width, mask_flag=True
        )  # mask padded pixels
        print("FFT box (qx, qz, qy): ", data.shape)

        temp_frames = -1 * np.ones(data.shape[0])
        temp_frames[pad_width[0] : pad_width[0] + nbz] = frames_logical
        frames_logical = temp_frames

        if len(q_values) != 0:
            dqx = qx[1] - qx[0]
            qx0 = qx[0] - pad_width[0] * dqx
            qx = qx0 + np.arange(pad_size[0]) * dqx
            qy = qy[ix0 - nx1 // 2 : ix0 + nx1 // 2]
            qz = qz[iy0 - ny1 // 2 : iy0 + ny1 // 2]

    elif fft_option == "pad_sym_Z_crop_asym_YX":
        # pad rocking angle based on 'pad_size' (Bragg peak centered)
        # and crop detector (Bragg peak non-centered)
        if len(pad_size) != 3:
            raise ValueError("pad_size should be a list of three elements")
        print("pad_size for 1st axis before binning: ", pad_size[0])
        if pad_size[0] != util.higher_primes(
            pad_size[0], maxprime=7, required_dividers=(2,)
        ):
            raise ValueError(pad_size[0], "does not meet FFT requirements")
        ny1, nx1 = util.smaller_primes(
            (max_ny, max_nx), maxprime=7, required_dividers=(2,)
        )

        data = data[
            :,
            nby // 2 - ny1 // 2 : nby // 2 + ny1 // 2,
            nbx // 2 - nx1 // 2 : nbx // 2 + nx1 // 2,
        ]
        mask = mask[
            :,
            nby // 2 - ny1 // 2 : nby // 2 + ny1 // 2,
            nbx // 2 - nx1 // 2 : nbx // 2 + nx1 // 2,
        ]
        pad_width = np.array(
            [
                int(min(pad_size[0] / 2 - iz0, pad_size[0] - nbz)),
                int(min(pad_size[0] / 2 - nbz + iz0, pad_size[0] - nbz)),
                0,
                0,
                0,
                0,
            ],
            dtype=int,
        )
        data = zero_pad(data, padding_width=pad_width, mask_flag=False)
        mask = zero_pad(
            mask, padding_width=pad_width, mask_flag=True
        )  # mask padded pixels
        print("FFT box (qx, qz, qy): ", data.shape)

        temp_frames = -1 * np.ones(data.shape[0])
        temp_frames[pad_width[0] : pad_width[0] + nbz] = frames_logical
        frames_logical = temp_frames

        if len(q_values) != 0:
            dqx = qx[1] - qx[0]
            qx0 = qx[0] - pad_width[0] * dqx
            qx = qx0 + np.arange(pad_size[0]) * dqx
            qy = qy[nbx // 2 - nx1 // 2 : nbx // 2 + nx1 // 2]
            qz = qz[nby // 2 - ny1 // 2 : nby // 2 + ny1 // 2]

    elif fft_option == "pad_asym_Z_crop_sym_YX":
        # pad rocking angle without centering the Bragg peak
        # and crop detector (Bragg peak centered)
        ny1, nx1 = util.smaller_primes(
            (max_ny, max_nx), maxprime=7, required_dividers=(2,)
        )
        nz1 = util.higher_primes(nbz, maxprime=7, required_dividers=(2,))

        data = data[:, iy0 - ny1 // 2 : iy0 + ny1 // 2, ix0 - nx1 // 2 : ix0 + nx1 // 2]
        mask = mask[:, iy0 - ny1 // 2 : iy0 + ny1 // 2, ix0 - nx1 // 2 : ix0 + nx1 // 2]
        pad_width = np.array(
            [
                int((nz1 - nbz + ((nz1 - nbz) % 2)) / 2),
                int((nz1 - nbz + 1) / 2 - ((nz1 - nbz) % 2)),
                0,
                0,
                0,
                0,
            ],
            dtype=int,
        )
        data = zero_pad(data, padding_width=pad_width, mask_flag=False)
        mask = zero_pad(
            mask, padding_width=pad_width, mask_flag=True
        )  # mask padded pixels
        print("FFT box (qx, qz, qy): ", data.shape)

        temp_frames = -1 * np.ones(data.shape[0])
        temp_frames[pad_width[0] : pad_width[0] + nbz] = frames_logical
        frames_logical = temp_frames

        if len(q_values) != 0:
            dqx = qx[1] - qx[0]
            qx0 = qx[0] - pad_width[0] * dqx
            qx = qx0 + np.arange(nz1) * dqx
            qy = qy[ix0 - nx1 // 2 : ix0 + nx1 // 2]
            qz = qz[iy0 - ny1 // 2 : iy0 + ny1 // 2]

    elif fft_option == "pad_asym_Z_crop_asym_YX":
        # pad rocking angle and crop detector without centering the Bragg peak
        ny1, nx1 = util.smaller_primes((nby, nbx), maxprime=7, required_dividers=(2,))
        nz1 = util.higher_primes(nbz, maxprime=7, required_dividers=(2,))

        data = data[
            :,
            nby // 2 - ny1 // 2 : nby // 2 + ny1 // 2,
            nbx // 2 - nx1 // 2 : nbx // 2 + nx1 // 2,
        ]
        mask = mask[
            :,
            nby // 2 - ny1 // 2 : nby // 2 + ny1 // 2,
            nbx // 2 - nx1 // 2 : nbx // 2 + nx1 // 2,
        ]
        pad_width = np.array(
            [
                int((nz1 - nbz + ((nz1 - nbz) % 2)) / 2),
                int((nz1 - nbz + 1) / 2 - ((nz1 - nbz) % 2)),
                0,
                0,
                0,
                0,
            ],
            dtype=int,
        )
        data = zero_pad(data, padding_width=pad_width, mask_flag=False)
        mask = zero_pad(
            mask, padding_width=pad_width, mask_flag=True
        )  # mask padded pixels
        print("FFT box (qx, qz, qy): ", data.shape)

        temp_frames = -1 * np.ones(data.shape[0])
        temp_frames[pad_width[0] : pad_width[0] + nbz] = frames_logical
        frames_logical = temp_frames

        if len(q_values) != 0:
            dqx = qx[1] - qx[0]
            qx0 = qx[0] - pad_width[0] * dqx
            qx = qx0 + np.arange(nz1) * dqx
            qy = qy[nbx // 2 - nx1 // 2 : nbx // 2 + nx1 // 2]
            qz = qz[nby // 2 - ny1 // 2 : nby // 2 + ny1 // 2]

    elif fft_option == "pad_sym_Z":
        # pad rocking angle based on 'pad_size'(Bragg peak centered)
        # and keep detector size
        if len(pad_size) != 3:
            raise ValueError("pad_size should be a list of three elements")
        print("pad_size for 1st axis before binning: ", pad_size[0])
        if pad_size[0] != util.higher_primes(
            pad_size[0], maxprime=7, required_dividers=(2,)
        ):
            raise ValueError(pad_size[0], "does not meet FFT requirements")

        pad_width = np.array(
            [
                int(min(pad_size[0] / 2 - iz0, pad_size[0] - nbz)),
                int(min(pad_size[0] / 2 - nbz + iz0, pad_size[0] - nbz)),
                0,
                0,
                0,
                0,
            ],
            dtype=int,
        )
        data = zero_pad(data, padding_width=pad_width, mask_flag=False)
        mask = zero_pad(
            mask, padding_width=pad_width, mask_flag=True
        )  # mask padded pixels
        print("FFT box (qx, qz, qy): ", data.shape)

        temp_frames = -1 * np.ones(data.shape[0])
        temp_frames[pad_width[0] : pad_width[0] + nbz] = frames_logical
        frames_logical = temp_frames

        if len(q_values) != 0:
            dqx = qx[1] - qx[0]
            qx0 = qx[0] - pad_width[0] * dqx
            qx = qx0 + np.arange(pad_size[0]) * dqx

    elif fft_option == "pad_asym_Z":
        # pad rocking angle without centering the Bragg peak, keep detector size
        nz1 = util.higher_primes(nbz, maxprime=7, required_dividers=(2,))

        pad_width = np.array(
            [
                int((nz1 - nbz + ((nz1 - nbz) % 2)) / 2),
                int((nz1 - nbz + 1) / 2 - ((nz1 - nbz) % 2)),
                0,
                0,
                0,
                0,
            ],
            dtype=int,
        )
        data = zero_pad(data, padding_width=pad_width, mask_flag=False)
        mask = zero_pad(
            mask, padding_width=pad_width, mask_flag=True
        )  # mask padded pixels
        print("FFT box (qx, qz, qy): ", data.shape)

        temp_frames = -1 * np.ones(data.shape[0])
        temp_frames[pad_width[0] : pad_width[0] + nbz] = frames_logical
        frames_logical = temp_frames

        if len(q_values) != 0:
            dqx = qx[1] - qx[0]
            qx0 = qx[0] - pad_width[0] * dqx
            qx = qx0 + np.arange(nz1) * dqx

    elif fft_option == "pad_sym_ZYX":
        # pad both dimensions based on 'pad_size' (Bragg peak centered)
        if len(pad_size) != 3:
            raise ValueError("pad_size should be a list of 3 integers")
        print("pad_size: ", pad_size)
        print(
            "The 1st axis (stacking dimension) is padded before binning,"
            " detector plane after binning."
        )
        if pad_size[0] != util.higher_primes(
            pad_size[0], maxprime=7, required_dividers=(2,)
        ):
            raise ValueError(pad_size[0], "does not meet FFT requirements")
        if pad_size[1] != util.higher_primes(
            pad_size[1], maxprime=7, required_dividers=(2,)
        ):
            raise ValueError(pad_size[1], "does not meet FFT requirements")
        if pad_size[2] != util.higher_primes(
            pad_size[2], maxprime=7, required_dividers=(2,)
        ):
            raise ValueError(pad_size[2], "does not meet FFT requirements")

        pad_width = [
            int(min(pad_size[0] / 2 - iz0, pad_size[0] - nbz)),
            int(min(pad_size[0] / 2 - nbz + iz0, pad_size[0] - nbz)),
            int(min(pad_size[1] / 2 - iy0, pad_size[1] - nby)),
            int(min(pad_size[1] / 2 - nby + iy0, pad_size[1] - nby)),
            int(min(pad_size[2] / 2 - ix0, pad_size[2] - nbx)),
            int(min(pad_size[2] / 2 - nbx + ix0, pad_size[2] - nbx)),
        ]
        pad_width = np.array(
            list((map(lambda value: max(value, 0), pad_width))), dtype=int
        )  # remove negative numbers
        data = zero_pad(data, padding_width=pad_width, mask_flag=False)
        mask = zero_pad(
            mask, padding_width=pad_width, mask_flag=True
        )  # mask padded pixels
        print("FFT box (qx, qz, qy): ", data.shape)

        temp_frames = -1 * np.ones(data.shape[0])
        temp_frames[pad_width[0] : pad_width[0] + nbz] = frames_logical
        frames_logical = temp_frames

        if len(q_values) != 0:
            dqx = qx[1] - qx[0]
            dqy = qy[1] - qy[0]
            dqz = qz[1] - qz[0]
            qx0 = qx[0] - pad_width[0] * dqx
            qy0 = qy[0] - pad_width[2] * dqy
            qz0 = qz[0] - pad_width[1] * dqz
            qx = qx0 + np.arange(pad_size[0]) * dqx
            qy = qy0 + np.arange(pad_size[2]) * dqy
            qz = qz0 + np.arange(pad_size[1]) * dqz

    elif fft_option == "pad_asym_ZYX":
        # pad both dimensions without centering the Bragg peak
        nz1, ny1, nx1 = [
            util.higher_primes(nbz, maxprime=7, required_dividers=(2,)),
            util.higher_primes(nby, maxprime=7, required_dividers=(2,)),
            util.higher_primes(nbx, maxprime=7, required_dividers=(2,)),
        ]

        pad_width = np.array(
            [
                int((nz1 - nbz + ((nz1 - nbz) % 2)) / 2),
                int((nz1 - nbz + 1) / 2 - ((nz1 - nbz) % 2)),
                int((ny1 - nby + ((pad_size[1] - nby) % 2)) / 2),
                int((ny1 - nby + 1) / 2 - ((ny1 - nby) % 2)),
                int((nx1 - nbx + ((nx1 - nbx) % 2)) / 2),
                int((nx1 - nbx + 1) / 2 - ((nx1 - nbx) % 2)),
            ]
        )
        data = zero_pad(data, padding_width=pad_width, mask_flag=False)
        mask = zero_pad(
            mask, padding_width=pad_width, mask_flag=True
        )  # mask padded pixels

        temp_frames = -1 * np.ones(data.shape[0])
        temp_frames[pad_width[0] : pad_width[0] + nbz] = frames_logical
        frames_logical = temp_frames

        if len(q_values) != 0:
            dqx = qx[1] - qx[0]
            dqy = qy[1] - qy[0]
            dqz = qz[1] - qz[0]
            qx0 = qx[0] - pad_width[0] * dqx
            qy0 = qy[0] - pad_width[2] * dqy
            qz0 = qz[0] - pad_width[1] * dqz
            qx = qx0 + np.arange(nz1) * dqx
            qy = qy0 + np.arange(nx1) * dqy
            qz = qz0 + np.arange(ny1) * dqz

    elif fft_option == "skip":
        # keep the full dataset or use 'fix_size' parameter
        pad_width = np.zeros(
            6, dtype=int
        )  # do nothing or crop the data, starting_frame should be 0
        if len(fix_size) == 6:
            # take binning into account
            fix_size[2] = int(fix_size[2] // detector.binning[1])
            fix_size[3] = int(fix_size[3] // detector.binning[1])
            fix_size[4] = int(fix_size[4] // detector.binning[2])
            fix_size[5] = int(fix_size[5] // detector.binning[2])
            # size of output array defined
            nbz, nby, nbx = np.shape(data)
            z_span = fix_size[1] - fix_size[0]
            y_span = fix_size[3] - fix_size[2]
            x_span = fix_size[5] - fix_size[4]
            if (
                z_span > nbz
                or y_span > nby
                or x_span > nbx
                or fix_size[1] > nbz
                or fix_size[3] > nby
                or fix_size[5] > nbx
            ):
                raise ValueError("Predefined fix_size uncorrect")
            data = data[
                fix_size[0] : fix_size[1],
                fix_size[2] : fix_size[3],
                fix_size[4] : fix_size[5],
            ]
            mask = mask[
                fix_size[0] : fix_size[1],
                fix_size[2] : fix_size[3],
                fix_size[4] : fix_size[5],
            ]

            if fix_size[0] > 0:  # if 0, the first frame is used
                frames_logical[0 : fix_size[0]] = 0
            if fix_size[1] < nbz:  # if nbz, the last frame is used
                frames_logical[fix_size[1] :] = 0

            if len(q_values) != 0:
                qx = qx[fix_size[0] : fix_size[1]]
                qy = qy[fix_size[4] : fix_size[5]]
                qz = qz[fix_size[2] : fix_size[3]]
    else:
        raise ValueError("Incorrect value for 'fft_option'")

    if len(q_values) != 0:
        q_values = list(q_values)
        q_values[0] = qx
        q_values[1] = qz
        q_values[2] = qy
    return data, mask, pad_width, q_values, frames_logical


def check_empty_frames(data, mask=None, monitor=None, frames_logical=None):
    """
    Check if there is intensity for all frames.

    In case of beam dump, some frames may be empty. The data and optional mask will be
    cropped to remove those empty frames.

    :param data: a numpy 3D array
    :param mask: a numpy 3D array of 0 (pixel not masked) and 1 (masked pixel),
     same shape as data
    :param monitor: a numpy 1D array of shape equal to data.shape[0]
    :param frames_logical: 1D array of length equal to the number of measured frames.
     In case of cropping the length of the stack of frames changes. A frame whose
     index is set to 1 means that it is used, 0 means not used.
    :return:
     - cropped data as a numpy 3D array
     - cropped mask as a numpy 3D array
     - cropped monitor as a numpy 1D array
     - updated frames_logical

    """
    valid.valid_ndarray(arrays=data, ndim=3)
    if mask is not None:
        valid.valid_ndarray(arrays=mask, shape=data.shape)
    if monitor is not None:
        if not isinstance(monitor, np.ndarray):
            raise TypeError("monitor should be a numpy array")
        if monitor.ndim != 1 or len(monitor) != data.shape[0]:
            raise ValueError("monitor be a 1D array of length data.shae[0]")

    if frames_logical is None:
        frames_logical = np.ones(data.shape[0])
    valid.valid_1d_array(
        frames_logical,
        allowed_types=Integral,
        allow_none=False,
        allowed_values=(0, 1),
        name="frames_logical",
    )

    # check if there are empty frames
    is_intensity = np.zeros(data.shape[0])
    is_intensity[np.argwhere(data.sum(axis=(1, 2)))] = 1
    if is_intensity.sum() != data.shape[0]:
        print("\nEmpty frame detected, cropping the data\n")

    # update frames_logical
    frames_logical = np.multiply(frames_logical, is_intensity)

    # remove empty frames from the data and update the mask and the monitor
    data = data[np.nonzero(frames_logical)]
    mask = mask[np.nonzero(frames_logical)]
    monitor = monitor[np.nonzero(frames_logical)]
    return data, mask, monitor, frames_logical


def check_pixels(data, mask, debugging=False):
    """
    Check for hot pixels in the data using the mean value and the variance.

    :param data: 3D diffraction data
    :param mask: 2D or 3D mask. Mask will summed along the first axis if a 3D array.
    :param debugging: set to True to see plots
    :type debugging: bool
    :return: the filtered 3D data and the updated 2D mask.
    """
    valid.valid_ndarray(arrays=data, ndim=3)
    valid.valid_ndarray(arrays=mask, ndim=(2, 3))
    nbz, nby, nbx = data.shape

    if mask.ndim == 3:  # 3D array
        print("Mask is a 3D array, summing it along axis 0")
        mask = mask.sum(axis=0)
        mask[np.nonzero(mask)] = 1
    valid.valid_ndarray(arrays=mask, shape=(nby, nbx))

    print(
        "\ncheck_pixels(): number of masked pixels due to detector gaps ="
        f" {int(mask.sum())} on a total of {nbx*nby}"
    )
    meandata = data.mean(axis=0)  # 2D
    vardata = 1 / data.var(axis=0)  # 2D
    var_mean = vardata[vardata != np.inf].mean()
    vardata[meandata == 0] = var_mean
    # pixels were data=0 (i.e. 1/variance=inf) are set to the mean of  1/var:
    # we do not want to mask pixels where there was no intensity during the scan

    if debugging:
        gu.combined_plots(
            tuple_array=(mask, meandata, vardata),
            tuple_sum_frames=False,
            tuple_sum_axis=0,
            tuple_width_v=None,
            tuple_width_h=None,
            tuple_colorbar=True,
            tuple_vmin=0,
            tuple_vmax=(1, 1, np.nan),
            tuple_scale=("linear", "linear", "linear"),
            tuple_title=(
                "Input mask",
                "check_pixels()\nmean(data) before masking",
                "check_pixels()\n1/var(data) before masking",
            ),
            reciprocal_space=True,
            position=(131, 132, 133),
        )

    # calculate the mean and variance of a single photon event along the rocking curve
    min_count = 0.99  # pixels with only 1 photon count along the rocking curve,
    # use the value 0.99 to be inclusive
    mean_singlephoton = min_count / nbz
    var_singlephoton = (
        ((nbz - 1) * mean_singlephoton ** 2 + (min_count - mean_singlephoton) ** 2)
        * 1
        / nbz
    )
    print(
        "check_pixels(): var_mean={:.2f}, 1/var_threshold={:.2f}".format(
            var_mean, 1 / var_singlephoton
        )
    )

    # mask hotpixels with zero variance
    temp_mask = np.zeros((nby, nbx))
    temp_mask[vardata == np.inf] = 1
    # this includes only hotpixels since zero intensity pixels were set to var_mean
    mask[np.nonzero(temp_mask)] = 1  # update the mask with zero variance hotpixels
    vardata[vardata == np.inf] = 0  # update the array
    print(
        "check_pixels(): number of zero variance hotpixels = {:d}".format(
            int(temp_mask.sum())
        )
    )

    # filter out pixels which have a variance smaller that the threshold
    # (note that  vardata = 1/data.var())
    indices_badpixels = np.nonzero(vardata > 1 / var_singlephoton)
    mask[indices_badpixels] = 1  # mask is 2D
    print(
        "check_pixels(): number of pixels with too low variance = {:d}\n".format(
            indices_badpixels[0].shape[0]
        )
    )

    # update the data array
    indices_badpixels = np.nonzero(mask)  # update indices
    for index in range(nbz):
        tempdata = data[index, :, :]
        tempdata[
            indices_badpixels
        ] = 0  # numpy array is mutable hence data will be modified

    if debugging:
        meandata = data.mean(axis=0)
        vardata = 1 / data.var(axis=0)
        vardata[meandata == 0] = var_mean  # 0 intensity pixels, not masked
        gu.combined_plots(
            tuple_array=(mask, meandata, vardata),
            tuple_sum_frames=False,
            tuple_sum_axis=0,
            tuple_width_v=None,
            tuple_width_h=None,
            tuple_colorbar=True,
            tuple_vmin=0,
            tuple_vmax=(1, 1, np.nan),
            tuple_scale="linear",
            tuple_title=(
                "Output mask",
                "check_pixels()\nmean(data) after masking",
                "check_pixels()\n1/var(data) after masking",
            ),
            reciprocal_space=True,
            position=(131, 132, 133),
        )
    return data, mask


def find_bragg(data, peak_method):
    """
    Find the Bragg peak position in data based on the centering method.

    :param data: 2D or 3D array. If complex, Bragg peak position is calculated for
     abs(array)
    :param peak_method: 'max', 'com' or 'maxcom'. For 'maxcom', it uses method 'max'
     for the first axis and 'com' for the other axes.
    :return: the centered data
    """
    valid.valid_ndarray(arrays=data, ndim=(2, 3))
    if all((peak_method != val for val in {"max", "com", "maxcom"})):
        raise ValueError('Incorrect value for "centering_method" parameter')

    if data.ndim == 2:
        z0 = 0
        if peak_method == "max":
            y0, x0 = np.unravel_index(abs(data).argmax(), data.shape)
            print(f"Max at (y, x): ({y0}, {x0})  Max = {int(data[y0, x0])}")
        else:  # 'com'
            y0, x0 = center_of_mass(data)
            print(
                f"Center of mass at (y, x): ({y0:.1f}, {x0:.1f})  "
                f"COM = {int(data[int(y0), int(x0)])}"
            )
    else:  # 3D
        if peak_method == "max":
            z0, y0, x0 = np.unravel_index(abs(data).argmax(), data.shape)
            print(
                f"Max at (z, y, x): ({z0}, {y0}, {x0})  Max = {int(data[z0, y0, x0])}"
            )
        elif peak_method == "com":
            z0, y0, x0 = center_of_mass(data)
            print(
                f"Center of mass at (z, y, x): ({z0:.1f}, {y0:.1f}, {x0:.1f})  "
                f"COM = {int(data[int(z0), int(y0), int(x0)])}"
            )
        else:  # 'maxcom'
            z0, _, _ = np.unravel_index(abs(data).argmax(), data.shape)
            y0, x0 = center_of_mass(data[z0, :, :])
            print(
                f"MaxCom at (z, y, x): ({z0:.1f}, {y0:.1f}, {x0:.1f})  "
                f"COM = {int(data[int(z0), int(y0), int(x0)])}"
            )

    return z0, y0, x0


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
     - 'follow_bragg': bool, True when for energy scans the detector was also scanned
       to follow the Bragg peak
     - 'fill_value': tuple of two real numbers, fill values to use for pixels outside
       of the interpolation range. The first value is for the data, the second for the
       mask. Default is (0, 0)

    :return: the data and mask interpolated in the laboratory frame, q values
     (downstream, vertical up, outboard). q values are in inverse angstroms.
    """
    valid.valid_ndarray(arrays=(data, mask), ndim=3)
    # check and load kwargs
    valid.valid_kwargs(
        kwargs=kwargs,
        allowed_kwargs={"follow_bragg", "fill_value", "reference_axis"},
        name="kwargs",
    )
    follow_bragg = kwargs.get("follow_bragg", False)
    valid.valid_item(follow_bragg, allowed_types=bool, name="follow_bragg")
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
    print(
        "Gridding the data using the linearized matrix,"
        " the result will be in the laboratory frame"
    )
    string = "linmat_reciprocal_space_"
    (interp_data, interp_mask), q_values = setup.ortho_reciprocal(
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
        )

    return interp_data, interp_mask, q_values


def grid_bcdi_xrayutil(
    data,
    mask,
    scan_number,
    logfile,
    detector,
    setup,
    frames_logical,
    hxrd,
    debugging=False,
    **kwargs,
):
    """
    Interpolate BCDI reciprocal space data using xrayutilities package.

    The resulting (qx, qy, qz) are in the crystal frame (qz vertical).

    :param data: the 3D data, already binned in the detector frame
    :param mask: the corresponding 3D mask
    :param scan_number: the scan number to load
    :param logfile: file containing the information about the scan and image numbers
     (specfile, .fio...)
    :param detector: an instance of the class Detector
    :param setup: instance of the Class experiment_utils.Setup()
    :param frames_logical: array of initial length the number of measured frames.
     In case of padding the length changes. A frame whose index is set to 1 means
     that it is used, 0 means not used, -1 means padded (added) frame.
    :param hxrd: an initialized xrayutilities HXRD object used for the orthogonalization
     of the dataset
    :param debugging: set to True to see plots
    :param kwargs:
     - follow_bragg (bool): True when for energy scans the detector was also scanned to
       follow the Bragg peak

    :return: the data and mask interpolated in the crystal frame, q values
     (downstream, vertical up, outboard). q values are in inverse angstroms.
    """
    valid.valid_ndarray(arrays=(data, mask), ndim=3)
    # check and load kwargs
    valid.valid_kwargs(
        kwargs=kwargs,
        allowed_kwargs={"follow_bragg"},
        name="preprocessing_utils.grid_bcdi_xrayutil",
    )
    follow_bragg = kwargs.get("follow_bragg", False)
    valid.valid_item(
        follow_bragg, allowed_types=bool, name="preprocessing_utils.grid_bcdi_xrayutil"
    )

    numz, numy, numx = data.shape
    print(
        "Gridding the data using xrayutilities package,"
        " the result will be in the crystal frame"
    )
    string = "xrutil_reciprocal_space_"
    if setup.filtered_data:
        print(
            "Trying to orthogonalize a filtered data,"
            " the corresponding detector ROI should be provided\n"
            "otherwise q values will be wrong."
        )
    qx, qz, qy, frames_logical = setup.calc_qvalues_xrutils(
        logfile=logfile,
        hxrd=hxrd,
        nb_frames=numz,
        scan_number=scan_number,
        frames_logical=frames_logical,
        follow_bragg=follow_bragg,
    )

    # below is specific to ID01 energy scans
    # where frames are duplicated for undulator gap change
    if setup.beamline == "ID01" and setup.rocking_angle == "energy":
        # frames need to be removed
        tempdata = np.zeros(((frames_logical != 0).sum(), numy, numx))
        offset_frame = 0
        for idx in range(numz):
            if frames_logical[idx] != 0:  # use frame
                tempdata[idx - offset_frame, :, :] = data[idx, :, :]
            else:  # average with the precedent frame
                offset_frame = offset_frame + 1
                tempdata[idx - offset_frame, :, :] = (
                    tempdata[idx - offset_frame, :, :] + data[idx, :, :]
                ) / 2
        data = tempdata
        mask = mask[
            0 : data.shape[0], :, :
        ]  # truncate the mask to have the correct size

    maxbins = []
    for dim in (qx, qy, qz):
        maxstep = max((abs(np.diff(dim, axis=j)).max() for j in range(3)))
        maxbins.append(int(abs(dim.max() - dim.min()) / maxstep))
    print(f"Maximum number of bins based on the sampling in q: {maxbins}")
    maxbins = util.smaller_primes(maxbins, maxprime=7, required_dividers=(2,))
    print(f"Maximum number of bins based on the shape requirements for FFT: {maxbins}")
    # only rectangular cuboidal voxels are supported in xrayutilities FuzzyGridder3D
    gridder = xu.FuzzyGridder3D(*maxbins)
    #
    # define the width of data points (rectangular datapoints, xrayutilities use half
    # of these values but there are artefacts sometimes)
    wx = (qx.max() - qx.min()) / maxbins[0]
    wz = (qz.max() - qz.min()) / maxbins[1]
    wy = (qy.max() - qy.min()) / maxbins[2]
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
        detector.preprocessing_binning[0] * detector.binning[0],
        detector.preprocessing_binning[1] * detector.binning[1],
        detector.preprocessing_binning[2] * detector.binning[2],
    )

    numz, numy, numx = interp_data.shape
    plot_comment = (f"_{numz}_{numy}_{numx}"
                    f"_{final_binning[0]}_{final_binning[1]}_{final_binning[2]}.png")

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
        )

    return interp_data, interp_mask, (qx, qz, qy), frames_logical


def load_bcdi_data(
    logfile,
    scan_number,
    detector,
    setup,
    flatfield=None,
    hotpixels=None,
    background=None,
    normalize="skip",
    debugging=False,
    **kwargs,
):
    """
    Load Bragg CDI data, apply optional threshold, normalization and binning.

    :param logfile: file containing the information about the scan and image numbers
     (specfile, .fio...)
    :param scan_number: the scan number to load
    :param detector: an instance of the class Detector
    :param setup: an instance of the class Setup
    :param flatfield: the 2D flatfield array
    :param hotpixels: the 2D hotpixels array. 1 for a hotpixel, 0 for normal pixels.
    :param background: the 2D background array to subtract to the data
    :param normalize: 'monitor' to return the default monitor values, 'sum_roi' to
     return a monitor based on the integrated intensity in the region of interest
     defined by detector.sum_roi, 'skip' to do nothing
    :param debugging:  set to True to see plots
    :parama kwargs:
     - 'photon_threshold' = float, photon threshold to apply before binning

    :return:
     - the 3D data and mask arrays
     - frames_logical: array of initial length the number of measured frames.
       In case of padding the length changes. A frame whose index is set to 1 means
       that it is used, 0 means not used, -1 means padded (added) frame.
     - the monitor values used for the intensity normalization

    """
    # check and load kwargs
    valid.valid_kwargs(
        kwargs=kwargs,
        allowed_kwargs={"photon_threshold"},
        name="preprocessing_utils.load_bcdi_data",
    )
    photon_threshold = kwargs.get("photon_threshold", 0)
    valid.valid_item(
        photon_threshold,
        allowed_types=Real,
        min_included=0,
        name="preprocessing_utils.load_bcdi_data",
    )

    rawdata, rawmask, monitor, frames_logical = load_data(
        logfile=logfile,
        scan_number=scan_number,
        detector=detector,
        setup=setup,
        flatfield=flatfield,
        hotpixels=hotpixels,
        background=background,
        normalize=normalize,
        debugging=debugging,
    )

    print(
        (rawdata < 0).sum(), " negative data points masked"
    )  # can happen when subtracting a background
    rawmask[rawdata < 0] = 1
    rawdata[rawdata < 0] = 0

    _, nby, nbx = rawdata.shape
    # pad the data to the shape defined by the ROI
    if (
        detector.roi[1] - detector.roi[0] > nby
        or detector.roi[3] - detector.roi[2] > nbx
    ):
        if detector.roi[0] < 0:  # padding on the left
            starty = abs(detector.roi[0])  # loaded data will start at this index
        else:  # padding on the right
            starty = 0
        if detector.roi[2] < 0:  # padding on the left
            startx = abs(detector.roi[2])  # loaded data will start at this index
        else:  # padding on the right
            startx = 0
        start = (0, starty, startx)
        print("Paddind the data to the shape defined by the ROI")
        rawdata = util.crop_pad(
            array=rawdata,
            pad_start=start,
            output_shape=(
                rawdata.shape[0],
                detector.roi[1] - detector.roi[0],
                detector.roi[3] - detector.roi[2],
            ),
        )
        rawmask = util.crop_pad(
            array=rawmask,
            pad_value=1,
            pad_start=start,
            output_shape=(
                rawmask.shape[0],
                detector.roi[1] - detector.roi[0],
                detector.roi[3] - detector.roi[2],
            ),
        )

    # apply optional photon threshold before binning
    if photon_threshold != 0:
        rawmask[rawdata < photon_threshold] = 1
        rawdata[rawdata < photon_threshold] = 0
        print("Applying photon threshold before binning: < ", photon_threshold)

    # bin data and mask in the detector plane if needed
    # binning in the stacking dimension is done at the very end of the data processing
    if (detector.binning[1] != 1) or (detector.binning[2] != 1):
        print(
            "Binning the data: detector vertical axis by",
            detector.binning[1],
            ", detector horizontal axis by",
            detector.binning[2],
        )
        rawdata = util.bin_data(
            rawdata, (1, detector.binning[1], detector.binning[2]), debugging=False
        )
        rawmask = util.bin_data(
            rawmask, (1, detector.binning[1], detector.binning[2]), debugging=False
        )
        rawmask[np.nonzero(rawmask)] = 1

    return rawdata, rawmask, frames_logical, monitor


def load_data(
    logfile,
    scan_number,
    detector,
    setup,
    flatfield=None,
    hotpixels=None,
    background=None,
    normalize="skip",
    bin_during_loading=False,
    debugging=False,
):
    """
    Load data, apply filters and concatenate it for phasing.

    :param logfile: the logfile created in Setup.create_logfile()
    :param scan_number: the scan number to load
    :param detector: an instance of the class Detector
    :param setup: an instance of the class Setup
    :param flatfield: the 2D flatfield array
    :param hotpixels: the 2D hotpixels array. 1 for a hotpixel, 0 for normal pixels.
    :param background: the 2D background array to subtract to the data
    :param normalize: 'monitor' to return the default monitor values, 'sum_roi' to
     return a monitor based on the integrated intensity in the region of interest
     defined by detector.sum_roi, 'skip' to do nothing
    :param bin_during_loading: if True, the data will be binned in the detector frame
     while loading. It saves a lot of memory for large detectors.
    :param debugging: set to True to see plots
    :return:

     - the 3D data array in the detector frame and the 3D mask array
     - the monitor values for normalization
     - frames_logical: 1D array of length equal to the number of measured frames.
       In case of cropping the length of the stack of frames changes. A frame whose
       index is set to 1 means that it is used, 0 means not used.

    """
    print(
        "User-defined ROI size (VxH):",
        detector.roi[1] - detector.roi[0],
        detector.roi[3] - detector.roi[2],
    )
    print(
        "Detector physical size without binning (VxH):",
        detector.nb_pixel_y,
        detector.nb_pixel_x,
    )
    print(
        "Detector size with binning (VxH):",
        detector.nb_pixel_y // detector.binning[1],
        detector.nb_pixel_x // detector.binning[2],
    )

    if setup.filtered_data:
        data, mask3d, monitor, frames_logical = load_filtered_data(detector=detector)

    else:
        data, mask2d, monitor, frames_logical = setup.diffractometer.load_data(
            logfile=logfile,
            setup=setup,
            scan_number=scan_number,
            detector=detector,
            flatfield=flatfield,
            hotpixels=hotpixels,
            background=background,
            normalize=normalize,
            bin_during_loading=bin_during_loading,
            debugging=debugging,
        )

        # bin the 2D mask if necessary
        if bin_during_loading:
            mask2d = util.bin_data(
                mask2d, (detector.binning[1], detector.binning[2]), debugging=False
            )
            mask2d[np.nonzero(mask2d)] = 1

        # check for abnormally behaving pixels
        data, mask2d = check_pixels(data=data, mask=mask2d, debugging=debugging)
        mask3d = np.repeat(mask2d[np.newaxis, :, :], data.shape[0], axis=0)
        mask3d[np.isnan(data)] = 1
        data[np.isnan(data)] = 0

        # check for empty frames (no beam)
        data, mask3d, monitor, frames_logical = check_empty_frames(
            data=data, mask=mask3d, monitor=monitor, frames_logical=frames_logical
        )

        # intensity normalization
        if normalize == "skip":
            print("Skip intensity normalization")
        else:
            print("Intensity normalization using " + normalize)
            data, monitor = normalize_dataset(
                array=data,
                monitor=monitor,
                norm_to_min=True,
                savedir=detector.savedir,
                debugging=debugging,
            )

    return data, mask3d, monitor, frames_logical.astype(int)


def load_filtered_data(detector):
    """
    Load a filtered dataset and the corresponding mask.

    :param detector: an instance of the class Detector
    :return: the data and the mask array
    """
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        initialdir=detector.datadir,
        title="Select data file",
        filetypes=[("NPZ", "*.npz")],
    )
    data = np.load(file_path)
    npz_key = data.files
    data = data[npz_key[0]]
    file_path = filedialog.askopenfilename(
        initialdir=detector.datadir,
        title="Select mask file",
        filetypes=[("NPZ", "*.npz")],
    )
    mask = np.load(file_path)
    npz_key = mask.files
    mask = mask[npz_key[0]]

    monitor = np.ones(data.shape[0])
    frames_logical = np.ones(data.shape[0])

    return data, mask, monitor, frames_logical


def mean_filter(
    data,
    nb_neighbours,
    mask=None,
    target_val=0,
    extent=1,
    min_count=3,
    interpolate="mask_isolated",
    debugging=False,
):
    """
    Mask or apply a mean filter to data.

    The procedure is applied only if the empty pixel is surrounded by nb_neighbours or
    more pixels with at least min_count intensity per pixel.

    :param data: 2D or 3D array to be filtered
    :param nb_neighbours: minimum number of non-zero neighboring pixels for median
     filtering
    :param mask: mask array of the same shape as data
    :param target_val: value where to interpolate, allowed values are int>=0 or np.nan
    :param extent: in pixels, extent of the averaging window from the reference pixel
     (extent=1, 2, 3 ... corresponds to window width=3, 5, 7 ... )
    :param min_count: minimum intensity (inclusive) in the neighboring pixels to
     interpolate, the pixel will be masked otherwise.
    :param interpolate: based on 'nb_neighbours', if 'mask_isolated' will mask
     isolated pixels, if 'interp_isolated' will interpolate isolated pixels
    :param debugging: set to True to see plots
    :type debugging: bool
    :return: updated data and mask, number of pixels treated
    """
    valid.valid_ndarray(arrays=data, ndim=(2, 3))
    # check some mparameters
    if mask is None:
        mask = np.zeros(data.shape)
    valid.valid_ndarray(arrays=mask, shape=data.shape)

    if not np.isnan(target_val) and not isinstance(target_val, int):
        raise ValueError(
            "target_val should be nan or an integer, cannot assess float equality"
        )

    valid.valid_item(
        nb_neighbours, allowed_types=int, min_excluded=0, name="mean_filter"
    )
    valid.valid_item(extent, allowed_types=int, min_excluded=0, name="mean_filter")
    valid.valid_item(min_count, allowed_types=int, min_included=0, name="mean_filter")

    if interpolate not in {"mask_isolated", "interp_isolated"}:
        raise ValueError(
            f"invalid value '{interpolate}' for interpolate,"
            f" allowed are 'mask_isolated' and 'interp_isolated'"
        )
    if not isinstance(debugging, bool):
        raise TypeError(f"debugging should be a boolean, got {type(debugging)}")

    # find all voxels to be processed
    if target_val is np.nan:
        target_pixels = np.argwhere(np.isnan(data))
    else:
        target_pixels = np.argwhere(data == target_val)
    nb_pixels = 0

    if debugging:
        gu.combined_plots(
            tuple_array=(data, mask),
            tuple_sum_frames=(False, False),
            tuple_sum_axis=(0, 0),
            tuple_width_v=(None, None),
            tuple_width_h=(None, None),
            tuple_colorbar=(True, True),
            tuple_vmin=(-1, 0),
            tuple_vmax=(np.nan, 1),
            tuple_scale=("log", "linear"),
            tuple_title=("Data before filtering", "Mask before filtering"),
            reciprocal_space=True,
        )

    if data.ndim == 2:
        for indx in range(target_pixels.shape[0]):
            pixrow = target_pixels[indx, 0]
            pixcol = target_pixels[indx, 1]
            temp = data[
                pixrow - extent : pixrow + extent + 1,
                pixcol - extent : pixcol + extent + 1,
            ]
            temp = temp[np.logical_and(~np.isnan(temp), temp != target_val)]
            if (
                temp.size >= nb_neighbours and (temp > min_count).all()
            ):  # nb_neighbours is >= 1
                nb_pixels += 1
                if interpolate == "interp_isolated":
                    data[pixrow, pixcol] = temp.mean()
                    mask[pixrow, pixcol] = 0
                else:
                    mask[pixrow, pixcol] = 1
    else:  # 3D
        for indx in range(target_pixels.shape[0]):
            pix_z = target_pixels[indx, 0]
            pix_y = target_pixels[indx, 1]
            pix_x = target_pixels[indx, 2]
            temp = data[
                pix_z - extent : pix_z + extent + 1,
                pix_y - extent : pix_y + extent + 1,
                pix_x - extent : pix_x + extent + 1,
            ]
            temp = temp[np.logical_and(~np.isnan(temp), temp != target_val)]
            if (
                temp.size >= nb_neighbours and (temp > min_count).all()
            ):  # nb_neighbours is >= 1
                nb_pixels += 1
                if interpolate == "interp_isolated":
                    data[pix_z, pix_y, pix_x] = temp.mean()
                    mask[pix_z, pix_y, pix_x] = 0
                else:
                    mask[pix_z, pix_y, pix_x] = 1

    if debugging:
        gu.combined_plots(
            tuple_array=(data, mask),
            tuple_sum_frames=(True, True),
            tuple_sum_axis=(0, 0),
            tuple_width_v=(None, None),
            tuple_width_h=(None, None),
            tuple_colorbar=(True, True),
            tuple_vmin=(-1, 0),
            tuple_vmax=(np.nan, 1),
            tuple_scale=("log", "linear"),
            tuple_title=("Data after filtering", "Mask after filtering"),
            reciprocal_space=True,
        )

    return data, nb_pixels, mask


def normalize_dataset(array, monitor, savedir=None, norm_to_min=True, debugging=False):
    """
    Normalize array using the monitor values.

    :param array: the 3D array to be normalized
    :param monitor: the monitor values
    :param savedir: path where to save the debugging figure
    :param norm_to_min: bool, True to normalize to min(monitor) instead of max(monitor),
     avoid multiplying the noise
    :param debugging: bool, True to see plots
    :return:

     - normalized dataset
     - updated monitor
     - a title for plotting

    """
    valid.valid_ndarray(arrays=array, ndim=3)
    ndim = array.ndim
    nbz, nby, nbx = array.shape
    original_max = None
    original_data = None

    if ndim != 3:
        raise ValueError("Array should be 3D")

    if debugging:
        original_data = np.copy(array)
        original_max = original_data.max()
        original_data[original_data < 5] = 0  # remove the background
        original_data = original_data.sum(
            axis=1
        )  # the first axis is the normalization axis

    print(
        "Monitor min, max, mean: {:.1f}, {:.1f}, {:.1f}".format(
            monitor.min(), monitor.max(), monitor.mean()
        )
    )

    if norm_to_min:
        print("Data normalization by monitor.min()/monitor\n")
        monitor = monitor.min() / monitor  # will divide higher intensities
    else:  # norm to max
        print("Data normalization by monitor.max()/monitor\n")
        monitor = monitor.max() / monitor  # will multiply lower intensities

    nbz = array.shape[0]
    if len(monitor) != nbz:
        raise ValueError(
            "The frame number and the monitor data length are different:",
            f"got {nbz} frames but {len(monitor)} monitor values",
        )

    for idx in range(nbz):
        array[idx, :, :] = array[idx, :, :] * monitor[idx]

    if debugging:
        norm_data = np.copy(array)
        # rescale norm_data to original_data for easier comparison
        norm_data = norm_data * original_max / norm_data.max()
        norm_data[norm_data < 5] = 0  # remove the background
        norm_data = norm_data.sum(axis=1)  # the first axis is the normalization axis
        fig = gu.combined_plots(
            tuple_array=(monitor, original_data, norm_data),
            tuple_sum_frames=False,
            tuple_colorbar=False,
            tuple_vmin=(np.nan, 0, 0),
            tuple_vmax=np.nan,
            tuple_title=(
                "monitor.min() / monitor",
                "Before norm (thres. 5)",
                "After norm (thres. 5)",
            ),
            tuple_scale=("linear", "log", "log"),
            xlabel=("Frame number", "Detector X", "Detector X"),
            is_orthogonal=False,
            ylabel=("Counts (a.u.)", "Frame number", "Frame number"),
            position=(211, 223, 224),
            reciprocal_space=True,
        )
        if savedir is not None:
            fig.savefig(savedir + f"monitor_{nbz}_{nby}_{nbx}.png")
        plt.close(fig)

    return array, monitor


def reload_bcdi_data(
    data,
    mask,
    logfile,
    scan_number,
    detector,
    setup,
    normalize=False,
    debugging=False,
    **kwargs,
):
    """
    Reload BCDI data, apply optional threshold, normalization and binning.

    :param data: the 3D data array
    :param mask: the 3D mask array
    :param logfile: file containing the information about the scan and image numbers
     (specfile, .fio...)
    :param scan_number: the scan number to load
    :param detector: an instance of the class Detector
    :param setup: an instance of the class Setup
    :param normalize: set to True to normalize by the default monitor of the beamline
    :param debugging:  set to True to see plots
    :parama kwargs:
     - 'photon_threshold' = float, photon threshold to apply before binning

    :return:
     - the updated 3D data and mask arrays
     - the monitor values used for the intensity normalization

    """
    valid.valid_ndarray(arrays=(data, mask), ndim=3)
    # check and load kwargs
    valid.valid_kwargs(
        kwargs=kwargs,
        allowed_kwargs={"photon_threshold"},
        name="preprocessing_utils.reload_bcdi_data",
    )
    photon_threshold = kwargs.get("photon_threshold", 0)
    valid.valid_item(
        photon_threshold,
        allowed_types=Real,
        min_included=0,
        name="preprocessing_utils.reload_bcdi_data",
    )

    normalize_method = "monitor" if normalize else "skip"

    nbz, nby, nbx = data.shape
    frames_logical = np.ones(nbz)

    print(
        (data < 0).sum(), " negative data points masked"
    )  # can happen when subtracting a background
    mask[data < 0] = 1
    data[data < 0] = 0

    # normalize by the incident X-ray beam intensity
    if normalize_method == "skip":
        print("Skip intensity normalization")
        monitor = []
    else:  # use the default monitor of the beamline
        monitor = setup.diffractometer.read_monitor(
            scan_number=scan_number,
            logfile=logfile,
            beamline=setup.beamline,
            actuators=setup.actuators,
        )

        print("Intensity normalization using " + normalize_method)
        data, monitor = normalize_dataset(
            array=data,
            monitor=monitor,
            norm_to_min=True,
            savedir=detector.savedir,
            debugging=True,
        )

    # pad the data to the shape defined by the ROI
    if (
        detector.roi[1] - detector.roi[0] > nby
        or detector.roi[3] - detector.roi[2] > nbx
    ):
        start = (np.nan, min(0, detector.roi[0]), min(0, detector.roi[2]))
        print("Paddind the data to the shape defined by the ROI")
        data = util.crop_pad(
            array=data,
            pad_start=start,
            output_shape=(
                data.shape[0],
                detector.roi[1] - detector.roi[0],
                detector.roi[3] - detector.roi[2],
            ),
        )
        mask = util.crop_pad(
            array=mask,
            pad_value=1,
            pad_start=start,
            output_shape=(
                mask.shape[0],
                detector.roi[1] - detector.roi[0],
                detector.roi[3] - detector.roi[2],
            ),
        )

    # apply optional photon threshold before binning
    if photon_threshold != 0:
        mask[data < photon_threshold] = 1
        data[data < photon_threshold] = 0
        print("Applying photon threshold before binning: < ", photon_threshold)

    # bin data and mask in the detector plane if needed
    # binning in the stacking dimension is done at the very end of the data processing
    if (detector.binning[1] != 1) or (detector.binning[2] != 1):
        print(
            "Binning the data: detector vertical axis by",
            detector.binning[1],
            ", detector horizontal axis by",
            detector.binning[2],
        )
        data = util.bin_data(
            data, (1, detector.binning[1], detector.binning[2]), debugging=debugging
        )
        mask = util.bin_data(
            mask, (1, detector.binning[1], detector.binning[2]), debugging=debugging
        )
        mask[np.nonzero(mask)] = 1

    return data, mask, frames_logical, monitor


def wrap(obj, start_angle, range_angle):
    """
    Wrap obj between start_angle and (start_angle + range_angle).

    :param obj: number or array to be wrapped
    :param start_angle: start angle of the range
    :param range_angle: range
    :return: wrapped angle in [start_angle, start_angle+range[
    """
    return (obj - start_angle + range_angle) % range_angle + start_angle


def zero_pad(array, padding_width=np.zeros(6), mask_flag=False, debugging=False):
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
