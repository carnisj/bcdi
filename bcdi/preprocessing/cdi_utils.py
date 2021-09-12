# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#   (c) 06/2021-present : DESY CFEL
#       authors:
#         Jerome Carnis, jerome.carnis@esrf.fr
"""Functions related to forward CDI data preprocessing, before phase retrieval."""


import matplotlib.pyplot as plt
from numbers import Real
import numpy as np

from ..experiment import diffractometer as diff
from ..graph import graph_utils as gu
from ..utils import utilities as util
from ..utils import validation as valid


def beamstop_correction(data, detector, setup, debugging=False):
    """
    Correct absorption from the beamstops during P10 forward CDI experiment.

    :param data: the 3D stack of 2D CDI images, shape = (nbz, nby, nbx) or 2D image of
     shape (nby, nbx)
    :param detector: the detector object: Class experiment_utils.Detector()
    :param setup: an instance of the class Setup
    :param debugging: set to True to see plots
    :return: the corrected data
    """
    valid.valid_ndarray(arrays=data, ndim=(2, 3))
    energy = setup.energy
    if not isinstance(energy, Real):
        raise TypeError(f"Energy should be a number in eV, not a {type(energy)}")

    print(f"Applying beamstop correction for the X-ray energy of {energy}eV")

    if energy not in [8200, 8700, 10000, 10235]:
        print(
            "no beam stop information for the X-ray energy of {:d}eV,"
            " defaulting to the correction for 8700 eV".format(int(energy))
        )
        energy = 8700

    ndim = data.ndim
    if ndim == 3:
        pass
    elif ndim == 2:
        data = data[np.newaxis, :, :]
    else:
        raise ValueError("2D or 3D data expected")
    nbz, nby, nbx = data.shape

    directbeam_y = setup.direct_beam[0] - detector.roi[0]  # vertical
    directbeam_x = setup.direct_beam[1] - detector.roi[2]  # horizontal

    # at 8200eV, the transmission of 100um Si is 0.26273
    # at 8700eV, the transmission of 100um Si is 0.32478
    # at 10000eV, the transmission of 100um Si is 0.47337
    # at 10235eV, the transmission of 100um Si is 0.51431
    if energy == 8200:
        factor_large = 1 / 0.26273  # 5mm*5mm (100um thick) Si wafer
        factor_small = 1 / 0.26273  # 3mm*3mm (100um thick) Si wafer
        pixels_large = [-33, 35, -31, 36]
        # boundaries of the large wafer relative to the direct beam (V x H)
        pixels_small = [-14, 14, -11, 16]
        # boundaries of the small wafer relative to the direct beam (V x H)
    elif energy == 8700:
        factor_large = 1 / 0.32478  # 5mm*5mm (100um thick) Si wafer
        factor_small = 1 / 0.32478  # 3mm*3mm (100um thick) Si wafer
        pixels_large = [-33, 35, -31, 36]
        # boundaries of the large wafer relative to the direct beam (V x H)
        pixels_small = [-14, 14, -11, 16]
        # boundaries of the small wafer relative to the direct beam (V x H)
    elif energy == 10000:
        factor_large = 2.1 / 0.47337  # 5mm*5mm (200um thick) Si wafer
        factor_small = 4.5 / 0.47337  # 3mm*3mm (300um thick) Si wafer
        pixels_large = [-36, 34, -34, 35]
        # boundaries of the large wafer relative to the direct beam (V x H)
        pixels_small = [-21, 21, -21, 21]
        # boundaries of the small wafer relative to the direct beam (V x H)
    else:  # energy = 10235
        factor_large = 2.1 / 0.51431  # 5mm*5mm (200um thick) Si wafer
        factor_small = 4.5 / 0.51431  # 3mm*3mm (300um thick) Si wafer
        pixels_large = [-34, 35, -33, 36]
        # boundaries of the large wafer relative to the direct beam (V x H)
        pixels_small = [-20, 22, -20, 22]
        # boundaries of the small wafer relative to the direct beam (V x H)

    # define boolean arrays for the large and the small square beam stops
    large_square = np.zeros((nby, nbx))
    large_square[
        directbeam_y + pixels_large[0] : directbeam_y + pixels_large[1],
        directbeam_x + pixels_large[2] : directbeam_x + pixels_large[3],
    ] = 1
    small_square = np.zeros((nby, nbx))
    small_square[
        directbeam_y + pixels_small[0] : directbeam_y + pixels_small[1],
        directbeam_x + pixels_small[2] : directbeam_x + pixels_small[3],
    ] = 1

    # define the boolean array for the border of the large square wafer
    # (the border is 1 pixel wide)
    temp_array = np.zeros((nby, nbx))
    temp_array[
        directbeam_y + pixels_large[0] + 1 : directbeam_y + pixels_large[1] - 1,
        directbeam_x + pixels_large[2] + 1 : directbeam_x + pixels_large[3] - 1,
    ] = 1
    large_border = large_square - temp_array

    # define the boolean array for the border of the small square wafer
    # (the border is 1 pixel wide)
    temp_array = np.zeros((nby, nbx))
    temp_array[
        directbeam_y + pixels_small[0] + 1 : directbeam_y + pixels_small[1] - 1,
        directbeam_x + pixels_small[2] + 1 : directbeam_x + pixels_small[3] - 1,
    ] = 1
    small_border = small_square - temp_array

    if debugging:
        gu.imshow_plot(
            data,
            sum_frames=True,
            sum_axis=0,
            vmin=0,
            vmax=11,
            plot_colorbar=True,
            scale="log",
            title="data before absorption correction",
            is_orthogonal=False,
            reciprocal_space=True,
        )

        gu.combined_plots(
            tuple_array=(large_square, small_square, large_border, small_border),
            tuple_sum_frames=(False, False, False, False),
            tuple_sum_axis=0,
            tuple_width_v=None,
            tuple_width_h=None,
            tuple_colorbar=False,
            tuple_vmin=0,
            tuple_vmax=11,
            is_orthogonal=False,
            reciprocal_space=True,
            tuple_title=(
                "large_square",
                "small_square",
                "larger border",
                "small border",
            ),
            tuple_scale=("linear", "linear", "linear", "linear"),
        )

    # absorption correction for the large and small square beam stops
    for idx in range(nbz):
        tempdata = data[idx, :, :]
        tempdata[np.nonzero(large_square)] = (
            tempdata[np.nonzero(large_square)] * factor_large
        )
        tempdata[np.nonzero(small_square)] = (
            tempdata[np.nonzero(small_square)] * factor_small
        )
        data[idx, :, :] = tempdata

    if debugging:
        width = 40
        _, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        ax0.plot(
            np.log10(
                data[:, directbeam_y, directbeam_x - width : directbeam_x + width].sum(
                    axis=0
                )
            )
        )
        ax0.set_title("horizontal cut after absorption correction")
        ax0.vlines(
            x=[
                width + pixels_large[2],
                width + pixels_large[3],
                width + pixels_small[2],
                width + pixels_small[3],
            ],
            ymin=ax0.get_ylim()[0],
            ymax=ax0.get_ylim()[1],
            colors="b",
            linestyle="dashed",
        )
        ax1.plot(
            np.log10(
                data[:, directbeam_y - width : directbeam_y + width, directbeam_x].sum(
                    axis=0
                )
            )
        )
        ax1.set_title("vertical cut after absorption correction")
        ax1.vlines(
            x=[
                width + pixels_large[0],
                width + pixels_large[1],
                width + pixels_small[0],
                width + pixels_small[1],
            ],
            ymin=ax1.get_ylim()[0],
            ymax=ax1.get_ylim()[1],
            colors="b",
            linestyle="dashed",
        )

        gu.imshow_plot(
            data,
            sum_frames=True,
            sum_axis=0,
            vmin=0,
            vmax=11,
            plot_colorbar=True,
            scale="log",
            title="data after absorption correction",
            is_orthogonal=False,
            reciprocal_space=True,
        )

    # interpolation for the border of the large square wafer
    indices = np.argwhere(large_border == 1)
    data[
        np.nonzero(np.repeat(large_border[np.newaxis, :, :], nbz, axis=0))
    ] = 0  # exclude border points
    for frame in range(nbz):  # loop over 2D images in the detector plane
        tempdata = data[frame, :, :]
        for idx in range(indices.shape[0]):
            pixrow = indices[idx, 0]
            pixcol = indices[idx, 1]
            counter = (
                9 - large_border[pixrow - 1 : pixrow + 2, pixcol - 1 : pixcol + 2].sum()
            )  # number of pixels in a 3x3 window
            # which do not belong to the border
            tempdata[pixrow, pixcol] = (
                tempdata[pixrow - 1 : pixrow + 2, pixcol - 1 : pixcol + 2].sum()
                / counter
            )
        data[frame, :, :] = tempdata

    # interpolation for the border of the small square wafer
    indices = np.argwhere(small_border == 1)
    data[
        np.nonzero(np.repeat(small_border[np.newaxis, :, :], nbz, axis=0))
    ] = 0  # exclude border points
    for frame in range(nbz):  # loop over 2D images in the detector plane
        tempdata = data[frame, :, :]
        for idx in range(indices.shape[0]):
            pixrow = indices[idx, 0]
            pixcol = indices[idx, 1]
            counter = (
                9 - small_border[pixrow - 1 : pixrow + 2, pixcol - 1 : pixcol + 2].sum()
            )  # number of pixels in a 3x3 window
            # which do not belong to the border
            tempdata[pixrow, pixcol] = (
                tempdata[pixrow - 1 : pixrow + 2, pixcol - 1 : pixcol + 2].sum()
                / counter
            )
        data[frame, :, :] = tempdata

    if debugging:
        width = 40
        _, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        ax0.plot(
            np.log10(
                data[:, directbeam_y, directbeam_x - width : directbeam_x + width].sum(
                    axis=0
                )
            )
        )
        ax0.set_title("horizontal cut after interpolating border")
        ax0.vlines(
            x=[
                width + pixels_large[2],
                width + pixels_large[3],
                width + pixels_small[2],
                width + pixels_small[3],
            ],
            ymin=ax0.get_ylim()[0],
            ymax=ax0.get_ylim()[1],
            colors="b",
            linestyle="dashed",
        )
        ax1.plot(
            np.log10(
                data[:, directbeam_y - width : directbeam_y + width, directbeam_x].sum(
                    axis=0
                )
            )
        )
        ax1.set_title("vertical cut after interpolating border")
        ax1.vlines(
            x=[
                width + pixels_large[0],
                width + pixels_large[1],
                width + pixels_small[0],
                width + pixels_small[1],
            ],
            ymin=ax1.get_ylim()[0],
            ymax=ax1.get_ylim()[1],
            colors="b",
            linestyle="dashed",
        )

        gu.imshow_plot(
            data,
            sum_frames=True,
            sum_axis=0,
            vmin=0,
            vmax=11,
            plot_colorbar=True,
            scale="log",
            title="data after interpolating the border of beam stops",
            is_orthogonal=False,
            reciprocal_space=True,
        )
    return data


def check_cdi_angle(data, mask, cdi_angle, frames_logical, debugging=False):
    """
    Check for overlaps of the sample rotation motor position in forward CDI experiment.

    It checks if there is no overlap in the measurement angles, and crops it otherwise.
    Flip the rotation direction to convert sample angles into detector angles. Update
    data, mask and frames_logical accordingly.

    :param data: 3D forward CDI dataset before gridding.
    :param mask: 3D mask
    :param cdi_angle: array of measurement sample angles in degrees
    :param frames_logical: array of initial length the number of measured frames.
     In case of padding the length changes. A frame whose index is set to 1 means
     that it is used, 0 means not used, -1 means padded (added) frame.
    :param debugging: True to have more printed comments
    :return: updated data, mask, detector cdi_angle, frames_logical
    """
    valid.valid_ndarray(arrays=(data, mask), ndim=3)
    detector_angle = np.zeros(len(cdi_angle))
    # flip the rotation axis in order to compensate the rotation of the Ewald sphere
    # due to sample rotation
    print(
        "Reverse the rotation direction to compensate the rotation of the Ewald sphere"
    )
    for idx, item in enumerate(cdi_angle):
        detector_angle[idx] = cdi_angle[0] - (item - cdi_angle[0])

    wrap_angle = util.wrap(
        obj=detector_angle, start_angle=detector_angle.min(), range_angle=180
    )
    for idx, item in enumerate(wrap_angle):
        duplicate = np.isclose(wrap_angle[:idx], item, rtol=1e-06, atol=1e-06).sum()
        # duplicate will be different from 0 if there is a duplicated angle
        frames_logical[idx] = frames_logical[idx] * (
            duplicate == 0
        )  # set frames_logical to 0 if duplicated angle

    if debugging:
        print("frames_logical after checking duplicated angles:\n", frames_logical)

    # find first duplicated angle
    try:
        index_duplicated = np.where(frames_logical == 0)[0][0]
        # change the angle by a negligeable amount
        # to still be able to use it for interpolation
        if cdi_angle[1] - cdi_angle[0] > 0:
            detector_angle[index_duplicated] = detector_angle[index_duplicated] - 0.0001
        else:
            detector_angle[index_duplicated] = detector_angle[index_duplicated] + 0.0001
        print(
            "RegularGridInterpolator cannot take duplicated values: shifting frame",
            index_duplicated,
            "by 1/10000 degrees for the interpolation",
        )

        frames_logical[index_duplicated] = 1
    except IndexError:  # no duplicated angle
        print("no duplicated angle")

    data = data[np.nonzero(frames_logical)[0], :, :]
    mask = mask[np.nonzero(frames_logical)[0], :, :]
    detector_angle = detector_angle[np.nonzero(frames_logical)]
    return data, mask, detector_angle, frames_logical


def grid_cdi(
    data,
    mask,
    logfile,
    detector,
    setup,
    frames_logical,
    correct_curvature=False,
    debugging=False,
    **kwargs,
):
    """
    Interpolate reciprocal space forward CDI data.

    The interpolation is done from the measurement cylindrical frame to the
    laboratory frame (cartesian coordinates). Note that it is based on PetraIII P10
    beamline ( counterclockwise rotation, detector seen from the front).

    :param data: the 3D data, already binned in the detector frame
    :param mask: the corresponding 3D mask
    :param logfile: file containing the information about the scan and image numbers
     (specfile, .fio...)
    :param detector: an instance of the class Detector.
     The detector orientation is supposed to follow the CXI convention: (z
     downstream, y vertical up, x outboard) Y opposite to y, X opposite to x
    :param setup: an instance of the class Setup
    :param frames_logical: array of initial length the number of measured frames.
     In case of padding the length changes. A frame whose index is set to 1 means
     that it is used, 0 means not used, -1 means padded (added) frame.
    :param correct_curvature: if True, will correct for the curvature of
     the Ewald sphere
    :param debugging: set to True to see plots
    :param kwargs:
     - 'fill_value': tuple of two real numbers, fill values to use for pixels outside
       of the interpolation range. The first value is for the data, the second for the
       mask. Default is (0, 0)

    :return: the data and mask interpolated in the laboratory frame, q values
     (downstream, vertical up, outboard)
    """
    fill_value = kwargs.get("fill_value", (0, 0))
    valid.valid_ndarray(arrays=(data, mask), ndim=3)
    if setup.beamline == "P10_SAXS":
        if setup.rocking_angle == "inplane":
            if setup.custom_scan:
                cdi_angle = setup.custom_motors["hprz"]
            else:
                cdi_angle, _ = setup.diffractometer.motor_positions(
                    setup=setup, logfile=logfile
                )  # second return value is the X-ray energy
        else:
            raise ValueError(
                "out-of-plane rotation not yet implemented for forward CDI data"
            )
    else:
        raise NotImplementedError("Not yet implemented for beamlines other than P10")

    data, mask, cdi_angle, frames_logical = check_cdi_angle(
        data=data,
        mask=mask,
        cdi_angle=cdi_angle,
        frames_logical=frames_logical,
        debugging=debugging,
    )
    if debugging:
        print("\ncdi_angle", cdi_angle)
    nbz, nby, nbx = data.shape
    print("\nData shape after check_cdi_angle and before regridding:", nbz, nby, nbx)
    print("\nAngle range:", cdi_angle.min(), cdi_angle.max())

    (interp_data, interp_mask), q_values, corrected_dirbeam = setup.ortho_cdi(
        arrays=(data, mask),
        cdi_angle=cdi_angle,
        fill_value=fill_value,
        correct_curvature=correct_curvature,
        debugging=debugging,
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

    # calculate the position in pixels of the origin of the reciprocal space
    pivot_z = int((setup.direct_beam[1] - detector.roi[2]) / detector.binning[2])
    # 90 degrees conter-clockwise rotation of detector X around qz, downstream
    _, numy, numx = interp_data.shape
    pivot_y = int(numy - corrected_dirbeam[0])
    # detector Y vertical down, opposite to qz vertical up
    pivot_x = int(numx - corrected_dirbeam[1])
    # detector X inboard at P10, opposite to qy outboard
    print(
        "\nOrigin of the reciprocal space (Qx,Qz,Qy): "
        f"({pivot_z}, {pivot_y}, {pivot_x})\n"
    )

    # plot the gridded data
    final_binning = (
        detector.preprocessing_binning[2] * detector.binning[2],
        detector.preprocessing_binning[1] * detector.binning[1],
        detector.preprocessing_binning[2] * detector.binning[2],
    )
    plot_comment = (
        f"_{numx}_{numy}_{numx}"
        f"_{final_binning[0]}_{final_binning[1]}_{final_binning[2]}.png"
    )
    # sample rotation around the vertical direction at P10: the effective
    # binning in axis 0 is binning[2]

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
    fig.text(
        0.55,
        0.30,
        "Origin of the reciprocal space (Qx,Qz,Qy):\n\n"
        + "     ({:d}, {:d}, {:d})".format(pivot_z, pivot_y, pivot_x),
        size=14,
    )
    fig.savefig(detector.savedir + "reciprocal_space_sum" + plot_comment)
    plt.close(fig)

    fig, _, _ = gu.contour_slices(
        interp_data,
        (qx, qz, qy),
        sum_frames=False,
        title="Regridded data",
        levels=np.linspace(
            0, np.ceil(np.log10(interp_data.max(initial=None))), 150, endpoint=True
        ),
        slice_position=(pivot_z, pivot_y, pivot_x),
        plot_colorbar=True,
        scale="log",
        is_orthogonal=True,
        reciprocal_space=True,
    )
    fig.text(
        0.55,
        0.30,
        "Origin of the reciprocal space (Qx,Qz,Qy):\n\n"
        + "     ({:d}, {:d}, {:d})".format(pivot_z, pivot_y, pivot_x),
        size=14,
    )
    fig.savefig(detector.savedir + "reciprocal_space_central" + plot_comment)
    plt.close(fig)

    fig, _, _ = gu.multislices_plot(
        interp_data,
        sum_frames=False,
        scale="log",
        plot_colorbar=True,
        vmin=0,
        slice_position=(pivot_z, pivot_y, pivot_x),
        title="Regridded data",
        is_orthogonal=True,
        reciprocal_space=True,
    )
    fig.text(
        0.55,
        0.30,
        "Origin of the reciprocal space (Qx,Qz,Qy):\n\n"
        + "     ({:d}, {:d}, {:d})".format(pivot_z, pivot_y, pivot_x),
        size=14,
    )
    fig.savefig(detector.savedir + "reciprocal_space_central_pix" + plot_comment)
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

    return interp_data, interp_mask, [qx, qz, qy], frames_logical


def load_cdi_data(
    logfile,
    scan_number,
    detector,
    setup,
    bin_during_loading=False,
    flatfield=None,
    hotpixels=None,
    background=None,
    normalize="skip",
    debugging=False,
    **kwargs,
):
    """
    Load forward CDI data and preprocess it.

    It applies beam stop correction and an optional photon threshold, normalization
    and binning.

    :param logfile: file containing the information about the scan and image numbers
     (specfile, .fio...)
    :param scan_number: the scan number to load
    :param detector: an instance of the class Detector
    :param setup: an instance of the class Setup
    :param bin_during_loading: True to bin the data during loading (faster)
    :param flatfield: the 2D flatfield array
    :param hotpixels: the 2D hotpixels array. 1 for a hotpixel, 0 for normal pixels.
    :param background: the 2D background array to subtract to the data
    :param normalize: 'skip' to skip, 'monitor'  to normalize by the default monitor,
     'sum_roi' to normalize by the integrated intensity in the region of interest
     defined by detector.sum_roi
    :param debugging:  set to True to see plots
    :param kwargs:
     - 'photon_threshold' = float, photon threshold to apply before binning

    :return:
     - the 3D data and mask arrays
     - frames_logical: array of initial length the number of measured frames.
       In case of padding the length changes. A frame whose index is set to 1 means
       that it is used, 0 means not used, -1 means padded (added) frame.
     - the monitor values used for the intensity normalization

    """
    valid.valid_item(bin_during_loading, allowed_types=bool, name="bin_during_loading")
    # check and load kwargs
    valid.valid_kwargs(
        kwargs=kwargs,
        allowed_kwargs={"photon_threshold"},
        name="kwargs",
    )
    photon_threshold = kwargs.get("photon_threshold", 0)
    valid.valid_item(
        photon_threshold,
        allowed_types=Real,
        min_included=0,
        name="photon_threshold",
    )

    rawdata, rawmask, monitor, frames_logical = setup.diffractometer.load_check_dataset(
        logfile=logfile,
        scan_number=scan_number,
        detector=detector,
        setup=setup,
        bin_during_loading=bin_during_loading,
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

    rawdata = beamstop_correction(
        data=rawdata, detector=detector, setup=setup, debugging=debugging
    )

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


def reload_cdi_data(
    data,
    mask,
    logfile,
    scan_number,
    detector,
    setup,
    normalize_method="skip",
    debugging=False,
    **kwargs,
):
    """
    Reload forward CDI data, apply optional threshold, normalization and binning.

    :param data: the 3D data array
    :param mask: the 3D mask array
    :param logfile: file containing the information about the scan and image numbers
     (specfile, .fio...)
    :param scan_number: the scan number to load
    :param detector: an instance of the class Detector
    :param setup: an instance of the class Setup
    :param normalize_method: 'skip' to skip, 'monitor'  to normalize by the default
     monitor, 'sum_roi' to normalize by the integrated intensity in a defined region
     of interest
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
        name="kwargs",
    )
    photon_threshold = kwargs.get("photon_threshold", 0)
    valid.valid_item(
        photon_threshold,
        allowed_types=Real,
        min_included=0,
        name="photon_threshold",
    )

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
    else:
        if normalize_method == "sum_roi":
            monitor = data[
                :,
                detector.sum_roi[0] : detector.sum_roi[1],
                detector.sum_roi[2] : detector.sum_roi[3],
            ].sum(axis=(1, 2))
        else:  # use the default monitor of the beamline
            monitor = setup.diffractometer.read_monitor(
                scan_number=scan_number,
                logfile=logfile,
                beamline=setup.beamline,
                actuators=setup.actuators,
            )

        print("Intensity normalization using " + normalize_method)
        data, monitor = diff.normalize_dataset(
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
        start = (0, max(0, abs(detector.roi[0])), max(0, abs(detector.roi[2])))
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
