#!/usr/bin/env python3

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

import bcdi.graph.graph_utils as gu
import bcdi.utils.image_registration as reg
import bcdi.utils.utilities as util

helptext = """
Average several BCDI or CDI scans after an optional alignement step, based on a
threshold on their Pearson correlation coefficient. The alignment of diffraction
patterns can be realized using Python regular grid interpolator, subpixel shift or a
simple roll of pixels. Note that there are many artefacts when interpolating in
reciprocal space, rolling is the best option.

The script expects that datasets have already been stacked, for example using the script
bcdi_preprocessing_BCDI. Relative to root_folder, the stacked data for scan S1 would be
in /root_folder/S1/S1_pynx_suffix and the mask in /root_folder/S1/S1_maskpynx_suffix

In this example, sample_name=["S"] and suffix=["_suffix"], but it could be different for
each dataset (hence the use of a list).
"""

scans = np.arange(314, 322 + 1, 4)
# list or array of scan numbers
# bad_indices = np.argwhere(scans == 738)
# scans = np.delete(scans, bad_indices)
sample_name = ["PtNP1"]
# list of sample names. If only one name is indicated,
# it will be repeated to match the length of scans
suffix = ["_norm_250_1100_900_1_1_1.npz"]
# list of sample names (end of the filename template after 'pynx'),
# it will be repeated to match the length of scans
root_folder = "C:/Users/Jerome/Documents/data/isosurface/"
# parent folder of scans folders
save_dir = "C:/Users/Jerome/Documents/data/isosurface/test"
# path of the folder to save data
shift_method = "raw"  # ' raw', 'modulus', 'support' or 'skip'
# Object to use for the determination of the shift. If 'raw', it uses the raw,
# eventually complex array. if 'modulus', it uses the modulus of the array.
# If 'support', it uses a support created by threshold the modulus of the array.
# if 'skip', it skips the alignment.
interpolation_method = "roll"  # 'rgi' for RegularGridInterpolator,
# 'subpixel' for subpixel shift, 'roll' to shift voxels by an integral number
# (the shifts are rounded to the nearest integer)
corr_roi = None
# [420, 520, 660, 760, 600, 700]
# region of interest where to calculate the correlation between scans.
# If None, it will use the full array. [zstart, zstop, ystart, ystop, xstart, xstop]
output_shape = (250, 1024, 800)
# (1160, 1083, 1160)  # the output dataset will be cropped/padded to this shape
apply_fft_constraint = True  # set to True to enforce the constraint on the FFT shape
# for phase retrieval, output_shape may be overriden.
crop_center = None  # [z, y, x] pixels position in the original array
# of the center of the cropped output
# if None, it will be set to the center of the original array
boundaries = "crop"  # 'mask', 'crop' or 'skip'.
# If 'mask', boundary pixels were not all scans are defined after
# alignement will be masked, if 'crop' output_shape will be modified to crop them.
# If 'skip', boundaries will not be processed.
partially_masked = "unmask"  # 'unmask' or 'mask'.
# If 'unmask', partially masked pixels will be set to their mean value
# and unmasked. If 'mask', partially masked pixels will be set to 0 and masked.
correlation_threshold = 0.95
# only scans having a correlation larger than this threshold will be combined
reference_scan = 0
# index in scans of the scan to be used as the reference for the correlation calculation
combine_masks = True  # if True, the output mask is the union of all masks.
# If False, the reference mask is used.
# If a pixel is defined only in part of the dataset, its value will be used with proper
# rescaling
is_orthogonal = False  # if True, it will look for the data in a folder named /pynx,
# otherwise in /pynxraw
plot_threshold = 0  # data below this will be set to 0, only in plots
comment = ""  # should start with _ , it will be added to the filename
# when saving the combined dataset
debug = False  # True or False
##################################
# end of user-defined parameters #
##################################

#########################
# check some parameters #
#########################
if reference_scan is None:
    reference_scan = 0

if type(output_shape) is tuple:
    output_shape = list(output_shape)
if len(output_shape) != 3:
    raise ValueError("output_shape should be a list or tuple of three numbers")
if apply_fft_constraint:
    output_shape = util.smaller_primes(output_shape, maxprime=7, required_dividers=(2,))
    print(
        "\nChecking that output_shape fits FFT shape requirements for phase retrieval, "
        f"output_shape={output_shape}"
    )
if isinstance(sample_name, (tuple, list)):
    if len(sample_name) == 1:
        sample_name = [sample_name[0] for idx in range(len(scans))]
    if len(sample_name) != len(scans):
        raise ValueError("sample_name and scans should have the same length")
elif isinstance(sample_name, str):
    sample_name = [sample_name for idx in range(len(scans))]
else:
    print("sample_name should be either a string or a list of strings")
    sys.exit()

if isinstance(suffix, (tuple, list)):
    if len(suffix) == 1:
        suffix = [suffix[0] for idx in range(len(scans))]
    if len(suffix) != len(scans):
        raise ValueError("sample_name and scans should have the same length")
elif isinstance(suffix, str):
    suffix = [suffix for idx in range(len(scans))]
else:
    print("suffix should be either a string or a list of strings")
    sys.exit()

if boundaries not in {
    "mask",
    "crop",
    "skip",
}:
    raise ValueError('boundaries should be either "mask", "crop" or "skip"')

if is_orthogonal:
    parent_folder = "/pynx/"
else:
    parent_folder = "/pynxraw/"

###########################
# load the reference scan #
###########################
plt.ion()
print(f"Scan list: {scans}")
samplename = sample_name[reference_scan] + "_" + f"{scans[reference_scan]:05d}"
print("Reference scan:", samplename)
refdata = np.load(
    root_folder
    + samplename
    + parent_folder
    + "S"
    + str(scans[reference_scan])
    + "_pynx"
    + suffix[reference_scan]
)["data"]
refmask = np.load(
    root_folder
    + samplename
    + parent_folder
    + "S"
    + str(scans[reference_scan])
    + "_maskpynx"
    + suffix[reference_scan]
)["mask"]
if not (refdata.ndim == 3 and refmask.ndim == 3):
    raise ValueError("data and mask should be 3D arrays")
nbz, nby, nbx = refdata.shape

#################################################################
# check parameters depending on the shape of the reference scan #
#################################################################
crop_center = list(
    crop_center or [nbz // 2, nby // 2, nbx // 2]
)  # if None, default to the middle of the array
if len(crop_center) != 3:
    raise ValueError("crop_center should be a list or tuple of three indices")
if not np.all(np.asarray(crop_center) - np.asarray(output_shape) // 2 >= 0):
    raise ValueError("crop_center incompatible with output_shape")
if not (
    crop_center[0] + output_shape[0] // 2 <= nbz
    and crop_center[1] + output_shape[1] // 2 <= nby
    and crop_center[2] + output_shape[2] // 2 <= nbx
):
    raise ValueError("crop_center incompatible with output_shape")

corr_roi = corr_roi or [
    0,
    nbz,
    0,
    nby,
    0,
    nbx,
]  # if None, use the full array for corr_roi

if len(corr_roi) != 6:
    raise ValueError("corr_roi should be a tuple or list of lenght 6")
if (
    not 0 <= corr_roi[0] < corr_roi[1] <= nbz
    or not 0 <= corr_roi[2] < corr_roi[3] <= nby
    or not 0 <= corr_roi[4] < corr_roi[5] <= nbx
):
    print("Incorrect value for the parameter corr_roi")
    sys.exit()

# crop the data directly to output_shape if no alignment is required,
# update corr_roi accordingly
if shift_method == "skip":
    refmask = util.crop_pad(
        array=refmask, output_shape=output_shape, crop_center=crop_center
    )
    refdata = util.crop_pad(
        array=refdata, output_shape=output_shape, crop_center=crop_center
    )
    # correct for the offset due to cropping
    corr_roi = [
        corr_roi[0] - (crop_center[0] - output_shape[0] // 2),
        corr_roi[1] - (crop_center[0] - output_shape[0] // 2),
        corr_roi[2] - (crop_center[1] - output_shape[1] // 2),
        corr_roi[3] - (crop_center[1] - output_shape[1] // 2),
        corr_roi[4] - (crop_center[2] - output_shape[2] // 2),
        corr_roi[5] - (crop_center[2] - output_shape[2] // 2),
    ]
    # check if this still fits the cropped data, default to the data range otherwise
    corr_roi[0] = max(0, corr_roi[0])
    corr_roi[1] = min(output_shape[0], corr_roi[1])
    corr_roi[2] = max(0, corr_roi[2])
    corr_roi[3] = min(output_shape[1], corr_roi[3])
    corr_roi[4] = max(0, corr_roi[4])
    corr_roi[5] = min(output_shape[2], corr_roi[5])
    print("Corrected corr_roi after cropping:", corr_roi)

# replace nans by 0 and mask them
refmask[np.isnan(refdata)] = 1
refdata[np.isnan(refdata)] = 0

gu.multislices_plot(
    refdata[
        corr_roi[0] : corr_roi[1], corr_roi[2] : corr_roi[3], corr_roi[4] : corr_roi[5]
    ],
    sum_frames=True,
    scale="log",
    plot_colorbar=True,
    title="refdata in corr_roi",
    vmin=0,
    reciprocal_space=True,
    is_orthogonal=is_orthogonal,
)

###################################################
# combine the other scans with the reference scan #
###################################################
shift_min = [0, 0, 0]  # min of the shift of the first axis after alignement
shift_max = [0, 0, 0]  # max of the shift of the first axis after alignement
combined_list = []  # list of scans with correlation coeeficient >= threshold
corr_coeff = []  # list of correlation coefficients
sumdata = np.copy(refdata)  # refdata must not be modified
summask = refmask  # refmask is not used elsewhere

for idx, item in enumerate(scans):
    if idx == reference_scan:
        combined_list.append(item)
        corr_coeff.append(1.0)
        continue  # sumdata and summask were already initialized with the reference scan
    samplename = sample_name[idx] + "_" + f"{item:05d}"
    print(
        f'\n{"#" * len(samplename)}\n' + samplename + "\n" + f'{"#" * len(samplename)}'
    )
    data = np.load(
        root_folder + samplename + parent_folder + f"S{item}_pynx" + suffix[idx]
    )["data"]
    mask = np.load(
        root_folder
        + samplename
        + parent_folder
        + "S"
        + str(item)
        + "_maskpynx"
        + suffix[idx]
    )["mask"]

    # replace nans by 0 and mask them
    mask[np.isnan(data)] = 1
    data[np.isnan(data)] = 0

    if debug:
        gu.multislices_plot(
            data,
            sum_frames=True,
            scale="log",
            plot_colorbar=True,
            title="S" + str(item) + "\n Data before shift",
            vmin=0,
            reciprocal_space=True,
            is_orthogonal=is_orthogonal,
        )

        gu.multislices_plot(
            mask,
            sum_frames=True,
            scale="linear",
            plot_colorbar=True,
            title="S" + str(item) + "\n Mask before shift",
            vmin=0,
            reciprocal_space=True,
            is_orthogonal=is_orthogonal,
        )
    ##################
    # align datasets #
    ##################
    if shift_method != "skip":
        data, mask, shifts = reg.align_diffpattern(
            reference_data=refdata,
            data=data,
            mask=mask,
            shift_method=shift_method,
            interpolation_method=interpolation_method,
        )
        shift_min = [min(shift_min[axis], shifts[axis]) for axis in range(3)]
        shift_max = [max(shift_max[axis], shifts[axis]) for axis in range(3)]
        if debug:
            gu.multislices_plot(
                data,
                sum_frames=True,
                scale="log",
                plot_colorbar=True,
                title="S" + str(item) + "\n Data after shift",
                vmin=0,
                reciprocal_space=True,
                is_orthogonal=is_orthogonal,
            )

            gu.multislices_plot(
                mask,
                sum_frames=True,
                scale="linear",
                plot_colorbar=True,
                title="S" + str(item) + "\n Mask after shift",
                vmin=0,
                reciprocal_space=True,
                is_orthogonal=is_orthogonal,
            )
    else:  # crop the data directly to output_shape
        mask = util.crop_pad(
            array=mask, output_shape=output_shape, crop_center=crop_center
        )
        data = util.crop_pad(
            array=data, output_shape=output_shape, crop_center=crop_center
        )

    gu.multislices_plot(
        data[
            corr_roi[0] : corr_roi[1],
            corr_roi[2] : corr_roi[3],
            corr_roi[4] : corr_roi[5],
        ],
        sum_frames=True,
        scale="log",
        plot_colorbar=True,
        title="data in corr_roi",
        vmin=0,
        reciprocal_space=True,
        is_orthogonal=is_orthogonal,
    )
    ########################################################
    # combine datasets if their correlation is good enough #
    ########################################################
    correlation = pearsonr(
        np.ndarray.flatten(
            abs(
                refdata[
                    corr_roi[0] : corr_roi[1],
                    corr_roi[2] : corr_roi[3],
                    corr_roi[4] : corr_roi[5],
                ]
            )
        ),
        np.ndarray.flatten(
            abs(
                data[
                    corr_roi[0] : corr_roi[1],
                    corr_roi[2] : corr_roi[3],
                    corr_roi[4] : corr_roi[5],
                ]
            )
        ),
    )[0]
    print(
        "Dataset ",
        idx + 1,
        ": Pearson correlation coefficient = ",
        str(f"{correlation:.3f}"),
    )
    corr_coeff.append(round(correlation, 2))
    if correlation >= correlation_threshold:
        combined_list.append(item)
        sumdata = sumdata + data
        if combine_masks:
            summask = summask + mask
    else:
        print("Scan ", item, ", correlation below threshold, skip concatenation")

###################################################################################
# process boundaries, where some voxels can be undefined after aligning a dataset #
###################################################################################
if shift_method != "skip":
    shift_min = [int(np.ceil(abs(shift_min[axis]))) for axis in range(3)]
    # shift_min is the number of pixels to remove at the end along each axis
    shift_max = [int(np.ceil(shift_max[axis])) for axis in range(3)]
    # shft_max is the number of pixels to remove at the beginning along each axis
    print("\nnumber of pixels to remove (start, end) = ", shift_max, ", ", shift_min)

    if boundaries == "mask":
        # when alignment is skipped, shift_min=[0, 0, 0] and shift_max=[0, 0, 0],
        # resulting in empty slices (nothing masked)
        sumdata[0 : shift_max[0], :, :] = 0
        sumdata[nbz - shift_min[0] :, :, :] = 0
        sumdata[:, 0 : shift_max[1], :] = 0
        sumdata[:, nby - shift_min[1] :, :] = 0
        sumdata[:, :, 0 : shift_max[2]] = 0
        sumdata[:, :, nbx - shift_min[2] :] = 0

        summask[0 : shift_max[0], :, :] = 1
        summask[nbz - shift_min[0] :, :, :] = 1
        summask[:, 0 : shift_max[1], :] = 1
        summask[:, nby - shift_min[1] :, :] = 1
        summask[:, :, 0 : shift_max[2]] = 1
        summask[:, :, nbx - shift_min[2] :] = 1

    elif (
        boundaries == "crop"
    ):  # will redefined output_shape and crop_center to remove boundaries
        # check along axis 0
        if (
            crop_center[0] - output_shape[0] // 2 < shift_max[0]
        ):  # not enough pixels on the lower indices side
            delta_z = shift_max[0] - crop_center[0] + output_shape[0] // 2
            center_z = crop_center[0] + delta_z
        else:
            center_z = crop_center[0]
        # check if this still fit on the larger indices side
        if (
            center_z + output_shape[0] // 2 > nbz - shift_min[0]
        ):  # not enough pixels on the larger indices side
            print(f"cannot crop the first axis to {output_shape[0]:d}")
            # find the correct output_shape[0]
            # taking into accournt FFT shape considerations
            output_shape[0] = util.smaller_primes(
                nbz - shift_min[0] - shift_max[0], maxprime=7, required_dividers=(2,)
            )
            # redefine crop_center[0] if needed
            if crop_center[0] - output_shape[0] // 2 < shift_max[0]:
                delta_z = shift_max[0] - crop_center[0] + output_shape[0] // 2
                crop_center[0] = crop_center[0] + delta_z

        # check along axis 1
        if (
            crop_center[1] - output_shape[1] // 2 < shift_max[1]
        ):  # not enough pixels on the lower indices side
            delta_y = shift_max[1] - crop_center[1] + output_shape[1] // 2
            center_y = crop_center[1] + delta_y
        else:
            center_y = crop_center[1]
        # check if this still fit on the larger indices side
        if (
            center_y + output_shape[1] // 2 > nby - shift_min[1]
        ):  # not enough pixels on the larger indices side
            print(f"cannot crop the second axis to {output_shape[1]:d}")
            # find the correct output_shape[1]
            # taking into accournt FFT shape considerations
            output_shape[1] = util.smaller_primes(
                nby - shift_min[1] - shift_max[1], maxprime=7, required_dividers=(2,)
            )
            # redefine crop_center[1] if needed
            if crop_center[1] - output_shape[1] // 2 < shift_max[1]:
                delta_y = shift_max[1] - crop_center[1] + output_shape[1] // 2
                crop_center[1] = crop_center[1] + delta_y

        # check along axis 2
        if (
            crop_center[2] - output_shape[2] // 2 < shift_max[2]
        ):  # not enough pixels on the lower indices side
            delta_x = shift_max[2] - crop_center[2] + output_shape[2] // 2
            center_x = crop_center[2] + delta_x
        else:
            center_x = crop_center[2]
        # check if this still fit on the larger indices side
        if (
            center_x + output_shape[2] // 2 > nbx - shift_min[2]
        ):  # not enough pixels on the larger indices side
            print(f"cannot crop the third axis to {output_shape[2]:d}")
            # find the correct output_shape[2]
            # taking into accournt FFT shape considerations
            output_shape[2] = util.smaller_primes(
                nbx - shift_min[2] - shift_max[2], maxprime=7, required_dividers=(2,)
            )
            # redefine crop_center[2] if needed
            if crop_center[2] - output_shape[2] // 2 < shift_max[2]:
                delta_x = shift_max[2] - crop_center[2] + output_shape[2] // 2
                crop_center[2] = crop_center[2] + delta_x

        print(
            "new crop size for the first axis=",
            output_shape,
            "new crop_center=",
            crop_center,
        )

    else:  # 'skip'
        print("no process of the boundaries")

    # crop the combined data and mask to the desired shape,
    # when alignment_method is 'skip' it is done beforehand
    summask = util.crop_pad(
        array=summask, output_shape=output_shape, crop_center=crop_center
    )
    sumdata = util.crop_pad(
        array=sumdata, output_shape=output_shape, crop_center=crop_center
    )

##################################################
# normalize sumdata using the counter in summask #
##################################################
mean_data = sumdata
if partially_masked == "unmask":
    unmask_ind = summask != len(
        combined_list
    )  # summask will be = len(combined_list) for pixels totally masked
    mean_data[unmask_ind] = np.divide(
        mean_data[unmask_ind], len(combined_list) - summask[unmask_ind]
    )
    summask[summask != len(combined_list)] = (
        0  # unmask voxels which are partially masked
    )
    summask[np.nonzero(summask)] = 1
else:  # 'mask', mask partially masked pixels
    summask[np.nonzero(summask)] = 1
    mean_data[np.nonzero(summask)] = 0
    mean_data = mean_data / len(combined_list)

###################################
# save the combined data and mask #
###################################
template = (
    "_S"
    + str(combined_list[0])
    + "toS"
    + str(combined_list[-1])
    + f"_{output_shape[0]:d}_{output_shape[1]:d}_{output_shape[2]:d=}"
    + comment
)


pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
np.savez_compressed(save_dir + "combined_pynx" + template + ".npz", data=mean_data)
np.savez_compressed(save_dir + "combined_maskpynx" + template + ".npz", mask=summask)
print("\nSum of ", len(combined_list), "scans")

###################################
# plot the combined data and mask #
###################################
gu.multislices_plot(
    mean_data[
        corr_roi[0] : corr_roi[1], corr_roi[2] : corr_roi[3], corr_roi[4] : corr_roi[5]
    ],
    sum_frames=True,
    scale="log",
    plot_colorbar=True,
    title="mean_data in corr_roi",
    vmin=0,
    reciprocal_space=True,
    is_orthogonal=is_orthogonal,
)

mean_data[np.nonzero(summask)] = 0
mean_data[mean_data < plot_threshold] = 0
fig, _, _ = gu.multislices_plot(
    mean_data,
    sum_frames=True,
    scale="log",
    plot_colorbar=True,
    is_orthogonal=is_orthogonal,
    reciprocal_space=True,
    title="Combined masked intensity",
    vmin=0,
)
fig.text(0.55, 0.40, "Scans tested:", size=12)
fig.text(0.55, 0.35, str(scans), size=8)
fig.text(0.55, 0.30, "Correlation coefficients:", size=12)
fig.text(0.55, 0.25, str(corr_coeff), size=8)
fig.text(
    0.55, 0.20, "Threshold for correlation: " + str(correlation_threshold), size=12
)
fig.text(0.55, 0.15, "Scans concatenated:", size=12)
fig.text(0.55, 0.10, str(combined_list), size=8)
if plot_threshold != 0:
    fig.text(0.60, 0.05, f"Threshold for plots only: {plot_threshold:d}", size=12)

plt.pause(0.1)
plt.savefig(save_dir + "data" + template + ".png")

fig, _, _ = gu.multislices_plot(
    mean_data,
    sum_frames=False,
    scale="log",
    plot_colorbar=True,
    is_orthogonal=is_orthogonal,
    reciprocal_space=True,
    slice_position=[crop_center[0], crop_center[1], crop_center[2]],
    title="Combined masked intensity",
    vmin=0,
)
fig.text(0.55, 0.40, "Scans tested:", size=12)
fig.text(0.55, 0.35, str(scans), size=8)
fig.text(0.55, 0.30, "Correlation coefficients:", size=12)
fig.text(0.55, 0.25, str(corr_coeff), size=8)
fig.text(
    0.55, 0.20, "Threshold for correlation: " + str(correlation_threshold), size=12
)
fig.text(0.55, 0.15, "Scans concatenated:", size=12)
fig.text(0.55, 0.10, str(combined_list), size=8)
if plot_threshold != 0:
    fig.text(0.60, 0.05, f"Threshold for plots only: {plot_threshold:d}", size=12)

plt.pause(0.1)
plt.savefig(save_dir + "data_slice" + template + ".png")

gu.multislices_plot(
    summask,
    sum_frames=True,
    scale="linear",
    plot_colorbar=True,
    title="Combined mask",
    vmin=0,
    reciprocal_space=True,
    is_orthogonal=is_orthogonal,
)
plt.savefig(save_dir + "mask" + template + ".png")

print("\nEnd of script")

plt.ioff()
plt.show()
