# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np
import pathlib
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import sys
sys.path.append('D:/myscripts/bcdi/')
import bcdi.graph.graph_utils as gu
import bcdi.preprocessing.preprocessing_utils as pru
import bcdi.postprocessing.postprocessing_utils as pu

helptext = """
Average several BCDI or CDI scans after an optional alignement step, based on a threshold on their Pearson correlation
coefficient.

The alignment of diffraction patterns is based on the center of mass shift or dft registration, using Python regular
grid interpolator or subpixel shift. Note thta there are many artefacts when using subpixel shift in reciprocal space.
"""

scans = np.arange(1138, 1141+1, 3)  # list or array of scan numbers
scans = np.concatenate((scans, np.arange(1147, 1195+1, 3)))
# bad_indices = np.argwhere(scans == 738)
# scans = np.delete(scans, bad_indices)
sample_name = ['dewet2_2']  # list of sample names. If only one name is indicated,
# it will be repeated to match the length of scans
suffix = '_norm_141_580_580_1_1_1.npz'  # '_ortho_norm_1160_1083_1160_2_2_2.npz'
# the end of the filename template after 'pynx'
homedir = "D:/data/P10_OER/data/"  # parent folder of scans folders
savedir = "D:/data/P10_OER/analysis/candidate_11/dewet2_2_S" + str(scans[0]) + "_to_S" + str(scans[-1]) + "/"
# path of the folder to save data
alignement_method = 'registration'
# method to find the translational offset, 'skip', 'center_of_mass' or 'registration'
combining_method = 'rgi'  # 'rgi' for RegularGridInterpolator or 'subpixel' for subpixel shift
corr_roi = None  # [325, 400, 845, 920, 410, 485]
# [420, 520, 660, 760, 600, 700]  # region of interest where to calculate the correlation between scans.
# If None, it will use the full
# array. [zstart, zstop, ystart, ystop, xstart, xstop]
output_shape = (140, 300, 300)  # (1160, 1083, 1160)  # the output dataset will be cropped/padded to this shape
crop_center = None  # [z, y, x] pixels position in the original array of the center of the cropped output
# if None, it will be set to the center of the original array
boundaries = 'crop'  # 'mask' or 'crop'. If 'mask', pixels were not all scans are defined after alignement will be
# masked, if 'crop' output_shape will be modified to remove these boundary pixels
correlation_threshold = 0.90  # only scans having a correlation larger than this threshold will be combined
reference_scan = 0  # index in scans of the scan to be used as the reference for the correlation calculation
combine_masks = False  # if True, the output mask is the combination of all masks. If False, the reference mask is used
is_orthogonal = False  # if True, it will look for the data in a folder named /pynx, otherwise in /pynxraw
plot_threshold = 0  # data below this will be set to 0, only in plots
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
assert len(output_shape) == 3, 'output_shape should be a list or tuple of three numbers'
assert np.all(np.asarray(output_shape) % 2 == 0), 'output_shape components should be all even due to FFT shape' \
                                                ' considerations for phase retrieval'
if type(sample_name) is list:
    if len(sample_name) == 1:
        sample_name = [sample_name[0] for idx in range(len(scans))]
    assert len(sample_name) == len(scans), 'sample_name and scans should have the same length'
elif type(sample_name) is str:
    sample_name = [sample_name for idx in range(len(scans))]
else:
    print('sample_name should be either a string or a list of strings')
    sys.exit()

assert boundaries in ['mask', 'crop'], 'boundaries should be either "mask" or "crop"'

if is_orthogonal:
    parent_folder = '/pynx/'
else:
    parent_folder = '/pynxraw/'

###########################
# load the reference scan #
###########################
plt.ion()
print(scans)
samplename = sample_name[reference_scan] + '_' + str('{:05d}').format(scans[reference_scan])
print('Reference scan:', samplename)
refdata = np.load(homedir + samplename + parent_folder +
                  'S' + str(scans[reference_scan]) + '_pynx' + suffix)['data']
refmask = np.load(homedir + samplename + parent_folder +
                  'S' + str(scans[reference_scan]) + '_maskpynx' + suffix)['mask']
assert refdata.ndim == 3 and refmask.ndim == 3, 'data and mask should be 3D arrays'
nbz, nby, nbx = refdata.shape

crop_center = list(crop_center or [nbz // 2, nby // 2, nbx // 2])  # if None, default to the middle of the array
assert len(crop_center) == 3, 'crop_center should be a list or tuple of three indices'
assert np.all(np.asarray(crop_center)-np.asarray(output_shape)//2 >= 0), 'crop_center incompatible with output_shape'
assert crop_center[0]+output_shape[0]//2 <= nbz and crop_center[1]+output_shape[1]//2 <= nby\
        and crop_center[2]+output_shape[2]//2 <= nbx, 'crop_center incompatible with output_shape'

corr_roi = corr_roi or [0, nbz, 0, nby, 0, nbx]  # if None, use the full array for corr_roi

assert len(corr_roi) == 6, 'corr_roi should be a tuple or list of lenght 6'
if not 0 <= corr_roi[0] < corr_roi[1] <= nbz\
        or not 0 <= corr_roi[2] < corr_roi[3] <= nby\
        or not 0 <= corr_roi[4] < corr_roi[5] <= nbx:
    print('Incorrect value for the parameter corr_roi')
    sys.exit()

gu.multislices_plot(refdata[corr_roi[0]:corr_roi[1], corr_roi[2]:corr_roi[3], corr_roi[4]:corr_roi[5]],
                    sum_frames=True, scale='log', plot_colorbar=True, title='refdata in corr_roi', vmin=0,
                    reciprocal_space=True, is_orthogonal=is_orthogonal)

###########################
# combine the other scans #
###########################
shift_min = [0, 0, 0]  # min of the shift of the first axis after alignement
shift_max = [0, 0, 0]  # max of the shift of the first axis after alignement
combined_list = []  # list of scans with correlation coeeficient >= threshold
corr_coeff = []  # list of correlation coefficients
sumdata = np.copy(refdata)  # refdata must not be modified
summask = refmask  # refmask is not used elsewhere

for idx in range(len(scans)):
    if idx == reference_scan:
        combined_list.append(scans[idx])
        corr_coeff.append(1.0)
        continue  # sumdata and summask were already initialized with the reference scan
    samplename = sample_name[idx] + '_' + str('{:05d}').format(scans[idx])
    print('\n Opening ', samplename)
    data = np.load(homedir + samplename + parent_folder +
                   'S' + str(scans[idx]) + '_pynx' + suffix)['data']
    mask = np.load(homedir + samplename + parent_folder +
                   'S' + str(scans[idx]) + '_maskpynx' + suffix)['mask']

    if debug:
        gu.multislices_plot(data, sum_frames=True, scale='log', plot_colorbar=True,
                            title='S' + str(scans[idx]) + '\n Data before shift', vmin=0,
                            reciprocal_space=True, is_orthogonal=is_orthogonal)

        gu.multislices_plot(mask, sum_frames=True, scale='linear', plot_colorbar=True,
                            title='S' + str(scans[idx]) + '\n Mask before shift', vmin=0,
                            reciprocal_space=True, is_orthogonal=is_orthogonal)
    ##################
    # align datasets #
    ##################
    if alignement_method is not 'skip':
        data, mask, shifts = pru.align_diffpattern(reference_data=refdata, data=data, mask=mask,
                                                   method=alignement_method, combining_method=combining_method,
                                                   return_shift=True)
        shift_min = [min(shift_min[axis], shifts[axis]) for axis in range(3)]
        shift_max = [max(shift_max[axis], shifts[axis]) for axis in range(3)]
        if debug:
            gu.multislices_plot(data, sum_frames=True, scale='log', plot_colorbar=True,
                                title='S' + str(scans[idx]) + '\n Data after shift', vmin=0,
                                reciprocal_space=True, is_orthogonal=is_orthogonal)

            gu.multislices_plot(mask, sum_frames=True, scale='linear', plot_colorbar=True,
                                title='S' + str(scans[idx]) + '\n Mask after shift', vmin=0,
                                reciprocal_space=True, is_orthogonal=is_orthogonal)

    correlation = pearsonr(
        np.ndarray.flatten(abs(refdata[corr_roi[0]:corr_roi[1], corr_roi[2]:corr_roi[3], corr_roi[4]:corr_roi[5]])),
        np.ndarray.flatten(abs(data[corr_roi[0]:corr_roi[1], corr_roi[2]:corr_roi[3], corr_roi[4]:corr_roi[5]])))[0]
    print('Rocking curve ', idx+1, ': Pearson correlation coefficient = ', str('{:.2f}'.format(correlation)))
    corr_coeff.append(round(correlation, 2))

    if correlation >= correlation_threshold:
        combined_list.append(scans[idx])
        sumdata = sumdata + data
        if combine_masks:
            summask = summask + mask
    else:
        print('Scan ', scans[idx], ', correlation below threshold, skip concatenation')

summask[np.nonzero(summask)] = 1  # mask should be 0 or 1
sumdata = sumdata / len(combined_list)

##########################################################################################
# exclude or mask boundaries where not all scans are defined after alignment (BCDI case) #
##########################################################################################
if alignement_method is not 'skip':
    shift_min = [int(np.ceil(abs(shift_min[axis]))) for axis in range(3)]
    # shift_min is the number of pixels to remove at the end along each axis
    shift_max = [int(np.ceil(shift_max[axis])) for axis in range(3)]
    # shft_max is the number of pixels to remove at the beginning along each axis
    print('\nnumber of pixels to remove (start, end) = ', shift_max, ', ', shift_min)

    if boundaries == 'mask':
        sumdata[0:shift_max[0], :, :] = 0
        sumdata[shift_min[0]:, :, :] = 0
        sumdata[:, 0:shift_max[1], :] = 0
        sumdata[:, shift_min[1]:, :] = 0
        sumdata[:, :, 0:shift_max[2]] = 0
        sumdata[:, :, shift_min[2]:] = 0

        summask[0:shift_max[0], :, :] = 1
        summask[shift_min[0]:, :, :] = 1
        summask[:, 0:shift_max[1], :] = 1
        summask[:, shift_min[1]:, :] = 1
        summask[:, :, 0:shift_max[2]] = 1
        summask[:, :, shift_min[2]:] = 1
    else:  # 'crop', will redefined output_shape and crop_center to remove boundaries
        # check along axis 0
        if crop_center[0] - output_shape[0] // 2 < shift_max[0]:  # not enough pixels on the lower indices side
            delta_z = shift_max[0] - crop_center[0] + output_shape[0] // 2
            center_z = crop_center[0] + delta_z
        else:
            center_z = crop_center[0]
        # check if this still fit on the larger indices side
        if center_z + output_shape[0] // 2 > nbz - shift_min[0]:  # not enough pixels on the larger indices side
            print('cannot crop the first axis to {:d}'.format(output_shape[0]))
            # find the correct output_shape[0] taking into accournt FFT shape considerations
            output_shape[0] = pru.smaller_primes(nbz - shift_min[0] - shift_max[0], maxprime=7, required_dividers=(2,))
            # redefine crop_center[0] if needed
            if crop_center[0] - output_shape[0] // 2 < shift_max[0]:
                delta_z = shift_max[0] - crop_center[0] + output_shape[0] // 2
                crop_center[0] = crop_center[0] + delta_z

        # check along axis 1
        if crop_center[1] - output_shape[1] // 2 < shift_max[1]:  # not enough pixels on the lower indices side
            delta_y = shift_max[1] - crop_center[1] + output_shape[1] // 2
            center_y = crop_center[1] + delta_y
        else:
            center_y = crop_center[1]
        # check if this still fit on the larger indices side
        if center_y + output_shape[1] // 2 > nby - shift_min[1]:  # not enough pixels on the larger indices side
            print('cannot crop the second axis to {:d}'.format(output_shape[1]))
            # find the correct output_shape[1] taking into accournt FFT shape considerations
            output_shape[1] = pru.smaller_primes(nby - shift_min[1] - shift_max[1], maxprime=7, required_dividers=(2,))
            # redefine crop_center[1] if needed
            if crop_center[1] - output_shape[1] // 2 < shift_max[1]:
                delta_y = shift_max[1] - crop_center[1] + output_shape[1] // 2
                crop_center[1] = crop_center[1] + delta_y

        # check along axis 2
        if crop_center[2] - output_shape[2] // 2 < shift_max[2]:  # not enough pixels on the lower indices side
            delta_x = shift_max[2] - crop_center[2] + output_shape[2] // 2
            center_x = crop_center[2] + delta_x
        else:
            center_x = crop_center[2]
        # check if this still fit on the larger indices side
        if center_x + output_shape[2] // 2 > nbx - shift_min[2]:  # not enough pixels on the larger indices side
            print('cannot crop the third axis to {:d}'.format(output_shape[2]))
            # find the correct output_shape[2] taking into accournt FFT shape considerations
            output_shape[2] = pru.smaller_primes(nbx - shift_min[2] - shift_max[2], maxprime=7, required_dividers=(2,))
            # redefine crop_center[2] if needed
            if crop_center[2] - output_shape[2] // 2 < shift_max[2]:
                delta_x = shift_max[2] - crop_center[2] + output_shape[2] // 2
                crop_center[2] = crop_center[2] + delta_x

        print('new crop size for the first axis=', output_shape, 'new crop_center=', crop_center)

########################################################
# crop the combined data and mask to the desired shape #
########################################################
summask = pu.crop_pad(array=summask, output_shape=output_shape, crop_center=crop_center)
sumdata = pu.crop_pad(array=sumdata, output_shape=output_shape, crop_center=crop_center)

###################################
# save the combined data and mask #
###################################
template = '_S' + str(combined_list[0]) + 'toS' + str(combined_list[-1]) +\
           '_{:d}_{:d}_{:d=}'.format(output_shape[0], output_shape[1], output_shape[2])


pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)
np.savez_compressed(savedir+'combined_pynx' + template + '.npz', data=sumdata)
np.savez_compressed(savedir+'combined_maskpynx' + template + '.npz', mask=summask)
print('\nSum of ', len(combined_list), 'scans')

###################################
# plot the combined data and mask #
###################################
gu.multislices_plot(sumdata[corr_roi[0]:corr_roi[1], corr_roi[2]:corr_roi[3], corr_roi[4]:corr_roi[5]],
                    sum_frames=True, scale='log', plot_colorbar=True, title='sumdata in corr_roi', vmin=0,
                    reciprocal_space=True, is_orthogonal=is_orthogonal)

sumdata[np.nonzero(summask)] = 0
sumdata[sumdata < plot_threshold] = 0
fig, _, _ = gu.multislices_plot(sumdata, sum_frames=True, scale='log', plot_colorbar=True, is_orthogonal=is_orthogonal,
                                title='Combined masked intensity', vmin=0, reciprocal_space=True)
fig.text(0.55, 0.40, "Scans tested:", size=12)
fig.text(0.55, 0.35, str(scans), size=8)
fig.text(0.55, 0.30, "Correlation coefficients:", size=12)
fig.text(0.55, 0.25, str(corr_coeff), size=8)
fig.text(0.55, 0.20, "Threshold for correlation: " + str(correlation_threshold), size=12)
fig.text(0.55, 0.15, 'Scans concatenated:', size=12)
fig.text(0.55, 0.10, str(combined_list), size=8)
if plot_threshold != 0:
    fig.text(0.60, 0.05, "Threshold for plots only: {:d}".format(plot_threshold), size=12)

plt.pause(0.1)
plt.savefig(savedir + 'data' + template + '.png')

fig, _, _ = gu.multislices_plot(sumdata, sum_frames=False, scale='log', plot_colorbar=True, is_orthogonal=is_orthogonal,
                                slice_position=crop_center, title='Combined masked intensity', vmin=0,
                                reciprocal_space=True)
fig.text(0.55, 0.40, "Scans tested:", size=12)
fig.text(0.55, 0.35, str(scans), size=8)
fig.text(0.55, 0.30, "Correlation coefficients:", size=12)
fig.text(0.55, 0.25, str(corr_coeff), size=8)
fig.text(0.55, 0.20, "Threshold for correlation: " + str(correlation_threshold), size=12)
fig.text(0.55, 0.15, 'Scans concatenated:', size=12)
fig.text(0.55, 0.10, str(combined_list), size=8)
if plot_threshold != 0:
    fig.text(0.60, 0.05, "Threshold for plots only: {:d}".format(plot_threshold), size=12)

plt.pause(0.1)
plt.savefig(savedir + 'data_slice' + template + '.png')

gu.multislices_plot(summask, sum_frames=True, scale='linear', plot_colorbar=True,
                    title='Combined mask', vmin=0, reciprocal_space=True, is_orthogonal=is_orthogonal)
plt.savefig(savedir + 'mask' + template + '.png')

print('\nEnd of script')

plt.ioff()
plt.show()
