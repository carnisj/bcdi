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

scan_list = [22, 32, 48, 55, 59, 71, 6, 15, 37]  # np.arange(404, 407+1, 3)  # list or array of scan numbers
sample_name = ['ht_pillar3', 'ht_pillar3', 'ht_pillar3', 'ht_pillar3', 'ht_pillar3', 'ht_pillar3',
               'ht_pillar3_1', 'ht_pillar3_1', 'ht_pillar3_1']  # list of sample names. If only one name is indicated,
# it will be repeated to match the length of scan_list
comment = '_553_1083_1035_1_2_2.npz'  # '_ortho_norm_1160_1083_1160_2_2_2.npz'
# the end of the filename template after 'pynx'
homedir = "/nfs/fs/fscxi/experiments/2020/PETRA/P10/11008562/raw/"  # parent folder of scans folders
savedir = '/home/carnisj/phasing/'  # path of the folder to save data
alignement_method = 'skip'  # method to find the translational offset, 'skip', 'center_of_mass' or 'registration'
combining_method = 'subpixel'  # 'rgi' for RegularGridInterpolator or 'subpixel' for subpixel shift
corr_roi = [325, 400, 845, 920, 410, 485]
# [420, 520, 660, 760, 600, 700]  # region of interest where to calculate the correlation between scans.
# If None, it will use the full
# array. [zstart, zstop, ystart, ystop, xstart, xstop]
output_shape = (553, 1083, 1035)  # (1160, 1083, 1160)  # the output dataset will be cropped/padded to this shape
correlation_threshold = 0.50  # only scans having a correlation larger than this threshold will be combined
reference_scan = 2  # index in scan_list of the scan to be used as the reference for the correlation calculation
combine_masks = False  # if True, the output mask is the combination of all masks. If False, the reference mask is used
is_orthogonal = True  # if True, it will look for the data in a folder named /pynx, otherwise in /pynxraw
plot_threshold = 5  # data below this will be set to 0, only in plots
debug = False  # True or False
##################################
# end of user-defined parameters #
##################################

#########################
# check some parameters #
#########################
if reference_scan is None:
    reference_scan = 0

if type(sample_name) is list:
    if len(sample_name) == 1:
        sample_name = [sample_name[0] for idx in range(len(scan_list))]
    assert len(sample_name) == len(scan_list), 'sample_name and scan_list should have the same length'
elif type(sample_name) is str:
    sample_name = [sample_name for idx in range(len(scan_list))]
else:
    print('sample_name should be either a string or a list of strings')
    sys.exit()
if is_orthogonal:
    parent_folder = '/pynx/'
else:
    parent_folder = '/pynxraw/'

###########################
# load the reference scan #
###########################
plt.ion()
print(scan_list)
samplename = sample_name[reference_scan] + '_' + str('{:05d}').format(scan_list[reference_scan])
print('Reference scan:', samplename)
refdata = np.load(homedir + samplename + parent_folder +
                  'S' + str(scan_list[reference_scan]) + '_pynx' + comment)['data']
refmask = np.load(homedir + samplename + parent_folder +
                  'S' + str(scan_list[reference_scan]) + '_maskpynx' + comment)['mask']
assert refdata.ndim == 3 and refmask.ndim == 3, 'data and mask should be 3D arrays'
nbz, nby, nbx = refdata.shape

if corr_roi is None:
    corr_roi = [0, nbz, 0, nby, 0, nbx]
else:
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
combined_list = []  # list of scans with correlation coeeficient >= threshold
corr_coeff = []  # list of correlation coefficients
sumdata = np.copy(refdata)  # refdata must not be modified
summask = refmask  # refmask is not used elsewhere

for idx in range(len(scan_list)):
    if idx == reference_scan:
        combined_list.append(scan_list[idx])
        corr_coeff.append(1.0)
        continue  # sumdata and summask were already initialized with the reference scan
    samplename = sample_name[idx] + '_' + str('{:05d}').format(scan_list[idx])
    print('\n Opening ', samplename)
    data = np.load(homedir + samplename + parent_folder +
                   'S' + str(scan_list[idx]) + '_pynx' + comment)['data']
    mask = np.load(homedir + samplename + parent_folder +
                   'S' + str(scan_list[idx]) + '_maskpynx' + comment)['mask']

    if debug:
        gu.multislices_plot(data, sum_frames=True, scale='log', plot_colorbar=True,
                            title='S' + str(scan_list[idx]) + '\n Data before shift', vmin=0,
                            reciprocal_space=True, is_orthogonal=is_orthogonal)

        gu.multislices_plot(mask, sum_frames=True, scale='linear', plot_colorbar=True,
                            title='S' + str(scan_list[idx]) + '\n Mask before shift', vmin=0,
                            reciprocal_space=True, is_orthogonal=is_orthogonal)
    ##################
    # align datasets #
    ##################
    if alignement_method is not 'skip':
        data, mask = pru.align_diffpattern(reference_data=refdata, data=data, mask=mask, method=alignement_method,
                                           combining_method=combining_method)
        data[data < 0.5] = 0  # remove interpolated noisy pixels

        if debug:
            gu.multislices_plot(data, sum_frames=True, scale='log', plot_colorbar=True,
                                title='S' + str(scan_list[idx]) + '\n Data after shift', vmin=0,
                                reciprocal_space=True, is_orthogonal=is_orthogonal)

            gu.multislices_plot(mask, sum_frames=True, scale='linear', plot_colorbar=True,
                                title='S' + str(scan_list[idx]) + '\n Mask after shift', vmin=0,
                                reciprocal_space=True, is_orthogonal=is_orthogonal)

    correlation = pearsonr(
        np.ndarray.flatten(abs(refdata[corr_roi[0]:corr_roi[1], corr_roi[2]:corr_roi[3], corr_roi[4]:corr_roi[5]])),
        np.ndarray.flatten(abs(data[corr_roi[0]:corr_roi[1], corr_roi[2]:corr_roi[3], corr_roi[4]:corr_roi[5]])))[0]
    print('Rocking curve ', idx+1, ': Pearson correlation coefficient = ', str('{:.2f}'.format(correlation)))
    corr_coeff.append(round(correlation, 2))

    if correlation >= correlation_threshold:
        combined_list.append(scan_list[idx])
        sumdata = sumdata + data
        if combine_masks:
            summask = summask + mask
    else:
        print('Scan ', scan_list[idx], ', correlation below threshold, skip concatenation')

summask[np.nonzero(summask)] = 1  # mask should be 0 or 1
sumdata = sumdata / len(combined_list)

summask = pu.crop_pad(array=summask, output_shape=output_shape)
sumdata = pu.crop_pad(array=sumdata, output_shape=output_shape)

template = ''.join("_S%s" % ''.join(str(val)) for val in combined_list) +\
           '_{:d}_{:d}_{:d=}'.format(output_shape[0], output_shape[1], output_shape[2])

pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)
np.savez_compressed(savedir+'pynx' + template + '.npz', obj=sumdata)
np.savez_compressed(savedir+'maskpynx' + template + '.npz', obj=summask)
print('\nSum of ', len(combined_list), 'scans')

gu.multislices_plot(sumdata[corr_roi[0]:corr_roi[1], corr_roi[2]:corr_roi[3], corr_roi[4]:corr_roi[5]],
                    sum_frames=True, scale='log', plot_colorbar=True, title='sumdata in corr_roi', vmin=0,
                    reciprocal_space=True, is_orthogonal=is_orthogonal)

sumdata[sumdata < plot_threshold] = 0
fig, _, _ = gu.multislices_plot(sumdata, sum_frames=True, scale='log', plot_colorbar=True, is_orthogonal=is_orthogonal,
                                title='Combined intensity (thresholded)', vmin=0, reciprocal_space=True)
fig.text(0.50, 0.40, "Scans tested:", size=12)
fig.text(0.50, 0.35, str(scan_list), size=12)
fig.text(0.50, 0.30, "Correlation coefficients:", size=12)
fig.text(0.50, 0.25, str(corr_coeff), size=12)
fig.text(0.50, 0.20, "Threshold for correlation: " + str(correlation_threshold), size=12)
fig.text(0.50, 0.15, 'Scans concatenated:', size=12)
fig.text(0.50, 0.10, str(combined_list), size=12)

plt.pause(0.1)
plt.savefig(savedir + 'data' + template + '.png')

gu.multislices_plot(summask, sum_frames=True, scale='linear', plot_colorbar=True,
                    title='Combined mask', vmin=0, reciprocal_space=True, is_orthogonal=is_orthogonal)
plt.savefig(savedir + 'mask' + template + '.png')

print('\nEnd of script')

plt.ioff()
plt.show()
