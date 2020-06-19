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

scan_list = [22, 32, 48, 55, 59, 71, 6, 15, 37,45]  # np.arange(404, 407+1, 3)  # list or array of scan numbers
sample_name = ['ht_pillar3', 'ht_pillar3', 'ht_pillar3', 'ht_pillar3', 'ht_pillar3', 'ht_pillar3',
               'ht_pillar3_1', 'ht_pillar3_1', 'ht_pillar3_1', 'ht_pillar3_1']
comment = '_ortho_norm_1134_1083_1134_2_2_2.npz'  # the end of the filename template after 'pynx'
homedir = "/nfs/fs/fscxi/experiments/2020/PETRA/P10/11008562/raw/"  # parent folder of scans folders
savedir = '/home/carnisj/phasing/'  # path of the folder to save data
alignement_method = 'skip'  # method to find the translational offset, 'skip', 'center_of_mass' or 'registration'
combining_method = 'subpixel'  # 'rgi' for RegularGridInterpolator or 'subpixel' for subpixel shift
output_shape = (140, 512, 350)  # the output dataset will be cropped/padded to this shape
correlation_threshold = 0.95  # only scans having a correlation larger than this threshold will be combined
reference_scan = None  # index in scan_list of the scan to be used as the reference for the correlation calculation
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
    assert len(sample_name) == len(scan_list), 'sample_name and scan_list should have the same length'
elif type(sample_name) is str:
    sample_name = [sample_name for idx in range(len(scan_list))]
else:
    print('sample_name should be either a string or a list of strings')
    sys.exit()

###########################
# load the reference scan #
###########################
plt.ion()
print(scan_list)
filename = sample_name[reference_scan] + str('{:05d}').format(scan_list[reference_scan])
print('Reference scan:', filename)
refdata = np.load(homedir + filename + '/pynxraw/S' + str(scan_list[reference_scan]) + '_pynx' + comment)['data']
nbz, nby, nbx = refdata.shape

###########################
# combine the other scans #
###########################
nb_scan = len(scan_list)
sumdata = np.zeros(refdata.shape)
summask = np.zeros(refdata.shape)
corr_coeff = []  # list of correlation coeeficients
scanlist = []  # list of scans with correlation coeeficient >= threshold

for idx in range(nb_scan):
    filename = sample_name[idx] + str('{:05d}').format(scan_list[idx])
    print('\n Opening ', filename)
    data = np.load(homedir + filename + '/pynxraw/S'+str(scan_list[idx]) + '_pynx' + comment)['data']
    mask = np.load(homedir + filename + '/pynxraw/S'+str(scan_list[idx]) + '_maskpynx' + comment)['mask']

    if debug:
        gu.multislices_plot(data, sum_frames=True, scale='log', plot_colorbar=True,
                            title='S' + str(scan_list[idx]) + '\n Data before shift', vmin=0,
                            reciprocal_space=True, is_orthogonal=False)

        gu.multislices_plot(mask, sum_frames=True, scale='linear', plot_colorbar=True,
                            title='S' + str(scan_list[idx]) + '\n Mask before shift', vmin=0,
                            reciprocal_space=True, is_orthogonal=False)
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
                                reciprocal_space=True, is_orthogonal=False)

            gu.multislices_plot(mask, sum_frames=True, scale='linear', plot_colorbar=True,
                                title='S' + str(scan_list[idx]) + '\n Mask after shift', vmin=0,
                                reciprocal_space=True, is_orthogonal=False)

    correlation = pearsonr(np.ndarray.flatten(abs(refdata)), np.ndarray.flatten(abs(data)))[0]
    print('Rocking curve ', idx+1, ': Pearson correlation coefficient = ', str('{:.2f}'.format(correlation)))
    corr_coeff.append(str('{:.2f}'.format(correlation)))

    if correlation >= correlation_threshold:
        scanlist.append(scan_list[idx])
        sumdata = sumdata + data
        summask = summask + mask
    else:
        print('Scan ', scan_list[idx], ', correlation below threshold, skip concatenation')

summask[np.nonzero(summask)] = 1  # mask should be 0 or 1
sumdata = sumdata / len(scanlist)

summask = pu.crop_pad(array=summask, output_shape=output_shape)
sumdata = pu.crop_pad(array=sumdata, output_shape=output_shape)

template = '_S'+str(scanlist[0]) + '_to_S' + str(scanlist[-1]) + '_' +\
           str(output_shape[0]) + '_' + str(output_shape[1]) + '_' + str(output_shape[2]) + '.npz'

pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)
np.savez_compressed(savedir+'pynx' + template, obj=sumdata)
np.savez_compressed(savedir+'maskpynx' + template, obj=summask)
print('Sum of ', len(scanlist), 'scans')

fig, _, _ = gu.multislices_plot(sumdata, sum_frames=True, scale='log', plot_colorbar=True,
                                title='sum(intensity)', vmin=0, reciprocal_space=True, is_orthogonal=False)
fig.text(0.50, 0.40, "Scans tested: " + str(scan_list), size=14)
fig.text(0.50, 0.35, 'Scans concatenated: ' + str(scanlist), size=14)
fig.text(0.50, 0.30, "Correlation coefficients: " + str(corr_coeff), size=14)
fig.text(0.50, 0.25, "Threshold for correlation: " + str(correlation_threshold), size=14)
plt.pause(0.1)
plt.savefig(savedir + 'sum_S' + str(scan_list[0]) + '_to_S' + str(scan_list[-1]) + '.png')

gu.multislices_plot(summask, sum_frames=True, scale='linear', plot_colorbar=True,
                    title='sum(mask)', vmin=0, reciprocal_space=True, is_orthogonal=False)
plt.savefig(savedir + 'sum_mask_S' + str(scan_list[0]) + '_to_S' + str(scan_list[-1]) + '.png')
plt.ioff()
plt.show()
