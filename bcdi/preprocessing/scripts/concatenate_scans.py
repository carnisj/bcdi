# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#       authors:
#         Jerome Carnis, jerome.carnis@esrf.fr

import numpy as np
import pathlib
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import sys
sys.path.append('C:\\Users\\carnis\\Work Folders\\Documents\\myscripts\\bcdi\\')
import bcdi.graph.graph_utils as gu
import bcdi.preprocessing.preprocessing_utils as pru

helptext = """
Align diffraction patterns using the center of mass or dft registration and subpixel shift. 

Average the scans if their correlation coefficient is larger than a threshold.

The first scan in the list serves as reference.
"""

scan_list = np.arange(164, 185+1, 3)  # list or array of scan numbers
sample_name = 'dewet5_'
comment = '_norm_180_512_480.npz'  # the end of the filename template after 'pynx'
homedir = "C:/Users/carnis/Work Folders/Documents/data/P10_2018/data/"
method = 'center_of_mass'  # 'center_of_mass' or 'registration',
correlation_threshold = 0.8
debug = False  # True or False
plt.ion()
##################################
# end of user-defined parameters #
##################################

print(scan_list)
filename = sample_name + str('{:05d}').format(scan_list[0])
refdata = np.load(homedir + filename + '/pynxraw/S' + str(scan_list[0]) + '_pynx' + comment)['data']
nbz, nby, nbx = refdata.shape

nb_scan = len(scan_list)
sumdata = np.zeros(refdata.shape)
summask = np.zeros(refdata.shape)
corr_coeff = []  # list of correlation coeeficients
scanlist = []  # list of scans with correlation coeeficient >= threshold

for idx in range(nb_scan):
    filename = sample_name + str('{:05d}').format(scan_list[idx])
    print('\n Opening ', filename)
    data = np.load(homedir + filename + '/pynxraw/S'+str(scan_list[idx]) + '_pynx' + comment)['data']
    mask = np.load(homedir + filename + '/pynxraw/S'+str(scan_list[idx]) + '_maskpynx' + comment)['mask']

    if debug:
        gu.multislices_plot(data, sum_frames=True, invert_yaxis=False, scale='log', plot_colorbar=True,
                            title='S' + str(scan_list[idx]) + '\n Data before shift', vmin=0,
                            reciprocal_space=True)

        gu.multislices_plot(mask, sum_frames=True, invert_yaxis=False, scale='linear', plot_colorbar=True,
                            title='S' + str(scan_list[idx]) + '\n Mask before shift', vmin=0,
                            reciprocal_space=True)

    data, mask = pru.align_diffpattern(reference_data=refdata, data=data, mask=mask, method=method)

    if debug:
        gu.multislices_plot(data, sum_frames=True, invert_yaxis=False, scale='log', plot_colorbar=True,
                            title='S' + str(scan_list[idx]) + '\n Data after shift', vmin=0,
                            reciprocal_space=True)

        gu.multislices_plot(mask, sum_frames=True, invert_yaxis=False, scale='linear', plot_colorbar=True,
                            title='S' + str(scan_list[idx]) + '\n Mask after shift', vmin=0,
                            reciprocal_space=True)

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
sumdata = np.rint(sumdata / len(scanlist))  # back to count in photons

savedir = homedir + sample_name + 'sum_S' + str(scanlist[0]) + '_to_S' + str(scanlist[-1])+'/'
pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)
np.savez_compressed(savedir+'pynx_S'+str(scanlist[0]) + '_to_S' + str(scanlist[-1])+'.npz', obj=sumdata)
np.savez_compressed(savedir+'maskpynx_S'+str(scanlist[0]) + '_to_S' + str(scanlist[-1])+'.npz', obj=summask)
print('Sum of ', len(corr_coeff), 'scans')

fig, _, _ = gu.multislices_plot(sumdata, sum_frames=True, invert_yaxis=False, scale='log', plot_colorbar=True,
                                title='sum(intensity)', vmin=0, reciprocal_space=True)
fig.text(0.50, 0.40, "Scans tested: " + str(scan_list), size=14)
fig.text(0.50, 0.35, 'Scans concatenated: ' + str(scanlist), size=14)
fig.text(0.50, 0.30, "Correlation coefficients: " + str(corr_coeff), size=14)
fig.text(0.50, 0.25, "Threshold for correlation: " + str(correlation_threshold), size=14)
plt.pause(0.1)
plt.savefig(savedir + 'sum_S' + str(scan_list[0]) + '_to_S' + str(scan_list[-1]) + '.png')

gu.multislices_plot(summask, sum_frames=True, invert_yaxis=False, scale='linear', plot_colorbar=True,
                    title='sum(mask)', vmin=0, reciprocal_space=True)
plt.savefig(savedir + 'sum_mask_S' + str(scan_list[0]) + '_to_S' + str(scan_list[-1]) + '.png')
plt.ioff()
plt.show()
