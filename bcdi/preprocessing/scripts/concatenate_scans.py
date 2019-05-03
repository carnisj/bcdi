# -*- coding: utf-8 -*-
"""
Align diffraction patterns and calculate their correlation.
The first scan in the list serves as reference.

Created on Fri Nov 30 03:44:26 2018
@author: Jerome Carnis @ ESRF ID01
"""

import numpy as np
import pathlib
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import sys
sys.path.append('C:/Users/carnis/Work Folders/Documents/myscripts/preprocessing_cdi/')
import image_registration as reg
from scipy.stats import pearsonr
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage.measurements import center_of_mass

scan_list = np.arange(282, 294+1, 3)  # list or array of scan numbers
sample_name = 'dewet5_'
comment = '_norm_180_512_480.npz'
specdir = "C:/Users/carnis/Work Folders/Documents/data/P10_2018/data/"
aligning_option = 'com'  # 'com' or 'reg', 'com' is better because it does not introduce artifacts
correlation_threshold = 0.8
debug = False  # True or False
##############################################################################
# parameters for plotting)
params = {'backend': 'Qt5Agg',
          'axes.labelsize': 20,
          'font.size': 20,
          'legend.fontsize': 20,
          'axes.titlesize': 20,
          'xtick.labelsize': 20,
          'ytick.labelsize': 20,
          'text.usetex': False,
          'figure.figsize': (11, 9)}
matplotlib.rcParams.update(params)
# define a colormap
cdict = {'red':  ((0.0, 1.0, 1.0),
                  (0.11, 0.0, 0.0),
                  (0.36, 0.0, 0.0),
                  (0.62, 1.0, 1.0),
                  (0.87, 1.0, 1.0),
                  (1.0, 0.0, 0.0)),
         'green': ((0.0, 1.0, 1.0),
                   (0.11, 0.0, 0.0),
                   (0.36, 1.0, 1.0),
                   (0.62, 1.0, 1.0),
                   (0.87, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),
         'blue': ((0.0, 1.0, 1.0),
                  (0.11, 1.0, 1.0),
                  (0.36, 1.0, 1.0),
                  (0.62, 0.0, 0.0),
                  (0.87, 0.0, 0.0),
                  (1.0, 0.0, 0.0))}
my_cmap = LinearSegmentedColormap('my_colormap', cdict, 256)
plot_title = ['YZ', 'XZ', 'XY']
plt.ion()
#################################################################################
print(scan_list)
filename = sample_name + str('{:05d}').format(scan_list[0])
refdata = np.load(specdir + filename + '/pynxraw/S' + str(scan_list[0]) + '_pynx' + comment)['data']
ref_piz, ref_piy, ref_pix = center_of_mass(refdata)
nbz, nby, nbx = refdata.shape
nb_scan = len(scan_list)
sumdata = np.zeros(refdata.shape)
summask = np.zeros(refdata.shape)
corr_coeff = []  # list of correlation coeeficients
scanlist = []  # list of scans with correlation coeeficient >= threshold
for idx in range(nb_scan):
    filename = sample_name + str('{:05d}').format(scan_list[idx])
    data = np.load(specdir + filename + '/pynxraw/S'+str(scan_list[idx]) + '_pynx' + comment)['data']
    mask = np.load(specdir + filename + '/pynxraw/S'+str(scan_list[idx]) + '_maskpynx' + comment)['mask']
    if debug:
        plt.figure(figsize=(18, 15))
        plt.subplot(2, 2, 1)
        plt.imshow(np.log10(abs(data.sum(axis=2))), cmap=my_cmap, vmin=0)
        plt.colorbar()
        plt.axis('scaled')
        plt.title('S' + str(scan_list[idx]) + '\nBefore shifting in ' + plot_title[0])
        plt.subplot(2, 2, 2)
        plt.imshow(np.log10(abs(data.sum(axis=1))), cmap=my_cmap, vmin=0)
        plt.colorbar()
        plt.axis('scaled')
        plt.title('S' + str(scan_list[idx]) + '\nBefore shifting in ' + plot_title[1])
        plt.subplot(2, 2, 3)
        plt.imshow(np.log10(abs(data.sum(axis=0))), cmap=my_cmap, vmin=0)
        plt.colorbar()
        plt.axis('scaled')
        plt.title('S' + str(scan_list[idx]) + '\nBefore shifting in ' + plot_title[2])
        plt.pause(0.1)

        plt.figure(figsize=(18, 15))
        plt.subplot(2, 2, 1)
        plt.imshow(mask.sum(axis=2), cmap=my_cmap, vmin=0)
        plt.colorbar()
        plt.axis('scaled')
        plt.title('S' + str(scan_list[idx]) + '\nBefore shifting in ' + plot_title[0])
        plt.subplot(2, 2, 2)
        plt.imshow(mask.sum(axis=1), cmap=my_cmap, vmin=0)
        plt.colorbar()
        plt.axis('scaled')
        plt.title('S' + str(scan_list[idx]) + '\nBefore shifting in ' + plot_title[1])
        plt.subplot(2, 2, 3)
        plt.imshow(mask.sum(axis=0), cmap=my_cmap, vmin=0)
        plt.colorbar()
        plt.axis('scaled')
        plt.title('S' + str(scan_list[idx]) + '\nBefore shifting in ' + plot_title[2])
        plt.pause(0.1)

    if aligning_option is 'com':
        piz, piy, pix = center_of_mass(data)
        offset_z = ref_piz - piz
        offset_y = ref_piy - piy
        offset_x = ref_pix - pix
        print('\nRocking curve ', idx+1, ': x shift', str('{:.2f}'.format(offset_x)),  ', y shift',
              str('{:.2f}'.format(offset_y)),  ', z shift', str('{:.2f}'.format(offset_z)))
        # re-sample data on a new grid based on COM shift of support
        old_z = np.arange(-nbz // 2, nbz // 2)
        old_y = np.arange(-nby // 2, nby // 2)
        old_x = np.arange(-nbx // 2, nbx // 2)
        myz, myy, myx = np.meshgrid(old_z, old_y, old_x, indexing='ij')
        new_z = myz + offset_z
        new_y = myy + offset_y
        new_x = myx + offset_x
        del myx, myy, myz
        rgi = RegularGridInterpolator((old_z, old_y, old_x), data, method='linear', bounds_error=False,
                                      fill_value=0)
        data = rgi(np.concatenate((new_z.reshape((1, new_z.size)), new_y.reshape((1, new_z.size)),
                                   new_x.reshape((1, new_z.size)))).transpose())
        data = data.reshape((nbz, nby, nbx)).astype(refdata.dtype)
        rgi = RegularGridInterpolator((old_z, old_y, old_x), mask, method='linear', bounds_error=False,
                                      fill_value=0)
        mask = rgi(np.concatenate((new_z.reshape((1, new_z.size)), new_y.reshape((1, new_z.size)),
                                   new_x.reshape((1, new_z.size)))).transpose())
        mask = mask.reshape((nbz, nby, nbx)).astype(refdata.dtype)
        mask = np.rint(mask)  # mask is integer 0 or 1
    else:
        shiftz, shifty, shiftx = reg.getimageregistration(abs(refdata), abs(data), precision=1000)
        print('\nRocking curve ', idx+1, ': x shift', shiftx, ', y shift', shifty, ', z shift', shiftz)
        data = abs(reg.subpixel_shift(data, shiftz, shifty, shiftx))  # data is a real number
        mask = np.rint(abs(reg.subpixel_shift(mask, shiftz, shifty, shiftx)))  # mask is integer 0 or 1
    if debug:
        plt.figure(figsize=(18, 15))
        plt.subplot(2, 2, 1)
        plt.imshow(np.log10(abs(data.sum(axis=2))), cmap=my_cmap, vmin=0)
        plt.colorbar()
        plt.axis('scaled')
        plt.title('S' + str(scan_list[idx]) + '\nAfter shifting in ' + plot_title[0])
        plt.subplot(2, 2, 2)
        plt.imshow(np.log10(abs(data.sum(axis=1))), cmap=my_cmap, vmin=0)
        plt.colorbar()
        plt.axis('scaled')
        plt.title('S' + str(scan_list[idx]) + '\nAfter shifting in ' + plot_title[1])
        plt.subplot(2, 2, 3)
        plt.imshow(np.log10(abs(data.sum(axis=0))), cmap=my_cmap, vmin=0)
        plt.colorbar()
        plt.axis('scaled')
        plt.title('S' + str(scan_list[idx]) + '\nAfter shifting in ' + plot_title[2])
        plt.pause(0.1)

        plt.figure(figsize=(18, 15))
        plt.subplot(2, 2, 1)
        plt.imshow(mask.sum(axis=2), cmap=my_cmap, vmin=0)
        plt.colorbar()
        plt.axis('scaled')
        plt.title('S' + str(scan_list[idx]) + '\nAfter shifting in ' + plot_title[0])
        plt.subplot(2, 2, 2)
        plt.imshow(mask.sum(axis=1), cmap=my_cmap, vmin=0)
        plt.colorbar()
        plt.axis('scaled')
        plt.title('S' + str(scan_list[idx]) + '\nAfter shifting in ' + plot_title[1])
        plt.subplot(2, 2, 3)
        plt.imshow(mask.sum(axis=0), cmap=my_cmap, vmin=0)
        plt.colorbar()
        plt.axis('scaled')
        plt.title('S' + str(scan_list[idx]) + '\nAfter shifting in ' + plot_title[2])
        plt.pause(0.1)

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

savedir = specdir + sample_name + 'sum_S' + str(scanlist[0]) + '_to_S' + str(scanlist[-1])+'/'
pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)
np.savez_compressed(savedir+'pynx_S'+str(scanlist[0]) + '_to_S' + str(scanlist[-1])+'.npz', obj=sumdata)
np.savez_compressed(savedir+'maskpynx_S'+str(scanlist[0]) + '_to_S' + str(scanlist[-1])+'.npz', obj=summask)
print('Sum of ', len(corr_coeff), 'scans')

fig = plt.figure(figsize=(18, 15))
plt.subplot(2, 2, 1)
plt.imshow(np.log10(abs(sumdata.sum(axis=2))), cmap=my_cmap, vmin=0)
plt.colorbar()
plt.axis('scaled')
plt.title('sum(intensity)\n in ' + plot_title[0])
plt.subplot(2, 2, 2)
plt.imshow(np.log10(abs(sumdata.sum(axis=1))), cmap=my_cmap, vmin=0)
plt.colorbar()
plt.axis('scaled')
plt.title('sum(intensity)\n in ' + plot_title[1])
plt.subplot(2, 2, 3)
plt.imshow(np.log10(abs(sumdata.sum(axis=0))), cmap=my_cmap, vmin=0)
plt.colorbar()
plt.axis('scaled')
plt.title('sum(intensity)\n in ' + plot_title[2])
fig.text(0.50, 0.40, "Scans tested: " + str(scan_list), size=14)
fig.text(0.50, 0.35, 'Scans concatenated: ' + str(scanlist), size=14)
fig.text(0.50, 0.30, "Correlation coefficients: " + str(corr_coeff), size=14)
fig.text(0.50, 0.25, "Threshold for correlation: " + str(correlation_threshold), size=14)
# plt.pause(0.1)
plt.savefig(savedir + 'sum_S' + str(scan_list[0]) + '_to_S' + str(scan_list[-1]) + '.png')

plt.figure(figsize=(18, 15))
plt.subplot(2, 2, 1)
plt.imshow(summask.sum(axis=2), cmap=my_cmap, vmin=0)
plt.colorbar()
plt.axis('scaled')
plt.title('sum(mask)\n in ' + plot_title[0])
plt.subplot(2, 2, 2)
plt.imshow(summask.sum(axis=1), cmap=my_cmap, vmin=0)
plt.colorbar()
plt.axis('scaled')
plt.title('sum(mask)\n in ' + plot_title[1])
plt.subplot(2, 2, 3)
plt.imshow(summask.sum(axis=0), cmap=my_cmap, vmin=0)
plt.colorbar()
plt.axis('scaled')
plt.title('sum(mask)\n in ' + plot_title[2])
plt.savefig(savedir + 'sum_mask_S' + str(scan_list[0]) + '_to_S' + str(scan_list[-1]) + '.png')
plt.ioff()
plt.show()