# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 03:44:26 2018

@author: p10user
"""

import numpy as np
import pathlib
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import image_registration as reg
from scipy.stats import pearsonr
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage.measurements import center_of_mass

a = np.arange(2295,2307+1,3)
b = np.arange(2313,2325+1,3)
scan_list = np.concatenate((a,b),axis=0)
#scan_list = np.arange(1954, 1981+1, 3)  # put the scan numbers here
#scan_list = np.arange(1879, 1951+1, 3)  # put the scan numbers here
#scan_list = np.arange(1849, 1876+1, 3)  # put the scan numbers here
correlation_threshold = 0.985
sample_name = 'dewet2_2_'
suffix = '_norm_119_410_260.npz'

datadir = 'T:/current/processed/analysis/'
aligning_option = 'reg'  # 'com' or 'reg'
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
#################################################################################
filename = sample_name + str('{:05d}').format(scan_list[0])
refdata = np.load(datadir+filename+'/pynxraw/S'+str(scan_list[0])+ '_pynx'+ suffix)['data']
ref_piz, ref_piy, ref_pix = center_of_mass(refdata)
nbz, nby, nbx = refdata.shape
nb_scan = len(scan_list)
sumdata = np.zeros(refdata.shape)
mask = np.load(datadir+filename+'/pynxraw/S'+str(scan_list[0])+ '_maskpynx'+ suffix)['mask']  

if aligning_option is 'com':
    print('\n Use center of mass to align diffraction patterns\n')
else:
    print('\n Use DFT registration to align diffraction patterns\n')

for idx in range(nb_scan):
    filename = sample_name + str('{:05d}').format(scan_list[idx])
    data = np.load(datadir+filename+'/pynxraw/S'+str(scan_list[idx])+ '_pynx'+ suffix)['data']

    if aligning_option is 'com':
        piz, piy, pix = center_of_mass(data)
        offset_z = ref_piz - piz
        offset_y = ref_piy - piy
        offset_x = ref_pix - pix
        print("center of mass offset with reference object: (", str('{:.2f}'.format(offset_z)), ',',
              str('{:.2f}'.format(offset_y)), ',', str('{:.2f}'.format(offset_x)), ') pixels')
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
    else:
        shiftz, shifty, shiftx = reg.getimageregistration(abs(refdata), abs(data), precision=1000)
        data = reg.subpixel_shift(data, shiftz, shifty, shiftx)
        print('\n x shift', shiftx,'y shift', shifty,'z shift', shiftz)
    
    correlation = pearsonr(np.ndarray.flatten(abs(refdata)), np.ndarray.flatten(abs(data)))[0]
    print('\n Rocking curve ', idx+1, ', pearson correlation coefficient = ', str('{:.3f}'.format(correlation)))
    if correlation>= correlation_threshold:
        sumdata = sumdata + data
    
savedir = datadir + sample_name + 'sum_S' + str(scan_list[0]) + '_to_S' + str(scan_list[-1])+'/'
pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)
np.savez_compressed(savedir+'pynx_S'+str(scan_list[0]) + '_to_S' + str(scan_list[-1])+'.npz', obj=sumdata)
np.savez_compressed(savedir+'maskpynx_S'+str(scan_list[0]) + '_to_S' + str(scan_list[-1])+'.npz', obj=mask)


plt.figure(figsize=(18, 15))
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
# plt.pause(0.1)
plt.savefig(savedir + 'sum_S' + str(scan_list[0]) + '_to_S' + str(scan_list[-1])+ '.png')
plt.ioff()
plt.show()