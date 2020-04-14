# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

helptext = """
Open images or series data at P10 beamline.
"""

import hdf5plugin  # for lz4 filter
import h5py
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import os

sample_name = "align_06"  # without _ at the end
scan = 123  # scan number as it appears in the folder name
image_nb = 1  # last number in the filename, e.g. 1 for gold_2_2_2_00022_data_000001.h5
rootdir = "D:/data/P10_March2020_CDI/test_april/data/"
savedir = ''  # images will be saved here, leave it to '' otherwise (default to data directory's parent)
series = True  # set to True if the measurement is a time series, False for a single image
save_mask = False  # True to save the mask as 'hotpixels.npz'

if series:
    datadir = rootdir + sample_name + '_' + str('{:05d}'.format(scan)) + '/e4m/'
    ccdfiletmp = os.path.join(datadir, sample_name + '_' + str('{:05d}'.format(scan)) + 
                              "_data_" + str('{:06d}'.format(image_nb))+".h5")
else:
    datadir = rootdir + sample_name + '/e4m/'
    ccdfiletmp = os.path.join(datadir, sample_name + '_take_' + str('{:05d}'.format(scan)) + 
                              "_data_" + str('{:06d}'.format(image_nb))+".h5")
if savedir == '':
    savedir = os.path.abspath(os.path.join(datadir, os.pardir))

h5file = h5py.File(ccdfiletmp, 'r')
data = h5file['entry']['data']['data'][:]
if series:
    nbz, nby, nbx = data.shape
    plot_title = 'masked data - sum of ' + str(nbz) + ' frames'
    filename = '/S' + str(scan) + '_series.png'
else:
    plot_title = 'masked data'
    filename = '/image_' + str(image_nb) + '.png'

print(data.shape)
data = data.sum(axis=0)
# data = data[109, :, :]
mask = np.zeros((2167, 2070))
mask[:, 1029:1041] = 1
mask[513:552, :] = 1
mask[1064:1103, :] = 1
mask[1614:1654, :] = 1

mask[np.log10(data) > 9] = 1  # look for hot pixels
data[mask == 1] = 0

if save_mask:
    plt.figure()
    plt.imshow(mask)
    plt.title('mask')
    plt.colorbar()
    np.savez_compressed(savedir+'hotpixels.npz', mask=mask)
    plt.savefig(savedir + '/mask.png')

y0, x0 = np.unravel_index(abs(data).argmax(), data.shape)
print("Max at (y, x): ", y0, x0, ' Max = ', int(data[y0, x0]))

plt.figure()
plt.imshow(np.log10(data), vmin=0)
plt.title(plot_title)
plt.colorbar()
plt.savefig(savedir + filename)
plt.show()
