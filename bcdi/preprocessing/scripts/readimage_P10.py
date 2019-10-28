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

sample_name = "gold_2_2_2"  # without _ at the end
scan = 22  # scan number as it appears in the folder name
image = 2  # last number in the filename, e.g. 1 for gold_2_2_2_00022_data_000001.h5
rootdir = "D:/data/P10_August2019/data/"
series = True  # set to True if the measurement is a time series, False for a single image
save_mask = True  # True to save the mask as 'hotpixels.npz'

if series: 
    datadir = rootdir + sample_name + '_' + str('{:05d}'.format(scan)) + '/e4m/'
    ccdfiletmp = os.path.join(datadir, sample_name + '_' + str('{:05d}'.format(scan)) + 
                              "_data_" + str('{:06d}'.format(image))+".h5")
else:
    datadir = rootdir + sample_name + '/e4m/'
    ccdfiletmp = os.path.join(datadir, sample_name + '_take_' + str('{:05d}'.format(scan)) + 
                              "_data_" + str('{:06d}'.format(image))+".h5")

h5file = h5py.File(ccdfiletmp, 'r')
data = h5file['entry']['data']['data'][:]
print(data.shape)
data = data.sum(axis=0)
mask = np.zeros((2167, 2070))
mask[:, 1029:1041] = 1
mask[513:552, :] = 1
mask[1064:1103, :] = 1
mask[1614:1654, :] = 1

mask[1345, 1319:1325] = 1
mask[1346, 1318:1326] = 1
mask[1347, 1317:1327] = 1
mask[1348, 1317:1327] = 1
mask[1349, 1317:1327] = 1
mask[1350, 1317:1327] = 1
mask[1351, 1318:1324] = 1

mask[np.log10(data) > 7] = 1  # look for hot pixels
data[mask == 1] = 0
plt.figure()
plt.imshow(mask)
plt.title('mask')
plt.colorbar()

if save_mask:
    np.savez_compressed(rootdir+'hotpixels.npz', mask=mask)
    plt.savefig(rootdir + 'mask.png')

y0, x0 = np.unravel_index(abs(data).argmax(), data.shape)
print("Max at (y, x): ", y0, x0, ' Max = ', int(data[y0, x0]))

plt.figure()
plt.imshow(np.log10(data), vmin=0)
plt.title('sum(masked data)')
plt.colorbar()

plt.show()
