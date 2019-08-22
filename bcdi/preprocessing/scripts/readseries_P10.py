# -*- coding: utf-8 -*-
"""
open data at P10 beamline
"""
import hdf5plugin  # for lz4 filter
import h5py
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import os

scan = 8
image = 1
sample_name = "magnetite_A2"
datadir = "D:/data/P10_August2019/data/" + sample_name + '_' + str('{:05d}'.format(scan)) + '/e4m/'
ccdfiletmp = os.path.join(datadir, sample_name + '_' + str('{:05d}'.format(scan)) + 
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
mask[np.log10(data) > 7] = 1  # look for hot pixels
data[mask == 1] = 0
# np.savez_compressed(datadir+'hotpixels.npz', mask=mask)
plt.figure()
plt.imshow(mask)
plt.title('mask')
plt.colorbar()

y0, x0 = np.unravel_index(abs(data).argmax(), data.shape)
print("Max at (y, x): ", y0, x0, ' Max = ', int(data[y0, x0]))

plt.figure()
plt.imshow(np.log10(data), vmin=0)
plt.title('sum(data)')
plt.colorbar()

plt.show()
