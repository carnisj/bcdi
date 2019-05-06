# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#       authors:
#         Jerome Carnis, jerome.carnis@esrf.fr

import hdf5plugin  # for lz4 filter
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

helptext = """
Open a rocking curve data at PETRAIII P10 beamline and plot the stack along first axis.
Usefull to localize the Bragg peak for ROI determination.
"""

scan = 164
sample_name = "dewet5_"  # "S"
specdir = "C:/Users/carnis/Work Folders/Documents/data/P10_2018/"
homedir = specdir + sample_name + str('{:05d}'.format(scan)) + "/"  # specdir + "S" + str(scan) + '/'
datadir = homedir + "e4m/"  # "data/"
ccdfiletmp = os.path.join(datadir, sample_name + str('{:05d}'.format(scan)) + "_data_%06d.h5")

h5file = h5py.File(ccdfiletmp % 1, 'r')
data = h5file['entry']['data']['data'][:].sum(axis=0)
print(data.shape)
mask = np.zeros((2167, 2070))
mask[:, 1029:1041] = 1
mask[513:552, :] = 1
mask[1064:1103, :] = 1
mask[1614:1654, :] = 1
mask[np.log10(data) > 8] = 1  # look for hot pixels
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
plt.title('sum(data) along the the rocking curve')
plt.colorbar()
plt.show()
