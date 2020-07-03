# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

helptext = """
Crop a stacked 3D dataset saved in NPZ format, to the desired region of interest.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('D:/myscripts/bcdi/')
import bcdi.postprocessing.postprocessing_utils as pu

homedir = "D:/data/P10_isosurface/data/p21_00038/pynxraw/"  # parent folder of scans folders
datadir = homedir  # + 'pynxraw/'
roi_center = [80, 278, 280]  # center of the region of interest
output_shape = [160, 512, 256]  # shape of the output file
load_mask = True  # True to load the mask and crop it
comment = '_1_2_2'  # should start with _
################################

npzfile = np.load(datadir + 'combined_pynx_S38_S41_160_512_256_1_2_2.npz')
try:
    data = npzfile['data']
except KeyError:
    print('Wrong key, available keys:', list(npzfile.keys()))
    sys.exit()

data = pu.crop_pad(data, output_shape=output_shape, crop_center=roi_center, debugging=True)
comment = str(output_shape[0]) + '_' + str(output_shape[1]) + '_' + str(output_shape[2]) + comment + '.npz'
np.savez_compressed(datadir + 'combined_pynx_' + comment, data=data)

if load_mask:
    npzfile = np.load(datadir + 'combined_maskpynx_S38_S41_160_512_256_1_2_2.npz')
    try:
        mask = npzfile['mask']
    except KeyError:
        print('Wrong key, available keys:', list(npzfile.keys()))
        sys.exit()
    mask = pu.crop_pad(mask, output_shape=output_shape, crop_center=roi_center, debugging=False)
    np.savez_compressed(datadir + 'combined_maskpynx_' + comment, mask=mask)

print('End of script')
plt.ioff()
plt.show()
