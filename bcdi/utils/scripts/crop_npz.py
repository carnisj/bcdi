# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np
import gc
import matplotlib.pyplot as plt
import sys
import tkinter as tk
from tkinter import filedialog
sys.path.append('D:/myscripts/bcdi/')
import bcdi.utils.utilities as util
import bcdi.graph.graph_utils as gu
import bcdi.postprocessing.postprocessing_utils as pu

helptext = """
Crop a stacked 3D dataset saved in NPZ format, to the desired region of interest.
"""

scan = 9999  # scan number, used in the filename when saving
homedir = "/nfs/fs/fscxi/experiments/2020/PETRA/P10/11009357/raw/mag_3_concat/"
# '/nfs/fs/fscxi/experiments/2020/PETRA/P10/11008562/raw/ht_pillar3_{:05d}'.format(scan) + '/pynx/'
# parent folder of scans folders
datadir = homedir  #  + 'ht_pillar3_combined/'
crop_center = [580, 456, 580]  # center of the region of interest
output_shape = [900, 972, 768]  # shape of the output file
load_mask = True  # True to load the mask and crop it
load_qvalues = True  # True to load the q values and crop it
is_orthogonal = True  # True if the data is in an orthogonal frame, only used for plots
reciprocal_space = True  # True if the data is in reciprocal space, only used for plots
debug = True  # True to see more plots
comment = '_combined'  # should start with _
##################################
# end of user-defined parameters #
##################################

###################
# define colormap #
###################
colormap = gu.Colormap()
my_cmap = colormap.cmap

#################
# load the data #
#################
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(initialdir=datadir, title="Select the data file",
                                       filetypes=[("NPZ", "*.npz"), ("CXI", "*.cxi"), ("HDF5", "*.h5")])
data, _ = util.load_file(file_path)
nbz, nby, nbx = data.shape

#################################################################
# check parameters depending on the shape of the reference scan #
#################################################################
crop_center = list(crop_center or [nbz // 2, nby // 2, nbx // 2])  # if None, default to the middle of the array
assert len(crop_center) == 3, 'crop_center should be a list or tuple of three indices'
assert np.all(np.asarray(crop_center)-np.asarray(output_shape)//2 >= 0), 'crop_center incompatible with output_shape'
assert crop_center[0]+output_shape[0]//2 <= nbz and crop_center[1]+output_shape[1]//2 <= nby\
        and crop_center[2]+output_shape[2]//2 <= nbx, 'crop_center incompatible with output_shape'

#######################################################
# crop the data, and optionally the mask and q values #
#######################################################
data = pu.crop_pad(data, output_shape=output_shape, crop_center=crop_center, debugging=debug)
comment = str(output_shape[0]) + '_' + str(output_shape[1]) + '_' + str(output_shape[2]) + comment
np.savez_compressed(datadir + 'S' + str(scan) + '_pynx_cropped_' + comment + '.npz', data=data)

fig, _, _ = gu.multislices_plot(data, sum_frames=True, scale='log', plot_colorbar=True, vmin=0,
                                title='Cropped data', is_orthogonal=is_orthogonal,
                                reciprocal_space=reciprocal_space)
fig.savefig(datadir + 'S' + str(scan) + '_pynx_cropped_' + comment + '.png')
del data
gc.collect()

if load_mask:
    file_path = filedialog.askopenfilename(initialdir=datadir, title="Select the mask file",
                                           filetypes=[("NPZ", "*.npz"), ("CXI", "*.cxi"), ("HDF5", "*.h5")])
    mask, _ = util.load_file(file_path)
    mask = pu.crop_pad(mask, output_shape=output_shape, crop_center=crop_center, debugging=False)
    np.savez_compressed(datadir + 'S' + str(scan) + '_maskpynx_cropped_' + comment + '.npz', mask=mask)
    fig, _, _ = gu.multislices_plot(mask, sum_frames=True, scale='linear', plot_colorbar=True, vmin=0,
                                    title='Cropped mask', is_orthogonal=is_orthogonal,
                                    reciprocal_space=reciprocal_space)
    fig.savefig(datadir + 'S' + str(scan) + '_maskpynx_cropped_' + comment + '.png')
    del mask
    gc.collect()
    
if load_qvalues:
    file_path = filedialog.askopenfilename(initialdir=datadir, title="Select the file containing q values",
                                           filetypes=[("NPZ", "*.npz")])
    q_values = np.load(file_path)
    qx = q_values['qx']  # 1D array
    qy = q_values['qy']  # 1D array
    qz = q_values['qz']  # 1D array
    qx = pu.crop_pad_1d(qx, output_shape[0], crop_center=crop_center[0])  # qx along z
    qy = pu.crop_pad_1d(qy, output_shape[2], crop_center=crop_center[2])  # qy along x
    qz = pu.crop_pad_1d(qz, output_shape[1], crop_center=crop_center[1])  # qz along y
    np.savez_compressed(datadir + 'S' + str(scan) + '_cropped_qvalues_' + comment + '.npz', qx=qx, qz=qz, qy=qy)

print('End of script')
plt.ioff()
plt.show()
