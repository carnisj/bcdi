# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np
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

homedir = "/nfs/fs/fscxi/experiments/2020/PETRA/P10/11008562/raw/"  # parent folder of scans folders
datadir = homedir  + 'ht_pillar3_combined/pynxraw/'
roi_center = [675, 470, 296]  # center of the region of interest
output_shape = [100, 100, 100]  # shape of the output file
load_mask = True  # True to load the mask and crop it
is_orthogonal = True  # True if the data is in an orthogonal frame, only used for plots
reciprocal_space = True  # True if the data is in reciprocal space, only used for plots
debug = False  # True to see more plots
comment = '_2_2_2'  # should start with _
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
                                       filetypes=[("NPZ", "*.npz"), ("CXI", "*.cxi"), ("HDF5", "*.h5") ])
data, _ = util.load_file(file_path)

data = pu.crop_pad(data, output_shape=output_shape, crop_center=roi_center, debugging=debug)
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

fig, _, _ = gu.multislices_plot(data, sum_frames=True, scale='log', plot_colorbar=True, vmin=0,
                                title='Cropped data', is_orthogonal=is_orthogonal,
                                reciprocal_space=reciprocal_space)

print('End of script')
plt.ioff()
plt.show()
