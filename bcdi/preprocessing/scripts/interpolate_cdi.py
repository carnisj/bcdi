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
import bcdi.graph.graph_utils as gu
import bcdi.utils.utilities as util
import bcdi.utils.validation as valid

helptext = """
This script can be used to interpolate the intensity of masked voxels suing the centrosymmetry property of the
diffraction part in the forward CDI geometry. The diffraction pattern should be in an orthonormal frame with identical
voxel sizes in all directions.
"""

datadir = ''  # location of the data and mask
savedir = None  # path where to save the result, will default to datadir if None
origin = (12, 23, 15)  # tuple of three integers, position in pixels of the origin of reciprocal space
##################################
# end of user-defined parameters #
##################################

###################
# define colormap #
###################
colormap = gu.Colormap()
my_cmap = colormap.cmap

###################################
# load experimental data and mask #
###################################
plt.ion()
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(initialdir=datadir, title="Select the diffraction pattern",
                                       filetypes=[("NPZ", "*.npz"), ("NPY", "*.npy")])
data, _ = util.load_file(file_path)
data = data.astype(float)

file_path = filedialog.askopenfilename(initialdir=datadir, title="Select the mask",
                                       filetypes=[("NPZ", "*.npz"), ("NPY", "*.npy")])
mask, _ = util.load_file(file_path)

#########################
# check some parameters #
#########################
savedir = savedir or datadir

nbz, nby, nbx = data.shape

if data.shape != mask.shape:
    raise ValueError(f'Incompatible shape for the data: {data.shape} and the mask: {mask.shape}')

if data.ndim != 3:
    raise ValueError('only 3D data is supported')

valid.valid_container(obj=origin, container_types=(tuple, list, np.ndarray), item_types=int, length=3,
                      name='interpolate_cdi.py')

####################################################################################################################
# loop over masked points to see if the centrosymmetric voxel is also masked, if not copy its intensity and unmask #
####################################################################################################################
ind_z, ind_y, ind_x = np.nonzero(mask)

for idx in range(len(ind_z)):
    # calculate the position of the centrosymmetric voxel
    