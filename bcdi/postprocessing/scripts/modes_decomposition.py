# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend("Qt5Agg")  # "Qt5Agg" or "Qt4Agg" depending on the version of Qt installer, bug with Tk
import os
import sys
import tkinter as tk
from tkinter import filedialog
sys.path.append('D:/myscripts/bcdi/')
import bcdi.graph.graph_utils as gu
import bcdi.utils.utilities as util
import bcdi.postprocessing.postprocessing_utils as pu

helptext = """
Eigendecomposition of a set of 3D reconstructed objects from phase retrieval,
ideally the first mode should be as high as possible. Adapted from PyNX.
"""

datadir = "D:/data/P10_August2019/data/gold_2_2_2_00022/pynx/1000_1000_1000_1_1_1/maximum_likelihood/good"
user_comment = '_test'  # string, should start with "_"
nb_mode = 2  # number of modes to return in the mode array (starting from 0)
################
# Load objects #
################
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilenames(initialdir=datadir,
                                        filetypes=[("CXI", "*.cxi"), ("NPZ", "*.npz"),
                                                   ("NPY", "*.npy"), ("HDF5", "*.h5")])
nbfiles = len(file_path)
print('Loading ', nbfiles, 'objects')
if nbfiles == 1:
    print('More than one array is needed.')
    sys.exit()

##################################################################
# align objects against the first one and stack it in a 4D array #
##################################################################
obj0, _ = util.load_file(file_path[0])
ndim = obj0.ndim
if ndim != 3:
    print('3D objects are expected')
    sys.exit()

nz, ny, nx = obj0.shape
obj0 = pu.crop_pad(array=obj0, output_shape=(nz+10, ny+10, nx+10))
nz, ny, nx = obj0.shape
stack = np.zeros((nbfiles, nz, ny, nx), dtype=complex)
stack[0, :, :, :] = obj0

for idx in range(1, nbfiles):
    print(os.path.basename(file_path[idx]))
    obj, _ = util.load_file(file_path[idx])
    obj = pu.crop_pad(array=obj, output_shape=obj0.shape)
    obj = pu.align_obj(reference_obj=obj0, obj=obj)

    stack[idx, :, :, :] = obj

############################
# decomposition into modes #
############################
modes, eigenvectors, weights = pu.ortho_modes(array_stack=stack, nb_mode=nb_mode)

print('\nWeights of the', len(weights), ' modes:', weights)

fig, _, _ = gu.multislices_plot(abs(modes[0]), scale='linear', sum_frames=False, plot_colorbar=True,
                                reciprocal_space=False, is_orthogonal=True, title='')
fig.text(0.60, 0.25, "1st mode =" + str('{:.2f}'.format(weights[0]*100) + "%"), size=20)
plt.show()
