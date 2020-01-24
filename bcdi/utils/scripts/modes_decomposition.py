# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend("Qt5Agg")  # "Qt5Agg" or "Qt4Agg" depending on the version of Qt installer, bug with Tk
import pathlib
import os
import sys
import tkinter as tk
from tkinter import filedialog
sys.path.append('D:/myscripts/bcdi/')
import bcdi.graph.graph_utils as gu
import bcdi.experiment.experiment_utils as exp
import bcdi.postprocessing.postprocessing_utils as pu
import bcdi.preprocessing.preprocessing_utils as pru

helptext = """
Decomposition of a set of reconstructed objects from phase retrieval in an orthogonal set,
the first mode is the most prominent feature of the solution space.
"""

datadir = "D:/data/P10_August2019/data/gold_2_2_2_00022/pynx/800_800_800_1_1_1/v5/"
user_comment = ''  # string, should start with "_"
nb_mode = None  # number of modes to save in the file (starting from 0)
################
# Load objects #
################
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilenames(initialdir=datadir,
                                        filetypes=[("NPZ", "*.npz"),
                                                   ("NPY", "*.npy"), ("CXI", "*.cxi"), ("HDF5", "*.h5")])
nbfiles = len(file_path)

if nbfiles == 1:
    print('More than one array is needed.')
    sys.exit()

################################################################
# align objects against the first one and stack it in an array #
################################################################
obj0, _ = pu.load_reconstruction(file_path[0])
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
    obj, _ = pu.load_reconstruction(file_path[idx])
    obj = pu.crop_pad(array=obj, output_shape=obj0.shape)
    obj = pu.align_obj(reference_obj=obj0, obj=obj)

    stack[idx, :, :, :] = obj

############################
# decomposition into modes #
############################
modes, eigenvectors, weights = pu.ortho_modes(stack=stack, nb_mode=nb_mode, return_matrix=False, return_weights=True)

print(weights)

gu.multislices_plot(abs(modes[0]), scale='linear', sum_frames=False, plot_colorbar=True, reciprocal_space=False,
                    is_orthogonal=True)

plt.show()



