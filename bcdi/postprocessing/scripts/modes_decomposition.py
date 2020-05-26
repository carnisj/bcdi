# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np
from scipy.ndimage.measurements import center_of_mass
import matplotlib.pyplot as plt
plt.switch_backend("Qt5Agg")  # "Qt5Agg" or "Qt4Agg" depending on the version of Qt installer, bug with Tk
import os
import gc
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

datadir = "D:/data/P10_August2019/data/gold_2_2_2_00022/pynx/1000_2_debug/"
user_comment = ''  # string, should start with "_"
nb_mode = 5  # number of modes to return in the mode array (starting from 0)
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
print('Array shape', obj0.shape)

amp0 = abs(obj0)
sum0 = amp0.sum()
phase0 = np.angle(obj0)
piz0, piy0, pix0 = center_of_mass(amp0)
piz0, piy0, pix0 = int(piz0), int(piy0), int(pix0)
phase0 = phase0 - phase0[piz0, piy0, pix0]  # set the phae to 0 at the COM of the support
obj0 = amp0 * np.exp(1j * phase0)
stack = np.zeros((nbfiles, nz, ny, nx), dtype=complex)
stack[0, :, :, :] = obj0
del amp0, phase0
gc.collect()

for idx in range(1, nbfiles):
    print(os.path.basename(file_path[idx]))
    obj, _ = util.load_file(file_path[idx])
    obj = pu.crop_pad(array=obj, output_shape=obj0.shape)
    obj = pu.align_obj(reference_obj=obj0, obj=obj)
    amp = abs(obj)
    phase = np.angle(obj) - np.angle(obj)[piz0, piy0, pix0]  # set the phase to 0 at the same pixel
    obj = amp * np.exp(1j * phase)
    stack[idx, :, :, :] = obj
    del amp, phase, obj
    gc.collect()
print('Summed modulus / summed modulus of the first:')
print([str('{:.4f}'.format(abs(stack[idx]).sum()/sum0)) for idx in range(nbfiles)])
for idx in range(nbfiles):
    stack[idx] = stack[idx] * sum0 / abs(stack[idx]).sum()
############################
# decomposition into modes #
############################
np.save(datadir + 'stack_normalized.npy', stack)
modes, weights = pu.ortho_modes(array_stack=stack, nb_mode=nb_mode, method='eig')
print('\nWeights of the', len(weights), ' modes:', weights)

highest_weight = np.unravel_index(np.argmax(weights), shape=weights.shape)[0]
fig, _, _ = gu.multislices_plot(abs(modes[highest_weight]), scale='linear', sum_frames=False, plot_colorbar=True,
                                reciprocal_space=False, is_orthogonal=True, title='')
fig.text(0.6, 0.30, "Strongest mode = {:d}".format(highest_weight), size=20)
fig.text(0.6, 0.25, "weight = {:.2f}%".format(weights[highest_weight]*100), size=20)
plt.pause(0.1)
plt.show()
