#!/usr/bin/env python3

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import gc
import os
import sys
import tkinter as tk
from numbers import Real
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.measurements import center_of_mass

import bcdi.graph.graph_utils as gu
import bcdi.postprocessing.postprocessing_utils as pu
import bcdi.utils.image_registration as reg
import bcdi.utils.utilities as util
import bcdi.utils.validation as valid

plt.switch_backend("Qt5Agg")
# "Qt5Agg" or "Qt4Agg" depending on the version of Qt installer, bug with Tk

helptext = """
Eigendecomposition of a set of 3D reconstructed objects from phase retrieval,
ideally the first mode should be as high as possible. Adapted from PyNX.
"""

datadir = "D:/data/P10_August2020_CDI/data/mag_3_macro1/centrosym/"
user_comment = ""  # string, should start with "_"
nb_mode = 1  # number of modes to return in the mode array
alignment_method = "support"  # 'modulus' or 'support'
# if 'modulus', use the center of mass of the modulus.
# If 'support', use the center of mass of a support object defined by support_threshold
support_threshold = 0.2  # threshold on the normalized modulus to define the support
# if alignement_method is 'support'
debug = True  # True to see debugging plots
#########################
# check some parameters #
#########################
valid.valid_container(user_comment, container_types=str, name="modes_decomposition")
valid.valid_item(
    value=nb_mode, allowed_types=int, min_excluded=0, name="modes_decomposition"
)
valid.valid_item(
    value=support_threshold,
    allowed_types=Real,
    min_included=0,
    name="modes_decomposition",
)
if alignment_method not in {"modulus", "support"}:
    raise ValueError(
        f"wrong value for alignment_method {alignment_method}, "
        'allowed are "support" and "modulus"'
    )
valid.valid_item(value=debug, allowed_types=bool, name="modes_decomposition")

################
# Load objects #
################
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilenames(
    initialdir=datadir,
    filetypes=[("CXI", "*.cxi"), ("NPZ", "*.npz"), ("NPY", "*.npy"), ("HDF5", "*.h5")],
)
nbfiles = len(file_path)
print("Loading ", nbfiles, "objects")
if nbfiles == 1:
    print("More than one array is needed.")
    sys.exit()

##################################################################
# align objects against the first one and stack it in a 4D array #
##################################################################
obj0, _ = util.load_file(file_path[0])
ndim = obj0.ndim
if ndim != 3:
    print("3D objects are expected")
    sys.exit()

nz, ny, nx = obj0.shape
obj0 = util.crop_pad(array=obj0, output_shape=(nz + 10, ny + 10, nx + 10))
nz, ny, nx = obj0.shape
print("Array shape", obj0.shape)

amp0 = abs(obj0)
sum0 = amp0.sum()
phase0 = np.angle(obj0)
if alignment_method in ["modulus", "skip"]:
    piz0, piy0, pix0 = center_of_mass(amp0)
else:  # 'support'
    support = np.zeros(amp0.shape)
    support[amp0 > support_threshold * amp0.max()] = 1
    piz0, piy0, pix0 = center_of_mass(support)

piz0, piy0, pix0 = int(piz0), int(piy0), int(pix0)
phase0 = (
    phase0 - phase0[piz0, piy0, pix0]
)  # set the phase to 0 at the COM of the support
obj0 = amp0 * np.exp(1j * phase0)
stack = np.zeros((nbfiles, nz, ny, nx), dtype=complex)
stack[0, :, :, :] = obj0
del amp0, phase0
gc.collect()

for idx in range(1, nbfiles):
    print("\n" + os.path.basename(file_path[idx]))
    obj, _ = util.load_file(file_path[idx])
    obj = util.crop_pad(array=obj, output_shape=obj0.shape)
    obj, _ = reg.align_arrays(
        reference_array=obj0,
        shifted_array=obj,
        shift_method=alignment_method,
        support_threshold=support_threshold,
        debugging=debug,
    )
    amp = abs(obj)
    phase = (
        np.angle(obj) - np.angle(obj)[piz0, piy0, pix0]
    )  # set the phase to 0 at the same pixel
    obj = amp * np.exp(1j * phase)
    stack[idx, :, :, :] = obj
    del amp, phase, obj
    gc.collect()
print("Summed modulus / summed modulus of the first:")
print([str(f"{abs(stack[idx]).sum() / sum0:.4f}") for idx in range(nbfiles)])
for idx in range(nbfiles):
    stack[idx] = stack[idx] * sum0 / abs(stack[idx]).sum()
############################
# decomposition into modes #
############################
modes, _, weights = pu.ortho_modes(array_stack=stack, nb_mode=nb_mode, method="eig")
print("\nWeights of the", len(weights), " modes:", weights)

highest_weight = np.unravel_index(np.argmax(weights), shape=weights.shape)[0]
np.savez_compressed(datadir + "highest_mode.npz", modes[highest_weight])

fig, _, _ = gu.multislices_plot(
    abs(modes[highest_weight]),
    scale="linear",
    sum_frames=False,
    plot_colorbar=True,
    reciprocal_space=False,
    is_orthogonal=True,
    title="",
)
fig.text(0.6, 0.30, f"Strongest mode = {highest_weight:d}", size=20)
fig.text(0.6, 0.25, f"weight = {weights[highest_weight] * 100:.2f}%", size=20)
plt.pause(0.1)
plt.show()
