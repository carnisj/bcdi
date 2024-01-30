#!/usr/bin/env python3

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import tkinter as tk
from tkinter import filedialog

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

import bcdi.utils.utilities as util
from bcdi.graph.colormap import ColormapFactory
from bcdi.utils import image_registration as reg

helptext = """
Compare the correlation between several 3D objects.
"""

datadir = ""
threshold_correlation = 0.05
# only points above that threshold will be considered for correlation calculation
###################
# define colormap #
###################
my_cmap = ColormapFactory().cmap

#############
# load data #
#############
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilenames(
    initialdir=datadir,
    filetypes=[("NPZ", "*.npz"), ("NPY", "*.npy"), ("CXI", "*.cxi"), ("HDF5", "*.h5")],
)
nbfiles = len(file_path)
print(nbfiles, "files selected")
#################################################################
# loop through files and calculate the correlation coefficients #
#################################################################
correlation = np.zeros((nbfiles, nbfiles))
for raw in range(nbfiles):
    reference_obj, _ = util.load_file(file_path[raw])
    reference_obj = abs(reference_obj) / abs(reference_obj).max()
    nbz, nby, nbx = reference_obj.shape
    reference_obj = util.crop_pad(
        array=reference_obj, output_shape=[nbz + 10, nby + 10, nbx + 10]
    )
    correlation[raw, raw] = 1
    for col in range(raw + 1, nbfiles):
        test_obj, _ = util.load_file(file_path[col])  # which index?
        test_obj = abs(test_obj) / abs(test_obj).max()
        test_obj = util.crop_pad(
            array=test_obj, output_shape=[nbz + 10, nby + 10, nbx + 10]
        )
        # align reconstructions
        shiftz, shifty, shiftx = reg.getimageregistration(
            abs(reference_obj), abs(test_obj), precision=100
        )
        test_obj = reg.subpixel_shift(test_obj, shiftz, shifty, shiftx)
        print("\nReference =", raw, "  Test =", col)
        print(
            "z shift",
            str(f"{shiftz:.2f}"),
            ", y shift",
            str(f"{shifty:.2f}"),
            ", x shift",
            str(f"{shiftx:.2f}"),
        )

        correlation[raw, col] = pearsonr(
            np.ndarray.flatten(
                abs(reference_obj[reference_obj > threshold_correlation])
            ),
            np.ndarray.flatten(abs(test_obj[reference_obj > threshold_correlation])),
        )[0]
        correlation[col, raw] = correlation[raw, col]
        print("Correlation=", str(f"{correlation[raw, col]:.2f}"))

plt.figure()
plt.imshow(correlation, cmap=my_cmap, vmin=0, vmax=1)
plt.colorbar()
plt.title("Pearson correlation coefficients")
plt.show()
