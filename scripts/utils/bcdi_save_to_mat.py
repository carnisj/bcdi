#!/usr/bin/env python3

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import os
import sys
import tkinter as tk
from tkinter import filedialog

import numpy as np
from scipy.io import savemat

import bcdi.utils.utilities as util

helptext = """
Load a 2D or 3D object and save it into Matlab .mat format.
"""

datadir = "D:/data/P10_August2019/data/gold_2_2_2_00022/pynx/1000_1000_1000_1_1_1/"
save_name = ""  # use this to change the filename,
# it will default to the actual file name if save_name=''
#############
# load data #
#############
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    initialdir=datadir,
    filetypes=[("NPZ", "*.npz"), ("NPY", "*.npy"), ("CXI", "*.cxi"), ("HDF5", "*.h5")],
)
obj, extension = util.load_file(file_path)

if save_name == "":
    save_name = os.path.splitext(os.path.basename(file_path))[0]


if obj.ndim == 2:
    savemat(datadir + save_name + ".mat", {"data": obj})
elif obj.ndim == 3:
    savemat(
        datadir + save_name + ".mat",
        {"data": np.moveaxis(obj, [0, 1, 2], [-1, -3, -2])},
    )
else:
    print("a 2D or 3D array is expected")
    sys.exit()
