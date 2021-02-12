# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import json
import matplotlib.pyplot as plt
from numbers import Real
import numpy as np
import os
import pathlib
from scipy.interpolate import interp1d
from scipy.ndimage.measurements import center_of_mass
import sys
import tkinter as tk
from tkinter import filedialog
sys.path.append('D:/myscripts/bcdi/')
import bcdi.graph.graph_utils as gu
import bcdi.utils.utilities as util
import bcdi.utils.validation as valid

helptext = """
This script can be used to compare the lateral sizes of an object measured by CDI and scanning electron micrscopy. Two
dictionary should be provided as input (one for each technique). The dictionary should contain the following items:
{'threshold': 1D array-like values of thresholds,
 'ang_width_threshold': 2D array-like values (one row for each threshold, the row is the width vs angle of the linecut)}

These dictionaries can be produced by the script angular_profile.py

After aligning the traces of the width vs angle (e.g. if the object was slightly rotated in one of the measurements),
the traces are overlaid in order to determine which threshold is correct.     
"""

datadir = "D:/data/P10_2nd_test_isosurface_Dec2020/data_nanolab/dataset_1/PtNP1_00128/result/linecuts/"
# "D:/data/P10_2nd_test_isosurface_Dec2020/data_nanolab/dataset_1/PtNP1_00128/result/"  # data folder
savedir = "D:/data/P10_2nd_test_isosurface_Dec2020/data_nanolab/AFM-SEM/P10 beamtime P2 particle size SEM/test/"
# results will be saved here, if None it will default to datadir
##################################
# end of user-defined parameters #
##################################

###############################
# list of colors for the plot #
###############################
colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')
markers = ('.', 'v', '^', '<', '>')

#########################
# check some parameters #
#########################
savedir = savedir or datadir
pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)

########################
# load the SEM profile #
########################
plt.ion()
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(initialdir=datadir, title='select the dictionary containing SEM profiles',
                                       filetypes=[("JSON", "*.json"), ("NPZ", "*.npz")])

_, ext = os.path.splitext(file_path)
if ext == '.json':
    sem_dict = json.load(file_path, object_hook=util.decode_json)
else:  # npz
    sem_dict, _ = util.load_file(file_path)

#########################
# load the BCDI profile #
#########################
plt.ion()
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(initialdir=datadir, title='select the dictionary containing BCDI profiles',
                                       filetypes=[("JSON", "*.json"), ("NPZ", "*.npz")])

_, ext = os.path.splitext(file_path)
if ext == '.json':
    bcdi_dict = json.load(file_path, object_hook=util.decode_json)
else:  # npz
    bcdi_dict, _ = util.load_file(file_path)

####################################################################
# get the angular shift between SEM and BCDI traces and align them #
####################################################################


##########################################################################
# plot the aligned BDCI traces for different thresholds vs the SEM trace #
##########################################################################
