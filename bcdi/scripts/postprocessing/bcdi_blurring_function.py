#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import pathlib
import tkinter as tk
from tkinter import filedialog
import sys
sys.path.append('D:/myscripts/bcdi/')
import bcdi.postprocessing.postprocessing_utils as pu
import bcdi.graph.graph_utils as gu
import bcdi.utils.utilities as util
import bcdi.utils.validation as valid

helptext = """
Load a 3D BCDI reconstruction (.npz file) containing the field 'amp'. After defining a support using a threshold on the 
normalized amplitude, calculate the blurring function by Richardson-Lucy deconvolution. Extract the resolution from 
this blurring function in arbitrary direction. See M. Cherukara et al. Anisotropic nano-scale resolution in 3D 
Bragg coherent diffraction imaging. Appl. Phys. Lett. 113, 203101 (2018); https://doi.org/10.1063/1.5055235
"""

datadir = "D:/data/P10_2nd_test_isosurface_Dec2020/data_nanolab/dataset_2_pearson97.5_newpsf/result/"
savedir = datadir + 'test/'
isosurface_threshold = 0.3
comment = ''  # string to add to the filename when saving, should start with "_"
tick_length = 10  # in plots
tick_width = 2  # in plots
#############################
# define default parameters #
#############################
colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')  # for plots
markers = ('.', 'v', '^', '<', '>', 'x', '+', 'o')  # for plots
mpl.rcParams['axes.linewidth'] = tick_width  # set the linewidth globally
validation_name = 'bcdi_blurring_function'

#########################
# check some parameters #
#########################
if not datadir.endswith('/'):
    datadir += '/'
valid.valid_container(comment, container_types=str, name=validation_name)
if len(comment) != 0 and not comment.startswith('_'):
    comment = '_' + comment

savedir = savedir or datadir
if not savedir.endswith('/'):
    savedir += '/'
pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)