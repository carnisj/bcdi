# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import hdf5plugin  # for P10, should be imported before h5py or PyTables
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend("Qt5Agg")  # "Qt5Agg" or "Qt4Agg" depending on the version of Qt installer, bug with Tk
import tkinter as tk
from tkinter import filedialog
import sys
sys.path.append('D:/myscripts/bcdi/')
import bcdi.graph.graph_utils as gu

datadir = ''
fit_option = 'manual'  # only 'manual' for now
xlim = [0, 1]  # limits used for the horizontal axis of the angular plot
ylim = [0, 7]  # limits used for the vertical axis of the angular plot

##############################
# load reciprocal space data #
##############################
plt.ion()
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(initialdir=datadir, title="Select the reciprocal space data",
                                       filetypes=[("NPZ", "*.npz"), ("NPY", "*.npy")])
npzfile = np.load(file_path)
distances = npzfile['distances']
average = npzfile['avergage']

#############
# plot data #
#############
# prepare for masking arrays - 'conventional' arrays won't do it
y_values = np.ma.array(average)
# mask values below a certain threshold
y_values_masked = np.ma.masked_where(np.isnan(y_values), y_values)
fig, ax0 = plt.subplots(1, 1)
plt0 = ax0.plot(distances, np.log10(y_values_masked), 'r')
plt.xlabel('q (1/nm)')
plt.ylabel('Angular average (A.U.)')
plt.xlim(xlim[0], xlim[1])
plt.ylim(ylim[0], ylim[1])

######################
# fit the background #
######################

###################################
# plot background subtracted data #
###################################

plt.ioff()
plt.show()
