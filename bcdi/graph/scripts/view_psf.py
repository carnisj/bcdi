# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

from numpy.fft import fftshift
import matplotlib.pyplot as plt
import h5py
import tkinter as tk
from tkinter import filedialog
import sys
sys.path.append('D:/myscripts/bcdi/')
import bcdi.graph.graph_utils as gu

helptext = """
Open and plot the psf from a .cxi reconstruction file (from PyNX). The psf has to be fftshifted.
"""

datadir = "D:/data/P10_August2019/data/gold_2_2_2_00022/pynx/1000_1000_1000_1_1_1/"
is_orthogonal = True  # True if the data was orthogonalized before phasing
comment = '_binning2x2x2'  # should start with _
###################
# define colormap #
###################
bad_color = '1.0'  # white background
colormap = gu.Colormap(bad_color=bad_color)
my_cmap = colormap.cmap

root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(initialdir=datadir, filetypes=[("CXI", "*.cxi")])

h5file = h5py.File(file_path, 'r')
dataset = fftshift(h5file['/entry_1/image_1/instrument_1/detector_1/point_spread_function'].value)

fig, _, _ = gu.multislices_plot(dataset[215:285, 215:285, 215:285], scale='log', sum_frames=True,
                                title='log(psf) in detector frame', reciprocal_space=False,
                                vmin=-5, is_orthogonal=is_orthogonal, plot_colorbar=True)
fig.savefig(datadir + 'psf_sum' + comment + '.png')

fig, _, _ = gu.multislices_plot(dataset[215:285, 215:285, 215:285], scale='log', sum_frames=False,
                                title='log(psf) in detector frame', reciprocal_space=False,
                                vmin=-5, is_orthogonal=is_orthogonal, plot_colorbar=True)
fig.savefig(datadir + 'psf_centralslice' + comment + '.png')

plt.show()
