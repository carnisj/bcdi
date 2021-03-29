#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

helptext = """
Multiply a diffraction pattern with a 3D apodization window. 
"""

scan = 2227
datadir = "C:/Users/carnis/Work Folders/Documents/data/CH4760_Pt/S"+str(scan)+"/simu/crop400phase/new/apodize_during_phasing/"
mu = np.array([0.0, 0.0, 0.0])
sigma = np.array([0.30, 0.30, 0.30])
covariance = np.diag(sigma**2)
comment = 'diff_apodize'
debug = True
######################################
plt.ion()
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(initialdir=datadir, filetypes=[("NPZ", "*.npz"), ("NPY", "*.npy")])
data = np.load(file_path)['data']
nbz, nby, nbx = data.shape
print(data.max())
maxdata = data.max()

plt.figure()
plt.imshow(np.log10(data.sum(axis=0)), vmin=0, vmax=6)
plt.colorbar()
plt.title('Initial diffraction pattern')
plt.pause(0.1)

grid_z, grid_y, grid_x = np.meshgrid(np.linspace(-1, 1, nbz), np.linspace(-1, 1, nby), np.linspace(-1, 1, nbx),
                                     indexing='ij')
window = multivariate_normal.pdf(np.column_stack([grid_z.flat, grid_y.flat, grid_x.flat]), mean=mu, cov=covariance)
window = window.reshape((nbz, nby, nbx))
if debug:
    plt.figure()
    plt.imshow(window[:, :, nbx//2], vmin=0, vmax=window.max())
    plt.title('Window at middle frame')
    plt.figure()
    plt.plot(window[nbz//2, nby//2, :])
    plt.plot(window[:, nby//2, nbx//2])
    plt.plot(window[nbz//2, :, nbx//2])
    plt.title('Window linecuts at array center')

new_data = np.multiply(data, window)
new_data = new_data * maxdata / new_data.max()

print(new_data.max())
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(np.log10(new_data.sum(axis=0)), vmin=0, vmax=6)
plt.colorbar()
plt.title('Apodized diffraction pattern')
plt.subplot(1, 2, 2)
plt.imshow((new_data-data).sum(axis=0))
plt.colorbar()
plt.title('(Apodized - initial) diffraction pattern')
plt.pause(0.1)

np.savez_compressed(datadir + comment + '.npz', data=new_data)

plt.ioff()
plt.show()