# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import os
import gc
import scipy.optimize as optimize
import numpy as np
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import filedialog
import sys
sys.path.append('C:/Users/Jerome/Documents/myscripts/bcdi/')
import bcdi.utils.utilities as util
import bcdi.graph.graph_utils as gu


helptext = """
Apply intensity normalization of additive aptially multiplexed patterns with n encoded phases, as described in:
R. Jaurez-Salazar et al. Optics and Lasers in Engineering 77, 225-229 (2016).

Input: a 3D real intensity array
"""

datadir = "D:/data/P10_August2019_CDI/data/gold_2_2_2_00022/pynx/1000_1000_1000_1_1_1/current_paper/"
savedir = "D:/data/P10_August2019_CDI/data/gold_2_2_2_00022/pynx/scratch/"
comment = ''  # should start with _
threshold = 0  # the intensity (normalized to 1) below this value will be set to 0
roll_modes = (0, -8, 0)   # axis=(0, 1, 2), correct a misalignement of the data
save = False  # True to save the result as a NPZ file
##############################################
# parameters for the normalization algorithm #
##############################################
nb_phases = 1  # number of encoded phases
background_order = 4  # degree of the polynomial for background fitting
modulation_order = 4  # degree of the polynomial for modulation fitting
##########################
# end of user parameters #
##########################


def fit3d_poly1(xdata, a, b, c, d):
    return a + b*xdata[0] + c*xdata[1] + d*xdata[2]


def fit3d_poly2(xdata, a, b, c, d, e, f, g):
    return a + b*xdata[0] + c*xdata[1] + d*xdata[2] + e*xdata[0]**2 + f*xdata[1]**2 + g*xdata[2]**2


def fit3d_poly3(xdata, a, b, c, d, e, f, g, h, i, j):
    return a + b*xdata[0] + c*xdata[1] + d*xdata[2] + e*xdata[0]**2 + f*xdata[1]**2 + g*xdata[2]**2 + h*xdata[0]**3 +\
           i*xdata[1]**3 + j*xdata[2]**3


def fit3d_poly4(xdata, a, b, c, d, e, f, g, h, i, j, k, l, m):
    return a + b*xdata[0] + c*xdata[1] + d*xdata[2] + e*xdata[0]**2 + f*xdata[1]**2 + g*xdata[2]**2 + h*xdata[0]**3 +\
           i*xdata[1]**3 + j*xdata[2]**3 + k*xdata[0]**4 + l*xdata[1]**4 + m*xdata[2]**4


####################
# check parameters #
####################
assert background_order <= 4, 'polynomial fitting of order > 4 not implemented'
assert modulation_order <= 4, 'polynomial fitting of order > 4 not implemented'

#################
# load the data #
#################
plt.ion()
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(initialdir=datadir, title="Select the CCF file",
                                       filetypes=[("HDF5", "*.h5"), ("NPZ", "*.npz"),
                                                  ("NPY", "*.npy"), ("CXI", "*.cxi")])
filename = os.path.splitext(os.path.basename(file_path))[0]  # the extension .npz is removed
obj, extension = util.load_file(file_path)
nz, ny, nx = obj.shape

######################
# data preprocessing #
######################
# correct a misalignement of the object
obj = np.roll(obj, roll_modes, axis=(0, 1, 2))
fig, _, _ = gu.multislices_plot(abs(obj), sum_frames=True, plot_colorbar=True, reciprocal_space=False,
                                is_orthogonal=True,  title='obj after centering')

obj = abs(obj)
obj = obj / obj[~np.isnan(obj)].max()  # normalize to 1
obj[obj < threshold] = 0  # apply intensity threshold

###########################
# set up the grid and fit #
###########################
grid_z, grid_y, grid_x = np.meshgrid(np.arange(0, nz, 1), np.arange(0, ny, 1), np.arange(0, nx, 1), indexing='ij')
xdata = np.concatenate((grid_z.reshape((1, obj.size)), grid_y.reshape((1, obj.size)),
                        grid_x.reshape((1, obj.size))), axis=0)  # xdata should have a 3xN array

if background_order == 1:
    guess = np.ones(4)
    params, cov = optimize.curve_fit(fit3d_poly1, xdata=xdata, ydata=obj.reshape(obj.size), p0=guess)
elif background_order == 2:
    guess = np.ones(7)
    params, cov = optimize.curve_fit(fit3d_poly2, xdata=xdata, ydata=obj.reshape(obj.size), p0=guess)
elif background_order == 3:
    guess = np.ones(10)
    params, cov = optimize.curve_fit(fit3d_poly3, xdata=xdata, ydata=obj.reshape(obj.size), p0=guess)
else:  # 4th order polynomial
    guess = np.ones(13)
    params, cov = optimize.curve_fit(fit3d_poly4, xdata=xdata, ydata=obj.reshape(obj.size), p0=guess)

##############################
# plot the fitted background #
##############################
if background_order == 1:
    background = fit3d_poly1(xdata, params[0], params[1], params[2], params[3])
elif background_order == 2:
    background = fit3d_poly2(xdata, params[0], params[1], params[2], params[3], params[4], params[5], params[6])
elif background_order == 3:
    background = fit3d_poly3(xdata, params[0], params[1], params[2], params[3], params[4], params[5], params[6],
                             params[7], params[8], params[9])
else:  # 4th order polynomial
    background = fit3d_poly4(xdata, params[0], params[1], params[2], params[3], params[4], params[5], params[6],
                             params[7], params[8], params[9], params[10], params[11], params[12])
del params, cov
gc.collect()

background = background.reshape((nz, ny, nx))
gu.multislices_plot(background, sum_frames=False, plot_colorbar=True, reciprocal_space=False, is_orthogonal=True,
                    title='fitted background')

#######################################################
# subtract the background to the intensity and square #
#######################################################
obj_bck = np.square(obj - background)
gu.multislices_plot(obj_bck, sum_frames=False, plot_colorbar=True, reciprocal_space=False,
                    is_orthogonal=True, title='(obj-background)**2')

######################
# fit the modulation #
######################
if modulation_order == 1:
    guess = np.ones(4)
    params, cov = optimize.curve_fit(fit3d_poly1, xdata=xdata, ydata=obj_bck.reshape(obj_bck.size), p0=guess)
elif modulation_order == 2:
    guess = np.ones(7)
    params, cov = optimize.curve_fit(fit3d_poly2, xdata=xdata, ydata=obj_bck.reshape(obj_bck.size), p0=guess)
elif modulation_order == 3:
    guess = np.ones(10)
    params, cov = optimize.curve_fit(fit3d_poly3, xdata=xdata, ydata=obj_bck.reshape(obj_bck.size), p0=guess)
else:  # 4th order polynomial
    guess = np.ones(13)
    params, cov = optimize.curve_fit(fit3d_poly4, xdata=xdata, ydata=obj_bck.reshape(obj_bck.size), p0=guess)

##############################
# plot the fitted modulation #
##############################
if modulation_order == 1:
    modulation = fit3d_poly1(xdata, params[0], params[1], params[2], params[3])
elif modulation_order == 2:
    modulation = fit3d_poly2(xdata, params[0], params[1], params[2], params[3], params[4], params[5], params[6])
elif modulation_order == 3:
    modulation = fit3d_poly3(xdata, params[0], params[1], params[2], params[3], params[4], params[5], params[6],
                             params[7], params[8], params[9])
else:  # 4th order polynomial
    modulation = fit3d_poly4(xdata, params[0], params[1], params[2], params[3], params[4], params[5], params[6],
                             params[7], params[8], params[9], params[10], params[11], params[12])
del params, cov
gc.collect()

modulation = np.sqrt(2*nb_phases*modulation)
modulation = modulation.reshape((nz, ny, nx))
gu.multislices_plot(modulation, sum_frames=False, plot_colorbar=True, reciprocal_space=False, is_orthogonal=True,
                    title='fitted modulation')

############################################
# calculate and plot the normalized object #
############################################
result = np.divide(obj_bck, modulation)
result = result / result[~np.isnan(result)].max()
if save:
    np.savez_compressed(savedir + filename + comment + '.npz', obj=result)

gu.combined_plots(tuple_array=(obj, result), tuple_sum_frames=False, tuple_sum_axis=0, tuple_colorbar=True,
                  tuple_vmin=0, tuple_vmax=1, tuple_title=('before', 'after'), tuple_scale='linear', is_orthogonal=True,
                  reciprocal_space=False, position=(121, 122))
plt.ioff()
plt.show()
