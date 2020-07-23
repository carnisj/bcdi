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
sys.path.append('D:/myscripts/bcdi/')
import bcdi.utils.utilities as util
import bcdi.graph.graph_utils as gu
import bcdi.postprocessing.postprocessing_utils as pu


helptext = """
Apply intensity normalization of additive aptially multiplexed patterns with n encoded phases, as described in:
R. Jaurez-Salazar et al. Optics and Lasers in Engineering 77, 225-229 (2016).

Input: a 3D real intensity array
"""

datadir = "D:/data/P10_August2019_CDI/data/gold_2_2_2_00022/pynx/1000_1000_1000_1_1_1/current_paper/"
savedir = "D:/data/P10_August2019_CDI/data/gold_2_2_2_00022/pynx/scratch/"
comment = '_gaussian_21_5'  # should start with _
threshold = 0.2  # threshold used to define the support for background fitting (intensity normalized to 1)
roll_modes = (0, -10, 0)   # axis=(0, 1, 2), correct a misalignement of the data
save = True  # True to save the result as a NPZ file
nb_phases = 1  # number of encoded phases, generally 1 if this is a single measurement
#########################################
# parameters for the background fitting #
#########################################
background_method = 'skip'  # 'gaussian', 'polyfit' or 'skip': 'gaussian' will convolve a gaussian with the object,
# 'polyfit' will fit a polynomial or order background_order to the object. 'skip' defines a zero background
background_order = 1  # degree of the polynomial for background fitting 1~4.
background_kernel = 41  # size of the kernel for the 'gaussian' method
background_sigma = 7  # standard deviation of the gaussian for the 'gaussian' method
#########################################
# parameters for the modulation fitting #
#########################################
modulation_method = 'gaussian'  # 'gaussian' or 'polyfit': 'gaussian' will convolve a gaussian with the object,
# 'polyfit' will fit a polynomial or order modulation_order to the object.
modulation_order = 4  # degree of the polynomial for modulation fitting 1~4
modulation_kernel = 21  # size of the kernel for the 'gaussian' method
modulation_sigma = 5  # standard deviation of the gaussian for the 'gaussian' method
##########################
# end of user parameters #
##########################


def fit3d_poly1(x_axis, a, b, c, d):
    return a + b*x_axis[0] + c*x_axis[1] + d*x_axis[2]


def fit3d_poly2(x_axis, a, b, c, d, e, f, g):
    return a + b*x_axis[0] + c*x_axis[1] + d*x_axis[2] + e*x_axis[0]**2 + f*x_axis[1]**2 + g*x_axis[2]**2


def fit3d_poly3(x_axis, a, b, c, d, e, f, g, h, i, j):
    return a + b*x_axis[0] + c*x_axis[1] + d*x_axis[2] + e*x_axis[0]**2 + f*x_axis[1]**2 + g*x_axis[2]**2 +\
           h*x_axis[0]**3 + i*x_axis[1]**3 + j*x_axis[2]**3


def fit3d_poly4(x_axis, a, b, c, d, e, f, g, h, i, j, k, l, m):
    return a + b*x_axis[0] + c*x_axis[1] + d*x_axis[2] + e*x_axis[0]**2 + f*x_axis[1]**2 + g*x_axis[2]**2 +\
           h*x_axis[0]**3 + i*x_axis[1]**3 + j*x_axis[2]**3 + k*x_axis[0]**4 + l*x_axis[1]**4 + m*x_axis[2]**4


####################
# check parameters #
####################
assert 1 <= background_order <= 4, 'polynomial fitting of order > 4 not implemented'
assert 1 <= modulation_order <= 4, 'polynomial fitting of order > 4 not implemented'
assert background_method in ['gaussian', 'polyfit', 'skip'], 'invalid setting for background_method'
assert modulation_method in ['gaussian', 'polyfit'], 'invalid setting for modulation_method'

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
filename = filename + comment
print('filename:', filename)
obj, extension = util.load_file(file_path)
nz, ny, nx = obj.shape

######################
# data preprocessing #
######################
# correct a misalignement of the object
obj = np.roll(obj, roll_modes, axis=(0, 1, 2))
gu.multislices_plot(abs(obj), sum_frames=True, plot_colorbar=True, reciprocal_space=False, is_orthogonal=True,
                    vmin=0, title='obj after centering')

obj[np.isnan(obj)] = 0
obj = abs(obj)
obj = obj / obj.max()  # normalize to 1

############################
# determine the background #
############################
support_bckg = np.zeros((nz, ny, nx))
support_bckg[obj >= threshold] = 1
xdata = np.nonzero(support_bckg)
nb_nonzero = xdata[0].size
temp_obj = np.copy(obj)
temp_obj[obj < threshold] = np.nan
gu.multislices_plot(temp_obj, sum_frames=False, plot_colorbar=True, reciprocal_space=False, is_orthogonal=True,
                    vmin=0, vmax=1, title='points used for background fitting')
del temp_obj
gc.collect()

if background_method == 'polyfit':
    grid_z, grid_y, grid_x = np.meshgrid(np.arange(0, nz, 1), np.arange(0, ny, 1), np.arange(0, nx, 1), indexing='ij')
    grid = np.concatenate((grid_z.reshape((1, obj.size)), grid_y.reshape((1, obj.size)),
                           grid_x.reshape((1, obj.size))), axis=0)  # xdata should have a 3xN array
    if background_order == 1:
        guess = np.ones(4)
        params, cov = optimize.curve_fit(fit3d_poly1, xdata=xdata, ydata=obj[xdata].reshape(nb_nonzero),
                                         p0=guess)
        background = fit3d_poly1(grid, params[0], params[1], params[2], params[3])
    elif background_order == 2:
        guess = np.ones(7)
        params, cov = optimize.curve_fit(fit3d_poly2, xdata=xdata, ydata=obj[xdata].reshape(nb_nonzero),
                                         p0=guess)
        background = fit3d_poly2(grid, params[0], params[1], params[2], params[3], params[4], params[5], params[6])
    elif background_order == 3:
        guess = np.ones(10)
        params, cov = optimize.curve_fit(fit3d_poly3, xdata=xdata, ydata=obj[xdata].reshape(nb_nonzero),
                                         p0=guess)
        background = fit3d_poly3(grid, params[0], params[1], params[2], params[3], params[4], params[5], params[6],
                                 params[7], params[8], params[9])
    else:
        guess = np.ones(13)
        params, cov = optimize.curve_fit(fit3d_poly4, xdata=xdata, ydata=obj[xdata].reshape(nb_nonzero),
                                         p0=guess)
        background = fit3d_poly4(grid, params[0], params[1], params[2], params[3], params[4], params[5], params[6],
                                 params[7], params[8], params[9], params[10], params[11], params[12])
    del grid_z, grid_y, grid_x, grid
    gc.collect()

elif background_method == 'gaussian':
    background = pu.filter_3d(obj, filter_name='gaussian', kernel_length=background_kernel,
                              sigma=background_sigma, debugging=False)

else:  # skip
    print('skipping background determination')
    background = np.zeros((nz, ny, nx))

##############################
# plot the fitted background #
##############################
background = background.reshape((nz, ny, nx))
background[support_bckg == 0] = 0
gu.multislices_plot(background, sum_frames=False, plot_colorbar=True, reciprocal_space=False, is_orthogonal=True,
                    vmin=0, vmax=1, title='fitted background')

#######################################################
# subtract the background to the intensity and square #
#######################################################
obj_bck = np.square(obj - background)
obj_bck[np.isnan(obj_bck)] = 0
obj_bck = obj_bck / obj_bck[xdata].max()
gu.multislices_plot(obj_bck, sum_frames=False, plot_colorbar=True, reciprocal_space=False,
                    vmin=0, vmax=1, is_orthogonal=True, title='(obj-background)**2')
del support_bckg, xdata
gc.collect()

#############################################################
# fit the modulation to the points belonging to the support #
#############################################################
threshold_modul = threshold**2  # square the threshold since we are fitting (obj-background)**2
obj_bck[obj_bck < threshold_modul] = 0
support_modul = np.zeros((nz, ny, nx))
support_modul[obj_bck >= threshold_modul] = 1
xdata = np.nonzero(support_modul)
nb_nonzero = xdata[0].size
temp_obj = np.copy(obj_bck)
temp_obj[obj_bck < threshold_modul] = np.nan
gu.multislices_plot(temp_obj, sum_frames=False, plot_colorbar=True, reciprocal_space=False, is_orthogonal=True,
                    vmin=0, vmax=1, title='points used for modulation fitting')
del temp_obj
gc.collect()

if background_method == 'polyfit':
    grid_z, grid_y, grid_x = np.meshgrid(np.arange(0, nz, 1), np.arange(0, ny, 1), np.arange(0, nx, 1), indexing='ij')
    grid = np.concatenate((grid_z.reshape((1, obj.size)), grid_y.reshape((1, obj.size)),
                           grid_x.reshape((1, obj.size))), axis=0)  # xdata should have a 3xN array
    if modulation_order == 1:
        guess = np.ones(4)
        params, cov = optimize.curve_fit(fit3d_poly1, xdata=xdata, ydata=obj_bck[xdata].reshape(nb_nonzero),
                                         p0=guess)
        modulation = fit3d_poly1(grid, params[0], params[1], params[2], params[3])
    elif modulation_order == 2:
        guess = np.ones(7)
        params, cov = optimize.curve_fit(fit3d_poly2, xdata=xdata, ydata=obj_bck[xdata].reshape(nb_nonzero),
                                         p0=guess)
        modulation = fit3d_poly2(grid, params[0], params[1], params[2], params[3], params[4], params[5], params[6])
    elif modulation_order == 3:
        guess = np.ones(10)
        params, cov = optimize.curve_fit(fit3d_poly3, xdata=xdata, ydata=obj_bck[xdata].reshape(nb_nonzero),
                                         p0=guess)
        modulation = fit3d_poly3(grid, params[0], params[1], params[2], params[3], params[4], params[5], params[6],
                                 params[7], params[8], params[9])
    else:  # 4th order
        guess = np.ones(13)
        params, cov = optimize.curve_fit(fit3d_poly4, xdata=xdata, ydata=obj_bck[xdata].reshape(nb_nonzero),
                                         p0=guess)
        modulation = fit3d_poly4(grid, params[0], params[1], params[2], params[3], params[4], params[5], params[6],
                                 params[7], params[8], params[9], params[10], params[11], params[12])
else:  # 'gaussian'
    modulation = pu.filter_3d(obj_bck, filter_name='gaussian', kernel_length=modulation_kernel,
                              sigma=modulation_sigma, debugging=False)

##############################
# plot the fitted modulation #
##############################
modulation = modulation.reshape((nz, ny, nx))
modulation = np.sqrt(2*nb_phases*modulation)
modulation[support_modul == 0] = 1
modulation[np.isnan(modulation)] = 1
gu.multislices_plot(modulation, sum_frames=False, plot_colorbar=True, reciprocal_space=False, is_orthogonal=True,
                    vmin=0, vmax=1, title='fitted modulation')

############################################
# calculate and plot the normalized object #
############################################
result = np.divide(obj - background, modulation)
result[np.isnan(result)] = 0
result = result / result[result >= threshold].max()
piz, piy, pix = np.unravel_index(result.argmax(), shape=(nz, ny, nx))
print('maximum at voxel:', piz, piy, pix, '   value=', result.max())
gu.multislices_plot(result, slice_position=(piz, piy, pix), sum_frames=False, plot_colorbar=True,
                    reciprocal_space=False, is_orthogonal=True, vmin=0, vmax=1, title='result at max')

del support_modul, threshold_modul, xdata
gc.collect()

fig = gu.combined_plots(tuple_array=(obj, obj, obj, result, result, result), tuple_sum_frames=False,
                        tuple_sum_axis=(0, 1, 2, 0, 1, 2), tuple_colorbar=True, tuple_vmin=0, tuple_vmax=1,
                        tuple_title=('Original', 'Original', 'Original', 'Result', 'Result', 'Result'),
                        tuple_scale='linear', is_orthogonal=True, reciprocal_space=False,
                        position=(321, 323, 325, 322, 324, 326))

if save:
    np.savez_compressed(savedir + filename + '.npz', obj=result)
    fig.savefig(savedir + filename + '.png')
plt.ioff()
plt.show()
