# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, report_fit
import tkinter as tk
from tkinter import filedialog
import sys
sys.path.append('D:/myscripts/bcdi/')

helptext = """
Fit a reciprocal space linecut over selected region using different lineshapes. The fit is performed simultaneously 
over all regions defined by the user, limiting the number of fitting parameters.
"""

datadir = 'D:/data/P10_August2019/data/magnetite_A2_new_00013/pynx/'
xlim = [0, 1]  # limits used for the horizontal axis of plots, leave None otherwise
ylim = [0, 7]  # limits used for the vertical axis of plots, leave None otherwise
lineshape = 'gaussian'  # lineshape to use for fitting, only 'gaussian' for now
scale = 'log'  # scale for plots
field_names = ['distances', 'average']  # names of the fields in the file
fit_range = [[0.30, 0.55], [0.70, 0.81]]  # list of ranges for simultaneous fit [[start1, stop1],[start2, stop2],...]

##################################
# end of user-defined parameters #
##################################


def gaussian_dataset(params, iterator, x_axis):
    """calc gaussian from params for data set i
    using simple, hardwired naming convention"""
    amp = params['amp_%i' % (iterator+1)].value
    cen = params['cen_%i' % (iterator+1)].value
    sig = params['sig_%i' % (iterator+1)].value
    return amp*np.exp(-(x_axis-cen)**2/(2*sig**2))


def objective(params, x_axis, data):
    """ calculate total residual for fits to several data sets held
    in a 2-D array, and modeled by Gaussian functions"""
    ndata, nx = data.shape
    resid = 0.0*data[:]
    # make residual per data set
    for ii in range(ndata):
        resid[ii, :] = data[ii, :] - gaussian_dataset(params, ii, x_axis)
    # now flatten this to a 1D array, as minimize() needs
    return resid.flatten()




#####################
# load the 1D curve #
#####################
plt.ion()
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(initialdir=datadir, title="Select the data to fit",
                                       filetypes=[("NPZ", "*.npz")])
npzfile = np.load(file_path)
distances = npzfile[field_names[0]]
average = npzfile[field_names[1]]

#############
# plot data #
#############
fig, ax = plt.subplots(1, 1)
if scale == 'linear':
    plt.plot(distances, average, 'r')
else:
    plt.plot(distances, np.log10(average), 'r')
plt.xlabel('q (1/nm)')
plt.ylabel('Angular average (A.U.)')
if xlim is not None:
    ax.set_xlim(xlim[0], xlim[1])
if ylim is not None:
    ax.set_ylim(ylim[0], ylim[1])

##################################################
# combine ranges of interest in a single dataset #
##################################################
nb_ranges = len(fit_range)
fit_range = np.asarray(fit_range)

combined_xaxis = []
combined_data = []
for idx in range(nb_ranges):
    # find indices of distances belonging to ranges of interest

    combined_xaxis = combined_xaxis.append(distances[indices])
    combined_data = combined_data.append(average[indices])

combined_xaxis = np.asarray(combined_xaxis)
combined_data = np.asarray(combined_data)

##############################################
# fit user-defined range using the lineshape #
##############################################
# create nb_fit sets of parameters, one per data set
fit_params = Parameters()
for idx, _ in enumerate(combined_data):
    mu = (fit_range[idx, 0] + fit_range[idx, 1]) / 2
    sigma = abs(fit_range[idx, 0] - fit_range[idx, 1]) / 4
    fit_params.add('amp_%i' % (idx+1), value=0.5, min=0.0,  max=200)
    fit_params.add('cen_%i' % (idx+1), value=mu, min=mu-0.5,  max=mu+0.5)
    fit_params.add('sig_%i' % (idx+1), value=sigma, min=sigma/2, max=sigma*2)

# run the global fit to all the data sets
result = minimize(objective, fit_params, args=(combined_xaxis, combined_data))
report_fit(result.params)

#####################
# plot data and fit #
#####################

# fig.savefig(datadir + lineshape + ' fit.png')
plt.ioff()
plt.show()
