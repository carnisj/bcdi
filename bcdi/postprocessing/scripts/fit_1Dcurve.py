# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from lmfit import minimize, Parameters, report_fit
import tkinter as tk
from tkinter import filedialog
import sys
sys.path.append('D:/myscripts/bcdi/')
import bcdi.utils.utilities as util

helptext = """
Fit a reciprocal space linecut over selected region using different lineshapes. The fit is performed simultaneously 
over all regions defined by the user, limiting the number of fitting parameters.
"""

datadir = 'D:/data/P10_August2019/data/magnetite_A2_new_00013/pynx/'
xlim = [0, 1]  # limits used for the horizontal axis of plots, leave None otherwise
ylim = [0, 3]  # limits used for the vertical axis of plots, leave None otherwise
lineshape = 'gaussian'  # lineshape to use for fitting, only 'gaussian' for now
scale = 'log'  # scale for plots
field_names = ['distances', 'average']  # names of the fields in the file
fit_range = [[0.30, 0.55], [0.70, 0.81]]  # list of ranges for simultaneous fit [[start1, stop1],[start2, stop2],...]
constraint_expr = ['1.6329931618554523 * cen_1']  # list of string constraints for the fit, leave [] otherwise
# if provided, len(constraint_expr) should be equal to len(fit_range)-1
# 1.6329931618554523 = sqrt(8)/sqrt(3), ratio of 220 to 111 in FCC materials
constraint_var = ['cen']  # list of variable to be constrained for the fit, leave [] otherwise
# if provided, len(constraint_expr) should be equal to len(fit_range)-1

##################################
# end of user-defined parameters #
##################################

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
if xlim is None:
    xlim = [distances[np.unravel_index(distances[~np.isnan(average)].argmin(), distances.shape)],
            distances[np.unravel_index(distances[~np.isnan(average)].argmax(), distances.shape)]]
if ylim is None:
    if scale == 'linear':
        ylim = [0, average[~np.isnan(average)].max() * 2]
    else:
        ylim = [0, np.log10(average[~np.isnan(average)].max()) + 1]
ax.set_xlim(xlim[0], xlim[1])
ax.set_ylim(ylim[0], ylim[1])

##################################################
# combine ranges of interest in a single dataset #
##################################################
nb_ranges = len(fit_range)
nb_points = np.zeros(nb_ranges, dtype=int)
fit_range = np.asarray(fit_range)


for idx in range(nb_ranges):
    # find indices of distances belonging to ranges of interest
    myrange = fit_range[idx]
    ind_min, ind_max = util.find_nearest(distances, [myrange.min(), myrange.max()])
    # TODO: add an option in find_nearest() to find the nearest within 2 values
    nb_points[idx] = ind_max - ind_min + 1

# check if the number of points in ranges in the same, interpolate otherwise
max_points = nb_points.max()
combined_xaxis = []
combined_data = []
for idx in range(nb_ranges):
    # find indices of distances belonging to ranges of interest
    myrange = fit_range[idx]
    ind_min, ind_max = util.find_nearest(distances, [myrange.min(), myrange.max()])
    indices = np.arange(ind_min, ind_max+1, 1)
    if (ind_max-ind_min+1) != max_points:
        interp = interp1d(distances[indices], average[indices], kind='linear', bounds_error=True)
        interp_dist = np.linspace(distances[ind_min], distances[ind_max], num=max_points, endpoint=True)
        interp_data = interp(interp_dist)
        combined_xaxis.append(interp_dist)
        combined_data.append(interp_data)
    else:
        combined_xaxis.append(distances[indices])
        combined_data.append(average[indices])

combined_xaxis = np.asarray(combined_xaxis)
combined_data = np.asarray(combined_data)

##############################################
# fit user-defined range using the lineshape #
##############################################
# create nb_fit sets of parameters, one per data set
fit_params = Parameters()
for idx in range(nb_ranges):
    if lineshape == 'gaussian':
        cen = (fit_range[idx, 0] + fit_range[idx, 1]) / 2
        sig = abs(fit_range[idx, 0] - fit_range[idx, 1]) / 4
        fit_params.add('amp_%i' % (idx+1), value=10, min=0.0,  max=200)
        fit_params.add('cen_%i' % (idx+1), value=cen, min=cen-0.5,  max=cen+0.5)
        fit_params.add('sig_%i' % (idx+1), value=sig, min=sig/2, max=sig*2)

# constrain values
if len(constraint_expr) != 0:
    if len(constraint_expr) != (nb_ranges - 1) or len(constraint_var) != (nb_ranges - 1):
        print('Number of constraints or constrained variables incompatible with the number of ranges')
        sys.exit()
    for idx in range(1, nb_ranges):
        fit_params[constraint_var[idx-1]+'_%i' % (idx+1)].expr = constraint_expr[idx-1]

# run the global fit to all the data sets
result = minimize(util.objective_lmfit, fit_params, args=(combined_xaxis, combined_data, lineshape))
report_fit(result.params)

#####################
# plot data and fit #
#####################
fig, ax = plt.subplots(1, 1)
if scale == 'linear':
    ax.plot(distances, average, 'r')
else:
    ax.plot(distances, np.log10(average), 'r')
plt.legend(['data'])
for idx in range(nb_ranges):
    if lineshape == 'gaussian':
        y_fit = util.function_lmfit(params=result.params, iterator=idx, x_axis=distances, distribution=lineshape)
    if scale == 'linear':
        ax.plot(distances, y_fit, '-')
    else:
        ax.plot(distances, np.log10(y_fit), '-')
ax.set_xlim(xlim[0], xlim[1])
ax.set_ylim(ylim[0], ylim[1])
ax.set_xlabel('q (1/nm)')
ax.set_ylabel('Angular average (A.U.)')
fig.text(0.15, 0.90, 'cen_1 = ' + str('{:.5f}'.format(result.params['cen_1'].value)) + '+/-' +
         str('{:.5f}'.format(result.params['cen_1'].stderr)) +
         '   sig_1 = ' + str('{:.5f}'.format(result.params['sig_1'].value)) + '+/-' +
         str('{:.5f}'.format(result.params['sig_1'].stderr)), size=12)
fig.text(0.15, 0.80, lineshape + ' fit', size=12)
for idx in range(len(constraint_var)):
    fig.text(0.15, (0.75-0.1*idx), constraint_var[idx] + ' = ' + constraint_expr[idx], size=12)
fig.savefig(datadir + lineshape + ' fit.png')

plt.ioff()
plt.show()
