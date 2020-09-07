# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np
from matplotlib import pyplot as plt
from lmfit import minimize, Parameters, report_fit
import tkinter as tk
from tkinter import filedialog
import os
import sys
sys.path.append('D:/myscripts/bcdi/')
import bcdi.graph.graph_utils as gu
import bcdi.utils.utilities as util
import bcdi.postprocessing.postprocessing_utils as pu

helptext = """
Extract the surface voxel layer of an object recontructed by BCDI phase retrieval and plot histograms of the strain at
the surface and in the remaining bulk.

Input: a .npz file containing fields 'amp' and 'strain' (e.g., S1301_amp_disp_strain.npz)
"""

scan = 78  # spec scan number
root_folder = "D:/data/Pt THH ex-situ/Data/HS4670/"
sample_name = "S"  # "S"
datadir = root_folder + sample_name + str(scan) + "/pynxraw/"
support_threshold = 0.25  # threshold applied to the modulus for reading the surface strain
normalize = True  # if True, will normalize the histograms to the respective number of points
bin_number = 1000  # number of bins between strain_min and strain_max
plot_scale = 'linear'  # 'log' or 'linear', Y scale for the histograms
xlim = [-0.002, 0.002]  # limits used for the horizontal axis of histograms, leave None otherwise
ylim = None  # limits used for the vertical axis of histograms, leave None otherwise
fit_pdf = 'skewed_gaussian'  # 'pseudovoigt' or 'skewed_gaussian'
save_txt = False  # True to save the strain values for the surface, the bulk and the full support in txt files
debug = True  # True to see more plots
##########################
# end of user parameters #
##########################

#########################
# check some parameters #
#########################
assert fit_pdf in ['pseudovoigt', 'skewed_gaussian'], 'invalid value for fit_pdf parameter'


###################
# define colormap #
###################
colormap = gu.Colormap()
my_cmap = colormap.cmap

#############
# load data #
#############
plt.ion()
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(initialdir=datadir, filetypes=[("NPZ", "*.npz")])
npzfile = np.load(file_path)
amp = npzfile['amp']
strain = npzfile['strain']
nbz, nby, nbx = amp.shape

######################################
# define the support, surface & bulk #
######################################
support = np.zeros(amp.shape)
support[amp > support_threshold*amp.max()] = 1
coordination_matrix = pu.calc_coordination(support=support, kernel=np.ones((3, 3, 3)), debugging=debug)
surface = np.copy(support)
surface[coordination_matrix > 22] = 0  # remove the bulk 22
bulk = support - surface
if debug:
    gu.multislices_plot(surface, sum_frames=False, plot_colorbar=True, cmap=my_cmap,
                        title='Surface layer', scale='linear', is_orthogonal=True,
                        reciprocal_space=False)
    gu.multislices_plot(bulk, sum_frames=False, plot_colorbar=True, cmap=my_cmap,
                        title='Bulk', scale='linear', is_orthogonal=True,
                        reciprocal_space=False)

##########################
# save the strain values #
##########################
if save_txt:
    file_surface = open(os.path.join(datadir, "S" + str(scan) +
                                     "_threshold" + str(support_threshold) + "_surface.dat"), "w")
    file_bulk = open(os.path.join(datadir, "S" + str(scan) +
                                  "_threshold" + str(support_threshold) + "_bulk.dat"), "w")
    file_total = open(os.path.join(datadir, "S" + str(scan) +
                                   "_threshold" + str(support_threshold) + "_bulk+surface.dat"), "w")
    # write surface points position / strain to file
    surface_indices = np.nonzero(surface)
    nb_surface = len(surface_indices[0])
    ind_z = surface_indices[0]
    ind_y = surface_indices[1]
    ind_x = surface_indices[2]
    for point in range(nb_surface):
        file_surface.write(
            '{0: <10}'.format(str('{:.7f}'.format(strain[ind_z[point], ind_y[point], ind_x[point]]))) + '\n')

    # write bulk points position / strain to file
    bulk_indices = np.nonzero(bulk)
    nb_bulk = len(bulk_indices[0])
    ind_z = bulk_indices[0]
    ind_y = bulk_indices[1]
    ind_x = bulk_indices[2]
    for point in range(nb_bulk):
        file_bulk.write(
            '{0: <10}'.format(str('{:.7f}'.format(strain[ind_z[point], ind_y[point], ind_x[point]]))) + '\n')
    file_surface.close()
    file_bulk.close()

    # write all points position / strain to file
    total_indices = np.nonzero(support)
    nb_total = len(total_indices[0])
    ind_z = total_indices[0]
    ind_y = total_indices[1]
    ind_x = total_indices[2]
    for point in range(nb_total):
        file_total.write(
            '{0: <10}'.format(str('{:.7f}'.format(strain[ind_z[point], ind_y[point], ind_x[point]]))) + '\n')

    file_surface.close()
    file_bulk.close()
    file_total.close()

####################################
# fit the bulk strain distribution #
####################################
nb_surface = len(np.nonzero(surface)[0])
print("Number of surface points = ", str(nb_surface))
print('Min surface strain = {:.5f}'.format(strain[np.nonzero(surface)].min()))
print('Max surface strain = {:.5f}'.format(strain[np.nonzero(surface)].max()))
hist, bin_edges = np.histogram(strain[np.nonzero(surface)], bins=bin_number)
hist = hist.astype(float)
if normalize:
    hist = hist / nb_surface  # normalize the histogram to the number of points

x_axis = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2

fit_params = Parameters()
if fit_pdf == 'skewed_gaussian':
    fit_params.add('amp_1', value=0.005, min=0.000001, max=100000)
    fit_params.add('loc_1', value=0, min=-0.1, max=0.1)
    fit_params.add('sig_1', value=0.0005, min=0.0000001, max=0.1)
    fit_params.add('alpha_1', value=0, min=-10, max=10)
else:  # 'pseudovoigt'
    fit_params.add('amp_1', value=0.005, min=0.000001, max=100000)
    fit_params.add('cen_1', value=0, min=-0.1, max=0.1)
    fit_params.add('sig_1', value=0.0005, min=0.0000001, max=0.1)
    fit_params.add('ratio_1', value=0.5, min=0, max=1)

# run the global fit to all the data sets
result = minimize(util.objective_lmfit, fit_params, args=(x_axis, hist, fit_pdf))
report_fit(result.params)
strain_fit = util.function_lmfit(params=result.params, x_axis=x_axis, distribution=fit_pdf)

if fit_pdf == 'skewed_gaussian':  # find the position of the mode (maximum of the pdf)
    x_mode = np.unravel_index(strain_fit.argmax(), x_axis.shape)
    step = x_axis[x_mode] - x_axis[x_mode[0]-1]
    fine_x = np.copy(x_axis)
    for idx in range(2):
        fine_x = np.linspace(fine_x[x_mode] - step, fine_x[x_mode] + step, endpoint=True, num=1000)
        fine_y = util.function_lmfit(params=result.params, x_axis=fine_x, distribution=fit_pdf)
        diff_fit = np.gradient(fine_y, fine_x[1] - fine_x[0])
        x_mode, = np.unravel_index(abs(diff_fit).argmin(), fine_x.shape)
        step = fine_x[x_mode] - fine_x[x_mode - 1]
    strain_mode = fine_x[x_mode]

###################################################
# plot the strain histogram for the surface layer #
###################################################
fig, ax = plt.subplots(nrows=1, ncols=1)
if plot_scale == 'log':
    hist[hist == 0] = np.nan
    ax.plot(x_axis, np.log10(hist), linestyle='', marker='o', markeredgecolor='r', fillstyle='none')
    fit, = ax.plot(x_axis, np.log10(strain_fit), linestyle='-', color='r')
else:
    ax.plot(x_axis, hist, linestyle='', marker='o', markeredgecolor='r', fillstyle='none')
    fit, = ax.plot(x_axis, strain_fit, linestyle='-', color='r')
if xlim is None:
    ax.set_xlim(-max(abs(x_axis)), max(abs(x_axis)))
else:
    assert len(xlim) == 2, 'xlim=[min, max] expected'
    ax.set_xlim(xlim[0], xlim[1])
if ylim is not None:
    assert len(ylim) == 2, 'ylim=[min, max] expected'
    ax.set_.ylim(ylim[0], ylim[1])
ax.set_xlabel('strain')
vline1 = ax.axvline(x=0, ymin=0, ymax=1, color='k', linestyle='dotted', linewidth=1.0)
vline2 = ax.axvline(x=np.mean(strain[np.nonzero(surface)]), ymin=0, ymax=1, color='r', linestyle='dashed')
legend_fit = ax.legend(handles=[fit], labels=[fit_pdf], loc='upper left', frameon=False)
ax.legend(handles=(vline1, vline2), labels=('strain=0', '<surface>'), loc='upper right', frameon=False)
ax.add_artist(legend_fit)
ax.set_title('S{:d} histogram of the strain for {:d} surface points'.format(scan, nb_surface)
             + "\nModulus threshold="+str(support_threshold))
fig.text(0.65, 0.70, '<strain>={:.2e}'.format(np.mean(strain[np.nonzero(surface)])))
fig.text(0.65, 0.65, 'std(strain)={:.2e}'.format(np.std(strain[np.nonzero(surface)])))

if fit_pdf == 'skewed_gaussian':
    fig.text(0.13, 0.76, 'SK_max @ strain={:.2e}'.format(strain_mode))
    fig.text(0.13, 0.66, 'SK std={:.2e}\n   +/-{:.2e}'.format(result.params['sig_1'].value,
                                                              result.params['sig_1'].stderr))
else:
    fig.text(0.15, 0.70, 'PDF center={:.2e}\n   +/-{:.2e}'.format(result.params['cen_1'].value,
                                                                  result.params['cen_1'].stderr))
    fig.text(0.15, 0.60, 'PDF std={:.2e}\n   +/-{:.2e}'.format(result.params['sig_1'].value,
                                                               result.params['sig_1'].stderr))
    fig.text(0.15, 0.50, 'PDF ratio={:.2e}\n   +/-{:.2e}'.format(result.params['ratio_1'].value,
                                                                 result.params['ratio_1'].stderr))
plt.pause(0.1)
fig.savefig(datadir + 'surface_strain_iso' + str(support_threshold)+'.png')

####################################
# fit the bulk strain distribution #
####################################
nb_bulk = len(np.nonzero(bulk)[0])
print("Number of bulk points = ", str(nb_bulk))
print('Min bulk strain = {:.5f}'.format(strain[np.nonzero(bulk)].min()))
print('Max bulk strain = {:.5f}'.format(strain[np.nonzero(bulk)].max()))
hist, bin_edges = np.histogram(strain[np.nonzero(bulk)], bins=bin_number)
hist = hist.astype(float)
if normalize:
    hist = hist / nb_bulk  # normalize the histogram to the number of points

x_axis = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2

fit_params = Parameters()
if fit_pdf == 'skewed_gaussian':
    fit_params.add('amp_1', value=0.001, min=0.00000001, max=100000)
    fit_params.add('loc_1', value=0, min=-0.1, max=0.1)
    fit_params.add('sig_1', value=0.0005, min=0.0000001, max=0.1)
    fit_params.add('alpha_1', value=0, min=-10, max=10)
else:  # 'pseudovoigt'
    fit_params.add('amp_1', value=0.001, min=0.00000001, max=100000)
    fit_params.add('cen_1', value=0, min=-0.1, max=0.1)
    fit_params.add('sig_1', value=0.0005, min=0.0000001, max=0.1)
    fit_params.add('ratio_1', value=0.5, min=0, max=1)

# run the global fit to all the data sets
result = minimize(util.objective_lmfit, fit_params, args=(x_axis, hist, fit_pdf))
report_fit(result.params)
strain_fit = util.function_lmfit(params=result.params, x_axis=x_axis, distribution=fit_pdf)

if fit_pdf == 'skewed_gaussian':  # find the position of the mode (maximum of the pdf)
    x_mode = np.unravel_index(strain_fit.argmax(), x_axis.shape)
    step = x_axis[x_mode] - x_axis[x_mode[0]-1]
    fine_x = np.copy(x_axis)
    for idx in range(2):
        fine_x = np.linspace(fine_x[x_mode] - step, fine_x[x_mode] + step, endpoint=True, num=1000)
        fine_y = util.function_lmfit(params=result.params, x_axis=fine_x, distribution=fit_pdf)
        diff_fit = np.gradient(fine_y, fine_x[1] - fine_x[0])
        x_mode, = np.unravel_index(abs(diff_fit).argmin(), fine_x.shape)
        step = fine_x[x_mode] - fine_x[x_mode - 1]
    strain_mode = fine_x[x_mode]

##########################################
# plot the strain histogram for the bulk #
##########################################
fig, ax = plt.subplots(nrows=1, ncols=1)
if plot_scale == 'log':
    hist[hist == 0] = np.nan
    ax.plot(x_axis, np.log10(hist), linestyle='', marker='o', markeredgecolor='b', fillstyle='none')
    fit, = ax.plot(x_axis, np.log10(strain_fit), linestyle='-', color='b')
else:
    ax.plot(x_axis, hist, linestyle='', marker='o', markeredgecolor='b', fillstyle='none')
    fit, = ax.plot(x_axis, strain_fit, linestyle='-', color='b')
if xlim is None:
    ax.set_xlim(-max(abs(x_axis)), max(abs(x_axis)))
else:
    assert len(xlim) == 2, 'xlim=[min, max] expected'
    ax.set_xlim(xlim[0], xlim[1])
if ylim is not None:
    assert len(ylim) == 2, 'ylim=[min, max] expected'
    ax.set_ylim(ylim[0], ylim[1])
ax.set_xlabel('strain')
vline1 = ax.axvline(x=0, ymin=0, ymax=1, color='k', linestyle='dotted', linewidth=1.0)
vline2 = ax.axvline(x=np.mean(strain[np.nonzero(bulk)]), ymin=0, ymax=1, color='b', linestyle='dashed')
legend_fit = ax.legend(handles=[fit], labels=[fit_pdf], loc='upper left', frameon=False)
ax.legend(handles=(vline1, vline2), labels=('strain=0', '<bulk>'), loc='upper right', frameon=False)
ax.add_artist(legend_fit)
ax.set_title('S{:d} histogram for {:d} bulk points'.format(scan, nb_bulk)
             + "\nModulus threshold="+str(support_threshold))
fig.text(0.65, 0.70, '<strain>={:.2e}'.format(np.mean(strain[np.nonzero(bulk)])))
fig.text(0.65, 0.65, 'std(strain)={:.2e}'.format(np.std(strain[np.nonzero(bulk)])))

if fit_pdf == 'skewed_gaussian':
    fig.text(0.13, 0.76, 'SK_max @ strain={:.2e}'.format(strain_mode))
    fig.text(0.13, 0.66, 'SK std={:.2e}\n   +/-{:.2e}'.format(result.params['sig_1'].value,
                                                              result.params['sig_1'].stderr))
else:
    fig.text(0.15, 0.70, 'PDF center={:.2e}\n   +/-{:.2e}'.format(result.params['cen_1'].value,
                                                                  result.params['cen_1'].stderr))
    fig.text(0.15, 0.60, 'PDF std={:.2e}\n   +/-{:.2e}'.format(result.params['sig_1'].value,
                                                               result.params['sig_1'].stderr))
    fig.text(0.15, 0.50, 'PDF ratio={:.2e}\n   +/-{:.2e}'.format(result.params['ratio_1'].value,
                                                                 result.params['ratio_1'].stderr))
plt.pause(0.1)
fig.savefig(datadir + 'bulk_strain_iso' + str(support_threshold)+'.png')

nb_total = len(np.nonzero(support)[0])
print("Sanity check: Total points = {:d}".format(nb_total), ", surface+bulk = {:d}".format(nb_surface+nb_bulk))
plt.ioff()
plt.show()
