# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os
import sys
sys.path.append('D:/myscripts/bcdi/')
import bcdi.graph.graph_utils as gu
import bcdi.postprocessing.postprocessing_utils as pu

helptext = """
# TODO refactor
"""

scan = 1301  # spec scan number
root_folder = "D:/data/SIXS_2019_Ni/"
sample_name = "S"  # "S"
datadir = root_folder + sample_name + str(scan) + "/pynxraw/"
support_threshold = 0.30  # threshold applied to the modulus for determining the support
normalize = False  # if True, will normalize the histograms to the respective number of points
plot_scale = 'linear'  # 'log' or 'linear', Y scale for the histograms
xlim = [-0.002, 0.002]  # limits used for the horizontal axis of histograms, leave None otherwise
ylim = None  # limits used for the vertical axis of histograms, leave None otherwise
save_txt = False  # True to save the strain values for the surface, the bulk and the full support in txt files
debug = True  # True to see more plots
##########################
# end of user parameters #
##########################

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

###################################################
# plot the strain histogram for the surface layer #
###################################################
nb_surface = len(np.nonzero(surface)[0])
print("Number of surface points = ", str(nb_surface))
print('Min surface strain = {:.5f}'.format(strain[np.nonzero(surface)].min()))
print('Max surface strain = {:.5f}'.format(strain[np.nonzero(surface)].max()))
hist, bin_edges = np.histogram(strain[np.nonzero(surface)], bins=100)
hist = hist.astype(float)
if normalize:
    hist = hist / nb_surface  # normalize the histogram to the number of points
hist[hist == 0] = np.nan
x_axis = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2
plt.figure()
if plot_scale == 'log':
    plt.plot(x_axis, np.log10(hist))
else:
    plt.plot(x_axis, hist)
if xlim is None:
    plt.xlim(-max(abs(x_axis)), max(abs(x_axis)))
else:
    assert len(xlim) == 2, 'xlim=[min, max] expected'
    plt.xlim(xlim[0], xlim[1])
if ylim is not None:
    assert len(ylim) == 2, 'ylim=[min, max] expected'
    plt.ylim(ylim[0], ylim[1])

plt.title('Histogram of the strain for {:d} surface points'.format(nb_surface)
          + "\nModulus threshold="+str(support_threshold))
plt.pause(0.1)

##########################################
# plot the strain histogram for the bulk #
##########################################
nb_bulk = len(np.nonzero(bulk)[0])
print("Number of bulk points = ", str(nb_bulk))
print('Min bulk strain = {:.5f}'.format(strain[np.nonzero(bulk)].min()))
print('Max bulk strain = {:.5f}'.format(strain[np.nonzero(bulk)].max()))
hist, bin_edges = np.histogram(strain[np.nonzero(bulk)], bins=100)
hist = hist.astype(float)
if normalize:
    hist = hist / nb_bulk  # normalize the histogram to the number of points
hist[hist == 0] = np.nan
x_axis = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2
plt.figure()
if plot_scale == 'log':
    plt.plot(x_axis, np.log10(hist))
else:
    plt.plot(x_axis, hist)
if xlim is None:
    plt.xlim(-max(abs(x_axis)), max(abs(x_axis)))
else:
    assert len(xlim) == 2, 'xlim=[min, max] expected'
    plt.xlim(xlim[0], xlim[1])
if ylim is not None:
    assert len(ylim) == 2, 'ylim=[min, max] expected'
    plt.ylim(ylim[0], ylim[1])
plt.title('Histogram of the strain for {:d} bulk points'.format(nb_bulk)
          + "\nModulus threshold="+str(support_threshold))
plt.pause(0.1)

nb_total = len(np.nonzero(support)[0])
print("Sanity check: Total points = {:d}".format(nb_total), ", surface+bulk = {:d}".format(nb_surface+nb_bulk))
plt.ioff()
plt.show()
