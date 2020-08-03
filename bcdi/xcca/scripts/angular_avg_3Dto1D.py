# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import sys
sys.path.append('D:/myscripts/bcdi/')
import bcdi.graph.graph_utils as gu
import bcdi.utils.utilities as util
import bcdi.xcca_utils as xcca
import bcdi.postprocessing.postprocessing_utils as pu

helptext = """
Plot a 1D angular average of a 3D reciprocal space map, based on the position of the origin (direct beam or Bragg peak). 

If q values are provided, the data can be in an orthonormal frame or not (detector frame in Bragg CDI). The unit
expected for q values is 1/nm.

If q values are not provided, the data is supposed to be in an orthonormal frame.
"""

root_folder = 'D:/data/P10_August2019_CDI/data/gold_2_2_2_00022/pynx/1000_1000_1000_1_1_1/'
load_qvalues = True  # True if the q values are provided
load_mask = True  # True to load a mask, masked points are not used for angular average
origin = [np.nan, np.nan, np.nan]  # [np.nan, np.nan, np.nan] #  # if np.nan, the origin is set at the center
bin_factor = 2  # the data will be binned by bin_factor is the three directions
vertical_lines = [0.104, 0.144, 0.172, 0.208]  # plot vertical dashed lines at these q values, leave [] otherwise
# position in pixels of the origin of the angular average in the array.
# if a nan value is used, the origin will be set at the middle of the array in the corresponding dimension.
threshold = 0  # data < threshold will be set to 0
debug = False  # True to show more plots
xlim = None  # limits used for the horizontal axis of the angular plot, leave None otherwise
ylim = None  # limits used for the vertical axis of++ plots, leave None otherwise
save_txt = True  # True to save q values and the average in .txt format
##########################
# end of user parameters #
##########################

###################
# define colormap #
###################
colormap = gu.Colormap()
my_cmap = colormap.cmap

##############################
# load reciprocal space data #
##############################
plt.ion()
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(initialdir=root_folder, title="Select the diffraction pattern",
                                       filetypes=[("NPZ", "*.npz")])
npzfile = np.load(file_path)
diff_pattern = pu.bin_data(npzfile[list(npzfile.files)[0]].astype(int), (bin_factor, bin_factor, bin_factor),
                           debugging=False)
diff_pattern[diff_pattern < threshold] = 0
nz, ny, nx = diff_pattern.shape
print('Data shape after binning:', nz, ny, nx)
gu.multislices_plot(diff_pattern, sum_frames=True, plot_colorbar=True, cmap=my_cmap,
                    title='diffraction pattern', scale='log', vmin=np.nan, vmax=np.nan,
                    reciprocal_space=True, is_orthogonal=True)
#############
# load mask #
#############
if load_mask:
    file_path = filedialog.askopenfilename(initialdir=root_folder, title="Select the mask",
                                           filetypes=[("NPZ", "*.npz")])
    npzfile = np.load(file_path)
    mask = pu.bin_data(npzfile[list(npzfile.files)[0]], (bin_factor, bin_factor, bin_factor), debugging=False)
else:
    mask = None
#######################
# check origin values #
#######################
if np.isnan(origin[0]):
    origin[0] = int(nz // 2)
if np.isnan(origin[1]):
    origin[1] = int(ny // 2)
if np.isnan(origin[2]):
    origin[2] = int(nx // 2)

#################
# load q values #
#################
if load_qvalues:
    file_path = filedialog.askopenfilename(initialdir=root_folder, title="Select q values",
                                           filetypes=[("NPZ", "*.npz")])
    npzfile = np.load(file_path)
    qx = npzfile['qx'][::bin_factor]  # downstream
    qz = npzfile['qz'][::bin_factor]  # vertical up
    qy = npzfile['qy'][::bin_factor]  # outboard
else:  # work with pixels, supposing that the data is in an orthonormal frame
    qx = np.arange(nz) - origin[0]
    qz = np.arange(ny) - origin[1]
    qy = np.arange(nx) - origin[2]

q_axis, y_mean_masked, y_median_masked = xcca.angular_avg(data=diff_pattern, q_values=(qx, qz, qy), origin=origin,
                                                          mask=mask, debugging=debug)
#############
# save data #
#############
np.savez_compressed(root_folder + 'q+angular_avg.npz', q=q_axis, avg=y_mean_masked, median=y_median_masked)
if save_txt:
    file = open(root_folder + 'q+angular_avg.txt', "w")
    file.write('{:8s}'.format('q') + '\t' + '{:10s}'.format('avg') + '\n')
    for idx in range(len(q_axis)):
        file.write('{:8.6f}'.format(q_axis[idx]) + '\t' + '{:10.1f}'.format(y_mean_masked[idx]) + '\n')
    file.close()

#############
# plot data #
#############
q_vline = util.find_nearest(q_axis, vertical_lines)

fig, ax0 = plt.subplots(1, 1)
plt0 = ax0.plot(q_axis, np.log10(y_mean_masked), 'r')
plt.xlabel('q (1/nm)')
plt.ylabel('Angular average (A.U.)')
if xlim is not None:
    plt.xlim(xlim[0], xlim[1])
if ylim is not None:
    ymin = ylim[0]
    plt.ylim(ylim[0], ylim[1])
else:
    ymin, _ = ax0.get_ylim()
for counter, value in enumerate(vertical_lines):
    ax0.vlines(x=value, ymin=ymin, ymax=np.log10(y_mean_masked[q_vline[counter]]),
               colors='r', linestyle='dashed')
plt.savefig(root_folder + 'angular_avg_labels.png')
ax0.tick_params(labelbottom=False, labelleft=False)
plt.xlabel('')
plt.ylabel('')
plt.savefig(root_folder + 'angular_avg.png')

fig, ax = plt.subplots(1, 1)
ax.plot(q_axis, np.log10(y_mean_masked), 'r', label='mean')
ax.plot(q_axis, np.log10(y_median_masked), 'b', label='median')
ax.set_xlabel('q (1/nm)')
ax.set_ylabel('Angular average (A.U.)')
ax.legend()

plt.ioff()
plt.show()
