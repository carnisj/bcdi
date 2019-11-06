# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import gc
import sys
sys.path.append('//win.desy.de/home/carnisj/My Documents/myscripts/bcdi/')
import bcdi.graph.graph_utils as gu

helptext = """
Plot a 1D radial average of a 3D reciprocal space map, based on the position of the origin (direct beam or Bragg peak). 

If q values are provided, the data can be in an orthonormal frame or not (detector frame in Bragg CDI). The unit
expected for q values is 1/nm.

If q values are not provided, the data is supposed to be in an orthonormal frame.
"""

root_folder = 'D:/data/P10_August2019/data/gold_2_2_2_00022/pynx/800_800_800_1_1_1/'
load_qvalues = True  # True if the q values are provided
load_mask = True  # True to load a mask, masked points are not used for radial average
origin = [np.nan, np.nan, np.nan]  # position in pixels of the origin of the radial average in the array.
# if a nan value is used, the origin will be set at the middle of the array in the corresponding dimension.
debug = False  # True to show more plots
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
diff_pattern = npzfile[list(npzfile.files)[0]]
nz, ny, nx = diff_pattern.shape

#############
# load mask #
#############
if load_mask:
    file_path = filedialog.askopenfilename(initialdir=root_folder, title="Select the mask",
                                           filetypes=[("NPZ", "*.npz")])
    npzfile = np.load(file_path)
    mask = npzfile[list(npzfile.files)[0]]
    diff_pattern[np.nonzero(mask)] = np.nan
    del mask
    gc.collect()

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
    qx = npzfile['qx']  # downstream
    qz = npzfile['qz']  # vertical up
    qy = npzfile['qy']  # outboard
else:  # work with pixels, supposing that the data is in an orthonormal frame
    qx = np.arange(nz) - origin[0]
    qz = np.arange(ny) - origin[1]
    qy = np.arange(nx) - origin[2]

qxCOM = qx[origin[0]]
qzCOM = qz[origin[1]]
qyCOM = qy[origin[2]]

############################
# calculate ditance matrix #
############################
distances = np.sqrt((qx[:, np.newaxis, np.newaxis] - qxCOM)**2 +
                    (qz[np.newaxis, :, np.newaxis] - qzCOM)**2 +
                    (qy[np.newaxis, np.newaxis, :] - qyCOM)**2)
if debug:
    gu.multislices_plot(distances, sum_frames=False, plot_colorbar=True, cmap=my_cmap,
                        title='distances_q', scale='linear', invert_yaxis=False, vmin=np.nan, vmax=np.nan,
                        reciprocal_space=True, is_orthogonal=True)

#################################
# average over spherical shells #
#################################
print('Distance max:', distances.max(), ' (1/nm) at voxel:', np.unravel_index(abs(distances).argmax(), distances.shape))
print('Distance:', distances[origin[0], origin[1], origin[2]], ' (1/nm) at voxel:', origin)
nb_bins = int(nz // 3)
radial_avg = np.zeros(nb_bins)
dq = distances.max() / nb_bins  # in 1/A
q_axis = np.linspace(0, distances.max(), endpoint=True, num=nb_bins+1)  # in pixels or 1/nm

for index in range(nb_bins):
    logical_array = np.logical_and((distances < q_axis[index+1]), (distances >= q_axis[index]))
    temp = diff_pattern[logical_array]
    radial_avg[index] = temp[~np.isnan(temp)].mean()
q_axis = q_axis[:-1]

del diff_pattern
gc.collect()

################################
# plot and save the 1D average #
################################
plt.figure()
plt.plot(q_axis, np.log10(radial_avg[~np.isnan(radial_avg)]), '.r')
plt.xlabel('q (1/nm)')
plt.ylabel('radial average')
plt.savefig(root_folder + 'radial_avg.png')
plt.ioff()
plt.show()
