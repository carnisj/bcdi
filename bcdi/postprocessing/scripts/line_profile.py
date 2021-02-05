# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import matplotlib.pyplot as plt
from numbers import Real
import numpy as np
import tkinter as tk
import sys
from tkinter import filedialog
sys.path.append('D:/myscripts/bcdi/')
import bcdi.graph.graph_utils as gu
import bcdi.utils.utilities as util
import bcdi.utils.validation as valid

helptext = """
This script allow to plot and save linecuts through a 2D or 3D object in function of a modulus threshold 
defining the object from the background. Must be given as input: the voxel size (possibly different in all directions), 
the direction of the cuts and a list of points where to apply the cut along this direction.   
"""

datadir = "D:/data/P10_2nd_test_isosurface_Dec2020/data_nanolab/"  # data folder
savedir = "D:/data/P10_2nd_test_isosurface_Dec2020/data_nanolab/linecuts/"
# results will be saved here, if None it will default to datadir
threshold = 0.2  # modulus threshold defining the normalized object from the background
binary = True  # True in order to perform the linecuts on a support (0 or 1) created from the thresholded object
direction = (0, 1, 0)  # tuple of 2 or 3 numbers (2 for 2D object, 3 for 3D) defining the direction of the cut
# in the orthonormal reference frame is given by the array axes. It will be corrected for anisotropic voxel sizes.
points = {(1, 2, 3)}  # list/tuple/set of 2 or 3 indices (2 for 2D object, 3 for 3D) corresponding to the points where
# the cut alond direction should be performed. The reference frame is given by the array axes.
voxel_size = 5  # positive real number  or tuple of 2 or 3 positive real number (2 for 2D object, 3 for 3D)
comment = ''  # string to add to the filename when saving
##################################
# end of user-defined parameters #
##################################

###############################
# list of colors for the plot #
###############################
colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')

#################
# load the data #
#################
plt.ion()
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(initialdir=datadir,
                                       filetypes=[("NPZ", "*.npz"), ("NPY", "*.npy"),
                                                  ("CXI", "*.cxi"), ("HDF5", "*.h5")])

obj, _ = util.load_file(file_path)
ndim = obj.ndim

#########################
# check some parameters #
#########################
if ndim not in {2, 3}:
    raise ValueError(f'Number of dimensions = {ndim}, expected 2 or 3')

valid.valid_container(direction, container_types=(list, tuple, np.ndarray), length=ndim, item_types=Real,
                      name='line_profile')

valid.valid_container(points, container_types=(list, tuple, set), min_length=1, name='line_profile')
for point in points:
    valid.valid_container(point, container_types=(list, tuple, np.ndarray), length=ndim, item_types=Real,
                          min_included=0, name='line_profile')
valid.valid_item(value=binary, allowed_types=bool, name='line_profile')

if isinstance(voxel_size, Real):
    voxel_size = (voxel_size,) * ndim
valid.valid_container(voxel_size, container_types=(list, tuple, np.ndarray), length=ndim, item_types=Real,
                      min_excluded=0, name='line_profile')

savedir = savedir or datadir

#################################################
# normalize the modulus and apply the threshold #
#################################################
obj = abs(obj) / abs(obj).max()  # normalize the modulus to 1
obj[np.isnan(obj)] = 0  # remove nans
obj[obj < threshold] = 0

#####################################
# create the linecut for each point #
#####################################
result = dict()
result['direction'] = direction
for point in points:
    # get the indices of all voxels belonging to the linecut
    indices = util.linecut(array_shape=obj.shape, point=point, direction=direction, voxel_size=voxel_size)
    # indices is a tuple of ndim ndarrays that can be used to directly slice obj

    cut_points = len(indices[0])
    x_axis = []
    for idx in range(cut_points):
        if ndim == 2:
            distance = np.sqrt(((indices[0][idx]-indices[0][0]) * voxel_size[0]) ** 2 +
                               ((indices[1][idx]-indices[1][0]) * voxel_size[1]) ** 2)
        else:  # 3D
            distance = np.sqrt(((indices[0][idx]-indices[0][0]) * voxel_size[0]) ** 2 +
                               ((indices[1][idx]-indices[1][0]) * voxel_size[1]) ** 2 +
                               ((indices[2][idx]-indices[2][0]) * voxel_size[2]) ** 2)
        x_axis.append(distance)

    cut = obj[indices]  # cut is now 1D

    # store the result in a dictionnary (cuts can have different lengths depending on the direction)
    result[point] = x_axis, cut

##############################
# save and plot the linecuts #
##############################
fig = plt.figure(figsize=(12, 9))
ax = plt.subplot(111)
plot_nb = 0
for point, (x_axis, cut) in result.items():
    col = colors[plot_nb % len(colors)]
    line, = ax.plot(x_axis, cut, colors[col])
    line.set_label(f'{point}')
    plot_nb += 1
ax.legend()

comment = f'cut_direction{direction[0]}_{direction[1]}_{direction[2]}_{comment}.npz'
np.savez_compressed(savedir + comment, result=result)
plt.ioff()
plt.show()
