# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np
from matplotlib import pyplot as plt
from mayavi import mlab
import tkinter as tk
from tkinter import filedialog
import sys
sys.path.append('//win.desy.de/home/carnisj/My Documents/myscripts/bcdi/')
import bcdi.postprocessing.postprocessing_utils as pu
import bcdi.graph.graph_utils as gu

helptext = """
Template for making 3d isosurface figures for paper
Open an npz file (reconstruction ampdispstrain.npz) and save individual figures including a length scale.
"""

scan = 1012    # spec scan number
root_folder = "D:/data/HC3207/"
sample_name = "SN"  # "S"  #
comment = ""

voxel_size = 6.0  # in nm, supposed isotropic
tick_spacing = 50  # for plots, in nm
field_of_view = 400  # in nm, can be larger than the total width (the array will be padded)

threshold_isosurface = 0.5

#############
# load data #
#############
homedir = root_folder + sample_name + str(scan) + '/'
plt.ion()
root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename(initialdir=homedir, title="Select reconstruction file",
                                       filetypes=[("NPZ", "*.npz")])
npzfile = np.load(file_path)
amp = npzfile['amp']
if amp.ndim != 3:
    print('a 3D reconstruction array is expected')
    sys.exit()

amp = amp / amp.max()
amp[amp < threshold_isosurface] = 0
amp[np.nonzero(amp)] = 1  # create a support for the isosurface

numz, numy, numx = amp.shape
print("Initial data size: (", numz, ',', numy, ',', numx, ')')

###################################################
#  pad arrays to obtain the desired field of view #
###################################################
pixel_spacing = tick_spacing / voxel_size
pixel_FOV = int(np.rint((field_of_view / voxel_size) / 2))  # half-number of pixels corresponding to the FOV
new_shape = [max(numz, 2*pixel_FOV), max(numy, 2*pixel_FOV), max(numx, 2*pixel_FOV)]
amp = pu.crop_pad(array=amp, output_shape=new_shape, debugging=False)
numz, numy, numx = amp.shape
print("Cropped/padded data size: (", numz, ',', numy, ',', numx, ')')

###################################
# plot 3D isosurface using mayavi #
###################################
grid_qx, grid_qz, grid_qy = np.mgrid[0:numz*voxel_size:voxel_size,
                                     0:numy*voxel_size:voxel_size,
                                     0:numx*voxel_size:voxel_size]
# in CXI convention, z is downstream, y vertical and x outboard

mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
mlab.contour3d(grid_qx, grid_qz, grid_qy, amp, contours=1, opacity=0.5, colormap="jet")
# mlab.colorbar(orientation="vertical", nb_labels=6)
mlab.outline(line_width=2.0)
# mlab.axes(ranges=[-0.1, 0.05, 2.71, 2.83, 0.05, 0.19], nb_labels=3, xlabel='z', ylabel='y', zlabel='x')  #
mlab.show()
