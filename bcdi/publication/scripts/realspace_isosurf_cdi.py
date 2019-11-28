# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np
from matplotlib import pyplot as plt
import mayavi
from mayavi import mlab
import tkinter as tk
from tkinter import filedialog
import sys
sys.path.append('//win.desy.de/home/carnisj/My Documents/myscripts/bcdi/')
# sys.path.append('C:/Users/Jerome/Documents/myscripts/bcdi/')
import bcdi.postprocessing.postprocessing_utils as pu

helptext = """
Template for 3d isosurface figures of a real space forward CDI reconstruction.

Open a CDI reconstruction file and save individual figures including a length scale.
"""

scan = 22    # spec scan number
root_folder = "D:/data/P10_August2019/data/"
sample_name = "gold_2_2_2_000"
comment = ""

voxel_size = 9.42  # in nm, supposed isotropic
tick_spacing = 200  # for plots, in nm
field_of_view = [1600, 1600, 1600]  # [z,y,x] in nm, can be larger than the total width (the array will be padded)
# the number of labels of mlab.axes() is an integer and is be calculated as: field_of_view[0]/tick_spacing
# therefore it is better to use an isotropic field_of_view
threshold_isosurface = 0.5

#############
# load data #
#############
homedir = root_folder + sample_name + str(scan) + '/pynx/1000_1000_1000_1_1_1/v1/'
plt.ion()
root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename(initialdir=homedir, title="Select reconstruction file",
                                       filetypes=[("NPZ", "*.npz"), ("CXI", "*.cxi"), ("HDF5", "*.h5")])
obj, _ = pu.load_reconstruction(file_path)

if obj.ndim != 3:
    print('a 3D reconstruction array is expected')
    sys.exit()

amp = abs(obj) / abs(obj).max()
amp[amp < threshold_isosurface] = 0

numz, numy, numx = amp.shape
print("Initial data size: (", numz, ',', numy, ',', numx, ')')

###################################################
#  pad arrays to obtain the desired field of view #
###################################################
z_pixel_FOV = int(np.rint((field_of_view[0] / voxel_size) / 2))  # half-number of pixels corresponding to the FOV
y_pixel_FOV = int(np.rint((field_of_view[1] / voxel_size) / 2))  # half-number of pixels corresponding to the FOV
x_pixel_FOV = int(np.rint((field_of_view[2] / voxel_size) / 2))  # half-number of pixels corresponding to the FOV
new_shape = [max(numz, 2*z_pixel_FOV), max(numy, 2*y_pixel_FOV), max(numx, 2*x_pixel_FOV)]
amp = pu.crop_pad(array=amp, output_shape=new_shape, debugging=False)
numz, numy, numx = amp.shape
print("Cropped/padded data size: (", numz, ',', numy, ',', numx, ')')

#########################################
# plot 3D isosurface (perspective view) #
#########################################
grid_qx, grid_qz, grid_qy = np.mgrid[0:2*z_pixel_FOV*voxel_size:voxel_size,
                                     0:2*y_pixel_FOV*voxel_size:voxel_size,
                                     0:2*x_pixel_FOV*voxel_size:voxel_size]
extent = [0, 2*z_pixel_FOV*voxel_size, 0, 2*y_pixel_FOV*voxel_size, 0, 2*x_pixel_FOV*voxel_size]
# in CXI convention, z is downstream, y vertical and x outboard

myfig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
mlab.contour3d(grid_qx, grid_qz, grid_qy,
               amp[numz // 2 - z_pixel_FOV:numz // 2 + z_pixel_FOV,
                   numy // 2 - y_pixel_FOV:numy // 2 + y_pixel_FOV,
                   numx // 2 - x_pixel_FOV:numx // 2 + x_pixel_FOV],
               contours=[threshold_isosurface], color=(0.7, 0.7, 0.7))
mlab.view(azimuth=0, elevation=50, distance=3*field_of_view[0])  # azimut is the rotation around z axis of mayavi (x)
mlab.roll(0)

ax = mlab.axes(extent=extent, line_width=2.0, nb_labels=int(1+field_of_view[0]/tick_spacing))
mlab.savefig(homedir + 'S' + str(scan) + '-perspective_labels.png', figure=myfig)
ax.label_text_property.opacity = 0.0
ax.title_text_property.opacity = 0.0
mlab.savefig(homedir + 'S' + str(scan) + '-perspective.png', figure=myfig)
mlab.close(myfig)
