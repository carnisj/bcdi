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
Template for 3d isosurface figures of a diffraction pattern.

The diffraction pattern is supposed to be in an orthonormal frame and q values need to be provided.
"""

scan = 22    # spec scan number
root_folder = "D:/data/P10_August2019/data/"
sample_name = "gold_2_2_2_000"
comment = ""
binning = [3, 3, 3]  # binning for the measured diffraction pattern in each dimension
# tick_spacing = 50  # for plots, in nm
threshold_isosurface = 5  # log scale

#############
# load data #
#############
homedir = root_folder + sample_name + str(scan) + '/pynx/800_800_800_1_1_1/'
plt.ion()
root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename(initialdir=homedir, title="Select the diffraction pattern",
                                       filetypes=[("NPZ", "*.npz")])
data = np.load(file_path)
npz_key = data.files
data = data[npz_key[0]].astype(float)

if data.ndim != 3:
    print('a 3D array is expected')
    sys.exit()

nz, ny, nx = data.shape
print("Initial data size: (", nz, ',', ny, ',', nx, ')')

file_path = filedialog.askopenfilename(initialdir=homedir, title="Select q values",
                                       filetypes=[("NPZ", "*.npz")])
q_values = np.load(file_path)
qx = q_values['qx']  # 1D array
qy = q_values['qy']  # 1D array
qz = q_values['qz']  # 1D array

############
# bin data #
############
qx = qx[:nz - (nz % binning[0]):binning[0]]
qz = qz[:ny - (ny % binning[1]):binning[1]]
qy = qy[:nx - (nx % binning[2]):binning[2]]
data = pu.bin_data(data, (binning[0], binning[1], binning[2]), debugging=False)
nz, ny, nx = data.shape
print('Diffraction data shape after binning', data.shape)

#########################################
# plot 3D isosurface (perspective view) #
#########################################
data[data == 0] = np.nan
grid_qx, grid_qz, grid_qy = np.mgrid[qx.min():qx.max():1j * nz, qz.min():qz.max():1j * ny, qy.min():qy.max():1j*nx]
# in CXI convention, z is downstream, y vertical and x outboard
# for q: classical convention qx downstream, qz vertical and qy outboard
myfig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
mlab.contour3d(grid_qx, grid_qz, grid_qy, np.log10(data), contours=[threshold_isosurface], color=(0.7, 0.7, 0.7))

mlab.view(azimuth=165, elevation=135, distance=3.5*np.sqrt(grid_qx**2+grid_qz**2+grid_qy**2).max())
# azimut is the rotation around z axis of mayavi (x)
mlab.roll(0)

ax = mlab.axes(line_width=2.0, nb_labels=5)
mlab.savefig(homedir + 'S' + str(scan) + '_labels.png', figure=myfig)
ax.label_text_property.opacity = 0.0
ax.title_text_property.opacity = 0.0
mlab.savefig(homedir + 'S' + str(scan) + '.png', figure=myfig)
mlab.show()
