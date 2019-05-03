# -*- coding: utf-8 -*-
"""
Calculate the diffraction pattern using a kinematic sum.
You need to use pynx as source
"""
import numpy as np
from matplotlib import pyplot as plt
# from pynx import scattering
from scipy.interpolate import RegularGridInterpolator
from matplotlib.colors import LinearSegmentedColormap
import tkinter as tk
from tkinter import filedialog

datadir = "C:/Users/carnis/Work Folders/Documents/data/CH4760_Pt/S2227/pynxraw/"
photon_number = 5e7  # total number of photons in the array, usually around 5e7
voxel_size = 3  # in nm, voxel size of the reconstruction, should be eaqual in each direction
pad_size = [1000, 1000, 1000]  # size of the array for kinematic sum calculation
########################################
# define a colormap
cdict = {'red':  ((0.0, 1.0, 1.0),
                  (0.11, 0.0, 0.0),
                  (0.36, 0.0, 0.0),
                  (0.62, 1.0, 1.0),
                  (0.87, 1.0, 1.0),
                  (1.0, 0.0, 0.0)),
         'green': ((0.0, 1.0, 1.0),
                   (0.11, 0.0, 0.0),
                   (0.36, 1.0, 1.0),
                   (0.62, 1.0, 1.0),
                   (0.87, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),
         'blue': ((0.0, 1.0, 1.0),
                  (0.11, 1.0, 1.0),
                  (0.36, 1.0, 1.0),
                  (0.62, 0.0, 0.0),
                  (0.87, 0.0, 0.0),
                  (1.0, 0.0, 0.0))}
my_cmap = LinearSegmentedColormap('my_colormap', cdict, 256)
########################################
########################################
# import support - should be centered
########################################
plt.ion()
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(initialdir=datadir, filetypes=[("NPZ", "*.npz")])
npzfile = np.load(file_path)

obj = npzfile['obj']
nz, ny, nx = obj.shape

plt.figure(figsize=(18, 15), num=1)
plt.subplot(2, 2, 1)
plt.imshow(abs(obj.sum(axis=2)), cmap=my_cmap)
plt.colorbar()
plt.axis('scaled')
plt.title('Support before interpolation in YZ')
plt.subplot(2, 2, 2)
plt.imshow(abs(obj.sum(axis=1)), cmap=my_cmap)
plt.colorbar()
plt.axis('scaled')
plt.title('Support before interpolation in XZ')
plt.subplot(2, 2, 3)
plt.imshow(abs(obj.sum(axis=0)), cmap=my_cmap)
plt.colorbar()
plt.axis('scaled')
plt.title('Support before interpolation in XY')

# # reduce size object
# # FIND COM in three directions
# y = obj.sum(axis=2).sum(axis=1)
# x = np.arange(0, 400, 1)
# xCOM = sum(x * y) / sum(y)
# y = obj.sum(axis=2).sum(axis=0)
# yCOM = sum(x * y) / sum(y)
# y = obj.sum(axis=0).sum(axis=0)
# zCOM = sum(x * y) / sum(y)
#
# newobj = np.zeros((200, 200, 200))
# newobj = obj[int(xCOM)-100:int(xCOM)+100, int(yCOM)-100:int(yCOM)+100, int(zCOM)-100:int(zCOM)+100]
#
# plt.figure(num=2)
# plt.subplot(2, 2, 1)
# plt.imshow(newobj.sum(axis=0))
# ax = plt.gca()
# ax.invert_yaxis()
# plt.title("sum in XY")
# plt.subplot(2, 2, 2)
# plt.imshow(newobj.sum(axis=1))
# plt.title("sum in XZ")
# plt.subplot(2, 2, 3)
# plt.imshow(newobj.sum(axis=2))
# plt.title("sum in YZ")
# plt.pause(0.1)

###################################################
# Interpolate the support to keep the real space voxel size constant
###################################################
nbz, nby, nbx = pad_size
rgi = RegularGridInterpolator((np.arange(-nz // 2, nz // 2) * voxel_size * nbz / nz,
                               np.arange(-ny // 2, ny // 2) * voxel_size * nbz / nz,
                               np.arange(-nx // 2, nx // 2) * voxel_size * nbz / nz),
                              obj, method='linear', bounds_error=False, fill_value=0)

myz, myy, myx = np.meshgrid(np.arange(-nz // 2, nz // 2, 1) * voxel_size,
                            np.arange(-ny // 2, ny // 2, 1) * voxel_size,
                            np.arange(-nx // 2, nx // 2, 1) * voxel_size, indexing='ij')

new_obj = rgi(np.concatenate((myz.reshape((1, myz.size)), myy.reshape((1, myy.size)),
                              myx.reshape((1, myx.size)))).transpose())
new_obj = new_obj.reshape((nz, ny, nx)).astype(obj.dtype)

plt.figure(figsize=(18, 15), num=2)
plt.subplot(2, 2, 1)
plt.imshow(abs(new_obj.sum(axis=2)), cmap=my_cmap)
plt.colorbar()
plt.axis('scaled')
plt.title('Support after interpolation in YZ')
plt.subplot(2, 2, 2)
plt.imshow(abs(new_obj.sum(axis=1)), cmap=my_cmap)
plt.colorbar()
plt.axis('scaled')
plt.title('Support after interpolation in XZ')
plt.subplot(2, 2, 3)
plt.imshow(abs(new_obj.sum(axis=0)), cmap=my_cmap)
plt.colorbar()
plt.axis('scaled')
plt.title('Support after interpolation in XY')

###################################################
# Create array of 3D coordinates, 200x200x200 cells
###################################################
nx, ny, nz = 200, 200, 200
x = np.arange(0, nx, dtype=np.float32)
y = np.arange(0, ny, dtype=np.float32)[:, np.newaxis]
z = np.arange(0, nz, dtype=np.float32)[:, np.newaxis, np.newaxis]
occ = obj

###############################################################
# HKL coordinates
dh = 1
nh, nk, nl = 128, 128, 128
h = np.arange(-dh, dh, 2*dh/nh) + 1
k = np.arange(-dh, dh, 2*dh/nk)[:, np.newaxis] + 1
l = np.arange(-dh, dh, 2*dh/nl)[:, np.newaxis, np.newaxis] + 1

# Kinematic sum
fhkl, dt = scattering.Fhkl_thread(h, k, l, x, y, z, occ, gpu_name="TITAN", language='OpenCL')
diff_pattern = abs(fhkl**2)
diff_pattern = diff_pattern / diff_pattern.sum() * photon_number  # convert into photon number

plt.figure(num=3)
plt.imshow(np.log10(diff_pattern.sum(axis=0)), vmin=0)
plt.title("sum in Qy Qz")
plt.savefig('diff_sum_dh=' + str(dh) + '_nh=' + str(nh) + '.png')

plt.figure(num=4)
plt.imshow(np.log10(diff_pattern[nh//2, :, :]), vmin=0)
plt.title("Middle frame in Qy Qz")
plt.savefig('diff_central_dh=' + str(dh) + '_nh=' + str(nh) + '.png')

plt.pause(0.1)
np.savez_compressed('diffpattern_dh=' + str(dh) + '_nh=' + str(nh) + '.npz', diff_pattern)

plt.ioff()
plt.show()