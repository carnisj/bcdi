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
from scipy.ndimage.measurements import center_of_mass
from scipy.interpolate import interp1d

helptext = """
Help to define the isosurface for the reconstructed scan depending on amplitude histogram.
"""

save = 1  # 1 to save the histogram, 0 otherwise
plot = 0  # 1 to plot the summed amplitude, 0 otherwise
# scan = 978
polyfit = 0  # fir with polynomial instead of spline
histogram_Yaxis = 'linear'  # 'log' or 'linear'
comment = '1'
datadir = 'D:/review paper/BCDI_isosurface/S2227/simu/crop600/'  # 'C:/users/CARNIS/Work Folders/Documents/data/HC3207/SN'+str(scan)+'/pynxraw/'
################################################################################


def center_max(myarray, debugging=0):
    """"
    :param myarray: array to be centered based on the max value
    :param debugging: 1 to show plots
    :return centered array
    """
    nbz, nby, nbx = myarray.shape
    if debugging == 1:
        plt.figure(figsize=(18, 15))
        plt.subplot(2, 2, 1)
        plt.imshow(abs(myarray).sum(axis=2))
        plt.colorbar()
        plt.title("Sum(amp) in YZ before Max centering")
        plt.subplot(2, 2, 2)
        plt.imshow(abs(myarray).sum(axis=1))
        plt.colorbar()
        plt.title("Sum(amp) in XZ before Max centering")
        plt.subplot(2, 2, 3)
        plt.imshow(abs(myarray).sum(axis=0))
        plt.colorbar()
        plt.title("Sum(amp) in XY before Max centering")
        plt.pause(0.1)
    piz, piy, pix = np.unravel_index(abs(myarray).argmax(), myarray.shape)
    print("Max at (z, y, x):", [piz, piy, pix])
    offset_z = int(np.rint(nbz / 2.0 - piz))
    offset_y = int(np.rint(nby / 2.0 - piy))
    offset_x = int(np.rint(nbx / 2.0 - pix))
    print("Max offset: ", [offset_z, offset_y, offset_x])
    myarray = np.roll(myarray, (offset_z, offset_y, offset_x), axis=(0, 1, 2))
    if debugging == 1:
        plt.figure(figsize=(18, 15))
        plt.subplot(2, 2, 1)
        plt.imshow(abs(myarray).sum(axis=2))
        plt.colorbar()
        plt.title("Sum(amp) in YZ after Max centering")
        plt.subplot(2, 2, 2)
        plt.imshow(abs(myarray).sum(axis=1))
        plt.colorbar()
        plt.title("Sum(amp) in XZ after Max centering")
        plt.subplot(2, 2, 3)
        plt.imshow(abs(myarray).sum(axis=0))
        plt.colorbar()
        plt.title("Sum(amp) in XY after Max centering")
        plt.pause(0.1)
    return myarray


def center_com(myarray, debugging=0):
    """"
    :param myarray: array to be centered based on the center of mass value
    :param debugging: 1 to show plots
    :return centered array
    """
    nbz, nby, nbx = myarray.shape
    if debugging == 1:
        plt.figure(figsize=(18, 15))
        plt.subplot(2, 2, 1)
        plt.imshow(abs(myarray).sum(axis=2))
        plt.colorbar()
        plt.title("Sum(amp) in YZ before COM centering")
        plt.subplot(2, 2, 2)
        plt.imshow(abs(myarray).sum(axis=1))
        plt.colorbar()
        plt.title("Sum(amp) in XZ before COM centering")
        plt.subplot(2, 2, 3)
        plt.imshow(abs(myarray).sum(axis=0))
        plt.colorbar()
        plt.title("Sum(amp) in XY before COM centering")
        plt.pause(0.1)
    piz, piy, pix = center_of_mass(abs(myarray))
    print("center of mass at (z, y, x):", [piz, piy, pix])
    offset_z = int(np.rint(nbz / 2.0 - piz))
    offset_y = int(np.rint(nby / 2.0 - piy))
    offset_x = int(np.rint(nbx / 2.0 - pix))
    print("center of mass offset: ", [offset_z, offset_y, offset_x])
    myarray = np.roll(myarray, (offset_z, offset_y, offset_x), axis=(0, 1, 2))
    if debugging == 1:
        plt.figure(figsize=(18, 15))
        plt.subplot(2, 2, 1)
        plt.imshow(abs(myarray).sum(axis=2))
        plt.colorbar()
        plt.title("Sum(amp) in YZ after COM centering")
        plt.subplot(2, 2, 2)
        plt.imshow(abs(myarray).sum(axis=1))
        plt.colorbar()
        plt.title("Sum(amp) in XZ after COM centering")
        plt.subplot(2, 2, 3)
        plt.imshow(abs(myarray).sum(axis=0))
        plt.colorbar()
        plt.title("Sum(amp) in XY after COM centering")
        plt.pause(0.1)
    return myarray


################################################
# load data
################################################
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(initialdir=datadir, filetypes=[("NPZ", "*.npz")])
npzfile = np.load(file_path)
# npzfile.files to check the content of npzfile
obj = npzfile['amp']
obj = center_max(obj)
obj = center_com(obj)
plt.ion()

if plot == 1:
    plt.figure(figsize=(18, 15))
    plt.subplot(2, 2, 1)
    plt.imshow(abs(obj).sum(axis=2))
    plt.colorbar()
    plt.axis('scaled')
    plt.title("Sum(amp) in YZ after COM centering")
    plt.subplot(2, 2, 2)
    plt.imshow(abs(obj).sum(axis=1))
    plt.colorbar()
    plt.axis('scaled')
    plt.title("Sum(amp) in XZ after COM centering")
    plt.subplot(2, 2, 3)
    plt.imshow(abs(obj).sum(axis=0))
    plt.colorbar()
    plt.axis('scaled')
    plt.title("Sum(amp) in XY after COM centering")


amp = abs(obj)
amp = amp / amp.max()
mean_amp = amp[amp > 0.01].mean()
std_amp = amp[amp > 0.01].std()
print("Mean amp=", mean_amp)
print("Std amp=", std_amp)
hist, bin_edges = np.histogram(amp[amp > 0.01].flatten(), bins=30)
bin_step = (bin_edges[1]-bin_edges[0])/2
bin_axis = bin_edges + bin_step
bin_axis = bin_axis[0:len(hist)]

newbin_axis = np.linspace(bin_axis.min(), bin_axis.max(), 120)
newbin_step = newbin_axis[1] - newbin_axis[0]
if polyfit == 0:
    fit_hist = interp1d(bin_axis, hist, kind='cubic')
    newhist = fit_hist(newbin_axis)
else:
    poly = np.poly1d(np.polyfit(bin_axis, hist, 10))
    newhist = poly(newbin_axis)

fig, ax = plt.subplots(1, 1)
plt.plot(bin_axis, hist, 'o', newbin_axis, newhist, '-')
if histogram_Yaxis == 'log':
    ax.set_yscale('log')
# plt.title('S'+str(scan)+', <amp>='+str('{:.2f}'.format(mean_amp))+', std='+str('{:.2f}'.format(std_amp))+comment)
plt.title('<amp>='+str('{:.2f}'.format(mean_amp))+', std='+str('{:.2f}'.format(std_amp))+comment)
if save == 1:
    plt.savefig(datadir + 'amp_histogram_' + comment + '.png')
plt.ioff()
plt.show()

