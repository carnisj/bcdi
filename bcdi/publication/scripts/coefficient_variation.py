# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.measurements import center_of_mass
import tkinter as tk
from tkinter import filedialog
import matplotlib.ticker as ticker
import sys
import gc
sys.path.append('D:/myscripts/bcdi/')
import bcdi.postprocessing.postprocessing_utils as pu
from bcdi.utils import image_registration as reg
import bcdi.graph.graph_utils as gu

helptext = """
Load several reconstructed complex objects and calculate the coefficient variation (CV = std/mean) 
on the modulus or the strain.

In the reconstruction file, the following fieldnames are expected: 'amp', 'bulk', 'phase' for simulated data or 'disp' 
for experimental data, 'strain'.

It is necessary to know the voxel size of the reconstruction in order to put ticks at the correct position.
Laboratory frame: z downstream, y vertical, x outboard (CXI convention)
"""


scans = [1301, 1304]  # spec scan number
rootfolder = 'D:/data/SIXS_2019_Ni/'
savedir = 'D:/data/SIXS_2019_Ni/comparison_S1301_S1304/'
comment = ''   # should start with _

voxel_size = 9.74  # in nm
tick_spacing = 50  # for plots, in nm
field_of_view = 900  # in nm, can be larger than the total width (the array will be padded)

tick_direction = 'in'  # 'out', 'in', 'inout'
tick_length = 10  # in plots
tick_width = 2  # in plots

strain_range = 0.002  # for plots
grey_background = True  # True to set the background to grey in phase and strain plots
background_strain = np.nan  # value outside of the crystal, np.nan will give grey if grey_background = True

save_YZ = True  # True to save the view in YZ plane
save_XZ = True  # True to save the view in XZ plane
save_XY = True  # True to save the view in XY plane

flag_strain = True  # True to plot and save the strain
flag_amp = True  # True to plot and save the amplitude
amp_threshold = 0  # amplitude below this value will be set to 0

center_object = False  # if True, will center the first object based on the COM of its support
# all other objects will be aligned on the first one
##########################
# end of user parameters #
##########################

###################
# define colormap #
###################
if grey_background:
    bad_color = '0.7'
else:
    bad_color = '1.0'  # white background
colormap = gu.Colormap(bad_color=bad_color)
my_cmap = colormap.cmap

#############
# load data #
#############
plt.ion()
root = tk.Tk()
root.withdraw()

nb_scans = len(scans)
fieldnames = ["scans", "modulus", "strain"]
modulus_list = []
strain_list = []
datasets = {new_list: [] for new_list in fieldnames}  # create dictionnary
datasets["scans"].append(scans)

for index in range(len(scans)):

    file_path = filedialog.askopenfilename(initialdir=rootfolder,
                                           title="Select amp-disp-strain file for S" + str(scans[index]),
                                           filetypes=[("NPZ", "*.npz")])
    npzfile = np.load(file_path)

    amp = npzfile['amp']
    strain = npzfile['strain']
    bulk = npzfile['bulk']
    # bulk is a support build from the amplitude minus the surface voxel layer were the strain is not defined

    numz, numy, numx = amp.shape

    # pad the dataset to the desired the field #
    pixel_spacing = tick_spacing / voxel_size  # TODO: allow for different voxel sizes
    pixel_FOV = int(np.rint((field_of_view / voxel_size) / 2))  # half-number of pixels corresponding to the FOV

    #  pad arrays to obtain the desired field of view
    new_shape = [max(numz, 2*pixel_FOV), max(numy, 2*pixel_FOV), max(numx, 2*pixel_FOV)]
    amp = pu.crop_pad(array=amp, output_shape=new_shape, debugging=False)
    strain = pu.crop_pad(array=strain, output_shape=new_shape, debugging=False)
    bulk = pu.crop_pad(array=bulk, output_shape=new_shape, debugging=False)
    numz, numy, numx = amp.shape

    strain[bulk == 0] = 0  # assign default values outside of the crystal

    if index == 0:
        # center the first object if needed
        if center_object is True:
            support = np.zeros(amp.shape)
            support[amp > amp_threshold * amp.max()] = 1

            zcom, ycom, xcom = center_of_mass(support)
            zcom, ycom, xcom = [int(np.rint(zcom)), int(np.rint(ycom)), int(np.rint(xcom))]
            # support = np.roll(support, (numz//2-zcom, numy//2-ycom, numx//2-xcom), axis=(0, 1, 2))
            amp = np.roll(amp, (numz//2-zcom, numy//2-ycom, numx//2-xcom), axis=(0, 1, 2))
            strain = np.roll(strain, (numz//2-zcom, numy//2-ycom, numx//2-xcom), axis=(0, 1, 2))

        ref_amp = np.copy(amp)

    else:  # align it with the reference object
        # TODO: interpolate object if the shape is different from ref_amp (different voxel sizes between datasets)
        shiftz, shifty, shiftx = reg.getimageregistration(ref_amp, amp, precision=1000)
        print('Shift of array', index, 'with the reference array:', shiftz, shifty, shiftx)
        amp = abs(reg.subpixel_shift(amp, shiftz, shifty, shiftx))
        strain = abs(reg.subpixel_shift(strain, shiftz, shifty, shiftx))

    amp = amp / amp.max()
    amp[amp < amp_threshold] = 0

    datasets['modulus'].append(amp)
    datasets['strain'].append(strain)

numz, numy, numx = ref_amp.shape

#############
# Amplitude #
#############
if flag_amp:
    amp_concat = np.zeros((nb_scans, ref_amp.size))
    for idx in range(nb_scans):
        amp_concat[idx, :] = datasets['modulus'][idx].flatten()

    CV_amp = np.divide(np.std(amp_concat, axis=0), np.mean(amp_concat, axis=0)).reshape(ref_amp.shape)

    fig, ax0 = plt.subplots(1, 1)
    plt0 = ax0.imshow(
        CV_amp[numz // 2 - pixel_FOV:numz // 2 + pixel_FOV, numy // 2 - pixel_FOV:numy // 2 + pixel_FOV, numx // 2],
        vmin=0, vmax=1, cmap=my_cmap)

    ax0.xaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
    ax0.yaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
    ax0.tick_params(labelbottom=False, labelleft=False, top=True, right=True, direction=tick_direction,
                    length=tick_length, width=tick_width)
    if save_YZ:
        fig.savefig(savedir + 'CV_amp_YZ' + comment + '.png', bbox_inches="tight")

    fig, ax1 = plt.subplots(1, 1)
    plt1 = ax1.imshow(
        CV_amp[numz // 2 - pixel_FOV:numz // 2 + pixel_FOV, numy // 2, numx // 2 - pixel_FOV:numx // 2 + pixel_FOV],
        vmin=0, vmax=1, cmap=my_cmap)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
    ax1.tick_params(labelbottom=False, labelleft=False, top=True, right=True, direction=tick_direction,
                    length=tick_length, width=tick_width)
    if save_XZ:
        fig.savefig(savedir + 'CV_amp_XZ' + comment + '.png', bbox_inches="tight")

    fig, ax2 = plt.subplots(1, 1)
    plt2 = ax2.imshow(
        CV_amp[numz // 2, numy // 2 - pixel_FOV:numy // 2 + pixel_FOV, numx // 2 - pixel_FOV:numx // 2 + pixel_FOV],
        vmin=0, vmax=1, cmap=my_cmap)
    ax2.invert_yaxis()
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
    ax2.tick_params(labelbottom=False, labelleft=False, top=True, right=True, direction=tick_direction,
                    length=tick_length, width=tick_width)
    if save_XY:
        fig.savefig(savedir + 'CV_amp_XY' + comment + '.png', bbox_inches="tight")
    plt.colorbar(plt2, ax=ax2)
    fig.savefig(savedir + 'CV_amp_XY' + comment + '_colorbar.png', bbox_inches="tight")
##########
# Strain #
##########
if flag_strain:
    strain_concat = np.zeros((nb_scans, ref_amp.size))
    for idx in range(nb_scans):
        strain_concat[idx, :] = datasets['strain'][idx].flatten()

    CV_strain = np.divide(np.std(strain_concat, axis=0), np.mean(strain_concat, axis=0)).reshape(ref_amp.shape)

    fig, ax0 = plt.subplots(1, 1)
    plt0 = ax0.imshow(CV_strain[numz//2-pixel_FOV:numz//2+pixel_FOV, numy//2-pixel_FOV:numy//2+pixel_FOV, numx // 2],
                      vmin=-strain_range, vmax=strain_range, cmap=my_cmap)
    ax0.xaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
    ax0.yaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
    ax0.tick_params(labelbottom=False, labelleft=False, top=True, right=True, direction=tick_direction,
                    length=tick_length, width=tick_width)
    plt.colorbar(plt0, ax=ax0)
    if save_YZ:
        fig.savefig(savedir + 'CV_strain_YZ' + comment + '.png', bbox_inches="tight")

    fig, ax1 = plt.subplots(1, 1)
    plt1 = ax1.imshow(CV_strain[numz//2-pixel_FOV:numz//2+pixel_FOV, numy // 2, numx//2-pixel_FOV:numx//2+pixel_FOV],
                      vmin=-strain_range, vmax=strain_range, cmap=my_cmap)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
    ax1.tick_params(labelbottom=False, labelleft=False, top=True, right=True, direction=tick_direction,
                    length=tick_length, width=tick_width)
    plt.colorbar(plt1, ax=ax1)
    if save_XZ:
        fig.savefig(savedir + 'CV_strain_XZ' + comment + '.png', bbox_inches="tight")

    fig, ax2 = plt.subplots(1, 1)
    plt2 = ax2.imshow(CV_strain[numz // 2, numy//2-pixel_FOV:numy//2+pixel_FOV, numx//2-pixel_FOV:numx//2+pixel_FOV],
                      vmin=-strain_range, vmax=strain_range, cmap=my_cmap)
    ax2.invert_yaxis()
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
    ax2.tick_params(labelbottom=False, labelleft=False, top=True, right=True, direction=tick_direction,
                    length=tick_length, width=tick_width)
    plt.colorbar(plt2, ax=ax2)
    if save_XY:
        fig.savefig(savedir + 'CV_strain_XY' + comment + '.png', bbox_inches="tight")

plt.ioff()
plt.show()
