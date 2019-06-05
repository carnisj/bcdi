# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#       authors:
#         Jerome Carnis, jerome.carnis@esrf.fr

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage.measurements import center_of_mass
import tkinter as tk
from tkinter import filedialog
import matplotlib.ticker as ticker
import sys
sys.path.append('C:\\Users\\carnis\\Work Folders\\Documents\\myscripts\\bcdi\\')
import bcdi.postprocessing.postprocessing_utils as pu

helptext = """
Template for making 2D slices figures for a publication about BCDI.

Open an amp_dist_strain.npz file and save individual figures.

In the reconstruction file, the following fieldnames are expected: 'amp', 'bulk', 'phase' for simulated data or 'disp' 
for experimental data, 'strain'.

It is necessary to know the voxel size of the reconstruction in order to put ticks at the correct position.
"""


scan = 978  # spec scan number
datadir = "C:/Users/carnis/Work Folders/Documents/data/HC3207/SN"+str(scan)+"/pynxraw/"
savedir = "C:/Users/carnis/Work Folders/Documents/data/HC3207/SN"+str(scan) + "/figures/"
comment = '_modes'   # should start with _
simulated_data = False

voxel_size = 3  # in nm
tick_spacing = 50  # for plots, in nm
field_of_view = 400  # in nm, can be larger than the total width (the array will be padded)

tick_direction = 'in'  # 'out', 'in', 'inout'
tick_length = 10  # in plots
tick_width = 2  # in plots

strain_range = 0.002  # for plots
phase_range = np.pi  # for plots
grey_background = True  # True to set the background to grey in phase and strain plots

save_YZ = 0  # True to save the strain in YZ plane
save_XZ = True  # True to save the strain in XZ plane
save_XY = True  # True to save the strain in XY plane

flag_strain = True  # True to plot and save the strain
flag_phase = True  # True to plot and save the phase
flag_amp = True  # True to plot and save the amplitude
amp_histogram_Yaxis = 'linear'  # 'log' or 'linear', Y axis scale for the amplitude histogram
flag_support = True  # True to plot and save the support
flag_linecut = False  # True to plot and save a linecut of the phase
##########################
# end of user parameters #
##########################

#######################
# define the colormap #
#######################
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
my_cmap.set_bad(color='0.7')

#######################################
plt.ion()
root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename(initialdir=datadir, title="Select data file", filetypes=[("NPZ", "*.npz")])
npzfile = np.load(file_path)
strain = npzfile['strain']
amp = npzfile['amp']
bulk = npzfile['bulk']  # bulk is the amplitude minus the surface voxel layer were the strain is not defined

amp = amp / amp.max()  # normalize amplitude
amp[amp < 0.01] = 0
support = np.zeros(amp.shape)
support[np.nonzero(amp)] = 1

if simulated_data:
    phase = npzfile['phase']
else:
    phase = npzfile['displacement']

if flag_phase:
    comment = comment + "_phaserange_" + str('{:.2f}'.format(phase_range))
if flag_strain:
    comment = comment + "_strainrange_" + str(strain_range)

numz, numy, numx = amp.shape
print("Initial data size: (", numz, ',', numy, ',', numx, ')')

#############################
#  pad arrays to obtain the desired field of view
#############################
pixel_spacing = tick_spacing / voxel_size
pixel_FOV = int(np.rint((field_of_view / voxel_size) / 2))  # half-number of pixels corresponding to the FOV
new_shape = [max(numz, 2*pixel_FOV), max(numy, 2*pixel_FOV), max(numx, 2*pixel_FOV)]
support = pu.crop_pad(array=support, output_shape=new_shape, debugging=False)
strain = pu.crop_pad(array=strain, output_shape=new_shape, debugging=False)
phase = pu.crop_pad(array=phase, output_shape=new_shape, debugging=False)
amp = pu.crop_pad(array=amp, output_shape=new_shape, debugging=False)
bulk = pu.crop_pad(array=bulk, output_shape=new_shape, debugging=False)
numz, numy, numx = amp.shape
print("Cropped/padded data size: (", numz, ',', numy, ',', numx, ')')

#############################
# center arrays based on the support
#############################
zcom, ycom, xcom = center_of_mass(support)
zcom, ycom, xcom = [int(np.rint(zcom)), int(np.rint(ycom)), int(np.rint(xcom))]
support = np.roll(support, (numz//2-zcom, numy//2-ycom, numx//2-xcom), axis=(0, 1, 2))
strain = np.roll(strain, (numz//2-zcom, numy//2-ycom, numx//2-xcom), axis=(0, 1, 2))
phase = np.roll(phase, (numz//2-zcom, numy//2-ycom, numx//2-xcom), axis=(0, 1, 2))
amp = np.roll(amp, (numz//2-zcom, numy//2-ycom, numx//2-xcom), axis=(0, 1, 2))
bulk = np.roll(bulk, (numz//2-zcom, numy//2-ycom, numx//2-xcom), axis=(0, 1, 2))

strain[bulk == 0] = -2 * strain_range
phase[bulk == 0] = np.nan

###########
# Support #
###########
if flag_support:
    fig, ax0 = plt.subplots(1, 1)
    plt0 = ax0.imshow(
        support[numz // 2 - pixel_FOV:numz // 2 + pixel_FOV, numy // 2 - pixel_FOV:numy // 2 + pixel_FOV, numx // 2],
        vmin=0, vmax=1, cmap=my_cmap)

    ax0.xaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
    ax0.yaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
    ax0.tick_params(labelbottom='off', labelleft='off', top='on', right='on', direction=tick_direction,
                    length=tick_length, width=tick_width)
    if save_YZ:
        fig.savefig(savedir + 'support_YZ' + comment + '.png', bbox_inches="tight")

    fig, ax1 = plt.subplots(1, 1)
    plt1 = ax1.imshow(
        support[numz // 2 - pixel_FOV:numz // 2 + pixel_FOV, numy // 2, numx // 2 - pixel_FOV:numx // 2 + pixel_FOV],
        vmin=0, vmax=1, cmap=my_cmap)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
    ax1.tick_params(labelbottom='off', labelleft='off', top='on', right='on', direction=tick_direction,
                    length=tick_length, width=tick_width)
    if save_XZ:
        fig.savefig(savedir + 'support_XZ' + comment + '.png', bbox_inches="tight")

    fig, ax2 = plt.subplots(1, 1)
    plt2 = ax2.imshow(
        support[numz // 2, numy // 2 - pixel_FOV:numy // 2 + pixel_FOV, numx // 2 - pixel_FOV:numx // 2 + pixel_FOV],
        vmin=0, vmax=1, cmap=my_cmap)
    ax2.invert_yaxis()
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
    ax2.tick_params(labelbottom='off', labelleft='off', top='on', right='on', direction=tick_direction,
                    length=tick_length, width=tick_width)

    if save_XY:
        fig.savefig(savedir + 'support_XY' + comment + '.png', bbox_inches="tight")

#############
# Amplitude #
#############
if flag_amp:
    fig, ax0 = plt.subplots(1, 1)
    plt0 = ax0.imshow(
        amp[numz // 2 - pixel_FOV:numz // 2 + pixel_FOV, numy // 2 - pixel_FOV:numy // 2 + pixel_FOV, numx // 2],
        vmin=0, vmax=1, cmap=my_cmap)

    ax0.xaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
    ax0.yaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
    ax0.tick_params(labelbottom='off', labelleft='off', top='on', right='on', direction=tick_direction,
                    length=tick_length, width=tick_width)
    if save_YZ:
        fig.savefig(savedir + 'amp_YZ' + comment + '.png', bbox_inches="tight")

    fig, ax1 = plt.subplots(1, 1)
    plt1 = ax1.imshow(
        amp[numz // 2 - pixel_FOV:numz // 2 + pixel_FOV, numy // 2, numx // 2 - pixel_FOV:numx // 2 + pixel_FOV],
        vmin=0, vmax=1, cmap=my_cmap)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
    ax1.tick_params(labelbottom='off', labelleft='off', top='on', right='on', direction=tick_direction,
                    length=tick_length, width=tick_width)
    if save_XZ:
        fig.savefig(savedir + 'amp_XZ' + comment + '.png', bbox_inches="tight")

    fig, ax2 = plt.subplots(1, 1)
    plt2 = ax2.imshow(
        amp[numz // 2, numy // 2 - pixel_FOV:numy // 2 + pixel_FOV, numx // 2 - pixel_FOV:numx // 2 + pixel_FOV],
        vmin=0, vmax=1, cmap=my_cmap)
    ax2.invert_yaxis()
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
    ax2.tick_params(labelbottom='off', labelleft='off', top='on', right='on', direction=tick_direction,
                    length=tick_length, width=tick_width)

    if save_XY:
        fig.savefig(savedir + 'amp_XY' + comment + '.png', bbox_inches="tight")
    plt.colorbar(plt2, ax=ax2)
    fig.savefig(savedir + 'amp_XY' + comment + '_colorbar.png', bbox_inches="tight")

    min_amp = amp.min()
    fig, ax = plt.subplots(1, 1)
    plt.hist(amp[amp > min_amp].flatten(), bins=250)
    plt.xlim(left=0.05)
    plt.ylim(bottom=1)
    if amp_histogram_Yaxis == 'log':
        ax.set_yscale('log')
        plt.ylim(top=100000)
    ax.tick_params(labelbottom='off', labelleft='off', direction='out', length=tick_length, width=tick_width)
    plt.savefig(savedir + 'phased_histogram_amp' + comment + '.png', bbox_inches="tight")
    ax.tick_params(labelbottom='on', labelleft='on', direction='out', length=tick_length, width=tick_width)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    fig.savefig(savedir + 'phased_histogram_amp' + comment + '_labels.png', bbox_inches="tight")

##########
# Strain #
##########
if flag_strain:
    if grey_background:
        strain_copy = np.copy(strain)
        strain_copy[bulk == 0] = np.nan
        strain = np.ma.array(strain_copy, mask=np.isnan(strain_copy))

    fig, ax0 = plt.subplots(1, 1)
    plt0 = ax0.imshow(strain[numz//2-pixel_FOV:numz//2+pixel_FOV, numy//2-pixel_FOV:numy//2+pixel_FOV, numx // 2],
                      vmin=-strain_range, vmax=strain_range, cmap=my_cmap)
    ax0.xaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
    ax0.yaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
    ax0.tick_params(labelbottom='off', labelleft='off', top='on', right='on', direction=tick_direction,
                    length=tick_length, width=tick_width)
    if save_YZ:
        fig.savefig(savedir + 'strain_YZ' + comment + '.png', bbox_inches="tight")

    fig, ax1 = plt.subplots(1, 1)
    plt1 = ax1.imshow(strain[numz//2-pixel_FOV:numz//2+pixel_FOV, numy // 2, numx//2-pixel_FOV:numx//2+pixel_FOV],
                      vmin=-strain_range, vmax=strain_range, cmap=my_cmap)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
    ax1.tick_params(labelbottom='off', labelleft='off', top='on', right='on', direction=tick_direction,
                    length=tick_length, width=tick_width)
    if save_XZ:
        fig.savefig(savedir + 'strain_XZ' + comment + '.png', bbox_inches="tight")

    fig, ax2 = plt.subplots(1, 1)
    plt2 = ax2.imshow(strain[numz // 2, numy//2-pixel_FOV:numy//2+pixel_FOV, numx//2-pixel_FOV:numx//2+pixel_FOV],
                      vmin=-strain_range, vmax=strain_range, cmap=my_cmap)
    ax2.invert_yaxis()
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
    ax2.tick_params(labelbottom='off', labelleft='off', top='on', right='on', direction=tick_direction,
                    length=tick_length, width=tick_width)

    if save_XY:
        fig.savefig(savedir + 'strain_XY' + comment + '.png', bbox_inches="tight")
    plt.colorbar(plt2, ax=ax2)
    fig.savefig(savedir + 'strain_XY' + comment + '_colorbar.png', bbox_inches="tight")

#########
# Phase #
#########
if flag_phase:
    if grey_background:
        phase_copy = np.copy(phase)
        phase_copy[bulk == 0] = np.nan
        phase = np.ma.array(phase_copy, mask=np.isnan(phase_copy))
    fig, ax0 = plt.subplots(1, 1)
    plt0 = ax0.imshow(phase[numz//2-pixel_FOV:numz//2+pixel_FOV, numy//2-pixel_FOV:numy//2+pixel_FOV, numx // 2],
                      vmin=-phase_range, vmax=phase_range, cmap=my_cmap)
    ax0.xaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
    ax0.yaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
    ax0.tick_params(labelbottom='off', labelleft='off', top='on', right='on', direction=tick_direction,
                    length=tick_length, width=tick_width)
    if save_YZ:
        fig.savefig(savedir + 'phase_YZ' + comment + '.png', bbox_inches="tight")

    fig, ax1 = plt.subplots(1, 1)
    plt1 = ax1.imshow(phase[numz//2-pixel_FOV:numz//2+pixel_FOV, numy // 2, numx//2-pixel_FOV:numx//2+pixel_FOV],
                      vmin=-phase_range, vmax=phase_range, cmap=my_cmap)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
    ax1.tick_params(labelbottom='off', labelleft='off', top='on', right='on', direction=tick_direction,
                    length=tick_length, width=tick_width)
    if save_XZ:
        fig.savefig(savedir + 'phase_XZ' + comment + '.png', bbox_inches="tight")

    fig, ax2 = plt.subplots(1, 1)
    plt2 = ax2.imshow(phase[numz // 2, numy//2-pixel_FOV:numy//2+pixel_FOV, numx//2-pixel_FOV:numx//2+pixel_FOV],
                      vmin=-phase_range, vmax=phase_range, cmap=my_cmap)
    ax2.invert_yaxis()
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
    ax2.tick_params(labelbottom='off', labelleft='off', top='on', right='on', direction=tick_direction,
                    length=tick_length, width=tick_width)

    if save_XY:
        fig.savefig(savedir + 'phase_XY' + comment + '.png', bbox_inches="tight")
    plt.colorbar(plt2, ax=ax2)
    fig.savefig(savedir + 'phase_XY' + comment + '_colorbar.png', bbox_inches="tight")

    ###################
    # example of a line cut on the phase, can also load more data for the lineplot for comparison
    ##################
    if flag_linecut:
        file_path = filedialog.askopenfilename(initialdir=datadir, title="Select avg7 file",
                                               filetypes=[("NPZ", "*.npz")])
        # npzfile = np.load(file_path)
        # phase2 = npzfile['displacement']
        # bulk = npzfile['bulk']
        # if phase2.shape != phase.shape:
        #     print('array2 shape not compatible')
        #     sys.exit()
        # phase2[bulk == 0] = np.nan
        #
        # file_path = filedialog.askopenfilename(initialdir=datadir, title="Select post_processing apodization file",
        #                                        filetypes=[("NPZ", "*.npz")])
        # npzfile = np.load(file_path)
        # phase3 = npzfile['displacement']
        # bulk = npzfile['bulk']
        # if phase3.shape != phase.shape:
        #     print('array3 shape not compatible')
        #     sys.exit()
        # phase3[bulk == 0] = np.nan
        #
        # file_path = filedialog.askopenfilename(initialdir=datadir, title="Select pre_processingapodization file",
        #                                        filetypes=[("NPZ", "*.npz")])
        # npzfile = np.load(file_path)
        # phase4 = npzfile['displacement']
        # bulk = npzfile['bulk']
        # if phase3.shape != phase.shape:
        #     print('array4 shape not compatible')
        #     sys.exit()
        # phase4[bulk == 0] = np.nan

        fig, ax3 = plt.subplots(1, 1)
        plt.plot(phase[numz // 2, 130, :], 'r', linestyle='-')  # (1, (4, 4)))  #
        # plt.plot(phase2[numz // 2, 130, :], 'k', linestyle='-.')  # , marker='D', fillstyle='none'
        # plt.plot(phase3[numz // 2, 130, :], 'b', linestyle=':')  # , marker='^', fillstyle='none'
        # plt.plot(phase4[numz // 2, 130, :], 'g', linestyle='--')  # , marker='^', fillstyle='none'
        ax3.spines['right'].set_linewidth(1)
        ax3.spines['left'].set_linewidth(1)
        ax3.spines['top'].set_linewidth(1)
        ax3.spines['bottom'].set_linewidth(1)
        ax3.set_xlim(50, 150)
        ax3.set_ylim(-np.pi, np.pi)
        ax3.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax3.tick_params(labelbottom='off', labelleft='off', top='off', bottom='on', direction='inout',
                        length=tick_length, width=tick_width)
        fig.savefig(savedir + 'Linecut_phase_X_Y=130' + comment + '.png', bbox_inches="tight")

plt.ioff()
plt.show()
