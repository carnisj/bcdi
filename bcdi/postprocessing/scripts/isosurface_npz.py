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
import sys
from scipy.interpolate import interp1d
sys.path.append('D:/myscripts/bcdi/')
import bcdi.graph.graph_utils as gu
import bcdi.utils.utilities as util

helptext = """
Plot the modulus histogram of a complex object reconstructed by phase retrieval.
"""

scan = 22    # spec scan number
root_folder = "D:/data/P10_August2019/data/"
sample_name = "gold_2_2_2"
homedir = root_folder + sample_name + '_' + str('{:05d}'.format(scan)) + '/pynx/1000_1000_1000_1_1_1/v1/'
comment = ""  # should start with _

histogram_Yaxis = 'linear'  # 'log' or 'linear'
threshold_amp = 0.05  # use only points with larger modulus to calculate mean, std and the histogram
save = True  # True to save the histogram plot
##########################
# end of user parameters #
##########################

#############
# load data #
#############
plt.ion()
root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename(initialdir=homedir, title="Select reconstruction file",
                                       filetypes=[("NPZ", "*.npz"), ("CXI", "*.cxi"), ("HDF5", "*.h5")])
obj, _ = util.load_file(file_path)

if obj.ndim != 3:
    print('a 3D reconstruction array is expected')
    sys.exit()

nbz, nby, nbx = obj.shape
print("Initial data size:", nbz, nby, nbx)

amp = abs(obj)
amp = amp / amp.max()

gu.multislices_plot(amp, sum_frames=False, title='Normalized modulus', vmin=0, vmax=1, plot_colorbar=True,
                    is_orthogonal=True, reciprocal_space=False)

mean_amp = amp[amp > threshold_amp].mean()
std_amp = amp[amp > threshold_amp].std()
print("Mean amp=", mean_amp)
print("Std amp=", std_amp)
hist, bin_edges = np.histogram(amp[amp > threshold_amp].flatten(), bins=50)
bin_step = (bin_edges[1]-bin_edges[0])/2
bin_axis = bin_edges + bin_step
bin_axis = bin_axis[0:len(hist)]

newbin_axis = np.linspace(bin_axis.min(), bin_axis.max(), 120)
newbin_step = newbin_axis[1] - newbin_axis[0]

fit_hist = interp1d(bin_axis, hist, kind='cubic')
newhist = fit_hist(newbin_axis)

fig, ax = plt.subplots(1, 1)
plt.plot(bin_axis, hist, 'o', newbin_axis, newhist, '-')
if histogram_Yaxis == 'log':
    ax.set_yscale('log')
# plt.title('S'+str(scan)+', <amp>='+str('{:.2f}'.format(mean_amp))+', std='+str('{:.2f}'.format(std_amp))+comment)
plt.title('<amp>='+str('{:.2f}'.format(mean_amp))+', std='+str('{:.2f}'.format(std_amp))+comment)

if save:
    fig.savefig(homedir + 'amp_histogram' + comment + '.png')
plt.ioff()
plt.show()
