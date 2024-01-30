#!/usr/bin/env python3

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import sys
import tkinter as tk
from tkinter import filedialog

import numpy as np
from lmfit import Parameters, minimize, report_fit
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

import bcdi.graph.graph_utils as gu
import bcdi.utils.utilities as util

helptext = """
Plot the modulus histogram of a complex object reconstructed by phase retrieval.
"""

scan = 11  # spec scan number
root_folder = "D:/data/Pt THH ex-situ/Data/CH4760/"
sample_name = "S"
homedir = root_folder + sample_name + str(scan) + "/pynxraw/"
# + '_' + str('{:05d}'.format(scan)) + '/pynx/1000_1000_1000_1_1_1/v1/'
comment = ""  # should start with _
fit = True  # if True, fit the histogram with lineshape
lineshape = "pseudovoigt"
fit_range = [0.5, 1.0]
histogram_Yaxis = "linear"  # 'log' or 'linear'
cutoff_amp = 0.05  # use only points with a modulus larger than this value
# to calculate mean, std and the histogram
save = False  # True to save the histogram plot
##########################
# end of user parameters #
##########################

#############
# load data #
#############
plt.ion()
root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename(
    initialdir=homedir,
    title="Select reconstruction file",
    filetypes=[("NPZ", "*.npz"), ("CXI", "*.cxi"), ("HDF5", "*.h5")],
)
obj, _ = util.load_file(file_path)

if obj.ndim != 3:
    print("a 3D reconstruction array is expected")
    sys.exit()

nbz, nby, nbx = obj.shape
print("Initial data size:", nbz, nby, nbx)

amp = abs(obj)
amp = amp / amp.max()

gu.multislices_plot(
    amp,
    sum_frames=False,
    title="Normalized modulus",
    vmin=0,
    vmax=1,
    plot_colorbar=True,
    is_orthogonal=True,
    reciprocal_space=False,
)

mean_amp = amp[amp > cutoff_amp].mean()
std_amp = amp[amp > cutoff_amp].std()
print("Mean amp=", mean_amp)
print("Std amp=", std_amp)
hist, bin_edges = np.histogram(amp[amp > cutoff_amp].flatten(), bins=50)
bin_step = (bin_edges[1] - bin_edges[0]) / 2
bin_axis = bin_edges + bin_step
bin_axis = bin_axis[0 : len(hist)]

# interpolate the histogram
newbin_axis = np.linspace(bin_axis.min(), bin_axis.max(), 500)
interp_hist = interp1d(bin_axis, hist, kind="cubic")
newhist = interp_hist(newbin_axis)

##############################################
# fit the peak with a pseudovoigt line shape #
##############################################
if fit:
    # find indices of the histogram points belonging to the range of interest
    ind_min, ind_max = util.find_nearest(newbin_axis, [min(fit_range), max(fit_range)])
    fit_axis = newbin_axis[np.arange(ind_min, ind_max + 1, 1)]
    fit_hist = newhist[np.arange(ind_min, ind_max + 1, 1)]
    # offset_hist = min(fit_hist)

    # define the initial parameters
    fit_params = Parameters()
    if lineshape == "pseudovoigt":
        cen = newbin_axis[np.unravel_index(newhist.argmax(), newhist.shape)]
        fit_params.add("amp_0", value=50000, min=100, max=1000000)
        fit_params.add("cen_0", value=cen, min=cen - 0.2, max=cen + 0.2)
        fit_params.add("sig_0", value=0.1, min=0.01, max=0.5)
        fit_params.add("ratio_0", value=0.5, min=0, max=1)

    # run the fit
    result = minimize(
        util.objective_lmfit, fit_params, args=(fit_axis, fit_hist, lineshape)
    )
    report_fit(result.params)
    y_fit = util.function_lmfit(
        params=result.params, iterator=0, x_axis=newbin_axis, distribution=lineshape
    )
else:
    y_fit = None
    result = None

##################################
# plot the histogram and the fit #
##################################
fig, ax = plt.subplots(1, 1)
plt.plot(bin_axis, hist, "o", newbin_axis, newhist, "-")
if histogram_Yaxis == "log":
    ax.set_yscale("log")
if fit:
    if histogram_Yaxis == "linear":
        ax.plot(newbin_axis, y_fit, "-")
    else:
        ax.plot(newbin_axis, np.log10(y_fit), "-")
    try:
        fig.text(
            0.15,
            0.95,
            "cen_0 = "
            + str("{:.5f}".format(result.params["cen_0"].value))
            + "+/-"
            + str("{:.5f}".format(result.params["cen_0"].stderr))
            + "   sig_0 = "
            + str("{:.5f}".format(result.params["sig_0"].value))
            + "+/-"
            + str("{:.5f}".format(result.params["sig_0"].stderr)),
            size=12,
        )
    except TypeError:  # one output is None
        fig.text(0.15, 0.95, "at least one output is None", size=12)
    fig.text(0.15, 0.80, lineshape + " fit", size=12)
plt.title(
    "<amp>=" + str(f"{mean_amp:.2f}") + ", std=" + str(f"{std_amp:.2f}") + comment
)
if save:
    fig.savefig(homedir + "amp_histogram" + comment + ".png")
plt.ioff()
plt.show()
