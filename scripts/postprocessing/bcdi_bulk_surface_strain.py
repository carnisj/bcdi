#!/usr/bin/env python3

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import os
import pathlib
import tkinter as tk
from tkinter import filedialog

import numpy as np
from lmfit import Parameters, minimize, report_fit
from matplotlib import pyplot as plt

import bcdi.graph.graph_utils as gu
import bcdi.postprocessing.postprocessing_utils as pu
import bcdi.utils.utilities as util
from bcdi.graph.colormap import ColormapFactory

helptext = """
Extract the surface voxel layer of an object recontructed by BCDI phase retrieval and
plot histograms of the strain at the surface and in the remaining bulk.
Input: a .npz file containing fields 'amp' and 'strain' (e.g., S130_amp_disp_strain.npz)
"""

scan = 1  # spec scan number
root_folder = "D:/data/P10_2nd_test_isosurface_Dec2020/data_nanolab/"
sample_name = "dataset_"  # "S"
datadir = root_folder + sample_name + str(scan) + "_newpsf/result/"
savedir = datadir + "test/"  # valid path or None (default to datadir in that case)
support_threshold = (
    0.496  # threshold applied to the modulus for reading the surface strain
)
normalize = (
    True  # if True, will normalize the histograms to the respective number of points
)
bin_step = 2e-5  # step size for the bins (in units of strain)
plot_scale = "linear"  # 'log' or 'linear', Y scale for the histograms
xlim = (
    -0.002,
    0.002,
)  # limits used for the horizontal axis of histograms, leave None otherwise
ylim = [
    0,
    0.04,
]  # limits used for the vertical axis of histograms, leave None otherwise
fit_pdf = "skewed_gaussian"  # 'pseudovoigt' or 'skewed_gaussian'
save_txt = False  # True to save the strain values for the surface,
# the bulk and the full support in txt files
debug = True  # True to see more plots
tick_length = 4  # in plots
tick_width = 1.5  # in plots
##########################
# end of user parameters #
##########################

#########################
# check some parameters #
#########################
if fit_pdf not in {
    "pseudovoigt",
    "skewed_gaussian",
}:
    raise ValueError("invalid value for fit_pdf parameter")
savedir = savedir or datadir
pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)

###################
# define colormap #
###################
my_cmap = ColormapFactory().cmap

#############
# load data #
#############
plt.ion()
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(initialdir=datadir, filetypes=[("NPZ", "*.npz")])
npzfile = np.load(file_path)
amp = npzfile["amp"]
strain = npzfile["strain"]
nbz, nby, nbx = amp.shape

######################################
# define the support, surface & bulk #
######################################
support = np.zeros(amp.shape)
support[amp > support_threshold * amp.max()] = 1
coordination_matrix = pu.calc_coordination(
    support=support, kernel=np.ones((3, 3, 3)), debugging=debug
)
surface = np.copy(support)
surface[coordination_matrix > 22] = 0  # remove the bulk 22
bulk = support - surface
nb_surface = len(np.nonzero(surface)[0])
nb_bulk = len(np.nonzero(bulk)[0])
print(f"Number of surface points = {nb_surface}")
print(f"Number of bulk points = {nb_bulk}")
if debug:
    gu.multislices_plot(
        surface,
        sum_frames=False,
        plot_colorbar=True,
        cmap=my_cmap,
        title="Surface layer",
        scale="linear",
        is_orthogonal=True,
        reciprocal_space=False,
    )
    gu.multislices_plot(
        bulk,
        sum_frames=False,
        plot_colorbar=True,
        cmap=my_cmap,
        title="Bulk",
        scale="linear",
        is_orthogonal=True,
        reciprocal_space=False,
    )

##########################
# save the strain values #
##########################
if save_txt:
    with open(
        os.path.join(
            savedir,
            f"S{scan}_threshold{support_threshold}_surface.dat",
        ),
        "w",
    ) as file_surface, open(
        os.path.join(
            savedir,
            f"S{scan}_threshold{support_threshold}_bulk.dat",
        ),
        "w",
    ) as file_bulk:
        # write surface points position / strain to file
        surface_indices = np.nonzero(surface)
        nb_surface = len(surface_indices[0])
        ind_z = surface_indices[0]
        ind_y = surface_indices[1]
        ind_x = surface_indices[2]
        for point in range(nb_surface):
            file_surface.write(
                f"{f'{strain[ind_z[point], ind_y[point], ind_x[point]]:.7f}': <10}\n"
            )

        # write bulk points position / strain to file
        bulk_indices = np.nonzero(bulk)
        nb_bulk = len(bulk_indices[0])
        ind_z = bulk_indices[0]
        ind_y = bulk_indices[1]
        ind_x = bulk_indices[2]
        for point in range(nb_bulk):
            file_bulk.write(
                f"{f'{strain[ind_z[point], ind_y[point], ind_x[point]]:.7f}': <10}\n"
            )
    file_surface.close()
    file_bulk.close()

    # write all points position / strain to file
    total_indices = np.nonzero(support)
    nb_total = len(total_indices[0])
    ind_z = total_indices[0]
    ind_y = total_indices[1]
    ind_x = total_indices[2]
    with open(
        os.path.join(
            savedir,
            "S"
            + str(scan)
            + "_threshold"
            + str(support_threshold)
            + "_bulk+surface.dat",
        ),
        "w",
    ) as file_total:
        for point in range(nb_total):
            file_total.write(
                f"{f'{strain[ind_z[point], ind_y[point], ind_x[point]]:.7f}': <10}\n"
            )

####################################
# fit the bulk strain distribution #
####################################
print(f"Min surface strain = {strain[np.nonzero(surface)].min():.5f}")
print(f"Max surface strain = {strain[np.nonzero(surface)].max():.5f}")
hist, bin_edges = np.histogram(
    strain[np.nonzero(surface)],
    bins=int(
        (strain[np.nonzero(surface)].max() - strain[np.nonzero(surface)].min())
        / bin_step
    ),
)
hist = hist.astype(float)
if normalize:
    hist = hist / nb_surface  # normalize the histogram to the number of points

x_axis = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2

fit_params = Parameters()
if fit_pdf == "skewed_gaussian":
    fit_params.add("amp_0", value=0.0005, min=0.000001, max=10000)
    fit_params.add("loc_0", value=0, min=-0.1, max=0.1)
    fit_params.add("sig_0", value=0.0005, min=0.0000001, max=0.1)
    fit_params.add("alpha_0", value=0, min=-10, max=10)
else:  # 'pseudovoigt'
    fit_params.add("amp_0", value=0.0005, min=0.000001, max=10000)
    fit_params.add("cen_0", value=0, min=-0.1, max=0.1)
    fit_params.add("sig_0", value=0.0005, min=0.0000001, max=0.1)
    fit_params.add("ratio_0", value=0.5, min=0, max=1)

# run the global fit to all the data sets
result = minimize(util.objective_lmfit, fit_params, args=(x_axis, hist, fit_pdf))
report_fit(result.params)
strain_fit = util.function_lmfit(
    params=result.params, x_axis=x_axis, distribution=fit_pdf
)

if fit_pdf == "skewed_gaussian":  # find the position of the mode (maximum of the pdf)
    x_mode = np.unravel_index(strain_fit.argmax(), x_axis.shape)
    step = x_axis[x_mode] - x_axis[x_mode[0] - 1]
    fine_x = np.copy(x_axis)
    for idx in range(2):
        fine_x = np.linspace(
            fine_x[x_mode] - step, fine_x[x_mode] + step, endpoint=True, num=1000
        )
        fine_y = util.function_lmfit(
            params=result.params, x_axis=fine_x, distribution=fit_pdf
        )
        diff_fit = np.gradient(fine_y, fine_x[1] - fine_x[0])
        (x_mode,) = np.unravel_index(abs(diff_fit).argmin(), fine_x.shape)
        step = fine_x[x_mode] - fine_x[x_mode - 1]
    strain_mode = fine_x[x_mode]

###################################################
# plot the strain histogram for the surface layer #
###################################################
surface_xaxis = np.copy(x_axis)  # will be used for the overlay plot
surface_hist = np.copy(hist)  # will be used for the overlay plot
fig, ax = plt.subplots(nrows=1, ncols=1)
if plot_scale == "log":
    hist[hist == 0] = np.nan
    ax.plot(
        x_axis,
        np.log10(hist),
        linestyle="",
        marker="o",
        markeredgecolor="r",
        fillstyle="none",
    )
    (fit,) = ax.plot(x_axis, np.log10(strain_fit), linestyle="-", color="r")
else:
    ax.plot(
        x_axis, hist, linestyle="", marker="o", markeredgecolor="r", fillstyle="none"
    )
    (fit,) = ax.plot(x_axis, strain_fit, linestyle="-", color="r")
if xlim is None:
    ax.set_xlim(-max(abs(x_axis)), max(abs(x_axis)))
else:
    if len(xlim) != 2:
        raise ValueError("xlim=[min, max] expected")
    ax.set_xlim(xlim[0], xlim[1])
if ylim is not None:
    if len(ylim) != 2:
        raise ValueError("ylim=[min, max] expected")
    ax.set_ylim(ylim[0], ylim[1])

vline1 = ax.axvline(x=0, ymin=0, ymax=1, color="k", linestyle="dotted", linewidth=1.0)
ax.tick_params(labelbottom=False, labelleft=False)
fig.savefig(
    savedir + "S" + str(scan) + "_surface_strain_iso" + str(support_threshold) + ".png"
)

ax.set_xlabel("strain")
vline2 = ax.axvline(
    x=np.mean(strain[np.nonzero(surface)]),
    ymin=0,
    ymax=1,
    color="r",
    linestyle="dashed",
)
legend_fit = ax.legend(handles=[fit], labels=[fit_pdf], loc="upper left", frameon=False)
ax.legend(
    handles=(vline1, vline2),
    labels=("strain=0", "<surface>"),
    loc="upper right",
    frameon=False,
)
ax.add_artist(legend_fit)
ax.set_title(
    f"S{scan:d} histogram of the strain for {nb_surface:d} surface points\n"
    f"Modulus threshold={support_threshold}"
)
fig.text(0.65, 0.70, f"<strain>={np.mean(strain[np.nonzero(surface)]):.2e}")
fig.text(0.65, 0.65, f"std(strain)={np.std(strain[np.nonzero(surface)]):.2e}")

if fit_pdf == "skewed_gaussian":
    fig.text(0.13, 0.76, f"SK_max @ strain={strain_mode:.2e}")
    fig.text(
        0.13,
        0.66,
        f"SK std={result.params['sig_0'].value:.2e}\n"
        f"   +/-{result.params['sig_0'].stderr:.2e}",
    )
else:
    fig.text(
        0.15,
        0.70,
        f"PDF center={result.params['cen_0'].value:.2e}\n"
        f"   +/-{result.params['cen_0'].stderr:.2e}",
    )
    fig.text(
        0.15,
        0.60,
        f"PDF std={result.params['sig_0'].value:.2e}\n"
        f"   +/-{result.params['sig_0'].stderr:.2e}",
    )
    fig.text(
        0.15,
        0.50,
        f"PDF ratio={result.params['ratio_0'].value:.2e}\n"
        f"   +/-{result.params['ratio_0'].stderr:.2e}",
    )
plt.pause(0.1)
fig.savefig(
    savedir
    + "S"
    + str(scan)
    + "_surface_strain_iso"
    + str(support_threshold)
    + "_labels.png"
)

####################################
# fit the bulk strain distribution #
####################################
print(f"Min bulk strain = {strain[np.nonzero(bulk)].min():.5f}")
print(f"Max bulk strain = {strain[np.nonzero(bulk)].max():.5f}")
hist, bin_edges = np.histogram(
    strain[np.nonzero(bulk)],
    bins=int(
        (strain[np.nonzero(bulk)].max() - strain[np.nonzero(bulk)].min()) / bin_step
    ),
)
hist = hist.astype(float)
if normalize:
    hist = hist / nb_bulk  # normalize the histogram to the number of points

x_axis = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2

fit_params = Parameters()
if fit_pdf == "skewed_gaussian":
    fit_params.add("amp_0", value=0.001, min=0.00000001, max=100000)
    fit_params.add("loc_0", value=0, min=-0.1, max=0.1)
    fit_params.add("sig_0", value=0.0005, min=0.0000001, max=0.1)
    fit_params.add("alpha_0", value=0, min=-10, max=10)
else:  # 'pseudovoigt'
    fit_params.add("amp_0", value=0.001, min=0.00000001, max=100000)
    fit_params.add("cen_0", value=0, min=-0.1, max=0.1)
    fit_params.add("sig_0", value=0.0005, min=0.0000001, max=0.1)
    fit_params.add("ratio_0", value=0.5, min=0, max=1)

# run the global fit to all the data sets
result = minimize(util.objective_lmfit, fit_params, args=(x_axis, hist, fit_pdf))
report_fit(result.params)
strain_fit = util.function_lmfit(
    params=result.params, x_axis=x_axis, distribution=fit_pdf
)

if fit_pdf == "skewed_gaussian":  # find the position of the mode (maximum of the pdf)
    x_mode = np.unravel_index(strain_fit.argmax(), x_axis.shape)
    step = x_axis[x_mode] - x_axis[x_mode[0] - 1]
    fine_x = np.copy(x_axis)
    for idx in range(2):
        fine_x = np.linspace(
            fine_x[x_mode] - step, fine_x[x_mode] + step, endpoint=True, num=1000
        )
        fine_y = util.function_lmfit(
            params=result.params, x_axis=fine_x, distribution=fit_pdf
        )
        diff_fit = np.gradient(fine_y, fine_x[1] - fine_x[0])
        (x_mode,) = np.unravel_index(abs(diff_fit).argmin(), fine_x.shape)
        step = fine_x[x_mode] - fine_x[x_mode - 1]
    strain_mode = fine_x[x_mode]

##########################################
# plot the strain histogram for the bulk #
##########################################
bulk_xaxis = np.copy(x_axis)  # will be used for the overlay plot
bulk_hist = np.copy(hist)  # will be used for the overlay plot
fig, ax = plt.subplots(nrows=1, ncols=1)
if plot_scale == "log":
    hist[hist == 0] = np.nan
    ax.plot(
        x_axis,
        np.log10(hist),
        linestyle="",
        marker="o",
        markeredgecolor="b",
        fillstyle="none",
    )
    (fit,) = ax.plot(x_axis, np.log10(strain_fit), linestyle="-", color="b")
else:
    ax.plot(
        x_axis, hist, linestyle="", marker="o", markeredgecolor="b", fillstyle="none"
    )
    (fit,) = ax.plot(x_axis, strain_fit, linestyle="-", color="b")
if xlim is None:
    ax.set_xlim(-max(abs(x_axis)), max(abs(x_axis)))
else:
    if len(xlim) != 2:
        raise ValueError("xlim=[min, max] expected")
    ax.set_xlim(xlim[0], xlim[1])
if ylim is not None:
    if len(ylim) != 2:
        raise ValueError("ylim=[min, max] expected")
    ax.set_ylim(ylim[0], ylim[1])

vline1 = ax.axvline(x=0, ymin=0, ymax=1, color="k", linestyle="dotted", linewidth=1.0)
ax.tick_params(labelbottom=False, labelleft=False)
fig.savefig(savedir + f"S{scan}_bulk_strain_iso{support_threshold}.png")

ax.set_xlabel("strain")
vline2 = ax.axvline(
    x=np.mean(strain[np.nonzero(bulk)]), ymin=0, ymax=1, color="b", linestyle="dashed"
)
legend_fit = ax.legend(handles=[fit], labels=[fit_pdf], loc="upper left", frameon=False)
ax.legend(
    handles=(vline1, vline2),
    labels=("strain=0", "<bulk>"),
    loc="upper right",
    frameon=False,
)
ax.add_artist(legend_fit)
ax.set_title(
    f"S{scan:d} histogram for {nb_bulk:d} bulk points\n"
    f"Modulus threshold={support_threshold}"
)
fig.text(0.65, 0.70, f"<strain>={np.mean(strain[np.nonzero(bulk)]):.2e}")
fig.text(0.65, 0.65, f"std(strain)={np.std(strain[np.nonzero(bulk)]):.2e}")

if fit_pdf == "skewed_gaussian":
    fig.text(0.13, 0.76, f"SK_max @ strain={strain_mode:.2e}")
    fig.text(
        0.13,
        0.66,
        f"SK std={result.params['sig_0'].value:.2e}\n"
        f"   +/-{result.params['sig_0'].stderr:.2e}",
    )
else:
    fig.text(
        0.15,
        0.70,
        f"PDF center={result.params['cen_0'].value:.2e}\n"
        f"   +/-{result.params['cen_0'].stderr:.2e}",
    )
    fig.text(
        0.15,
        0.60,
        f"PDF std={result.params['sig_0'].value:.2e}\n"
        f"   +/-{result.params['sig_0'].stderr:.2e}",
    )
    fig.text(
        0.15,
        0.50,
        f"PDF ratio={result.params['ratio_0'].value:.2e}\n"
        f"   +/-{result.params['ratio_0'].stderr:.2e}",
    )
plt.pause(0.1)
fig.savefig(
    savedir
    + "S"
    + str(scan)
    + "_bulk_strain_iso"
    + str(support_threshold)
    + "_labels.png"
)

nb_total = len(np.nonzero(support)[0])
print(
    f"Sanity check: Total points = {nb_total:d}, "
    f"surface+bulk = {nb_surface + nb_bulk:d}",
)
######################################################################
# plot the overlay of strain histograms for the bulk and the surface #
######################################################################
fig, ax = plt.subplots(nrows=1, ncols=1)
if plot_scale == "log":
    surface_hist[surface_hist == 0] = np.nan
    bulk_hist[bulk_hist == 0] = np.nan
    ax.bar(
        x=surface_xaxis,
        height=np.log10(surface_hist),
        width=surface_xaxis[1] - surface_xaxis[0],
        color="r",
        alpha=0.5,
    )
    ax.bar(
        x=bulk_xaxis,
        height=np.log10(bulk_hist),
        width=bulk_xaxis[1] - bulk_xaxis[0],
        color="b",
        alpha=0.5,
    )
else:
    ax.bar(
        x=surface_xaxis,
        height=surface_hist,
        width=surface_xaxis[1] - surface_xaxis[0],
        color="r",
        alpha=0.5,
    )
    ax.bar(
        x=bulk_xaxis,
        height=bulk_hist,
        width=bulk_xaxis[1] - bulk_xaxis[0],
        color="b",
        alpha=0.5,
    )

if xlim is None:
    ax.set_xlim(
        -max(abs(surface_xaxis), abs(bulk_xaxis)),
        max(abs(surface_xaxis), abs(bulk_xaxis)),
    )
else:
    if len(xlim) != 2:
        raise ValueError("xlim=[min, max] expected")
    ax.set_xlim(xlim[0], xlim[1])
if ylim is not None:
    if len(ylim) != 2:
        raise ValueError("ylim=[min, max] expected")
    ax.set_ylim(ylim[0], ylim[1])

ax.axvline(x=0, ymin=0, ymax=1, color="k", linestyle="dotted", linewidth=1.0)
ax.tick_params(length=tick_length, width=tick_width)
fig.savefig(
    savedir
    + "S"
    + str(scan)
    + "_overlay_strain_iso"
    + str(support_threshold)
    + "_labels.png"
)
ax.tick_params(labelbottom=False, labelleft=False)
ax.spines["right"].set_linewidth(tick_width)
ax.spines["left"].set_linewidth(tick_width)
ax.spines["top"].set_linewidth(tick_width)
ax.spines["bottom"].set_linewidth(tick_width)
fig.savefig(
    savedir + "S" + str(scan) + "_overlay_strain_iso" + str(support_threshold) + ".png"
)
plt.ioff()
plt.show()
