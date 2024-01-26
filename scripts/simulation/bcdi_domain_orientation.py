#!/usr/bin/env python3

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import datetime
import gc
import sys
import time
import tkinter as tk
from tkinter import filedialog

import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import peak_local_max

import bcdi.graph.graph_utils as gu
import bcdi.postprocessing.postprocessing_utils as pu
import bcdi.simulation.simulation_utils as simu
import bcdi.utils.utilities as util
from bcdi.experiment.detector import create_detector
from bcdi.graph.colormap import ColormapFactory

helptext = """
Calculate the position of the Bragg peaks for a mesocrystal given the lattice type,
the unit cell parameter and beamline-related parameters. Assign 3D Gaussians to each
lattice point and rotates the unit cell in order to maximize the cross-correlation of
the simulated data with experimental data. The experimental data should be sparse
(using a photon threshold), and Bragg peaks maximum must be clearly identifiable.

Laboratory frame convention (CXI): z downstream, y vertical up, x outboard.
Reciprocal space basis:            qx downstream, qz vertical up, qy outboard.
"""

datadir = "D:/data/P10_March2020_CDI/test_april/data/align_06_00248/pynx/"
savedir = "D:/data/P10_March2020_CDI/test_april/data/align_06_00248/simu/"
comment = ""  # should start with _
################
# sample setup #
################
unitcell = "fcc"  # supported unit cells: 'cubic', 'bcc', 'fcc', 'bct'
# It can be a number or tuple of numbers depending on the unit cell.
unitcell_ranges = [22.9, 22.9]  # in nm, values of the unit cell parameters to test
# cubic, FCC or BCC unit cells: [start, stop].
# BCT unit cell: [start1, stop1, start2, stop2]   (stop is included)
unitcell_step = 0.05  # in nm
#########################
# unit cell orientation #
#########################
angles_ranges = [
    -45,
    -45,
    -45,
    45,
    -45,
    45,
]  # [start, stop, start, stop, start, stop], in degrees
# ranges to span for the rotation around qx downstream, qz vertical up and
# qy outboard respectively (stop is included)
angular_step = 5  # in degrees
#######################
# beamline parameters #
#######################
sdd = 4.95  # in m, sample to detector distance
energy = 8250  # in ev X-ray energy
##################
# detector setup #
##################
detector = "Eiger4M"  # "Eiger2M" or "Maxipix" or "Eiger4M"
direct_beam = (
    1303,
    1127,
)  # tuple of int (vertical, horizontal): position of the direct beam in pixels
# this parameter is important for gridding the data onto the laboratory frame
roi_detector = []
# [direct_beam[0] - 972, direct_beam[0] + 972,
# direct_beam[1] - 883, direct_beam[1] + 883]
# [Vstart, Vstop, Hstart, Hstop], leave [] to use the full detector
binning = [4, 4, 4]  # binning of the detector
##########################
# peak detection options #
##########################
photon_threshold = 1000  # intensity below this value will be set to 0
min_distance = 50  # minimum distance between Bragg peaks in pixels
peak_width = 0  # the total width will be (2*peak_width+1)
###########
# options #
###########
kernel_length = 11  # width of the 3D gaussian window
debug = True  # True to see more plots
correct_background = False  # True to create a 3D background
bckg_method = "normalize"  # 'subtract' or 'normalize'

##################################
# end of user-defined parameters #
##################################

#######################
# Initialize detector #
#######################
detector = create_detector(name=detector, binning=binning, roi=roi_detector)

###################
# define colormap #
###################
bad_color = "1.0"  # white background
my_cmap = ColormapFactory(bad_color=bad_color).cmap
plt.ion()

###################################
# load experimental data and mask #
###################################
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    initialdir=datadir, title="Select the data to fit", filetypes=[("NPZ", "*.npz")]
)
data = np.load(file_path)["data"]
nz, ny, nx = data.shape
print(
    "Sparsity of the data:",
    str(f"{(data == 0).sum() / (nz * ny * nx) * 100:.2f}"),
    "%",
)

try:
    file_path = filedialog.askopenfilename(
        initialdir=datadir, title="Select the mask", filetypes=[("NPZ", "*.npz")]
    )
    mask = np.load(file_path)["mask"]

    data[np.nonzero(mask)] = 0
    del mask
    gc.collect()
except FileNotFoundError:
    pass

try:
    file_path = filedialog.askopenfilename(
        initialdir=datadir, title="Select q values", filetypes=[("NPZ", "*.npz")]
    )
    exp_qvalues = np.load(file_path)
    qvalues_flag = True
except FileNotFoundError:
    exp_qvalues = None
    qvalues_flag = False

##########################
# apply photon threshold #
##########################
data[data < photon_threshold] = 0
print(
    "Sparsity of the data after photon threshold:",
    str(f"{(data == 0).sum() / (nz * ny * nx) * 100:.2f}"),
    "%",
)

######################
# calculate q values #
######################
if unitcell == "bct":
    pivot, _, q_values, _, _ = simu.lattice(
        energy=energy,
        sdd=sdd,
        direct_beam=direct_beam,
        detector=detector,
        unitcell=unitcell,
        unitcell_param=[unitcell_ranges[0], unitcell_ranges[2]],
        euler_angles=[0, 0, 0],
        offset_indices=True,
    )
else:
    pivot, _, q_values, _, _ = simu.lattice(
        energy=energy,
        sdd=sdd,
        direct_beam=direct_beam,
        detector=detector,
        unitcell=unitcell,
        unitcell_param=unitcell_ranges[0],
        euler_angles=[0, 0, 0],
        offset_indices=True,
    )

nbz, nby, nbx = len(q_values[0]), len(q_values[1]), len(q_values[2])
comment = (
    comment
    + str(nbz)
    + "_"
    + str(nby)
    + "_"
    + str(nbx)
    + "_"
    + str(binning[0])
    + "_"
    + str(binning[1])
    + "_"
    + str(binning[2])
)

if (nbz != nz) or (nby != ny) or (nbx != nx):
    print(
        "The experimental data and calculated q values have different shape,"
        ' check "roi_detector" parameter!'
    )
    sys.exit()

print("Origin of the reciprocal space at pixel", pivot)

##########################
# plot experimental data #
##########################
if debug:
    gu.multislices_plot(
        data,
        sum_frames=True,
        title="data",
        vmin=0,
        vmax=np.log10(data).max(),
        scale="log",
        plot_colorbar=True,
        cmap=my_cmap,
        is_orthogonal=True,
        reciprocal_space=True,
    )

    if qvalues_flag:
        gu.contour_slices(
            data,
            q_coordinates=(exp_qvalues["qx"], exp_qvalues["qz"], exp_qvalues["qy"]),
            sum_frames=True,
            title="Experimental data",
            levels=np.linspace(0, np.log10(data.max()) + 1, 20, endpoint=False),
            scale="log",
            plot_colorbar=False,
            is_orthogonal=True,
            reciprocal_space=True,
        )
    else:
        gu.contour_slices(
            data,
            q_coordinates=q_values,
            sum_frames=True,
            title="Experimental data",
            levels=np.linspace(0, np.log10(data.max()) + 1, 20, endpoint=False),
            scale="log",
            plot_colorbar=False,
            is_orthogonal=True,
            reciprocal_space=True,
        )

################################################
# remove background from the experimental data #
################################################
if correct_background:
    file_path = filedialog.askopenfilename(
        initialdir=datadir,
        title="Select the 1D background file",
        filetypes=[("NPZ", "*.npz")],
    )
    avg_background = np.load(file_path)["background"]
    distances = np.load(file_path)["distances"]

    if qvalues_flag:
        data = util.remove_avg_background(
            array=data,
            avg_background=avg_background,
            avg_qvalues=distances,
            q_values=(exp_qvalues["qx"], exp_qvalues["qz"], exp_qvalues["qy"]),
            method=bckg_method,
        )
    else:
        print("Using calculated q values for background subtraction")
        data = util.remove_avg_background(
            array=data,
            q_values=q_values,
            avg_background=avg_background,
            avg_qvalues=distances,
            method=bckg_method,
        )

    np.savez_compressed(datadir + "data-background_" + comment + ".npz", data=data)

    gu.multislices_plot(
        data,
        sum_frames=True,
        title="Background subtracted data",
        vmin=0,
        vmax=np.log10(data).max(),
        scale="log",
        plot_colorbar=True,
        cmap=my_cmap,
        is_orthogonal=True,
        reciprocal_space=True,
    )

#############################################
# find Bragg peaks in the experimental data #
#############################################
density_map = np.copy(data)

# find peaks
local_maxi = peak_local_max(
    density_map, exclude_border=False, min_distance=min_distance, indices=True
)
nb_peaks = local_maxi.shape[0]
print("Number of Bragg peaks isolated:", nb_peaks)
print("Bragg peaks positions:")
print(local_maxi)

density_map[:] = 0

for idx in range(nb_peaks):
    piz, piy, pix = local_maxi[idx]
    density_map[
        piz - peak_width : piz + peak_width + 1,
        piy - peak_width : piy + peak_width + 1,
        pix - peak_width : pix + peak_width + 1,
    ] = 1

nonzero_indices = np.nonzero(density_map)
bragg_peaks = density_map[
    nonzero_indices
]  # 1D array of length: nb_peaks*(2*peak_width+1)**3

if debug:
    gu.multislices_plot(
        density_map,
        sum_frames=True,
        title="Bragg peaks positions",
        slice_position=pivot,
        vmin=0,
        vmax=1,
        scale="linear",
        cmap=my_cmap,
        is_orthogonal=True,
        reciprocal_space=True,
    )
    plt.pause(0.1)

#########################
# define the peak shape #
#########################
peak_shape = pu.blackman_window(
    shape=(kernel_length, kernel_length, kernel_length), normalization=100
)

#####################################
# define the list of angles to test #
#####################################
angles_qx = np.linspace(
    start=angles_ranges[0],
    stop=angles_ranges[1],
    num=max(1, np.rint((angles_ranges[1] - angles_ranges[0]) / angular_step) + 1),
)
angles_qz = np.linspace(
    start=angles_ranges[2],
    stop=angles_ranges[3],
    num=max(1, np.rint((angles_ranges[3] - angles_ranges[2]) / angular_step) + 1),
)
angles_qy = np.linspace(
    start=angles_ranges[4],
    stop=angles_ranges[5],
    num=max(1, np.rint((angles_ranges[5] - angles_ranges[4]) / angular_step) + 1),
)
nb_angles = len(angles_qx) * len(angles_qz) * len(angles_qy)
print("Number of angles to test: ", nb_angles)

####################################################
# loop over rotation angles and lattice parameters #
####################################################
start = time.time()
if unitcell == "bct":
    a_values = np.linspace(
        start=unitcell_ranges[0],
        stop=unitcell_ranges[1],
        num=max(
            1, np.rint((unitcell_ranges[1] - unitcell_ranges[0]) / unitcell_step) + 1
        ),
    )
    c_values = np.linspace(
        start=unitcell_ranges[2],
        stop=unitcell_ranges[3],
        num=max(
            1, np.rint((unitcell_ranges[3] - unitcell_ranges[2]) / unitcell_step) + 1
        ),
    )
    nb_lattices = len(a_values) * len(c_values)
    print("Number of lattice parameters to test: ", nb_lattices)
    print("Total number of iterations: ", nb_angles * nb_lattices)
    corr = np.zeros(
        (len(angles_qx), len(angles_qz), len(angles_qy), len(a_values), len(c_values))
    )
    for idz, alpha in enumerate(angles_qx):
        for idy, beta in enumerate(angles_qz):
            for idx, gamma in enumerate(angles_qy):
                for idw, a in enumerate(a_values):
                    for idv, c in enumerate(c_values):
                        _, _, _, rot_lattice, _ = simu.lattice(
                            energy=energy,
                            sdd=sdd,
                            direct_beam=direct_beam,
                            detector=detector,
                            unitcell=unitcell,
                            unitcell_param=(a, c),
                            euler_angles=(alpha, beta, gamma),
                            offset_indices=False,
                        )
                        # peaks in the format [[h, l, k], ...]:
                        # CXI convention downstream , vertical up, outboard

                        # assign the peak shape to each lattice point
                        struct_array = simu.assign_peakshape(
                            array_shape=(nbz, nby, nbx),
                            lattice_list=rot_lattice,
                            peak_shape=peak_shape,
                            pivot=pivot,
                        )

                        # calculate the correlation between experimental data
                        # and simulated data
                        corr[idz, idy, idx, idw, idv] = np.multiply(
                            bragg_peaks, struct_array[nonzero_indices]
                        ).sum()
else:
    a_values = np.linspace(
        start=unitcell_ranges[0],
        stop=unitcell_ranges[1],
        num=max(
            1, np.rint((unitcell_ranges[1] - unitcell_ranges[0]) / unitcell_step) + 1
        ),
    )
    nb_lattices = len(a_values)
    print("Number of lattice parameters to test: ", nb_lattices)
    print("Total number of iterations: ", nb_angles * nb_lattices)
    corr = np.zeros((len(angles_qx), len(angles_qz), len(angles_qy), len(a_values)))
    for idz, alpha in enumerate(angles_qx):
        for idy, beta in enumerate(angles_qz):
            for idx, gamma in enumerate(angles_qy):
                for idw, a in enumerate(a_values):
                    _, _, _, rot_lattice, _ = simu.lattice(
                        energy=energy,
                        sdd=sdd,
                        direct_beam=direct_beam,
                        detector=detector,
                        unitcell=unitcell,
                        unitcell_param=a,
                        euler_angles=(alpha, beta, gamma),
                        offset_indices=False,
                    )
                    # peaks in the format [[h, l, k], ...]:
                    # CXI convention downstream , vertical up, outboard

                    # assign the peak shape to each lattice point
                    struct_array = simu.assign_peakshape(
                        array_shape=(nbz, nby, nbx),
                        lattice_list=rot_lattice,
                        peak_shape=peak_shape,
                        pivot=pivot,
                    )

                    # calculate the correlation between experimental data
                    # and simulated data
                    corr[idz, idy, idx, idw] = np.multiply(
                        bragg_peaks, struct_array[nonzero_indices]
                    ).sum()

end = time.time()
print(
    "\nTime ellapsed in the loop over angles and lattice parameters:",
    str(datetime.timedelta(seconds=int(end - start))),
)

##########################################
# plot the correlation matrix at maximum #
##########################################
comment = comment + "_" + unitcell

if unitcell == "bct":  # corr is 5D
    piz, piy, pix, piw, piv = np.unravel_index(abs(corr).argmax(), corr.shape)
    alpha, beta, gamma = angles_qx[piz], angles_qz[piy], angles_qy[pix]
    best_param = a_values[piw], c_values[piv]
    text = (
        unitcell
        + f" unit cell of parameter(s) = {best_param[0]:.2f} nm, {best_param[1]:.2f}"
        + " nm"
    )
    print(
        "Maximum correlation for (angle_qx, angle_qz, angle_qy) = "
        f"{alpha:.2f}, {beta:.2f}, {gamma:.2f}"
    )
    print("Maximum correlation for a", text)
    corr_angles = np.copy(corr[:, :, :, piw, piv])
    corr_lattice = np.copy(corr[piz, piy, pix, :, :])

    vmin = corr_lattice.min()
    vmax = 1.1 * corr_lattice.max()
    save_lattice = True
    if all(corr_lattice.shape[idx] > 1 for idx in range(corr_lattice.ndim)):  # 2D
        fig, ax = plt.subplots(nrows=1, ncols=1)
        plt0 = ax.contourf(
            c_values,
            a_values,
            corr_lattice,
            np.linspace(vmin, vmax, 20, endpoint=False),
            cmap=my_cmap,
        )
        plt.colorbar(plt0, ax=ax)
        ax.set_ylabel("a parameter (nm)")
        ax.set_xlabel("c parameter (nm)")
        ax.set_title("Correlation map for lattice parameters")
    else:  # 1D or 0D
        nonzero_dim = np.nonzero(np.asarray(corr_lattice.shape) != 1)[0]
        if len(nonzero_dim) == 0:  # 0D
            print("The unit cell lattice parameters are not scanned")
            save_lattice = False
        else:  # 1D
            corr_lattice = np.squeeze(corr_lattice)
            labels = ["a parameter (nm)", "c parameter (nm)"]
            fig = plt.figure()
            if nonzero_dim[0] == 0:
                plt.plot(a_values, corr_lattice, ".-r")
            else:  # index 1
                plt.plot(c_values, corr_lattice, ".-r")
            plt.xlabel(labels[nonzero_dim[0]])
            plt.ylabel("Correlation")
    plt.pause(0.1)
    if save_lattice:
        plt.savefig(
            savedir
            + "correlation_lattice_"
            + comment
            + f"_param a={best_param[0]:.2f}nm,c={best_param[1]:.2f}nm"
            + ".png"
        )

else:  # corr is 4D
    piz, piy, pix, piw = np.unravel_index(abs(corr).argmax(), corr.shape)
    alpha, beta, gamma = angles_qx[piz], angles_qz[piy], angles_qy[pix]
    best_param = a_values[piw]
    text = unitcell + " unit cell of parameter = " + str(f"{best_param:.2f}") + " nm"
    print(
        "Maximum correlation for (angle_qx, angle_qz, angle_qy) = "
        f"{alpha:.2f}, {beta:.2f}, {gamma:.2f}"
    )
    print("Maximum correlation for a", text)
    corr_angles = np.copy(corr[:, :, :, piw])
    corr_lattice = np.copy(corr[piz, piy, pix, :])

    fig = plt.figure()
    plt.plot(a_values, corr_lattice, ".r")
    plt.xlabel("a parameter (nm)")
    plt.ylabel("Correlation")
    plt.pause(0.1)
    plt.savefig(
        savedir
        + "correlation_lattice_"
        + comment
        + f"_param a={best_param:.2f}nm"
        + ".png"
    )

vmin = corr_angles.min()
vmax = 1.1 * corr_angles.max()
save_angles = True
if all(corr_angles.shape[idx] > 1 for idx in range(corr_angles.ndim)):  # 3D
    fig, _, _ = gu.contour_slices(
        corr_angles,
        (angles_qx, angles_qz, angles_qy),
        sum_frames=False,
        title="Correlation map for rotation angles",
        slice_position=[piz, piy, pix],
        plot_colorbar=True,
        levels=np.linspace(vmin, vmax, 20, endpoint=False),
        is_orthogonal=True,
        reciprocal_space=True,
        cmap=my_cmap,
    )
    fig.text(0.60, 0.25, "Kernel size = " + str(kernel_length) + " pixels", size=12)
else:
    # find which angle is 1D
    nonzero_dim = np.nonzero(np.asarray(corr_angles.shape) != 1)[0]
    corr_angles = np.squeeze(corr_angles)
    labels = [
        "rotation around qx (deg)",
        "rotation around qz (deg)",
        "rotation around qy (deg)",
    ]
    if corr_angles.ndim == 2:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        if (nonzero_dim[0] == 0) and (nonzero_dim[1] == 1):
            plt0 = ax.contourf(
                angles_qz,
                angles_qx,
                corr_angles,
                np.linspace(vmin, vmax, 20, endpoint=False),
                cmap=my_cmap,
            )
        elif (nonzero_dim[0] == 0) and (nonzero_dim[1] == 2):
            plt0 = ax.contourf(
                angles_qy,
                angles_qx,
                corr_angles,
                np.linspace(vmin, vmax, 20, endpoint=False),
                cmap=my_cmap,
            )
        else:
            plt0 = ax.contourf(
                angles_qy,
                angles_qz,
                corr_angles,
                np.linspace(vmin, vmax, 20, endpoint=False),
                cmap=my_cmap,
            )
        plt.colorbar(plt0, ax=ax)
        ax.set_ylabel(labels[nonzero_dim[0]])
        ax.set_xlabel(labels[nonzero_dim[1]])
        ax.set_title("Correlation map for rotation angles")
    else:  # 1D or 0D
        if len(nonzero_dim) == 0:  # 0D
            print("The unit cell rotation angles are not scanned")
            save_angles = False
        else:  # 1D
            fig = plt.figure()
            if nonzero_dim[0] == 0:
                plt.plot(angles_qx, corr_angles, ".-r")
            elif nonzero_dim[0] == 1:
                plt.plot(angles_qz, corr_angles, ".-r")
            else:  # index 2
                plt.plot(angles_qy, corr_angles, ".-r")
            plt.xlabel(labels[nonzero_dim[0]])
            plt.ylabel("Correlation")

plt.pause(0.1)
if save_angles:
    plt.savefig(
        savedir
        + "correlation_angles_"
        + comment
        + f"_rot_{alpha:.2f}_{beta:.2f}_{gamma:.2f}"
        + ".png"
    )

###################################################
# calculate the lattice at calculated best values #
###################################################
_, _, _, rot_lattice, peaks = simu.lattice(
    energy=energy,
    sdd=sdd,
    direct_beam=direct_beam,
    detector=detector,
    unitcell=unitcell,
    unitcell_param=best_param,
    euler_angles=(alpha, beta, gamma),
    offset_indices=False,
)
# peaks in the format [[h, l, k], ...]:
# CXI convention downstream , vertical up, outboard

nb_peaks = len(peaks)
print("Simulated Bragg peaks hkls and position:")
print("hlk (qx, qz, qy)       indices (in pixels)")
for idx in range(nb_peaks):
    print(peaks[idx], " : ", rot_lattice[idx])
# assign the peak shape to each lattice point
struct_array = simu.assign_peakshape(
    array_shape=(nbz, nby, nbx),
    lattice_list=rot_lattice,
    peak_shape=peak_shape,
    pivot=pivot,
)

#######################################################
# plot the overlay of experimental and simulated data #
#######################################################
if unitcell == "bct":
    text = (
        unitcell
        + f" unit cell of parameter(s) = {best_param[0]:.2f} nm, {best_param[1]:.2f}"
        + " nm"
    )
else:
    text = unitcell + " unit cell of parameter(s) = " + str(f"{best_param:.2f}") + " nm"

plot_max = 2 * peak_shape.sum(axis=0).max()
density_map[np.nonzero(density_map)] = 10 * plot_max
fig, _, _ = gu.multislices_plot(
    struct_array + density_map,
    sum_frames=True,
    title="Overlay",
    vmin=0,
    vmax=plot_max,
    plot_colorbar=True,
    scale="linear",
    is_orthogonal=True,
    reciprocal_space=True,
)
fig.text(0.5, 0.25, "Energy = " + str(energy / 1000) + " keV", size=12)
fig.text(0.5, 0.20, "SDD = " + str(sdd) + " m", size=12)
fig.text(0.5, 0.15, text, size=12)
fig.text(
    0.5,
    0.10,
    "Rotation of the unit cell in degrees (Qx, Qz, Qy) ="
    f" {alpha:.2f}, {beta:.2f}, {gamma:.2f}",
    size=12,
)
plt.pause(0.1)
plt.savefig(
    savedir + "Overlay_" + comment + "_corr=" + str(f"{corr.max():.2f}") + ".png"
)

if debug:
    fig, _, _ = gu.multislices_plot(
        struct_array,
        sum_frames=True,
        title="Simulated diffraction pattern",
        vmin=0,
        vmax=plot_max,
        plot_colorbar=False,
        scale="linear",
        is_orthogonal=True,
        reciprocal_space=True,
    )
    fig.text(0.5, 0.25, "Energy = " + str(energy / 1000) + " keV", size=12)
    fig.text(0.5, 0.20, "SDD = " + str(sdd) + " m", size=12)
    fig.text(0.5, 0.15, text, size=12)
    fig.text(
        0.5,
        0.10,
        "Rotation of the unit cell in degrees (Qx, Qz, Qy) ="
        f" {alpha:.2f}, {beta:.2f}, {gamma:.2f}",
        size=12,
    )
    plt.pause(0.1)

    fig, _, _ = gu.contour_slices(
        struct_array,
        q_coordinates=q_values,
        sum_frames=True,
        title="Simulated diffraction pattern",
        cmap=my_cmap,
        levels=np.linspace(
            struct_array.min() + plot_max / 100, plot_max, 20, endpoint=False
        ),
        plot_colorbar=True,
        scale="linear",
        is_orthogonal=True,
        reciprocal_space=True,
    )
    fig.text(0.5, 0.25, "Energy = " + str(energy / 1000) + " keV", size=12)
    fig.text(0.5, 0.20, "SDD = " + str(sdd) + " m", size=12)
    fig.text(0.5, 0.15, text, size=12)
    fig.text(
        0.5,
        0.10,
        "Rotation of the unit cell in degrees (Qx, Qz, Qy) ="
        f" {alpha:.2f}, {beta:.2f}, {gamma:.2f}",
        size=12,
    )
    plt.pause(0.1)

plt.ioff()
plt.show()
