#!/usr/bin/env python3

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import gc
import os
import sys
import tkinter as tk
from tkinter import filedialog

import numpy as np
from matplotlib import pyplot as plt
from numpy.fft import fftn, fftshift
from numpy.random import poisson
from scipy.interpolate import RegularGridInterpolator

import bcdi.graph.graph_utils as gu
import bcdi.postprocessing.postprocessing_utils as pu
import bcdi.simulation.simulation_utils as simu
import bcdi.utils.utilities as util
from bcdi.experiment.setup import Setup
from bcdi.graph.colormap import ColormapFactory

helptext = """
Using a support created from a reconstructed object (real space), calculate the
diffraction pattern depending on several parameters: detector size, detector distance,
presence/width of a detector gap, Poisson noise, user-defined phase.

The provided reconstructed object is expected to be orthogonalized, in the laboratory
frame.
"""

scan = 2227  # spec scan number
datadir = "C:/Users/Jerome/Documents/data/BCDI_isosurface/S" + str(scan) + "/test/"
# "D:/data/BCDI_isosurface/S"+str(scan)+"/test/"

original_sdd = 0.50678  # 1.0137
# in m, sample to detector distance of the provided reconstruction
simulated_sdd = (
    0.50678  # in m, sample to detector distance for the simulated diffraction pattern
)
sdd_change_mode = "real_space"  # 'real_space' or 'reciprocal_space',
# for compensating the detector distance change
# in real_space, it will interpolate the support
# if 'reciprocal_space', it will interpolate the diffraction calculated on pad_size
energy = 9000.0 - 6  # x-ray energy in eV, 6eV offset at ID01
voxel_size = 3
# in nm, voxel size of the reconstruction, should be eaqual in each direction
photon_threshold = 0  # 0.75
photon_number = (
    5e7  # * 1011681 / 469091 # total number of photons in the array, usually around 5e7
)
pad_ortho = False  # True to pad before interpolating into detector frame,
# False after (saves memory)
# True is the only choice if the compensated object is larger than the original
# array shape (it gets truncated)
orthogonal_frame = (
    False  # set to False to interpolate the diffraction pattern in the detector frame
)
rotate_crystal = (
    True  # if True, the crystal will be rotated as it was during the experiment
)
support_threshold = 0.24  # threshold for support determination
beamline = "ID01"  # name of the beamline, used for orthogonalisation
# supported beamlines: 'ID01', 'SIXS_2018', 'SIXS_2019', 'CRISTAL', 'P10'
beam_direction = np.array([1, 0, 0])  # incident beam along z
rocking_angle = "outofplane"  # "outofplane" or "inplane"
outofplane_angle = 35.3240  # detector delta ID01
inplane_angle = -1.6029  # detector nu ID01
grazing_angle = 0  # in degrees, incident angle for in-plane rocking curves (eta ID01)
tilt_angle = 0.0102  # angular step size for rocking angle, eta ID01
detector = "Maxipix"  # "Eiger2M", "Maxipix", "Eiger4M", "Merlin", "Timepix" or "Dummy"

set_gap = True
# set to True if you want to use the detector gap in the simulation (updates the mask)
gap_width = 6  # number of pixels to mask
gap_pixel_start = 550

flat_phase = True  # set to True to use a phase flat (0 everywhere)

include_noise = False  # set to True to include poisson noise on the data

original_size = [
    400,
    400,
    400,
]  # size of the FFT array before binning.
# It will be modify to take into account binning
# during phasing automatically. Leave it to () if the shape did not change.
binning = (1, 1, 1)  # binning factor during phasing
pad_size = [
    1000,
    1000,
    1000,
]  # will pad the array by this amount of zeroed pixels in z, y, x at both ends
# if only a number (e.g. 3), will pad to get three times the initial array size
# ! max size ~ [800, 800, 800]
crop_size = [300, 300, 300]  # will crop the array to this size

ref_axis_outplane = "y"  # "y"  # "z"  # q is supposed to be aligned along that axis
# before rotating back (nexus)
phase_range = np.pi  # for plots
strain_range = 0.001  # for plots
debug = False  # True to see all plots
save_fig = True  # if True save figures
save_data = True  # if True save data as npz and VTK
comment = ""  # should start with _

##################################
# end of user-defined parameters #
##################################

###################
# define colormap #
###################
bad_color = "1.0"  # white background
my_cmap = ColormapFactory(bad_color=bad_color).cmap

################
# define setup #
################
setup = Setup(
    beamline_name=beamline,
    energy=energy,
    outofplane_angle=outofplane_angle,
    inplane_angle=inplane_angle,
    tilt_angle=tilt_angle,
    rocking_angle=rocking_angle,
    grazing_angle=grazing_angle,
    distance=original_sdd,
    detector_name=detector,
)

#########################
# load a reconstruction #
#########################
plt.ion()
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(initialdir=datadir, filetypes=[("NPZ", "*.npz")])
npzfile = np.load(file_path)
amp = npzfile["amp"]
gu.multislices_plot(
    amp,
    sum_frames=False,
    plot_colorbar=False,
    vmin=0,
    vmax=1,
    cmap=my_cmap,
    title="Input amplitude",
)

#################################
# pad data to the original size #
#################################
print("Initial data size:", amp.shape)
if len(original_size) == 0:
    original_size = amp.shape
print("FFT size before accounting for binning", original_size)
original_size = tuple(
    [original_size[index] // binning[index] for index in range(len(binning))]
)
print("Binning used during phasing:", binning)
print("Padding back to original FFT size", original_size, "\n")
amp = util.crop_pad(array=amp, output_shape=original_size)
nz, ny, nx = amp.shape

##########################################################
# calculate q for later regridding in the detector frame #
##########################################################
kin = (
    2 * np.pi / setup.wavelength * beam_direction
)  # in laboratory frame z downstream, y vertical, x outboard
kout = (
    setup.exit_wavevector()
)  # in laboratory frame z downstream, y vertical, x outboard
q = kout - kin
Qnorm = np.linalg.norm(q)
q = q / Qnorm
Qnorm = Qnorm * 1e-10  # switch to angstroms
planar_dist = 2 * np.pi / Qnorm  # Qnorm should be in angstroms
print("Wavevector transfer [z, y, x]:", q * Qnorm)
print("Wavevector transfer: (angstroms)", str(f"{Qnorm:.4f}"))
print("Interplanar distance: (angstroms)", str(f"{planar_dist:.4f}"), "angstroms")
planar_dist = planar_dist / 10  # switch to nm

#########################################
# define the support and impose a phase #
#########################################
support = np.ones((nz, ny, nx))

if flat_phase:
    phase = np.zeros((nz, ny, nx))
else:
    comment = comment + "_phase"
    # model for paper about artefacts in BCDI
    oscillation_period = 100  # in pixels
    z, y, x = np.meshgrid(
        np.cos(np.arange(-nz // 2, nz // 2, 1) * 2 * np.pi / oscillation_period),
        np.cos(np.arange(-ny // 2, ny // 2, 1) * 2 * np.pi / oscillation_period),
        np.cos(np.arange(-nx // 2, nx // 2, 1) * 2 * np.pi / oscillation_period),
        indexing="ij",
    )
    phase = z + y + x

if debug and not flat_phase:
    gu.multislices_plot(
        phase,
        sum_frames=False,
        plot_colorbar=True,
        width_z=200,
        width_y=200,
        width_x=200,
        vmin=-phase_range,
        vmax=phase_range,
        cmap=my_cmap,
        title="Phase before wrapping\n",
    )

phase = util.wrap(phase, start_angle=-np.pi, range_angle=2 * np.pi)

support[abs(amp) < support_threshold * abs(amp).max()] = 0
del amp

volume = support.sum() * voxel_size**3  # in nm3
print("estimated volume", volume, " nm3")

phase[support == 0] = 0

if ref_axis_outplane == "x":
    _, _, strain = np.gradient(
        planar_dist / (2 * np.pi) * phase, voxel_size
    )  # q is along x after rotating the crystal
elif ref_axis_outplane == "y":
    _, strain, _ = np.gradient(
        planar_dist / (2 * np.pi) * phase, voxel_size
    )  # q is along y after rotating the crystal
elif ref_axis_outplane == "z":
    strain, _, _ = np.gradient(
        planar_dist / (2 * np.pi) * phase, voxel_size
    )  # q is along y after rotating the crystal
else:  # default is ref_axis_outplane = "y"
    _, strain, _ = np.gradient(
        planar_dist / (2 * np.pi) * phase, voxel_size
    )  # q is along y after rotating the crystal

# remove the outer layer of support for saving, because strain is undefined there
coordination_matrix = pu.calc_coordination(support, debugging=debug)
surface = np.copy(support)
surface[coordination_matrix > 23] = 0  # remove the bulk 22
bulk = support - surface
bulk[np.nonzero(bulk)] = 1

if debug and not flat_phase:
    gu.multislices_plot(
        surface,
        sum_frames=False,
        plot_colorbar=False,
        width_z=200,
        width_y=200,
        width_x=200,
        vmin=0,
        vmax=1,
        cmap=my_cmap,
        title="surface",
    )

    surface = np.multiply(surface, strain)

    gu.multislices_plot(
        surface,
        sum_frames=False,
        plot_colorbar=True,
        width_z=200,
        width_y=200,
        width_x=200,
        vmin=-strain_range,
        vmax=strain_range,
        cmap=my_cmap,
        title="surface strain",
    )

    gu.multislices_plot(
        support,
        sum_frames=True,
        plot_colorbar=False,
        cmap=my_cmap,
        title="Orthogonal support\n",
    )

    gu.multislices_plot(
        phase,
        sum_frames=False,
        plot_colorbar=True,
        width_z=200,
        width_y=200,
        width_x=200,
        vmin=-phase_range,
        vmax=phase_range,
        cmap=my_cmap,
        title="Orthogonal phase",
    )

    strain[bulk == 0] = 0  # for easier visualization
    if save_fig:
        plt.savefig(
            datadir
            + "S"
            + str(scan)
            + "_phase_"
            + str(f"{photon_number:.0e}")
            + comment
            + ".png"
        )
    if save_data:
        np.savez_compressed(
            datadir + "S" + str(scan) + "_amp-phase-strain_SIMU" + comment,
            amp=support,
            phase=phase,
            bulk=bulk,
            strain=strain,
        )

        # save amp & phase to VTK
        # in VTK, x is downstream, y vertical, z inboard,
        # thus need to flip the last axis
        gu.save_to_vti(
            filename=os.path.join(
                datadir, "S" + str(scan) + "_amp-phase-strain_SIMU" + comment + ".vti"
            ),
            voxel_size=(voxel_size, voxel_size, voxel_size),
            tuple_array=(support, bulk, phase, strain),
            tuple_fieldnames=("amp", "bulk", "phase", "strain"),
            amplitude_threshold=0.01,
        )

    gu.multislices_plot(
        strain,
        sum_frames=False,
        plot_colorbar=True,
        width_z=200,
        width_y=200,
        width_x=200,
        vmin=-strain_range,
        vmax=strain_range,
        cmap=my_cmap,
        title="strain",
    )
    if save_fig:
        plt.savefig(
            datadir
            + "S"
            + str(scan)
            + "_strain_"
            + str(f"{photon_number:.0e}")
            + comment
            + ".png"
        )

del strain, bulk, surface, coordination_matrix
gc.collect()

##############################################################################
# rotate the object to have q in the same direction as during the experiment #
##############################################################################
if rotate_crystal:
    print("\nRotating the crystal to match experimental conditions")
    if ref_axis_outplane == "x":
        myaxis = np.array([1, 0, 0])  # must be in [x, y, z] order
    elif ref_axis_outplane == "y":
        myaxis = np.array([0, 1, 0])  # must be in [x, y, z] order
    elif ref_axis_outplane == "z":
        myaxis = np.array([0, 0, 1])  # must be in [x, y, z] order
    else:
        ref_axis_outplane = "y"
        myaxis = np.array([0, 1, 0])  # must be in [x, y, z] order
    print("Q aligned along ", ref_axis_outplane, ":", myaxis)
    angle = simu.angle_vectors(
        ref_vector=np.array([q[2], q[1], q[0]]) / np.linalg.norm(q), test_vector=myaxis
    )
    print("Angle between q and", ref_axis_outplane, "=", angle, "deg")
    print("Angle with y in zy plane", np.arctan(q[0] / q[1]) * 180 / np.pi, "deg")
    print("Angle with y in xy plane", np.arctan(-q[2] / q[1]) * 180 / np.pi, "deg")
    print("Angle with z in xz plane", 180 + np.arctan(q[2] / q[0]) * 180 / np.pi, "deg")

    support, phase = util.rotate_crystal(
        arrays=(support, phase),
        axis_to_align=myaxis,
        debugging=(True, False),
        title=("support", "phase"),
        reference_axis=np.array([q[2], q[1], q[0]]) / np.linalg.norm(q),
    )

original_obj = support * np.exp(1j * phase)
del phase, support
gc.collect()

###################################################
# compensate padding in order to keep reciprocal  #
# space resolution (detector pixel size) constant #
###################################################
# compensate padding in real space
print("\nOriginal voxel size", voxel_size, "nm")
dqz = 2 * np.pi / (nz * voxel_size * 10)  # in inverse angstroms
dqy = 2 * np.pi / (ny * voxel_size * 10)  # in inverse angstroms
dqx = 2 * np.pi / (nx * voxel_size * 10)  # in inverse angstroms
print(
    "Original reciprocal space resolution (z, y, x): (",
    str(f"{dqz:.5f}"),
    "A-1,",
    str(f"{dqy:.5f}"),
    "A-1,",
    str(f"{dqx:.5f}"),
    "A-1 )",
)
print(
    "Original q range (z, y, x): (",
    str(f"{dqz * nz:.5f}"),
    "A-1,",
    str(f"{dqy * ny:.5f}"),
    "A-1,",
    str(f"{dqx * nx:.5f}"),
    "A-1 )\n",
)
dqz_pad = 2 * np.pi / (pad_size[0] * voxel_size * 10)  # in inverse angstroms
dqy_pad = 2 * np.pi / (pad_size[1] * voxel_size * 10)  # in inverse angstroms
dqx_pad = 2 * np.pi / (pad_size[2] * voxel_size * 10)  # in inverse angstroms
print(
    "Reciprocal space resolution (z, y, x) after padding: (",
    str(f"{dqz_pad:.5f}"),
    "A-1,",
    str(f"{dqy_pad:.5f}"),
    "A-1,",
    str(f"{dqx_pad:.5f}"),
    "A-1 )",
)
print(
    "q range after padding (z, y, x): (",
    str(f"{dqz_pad * pad_size[0]:.5f}"),
    "A-1,",
    str(f"{dqy_pad * pad_size[1]:.5f}"),
    "A-1,",
    str(f"{dqx_pad * pad_size[2]:.5f}"),
    "A-1 )\n",
)
voxelsize_z = 2 * np.pi / (pad_size[0] * dqz_pad * 10)  # in nm
voxelsize_y = 2 * np.pi / (pad_size[1] * dqy_pad * 10)  # in nm
voxelsize_x = 2 * np.pi / (pad_size[2] * dqx_pad * 10)  # in nm
print(
    "Real-space voxel sizes (z, y, x) after padding: (",
    str(f"{voxelsize_z:.2f}"),
    "nm,",
    str(f"{voxelsize_y:.2f}"),
    "nm,",
    str(f"{voxelsize_x:.2f}"),
    "nm )",
)
print("Padding has no effect on real-space voxel size.\n")
print(
    "Interpolating the object to keep the q resolution constant "
    "(i.e. the detector pixel size constant)."
)
print("Multiplication factor for the real-space voxel size:  pad_size/original_size")

# compensate change in detector distance
comment = comment + "_sdd_" + str(f"{simulated_sdd:.2f}")
print(
    "\nCurrent detector pixel size", setup.detector.unbinned_pixel_size[0] * 1e6, "um"
)
print(
    "Detector pixel size to compensate the change in detector distance",
    str(
        "{:.2f}".format(
            setup.detector.unbinned_pixel_size[0] * 1e6 * original_sdd / simulated_sdd
        )
    ),
    "um",
)
print(
    "Reciprocal space resolution before detector distance change (z, y, x): (",
    str(f"{dqz:.5f}"),
    "A-1,",
    str(f"{dqy:.5f}"),
    "A-1,",
    str(f"{dqx:.5f}"),
    "A-1 )",
)
print(
    "q range before detector distance change (z, y, x): (",
    str(f"{dqz * nz:.5f}"),
    "A-1,",
    str(f"{dqy * ny:.5f}"),
    "A-1,",
    str(f"{dqx * nx:.5f}"),
    "A-1 )",
)
voxelsize_z = 2 * np.pi / (nz * dqz * 10)  # in nm
voxelsize_y = 2 * np.pi / (ny * dqy * 10)  # in nm
voxelsize_x = 2 * np.pi / (nx * dqx * 10)  # in nm
print(
    "Real-space voxel sizes before detector distance change (z, y, x): (",
    str(f"{voxelsize_z:.2f}"),
    "nm,",
    str(f"{voxelsize_y:.2f}"),
    "nm,",
    str(f"{voxelsize_x:.2f}"),
    "nm )\n",
)

dqz_simu, dqy_simu, dqx_simu = (
    dqz * original_sdd / simulated_sdd,
    dqy * original_sdd / simulated_sdd,
    dqx * original_sdd / simulated_sdd,
)
print(
    "Reciprocal space resolution after detector distance change (z, y, x): (",
    str(f"{dqz_simu:.5f}"),
    "A-1,",
    str(f"{dqy_simu:.5f}"),
    "A-1,",
    str(f"{dqx_simu:.5f}"),
    "A-1 )",
)
print(
    "q range after detector distance change (z, y, x): (",
    str(f"{dqz_simu * nz:.5f}"),
    "A-1,",
    str(f"{dqy_simu * ny:.5f}"),
    "A-1,",
    str(f"{dqx_simu * nx:.5f}"),
    "A-1 )",
)
voxelsize_z = 2 * np.pi / (nz * dqz_simu * 10)  # in nm
voxelsize_y = 2 * np.pi / (ny * dqy_simu * 10)  # in nm
voxelsize_x = 2 * np.pi / (nx * dqx_simu * 10)  # in nm
print(
    "Real-space voxel sizes after detector distance change (z, y, x): (",
    str(f"{voxelsize_z:.2f}"),
    "nm,",
    str(f"{voxelsize_y:.2f}"),
    "nm,",
    str(f"{voxelsize_x:.2f}"),
    "nm )\n",
)

# interpolate the support
if pad_ortho:  # pad before interpolating into detector frame
    # this is the only choice if the compensated object is larger
    # than the initial array shape
    print(
        "Padding to data size: ",
        pad_size,
        " before interpolating into the detector frame",
    )
    nz_interp = pad_size[0]
    ny_interp = pad_size[1]
    nx_interp = pad_size[2]
    if pad_size[0] < nz or pad_size[1] < ny or pad_size[2] < nx:
        print("Pad size smaller than initial array size")
        sys.exit()
    original_obj = util.crop_pad(original_obj, pad_size)
else:  # pad after interpolating into detector frame - saves memory
    nz_interp = nz
    ny_interp = ny
    nx_interp = nx

newz, newy, newx = np.meshgrid(
    np.arange(-nz_interp // 2, nz_interp // 2, 1) * voxel_size,
    np.arange(-ny_interp // 2, ny_interp // 2, 1) * voxel_size,
    np.arange(-nx_interp // 2, nx_interp // 2, 1) * voxel_size,
    indexing="ij",
)

if sdd_change_mode == "real_space":
    print(
        "Interpolating the real-space object to accomodate "
        "the change in detector distance."
    )
    print(
        "Multiplication factor for the real-space voxel size:  "
        "original_sdd / simulated_sdd\n"
    )
    # if the detector is 2 times farther away, the pixel size is two times smaller
    # (2 times better sampling) hence the q range is two times smaller and the
    # real-space voxel size two times larger

    rgi = RegularGridInterpolator(
        (
            np.arange(-nz_interp // 2, nz_interp // 2)
            * voxel_size
            * pad_size[0]
            / nz_interp
            * original_sdd
            / simulated_sdd,
            np.arange(-ny_interp // 2, ny_interp // 2)
            * voxel_size
            * pad_size[1]
            / ny_interp
            * original_sdd
            / simulated_sdd,
            np.arange(-nx_interp // 2, nx_interp // 2)
            * voxel_size
            * pad_size[2]
            / nx_interp
            * original_sdd
            / simulated_sdd,
        ),
        original_obj,
        method="linear",
        bounds_error=False,
        fill_value=0,
    )

else:  # 'reciprocal_space'
    rgi = RegularGridInterpolator(
        (
            np.arange(-nz_interp // 2, nz_interp // 2)
            * voxel_size
            * pad_size[0]
            / nz_interp,
            np.arange(-ny_interp // 2, ny_interp // 2)
            * voxel_size
            * pad_size[1]
            / ny_interp,
            np.arange(-nx_interp // 2, nx_interp // 2)
            * voxel_size
            * pad_size[2]
            / nx_interp,
        ),
        original_obj,
        method="linear",
        bounds_error=False,
        fill_value=0,
    )

obj = rgi(
    np.concatenate(
        (
            newz.reshape((1, newz.size)),
            newy.reshape((1, newz.size)),
            newx.reshape((1, newz.size)),
        )
    ).transpose()
)
del newx, newy, newz, rgi
gc.collect()

obj = obj.reshape((nz_interp, ny_interp, nx_interp)).astype(original_obj.dtype)

if debug:
    gu.multislices_plot(
        abs(obj),
        sum_frames=True,
        cmap=my_cmap,
        title="Orthogonal support interpolated for \npadding & detector distance "
        "change compensation\n",
    )
    if orthogonal_frame:
        data = fftshift(abs(fftn(original_obj)) ** 2)
        data = data / data.sum() * photon_number  # convert into photon number
        gu.multislices_plot(
            data,
            sum_frames=False,
            scale="log",
            plot_colorbar=True,
            vmin=-5,
            cmap=my_cmap,
            reciprocal_space=True,
            is_orthogonal=False,
            title="FFT before padding & detector distance change\n",
        )
        del original_obj, data
        gc.collect()

else:
    del original_obj
    gc.collect()

###################################################
# interpolate the object back into detector frame #
###################################################
if not orthogonal_frame:
    if debug:
        original_obj = setup.detector_frame(
            obj=original_obj,
            voxel_size=voxel_size,
            debugging=debug,
            title="Original object",
        )
        data = fftshift(abs(fftn(original_obj)) ** 2)
        gu.multislices_plot(
            data,
            sum_frames=False,
            scale="log",
            plot_colorbar=True,
            vmin=-5,
            cmap=my_cmap,
            reciprocal_space=True,
            is_orthogonal=False,
            title="FFT before padding & detector distance change\n",
        )
        del original_obj, data
        gc.collect()

    obj = setup.detector_frame(
        obj=obj, voxel_size=voxel_size, debugging=debug, title="Rescaled object"
    )

    #################################################################
    # uncomment this if you want to save the non-orthogonal support #
    # in that case pad_size and crop_size should be identical       #
    #################################################################
    # support = abs(obj)
    # support = support / support.max()
    # support[support < 0.05] = 0
    # support[np.nonzero(support)] = 1
    # np.savez_compressed(datadir + 'S' + str(scan) +
    # 'support_nonortho400.npz', obj=support)

##############################################################
# pad the array (after interpolation because of memory cost) #
##############################################################
if not pad_ortho:
    print(
        "Padding to data size: ",
        pad_size,
        " after interpolating into the detector frame",
    )
    if pad_size[0] < nz or pad_size[1] < ny or pad_size[2] < nx:
        print("Pad size smaller than initial array size")
        sys.exit()
    newobj = util.crop_pad(obj, pad_size)
else:
    newobj = obj

nz, ny, nx = newobj.shape
comment = comment + "_pad_" + str(nz) + "," + str(ny) + "," + str(nx)
del obj
gc.collect()

gu.multislices_plot(
    abs(newobj), sum_frames=True, cmap=my_cmap, title="Support before FFT calculation"
)
if save_fig:
    plt.savefig(
        datadir + "S" + str(scan) + "_support_before_FFT" + comment + "_sum.png"
    )

###########################################
# normalize and apply amplitude threshold #
###########################################
newobj = newobj / abs(newobj).max()
newobj[abs(newobj) < support_threshold] = 0

#####################################
# calculate the diffraction pattern #
#####################################
data = fftshift(abs(fftn(newobj)) ** 2)
gu.multislices_plot(
    data,
    sum_frames=False,
    scale="log",
    plot_colorbar=True,
    vmin=-5,
    cmap=my_cmap,
    reciprocal_space=True,
    is_orthogonal=False,
    title="FFT on the padded object\n",
)
del newobj
gc.collect()

#################################################################################
# interpolate the diffraction pattern to accomodate change in detector distance #
#################################################################################
if (sdd_change_mode == "reciprocal_space") and (original_sdd != simulated_sdd):
    print(
        "Interpolating the diffraction pattern to accomodate "
        "the change in detector distance."
    )
    print(
        "Multiplication factor for the detector pixel size:  "
        "simulated_sdd/original_sdd\n"
    )
    # if the detector is 2 times farther away,
    # the pixel size is two times smaller (2 times better sampling)
    # and the q range is two times smaller

    newz, newy, newx = np.meshgrid(
        np.arange(-nz // 2, nz // 2, 1) * dqz,
        np.arange(-ny // 2, ny // 2, 1) * dqy,
        np.arange(-nx // 2, nx // 2, 1) * dqx,
        indexing="ij",
    )

    rgi = RegularGridInterpolator(
        (
            np.arange(-nz // 2, nz // 2) * dqz * simulated_sdd / original_sdd,
            np.arange(-ny // 2, ny // 2) * dqy * simulated_sdd / original_sdd,
            np.arange(-nx // 2, nx // 2) * dqx * simulated_sdd / original_sdd,
        ),
        data,
        method="linear",
        bounds_error=False,
        fill_value=0,
    )

    simu_data = rgi(
        np.concatenate(
            (
                newz.reshape((1, newz.size)),
                newy.reshape((1, newz.size)),
                newx.reshape((1, newz.size)),
            )
        ).transpose()
    )
    del newx, newy, newz, rgi
    gc.collect()

    simu_data = simu_data.reshape((nz, ny, nx)).astype(data.dtype)

    gu.multislices_plot(
        simu_data,
        sum_frames=False,
        scale="log",
        plot_colorbar=True,
        vmin=-5,
        cmap=my_cmap,
        reciprocal_space=True,
        is_orthogonal=False,
        title="FFT for simulated detector distance\n",
    )
else:
    simu_data = data

del data
gc.collect()

#######################################################
# convert into photons and apply the photon threshold #
#######################################################
simu_data = simu_data / simu_data.sum() * photon_number  # convert into photon number

mask = np.zeros((nz, ny, nx))
mask[simu_data <= photon_threshold] = 1
simu_data[simu_data <= photon_threshold] = 0
temp_data = np.rint(simu_data).astype(int)
filled_pixels = (temp_data != 0).sum()
print("Number of pixels filled with non-zero intensity= ", filled_pixels)
del temp_data
gc.collect()

gu.multislices_plot(
    simu_data,
    sum_frames=False,
    scale="log",
    plot_colorbar=True,
    vmin=-5,
    cmap=my_cmap,
    reciprocal_space=True,
    is_orthogonal=False,
    title="FFT converted into photons\n",
)
if save_fig:
    plt.savefig(
        datadir
        + "S"
        + str(scan)
        + "_diff_float_"
        + str(f"{photon_number:.0e}")
        + comment
        + "_sum.png"
    )

#########################
# include Poisson noise #
#########################
if include_noise:
    simu_data = np.rint(poisson(simu_data)).astype(int)
    comment = comment + "_noise"
else:
    simu_data = np.rint(simu_data).astype(int)

#####################
# add detector gaps #
#####################
if set_gap:
    comment = comment + "_gap" + str(gap_pixel_start)
    simu_data, mask = simu.gap_detector(
        data=simu_data, mask=mask, start_pixel=gap_pixel_start, width_gap=gap_width
    )
else:
    comment = comment + "_nogap"

gu.multislices_plot(
    simu_data,
    sum_frames=False,
    scale="log",
    plot_colorbar=True,
    vmin=-1,
    cmap=my_cmap,
    reciprocal_space=True,
    is_orthogonal=False,
    title="FFT after rounding",
)

myfig, _, _ = gu.multislices_plot(
    simu_data,
    sum_frames=True,
    scale="log",
    plot_colorbar=True,
    vmin=-1,
    cmap=my_cmap,
    reciprocal_space=True,
    is_orthogonal=False,
    title="Masked intensity",
)
myfig.text(0.60, 0.30, "Pad size =" + str(pad_size), size=20)
if save_fig:
    myfig.savefig(
        datadir
        + "S"
        + str(scan)
        + "_diff_"
        + str(f"{photon_number:.0e}")
        + comment
        + "_sum.png"
    )

#################################################
# crop arrays to obtain the final detector size #
#################################################
voxelsizez_crop = 2 * np.pi / (crop_size[0] * dqz_simu * 10)  # in nm
voxelsizey_crop = 2 * np.pi / (crop_size[1] * dqy_simu * 10)  # in nm
voxelsizex_crop = 2 * np.pi / (crop_size[2] * dqx_simu * 10)  # in nm
print(
    "Real-space voxel sizes (z, y, x) after cropping: (",
    str(f"{voxelsizez_crop:.2f}"),
    "nm,",
    str(f"{voxelsizey_crop:.2f}"),
    "nm,",
    str(f"{voxelsizex_crop:.2f}"),
    "nm )",
)

nz, ny, nx = simu_data.shape
nz_crop, ny_crop, nx_crop = crop_size
if nz < nz_crop or ny < ny_crop or nx < nx_crop:
    print("Crop size larger than initial array size")
    sys.exit()

simu_data = util.crop_pad(simu_data, crop_size)
mask = util.crop_pad(mask, crop_size)

##########################################################
# crop arrays to fulfill FFT requirements during phasing #
##########################################################
nz, ny, nx = simu_data.shape
nz_crop, ny_crop, nx_crop = util.smaller_primes(
    (nz, ny, nx), maxprime=7, required_dividers=(2,)
)

simu_data = util.crop_pad(simu_data, (nz_crop, ny_crop, nx_crop))
mask = util.crop_pad(mask, (nz_crop, ny_crop, nx_crop))

nz, ny, nx = simu_data.shape
print("cropped FFT data size:", simu_data.shape)
print("Total number of photons:", simu_data.sum())
comment = comment + "_crop_" + str(nz) + "," + str(ny) + "," + str(nx)

##############
# save files #
##############
if save_data:
    np.savez_compressed(
        datadir + "S" + str(scan) + "_diff_" + str(f"{photon_number:.0e}") + comment,
        data=simu_data,
    )
    np.savez_compressed(
        datadir + "S" + str(scan) + "_mask_" + str(f"{photon_number:.0e}") + comment,
        mask=mask,
    )

#####################################
# plot mask and diffraction pattern #
#####################################
plt.ioff()
if debug:
    gu.multislices_plot(
        mask,
        sum_frames=True,
        scale="linear",
        plot_colorbar=False,
        cmap=my_cmap,
        reciprocal_space=True,
        is_orthogonal=False,
        title="Mask",
    )

myfig, _, _ = gu.multislices_plot(
    simu_data,
    sum_frames=False,
    scale="log",
    plot_colorbar=True,
    vmin=-1,
    cmap=my_cmap,
    reciprocal_space=True,
    is_orthogonal=False,
    title="Masked intensity",
)
myfig.text(0.60, 0.35, "Pad size =" + str(pad_size), size=20)
myfig.text(0.60, 0.30, "Crop size =" + str(crop_size), size=20)
myfig.text(0.60, 0.25, "Filled pixels =" + str(filled_pixels), size=20)
myfig.text(
    0.60,
    0.20,
    "Detector distance =" + str(f"{simulated_sdd:.5f}") + " m",
    size=20,
)
myfig.text(
    0.60,
    0.15,
    "Voxel size ="
    + str(f"{voxelsizez_crop:.2f}")
    + ", "
    + str(f"{voxelsizey_crop:.2f}")
    + ", "
    + str(f"{voxelsizex_crop:.2f}")
    + " nm",
    size=20,
)
myfig.text(0.60, 0.10, "Volume =" + str(volume) + " nm3", size=20)
if set_gap:
    myfig.text(0.60, 0.05, "Gap width =" + str(gap_width) + " pixels", size=20)
if save_fig:
    myfig.savefig(
        datadir
        + "S"
        + str(scan)
        + "_diff_"
        + str(f"{photon_number:.0e}")
        + comment
        + "_center.png"
    )

myfig, _, _ = gu.multislices_plot(
    simu_data,
    sum_frames=True,
    scale="log",
    plot_colorbar=True,
    vmin=-1,
    cmap=my_cmap,
    reciprocal_space=True,
    is_orthogonal=False,
    title="Masked intensity",
)
myfig.text(0.60, 0.35, "Pad size =" + str(pad_size), size=20)
myfig.text(0.60, 0.30, "Crop size =" + str(crop_size), size=20)
myfig.text(0.60, 0.25, "Filled pixels =" + str(filled_pixels), size=20)
myfig.text(
    0.60,
    0.20,
    "Detector distance =" + str(f"{simulated_sdd:.5f}") + " m",
    size=20,
)
myfig.text(
    0.60,
    0.15,
    "Voxel size ="
    + str(f"{voxelsizez_crop:.2f}")
    + ", "
    + str(f"{voxelsizey_crop:.2f}")
    + ", "
    + str(f"{voxelsizex_crop:.2f}")
    + " nm",
    size=20,
)
myfig.text(0.60, 0.10, "Volume =" + str(volume) + " nm3", size=20)
if set_gap:
    myfig.text(0.60, 0.05, "Gap width =" + str(gap_width) + " pixels", size=20)
if save_fig:
    myfig.savefig(
        datadir
        + "S"
        + str(scan)
        + "_diff_"
        + str(f"{photon_number:.0e}")
        + comment
        + "_sum.png"
    )
plt.show()
