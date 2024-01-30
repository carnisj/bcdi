#!/usr/bin/env python3

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import gc
import sys
import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal  # for medfilt
import xrayutilities as xu
from numpy.fft import fftn, fftshift
from scipy.interpolate import RegularGridInterpolator

import bcdi.graph.graph_utils as gu
import bcdi.postprocessing.facet_recognition as fu
import bcdi.preprocessing.bcdi_utils as bu
import bcdi.simulation.simulation_utils as simu
import bcdi.utils.utilities as util
from bcdi.experiment.setup import Setup
from bcdi.graph.colormap import ColormapFactory

helptext = """
Stereographic projection of a measured 3D diffraction pattern or calculated from a
real-space BCDI reconstruction. A shell dq of reciprocal space located a radius_mean
(in q) from the Bragg peak is projected from the South pole and the North onto the
equatorial plane.

The coordinate system follows the CXI convention: Z downstream, Y vertical up and X
outboard. Q values follow the more classical convention: qx downstream, qz vertical
up, qy outboard.
"""
######################
# generic parameters #
######################
scan = 78  # spec scan number
root_folder = "D:/data/Pt THH ex-situ/Data/HS4670/"
sample_name = "S"  # "S"  #
comment = ""
reflection = np.array([0, 2, 0])  # np.array([0, 0, 2])  #   # reflection measured
projection_axis = 1  # the projection will be performed on the equatorial plane
# perpendicular to that axis (0, 1 or 2)
radius_mean = 0.030  # q from Bragg peak
dq = 0.001  # width in q of the shell to be projected
sample_offsets = None  # tuple of offsets in degrees of the sample
# for each sample circle (outer first).
# the sample offsets will be subtracted to the motor values. Leave None if no offset.
q_offset = [
    0,
    0,
    0,
]  # offset of the projection plane in [qx, qy, qz] (0 = equatorial plane)
# q_offset applies only to measured diffraction pattern
# (not obtained from a reconstruction)
photon_threshold = 0  # threshold applied to the measured diffraction pattern
contour_range = None  # range(250, 2600, 250)
# range for the plot contours range(min, max, step), leave it to None for default
max_angle = 100  # maximum angle in degrees of the stereographic projection
# (should be larger than 90)
medianfilter_kernel = 3  # size in each dimension of the 3D kernel for median filtering,
# leave None otherwise
plot_planes = True  # if True, plot dotted circles corresponding to
# planes_south and planes_north indices
hide_axis = (
    False  # if True, the default axis frame, ticks and ticks labels will be hidden
)
planes_south = {}  # create dictionnary for the projection from the South pole,
# the reference is +reflection
planes_south["0 2 0"] = simu.angle_vectors(
    ref_vector=reflection, test_vector=np.array([0, 2, 0])
)
planes_south["1 1 1"] = simu.angle_vectors(
    ref_vector=reflection, test_vector=np.array([1, 1, 1])
)
# planes_south['1 0 0'] =
# simu.angle_vectors(ref_vector=reflection, test_vector=np.array([1, 0, 0]))
# planes_south['1 0 0'] =
# simu.angle_vectors(ref_vector=reflection, test_vector=np.array([1, 0, 0]))
# planes_south['1 1 0'] =
# simu.angle_vectors(ref_vector=reflection, test_vector=np.array([1, 1, 0]))
# planes_south['-1 1 0'] =
# simu.angle_vectors(ref_vector=reflection, test_vector=np.array([-1, 1, 0]))
# planes_south['1 -1 1'] =
# simu.angle_vectors(ref_vector=reflection, test_vector=np.array([1, -1, 1]))
# planes_south['-1 -1 1'] =
# simu.angle_vectors(ref_vector=reflection, test_vector=np.array([-1, -1, 1]))
planes_south["1 2 0"] = simu.angle_vectors(
    ref_vector=reflection, test_vector=np.array([1, 2, 0])
)
planes_south["2 1 0"] = simu.angle_vectors(
    ref_vector=reflection, test_vector=np.array([2, 1, 0])
)
planes_south["2 0 1"] = simu.angle_vectors(
    ref_vector=reflection, test_vector=np.array([2, 0, 1])
)

planes_north = {}  # create dictionnary for the projection from the North pole,
# the reference is -reflection
planes_north["0 -2 0"] = simu.angle_vectors(
    ref_vector=-reflection, test_vector=np.array([0, -2, 0])
)
planes_north["-1 -1 -1"] = simu.angle_vectors(
    ref_vector=-reflection, test_vector=np.array([-1, -1, -1])
)
# planes_north['-1 0 0'] =
# simu.angle_vectors(ref_vector=-reflection, test_vector=np.array([-1, 0, 0]))
# planes_north['-1 -1 0'] =
# simu.angle_vectors(ref_vector=-reflection, test_vector=np.array([-1, -1, 0]))
# planes_north['-1 1 0'] =
# simu.angle_vectors(ref_vector=-reflection, test_vector=np.array([-1, 1, 0]))
# planes_north['-1 -1 1'] =
# simu.angle_vectors(ref_vector=-reflection, test_vector=np.array([-1, -1, 1]))
# planes_north['-1 1 1'] =
# simu.angle_vectors(ref_vector=-reflection, test_vector=np.array([-1, 1, 1]))
planes_north["1 -2 0"] = simu.angle_vectors(
    ref_vector=-reflection, test_vector=np.array([1, -2, 0])
)
planes_north["2 -1 0"] = simu.angle_vectors(
    ref_vector=-reflection, test_vector=np.array([2, -1, 0])
)
planes_north["2 0 1"] = simu.angle_vectors(
    ref_vector=-reflection, test_vector=np.array([2, 0, 1])
)
debug = False  # True to show more plots, False otherwise
########################################################
# parameters for plotting the stereographic projection #
# starting from the phased real space object only      #
########################################################
reconstructed_data = (
    False  # set it to True if the data is a BCDI reconstruction (real space)
)
# the reconstruction should be in the crystal orthogonal frame
threshold_amp = (
    0.3  # threshold for support determination from amplitude, if reconstructed_data=1
)
use_phase = (
    False  # set to False to use only a support, True to use the complex amplitude
)
binary_support = (
    False  # if True, the modulus of the reconstruction will be set to a binary support
)
phase_factor = -1  # 1, -1, -2*np.pi/d depending on what is in the field phase
# (phase, -phase, displacement...)
voxel_size = [
    3.0,
    3.0,
    3.0,
]  # in nm, voxel size of the CDI reconstruction in each directions.  Put [] if unknown
pad_size = [
    2,
    2,
    2,
]  # list of three int >= 1, will pad to get this number times the initial array size
# voxel size does not change, hence it corresponds to upsampling the diffraction pattern
upsampling_ratio = 2  # int >=1, upsample the real space object by this factor
# (voxel size divided by upsampling_ratio)
# it corresponds to increasing the size of the detector while keeping
# detector pixel size constant
#################################################################################
# define beamline related parameters, not used for the phased real space object #
#################################################################################
beamline = (
    "ID01"  # name of the beamline, used for data loading and normalization by monitor
)
# supported beamlines: 'ID01', 'SIXS_2018', 'SIXS_2019', 'CRISTAL', 'P10'
custom_scan = False  # True for a stack of images acquired without scan,
# e.g. with ct in a macro (no info in spec file)
custom_images = (
    None  # np.arange(11665, 11764, 1)  # list of image numbers for the custom_scan
)
custom_monitor = None  # np.ones(len(custom_images))
# monitor values for normalization for the custom_scan
custom_motors = None
# {"eta": np.linspace(16.989, 18.969596, num=100, endpoint=False),
# "phi": 0, "nu": -0.75, "delta": 35.978}
# ID01: eta, phi, nu, delta
# CRISTAL: mgomega, gamma, delta
# P10: om, phi, chi, mu, gamma, delta
# SIXS: beta, mu, gamma, delta
rocking_angle = "outofplane"  # "outofplane" or "inplane" or "energy"
specfile_name = "psic_nano_20141204"
# .spec for ID01, .fio for P10, alias_dict.txt for SIXS_2018,
# not used for CRISTAL and SIXS_2019
# template for ID01: name of the spec file without '.spec'
# template for SIXS_2018: full path of the alias dictionnary,
# typically root_folder + 'alias_dict_2019.txt'
# template for SIXS_2019: ''
# template for P10: sample_name + '_%05d'
# template for CRISTAL: ''
filtered_data = True  # set to True if the data is already a 3D array, False otherwise
is_orthogonal = False  # True is the filtered_data is already orthogonalized,
# q values need to be provided
normalize_flux = "skip"  # 'monitor' to normalize the intensity by the default
# monitor values, 'skip' to do nothing
#######################################################
# define detector related parameters and region of    #
# interest, not used for the phased real space object #
#######################################################
detector = "Maxipix"  # "Eiger2M" or "Maxipix" or "Eiger4M"
# x_bragg = 451  # horizontal pixel number of the Bragg peak
# y_bragg = 1450  # vertical pixel number of the Bragg peak
# roi_detector = [1202, 1610, x_bragg - 256, x_bragg + 256]  # HC3207  x_bragg = 430
roi_detector = []  # [y_bragg - 290, y_bragg + 350, x_bragg - 350, x_bragg + 350]  # Ar
# roi_detector = [552, 1064, x_bragg - 240, x_bragg + 240]  # P10 2018
# leave it as [] to use the full detector.
# Use with center_fft='do_nothing' if you want this exact size.
hotpixels_file = ""  # root_folder + 'hotpixels.npz'  #
flatfield_file = root_folder + "flatfield_maxipix_8kev.npz"  #
template_imagefile = "Pt4_%04d.edf"  # .gz'
# template for ID01: 'data_mpx4_%05d.edf.gz' or 'align_eiger2M_%05d.edf.gz'
# template for SIXS_2018: 'align.spec_ascan_mu_%05d.nxs'
# template for SIXS_2019: 'spare_ascan_mu_%05d.nxs'
# template for Cristal: 'S%d.nxs'
# template for P10: '_data_%06d.h5'
binning = [
    1,
    1,
    1,
]  # binning to apply to the measured diffraction pattern in each dimension
###################################################################
# define parameters for xrayutilities, used for orthogonalization #
# not used for the phased real space object                       #
###################################################################
# xrayutilities uses the xyz crystal frame: for incident angle = 0,
# x is downstream, y outboard, and z vertical up
sdd = (
    1.26  # 0.865  # sample to detector distance in m, not important if you use raw data
)
energy = 9000  # x-ray energy in eV, not important if you use raw data
beam_direction = (1, 0, 0)  # beam along z
sample_inplane = (
    1,
    0,
    0,
)  # sample inplane reference direction along the beam at 0 angles
sample_outofplane = (0, 0, 1)  # surface normal of the sample at 0 angles
offset_inplane = 0  # outer detector angle offset, not important if you use raw data
cch1 = 369.5  # vertical
# cch1 parameter from xrayutilities 2D detector calibration,
# the detector roi is taken into account below
cch2 = 138.5  # horizontal
# cch2 parameter from xrayutilities 2D detector calibration,
# the detector roi is taken into account below
detrot = 0  # detrot parameter from xrayutilities 2D detector calibration
tiltazimuth = 0  # tiltazimuth parameter from xrayutilities 2D detector calibration
tilt = 0  # tilt parameter from xrayutilities 2D detector calibration
##################################
# end of user-defined parameters #
##################################

###################
# define colormap #
###################
bad_color = "1.0"  # white background
my_cmap = ColormapFactory(bad_color=bad_color).cmap

####################
# Initialize setup #
####################
setup = Setup(
    beamline_name=beamline,
    energy=energy,
    rocking_angle=rocking_angle,
    distance=sdd,
    beam_direction=beam_direction,
    sample_inplane=sample_inplane,
    sample_outofplane=sample_outofplane,
    sample_offsets=sample_offsets,
    offset_inplane=offset_inplane,
    custom_scan=custom_scan,
    custom_images=custom_images,
    custom_monitor=custom_monitor,
    custom_motors=custom_motors,
    filtered_data=filtered_data,
    is_orthogonal=is_orthogonal,
    detector_name=detector,
    datadir="",
    template_imagefile=template_imagefile,
    roi=roi_detector,
)

#############################################
# Initialize geometry for orthogonalization #
#############################################
qconv, offsets = setup.init_qconversion()
setup.detector.offsets = offsets
hxrd = xu.experiment.HXRD(sample_inplane, sample_outofplane, qconv=qconv)
# x downstream, y outboard, z vertical
# first two arguments in HXRD are the inplane reference direction
# along the beam and surface normal of the sample
cch1 = cch1 - setup.detector.roi[0]  # take into account the roi if the image is cropped
cch2 = cch2 - setup.detector.roi[2]  # take into account the roi if the image is cropped
hxrd.Ang2Q.init_area(
    "z-",
    "y+",
    cch1=cch1,
    cch2=cch2,
    Nch1=setup.detector.roi[1] - setup.detector.roi[0],
    Nch2=setup.detector.roi[3] - setup.detector.roi[2],
    pwidth1=setup.detector.pixelsize_y,
    pwidth2=setup.detector.pixelsize_x,
    distance=sdd,
    detrot=detrot,
    tiltazimuth=tiltazimuth,
    tilt=tilt,
)
# the first two arguments in init_area are the direction of the detector,
# checked for ID01 and SIXS

#############
# load data #
#############
plt.ion()
root = tk.Tk()
root.withdraw()
if setup.beamline != "P10":
    homedir = root_folder + sample_name + str(scan) + "/"
    setup.detector.datadir = homedir + "data/"
else:
    specfile_name = specfile_name % scan
    homedir = root_folder + specfile_name + "/"
    setup.detector.datadir = homedir + "e4m/"
    template_imagefile = specfile_name + template_imagefile
    setup.detector.template_imagefile = template_imagefile

setup.detector.savedir = homedir

if not reconstructed_data:  # load reciprocal space data
    flatfield = util.load_flatfield(flatfield_file)
    hotpix_array = util.load_hotpixels(hotpixels_file)
    setup.create_logfile(
        scan_number=scan, root_folder=root_folder, filename=specfile_name
    )

    data, mask, frames_logical, monitor = bu.load_bcdi_data(
        scan_number=scan,
        setup=setup,
        flatfield=flatfield,
        hotpixels=hotpix_array,
        normalize=normalize_flux,
        debugging=debug,
    )
    data, _, q_values, _ = bu.grid_bcdi_xrayutil(
        data=data,
        mask=mask,
        scan_number=scan,
        setup=setup,
        frames_logical=frames_logical,
        hxrd=hxrd,
        debugging=debug,
    )

    nz, ny, nx = data.shape  # CXI convention: z downstream, y vertical up, x outboard
    print("Diffraction data shape", data.shape)
    qx = q_values[0]  # axis=0, z downstream, qx in reciprocal space
    qz = q_values[1]  # axis=1, y vertical, qz in reciprocal space
    qy = q_values[2]  # axis=2, x outboard, qy in reciprocal space
    ############
    # bin data #
    ############
    qx = qx[: nz - (nz % binning[0]) : binning[0]]
    qz = qz[: ny - (ny % binning[1]) : binning[1]]
    qy = qy[: nx - (nx % binning[2]) : binning[2]]
    data = util.bin_data(data, (binning[0], binning[1], binning[2]), debugging=False)
    nz, ny, nx = data.shape
    print("Diffraction data shape after binning", data.shape)

    # apply photon threshold
    data[data < photon_threshold] = 0
else:  # load a reconstructed real space object
    comment = comment + "_CDI"
    file_path = filedialog.askopenfilename(
        initialdir=homedir, title="Select 3D data", filetypes=[("NPZ", "*.npz")]
    )
    amp = np.load(file_path)["amp"]
    amp = amp / abs(amp).max()  # normalize amp
    nz, ny, nx = amp.shape  # CXI convention: z downstream, y vertical up, x outboard
    print("CDI data shape", amp.shape)
    # nz1, ny1, nx1 = [value * pad_size for value in amp.shape]
    nz1, ny1, nx1 = np.multiply(np.asarray(amp.shape), np.asarray(pad_size))
    nz1, ny1, nx1 = (
        nz1 + nz1 % 2,
        ny1 + ny1 % 2,
        nx1 + nx1 % 2,
    )  # ensure even pixel numbers in each direction

    if use_phase:  # calculate the complex amplitude
        comment = comment + "_complex"
        try:
            phase = np.load(file_path)["phase"]
        except KeyError:
            try:
                phase = np.load(file_path)["displacement"]
            except KeyError:
                print('No field named "phase" or "disp" in the reconstruction file')
                sys.exit()
        phase = phase * phase_factor
        amp = amp * np.exp(1j * phase)  # amp is the complex amplitude
        del phase
        gc.collect()
    else:
        comment = comment + "_support"

    gu.multislices_plot(
        abs(amp),
        sum_frames=False,
        reciprocal_space=False,
        is_orthogonal=True,
        title="abs(amp)",
    )

    ####################################################
    # pad array to improve reciprocal space resolution #
    ####################################################
    amp = util.crop_pad(amp, (nz1, ny1, nx1))
    nz, ny, nx = amp.shape
    print("CDI data shape after padding", amp.shape)

    ####################################################
    # interpolate the array with isotropic voxel sizes #
    ####################################################
    if len(voxel_size) == 0:
        print("Using isotropic voxel size of 1 nm")
        voxel_size = [1, 1, 1]  # nm

    if not all(voxsize == voxel_size[0] for voxsize in voxel_size):
        newvoxelsize = min(voxel_size)  # nm
        # size of the original object
        rgi = RegularGridInterpolator(
            (
                np.arange(-nz // 2, nz // 2, 1) * voxel_size[0],
                np.arange(-ny // 2, ny // 2, 1) * voxel_size[1],
                np.arange(-nx // 2, nx // 2, 1) * voxel_size[2],
            ),
            amp,
            method="linear",
            bounds_error=False,
            fill_value=0,
        )
        # points were to interpolate the object
        new_z, new_y, new_x = np.meshgrid(
            np.arange(-nz // 2, nz // 2, 1) * newvoxelsize,
            np.arange(-ny // 2, ny // 2, 1) * newvoxelsize,
            np.arange(-nx // 2, nx // 2, 1) * newvoxelsize,
            indexing="ij",
        )

        obj = rgi(
            np.concatenate(
                (
                    new_z.reshape((1, new_z.size)),
                    new_y.reshape((1, new_z.size)),
                    new_x.reshape((1, new_z.size)),
                )
            ).transpose()
        )
        obj = obj.reshape((nz, ny, nx)).astype(amp.dtype)
        gu.multislices_plot(
            abs(obj),
            sum_frames=False,
            reciprocal_space=False,
            is_orthogonal=True,
            title="obj with isotropic voxel sizes",
        )
    else:
        obj = amp
        newvoxelsize = voxel_size[0]  # nm
    print(
        "Original voxel sizes (nm): "
        f"{voxel_size[0]:.2f}, {voxel_size[1]:.2f}, {voxel_size[2]:.2f}"
    )
    print(
        "Output voxel sizes (nm): "
        f"{newvoxelsize:.2f}, {newvoxelsize:.2f}, {newvoxelsize:.2f}"
    )

    ########################################################################
    # upsample array to increase the size of the detector (limit aliasing) #
    ########################################################################

    if upsampling_ratio != 1:
        # size of the original object
        rgi = RegularGridInterpolator(
            (
                np.arange(-nz // 2, nz // 2, 1) * newvoxelsize,
                np.arange(-ny // 2, ny // 2, 1) * newvoxelsize,
                np.arange(-nx // 2, nx // 2, 1) * newvoxelsize,
            ),
            obj,
            method="linear",
            bounds_error=False,
            fill_value=0,
        )
        # points were to interpolate the object
        new_z, new_y, new_x = np.meshgrid(
            np.arange(-nz // 2, nz // 2, 1) * newvoxelsize / upsampling_ratio,
            np.arange(-ny // 2, ny // 2, 1) * newvoxelsize / upsampling_ratio,
            np.arange(-nx // 2, nx // 2, 1) * newvoxelsize / upsampling_ratio,
            indexing="ij",
        )

        obj = rgi(
            np.concatenate(
                (
                    new_z.reshape((1, new_z.size)),
                    new_y.reshape((1, new_z.size)),
                    new_x.reshape((1, new_z.size)),
                )
            ).transpose()
        )
        obj = obj.reshape((nz, ny, nx)).astype(amp.dtype)
        print(f"voxel size after upsampling (nm) {newvoxelsize / upsampling_ratio}")

        if debug:
            gu.multislices_plot(
                abs(obj),
                sum_frames=False,
                reciprocal_space=False,
                is_orthogonal=True,
                title="upsampled object",
            )

    #########################################
    # normalize and apply modulus threshold #
    #########################################
    # It is important to apply the threshold just before FFT calculation,
    # otherwise the FFT is noisy because of
    # interpolation artefacts
    obj = obj / abs(obj).max()
    obj[abs(obj) < threshold_amp] = 0
    if not use_phase and binary_support:  # phase is 0, obj is real
        # create a binary support
        obj[np.nonzero(obj)] = 1
        comment = comment + "_binary"
    if debug:
        gu.multislices_plot(
            abs(obj),
            sum_frames=False,
            reciprocal_space=False,
            is_orthogonal=True,
            title="abs(object) after threshold",
        )

    #######################################
    # calculate the diffraction intensity #
    #######################################
    data = fftshift(abs(fftn(obj)) ** 2) / (nz * ny * nx)
    gu.multislices_plot(
        abs(data),
        scale="log",
        vmin=-5,
        sum_frames=False,
        reciprocal_space=True,
        is_orthogonal=True,
        title="FFT(obj)",
    )
    del obj
    gc.collect()

    #############################
    # create qx, qy, qz vectors #
    #############################
    dqx = 2 * np.pi / (newvoxelsize / upsampling_ratio * 10 * nz)  # qx downstream
    dqy = 2 * np.pi / (newvoxelsize / upsampling_ratio * 10 * nx)  # qy outboard
    dqz = 2 * np.pi / (newvoxelsize / upsampling_ratio * 10 * ny)  # qz vertical up
    print(f"dqx {dqx:.5f}, dqy {dqy:.5f}, dqz {dqz:.5f}")
    qx = np.arange(-nz // 2, nz // 2) * dqx
    qy = np.arange(-nx // 2, nx // 2) * dqy
    qz = np.arange(-ny // 2, ny // 2) * dqz

nz, ny, nx = data.shape
if medianfilter_kernel:  # apply some noise filtering
    print(
        f"Applying median filtering {medianfilter_kernel}x"
        f"{medianfilter_kernel}x{medianfilter_kernel}"
    )
    data = scipy.signal.medfilt(data, medianfilter_kernel)

###################################
# define the center of the sphere #
###################################
if not reconstructed_data:
    qzCOM = (
        1 / data.sum() * (qz * data.sum(axis=0).sum(axis=1)).sum() + q_offset[2]
    )  # COM in qz
    qyCOM = (
        1 / data.sum() * (qy * data.sum(axis=0).sum(axis=0)).sum() + q_offset[1]
    )  # COM in qy
    qxCOM = (
        1 / data.sum() * (qx * data.sum(axis=1).sum(axis=1)).sum() + q_offset[0]
    )  # COM in qx
else:
    qzCOM, qyCOM, qxCOM = (
        0,
        0,
        0,
    )  # data is centered because it is the FFT of the object
print(f"Center of mass [qx, qy, qz]: [{qxCOM:.5f}, {qyCOM:.5f}, {qzCOM:.5f}]")

###############################################################
# create a 3D array of distances in q from the center of mass #
###############################################################
distances = np.sqrt(
    (qx[:, np.newaxis, np.newaxis] - qxCOM) ** 2
    + (qy[np.newaxis, np.newaxis, :] - qyCOM) ** 2
    + (qz[np.newaxis, :, np.newaxis] - qzCOM) ** 2
)
if debug:
    gu.multislices_plot(
        distances,
        sum_frames=False,
        reciprocal_space=True,
        is_orthogonal=True,
        title="distances",
    )

#########################################
# define the mask at radius radius_mean #
#########################################
mask = np.logical_and(
    (distances < (radius_mean + dq)), (distances > (radius_mean - dq))
)
if debug:
    gu.multislices_plot(
        mask, sum_frames=False, reciprocal_space=True, is_orthogonal=True, title="mask"
    )

####################################
# plot 2D diffrated intensity maps #
####################################
fig, (ax0, ax1, ax2, ax3), (plt0, plt1, plt2) = gu.contour_slices(
    data,
    (qx, qz, qy),
    sum_frames=True,
    title="Regridded data",
    levels=150,
    plot_colorbar=True,
    scale="log",
    is_orthogonal=True,
    reciprocal_space=True,
)

circle = plt.Circle(
    (qyCOM, qzCOM), radius_mean + dq, color="0", fill=False, linestyle="dotted"
)
ax0.add_artist(circle)
circle = plt.Circle(
    (qyCOM, qzCOM), radius_mean - dq, color="0", fill=False, linestyle="dotted"
)
ax0.add_artist(circle)
circle = plt.Circle(
    (qyCOM, qxCOM), radius_mean + dq, color="0", fill=False, linestyle="dotted"
)
ax1.add_artist(circle)
circle = plt.Circle(
    (qyCOM, qxCOM), radius_mean - dq, color="0", fill=False, linestyle="dotted"
)
ax1.add_artist(circle)
circle = plt.Circle(
    (qzCOM, qxCOM), radius_mean + dq, color="0", fill=False, linestyle="dotted"
)
ax2.add_artist(circle)
circle = plt.Circle(
    (qzCOM, qxCOM), radius_mean - dq, color="0", fill=False, linestyle="dotted"
)
ax2.add_artist(circle)
fig.text(0.60, 0.30, "Scan " + str(scan), size=20)
plt.pause(0.1)
plt.savefig(
    homedir
    + "diffpattern"
    + comment
    + "_S"
    + str(scan)
    + "_q="
    + str(radius_mean)
    + ".png"
)

###########################################
# apply the mask to the data and q values #
###########################################
# qx downstream, qz vertical up, qy outboard
data_masked = data[mask]
if projection_axis == 0:
    qx, qz, qy = np.meshgrid(qx, qz - qzCOM, qy - qyCOM, indexing="ij")
    stereo_center = qxCOM
elif projection_axis == 1:
    qx, qz, qy = np.meshgrid(qx - qxCOM, qz, qy - qyCOM, indexing="ij")
    stereo_center = qzCOM
elif projection_axis == 2:
    qx, qz, qy = np.meshgrid(qx - qxCOM, qz - qzCOM, qy, indexing="ij")
    stereo_center = qyCOM
else:
    print('Invalid value for the parameter "projection_axis"')
    sys.exit()

qx, qz, qy = (
    qx[mask].reshape((data_masked.size, 1)),
    qz[mask].reshape((data_masked.size, 1)),
    qy[mask].reshape((data_masked.size, 1)),
)

##########################################
# calculate the stereographic projection #
##########################################
stereo_proj, uv_labels = fu.calc_stereoproj_facet(
    projection_axis=projection_axis,
    vectors=np.concatenate((qx, qz, qy), axis=1),
    radius_mean=radius_mean,
    stereo_center=stereo_center,
)

###########################################
# plot the projection from the South pole #
###########################################
fig, _ = gu.contour_stereographic(
    euclidian_u=stereo_proj[:, 0],
    euclidian_v=stereo_proj[:, 1],
    color=data_masked,
    radius_mean=radius_mean,
    planes=planes_south,
    title="Projection from the South pole",
    hide_axis=hide_axis,
    plot_planes=plot_planes,
    max_angle=max_angle,
    uv_labels=uv_labels,
    contour_range=contour_range,
)


if not reconstructed_data:
    fig.text(0.05, 0.02, "q=" + str(radius_mean) + " dq=" + str(dq), size=14)
else:
    fig.text(0.05, 0.92, "q=" + str(radius_mean) + " dq=" + str(dq), size=14)

fig.savefig(homedir + "South pole" + comment + "_S" + str(scan) + ".png")

############################################
# plot the projection from the  North pole #
############################################
fig, _ = gu.contour_stereographic(
    euclidian_u=stereo_proj[:, 2],
    euclidian_v=stereo_proj[:, 3],
    color=data_masked,
    radius_mean=radius_mean,
    planes=planes_north,
    title="Projection from the North pole",
    hide_axis=hide_axis,
    plot_planes=plot_planes,
    max_angle=max_angle,
    uv_labels=uv_labels,
    contour_range=contour_range,
)

if not reconstructed_data:
    fig.text(0.05, 0.02, "q=" + str(radius_mean) + " dq=" + str(dq), size=14)

else:
    fig.text(0.05, 0.92, "q=" + str(radius_mean) + " dq=" + str(dq), size=14)

plt.savefig(homedir + "North pole" + comment + "_S" + str(scan) + ".png")

################################
# save grid points in txt file #
################################
with open(homedir + "Poles" + comment + "_S" + str(scan) + ".dat", "w") as file:
    # save metric coordinates in text file
    nb_points = stereo_proj.shape[0]
    for ii in range(nb_points):
        file.write(
            str(stereo_proj[ii, 0])
            + "\t"
            + str(stereo_proj[ii, 1])
            + "\t"
            + str(stereo_proj[ii, 2])
            + "\t"
            + str(stereo_proj[ii, 3])
            + "\t"
            + str(data_masked[ii])
            + "\n"
        )

plt.ioff()
plt.show()
