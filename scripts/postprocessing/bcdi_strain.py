#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import argparse
from datetime import datetime
from functools import reduce
import gc

try:
    import hdf5plugin  # for P10, should be imported before h5py or PyTables
except ModuleNotFoundError:
    pass
import h5py
from matplotlib import pyplot as plt
import numpy as np
import os
import pprint
import tkinter as tk
from tkinter import filedialog

import bcdi.graph.graph_utils as gu
from bcdi.experiment.detector import create_detector
from bcdi.experiment.setup import Setup
import bcdi.preprocessing.bcdi_utils as bu
import bcdi.postprocessing.postprocessing_utils as pu
import bcdi.simulation.simulation_utils as simu
import bcdi.utils.image_registration as reg
from bcdi.utils.parser import add_cli_parameters, ConfigParser
import bcdi.utils.utilities as util

CONFIG_FILE = "C:/Users/Jerome/Documents/myscripts/bcdi/conf/config_postprocessing.yml"

helptext = """
Interpolate the output of the phase retrieval into an orthonormal frame,
and calculate the strain component along the direction of the experimental diffusion
vector q.

Input: complex amplitude array, output from a phase retrieval program.
Output: data in an orthonormal frame (laboratory or crystal frame), amp_disp_strain
array.The disp array should be divided by q to get the displacement (disp = -1*phase
here).

Laboratory frame: z downstream, y vertical, x outboard (CXI convention)
Crystal reciprocal frame: qx downstream, qz vertical, qy outboard
Detector convention: when out_of_plane angle=0   Y=-y , when in_plane angle=0   X=x

In arrays, when plotting the first parameter is the row (vertical axis), and the
second the column (horizontal axis). Therefore the data structure is data[qx, qz,
qy] for reciprocal space, or data[z, y, x] for real space

Usage:

 - from the command line:
   `python path_to/bcdi_strain.py --config_file path_to/config.yml`
 - directly from a code editor:
   update the constant CONFIG_FILE at the top of the file

    Parameters related to path names:

    :param scan: e.g. 11
     scan number
    :param root_folder: e.g. "C:/Users/Jerome/Documents/data/dataset_ID01/"
     folder of the experiment, where all scans are stored
    :param save_dir: e.g. "C:/Users/Jerome/Documents/data/dataset_ID01/test/"
     images will be saved here, leave it to None otherwise
    :param data_dir: e.g. None
     use this to override the beamline default search path for the data
    :param sample_name: e.g. "S"
     str or list of str of sample names (string in front of the scan number in the
     folder name). If only one name is indicated, it will be repeated to match the
     number of scans.
    :param comment: string use in filenames when saving
    :param debug: e.g. False
     True to see plots
    :param reconstruction_file: e.g. "modes.h5"
     path to a reconstruction file, to avoid opening a pop-up window

    Parameters used in the interactive masking GUI:

    :param backend: e.g. "Qt5Agg"
     Backend used in script, change to "Agg" to make sure the figures are saved, not
     compaticle with interactive masking. Other possibilities are
     'module://matplotlib_inline.backend_inline'
     default value is "Qt5Agg"

    Parameters used when averaging several reconstruction:

    :param sort_method: e.g. "variance/mean"
     'mean_amplitude' or 'variance' or 'variance/mean' or 'volume', metric for averaging
    :param averaging_space: e.g. "reciprocal_space"
     in which space to average, 'direct_space' or 'reciprocal_space'
    :param correlation_threshold: e.g. 0.90
     minimum correlation between two arrays to average them

    Parameters related to centering:

    :param centering_method: e.g. "max_com"
    'com' (center of mass), 'max', 'max_com' (max then com), 'do_nothing'
    :param roll_modes: e.g. [0, 0, 0]
    correct a roll of few pixels after the decomposition into modes in PyNX
    axis=(0, 1, 2)

    Prameters relative to the FFT window and voxel sizes:

    :param original_size: e.g. [150, 256, 500]
     size of the FFT array before binning. It will be modified to take into account
     binning during phasing automatically. Leave it to None if the shape did not change.
    :param phasing_binning: e.g. [1, 1, 1]
     binning factor applied during phase retrieval
    :param preprocessing_binning: e.g. [1, 2, 2]
     binning factors in each dimension used in preprocessing (not phase retrieval)
    :param output_size: e.g. [100, 100, 100]
     (z, y, x) Fix the size of the output array, leave None to use the object size
    :param keep_size: e.g. False
     True to keep the initial array size for orthogonalization (slower), it will be
     cropped otherwise
    :param fix_voxel: e.g. 10
     voxel size in nm for the interpolation during the geometrical transformation.
     If a single value is provided, the voxel size will be identical in all 3
     directions. Set it to None to use the default voxel size (calculated from q values,
     it will be different in each dimension).

    Parameters related to the strain calculation:

    :param data_frame: e.g. "detector"
     in which frame is defined the input data, available options:

     - 'crystal' if the data was interpolated into the crystal frame using
       xrayutilities or (transformation matrix + align_q=True)
     - 'laboratory' if the data was interpolated into the laboratory frame using
       the transformation matrix (align_q: False)
     - 'detector' if the data is still in the detector frame

    :param ref_axis_q: e.g. "y"
     axis along which q will be aligned (data_frame= 'detector' or 'laboratory') or is
     already aligned (data_frame='crystal')
    :param save_frame: e.g. "laboratory"
     in which frame should be saved the data, available options:

     - 'crystal' to save the data with q aligned along ref_axis_q
     - 'laboratory' to save the data in the laboratory frame (experimental geometry)
     - 'lab_flat_sample' to save the data in the laboratory frame, with all sample
       angles rotated back to 0. The rotations for 'laboratory' and 'lab_flat_sample'
       are realized after the strain calculation (which is always done in the crystal
       frame along ref_axis_q)

    :param isosurface_strain: e.g. 0.2
     threshold use for removing the outer layer (the strain is undefined at the exact
     surface voxel)
    :param strain_method: e.g. "default"
     how to calculate the strain, available options:

     - 'default': use the single value calculated from the gradient of the phase
     - 'defect': it will offset the phase in a loop and keep the smallest magnitude
       value for the strain. See: F. Hofmann et al. PhysRevMaterials 4, 013801 (2020)

    Parameters for the beamline:

    :param beamline: e.g. "ID01"
     name of the beamline, used for data loading and normalization by monitor
    :param is_series: e.g. True
     specific to series measurement at P10.
    :param actuators: e.g. {'rocking_angle': 'actuator_1_1'}
     optional dictionary that can be used to define the entries corresponding to
     actuators in data files (useful at CRISTAL where the location of data keeps
     changing, or to declare a non-standard monitor)
    :param is_series: e.g. True
     specific to series measurement at P10
    :param custom_scan: e.g. False
     True for a stack of images acquired without scan, e.g. with ct in a
     macro, or when there is no spec/log file available
    :param custom_images: list of image numbers for the custom_scan, None otherwise
    :param custom_monitor: list of monitor values for normalization for the custom_scan,
     None otherwise
    :param rocking_angle: e.g. "outofplane"
     "outofplane" for a sample rotation around x outboard, "inplane" for a sample
     rotation around y vertical up
    :param sdd: e.g. 0.50678
     in m, sample to detector distance in m
    :param energy: e.g. 9000
     X-ray energy in eV, leave None to use the default from the log file.
    :param beam_direction: e.g. [1, 0, 0]
     beam direction in the laboratory frame (downstream, vertical up, outboard)
    :param sample_offsets: e.g. None
     tuple of offsets in degrees of the sample for each sample circle (outer first).
     convention: the sample offsets will be subtracted to the motor values. Leave None
     if there is no offset.
    :param tilt_angle: e.g. 0.00537
     angular step size in degrees for the rocking angle
    :param direct_beam: e.g. [125, 362]
     [vertical, horizontal], direct beam position on the unbinned, full detector
     measured with detector angles given by `dirbeam_detector_angles`. It will be used
     to calculate the real detector angles for the measured Bragg peak. Leave None for
     no correction.
    :param dirbeam_detector_angles: e.g. [1, 25]
     [outofplane, inplane] detector angles in degrees for the direct beam measurement.
     Leave None for no correction
    :param bragg_peak: e.g. [121, 321, 256]
     Bragg peak position [z_bragg, y_bragg, x_bragg] considering the unbinned full
     detector. If 'outofplane_angle' and 'inplane_angle' are None and the direct beam
     position is provided, it will be used to calculate the correct detector angles.
     It is useful if there are hotpixels or intense aliens. Leave None otherwise.
    :param outofplane_angle: e.g. 42.6093
     detector angle in deg (rotation around x outboard, typically delta), corrected for
     the direct beam position. Leave None to use the uncorrected position.
    :param inplane_angle: e.g. -0.5783
     detector angle in deg(rotation around y vertical up, typically gamma), corrected
     for the direct beam position. Leave None to use the uncorrected position.
    :param specfile_name: e.g. "l5.spec"
     beamline-dependent parameter, use the following template:

     - template for ID01 and 34ID: name of the spec file if it is at the default
      location (in root_folder) or full path to the spec file
     - template for SIXS: full path of the alias dictionnary or None to use the one in
      the package folder
     - for P10, either None (if you are using the same directory structure as the
      beamline) or the full path to the .fio file
     - template for all other beamlines: None

    Parameters for custom scans:

    :param custom_scan: e.g. False
     True for a stack of images acquired without scan, e.g. with ct in a
     macro, or when there is no spec/log file available
    :param custom_motors: e.g. {"delta": 5.5, "gamma": 42.2, "theta": 1.1, "phi": 0}
     dictionary providing the goniometer positions of the beamline

    Parameters for the detector:

    :param detector: e.g. "Maxipix"
     name of the detector
    :param pixel_size: e.g. 100e-6
     use this to declare the pixel size of the "Dummy" detector if different from 55e-6
    :param center_roi_x: e.g. 1577
     horizontal pixel number of the center of the ROI for data loading.
     Leave None to use the full detector.
    :param center_roi_y: e.g. 833
     vertical pixel number of the center of the ROI for data loading.
     Leave None to use the full detector.
    :param roi_detector: e.g.[0, 250, 10, 210]
     region of interest of the detector to load. If "center_roi_x" or "center_roi_y" are
     not None, it will consider that the current values in roi_detector define a window
     around the pixel [center_roi_y, center_roi_x] and the final output will be
     [center_roi_y - roi_detector[0], center_roi_y + roi_detector[1],
     center_roi_x - roi_detector[2], center_roi_x + roi_detector[3]].
     Leave None to use the full detector. Use with center_fft='skip' if you want this
     exact size for the output.
    :param template_imagefile: e.g. "data_mpx4_%05d.edf.gz"
     use one of the following template:

     - template for ID01: 'data_mpx4_%05d.edf.gz' or 'align_eiger2M_%05d.edf.gz'
     - template for SIXS_2018: 'align.spec_ascan_mu_%05d.nxs'
     - template for SIXS_2019: 'spare_ascan_mu_%05d.nxs'
     - template for Cristal: 'S%d.nxs'
     - template for P10: '_master.h5'
     - template for NANOMAX: '%06d.h5'
     - template for 34ID: 'Sample%dC_ES_data_51_256_256.npz'

    Parameters related to the refraction correction:

    :param correct_refraction: e.g. True
     True for correcting the phase shift due to refraction
    :param optical_path_method: e.g. "threshold"
     'threshold' or 'defect', if 'threshold' it uses isosurface_strain to define the
     support  for the optical path calculation, if 'defect' (holes) it tries to remove
     only outer layers even if the amplitude is lower than isosurface_strain inside
     the crystal
    :param dispersion: e.g. 5.0328e-05
     delta value used for refraction correction, for Pt:  3.0761E-05 @ 10300eV,
     5.0328E-05 @ 8170eV, 3.2880E-05 @ 9994eV, 4.1184E-05 @ 8994eV, 5.2647E-05 @ 7994eV,
     4.6353E-05 @ 8500eV / Ge 1.4718E-05 @ 8keV
    :param absorption: e.g. 4.1969e-06
     beta value, for Pt:  2.0982E-06 @ 10300eV, 4.8341E-06 @ 8170eV,
     2.3486E-06 @ 9994eV, 3.4298E-06 @ 8994eV, 5.2245E-06 @ 7994eV, 4.1969E-06 @ 8500eV
    :param threshold_unwrap_refraction: e.g. 0.05
     threshold used to calculate the optical path. The threshold for refraction
     correction should be low, to correct for an object larger than the real one,
     otherwise it messes up the phase

    Parameters related to the phase:

    :param simulation: e.g. False
     True if it is a simulation, the parameter invert_phase will be set to 0 (see below)
    :param invert_phase: e.g. True
    True for the displacement to have the right sign (FFT convention), it is False only
    for simulations
    :param flip_reconstruction: e.g. True
     True if you want to get the conjugate object
    :param phase_ramp_removal: e.g. "gradient"
     'gradient' or 'upsampling', 'gradient' is much faster
    :param threshold_gradient: e.g. 1.0
     upper threshold of the gradient of the phase, use for ramp removal
    :param phase_offset: e.g. 0
     manual offset to add to the phase, should be 0 in most cases
    :param phase_offset_origin: e.g. [12, 32, 65]
     the phase at this voxel will be set to phase_offset, leave None to use the default
     position computed using offset_method (see below)
    :param offset_method: e.g. "mean"
     'com' (center of mass) or 'mean', method for determining the phase offset origin

    Parameters related to data visualization:

    :param debug: e.g. False
     True to show all plots for debugging
    :param align_axis: e.g. False
     True to rotate the crystal to align axis_to_align along ref_axis for visualization.
     This is done after the calculation of the strain and has no effect on it.
    :param ref_axis: e.g. "y"
     it will align axis_to_align to that axis if align_axis is True
    :param axis_to_align: e.g. [-0.01166, 0.9573, -0.2887]
     axis to align with ref_axis in the order x y z (axis 2, axis 1, axis 0)
    :param strain_range: e.g. 0.001
     range of the colorbar for strain plots
    :param phase_range: e.g. 0.4
     range of the colorbar for phase plots
    :param grey_background: e.g. True
     True to set the background to grey in phase and strain plots
    :param tick_spacing: e.g. 50
     spacing between axis ticks in plots, in nm
    :param tick_direction: e.g. "inout"
     direction of the ticks in plots: 'out', 'in', 'inout'
    :param tick_length: e.g. 3
     length of the ticks in plots
    :param tick_width: e.g. 1
     width of the ticks in plots

    Parameters for temperature estimation:

    :param get_temperature: e.g. False
     True to estimate the temperature, only available for platinum at the moment
    :param reflection: e.g. [1, 1, 1]
    measured reflection, use for estimating the temperature from the lattice parameter
    :param reference_spacing: 3.9236
     for calibrating the thermal expansion, if None it is fixed to the one of Platinum
     3.9236/norm(reflection)
    :param reference_temperature: 325
     temperature in Kelvins used to calibrate the thermal expansion, if None it is fixed
     to 293.15K (room temperature)

    Parameters for phase averaging or apodization:

    :param half_width_avg_phase: e.g. 0
     (width-1)/2 of the averaging window for the phase, 0 means no phase averaging
    :param apodize: e.g. False
     True to multiply the diffraction pattern by a filtering window
    :param apodization_window: e.g. "blackman"
     filtering window, multivariate 'normal' or 'tukey' or 'blackman'
    :param apodization_mu: e.g. [0.0, 0.0, 0.0]
     mu of the gaussian window
    :param apodization_sigma: e.g. [0.30, 0.30, 0.30]
     sigma of the gaussian window
    :param apodization_alpha: e.g. [1.0, 1.0, 1.0]
     shape parameter of the tukey window

    Parameters related to saving:

    :param save_rawdata: e.g. False
     True to save the amp-phase.vti before orthogonalization
    :param save_support: e.g. False
     True to save the non-orthogonal support for later phase retrieval
    :param save: e.g. True
     True to save amp.npz, phase.npz, strain.npz and vtk files

"""


def run(prm):
    """
    Run the postprocessing.

    :param prm: the parsed parameters
    """
    pretty = pprint.PrettyPrinter(indent=4)

    ################################
    # assign often used parameters #
    ################################
    bragg_peak = prm.get("bragg_peak")
    debug = prm.get("debug", False)
    comment = prm.get("comment", "")
    centering_method = prm.get("centering_method", "max_com")
    original_size = prm.get("original_size")
    phasing_binning = prm.get("phasing_binning", [1, 1, 1])
    preprocessing_binning = prm.get("preprocessing_binning", [1, 1, 1])
    ref_axis_q = prm.get("ref_axis_q", "y")
    fix_voxel = prm.get("fix_voxel")
    save = prm.get("save", True)
    tick_spacing = prm.get("tick_spacing", 50)
    tick_direction = prm.get("tick_direction", "inout")
    tick_length = prm.get("tick_length", 10)
    tick_width = prm.get("tick_width", 2)
    invert_phase = prm.get("invert_phase", True)
    correct_refraction = prm.get("correct_refraction", False)
    threshold_unwrap_refraction = prm.get("threshold_unwrap_refraction", 0.05)
    threshold_gradient = prm.get("threshold_gradient", 1.0)
    offset_method = prm.get("offset_method", "mean")
    phase_offset = prm.get("phase_offset", 0)
    offset_origin = prm.get("phase_offset_origin")
    sort_method = prm.get("sort_method", "variance/mean")
    correlation_threshold = prm.get("correlation_threshold", 0.90)

    # parameters below must be provided
    try:
        detector_name = prm["detector"]
        beamline_name = prm["beamline"]
        rocking_angle = prm["rocking_angle"]
        isosurface_strain = prm["isosurface_strain"]
        output_size = prm["output_size"]
        save_frame = prm["save_frame"]
        data_frame = prm["data_frame"]
        scan = prm["scan"]
        sample_name = prm["sample_name"]
        root_folder = prm["root_folder"]
    except KeyError as ex:
        print("Required parameter not defined")
        raise

    prm["sample"] = (f"{sample_name}+{scan}",)
    #########################
    # Check some parameters #
    #########################
    if prm.get("simulation", False):
        invert_phase = False
        correct_refraction = 0
    if invert_phase:
        phase_fieldname = "disp"
    else:
        phase_fieldname = "phase"

    if data_frame == "detector":
        is_orthogonal = False
    else:
        is_orthogonal = True

    if data_frame == "crystal" and save_frame != "crystal":
        print(
            "data already in the crystal frame before phase retrieval,"
            " it is impossible to come back to the laboratory "
            "frame, parameter 'save_frame' defaulted to 'crystal'"
        )
        save_frame = "crystal"

    axis_to_array_xyz = {
        "x": np.array([1, 0, 0]),
        "y": np.array([0, 1, 0]),
        "z": np.array([0, 0, 1]),
    }  # in xyz order

    ###############
    # Set backend #
    ###############
    if prm.get("backend") is not None:
        try:
            plt.switch_backend(prm["backend"])
        except ModuleNotFoundError:
            print(f"{prm['backend']} backend is not supported.")

    ###################
    # define colormap #
    ###################
    if prm.get("grey_background"):
        bad_color = "0.7"
    else:
        bad_color = "1.0"  # white background
    colormap = gu.Colormap(bad_color=bad_color)
    my_cmap = colormap.cmap

    #######################
    # Initialize detector #
    #######################
    detector = create_detector(
        name=detector_name,
        template_imagefile=prm.get("template_imagefile"),
        binning=phasing_binning,
        preprocessing_binning=preprocessing_binning,
        pixel_size=prm.get("pixel_size"),
    )

    ####################################
    # define the experimental geometry #
    ####################################
    setup = Setup(
        beamline=beamline_name,
        detector=detector,
        energy=prm.get("energy"),
        outofplane_angle=prm.get("outofplane_angle"),
        inplane_angle=prm.get("inplane_angle"),
        tilt_angle=prm.get("tilt_angle"),
        rocking_angle=rocking_angle,
        distance=prm.get("sdd"),
        sample_offsets=prm.get("sample_offsets"),
        actuators=prm.get("actuators"),
        custom_scan=prm.get("custom_scan", False),
        custom_motors=prm.get("custom_motors"),
        dirbeam_detector_angles=prm.get("dirbeam_detector_angles"),
        direct_beam=prm.get("direct_beam"),
        is_series=prm.get("is_series", False),
    )

    ########################################
    # Initialize the paths and the logfile #
    ########################################
    setup.init_paths(
        sample_name=sample_name,
        scan_number=scan,
        root_folder=root_folder,
        data_dir=prm.get("data_dir"),
        save_dir=prm.get("save_dir"),
        specfile_name=prm.get("specfile_name"),
        template_imagefile=prm.get("template_imagefile"),
    )

    setup.create_logfile(
        scan_number=scan, root_folder=root_folder, filename=detector.specfile
    )

    # load the goniometer positions needed in the calculation
    # of the transformation matrix
    setup.read_logfile(scan_number=scan)

    ###################
    # print instances #
    ###################
    print(f'{"#"*(5+len(str(scan)))}\nScan {scan}\n{"#"*(5+len(str(scan)))}')
    print("\n##############\nSetup instance\n##############")
    pretty.pprint(setup.params)
    print("\n#################\nDetector instance\n#################")
    pretty.pprint(detector.params)

    ################
    # preload data #
    ################
    if prm.get("reconstruction_file") is not None:
        file_path = (prm["reconstruction_file"],)
    else:
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilenames(
            initialdir=detector.scandir
            if prm.get("data_dir") is None
            else detector.datadir,
            filetypes=[
                ("NPZ", "*.npz"),
                ("NPY", "*.npy"),
                ("CXI", "*.cxi"),
                ("HDF5", "*.h5"),
            ],
        )

    nbfiles = len(file_path)
    plt.ion()

    obj, extension = util.load_file(file_path[0])
    if extension == ".h5":
        comment = comment + "_mode"

    print("\n###############\nProcessing data\n###############")
    nz, ny, nx = obj.shape
    print("Initial data size: (", nz, ",", ny, ",", nx, ")")
    if not original_size:
        original_size = obj.shape
    print("FFT size before accounting for phasing_binning", original_size)
    original_size = tuple(
        [
            original_size[index] // phasing_binning[index]
            for index in range(len(phasing_binning))
        ]
    )
    print("Binning used during phasing:", detector.binning)
    print("Padding back to original FFT size", original_size)
    obj = util.crop_pad(array=obj, output_shape=original_size)

    ###########################################################################
    # define range for orthogonalization and plotting - speed up calculations #
    ###########################################################################
    zrange, yrange, xrange = pu.find_datarange(
        array=obj, amplitude_threshold=0.05, keep_size=prm.get("keep_size", False)
    )

    numz = zrange * 2
    numy = yrange * 2
    numx = xrange * 2
    print(
        f"Data shape used for orthogonalization and plotting: ({numz}, {numy}, {numx})"
    )

    ####################################################################################
    # find the best reconstruction from the list, based on mean amplitude and variance #
    ####################################################################################
    if nbfiles > 1:
        print(
            "\nTrying to find the best reconstruction\nSorting by ",
            sort_method
        )
        sorted_obj = pu.sort_reconstruction(
            file_path=file_path,
            amplitude_threshold=isosurface_strain,
            data_range=(zrange, yrange, xrange),
            sort_method=sort_method,
        )
    else:
        sorted_obj = [0]

    #######################################
    # load reconstructions and average it #
    #######################################
    avg_obj = np.zeros((numz, numy, numx))
    ref_obj = np.zeros((numz, numy, numx))
    avg_counter = 1
    print("\nAveraging using", nbfiles, "candidate reconstructions")
    for counter, value in enumerate(sorted_obj):
        obj, extension = util.load_file(file_path[value])
        print("\nOpening ", file_path[value])
        prm[f"from_file_{counter}"] = file_path[value]

        if prm.get("flip_reconstruction", False):
            obj = pu.flip_reconstruction(obj, debugging=True)

        if extension == ".h5":
            centering_method = "do_nothing"  # do not center, data is already cropped
            # just on support for mode decomposition
            # correct a roll after the decomposition into modes in PyNX
            obj = np.roll(obj, prm.get("roll_modes", [0, 0, 0]), axis=(0, 1, 2))
            fig, _, _ = gu.multislices_plot(
                abs(obj),
                sum_frames=True,
                plot_colorbar=True,
                title="1st mode after centering",
            )

        # use the range of interest defined above
        obj = util.crop_pad(obj, [2 * zrange, 2 * yrange, 2 * xrange], debugging=False)

        # align with average reconstruction
        if counter == 0:  # the fist array loaded will serve as reference object
            print("This reconstruction will be used as reference.")
            ref_obj = obj

        avg_obj, flag_avg = reg.average_arrays(
            avg_obj=avg_obj,
            ref_obj=ref_obj,
            obj=obj,
            support_threshold=0.25,
            correlation_threshold=correlation_threshold,
            aligning_option="dft",
            space=prm.get("averaging_space", "reciprocal_space"),
            reciprocal_space=False,
            is_orthogonal=is_orthogonal,
            debugging=debug,
        )
        avg_counter = avg_counter + flag_avg

    avg_obj = avg_obj / avg_counter
    if avg_counter > 1:
        print("\nAverage performed over ", avg_counter, "reconstructions\n")
    del obj, ref_obj
    gc.collect()

    ################
    # unwrap phase #
    ################
    phase, extent_phase = pu.unwrap(
        avg_obj,
        support_threshold=threshold_unwrap_refraction,
        debugging=debug,
        reciprocal_space=False,
        is_orthogonal=is_orthogonal,
    )

    print(
        "Extent of the phase over an extended support (ceil(phase range)) ~ ",
        int(extent_phase),
        "(rad)",
    )
    phase = util.wrap(phase, start_angle=-extent_phase / 2, range_angle=extent_phase)
    if debug:
        gu.multislices_plot(
            phase,
            width_z=2 * zrange,
            width_y=2 * yrange,
            width_x=2 * xrange,
            plot_colorbar=True,
            title="Phase after unwrap + wrap",
            reciprocal_space=False,
            is_orthogonal=is_orthogonal,
        )

    #############################################
    # phase ramp removal before phase filtering #
    #############################################
    amp, phase, rampz, rampy, rampx = pu.remove_ramp(
        amp=abs(avg_obj),
        phase=phase,
        initial_shape=original_size,
        method="gradient",
        amplitude_threshold=isosurface_strain,
        threshold_gradient=threshold_gradient,
    )
    del avg_obj
    gc.collect()

    if debug:
        gu.multislices_plot(
            phase,
            width_z=2 * zrange,
            width_y=2 * yrange,
            width_x=2 * xrange,
            plot_colorbar=True,
            title="Phase after ramp removal",
            reciprocal_space=False,
            is_orthogonal=is_orthogonal,
        )

    ########################
    # phase offset removal #
    ########################
    support = np.zeros(amp.shape)
    support[amp > isosurface_strain * amp.max()] = 1
    phase = pu.remove_offset(
        array=phase,
        support=support,
        offset_method=offset_method,
        phase_offset=phase_offset,
        offset_origin=offset_origin,
        title="Phase",
        debugging=debug,
    )
    del support
    gc.collect()

    phase = util.wrap(
        obj=phase, start_angle=-extent_phase / 2, range_angle=extent_phase
    )

    ##############################################################################
    # average the phase over a window or apodize to reduce noise in strain plots #
    ##############################################################################
    half_width_avg_phase = prm.get("half_width_avg_phase", 0)
    if half_width_avg_phase != 0:
        bulk = pu.find_bulk(
            amp=amp, support_threshold=isosurface_strain, method="threshold"
        )
        # the phase should be averaged only in the support defined by the isosurface
        phase = pu.mean_filter(
            array=phase, support=bulk, half_width=half_width_avg_phase
        )
        del bulk
        gc.collect()

    if half_width_avg_phase != 0:
        comment = comment + "_avg" + str(2 * half_width_avg_phase + 1)

    gridz, gridy, gridx = np.meshgrid(
        np.arange(0, numz, 1),
        np.arange(0, numy, 1),
        np.arange(0, numx, 1),
        indexing="ij",
    )

    phase = (
        phase + gridz * rampz + gridy * rampy + gridx * rampx
    )  # put back the phase ramp otherwise the diffraction
    # pattern will be shifted and the prtf messed up

    if prm.get("apodize", False):
        amp, phase = pu.apodize(
            amp=amp,
            phase=phase,
            initial_shape=original_size,
            window_type=prm.get("apodization_window", "blackman"),
            sigma=prm.get("apodization_sigma", [0.30, 0.30, 0.30]),
            mu=prm.get("apodization_mu", [0.0, 0.0, 0.0]),
            alpha=prm.get("apodization_alpha", [1.0, 1.0, 1.0]),
            is_orthogonal=is_orthogonal,
            debugging=True,
        )
        comment = comment + "_apodize_" + prm.get("apodization_window", "blackman")

    ################################################################
    # save the phase with the ramp for PRTF calculations,          #
    # otherwise the object will be misaligned with the measurement #
    ################################################################
    np.savez_compressed(
        detector.savedir + "S" + str(scan) + "_avg_obj_prtf" + comment,
        obj=amp * np.exp(1j * phase),
    )

    ####################################################
    # remove again phase ramp before orthogonalization #
    ####################################################
    phase = phase - gridz * rampz - gridy * rampy - gridx * rampx

    avg_obj = amp * np.exp(1j * phase)  # here the phase is again wrapped in [-pi pi[

    del amp, phase, gridz, gridy, gridx, rampz, rampy, rampx
    gc.collect()

    ######################
    # centering of array #
    ######################
    if centering_method == "max":
        avg_obj = pu.center_max(avg_obj)
        # shift based on max value,
        # required if it spans across the edge of the array before COM
    elif centering_method == "com":
        avg_obj = pu.center_com(avg_obj)
    elif centering_method == "max_com":
        avg_obj = pu.center_max(avg_obj)
        avg_obj = pu.center_com(avg_obj)

    #######################
    #  save support & vti #
    #######################
    if prm.get("save_support", False):
        # to be used as starting support in phasing, hence still in the detector frame
        support = np.zeros((numz, numy, numx))
        support[abs(avg_obj) / abs(avg_obj).max() > 0.01] = 1
        # low threshold because support will be cropped by shrinkwrap during phasing
        np.savez_compressed(
            detector.savedir + "S" + str(scan) + "_support" + comment, obj=support
        )
        del support
        gc.collect()

    if prm.get("save_rawdata", False):
        np.savez_compressed(
            detector.savedir + "S" + str(scan) + "_raw_amp-phase" + comment,
            amp=abs(avg_obj),
            phase=np.angle(avg_obj),
        )

        # voxel sizes in the detector frame
        voxel_z, voxel_y, voxel_x = setup.voxel_sizes_detector(
            array_shape=original_size,
            tilt_angle=(
                prm.get("tilt_angle")
                * detector.preprocessing_binning[0]
                * detector.binning[0]
            ),
            pixel_x=detector.pixelsize_x,
            pixel_y=detector.pixelsize_y,
            verbose=True,
        )
        # save raw amp & phase to VTK
        # in VTK, x is downstream, y vertical, z inboard,
        # thus need to flip the last axis
        gu.save_to_vti(
            filename=os.path.join(
                detector.savedir, "S" + str(scan) + "_raw_amp-phase" + comment + ".vti"
            ),
            voxel_size=(voxel_z, voxel_y, voxel_x),
            tuple_array=(abs(avg_obj), np.angle(avg_obj)),
            tuple_fieldnames=("amp", "phase"),
            amplitude_threshold=0.01,
        )

    #########################################################
    # calculate q of the Bragg peak in the laboratory frame #
    #########################################################
    q_lab = (
        setup.q_laboratory
    )  # (1/A), in the laboratory frame z downstream, y vertical, x outboard
    qnorm = np.linalg.norm(q_lab)
    q_lab = q_lab / qnorm

    angle = simu.angle_vectors(
        ref_vector=[q_lab[2], q_lab[1], q_lab[0]],
        test_vector=axis_to_array_xyz[ref_axis_q],
    )
    print(
        f"\nNormalized diffusion vector in the laboratory frame (z*, y*, x*): "
        f"({q_lab[0]:.4f} 1/A, {q_lab[1]:.4f} 1/A, {q_lab[2]:.4f} 1/A)"
    )

    planar_dist = 2 * np.pi / qnorm  # qnorm should be in angstroms
    print(f"Wavevector transfer: {qnorm:.4f} 1/A")
    print(f"Atomic planar distance: {planar_dist:.4f} A")
    print(f"\nAngle between q_lab and {ref_axis_q} = {angle:.2f} deg")
    if debug:
        print(
            "Angle with y in zy plane = "
            f"{np.arctan(q_lab[0]/q_lab[1])*180/np.pi:.2f} deg"
        )
        print(
            "Angle with y in xy plane = "
            f"{np.arctan(-q_lab[2]/q_lab[1])*180/np.pi:.2f} deg"
        )
        print(
            "Angle with z in xz plane = "
            f"{180+np.arctan(q_lab[2]/q_lab[0])*180/np.pi:.2f} deg\n"
        )

    planar_dist = planar_dist / 10  # switch to nm

    #######################
    #  orthogonalize data #
    #######################
    print("\nShape before orthogonalization", avg_obj.shape)
    if data_frame == "detector":
        if debug:
            phase, _ = pu.unwrap(
                avg_obj,
                support_threshold=threshold_unwrap_refraction,
                debugging=True,
                reciprocal_space=False,
                is_orthogonal=False,
            )
            gu.multislices_plot(
                phase,
                width_z=2 * zrange,
                width_y=2 * yrange,
                width_x=2 * xrange,
                sum_frames=False,
                plot_colorbar=True,
                reciprocal_space=False,
                is_orthogonal=False,
                title="unwrapped phase before orthogonalization",
            )
            del phase
            gc.collect()

        if not prm.get("outofplane_angle") and not prm.get("inplane_angle"):
            # corrected detector angles not provided
            if bragg_peak is None:
                # Bragg peak position not provided, find it from the data
                # TODO: put this in a try except block (templateimagefile not defined)
                data, _, _, _ = setup.diffractometer.load_check_dataset(
                    scan_number=scan,
                    detector=detector,
                    setup=setup,
                    frames_pattern=prm.get("frames_pattern"),
                    bin_during_loading=False,
                    flatfield=prm.get("flatfield"),
                    hotpixels=prm.get("hotpix_array"),
                    background=prm.get("background"),
                    normalize=prm.get("normalize_flux", "skip"),
                )
                bragg_peak = bu.find_bragg(
                    data=data,
                    peak_method='maxcom',
                    roi=detector.roi,
                    binning=None,
                )
            setup.correct_detector_angles(bragg_peak_position=bragg_peak)

        obj_ortho, voxel_size, transfer_matrix = setup.ortho_directspace(
            arrays=avg_obj,
            q_com=np.array([q_lab[2], q_lab[1], q_lab[0]]),
            initial_shape=original_size,
            voxel_size=fix_voxel,
            reference_axis=axis_to_array_xyz[ref_axis_q],
            fill_value=0,
            debugging=True,
            title="amplitude",
        )
        prm["transformation_matrix"] = transfer_matrix
    else:  # data already orthogonalized using xrayutilities
        # or the linearized transformation matrix
        obj_ortho = avg_obj
        try:
            print("Select the file containing QxQzQy")
            file_path = filedialog.askopenfilename(
                title="Select the file containing QxQzQy",
                initialdir=detector.savedir,
                filetypes=[("NPZ", "*.npz")],
            )
            npzfile = np.load(file_path)
            qx = npzfile["qx"]
            qy = npzfile["qy"]
            qz = npzfile["qz"]
        except FileNotFoundError:
            raise FileNotFoundError(
                "q values not provided, the voxel size cannot be calculated"
            )
        dy_real = (
            2 * np.pi / abs(qz.max() - qz.min()) / 10
        )  # in nm qz=y in nexus convention
        dx_real = (
            2 * np.pi / abs(qy.max() - qy.min()) / 10
        )  # in nm qy=x in nexus convention
        dz_real = (
            2 * np.pi / abs(qx.max() - qx.min()) / 10
        )  # in nm qx=z in nexus convention
        print(
            f"direct space voxel size from q values: ({dz_real:.2f} nm,"
            f" {dy_real:.2f} nm, {dx_real:.2f} nm)"
        )
        if fix_voxel:
            voxel_size = fix_voxel
            print(f"Direct space pixel size for the interpolation: {voxel_size} (nm)")
            print("Interpolating...\n")
            obj_ortho = pu.regrid(
                array=obj_ortho,
                old_voxelsize=(dz_real, dy_real, dx_real),
                new_voxelsize=voxel_size,
            )
        else:
            # no need to interpolate
            voxel_size = dz_real, dy_real, dx_real  # in nm

        if (
            data_frame == "laboratory"
        ):  # the object must be rotated into the crystal frame
            # before the strain calculation
            print("Rotating the object in the crystal frame for the strain calculation")

            amp, phase = util.rotate_crystal(
                arrays=(abs(obj_ortho), np.angle(obj_ortho)),
                is_orthogonal=True,
                reciprocal_space=False,
                voxel_size=voxel_size,
                debugging=(True, False),
                axis_to_align=q_lab[::-1],
                reference_axis=axis_to_array_xyz[ref_axis_q],
                title=("amp", "phase"),
            )

            obj_ortho = amp * np.exp(
                1j * phase
            )  # here the phase is again wrapped in [-pi pi[
            del amp, phase

    del avg_obj
    gc.collect()

    ######################################################
    # center the object (centering based on the modulus) #
    ######################################################
    print("\nCentering the crystal")
    obj_ortho = pu.center_com(obj_ortho)

    ####################
    # Phase unwrapping #
    ####################
    print("\nPhase unwrapping")
    phase, extent_phase = pu.unwrap(
        obj_ortho,
        support_threshold=threshold_unwrap_refraction,
        debugging=True,
        reciprocal_space=False,
        is_orthogonal=True,
    )
    amp = abs(obj_ortho)
    del obj_ortho
    gc.collect()

    #############################################
    # invert phase: -1*phase = displacement * q #
    #############################################
    if invert_phase:
        phase = -1 * phase

    ########################################
    # refraction and absorption correction #
    ########################################
    if correct_refraction:  # or correct_absorption:
        bulk = pu.find_bulk(
            amp=amp,
            support_threshold=threshold_unwrap_refraction,
            method=prm.get("optical_path_method", "threshold"),
            debugging=debug,
        )

        kin = setup.incident_wavevector
        kout = setup.exit_wavevector
        # kin and kout were calculated in the laboratory frame,
        # but after the geometric transformation of the crystal, this
        # latter is always in the crystal frame (for simpler strain calculation).
        # We need to transform kin and kout back
        # into the crystal frame (also, xrayutilities output is in crystal frame)
        kin = util.rotate_vector(
            vectors=[kin[2], kin[1], kin[0]],
            axis_to_align=axis_to_array_xyz[ref_axis_q],
            reference_axis=[q_lab[2], q_lab[1], q_lab[0]],
        )
        kout = util.rotate_vector(
            vectors=[kout[2], kout[1], kout[0]],
            axis_to_align=axis_to_array_xyz[ref_axis_q],
            reference_axis=[q_lab[2], q_lab[1], q_lab[0]],
        )

        # calculate the optical path of the incoming wavevector
        path_in = pu.get_opticalpath(
            support=bulk, direction="in", k=kin, debugging=debug
        )  # path_in already in nm

        # calculate the optical path of the outgoing wavevector
        path_out = pu.get_opticalpath(
            support=bulk, direction="out", k=kout, debugging=debug
        )  # path_our already in nm

        optical_path = path_in + path_out
        del path_in, path_out
        gc.collect()

        if correct_refraction:
            phase_correction = (
                2 * np.pi / (1e9 * setup.wavelength) * prm["dispersion"] * optical_path
            )
            phase = phase + phase_correction

            gu.multislices_plot(
                np.multiply(phase_correction, bulk),
                width_z=2 * zrange,
                width_y=2 * yrange,
                width_x=2 * xrange,
                sum_frames=False,
                plot_colorbar=True,
                vmin=0,
                vmax=np.nan,
                title="Refraction correction on the support",
                is_orthogonal=True,
                reciprocal_space=False,
            )
        correct_absorption = False
        if correct_absorption:
            amp_correction = np.exp(
                2 * np.pi / (1e9 * setup.wavelength) * prm["absorption"] * optical_path
            )
            amp = amp * amp_correction

            gu.multislices_plot(
                np.multiply(amp_correction, bulk),
                width_z=2 * zrange,
                width_y=2 * yrange,
                width_x=2 * xrange,
                sum_frames=False,
                plot_colorbar=True,
                vmin=1,
                vmax=1.1,
                title="Absorption correction on the support",
                is_orthogonal=True,
                reciprocal_space=False,
            )

        del bulk, optical_path
        gc.collect()

    ##############################################
    # phase ramp and offset removal (mean value) #
    ##############################################
    print("\nPhase ramp removal")
    amp, phase, _, _, _ = pu.remove_ramp(
        amp=amp,
        phase=phase,
        initial_shape=original_size,
        method=prm.get("phase_ramp_removal", "gradient"),
        amplitude_threshold=isosurface_strain,
        threshold_gradient=threshold_gradient,
        debugging=debug,
    )

    ########################
    # phase offset removal #
    ########################
    print("\nPhase offset removal")
    support = np.zeros(amp.shape)
    support[amp > isosurface_strain * amp.max()] = 1
    phase = pu.remove_offset(
        array=phase,
        support=support,
        offset_method=offset_method,
        phase_offset=phase_offset,
        offset_origin=offset_origin,
        title="Orthogonal phase",
        debugging=debug,
        reciprocal_space=False,
        is_orthogonal=True,
    )
    del support
    gc.collect()
    # Wrap the phase around 0 (no more offset)
    phase = util.wrap(
        obj=phase, start_angle=-extent_phase / 2, range_angle=extent_phase
    )

    ################################################################
    # calculate the strain depending on which axis q is aligned on #
    ################################################################
    print(f"\nCalculation of the strain along {ref_axis_q}")
    strain = pu.get_strain(
        phase=phase,
        planar_distance=planar_dist,
        voxel_size=voxel_size,
        reference_axis=ref_axis_q,
        extent_phase=extent_phase,
        method=prm.get("strain_method", "default"),
        debugging=debug,
    )

    ################################################
    # optionally rotates back the crystal into the #
    # laboratory frame (for debugging purpose)     #
    ################################################
    q_final = None
    if save_frame in {"laboratory", "lab_flat_sample"}:
        comment = comment + "_labframe"
        print("\nRotating back the crystal in laboratory frame")
        amp, phase, strain = util.rotate_crystal(
            arrays=(amp, phase, strain),
            axis_to_align=axis_to_array_xyz[ref_axis_q],
            voxel_size=voxel_size,
            is_orthogonal=True,
            reciprocal_space=False,
            reference_axis=[q_lab[2], q_lab[1], q_lab[0]],
            debugging=(True, False, False),
            title=("amp", "phase", "strain"),
        )
        # q_lab is already in the laboratory frame
        q_final = q_lab

    if save_frame == "lab_flat_sample":
        comment = comment + "_flat"
        print("\nSending sample stage circles to 0")
        (amp, phase, strain), q_final = setup.diffractometer.flatten_sample(
            arrays=(amp, phase, strain),
            voxel_size=voxel_size,
            q_com=q_lab[::-1],  # q_com needs to be in xyz order
            is_orthogonal=True,
            reciprocal_space=False,
            rocking_angle=setup.rocking_angle,
            debugging=(True, False, False),
            title=("amp", "phase", "strain"),
        )
    if save_frame == "crystal":
        # rotate also q_lab to have it along ref_axis_q,
        # as a cross-checkm, vectors needs to be in xyz order
        comment = comment + "_crystalframe"
        q_final = util.rotate_vector(
            vectors=q_lab[::-1],
            axis_to_align=axis_to_array_xyz[ref_axis_q],
            reference_axis=q_lab[::-1],
        )

    ###############################################
    # rotates the crystal e.g. for easier slicing #
    # of the result along a particular direction  #
    ###############################################
    # typically this is an inplane rotation, q should stay aligned with the axis
    # along which the strain was calculated
    if prm.get("align_axis", False):
        print("\nRotating arrays for visualization")
        amp, phase, strain = util.rotate_crystal(
            arrays=(amp, phase, strain),
            reference_axis=axis_to_array_xyz[prm["ref_axis"]],
            axis_to_align=prm["axis_to_align"],
            voxel_size=voxel_size,
            debugging=(True, False, False),
            is_orthogonal=True,
            reciprocal_space=False,
            title=("amp", "phase", "strain"),
        )
        # rotate q accordingly, vectors needs to be in xyz order
        q_final = util.rotate_vector(
            vectors=q_final[::-1],
            axis_to_align=axis_to_array_xyz[prm["ref_axis"]],
            reference_axis=prm["axis_to_align"],
        )

    print(
        f"\nq_final = ({q_final[0]:.4f} 1/A,"
        f" {q_final[1]:.4f} 1/A, {q_final[2]:.4f} 1/A)"
    )

    ##############################################
    # pad array to fit the output_size parameter #
    ##############################################
    if output_size is not None:
        amp = util.crop_pad(array=amp, output_shape=output_size)
        phase = util.crop_pad(array=phase, output_shape=output_size)
        strain = util.crop_pad(array=strain, output_shape=output_size)
    print(f"\nFinal data shape: {amp.shape}")

    ######################
    # save result to vtk #
    ######################
    print(
        f"\nVoxel size: ({voxel_size[0]:.2f} nm, {voxel_size[1]:.2f} nm,"
        f" {voxel_size[2]:.2f} nm)"
    )
    bulk = pu.find_bulk(
        amp=amp, support_threshold=isosurface_strain, method="threshold"
    )
    if save:
        prm["comment"] = comment
        np.savez_compressed(
            f"{detector.savedir}S{scan}_amp{phase_fieldname}strain{comment}",
            amp=amp,
            phase=phase,
            bulk=bulk,
            strain=strain,
            q_com=q_final,
            voxel_sizes=voxel_size,
            detector=detector.params,
            setup=setup.params,
            params=prm,
        )

        # save results in hdf5 file
        with h5py.File(
            f"{detector.savedir}S{scan}_amp{phase_fieldname}strain{comment}.h5", "w"
        ) as hf:
            out = hf.create_group("output")
            par = hf.create_group("params")
            out.create_dataset("amp", data=amp)
            out.create_dataset("bulk", data=bulk)
            out.create_dataset("phase", data=phase)
            out.create_dataset("strain", data=strain)
            out.create_dataset("q_com", data=q_final)
            out.create_dataset("voxel_sizes", data=voxel_size)
            par.create_dataset("detector", data=str(detector.params))
            par.create_dataset("setup", data=str(setup.params))
            par.create_dataset("parameters", data=str(prm))

        # save amp & phase to VTK
        # in VTK, x is downstream, y vertical, z inboard,
        # thus need to flip the last axis
        gu.save_to_vti(
            filename=os.path.join(
                detector.savedir,
                "S"
                + str(scan)
                + "_amp-"
                + phase_fieldname
                + "-strain"
                + comment
                + ".vti",
            ),
            voxel_size=voxel_size,
            tuple_array=(amp, bulk, phase, strain),
            tuple_fieldnames=("amp", "bulk", phase_fieldname, "strain"),
            amplitude_threshold=0.01,
        )

    ######################################
    # estimate the volume of the crystal #
    ######################################
    amp = amp / amp.max()
    temp_amp = np.copy(amp)
    temp_amp[amp < isosurface_strain] = 0
    temp_amp[np.nonzero(temp_amp)] = 1
    volume = temp_amp.sum() * reduce(lambda x, y: x * y, voxel_size)  # in nm3
    del temp_amp
    gc.collect()

    ##############################
    # plot slices of the results #
    ##############################
    pixel_spacing = [tick_spacing / vox for vox in voxel_size]
    print(
        "\nPhase extent without / with thresholding the modulus "
        f"(threshold={isosurface_strain}): {phase.max()-phase.min():.2f} rad, "
        f"{phase[np.nonzero(bulk)].max()-phase[np.nonzero(bulk)].min():.2f} rad"
    )
    piz, piy, pix = np.unravel_index(phase.argmax(), phase.shape)
    print(
        f"phase.max() = {phase[np.nonzero(bulk)].max():.2f} "
        f"at voxel ({piz}, {piy}, {pix})"
    )
    strain[bulk == 0] = np.nan
    phase[bulk == 0] = np.nan

    # plot the slice at the maximum phase
    gu.combined_plots(
        (phase[piz, :, :], phase[:, piy, :], phase[:, :, pix]),
        tuple_sum_frames=False,
        tuple_sum_axis=0,
        tuple_width_v=None,
        tuple_width_h=None,
        tuple_colorbar=True,
        tuple_vmin=np.nan,
        tuple_vmax=np.nan,
        tuple_title=("phase at max in xy", "phase at max in xz", "phase at max in yz"),
        tuple_scale="linear",
        cmap=my_cmap,
        is_orthogonal=True,
        reciprocal_space=False,
    )

    # bulk support
    fig, _, _ = gu.multislices_plot(
        bulk,
        sum_frames=False,
        title="Orthogonal bulk",
        vmin=0,
        vmax=1,
        is_orthogonal=True,
        reciprocal_space=False,
    )
    fig.text(0.60, 0.45, "Scan " + str(scan), size=20)
    fig.text(
        0.60,
        0.40,
        "Bulk - isosurface=" + str("{:.2f}".format(isosurface_strain)),
        size=20,
    )
    plt.pause(0.1)
    if save:
        plt.savefig(detector.savedir + "S" + str(scan) + "_bulk" + comment + ".png")

    # amplitude
    fig, _, _ = gu.multislices_plot(
        amp,
        sum_frames=False,
        title="Normalized orthogonal amp",
        vmin=0,
        vmax=1,
        tick_direction=tick_direction,
        tick_width=tick_width,
        tick_length=tick_length,
        pixel_spacing=pixel_spacing,
        plot_colorbar=True,
        is_orthogonal=True,
        reciprocal_space=False,
    )
    fig.text(0.60, 0.45, f"Scan {scan}", size=20)
    fig.text(
        0.60,
        0.40,
        f"Voxel size=({voxel_size[0]:.1f}, {voxel_size[1]:.1f}, "
        f"{voxel_size[2]:.1f}) (nm)",
        size=20,
    )
    fig.text(0.60, 0.35, f"Ticks spacing={tick_spacing} nm", size=20)
    fig.text(0.60, 0.30, f"Volume={int(volume)} nm3", size=20)
    fig.text(0.60, 0.25, "Sorted by " + sort_method, size=20)
    fig.text(
        0.60, 0.20, f"correlation threshold={correlation_threshold}", size=20
    )
    fig.text(0.60, 0.15, f"average over {avg_counter} reconstruction(s)", size=20)
    fig.text(0.60, 0.10, f"Planar distance={planar_dist:.5f} nm", size=20)
    if prm.get("get_temperature", False):
        temperature = pu.bragg_temperature(
            spacing=planar_dist * 10,
            reflection=prm["reflection"],
            spacing_ref=prm.get("reference_spacing"),
            temperature_ref=prm.get("reference_temperature"),
            use_q=False,
            material="Pt",
        )
        fig.text(0.60, 0.05, f"Estimated T={temperature} C", size=20)
    if save:
        plt.savefig(detector.savedir + f"S{scan}_amp" + comment + ".png")

    # amplitude histogram
    fig, ax = plt.subplots(1, 1)
    ax.hist(amp[amp > 0.05 * amp.max()].flatten(), bins=250)
    ax.set_ylim(bottom=1)
    ax.tick_params(
        labelbottom=True,
        labelleft=True,
        direction="out",
        length=tick_length,
        width=tick_width,
    )
    ax.spines["right"].set_linewidth(1.5)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["top"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    fig.savefig(detector.savedir + f"S{scan}_histo_amp" + comment + ".png")

    # phase
    fig, _, _ = gu.multislices_plot(
        phase,
        sum_frames=False,
        title="Orthogonal displacement",
        vmin=-prm.get("phase_range", np.pi/2),
        vmax=prm.get("phase_range", np.pi/2),
        tick_direction=tick_direction,
        cmap=my_cmap,
        tick_width=tick_width,
        tick_length=tick_length,
        pixel_spacing=pixel_spacing,
        plot_colorbar=True,
        is_orthogonal=True,
        reciprocal_space=False,
    )
    fig.text(0.60, 0.30, f"Scan {scan}", size=20)
    fig.text(
        0.60,
        0.25,
        f"Voxel size=({voxel_size[0]:.1f}, {voxel_size[1]:.1f}, "
        f"{voxel_size[2]:.1f}) (nm)",
        size=20,
    )
    fig.text(0.60, 0.20, f"Ticks spacing={tick_spacing} nm", size=20)
    fig.text(0.60, 0.15, f"average over {avg_counter} reconstruction(s)", size=20)
    if half_width_avg_phase > 0:
        fig.text(
            0.60, 0.10, f"Averaging over {2*half_width_avg_phase+1} pixels", size=20
        )
    else:
        fig.text(0.60, 0.10, "No phase averaging", size=20)
    if save:
        plt.savefig(detector.savedir + f"S{scan}_displacement" + comment + ".png")

    # strain
    fig, _, _ = gu.multislices_plot(
        strain,
        sum_frames=False,
        title="Orthogonal strain",
        vmin=-prm.get("strain_range", 0.002),
        vmax=prm.get("strain_range", 0.002),
        tick_direction=tick_direction,
        tick_width=tick_width,
        tick_length=tick_length,
        plot_colorbar=True,
        cmap=my_cmap,
        pixel_spacing=pixel_spacing,
        is_orthogonal=True,
        reciprocal_space=False,
    )
    fig.text(0.60, 0.30, f"Scan {scan}", size=20)
    fig.text(
        0.60,
        0.25,
        f"Voxel size=({voxel_size[0]:.1f}, "
        f"{voxel_size[1]:.1f}, {voxel_size[2]:.1f}) (nm)",
        size=20,
    )
    fig.text(0.60, 0.20, f"Ticks spacing={tick_spacing} nm", size=20)
    fig.text(0.60, 0.15, f"average over {avg_counter} reconstruction(s)", size=20)
    if half_width_avg_phase > 0:
        fig.text(
            0.60, 0.10, f"Averaging over {2*half_width_avg_phase+1} pixels", size=20
        )
    else:
        fig.text(0.60, 0.10, "No phase averaging", size=20)
    if save:
        plt.savefig(detector.savedir + f"S{scan}_strain" + comment + ".png")


if __name__ == "__main__":
    # construct the argument parser and parse the command-line arguments
    ap = argparse.ArgumentParser()
    ap = add_cli_parameters(ap)
    cli_args = vars(ap.parse_args())

    # load the config file
    file = cli_args.get("config_file") or CONFIG_FILE
    parser = ConfigParser(file, cli_args)
    args = parser.load_arguments()
    args["time"] = f"{datetime.now()}"
    run(prm=args)

    print("\nEnd of script")
    plt.ioff()
    plt.show()
