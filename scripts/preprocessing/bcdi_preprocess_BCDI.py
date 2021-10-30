#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import argparse
from datetime import datetime

try:
    import hdf5plugin  # for P10, should be imported before h5py or PyTables
except ModuleNotFoundError:
    pass
import gc
import matplotlib.pyplot as plt
import numpy as np
import os
import pprint
import scipy.signal  # for medfilt2d
from scipy.ndimage.measurements import center_of_mass
import sys
from scipy.io import savemat
import tkinter as tk
from tkinter import filedialog
import xrayutilities as xu

import bcdi.graph.graph_utils as gu
from bcdi.experiment.detector import create_detector, create_roi
from bcdi.experiment.setup import Setup
import bcdi.postprocessing.postprocessing_utils as pu
import bcdi.preprocessing.bcdi_utils as bu
from bcdi.utils.parser import add_cli_parameters, ConfigParser
import bcdi.utils.utilities as util

CONFIG_FILE = (
    "C:/Users/Jerome/Documents/myscripts/clement/bcdi/conf/config_preprocessing.yml"
)

helptext = """
Prepare experimental data for Bragg CDI phasing: crop/pad, center, mask, normalize and
filter the data.

Beamlines currently supported: ESRF ID01, SOLEIL CRISTAL, SOLEIL SIXS, PETRAIII P10 and
APS 34ID-C.

The directory structure expected by default is (e.g. scan 1):
specfile, hotpixels file and flatfield file in:    /rootdir/
data in:                                           /rootdir/S1/data/

output files saved in:   /rootdir/S1/pynxraw/ or /rootdir/S1/pynx/ depending on the
'use_rawdata' option.

If you directory structure is different, you can use the parameter data_dir to indicate
where the data is.

Usage:

 - command line: `python path_to/bcdi_preprocess_BCDI.py --config path_to/config.yml`
 - directly from a code editor: update the constant CONFIG_FILE at the top of the file

    Parameters related to path names:

    :param scans: e.g. 11
     scan number or list of scan numbers
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


    Parameters used in the interactive masking GUI:

    :param flag_interact: e.g. True
     True to interact with plots, False to close it automatically
    :param background_plot: e.g. "0.5"
     background color for the GUI in level of grey in [0,1], 0 being dark. For visual
     comfort during interactive masking.

    Parameters related to data cropping/padding/centering #

    :param centering_method: e.g. "max"
     Bragg peak determination: 'max' or 'com', 'max' is better usually. It will be
     overridden by 'fix_bragg' if not empty
    :param fix_bragg: e.g. [121, 321, 256]
     Bragg peak position [z_bragg, y_bragg, x_bragg] considering the full detector.
     It is useful if hotpixels or intense aliens. Leave None otherwise.
    :param fix_size: e.g. [0, 256, 10, 240, 50, 350]
     crop the array to that predefined size considering the full detector.
     [zstart, zstop, ystart, ystop, xstart, xstop], ROI will be defaulted to [] if
     fix_size is provided. Leave None otherwise
    :param center_fft: e.g. "skip"
     how to crop/pad/center the data, available options: 'crop_sym_ZYX','crop_asym_ZYX',
     'pad_asym_Z_crop_sym_YX', 'pad_sym_Z_crop_asym_YX', 'pad_sym_Z', 'pad_asym_Z',
     'pad_sym_ZYX','pad_asym_ZYX' or 'skip'
    :param pad_size: e.g. [256, 512, 512]
     Use this to pad the array. Used in 'pad_sym_Z_crop_sym_YX', 'pad_sym_Z' and
     'pad_sym_ZYX'. Leave None otherwise.

    Parameters for data filtering

    :param mask_zero_event: e.g. False
    mask pixels where the sum along the rocking curve is zero may be dead pixels
    :param median_filter: e.g. "skip"
     which filter to apply, available filters:

     - 'median': to apply a med2filter [3,3]
     - 'interp_isolated': to interpolate isolated empty pixels based on 'medfilt_order'
       parameter
     - 'mask_isolated': mask isolated empty pixels
     - 'skip': skip filtering

    :param median_filter_order: e.g. 7
     minimum number of non-zero neighboring pixels to apply filtering

    Parameters used when reloading processed data

    :param reload_previous: e.g. False
     True to resume a previous masking (load data and mask)
    :param reload_orthogonal: e.g. False
     True if the reloaded data is already intepolated in an orthonormal frame
    :param preprocessing_binning: e.g. [1, 1, 1]
     binning factors in each dimension of the binned data to be reloaded

    Options for saving:

    :param save_rawdata: e.g. False
     True to save also the raw data when use_rawdata is False
    :param save_to_npz: e.g. True
     True to save the processed data in npz format
    :param save_to_mat: e.g. False
     True to save also in .mat format
    :param save_to_vti: e.g. False
     True to save the orthogonalized diffraction pattern to VTK file
    :param save_as_int: e.g. False
     True to save the result as an array of integers (save space)

    Parameters for the beamline:

    :param beamline: e.g. "ID01"
     name of the beamline, used for data loading and normalization by monitor
    :param actuators: e.g. {'rocking_angle': 'actuator_1_1'}
     optional dictionary that can be used to define the entries corresponding to
     actuators in data files (useful at CRISTAL where the location of data keeps
     changing, or to declare a non-standard monitor)
    :param is_series: e.g. True
     specific to series measurement at P10
    :param rocking_angle: e.g. "outofplane"
     "outofplane" for a sample rotation around x outboard, "inplane" for a sample
     rotation around y vertical up, "energy"
    :param specfile_name: e.g. "l5"
     beamline-dependent parameter, use the following template:

     - template for ID01 and 34ID-C: name of the spec file without '.spec'
     - template for SIXS: full path of the alias dictionnary or None to use the one in
       the package folder
     - template for all other beamlines: None

    Parameters for custom scans:

    :param custom_scan: e.g. False
     True for a stack of images acquired without scan, e.g. with ct in a
     macro, or when there is no spec/log file available
    :param custom_images: list of image numbers for the custom_scan, None otherwise
    :param custom_monitor: list of monitor values for normalization for the custom_scan,
     None otherwise

    Parameters for the detector:

    :param detector: e.g. "Maxipix"
     name of the detector
    :param phasing_binning: e.g. [1, 2, 2]
     binning to apply to the data (stacking dimension, detector vertical axis, detector
     horizontal axis)
    :param linearity_func: name of the linearity correction for the detector, leave None
     otherwise.
    :param x_bragg: e.g. 1577
     horizontal pixel number of the Bragg peak, used for the definition of roi_detector
     (see below). Leave None otherwise.
    :param y_bragg: e.g. 833
     vertical pixel number of the Bragg peak, used for the definition of roi_detector
     (see below). Leave None otherwise.
    :param roi_detector: e.g.[0, 250, 10, 210]
     region of interest of the detector to load. If "x_bragg" or "y_bragg" are not None,
     it will consider that the current values in roi_detector define a window around the
     Bragg peak position and the final output will be:
     [y_bragg - roi_detector[0], y_bragg + roi_detector[1],
     x_bragg - roi_detector[2], x_bragg + roi_detector[3]]. Leave None to use the full
     detector. Use with center_fft='skip' if you want this exact size for the output.
    :param normalize_flux: e.g. "monitor"
     'monitor' to normalize the intensity by the default monitor values,
     'skip' to do nothing
    :param photon_threshold: e.g. 0
     voxels with a smaller intensity will be set to 0.
    :param photon_filter: e.g. "loading"
     'loading' or 'postprocessing', when the photon threshold should be applied.
     If 'loading', it is applied before binning; if 'postprocessing', it is applied at
     the end of the script before saving
    :param bin_during_loading: e.g. False
     True to bin during loading, faster
    :param frames_pattern:  list of int, of length data.shape[0].
     If frames_pattern is 0 at index, the frame at data[index] will be skipped, if 1
     the frame will be added to the stack. Use this if you need to remove some frames
     and you know it in advance.
    :param background_file: non-empty file path or None
    :param hotpixels_file: non-empty file path or None
    :param flatfield_file: non-empty file path or None
    :param template_imagefile: e.g. "data_mpx4_%05d.edf.gz"
     use one of the following template:

     - template for ID01: 'data_mpx4_%05d.edf.gz' or 'align_eiger2M_%05d.edf.gz'
     - template for SIXS_2018: 'align.spec_ascan_mu_%05d.nxs'
     - template for SIXS_2019: 'spare_ascan_mu_%05d.nxs'
     - template for Cristal: 'S%d.nxs'
     - template for P10: '_master.h5'
     - template for NANOMAX: '%06d.h5'
     - template for 34ID: 'Sample%dC_ES_data_51_256_256.npz'

    Parameters below if you want to orthogonalize the data before phasing:

    :param use_rawdata: e.g. True
     False for using data gridded in laboratory frame, True for using data in detector
     frame
    :param interpolation_method: e.g. "xrayutilities"
     'xrayutilities' or 'linearization'
    :param fill_value_mask: e.g. 0
     0 (not masked) or 1 (masked). It will define how the pixels outside of the data
     range are processed during the interpolation. Because of the large number of masked
     pixels, phase retrieval converges better if the pixels are not masked (0 intensity
     imposed). The data is by default set to 0 outside of the defined range.
    :param beam_direction: e.g. [1, 0, 0]
     beam direction in the laboratory frame (downstream, vertical up, outboard)
    :param sample_offsets: e.g. None
     tuple of offsets in degrees of the sample for each sample circle (outer first).
     convention: the sample offsets will be subtracted to the motor values. Leave None
     if there is no offset.
    :param sdd: e.g. 0.50678
     in m, sample to detector distance in m
    :param energy: e.g. 9000
     X-ray energy in eV, it can be a number or a list in case of energy scans.
    :param custom_motors: e.g. {"mu": 0, "phi": -15.98, "chi": 90, "theta": 0,
     "delta": -0.5685, "gamma": 33.3147}
     use this to declare motor positions if there is not log file, None otherwise

    Parameters when orthogonalizing the data before phasing  using the linearized
    transformation matrix:

    :param align_q: e.g. True
     if True it rotates the crystal to align q, along one axis of the array. It is used
     only when interp_method is 'linearization'
    :param ref_axis_q: e.g. "y"  # q will be aligned along that axis
    :param outofplane_angle: e.g. 42.6093
     detector angle in deg (rotation around x outboard, typically delta), corrected for
     the direct beam position. Leave None to use the uncorrected position.
    :param inplane_angle: e.g. -0.5783
     detector angle in deg(rotation around y vertical up, typically gamma), corrected
     for the direct beam position. Leave None to use the uncorrected position.

    Parameters when orthogonalizing the data before phasing  using xrayutilities.
    xrayutilities uses the xyz crystal frame (for zero incident angle x is downstream,
    y outboard, and z vertical up):

    :param sample_inplane: e.g. [1, 0, 0]
     sample inplane reference direction along the beam at 0 angles in xrayutilities
     frame
    :param sample_outofplane: e.g. [0, 0, 1]
     surface normal of the sample at 0 angles in xrayutilities frame
    :param offset_inplane: e.g. 0
     outer detector angle offset as determined by xrayutilities area detector
     initialization
    :param cch1: e.g. 208
     direct beam vertical position in the full unbinned detector for xrayutilities 2D
     detector calibration
    :param cch2: e.g. 154
     direct beam horizontal position in the full unbinned detector for xrayutilities 2D
     detector calibration
    :param detrot: e.g. 0
     detrot parameter from xrayutilities 2D detector calibration
    :param tiltazimuth: e.g. 360
     tiltazimuth parameter from xrayutilities 2D detector calibration
    :param tilt_detector: e.g. 0
     tilt parameter from xrayutilities 2D detector calibration

"""


def close_event(event):
    """
    This function handles closing events on plots.

    :return: nothing
    """
    print(event, "Click on the figure instead of closing it!")
    sys.exit()


def run(prm):
    """
    Run the postprocessing.

    :param prm: the parsed parameters
    """

    def on_click(event):
        """
        Function to interact with a plot, return the position of clicked pixel.

        If flag_pause==1 or if the mouse is out of plot axes, it will not register
        the click.

        :param event: mouse click event
        """
        nonlocal xy, flag_pause, previous_axis

        if not event.inaxes:
            return
        if not flag_pause:

            if (previous_axis == event.inaxes) or (
                previous_axis is None
            ):  # collect points
                _x, _y = int(np.rint(event.xdata)), int(np.rint(event.ydata))
                xy.append([_x, _y])
                if previous_axis is None:
                    previous_axis = event.inaxes
            else:  # the click is not in the same subplot, restart collecting points
                print(
                    "Please select mask polygon vertices within "
                    "the same subplot: restart masking..."
                )
                xy = []
                previous_axis = None

    def press_key(event):
        """
        Interact with a plot for masking parasitic intensity or detector gaps

        :param event: button press event
        """
        nonlocal original_data, original_mask, updated_mask, data, mask, frame_index
        nonlocal flag_aliens, flag_mask, flag_pause, xy, fig_mask, max_colorbar
        nonlocal ax0, ax1, ax2, ax3, previous_axis, info_text, my_cmap, width

        try:
            if event.inaxes == ax0:
                dim = 0
                inaxes = True
            elif event.inaxes == ax1:
                dim = 1
                inaxes = True
            elif event.inaxes == ax2:
                dim = 2
                inaxes = True
            else:
                dim = -1
                inaxes = False

            if inaxes:
                if flag_aliens:
                    (
                        data,
                        mask,
                        width,
                        max_colorbar,
                        frame_index,
                        stop_masking,
                    ) = gu.update_aliens_combined(
                        key=event.key,
                        pix=int(np.rint(event.xdata)),
                        piy=int(np.rint(event.ydata)),
                        original_data=original_data,
                        original_mask=original_mask,
                        updated_data=data,
                        updated_mask=mask,
                        axes=(ax0, ax1, ax2, ax3),
                        width=width,
                        dim=dim,
                        frame_index=frame_index,
                        vmin=0,
                        vmax=max_colorbar,
                        cmap=my_cmap,
                        invert_yaxis=not use_rawdata,
                    )
                elif flag_mask:
                    if previous_axis == ax0:
                        click_dim = 0
                        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
                        points = np.stack((x.flatten(), y.flatten()), axis=0).T
                    elif previous_axis == ax1:
                        click_dim = 1
                        x, y = np.meshgrid(np.arange(nx), np.arange(nz))
                        points = np.stack((x.flatten(), y.flatten()), axis=0).T
                    elif previous_axis == ax2:
                        click_dim = 2
                        x, y = np.meshgrid(np.arange(ny), np.arange(nz))
                        points = np.stack((x.flatten(), y.flatten()), axis=0).T
                    else:
                        click_dim = None
                        points = None

                    (
                        data,
                        updated_mask,
                        flag_pause,
                        xy,
                        width,
                        max_colorbar,
                        click_dim,
                        stop_masking,
                        info_text,
                    ) = gu.update_mask_combined(
                        key=event.key,
                        pix=int(np.rint(event.xdata)),
                        piy=int(np.rint(event.ydata)),
                        original_data=original_data,
                        original_mask=mask,
                        updated_data=data,
                        updated_mask=updated_mask,
                        axes=(ax0, ax1, ax2, ax3),
                        flag_pause=flag_pause,
                        points=points,
                        xy=xy,
                        width=width,
                        dim=dim,
                        click_dim=click_dim,
                        info_text=info_text,
                        vmin=0,
                        vmax=max_colorbar,
                        cmap=my_cmap,
                        invert_yaxis=not use_rawdata,
                    )
                    if click_dim is None:
                        previous_axis = None
                else:
                    stop_masking = False

                if stop_masking:
                    plt.close("all")

        except AttributeError:  # mouse pointer out of axes
            pass

    pretty = pprint.PrettyPrinter(indent=4)

    ################################
    # assign often used parameters #
    ################################
    scans = prm["scans"]
    fix_size = prm["fix_size"]
    sample_name = prm["sample_name"]
    debug = prm["debug"]
    user_comment = prm["comment"]
    root_folder = prm["root_folder"]
    align_q = prm["align_q"]
    ref_axis_q = prm["ref_axis_q"]
    phasing_binning = prm["phasing_binning"]
    preprocessing_binning = prm["preprocessing_binning"]
    interpolation_method = prm["interpolation_method"]
    save_dir = prm["save_dir"]
    flag_interact = prm["flag_interact"]
    center_fft = prm["center_fft"]
    median_filter = prm["median_filter"]
    rocking_angle = prm["rocking_angle"]
    photon_threshold = prm["photon_threshold"]
    reload_orthogonal = prm["reload_orthogonal"]
    inplane_angle = prm["inplane_angle"]
    outofplane_angle = prm["outofplane_angle"]
    roi_detector = create_roi(dic=prm)
    use_rawdata = prm["use_rawdata"]
    normalize_flux = prm["normalize_flux"]
    #########################
    # check some parameters #
    #########################
    if len(scans) > 1 and center_fft not in [
        "crop_asymmetric_ZYX",
        "pad_Z",
        "pad_asymmetric_ZYX",
    ]:
        center_fft = "skip"
        # avoid croping the detector plane XY while centering the Bragg peak
        # otherwise outputs may have a different size,
        # which will be problematic for combining or comparing them
    if fix_size:
        print('"fix_size" parameter provided, roi_detector will be set to []')
        roi_detector = []
        print("'fix_size' parameter provided, defaulting 'center_fft' to 'skip'")
        center_fft = "skip"

    if prm["photon_filter"] == "loading":
        loading_threshold = photon_threshold
    else:
        loading_threshold = 0

    if prm["reload_previous"]:
        user_comment += "_reloaded"
    else:
        preprocessing_binning = (1, 1, 1)
        reload_orthogonal = False

    if rocking_angle == "energy":
        use_rawdata = False
        # you need to interpolate the data in QxQyQz for energy scans
        print(
            "Energy scan: defaulting use_rawdata to False,"
            " the data will be interpolated using xrayutilities"
        )

    if reload_orthogonal:
        use_rawdata = False

    if use_rawdata:
        save_dirname = "pynxraw"
        print("Output will be non orthogonal, in the detector frame")
    else:
        if interpolation_method not in {"xrayutilities", "linearization"}:
            raise ValueError(
                "Incorrect value for interp_method,"
                ' allowed values are "xrayutilities" and "linearization"'
            )
        if rocking_angle == "energy":
            interpolation_method = "xrayutilities"
            print(f"Defaulting interp_method to {interpolation_method}")
        if not reload_orthogonal and preprocessing_binning[0] != 1:
            raise ValueError(
                "preprocessing_binning along axis 0 should be 1"
                " when gridding reloaded data (angles won't match)"
            )
        save_dirname = "pynx"
        print(f"Output will be orthogonalized using {interpolation_method}")

    if isinstance(sample_name, str):
        sample_name = (sample_name,) * len(scans)

    if align_q:
        user_comment += f"_align-q-{ref_axis_q}"
        if ref_axis_q not in {"x", "y", "z"}:
            raise ValueError("ref_axis_q should be either 'x', 'y' or 'z'")
    else:
        ref_axis_q = "y"  # ref_axis_q will not be used
    axis_to_array_xyz = {
        "x": np.array([1, 0, 0]),
        "y": np.array([0, 1, 0]),
        "z": np.array([0, 0, 1]),
    }  # in xyz order

    ###################
    # define colormap #
    ###################
    colormap = gu.Colormap()
    my_cmap = colormap.cmap
    plt.rcParams["keymap.fullscreen"] = [""]

    #######################
    # Initialize detector #
    #######################
    detector = create_detector(
        name=prm["detector"],
        template_imagefile=prm["template_imagefile"],
        roi=roi_detector,
        binning=phasing_binning,
        preprocessing_binning=preprocessing_binning,
        linearity_func=prm["linearity_func"],
    )

    ####################
    # Initialize setup #
    ####################
    setup = Setup(
        beamline=prm["beamline"],
        detector=detector,
        energy=prm["energy"],
        rocking_angle=rocking_angle,
        distance=prm["sdd"],
        beam_direction=prm["beam_direction"],
        sample_inplane=prm["sample_inplane"],
        sample_outofplane=prm["sample_outofplane"],
        offset_inplane=prm["offset_inplane"],
        custom_scan=prm["custom_scan"],
        custom_images=prm["custom_images"],
        sample_offsets=prm["sample_offsets"],
        custom_monitor=prm["custom_monitor"],
        custom_motors=prm["custom_motors"],
        actuators=prm["actuators"],
        is_series=prm["is_series"],
    )

    ########################################
    # print the current setup and detector #
    ########################################
    print("\n##############\nSetup instance\n##############")
    pretty.pprint(setup.params)
    print("\n#################\nDetector instance\n#################")
    pretty.pprint(detector.params)

    ############################################
    # Initialize values for callback functions #
    ############################################
    flag_mask = False
    flag_aliens = False
    plt.rcParams["keymap.quit"] = [
        "ctrl+w",
        "cmd+w",
    ]  # this one to avoid that q closes window (matplotlib default)

    ############################
    # start looping over scans #
    ############################
    root = tk.Tk()
    root.withdraw()

    for scan_idx, scan_nb in enumerate(scans, start=1):
        plt.ion()

        comment = user_comment  # re-initialize comment
        tmp_str = f"Scan {scan_idx}/{len(scans)}: S{scan_nb}"
        print(f'\n{"#" * len(tmp_str)}\n' + tmp_str + "\n" + f'{"#" * len(tmp_str)}')

        # initialize the paths
        setup.init_paths(
            sample_name=sample_name[scan_idx - 1],
            scan_number=scan_nb,
            data_dir=prm["data_dir"],
            root_folder=root_folder,
            save_dir=save_dir,
            save_dirname=save_dirname,
            specfile_name=prm["specfile_name"],
            template_imagefile=prm["template_imagefile"],
        )

        logfile = setup.create_logfile(
            scan_number=scan_nb, root_folder=root_folder, filename=detector.specfile
        )

        if not use_rawdata:
            comment += "_ortho"
            if interpolation_method == "linearization":
                comment += "_lin"
                # load the goniometer positions needed in the calculation
                # of the transformation matrix
                (
                    tilt_angle,
                    setup.grazing_angle,
                    inplane,
                    outofplane,
                ) = setup.diffractometer.goniometer_values(
                    logfile=logfile,
                    scan_number=scan_nb,
                    setup=setup,
                )
                setup.tilt_angle = (tilt_angle[1:] - tilt_angle[0:-1]).mean()
                # override detector motor positions if the corrected values
                # (taking into account the direct beam position)
                # are provided by the user
                setup.inplane_angle = (
                    inplane_angle if inplane_angle is not None else inplane
                )
                setup.outofplane_angle = (
                    outofplane_angle if outofplane_angle is not None else outofplane
                )
            else:  # 'xrayutilities'
                comment += "_xrutil"
        if normalize_flux:
            comment = comment + "_norm"

        #############
        # Load data #
        #############
        if prm["reload_previous"]:  # resume previous masking
            print("Resuming previous masking")
            file_path = filedialog.askopenfilename(
                initialdir=detector.scandir,
                title="Select data file",
                filetypes=[("NPZ", "*.npz")],
            )
            data = np.load(file_path)
            npz_key = data.files
            data = data[npz_key[0]]
            nz, ny, nx = np.shape(data)

            # check that the ROI is correctly defined
            detector.roi = roi_detector or [0, ny, 0, nx]
            print("Detector ROI:", detector.roi)
            # update savedir to save the data in the same directory as the reloaded data
            if not save_dir:
                detector.savedir = os.path.dirname(file_path) + "/"
                print(f"Updated saving directory: {detector.savedir}")

            file_path = filedialog.askopenfilename(
                initialdir=os.path.dirname(file_path) + "/",
                title="Select mask file",
                filetypes=[("NPZ", "*.npz")],
            )
            mask = np.load(file_path)
            npz_key = mask.files
            mask = mask[npz_key[0]]

            if reload_orthogonal:
                # the data is gridded in the orthonormal laboratory frame
                use_rawdata = False
                try:
                    file_path = filedialog.askopenfilename(
                        initialdir=detector.savedir,
                        title="Select q values",
                        filetypes=[("NPZ", "*.npz")],
                    )
                    reload_qvalues = np.load(file_path)
                    q_values = [
                        reload_qvalues["qx"],
                        reload_qvalues["qz"],
                        reload_qvalues["qy"],
                    ]
                except FileNotFoundError:
                    q_values = []

                normalize_flux = (
                    "skip"  # we assume that normalization was already performed
                )
                monitor = []  # we assume that normalization was already performed
                center_fft = "skip"
                # we assume that crop/pad/centering was already performed
                fix_size = []  # we assume that crop/pad/centering was already performed

                # bin data and mask if needed
                if (
                    (detector.binning[0] != 1)
                    or (detector.binning[1] != 1)
                    or (detector.binning[2] != 1)
                ):
                    print("Binning the reloaded orthogonal data by", detector.binning)
                    data = util.bin_data(
                        data, binning=detector.binning, debugging=False
                    )
                    mask = util.bin_data(
                        mask, binning=detector.binning, debugging=False
                    )
                    mask[np.nonzero(mask)] = 1
                    if len(q_values) != 0:
                        qx = q_values[0]
                        qz = q_values[1]
                        qy = q_values[2]
                        numz, numy, numx = len(qx), len(qz), len(qy)
                        qx = qx[
                            : numz - (numz % detector.binning[0]) : detector.binning[0]
                        ]  # along z downstream
                        qz = qz[
                            : numy - (numy % detector.binning[1]) : detector.binning[1]
                        ]  # along y vertical
                        qy = qy[
                            : numx - (numx % detector.binning[2]) : detector.binning[2]
                        ]  # along x outboard
                        del numz, numy, numx
            else:  # the data is in the detector frame
                data, mask, frames_logical, monitor = bu.reload_bcdi_data(
                    logfile=logfile,
                    scan_number=scan_nb,
                    data=data,
                    mask=mask,
                    detector=detector,
                    setup=setup,
                    debugging=debug,
                    normalize=normalize_flux,
                    photon_threshold=loading_threshold,
                )

        else:  # new masking process
            reload_orthogonal = False  # the data is in the detector plane
            flatfield = util.load_flatfield(prm["flatfield_file"])
            hotpix_array = util.load_hotpixels(prm["hotpixels_file"])
            background = util.load_background(prm["background_file"])

            data, mask, frames_logical, monitor = bu.load_bcdi_data(
                logfile=logfile,
                scan_number=scan_nb,
                detector=detector,
                setup=setup,
                frames_pattern=prm["frames_pattern"],
                bin_during_loading=prm["bin_during_loading"],
                flatfield=flatfield,
                hotpixels=hotpix_array,
                background=background,
                normalize=normalize_flux,
                debugging=debug,
                photon_threshold=loading_threshold,
            )

        nz, ny, nx = np.shape(data)
        print("\nInput data shape:", nz, ny, nx)

        binning_comment = (
            f"_{detector.preprocessing_binning[0]*detector.binning[0]}"
            f"_{detector.preprocessing_binning[1]*detector.binning[1]}"
            f"_{detector.preprocessing_binning[2]*detector.binning[2]}"
        )

        if not reload_orthogonal:
            if prm["save_rawdata"]:
                np.savez_compressed(
                    detector.savedir + f"S{scan_nb}" + "_data_before_masking_stack",
                    data=data,
                )
                if prm["save_to_mat"]:
                    # save to .mat, the new order is x y z
                    # (outboard, vertical up, downstream)
                    savemat(
                        detector.savedir
                        + "S"
                        + str(scan_nb)
                        + "_data_before_masking_stack.mat",
                        {"data": np.moveaxis(data, [0, 1, 2], [-1, -2, -3])},
                    )

            if use_rawdata:
                q_values = []
                # binning along axis 0 is done after masking
                data[np.nonzero(mask)] = 0
            else:
                tmp_data = np.copy(
                    data
                )  # do not modify the raw data before the interpolation
                tmp_data[mask == 1] = 0
                fig, _, _ = gu.multislices_plot(
                    tmp_data,
                    sum_frames=True,
                    scale="log",
                    plot_colorbar=True,
                    vmin=0,
                    title="Data before gridding\n",
                    is_orthogonal=False,
                    reciprocal_space=True,
                )
                plt.savefig(
                    detector.savedir
                    + f"data_before_gridding_S{scan_nb}_{nz}_{ny}_{nx}"
                    + binning_comment
                    + ".png"
                )
                plt.close(fig)
                del tmp_data
                gc.collect()

                if interpolation_method == "xrayutilities":
                    qconv, offsets = setup.init_qconversion()
                    detector.offsets = offsets
                    hxrd = xu.experiment.HXRD(
                        prm["sample_inplane"],
                        prm["sample_outofplane"],
                        en=prm["energy"],
                        qconv=qconv,
                    )
                    # the first 2 arguments in HXRD are the inplane reference direction
                    # along the beam and surface normal of the sample

                    # Update the direct beam vertical position,
                    # take into account the roi and binning
                    cch1 = (prm["cch1"] - detector.roi[0]) / (
                        detector.preprocessing_binning[1] * detector.binning[1]
                    )
                    # Update the direct beam horizontal position,
                    # take into account the roi and binning
                    cch2 = (prm["cch2"] - detector.roi[2]) / (
                        detector.preprocessing_binning[2] * detector.binning[2]
                    )
                    # number of pixels after taking into account the roi and binning
                    nch1 = (detector.roi[1] - detector.roi[0]) // (
                        detector.preprocessing_binning[1] * detector.binning[1]
                    ) + (detector.roi[1] - detector.roi[0]) % (
                        detector.preprocessing_binning[1] * detector.binning[1]
                    )
                    nch2 = (detector.roi[3] - detector.roi[2]) // (
                        detector.preprocessing_binning[2] * detector.binning[2]
                    ) + (detector.roi[3] - detector.roi[2]) % (
                        detector.preprocessing_binning[2] * detector.binning[2]
                    )
                    # detector init_area method, pixel sizes are the binned ones
                    hxrd.Ang2Q.init_area(
                        setup.detector_ver_xrutil,
                        setup.detector_hor_xrutil,
                        cch1=cch1,
                        cch2=cch2,
                        Nch1=nch1,
                        Nch2=nch2,
                        pwidth1=detector.pixelsize_y,
                        pwidth2=detector.pixelsize_x,
                        distance=setup.distance,
                        detrot=prm["detrot"],
                        tiltazimuth=prm["tiltazimuth"],
                        tilt=prm["tilt_detector"],
                    )
                    # the first two arguments in init_area are
                    # the direction of the detector

                    data, mask, q_values, frames_logical = bu.grid_bcdi_xrayutil(
                        data=data,
                        mask=mask,
                        scan_number=scan_nb,
                        logfile=logfile,
                        detector=detector,
                        setup=setup,
                        frames_logical=frames_logical,
                        hxrd=hxrd,
                        debugging=debug,
                    )
                else:  # 'linearization'
                    # for q values, the frame used is
                    # (qx downstream, qy outboard, qz vertical up)
                    # for reference_axis, the frame is z downstream, y vertical up,
                    # x outboard but the order must be x,y,z
                    data, mask, q_values = bu.grid_bcdi_labframe(
                        data=data,
                        mask=mask,
                        detector=detector,
                        setup=setup,
                        align_q=align_q,
                        reference_axis=axis_to_array_xyz[ref_axis_q],
                        debugging=debug,
                        fill_value=(0, prm["fill_value_mask"]),
                    )
                nz, ny, nx = data.shape
                print(
                    "\nData size after interpolation into an orthonormal frame:"
                    f"{nz}, {ny}, {nx}"
                )

                # plot normalization by incident monitor for the gridded data
                if normalize_flux:
                    plt.ion()
                    tmp_data = np.copy(
                        data
                    )  # do not modify the raw data before the interpolation
                    tmp_data[tmp_data < 5] = 0  # threshold the background
                    tmp_data[mask == 1] = 0
                    fig = gu.combined_plots(
                        tuple_array=(monitor, tmp_data),
                        tuple_sum_frames=(False, True),
                        tuple_sum_axis=(0, 1),
                        tuple_width_v=None,
                        tuple_width_h=None,
                        tuple_colorbar=(False, False),
                        tuple_vmin=(np.nan, 0),
                        tuple_vmax=(np.nan, np.nan),
                        tuple_title=(
                            "monitor.min() / monitor",
                            "Gridded normed data (threshold 5)\n",
                        ),
                        tuple_scale=("linear", "log"),
                        xlabel=("Frame number", "Q$_y$"),
                        ylabel=("Counts (a.u.)", "Q$_x$"),
                        position=(323, 122),
                        is_orthogonal=not use_rawdata,
                        reciprocal_space=True,
                    )

                    fig.savefig(
                        detector.savedir
                        + f"monitor_gridded_S{scan_nb}_{nz}_{ny}_{nx}"
                        + binning_comment
                        + ".png"
                    )
                    if flag_interact:
                        fig.canvas.mpl_disconnect(
                            fig.canvas.manager.key_press_handler_id
                        )
                        cid = plt.connect("close_event", close_event)
                        fig.waitforbuttonpress()
                        plt.disconnect(cid)
                    plt.close(fig)
                    plt.ioff()
                    del tmp_data
                    gc.collect()

        ########################
        # crop/pad/center data #
        ########################
        data, mask, pad_width, q_values, frames_logical = bu.center_fft(
            data=data,
            mask=mask,
            detector=detector,
            frames_logical=frames_logical,
            centering=prm["centering_method"],
            fft_option=center_fft,
            pad_size=prm["pad_size"],
            fix_bragg=prm["fix_bragg"],
            fix_size=fix_size,
            q_values=q_values,
        )

        starting_frame = [
            pad_width[0],
            pad_width[2],
            pad_width[4],
        ]  # no need to check padded frames
        print("\nPad width:", pad_width)
        nz, ny, nx = data.shape
        print("\nData size after cropping / padding:", nz, ny, nx)

        ##########################################
        # optional masking of zero photon events #
        ##########################################
        if prm["mask_zero_event"]:
            # mask points when there is no intensity along the whole rocking curve
            # probably dead pixels
            temp_mask = np.zeros((ny, nx))
            temp_mask[np.sum(data, axis=0) == 0] = 1
            mask[np.repeat(temp_mask[np.newaxis, :, :], repeats=nz, axis=0) == 1] = 1
            del temp_mask

        ###########################################
        # save data and mask before alien removal #
        ###########################################
        fig, _, _ = gu.multislices_plot(
            data,
            sum_frames=True,
            scale="log",
            plot_colorbar=True,
            vmin=0,
            title="Data before aliens removal\n",
            is_orthogonal=not use_rawdata,
            reciprocal_space=True,
        )
        if debug:
            plt.savefig(
                detector.savedir + f"data_before_masking_sum_S{scan_nb}_{nz}_{ny}_{nx}_"
                f"{detector.binning[0]}_"
                f"{detector.binning[1]}_{detector.binning[2]}.png"
            )
        if flag_interact:
            fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
            cid = plt.connect("close_event", close_event)
            fig.waitforbuttonpress()
            plt.disconnect(cid)
        plt.close(fig)

        piz, piy, pix = np.unravel_index(data.argmax(), data.shape)
        fig = gu.combined_plots(
            (data[piz, :, :], data[:, piy, :], data[:, :, pix]),
            tuple_sum_frames=False,
            tuple_sum_axis=0,
            tuple_width_v=None,
            tuple_width_h=None,
            tuple_colorbar=True,
            tuple_vmin=0,
            tuple_vmax=np.nan,
            tuple_scale="log",
            tuple_title=("data at max in xy", "data at max in xz", "data at max in yz"),
            is_orthogonal=not use_rawdata,
            reciprocal_space=False,
        )
        if debug:
            plt.savefig(
                detector.savedir
                + f"data_before_masking_S{scan_nb}_{nz}_{ny}_{nx}_{detector.binning[0]}"
                f"_{detector.binning[1]}_{detector.binning[2]}.png"
            )
        if flag_interact:
            fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
            cid = plt.connect("close_event", close_event)
            fig.waitforbuttonpress()
            plt.disconnect(cid)
        plt.close(fig)

        fig, _, _ = gu.multislices_plot(
            mask,
            sum_frames=True,
            scale="linear",
            plot_colorbar=True,
            vmin=0,
            vmax=(nz, ny, nx),
            title="Mask before aliens removal\n",
            is_orthogonal=not use_rawdata,
            reciprocal_space=True,
        )
        if debug:
            plt.savefig(
                detector.savedir
                + f"mask_before_masking_S{scan_nb}_{nz}_{ny}_{nx}_{detector.binning[0]}"
                f"_{detector.binning[1]}_{detector.binning[2]}.png"
            )

        if flag_interact:
            fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
            cid = plt.connect("close_event", close_event)
            fig.waitforbuttonpress()
            plt.disconnect(cid)
        plt.close(fig)

        ###############################################
        # save the orthogonalized diffraction pattern #
        ###############################################
        if not use_rawdata and len(q_values) != 0:
            qx = q_values[0]
            qz = q_values[1]
            qy = q_values[2]

            if prm["save_to_vti"]:
                # save diffraction pattern to vti
                (
                    nqx,
                    nqz,
                    nqy,
                ) = (
                    data.shape
                )  # in nexus z downstream, y vertical / in q z vertical, x downstream
                print("\ndqx, dqy, dqz = ", qx[1] - qx[0], qy[1] - qy[0], qz[1] - qz[0])
                # in nexus z downstream, y vertical / in q z vertical, x downstream
                qx0 = qx.min()
                dqx = (qx.max() - qx0) / nqx
                qy0 = qy.min()
                dqy = (qy.max() - qy0) / nqy
                qz0 = qz.min()
                dqz = (qz.max() - qz0) / nqz

                gu.save_to_vti(
                    filename=os.path.join(
                        detector.savedir, f"S{scan_nb}_ortho_int" + comment + ".vti"
                    ),
                    voxel_size=(dqx, dqz, dqy),
                    tuple_array=data,
                    tuple_fieldnames="int",
                    origin=(qx0, qz0, qy0),
                )

        if flag_interact:
            plt.ioff()
            #############################################
            # remove aliens
            #############################################
            nz, ny, nx = np.shape(data)
            width = 5
            max_colorbar = 5
            flag_mask = False
            flag_aliens = True

            fig_mask, ((ax0, ax1), (ax2, ax3)) = plt.subplots(
                nrows=2, ncols=2, figsize=(12, 6)
            )
            fig_mask.canvas.mpl_disconnect(fig_mask.canvas.manager.key_press_handler_id)
            original_data = np.copy(data)
            original_mask = np.copy(mask)
            frame_index = starting_frame
            ax0.imshow(
                data[frame_index[0], :, :], vmin=0, vmax=max_colorbar, cmap=my_cmap
            )
            ax1.imshow(
                data[:, frame_index[1], :], vmin=0, vmax=max_colorbar, cmap=my_cmap
            )
            ax2.imshow(
                data[:, :, frame_index[2]], vmin=0, vmax=max_colorbar, cmap=my_cmap
            )
            ax3.set_visible(False)
            ax0.axis("scaled")
            ax1.axis("scaled")
            ax2.axis("scaled")
            if not use_rawdata:
                ax0.invert_yaxis()  # detector Y is vertical down
            ax0.set_title(f"XY - Frame {frame_index[0] + 1} / {nz}")
            ax1.set_title(f"XZ - Frame {frame_index[1] + 1} / {ny}")
            ax2.set_title(f"YZ - Frame {frame_index[2] + 1} / {nx}")
            fig_mask.text(
                0.60,
                0.30,
                "m mask ; b unmask ; u next frame ; d previous frame",
                size=12,
            )
            fig_mask.text(
                0.60,
                0.25,
                "up larger ; down smaller ; right darker ; left brighter",
                size=12,
            )
            fig_mask.text(0.60, 0.20, "p plot full image ; q quit", size=12)
            plt.tight_layout()
            plt.connect("key_press_event", press_key)
            fig_mask.set_facecolor(prm["background_plot"])
            plt.show()
            del fig_mask, original_data, original_mask
            gc.collect()

            mask[np.nonzero(mask)] = 1

            fig, _, _ = gu.multislices_plot(
                data,
                sum_frames=True,
                scale="log",
                plot_colorbar=True,
                vmin=0,
                title="Data after aliens removal\n",
                is_orthogonal=not use_rawdata,
                reciprocal_space=True,
            )

            fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
            cid = plt.connect("close_event", close_event)
            fig.waitforbuttonpress()
            plt.disconnect(cid)
            plt.close(fig)

            fig, _, _ = gu.multislices_plot(
                mask,
                sum_frames=True,
                scale="linear",
                plot_colorbar=True,
                vmin=0,
                vmax=(nz, ny, nx),
                title="Mask after aliens removal\n",
                is_orthogonal=not use_rawdata,
                reciprocal_space=True,
            )

            fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
            cid = plt.connect("close_event", close_event)
            fig.waitforbuttonpress()
            plt.disconnect(cid)
            plt.close(fig)

            #############################################
            # define mask
            #############################################
            width = 0
            max_colorbar = 5
            flag_aliens = False
            flag_mask = True
            flag_pause = False  # press x to pause for pan/zoom
            previous_axis = None
            xy = []  # list of points for mask

            fig_mask, ((ax0, ax1), (ax2, ax3)) = plt.subplots(
                nrows=2, ncols=2, figsize=(12, 6)
            )
            fig_mask.canvas.mpl_disconnect(fig_mask.canvas.manager.key_press_handler_id)
            original_data = np.copy(data)
            updated_mask = np.zeros((nz, ny, nx))
            data[mask == 1] = 0  # will appear as grey in the log plot (nan)
            ax0.imshow(
                np.log10(abs(data).sum(axis=0)), vmin=0, vmax=max_colorbar, cmap=my_cmap
            )
            ax1.imshow(
                np.log10(abs(data).sum(axis=1)), vmin=0, vmax=max_colorbar, cmap=my_cmap
            )
            ax2.imshow(
                np.log10(abs(data).sum(axis=2)), vmin=0, vmax=max_colorbar, cmap=my_cmap
            )
            ax3.set_visible(False)
            ax0.axis("scaled")
            ax1.axis("scaled")
            ax2.axis("scaled")
            if not use_rawdata:
                ax0.invert_yaxis()  # detector Y is vertical down
            ax0.set_title("XY")
            ax1.set_title("XZ")
            ax2.set_title("YZ")
            fig_mask.text(
                0.60, 0.45, "click to select the vertices of a polygon mask", size=12
            )
            fig_mask.text(
                0.60, 0.40, "x to pause/resume polygon masking for pan/zoom", size=12
            )
            fig_mask.text(0.60, 0.35, "p plot mask ; r reset current points", size=12)
            fig_mask.text(
                0.60,
                0.30,
                "m square mask ; b unmask ; right darker ; left brighter",
                size=12,
            )
            fig_mask.text(
                0.60, 0.25, "up larger masking box ; down smaller masking box", size=12
            )
            fig_mask.text(0.60, 0.20, "a restart ; q quit", size=12)
            info_text = fig_mask.text(0.60, 0.05, "masking enabled", size=16)
            plt.tight_layout()
            plt.connect("key_press_event", press_key)
            plt.connect("button_press_event", on_click)
            fig_mask.set_facecolor(prm["background_plot"])
            plt.show()

            mask[np.nonzero(updated_mask)] = 1
            data = original_data

            del fig_mask, flag_pause, flag_mask, original_data, updated_mask
            gc.collect()

        mask[np.nonzero(mask)] = 1
        data[mask == 1] = 0

        #############################################
        # mask or median filter isolated empty pixels
        #############################################
        if median_filter in {"mask_isolated", "interp_isolated"}:
            print("\nFiltering isolated pixels")
            nb_pix = 0
            for idx in range(
                pad_width[0], nz - pad_width[1]
            ):  # filter only frames whith data (not padded)
                data[idx, :, :], processed_pix, mask[idx, :, :] = util.mean_filter(
                    data=data[idx, :, :],
                    nb_neighbours=prm["median_filter_order"],
                    mask=mask[idx, :, :],
                    interpolate=median_filter,
                    min_count=3,
                    debugging=debug,
                )
                nb_pix += processed_pix
                sys.stdout.write(
                    f"\rImage {idx}, number of filtered pixels: {processed_pix}"
                )
                sys.stdout.flush()
            print("\nTotal number of filtered pixels: ", nb_pix)
        elif median_filter == "median":  # apply median filter
            print("\nApplying median filtering")
            for idx in range(
                pad_width[0], nz - pad_width[1]
            ):  # filter only frames whith data (not padded)
                data[idx, :, :] = scipy.signal.medfilt2d(data[idx, :, :], [3, 3])
        else:
            print("\nSkipping median filtering")

        ##########################
        # apply photon threshold #
        ##########################
        if photon_threshold != 0:
            mask[data < photon_threshold] = 1
            data[data < photon_threshold] = 0
            print("\nApplying photon threshold < ", photon_threshold)

        ################################################
        # check for nans and infs in the data and mask #
        ################################################
        nz, ny, nx = data.shape
        print("\nData size after masking:", nz, ny, nx)

        data, mask = util.remove_nan(data=data, mask=mask)

        data[mask == 1] = 0

        ####################
        # debugging plots  #
        ####################
        plt.ion()
        if debug:
            z0, y0, x0 = center_of_mass(data)
            fig, _, _ = gu.multislices_plot(
                data,
                sum_frames=False,
                scale="log",
                plot_colorbar=True,
                vmin=0,
                title="Masked data",
                slice_position=[int(z0), int(y0), int(x0)],
                is_orthogonal=not use_rawdata,
                reciprocal_space=True,
            )
            plt.savefig(
                detector.savedir
                + f"middle_frame_S{scan_nb}_{nz}_{ny}_{nx}_{detector.binning[0]}_"
                f"{detector.binning[1]}_{detector.binning[2]}" + comment + ".png"
            )
            if not flag_interact:
                plt.close(fig)

            fig, _, _ = gu.multislices_plot(
                data,
                sum_frames=True,
                scale="log",
                plot_colorbar=True,
                vmin=0,
                title="Masked data",
                is_orthogonal=not use_rawdata,
                reciprocal_space=True,
            )
            plt.savefig(
                detector.savedir
                + f"sum_S{scan_nb}_{nz}_{ny}_{nx}_{detector.binning[0]}_"
                f"{detector.binning[1]}_{detector.binning[2]}" + comment + ".png"
            )
            if not flag_interact:
                plt.close(fig)

            fig, _, _ = gu.multislices_plot(
                mask,
                sum_frames=True,
                scale="linear",
                plot_colorbar=True,
                vmin=0,
                vmax=(nz, ny, nx),
                title="Mask",
                is_orthogonal=not use_rawdata,
                reciprocal_space=True,
            )
            plt.savefig(
                detector.savedir + f"mask_S{scan_nb}_{nz}_{ny}_{nx}_"
                f"{detector.binning[0]}_{detector.binning[1]}_"
                f"{detector.binning[2]}" + comment + ".png"
            )
            if not flag_interact:
                plt.close(fig)

        ##################################################
        # bin the stacking axis if needed, the detector  #
        # plane was already binned when loading the data #
        ##################################################
        if (
            detector.binning[0] != 1 and not reload_orthogonal
        ):  # data was already binned for reload_orthogonal
            data = util.bin_data(data, (detector.binning[0], 1, 1), debugging=False)
            mask = util.bin_data(mask, (detector.binning[0], 1, 1), debugging=False)
            mask[np.nonzero(mask)] = 1
            if not use_rawdata and len(q_values) != 0:
                numz = len(qx)
                qx = qx[
                    : numz - (numz % detector.binning[0]) : detector.binning[0]
                ]  # along Z
                del numz
        print("\nData size after binning the stacking dimension:", data.shape)

        ##################################################################
        # final check of the shape to comply with FFT shape requirements #
        ##################################################################
        final_shape = util.smaller_primes(
            data.shape, maxprime=7, required_dividers=(2,)
        )
        com = tuple(map(lambda x: int(np.rint(x)), center_of_mass(data)))
        crop_center = pu.find_crop_center(
            array_shape=data.shape, crop_shape=final_shape, pivot=com
        )
        data = util.crop_pad(data, output_shape=final_shape, crop_center=crop_center)
        mask = util.crop_pad(mask, output_shape=final_shape, crop_center=crop_center)
        print("\nData size after considering FFT shape requirements:", data.shape)
        nz, ny, nx = data.shape
        comment = f"{comment}_{nz}_{ny}_{nx}" + binning_comment

        ############################
        # save final data and mask #
        ############################
        print("\nSaving directory:", detector.savedir)
        if prm["save_as_int"]:
            data = data.astype(int)
        print("Data type before saving:", data.dtype)
        mask[np.nonzero(mask)] = 1
        mask = mask.astype(int)
        print("Mask type before saving:", mask.dtype)
        if not use_rawdata and len(q_values) != 0:
            if prm["save_to_npz"]:
                np.savez_compressed(
                    detector.savedir + f"QxQzQy_S{scan_nb}" + comment,
                    qx=qx,
                    qz=qz,
                    qy=qy,
                )
            if prm["save_to_mat"]:
                savemat(detector.savedir + f"S{scan_nb}_qx.mat", {"qx": qx})
                savemat(detector.savedir + f"S{scan_nb}_qz.mat", {"qz": qz})
                savemat(detector.savedir + f"S{scan_nb}_qy.mat", {"qy": qy})
            max_z = data.sum(axis=0).max()
            fig, _, _ = gu.contour_slices(
                data,
                (qx, qz, qy),
                sum_frames=True,
                title="Final data",
                plot_colorbar=True,
                scale="log",
                is_orthogonal=True,
                levels=np.linspace(0, np.ceil(np.log10(max_z)), 150, endpoint=False),
                reciprocal_space=True,
            )
            fig.savefig(
                detector.savedir
                + f"final_reciprocal_space_S{scan_nb}"
                + comment
                + ".png"
            )
            plt.close(fig)

        if prm["save_to_npz"]:
            np.savez_compressed(
                detector.savedir + f"S{scan_nb}_pynx" + comment, data=data
            )
            np.savez_compressed(
                detector.savedir + f"S{scan_nb}_maskpynx" + comment, mask=mask
            )

        if prm["save_to_mat"]:
            # save to .mat, the new order is x y z (outboard, vertical up, downstream)
            savemat(
                detector.savedir + f"S{scan_nb}_data.mat",
                {"data": np.moveaxis(data.astype(np.float32), [0, 1, 2], [-1, -2, -3])},
            )
            savemat(
                detector.savedir + f"S{scan_nb}_mask.mat",
                {"data": np.moveaxis(mask.astype(np.int8), [0, 1, 2], [-1, -2, -3])},
            )

        ############################
        # plot final data and mask #
        ############################
        data[np.nonzero(mask)] = 0
        fig, _, _ = gu.multislices_plot(
            data,
            sum_frames=True,
            scale="log",
            plot_colorbar=True,
            vmin=0,
            title="Final data",
            is_orthogonal=not use_rawdata,
            reciprocal_space=True,
        )
        plt.savefig(detector.savedir + f"finalsum_S{scan_nb}" + comment + ".png")
        if not flag_interact:
            plt.close(fig)

        fig, _, _ = gu.multislices_plot(
            mask,
            sum_frames=True,
            scale="linear",
            plot_colorbar=True,
            vmin=0,
            vmax=(nz, ny, nx),
            title="Final mask",
            is_orthogonal=not use_rawdata,
            reciprocal_space=True,
        )
        plt.savefig(detector.savedir + f"finalmask_S{scan_nb}" + comment + ".png")
        if not flag_interact:
            plt.close(fig)

        del data, mask
        gc.collect()

        if len(scans) > 1:
            plt.close("all")


if __name__ == "__main__":
    # construct the argument parser and parse the command-line arguments
    ap = argparse.ArgumentParser()
    ap = add_cli_parameters(ap)
    cli_args = vars(ap.parse_args())

    # load the config file
    file = cli_args.get("config") or CONFIG_FILE
    parser = ConfigParser(CONFIG_FILE, cli_args)
    args = parser.load_arguments()
    args["time"] = f"{datetime.now()}"
    run(prm=args)

    print("\nEnd of script")
    plt.ioff()
    plt.show()
