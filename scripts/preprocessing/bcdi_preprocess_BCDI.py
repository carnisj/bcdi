#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import argparse
from datetime import datetime
import matplotlib.pyplot as plt

from bcdi.preprocessing.preprocessing_runner import run
from bcdi.utils.parser import add_cli_parameters, ConfigParser

CONFIG_FILE = "C:/Users/Jerome/Documents/myscripts/bcdi/conf/config_preprocessing.yml"

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

 - command line:
   `python path_to/bcdi_preprocess_BCDI.py --config_file path_to/config.yml`
 - directly from a code editor:
   update the constant CONFIG_FILE at the top of the file

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
    :param backend: e.g. "Qt5Agg"
     Backend used in script, change to "Agg" to make sure the figures are saved, not
     compaticle with interactive masking. Other possibilities are
     'module://matplotlib_inline.backend_inline'
     default value is "Qt5Agg"

    Parameters related to data cropping/padding/centering #

    :param centering_method: e.g. "max"
     Bragg peak determination: 'max' or 'com', 'max' is better usually. It will be
     overridden by 'bragg_peak' if not empty
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
    :param custom_images: list of image numbers for the custom_scan, None otherwise
    :param custom_monitor: list of monitor values for normalization for the custom_scan,
     None otherwise

    Parameters for the detector:

    :param detector: e.g. "Maxipix"
     name of the detector
    :param phasing_binning: e.g. [1, 2, 2]
     binning to apply to the data (stacking dimension, detector vertical axis, detector
     horizontal axis)
    :param linearity_func: e.g. [1, -2, -0.0021, 32.0, 1.232]
     coefficients of the 4th order polynomial ax^4 + bx^3 + cx^2 + dx + e which it used
     to correct the non-linearity of the detector at high intensities. Leave None
     otherwise.
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
     - template for 34ID: 'some_name_%05d.tif'

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
     X-ray energy in eV, it can be a number or a list in case of energy scans. Leave
     None to use the default from the log file.
    :param custom_motors: e.g. {"mu": 0, "phi": -15.98, "chi": 90, "theta": 0,
     "delta": -0.5685, "gamma": 33.3147}
     use this to declare motor positions if there is not log file, None otherwise

    Parameters when orthogonalizing the data before phasing  using the linearized
    transformation matrix:

    :param align_q: e.g. True
     if True it rotates the crystal to align q, along one axis of the array. It is used
     only when interp_method is 'linearization'
    :param ref_axis_q: e.g. "y"  # q will be aligned along that axis
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
