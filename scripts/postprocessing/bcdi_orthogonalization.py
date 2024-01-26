#!/usr/bin/env python3

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr
#         Clement Atlan, c.atlan@outlook.com

import argparse
import logging
from datetime import datetime
from pathlib import Path

from matplotlib import pyplot as plt

from bcdi.postprocessing.postprocessing_runner import run
from bcdi.utils.parser import ConfigParser, add_cli_parameters
from bcdi.utils.snippets_logging import configure_logging

here = Path(__file__).parent
CONFIG_FILE = str(here.parents[1] / "bcdi/examples/S11_config_postprocessing.yml")

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

    :param scans: e.g. 11
     scan number or list of scan numbers
    :param root_folder: e.g. "C:/Users/Jerome/Documents/data/dataset_ID01/"
     folder of the experiment, where all scans are stored
    :param save_dir: e.g. "C:/Users/Jerome/Documents/data/dataset_ID01/test/"
     images will be saved here, leave it to None otherwise. Provide a single path or a
     list of paths for multiple scans.
    :param data_dir: e.g. None
     use this to override the beamline default search path for the data. Provide
     a list of paths for multiple scans.
    :param sample_name: e.g. "S"
     str or list of str of sample names (string in front of the scan number in the
     folder name). If only one name is indicated, it will be repeated to match the
     number of scans.
    :param comment: string use in filenames when saving
    :param debug: e.g. False
     True to see plots
    :param colormap: e.g. "turbo"
     "turbo", "custom" or colormap defined in the colorcet package, see
     https://colorcet.holoviz.org/
    :param reconstruction_files: e.g. "C:/Users/Jerome/Documents/data/modes.h5"
     full path to the output of phase retrieval, or list of such paths if 'scans' is a
     list. Providing several reconstructions for each scan is not supported.
     If None an interactive window will open to choose a file.

    Parameters used in the interactive masking GUI:

    :param backend: e.g. "Qt5Agg"
     Backend used in script, change to "Agg" if you want to save the figures without
     showing them. Other possibilities are "Qt5Agg" (default) and
     "module://matplotlib_inline.backend_inline"

    Parameters used when averaging several reconstruction:

    :param sort_method: e.g. "variance/mean"
     'mean_amplitude' or 'variance' or 'variance/mean' or 'volume', metric for averaging
    :param averaging_space: e.g. "reciprocal_space"
     in which space to average, 'direct_space' or 'reciprocal_space'
    :param correlation_threshold: e.g. 0.90
     minimum correlation between two arrays to average them

    Parameters related to centering:

    :param centering_method: e.g. {"direct_space": "max_com", "reciprocal_space": "max"}
     dictionary with the centering methods for direct and reciprocal space. The methods
     supported are 'com' (center of mass), 'max', 'max_com' (max along the first axis,
     center of mass in the detector plane), 'skip'. If a simple string is provided, it
     will use that method for both direct and reciprocal space.
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
    :plot_margin: e.g. 30
     margin to add on each side of the smallest box around the crystal. Use this to
     increase the box size in case the interpolated crystal is cropped.
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
    :param skip_unwrap: e.g. False
     If 'skip_unwrap', it will not unwrap the phase. It can be used when there is a
     defect and phase unwrapping does not work well.
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
     string or list of strings for multiple scans. beamline-dependent parameter,
     use the following template:

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
     string or list of strings for multiple scans. beamline-dependent parameter,
     use the following template:

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


if __name__ == "__main__":
    now = datetime.now()
    configure_logging()
    logging.info(f"Start script at {now}")

    # construct the argument parser and parse the command-line arguments
    ap = argparse.ArgumentParser()
    ap = add_cli_parameters(ap)
    cli_args = vars(ap.parse_args())

    # load the config file
    file = cli_args.get("config_file") or CONFIG_FILE
    logging.info(f"Loading config file '{file}'")

    parser = ConfigParser(file, cli_args)
    args = parser.load_arguments()
    args["time"] = f"{now}"
    run(prm=args, procedure="orthogonalization")

    now = datetime.now()
    logging.info(f"End of script at {now}")
    plt.ioff()
    plt.show()
