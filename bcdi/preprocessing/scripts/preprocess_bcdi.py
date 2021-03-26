# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

try:
    import hdf5plugin  # for P10, should be imported before h5py or PyTables
except ModuleNotFoundError:
    pass
import xrayutilities as xu
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend("Qt5Agg")  # "Qt5Agg" or "Qt4Agg" depending on the version of Qt installer, bug with Tk
import os
import scipy.signal  # for medfilt2d
from scipy.ndimage.measurements import center_of_mass
import sys
from scipy.io import savemat
import tkinter as tk
from tkinter import filedialog
import gc
sys.path.append('D:/myscripts/bcdi/')
import bcdi.graph.graph_utils as gu
import bcdi.experiment.experiment_utils as exp
import bcdi.postprocessing.postprocessing_utils as pu
import bcdi.preprocessing.preprocessing_utils as pru
import bcdi.utils.utilities as util
import bcdi.utils.validation as valid


helptext = """
Prepare experimental data for Bragg CDI phasing: crop/pad, center, mask, normalize and filter the data.

Beamlines currently supported: ESRF ID01, SOLEIL CRISTAL, SOLEIL SIXS, PETRAIII P10 and APS 34ID-C.

Output: data and mask as numpy .npz or Matlab .mat 3D arrays for phasing

File structure should be (e.g. scan 1):
specfile, hotpixels file and flatfield file in:    /rootdir/
data in:                                           /rootdir/S1/data/

output files saved in:   /rootdir/S1/pynxraw/ or /rootdir/S1/pynx/ depending on 'use_rawdata' option
"""

scans = 78  # np.arange(1401, 1419+1, 3)  # scan number or list of scan numbers
# scans = np.concatenate((scans, np.arange(1147, 1195+1, 3)))
# bad_indices = np.argwhere(scans == 738)
# scans = np.delete(scans, bad_indices)

root_folder = "D:/data/CRISTAL_March2021/"  # folder of the experiment, where all scans are stored
save_dir = None  # root_folder + '/dataset_2_pearson97.5_newpsf/result/diffpattern_from_reconstruction/'  # images will be saved here, leave it to None otherwise
data_dirname = None  # leave None to use the beamline default, '' empty string when there is no subfolder
# (data directly in the scan folder), or a non-empty string for the subfolder name
# (default to scan_folder/pynx/ or scan_folder/pynxraw/ depending on the setting of use_rawdata)
sample_name = "S"  # str or list of str of sample names (string in front of the scan number in the folder name).
# If only one name is indicated, it will be repeated to match the number of scans.
user_comment = ''  # string, should start with "_"
debug = False  # set to True to see plots
binning = (1, 1, 1)  # binning to apply to the data
# (stacking dimension, detector vertical axis, detector horizontal axis)
##############################
# parameters used in masking #
##############################
flag_interact = True  # True to interact with plots, False to close it automatically
background_plot = '0.5'  # in level of grey in [0,1], 0 being dark. For visual comfort during masking
#########################################################
# parameters related to data cropping/padding/centering #
#########################################################
centering = 'max'  # Bragg peak determination: 'max' or 'com', 'max' is better usually.
#  It will be overridden by 'fix_bragg' if not empty
fix_bragg = []  # fix the Bragg peak position [z_bragg, y_bragg, x_bragg] considering the full detector
# It is useful if hotpixels or intense aliens. Leave it [] otherwise.
fix_size = []  # crop the array to predefined size considering the full detector,
# leave it to [] otherwise [zstart, zstop, ystart, ystop, xstart, xstop]. ROI will be defaulted to []
center_fft = 'skip'
# 'crop_sym_ZYX','crop_asym_ZYX','pad_asym_Z_crop_sym_YX', 'pad_sym_Z_crop_asym_YX',
# 'pad_sym_Z', 'pad_asym_Z', 'pad_sym_ZYX','pad_asym_ZYX' or 'skip'
pad_size = []  # size after padding, e.g. [256, 512, 512]. Use this to pad the array.
# used in 'pad_sym_Z_crop_sym_YX', 'pad_sym_Z', 'pad_sym_ZYX'
##############################################
# parameters used in intensity normalization #
##############################################
normalize_flux = 'monitor'  # 'skip'  # 'monitor' to normalize the intensity by the default monitor values, 'skip' to do nothing
#################################
# parameters for data filtering #
#################################
mask_zero_event = False  # mask pixels where the sum along the rocking curve is zero - may be dead pixels
flag_medianfilter = 'interp_isolated'
# set to 'median' for applying med2filter [3,3]
# set to 'interp_isolated' to interpolate isolated empty pixels based on 'medfilt_order' parameter
# set to 'mask_isolated' it will mask isolated empty pixels
# set to 'skip' will skip filtering
medfilt_order = 7    # for custom median filter, number of pixels with intensity surrounding the empty pixel
#################################################
# parameters used when reloading processed data #
#################################################
reload_previous = False  # True to resume a previous masking (load data and mask)
reload_orthogonal = False  # True if the reloaded data is already intepolated in an orthonormal frame
preprocessing_binning = (1, 1, 1)  # binning factors in each dimension of the binned data to be reloaded
##################
# saving options #
##################
save_rawdata = False  # save also the raw data when use_rawdata is False
save_to_npz = True  # True to save the processed data in npz format
save_to_mat = False  # True to save also in .mat format
save_to_vti = False  # save the orthogonalized diffraction pattern to VTK file
save_asint = False  # if True, the result will be saved as an array of integers (save space)
######################################
# define beamline related parameters #
######################################
beamline = 'CRISTAL'  # name of the beamline, used for data loading and normalization by monitor
# supported beamlines: 'ID01', 'SIXS_2018', 'SIXS_2019', 'CRISTAL', 'P10', 'NANOMAX', '34ID'
actuators = {'rocking_angle': 'actuator_1_3'}
# Optional dictionary that can be used to define the entries corresponding to actuators in data files
# (useful at CRISTAL where the location of data keeps changing)
# e.g.  {'rocking_angle': 'actuator_1_3', 'detector': 'data_04', 'monitor': 'data_05'}
is_series = True  # specific to series measurement at P10

custom_scan = False  # set it to True for a stack of images acquired without scan, e.g. with ct in a macro, or when
# there is no spec/log file available
custom_images = [3]  # np.arange(11353, 11453, 1)  # list of image numbers for the custom_scan, None otherwise
custom_monitor = np.ones(51)  # monitor values for normalization for the custom_scan, None otherwise

rocking_angle = "inplane"  # "outofplane" for a sample rotation around x outboard, "inplane" for a sample rotation
# around y vertical up, "energy"

follow_bragg = False  # only for energy scans, set to True if the detector was also scanned to follow the Bragg peak
specfile_name = ''
# template for ID01: name of the spec file without '.spec'
# template for SIXS: full path of the alias dictionnary or None to use the one in the package folder
# template for all other beamlines: ''
###############################
# detector related parameters #
###############################
detector = "Maxipix"    # "Eiger2M", "Maxipix", "Eiger4M", "Merlin" or "Timepix"
linearity_func = None  # lambda array_1d: (array_1d*(7.484e-22*array_1d**4 - 3.447e-16*array_1d**3 +
# 5.067e-11*array_1d**2 - 6.022e-07*array_1d + 0.889)) # MIR
# np.divide(array_1d, (1-array_1d*1.3e-6))  # Sarah_1
# (array_1d*(7.484e-22*array_1d**4 - 3.447e-16*array_1d**3 + 5.067e-11*array_1d**2 - 6.022e-07*array_1d + 0.889)) # MIR
# linearity correction for the detector, leave None otherwise.
# You can use def instead of a lambda expression but the input array should be 1d (flattened 2D detector array).
x_bragg = 184  # horizontal pixel number of the Bragg peak, can be used for the definition of the ROI
y_bragg = 368  # vertical pixel number of the Bragg peak, can be used for the definition of the ROI
roi_detector = [y_bragg - 128, y_bragg + 128, x_bragg - 180, x_bragg + 180]  #
# [Vstart, Vstop, Hstart, Hstop]
# leave None to use the full detector. Use with center_fft='skip' if you want this exact size.
photon_threshold = 0  # data[data < photon_threshold] = 0
photon_filter = 'loading'  # 'loading' or 'postprocessing', when the photon threshold should be applied
# if 'loading', it is applied before binning; if 'postprocessing', it is applied at the end of the script before saving
background_file = None  # root_folder + 'background.npz'  # non empty file path or None
hotpixels_file = root_folder + 'hotpixels_cristal.npz'  # root_folder + 'mask_merlin.npy'  # non empty file path or None
flatfield_file = None  # root_folder + "flatfield_maxipix_8kev.npz"  # non empty file path or None
template_imagefile = 'mgphi-2021_%04d.nxs'
# template for ID01: 'data_mpx4_%05d.edf.gz' or 'align_eiger2M_%05d.edf.gz'
# template for SIXS_2018: 'align.spec_ascan_mu_%05d.nxs'
# template for SIXS_2019: 'spare_ascan_mu_%05d.nxs'
# template for Cristal: 'S%d.nxs'
# template for P10: '_master.h5'
# template for NANOMAX: '%06d.h5'
# template for 34ID: 'Sample%dC_ES_data_51_256_256.npz'
nb_pixel_x = None  # fix to declare a known detector but with less pixels (e.g. one tile HS), leave None otherwise
nb_pixel_y = None  # fix to declare a known detector but with less pixels (e.g. one tile HS), leave None otherwise
################################################################################
# define parameters below if you want to orthogonalize the data before phasing #
################################################################################
use_rawdata = False  # False for using data gridded in laboratory frame/ True for using data in detector frame
interp_method = 'xrayutilities'  # 'xrayutilities' or 'linearization'
beam_direction = (1, 0, 0)  # beam direction in the laboratory frame (downstream, vertical up, outboard)
sample_offsets = (0, 0, 0)  # -6*0.007821)  # tuple of offsets in degrees of the sample around (downstream, vertical up, outboard)
# convention: the sample offsets will be subtracted to the motor values
sdd = 0.914  # in m, sample to detector distance in m
energy = 8500  # np.linspace(11100, 10900, num=51)  # x-ray energy in eV
custom_motors = None  # {"mu": 0, "phi": -15.98, "chi": 90, "theta": 0, "delta": -0.5685, "gamma": 33.3147}
# use this to declare motor positions if there is not log file, None otherwise
# example: {"eta": np.linspace(16.989, 18.989, num=100, endpoint=False), "phi": 0, "nu": -0.75, "delta": 36.65}
# ID01: eta, chi, phi, nu, delta
# CRISTAL: mgomega, gamma, delta
# SIXS: beta, mu, gamma, delta
# P10: om, phi, chi, mu, gamma, delta
# NANOMAX: theta, phi, gamma, delta, energy, radius
# 34ID: mu, phi (incident angle), chi, theta (inplane), delta (inplane), gamma (outofplane)
#######################################################################################################
# parameters when orthogonalizing the data before phasing  using the linearized transformation matrix #
#######################################################################################################
align_q = True  # used only when interp_method is 'linearization', if True it rotates the crystal to align q
# along one axis of the array
ref_axis_q = "y"  # q will be aligned along that axis
outofplane_angle = None  # detector angle in deg (rotation around x outboard, typically delta),
# corrected for the direct beam position. Leave None to use the uncorrected position.
inplane_angle = None  # detector angle in deg(rotation around y vertical up, typically gamma),
# corrected for the direct beam position. Leave None to use the uncorrected position.
################################################################################
# parameters when orthogonalizing the data before phasing  using xrayutilities #
################################################################################
# xrayutilities uses the xyz crystal frame: for incident angle = 0, x is downstream, y outboard, and z vertical up
sample_inplane = (1, 0, 0)  # sample inplane reference direction along the beam at 0 angles in xrayutilities frame
sample_outofplane = (0, 0, 1)  # surface normal of the sample at 0 angles in xrayutilities frame
offset_inplane = 0  # outer detector angle offset as determined by xrayutilities area detector initialization
cch1 = 183  # direct beam vertical position in the full unbinned detector for xrayutilities 2D detector calibration
cch2 = 171  # direct beam horizontal position in the full unbinned detector for xrayutilities 2D detector calibration
detrot = 0  # detrot parameter from xrayutilities 2D detector calibration
tiltazimuth = 360  # tiltazimuth parameter from xrayutilities 2D detector calibration
tilt = 0  # tilt parameter from xrayutilities 2D detector calibration
##################################
# end of user-defined parameters #
##################################


def close_event(event):
    """
    This function handles closing events on plots.

    :return: nothing
    """
    print(event, 'Click on the figure instead of closing it!')
    sys.exit()


def on_click(event):
    """
    Function to interact with a plot, return the position of clicked pixel. If flag_pause==1 or
    if the mouse is out of plot axes, it will not register the click

    :param event: mouse click event
    """
    global xy, flag_pause, previous_axis
    if not event.inaxes:
        return
    if not flag_pause:

        if (previous_axis == event.inaxes) or (previous_axis is None):  # collect points
            _x, _y = int(np.rint(event.xdata)), int(np.rint(event.ydata))
            xy.append([_x, _y])
            if previous_axis is None:
                previous_axis = event.inaxes
        else:  # the click is not in the same subplot, restart collecting points
            print('Please select mask polygon vertices within the same subplot: restart masking...')
            xy = []
            previous_axis = None


def press_key(event):
    """
    Interact with a plot for masking parasitic diffraction intensity or detector gaps

    :param event: button press event
    """
    global original_data, original_mask, updated_mask, data, mask, frame_index, width, flag_aliens, flag_mask
    global flag_pause, xy, fig_mask, max_colorbar, ax0, ax1, ax2, ax3, previous_axis, info_text, my_cmap

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
                data, mask, width, max_colorbar, frame_index, stop_masking = \
                    gu.update_aliens_combined(key=event.key, pix=int(np.rint(event.xdata)),
                                              piy=int(np.rint(event.ydata)), original_data=original_data,
                                              original_mask=original_mask, updated_data=data, updated_mask=mask,
                                              axes=(ax0, ax1, ax2, ax3), width=width, dim=dim, frame_index=frame_index,
                                              vmin=0, vmax=max_colorbar, cmap=my_cmap, invert_yaxis=not use_rawdata)
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

                data, updated_mask, flag_pause, xy, width, max_colorbar, click_dim, stop_masking, info_text = \
                    gu.update_mask_combined(key=event.key, pix=int(np.rint(event.xdata)),
                                            piy=int(np.rint(event.ydata)), original_data=original_data,
                                            original_mask=mask, updated_data=data, updated_mask=updated_mask,
                                            axes=(ax0, ax1, ax2, ax3), flag_pause=flag_pause, points=points,
                                            xy=xy, width=width, dim=dim, click_dim=click_dim, info_text=info_text,
                                            vmin=0, vmax=max_colorbar, cmap=my_cmap, invert_yaxis=not use_rawdata)
                if click_dim is None:
                    previous_axis = None
            else:
                stop_masking = False

            if stop_masking:
                plt.close('all')

    except AttributeError:  # mouse pointer out of axes
        pass


#########################
# check some parameters #
#########################
if isinstance(scans, int):
    scans = (scans,)

if len(scans) > 1:
    if center_fft not in ['crop_asymmetric_ZYX', 'pad_Z', 'pad_asymmetric_ZYX']:
        center_fft = 'skip'
        # avoid croping the detector plane XY while centering the Bragg peak
        # otherwise outputs may have a different size, which will be problematic for combining or comparing them
if len(fix_size) != 0:
    print('"fix_size" parameter provided, roi_detector will be set to []')
    roi_detector = []
    print("'fix_size' parameter provided, defaulting 'center_fft' to 'skip'")
    center_fft = 'skip'

if photon_filter == 'loading':
    loading_threshold = photon_threshold
else:
    loading_threshold = 0

create_savedir = True

valid.valid_container(user_comment, container_types=str, name='preprocess_bcdi')
if len(user_comment) != 0 and not user_comment.startswith('_'):
    user_comment = '_' + user_comment
if reload_previous:
    user_comment += '_reloaded'
else:
    preprocessing_binning = (1, 1, 1)
    reload_orthogonal = False

if rocking_angle == "energy":
    use_rawdata = False  # you need to interpolate the data in QxQyQz for energy scans
    print("Energy scan: defaulting use_rawdata to False, the data will be interpolated using xrayutilities")

if reload_orthogonal:
    use_rawdata = False

if use_rawdata:
    save_dirname = 'pynxraw'
    print('Output will be non orthogonal, in the detector frame')
    plot_title = ['YZ', 'XZ', 'XY']
else:
    save_dirname = 'pynx'
    print('Output will be orthogonalized by xrayutilities')
    plot_title = ['QzQx', 'QyQx', 'QyQz']
    if interp_method not in {'xrayutilities', 'linearization'}:
        raise ValueError('Incorrect value for interp_method, allowed values are "xrayutilities" and "linearization"')
    if rocking_angle == 'energy':
        interp_method = 'xrayutilities'
        print(f'Defaulting interp_method to {interp_method}')
    if not reload_orthogonal and preprocessing_binning[0] != 1:
        raise ValueError('preprocessing_binning along axis 0 should be 1 when gridding reloaded data'
                         ' (angles won\'t match)')

if isinstance(sample_name, str):
    sample_name = [sample_name for idx in range(len(scans))]
valid.valid_container(sample_name, container_types=(tuple, list), length=len(scans), item_types=str,
                      name='preprocess_bcdi')

###################
# define colormap #
###################
colormap = gu.Colormap()
my_cmap = colormap.cmap
plt.rcParams["keymap.fullscreen"] = [""]

#######################
# Initialize detector #
#######################
kwargs = dict()  # create dictionnary
kwargs['is_series'] = is_series
kwargs['preprocessing_binning'] = preprocessing_binning
kwargs['nb_pixel_x'] = nb_pixel_x  # fix to declare a known detector but with less pixels (e.g. one tile HS)
kwargs['nb_pixel_y'] = nb_pixel_y  # fix to declare a known detector but with less pixels (e.g. one tile HS)
kwargs['linearity_func'] = linearity_func
detector = exp.Detector(name=detector, template_imagefile=template_imagefile, roi=roi_detector,
                        binning=binning, **kwargs)

####################
# Initialize setup #
####################
setup = exp.Setup(beamline=beamline, energy=energy, rocking_angle=rocking_angle, distance=sdd,
                  beam_direction=beam_direction, sample_inplane=sample_inplane,
                  sample_outofplane=sample_outofplane, offset_inplane=offset_inplane,
                  custom_scan=custom_scan, custom_images=custom_images, sample_offsets=sample_offsets,
                  custom_monitor=custom_monitor, custom_motors=custom_motors,
                  pixel_x=detector.pixelsize_x, pixel_y=detector.pixelsize_y, actuators=actuators)

########################################
# print the current setup and detector #
########################################
print('\n##############\nSetup instance\n##############')
print(setup)
print('\n#################\nDetector instance\n#################')
print(detector)

############################################
# Initialize values for callback functions #
############################################
flag_mask = False
flag_aliens = False
plt.rcParams["keymap.quit"] = ["ctrl+w", "cmd+w"]  # this one to avoid that q closes window (matplotlib default)

############################
# start looping over scans #
############################
root = tk.Tk()
root.withdraw()

for scan_idx, scan_nb in enumerate(scans, start=1):
    plt.ion()

    comment = user_comment  # re-initialize comment
    tmp_str = f'Scan {scan_idx}/{len(scans)}: S{scan_nb}'
    print(f'\n{"#" * len(tmp_str)}\n' + tmp_str + '\n' + f'{"#" * len(tmp_str)}')

    # initialize the paths
    setup.init_paths(detector=detector, sample_name=sample_name[scan_idx-1], scan_number=scan_nb,
                     data_dirname=data_dirname, root_folder=root_folder, save_dir=save_dir, save_dirname=save_dirname,
                     verbose=True, create_savedir=create_savedir, specfile_name=specfile_name,
                     template_imagefile=template_imagefile)

    logfile = pru.create_logfile(setup=setup, detector=detector, scan_number=scan_nb,
                                 root_folder=root_folder, filename=detector.specfile)

    if not use_rawdata:
        comment += '_ortho'
        if interp_method == 'linearization':
            comment += '_lin'
            # load the goniometer positions needed in the calculation of the transformation matrix
            tilt_angle, setup.grazing_angle, inplane, outofplane =\
                pru.goniometer_values(logfile=logfile, scan_number=scan_nb, setup=setup, follow_bragg=follow_bragg)
            setup.tilt_angle = tilt_angle[1] - tilt_angle[0]
            # override detector motor positions if the corrected values (taking into account the direct beam position)
            # are provided by the user
            setup.inplane_angle = inplane_angle if inplane_angle is not None else inplane
            setup.outofplane_angle = outofplane_angle if outofplane_angle is not None else outofplane
        else:  # 'xrayutilities'
            comment += '_xrutil'
    if normalize_flux:
        comment = comment + '_norm'

    #############
    # Load data #
    #############
    if reload_previous:  # resume previous masking
        print('Resuming previous masking')
        file_path = filedialog.askopenfilename(initialdir=detector.scandir, title="Select data file",
                                               filetypes=[("NPZ", "*.npz")])
        data = np.load(file_path)
        npz_key = data.files
        data = data[npz_key[0]]
        nz, ny, nx = np.shape(data)
        # from numpy.fft import fftn, fftshift
        # print('data.shape before padding', data.shape)
        # # first dataset (168, 1024, 800)
        # data = pu.crop_pad(array=data, output_shape=(224, 1024, 800), debugging=False)
        # data = fftshift(fftn(data)) / (np.sqrt(224) * np.sqrt(1024) * np.sqrt(800))  # complex amplitude
        # # second dataset (168, 1024, 800)
        # data = abs(np.multiply(data, np.conjugate(data)))  # intensity
        # # data = data * 0.2740**2  # dataset_1_newpsf   due to mode decomposition
        # # data = data * 0.2806**2  # dataset_1_nopsf   due to mode decomposition
        # data = data * 0.2744**2  # dataset_2_pearson97.5_newpsf   due to mode decomposition
        # check that the ROI is correctly defined
        detector.roi = roi_detector or [0, ny, 0, nx]
        print('Detector ROI:', detector.roi)
        # update savedir to save the data in the same directory as the reloaded data
        if not save_dir:
            detector.savedir = os.path.dirname(file_path) + '/'
            print(f'Updated saving directory: {detector.savedir}')

        file_path = filedialog.askopenfilename(initialdir=os.path.dirname(file_path) + '/', title="Select mask file",
                                               filetypes=[("NPZ", "*.npz")])
        mask = np.load(file_path)
        npz_key = mask.files
        mask = mask[npz_key[0]]

        if reload_orthogonal:  # the data is gridded in the orthonormal laboratory frame
            use_rawdata = False
            try:
                file_path = filedialog.askopenfilename(initialdir=detector.savedir, title="Select q values",
                                                       filetypes=[("NPZ", "*.npz")])
                reload_qvalues = np.load(file_path)
                q_values = [reload_qvalues['qx'], reload_qvalues['qz'], reload_qvalues['qy']]
            except FileNotFoundError:
                q_values = []

            normalize_flux = 'skip'  # we assume that normalization was already performed
            monitor = []  # we assume that normalization was already performed
            center_fft = 'skip'  # we assume that crop/pad/centering was already performed
            fix_size = []  # we assume that crop/pad/centering was already performed

            # bin data and mask if needed
            if (detector.binning[0] != 1) or (detector.binning[1] != 1) or (detector.binning[2] != 1):
                print('Binning the reloaded orthogonal data by', detector.binning)
                data = pu.bin_data(data, binning=detector.binning, debugging=False)
                mask = pu.bin_data(mask, binning=detector.binning, debugging=False)
                mask[np.nonzero(mask)] = 1
                if len(q_values) != 0:
                    qx = q_values[0]
                    qz = q_values[1]
                    qy = q_values[2]
                    numz, numy, numx = len(qx), len(qz), len(qy)
                    qx = qx[:numz - (numz % detector.binning[0]):detector.binning[0]]  # along z downstream
                    qz = qz[:numy - (numy % detector.binning[1]):detector.binning[1]]  # along y vertical
                    qy = qy[:numx - (numx % detector.binning[2]):detector.binning[2]]  # along x outboard
                    del numz, numy, numx
        else:  # the data is in the detector frame
            data, mask, frames_logical, monitor = pru.reload_bcdi_data(logfile=logfile, scan_number=scan_nb,
                                                                       data=data, mask=mask, detector=detector,
                                                                       setup=setup, debugging=debug,
                                                                       normalize=normalize_flux,
                                                                       photon_threshold=loading_threshold)

    else:  # new masking process
        reload_orthogonal = False  # the data is in the detector plane
        flatfield = pru.load_flatfield(flatfield_file)
        hotpix_array = pru.load_hotpixels(hotpixels_file)
        background = pru.load_background(background_file)

        data, mask, frames_logical, monitor = pru.load_bcdi_data(logfile=logfile, scan_number=scan_nb,
                                                                 detector=detector, setup=setup,
                                                                 flatfield=flatfield, hotpixels=hotpix_array,
                                                                 background=background, normalize=normalize_flux,
                                                                 debugging=debug, photon_threshold=loading_threshold)

    nz, ny, nx = np.shape(data)
    print('\nInput data shape:', nz, ny, nx)

    binning_comment = (f'_{detector.preprocessing_binning[0]*detector.binning[0]}'
                       f'_{detector.preprocessing_binning[1]*detector.binning[1]}'
                       f'_{detector.preprocessing_binning[2]*detector.binning[2]}')

    if not reload_orthogonal:
        if save_rawdata:
            np.savez_compressed(detector.savedir+'S'+str(scan_nb)+'_data_before_masking_stack', data=data)
            if save_to_mat:
                # save to .mat, the new order is x y z (outboard, vertical up, downstream)
                savemat(detector.savedir+'S'+str(scan_nb)+'_data_before_masking_stack.mat',
                        {'data': np.moveaxis(data, [0, 1, 2], [-1, -2, -3])})

        if use_rawdata:
            q_values = []
            # binning along axis 0 is done after masking
            data[np.nonzero(mask)] = 0
        else:
            tmp_data = np.copy(data)  # do not modify the raw data before the interpolation
            tmp_data[mask == 1] = 0
            fig, _, _ = gu.multislices_plot(tmp_data, sum_frames=True, scale='log', plot_colorbar=True, vmin=0,
                                            title='Data before gridding\n', is_orthogonal=False, reciprocal_space=True)
            plt.savefig(detector.savedir + f'data_before_gridding_S{scan_nb}_{nz}_{ny}_{nx}' + binning_comment + '.png')
            plt.close(fig)
            del tmp_data
            gc.collect()

            if interp_method == 'xrayutilities':
                qconv, offsets = pru.init_qconversion(setup)
                detector.offsets = offsets
                hxrd = xu.experiment.HXRD(sample_inplane, sample_outofplane, en=energy, qconv=qconv)
                # the first 2 arguments in HXRD are the inplane reference direction along the beam and surface normal
                # of the sample

                # Update the direct beam vertical position, take into account the roi and binning
                cch1 = (cch1 - detector.roi[0]) / (detector.preprocessing_binning[1] * detector.binning[1])
                # Update the direct beam horizontal position, take into account the roi and binning
                cch2 = (cch2 - detector.roi[2]) / (detector.preprocessing_binning[2] * detector.binning[2])
                # number of pixels after taking into account the roi and binning
                nch1 = (detector.roi[1] - detector.roi[0]) // (
                            detector.preprocessing_binning[1] * detector.binning[1]) + \
                       (detector.roi[1] - detector.roi[0]) % (detector.preprocessing_binning[1] * detector.binning[1])
                nch2 = (detector.roi[3] - detector.roi[2]) // (
                            detector.preprocessing_binning[2] * detector.binning[2]) + \
                       (detector.roi[3] - detector.roi[2]) % (detector.preprocessing_binning[2] * detector.binning[2])
                # detector init_area method, pixel sizes are the binned ones
                hxrd.Ang2Q.init_area(setup.detector_ver, setup.detector_hor, cch1=cch1, cch2=cch2,
                                     Nch1=nch1, Nch2=nch2, pwidth1=detector.pixelsize_y, pwidth2=detector.pixelsize_x,
                                     distance=setup.distance, detrot=detrot, tiltazimuth=tiltazimuth, tilt=tilt)
                # first two arguments in init_area are the direction of the detector, checked for ID01 and SIXS

                data, mask, q_values, frames_logical = \
                    pru.grid_bcdi_xrayutil(data=data, mask=mask, scan_number=scan_nb, logfile=logfile,
                                           detector=detector, setup=setup, frames_logical=frames_logical, hxrd=hxrd,
                                           follow_bragg=follow_bragg, debugging=debug)
            else:  # 'linearization'
                data, mask, q_values = \
                    pru.grid_bcdi_labframe(data=data, mask=mask, detector=detector, setup=setup,
                                           debugging=debug, follow_bragg=follow_bragg, method_shape='fix_sampling')
            nz, ny, nx = data.shape
            print('\nData size after interpolation into an orthonormal frame:', nz, ny, nx)

            # plot normalization by incident monitor for the gridded data
            if normalize_flux:
                plt.ion()
                tmp_data = np.copy(data)  # do not modify the raw data before the interpolation
                tmp_data[tmp_data < 5] = 0  # threshold the background
                tmp_data[mask == 1] = 0
                fig = gu.combined_plots(tuple_array=(monitor, tmp_data), tuple_sum_frames=(False, True),
                                        tuple_sum_axis=(0, 1), tuple_width_v=None,
                                        tuple_width_h=None, tuple_colorbar=(False, False),
                                        tuple_vmin=(np.nan, 0), tuple_vmax=(np.nan, np.nan),
                                        tuple_title=('monitor.min() / monitor', 'Gridded normed data (threshold 5)\n'),
                                        tuple_scale=('linear', 'log'), xlabel=('Frame number', "Q$_y$"),
                                        ylabel=('Counts (a.u.)', "Q$_x$"), position=(323, 122),
                                        is_orthogonal=not use_rawdata, reciprocal_space=True)

                fig.savefig(detector.savedir + f'monitor_gridded_S{scan_nb}_{nz}_{ny}_{nx}' +
                            binning_comment + '.png')
                if flag_interact:
                    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
                    cid = plt.connect('close_event', close_event)
                    fig.waitforbuttonpress()
                    plt.disconnect(cid)
                plt.close(fig)
                plt.ioff()
                del tmp_data
                gc.collect()

    ########################
    # crop/pad/center data #
    ########################
    data, mask, pad_width, q_values, frames_logical = \
        pru.center_fft(data=data, mask=mask, detector=detector, frames_logical=frames_logical, centering=centering,
                       fft_option=center_fft, pad_size=pad_size, fix_bragg=fix_bragg, fix_size=fix_size,
                       q_values=q_values)

    starting_frame = [pad_width[0], pad_width[2], pad_width[4]]  # no need to check padded frames
    print('\nPad width:', pad_width)
    nz, ny, nx = data.shape
    print('\nData size after cropping / padding:', nz, ny, nx)

    ##########################################
    # optional masking of zero photon events #
    ##########################################
    if mask_zero_event:
        # mask points when there is no intensity along the whole rocking curve - probably dead pixels
        temp_mask = np.zeros((ny, nx))
        temp_mask[np.sum(data, axis=0) == 0] = 1
        mask[np.repeat(temp_mask[np.newaxis, :, :], repeats=nz, axis=0) == 1] = 1
        del temp_mask

    ###########################################
    # save data and mask before alien removal #
    ###########################################
    fig, _, _ = gu.multislices_plot(data, sum_frames=True, scale='log', plot_colorbar=True, vmin=0,
                                    title='Data before aliens removal\n',
                                    is_orthogonal=not use_rawdata, reciprocal_space=True)
    if debug:
        plt.savefig(detector.savedir + f'data_before_masking_sum_S{scan_nb}_{nz}_{ny}_{nx}_{detector.binning[0]}_'
                                       f'{detector.binning[1]}_{detector.binning[2]}.png')
    if flag_interact:
        fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
        cid = plt.connect('close_event', close_event)
        fig.waitforbuttonpress()
        plt.disconnect(cid)
    plt.close(fig)

    piz, piy, pix = np.unravel_index(data.argmax(), data.shape)
    fig = gu.combined_plots((data[piz, :, :], data[:, piy, :], data[:, :, pix]), tuple_sum_frames=False,
                            tuple_sum_axis=0, tuple_width_v=None, tuple_width_h=None, tuple_colorbar=True,
                            tuple_vmin=0, tuple_vmax=np.nan, tuple_scale='log',
                            tuple_title=('data at max in xy', 'data at max in xz', 'data at max in yz'),
                            is_orthogonal=not use_rawdata, reciprocal_space=False)
    if debug:
        plt.savefig(detector.savedir + f'data_before_masking_S{scan_nb}_{nz}_{ny}_{nx}_{detector.binning[0]}_'
                                       f'{detector.binning[1]}_{detector.binning[2]}.png')
    if flag_interact:
        fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
        cid = plt.connect('close_event', close_event)
        fig.waitforbuttonpress()
        plt.disconnect(cid)
    plt.close(fig)

    fig, _, _ = gu.multislices_plot(mask, sum_frames=True, scale='linear', plot_colorbar=True, vmin=0,
                                    vmax=(nz, ny, nx), title='Mask before aliens removal\n',
                                    is_orthogonal=not use_rawdata, reciprocal_space=True)
    if debug:
        plt.savefig(detector.savedir + f'mask_before_masking_S{scan_nb}_{nz}_{ny}_{nx}_{detector.binning[0]}_'
                                       f'{detector.binning[1]}_{detector.binning[2]}.png')

    if flag_interact:
        fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
        cid = plt.connect('close_event', close_event)
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

        if save_to_vti:
            # save diffraction pattern to vti
            nqx, nqz, nqy = data.shape  # in nexus z downstream, y vertical / in q z vertical, x downstream
            print('\ndqx, dqy, dqz = ', qx[1] - qx[0], qy[1] - qy[0], qz[1] - qz[0])
            # in nexus z downstream, y vertical / in q z vertical, x downstream
            qx0 = qx.min()
            dqx = (qx.max() - qx0) / nqx
            qy0 = qy.min()
            dqy = (qy.max() - qy0) / nqy
            qz0 = qz.min()
            dqz = (qz.max() - qz0) / nqz

            gu.save_to_vti(filename=os.path.join(detector.savedir, f'S{scan_nb}_ortho_int' + comment + ".vti"),
                           voxel_size=(dqx, dqz, dqy), tuple_array=data, tuple_fieldnames='int', origin=(qx0, qz0, qy0))

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

        fig_mask, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))
        fig_mask.canvas.mpl_disconnect(fig_mask.canvas.manager.key_press_handler_id)
        original_data = np.copy(data)
        original_mask = np.copy(mask)
        frame_index = starting_frame
        ax0.imshow(data[frame_index[0], :, :], vmin=0, vmax=max_colorbar, cmap=my_cmap)
        ax1.imshow(data[:, frame_index[1], :], vmin=0, vmax=max_colorbar, cmap=my_cmap)
        ax2.imshow(data[:, :, frame_index[2]], vmin=0, vmax=max_colorbar, cmap=my_cmap)
        ax3.set_visible(False)
        ax0.axis('scaled')
        ax1.axis('scaled')
        ax2.axis('scaled')
        if not use_rawdata:
            ax0.invert_yaxis()  # detector Y is vertical down
        ax0.set_title(f'XY - Frame {frame_index[0] + 1} / {nz}')
        ax1.set_title(f'XZ - Frame {frame_index[1] + 1} / {ny}')
        ax2.set_title(f'YZ - Frame {frame_index[2] + 1} / {nx}')
        fig_mask.text(0.60, 0.30, "m mask ; b unmask ; u next frame ; d previous frame", size=12)
        fig_mask.text(0.60, 0.25, "up larger ; down smaller ; right darker ; left brighter", size=12)
        fig_mask.text(0.60, 0.20, "p plot full image ; q quit", size=12)
        plt.tight_layout()
        plt.connect('key_press_event', press_key)
        fig_mask.set_facecolor(background_plot)
        plt.show()
        del fig_mask, original_data, original_mask
        gc.collect()

        mask[np.nonzero(mask)] = 1

        fig, _, _ = gu.multislices_plot(data, sum_frames=True, scale='log', plot_colorbar=True, vmin=0,
                                        title='Data after aliens removal\n',
                                        is_orthogonal=not use_rawdata, reciprocal_space=True)

        fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
        cid = plt.connect('close_event', close_event)
        fig.waitforbuttonpress()
        plt.disconnect(cid)
        plt.close(fig)

        fig, _, _ = gu.multislices_plot(mask, sum_frames=True, scale='linear', plot_colorbar=True, vmin=0,
                                        vmax=(nz, ny, nx), title='Mask after aliens removal\n',
                                        is_orthogonal=not use_rawdata, reciprocal_space=True)

        fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
        cid = plt.connect('close_event', close_event)
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

        fig_mask, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))
        fig_mask.canvas.mpl_disconnect(fig_mask.canvas.manager.key_press_handler_id)
        original_data = np.copy(data)
        updated_mask = np.zeros((nz, ny, nx))
        data[mask == 1] = 0  # will appear as grey in the log plot (nan)
        ax0.imshow(np.log10(abs(data).sum(axis=0)), vmin=0, vmax=max_colorbar, cmap=my_cmap)
        ax1.imshow(np.log10(abs(data).sum(axis=1)), vmin=0, vmax=max_colorbar, cmap=my_cmap)
        ax2.imshow(np.log10(abs(data).sum(axis=2)), vmin=0, vmax=max_colorbar, cmap=my_cmap)
        ax3.set_visible(False)
        ax0.axis('scaled')
        ax1.axis('scaled')
        ax2.axis('scaled')
        if not use_rawdata:
            ax0.invert_yaxis()  # detector Y is vertical down
        ax0.set_title("XY")
        ax1.set_title("XZ")
        ax2.set_title("YZ")
        fig_mask.text(0.60, 0.45, "click to select the vertices of a polygon mask", size=12)
        fig_mask.text(0.60, 0.40, "x to pause/resume polygon masking for pan/zoom", size=12)
        fig_mask.text(0.60, 0.35, "p plot mask ; r reset current points", size=12)
        fig_mask.text(0.60, 0.30, "m square mask ; b unmask ; right darker ; left brighter", size=12)
        fig_mask.text(0.60, 0.25, "up larger masking box ; down smaller masking box", size=12)
        fig_mask.text(0.60, 0.20, "a restart ; q quit", size=12)
        info_text = fig_mask.text(0.60, 0.05, "masking enabled", size=16)
        plt.tight_layout()
        plt.connect('key_press_event', press_key)
        plt.connect('button_press_event', on_click)
        fig_mask.set_facecolor(background_plot)
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
    if flag_medianfilter == 'mask_isolated' or flag_medianfilter == 'interp_isolated':
        print("\nFiltering isolated pixels")
        nb_pix = 0
        for idx in range(pad_width[0], nz-pad_width[1]):  # filter only frames whith data (not padded)
            data[idx, :, :], processed_pix, mask[idx, :, :] = \
                pru.mean_filter(data=data[idx, :, :], nb_neighbours=medfilt_order, mask=mask[idx, :, :],
                                interpolate=flag_medianfilter, min_count=3, debugging=debug)
            nb_pix += processed_pix
            sys.stdout.write(f'\rImage {idx}, number of filtered pixels: {processed_pix}')
            sys.stdout.flush()
        print("\nTotal number of filtered pixels: ", nb_pix)
    elif flag_medianfilter == 'median':  # apply median filter
        print("\nApplying median filtering")
        for idx in range(pad_width[0], nz-pad_width[1]):  # filter only frames whith data (not padded)
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

    #################################################################################################################
    # rotate the diffraction pattern interpolated into the laboratory frame, to have q aligned along one array axis #
    #################################################################################################################
    # for q values, the frame used is (qx downstream, qy outboard, qz vertical up)
    if not use_rawdata and interp_method == 'linearization' and align_q:
        comment += f'_align-q-{ref_axis_q}'
        if ref_axis_q == "x":
            myaxis = np.array([1, 0, 0])  # must be in [x, y, z] order
        elif ref_axis_q == "y":
            myaxis = np.array([0, 1, 0])  # must be in [x, y, z] order
        elif ref_axis_q == "z":
            myaxis = np.array([0, 0, 1])  # must be in [x, y, z] order
        else:
            ref_axis_q = "y"
            myaxis = np.array([0, 1, 0])  # must be in [x, y, z] order

        nqx, nqz, nqy = data.shape
        qx, qz, qy = q_values
        dqx = (qx.max() - qx.min()) / nqx
        dqy = (qy.max() - qy.min()) / nqy
        dqz = (qz.max() - qz.min()) / nqz

        # find the position in q of the center of mass
        piz, piy, pix = center_of_mass(data)
        qx_com = qx[int(np.rint(piz))]
        qz_com = qz[int(np.rint(piy))]
        qy_com = qy[int(np.rint(pix))]
        q_com = np.array([qx_com, qy_com, qz_com])
        qnorm = np.linalg.norm(q_com)  # in 1/A
        planar_dist = 2 * np.pi / qnorm  # in A
        print(f"\nWavevector transfer (qx, qz, qy): {qx_com:.4f}, {qz_com:.4f}, {qy_com:.4f}")
        print("Wavevector transfer: (1/A)", str('{:.4f}'.format(qnorm)))
        print("Atomic plane distance: (A)", str('{:.4f}'.format(planar_dist)), "angstroms")
        print(f'Aligning Q along {ref_axis_q}: {myaxis}')

        # axes in rotate_crystal must be in [x, y, z] order, i.e. [qy, qz, qx]
        data = pu.rotate_crystal(array=data, axis_to_align=np.array([qy_com, qz_com, qx_com]) / np.linalg.norm(qnorm),
                                 reference_axis=myaxis, voxel_size=(dqy, dqz, dqx), scale='log', debugging=debug)
        mask = pu.rotate_crystal(array=mask, axis_to_align=np.array([qy_com, qz_com, qx_com]) / np.linalg.norm(qnorm),
                                 reference_axis=myaxis, voxel_size=(dqy, dqz, dqx), fill_value=1, debugging=debug)

    ################################################
    # check for nans and infs in the data and mask #
    ################################################
    nz, ny, nx = data.shape
    print('\nData size after masking:', nz, ny, nx)

    data, mask = util.remove_nan(data=data, mask=mask)

    data[mask == 1] = 0

    ####################
    # debugging plots  #
    ####################
    plt.ion()
    if debug:
        z0, y0, x0 = center_of_mass(data)
        fig, _, _ = gu.multislices_plot(data, sum_frames=False, scale='log', plot_colorbar=True, vmin=0,
                                        title='Masked data', slice_position=[int(z0), int(y0), int(x0)],
                                        is_orthogonal=not use_rawdata, reciprocal_space=True)
        plt.savefig(detector.savedir + f'middle_frame_S{scan_nb}_{nz}_{ny}_{nx}_{detector.binning[0]}_'
                                       f'{detector.binning[1]}_{detector.binning[2]}' + comment + '.png')
        if not flag_interact:
            plt.close(fig)

        fig, _, _ = gu.multislices_plot(data, sum_frames=True, scale='log', plot_colorbar=True, vmin=0,
                                        title='Masked data', is_orthogonal=not use_rawdata, reciprocal_space=True)
        plt.savefig(detector.savedir + f'sum_S{scan_nb}_{nz}_{ny}_{nx}_{detector.binning[0]}_'
                                       f'{detector.binning[1]}_{detector.binning[2]}' + comment + '.png')
        if not flag_interact:
            plt.close(fig)

        fig, _, _ = gu.multislices_plot(mask, sum_frames=True, scale='linear', plot_colorbar=True, vmin=0,
                                        vmax=(nz, ny, nx), title='Mask', is_orthogonal=not use_rawdata,
                                        reciprocal_space=True)
        plt.savefig(detector.savedir + f'mask_S{scan_nb}_{nz}_{ny}_{nx}_{detector.binning[0]}_'
                                       f'{detector.binning[1]}_{detector.binning[2]}' + comment + '.png')
        if not flag_interact:
            plt.close(fig)

    ################################################################################################
    # bin the stacking axis if needed, the detector plane was already binned when loading the data #
    ################################################################################################
    if detector.binning[0] != 1 and not reload_orthogonal:  # data was already binned for reload_orthogonal
        data = pu.bin_data(data, (detector.binning[0], 1, 1), debugging=False)
        mask = pu.bin_data(mask, (detector.binning[0], 1, 1), debugging=False)
        mask[np.nonzero(mask)] = 1
        if not use_rawdata and len(q_values) != 0:
            numz = len(qx)
            qx = qx[:numz - (numz % detector.binning[0]):detector.binning[0]]  # along Z
            del numz
    nz, ny, nx = data.shape
    print('\nData size after binning the stacking dimension:', data.shape)
    comment = f'{comment}_{nz}_{ny}_{nx}' + binning_comment

    ############################
    # save final data and mask #
    ############################
    print('\nSaving directory:', detector.savedir)
    if save_asint:
        data = data.astype(int)
    print('Data type before saving:', data.dtype)
    mask[np.nonzero(mask)] = 1
    mask = mask.astype(int)
    print('Mask type before saving:', mask.dtype)
    if not use_rawdata and len(q_values) != 0:
        if save_to_npz:
            np.savez_compressed(detector.savedir + f'QxQzQy_S{scan_nb}' + comment,
                                qx=q_values[0], qz=q_values[1], qy=q_values[2])
        if save_to_mat:
            savemat(detector.savedir + f'S{scan_nb}_qx.mat', {'qx': q_values[0]})
            savemat(detector.savedir + f'S{scan_nb}_qz.mat', {'qz': q_values[1]})
            savemat(detector.savedir + f'S{scan_nb}_qy.mat', {'qy': q_values[2]})
        max_z = data.sum(axis=0).max()
        fig, _, _ = gu.contour_slices(data, (q_values[0], q_values[1], q_values[2]), sum_frames=True,
                                      title='Final data', plot_colorbar=True, scale='log', is_orthogonal=True,
                                      levels=np.linspace(0, np.ceil(np.log10(max_z)), 150, endpoint=False),
                                      reciprocal_space=True)
        fig.savefig(detector.savedir + f'final_reciprocal_space_S{scan_nb}' + comment + '.png')
        plt.close(fig)

    if save_to_npz:
        np.savez_compressed(detector.savedir + f'S{scan_nb}_pynx' + comment, data=data)
        np.savez_compressed(detector.savedir + f'S{scan_nb}_maskpynx' + comment, mask=mask)

    if save_to_mat:
        # save to .mat, the new order is x y z (outboard, vertical up, downstream)
        savemat(detector.savedir + f'S{scan_nb}_data.mat',
                {'data': np.moveaxis(data.astype(np.float32), [0, 1, 2], [-1, -2, -3])})
        savemat(detector.savedir + f'S{scan_nb}_mask.mat',
                {'data': np.moveaxis(mask.astype(np.int8), [0, 1, 2], [-1, -2, -3])})

    ############################
    # plot final data and mask #
    ############################
    data[np.nonzero(mask)] = 0
    fig, _, _ = gu.multislices_plot(data, sum_frames=True, scale='log', plot_colorbar=True, vmin=0,
                                    title='Final data', is_orthogonal=not use_rawdata,
                                    reciprocal_space=True)
    plt.savefig(detector.savedir + f'finalsum_S{scan_nb}' + comment + '.png')
    if not flag_interact:
        plt.close(fig)

    fig, _, _ = gu.multislices_plot(mask, sum_frames=True, scale='linear', plot_colorbar=True, vmin=0,
                                    vmax=(nz, ny, nx), title='Final mask',
                                    is_orthogonal=not use_rawdata, reciprocal_space=True)
    plt.savefig(detector.savedir + f'finalmask_S{scan_nb}' + comment + '.png')
    if not flag_interact:
        plt.close(fig)

    del data, mask
    gc.collect()

    if len(scans) > 1:
        plt.close('all')

print('\nEnd of script')
plt.ioff()
plt.show()
