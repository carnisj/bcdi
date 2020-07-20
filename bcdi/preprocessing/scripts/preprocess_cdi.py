# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import hdf5plugin  # for P10, should be imported before h5py or PyTables
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend("Qt5Agg")  # "Qt5Agg" or "Qt4Agg" depending on the version of Qt installer, bug with Tk
import pathlib
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


helptext = """
Prepare experimental data for forward CDI phasing: crop/pad, center, mask, normalize, filter and regrid the data.

Beamlines currently supported: ESRF ID01, SOLEIL CRISTAL, SOLEIL SIXS and PETRAIII P10.

Output: data and mask as numpy .npz or Matlab .mat 3D arrays for phasing

File structure should be (e.g. scan 1):
specfile, background, hotpixels file and flatfield file in:    /rootdir/
data in:                                                       /rootdir/S1/data/

output files saved in:   /rootdir/S1/pynxraw/ or /rootdir/S1/pynx/ depending on 'use_rawdata' option
"""

scans = [32, 48, 55, 59, 71, 6, 15, 20, 37]  # list or array of scan numbers
root_folder = "/nfs/fs/fscxi/experiments/2020/PETRA/P10/11008562/raw/"
sample_name = ['ht_pillar3', 'ht_pillar3', 'ht_pillar3', 'ht_pillar3', 'ht_pillar3',
               'ht_pillar3_1', 'ht_pillar3_1', 'ht_pillar3_1', 'ht_pillar3_1']  # "S"  # # list of sample names. If only one name is indicated,
# it will be repeated to match the number of scans
user_comment = ''  # string, should start with "_"
debug = False  # set to True to see plots
binning = [1, 2, 2]  # binning that will be used for phasing
# (stacking dimension, detector vertical axis, detector horizontal axis)
##############################
# parameters used in masking #
##############################
flag_interact = True  # True to interact with plots, False to close it automatically
background_plot = '0.5'  # in level of grey in [0,1], 0 being dark. For visual comfort during masking
##############################################
# parameters used in intensity normalization #
##############################################
normalize_method = 'skip'  # 'skip' for no normalization, 'monitor' to use the default monitor, 'sum_roi' to normalize
# by the intensity summed in normalize_roi
normalize_roi = []  # roi for the integration of intensity used as a monitor for data normalization
# [Vstart, Vstop, Hstart, Hstop]
#################################
# parameters for data filtering #
#################################
mask_zero_event = False  # mask pixels where the sum along the rocking curve is zero - may be dead pixels
flag_medianfilter = 'skip'
# set to 'median' for applying med2filter [3,3]
# set to 'interp_isolated' to interpolate isolated empty pixels based on 'medfilt_order' parameter
# set to 'mask_isolated' it will mask isolated empty pixels
# set to 'skip' will skip filtering
medfilt_order = 8    # for custom median filter, number of pixels with intensity surrounding the empty pixel
#################################################
# parameters used when reloading processed data #
#################################################
reload_previous = False  # True to resume a previous masking (load data and mask)
reload_orthogonal = True  # True if the reloaded data is already intepolated in an orthonormal frame
previous_binning = [1, 1, 1]  # binning factors in each dimension of the binned data to be reloaded
save_previous = False  # if True, will save the previous data and mask
######################################################################
# parameters used for interpolating the data in an orthonormal frame #
######################################################################
use_rawdata = False  # False for using data gridded in laboratory frame/ True for using data in detector frame
correct_curvature = False  # True to correcture q values for the curvature of Ewald sphere
fit_datarange = False  # if True, crop the final array within data range, avoiding areas at the corners of the window
# viewed from the top, data is circular, but the interpolation window is rectangular, with nan values outside of data
sdd = 5.0  # sample to detector distance in m, used only if use_rawdata is False
energy = 10000  # x-ray energy in eV, used only if use_rawdata is False
##################
# saving options #
##################
save_rawdata = False  # save also the raw data when use_rawdata is False
save_to_npz = False  # True to save the processed data in npz format
save_to_mat = False  # True to save the processed data in mat format
save_to_vti = False  # save the orthogonalized diffraction pattern to VTK file
save_asint = False  # if True, the result will be saved as an array of integers (save space)
###############################
# beamline related parameters #
###############################
beamline = 'P10'  # name of the beamline, used for data loading and normalization by monitor
# supported beamlines: 'ID01', 'SIXS_2018', 'SIXS_2019', 'CRISTAL', 'P10'
rocking_angle = "inplane"  # "outofplane" or "inplane"
is_series = True  # specific to series measurement at P10
specfile_name = ''
# .spec for ID01, .fio for P10, alias_dict.txt for SIXS_2018, not used for CRISTAL and SIXS_2019
# template for ID01: name of the spec file without '.spec'
# template for SIXS_2018: full path of the alias dictionnary, typically root_folder + 'alias_dict_2019.txt'
# template for SIXS_2019: ''
# template for P10: ''
# template for CRISTAL: ''
###############################
# detector related parameters #
###############################
detector = "Eiger4M"    # "Eiger2M" or "Maxipix" or "Eiger4M"
direct_beam = (1255, 1161)  # tuple of int (vertical, horizontal): position of the direct beam in pixels
# this parameter is important for gridding the data onto the laboratory frame
roi_detector = []  # [direct_beam[0] - 200, direct_beam[0] + 200, direct_beam[1] - 200, direct_beam[1] + 200]
# [Vstart, Vstop, Hstart, Hstop]
# leave it as [] to use the full detector.
photon_threshold = 0  # data[data < photon_threshold] = 0
photon_filter = 'loading'  # 'loading' or 'postprocessing', when the photon threshold should be applied
# if 'loading', it is applied before binning; if 'postprocessing', it is applied at the end of the script before saving
background_file = ''  # root_folder + 'background.npz'  #
hotpixels_file = ''  # root_folder + 'hotpixels.npz'  #
flatfield_file = ''  # root_folder + "flatfield_eiger.npz"  #
template_imagefile = '_master.h5'  # ''_data_%06d.h5'
# template for ID01: 'data_mpx4_%05d.edf.gz' or 'align_eiger2M_%05d.edf.gz'
# template for SIXS_2018: 'align.spec_ascan_mu_%05d.nxs'
# template for SIXS_2019: 'spare_ascan_mu_%05d.nxs'
# template for Cristal: 'S%d.nxs'
# template for P10: '_master.h5'
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
    global original_data, updated_mask, data, mask, frame_index, width, flag_aliens, flag_mask, flag_pause
    global xy, fig_mask, max_colorbar, ax0, ax1, ax2, previous_axis, detector_plane, info_text

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
            invert_yaxis = (not use_rawdata) and (not detector_plane)
            if flag_aliens:
                data, mask, width, max_colorbar, frame_index, stop_masking = \
                    gu.update_aliens_combined(key=event.key, pix=int(np.rint(event.xdata)),
                                              piy=int(np.rint(event.ydata)), original_data=original_data,
                                              original_mask=original_mask, updated_data=data, updated_mask=mask,
                                              axes=(ax0, ax1, ax2), width=width, dim=dim, frame_index=frame_index,
                                              vmin=0, vmax=max_colorbar, invert_yaxis=invert_yaxis)
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
                                            axes=(ax0, ax1, ax2), flag_pause=flag_pause, points=points,
                                            xy=xy, width=width, dim=dim, click_dim=click_dim, info_text=info_text,
                                            vmin=0, vmax=max_colorbar, invert_yaxis=invert_yaxis)

                if click_dim is None:
                    previous_axis = None
            else:
                stop_masking = False

            if stop_masking:
                plt.close(fig_mask)

    except AttributeError:  # mouse pointer out of axes
        pass


#########################
# check some parameters #
#########################
if not reload_previous:
    previous_binning = [1, 1, 1]
    reload_orthogonal = False

if reload_orthogonal:
    use_rawdata = False

if not use_rawdata:
    if reload_orthogonal:  # data already gridded, one can bin the first axis
        pass
    else:  # data in the detector frame, one cannot bin the first axis because it is done during interpolation
        print('use_rawdata=False: defaulting the binning factor along the stacking dimension to 1')
        # the vertical axis y being the rotation axis, binning along z downstream and x outboard will be the same
        binning[0] = 1
        if previous_binning[0] != 1:
            print('previous_binning along axis 0 should be 1 for reloaded data to be gridded (angles will not match)')
            sys.exit()

if type(sample_name) is list:
    if len(sample_name) == 1:
        sample_name = [sample_name[0] for idx in range(len(scans))]
    assert len(sample_name) == len(scans), 'sample_name and scan_list should have the same length'
elif type(sample_name) is str:
    sample_name = [sample_name for idx in range(len(scans))]
else:
    print('sample_name should be either a string or a list of strings')
    sys.exit()

#######################
# Initialize detector #
#######################
kwargs = dict()  # create dictionnary
kwargs['is_series'] = is_series
kwargs['previous_binning'] = previous_binning
try:
    kwargs['nb_pixel_x'] = nb_pixel_x  # fix to declare a known detector but with less pixels (e.g. one tile HS)
except NameError:  # nb_pixel_x not declared
    pass
try:
    kwargs['nb_pixel_y'] = nb_pixel_y  # fix to declare a known detector but with less pixels (e.g. one tile HS)
except NameError:  # nb_pixel_y not declared
    pass

detector = exp.Detector(name=detector, datadir='', template_imagefile=template_imagefile, roi=roi_detector,
                        sum_roi=normalize_roi, binning=binning, **kwargs)

####################
# Initialize setup #
####################
setup = exp.SetupPreprocessing(beamline=beamline, energy=energy, rocking_angle=rocking_angle, distance=sdd,
                               direct_beam=direct_beam)

############################################
# Initialize values for callback functions #
############################################
detector_plane = False
flag_mask = False
flag_aliens = False
plt.rcParams["keymap.quit"] = ["ctrl+w", "cmd+w"]  # this one to avoid that q closes window (matplotlib default)

############################
# start looping over scans #
############################
root = tk.Tk()
root.withdraw()

for scan_nb in range(len(scans)):
    plt.ion()

    comment = user_comment  # initialize comment

    if setup.beamline != 'P10':
        homedir = root_folder + sample_name[scan_nb] + str(scans[scan_nb]) + '/'
        detector.datadir = homedir + "data/"
        specfile = specfile_name
    else:
        specfile = sample_name[scan_nb] + '_{:05d}'.format(scans[scan_nb])
        homedir = root_folder + specfile + '/'
        detector.datadir = homedir + 'e4m/'
        imagefile = specfile + template_imagefile
        detector.template_imagefile = imagefile

    logfile = pru.create_logfile(setup=setup, detector=detector, scan_number=scans[scan_nb],
                                 root_folder=root_folder, filename=specfile)

    print('\nScan', scans[scan_nb])
    print('Setup: ', setup.beamline)
    print('Direct beam (VxH)', direct_beam)
    print('Detector: ', detector.name)
    print('Pixel number (VxH): ', detector.nb_pixel_y, detector.nb_pixel_x)
    print('Detector ROI:', roi_detector)
    print('Horizontal pixel size with binning: ', detector.pixelsize_x, 'm')
    print('Vertical pixel size with binning: ', detector.pixelsize_y, 'm')
    print('Specfile: ', specfile)
    print('Scan type: ', setup.rocking_angle)
    print('Sample to detector distance: ', setup.distance, 'm')
    print('Energy:', setup.energy, 'ev')

    if not use_rawdata:
        comment = comment + '_ortho'
        savedir = homedir + "pynx/"
        pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)
        print('Output will interpolated in the orthogonal laboratory frame')
        plot_title = ['QzQx', 'QyQx', 'QyQz']
    else:
        savedir = homedir + "pynxraw/"
        pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)
        print('Output will be non orthogonal, in the detector frame')
        plot_title = ['YZ', 'XZ', 'XY']

    detector.savedir = savedir

    if normalize_method != 'skip':
        comment = comment + '_norm'

    #############
    # Load data #
    #############
    if reload_previous:  # resume previous masking
        print('Resuming previous masking')
        file_path = filedialog.askopenfilename(initialdir=homedir, title="Select data file",
                                               filetypes=[("NPZ", "*.npz")])
        data = np.load(file_path)
        npz_key = data.files
        data = data[npz_key[0]]
        nz, ny, nx = np.shape(data)

        # update savedir to save the data in the same directory as the reloaded data
        savedir = os.path.dirname(file_path) + '/'
        detector.savedir = savedir

        file_path = filedialog.askopenfilename(initialdir=savedir, title="Select mask file",
                                               filetypes=[("NPZ", "*.npz")])
        mask = np.load(file_path)
        npz_key = mask.files
        mask = mask[npz_key[0]]

        if save_previous:
            np.savez_compressed(savedir + 'S' + str(scans[scan_nb]) + '_pynx_previous', data=data)
            np.savez_compressed(savedir + 'S' + str(scans[scan_nb]) + '_maskpynx_previous', mask=mask)

        if reload_orthogonal:  # the data is gridded in the orthonormal laboratory frame
            use_rawdata = False
            try:
                file_path = filedialog.askopenfilename(initialdir=homedir, title="Select q values",
                                                       filetypes=[("NPZ", "*.npz")])
                reload_qvalues = np.load(file_path)
                q_values = [reload_qvalues['qx'], reload_qvalues['qz'], reload_qvalues['qy']]
            except FileNotFoundError:
                q_values = []

            normalize_method = 'skip'  # we assume that normalization was already performed
            monitor = []  # we assume that normalization was already performed
            min_range = (nx / 2) * np.sqrt(2)  # used when fit_datarange is True, keep the full array because we do not
            # know the position of the origin of reciprocal space

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
                    qx = qx[:numz - (numz % binning[2]):binning[2]]  # along z downstream, same binning as along x
                    qz = qz[:numy - (numy % binning[1]):binning[1]]  # along y vertical, the axis of rotation
                    qy = qy[:numx - (numx % binning[2]):binning[2]]  # along x outboard
                    del numz, numy, numx
        else:  # the data is in the detector frame
            if photon_filter == 'loading':
                data, mask, frames_logical, monitor = pru.reload_cdi_data(logfile=logfile, scan_number=scans[scan_nb],
                                                                          data=data, mask=mask, detector=detector,
                                                                          setup=setup, debugging=debug,
                                                                          normalize_method=normalize_method,
                                                                          photon_threshold=photon_threshold)
            else:  # photon_filter = 'postprocessing'
                data, mask, frames_logical, monitor = pru.reload_cdi_data(logfile=logfile, scan_number=scans[scan_nb],
                                                                          data=data, mask=mask, detector=detector,
                                                                          setup=setup, debugging=debug,
                                                                          normalize_method=normalize_method)

    else:  # new masking process
        reload_orthogonal = False  # the data is in the detector plane
        flatfield = pru.load_flatfield(flatfield_file)
        hotpix_array = pru.load_hotpixels(hotpixels_file)
        background = pru.load_background(background_file)

        if photon_filter == 'loading':
            data, mask, frames_logical, monitor = pru.load_cdi_data(logfile=logfile, scan_number=scans[scan_nb],
                                                                    detector=detector, setup=setup, flatfield=flatfield,
                                                                    hotpixels=hotpix_array, background=background,
                                                                    normalize=normalize_method, debugging=debug,
                                                                    photon_threshold=photon_threshold)
        else:  # photon_filter = 'postprocessing'
            data, mask, frames_logical, monitor = pru.load_cdi_data(logfile=logfile, scan_number=scans[scan_nb],
                                                                    detector=detector, setup=setup, flatfield=flatfield,
                                                                    hotpixels=hotpix_array, background=background,
                                                                    normalize=normalize_method, debugging=debug)
    nz, ny, nx = np.shape(data)
    print('\nInput data shape:', nz, ny, nx)

    if not reload_orthogonal:
        dirbeam = int((setup.direct_beam[1] - detector.roi[2]) / detector.binning[2])  # updated horizontal direct beam
        min_range = min(dirbeam, nx - dirbeam)  # maximum symmetrical range with defined data
        print('\nMaximum symmetrical range with defined data along detector horizontal direction:', min_range*2,
              'pixels')

        if save_rawdata:
            np.savez_compressed(savedir + 'S' + str(scans[scan_nb]) + '_data_before_masking_stack', data=data)
            if save_to_mat:
                # save to .mat, the new order is x y z (outboard, vertical up, downstream)
                savemat(savedir + 'S' + str(scans[scan_nb]) + '_data_before_masking_stack.mat',
                        {'data': np.moveaxis(data, [0, 1, 2], [-1, -2, -3])})

        if flag_interact:
            # masking step in the detector plane
            plt.ioff()
            width = 0
            max_colorbar = 5
            detector_plane = True
            flag_aliens = False
            flag_mask = True
            flag_pause = False  # press x to pause for pan/zoom
            previous_axis = None
            xy = []  # list of points for mask

            fig_mask = plt.figure(figsize=(12, 9))
            ax0 = fig_mask.add_subplot(121)
            ax1 = fig_mask.add_subplot(322)
            ax2 = fig_mask.add_subplot(324)
            fig_mask.canvas.mpl_disconnect(fig_mask.canvas.manager.key_press_handler_id)
            original_data = np.copy(data)
            updated_mask = np.zeros((nz, ny, nx))
            data[mask == 1] = 0  # will appear as grey in the log plot (nan)
            ax0.imshow(np.log10(abs(data).sum(axis=0)), vmin=0, vmax=max_colorbar)
            ax1.imshow(np.log10(abs(data).sum(axis=1)), vmin=0, vmax=max_colorbar)
            ax2.imshow(np.log10(abs(data).sum(axis=2)), vmin=0, vmax=max_colorbar)
            ax0.axis('scaled')
            ax1.axis('scaled')
            ax2.axis('scaled')
            ax0.set_title("XY")
            ax1.set_title("XZ")
            ax2.set_title("YZ")
            fig_mask.text(0.60, 0.27, "click to select the vertices of a polygon mask", size=10)
            fig_mask.text(0.60, 0.24, "x to pause/resume polygon masking for pan/zoom", size=10)
            fig_mask.text(0.60, 0.21, "p plot mask ; r reset current points", size=10)
            fig_mask.text(0.60, 0.18, "m square mask ; b unmask ; right darker ; left brighter", size=10)
            fig_mask.text(0.60, 0.15, "up larger masking box ; down smaller masking box", size=10)
            fig_mask.text(0.60, 0.12, "a restart ; q quit", size=10)
            info_text = fig_mask.text(0.60, 0.05, "masking enabled", size=16)
            plt.tight_layout()
            plt.connect('key_press_event', press_key)
            plt.connect('button_press_event', on_click)
            fig_mask.set_facecolor(background_plot)
            plt.show()

            mask[np.nonzero(updated_mask)] = 1
            data = original_data
            detector_plane = False
            del fig_mask, original_data, updated_mask
            gc.collect()

        if use_rawdata:
            q_values = []
            binning_comment = '_' + str(previous_binning[0] * binning[0]) + '_' + str(previous_binning[1] * binning[1])\
                              + '_' + str(previous_binning[2] * binning[2])
            # binning along axis 0 is done after masking
            data[np.nonzero(mask)] = 0
        else:  # the data will be gridded, binning[0] is already set to 1
            # sample rotation around the vertical direction at P10:
            # the effective binning in axis 0 is previous_binning[2]*binning[2]
            binning_comment = '_' + str(previous_binning[2] * binning[2]) + '_' + str(previous_binning[1] * binning[1])\
                              + '_' + str(previous_binning[2] * binning[2])

            tmp_data = np.copy(data)  # do not modify the raw data before the interpolation
            tmp_data[mask == 1] = 0
            fig, _, _ = gu.multislices_plot(tmp_data, sum_frames=False, scale='log', plot_colorbar=True, vmin=0,
                                            title='Data before gridding\n', is_orthogonal=False, reciprocal_space=True)
            plt.savefig(savedir + 'data_before_gridding_S' + str(scans[scan_nb]) + '_' + str(nz) + '_' + str(ny) + '_' +
                        str(nx) + binning_comment + '.png')
            plt.close(fig)
            del tmp_data
            gc.collect()

            print('\nGridding the data in the orthonormal laboratory frame')
            data, mask, q_values, frames_logical = \
                pru.grid_cdi(data=data, mask=mask, logfile=logfile, detector=detector, setup=setup,
                             frames_logical=frames_logical, correct_curvature=correct_curvature, debugging=debug)

            # plot normalization by incident monitor for the gridded data
            if normalize_method != 'skip':
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

                fig.savefig(savedir + 'monitor_gridded_S' + str(scans[scan_nb]) + '_' + str(nz) + '_' + str(ny) + '_' +
                            str(nx) + binning_comment + '.png')
                if flag_interact:
                    cid = plt.connect('close_event', close_event)
                    fig.waitforbuttonpress()
                    plt.disconnect(cid)
                plt.close(fig)
                plt.ioff()
                del tmp_data
                gc.collect()

    else:  # reload_orthogonal=True, the data is already gridded, binning was realized along each axis
        binning_comment = '_' + str(previous_binning[0] * binning[0]) + '_' + str(previous_binning[1] * binning[1]) +\
                          '_' + str(previous_binning[2] * binning[2])

    nz, ny, nx = np.shape(data)
    plt.ioff()

    ##########################################
    # optional masking of zero photon events #
    ##########################################
    if mask_zero_event:
        # mask points when there is no intensity along the whole rocking curve - probably dead pixels
        temp_mask = np.zeros((ny, nx))
        temp_mask[np.sum(data, axis=0) == 0] = 1
        mask[np.repeat(temp_mask[np.newaxis, :, :], repeats=nz, axis=0) == 1] = 1
        del temp_mask

    #####################################
    # save data and mask before masking #
    #####################################
    fig, _, _ = gu.multislices_plot(data, sum_frames=True, scale='log', plot_colorbar=True, vmin=0,
                                    title='Data before aliens removal\n',
                                    is_orthogonal=not use_rawdata, reciprocal_space=True)
    if debug:
        plt.savefig(savedir + 'data_before_masking_S' + str(scans[scan_nb]) + '_' + str(nz) + '_' + str(ny) + '_' +
                    str(nx) + binning_comment + '.png')

    if flag_interact:
        cid = plt.connect('close_event', close_event)
        fig.waitforbuttonpress()
        plt.disconnect(cid)
    plt.close(fig)

    fig, _, _ = gu.multislices_plot(mask, sum_frames=True, scale='linear', plot_colorbar=True, vmin=0,
                                    vmax=(nz, ny, nx), title='Mask before aliens removal\n',
                                    is_orthogonal=not use_rawdata, reciprocal_space=True)
    if debug:
        plt.savefig(savedir + 'mask_before_masking_S' + str(scans[scan_nb]) + '_' + str(nz) + '_' + str(ny) + '_' +
                    str(nx) + binning_comment + '.png')

    if flag_interact:
        cid = plt.connect('close_event', close_event)
        fig.waitforbuttonpress()
        plt.disconnect(cid)
    plt.close(fig)

    ###############################################
    # save the orthogonalized diffraction pattern #
    ###############################################
    if not use_rawdata and len(q_values) != 0:
        qx = q_values[0]  # downstream
        qz = q_values[1]  # vertical up
        qy = q_values[2]  # outboard

        if save_to_vti:
            nqx, nqz, nqy = data.shape  # in nexus z downstream, y vertical / in q z vertical, x downstream
            print('\ndqx, dqy, dqz = ', qx[1] - qx[0], qy[1] - qy[0], qz[1] - qz[0])
            # in nexus z downstream, y vertical / in q z vertical, x downstream
            qx0 = qx.min()
            dqx = (qx.max() - qx0) / nqx
            qy0 = qy.min()
            dqy = (qy.max() - qy0) / nqy
            qz0 = qz.min()
            dqz = (qz.max() - qz0) / nqz

            gu.save_to_vti(filename=os.path.join(savedir, "S"+str(scans[scan_nb])+"_ortho_int"+comment+".vti"),
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
        frame_index = [0, 0, 0]
        ax0.imshow(data[frame_index[0], :, :], vmin=0, vmax=max_colorbar)
        ax1.imshow(data[:, frame_index[1], :], vmin=0, vmax=max_colorbar)
        ax2.imshow(data[:, :, frame_index[2]], vmin=0, vmax=max_colorbar)
        ax3.set_visible(False)
        ax0.axis('scaled')
        ax1.axis('scaled')
        ax2.axis('scaled')
        if not use_rawdata:
            ax0.invert_yaxis()  # detector Y is vertical down
        ax0.set_title("XY - Frame " + str(frame_index[0] + 1) + "/" + str(nz))
        ax1.set_title("XZ - Frame " + str(frame_index[1] + 1) + "/" + str(ny))
        ax2.set_title("YZ - Frame " + str(frame_index[2] + 1) + "/" + str(nx))
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

        if flag_interact:
            cid = plt.connect('close_event', close_event)
            fig.waitforbuttonpress()
            plt.disconnect(cid)
        plt.close(fig)

        fig, _, _ = gu.multislices_plot(mask, sum_frames=True, scale='linear', plot_colorbar=True, vmin=0,
                                        vmax=(nz, ny, nx), title='Mask after aliens removal\n',
                                        is_orthogonal=not use_rawdata, reciprocal_space=True)

        if flag_interact:
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
        ax0.imshow(np.log10(abs(data).sum(axis=0)), vmin=0, vmax=max_colorbar)
        ax1.imshow(np.log10(abs(data).sum(axis=1)), vmin=0, vmax=max_colorbar)
        ax2.imshow(np.log10(abs(data).sum(axis=2)), vmin=0, vmax=max_colorbar)
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
        fig_mask.text(0.60, 0.40, "then p to apply and see the result", size=12)
        fig_mask.text(0.60, 0.30, "x to pause/resume masking for pan/zoom", size=12)
        fig_mask.text(0.60, 0.25, "up larger masking box ; down smaller masking box", size=12)
        fig_mask.text(0.60, 0.20, "m mask ; b unmask ; right darker ; left brighter", size=12)
        fig_mask.text(0.60, 0.15, "p plot full masked data ; a restart ; q quit", size=12)
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

    ###############################################
    # mask or median filter isolated empty pixels #
    ###############################################
    if flag_medianfilter == 'mask_isolated' or flag_medianfilter == 'interp_isolated':
        print("\nFiltering isolated pixels")
        nb_pix = 0
        for idx in range(nz):  # filter only frames whith data (not padded)
            data[idx, :, :], numb_pix, mask[idx, :, :] = \
                pru.mean_filter(data=data[idx, :, :], nb_neighbours=medfilt_order, mask=mask[idx, :, :],
                                interpolate=flag_medianfilter, min_count=3, debugging=debug)
            nb_pix = nb_pix + numb_pix
            print("Processed image nb: ", idx)
        if flag_medianfilter == 'mask_isolated':
            print("\nTotal number of masked isolated pixels: ", nb_pix)
        if flag_medianfilter == 'interp_isolated':
            print("\nTotal number of interpolated isolated pixels: ", nb_pix)

    elif flag_medianfilter == 'median':  # apply median filter
        for idx in range(nz):  # filter only frames whith data (not padded)
            data[idx, :, :] = scipy.signal.medfilt2d(data[idx, :, :], [3, 3])
        print("\nApplying median filtering")
    else:
        print("\nSkipping median filtering")

    ##########################
    # apply photon threshold #
    ##########################
    if photon_threshold != 0:
        mask[data < photon_threshold] = 1
        data[data < photon_threshold] = 0
        print("\nApplying photon threshold < ", photon_threshold)

    ########################################
    # check for nans / inf, convert to int #
    ########################################
    plt.ion()
    nz, ny, nx = np.shape(data)
    print('\nData size after masking:', nz, ny, nx)

    # check for Nan
    mask[np.isnan(data)] = 1
    data[np.isnan(data)] = 0
    mask[np.isnan(mask)] = 1
    # check for Inf
    mask[np.isinf(data)] = 1
    data[np.isinf(data)] = 0
    mask[np.isinf(mask)] = 1

    data[mask == 1] = 0
    if save_asint:
        data = data.astype(int)

    ####################
    # debugging plots  #
    ####################
    if debug:
        z0, y0, x0 = center_of_mass(data)
        fig, _, _ = gu.multislices_plot(data, sum_frames=False, scale='log', plot_colorbar=True, vmin=0,
                                        title='Masked data', slice_position=[int(z0), int(y0), int(x0)],
                                        is_orthogonal=not use_rawdata, reciprocal_space=True)
        plt.savefig(savedir + 'middle_frame_S' + str(scans[scan_nb]) + '_' + str(nz) + '_' + str(ny) + '_' +
                    str(nx) + binning_comment + '.png')
        if not flag_interact:
            plt.close(fig)

        fig, _, _ = gu.multislices_plot(data, sum_frames=True, scale='log', plot_colorbar=True, vmin=0,
                                        title='Masked data', is_orthogonal=not use_rawdata, reciprocal_space=True)
        plt.savefig(savedir + 'sum_S' + str(scans[scan_nb]) + '_' + str(nz) + '_' + str(ny) + '_' +
                    str(nx) + binning_comment + '.png')
        if not flag_interact:
            plt.close(fig)

        fig, _, _ = gu.multislices_plot(mask, sum_frames=True, scale='linear', plot_colorbar=True, vmin=0,
                                        vmax=(nz, ny, nx), title='Mask', is_orthogonal=not use_rawdata,
                                        reciprocal_space=True)
        plt.savefig(savedir + 'mask_S' + str(scans[scan_nb]) + '_' + str(nz) + '_' + str(ny) + '_' +
                    str(nx) + binning_comment + '.png')
        if not flag_interact:
            plt.close(fig)

    if not use_rawdata and fit_datarange:
        ############################################################
        # select the largest cubic array fitting inside data range #
        ############################################################
        # this is to avoid having large masked areas near the corner of the area
        # which is a side effect of regridding the data from cylindrical coordinates
        final_nxz = int(np.floor(min_range*2 / np.sqrt(2)))
        if (final_nxz % 2) != 0:
            final_nxz = final_nxz - 1  # we want the number of pixels to be even
        data = data[(nz-final_nxz)//2:(nz-final_nxz)//2 + final_nxz, :, (nz-final_nxz)//2:(nz-final_nxz)//2 + final_nxz]
        mask = mask[(nz-final_nxz)//2:(nz-final_nxz)//2 + final_nxz, :, (nz-final_nxz)//2:(nz-final_nxz)//2 + final_nxz]
        print('\nData size after taking the largest data-defined area:', data.shape)
        if len(q_values) != 0:
            qx = qx[(nz-final_nxz)//2:(nz-final_nxz)//2 + final_nxz]  # along Z
            qy = qy[(nz-final_nxz)//2:(nz-final_nxz)//2 + final_nxz]  # along X
            # qz (along Y) keeps the same number of pixels
        else:
            print('fit_datarange: q values are not provided')

    ###############################################################################################################
    # only for non gridded data, bin the stacking axis, the detector plane was already binned during data loading #
    ###############################################################################################################
    if detector.binning[0] != 1 and not reload_orthogonal:  # for data to be gridded, binning[0] is set to 1
        data = pu.bin_data(data, (detector.binning[0], 1, 1), debugging=False)
        mask = pu.bin_data(mask, (detector.binning[0], 1, 1), debugging=False)
        mask[np.nonzero(mask)] = 1

    nz, ny, nx = data.shape
    print('\nData size after binning the stacking dimension:', data.shape)
    comment = comment + "_" + str(nz) + "_" + str(ny) + "_" + str(nx) + binning_comment

    ############################
    # save final data and mask #
    ############################
    print('\nSaving directory:', savedir)
    if not use_rawdata and len(q_values) != 0:
        if save_to_npz:
            np.savez_compressed(savedir + 'QxQzQy_S' + str(scans[scan_nb]) + comment, qx=qx, qz=qz, qy=qy)
        if save_to_mat:
            savemat(savedir + 'S' + str(scans[scan_nb]) + '_qx.mat', {'qx': qx})
            savemat(savedir + 'S' + str(scans[scan_nb]) + '_qy.mat', {'qy': qy})
            savemat(savedir + 'S' + str(scans[scan_nb]) + '_qz.mat', {'qz': qz})
        fig, _, _ = gu.contour_slices(data, (qx, qz, qy), sum_frames=True, title='Final data',
                                      levels=np.linspace(0, int(np.log10(data.max())), 150, endpoint=False),
                                      plot_colorbar=True, scale='log', is_orthogonal=True, reciprocal_space=True)
        fig.savefig(detector.savedir + 'final_reciprocal_space_S' + str(scans[scan_nb]) + comment + '.png')
        plt.close(fig)

    if save_to_npz:
        np.savez_compressed(savedir + 'S' + str(scans[scan_nb]) + '_pynx' + comment, data=data)
        np.savez_compressed(savedir + 'S' + str(scans[scan_nb]) + '_maskpynx' + comment, mask=mask)

    if save_to_mat:
        # save to .mat, the new order is x y z (outboard, vertical up, downstream)
        savemat(savedir + 'S' + str(scans[scan_nb]) + '_data.mat',
                {'data': np.moveaxis(data.astype(np.float32), [0, 1, 2], [-1, -2, -3])})
        savemat(savedir + 'S' + str(scans[scan_nb]) + '_mask.mat',
                {'data': np.moveaxis(mask.astype(np.int8), [0, 1, 2], [-1, -2, -3])})

    ############################
    # plot final data and mask #
    ############################
    data[np.nonzero(mask)] = 0
    fig, _, _ = gu.multislices_plot(data, sum_frames=True, scale='log', plot_colorbar=True, vmin=0,
                                    title='Final data', is_orthogonal=not use_rawdata,
                                    reciprocal_space=True)
    plt.savefig(savedir + 'finalsum_S' + str(scans[scan_nb]) + comment + '.png')
    if not flag_interact:
        plt.close(fig)

    fig, _, _ = gu.multislices_plot(mask, sum_frames=True, scale='linear', plot_colorbar=True, vmin=0,
                                    vmax=(nz, ny, nx), title='Final mask',
                                    is_orthogonal=not use_rawdata, reciprocal_space=True)
    plt.savefig(savedir + 'finalmask_S' + str(scans[scan_nb]) + comment + '.png')
    if not flag_interact:
        plt.close(fig)

    del data, mask
    gc.collect()
print('\nEnd of script')
plt.ioff()
plt.show()
