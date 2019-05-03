# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#       authors:
#         Jerome Carnis, jerome.carnis@esrf.fr

import hdf5plugin  # for P10, should be imported before h5py or PyTables
import xrayutilities as xu
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os
import scipy.signal  # for medfilt2d
import sys
from scipy.io import savemat
import tkinter as tk
from tkinter import filedialog
import gc
# sys.path.append('C:\\Users\\carnis\\Work Folders\\Documents\\myscripts\\bcdi\\')
import bcdi.graph.graph_utils as gu
import bcdi.experiment.experiment_utils as exp
import bcdi.preprocessing.preprocessing_utils as pru


helptext = """
prepare_cdi_mask.py
Prepare experimental data for CDI phasing
Output: data and mask as numpy .npz arrays for phasing

File structure should be (e.g. scan 1):
specfile, hotpixels file and flatfield file in:    /rootdir/
data in:                                           /rootdir/S1/data/


output files will be saved in:   /rootdir/S1/pynxraw/ or /rootdir/S1/pynx/ depending on 'use_rawdata' option

DOCUMENTATION TO BE IMPROVED, SEE EXPLANATIONS IN SCRIPT

"""

scans = [1067]  # list or array of scan numbers    np.arange(173, 185+1, 3)
root_folder = "C:\\Users\\carnis\\Work Folders\\Documents\\data\\HC3207\\"
sample_name = "SN"  # "S"
comment = '_'  # string, should start with "_"
###########################
flag_interact = False  # True to interact with plots, False to close it automatically
###########################
centering = 'max'  # Bragg peak determination: 'max' or 'com', 'max' is better usually.
#  It will be overridden by 'fix_bragg' if not empty
fix_bragg = []  # fix the Bragg peak position [z_bragg, y_bragg, x_bragg]
# It is useful if hotpixels or intense aliens. Leave it [] otherwise.
###########################
fix_size = []  # [0, 400, 68, 516, 0, 400]  # crop the array to predefined size, leave it to [] otherwise
# [zstart, zstop, ystart, ystop, xstart, xstop]
###########################
center_fft = 'pad_asymmetric_Z_crop_YX'
# 'crop_symmetric_ZYX','crop_asymmetric_ZYX','pad_asymmetric_Z_crop_YX', 'pad_symmetric_Z_crop_YX',
# 'pad_symmetric_Z', 'pad_asymmetric_Z', 'pad_symmetric_ZYX','pad_asymmetric_ZYX' or 'do_nothing'
pad_size = []  # size after padding, e.g. [256, 512, 512]
# used in 'pad_symmetric_Z_crop_YX', 'pad_symmetric_Z', 'pad_symmetric_ZYX'
###########################
normalize_flux = True  # will normalize the intensity by the default monitor
# exp1 or mon2 monitor on ID01, imon1 for SIXS, ipetra fro P10
###########################
mask_zero_event = False  # mask pixels where the sum along the rocking curve is zero - may be dead pixels
###########################
flag_medianfilter = 'interp_isolated'
# set to 'median' for applying med2filter [3,3]
# set to 'interp_isolated' to interpolate isolated empty pixels based on 'medfilt_order' parameter
# set to 'mask_isolated' it will mask isolated empty pixels
# set to 'skip' will skip filtering
medfilt_order = 8    # for custom median filter, number of pixels with intensity surrounding the empty pixel
###########################
reload_previous = 0  # set to 1 to resume a previous masking (load data and mask)
###########################
use_rawdata = True  # 0 for using data orthogonalized by xrayutilities/ 1 for using data in detector reference frame
save_to_mat = False  # set to 1 to save also in .mat format
######################################
# define beamline related parameters #
######################################
beamline = 'ID01'  # 'ID01' or 'SIXS' or 'CRISTAL' or 'P10', used for data loading and normalization by monitor
header_cristal = 'test'  # prefix of the first entry in .nxs file for CRISTAL
headerline_p10 = 57  # nb of header lines in .fio file before data
rocking_angle = "outofplane"  # "outofplane" or "inplane" or "energy"
follow_bragg = False  # only for energy scans, set to True if the detector was also scanned to follow the Bragg peak
specfile_name = 'align2'  # .spec for ID01 and SIXS, .fio for P10, not used for CRISTAL
# template for ID01: name of the spec file without '.spec'
# template for SIXS: name of the spec file including '.spec'
# template for P10: sample_name + '_%05d'
#############################################################
# define detector related parameters and region of interest #
#############################################################
detector = "Eiger2M"    # "Eiger2M" or "Maxipix" or "Eiger4M"
x_bragg = 430  # horizontal pixel number of the Bragg peak
roi_detector = [1202, 1610, x_bragg - 256, x_bragg + 256]  # HC3207
# roi_detector = []
# leave it as [] to use the full detector. Use with center_fft='do_nothing' if you want this exact size.
photon_threshold = 0  # data[data <= photon_threshold] = 0
hotpixels_file = ''  # root_folder + 'hotpixels.npz'
flatfield_file = root_folder + "flatfield_eiger.npz"  #
template_imagefile = 'align_eiger2M_%05d.edf.gz'
# ID01: 'data_mpx4_%05d.edf.gz' or 'align_eiger2M_%05d.edf.gz'
# SIXS: specfile_name + '_ascan_mu_%05d.nxs'
# Cristal: 'S%d.nxs'
# P10: '_data_%06d.h5'
################################################################################
# define paramteres below if you want to orthogonalize the data before phasing #
################################################################################
sdd = 1.25873  # sample to detector distance in m, not important if you use raw data
energy = 8500  # x-ray energy in eV, not important if you use raw data
grazing_angle = 0  # incident angle of diffractometer at SIXS or incident angle below phi for in-plane rocking curves
beam_direction = [1, 0, 0]  # beam along x
sample_inplane = [1, 0, 0]  # sample inplane reference direction along the beam at 0 angles
sample_outofplane = [0, 0, 1]  # surface normal of the sample at 0 angles
offset_inplane = 3.0069  # outer detector angle offset, not important if you use raw data
cch1 = 71.61  # cch1 parameter from xrayutilities 2D detector calibration, detector roi is taken into account below
cch2 = 1656.65  # cch2 parameter from xrayutilities 2D detector calibration, detector roi is taken into account below
detrot = -0.897  # detrot parameter from xrayutilities 2D detector calibration
tiltazimuth = 28.4  # tiltazimuth parameter from xrayutilities 2D detector calibration
tilt = 3.772  # tilt parameter from xrayutilities 2D detector calibration
##################################
# end of user-defined parameters #
##################################


def on_click(event):
    """
    Function to interact with a plot, return the position of clicked pixel. If flag_pause==1 or
    if the mouse is out of plot axes, it will not register the click

    :param event: mouse click event
    :return: updated list of vertices which defines a polygon to be masked
    """
    global xy, flag_pause
    if not event.inaxes:
        return
    if not flag_pause:
        _x, _y = int(np.rint(event.xdata)), int(np.rint(event.ydata))
        xy.append([_x, _y])
    return


def press_key(event):
    """
    Interact with a plot for masking parasitic diffraction intensity or detector gaps

    :param event: button press event
    :return: updated data, mask and controls
    """
    global original_data, data, mask, temp_mask, dim, idx, width, flag_aliens, flag_mask, flag_pause, max_colorbar
    global xy, points, fig_mask, masked_color

    try:
        if flag_aliens:
            data, mask, width, max_colorbar, idx, stop_masking = \
                pru.update_aliens(key=event.key, pix=int(np.rint(event.xdata)), piy=int(np.rint(event.ydata)),
                                  original_data=original_data, updated_data=data, updated_mask=mask,
                                  figure=fig_mask, width=width, dim=dim, idx=idx, vmin=0, vmax=max_colorbar)
        elif flag_mask:
            data, temp_mask, flag_pause, xy, width, vmax, stop_masking = \
                pru.update_mask(key=event.key, pix=int(np.rint(event.xdata)), piy=int(np.rint(event.ydata)),
                                original_data=original_data, original_mask=mask, updated_data=data,
                                updated_mask=temp_mask, figure=fig_mask, flag_pause=flag_pause, points=points,
                                xy=xy, width=width, dim=dim, vmin=0, vmax=max_colorbar, masked_color=masked_color)
        else:
            stop_masking = False

        if stop_masking:
            plt.close(fig_mask)

    except AttributeError:  # mouse pointer out of axes
        pass


#######################
# Initialize detector #
#######################
detector = exp.Detector(name=detector, datadir='', template_imagefile=template_imagefile, roi=roi_detector)

####################
# Initialize setup #
####################
setup = exp.SetupPreprocessing(beamline=beamline, energy=energy, rocking_angle=rocking_angle, distance=sdd,
                               grazing_angle=grazing_angle, beam_direction=beam_direction,
                               sample_inplane=sample_inplane, sample_outofplane=sample_outofplane,
                               offset_inplane=offset_inplane)

#############################################
# Initialize geometry for orthogonalization #
#############################################
qconv, offsets = pru.init_qconversion(setup)
detector.offsets = offsets
cch1 = 71.61 - detector.roi[0]  # take into account the roi if the image is cropped
cch2 = 1656.65 - detector.roi[2]  # take into account the roi if the image is cropped
hxrd = xu.experiment.HXRD(sample_inplane, sample_outofplane, qconv=qconv)  # x downstream, y outboard, z vertical
# first two arguments in HXRD are the inplane reference direction along the beam and surface normal of the sample
hxrd.Ang2Q.init_area('z-', 'y+', cch1=cch1, cch2=cch2, Nch1=detector.roi[1] - detector.roi[0],
                     Nch2=detector.roi[3] - detector.roi[2], pwidth1=detector.pixelsize,
                     pwidth2=detector.pixelsize, distance=sdd, detrot=detrot, tiltazimuth=tiltazimuth, tilt=tilt)
# first two arguments in init_area are the direction of the detector, checked for ID01 and SIXS

############################
# start looping over scans #
############################
root = tk.Tk()
root.withdraw()
if len(scans) > 1:
    if center_fft not in ['crop_asymmetric_ZYX', 'pad_Z', 'pad_asymmetric_ZYX']:
        center_fft = 'do_nothing'
        # avoid croping the detector plane XY while centering the Bragg peak
        # otherwise outputs may have a different size, which will be problematic for combining or comparing them
if rocking_angle == "energy":
    use_rawdata = False  # you need to interpolate the data in QxQyQz for energy scans
    print("Energy scan implemented only for ID01")

if not use_rawdata and setup.beamline == 'P10':
    print('Orthogonalization not yet implemented for P10')
    sys.exit()

for scan_nb in range(len(scans)):
    plt.ion()

    if setup.beamline != 'P10':
        homedir = root_folder + sample_name + str(scans[scan_nb]) + '/'
        detector.datadir = homedir + "data/"
    else:
        specfile_name = specfile_name % scans[scan_nb]
        homedir = root_folder + specfile_name + '/'
        detector.datadir = homedir + 'e4m/'
        template_imagefile = specfile_name + template_imagefile
        detector.template_imagefile = template_imagefile

    if not use_rawdata:
        comment = comment + '_ortho'
        savedir = homedir + "pynx/"
        pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)
    else:
        savedir = homedir + "pynxraw/"
        pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)

    print('\nScan', scans[scan_nb])
    print('Setup: ', setup.beamline)
    print('Detector: ', detector.name)
    print('Pixel Size: ', detector.pixelsize)
    print('Specfile: ', specfile_name)
    print('Scan type: ', setup.rocking_angle)

    if not use_rawdata:
        print('Output will be orthogonalized by xrayutilities')
        plot_title = ['QzQx', 'QyQx', 'QyQz']
        if beamline == 'P10':
            print('Gridder not yet implemented for P10')
            print('Switch to raw data instead of orthogonal data')
            use_rawdata = True
    else:
        print('Output will be non orthogonal, in the detector frame')
        plot_title = ['YZ', 'XZ', 'XY']

    if not fix_size:  # output_size not defined, default to actual size
        pass
    else:
        print("'fix_size' parameter provided, defaulting 'center_fft' to 'do_nothing'")
        center_fft = 'do_nothing'

    ####################################
    # Load data
    ####################################
    if reload_previous:  # resume previous masking
        print('Resuming previous masking')
        file_path = filedialog.askopenfilename(initialdir=homedir, title="Select data file",
                                               filetypes=[("NPZ", "*.npz")])
        data = np.load(file_path)
        npz_key = data.files
        data = data[npz_key[0]]
        file_path = filedialog.askopenfilename(initialdir=homedir, title="Select mask file",
                                               filetypes=[("NPZ", "*.npz")])
        mask = np.load(file_path)
        npz_key = mask.files
        mask = mask[npz_key[0]]

        q_values = []  # cannot orthogonalize since we do not know the original array size
        center_fft = 'do_nothing'  # we assume that crop/pad/centering was already performed
        frames_logical = np.ones(data.shape[0])  # we assume that all frames will be used
        fix_size = []  # we assume that crop/pad/centering was already performed
        normalize_flux = False  # we assume that normalization was already performed
        monitor = []  # we assume that normalization was already performed

        np.savez_compressed(savedir + 'S' + str(scans[scan_nb]) + '_pynx_previous' + comment, data=data)
        np.savez_compressed(savedir + 'S' + str(scans[scan_nb]) + '_maskpynx_previous', mask=mask)

    else:  # new masking process

        flatfield = pru.load_flatfield(flatfield_file)
        hotpix_array = pru.load_hotpixels(hotpixels_file)

        logfile = pru.create_logfile(setup=setup, detector=detector, scan_number=scans[scan_nb],
                                     root_folder=root_folder, filename=specfile_name)

        if use_rawdata:
            q_values, data, _, mask, _, frames_logical, monitor = \
                pru.gridmap(logfile=logfile, scan_number=scans[scan_nb], detector=detector, setup=setup,
                            flatfield=flatfield, hotpixels=hotpix_array, hxrd=None, header_cristal=header_cristal,
                            follow_bragg=follow_bragg, headerline_p10=headerline_p10)
        else:
            q_values, rawdata, data, _, mask, frames_logical, monitor = \
                pru.gridmap(logfile=logfile, scan_number=scans[scan_nb], detector=detector, setup=setup,
                            flatfield=flatfield, hotpixels=hotpix_array, hxrd=hxrd, header_cristal=header_cristal,
                            follow_bragg=follow_bragg, headerline_p10=headerline_p10)

            np.savez_compressed(savedir+'S'+str(scans[scan_nb])+'_rawdata_stack', data=rawdata)
            if save_to_mat:
                # save to .mat, x becomes z for Matlab phasing code
                savemat(savedir+'S'+str(scans[scan_nb])+'_rawdata_stack.mat',
                        {'data': np.moveaxis(rawdata, [0, 1, 2], [-1, -3, -2])})
            del rawdata
            gc.collect()

    ########################
    # crop/pad/center data #
    ########################
    nz, ny, nx = np.shape(data)
    print('Data size:', nz, ny, nx)

    data, mask, pad_width, q_vector, frames_logical = \
        pru.center_fft(data=data, mask=mask, frames_logical=frames_logical, centering=centering, fft_option=center_fft,
                       pad_size=pad_size, fix_bragg=fix_bragg, fix_size=fix_size, q_values=q_values)

    starting_frame = [pad_width[0], pad_width[2], pad_width[4]]  # no need to check padded frames
    print('Pad width:', pad_width)
    nz, ny, nx = data.shape
    print('Data size after cropping / padding:', nz, ny, nx)

    if mask_zero_event:
        # mask points when there is no intensity along the whole rocking curve - probably dead pixels
        for idx in range(nz):
            temp_mask = mask[idx, :, :]
            temp_mask[np.sum(data, axis=0) == 0] = 1  # enough, numpy array is mutable hence mask will be modified
        del temp_mask

    plt.ioff()

    ##############################
    # save the raw data and mask #
    ##############################
    fig, _, _ = gu.multislices_plot(data, sum_frames=True, scale='log', plot_colorbar=True, vmin=0,
                                    title='Data before aliens removal\n', invert_yaxis=False, reciprocal_space=True)
    plt.savefig(savedir + 'rawdata_S' + str(scans[scan_nb]) + '.png')

    if flag_interact:
        fig.waitforbuttonpress()
    plt.close(fig)

    fig, _, _ = gu.multislices_plot(mask, sum_frames=True, scale='linear', plot_colorbar=True, vmin=0,
                                    vmax=(nz, ny, nx), title='Mask before aliens removal\n', invert_yaxis=False,
                                    reciprocal_space=True)
    plt.savefig(savedir + 'rawmask_S' + str(scans[scan_nb]) + '.png')

    if flag_interact:
        fig.waitforbuttonpress()
    plt.close(fig)

    ###############################################
    # save the orthogonalized diffraction pattern #
    ###############################################
    if not use_rawdata:
        qx = q_vector[0]
        qz = q_vector[1]
        qy = q_vector[2]

        # save diffraction pattern to vti
        nqx, nqz, nqy = data.shape  # in nexus z downstream, y vertical / in q z vertical, x downstream
        print('dqx, dqy, dqz = ', qx[1] - qx[0], qy[1] - qy[0], qz[1] - qz[0])
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

        #############################################
        # remove aliens
        #############################################
        nz, ny, nx = np.shape(data)
        width = 5
        max_colorbar = 5
        flag_aliens = True

        # in XY
        dim = 0
        fig_mask = plt.figure()
        idx = starting_frame[0]
        original_data = np.copy(data)
        plt.imshow(data[idx, :, :], vmin=0, vmax=max_colorbar)
        plt.title("Frame " + str(idx+1) + "/" + str(nz) + "\n"
                  "m mask ; b unmask ; q quit ; u next frame ; d previous frame\n"
                  "up larger ; down smaller ; right darker ; left brighter")
        plt.connect('key_press_event', press_key)
        plt.show()
        del dim, fig_mask

        # in XZ
        dim = 1
        fig_mask = plt.figure()
        idx = starting_frame[1]
        plt.imshow(data[:, idx, :], vmin=0, vmax=max_colorbar)
        plt.title("Frame " + str(idx+1) + "/" + str(ny) + "\n"
                  "m mask ; b unmask ; q quit ; u next frame ; d previous frame\n"
                  "up larger ; down smaller ; right darker ; left brighter")
        plt.connect('key_press_event', press_key)
        plt.show()
        del dim, fig_mask

        # in YZ
        dim = 2
        fig_mask = plt.figure()
        idx = starting_frame[2]
        plt.imshow(data[:, :, idx], vmin=0, vmax=max_colorbar)
        plt.title("Frame " + str(idx+1) + "/" + str(nx) + "\n"
                  "m mask ; b unmask ; q quit ; u next frame ; d previous frame\n"
                  "up larger ; down smaller ; right darker ; left brighter")
        plt.connect('key_press_event', press_key)
        plt.show()

        del dim, width, fig_mask, original_data, flag_aliens

        fig, _, _ = gu.multislices_plot(data, sum_frames=True, scale='log', plot_colorbar=True, vmin=0,
                                        title='Data after aliens removal\n', invert_yaxis=False, reciprocal_space=True)

        if flag_interact:
            fig.waitforbuttonpress()
        plt.close(fig)

        fig, _, _ = gu.multislices_plot(mask, sum_frames=True, scale='linear', plot_colorbar=True, vmin=0,
                                        vmax=(nz, ny, nx), title='Mask after aliens removal\n', invert_yaxis=False,
                                        reciprocal_space=True)

        if flag_interact:
            fig.waitforbuttonpress()
        plt.close(fig)

        #############################################
        # define mask
        #############################################
        masked_color = 0.1  # will appear as -1 on the plot
        width = 0
        max_colorbar = 5
        flag_aliens = False
        flag_mask = True
        flag_pause = False  # press x to pause for pan/zoom

        nz, ny, nx = np.shape(data)
        original_data = np.copy(data)

        # in XY
        dim = 0
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        x, y = x.flatten(), y.flatten()
        points = np.stack((x, y), axis=0).T
        xy = []  # list of points for mask
        temp_mask = np.zeros((ny, nx))
        data[mask == 1] = masked_color / nz  # will appear as -1 on the plot
        print('Select vertices of mask. Press a to restart;p to plot; q to quit.')
        fig_mask = plt.figure()
        plt.imshow(np.log10(abs(data.sum(axis=0))), vmin=0, vmax=max_colorbar)
        plt.title('x to pause/resume masking for pan/zoom \n'
                  'p plot mask ; a restart ; click to select vertices\n'
                  "m mask ; b unmask ; q quit ; u next frame ; d previous frame\n"
                  "up larger ; down smaller ; right darker ; left brighter")
        plt.connect('key_press_event', press_key)
        plt.connect('button_press_event', on_click)
        plt.show()
        data = np.copy(original_data)

        for idx in range(nz):
            temp_array = mask[idx, :, :]
            temp_array[np.nonzero(temp_mask)] = 1  # enough, numpy array is mutable hence mask will be modified
        del temp_mask

        # in XZ
        dim = 1
        flag_pause = False  # press x to pause for pan/zoom
        x, y = np.meshgrid(np.arange(nx), np.arange(nz))
        x, y = x.flatten(), y.flatten()
        points = np.stack((x, y), axis=0).T
        xy = []  # list of points for mask
        temp_mask = np.zeros((nz, nx))
        data[mask == 1] = masked_color / ny  # will appear as -1 on the plot
        print('Select vertices of mask. Press a to restart;p to plot; q to quit.')
        fig_mask = plt.figure()
        plt.imshow(np.log10(abs(data.sum(axis=1))), vmin=0, vmax=max_colorbar)
        plt.title('x to pause/resume masking for pan/zoom \n'
                  'p plot mask ; a restart ; click to select vertices\n'
                  "m mask ; b unmask ; q quit ; u next frame ; d previous frame\n"
                  "up larger ; down smaller ; right darker ; left brighter")
        plt.connect('key_press_event', press_key)
        plt.connect('button_press_event', on_click)
        plt.show()
        data = np.copy(original_data)

        for idx in range(ny):
            temp_array = mask[:, idx, :]
            temp_array[np.nonzero(temp_mask)] = 1  # enough, numpy array is mutable hence mask will be modified
        del temp_mask

        # in YZ
        dim = 2
        flag_pause = False  # press x to pause for pan/zoom
        x, y = np.meshgrid(np.arange(ny), np.arange(nz))
        x, y = x.flatten(), y.flatten()
        points = np.stack((x, y), axis=0).T
        xy = []  # list of points for mask
        temp_mask = np.zeros((nz, ny))
        data[mask == 1] = masked_color / nx  # will appear as -1 on the plot
        print('Select vertices of mask. Press a to restart;p to plot; q to quit.')
        fig_mask = plt.figure()
        plt.imshow(np.log10(abs(data.sum(axis=2))), vmin=0, vmax=max_colorbar)
        plt.title('x to pause/resume masking for pan/zoom \n'
                  'p plot mask ; a restart ; click to select vertices\n'
                  "m mask ; b unmask ; q quit ; u next frame ; d previous frame\n"
                  "up larger ; down smaller ; right darker ; left brighter")
        plt.connect('key_press_event', press_key)
        plt.connect('button_press_event', on_click)
        plt.show()

        for idx in range(nx):
            temp_array = mask[:, :, idx]
            temp_array[np.nonzero(temp_mask)] = 1  # enough, numpy array is mutable hence mask will be modified
        del temp_mask, dim

        data = original_data
        del original_data, flag_aliens, flag_mask, flag_pause

    data[mask == 1] = 0

    #################################
    # normalize by incident monitor #
    #################################
    if normalize_flux:
        data, monitor, monitor_title = pru.normalize_dataset(array=data, raw_monitor=monitor,
                                                             frames_logical=frames_logical,
                                                             norm_to_min=True, debugging=False)
        plt.ion()
        fig = gu.combined_plots(tuple_array=(monitor, data), tuple_sum_frames=(False, True),
                                tuple_sum_axis=(0, 1), tuple_width_v=(np.nan, np.nan),
                                tuple_width_h=(np.nan, np.nan), tuple_colorbar=(False, False),
                                tuple_vmin=(np.nan, 0), tuple_vmax=(np.nan, np.nan),
                                tuple_title=('monitor.min() / monitor', 'Data after normalization'),
                                tuple_scale=('linear', 'log'), xlabel=('Frame number', 'Frame number'),
                                ylabel=('Counts (a.u.)', 'Rocking dimension'))

        fig.savefig(savedir + 'monitor_S' + str(scans[scan_nb]) + '.png')
        if flag_interact:
            fig.waitforbuttonpress()
        plt.close(fig)
        plt.ioff()
        comment = comment + '_norm'

    #############################################
    # mask or median filter isolated empty pixels
    #############################################
    if flag_medianfilter == 'mask_isolated' or flag_medianfilter == 'interp_isolated':
        nb_pix = 0
        for idx in range(pad_width[0], nz-pad_width[1]):  # filter only frames whith data (not padded)
            data[idx, :, :], numb_pix, mask[idx, :, :] = \
                pru.mean_filter(data=data[idx, :, :], nb_neighbours=medfilt_order, mask=mask[idx, :, :],
                                interpolate=flag_medianfilter, debugging=False)
            nb_pix = nb_pix + numb_pix
            print("Processed image nb: ", idx)
        if flag_medianfilter == 'mask_isolated':
            print("Total number of masked isolated pixels: ", nb_pix)
        if flag_medianfilter == 'interp_isolated':
            print("Total number of interpolated isolated pixels: ", nb_pix)

    elif flag_medianfilter == 'median':  # apply median filter
        for idx in range(pad_width[0], nz-pad_width[1]):  # filter only frames whith data (not padded)
            data[idx, :, :] = scipy.signal.medfilt2d(data[idx, :, :], [3, 3])
        print("Applying median filtering")
    else:
        print("Skipping median filtering")

    #############################################
    # apply photon threshold
    #############################################
    if photon_threshold != 0:
        mask[np.nonzero(data) == photon_threshold] = 1
        data[data <= photon_threshold] = 0
        print("Applying photon threshold")

    #############################################
    # save prepared data and mask
    #############################################
    plt.ion()
    nz, ny, nx = np.shape(data)
    print('Data size after cropping / padding:', nz, ny, nx)
    comment = comment + "_" + str(nz) + "_" + str(ny) + "_" + str(nx)  # need these numbers to calculate the voxel size

    # check for Nan
    mask[np.isnan(data)] = 1
    data[np.isnan(data)] = 0
    mask[np.isnan(mask)] = 1
    # check for Inf
    mask[np.isinf(data)] = 1
    data[np.isinf(data)] = 0
    mask[np.isinf(mask)] = 1

    data[mask == 1] = 0

    if not use_rawdata:
        np.savez_compressed(savedir + 'QxQzQy_S' + str(scans[scan_nb]) + comment,
                            qx=q_vector[0], qz=q_vector[1], qy=q_vector[2])
    np.savez_compressed(savedir + 'S' + str(scans[scan_nb]) + '_pynx' + comment, data=data)
    np.savez_compressed(savedir + 'S' + str(scans[scan_nb]) + '_maskpynx' + comment, mask=mask)

    if save_to_mat:
        # save to .mat, x becomes z for Matlab phasing code
        savemat(savedir + 'S' + str(scans[scan_nb]) + '_data.mat',
                {'data': np.moveaxis(data, [0, 1, 2], [-1, -3, -2])})
        savemat(savedir + 'S' + str(scans[scan_nb]) + '_mask.mat',
                {'data': np.moveaxis(mask, [0, 1, 2], [-1, -3, -2])})

    ###################################
    # plot the prepared data and mask #
    ###################################
    if center_fft in ['crop_symmetric_ZYX', 'pad_symmetric_ZYX']:
        # in other cases the diffraction pattern will not be centered
        fig, _, _ = gu.multislices_plot(data, sum_frames=False, scale='log', plot_colorbar=True, vmin=0,
                                        title='Masked data', invert_yaxis=False, reciprocal_space=True)
        plt.savefig(savedir + 'middle_frame_S' + str(scans[scan_nb]) + comment + '.png')
        if not flag_interact:
            plt.close(fig)

    fig, _, _ = gu.multislices_plot(data, sum_frames=True, scale='log', plot_colorbar=True, vmin=0, title='Masked data',
                                    invert_yaxis=False, reciprocal_space=True)
    plt.savefig(savedir + 'sum_S' + str(scans[scan_nb]) + comment + '.png')
    if not flag_interact:
        plt.close(fig)

    fig, _, _ = gu.multislices_plot(mask, sum_frames=True, scale='linear', plot_colorbar=True, vmin=0,
                                    vmax=(nz, ny, nx), title='Mask', invert_yaxis=False, reciprocal_space=True)
    plt.savefig(savedir + 'mask_S' + str(scans[scan_nb]) + comment + '.png')
    if not flag_interact:
        plt.close(fig)

plt.ioff()
plt.show()
