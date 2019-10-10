# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import hdf5plugin  # for P10, should be imported before h5py or PyTables
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import sys
sys.path.append('//win.desy.de/home/carnisj/My Documents/myscripts/bcdi/')
import bcdi.experiment.experiment_utils as exp
import bcdi.preprocessing.preprocessing_utils as pru
import bcdi.graph.graph_utils as gu

helptext = """
Open a rocking curve data, plot the mask, the monitor and the stack along the first axis.

It is usefull when you want to localize the Bragg peak for ROI determination.

Supported beamlines: ESRF ID01, PETRAIII P10, SOLEIL SIXS, SOLEIL CRISTAL.
"""

scan = 936
root_folder = "D:/data/HC3207/"
sample_name = "SN"  # "S"
save_mask = False  # set to True to save the mask
fit_rockingcurve = False  # set to True if you want a fit of the rocking curve
######################################
# define beamline related parameters #
######################################
beamline = 'ID01'  # name of the beamlis[scan_nb]ne, used for data loading and normalization by monitor
# supported beamlines: 'ID01', 'SIXS_2018', 'SIXS_2019', 'CRISTAL', 'P10'
rocking_angle = "outofplane"  # "outofplane" or "inplane"
specfile_name = 'align2'
# .spec for ID01, .fio for P10, alias_dict.txt for SIXS_2018, not used for CRISTAL and SIXS_2019
# template for ID01: name of the spec file without '.spec'
# template for SIXS_2018: full path of the alias dictionnary 'alias_dict.txt', typically: root_folder + 'alias_dict.txt'
# template for SIXS_2019: ''
# template for P10: sample_name + '_%05d'
# template for CRISTAL: ''
#############################################################
# define detector related parameters and region of interest #
#############################################################
detector = "Eiger2M"    # "Eiger2M" or "Maxipix" or "Eiger4M"
bragg_position = []  # Bragg peak position [vertical, horizontal], leave it as [] if there is a single peak
peak_method = 'max'  # Bragg peak determination: 'max', 'com' or 'maxcom'.
hotpixels_file = ''  # root_folder + 'hotpixels.npz'  #
flatfield_file = ''  # root_folder + "flatfield_8.5kev.npz"  #
template_imagefile = 'align_eiger2M_%05d.edf.gz'
# template for ID01: 'data_mpx4_%05d.edf.gz' or 'align_eiger2M_%05d.edf.gz'
# template for SIXS_2018: 'align.spec_ascan_mu_%05d.nxs'
# template for SIXS_2019: 'spare_ascan_mu_%05d.nxs'
# template for Cristal: 'S%d.nxs'
# template for P10: '_data_%06d.h5'
##################################
# end of user-defined parameters #
##################################

#################################################
# Initialize detector, setup, paths and logfile #
#################################################
detector = exp.Detector(name=detector, datadir='', template_imagefile=template_imagefile)

setup = exp.SetupPreprocessing(beamline=beamline, rocking_angle=rocking_angle)

if setup.beamline == 'P10':
    specfile_name = specfile_name % scan
    homedir = root_folder + specfile_name + '/'
    detector.datadir = homedir + 'e4m/'
    template_imagefile = specfile_name + template_imagefile
    detector.template_imagefile = template_imagefile
elif setup.beamline == 'SIXS_2018' or setup.beamline == 'SIXS_2019':
    homedir = root_folder 
    detector.datadir = homedir + "align/"
else:
    homedir = root_folder + sample_name + str(scan) + '/'
    detector.datadir = homedir + "data/"

flatfield = pru.load_flatfield(flatfield_file)
hotpix_array = pru.load_hotpixels(hotpixels_file)

logfile = pru.create_logfile(setup=setup, detector=detector, scan_number=scan, root_folder=root_folder,
                             filename=specfile_name)

data, mask, monitor, frames_logical = pru.load_data(logfile=logfile, scan_number=scan, detector=detector,
                                                    beamline=beamline, flatfield=flatfield, hotpixels=hotpix_array,
                                                    debugging=False)

numz, numy, numx = data.shape
print('Data shape: ', numz, numy, numx)

######################################################
# calculate rocking curve and fit it to get the FWHM #
######################################################
if data.ndim == 3 and fit_rockingcurve:
    tilt, _, _, _ = pru.motor_values(frames_logical=frames_logical, logfile=logfile, scan_number=scan, setup=setup,
                                     follow_bragg=False)
    rocking_curve = np.zeros(numz)

    z0, y0, x0 = pru.find_bragg(data, peak_method=peak_method)
    z0 = np.rint(z0).astype(int)
    y0 = np.rint(y0).astype(int)
    x0 = np.rint(x0).astype(int)

    if len(bragg_position) == 0:  # Bragg peak position not defined by te user, find the max
        bragg_position.append(y0)
        bragg_position.append(x0)

    print("Bragg peak (full detector) at (z, y, x): ", z0, y0, x0)

    for idx in range(numz):
        rocking_curve[idx] = data[idx, bragg_position[0] - 50:bragg_position[0] + 50,
                                  bragg_position[1] - 50:bragg_position[1] + 50].sum()
    plot_title = "Rocking curve for a ROI centered on (y, x): " + str(bragg_position[0]) + ',' + str(bragg_position[1])

    z0 = np.unravel_index(rocking_curve.argmax(), rocking_curve.shape)[0]

    interpolation = interp1d(tilt, rocking_curve, kind='cubic')
    interp_points = 5 * numz
    interp_tilt = np.linspace(tilt.min(), tilt.max(), interp_points)
    interp_curve = interpolation(interp_tilt)
    interp_fwhm = len(np.argwhere(interp_curve >= interp_curve.max() / 2)) * \
                  (tilt.max() - tilt.min()) / (interp_points - 1)
    print('FWHM by interpolation', str('{:.3f}'.format(interp_fwhm)), 'deg')

    fig, (ax0, ax1) = plt.subplots(2, 1, sharex='col', figsize=(10, 5))
    ax0.plot(tilt, rocking_curve, '.')
    ax0.plot(interp_tilt, interp_curve)
    ax0.set_ylabel('Integrated intensity')
    ax0.legend(('data', 'interpolation'))
    ax0.set_title(plot_title)
    ax1.plot(tilt, np.log10(rocking_curve), '.')
    ax1.plot(interp_tilt, np.log10(interp_curve))
    ax1.set_xlabel('Rocking angle (deg)')
    ax1.set_ylabel('Log(integrated intensity)')
    ax0.legend(('data', 'interpolation'))
    plt.pause(0.1)

############################################
# plot mask, monitor and concatenated data #
############################################
#data = data[data.shape[0]//2, :, :]  # select the first frame e.g. for detector mesh scan
data = data.sum(axis=0)  # concatenate along the axis of the rocking curve

if save_mask:
    np.savez_compressed(detector.datadir+'hotpixels.npz', mask=mask)

gu.combined_plots(tuple_array=(monitor, mask), tuple_sum_frames=False, tuple_sum_axis=(0, 0),
                  tuple_width_v=np.nan, tuple_width_h=np.nan, tuple_colorbar=(True, False), tuple_vmin=np.nan,
                  tuple_vmax=np.nan, tuple_title=('monitor', 'mask'), tuple_scale='linear',
                  ylabel=('Counts (a.u.)', ''))

y0, x0 = np.unravel_index(abs(data).argmax(), data.shape)
print("Max at (y, x): ", y0, x0, ' Max = ', int(data[y0, x0]))

fig = plt.figure()
plt.imshow(np.log10(data), vmin=0)
plt.title('data.sum(axis=0)\nMax at (y, x): (' + str(y0) + ',' + str(x0) + ')   Max = ' + str(int(data[y0, x0])))
plt.colorbar()
plt.savefig(detector.datadir + 'sum_S' + str(scan) + '.png')
plt.show()
