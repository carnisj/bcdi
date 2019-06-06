# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#       authors:
#         Jerome Carnis, jerome.carnis@esrf.fr

import hdf5plugin  # for lz4 filter
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('C:\\Users\\carnis\\Work Folders\\Documents\\myscripts\\bcdi\\')
import bcdi.experiment.experiment_utils as exp
import bcdi.preprocessing.preprocessing_utils as pru
import bcdi.graph.graph_utils as gu

helptext = """
Open a rocking curve data, plot the mask, the monitor and the stack along the first axis.

It is usefull when you want to localize the Bragg peak for ROI determination.

Supported beamlines: ESRF ID01, PETRAIII P10, SOLEIL SIXS, SOLEIL CRISTAL.
"""

scan = 1351
root_folder = "C:/Users/carnis/Work Folders/Documents/data/P10_2019/"
sample_name = "align_02"  # "S"
save_mask = False  # set to True to save the mask
######################################
# define beamline related parameters #
######################################
beamline = 'P10'  # 'ID01' or 'SIXS' or 'CRISTAL' or 'P10', used for data loading and normalization by monitor
specfile_name = sample_name + '_%05d'
# .spec for ID01, .fio for P10, alias_dict.txt for SIXS, not used for CRISTAL
# template for ID01: name of the spec file without '.spec'
# template for SIXS: full path of the alias dictionnary 'alias_dict.txt', typically root_folder + 'alias_dict.txt'
# template for P10: sample_name + '_%05d'
#############################################################
# define detector related parameters and region of interest #
#############################################################
detector = "Eiger4M"    # "Eiger2M" or "Maxipix" or "Eiger4M"
x_bragg = 1495  # horizontal pixel number of the Bragg peak
# roi_detector = [1202, 1610, x_bragg - 256, x_bragg + 256]  # HC3207  x_bragg = 430
roi_detector = [552, 1064, x_bragg - 240, x_bragg + 240]  # P10 2018
# roi_detector = []
# leave it as [] to use the full detector. Use with center_fft='do_nothing' if you want this exact size.
hotpixels_file = ''  # root_folder + 'hotpixels.npz'  #
flatfield_file = ''  # root_folder + "flatfield_eiger.npz"  #
template_imagefile = '_data_%06d.h5'
# ID01: 'data_mpx4_%05d.edf.gz' or 'align_eiger2M_%05d.edf.gz'
# SIXS: 'align.spec_ascan_mu_%05d.nxs'
# Cristal: 'S%d.nxs'
# P10: '_data_%06d.h5'
##################################
# end of user-defined parameters #
##################################

#################################################
# Initialize paths, detector, setup and logfile #
#################################################
detector = exp.Detector(name=detector, datadir='', template_imagefile=template_imagefile)

if beamline != 'P10':
    homedir = root_folder + sample_name + str(scan) + '/'
    detector.datadir = homedir + "data/"
else:
    specfile_name = specfile_name % scan
    homedir = root_folder + specfile_name + '/'
    detector.datadir = homedir + 'e4m/'
    template_imagefile = specfile_name + template_imagefile
    detector.template_imagefile = template_imagefile

flatfield = pru.load_flatfield(flatfield_file)
hotpix_array = pru.load_hotpixels(hotpixels_file)

logfile = pru.create_logfile(beamline=beamline, detector=detector, scan_number=scan, root_folder=root_folder,
                             filename=specfile_name)

data, mask, monitor, _ = pru.load_data(logfile=logfile, scan_number=scan, detector=detector, beamline=beamline,
                                       flatfield=flatfield, hotpixels=hotpix_array, debugging=False)

print(data.shape)

if data.ndim == 3:
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
