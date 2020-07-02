# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

helptext = """
Open images or series data at P10 beamline.
"""

import hdf5plugin  # for lz4 filter
import h5py
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('D:/myscripts/bcdi/')
import bcdi.graph.graph_utils as gu
import bcdi.experiment.experiment_utils as exp
import bcdi.preprocessing.preprocessing_utils as pru

scan = 38  # scan number as it appears in the folder name
sample_name = "p15"  # without _ at the end
root_folder = "D:/data/P10_isosurface/data/"
savedir = ''  # images will be saved here, leave it to '' otherwise (default to data directory's parent)
###########################
# mesh related parameters #
###########################
fast_motor = 'hpy'  # fast scanning motor for the mesh
slow_motor = 'hpx'  # slow scanning motor for the mesh
###############################
# beamline related parameters #
###############################
beamline = 'P10'  # name of the beamlisne, used for data loading and normalization by monitor
# supported beamlines: 'ID01', 'SIXS_2018', 'SIXS_2019', 'CRISTAL', 'P10'
is_series = True  # specific to series measurement at P10
specfile_name = ''
# .spec for ID01, .fio for P10, alias_dict.txt for SIXS_2018, not used for CRISTAL and SIXS_2019
# template for ID01: name of the spec file without '.spec'
# template for SIXS_2018: full path of the alias dictionnary 'alias_dict.txt', typically: root_folder + 'alias_dict.txt'
# template for SIXS_2019: ''
# template for P10: ''
# template for CRISTAL: ''
###############################
# detector related parameters #
###############################
detector = "Eiger4M"    # "Eiger2M" or "Maxipix" or "Eiger4M"
template_imagefile = '_master.h5'
# template for ID01: 'data_mpx4_%05d.edf.gz' or 'align_eiger2M_%05d.edf.gz'
# template for SIXS_2018: 'align.spec_ascan_mu_%05d.nxs'
# template for SIXS_2019: 'spare_ascan_mu_%05d.nxs'
# template for Cristal: 'S%d.nxs'
# template for P10: '_master.h5'
##########################
# end of user parameters #
##########################

#################################################
# Initialize detector, setup, paths and logfile #
#################################################
detector = exp.Detector(name=detector, datadir='', template_imagefile=template_imagefile, is_series=is_series)
nb_pixel_y, nb_pixel_x = detector.nb_pixel_y, detector.nb_pixel_x
setup = exp.SetupPreprocessing(beamline=beamline)

if setup.beamline == 'P10':
    specfile_name = sample_name + '_{:05d}'.format(scan)
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

if savedir == '':
    savedir = os.path.abspath(os.path.join(detector.datadir, os.pardir)) + '/'

logfile = pru.create_logfile(setup=setup, detector=detector, scan_number=scan, root_folder=root_folder,
                             filename=specfile_name)

#############
# Load data #
#############
data, mask, monitor, frames_logical = pru.load_data(logfile=logfile, scan_number=scan, detector=detector,
                                                    setup=setup, debugging=False)
numz, numy, numx = data.shape
print('Data shape: ', numz, numy, numx)

########################
# Load motor positions #
########################
fast_positions = pru.get_motor_pos(logfile=logfile, scan_number=scan, setup=setup, motor_name=fast_motor)
slow_positions = pru.get_motor_pos(logfile=logfile, scan_number=scan, setup=setup, motor_name=slow_motor)

####################
# interactive plot #
####################
counter_roi = [0, nb_pixel_y, 0, nb_pixel_x]
plt.ioff()

plt.show()


