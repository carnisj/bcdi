# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import os
import h5py
import numpy as np


def load_file(file_path, fieldname):
    """
    Load a file.

    :param file_path: the path of the reconstruction to load. Format supported: .npy .npz .cxi .h5
    :param fieldname: the name of the field to be loaded
    :return: the loaded data and the extension of the file
    """
    _, extension = os.path.splitext(file_path)
    if extension == '.npz':  # could be anything
        dataset = np.load(file_path)[fieldname]
    elif extension == '.npy':  # could be anything
        dataset = np.load(file_path)
    elif extension == '.cxi':  # output of PyNX phasing
        h5file = h5py.File(file_path, 'r')
        group_key = list(h5file.keys())[1]
        subgroup_key = list(h5file[group_key])
        dataset = h5file['/'+group_key+'/'+subgroup_key[0]+'/data'].value
    elif extension == '.h5':  # modes.h5
        h5file = h5py.File(file_path, 'r')
        group_key = list(h5file.keys())[0]
        subgroup_key = list(h5file[group_key])
        dataset = h5file['/' + group_key + '/' + subgroup_key[0] + '/data'][0]  # select only first mode
    else:
        raise ValueError("File format not supported: can load only '.npy', '.npz', '.cxi' or '.h5' files")
    return dataset, extension
