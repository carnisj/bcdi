# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

"""
Definition of the correct parameters for the config files.

Parameter validation is performed over the excepted parameters.
"""
import numpy as np
import bcdi.utils.validation as valid


def valid_param(key, value):
    if key == "scans":
        valid.valid_container(
            value,
            container_types=(tuple, list, np.ndarray),
            min_length=1,
            name=key
        )
    elif key == "scan":
        valid.valid_item(
            value,
            allowed_types=int,
            min_included=0,
            name=key
        )
    # here we will list all the possible parameters used in scripts (we need to unify
    # as much as possible the names)
    else:
        # this key is not in the known parameters
        return False
    return True
