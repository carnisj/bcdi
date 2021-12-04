# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

"""
Implementation of the Dataset class.

This class stores data and optionally other arrays such as the mask, the optical path...
"""
from numbers import Real
from typing import List, Optional, Tuple, Union

import bcdi.preprocessing.bcdi_utils as bu


class Dataset:
    """


    """

    def __init__(
            self,
            data,
            mask=None,
            original_size=None
    ):
        self.data = data
        self.mask = mask
        self.original_size = original_size

    def find_bragg_peak(
            self, binning=(1, 1, 1), direct_space=False, peak_method="maxcom",
    ) -> Tuple[Real, Real, Real]:
        if direct_space:
            # need to take the FFT of the full size object and unbin it
            pass
        return bu.find_bragg(reciprocal_data, peak_method=peak_method)