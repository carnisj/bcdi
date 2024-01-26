# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, jerome.carnis@esrf.fr
"""Constants used throughout the package."""

import numpy as np

AXIS_TO_ARRAY = {
    "x": np.array([1, 0, 0]),
    "y": np.array([0, 1, 0]),
    "z": np.array([0, 0, 1]),
}  # in xyz order


BEAMLINES_BCDI = [
    "BM02",
    "ID01",
    "ID01BLISS",
    "SIXS_2018",
    "SIXS_2019",
    "34ID",
    "P10",
    "CRISTAL",
    "NANOMAX",
]
BEAMLINES_SAXS = ["P10_SAXS", "ID27"]
