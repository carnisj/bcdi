# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, jerome.carnis@esrf.fr
"""Workflow for CDI data preprocessing of a single scan, before phase retrieval.

The detector is expected to be fixed, its plane being always perpendicular to the direct
beam independently of the detector position.
"""

import gc

try:
    import hdf5plugin  # for P10, should be imported before h5py or PyTables
except ModuleNotFoundError:
    pass

import logging
import os
import tkinter as tk
from logging import Logger
from pathlib import Path
from tkinter import filedialog
from typing import Any, Dict, List, Optional, Tuple

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal  # for medfilt2d
import xrayutilities as xu
from scipy.io import savemat
from scipy.ndimage.measurements import center_of_mass

import bcdi.graph.graph_utils as gu
import bcdi.postprocessing.postprocessing_utils as pu
import bcdi.preprocessing.bcdi_utils as bu
import bcdi.utils.utilities as util
import bcdi.utils.validation as valid
from bcdi.experiment.setup import Setup
from bcdi.utils.constants import AXIS_TO_ARRAY
from bcdi.utils.snippets_logging import FILE_FORMATTER

logger = logging.getLogger(__name__)


def process_scan_cdi(
    scan_idx: int, prm: Dict[str, Any]
) -> Tuple[Path, Path, Optional[Logger]]:
    pass