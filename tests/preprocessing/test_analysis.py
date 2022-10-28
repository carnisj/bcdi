# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr
import os.path
import tempfile
import unittest
from copy import deepcopy
from logging import Logger
from pathlib import Path
from unittest.mock import patch

import matplotlib
import numpy as np

import bcdi.preprocessing.analysis as analysis
from bcdi.experiment.setup import Setup
from bcdi.preprocessing.preprocessing_runner import initialize_parameters_bcdi
from bcdi.utils.parser import ConfigParser
from tests.config import run_tests

here = Path(__file__).parent
THIS_DIR = str(here)
CONFIG = str(here.parents[1] / "bcdi/examples/S11_config_preprocessing.yml")
try:
    parameters = initialize_parameters_bcdi(ConfigParser(CONFIG).load_arguments())
    parameters.update({"backend": "agg"})
    matplotlib.use(parameters["backend"])
    skip_tests = False
except ValueError:
    skip_tests = True
