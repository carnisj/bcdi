# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import unittest
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import matplotlib

from bcdi.postprocessing.postprocessing_runner import initialize_parameters
from bcdi.preprocessing.preprocessing_runner import initialize_parameters_bcdi
from bcdi.utils.parser import ConfigParser


def has_backend(backend: str) -> bool:
    """Check if the desired backend is available on the runner."""
    try:
        matplotlib.use(backend)
    except ImportError:
        return False
    return True


def get_config(workflow) -> Tuple[str, Callable]:
    if workflow == "preprocessing":
        return "bcdi/examples/S11_config_preprocessing.yml", initialize_parameters_bcdi
    if workflow == "postprocessing":
        return "bcdi/examples/S11_config_postprocessing.yml", initialize_parameters
    raise NotImplementedError(f"workflow {workflow} not implemented")


def load_config(workflow: str) -> Tuple[Dict[str, Any], bool]:
    allowed_workflows = {"preprocessing", "postprocessing"}
    if workflow not in allowed_workflows:
        raise ValueError(
            f"Unknown worklow '{workflow}', allowed is " f"{allowed_workflows}."
        )

    filename, function = get_config(workflow)
    here = Path(__file__).parent

    config = str(here.parents[0] / filename)
    try:
        parameters = function(ConfigParser(config).load_arguments())
        parameters.update({"backend": "agg"})
        matplotlib.use(parameters["backend"])
        return parameters, False
    except ValueError:
        return {}, True


def run_tests(test_class):
    suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)
