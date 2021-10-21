# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#   (c) 06/2021-present : DESY CFEL
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

from pathlib import Path
import unittest
from bcdi.utils.parameters import valid_param

here = Path(__file__).parent
CONFIG = str(here.parents[1] / "conf/default_config.yml")


def run_tests(test_class):
    suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


class TestParameters(unittest.TestCase):
    """
    Tests on the function valid_param.

    def __init__(self, file_path : str, script_type : str = "preprocessing") -> None :
    """
    pass


if __name__ == "__main__":
    run_tests(TestParameters)
