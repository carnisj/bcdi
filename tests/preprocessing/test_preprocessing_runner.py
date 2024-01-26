# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import os
import tempfile
import unittest
from pathlib import Path
from typing import Dict, Optional

import h5py
import numpy as np
import yaml

from bcdi.preprocessing.preprocessing_runner import run
from bcdi.utils.parser import ConfigParser
from tests.config import run_tests

here = Path(__file__).parent
THIS_DIR = str(here)
CONFIG = str(here.parents[1] / "bcdi/examples/S11_config_preprocessing.yml")


class TestRun(unittest.TestCase):
    """Large test for the preprocessing script bcdi_preprocessing_BCDI.py."""

    def setUp(self) -> None:
        self.args: Optional[Dict] = None
        self.command_line_args = {
            "backend": "Agg",
            "flag_interact": False,
        }
        self.parser = ConfigParser(CONFIG, self.command_line_args)
        if not Path(
            yaml.load(self.parser.raw_config, Loader=yaml.SafeLoader)["root_folder"]
        ).is_dir():
            self.skipTest(
                reason="This test can only run locally with the example dataset"
            )

    def test_run(self):
        expected_bragg_inplane = 0.4926488901267543
        expected_bragg_outofplane = 35.36269069963432
        expected_bragg_peak = [127, 214, 316]
        expected_q = [-0.84164063, 2.63974482, -0.03198209]
        with tempfile.TemporaryDirectory() as tmpdir:
            self.args = self.parser.load_arguments()
            self.args["save_dir"] = [tmpdir]
            run(self.args)
            self.assertTrue(os.path.isfile(f"{tmpdir}/preprocessing_run0_S11.log"))
            with h5py.File(
                f"{tmpdir}/S11_preprocessing_norm_256_256_256_1_2_2.h5",
                "r",
            ) as h5file:
                bragg_inplane = h5file["output/bragg_inplane"][()]
                bragg_outofplane = h5file["output/bragg_outofplane"][()]
                bragg_peak = h5file["output/bragg_peak"][()]
                q = h5file["output/q_bragg"][()]

        self.assertAlmostEqual(bragg_inplane, expected_bragg_inplane)
        self.assertAlmostEqual(bragg_outofplane, expected_bragg_outofplane)
        self.assertTrue(
            val1 == val2 for val1, val2 in zip(bragg_peak, expected_bragg_peak)
        )
        self.assertTrue(np.allclose(q, expected_q))


if __name__ == "__main__":
    run_tests(TestRun)
