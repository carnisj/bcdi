# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

from functools import reduce
import h5py
import numpy as np
from pathlib import Path
import tempfile
import unittest
import yaml

from bcdi.postprocessing.postprocessing_runner import run
from bcdi.utils.parser import ConfigParser
from tests.config import run_tests

here = Path(__file__).parent
THIS_DIR = str(here)
CONFIG = str(here.parents[1] / "bcdi/examples/S11_config_postprocessing.yml")


class TestRun(unittest.TestCase):
    """Large test for the postprocessing script bcdi_strain.py."""

    def setUp(self) -> None:
        self.command_line_args = {
            "backend": "Agg",
            "reconstruction_files": str(
                here.parents[1]
                / "bcdi/examples/S11_modes_252_420_392_prebinning_1_1_1.h5"
            ),
        }
        self.parser = ConfigParser(CONFIG, self.command_line_args)
        if not Path(
            yaml.load(self.parser.raw_config, Loader=yaml.SafeLoader)["root_folder"]
        ).is_dir():
            self.skipTest(
                reason="This test can only run locally with the example dataset"
            )

    def test_run(self):
        expected_q_com = [0, 2.77555, 0]
        expected_volume = 23217408
        with tempfile.TemporaryDirectory() as tmpdir:
            self.args = self.parser.load_arguments()
            self.args["save_dir"] = (tmpdir,)
            run(self.args)

            with h5py.File(
                f"{tmpdir}/S11_ampdispstrain_mode_crystalframe.h5",
                "r",
            ) as h5file:
                amp = h5file["output/amp"][:]
                voxel_sizes = h5file["output/voxel_sizes"][:]
                q_com = h5file["output/q_com"][:]
        amp = amp / amp.max()
        amp[amp < self.args["isosurface_strain"]] = 0
        amp[np.nonzero(amp)] = 1
        volume = amp.sum() * reduce(lambda x, y: x * y, voxel_sizes)

        self.assertEqual(volume, expected_volume)
        self.assertTrue(np.allclose(q_com, expected_q_com))


if __name__ == "__main__":
    run_tests(TestRun)
