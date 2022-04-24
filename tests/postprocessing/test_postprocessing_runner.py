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
        self.args = self.parser.load_arguments()
        if not Path(self.args["root_folder"]).is_dir():
            self.skipTest(
                reason="This test can only run locally with the example dataset"
            )

    def test_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.args["save_dir"] = (tmpdir,)
            run(self.args)

            with h5py.File(
                f"{tmpdir}/S11_ampdispstrain_mode_crystalframe.h5",
                "r",
            ) as h5file:
                amp = h5file["output/amp"][:]
                voxel_sizes = h5file["output/voxel_sizes"][:]
        amp = amp / amp.max()
        amp[amp < self.args["isosurface_strain"]] = 0
        amp[np.nonzero(amp)] = 1
        volume = amp.sum() * reduce(lambda x, y: x * y, voxel_sizes)

        self.assertEqual(volume, 23217408)


if __name__ == "__main__":
    run_tests(TestRun)
