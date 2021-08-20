# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#   (c) 06/2021-present : DESY CFEL
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np
import unittest
from bcdi.experiment.beamline import create_beamline, Beamline


def run_tests(test_class):
    suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


class TestBeamline(unittest.TestCase):
    """Tests related to beamline instantiation."""

    def test_create_beamline_from_abc(self):
        with self.assertRaises(TypeError):
            Beamline(name="ID01")


class TestBeamlineCRISTAL(unittest.TestCase):
    """Tests related to CRISTAL beamline instantiation."""

    def setUp(self):
        self.beamline = create_beamline("ID01")

    def test_detector_hor(self):
        self.assertTrue(
            self.beamline.detector_hor == "y+"
        )

    def test_detector_ver(self):
        self.assertTrue(
            self.beamline.detector_hor == "z-"
        )

    def test_exit_wavevector(self):
        params = {
            "inplane_angle": 0.0,
            "outofplane_angle": 90.0,
            "wavelength_m": 2 * np.pi,
        }
        print(self.beamline.exit_wavevector(params))
        self.assertTrue(
            np.allclose(
                self.beamline.exit_wavevector(params),
                np.array([0.0, 1.0, 0.0]),
                rtol=1e-09,
                atol=1e-09
            )
        )


if __name__ == "__main__":
    run_tests(TestBeamline)
