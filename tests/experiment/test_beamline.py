# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#   (c) 06/2021-present : DESY CFEL
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np
import os
from pyfakefs import fake_filesystem_unittest
import unittest
from bcdi.experiment.beamline import create_beamline, Beamline
from bcdi.experiment.diffractometer import DiffractometerCRISTAL

def run_tests(test_class):
    suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


class TestBeamline(unittest.TestCase):
    """Tests related to beamline instantiation."""

    def test_create_beamline_from_abc(self):
        with self.assertRaises(TypeError):
            Beamline(name="ID01")


class TestBeamlineCRISTAL(fake_filesystem_unittest.TestCase):
    """Tests related to CRISTAL beamline instantiation."""

    def setUp(self):
        self.setUpPyfakefs()
        self.root_dir = "D:/data/Cristal/"
        self.sample_name = "S"
        self.scan_number = 1
        self.template_imagefile = self.sample_name + "%d.nxs"
        datadir = self.root_dir + self.sample_name + str(self.scan_number)
        os.makedirs(datadir)
        with open(datadir + "test.nxs", "w") as f:
            f.write("dummy")
        self.beamline = create_beamline("CRISTAL")
        self.diffractometer = DiffractometerCRISTAL(sample_offsets=(0, 0))

    def test_detector_hor(self):
        self.assertTrue(self.beamline.detector_hor == "x+")

    def test_detector_ver(self):
        self.assertTrue(self.beamline.detector_ver == "y-")

    def test_exit_wavevector(self):
        params = {
            "inplane_angle": 0.0,
            "outofplane_angle": 90.0,
            "wavelength": 2 * np.pi,
        }
        self.assertTrue(
            np.allclose(
                self.beamline.exit_wavevector(**params),
                np.array([0.0, 1.0, 0.0]),
                rtol=1e-09,
                atol=1e-09,
            )
        )

    def test_init_paths(self):
        params = {
            "root_folder": self.root_dir,
            "sample_name": self.sample_name,
            "scan_number": self.scan_number,
            "specfile_name": "",
            "template_imagefile": self.template_imagefile,
        }
        (
            homedir,
            default_dirname,
            specfile,
            template_imagefile,
        ) = self.beamline.init_paths(**params)
        self.assertEqual(
            homedir, self.root_dir + self.sample_name + str(self.scan_number) + "/"
        )
        self.assertEqual(default_dirname, "data/")
        self.assertEqual(specfile, "")
        self.assertEqual(template_imagefile, self.sample_name + "%d.nxs")

    def test_inplane_coeff(self):
        self.assertEqual(self.beamline.inplane_coeff(self.diffractometer), 1)

    def test_outofplane_coeff(self):
        self.assertEqual(self.beamline.outofplane_coeff(self.diffractometer), 1)


if __name__ == "__main__":
    run_tests(TestBeamline)
