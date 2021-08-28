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
from bcdi.experiment.diffractometer import (
    DiffractometerCRISTAL,
    DiffractometerSIXS
)


labframe_to_xrayutil = {
    "x+": "y+",
    "x-": "y-",
    "y+": "z+",
    "y-": "z-",
    "z+": "x+",
    "z-": "x-",
}


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
        self.conversion_table = labframe_to_xrayutil
        self.root_dir = "D:/data/Cristal/"
        self.sample_name = "S"
        self.scan_number = 1
        self.template_imagefile = self.sample_name + "%d.nxs"
        self.beamline = create_beamline("CRISTAL")
        self.diffractometer = DiffractometerCRISTAL(sample_offsets=(0, 0))
        self.beam_direction = np.array([1, 0, 0])
        self.offset_inplane = 1

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

    def test_init_qconversion(self):
        qconv, offsets = self.beamline.init_qconversion(
            conversion_table=self.conversion_table,
            beam_direction=self.beam_direction,
            offset_inplane=self.offset_inplane,
            diffractometer=self.diffractometer,
        )
        nb_circles = len(self.diffractometer.sample_circles) +\
            len(self.diffractometer.detector_circles)
        print(offsets)
        self.assertEqual(len(offsets), nb_circles)
        self.assertEqual(offsets, [0, 0, self.offset_inplane, 0])

    def test_inplane_coeff(self):
        self.assertEqual(self.beamline.inplane_coeff(self.diffractometer), 1)

    def test_outofplane_coeff(self):
        self.assertEqual(self.beamline.outofplane_coeff(self.diffractometer), 1)


class TestBeamlineSIXS2019(unittest.TestCase):
    """Tests related to CRISTAL beamline instantiation."""

    def setUp(self):
        self.conversion_table = labframe_to_xrayutil
        self.root_dir = "D:/data/Sixs/"
        self.sample_name = "S"
        self.scan_number = 1
        self.specfile_name = self.root_dir + "alias_dict.txt"
        self.template_imagefile = 'spare_ascan_mu_%05d.nxs'
        self.beamline = create_beamline("SIXS_2019")
        self.diffractometer = DiffractometerSIXS(sample_offsets=(0, 0))
        self.beam_direction = np.array([1, 0, 0])
        self.offset_inplane = 1

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
            "specfile_name": self.specfile_name,
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
        self.assertEqual(specfile, self.specfile_name)
        self.assertEqual(template_imagefile, self.template_imagefile)

    def test_init_qconversion(self):
        qconv, offsets = self.beamline.init_qconversion(
            conversion_table=self.conversion_table,
            beam_direction=self.beam_direction,
            offset_inplane=self.offset_inplane,
            diffractometer=self.diffractometer,
        )
        nb_circles = len(self.diffractometer.sample_circles) +\
            len(self.diffractometer.detector_circles)
        print(offsets)
        self.assertEqual(len(offsets), nb_circles)
        self.assertEqual(offsets, [0, 0, 0, self.offset_inplane, 0])

    def test_inplane_coeff(self):
        self.assertEqual(self.beamline.inplane_coeff(self.diffractometer), 1)

    def test_outofplane_coeff(self):
        self.assertEqual(self.beamline.outofplane_coeff(self.diffractometer), 1)


if __name__ == "__main__":
    run_tests(TestBeamline)
    run_tests(TestBeamlineCRISTAL)
    run_tests(TestBeamlineSIXS2019)
