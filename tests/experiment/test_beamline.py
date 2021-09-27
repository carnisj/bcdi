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
    DiffractometerNANOMAX,
    DiffractometerID01,
    DiffractometerP10,
    Diffractometer34ID,
    DiffractometerSIXS,
)

# conversion table from the laboratory frame (CXI convention)
# (z downstream, y vertical up, x outboard) to the frame of xrayutilities
# (x downstream, y outboard, z vertical up)
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

    def test_find_inplane_CRISTAL(self):
        beamline = create_beamline("CRISTAL")
        diffractometer = DiffractometerCRISTAL(sample_offsets=(0, 0))
        self.assertTrue(beamline.find_inplane(diffractometer) == 0)

    def test_find_outofplane_CRISTAL(self):
        beamline = create_beamline("CRISTAL")
        diffractometer = DiffractometerCRISTAL(sample_offsets=(0, 0))
        self.assertTrue(beamline.find_outofplane(diffractometer) == 1)

    def test_find_inplane_ID01(self):
        beamline = create_beamline("ID01")
        diffractometer = DiffractometerID01(sample_offsets=(0, 0, 0))
        self.assertTrue(beamline.find_inplane(diffractometer) == 0)

    def test_find_outofplane_ID01(self):
        beamline = create_beamline("ID01")
        diffractometer = DiffractometerID01(sample_offsets=(0, 0, 0))
        self.assertTrue(beamline.find_outofplane(diffractometer) == 1)

    def test_find_inplane_NANOMAX(self):
        beamline = create_beamline("NANOMAX")
        diffractometer = DiffractometerNANOMAX(sample_offsets=(0, 0))
        self.assertTrue(beamline.find_inplane(diffractometer) == 0)

    def test_find_outofplane_NANOMAX(self):
        beamline = create_beamline("NANOMAX")
        diffractometer = DiffractometerNANOMAX(sample_offsets=(0, 0))
        self.assertTrue(beamline.find_outofplane(diffractometer) == 1)

    def test_find_inplane_P10(self):
        beamline = create_beamline("P10")
        diffractometer = DiffractometerP10(sample_offsets=(0, 0, 0, 0))
        self.assertTrue(beamline.find_inplane(diffractometer) == 0)

    def test_find_outofplane_P10(self):
        beamline = create_beamline("P10")
        diffractometer = DiffractometerP10(sample_offsets=(0, 0, 0, 0))
        self.assertTrue(beamline.find_outofplane(diffractometer) == 1)

    def test_find_inplane_SIXS(self):
        beamline = create_beamline("SIXS_2019")
        diffractometer = DiffractometerSIXS(sample_offsets=(0, 0))
        self.assertTrue(beamline.find_inplane(diffractometer) == 1)

    def test_find_outofplane_SIXS(self):
        beamline = create_beamline("SIXS_2019")
        diffractometer = DiffractometerSIXS(sample_offsets=(0, 0))
        self.assertTrue(beamline.find_outofplane(diffractometer) == 2)

    def test_find_inplane_34ID(self):
        beamline = create_beamline("34ID")
        diffractometer = Diffractometer34ID(sample_offsets=(0, 0))
        self.assertTrue(beamline.find_inplane(diffractometer) == 0)

    def test_find_outofplane_34ID(self):
        beamline = create_beamline("34ID")
        diffractometer = Diffractometer34ID(sample_offsets=(0, 0))
        self.assertTrue(beamline.find_outofplane(diffractometer) == 1)


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
            "diffractometer": self.diffractometer,
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
        _, offsets = self.beamline.init_qconversion(
            conversion_table=self.conversion_table,
            beam_direction=self.beam_direction,
            offset_inplane=self.offset_inplane,
            diffractometer=self.diffractometer,
        )
        nb_circles = len(self.diffractometer.sample_circles) + len(
            self.diffractometer.detector_circles
        )
        print(offsets)
        self.assertEqual(len(offsets), nb_circles)
        self.assertEqual(offsets, [0, 0, self.offset_inplane, 0])

    def test_inplane_coeff(self):
        self.assertEqual(self.beamline.inplane_coeff(self.diffractometer), 1)

    def test_outofplane_coeff(self):
        self.assertEqual(self.beamline.outofplane_coeff(self.diffractometer), 1)


class TestBeamlineID01(unittest.TestCase):
    """Tests related to ID01 beamline instantiation."""

    def setUp(self):
        self.conversion_table = labframe_to_xrayutil
        self.root_dir = "D:/data/ID01/"
        self.sample_name = "S"
        self.scan_number = 1
        self.template_imagefile = "data_mpx4_%05d.edf.gz"
        self.specfile_name = "test"
        self.beamline = create_beamline("ID01")
        self.diffractometer = DiffractometerID01(sample_offsets=(0, 0, 0))
        self.beam_direction = np.array([1, 0, 0])
        self.offset_inplane = 1

    def test_detector_hor(self):
        self.assertTrue(self.beamline.detector_hor == "x+")

    def test_detector_ver(self):
        self.assertTrue(self.beamline.detector_ver == "y-")

    def test_exit_wavevector(self):
        params = {
            "diffractometer": self.diffractometer,
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
        self.assertEqual(specfile, "test")
        self.assertEqual(template_imagefile, "data_mpx4_%05d.edf.gz")

    def test_init_qconversion(self):
        _, offsets = self.beamline.init_qconversion(
            conversion_table=self.conversion_table,
            beam_direction=self.beam_direction,
            offset_inplane=self.offset_inplane,
            diffractometer=self.diffractometer,
        )
        nb_circles = len(self.diffractometer.sample_circles) + len(
            self.diffractometer.detector_circles
        )
        print(offsets)
        self.assertEqual(len(offsets), nb_circles)
        self.assertEqual(offsets, [0, 0, 0, self.offset_inplane, 0])

    def test_inplane_coeff(self):
        self.assertEqual(self.beamline.inplane_coeff(self.diffractometer), -1)

    def test_outofplane_coeff(self):
        self.assertEqual(self.beamline.outofplane_coeff(self.diffractometer), 1)


class TestBeamlineNANOMAX(unittest.TestCase):
    """Tests related to NANOMAX beamline instantiation."""

    def setUp(self):
        self.conversion_table = labframe_to_xrayutil
        self.root_dir = "D:/data/Nanomax/"
        self.sample_name = "S"
        self.scan_number = 1
        self.template_imagefile = "%06d.h5"
        self.specfile_name = ""
        self.beamline = create_beamline("NANOMAX")
        self.diffractometer = DiffractometerNANOMAX(sample_offsets=(0, 0))
        self.beam_direction = np.array([1, 0, 0])
        self.offset_inplane = 1

    def test_detector_hor(self):
        self.assertTrue(self.beamline.detector_hor == "x+")

    def test_detector_ver(self):
        self.assertTrue(self.beamline.detector_ver == "y-")

    def test_exit_wavevector(self):
        params = {
            "diffractometer": self.diffractometer,
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
            homedir,
            self.root_dir + self.sample_name + "{:06d}".format(self.scan_number) + "/"
        )
        self.assertEqual(default_dirname, "data/")
        self.assertEqual(specfile, "")
        self.assertEqual(template_imagefile, "%06d.h5")

    def test_init_qconversion(self):
        _, offsets = self.beamline.init_qconversion(
            conversion_table=self.conversion_table,
            beam_direction=self.beam_direction,
            offset_inplane=self.offset_inplane,
            diffractometer=self.diffractometer,
        )
        nb_circles = len(self.diffractometer.sample_circles) + len(
            self.diffractometer.detector_circles
        )
        print(offsets)
        self.assertEqual(len(offsets), nb_circles)
        self.assertEqual(offsets, [0, 0, self.offset_inplane, 0])

    def test_inplane_coeff(self):
        self.assertEqual(self.beamline.inplane_coeff(self.diffractometer), -1)

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
        self.template_imagefile = "spare_ascan_mu_%05d.nxs"
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
            "diffractometer": self.diffractometer,
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
        _, offsets = self.beamline.init_qconversion(
            conversion_table=self.conversion_table,
            beam_direction=self.beam_direction,
            offset_inplane=self.offset_inplane,
            diffractometer=self.diffractometer,
        )
        nb_circles = len(self.diffractometer.sample_circles) + len(
            self.diffractometer.detector_circles
        )
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
