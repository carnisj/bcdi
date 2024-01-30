# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr
import os
import unittest

import numpy as np
from pyfakefs import fake_filesystem_unittest

from bcdi.experiment.beamline import create_beamline
from bcdi.experiment.loader import LoaderID01, create_loader
from tests.config import load_config, run_tests

parameters, skip_tests = load_config("preprocessing")


class TestInitPath(fake_filesystem_unittest.TestCase):
    """Tests related to Loader.init_paths."""

    def setUp(self):
        self.beamline = None
        self.specfile_name = None
        self.template_imagefile = None
        self.root_dir = "D:/data/test/"
        self.sample_name = "S"
        self.scan_number = 1
        self.beam_direction = np.array([1, 0, 0])
        self.offset_inplane = 1
        self.params = {
            "wavelength": 1,
            "distance": 1,
            "pixel_x": 55000,
            "pixel_y": 55000,
            "inplane": 32,
            "outofplane": 28,
            "tilt": 0.005,
            "verbose": False,
        }

    def test_init_paths_CRISTAL(self):
        self.template_imagefile = self.sample_name + "%d.nxs"
        self.specfile_name = "anything"
        self.beamline = create_beamline("CRISTAL")
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
        ) = self.beamline.loader.init_paths(**params)
        self.assertEqual(
            homedir, self.root_dir + self.sample_name + str(self.scan_number) + "/"
        )
        self.assertEqual(default_dirname, "data/")
        self.assertEqual(specfile, None)
        self.assertEqual(template_imagefile, self.sample_name + "%d.nxs")

    def test_init_paths_ID01(self):
        self.template_imagefile = "data_mpx4_%05d.edf.gz"
        self.specfile_name = "test"
        self.beamline = create_beamline("ID01")
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
        ) = self.beamline.loader.init_paths(**params)
        self.assertEqual(
            homedir, self.root_dir + self.sample_name + str(self.scan_number) + "/"
        )
        self.assertEqual(default_dirname, "data/")
        self.assertEqual(specfile, "test")
        self.assertEqual(template_imagefile, "data_mpx4_%05d.edf.gz")

    def test_init_paths_BM02(self):
        self.template_imagefile = "sample_name%04d.edf"
        self.specfile_name = "test"
        self.beamline = create_beamline("BM02")
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
        ) = self.beamline.loader.init_paths(**params)
        self.assertEqual(
            homedir, self.root_dir + self.sample_name + str(self.scan_number) + "/"
        )
        self.assertEqual(default_dirname, "data/")
        self.assertEqual(specfile, "test")
        self.assertEqual(template_imagefile, "sample_name%04d.edf")

    def test_init_paths_ID01_BLISS(self):
        self.template_imagefile = "exp1_scan5.h5"
        self.specfile_name = None
        self.beamline = create_beamline("ID01BLISS")
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
        ) = self.beamline.loader.init_paths(**params)
        self.assertEqual(homedir, self.root_dir)
        self.assertEqual(default_dirname, "")
        self.assertEqual(specfile, None)
        self.assertEqual(template_imagefile, self.template_imagefile)

    def test_init_paths_NANOMAX(self):
        self.template_imagefile = "%06d.h5"
        self.specfile_name = "anything"
        self.beamline = create_beamline("NANOMAX")
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
        ) = self.beamline.loader.init_paths(**params)
        self.assertEqual(
            homedir,
            f"{self.root_dir}{self.sample_name}{self.scan_number:06d}/",
        )
        self.assertEqual(default_dirname, "data/")
        self.assertEqual(specfile, None)
        self.assertEqual(template_imagefile, "%06d.h5")

    def test_init_paths_P10(self):
        self.template_imagefile = "_master.h5"
        self.specfile_name = "anything"
        self.beamline = create_beamline("P10")
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
        ) = self.beamline.loader.init_paths(**params)
        self.assertEqual(
            homedir,
            f"{self.root_dir}{self.sample_name}_{self.scan_number:05d}/",
        )
        self.assertEqual(default_dirname, "e4m/")
        self.assertEqual(specfile, f"{self.sample_name}_{self.scan_number:05d}")
        self.assertEqual(template_imagefile, "S_00001_master.h5")

    def test_init_paths_specfile_P10_full_path(self):
        self.template_imagefile = "_master.h5"
        self.specfile_name = "anything"
        self.beamline = create_beamline("P10")
        self.setUpPyfakefs()
        valid_path = "/gpfs/bcdi/data"
        os.makedirs(valid_path)
        with open(valid_path + "/dummy.fio", "w") as f:
            f.write("dummy")

        params = {
            "root_folder": self.root_dir,
            "sample_name": self.sample_name,
            "scan_number": self.scan_number,
            "specfile_name": valid_path + "/dummy.fio",
            "template_imagefile": self.template_imagefile,
        }
        (
            homedir,
            default_dirname,
            specfile,
            template_imagefile,
        ) = self.beamline.loader.init_paths(**params)
        self.assertEqual(
            homedir,
            f"{self.root_dir}{self.sample_name}_{self.scan_number:05d}/",
        )
        self.assertEqual(default_dirname, "e4m/")
        self.assertEqual(specfile, params["specfile_name"])
        self.assertEqual(template_imagefile, "S_00001_master.h5")

    def test_init_paths_SIXS(self):
        self.template_imagefile = "spare_ascan_mu_%05d.nxs"
        self.specfile_name = self.root_dir + "alias_dict.txt"
        self.beamline = create_beamline("SIXS_2019")
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
        ) = self.beamline.loader.init_paths(**params)
        self.assertEqual(
            homedir, self.root_dir + self.sample_name + str(self.scan_number) + "/"
        )
        self.assertEqual(default_dirname, "data/")
        self.assertEqual(specfile, self.specfile_name)
        self.assertEqual(template_imagefile, self.template_imagefile)

    def test_init_paths_34ID(self):
        self.template_imagefile = None
        self.specfile_name = "test_spec"
        self.beamline = create_beamline("34ID")
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
        ) = self.beamline.loader.init_paths(**params)
        self.assertEqual(
            homedir, self.root_dir + self.sample_name + str(self.scan_number) + "/"
        )
        self.assertEqual(default_dirname, "data/")
        self.assertEqual(specfile, self.specfile_name)
        self.assertEqual(template_imagefile, None)


class TestRetrieveDistance(fake_filesystem_unittest.TestCase):
    """
    Tests related to DiffractometerID01.retrieve_distance.

    def retrieve_distance(setup) -> Optional[float]:
    """

    def setUp(self):
        self.setUpPyfakefs()
        self.valid_path = "/gpfs/bcdi/data/"
        os.makedirs(self.valid_path)
        self.beamline = create_beamline("ID01")

    def test_distance_defined(self):
        with open(self.valid_path + "defined.spec", "w") as f:
            f.write(
                "test\n#UDETCALIB cen_pix_x=11.195,cen_pix_y=281.115,"
                "pixperdeg=455.257,"
                "det_distance_CC=1.434,det_distance_COM=1.193,"
                "timestamp=2021-02-28T13:01:16.615422"
            )
        distance = self.beamline.loader.retrieve_distance(
            filename="defined.spec", default_folder=self.valid_path
        )
        self.assertTrue(np.isclose(distance, 1.193))

    def test_distance_undefined(self):
        with open(self.valid_path + "undefined.spec", "w") as f:
            f.write("test\n#this,is,bad")
        distance = self.beamline.loader.retrieve_distance(
            filename="undefined.spec", default_folder=self.valid_path
        )
        self.assertTrue(distance is None)

    def test_distance_not_a_spec_file(self):
        with self.assertRaises(ValueError):
            self.beamline.loader.retrieve_distance(
                filename="undefined.txt", default_folder=self.valid_path
            )


class TestRepr(unittest.TestCase):
    """Tests related to __repr__."""

    def setUp(self) -> None:
        self.loader = create_loader(name="ID01", sample_offsets=(0, 0, 0))

    def test_return_type(self):
        self.assertIsInstance(eval(repr(self.loader)), LoaderID01)


if __name__ == "__main__":
    run_tests(TestRetrieveDistance)
    run_tests(TestInitPath)
    run_tests(TestRepr)
