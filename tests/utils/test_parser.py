# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#   (c) 06/2021-present : DESY CFEL
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

from pathlib import Path
import unittest
from bcdi.utils.parser import ConfigParser

here = Path(__file__).parent
CONFIG = str(here.parents[1] / "conf/config_postprocessing.yml")


def run_tests(test_class):
    suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


class TestConfigParser(unittest.TestCase):
    """
    Tests on the class ConfigParser.

    def __init__(self, file_path : str, script_type : str = "preprocessing") -> None :
    """

    def setUp(self) -> None:
        self.command_line_args = {"scan": "9999"}
        self.parser = ConfigParser(CONFIG, self.command_line_args)

    def test_init_file_path(self):
        self.assertTrue(self.parser.file_path == CONFIG)

    def test_init_file_path_2(self):
        self.assertTrue(self.parser.arguments is None)

    def test_init_file_path_wrong_type(self):
        with self.assertRaises(TypeError):
            ConfigParser(1234, self.command_line_args)

    def test_init_file_path_wrong_file_extension(self):
        with self.assertRaises(ValueError):
            ConfigParser("C:/test.txt", self.command_line_args)

    def test_init_file_path_not_existing(self):
        with self.assertRaises(ValueError):
            ConfigParser("C:/test.yml", self.command_line_args)

    def test_init_raw_config(self):
        self.assertIsInstance(self.parser.raw_config, bytes)


if __name__ == "__main__":
    run_tests(TestConfigParser)
