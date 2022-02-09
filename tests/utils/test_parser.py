# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

from pathlib import Path
import unittest
from bcdi.utils.parser import ConfigParser

here = Path(__file__).parent
CONFIG = str(here.parents[1] / "bcdi/examples/config_postprocessing.yml")


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
        self.command_line_args = {"scan": 999999999, "root_folder": str(here)}
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

    def test_init_command_line_args(self):
        self.assertTrue(self.parser.command_line_args == self.command_line_args)

    def test_init_command_line_args_none(self):
        parser = ConfigParser(CONFIG, None)
        self.assertTrue(parser.command_line_args is None)

    def test_init_raw_config(self):
        self.assertIsInstance(self.parser.raw_config, bytes)

    def test_filter_dict(self):
        dic = {"scan": "9999", "sdd": None}
        output = self.parser.filter_dict(dic)
        self.assertTrue(output == {"scan": "9999"})

    def test_filter_dict_filter_value(self):
        dic = {"scan": "9999", "sdd": None, "test": True}
        output = self.parser.filter_dict(dic, filter_value=True)
        self.assertTrue(output == {"scan": "9999", "sdd": None})

    def test_load_arguments(self):
        args = self.parser.load_arguments()
        # "scan" is also key in CONFIG, which means that the overriding by the optional
        # --scan argument from the command line works as expected
        self.assertTrue(args.get("scan") == self.command_line_args["scan"])

    def test_load_arguments_no_cl_params_flip(self):
        args = self.parser.load_arguments()
        self.assertTrue(args.get("flip_reconstruction") is True)

    def test_load_arguments_cl_params_flip(self):
        self.parser = ConfigParser(
            CONFIG, {"flip_reconstruction": "False", "root_folder": str(here)}
        )
        # "flip_reconstruction" is also key in CONFIG, which means that the overriding
        # by the optional --flip_reconstruction argument from the command line works as
        # expected
        args = self.parser.load_arguments()
        self.assertTrue(args.get("flip_reconstruction") is False)

    def test_load_arguments_cl_params_flip_no_bool(self):
        self.parser = ConfigParser(
            CONFIG, {"flip_reconstruction": "weirdstring", "root_folder": str(here)}
        )
        with self.assertRaises(TypeError):
            self.parser.load_arguments()


if __name__ == "__main__":
    run_tests(TestConfigParser)
