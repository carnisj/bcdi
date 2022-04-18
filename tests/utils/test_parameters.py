# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

from pathlib import Path
import unittest
from unittest.mock import patch
from bcdi.utils.parameters import (
    ConfigChecker,
    PreprocessingChecker,
    PostprocessingChecker,
    valid_param,
)
from bcdi.utils.parser import ConfigParser
from tests.config import run_tests

here = Path(__file__).parent
THIS_DIR = str(here)
CONFIG = str(here.parents[1] / "bcdi/examples/S11_config_postprocessing.yml")


class TestConfigChecker(unittest.TestCase):
    """Tests related to the abstract class ConfigChecker."""

    @patch("bcdi.utils.parameters.ConfigChecker.__abstractmethods__", set())
    def setUp(self) -> None:
        self.parser = ConfigParser(CONFIG, {})
        self.args = self.parser.load_arguments()
        self.checker = ConfigChecker(initial_params=self.args)

    def test_create_roi_none(self):
        self.checker.initial_params["roi_detector"] = None
        self.assertEqual(self.checker._create_roi(), None)

    def test_create_roi_center_roi_x_center_roi_y_none(self):
        self.checker.initial_params["roi_detector"] = [10, 200, 20, 50]
        correct = [10, 200, 20, 50]
        output = self.checker._create_roi()
        self.assertTrue(all(out == correct[idx] for idx, out in enumerate(output)))

    def test_create_roi_center_roi_x_not_none(self):
        self.checker.initial_params["roi_detector"] = [10, 200, 20, 50]
        self.checker.initial_params["center_roi_x"] = 150
        correct = [10, 200, 130, 200]
        output = self.checker._create_roi()
        self.assertTrue(all(out == correct[idx] for idx, out in enumerate(output)))

    def test_create_roi_center_roi_y_not_none(self):
        self.checker.initial_params["roi_detector"] = [10, 200, 20, 50]
        self.checker.initial_params["center_roi_y"] = 150
        correct = [140, 350, 20, 50]
        output = self.checker._create_roi()
        self.assertTrue(all(out == correct[idx] for idx, out in enumerate(output)))

    def test_create_roi_center_roi_x_y_not_none(self):
        self.checker.initial_params["roi_detector"] = [10, 200, 20, 50]
        self.checker.initial_params["center_roi_x"] = 10
        self.checker.initial_params["center_roi_y"] = 150
        correct = [140, 350, -10, 60]
        output = self.checker._create_roi()
        self.assertTrue(all(out == correct[idx] for idx, out in enumerate(output)))


class TestParameters(unittest.TestCase):
    """
    Tests on the function valid_param.

    def valid_param(key: str, value: Any) -> Tuple[Any, bool]:
    """

    def test_none_str_unexpected(self):
        val, flag = valid_param(key="not_expected", value="None")
        self.assertTrue(val is None and flag is False)

    def test_none_str_expected(self):
        val, flag = valid_param(key="actuators", value="None")
        self.assertTrue(val is None and flag is True)

    def test_true_expected_1(self):
        val, flag = valid_param(key="align_axis", value="True")
        self.assertTrue(val is True and flag is True)

    def test_true_expected_2(self):
        val, flag = valid_param(key="align_axis", value="true")
        self.assertTrue(val is True and flag is True)

    def test_true_expected_3(self):
        val, flag = valid_param(key="align_axis", value="TRUE")
        self.assertTrue(val is True and flag is True)

    def test_false_expected_1(self):
        val, flag = valid_param(key="align_axis", value="False")
        print(val, flag)
        self.assertTrue(val is False and flag is True)

    def test_false_expected_2(self):
        val, flag = valid_param(key="align_axis", value="false")
        self.assertTrue(val is False and flag is True)

    def test_false_expected_3(self):
        val, flag = valid_param(key="align_axis", value="FALSE")
        self.assertTrue(val is False and flag is True)

    def test_data_dir_none(self):
        val, flag = valid_param(key="data_dir", value="None")
        self.assertTrue(val is None and flag is True)

    def test_data_dir_not_existing(self):
        with self.assertRaises(ValueError):
            valid_param(key="data_dir", value="this_is_not_a_valid_path")

    def test_data_dir_existing(self):
        val, flag = valid_param(key="data_dir", value=THIS_DIR)
        self.assertTrue(val == THIS_DIR and flag is True)


if __name__ == "__main__":
    run_tests(TestParameters)
