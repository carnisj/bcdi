# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

from pathlib import Path
import unittest
from bcdi.utils.parameters import valid_param
from tests.config import run_tests

here = Path(__file__).parent
THIS_DIR = str(here)


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
