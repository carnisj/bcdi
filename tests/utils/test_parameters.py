# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr
import copy

import numpy as np
from pathlib import Path
import unittest
from unittest.mock import patch
from bcdi.utils.parameters import (
    ConfigChecker,
    MissingKeyError,
    PreprocessingChecker,
    PostprocessingChecker,
    valid_param,
)
from bcdi.graph.colormap import ColormapFactory
from bcdi.utils.parser import ConfigParser
from tests.config import has_backend, run_tests

here = Path(__file__).parent
THIS_DIR = str(here)
CONFIG_PRE = str(here.parents[1] / "bcdi/examples/S11_config_preprocessing.yml")
CONFIG_POST = str(here.parents[1] / "bcdi/examples/S11_config_postprocessing.yml")


class TestConfigChecker(unittest.TestCase):
    """Tests related to the abstract class ConfigChecker."""

    @patch("bcdi.utils.parameters.ConfigChecker.__abstractmethods__", set())
    def setUp(self) -> None:
        self.command_line_args = {
            "root_folder": str(here),
            "data_dir": str(here),
            "save_dir": str(here),
            "backend": "Agg",
            "flag_interact": False,
        }
        self.parser = ConfigParser(CONFIG_POST, self.command_line_args)
        self.args = self.parser.load_arguments()
        self.checker = ConfigChecker(initial_params=self.args)
        if not has_backend(self.checker.initial_params["backend"]):
            self.skipTest(
                reason=f"cannot load backend {self.checker.initial_params['backend']}"
            )

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

    def test_assign_default_value_copy(self):
        self.checker._assign_default_value()
        for key, val in self.checker.initial_params.items():
            if isinstance(val, (list, tuple, np.ndarray)):
                self.assertTrue(
                    all(
                        item1 == item2
                        for item1, item2 in zip(
                            self.checker.initial_params[key],
                            self.checker._checked_params[key],
                        )
                    )
                )
            else:
                self.assertTrue(
                    self.checker._checked_params[key]
                    == self.checker.initial_params[key]
                )

    def test_assign_default_value_defined_in_config(self):
        original = self.checker.initial_params["reflection"]
        default = [5, 2, 1]
        self.checker.default_values = {"reflection": default}
        self.checker._assign_default_value()
        self.assertTrue(
            all(
                item1 == item2
                for item1, item2 in zip(
                    original,
                    self.checker._checked_params["reflection"],
                )
            )
        )

    def test_assign_default_value_undefined_in_config(self):
        del self.checker._checked_params["reflection"]
        default = [5, 2, 1]
        self.checker.default_values = {"reflection": default}
        self.checker._assign_default_value()
        self.assertTrue(
            all(
                item1 == item2
                for item1, item2 in zip(
                    default,
                    self.checker._checked_params["reflection"],
                )
            )
        )

    def test_check_backend_not_supported(self):
        self.checker.initial_params["backend"] = "bad_backend"
        with self.assertRaises(ValueError):
            self.checker._check_backend()

    def test_check_length_wrong_length(self):
        self.checker.initial_params["specfile_name"] = ["test.spec", "test2.spec"]
        with self.assertRaises(ValueError):
            self.checker._check_length("specfile_name", length=3)

    def test_check_length_unique_value(self):
        self.checker._check_length("specfile_name", length=3)
        self.assertTrue(len(self.checker._checked_params["specfile_name"]) == 3)

    def test_check_length_wrong_type(self):
        self.checker.initial_params["specfile_name"] = "test.spec"
        with self.assertRaises(TypeError):
            self.checker._check_length("specfile_name", length=3)

    def test_check_length_none(self):
        self.checker.initial_params["specfile_name"] = None
        self.checker._check_length("specfile_name", length=3)
        self.assertTrue(len(self.checker._checked_params["specfile_name"]) == 3)
        self.assertTrue(
            all(val is None for val in self.checker._checked_params["specfile_name"])
        )

    def test__check_mandatory_params_valid_key(self):
        self.checker.required_params = ("specfile_name",)
        self.assertTrue(self.checker._check_mandatory_params() is None)

    def test__check_mandatory_params_key_absent(self):
        self.checker.required_params = ("required_key",)
        with self.assertRaises(MissingKeyError):
            self.checker._check_mandatory_params()

    def test_create_colormap(self):
        self.checker._create_colormap()
        self.assertIsInstance(self.checker._checked_params["colormap"], ColormapFactory)
        self.assertTrue(self.checker._checked_params["colormap"].colormap == "turbo")

    def test_create_colormap_grey_background(self):
        self.checker.initial_params["grey_background"] = True
        self.checker._create_colormap()
        self.assertTrue(self.checker._checked_params["colormap"].bad_color == "0.7")

    def test_check_config_scans_none(self):
        self.checker.initial_params["scans"] = None
        with self.assertRaises(ValueError):
            self.checker.check_config()

    def test_check_config_scans_instantiation_error(self):
        with self.assertRaises(NotImplementedError):
            self.checker.check_config()


class TestPreprocessingChecker(unittest.TestCase):
    """Tests related to the abstract class PostprocessingChecker."""

    def setUp(self) -> None:
        self.command_line_args = {
            "root_folder": str(here),
            "data_dir": str(here),
            "save_dir": str(here),
            "backend": "Agg",
            "flag_interact": False,
        }
        self.parser = ConfigParser(CONFIG_PRE, self.command_line_args)
        self.args = self.parser.load_arguments()
        self.checker = PreprocessingChecker(initial_params=self.args)
        if not has_backend(self.checker.initial_params["backend"]):
            self.skipTest(
                reason=f"cannot load backend {self.checker.initial_params['backend']}"
            )

    def test_check_config(self):
        out = self.checker.check_config()
        self.assertEqual(out["scans"], (11,))

    def test_check_config_several_scans(self):
        self.checker.initial_params["scans"] = (
            11,
            12,
        )
        self.checker.initial_params["center_fft"] = "crop_sym_ZYX"
        out = self.checker.check_config()
        self.assertEqual(out["center_fft"], "skip")

    def test_check_config_fix_size(self):
        self.checker.initial_params["fix_size"] = [2, 127, 25, 326, 56, 95]
        self.checker.initial_params["center_fft"] = "crop_sym_ZYX"
        self.checker.initial_params["roi_detector"] = [2, 326, 5, 956]
        out = self.checker.check_config()
        self.assertEqual(out["center_fft"], "skip")
        self.assertTrue(len(out["roi_detector"]) == 0)

    def test_check_config_photon_filter_loading(self):
        self.checker.initial_params["photon_filter"] = "loading"
        out = self.checker.check_config()
        self.assertEqual(
            out["loading_threshold"], self.checker.initial_params["photon_threshold"]
        )

    def test_check_config_photon_filter_postprocessing(self):
        self.checker.initial_params["photon_filter"] = "postprocessing"
        out = self.checker.check_config()
        self.assertEqual(out["loading_threshold"], 0)

    def test_check_config_reload_previous(self):
        self.checker.initial_params["comment"] = "IronMaiden"
        self.checker.initial_params["reload_previous"] = True
        self.checker.initial_params["align_q"] = False
        self.checker._checked_params = copy.deepcopy(self.checker.initial_params)
        out = self.checker.check_config()
        self.assertEqual(out["comment"], "IronMaiden_reloaded")

    def test_check_config_not_reload_previous(self):
        out = self.checker.check_config()
        self.assertEqual(out["preprocessing_binning"], (1, 1, 1))
        self.assertFalse(out["reload_orthogonal"])

    def test_check_config_rocking_angle_energy(self):
        self.checker.initial_params["rocking_angle"] = "energy"
        self.checker.initial_params["use_rawdata"] = True
        out = self.checker.check_config()
        self.assertFalse(out["use_rawdata"])

    def test_check_config_reload_orthogonal(self):
        self.checker.initial_params["reload_previous"] = True
        self.checker.initial_params["reload_orthogonal"] = True
        self.checker.initial_params["use_rawdata"] = True
        self.checker._checked_params = copy.deepcopy(self.checker.initial_params)
        out = self.checker.check_config()
        self.assertFalse(out["use_rawdata"])

    def test_check_config_use_rawdata(self):
        self.checker.initial_params["use_rawdata"] = True
        out = self.checker.check_config()
        self.assertEqual(out["save_dirname"], "pynxraw")

    def test_check_config_interpolate_energy(self):
        self.checker.initial_params["use_rawdata"] = False
        self.checker.initial_params["rocking_angle"] = "energy"
        out = self.checker.check_config()
        self.assertEqual(out["save_dirname"], "pynx")
        self.assertEqual(out["interpolation_method"], "xrayutilities")

    def test_check_config_interpolate_undefined_method(self):
        self.checker.initial_params["use_rawdata"] = False
        self.checker.initial_params["interpolation_method"] = "bad_method"
        self.checker._checked_params = copy.deepcopy(self.checker.initial_params)
        with self.assertRaises(ValueError):
            self.checker.check_config()

    def test_check_config_interpolate_reload_orthogonal(self):
        self.checker.initial_params["use_rawdata"] = False
        self.checker.initial_params["reload_previous"] = True
        self.checker.initial_params["reload_orthogonal"] = True
        self.checker.initial_params["preprocessing_binning"] = (2, 2, 2)
        self.checker._checked_params = copy.deepcopy(self.checker.initial_params)
        with self.assertRaises(ValueError):
            self.checker.check_config()

    def test_check_config_align_q(self):
        self.checker.initial_params["align_q"] = True
        out = self.checker.check_config()
        self.assertEqual(
            out["comment"],
            f"_align-q-{self.checker.initial_params['ref_axis_q']}",
        )

    def test_check_config_align_q_undefined_reference_axis(self):
        self.checker.initial_params["align_q"] = True
        self.checker.initial_params["ref_axis_q"] = "a"
        with self.assertRaises(ValueError):
            self.checker.check_config()

    def test_check_config_incompatible_backend(self):
        self.checker.initial_params["flag_interact"] = True
        with self.assertRaises(ValueError):
            self.checker.check_config()


class TestPostprocessingChecker(unittest.TestCase):
    """Tests related to the abstract class PostprocessingChecker."""

    def setUp(self) -> None:
        self.command_line_args = {
            "root_folder": str(here),
            "data_dir": str(here),
            "save_dir": str(here),
            "backend": "Agg",
            "flag_interact": False,
        }
        self.parser = ConfigParser(CONFIG_POST, self.command_line_args)
        self.args = self.parser.load_arguments()
        self.checker = PostprocessingChecker(initial_params=self.args)
        if not has_backend(self.checker.initial_params["backend"]):
            self.skipTest(
                reason=f"cannot load backend {self.checker.initial_params['backend']}"
            )

    def test_check_config(self):
        out = self.checker.check_config()
        self.assertEqual(out["scans"], (11,))

    def test_check_config_not_simulation(self):
        out = self.checker.check_config()
        self.assertTrue(out["invert_phase"])
        self.assertEqual(out["phase_fieldname"], "disp")

    def test_check_config_simulation(self):
        self.checker.initial_params["simulation"] = True
        out = self.checker.check_config()
        self.assertFalse(out["invert_phase"])
        self.assertFalse(out["correct_refraction"])
        self.assertEqual(out["phase_fieldname"], "phase")

    def test_check_config_detector_frame(self):
        out = self.checker.check_config()
        self.assertFalse(out["is_orthogonal"])

    def test_check_config_crystal_frame(self):
        self.checker.initial_params["data_frame"] = "crystal"
        out = self.checker.check_config()
        self.assertTrue(out["is_orthogonal"])
        self.assertEqual(out["save_frame"], "crystal")

    def test_check_config_crystal_frame_save_frame_bad_config(self):
        self.checker.initial_params["data_frame"] = "crystal"
        self.checker.initial_params["save_frame"] = "laboratory"
        out = self.checker.check_config()
        self.assertEqual(out["save_frame"], "crystal")


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
        self.assertTrue(val == (THIS_DIR,) and flag is True)


if __name__ == "__main__":
    run_tests(TestParameters)
    run_tests(TestConfigChecker)
    run_tests(TestPostprocessingChecker)
    run_tests(TestPreprocessingChecker)
