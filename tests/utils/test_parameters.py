# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr
import copy
import pathlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bcdi.graph.colormap import ColormapFactory
from bcdi.utils.parameters import (
    ConfigChecker,
    MissingKeyError,
    ParameterError,
    PostprocessingChecker,
    PreprocessingChecker,
    valid_param,
)
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
        self.checker = ConfigChecker(initial_params=self.args)  # type: ignore
        if not has_backend(self.checker._checked_params["backend"]):
            self.skipTest(
                reason=f"cannot load backend {self.checker._checked_params['backend']}"
            )

    def test_create_roi_none(self):
        self.checker._checked_params["roi_detector"] = None
        self.assertEqual(self.checker._create_roi(), None)

    def test_create_roi_center_roi_x_center_roi_y_none(self):
        self.checker._checked_params["roi_detector"] = [10, 200, 20, 50]
        correct = [10, 200, 20, 50]
        output = self.checker._create_roi()
        self.assertTrue(all(out == correct[idx] for idx, out in enumerate(output)))

    def test_create_roi_center_roi_x_not_none(self):
        self.checker._checked_params["roi_detector"] = [10, 200, 20, 50]
        self.checker._checked_params["center_roi_x"] = 150
        correct = [10, 200, 130, 200]
        output = self.checker._create_roi()
        self.assertTrue(all(out == correct[idx] for idx, out in enumerate(output)))

    def test_create_roi_center_roi_y_not_none(self):
        self.checker._checked_params["roi_detector"] = [10, 200, 20, 50]
        self.checker._checked_params["center_roi_y"] = 150
        correct = [140, 350, 20, 50]
        output = self.checker._create_roi()
        self.assertTrue(all(out == correct[idx] for idx, out in enumerate(output)))

    def test_create_roi_center_roi_x_y_not_none(self):
        self.checker._checked_params["roi_detector"] = [10, 200, 20, 50]
        self.checker._checked_params["center_roi_x"] = 10
        self.checker._checked_params["center_roi_y"] = 150
        correct = [140, 350, -10, 60]
        output = self.checker._create_roi()
        self.assertTrue(all(out == correct[idx] for idx, out in enumerate(output)))

    def test_assign_default_value_copy(self):
        self.checker._assign_default_value()
        for key, val in self.checker._checked_params.items():
            if isinstance(val, (list, tuple, np.ndarray)):
                self.assertTrue(
                    all(
                        item1 == item2
                        for item1, item2 in zip(
                            self.checker._checked_params[key],
                            self.checker._checked_params[key],
                        )
                    )
                )
            else:
                self.assertTrue(
                    self.checker._checked_params[key]
                    == self.checker._checked_params[key]
                )

    def test_assign_default_value_defined_in_config(self):
        original = self.checker._checked_params["reflection"]
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
        self.checker._checked_params["backend"] = "bad_backend"
        with self.assertRaises(ValueError):
            self.checker._check_backend()

    def test_check_length_wrong_length(self):
        self.checker._checked_params["specfile_name"] = ["test.spec", "test2.spec"]
        with self.assertRaises(ValueError):
            self.checker._check_length("specfile_name", length=3)

    def test_check_length_unique_value(self):
        self.checker._check_length("specfile_name", length=3)
        self.assertTrue(len(self.checker._checked_params["specfile_name"]) == 3)

    def test_check_length_wrong_type(self):
        self.checker._checked_params["specfile_name"] = "test.spec"
        with self.assertRaises(TypeError):
            self.checker._check_length("specfile_name", length=3)

    def test_check_length_none(self):
        self.checker._checked_params["specfile_name"] = None
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
        self.checker._checked_params["grey_background"] = True
        self.checker._create_colormap()
        self.assertTrue(self.checker._checked_params["colormap"].bad_color == "0.7")

    def test_create_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.checker._checked_params["save_dir"] = [tmpdir]
            self.checker._create_dirs()
            self.assertTrue(pathlib.Path(tmpdir).is_dir())

    def test_check_config_scans_none(self):
        self.checker._checked_params["scans"] = None
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
        if not has_backend(self.checker._checked_params["backend"]):
            self.skipTest(
                reason=f"cannot load backend {self.checker._checked_params['backend']}"
            )

    def test_bragg_peak_not_provided(self):
        self.checker._checked_params["bragg_peak"] = None
        out = self.checker.check_config()
        self.assertEqual(
            out["centering_method"], self.checker.initial_params["centering_method"]
        )

    def test_bragg_peak_provided(self):
        self.checker._checked_params["bragg_peak"] = [100, 252, 321]
        out = self.checker.check_config()
        self.assertEqual(out["centering_method"]["reciprocal_space"], "user")

    def test_check_config(self):
        out = self.checker.check_config()
        self.assertEqual(out["scans"], (11,))

    def test_check_config_several_scans(self):
        self.checker._checked_params["scans"] = (
            11,
            12,
        )
        self.checker._checked_params["center_fft"] = "crop_sym_ZYX"
        out = self.checker.check_config()
        self.assertEqual(out["center_fft"], "skip")

    def test_check_config_photon_filter_loading(self):
        self.checker._checked_params["photon_filter"] = "loading"
        out = self.checker.check_config()
        self.assertEqual(
            out["loading_threshold"], self.checker._checked_params["photon_threshold"]
        )

    def test_check_config_photon_filter_postprocessing(self):
        self.checker._checked_params["photon_filter"] = "postprocessing"
        out = self.checker.check_config()
        self.assertEqual(out["loading_threshold"], 0)

    def test_check_config_reload_previous(self):
        self.checker._checked_params["comment"] = "IronMaiden"
        self.checker._checked_params["reload_previous"] = True
        self.checker._checked_params["align_q"] = False
        self.checker._checked_params = copy.deepcopy(self.checker._checked_params)
        out = self.checker.check_config()
        self.assertEqual(out["comment"], "IronMaiden_reloaded")

    def test_check_config_not_reload_previous(self):
        out = self.checker.check_config()
        self.assertEqual(out["preprocessing_binning"], (1, 1, 1))
        self.assertFalse(out["reload_orthogonal"])

    def test_check_config_rocking_angle_energy(self):
        self.checker._checked_params["rocking_angle"] = "energy"
        self.checker._checked_params["use_rawdata"] = True
        out = self.checker.check_config()
        self.assertFalse(out["use_rawdata"])

    def test_check_config_reload_orthogonal(self):
        self.checker._checked_params["reload_previous"] = True
        self.checker._checked_params["reload_orthogonal"] = True
        self.checker._checked_params["use_rawdata"] = True
        self.checker._checked_params = copy.deepcopy(self.checker._checked_params)
        out = self.checker.check_config()
        self.assertFalse(out["use_rawdata"])

    def test_check_config_use_rawdata(self):
        self.checker._checked_params["use_rawdata"] = True
        out = self.checker.check_config()
        self.assertEqual(out["save_dirname"], "pynxraw")

    def test_check_config_interpolate_energy(self):
        self.checker._checked_params["use_rawdata"] = False
        self.checker._checked_params["rocking_angle"] = "energy"
        out = self.checker.check_config()
        self.assertEqual(out["save_dirname"], "pynx")
        self.assertEqual(out["interpolation_method"], "xrayutilities")

    def test_check_config_interpolate_undefined_method(self):
        self.checker._checked_params["use_rawdata"] = False
        self.checker._checked_params["interpolation_method"] = "bad_method"
        self.checker._checked_params = copy.deepcopy(self.checker._checked_params)
        with self.assertRaises(ValueError):
            self.checker.check_config()

    def test_check_config_interpolate_reload_orthogonal(self):
        self.checker._checked_params["use_rawdata"] = False
        self.checker._checked_params["reload_previous"] = True
        self.checker._checked_params["reload_orthogonal"] = True
        self.checker._checked_params["preprocessing_binning"] = (2, 2, 2)
        self.checker._checked_params = copy.deepcopy(self.checker._checked_params)
        with self.assertRaises(ValueError):
            self.checker.check_config()

    def test_check_config_align_q(self):
        self.checker._checked_params["align_q"] = True
        self.checker._checked_params["use_rawdata"] = False
        self.checker._checked_params["interpolation_method"] = "linearization"
        out = self.checker.check_config()
        self.assertEqual(
            out["comment"],
            f"_align-q-{self.checker._checked_params['ref_axis_q']}",
        )

    def test_check_config_align_q_undefined_reference_axis(self):
        self.checker._checked_params["align_q"] = True
        self.checker._checked_params["ref_axis_q"] = "a"
        with self.assertRaises(ValueError):
            self.checker.check_config()

    def test_check_config_incompatible_backend(self):
        self.checker._checked_params["flag_interact"] = True
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
        if not has_backend(self.checker._checked_params["backend"]):
            self.skipTest(
                reason=f"cannot load backend {self.checker._checked_params['backend']}"
            )

    def test_check_config(self):
        out = self.checker.check_config()
        self.assertEqual(out["scans"], (11,))

    def test_check_config_not_simulation(self):
        out = self.checker.check_config()
        self.assertTrue(out["invert_phase"])
        self.assertEqual(out["phase_fieldname"], "disp")

    def test_check_config_simulation(self):
        self.checker._checked_params["simulation"] = True
        out = self.checker.check_config()
        self.assertFalse(out["invert_phase"])
        self.assertFalse(out["correct_refraction"])
        self.assertEqual(out["phase_fieldname"], "phase")

    def test_check_config_detector_frame(self):
        out = self.checker.check_config()
        self.assertFalse(out["is_orthogonal"])

    def test_check_config_crystal_frame(self):
        self.checker._checked_params["data_frame"] = "crystal"
        out = self.checker.check_config()
        self.assertTrue(out["is_orthogonal"])
        self.assertEqual(out["save_frame"], "crystal")

    def test_check_config_crystal_frame_save_frame_bad_config(self):
        self.checker._checked_params["data_frame"] = "crystal"
        self.checker._checked_params["save_frame"] = "laboratory"
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

    def test_save_dir_none(self):
        val, flag = valid_param(key="save_dir", value=None)
        self.assertTrue(val == [None] and flag is True)

    def test_save_dir_str(self):
        val, flag = valid_param(key="save_dir", value=THIS_DIR)
        self.assertTrue(val == [THIS_DIR + "/"] and flag is True)

    def test_save_dir_str_2(self):
        folder = THIS_DIR + "/"
        val, flag = valid_param(key="save_dir", value=folder)
        self.assertTrue(val == [folder] and flag is True)

    def test_save_dir_list(self):
        folder = THIS_DIR + "/"
        val, flag = valid_param(key="save_dir", value=[folder, THIS_DIR])
        self.assertTrue(val == [folder, folder] and flag is True)

    def test_save_dir_tuple(self):
        folder = THIS_DIR + "/"
        val, flag = valid_param(key="save_dir", value=(folder, THIS_DIR))
        self.assertTrue(val == [folder, folder] and flag is True)

    def test_save_dir_list_none(self):
        folder = THIS_DIR + "/"
        val, flag = valid_param(key="save_dir", value=[folder, None])
        self.assertTrue(val == [folder, None] and flag is True)

    def test_tick_direction(self):
        with self.assertRaises(ParameterError):
            valid_param(key="tick_direction", value="skip")

    def test_strain_method(self):
        with self.assertRaises(ParameterError):
            valid_param(key="strain_method", value="skip")

    def test_save_frame(self):
        with self.assertRaises(ParameterError):
            valid_param(key="save_frame", value="skip")

    def test_center_fft(self):
        with self.assertRaises(ParameterError):
            valid_param(key="center_fft", value="fake")

    def test_rocking_angle(self):
        with self.assertRaises(ParameterError):
            valid_param(key="rocking_angle", value="skip")

    def test_ref_axis_q(self):
        with self.assertRaises(ParameterError):
            valid_param(key="ref_axis_q", value="skip")

    def test_ref_axis(self):
        with self.assertRaises(ParameterError):
            valid_param(key="ref_axis", value="skip")

    def test_photon_filter(self):
        with self.assertRaises(ParameterError):
            valid_param(key="photon_filter", value="skip")

    def test_phase_ramp_removal(self):
        with self.assertRaises(ParameterError):
            valid_param(key="phase_ramp_removal", value="skip")

    def test_optical_path_method(self):
        with self.assertRaises(ParameterError):
            valid_param(key="optical_path_method", value="skip")

    def test_normalize_flux(self):
        with self.assertRaises(ParameterError):
            valid_param(key="normalize_flux", value="crash")

    def test_median_filter(self):
        with self.assertRaises(ParameterError):
            valid_param(key="median_filter", value="custom")

    def test_interpolation_method(self):
        with self.assertRaises(ParameterError):
            valid_param(key="interpolation_method", value="skip")

    def test_fill_value_mask(self):
        with self.assertRaises(ParameterError):
            valid_param(key="fill_value_mask", value=2)

    def test_backend(self):
        with self.assertRaises(ParameterError):
            valid_param(key="backend", value="skip")

    def test_data_frame(self):
        with self.assertRaises(ParameterError):
            valid_param(key="data_frame", value="skip")

    def test_averaging_space(self):
        with self.assertRaises(ParameterError):
            valid_param(key="averaging_space", value="skip")

    def test_apodization_window(self):
        with self.assertRaises(ParameterError):
            valid_param(key="apodization_window", value="skip")

    def test_sort_method(self):
        with self.assertRaises(ParameterError):
            valid_param(key="sort_method", value="skip")

    def test_offset_method(self):
        with self.assertRaises(ParameterError):
            valid_param(key="offset_method", value="skip")

    def test_centering_method(self):
        with self.assertRaises(ParameterError):
            valid_param(key="centering_method", value="do_nothing")

    def test_root_folder(self):
        with self.assertRaises(ValueError):
            valid_param(key="root_folder", value="this_dir_does_not_exist")

    def test_reconstruction_files_none(self):
        val, flag = valid_param(key="reconstruction_files", value=None)
        self.assertTrue(val is None and flag is True)

    def test_reconstruction_files_str(self):
        val, flag = valid_param(key="reconstruction_files", value=CONFIG_PRE)
        self.assertTrue(val == (CONFIG_PRE,) and flag is True)

    def test_reconstruction_files_list_unexisting_file(self):
        with self.assertRaises(ValueError):
            valid_param(key="reconstruction_files", value=[CONFIG_PRE, "fake_file.h5"])

    def test_reconstruction_files_list_none(self):
        val, flag = valid_param(key="reconstruction_files", value=[CONFIG_PRE, None])
        self.assertTrue(val == [CONFIG_PRE, None] and flag is True)

    def test_mask(self):
        val, flag = valid_param(key="mask", value=None)
        self.assertTrue(val is None and flag is True)

    def test_mask_unexisting_file(self):
        with self.assertRaises(ValueError):
            valid_param(key="mask", value="fake_file.h5")

    def test_comment_empty(self):
        val, flag = valid_param(key="comment", value="")
        self.assertTrue(val == "" and flag is True)

    def test_comment(self):
        val, flag = valid_param(key="comment", value="test")
        self.assertTrue(val == "_test" and flag is True)

    def test_comment_underscore(self):
        val, flag = valid_param(key="comment", value="_test")
        self.assertTrue(val == "_test" and flag is True)

    def test_colormap_not_supported(self):
        with self.assertRaises(ValueError):
            valid_param(key="colormap", value="fake")

    def test_config_file_unexisting_file(self):
        with self.assertRaises(ValueError):
            valid_param(key="config_file", value="fake_file.h5")

    def test_energy_none(self):
        val, flag = valid_param(key="energy", value=None)
        self.assertTrue(val is None and flag is True)

    def test_energy_number(self):
        val, flag = valid_param(key="energy", value=10000)
        self.assertTrue(val == 10000 and flag is True)

    def test_energy_list(self):
        val, flag = valid_param(key="energy", value=[1, 2, 3])
        self.assertTrue(val == [1, 2, 3] and flag is True)

    def test_frames_pattern_binary_list(self):
        expected = [0, 1, 1, 0]
        output, flag = valid_param(key="frames_pattern", value=expected)
        self.assertTrue(
            all(val1 == val2 for val1, val2 in zip(expected, output)) and flag is True
        )

    def test_frames_pattern_none(self):
        val, flag = valid_param(key="frames_pattern", value=None)
        self.assertTrue(val is None and flag is True)

    def test_frames_pattern_list_of_indices(self):
        expected = [126, 138]
        output, flag = valid_param(key="frames_pattern", value=[126, 138])
        self.assertTrue(
            all(val1 == val2 for val1, val2 in zip(expected, output)) and flag is True
        )


if __name__ == "__main__":
    run_tests(TestParameters)
    run_tests(TestConfigChecker)
    run_tests(TestPostprocessingChecker)
    run_tests(TestPreprocessingChecker)
