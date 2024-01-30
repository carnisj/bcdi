# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import unittest

import bcdi.preprocessing.analysis as analysis
from tests.config import load_config, run_tests

parameters, skip_tests = load_config("preprocessing")


class TestDefineAnalysisType(unittest.TestCase):
    def test_reload_orthogonal_true(self) -> None:
        self.assertTrue(
            analysis.define_analysis_type(
                reload_orthogonal=True, use_rawdata=True, interpolation_method=""
            ),
            "interpolated",
        )
        self.assertTrue(
            analysis.define_analysis_type(
                reload_orthogonal=True, use_rawdata=False, interpolation_method=""
            ),
            "interpolated",
        )

    def test_reload_orthogonal_false_use_rawdata(self) -> None:
        self.assertTrue(
            analysis.define_analysis_type(
                reload_orthogonal=False, use_rawdata=True, interpolation_method=""
            ),
            "detector_frame",
        )

    def test_reload_orthogonal_false_interpolate_rawdata(self) -> None:
        method = "this_method"
        self.assertTrue(
            analysis.define_analysis_type(
                reload_orthogonal=False, use_rawdata=False, interpolation_method=method
            ),
            method,
        )


if __name__ == "__main__":
    run_tests(TestDefineAnalysisType)
