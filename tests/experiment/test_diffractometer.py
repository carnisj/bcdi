# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import unittest

from bcdi.experiment.diffractometer import DiffractometerFactory, FullDiffractometer
from tests.config import run_tests


class TestRepr(unittest.TestCase):
    """Tests related to __repr__."""

    def setUp(self) -> None:
        self.diffractometer = DiffractometerFactory.create_diffractometer(name="ID01")

    def test_return_type(self):
        self.assertIsInstance(eval(repr(self.diffractometer)), FullDiffractometer)


if __name__ == "__main__":
    run_tests(TestRepr)
