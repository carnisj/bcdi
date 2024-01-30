# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import unittest

from bcdi.experiment.rotation_matrix import RotationMatrix
from tests.config import run_tests


class Test(unittest.TestCase):
    """Tests related to rotation matrix instantiation."""

    def setUp(self) -> None:
        self.rotmat = RotationMatrix(circle="x+", angle=34)

    def test_instantiation_missing_parameter(self):
        with self.assertRaises(TypeError):
            RotationMatrix()

    def test_repr(self):
        self.assertIsInstance(eval(repr(self.rotmat)), RotationMatrix)


if __name__ == "__main__":
    run_tests(Test)
