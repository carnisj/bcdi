# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np
import unittest

from bcdi.preprocessing.bcdi_utils import find_bragg
from bcdi.utils.utilities import gaussian_window


def run_tests(test_class):
    suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


class TestFindBragg(unittest.TestCase):
    """
    Tests related to the function find_bragg.

    def find_bragg(
            data: np.ndarray,
            peak_method: str,
            roi: Optional[Tuple[int, int, int, int]] = None,
            binning: Optional[Tuple[int, ...]] = None,
    ) -> Tuple[int, ...]:
    """
    def setUp(self) -> None:
        data = np.zeros((4, 32, 32))
        data[:-1, -12:, 18:30] = gaussian_window(window_shape=(3, 12, 12))
        self.data = data

    def test_no_method(self):
        with self.assertRaises(ValueError):
            find_bragg(data=self.data, peak_method="wrong")


if __name__ == "__main__":
    run_tests(TestFindBragg)
