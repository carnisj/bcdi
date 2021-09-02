# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#   (c) 06/2021-present : DESY CFEL
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import unittest
from bcdi.experiment.detector import create_detector, Detector, Dummy


def run_tests(test_class):
    suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


class TestDetector(unittest.TestCase):
    """Tests related to detector instantiation."""

    def test_create_detector_from_abc(self):
        with self.assertRaises(TypeError):
            Detector(name="Maxipix")

    def test_instantiation_missing_parameter(self):
        with self.assertRaises(TypeError):
            create_detector()


class TestDummy(unittest.TestCase):
    """Tests related to the Dummy detector."""

    def test_create_instance(self):
        self.assertIsInstance(Dummy("dummy"), Detector)

    def test_unbinned_pixel_number_wrong_type(self):
        with self.assertRaises(TypeError):
            Dummy("dummy", custom_pixelnumber=2)

    def test_unbinned_pixel_number_wrong_value(self):
        with self.assertRaises(ValueError):
            Dummy("dummy", custom_pixelnumber=(0, 2))

    def test_unbinned_pixel_number_partial_none(self):
        det = Dummy("dummy", custom_pixelnumber=(None, 2))
        self.assertTupleEqual(det.unbinned_pixel_number, (516, 516))

    def test_unbinned_pixel_number_default(self):
        det = Dummy("dummy")
        self.assertTupleEqual(det.unbinned_pixel_number, (516, 516))

    def test_unbinned_pixel_number_no_error(self):
        det = Dummy("dummy", custom_pixelnumber=(128, 256))
        self.assertTrue(det.unbinned_pixel_number[0] == 128 and
                        det.unbinned_pixel_number[1] == 256)

    def test_unbinned_pixel_size_wrong_type(self):
        with self.assertRaises(TypeError):
            Dummy("dummy", custom_pixelsize=(55e-6, 55e-6))

    def test_unbinned_pixel_size_wrong_value(self):
        with self.assertRaises(ValueError):
            Dummy("dummy", custom_pixelsize=0)

    def test_unbinned_pixel_size_none(self):
        det = Dummy("dummy", custom_pixelsize=None)
        self.assertTupleEqual(det.unbinned_pixel_size, (55e-06, 55e-06))

    def test_unbinned_pixel_size_default(self):
        det = Dummy("dummy")
        self.assertTupleEqual(det.unbinned_pixel_size, (55e-06, 55e-06))

    def test_unbinned_pixel_size_no_error(self):
        det = Dummy("dummy", custom_pixelsize=100e-6)
        self.assertAlmostEqual(det.unbinned_pixel_size[0], 100e-6)
        self.assertTrue(det.unbinned_pixel_size[0] == det.unbinned_pixel_size[1])


if __name__ == "__main__":
    run_tests(TestDetector)
    run_tests(TestDummy)
