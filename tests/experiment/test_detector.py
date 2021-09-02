# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#   (c) 06/2021-present : DESY CFEL
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import unittest
from bcdi.experiment.detector import (
    create_detector, Detector, Dummy, Merlin, Timepix, Eiger2M, Eiger4M, Maxipix
)


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


class TestMaxipix(unittest.TestCase):
    """Tests related to the Maxipix detector."""

    def setUp(self) -> None:
        self.det = Maxipix("Maxipix")

    def test_create_instance(self):
        self.assertIsInstance(self.det, Detector)

    def test_unbinned_pixel_number_default(self):
        self.assertTupleEqual(self.det.unbinned_pixel_number, (516, 516))


def test_unbinned_pixel_size_default(self):
    self.assertTupleEqual(self.det.unbinned_pixel_size, (55e-06, 55e-06))


class TestEiger2M(unittest.TestCase):
    """Tests related to the Eiger2M detector."""

    def setUp(self) -> None:
        self.det = Eiger2M("Eiger2M")

    def test_create_instance(self):
        self.assertIsInstance(self.det, Detector)

    def test_unbinned_pixel_number_default(self):
        self.assertTupleEqual(self.det.unbinned_pixel_number, (2164, 1030))

    def test_unbinned_pixel_size_default(self):
        self.assertTupleEqual(self.det.unbinned_pixel_size, (75e-06, 75e-06))


class TestEiger4M(unittest.TestCase):
    """Tests related to the Eiger4M detector."""

    def setUp(self) -> None:
        self.det = Eiger4M("Eiger4M")

    def test_create_instance(self):
        self.assertIsInstance(self.det, Detector)

    def test_unbinned_pixel_number_default(self):
        self.assertTupleEqual(self.det.unbinned_pixel_number, (2167, 2070))

    def test_unbinned_pixel_size_default(self):
        self.assertTupleEqual(self.det.unbinned_pixel_size, (75e-06, 75e-06))


class TestTimepix(unittest.TestCase):
    """Tests related to the Timepix detector."""

    def setUp(self) -> None:
        self.det = Timepix("Timepix")

    def test_create_instance(self):
        self.assertIsInstance(self.det, Detector)

    def test_unbinned_pixel_number_default(self):
        self.assertTupleEqual(self.det.unbinned_pixel_number, (256, 256))

    def test_unbinned_pixel_size_default(self):
        self.assertTupleEqual(self.det.unbinned_pixel_size, (55e-06, 55e-06))


class TestMerlin(unittest.TestCase):
    """Tests related to the Merlin detector."""

    def setUp(self) -> None:
        self.det = Merlin("Merlin")

    def test_create_instance(self):
        self.assertIsInstance(self.det, Detector)

    def test_unbinned_pixel_number_default(self):
        self.assertTupleEqual(self.det.unbinned_pixel_number, (515, 515))

    def test_unbinned_pixel_size_default(self):
        self.assertTupleEqual(self.det.unbinned_pixel_size, (55e-06, 55e-06))


class TestDummy(unittest.TestCase):
    """Tests related to the Dummy detector."""

    def setUp(self) -> None:
        self.det = Dummy("dummy")

    def test_create_instance(self):
        self.assertIsInstance(self.det, Detector)

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
        self.assertTupleEqual(self.det.unbinned_pixel_number, (516, 516))

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
        self.assertTupleEqual(self.det.unbinned_pixel_size, (55e-06, 55e-06))

    def test_unbinned_pixel_size_no_error(self):
        det = Dummy("dummy", custom_pixelsize=100e-6)
        self.assertAlmostEqual(det.unbinned_pixel_size[0], 100e-6)
        self.assertTrue(det.unbinned_pixel_size[0] == det.unbinned_pixel_size[1])


if __name__ == "__main__":
    run_tests(TestDetector)
    run_tests(TestMaxipix)
    run_tests(TestEiger2M)
    run_tests(TestEiger4M)
    run_tests(TestTimepix)
    run_tests(TestMerlin)
    run_tests(TestDummy)
