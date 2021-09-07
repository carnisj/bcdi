# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#   (c) 06/2021-present : DESY CFEL
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np
import unittest
from bcdi.experiment.detector import (
    create_detector,
    Detector,
    Dummy,
    Merlin,
    Timepix,
    Eiger2M,
    Eiger4M,
    Maxipix,
)


def run_tests(test_class):
    suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


class TestCreateDetector(unittest.TestCase):
    """Tests related to create_detector."""

    def test_create_maxipix(self):
        self.assertIsInstance(create_detector("Maxipix"), Maxipix)

    def test_create_eiger2m(self):
        self.assertIsInstance(create_detector("Eiger2M"), Eiger2M)

    def test_create_eiger4m(self):
        self.assertIsInstance(create_detector("Eiger4M"), Eiger4M)

    def test_create_timepix(self):
        self.assertIsInstance(create_detector("Timepix"), Timepix)

    def test_create_merlin(self):
        self.assertIsInstance(create_detector("Merlin"), Merlin)

    def test_create_dummy(self):
        self.assertIsInstance(create_detector("Dummy"), Dummy)

    def test_create_unknown_detector(self):
        with self.assertRaises(NotImplementedError):
            create_detector("unknown")

    def test_name_wrong_type(self):
        with self.assertRaises(NotImplementedError):
            create_detector(777)

    def test_name_wrong_none(self):
        with self.assertRaises(NotImplementedError):
            create_detector(None)

    def test_name_missing(self):
        with self.assertRaises(TypeError):
            create_detector()


class TestDetector(unittest.TestCase):
    """
    Tests related to the properties of the base class.

     Tests are performed via the instantiation of the Maxipix.
     """

    def setUp(self) -> None:
        self.det = Maxipix("Maxipix")

    def test_create_detector_from_abc(self):
        with self.assertRaises(TypeError):
            Detector(name="Maxipix")

    def test_linearity_function_not_callable(self):
        with self.assertRaises(TypeError):
            Maxipix("Maxipix", linearity_func=0)

    def test_linearity_function_callable(self):
        func = lambda x: x**2
        det = Maxipix("Maxipix", linearity_func=func)
        self.assertEqual(det._linearity_func, func)

    def test_binning_number(self):
        with self.assertRaises(TypeError):
            Maxipix(name="Maxipix", binning=2)

    def test_binning_list_wrong_type(self):
        with self.assertRaises(TypeError):
            Maxipix(name="Maxipix", binning=(2.0, 2.0, 1.0))

    def test_binning_list_wrong_length(self):
        with self.assertRaises(ValueError):
            Maxipix(name="Maxipix", binning=(2, 2))

    def test_binning_list_wrong_value(self):
        with self.assertRaises(ValueError):
            Maxipix(name="Maxipix", binning=(2, 2, 0))

    def test_binning_list_wrong_value_none(self):
        with self.assertRaises(ValueError):
            Maxipix(name="Maxipix", binning=(2, 2, None))

    def test_binning_none(self):
        with self.assertRaises(ValueError):
            Maxipix(name="Maxipix", binning=None)

    def test_binning_correct(self):
        det = Maxipix(name="Maxipix", binning=(2, 2, 1))
        self.assertEqual(det.binning, (2, 2, 1))

    def test_counter_beamline_wrong_type(self):
        with self.assertRaises(TypeError):
            self.det.counter(1)

    def test_counter_beamline_not_supported(self):
        self.assertEqual(self.det.counter("test"), None)

    def test_counter_beamline_supported(self):
        self.assertEqual(self.det.counter("ID01"), "mpx4inr")

    def test_datadir_wrong_type(self):
        with self.assertRaises(TypeError):
            Maxipix(name="Maxipix", datadir=0)

    def test_datadir_none(self):
        det = Maxipix(name="Maxipix", datadir=None)
        self.assertEqual(det.datadir, None)

    def test_datadir_correct(self):
        det = Maxipix(name="Maxipix", datadir="C:/test/")
        self.assertEqual(det.datadir, "C:/test/")

    def test_name(self):
        self.assertEqual(self.det.name, "Maxipix")

    def test_nb_pixel_x(self):
        # for Maxipix, unbinned_pixel_number = (516, 516)
        self.det.preprocessing_binning = (1, 1, 2)
        self.assertEqual(self.det.nb_pixel_x, 258)

    def test_nb_pixel_y(self):
        # for Maxipix, unbinned_pixel_number = (516, 516)
        self.det.preprocessing_binning = (1, 3, 2)
        self.assertEqual(self.det.nb_pixel_y, 172)

    def test_nb_pixel_y_truncated(self):
        # for Maxipix, unbinned_pixel_number = (516, 516)
        self.det.preprocessing_binning = (1, 7, 2)
        self.assertEqual(self.det.nb_pixel_y, 73)

    def test_params(self):
        self.assertIsInstance(self.det.params, dict)

    def test_pixelsize_x(self):
        # for Maxipix, unbinned_pixel_number = (55e-06, 55e-06)
        self.det.preprocessing_binning = (2, 2, 3)
        self.det.binning = (1, 2, 2)
        self.assertEqual(self.det.pixelsize_x, 330e-6)

    def test_pixelsize_y(self):
        # for Maxipix, unbinned_pixel_number = (55e-06, 55e-06)
        self.det.preprocessing_binning = (2, 2, 3)
        self.det.binning = (1, 2, 2)
        self.assertEqual(self.det.pixelsize_y, 220e-6)

    def test_preprocessing_binning_number(self):
        with self.assertRaises(TypeError):
            Maxipix(name="Maxipix", preprocessing_binning=2)

    def test_preprocessing_binning_list_wrong_type(self):
        with self.assertRaises(TypeError):
            Maxipix(name="Maxipix", preprocessing_binning=(2.0, 2.0, 1.0))

    def test_preprocessing_binning_list_wrong_length(self):
        with self.assertRaises(ValueError):
            Maxipix(name="Maxipix", preprocessing_binning=(2, 2))

    def test_preprocessing_binning_list_wrong_value(self):
        with self.assertRaises(ValueError):
            Maxipix(name="Maxipix", preprocessing_binning=(2, 2, 0))

    def test_preprocessing_binning_list_wrong_value_none(self):
        with self.assertRaises(ValueError):
            Maxipix(name="Maxipix", preprocessing_binning=(2, 2, None))

    def test_preprocessing_binning_none_default(self):
        det = Maxipix(name="Maxipix", preprocessing_binning=None)
        self.assertEqual(det.preprocessing_binning, (1, 1, 1))

    def test_preprocessing_binning_correct(self):
        det = Maxipix(name="Maxipix", preprocessing_binning=(2, 2, 1))
        self.assertEqual(det.preprocessing_binning, (2, 2, 1))

class TestMaxipix(unittest.TestCase):
    """Tests related to the Maxipix detector."""

    def setUp(self) -> None:
        self.det = Maxipix("Maxipix")
        self.data = np.ones(self.det.unbinned_pixel_number)
        self.mask = np.zeros(self.det.unbinned_pixel_number)

    def test_unbinned_pixel_number_default(self):
        self.assertTupleEqual(self.det.unbinned_pixel_number, (516, 516))

    def test_unbinned_pixel_size_default(self):
        self.assertTupleEqual(self.det.unbinned_pixel_size, (55e-06, 55e-06))

    def test_mask_gaps(self):
        data, mask = self.det._mask_gaps(data=self.data, mask=self.mask)
        self.assertTrue(np.all(data[:, 255:261]) == 0)
        self.assertTrue(np.all(data[255:261, :]) == 0)
        self.assertTrue(np.all(mask[:, 255:261]) == 1)
        self.assertTrue(np.all(mask[255:261, :]) == 1)


class TestEiger2M(unittest.TestCase):
    """Tests related to the Eiger2M detector."""

    def setUp(self) -> None:
        self.det = Eiger2M("Eiger2M")
        self.data = np.ones(self.det.unbinned_pixel_number)
        self.mask = np.zeros(self.det.unbinned_pixel_number)

    def test_unbinned_pixel_number_default(self):
        self.assertTupleEqual(self.det.unbinned_pixel_number, (2164, 1030))

    def test_unbinned_pixel_size_default(self):
        self.assertTupleEqual(self.det.unbinned_pixel_size, (75e-06, 75e-06))

    def test_mask_gaps(self):
        data, mask = self.det._mask_gaps(data=self.data, mask=self.mask)
        self.assertTrue(np.all(data[:, 255:259]) == 0)
        self.assertTrue(np.all(data[:, 513:517]) == 0)
        self.assertTrue(np.all(data[:, 771:775]) == 0)
        self.assertTrue(np.all(data[0:257, 72:80]) == 0)
        self.assertTrue(np.all(data[255:259, :]) == 0)
        self.assertTrue(np.all(data[511:552, :]) == 0)
        self.assertTrue(np.all(data[804:809, :]) == 0)
        self.assertTrue(np.all(data[1061:1102, :]) == 0)
        self.assertTrue(np.all(data[1355:1359, :]) == 0)
        self.assertTrue(np.all(data[1611:1652, :]) == 0)
        self.assertTrue(np.all(data[1905:1909, :]) == 0)
        self.assertTrue(np.all(data[1248:1290, 478]) == 0)
        self.assertTrue(np.all(data[1214:1298, 481]) == 0)
        self.assertTrue(np.all(data[1649:1910, 620:628]) == 0)

        self.assertTrue(np.all(mask[:, 255:259]) == 1)
        self.assertTrue(np.all(mask[:, 513:517]) == 1)
        self.assertTrue(np.all(mask[:, 771:775]) == 1)
        self.assertTrue(np.all(mask[0:257, 72:80]) == 1)
        self.assertTrue(np.all(mask[255:259, :]) == 1)
        self.assertTrue(np.all(mask[511:552, :]) == 1)
        self.assertTrue(np.all(mask[804:809, :]) == 1)
        self.assertTrue(np.all(mask[1061:1102, :]) == 1)
        self.assertTrue(np.all(mask[1355:1359, :]) == 1)
        self.assertTrue(np.all(mask[1611:1652, :]) == 1)
        self.assertTrue(np.all(mask[1905:1909, :]) == 1)
        self.assertTrue(np.all(mask[1248:1290, 478]) == 1)
        self.assertTrue(np.all(mask[1214:1298, 481]) == 1)
        self.assertTrue(np.all(mask[1649:1910, 620:628]) == 1)


class TestEiger4M(unittest.TestCase):
    """Tests related to the Eiger4M detector."""

    def setUp(self) -> None:
        self.det = Eiger4M("Eiger4M")
        self.data = np.ones(self.det.unbinned_pixel_number)
        self.mask = np.zeros(self.det.unbinned_pixel_number)

    def test_unbinned_pixel_number_default(self):
        self.assertTupleEqual(self.det.unbinned_pixel_number, (2167, 2070))

    def test_unbinned_pixel_size_default(self):
        self.assertTupleEqual(self.det.unbinned_pixel_size, (75e-06, 75e-06))

    def test_mask_gaps(self):
        data, mask = self.det._mask_gaps(data=self.data, mask=self.mask)
        self.assertTrue(np.all(data[:, 0:1]) == 0)
        self.assertTrue(np.all(data[:, 0:1]) == 0)
        self.assertTrue(np.all(data[:, -1:]) == 0)
        self.assertTrue(np.all(data[0:1, :]) == 0)
        self.assertTrue(np.all(data[-1:, :]) == 0)
        self.assertTrue(np.all(data[:, 1029:1041]) == 0)
        self.assertTrue(np.all(data[513:552, :]) == 0)
        self.assertTrue(np.all(data[1064:1103, :]) == 0)
        self.assertTrue(np.all(data[1615:1654, :]) == 0)

        self.assertTrue(np.all(mask[:, 0:1]) == 1)
        self.assertTrue(np.all(mask[:, -1:]) == 1)
        self.assertTrue(np.all(mask[0:1, :]) == 1)
        self.assertTrue(np.all(mask[-1:, :]) == 1)
        self.assertTrue(np.all(mask[:, 1029:1041]) == 1)
        self.assertTrue(np.all(mask[513:552, :]) == 1)
        self.assertTrue(np.all(mask[1064:1103, :]) == 1)
        self.assertTrue(np.all(mask[1615:1654, :]) == 1)


class TestTimepix(unittest.TestCase):
    """Tests related to the Timepix detector."""

    def setUp(self) -> None:
        self.det = Timepix("Timepix")

    def test_unbinned_pixel_number_default(self):
        self.assertTupleEqual(self.det.unbinned_pixel_number, (256, 256))

    def test_unbinned_pixel_size_default(self):
        self.assertTupleEqual(self.det.unbinned_pixel_size, (55e-06, 55e-06))


class TestMerlin(unittest.TestCase):
    """Tests related to the Merlin detector."""

    def setUp(self) -> None:
        self.det = Merlin("Merlin")
        self.data = np.ones(self.det.unbinned_pixel_number)
        self.mask = np.zeros(self.det.unbinned_pixel_number)

    def test_unbinned_pixel_number_default(self):
        self.assertTupleEqual(self.det.unbinned_pixel_number, (515, 515))

    def test_unbinned_pixel_size_default(self):
        self.assertTupleEqual(self.det.unbinned_pixel_size, (55e-06, 55e-06))

    def test_mask_gaps(self):
        data, mask = self.det._mask_gaps(data=self.data, mask=self.mask)
        self.assertTrue(np.all(data[:, 255:260]) == 0)
        self.assertTrue(np.all(data[255:260, :]) == 0)
        self.assertTrue(np.all(mask[:, 255:260]) == 1)
        self.assertTrue(np.all(mask[255:260, :]) == 1)


class TestDummy(unittest.TestCase):
    """Tests related to the Dummy detector."""

    def setUp(self) -> None:
        self.det = Dummy("dummy")

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
        self.assertTrue(
            det.unbinned_pixel_number[0] == 128 and det.unbinned_pixel_number[1] == 256
        )

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
    run_tests(TestCreateDetector)
    run_tests(TestDetector)
    run_tests(TestMaxipix)
    run_tests(TestEiger2M)
    run_tests(TestEiger4M)
    run_tests(TestTimepix)
    run_tests(TestMerlin)
    run_tests(TestDummy)
