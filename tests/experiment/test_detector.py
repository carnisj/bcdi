# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import os
import unittest
from unittest.mock import patch

import numpy as np
from pyfakefs import fake_filesystem_unittest

from bcdi.experiment.detector import (
    Detector,
    Dummy,
    Eiger2M,
    Eiger4M,
    Lambda,
    Maxipix,
    Merlin,
    Timepix,
    create_detector,
)
from tests.config import run_tests


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

    def test_create_lambda(self):
        self.assertIsInstance(create_detector("Lambda"), Lambda)

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


class TestDetector(fake_filesystem_unittest.TestCase):
    """
    Tests related to the properties of the base class.

     Tests are performed via the instantiation of the Maxipix.
    """

    def setUp(self) -> None:
        self.setUpPyfakefs()
        self.valid_path = "/gpfs/bcdi/data"
        os.makedirs(self.valid_path)
        self.det = Maxipix("Maxipix")

    def test_create_detector_from_abc(self):
        with self.assertRaises(TypeError):
            Detector(name="Maxipix")

    def test_binning_number(self):
        with self.assertRaises(TypeError):
            Maxipix(name="Maxipix", binning=2)

    def test_binning_list_wrong_type(self):
        with self.assertRaises(TypeError):
            Maxipix(name="Maxipix", binning=[2.0, 2.0, 1.0])

    def test_binning_list_wrong_length(self):
        with self.assertRaises(ValueError):
            Maxipix(name="Maxipix", binning=[2, 2])

    def test_binning_list_wrong_value(self):
        with self.assertRaises(ValueError):
            Maxipix(name="Maxipix", binning=[2, 2, 0])

    def test_binning_list_wrong_value_none(self):
        with self.assertRaises(ValueError):
            Maxipix(name="Maxipix", binning=[2, 2, None])

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
        det = Maxipix(name="Maxipix", datadir=self.valid_path)
        self.assertEqual(det.datadir, self.valid_path)

    def test_datadir_not_exist(self):
        with self.assertRaises(ValueError):
            Maxipix(name="Maxipix", datadir="this directory does not exist")

    def test_linearity_function_wrong_type(self):
        with self.assertRaises(TypeError):
            Maxipix("Maxipix", linearity_func=0)

    def test_linearity_function_none(self):
        det = Maxipix("Maxipix", linearity_func=None)
        self.assertEqual(det.linearity_func, None)

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
            Maxipix(name="Maxipix", preprocessing_binning=[2.0, 2.0, 1.0])

    def test_preprocessing_binning_list_wrong_length(self):
        with self.assertRaises(ValueError):
            Maxipix(name="Maxipix", preprocessing_binning=[2, 2])

    def test_preprocessing_binning_list_wrong_value(self):
        with self.assertRaises(ValueError):
            Maxipix(name="Maxipix", preprocessing_binning=[2, 2, 0])

    def test_preprocessing_binning_list_wrong_value_none(self):
        with self.assertRaises(ValueError):
            Maxipix(name="Maxipix", preprocessing_binning=[2, 2, None])

    def test_preprocessing_binning_none_default(self):
        with self.assertRaises(ValueError):
            Maxipix(name="Maxipix", preprocessing_binning=None)

    def test_preprocessing_binning_correct(self):
        det = Maxipix(name="Maxipix", preprocessing_binning=(2, 2, 1))
        self.assertEqual(det.preprocessing_binning, (2, 2, 1))

    def test_roi_number(self):
        with self.assertRaises(TypeError):
            Maxipix(name="Maxipix", roi=2)

    def test_roi_list_wrong_type(self):
        with self.assertRaises(TypeError):
            Maxipix(name="Maxipix", roi=[2.0, 512, 12, 35])

    def test_roi_list_wrong_length(self):
        with self.assertRaises(ValueError):
            Maxipix(name="Maxipix", roi=[2, 2])

    def test_roi_list_wrong_value_none(self):
        with self.assertRaises(ValueError):
            Maxipix(name="Maxipix", roi=[None, 512, 12, 35])

    def test_roi_list_wrong_value_decreasing_y(self):
        with self.assertRaises(ValueError):
            Maxipix(name="Maxipix", roi=[128, 0, 12, 35])

    def test_roi_list_wrong_value_decreasing_x(self):
        with self.assertRaises(ValueError):
            Maxipix(name="Maxipix", roi=[0, 256, 128, 35])

    def test_roi_none_default(self):
        det = Maxipix(name="Maxipix", roi=None)
        self.assertEqual(det.roi, [0, self.det.nb_pixel_y, 0, self.det.nb_pixel_x])

    def test_roi_correct_tuple(self):
        det = Maxipix(name="Maxipix", roi=(2, 252, 1, 35))
        self.assertEqual(det.roi, (2, 252, 1, 35))

    def test_rootdir_exists(self):
        det = Maxipix(name="Maxipix", rootdir=self.valid_path)
        self.assertEqual(det.rootdir, self.valid_path)

    def test_rootdir_not_exist(self):
        with self.assertRaises(ValueError):
            Maxipix(name="Maxipix", rootdir="this directory does not exist")

    def test_rootdir_wrong_type(self):
        with self.assertRaises(TypeError):
            Maxipix(name="Maxipix", rootdir=777)

    def test_rootdir_wrong_length(self):
        with self.assertRaises(ValueError):
            Maxipix(name="Maxipix", rootdir="")

    def test_rootdir_None(self):
        det = Maxipix(name="Maxipix", rootdir=None)
        self.assertEqual(det.rootdir, None)

    def test_sample_name_wrong_type(self):
        with self.assertRaises(TypeError):
            Maxipix(name="Maxipix", sample_name=777)

    def test_sample_name_wrong_length(self):
        det = Maxipix(name="Maxipix", sample_name="")
        self.assertEqual(det.sample_name, "")

    def test_sample_name_None(self):
        det = Maxipix(name="Maxipix", sample_name=None)
        self.assertEqual(det.sample_name, None)

    def test_sample_name_correct(self):
        det = Maxipix(name="Maxipix", sample_name="S")
        self.assertEqual(det.sample_name, "S")

    def test_scandir_datadir_defined(self):
        dir_path = os.path.abspath(os.path.join(self.valid_path, os.pardir)) + "/"
        det = Maxipix(name="Maxipix", datadir=self.valid_path)
        self.assertEqual(det.scandir, dir_path.replace("\\", "/"))

    def test_scandir_datadir_none(self):
        det = Maxipix(name="Maxipix", datadir=None)
        self.assertEqual(det.scandir, None)

    def test_sum_roi_number(self):
        with self.assertRaises(TypeError):
            Maxipix(name="Maxipix", sum_roi=2)

    def test_sum_roi_list_wrong_type(self):
        with self.assertRaises(TypeError):
            Maxipix(name="Maxipix", sum_roi=[2.0, 512, 12, 35])

    def test_sum_roi_list_wrong_length(self):
        with self.assertRaises(ValueError):
            Maxipix(name="Maxipix", sum_roi=[2, 2])

    def test_sum_roi_list_wrong_value_none(self):
        with self.assertRaises(ValueError):
            Maxipix(name="Maxipix", sum_roi=[None, 512, 12, 35])

    def test_sum_roi_list_wrong_value_decreasing_y(self):
        with self.assertRaises(ValueError):
            Maxipix(name="Maxipix", sum_roi=[128, 0, 12, 35])

    def test_sum_roi_list_wrong_value_decreasing_x(self):
        with self.assertRaises(ValueError):
            Maxipix(name="Maxipix", sum_roi=[0, 256, 128, 35])

    def test_sum_roi_none_default(self):
        det = Maxipix(name="Maxipix", sum_roi=None)
        self.assertEqual(det.sum_roi, [0, self.det.nb_pixel_y, 0, self.det.nb_pixel_x])

    def test_sum_roi_none_default_roi_defined(self):
        det = Maxipix(name="Maxipix", sum_roi=None, roi=(2, 252, 1, 35))
        self.assertEqual(det.sum_roi, det.roi)
        self.assertEqual(det.sum_roi, (2, 252, 1, 35))

    def test_sum_roi_empty_list(self):
        det = Maxipix(name="Maxipix", sum_roi=(), roi=(2, 252, 1, 35))
        self.assertEqual(det.sum_roi, det.roi)
        self.assertEqual(det.sum_roi, (2, 252, 1, 35))

    def test_template_imagefile_wrong_type(self):
        with self.assertRaises(TypeError):
            Maxipix(name="Maxipix", template_imagefile=777)

    def test_template_imagefile_None(self):
        det = Maxipix(name="Maxipix", template_imagefile=None)
        self.assertEqual(det.template_imagefile, None)

    def test_template_imagefile_correct(self):
        det = Maxipix(name="Maxipix", template_imagefile="S")
        self.assertEqual(det.template_imagefile, "S")

    def test_background_subtraction_correct(self):
        data = np.ones((3, 3))
        background = np.ones((3, 3))
        self.assertTrue(
            np.all(
                np.isclose(
                    self.det._background_subtraction(data, background), np.zeros((3, 3))
                )
            )
        )

    def test_background_subtraction_float(self):
        data = np.ones((3, 3))
        background = 0.5 * np.ones((3, 3))
        self.assertTrue(
            np.all(
                np.isclose(
                    self.det._background_subtraction(data, background), background
                )
            )
        )

    def test_background_subtraction_wrong_ndim(self):
        data = np.ones((3, 3, 3))
        background = 0.5 * np.ones((3, 3))
        with self.assertRaises(ValueError):
            self.det._background_subtraction(data, background)

    def test_background_subtraction_wrong_type(self):
        data = 5
        background = 0.5 * np.ones((3, 3))
        with self.assertRaises(TypeError):
            self.det._background_subtraction(data, background)

    def test_background_subtraction_wrong_shape(self):
        data = np.ones((3, 3))
        background = 0.5 * np.ones((3, 4))
        with self.assertRaises(ValueError):
            self.det._background_subtraction(data, background)

    def test_background_subtraction_none(self):
        data = np.ones((3, 3))
        background = None
        self.assertTrue(
            np.all(np.isclose(self.det._background_subtraction(data, background), data))
        )

    def test_flatfield_correction_correct(self):
        data = np.ones((3, 3))
        flatfield = np.ones((3, 3))
        self.assertTrue(
            np.all(np.isclose(self.det._flatfield_correction(data, flatfield), data))
        )

    def test_flatfield_correction_float(self):
        data = np.ones((3, 3))
        flatfield = 0.5 * np.ones((3, 3))
        self.assertTrue(
            np.all(
                np.isclose(self.det._flatfield_correction(data, flatfield), flatfield)
            )
        )

    def test_flatfield_correction_wrong_ndim(self):
        data = np.ones((3, 3, 3))
        flatfield = 0.5 * np.ones((3, 3))
        with self.assertRaises(ValueError):
            self.det._flatfield_correction(data, flatfield)

    def test_flatfield_correction_wrong_type(self):
        data = 5
        flatfield = 0.5 * np.ones((3, 3))
        with self.assertRaises(TypeError):
            self.det._flatfield_correction(data, flatfield)

    def test_flatfield_correction_wrong_shape(self):
        data = np.ones((3, 3))
        flatfield = 0.5 * np.ones((3, 4))
        with self.assertRaises(ValueError):
            self.det._flatfield_correction(data, flatfield)

    def test_flatfield_correction_none(self):
        data = np.ones((3, 3))
        flatfield = None
        self.assertTrue(
            np.all(np.isclose(self.det._flatfield_correction(data, flatfield), data))
        )

    def test_hotpixels_correction_correct(self):
        data = np.ones((3, 3))
        mask = np.zeros((3, 3))
        hotpixels = np.ones((3, 3))
        output = self.det._hotpixels_correction(data, mask, hotpixels)
        self.assertTrue(np.all(np.isclose(output[0], np.zeros((3, 3)))))
        self.assertTrue(np.all(np.isclose(output[1], np.ones((3, 3)))))

    def test_hotpixels_correction_wrong_ndim(self):
        data = np.ones((3, 3, 3))
        mask = np.zeros((3, 3))
        hotpixels = np.ones((3, 3))
        with self.assertRaises(ValueError):
            self.det._hotpixels_correction(data, mask, hotpixels)

    def test_hotpixels_correction_wrong_type(self):
        data = 5
        mask = np.zeros((3, 3))
        hotpixels = np.ones((3, 3))
        with self.assertRaises(TypeError):
            self.det._hotpixels_correction(data, mask, hotpixels)

    def test_hotpixels_correction_wrong_shape(self):
        data = np.ones((3, 3))
        mask = np.zeros((3, 3))
        hotpixels = np.ones((3, 4))
        with self.assertRaises(ValueError):
            self.det._hotpixels_correction(data, mask, hotpixels)

    def test_hotpixels_correction_wrong_value(self):
        data = np.ones((3, 3))
        mask = np.zeros((3, 3))
        hotpixels = 2 * np.ones((3, 3))
        with self.assertRaises(ValueError):
            self.det._hotpixels_correction(data, mask, hotpixels)

    def test_hotpixels_correction_none(self):
        data = np.ones((3, 3))
        mask = np.zeros((3, 3))
        hotpixels = None
        output = self.det._hotpixels_correction(data, mask, hotpixels)
        self.assertTrue(np.all(np.isclose(output[0], data)))
        self.assertTrue(np.all(np.isclose(output[1], mask)))

    def test_linearity_correction_correct(self):
        data = np.ones((3, 3))
        self.det.linearity_func = [0, 0, 1, 0, 0]
        self.assertTrue(np.all(np.isclose(self.det._linearity_correction(data), data)))

    def test_linearity_correction_correct_2(self):
        data = np.ones((3, 3))
        self.det.linearity_func = [5, 4, 3, 2, 1]
        self.assertTrue(
            np.all(np.isclose(self.det._linearity_correction(data), 15 * data))
        )

    def test_linearity_correction_correct_3(self):
        np.random.seed(0)
        data = np.random.rand(3, 3)
        correct = np.array(
            [
                [0.47218707, 0.86316156, 0.55738783],
                [0.4674014, 0.40471678, 0.65317796],
                [0.40416385, 1.75428095, 2.28272669],
            ]
        )
        self.det.linearity_func = [-0.23, 4.12, -0.3, -2, 1]
        self.assertTrue(
            np.all(np.isclose(self.det._linearity_correction(data), correct))
        )

    def test_linearity_correction_zero(self):
        data = np.ones((3, 3))
        self.det.linearity_func = [0, 0, 0, 0, 0]
        self.assertTrue(
            np.all(np.isclose(self.det._linearity_correction(data), np.zeros((3, 3))))
        )

    def test_linearity_correction_none(self):
        data = np.ones((3, 3))
        self.det.linearity_func = None
        self.assertTrue(np.all(np.isclose(self.det._linearity_correction(data), data)))

    def test_linearity_correction_wrong_ndim(self):
        data = np.ones((3, 3, 3))
        self.det.linearity_func = [3, 5, 6, 1, 2]
        with self.assertRaises(ValueError):
            self.det._linearity_correction(data)

    def test_mask_detector_no_linearity_correction(self):
        det = Timepix("Timepix", linearity_func=None)
        data = np.ones(det.unbinned_pixel_number)
        mask = np.zeros(det.unbinned_pixel_number)
        output = det.mask_detector(data, mask, nb_frames=1)
        self.assertTrue(np.all(np.isclose(output[0], data)))
        self.assertTrue(np.all(np.isclose(output[1], mask)))

    def test_mask_detector_linearity_correction(self):
        det = Timepix("Timepix", linearity_func=[0, 0, 0, 2, 0])
        data = np.ones(det.unbinned_pixel_number)
        mask = np.zeros(det.unbinned_pixel_number)
        output = det.mask_detector(data, mask, nb_frames=1)
        self.assertTrue(np.all(np.isclose(output[0], 2 * data)))
        self.assertTrue(np.all(np.isclose(output[1], mask)))

    def test_mask_detector_wrong_type(self):
        data = 1
        mask = np.zeros((3, 3, 3))
        with self.assertRaises(TypeError):
            self.det.mask_detector(data, mask)

    def test_mask_detector_wrong_ndim(self):
        data = np.ones((3, 3))
        mask = np.zeros((3, 3, 3))
        with self.assertRaises(ValueError):
            self.det.mask_detector(data, mask)

    def test_mask_detector_shape_mismatch(self):
        data = np.ones((3, 3))
        mask = np.zeros((3, 4))
        with self.assertRaises(ValueError):
            self.det.mask_detector(data, mask)

    def test_mask_detector_invalid_shape(self):
        """Shape of Maxipix is (516, 516)."""
        data = np.ones((3, 3))
        mask = np.zeros((3, 3))
        with self.assertRaises(ValueError):
            self.det.mask_detector(data, mask)

    @patch("bcdi.experiment.detector.Detector.__abstractmethods__", set())
    def test_mask_gaps_base_class(self):
        det = Detector("Maxipix")
        data = np.ones((1, 1))
        mask = np.zeros((1, 1))
        output = det._mask_gaps(data, mask)
        self.assertTrue(np.all(np.isclose(output[0], data)))
        self.assertTrue(np.all(np.isclose(output[1], mask)))

    @patch("bcdi.experiment.detector.Detector.__abstractmethods__", set())
    def test_mask_gaps_base_class_wrong_ndim(self):
        det = Detector("Maxipix")
        data = np.ones((1, 1))
        mask = np.zeros((1, 1, 3))
        with self.assertRaises(ValueError):
            det._mask_gaps(data, mask)

    @patch("bcdi.experiment.detector.Detector.__abstractmethods__", set())
    def test_mask_gaps_base_class_wrong_type(self):
        det = Detector("Maxipix")
        data = 1
        mask = np.zeros((1, 1))
        with self.assertRaises(TypeError):
            det._mask_gaps(data, mask)

    @patch("bcdi.experiment.detector.Detector.__abstractmethods__", set())
    def test_mask_gaps_base_class_wrong_shape(self):
        det = Detector("Maxipix")
        data = np.ones((1, 1))
        mask = np.zeros((1, 2))
        with self.assertRaises(ValueError):
            det._mask_gaps(data, mask)

    def test_saturation_correction_above(self):
        det = Maxipix("Maxipix")
        det.saturation_threshold = 10
        data = np.ones((3, 3)) * 11
        mask = np.zeros((3, 3))
        output = det._saturation_correction(data, mask, nb_frames=1)
        self.assertTrue(np.all(np.isclose(output[0], np.zeros(data.shape))))
        self.assertTrue(np.all(np.isclose(output[1], np.ones(data.shape))))

    def test_saturation_correction_edge_case(self):
        det = Maxipix("Maxipix")
        det.saturation_threshold = 10
        data = np.ones((3, 3)) * 10
        mask = np.zeros((3, 3))
        output = det._saturation_correction(data, mask, nb_frames=1)
        self.assertTrue(np.all(np.isclose(output[0], data)))
        self.assertTrue(np.all(np.isclose(output[1], mask)))

    def test_saturation_correction_shape_mismatch(self):
        det = Maxipix("Maxipix")
        det.saturation_threshold = 10
        data = np.ones((3, 3)) * 11
        mask = np.zeros((3, 4))
        with self.assertRaises(ValueError):
            det._saturation_correction(data, mask, nb_frames=1)

    def test_saturation_correction_wrong_ndim(self):
        det = Maxipix("Maxipix")
        det.saturation_threshold = 10
        data = np.ones((3, 3)) * 11
        mask = np.zeros((3, 3, 3))
        with self.assertRaises(ValueError):
            det._saturation_correction(data, mask, nb_frames=1)

    def test_saturation_correction_wrong_type(self):
        det = Maxipix("Maxipix")
        det.saturation_threshold = 10
        data = 11
        mask = np.zeros((3, 3))
        with self.assertRaises(TypeError):
            det._saturation_correction(data, mask, nb_frames=1)

    def test_saturation_correction_nbframes_2(self):
        det = Maxipix("Maxipix")
        det.saturation_threshold = 10
        data = np.ones((3, 3)) * 11
        mask = np.zeros((3, 3))
        output = det._saturation_correction(data, mask, nb_frames=2)
        self.assertTrue(np.all(np.isclose(output[0], data)))
        self.assertTrue(np.all(np.isclose(output[1], mask)))

    def test_saturation_correction_nbframes_wrong_type(self):
        det = Maxipix("Maxipix")
        det.saturation_threshold = 10
        data = np.ones((3, 3)) * 11
        mask = np.zeros((3, 3))
        with self.assertRaises(TypeError):
            det._saturation_correction(data, mask, nb_frames=2.0)

    def test_saturation_correction_nbframes_wrong_value(self):
        det = Maxipix("Maxipix")
        det.saturation_threshold = 10
        data = np.ones((3, 3)) * 11
        mask = np.zeros((3, 3))
        with self.assertRaises(ValueError):
            det._saturation_correction(data, mask, nb_frames=0)

    def test_repr(self):
        self.assertIsInstance(eval(repr(self.det)), Maxipix)

    def test_repr_str_not_None(self):
        self.det.template_file = "test"
        self.assertIsInstance(eval(repr(self.det)), Maxipix)


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


class TestLambda(unittest.TestCase):
    """Tests related to the Lambda detector."""

    def setUp(self) -> None:
        self.det = Lambda("Lambda")
        self.saturation_threshold = 1.5e6

    def test_name(self):
        self.assertEqual(self.det.name, "Lambda")

    def test_unbinned_pixel_size(self):
        self.assertTupleEqual(self.det.unbinned_pixel_size, (55e-06, 55e-06))

    def test_unbinned_pixel_number(self):
        self.assertTupleEqual(self.det.unbinned_pixel_number, (516, 516))

    def test_saturation_threshold(self):
        self.assertEqual(self.det.saturation_threshold, self.saturation_threshold)


if __name__ == "__main__":
    run_tests(TestCreateDetector)
    run_tests(TestDetector)
    run_tests(TestMaxipix)
    run_tests(TestEiger2M)
    run_tests(TestEiger4M)
    run_tests(TestTimepix)
    run_tests(TestMerlin)
    run_tests(TestDummy)
    run_tests(TestLambda)
