# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import os
import unittest

import numpy as np
from pyfakefs import fake_filesystem_unittest

import bcdi.utils.format as fmt
from bcdi.experiment.detector import Detector, create_detector
from bcdi.experiment.setup import Setup
from bcdi.utils.io_helper import ContextFile
from tests.config import load_config, run_tests

parameters, skip_tests = load_config("preprocessing")


class TestCreateRepr(unittest.TestCase):
    """
    Tests on the function format.create_repr.

    def create_repr(obj: type) -> str
    """

    def test_repr_setup(self):
        if skip_tests:
            self.skipTest(
                reason="This test can only run locally with the example dataset"
            )
        setup = Setup(parameters=parameters)
        valid = (
            "Setup(parameters={'scans': (11,), "
            "'root_folder': 'C:/Users/Jerome/Documents/data/CXIDB-I182/CH4760/', "
            "'save_dir': ['C:/Users/Jerome/Documents/data/CXIDB-I182/CH4760/test/'], "
            "'data_dir': ('C:/Users/Jerome/Documents/data/CXIDB-I182/CH4760/S11/',), "
            "'sample_name': ('S',), 'comment': ''"
            ""
        )
        out = fmt.create_repr(obj=setup, cls=Setup)
        self.assertTrue(out.startswith(valid))


class TestCreateReprFake(fake_filesystem_unittest.TestCase):
    """
    Tests on the function format.create_repr.

    def create_repr(obj: type) -> str
    """

    def setUp(self):
        self.setUpPyfakefs()
        self.valid_path = "/gpfs/bcdi/data/"
        self.filename = "dummy.spec"
        os.makedirs(self.valid_path)
        with open(self.valid_path + self.filename, "w") as f:
            f.write("dummy")

    def test_contextfile(self):
        ctx = ContextFile(filename=self.valid_path + self.filename, open_func=open)
        valid = (
            'ContextFile(filename="/gpfs/bcdi/data/dummy.spec", '
            'open_func=pyfakefs.fake_io.open, scan_number=None, mode="r", '
            'encoding="utf-8", longname=None, shortname=None, directory=None, )'
        )
        out = fmt.create_repr(obj=ctx, cls=ContextFile)
        self.assertEqual(out, valid)

    def test_detector(self):
        det = create_detector(name="Maxipix")
        valid = (
            'Maxipix(name="Maxipix", rootdir=None, datadir=None, savedir=None, '
            "template_imagefile=None, specfile=None, sample_name=None, "
            "roi=[0, 516, 0, 516], sum_roi=[0, 516, 0, 516], binning=(1, 1, 1), "
            "preprocessing_binning=(1, 1, 1), offsets=None, linearity_func=None, )"
        )
        out = fmt.create_repr(obj=det, cls=Detector)
        self.assertEqual(out, valid)

    def test_not_a_class(self):
        det = create_detector(name="Maxipix")
        with self.assertRaises(TypeError):
            fmt.create_repr(obj=det, cls="Detector")

    def test_empty_init(self):
        valid = "Empty()"

        class Empty:
            """This is an empty class"""

        out = fmt.create_repr(obj=Empty(), cls=Empty)
        self.assertEqual(out, valid)


class TestFormatRepr(unittest.TestCase):
    """
    Tests on the function format.format_repr.

    def format_repr(value: Optional[Any], quote_mark: bool = True) -> str:
    """

    def test_str(self):
        out = fmt.format_repr("test")
        self.assertEqual(out, '"test", ')

    def test_str_quote_mark_false(self):
        out = fmt.format_repr("test", quote_mark=False)
        self.assertEqual(out, "test, ")

    def test_float(self):
        out = fmt.format_repr(0.4)
        self.assertEqual(out, "0.4, ")

    def test_none(self):
        out = fmt.format_repr(None)
        self.assertEqual(out, "None, ")

    def test_tuple(self):
        out = fmt.format_repr((1.0, 2.0))
        self.assertEqual(out, "(1.0, 2.0), ")


class TestNdarrayToList(unittest.TestCase):
    """
    Tests on the function format.ndarray_to_list.

    def ndarray_to_list(array: np.ndarray) -> List
    """

    def test_not_an_array(self):
        with self.assertRaises(TypeError):
            fmt.ndarray_to_list(array=2.3)

    def test_none(self):
        with self.assertRaises(TypeError):
            fmt.ndarray_to_list(array=None)

    def test_1d_array_int(self):
        valid = [1, 2, 3]
        out = fmt.ndarray_to_list(array=np.array(valid))
        self.assertTrue(out == valid)

    def test_1d_array_float(self):
        valid = [1.12333333333333333333333333, 2.77, 3.5]
        out = fmt.ndarray_to_list(array=np.array(valid))
        self.assertTrue(out == valid)

    def test_2d_array_int(self):
        valid = [[1, 2, 3], [1.2, 3.333333333, 0]]
        out = fmt.ndarray_to_list(array=np.array(valid))
        self.assertTrue(out == valid)


if __name__ == "__main__":
    run_tests(TestCreateRepr)
    run_tests(TestFormatRepr)
    run_tests(TestNdarrayToList)
