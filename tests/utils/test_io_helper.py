# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import io
import unittest
from pathlib import Path

from bcdi.utils.io_helper import ContextFile
from tests.config import run_tests

here = Path(__file__).parent
CONFIG = str(here.parents[1] / "bcdi/examples/S11_config_preprocessing.yml")


class TestContextFile(unittest.TestCase):
    """Tests related to ContextFile class."""

    def setUp(self):
        self.filename = CONFIG
        self.open_func = io.open

    def test_instantiate_class(self):
        ctx = ContextFile(filename=self.filename, open_func=self.open_func)
        self.assertTrue(ctx.filename == self.filename)
        self.assertTrue(ctx.mode == "r")
        self.assertTrue(ctx.encoding == "utf-8")
        self.assertIsNone(ctx.file)
        self.assertIsNone(ctx.longname)
        self.assertIsNone(ctx.scan_number)
        self.assertIsNone(ctx.shortname)
        self.assertIsNone(ctx.directory)

    def test_open_function(self):
        with ContextFile(filename=self.filename, open_func=self.open_func) as file:
            try:
                first_line = next(file).split()
            except StopIteration:
                pass
        self.assertTrue(first_line[0] == "scans:" and first_line[1] == "11")

    def test_open_function_wrong_type(self):
        with self.assertRaises(TypeError):
            ContextFile(filename=self.filename, open_func="weird function")

    def test_open_function_not_supported(self):
        ctx = ContextFile(filename=self.filename, open_func=int)
        with self.assertRaises(NotImplementedError):
            ctx.__enter__()

    def test_scan_number_wrong_type(self):
        with self.assertRaises(TypeError):
            ContextFile(filename=self.filename, open_func=open, scan_number="1")

    def test_scan_number_wrong_type_float(self):
        with self.assertRaises(TypeError):
            ContextFile(filename=self.filename, open_func=open, scan_number=1.0)

    def test_exit_function(self):
        with ContextFile(filename=self.filename, open_func=self.open_func) as file:
            pass
        with self.assertRaises(ValueError):
            # file already closed
            try:
                next(file)
            except StopIteration:
                pass

    def test_repr(self):
        ctx = ContextFile(
            filename=self.filename, open_func=self.open_func, shortname="test"
        )
        self.assertIsInstance(eval(repr(ctx)), ContextFile)

    def test_repr_str_none(self):
        ctx = ContextFile(
            filename=self.filename, open_func=self.open_func, shortname=None
        )
        self.assertIsInstance(eval(repr(ctx)), ContextFile)


if __name__ == "__main__":
    run_tests(TestContextFile)
