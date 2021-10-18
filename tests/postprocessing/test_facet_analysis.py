# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#   (c) 06/2021-present : DESY CFEL
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import os
import unittest
from bcdi.postprocessing.facet_analysis import Facets

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FILENAME = "3572_fa.vtk"


def run_tests(test_class):
    suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


class TestInitFacets(unittest.TestCase):
    """
    Tests on the class Facets.

    __init__(self, filename : str,pathdir : str = "./",lattice : float = 3.912) -> None:
    """

    # def setUp(self):
    #     # executed before each test
    #     facet = Facets(pathdir=THIS_DIR, filename="3572_fa.vtk")

    def test_init_pathdir_empty(self):
        with self.assertRaises(ValueError):
            Facets(pathdir="", filename=FILENAME)

    def test_init_pathdir_None(self):
        with self.assertRaises(ValueError):
            Facets(pathdir=None, filename=FILENAME)

    def test_init_pathdir_wrong_type(self):
        with self.assertRaises(TypeError):
            Facets(pathdir=0, filename=FILENAME)

    def test_init_filename_empty(self):
        with self.assertRaises(ValueError):
            Facets(pathdir=THIS_DIR, filename="")

    def test_init_filename_None(self):
        with self.assertRaises(ValueError):
            Facets(pathdir=THIS_DIR, filename=None)

    def test_init_filename_wrong_type(self):
        with self.assertRaises(TypeError):
            Facets(pathdir=THIS_DIR, filename=0)

    def test_init_lattice_None(self):
        with self.assertRaises(ValueError):
            Facets(pathdir=THIS_DIR, filename=FILENAME, lattice=None)

    def test_init_lattice_int(self):
        with self.assertRaises(TypeError):
            Facets(pathdir=THIS_DIR, filename=FILENAME, lattice=int(1))

    def test_init_lattice_str(self):
        with self.assertRaises(TypeError):
            Facets(pathdir=THIS_DIR, filename=FILENAME, lattice=str(3.912))


if __name__ == "__main__":
    run_tests(TestInitFacets)