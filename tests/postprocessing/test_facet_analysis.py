# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#   (c) 06/2021-present : DESY CFEL
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

from pandas import DataFrame
from pathlib import Path
import unittest
from bcdi.postprocessing.facet_analysis import Facets

here = Path(__file__).parent
THIS_DIR = str(here)
SAVEDIR = str(here.parents[1] / "test_output/")
FILENAME = "3572_fa.vtk"


def run_tests(test_class):
    suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


class TestInitFacetsParams(unittest.TestCase):
    """
    Tests on the class Facets.

    __init__(self, filename : str,pathdir : str = "./",lattice : float = 3.912) -> None:
    """

    def test_init_pathdir_empty(self):
        with self.assertRaises(ValueError):
            Facets(pathdir="", filename=FILENAME, savedir=SAVEDIR)

    def test_init_pathdir_None(self):
        with self.assertRaises(ValueError):
            Facets(pathdir=None, filename=FILENAME, savedir=SAVEDIR)

    def test_init_pathdir_wrong_type(self):
        with self.assertRaises(TypeError):
            Facets(pathdir=0, filename=FILENAME, savedir=SAVEDIR)

    def test_init_filename_empty(self):
        with self.assertRaises(ValueError):
            Facets(pathdir=THIS_DIR, filename="", savedir=SAVEDIR)

    def test_init_filename_None(self):
        with self.assertRaises(ValueError):
            Facets(pathdir=THIS_DIR, filename=None, savedir=SAVEDIR)

    def test_init_filename_wrong_type(self):
        with self.assertRaises(TypeError):
            Facets(pathdir=THIS_DIR, filename=0, savedir=SAVEDIR)

    def test_init_lattice_None(self):
        with self.assertRaises(ValueError):
            Facets(pathdir=THIS_DIR, filename=FILENAME, savedir=SAVEDIR, lattice=None)

    def test_init_lattice_int(self):
        with self.assertRaises(TypeError):
            Facets(pathdir=THIS_DIR, filename=FILENAME, savedir=SAVEDIR, lattice=int(1))

    def test_init_lattice_str(self):
        with self.assertRaises(TypeError):
            Facets(pathdir=THIS_DIR, filename=FILENAME, savedir=SAVEDIR, lattice=str(3.912))


class TestInitFacetsAttributes(unittest.TestCase):
    """
    Tests on the class Facets.

    __init__(self, filename : str,pathdir : str = "./",lattice : float = 3.912) -> None:
    """
    def setUp(self):
        # executed before each test
        self.facets = Facets(pathdir=THIS_DIR, filename=FILENAME, savedir=SAVEDIR)

    def test_init_nb_facets(self):
        self.assertTrue(self.facets.nb_facets == 22)

    def test_init_vtk_data(self):
        self.assertIsInstance(self.facets.vtk_data, dict)
        self.assertTrue(all(
            key in self.facets.vtk_data.keys() for key in
            {"x", "y", "z", "strain", "disp", "facet_probabilities", "facet_id", "x0", "y0", "z0"}))

    def test_init_strain_mean_facets(self):
        self.assertTrue(isinstance(self.facets.strain_mean_facets, list) and len(
            self.facets.strain_mean_facets) == 0)

    def test_init_disp_mean_facets(self):
        self.assertTrue(isinstance(self.facets.disp_mean_facets, list) and len(
            self.facets.disp_mean_facets) == 0)

    def test_init_field_data(self):
        self.assertIsInstance(self.facets.field_data, DataFrame)

    def test_init_parameters_u0(self):
        self.assertTrue(self.facets.u0 is None)

    def test_init_parameters_v0(self):
        self.assertTrue(self.facets.v0 is None)

    def test_init_parameters_w0(self):
        self.assertTrue(self.facets.w0 is None)

    def test_init_parameters_u(self):
        self.assertTrue(self.facets.u is None)

    def test_init_parameters_v(self):
        self.assertTrue(self.facets.v is None)

    def test_init_parameters_norm_u(self):
        self.assertTrue(self.facets.norm_u is None)

    def test_init_parameters_norm_v(self):
        self.assertTrue(self.facets.norm_v is None)

    def test_init_parameters_norm_w(self):
        self.assertTrue(self.facets.norm_w is None)

    def test_init_parameters_rotation_matrix(self):
        self.assertTrue(self.facets.rotation_matrix is None)

    def test_init_parameters_hkl_reference(self):
        self.assertTrue(self.facets.hkl_reference is None)

    def test_init_parameters_hkls(self):
        self.assertTrue(self.facets.hkls is None)

    def test_init_parameters_planar_dist(self):
        self.assertTrue(self.facets.planar_dist is None)

    def test_init_parameters_ref_normal(self):
        self.assertTrue(self.facets.ref_normal is None)

    def test_init_parameters_theoretical_angles(self):
        self.assertTrue(self.facets.theoretical_angles is None)


if __name__ == "__main__":
    run_tests(TestInitFacetsParams)
    run_tests(TestInitFacetsAttributes)
