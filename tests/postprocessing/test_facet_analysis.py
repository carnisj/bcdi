# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import unittest
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame

from bcdi.postprocessing.facet_analysis import Facets
from tests.config import run_tests

here = Path(__file__).parent
THIS_DIR = str(here)
SAVEDIR = str(here.parents[1] / "test_output/")
FILENAME = "3572_fa.vtk"
matplotlib.use("Agg")


class TestInitFacetsParams(unittest.TestCase):
    """
    Tests on the class Facets.

    __init__(self, filename : str,pathdir : str = "./",lattice : float = 3.912) -> None:
    """

    def setUp(self):
        # executed before each test
        plt.ion()

    def tearDown(self) -> None:
        plt.close("all")

    def test_init_pathdir_empty(self):
        with self.assertRaises(ValueError):
            Facets(pathdir="", filename=FILENAME, savedir=SAVEDIR)

    def test_init_pathdir_none(self):
        with self.assertRaises(ValueError):
            Facets(pathdir=None, filename=FILENAME, savedir=SAVEDIR)

    def test_init_pathdir_wrong_type(self):
        with self.assertRaises(TypeError):
            Facets(pathdir=0, filename=FILENAME, savedir=SAVEDIR)

    def test_init_savedir_empty(self):
        with self.assertRaises(ValueError):
            Facets(pathdir=THIS_DIR, filename=FILENAME, savedir="")

    def test_init_savedir_none(self):
        facets = Facets(pathdir=THIS_DIR, filename=FILENAME, savedir=None)
        self.assertTrue(facets.pathsave == THIS_DIR + "/facets_analysis/")

    def test_init_savedir_wrong_type(self):
        with self.assertRaises(TypeError):
            Facets(pathdir=THIS_DIR, filename=FILENAME, savedir=0)

    def test_init_filename_empty(self):
        with self.assertRaises(ValueError):
            Facets(pathdir=THIS_DIR, filename="", savedir=SAVEDIR)

    def test_init_filename_none(self):
        with self.assertRaises(ValueError):
            Facets(pathdir=THIS_DIR, filename=None, savedir=SAVEDIR)

    def test_init_filename_wrong_type(self):
        with self.assertRaises(TypeError):
            Facets(pathdir=THIS_DIR, filename=0, savedir=SAVEDIR)

    def test_init_lattice_none(self):
        with self.assertRaises(ValueError):
            Facets(pathdir=THIS_DIR, filename=FILENAME, savedir=SAVEDIR, lattice=None)

    def test_init_lattice_int(self):
        with self.assertRaises(TypeError):
            Facets(pathdir=THIS_DIR, filename=FILENAME, savedir=SAVEDIR, lattice=1)

    def test_init_lattice_str(self):
        with self.assertRaises(TypeError):
            Facets(
                pathdir=THIS_DIR, filename=FILENAME, savedir=SAVEDIR, lattice=str(3.912)
            )


class TestInitFacetsAttributes(unittest.TestCase):
    """
    Tests on the class Facets.

    __init__(self, filename : str,pathdir : str = "./",lattice : float = 3.912) -> None:
    """

    def setUp(self):
        # executed before each test
        plt.ion()
        self.facets = Facets(pathdir=THIS_DIR, filename=FILENAME, savedir=SAVEDIR)

    def tearDown(self) -> None:
        plt.close("all")

    def test_init_nb_facets(self):
        self.assertTrue(self.facets.nb_facets == 22)

    def test_init_vtk_data(self):
        self.assertIsInstance(self.facets.vtk_data, dict)
        self.assertTrue(
            all(
                key in self.facets.vtk_data
                for key in {
                    "x",
                    "y",
                    "z",
                    "strain",
                    "disp",
                    "facet_probabilities",
                    "facet_id",
                    "x0",
                    "y0",
                    "z0",
                }
            )
        )

    def test_init_strain_mean_facets(self):
        self.assertTrue(
            isinstance(self.facets.strain_mean_facets, list)
            and len(self.facets.strain_mean_facets) == 0
        )

    def test_init_disp_mean_facets(self):
        self.assertTrue(
            isinstance(self.facets.disp_mean_facets, list)
            and len(self.facets.disp_mean_facets) == 0
        )

    def test_init_field_data(self):
        self.assertIsInstance(self.facets.field_data, DataFrame)

    def test_init_u0(self):
        self.assertTrue(
            isinstance(self.facets.u0, np.ndarray) and self.facets.u0.shape == (3,)
        )

    def test_init_v0(self):
        self.assertTrue(
            isinstance(self.facets.v0, np.ndarray) and self.facets.v0.shape == (3,)
        )

    def test_init_w0(self):
        self.assertTrue(
            isinstance(self.facets.w0, np.ndarray) and self.facets.w0.shape == (3,)
        )

    def test_init_u(self):
        self.assertTrue(
            isinstance(self.facets.u, np.ndarray) and self.facets.u.shape == (3,)
        )

    def test_init_v(self):
        self.assertTrue(
            isinstance(self.facets.v, np.ndarray) and self.facets.v.shape == (3,)
        )

    def test_init_norm_u(self):
        self.assertTrue(
            isinstance(self.facets.norm_u, np.ndarray)
            and self.facets.norm_u.shape == (3,)
        )

    def test_init_norm_v(self):
        self.assertTrue(
            isinstance(self.facets.norm_v, np.ndarray)
            and self.facets.norm_v.shape == (3,)
        )

    def test_init_norm_w(self):
        self.assertTrue(
            isinstance(self.facets.norm_w, np.ndarray)
            and self.facets.norm_w.shape == (3,)
        )

    def test_init_rotation_matrix(self):
        self.assertTrue(self.facets.rotation_matrix is None)

    def test_init_hkl_reference(self):
        self.assertTrue(self.facets.hkl_reference is None)

    def test_init_hkls(self):
        self.assertTrue(self.facets.hkls == "")

    def test_init_planar_dist(self):
        self.assertTrue(self.facets.planar_dist is None)

    def test_init_ref_normal(self):
        self.assertTrue(self.facets.ref_normal is None)

    def test_init_theoretical_angles(self):
        self.assertTrue(self.facets.theoretical_angles is None)

    def test_init_strain_range(self):
        self.assertTrue(np.isclose(self.facets.strain_range, 0.001))

    def test_init_disp_range_avg(self):
        self.assertTrue(np.isclose(self.facets.disp_range_avg, 0.2))

    def test_init_disp_range(self):
        self.assertTrue(np.isclose(self.facets.disp_range, 0.35))

    def test_init_strain_range_avg(self):
        self.assertTrue(np.isclose(self.facets.strain_range_avg, 0.0005))

    def test_init_comment(self):
        self.assertTrue(self.facets.comment == "")

    def test_init_title_fontsize(self):
        self.assertTrue(self.facets.title_fontsize == 24)

    def test_init_axes_fontsize(self):
        self.assertTrue(self.facets.axes_fontsize == 18)

    def test_init_legend_fontsize(self):
        self.assertTrue(self.facets.legend_fontsize == 11)

    def test_init_ticks_fontsize(self):
        self.assertTrue(self.facets.ticks_fontsize == 14)

    def test_init_cmap(self):
        self.assertTrue(self.facets.cmap == "viridis")

    def test_init_particle_cmap(self):
        self.assertTrue(self.facets.particle_cmap == "gist_ncar")

    def test_repr(self):
        self.assertIsInstance(eval(repr(self.facets)), Facets)


if __name__ == "__main__":
    run_tests(TestInitFacetsParams)
    run_tests(TestInitFacetsAttributes)
