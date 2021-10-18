# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#   (c) 06/2021-present : DESY CFEL
#       authors:
#         David Simonne, david.simonne@universite-paris-saclay.fr
#         Marie-Ingrid Richard, mrichard@esrf.fr
#         Maxime Dupraz, maxime.dupraz@esrf.fr

"""Postprocessing of the output from the facet analyzer plugin for Paraview."""

import h5py
import ipywidgets as widgets
from ipywidgets import Layout, interactive
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from typing import Tuple, Union
import vtk

from bcdi.utils import validation as valid


class Facets:
    """
    Import and stores data output of facet analyzer plugin for further analysis.

    Extract the strain component and the displacement on the facets, and retrieves the
    correct facet normals based on a user input (geometric transformation into the
    crystal frame). It requries as input a VTK file extracted from the FacetAnalyser
    plugin from ParaView. See: https://doi.org/10.1016/j.ultramic.2012.07.024

    Original tutorial on how to open vtk files:
    http://forrestbao.blogspot.com/2011/12/reading-vtk-files-in-python-via-python.html

    Expected directory structure:
     - vtk file should have been saved in in Sxxxx/postprocessing
     - the analysis output will be saved in Sxxxx/postprocessing/facet_analysis

    Several plotting options are attributes of this class, feel free to change them
    (cmap, strain_range, disp_range_avg, disp_range, strain_range_avg, comment,
    title_fontsize, axes_fontsize, legend_fontsize, ticks_fontsize)

    :param filename: str, name of the VTK file
    :param pathdir: str, path to the VTK file
    :param lattice: float, atomic spacing of the material in angstroms
     (only cubic lattices are supported).
    """

    def __init__(
        self,
        filename : str,
        pathdir : str = "./",
        lattice : float = 3.912,
    ) -> None:
        # Create other required parameters with default None value
        self.nb_facets = None
        self.vtk_data = None
        self.strain_mean_facets = None
        self.disp_mean_facets = None
        self.field_data = None
        self.u0 = None
        self.v0 = None
        self.w0 = None
        self.u = None
        self.v = None
        self.norm_u = None
        self.norm_v = None
        self.norm_w = None
        self.rotation_matrix = None
        self.hkl_reference = None
        self.hkls = None
        self.planar_dist = None
        self.ref_normal = None
        self.theoretical_angles = None

        # Check input parameters
        valid.valid_container(
            filename,
            container_types=str,
            min_length=1,
            name="filename"
        )
        valid.valid_container(
            pathdir,
            container_types=str,
            min_length=1,
            name="pathdir")
        valid.valid_item(lattice, allowed_types=float, min_excluded=0, name="lattice")
        self.pathsave = pathdir + "facets_analysis/"
        self.path_to_data = pathdir + filename
        self.filename = filename
        self.lattice = lattice

        # Plotting options
        self.strain_range = 0.001
        self.disp_range_avg = 0.2
        self.disp_range = 0.35
        self.strain_range_avg = 0.0005
        self.comment = ""

        self.title_fontsize = 24
        self.axes_fontsize = 18
        self.legend_fontsize = 11
        self.ticks_fontsize = 14

        self.cmap = "viridis"
        self.particle_cmap = "gist_ncar"

        # Load the data
        self.load_vtk()

        # Add edges and corners data if not there already
        self.save_edges_corners_data()

        # Create widget for particle viewing
        self.window = interactive(
            self.view_particle,
            elev=widgets.IntSlider(
                value=0,
                step=1,
                min=0,
                max=360,
                continuous_update=False,
                description="Elevation angle in the z plane:",
                layout=Layout(width="45%"),
                readout=True,
                style={"description_width": "initial"},
                orientation="horizontal",
            ),
            azim=widgets.IntSlider(
                value=0,
                step=1,
                min=0,
                max=360,
                continuous_update=False,
                description="Azimuth angle in the (x, y) plane:",
                layout=Layout(width="45%"),
                readout=True,
                style={"description_width": "initial"},
                orientation="horizontal",
            ),
            elev_axis=widgets.Dropdown(
                options=["x", "y", "z"],
                value="z",
                description="Elevated axis",
                continuous_update=False,
                style={"description_width": "initial"},
            ),
            facet_id_range=widgets.IntRangeSlider(
                value=[1, self.nb_facets],
                step=1,
                min=1,
                max=self.nb_facets,
                continuous_update=False,
                description="Facets ids to show:",
                layout=Layout(width="45%"),
                readout=True,
                style={"description_width": "initial"},
                orientation="horizontal",
            ),
            show_edges_corners=widgets.Checkbox(
                value=False,
                description="Show edges and corners",
                layout=Layout(width="40%"),
                style={"description_width": "initial"},
            ),
        )

    def load_vtk(self) -> None:
        """
        Load the VTK file.

        In paraview, the facets have an index that starts at 1, the index 0 corresponds
        to the edges and corners of the facets.
        """
        if not os.path.exists(self.pathsave):
            os.makedirs(self.pathsave)

        reader = vtk.vtkGenericDataObjectReader()
        reader.SetFileName(self.path_to_data)
        reader.ReadAllScalarsOn()
        reader.ReadAllVectorsOn()
        reader.ReadAllTensorsOn()
        reader.Update()
        vtkdata = reader.GetOutput()

        # Get point data
        try:
            point_data = vtkdata.GetPointData()
            print("Loading data...")
        except AttributeError:
            raise NameError("This file does not exist or is not right.")

        print("Number of points = %s" % str(vtkdata.GetNumberOfPoints()))
        print("Number of cells = %s" % str(vtkdata.GetNumberOfCells()))

        self.vtk_data = {
            "x": [vtkdata.GetPoint(i)[0] for i in range(vtkdata.GetNumberOfPoints())],
            "y": [vtkdata.GetPoint(i)[1] for i in range(vtkdata.GetNumberOfPoints())],
            "z": [vtkdata.GetPoint(i)[2] for i in range(vtkdata.GetNumberOfPoints())],
            "strain": [
                point_data.GetArray("strain").GetValue(i)
                for i in range(vtkdata.GetNumberOfPoints())
            ],
            "disp": [
                point_data.GetArray("disp").GetValue(i)
                for i in range(vtkdata.GetNumberOfPoints())
            ],
        }

        # Get cell data
        cell_data = vtkdata.GetCellData()

        self.vtk_data["facet_probabilities"] = [
            cell_data.GetArray("FacetProbabilities").GetValue(i)
            for i in range(vtkdata.GetNumberOfCells())
        ]
        self.vtk_data["facet_id"] = [
            cell_data.GetArray("FacetIds").GetValue(i)
            for i in range(vtkdata.GetNumberOfCells())
        ]
        self.vtk_data["x0"] = [
            vtkdata.GetCell(i).GetPointId(0) for i in range(vtkdata.GetNumberOfCells())
        ]
        self.vtk_data["y0"] = [
            vtkdata.GetCell(i).GetPointId(1) for i in range(vtkdata.GetNumberOfCells())
        ]
        self.vtk_data["z0"] = [
            vtkdata.GetCell(i).GetPointId(2) for i in range(vtkdata.GetNumberOfCells())
        ]

        self.nb_facets = int(max(self.vtk_data["facet_id"]))
        print("Number of facets = %s" % str(self.nb_facets))

        # Get means
        facet_indices = np.arange(1, int(self.nb_facets) + 1, 1)
        # indices from 1 to n_facets

        strain_mean = np.zeros(self.nb_facets)  # stored later in field data
        strain_std = np.zeros(self.nb_facets)  # stored later in field data
        disp_mean = np.zeros(self.nb_facets)  # stored later in field data
        disp_std = np.zeros(self.nb_facets)  # stored later in field data

        # For future analysis
        self.strain_mean_facets = []
        self.disp_mean_facets = []

        for ind in facet_indices:
            print("Facet = %d" % ind)
            results = self.extract_facet(int(ind), plot=False)
            strain_mean[ind - 1] = results["strain_mean"]
            strain_std[ind - 1] = results["strain_std"]
            disp_mean[ind - 1] = results["disp_mean"]
            disp_std[ind - 1] = results["disp_std"]

        # Get field data
        self.field_data = pd.DataFrame()
        field_data = vtkdata.GetFieldData()

        self.field_data["facet_id"] = [
            field_data.GetArray("FacetIds").GetValue(i) for i in range(self.nb_facets)
        ]
        self.field_data["strain_mean"] = strain_mean
        self.field_data["strain_std"] = strain_std
        self.field_data["disp_mean"] = disp_mean
        self.field_data["disp_std"] = disp_std
        self.field_data["n0"] = [
            field_data.GetArray("facetNormals").GetValue(3 * i)
            for i in range(self.nb_facets)
        ]
        self.field_data["n1"] = [
            field_data.GetArray("facetNormals").GetValue(3 * i + 1)
            for i in range(self.nb_facets)
        ]
        self.field_data["n2"] = [
            field_data.GetArray("facetNormals").GetValue(3 * i + 2)
            for i in range(self.nb_facets)
        ]
        self.field_data["c0"] = [
            field_data.GetArray("FacetCenters").GetValue(3 * i)
            for i in range(self.nb_facets)
        ]
        self.field_data["c1"] = [
            field_data.GetArray("FacetCenters").GetValue(3 * i + 1)
            for i in range(self.nb_facets)
        ]
        self.field_data["c2"] = [
            field_data.GetArray("FacetCenters").GetValue(3 * i + 2)
            for i in range(self.nb_facets)
        ]
        self.field_data["interplanar_angles"] = [
            field_data.GetArray("interplanarAngles").GetValue(i)
            for i in range(self.nb_facets)
        ]
        self.field_data["abs_facet_size"] = [
            field_data.GetArray("absFacetSize").GetValue(i)
            for i in range(self.nb_facets)
        ]
        self.field_data["rel_facet_size"] = [
            field_data.GetArray("relFacetSize").GetValue(i)
            for i in range(self.nb_facets)
        ]

        self.field_data = self.field_data.astype({"facet_id": np.int8})

        # Get normals
        # Don't use array index but facet number in case we sort the dataframe !!
        normals = {
            f"facet_{row.facet_id}": np.array([row["n0"], row["n1"], row["n2"]])
            for j, row in self.field_data.iterrows()
        }

        # Update legend
        legend = []
        for e in normals.keys():
            legend = legend + [" ".join(str("{:.2f}".format(e)) for e in normals[e])]
        self.field_data["legend"] = legend

    def set_rotation_matrix(
        self,
        u0 : np.ndarray,
        v0 : np.ndarray,
        w0 : np.ndarray,
        u : np.ndarray,
        v : np.ndarray,
    ) -> None :
        """
        Define the rotation matrix.

        u and v should be the vectors perpendicular to two facets. The rotation matrix
        is then used if the argument rotate_particle is set to True in the method
        load_vtk.

        :param u0: numpy.ndarray, shape (3,)
        :param v0: numpy.ndarray, shape (3,)
        :param w0: numpy.ndarray, shape (3,)
        :param u: numpy.ndarray, shape (3,)
        :param v: numpy.ndarray, shape (3,)
        """
        # Check parameters
        valid.valid_ndarray(arrays=(u0, v0, w0, u, v), shape=(3,))

        # Input theoretical values for three facets' normals
        self.u0 = u0
        self.v0 = v0
        self.w0 = w0
        print("Cross product of u0 and v0:", np.cross(self.u0, self.v0))

        # Current values for the first two facets' normals,
        # to compute the rotation matrix
        self.u = u
        self.v = v

        self.norm_u = self.u / np.linalg.norm(self.u)
        self.norm_v = self.v / np.linalg.norm(self.v)
        self.norm_w = np.cross(self.norm_u, self.norm_v)
        print("Normalized cross product of u and v:", self.norm_w)

        # Transformation matrix
        tensor0 = np.array([self.u0, self.v0, self.w0])
        tensor1 = np.array([self.norm_u, self.norm_v, self.norm_w])
        inv_tensor1 = np.linalg.inv(tensor1)
        self.rotation_matrix = np.dot(np.transpose(tensor0), np.transpose(inv_tensor1))

    def rotate_particle(self) -> None:
        """
        Rotate the nanocrystal.

        The rotation is so that the base of the normals to the facets is computed with
        the new rotation matrix.
        """
        # Get normals, again to make sure that we have the good ones
        normals = {
            f"facet_{row.facet_id}": np.array([row["n0"], row["n1"], row["n2"]])
            for j, row in self.field_data.iterrows()
            if row.facet_id != 0
        }

        try:
            for e in normals.keys():
                normals[e] = np.dot(self.rotation_matrix, normals[e])
        except AttributeError:
            print(
                """You need to define the rotation matrix first if you want to rotate
                the particle. Please choose vectors from the normals in field data"""
            )

        # Save the new normals
        for k, v in normals.items():
            # we make sure that we use the same facets !!
            mask = self.field_data["facet_id"] == int(k.split("facet_")[-1])
            self.field_data.loc[mask, "n0"] = v[0]
            self.field_data.loc[mask, "n1"] = v[1]
            self.field_data.loc[mask, "n2"] = v[2]

            # Update legend
            self.field_data.loc[mask, "legend"] = " ".join(
                ["{:.2f}".format(e) for e in v]
            )

    def fixed_reference(
        self,
        hkl_reference : Tuple[float, float, float] = (1, 1, 1),
        plot : bool = True,
    ) -> None :
        """
        Compute the interplanar angles between each normal and a fixed reference vector.

        :param hkl_reference: tuple of three real numbers, reference crystallographic
         direction
        :param plot: True to see plots
        """
        # Check parameters
        valid.valid_container(
            hkl_reference,
            container_types=(tuple, list),
            item_types=(int, float),
            length=3,
            name="hkl_reference"
        )
        valid.valid_item(plot, allowed_types=bool, name="plot")

        self.hkl_reference = hkl_reference
        self.hkls = " ".join(str(e) for e in self.hkl_reference)
        self.planar_dist = self.lattice / np.sqrt(
            self.hkl_reference[0] ** 2
            + self.hkl_reference[1] ** 2
            + self.hkl_reference[2] ** 2
        )
        self.ref_normal = self.hkl_reference / np.linalg.norm(self.hkl_reference)

        # Get normals, again to make sure that we have the good ones
        normals = {
            f"facet_{row.facet_id}": np.array([row["n0"], row["n1"], row["n2"]])
            for j, row in self.field_data.iterrows()
            if row.facet_id != 0
        }

        # Interplanar angle recomputed from a fixed reference plane,
        # between the experimental facets
        new_angles = [np.nan]
        for e in normals.keys():
            value = np.rad2deg(
                np.arccos(
                    np.dot(self.ref_normal, normals[e] / np.linalg.norm(normals[e]))
                )
            )

            new_angles.append(value)

        # Convert nan to zeros
        mask = np.isnan(new_angles)
        for j, m in enumerate(mask):
            if m:
                new_angles[j] = 0

        self.field_data["interplanar_angles"] = new_angles

        # Save angles for indexation, using facets that we should see or
        # usually see on Pt nanoparticles (WK form)
        normals = [
            [1, 0, 0],
            [-1, 0, 0],
            [1, 1, 0],
            [-1, 1, 0],
            [-1, -1, 0],
            [1, -1, 1],
            [-1, 1, -1],
            [2, 1, 0],
            [1, 1, 3],
            [1, -1, 3],
            [1, -1, -3],
            [-1, -1, 3],
            [1, 1, -3],
            [-1, -1, -3],
            [1, 1, 5],
            [1, -1, 5],
            [1, -1, -5],
            [-1, -1, 5],
            [1, 1, -5],
            [-1, -1, -5],
        ]

        # Stores the theoretical angles between normals
        self.theoretical_angles = {}
        for n in normals:
            self.theoretical_angles[str(n)] = np.rad2deg(
                np.arccos(np.dot(self.ref_normal, n / np.linalg.norm(n)))
            )

        # Make a plot
        if plot is True:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(1, 1, 1)

            ax.set_title(
                "Interplanar angles between [111] and other possible facets",
                fontsize=self.title_fontsize,
            )
            # Default value is red
            for norm, (norm_str, angle) in zip(
                normals, self.theoretical_angles.items()
            ):
                # add colors ass a fct of multiplicity
                if [abs(x) for x in norm] == [1, 1, 1]:
                    color = "#7fc97f"
                elif [abs(x) for x in norm] == [1, 1, 0]:
                    color = "#beaed4"
                elif [abs(x) for x in norm] == [1, 0, 0]:
                    color = "#fdc086"
                elif [abs(x) for x in norm] == [2, 1, 0]:
                    color = "#f0027f"
                elif [abs(x) for x in norm] == [1, 1, 3]:
                    color = "#386cb0"
                elif [abs(x) for x in norm] == [1, 1, 5]:
                    color = "k"
                else:
                    color = "r"

                ax.scatter(angle, norm_str, color=color)

            # Major ticks every 20, minor ticks every 5
            major_ticks = np.arange(0, 180, 20)
            minor_ticks = np.arange(0, 180, 5)

            ax.set_xticks(major_ticks)
            ax.set_xticks(minor_ticks, minor=True)
            # ax.set_yticks(major_ticks)
            # ax.set_yticks(minor_ticks, minor=True)

            # Or if you want different settings for the grids:
            ax.grid(which="minor", alpha=0.2)
            ax.grid(which="major", alpha=0.5)
            plt.show()

    def test_vector(self, vec : np.ndarray) -> None :
        """
        Computes value of a vector passed through the rotation matrix.

        :param vec: numpy ndarray of shape (3,).
         e.g. np.array([-0.833238, -0.418199, -0.300809])
        """
        # Check parameter
        valid.valid_ndarray(vec, shape=(3,), name="vec")

        try:
            print(np.dot(self.rotation_matrix, vec / np.linalg.norm(vec)))
        except AttributeError:
            print("You need to define the rotation matrix before")
        except TypeError:
            print("You need to define the rotation matrix before")

    def extract_facet(
            self,
            facet_id : int,
            plot : bool = False,
            elev : int = 0,
            azim : int = 0,
            output : bool = True,
            save : bool = True
    ) -> Union[None, dict] :
        """
        Extract data from one facet.

        It extracts the facet direction [x, y, z], the strain component, the
        displacement and their means, and also plots it.

        :param facet_id: id of facet in paraview
        :param plot: True to see plots:
        :param elev: elevation angle in the z plane (in degrees).
        :param azim: azimuth angle in the (x, y) plane (in degrees).
        :param output: True to return facet data
        :param save: True to save plot
        """
        # Check parameters
        valid.valid_item(facet_id, allowed_types=int, name="facet_id")
        valid.valid_item(elev, allowed_types=int, name="elev")
        valid.valid_item(azim, allowed_types=int, name="azim")
        valid.valid_item(plot, allowed_types=bool, name="plot")
        valid.valid_item(output, allowed_types=bool, name="output")
        valid.valid_item(save, allowed_types=bool, name="save")

        # Retrieve voxels that correspond to that facet index
        voxel_indices = []
        for i, _ in enumerate(self.vtk_data["facet_id"]):
            if int(self.vtk_data["facet_id"][i]) == facet_id:
                voxel_indices.append(self.vtk_data["x0"][i])
                voxel_indices.append(self.vtk_data["y0"][i])
                voxel_indices.append(self.vtk_data["z0"][i])

        #
        voxel_indices_new = list(set(voxel_indices))
        results = {
            "x": np.zeros(len(voxel_indices_new)),
            "y": np.zeros(len(voxel_indices_new)),
            "z": np.zeros(len(voxel_indices_new)),
            "strain": np.zeros(len(voxel_indices_new)),
            "disp": np.zeros(len(voxel_indices_new))
        }

        for j, _ in enumerate(voxel_indices_new):
            results["x"][j] = self.vtk_data["x"][int(voxel_indices_new[j])]
            results["y"][j] = self.vtk_data["y"][int(voxel_indices_new[j])]
            results["z"][j] = self.vtk_data["z"][int(voxel_indices_new[j])]
            results["strain"][j] = self.vtk_data["strain"][int(voxel_indices_new[j])]
            results["disp"][j] = self.vtk_data["disp"][int(voxel_indices_new[j])]
        results["strain_mean"] = np.mean(results["strain"])
        results["strain_std"] = np.std(results["strain"])
        results["disp_mean"] = np.mean(results["disp"])
        results["disp_std"] = np.std(results["disp"])

        # plot single result
        if plot:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(projection="3d")
            ax.view_init(elev=elev, azim=azim)

            ax.scatter(
                self.vtk_data["x"],
                self.vtk_data["y"],
                self.vtk_data["z"],
                s=0.2,
                antialiased=True,
                depthshade=True,
            )

            ax.scatter(
                results["x"],
                results["y"],
                results["z"],
                s=50,
                c=results["strain"],
                cmap=self.cmap,
                vmin=-0.025,
                vmax=0.025,
                antialiased=True,
                depthshade=True,
            )

            plt.tick_params(axis="both", which="major", labelsize=self.ticks_fontsize)
            plt.tick_params(axis="both", which="minor", labelsize=self.ticks_fontsize)
            plt.title(f"Strain for facet n°{facet_id}", fontsize=self.title_fontsize)
            plt.tight_layout()
            if save:
                plt.savefig(
                    f"{self.pathsave}facet_n°{facet_id}.png", bbox_inches="tight"
                )
            plt.show()
            plt.close()

            try:
                row = self.field_data.loc[self.field_data["facet_id"] == facet_id]

                n0 = row.n0.values[0]
                n1 = row.n1.values[0]
                n2 = row.n2.values[0]
                n = np.array([n0, n1, n2])
                print(f"Facet normal: {np.round(n, 2)}")
            except IndexError:
                pass  # we are on the corners and edges
            except Exception as e:
                raise e
                # pass

        if not output:
            results = None
        return results

    def view_particle(
        self,
        facet_id_range : Tuple[int, int],
        elev_axis : str,
        show_edges_corners : bool,
        elev : int = 0,
        azim : int = 0,
    ) -> None:
        """
        Visualization of the nanocrystal.

        x, y and z correspond to the frame used in paraview before saving the facet
        analyser plugin data.

        :param elev: elevation angle in the z plane (in degrees).
        :param azim: azimuth angle in the (x, y) plane (in degrees).
        :param facet_id_range: tuple of two facets numbers, facets with numbers between
         these two values will be plotted (higher boundary is excluded)
        :param elev_axis: "x", "y" or "z"
        :param show_edges_corners: set it to True to plot also edges and corners
        """
        # Check some parameters
        valid.valid_container(
            facet_id_range,
            container_types=(tuple, list),
            item_types=int,
            length=2,
            min_included=0,
            name="facet_id_range"
        )
        valid.valid_item(elev, allowed_types=int, name="elev")
        valid.valid_item(azim, allowed_types=int, name="azim")
        valid.valid_item(
            show_edges_corners,
            allowed_types=bool,
            name="show_edges_corners"
        )
        if elev_axis not in {"x", "y", "z"}:
            raise ValueError(f"unsupported value for 'elev_axis': {elev_axis}")

        plt.close()
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(projection="3d")

        ax.view_init(elev, azim)
        ax.set_xlabel("X axis", fontsize=self.axes_fontsize)
        ax.set_ylabel("Y axis", fontsize=self.axes_fontsize)
        ax.set_zlabel("Z axis", fontsize=self.axes_fontsize)

        def plot_facet_id(facet_id : int) -> None:
            """
            Plots the voxels belonging to a specific facet.

            It plots together the normal to that facet and it's id.

            :param facet_id: number of the facet
            """
            # Retrieve voxels for each facet
            voxel_indices = []
            for idx, _ in enumerate(self.vtk_data["facet_id"]):
                if int(self.vtk_data["facet_id"][idx]) == facet_id:
                    voxel_indices.append(self.vtk_data["x0"][idx])
                    voxel_indices.append(self.vtk_data["y0"][idx])
                    voxel_indices.append(self.vtk_data["z0"][idx])

            # Delete doubles
            voxel_indices_new = list(set(voxel_indices))
            results = {
                "x": np.zeros(len(voxel_indices_new)),
                "y": np.zeros(len(voxel_indices_new)),
                "z": np.zeros(len(voxel_indices_new)),
                "facet_id": np.zeros(len(voxel_indices_new)),
            }

            for idx, _ in enumerate(voxel_indices_new):
                results["x"][idx] = self.vtk_data["x"][int(voxel_indices_new[idx])]
                results["y"][idx] = self.vtk_data["y"][int(voxel_indices_new[idx])]
                results["z"][idx] = self.vtk_data["z"][int(voxel_indices_new[idx])]
                results["facet_id"][idx] = facet_id

            # Plot all the voxels with the color of their facet
            if elev_axis == "z":
                ax.scatter(
                    results["x"],
                    results["y"],
                    results["z"],
                    s=50,
                    c=results["facet_id"],
                    cmap=self.particle_cmap,
                    vmin=facet_id_range[0],
                    vmax=facet_id_range[1],
                    antialiased=True,
                    depthshade=True,
                )

            if elev_axis == "x":
                ax.scatter(
                    results["y"],
                    results["z"],
                    results["x"],
                    s=50,
                    c=results["facet_id"],
                    cmap=self.particle_cmap,
                    vmin=facet_id_range[0],
                    vmax=facet_id_range[1],
                    antialiased=True,
                    depthshade=True,
                )

            if elev_axis == "y":
                ax.scatter(
                    results["z"],
                    results["x"],
                    results["y"],
                    s=50,
                    c=results["facet_id"],
                    cmap=self.particle_cmap,
                    vmin=facet_id_range[0],
                    vmax=facet_id_range[1],
                    antialiased=True,
                    depthshade=True,
                )

            # Plot the normal to each facet at their center,
            # do it after so that is it the top layer
            row = self.field_data.loc[self.field_data["facet_id"] == facet_id]
            if facet_id != 0:
                if elev_axis == "x":
                    # Normal
                    n = np.array([row.n1.values[0], row.n2.values[0], row.n0.values[0]])

                    # Center of mass
                    com = np.array(
                        [row.c1.values[0], row.c2.values[0], row.c0.values[0]]
                    )

                elif elev_axis == "y":
                    # Normal
                    n = np.array([row.n2.values[0], row.n0.values[0], row.n1.values[0]])

                    # Center of mass
                    com = np.array(
                        [row.c2.values[0], row.c0.values[0], row.c1.values[0]]
                    )

                else:  # "z":
                    # Normal
                    n = np.array([row.n0.values[0], row.n1.values[0], row.n2.values[0]])

                    # Center of mass
                    com = np.array(
                        [row.c0.values[0], row.c1.values[0], row.c2.values[0]]
                    )

                n_str = str(facet_id) + str(n.round(2).tolist())
                ax.text(com[0], com[1], com[2], n_str, color="red", fontsize=20)

        for i in range(facet_id_range[0], facet_id_range[1]):
            plot_facet_id(i)

        if show_edges_corners:
            plot_facet_id(0)

        plt.tick_params(axis="both", which="major", labelsize=self.ticks_fontsize)
        plt.tick_params(axis="both", which="minor", labelsize=self.ticks_fontsize)
        ax.set_title("Particle voxels", fontsize=self.title_fontsize)
        plt.tight_layout()
        plt.show()

    def plot_strain(
            self,
            figsize : Tuple[float, float] = (12, 10),
            elev : int = 0,
            azim : int = 0,
            save : bool = True
    ) -> None :
        """
        Plot two views of the surface strain of the nanocrystal.

        The first one with the surface coloured by the mean strain per facet. The second
        one with the surface coloured by the strain per voxel.

        :param figsize: figure size in inches (width, height)
        :param elev: elevation angle in the z plane (in degrees).
        :param azim: azimuth angle in the (x, y) plane (in degrees).
        :param save: True to save the figures
        """
        # Check parameters
        valid.valid_container(
            figsize,
            container_types=(tuple, list),
            item_types=(int, float),
            length=2,
            min_included=0,
            name="figsize"
        )
        valid.valid_item(elev, allowed_types=int, name="elev")
        valid.valid_item(azim, allowed_types=int, name="azim")
        valid.valid_item(save, allowed_types=bool, name="save")

        # 3D strain
        p = None
        fig_name = (
            "strain_3D_" + self.hkls + self.comment + "_" + str(self.strain_range)
        )
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection="3d")

        for ind in range(1, self.nb_facets):
            results = self.extract_facet(ind, plot=False)

            p = ax.scatter(
                results["x"],
                results["y"],
                results["z"],
                s=50,
                c=results["strain"],
                cmap=self.cmap,
                vmin=-self.strain_range,
                vmax=self.strain_range,
                antialiased=True,
                depthshade=True,
            )

        fig.colorbar(p)
        ax.view_init(elev=elev, azim=azim)
        plt.title("Strain for each voxel", fontsize=self.title_fontsize)
        ax.tick_params(axis="both", which="major", labelsize=self.ticks_fontsize)
        ax.tick_params(axis="both", which="minor", labelsize=self.ticks_fontsize)

        if save:
            plt.savefig(self.pathsave + fig_name + ".png", bbox_inches="tight")
        plt.show()

        # Average strain
        fig_name = (
            "strain_3D_avg_"
            + self.hkls
            + self.comment
            + "_"
            + str(self.strain_range_avg)
        )
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection="3d")

        for ind in range(1, self.nb_facets):
            results = self.extract_facet(ind, plot=False)

            strain_mean_facet = np.zeros(results["strain"].shape)
            strain_mean_facet.fill(results["strain_mean"])
            self.strain_mean_facets = np.append(
                self.strain_mean_facets, strain_mean_facet, axis=0
            )

            p = ax.scatter(
                results["x"],
                results["y"],
                results["z"],
                s=50,
                c=strain_mean_facet,
                cmap=self.cmap,
                vmin=-self.strain_range_avg,
                vmax=self.strain_range_avg,
                antialiased=True,
                depthshade=True,
            )

        fig.colorbar(p)
        ax.view_init(elev=elev, azim=azim)
        plt.title("Mean strain per facet", fontsize=self.title_fontsize)
        ax.tick_params(axis="both", which="major", labelsize=self.ticks_fontsize)
        ax.tick_params(axis="both", which="minor", labelsize=self.ticks_fontsize)

        if save:
            plt.savefig(self.pathsave + fig_name + ".png", bbox_inches="tight")
        plt.show()

    def plot_displacement(
            self,
            figsize: Tuple[float, float] = (12, 10),
            elev : int = 0,
            azim : int = 0,
            save: bool = True
    ) -> None:
        """
        Plot two views of the surface dispalcement of the nanocrystal.

        The first one with the surface coloured by the mean displacement per facet.
        The second one with the surface coloured by the displacement per voxel.

        :param figsize: figure size in inches (width, height)
        :param elev: elevation angle in the z plane (in degrees).
        :param azim: azimuth angle in the (x, y) plane (in degrees).
        :param save: True to save the figures
        """
        # Check parameters
        valid.valid_container(
            figsize,
            container_types=(tuple, list),
            item_types=(int, float),
            length=2,
            min_included=0,
            name="figsize"
        )
        valid.valid_item(elev, allowed_types=int, name="elev")
        valid.valid_item(azim, allowed_types=int, name="azim")
        valid.valid_item(save, allowed_types=bool, name="save")

        # 3D displacement
        p = None
        fig_name = "disp_3D_" + self.hkls + self.comment + "_" + str(self.disp_range)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection="3d")

        for ind in range(1, self.nb_facets):
            results = self.extract_facet(ind, plot=False)

            p = ax.scatter(
                results["x"],
                results["y"],
                results["z"],
                s=50,
                c=results["disp"],
                cmap=self.cmap,
                vmin=-self.disp_range,
                vmax=self.disp_range,
                antialiased=True,
                depthshade=True,
            )

        fig.colorbar(p)
        ax.view_init(elev=elev, azim=azim)
        plt.title("Displacement for each voxel", fontsize=self.title_fontsize)
        ax.tick_params(axis="both", which="major", labelsize=self.ticks_fontsize)
        ax.tick_params(axis="both", which="minor", labelsize=self.ticks_fontsize)

        if save:
            plt.savefig(self.pathsave + fig_name + ".png", bbox_inches="tight")
        plt.show()

        # Average disp
        fig_name = (
            "disp_3D_avg_" + self.hkls + self.comment + "_" + str(self.disp_range_avg)
        )
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection="3d")

        for ind in range(1, self.nb_facets):
            results = self.extract_facet(ind, plot=False)

            disp_mean_facet = np.zeros(results["disp"].shape)
            disp_mean_facet.fill(results["disp_mean"])
            self.disp_mean_facets = np.append(
                self.disp_mean_facets, disp_mean_facet, axis=0
            )

            p = ax.scatter(
                results["x"],
                results["y"],
                results["z"],
                s=50,
                c=disp_mean_facet,
                cmap=self.cmap,
                vmin=-self.disp_range_avg / 2,
                vmax=self.disp_range_avg / 2,
                antialiased=True,
                depthshade=True,
            )

        fig.colorbar(p)
        ax.view_init(elev=elev, azim=azim)
        plt.title("Mean displacement per facet", fontsize=self.title_fontsize)
        ax.tick_params(axis="both", which="major", labelsize=self.ticks_fontsize)
        ax.tick_params(axis="both", which="minor", labelsize=self.ticks_fontsize)

        if save:
            plt.savefig(self.pathsave + fig_name + ".png", bbox_inches="tight")
        plt.show()

    def evolution_curves(self, ncol : int = 1) -> None:
        """
        Plot strain and displacement evolution for each facet.

        :param ncol: number of columns in the plot
        """
        # Check parameters
        valid.valid_item(ncol, allowed_types=int, min_included=1, name="ncol")

        # 1D plot: average displacement vs facet index
        fig_name = "avg_disp_vs_facet_id_" + self.hkls + self.comment
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1)

        # Major x ticks every 5, minor ticks every 1
        major_x_ticks_facet = np.arange(0, self.nb_facets + 5, 5)
        minor_x_ticks_facet = np.arange(0, self.nb_facets + 5, 1)

        ax.set_xticks(major_x_ticks_facet)
        ax.set_xticks(minor_x_ticks_facet, minor=True)
        plt.xticks(fontsize=self.ticks_fontsize)

        # Major y ticks every 0.5, minor ticks every 0.1
        major_y_ticks_facet = np.arange(-3, 3, 0.5)
        minor_y_ticks_facet = np.arange(-3, 3, 0.1)

        ax.set_yticks(major_y_ticks_facet)
        ax.set_yticks(minor_y_ticks_facet, minor=True)
        plt.yticks(fontsize=self.ticks_fontsize)

        for _, row in self.field_data.iterrows():
            ax.errorbar(
                row["facet_id"],
                row["disp_mean"],
                row["disp_std"],
                fmt="o",
                label=row["legend"],
            )

        ax.set_title(
            "Average displacement vs facet index", fontsize=self.title_fontsize
        )
        ax.set_xlabel("Facet index", fontsize=self.axes_fontsize)
        ax.set_ylabel("Average retrieved displacement", fontsize=self.axes_fontsize)

        ax.legend(
            bbox_to_anchor=(1, 1),
            loc="upper left",
            borderaxespad=0,
            fontsize=self.legend_fontsize,
            ncol=ncol,
        )

        ax.grid(which="minor", alpha=0.2)
        ax.grid(which="major", alpha=0.5)

        plt.savefig(self.pathsave + fig_name + ".png", bbox_inches="tight")
        plt.show()

        # 1D plot: average strain vs facet index
        fig_name = "avg_strain_vs_facet_id_" + self.hkls + self.comment
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1)

        # Major x ticks every 5, minor ticks every 1
        ax.set_xticks(major_x_ticks_facet)
        ax.set_xticks(minor_x_ticks_facet, minor=True)
        plt.xticks(fontsize=self.ticks_fontsize)

        # Major y ticks every 0.5, minor ticks every 0.1
        major_y_ticks_facet = np.arange(-0.0004, 0.0004, 0.0001)
        minor_y_ticks_facet = np.arange(-0.0004, 0.0004, 0.00005)

        ax.set_yticks(major_y_ticks_facet)
        ax.set_yticks(minor_y_ticks_facet, minor=True)
        plt.yticks(fontsize=self.ticks_fontsize)

        for _, row in self.field_data.iterrows():
            ax.errorbar(
                row["facet_id"],
                row["strain_mean"],
                row["strain_std"],
                fmt="o",
                label=row["legend"],
            )

        ax.set_title("Average strain vs facet index", fontsize=self.title_fontsize)
        ax.set_xlabel("Facet index", fontsize=self.axes_fontsize)
        ax.set_ylabel("Average retrieved strain", fontsize=self.axes_fontsize)

        ax.legend(
            bbox_to_anchor=(1, 1),
            loc="upper left",
            borderaxespad=0,
            fontsize=self.legend_fontsize,
            ncol=ncol,
        )

        ax.grid(which="minor", alpha=0.2)
        ax.grid(which="major", alpha=0.5)

        plt.savefig(self.pathsave + fig_name + ".png", bbox_inches="tight")
        plt.show()

        # disp, strain & size vs angle planes,
        # change line style as a fct of the planes indices
        fig_name = "disp_strain_size_vs_angle_planes_" + self.hkls + self.comment
        fig, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex="all", figsize=(10, 12))

        plt.xticks(fontsize=self.ticks_fontsize)
        plt.yticks(fontsize=self.ticks_fontsize)

        # Major ticks every 20, minor ticks every 5
        major_x_ticks = np.arange(0, 200, 20)
        minor_x_ticks = np.arange(0, 200, 5)

        ax0.set_xticks(major_x_ticks)
        ax0.set_xticks(minor_x_ticks, minor=True)

        # Major y ticks every 0.5, minor ticks every 0.1
        major_y_ticks = np.arange(-3, 3, 0.5)
        minor_y_ticks = np.arange(-3, 3, 0.1)

        ax0.set_yticks(major_y_ticks)
        ax0.set_yticks(minor_y_ticks, minor=True)

        for _, row in self.field_data.iterrows():
            try:
                lx, ly = (
                    float(row.legend.split()[0]),
                    float(row.legend.split()[1]),
                )
                if lx >= 0:
                    if ly >= 0:
                        fmt = "o"
                    else:
                        fmt = "d"
                elif ly >= 0:
                    fmt = "s"
                else:
                    fmt = "+"
            except AttributeError:
                fmt = "+"
            ax0.errorbar(
                row["interplanar_angles"],
                row["disp_mean"],
                row["disp_std"],
                fmt=fmt,
                capsize=2,
                label=row["legend"],
            )
        ax0.set_ylabel("Retrieved <disp> (A)", fontsize=self.axes_fontsize)
        ax0.legend(
            loc="upper left",
            bbox_to_anchor=(1, 1),
            ncol=1,
            fancybox=True,
            shadow=True,
            fontsize=self.legend_fontsize + 3,
        )
        ax0.grid(which="minor", alpha=0.2)
        ax0.grid(which="major", alpha=0.5)

        # Major ticks every 20, minor ticks every 5
        ax1.set_xticks(major_x_ticks)
        ax1.set_xticks(minor_x_ticks, minor=True)

        # Major y ticks every 0.5, minor ticks every 0.1
        major_y_ticks = np.arange(-0.0004, 0.0004, 0.0001)
        minor_y_ticks = np.arange(-0.0004, 0.0004, 0.00005)

        ax1.set_yticks(major_y_ticks)
        ax1.set_yticks(minor_y_ticks, minor=True)

        for _, row in self.field_data.iterrows():
            try:
                lx, ly = (
                    float(row.legend.split()[0]),
                    float(row.legend.split()[1]),
                )
                if lx >= 0:
                    if ly >= 0:
                        fmt = "o"
                    else:
                        fmt = "d"
                elif ly >= 0:
                    fmt = "s"
                else:
                    fmt = "+"
            except AttributeError:
                fmt = "+"
            ax1.errorbar(
                row["interplanar_angles"],
                row["strain_mean"],
                row["strain_std"],
                fmt=fmt,
                capsize=2,
                label=row["legend"],
            )
        ax1.set_ylabel("Retrieved <strain>", fontsize=self.axes_fontsize)
        ax1.grid(which="minor", alpha=0.2)
        ax1.grid(which="major", alpha=0.5)

        # Major ticks every 20, minor ticks every 5
        ax2.set_xticks(major_x_ticks)
        ax2.set_xticks(minor_x_ticks, minor=True)

        # Major y ticks every 0.5, minor ticks every 0.1
        major_y_ticks = np.arange(-0, 0.3, 0.05)
        minor_y_ticks = np.arange(-0, 0.3, 0.01)

        ax2.set_yticks(major_y_ticks)
        ax2.set_yticks(minor_y_ticks, minor=True)

        for _, row in self.field_data.iterrows():
            ax2.plot(
                row["interplanar_angles"],
                row["rel_facet_size"],
                "o",
                label=row["legend"],
            )
        ax2.set_xlabel("Angle (deg.)", fontsize=self.axes_fontsize)
        ax2.set_ylabel("Relative facet size", fontsize=self.axes_fontsize)
        ax2.grid(which="minor", alpha=0.2)
        ax2.grid(which="major", alpha=0.5)

        plt.savefig(self.pathsave + fig_name + ".png", bbox_inches="tight")
        plt.show()

    def save_edges_corners_data(self) -> None:
        """Extract the edges and corners data, i.e. the mean strain and displacement."""
        if 0 not in self.field_data.facet_id.values:
            result = self.extract_facet(0)

            edges_cornes_df = pd.DataFrame(
                {
                    "facet_id": [0],
                    "strain_mean": result["strain_mean"],
                    "strain_std": result["strain_std"],
                    "disp_mean": result["disp_mean"],
                    "disp_std": result["disp_std"],
                    "n0": None,
                    "n1": None,
                    "n2": None,
                    "c0": None,
                    "c1": None,
                    "c2": None,
                    "interplanar_angles": None,
                    "abs_facet_size": None,
                    "rel_facet_size": None,
                    "legend": None,
                }
            )

            self.field_data = self.field_data.append(edges_cornes_df, ignore_index=True)
            self.field_data = self.field_data.sort_values(by="facet_id")
            self.field_data = self.field_data.reset_index(drop=True)

    def save_data(self, path_to_data : str) -> None :
        """
        Save the field data as a csv file.

        :param path_to_data: path where to save the data
        """
        # Check parameters
        valid.valid_item(path_to_data, allowed_types=str, name="path_to_data")

        # Save field data
        self.field_data.to_csv(path_to_data, index=False)

    def to_hdf5(self, path_to_data : str) -> None :
        """
        Save the facets object as an hdf5 file.

        Can be combined with the file saved by the gwaihir gui.

        :param path_to_data: path where to save the data
        """
        # Check parameters
        valid.valid_item(path_to_data, allowed_types=str, name="path_to_data")

        # Save attributes
        with h5py.File(path_to_data, mode="a") as f:
            try:
                facets = f.create_group("/data/facets")

                facets.create_dataset("path_to_data", data=self.path_to_data)
                facets.create_dataset("nb_facets", data=self.nb_facets)
                facets.create_dataset("comment", data=self.comment)
                facets.create_dataset("lattice", data=self.lattice)
                facets.create_dataset("u0", data=self.u0)
                facets.create_dataset("v0", data=self.v0)
                facets.create_dataset("w0", data=self.w0)
                facets.create_dataset("u", data=self.u)
                facets.create_dataset("v", data=self.v)
                facets.create_dataset("norm_u", data=self.norm_u)
                facets.create_dataset("norm_v", data=self.norm_v)
                facets.create_dataset("norm_w", data=self.norm_w)
                facets.create_dataset("rotation_matrix", data=self.rotation_matrix)
                facets.create_dataset("hkl_reference", data=self.hkl_reference)
                facets.create_dataset("planar_dist", data=self.planar_dist)
                facets.create_dataset("ref_normal", data=self.ref_normal)

            except ValueError:
                print("Data already exists, overwriting ...")

                f["/data/facets/path_to_data"][...] = self.path_to_data
                f["/data/facets/nb_facets"][...] = self.nb_facets
                f["/data/facets/comment"][...] = self.comment
                f["/data/facets/lattice"][...] = self.lattice
                f["/data/facets/u0"][...] = self.u0
                f["/data/facets/v0"][...] = self.v0
                f["/data/facets/w0"][...] = self.w0
                f["/data/facets/u"][...] = self.u
                f["/data/facets/v"][...] = self.v
                f["/data/facets/norm_u"][...] = self.norm_u
                f["/data/facets/norm_v"][...] = self.norm_v
                f["/data/facets/norm_w"][...] = self.norm_w
                f["/data/facets/rotation_matrix"][...] = self.rotation_matrix
                f["/data/facets/hkl_reference"][...] = self.hkl_reference
                f["/data/facets/planar_dist"][...] = self.planar_dist
                f["/data/facets/ref_normal"][...] = self.ref_normal

            except AttributeError:
                print("Particle not rotated, some attributes could not be saved ...")
                facets.create_dataset("u0", data=np.zeros(3))
                facets.create_dataset("v0", data=np.zeros(3))
                facets.create_dataset("w0", data=np.zeros(3))
                facets.create_dataset("u", data=np.zeros(3))
                facets.create_dataset("v", data=np.zeros(3))
                facets.create_dataset("norm_u", data=np.zeros(3))
                facets.create_dataset("norm_v", data=np.zeros(3))
                facets.create_dataset("norm_w", data=np.zeros(3))
                facets.create_dataset("rotation_matrix", data=np.zeros((3, 3)))
                facets.create_dataset("hkl_reference", data=[])
                facets.create_dataset("planar_dist", data="")
                facets.create_dataset("ref_normal", data=np.zeros(3))

        # Save field data
        try:
            self.field_data.to_hdf(
                path_to_data,
                key="data/facets/tables/field_data",
                mode="a",
                append=True,
                format="table",
                data_columns=True,
            )
        except Exception as e:
            raise e

        # Save theoretical angles
        try:
            df = pd.DataFrame(
                {
                    "miller_indices": list(self.theoretical_angles.keys()),
                    "interplanar_angles": list(self.theoretical_angles.values()),
                }
            )
            df.to_hdf(
                path_to_data,
                key="data/facets/tables/theoretical_angles",
                mode="a",
                append=True,
                format="table",
                data_columns=True,
            )
        except AttributeError:
            print("Facets has no attribute theoretical_angles yet")
        except Exception as e:
            raise e

    def __repr__(self):
        """Unambiguous representation of the class."""
        return "Facets {}\n".format(
            self.filename,
        )

    def __str__(self):
        """Readable representation of the class."""
        return repr(self)
