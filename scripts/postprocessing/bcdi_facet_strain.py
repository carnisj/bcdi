#!/usr/bin/env python3

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import collections
import gc
import logging
import os
import pathlib
import sys
import tkinter as tk
from tkinter import filedialog

import numpy as np
import vtk
from matplotlib import pyplot as plt
from scipy.ndimage.measurements import center_of_mass
from skimage import measure
from vtk.util import numpy_support

import bcdi.graph.graph_utils as gu
import bcdi.postprocessing.facet_recognition as fu
import bcdi.postprocessing.postprocessing_utils as pu
import bcdi.simulation.simulation_utils as simu
from bcdi.graph.colormap import ColormapFactory

helptext = """
Script for detecting facets on a 3D crytal reconstructed by a phasing algorithm
(Bragg CDI) and making some statistics about strain by facet. The correct threshold
for support determination should be given as input, as well as voxel sizes for a
correct calculation of facet angle.

Input: a reconstruction .npz file with fields: 'amp' and 'strain'
Output: a log file with strain statistics by plane, a VTK file for 3D visualization of
detected planes.
"""

scan = 11  # spec scan number
datadir = "C:/Users/Jerome/Documents/data/dataset_facet_recognition/"
support_threshold = 0.48  # threshold for support determination
voxel_size = [3.63, 5.31, 2.62]
# tuple of 3 numbers, voxel size of the real-space reconstruction in each dimension
upsampling_factor = 2  # integer, factor for upsampling the reconstruction
# in order to have a smoother surface
savedir = datadir + "/test/"
reflection = np.array([1, 1, 1])  # measured crystallographic reflection
projection_axis = 2  # the projection will be performed on the equatorial plane
# perpendicular to that axis (0, 1 or 2)
debug = False  # set to True to see all plots for debugging
smoothing_iterations = 5  # number of iterations in Taubin smoothing,
# bugs if smoothing_iterations larger than 10
smooth_lamda = 0.33  # lambda parameter in Taubin smoothing
smooth_mu = 0.34  # mu parameter in Taubin smoothing
radius_normals = 0.1
# radius of integration for the calculation of the density of normals
projection_method = "stereographic"  # 'stereographic' or 'equirectangular'
peak_min_distance = 10  # pixel separation between peaks in corner_peaks()
max_distance_plane = 0.75
# in pixels, maximum allowed distance to the facet plane of a voxel
edges_coord = 350
# coordination threshold for isolating edges, 360 seems to work reasonably well
corners_coord = 300  # coordination threshold for isolating corners,
# 310 seems to work reasonably well
########################################################
# parameters only used in the stereographic projection #
########################################################
threshold_south = -1400  # background threshold in the stereographic projection
# from South of the density of normals
threshold_north = -1200
# background threshold in the stereographic projection from North
# of the density of normals
max_angle = 95  # maximum angle in degree of the stereographic projection
# (should be larger than 90)
stereo_scale = "linear"
# 'linear' or 'log', scale of the colorbar in the stereographic plot
##########################################################
# parameters only used in the equirectangular projection #
##########################################################
bw_method = 0.03  # bandwidth in the gaussian kernel density estimation
kde_threshold = -0.2
# threshold for defining the background in the density estimation of normals
##################################################
# define crystallographic planes of interest for #
# the stereographic projection (cubic lattice)   #
##################################################
planes_south = {}  # create dictionnary for the projection from the South pole,
# the reference is +reflection
# planes_south['0 2 0'] =
# simu.angle_vectors(ref_vector=reflection, test_vector=np.array([0, 2, 0]))
planes_south["1 1 1"] = simu.angle_vectors(
    ref_vector=reflection, test_vector=np.array([1, 1, 1])
)
planes_south["1 0 0"] = simu.angle_vectors(
    ref_vector=reflection, test_vector=np.array([1, 0, 0])
)
planes_south["1 1 0"] = simu.angle_vectors(
    ref_vector=reflection, test_vector=np.array([1, 1, 0])
)
planes_south["-1 1 0"] = simu.angle_vectors(
    ref_vector=reflection, test_vector=np.array([-1, 1, 0])
)
planes_south["1 -1 1"] = simu.angle_vectors(
    ref_vector=reflection, test_vector=np.array([1, -1, 1])
)
# planes_south['-1 -1 1'] =
# simu.angle_vectors(ref_vector=reflection, test_vector=np.array([-1, -1, 1]))
# planes_south['2 1 0'] =
# simu.angle_vectors(ref_vector=reflection, test_vector=np.array([2, 1, 0]))
# planes_south['2 -1 0'] =
# simu.angle_vectors(ref_vector=reflection, test_vector=np.array([2, -1, 0]))
# planes_south['1 2 0'] =
# simu.angle_vectors(ref_vector=reflection, test_vector=np.array([1, 2, 0]))

planes_north = {}  # create dictionnary for the projection from the North pole,
# the reference is -reflection
# planes_north['0 -2 0'] =
# simu.angle_vectors(ref_vector=-reflection, test_vector=np.array([0, -2, 0]))
planes_north["-1 -1 -1"] = simu.angle_vectors(
    ref_vector=-reflection, test_vector=np.array([-1, -1, -1])
)
planes_north["-1 0 0"] = simu.angle_vectors(
    ref_vector=-reflection, test_vector=np.array([-1, 0, 0])
)
planes_north["-1 -1 0"] = simu.angle_vectors(
    ref_vector=-reflection, test_vector=np.array([-1, -1, 0])
)
planes_north["-1 1 0"] = simu.angle_vectors(
    ref_vector=-reflection, test_vector=np.array([-1, 1, 0])
)
planes_north["-1 -1 1"] = simu.angle_vectors(
    ref_vector=-reflection, test_vector=np.array([-1, -1, 1])
)
# planes_north['-1 1 1'] =
# simu.angle_vectors(ref_vector=-reflection, test_vector=np.array([-1, 1, 1]))
planes_north["-2 1 0"] = simu.angle_vectors(
    ref_vector=-reflection, test_vector=np.array([-2, 1, 0])
)
planes_north["-2 -1 0"] = simu.angle_vectors(
    ref_vector=-reflection, test_vector=np.array([-2, -1, 0])
)
# planes_north['1 -2 0'] =
# simu.angle_vectors(ref_vector=-reflection, test_vector=np.array([1, -2, 0]))
##########################
# end of user parameters #
##########################

###########################################################
# create directory and initialize error logger #
###########################################################
pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)
logger = logging.getLogger()

###################
# define colormap #
###################
my_cmap = ColormapFactory().cmap

#############
# load data #
#############
plt.ion()
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(initialdir=datadir, filetypes=[("NPZ", "*.npz")])
npzfile = np.load(file_path)
amp = npzfile["amp"]
amp = amp / amp.max()
nz, ny, nx = amp.shape
print("Initial data size: (", nz, ",", ny, ",", nx, ")")
strain = npzfile["strain"]

#################
# upsample data #
#################
if upsampling_factor > 1:
    amp, _ = fu.upsample(
        array=amp,
        upsampling_factor=upsampling_factor,
        voxelsizes=voxel_size,
        title="modulus",
        debugging=debug,
    )
    strain, voxel_size = fu.upsample(
        array=strain,
        upsampling_factor=upsampling_factor,
        voxelsizes=voxel_size,
        title="strain",
        debugging=debug,
    )
    nz, ny, nx = amp.shape
    print("Upsampled data size: (", nz, ",", ny, ",", nx, ")")
    print("New voxel sizes: ", voxel_size)

#####################################################################
# Use marching cubes to obtain the surface mesh of these ellipsoids #
#####################################################################
vertices_old, faces, _, _ = measure.marching_cubes(
    amp, level=support_threshold, allow_degenerate=False, step_size=1
)
# vertices_old is a list of 3d coordinates of all vertices points
# faces is a list of facets defined by the indices of 3 vertices_old

# from scipy.io import savemat
# savemat('//win.desy.de/home/carnisj/My Documents/MATLAB/TAUBIN/vertices.mat',
# {'V': vertices_old})
# savemat('//win.desy.de/home/carnisj/My Documents/MATLAB/TAUBIN/faces.mat',
# {'F': faces})

# Display mesh before smoothing
if debug:
    gu.plot_3dmesh(vertices_old, faces, (nz, ny, nx), title="Mesh after marching cubes")
    plt.ion()

#######################################
# smooth the mesh using taubin_smooth #
#######################################
vertices_new, normals, _, intensity, faces, _ = fu.taubin_smooth(
    faces,
    vertices_old,
    iterations=smoothing_iterations,
    lamda=smooth_lamda,
    mu=smooth_mu,
    radius=radius_normals,
    debugging=debug,
)
del vertices_old
gc.collect()

nb_vertices = vertices_new.shape[0]

# Display smoothed triangular mesh
if debug:
    gu.plot_3dmesh(
        vertices_new, faces, (nz, ny, nx), title="Mesh after Taubin smoothing"
    )
    plt.ion()

#####################################################################
# 2D projection of normals, peak finding and watershed segmentation #
#####################################################################
nb_normals = normals.shape[0]
if projection_method == "stereographic":
    labels_top, labels_bottom, stereo_proj, remove_row = fu.stereographic_proj(
        normals=normals,
        intensity=intensity,
        background_south=threshold_south,
        background_north=threshold_north,
        min_distance=peak_min_distance,
        savedir=savedir,
        save_txt=False,
        plot_planes=True,
        planes_south=planes_south,
        planes_north=planes_north,
        max_angle=max_angle,
        voxel_size=voxel_size,
        projection_axis=projection_axis,
        scale=stereo_scale,
        debugging=debug,
    )
    # labels_south and labels_north are 2D arrays for projections from South and North
    # stereo_proj is a (Nx4) array containint the projected coordinates of normals
    # from South (u column 0, v column 1) and North (u column2 , v column 3). The
    # coordinates are in degrees, not indices.

    # remove rows containing nan values
    normals = np.delete(normals, remove_row, axis=0)
    faces = np.delete(faces, remove_row, axis=0)

    nb_normals = normals.shape[0]
    numy, numx = labels_top.shape  # identical to labels_bottom.shape
    max_label = max(labels_top.max(), labels_bottom.max())

    if stereo_proj.shape[0] != nb_normals:
        print(projection_method, "projection output: incompatible number of normals")
        sys.exit()

    # look for potentially duplicated labels (labels crossing the 90 degree circle)
    duplicated_labels = [
        0
    ]  # do not consider background points when looking for duplicates
    # (label 0 is the background)
    # duplicated_labels stores bottom_labels which are duplicate from top_labels
    # [0 duplicated_labels unique_label ...]
    for label in range(1, labels_top.max() + 1, 1):
        label_points = np.argwhere(labels_top == label)
        # rescale label_points to angles instead of indices,
        # the angular range is [-max_angle max_angle]
        label_points[:, 0] = (label_points[:, 0] * 2 * max_angle / numy) - max_angle
        label_points[:, 1] = (label_points[:, 1] * 2 * max_angle / numx) - max_angle

        label_distances = np.sqrt(
            label_points[:, 0] ** 2 + label_points[:, 1] ** 2
        )  # distance in angle from the origin
        if (label_distances <= 90).sum() == label_points.shape[
            0
        ]:  # all points inside the 90deg border
            continue  # do nothing, the facet is valid
        if (label_distances > 90).sum() == label_points.shape[
            0
        ]:  # all points outside the 90deg border
            continue  # do nothing, the facet will be filtered out
            # in next section by distance check
        print("Label ", str(label), "is potentially duplicated")
        # look for the corresponding label in the bottom projection
        for idx in range(nb_normals):
            # calculate the corresponding index coordinates
            # by rescaling from [-max_angle max_angle] to [0 numy] or [0 numx]
            u_top = int(
                np.rint((stereo_proj[idx, 0] + max_angle) * numx / (2 * max_angle))
            )  # u axis horizontal
            v_top = int(
                np.rint((stereo_proj[idx, 1] + max_angle) * numy / (2 * max_angle))
            )  # v axis vertical
            u_bottom = int(
                np.rint((stereo_proj[idx, 2] + max_angle) * numx / (2 * max_angle))
            )  # u axis horizontal
            v_bottom = int(
                np.rint((stereo_proj[idx, 3] + max_angle) * numy / (2 * max_angle))
            )  # v axis vertical

            try:
                if (
                    labels_top[v_top, u_top] == label
                    and labels_bottom[v_bottom, u_bottom] not in duplicated_labels
                ):
                    # only the first duplicated point will be checked,
                    # then the whole bottom_label is changed
                    # to label and there is no need to check anymore
                    duplicated_labels.append(labels_bottom[v_bottom, u_bottom])
                    duplicated_labels.append(label)
                    print(
                        "  Corresponding label :",
                        labels_bottom[v_bottom, u_bottom],
                        "changed to",
                        label,
                    )
                    labels_bottom[
                        labels_bottom == labels_bottom[v_bottom, u_bottom]
                    ] = label
            except IndexError:
                # the IndexError exception arises because we are spanning all normals
                # for labels_top, even those whose stereographic projection is
                # farther than max_angle.
                continue

    del label_points, label_distances
    gc.collect()

    # reorganize stereo_proj to keep only the projected point
    # which is in the angular range [-90 90]
    # stereo_proj coordinates are in polar degrees, we want coordinates to be in indices
    coordinates = np.zeros((nb_normals, 3), dtype=stereo_proj.dtype)
    # 1st and 2nd columns are coordinates
    # the 3rd column is a flag for using the South (0)
    # or North (1) projected coordinates
    for idx in range(nb_normals):
        if np.sqrt(stereo_proj[idx, 0] ** 2 + stereo_proj[idx, 1] ** 2) > 90:
            coordinates[idx, 0] = stereo_proj[
                idx, 3
            ]  # use v values for the projection from North pole
            coordinates[idx, 1] = stereo_proj[
                idx, 2
            ]  # use u values for the projection from North pole
            coordinates[idx, 2] = (
                1  # use values from labels_bottom (projection from North pole)
            )
        else:
            coordinates[idx, 0] = stereo_proj[
                idx, 1
            ]  # use v values for the projection from South pole
            coordinates[idx, 1] = stereo_proj[
                idx, 0
            ]  # use u values for the projection from South pole
            coordinates[idx, 2] = (
                0  # use values from labels_top (projection from South pole)
            )
    del stereo_proj
    gc.collect()

    # rescale euclidian v axis from [-max_angle max_angle] to [0 numy]
    coordinates[:, 0] = (coordinates[:, 0] + max_angle) * numy / (2 * max_angle)
    # rescale euclidian u axis from [-max_angle max_angle] to [0 numx]
    coordinates[:, 1] = (coordinates[:, 1] + max_angle) * numx / (2 * max_angle)
    # change coordinates to an array of integer indices
    coordinates = coordinates.astype(int)

    ##########################################################
    # now that we have the labels and coordinates in indices #
    # we can assign back labels to normals and vertices      #
    ##########################################################
    normals_label = np.zeros(nb_normals, dtype=int)
    vertices_label = np.zeros(
        nb_vertices, dtype=int
    )  # the number of vertices is: vertices_new.shape[0]
    for idx in range(nb_normals):
        # check to which label belongs this normal
        if (
            coordinates[idx, 2] == 0
        ):  # use values from labels_top (projection from South pole)
            label_idx = labels_top[coordinates[idx, 0], coordinates[idx, 1]]
        elif (
            coordinates[idx, 2] == 1
        ):  # use values from labels_bottom (projection from North pole)
            label_idx = labels_bottom[coordinates[idx, 0], coordinates[idx, 1]]
        else:
            label_idx = 0  # duplicated facet, set it to the background
        normals_label[idx] = label_idx  # attribute the label to the normal
        vertices_label[faces[idx, :]] = (
            label_idx  # attribute the label to the corresponding vertices
        )
    del labels_top, labels_bottom
elif projection_method == "equirectangular":
    labels, longitude_latitude = fu.equirectangular_proj(
        normals=normals,
        intensity=intensity,
        bw_method=bw_method,
        background_threshold=kde_threshold,
        min_distance=peak_min_distance,
        debugging=debug,
    )
    if longitude_latitude.shape[0] != nb_normals:
        print(projection_method, "projection output: incompatible number of normals")
        sys.exit()
    numy, numx = labels.shape
    # rescale the horizontal axis from [-pi pi] to [0 numx]
    longitude_latitude[:, 0] = (
        (longitude_latitude[:, 0] + np.pi) * numx / (2 * np.pi)
    )  # longitude
    # rescale the vertical axis from [-pi/2 pi/2] to [0 numy]
    longitude_latitude[:, 1] = (
        (longitude_latitude[:, 1] + np.pi / 2) * numy / np.pi
    )  # latitude
    # change longitude_latitude to an array of integer indices
    coordinates = np.rint(np.fliplr(longitude_latitude)).astype(
        int
    )  # put the vertical axis in first position
    duplicated_labels = []
    max_label = labels.max()

    del longitude_latitude
    gc.collect()

    ##############################################
    # assign back labels to normals and vertices #
    ##############################################
    normals_label = np.zeros(nb_normals, dtype=int)
    vertices_label = np.zeros(
        nb_vertices, dtype=int
    )  # the number of vertices is: vertices_new.shape[0]
    for idx in range(nb_normals):
        label_idx = labels[coordinates[idx, 0], coordinates[idx, 1]]
        normals_label[idx] = label_idx  # attribute the label to the normal
        vertices_label[faces[idx, :]] = (
            label_idx  # attribute the label to the corresponding vertices
        )

else:
    print("Invalid value for projection_method")
    sys.exit()

unique_labels = [
    label
    for label in np.arange(1, max_label + 1)
    if label not in duplicated_labels[1::2]
]  # label 0 is the background
if len(duplicated_labels[1::2]) == 0:
    print("\nNo duplicated label")
print("\nBackground: ", str((normals_label == 0).sum()), "normals")
for label in unique_labels:
    print(
        "Facet",
        str(label),
        ": ",
        str((normals_label == label).sum()),
        "normals detected",
    )
del normals, normals_label, coordinates, faces, duplicated_labels, intensity
gc.collect()

###############################################
# assign back labels to voxels using vertices #
###############################################
all_planes = np.zeros((nz, ny, nx), dtype=int)
planes_counter = np.zeros(
    (nz, ny, nx), dtype=int
)  # check if a voxel is used several times
duplicated_counter = 0
for idx in range(nb_vertices):
    temp_indices = np.rint(vertices_new[idx, :]).astype(int)
    planes_counter[temp_indices[0], temp_indices[1], temp_indices[2]] = (
        planes_counter[temp_indices[0], temp_indices[1], temp_indices[2]] + 1
    )
    # check duplicated voxels and discard them if they belong to different planes
    # it happens when some vertices are close and they give
    # the same voxel after rounding their position to integers
    # one side effect is that the border of areas obtained by
    # watershed segmentation will be set to the background
    if (
        planes_counter[temp_indices[0], temp_indices[1], temp_indices[2]] > 1
    ):  # a rounded voxel was already added
        if (
            all_planes[temp_indices[0], temp_indices[1], temp_indices[2]]
            != vertices_label[idx]
        ):
            # belongs to different labels, therefore it is set as background (label 0)
            all_planes[temp_indices[0], temp_indices[1], temp_indices[2]] = 0
            duplicated_counter = duplicated_counter + 1
    else:  # non duplicated pixel
        all_planes[temp_indices[0], temp_indices[1], temp_indices[2]] = vertices_label[
            idx
        ]
print("\nRounded vertices belonging to multiple labels = ", duplicated_counter, "\n")
del planes_counter, vertices_label, vertices_new
gc.collect()

for label in unique_labels:
    print(
        "Facet", str(label), ": ", str((all_planes == label).sum()), "voxels detected"
    )

############################################
# define the support, surface layer & bulk #
############################################
support = np.zeros(amp.shape)
support[abs(amp) > support_threshold * abs(amp).max()] = 1
zcom_support, ycom_support, xcom_support = center_of_mass(support)
print(
    "\nCOM at (z, y, x): (",
    str(f"{zcom_support:.2f}"),
    ",",
    str(f"{ycom_support:.2f}"),
    ",",
    str(f"{xcom_support:.2f}"),
    ")",
)
coordination_matrix = pu.calc_coordination(
    support, kernel=np.ones((3, 3, 3)), debugging=False
)
surface = np.copy(support)
surface[coordination_matrix > 22] = 0  # remove the bulk 22
del coordination_matrix
gc.collect()

########################################################
# define edges using the coordination number of voxels #
########################################################
edges = pu.calc_coordination(support, kernel=np.ones((9, 9, 9)), debugging=False)
edges[support == 0] = 0
if debug:
    gu.multislices_plot(edges, vmin=0, title="Coordination matrix")
edges[edges > edges_coord] = 0  # remove facets and bulk
edges[np.nonzero(edges)] = 1  # edge support
gu.scatter_plot(
    array=np.asarray(np.nonzero(edges)).T,
    markersize=2,
    markercolor="b",
    labels=("axis 0", "axis 1", "axis 2"),
    title="edges",
)

########################################################
# define corners using the coordination number of voxels #
########################################################
corners = pu.calc_coordination(support, kernel=np.ones((9, 9, 9)), debugging=False)
corners[support == 0] = 0
if debug:
    gu.multislices_plot(corners, vmin=0, title="Coordination matrix")
corners[corners > corners_coord] = 0  # remove edges, facets and bulk
corners[np.nonzero(corners)] = 1  # corner support
gu.scatter_plot(
    array=np.asarray(np.nonzero(corners)).T,
    markersize=2,
    markercolor="b",
    labels=("axis 0", "axis 1", "axis 2"),
    title="corners",
)

######################################
# Initialize log files and .vti file #
######################################
with open(
    os.path.join(
        savedir, "S" + str(scan) + "_planes_iso" + str(support_threshold) + ".dat"
    ),
    "w",
) as summary_file, open(
    os.path.join(
        savedir, "S" + str(scan) + "_strain_iso" + str(support_threshold) + ".dat"
    ),
    "w",
) as allpoints_file:
    summary_file.write(
        "{: <10}".format("Plane #")
        + "\t"
        + "{: <10}".format("angle")
        + "\t"
        + "{: <10}".format("points #")
        + "\t"
        + "{: <10}".format("<strain>")
        + "\t"
        + "{: <10}".format("std dev")
        + "\t"
        + "{: <10}".format("A (x)")
        + "\t"
        + "{: <10}".format("B (y)")
        + "\t"
        + "{: <10}".format("C (z)")
        + "\t"
        + "D (Ax+By+CZ+D=0)"
        + "\t"
        "normal X" + "\t" + "normal Y" + "\t" + "normal Z" + "\n"
    )

    allpoints_file.write(
        "{: <10}".format("Plane #")
        + "\t"
        + "{: <10}".format("angle")
        + "\t"
        + "{: <10}".format("strain")
        + "\t"
        + "{: <10}".format("Z")
        + "\t"
        + "{: <10}".format("Y")
        + "\t"
        + "{: <10}".format("X")
        + "\n"
    )

    # prepare amp for vti file
    amp_array = np.transpose(np.flip(amp, 2)).reshape(amp.size)  # VTK axis 2 is flipped
    amp_array = numpy_support.numpy_to_vtk(amp_array)
    image_data = vtk.vtkImageData()
    image_data.SetOrigin(0, 0, 0)
    image_data.SetSpacing(voxel_size[0], voxel_size[1], voxel_size[2])
    image_data.SetExtent(0, nz - 1, 0, ny - 1, 0, nx - 1)
    pd = image_data.GetPointData()
    pd.SetScalars(amp_array)
    pd.GetArray(0).SetName("amp")

    # update vti file with edges
    edges_array = np.transpose(np.flip(edges, 2)).reshape(edges.size)
    edges_array = numpy_support.numpy_to_vtk(edges_array)
    pd.AddArray(edges_array)
    pd.GetArray(1).SetName("edges")
    pd.Update()

    index_vti = 2
    del amp, amp_array, edges_array
    gc.collect()
    #########################################################
    # save surface, edges and corners strain to the logfile #
    #########################################################
    fu.update_logfile(
        support=surface,
        strain_array=strain,
        summary_file=summary_file,
        allpoints_file=allpoints_file,
        label="surface",
    )

    fu.update_logfile(
        support=edges,
        strain_array=strain,
        summary_file=summary_file,
        allpoints_file=allpoints_file,
        label="edges",
    )

    fu.update_logfile(
        support=corners,
        strain_array=strain,
        summary_file=summary_file,
        allpoints_file=allpoints_file,
        label="corners",
    )

    del corners
    gc.collect()

    ###############################################################################
    # Iterate over the planes to find the corresponding surface facet             #
    # Effect of smoothing: the meshed support is smaller than the initial support #
    ###############################################################################
    summary_dict = {}
    for label in unique_labels:
        print("\nPlane", label)
        # raw fit including all points
        plane = np.copy(all_planes)
        plane[plane != label] = 0
        plane[plane == label] = 1
        if plane[plane == 1].sum() == 0:  # no points on the plane
            print("Raw fit: no points for plane", label)
            continue
        ################################################################
        # fit a plane to the voxels isolated by watershed segmentation #
        ################################################################
        # Why not using directly the centroid to find plane equation?
        # Because it does not distinguish pixels coming from
        # different but parallel facets
        coeffs, plane_indices, errors, stop = fu.fit_plane(
            plane=plane, label=label, debugging=debug
        )
        if stop:
            print("No points remaining after raw fit for plane", label)
            continue

        # update plane by filtering out pixels too far from the fit plane
        plane, stop = fu.distance_threshold(
            fit=coeffs,
            indices=plane_indices,
            plane_shape=plane.shape,
            max_distance=max_distance_plane,
        )
        grown_points = plane[plane == 1].sum().astype(int)
        if stop:  # no points on the plane
            print("Refined fit: no points for plane", label)
            continue
        print(
            "Plane",
            label,
            ", ",
            str(grown_points),
            "points after checking distance to plane",
        )
        plane_indices = np.nonzero(plane)  # plane_indices is a tuple of 3 arrays

        # check that the plane normal is not flipped using
        # the support gradient at the center of mass of the facet
        zcom_facet, ycom_facet, xcom_facet = center_of_mass(plane)
        mean_gradient = fu.surface_gradient(
            (zcom_facet, ycom_facet, xcom_facet), support=support
        )[0]
        plane_normal = np.array(
            [coeffs[0], coeffs[1], coeffs[2]]
        )  # normal is [a, b, c] if ax+by+cz+d=0
        plane_normal = plane_normal / np.linalg.norm(plane_normal)
        if (
            np.dot(plane_normal, mean_gradient) < 0
        ):  # normal is in the reverse direction
            print("Flip normal direction plane", str(label))
            plane_normal = -1 * plane_normal
            coeffs = (-coeffs[0], -coeffs[1], -coeffs[2], -coeffs[3])

        #########################################################
        # Look for the surface: correct for the offset between  #
        # the plane equation and the outer shell of the support #
        #########################################################
        # crop the support to a small ROI included in the plane box
        surf0, surf1, surf2 = fu.surface_indices(
            surface=surface, plane_indices=plane_indices, margin=3
        )
        if debug:
            gu.scatter_plot_overlaid(
                arrays=(
                    np.asarray(plane_indices).T,
                    np.concatenate(
                        (
                            surf0[:, np.newaxis],
                            surf1[:, np.newaxis],
                            surf2[:, np.newaxis],
                        ),
                        axis=1,
                    ),
                ),
                markersizes=(8, 2),
                markercolors=("b", "r"),
                labels=("axis 0", "axis 1", "axis 2"),
                title="Plane" + str(label) + " before shifting",
            )

        # find the direction in which the plane should be shifted to cross the surface
        dist = np.zeros(len(surf0))
        for point, _ in enumerate(surf0):
            dist[point] = (
                coeffs[0] * surf0[point]
                + coeffs[1] * surf1[point]
                + coeffs[2] * surf2[point]
                + coeffs[3]
            ) / np.linalg.norm(plane_normal)
        mean_dist = dist.mean()
        print(
            "Mean distance of plane ",
            label,
            " to outer shell = " + str(f"{mean_dist:.2f}") + " pixels",
        )

        # offset the plane by mean_dist/2 and see if the plane is closer to
        # the surface or went in the wrong direction
        dist = np.zeros(len(surf0))
        offset = mean_dist / 2

        for point, _ in enumerate(surf0):
            dist[point] = (
                coeffs[0] * surf0[point]
                + coeffs[1] * surf1[point]
                + coeffs[2] * surf2[point]
                + coeffs[3]
                - offset
            ) / np.linalg.norm(plane_normal)
        new_dist = dist.mean()
        print(
            "<distance> of plane ",
            label,
            " to outer shell after offsetting by mean_distance/2 = "
            + str(f"{new_dist:.2f}")
            + " pixels",
        )
        # these directions are for a mesh smaller than the support
        step_shift = 0.5  # will scan with subpixel step through the crystal
        # in order to not miss voxels
        if mean_dist * new_dist < 0:  # crossed the support surface, correct direction
            step_shift = np.sign(mean_dist) * step_shift
        elif (
            abs(new_dist) - abs(mean_dist) < 0
        ):  # moving towards the surface, correct direction
            step_shift = np.sign(mean_dist) * step_shift
        else:  # moving away from the surface, wrong direction
            step_shift = -1 * np.sign(mean_dist) * step_shift

        ##############################################
        # shift the fit plane along its normal until #
        # the normal is crossed and go back one step #
        ##############################################
        total_offset = fu.find_facet(
            refplane_indices=plane_indices,
            surf_indices=(surf0, surf1, surf2),
            original_shape=surface.shape,
            step_shift=step_shift,
            plane_label=label,
            plane_coeffs=coeffs,
            min_points=grown_points / 5,
            debugging=debug,
        )

        # correct the offset of the fit plane to match the surface
        coeffs = list(coeffs)
        coeffs[3] = coeffs[3] - total_offset
        # shift plane indices
        plane_newindices0, plane_newindices1, plane_newindices2 = fu.offset_plane(
            indices=plane_indices, offset=total_offset, plane_normal=plane_normal
        )

        plane = np.zeros(surface.shape)
        plane[plane_newindices0, plane_newindices1, plane_newindices2] = 1

        # use only pixels belonging to the outer shell of the support
        plane = plane * surface

        if debug:
            # plot plane points overlaid with the support
            plane_indices = np.nonzero(plane == 1)
            gu.scatter_plot_overlaid(
                arrays=(
                    np.asarray(plane_indices).T,
                    np.concatenate(
                        (
                            surf0[:, np.newaxis],
                            surf1[:, np.newaxis],
                            surf2[:, np.newaxis],
                        ),
                        axis=1,
                    ),
                ),
                markersizes=(8, 2),
                markercolors=("b", "r"),
                labels=("axis 0", "axis 1", "axis 2"),
                title="Plane"
                + str(label)
                + " after finding the surface\n"
                + "Points number="
                + str(len(plane_indices[0])),
            )

        if plane[plane == 1].sum() == 0:  # no point belongs to the support
            print("Plane ", label, " , no point belongs to support")
            continue

        #################################
        # grow the facet on the surface #
        #################################
        print("Growing the facet at the surface")
        iterate = 0
        while stop == 0:
            previous_nb = plane[plane == 1].sum()
            plane, stop = fu.grow_facet(
                fit=coeffs,
                plane=plane,
                label=label,
                support=support,
                max_distance=1.5 * max_distance_plane,
                debugging=debug,
            )
            # here the distance threshold is larger in order to reach
            # voxels missed by the first plane fit when rounding vertices to integer.
            # Anyway we intersect it with the surface therefore it can not go crazy.
            plane_indices = np.nonzero(plane)
            iterate = iterate + 1
            plane = (
                plane * surface
            )  # use only pixels belonging to the outer shell of the support
            if plane[plane == 1].sum() == previous_nb:
                print("Growth: maximum size reached")
                break
            if iterate == 50:  # it is likely that we are stuck in an infinite loop
                # after this number of iterations
                print("Growth: maximum iteration number reached")
                break
        grown_points = plane[plane == 1].sum().astype(int)
        print(
            "Plane ",
            label,
            ", ",
            str(grown_points),
            "points after growing facet at the surface",
        )

        if debug:
            plane_indices = np.nonzero(plane == 1)
            surf0, surf1, surf2 = fu.surface_indices(
                surface=surface, plane_indices=plane_indices, margin=3
            )
            gu.scatter_plot_overlaid(
                arrays=(
                    np.asarray(plane_indices).T,
                    np.concatenate(
                        (
                            surf0[:, np.newaxis],
                            surf1[:, np.newaxis],
                            surf2[:, np.newaxis],
                        ),
                        axis=1,
                    ),
                ),
                markersizes=(8, 2),
                markercolors=("b", "r"),
                labels=("axis 0", "axis 1", "axis 2"),
                title="Plane"
                + str(label)
                + " after 1st growth at the surface\n iteration"
                + str(iterate)
                + "- Points number="
                + str(len(plane_indices[0])),
            )

        ################################################################
        # refine plane fit, now we are sure that we are at the surface #
        ################################################################
        coeffs, plane_indices, errors, stop = fu.fit_plane(
            plane=plane, label=label, debugging=debug
        )
        if stop:
            print("No points remaining after refined fit for plane", label)
            continue

        if debug:
            surf0, surf1, surf2 = fu.surface_indices(
                surface=surface, plane_indices=plane_indices, margin=3
            )
            gu.scatter_plot_overlaid(
                arrays=(
                    np.asarray(plane_indices).T,
                    np.concatenate(
                        (
                            surf0[:, np.newaxis],
                            surf1[:, np.newaxis],
                            surf2[:, np.newaxis],
                        ),
                        axis=1,
                    ),
                ),
                markersizes=(8, 2),
                markercolors=("b", "r"),
                labels=("axis 0", "axis 1", "axis 2"),
                title="Plane"
                + str(label)
                + " after refined fit at surface\n iteration"
                + str(iterate)
                + "- Points number="
                + str(len(plane_indices[0])),
            )

        # update plane by filtering out pixels too far from the fit plane
        plane, stop = fu.distance_threshold(
            fit=coeffs,
            indices=plane_indices,
            plane_shape=plane.shape,
            max_distance=max_distance_plane,
        )
        if stop:  # no points on the plane
            print("Refined fit: no points for plane", label)
            continue
        print(
            "Plane",
            label,
            ", ",
            str(plane[plane == 1].sum()),
            "points after refined fit",
        )
        plane_indices = np.nonzero(plane)

        if debug:
            surf0, surf1, surf2 = fu.surface_indices(
                surface=surface, plane_indices=plane_indices, margin=3
            )
            gu.scatter_plot_overlaid(
                arrays=(
                    np.asarray(plane_indices).T,
                    np.concatenate(
                        (
                            surf0[:, np.newaxis],
                            surf1[:, np.newaxis],
                            surf2[:, np.newaxis],
                        ),
                        axis=1,
                    ),
                ),
                markersizes=(8, 2),
                markercolors=("b", "r"),
                labels=("axis 0", "axis 1", "axis 2"),
                title="Plane"
                + str(label)
                + " after distance threshold at surface\n"
                + "Points number="
                + str(len(plane_indices[0])),
            )

        # check that the plane normal is not flipped using the support gradient
        # at the center of mass of the facet
        zcom_facet, ycom_facet, xcom_facet = center_of_mass(plane)
        mean_gradient = fu.surface_gradient(
            (zcom_facet, ycom_facet, xcom_facet), support=support
        )[0]
        plane_normal = np.array(
            [coeffs[0], coeffs[1], coeffs[2]]
        )  # normal is [a, b, c] if ax+by+cz+d=0
        plane_normal = plane_normal / np.linalg.norm(plane_normal)
        if (
            np.dot(plane_normal, mean_gradient) < 0
        ):  # normal is in the reverse direction
            print("Flip normal direction plane", str(label))
            plane_normal = -1 * plane_normal
            coeffs = (-coeffs[0], -coeffs[1], -coeffs[2], -coeffs[3])

        ############################################
        # final growth of the facet on the surface #
        ############################################
        print("Final growth of the facet")
        iterate = 0
        while stop == 0:
            previous_nb = plane[plane == 1].sum()
            plane, stop = fu.grow_facet(
                fit=coeffs,
                plane=plane,
                label=label,
                support=support,
                max_distance=1.5 * max_distance_plane,
                debugging=debug,
            )
            plane = (
                plane * surface
            )  # use only pixels belonging to the outer shell of the support
            iterate = iterate + 1
            if plane[plane == 1].sum() == previous_nb:
                print("Growth: maximum size reached")
                break
            if iterate == 50:  # it is likely that we are stuck in an infinite loop
                # after this number of iterations
                print("Growth: maximum iteration number reached")
                break
        grown_points = plane[plane == 1].sum().astype(int)
        # plot plane points overlaid with the support
        print(
            "Plane ",
            label,
            ", ",
            str(grown_points),
            "points after the final growth of the facet",
        )

        if debug:
            plane_indices = np.nonzero(plane)
            surf0, surf1, surf2 = fu.surface_indices(
                surface=surface, plane_indices=plane_indices, margin=3
            )
            gu.scatter_plot_overlaid(
                arrays=(
                    np.asarray(plane_indices).T,
                    np.concatenate(
                        (
                            surf0[:, np.newaxis],
                            surf1[:, np.newaxis],
                            surf2[:, np.newaxis],
                        ),
                        axis=1,
                    ),
                ),
                markersizes=(8, 2),
                markercolors=("b", "r"),
                labels=("axis 0", "axis 1", "axis 2"),
                title="Plane"
                + str(label)
                + " final growth at the surface\nPoints number="
                + str(len(plane_indices[0])),
            )

        #####################################
        # remove point belonging to an edge #
        #####################################
        plane[np.nonzero(edges)] = 0
        plane_indices = np.nonzero(plane)
        if debug:
            surf0, surf1, surf2 = fu.surface_indices(
                surface=surface, plane_indices=plane_indices, margin=3
            )
            gu.scatter_plot_overlaid(
                arrays=(
                    np.asarray(plane_indices).T,
                    np.concatenate(
                        (
                            surf0[:, np.newaxis],
                            surf1[:, np.newaxis],
                            surf2[:, np.newaxis],
                        ),
                        axis=1,
                    ),
                ),
                markersizes=(8, 2),
                markercolors=("b", "r"),
                labels=("axis 0", "axis 1", "axis 2"),
                title="Plane"
                + str(label)
                + " after edge removal\nPoints number="
                + str(len(plane_indices[0])),
            )
        print(
            "Plane ",
            label,
            ", ",
            str(len(plane_indices[0])),
            "points after removing edges",
        )

        ##############################################################################
        # calculate the angle between the plane normal and the measurement direction #
        ##############################################################################
        # correct plane_normal for the eventual anisotropic voxel size
        plane_normal = np.array(
            [
                plane_normal[0] * 2 * np.pi / voxel_size[0],
                plane_normal[1] * 2 * np.pi / voxel_size[1],
                plane_normal[2] * 2 * np.pi / voxel_size[2],
            ]
        )
        plane_normal = plane_normal / np.linalg.norm(plane_normal)

        # check where is the measurement direction
        if projection_axis == 0:  # q aligned along the 1st axis
            ref_axis = np.array([1, 0, 0])
        elif projection_axis == 1:  # q aligned along the 2nd axis
            ref_axis = np.array([0, 1, 0])
        elif projection_axis == 2:  # q aligned along the 3rd axis
            ref_axis = np.array([0, 0, 1])
        else:
            ref_axis = np.zeros(3)
            print("projection_axis should be a basis axis of the reconstructed array")
            sys.exit()

        # calculate the angle of the plane normal to the measurement direction,
        # which is aligned along projection_axis
        angle_plane = 180 / np.pi * np.arccos(np.dot(ref_axis, plane_normal))
        print(
            f"Angle between plane {label} and "
            f"the measurement direction = {angle_plane:.2f} degrees"
        )
        # update the dictionnary
        summary_dict[label] = {
            "angle_plane": angle_plane,
            "plane_coeffs": coeffs,
            "plane_normal": plane_normal,
            "plane_indices": plane_indices,
        }

    del support, all_planes
    gc.collect()

    #################################################
    # look for voxels attributed to multiple facets #
    #################################################
    # full_indices is a list a tuples, each tuple being a point (z, y, x)
    full_indices = [
        list(
            zip(
                summary_dict[label]["plane_indices"][0],
                summary_dict[label]["plane_indices"][1],
                summary_dict[label]["plane_indices"][2],
            )
        )[point]
        for label in summary_dict
        for point in range(
            len(
                list(
                    zip(
                        summary_dict[label]["plane_indices"][0],
                        summary_dict[label]["plane_indices"][1],
                        summary_dict[label]["plane_indices"][2],
                    )
                )
            )
        )
    ]
    # count the number of times each point appears in the list
    counter_dict = collections.Counter(full_indices)

    # creates the list of duplicated voxels,
    # these will be set to the background later on
    duplicates = [key for key, value in counter_dict.items() if value > 1]

    # modify the structure of the list so that it can be directly
    # used for array indexing (like the output of np.nonzero)
    ind_0 = np.asarray([duplicates[point][0] for point in range(len(duplicates))])
    ind_1 = np.asarray([duplicates[point][1] for point in range(len(duplicates))])
    ind_2 = np.asarray([duplicates[point][2] for point in range(len(duplicates))])
    duplicates = (ind_0, ind_1, ind_2)

    ################################
    # update the log and VTK files #
    ################################
    print("\nFiltering out voxels attributed to multiple facets")
    print(f"{len(duplicates[0])} points attributed to multiple facets will be removed")
    for label in summary_dict:
        plane = np.zeros((nz, ny, nx), dtype=int)
        plane[summary_dict[label]["plane_indices"]] = 1
        number_before = (plane == 1).sum()
        # remove voxels attributed to multiple facets
        plane[duplicates] = 0
        plane_indices = np.nonzero(plane)

        nb_points = len(plane_indices[0])
        if nb_points == 0:  # no point belongs to the support
            print(
                "Plane ", label, " , no point remaining after checking for duplicates"
            )
            continue
        print(
            "Plane ",
            label,
            " : {} points before, {} points after checking for duplicates".format(
                number_before, nb_points
            ),
        )

        surf0, surf1, surf2 = fu.surface_indices(
            surface=surface, plane_indices=plane_indices, margin=3
        )
        gu.scatter_plot_overlaid(
            arrays=(
                np.asarray(plane_indices).T,
                np.concatenate(
                    (surf0[:, np.newaxis], surf1[:, np.newaxis], surf2[:, np.newaxis]),
                    axis=1,
                ),
            ),
            markersizes=(8, 2),
            markercolors=("b", "r"),
            labels=("axis 0", "axis 1", "axis 2"),
            title="Final plane"
            + str(label)
            + " after checking for duplicates\nPoints number="
            + str(nb_points),
        )

        fu.update_logfile(
            support=plane,
            strain_array=strain,
            summary_file=summary_file,
            allpoints_file=allpoints_file,
            label=label,
            angle_plane=summary_dict[label]["angle_plane"],
            plane_coeffs=summary_dict[label]["plane_coeffs"],
            plane_normal=summary_dict[label]["plane_normal"],
        )

        # update vti file
        plane_array = np.transpose(np.flip(plane, 2)).reshape(
            plane.size
        )  # VTK axis 2 is flipped
        plane_array = numpy_support.numpy_to_vtk(plane_array)
        pd.AddArray(plane_array)
        pd.GetArray(index_vti).SetName("plane_" + str(label))
        pd.Update()
        index_vti = index_vti + 1

    ################
    # update files #
    ################
    summary_file.write(
        "\n" + "Isosurface value" + "\t" f"{str(support_threshold): <10}"
    )
    allpoints_file.write(
        "\n" + "Isosurface value" + "\t" f"{str(support_threshold): <10}"
    )

# export data to file
writer = vtk.vtkXMLImageDataWriter()
writer.SetFileName(
    os.path.join(
        savedir,
        "S" + str(scan) + "_refined planes_iso" + str(support_threshold) + ".vti",
    )
)
writer.SetInputData(image_data)
writer.Write()
print("\nEnd of script")
plt.ioff()
plt.show()
