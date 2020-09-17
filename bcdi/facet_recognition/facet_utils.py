# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

from scipy.ndimage.measurements import center_of_mass
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patches
import gc
import sys
sys.path.append('//win.desy.de/home/carnisj/My Documents/myscripts/bcdi/')
import bcdi.graph.graph_utils as gu
import bcdi.utils.utilities as util

colormap = gu.Colormap()
default_cmap = colormap.cmap


def calc_stereoproj_facet(projection_axis, vectors, radius_mean, stereo_center):
    """
    Calculate the coordinates of normals in the stereographic projection depending on the reference axis
     see Nanoscale 10, 4833 (2018).

    :param projection_axis: the projection is performed on q plane perpendicular to that axis (0, 1 or 2)
    :param vectors: array of vectors to be projected (nb_vectors rows x 3 columns)
    :param radius_mean: q radius from which the projection will be done
    :param stereo_center: offset of the projection plane along the reflection axis, in the same unit as radius_mean
     If stereo_center = 0, the projection plane will be the equator.
    :return: the coordinates of the stereographic projection for the projection from the South pole(1st and 2nd columns)
      and from the North pole (3rd and 4th columns) projection, rescaled from radius_mean to 90 degrees
    """
    if projection_axis not in [0, 1, 2]:
        raise ValueError('reflection_axis should be a basis axis of the reconstructed array')

    # calculate u and v from xyz
    stereo_proj = np.zeros((vectors.shape[0], 4), dtype=vectors.dtype)
    # stereo_proj[:, 0] is the euclidian u_south, stereo_proj[:, 1] is the euclidian v_south
    # stereo_proj[:, 2] is the euclidian u_north, stereo_proj[:, 3] is the euclidian v_north

    if projection_axis == 0:  # q aligned along the 1st axis (Z downstream in CXI convention)
        for idx in range(vectors.shape[0]):
            stereo_proj[idx, 0] = radius_mean * vectors[idx, 1] / (radius_mean + vectors[idx, 0] - stereo_center)  # u_s
            stereo_proj[idx, 1] = radius_mean * vectors[idx, 2] / (radius_mean + vectors[idx, 0] - stereo_center)  # v_s
            stereo_proj[idx, 2] = radius_mean * vectors[idx, 1] / (radius_mean + stereo_center - vectors[idx, 0])  # u_n
            stereo_proj[idx, 3] = radius_mean * vectors[idx, 2] / (radius_mean + stereo_center - vectors[idx, 0])  # v_n
        uv_labels = ('axis 1', 'axis 2')  # axes corresponding to u and v respectively, used in plots

    elif projection_axis == 1:  # q aligned along the 2nd axis (Y vertical up in CXI convention)
        for idx in range(vectors.shape[0]):
            stereo_proj[idx, 0] = radius_mean * vectors[idx, 0] / (radius_mean + vectors[idx, 1] - stereo_center)  # u_s
            stereo_proj[idx, 1] = radius_mean * vectors[idx, 2] / (radius_mean + vectors[idx, 1] - stereo_center)  # v_s
            stereo_proj[idx, 2] = radius_mean * vectors[idx, 0] / (radius_mean + stereo_center - vectors[idx, 1])  # u_n
            stereo_proj[idx, 3] = radius_mean * vectors[idx, 2] / (radius_mean + stereo_center - vectors[idx, 1])  # v_n
        uv_labels = ('axis 0', 'axis 2')  # axes corresponding to u and v respectively, used in plots

    else:  # q aligned along the 3rd axis (X outboard in CXI convention)
        for idx in range(vectors.shape[0]):
            stereo_proj[idx, 0] = radius_mean * vectors[idx, 0] / (radius_mean + vectors[idx, 2] - stereo_center)  # u_s
            stereo_proj[idx, 1] = radius_mean * vectors[idx, 1] / (radius_mean + vectors[idx, 2] - stereo_center)  # v_s
            stereo_proj[idx, 2] = radius_mean * vectors[idx, 0] / (radius_mean + stereo_center - vectors[idx, 2])  # u_n
            stereo_proj[idx, 3] = radius_mean * vectors[idx, 1] / (radius_mean + stereo_center - vectors[idx, 2])  # v_n
        uv_labels = ('axis 0', 'axis 1')  # axes corresponding to u and v respectively, used in plots
    stereo_proj = stereo_proj / radius_mean * 90  # rescale from radius_mean to 90

    return stereo_proj, uv_labels


def detect_edges(faces):
    """
    Find indices of vertices defining non-shared edges

    :param faces: ndarray of m*3 faces
    :return: 1D list of indices of vertices defining non-shared edges (near hole...)
    """
    # Get the three edges per triangle
    edge1 = np.copy(faces[:, 0:2])
    edge2 = np.array([np.copy(faces[:, 0]), np.copy(faces[:, 2])]).T
    edge3 = np.array([np.copy(faces[:, 1]), np.copy(faces[:, 2])]).T
    edge1.sort(axis=1)
    edge2.sort(axis=1)
    edge3.sort(axis=1)

    # list of edges without redundancy
    edges = np.concatenate((edge1, edge2, edge3), axis=0)
    edge_list, edges_indices, edges_counts = np.unique(edges, return_index=True, return_counts=True, axis=0)

    # isolate non redundant edges
    unique_edges = edge_list[edges_counts == 1].flatten()
    return unique_edges


def distance_threshold(fit, indices, shape, max_distance=0.90):
    """
    Filter out pixels depending on their distance to a fit plane

    :param fit: coefficients of the plane (tuple of 3 numbers)
    :param indices: plane indices
    :param shape: shape of the intial plane array
    :param max_distance: max distance allowed from the fit plane in pixels
    :return: the updated plane, a stop flag
    """
    plane = np.zeros(shape, dtype=int)
    no_points = 0
    indx = indices[0]
    indy = indices[1]
    indz = indices[2]
    if len(indices[0]) == 0:
        no_points = 1
        return plane, no_points
    # remove outsiders based on distance to plane
    plane_normal = np.array([fit[0], fit[1], -1])  # normal is [a, b, c] if ax+by+cz+d=0
    for point in range(len(indices[0])):
        dist = abs(fit[0]*indx[point] + fit[1]*indy[point] -
                   indz[point] + fit[2])/np.linalg.norm(plane_normal)
        if dist < max_distance:
            plane[indx[point], indy[point], indz[point]] = 1
    if plane[plane == 1].sum() == 0:
        print('Distance_threshold: no points for plane')
        no_points = 1
        return plane, no_points
    return plane, no_points


def equirectangular_proj(normals, intensity, cmap=default_cmap, bw_method=0.03, min_distance=10,
                         background_threshold=-0.35, debugging=False):
    """
    Detect facets in an object using an equirectangular projection of normals to mesh triangles
     and watershed segmentation.

    :param normals: normals array
    :param intensity: intensity array
    :param cmap: colormap used for plotting
    :param bw_method: bw_method of gaussian_kde
    :param min_distance: min_distance of corner_peaks()
    :param background_threshold: threshold for background determination (depth of the KDE)
    :param debugging: if True, show plots for debugging
    :return: ndarray of labelled regions
    """
    from scipy import stats
    from scipy import ndimage
    from skimage.feature import corner_peaks
    from skimage.morphology import watershed

    # check normals for nan
    list_nan = np.argwhere(np.isnan(normals))
    normals = np.delete(normals, list_nan[::3, 0], axis=0)
    intensity = np.delete(intensity, list_nan[::3, 0], axis=0)

    # calculate latitude and longitude from xyz, this is equal to the equirectangular flat square projection
    long_lat = np.zeros((normals.shape[0], 2), dtype=normals.dtype)
    for i in range(normals.shape[0]):
        if normals[i, 1] == 0 and normals[i, 0] == 0:
            continue
        long_lat[i, 0] = np.arctan2(normals[i, 1], normals[i, 0])  # longitude
        long_lat[i, 1] = np.arcsin(normals[i, 2])  # latitude
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(long_lat[:, 0], long_lat[:, 1], c=intensity, cmap=cmap)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi / 2, np.pi / 2)
    plt.axis('scaled')
    plt.title('Equirectangular projection of the weighted point densities before KDE')
    plt.pause(0.1)

    del intensity
    gc.collect()

    # kernel density estimation
    kde = stats.gaussian_kde(long_lat.T, bw_method=bw_method)
    # input should be a 2D array with shape (# of dims, # of data)

    # Create a regular 3D grid
    yi, xi = np.mgrid[-np.pi/2:np.pi/2:150j, -np.pi:np.pi:300j]  # vertical, horizontal

    # Evaluate the KDE on a regular grid...
    coords = np.vstack([item.ravel() for item in [xi, yi]])
    # coords is a contiguous flattened array of coordinates of shape (2, size(xi))

    density = -1 * kde(coords).reshape(xi.shape)  # inverse density for later watershed segmentation

    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter = ax.scatter(xi, yi, c=density, cmap=cmap, vmin=-1.5, vmax=0)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi / 2, np.pi / 2)
    fig.colorbar(scatter)
    plt.axis('scaled')
    plt.title('Equirectangular projection of the KDE')
    plt.pause(0.1)

    # identification of local minima
    density[density > background_threshold] = 0  # define the background
    mask = np.copy(density)
    mask[mask != 0] = 1

    plt.figure()
    plt.imshow(mask, cmap=cmap, interpolation='nearest')
    plt.title('Background mask')
    plt.gca().invert_yaxis()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter = ax.scatter(xi, yi, c=density, cmap=cmap)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi / 2, np.pi / 2)
    fig.colorbar(scatter)
    plt.axis('scaled')
    plt.title('KDE after background definition')
    plt.pause(0.1)

    # Generate the markers as local minima of the distance to the background
    distances = ndimage.distance_transform_edt(density)
    if debugging:
        plt.figure()
        plt.imshow(distances, cmap=cmap, interpolation='nearest')
        plt.title('Distances')
        plt.gca().invert_yaxis()
        plt.pause(0.1)

    # find peaks
    local_maxi = corner_peaks(distances, exclude_border=False, min_distance=min_distance, indices=False)  #
    if debugging:
        plt.figure()
        plt.imshow(local_maxi, interpolation='nearest')
        plt.title('local_maxi')
        plt.gca().invert_yaxis()
        plt.pause(0.1)

    # define markers for each peak
    markers = ndimage.label(local_maxi)[0]
    if debugging:
        plt.figure()
        plt.imshow(markers, interpolation='nearest')
        plt.title('markers')
        plt.colorbar()
        plt.gca().invert_yaxis()
        plt.pause(0.1)

    # watershed segmentation
    labels = watershed(-distances, markers, mask=mask)
    print('There are', str(labels.max()), 'facets')  # label 0 is the background

    plt.figure()
    plt.imshow(labels, cmap=cmap, interpolation='nearest')
    plt.title('Separated objects')
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.pause(0.1)

    return labels, long_lat


def find_neighbours(vertices, faces):
    """
    Get the list of neighbouring vertices for each vertex.

    :param vertices: ndarray of n*3 vertices
    :param faces: ndarray of m*3 faces
    :return: list of lists of indices
    """
    neighbors = [None]*vertices.shape[0]

    for indx in range(faces.shape[0]):
        if neighbors[faces[indx, 0]] is None:
            neighbors[faces[indx, 0]] = [faces[indx, 1], faces[indx, 2]]
        else:
            neighbors[faces[indx, 0]].append(faces[indx, 1])
            neighbors[faces[indx, 0]].append(faces[indx, 2])
        if neighbors[faces[indx, 1]] is None:
            neighbors[faces[indx, 1]] = [faces[indx, 2], faces[indx, 0]]
        else:
            neighbors[faces[indx, 1]].append(faces[indx, 2])
            neighbors[faces[indx, 1]].append(faces[indx, 0])
        if neighbors[faces[indx, 2]] is None:
            neighbors[faces[indx, 2]] = [faces[indx, 0], faces[indx, 1]]
        else:
            neighbors[faces[indx, 2]].append(faces[indx, 0])
            neighbors[faces[indx, 2]].append(faces[indx, 1])

    for indx in range(len(neighbors)):
        temp_list = [point for point in neighbors[indx] if point is not None]  # remove None values
        neighbors[indx] = list(set(temp_list))  # remove redundant indices in each sublist

    return neighbors


def fit_plane(plane, label, debugging=False):
    """
    Fit a plane to labelled indices using the equation a*x+ b*y + c*z + d = 0.

    :param plane: 3D binary array, where the voxels belonging to the plane are set to 1 and others are set to 0.
    :param label: int, label of the plane used for the title in plots
    :param debugging: show plots for debugging
    :return: fit parameters (a, b, c, d), plane indices after filtering, errors associated, a stop flag
    """
    indices = np.asarray(np.nonzero(plane))
    no_points = 0
    if len(indices[0]) == 0:
        no_points = 1
        return 0, indices, 0, no_points
    x_com, y_com, z_com = center_of_mass(plane)

    # remove isolated points, which probably do not belong to the plane
    if debugging:
        gu.scatter_plot(np.asarray(np.nonzero(plane)).transpose(), labels=('axis 0', 'axis 1', 'axis 2'),
                        title='Points before coordination threshold plane ' + str(label))

    for point in range(indices.shape[1]):
        neighbors = plane[indices[0, point]-2:indices[0, point]+3, indices[1, point]-2:indices[1, point]+3,
                          indices[2, point]-2:indices[2, point]+3].sum()
        if neighbors < 5:
            plane[indices[0, point], indices[1, point], indices[2, point]] = 0

    print('Fit plane', label, ', ', str(indices.shape[1]-plane[plane == 1].sum()), 'points isolated, ',
          str(plane[plane == 1].sum()), 'remaining')
    if debugging:
        gu.scatter_plot(np.asarray(np.nonzero(plane)).transpose(), labels=('axis 0', 'axis 1', 'axis 2'),
                        title='Points after coordination threshold plane ' + str(label))

    # update plane indices
    indices = np.asarray(np.nonzero(plane))
    if len(indices[0]) == 0:
        no_points = 1
        return 0, indices, 0, no_points

    # remove also points farther than 2 times the mean distance to the COM
    dist = np.zeros(indices.shape[1])
    for point in range(indices.shape[1]):
        dist[point] = np.sqrt((indices[0, point]-x_com)**2+(indices[1, point]-y_com)**2+(indices[2, point]-z_com)**2)
    average_dist = np.mean(dist)
    debugging = True
    if debugging:
        gu.scatter_plot(np.asarray(np.nonzero(plane)).transpose(), labels=('axis 0', 'axis 1', 'axis 2'),
                        title='Points before distance threshold plane ' + str(label))

    for point in range(indices.shape[1]):
        if dist[point] > 2 * average_dist:
            plane[indices[0, point], indices[1, point], indices[2, point]] = 0
    print('Fit plane', label, ', ', str(indices.shape[1] - plane[plane == 1].sum()), 'points too far from COM, ',
          str(plane[plane == 1].sum()), 'remaining')
    if debugging:
        gu.scatter_plot(np.asarray(np.nonzero(plane)).transpose(), labels=('axis 0', 'axis 1', 'axis 2'),
                        title='Points after distance threshold plane ' + str(label))

    # update plane indices and check if enough points remain
    indices = np.asarray(np.nonzero(plane))
    if len(indices[0]) < 5:
        no_points = 1
        return 0, indices, 0, no_points

    # the fit parameters are (a, b, c, d) such that a*x + b*y + c*z + d = 0
    params, std_param = util.plane_fit(indices=indices, label=label, debugging=debugging)
    return params, indices, std_param, no_points


def grow_facet(fit, plane, label, support, max_distance=0.90, debugging=True):
    """
    Find voxels of the object which belong to a facet using the facet plane equation and the distance to the plane.

    :param fit: coefficients of the plane (tuple of 3 numbers)
    :param plane: 3D binary array of the shape of the data
    :param label: the label of the plane processed
    :param support: binary support of the object
    :param max_distance: in pixels, maximum allowed distance to the facet plane of a voxel
    :param debugging: set to True to see plots
    :return: the updated plane, a stop flag
    """
    from scipy.signal import convolve
    nbz, nby, nbx = plane.shape
    indices = np.nonzero(plane)
    if len(indices[0]) == 0:
        no_points = 1
        return plane, no_points
    kernel = np.ones((3, 3, 3))

    start_z = max(indices[0].min()-20, 0)
    stop_z = min(indices[0].max()+21, nbz)
    start_y = max(indices[1].min()-20, 0)
    stop_y = min(indices[1].max()+21, nby)
    start_x = max(indices[2].min()-20, 0)
    stop_x = min(indices[2].max()+21, nbx)

    # find nearby voxels using the coordination number
    obj = np.copy(plane[start_z:stop_z, start_y:stop_y, start_x: stop_x])
    coord = np.rint(convolve(obj, kernel, mode='same'))
    coord = coord.astype(int)
    coord[np.nonzero(coord)] = 1

    # update plane with new voxels
    temp_plane = np.copy(plane)
    temp_plane[start_z:stop_z, start_y:stop_y, start_x: stop_x] = coord
    # remove voxels not belonging to the support
    temp_plane[support == 0] = 0
    # check distance of new voxels to the plane
    new_indices = np.nonzero(temp_plane)

    plane, no_points = distance_threshold(fit=fit, indices=new_indices, shape=temp_plane.shape,
                                          max_distance=max_distance)

    if debugging and len(new_indices[0]) != 0:
        indices = np.nonzero(plane)
        gu.scatter_plot(array=np.asarray(indices).T, labels=('x', 'y', 'z'),
                        title='Plane' + str(label) + ' after 1 cycle of facet growing')
        print(str(len(indices[0])) + ' after 1 cycle of facet growing')
    return plane, no_points


def offset_plane(indices, offset, plane_normal):
    """
    Shift plane indices by the offset value in order to scan perpendicular to the plane.

    :param indices: tuple of 3 1D ndarrays (array shape = nb_points)
    :param offset: offset to be applied to the indices (offset of the plane)
    :param plane_normal: ndarray of 3 elements, normal to the plane
    :return: offseted indices
    """
    if not isinstance(indices, tuple):
        raise ValueError('indices should be a tuple of 3 1D ndarrays')
    new_indices0 = np.rint(indices[0] +
                           offset * np.dot(np.array([1, 0, 0]), plane_normal /
                                           np.linalg.norm(plane_normal))).astype(int)
    new_indices1 = np.rint(indices[1] +
                           offset * np.dot(np.array([0, 1, 0]), plane_normal /
                                           np.linalg.norm(plane_normal))).astype(int)
    new_indices2 = np.rint(indices[2] +
                           offset * np.dot(np.array([0, 0, 1]), plane_normal /
                                           np.linalg.norm(plane_normal))).astype(int)
    return new_indices0, new_indices1, new_indices2


def remove_duplicates(vertices, faces, debugging=False):
    """
    Remove duplicates in a list of vertices and faces (a face is atriangle made of tree vertices).

    :param vertices: a ndarray of vertices, shape (N, 3)
    :param faces: a ndarray of vertex indices, shape (M, 3)
    :param debugging: True to see which vertices are duplicated and how lists are modified
    :return: the updated vertices and faces with duplicates removed in place
    """
    # find indices which are duplicated
    uniq_vertices, uniq_inverse = np.unique(vertices, axis=0, return_inverse=True)
    indices, count = np.unique(uniq_inverse, return_counts=True)
    duplicated_indices = indices[count != 1]  # list of vertices which are not unique

    # for each duplicated vertex, build the list of the corresponding identical vertices
    list_duplicated = []
    for idx in range(len(duplicated_indices)):
        same_vertices = np.argwhere(vertices == uniq_vertices[duplicated_indices[idx], :])
        # same_vertices is a ndarray of the form [[ind0, 0], [ind0, 1], [ind0, 2], [ind1, 0], [ind1, 1], [ind1, 2],...]
        list_duplicated.append(list(same_vertices[::3, 0]))

    # remove duplicates in vertices
    remove_vertices = [value for sublist in list_duplicated for value in sublist[1:]]
    vertices = np.delete(vertices, remove_vertices, axis=0)
    print(len(remove_vertices), 'duplicated vertices removed')

    # remove duplicated_vertices in faces
    for idx in range(len(list_duplicated)):
        temp_array = list_duplicated[idx]
        for idy in range(1, len(temp_array)):
            duplicated_value = temp_array[idy]
            faces[faces == duplicated_value] = temp_array[0]  # temp_array[0] is the unique value, others are duplicates

            # all indices above duplicated_value have to be decreased by 1 to keep the match with the number of vertices
            faces[faces > duplicated_value] = faces[faces > duplicated_value] - 1

            # update accordingly all indices above temp_array[idy]
            if debugging:
                print('temp_array before', temp_array)
                print('list_duplicated before', list_duplicated)
            temp_array = [(value-1) if value > duplicated_value else value for value in temp_array]
            list_duplicated = [[(value-1) if value > duplicated_value else value for value in sublist]
                               for sublist in list_duplicated]
            if debugging:
                print('temp_array after', temp_array)
                print('list_duplicated after', list_duplicated)

    # look for faces with 2 identical vertices (cannot define later a normal to these faces)
    remove_faces = []
    for idx in range(faces.shape[0]):
        if np.unique(faces[idx, :], axis=0).shape[0] != faces[idx, :].shape[0]:
            remove_faces.append(idx)
    faces = np.delete(faces, remove_faces, axis=0)
    print(len(remove_faces), 'faces with identical vertices removed')

    return vertices, faces


def surface_indices(surface, plane_indices, margin=3):
    """
    Crop surface around the plane with a certain margin, and find corresponding surface indices.

    :param surface: the 3D surface binary array
    :param plane_indices: tuple of 3 1D-arrays of plane indices
    :param margin: margin to include aroung plane indices, in pixels
    :return: 3*1D arrays of surface indices
    """
    if surface.ndim != 3:
        raise ValueError('Surface should be a 3D array')
    if len(plane_indices) != 3:
        raise ValueError('plane_indices should be a tuple of 3 1D-arrays')
    surf_indices = np.nonzero(surface[
                                 plane_indices[0].min() - margin:plane_indices[0].max() + margin,
                                 plane_indices[1].min() - margin:plane_indices[1].max() + margin,
                                 plane_indices[2].min() - margin:plane_indices[2].max() + margin])
    surf0 = surf_indices[0] + plane_indices[0].min() - margin  # add margin plane_indices[0].min() - margin
    surf1 = surf_indices[1] + plane_indices[1].min() - margin  # add margin plane_indices[1].min() - margin
    surf2 = surf_indices[2] + plane_indices[2].min() - margin  # add margin plane_indices[2].min() - margin
    return surf0, surf1, surf2


def stereographic_proj(normals, intensity, max_angle, savedir, voxel_size, projection_axis, min_distance=10,
                       background_south=-1000, background_north=-1000, save_txt=False, cmap=default_cmap,
                       planes_south=None, planes_north=None, plot_planes=True, scale='linear',
                       comment_fig='', debugging=False):
    """
    Detect facets in an object using a stereographic projection of normals to mesh triangles
     and watershed segmentation.

    :param normals: array of normals to mesh triangles (nb_normals rows x 3 columns)
    :param intensity: array of intensities (nb_normals rows x 1 column)
    :param max_angle: maximum angle in degree of the stereographic projection (should be larger than 90)
    :param savedir: directory for saving figures
    :param voxel_size: tuple of three numbers corresponding to the real-space voxel size in each dimension
    :param projection_axis: the projection is performed on a plane perpendicular to that axis (0, 1 or 2)
    :param min_distance: min_distance of corner_peaks()
    :param background_south: threshold for background determination in the projection from South
    :param background_north: threshold for background determination in the projection from North
    :param save_txt: if True, will save coordinates in a .txt file
    :param cmap: colormap used for plotting pole figures
    :param planes_south: dictionnary of crystallographic planes, e.g. {'111':angle_with_reflection}
    :param planes_north: dictionnary of crystallographic planes, e.g. {'111':angle_with_reflection}
    :param plot_planes: if True, will draw circles corresponding to crystallographic planes in the pole figure
    :param scale: 'linear' or 'log', scale for the colorbar of the plot
    :param comment_fig: string, comment for the filename when saving figures
    :param debugging: show plots for debugging
    :return: labels for each projection from South and North, one array for each projection from South and North,
     list of rows to remove
    """
    from scipy.interpolate import griddata
    from scipy import ndimage
    from skimage.feature import corner_peaks
    from skimage.morphology import watershed

    if comment_fig and comment_fig[-1] != '_':
        comment_fig = comment_fig + '_'
    radius_mean = 1  # normals are normalized
    stereo_center = 0  # COM of the weighted point density, where the projection plane intersects the reference axis
    # since the normals have their origin at 0, the projection plane is the equator and stereo_center=0

    # check normals for nan
    list_nan = np.argwhere(np.isnan(normals))
    normals = np.delete(normals, list_nan[::3, 0], axis=0)
    intensity = np.delete(intensity, list_nan[::3, 0], axis=0)

    # recalculate normals considering the anisotropy of voxel sizes (otherwise angles are wrong)
    # the stereographic projection is in reciprocal space, therefore we need to use the reciprocal voxel sizes
    iso_normals = np.copy(normals)
    iso_normals[:, 0] = iso_normals[:, 0] * 2 * np.pi / voxel_size[0]
    iso_normals[:, 1] = iso_normals[:, 1] * 2 * np.pi / voxel_size[1]
    iso_normals[:, 2] = iso_normals[:, 2] * 2 * np.pi / voxel_size[2]
    # normalize iso_normals
    iso_normals_length = np.sqrt(iso_normals[:, 0] ** 2 + iso_normals[:, 1] ** 2 + iso_normals[:, 2] ** 2)
    iso_normals = iso_normals / iso_normals_length[:, np.newaxis]

    # calculate the normalized Euclidian metric coordinates u and v from xyz
    stereo_proj, uv_labels = calc_stereoproj_facet(projection_axis=projection_axis, vectors=iso_normals,
                                                   radius_mean=radius_mean, stereo_center=stereo_center)
    # stereo_proj[:, 0] is the euclidian u_south, stereo_proj[:, 1] is the euclidian v_south
    # stereo_proj[:, 2] is the euclidian u_north, stereo_proj[:, 3] is the euclidian v_north

    # remove intensity where stereo_proj is infinite
    list_bad = np.argwhere(np.isinf(stereo_proj) | np.isnan(stereo_proj))  # elementwise or
    remove_row = list(set(list_bad[:, 0]))  # remove duplicated row indices
    print('remove_row indices (the stereographic projection is infinite or nan): ', remove_row, '\n')
    stereo_proj = np.delete(stereo_proj, remove_row, axis=0)
    intensity = np.delete(intensity, remove_row, axis=0)

    fig, _ = gu.contour_stereographic(euclidian_u=stereo_proj[:, 0], euclidian_v=stereo_proj[:, 1], color=intensity,
                                      radius_mean=radius_mean, planes=planes_south, max_angle=max_angle, scale=scale,
                                      title="Projection from\nSouth pole", plot_planes=plot_planes, uv_labels=uv_labels)
    fig.savefig(savedir + comment_fig + 'South pole_' + scale + '.png')
    fig, _ = gu.contour_stereographic(euclidian_u=stereo_proj[:, 2], euclidian_v=stereo_proj[:, 3], color=intensity,
                                      radius_mean=radius_mean, planes=planes_north, max_angle=max_angle, scale=scale,
                                      title="Projection from\nNorth pole", plot_planes=plot_planes, uv_labels=uv_labels)
    fig.savefig(savedir + comment_fig + 'North pole_' + scale + '.png')

    # regrid stereo_proj
    # stereo_proj[:, 0] is the euclidian u_south, stereo_proj[:, 1] is the euclidian v_south
    # stereo_proj[:, 2] is the euclidian u_north, stereo_proj[:, 3] is the euclidian v_north
    nb_points = 4 * max_angle + 1
    v_grid, u_grid = np.mgrid[-max_angle:max_angle:(nb_points*1j), -max_angle:max_angle:(nb_points*1j)]
    # v_grid changes vertically, u_grid horizontally
    nby, nbx = u_grid.shape
    density_south = griddata((stereo_proj[:, 0], stereo_proj[:, 1]), intensity, (u_grid, v_grid), method='linear')  # S
    density_north = griddata((stereo_proj[:, 2], stereo_proj[:, 3]), intensity, (u_grid, v_grid), method='linear')  # N
    density_south = density_south / density_south[density_south > 0].max() * 10000  # normalize for plotting
    density_north = density_north / density_north[density_north > 0].max() * 10000  # normalize for plotting

    if save_txt:
        # save metric coordinates in text file
        density_south[np.isnan(density_south)] = 0.0
        density_north[np.isnan(density_north)] = 0.0
        fichier = open(savedir + 'CDI_poles.dat', "w")
        for ii in range(len(v_grid)):
            for jj in range(len(u_grid)):
                fichier.write(str(v_grid[ii, 0]) + '\t' + str(u_grid[0, jj]) + '\t' +
                              str(density_south[ii, jj]) + '\t' + str(v_grid[ii, 0]) + '\t' +
                              str(u_grid[0, jj]) + '\t' + str(density_north[ii, jj]) + '\n')
        fichier.close()
        del intensity
        gc.collect()

    # inverse densities for watershed segmentation
    density_south = -1 * density_south
    density_north = -1 * density_north

    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(12, 9))
    img0 = ax0.scatter(u_grid, v_grid, c=density_south, cmap=cmap)
    ax0.set_xlim(-max_angle, max_angle)
    ax0.set_ylim(-max_angle, max_angle)
    ax0.axis('scaled')
    gu.colorbar(img0)
    ax0.set_title('KDE \nSouth pole')
    img1 = ax1.scatter(u_grid, v_grid, c=density_north, cmap=cmap)
    ax1.set_xlim(-max_angle, max_angle)
    ax1.set_ylim(-max_angle, max_angle)
    ax1.axis('scaled')
    gu.colorbar(img1)
    ax1.set_title('KDE \nNorth pole')
    fig.tight_layout()
    plt.pause(0.1)
    fig.waitforbuttonpress()
    plt.close(fig)
    
    # identification of local minima
    density_south[density_south > background_south] = 0  # define the background in the density of normals
    mask_south = np.copy(density_south)
    mask_south[mask_south != 0] = 1

    density_north[density_north > background_north] = 0  # define the background in the density of normals
    mask_north = np.copy(density_north)
    mask_north[mask_north != 0] = 1

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2, figsize=(12, 9))
    ax0.imshow(mask_south, cmap=cmap, interpolation='nearest')
    ax0.set_title('Background mask South')
    ax0.invert_yaxis()
    img1 = ax1.scatter(u_grid, v_grid, c=density_south, cmap=cmap)
    ax1.set_xlim(-max_angle, max_angle)
    ax1.set_ylim(-max_angle, max_angle)
    ax1.axis('scaled')
    gu.colorbar(img1)
    ax1.set_title('KDE South pole\nafter background definition')
    circle = patches.Circle((0, 0), 90, color='w', fill=False, linewidth=1.5)
    ax1.add_artist(circle)
    ax2.imshow(mask_north, cmap=cmap, interpolation='nearest')
    ax2.set_title('Background mask North')
    ax2.invert_yaxis()
    img3 = ax3.scatter(u_grid, v_grid, c=density_north, cmap=cmap)
    ax3.set_xlim(-max_angle, max_angle)
    ax3.set_ylim(-max_angle, max_angle)
    ax3.axis('scaled')
    gu.colorbar(img3)
    ax3.set_title('KDE North pole\nafter background definition')
    circle = patches.Circle((0, 0), 90, color='w', fill=False, linewidth=1.5)
    ax3.add_artist(circle)
    fig.tight_layout()
    plt.pause(0.1)

    ##########################################################################
    # Generate the markers as local maxima of the distance to the background #
    ##########################################################################
    distances_south = ndimage.distance_transform_edt(density_south)
    distances_north = ndimage.distance_transform_edt(density_north)
    if debugging:
        fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
        img0 = ax0.imshow(distances_south, cmap=cmap, interpolation='nearest')
        ax0.set_title('Distances South')
        gu.colorbar(img0)
        ax0.invert_yaxis()
        img1 = ax1.imshow(distances_north, cmap=cmap, interpolation='nearest')
        ax1.set_title('Distances North')
        gu.colorbar(img1)
        ax1.invert_yaxis()
        fig.tight_layout()
        plt.pause(0.1)

    local_maxi_south = corner_peaks(distances_south, exclude_border=False, min_distance=min_distance, indices=False)
    local_maxi_north = corner_peaks(distances_north, exclude_border=False, min_distance=min_distance, indices=False)
    if debugging:
        fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
        ax0.imshow(local_maxi_south, interpolation='nearest')
        ax0.set_title('local_maxi South before filtering')
        ax0.invert_yaxis()
        circle = patches.Ellipse((nbx // 2, nby // 2), 361, 361, color='r', fill=False, linewidth=1.5)
        ax0.add_artist(circle)
        ax1.imshow(local_maxi_north, interpolation='nearest')
        ax1.set_title('local_maxi North before filtering')
        ax1.invert_yaxis()
        circle = patches.Ellipse((nbx // 2, nby // 2), 361, 361, color='r', fill=False, linewidth=1.5)
        ax1.add_artist(circle)
        fig.tight_layout()
        plt.pause(0.1)

    # define the marker for each peak
    markers_south = ndimage.label(local_maxi_south)[0]  # range from 0 to nb_peaks
    # define non overlaping markers for the North projection: the first marker value is (markers_south.max()+1)
    markers_north = ndimage.label(local_maxi_north)[0] + markers_south.max(initial=None)
    # markers_north.min() is 0 since it is the background
    markers_north[markers_north == markers_south.max(initial=None)] = 0
    if debugging:
        fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
        ax0.imshow(markers_south, interpolation='nearest', cmap='binary', vmin=0, vmax=1)
        ax0.set_title('markers South')
        ax0.invert_yaxis()
        circle = patches.Ellipse((nbx // 2, nby // 2), 361, 361, color='r', fill=False, linewidth=1.5)
        ax0.add_artist(circle)
        ax1.imshow(markers_north, interpolation='nearest', cmap='binary', vmin=0, vmax=1)
        ax1.set_title('markers North')
        ax1.invert_yaxis()
        circle = patches.Ellipse((nbx // 2, nby // 2), 361, 361, color='r', fill=False, linewidth=1.5)
        ax1.add_artist(circle)
        fig.tight_layout()
        plt.pause(0.1)

    ##########################
    # watershed segmentation #
    ##########################
    labels_south = watershed(-distances_south, markers_south, mask=mask_south)
    labels_north = watershed(-distances_north, markers_north, mask=mask_north)
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(12, 9))
    ax0.imshow(labels_south, cmap=cmap, interpolation='nearest')
    ax0.set_title('Separated objects South')
    ax0.invert_yaxis()
    circle = patches.Ellipse((nbx // 2, nby // 2), 361, 361, color='r', fill=False, linewidth=1.5)
    ax0.add_artist(circle)
    ax1.imshow(labels_north, cmap=cmap, interpolation='nearest')
    ax1.set_title('Separated objects North')
    ax1.invert_yaxis()
    circle = patches.Ellipse((nbx // 2, nby // 2), 361, 361, color='r', fill=False, linewidth=1.5)
    ax1.add_artist(circle)
    fig.tight_layout()
    plt.pause(0.1)

    return labels_south, labels_north, stereo_proj, remove_row


def taubin_smooth(faces, vertices, cmap=default_cmap, iterations=10, lamda=0.33, mu=0.34, radius=0.1,
                  debugging=False):
    """
    Performs a back and forward Laplacian smoothing "without shrinking" of a triangulated mesh,
     as described by Gabriel Taubin (ICCV '95)

    :param faces: m*3 ndarray of m faces defined by 3 indices of vertices
    :param vertices: n*3 ndarray of n vertices defined by 3 positions
    :param cmap: colormap used for plotting
    :param iterations: number of iterations for smoothing
    :param lamda: smoothing variable 0 < lambda < mu < 1
    :param mu: smoothing variable 0 < lambda < mu < 1
    :param radius: radius around which the normals are integrated in the calculation of the density of normals
    :param debugging: show plots for debugging
    :return: smoothened vertices (ndarray n*3), normals to triangle (ndarray m*3), weighted density of normals,
     updated faces, errors
    """
    from mpl_toolkits.mplot3d import Axes3D
    plt.ion()

    print('Original number of vertices:', vertices.shape[0])
    print('Original number of faces:', faces.shape[0])
    new_vertices = np.copy(vertices)

    for k in range(iterations):
        # check the unicity of vertices otherwise 0 distance would happen
        if np.unique(new_vertices, axis=0).shape[0] != new_vertices.shape[0]:
            print('\nTaubin smoothing / lambda: duplicated vertices at iteration', k)
            new_vertices, faces = remove_duplicates(vertices=new_vertices, faces=faces)
        vertices = np.copy(new_vertices)
        neighbours = find_neighbours(vertices, faces)  # get the indices of neighboring vertices for each vertex
        indices_edges = detect_edges(faces)  # find indices of vertices defining non-shared edges (near hole...)

        for i in range(vertices.shape[0]):
            indices = neighbours[i]  # list of indices
            distances = np.sqrt(np.sum((vertices[indices, :]-vertices[i, :]) ** 2, axis=1))
            weights = distances**(-1)
            vectoren = weights[:, np.newaxis] * vertices[indices, :]
            totaldist = sum(weights)
            new_vertices[i, :] = vertices[i, :] + lamda*(np.sum(vectoren, axis=0)/totaldist-vertices[i, :])

        if indices_edges.size != 0:
            new_vertices[indices_edges, :] = vertices[indices_edges, :]

        # check the unicity of vertices otherwise 0 distance would happen
        if np.unique(new_vertices, axis=0).shape[0] != new_vertices.shape[0]:
            print('\nTaubin smoothing / mu: duplicated vertices at iteration', k)
            new_vertices, faces = remove_duplicates(vertices=new_vertices, faces=faces)
        vertices = np.copy(new_vertices)
        neighbours = find_neighbours(vertices, faces)  # get the indices of neighboring vertices for each vertex
        indices_edges = detect_edges(faces)  # find indices of vertices defining non-shared edges (near hole...)

        for i in range(vertices.shape[0]):

            indices = neighbours[i]  # list of indices
            distances = np.sqrt(np.sum((vertices[indices, :]-vertices[i, :])**2, axis=1))
            weights = distances**(-1)
            vectoren = weights[:, np.newaxis] * vertices[indices, :]
            totaldist = sum(weights)
            new_vertices[i, :] = vertices[i, :] - mu*(sum(vectoren)/totaldist - vertices[i, :])

        if indices_edges.size != 0:
            new_vertices[indices_edges, :] = vertices[indices_edges, :]

    # check the unicity of vertices otherwise 0 distance would happen
    if np.unique(new_vertices, axis=0).shape[0] != new_vertices.shape[0]:
        print('\nTaubin smoothing / exiting loop: duplicated vertices')
        new_vertices, faces = remove_duplicates(vertices=new_vertices, faces=faces)

    nan_vertices = np.argwhere(np.isnan(new_vertices[:, 0]))
    print('Number of nan in new_vertices:', nan_vertices.shape[0], '; Total number of vertices:', new_vertices.shape[0])

    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = new_vertices[faces]
    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0,
    # and v2-v0 in each triangle
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[::, 0])
    areas = np.array([1/2 * np.linalg.norm(normal) for normal in normals])
    normals_length = np.sqrt(normals[:, 0]**2 + normals[:, 1]**2 + normals[:, 2]**2)
    normals = -1 * normals / normals_length[:, np.newaxis]   # flip and normalize normals
    # n is now an array of normalized normals, one per triangle.

    # calculate the colormap for plotting the weighted point density of normals on a sphere
    intensity = np.zeros(normals.shape[0], dtype=normals.dtype)
    for i in range(normals.shape[0]):
        distances = np.sqrt(np.sum((normals - normals[i, :]) ** 2, axis=1))  # ndarray of  normals.shape[0]
        intensity[i] = np.multiply(areas[distances < radius], distances[distances < radius]).sum()
        # normals are weighted by the area of mesh triangles

    intensity = intensity / max(intensity)
    if debugging:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(normals[:, 0], normals[:, 1], normals[:, 2], c=intensity, cmap=cmap)
        ax.set_xlim(-1, 1)
        ax.set_xlabel('z')
        ax.set_ylim(-1, 1)
        ax.set_ylabel('y')
        ax.set_zlim(-1, 1)
        ax.set_zlabel('x')
        plt.title('Weighted point densities before KDE')
        plt.pause(0.1)
    err_normals = np.argwhere(np.isnan(normals[:, 0]))
    normals[err_normals, :] = normals[err_normals-1, :]
    plt.ioff()

    # check normals for nan
    list_nan = np.argwhere(np.isnan(normals))
    normals = np.delete(normals, list_nan[::3, 0], axis=0)
    intensity = np.delete(intensity, list_nan[::3, 0], axis=0)

    return new_vertices, normals, areas, intensity, faces, err_normals


def update_logfile(support, strain_array, summary_file, allpoints_file, label=0, angle_plane=0, plane_coeffs=(0, 0, 0),
                   plane_normal=(0, 0, 0), top_part=False, z_cutoff=0):
    """
    Update log files use in the facet_strain.py script.

    :param support: the 3D binary support defining voxels to be saved in the logfile
    :param strain_array: the 3D strain array
    :param summary_file: the handle for the file summarizing strain statistics per facet
    :param allpoints_file: the handle for the file giving the strain and the label for each voxel
    :param label: the label of the plane
    :param angle_plane: the angle of the plane with the measurement direction
    :param plane_coeffs: the fit coefficients of the plane
    :param plane_normal: the normal to the plane
    :param top_part: if True, it will save values only for the top part of the nanoparticle
    :param z_cutoff: if top_pat=True, will set all support points below this value to 0
    :return: nothing
    """
    if (support.ndim != 3) or (strain_array.ndim != 3):
        raise ValueError('The support and the strain arrays should be 3D arrays')

    support_indices = np.nonzero(support == 1)
    ind_z = support_indices[0]
    ind_y = support_indices[1]
    ind_x = support_indices[2]
    nb_points = len(support_indices[0])
    for idx in range(nb_points):
        if strain_array[ind_z[idx], ind_y[idx], ind_x[idx]] != 0:
            # remove the artefact from YY reconstrutions at the bottom facet
            allpoints_file.write('{0: <10}'.format(str(label)) + '\t' +
                                 '{0: <10}'.format(str(ind_z[idx])) + '\t' +
                                 '{0: <10}'.format(str(ind_y[idx])) + '\t' +
                                 '{0: <10}'.format(str(ind_x[idx])) + '\t' +
                                 '{0: <10}'.format(str('{:.7f}'.format(strain_array[ind_z[idx],
                                                                                    ind_y[idx],
                                                                                    ind_x[idx]])))
                                 + '\n')

    str_array = strain_array[support == 1]
    str_array[str_array == 0] = np.nan  # remove the artefact from YY reconstrutions at the bottom facet
    support_strain = np.mean(str_array[~np.isnan(str_array)])
    support_deviation = np.std(str_array[~np.isnan(str_array)])

    # support_strain = np.mean(strain_array[support == 1])
    # support_deviation = np.std(strain_array[support == 1])
    summary_file.write('{0: <10}'.format(str(label)) + '\t' +
                       '{0: <10}'.format(str('{:.3f}'.format(angle_plane))) + '\t' +
                       '{0: <10}'.format(str(nb_points)) + '\t' +
                       '{0: <10}'.format(str('{:.7f}'.format(support_strain))) + '\t' +
                       '{0: <10}'.format(str('{:.7f}'.format(support_deviation))) + '\t' +
                       '{0: <10}'.format(str('{:.5f}'.format(plane_coeffs[0]))) + '\t' +
                       '{0: <10}'.format(str('{:.5f}'.format(plane_coeffs[1]))) + '\t' +
                       '{0: <10}'.format(str('{:.5f}'.format(plane_coeffs[2]))) + '\t' +
                       '{0: <10}'.format(str('{:.5f}'.format(plane_normal[0]))) + '\t' +
                       '{0: <10}'.format(str('{:.5f}'.format(plane_normal[1]))) + '\t' +
                       '{0: <10}'.format(str('{:.5f}'.format(plane_normal[2]))) + '\n')

    if top_part:
        new_support = np.copy(support)  # support is mutable, need to make a copy
        new_support[:, :, :z_cutoff] = 0
        new_label = str(label) + '_top'
        support_indices = np.nonzero(new_support == 1)
        ind_z = support_indices[0]
        ind_y = support_indices[1]
        ind_x = support_indices[2]
        nb_points = len(support_indices[0])
        for idx in range(nb_points):
            if strain_array[ind_z[idx], ind_y[idx], ind_x[idx]] != 0:
                # remove the artefact from YY reconstrutions at the bottom facet
                allpoints_file.write('{0: <10}'.format(new_label) + '\t' +
                                     '{0: <10}'.format(str(ind_z[idx])) + '\t' +
                                     '{0: <10}'.format(str(ind_y[idx])) + '\t' +
                                     '{0: <10}'.format(str(ind_x[idx])) + '\t' +
                                     '{0: <10}'.format(str('{:.7f}'.format(strain_array[ind_z[idx],
                                                                                        ind_y[idx],
                                                                                        ind_x[idx]])))
                                     + '\n')

        str_array = strain_array[new_support == 1]
        str_array[str_array == 0] = np.nan  # remove the artefact from YY reconstrutions at the bottom facet
        support_strain = np.mean(str_array[~np.isnan(str_array)])
        support_deviation = np.std(str_array[~np.isnan(str_array)])

        # support_strain = np.mean(strain_array[support == 1])
        # support_deviation = np.std(strain_array[support == 1])
        summary_file.write('{0: <10}'.format(new_label) + '\t' +
                           '{0: <10}'.format(str('{:.3f}'.format(angle_plane))) + '\t' +
                           '{0: <10}'.format(str(nb_points)) + '\t' +
                           '{0: <10}'.format(str('{:.7f}'.format(support_strain))) + '\t' +
                           '{0: <10}'.format(str('{:.7f}'.format(support_deviation))) + '\t' +
                           '{0: <10}'.format(str('{:.5f}'.format(plane_coeffs[0]))) + '\t' +
                           '{0: <10}'.format(str('{:.5f}'.format(plane_coeffs[1]))) + '\t' +
                           '{0: <10}'.format(str('{:.5f}'.format(plane_coeffs[2]))) + '\t' +
                           '{0: <10}'.format(str('{:.5f}'.format(plane_normal[0]))) + '\t' +
                           '{0: <10}'.format(str('{:.5f}'.format(plane_normal[1]))) + '\t' +
                           '{0: <10}'.format(str('{:.5f}'.format(plane_normal[2]))) + '\n')


def upsample(array, upsampling_factor, voxelsizes, title='', debugging=False):
    """
    Upsample array using a factor of upsampling.

    :param array: the real array to be upsampled
    :param upsampling_factor: int, the upsampling factor
    :param voxelsizes: list, the voxel sizes of array
    :param title: title for the debugging plot
    :param debugging: True to see plots
    :return: the upsampled array
    """
    from scipy.interpolate import RegularGridInterpolator

    if array.ndim != 3:
        raise ValueError('Expecting a 3D array as input')

    if not isinstance(upsampling_factor, int):
        raise ValueError('upsampling_factor should be an integer')

    vmin, vmax = array.min(), array.max()

    if debugging:
        gu.multislices_plot(array, sum_frames=False, title=title+' before upsampling', vmin=vmin, vmax=vmax,
                            scale='linear', plot_colorbar=True, reciprocal_space=False, is_orthogonal=True)
    nbz, nby, nbx = array.shape
    numz, numy, numx = nbz * upsampling_factor, nby * upsampling_factor, nbx * upsampling_factor
    newvoxelsizes = [voxsize/upsampling_factor for voxsize in voxelsizes]

    newz, newy, newx = np.meshgrid(np.arange(-numz // 2, numz // 2, 1) * newvoxelsizes[0],
                                   np.arange(-numy // 2, numy // 2, 1) * newvoxelsizes[1],
                                   np.arange(-numx // 2, numx // 2, 1) * newvoxelsizes[2], indexing='ij')

    rgi = RegularGridInterpolator(
        (np.arange(-nbz // 2, nbz // 2)*voxelsizes[0],
         np.arange(-nby // 2, nby // 2)*voxelsizes[1],
         np.arange(-nbx // 2, nbx // 2)*voxelsizes[2]),
        array, method='linear', bounds_error=False, fill_value=0)

    obj = rgi(np.concatenate((newz.reshape((1, newz.size)), newy.reshape((1, newz.size)),
                              newx.reshape((1, newz.size)))).transpose())

    obj = obj.reshape((numz, numy, numx)).astype(array.dtype)

    if debugging:
        gu.multislices_plot(obj, sum_frames=False, title=title+' after upsampling', vmin=vmin, vmax=vmax,
                            scale='linear', plot_colorbar=True, reciprocal_space=False, is_orthogonal=True)

    return obj, newvoxelsizes


# if __name__ == "__main__":
