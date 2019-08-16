# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patches
import gc
import sys
sys.path.append('//win.desy.de/home/carnisj/My Documents/myscripts/bcdi/')
import bcdi.graph.graph_utils as gu

colormap = gu.Colormap()
default_cmap = colormap.cmap


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
    plane_normal = np.array([fit[0, 0], fit[1, 0], -1])  # normal is [a, b, c] if ax+by+cz+d=0
    for point in range(len(indices[0])):
        dist = abs(fit[0, 0]*indx[point] + fit[1, 0]*indy[point] -
                   indz[point] + fit[2, 0])/np.linalg.norm(plane_normal)
        if dist < max_distance:
            plane[indx[point], indy[point], indz[point]] = 1
    if plane[plane == 1].sum() == 0:
        print('Distance_threshold: no points for plane')
        no_points = 1
        return plane, no_points
    return plane, no_points


def equirectangular_proj(normals, color, weights, cmap=default_cmap, bw_method=0.03, min_distance=10,
                         background_threshold=-0.35, debugging=False):
    """
    Detect facets in an object using an equirectangular projection of normals to mesh triangles
     and watershed segmentation.

    :param normals: normals array
    :param color: intensity array
    :param weights: weights used in the gaussian kernel density estimation
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
    if len(list_nan) != 0:
        for i in range(list_nan.shape[0]//3):
            normals = np.delete(normals, list_nan[i*3, 0], axis=0)
            color = np.delete(color, list_nan[i*3, 0], axis=0)
    # calculate latitude and longitude from xyz, this is equal to the equirectangular flat square projection
    long_lat = np.zeros((normals.shape[0], 2), dtype=normals.dtype)
    for i in range(normals.shape[0]):
        if normals[i, 1] == 0 and normals[i, 0] == 0:
            continue
        long_lat[i, 0] = np.arctan2(normals[i, 1], normals[i, 0])  # longitude
        long_lat[i, 1] = np.arcsin(normals[i, 2])  # latitude
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(long_lat[:, 0], long_lat[:, 1], c=color, cmap=cmap)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi / 2, np.pi / 2)
    plt.axis('scaled')
    plt.title('Equirectangular projection of the weighted point densities before KDE')
    plt.pause(0.1)

    del color
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
    neighbors = [point for point in neighbors if point is not None]
    for indx in range(vertices.shape[0]):
        neighbors[indx] = list(set(neighbors[indx]))  # remove redundant indices in each sublist
    return neighbors


def fit_plane(plane, label, debugging=1):
    """
    Fit a plane to labelled indices, ax+by+c=z

    :param plane: 3D binary array of the shape of the data
    :param label: int, only used for title in plot
    :param debugging: show plots for debugging
    :return: matrix of fit parameters [a, b, c], plane indices, errors associated, a stop flag
    """
    from scipy.ndimage.measurements import center_of_mass

    indices = np.nonzero(plane)
    no_points = 0
    if len(indices[0]) == 0:
        no_points = 1
        return 0, indices, 0, no_points
    tmp_x = indices[0]
    tmp_y = indices[1]
    tmp_z = indices[2]
    x_com, y_com, z_com = center_of_mass(plane)

    # remove isolated points, which probably do not belong to the plane
    for point in range(tmp_x.shape[0]):
        neighbors = plane[tmp_x[point]-2:tmp_x[point]+3, tmp_y[point]-2:tmp_y[point]+3,
                          tmp_z[point]-2:tmp_z[point]+3].sum()
        if neighbors < 5:
            plane[tmp_x[point], tmp_y[point], tmp_z[point]] = 0
    print('Plane', label, ', ', str(tmp_x.shape[0]-plane[plane == 1].sum()), 'points isolated, ',
          str(plane[plane == 1].sum()), 'remaining')

    # update plane indices
    indices = np.nonzero(plane)
    if len(indices[0]) == 0:
        no_points = 1
        return 0, indices, 0, no_points
    tmp_x = indices[0]
    tmp_y = indices[1]
    tmp_z = indices[2]

    # remove also points farther than 2 times the mean distance to the COM
    dist = np.zeros(tmp_x.shape[0])
    for point in range(tmp_x.shape[0]):
        dist[point] = np.sqrt((tmp_x[point]-x_com)**2+(tmp_y[point]-y_com)**2+(tmp_z[point]-z_com)**2)
    average_dist = np.mean(dist)
    for point in range(tmp_x.shape[0]):
        if dist[point] > 2 * average_dist:
            plane[tmp_x[point], tmp_y[point], tmp_z[point]] = 0
    print('Plane', label, ', ', str(tmp_x.shape[0] - plane[plane == 1].sum()), 'points too far from COM, ',
          str(plane[plane == 1].sum()), 'remaining')

    # update plane indices and check if enough points remain
    indices = np.nonzero(plane)
    if len(indices[0]) < 5:
        no_points = 1
        return 0, indices, 0, no_points
    tmp_x = indices[0]
    tmp_y = indices[1]
    tmp_z = indices[2]

    tmp_x = tmp_x[:, np.newaxis]
    tmp_y = tmp_y[:, np.newaxis]
    tmp_1 = np.ones((tmp_y.shape[0], 1))

    # calculate plane parameters
    # TODO: update this part (numpy.matrix will not be supported anymore in the future)
    a = np.matrix(np.concatenate((tmp_x, tmp_y, tmp_1), axis=1))
    b = np.matrix(tmp_z).T
    fit = (a.T * a).I * a.T * b
    errors = b - a * fit

    if debugging:
        plt.figure()
        ax = plt.subplot(111, projection='3d')
        ax.scatter(tmp_x, tmp_y, tmp_z, color='b')
        ax.set_xlabel('x')  # first dimension is x for plots, but z for NEXUS convention
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        xlim = ax.get_xlim()  # first dimension is x for plots, but z for NEXUS convention
        ylim = ax.get_ylim()
        meshx, meshy = np.meshgrid(np.arange(xlim[0], xlim[1]+1, 1), np.arange(ylim[0], ylim[1]+1, 1))
        meshz = np.zeros(meshx.shape)
        for row in range(meshx.shape[0]):
            for col in range(meshx.shape[1]):
                meshz[row, col] = fit[0, 0] * meshx[row, col] +\
                                      fit[1, 0] * meshy[row, col] + fit[2, 0]
        ax.plot_wireframe(meshx, meshy, meshz, color='k')
        plt.title("Points and fitted plane" + str(label))
        plt.pause(0.1)
    return fit, indices, errors, no_points


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

    start_z = min(indices[0].min()-20, 0)
    stop_z = min(indices[0].max()+21, nbz)
    start_y = min(indices[1].min()-20, 0)
    stop_y = min(indices[1].max()+21, nby)
    start_x = min(indices[2].min()-20, 0)
    stop_x = min(indices[2].max()+21, nbx)

    # find nearby voxels using the coordination number
    obj = np.copy(plane[start_z:stop_z, start_y:stop_y, start_x: stop_x])
    coord = np.rint(convolve(obj, kernel, mode='same'))
    coord = coord.astype(int)
    coord[coord < 1] = 0

    # update plane with new voxels
    temp_plane = np.copy(plane)
    temp_plane[start_z:stop_z, start_y:stop_y, start_x: stop_x] = coord
    # remove voxels not belonging to the support
    temp_plane[support == 0] = 0
    # check distance of new voxels to the plane
    new_indices = np.nonzero(temp_plane)

    temp_plane, no_points = distance_threshold(fit=fit, indices=new_indices, shape=temp_plane.shape,
                                               max_distance=max_distance)

    # update indices and plane voxels
    new_indices = np.nonzero(temp_plane)
    plane[new_indices[0], new_indices[1], new_indices[2]] = 1

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


def plane_angle_cubic(ref_plane, plane):
    """
    Calculate the angle between two crystallographic planes in cubic materials

    :param ref_plane: measured reflection
    :param plane: plane for which angle should be calculated
    :return: the angle in degrees
    """
    if np.array_equal(ref_plane, plane):
        angle = 0.0
    else:
        angle = 180 / np.pi * np.arccos(sum(np.multiply(ref_plane, plane)) /
                                        (np.linalg.norm(ref_plane) * np.linalg.norm(plane)))
    if angle > 90.0:
        angle = 180.0 - angle
    return angle


def stereographic_proj(normals, color, weights, max_angle, savedir, min_distance=10, background_threshold=-1000,
                       save_txt=False, cmap=default_cmap, planes={}, plot_planes=True, debugging=False):
    """
    Detect facets in an object using a stereographic projection of normals to mesh triangles
     and watershed segmentation.

    :param normals: array of normals (nb_normals rows x 3 columns)
    :param color: array of intensities (nb_normals rows x 1 column)
    :param weights: weights used in the density estimation
    :param max_angle: maximum angle in degree of the stereographic projection (should be larger than 90)
    :param savedir: directory for saving figures
    :param min_distance: min_distance of corner_peaks()
    :param background_threshold: threshold for background determination (depth of the KDE)
    :param save_txt: if True, will save coordinates in a .txt file
    :param cmap: colormap used for plotting pole figures
    :param planes: dictionnary of crystallographic planes, e.g. {'111':angle_with_reflection}
    :param plot_planes: if True, will draw circles corresponding to crystallographic planes in the pole figure
    :param debugging: show plots for debugging
    :return:
    """
    from scipy.interpolate import griddata
    from scipy import ndimage
    from skimage.feature import corner_peaks
    from skimage.morphology import watershed

    # check normals for nan
    radius_mean = 1  # normals are normalized
    stereo_centerz = 0  # COM of the weighted point density
    list_nan = np.argwhere(np.isnan(normals))
    if len(list_nan) != 0:
        for i in range(list_nan.shape[0]//3):
            normals = np.delete(normals, list_nan[i*3, 0], axis=0)
            color = np.delete(color, list_nan[i*3, 0], axis=0)

    # calculate u and v from xyz, this is equal to the stereographic projection from South pole
    stereo_proj = np.zeros((normals.shape[0], 4), dtype=normals.dtype)
    for i in range(normals.shape[0]):
        if normals[i, 1] == 0 and normals[i, 0] == 0:
            continue
        stereo_proj[i, 0] = radius_mean * normals[i, 0] / (radius_mean+normals[i, 1] - stereo_centerz)  # u_top qx
        stereo_proj[i, 1] = radius_mean * normals[i, 2] / (radius_mean+normals[i, 1] - stereo_centerz)  # v_top qy
        stereo_proj[i, 2] = radius_mean * normals[i, 0] / (stereo_centerz - radius_mean+normals[i, 1])  # u_bottom qx
        stereo_proj[i, 3] = radius_mean * normals[i, 2] / (stereo_centerz - radius_mean+normals[i, 1])  # v_bottom qy
    stereo_proj = stereo_proj / radius_mean * 90  # rescaling from radius_mean to 90

    if debugging:
        fig, _ = gu.plot_stereographic(euclidian_u=stereo_proj[:, 0], euclidian_v=stereo_proj[:, 1], color=color,
                                       radius_mean=radius_mean, planes=planes, title="South pole",
                                       plot_planes=plot_planes)
        fig.savefig(savedir + 'South pole.png')
        fig, _ = gu.plot_stereographic(euclidian_u=stereo_proj[:, 2], euclidian_v=stereo_proj[:, 3], color=color,
                                       radius_mean=radius_mean, planes=planes, title="North pole",
                                       plot_planes=plot_planes)
        fig.savefig(savedir + 'North pole.png')

    yi, xi = np.mgrid[-max_angle:max_angle:381j, -max_angle:max_angle:381j]  # vertical, horizontal
    density_top = griddata((stereo_proj[:, 0], stereo_proj[:, 1]), color, (yi, xi), method='linear')
    density_bottom = griddata((stereo_proj[:, 2], stereo_proj[:, 3]), color, (yi, xi), method='linear')
    density_top = density_top / density_top[density_top > 0].max() * 10000  # normalize for plotting
    density_bottom = density_bottom / density_bottom[density_bottom > 0].max() * 10000  # normalize for plotting

    if save_txt:
        # save metric coordinates in text file
        density_top[np.isnan(density_top)] = 0.0
        density_bottom[np.isnan(density_bottom)] = 0.0
        fichier = open(savedir + 'CDI_poles.dat', "w")
        for ii in range(len(yi)):
            for jj in range(len(xi)):
                fichier.write(str(yi[ii, 0]) + '\t' + str(xi[0, jj]) + '\t' +
                              str(density_top[ii, jj]) + '\t' + str(yi[ii, 0]) + '\t' +
                              str(xi[0, jj]) + '\t' + str(density_bottom[ii, jj]) + '\n')
        fichier.close()
        del color
        gc.collect()

    # inverse densities for watershed segmentation
    density_top = -1 * density_top
    density_bottom = -1 * density_bottom

    fig = plt.figure(figsize=(15, 10))
    ax0 = fig.add_subplot(121)
    scatter_top = ax0.scatter(xi, yi, c=density_top, cmap=cmap)
    ax0.set_xlim(-max_angle, max_angle)
    ax0.set_ylim(-max_angle, max_angle)
    fig.colorbar(scatter_top)
    plt.axis('scaled')
    plt.title('KDE \nSouth pole')
    plt.pause(0.1)

    ax1 = fig.add_subplot(122)
    scatter_bottom = ax1.scatter(xi, yi, c=density_bottom, cmap=cmap)
    ax1.set_xlim(-max_angle, max_angle)
    ax1.set_ylim(-max_angle, max_angle)
    fig.colorbar(scatter_bottom)
    plt.axis('scaled')
    plt.title('KDE \nNorth pole')
    plt.pause(0.1)

    # identification of local minima
    density_top[density_top > background_threshold] = 0  # define the background
    mask_top = np.copy(density_top)
    mask_top[mask_top != 0] = 1

    density_bottom[density_bottom > background_threshold] = 0  # define the background
    mask_bottom = np.copy(density_bottom)
    mask_bottom[mask_bottom != 0] = 1

    fig = plt.figure(figsize=(15, 10))
    ax0 = fig.add_subplot(221)
    ax0.imshow(mask_top, cmap=cmap, interpolation='nearest')
    plt.title('Background mask South')
    plt.gca().invert_yaxis()

    ax1 = fig.add_subplot(223)
    scatter_top = ax1.scatter(xi, yi, c=density_top, cmap=cmap)
    ax1.set_xlim(-max_angle, max_angle)
    ax1.set_ylim(-max_angle, max_angle)
    fig.colorbar(scatter_top)
    plt.axis('scaled')
    plt.title('KDE South pole\nafter background definition')
    circle = patches.Circle((0, 0), 90, color='w', fill=False, linewidth=1.5)
    ax1.add_artist(circle)

    ax2 = fig.add_subplot(222)
    ax2.imshow(mask_bottom, cmap=cmap, interpolation='nearest')
    plt.title('Background mask North')
    plt.gca().invert_yaxis()

    ax3 = fig.add_subplot(224)
    scatter_bottom = ax3.scatter(xi, yi, c=density_bottom, cmap=cmap)
    ax3.set_xlim(-max_angle, max_angle)
    ax3.set_ylim(-max_angle, max_angle)
    fig.colorbar(scatter_bottom)
    plt.axis('scaled')
    plt.title('KDE North pole\nafter background definition')
    circle = patches.Circle((0, 0), 90, color='w', fill=False, linewidth=1.5)
    ax3.add_artist(circle)
    plt.pause(0.1)

    # Generate the markers as local minima of the distance to the background
    distances_top = ndimage.distance_transform_edt(density_top)
    distances_bottom = ndimage.distance_transform_edt(density_bottom)

    if debugging:
        fig = plt.figure(figsize=(15, 10))
        fig.add_subplot(121)
        plt.imshow(distances_top, cmap=cmap, interpolation='nearest')
        plt.title('Distances South')
        plt.gca().invert_yaxis()
        fig.add_subplot(122)
        plt.imshow(distances_bottom, cmap=cmap, interpolation='nearest')
        plt.title('Distances North')
        plt.gca().invert_yaxis()
        plt.pause(0.1)

    # find peaks
    local_maxi_top = corner_peaks(distances_top, exclude_border=False, min_distance=min_distance, indices=False)
    local_maxi_bottom = corner_peaks(distances_bottom, exclude_border=False, min_distance=min_distance, indices=False)

    if debugging:
        fig = plt.figure(figsize=(15, 10))
        fig.add_subplot(121)
        plt.imshow(local_maxi_top, interpolation='nearest')
        plt.title('local_maxi South')
        plt.gca().invert_yaxis()
        fig.add_subplot(122)
        plt.imshow(local_maxi_bottom, interpolation='nearest')
        plt.title('local_maxi North')
        plt.gca().invert_yaxis()
        plt.pause(0.1)

    # define markers for each peak
    markers_top = ndimage.label(local_maxi_top)[0]  # range from 0 to nb_peaks
    # define non overlaping markers for bottom projection
    markers_bottom = ndimage.label(local_maxi_bottom)[0] + markers_top.max()
    # markers_bottom.min() should be 0 since it is the background
    markers_bottom[markers_bottom == markers_top.max()] = 0

    if debugging:
        fig = plt.figure(figsize=(15, 10))
        fig.add_subplot(121)
        plt.imshow(markers_top, interpolation='nearest')
        plt.title('markers South')
        plt.gca().invert_yaxis()
        fig.add_subplot(122)
        plt.imshow(markers_bottom, interpolation='nearest')
        plt.title('markers North')
        plt.gca().invert_yaxis()
        plt.pause(0.1)

    # watershed segmentation
    labels_top = watershed(-distances_top, markers_top, mask=mask_top)
    labels_bottom = watershed(-markers_bottom, markers_bottom, mask=mask_bottom)
    nby, nbx = xi.shape
    fig = plt.figure(figsize=(15, 10))
    ax0 = fig.add_subplot(121)
    plt.imshow(labels_top, cmap=cmap, interpolation='nearest')
    plt.title('Separated objects South')
    plt.gca().invert_yaxis()
    circle = patches.Ellipse((nbx // 2, nby // 2), 361, 361, color='r', fill=False, linewidth=1.5)
    ax0.add_artist(circle)
    ax1 = fig.add_subplot(122)
    plt.imshow(labels_bottom, cmap=cmap, interpolation='nearest')
    plt.title('Separated objects North')
    plt.gca().invert_yaxis()
    circle = patches.Ellipse((nbx // 2, nby // 2), 361, 361, color='r', fill=False, linewidth=1.5)
    ax1.add_artist(circle)
    plt.pause(0.1)

    return labels_top, labels_bottom, stereo_proj


def taubin_smooth(faces, vertices, cmap=default_cmap, iterations=10, lamda=0.5, mu=0.53, debugging=0):
    """
    Taubinsmooth: performs a back and forward Laplacian smoothing "without shrinking" of a triangulated mesh,
     as described by Gabriel Taubin (ICCV '95)

    :param faces: m*3 ndarray of m faces defined by 3 indices of vertices
    :param vertices: n*3 ndarray of n vertices defined by 3 positions
    :param cmap: colormap used for plotting
    :param iterations: number of iterations for smoothing (default 30)
    :param lamda: smoothing variable 0 < lambda < mu < 1 (default 0.5)
    :param mu: smoothing variable 0 < lambda < mu < 1 (default 0.53)
    :param debugging: show plots for debugging
    :return: smoothened vertices (ndarray n*3), normals to triangle (ndarray m*3)
    """
    from mpl_toolkits.mplot3d import Axes3D
    plt.ion()

    neighbours = find_neighbours(vertices, faces)  # get the indices of neighboring vertices for each vertex
    old_vertices = np.copy(vertices)
    indices_edges = detect_edges(faces)  # find indices of vertices defining non-shared edges (near hole...)
    new_vertices = np.copy(vertices)

    for k in range(iterations):
        vertices = np.copy(new_vertices)
        for i in range(vertices.shape[0]):
            indices = neighbours[i]  # list of indices
            distances = np.sqrt(np.sum((vertices[indices, :]-vertices[i, :]) ** 2, axis=1))
            weights = distances**(-1)
            vectoren = weights[:, np.newaxis] * vertices[indices, :]
            totaldist = sum(weights)
            new_vertices[i, :] = vertices[i, :] + lamda*(np.sum(vectoren, axis=0)/totaldist-vertices[i, :])
        if indices_edges.size != 0:
            new_vertices[indices_edges, :] = vertices[indices_edges, :]

        vertices = np.copy(new_vertices)
        for i in range(vertices.shape[0]):
            indices = neighbours[i]  # list of indices
            distances = np.sqrt(np.sum((vertices[indices, :]-vertices[i, :])**2, axis=1))
            weights = distances**(-1)
            # weights[np.argwhere(np.isnan(weights))] = 0
            vectoren = weights[:, np.newaxis] * vertices[indices, :]
            totaldist = sum(weights)
            new_vertices[i, :] = vertices[i, :] - mu*(sum(vectoren)/totaldist - vertices[i, :])
        if indices_edges.size != 0:
            new_vertices[indices_edges, :] = vertices[indices_edges, :]

    tfind = np.argwhere(np.isnan(new_vertices[:, 0]))
    new_vertices[tfind, :] = old_vertices[tfind, :]

    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = new_vertices[faces]
    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0,
    # and v2-v0 in each triangle
    # TODO: check direction of normals by dot product with the gradient of the support
    normals = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    areas = np.array([1/2 * np.linalg.norm(normal) for normal in normals])
    normals_length = np.sqrt(normals[:, 0]**2 + normals[:, 1]**2 + normals[:, 2]**2)
    normals = -1 * normals / normals_length[:, np.newaxis]   # flip and normalize normals
    # n is now an array of normalized normals, one per triangle.

    # calculate the colormap for plotting the weighted point density of normals on a sphere
    local_radius = 0.1
    color = np.zeros(normals.shape[0], dtype=normals.dtype)
    for i in range(normals.shape[0]):
        distances = np.sqrt(np.sum((normals - normals[i, :]) ** 2, axis=1))  # ndarray of  normals.shape[0]
        color[i] = distances[distances < local_radius].sum()
    color = color / max(color)
    if debugging:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(normals[:, 0], normals[:, 1], normals[:, 2], c=color, cmap=cmap)
        # ax.scatter(normals[:, 2], normals[:, 1], normals[:, 0], c=color, cmap=cmap)
        ax.set_xlim(-1, 1)
        ax.set_xlabel('z')
        ax.set_ylim(-1, 1)
        ax.set_ylabel('y')
        ax.set_zlim(-1, 1)
        ax.set_zlabel('x')
#         ax.set_aspect('equal', 'box')
        plt.title('Weighted point densities before KDE')
        plt.pause(0.1)
    err_normals = np.argwhere(np.isnan(normals[:, 0]))
    normals[err_normals, :] = normals[err_normals-1, :]
    plt.ioff()
    return new_vertices, normals, areas, color, err_normals


def save_planes_vti(filename, voxel_size, tuple_array, tuple_fieldnames, plane_labels, planes, origin=(0, 0, 0),
                    amplitude_threshold=0.01):
    """
    Save arrays defined by their name in a single vti file.

    :param filename: the file name of the vti file
    :param voxel_size: tuple (voxel_size_axis0, voxel_size_axis1, voxel_size_axis2)
    :param tuple_array: tuple of arrays of the same dimension
    :param tuple_fieldnames: tuple of name containing the same number of elements as tuple_array
    :param plane_labels: range of labels (label 0 is the background)
    :param planes: array of the same shape as arrays in 'tuple_array', with labeled voxels
    :param origin: tuple of points for vtk SetOrigin()
    :param amplitude_threshold: lower threshold for saving the reconstruction modulus (save memory space)
    :return: nothing
    """
    import vtk
    from vtk.util import numpy_support

    if type(tuple_fieldnames) is tuple:
        nb_fieldnames = len(tuple_fieldnames)
    elif type(tuple_fieldnames) is str:
        nb_fieldnames = 1
    else:
        raise TypeError('Invalid input for tuple_fieldnames')

    if type(tuple_array) is tuple:
        nb_arrays = len(tuple_array)
        nb_dim = tuple_array[0].ndim
        if nb_dim != 3:  # wrong array dimension
            raise ValueError('save_to_vti() needs a 3D array')
        nbz, nby, nbx = tuple_array[0].shape
    elif type(tuple_array) is np.ndarray:
        nb_arrays = 1
        nb_dim = tuple_array.ndim
        if nb_dim != 3:  # wrong array dimension
            raise ValueError('save_to_vti() needs a 3D array')
        nbz, nby, nbx = tuple_array.shape
    else:
        raise TypeError('Invalid input for tuple_array')

    if nb_arrays != nb_fieldnames:
        print('Different number of arrays and field names')
        return

    image_data = vtk.vtkImageData()
    image_data.SetOrigin(origin[0], origin[1], origin[2])
    image_data.SetSpacing(voxel_size[0], voxel_size[1], voxel_size[2])
    image_data.SetExtent(0, nbz - 1, 0, nby - 1, 0, nbx - 1)

    try:
        amp_index = tuple_fieldnames.index('amp')  # look for the substring 'amp'
        if nb_arrays > 1:
            amp_array = tuple_array[amp_index]
        else:
            amp_array = tuple_array
        amp_array = amp_array / amp_array.max()
        amp_array[amp_array < amplitude_threshold] = 0  # save disk space
        amp_array = np.transpose(np.flip(amp_array, 2)).reshape(amp_array.size)
        amp_array = numpy_support.numpy_to_vtk(amp_array)
        pd = image_data.GetPointData()
        pd.SetScalars(amp_array)
        pd.GetArray(0).SetName("amp")
        counter = 1
        if nb_arrays > 1:
            for idx in range(nb_arrays):
                if idx == amp_index:
                    continue
                temp_array = tuple_array[idx]
                temp_array[amp_array == 0] = 0
                temp_array = np.transpose(np.flip(temp_array, 2)).reshape(temp_array.size)
                temp_array = numpy_support.numpy_to_vtk(temp_array)
                pd.AddArray(temp_array)
                pd.GetArray(counter).SetName(tuple_fieldnames[idx])
                pd.Update()
                counter = counter + 1
    except ValueError:
        print('amp not in fieldnames, will save arrays without thresholding')
        if nb_arrays > 1:
            temp_array = tuple_array[0]
        else:
            temp_array = tuple_array
        temp_array = np.transpose(np.flip(temp_array, 2)).reshape(temp_array.size)
        temp_array = numpy_support.numpy_to_vtk(temp_array)
        pd = image_data.GetPointData()
        pd.SetScalars(temp_array)
        if nb_arrays > 1:
            pd.GetArray(0).SetName(tuple_fieldnames[0])
            for idx in range(1, nb_arrays):
                temp_array = tuple_array[idx]
                temp_array = np.transpose(np.flip(temp_array, 2)).reshape(temp_array.size)
                temp_array = numpy_support.numpy_to_vtk(temp_array)
                pd.AddArray(temp_array)
                pd.GetArray(idx).SetName(tuple_fieldnames[idx])
                pd.Update()
        else:
            pd.GetArray(0).SetName(tuple_fieldnames)
        counter = nb_arrays

    # save planes
    for label in plane_labels:
        plane = np.copy(planes)
        plane[plane != label] = 0
        plane[plane == label] = 1
        plane_array = np.transpose(plane).reshape(plane.size)
        plane_array = numpy_support.numpy_to_vtk(plane_array)
        pd.AddArray(plane_array)
        pd.GetArray(label + counter).SetName("plane_" + str(label))
        pd.Update()
    plane_array = np.transpose(planes).reshape(planes.size)
    plane_array = numpy_support.numpy_to_vtk(plane_array)
    pd.AddArray(plane_array)
    pd.GetArray(label + 1 + counter).SetName("all_planes")
    pd.Update()

    # export data to file
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(image_data)
    writer.Write()
    return


# if __name__ == "__main__":
#     ref_plane = np.array([1, 1, 1])
#     my_plane = np.array([1, 1, -1])
#     print(plane_angle_cubic(ref_plane, my_plane))
