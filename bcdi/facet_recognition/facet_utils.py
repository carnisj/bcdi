# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np
from matplotlib import pyplot as plt
import gc
import sys
sys.path.append('//win.desy.de/home/carnisj/My Documents/myscripts/bcdi/')
import bcdi.graph.graph_utils as gu

colormap = gu.Colormap()
default_cmap = colormap.cmap


def detect_edges(myfaces):
    """
    find indices of vertices defining non-shared edges
    :param myfaces: ndarray of m*3 faces
    :return: 1D list of indices of vertices defining non-shared edges (near hole...)
    """
    # Get the three edges per triangle
    edge1 = np.copy(myfaces[:, 0:2])
    edge2 = np.array([np.copy(myfaces[:, 0]), np.copy(myfaces[:, 2])]).T
    edge3 = np.array([np.copy(myfaces[:, 1]), np.copy(myfaces[:, 2])]).T
    edge1.sort(axis=1)
    edge2.sort(axis=1)
    edge3.sort(axis=1)

    # list of edges without redundancy
    edges = np.concatenate((edge1, edge2, edge3), axis=0)
    edge_list, edges_indices, edges_counts = np.unique(edges, return_index=True, return_counts=True, axis=0)

    # isolate non redundant edges
    unique_edges = edge_list[edges_counts == 1].flatten()
    return unique_edges


def distance_threshold(myfit, myindices, mythreshold, myshape):
    myplane = np.zeros(myshape, dtype=int)
    no_points = 0
    indx = myindices[0]
    indy = myindices[1]
    indz = myindices[2]
    if len(myindices[0]) == 0:
        no_points = 1
        return myplane, no_points
    # remove outsiders based on distance to plane
    myplane_normal = np.array([myfit[0, 0], myfit[1, 0], -1])  # normal is [a, b, c] if ax+by+cz+d=0
    for mypoint in range(len(myindices[0])):
        mydist = abs(myfit[0, 0]*indx[mypoint] + myfit[1, 0]*indy[mypoint] -
                     indz[mypoint] + myfit[2, 0])/np.linalg.norm(myplane_normal)
        if mydist < mythreshold:
            myplane[indx[mypoint], indy[mypoint], indz[mypoint]] = 1
    if myplane[myplane == 1].sum() == 0:
        print('Distance_threshold: no points for plane')
        no_points = 1
        return myplane, no_points
    return myplane, no_points


def equirectangular_proj(mynormals, color, weights, cmap=default_cmap, bw_method=0.03, min_distance=10,
                         background_threshold=-0.35, debugging=False):
    """

    :param mynormals: normals array
    :param color: intensity array
    :param weights: weights used in the gaussian kernel density estimation
    :param cmap: colormap used for plotting
    :param bw_method: bw_method of gaussian_kde
    :param min_distance: min_distance of corner_peaks()
    :param background_threshold: threshold for background determination (depth of the KDE)
    :param debugging: if True, show plots for debugging
    :return: ndarray of labelled regions
    """
    from matplotlib import cm
    from scipy import stats
    from scipy import ndimage
    from skimage.feature import corner_peaks
    from skimage.morphology import watershed

    # check normals for nan
    list_nan = np.argwhere(np.isnan(mynormals))
    if len(list_nan) != 0:
        for i in range(list_nan.shape[0]//3):
            mynormals = np.delete(mynormals, list_nan[i*3, 0], axis=0)
            color = np.delete(color, list_nan[i*3, 0], axis=0)
    # calculate latitude and longitude from xyz, this is equal to the equirectangular flat square projection
    long_lat = np.zeros((mynormals.shape[0], 2), dtype=mynormals.dtype)
    for i in range(mynormals.shape[0]):
        if mynormals[i, 1] == 0 and mynormals[i, 0] == 0:
            continue
        long_lat[i, 0] = np.arctan2(mynormals[i, 1], mynormals[i, 0])  # longitude
        long_lat[i, 1] = np.arcsin(mynormals[i, 2])  # latitude
    myfig = plt.figure()
    myax = myfig.add_subplot(111)
    myax.scatter(long_lat[:, 0], long_lat[:, 1], c=color, cmap=cmap)
    myax.set_xlim(-np.pi, np.pi)
    myax.set_ylim(-np.pi / 2, np.pi / 2)
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

    myfig = plt.figure()
    myax = myfig.add_subplot(111)
    scatter = myax.scatter(xi, yi, c=density, cmap=cmap, vmin=-1.5, vmax=0)
    myax.set_xlim(-np.pi, np.pi)
    myax.set_ylim(-np.pi / 2, np.pi / 2)
    myfig.colorbar(scatter)
    plt.axis('scaled')
    plt.title('Equirectangular projection of the KDE')
    plt.pause(0.1)

    # identification of local minima
    density[density > background_threshold] = 0  # define the background
    mymask = np.copy(density)
    mymask[mymask != 0] = 1

    plt.figure()
    plt.imshow(mymask, cmap=cm.gray, interpolation='nearest')
    plt.title('Background mask')
    plt.gca().invert_yaxis()
    myfig = plt.figure()
    myax = myfig.add_subplot(111)
    scatter = myax.scatter(xi, yi, c=density, cmap=cmap)
    myax.set_xlim(-np.pi, np.pi)
    myax.set_ylim(-np.pi / 2, np.pi / 2)
    myfig.colorbar(scatter)
    plt.axis('scaled')
    plt.title('KDE after background definition')
    plt.pause(0.1)

    # Generate the markers as local minima of the distance to the background
    mydistances = ndimage.distance_transform_edt(density)
    if debugging:
        plt.figure()
        plt.imshow(mydistances, cmap=cm.gray, interpolation='nearest')
        plt.title('Distances')
        plt.gca().invert_yaxis()

    # find peaks
    local_maxi = corner_peaks(mydistances, exclude_border=False, min_distance=min_distance, indices=False)  #
    if debugging:
        plt.figure()
        plt.imshow(local_maxi, interpolation='nearest')
        plt.title('local_maxi')
        plt.gca().invert_yaxis()

    # define markers for each peak
    mymarkers = ndimage.label(local_maxi)[0]
    if debugging:
        plt.figure()
        plt.imshow(mymarkers, interpolation='nearest')
        plt.title('mymarkers')
        plt.colorbar()
        plt.gca().invert_yaxis()

    # watershed segmentation
    mylabels = watershed(-mydistances, mymarkers, mask=mymask)
    print('There are', str(mylabels.max()), 'facets')  # label 0 is the background

    plt.figure()
    plt.imshow(mylabels, cmap=cm.Spectral, interpolation='nearest')
    plt.title('Separated objects')
    plt.colorbar()
    plt.gca().invert_yaxis()

    return mylabels, long_lat


def find_neighbours(myvertices, myfaces):
    """
    Get the list of neighbouring vertices for each vertex
    :param myvertices: ndarray of n*3 vertices
    :param myfaces: ndarray of m*3 faces
    :return: list of lists of indices
    """
    neighbors = [None]*myvertices.shape[0]

    for indx in range(myfaces.shape[0]):
        if neighbors[myfaces[indx, 0]] is None:
            neighbors[myfaces[indx, 0]] = [myfaces[indx, 1], myfaces[indx, 2]]
        else:
            neighbors[myfaces[indx, 0]].append(myfaces[indx, 1])
            neighbors[myfaces[indx, 0]].append(myfaces[indx, 2])
        if neighbors[myfaces[indx, 1]] is None:
            neighbors[myfaces[indx, 1]] = [myfaces[indx, 2], myfaces[indx, 0]]
        else:
            neighbors[myfaces[indx, 1]].append(myfaces[indx, 2])
            neighbors[myfaces[indx, 1]].append(myfaces[indx, 0])
        if neighbors[myfaces[indx, 2]] is None:
            neighbors[myfaces[indx, 2]] = [myfaces[indx, 0], myfaces[indx, 1]]
        else:
            neighbors[myfaces[indx, 2]].append(myfaces[indx, 0])
            neighbors[myfaces[indx, 2]].append(myfaces[indx, 1])
    neighbors = [mylist for mylist in neighbors if mylist is not None]
    for indx in range(myvertices.shape[0]):
        neighbors[indx] = list(set(neighbors[indx]))  # remove redundant indices in each sublist
    return neighbors


def fit_plane(myplane, mylabel, debugging=1):
    """
    fit a plane to labelled indices, ax+by+c=z
    :param myplane: 3D binary array of the shape of the data
    :param mylabel: int, only used for title in plot
    :param debugging: show plots for debugging
    :return: matrix of fit parameters [a, b, c], plane indices and errors associated
    """
    from scipy.ndimage.measurements import center_of_mass

    myindices = np.nonzero(myplane == 1)
    no_points = 0
    if len(myindices[0]) == 0:
        no_points = 1
        return 0, myindices, no_points
    tmp_x = myindices[0]
    tmp_y = myindices[1]
    tmp_z = myindices[2]
    x_com, y_com, z_com = center_of_mass(myplane)

    # remove isolated points, which probably do not belong to the plane
    for mypoint in range(tmp_x.shape[0]):
        my_neighbors = myplane[tmp_x[mypoint]-2:tmp_x[mypoint]+3, tmp_y[mypoint]-2:tmp_y[mypoint]+3,
                               tmp_z[mypoint]-2:tmp_z[mypoint]+3].sum()
        # if debugging == 1:
        #     print(my_neighbors)
        if my_neighbors < 5:
            myplane[tmp_x[mypoint], tmp_y[mypoint], tmp_z[mypoint]] = 0
    print('Plane', mylabel, ', ', str(tmp_x.shape[0]-myplane[myplane == 1].sum()), 'points isolated, ',
          str(myplane[myplane == 1].sum()), 'remaining')
    myindices = np.nonzero(myplane == 1)
    if len(myindices[0]) == 0:
        no_points = 1
        return 0, myindices, no_points
    tmp_x = myindices[0]
    tmp_y = myindices[1]
    tmp_z = myindices[2]

    # remove points farther than 1.8 times the mean distance to COM
    mydist = np.zeros(tmp_x.shape[0])
    for mypoint in range(tmp_x.shape[0]):
        mydist[mypoint] = np.sqrt((tmp_x[mypoint]-x_com)**2+(tmp_y[mypoint]-y_com)**2+(tmp_z[mypoint]-z_com)**2)
    average_dist = np.mean(mydist)
    # plt.figure()
    # myax = plt.subplot(111, projection='3d')
    # myax.scatter(tmp_x, tmp_y, tmp_z, color='b')
    for mypoint in range(tmp_x.shape[0]):
        if mydist[mypoint] > 1.8 * average_dist:
            myplane[tmp_x[mypoint], tmp_y[mypoint], tmp_z[mypoint]] = 0
    print('Plane', mylabel, ', ', str(tmp_x.shape[0] - myplane[myplane == 1].sum()), 'points too far from COM, ',
          str(myplane[myplane == 1].sum()), 'remaining')
    myindices = np.nonzero(myplane == 1)
    if len(myindices[0]) < 5:
        no_points = 1
        return 0, myindices, no_points
    tmp_x = myindices[0]
    tmp_y = myindices[1]
    tmp_z = myindices[2]

    tmp_x = tmp_x[:, np.newaxis]
    tmp_y = tmp_y[:, np.newaxis]
    tmp_1 = np.ones((tmp_y.shape[0], 1))

    a = np.matrix(np.concatenate((tmp_x, tmp_y, tmp_1), axis=1))
    b = np.matrix(tmp_z).T
    myfit = (a.T * a).I * a.T * b
    # myerrors = b - a * myfit

    if debugging == 1:
        plt.figure()
        myax = plt.subplot(111, projection='3d')
        myax.scatter(tmp_x, tmp_y, tmp_z, color='b')
        myax.set_xlabel('x')  # first dimension is x for plots, but z for NEXUS convention
        myax.set_ylabel('y')
        myax.set_zlabel('z')
        xlim = myax.get_xlim()  # first dimension is x for plots, but z for NEXUS convention
        ylim = myax.get_ylim()
        meshx, meshy = np.meshgrid(np.arange(xlim[0], xlim[1]+1, 1), np.arange(ylim[0], ylim[1]+1, 1))
        meshz = np.zeros(meshx.shape)
        for myrow in range(meshx.shape[0]):
            for mycol in range(meshx.shape[1]):
                meshz[myrow, mycol] = myfit[0, 0] * meshx[myrow, mycol] +\
                                      myfit[1, 0] * meshy[myrow, mycol] + myfit[2, 0]
        myax.plot_wireframe(meshx, meshy, meshz, color='k')
        plt.title("Points and fitted plane" + str(mylabel))
        plt.pause(0.1)
    return myfit, myindices, no_points


def grow_facet(fit, plane, label, debugging=1):
    """

    :param fit:
    :param plane:
    :param label:
    :param debugging:
    :return:
    """
    from scipy.signal import convolve

    myindices = np.nonzero(plane == 1)
    if len(myindices[0]) == 0:
        no_points = 1
        return plane, no_points
    mykernel = np.ones((10, 10, 10))
    myobject = np.copy(plane[myindices[0].min():myindices[0].max()+1, myindices[1].min():myindices[1].max()+1,
                       myindices[2].min(): myindices[2].max() + 1])
    mycoord = np.rint(convolve(myobject, mykernel, mode='same'))

    # determine the threshold for growing the facet
    mycoord[myobject == 0] = 0
    mean_coord = mycoord.sum() / len(myindices[0])
    print('Plane ' + str(label) + ', mean coordination number = ' + str(mean_coord))
    mythreshold = mean_coord - 12  # -10 extension starting outwards

    # apply the estimated threshold
    mycoord = np.rint(convolve(myobject, mykernel, mode='same'))
    mycoord = mycoord.astype(int)
    mycoord[mycoord < mythreshold] = 0

    temp_plane = np.copy(plane)
    temp_plane[myindices[0].min():myindices[0].max() + 1, myindices[1].min():myindices[1].max() + 1,
               myindices[2].min(): myindices[2].max() + 1] = mycoord
    new_indices = np.nonzero(temp_plane)
    temp_plane, no_points = distance_threshold(fit, new_indices, 0.5, temp_plane.shape)

    new_indices = np.nonzero(temp_plane)
    plane[new_indices[0], new_indices[1], new_indices[2]] = 1

    if debugging == 1 and len(new_indices[0]) != 0:
        # myindices = np.nonzero(mycoord)
        # color = mycoord[myindices[0], myindices[1], myindices[2]]
        # myfig = plt.figure()
        # myax = plt.subplot(111, projection='3d')
        # myscatter = myax.scatter(myindices[0], myindices[1], myindices[2], s=2, c=color,
        #                          cmap=my_cmap, vmin=0, vmax=mythreshold)
        # myax.set_xlabel('x')  # first dimension is x for plots, but z for NEXUS convention
        # myax.set_ylabel('y')
        # myax.set_zlabel('z')
        # plt.title("Convolution for plane " + str(mylabel) + ' after distance threshold')
        # myfig.colorbar(myscatter)

        myindices = np.nonzero(plane)
        plt.figure()
        myax = plt.subplot(111, projection='3d')
        myax.scatter(myindices[0], myindices[1], myindices[2], color='b')
        myax.set_xlabel('x')  # first dimension is x for plots, but z for NEXUS convention
        myax.set_ylabel('y')
        myax.set_zlabel('z')
        plt.title("Plane " + str(label) + ' after 1 cycle of facet growing')
        plt.pause(0.1)
        print(str(len(myindices[0])) + ' after 1 cycle of facet growing')
    return plane, no_points


def plane_angle(ref_plane, myplane):
    """
    Calculate the angle between two crystallographic planes in cubic materials
    :param ref_plane: measured reflection
    :param myplane: plane for which angle should be calculated
    :return: the angle in degrees
    """
    if np.array_equal(ref_plane, myplane):
        angle = 0.0
    else:
        angle = 180 / np.pi * np.arccos(sum(np.multiply(ref_plane, myplane)) /
                                        (np.linalg.norm(ref_plane) * np.linalg.norm(myplane)))
    if angle > 90.0:
        angle = 180.0 - angle
    return angle


def stereographic_proj(normals, color, weights, savedir, min_distance=10, background_threshold=-1000,
                       save_txt=False, cmap=default_cmap, planes={}, plot_planes=True, debugging=False):
    """

    :param normals: array of normals (nb_normals rows x 3 columns)
    :param color: array of intensities (nb_normals rows x 1 column)
    :param weights: weights used in the density estimation
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

    yi, xi = np.mgrid[-91:91:364j, -91:91:365j]  # vertical, horizontal
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

    myfig = plt.figure(figsize=(15, 10))
    myax0 = myfig.add_subplot(121)
    scatter_top = myax0.scatter(xi, yi, c=density_top, cmap=cmap)
    myax0.set_xlim(-91, 91)
    myax0.set_ylim(-91, 91)
    myfig.colorbar(scatter_top)
    plt.axis('scaled')
    plt.title('KDE \nSouth pole')
    plt.pause(0.1)

    myax1 = myfig.add_subplot(122)
    scatter_bottom = myax1.scatter(xi, yi, c=density_bottom, cmap=cmap)
    myax1.set_xlim(-91, 91)
    myax1.set_ylim(-91, 91)
    myfig.colorbar(scatter_bottom)
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

    myfig = plt.figure(figsize=(15, 10))
    myax0 = myfig.add_subplot(221)
    myax0.imshow(mask_top, cmap=cmap, interpolation='nearest')  # cm.gray
    plt.title('Background mask South')
    plt.gca().invert_yaxis()
    myax1 = myfig.add_subplot(223)
    scatter_top = myax1.scatter(xi, yi, c=density_top, cmap=cmap)
    myax1.set_xlim(-91, 91)
    myax1.set_ylim(-91, 91)
    myfig.colorbar(scatter_top)
    plt.axis('scaled')
    plt.title('KDE South pole\nafter background definition')

    myax2 = myfig.add_subplot(222)
    myax2.imshow(mask_bottom, cmap=cmap, interpolation='nearest')  # cm.gray
    plt.title('Background mask North')
    plt.gca().invert_yaxis()
    myax3 = myfig.add_subplot(224)
    scatter_bottom = myax3.scatter(xi, yi, c=density_bottom, cmap=cmap)
    myax3.set_xlim(-91, 91)
    myax3.set_ylim(-91, 91)
    myfig.colorbar(scatter_bottom)
    plt.axis('scaled')
    plt.title('KDE North pole\nafter background definition')
    plt.pause(0.1)

    # Generate the markers as local minima of the distance to the background
    distances_top = ndimage.distance_transform_edt(density_top)
    distances_bottom = ndimage.distance_transform_edt(density_bottom)

    if debugging:
        myfig = plt.figure(figsize=(15, 10))
        myfig.add_subplot(121)
        plt.imshow(distances_top, cmap=cmap, interpolation='nearest')  # cm.gray
        plt.title('Distances South')
        plt.gca().invert_yaxis()
        myfig.add_subplot(122)
        plt.imshow(distances_bottom, cmap=cmap, interpolation='nearest')
        plt.title('Distances North')
        plt.gca().invert_yaxis()
        plt.pause(0.1)

    # find peaks
    local_maxi_top = corner_peaks(distances_top, exclude_border=False, min_distance=min_distance, indices=False)
    local_maxi_bottom = corner_peaks(distances_bottom, exclude_border=False, min_distance=min_distance, indices=False)

    if debugging:
        myfig = plt.figure(figsize=(15, 10))
        myfig.add_subplot(121)
        plt.imshow(local_maxi_top, interpolation='nearest')
        plt.title('local_maxi South')
        plt.gca().invert_yaxis()
        myfig.add_subplot(122)
        plt.imshow(local_maxi_bottom, interpolation='nearest')
        plt.title('local_maxi North')
        plt.gca().invert_yaxis()
        plt.pause(0.1)

    # define markers for each peak
    markers_top = ndimage.label(local_maxi_top)[0]
    markers_bottom = ndimage.label(local_maxi_bottom)[0]

    if debugging:
        myfig = plt.figure(figsize=(15, 10))
        myfig.add_subplot(121)
        plt.imshow(markers_top, interpolation='nearest')
        plt.title('markers South')
        plt.gca().invert_yaxis()
        myfig.add_subplot(122)
        plt.imshow(markers_bottom, interpolation='nearest')
        plt.title('markers North')
        plt.gca().invert_yaxis()
        plt.pause(0.1)

    # watershed segmentation
    labels_top = watershed(-distances_top, markers_top, mask=mask_top)
    labels_bottom = watershed(-markers_bottom, markers_bottom, mask=mask_bottom)

    myfig = plt.figure(figsize=(15, 10))
    myfig.add_subplot(121)
    plt.imshow(labels_top, cmap=cmap, interpolation='nearest')  # cm.Spectral
    plt.title('Separated objects South')
    plt.gca().invert_yaxis()
    myfig.add_subplot(122)
    plt.imshow(labels_bottom, cmap=cmap, interpolation='nearest')
    plt.title('Separated objects North')
    plt.gca().invert_yaxis()
    plt.pause(0.1)

    return labels_top, labels_bottom, stereo_proj


def taubin_smooth(myfaces, myvertices, cmap=default_cmap, iterations=10, lamda=0.5, mu=0.53, debugging=0):
    """
    taubinsmooth: performs a back and forward Laplacian smoothing "without shrinking" of a triangulated mesh,
    as described by Gabriel Taubin (ICCV '95)
    :param myfaces: m*3 ndarray of m faces defined by 3 indices of vertices
    :param myvertices: n*3 ndarray of n vertices defined by 3 positions
    :param cmap: colormap used for plotting
    :param iterations: number of iterations for smoothing (default 30)
    :param lamda: smoothing variable 0 < lambda < mu < 1 (default 0.5)
    :param mu: smoothing variable 0 < lambda < mu < 1 (default 0.53)
    :param debugging: show plots for debugging
    :return: smoothened vertices (ndarray n*3), normals to triangle (ndarray m*3)
    """
    from mpl_toolkits.mplot3d import Axes3D
    plt.ion()

    neighbours = find_neighbours(myvertices, myfaces)  # get the indices of neighboring vertices for each vertex
    old_vertices = np.copy(myvertices)
    indices_edges = detect_edges(myfaces)  # find indices of vertices defining non-shared edges (near hole...)
    new_vertices = np.copy(myvertices)

    for k in range(iterations):
        myvertices = np.copy(new_vertices)
        for i in range(myvertices.shape[0]):
            indices = neighbours[i]  # list of indices
            mydistances = np.sqrt(np.sum((myvertices[indices, :]-myvertices[i, :]) ** 2, axis=1))
            weights = mydistances**(-1)
            vectoren = weights[:, np.newaxis] * myvertices[indices, :]
            totaldist = sum(weights)
            new_vertices[i, :] = myvertices[i, :] + lamda*(np.sum(vectoren, axis=0)/totaldist-myvertices[i, :])
        if indices_edges.size != 0:
            new_vertices[indices_edges, :] = myvertices[indices_edges, :]

        myvertices = np.copy(new_vertices)
        for i in range(myvertices.shape[0]):
            indices = neighbours[i]  # list of indices
            mydistances = np.sqrt(np.sum((myvertices[indices, :]-myvertices[i, :])**2, axis=1))
            weights = mydistances**(-1)
            # weights[np.argwhere(np.isnan(weights))] = 0
            vectoren = weights[:, np.newaxis] * myvertices[indices, :]
            totaldist = sum(weights)
            new_vertices[i, :] = myvertices[i, :] - mu*(sum(vectoren)/totaldist - myvertices[i, :])
        if indices_edges.size != 0:
            new_vertices[indices_edges, :] = myvertices[indices_edges, :]

    tfind = np.argwhere(np.isnan(new_vertices[:, 0]))
    new_vertices[tfind, :] = old_vertices[tfind, :]

    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = new_vertices[myfaces]
    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0,
    # and v2-v0 in each triangle
    # TODO: check direction of normals by dot product with the gradient of the support
    mynormals = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    areas = np.array([1/2 * np.linalg.norm(normal) for normal in mynormals])
    normals_length = np.sqrt(mynormals[:, 0]**2 + mynormals[:, 1]**2 + mynormals[:, 2]**2)
    mynormals = -1 * mynormals / normals_length[:, np.newaxis]   # flip and normalize normals
    # n is now an array of normalized normals, one per triangle.

    # calculate the colormap for plotting the weighted point density of normals on a sphere
    local_radius = 0.1
    color = np.zeros(mynormals.shape[0], dtype=mynormals.dtype)
    for i in range(mynormals.shape[0]):
        mydistances = np.sqrt(np.sum((mynormals - mynormals[i, :]) ** 2, axis=1))  # ndarray of my mynormals.shape[0]
        color[i] = mydistances[mydistances < local_radius].sum()
    color = color / max(color)
    if debugging:
        myfig = plt.figure()
        myax = Axes3D(myfig)
        myax.scatter(mynormals[:, 0], mynormals[:, 1], mynormals[:, 2], c=color, cmap=cmap)
        # myax.scatter(mynormals[:, 2], mynormals[:, 1], mynormals[:, 0], c=color, cmap=cmap)
        myax.set_xlim(-1, 1)
        myax.set_xlabel('z')
        myax.set_ylim(-1, 1)
        myax.set_ylabel('y')
        myax.set_zlim(-1, 1)
        myax.set_zlabel('x')
#         myax.set_aspect('equal', 'box')
        plt.title('Weighted point densities before KDE')
        plt.pause(0.1)
    err_normals = np.argwhere(np.isnan(mynormals[:, 0]))
    mynormals[err_normals, :] = mynormals[err_normals-1, :]
    plt.ioff()
    return new_vertices, mynormals, areas, color, err_normals


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



