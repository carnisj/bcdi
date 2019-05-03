# -*- coding: utf-8 -*-
"""
last correction on 24th August 2018: flipped normals, include stereographic projection
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage.measurements import center_of_mass
import pathlib
import vtk
from vtk.util import numpy_support
import os
import tkinter as tk
from tkinter import filedialog
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import stats
from scipy import ndimage
from scipy.signal import convolve
from skimage import measure
from skimage.feature import corner_peaks
from skimage.morphology import watershed
import logging
from scipy.interpolate import griddata

scan = 2227  # spec scan number
datadir = "C:/Users/carnis/Work Folders/Documents/data/CH4760_Pt/S"+str(scan)+"/simu/new_model/"
support_threshold = 0.75  # threshold for support determination
savedir = datadir + "isosurface_" + str(support_threshold) + "/"
# datadir = "C:/Users/carnis/Work Folders/Documents/data/CH4760_Pt/S"+str(scan)+"/pynxraw/"
# datadir = "C:/Users/carnis/Work Folders/Documents/data/CH5309/data/S"+str(scan)+"/pynxraw/"
reflection = np.array([1, 1, 1])  # measured crystallographic reflection
debug = 0  # 1 to see all plots
smoothing_iterations = 10
smooth_lamda = 0.65  # with lamda=0.65 and mu=0.6, the mesh is smaller than the support
smooth_mu = 0.60
kde_threshold = -0.35  # threshold for defining the background in the kernel density estimation of normals
############################################################
pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)
logger = logging.getLogger()
# define a colormap
cdict = {'red':  ((0.0, 1.0, 1.0),
                  (0.11, 0.0, 0.0),
                  (0.36, 0.0, 0.0),
                  (0.62, 1.0, 1.0),
                  (0.87, 1.0, 1.0),
                  (1.0, 0.0, 0.0)),
         'green': ((0.0, 1.0, 1.0),
                   (0.11, 0.0, 0.0),
                   (0.36, 1.0, 1.0),
                   (0.62, 1.0, 1.0),
                   (0.87, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),
         'blue': ((0.0, 1.0, 1.0),
                  (0.11, 1.0, 1.0),
                  (0.36, 1.0, 1.0),
                  (0.62, 0.0, 0.0),
                  (0.87, 0.0, 0.0),
                  (1.0, 0.0, 0.0))}
my_cmap = LinearSegmentedColormap('my_colormap', cdict, 256)


def calc_coordination(myamp, mythreshold, debugging=0):
    nbz, nby, nbx = myamp.shape
    mysupport = np.zeros((nbz, nby, nbx))
    mysupport[myamp > mythreshold * abs(myamp).max()] = 1

    mykernel = np.ones((3, 3, 3))
    mycoord = np.rint(convolve(mysupport, mykernel, mode='same'))
    mycoord = mycoord.astype(int)

    if debugging == 1:
        plt.figure(figsize=(18, 15))
        plt.subplot(2, 2, 1)
        plt.imshow(mycoord[:, :, nbx // 2])
        plt.colorbar()
        plt.axis('scaled')
        plt.title("Coordination matrix in middle slice in YZ")
        plt.subplot(2, 2, 2)
        plt.imshow(mycoord[:, nby // 2, :])
        plt.colorbar()
        plt.title("Coordination matrix in middle slice in XZ")
        plt.axis('scaled')
        plt.subplot(2, 2, 3)
        plt.imshow(mycoord[nbz // 2, :, :])
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.title("Coordination matrix in middle slice in XY")
        plt.axis('scaled')
        plt.pause(0.1)
    return mycoord


def taubin_smooth(myfaces, myvertices, iterations=10, lamda=0.5, mu=0.53, debugging=0):
    """
    taubinsmooth: performs a back and forward Laplacian smoothing "without shrinking" of a triangulated mesh,
    as described by Gabriel Taubin (ICCV '95)
    :param myfaces: ndarray of m*3 faces
    :param myvertices: ndarray of n*3 vertices
    :param iterations: number of iterations for smoothing (default 10)
    :param lamda: smoothing variable 0 < lambda < mu < 1 (default 0.5)
    :param mu: smoothing variable 0 < lambda < mu < 1 (default 0.53)
    :param debugging: show plots for debugging
    :return: smoothened vertices (ndarray n*3), normals to triangle (ndarray m*3)
    """
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
    normals_length = np.sqrt(mynormals[:, 0]**2 + mynormals[:, 1]**2 + mynormals[:, 2]**2)
    mynormals = -1 * mynormals / normals_length[:, np.newaxis]   # flip and normalize normals
    # n is now an array of normalized normals, one per triangle.

    # calculate the colormap for plotting the weighted point density of normals on a sphere
    local_radius = 0.1
    mycolor = np.zeros(mynormals.shape[0], dtype=mynormals.dtype)
    for i in range(mynormals.shape[0]):
        mydistances = np.sqrt(np.sum((mynormals - mynormals[i, :]) ** 2, axis=1))  # ndarray of my mynormals.shape[0]
        mycolor[i] = mydistances[mydistances < local_radius].sum()
    mycolor = mycolor / max(mycolor)
    if debugging == 0:
        myfig = plt.figure()
        myax = Axes3D(myfig)
        myax.scatter(mynormals[:, 0], mynormals[:, 1], mynormals[:, 2], c=mycolor, cmap=my_cmap)
        # myax.scatter(mynormals[:, 2], mynormals[:, 1], mynormals[:, 0], c=mycolor, cmap=my_cmap)
        myax.set_xlim(-1, 1)
        myax.set_xlabel('z')
        myax.set_ylim(-1, 1)
        myax.set_ylabel('y')
        myax.set_zlim(-1, 1)
        myax.set_zlabel('x')
        myax.set_aspect('equal', 'box')
        plt.title('Weighted point densities before KDE')
        plt.pause(0.1)
    err_normals = np.argwhere(np.isnan(mynormals[:, 0]))
    mynormals[err_normals, :] = mynormals[err_normals-1, :]
    return new_vertices, mynormals, mycolor, err_normals


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


def stereographic_proj(mynormals, mycolor, myreflection, background_threshold=-0.35, flag_plotplanes=1, debugging=0):
    """

    :param mynormals: array of normals (nb_normals rows x 3 columns)
    :param mycolor: array of intensities (nb_normals rows x 1 column)
    :param myreflection: measured crystallographic reflection
    :param flag_plotplanes: plot circles corresponding to crystallogrpahic orientations in the pole figure
    :param debugging: show plots for debugging
    :return:
    """
    # define crystallographic planes of interest
    planes = {}
    planes['1 0 0'] = plane_angle(myreflection, np.array([1, 0, 0]))
    # planes['-1 0 0'] = plane_angle(myreflection, np.array([-1, 0, 0]))
    planes['1 1 0'] = plane_angle(myreflection, np.array([1, 1, 0]))
    # planes['1 -1 0'] = plane_angle(myreflection, np.array([1, -1, 0]))
    # planes['1 1 1'] = plane_angle(myreflection, np.array([1, 1, 1]))
    planes['1 -1 1'] = plane_angle(myreflection, np.array([1, -1, 1]))
    # planes['1 -1 -1'] = plane_angle(myreflection, np.array([1, -1, -1]))
    planes['2 1 0'] = plane_angle(myreflection, np.array([2, 1, 0]))
    planes['2 -1 0'] = plane_angle(myreflection, np.array([2, -1, 0]))
    # planes['2 -1 1'] = plane_angle(myreflection, np.array([2, -1, 1]))
    # planes['3 0 1'] = plane_angle(myreflection, np.array([3, 0, 1]))
    # planes['3 -1 0'] = plane_angle(myreflection, np.array([3, -1, 0]))
    planes['3 2 1'] = plane_angle(myreflection, np.array([3, 2, 1]))
    # planes['3 -2 -1'] = plane_angle(myreflection, np.array([3, -2, -1]))
    # planes['-3 0 -1'] = plane_angle(myreflection, np.array([-3, 0, -1]))
    planes['4 0 -1'] = plane_angle(myreflection, np.array([4, 0, -1]))
    planes['5 2 0'] = plane_angle(myreflection, np.array([5, 2, 0]))
    # planes['5 -2 0'] = plane_angle(myreflection, np.array([5, -2, 0]))
    planes['5 2 1'] = plane_angle(myreflection, np.array([5, 2, 1]))
    planes['5 -2 -1'] = plane_angle(myreflection, np.array([5, -2, -1]))
    # planes['-5 0 -2'] = plane_angle(myreflection, np.array([-5, 0, -2]))
    # planes['7 0 3'] = plane_angle(myreflection, np.array([7, 0, 3]))
    # planes['7 -3 0'] = plane_angle(myreflection, np.array([-7, 0, 3]))
    # planes['-7 0 -3'] = plane_angle(myreflection, np.array([-7, 0, -3]))
    # planes['1 3 6'] = plane_angle(myreflection, np.array([1, 3, 6]))

    # check normals for nan
    radius_mean = 1  # normals are normalized
    stereo_centerz = 0  # COM of the weighted point density
    list_nan = np.argwhere(np.isnan(mynormals))
    if len(list_nan) != 0:
        for i in range(list_nan.shape[0]//3):
            mynormals = np.delete(mynormals, list_nan[i*3, 0], axis=0)
            mycolor = np.delete(mycolor, list_nan[i*3, 0], axis=0)

    # calculate u and v from xyz, this is equal to the stereographic projection from South pole
    stereo_proj = np.zeros((mynormals.shape[0], 4), dtype=mynormals.dtype)
    for i in range(mynormals.shape[0]):
        if mynormals[i, 1] == 0 and mynormals[i, 0] == 0:
            continue
        stereo_proj[i, 0] = radius_mean * mynormals[i, 0] / (radius_mean+mynormals[i, 1] - stereo_centerz)  # u_top
        stereo_proj[i, 1] = radius_mean * mynormals[i, 2] / (radius_mean+mynormals[i, 1] - stereo_centerz)  # v_top
        stereo_proj[i, 2] = radius_mean * mynormals[i, 0] / (stereo_centerz - radius_mean+mynormals[i, 1])  # u_bottom
        stereo_proj[i, 3] = radius_mean * mynormals[i, 2] / (stereo_centerz - radius_mean+mynormals[i, 1])  # v_bottom
    stereo_proj = stereo_proj / radius_mean * 90  # rescaling from radius_mean to 90

    u_grid_top, v_grid_top = np.mgrid[-91:91:183j, -91:91:183j]
    u_grid_bottom, v_grid_bottom = np.mgrid[-91:91:183j, -91:91:183j]
    int_grid_top = griddata((stereo_proj[:, 0], stereo_proj[:, 1]), mycolor,
                            (u_grid_top, v_grid_top), method='linear')
    int_grid_bottom = griddata((stereo_proj[:, 2], stereo_proj[:, 3]), mycolor,
                               (u_grid_bottom, v_grid_bottom), method='linear')
    int_grid_top = int_grid_top / int_grid_top[int_grid_top > 0].max() * 10000  # normalize for easier plotting
    int_grid_bottom = int_grid_bottom / int_grid_bottom[int_grid_bottom > 0].max() * 10000  # normalize for plotting

    # plot the stereographic projection
    myfig, (myax0, myax1) = plt.subplots(1, 2, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
    # plot top part (projection from South pole on equator)
    plt0 = myax0.contourf(u_grid_top, v_grid_top, abs(int_grid_top), range(100, 6100, 250), cmap='hsv')
    # plt.colorbar(plt0, ax=myax0)
    myax0.axis('equal')
    myax0.axis('off')

    # # add the projection of the elevation angle, depending on the center of projection
    for ii in range(15, 90, 5):
        circle = plt.Circle((0, 0),
                            radius_mean * np.sin(ii * np.pi / 180) / (1 + np.cos(ii * np.pi / 180)) * 90 / radius_mean,
                            color='grey', fill=False, linestyle='dotted', linewidth=0.5)
        myax0.add_artist(circle)
    for ii in range(10, 90, 20):
        circle = plt.Circle((0, 0),
                            radius_mean * np.sin(ii * np.pi / 180) / (1 + np.cos(ii * np.pi / 180)) * 90 / radius_mean,
                            color='grey', fill=False, linestyle='dotted', linewidth=1)
        myax0.add_artist(circle)
    for ii in range(10, 95, 20):
        myax0.text(-radius_mean * np.sin(ii * np.pi / 180) / (1 + np.cos(ii * np.pi / 180)) * 90 / radius_mean, 0,
                   str(ii) + '$^\circ$', fontsize=10, color='k')
    circle = plt.Circle((0, 0), 90, color='k', fill=False, linewidth=1.5)
    myax0.add_artist(circle)

    # add azimutal lines every 5 and 45 degrees
    for ii in range(5, 365, 5):
        myax0.plot([0, 90 * np.cos(ii * np.pi / 180)], [0, 90 * np.sin(ii * np.pi / 180)], color='grey',
                   linestyle='dotted', linewidth=0.5)
    for ii in range(0, 365, 20):
        myax0.plot([0, 90 * np.cos(ii * np.pi / 180)], [0, 90 * np.sin(ii * np.pi / 180)], color='grey',
                   linestyle='dotted', linewidth=1)

    # draw circles corresponding to particular reflection
    if flag_plotplanes == 1:
        indx = 0
        for key, value in planes.items():
            circle = plt.Circle((0, 0), radius_mean * np.sin(value * np.pi / 180) /
                                (1 + np.cos(value * np.pi / 180)) * 90 / radius_mean,
                                color='g', fill=False, linestyle='dotted', linewidth=1.5)
            myax0.add_artist(circle)
            myax0.text(np.cos(indx * np.pi / 180) * radius_mean * np.sin(value * np.pi / 180) /
                       (1 + np.cos(value * np.pi / 180)) * 90 / radius_mean,
                       np.sin(indx * np.pi / 180) * radius_mean * np.sin(value * np.pi / 180) /
                       (1 + np.cos(value * np.pi / 180)) * 90 / radius_mean,
                       key, fontsize=10, color='k', fontweight='bold')
            indx = indx + 6
            print(key + ": ", str('{:.2f}'.format(value)))
    myax0.set_title('Top projection\nfrom South pole\nS' + str(scan))

    # plot bottom part (projection from North pole on equator)
    plt1 = myax1.contourf(u_grid_bottom, v_grid_bottom, abs(int_grid_bottom), range(100, 6100, 250), cmap='hsv')
    # plt.colorbar(plt1, ax=ax1)
    myax1.axis('equal')
    myax1.axis('off')

    # # add the projection of the elevation angle, depending on the center of projection
    for ii in range(15, 90, 5):
        circle = plt.Circle((0, 0),
                            radius_mean * np.sin(ii * np.pi / 180) / (1 + np.cos(ii * np.pi / 180)) * 90 / radius_mean,
                            color='grey', fill=False, linestyle='dotted', linewidth=0.5)
        myax1.add_artist(circle)
    for ii in range(10, 90, 20):
        circle = plt.Circle((0, 0),
                            radius_mean * np.sin(ii * np.pi / 180) / (1 + np.cos(ii * np.pi / 180)) * 90 / radius_mean,
                            color='grey', fill=False, linestyle='dotted', linewidth=1)
        myax1.add_artist(circle)
    for ii in range(10, 95, 20):
        myax1.text(-radius_mean * np.sin(ii * np.pi / 180) / (1 + np.cos(ii * np.pi / 180)) * 90 / radius_mean, 0,
                   str(ii) + '$^\circ$', fontsize=10, color='k')
    circle = plt.Circle((0, 0), 90, color='k', fill=False, linewidth=1.5)
    myax1.add_artist(circle)

    # add azimutal lines every 5 and 45 degrees
    for ii in range(5, 365, 5):
        myax1.plot([0, 90 * np.cos(ii * np.pi / 180)], [0, 90 * np.sin(ii * np.pi / 180)], color='grey',
                   linestyle='dotted', linewidth=0.5)
    for ii in range(0, 365, 20):
        myax1.plot([0, 90 * np.cos(ii * np.pi / 180)], [0, 90 * np.sin(ii * np.pi / 180)], color='grey',
                   linestyle='dotted', linewidth=1)

    # draw circles corresponding to particular reflection
    if flag_plotplanes == 1:
        indx = 0
        for key, value in planes.items():
            circle = plt.Circle((0, 0), radius_mean * np.sin(value * np.pi / 180) /
                                (1 + np.cos(value * np.pi / 180)) * 90 / radius_mean,
                                color='g', fill=False, linestyle='dotted', linewidth=1.5)
            myax1.add_artist(circle)
            myax1.text(np.cos(indx * np.pi / 180) * radius_mean * np.sin(value * np.pi / 180) /
                       (1 + np.cos(value * np.pi / 180)) * 90 / radius_mean,
                       np.sin(indx * np.pi / 180) * radius_mean * np.sin(value * np.pi / 180) /
                       (1 + np.cos(value * np.pi / 180)) * 90 / radius_mean,
                       key, fontsize=10, color='k', fontweight='bold')
            indx = indx + 6
            print(key + ": ", str('{:.2f}'.format(value)))
    plt.title('Bottom projection\nfrom North pole\nS' + str(scan))

    plt.pause(0.1)

    # save figure
    plt.savefig(savedir + 'CDI_poles_S' + str(scan) + '.png')
    # save metric coordinates in text file
    int_grid_top[np.isnan(int_grid_top)] = 0.0
    fichier = open(savedir + 'CDI_poles_S' + str(scan) + '.dat', "w")
    for ii in range(len(u_grid_top)):
        for jj in range(len(v_grid_top)):
            fichier.write(str(u_grid_top[ii, 0]) + '\t' + str(v_grid_top[0, jj]) + '\t' +
                          str(int_grid_top[ii, jj]) + '\t' + str(u_grid_bottom[ii, 0]) + '\t' +
                          str(v_grid_bottom[0, jj]) + '\t' + str(int_grid_bottom[ii, jj]) + '\n')
    fichier.close()
    return 0


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


def equiproj_splatt_segment(mynormals, mycolor, background_threshold=-0.35, debugging=0):
    """

    :param mynormals: normals array
    :param mycolor: intensity array
    :param background_threshold: threshold for background determination (depth of the KDE)
    :param debugging: show plots for debugging
    :return: ndarray of labelled regions
    """
    # check normals for nan
    list_nan = np.argwhere(np.isnan(mynormals))
    if len(list_nan) != 0:
        for i in range(list_nan.shape[0]//3):
            mynormals = np.delete(mynormals, list_nan[i*3, 0], axis=0)
            mycolor = np.delete(mycolor, list_nan[i*3, 0], axis=0)
    # calculate latitude and longitude from xyz, this is equal to the equirectangular flat square projection
    long_lat = np.zeros((mynormals.shape[0], 2), dtype=mynormals.dtype)
    for i in range(mynormals.shape[0]):
        if mynormals[i, 1] == 0 and mynormals[i, 0] == 0:
            continue
        long_lat[i, 0] = np.arctan2(mynormals[i, 1], mynormals[i, 0])  # longitude
        long_lat[i, 1] = np.arcsin(mynormals[i, 2])  # latitude
    myfig = plt.figure()
    myax = myfig.add_subplot(111)
    myax.scatter(long_lat[:, 0], long_lat[:, 1], c=mycolor, cmap=my_cmap)
    myax.set_xlim(-np.pi, np.pi)
    myax.set_ylim(-np.pi / 2, np.pi / 2)
    plt.axis('scaled')
    plt.title('Equirectangular projection of the weighted point densities before KDE')
    plt.pause(0.1)

    # kernel density estimation
    kde = stats.gaussian_kde(long_lat.T, bw_method=0.03)
    # Create a regular 3D grid
    yi, xi = np.mgrid[-np.pi/2:np.pi/2:150j, -np.pi:np.pi:300j]
    # Evaluate the KDE on a regular grid...
    coords = np.vstack([item.ravel() for item in [xi, yi]])
    density = -1 * kde(coords).reshape(xi.shape)  # inverse density for later watershed segmentation
    if debugging == 1:
        myfig = plt.figure()
        myax = myfig.add_subplot(111)
        scatter = myax.scatter(xi, yi, c=density, cmap=my_cmap, vmin=-1.5, vmax=0)
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
    if debugging == 1:
        plt.figure()
        plt.imshow(mymask, cmap=cm.gray, interpolation='nearest')
        plt.title('Background mask')
        plt.gca().invert_yaxis()
        myfig = plt.figure()
        myax = myfig.add_subplot(111)
        scatter = myax.scatter(xi, yi, c=density, cmap=my_cmap)
        myax.set_xlim(-np.pi, np.pi)
        myax.set_ylim(-np.pi / 2, np.pi / 2)
        myfig.colorbar(scatter)
        plt.axis('scaled')
        plt.title('KDE after background definition')
        plt.pause(0.1)

    # Generate the markers as local minima of the distance to the background
    mydistances = ndimage.distance_transform_edt(density)
    if debugging == 1:
        plt.figure()
        plt.imshow(mydistances, cmap=cm.gray, interpolation='nearest')
        plt.title('Distances')
        plt.gca().invert_yaxis()
    local_maxi = corner_peaks(mydistances, min_distance=9, indices=False)  #
    mymarkers = ndimage.label(local_maxi)[0]

    # watershed segmentation
    mylabels = watershed(-mydistances, mymarkers, mask=mymask)
    print('There are', str(mylabels.max()), 'facets')  # label 0 is the background
    plt.figure()
    plt.imshow(mylabels, cmap=cm.spectral, interpolation='nearest')
    plt.title('Separated objects')
    plt.colorbar()
    plt.gca().invert_yaxis()

    return mylabels, long_lat


def fit_plane(myplane, mylabel, debugging=1):
    """
    fit a plane to labelled indices, ax+by+c=z
    :param myplane: 3D binary array of the shape of the data
    :param mylabel: int, only used for title in plot
    :param debugging: show plots for debugging
    :return: matrix of fit parameters [a, b, c], plane indices and errors associated
    """
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


def grow_facet(myfit, myplane, mylabel, debugging=1):
    myindices = np.nonzero(myplane == 1)
    if len(myindices[0]) == 0:
        no_points = 1
        return myplane, no_points
    mykernel = np.ones((10, 10, 10))
    myobject = np.copy(myplane[myindices[0].min():myindices[0].max()+1, myindices[1].min():myindices[1].max()+1,
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

    temp_plane = np.copy(myplane)
    temp_plane[myindices[0].min():myindices[0].max() + 1, myindices[1].min():myindices[1].max() + 1,
               myindices[2].min(): myindices[2].max() + 1] = mycoord
    new_indices = np.nonzero(temp_plane)
    temp_plane, no_points = distance_threshold(myfit, new_indices, 0.25, temp_plane.shape)

    new_indices = np.nonzero(temp_plane)
    myplane[new_indices[0], new_indices[1], new_indices[2]] = 1

    if debugging == 1 and len(new_indices[0]) != 0:
        # myindices = np.nonzero(mycoord)
        # mycolor = mycoord[myindices[0], myindices[1], myindices[2]]
        # myfig = plt.figure()
        # myax = plt.subplot(111, projection='3d')
        # myscatter = myax.scatter(myindices[0], myindices[1], myindices[2], s=2, c=mycolor,
        #                          cmap=my_cmap, vmin=0, vmax=mythreshold)
        # myax.set_xlabel('x')  # first dimension is x for plots, but z for NEXUS convention
        # myax.set_ylabel('y')
        # myax.set_zlabel('z')
        # plt.title("Convolution for plane " + str(mylabel) + ' after distance threshold')
        # myfig.colorbar(myscatter)

        myindices = np.nonzero(myplane)
        plt.figure()
        myax = plt.subplot(111, projection='3d')
        myax.scatter(myindices[0], myindices[1], myindices[2], color='b')
        myax.set_xlabel('x')  # first dimension is x for plots, but z for NEXUS convention
        myax.set_ylabel('y')
        myax.set_zlabel('z')
        plt.title("Plane " + str(mylabel) + ' after 1 cycle of facet growing')
        plt.pause(0.1)
        print(str(len(myindices[0])) + ' after 1 cycle of facet growing')
    return myplane, no_points


###################################################################
plt.ion()
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(initialdir=datadir, filetypes=[("NPZ", "*.npz")])
npzfile = np.load(file_path)
amp = npzfile['bulk']
amp = amp / amp.max()
strain = npzfile['strain']
support = np.zeros(amp.shape)
support[amp > support_threshold*amp.max()] = 1
nz, ny, nx = amp.shape
print("Initial data size: (", nz, ',', ny, ',', nx, ')')
coordination_matrix = calc_coordination(amp, support_threshold, debug)
surface = np.copy(support)
surface[coordination_matrix > 22] = 0  # remove the bulk 22

support = np.ones((nz, ny, nx))  # this support is 1 outside, 0 inside so that the gradient points towards exterior
support[abs(amp) > support_threshold * abs(amp).max()] = 0

zCOM, yCOM, xCOM = center_of_mass(support)
print("COM at (z, y, x): (", str('{:.2f}'.format(zCOM)), ',', str('{:.2f}'.format(yCOM)), ',',
      str('{:.2f}'.format(xCOM)), ')')
# Use marching cubes to obtain the surface mesh of these ellipsoids
vertices_old, faces, _, _ = measure.marching_cubes_lewiner(amp, level=support_threshold, step_size=2)
# TODO: check the effect of isosurface on marching cubes
# Display resulting triangular mesh using Matplotlib.
if debug == 1:
    fig = plt.figure(figsize=(10, 10))
    ax = Axes3D(fig)
    mesh = Poly3DCollection(vertices_old[faces])
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)
    ax.set_xlim(0, nz)
    ax.set_xlabel('Z')
    ax.set_ylim(0, ny)
    ax.set_ylabel('Y')
    ax.set_zlim(0, nx)
    # plt.tight_layout()
    ax.set_zlabel('X')
    plt.title('Mesh - z axis flipped because of nexus convention')
    plt.pause(0.1)

vertices_new, normals, color, error_normals = taubin_smooth(faces, vertices_old, iterations=smoothing_iterations,
                                                            lamda=smooth_lamda, mu=smooth_mu, debugging=debug)

# stereographic_proj(normals, color, reflection, flag_plotplanes=0, debugging=0, background_threshold=kde_threshold)

labels, longitude_latitude = equiproj_splatt_segment(normals, color, background_threshold=kde_threshold,
                                                     debugging=1)  # debug)
numy, numx = labels.shape
# TODO: calculate the stereographic projection of normals, for direction determination
# check if a normal belongs to a particular label, assigns label to the triangle vertices if this is the case
normals_label = np.zeros(longitude_latitude.shape[0], dtype=int)
vertices_label = np.zeros(vertices_new.shape[0], dtype=int)
for idx in range(longitude_latitude.shape[0]):
    row = int((longitude_latitude[idx, 1] + np.pi/2) * numy/np.pi)
    col = int((longitude_latitude[idx, 0] + np.pi) * numx/(2*np.pi))
    for label in range(1, labels.max()+1, 1):  # label 0 is the background
        try:
            if labels[row, col] == label:
                normals_label[idx] = label
                vertices_label[faces[idx, :]] = label
        except BaseException as e:
            logger.error(str(e))
            continue

# assign back labels to voxels using vertices
all_planes = np.zeros((nz, ny, nx), dtype=int)
planes_counter = np.zeros((nz, ny, nx), dtype=int)  # check if a voxel is used several times
for idx in range(vertices_new.shape[0]):
    temp_indices = np.rint(vertices_old[idx, :]).astype(int)
    planes_counter[temp_indices[0], temp_indices[1],
                   temp_indices[2]] = planes_counter[temp_indices[0], temp_indices[1], temp_indices[2]] + 1
    # check duplicated pixels (appearing several times) and remove them if they belong to different planes
    if planes_counter[temp_indices[0], temp_indices[1], temp_indices[2]] > 1:
        if all_planes[temp_indices[0], temp_indices[1], temp_indices[2]] != vertices_label[idx]:
            # belongs to different groups, therefore it is set as background (label 0)
            all_planes[temp_indices[0], temp_indices[1], temp_indices[2]] = 0
    else:  # non duplicated pixel
        all_planes[temp_indices[0], temp_indices[1], temp_indices[2]] = \
                vertices_label[idx]  # * support[temp_indices[0], temp_indices[1], temp_indices[2]]

if debug == 1:
    # prepare amp for vti file
    AMP = np.transpose(amp).reshape(amp.size)
    data_array = numpy_support.numpy_to_vtk(AMP)
    image_data = vtk.vtkImageData()
    image_data.SetOrigin(0, 0, 0)
    image_data.SetSpacing(1, 1, 1)
    image_data.SetExtent(0, nz - 1, 0, ny - 1, 0, nx - 1)
    pd = image_data.GetPointData()
    pd.SetScalars(data_array)
    pd.GetArray(0).SetName("amp")
    for label in range(1, labels.max()+1, 1):  # label 0 is the background
        plane = np.copy(all_planes)
        plane[plane != label] = 0
        plane[plane == label] = 1
        print(int(vertices_label[vertices_label == label].sum() / label), plane[plane == 1].sum())
        PLANE = np.transpose(plane).reshape(plane.size)
        plane_array = numpy_support.numpy_to_vtk(PLANE)
        pd.AddArray(plane_array)
        pd.GetArray(label).SetName("plane_" + str(label))
        pd.Update()
    PLANE = np.transpose(all_planes).reshape(all_planes.size)
    plane_array = numpy_support.numpy_to_vtk(PLANE)
    pd.AddArray(plane_array)
    pd.GetArray(labels.max()+1).SetName("all_planes")
    pd.Update()
    # export data to file
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(os.path.join(savedir, "S" + str(scan) + "_planes before refinement.vti"))
    writer.SetInputData(image_data)
    writer.Write()

# fit points by a plane, exclude points far away, refine the fit
gradz, grady, gradx = np.gradient(support, 1)  # support
file = open(os.path.join(savedir, "S" + str(scan) + "_planes.dat"), "w")
file.write('{0: <10}'.format('Plane #') + '\t' + '{0: <10}'.format('angle') + '\t' +
           '{0: <10}'.format('points #') + '\t' + '{0: <10}'.format('<strain>') + '\t' +
           '{0: <10}'.format('std dev') + '\t' + '{0: <10}'.format('A (x)') + '\t' +
           '{0: <10}'.format('B (y)') + '\t' + 'C (Ax+By+C=z)' + '\t' + 'normal X' + '\t' +
           'normal Y' + '\t' + 'normal Z' + '\n')
strain_file = open(os.path.join(savedir, "S" + str(scan) + "_strain.dat"), "w")
strain_file.write('{0: <10}'.format('Plane #') + '\t' + '{0: <10}'.format('Z') + '\t' + '{0: <10}'.format('Y') + '\t' +
                  '{0: <10}'.format('X') + '\t' + '{0: <10}'.format('strain')+'\n')
# prepare amp for vti file
AMP = np.transpose(amp).reshape(amp.size)
data_array = numpy_support.numpy_to_vtk(AMP)
image_data = vtk.vtkImageData()
image_data.SetOrigin(0, 0, 0)
image_data.SetSpacing(1, 1, 1)
image_data.SetExtent(0, nz - 1, 0, ny - 1, 0, nx - 1)
pd = image_data.GetPointData()
pd.SetScalars(data_array)
pd.GetArray(0).SetName("amp")
index_vti = 1
for label in range(1, labels.max()+1, 1):  # label 0 is the background

    # raw fit including all points
    plane = np.copy(all_planes)
    plane[plane != label] = 0
    plane[plane == label] = 1
    if plane[plane == 1].sum() == 0:  # no points on the plane
        print('Raw fit: no points for plane', label)
        continue
    coeffs,  plane_indices, stop = fit_plane(plane, label, debugging=debug)
    if stop == 1:
        print('No points remaining after raw fit for plane', label)
        continue

    # update plane
    plane, stop = distance_threshold(coeffs,  plane_indices, 1, plane.shape)
    if stop == 1:  # no points on the plane
        print('Refined fit: no points for plane', label)
        continue
    else:
        print('Plane', label, ', ', str(plane[plane == 1].sum()), 'points after checking distance to plane')

    coeffs, plane_indices, stop = fit_plane(plane, label, debugging=debug)
    if stop == 1:
        print('No points remaining after refined fit for plane', label)
        continue

    # update plane
    plane, stop = distance_threshold(coeffs, plane_indices, 0.45, plane.shape)
    if stop == 1:  # no points on the plane
        print('Refined fit: no points for plane', label)
        continue
    print('Plane', label, ', ', str(plane[plane == 1].sum()), 'points after refined fit')
    if debug == 1:
        plane_indices = np.nonzero(plane == 1)
        plt.figure()
        ax = plt.subplot(111, projection='3d')
        ax.scatter(plane_indices[0], plane_indices[1], plane_indices[2], color='b')
        plt.title('Plane' + str(label) + ' after refined fit')
        ax.set_xlabel('x')  # first dimension is x for plots, but z for NEXUS convention
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.pause(0.1)

    # grow the facet towards the interior
    iterate = 0
    while stop == 0:
        previous_nb = plane[plane == 1].sum()
        plane, stop = grow_facet(coeffs, plane, label, debugging=debug)
        plane_indices = np.nonzero(plane == 1)
        iterate = iterate + 1
        if plane[plane == 1].sum() == previous_nb:
            break
    grown_points = plane[plane == 1].sum()
    print('Plane ', label, ', ', str(grown_points), 'points after growing facet')
    plane_indices = np.nonzero(plane == 1)
    if debug == 1:
        plt.figure()
        ax = plt.subplot(111, projection='3d')
        ax.scatter(plane_indices[0], plane_indices[1], plane_indices[2], color='b')
        plt.title('Plane' + str(label) + ' after growing the facet')
        ax.set_xlabel('x')  # first dimension is x for plots, but z for NEXUS convention
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.pause(0.1)

    # correct for the offset between plane equation and the outer shell of the support (effect of meshing/smoothing)
    # crop the support to a small ROI included in the plane box
    support_indices = np.nonzero(surface[
                                 plane_indices[0].min() - 3:plane_indices[0].max() + 3,
                                 plane_indices[1].min() - 3:plane_indices[1].max() + 3,
                                 plane_indices[2].min() - 3:plane_indices[2].max() + 3])
    sup0 = support_indices[0] + plane_indices[0].min() - 3  # add offset plane_indices[0].min() - 3
    sup1 = support_indices[1] + plane_indices[1].min() - 3  # add offset plane_indices[1].min() - 3
    sup2 = support_indices[2] + plane_indices[2].min() - 3  # add offset plane_indices[2].min() - 3
    plane_normal = np.array([coeffs[0, 0], coeffs[1, 0], -1])  # normal is [a, b, c] if ax+by+cz+d=0
    dist = np.zeros(len(support_indices[0]))
    for point in range(len(support_indices[0])):
        dist[point] = (coeffs[0, 0]*sup0[point] + coeffs[1, 0]*sup1[point] - sup2[point] + coeffs[2, 0]) \
               / np.linalg.norm(plane_normal)
    mean_dist = dist.mean()
    print('Mean distance of plane ', label, ' to outer shell = ' + str('{:.2f}'.format(mean_dist)) + 'pixels')
    dist = np.zeros(len(support_indices[0]))
    for point in range(len(support_indices[0])):
        dist[point] = (coeffs[0, 0]*sup0[point] + coeffs[1, 0]*sup1[point] - sup2[point] +
                       (coeffs[2, 0] - mean_dist / 2)) / np.linalg.norm(plane_normal)
    new_dist = dist.mean()
    # these directions are for a mesh smaller than the support
    if mean_dist*new_dist < 0:  # crossed the support surface
        step_shift = np.sign(mean_dist) * 0.5
    elif abs(new_dist) - abs(mean_dist) < 0:
        step_shift = np.sign(mean_dist) * 0.5
    else:  # going away from surface, wrong direction
        step_shift = -1 * np.sign(mean_dist) * 0.5

    step_shift = -1*step_shift  # added JCR 24082018 because the direction of normals was flipped

    common_previous = 0
    found_plane = 0
    nbloop = 1
    crossed_surface = 0
    shift_direction = 0
    while found_plane == 0:
        common_points = 0
        plane_newindices0 = np.rint(plane_indices[0] +
                                    nbloop*step_shift * np.dot(np.array([1, 0, 0]), plane_normal /
                                                               np.linalg.norm(plane_normal))).astype(int)
        plane_newindices1 = np.rint(plane_indices[1] +
                                    nbloop*step_shift * np.dot(np.array([0, 1, 0]), plane_normal /
                                                               np.linalg.norm(plane_normal))).astype(int)
        plane_newindices2 = np.rint(plane_indices[2] +
                                    nbloop*step_shift * np.dot(np.array([0, 0, 1]), plane_normal /
                                                               np.linalg.norm(plane_normal))).astype(int)
        for point in range(len(plane_newindices0)):
            for point2 in range(len(sup0)):
                if plane_newindices0[point] == sup0[point2] and plane_newindices1[point] == sup1[point2]\
                        and plane_newindices2[point] == sup2[point2]:
                    common_points = common_points + 1

        if debug == 1:
            tempcoeff2 = coeffs[2, 0] - nbloop * step_shift
            dist = np.zeros(len(support_indices[0]))
            for point in range(len(support_indices[0])):
                dist[point] = (coeffs[0, 0] * sup0[point] + coeffs[1, 0] * sup1[point] - sup2[point] + tempcoeff2) \
                              / np.linalg.norm(plane_normal)
            temp_mean_dist = dist.mean()
            plane = np.zeros(surface.shape)
            plane[plane_newindices0, plane_newindices1, plane_newindices2] = 1
            plt.figure()
            ax = plt.subplot(111, projection='3d')
            ax.scatter(plane_newindices0, plane_newindices1, plane_newindices2, s=8, color='b')

            ax.scatter(sup0, sup1, sup2, s=2, color='r')
            plt.title('Plane ' + str(label) + ' after shifting facet - iteration' + str(nbloop))
            ax.set_xlabel('x')  # first dimension is x for plots, but z for NEXUS convention
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            plt.pause(0.1)
            print('(while) iteration ', nbloop, '- Mean distance of the plane to outer shell = ' +
                  str('{:.2f}'.format(temp_mean_dist)) + '\n pixels - common_points = ', common_points)

        if common_points != 0:
            if common_points >= common_previous:
                found_plane = 0
                common_previous = common_points
                print('(while) iteration ', nbloop, ' - ', common_previous, 'points belonging to the facet for plane ',
                      label)
                nbloop = nbloop + 1
                crossed_surface = 1
            elif common_points < grown_points / 5:  # try to keep enough points for statistics, half step back
                found_plane = 1
                print('Exiting while loop after threshold reached - ', common_previous,
                      'points belonging to the facet for plane ', label, '- next step common points=', common_points)
            else:
                found_plane = 0
                common_previous = common_points
                print('(while) iteration ', nbloop, ' - ', common_previous, 'points belonging to the facet for plane ',
                      label)
                nbloop = nbloop + 1
                crossed_surface = 1
        else:
            if crossed_surface == 1:  # found the outer shell
                found_plane = 1
                print('Exiting while loop - ', common_previous, 'points belonging to the facet for plane ', label,
                      '- next step common points=', common_points)
            elif nbloop < 5:
                print('(while) iteration ', nbloop, ' - ', common_previous, 'points belonging to the facet for plane ',
                      label)
                nbloop = nbloop + 1
            else:
                if shift_direction == 1:  # already unsuccessful in the other direction
                    print('No point from support is intersecting the plane ', label)
                    stop = 1
                    break
                else:  # distance to support metric not reliable, start again in the other direction
                    shift_direction = 1
                    print('Shift scanning direction')
                    step_shift = -1 * step_shift
                    nbloop = 1

    if stop == 1:  # no points on the plane
        print('Intersecting with support: no points for plane', label)
        continue
    # go back one step
    coeffs[2, 0] = coeffs[2, 0] - (nbloop-1)*step_shift
    plane_newindices0 = np.rint(plane_indices[0] +
                                (nbloop-1)*step_shift * np.dot(np.array([1, 0, 0]), plane_normal /
                                                               np.linalg.norm(plane_normal))).astype(int)
    plane_newindices1 = np.rint(plane_indices[1] +
                                (nbloop - 1)*step_shift * np.dot(np.array([0, 1, 0]), plane_normal /
                                                                 np.linalg.norm(plane_normal))).astype(int)
    plane_newindices2 = np.rint(plane_indices[2] +
                                (nbloop - 1)*step_shift * np.dot(np.array([0, 0, 1]), plane_normal /
                                                                 np.linalg.norm(plane_normal))).astype(int)

    plane = np.zeros(surface.shape)
    plane[plane_newindices0, plane_newindices1, plane_newindices2] = 1
    # use only pixels belonging to the outer shell of the support
    plane = plane * surface
    # plot result
    plane_indices = np.nonzero(plane == 1)
    plt.figure()
    ax = plt.subplot(111, projection='3d')
    ax.scatter(plane_indices[0], plane_indices[1], plane_indices[2], s=8, color='b')

    ax.scatter(sup0, sup1, sup2, s=2, color='r')
    plt.title('Plane ' + str(label) + ' after growing facet and matching to support\n iteration' +
              str(iterate) + '- Points number=' + str(len(plane_indices[0])))
    ax.set_xlabel('x')  # first dimension is x for plots, but z for NEXUS convention
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.pause(0.1)

    if plane[plane == 1].sum() == 0:  # no point belongs to the support
        print('Plane ', label, ' , no point belongs to support')
        continue

    # grow again the facet on the support towards the interior
    print('Growing again the facet')
    while stop == 0:
        previous_nb = plane[plane == 1].sum()
        plane, stop = grow_facet(coeffs, plane, label, debug)
        plane_indices = np.nonzero(plane == 1)
        plane = plane * surface  # use only pixels belonging to the outer shell of the support
        if plane[plane == 1].sum() == previous_nb:
            break
    grown_points = plane[plane == 1].sum().astype(int)
    print('Plane ', label, ', ', str(grown_points), 'points after growing facet on support\n')
    plane_indices = np.nonzero(plane == 1)
    if plane[plane == 1].sum() < 20:  # not enough point belongs to the support
        print('Plane ', label, ' , not enough points belong to support')
        continue
    if debug == 1:
        plt.figure()
        ax = plt.subplot(111, projection='3d')
        ax.scatter(plane_indices[0], plane_indices[1], plane_indices[2], color='b')
        plt.title('Plane'+str(label)+' after growing the facet on support - Points number='+str(len(plane_indices[0])))
        ax.set_xlabel('x')  # first dimension is x for plots, but z for NEXUS convention
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.pause(0.1)
    # calculate mean gradient
    mean_gradient = np.zeros(3)
    # mean_gradient2 = np.zeros(3)
    ind_z = plane_indices[0]
    ind_y = plane_indices[1]
    ind_x = plane_indices[2]
    for point in range(len(plane_indices[0])):
        mean_gradient[0] = mean_gradient[0] + (ind_z[point] - zCOM)
        mean_gradient[1] = mean_gradient[1] + (ind_y[point] - yCOM)
        mean_gradient[2] = mean_gradient[2] + (ind_x[point] - xCOM)
        # mean_gradient2[0] = mean_gradient2[0] + gradz[ind_z[point], ind_y[point], ind_x[point]]
        # mean_gradient2[1] = mean_gradient2[1] + grady[ind_z[point], ind_y[point], ind_x[point]]
        # mean_gradient2[2] = mean_gradient2[2] + gradx[ind_z[point], ind_y[point], ind_x[point]]
    if np.linalg.norm(mean_gradient) == 0:
        print('gradient at surface is 0, cannot determine the correct direction of surface normal')
    else:
        mean_gradient = mean_gradient / np.linalg.norm(mean_gradient)
        # mean_gradient2 = mean_gradient2 / np.linalg.norm(mean_gradient2)
    # check the correct direction of the normal using the gradient of the support
    ref_direction = np.array([0, 1, 0])  # [111] is vertical
    plane_normal = np.array([coeffs[0, 0], coeffs[1, 0], -1])  # normal is [a, b, c] if ax+by+cz+d=0
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    if np.dot(plane_normal, mean_gradient) < 0:  # normal is in the reverse direction
        print('Flip normal direction plane', str(label),'\n')
        plane_normal = -1 * plane_normal
    # calculate the angle of the plane normal to [111]
    angle_plane = 180 / np.pi * np.arccos(np.dot(ref_direction, plane_normal))
    # calculate the average strain for plane voxels
    plane_indices = np.nonzero(plane == 1)
    ind_z = plane_indices[0]
    ind_y = plane_indices[1]
    ind_x = plane_indices[2]
    nb_points = len(plane_indices[0])
    for idx in range(nb_points):
        strain_file.write('{0: <10}'.format(str(label)) + '\t' +
                          '{0: <10}'.format(str(ind_z[idx])) + '\t' +
                          '{0: <10}'.format(str(ind_y[idx])) + '\t' +
                          '{0: <10}'.format(str(ind_x[idx])) + '\t' +
                          '{0: <10}'.format(str('{:.7f}'.format(strain[ind_z[idx], ind_y[idx], ind_x[idx]])))+'\n')

    plane_strain = np.mean(strain[plane == 1])
    plane_deviation = np.std(strain[plane == 1])
    file.write('{0: <10}'.format(str(label)) + '\t' +
               '{0: <10}'.format(str('{:.3f}'.format(angle_plane))) + '\t' +
               '{0: <10}'.format(str(nb_points)) + '\t' +
               '{0: <10}'.format(str('{:.7f}'.format(plane_strain))) + '\t' +
               '{0: <10}'.format(str('{:.7f}'.format(plane_deviation))) + '\t' +
               '{0: <10}'.format(str('{:.5f}'.format(coeffs[0, 0]))) + '\t' +
               '{0: <10}'.format(str('{:.5f}'.format(coeffs[1, 0]))) + '\t' +
               '{0: <10}'.format(str('{:.5f}'.format(coeffs[2, 0]))) + '\t' +
               '{0: <10}'.format(str('{:.5f}'.format(plane_normal[0]))) + '\t' +
               '{0: <10}'.format(str('{:.5f}'.format(plane_normal[1]))) + '\t' +
               '{0: <10}'.format(str('{:.5f}'.format(plane_normal[2]))) + '\n')
    # update vti file
    PLANE = np.transpose(plane).reshape(plane.size)
    plane_array = numpy_support.numpy_to_vtk(PLANE)
    pd.AddArray(plane_array)
    pd.GetArray(index_vti).SetName("plane_"+str(label))
    pd.Update()
    index_vti = index_vti + 1
file.write('\n'+'Isosurface value'+'\t' '{0: <10}'.format(str(support_threshold)))
strain_file.write('\n'+'Isosurface value'+'\t' '{0: <10}'.format(str(support_threshold)))
file.close()
strain_file.close()
# export data to file
writer = vtk.vtkXMLImageDataWriter()
writer.SetFileName(os.path.join(savedir, "S" + str(scan) + "_refined planes.vti"))
writer.SetInputData(image_data)
writer.Write()
print('End of script')
plt.ioff()
plt.show()