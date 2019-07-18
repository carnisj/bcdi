# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.measurements import center_of_mass
import pathlib
import vtk
from vtk.util import numpy_support
import os
import tkinter as tk
from tkinter import filedialog
from skimage import measure
import logging
import sys
sys.path.append('//win.desy.de/home/carnisj/My Documents/myscripts/bcdi/')
import bcdi.graph.graph_utils as gu
import bcdi.facet_recognition.facet_utils as fu
import bcdi.postprocessing.postprocessing_utils as pu

helptext = """
help text comes here
"""

scan = 2227  # spec scan number
datadir = 'D:/data/PtRh/PtRh(103x98x157)/'
# datadir = "C:/Users/carnis/Work Folders/Documents/data/CH4760_Pt/S"+str(scan)+"/simu/new_model/"
support_threshold = 0.55  # threshold for support determination
savedir = datadir + "isosurface_" + str(support_threshold) + "/"
# datadir = "C:/Users/carnis/Work Folders/Documents/data/CH4760_Pt/S"+str(scan)+"/pynxraw/"
# datadir = "C:/Users/carnis/Work Folders/Documents/data/CH5309/data/S"+str(scan)+"/pynxraw/"
reflection = np.array([1, 1, 1])  # measured crystallographic reflection
debug = 0  # 1 to see all plots
smoothing_iterations = 10  # number of iterations in Taubin smoothing
smooth_lamda = 0.5  # lambda parameter in Taubin smoothing
smooth_mu = 0.51  # mu parameter in Taubin smoothing
kde_threshold = -0.2  # threshold for defining the background in the kernel density estimation of normals
my_bw_method = 0.06  # bandwidth in the gaussian kernel density estimation
my_min_distance = 15  # pixel separation between peaks in corner_peaks()
##########################
# end of user parameters #
##########################

###########################################################
# create directory and initialize error logger #
###########################################################
pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)
logger = logging.getLogger()

#############
# load data #
#############
plt.ion()
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(initialdir=datadir, filetypes=[("NPZ", "*.npz")])
npzfile = np.load(file_path)
amp = npzfile['amp']
amp = amp / amp.max()
nz, ny, nx = amp.shape
print("Initial data size: (", nz, ',', ny, ',', nx, ')')
strain = npzfile['strain']

# define the support and the surface layer
support = np.zeros(amp.shape)
support[amp > support_threshold*amp.max()] = 1
coordination_matrix = pu.calc_coordination(support, kernel=np.ones((3, 3, 3)), debugging=False)
surface = np.copy(support)
surface[coordination_matrix > 22] = 0  # remove the bulk 22


# Use marching cubes to obtain the surface mesh of these ellipsoids
vertices_old, faces, _, _ = measure.marching_cubes_lewiner(amp, level=support_threshold, step_size=2)
# vertices is a list of 3d coordinates of all vertices points
# faces is a list of facets defined by the indices of 3 vertices

# from scipy.io import savemat
# savemat('//win.desy.de/home/carnisj/My Documents/MATLAB/TAUBIN/vertices.mat', {'V': vertices_old})
# savemat('//win.desy.de/home/carnisj/My Documents/MATLAB/TAUBIN/faces.mat', {'F': faces})

# Display resulting triangular mesh using Matplotlib.
gu.plot_3dmesh(vertices_old, faces, (nz, ny, nx), title='Mesh after marching cubes')
plt.ion()

# estimate the probability density using gaussian_kde
vertices_new, normals, areas, color, error_normals = fu.taubin_smooth(faces, vertices_old, iterations=smoothing_iterations,
                                                               lamda=smooth_lamda, mu=smooth_mu, debugging=1)
# Display resulting triangular mesh using Matplotlib.
# gu.plot_3dmesh(vertices_new, faces, (nz, ny, nx), title='Mesh after Taubin smoothing')
# plt.ion()

# fu.stereographic_proj(normals, color, reflection, flag_plotplanes=0, debugging=False)

labels, longitude_latitude = fu.equiproj_splatt_segment(normals, color, weights=areas, bw_method=my_bw_method,
                                                        background_threshold=kde_threshold,
                                                        min_distance=my_min_distance,
                                                        debugging=1)
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

###############################################
# assign back labels to voxels using vertices #
###############################################
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
                vertices_label[idx]

########################################
# save planes before refinement in vti #
########################################
fu.save_planes_vti(filename=os.path.join(savedir, "S" + str(scan) + "_planes before refinement.vti"),
                   voxel_size=(1, 1, 1), tuple_array=(amp, support), tuple_fieldnames=('amp', 'support'),
                   plane_labels=range(0, labels.max()+1, 1), planes=all_planes, amplitude_threshold=0.01)

##############################
# define a conjugate support #
##############################
# this support is 1 outside, 0 inside so that the gradient points towards exterior
support = np.ones((nz, ny, nx))
support[abs(amp) > support_threshold * abs(amp).max()] = 0
zCOM, yCOM, xCOM = center_of_mass(support)
print("COM at (z, y, x): (", str('{:.2f}'.format(zCOM)), ',', str('{:.2f}'.format(yCOM)), ',',
      str('{:.2f}'.format(xCOM)), ')')
gradz, grady, gradx = np.gradient(support, 1)  # support

######################################
# Initialize log files and .vti file #
######################################
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
amp_array = np.transpose(amp).reshape(amp.size)
amp_array = numpy_support.numpy_to_vtk(amp_array)
image_data = vtk.vtkImageData()
image_data.SetOrigin(0, 0, 0)
image_data.SetSpacing(1, 1, 1)
image_data.SetExtent(0, nz - 1, 0, ny - 1, 0, nx - 1)
pd = image_data.GetPointData()
pd.SetScalars(amp_array)
pd.GetArray(0).SetName("amp")
index_vti = 1

##################################################################
# fit points by a plane, exclude points far away, refine the fit #
##################################################################
for label in range(1, labels.max()+1, 1):  # label 0 is the background

    # raw fit including all points
    plane = np.copy(all_planes)
    plane[plane != label] = 0
    plane[plane == label] = 1
    if plane[plane == 1].sum() == 0:  # no points on the plane
        print('Raw fit: no points for plane', label)
        continue
    # TODO: why not using direclty the centroid?
    coeffs,  plane_indices, stop = fu.fit_plane(plane, label, debugging=debug)
    if stop == 1:
        print('No points remaining after raw fit for plane', label)
        continue

    # update plane
    plane, stop = fu.distance_threshold(coeffs,  plane_indices, 1, plane.shape)
    if stop == 1:  # no points on the plane
        print('Refined fit: no points for plane', label)
        continue
    else:
        print('Plane', label, ', ', str(plane[plane == 1].sum()), 'points after checking distance to plane')

    coeffs, plane_indices, stop = fu.fit_plane(plane, label, debugging=debug)
    if stop == 1:
        print('No points remaining after refined fit for plane', label)
        continue

    # update plane
    plane, stop = fu.distance_threshold(coeffs, plane_indices, 0.45, plane.shape)
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
        plane, stop = fu.grow_facet(coeffs, plane, label, debugging=debug)
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
        plane, stop = fu.grow_facet(coeffs, plane, label, debug)
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
