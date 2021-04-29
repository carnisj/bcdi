#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np
import moviepy.editor as mpy
from mayavi import mlab
import pathlib
import tkinter as tk
from tkinter import filedialog
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage.measurements import center_of_mass
from traits.api import push_exception_handler
import gc
import sys
sys.path.append('D:/myscripts/bcdi/')
import bcdi.utils.utilities as util
import bcdi.graph.graph_utils as gu
import bcdi.postprocessing.postprocessing_utils as pu

helptext = """
Template for 3d isosurface figures of a diffraction pattern.
The data will be interpolated on an isotropic range in order to fulfill the desired tick spacing.
Note that qy values (axis 2) are opposite to the correct ones, because there is no easy way to flip an axis in mayvi.
The diffraction pattern is supposed to be in an orthonormal frame and q values need to be provided.
Optionally creates a movie from a 3D real space reconstruction in each direction. This requires moviepy.
"""

scan = 1    # spec scan number
root_folder = "D:/data/P10_2nd_test_isosurface_Dec2020/data_nanolab/"
sample_name = "dataset_"
homedir = root_folder + sample_name + str(scan) + "_newpsf/result/"
savedir = homedir + '3D_diffpattern/'  # saving directory
comment = ""  # should start with _
binning = (2, 2, 2)  # binning for the measured diffraction pattern in each dimension
geometry = 'Bragg'  # 'SAXS' or 'Bragg'
crop_symmetric = False  # if True, will crop the data ot the largest symmetrical range around the direct beam
# (geometry = 'SAXS') or the Brapp peak (geometry = 'Bragg')
tick_spacing = 0.05  # in 1/nm, spacing between ticks
contours = [-0.1, 0.5, 1.5, 3]  # contours for the isosurface in log scale.
# contours = [3.6, 4.05, 4.5, 4.95, 5.4]  # gold_2_2_2_00022
fig_size = (500, 500)  # figure size in pixels (horizontal, vertical)
distance = 0.5  # distance of the camera in q, leave None for default
debug = True  # True to see contour plots for debugging
##########################
# settings for the movie #
##########################
make_movie = False  # True to save a movie
duration = 10  # duration of the movie in s
frame_per_second = 20  # number of frames per second, there will be duration*frame_per_second frames in total
output_format = 'mp4'  # 'gif', 'mp4' or None for no movie
##########################
# end of user parameters #
##########################


def rotate_scene(t):
    """
    Rotate the camera of the mayavi scene at time t.

    :param t: time in the range [0, duration]
    :return: a screenshot of the scene
    """
    mlab.view(azimuth=360/duration*t, elevation=63, distance=distance)
    return mlab.screenshot(figure=myfig, mode='rgb', antialiased=True)  # return a RGB image


#########################
# check some parameters #
#########################
if not savedir.endswith('/'):
    savedir += '/'
pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)

#############
# load data #
#############
push_exception_handler(reraise_exceptions=True)  # force exceptions to be re-raised in Traits
root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename(initialdir=homedir, title="Select the diffraction pattern",
                                       filetypes=[("NPZ", "*.npz")])
data, _ = util.load_file(file_path)
assert data.ndim == 3, 'data should be a 3D array'

nz, ny, nx = data.shape
print("Initial data shape: ", nz, ny, nx)

file_path = filedialog.askopenfilename(initialdir=homedir, title="Select q values",
                                       filetypes=[("NPZ", "*.npz")])
q_values = np.load(file_path)
qx = q_values['qx']  # 1D array
qy = q_values['qy']  # 1D array
qz = q_values['qz']  # 1D array

############
# bin data #
############
qx = qx[:nz - (nz % binning[0]):binning[0]]
qz = qz[:ny - (ny % binning[1]):binning[1]]
qy = qy[:nx - (nx % binning[2]):binning[2]]
data = pu.bin_data(data, (binning[0], binning[1], binning[2]), debugging=False)
print('Diffraction data shape after binning', data.shape)
nz, ny, nx = data.shape

##########################################
# take the largest data symmetrical in q #
##########################################
if crop_symmetric:
    if geometry == 'SAXS':
        # the reference if the center of reciprocal space at qx=qy=qz=0
        qx_com, qz_com, qy_com = 0, 0, 0
    elif geometry == 'Bragg':
        # find the position of the Bragg peak
        zcom, ycom, xcom = center_of_mass(data)
        zcom, ycom, xcom = int(np.rint(zcom)), int(np.rint(ycom)), int(np.rint(xcom))
        print('Center of mass of the diffraction pattern at pixel:', zcom, ycom, xcom)
        qx_com, qz_com, qy_com = qx[zcom], qz[ycom], qy[xcom]
    else:
        raise ValueError('supported geometry: "SAXS" or "Bragg"')

    min_range = min(min(abs(qx.min()-qx_com,), qx.max()-qx_com,),
                    min(abs(qz.min()-qz_com), qz.max()-qz_com),
                    min(abs(qy.min()-qy_com), qy.max()-qy_com))
    indices_qx = np.argwhere(abs(qx-qx_com) < min_range)[:, 0]
    indices_qz = np.argwhere(abs(qz-qz_com) < min_range)[:, 0]
    indices_qy = np.argwhere(abs(qy-qy_com) < min_range)[:, 0]

    qx = qx[indices_qx]
    qz = qz[indices_qz]
    qy = qy[indices_qy]
    data = data[indices_qx.min():indices_qx.max()+1,
                indices_qz.min():indices_qz.max()+1,
                indices_qy.min():indices_qy.max()+1]
    nz, ny, nx = data.shape
    print("Shape of the largest symmetrical dataset:", nz, ny, nx)
    del indices_qx, indices_qy, indices_qz

    ##############################################################
    # interpolate the data to have ticks at the desired location #
    ##############################################################
    rgi = RegularGridInterpolator((np.linspace(qx.min(), qx.max(), num=nz),
                                   np.linspace(qz.min(), qz.max(), num=ny),
                                   np.linspace(qy.min(), qy.max(), num=nx)),
                                  data, method='linear', bounds_error=False, fill_value=0)

    half_labels = int(min_range // tick_spacing)  # the number of labels is (2*half_labels+1)

    new_z, new_y, new_x = np.meshgrid(
        np.linspace(qx_com-tick_spacing*half_labels, qx_com+tick_spacing*half_labels, num=nz),
        np.linspace(qz_com-tick_spacing*half_labels, qz_com + tick_spacing*half_labels, num=ny),
        np.linspace(qy_com-tick_spacing*half_labels, qy_com+tick_spacing*half_labels, num=nx),
        indexing='ij')

    data = rgi(np.concatenate((new_z.reshape((1, new_z.size)),
                               new_y.reshape((1, new_z.size)),
                               new_x.reshape((1, new_z.size)))).transpose())
    data = data.reshape((nz, ny, nx))
    print('Interpolation done')
    del new_z, new_y, new_x, rgi
    gc.collect()

if debug:
    gu.contour_slices(data, (qx, qz, qy), sum_frames=True, title='data',
                      levels=np.linspace(0, np.ceil(np.log10(data.max())), 150, endpoint=True),
                      plot_colorbar=True, scale='log', is_orthogonal=True, reciprocal_space=True)

###################
# filter out nans #
###################
data[np.isnan(data)] = 1e-20
data[data == 0] = 1e-20

#########################################
# plot 3D isosurface (perspective view) #
#########################################
data = np.flip(data, 2)  # mayavi expects xyz, data order is downstream/upward/outboard
if crop_symmetric:
    grid_qx, grid_qz, grid_qy = np.mgrid[qx_com-tick_spacing*half_labels:qx_com+tick_spacing*half_labels:1j * nz,
                                         qz_com-tick_spacing*half_labels:qz_com+tick_spacing*half_labels:1j * ny,
                                         qy_com-tick_spacing*half_labels:qy_com+tick_spacing*half_labels:1j * nx]
else:
    grid_qx, grid_qz, grid_qy = np.mgrid[qx.min():qx.max():1j*nz,
                                         qz.min():qz.max():1j*ny,
                                         qy.min():qy.max():1j*nx]

# in CXI convention, z is downstream, y vertical and x outboard
# for q: classical convention qx downstream, qz vertical and qy outboard
myfig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=fig_size)
mlab.contour3d(grid_qx, grid_qz, grid_qy, np.log10(data), contours=contours, opacity=0.2, colormap='hsv',
               vmin=0, vmax=np.ceil(np.log10(data.max())))
if distance:
    mlab.view(azimuth=38, elevation=63, distance=distance)
else:
    mlab.view(azimuth=38, elevation=63, distance=np.sqrt(qx.max()**2+qz.max()**2+qy.max()**2))

# azimut is the rotation around z axis of mayavi (x)
mlab.roll(0)

if crop_symmetric:
    ax = mlab.axes(line_width=2.0, nb_labels=2*half_labels+1)
else:
    ax = mlab.axes(line_width=2.0, nb_labels=4)

mlab.xlabel('Qx (1/nm)')
mlab.ylabel('Qz (1/nm)')
mlab.zlabel('-Qy (1/nm)')
mlab.savefig(savedir + 'S' + str(scan) + comment + '_labels.png', figure=myfig)
ax.label_text_property.opacity = 0.0
ax.title_text_property.opacity = 0.0
mlab.savefig(savedir + 'S' + str(scan) + comment + '_axes.png', figure=myfig)
ax.axes.x_axis_visibility = False
ax.axes.y_axis_visibility = False
ax.axes.z_axis_visibility = False
mlab.savefig(savedir + 'S' + str(scan) + comment + '.png', figure=myfig)
mlab.draw(myfig)

if make_movie:
    if output_format == 'mp4':
        animation = mpy.VideoClip(rotate_scene, duration=duration).resize(width=fig_size[0], height=fig_size[1])
        fname = savedir + "S" + str(scan) + "_movie.mp4"
        animation.write_videofile(fname, fps=frame_per_second)
    elif output_format == 'gif':
        animation = mpy.VideoClip(rotate_scene, duration=duration).resize(width=fig_size[0], height=fig_size[1])
        fname = savedir + "S" + str(scan) + "_movie.gif"
        animation.write_gif(fname, fps=frame_per_second)

mlab.show()

