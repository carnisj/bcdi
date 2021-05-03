#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

from numbers import Number, Real
import numpy as np
from matplotlib import pyplot as plt
from mayavi import mlab
import pathlib
import sys
import tkinter as tk
from tkinter import filedialog
sys.path.append('D:/myscripts/bcdi/')
import bcdi.postprocessing.postprocessing_utils as pu
import bcdi.graph.graph_utils as gu
import bcdi.utils.validation as valid

helptext = """
Template for 3d isosurface figures of a real space BCDI reconstruction.

Open an npz file (reconstruction ampdispstrain.npz) and save individual figures including a length scale.
"""

scan = 1    # scan number
root_folder = "D:/data/P10_2nd_test_isosurface_Dec2020/data_nanolab/"
sample_name = "dataset_"  #
homedir = root_folder + sample_name + str(scan) + "_newpsf/result/"
savedir = homedir + "isosurfaces/mayavi/"  # saving directory
flag_support = False  # True to plot and save the support
flag_amp = True  # True to plot and save the amplitude
flag_phase = False  # True to plot and save the phase
flag_strain = False  # True to plot and save the strain
voxel_size = 5.0  # in nm, supposed isotropic
tick_spacing = 50  # for plots, in nm
field_of_view = [150, 150, 150]  # [z,y,x] in nm, can be larger than the total width (the array will be padded)
# the number of labels of mlab.axes() is an integer and is be calculated as: field_of_view[0]/tick_spacing
# therefore it is better to use an isotropic field_of_view
strain_isosurface = 0.2
strain_range = 0.0005  # for plots
phase_range = np.pi  # for plots
plot_method = 'points3d'  # 'contour3d' or 'points3d'. The support is always plotted with 'contour3d' because there is
# no contrast with 'points3d'
fig_size = (1200, 1050)  # mayavi figure size in pixels (hor, ver), leave None for the default
azimuth = [90, -90, 180, -90, 0, 0, 150]  # azimuthal angle or list of azimuthal angles for the Mayavi scene views
elevation = [90, 90, 90, 0, 180, 90, 70]  # zenith angle or list of zenith angles for the Mayavi scene views
roll = [90, -90, 0, 0, 0, 0, 0]  # roll angle or list of roll angles for the Mayavi scene views
comment = [sample_name + "_{:5d}".format(scan) + '_top', sample_name + "_{:5d}".format(scan) + '_bottom',
           sample_name + "_{:5d}".format(scan) + '_side', sample_name + "_{:5d}".format(scan) + '_revside',
           sample_name + "_{:5d}".format(scan) + '_front', sample_name + "_{:5d}".format(scan) + '_back',
           sample_name + "_{:5d}".format(scan) + '_tilt']
# list of comments used in the filename of saved figures (len(comment) must be equal to len(azimuth))
colormap = 'jet'  # colormap for the Mayavi scene of phase and strain. 'binary' is the default for the amplitude
simulated_data = False  # if yes, it will look for a field 'phase' in the reconstructed file, otherwise for field 'disp'
##########################
# end of user parameters #
##########################

#########################
# check some parameters #
#########################
if plot_method not in ['contour3d', 'points3d']:
    raise ValueError('invalid value for the parameter plot_method')

valid.valid_item(voxel_size, allowed_types=Real, min_excluded=0, name='voxel_size')
valid.valid_item(tick_spacing, allowed_types=Real, min_excluded=0, name='tick_spacing')
valid.valid_item(strain_isosurface, allowed_types=Real, min_included=0, max_included=1, name='strain_isosurface')
valid.valid_item(strain_range, allowed_types=Real, min_excluded=0, name='strain_range')
valid.valid_item(phase_range, allowed_types=Real, min_excluded=0, name='phase_range')

if isinstance(field_of_view, Number):  # convert it to a tuple
    field_of_view = (field_of_view,) * 3
valid.valid_container(field_of_view, container_types=(tuple, list), length=3, item_types=Real, min_excluded=0,
                      name='field_of_view')

if isinstance(azimuth, Number):  # convert it to a tuple
    azimuth = (azimuth,)
valid.valid_container(azimuth, container_types=(tuple, list), min_length=1, item_types=Real, name='azimuth')

if isinstance(elevation, Number):  # convert it to a tuple
    elevation = (elevation,) * len(azimuth)
valid.valid_container(elevation, container_types=(tuple, list), length=len(azimuth), item_types=Real, name='elevation')

if isinstance(roll, Number):  # convert it to a tuple
    roll = (roll,) * len(azimuth)
valid.valid_container(roll, container_types=(tuple, list), length=len(azimuth), item_types=Real, name='roll')

if isinstance(comment, str):  # convert it to a tuple
    comment = (comment,) * len(azimuth)
valid.valid_container(comment, container_types=(tuple, list), length=len(azimuth), item_types=str, name='comment')

valid.valid_container((flag_support, flag_amp, flag_phase, flag_strain, simulated_data), container_types=tuple,
                      item_types=bool, name='boolean parameters')

if fig_size is None:
    fig_size = (400, 350)
valid.valid_container(fig_size, container_types=(tuple, list), length=2, item_types=Real, min_excluded=0,
                      name='fig_size')

if not savedir.endswith('/'):
    savedir += '/'
pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)

#############
# load data #
#############
plt.ion()
root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename(initialdir=homedir, title="Select reconstruction file",
                                       filetypes=[("NPZ", "*.npz")])
npzfile = np.load(file_path)
amp = npzfile['amp']
bulk = npzfile['bulk']
if simulated_data:
    phase = npzfile['phase']
else:
    phase = npzfile['displacement']
strain = npzfile['strain']

if amp.ndim != 3:
    print('a 3D reconstruction array is expected')
    sys.exit()

amp = amp / amp.max()
amp[amp < strain_isosurface] = 0

# flip the last axis: mayavi expect xyz, but we provide downstream/upward/outboard which is not in the correct order
amp = np.flip(amp, 2)
phase = np.flip(phase, 2)
strain = np.flip(strain, 2)

numz, numy, numx = amp.shape
print(f"Initial data size: ({numz}, {numy}, {numx})")

##################################################
# pad arrays to obtain the desired field of view #
##################################################
z_pixel_FOV = int(np.rint((field_of_view[0] / voxel_size) / 2))  # half-number of pixels corresponding to the FOV
y_pixel_FOV = int(np.rint((field_of_view[1] / voxel_size) / 2))  # half-number of pixels corresponding to the FOV
x_pixel_FOV = int(np.rint((field_of_view[2] / voxel_size) / 2))  # half-number of pixels corresponding to the FOV
new_shape = [max(numz, 2*z_pixel_FOV), max(numy, 2*y_pixel_FOV), max(numx, 2*x_pixel_FOV)]
amp = pu.crop_pad(array=amp, output_shape=new_shape, debugging=False)
phase = pu.crop_pad(array=phase, output_shape=new_shape, debugging=False)
strain = pu.crop_pad(array=strain, output_shape=new_shape, debugging=False)
numz, numy, numx = amp.shape
print(f"Cropped/padded data size: ({numz}, {numy}, {numx})")

##########################################################
# set the strain and phase to NAN outside of the support #
##########################################################
support = np.zeros((numz, numy, numx))
support[np.nonzero(amp)] = 1

strain[support == 0] = np.nan
phase[support == 0] = np.nan

############################################
# create the grid and calculate the extent #
############################################
grid_z, grid_y, grid_x = np.mgrid[0:2*z_pixel_FOV*voxel_size:voxel_size,
                                  0:2*y_pixel_FOV*voxel_size:voxel_size,
                                  0:2*x_pixel_FOV*voxel_size:voxel_size]
extent = [0, 2*z_pixel_FOV*voxel_size, 0, 2*y_pixel_FOV*voxel_size, 0, 2*x_pixel_FOV*voxel_size]
# in CXI convention, z is downstream, y vertical and x outboard

#####################################
# plot 3D isosurface of the support #
#####################################
if flag_support:
    title = [comment[idx] + '_sup' for idx in range(len(comment))]
    fig, _, _ = gu.mlab_contour3d(x=grid_z, y=grid_y, z=grid_x, contours=[strain_isosurface],
                                  scalars=support[numz // 2 - z_pixel_FOV:numz // 2 + z_pixel_FOV,
                                                  numy // 2 - y_pixel_FOV:numy // 2 + y_pixel_FOV,
                                                  numx // 2 - x_pixel_FOV:numx // 2 + x_pixel_FOV],
                                  extent=extent, nb_labels=int(1+field_of_view[0]/tick_spacing),
                                  fig_size=fig_size, azimuth=azimuth, elevation=elevation, distance=3*field_of_view[0],
                                  roll=roll, title=title, vmin=0, vmax=1, opacity=1, color=(0.7, 0.7, 0.7),
                                  savedir=savedir)

    mlab.close(fig)

#######################################
# plot 3D isosurface of the amplitude #
#######################################
if flag_amp:
    title = [comment[idx] + '_amp' for idx in range(len(comment))]
    if plot_method == 'points3d':
        fig, _, _ = gu.mlab_points3d(x=grid_z, y=grid_y, z=grid_x, mode='cube',
                                     scalars=amp[numz // 2 - z_pixel_FOV:numz // 2 + z_pixel_FOV,
                                                 numy // 2 - y_pixel_FOV:numy // 2 + y_pixel_FOV,
                                                 numx // 2 - x_pixel_FOV:numx // 2 + x_pixel_FOV],
                                     extent=extent, nb_labels=int(1 + field_of_view[0] / tick_spacing),
                                     fig_size=fig_size, azimuth=azimuth, elevation=elevation,
                                     distance=3 * field_of_view[0], roll=roll, title=title, vmin=0, vmax=1,
                                     opacity=1, colormap='binary', savedir=savedir)
    else:  # 'contour3d'
        contours = list(np.linspace(strain_isosurface, 1, num=10, endpoint=True))
        fig, _, _ = gu.mlab_contour3d(x=grid_z, y=grid_y, z=grid_x, contours=contours,
                                      scalars=amp[numz // 2 - z_pixel_FOV:numz // 2 + z_pixel_FOV,
                                                  numy // 2 - y_pixel_FOV:numy // 2 + y_pixel_FOV,
                                                  numx // 2 - x_pixel_FOV:numx // 2 + x_pixel_FOV],
                                      extent=extent, nb_labels=int(1 + field_of_view[0] / tick_spacing),
                                      fig_size=fig_size, azimuth=azimuth, elevation=elevation,
                                      distance=3 * field_of_view[0], roll=roll, title=title, vmin=0, vmax=1,
                                      opacity=1, colormap='binary', savedir=savedir)

    mlab.close(fig)

###################################
# plot 3D isosurface of the phase #
###################################
if flag_phase:
    title = [comment[idx] + '_phase' for idx in range(len(comment))]
    if plot_method == 'points3d':
        fig, _, _ = gu.mlab_points3d(x=grid_z, y=grid_y, z=grid_x, mode='cube',
                                     scalars=phase[numz // 2 - z_pixel_FOV:numz // 2 + z_pixel_FOV,
                                                   numy // 2 - y_pixel_FOV:numy // 2 + y_pixel_FOV,
                                                   numx // 2 - x_pixel_FOV:numx // 2 + x_pixel_FOV],
                                     extent=extent, nb_labels=int(1 + field_of_view[0] / tick_spacing),
                                     fig_size=fig_size, azimuth=azimuth, elevation=elevation,
                                     distance=3 * field_of_view[0], roll=roll, title=title, vmin=-phase_range,
                                     vmax=phase_range, opacity=1, colormap=colormap, savedir=savedir)
    else:  # 'contour3d'
        contours = list(np.linspace(-phase_range, phase_range, num=50, endpoint=True))
        fig, _, _ = gu.mlab_contour3d(x=grid_z, y=grid_y, z=grid_x, contours=contours,
                                      scalars=phase[numz // 2 - z_pixel_FOV:numz // 2 + z_pixel_FOV,
                                                    numy // 2 - y_pixel_FOV:numy // 2 + y_pixel_FOV,
                                                    numx // 2 - x_pixel_FOV:numx // 2 + x_pixel_FOV],
                                      extent=extent, nb_labels=int(1 + field_of_view[0] / tick_spacing),
                                      fig_size=fig_size, azimuth=azimuth, elevation=elevation,
                                      distance=3 * field_of_view[0], roll=roll, title=title, vmin=-phase_range,
                                      vmax=phase_range, opacity=1, colormap=colormap, savedir=savedir)

    mlab.close(fig)

####################################
# plot 3D isosurface of the strain #
####################################
if flag_strain:
    title = [comment[idx] + '_strain' for idx in range(len(comment))]
    if plot_method == 'points3d':
        fig, _, _ = gu.mlab_points3d(x=grid_z, y=grid_y, z=grid_x, mode='cube',
                                     scalars=strain[numz // 2 - z_pixel_FOV:numz // 2 + z_pixel_FOV,
                                                    numy // 2 - y_pixel_FOV:numy // 2 + y_pixel_FOV,
                                                    numx // 2 - x_pixel_FOV:numx // 2 + x_pixel_FOV],
                                     extent=extent, nb_labels=int(1 + field_of_view[0] / tick_spacing),
                                     fig_size=fig_size, azimuth=azimuth, elevation=elevation,
                                     distance=3 * field_of_view[0], roll=roll, title=title, vmin=-strain_range,
                                     vmax=strain_range, opacity=1, colormap=colormap, savedir=savedir)
    else:  # 'contour3d'
        contours = list(np.linspace(-strain_range, strain_range, num=50, endpoint=True))
        fig, _, _ = gu.mlab_contour3d(x=grid_z, y=grid_y, z=grid_x, contours=contours,
                                      scalars=strain[numz // 2 - z_pixel_FOV:numz // 2 + z_pixel_FOV,
                                                     numy // 2 - y_pixel_FOV:numy // 2 + y_pixel_FOV,
                                                     numx // 2 - x_pixel_FOV:numx // 2 + x_pixel_FOV],
                                      extent=extent, nb_labels=int(1 + field_of_view[0] / tick_spacing),
                                      fig_size=fig_size, azimuth=azimuth, elevation=elevation,
                                      distance=3 * field_of_view[0], roll=roll, title=title, vmin=-strain_range,
                                      vmax=strain_range, opacity=1, colormap=colormap, savedir=savedir)

    mlab.close(fig)
