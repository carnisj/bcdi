# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import pathlib
import numpy as np
from matplotlib import pyplot as plt
from mayavi import mlab
import tkinter as tk
from tkinter import filedialog
import sys
sys.path.append('D:/myscripts/bcdi/')
import bcdi.postprocessing.postprocessing_utils as pu

helptext = """
Template for 3d isosurface figures of a real space BCDI reconstruction.

Open an npz file (reconstruction ampdispstrain.npz) and save individual figures including a length scale.
"""

scan = 1138    # scan number
root_folder = 'D:/data/P10_OER/analysis/candidate_12/dewet2_2_S1484_to_S1511/'
sample_name = "dewet2_2"  #
homedir = root_folder  # + sample_name + str(scan) + '/pynxraw/'
# homedir = root_folder + sample_name
comment = sample_name + "_{:5d}".format(scan)
flag_support = True  # True to plot and save the support
flag_amp = True  # True to plot and save the amplitude
flag_phase = True  # True to plot and save the phase
flag_strain = True  # True to plot and save the strain
voxel_size = 6.0  # in nm, supposed isotropic
tick_spacing = 50  # for plots, in nm
field_of_view = [500, 500, 500]  # [z,y,x] in nm, can be larger than the total width (the array will be padded)
# the number of labels of mlab.axes() is an integer and is be calculated as: field_of_view[0]/tick_spacing
# therefore it is better to use an isotropic field_of_view
strain_isosurface = 0.45
strain_range = 0.002  # for plots
phase_range = np.pi  # for plots
plot_method = 'points3d'  # 'contour3d' or 'points3d'. The support is always plotted with 'contour3d' because there is
# no contrast with 'points3d'
fig_size = (1200, 1050)  # mayavi figure size in pixels (hor, ver), leave None for the default
simulated_data = False  # if yes, it will look for a field 'phase' in the reconstructed file, otherwise for field 'disp'
##########################
# end of user parameters #
##########################

#################################################################
# check few parameters and create the folder for saving results #
#################################################################
assert plot_method in ['contour3d', 'points3d'], 'invalid value for the parameter plot_method'
if fig_size is None:
    fig_size = (400, 350)

savedir = homedir + "isosurfaces/"
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

amp = np.flip(amp, 2)  # mayavi expect xyz, but we provide downstream/upward/outboard which is not in the correct order
phase = np.flip(phase, 2)
strain = np.flip(strain, 2)

numz, numy, numx = amp.shape
print("Initial data size: (", numz, ',', numy, ',', numx, ')')

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
print("Cropped/padded data size: (", numz, ',', numy, ',', numx, ')')

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
    fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=fig_size)
    mlab.contour3d(grid_z, grid_y, grid_x, support[numz // 2 - z_pixel_FOV:numz // 2 + z_pixel_FOV,
                                                   numy // 2 - y_pixel_FOV:numy // 2 + y_pixel_FOV,
                                                   numx // 2 - x_pixel_FOV:numx // 2 + x_pixel_FOV],
                   contours=[strain_isosurface], color=(0.7, 0.7, 0.7))

    # top view
    mlab.view(azimuth=90, elevation=90, distance=3*field_of_view[0])
    # azimut is the rotation around z axis of mayavi (x)
    mlab.roll(90)
    ax = mlab.axes(extent=extent, line_width=2.0, nb_labels=int(1+field_of_view[0]/tick_spacing))
    ax.label_text_property.opacity = 1.0
    ax.title_text_property.opacity = 1.0
    mlab.savefig(savedir + comment + '_sup_top_labels.png', figure=fig)
    ax.label_text_property.opacity = 0.0
    ax.title_text_property.opacity = 0.0
    mlab.savefig(savedir + comment + '_sup_top.png', figure=fig)

    # side view
    mlab.view(azimuth=180, elevation=90, distance=3*field_of_view[0])
    # azimut is the rotation around z axis of mayavi (x)
    mlab.roll(0)
    ax.label_text_property.opacity = 1.0
    ax.title_text_property.opacity = 1.0
    mlab.savefig(savedir + comment + '_sup_side_labels.png', figure=fig)
    ax.label_text_property.opacity = 0.0
    ax.title_text_property.opacity = 0.0
    mlab.savefig(savedir + comment + '_sup_side.png', figure=fig)

    # front view
    mlab.view(azimuth=0, elevation=180, distance=3*field_of_view[0])
    # azimut is the rotation around z axis of mayavi (x)
    mlab.roll(0)
    ax.label_text_property.opacity = 1.0
    ax.title_text_property.opacity = 1.0
    mlab.savefig(savedir + comment + '_sup_front_labels.png', figure=fig)
    ax.label_text_property.opacity = 0.0
    ax.title_text_property.opacity = 0.0
    mlab.savefig(savedir + comment + '_sup_front.png', figure=fig)

    # perspective view
    mlab.view(azimuth=150, elevation=70, distance=3*field_of_view[0])
    # azimut is the rotation around z axis of mayavi (x)
    mlab.roll(0)
    ax.label_text_property.opacity = 1.0
    ax.title_text_property.opacity = 1.0
    mlab.savefig(savedir + comment + '_sup_tilt_labels.png', figure=fig)
    ax.label_text_property.opacity = 0.0
    ax.title_text_property.opacity = 0.0
    mlab.savefig(savedir + comment + '_sup_tilt.png', figure=fig)

    mlab.close(fig)

#######################################
# plot 3D isosurface of the amplitude #
#######################################
if flag_amp:
    fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=fig_size)
    if plot_method == 'points3d':
        mlab.points3d(grid_z, grid_y, grid_x, amp[numz // 2 - z_pixel_FOV:numz // 2 + z_pixel_FOV,
                                                  numy // 2 - y_pixel_FOV:numy // 2 + y_pixel_FOV,
                                                  numx // 2 - x_pixel_FOV:numx // 2 + x_pixel_FOV],
                      mode='cube', opacity=1, vmin=0, vmax=1,  colormap='jet')
    else:  # 'contour3d'
        contours = list(np.linspace(strain_isosurface, 1, num=10, endpoint=True))
        mlab.contour3d(grid_z, grid_y, grid_x, amp[numz // 2 - z_pixel_FOV:numz // 2 + z_pixel_FOV,
                                                   numy // 2 - y_pixel_FOV:numy // 2 + y_pixel_FOV,
                                                   numx // 2 - x_pixel_FOV:numx // 2 + x_pixel_FOV],
                       contours=contours, opacity=1, vmin=0, vmax=1, colormap='jet')

    # top view
    mlab.view(azimuth=90, elevation=90, distance=3*field_of_view[0])
    # azimut is the rotation around z axis of mayavi (x)
    mlab.roll(90)
    ax = mlab.axes(extent=extent, line_width=2.0, nb_labels=int(1+field_of_view[0]/tick_spacing))
    cbar = mlab.colorbar(orientation='vertical')
    ax.label_text_property.opacity = 1.0
    ax.title_text_property.opacity = 1.0
    mlab.savefig(savedir + comment + '_amp_top_labels.png', figure=fig)
    cbar.visible = False
    ax.label_text_property.opacity = 0.0
    ax.title_text_property.opacity = 0.0
    mlab.savefig(savedir + comment + '_amp_top.png', figure=fig)

    # side view
    mlab.view(azimuth=180, elevation=90, distance=3*field_of_view[0])
    # azimut is the rotation around z axis of mayavi (x)
    mlab.roll(0)
    cbar.visible = True
    ax.label_text_property.opacity = 1.0
    ax.title_text_property.opacity = 1.0
    mlab.savefig(savedir + comment + '_amp_side_labels.png', figure=fig)
    cbar.visible = False
    ax.label_text_property.opacity = 0.0
    ax.title_text_property.opacity = 0.0
    mlab.savefig(savedir + comment + '_amp_side.png', figure=fig)

    # front view
    mlab.view(azimuth=0, elevation=180, distance=3*field_of_view[0])
    # azimut is the rotation around z axis of mayavi (x)
    mlab.roll(0)
    cbar.visible = True
    ax.label_text_property.opacity = 1.0
    ax.title_text_property.opacity = 1.0
    mlab.savefig(savedir + comment + '_amp_front_labels.png', figure=fig)
    cbar.visible = False
    ax.label_text_property.opacity = 0.0
    ax.title_text_property.opacity = 0.0
    mlab.savefig(savedir + comment + '_amp_front.png', figure=fig)

    # perspective view
    mlab.view(azimuth=150, elevation=70, distance=3*field_of_view[0])
    # azimut is the rotation around z axis of mayavi (x)
    mlab.roll(0)
    cbar.visible = True
    ax.label_text_property.opacity = 1.0
    ax.title_text_property.opacity = 1.0
    mlab.savefig(savedir + comment + '_amp_tilt_labels.png', figure=fig)
    cbar.visible = False
    ax.label_text_property.opacity = 0.0
    ax.title_text_property.opacity = 0.0
    mlab.savefig(savedir + comment + '_amp_tilt.png', figure=fig)

    mlab.close(fig)

###################################
# plot 3D isosurface of the phase #
###################################
if flag_phase:
    fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=fig_size)
    if plot_method == 'points3d':
        mlab.points3d(grid_z, grid_y, grid_x, phase[numz // 2 - z_pixel_FOV:numz // 2 + z_pixel_FOV,
                                                    numy // 2 - y_pixel_FOV:numy // 2 + y_pixel_FOV,
                                                    numx // 2 - x_pixel_FOV:numx // 2 + x_pixel_FOV],
                      mode='cube', opacity=1, vmin=-phase_range, vmax=phase_range, colormap='jet')
    else:  # 'contour3d'
        contours = list(np.linspace(-phase_range, phase_range, num=50, endpoint=True))
        mlab.contour3d(grid_z, grid_y, grid_x, phase[numz // 2 - z_pixel_FOV:numz // 2 + z_pixel_FOV,
                                                     numy // 2 - y_pixel_FOV:numy // 2 + y_pixel_FOV,
                                                     numx // 2 - x_pixel_FOV:numx // 2 + x_pixel_FOV],
                       contours=contours, opacity=1, vmin=-phase_range, vmax=phase_range, colormap='jet')

    # top view
    mlab.view(azimuth=90, elevation=90, distance=3*field_of_view[0])
    # azimut is the rotation around z axis of mayavi (x)
    mlab.roll(90)
    ax = mlab.axes(extent=extent, line_width=2.0, nb_labels=int(1+field_of_view[0]/tick_spacing))
    cbar = mlab.colorbar(orientation='vertical')
    ax.label_text_property.opacity = 1.0
    ax.title_text_property.opacity = 1.0
    mlab.savefig(savedir + comment + '_phase_top_labels.png', figure=fig)
    cbar.visible = False
    ax.label_text_property.opacity = 0.0
    ax.title_text_property.opacity = 0.0
    mlab.savefig(savedir + comment + '_phase_top.png', figure=fig)

    # side view
    mlab.view(azimuth=180, elevation=90, distance=3*field_of_view[0])
    # azimut is the rotation around z axis of mayavi (x)
    mlab.roll(0)
    cbar.visible = True
    ax.label_text_property.opacity = 1.0
    ax.title_text_property.opacity = 1.0
    mlab.savefig(savedir + comment + '_phase_side_labels.png', figure=fig)
    cbar.visible = False
    ax.label_text_property.opacity = 0.0
    ax.title_text_property.opacity = 0.0
    mlab.savefig(savedir + comment + '_phase_side.png', figure=fig)

    # front view
    mlab.view(azimuth=0, elevation=180, distance=3*field_of_view[0])
    # azimut is the rotation around z axis of mayavi (x)
    mlab.roll(0)
    cbar.visible = True
    ax.label_text_property.opacity = 1.0
    ax.title_text_property.opacity = 1.0
    mlab.savefig(savedir + comment + '_phase_front_labels.png', figure=fig)
    cbar.visible = False
    ax.label_text_property.opacity = 0.0
    ax.title_text_property.opacity = 0.0
    mlab.savefig(savedir + comment + '_phase_front.png', figure=fig)

    # perspective view
    mlab.view(azimuth=150, elevation=70, distance=3*field_of_view[0])
    # azimut is the rotation around z axis of mayavi (x)
    mlab.roll(0)
    cbar.visible = True
    ax.label_text_property.opacity = 1.0
    ax.title_text_property.opacity = 1.0
    mlab.savefig(savedir + comment + '_phase_tilt_labels.png', figure=fig)
    cbar.visible = False
    ax.label_text_property.opacity = 0.0
    ax.title_text_property.opacity = 0.0
    mlab.savefig(savedir + comment + '_phase_tilt.png', figure=fig)

    mlab.close(fig)

####################################
# plot 3D isosurface of the strain #
####################################
if flag_strain:
    fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=fig_size)
    if plot_method == 'points3d':
        mlab.points3d(grid_z, grid_y, grid_x, strain[numz // 2 - z_pixel_FOV:numz // 2 + z_pixel_FOV,
                                                     numy // 2 - y_pixel_FOV:numy // 2 + y_pixel_FOV,
                                                     numx // 2 - x_pixel_FOV:numx // 2 + x_pixel_FOV],
                      mode='cube', opacity=1, vmin=-strain_range, vmax=strain_range, colormap='jet')
    else:  # 'contour3d'
        contours = list(np.linspace(-strain_range, strain_range, num=50, endpoint=True))
        mlab.contour3d(grid_z, grid_y, grid_x, strain[numz // 2 - z_pixel_FOV:numz // 2 + z_pixel_FOV,
                                                      numy // 2 - y_pixel_FOV:numy // 2 + y_pixel_FOV,
                                                      numx // 2 - x_pixel_FOV:numx // 2 + x_pixel_FOV],
                       contours=contours, opacity=1, vmin=-strain_range, vmax=strain_range, colormap='jet')

    # top view
    mlab.view(azimuth=90, elevation=90, distance=3*field_of_view[0])
    # azimut is the rotation around z axis of mayavi (x)
    mlab.roll(90)
    ax = mlab.axes(extent=extent, line_width=2.0, nb_labels=int(1+field_of_view[0]/tick_spacing))
    cbar = mlab.colorbar(orientation='vertical')
    ax.label_text_property.opacity = 1.0
    ax.title_text_property.opacity = 1.0
    mlab.savefig(savedir + comment + '_strain_top_labels.png', figure=fig)
    cbar.visible = False
    ax.label_text_property.opacity = 0.0
    ax.title_text_property.opacity = 0.0
    mlab.savefig(savedir + comment + '_strain_top.png', figure=fig)

    # side view
    mlab.view(azimuth=180, elevation=90, distance=3*field_of_view[0])
    # azimut is the rotation around z axis of mayavi (x)
    mlab.roll(0)
    cbar.visible = True
    ax.label_text_property.opacity = 1.0
    ax.title_text_property.opacity = 1.0
    mlab.savefig(savedir + comment + '_strain_side_labels.png', figure=fig)
    cbar.visible = False
    ax.label_text_property.opacity = 0.0
    ax.title_text_property.opacity = 0.0
    mlab.savefig(savedir + comment + '_strain_side.png', figure=fig)

    # front view
    mlab.view(azimuth=0, elevation=180, distance=3*field_of_view[0])
    # azimut is the rotation around z axis of mayavi (x)
    mlab.roll(0)
    cbar.visible = True
    ax.label_text_property.opacity = 1.0
    ax.title_text_property.opacity = 1.0
    mlab.savefig(savedir + comment + '_strain_front_labels.png', figure=fig)
    cbar.visible = False
    ax.label_text_property.opacity = 0.0
    ax.title_text_property.opacity = 0.0
    mlab.savefig(savedir + comment + '_strain_front.png', figure=fig)

    # perspective view
    mlab.view(azimuth=150, elevation=70, distance=3*field_of_view[0])
    # azimut is the rotation around z axis of mayavi (x)
    mlab.roll(0)
    cbar.visible = True
    ax.label_text_property.opacity = 1.0
    ax.title_text_property.opacity = 1.0
    mlab.savefig(savedir + comment + '_strain_tilt_labels.png', figure=fig)
    cbar.visible = False
    ax.label_text_property.opacity = 0.0
    ax.title_text_property.opacity = 0.0
    mlab.savefig(savedir + comment + '_strain_tilt.png', figure=fig)

    mlab.close(fig)
