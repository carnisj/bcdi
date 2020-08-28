# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from scipy.interpolate import RegularGridInterpolator
import gc
import sys
sys.path.append('D:/myscripts/bcdi/')
import bcdi.postprocessing.postprocessing_utils as pu
import bcdi.graph.graph_utils as gu
import bcdi.algorithms.algorithms_utils as au
import bcdi.utils.utilities as util

helptext = """
Create a support from a reconstruction, using the indicated threshold.
The support can be cropped/padded to a desired shape.
In real space the CXI convention is used: z downstream, y vertical up, x outboard.
In reciprocal space, the following convention is used: qx downtream, qz vertical up, qy outboard

"""

root_folder = "D:/data/P10_August2020_CDI/data/gold_trunc_custom/"
support_threshold = 0.05  # in % of the normalized absolute value
original_shape = [500, 500, 500]  # shape of the array used for phasing and finding the support (after binning_original)
binning_pynx = (1, 1, 1)  # binning that was used in PyNX during phasing
output_shape = [500, 500, 500]  # shape of the array for later phasing (before binning_output)
# if the data and q-values were binned beforehand, use the binned shape and binning_output=(1,1,1)
binning_output = (1, 1, 1)  # binning that will be used in PyNX for later phasing
flag_interact = True  # if False, will skip thresholding and masking
filter_name = 'gaussian_highpass'  # apply a filtering kernel to the support, 'do_nothing' or 'gaussian_highpass'
gaussian_sigma = 3.0  # sigma of the gaussian filter
binary_support = True  # True to save the support as an array of 0 and 1
reload_support = False  # if True, will load the support which shape is assumed to be the shape after binning_output
# it is usefull to redo some masking without interpolating again.
is_ortho = True  # True if the data is already orthogonalized
center = True  # will center the support based on the center of mass
flip_reconstruction = False  # True if you want to get the conjugate object
roll_modes = (0, 0, 0)  # correct a roll of few pixels after the decomposition into modes in PyNX. axis=(0, 1, 2)
roll_centering = (0, 0, 0)  # roll applied after masking when centering by center of mass is not optimal axis=(0, 1, 2)
background_plot = '0.5'  # in level of grey in [0,1], 0 being dark. For visual comfort during masking
save_fig = True  # if True, will save the figure of the final support
comment = ''  # should start with _
###############################################################################################
# parameters used when (original_shape*binning_original != output_shape) and (is_ortho=False) #
###############################################################################################
energy = 10235  # in eV
tilt_angle = 0.5  # in degrees
distance = 5  # in m
pixel_x = 75e-06  # in m
pixel_y = 75e-06  # in m
######################################################################
# parameters for image deconvolution using Richardson-Lucy algorithm #
######################################################################
psf_iterations = 0  # number of iterations of Richardson-Lucy deconvolution, leave it to 0 if unwanted
psf_shape = (10, 10, 10)
psf = pu.gaussian_window(window_shape=psf_shape, sigma=0.3, mu=0.0, debugging=False)
##################################
# end of user-defined parameters #
##################################
###################
# define colormap #
###################
colormap = gu.Colormap()
my_cmap = colormap.cmap


def close_event(event):
    """
    This function handles closing events on plots.

    :return: nothing
    """
    print(event, 'Click on the figure instead of closing it!')
    sys.exit()


def on_click(event):
    """
    Function to interact with a plot, return the position of clicked pixel. If flag_pause==1 or
    if the mouse is out of plot axes, it will not register the click

    :param event: mouse click event
    """
    global xy, flag_pause, previous_axis
    if not event.inaxes:
        return
    if not flag_pause:

        if (previous_axis == event.inaxes) or (previous_axis is None):  # collect points
            _x, _y = int(np.rint(event.xdata)), int(np.rint(event.ydata))
            xy.append([_x, _y])
            if previous_axis is None:
                previous_axis = event.inaxes
        else:  # the click is not in the same subplot, restart collecting points
            print('Please select mask polygon vertices within the same subplot: restart masking...')
            xy = []
            previous_axis = None


def press_key(event):
    """
    Interact with a plot for masking parasitic diffraction intensity or detector gaps

    :param event: button press event
    """
    global original_data, updated_mask, data, mask, frame_index, width, flag_aliens, flag_mask, flag_pause
    global xy, fig_mask, max_colorbar, ax0, ax1, ax2, previous_axis, info_text, is_ortho, my_cmap

    try:
        if event.inaxes == ax0:
            dim = 0
            inaxes = True
        elif event.inaxes == ax1:
            dim = 1
            inaxes = True
        elif event.inaxes == ax2:
            dim = 2
            inaxes = True
        else:
            dim = -1
            inaxes = False

        if inaxes:
            invert_yaxis = is_ortho
            if flag_aliens:
                data, mask, width, max_colorbar, frame_index, stop_masking = \
                    gu.update_aliens_combined(key=event.key, pix=int(np.rint(event.xdata)),
                                              piy=int(np.rint(event.ydata)), original_data=original_data,
                                              original_mask=original_mask, updated_data=data, updated_mask=mask,
                                              axes=(ax0, ax1, ax2), width=width, dim=dim, frame_index=frame_index,
                                              vmin=0, vmax=max_colorbar, cmap=my_cmap, invert_yaxis=invert_yaxis)
            elif flag_mask:
                if previous_axis == ax0:
                    click_dim = 0
                    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
                    points = np.stack((x.flatten(), y.flatten()), axis=0).T
                elif previous_axis == ax1:
                    click_dim = 1
                    x, y = np.meshgrid(np.arange(nx), np.arange(nz))
                    points = np.stack((x.flatten(), y.flatten()), axis=0).T
                elif previous_axis == ax2:
                    click_dim = 2
                    x, y = np.meshgrid(np.arange(ny), np.arange(nz))
                    points = np.stack((x.flatten(), y.flatten()), axis=0).T
                else:
                    click_dim = None
                    points = None

                data, updated_mask, flag_pause, xy, width, max_colorbar, click_dim, stop_masking, info_text = \
                    gu.update_mask_combined(key=event.key, pix=int(np.rint(event.xdata)),
                                            piy=int(np.rint(event.ydata)), original_data=original_data,
                                            original_mask=mask, updated_data=data, updated_mask=updated_mask,
                                            axes=(ax0, ax1, ax2), flag_pause=flag_pause, points=points,
                                            xy=xy, width=width, dim=dim, click_dim=click_dim, info_text=info_text,
                                            vmin=0, vmax=max_colorbar, cmap=my_cmap, invert_yaxis=invert_yaxis)

                if click_dim is None:
                    previous_axis = None
            else:
                stop_masking = False

            if stop_masking:
                plt.close(fig_mask)

    except AttributeError:  # mouse pointer out of axes
        pass


###############################################
plt.rcParams["keymap.fullscreen"] = [""]

root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(initialdir=root_folder, title="Select the reconstruction",
                                       filetypes=[("HDF5", "*.h5"), ("NPZ", "*.npz"), ("CXI", "*.cxi")])
data, _ = util.load_file(file_path)
nz, ny, nx = data.shape
data = np.roll(data, roll_modes, axis=(0, 1, 2))

if flip_reconstruction:
    data = pu.flip_reconstruction(data, debugging=True)

data = abs(data)  # take the real part

if flag_interact:
    data = data / data.max(initial=None)  # normalize
    data[data < support_threshold] = 0

    fig, _, _ = gu.multislices_plot(data, sum_frames=False, scale='linear', plot_colorbar=True, vmin=0, vmax=1,
                                    title='Support before masking', is_orthogonal=True,
                                    reciprocal_space=False)
    cid = plt.connect('close_event', close_event)
    fig.waitforbuttonpress()
    plt.disconnect(cid)
    plt.close(fig)

    ###################################
    # clean interactively the support #
    ###################################
    plt.ioff()

    #############################################
    # mask the projected data in each dimension #
    #############################################
    width = 0
    max_colorbar = 5
    flag_aliens = False
    flag_mask = True
    flag_pause = False  # press x to pause for pan/zoom
    previous_axis = None
    mask = np.zeros(data.shape)
    xy = []  # list of points for mask

    fig_mask, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))
    fig_mask.canvas.mpl_disconnect(fig_mask.canvas.manager.key_press_handler_id)
    original_data = np.copy(data)
    updated_mask = np.zeros((nz, ny, nx))
    data[mask == 1] = 0  # will appear as grey in the log plot (nan)
    ax0.imshow(np.log10(abs(data).sum(axis=0)), vmin=0, vmax=max_colorbar, cmap=my_cmap)
    ax1.imshow(np.log10(abs(data).sum(axis=1)), vmin=0, vmax=max_colorbar, cmap=my_cmap)
    ax2.imshow(np.log10(abs(data).sum(axis=2)), vmin=0, vmax=max_colorbar, cmap=my_cmap)
    ax3.set_visible(False)
    ax0.axis('scaled')
    ax1.axis('scaled')
    ax2.axis('scaled')
    if is_ortho:
        ax0.invert_yaxis()  # detector Y is vertical down
    ax0.set_title("XY")
    ax1.set_title("XZ")
    ax2.set_title("YZ")
    fig_mask.text(0.60, 0.45, "click to select the vertices of a polygon mask", size=12)
    fig_mask.text(0.60, 0.40, "then p to apply and see the result", size=12)
    fig_mask.text(0.60, 0.30, "x to pause/resume masking for pan/zoom", size=12)
    fig_mask.text(0.60, 0.25, "up larger masking box ; down smaller masking box", size=12)
    fig_mask.text(0.60, 0.20, "m mask ; b unmask ; right darker ; left brighter", size=12)
    fig_mask.text(0.60, 0.15, "p plot full masked data ; a restart ; q quit", size=12)
    info_text = fig_mask.text(0.60, 0.05, "masking enabled", size=16)
    plt.tight_layout()
    plt.connect('key_press_event', press_key)
    plt.connect('button_press_event', on_click)
    fig_mask.set_facecolor(background_plot)
    plt.show()

    mask[np.nonzero(updated_mask)] = 1
    data = original_data
    data[mask == 1] = 0
    del fig_mask, flag_pause, flag_mask, original_data, updated_mask
    gc.collect()

    ############################################
    # mask individual frames in each dimension #
    ############################################
    nz, ny, nx = np.shape(data)
    width = 5
    max_colorbar = 5
    flag_mask = False
    flag_aliens = True

    fig_mask, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))
    fig_mask.canvas.mpl_disconnect(fig_mask.canvas.manager.key_press_handler_id)
    original_data = np.copy(data)
    original_mask = np.copy(mask)
    frame_index = [0, 0, 0]
    ax0.imshow(data[frame_index[0], :, :], vmin=0, vmax=max_colorbar, cmap=my_cmap)
    ax1.imshow(data[:, frame_index[1], :], vmin=0, vmax=max_colorbar, cmap=my_cmap)
    ax2.imshow(data[:, :, frame_index[2]], vmin=0, vmax=max_colorbar, cmap=my_cmap)
    ax3.set_visible(False)
    ax0.axis('scaled')
    ax1.axis('scaled')
    ax2.axis('scaled')
    if is_ortho:
        ax0.invert_yaxis()  # detector Y is vertical down
    ax0.set_title("XY - Frame " + str(frame_index[0] + 1) + "/" + str(nz))
    ax1.set_title("XZ - Frame " + str(frame_index[1] + 1) + "/" + str(ny))
    ax2.set_title("YZ - Frame " + str(frame_index[2] + 1) + "/" + str(nx))
    fig_mask.text(0.60, 0.30, "m mask ; b unmask ; u next frame ; d previous frame", size=12)
    fig_mask.text(0.60, 0.25, "up larger ; down smaller ; right darker ; left brighter", size=12)
    fig_mask.text(0.60, 0.20, "p plot full image ; q quit", size=12)
    plt.tight_layout()
    plt.connect('key_press_event', press_key)
    fig_mask.set_facecolor(background_plot)
    plt.show()

    mask[np.nonzero(mask)] = 1
    data[mask == 1] = 0
    del fig_mask, original_data, original_mask, mask
    gc.collect()

############################################
# plot the support with the original shape #
############################################
fig, _, _ = gu.multislices_plot(data, sum_frames=False, scale='linear', plot_colorbar=True, vmin=0,
                                title='Support after masking\n', is_orthogonal=True, reciprocal_space=False)
cid = plt.connect('close_event', close_event)
fig.waitforbuttonpress()
plt.disconnect(cid)
plt.close(fig)

#################################
# Richardson-Lucy deconvolution #
#################################
if psf_iterations > 0:
    data = au.deconvolution_rl(data, psf=psf, iterations=psf_iterations, debugging=True)

##################
# apply a filter #
##################
if filter_name != 'do_nothing':

    comment = comment + '_' + filter_name
    data = pu.filter_3d(data, filter_name=filter_name, sigma=gaussian_sigma, debugging=True)
    fig, _, _ = gu.multislices_plot(data, sum_frames=False, scale='linear', plot_colorbar=True, vmin=0,
                                    title='Support after filtering\n', is_orthogonal=True,
                                    reciprocal_space=False)
    cid = plt.connect('close_event', close_event)
    fig.waitforbuttonpress()
    plt.disconnect(cid)
    plt.close(fig)

data = data / data.max(initial=None)  # normalize
data[data < support_threshold] = 0
if binary_support:
    data[np.nonzero(data)] = 1  # change data into a support

############################################
# go back to original shape before binning #
############################################
if reload_support:
    binned_shape = [int(output_shape[idx] / binning_output[idx]) for idx in range(0, len(binning_output))]
else:
    binned_shape = [int(original_shape[idx] * binning_pynx[idx]) for idx in range(0, len(binning_pynx))]
nz, ny, nx = binned_shape

data = pu.crop_pad(data, binned_shape)
print('Data shape after considering original binning and shape:', data.shape)

######################
# center the support #
######################
if center:
    data = pu.center_com(data)
# Use user-defined roll when the center by COM is not optimal
data = np.roll(data, roll_centering, axis=(0, 1, 2))

#################################
# rescale the support if needed #
#################################
nbz, nby, nbx = output_shape
if ((nbz != nz) or (nby != ny) or (nbx != nx)) and not reload_support:
    print('Interpolating the support to match the output shape of', output_shape)
    if is_ortho:
        # load the original q values to calculate actual real space voxel sizes
        file_path = filedialog.askopenfilename(initialdir=root_folder, title="Select original q values",
                                               filetypes=[("NPZ", "*.npz")])
        q_values = np.load(file_path)
        qx = q_values['qx']  # 1D array
        qy = q_values['qy']  # 1D array
        qz = q_values['qz']  # 1D array
        # crop q to accomodate a shape change of the original array (e.g. cropping to fit FFT shape requirement)
        qx = pu.crop_pad_1d(qx, binned_shape[0])  # qx along z
        qy = pu.crop_pad_1d(qy, binned_shape[2])  # qy along x
        qz = pu.crop_pad_1d(qz, binned_shape[1])  # qz along y
        print('Length(q_original)=', len(qx), len(qz), len(qy), '(qx, qz, qy)')
        voxelsize_z = 2 * np.pi / (qx.max() - qx.min())  # qx along z
        voxelsize_x = 2 * np.pi / (qy.max() - qy.min())  # qy along x
        voxelsize_y = 2 * np.pi / (qz.max() - qz.min())  # qz along y

        # load the q values of the desired shape and calculate corresponding real space voxel sizes
        file_path = filedialog.askopenfilename(initialdir=root_folder, title="Select q values for the new shape",
                                               filetypes=[("NPZ", "*.npz")])
        q_values = np.load(file_path)
        newqx = q_values['qx']  # 1D array
        newqy = q_values['qy']  # 1D array
        newqz = q_values['qz']  # 1D array
        # crop q to accomodate a shape change of the original array (e.g. cropping to fit FFT shape requirement)
        # binning has no effect on the voxel size
        newqx = pu.crop_pad_1d(newqx, output_shape[0])  # qx along z
        newqy = pu.crop_pad_1d(newqy, output_shape[2])  # qy along x
        newqz = pu.crop_pad_1d(newqz, output_shape[1])  # qz along y
        print('Length(q_output)=', len(newqx), len(newqz), len(newqy), '(qx, qz, qy)')
        newvoxelsize_z = 2 * np.pi / (newqx.max() - newqx.min())  # qx along z
        newvoxelsize_x = 2 * np.pi / (newqy.max() - newqy.min())  # qy along x
        newvoxelsize_y = 2 * np.pi / (newqz.max() - newqz.min())  # qz along y

    else:  # data in detector frame
        wavelength = 12.398 * 1e-7 / energy  # in m
        voxelsize_z = wavelength / (nz * abs(tilt_angle) * np.pi / 180) * 1e9  # in nm
        voxelsize_y = wavelength * distance / (ny * pixel_y) * 1e9  # in nm
        voxelsize_x = wavelength * distance / (nx * pixel_x) * 1e9  # in nm

        newvoxelsize_z = wavelength / (nbz * abs(tilt_angle) * np.pi / 180) * 1e9  # in nm
        newvoxelsize_y = wavelength * distance / (nby * pixel_y) * 1e9  # in nm
        newvoxelsize_x = wavelength * distance / (nbx * pixel_x) * 1e9  # in nm

    print('Original voxel sizes zyx (nm):', str('{:.2f}'.format(voxelsize_z)), str('{:.2f}'.format(voxelsize_y)),
          str('{:.2f}'.format(voxelsize_x)))
    print('Output voxel sizes zyx (nm):', str('{:.2f}'.format(newvoxelsize_z)), str('{:.2f}'.format(newvoxelsize_y)),
          str('{:.2f}'.format(newvoxelsize_x)))

    rgi = RegularGridInterpolator((np.arange(-nz // 2, nz // 2, 1) * voxelsize_z,
                                   np.arange(-ny // 2, ny // 2, 1) * voxelsize_y,
                                   np.arange(-nx // 2, nx // 2, 1) * voxelsize_x),
                                  data, method='linear', bounds_error=False, fill_value=0)

    new_z, new_y, new_x = np.meshgrid(np.arange(-nbz // 2, nbz // 2, 1) * newvoxelsize_z,
                                      np.arange(-nby // 2, nby // 2, 1) * newvoxelsize_y,
                                      np.arange(-nbx // 2, nbx // 2, 1) * newvoxelsize_x, indexing='ij')

    new_support = rgi(np.concatenate((new_z.reshape((1, new_z.size)), new_y.reshape((1, new_z.size)),
                                      new_x.reshape((1, new_z.size)))).transpose())
    new_support = new_support.reshape((nbz, nby, nbx)).astype(data.dtype)

    print('Shape after interpolating the support:', new_support.shape)

else:  # no need for interpolation
    new_support = data

if binary_support:
    new_support[np.nonzero(new_support)] = 1
##########################################
# plot the support with the output shape #
##########################################
fig, _, _ = gu.multislices_plot(new_support, sum_frames=False, scale='linear', plot_colorbar=True, vmin=0,
                                title='Support after interpolation\n', is_orthogonal=True,
                                reciprocal_space=False)
filename = 'support_' + str(nbz) + '_' + str(nby) + '_' + str(nbx) +\
           '_bin_' + str(binning_output[0]) + '_' + str(binning_output[1]) + '_' + str(binning_output[2]) + comment

if save_fig:
    fig.savefig(root_folder + filename + '.png')

##########################################################################
# crop the new support to accomodate the binning factor in later phasing #
##########################################################################
binned_shape = [int(output_shape[idx] / binning_output[idx]) for idx in range(0, len(binning_output))]
new_support = pu.crop_pad(new_support, binned_shape)
print('Final shape after accomodating for later binning:', binned_shape)

###################################
# save support with the new shape #
###################################
np.savez_compressed(root_folder + filename + '.npz', obj=new_support)

plt.ioff()
plt.show()
