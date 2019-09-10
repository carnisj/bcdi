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
import sys
sys.path.append('//win.desy.de/home/carnisj/My Documents/myscripts/bcdi/')
import bcdi.postprocessing.postprocessing_utils as pu
import bcdi.preprocessing.preprocessing_utils as pru
import bcdi.graph.graph_utils as gu

helptext = """
Create a support from a reconstruction, using the indicated threshold.
The support can be cropped/padded to a desired shape.
"""

root_folder = "D:/data/P10_August2019/data/gold_2_2_2_00022/pynx/"
support_threshold = 0.1  # in % of the normalized absolute value
original_shape = [162, 492, 162]  # shape of the array used for phasing and finding the support
output_shape = [162, 800, 162]  # shape of the array for later phasing
reload_support = True  # if True, will load the support and skip masking
is_ortho = True  # True if the data is already orthogonalized
background_plot = '0.5'  # in level of grey in [0,1], 0 being dark. For visual comfort during masking
##############################################################################
# parameters used when (original_shape != output_shape) and (is_ortho=False) #
##############################################################################
energy = 8700  # in eV
tilt_angle = 0.5  # in degrees
distance = 4.95  # in m
pixel_x = 75e-06  # in m
pixel_y = 75e-06  # in m
##################################
# end of user-defined parameters #
##################################


def close_event(event):
    """
    This function handles closing events on plots.

    :return: nothing
    """
    print(event, 'Click on the figure instead of closing it!')
    sys.exit()


def press_key(event):
    """
    Interact with a plot for masking parasitic diffraction intensity or detector gaps

    :param event: button press event
    :return: updated data, mask and controls
    """
    global original_data, data, mask, fig_mask, dim, idx, width, max_colorbar

    try:
        data, mask, width, max_colorbar, idx, stop_masking = \
            pru.update_aliens(key=event.key, pix=int(np.rint(event.xdata)), piy=int(np.rint(event.ydata)),
                              original_data=original_data, updated_data=data, updated_mask=mask,
                              figure=fig_mask, width=width, dim=dim, idx=idx, vmin=0, vmax=max_colorbar)
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
nz, ny, nx = original_shape
data, _ = pu.load_reconstruction(file_path)

if not reload_support:
    # data[60:, :, :] = 0
    data = abs(data)  # take the real part
    data = data / data.max()  # normalize
    data[data < support_threshold] = 0
    data[np.nonzero(data)] = 1

    data = pu.crop_pad(data, original_shape)
    mask = np.zeros(data.shape)
    print('output data shape', data.shape)

    fig, _, _ = gu.multislices_plot(data, sum_frames=False, scale='linear', plot_colorbar=True, vmin=0, vmax=1,
                                    title='Support before masking', invert_yaxis=False, is_orthogonal=True,
                                    reciprocal_space=False)
    cid = plt.connect('close_event', close_event)
    fig.waitforbuttonpress()
    plt.disconnect(cid)
    plt.close(fig)

    ###################################
    # clean interactively the support #
    ###################################
    plt.ioff()
    width = 5
    max_colorbar = 1
    flag_aliens = True

    # in XY
    dim = 0
    fig_mask = plt.figure()
    idx = 0
    original_data = np.copy(data)
    plt.imshow(data[idx, :, :], vmin=0, vmax=max_colorbar)
    plt.title("Frame " + str(idx+1) + "/" + str(nz) + "\n"
              "m mask ; b unmask ; q quit ; u next frame ; d previous frame\n"
              "up larger ; down smaller ; right darker ; left brighter")
    plt.connect('key_press_event', press_key)
    fig_mask.set_facecolor(background_plot)
    plt.show()
    del dim, fig_mask

    # in XZ
    dim = 1
    fig_mask = plt.figure()
    idx = 0
    plt.imshow(data[:, idx, :], vmin=0, vmax=max_colorbar)
    plt.title("Frame " + str(idx+1) + "/" + str(ny) + "\n"
              "m mask ; b unmask ; q quit ; u next frame ; d previous frame\n"
              "up larger ; down smaller ; right darker ; left brighter")
    plt.connect('key_press_event', press_key)
    fig_mask.set_facecolor(background_plot)
    plt.show()
    del dim, fig_mask

    # in YZ
    dim = 2
    fig_mask = plt.figure()
    idx = 0
    plt.imshow(data[:, :, idx], vmin=0, vmax=max_colorbar)
    plt.title("Frame " + str(idx+1) + "/" + str(nx) + "\n"
              "m mask ; b unmask ; q quit ; u next frame ; d previous frame\n"
              "up larger ; down smaller ; right darker ; left brighter")
    plt.connect('key_press_event', press_key)
    fig_mask.set_facecolor(background_plot)
    plt.show()

    del dim, width, fig_mask, original_data

############################################
# plot the support with the original shape #
############################################
gu.multislices_plot(data, sum_frames=True, scale='log', plot_colorbar=True, vmin=0, invert_yaxis=False,
                    title='Support with original shape\n', is_orthogonal=True, reciprocal_space=True)

########################################
# save support with the original shape #
########################################
filename = 'support_' + str(nz) + '_' + str(ny) + '_' + str(nx) + '.npz'
np.savez_compressed(root_folder+filename, obj=data)

#################################
# rescale the support if needed #
#################################
nbz, nby, nbx = output_shape
if (nbz != nz) or (nby != ny) or (nbx != nx):
    print('Interpolating the support to match the output shape')
    if is_ortho:
        # load the original q values to calculate actual real space voxel sizes
        file_path = filedialog.askopenfilename(initialdir=root_folder, title="Select original q values",
                                               filetypes=[("NPZ", "*.npz")])
        q_values = np.load(file_path)
        qx = q_values['qx']
        qy = q_values['qy']
        qz = q_values['qz']
        voxelsize_z = 2 * np.pi / (qx.max() - qx.min())
        voxelsize_x = 2 * np.pi / (qy.max() - qy.min())
        voxelsize_y = 2 * np.pi / (qz.max() - qz.min())

        # load the q values of the desired shape and calculate corresponding real space voxel sizes
        file_path = filedialog.askopenfilename(initialdir=root_folder, title="Select q values for the new shape",
                                               filetypes=[("NPZ", "*.npz")])
        q_values = np.load(file_path)
        newqx = q_values['qx']
        newqy = q_values['qy']
        newqz = q_values['qz']
        newvoxelsize_z = 2 * np.pi / (newqx.max() - newqx.min())
        newvoxelsize_x = 2 * np.pi / (newqy.max() - newqy.min())
        newvoxelsize_y = 2 * np.pi / (newqz.max() - newqz.min())
        print('Output voxel sizes:', newvoxelsize_z, newvoxelsize_y, newvoxelsize_x)

    else:  # data in detector frame
        # TODO: check this part
        wavelength = 12.398 * 1e-7 / energy  # in m
        voxelsize_z = wavelength / (nz * abs(tilt_angle) * np.pi / 180) * 1e9  # in nm
        voxelsize_y = wavelength * distance / (ny * pixel_y) * 1e9  # in nm
        voxelsize_x = wavelength * distance / (nx * pixel_x) * 1e9  # in nm

        newvoxelsize_z = wavelength / (nbz * abs(tilt_angle) * np.pi / 180) * 1e9  # in nm
        newvoxelsize_x = wavelength * distance / (nby * pixel_y) * 1e9  # in nm
        newvoxelsize_y = wavelength * distance / (nbx * pixel_x) * 1e9  # in nm

    print('Original voxel sizes (nm):', str('{:.2f}'.format(voxelsize_z)), str('{:.2f}'.format(voxelsize_y)),
          str('{:.2f}'.format(voxelsize_x)))
    print('Output voxel sizes (nm):', str('{:.2f}'.format(newvoxelsize_z)), str('{:.2f}'.format(newvoxelsize_y)),
          str('{:.2f}'.format(newvoxelsize_x)))

    rgi = RegularGridInterpolator((np.arange(-nz // 2, nz // 2, 1) * voxelsize_z,
                                   np.arange(-ny // 2, ny // 2, 1) * voxelsize_y,
                                   np.arange(-nx // 2, nx // 2, 1) * voxelsize_x),
                                  data, method='linear', bounds_error=False, fill_value=0)

    new_z, new_y, new_x = np.meshgrid(np.arange(-nbz // 2, nbz // 2, 1) * newvoxelsize_z,
                                      np.arange(-nby // 2, nby // 2, 1) * newvoxelsize_x,
                                      np.arange(-nbx // 2, nbx // 2, 1) * newvoxelsize_y, indexing='ij')

    new_support = rgi(np.concatenate((new_z.reshape((1, new_z.size)), new_y.reshape((1, new_z.size)),
                                      new_x.reshape((1, new_z.size)))).transpose())
    new_support = new_support.reshape((nbz, nby, nbx)).astype(data.dtype)

    gu.multislices_plot(new_support, sum_frames=True, scale='log', plot_colorbar=True, vmin=0, invert_yaxis=False,
                        title='Support with output shape\n', is_orthogonal=True, reciprocal_space=True)

    ###################################
    # save support with the new shape #
    ###################################
    filename = 'support_' + str(nbz) + '_' + str(nby) + '_' + str(nbx) + '.npz'
    np.savez_compressed(root_folder+filename, obj=new_support)

plt.ioff()
plt.show()
