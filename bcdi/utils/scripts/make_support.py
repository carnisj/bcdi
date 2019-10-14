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
In real space the CXI convention is used: z downstream, y vertical up, x outboard.
In reciprocal space, the following convention is used: qx downtream, qz vertical up, qy outboard

"""

root_folder = "D:/data/P10_August2019/data/gold_2_2_2_00022/pynx/830_1000_830_1_1_1/"
support_threshold = 0.1  # in % of the normalized absolute value
original_shape = [564, 800, 564]  # shape of the array used for phasing and finding the support (after binning_original)
binning_original = (2, 2, 2)  # binning that was used in PyNX during phasing
output_shape = [800, 1000, 800]  # shape of the array for later phasing (before binning_output)
binning_output = (2, 2, 2)  # binning that will be used in PyNX for later phasing
skip_masking = False  # if True, will skip thresholding and masking
reload_support = True  # if True, will load the support which shape is assumed to be the shape after binning_output
# it is usefull to redo some masking without interpolating again.
is_ortho = True  # True if the data is already orthogonalized
roll_modes = (0, 0, 0)  # correct a roll of few pixels after the decomposition into modes in PyNX. axis=(0, 1, 2)
roll_centering = (0, 0, 0)  # roll applied after masking when centering by center of mass is not optimal axis=(0, 1, 2)
background_plot = '0.5'  # in level of grey in [0,1], 0 being dark. For visual comfort during masking
##############################################################################
# parameters used when (original_shape != output_shape) and (is_ortho=False) #
##############################################################################
energy = 9000  # in eV
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
data, _ = pu.load_reconstruction(file_path)

if reload_support:
    binned_shape = [int(output_shape[idx] / binning_output[idx]) for idx in range(0, len(binning_output))]
else:
    binned_shape = [int(original_shape[idx] * binning_original[idx]) for idx in range(0, len(binning_original))]
nz, ny, nx = binned_shape

if not skip_masking:

    data = np.roll(data, roll_modes, axis=(0, 1, 2))

    data = abs(data)  # take the real part
    data = data / data.max()  # normalize
    data[data < support_threshold] = 0

    # go back to original shape before binning
    data = pu.crop_pad(data, binned_shape)
    mask = np.zeros(data.shape)
    print('Data shape after considering original binning and shape:', data.shape)

    fig, _, _ = gu.multislices_plot(data, sum_frames=False, scale='linear', plot_colorbar=True, vmin=0, vmax=1,
                                    title='Support before masking', invert_yaxis=True, is_orthogonal=True,
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
    data[np.nonzero(data)] = 1

######################
# center the support #
######################
data = pu.center_com(data)
# Use user-defined roll when the center by COM is not optimal
data = np.roll(data, roll_centering, axis=(0, 1, 2))

############################################
# plot the support with the original shape #
############################################
fig, _, _ = gu.multislices_plot(data, sum_frames=True, scale='log', plot_colorbar=True, vmin=0, invert_yaxis=True,
                                title='Support after masking\n', is_orthogonal=True, reciprocal_space=True)
cid = plt.connect('close_event', close_event)
fig.waitforbuttonpress()
plt.disconnect(cid)
plt.close(fig)

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
        # TODO: check this part especially dq considering cropping/binning
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
                                      np.arange(-nby // 2, nby // 2, 1) * newvoxelsize_y,
                                      np.arange(-nbx // 2, nbx // 2, 1) * newvoxelsize_x, indexing='ij')

    new_support = rgi(np.concatenate((new_z.reshape((1, new_z.size)), new_y.reshape((1, new_z.size)),
                                      new_x.reshape((1, new_z.size)))).transpose())
    new_support = new_support.reshape((nbz, nby, nbx)).astype(data.dtype)

    print('Shape after interpolating the support:', new_support.shape)

else:  # no need for interpolation
    new_support = data

##########################################
# plot the support with the output shape #
##########################################
fig, _, _ = gu.multislices_plot(new_support, sum_frames=True, scale='log', plot_colorbar=True, vmin=0,
                                invert_yaxis=True, title='Support after interpolation\n', is_orthogonal=True,
                                reciprocal_space=True)

##########################################################################
# crop the new support to accomodate the binning factor in later phasing #
##########################################################################
binned_shape = [int(output_shape[idx] / binning_output[idx]) for idx in range(0, len(binning_output))]
new_support = pu.crop_pad(new_support, binned_shape)
print('Final shape after accomodating for later binning:', binned_shape)

###################################
# save support with the new shape #
###################################
filename = 'support_' + str(nbz) + '_' + str(nby) + '_' + str(nbx) +\
           '_bin_' + str(binning_output[0]) + '_' + str(binning_output[1]) + '_' + str(binning_output[2]) + '.npz'
np.savez_compressed(root_folder+filename, obj=new_support)

plt.ioff()
plt.show()
