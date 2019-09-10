# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
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
output_shape = [162, 492, 162]
background_plot = '0.5'  # in level of grey in [0,1], 0 being dark. For visual comfort during masking
###################################################################


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
file_path = filedialog.askopenfilenames(initialdir=root_folder,
                                        filetypes=[("NPZ", "*.npz"), ("NPY", "*.npy"), ("CXI", "*.cxi"),
                                                   ("HDF5", "*.h5")])

data, _ = pu.load_reconstruction(file_path[0])
data[60:, :, :] = 0
data = abs(data)  # take the real part
data = data / data.max()  # normalize
data[data < support_threshold] = 0
data[np.nonzero(data)] = 1

data = pu.crop_pad(data, output_shape)
mask = np.zeros(data.shape)
print('output data shape', data.shape)

fig, _, _ = gu.multislices_plot(data, sum_frames=False, scale='linear', plot_colorbar=True, vmin=0, vmax=1,
                                title='Support before masking', invert_yaxis=False, is_orthogonal=True,
                                reciprocal_space=False)
cid = plt.connect('close_event', close_event)
fig.waitforbuttonpress()
plt.disconnect(cid)
plt.close(fig)

##########################
# remove unwanted pixels #
##########################
plt.ioff()
nz, ny, nx = np.shape(data)
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

fig, _, _ = gu.multislices_plot(data, sum_frames=True, scale='log', plot_colorbar=True, vmin=0,
                                title='Data after aliens removal\n', invert_yaxis=False,
                                is_orthogonal=True, reciprocal_space=True)

################
# save support #
################
filename = 'support_' + str(output_shape[0]) + '_' + str(output_shape[1]) + '_' + str(output_shape[2]) + '.npz'
np.savez_compressed(root_folder+filename, obj=data)

plt.ioff()
plt.show()
