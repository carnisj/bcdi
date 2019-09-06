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
import bcdi.graph.graph_utils as gu
helptext = """
Create a support from a reconstruction, using the indicated threshold.
"""

root_folder = "D:/data/P10_August2019/data/gold_2_2_2_00022/pynx/"
support_threshold = 0.2  # in % of the normalized absolute value
output_shape = [162, 490, 300]
###################################################################
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilenames(initialdir=root_folder,
                                        filetypes=[("NPZ", "*.npz"), ("NPY", "*.npy"), ("CXI", "*.cxi"),
                                                   ("HDF5", "*.h5")])

data, _ = pu.load_reconstruction(file_path[0])

data = abs(data)  # take the real part
data = data / data.max()  # normalize
data[data < support_threshold] = 0
data[np.nonzero(data)] = 1

data = pu.crop_pad(data, output_shape)
print('output data shape', data.shape)

filename = 'support_' + str(output_shape[0]) + '_' + str(output_shape[1]) + '_' + str(output_shape[2]) + '.npz'
np.savez_compressed(root_folder+filename, obj=data)

gu.multislices_plot(data, sum_frames=False, scale='linear', plot_colorbar=True, vmin=0, vmax=1, title='Support',
                    invert_yaxis=False, is_orthogonal=False, reciprocal_space=False)
plt.ioff()
plt.show()
