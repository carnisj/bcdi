# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import filedialog
from scipy.stats import pearsonr
import sys
sys.path.append('D:/myscripts/bcdi/')
import bcdi.utils.utilities as util
from bcdi.utils import image_registration as reg
import bcdi.graph.graph_utils as gu
import bcdi.postprocessing.postprocessing_utils as pu

helptext = """
Compare the correlation between several 3D objects.
"""

datadir = 'D:/data/P10_August2019/data/gold_2_2_2_00022/pynx/1000_1000_1000_1_1_1/maximum_likelihood/good/'
threshold_correlation = 0.05  # only points above that threshold will be considered for correlation calculation
###################
# define colormap #
###################
colormap = gu.Colormap()
my_cmap = colormap.cmap

#############
# load data #
#############
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilenames(initialdir=datadir, filetypes=[("NPZ", "*.npz"), ("NPY", "*.npy"),
                                                                       ("CXI", "*.cxi"), ("HDF5", "*.h5")])
nbfiles = len(file_path)
print(nbfiles, 'files selected')
#################################################################
# loop through files and calculate the correlation coefficients #
#################################################################
correlation = np.zeros((nbfiles, nbfiles))
for raw in range(nbfiles):
    reference_obj, _ = util.load_file(file_path[raw])
    reference_obj = abs(reference_obj) / abs(reference_obj).max()
    nbz, nby, nbx = reference_obj.shape
    reference_obj = pu.crop_pad(array=reference_obj, output_shape=[nbz+10, nby+10, nbx+10])
    correlation[raw, raw] = 1
    for col in range(raw+1, nbfiles):
        test_obj, _ = util.load_file(file_path[col])  # which index?
        test_obj = abs(test_obj) / abs(test_obj).max()
        test_obj = pu.crop_pad(array=test_obj, output_shape=[nbz + 10, nby + 10, nbx + 10])
        # align reconstructions
        shiftz, shifty, shiftx = reg.getimageregistration(abs(reference_obj), abs(test_obj), precision=100)
        test_obj = reg.subpixel_shift(test_obj, shiftz, shifty, shiftx)
        print('\nReference =', raw, '  Test =', col)
        print('z shift', str('{:.2f}'.format(shiftz)), ', y shift',
              str('{:.2f}'.format(shifty)), ', x shift', str('{:.2f}'.format(shiftx)))

        correlation[raw, col] = pearsonr(np.ndarray.flatten(abs(reference_obj[reference_obj > threshold_correlation])),
                                         np.ndarray.flatten(abs(test_obj[reference_obj > threshold_correlation])))[0]
        correlation[col, raw] = correlation[raw, col]
        print('Correlation=', str('{:.2f}'.format(correlation[raw, col])))

plt.figure()
plt.imshow(correlation, cmap=my_cmap, vmin=0, vmax=1)
plt.colorbar()
plt.title('Pearson correlation coefficients')
plt.show()
