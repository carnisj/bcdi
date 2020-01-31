# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.colors import LinearSegmentedColormap
import sys
sys.path.append('D:/myscripts/bcdi/')
from bcdi.utils import image_registration as reg
from scipy.stats import pearsonr

helptext = """
Compare the correlation between 3D diffraction patterns.
"""


previous_file = "T:/current/processed/analysis/dewet5_sum_S173_to_S185/pynx_S173_to_S185.npz"

next_file = "T:/current/processed/analysis/dewet5_sum_S194_to_S203/pynx_S194_to_S203.npz"
################################################
# parameters for plotting)
params = {'backend': 'Qt5Agg',
          'axes.labelsize': 20,
          'font.size': 20,
          'legend.fontsize': 20,
          'axes.titlesize': 20,
          'xtick.labelsize': 20,
          'ytick.labelsize': 20,
          'text.usetex': False,
          'figure.figsize': (11, 9)}
matplotlib.rcParams.update(params)
# define a colormap
cdict = {'red':  ((0.0, 1.0, 1.0),
                  (0.11, 0.0, 0.0),
                  (0.36, 0.0, 0.0),
                  (0.62, 1.0, 1.0),
                  (0.87, 1.0, 1.0),
                  (1.0, 0.0, 0.0)),
         'green': ((0.0, 1.0, 1.0),
                   (0.11, 0.0, 0.0),
                   (0.36, 1.0, 1.0),
                   (0.62, 1.0, 1.0),
                   (0.87, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),
         'blue': ((0.0, 1.0, 1.0),
                  (0.11, 1.0, 1.0),
                  (0.36, 1.0, 1.0),
                  (0.62, 0.0, 0.0),
                  (0.87, 0.0, 0.0),
                  (1.0, 0.0, 0.0))}
my_cmap = LinearSegmentedColormap('my_colormap', cdict, 256)
plot_title = ['YZ', 'XZ', 'XY']
#####################################################
previous_data = np.load(previous_file)['obj']
previous_data = previous_data / abs(previous_data).max()
next_data = np.load(next_file)['obj']
next_data = next_data / abs(next_data).max()

shiftz, shifty, shiftx = reg.getimageregistration(abs(previous_data), abs(next_data), precision=1000)
next_data = reg.subpixel_shift(next_data, shiftz, shifty, shiftx)
print('\n x shift', shiftx,'y shift', shifty,'z shift', shiftz)

correlation = pearsonr(np.ndarray.flatten(abs(previous_data[previous_data>1])), np.ndarray.flatten(abs(next_data[previous_data>1])))[0]
print('\n Pearson correlation coefficient = ', str('{:.3f}'.format(correlation)))