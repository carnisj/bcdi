#!/usr/bin/env python3

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import os
import sys

import fabio
import matplotlib
import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import numpy as np
import xrayutilities as xu
from matplotlib.colors import LinearSegmentedColormap
from silx.io.specfile import SpecFile

matplotlib.use("Qt5Agg")

helptext = """
Create a movie from a scan (i.e. timescan). Requires ffmpeg
(http://ffmpeg.zeranoe.com/builds/).
"""

scan = 99  # spec scan number
specdir = "C:/Users/carnis/Work Folders/Documents/data/HC3796/OER/"
spec_prefix = "2018_11_01_022929_OER"  #
homedir = specdir + "S" + str(scan) + "/"
datadir = homedir + "data/"
detector = 1  # 0 for eiger, 1 for maxipix
roi = [
    0,
    516,
    0,
    516,
]
# ROI of the detector to be plotted [0, 516, 0, 516] Maxipix, [0, 2164, 0, 1030] Eiger2M
ccdfiletmp = os.path.join(
    datadir, "data_mpx4_%05d.edf.gz"
)  # ID01 template for the CCD file names
flatfield_file = specdir + "flatfield_maxipix_8kev.npz"
nav = [1, 1]  # reduce data: number of pixels to average in each detector direction
movie_title = "HC3796 OER Timescan S" + str(scan)
##############################################################################
# parameters for plotting)
params = {
    "backend": "Qt5Agg",
    "axes.labelsize": 20,
    "font.size": 20,
    "legend.fontsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "text.usetex": False,
    "figure.figsize": (11, 9),
}
matplotlib.rcParams.update(params)
# define a colormap
cdict = {
    "red": (
        (0.0, 1.0, 1.0),
        (0.11, 0.0, 0.0),
        (0.36, 0.0, 0.0),
        (0.62, 1.0, 1.0),
        (0.87, 1.0, 1.0),
        (1.0, 0.0, 0.0),
    ),
    "green": (
        (0.0, 1.0, 1.0),
        (0.11, 0.0, 0.0),
        (0.36, 1.0, 1.0),
        (0.62, 1.0, 1.0),
        (0.87, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    ),
    "blue": (
        (0.0, 1.0, 1.0),
        (0.11, 1.0, 1.0),
        (0.36, 1.0, 1.0),
        (0.62, 0.0, 0.0),
        (0.87, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    ),
}
my_cmap = LinearSegmentedColormap("my_colormap", cdict, 256)

####################################
try:
    FFMpegWriter = manimation.writers["ffmpeg"]
except KeyError:
    print("KeyError: 'ffmpeg'")
    sys.exit()
except RuntimeError:
    print("Could not import FFMpeg writer for movie generation")
    sys.exit()

if flatfield_file != "":
    flatfield = np.load(flatfield_file)["flatfield"]
else:
    flatfield = None
spec_file = SpecFile(specdir + spec_prefix + ".spec")
motor_names = spec_file[str(scan) + ".1"].motor_names  # positioners
motor_positions = spec_file[str(scan) + ".1"].motor_positions  # positioners
labels = spec_file[str(scan) + ".1"].labels  # motor scanned
labels_data = spec_file[str(scan) + ".1"].data  # motor scanned
if detector == 0:
    counter = "ei2minr"
elif detector == 1:
    counter = "mpx4inr"
else:
    print("Wrong detector type")
    sys.exit()
ccdn = labels_data[labels.index(counter), :]

metadata = dict(title=movie_title)
writer = FFMpegWriter(fps=5, metadata=metadata)
fontsize = 10
sum_img = np.zeros((roi[1] - roi[0], roi[3] - roi[2]))
fig = plt.figure(figsize=(6, 5))
with writer.saving(fig, homedir + "S" + str(scan) + "_movie.mp4", dpi=100):
    for index, item in enumerate(ccdn):
        i = int(item)
        e = fabio.open(ccdfiletmp % i)
        ccdraw = e.data
        ccdraw = flatfield * ccdraw
        ccd = xu.blockAverage2D(ccdraw, nav[0], nav[1], roi=roi)
        sum_img = sum_img + ccd
        ccd = ccd + 0.0001
        plt.clf()
        plt.title("Img %3d" % i, fontsize=fontsize)
        plt.imshow(np.log10(ccd), vmin=0, vmax=4, cmap=my_cmap)
        writer.grab_frame()

sum_img = sum_img + 0.0001
plt.imshow(np.log10(sum_img), vmin=0, cmap=my_cmap)
plt.title("Sum of all images")
plt.show()
