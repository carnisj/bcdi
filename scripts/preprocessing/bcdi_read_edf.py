#!/usr/bin/env python3

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import os
import sys
import tkinter as tk

import fabio
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage.measurements import center_of_mass

helptext = """
Read .edf image (ESRF data format).
"""

img = 8758
centering = 1  # 0 max, 1 center of mass
savedir = "D:/data/PtRh/detector_calibration/"
datadir = "D:/data/PtRh/detector_calibration/"
ccdfiletmp = os.path.join(
    datadir, "BCDI_eiger2M_%05d.edf.gz"
)  # template for the CCD file names
save = 1  # 1 to save image
photon_threshold = 0
comment = str(img)
region = [
    0,
    2164,
    0,
    1030,
]  # [130, 200, 278, 347]  # Maxipix [0, 516, 0, 516]  # Eiger2M [0, 2164, 0, 1030]
normalize_method = (
    0  # 1 to normalize using max, 2 to normalize the data using filter, 0 otherwise
)
logscale = 1  # 1 for log scale
# ####### below is useful only if you want to normalize
# with filter (normalize_method = 2)
filter_nb = 1
filter_factor = 3.8  # 3.8 at 8keV, 2.5 at 9keV
##################################################################################
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
##################################################################################
root = tk.Tk()
root.withdraw()
# file_path = filedialog.askopenfilenames(initialdir=datadir,
# filetypes=[("EDF.GZ", "*.edf.gz"), ("EDF", "*.edf")])
# e = fabio.open(file_path[0])
e = fabio.open(ccdfiletmp % img)
data = e.data
# if save == 1:
#     np.save(savedir+'img'+comment+'.npy', data)

if normalize_method == 1:
    data = data / data.max()
    colorscale_max = 0
elif normalize_method == 2 and filter_nb > 0:
    data = data * filter_nb * filter_factor
    if logscale == 1:
        colorscale_max = round(np.log10(data.max()))
    else:
        colorscale_max = round(data.max())
else:
    if logscale == 1:
        colorscale_max = round(np.log10(data.max()))
    else:
        colorscale_max = round(data.max())
data[data <= photon_threshold] = 0
data_min = data[np.nonzero(data)].min()
if logscale == 1:
    colorscale_min = round(np.log10(data_min))
else:
    colorscale_min = round(data_min)
data[data == 0] = data_min / 10  # to avoid nan in log10

if centering == 0:
    y0, x0 = np.unravel_index(abs(data).argmax(), data.shape)
    print("Max at (y, x): ", y0, x0)
elif centering == 1:
    y0, x0 = center_of_mass(data)
    print(
        "Center of mass at (y, x): ",
        str(f"{y0:.2f}"),
        " , ",
        str(f"{x0:.2f}"),
    )
else:
    sys.exit("Incorrect value for 'centering' parameter")
print("Max = ", str(abs(data).max()))
fig = plt.figure()
if logscale == 1:
    plt.imshow(
        np.log10(data[region[0] : region[1], region[2] : region[3]]),
        cmap=my_cmap,
        vmin=colorscale_min,
        vmax=colorscale_max,
    )
else:
    plt.imshow(
        data[region[0] : region[1], region[2] : region[3]],
        cmap=my_cmap,
        vmin=colorscale_min,
        vmax=colorscale_max,
    )
# plt.title(str(region))
plt.title(
    "Img "
    + str(img)
    + " Max="
    + str(f"{abs(data).max():.2f}")
    + " @ H="
    + str(f"{x0:.2f}")
    + "/V="
    + str(f"{y0:.2f}")
)
plt.colorbar()
if save == 1:
    fig.savefig(savedir + "img" + comment + ".png")
    # np.save(savedir+'img'+comment+'.npy', data)
plt.show()
