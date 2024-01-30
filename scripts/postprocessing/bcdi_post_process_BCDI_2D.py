#!/usr/bin/env python3

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#       authors:
#         Jerome Carnis, jerome.carnis@esrf.fr

import sys
import tkinter as tk
from tkinter import filedialog

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.measurements import center_of_mass

matplotlib.use("Qt5Agg")

helptext = """
starting from a 2D complex object (output of phasing program), center the object,
remove the phase ramp, the phase offset and wrap the phase.
"""


datadir = (
    "C:/Users/Jerome/Documents/data/BCDI_isosurface/S2227/simu/Figures/phasing_kin_FFT/"
)
savedir = datadir
original_size = [512, 512]  # size of the FFT window used for phasing
phase_range = np.pi / 30  # in radians, for plots
save_colorbar = 1  # to save the colorbar
comment = "_pynx_fft_negative"  # should start with _
aGe = 0.5658  # lattice spacing of Ge in nm
d400_Ge = aGe / 4
q400_Ge = 2 * np.pi / d400_Ge  # inverse nm
print("q=", str(q400_Ge), " inverse nm")
# parameters for plotting
params = {
    "backend": "ps",
    "axes.labelsize": 20,
    "text.fontsize": 20,
    "legend.fontsize": 20,
    "title.fontsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "text.usetex": False,
    "figure.figsize": (11, 9),
}

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
my_cmap = matplotlib.colors.LinearSegmentedColormap("my_colormap", cdict, 256)


def crop_pad_2d(myobj, myshape, debugging=0):
    """
    Will crop or pad my obj depending on myshape

    :param myobj: 2d complex array to be padded
    :param myshape: list of desired output shape [y, x]
    :param debugging: to plot myobj before and after padding
    :return: myobj padded with zeros
    """
    nby, nbx = myobj.shape
    newy, newx = myshape
    if debugging == 1:
        plt.figure()
        plt.imshow(abs(myobj), vmin=0, vmax=1)
        plt.colorbar()
        plt.axis("scaled")
        plt.title("before crop/pad")
        plt.pause(0.1)
    # y
    if newy >= nby:  # pad
        temp_y = np.zeros((newy, nbx), dtype=myobj.dtype)
        temp_y[(newy - nby) // 2 : (newy + nby) // 2, :] = myobj
    else:  # crop
        temp_y = myobj[(nby - newy) // 2 : (newy + nby) // 2, :]
    # x
    if newx >= nbx:  # pad
        newobj = np.zeros((newy, newx), dtype=myobj.dtype)
        newobj[:, (newx - nbx) // 2 : (newx + nbx) // 2] = temp_y
    else:  # crop
        newobj = temp_y[:, (nbx - newx) // 2 : (newx + nbx) // 2]

    if debugging == 1:
        plt.figure()
        plt.imshow(abs(newobj), vmin=0, vmax=1)
        plt.colorbar()
        plt.axis("scaled")
        plt.title("after crop/pad")
        plt.pause(0.1)
    return newobj


def center_com_2d(myarray, debugging=0):
    """
    Center myarray using the center of mass

    :param myarray: array to be centered based on the center of mass value
    :param debugging: 1 to show plots
    :return centered array
    """
    nby, nbx = myarray.shape
    if debugging == 1:
        plt.figure()
        plt.imshow(abs(myarray), cmap=my_cmap)
        plt.colorbar()
        plt.title("Sum(amp)before COM centering")
        plt.pause(0.1)
    piy, pix = center_of_mass(abs(myarray))
    print(
        "center of mass at (y, x): (",
        str(f"{piy:.2f}"),
        ",",
        str(f"{pix:.2f}"),
        ")",
    )
    offset_y = int(np.rint(nby / 2.0 - piy))
    offset_x = int(np.rint(nbx / 2.0 - pix))
    print("center of mass offset: (", offset_y, ",", offset_x, ") pixels")
    myarray = np.roll(myarray, (offset_y, offset_x), axis=(0, 1))
    if debugging == 1:
        plt.figure()

        plt.imshow(abs(myarray), cmap=my_cmap)
        plt.colorbar()
        plt.title("Sum(amp) after COM centering")
        plt.pause(0.1)
    return myarray


def remove_ramp_2d(myamp, myphase, threshold, gradient_threshold, debugging=0):
    """
    remove_ramp: remove the linear trend in the ramp using its gradient and a threshold
    :param myamp: amplitude of the object
    :param myphase: phase of the object, to be detrended
    :param threshold: threshold used to define the support of the object
    :param gradient_threshold: higher threshold used to select valid voxels in the
     gradient array
    :param debugging: 1 to show plots
    :return: the detrended phase
    """
    grad_threshold = gradient_threshold
    nby, nbx = myamp.shape
    mysupport = np.zeros((nby, nbx))
    mysupport[myamp > threshold * abs(myamp).max()] = 1

    mygrady, mygradx = np.gradient(myphase, 1)

    mysupporty = np.zeros((nby, nbx))
    mysupporty[abs(mygrady) < grad_threshold] = 1
    mysupporty = mysupporty * mysupport
    myrampy = mygrady[mysupporty == 1].mean()
    if debugging == 1:
        plt.figure()
        plt.imshow(mygrady, vmin=-0.2, vmax=0.2)
        plt.colorbar()
        plt.axis("scaled")
        plt.title("mygrady")
        plt.pause(0.1)

        plt.figure()
        plt.imshow(mysupporty, vmin=0, vmax=1)
        plt.colorbar()
        plt.axis("scaled")
        plt.title("mysupporty")
        plt.pause(0.1)

    mysupportx = np.zeros((nby, nbx))
    mysupportx[abs(mygradx) < grad_threshold] = 1
    mysupportx = mysupportx * mysupport
    myrampx = mygradx[mysupportx == 1].mean()
    if debugging == 1:
        plt.figure()
        plt.imshow(mygradx, vmin=-0.2, vmax=0.2)
        plt.colorbar()
        plt.axis("scaled")
        plt.title("mygradx")
        plt.pause(0.1)

        plt.figure()
        plt.imshow(mysupportx, vmin=0, vmax=1)
        plt.colorbar()
        plt.axis("scaled")
        plt.title("mysupportx")
        plt.pause(0.1)

    myy, myx = np.meshgrid(np.arange(0, nby, 1), np.arange(0, nbx, 1), indexing="ij")
    print(
        "Phase_ramp_y, Phase_ramp_x: (",
        str(f"{myrampy:.3f}"),
        str(f"{myrampx:.3f}"),
        ") rad",
    )
    myphase = myphase - myy * myrampy - myx * myrampx
    return myphase


def wrap(myphase):
    """
    Wrap the phase in [-pi pi] interval

    :param myphase:
    :return:
    """
    myphase = (myphase + np.pi) % (2 * np.pi) - np.pi
    return myphase


plt.ion()
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    initialdir=datadir,
    filetypes=[("NPZ", "*.npz"), ("NPY", "*.npy"), ("CXI", "*.cxi"), ("HDF5", "*.h5")],
)
if file_path.lower().endswith(".npz"):
    ext = ".npz"
    npzfile = np.load(file_path)
    obj = npzfile["obj"]
elif file_path.lower().endswith(".npy"):
    ext = ".npy"
    obj = np.load(file_path[0])
elif file_path.lower().endswith(".cxi"):
    ext = ".cxi"
    h5file = h5py.File(file_path, "r")
    group_key = list(h5file.keys())[1]
    subgroup_key = list(h5file[group_key])
    obj = h5file["/" + group_key + "/" + subgroup_key[0] + "/data"].value
elif file_path.lower().endswith(".h5"):
    ext = ".h5"
    h5file = h5py.File(file_path, "r")
    group_key = list(h5file.keys())[0]
    subgroup_key = list(h5file[group_key])
    obj = h5file["/" + group_key + "/" + subgroup_key[0] + "/data"].value[0]
    comment = comment + "_1stmode"
else:
    sys.exit("wrong file format")


################
# center object
################
obj = center_com_2d(obj, debugging=1)

if len(original_size) != 0:
    print("Original FFT window size: ", original_size)
    print("Padding back to initial size")
    obj = crop_pad_2d(myobj=obj, myshape=original_size, debugging=1)

ny, nx = obj.shape
half_window = int(nx / 2)

####################
# remove phase ramp
####################
amp = abs(obj)
amp = amp / amp.max()
plt.figure()
plt.imshow(amp, cmap=my_cmap)
plt.colorbar()
plt.axis("scaled")
plt.title("amplitude")
plt.pause(0.1)

phase = np.angle(obj)

plt.figure()
plt.imshow(phase, cmap=my_cmap)
plt.colorbar()
plt.axis("scaled")
plt.title("phase before ramp removal")
plt.pause(0.1)

# phase = remove_ramp_2d(amp, phase, threshold=0.0005,
# gradient_threshold=0.01, debugging=1)

plt.figure()
plt.imshow(phase, cmap=my_cmap)
plt.colorbar()
plt.axis("scaled")
plt.title("phase after ramp removal")
plt.pause(0.1)
####################
# remove phase offset
####################
support = np.zeros(amp.shape)
support[amp > 0.05] = 1
plt.figure()
plt.imshow(support, cmap=my_cmap)
plt.colorbar()
plt.axis("scaled")
plt.title("support used for offset removal")
plt.pause(0.1)

ycom, xcom = center_of_mass(support)
print("Mean phase:", phase[support == 1].mean(), "rad")
print(
    "COM at (y, x): (",
    ",",
    str(f"{ycom:.2f}"),
    ",",
    str(f"{xcom:.2f}"),
    ")",
)
print(
    "Phase offset at COM(amp) of:",
    str(f"{phase[int(ycom), int(xcom)]:.2f}"),
    "rad",
)
phase = phase - phase[int(ycom), int(xcom)]

plt.figure()
plt.imshow(phase, cmap=my_cmap)
plt.colorbar()
plt.axis("scaled")
plt.title("phase after offset removal")
plt.pause(0.1)
####################
# wrap phase
####################
phase = wrap(phase)

plt.figure()
plt.imshow(phase, cmap=my_cmap)
plt.colorbar()
plt.axis("scaled")
plt.title("phase after wrapping")
plt.pause(0.1)
####################
# scale back to displacement
####################
phase = phase / q400_Ge
phase[support == 0] = np.nan
plt.figure()
plt.imshow(phase, cmap=my_cmap)
plt.colorbar()
plt.axis("scaled")
plt.title("phase after rescaling to displacement")
plt.pause(0.1)
####################
# plot the phase
####################
fig, ax0 = plt.subplots(1, 1)
plt0 = ax0.imshow(
    phase[half_window - 100 : half_window + 100, half_window - 100 : half_window + 100],
    cmap=my_cmap,
    vmin=-phase_range,
    vmax=phase_range,
)
ax0.tick_params(
    labelbottom="off", labelleft="off", bottom="off", left="off", top="off", right="off"
)
plt.pause(0.5)
plt.savefig(savedir + comment + ".png", bbox_inches="tight")
if save_colorbar == 1:
    plt.colorbar(plt0, ax=ax0)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.pause(0.5)
    plt.savefig(savedir + comment + "_colorbar.png", bbox_inches="tight")
plt.ioff()
plt.show()
