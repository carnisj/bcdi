#!/usr/bin/env python3

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from numpy.fft import fftn, fftshift

matplotlib.use("Qt5Agg")

helptext = """
calculation of the diffraction pattern using FFTs with both conventions and kinematic
sum, to show the relationship between the phase and the displacement.
The object is a Ge-core / Si-shell nanowire.
"""

savedir = "C:/Users/carnis/Documents/data/CH4760_Pt/S2227/simu/Figures/phasing_kin_FFT/"
colorbar_range = [-7, 4]  # [0, 9.5]  # [vmin, vmax] log scale in photon counts
comment = "_GeSi_NW_scale" + str(colorbar_range)  # should start with _
tick_spacing = 25  # for plots in real space, in nm
tick_length = 5  # in plots
tick_width = 2  # in plots
save_colorbar = 1  # to save the colorbar
phase_range = np.pi / 30  # in radians, for plots

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
plt.ion()

##################
# Create the shape of the object
##################
half_window = 256  # half number of pixels in x (horizontal axis) and y (vertical axis)

aSi = 0.54309  # lattice spacing of Si in nm
aGe = 0.5658  # lattice spacing of Ge in nm
d400_Ge = aGe / 4  # the diffraction is calculated at Ge 400 peak
misfit = (aSi - aGe) / aGe  # dimensionless
Zc = 32  # atomic number of Germanium core
Zs = 14  # atomic number of  Silicon shell

voxel_size = aGe  # in nm
radius_core = 20 * voxel_size
radius_NW = 40 * voxel_size
alpha = np.arccos(1 / np.sqrt(3))

tmp = np.mgrid[-half_window:half_window, -half_window:half_window]
ygrid, xgrid = tmp[0] * voxel_size, tmp[1] * voxel_size

area_nanowire = np.where(
    (ygrid < radius_NW)
    & (ygrid > -radius_NW)
    & (ygrid < -np.tan(alpha - 10 * np.pi / 180) * (-xgrid - radius_NW))
    & (ygrid > -np.tan(alpha) * (-xgrid + radius_NW))
    & (ygrid > np.tan(alpha - 0 * np.pi / 180) * (xgrid - radius_NW))  #
    & (ygrid < -np.tan(alpha + 30 * np.pi / 180) * (xgrid - radius_NW))
    & (ygrid < np.tan(alpha) * (xgrid + radius_NW))
    & (ygrid > -np.tan(alpha) * (xgrid + radius_NW)),
    1,
    0,
)

area_core = np.where(
    (ygrid < radius_core)
    & (ygrid > -radius_core)
    & (ygrid < -np.tan(alpha - 10 * np.pi / 180) * (-xgrid - radius_core))
    & (ygrid > -np.tan(alpha) * (-xgrid + radius_core))
    & (ygrid > np.tan(alpha - 0 * np.pi / 180) * (xgrid - radius_core))
    & (ygrid < -np.tan(alpha + 30 * np.pi / 180) * (xgrid - radius_core))
    & (ygrid < np.tan(alpha) * (xgrid + radius_core))
    & (ygrid > -np.tan(alpha) * (xgrid + radius_core)),
    1,
    0,
)

nanowire = area_core * abs(Zc) + (area_nanowire - area_core) * abs(Zs)
np.savez_compressed(savedir + "GeSi_NW_support.npz", obj=nanowire)

pixel_spacing = tick_spacing / voxel_size
fig, ax0 = plt.subplots(1, 1)
plt0 = ax0.imshow(
    nanowire[
        half_window - 100 : half_window + 100, half_window - 100 : half_window + 100
    ],
    cmap=my_cmap,
    vmin=0,
    vmax=35,
)
ax0.xaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
ax0.yaxis.set_major_locator(ticker.MultipleLocator(pixel_spacing))
ax0.tick_params(
    labelbottom="off",
    labelleft="off",
    direction="in",
    top="on",
    right="on",
    length=tick_length,
    width=tick_width,
)
plt.pause(0.5)
plt.savefig(savedir + "density.png", bbox_inches="tight")
if save_colorbar == 1:
    plt.colorbar(plt0, ax=ax0)
    plt.xlabel("X")
    plt.ylabel("Y")
    ax0.tick_params(
        labelbottom="on",
        labelleft="on",
        labelsize=12,
        direction="in",
        top="on",
        right="on",
        length=tick_length,
        width=tick_width,
    )
    plt.pause(0.5)
    plt.savefig(savedir + "density_colorbar.png", bbox_inches="tight")

##################
# displacement
##################
nu = 0.27
theta = np.arctan2(xgrid, ygrid)
r = np.sqrt(ygrid * ygrid + xgrid * xgrid)  # in nm
alpha = (
    -radius_core
    * radius_core
    * misfit
    * (1 + nu)
    / (2 * radius_NW * radius_NW * (1 - nu))
)  # dimensionless
beta = alpha * radius_NW * radius_NW  # nm2
epsilonR = alpha - beta / (r * r)  # dimensionless
epsilonT = alpha + beta / (r * r)  # dimensionless
epsilonXX = misfit + epsilonR * np.cos(theta) ** 2 + epsilonT * np.sin(theta) ** 2
epsilonXX_Si = epsilonR * np.cos(theta) ** 2 + epsilonT * np.sin(theta) ** 2
epsilon_xx = np.zeros((2 * half_window, 2 * half_window))
# calculation based on the calculation of elastic strain in radial and
# transverse direction for a core-shell SiGe NW
# reference:

displacement = ((area_core - area_nanowire) * epsilonXX_Si + area_core * epsilon_xx) * 2
# plt.figure()
# plt.imshow(disp)

displacement[np.isnan(displacement)] = 0  # for central pixel which is not defined
ux = np.copy(displacement)

displacement[nanowire == 0] = np.nan  # for plots
# no displacement along y
fig, ax0 = plt.subplots(1, 1)
plt0 = ax0.imshow(
    displacement[
        half_window - 100 : half_window + 100, half_window - 100 : half_window + 100
    ],
    cmap=my_cmap,
    vmin=-phase_range,
    vmax=phase_range,
)
ax0.tick_params(
    labelbottom="off", labelleft="off", bottom="off", left="off", top="off", right="off"
)
plt.pause(0.5)
plt.savefig(savedir + "ux.png", bbox_inches="tight")
if save_colorbar == 1:
    plt.colorbar(plt0, ax=ax0)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.pause(0.5)
    plt.savefig(savedir + "ux_colorbar.png", bbox_inches="tight")

##################
# diffraction on Ge 400 peak
##################
q400_Ge = 2 * np.pi / d400_Ge  # inverse nm
avg_q = np.matrix([q400_Ge, 0])
dq = 2 * np.pi / (2 * half_window * aGe)  # inverse nm
qx = q400_Ge + np.arange(-dq * half_window, dq * half_window, dq)
qy = np.arange(-dq * half_window, dq * half_window, dq)

########################
# FFT with displacement field and symmetric
# normalization for comparison with mathematica
########################
complex_object = nanowire * np.exp(1j * (ux * avg_q[0, 0] + 0))
np.save(savedir + "GeSi_NW_complex_object.npy", complex_object)

print("Min(abs(object)", abs(complex_object).min())
print("Max(abs(object)", abs(complex_object).max())
amplitude = fftshift(fftn(nanowire * np.exp(1j * (ux * avg_q[0, 0] + 0)), norm="ortho"))
print("Min(abs(amplitude)", abs(amplitude).min())  # should be same as mathematica
print("Max(abs(amplitude)", abs(amplitude).max())  # should be same as mathematica
intensity = abs(amplitude) ** 2
print(
    "Min(log10(intensity)", np.log10(intensity).min()
)  # should be same as mathematica
print(
    "Max(log10(intensity)", np.log10(intensity).max()
)  # should be same as mathematica

fig, ax0 = plt.subplots(1, 1)
plt0 = ax0.imshow(
    np.log10(intensity),
    extent=(qx.min(), qx.max(), qy.min(), qy.max()),
    cmap="jet",
    vmin=-7,
    vmax=4,
)
ax0.tick_params(
    labelbottom="off", labelleft="off", bottom="off", left="off", top="off", right="off"
)
plt.pause(0.5)
plt.savefig(savedir + "FFT_positive" + comment + "_ortho_jet.png", bbox_inches="tight")
if save_colorbar == 1:
    plt.colorbar(plt0, ax=ax0)
    plt.xlabel("Qx")
    plt.ylabel("Qy")
    ax0.tick_params(
        labelbottom="on",
        labelleft="on",
        labelsize=12,
        direction="out",
        top="on",
        right="on",
        length=tick_length,
        width=tick_width,
    )
    plt.pause(0.5)
    plt.savefig(
        savedir + "FFT_positive" + comment + "_ortho_colorbar_jet.png",
        bbox_inches="tight",
    )

fig, ax0 = plt.subplots(1, 1)
plt0 = ax0.imshow(
    np.log10(
        intensity[
            half_window - 20 : half_window + 20, half_window - 20 : half_window + 20
        ]
    ),
    cmap="jet",
    vmin=-7,
    vmax=4,
)
ax0.tick_params(
    labelbottom="off", labelleft="off", bottom="off", left="off", top="off", right="off"
)
plt.pause(0.5)
plt.savefig(
    savedir + "FFT_positive" + comment + "_ortho_zoom_jet.png", bbox_inches="tight"
)

########################
# FFT with displacement field of opposite sign and
# symmetric normalization for comparison with mathematica
########################
amplitude = fftshift(
    fftn(nanowire * np.exp(1j * (-ux * avg_q[0, 0] + 0)), norm="ortho")
)
print("Min(abs(amplitude)", abs(amplitude).min())  # should be same as mathematica
print("Max(abs(amplitude)", abs(amplitude).max())  # should be same as mathematica
intensity = abs(amplitude) ** 2
print(
    "Min(log10(intensity)", np.log10(intensity).min()
)  # should be same as mathematica
print(
    "Max(log10(intensity)", np.log10(intensity).max()
)  # should be same as mathematica

fig, ax0 = plt.subplots(1, 1)
plt0 = ax0.imshow(
    np.log10(intensity),
    extent=(qx.min(), qx.max(), qy.min(), qy.max()),
    cmap="jet",
    vmin=-7,
    vmax=4,
)
ax0.tick_params(
    labelbottom="off", labelleft="off", bottom="off", left="off", top="off", right="off"
)
plt.pause(0.5)
plt.savefig(savedir + "FFT_negative" + comment + "_ortho_jet.png", bbox_inches="tight")

fig, ax0 = plt.subplots(1, 1)
plt0 = ax0.imshow(
    np.log10(
        intensity[
            half_window - 20 : half_window + 20, half_window - 20 : half_window + 20
        ]
    ),
    cmap="jet",
    vmin=-7,
    vmax=4,
)
ax0.tick_params(
    labelbottom="off", labelleft="off", bottom="off", left="off", top="off", right="off"
)
plt.pause(0.5)
plt.savefig(
    savedir + "FFT_negative" + comment + "_ortho_zoom_jet.png", bbox_inches="tight"
)

########################
# FFT with displacement field and default normalization
########################
intensity = abs(fftshift(fftn(nanowire * np.exp(1j * (ux * avg_q[0, 0] + 0))))) ** 2

fig, ax0 = plt.subplots(1, 1)
plt0 = ax0.imshow(
    np.log10(intensity),
    extent=(qx.min(), qx.max(), qy.min(), qy.max()),
    cmap=my_cmap,
    vmin=colorbar_range[0],
    vmax=colorbar_range[1],
)
ax0.tick_params(
    labelbottom="off",
    labelleft="off",
    top="on",
    right="on",
    labelsize=12,
    direction="out",
    length=tick_length,
    width=tick_width,
)
plt.pause(0.5)
plt.savefig(savedir + "FFT_positive" + comment + ".png", bbox_inches="tight")
np.savez_compressed(savedir + "GeSi_NW_FFT_positive.npz", obj=intensity)
if save_colorbar == 1:
    plt.colorbar(plt0, ax=ax0)
    plt.xlabel("Qx")
    plt.ylabel("Qy")
    ax0.tick_params(
        labelbottom="on",
        labelleft="on",
        labelsize=12,
        direction="out",
        top="on",
        right="on",
        length=tick_length,
        width=tick_width,
    )
    plt.pause(0.5)
    plt.savefig(
        savedir + "FFT_positive" + comment + "_colorbar.png", bbox_inches="tight"
    )

fig, ax0 = plt.subplots(1, 1)
plt0 = ax0.imshow(
    np.log10(
        intensity[
            half_window - 20 : half_window + 20, half_window - 20 : half_window + 20
        ]
    ),
    cmap=my_cmap,
    vmin=colorbar_range[0],
    vmax=colorbar_range[1],
)
ax0.tick_params(
    labelbottom="off", labelleft="off", bottom="off", left="off", top="off", right="off"
)
plt.pause(0.5)
plt.savefig(savedir + "FFT_positive" + comment + "_zoom.png", bbox_inches="tight")

########################
# FFT with displacement field of opposite sign and default normalization
########################
intensity = abs(fftshift(fftn(nanowire * np.exp(1j * (-ux * avg_q[0, 0] + 0))))) ** 2

fig, ax0 = plt.subplots(1, 1)
plt0 = ax0.imshow(
    np.log10(intensity),
    extent=(qx.min(), qx.max(), qy.min(), qy.max()),
    cmap=my_cmap,
    vmin=colorbar_range[0],
    vmax=colorbar_range[1],
)
ax0.tick_params(
    labelbottom="off",
    labelleft="off",
    direction="out",
    top="on",
    right="on",
    length=tick_length,
    width=tick_width,
)
plt.pause(0.5)
plt.savefig(savedir + "FFT_negative" + comment + ".png", bbox_inches="tight")
np.savez_compressed(savedir + "GeSi_NW_FFT_negative.npz", obj=intensity)

fig, x0 = plt.subplots(1, 1)
plt0 = x0.imshow(
    np.log10(
        intensity[
            half_window - 20 : half_window + 20, half_window - 20 : half_window + 20
        ]
    ),
    cmap=my_cmap,
    vmin=colorbar_range[0],
    vmax=colorbar_range[1],
)
x0.tick_params(
    labelbottom="off", labelleft="off", bottom="off", left="off", top="off", right="off"
)
plt.pause(0.5)
plt.savefig(savedir + "FFT_negative" + comment + "_zoom.png", bbox_inches="tight")

#######################
# kinematic sums
#######################
nanowire_zoom = nanowire[
    half_window - 50 : half_window + 50, half_window - 50 : half_window + 50
]
plt.figure()
plt.imshow(nanowire_zoom)

qx = q400_Ge + np.arange(-dq * half_window, dq * half_window, dq)
qy = np.arange(-dq * half_window, dq * half_window, dq)

grid_x = xgrid + ux
grid_y = ygrid

grid_x = grid_x[
    half_window - 50 : half_window + 50, half_window - 50 : half_window + 50
]
grid_y = grid_y[
    half_window - 50 : half_window + 50, half_window - 50 : half_window + 50
]

qx1 = np.repeat(qx[np.newaxis, :], len(qy), axis=0)
qy1 = np.repeat(qy[:, np.newaxis], len(qx), axis=1)

##############################
# calculate the centered kinematic sum +1j +ux
##############################
Fhk1 = np.zeros((len(qy), len(qx))).astype(np.complex64)
for ii in range(len(qy)):
    for jj in range(len(qx)):
        Fhk1[ii, jj] = (
            Fhk1[ii, jj]
            + (
                nanowire_zoom
                * np.exp(+1j * (qx1[ii, jj] * grid_x + qy1[ii, jj] * grid_y))
            ).sum()
        )

intensity = abs(Fhk1) ** 2
fig, ax0 = plt.subplots(1, 1)
ax0.imshow(
    np.log10(intensity),
    extent=(qx.min(), qx.max(), qy.min(), qy.max()),
    cmap=my_cmap,
    vmin=colorbar_range[0],
    vmax=colorbar_range[1],
)
ax0.tick_params(
    labelbottom="off",
    labelleft="off",
    direction="out",
    top="on",
    right="on",
    length=tick_length,
    width=tick_width,
)
plt.pause(0.5)
plt.savefig(savedir + "Kinematic_+1j_+ux" + comment + ".png", bbox_inches="tight")
np.savez_compressed(savedir + "Kinematic_+1j_+ux.npz", obj=abs(Fhk1) ** 2)
if save_colorbar == 1:
    plt.colorbar(plt0, ax=ax0)
    plt.xlabel("Qx")
    plt.ylabel("Qy")
    ax0.tick_params(
        labelbottom="on",
        labelleft="on",
        labelsize=12,
        direction="out",
        top="on",
        right="on",
        length=tick_length,
        width=tick_width,
    )
    plt.pause(0.5)
    plt.savefig(
        savedir + "Kinematic_+1j_+ux" + comment + "_colorbar.png", bbox_inches="tight"
    )

fig, x0 = plt.subplots(1, 1)
plt0 = x0.imshow(
    np.log10(
        intensity[
            half_window - 20 : half_window + 20, half_window - 20 : half_window + 20
        ]
    ),
    cmap=my_cmap,
    vmin=colorbar_range[0],
    vmax=colorbar_range[1],
)
x0.tick_params(
    labelbottom="off", labelleft="off", bottom="off", left="off", top="off", right="off"
)
plt.pause(0.5)
plt.savefig(
    savedir + "GeSi_NW_kinsum_+1j_+ux" + comment + "_zoom.png", bbox_inches="tight"
)

##############################
# calculate the centered kinematic sum -1j +ux
##############################
Fhk1 = np.zeros((len(qy), len(qx))).astype(np.complex64)
for ii in range(len(qy)):
    for jj in range(len(qx)):
        Fhk1[ii, jj] = (
            Fhk1[ii, jj]
            + (
                nanowire_zoom
                * np.exp(-1j * (qx1[ii, jj] * grid_x + qy1[ii, jj] * grid_y))
            ).sum()
        )
intensity = abs(Fhk1) ** 2

fig, ax0 = plt.subplots(1, 1)
ax0.imshow(
    np.log10(intensity),
    extent=(qx.min(), qx.max(), qy.min(), qy.max()),
    cmap=my_cmap,
    vmin=colorbar_range[0],
    vmax=colorbar_range[1],
)
ax0.tick_params(
    labelbottom="off",
    labelleft="off",
    direction="out",
    top="on",
    right="on",
    length=tick_length,
    width=tick_width,
)
plt.pause(0.5)
plt.savefig(savedir + "Kinematic_-1j_+ux" + comment + ".png", bbox_inches="tight")
np.savez_compressed(savedir + "GeSi_NW_kinsum_-1j_+ux.npz", obj=abs(Fhk1) ** 2)

fig, x0 = plt.subplots(1, 1)
plt0 = x0.imshow(
    np.log10(
        intensity[
            half_window - 20 : half_window + 20, half_window - 20 : half_window + 20
        ]
    ),
    cmap=my_cmap,
    vmin=colorbar_range[0],
    vmax=colorbar_range[1],
)
x0.tick_params(
    labelbottom="off", labelleft="off", bottom="off", left="off", top="off", right="off"
)
plt.pause(0.5)
plt.savefig(savedir + "Kinematic_-1j_+ux" + comment + "_zoom.png", bbox_inches="tight")

plt.ioff()
plt.show()
print("end")
