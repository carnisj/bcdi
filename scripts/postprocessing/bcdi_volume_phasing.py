#!/usr/bin/env python3

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import os

import matplotlib.pyplot as plt
import numpy as np

helptext = """
Calculate the volume of the reconstruction for different isosurfaces, save and plot
the result.
"""

datadir = "D:/review paper/BCDI_isosurface/S2227/simu/crop400phase/no_apodization/avg1/"
savedir = "D:/review paper/BCDI_isosurface/S2227/simu/crop400phase/no_apodization/"
voxel_size = (
    3  # voxel size of the reconstruction in nm (voxels are supposed to be isotropic)
)
isosurface = np.arange(1, 40) / 40
filename = "S2227_ampphasestrain_4_threshold_iso_0.73_avg1_crystal-frame.npz"
comment = "_5"  # should start with _
##########################
# end of user parameters #
##########################
amp = np.load(datadir + filename)["amp"]
amp = amp / amp.max()


volume_file = open(
    os.path.join(savedir, "volume_vs_isosurface_noapod" + comment + ".dat"), "w"
)
volume_file.write(f"{'isosurface': <10}" + "\t" + f"{'volume (um3)': <10}" + "\n")


volume = np.zeros(len(isosurface))
index = 0
for iso in isosurface:
    temp_obj = np.copy(amp)
    temp_obj[temp_obj < iso] = 0
    temp_obj[np.nonzero(temp_obj)] = 1
    volume[index] = (0.001 * voxel_size) ** 3 * temp_obj.sum()  # convert to um3
    volume_file.write(f"{str(iso): <10}" + "\t" + f"{str(volume[index]): <10}" + "\n")
    index = index + 1
volume_file.close()

amp[amp == 0] = np.nan  # avoid the 0 peak in amplitude histogram
plt.ion()
plt.figure()
plt.hist(amp.flatten(), bins=50)
plt.xlim(left=0.05)
plt.savefig(savedir + "amplitude_histogram" + comment + ".png")

plt.figure()
plt.plot(isosurface, volume)
plt.xlim((0, 1))
plt.ylim(bottom=0)
plt.xlabel("Normalized isosurface")
plt.ylabel("Volume (um3)")
plt.savefig(savedir + "volume_vs_isosurface" + comment + ".png")
plt.ioff()
plt.show()
