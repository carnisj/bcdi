#!/usr/bin/env python3

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import matplotlib.pyplot as plt
import numpy as np

helptext = """
Calculate the FT of an object with displacement, using kinematic sum.

Author: Stephane Labat @ IM2NP
Modif: Jerome Carnis
"""

support = np.ones((32, 32))
x, y = np.indices((32, 32))
sample = support * (x - 16 - 2 * y) < 0
plt.imshow(sample)
plt.show()

x0 = 24
y0 = 1
# displ = np.sin(2*pi*(x-20)/40)*((x-x0)**2+(y-y0)**2)/2000
displacement = ((x - x0) ** 2 + (y - y0) ** 2) / 3000 - 0.2

plt.imshow(sample)
plt.colorbar()
plt.show()

repres = np.zeros((40, 40))
repres[4:36, 4:36] = displacement * sample

plt.imshow(repres)
plt.colorbar()
plt.show()

tay = np.array([256, 256])
qx = 5
qy = 12
# amp=sum(obj*exp(1j*(qx*(x+displ)+qy*y)))
# amp=sum(samp*exp(1j*(qx*(x+displ)+qy*y)))
# amplit=sum(samp*np.exp(1j*2*np.pi*(qx*(x+displ)/tay[0]+qy*y/tay[1])))
intensite = (
    abs(
        sum(
            sample
            * np.exp(
                -1j * 2 * np.pi * (qx * (x / tay[0] + displacement) + qy * y / tay[1])
            )
        )
    )
    ** 2
)

h = 1
k = 1
tay = np.array([256, 256])
intensity = np.zeros(tay)
for qx in range(tay[0]):
    for qy in range(tay[1]):
        intensity[qx, qy] = (
            abs(
                sum(
                    sample
                    * np.exp(
                        -1j
                        * 2
                        * np.pi
                        * (
                            (qx / tay[0] + h - 1 / 2) * (x + displacement)
                            + (qy / tay[1] + k - 1 / 2) * y
                        )
                    )
                )
            )
            ** 2
        )
plt.imshow(np.log10(intensity), vmax=6, vmin=1)
plt.show()

# newintens=np.roll(np.roll(intens, 35, axis=0), 35, axis=1)
# plt.imshow(np.log10(newintens), vmax=4, vmin=-1)
# plt.show()

# save intensity file
filename = "C:/Users/carnis/Work Folders/Documents/data/simulationintensite_test.npy"
np.save(filename, intensity)
