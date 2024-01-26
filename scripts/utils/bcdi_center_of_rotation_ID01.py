#!/usr/bin/env python3

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Marie-Ingrid Richard, mrichard@esrf.fr

from math import pi

import numpy as np

helptext = """
Calculate the vertical correction and correction along the beam to apply to a
nanocrystal to move it at the center of rotation along the beam direction.
At two eta angles, perform a piy-scan of the particle.
"""


piy_array = [72.026, 70.826]  # in um
eta_array = [24.63, 25.03]  # in degrees

piz = (
    (piy_array[1] - piy_array[0])
    * np.tan(eta_array[0] * pi / 180)
    * np.tan(eta_array[1] * pi / 180)
    / (np.tan(eta_array[1] * pi / 180) - np.tan(eta_array[0] * pi / 180))
)
piy = piz / np.tan((eta_array[0] + eta_array[1]) * pi / 180 / 2.0)
print(f"Move relatively piz by {piz:.2f} um, piy by {piy:.2f} um")
