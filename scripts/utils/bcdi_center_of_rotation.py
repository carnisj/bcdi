#!/usr/bin/env python3

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

from math import pi, tan

helptext = """
Calculate the vertical correction and correction along the beam to apply to a
nanocrystal, given the drift along the beam at -/+ delta_angle from the center of the
rocking curve.
"""

incident_angle = 21.0326  # in degrees
delta_angle = 0.4  # in degrees, the motor position is checked at -/+ delta_angle
# from the incident_angle
drift_um = 2.4  # drift of the motor position along the beam, in micrometers

delta_vertical = (
    -1
    * drift_um
    * tan(incident_angle * pi / 180) ** 2
    / (
        tan((incident_angle + delta_angle) * pi / 180)
        - tan((incident_angle - delta_angle) * pi / 180)
    )
)
print(f"delta_vertical = {delta_vertical} um")

delta_horizontal = -1 * delta_vertical / tan(incident_angle * pi / 180)
print(f"delta_horizontal = {delta_horizontal} um")
