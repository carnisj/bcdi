# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import sys
sys.path.append('D:/myscripts/bcdi/')
import bcdi.facet_recognition.facet_utils as fu

helptext = """
Calculate the angle between to crystallographic planes in cubic materials.
"""

reference_plane = [1, 1, 1]  # [0.37182, 0.78376, -0.49747]  # [0.40975, 0.29201, -0.86420]
second_plane = [1, -1, 1]  # [-0.22923, 0.76727, -0.59896]  # [-0.19695, 0.27933, -0.93978]
angle = fu.plane_angle_cubic(reference_plane, second_plane)
print('angle=', str(angle), 'deg')
