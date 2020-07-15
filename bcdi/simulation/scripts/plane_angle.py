# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np
import sys
sys.path.append('D:/myscripts/bcdi/')
import bcdi.simulation.simulation_utils as simu

helptext = """
Calculate the angle between to crystallographic planes expressed in the crystal basis.
"""

reference_plane = [0, 1, 0]  # [0.37182, 0.78376, -0.49747]  # [0.40975, 0.29201, -0.86420]
second_plane = [1, 1, 0]  # [-0.22923, 0.76727, -0.59896]  # [-0.19695, 0.27933, -0.93978]
basis = (np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]))  # components of the basis vectors in the
# orthonormal basis ((1, 0, 0), (0, 1, 0), (0, 0, 1))

angle = simu.angle_vectors(ref_vector=reference_plane, test_vector=second_plane, basis_vectors=basis)
print('angle = {:.2f} deg'.format(angle))
