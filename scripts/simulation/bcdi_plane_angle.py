#!/usr/bin/env python3

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np

import bcdi.simulation.simulation_utils as simu

helptext = """
Calculate the angle between to crystallographic planes expressed in the triclinic
crystal system.
"""

reference_planes = [[1, 0, 1], [-1, 0, 1], [1, 0, -1], [1, 0, 0]]

# list of reference planes in the basis (b1, b2, b3)
test_planes = [
    [1, 0, 0],
    [-1, 0, 0],
    [-1, 0, 1],
]  # [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # list of test planes in the basis (b1, b2, b3)
use_directlattice = False
# if True, it will use the direct lattice parameters to calculate the reciprocal lattice
#############################
# define the direct lattice #
#############################
alpha = 75  # in degrees, angle between a2 and a3
beta = 75  # in degrees, angle between a1 and a3
gamma = 90  # in degrees, angle between a1 and a2
a1 = 63.2  # length of a1 in nm
a2 = 63.2  # length of a2 in nm
a3 = 61.2  # length of a3 in nm
#############################################
# or define direclty the reciprocal lattice #
#############################################
alpha_r = 75  # in degrees, angle between b2 and b3
beta_r = 75  # in degrees, angle between b1 and b3
gamma_r = 86.5  # in degrees, angle between b1 and b2
b1 = 0.103  # length of b1 in 1/nm
b2 = 0.103  # length of b2 in 1/nm
b3 = 0.108  # length of b3 in 1/nm
##################################
# end of user-defined parameters #
##################################

#######################################################
# calculate the basis vector components in the        #
# orthonormal basis [[1, 0, 0], [0, 1, 0], [0, 0, 1]] #
#######################################################
if use_directlattice:
    print("Using the direct lattice to calculate the reciprocal lattice")
    alpha_r, beta_r, gamma_r, b1, b2, b3 = simu.reciprocal_lattice(
        alpha, beta, gamma, a1, a2, a3, input_lattice="direct", verbose=True
    )
basis = simu.triclinic_to_basis(alpha_r, beta_r, gamma_r, b1, b2, b3)
volume = basis[0].dot(np.cross(basis[1], basis[2]))
print(f"Volume of the reciprocal unit cell: {volume:.6f} nm\u207B\u00B3")
##############################################################
# calculate the angle between reference_plane and test_plane #
##############################################################
for idx in range(len(reference_planes)):
    for idy in range(len(test_planes)):
        print(
            "ref_plane = "
            + str(reference_planes[idx])
            + "\ttest_plane = "
            + str(test_planes[idy])
            + "\tangle = {:.2f} deg".format(
                simu.angle_vectors(
                    ref_vector=reference_planes[idx],
                    test_vector=test_planes[idy],
                    basis_vectors=basis,
                )
            )
        )
