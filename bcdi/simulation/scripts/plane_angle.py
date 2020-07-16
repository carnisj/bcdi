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
Calculate the angle between to crystallographic planes expressed in the triclinic crystal system.
"""

reference_plane = [1, 0, 0]  # in the basis (b1, b2, b3)
test_plane = [1, 0, 1]  # in the basis (b1, b2, b3)
use_directlattice = True  # if True, it will use the direct lattice parameters to calculate the reciprocal lattice
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
alpha_r = 76  # in degrees, angle between b2 and b3
beta_r = 78  # in degrees, angle between b1 and b3
gamma_r = 87  # in degrees, angle between b1 and b2
b1 = 0.103  # length of b1 in 1/nm
b2 = 0.103  # length of b2 in 1/nm
b3 = 0.108  # length of b3 in 1/nm
##################################
# end of user-defined parameters #
##################################

####################################################################################################
# calculate the basis vector components in the orthonormal basis [[1, 0, 0], [0, 1, 0], [0, 0, 1]] #
####################################################################################################
if use_directlattice:
    alpha_r, beta_r, gamma_r, b1, b2, b3 = simu.reciprocal_lattice(alpha, beta, gamma, a1, a2, a3,
                                                                   input_lattice='direct', verbose=True)
basis = simu.triclinic_to_basis(alpha_r, beta_r, gamma_r, b1, b2, b3)
volume = basis[0].dot(np.cross(basis[1], basis[2]))
print('Volume of the reciprocal unit cell: {:.6f} nm\u207B\u00B3'.format(volume))
##############################################################
# calculate the angle between reference_plane and test_plane #
##############################################################
angle = simu.angle_vectors(ref_vector=reference_plane, test_vector=test_plane, basis_vectors=basis)
print('angle = {:.2f} deg'.format(angle))
