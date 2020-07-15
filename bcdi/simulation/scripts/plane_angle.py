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

alpha = 76  # in degrees, angle between b2 and b3
beta = 78  # in degrees, angle between b1 and b3
gamma = 87  # in degrees, angle between b1 and b2
b1 = 0.103  # length of b1
b2 = 0.103  # length of b2
b3 = 0.108  # length of b3
##################################
# end of user-defined parameters #
##################################

####################################################################################################
# calculate the basis vector components in the orthonormal basis [[1, 0, 0], [0, 1, 0], [0, 0, 1]] #
####################################################################################################
first_vector = b1 * np.array([1, 0, 0])  # the convention here is to align b1 along [1, 0, 0]
second_vector = b2 * np.cos(gamma*np.pi/180) * np.array([1, 0, 0]) + b2 * np.sin(gamma*np.pi/180) * np.array([0, 1, 0])
# b2 is in the plane defined by the vectors [1, 0, 0] and [0, 1, 0]
cx = b3 * np.cos(beta*np.pi/180)
cy = b3 * (np.cos(alpha*np.pi/180) - np.cos(beta*np.pi/180)*np.cos(gamma*np.pi/180))/np.sin(gamma*np.pi/180)
cz = np.sqrt(b3**2 - cx**2 - cy**2)
third_vector = cx * np.array([1, 0, 0]) + cy * np.array([0, 1, 0]) + cz * np.array([0, 0, 1])
basis = (first_vector, second_vector, third_vector)
print('Volume of the unit cell: {:.6f}'.format(b1*b2*cz*np.sin(gamma*np.pi/180)))
##############################################################
# calculate the angle between reference_plane and test_plane #
##############################################################
angle = simu.angle_vectors(ref_vector=reference_plane, test_vector=test_plane, basis_vectors=basis)
print('angle = {:.2f} deg'.format(angle))
