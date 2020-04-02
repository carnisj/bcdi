# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np
import xrayutilities as xu
import sys
sys.path.append('C:/Users/Jerome/Documents/myscripts/bcdi/')
import bcdi.experiment.experiment_utils as exp
import bcdi.postprocessing.postprocessing_utils as pu
import bcdi.preprocessing.preprocessing_utils as pru

helptext = """
Calculate the position of the Bragg peaks for a material and a particular diffractometer setup. 
The crystal frame uses the following convention: x downstream, y outboard, z vertical
Supported beamlines:  ESRF ID01, SOLEIL CRISTAL, SOLEIL SIXS and PETRAIII P10"""

material = xu.materials.Pt  # load material from materials submodule
beamline = 'ID01'  # 'ID01' or 'P10'
energy = 9000  # x-ray energy in eV
beam_direction = (1, 0, 0)  # beam along z
sample_inplane = material.Q(-2, 1, 1)  # sample Bragg reflection along the primary beam at 0 angles
sample_outofplane = material.Q(1, 1, 1)  # sample Bragg reflection perpendicular to the primary beam and
# the innermost detector rotation axis
sample_offsets = (0, 0, 0)
reflections = [[1, 1, 2]]  # list of reflections to calculate [[2,1,1], [1,-1,-1],...]
bounds = ((-1, 90), 0, 0, (-90, 90), (-1, 90))  # bound values for the goniometer angles.
# (min,max) pair or fixed value for all motors, with a maximum of three free motors
# ID01      sample: eta, chi, phi      detector: nu,del
# SIXS      sample: beta, mu     detector: beta, gamma, del
# CRISTAL   sample: mgomega    detector: gamma, delta
# P10       sample: mu, omega, chi,phi   detector: gamma, delta
##################################
# end of user-defined parameters #
##################################

####################
# Initialize setup #
####################
setup = exp.SetupPreprocessing(beamline=beamline, energy=energy, rocking_angle=None, distance=None,
                               beam_direction=beam_direction, sample_inplane=sample_inplane,
                               sample_outofplane=sample_outofplane)

qconv, _ = pru.init_qconversion(setup)

#################################################################
# initialize experimental class with directions from experiment #
#################################################################
hxrd = xu.experiment.HXRD(sample_inplane, sample_outofplane, qconv=qconv,  en=energy)

if pu.plane_angle(sample_inplane, sample_outofplane) != 90.0:
    print("The angle between reference directions is not 90 degrees", )
    sys.exit()

#############################################
# calculate the angles of Bragg reflections #
#############################################
comment = material.name
nb_reflex = len(reflections)
for idx in range(nb_reflex):
    hkl = reflections[idx]
    q_material = material.Q(hkl)
    q_laboratory = hxrd.Transform(q_material)
    print(comment, '   hkl=', hkl, '   q=', np.round(q_material, 5),
          '   lattice spacing= %.4f' % material.planeDistance(hkl), 'angstroms')

    # determine the goniometer angles with the correct geometry restrictions
    ang, qerror, errcode = xu.Q2AngFit(q_laboratory, hxrd, bounds)
    print('angles %s' % (str(np.round(ang, 5))))
    # check that qerror is small!!
    print('sanity check with back-transformation (hkl): ',
          np.round(hxrd.Ang2HKL(*ang, mat=material), 5), '\n')
