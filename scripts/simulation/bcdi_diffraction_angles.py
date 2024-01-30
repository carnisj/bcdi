#!/usr/bin/env python3

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import sys

import numpy as np
import xrayutilities as xu

import bcdi.experiment.setup as exp
import bcdi.simulation.simulation_utils as simu

helptext = """
Calculate the position of the Bragg peaks for a material and a particular
diffractometer setup. The crystal frame uses the following convention: x downstream,
y outboard, z vertical.
Supported beamlines:  ESRF ID01, SOLEIL CRISTAL, SOLEIL SIXS and PETRAIII P10.
"""

material = xu.materials.Pt  # load material from materials submodule
beamline = "P10"  # 'ID01' or 'P10'
energy = 8200  # x-ray energy in eV
beam_direction = (1, 0, 0)  # beam along z
sample_inplane = material.Q(
    -2, 1, 1
)  # sample Bragg reflection along the primary beam at 0 angles
sample_outofplane = material.Q(
    1, 1, 1
)  # sample Bragg reflection perpendicular to the primary beam and
# the innermost detector rotation axis
basis_vectors = (
    np.array([1, 0, 0]),
    np.array([0, 1, 0]),
    np.array([0, 0, 1]),
)  # tuple of the three components of the basis vectors expressed in the orthonormal
# basis. The convention used for the orthonormal basis is
# ([1, 0, 0], [0, 1, 0], [0, 0, 1]).
reflections = [[1, -1, 1]]  # list of reflections to calculate [[2,1,1], [1,-1,-1],...]
bounds = (
    0,
    5,
    0,
    (-45, 45),
    (-2, 90),
    (-1, 70),
)  # bound values for the goniometer angles.
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
setup = exp.Setup(
    beamline_name=beamline,
    energy=energy,
    beam_direction=beam_direction,
    sample_inplane=sample_inplane,
    sample_outofplane=sample_outofplane,
)

qconv, _ = setup.init_qconversion()

#################################################################
# initialize experimental class with directions from experiment #
#################################################################
hxrd = xu.experiment.HXRD(sample_inplane, sample_outofplane, qconv=qconv, en=energy)

if (
    simu.angle_vectors(sample_inplane, sample_outofplane, basis_vectors=basis_vectors)
    != 90.0
):
    print("The angle between reference directions is not 90 degrees")
    sys.exit()
if len(qconv.sampleAxis) + len(qconv.detectorAxis) != len(bounds):
    raise ValueError("number of specified bounds invalid")
nb_free = 0
for idx, item in enumerate(bounds):
    if type(item) is tuple:
        nb_free = nb_free + 1
if nb_free > 3:
    raise ValueError("maximum of three free motors exceeded")

#############################################
# calculate the angles of Bragg reflections #
#############################################
comment = material.name
nb_reflex = len(reflections)
for idx in range(nb_reflex):
    hkl = reflections[idx]
    q_material = material.Q(hkl)
    q_laboratory = hxrd.Transform(q_material)
    print(
        comment,
        "   hkl=",
        hkl,
        "   q=",
        np.round(q_material, 5),
        "   lattice spacing= %.4f" % material.planeDistance(hkl),
        "angstroms",
    )

    # determine the goniometer angles with the correct geometry restrictions
    ang, qerror, errcode = xu.Q2AngFit(q_laboratory, hxrd, bounds)
    print("angles %s" % (str(np.round(ang, 5))))
    # check that qerror is small!!
    print(
        "sanity check with back-transformation (hkl): ",
        np.round(hxrd.Ang2HKL(*ang, mat=material), 5),
        "\n",
    )
