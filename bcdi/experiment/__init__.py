# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction
# imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#   (c) 06/2021-present : DESY CFEL
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

"""
BCDI experiment.

This package contains classes related to the experimental setup.

Classes
-------

- ``Detector``         -- detector declaration
- ``Diffractometer``   -- diffractometer, one child class for each
                          supported beamline
- ``Setup``            -- class gathering various parameters about the
                          setup
- ``RotationMatrix``   -- rotation matrix given the rotation axis and
                          the angle

"""
