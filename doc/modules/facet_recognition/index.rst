.. py:module:: bcdi.facet_recognition

.. toctree::
   :maxdepth: 2

:mod:`bcdi.facet_recognition`: Stereographic projection, facet detection and statistics on strain
=================================================================================================

Description
-----------

This module provides tools for plotting the stereographic projection of a diffraction
peak or an object. There is also a script for facet detection on a reconstructed
object, and for calculating statistics on facet strain. After meshing the object,
facets are found using a density estimation of mesh triangles normals, followed by
,watershed segmentation.
See Carnis et al. Small 17, 2007702 (2021)
https://doi.org/10.1002/smll.202007702

API Reference
-------------

facet_utils
^^^^^^^^^^^

.. automodule:: bcdi.facet_recognition.facet_utils
   :members:
