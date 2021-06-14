![python-packages](https://github.com/carnisj/bcdi/actions/workflows/python-packages/badge.svg)
.. image:: https://readthedocs.org/projects/bcdi/badge/?version=latest
[![DeepSource](https://deepsource.io/gh/carnisj/bcdi.svg/?label=active+issues&show_trend=true&token=N3Z0cklmQrG8kzZOVwGJhLd9)](https://deepsource.io/gh/carnisj/bcdi/?ref=repository-badge)

BCDI: tools for pre(post)-processing Bragg and forward coherent X-ray diffraction imaging data
==============================================================================================

Introduction
============

BCDI stands for *Bragg coherent X-ray diffraction imaging*. It can be used for:

* pre-processing BCDI and forward CDI data (masking aliens, detector gaps ...) before phase retrieval

* post-processing phased data (phase offset and phase ramp removal, averaging, apodization, ...)

* data analysis on diffraction data (stereographic projection, angular cross-correlation analysis, domain orientation fitting ...)

* data analysis on phased data (resolution calculation, statistics on the retrieved strain ...)

* simulation of diffraction intensity (including noise, detector gaps, displacement field ...)

* creating figures for publication using templates


BCDI as a python toolkit
========================

BCDI can be used as a python library with the following main modules:

1) :mod:`bcdi.algorithms`: PSF and image deconvolution using Richardson-Lucy algorithm

2) :mod:`bcdi.facet_recognition`: Stereographic projection of a diffraction peak or a reconstructed crystal. Automatic detection of reconstructed facets and statistics on facet strain.

3) :mod:`bcdi.experiment`: definition of the experimental geometry (beamline, setup, detector...).

4) :mod:`bcdi.graph` : generation of plots using predefined templates.

5) :mod:`bcdi.postprocessing`: various methods for post-processing the complex output of a phasing algorithm.

6) :mod:`bcdi.preprocessing`: various methods for pre-processing the diffraction intensity.

7) :mod:`bcdi.simulation`: in BCDI geometry, calculation of the diffraction intensity based on FFT or kinematical sum.
   It can include a displacement field, noise, detector gaps etc... In forward CDI geometry, calculation of the Bragg
   peak positions in 3D for a mesocrystal, knowing the unit cell and unit cell parameter.

8) :mod:`bcdi.utils`: data loading, fitting functions ...

9) :mod:`bcdi.xcca`: X-ray cross-correlation analysis related methods

Acknowledgment and third party packages
=======================================

We would like to acknowledge the following packages:

* xrayutilities: (c) Dominik Kriegner, Eugen Wintersberger. See: J. Appl. Cryst. 46, 1162-1170 (2013).

* nxsReady: (c) Andrea Resta @ SOLEIL SIXS

* image_registration.py: original code from Xianhui Xiao @ APS Sector 2. See: Opt. Lett. 33, 156-158 (2008).

* Some functions were adapted from PyNX: (c) Vincent Favre-Nicolin. See: http://ftp.esrf.fr/pub/scisoft/PyNX/ and J. Appl. Cryst. 49, 1842-1848 (2016).

The following third-party packages are required:

* numpy

* scipy

* scikit-image

* matplotlib

* mayavi

* vtk

* h5py

* hdf5plugin

* fabio

* silx

* xrayutilities

* lmfit: for scripts performing fits

* pytest if you want to execute the tests

* pytables: when loading the dictionnary for SIXS data

* moviepy, imagemagick (https://imagemagick.org) or ffmpeg (http://ffmpeg.zeranoe.com/builds/) for creating movies

Download & Installation
=======================

BCDI is available from:
 * Python Package Index: pip install bcdi
 * https://github.com/carnisj  for the most updated version
 * upgrade existing version from GitHub: pip install --upgrade git+https://github.com/carnisj/bcdi.git

Please send feedback in GitHub or at carnis_jerome@yahoo.fr

Changelog
=========

.. include:: ../HISTORY.rst
  :end-before: Version 0.0.8

See the full :doc:`Changelog<changelog>`

Citation & Bibliography
=======================

If you use BCDI for scientific work, please consider including a citation (DOI: 10.5281/zenodo.3257616).

License
=======
The BCDI library is distributed with a CeCILL-B license (an open-source license similar to the FreeBSD one).
See http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html

Documentation
=============

The documentation is available at: https://bcdi.readthedocs.io/en/latest/

BCDI.algorithms: psf and image deconvolution algorithms
=======================================================

.. bcdi.algorithms section

Description
-----------

This module includes routines using Richardson-Lucy deconvolution algorithm.

.. bcdi.algorithms end

BCDI.facet_recognition: automatic facet detection in BCDI 3D reconstructions
============================================================================

.. bcdi.facet_recognition section

Description
-----------

This module provides tools for plotting the stereographic projection of a diffraction peak or an object. 
There is also a script for facet detection on a reconstructed object, and for calculating statistics on facet strain.
After meshing the object, facets are found using a density estimation of mesh triangles normals, followed by watershed segmentation.

.. bcdi.facet_recognition end

BCDI.experiment: class and methods defining the experimental setup
================================================================

.. bcdi.experiment section

Description
-----------

This module provides a class and methods for the definition of the experimental setup.
The geometry of the following beamlines is implemented:

 * ID01 (ESRF)
 * P10 (PETRA III)
 * CRISTAL (SOLEIL)
 * SIXS (SOLEIL)
 * NANOMAX (MAX IV)
 * 34ID-C (APS): only for postprocessing

The following detectors are implemented:

 * Maxipix
 * Timepix
 * Merlin
 * Eiger2M
 * Eiger4M

.. bcdi.experiment end

BCDI.preprocessing: preprocessing utilities on the diffraction data before phasing
==================================================================================

.. bcdi.preprocessing section

Description
-----------

This module provides methods used for pre-processing phased data. For example (but not limited to):
hotpixels removal, filtering, masking...

.. bcdi.preprocessing end

BCDI.postprocessing: postprocessing utilities on the complex object after phasing
=================================================================================

.. bcdi.postprocessing section

Description
-----------

This module provides methods used for post-processing phased data. For example (but not limited to):
phase offset and ramp removal, centering, cropping, padding, aligning reconstructions, filtering...

.. bcdi.postprocessing end

BCDI.publication: utilities to make formatted figure for publication
====================================================================

.. bcdi.publication section

Description
-----------

This module provides scripts with templates for figures that can be used in presentations.

.. bcdi.publication end

BCDI.simulation: simulation of diffraction patterns
===================================================

.. bcdi.simulation section

Description
-----------

In Bragg geometry, calculation of the diffraction intensity based on FFT or kinematical sum. It can include
a displacement field, noise, detector gaps etc...
In forward CDI geometry, calculation of the Bragg peak positions in 3D for a mesocrystal, knowing the unit cell
and unit cell parameter. It can be used to fit experimental data.

.. bcdi.simulation end

BCDI.graph: plotting utilities
==============================

.. bcdi.graph section

Description
-----------

This module provides methods to plot 2D and 3D data using templates, and to save it as a .vti file.

.. bcdi.graph end

BCDI.utils: various utilities for data analysis
===============================================

.. bcdi.utils section

Description
-----------

Various non-specific utilities (i/o, ...)

.. bcdi.utils end

BCDI.xcca: X-ray cross-correlation analysis
===========================================

.. bcdi.xcca section

Description
-----------

This module provides methods to calculate the angular cross-correlation function for a 3D reciprocal space dataset.

.. bcdi.xcca end
