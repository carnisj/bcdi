.. image:: https://readthedocs.org/projects/bcdi/badge/?version=latest
   :target: https://bcdi.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
.. image:: https://img.shields.io/github/workflow/status/carnisj/bcdi/python-package-3.9?logo=GitHub
   :alt: GitHub Workflow Status
.. image:: https://deepsource.io/gh/carnisj/bcdi.svg/?label=active+issues&show_trend=true&token=N3Z0cklmQrG8kzZOVwGJhLd9
.. image:: https://img.shields.io/pypi/pyversions/bcdi?logo=PyPI&logoColor=%23FFFF00
   :alt: PyPI - Python Version
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3257616.svg
   :target: https://doi.org/10.5281/zenodo.3257616

BCDI: tools for pre(post)-processing Bragg and forward coherent X-ray diffraction imaging data
==============================================================================================

Introduction
============

BCDI stands for *Bragg coherent X-ray diffraction imaging*. It can be used for:

* pre-processing BCDI and forward CDI data (masking aliens, detector gaps ...) before
  phase retrieval

* post-processing phased data (phase offset and phase ramp removal, averaging,
  apodization, ...)

* data analysis on diffraction data (stereographic projection, angular
  cross-correlation analysis, domain orientation fitting ...)

* data analysis on phased data (resolution calculation, statistics on the retrieved
  strain ...)

* simulation of diffraction intensity (including noise, detector gaps, displacement
  field ...)

* creating figures for publication using templates

Considering that most parts of the analysis pipeline are actually beamline-independent,
we tried to reuse as much as possible code, and leverage inheritance when it comes to
facility or beamline-dependent details.

BCDI as a python toolkit
========================

BCDI can be used as a python library with the following main modules:

1) :mod:`bcdi.algorithms`: PSF and image deconvolution using Richardson-Lucy algorithm

2) :mod:`bcdi.facet_recognition`: Stereographic projection of a diffraction peak or a
   reconstructed crystal. Automatic detection of reconstructed facets and statistics on
   facet strain.

3) :mod:`bcdi.experiment`: definition of the experimental geometry
   (beamline, setup, detector, diffractometer...).

4) :mod:`bcdi.graph` : generation of plots using predefined templates.

5) :mod:`bcdi.postprocessing`: methods for post-processing the complex output
   of a phasing algorithm.

6) :mod:`bcdi.preprocessing`: methods for pre-processing the diffraction
   intensity in Bragg CDI or forward CDI geometry.

7) :mod:`bcdi.simulation`: in BCDI geometry, calculation of the diffraction intensity
   based on FFT or kinematical sum. It can include a displacement field, noise,
   detector gaps etc... In forward CDI geometry, calculation of the Bragg peak
   positions in 3D for a mesocrystal, knowing the unit cell and unit cell parameter.

8) :mod:`bcdi.utils`: data loading, fitting functions, validation functions ...

9) :mod:`bcdi.xcca`: X-ray cross-correlation analysis related methods

The central module is :mod:`bcdi.experiment`, which contains all setup-related
implementation. This is the place where to look at if you want to add support for a new
beamline or detector.

Acknowledgment and third party packages
=======================================

We would like to acknowledge the following packages:

* xrayutilities: (c) Dominik Kriegner, Eugen Wintersberger.
  See: J. Appl. Cryst. 46, 1162-1170 (2013).

* nxsReady: (c) Andrea Resta @ SOLEIL SIXS

* image_registration.py: original code from Xianhui Xiao @ APS Sector 2.
  See: Opt. Lett. 33, 156-158 (2008).

* Some functions were adapted from PyNX: (c) Vincent Favre-Nicolin.
  See: http://ftp.esrf.fr/pub/scisoft/PyNX/ and J. Appl. Cryst. 49, 1842-1848 (2016).

The following third-party packages are required:

* numpy, scipy, scikit-image, matplotlib, pyqt5, vtk, h5py, hdf5plugin, fabio,
  silx, xrayutilities

* lmfit: for scripts performing fits

* pytest: to run the tests

* pytables: to load the devices dictionnary for SIXS data

* moviepy, `imagemagick <https://imagemagick.org>`_ or
  `ffmpeg <http://ffmpeg.zeranoe.com/builds/>`_ for creating movies

Download & Installation
=======================

BCDI is available from:

 * Python Package Index: ``pip install bcdi``
 * `Most updated version on GitHub <https://github.com/carnisj/>`_
 * upgrade your version with the latest changes from GitHub:
   ``pip install --upgrade git+https://github.com/carnisj/bcdi.git``

Not that there are issues with installing scikit-image within an Anaconda environment.
In such situation, the workaround is to create instead a virtual environment using pip.

Please send feedback in `GitHub <https://github.com/carnisj/>`_.

Citation & Bibliography
=======================

If you use BCDI for scientific work, please consider including a citation
(DOI: 10.5281/zenodo.3257616).

License
=======
The BCDI library is distributed with a CeCILL-B license
(an open-source license similar to the FreeBSD one).
See http://cecill.info/licences/Licence_CeCILL-B_V1-en.html

Documentation
=============

The documentation is available at: https://bcdi.readthedocs.io/en/latest/
