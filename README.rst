.. image:: https://readthedocs.org/projects/bcdi/badge/?version=latest
   :target: https://bcdi.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
.. image:: https://img.shields.io/github/actions/workflow/status/carnisj/bcdi/pr_lint_test.yml?logo=GitHub
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

* post-processing after phase retrieval (phase offset and phase ramp removal, averaging,
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

2) :mod:`bcdi.experiment`: definition of the experimental geometry
   (beamline, setup, detector, diffractometer...).

3) :mod:`bcdi.graph` : generation of plots using predefined templates.

4) :mod:`bcdi.postprocessing`: methods for post-processing the complex output
   of a phasing algorithm. Stereographic projection of a diffraction peak or a
   reconstructed crystal. Automatic detection of reconstructed facets and statistics on
   facet strain.

5) :mod:`bcdi.preprocessing`: methods for pre-processing the diffraction
   intensity in Bragg CDI or forward CDI geometry.

6) :mod:`bcdi.simulation`: in BCDI geometry, calculation of the diffraction intensity
   based on FFT or kinematical sum. It can include a displacement field, noise,
   detector gaps etc... In forward CDI geometry, calculation of the Bragg peak
   positions in 3D for a mesocrystal, knowing the unit cell and unit cell parameter.

7) :mod:`bcdi.utils`: generic functions about data loading, fitting functions, cropping/
   padding, image registration, validation functions ...

8) :mod:`bcdi.xcca`: X-ray cross-correlation analysis related methods

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

* Anton Mikhailov for the turbo colormap.
  See https://ai.googleblog.com/2019/08/turbo-improved-rainbow-colormap-for.html

The following third-party packages are required:

* numpy, scipy, scikit-image, matplotlib, pyqt5, vtk, h5py, hdf5plugin, fabio,
  silx, xrayutilities

* lmfit: for scripts performing fits

* pytest: to run the tests

* pytables: to load the devices dictionnary for SIXS data

* moviepy, `imagemagick <https://imagemagick.org>`_ or
  `ffmpeg <https://ffmpeg.org/download.html>`_ for creating movies

Download & Installation
=======================

BCDI is available from:

* the Python Package Index: ``python -m pip install bcdi``
* or on `GitHub <https://github.com/carnisj/bcdi>`_, where you will find the latest
  version:

|  - to install the main branch, type:
|    ``python -m pip install git+https://github.com/carnisj/bcdi.git``
|  - to install a specific branch, type:
|    ``python -m pip install git+https://github.com/carnisj/bcdi.git@branch_name``

Add the flag ``--upgrade`` to the commands above in order to update an existing
installation.

Note that there are issues with installing scikit-image within an Anaconda environment.
In such situation, the workaround is to create instead a virtual environment using pip.

If you want to contribute to bcdi development, install also extra dependencies:
``python -m pip install bcdi[dev]``

Please send feedback in `GitHub issues <https://github.com/carnisj/bcdi/issues>`_.

Documentation
=============

The documentation is available at: https://bcdi.readthedocs.io/en/latest/

Video Documentation
===================

All talks from the bcdiHackweek 2021 are available at the following links:

* Carnis, J. - BCDI package overview: https://youtu.be/g4jkzmz8JGk

* Li, N.  - data preprocessing: https://youtu.be/D-fl19Mi7Ao

* Carnis, J. - data preprocessing interactive: https://youtu.be/ddipN43HR1w

* Dupraz, M. - data postprocessing + viz: https://youtu.be/WyDzOkJJu8c

* Carnis, J. - all that is left in the BCDI package: https://youtu.be/egh8X6iI4Nw

* Richard, M.-I. - paraview Facet analyser: https://youtu.be/RarHeUIOu08

* Carnis, J. - bcdi Facet analyser: https://youtu.be/gucQk8p3vyk

* Simonne, D. - jupyter GUI for bcdi package: https://youtu.be/9SDcGfJqiVw

Related package `Cohere <https://github.com/AdvancedPhotonSource/cohere>`_:

* Harder, R. - Cohere package overview https://youtu.be/I1YOZoxddlE

License
=======

The BCDI library is distributed with a CeCILL-B license
(an open-source license similar to the FreeBSD one).
See http://cecill.info/licences/Licence_CeCILL-B_V1-en.html

Citation & Bibliography
=======================

If you use this package for scientific work, please consider including a citation using
the following DOI: 10.5281/zenodo.3257616

This package contributed to the following peer-reviewed publications:

* Y. Y. Kim, et al. Single Alloy Nanoparticle X-Ray Imaging during a Catalytic Reaction.
  Science Advances 7 (2021). DOI: 10.1126/sciadv.abh0757

* J. Carnis, et al. Facet-dependent strain determination in electrochemically
  synthetized platinum model catalytic nanoparticles. Small 2007702 (2021).
  DOI: 10.1002/smll.202007702
  Data available at CXIDB ID182: https://www.cxidb.org/id-182.html

* J. Carnis, et al. Twinning/detwinning in an individual platinum nanocrystal during
  catalytic CO oxidation. Nature Communications 12 (1), 1-10 (2021).
  DOI: 10.1038/s41467-021-25625-0

* J. Carnis, et al. Structural Study of a Self-Assembled Gold Mesocrystal Grain by
  Coherent X-ray Diffraction Imaging. Nanoscale 13, 10425-10435 (2021).
  DOI: 10.1039/D1NR01806J
  Data available at CXIDB ID183:  https://www.cxidb.org/id-183.html

* N. Li, et al. Mapping Inversion Domain Boundaries Along Single GaN Wires with Bragg
  Coherent X-ray Imaging. ACS Nano 14, 10305–10312 (2020). DOI: 10.1021/acsnano.0c03775

* N. Li, et al. Continu-ous scanning for Bragg coherent X ray imaging.
  Sci. Rep. 10, 12760 (2020). DOI: 10.1038/s41598-020-69678-5

* J. Carnis, et al. Towards a quantitative determination of strain in Bragg Coherent
  X-ray Diffraction Imaging: artefacts and sign convention in reconstructions.
  Scientific reports 9, 17357 (2019). DOI: 10.1038/s41598-019-53774-2

* W. Hua, et al. Structural insights into the formation and voltage degradation of
  lithium- and manganese-rich layered oxides. Nat Commun 10, 5365 (2019).
  DOI: 10.1038/s41467-019-13240-z

* G. Niu, et al. Advanced coherent X-ray diffraction and electron microscopy of
  individual InP nanocrystals on Si nanotips for III-V on Si electronics and
  optoelectronics. Phys. Rev. Applied 11, 064046 (2019).
  DOI: 10.1103/PhysRevApplied.11.064046

* S. Fernández, et al. In situ structural evolution of single particle model catalysts
  under ambient pressure reaction conditions. Nanoscale 11, 331-338 (2019).
  DOI: 10.1039/c8nr08414a
