.. BCDI documentation main file, created by
   sphinx-quickstart on Wed Apr 24 10:57:29 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. |bcdi icon| image:: ../diffract.png
  :width: 50
  :alt: Nice diffraction pattern

|bcdi icon| Welcome to BCDI's documentation!
============================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:
    
.. include:: ../README.rst

Examples
========

In this section you will find two examples showing how to pre-process and post-process
experimental BCDI using this package. In order to run the example, you need first to
get the data, which is available on the Coherent X-ray Imaging DataBank. Please
download the dataset ID182 from the following location:
https://www.cxidb.org/id-182.html

Extract files, you should get the following::

    CXIDB-182/
        Readme ID 182.txt
        CH4760/
            l5.spec
            S11/
            ...
        HS4670/
            ...


For the demonstration, we will use the scan ``S11`` from the experiment ``CH4760``.
The spec file for this scan is ``l5.spec``.

.. include:: EXAMPLE_PREPROCESSING.rst

.. include:: EXAMPLE_POSTPROCESSING.rst

Changelog
=========

.. include:: HISTORY.rst
  :end-before: Version 0.1.5

See the full  :doc:`Changelog<HISTORY>`

Command-line scripts
====================

Some sample scripts are provided with the package. However, the authors make no warranty
as to the correctness or accuracy of the analysis provided.

Documentation of the scripts included in BCDI.

.. toctree::
   :maxdepth: 1

   scripts/index.rst

API Documentation
=================

Documentation of the modules included in BCDI.

.. toctree::
   :maxdepth: 1

   modules/index.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
