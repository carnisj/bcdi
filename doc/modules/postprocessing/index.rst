.. py:module:: bcdi.postprocessing

.. toctree::
   :maxdepth: 2

:mod:`bcdi.postprocessing`: Post-processing of data after phase retrieval
=========================================================================

facet_recognition
^^^^^^^^^^^^^^^^^

This module provides tools for facet segmentation on nanocrystals, and for plotting the
stereographic projection of a diffraction peak or an object. For more details, see:
Carnis et al. Small 17, 2007702 (2021) https://doi.org/10.1002/smll.202007702

API Reference
-------------

.. automodule:: bcdi.postprocessing.facet_recognition
   :members:

postprocessing_utils
^^^^^^^^^^^^^^^^^^^^

This module provides methods used for post-processing data after phase retrieval.
For example (but not limited to): phase offset and ramp removal, centering, 
cropping, padding, aligning reconstructions, filtering...

API Reference
-------------

.. automodule:: bcdi.postprocessing.postprocessing_utils
   :members:
