.. py:module:: bcdi.preprocessing

.. toctree::
   :maxdepth: 2

:mod:`bcdi.preprocessing`: Preprocessing of (B)CDI data before phase retrieval
==============================================================================

preprocessing_runner
^^^^^^^^^^^^^^^^^^^^

This module provides the function which manage the whole preprocessing.

API Reference
-------------

.. automodule:: bcdi.preprocessing.preprocessing_runner
   :members:

bcdi_utils
^^^^^^^^^^

This module provides generic methods used for pre-processing data before phase
retrieval. For example (but not limited to): centering, hotpixels removal, filtering,
masking... Functions are optimized for a Bragg CDI dataset.

API Reference
-------------

.. automodule:: bcdi.preprocessing.bcdi_utils
   :members:

cdi_utils
^^^^^^^^^

This module provides generic methods used for pre-processing data before phase
retrieval. For example (but not limited to): centering, hotpixels removal, filtering,
masking... Functions are optimized for a forward CDI dataset.

API Reference
-------------

.. automodule:: bcdi.preprocessing.cdi_utils
   :members:
