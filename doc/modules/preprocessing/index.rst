.. py:module:: bcdi.preprocessing

.. toctree::
   :maxdepth: 2

:mod:`bcdi.preprocessing`: Preprocessing of (B)CDI data before phase retrieval
==============================================================================

preprocessing_runner
^^^^^^^^^^^^^^^^^^^^

This module provides the function which manage the preprocessing.

API Reference
-------------

.. automodule:: bcdi.preprocessing.preprocessing_runner
   :members:

analysis
^^^^^^^^

This module provides the classes used in data preprocessing. It is mainly an abstration
layer making analysis steps more understandable.

API Reference
-------------

.. automodule:: bcdi.preprocessing.analysis
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

center_fft
^^^^^^^^^^

This module provides classes related to centering the diffraction pattern in
preprocessing.

API Reference
-------------

.. automodule:: bcdi.preprocessing.center_fft
   :members:

process_scan
^^^^^^^^^^^^

This module provides the function which manage the preprocessing for a single scan.

API Reference
-------------

.. automodule:: bcdi.preprocessing.process_scan
   :members:

process_scan_cdi
^^^^^^^^^^^^^^^^

This module provides the function which manage the preprocessing for a single scan when
the detector is not on a goniometer (detector plane always perpendicular to the direct
beam independently of its position).

API Reference
-------------

.. automodule:: bcdi.preprocessing.process_scan_cdi
   :members:
