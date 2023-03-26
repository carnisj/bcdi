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

analysis
^^^^^^^^

This module provides the classes used in postprocessing strain analysis. It is mainly
an abstration layer making analysis steps more understandable.

API Reference
-------------

.. automodule:: bcdi.postprocessing.analysis
   :members:

facet_analysis
^^^^^^^^^^^^^^

This module provides tools to import and stores the output of the facet analyzer plugin
from Paraview for further analysis. See Ultramicroscopy 122, 65-75 (2012)
https://doi.org/10.1016/j.ultramic.2012.07.024

One can extract the strain component and the displacement on the facets, and retrieve
the correct facet normals based on a user input (geometric transformation into the
crystal frame). Only cubic lattices are supported.

API Reference
-------------

.. automodule:: bcdi.postprocessing.facet_analysis
   :members:

postprocessing_runner
^^^^^^^^^^^^^^^^^^^^^

This module provides the function which manage the whole postprocessing.

API Reference
-------------

.. automodule:: bcdi.postprocessing.postprocessing_runner
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

process_scan
^^^^^^^^^^^^

This module provides the function which manage the postprocessing for a single scan.

API Reference
-------------

.. automodule:: bcdi.postprocessing.process_scan
   :members:

raw_orthogonalization
^^^^^^^^^^^^^^^^^^^^^

This module provides the function for BCDI data orthogonalization of a single scan,
after phase retrieval (skipping centering, phase offset and ramp removal).

API Reference
-------------

.. automodule:: bcdi.postprocessing.raw_orthogonalization
   :members:
