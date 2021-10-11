.. py:module:: bcdi.simulation

.. toctree::
   :maxdepth: 2

:mod:`bcdi.simulation`: calculation of the diffraction intensity based on FFT or kinematical sum
================================================================================================

simulation_utils
^^^^^^^^^^^^^^^^

In Bragg geometry, calculation of the diffraction intensity based on FFT or
kinematical sum. It can include a displacement field, noise, detector gaps etc...
See Carnis et al. Scientific Reports 9, 17357 (2019)
https://doi.org/10.1038/s41598-019-53774-2

In forward CDI geometry, calculation of the Bragg peak positions in 3D for a
mesocrystal, knowing the unit cell and unit cell parameter. It can be used to fit
experimental data.

API Reference
-------------

.. automodule:: bcdi.simulation.simulation_utils
   :members:

supportMaker
^^^^^^^^^^^^

This module provides tools to create a simple support using polygons. It is possible
to define distances in pixels or nanometers.

API Reference
-------------

.. automodule:: bcdi.simulation.supportMaker
   :members:
