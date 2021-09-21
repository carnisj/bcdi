.. py:module:: bcdi.experiment

.. toctree::
   :maxdepth: 2

:mod:`bcdi.experiment`: Description of the experimental setup
=============================================================

Description
-----------

This module provides classes and methods for the definition of the experimental setup.
The following classes are implemented:

 * Beamline and corresponding child classes (one per supported beamline)
 * Detector and corresponding child classes (one per supported detector)
 * Diffractometer and corresponding child classes (one per supported beamline)
 * RotationMatrix: used in methods from Diffractometer to generate rotation matrices
 * Setup

.. mermaid::
  :align: center

  classDiagram
    class Setup{
      +str beamline
  }
    class Beamline{
      +str name
  }
    class Diffractometer{
      +tuple sample_offsets
  }
    class Detector{
      +str name
  }
    class RotationMatrix{
      +str circle
      +float angle
  }
    Setup *-- Beamline : create_beamline()
    Setup *-- Diffractometer : create_diffractometer()
    Setup o-- Detector : create_detector()
    Diffractometer o-- RotationMatrix

In scripts, the initial step is to declare a detector instance and a setup instance with
the related parameters (see the class documentation). The beamline and the
diffractometer are not meant to be instantiated directly, this is done internally in
Setup.

The geometry of the following beamlines is implemented:

 * ID01 (ESRF)
 * P10 (PETRA III): 6-circle and USAXS setups
 * CRISTAL (SOLEIL)
 * SIXS (SOLEIL)
 * NANOMAX (MAX IV)
 * 34ID-C (APS): only for postprocessing

.. autoclasstree:: bcdi.experiment.beamline

.. autoclasstree:: bcdi.experiment.diffractometer

The following detectors are implemented:

 * Maxipix
 * Timepix
 * Merlin
 * Eiger2M
 * Eiger4M
 * Dummy (user-defined pixel size and pixel number)

.. autoclasstree:: bcdi.experiment.detector

API Reference
-------------

beamline
^^^^^^^^

.. automodule:: bcdi.experiment.beamline
   :members:

detector
^^^^^^^^

.. automodule:: bcdi.experiment.detector
    :members:

diffractometer
^^^^^^^^^^^^^^

.. automodule:: bcdi.experiment.diffractometer
    :members:

rotation_matrix
^^^^^^^^^^^^^^^

.. automodule:: bcdi.experiment.rotation_matrix
    :members:

setup
^^^^^

.. automodule:: bcdi.experiment.setup
   :members:
