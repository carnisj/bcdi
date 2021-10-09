.. py:module:: bcdi.experiment

.. toctree::
   :maxdepth: 2

:mod:`bcdi.experiment`: Description of the experimental setup
=============================================================

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

The following detectors are implemented:

 * Maxipix
 * Timepix
 * Merlin
 * Eiger2M
 * Eiger4M
 * Dummy (user-defined pixel size and pixel number)

beamline
^^^^^^^^

.. mermaid::
  :align: center

  classDiagram
    class Beamline{
      +str name
  }
    ABC <|-- Beamline
    Beamline <|-- BeamlineID01
    Beamline <|-- BeamlineSIXS
    Beamline <|-- Beamline34ID
    Beamline <|-- BeamlineP10
    BeamlineP10 <|-- BeamlineP10SAXS
    Beamline <|-- BeamlineCRISTAL
    Beamline <|-- BeamlineNANOMAX


API Reference
-------------

.. automodule:: bcdi.experiment.beamline
   :members:

detector
^^^^^^^^

.. mermaid::
  :align: center

  classDiagram
    class Detector{
      +str name : detector_name
  }
    ABC <|-- Detector
    Detector <|-- Maxipix
    Detector <|-- Eiger2M
    Detector <|-- Eiger4M
    Detector <|-- Timepix
    Detector <|-- Merlin
    Detector <|-- Dummy

API Reference
-------------

.. automodule:: bcdi.experiment.detector
    :members:

diffractometer
^^^^^^^^^^^^^^

.. mermaid::
  :align: center

  classDiagram
    class Diffractometer{
      +tuple sample_offsets
  }
    ABC <|-- Diffractometer
    Diffractometer <|-- DiffractometerID01
    Diffractometer <|-- DiffractometerSIXS
    Diffractometer <|-- Diffractometer34ID
    Diffractometer <|-- DiffractometerP10
    DiffractometerP10 <|-- DiffractometerP10SAXS
    Diffractometer <|-- DiffractometerCRISTAL
    Diffractometer <|-- DiffractometerNANOMAX

API Reference
-------------

.. automodule:: bcdi.experiment.diffractometer
    :members:

rotation_matrix
^^^^^^^^^^^^^^^

This class is used to define 3D rotation matrices.

API Reference
-------------

.. automodule:: bcdi.experiment.rotation_matrix
    :members:

setup
^^^^^

This class is the "manager" or public interface of the analysis workflow. Access to
the instances of the child classes inheriting from Beamline, Diffractometer and Detector
is realized through Setup.

API Reference
-------------

.. automodule:: bcdi.experiment.setup
   :members:
