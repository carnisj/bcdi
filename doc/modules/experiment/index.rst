.. py:module:: bcdi.experiment

.. toctree::
   :maxdepth: 2

:mod:`bcdi.experiment`: Description of the experimental setup
=============================================================

This module provides classes and methods for the definition of the experimental setup.
The following classes are implemented:

 * Beamline and corresponding child classes: calculations related to reciprocal or
   direct space transformation.
 * Detector and corresponding child classes: implementation of the 2D detectors.
 * Diffractometer and Geometry with corresponding child classes: implementation of the
   diffractometer geometry.
 * Loader and corresponding child classes: initialization of the file system and loading
   of data and motor positions.
 * RotationMatrix: used in methods from Diffractometer to generate rotation matrices
 * Setup: the manager of the analysis

.. mermaid::
  :align: center

  classDiagram
    class Setup{
      +name : beamline name
      +detector_name
  }
    class Beamline{
      <<abstract>>
      +name
  }
    class Loader{
      <<abstract>>
      +name
      +sample_offsets
  }
    class Diffractometer{
      +name
      +sample_offsets
  }
    class Geometry{
      +name
  }
    class Detector{
      <<abstract>>
      +name
  }
    class RotationMatrix{
      +angle
      +circle
  }
    Setup *-- Beamline : create_beamline()
    Setup *-- Detector : create_detector()
    Beamline *-- Diffractometer
    Diffractometer *-- Geometry : create_geometry()
    Beamline *-- Loader : create_loader()
    Diffractometer *-- RotationMatrix

In scripts, the initial step is to declare a setup instance with the related parameters
(see the class documentation). The beamline and the detector are instantiated in the
Setup instance. The Loader and Diffractometer are instantiated in the Beamline instance.
However, you are free to instantiate these classes outside of a Setup instance if
needed.

The geometry of the following beamlines is implemented:

 * ID01 (ESRF)
 * P10 (PETRA III): 6-circle and USAXS setups
 * CRISTAL (SOLEIL)
 * SIXS (SOLEIL)
 * NANOMAX (MAX IV)
 * 34ID-C (APS)

The following detectors are implemented:

 * Maxipix
 * Timepix
 * Merlin
 * Eiger2M
 * Eiger4M
 * Dummy (user-defined pixel size and pixel number)

beamline
^^^^^^^^

General organization of the module:

.. mermaid::
  :align: center

  classDiagram
    class Beamline{
      <<abstract>>
      +name : beamline name
  }
    ABC <|-- Beamline
    Beamline <|-- BeamlineID01
    Beamline <|-- BeamlineSIXS
    Beamline <|-- Beamline34ID
    Beamline <|-- BeamlineP10
    BeamlineP10 <|-- BeamlineP10SAXS
    Beamline <|-- BeamlineCRISTAL
    Beamline <|-- BeamlineNANOMAX

.. automodule:: bcdi.experiment.beamline
   :members:

beamline_factory
^^^^^^^^^^^^^^^^

This module provides beamline abstract classes.

API Reference
-------------

.. automodule:: bcdi.experiment.beamline_factory
   :members:

detector
^^^^^^^^

General organization of the module:

.. mermaid::
  :align: center

  classDiagram
    class Detector{
      <<abstract>>
      +name : detector_name
  }
    ABC <|-- Detector
    Detector <|-- Maxipix
    Detector <|-- Eiger2M
    Detector <|-- Eiger4M
    Detector <|-- Timepix
    Detector <|-- Merlin
    Detector <|-- Dummy

.. automodule:: bcdi.experiment.detector
    :members:

diffractometer
^^^^^^^^^^^^^^

.. automodule:: bcdi.experiment.diffractometer
    :members:

loader
^^^^^^

General organization of the module:

.. mermaid::
  :align: center

  classDiagram
    class Loader{
      <<abstract>>
      +name
      +sample_offsets
  }
    ABC <|-- Loader
    Loader <|-- LoaderID01
    Loader <|-- LoaderID01BLISS
    Loader <|-- LoaderSIXS
    Loader <|-- Loader34ID
    Loader <|-- LoaderP10
    LoaderP10 <|-- LoaderP10SAXS
    Loader <|-- LoaderCRISTAL
    Loader <|-- LoaderNANOMAX

.. automodule:: bcdi.experiment.loader
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
