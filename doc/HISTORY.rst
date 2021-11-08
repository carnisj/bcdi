Future:
-------

* Support APS 34ID-C for data preprocessing (loading of TIFF images)

* Update the parameter "linearity_function" in preprocessing. Now this can be None (if
  unused) or a sequence of 5 real numbers corresponding to the coefficients of a 4th
  order polynomial.

* Implement a parser for YAML config files. Now the scripts ``bcdi_strain.py`` and
  ``bcdi_preprocess_BCDI.py`` can be run like scripts, from the command line, with
  optional command line arguments.

Version 0.1.7
-------------

* Bug: apply correctly the non-linearity correction function to the detector frames
  (typo in the function name)

* Modify the behavior of the parameter `specfile_name`: for beamlines relying on a
  separate file for logging motor positions (specfiles at ID01 and 34ID, fio file at
  P10), the user can provide the full path to the file

* Implement loading motor positions from a specfile at 34ID.

* Add mypy for type checking in doit and GitHub workflows

* Bug: correct the detector horizontal direction in Beamline34ID, it was flipped.

* Rename the parameter data_dirname to data_dir for the function setup.init_paths.
  Now the user can provide directly the path to the data directory.

* Move all functions related to dataset alignment in the module
  ``utils.image_registration.py`` and create generic functions

* Enable preprocessing d2scan using xrayutilities for ID01. The parameter `follow_bragg`
  become obsolete and is removed.

* Add the module ``simulation.supportMaker.py``, which allows to create a support using
  polygons. Lengths can be defined either in pixels or in nanometers.

* Merge the subpackage facet_recognition into postprocessing and rename the module
  ``facet_utils.py`` to ``facet_recognition.py``

* Add the list of publications related to the package in the documentation

* Add class diagrams in the documentation using sphinxcontrib-mermaid

* Solve issues with numpy when building the documentation (enable *Use system packages*
  in ReadTheDocs advanced settings)

Version 0.1.6
-------------

* Write unit tests for ``experiment.detector.py``, now coverage is > 99% for this
  module

* move forward CDI gridding function to ``Setup``

* implement ``DiffractometerP10SAXS`` and ``BeamlineP10SAXS`` classes for forward CDI
  experiments at P10

* split the module ``preprocessing_utils`` in two modules, ``bcdi_utils`` and
  ``cdi_utils``

* Move generic functions from ``preprocessing_utils`` to ``utilities``

* Create new validations functions ``valid_ndarray`` and ``valid_1d_array``, implement
  the corresponding unit tests.

* Refactor: rename ``preprocessing_utils.regrid`` to ``calc_qvalues_xrutils`` and move
  it to ``Setup``. Put all the beamline dependent code in the corresponding ``Beamline``
  child class.

Version 0.1.5
-------------

* Bug: convert arrays to a tuple before checking the length in Setup.ortho_directspace

Version 0.1.4
-------------

* Feature: implement a new validation function valid_ndarray, implement tests and remove
  the redundant code in modules.

* Refactor: split the Detector class using inheritance, refactor scripts accordingly and
  implement tests.

* Feature: create a Beamline class with one child class for each beamline, move
  all beamline-dependent methods from Setup to the respective class and implement some
  tests.

Version 0.1.3
-------------

* Refactor: allow the user to not provide a mask in the BCDI PRTF calculations (3D and
  2D)

* Refractor: split bcdi.experiment.experiment_utils module into smaller modules

* Refactor: enforce project's guidelines for the code style and the docstrings

* Create a dodo.py file (doit package) to simplify the life of contributors: now they
  just need to run doit at the same level as setup.py and verify that all checks pass
  before sending their pull request

* Create a CONTRIBUTING.md file

Version 0.1.2
-------------

* Refactor: remove circular imports from modules

* Refactor: ``move crop_pad``, ``bin_data`` and ``gaussian_window functions`` from
  ``postprocessing_utils.py`` to another module in order to avoid circular imports

* Feature: create a Diffractometer class with one child class for each beamline, move
  all functions related to the goniometer positions in the class

* Feature: add an option in ``strain.py`` to put back the sample in the laboratory
  frame with all sample circles rotated back to 0 deg

* Refactor: show only necessary plots and console output in ``strain.py``

* Refactor: create Setup calculated properties and transfer calculations in scripts to
  these properties

* Refactor: perform the geometrical transformation and rotation of the reconstructed
  crystal in a single step

* Refactor: perform the geometrical transformation and rotation of the diffraction
  pattern in a single step

* Bug: provide voxel sizes in the correct order when rotating the diffraction pattern
  in ``preprocess_bcdi.py``

Version 0.1.1
-------------

* code cleaning

Version 0.1.0
-------------

* Feature: implement ``publication/bcdi_diffpattern_from_reconstruction.py``, to
  compare with the experimental measurement in the crystal frame

* Refactor: simplify PRTF calculations

* Feature: implement the inplane rocking curve at CRISTAL

* Feature: implement ``graph_utils.savefig`` to save figures for publication with and
  without labels

* Feature: implement ``angular_profile.py`` to calculate the width of linecuts through
  the center of mass of a 2D object at different angles

* Feature: implement ``line_profile.py`` to calculate line profiles along particular
  directions in 2D or 3D objects

Version 0.0.10a2
----------------

* Feature: implement ``interpolate_cdi.py``, to interpolate the intensity of masked
  voxels using the centrosymmetry property

* Feature: implement the interpolation of the reciprocal space data in the laboratory
  frame using the linearized transformation matrix

* Refactor: update the calculation of the transformation matrices when chi is non-zero

* Feature: allow different voxel sizes in each dimension in ``strain.py``
  (NOT BACK COMPATIBLE)

* Feature: implement validation functions in ``utils.validation.py`` for commonly used
  parameters, implement related unit tests

* Refactor: merge the class SetupPostprocessing and SetupPreprocessing in a single
  class Setup due to code redundances

* Feature: implement ``linecut_diffpattern.py``, a GUI to get a linecut of a 3D
  diffraction pattern along a desired direction

* Feature: add a GUI to ``prtf_bcdi.py``, to get a linecut of the 3D PRTF along a
  desired direction

* Feature: implement ``center_of_rotation.py``, to calculate the distance of the
  crystal to the center of rotation

* Bug: in ``facet_strain.py``, solve bugs in plane fitting when the facet is parallel
  to an axis

* Feature: implement ``rotate_scan.py``, to rotate a 3D reciprocal space map around a
  vector

* Refactor: in ``modes_decomposition.py``, implement skipping alignment between datasets
  or aligning based on a support

Version 0.0.9
-------------

* Feature: implement support for MAXIV NANOMAX beamline

* Feature: implement ``rocking_curves.py`` to follow the evolution of the Bragg peak
  between several rocking curves

* Feature: implement ``flatten_modulus.py`` to remove low frequency artefacts in the
  modulus reconstructed by phase retrieval

* Feature: implement ``xcca_3D_map.py`` to calculate the angular cross-correlation
  CCF(q,q) over a range in q

* Feature: implement ``view_ccf.py`` and ``view_ccf_map.py`` to plot the
  cross-correlation function output

* Feature: implement the 3D angular X-ray cross-correlation analysis

* Refactor: allow the reloading of binned data and its orthogonalization in
  ``preprocess_cdi.py`` and ``preprocess_bcdi.py``

* Feature: implement ``crop_npz.py`` to crop combined stacked data to the desired size

* Feature: implement ``scan_analysis.py`` to plot interactively the integrated
  intensity in a region of interest for a 1D scan

* Feature: implement ``view_mesh.py`` to plot interactively the integrated intensity
  in a region of interest for a 2D mesh

* Refactor: when gridding forward CDI data, reverse the rotation direction to
  compensate the rotation of Ewald sphere

* Refactor: updated ``extract_bulk_surface.py`` to use module functions

* Bug: treat correctly the case angle=pi/2 during the interpolation of CDI data onto
  the laboratory frame

* Refactor: solve instabilities resulting from duplicate vertices after smoothing in
  ``facet_strain.py``

* Refactor: modify ``polarplot.py`` to use module functions instead of inline script

* Feature: implement ``coefficient_variation.py`` to compare several reconstructed
  modulus of a BCDI dataset

* Feature: implement diffraction_angles.py`` to find Bragg reflections for a particular
  goniometer setup, based on xrayutilities

* Feature: add the option of restarting masking the aliens during preprocessing,
  not back compatible with previous versions

* Feature: implement simultaneous masking over the 3 axes in two new preprocessing
  scripts ``preprocess_bcdi.py`` and ``preprocess_cdi.py``

* Feature: implement ``domain_orientation.py`` to find the orientation of domains in a
  3D forward CDI dataset of mesocrystal

* Feature: implement ``simu_diffpattern_CDI.py`` to find in 3D the Bragg peaks positions
  of a mesocrystal (supported unit cells: FCC, simple cubic, BCC and BCT)

* Feature: implement ``fit_1D curve.py`` to fit simultaneously ranges of a 1D curve with
  gaussian lineshapes

* Feature: implement ``fit_background.py`` to interactively determine the background in
  1D reciprocal space curves

* Refactor: in ``multislices_plot()`` and ``contour_slices()``, allow to plot the data
  at user-defined slices positions.

* Feature: implement ``prtf_bcdi_2D.py`` to calculate the PRTF also for 2D cases.

Version 0.0.8
-------------

* Feature: implement ``3Dobject_movie.py``, creating movies of a real-space 3D object.

* Feature: implement ``modes_decomposition.py``, decomposition of a set of reconstructed
  object in orthogonal modes (adapted from PyNX)

* Bug: correct the calculation of q when data is binned

* implement scripts to visualize isosurfaces of reciprocal/real space including
  publication options (in /publication/)

* implement ``algorithms_utils.py``, featuring psf and image deconvolution using
  Richardson-Lucy algorithm

* implement separate PRTF resolution estimation for CDI (``prtf_cdi.py``) and BCDI
  (``bcdi_prtf.py``) datasets

* Feature: implement ``angular_average.py`` to average 3D CDI reciprocal space data in
  1D curve

* Feature: implement view_psf to plot the psf output of a phase retrieval with partial
  coherence

* Refactor: change name of ``make_support.py`` to ``rescale_support.py``

Version 0.0.7
-------------
* Feature: implement ``supportMaker()`` class to define a support from a set of planes

* Feature: implement ``maskMaker()`` class for easier implementation of new masking
  features

* Debug ``prepare_bcdi_mask.py`` for energy scans at ID01

* Feature: implement ``utils/scripts/make_support.py``, to rescale a support for phasing
  with a larger FFT window

* Feature/refactor: implement ``prepare_cdi_mask.py`` for forward CDI, rename existing
  as ``prepare_bcdi_mask.py`` for Bragg CDI

* Feature: add the possibility to change the detector distance in ``simu_noise.py``

* Feature: add the possibility to pre-process data acquired without scans, e.g. in a
  macro (no spec file)

* Feature: in ``strain.py``, implement phase unwrapping so that the phase range can be
  larger than 2*pi

* Feature: in ``facet_strain.py``, implement edge removal for more precise statistics
  on facet strain

* Feature: in ``facet_strain.py``, allow anisotropic voxel size and user-defined
  reference axis in the stereographic projection

Version 0.0.6
-------------

* Feature: implement facet detection using a stereographic projection in
  ``facet_recognition/scripts/facet_strain.py``

* Feature: Converted ``bcdi/facet_recognition/scripts/facet_strain.py``

* Feature: implement ``bcdi/facet_recognition/facets_utils.py``

* Refactor: exclude voxels left over by coordination number selection in
  ``postprocessing/postprocessing_utils.find_bulk()``

* Refactor: use the mean amplitude of the surface layer to define the bulk in
  ``postprocessing/postprocessing_utils.find_bulk()``

* Feature: enable PRTF resolution calculation for simulated data

* Feature: create ``preprocessing/scripts/apodize.py`` to apodize reciprocal space data

* Feature: implement 3d Tukey and 3d Blackman windows for apodization in
  ``postprocessing_utils()``

* Feature: in ``postprocessing/scripts/resolution_prtf.py``, allow for binning the
  detector plane

* Bug: in ``postprocessing/scripts/strain.py``, correct the original array size taking
  into account the binning factor

* Feature: implement ``postprocessing_utils.bin_data()``

Version 0.0.5
-------------

* Feature: implement support for SIXS data measured after the 11/03/2019 with the new
  data recorder.

* Refactor: ``modify preprocessing/scripts/readdata_P10.py`` to support several
  beamlines and rename it ``read_data.py``

* Feature: implement support for multiple beamlines in
  ``postprocessing/script/resolution_prtf.py``

* Refactor: merge all ``preprocessing/preprocessing_utils.regrid_*.py`` in
  ``preprocessing/preprocessing_utils.regrid()``

* Converted ``postprocessing/scripts/resolution_prtf.py``

* Refactor: add the possibility of giving a single element instead of the full tuple
  in ``graph/graph_utils.combined_plots()``

* Converted ``postprocessing/scripts/resolution_prtf.py``

* Feature: create a ``Colormap()`` class in ``graph/graph_utils.py``

* Refactor: merge all ``postprocessing/scripts/calc_angles_beam_*.py`` in
  ``postprocessing/scripts/correct_angles_detector.py``

* Feature: Implement ``motor_values()`` and ``load_data()`` in
  ``preprocessing/preprocessing_utils.py``

* Feature: Implement ``SetupPostprocessing.rotation_direction()`` in
  ``experiment/experiment_utils.py``

* Feature: add other counter name 'curpetra' for beam intensity monitor at P10

* Bug: ``postprocessing/scripts/calc_angles_beam_*.py``: correct bug when roi_detector
  is not defined, and round the Bragg peak COM to integer pixels

Version 0.0.4
-------------

* Implement ``motor_positions_p10()``, ``motor_positions_cristal()`` in
  ``preprocessing/preprocessing_utils.py``

* Implement ``motor_positions_sixs()`` and ``motor_positions_id01()`` in
  ``preprocessing/preprocessing_utils.py``

* Implement ``find_bragg()`` in ``preprocessing/preprocessing_utils.py``

* New parameter 'binning' in ``postprocessing/strain.py`` to account for binning during
  phasing.

* Converted ``postprocessing/scripts/calc_angles_beam_P10.py`` and
  ``postprocessing/scripts/calc_angles_beam_CRISTAL.py``

* Converted ``postprocessing/scripts/calc_angles_beam_SIXS.py`` and
  ``postprocessing/scripts/calc_angles_beam_ID01.py``

* Converted ``publication/scripts/paper_figure_strain.py``

* Feat: implement ``postprocessing_utils.flip_reconstruction()`` to calculate the
  conjugate object giving the same diffracted intensity.

* Switch the backend to Qt4Agg or Qt5Agg in ``prepare_cdi_mask.py`` to avoid Tk bug
  with interactive interface.

* Correct bug in ``preprocessing_utils.center_fft()`` when 'fix_size' is not empty.

Version 0.0.3
-------------

* Removed cumbersome argument header_cristal in prepare_mask_cdi.py.

* Implement optical path calculation when the data is in crystal frame.

* Correct bugs in ``preprocessing_utils.center_fft()``

* Correct bugs and check consistency in ``postprocessing_utils.get_opticalpath()``.

* Add dataset combining option in ``preprocessing_utils.align_diffpattern()``.

* Checked TODOs in preprocessing_utils

Version 0.0.2
-------------

* Converted ``bcdi/preprocessing/scripts/concatenate_scans.py``

* Converted ``bcdi/preprocessing/scripts/readdata_P10.py``

* Created ``align_diffpattern()`` in ``bcdi/preprocessing/preprocessing_utils.py``

* Created ``find_datarange()`` in ``bcdi/postprocessing/postprocessing_utils.py``

* Created ``sort_reconstruction()`` in ``bcdi/postprocessing/postprocessing_utils.py``

* Implemented regridding on the orthogonal frame of the diffraction pattern for P10
  dataset.

* Removed cumbersome argument headerlines_P10 in prepare_mask_cdi.py, use string parsing
  instead.

Version 0.0.1
-------------
* Initial add, for the moment only the main scripts have been converted and checked:
  ``strain.py`` and ``prepare_cdi_mask.py``

EOF
