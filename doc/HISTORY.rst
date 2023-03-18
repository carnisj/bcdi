Version 0.3.1:
--------------

* Update dependencies.

* Allow the config parameter 'frames_pattern' to be a list containing only the indices
  of the detector frames to skip.

Version 0.3.0:
--------------

* `bcdi_utils.find_bragg`: a parameter name in the function signature was change from
  "roi" to "region_of_interest", therefore the region of interest was not taken into
  account anymore when calculating the peak position.

* Postprocessing: fix bragg peak calculation for energy scans.

* Postprocessing: raise NotImplementedError for energy scans not interpolated during
  preprocessing.

Version 0.2.9:
--------------

* Add support for Python3.10 and update dependencies.

* Refactor the function `bcdi_utils.center_fft` in separate classes using
  inheritance.

* Split `setup.ortho_directspace` into methods to get the transformation matrix in the
  laboratory frame or crystal frame. Implement `setup.ortho_directspace_labframe`.

* Create a new script `postprocessing/bcdi_orthogonalization.py`, which only
  interpolates the output of phase retrieval (no processing on the phase)

* Remove temporal couping in the initialization of the Setup instance. Set the paths,
  create the logfile and read it directly in `setup.__init__`.

* Create classes Analysis, PreprocessingLoader and InteractiveMasker for preprocessing.
  Refactor `preprocessing.process_scan` to use these classes.

* Add support for ESRF BM02 beamline (th rocking curve).

* Create classes Analysis, PhaseManipulator and InterpolatedCrystal for postprocessing.
  Refactor `postprocessing.process_scan` to use these classes.

* Remove the deprecated parameter 'fix_size'. THe same functionality can be obtained by
  using the set of parameters `roi_detector`, `center_roi_x` and `center_roi_y`.

* Previously the functionalities regarding Bragg peak finding, rocking curve fitting and
  plotting were located in tightly binded functions. A class PeakFinder is implemented to
  gather these functionalities and manage its own state properly.

* Add unit tests to postprocessing_utils.find_datarange and make it dimension-agnostic.

* Fix a bug with data path formatting in ID01BLISS loader (missing parentheses when
  creating the data path in the h5 file).

Version 0.2.8:
--------------

* Add plot in `bcdi_utils.find_bragg` to check qualitatively the configured centering
  method. Save it automatically with the title 'centering_method.png'.

* Refactor: move all methods related to linecuts in a dedicated module and class.
  Improve the robustness of linecuts fits in postprocessing, by allowing each fit to
  have a different length.

* Bug: allow `sample_name` to be an empty string, in order to match the data internal
  path in ID01BLISS h5 files.

* Bug: override  `centering_method` in PreprocessingChecker when the Bragg peak position
  is provided (the new setting becomes "user" for reciprocal space).

* Bug: add the method `utils.parameters.ConfigChecker._create_dirs`, which creates the
  saving directory when the folder does not exist yet.

* Bug: provide the correct binning factors to `bcdi_utils.find_bragg` by introducing a
  new detector property `current_binning` and using it as input parameter.

* Calculate the Bragg peak position for the three methods "max", "com" and "max_com"
  and plot the results to compare their efficiency. The type of `centering_method` is
  changed to `dict` in order to provide different methods for direct and reciprocal
  space (e.g. `{"direct_space": "max_com", "reciprocal_space": "max"}`).

* Use distinct names for the log files from preprocessing and postprocessing.

* Move the calculation of corrected detector angles before the calculation of `q_lab`
  in `postprocessing.process_scan`. Previously it was using the values from the beamline
  log file, resulting in a small discrepancy.

* Add a placeholder "-f" command line parameter to host the automatically generated
  parameter in Jupyter notebooks (path of the kernel json file).

* Add support for high-energy BCDI at ESRF ID27 beamline. Add an example configuration
  file to the package.

* Rename `bcdi_preprocess_BCDI.py` to `bcdi_preprocess.py`, add support for CDI in the
  script.

* Add option to skip unwrapping: add the boolean parameter `skip_unwrap` to the
  postprocessing. This can be useful when there are defects in the crystal and phase
  unwrapping does not work well.

* Bug: provide the correct number of motor positions to xrayutilities Qconv for the SIXS
  beamline (beta needs to be duplicated because it is also below the detector circles.)

* Modify the saturation threshold of the Eiger2M detector from 1e6 to 1e7 photons.

* The log file will be created in `save_dir` if this parameter is defined, otherwise
  it will be created in `root_folder`.

* Allow loading of Eiger2M detector data at ID01BLISS.

Version 0.2.7:
--------------

* Use the package `isort` to format imports in a standard way.

* Use multiprocessing for the analysis of several scans in `bcdi_preprocessing_BCDI.py`.
  In this case, `flag_interact` has to be False, and an optional mask can be loaded to
  be applied on the data. The use case is when one runs once the preprocessing manually
  and creates such a mask (it is saved automatically after the interactive masking as
  `interactive_mask.npz`), and afterwards wants to apply it to a series of scans
  measured in the same geometry. For each scan, a separate log file is created and
  saved in `save_dir` along with processing results.

* A parameter `mask` was added to the config file for preprocessing. The loaded mask
  will be combined with the one generated automatically and used independently of
  `flag_interact`.

* Use multiprocessing for the analysis of several scans in ``bcdi_strain.py``. The
  use-case is when there is a series of scans measured with the same geometry and only
  an external parameter (gaz, temperature,...) changes. For each scan, a separate log
  file is created and saved in `save_dir` along with processing results.

Version 0.2.6:
--------------

* Fix deprecation warning with the method `pandas.DataFrame.append`

* Implement the `ConfigChecker` class and child classes to gather in a single location
  the configuration check before processing.

* rename `scan` to `scans` in `bcdi_strain.py`, and allow the user to process several
  scans. Allow to provide a list for `specfile_name`, `template_imagefile`, `data_dir`,
  `save_dir`, `reconstruction_files` with the same number of elements as the number of
  scans.

* rename the parameter `sdd` to `detector_distance` to match PyNX terminology.

* Add a `colormap` parameter to the config. The user can choose a colormap between
  "turbo", "custom" or any colormap from the `colorcet` package

* Add an estimation of the resolution in `bcdi_strain.py` using a linecut of the
  reconstructed modulus and Gaussian fits to the derivative of the linecut at crystal
  edges.

* Add a CITATION.cff file for easier citation of the package in scientific publications.

* Change the default colormap for the turbo colormap.
  See https://ai.googleblog.com/2019/08/turbo-improved-rainbow-colormap-for.html

Version 0.2.5:
--------------

* Remove support for Python 3.6 and Python 3.7

* Use exact versions of dependencies in `setup.py` and `requirements.txt` to improve the
  reproducibility of results.

* Bug: use explicitely the figure handle to save figures in `bcdi_strain.py` and
  `bcdi_preprocessing_BCDI.py`

* Refactor: implement correctly the `__repr__` methods so that eval(repr(instance)) is
  itself an instance.

Version 0.2.4:
--------------

* Refactor: open files within context managers in `utilities.load_file`.

* Create the class ContextFile implementing the context manager protocol. The method
  `loader.create_logfile` returns an instance of ContextFile, allowing its safe usage
  in other methods via the decorators `@safeload` and `@safeload_static`.

* Refactor the signature of `loader.create_logfile`, so that all overriding methods
  share the same signature with the overriden base class abstract method.

* Implement the function `utilities.unpack_array`, to avoid hard-coding array slices
  in methods using motor positions.

* Feature: add support for the BLISS flavor of ID01 beamline.

Version 0.2.3:
--------------

* Refactor: allow the command line parameters to be None. ConfigParser can now be
  instantiated providing the path to the config file only.

* Bug: use the binned detector pixel size to calculate the transformation matrices

Version 0.2.2:
--------------

* Move the configuration files and example modes data to a subpackage ``bcdi.examples``
  so that they are distributed with the builds and wheels.

* Move the run functions from `bcdi_strain.py` and `bcdi_preprocessing_BCDI.py` to new
  dedicated modules `postprocessing.postprocessing_runner` and
  `preprocessing.preprocessing_runner`, so that it can be imported.

* Add an example in the documentation, using the dataset ID182 from the CXIDB.

* Bug with interpolation with xrayutilities: use the correct number of values to unpack
  from the call to `beamline.Beamline.process_position` (the detector distance was
  missing)

* Allow the correction of detector angles directly in `bcdi_preprocessing_BCDI.py` and
  `bcdi_strain.py`. The user can either provide the Bragg peak position, or this one
  will be calculated from the provided direct beam position and the setup geometry.
  The user can still provide already corrected detector angles.

* Bug: use the unbinned detector pixel size in `detector.create_roi`,
  `diffractometer.init_data_mask` and `diffractometer.load_check_dataset` instead of
  the calculated pixel size, which includes the preprocessing binning.

* Rename the parameters for the definition of the detector ROI for data loading
  `x_bragg` and `y_bragg` as `center_roi_x` and `center_roi_y`, to avoid the confusion
  with the Bragg peak position.

Version 0.2.1:
--------------

* Allow lazy loading of the experimental parameters: energy, detector distance, tilt
  angle, detector inplane and out-of-plane angles. If not provided by the user, the
  values will be looked for in the log/spec file. An exception is raised if they are
  not available.

* Implement the chi circle at 34ID-C and update the calculation of the transformation
  matrix.

* Add a task in `doit` to check for broken external links in the documentation.

* The methods setup.ortho_directspace and setup.ortho_reciprocal now return also the
  transformation matrix from the detector frame to the laboratory frame.

* Support APS 34ID-C for data preprocessing (loading of TIFF images).

* Update the parameter "linearity_function" in preprocessing. Now this can be None (if
  unused) or a sequence of 5 real numbers corresponding to the coefficients of a 4th
  order polynomial.

* Implement a parser for YAML config files. Now the scripts ``bcdi_strain.py`` and
  ``bcdi_preprocess_BCDI.py`` can be run like scripts, from the command line, with
  optional command line arguments.

Version 0.1.7
-------------

* Bug: apply correctly the non-linearity correction function to the detector frames
  (typo in the function name).

* Modify the behavior of the parameter `specfile_name`: for beamlines relying on a
  separate file for logging motor positions (specfiles at ID01 and 34ID, fio file at
  P10), the user can provide the full path to the file

* Implement loading motor positions from a specfile at 34ID.

* Add mypy for type checking in doit and GitHub workflows.

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
  ``facet_utils.py`` to ``facet_recognition.py``.

* Add the list of publications related to the package in the documentation.

* Add class diagrams in the documentation using sphinxcontrib-mermaid.

* Solve issues with numpy when building the documentation (enable *Use system packages*
  in ReadTheDocs advanced settings).

Version 0.1.6
-------------

* Write unit tests for ``experiment.detector.py``, now coverage is > 99% for this
  module.

* move forward CDI gridding function to ``Setup``.

* implement ``DiffractometerP10SAXS`` and ``BeamlineP10SAXS`` classes for forward CDI
  experiments at P10.

* split the module ``preprocessing_utils`` in two modules, ``bcdi_utils`` and
  ``cdi_utils``.

* Move generic functions from ``preprocessing_utils`` to ``utilities``.

* Create new validations functions ``valid_ndarray`` and ``valid_1d_array``, implement
  the corresponding unit tests.

* Refactor: rename ``preprocessing_utils.regrid`` to ``calc_qvalues_xrutils`` and move
  it to ``Setup``. Put all the beamline dependent code in the corresponding ``Beamline``
  child class.

Version 0.1.5
-------------

* Bug: convert arrays to a tuple before checking the length in Setup.ortho_directspace.

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
  2D).

* Refractor: split bcdi.experiment.experiment_utils module into smaller modules.

* Refactor: enforce project's guidelines for the code style and the docstrings.

* Create a dodo.py file (doit package) to simplify the life of contributors: now they
  just need to run doit at the same level as setup.py and verify that all checks pass
  before sending their pull request.

* Create a CONTRIBUTING.md file.

Version 0.1.2
-------------

* Refactor: remove circular imports from modules.

* Refactor: ``move crop_pad``, ``bin_data`` and ``gaussian_window functions`` from
  ``postprocessing_utils.py`` to another module in order to avoid circular imports.

* Feature: create a Diffractometer class with one child class for each beamline, move
  all functions related to the goniometer positions in the class.

* Feature: add an option in ``strain.py`` to put back the sample in the laboratory
  frame with all sample circles rotated back to 0 deg.

* Refactor: show only necessary plots and console output in ``strain.py``.

* Refactor: create Setup calculated properties and transfer calculations in scripts to
  these properties.

* Refactor: perform the geometrical transformation and rotation of the reconstructed
  crystal in a single step.

* Refactor: perform the geometrical transformation and rotation of the diffraction
  pattern in a single step.

* Bug: provide voxel sizes in the correct order when rotating the diffraction pattern
  in ``preprocess_bcdi.py``.

Version 0.1.1
-------------

* code cleaning.

Version 0.1.0
-------------

* Feature: implement ``publication/bcdi_diffpattern_from_reconstruction.py``, to
  compare with the experimental measurement in the crystal frame.

* Refactor: simplify PRTF calculations.

* Feature: implement the inplane rocking curve at CRISTAL.

* Feature: implement ``graph_utils.savefig`` to save figures for publication with and
  without labels.

* Feature: implement ``angular_profile.py`` to calculate the width of linecuts through
  the center of mass of a 2D object at different angles.

* Feature: implement ``line_profile.py`` to calculate line profiles along particular
  directions in 2D or 3D objects.

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
