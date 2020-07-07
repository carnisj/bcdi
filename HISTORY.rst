* Refactor: allow the reloading of binned data and its orthogonalization in preprocess_cdi_combined.py and preprocess_bcdi_combined.py

* Feature: implement crop_npz.py to crop combined stacked data to the desired size

* Feature: implement scan_analysis.py to plot interactively the integrated intensity in a region of interest for a 1D scan

* Feature: implement view_mesh.py to plot interactively the integrated intensity in a region of interest for a 2D mesh

* Refactor: when gridding forward CDI data, reverse the rotation direction to compensate the rotation of Ewald sphere

* Refactor: converted /postprocessing/scripts/extract_bulk_surface.py

* Bug: treat correctly the case angle=pi/2 during the interpolation of CDI data onto the laboratory frame

* Refactor: solve instabilities resulting from duplicate vertices after smoothing in facet_strain.py

* Refactor: modify polarplot.py to use module functions instead of inline script

* Feature: implement coefficient_variation.py to compare several reconstructed modulus of a BCDI dataset

* Feature: implement diffraction_angles.py to find Bragg reflections for a particular goniometer setup, based on xrayutilities

* Feature: add the option of restarting masking the aliens during preprocessing, not back compatible with previous versions

* Refactor: rename prepare_(b)cdi_mask.py preprocess_(b)cdi_sequential.py

* Feature: implement simultaneous masking over the 3 axes in two new preprocessing scripts preprocess_(b)cdi_combinedl.py

* Feature: implement domain_orientation.py to find the orientation of domains in a 3D forward CDI dataset of mesocrystal

* Feature: implement simu_diffpattern_CDI.py to find in 3D the Bragg peaks positions of a mesocrystal (supported unit cells: FCC, simple cubic, BCC and BCT)

* Feature: implement fit_1D curve.py to fit simultaneously ranges of a 1D curve with gaussian lineshapes

* Feature: implement fit_background.py to interactively determine the background in 1D reciprocal space curves

* Refactor: in multislices_plot() and contour_slices(), allow to plot the data at user-defined slices positions.

* Feature: implement prtf_bcdi_2D.py to calculate the PRTF also for 2D cases.

Version 0.0.8
-------------

* Feature: implement 3Dobject_movie.py, creating movies of a real-space 3D object.

* Feature: implement modes_decomposition.py, decomposition of a set of reconstructed object in orthogonal modes (adapted from PyNX)

* Bug: correct the calculation of q when data is binned

* implement scripts to visualize isosurfaces of reciprocal/real space including publication options (in /publication/)

* implement algorithms_utils.py, featuring psf and image deconvolution using Richardson-Lucy algorithm

* implement separate PRTF resolution estimation for CDI (prtf_cdi.py) and BCDI (bcdi_prtf.py) datasets

* Feature: implement angular_average.py to average 3D CDI reciprocal space data in 1D curve

* Feature: implement view_psf to plot the psf output of a phase retrieval with partial coherence

* Refactor: change name of make_support.py to rescale_support.py

Version 0.0.7
-------------
* Feature: implement supportMaker() class to define a support from a set of planes

* Feature: implement maskMaker() class for easier implementation of new masking features

* Debug prepare_bcdi_mask.py for energy scans at ID01

* Feature: implement utils/scripts/make_support.py, to rescale a support for phasing with a larger FFT window

* Feature/refactor: implement prepare_cdi_mask.py for forward CDI, rename existing as prepare_bcdi_mask.py for Bragg CDI

* Feature: add the possibility to change the detector distance in simu_noise.py

* Feature: add the possibility to pre-process data acquired without scans, e.g. in a macro (no spec file)

* Feature: in strain.py, implement phase unwrapping so that the phase range can be larger than 2*pi

* Feature: in facet_strain.py, implement edge removal for more precise statistics on facet strain

* Feature: in facet_strain.py, allow anisotropic voxel size and user-defined reference axis in the stereographic projection

Version 0.0.6
-------------

* Feature: implement facet detection using a stereographic projection in facet_recognition/scripts/facet_strain.py

* Feature: Converted bcdi/facet_recognition/scripts/facet_strain.py

* Feature: implement bcdi/facet_recognition/facets_utils.py

* Refactor: exclude voxels left over by coordination number selection in postprocessing/postprocessing_utils.find_bulk()

* Refactor: use the mean amplitude of the surface layer to define the bulk in postprocessing/postprocessing_utils.find_bulk()

* Feature: enable PRTF resolution calculation for simulated data

* Feature: create preprocessing/scripts/apodize.py to apodize reciprocal space data

* Feature: implement 3d Tukey and 3d Blackman windows for apodization in postprocessing_utils()

* Feature: in postprocessing/scripts/resolution_prtf.py, allow for binning the detector plane

* Bug: in postprocessing/scripts/strain.py, correct the original array size taking into account the binning factor

* Feature: implement postprocessing_utils.bin_data()

Version 0.0.5
-------------

* Feature: implement support for SIXS data measured after the 11/03/2019 with the new data recorder.

* Refactor: modify preprocessing/scripts/readdata_P10.py to support several beamlines and rename it 'read_data.py'

* Feature: implement support for multiple beamlines in postprocessing/script/resolution_prtf.py

* Refactor: merge all preprocessing/preprocessing_utils.regrid_*.py in preprocessing/preprocessing_utils.regrid()

* Converted postprocessing/scripts/resolution_prtf.py

* Refactor: add the possibility of giving a single element instead of the full tuple in graph/graph_utils.combined_plots()

* Converted postprocessing/scripts/resolution_prtf.py

* Feature: create a Colormap() class in graph/graph_utils.py

* Refactor: merge all postprocessing/scripts/calc_angles_beam_*.py in postprocessing/scripts/correct_angles_detector.py

* Feature: Implement motor_values() and load_data() in preprocessing/preprocessing_utils.py

* Feature: Implement SetupPostprocessing.rotation_direction() in experiment/experiment_utils.py

* Feature: add other counter name 'curpetra' for beam intensity monitor at P10

* Bug: postprocessing/scripts/calc_angles_beam_*.py: correct bug when roi_detector is not defined, and round the Bragg peak COM to integer pixels

Version 0.0.4
-------------

* Implement motor_positions_p10(), motor_positions_cristal() in preprocessing/preprocessing_utils.py

* Implement motor_positions_sixs() and motor_positions_id01() in preprocessing/preprocessing_utils.py

* Implement find_bragg() in preprocessing/preprocessing_utils.py

* New parameter 'binning' in postprocessing/strain.py to account for binning during phasing.

* Converted postprocessing/scripts/calc_angles_beam_P10.py and postprocessing/scripts/calc_angles_beam_CRISTAL.py

* Converted postprocessing/scripts/calc_angles_beam_SIXS.py and postprocessing/scripts/calc_angles_beam_ID01.py

* Converted publication/scripts/paper_figure_strain.py

* Feat: implement postprocessing_utils.flip_reconstruction() to calculate the conjugate object giving the same diffracted intensity.

* Switch the backend to Qt4Agg or Qt5Agg in prepare_cdi_mask.py to avoid Tk bug with interactive interface.

* Correct bug in preprocessing_utils.center_fft() when 'fix_size' is not empty.

Version 0.0.3
-------------

* Removed cumbersome argument header_cristal in prepare_mask_cdi.py.

* Implement optical path calculation when the data is in crystal frame.

* Correct bugs in preprocessing_utils.center_fft().

* Correct bugs and check consistency in postprocessing_utils.get_opticalpath().

* Add dataset combining option in preprocessing_utils.align_diffpattern().

* Checked TODOs in preprocessing_utils

Version 0.0.2
-------------

* Converted bcdi/preprocessing/scripts/concatenate_scans.py

* Converted bcdi/preprocessing/scripts/readdata_P10.py

* Created align_diffpattern() in bcdi/preprocessing/preprocessing_utils.py

* Created find_datarange() in bcdi/postprocessing/postprocessing_utils.py

* Created sort_reconstruction() in bcdi/postprocessing/postprocessing_utils.py

* Implemented regridding on the orthogonal frame of the diffraction pattern for P10 dataset.

* Removed cumbersome argument headerlines_P10 in prepare_mask_cdi.py, use string parsing instead.

Version 0.0.1
-------------
* Initial add, for the moment only the main scripts have been converted and checked: strain.py and prepare_cdi_mask.py 

EOF
