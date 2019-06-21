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
