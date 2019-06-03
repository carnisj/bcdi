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
