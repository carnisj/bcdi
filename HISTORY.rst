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
