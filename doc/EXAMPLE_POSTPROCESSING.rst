Example: postprocessing
-----------------------

This example will guide you throught the post-processing of an experimental BCDI dataset
measured at the European Synchrotron Radiation Facility, beamline ID01. Postprocessing
occurs after phase retrieval, and typically include the following steps: phase ramp
removal, refraction correction, interpolation of the object from the detector frame to
an orthonormal frame etc...

It is assumed that the phase retrieval of this dataset has been realized
(e.g. using PyNX), and that the data provided as input is in direct space. If it is
still in the detector frame, the first axis is expected to correspond to the rocking
dimension, the second axis corresponds to the detector vertical axis and the third axis
to the detector horizontal axis.

The most usefull script is ``bcdi_strain.py``. It requires a YAML config
file, which for this example is ``bcdi/doc/example/S11_config_postprocessing.yml``.

In order to have it running correctly on your machine, you will have to modify the paths
corresponding to the following parameters::

    root_folder: "path_to/CXIDB-I182/CH4760/"
    save_dir: "path_to_saving_directory"
    data_dir: "path_to/bcdi/doc/example/modes_S11.h5"

The script should run properly with other parameters unchanged.

After activating your virtual environment (assuming you created one, which is a good
practice), run:

``python path_to/bcdi_strain.py --conf path_to/S11_config_postprocessing.yml``

The phase retrieval output will be loaded and several postprocessing steps are applied
automatically. Below are few important points about parameters:

  - the interpolation has been extensively tested on several datasets, although we can't
    guarantee that the code is 100% bug-free. If the reconstructed crystal after
    interpolation looks distorted, most probably the parameters `original_size`,
    `phasing_binning` and `preprocessing_binning` are incorrect.

  -  in case of asymmetric crystal where the orientation is known, and if your crystal
     looks flipped after interpolation, use the parameter `flip_reconstruction` to
     use the complex conjugate instead.

  -  `invert_phase` should be True for experimental data. This is due to the sign
     convention of the FFT in Python. For more details, see Scientific reports 9, 17357
     (2019) DOI: 10.1038/s41598-019-53774-2

  -  you have the possibility to declare detector angles (corrected for the direct beam
     position) with the parameters `outofplane_angle` and `inplane_angle`. If these
     parameters are None, the script will try to load them from the log file and will
     apply the correction if the direct beam position is provided in the config file.
