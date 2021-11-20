Example: postprocessing
-----------------------

This example will guide you throught the post-processing of an experimental BCDI dataset
measured at the European Synchrotron Radiation Facility, beamline ID01. Postprocessing
occurs after phase retrieval, and typically include the following steps: phase rampe
removal, refraction correction, interpolation of the object from the detector frame to
an orthonormal frame etc...

It is assumed that the phase retrieval of this dataset has been realized
(e.g. using PyNX), and that the data provided as input is in direct space, with the
first axis corresponding to the rocking dimension. If it is still in the detector frame,
the second axis corresponds to the detector vertical axis and the third axis to the
detector horizontal axis.

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

The phase retrieval output will be loaded and a plot is opened.

