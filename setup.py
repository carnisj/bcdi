# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


setup(name='bcdi', version='0.1.1',
      packages=find_packages(),
      include_package_data=True,
      # package_data={'bcdi/preprocessing': ['bcdi/preprocessing/alias_dict.txt']},  # the file needs to be in a package
      # data_files=[('bcdi/data', ['bcdi/data/S978_LLKf000.460.cxi'])], # data files will be installed
      # outtside of the package, which is not ideal
      scripts=['bcdi/scripts/algorithms/bcdi_flatten_intensity.py',
               'bcdi/scripts/algorithms/bcdi_blurring_function.py',
               'bcdi/scripts/experiment/bcdi_ccd_calib.py',
               'bcdi/scripts/experiment/bcdi_ccd_calib_cristal.py',
               'bcdi/scripts/experiment/bcdi_ccd_calib_sixs.py',
               'bcdi/scripts/facet_recognition/bcdi_facet_strain.py',
               'bcdi/scripts/facet_recognition/bcdi_polarplot.py',
               'bcdi/scripts/graph/bcdi_3D_object_movie.py',
               'bcdi/scripts/graph/bcdi_linecut_diffpattern.py',
               'bcdi/scripts/graph/bcdi_merge_3D_qspace.py',
               'bcdi/scripts/graph/bcdi_scan_analysis.py',
               'bcdi/scripts/graph/bcdi_scan_movie.py',
               'bcdi/scripts/graph/bcdi_view_ccf.py',
               'bcdi/scripts/graph/bcdi_view_ccf_map.py',
               'bcdi/scripts/graph/bcdi_view_mesh.py',
               'bcdi/scripts/graph/bcdi_view_psf.py',
               'bcdi/scripts/graph/bcdi_visu_2D_slice.py',
               'bcdi/scripts/postprocessing/bcdi_amp_histogram.py',
               'bcdi/scripts/postprocessing/bcdi_angular_profile.py',
               'bcdi/scripts/postprocessing/bcdi_compare_BCDI_SEM.py',
               'bcdi/scripts/postprocessing/bcdi_correct_angles_detector.py',
               'bcdi/scripts/postprocessing/bcdi_bulk_surface_strain.py',
               'bcdi/scripts/postprocessing/bcdi_angular_profile.py',
               'bcdi/scripts/postprocessing/bcdi_line_profile.py',
               'bcdi/scripts/postprocessing/bcdi_modes_decomposition.py',
               'bcdi/scripts/postprocessing/bcdi_post_process_BCDI_2D.py',
               'bcdi/scripts/postprocessing/bcdi_prtf_BCDI.py',
               'bcdi/scripts/postprocessing/bcdi_prtf_BCDI_2D.py',
               'bcdi/scripts/postprocessing/bcdi_prtf_CDI.py',
               'bcdi/scripts/postprocessing/bcdi_rocking_curves.py',
               'bcdi/scripts/postprocessing/bcdi_strain.py',
               'bcdi/scripts/postprocessing/bcdi_strain_mean_var_rms.py',
               'bcdi/scripts/postprocessing/bcdi_volume_phasing.py',
               'bcdi/scripts/preprocessing/bcdi_apodize.py',
               'bcdi/scripts/preprocessing/bcdi_concatenate_scans.py',
               'bcdi/scripts/preprocessing/bcdi_interpolate_CDI.py',
               'bcdi/scripts/preprocessing/bcdi_preprocess_BCDI.py',
               'bcdi/scripts/preprocessing/bcdi_preprocess_CDI.py',
               'bcdi/scripts/preprocessing/bcdi_read_BCDI_scan.py',
               'bcdi/scripts/preprocessing/bcdi_read_edf.py',
               'bcdi/scripts/preprocessing/bcdi_read_data_P10.py',
               'bcdi/scripts/preprocessing/bcdi_rescale_support.py',
               'bcdi/scripts/preprocessing/bcdi_rotate_scan.py',
               'bcdi/scripts/publication/bcdi_coefficient_variation.py',
               'bcdi/scripts/publication/bcdi_compa_simu_exp.py',
               'bcdi/scripts/publication/bcdi_diffpattern_from_reconstruction.py',
               'bcdi/scripts/publication/bcdi_diffpattern_isosurf_3D.py',
               'bcdi/scripts/publication/bcdi_plot_diffpattern_2D.py',
               'bcdi/scripts/publication/bcdi_plot_strain.py',
               'bcdi/scripts/publication/bcdi_realspace_isosurf_BCDI.py',
               'bcdi/scripts/publication/bcdi_realspace_isosurf_CDI.py',
               'bcdi/scripts/simulation/bcdi_diffraction_angles.py',
               'bcdi/scripts/simulation/bcdi_domain_orientation.py',
               'bcdi/scripts/simulation/bcdi_kinematic_sum_forloop.py',
               'bcdi/scripts/simulation/bcdi_plane_angle.py',
               'bcdi/scripts/simulation/bcdi_simu_diffpattern_BCDI.py',
               'bcdi/scripts/simulation/bcdi_simu_diffpattern_CDI.py',
               'bcdi/scripts/simulation/bcdi_simu_signe_phase.py',
               'bcdi/scripts/utils/bcdi_apodize.py',
               'bcdi/scripts/utils/bcdi_angular_avg_3Dto1D.py',
               'bcdi/scripts/utils/bcdi_calibration_grid_SEM.py',
               'bcdi/scripts/utils/bcdi_center_of_rotation.py',
               'bcdi/scripts/utils/bcdi_correlation_realspace.py',
               'bcdi/scripts/utils/bcdi_crop_npz.py',
               'bcdi/scripts/utils/bcdi_cross_corr_fast_live.py',
               'bcdi/scripts/utils/bcdi_fit_1D_curve.py',
               'bcdi/scripts/utils/bcdi_fit_1D_background.py',
               'bcdi/scripts/utils/bcdi_parse_fio2spec.py',
               'bcdi/scripts/utils/bcdi_primes.py',
               'bcdi/scripts/utils/bcdi_save_to_mat.py',
               'bcdi/scripts/xcca/bcdi_xcca_3D_map_polar.py',
               'bcdi/scripts/xcca/bcdi_xcca_3D_map_rect.py',
               'bcdi/scripts/xcca/bcdi_xcca_3D_polar.py',
               'bcdi/scripts/xcca/bcdi_xcca_3D_rect.py',
               ],
      # metadata
      author="Jerome Carnis",
      author_email="carnis_jerome@yahoo.fr",
      description="BCDI: tools for pre(post)-processing Bragg and forward coherent X-ray diffraction imaging data",
      license="CeCILL-B",
      keywords="BCDI Bragg coherent X-rays diffraction imaging",
      long_description_content_type='text/x-rst',
      long_description="BCDI: tools for pre(post)-processing Bragg and forward coherent X-ray diffraction imaging data.\
                       \n\
                        BCDI stands for *Bragg coherent X-ray diffraction imaging*. It can be used for:\n\n\
                        1. Pre-processing BCDI and forward CDI data (masking aliens, detector gaps...)\
                         before phase retrieval.\n\
                        2. Post-processing phased data (phase offset and phase ramp removal, averaging,\
                         apodization...).\n\
                        3. Data analysis on diffraction data (stereographic projection, angular cross correlation\
                         analysis, domain orientation fitting ...).\n\
                        4. Data analysis on phased data (resolution calculation, statistics on the\
                         retrieved strain...).\n\
                        5. Simulation of diffraction intensity (including noise, detector gaps, displacement...).\n\
                        6. Making figures for publication using templates.\n\n",
      url='https://github.com/carnisj/bcdi',
      project_urls={
          'Documentation': 'https://bcdi.readthedocs.io/en/latest/'},
      python_requires='>=3.6, !=3.9.*',
      install_requires=['numpy', 'scipy', 'scikit-image', 'matplotlib', 'hdf5plugin', 'h5py', 'traits', 'vtk',
                        'importlib_resources', 'mayavi', 'xrayutilities', 'fabio', 'silx'],
      extra_requires=['tables', 'lmfit', 'moviepy'],
      classifiers=[
          # How mature is this project? Common values are
          #   3 - Alpha
          #   4 - Beta
          #   5 - Production/Stable
          'Development Status :: 3 - Alpha',

          # Indicate who your project is intended for
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering :: Physics',
          # Pick your license as you wish
          'License :: CeCILL-B Free Software License Agreement (CECILL-B)',

          # Specify the Python versions you support here. In particular, ensure
          # that you indicate whether you support Python 2, Python 3 or both.
          # These classifiers are *not* checked by 'pip install'. See instead
          # 'python_requires' below.
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
      ],
      )
