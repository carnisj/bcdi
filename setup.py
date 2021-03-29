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


setup(name='bcdi', version='0.1.0',
      packages=find_packages(),
      include_package_data=True,
      # package_data={'bcdi/preprocessing': ['bcdi/preprocessing/alias_dict.txt']},  # the file needs to be in a package
      # data_files=[('bcdi/data', ['bcdi/data/S978_LLKf000.460.cxi'])], # data files will be installed
      # outtside of the package, which is not ideal
      scripts=['bcdi/scripts/algorithms/flatten_intensity.py',
               'bcdi/scripts/experiment/ccd_calib.py',
               'bcdi/scripts/experiment/ccd_calib_cristal.py',
               'bcdi/scripts/experiment/ccd_calib_sixs.py',
               'bcdi/scripts/facet_recognition/facet_strain.py',
               'bcdi/scripts/facet_recognition/polarplot.py',
               'bcdi/scripts/graph/3Dobject_movie.py',
               'bcdi/scripts/graph/linecut_diffpattern.py',
               'bcdi/scripts/graph/merge3D_qspace.py',
               'bcdi/scripts/graph/scan_analysis.py',
               'bcdi/scripts/graph/scan_movie.py',
               'bcdi/scripts/graph/view_ccf.py',
               'bcdi/scripts/graph/view_ccf_map.py',
               'bcdi/scripts/graph/view_mesh.py',
               'bcdi/scripts/graph/view_psf.py',
               'bcdi/scripts/graph/visu_2Dslice.py',
               'bcdi/scripts/postprocessing/amp_histogram.py',
               'bcdi/scripts/postprocessing/angular_profile.py',
               'bcdi/scripts/postprocessing/bcdi_blurring_function.py',
               'bcdi/scripts/postprocessing/compare_CDI_SEM.py',
               'bcdi/scripts/postprocessing/correct_angles_detector.py',
               'bcdi/scripts/postprocessing/bulk_surface_strain.py',
               'bcdi/scripts/postprocessing/angular_profile.py',
               'bcdi/scripts/postprocessing/line_profile.py',
               'bcdi/scripts/postprocessing/modes_decomposition.py',
               'bcdi/scripts/postprocessing/post_process_CDI_2D.py',
               'bcdi/scripts/postprocessing/prtf_bcdi.py',
               'bcdi/scripts/postprocessing/prtf_bcdi_2D.py',
               'bcdi/scripts/postprocessing/prtf_cdi.py',
               'bcdi/scripts/postprocessing/rocking_curves.py',
               'bcdi/scripts/postprocessing/strain.py',
               'bcdi/scripts/postprocessing/strain_mean_var_rms.py',
               'bcdi/scriptspostprocessing//volume_phasing.py',
               'bcdi/scripts/preprocessing/apodize.py',
               'bcdi/scripts/preprocessing/concatenate_scans.py',
               'bcdi/scripts/preprocessing/interpolate_cdi.py',
               'bcdi/scripts/preprocessing/preprocess_cdi.py',
               'bcdi/scripts/preprocessing/preprocess_bcdi.py',
               'bcdi/scripts/preprocessing/read_bcdi_data.py',
               'bcdi/scripts/preprocessing/read_edf.py',
               'bcdi/scripts/preprocessing/read_data_P10.py',
               'bcdi/scripts/preprocessing/rescale_support.py',
               'bcdi/scripts/preprocessing/rotate_scan.py',
               'bcdi/scripts/publication/coefficient_variation.py',
               'bcdi/scripts/publication/plot_diffpattern_2d.py',
               'bcdi/scripts/publication/compa_simu_exp.py',
               'bcdi/scripts/publication/plot_strain.py',
               'bcdi/scripts/publication/realspace_isosurf_bcdi.py',
               'bcdi/scripts/publication/realspace_isosurf_cdi.py',
               'bcdi/scripts/publication/diffpattern_isosurf_3d.py',
               'bcdi/scripts/simulation/diffraction_angles.py',
               'bcdi/scripts/simulation/domain_orientation.py',
               'bcdi/scripts/simulation/kinematic_sum_forloop.py',
               'bcdi/scripts/simulation/kinematic_sum_pynx.py',
               'bcdi/scripts/simulation/plane_angle.py',
               'bcdi/scripts/simulation/simu_diffpattern_BCDI.py',
               'bcdi/scripts/simulation/simu_diffpattern_CDI.py',
               'bcdi/scripts/simulation/simu_signe_phase.py',
               'bcdi/scripts/utils/bcdi_apodize.py',
               'bcdi/scripts/utils/angular_avg_3Dto1D.py',
               'bcdi/scripts/utils/calibration_grid_SEM.py',
               'bcdi/scripts/utils/center_of_rotation.py',
               'bcdi/scripts/utils/correlation_realspace.py',
               'bcdi/scripts/utils/crop_npz.py',
               'bcdi/scripts/utils/cross_corr_fast_live_macro.py',
               'bcdi/scripts/utils/fit_1Dcurve.py',
               'bcdi/scripts/utils/fit_1D_background.py',
               'bcdi/scripts/utils/parse_fio2spec.py',
               'bcdi/scripts/utils/bcdi_primes.py',
               'bcdi/scripts/utils/save_to_mat.py',
               'bcdi/scripts/xcca/xcca_3D_map_polar.py',
               'bcdi/scripts/xcca/xcca_3D_map_rect.py',
               'bcdi/scripts/xcca/xcca_3D_polar.py',
               'bcdi/scripts/xcca/xcca_3D_rect.py',
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
      python_requires='==3.6.*',
      install_requires=['numpy', 'scipy', 'scikit-image', 'matplotlib', 'hdf5plugin', 'h5py', 'vtk',
                        'mayavi', 'xrayutilities', 'fabio', 'silx', 'lmfit', 'moviepy'],
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
      ],
      )
