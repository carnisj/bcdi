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


setup(name='bcdi', version='0.0.8',
      packages=find_packages(),
      include_package_data=True,
      # package_data={'bcdi/preprocessing': ['bcdi/preprocessing/alias_dict.txt']},  # the file needs to be in a package
      # data_files=[('bcdi/data', ['bcdi/data/S978_LLKf000.460.cxi'])], # data files will be installed
      # outtside of the package, which is not ideal
      scripts=['bcdi/algorithms/scripts/flatten_modulus.py',
               'bcdi/experiment/scripts/ccd_calib.py',
               'bcdi/experiment/scripts/ccd_calib_cristal.py',
               'bcdi/experiment/scripts/ccd_calib_sixs.py',
               'bcdi/facet_recognition/scripts/facet_strain.py',
               'bcdi/facet_recognition/scripts/polarplot.py',
               'bcdi/graph/scripts/3Dobject_movie.py',
               'bcdi/graph/scripts/scan_movie.py',
               'bcdi/graph/scripts/merge3D_qspace.py',
               'bcdi/graph/scripts/view_ccf.py',
               'bcdi/graph/scripts/view_ccf_map.py',
               'bcdi/graph/scripts/view_psf.py',
               'bcdi/graph/scripts/visu_2Dslice.py',
               'bcdi/graph/scripts/xrutils_Qplot_3Dmayavi.py',
               'bcdi/postprocessing/scripts/amp_histogram.py',
               'bcdi/postprocessing/scripts/modes_decomposition.py',
               'bcdi/postprocessing/scripts/post_process_CDI_2D.py',
               'bcdi/postprocessing/scripts/strain.py',
               'bcdi/postprocessing/scripts/prtf_bcdi.py',
               'bcdi/postprocessing/scripts/prtf_bcdi_2D.py',
               'bcdi/postprocessing/scripts/prtf_cdi.py',
               'bcdi/postprocessing/scripts/strain_mean_var_rms.py',
               'bcdi/postprocessing/scripts/correct_angles_detector.py',
               'bcdi/postprocessing/scripts/extract_bulk_surface.py',
               'bcdi/postprocessing/scripts/volume_phasing.py',
               'bcdi/preprocessing/scripts/apodize.py',
               'bcdi/preprocessing/scripts/concatenate_scans.py',
               'bcdi/preprocessing/scripts/preprocess_cdi.py',
               'bcdi/preprocessing/scripts/preprocess_bcdi.py',
               'bcdi/preprocessing/scripts/read_bcdi_data.py',
               'bcdi/preprocessing/scripts/read_edf.py',
               'bcdi/preprocessing/scripts/read_data_P10.py',
               'bcdi/preprocessing/scripts/rescale_support.py',
               'bcdi/publication/scripts/coefficient_variation.py',
               'bcdi/publication/scripts/paper_figure_diffpattern.py',
               'bcdi/publication/scripts/paper_figure_isosurface.py',
               'bcdi/publication/scripts/paper_figure_strain.py',
               'bcdi/publication/scripts/realspace_isosurf_bcdi.py',
               'bcdi/publication/scripts/realspace_isosurf_cdi.py',
               'bcdi/publication/scripts/diffpattern_isosurf_3d.py',
               'bcdi/simulation/scripts/diffraction_angles.py',
               'bcdi/simulation/scripts/domain_orientation.py',
               'bcdi/simulation/scripts/kinematic_sum_forloop.py',
               'bcdi/simulation/scripts/kinematic_sum_pynx.py',
               'bcdi/simulation/scripts/plane_angle.py',
               'bcdi/simulation/scripts/simu_diffpattern_BCDI.py',
               'bcdi/simulation/scripts/simu_diffpattern_CDI.py',
               'bcdi/simulation/scripts/simu_signe_phase.py',
               'bcdi/utils/scripts/apodize.py',
               'bcdi/utils/scripts/correlation_realspace.py',
               'bcdi/utils/scripts/crop_npz.py',
               'bcdi/utils/scripts/cross_corr_fast_live_macro.py',
               'bcdi/utils/scripts/fit_1Dcurve.py',
               'bcdi/utils/scripts/fit_1D_background.py',
               'bcdi/utils/scripts/parse_fio2spec.py',
               'bcdi/utils/scripts/primes.py',
               'bcdi/utils/scripts/save_to_mat.py',
               'bcdi/utils/scripts/scan_analysis.py',
               'bcdi/utils/scripts/view_mesh.py',
               'bcdi/xcca/scripts/angular_avg_3Dto1D.py',
               'bcdi/xcca/scripts/xcca_3D.py',
               'bcdi/xcca/scripts/xcca_3D_map.py',
               ],
      # metadata
      author="Jerome Carnis",
      author_email="carnis_jerome@yahoo.fr",
      description="BCDI: tools for pre(post)-processing Bragg and forward coherent X-ray diffraction imaging data",
      license="CeCILL-B",
      keywords="BCDI Bragg coherent X-rays diffraction imaging",
      long_description_content_type='text/x-rst',
      long_description="BCDI: tools for pre(post)-processing Bragg and forward coherent X-ray diffraction imaging data.\n\
                        BCDI stands for *Bragg coherent X-ray diffraction imaging*. It can be used for:\n\n\
                        1. Pre-processing BCDI and forward CDI data (masking aliens, detector gaps...) before phasing.\n\
                        2. Post-processing phased data (phase offset and phase ramp removal, averaging, apodization, ...).\n\
                        3. Data analysis on diffraction data (stereographic projection).\n\
                        4. Data analysis on phased data (resolution calculation, statistics on the retrieved strain...).\n\
                        5. Simulation of diffraction intensity (including noise, detector gaps, displacement...).\n\
                        6. Making figures for publication using templates.\n\n",
      url='https://github.com/carnisj/bcdi',
      project_urls={
          'Documentation': 'https://bcdi.readthedocs.io/en/latest/'},
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
          'Programming Language :: Python :: 3 :: Only',
      ],
      )
