# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP 
#       authors:
#         Jerome Carnis, jerome.carnis@esrf.fr

from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


setup(name='bcdii', version='0.0.7',
      packages=find_packages(),
      include_package_data=True,
      # package_data={'bcdi/data': ['bcdi/data/*.cxi']},  # data is not a package, it will not work
      # data_files=[('bcdi/data', ['bcdi/data/S978_LLKf000.460_LLK000.508.cxi'])], # data files will be installed
      # outtside of the package, which is not ideal
      scripts=['bcdi/facet_recognition/scripts/crystal_shape.py',
               'bcdi/graph/scripts/make_movie.py',
               'bcdi/graph/scripts/merge3D_qspace.py',
               'bcdi/graph/scripts/xrutils_Qplot_3Dmayavi.py',
               'bcdi/polarplot/scripts/xrutils_polarplot.py',
               'bcdi/postprocessing/scripts/isosurface_npz.py',
               'bcdi/postprocessing/scripts/post_process_CDI_2D.py',
               'bcdi/postprocessing/scripts/strain.py',
               'bcdi/postprocessing/scripts/resolution_prtf.py',
               'bcdi/postprocessing/scripts/strain_mean_var_rms.py',
               'bcdi/postprocessing/scripts/calc_angles_beam_CRISTAL.py',
               'bcdi/postprocessing/scripts/calc_angles_beam_ID01.py',
               'bcdi/preprocessing/scripts/concatenate_scans.py',
               'bcdi/preprocessing/scripts/prepare_cdi_mask.py',
               'bcdi/publication/scripts/paper_figure_diffpattern.py',
               'bcdi/publication/scripts/paper_figure_isosurface.py',
               'bcdi/publication/scripts/paper_figure_strain.py',
               'bcdi/simulation/scripts/kinematic_sum_forloop.py',
               'bcdi/simulation/scripts/kinematic_sum_pynx.py',
               'bcdi/simulation/scripts/simu_noise_CDI.py',
               'bcdi/simulation/scripts/simu_signe_phase.py',
               'bcdi/utils/scripts/apodize.py',
               'bcdi/utils/scripts/ccd_calib.py',
               'bcdi/utils/scripts/ccd_calib_cristal.py',
               'bcdi/utils/scripts/ccd_calib_sixs.py',
               'bcdi/utils/scripts/compare_dataset_corr.py',
               'bcdi/utils/scripts/concatenate_scans.py',
               'bcdi/utils/scripts/cross_corr_fast_live_macro.py',
               'bcdi/utils/scripts/parse_fio2spec.py',
               'bcdi/utils/scripts/plane_angle.py',
               'bcdi/utils/scripts/primes.py',
               'bcdi/utils/scripts/read_edf.py',
               'bcdi/utils/scripts/readdata_P10.py',
               ],
      # metadata
      author="carnisj",
      author_email="jerome.carnis@esrf.fr",
      description="BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data",
      license="CeCILL-B",
      keywords="BCDI Bragg coherent X-rays diffraction imaging",
      long_description_content_type='text/x-rst',
      long_description="BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data.\n\
                        BCDI stands for *Bragg coherent X-ray diffraction imaging*. It can be used for:\n\n\
                        1. Pre-processing BCDI data (masking aliens, detector gaps...) before phasing.\n\
                        2. Post-processing phased data (phase offset and phase ramp removal, averaging...).\n\
                        3. Data analysis on diffraction data (stereographic projection).\n\
                        4. Data analysis on phased data (resolution calculation, statistics on the strain...).\n\
                        5. Simulation of diffraction intensity (including noise, detector gaps, displacement...).\n\
                        6. Making figures for publication using templates.\n\n",
      url='https://github.com/carnisj/bcdi',
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
