# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np
from numpy.fft import fftshift
from skimage.restoration.deconvolution import richardson_lucy
import sys
sys.path.append('D:/myscripts/bcdi/')
import bcdi.postprocessing.postprocessing_utils as pu
import bcdi.graph.graph_utils as gu


def deconvolution_rl(image, psf=None, iterations=20, debugging=False):
    """
    Image deconvolution using Richardson-Lucy algorithm. The algorithm is based on a PSF (Point Spread Function),
     where PSF is described as the impulse response of the optical system.

    :param image: image to be deconvoluted
    :param psf: psf to be used as a first guess
    :param iterations: number of iterations for the Richardson-Lucy algorithm
    :param debugging: True to see plots
    :return:
    """
    image = image.astype(np.float)
    ndim = image.ndim
    if psf is None:
        print('Initializing the psf using a', ndim, 'D multivariate normal window\n')
        print('sigma =', 0.3, ' mu =', 0.0)
        psf = pu.gaussian_window(image.shape, sigma=0.3, mu=0.0, debugging=True)
    psf = psf.astype(float)

    im_deconv = np.abs(richardson_lucy(image=image, psf=psf, iterations=iterations, clip=False))

    if debugging:
        gu.combined_plots(tuple_array=(image, im_deconv), tuple_sum_frames=False, tuple_colorbar=True,
                          tuple_scale='linear', tuple_width_v=None, tuple_width_h=None, tuple_vmin=0, tuple_vmax=1,
                          tuple_title=('Before RL', 'After'+str(iterations)+'iterations of RL'))

    return im_deconv


def psf_rl(measured_intensity, coherent_intensity, iterations=20, debugging=False):
    """
    Partial coherence deconvolution using Richardson-Lucy algorithm.

    :param measured_intensity: measured object with partial coherent illumination
    :param coherent_intensity: estimate of the object measured by a fully coherent illumination
    :param iterations: number of iterations for the Richardson-Lucy algorithm
    :param debugging: True to see plots
    :return:
    """
    psf = np.abs(richardson_lucy(image=measured_intensity, psf=coherent_intensity, iterations=iterations, clip=False))
    psf = (psf / psf.sum()).astype(np.float32)
    if debugging:
        gu.multislices_plot(fftshift(psf), scale='log', sum_frames=False, title='log(psf) in detector frame',
                            reciprocal_space=True, vmin=-5, is_orthogonal=False, plot_colorbar=True)
    return psf


# def er:
#     return
#
#
# def hio:
#     return
#
#
# def hio_or:
#     return
# def raar:
#     return

