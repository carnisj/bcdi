# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

from numbers import Real
import numpy as np
from numpy.fft import fftshift
from skimage.restoration.deconvolution import richardson_lucy
import sys
sys.path.append('D:/myscripts/bcdi/')
import bcdi.postprocessing.postprocessing_utils as pu
import bcdi.graph.graph_utils as gu
import bcdi.utils.validation as valid


def deconvolution_rl(image, psf=None, psf_shape=(10, 10, 10), iterations=20, debugging=False):
    """
    Image deconvolution using Richardson-Lucy algorithm. The algorithm is based on a PSF (Point Spread Function),
    where PSF is described as the impulse response of the optical system.

    :param image: image to be deconvoluted
    :param psf: psf to be used as a first guess
    :param psf_shape: shape of the kernel used for deconvolution
    :param iterations: number of iterations for the Richardson-Lucy algorithm
    :param debugging: True to see plots
    :return:
    """
    image = image.astype(np.float)
    max_img = image.max(initial=None)
    min_img = image[np.nonzero(image)].min(initial=None)
    image = image / min_img  # the new min is 1, to avoid dividing by values close to 0

    ndim = image.ndim
    if psf is None:
        print('Initializing the psf using a', ndim, 'D multivariate normal window\n')
        print('sigma =', 0.3, ' mu =', 0.0)
        psf = pu.gaussian_window(window_shape=psf_shape, sigma=0.3, mu=0.0, debugging=False)
    psf = psf.astype(float)
    if debugging:
        gu.multislices_plot(array=psf, sum_frames=False, plot_colorbar=True, scale='linear', title='Gaussian window',
                            reciprocal_space=False, is_orthogonal=True)

    im_deconv = np.abs(richardson_lucy(image=image, psf=psf, iterations=iterations, clip=False))
    im_deconv / im_deconv.max(initial=None) * max_img  # normalize back to max_img

    if debugging:
        image = abs(image) / abs(image).max()
        im_deconv = abs(im_deconv) / abs(im_deconv).max()
        gu.combined_plots(tuple_array=(image, im_deconv), tuple_sum_frames=False, tuple_colorbar=True,
                          tuple_title=('Before RL', 'After '+str(iterations)+' iterations of RL (normalized)'),
                          tuple_scale='linear', tuple_vmin=0, tuple_vmax=1)
    return im_deconv


def psf_rl(measured_intensity, coherent_intensity, iterations=20, debugging=False, **kwargs):
    """
    Partial coherence deconvolution using Richardson-Lucy algorithm. See J.N. Clark et al., Nat. Comm. 3, 993 (2012).

    :param measured_intensity: measured object with partial coherent illumination
    :param coherent_intensity: estimate of the object measured by a fully coherent illumination
    :param iterations: number of iterations for the Richardson-Lucy algorithm
    :param debugging: True to see plots
    :param kwargs:
     - 'scale': scale for the plot, 'linear' or 'log'
     - 'reciprocal_space': True if the data is in reciprocal space, False otherwise.
     - 'is_orthogonal': set to True is the frame is orthogonal, False otherwise (detector frame) Used for plot labels.
     - 'vmin' = user defined output array size [nbz, nby, nbx]
     - 'vmax' = [qx, qz, qy], each component being a 1D array
    :return:
    """
    validation_name = 'algorithms_utils.psf_rl'
    # check and load kwargs
    valid.valid_kwargs(kwargs=kwargs, allowed_kwargs={'scale', 'reciprocal_space', 'is_orthogonal', 'vmin', 'vmax'},
                       name=validation_name)
    scale = kwargs.get('scale', 'log')
    if scale not in {'log', 'linear'}:
        raise ValueError('"scale" should be either "log" or "linear"')
    reciprocal_space = kwargs.get('reciprocal_space', True)
    if not isinstance(reciprocal_space, bool):
        raise TypeError('"reciprocal_space" should be a boolean')
    is_orthogonal = kwargs.get('is_orthogonal', True)
    if not isinstance(is_orthogonal, bool):
        raise TypeError('"is_orthogonal" should be a boolean')
    vmin = kwargs.get('vmin', np.nan)
    valid.valid_item(vmin, allowed_types=Real, name=validation_name)
    vmax = kwargs.get('vmax', np.nan)
    valid.valid_item(vmax, allowed_types=Real, name=validation_name)

    # calculate the psf
    psf = np.abs(richardson_lucy(image=measured_intensity, psf=coherent_intensity, iterations=iterations, clip=False))
    psf = (psf / psf.sum()).astype(np.float32)

    # debugging plot
    if debugging:
        if scale == 'linear':
            title = 'psf in detector frame'
        else:  # 'log'
            title = 'log(psf) in detector frame'

        gu.multislices_plot(fftshift(psf), scale=scale, sum_frames=False, title=title, vmin=vmin, vmax=vmax,
                            reciprocal_space=reciprocal_space, is_orthogonal=is_orthogonal,
                            plot_colorbar=True)
    return psf


# if __name__ == "__main__":
