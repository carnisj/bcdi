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

    if debugging:
        image = abs(image) / abs(image) .max()
        im_deconv = abs(im_deconv) / abs(im_deconv) .max()
        gu.combined_plots(tuple_array=(image, im_deconv), tuple_sum_frames=False, tuple_colorbar=True,
                          tuple_scale='linear', tuple_width_v=np.nan, tuple_width_h=np.nan, tuple_vmin=0, tuple_vmax=1,
                          tuple_title=('Before RL', 'After '+str(iterations)+' iterations of RL (normalized)'))

    return im_deconv


def psf_rl(measured_intensity, coherent_intensity, iterations=20, debugging=False):
    """
    Partial coherence deconvolution using Richardson-Lucy algorithm. See J.N. Clark et al., Nat. Comm. 3, 993 (2012).

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

if __name__ == "__main__":
    import h5py
    import matplotlib.pyplot as plt

    datadir = 'D:/data/P10_August2019/data/gold_2_2_2_00022/pynx/1000_1000_1000_1_1_1/maximum_likelihood/'
    filename = 'modes_ml.h5'
    h5file = h5py.File(datadir+filename, 'r')
    group_key = list(h5file.keys())[0]
    subgroup_key = list(h5file[group_key])
    dataset = h5file['/' + group_key + '/' + subgroup_key[0] + '/data'][0]  # select only first mode
    dataset = abs(dataset) / abs(dataset).max()
    # my_psf = pu.tukey_window((10, 10, 10), alpha=(0.6, 0.6, 0.6))
    output = deconvolution_rl(dataset, psf=None, iterations=10, debugging=True)
    # psf = pu.gaussian_window(window_shape=(3, 3, 3), sigma=0.7, mu=0.0, debugging=False)
    # output = deconvolution_rl(output, psf=psf, iterations=20, debugging=True)
    plt.show()
