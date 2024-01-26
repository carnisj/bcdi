# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr
"""
Functions related to image deconvolution.

Richardson-Lucy algorithm, blind deconvolution...
"""

import sys
from numbers import Real

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve, fftconvolve

from ..graph import graph_utils as gu
from ..utils import utilities as util
from ..utils import validation as valid


def blind_deconvolution_rl(
    blurred_object,
    perfect_object,
    psf,
    nb_cycles=10,
    sub_iterations=10,
    update_psf_first=True,
    debugging=False,
    **kwargs,
):
    """
    Blind deconvolution using Richardson-Lucy algorithm.

    Estimates of the perfect object and the psf have to be provided. See Figure 1 and
    equations (4) & (5) in  D. A. Fish et al. J. Opt. Soc. Am. A, 12, 58 (1995).

    :param blurred_object: ndarray, measured object with partial coherent illumination
    :param perfect_object: ndarray, estimate of the object measured by a fully coherent
     illumination, same shape as blurred_object
    :param psf: ndarray, estimate of the psf, same shape as blurred_object
    :param nb_cycles: number of blind deconvolution interations
    :param sub_iterations: number of iterations of the Richardson-Lucy algorithm during
     a single blind iteration
    :param update_psf_first: bool, if True the psf estimate is updated first and then
     the perfect object estimate
    :param debugging: True to see plots
    :param kwargs:
     - 'scale': tuple, scale for the plots, 'linear' or 'log'
     - 'reciprocal_space': bool, True if the data is in reciprocal space,
       False otherwise.
     - 'is_orthogonal': bool, True is the frame is orthogonal, False otherwise
       (detector frame) Used for plot labels.
     - 'vmin' = tuple of two floats (np.nan to use default), lower boundary for the
       colorbars
     - 'vmax' = tuple of two floats (np.nan to use default), higher boundary for the
       colorbars

    :return: the psf
    """
    validation_name = "algorithms_utils.psf_rl"
    # check and load kwargs
    valid.valid_kwargs(
        kwargs=kwargs,
        allowed_kwargs={"scale", "reciprocal_space", "is_orthogonal", "vmin", "vmax"},
        name=validation_name,
    )
    scale = kwargs.get("scale", ("linear", "log"))
    valid.valid_container(
        scale, container_types=(tuple, list), length=2, name=validation_name
    )
    if not all(val in {"log", "linear"} for val in scale):
        raise ValueError('"scale" should be either "log" or "linear"')
    reciprocal_space = kwargs.get("reciprocal_space", True)
    if not isinstance(reciprocal_space, bool):
        raise TypeError('"reciprocal_space" should be a boolean')
    is_orthogonal = kwargs.get("is_orthogonal", True)
    if not isinstance(is_orthogonal, bool):
        raise TypeError('"is_orthogonal" should be a boolean')
    vmin = kwargs.get("vmin", (np.nan, np.nan))
    valid.valid_container(
        vmin, container_types=(tuple, list), item_types=Real, name=validation_name
    )
    vmax = kwargs.get("vmax", (np.nan, np.nan))
    valid.valid_container(
        vmax, container_types=(tuple, list), item_types=Real, name=validation_name
    )

    # check parameters
    valid.valid_ndarray((blurred_object, perfect_object, psf))
    if not isinstance(debugging, bool):
        raise TypeError('"debugging" should be a boolean')
    if not isinstance(update_psf_first, bool):
        raise TypeError('"update_psf_first" should be a boolean')

    ########################
    # plot initial guesses #
    ########################
    if debugging:
        gu.multislices_plot(
            perfect_object,
            scale=scale[0],
            sum_frames=False,
            title="guessed perfect object",
            reciprocal_space=reciprocal_space,
            is_orthogonal=is_orthogonal,
            vmin=vmin[0],
            vmax=vmax[0],
            plot_colorbar=True,
        )

        gu.multislices_plot(
            psf,
            scale=scale[1],
            sum_frames=False,
            title="guessed psf",
            vmin=vmin[1],
            vmax=vmax[1],
            reciprocal_space=reciprocal_space,
            is_orthogonal=is_orthogonal,
            plot_colorbar=True,
        )

    ###########################################
    # loop over the blind deconvolution steps #
    ###########################################
    for _ in range(nb_cycles):
        if update_psf_first:
            # update the estimate of the psf
            psf, _ = richardson_lucy(
                image=blurred_object,
                psf=perfect_object,
                iterations=sub_iterations,
                clip=False,
                guess=psf,
            )
            # udpate the estimate of the perfect object
            perfect_object, _ = richardson_lucy(
                image=blurred_object,
                psf=psf,
                iterations=sub_iterations,
                clip=True,
                guess=perfect_object,
            )
        else:
            # udpate the estimate of the perfect object
            perfect_object, _ = richardson_lucy(
                image=blurred_object,
                psf=psf,
                iterations=sub_iterations,
                clip=True,
                guess=perfect_object,
            )
            # update the estimate of the psf
            psf, _ = richardson_lucy(
                image=blurred_object,
                psf=perfect_object,
                iterations=sub_iterations,
                clip=False,
                guess=psf,
            )
    psf = (np.abs(psf) / np.abs(psf).sum()).astype(np.float)

    ###############
    # plot result #
    ###############
    if debugging:
        gu.multislices_plot(
            perfect_object,
            scale=scale[0],
            sum_frames=False,
            title="retrieved perfect object",
            reciprocal_space=reciprocal_space,
            is_orthogonal=is_orthogonal,
            vmin=vmin[0],
            vmax=vmax[0],
            plot_colorbar=True,
        )

        gu.multislices_plot(
            psf,
            scale=scale[1],
            sum_frames=False,
            title="retrieved psf",
            vmin=vmin[1],
            vmax=vmax[1],
            reciprocal_space=reciprocal_space,
            is_orthogonal=is_orthogonal,
            plot_colorbar=True,
        )
    return psf


def deconvolution_rl(
    image, psf=None, psf_shape=(10, 10, 10), iterations=20, debugging=False
):
    """
    Image deconvolution using Richardson-Lucy algorithm.

    The algorithm is based on a PSF (Point Spread Function), where PSF is described
    as the impulse response of the optical system.

    :param image: image to be deconvoluted
    :param psf: ndarray, psf if known. Leave None to use a Gaussian kernel of shape
     psf_shape.
    :param psf_shape: shape of the kernel used for deconvolution
    :param iterations: number of iterations for the Richardson-Lucy algorithm
    :param debugging: True to see plots
    :return: the deconvoluted image
    """
    image = image.astype(np.float)
    max_img = image.max(initial=None)
    min_img = image[np.nonzero(image)].min(initial=None)
    image = image / min_img  # the new min is 1, to avoid dividing by values close to 0

    ndim = image.ndim
    if psf is None:
        print("Initializing the psf using a", ndim, "D multivariate normal window\n")
        print("sigma =", 0.3, " mu =", 0.0)
        psf = util.gaussian_window(
            window_shape=psf_shape, sigma=0.3, mu=0.0, debugging=False
        )
    psf = psf.astype(float)
    if debugging:
        gu.multislices_plot(
            array=psf,
            sum_frames=False,
            plot_colorbar=True,
            scale="linear",
            title="Gaussian window",
            reciprocal_space=False,
            is_orthogonal=True,
        )

    im_deconv, _ = np.abs(
        richardson_lucy(image=image, psf=psf, iterations=iterations, clip=False)
    )
    im_deconv = (
        abs(im_deconv) / abs(im_deconv).max(initial=None) * max_img
    )  # normalize back to max_img

    if debugging:
        image = abs(image) / abs(image).max()
        im_deconv = abs(im_deconv) / abs(im_deconv).max()
        gu.combined_plots(
            tuple_array=(image, im_deconv),
            tuple_sum_frames=False,
            tuple_colorbar=True,
            tuple_title=(
                "Before RL",
                "After " + str(iterations) + " iterations of RL (normalized)",
            ),
            tuple_scale="linear",
            tuple_vmin=0,
            tuple_vmax=1,
        )
    return im_deconv


def partial_coherence_rl(
    measured_intensity, coherent_intensity, iterations=20, debugging=False, **kwargs
):
    """
    Partial coherence deconvolution using Richardson-Lucy algorithm.

    See J.N. Clark et al., Nat. Comm. 3, 993 (2012).

    :param measured_intensity: measured object with partial coherent illumination
    :param coherent_intensity: estimate of the object measured by a fully coherent
     illumination
    :param iterations: number of iterations for the Richardson-Lucy algorithm
    :param debugging: True to see plots
    :param kwargs:
     - 'scale': scale for the plot, 'linear' or 'log'
     - 'reciprocal_space': True if the data is in reciprocal space, False otherwise.
     - 'is_orthogonal': set to True is the frame is orthogonal, False otherwise
       (detector frame) Used for plot labels.
     - 'vmin' = lower boundary for the colorbar. Float or tuple of 3 floats
     - 'vmax' = [higher boundary for the colorbar. Float or tuple of 3 floats
     - 'guess': ndarray, initial guess for the psf, of the same shape as
       measured_intensity

    :return: the retrieved psf (ndarray), the error metric
     (1D ndarray of len=iterations)
    """
    validation_name = "algorithms_utils.psf_rl"
    # check and load kwargs
    valid.valid_kwargs(
        kwargs=kwargs,
        allowed_kwargs={
            "scale",
            "reciprocal_space",
            "is_orthogonal",
            "vmin",
            "vmax",
            "guess",
        },
        name=validation_name,
    )
    scale = kwargs.get("scale", "log")
    if scale not in {"log", "linear"}:
        raise ValueError('"scale" should be either "log" or "linear"')
    reciprocal_space = kwargs.get("reciprocal_space", True)
    if not isinstance(reciprocal_space, bool):
        raise TypeError('"reciprocal_space" should be a boolean')
    is_orthogonal = kwargs.get("is_orthogonal", True)
    if not isinstance(is_orthogonal, bool):
        raise TypeError('"is_orthogonal" should be a boolean')
    vmin = kwargs.get("vmin", np.nan)
    valid.valid_item(vmin, allowed_types=Real, name=validation_name)
    vmax = kwargs.get("vmax", np.nan)
    valid.valid_item(vmax, allowed_types=Real, name=validation_name)
    valid.valid_ndarray((measured_intensity, coherent_intensity))
    guess = kwargs.get("guess")
    if guess is not None:
        valid.valid_ndarray(guess, shape=measured_intensity.shape)

    # calculate the psf
    psf, error = richardson_lucy(
        image=measured_intensity,
        psf=coherent_intensity,
        iterations=iterations,
        clip=False,
        guess=guess,
    )

    # optional plot
    if debugging:
        gu.multislices_plot(
            psf,
            scale=scale,
            sum_frames=False,
            title="psf",
            vmin=vmin,
            vmax=vmax,
            reciprocal_space=reciprocal_space,
            is_orthogonal=is_orthogonal,
            plot_colorbar=True,
        )
        _, ax = plt.subplots(figsize=(12, 9))
        ax.plot(error, "r.")
        ax.set_yscale("log")
        ax.set_xlabel("iteration number")
        ax.set_ylabel("difference between consecutive iterates")
    return psf, error


def richardson_lucy(image, psf, iterations=50, clip=True, guess=None):
    """
    Richardson-Lucy algorithm.

    The algorithm is as implemented in scikit-image.restoration.deconvolution with an
    additional parameter for the initial guess of the psf.

    :param image: ndarray, input degraded image (can be N dimensional).
    :param psf: ndarray, the point spread function.
    :param iterations: int, number of iterations. This parameter plays the role of
     regularisation.
    :param clip: boolean. If true, pixel values of the result above 1 or under -1 are
     thresholded for skimage pipeline compatibility.
    :param guess: ndarray, the initial guess for the deconvoluted image.
     Leave None to use the default (flat array of 0.5)
    :return: the deconvolved image (ndarray) and the error metric (1D ndarray,
     len = iterations). The error is given by
     np.linalg.norm(previous_deconv-new_deconv) / np.linalg.norm(previous_deconv)
    """
    # compute the times for direct convolution and the fft method. The fft is of
    # complexity O(N log(N)) for each dimension and the direct method does
    # straight arithmetic (and is O(n*k) to add n elements k times)
    direct_time = np.prod(image.shape + psf.shape)
    fft_time = np.sum([n * np.log(n) for n in image.shape + psf.shape])

    # see whether the fourier transform convolution method or the direct
    # convolution method is faster (discussed in scikit-image PR #1792)
    time_ratio = 40.032 * fft_time / direct_time

    if time_ratio <= 1 or len(image.shape) > 2:
        convolve_method = fftconvolve
    else:
        convolve_method = convolve

    valid.valid_ndarray((image, psf))
    image = image.astype(np.float)
    psf = psf.astype(np.float)

    if guess is not None:
        valid.valid_ndarray(guess, shape=image.shape)
        im_deconv = guess
    else:
        im_deconv = np.full(image.shape, 0.5)

    psf_mirror = psf[::-1, ::-1]

    error = np.empty(iterations)
    for idx in range(iterations):
        if (idx % 10) == 0:
            sys.stdout.write(f"\rRL iteration {idx}")
            sys.stdout.flush()
        previous_deconv = np.copy(im_deconv)
        relative_blur = image / convolve_method(im_deconv, psf, "same")
        im_deconv *= convolve_method(relative_blur, psf_mirror, "same")
        error[idx] = np.linalg.norm(previous_deconv - im_deconv) / np.linalg.norm(
            previous_deconv
        )
    print("\n")
    if clip:
        im_deconv[im_deconv > 1] = 1
        im_deconv[im_deconv < -1] = -1

    return im_deconv, error
