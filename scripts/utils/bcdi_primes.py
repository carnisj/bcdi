#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np

helptext = """
Check smaller or higher prime of a number, in order to determine the correct FFT window size for phase retrieval.
Adapted from PyNX
"""
# 428 * 294 * 599
my_nb = 368


def primes(n):
    """ Returns the prime decomposition of n as a list
    """
    v = [1]
    assert n > 0
    i = 2
    while i * i <= n:
        while n % i == 0:
            v.append(i)
            n //= i
        i += 1
    if n > 1:
        v.append(n)
    return v


def try_smaller_primes(n, maxprime=13, required_dividers=(4,)):
    """
    Check if the largest prime divider is <=maxprime, and optionally includes some dividers.

    Args:
        n: the integer number for which the prime decomposition will be checked
        maxprime: the maximum acceptable prime number. This defaults to the largest integer accepted by the clFFT
        library for OpenCL GPU FFT.
        required_dividers: list of required dividers in the prime decomposition. If None, this check is skipped.
    Returns:
        True if the conditions are met.
    """
    p = primes(n)
    if max(p) > maxprime:
        return False
    if required_dividers is not None:
        for k in required_dividers:
            if n % k != 0:
                return False
    return True


def smaller_primes(n, maxprime=13, required_dividers=(4,)):
    """ Find the closest integer <=n (or list/array of integers), for which the largest prime divider is <=maxprime,
    and has to include some dividers.
    The default values for maxprime is the largest integer accepted by the clFFT library for OpenCL GPU FFT.

    Args:
        n: the integer number
        maxprime: the largest prime factor acceptable
        required_dividers: a list of required dividers for the returned integer.
    Returns:
        the integer (or list/array of integers) fulfilling the requirements
    """
    if (type(n) is list) or (type(n) is tuple) or (type(n) is np.ndarray):
        vn = []
        for i in n:
            assert i > 1 and maxprime <= i
            while (
                try_smaller_primes(
                    i, maxprime=maxprime, required_dividers=required_dividers
                )
                is False
            ):
                i = i - 1
                if i == 0:
                    return 0
            vn.append(i)
        if type(n) is np.ndarray:
            return np.array(vn)
        return vn
    assert n > 1 and maxprime <= n
    while (
        try_smaller_primes(n, maxprime=maxprime, required_dividers=required_dividers)
        is False
    ):
        n = n - 1
        if n == 0:
            return 0
    return n


def higher_primes(number, maxprime=13, required_dividers=(4,)):
    """
    Find the closest integer >=n (or list/array of integers), for which the largest prime divider is <=maxprime,
    and has to include some dividers. The default values for maxprime is the largest integer accepted
    by the clFFT library for OpenCL GPU FFT. Adapted from PyNX.

    :param number: the integer number
    :param maxprime: the largest prime factor acceptable
    :param required_dividers: a list of required dividers for the returned integer.
    :return: the integer (or list/array of integers) fulfilling the requirements
    """
    if isinstance(number, (list, tuple, np.ndarray)):
        vn = []
        for i in number:
            limit = i
            assert i > 1 and maxprime <= i
            while (
                try_smaller_primes(
                    i, maxprime=maxprime, required_dividers=required_dividers
                )
                is False
            ):
                i = i + 1
                if i == limit:
                    return limit
            vn.append(i)
        if isinstance(number, np.ndarray):
            return np.array(vn)
        return vn
    limit = number
    assert number > 1 and maxprime <= number
    while (
        try_smaller_primes(
            number, maxprime=maxprime, required_dividers=required_dividers
        )
        is False
    ):
        number = number + 1
        if number == limit:
            return limit
    return number


nb_low = smaller_primes(my_nb, maxprime=7, required_dividers=(2,))
nb_high = higher_primes(my_nb, maxprime=7, required_dividers=(2,))
print("Smaller prime=", str(nb_low), "  Higher prime=", str(nb_high))
