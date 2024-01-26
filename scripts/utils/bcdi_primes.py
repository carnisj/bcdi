#!/usr/bin/env python3

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr


import bcdi.utils.utilities as util

helptext = """
Check smaller or higher prime of a number, in order to determine the correct FFT
window size for phase retrieval. Adapted from PyNX.
"""
my_nb = 368


if __name__ == "__main__":
    nb_low = util.smaller_primes(my_nb, maxprime=7, required_dividers=(2,))
    nb_high = util.higher_primes(my_nb, maxprime=7, required_dividers=(2,))
    print("Smaller prime=", str(nb_low), "  Higher prime=", str(nb_high))
