# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr
"""The main bcdi package, which contains the whole framework."""
__version__ = "0.2.7"

from subprocess import PIPE, Popen


def get_git_version() -> str:
    """
    Get the full version name with git hash, e.g. "2020.1-65-g958b7254-dirty".

    Only works if the current directory is part of the git repository.

    :return: the version name
    """
    try:
        p = Popen(
            ["git", "describe", "--tags", "--dirty", "--always"],
            stdout=PIPE,
            stderr=PIPE,
        )
        if p.stdout is not None:
            return p.stdout.readlines()[0].strip().decode("UTF-8")
        raise IndexError
    except IndexError:
        # in distributed & installed versions this is replaced by a string
        __git_version_static__ = "git_version_placeholder"
        if "placeholder" in __git_version_static__:
            return __version__
        return __git_version_static__
