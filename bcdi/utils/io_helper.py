# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

"""Module containing decorators and context manager classes for input-output."""

from functools import wraps
from inspect import signature


class ContextFile:

    def __init__(self, filename, open_func, mode, encoding="utf-8"):
        self.filename = filename
        self.open_func = open_func
        self.mode = mode
        self.encoding = encoding

    def __enter__(self, open_func, mode, encoding="utf-8"):
        self.file = open_func(self.filename, mode=mode, encoding=encoding)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
        return False


def decorator(func):
    @wraps(func)
    def helper(*args, setup, **kwargs):
        if not isinstance(setup.logfile, ContextFile):
            raise TypeError("setup.logfile undefined")
        with setup.logfile as file:
            return func(*args, file=file, **kwargs)
    return helper
