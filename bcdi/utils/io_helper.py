# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

"""Module containing decorators and context manager classes for input-output."""

from functools import wraps
from typing import Callable, Optional, Union


class ContextFile:
    """
    Convenience context manager to open files.

    The supported opening callables are silx.io.specfile.Specfile, io.open, h5py.File
    and SIXS (nxsReady.Dataset, ReadNxs3.Dataset).
    """

    def __init__(
        self,
        filename: str,
        open_func: Union[type, Callable],
        scan_number: Optional[int] = None,
        mode: str = "r",
        encoding: str = "utf-8",
        longname: Optional[str] = None,
        shortname: Optional[str] = None,
        directory: Optional[str] = None,
    ):
        self.filename = filename
        self.file = None
        self.open_func = open_func
        self.scan_number = scan_number
        self.mode = mode
        self.encoding = encoding
        self.longname = longname
        self.shortname = shortname
        self.directory = directory

    def __enter__(self):
        if (
            self.open_func.__module__ == "silx.io.specfile"
            and self.open_func.__name__ == "SpecFile"
        ):
            self.file = self.open_func(self.filename)
        elif self.open_func.__module__ == "io" and self.open_func.__name__ == "open":
            self.file = self.open_func(
                self.filename, mode=self.mode, encoding=self.encoding
            )
        elif (
            self.open_func.__module__ == "h5py._hl.files"
            and self.open_func.__name__ == "File"
        ):
            self.file = self.open_func(self.filename, mode=self.mode)
        elif (
            self.open_func.__module__ == "nxsReady"
            and self.open_func.__name__ == "DataSet"
        ):
            self.file = self.open_func(
                longname=self.longname,
                shortname=self.shortname,
                alias_dict=self.filename,
                scan="SBS",
            )
        elif (
            self.open_func.__module__ == "ReadNxs3"
            and self.open_func.__name__ == "DataSet"
        ):
            self.file = self.open_func(
                directory=self.directory,
                filename=self.shortname,
                alias_dict=self.filename,
            )
        else:
            raise NotImplementedError(f"open function {self.open_func} not supported")
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
        return False


def safeload(func: Callable) -> Callable:
    """
    Decorator for safely opening files within class methods.

    :param func: a class method accessing the file
    """
    if not isinstance(func, Callable):
        raise ValueError("func should be a callable")

    @wraps(func)
    def helper(self, *args, **kwargs):
        setup = kwargs.get("setup")
        if setup is None:
            raise ValueError
        if not isinstance(setup.logfile, ContextFile):
            raise TypeError(
                "setup.logfile should be a ContextFile, " f"got {type(setup.logfile)}"
            )
        with setup.logfile as file:
            return func(self, *args, file=file, **kwargs)

    return helper


def safeload_static(func: Callable) -> Callable:
    """
    Decorator for safely opening files within class static methods.

    :param func: a class static method accessing the file
    """
    if not isinstance(func, Callable):
        raise ValueError("func should be a callable")

    @wraps(func)
    def helper(*args, **kwargs):
        setup = kwargs.get("setup")
        if setup is None:
            raise ValueError
        if not isinstance(setup.logfile, ContextFile):
            raise TypeError(
                "setup.logfile should be a ContextFile, " f"got {type(setup.logfile)}"
            )
        with setup.logfile as file:
            return func(*args, file=file, **kwargs)

    return helper
