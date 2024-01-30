# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

"""Module containing decorators and context manager classes for input-output."""

import logging
import os
from functools import wraps
from typing import Callable, Optional, Union

import bcdi.utils.format as fmt
import bcdi.utils.validation as valid

module_logger = logging.getLogger(__name__)


class ContextFile:
    """
    Convenience context manager to open files.

    The supported opening callables are silx.io.specfile.Specfile, io.open, h5py.File
    and SIXS (nxsReady.Dataset, ReadNxs3.Dataset).

    :param kwargs:

     - 'logger': an optional logger

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
        **kwargs,
    ):
        self.logger = kwargs.get("logger", module_logger)
        self.filename = filename
        self.file = None
        self.open_func = open_func
        self.scan_number = scan_number
        self.mode = mode
        self.encoding = encoding
        self.longname = longname
        self.shortname = shortname
        self.directory = directory

    @property
    def directory(self):
        return self._directory

    @directory.setter
    def directory(self, value):
        if value is not None and not isinstance(value, str):
            raise TypeError("directory should be a str")
        self._directory = value

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, value):
        if not isinstance(value, str):
            raise TypeError("filename should be a str")
        if not os.path.isfile(value):
            raise ValueError(f"Could not find the file at: {value}")
        self._filename = value

    @property
    def longname(self):
        return self._longname

    @longname.setter
    def longname(self, value):
        if value is not None and not isinstance(value, str):
            raise TypeError("longname should be a str")
        self._longname = value

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        if not isinstance(value, str):
            raise TypeError("mode should be a str")
        self._mode = value

    @property
    def open_func(self):
        return self._open_func

    @open_func.setter
    def open_func(self, value):
        if not isinstance(value, type) and not callable(value):
            raise TypeError("open_func should be a class or a function")
        self._open_func = value

    @property
    def scan_number(self):
        return self._scan_number

    @scan_number.setter
    def scan_number(self, value):
        valid.valid_item(
            value,
            allowed_types=int,
            min_included=1,
            allow_none=True,
            name="scan_number",
        )
        self._scan_number = value

    @property
    def shortname(self):
        return self._shortname

    @shortname.setter
    def shortname(self, value):
        if value is not None and not isinstance(value, str):
            raise TypeError("shortname should be a str")
        self._shortname = value

    def __enter__(self):
        """
        Enter the context manager.

        This method returns a handle to the opened file.
        """
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
            "nxsReady" in self.open_func.__module__
            and self.open_func.__name__ == "DataSet"
        ):
            self.file = self.open_func(
                longname=self.longname,
                shortname=self.shortname,
                alias_dict=self.filename,
                scan="SBS",
            )
        elif (
            "ReadNxs3" in self.open_func.__module__
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
        """
        Exit the context manager.

        The open_func needs to implement a method 'close'.
        """
        try:
            self.file.close()
        except AttributeError:
            raise NotImplementedError(
                "couldn't close the file, 'close' is not implemented"
            )
        return False

    def __repr__(self):
        """Representation string of the ContextFile instance."""
        return fmt.create_repr(obj=self, cls=ContextFile)


def safeload(func: Callable) -> Callable:
    """
    Decorate a class method to safely opening files.

    :param func: a class method accessing the file
    """
    if not callable(func):
        raise ValueError("func should be a callable")

    @wraps(func)
    def helper(self, *args, **kwargs):
        setup = kwargs.get("setup")
        if setup is None:
            raise ValueError("setup undefined")
        if not isinstance(setup.logfile, ContextFile):
            raise TypeError(
                f"setup.logfile should be a ContextFile, got {type(setup.logfile)}"
            )
        with setup.logfile as file:
            return func(self, *args, file=file, **kwargs)

    return helper


def safeload_static(func: Callable) -> Callable:
    """
    Decorate a class static method or a function to safely opening files.

    :param func: a class static method accessing the file
    """
    if not callable(func):
        raise ValueError("func should be a callable")

    @wraps(func)
    def helper(*args, **kwargs):
        setup = kwargs.get("setup")
        if setup is None:
            raise ValueError("setup undefined")
        if not isinstance(setup.logfile, ContextFile):
            raise TypeError(
                f"setup.logfile should be a ContextFile, got {type(setup.logfile)}"
            )
        with setup.logfile as file:
            return func(*args, file=file, **kwargs)

    return helper
