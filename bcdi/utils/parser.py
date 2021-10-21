# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Clément Atlan, c.atlan@outlook.com
#         Jerome Carnis, carnis_jerome@yahoo.fr

import yaml
import pathlib

from bcdi.utils.parameters import valid_param
import bcdi.utils.validation as valid


class ArgumentParser:
    """
    Base class for parsing arguments.

    Some validation is also realized in the class.

    :param file_path: path of the configuration file that contains
    the arguments, str.
    """

    def __init__(self, file_path : str) -> None :
        self.file_path = file_path
        self.raw_config = self._open_file()
        self.arguments = None

    @property
    def file_path(self):
        """Path of the configuration file."""
        return self._file_path

    @file_path.setter
    def file_path(self, value):
        valid.valid_container(
            value,
            container_types=str,
            min_length=1,
            name="file_path"
        )
        self._file_path = value

    def _open_file(self):
        """Open the file and return it."""
        with open(self.file_path, "rb") as f:
            raw_config = f.read()
        return raw_config

    def load_arguments(self):
        extension = self._get_extension()
        if extension == ".yml":
            args = yaml.load(self.raw_config, Loader=yaml.FullLoader)
            self.arguments = self._check_args(args)
            return self.arguments
        else:
            return None

    def _get_extension(self):
        """return the extension of the the file_path attribute"""
        return pathlib.Path(self.file_path).suffix

    @staticmethod
    def _check_args(dic):
        checked_keys = []
        for key, value in dic.items():
            if valid_param(key, value):
                checked_keys.append(key)
            else:
                print(f"'{key}' is an unexpected key, "
                      "its value won't be considered.")
        return {key: dic[key] for key in checked_keys}

    # For now the yaml Loader already returns a dic, so not useful
    # but we may need it if we use other file format
    def to_dict(self):
        pass

    def dump(self, output_path, extension):
        pass
