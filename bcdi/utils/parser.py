# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Clément Atlan, c.atlan@outlook.com

import yaml
import pathlib

_AUTHORIZED_KEYS = {
    'preprocessing': [
        'beamline',
        'binning',
        'rocking-angle',
        'data-root-folder',
        'debug',
        'detector',
        'filter',
        'interact',
        'mask',
        'orthogonalize',
        'output-dir',
        'sample-name',
        'save-dir',
        'sample_detector_distance',
        'scan',
        'specfile-path',
        'template-data-file'
    ],
    'postprocessing': [
        'angle-step',
        'beamline',
        'data-root-folderscan',
        'debug',
        'energy',
        'flip',
        'incidence-angle',
        'inplane-angle',
        'is-orthogonalized',
        'isosurface-threshold',
        'modes',
        'outofplane-angle',
        'rocking-angle',
        'rocking-angle',
        'sample-detector-distance',
        'save-dir',
        'specfile-path',
        'voxel-size'
    ]
}


class ArgumentParser:
    """
    Base class for parsing arguments.

    Some validation is also realized in the class.

    :param file_path: path of the configuration file that contains
    the arguments, str.
    :param script_type: the type of the script that the arguments will
    be parsed into, str.
    """
    def __init__(self, file_path : str, script_type : str = "preprocessing") -> None :
        self.file_path = file_path
        if script_type not in _AUTHORIZED_KEYS.keys():
            print("Please, provide a script_type from "
                  f"{_AUTHORIZED_KEYS.keys()}")
        else:
            self.script_type = script_type

        self.raw_config = self._open_file()
        self.arguments = None

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

    def _check_args(self, dic):
        checked_keys = []
        for key in dic.keys():
            if key not in _AUTHORIZED_KEYS[self.script_type]:
                print(f"'{key}' is an unexpected key, "
                      "its value won't be considered.")
            else:
                checked_keys.append(key)
        return {key: dic[key] for key in checked_keys}

    # For now the yaml Loader already returns a dic, so not useful
    # but we may need it if we use other file format
    def to_dict(self):
        pass

    def dump(self, output_path, extension):
        pass
