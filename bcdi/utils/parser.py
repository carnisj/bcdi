# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Clément Atlan, c.atlan@outlook.com
#         Jerome Carnis, carnis_jerome@yahoo.fr

from argparse import ArgumentParser
import pathlib
from typing import Any, ByteString, Dict, Union
import yaml

from bcdi.utils.parameters import valid_param
import bcdi.utils.validation as valid


def add_cli_parameters(argument_parser: ArgumentParser) -> ArgumentParser:
    """
    Add generic parameters to the argument parser.

    :param argument_parser: an instance of argparse.ArgumentParser
    :return: the updated instance
    """
    argument_parser.add_argument(
        "--align_q",
        default="False",
        type=str,
        help="If orthogonalized, do not align q",
    )

    argument_parser.add_argument(
        "--background_file", type=str, help="optional background_file"
    )

    argument_parser.add_argument(
        "--beamline",
        type=str,
        help="beamline where the measurement was made",
    )

    argument_parser.add_argument(
        "--binning",
        type=str,
        help="binning factor applied during phasing",
    )

    argument_parser.add_argument("--debug", type=str, help="debugging option")

    argument_parser.add_argument(
        "--detector", type=str, help="detector used during measurement"
    )

    argument_parser.add_argument("--energy", type=float, help="beam energy")

    argument_parser.add_argument("--median_filter", type=str, help="apply a filter")

    argument_parser.add_argument(
        "--fix_voxel",
        type=str,
        help="the voxel size used for interpolation",
    )

    argument_parser.add_argument(
        "--flip_reconstruction",
        type=str,
        help="choose to flip or not the reconstruction",
    )

    argument_parser.add_argument("--grazing_angle", type=float, help="incidence angle")

    argument_parser.add_argument(
        "--inplane-angle", type=float, help="detector in plane angle"
    )

    argument_parser.add_argument(
        "--interact",
        type=str,
        help="choose if interaction mode is enabled",
    )

    argument_parser.add_argument(
        "--isosurface_strain",
        type=float,
        help="the isosurface threshold used for postprocessing",
    )

    argument_parser.add_argument(
        "--orthogonalize", type=str, help="Whether to orthogonalize or not"
    )

    argument_parser.add_argument(
        "--outofplane-angle", type=float, help="detector out of plane angle"
    )

    argument_parser.add_argument(
        "--phasing_binning",
        type=str,
        help="binning factor applied during phasing",
    )

    argument_parser.add_argument(
        "--rocking-angle",
        type=str,
        choices=["inplane", "outofplane"],
        help="rocking angle",
    )

    argument_parser.add_argument(
        "--root-folder", type=str, help="where to find experiment data"
    )

    argument_parser.add_argument("--sample-name", type=str, help="name of the sample")

    argument_parser.add_argument(
        "--save-dir", type=str, help="directory path where to save"
    )

    argument_parser.add_argument(
        "--scan", type=int, help="number of the scan to process"
    )

    argument_parser.add_argument(
        "--sdd",
        type=float,
        help="sample to detector distance",
    )

    argument_parser.add_argument(
        "--specfile_name", required=True, type=str, help="path to '.spec' file"
    )

    argument_parser.add_argument(
        "--template_imagefile",
        type=str,
        help="template of the data image files",
    )

    argument_parser.add_argument(
        "--tilt_angle", type=float, help="angle step used during scan"
    )

    return argument_parser


class ConfigParser:
    """
    Base class for parsing arguments.

    Some validation is also realized in the class.

    :param file_path: path of the configuration file that contains
    the arguments, str.
    """

    def __init__(
        self, file_path: str, command_line_args: Union[Dict[str, Any], None] = None
    ) -> None:
        self.file_path = file_path
        self.command_line_args = command_line_args
        self.raw_config = self._open_file()
        self.arguments = None

    @property
    def command_line_args(self):
        """Optional dictionary of command line parameters."""
        return self._command_line_args

    @command_line_args.setter
    def command_line_args(self, value):
        valid.valid_container(
            value, container_types=dict, allow_none=True, name="command_line_args"
        )
        self._command_line_args = value

    @property
    def file_path(self):
        """Path of the configuration file."""
        return self._file_path

    @file_path.setter
    def file_path(self, value):
        valid.valid_container(
            value, container_types=str, min_length=1, name="file_path"
        )
        if pathlib.Path(value).suffix != ".yml":
            raise ValueError("Expecting a YAML config file")
        self._file_path = value

    @staticmethod
    def _check_args(dic: Dict[str, Any]) -> Dict[str, Any]:
        checked_keys = []
        for key, value in dic.items():
            value, is_valid = valid_param(key, value)
            if is_valid:
                dic[key] = value
                checked_keys.append(key)
            else:
                print(
                    f"'{key}' is an unexpected key, " "its value won't be considered."
                )
        return {key: dic[key] for key in checked_keys}

    def load_arguments(self) -> Dict:
        """Parse the byte string, eventually override defaults and check parameters."""
        args = yaml.load(self.raw_config, Loader=yaml.SafeLoader)
        if self.command_line_args is not None:
            args.update(self.command_line_args)
        self.arguments = self._check_args(args)
        return self.arguments

    def _open_file(self) -> ByteString:
        """Open the file and return it."""
        with open(self.file_path, "rb") as f:
            raw_config = f.read()
        return raw_config
