# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Clément Atlan, c.atlan@outlook.com
#         Jerome Carnis, carnis_jerome@yahoo.fr

"""Parsing of command-line arguments and config files."""

import logging
import os
import pathlib
from argparse import ArgumentParser
from functools import partial
from typing import Any, Dict, List, Optional, Type

import yaml

import bcdi.utils.format as fmt
import bcdi.utils.validation as valid
from bcdi.utils.parameters import valid_param

logger = logging.getLogger(__name__)


def add_cli_parameters(argument_parser: ArgumentParser) -> ArgumentParser:
    """
    Add generic parameters to the argument parser.

    :param argument_parser: an instance of argparse.ArgumentParser
    :return: the updated instance
    """
    argument_parser.add_argument(
        "-f",
        type=str,
        help="don't use it, kernel json file in Jupyter notebooks",
    )
    argument_parser.add_argument(
        "--align_q",
        type=str,
        help="If orthogonalized, do not align q",
    )

    argument_parser.add_argument(
        "-bckg", "--background_file", type=str, help="optional background_file"
    )

    argument_parser.add_argument(
        "-bl",
        "--beamline",
        type=str,
        help="beamline where the measurement was made",
    )

    argument_parser.add_argument(
        "-bin",
        "--phasing_binning",
        type=str,
        help="binning factor applied during phasing",
    )

    argument_parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        help="path to the config file",
    )

    argument_parser.add_argument("--debug", type=str, help="debugging option")

    argument_parser.add_argument(
        "-det", "--detector", type=str, help="detector used during measurement"
    )

    argument_parser.add_argument("-en", "--energy", type=float, help="beam energy")

    argument_parser.add_argument(
        "-med", "--median_filter", type=str, help="apply a filtering method"
    )

    argument_parser.add_argument(
        "--fix_voxel",
        type=str,
        help="the voxel size used for interpolation",
    )
    argument_parser.add_argument(
        "-flip",
        "--flip_reconstruction",
        type=str,
        help="choose to flip or not the reconstruction",
    )

    argument_parser.add_argument("--grazing_angle", type=float, help="incidence angle")

    argument_parser.add_argument(
        "--inplane_angle", type=float, help="detector in-plane angle"
    )

    argument_parser.add_argument(
        "--interact",
        type=str,
        help="choose if interaction mode is enabled",
    )

    argument_parser.add_argument(
        "-iso",
        "--isosurface_strain",
        type=float,
        help="the isosurface threshold used for postprocessing",
    )

    argument_parser.add_argument(
        "--outofplane_angle", type=float, help="detector out-of-plane angle"
    )

    argument_parser.add_argument(
        "--rocking_angle",
        type=str,
        choices=["inplane", "outofplane"],
        help="rocking angle",
    )

    argument_parser.add_argument(
        "--root_folder",
        type=str,
        help="path to the directory where all scans are stored",
    )

    argument_parser.add_argument("--sample_name", type=str, help="name of the sample")

    argument_parser.add_argument(
        "--save_dir", type=str, help="directory path where to save"
    )

    argument_parser.add_argument(
        "--scans",
        type=partial(str_to_list, item_type=int),
        help="comma-separated list of scans, e.g. '11,12,13'",
    )

    argument_parser.add_argument(
        "--detector_distance",
        type=float,
        help="sample to detector distance",
    )

    argument_parser.add_argument(
        "--specfile_name",
        type=partial(str_to_list, item_type=str),
        help="comma-separated list of paths to the log (e.g. .spec or .fio) file",
    )

    argument_parser.add_argument(
        "--template_imagefile",
        type=partial(str_to_list, item_type=str),
        help="comma-separated list of templates for the data image files",
    )

    argument_parser.add_argument(
        "--tilt_angle", type=float, help="angle step used during scan"
    )

    return argument_parser


def str_to_list(string: str, item_type: Type) -> List:
    """
    Convert a comma-separated string to a list.

    :param string: the string to convert, e.g. "11,12,13" or "file1.spec,file2.spec"
    :param item_type: type to which the elements should be cast
    :return: a list of converted values
    """
    elements = string.split(",")
    for idx, val in enumerate(elements):
        try:
            elements[idx] = item_type(val)
        except ValueError:
            raise ValueError(f"can't cast {val} from {str} to {item_type}")
    return elements


class ConfigParser:
    """
    Base class for parsing arguments.

    If provided, command line arguments will override the parameters from the config
    file. Some validation is also realized in the class.

    :param file_path: path of the configuration file that contains
     the arguments, str.
    :param command_line_args: an optional dictionary of command-line arguments
    """

    def __init__(
        self, file_path: str, command_line_args: Optional[Dict[str, Any]] = None
    ) -> None:
        self.file_path = file_path
        self.command_line_args = command_line_args
        self.raw_config = self._open_file()
        self.arguments: Optional[Dict] = None

    @property
    def command_line_args(self):
        """Return an optional dictionary of command line parameters."""
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
        if not os.path.isfile(value):
            raise ValueError(f"The config file '{value}' does not exist")
        self._file_path = value

    @staticmethod
    def _check_args(dic: Dict[str, Any]) -> Dict[str, Any]:
        """Apply some validation on each parameter."""
        checked_keys = []
        for key, value in dic.items():
            value, is_valid = valid_param(key, value)
            if is_valid:
                dic[key] = value
                checked_keys.append(key)
            else:
                logger.info(
                    f"'{key}' is an unexpected key, " "its value won't be considered."
                )
        return {key: dic[key] for key in checked_keys}

    @staticmethod
    def filter_dict(dic: Optional[Dict], filter_value: Any = None) -> Dict:
        """
        Filter out key where the value is None.

        :param dic: a dictionary
        :param filter_value: value to be filtered out
        :return: a dictionary with only keys where the value is not None
        """
        if dic is not None:
            return {k: v for k, v in dic.items() if v is not filter_value}
        return {}

    def load_arguments(self) -> Dict:
        """Parse the byte string, eventually override defaults and check parameters."""
        args = yaml.load(self.raw_config, Loader=yaml.SafeLoader)

        # keep only command line arguments which are defined (not None)
        valid_cla = self.filter_dict(self.command_line_args)

        # override the default config by the valid command line arguments
        args.update(valid_cla)

        # validate the parameters
        self.arguments = self._check_args(args)
        return self.arguments

    def _open_file(self) -> bytes:
        """Open the file and return it."""
        with open(self.file_path, "rb") as f:
            raw_config = f.read()
        return raw_config

    def __repr__(self):
        """Representation string of the ConfigParser instance."""
        return fmt.create_repr(self, ConfigParser)
