# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, jerome.carnis@esrf.fr

"""Classes and functions for the configuration of logging."""

from dataclasses import dataclass
from logging import (
    CRITICAL,
    DEBUG,
    ERROR,
    INFO,
    WARNING,
    FileHandler,
    Formatter,
    StreamHandler,
    basicConfig,
)
from typing import Optional

FILE_FORMATTER = Formatter(
    "\n%(asctime)s-%(levelname)s-%(name)s.%(funcName)s:\n%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
VALID_LEVELS = [CRITICAL, DEBUG, ERROR, INFO, WARNING]


@dataclass
class LoggingColor:
    """Define terminal color codes."""

    BOLD = "\033[1m"
    BLUE = "\033[94m"
    WHITE = "\033[97m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD_WHITE = BOLD + WHITE
    BOLD_BLUE = BOLD + BLUE
    BOLD_GREEN = BOLD + GREEN
    BOLD_YELLOW = BOLD + YELLOW
    BOLD_RED = BOLD + RED
    END = "\033[0m"


class ColorLogFormatter(Formatter):
    """Format colored logs."""

    FORMAT = "\n%(prefix)s%(levelname)s-%(module)s.%(funcName)s:\n%(message)s%(suffix)s"

    LOG_LEVEL_COLOR = {
        "DEBUG": {"prefix": LoggingColor.WHITE, "suffix": LoggingColor.END},
        "INFO": {"prefix": LoggingColor.GREEN, "suffix": LoggingColor.END},
        "WARNING": {"prefix": LoggingColor.BOLD_YELLOW, "suffix": LoggingColor.END},
        "ERROR": {"prefix": LoggingColor.BOLD_RED, "suffix": LoggingColor.END},
        "CRITICAL": {"prefix": LoggingColor.BOLD_RED, "suffix": LoggingColor.END},
    }

    def format(self, record):
        """
        Format log records.

        It uses a default prefix and suffix to terminal color codes that corresponds
        to the log level name.
        """
        if not hasattr(record, "prefix"):
            record.prefix = self.LOG_LEVEL_COLOR.get(
                record.levelname.upper(), "INFO"
            ).get("prefix")

        if not hasattr(record, "suffix"):
            record.suffix = self.LOG_LEVEL_COLOR.get(
                record.levelname.upper(), "INFO"
            ).get("suffix")

        formatter = Formatter(self.FORMAT, datefmt="%m/%d/%Y %H:%M:%S")
        return formatter.format(record)


def configure_logging(
    path: Optional[str] = None, console_level: int = INFO, file_level: int = DEBUG
) -> None:
    """
    Create handlers and configure logging.

    :param path: path of the file where to write logs
    :param console_level: valid level for logging to the console
    :param file_level: valid level for logging to a file
    """
    if console_level not in VALID_LEVELS:
        raise ValueError(f"invalid logging level for the console: {console_level}")
    if file_level not in VALID_LEVELS:
        raise ValueError(f"invalid logging level for the file: {file_level}")

    # create console and file handlers
    console_hdl = StreamHandler()
    console_hdl.setLevel(console_level)
    console_hdl.setFormatter(ColorLogFormatter())

    if path is not None:
        file_hdl = FileHandler(path, mode="w", encoding="utf-8")
        file_hdl.setLevel(file_level)
        file_hdl.setFormatter(FILE_FORMATTER)
        # configure the root logger
        basicConfig(level=DEBUG, handlers=[console_hdl, file_hdl])
    else:
        basicConfig(level=DEBUG, handlers=[console_hdl])
