# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, jerome.carnis@esrf.fr

"""Configure logging."""

import logging
from typing import Optional


def configure_logging(path: Optional[str], verbose: bool = False):
    """Create handlers and configure logging."""
    if not isinstance(path, str):
        raise TypeError(f"'path' should be a string, got {type(path)}")

    # create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    # create console and file handlers
    console_hdl = logging.StreamHandler()

    # set levels
    if verbose:
        console_hdl.setLevel(logging.DEBUG)
    else:
        console_hdl.setLevel(logging.ERROR)
    # add formatter to handlers
    console_hdl.setFormatter(formatter)

    if path is not None:
        file_hdl = logging.FileHandler(path, mode="w", encoding="utf-8")
        file_hdl.setLevel(logging.DEBUG)
        file_hdl.setFormatter(formatter)
        # configure the root logger
        logging.basicConfig(level=logging.DEBUG, handlers=[console_hdl, file_hdl])
    else:
        logging.basicConfig(level=logging.DEBUG, handlers=[console_hdl])
