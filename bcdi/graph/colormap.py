# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr
"""
Implementation of the colormap class.

New colormaps can be added to the method generate_colormap.
"""
from matplotlib.colors import Colormap, LinearSegmentedColormap, ListedColormap
from typing import Optional

from bcdi.graph.turbo_colormap import turbo_colormap_data
from bcdi.utils.validation import is_float

data_table = (
    (0.0, 1.0, 1.0),
    (0.11, 0.0, 0.0),
    (0.36, 0.0, 0.0),
    (0.62, 1.0, 1.0),
    (0.87, 1.0, 1.0),
    (1.0, 0.0, 0.0),
)

custom_colormap_data = {
    "red": data_table,
    "green": data_table,
    "blue": data_table,
}


class ColormapFactory:
    """
    Class to define a colormap.

    :param colormap: a colormap string. Available choices: 'turbo',
     'custom'
    :param bad_color: a string which defines the grey level for nan pixels, e.g. '0.7'
    """

    valid_colormaps = ["turbo", "custom"]

    def __init__(self, bad_color="0.7", colormap="turbo"):
        self.bad_color = bad_color
        self.colormap = colormap
        self.cmap: Optional[Colormap] = None

    @property
    def bad_color(self):
        """Color for masked values."""
        return self._bad_color

    @bad_color.setter
    def bad_color(self, val: str):
        if not isinstance(val, str):
            raise TypeError(f"bad_color should be a str, got {type(val)})")
        if not is_float(val) or not (0.0 <= float(val) <= 1.0):
            raise ValueError("float(bad_color) should be a number between 0 and 1")
        self._bad_color = val

    @property
    def colormap(self):
        """Name of the colormap."""
        return self._colormap

    @colormap.setter
    def colormap(self, val: str):
        if not isinstance(val, str):
            raise TypeError(f"colormap should be a str, got {type(val)})")
        if val not in self.valid_colormaps:
            raise NotImplementedError(f"colormap {val} not implemented")
        self._colormap = val

    def generate_cmap(self):
        if self.colormap == "turbo":
            self.cmap = ListedColormap(turbo_colormap_data)
        elif self.colormap == "custom":
            self.cmap = LinearSegmentedColormap(
                "my_colormap", custom_colormap_data, 256
            )
        else:
            raise NotImplementedError(f"colormap {self.colormap} not implemented")
        self.cmap.set_bad(color=self.bad_color)
