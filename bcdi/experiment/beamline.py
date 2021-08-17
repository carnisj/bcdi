# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#   (c) 06/2021-present : DESY CFEL
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

"""Beamline-related classes."""
from abc import ABC, abstractmethod


class Beamline(ABC):
    """
    Base class for defining a beamline.

    :param beamline_name: str, name of the beamline
    """

    valid_beamlines = {
            "ID01",
            "SIXS_2018",
            "SIXS_2019",
            "34ID",
            "P10",
            "CRISTAL",
            "NANOMAX",
        }

    def __init__(self, beamline_name):
        self.beamline_name = beamline_name

    @property
    def beamline_name(self):
        """Name of the beamline."""
        return self._beamline_name

    @beamline_name.setter
    def beamline_name(self, name):
        if name not in self.valid_beamlines:
            raise ValueError(
                f"Invalid beamline name, valid are {self.valid_beamlines}"
            )
        self._beamline_name = name

