# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr
"""Implementation of the class used for comments in plots and file names."""

from dataclasses import dataclass


@dataclass
class Comment:
    """Comment used in logging message during the analysis."""

    text: str

    def concatenate(self, val: str, separator: str = "_") -> None:
        self.text += separator + val
