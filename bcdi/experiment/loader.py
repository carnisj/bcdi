# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

"""
Implementation of beanlime-dependent data loading classes.

The class methods manage the initialization of the file system and data loading.
Generic method are implemented in the abstract base class Beamline, and
beamline-dependent methods need to be implemented in each child class (they are
decorated by @abstractmethod in the base class; they are indicated using @ in the
following diagram).

.. mermaid::
  :align: center

  classDiagram
    class Loader{
      +str name
      create_logile(@)
  }
    ABC <|-- Beamline

API Reference
-------------

"""

from abc import ABC, abstractmethod
import h5py
import os
from silx.io.specfile import SpecFile

from bcdi.utils import utilities as util
from bcdi.utils import validation as valid


def create_loader(name, **kwargs):
    """
    Create the instance of the beamline.

    :param name: str, name of the beamline
    :param kwargs: optional beamline-dependent parameters
    :return: the corresponding beamline instance
    """
    if name == "ID01":
        return LoaderID01(name=name, **kwargs)
    if name == "ID01BLISS":
        return LoaderID01BLISS(name=name, **kwargs)
    if name in {"SIXS_2018", "SIXS_2019"}:
        return LoaderSIXS(name=name, **kwargs)
    if name == "34ID":
        return Loader34ID(name=name, **kwargs)
    if name in {"P10", "P10_SAXS"}:
        return LoaderP10(name=name, **kwargs)
    if name == "CRISTAL":
        return LoaderCRISTAL(name=name, **kwargs)
    if name == "NANOMAX":
        return LoaderNANOMAX(name=name, **kwargs)
    raise ValueError(f"Loader {name} not supported")


class Loader(ABC):

    def __init__(self, name, **kwargs):
        self._name = name

    @property
    def name(self):
        """Name of the beamline."""
        return self._name

    @staticmethod
    @abstractmethod
    def create_logfile(**kwargs):
        """
        Create the logfile, which can be a log/spec file or the data itself.

        The nature of this file is beamline dependent.

        :param kwargs: beamline_specific parameters, which may include part of the
         totality of the following keys:

          - 'scan_number': the scan number to load.
          - 'root_folder': the root directory of the experiment, where is e.g. the
            specfile/.fio file.
          - 'filename': the file name to load, or the path of 'alias_dict.txt' for SIXS.
          - 'datadir': the data directory
          - 'template_imagefile': the template for data/image file names
          - 'name': str, the name of the beamline, e.g. 'SIXS_2019'

        :return: logfile
        """


class LoaderID01(Loader):

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

    @staticmethod
    def create_logfile(**kwargs):
        """
        Create the logfile, which is the spec file for ID01.

        :param kwargs:
         - 'root_folder': str, the root directory of the experiment, where is e.g. the
           specfile file.
         - 'filename': str, name of the spec file or full path of the spec file

        :return: logfile
        """
        root_folder = kwargs.get("root_folder")
        filename = kwargs.get("filename")

        valid.valid_container(
            filename,
            container_types=str,
            min_length=1,
            name="filename",
        )

        path = util.find_file(filename=filename, default_folder=root_folder)
        return SpecFile(path)


class LoaderID01BLISS(Loader):

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

    @staticmethod
    def create_logfile(**kwargs):
        """
        Create the logfile, which is the spec file for ID01.

        :param kwargs:
         - 'root_folder': str, the root directory of the experiment, where is e.g. the
           specfile file.
         - 'filename': str, name of the spec file or full path of the spec file

        :return: logfile
        """
        pass  # TODO


class LoaderSIXS(Loader):

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

    @staticmethod
    def create_logfile(**kwargs):
        """
        Create the logfile, which is the data itself for SIXS.

        :param kwargs:
         - 'datadir': str, the data directory
         - 'template_imagefile': str, template for data file name:

           - SIXS_2018: 'align.spec_ascan_mu_%05d.nxs'
           - SIXS_2019: 'spare_ascan_mu_%05d.nxs'

         - 'scan_number': int, the scan number to load
         - 'filename': str, absolute path of 'alias_dict.txt'
         - 'name': str, the name of the beamline, e.g. 'SIXS_2019'

        :return: logfile
        """
        datadir = kwargs.get("datadir")
        template_imagefile = kwargs.get("template_imagefile")
        scan_number = kwargs.get("scan_number")
        filename = kwargs.get("filename")
        name = kwargs.get("name")

        if not os.path.isdir(datadir):
            raise ValueError(f"The directory {datadir} does not exist")
        valid.valid_container(
            template_imagefile, container_types=str, name="template_imagefile"
        )
        valid.valid_container(filename, container_types=str, name="filename")
        valid.valid_item(
            scan_number, allowed_types=int, min_included=0, name="scan_number"
        )

        shortname = template_imagefile % scan_number
        if name == "SIXS_2018":
            # no specfile, load directly the dataset
            import bcdi.preprocessing.nxsReady as nxsReady

            return nxsReady.DataSet(
                longname=datadir + shortname,
                shortname=shortname,
                alias_dict=filename,
                scan="SBS",
            )
        if name == "SIXS_2019":
            # no specfile, load directly the dataset
            import bcdi.preprocessing.ReadNxs3 as ReadNxs3

            return ReadNxs3.DataSet(
                directory=datadir,
                filename=shortname,
                alias_dict=filename,
            )
        raise NotImplementedError(f"{name} is not implemented")


class Loader34ID(Loader):

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

    @staticmethod
    def create_logfile(**kwargs):
        """
        Create the logfile, which is the spec file for 34ID-C.

        :param kwargs:
         - 'root_folder': str, the root directory of the experiment, where is e.g. the
           specfile file.
         - 'filename': str, name of the spec file or full path of the .spec file

        :return: logfile
        """
        root_folder = kwargs.get("root_folder")
        filename = kwargs.get("filename")

        valid.valid_container(
            filename,
            container_types=str,
            min_length=1,
            name="filename",
        )

        path = util.find_file(filename=filename, default_folder=root_folder)
        return SpecFile(path)


class LoaderP10(Loader):

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

    @staticmethod
    def create_logfile(**kwargs):
        """
        Create the logfile, which is the .fio file for P10.

        :param kwargs:
         - 'root_folder': str, the root directory of the experiment, where the scan
           folders are located.
         - 'filename': str, name of the .fio file or full path of the .fio file

        :return: logfile
        """
        root_folder = kwargs.get("root_folder")
        filename = kwargs.get("filename")

        valid.valid_container(
            filename,
            container_types=str,
            min_length=1,
            name="filename",
        )

        if os.path.isfile(filename):
            # filename is already the full path to the .fio file
            return filename
        print(f"Could not find the fio file at: {filename}")

        if not os.path.isdir(root_folder):
            raise ValueError(f"The directory {root_folder} does not exist")

        # return the path to the .fio file
        path = root_folder + filename + "/" + filename + ".fio"
        print(f"Trying to load the fio file at: {path}")
        return path


class LoaderCRISTAL(Loader):

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

    @staticmethod
    def create_logfile(**kwargs):
        """
        Create the logfile, which is the data itself for CRISTAL.

        :param kwargs:
         - 'datadir': str, the data directory
         - 'template_imagefile': str, template for data file name, e.g. 'S%d.nxs'
         - 'scan_number': int, the scan number to load

        :return: logfile
        """
        datadir = kwargs.get("datadir")
        template_imagefile = kwargs.get("template_imagefile")
        scan_number = kwargs.get("scan_number")

        if not os.path.isdir(datadir):
            raise ValueError(f"The directory {datadir} does not exist")
        valid.valid_container(
            template_imagefile, container_types=str, name="template_imagefile"
        )
        valid.valid_item(
            scan_number, allowed_types=int, min_included=0, name="scan_number"
        )

        # no specfile, load directly the dataset
        ccdfiletmp = os.path.join(datadir + template_imagefile % scan_number)
        return h5py.File(ccdfiletmp, "r")


class LoaderNANOMAX(Loader):

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

    @staticmethod
    def create_logfile(**kwargs):
        """
        Create the logfile, which is the data itself for Nanomax.

        :param kwargs:
         - 'datadir': str, the data directory
         - 'template_imagefile': str, template for data file name, e.g. '%06d.h5'
         - 'scan_number': int, the scan number to load

        :return: logfile
        """
        datadir = kwargs.get("datadir")
        template_imagefile = kwargs.get("template_imagefile")
        scan_number = kwargs.get("scan_number")

        if not os.path.isdir(datadir):
            raise ValueError(f"The directory {datadir} does not exist")
        valid.valid_container(
            template_imagefile, container_types=str, name="template_imagefile"
        )
        valid.valid_item(
            scan_number, allowed_types=int, min_included=0, name="scan_number"
        )

        ccdfiletmp = os.path.join(datadir + template_imagefile % scan_number)
        return h5py.File(ccdfiletmp, "r")
