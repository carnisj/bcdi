#!/usr/bin/env python
# Created on Mon Mar 18 14:39:36 2019
# Meant to open the data generated from the datarecorder upgrade of january 2019
# Modified again the 24/06/2020
# @author: Andrea Resta
"""
ReadNxs3.

This module contains Classes and functions to load data at SIXS beamline. It is meant to
be used on the data produced after the 11/03/2019 data of the upgrade of the
datarecorder.
"""


import inspect
import os
import pickle
import time
from typing import Optional

import numpy as np
import tables
from matplotlib import pyplot as plt

import bcdi
import bcdi.utils.utilities as util


class EmptyO:
    """Empty class used as container in the nxs2spec case."""


class DataSet:
    """
    Class dealing with datasets.

    It reads the file and stores it in an object, from this object we can
    retrive the data to use it.

    Use as: dataObject = nxsRead3.DataSet( path/filename, path )

    Filename can be '/path/filename.nxs' or just 'filename.nxs'. Directory is
    optional and it can contain '/dir00/dir01', both are meant to be string.

     - It returns an object with attibutes all the recorded sensor/motors listed in
       the nxs file.
     - If the sensor/motor is not in the "aliad_dict" file it generate a name from
       the device name.

    The object will also contains some basic methods for data reuction such as ROI
    extraction, attenuation correction and plotting. Meant to be used on the data
    produced after the 11/03/2019 data of the upgrade of the datarecorder.
    """

    allowed_alias_dict = [
        "alias_dict_2018.txt",
        "alias_dict_2019.txt",
        "alias_dict_2020.txt",
        "alias_dict_2021.txt",
        "alias_dict_2022.txt",
    ]

    def __init__(self, filename, directory="", nxs2spec=False, alias_dict=None):
        self.directory = directory
        self.filename = filename
        self.end_time = 2
        self.file: Optional[tables.File] = None
        self.start_time = 1
        self.attlist = []
        self._list2d = []
        self._SpecNaNs = (
            nxs2spec  # Remove the NaNs if the spec file need to be generated
        )
        attlist = []  # used for self generated file attribute list
        aliases = []  # used for SBS imported with alias_dict file

        self.path_aliases = inspect.getfile(bcdi).split("__")[0] + "preprocessing/"

        print("### ReadNxs3 ###")
        if not alias_dict:
            print("Defaulting to alias_dict_2022.txt")
            alias_dict = self.path_aliases + "alias_dict_2022.txt"
        if os.path.split(alias_dict)[-1] not in self.allowed_alias_dict:
            raise ValueError(f"{alias_dict} is not in the list of allowed names")
        try:
            self._alias_dict = pickle.load(open(alias_dict, "rb"))
        except pickle.UnpicklingError:
            # need to convert DOS linefeeds (crlf) to UNIX (lf)
            dirname = os.path.dirname(alias_dict)
            dict_name = os.path.splitext(os.path.basename(alias_dict))[
                0
            ]  # e.g.'alias_dict_2021'
            util.dos2unix(
                input_file=alias_dict,
                savedir=dirname + f"\\{dict_name}_unix.txt",
            )
            alias_dict = dirname + f"\\{dict_name}_unix.txt"
            self._alias_dict = pickle.load(open(alias_dict, "rb"))

        def is_empty(any_structure):
            """Quick function to determine if an array, tuple or string is empty."""
            if any_structure:
                return False
            return True

        # Load the file
        fullpath = os.path.join(self.directory, self.filename)
        self.file = tables.open_file(fullpath, "r")
        f = self.file.list_nodes("/")[0]

        # check if any scanned data a are present
        if not hasattr(f, "scan_data"):
            self.scantype = "unknown"
            self.file.close()
            return

        # Discriminating between SBS or FLY scans
        try:
            if f.scan_data.data_01.name == "data_01":
                self.scantype = "SBS"
        except tables.NoSuchNodeError:
            self.scantype = "FLY"

        ###############
        # Reading FLY #
        ###############
        if self.scantype == "FLY":
            # generating the attributes with the recorded scanned data
            for leaf in f.scan_data:
                list.append(attlist, leaf.name)
                self.__dict__[leaf.name] = leaf[:]
                time.sleep(0.1)
            self.attlist = attlist

        ###############
        # Reading SBS #
        ###############
        if self.scantype == "SBS":
            if self._alias_dict:  # Reading with dictionary
                for leaf in f.scan_data:
                    try:
                        alias = self._alias_dict[leaf.attrs.long_name.decode("UTF-8")]
                        if alias not in aliases:
                            aliases.append(alias)
                            self.__dict__[alias] = leaf[:]
                    except KeyError:
                        self.__dict__[leaf.attrs.long_name.decode("UTF-8")] = leaf[:]
                        aliases.append(leaf.attrs.long_name.decode("UTF-8"))
                self.attlist = aliases

            else:
                for leaf in f.scan_data:  # Reading with dictionary
                    # generating the attributes with the recorded scanned data
                    attr = leaf.attrs.long_name.decode("UTF-8")
                    attrshort = leaf.attrs.long_name.decode("UTF-8").split("/")[-1]
                    attrlong = leaf.attrs.long_name.decode("UTF-8").split("/")[-2:]
                    if attrshort not in attlist:
                        if (
                            attr.split("/")[-1] == "sensorsTimestamps"
                        ):  # rename the sensortimestamps as epoch
                            list.append(attlist, "epoch")
                            self.__dict__["epoch"] = leaf[:]
                        else:
                            list.append(attlist, attr.split("/")[-1])
                            self.__dict__[attr.split("/")[-1]] = leaf[:]
                    else:  # Dealing with for double naming
                        list.append(attlist, "_".join(attrlong))
                        self.__dict__["_".join(attrlong)] = leaf[:]
                self.attlist = attlist

        ##########################
        # patch xpad140 / xpad70 #
        ##########################
        # try to correct wrong/inconsistency naming coming from FLY/SBS name system
        bl2d = {
            120: "xpad70",
            240: "xpad140",
            515: "merlin",
            512: "merlin",
            1065: "eiger",
            1040: "cam2",
        }

        self.det2d()  # generating the list self._list2d
        for el in self._list2d:
            detarray = self.get_stack(el)
            detsize = detarray.shape[1]  # the key for the dictionary

            if detsize in bl2d:
                detname = bl2d[detsize]  # the detector name from the size
                if not hasattr(self, detname):
                    self.__setattr__(detname, detarray)  # adding the new attrinute name
                    self.attlist.append(detname)
            else:
                print("Detected a not standard detector: check ReadNxs3")

        self.det2d()  # re-generating the list self._list2d

        #######################################
        # this is for nxs2spec transformations#
        #######################################
        if nxs2spec:
            # import nxs2spec3 as n2s3
            # ideally those lines should be moved into the nxs2spec3
            self._nxs2spec = EmptyO()
            hkl_pre = False
            print("nxs2spec", self.filename)
            try:  # try reading for h k l coordinates at the beginning of the scan.
                # even without hkl transformer  UHV used in nxs2spec
                h_tmp = (
                    f.sample_info._f_get_child("i14-c-cx2-ex-diff-uhv-h")
                    ._f_get_child("position_pre")
                    .read()[0]
                )
                k_tmp = (
                    f.sample_info._f_get_child("i14-c-cx2-ex-diff-uhv-k")
                    ._f_get_child("position_pre")
                    .read()[0]
                )
                l_tmp = (
                    f.sample_info._f_get_child("i14-c-cx2-ex-diff-uhv-l")
                    ._f_get_child("position_pre")
                    .read()[0]
                )
                self._nxs2spec.hkl_string = (
                    "#Q " + str(h_tmp) + " " + str(k_tmp) + " " + str(l_tmp) + "\n"
                )
                self._setup = "UHV"
                hkl_pre = True
            except tables.NoSuchNodeError:
                pass

            try:  # try reading for h k l coordinates at the beginning of the scan.
                # even without hkl transformer  MED_h  used in nxs2spec
                h_tmp = (
                    f.sample_info._f_get_child("i14-c-cx1-ex-dif-med.h-h")
                    ._f_get_child("position_pre")
                    .read()[0]
                )
                k_tmp = (
                    f.sample_info._f_get_child("i14-c-cx1-ex-dif-med.h-k")
                    ._f_get_child("position_pre")
                    .read()[0]
                )
                l_tmp = (
                    f.sample_info._f_get_child("i14-c-cx1-ex-dif-med.h-l")
                    ._f_get_child("position_pre")
                    .read()[0]
                )
                self._nxs2spec.hkl_string = (
                    "#Q " + str(h_tmp) + " " + str(k_tmp) + " " + str(l_tmp) + "\n"
                )
                self._setup = "MED_H"
                hkl_pre = True
            except tables.NoSuchNodeError:
                pass

            try:  # try reading for h k l coordinates at the beginning of the scan.
                # even without hkl transformer  MED_h  used in nxs2spec
                h_tmp = (
                    f.sample_info._f_get_child("i14-c-cx1-ex-dif-med.1-h")
                    ._f_get_child("position_pre")
                    .read()[0]
                )
                k_tmp = (
                    f.sample_info._f_get_child("i14-c-cx1-ex-dif-med.1-k")
                    ._f_get_child("position_pre")
                    .read()[0]
                )
                l_tmp = (
                    f.sample_info._f_get_child("i14-c-cx1-ex-dif-med.1-l")
                    ._f_get_child("position_pre")
                    .read()[0]
                )
                self._nxs2spec.hkl_string = (
                    "#Q " + str(h_tmp) + " " + str(k_tmp) + " " + str(l_tmp) + "\n"
                )
                self._setup = "MED"
                hkl_pre = True
            except tables.NoSuchNodeError:
                pass

            if not hkl_pre:  # communicate it to user
                print("No hkl_prescan")

            # building the P lines for the nxs2spec #
            p0 = False
            try:
                mu = (
                    f.SIXS._f_get_child("i14-c-cx2-ex-mu-uhv")
                    ._f_get_child("position_pre")
                    .read()[0]
                )
                omega = (
                    f.SIXS._f_get_child("i14-c-cx2-ex-omega-uhv")
                    ._f_get_child("position_pre")
                    .read()[0]
                )
                delta = (
                    f.SIXS._f_get_child("i14-c-cx2-ex-delta-uhv")
                    ._f_get_child("position_pre")
                    .read()[0]
                )
                gamma = (
                    f.SIXS._f_get_child("i14-c-cx2-ex-gamma-uhv")
                    ._f_get_child("position_pre")
                    .read()[0]
                )
                p1 = str(delta)
                p2 = str(omega)
                p3 = str(0.000)
                p4 = str(0.000)
                p5 = str(mu)
                p6 = str(gamma)
                p7 = str(gamma)
                self._nxs2spec.P_line = (
                    "#P0 "
                    + p1
                    + " "
                    + p2
                    + " "
                    + p3
                    + " "
                    + p4
                    + " "
                    + p5
                    + " "
                    + p6
                    + " "
                    + p7
                    + "\n"
                )
                p0 = True
            except tables.NoSuchNodeError:
                pass
            try:
                mu = (
                    f.SIXS._f_get_child("i14-c-cx1-ex-mu-med-h")
                    ._f_get_child("position_pre")
                    .read()[0]
                )
                delta = (
                    f.SIXS._f_get_child("i14-c-cx1-ex-delta-med-h")
                    ._f_get_child("position_pre")
                    .read()[0]
                )
                gamma = (
                    f.SIXS._f_get_child("i14-c-cx1-ex-gamma-med-h")
                    ._f_get_child("position_pre")
                    .read()[0]
                )
                beta = (
                    f.SIXS._f_get_child("i14-c-cx1-ex-diff-med-tpp")
                    ._f_get_child("position_pre")
                    .read()[0]
                )
                p1 = str(gamma)
                p2 = str(mu)
                p3 = str(0.000)
                p4 = str(0.000)
                p5 = str(beta)
                p6 = str(delta)
                p7 = str(delta)
                self._nxs2spec.P_line = (
                    "#P0 "
                    + p1
                    + " "
                    + p2
                    + " "
                    + p3
                    + " "
                    + p4
                    + " "
                    + p5
                    + " "
                    + p6
                    + " "
                    + p7
                    + "\n"
                )
                p0 = True
            except tables.NoSuchNodeError:
                pass
            try:
                mu = (
                    f.SIXS._f_get_child("i14-c-cx1-ex-mu-med-v")
                    ._f_get_child("position_pre")
                    .read()[0]
                )
                delta = (
                    f.SIXS._f_get_child("i14-c-cx1-ex-delta-med-v")
                    ._f_get_child("position_pre")
                    .read()[0]
                )
                gamma = (
                    f.SIXS._f_get_child("i14-c-cx1-ex-gamma-med-v")
                    ._f_get_child("position_pre")
                    .read()[0]
                )
                omega = (
                    f.SIXS._f_get_child("i14-c-cx1-ex-omega-med-v")
                    ._f_get_child("position_pre")
                    .read()[0]
                )
                p1 = str(delta)
                p2 = str(omega)
                p3 = str(0.000)
                p4 = str(0.000)
                p5 = str(mu)
                p6 = str(gamma)
                p7 = str(gamma)
                self._nxs2spec.P_line = (
                    "#P0 "
                    + p1
                    + " "
                    + p2
                    + " "
                    + p3
                    + " "
                    + p4
                    + " "
                    + p5
                    + " "
                    + p6
                    + " "
                    + p7
                    + "\n"
                )
                p0 = True
            except tables.NoSuchNodeError:
                pass
            if not p0:
                print("No P0 Line")

        ############################################################
        # adding some useful attributes common between SBS and FLY #
        ############################################################
        try:
            mono = f.SIXS.__getattr__("i14-c-c02-op-mono")
            self.waveL = mono.__getattr__("lambda")[0]
            self.energymono = mono.energy[0]
        except tables.NoSuchNodeError:
            self.energymono = f.SIXS.Monochromator.energy[0]
            self.waveL = f.SIXS.Monochromator.wavelength[0]
        # probing time stamps and eventually use epoch to rebuild them
        if hasattr(
            f, "end_time"
        ):  # sometimes this attribute is absent, especially on the ctrl+C scans
            try:
                self.end_time = f.end_time._get_obj_timestamps().ctime
            except Exception:
                if is_empty(np.shape(f.end_time)):
                    try:
                        self.end_time = max(self.epoch)
                    except AttributeError:
                        self.end_time = 1.7e9 + 2
                        print("File has time stamps issues")
                else:
                    self.end_time = f.end_time[0]
        elif not hasattr(f, "end_time"):
            print("File has time stamps issues")
            self.end_time = 1.7e9 + 2  # necessary for nxs2spec conversion

        if hasattr(f, "start_time"):
            try:
                self.start_time = f.start_time._get_obj_timestamps().ctime
            except Exception:
                if is_empty(np.shape(f.start_time)):
                    try:
                        self.start_time = min(self.epoch)
                    except AttributeError:
                        self.start_time = 1.7e9
                        print("File has time stamps issues")
                else:
                    self.start_time = f.start_time[0]
        elif not hasattr(
            f, "start_time"
        ):  # sometimes this attribute is absent, especially on the ctrl+C scans
            print("File has time stamps issues")
            self.start_time = 1.7e9  # necessary for nxs2spec conversion
        try:  # att_coef
            self._coef = f.SIXS._f_get_child("i14-c-c00-ex-config-att").att_coef[0]
            self.attlist.append("_coef")
        except tables.NoSuchNodeError:
            print("No att coef")
        try:
            self._integration_time = f.SIXS._f_get_child(
                "i14-c-c00-ex-config-publisher"
            ).integration_time[0]
            self.attlist.append("_integration_time")
        except tables.NoSuchNodeError:
            self._integration_time = 1
            print("No integration time defined")

            ##################
            # XPADS/ 2D ROIs #
            # ################

        try:  # kept for "files before 18/11/2020   related to xpad70/140 transition"
            if (
                self.start_time < 1605740000
            ):  # apply to file older than Wed Nov 18 23:53:20 2020
                self._roi_limits_xpad140 = f.SIXS._f_get_child(
                    "i14-c-c00-ex-config-publisher"
                ).roi_limits[:][:]
                self.attlist.append("_roi_limits_xpad140")
                roi_names_cell = f.SIXS._f_get_child(
                    "i14-c-c00-ex-config-publisher"
                ).roi_name.read()
                self._roi_names_xpad140 = roi_names_cell.tolist().decode().split("\n")
                self.attlist.append("_roi_names_xpad140")
                self._ifmask_xpad140 = f.SIXS._f_get_child(
                    "i14-c-c00-ex-config-publisher"
                ).ifmask[:]
                self.attlist.append("_ifmask_xpad140")
                try:
                    self._mask_xpad140 = f.SIXS._f_get_child(
                        "i14-c-c00-ex-config-publisher"
                    ).mask[:]
                    self.attlist.append("_mask_xpad140")
                except tables.NoSuchNodeError:
                    print("No Mask")
            if (
                self.start_time > 1605740000
            ):  # apply to file after  Wed Nov 18 23:53:20 2020
                dets = (
                    self._list2d
                )  # the 2D detector list potentially extend here for the eiger ROIs
                for el in dets:
                    if el == "xpad70":
                        self._roi_limits_xpad70 = f.SIXS._f_get_child(
                            "i14-c-c00-ex-config-xpads70"
                        ).roi_limits[:][:]
                        self.attlist.append("_roi_limits_xpad70")
                        self._distance_xpad70 = f.SIXS._f_get_child(
                            "i14-c-c00-ex-config-xpads70"
                        ).distance_xpad[:]
                        self.attlist.append("_distance_xpad70")
                        self._ifmask_xpad70 = f.SIXS._f_get_child(
                            "i14-c-c00-ex-config-xpads70"
                        ).ifmask[:]
                        self.attlist.append("_ifmask_xpad70")
                        try:
                            self._mask_xpad70 = f.SIXS._f_get_child(
                                "i14-c-c00-ex-config-xpads70"
                            ).mask[:]
                            self.attlist.append("_mask_xpad70")
                        except tables.NoSuchNodeError:
                            print("no mask xpad70")
                        roi_names_cell = f.SIXS._f_get_child(
                            "i14-c-c00-ex-config-xpads70"
                        ).roi_name.read()
                        self._roi_names_xpad70 = (
                            roi_names_cell.tolist().decode().split("\n")
                        )
                        self.attlist.append("_roi_names_xpad70")

                    if el == "xpad140":
                        self._roi_limits_xpad140 = f.SIXS._f_get_child(
                            "i14-c-c00-ex-config-xpads140"
                        ).roi_limits[:][:]
                        self.attlist.append("_roi_limits_xpad140")
                        self._distance_xpad140 = f.SIXS._f_get_child(
                            "i14-c-c00-ex-config-xpads140"
                        ).distance_xpad[:]
                        self.attlist.append("_distance_xpad140")
                        self._ifmask_xpad140 = f.SIXS._f_get_child(
                            "i14-c-c00-ex-config-xpads140"
                        ).ifmask[:]
                        self.attlist.append("_ifmask_xpad140")
                        try:
                            self._mask_xpad140 = f.SIXS._f_get_child(
                                "i14-c-c00-ex-config-xpads140"
                            ).mask[:]
                            self.attlist.append("_mask_xpad140")
                        except tables.NoSuchNodeError:
                            print("no mask xpad140")
                        roi_names_cell = f.SIXS._f_get_child(
                            "i14-c-c00-ex-config-xpads140"
                        ).roi_name.read()
                        self._roi_names_xpad140 = (
                            roi_names_cell.tolist().decode().split("\n")
                        )
                        self.attlist.append("_roi_names_xpad140")
                    if el == "merlin":
                        self._roi_limits_merlin = f.SIXS._f_get_child(
                            "i14-c-c00-ex-config-merlin"
                        ).roi_limits[:][:]
                        self.attlist.append("_roi_limits_merlin")
                        self._distance_merlin = f.SIXS._f_get_child(
                            "i14-c-c00-ex-config-merlin"
                        ).distance_xpad[:]
                        self.attlist.append("_distance_merlin")
                        self._ifmask_merlin = f.SIXS._f_get_child(
                            "i14-c-c00-ex-config-merlin"
                        ).ifmask[:]
                        self.attlist.append("_ifmask_merlin")
                        try:
                            self._mask_merlin = f.SIXS._f_get_child(
                                "i14-c-c00-ex-config-merlin"
                            ).mask[:]
                            self.attlist.append("_mask_merlin")
                        except tables.NoSuchNodeError:
                            print("no mask merlin")
                        roi_names_cell = f.SIXS._f_get_child(
                            "i14-c-c00-ex-config-merlin"
                        ).roi_name.read()
                        self._roi_names_merlin = (
                            roi_names_cell.tolist().decode().split("\n")
                        )
                        self.attlist.append("_roi_names_merlin")
        except tables.NoSuchNodeError:
            print("No Xpad Publisher defined")
        print("### End of ReadNxs3 ###\n")
        self.file.close()

    ############################################
    # down here useful function in the NxsRead #
    ############################################
    def close(self):
        """Close the data file."""
        self.file.close()

    def get_stack(self, det2d_name):
        """
        Check in the attribute-list for a given 2D detector.

        :param det2d_name: string, detector name
        :return: the stack of images
        """
        try:
            stack = self.__getattribute__(det2d_name)
            return stack
        except AttributeError:
            print("There is no such attribute")
            return None

    def make_mask_frame_xpad(self):
        """
        Correction for XPAD detector.

        It generates a new attribute 'mask0_xpad' to remove the double pixels. It can
        be applied only to xpads140 for now.
        """
        detlist = self.det2d()

        if "xpad140" in detlist:
            mask = np.zeros((240, 560), dtype="bool")
            mask[:, 79:560:80] = True
            mask[:, 80:561:80] = True
            mask[119, :] = True
            mask[120, :] = True
            mask[:, 559] = False
            self.mask0_xpad140 = mask
            self.attlist.append("mask0_xpad140")
        if "xpad70" in detlist:
            mask = np.zeros((120, 560), dtype="bool")
            mask[:, 79:559:80] = True
            mask[:, 80:560:80] = True
            self.mask0_xpad70 = mask
            self.attlist.append("mask0_xpad70")

    @staticmethod
    def roi_sum(stack, roi):
        """
        Integrate intensity in a ROI for a stack of images.

        :param stack: the stack of images
        :param roi: ROI for the integration, e.g. [257, 126,  40,  40]
        :return: the summed intensity for the stack of images
        """
        return (
            stack[:, roi[1] : roi[1] + roi[3], roi[0] : roi[0] + roi[2]]
            .sum(axis=1)
            .sum(axis=1)
            .copy()
        )

    @staticmethod
    def roi_sum_mask(stack, roi, mask):
        """
        Integrate intensity in a ROI for a stack of images, neglecting masked pixels.

        :param stack: the stack of images
        :param roi: ROI for the integration, e.g. [257, 126,  40,  40]
        :param mask: the mask
        :return: the summed intensity for the stack of images
        """
        _stack = stack[:] * (1 - mask.astype("uint16"))
        return (
            _stack[:, roi[1] : roi[1] + roi[3], roi[0] : roi[0] + roi[2]]
            .sum(axis=1)
            .sum(axis=1)
        )

    def calc_roi(
        self, stack, roiextent, maskname, attcoef, filters, acq_time, roi_name
    ):
        """
        Calculate a ROI.

        To calculate the roi corrected by attcoef, mask, filters, acquisition_time
        roi_name is the name of the attribute that will be attached to the dataset
        object mind that there might be a shift between motors and filters in the SBS
        scans the ROI is expected as eg: [257, 126,  40,  40]
        """
        if hasattr(self, maskname):
            mask = self.__getattribute__(maskname)
            integrals = self.roi_sum_mask(stack, roiextent, mask)
        if not hasattr(self, maskname):
            integrals = self.roi_sum(stack, roiextent)
            print("Mask Not Used")

        if (
            self.scantype == "SBS"
        ):  # here handling the data shift between data and filters SBS
            _filterchanges = np.where((filters[1:] - filters[:-1]) != 0)
            roi_c = (integrals[:] * (attcoef ** filters[:])) / acq_time
            _filterchanges = np.asanyarray(_filterchanges)
            if self._SpecNaNs:  # PyMCA do not like NaNs in the last column
                np.put(roi_c, _filterchanges + 1, 0)
            if not self._SpecNaNs:  # but for data analysis NaNs are better
                np.put(roi_c, _filterchanges + 1, np.NaN)
            setattr(self, roi_name, roi_c)
            self.attlist.append(roi_name)
        if self.scantype == "FLY":
            f_shi = np.concatenate((filters[2:], filters[-1:], filters[-1:]))
            # here handling the data shift between data and filters FLY
            roi_c = (integrals[:] * (attcoef ** f_shi[:])) / acq_time
            setattr(self, roi_name, roi_c)
            self.attlist.append(roi_name)

    def plot_roi(self, motor, roi, color="-og", detname=None, label=None):
        """
        Integrate the desired roi and plot it as quick verification.

        :param motor: motor name string
        :param roi: roi name string of the desired region measured or in the
         form: [257, 126,  40,  40]
        :param color: color for the plot
        :param detname: detector name. It uses the first detector it finds if not
         differently specified
        :param label: label for the plot
        """
        if not detname:
            detname = self.det2d()[0]
            print(detname)

        if motor in self.attlist:
            xmot = getattr(self, motor)
        else:
            xmot = 0
        if detname:
            stack = self.get_stack(detname)
            if isinstance(roi, str):
                roi_arr = self._roi_limits[self._roi_names.index(roi)]
                yint = self.roi_sum(stack, roi_arr)
            elif isinstance(roi, list):
                roi_arr = roi
                yint = self.roi_sum(stack, roi_arr)
            else:
                yint = 0
        else:
            yint = 0
        plt.plot(xmot, yint, color, label=label)

    def plotscan(self, xvar, yvar, color="-og", label=None):
        """
        Plot xvar vs yvar.

        xvar and yvar must be in the attributes list.
        """
        if xvar in self.attlist:
            x = getattr(self, xvar)
            print("x ok")
        else:
            x = 0
        if yvar in self.attlist:
            y = getattr(self, yvar)
            print("y ok")
        else:
            y = 0
        plt.plot(x, y, color, label=label)

    def calc_roi_new2(self):
        """
        Calculate of the ROI.

        If exist _coef, _integration_time, _roi_limits, _roi_names it can be applied
        to recalculate the roi on one or more 2D detectors. filters and motors are
        shifted of one points for the FLY. corrected in the self.calc_roi. For SBS
        the data point when the filter is changed is collected with no constant
        absorber and therefore is rejected.
        """
        list2d = self._list2d
        common_roots = [
            "_roi_limits",
            "_roi_names",
            "_ifmask",
        ]  # common root names for attributes
        commons = ["_coef", "_integration_time"]  # common attributes
        possible = True
        if not list2d:
            possible = possible * list2d
        if list2d:
            for el in list2d:  # verify the conditions for 6 attributes
                for el2 in common_roots:
                    to_check = el2 + "_" + el
                    if not hasattr(self, to_check):
                        print("missing ", to_check)
                    possible = possible * hasattr(self, to_check)
                    # Bool variable used to keep track
                    # if all the attributes are presents
        for el in commons:
            possible = possible * hasattr(self, el)

        if self.scantype == "SBS":  # SBS Correction
            possible = possible * hasattr(self, "att_sbs_xpad")
            if not possible:
                print(
                    "No correction applied: missing  some data.... check for detector,"
                    " attenuation, integration time, att_coef ROIs..."
                )
            if possible:
                if hasattr(self, "_npts"):  # modified until here.
                    print("Correction already applied")
                if not hasattr(
                    self, "_npts"
                ):  # check if the process was alredy runned once on this object
                    self._npts = len(self.__getattribute__(list2d[0]))
                    for el in list2d:
                        if self.__getattribute__("_ifmask_" + el):
                            maskname = "_mask_" + el
                        if not self.__getattribute__("_ifmask_" + el):
                            maskname = "NO_mask_"  # not existent attribute filtered
                            # away from the roi_sum function
                        for pos, roi in enumerate(
                            self.__getattribute__("_roi_limits_" + el), start=0
                        ):
                            roi_name = (
                                self.__getattribute__("_roi_names_" + el)[pos]
                                + "_"
                                + el
                                + "c_new"
                            )
                            stack = self.__getattribute__(el)
                            attenuators = self.att_sbs_xpad[:]
                            self.calc_roi(
                                stack,
                                roi,
                                maskname,
                                self._coef,
                                attenuators,
                                self._integration_time,
                                roi_name,
                            )

        if self.scantype == "FLY":  # FLY correction
            possible = possible * hasattr(self, "attenuation")
            if not possible:
                print(
                    "No correction applied: missing  some data, check for xpad,"
                    " attenuation, integration time, att_coef and ROIs."
                )

            if possible:
                if hasattr(self, "_npts"):
                    print("Correction already applied")
                if not hasattr(
                    self, "_npts"
                ):  # check if the process was alredy runned once on this object
                    self._npts = len(self.__getattribute__(self._list2d[0]))
                    for el in self._list2d:
                        if self.__getattribute__("_ifmask_" + el):
                            maskname = "_mask_" + el
                        if not self.__getattribute__("_ifmask_" + el):
                            maskname = "NO_mask_"  # not existent attribute filtered
                            # away from the roi_sum function
                        for pos, roi in enumerate(
                            self.__getattribute__("_roi_limits_" + el), start=0
                        ):
                            roi_name = (
                                self.__getattribute__("_roi_names_" + el)[pos]
                                + "_"
                                + el
                                + "c_new"
                            )
                            stack = self.__getattribute__(el)
                            attenuators = self.attenuation[
                                :
                            ]  # filters and motors are shifted of one points
                            self.calc_roi(
                                stack,
                                roi,
                                maskname,
                                self._coef,
                                attenuators,
                                self._integration_time,
                                roi_name,
                            )

    def prj(self, axe=0, mask_extra=None):
        """
        Project a 3D dataset along one axis of the detector.

        :param axe:
         - axe = 0 ==> x axe detector image
         - axe = 1 ==> y axe detector image

        :param mask_extra: specify a mask_extra variable if you like. Mask extra must
         be a the result of np.load(YourMask.npy).
        :return: a matrix of size: 'side detector pixels' x 'number of images'
        """
        if hasattr(self, "mask"):
            mask = self.__getattribute__("mask")
        else:
            mask = 1
        if np.shape(mask_extra):
            mask = mask_extra
            if np.shape(mask) == (240, 560):
                self.make_mask_frame_xpad()
        for el in self.attlist:
            bla = self.__getattribute__(el)
            # get the attributes from list one by one
            if len(bla.shape) == 3:  # check for image stacks
                # Does Not work if you have more than one 2D detectors
                mat = []
                if np.shape(mask) != np.shape(bla[0]):  # verify mask size
                    print(
                        np.shape(mask),
                        "different from ",
                        np.shape(bla[0]),
                        " verify mask size",
                    )
                    mask = 1
                for img in bla:
                    if np.shape(mat)[0] == 0:  # fill the first line element
                        mat = np.sum(img ^ mask, axis=axe)
                    if np.shape(mat)[0] > 0:
                        mat = np.c_[mat, np.sum(img ^ mask, axis=axe)]
                setattr(self, str(el + "_prjX"), mat)  # generate the new attribute

    def det2d(self):
        """Return the name of the 2D detector."""
        list2d = []
        for el in self.attlist:
            bla = self.__getattribute__(el)
            # get the attributes from list one by one
            if (
                isinstance(bla, (np.ndarray, np.generic)) and len(bla.shape) == 3
            ):  # check for image stacks
                list2d.append(el)
        if len(list2d) > 0:
            self._list2d = list2d
            return list2d
        return False
