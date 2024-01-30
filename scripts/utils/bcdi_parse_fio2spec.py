#!/usr/bin/env python3

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Steven Leake, steven.leake@esrf.fr

import os
import re

import h5py
import numpy as np

helptext = """
Script parsing the log file of P10.
"""

basepath = "/gpfs/current/raw/"
# basepath = "/mntdirect/_data_id01_inhouse/otherlightsources/2018_P10_JC/misc/"
sampleid = "dewet2_2_%05i"
logfilename = basepath + "SPOCK/spock_output_e1.log"
##########################################################


def open_log(fn="SPOCK/spock_output_e1.log"):
    """Open P10 experimental log file."""
    if not fn.endswith(".log"):
        raise ValueError(f"file {fn} is not a regular .log file")
    with open(fn, encoding="utf-8") as file:
        rows = file.readlines()
    list1 = []
    list2 = []
    my_regions = {}
    region_temp = [0, 0, 0, 0]
    for myrow in rows[:]:
        if myrow.count("senv ROI") > 0 and myrow.count("p10/door/haspp10e1.01") > 0:
            try:
                # print(myrow)
                roi_tmp1 = myrow.split()[-4:]
                roi_tmp1[3] = roi_tmp1[3][:4]
                # print(region_temp)
                region_temp = map(int, roi_tmp1)
                print(region_temp)
                # print(myrow)
            except IndexError:
                pass

        if myrow.count("Scan #") > 0:
            print(myrow)
            scanno = re.findall(r"Scan #\d+", myrow)[0].split("#")[1]
            print(scanno)
            try:
                print(int(scanno))
                list1.append(scanno)
                list2.append(region_temp)
                print(region_temp)
                my_regions[str(scanno)] = region_temp
                # print(myrow)
            except TypeError:
                pass
    return my_regions


def parsefio(fio_filename: str):
    """Parse the P10 experimental fio file."""
    if not fio_filename.endswith(".fio"):
        raise ValueError(f"file {fio_filename} is not a regular .fio file")
    with open(fio_filename, encoding="utf-8") as file:
        rows = file.readlines()
    index = rows.index("%c\n")
    ii = 0
    myheader = {}

    for myrow in rows[index + 1 :]:
        if myrow.startswith("!"):
            break
        myrow = myrow.split("\n")[0]
        myheader[ii] = myrow
        ii += 1

    index = rows.index("%p\n")
    ii = 0
    mymotors = {}

    for myrow in rows[index + 1 :]:
        if myrow.startswith("!"):
            break
        myrow = myrow.split("\n")[0]
        mymotors[myrow.split(" = ")[0]] = float(myrow.split(" = ")[1]) - 1
        ii += 1

    index = rows.index("%d\n")
    ii = 0
    mycols = {}

    for myrow in rows[index:]:
        if myrow.startswith(" Col"):
            mycols[myrow.split()[2]] = int(myrow.split()[1]) - 1
            ii += 1
    scan_data = rows[index + ii + 1 : -1]

    out = []
    for myrow in scan_data:
        out.append(map(float, myrow.split()))
    scandata_array = np.array(out)
    return myheader, mymotors, mycols, scandata_array, scan_data


#####################################################################
fns = os.listdir(basepath)
nos = np.arange(30000)
for no in nos:
    targetdir = sampleid % no
    try:
        fns.index(targetdir)
        add2file = True
    except ValueError:  # targetdir not in path
        # print("no data for this one: ",targetdir)
        add2file = False

    if add2file:
        print(targetdir)
        # lookup table for ROI values
        ROIs = open_log(logfilename)
        # add fio file
        fio_fn = basepath + sampleid % no + "/" + sampleid % no + ".fio"
        header, motors, cols, scandataarr, scandata = parsefio(fio_fn)
        # add scans file
        h5_fn = (
            basepath
            + sampleid % no
            + "/"
            + "e4m/"
            + sampleid % no
            + "_data_%06i.h5" % 1
        )
        with h5py.File(h5_fn, "r") as f:
            Mydataset = f["/entry/data/data"].value
            Mydataset[Mydataset > 10000] = 0
        ROI = ROIs[str(no)]
        sumData = (
            Mydataset[:, 2167 - ROI[3] : 2167 - ROI[2], ROI[0] : ROI[1]]
            .sum(axis=1)
            .sum(axis=1)
        )
        # find the maxima
        # maxVal = np.argmax(sumData)
        # cen_index=int(scind.center_of_mass(sumData)[0])

        cols_str = "#L "
        motor_name = "None"
        for i in range(len(cols.keys())):
            for key in cols:
                if cols[key] == i:
                    cols_str = cols_str + key + "  "
                if cols[key] == 0:
                    motor_name = key
        cols_str = cols_str + "  sumInt\n"

        outputarr = []
        for row in scandata:
            outputarr.append(map(float, row.split()))
        scandataarr = np.array(outputarr)
        # self.output(scandataarr[:,0])
        # cen_pos = scandataarr[:,0][cen_index]

        mot_str = "#O0"
        pos_str = "#P0"
        for mot in motors:
            mot_str += "  %s" % mot
            pos_str += "  %.5f" % motors[mot]
        mot_str += "\n"
        pos_str += "\n"
        logfileName = basepath + "testdelete.spec"

        with open(logfileName, "a") as logfile:
            logfile.write(
                "#S %i %s %s %f %f \n"
                % (
                    no,
                    header[0].split(" ")[0],
                    motor_name,
                    scandataarr[:, 0][0],
                    scandataarr[:, 0][1],
                )
            )

            logfile.write(mot_str)
            logfile.write(pos_str)

            out_str = cols_str
            logfile.write(out_str)
            for jj, row in enumerate(scandata):
                out_str = row.split("\n")[0] + " %i\n" % sumData[jj]
                logfile.write(out_str)
