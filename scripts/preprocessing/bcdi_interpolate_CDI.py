#!/usr/bin/env python3

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import multiprocessing as mp
import sys
import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np

import bcdi.graph.graph_utils as gu
import bcdi.utils.utilities as util
import bcdi.utils.validation as valid

helptext = """
This script can be used to interpolate the intensity of masked voxels suing the
centrosymmetry property of a 3D diffraction pattern in the forward CDI geometry.
The diffraction pattern should be in an orthonormal frame with identical voxel sizes
in all directions. The mask should be an array of integers (0 or 1) of the same shape
as the diffraction pattern.
"""

data_dir = "D:/data/P10_August2019_CDI/test/gold_2_2_2_00022/pynx/"
# location of the data and mask
save_dir = None  # path where to save the result, will default to datadir if None
user_comment = (
    "_500_500_500_1_1_1"  # comment for the file name when saving, should start with _
)
origin_voxel = (
    250,
    250,
    250,
)  # tuple of three integers, position in pixels of the origin of reciprocal space
plot_data = (
    True  # True to show plots of the data and mask, before and after the interpolation
)
##################################
# end of user-defined parameters #
##################################

##############################################
# create the dictionnary of input parameters #
##############################################
params = {
    "datadir": data_dir,
    "savedir": save_dir,
    "comment": user_comment,
    "origin": origin_voxel,
    "plot": plot_data,
}


def check_voxel(mask_index, ref_voxel, datarange):
    """
    Check if the voxel centrosymmetric to ref_voxel belongs also to the datarange.

    :param mask_index: tuple of three integers, indices of the masked voxel
    :param ref_voxel: tuple of three integers, indices of the origin of reciprocal space
    :param datarange: tuple of six integers (z_start, z_stop, y_start, y_stop, x_tart,
     x_stop) representing the range of valid indices
    :return: tuple (boolean, mask_index, sym_index) where boolean is True if the
     centrosymmetric voxel belongs to the datarange and sym_index is a tuple of three
     integers representing it's indices.
    """
    # calculate the position of the centrosymmetric voxel
    sym_z, sym_y, sym_x = (
        2 * ref_voxel[0] - mask_index[0],
        2 * ref_voxel[1] - mask_index[1],
        2 * ref_voxel[2] - mask_index[2],
    )

    # check if this voxel is masked. Copy its intensity if not.
    if util.in_range(point=(sym_z, sym_y, sym_x), extent=datarange):
        return True, mask_index, (sym_z, sym_y, sym_x)
    return False, None, None


def main(parameters):
    """
    Protection for multiprocessing.

    :param parameters: dictionnary containing input parameters
    """

    def collect_result(result):
        """
        Callback processing the result after asynchronous multiprocessing.

        Update the global arrays.

        :param result: tuple output of check_voxel, (boolean, masked voxel indices,
         centrosymmetric voxel indices). The boolean will be True if the
         centrosymmetrix voxel intensity can be used
        """
        nonlocal data, mask, current_point, nb_points
        current_point += 1
        if result[0] and not mask[result[2][0], result[2][1], result[2][2]]:
            data[result[1][0], result[1][1], result[1][2]] = data[
                result[2][0], result[2][1], result[2][2]
            ]
            mask[result[1][0], result[1][1], result[1][2]] = 0
        if (current_point % 10000) == 0:
            sys.stdout.write(f"\rPoint {current_point:d} / {nb_points:d},")
            sys.stdout.flush()

    ######################################
    # load the dictionnary of parameters #
    ######################################
    datadir = parameters["datadir"]
    savedir = parameters["savedir"]
    comment = parameters["comment"]
    origin = parameters["origin"]
    plot = parameters["plot"]

    ###################################
    # load experimental data and mask #
    ###################################
    plt.ion()
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        initialdir=datadir,
        title="Select the diffraction pattern",
        filetypes=[("NPZ", "*.npz"), ("NPY", "*.npy")],
    )
    data, _ = util.load_file(file_path)
    data = data.astype(float)

    file_path = filedialog.askopenfilename(
        initialdir=datadir,
        title="Select the mask",
        filetypes=[("NPZ", "*.npz"), ("NPY", "*.npy")],
    )
    mask, _ = util.load_file(file_path)
    mask = mask.astype(int)

    #########################
    # check some parameters #
    #########################
    savedir = savedir or datadir

    if data.shape != mask.shape:
        raise ValueError(
            f"Incompatible shape for the data: {data.shape} and the mask: {mask.shape}"
        )

    if data.ndim != 3:
        raise ValueError("only 3D data is supported")

    valid.valid_container(
        obj=origin,
        container_types=(tuple, list, np.ndarray),
        item_types=int,
        length=3,
        name="interpolate_cdi.py",
    )

    nbz, nby, nbx = data.shape
    # calculate the range of pixels indices covered by the data,
    # taking into account the origin of reciprocal space
    data_extent = (0, nbz - 1, 0, nby - 1, 0, nbx - 1)
    print(f"data shape: {data.shape}")
    print(f"origin of reciprocal space: {origin}")
    print(f"data extent: {data_extent}")

    ###################################################
    # plot the data and mask before the interpolation #
    ###################################################
    if plot:
        gu.multislices_plot(
            array=data,
            sum_frames=True,
            plot_colorbar=True,
            scale="log",
            slice_position=origin,
            is_orthogonal=True,
            reciprocal_space=True,
            vmin=0,
            title="data before interpolation",
        )
        gu.multislices_plot(
            array=mask,
            sum_frames=True,
            plot_colorbar=False,
            scale="linear",
            slice_position=origin,
            is_orthogonal=True,
            reciprocal_space=True,
            vmin=0,
            title="mask before interpolation",
        )

    ##############################################################
    # loop over masked points to see if the centrosymmetric      #
    # voxel is also masked, if not copy its intensity and unmask #
    ##############################################################
    ind_z, ind_y, ind_x = np.nonzero(
        mask
    )  # np.nonzero returns a tuple of three 1D arrays
    current_point = 0
    nb_points = len(ind_z)
    print(f"\nnumber of masked points before interpolation: {nb_points}")

    print(f"number of processors used: {mp.cpu_count()}")
    mp.freeze_support()
    pool = mp.Pool(processes=mp.cpu_count())  # use this number of processes
    for idx in range(nb_points):
        pool.apply_async(
            check_voxel,
            args=((ind_z[idx], ind_y[idx], ind_x[idx]), origin, data_extent),
            callback=collect_result,
            error_callback=util.catch_error,
        )

    pool.close()
    pool.join()  # postpones the execution of next line of code until
    # all processes in the queue are done.
    print(f"\nnumber of masked points after interpolation: {len(np.nonzero(mask)[0])}")

    ##################################################
    # plot the data and mask after the interpolation #
    ##################################################
    gu.multislices_plot(
        array=data,
        sum_frames=True,
        plot_colorbar=True,
        scale="log",
        slice_position=origin,
        is_orthogonal=True,
        reciprocal_space=True,
        vmin=0,
        title="data after interpolation",
    )
    plt.savefig(savedir + "centrosym_finalsum" + comment + ".png")
    gu.multislices_plot(
        array=mask,
        sum_frames=True,
        plot_colorbar=False,
        scale="linear",
        slice_position=origin,
        is_orthogonal=True,
        reciprocal_space=True,
        vmin=0,
        title="mask after interpolation",
    )
    plt.savefig(savedir + "centrosym_finalmask" + comment + ".png")

    ##################################
    # save the updated data and mask #
    ##################################
    print("\nSaving directory:", savedir)
    print("Data type before saving:", data.dtype)
    print("Mask type before saving:", mask.dtype)
    np.savez_compressed(savedir + "centrosym_data" + comment, data=data)
    np.savez_compressed(savedir + "centrosym_mask" + comment, mask=mask)
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main(parameters=params)
