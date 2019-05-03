# -*- coding: utf-8 -*-
"""
plane_angle.py
Calculate the angle between to crystallographic planes in cubic materials
@author: CARNIS
"""
import numpy as np


def plane_angle(ref_plane, plane):
    """
    Calculate the angle between two crystallographic planes in cubic materials
    :param ref_plane: measured reflection
    :param plane: plane for which angle should be calculated
    :return: the angle in degrees
    """
    if np.array_equal(ref_plane, plane):
        my_angle = 0.0
    else:
        my_angle = 180/np.pi*np.arccos(sum(np.multiply(ref_plane, plane)) /
                                       (np.linalg.norm(ref_plane)*np.linalg.norm(plane)))
    # if my_angle > 90.0:
    #     my_angle = 180.0 - my_angle
    return my_angle


reference_plane = [1, 1, 1]  # [0.37182, 0.78376, -0.49747]  # [0.40975, 0.29201, -0.86420]
second_plane = [1, -1, 1]  # [-0.22923, 0.76727, -0.59896]  # [-0.19695, 0.27933, -0.93978]
angle = plane_angle(reference_plane, second_plane)
print('angle=', str(angle))