# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Steven Leake, steven.leake@esrf.fr
#         makePoly commands adapted from Ross Harder scripts decades old
"""Make a 3D polygon."""

# The concept is to be able to build a support from a set of defined planes these
# planes can be positioned based on physical size (nm) if
# known or eventually made into a fancy tool with a 3D view of the support versus
# the data so you can match fringes. It may also be interesting to consider how to
# make 3D supports from other characterisation methods (SEM etc...).

import sys

import numpy as np


def AddPolyCen(array, center, planes):
    """

    Make the polygon.

    :param array:  input array
    :param center: origin of polygon
    :param planes: array of planes

    """
    dims = array.shape

    griddims = []
    for d in dims:
        griddims.append(slice(0, d))
    grid = np.ogrid[griddims]

    for plane in planes:
        sum1 = np.zeros(dims)
        sum2 = 0
        for d in range(len(dims)):
            sum1 += plane[d] * (grid[d] - center[d])
            sum2 += plane[d] ** 2
        array += (sum1 <= sum2) * 1

    return ((array >= len(planes)) * 1).astype(array.dtype)


def MakePoly(dims, planes):
    """

    Make a polygon.

    :param dims: dimensions of array in pixels
    :param planes: array of planes

    """
    return make_poly(dims, planes)


def make_poly(dims, planes):
    """

    Make a polygon.

    :param dims: dimensions of array in pixels
    :param planes: array of planes

    """
    cen = []
    array = np.zeros(dims)
    for dim in dims:
        cen.append(dim / 2)
    return AddPolyCen(array, cen, planes)


def MakePolyCen(dims, center, planes):
    """

    Make the polygon.

    :param array: input array
    :param center: origin of polygon
    :param planes: array of planes

    """
    array = np.zeros(dims)
    return AddPolyCen(array, center, planes)


class supportMaker:
    """

    A masking class for support creation.

    :param rawdata: raw experimental data
    :param wavelength: x-ray wavelength
    :param detector_distance: sample to detector (m)
    :param detector_pixel_size: detector pixel size - 2D (m)
    :param ang_step: angular step (degrees)
    :param braggAng: bragg angle (degrees)
    :param planes: array of planes np.array([x,y,z])
    :param planesDist: array of plane distance to origin (m)
    :param voxel_size: set the voxel size to some arbitrary size,
             np.array([x,y,z]) (m)

    """

    def __init__(
        self,
        rawdata,
        wavelength=None,
        detector_distance=None,
        detector_pixel_size=None,
        ang_step=None,
        braggAng=None,
        planes=None,
        planesDist=None,
        voxel_size=None,
    ):
        # set all parameters

        self.rawdata = rawdata
        self.wavelength = wavelength
        self.detDist = detector_distance
        self.detector_pixel_size = detector_pixel_size
        self.braggAng = braggAng
        self.ang_step = ang_step

        # create the support
        self.set_support(rawdata)
        if voxel_size is None:
            self.calc_voxel_size(
                wavelength, detector_distance, detector_pixel_size, ang_step, braggAng
            )
        else:
            self.set_voxel_size(voxel_size)

        print("voxel_size: ", self.vox_size)
        self.set_planes(planes, planesDist)
        self.support = MakePoly(self.support.shape, self.scaled_planes)

    def set_support(self, rawdata):
        """Set the support to match rawdata dimensions."""
        self.support = np.zeros_like(rawdata)

    def set_voxel_size(self, voxel_size):
        """Set the voxel size."""
        self.vox_size = voxel_size

    def set_planes(self, planes_list, plane_distance_origin_list):
        """Set the planes."""
        if len(planes_list) != len(plane_distance_origin_list):
            print("the number of planes does not match the number of distances.")
            sys.exit()

        self.planes = np.array(planes_list)
        self.planesDist = np.array(plane_distance_origin_list)

        # based on voxel size - scale distance to plane metres to pixels
        # convert existing plane distance
        d_pix = np.sqrt(np.sum((self.planes) ** 2, axis=1))
        print("\nD_pix", d_pix)

        # convert existing plane distance to metres to user defined size
        # with a scalefactor
        d_m = np.sqrt(np.sum((self.vox_size * self.planes) ** 2, axis=1))
        print("\nDM", d_m)

        sf = (self.planesDist.reshape(1, self.planes.shape[0]) / d_m).reshape(
            self.planes.shape[0], 1
        )
        self.scaled_planes = self.planes * sf

        print(self.planesDist.reshape(1, self.planes.shape[0]))
        print("\n###", self.scaled_planes)
        print("\n### scale factor:", sf)
        print("\n###", self.planes)
        print("\n###", self.rawdata.shape)

    def get_support(
        self,
    ):
        """Return the generated support."""
        return self.support

    def get_planes(
        self,
    ):
        """Return planes array."""
        return self.planes

    def get_planesDist(
        self,
    ):
        """Return planes distance from origin."""
        return self.planesDist

    def calc_voxel_size(
        self,
        wavelength=None,
        detDist=None,
        pixel_size=None,
        ang_step=None,
        braggAng=None,
    ):
        """Calculate the voxel size."""
        # use the experiment parameters to determine the voxel size
        ss = self.support.shape

        # calculate the angular dimension first
        q = 4 * np.pi * np.sin(np.deg2rad(braggAng)) / wavelength
        deltaQ = q * np.deg2rad(ang_step) * ss[0]
        a = np.pi * 2 / deltaQ

        # add the detector dimensions - don't forget to reverse -
        # but pixels tend to be symmetric
        pixel_size.reverse()
        self.vox_size = np.r_[
            a, wavelength * detDist / (np.array(ss[1:]) * np.array(pixel_size))
        ]

        print("Voxel dimensions: ", self.vox_size * 1e9, " (nm)")


def generatePlanesCuboid(x, y, z):
    """Make a cuboid of size x,y,z."""
    planes = np.array(
        [
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ]
    )
    planesDist = np.array(
        [
            [x / 2],
            [x / 2],
            [y / 2],
            [y / 2],
            [z / 2],
            [z / 2],
        ]
    )
    return planes, planesDist


def generatePlanesTetrahedra(x):
    """Make a tetrahedra of dimension x."""
    planes = np.array(
        [
            [1, 1, 1],
            [-1, 1, -1],
            [1, -1, -1],
            [-1, -1, 1],
        ]
    )
    planesDist = np.array(
        [
            [x],
            [x],
            [x],
            [x],
        ]
    )
    return planes, planesDist


def generatePlanesPrism(x, y):
    """Make a Prism of thickness x, y is somewhat arbitrary."""
    planes = np.array(
        [
            [-1, np.sqrt(3) / 2.0, 0],
            [1, np.sqrt(3) / 2.0, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ]
    )
    planesDist = np.array(
        [
            [y],
            [y],
            [y],
            [x],
            [x],
        ]
    )
    return planes, planesDist


def rot_planes(planes, rot):
    """
    Rotate planes with some rotation matrix.

    something is wrong here if I import this it doesnt work
    """
    # should probably move to the rotation matrix module
    print(planes)
    rp = [np.dot(rot, v) for v in planes]
    npl = []
    npl = [npl.append(p.tolist()) for p in rp]
    planes = np.array(npl)
    print(planes)
    return planes
