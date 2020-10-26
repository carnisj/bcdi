# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np
import bcdi.graph.graph_utils as gu
from scipy.interpolate import RegularGridInterpolator
import gc


class SetupPostprocessing(object):
    """
    Class to handle the experimental geometry for postprocessing.
    """
    def __init__(self, beamline, energy, outofplane_angle, inplane_angle, tilt_angle, rocking_angle, distance,
                 grazing_angle=0, pixel_x=55e-6, pixel_y=55e-6):
        """
        Initialize parameters of the experiment.

        :param beamline: name of the beamline: 'ID01', 'SIXS_2018', 'SIXS_2019', '34ID', 'P10', 'CRISTAL', 'NANOMAX'
        :param energy: X-ray energy in eV
        :param outofplane_angle: out of plane angle of the detector in degrees
        :param inplane_angle: inplane angle of the detector in degrees
        :param tilt_angle: angular step of the sample during the rocking curve, in degrees
        :param rocking_angle: name of the angle which is tilted during the rocking curve, 'outofplane' or 'inplane'
        :param distance: sample to detector distance in meters
        :param grazing_angle: grazing angle for in-plane rocking curves (eta ID01, th 34ID, beta SIXS)
        :param pixel_x: horizontal pixel size, in meters
        :param pixel_y: vertical pixel size, in meters
        """
        self.beamline = beamline  # string
        self.energy = energy  # in eV
        self.wavelength = 12.398 * 1e-7 / energy  # in m
        self.outofplane_angle = outofplane_angle  # in degrees
        self.inplane_angle = inplane_angle  # in degrees
        self.tilt_angle = tilt_angle  # in degrees
        self.rocking_angle = rocking_angle  # string
        self.grazing_angle = grazing_angle  # string
        self.distance = distance  # in meters
        self.pixel_x = pixel_x  # in meters
        self.pixel_y = pixel_y  # in meters

        #############################################################
        # detector orientation convention depending on the beamline #
        #############################################################
        # the frame convention is the one of xrayutilities: x downstream, y outboard, z vertical up

        # horizontal axis:
        if beamline in ['ID01', 'SIXS_2018', 'SIXS_2019', 'CRISTAL', 'NANOMAX']:
            # we look at the detector from downstream
            self.detector_hor = 'y+'
        else:  # 'P10', '34ID'
            # we look at the detector from upstream
            self.detector_hor = 'y-'

        # vertical axis:
        if beamline in ['NANOMAX']:
            # detector in flip upside-down on the robot arm at Nanomax, the origin is at the bottom
            self.detector_ver = 'z+'
        else:
            # origin is at the top
            self.detector_ver = 'z-'

    def __repr__(self):
        """
        :return: a nicely formatted representation string
        """
        return f"{self.__class__.__name__}: beamline={self.beamline}, energy={self.energy}eV," \
               f" sample to detector distance={self.distance}m, pixel size (VxH)=({self.pixel_y},{self.pixel_x})"

    def detector_frame(self, obj, voxelsize, width_z=None, width_y=None, width_x=None,
                       debugging=False, **kwargs):
        """
        Interpolate the orthogonal object back into the non-orthogonal detector frame

        :param obj: real space object, in the orthogonal laboratory frame
        :param voxelsize: voxel size of the original object
        :param width_z: size of the area to plot in z (axis 0), centered on the middle of the initial array
        :param width_y: size of the area to plot in y (axis 1), centered on the middle of the initial array
        :param width_x: size of the area to plot in x (axis 2), centered on the middle of the initial array
        :param debugging: True to show plots before and after interpolation
        :param kwargs:
         - 'title': title for the debugging plots
        :return: object interpolated on an orthogonal grid
        """
        # default values for kwargs
        title = 'Object'

        for k in kwargs.keys():
            if k in ['title']:
                title = kwargs['title']
            else:
                raise Exception("unknown keyword argument given: allowed is 'title'")

        nbz, nby, nbx = obj.shape

        if debugging:
            gu.multislices_plot(abs(obj), sum_frames=True, width_z=width_z, width_y=width_y, width_x=width_x,
                                title=title + ' before interpolation\n')

        ortho_matrix = self.update_coords(array_shape=(nbz, nby, nbx), tilt_angle=self.tilt_angle,
                                          pixel_x=self.pixel_x, pixel_y=self.pixel_y)

        ################################################
        # interpolate the data into the detector frame #
        ################################################
        myz, myy, myx = np.meshgrid(np.arange(-nbz // 2, nbz // 2, 1),
                                    np.arange(-nby // 2, nby // 2, 1),
                                    np.arange(-nbx // 2, nbx // 2, 1), indexing='ij')

        new_x = ortho_matrix[0, 0] * myx + ortho_matrix[0, 1] * myy + ortho_matrix[0, 2] * myz
        new_y = ortho_matrix[1, 0] * myx + ortho_matrix[1, 1] * myy + ortho_matrix[1, 2] * myz
        new_z = ortho_matrix[2, 0] * myx + ortho_matrix[2, 1] * myy + ortho_matrix[2, 2] * myz
        del myx, myy, myz
        # la partie rgi est sure: c'est la taille de l'objet orthogonal de depart
        rgi = RegularGridInterpolator((np.arange(-nbz // 2, nbz // 2) * voxelsize,
                                       np.arange(-nby // 2, nby // 2) * voxelsize,
                                       np.arange(-nbx // 2, nbx // 2) * voxelsize),
                                      obj, method='linear', bounds_error=False, fill_value=0)
        detector_obj = rgi(np.concatenate((new_z.reshape((1, new_z.size)), new_y.reshape((1, new_z.size)),
                                           new_x.reshape((1, new_z.size)))).transpose())
        detector_obj = detector_obj.reshape((nbz, nby, nbx)).astype(obj.dtype)

        if debugging:
            gu.multislices_plot(abs(detector_obj), sum_frames=True, width_z=width_z, width_y=width_y, width_x=width_x,
                                title=title + ' interpolated in detector frame\n')

        return detector_obj

    def exit_wavevector(self):
        """
        Calculate the exit wavevector kout depending on the setup parameters, in laboratory frame (z downstream,
         y vertical, x outboard).

        :return: kout vector
        """
        if self.beamline == 'SIXS_2018' or self.beamline == 'SIXS_2019':
            # gamma is anti-clockwise
            kout = 2 * np.pi / self.wavelength * np.array(
                [np.cos(np.pi * self.inplane_angle / 180) * np.cos(np.pi * self.outofplane_angle / 180),  # z
                 np.sin(np.pi * self.outofplane_angle / 180),  # y
                 np.sin(np.pi * self.inplane_angle / 180) * np.cos(np.pi * self.outofplane_angle / 180)])  # x
        elif self.beamline == 'ID01':
            # nu is clockwise
            kout = 2 * np.pi / self.wavelength * np.array(
                [np.cos(np.pi * self.inplane_angle / 180) * np.cos(np.pi * self.outofplane_angle / 180),  # z
                 np.sin(np.pi * self.outofplane_angle / 180),  # y
                 -np.sin(np.pi * self.inplane_angle / 180) * np.cos(np.pi * self.outofplane_angle / 180)])  # x
        elif self.beamline == '34ID':
            # gamma is anti-clockwise
            kout = 2 * np.pi / self.wavelength * np.array(
                [np.cos(np.pi * self.inplane_angle / 180) * np.cos(np.pi * self.outofplane_angle / 180),  # z
                 np.sin(np.pi * self.outofplane_angle / 180),  # y
                 np.sin(np.pi * self.inplane_angle / 180) * np.cos(np.pi * self.outofplane_angle / 180)])  # x
        elif self.beamline == 'NANOMAX':
            # gamma is clockwise
            kout = 2 * np.pi / self.wavelength * np.array(
                [np.cos(np.pi * self.inplane_angle / 180) * np.cos(np.pi * self.outofplane_angle / 180),  # z
                 np.sin(np.pi * self.outofplane_angle / 180),  # y
                 -np.sin(np.pi * self.inplane_angle / 180) * np.cos(np.pi * self.outofplane_angle / 180)])  # x
        elif self.beamline == 'P10':
            # gamma is anti-clockwise
            kout = 2 * np.pi / self.wavelength * np.array(
                [np.cos(np.pi * self.inplane_angle / 180) * np.cos(np.pi * self.outofplane_angle / 180),  # z
                 np.sin(np.pi * self.outofplane_angle / 180),  # y
                 np.sin(np.pi * self.inplane_angle / 180) * np.cos(np.pi * self.outofplane_angle / 180)])  # x
        elif self.beamline == 'CRISTAL':
            # gamma is anti-clockwise
            kout = 2 * np.pi / self.wavelength * np.array(
                [np.cos(np.pi * self.inplane_angle / 180) * np.cos(np.pi * self.outofplane_angle / 180),  # z
                 np.sin(np.pi * self.outofplane_angle / 180),  # y
                 np.sin(np.pi * self.inplane_angle / 180) * np.cos(np.pi * self.outofplane_angle / 180)])  # x
        else:
            raise ValueError('setup parameter: ', self.beamline, 'not defined')
        return kout

    def inplane_coeff(self):
        """
        Define a coefficient +/- 1 depending on the detector inplane rotation direction and the detector inplane
         orientation. The frame convention is the one of xrayutilities: x downstream, y outboard, z vertical up.
         See postprocessing/scripts/correct_angles_detector.py for an example.

        :return: +1 or -1
        """
        if self.detector_hor == 'y+':
            hor_coeff = 1
        else:  # 'y-'
            hor_coeff = -1

        if self.beamline == 'SIXS_2018' or self.beamline == 'SIXS_2019':
            # gamma is anti-clockwise, we see the detector from downstream
            coeff_inplane = 1 * hor_coeff
        elif self.beamline == 'ID01':
            # nu is clockwise, we see the detector from downstream
            coeff_inplane = -1 * hor_coeff
        elif self.beamline == '34ID':
            # delta is anti-clockwise, we see the detector from the front
            coeff_inplane = 1 * hor_coeff
        elif self.beamline == 'P10':
            # gamma is anti-clockwise, we see the detector from the front
            coeff_inplane = 1 * hor_coeff
        elif self.beamline == 'CRISTAL':
            # gamma is anti-clockwise, we see the detector from downstream
            coeff_inplane = 1 * hor_coeff
        elif self.beamline == 'NANOMAX':
            # gamma is clockwise, we see the detector from downstream
            coeff_inplane = -1 * hor_coeff
        else:
            raise ValueError('setup parameter: ', self.beamline, 'not defined')
        return coeff_inplane

    def orthogonalize(self, obj, initial_shape=(), voxel_size=np.nan, width_z=None, width_y=None,
                      width_x=None, verbose=True, debugging=False, **kwargs):
        """
        Interpolate obj on the orthogonal reference frame defined by the setup.

        :param obj: real space object, in a non-orthogonal frame (output of phasing program)
        :param initial_shape: shape of the FFT used for phasing
        :param voxel_size: user-defined voxel size, in nm
        :param width_z: size of the area to plot in z (axis 0), centered on the middle of the initial array
        :param width_y: size of the area to plot in y (axis 1), centered on the middle of the initial array
        :param width_x: size of the area to plot in x (axis 2), centered on the middle of the initial array
        :param verbose: True to have printed comments
        :param debugging: True to show plots before and after interpolation
        :param kwargs:
         - 'title': title for the debugging plots
        :return: object interpolated on an orthogonal grid
        """
        # default values for kwargs
        title = 'Object'

        for k in kwargs.keys():
            if k in ['title']:
                title = kwargs['title']
            else:
                raise Exception("unknown keyword argument given: allowed is 'title'")

        if len(initial_shape) == 0:
            initial_shape = obj.shape

        if debugging:
            gu.multislices_plot(abs(obj), sum_frames=True, width_z=width_z, width_y=width_y, width_x=width_x,
                                title=title+' in detector frame')

        # estimate the direct space voxel sizes in nm based on the FFT window shape used in phase retrieval
        dz_realspace, dy_realspace, dx_realspace = self.voxel_sizes(initial_shape, tilt_angle=abs(self.tilt_angle),
                                                                    pixel_x=self.pixel_x, pixel_y=self.pixel_y)

        if verbose:
            print('Direct space voxel sizes (z, y, x) based on initial FFT shape: (',
                  str('{:.2f}'.format(dz_realspace)), 'nm,',
                  str('{:.2f}'.format(dy_realspace)), 'nm,',
                  str('{:.2f}'.format(dx_realspace)), 'nm )')

        nbz, nby, nbx = obj.shape  # could be smaller if the object was cropped around the support
        if nbz != initial_shape[0] or nby != initial_shape[1] or nbx != initial_shape[2]:
            # recalculate the tilt and pixel sizes to accomodate a shape change
            tilt = self.tilt_angle * initial_shape[0] / nbz
            pixel_y = self.pixel_y * initial_shape[1] / nby
            pixel_x = self.pixel_x * initial_shape[2] / nbx
            if verbose:
                print('Tilt, pixel_y, pixel_x based on cropped array shape: (',
                      str('{:.4f}'.format(tilt)), 'deg,',
                      str('{:.2f}'.format(pixel_y * 1e6)), 'um,',
                      str('{:.2f}'.format(pixel_x * 1e6)), 'um)')

            # sanity check, the direct space voxel sizes calculated below should be equal to the original ones
            dz_realspace, dy_realspace, dx_realspace = self.voxel_sizes((nbz, nby, nbx),
                                                                        tilt_angle=abs(tilt),
                                                                        pixel_x=pixel_x, pixel_y=pixel_y)
            if verbose:
                print('Sanity check, recalculated direct space voxel sizes: (',
                      str('{:.2f}'.format(dz_realspace)), ' nm,',
                      str('{:.2f}'.format(dy_realspace)), 'nm,',
                      str('{:.2f}'.format(dx_realspace)), 'nm )')
        else:
            tilt = self.tilt_angle
            pixel_y = self.pixel_y
            pixel_x = self.pixel_x

        if np.isnan(voxel_size):
            voxel = np.mean([dz_realspace, dy_realspace, dx_realspace])  # in nm
        else:
            voxel = voxel_size

        ortho_matrix = self.update_coords(array_shape=(nbz, nby, nbx), tilt_angle=tilt,
                                          pixel_x=pixel_x, pixel_y=pixel_y)

        ###############################################################
        # Vincent Favre-Nicolin's method using inverse transformation #
        ###############################################################
        myz, myy, myx = np.meshgrid(np.arange(-nbz // 2, nbz // 2, 1) * voxel,
                                    np.arange(-nby // 2, nby // 2, 1) * voxel,
                                    np.arange(-nbx // 2, nbx // 2, 1) * voxel, indexing='ij')

        # ortho_matrix is the transformation matrix from the detector coordinates to the laboratory frame
        # in RGI, we want to calculate the coordinates that would have a grid of the laboratory frame expressed in the
        # detector frame, i.e. one has to inverse the transformation matrix.
        ortho_imatrix = np.linalg.inv(ortho_matrix)
        new_x = ortho_imatrix[0, 0] * myx + ortho_imatrix[0, 1] * myy + ortho_imatrix[0, 2] * myz
        new_y = ortho_imatrix[1, 0] * myx + ortho_imatrix[1, 1] * myy + ortho_imatrix[1, 2] * myz
        new_z = ortho_imatrix[2, 0] * myx + ortho_imatrix[2, 1] * myy + ortho_imatrix[2, 2] * myz
        del myx, myy, myz
        gc.collect()

        rgi = RegularGridInterpolator((np.arange(-nbz // 2, nbz // 2), np.arange(-nby // 2, nby // 2),
                                       np.arange(-nbx // 2, nbx // 2)), obj, method='linear',
                                      bounds_error=False, fill_value=0)
        ortho_obj = rgi(np.concatenate((new_z.reshape((1, new_z.size)), new_y.reshape((1, new_z.size)),
                                        new_x.reshape((1, new_z.size)))).transpose())
        ortho_obj = ortho_obj.reshape((nbz, nby, nbx)).astype(obj.dtype)

        if debugging:
            gu.multislices_plot(abs(ortho_obj), sum_frames=True, width_z=width_z, width_y=width_y, width_x=width_x,
                                title=title+' in the orthogonal laboratory frame')
        return ortho_obj, voxel

    def orthogonalize_vector(self, vector, array_shape, tilt_angle, pixel_x, pixel_y):
        """
        Calculate the direct space voxel sizes in the laboratory frame (z downstream, y vertical up, x outboard).

        :param vector: tuple of 3 coordinates, vector to be transformed in the detector frame
        :param array_shape: shape of the 3D array to orthogonalize
        :param tilt_angle: angular step during the rocking curve, in degrees
        :param pixel_x: horizontal pixel size, in meters
        :param pixel_y: vertical pixel size, in meters
        :return: the direct space voxel sizes in nm, in the laboratory frame (voxel_z, voxel_y, voxel_x)
        """
        ortho_matrix = self.update_coords(array_shape=array_shape, tilt_angle=tilt_angle,
                                          pixel_x=pixel_x, pixel_y=pixel_y)
        # ortho_matrix is the transformation matrix from the detector coordinates to the laboratory frame
        # Here, we want to calculate the coordinates that would have a vector of the laboratory frame expressed in the
        # detector frame, i.e. one has to inverse the transformation matrix.
        ortho_imatrix = np.linalg.inv(ortho_matrix)
        new_x = ortho_imatrix[0, 0] * vector[2] + ortho_imatrix[0, 1] * vector[1] + ortho_imatrix[0, 2] * vector[0]
        new_y = ortho_imatrix[1, 0] * vector[2] + ortho_imatrix[1, 1] * vector[1] + ortho_imatrix[1, 2] * vector[0]
        new_z = ortho_imatrix[2, 0] * vector[2] + ortho_imatrix[2, 1] * vector[1] + ortho_imatrix[2, 2] * vector[0]
        return new_z, new_y, new_x

    def outofplane_coeff(self):
        """
        Define a coefficient +/- 1 depending on the detector out of plane rotation direction and the detector out of
         plane orientation. The frame convention is the one of xrayutilities: x downstream, y outboard, z vertical up.
         See postprocessing/scripts/correct_angles_detector.py for an example.

        :return: +1 or -1
        """
        if self.detector_ver == 'z+':  # origin of pixels at the bottom
            ver_coeff = 1
        else:  # 'z-'  origin of pixels at the top
            ver_coeff = -1

        # the out of plane detector rotation is clockwise for all beamlines
        coeff_outofplane = -1 * ver_coeff

        return coeff_outofplane

    def update_coords(self, array_shape, tilt_angle, pixel_x, pixel_y):
        """
        Calculate the pixel non-orthogonal coordinates in the orthogonal reference frame.

        :param array_shape: shape of the 3D array to orthogonalize
        :param tilt_angle: angular step during the rocking curve, in degrees
        :param pixel_x: horizontal pixel size, in meters
        :param pixel_y: vertical pixel size, in meters
        :return: the transfer matrix from the detector frame to the laboratory frame
        """
        wavelength = self.wavelength * 1e9  # convert to nm
        distance = self.distance * 1e9  # convert to nm
        pixel_x = pixel_x * 1e9  # convert to nm
        pixel_y = pixel_y * 1e9  # convert to nm
        outofplane = np.radians(self.outofplane_angle)
        inplane = np.radians(self.inplane_angle)
        mygrazing_angle = np.radians(self.grazing_angle)
        lambdaz = wavelength * distance
        mymatrix = np.zeros((3, 3))
        tilt = np.radians(tilt_angle)

        nbz, nby, nbx = array_shape

        if self.beamline == 'ID01':
            print('using ESRF ID01 PSIC geometry')
            if self.rocking_angle == "outofplane":
                print('rocking angle is eta')
                # rocking eta angle clockwise around x (phi does not matter, above eta)
                mymatrix[:, 0] = 2 * np.pi * nbx / lambdaz * np.array([pixel_x * np.cos(inplane),
                                                                       0,
                                                                       pixel_x * np.sin(inplane)])
                mymatrix[:, 1] = 2 * np.pi * nby / lambdaz * np.array([-pixel_y * np.sin(inplane) * np.sin(outofplane),
                                                                       -pixel_y * np.cos(outofplane),
                                                                       pixel_y * np.cos(inplane) * np.sin(outofplane)])
                mymatrix[:, 2] = 2 * np.pi * nbz / lambdaz * np.array([0,
                                                                       tilt * distance * (1 - np.cos(inplane) * np.cos(
                                                                           outofplane)),
                                                                       tilt * distance * np.sin(outofplane)])
            elif self.rocking_angle == "inplane" and mygrazing_angle == 0:
                print('rocking angle is phi, eta=0')
                # rocking phi angle clockwise around y, assuming incident angle eta is zero (eta below phi)
                mymatrix[:, 0] = 2 * np.pi * nbx / lambdaz * np.array([pixel_x * np.cos(inplane),
                                                                       0,
                                                                       pixel_x * np.sin(inplane)])
                mymatrix[:, 1] = 2 * np.pi * nby / lambdaz * np.array([-pixel_y * np.sin(inplane) * np.sin(outofplane),
                                                                       -pixel_y * np.cos(outofplane),
                                                                       pixel_y * np.cos(inplane) * np.sin(outofplane)])
                mymatrix[:, 2] = 2 * np.pi * nbz / lambdaz * np.array(
                    [-tilt * distance * (1 - np.cos(inplane) * np.cos(outofplane)),
                     0,
                     tilt * distance * np.sin(inplane) * np.cos(outofplane)])
            elif self.rocking_angle == "inplane" and mygrazing_angle != 0:
                print('rocking angle is phi, with eta non zero')
                # rocking phi angle clockwise around y, incident angle eta is non zero (eta below phi)
                mymatrix[:, 0] = 2 * np.pi * nbx / lambdaz * np.array([pixel_x * np.cos(inplane),
                                                                       0,
                                                                       pixel_x * np.sin(inplane)])
                mymatrix[:, 1] = 2 * np.pi * nby / lambdaz * np.array([-pixel_y * np.sin(inplane) * np.sin(outofplane),
                                                                       -pixel_y * np.cos(outofplane),
                                                                       pixel_y * np.cos(inplane) * np.sin(outofplane)])
                mymatrix[:, 2] = 2 * np.pi * nbz / lambdaz * tilt * distance * \
                    np.array([(np.sin(mygrazing_angle) * np.sin(outofplane) +
                             np.cos(mygrazing_angle) * (np.cos(inplane) * np.cos(outofplane) - 1)),
                             np.sin(mygrazing_angle) * np.sin(inplane) * np.cos(outofplane),
                             np.cos(mygrazing_angle) * np.sin(inplane) * np.cos(outofplane)])
        if self.beamline == 'P10':
            print('using PETRAIII P10 geometry')
            if self.rocking_angle == "outofplane":
                print('rocking angle is omega')
                # rocking omega angle clockwise around x at mu=0 (phi does not matter, above eta)
                mymatrix[:, 0] = 2 * np.pi * nbx / lambdaz * np.array([-pixel_x * np.cos(inplane),
                                                                       0,
                                                                       pixel_x * np.sin(inplane)])
                mymatrix[:, 1] = 2 * np.pi * nby / lambdaz * np.array([pixel_y * np.sin(inplane) * np.sin(outofplane),
                                                                       -pixel_y * np.cos(outofplane),
                                                                       pixel_y * np.cos(inplane) * np.sin(outofplane)])
                mymatrix[:, 2] = 2 * np.pi * nbz / lambdaz * np.array([0,
                                                                       tilt * distance * (1 - np.cos(inplane) * np.cos(
                                                                           outofplane)),
                                                                       tilt * distance * np.sin(outofplane)])
            elif self.rocking_angle == "inplane" and mygrazing_angle == 0:
                print('rocking angle is phi, omega=0')
                # rocking phi angle clockwise around y, incident angle omega is zero (omega below phi)
                mymatrix[:, 0] = 2 * np.pi * nbx / lambdaz * np.array([-pixel_x * np.cos(inplane),
                                                                       0,
                                                                       pixel_x * np.sin(inplane)])
                mymatrix[:, 1] = 2 * np.pi * nby / lambdaz * np.array([pixel_y * np.sin(inplane) * np.sin(outofplane),
                                                                       -pixel_y * np.cos(outofplane),
                                                                       pixel_y * np.cos(inplane) * np.sin(outofplane)])
                mymatrix[:, 2] = 2 * np.pi * nbz / lambdaz * np.array(
                    [tilt * distance * (np.cos(inplane) * np.cos(outofplane) - 1),
                     0,
                     - tilt * distance * np.sin(inplane) * np.cos(outofplane)])

            elif self.rocking_angle == "inplane" and mygrazing_angle != 0:
                print('rocking angle is phi, with omega non zero')
                # rocking phi angle clockwise around y, incident angle omega is non zero (omega below phi)
                mymatrix[:, 0] = 2 * np.pi * nbx / lambdaz * np.array([-pixel_x * np.cos(inplane),
                                                                       0,
                                                                       pixel_x * np.sin(inplane)])
                mymatrix[:, 1] = 2 * np.pi * nby / lambdaz * np.array([pixel_y * np.sin(inplane) * np.sin(outofplane),
                                                                       -pixel_y * np.cos(outofplane),
                                                                       pixel_y * np.cos(inplane) * np.sin(outofplane)])
                mymatrix[:, 2] = 2 * np.pi * nbz / lambdaz * tilt * distance * \
                    np.array([(np.sin(mygrazing_angle) * np.sin(outofplane) +
                             np.cos(mygrazing_angle) * (np.cos(inplane) * np.cos(outofplane) - 1)),
                             - np.sin(mygrazing_angle) * np.sin(inplane) * np.cos(outofplane),
                             - np.cos(mygrazing_angle) * np.sin(inplane) * np.cos(outofplane)])

        if self.beamline == 'NANOMAX':
            print('using NANOMAX geometry')
            if self.rocking_angle == "outofplane":
                print('rocking angle is theta')
                # rocking eta angle clockwise around x (phi does not matter, above eta)
                mymatrix[:, 0] = 2 * np.pi * nbx / lambdaz * np.array([pixel_x * np.cos(inplane),
                                                                       0,
                                                                       pixel_x * np.sin(inplane)])
                mymatrix[:, 1] = 2 * np.pi * nby / lambdaz * np.array([pixel_y * np.sin(inplane) * np.sin(outofplane),
                                                                       pixel_y * np.cos(outofplane),
                                                                       -pixel_y * np.cos(inplane) * np.sin(outofplane)])
                mymatrix[:, 2] = 2 * np.pi * nbz / lambdaz * np.array([0,
                                                                       tilt * distance * (1 - np.cos(inplane) * np.cos(
                                                                           outofplane)),
                                                                       tilt * distance * np.sin(outofplane)])
            elif self.rocking_angle == "inplane" and mygrazing_angle == 0:
                print('rocking angle is phi, theta=0')
                # rocking phi angle clockwise around y, assuming incident angle eta is zero (eta below phi)
                mymatrix[:, 0] = 2 * np.pi * nbx / lambdaz * np.array([pixel_x * np.cos(inplane),
                                                                       0,
                                                                       pixel_x * np.sin(inplane)])
                mymatrix[:, 1] = 2 * np.pi * nby / lambdaz * np.array([pixel_y * np.sin(inplane) * np.sin(outofplane),
                                                                       pixel_y * np.cos(outofplane),
                                                                       -pixel_y * np.cos(inplane) * np.sin(outofplane)])
                mymatrix[:, 2] = 2 * np.pi * nbz / lambdaz * np.array(
                    [-tilt * distance * (1 - np.cos(inplane) * np.cos(outofplane)),
                     0,
                     tilt * distance * np.sin(inplane) * np.cos(outofplane)])
            elif self.rocking_angle == "inplane" and mygrazing_angle != 0:
                print('rocking angle is phi, with theta non zero')
                # rocking phi angle clockwise around y, incident angle eta is non zero (eta below phi)
                mymatrix[:, 0] = 2 * np.pi * nbx / lambdaz * np.array([pixel_x * np.cos(inplane),
                                                                       0,
                                                                       pixel_x * np.sin(inplane)])
                mymatrix[:, 1] = 2 * np.pi * nby / lambdaz * np.array([pixel_y * np.sin(inplane) * np.sin(outofplane),
                                                                       pixel_y * np.cos(outofplane),
                                                                       -pixel_y * np.cos(inplane) * np.sin(outofplane)])
                mymatrix[:, 2] = 2 * np.pi * nbz / lambdaz * tilt * distance * \
                    np.array([(np.sin(mygrazing_angle) * np.sin(outofplane) +
                               np.cos(mygrazing_angle) * (np.cos(inplane) * np.cos(outofplane) - 1)),
                              np.sin(mygrazing_angle) * np.sin(inplane) * np.cos(outofplane),
                              np.cos(mygrazing_angle) * np.sin(inplane) * np.cos(outofplane)])

        if self.beamline == '34ID':
            print('using APS 34ID geometry')
            if self.rocking_angle == "outofplane":
                print('rocking angle is phi')
                # rocking phi angle anti-clockwise around x (theta does not matter, above phi)
                mymatrix[:, 0] = 2 * np.pi * nbx / lambdaz * np.array([pixel_x * np.cos(inplane),
                                                                       0,
                                                                       -pixel_x * np.sin(inplane)])
                mymatrix[:, 1] = 2 * np.pi * nby / lambdaz * np.array([pixel_y * np.sin(inplane) * np.sin(outofplane),
                                                                       -pixel_y * np.cos(outofplane),
                                                                       pixel_y * np.cos(inplane) * np.sin(outofplane)])
                mymatrix[:, 2] = 2 * np.pi * nbz / lambdaz * np.array([0,
                                                                       -tilt * distance * (1 - np.cos(inplane) * np.cos(
                                                                           outofplane)),
                                                                       -tilt * distance * np.sin(outofplane)])

            elif self.rocking_angle == "inplane" and mygrazing_angle != 0:
                print('rocking angle is theta, with phi non zero')
                # rocking theta angle anti-clockwise around y, incident angle is non zero (theta is above phi)
                mymatrix[:, 0] = 2 * np.pi * nbx / lambdaz * np.array([pixel_x * np.cos(inplane),
                                                                       0,
                                                                       -pixel_x * np.sin(inplane)])
                mymatrix[:, 1] = 2 * np.pi * nby / lambdaz * np.array([pixel_y * np.sin(inplane) * np.sin(outofplane),
                                                                       -pixel_y * np.cos(outofplane),
                                                                       pixel_y * np.cos(inplane) * np.sin(outofplane)])
                mymatrix[:, 2] = 2 * np.pi * nbz / lambdaz * tilt * distance * \
                    np.array([(np.sin(mygrazing_angle) * np.sin(outofplane) +
                              np.cos(mygrazing_angle) * (1 - np.cos(inplane) * np.cos(outofplane))),
                              -np.sin(mygrazing_angle) * np.sin(inplane) * np.cos(outofplane),
                              np.cos(mygrazing_angle) * np.sin(inplane) * np.cos(outofplane)])

            elif self.rocking_angle == "inplane" and mygrazing_angle == 0:
                print('rocking angle is theta, phi=0')
                # rocking theta angle anti-clockwise around y, assuming incident angle is zero (theta is above phi)
                mymatrix[:, 0] = 2 * np.pi * nbx / lambdaz * np.array([pixel_x * np.cos(inplane),
                                                                       0,
                                                                       -pixel_x * np.sin(inplane)])
                mymatrix[:, 1] = 2 * np.pi * nby / lambdaz * np.array([pixel_y * np.sin(inplane) * np.sin(outofplane),
                                                                       -pixel_y * np.cos(outofplane),
                                                                       pixel_y * np.cos(inplane) * np.sin(outofplane)])
                mymatrix[:, 2] = 2 * np.pi * nbz / lambdaz * np.array(
                    [tilt * distance * (1 - np.cos(inplane) * np.cos(outofplane)),
                     0,
                     tilt * distance * np.sin(inplane) * np.cos(outofplane)])
        if self.beamline == 'SIXS_2018' or self.beamline == 'SIXS_2019':
            print('using SIXS geometry')
            if self.rocking_angle == "inplane" and mygrazing_angle != 0:
                print('rocking angle is mu, with beta non zero')
                # rocking mu angle anti-clockwise around y
                mymatrix[:, 0] = 2 * np.pi * nbx / lambdaz * pixel_x * np.array([np.cos(inplane),
                                                                                 -np.sin(mygrazing_angle) * np.sin(
                                                                                     inplane),
                                                                                 -np.cos(mygrazing_angle) * np.sin(
                                                                                     inplane)])
                mymatrix[:, 1] = 2 * np.pi * nby / \
                    lambdaz * pixel_y * np.array([np.sin(inplane) * np.sin(outofplane),
                                                  (np.sin(mygrazing_angle) * np.cos(inplane) * np.sin(outofplane)
                                                   - np.cos(mygrazing_angle) * np.cos(outofplane)),
                                                  (np.cos(mygrazing_angle) * np.cos(inplane) * np.sin(outofplane)
                                                   + np.sin(mygrazing_angle) * np.cos(outofplane))])
                mymatrix[:, 2] = 2 * np.pi * nbz / lambdaz * tilt * distance \
                    * np.array([np.cos(mygrazing_angle) - np.cos(inplane) * np.cos(outofplane),
                                np.sin(mygrazing_angle) * np.sin(inplane) * np.cos(outofplane),
                                np.cos(mygrazing_angle) * np.sin(inplane) * np.cos(outofplane)])

            elif self.rocking_angle == "inplane" and mygrazing_angle == 0:
                print('rocking angle is mu, beta=0')
                # rocking th angle anti-clockwise around y, assuming incident angle is zero (th above tilt)
                mymatrix[:, 0] = 2 * np.pi * nbx / lambdaz * np.array([pixel_x * np.cos(inplane),
                                                                       0,
                                                                       -pixel_x * np.sin(inplane)])
                mymatrix[:, 1] = 2 * np.pi * nby / lambdaz * np.array([pixel_y * np.sin(inplane) * np.sin(outofplane),
                                                                       -pixel_y * np.cos(outofplane),
                                                                       pixel_y * np.cos(inplane) * np.sin(outofplane)])
                mymatrix[:, 2] = 2 * np.pi * nbz / lambdaz * np.array(
                    [tilt * distance * (1 - np.cos(inplane) * np.cos(outofplane)),
                     0,
                     tilt * distance * np.sin(inplane) * np.cos(outofplane)])
        if self.beamline == 'CRISTAL':
            print('using CRISTAL geometry')
            if self.rocking_angle == "outofplane":
                print('rocking angle is komega')
                # rocking tilt angle clockwise around x
                mymatrix[:, 0] = 2 * np.pi * nbx / lambdaz * np.array([pixel_x * np.cos(inplane),
                                                                       0,
                                                                       -pixel_x * np.sin(inplane)])
                mymatrix[:, 1] = 2 * np.pi * nby / lambdaz * np.array([pixel_y * np.sin(inplane) * np.sin(outofplane),
                                                                       -pixel_y * np.cos(outofplane),
                                                                       pixel_y * np.cos(inplane) * np.sin(outofplane)])
                mymatrix[:, 2] = 2 * np.pi * nbz / lambdaz * np.array([0,
                                                                       tilt * distance * (1 - np.cos(inplane) * np.cos(
                                                                           outofplane)),
                                                                       tilt * distance * np.sin(outofplane)])

        transfer_matrix = 2 * np.pi * np.linalg.inv(mymatrix).transpose()
        return transfer_matrix

    def voxel_sizes(self, array_shape, tilt_angle, pixel_x, pixel_y, debug=False):
        """
        Calculate the direct space voxel sizes in the laboratory frame (z downstream, y vertical up, x outboard).

        :param array_shape: shape of the 3D array to orthogonalize
        :param tilt_angle: angular step during the rocking curve, in degrees
        :param pixel_x: horizontal pixel size, in meters
        :param pixel_y: vertical pixel size, in meters
        :param debug: True to have printed comments
        :return: the direct space voxel sizes in nm, in the laboratory frame (voxel_z, voxel_y, voxel_x)
        """
        transfer_matrix = self.update_coords(array_shape=array_shape, tilt_angle=tilt_angle,
                                             pixel_x=pixel_x, pixel_y=pixel_y)
        rec_matrix = 2 * np.pi * np.linalg.inv(transfer_matrix).transpose()
        qx_range = np.linalg.norm(rec_matrix[0, :])
        qy_range = np.linalg.norm(rec_matrix[1, :])
        qz_range = np.linalg.norm(rec_matrix[2, :])
        if debug:
            print('q_range_z, q_range_y, q_range_x=({0:.5f}, {1:.5f}, {2:.5f}) (1/nm)'.format(qz_range, qy_range,
                                                                                              qx_range))
            print('voxelsize_z, voxelsize_y, voxelsize_x='
                  '({0:.2f}, {1:.2f}, {2:.2f}) (1/nm)'.format(2 * np.pi / qz_range, 2 * np.pi / qy_range,
                                                              2 * np.pi / qx_range))
        return 2 * np.pi / qz_range, 2 * np.pi / qy_range, 2 * np.pi / qx_range

    def voxel_sizes_detector(self, array_shape, tilt_angle, pixel_x, pixel_y, debug=False):
        """
        Calculate the direct space voxel sizes in the detector frame
         (z rocking angle, y detector vertical axis, x detector horizontal axis).

        :param array_shape: shape of the 3D array used in phase retrieval
        :param tilt_angle: angular step during the rocking curve, in degrees
        :param pixel_x: horizontal pixel size, in meters
        :param pixel_y: vertical pixel size, in meters
        :param debug: True to have printed comments
        :return: the direct space voxel sizes in nm, in the detector frame (voxel_z, voxel_y, voxel_x)
        """
        voxel_z = self.wavelength / (array_shape[0] * abs(tilt_angle) * np.pi / 180) * 1e9  # in nm
        voxel_y = self.wavelength * self.distance / (array_shape[1] * pixel_y) * 1e9  # in nm
        voxel_x = self.wavelength * self.distance / (array_shape[2] * pixel_x) * 1e9  # in nm
        if debug:
            print('voxelsize_z, voxelsize_y, voxelsize_x='
                  '({0:.2f}, {1:.2f}, {2:.2f}) (1/nm)'.format(voxel_z, voxel_y, voxel_x))
        return voxel_z, voxel_y, voxel_x


class SetupPreprocessing(object):
    """
    Class to handle the experimental geometry for preprocessing.
    """
    def __init__(self, beamline, rocking_angle=None, distance=1, energy=8000, direct_beam=(0, 0),
                 beam_direction=(1, 0, 0), sample_inplane=(1, 0, 0), sample_outofplane=(0, 0, 1),
                 sample_offsets=(0, 0, 0), offset_inplane=0, **kwargs):
        """
        Initialize parameters of the experiment.

        :param beamline: name of the beamline: 'ID01', 'SIXS_2018', 'SIXS_2019', '34ID', 'P10', 'CRISTAL'
        :param rocking_angle: angle which is tilted during the scan. 'outofplane', 'inplane', or 'energy'
        :param distance: sample to detector distance in meters, default=1m
        :param energy: X-ray energy in eV, default=8000eV
        :param direct_beam: tuple describing the position of the direct beam in pixels (vertical, horizontal)
        :param beam_direction: x-ray beam direction
        :param sample_inplane: sample inplane reference direction along the beam at 0 angles
        :param sample_outofplane: surface normal of the sample at 0 angles
        :param sample_offsets: tuple of offsets in degree of the sample around z (downstream), y (vertical up) and x
         (outboard). This corresponds to (chi, phi, incident angle) in a standard diffractometer.
        :param offset_inplane: outer angle offset as defined by xrayutilities detector calibration
        :param kwargs:
         - 'filtered_data' = True when the data is a 3D npy/npz array already cleaned up
         - 'is_orthogonal' = True if 'filtered_data' is already orthogonalized
         - 'custom_scan' = True for a stack of images acquired without scan, (no motor data in the spec file)
         - 'custom_images' = list of image numbers for the custom_scan
         - 'custom_monitor' = list of monitor values for normalization for the custom_scan
         - 'custom_motors' = dictionnary of motors values during the scan
        """
        # default values for kwargs
        filtered_data = False
        is_orthogonal = False
        custom_scan = False
        custom_images = []
        custom_monitor = []
        custom_motors = {}

        for k in kwargs.keys():
            if k in ['filtered_data']:
                filtered_data = kwargs['filtered_data']
            elif k in ['is_orthogonal']:
                is_orthogonal = kwargs['is_orthogonal']
            elif k in ['custom_scan']:
                custom_scan = kwargs['custom_scan']
            elif k in ['custom_images']:
                custom_images = kwargs['custom_images']
            elif k in ['custom_monitor']:
                custom_monitor = kwargs['custom_monitor']
            elif k in ['custom_motors']:
                custom_motors = kwargs['custom_motors']
            else:
                raise Exception("unknown keyword argument given: allowed is"
                                "'custom_images', 'custom_monitor', 'custom_motors'")

        self.beamline = beamline  # string
        self.filtered_data = filtered_data  # boolean
        self.is_orthogonal = is_orthogonal
        self.custom_scan = custom_scan  # boolean
        self.custom_images = custom_images  # list
        self.custom_monitor = custom_monitor  # list
        self.custom_motors = custom_motors  # dictionnary
        self.energy = energy  # in eV
        self.wavelength = 12.398 * 1e-7 / energy  # in m
        self.rocking_angle = rocking_angle  # string
        self.distance = distance  # in meters
        self.direct_beam = direct_beam  # in pixels (vertical, horizontal)
        self.beam_direction = beam_direction  # tuple
        self.sample_inplane = sample_inplane  # tuple
        self.sample_outofplane = sample_outofplane  # tuple
        self.sample_offsets = sample_offsets  # tuple
        self.offset_inplane = offset_inplane  # in degrees

        #############################################################
        # detector orientation convention depending on the beamline #
        #############################################################
        # the frame convention is the one of xrayutilities: x downstream, y outboard, z vertical up

        # horizontal axis:
        if beamline in ['ID01', 'SIXS_2018', 'SIXS_2019', 'CRISTAL', 'NANOMAX']:
            # we look at the detector from downstream
            self.detector_hor = 'y+'
        else:  # 'P10', '34ID'
            # we look at the detector from upstream
            self.detector_hor = 'y-'

        # vertical axis:
        if beamline in ['NANOMAX']:
            # detector in flip upside-down on the robot arm at Nanomax, the origin is at the bottom
            self.detector_ver = 'z+'
        else:
            # origin is at the top
            self.detector_ver = 'z-'

    def __repr__(self):
        """
        :return: a nicely formatted representation string
        """
        return f"{self.__class__.__name__}: beamline={self.beamline}, energy={self.energy}eV," \
               f" sample to detector distance={self.distance}m"


class Detector(object):
    """
    Class to handle the configuration of the detector used for data acquisition.
    """
    def __init__(self, name, datadir='', savedir='', template_imagefile='', roi=(), sum_roi=(), binning=(1, 1, 1),
                 **kwargs):
        """
        Initialize parameters of the detector.

        :param name: name of the detector: 'Maxipix'or 'Eiger2M' or 'Eiger4M'
        :param datadir: directory where the data is saved
        :param savedir: directory where to save files if needed
        :param template_imagefile: template for the name of image files
         - ID01: 'data_mpx4_%05d.edf.gz'
         - SIXS: 'spare_ascan_mu_%05d.nxs'
         - Cristal: 'S%d.nxs'
         - P10: sample_name + str('{:05d}'.format(scans[scan_nb])) + '_data_%06d.h5'
        :param roi: region of interest in the detector, use [] to use the full detector
        :param sum_roi: optional region of interest used for calculated an integrated intensity
        :param binning: binning of the 3D dataset (stacking dimension, detector vertical axis, detector horizontal axis)
        :param kwargs:
         - 'is_series' = boolean, True is the measurement is a series at P10 beamline
         - 'nb_pixel_x' and 'nb_pixel_y': useful when part of the detector is broken (less pixels than expected)
         - 'previous_binning': tuple or list of the three binning factors for reloaded binned data
        """
        # default values for kwargs
        nb_pixel_x = None
        nb_pixel_y = None
        previous_binning = None
        is_series = False

        for k in kwargs.keys():
            if k in ['is_series']:
                is_series = kwargs['is_series']
            elif k in ['nb_pixel_x']:
                nb_pixel_x = kwargs['nb_pixel_x']
            elif k in ['nb_pixel_y']:
                nb_pixel_y = kwargs['nb_pixel_y']
            elif k in ['previous_binning']:
                previous_binning = kwargs['previous_binning']
            else:
                raise Exception("unknown keyword argument given:", k)

        self.previous_binning = previous_binning or (1, 1, 1)
        self.is_series = is_series
        self.name = name  # string
        self.offsets = ()

        if name == 'Maxipix':
            nb_pixel_x = nb_pixel_x or 516
            nb_pixel_y = nb_pixel_y or 516
            self.nb_pixel_x = nb_pixel_x // self.previous_binning[2]
            self.nb_pixel_y = nb_pixel_y // self.previous_binning[1]
            self.pixelsize_x = 55e-06  # m
            self.pixelsize_y = 55e-06  # m
            self.counter = 'mpx4inr'
        elif name == 'Eiger2M':
            nb_pixel_x = nb_pixel_x or 1030
            nb_pixel_y = nb_pixel_y or 2164
            self.nb_pixel_x = nb_pixel_x // self.previous_binning[2]
            self.nb_pixel_y = nb_pixel_y // self.previous_binning[1]
            self.pixelsize_x = 75e-06  # m
            self.pixelsize_y = 75e-06  # m
            self.counter = 'ei2minr'
        elif name == 'Eiger4M':
            nb_pixel_x = nb_pixel_x or 2070
            nb_pixel_y = nb_pixel_y or 2167
            self.nb_pixel_x = nb_pixel_x // self.previous_binning[2]
            self.nb_pixel_y = nb_pixel_y // self.previous_binning[1]
            self.pixelsize_x = 75e-06  # m
            self.pixelsize_y = 75e-06  # m
            self.counter = ''  # unused
        elif name == 'Timepix':
            nb_pixel_x = nb_pixel_x or 256
            nb_pixel_y = nb_pixel_y or 256
            self.nb_pixel_x = nb_pixel_x // self.previous_binning[2]
            self.nb_pixel_y = nb_pixel_y // self.previous_binning[1]
            self.pixelsize_x = 55e-06  # m
            self.pixelsize_y = 55e-06  # m
            self.counter = ''  # unused
        elif name == 'Merlin':
            nb_pixel_x = nb_pixel_x or 515
            nb_pixel_y = nb_pixel_y or 515
            self.nb_pixel_x = nb_pixel_x // self.previous_binning[2]
            self.nb_pixel_y = nb_pixel_y // self.previous_binning[1]
            self.pixelsize_x = 55e-06  # m
            self.pixelsize_y = 55e-06  # m
            self.counter = 'alba2'
        else:
            raise ValueError('Unknown detector name')

        # define paths
        self.datadir = datadir
        self.savedir = savedir
        self.template_imagefile = template_imagefile

        # define regions of interest
        if len(roi) == 0:
            self.roi = [0, self.nb_pixel_y, 0, self.nb_pixel_x]
        elif len(roi) == 4:
            self.roi = roi
        else:
            raise ValueError("Incorrect value for parameter 'roi'")

        if len(sum_roi) == 0:
            self.sum_roi = [0, self.nb_pixel_y, 0, self.nb_pixel_x]
        elif len(sum_roi) == 4:
            self.sum_roi = sum_roi
        else:
            raise ValueError("Incorrect value for parameter 'sum_roi'")

        # define binning and correct the pixel sizes taking into account past and future binning
        self.binning = binning  # (stacking dimension, detector vertical axis, detector horizontal axis)
        self.pixelsize_y = self.pixelsize_y * self.previous_binning[1] * self.binning[1]
        self.pixelsize_x = self.pixelsize_x * self.previous_binning[2] * self.binning[2]

    def __repr__(self):
        """
        :return: a nicely formatted representation string
        """
        return f"{self.__class__.__name__}: {self.name}"

    def mask_detector(self, data, mask, nb_img=1, flatfield=None, background=None, hotpixels=None):
        """
        Mask data measured with a 2D detector (flatfield, background, hotpixels, gaps).

        :param data: the 2D data to mask
        :param mask: the 2D mask to be updated
        :param nb_img: number of images summed to yield the 2D data (e.g. in a series measurement)
        :param flatfield: the 2D flatfield array to be multiplied with the data
        :param background: a 2D array to be subtracted to the data
        :param hotpixels: a 2D array with hotpixels to be masked (1=hotpixel, 0=normal pixel)
        :return: the masked data and the updated mask
        """

        assert isinstance(data, np.ndarray) and isinstance(mask, np.ndarray), 'data and mask should be numpy arrays'
        if data.ndim != 2 or mask.ndim != 2:
            raise ValueError('data and mask should be 2D arrays')

        if data.shape != mask.shape:
            raise ValueError('data and mask must have the same shape\n data is ', data.shape,
                             ' while mask is ', mask.shape)

        # flatfield correction
        if flatfield is not None:
            if flatfield.shape != data.shape:
                raise ValueError('flatfield and data must have the same shape\n data is ', flatfield.shape,
                                 ' while data is ', data.shape)
            data = np.multiply(flatfield, data)

        # remove the background
        if background is not None:
            if background.shape != data.shape:
                raise ValueError('background and data must have the same shape\n data is ', background.shape,
                                 ' while data is ', data.shape)
            data = data - background

        # mask hotpixels
        if hotpixels is not None:
            if hotpixels.shape != data.shape:
                raise ValueError('hotpixels and data must have the same shape\n data is ', hotpixels.shape,
                                 ' while data is ', data.shape)
            data[hotpixels == 1] = 0
            mask[hotpixels == 1] = 1

        if self.name == 'Eiger2M':
            data[:, 255: 259] = 0
            data[:, 513: 517] = 0
            data[:, 771: 775] = 0
            data[0: 257, 72: 80] = 0
            data[255: 259, :] = 0
            data[511: 552, :0] = 0
            data[804: 809, :] = 0
            data[1061: 1102, :] = 0
            data[1355: 1359, :] = 0
            data[1611: 1652, :] = 0
            data[1905: 1909, :] = 0
            data[1248:1290, 478] = 0
            data[1214:1298, 481] = 0
            data[1649:1910, 620:628] = 0

            mask[:, 255: 259] = 1
            mask[:, 513: 517] = 1
            mask[:, 771: 775] = 1
            mask[0: 257, 72: 80] = 1
            mask[255: 259, :] = 1
            mask[511: 552, :] = 1
            mask[804: 809, :] = 1
            mask[1061: 1102, :] = 1
            mask[1355: 1359, :] = 1
            mask[1611: 1652, :] = 1
            mask[1905: 1909, :] = 1
            mask[1248:1290, 478] = 1
            mask[1214:1298, 481] = 1
            mask[1649:1910, 620:628] = 1

            # mask hot pixels
            mask[data > 1e6 * nb_img] = 1
            data[data > 1e6 * nb_img] = 0

        elif self.name == 'Eiger4M':
            data[:, 0:1] = 0
            data[:, -1:] = 0
            data[0:1, :] = 0
            data[-1:, :] = 0
            data[:, 1030:1040] = 0
            data[514:551, :] = 0
            data[1065:1102, :] = 0
            data[1616:1653, :] = 0

            mask[:, 0:1] = 1
            mask[:, -1:] = 1
            mask[0:1, :] = 1
            mask[-1:, :] = 1
            mask[:, 1030:1040] = 1
            mask[514:551, :] = 1
            mask[1065:1102, :] = 1
            mask[1616:1653, :] = 1

            # mask hot pixels, 4000000000 for the Eiger4M
            mask[data > 4000000000 * nb_img] = 1
            data[data > 4000000000 * nb_img] = 0

        elif self.name == 'Maxipix':
            data[:, 255:261] = 0
            data[255:261, :] = 0

            mask[:, 255:261] = 1
            mask[255:261, :] = 1

            # mask hot pixels
            mask[data > 1e6 * nb_img] = 1
            data[data > 1e6 * nb_img] = 0

        elif self.name == 'Merlin':
            data[:, 255:260] = 0
            data[255:260, :] = 0

            mask[:, 255:260] = 1
            mask[255:260, :] = 1

            # mask hot pixels
            mask[data > 1e6 * nb_img] = 1
            data[data > 1e6 * nb_img] = 0

        elif self.name == 'Timepix':
            pass  # no gaps

        else:
            raise NotImplementedError('Detector not implemented')

        return data, mask


if __name__ == "__main__":
    my = SetupPostprocessing(beamline='ID01', energy=8800, outofplane_angle=0, inplane_angle=7.248, tilt_angle=0.01,
                             rocking_angle='inplane', distance=7.25, grazing_angle=0, pixel_x=110e-6, pixel_y=110e-6)
    print(my.voxel_sizes((41, 256, 256), tilt_angle=0.01, pixel_x=110e-6, pixel_y=110e-6, debug=True))
