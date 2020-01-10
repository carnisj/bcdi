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

        :param beamline: name of the beamline: 'ID01', 'SIXS_2018', 'SIXS_2019', '34ID', 'P10', 'CRISTAL'
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

    def rotation_direction(self):
        """
        Define a coefficient +/- 1 depending on the detector rotation direction and the detector inplane orientation.

        :return: a coefficient  which is 1 for anticlockwise rotation or -1 for clockwise rotation.
        """
        if self.beamline == 'SIXS_2018' or self.beamline == 'SIXS_2019':
            # gamma is anti-clockwise
            coeff_inplane = 1
            # TODO: check this
        elif self.beamline == 'ID01':
            # nu is clockwise, we see the detector from downstream (behind)
            coeff_inplane = -1
        elif self.beamline == '34ID':
            coeff_inplane = 1
            # gamma is anti-clockwise
            # TODO: check this
        elif self.beamline == 'P10':
            coeff_inplane = -1
            # gamma is anti-clockwise, we see the detector from the front
        elif self.beamline == 'CRISTAL':
            coeff_inplane = 1
            # gamma is anti-clockwise
            # TODO: check this
        else:
            raise ValueError('setup parameter: ', self.beamline, 'not defined')
        return coeff_inplane

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

    def detector_frame(self, obj, voxelsize, width_z=np.nan, width_y=np.nan, width_x=np.nan,
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
        for k in kwargs.keys():
            if k in ['title']:
                title = kwargs['title']
            else:
                raise Exception("unknown keyword argument given: allowed is 'title'")
        try:
            title
        except NameError:  # title not declared
            title = 'Object'

        nbz, nby, nbx = obj.shape

        if debugging:
            gu.multislices_plot(abs(obj), sum_frames=True, width_z=width_z, width_y=width_y, width_x=width_x,
                                invert_yaxis=True, title=title + ' before interpolation\n')

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
                                invert_yaxis=True, title=title + ' interpolated in detector frame\n')

        return detector_obj

    def orthogonalize(self, obj, initial_shape=(), voxel_size=np.nan, width_z=np.nan, width_y=np.nan,
                      width_x=np.nan, debugging=False, **kwargs):
        """
        Interpolate obj on the orthogonal reference frame defined by the setup.

        :param obj: real space object, in a non-orthogonal frame (output of phasing program)
        :param initial_shape: shape of the FFT used for phasing
        :param voxel_size: user-defined voxel size, in nm
        :param width_z: size of the area to plot in z (axis 0), centered on the middle of the initial array
        :param width_y: size of the area to plot in y (axis 1), centered on the middle of the initial array
        :param width_x: size of the area to plot in x (axis 2), centered on the middle of the initial array
        :param debugging: True to show plots before and after interpolation
        :param kwargs:
         - 'title': title for the debugging plots
        :return: object interpolated on an orthogonal grid
        """
        for k in kwargs.keys():
            if k in ['title']:
                title = kwargs['title']
            else:
                raise Exception("unknown keyword argument given: allowed is 'title'")
        try:
            title
        except NameError:  # title not declared
            title = 'Object'

        if len(initial_shape) == 0:
            initial_shape = obj.shape

        if debugging:
            gu.multislices_plot(abs(obj), sum_frames=True, width_z=width_z, width_y=width_y, width_x=width_x,
                                invert_yaxis=True, title=title+' in detector frame')

        tilt_sign = np.sign(self.tilt_angle)
        wavelength = 12.398 * 1e-7 / self.energy  # in m

        dz_realspace = wavelength / (initial_shape[0] * abs(self.tilt_angle) * np.pi / 180) * 1e9  # in nm
        dy_realspace = wavelength * self.distance / (initial_shape[1] * self.pixel_y) * 1e9  # in nm
        dx_realspace = wavelength * self.distance / (initial_shape[2] * self.pixel_x) * 1e9  # in nm
        print('Real space pixel size (z, y, x) based on initial FFT shape: (',
              str('{:.2f}'.format(dz_realspace)), 'nm,',
              str('{:.2f}'.format(dy_realspace)), 'nm,',
              str('{:.2f}'.format(dx_realspace)), 'nm )')

        nbz, nby, nbx = obj.shape  # could be smaller if the object was cropped around the support
        if nbz != initial_shape[0] or nby != initial_shape[1] or nbx != initial_shape[2]:
            tilt = tilt_sign * wavelength / (nbz * dz_realspace * np.pi / 180) * 1e9  # in m
            pixel_y = wavelength * self.distance / (nby * dy_realspace) * 1e9  # in m
            pixel_x = wavelength * self.distance / (nbx * dx_realspace) * 1e9  # in m
            print('Tilt, pixel_y, pixel_x based on actual array shape: (',
                  str('{:.4f}'.format(tilt)), 'deg,',
                  str('{:.2f}'.format(pixel_y * 1e6)), 'um,',
                  str('{:.2f}'.format(pixel_x * 1e6)), 'um)')
            dz_realspace = wavelength / (nbz * abs(tilt) * np.pi / 180) * 1e9  # in nm
            dy_realspace = wavelength * self.distance / (nby * pixel_y) * 1e9  # in nm
            dx_realspace = wavelength * self.distance / (nbx * pixel_x) * 1e9  # in nm
            print('New real space pixel size (z, y, x) based on actual array shape: (',
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
                                invert_yaxis=True, title=title+' in the orthogonal laboratory frame')
        return ortho_obj, voxel

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
                             np.sin(mygrazing_angle) * np.sin(inplane) * np.sin(outofplane),
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
                print('rocking angle is mu')
                # rocking mu angle anti-clockwise around y, mu below all other sample rotations
                mymatrix[:, 0] = 2 * np.pi * nbx / lambdaz * np.array([-pixel_x * np.cos(inplane),
                                                                       0,
                                                                       pixel_x * np.sin(inplane)])
                mymatrix[:, 1] = 2 * np.pi * nby / lambdaz * np.array([pixel_y * np.sin(inplane) * np.sin(outofplane),
                                                                       -pixel_y * np.cos(outofplane),
                                                                       pixel_y * np.cos(inplane) * np.sin(outofplane)])
                mymatrix[:, 2] = 2 * np.pi * nbz / lambdaz * np.array(
                    [tilt * distance * (1 - np.cos(inplane) * np.cos(outofplane)),
                     0,
                     tilt * distance * np.sin(inplane) * np.cos(outofplane)])
            else:
                raise ValueError('inplane rocking for phi not yet implemented for P10')

        if self.beamline == '34ID':
            print('using APS 34ID geometry')
            if self.rocking_angle == "outofplane":
                print('rocking angle is tilt')
                # rocking tilt angle anti-clockwise around x (th does not matter, above tilt)
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
                print('rocking angle is th, with tilt non zero')
                # rocking th angle anti-clockwise around y, incident angle is non zero
                mymatrix[:, 0] = 2 * np.pi * nbx / lambdaz * np.array([pixel_x * np.cos(inplane),
                                                                       0,
                                                                       -pixel_x * np.sin(inplane)])
                mymatrix[:, 1] = 2 * np.pi * nby / lambdaz * np.array([pixel_y * np.sin(inplane) * np.sin(outofplane),
                                                                       -pixel_y * np.cos(outofplane),
                                                                       pixel_y * np.cos(inplane) * np.sin(outofplane)])
                mymatrix[:, 2] = 2 * np.pi * nbz / lambdaz * tilt * distance * \
                    np.array([(np.sin(mygrazing_angle) * np.sin(outofplane) +
                              np.cos(mygrazing_angle) * (1 - np.cos(inplane) * np.cos(outofplane))),
                              -np.sin(mygrazing_angle) * np.sin(inplane) * np.sin(outofplane),
                              np.cos(mygrazing_angle) * np.sin(inplane) * np.cos(outofplane)])

            elif self.rocking_angle == "inplane" and mygrazing_angle == 0:
                print('rocking angle is th, tilt=0')
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


class SetupPreprocessing(object):
    """
    Class to handle the experimental geometry for preprocessing.
    """
    def __init__(self, beamline, rocking_angle, distance=1, energy=8000, direct_beam=(0, 0), beam_direction=(1, 0, 0),
                 sample_inplane=(1, 0, 0), sample_outofplane=(0, 0, 1), sample_offsets=(0, 0, 0), offset_inplane=0,
                 **kwargs):
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
        try:
            filtered_data
        except NameError:  # filtered_data not declared
            filtered_data = False
        try:
            is_orthogonal
        except NameError:  # is_orthogonal not declared
            is_orthogonal = False
        try:
            custom_scan
        except NameError:  # custom_scan not declared
            custom_scan = False
        try:
            custom_images
        except NameError:  # custom_images not declared
            custom_images = []
        try:
            custom_monitor
        except NameError:  # custom_monitor not declared
            custom_monitor = []
        try:
            custom_motors
        except NameError:  # custom_motors not declared
            custom_motors = {}

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


class Detector(object):
    """
    Class to handle the configuration of the detector used for data acquisition.
    """
    def __init__(self, name, datadir='', savedir='', template_imagefile='', roi=(), binning=(1, 1, 1), **kwargs):
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
        :param binning: binning of the 3D dataset (stacking dimension, detector vertical axis, detector horizontal axis)
        :param kwargs:
         - 'is_series' = boolean, True is the measurement is a series at P10 beamline
         - 'nb_pixel_x' and 'nb_pixel_y': useful when part of the detector is broken (less pixels than expected)
        """
        for k in kwargs.keys():
            if k in ['is_series']:
                self.is_series = kwargs['is_series']
            elif k in ['nb_pixel_x']:
                nb_pixel_x = kwargs['nb_pixel_x']
            elif k in ['nb_pixel_y']:
                nb_pixel_y = kwargs['nb_pixel_y']
            else:
                raise Exception("unknown keyword argument given:", k)

        self.name = name  # string
        self.offsets = ()
        if name == 'Maxipix':
            try:
                self.nb_pixel_x = nb_pixel_x
            except NameError:  # nb_pixel_x not declared
                self.nb_pixel_x = 516
            try:
                self.nb_pixel_y = nb_pixel_y
            except NameError:  # nb_pixel_y not declared
                self.nb_pixel_y = 516
            self.pixelsize_x = 55e-06  # m
            self.pixelsize_y = 55e-06  # m
            self.counter = 'mpx4inr'
        elif name == 'Eiger2M':
            try:
                self.nb_pixel_x = nb_pixel_x
            except NameError:  # nb_pixel_x not declared
                self.nb_pixel_x = 1030
            try:
                self.nb_pixel_y = nb_pixel_y
            except NameError:  # nb_pixel_y not declared
                self.nb_pixel_y = 2164
            self.pixelsize_x = 75e-06  # m
            self.pixelsize_y = 75e-06  # m
            self.counter = 'ei2minr'
        elif name == 'Eiger4M':
            try:
                self.nb_pixel_x = nb_pixel_x
            except NameError:  # nb_pixel_x not declared
                self.nb_pixel_x = 2070
            try:
                self.nb_pixel_y = nb_pixel_y
            except NameError:  # nb_pixel_y not declared
                self.nb_pixel_y = 2167
            self.pixelsize_x = 75e-06  # m
            self.pixelsize_y = 75e-06  # m
            self.counter = ''  # unused
        else:
            raise ValueError('Unknown detector name')

        self.datadir = datadir
        self.savedir = savedir
        self.template_imagefile = template_imagefile

        if len(roi) == 0:
            self.roi = [0, self.nb_pixel_y, 0, self.nb_pixel_x]
            self.roiUser = False
        elif len(roi) == 4:
            self.roi = roi
            self.roiUser = True
        else:
            raise ValueError("Incorrect value for parameter 'roi'")

        self.binning = binning  # (stacking dimension, detector vertical axis, detector horizontal axis)
        self.pixelsize_y = self.pixelsize_y * binning[1]
        self.pixelsize_x = self.pixelsize_x * binning[2]
