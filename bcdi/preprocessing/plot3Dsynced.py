# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Steven Leake, steven.leake@esrf.fr

import functools
import sys
import numpy

from silx.gui import qt
from silx.gui.plot3d.SceneWindow import SceneWindow, items

qapp = qt.QApplication.instance() or qt.QApplication([])


# Create two SceneWindow widgets
def create_plot3d(data):
    window = SceneWindow()
    sceneWidget = window.getSceneWidget()
    #x, y, z = numpy.meshgrid(numpy.linspace(-10, 10, 64),
    #                         numpy.linspace(-10, 10, 64),
    #                         numpy.linspace(-10, 10, 64))
    #data = numpy.sin(x * y * z) / (x * y * z)

    volume = sceneWidget.addVolume(data)
    volume.addIsosurface(0.4*np.max(data), '#FF000080')
    return window

import h5py as h5
import numpy as np
with h5.File('support.h5','r') as outf:
	
	window1 = create_plot3d(np.log10(outf['rawdata']))
	window2 = create_plot3d(outf['poly'])
	window3 = create_plot3d(np.log10(outf['poly_fft_shift']))

# Synchronize the 2 cameras
class SyncCameras:
    """Synchronize direction and position of the camera between multiple SceneWidgets
    
    :param List[~silx.gui.plot3d.SceneWidget.SceneWidget] sceneWidgets:
    """

    def __init__(self, *sceneWidgets):
        self.__cameras = [sw.viewport.camera for sw in sceneWidgets]
        self.__updating = False
        for camera in self.__cameras:
            camera.addListener(self._camera_changed)

    def _camera_changed(self, source):
        """Camera changes listener"""
        if self.__updating:
            return

        self.__updating = True
        position = source.extrinsic.position
        direction = source.extrinsic.direction
        up = source.extrinsic.up

        for camera in self.__cameras:
            if camera is source:
                continue
            if not (numpy.array_equal(direction, camera.extrinsic.direction) and
                    numpy.array_equal(up, camera.extrinsic.up)):
                camera.extrinsic.setOrientation(direction, up)
            if not numpy.array_equal(position, camera.extrinsic.position):
                camera.extrinsic.position = position

        self.__updating = False


sync_cam = SyncCameras(
    window1.getSceneWidget(),
    window2.getSceneWidget(),
    window3.getSceneWidget())

# Run example

window1.show()
window2.show()
window3.show()

sys.excepthook = qt.exceptionHandler
qapp.exec_()
