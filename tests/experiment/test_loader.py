# -*- coding: utf-8 -*-

# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import numpy as np
import os
from pyfakefs import fake_filesystem_unittest
import unittest
from bcdi.experiment.beamline import create_beamline
from bcdi.experiment.detector import create_detector
from bcdi.experiment.setup import Setup


def run_tests(test_class):
    suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


class TestRetrieveDistance(fake_filesystem_unittest.TestCase):
    """
    Tests related to DiffractometerID01.retrieve_distance.

    def retrieve_distance(setup) -> Optional[float]:
    """

    def setUp(self):
        self.setUpPyfakefs()
        self.valid_path = "/gpfs/bcdi/data/"
        os.makedirs(self.valid_path)
        with open(self.valid_path + "defined.spec", "w") as f:
            f.write(
                "test\n#UDETCALIB cen_pix_x=11.195,cen_pix_y=281.115,"
                "pixperdeg=455.257,"
                "det_distance_CC=1.434,det_distance_COM=1.193,"
                "timestamp=2021-02-28T13:01:16.615422"
            )

        with open(self.valid_path + "undefined.spec", "w") as f:
            f.write("test\n#this,is,bad")
        detector = create_detector("Maxipix")
        detector.rootdir = self.valid_path
        self.setup = Setup("ID01", detector=detector)
        self.beamline = create_beamline(name="ID01")

    def test_distance_defined(self):
        self.setup.detector.specfile = "defined.spec"

        distance = self.beamline.loader.retrieve_distance(self.setup)
        self.assertTrue(np.isclose(distance, 1.193))

    def test_distance_undefined(self):
        self.setup.detector.specfile = "undefined.spec"

        distance = self.beamline.loader.retrieve_distance(self.setup)
        self.assertTrue(distance is None)


if __name__ == "__main__":
    run_tests(TestRetrieveDistance)
