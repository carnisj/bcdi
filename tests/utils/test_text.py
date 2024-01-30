# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-05/2021 : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

import unittest

from bcdi.utils.text import Comment
from tests.config import run_tests


class TestComment(unittest.TestCase):
    def setUp(self) -> None:
        self.text = "test"
        self.comment = Comment(self.text)

    def test_update_comment(self):
        text_to_add = "blue"
        self.comment.concatenate(text_to_add)
        self.assertEqual(self.comment.text, self.text + "_" + text_to_add)


if __name__ == "__main__":
    run_tests(TestComment)
