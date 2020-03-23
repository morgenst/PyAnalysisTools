import unittest
import os
from PyAnalysisTools.base import OutputHandle
from tests.unit import tearDownModule  # noqa: F401

cwd = os.path.dirname(__file__)


class TestOutputSysFileHandle(unittest.TestCase):
    def setUp(self):
        pass

    def test_default_argument_output_dir(self):
        h = OutputHandle.SysOutputHandle()
        self.assertEqual(h.base_output_dir, './')

    def test_output_tag(self):
        h = OutputHandle.SysOutputHandle(output_tag='foo')
        self.assertTrue(h.output_dir.endswith('foo'))


class TestOutputHandle(unittest.TestCase):
    def setUp(self):
        pass


class TestOutputFileHandle(unittest.TestCase):
    def setUp(self):
        pass
