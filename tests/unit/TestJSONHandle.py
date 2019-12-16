import os
import unittest

from PyAnalysisTools.base.JSONHandle import JSONHandle
from PyAnalysisTools.base.Singleton import Singleton


def clear_instance(cls):
    try:
        del Singleton._instances[cls]
    except KeyError:
        pass


class TestJSONHandle(unittest.TestCase):
    def setUp(self):pass

    def tearDown(self):
        clear_instance(JSONHandle)

    def test_ctor(self):
        handle = JSONHandle('foo')
        self.assertEqual({}, handle.data)
        self.assertEqual('foo/config.json', handle.file_name)
        self.assertIsNone(handle.input_file)
        self.assertFalse(handle.copy)

    def test_add_args(self):
        handle = JSONHandle('foo')
        new_data = {'foo': 'bar'}
        handle.add_args(**new_data)
        self.assertEqual(new_data, handle.data)

    def test_reset_path(self):
        handle = JSONHandle('foo')
        handle.reset_path('bar')
        self.assertEqual('bar/config.json', handle.file_name)

    def test_dump_copy_no_input_file(self):
        handle = JSONHandle('foo', copy=True)
        self.assertIsNone(handle.dump())
