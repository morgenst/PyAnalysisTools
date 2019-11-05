from __future__ import print_function
import unittest
import yaml
import os
from PyAnalysisTools.base.YAMLHandle import YAMLLoader as YL
from PyAnalysisTools.base.YAMLHandle import YAMLDumper as YD
from PyAnalysisTools.base import _logger


class TestYAMLLoader(unittest.TestCase):
    def setUp(self):
        _logger.setLevel(50)
        self.data = {"a": 1, "b": 2}
        self.loader = YL()
        self.test_file = open("yaml_loader.yml", "w+")
        yaml.dump(self.data, self.test_file)
        self.test_file.close()

    def test_parsing(self):
        result = YL.read_yaml("yaml_loader.yml")
        self.assertEqual(result, self.data)

    def test_io_exception(self):
        self.assertRaises(IOError, YL.read_yaml, "non_existing_file")

    def test_invalid_input_file(self):
        test_file = open("invalid_yaml_file.yml", "w+")
        print("foo:--:\nsome invalid input", file=test_file)
        test_file.close()
        self.assertRaises(Exception, YL.read_yaml, "invalid_yaml_file.yml")

    def test_accept_None(self):
        self.assertEqual(None, YL.read_yaml(None, True))


class TestYAMLDumper(unittest.TestCase):
    def setUp(self):
        _logger.setLevel(50)
        self.data = {"a": 1, "b": 2}
        self.dumper = YD()
        self.test_file_name = "yaml_dumper.yml"

    def test_dump_dict(self):
        YD.dump_yaml(self.data, self.test_file_name)
        self.assertTrue(os.path.exists(self.test_file_name))

    def test_dump_failure(self):
        self.assertRaises(Exception, YD.dump_yaml, "foo:--:\nsome invalid input", "/usr/bin/test.yml")
