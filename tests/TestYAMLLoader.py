import unittest
import yaml
from PyAnalysisTools.base.YAMLLoader import YAMLLoader as YL


class TestYAMLLoader(unittest.TestCase):
    def setUp(self):
        self.data = {"a": 1, "b": 2}
        self.loader = YL()
        self.test_file = open("yaml_loader.yml", "w+")
        yaml.dump(self.data, self.test_file)
        self.test_file.close()

    def test_parsing(self):
        result = self.loader.read_yaml("yaml_loader.yml")
        self.assertEqual(result, self.data)

    def test_io_exception(self):
        self.assertRaises(IOError, self.loader.read_yaml, "non_existing_file")
