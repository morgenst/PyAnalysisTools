import unittest
from PyAnalysisTools.base import Utilities


class TestFileHandle(unittest.TestCase):
    def setUp(self):
        pass

    def test_flatted_dict(self):
        d1 = {"a": 1}
        d2 = {"b": 2}
        result = {"a": 1, "b": 2}
        test = Utilities.merge_dictionaries(d1, d2)
        self.assertEqual(test, result, "Pass flatted dict test")

    def test_recursive_glob(self):
        pass
