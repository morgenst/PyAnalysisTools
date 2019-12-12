
import unittest
import os
from PyAnalysisTools.AnalysisTools import Utilities

cwd = os.path.dirname(__file__)


class TestAnalysisUtilities(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_find_data_period_None(self):
        period, year = Utilities.find_data_period(100, None)
        self.assertIsNone(period)
        self.assertIsNone(year)

    def test_find_data_period(self):
        period, year = Utilities.find_data_period(100, {'2018': {'periodB': [100]}})
        self.assertEqual('periodB', period)
        self.assertEqual('2018', year)

    def test_find_data_period_no_match(self):
        period, year = Utilities.find_data_period(100, {'2018': {'periodB': [200]}})
        self.assertIsNone(period)
        self.assertIsNone(year)
