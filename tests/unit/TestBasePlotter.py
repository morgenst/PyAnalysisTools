__author__ = 'marcusmorgenstern'
__mail__ = ''

import unittest
from PyAnalysisTools.base import InvalidInputError
from PyAnalysisTools.PlottingUtils.BasePlotter import BasePlotter as BP


class TestBasePlotter(unittest.TestCase):
    def setUp(self):
        pass

    def test_ctor_plot_config_file_pass(self):
        plotter = BP(input_files=[], plot_config_file="foo")
        self.assertEqual(plotter.plot_config_file, "foo")

    @unittest.skip("Check why this test is raising exception itself")
    def test_ctor_plot_config_file_fail(self):
        self.assertRaises(InvalidInputError, BP(), "")
