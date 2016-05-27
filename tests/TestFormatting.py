__author__ = 'marcusmorgenstern'
__mail__ = ''

import unittest

import ROOT

from PyAnalysisTools.PlottingUtils import Formatting as FM
from PyAnalysisTools.base import InvalidInputError


class TestFormatting(unittest.TestCase):
    def setUp(self):
        root_file = ROOT.TFile.Open("test_data/test.root", "READ")
        self.unformatted_hist_1d = root_file.Get("test_hist_1")
        self.unformatted_hist_1d.SetDirectory(0)
        ROOT.SetOwnership(self.unformatted_hist_1d, False)

    def test_set_xtitle_hist_1d(self):
        FM.set_title_x(self.unformatted_hist_1d, "x_title")
        self.assertEqual(self.unformatted_hist_1d.GetXaxis().GetTitle(), "x_title")

    def test_set_ytitle_hist_1d(self):
        FM.set_title_y(self.unformatted_hist_1d, "y_title")
        self.assertEqual(self.unformatted_hist_1d.GetYaxis().GetTitle(), "y_title")

    @unittest.skip("Not implemented")
    def test_set_ztitle_hist_1d(self):
        FM.set_title_y(self.unformatted_hist_1d, "y_title")
        self.assertEqual(self.unformatted_hist_1d.GetYaxis().GetTitle(), "y_title")

    @unittest.skip("Not implemented")
    def test_set_xtitle_hist_2d(self):
        FM.set_title_x(self.unformatted_hist_2d, "x_title")
        self.assertEqual(self.unformatted_hist_2d.GetXaxis().GetTitle(), "x_title")

    @unittest.skip("Not implemented")
    def test_set_ytitle_hist_2d(self):
        FM.set_title_y(self.unformatted_hist_2d, "y_title")
        self.assertEqual(self.unformatted_hist_2d.GetYaxis().GetTitle(), "y_title")

    @unittest.skip("Not implemented")
    def test_set_ztitle_hist_2d(self):
        FM.set_title_y(self.unformatted_hist_2d, "z_title")
        self.assertEqual(self.unformatted_hist_1d.GetYaxis().GetTitle(), "y_title")

    @unittest.skip("Not implemented")
    def test_set_xtitle_hist_graph(self):
        FM.set_title_x(self.unformatted_hist_graph, "x_title")
        self.assertEqual(self.unformatted_hist_graph.GetXaxis().GetTitle(), "x_title")

    @unittest.skip("Not implemented")
    def test_set_ytitle_hist_graph(self):
        FM.set_title_y(self.unformatted_hist_graph, "y_title")
        self.assertEqual(self.unformatted_hist_graph.GetYaxis().GetTitle(), "y_title")

    @unittest.skip("Not implemented")
    def test_set_ztitle_hist_graph(self):
        FM.set_title_y(self.unformatted_hist_graph, "z_title")
        self.assertEqual(self.unformatted_hist_1d.GetYaxis().GetTitle(), "y_title")

    def test_set_xtitle_invalid_object(self):
        invalid_object = ROOT.TCanvas("x", "x")
        self.assertRaises(TypeError, FM.set_title_x, *[invalid_object, "x_title"])

    def test_set_ytitle_invalid_object(self):
        invalid_object = ROOT.TCanvas("x", "x")
        self.assertRaises(TypeError, FM.set_title_y, *[invalid_object, "y_title"])

    def test_set_marker_style_1d(self):
        config = {"marker": {"color": ROOT.kRed, "style": 22, "size": 2}}
        FM.set_style_options(self.unformatted_hist_1d, config)
        self.assertEqual(self.unformatted_hist_1d.GetMarkerColor(), ROOT.kRed)
        self.assertEqual(self.unformatted_hist_1d.GetMarkerStyle(), 22)
        self.assertEqual(self.unformatted_hist_1d.GetMarkerSize(), 2)

    def test_set_line_style_1d(self):
        config = {"line": {"color": ROOT.kRed, "style": 22, "width": 2}}
        FM.set_style_options(self.unformatted_hist_1d, config)
        self.assertEqual(self.unformatted_hist_1d.GetLineColor(), ROOT.kRed)
        self.assertEqual(self.unformatted_hist_1d.GetLineStyle(), 22)
        self.assertEqual(self.unformatted_hist_1d.GetLineWidth(), 2)
        
    def test_set_style_invalid_config(self):
        self.assertRaises(InvalidInputError, FM.set_style_options, *[None, []])

    def test_set_style_invalid_config_option(self):
        self.assertRaises(InvalidInputError, FM.set_style_options, *[None, {"marker": []}])

    def test_set_marker_style_misspell(self):
        config = {"marker": {"sizes": 2}}
        FM.set_style_options(self.unformatted_hist_1d, config)
        self.assertEqual(self.unformatted_hist_1d.GetMarkerSize(), 1)

    @unittest.skip("Not implemented")
    def test_set_bin_label_x(self):
        pass
