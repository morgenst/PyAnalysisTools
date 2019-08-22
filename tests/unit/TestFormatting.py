import unittest
import ROOT
from PyAnalysisTools.PlottingUtils import Formatting as fm
from PyAnalysisTools.base import InvalidInputError
from random import random


class TestFormatting(unittest.TestCase):
    def setUp(self):
        self.hist = ROOT.TH1F('h', '', 10, -1., 1.)
        self.hist.FillRandom('gaus', 10000)

    def tearDown(self):
        del self.hist

    def get_2d_hist(self):
        h = ROOT.TH2F('h2', '', 10, -1, 1, 10, -1, 1)
        for i in range(10000):
            h.Fill(random(), random())
        return h

    def test_set_xtitle_hist_1d(self):
        fm.set_title_x(self.hist, "x_title")
        self.assertEqual(self.hist.GetXaxis().GetTitle(), "x_title")

    def test_set_ytitle_hist_1d(self):
        fm.set_title_y(self.hist, "y_title")
        self.assertEqual(self.hist.GetYaxis().GetTitle(), "y_title")

    def test_set_ztitle_hist_1d(self):
        fm.set_title_z(self.hist, "z_title")
        self.assertEqual(self.hist.GetZaxis().GetTitle(), "z_title")

    def test_set_invalid_title_hist_1d(self):
        self.assertRaises(InvalidInputError, fm.set_axis_title, self.hist, 'title', 'a')

    def test_set_title_x_offset(self):
        fm.set_title_x_offset(self.hist, 10)
        self.assertEqual(self.hist.GetXaxis().GetTitleOffset(), 10)

    def test_set_title_y_offset(self):
        fm.set_title_y_offset(self.hist, 10)
        self.assertEqual(self.hist.GetYaxis().GetTitleOffset(), 10)

    def test_set_title_z_offset(self):
        fm.set_title_z_offset(self.hist, 10)
        self.assertEqual(self.hist.GetZaxis().GetTitleOffset(), 10)

    def test_set_title_x_size(self):
        fm.set_title_x_size(self.hist, 10)
        self.assertEqual(self.hist.GetXaxis().GetTitleSize(), 10)

    def test_set_title_y_size(self):
        fm.set_title_y_size(self.hist, 10)
        self.assertEqual(self.hist.GetYaxis().GetTitleSize(), 10)

    def test_set_title_z_size(self):
        fm.set_title_z_size(self.hist, 10)
        self.assertEqual(self.hist.GetZaxis().GetTitleSize(), 10)

    def test_set_title_size_fail(self):
        self.assertRaises(InvalidInputError, fm.set_axis_title_size, self.hist, 10, 'a')

    def test_set_title_size_fail_type(self):
        self.assertRaises(TypeError, fm.set_axis_title_size, ROOT.TCanvas(), 10, 'z')

    def test_set_title_size_fail_no_axis(self):
        self.assertIsNone(fm.set_axis_title_size(ROOT.TGraph(), 10, 'x'))

    def test_add_lumi_text(self):
        c = ROOT.TCanvas()
        fm.add_lumi_text(c, 100.)
        lumi_text = c.GetListOfPrimitives()[-1]
        self.assertEqual(lumi_text.GetTitle(), '#scale[0.7]{#int}dt L = 100.00 fb^{-1} #sqrt{s} = 13 TeV')
        del c

    def test_add_lumi_text_custom(self):
        c = ROOT.TCanvas()
        fm.add_lumi_text(c, 100., lumi_text='foo')
        lumi_text = c.GetListOfPrimitives()[-1]
        self.assertEqual(lumi_text.GetTitle(), 'foo ')
        del c

    def test_add_lumi_text_split(self):
        c = ROOT.TCanvas()
        fm.add_lumi_text(c, 100., split_lumi_text=True)
        self.assertEqual(2, len(c.GetListOfPrimitives()))
        del c

    def test_add_atlas_label(self):
        c = ROOT.TCanvas()
        fm.add_atlas_label(c, 'descr')
        self.assertEqual(2, len(c.GetListOfPrimitives()))
        self.assertEqual(c.GetListOfPrimitives()[0].GetTitle(), 'ATLAS')
        self.assertEqual(c.GetListOfPrimitives()[1].GetTitle(), 'descr')
        del c

    def test_set_minimum_y(self):
        fm.set_minimum_y(self.hist, 2.)
        self.assertEqual(2., self.hist.GetMinimum())

    def test_set_minimum_x(self):
        fm.set_minimum_x(self.hist, 1.)
        self.assertEqual(1., self.hist.GetXaxis().GetXmin())

    def test_set_maximum_x_above_range(self):
        fm.set_maximum_x(self.hist, 2.)
        self.assertEqual(2., self.hist.GetXaxis().GetXmax())

    def test_set_maximum_x(self):
        fm.set_maximum_x(self.hist, 0.5)
        self.assertEqual(0.5, self.hist.GetXaxis().GetXmax())

    def test_set_maximum_y(self):
        fm.set_maximum_y(self.hist, 2.)
        self.assertEqual(2., self.hist.GetMaximum())

    def test_add_text_to_canvas(self):
        c = ROOT.TCanvas()
        fm.add_text_to_canvas(c, 'descr')
        self.assertEqual(c.GetListOfPrimitives()[0].GetTitle(), 'descr')
        del c

    def test_format_canvas_margin(self):
        c = ROOT.TCanvas()
        c = fm.format_canvas(c, margin={'top': 0.2})
        self.assertAlmostEqual(0.2, c.GetTopMargin(), 2)

    def test_format_canvas_margin_wrong_input(self):
        c = ROOT.TCanvas()
        self.assertRaises(AttributeError, fm.format_canvas, c, margin={'x': 0.2})

    def test_format_canvas_margin_wrong_input_type(self):
        c = ROOT.TCanvas()
        self.assertRaises(AttributeError, fm.format_canvas, c, margin=['x', 0.2])

    def test_range_z_all_none(self):
        h = self.get_2d_hist()
        init_min, init_max = h.GetMinimum(), h.GetMaximum()
        fm.set_range_z(h)
        self.assertEqual(init_min, h.GetMinimum())
        self.assertEqual(init_min, h.GetZaxis().GetXmin())
        self.assertEqual(init_max, h.GetMaximum())
        self.assertEqual(init_max, h.GetZaxis().GetXmax())
        del h

    def test_range_z_min_only(self):
        h = self.get_2d_hist()
        init_max = h.GetMaximum()
        fm.set_range_z(h, 10.)
        self.assertEqual(10., h.GetMinimum())
        self.assertEqual(10., h.GetZaxis().GetXmin())
        self.assertEqual(init_max, h.GetMaximum())
        self.assertEqual(init_max, h.GetZaxis().GetXmax())
        del h

    def test_range_z_max_only(self):
        h = self.get_2d_hist()
        init_min = h.GetMinimum()
        fm.set_range_z(h, maximum=20.)
        self.assertEqual(init_min, h.GetMinimum())
        self.assertEqual(init_min, h.GetZaxis().GetXmin())
        self.assertEqual(20., h.GetMaximum())
        self.assertEqual(20., h.GetZaxis().GetXmax())
        del h

    def test_range_z_min_max(self):
        h = self.get_2d_hist()
        fm.set_range_z(h, 10., 20.)
        self.assertEqual(10., h.GetMinimum())
        self.assertEqual(10., h.GetZaxis().GetXmin())
        self.assertEqual(20., h.GetMaximum())
        self.assertEqual(20., h.GetZaxis().GetXmax())
        del h

    def test_range_z_graph(self):
        self.assertEqual(None, fm.set_range_z(ROOT.TGraph))

    @unittest.skip("Not implemented")
    def test_set_xtitle_hist_2d(self):
        fm.set_title_x(self.unformatted_hist_2d, "x_title")
        self.assertEqual(self.unformatted_hist_2d.GetXaxis().GetTitle(), "x_title")

    @unittest.skip("Not implemented")
    def test_set_ytitle_hist_2d(self):
        fm.set_title_y(self.unformatted_hist_2d, "y_title")
        self.assertEqual(self.unformatted_hist_2d.GetYaxis().GetTitle(), "y_title")

    @unittest.skip("Not implemented")
    def test_set_ztitle_hist_2d(self):
        fm.set_title_y(self.unformatted_hist_2d, "z_title")
        self.assertEqual(self.unformatted_hist_1d.GetYaxis().GetTitle(), "y_title")

    @unittest.skip("Not implemented")
    def test_set_xtitle_hist_graph(self):
        fm.set_title_x(self.unformatted_hist_graph, "x_title")
        self.assertEqual(self.unformatted_hist_graph.GetXaxis().GetTitle(), "x_title")

    @unittest.skip("Not implemented")
    def test_set_ytitle_hist_graph(self):
        fm.set_title_y(self.unformatted_hist_graph, "y_title")
        self.assertEqual(self.unformatted_hist_graph.GetYaxis().GetTitle(), "y_title")

    @unittest.skip("Not implemented")
    def test_set_ztitle_hist_graph(self):
        fm.set_title_y(self.unformatted_hist_graph, "z_title")
        self.assertEqual(self.unformatted_hist_1d.GetYaxis().GetTitle(), "y_title")

    def test_set_xtitle_invalid_object(self):
        invalid_object = ROOT.TCanvas("x", "x")
        self.assertRaises(TypeError, fm.set_title_x, *[invalid_object, "x_title"])

    def test_set_ytitle_invalid_object(self):
        invalid_object = ROOT.TCanvas("x", "x")
        self.assertRaises(TypeError, fm.set_title_y, *[invalid_object, "y_title"])

    def test_set_marker_style_1d(self):
        config = {"marker": {"color": ROOT.kRed, "style": 22, "size": 2}}
        fm.set_style_options(self.hist, config)
        self.assertEqual(self.hist.GetMarkerColor(), ROOT.kRed)
        self.assertEqual(self.hist.GetMarkerStyle(), 22)
        self.assertEqual(self.hist.GetMarkerSize(), 2)

    def test_set_line_style_1d(self):
        config = {"lines": {"color": ROOT.kRed, "style": 22, "width": 2}}
        fm.set_style_options(self.hist, config)
        self.assertEqual(self.hist.GetLineColor(), ROOT.kBlack)

    def test_set_style_invalid_config(self):
        self.assertRaises(InvalidInputError, fm.set_style_options, *[None, []])

    def test_set_style_invalid_config_option(self):
        self.assertRaises(InvalidInputError, fm.set_style_options, *[None, {"marker": []}])

    def test_set_style_invalid_(self):
        self.assertRaises(InvalidInputError, fm.set_style_options, *[None, {"marker": []}])

    @unittest.skip("Check")
    def test_set_marker_style_misspell(self):
        config = {"marker": {"sizes": 2}}
        fm.set_style_options(self.unformatted_hist_1d, config)
        self.assertEqual(self.unformatted_hist_1d.GetMarkerSize(), 1)

    @unittest.skip("Not implemented")
    def test_set_bin_label_x(self):
        pass

    def test_set_range_hist_min_only(self):
        fm.set_range(self.hist, 1.)
        self.assertEqual(self.hist.GetMinimum(), 1.)

    def test_set_range_hist_max_only(self):
        fm.set_range(self.hist, maximum=10.)
        self.assertEqual(self.hist.GetMaximum(), 10.)

    def test_set_range_hist(self):
        fm.set_range(self.hist, 1., 10.)
        self.assertEqual(self.hist.GetMinimum(), 1.)
        self.assertEqual(self.hist.GetMaximum(), 10.)

    @unittest.skip("Not implemented")
    def test_set_range_graph_min_only(self):
        fm.set_range(self.hist, 1.)
        self.assertEqual(self.hist.GetMinimum(), 1.)

    @unittest.skip("Not implemented")
    def test_set_range_graph_max_only(self):
        fm.set_range(self.unformatted_graph_1d, maximum=10.)
        self.assertEqual(self.unformatted_graph_1d.GetMaximum(), 10.)

    @unittest.skip("Not implemented")
    def test_set_range_hist(self):
        fm.set_range(self.unformatted_graph_1d, 1., 10.)
        self.assertEqual(self.unformatted_graph_1d.GetMinimum(), 1.)
        self.assertEqual(self.unformatted_graph_1d.GetMaximum(), 10.)

    def test_make_text(self):
        res = fm.make_text(0., 1., "TexText", size=10, angle=10., color=ROOT.kBlue)
        self.assertEqual(res.GetX(), 0.)
        self.assertEqual(res.GetY(), 1.)
        self.assertEqual(res.GetTextSize(), 10.)
        self.assertEqual(res.GetTextColor(), ROOT.kBlue)
