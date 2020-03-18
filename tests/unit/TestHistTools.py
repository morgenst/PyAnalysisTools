from builtins import map
from builtins import range
from builtins import object
import unittest
import ROOT
import os
from PyAnalysisTools.PlottingUtils import HistTools as ht
from PyAnalysisTools.base import InvalidInputError

cwd = os.path.dirname(__file__)
ROOT.gROOT.SetBatch(True)


class PlotConfig(object):
    pass


class TestHistTools(unittest.TestCase):
    def setUp(self):
        self.hist = ROOT.TH1F('h', '', 10, -1., 1.)
        self.hist.FillRandom('gaus', 10000)
        self.plot_config = PlotConfig()
        self.plot_config.axis_labels = None

    def tearDown(self):
        del self.hist

    def test_normalise(self):
        ht.normalise(self.hist)
        self.assertAlmostEqual(self.hist.Integral(), 1., delta=1e-5)

    def test_normalise_empty(self):
        hist = ROOT.TH1F('h', '', 10, -1., 1.)
        res = ht._normalise_1d_hist(hist)
        self.assertEqual(0., res.Integral())

    def test_normalise_asymmetric(self):
        h = ht.rebin(self.hist, [-1, 0.4, 1.])
        ht.normalise(h)
        self.assertAlmostEqual(h.Integral(), 1., delta=1e-5)

    def test_normalise_stack(self):
        stack = ROOT.THStack()
        stack.Add(self.hist)
        res = ht._normalise_1d_hist(stack)
        self.assertEqual(sum([i.Integral() for i in res.GetHists()]), self.hist.Integral())

    def test_normalise_list(self):
        ht.normalise([self.hist])
        self.assertAlmostEqual(self.hist.Integral(), 1., delta=1.e-5)

    def test_scale_hist(self):
        ht.scale(self.hist, 10.)
        self.assertEqual(self.hist.Integral(), 100000.)

    def test_get_color(self):
        self.hist.Draw('hist')
        colors = ht.get_colors([self.hist])
        self.assertTrue(colors == [602] or ht.get_colors([self.hist]) == 1)

    def test_read_bin_from_label(self):
        self.hist.GetXaxis().SetBinLabel(2, 'label')
        self.assertEqual(ht.read_bin_from_label(self.hist, 'label'), 2)

    def test_read_bin_from_label_non_existing(self):
        self.assertEqual(ht.read_bin_from_label(self.hist, 'label'), None)

    def test_read_bin_from_multi_label(self):
        self.hist.GetXaxis().SetBinLabel(2, 'label')
        self.hist.GetXaxis().SetBinLabel(3, 'label')
        self.assertEqual(ht.read_bin_from_label(self.hist, 'label'), 2)

    def test_set_axis_labels_no_labels(self):
        self.assertEqual(ht.set_axis_labels(self.hist, self.plot_config), None)

    def test_set_axis_labels(self):
        self.plot_config.axis_labels = list(map(str, list(range(self.hist.GetNbinsX()))))
        self.assertEqual(ht.set_axis_labels(self.hist, self.plot_config), None)
        self.assertEqual(self.hist.GetXaxis().GetBinLabel(5), '4')

    def test_rebin_const(self):
        h = ht.rebin(self.hist, 5)
        self.assertEqual(h.GetNbinsX(), 2)

    def test_rebin_list(self):
        h = ht.rebin(self.hist, [-1, 0.4, 1.])
        self.assertEqual(h.GetNbinsX(), 2)
        self.assertEqual(h.GetXaxis().GetBinLowEdge(1), -1.)
        self.assertEqual(h.GetXaxis().GetBinLowEdge(2), 0.4)
        self.assertEqual(h.GetXaxis().GetBinLowEdge(3), 1.0)

    def test_rebin_const_list(self):
        h = ht.rebin([self.hist], 5)
        self.assertIsInstance(h, list)
        self.assertEqual(h[0].GetNbinsX(), 2)

    def test_rebin_list_list(self):
        h = ht.rebin([self.hist], [-1, 0.4, 1.])
        self.assertIsInstance(h, list)
        self.assertEqual(h[0].GetNbinsX(), 2)
        self.assertEqual(h[0].GetXaxis().GetBinLowEdge(2), 0.4)

    def test_rebin_const_dict(self):
        h = ht.rebin({'foo': self.hist}, 5)
        self.assertIsInstance(h, dict)
        self.assertEqual(h['foo'].GetNbinsX(), 2)

    def test_rebin_list_dict(self):
        h = ht.rebin({'foo': self.hist}, [-1, 0.4, 1.])
        self.assertIsInstance(h, dict)
        self.assertEqual(h['foo'].GetNbinsX(), 2)
        self.assertEqual(h['foo'].GetXaxis().GetBinLowEdge(2), 0.4)

    def test_rebin_const_dict_list(self):
        h = ht.rebin({'foo': [self.hist]}, 5)
        self.assertIsInstance(h, dict)
        self.assertEqual(h['foo'][0].GetNbinsX(), 2)

    def test_rebin_list_dict_list(self):
        h = ht.rebin({'foo': [self.hist]}, [-1, 0.4, 1.])
        self.assertIsInstance(h, dict)
        self.assertEqual(h['foo'][0].GetNbinsX(), 2)
        self.assertEqual(h['foo'][0].GetXaxis().GetBinLowEdge(2), 0.4)

    def test_rebin_asymmetric(self):
        h = ht.rebin(self.hist, [-1, 0.4, 1.])
        self.assertEqual(h.GetNbinsX(), 2)
        self.assertAlmostEqual(h.GetBinContent(1), self.hist.Integral(self.hist.FindBin(-1.),
                                                                      self.hist.FindBin(0.39))/1.4, delta=1e-3)

    def test_rebin_asymetric_disable_binwidth_division(self):
        h = ht.rebin(self.hist, [-1, 0.4, 1.], disable_bin_width_division=True)
        self.assertEqual(h.GetNbinsX(), 2)
        self.assertEqual(h.GetNbinsX(), 2)
        self.assertEqual(h.GetBinContent(1), self.hist.Integral(self.hist.FindBin(-1.), self.hist.FindBin(0.39)))

    def test_rebin_invalid_dict(self):
        self.assertRaises(InvalidInputError, ht.rebin, {'foo': tuple(self.hist)}, [-1, 0.4, 1.])

    def test_rebin_invalid_factor(self):
        self.assertRaises(InvalidInputError, ht.rebin, self.hist, (-1, 0.4, 1.))

    def test_rebin_entity(self):
        self.assertEqual(ht.rebin(self.hist, None), self.hist)
        self.assertEqual(ht.rebin(self.hist, 1.), self.hist)

    def test_has_asymmetric_binning(self):
        h = ht.rebin(self.hist, [-1, -0.4, 1.])
        self.assertTrue(ht.has_asymmetric_binning(h))

    def test_has_asymmetric_binning_sym_binning(self):
        self.assertFalse(ht.has_asymmetric_binning(self.hist))

    def test_overflow_merge(self):
        self.hist.Fill(100)
        expected = self.hist.GetBinContent(10) + self.hist.GetBinContent(11)
        ht.merge_overflow_bins(self.hist)
        self.assertEqual(self.hist.GetBinContent(10), expected)

    def test_overflow_merge_dict(self):
        self.hist.Fill(100)
        expected = self.hist.GetBinContent(10) + self.hist.GetBinContent(11)
        ht.merge_overflow_bins({'foo': self.hist})
        self.assertEqual(self.hist.GetBinContent(10), expected)

    def test_underflow_merge(self):
        self.hist.Fill(-1100)
        expected = self.hist.GetBinContent(0) + self.hist.GetBinContent(1)
        ht.merge_underflow_bins(self.hist)
        self.assertEqual(self.hist.GetBinContent(1), expected)

    def test_underflow_merge_list(self):
        self.hist.Fill(-1100)
        expected = self.hist.GetBinContent(0) + self.hist.GetBinContent(1)
        ht.merge_underflow_bins({'foo': self.hist})
        self.assertEqual(self.hist.GetBinContent(1), expected)
