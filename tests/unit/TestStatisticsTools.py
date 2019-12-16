import unittest
import os
import pandas as pd
from copy import deepcopy

from mock import MagicMock, Mock, patch

import ROOT
from PyAnalysisTools.PlottingUtils import PlottingTools
from PyAnalysisTools.base import InvalidInputError
from .Mocks import hist
from PyAnalysisTools.AnalysisTools import StatisticsTools as st
from PyAnalysisTools.ROOTUtils.ObjectHandle import get_objects_from_canvas_by_type
cwd = os.path.dirname(__file__)


class TestStatisticTools(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_consistency_check_bins_pass(self):
        h1 = hist
        h1.GetNbinsX = MagicMock(return_value=3)
        self.assertTrue(st.consistency_check_bins(h1, h1))

    def test_consistency_check_bins_fail(self):
        h1 = deepcopy(hist)
        h1.GetNbinsX = MagicMock(return_value=3)
        h2 = deepcopy(hist)
        h2.GetNbinsX = MagicMock(return_value=4)
        self.assertFalse(st.consistency_check_bins(h1, h2))

    def test_consistency_check_bins_invalid(self):
        h1 = hist
        h1.GetNbinsX = MagicMock(return_value=3)
        self.assertRaises(InvalidInputError, st.consistency_check_bins, h1, 5)

    def test_calc_significance(self):
        self.assertEqual(2./3., st.calculate_significance(2., 9.))

    def test_calc_significance_zero_div(self):
        self.assertEqual(0., st.calculate_significance(2., 0.))

    def test_calc_significance_invalid(self):
        self.assertEqual(0., st.calculate_significance(2., -1.))

    def test_get_stat_unc_empty(self):
        self.assertIsNone(st.get_statistical_uncertainty_hist([]))

    def test_get_stat_unc(self):
        h = hist
        h.val = MagicMock(return_value=1)
        h.Clone = MagicMock(return_value=hist)
        h.Add = MagicMock(side_effect=lambda i: i.val + 1)
        stat_unc = st.get_statistical_uncertainty_hist([h, h])
        self.assertIsNotNone(stat_unc)

    def test_get_statistical_uncertainty_from_stack(self):
        h = hist
        h.val = MagicMock(return_value=1)
        h.Clone = MagicMock(return_value=hist)
        h.Add = MagicMock(side_effect=lambda i: i.val + 1)
        stack = Mock()
        stack.GetHists = MagicMock(return_value=[h, h])
        stat_unc = st. get_statistical_uncertainty_from_stack(stack)
        self.assertIsNotNone(stat_unc)

    def test_get_statistical_uncertainty_ratio_invalid(self):
        self.assertIsNone(st.get_statistical_uncertainty_ratio(None))

    def test_get_statistical_uncertainty_ratio(self):
        h = hist
        h.val = MagicMock(return_value=1)
        h.GetBinContent = MagicMock(return_value=1)
        h.GetBinError = MagicMock(return_value=1)
        h.Clone = MagicMock(return_value=hist)
        h.Add = MagicMock(side_effect=lambda i: i.val + 1)
        stat_unc = st.get_statistical_uncertainty_ratio(h)
        self.assertIsNotNone(stat_unc)

    def test_get_statistical_uncertainty_ratio_zero(self):
        h = hist
        h.val = MagicMock(return_value=1)
        h.GetBinContent = MagicMock(return_value=0)
        h.GetBinError = MagicMock(return_value=0)
        h.Clone = MagicMock(return_value=hist)
        h.Add = MagicMock(side_effect=lambda i: i.val + 1)
        stat_unc = st.get_statistical_uncertainty_ratio(h)
        self.assertIsNotNone(stat_unc)

    def test_KS(self):
        h = Mock()
        h.KolmogorovTest = MagicMock(return_value=3.)
        self.assertEqual(3., st.get_KS(h, h))

    def test_signal_acceptance(self):
        c, cl, cf = st.get_signal_acceptance({'foo1000': pd.DataFrame([{'cut': 'foo', 'yield': 10.}]),
                                              'foo2000': pd.DataFrame([{'cut': 'foo', 'yield': 20.}])},
                                             {'foo1000': 100., 'foo2000': 200.})
        self.assertTrue(isinstance(c, ROOT.TCanvas))
        self.assertTrue(isinstance(cl, ROOT.TCanvas))
        self.assertTrue(isinstance(cf, ROOT.TCanvas))
        self.assertTrue(cl.GetLogy())
        h = get_objects_from_canvas_by_type(c, 'TGraph')[0]
        self.assertEqual(2, h.GetN())

    @patch.object(PlottingTools, 'plot_obj', lambda *args: args)
    def test_get_significance(self):
        hist.GetNbinsX = MagicMock(return_value=1)
        sig_hist = deepcopy(hist)
        sig_hist.Clone = MagicMock(return_value=sig_hist)
        sig_hist.Integral = MagicMock(return_value=10.)
        bkg_hist = deepcopy(hist)
        bkg_hist.Integral = MagicMock(return_value=20.)
        x, y = st.get_significance(sig_hist, bkg_hist, None)
        self.assertEqual(sig_hist, x)
        self.assertIsNone(y)

    @patch.object(PlottingTools, 'add_object_to_canvas', lambda *args: args)
    def test_get_significance_add(self):
        hist.GetNbinsX = MagicMock(return_value=1)
        sig_hist = deepcopy(hist)
        sig_hist.Clone = MagicMock(return_value=sig_hist)
        sig_hist.Integral = MagicMock(return_value=10.)
        bkg_hist = deepcopy(hist)
        bkg_hist.Integral = MagicMock(return_value=20.)
        x = st.get_significance(sig_hist, bkg_hist, None, 'foo', upper_cut=True)
        self.assertEqual('foo', x)

    def test_get_significance_invalid_binning(self):
        sig_hist = deepcopy(hist)
        sig_hist.GetNbinsX = MagicMock(return_value=2)
        bkg_hist = deepcopy(hist)
        bkg_hist.GetNbinsX = MagicMock(return_value=1)
        self.assertRaises(InvalidInputError, st.get_significance, sig_hist, bkg_hist, None)

    def test_get_relative_systematics_ratio(self):
        hist.GetNbinsX = MagicMock(return_value=2)
        nominal = deepcopy(hist)
        stat_unc = deepcopy(hist)
        h_syst = deepcopy(hist)
        nominal.GetBinContent = MagicMock(return_value=10.)
        nominal.Clone = MagicMock(return_value=ROOT.TH1F())
        h_syst.GetBinContent = MagicMock(return_value=5.)
        stat_unc.GetBinError = MagicMock(return_value=2.)
        h_syst.GetName = MagicMock(return_value='foo')
        ratio = st.get_relative_systematics_ratio(nominal, stat_unc, [h_syst])
        self.assertEqual(1, len(ratio))
        self.assertEqual(1., ratio[0].GetBinContent(0))
        self.assertEqual(1.5, ratio[0].GetBinError(0))
        nominal.GetBinContent = MagicMock(return_value=0.)
        stat_unc.GetBinContent = MagicMock(return_value=3.5)
        ratio = st.get_relative_systematics_ratio(nominal, stat_unc, [h_syst])
        self.assertEqual(1, len(ratio))
        self.assertEqual(2.5, ratio[0].GetBinContent(0))
