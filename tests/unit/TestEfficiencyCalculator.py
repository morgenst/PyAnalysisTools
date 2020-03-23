import unittest
try:
    from functools import partialmethod
except ImportError:
    pass

import six

import ROOT
from mock import patch
from PyAnalysisTools.AnalysisTools.EfficiencyCalculator import EfficiencyCalculator as ec


@classmethod
def fct_name_patch(*_, **kwargs):
    kwargs.setdefault('ret_val', None)
    return kwargs['ret_val']


class TestEfficiencyCalculator(unittest.TestCase):
    def setUp(self):
        self.calculator = ec()

    def test_ctor(self):
        calculator = ec()
        self.assertIsInstance(calculator, ec)

    @unittest.skipIf(six.PY2, "not compatible with Python 2")
    @patch.object(ec, "calculate_1d_efficiency", partialmethod(fct_name_patch, ret_val='1D'))
    @patch.object(ec, "calculate_2d_efficiency", partialmethod(fct_name_patch, ret_val='2D'))
    def test_calculate_efficiency(self):
        self.assertEqual("1D", self.calculator.calculate_efficiency(ROOT.TH1F(), ROOT.TH1F(), 'foo'))
        self.assertEqual("2D", self.calculator.calculate_efficiency(ROOT.TH2F(), ROOT.TH2F(), 'foo'))
        self.assertIsNone(self.calculator.calculate_efficiency(ROOT.TGraph(), ROOT.TGraph(), 'foo'))

    def test_calculate_1d_efficiency(self):
        h_nom = ROOT.TH1F("h_nom", "", 10, 0., 10.)
        h_denom = ROOT.TH1F("h_denom", "", 10, 0., 10.)
        for i in range(10):
            h_nom.SetBinContent(i, i)
            h_denom.SetBinContent(i, i*i)
        eff = ec.calculate_1d_efficiency(h_nom, h_denom, name='foo')
        self.assertEqual('foo', eff.GetName())
        self.assertIsInstance(eff, ROOT.TEfficiency)
        for i in range(1, 10):
            self.assertAlmostEqual(float(i)/(i*i), eff.GetEfficiency(i), delta=1.e-4)

    def test_calculate_2d_efficiency(self):
        h_nom = ROOT.TH2F("h_nom", "", 3, 0., 3., 3, 0., 3.)
        h_denom = ROOT.TH2F("h_denom", "", 3, 0., 3., 3, 0., 3.)
        for i in range(9):
            h_nom.SetBinContent(i, i)
            h_denom.SetBinContent(i, i*i)
        eff = ec.calculate_2d_efficiency(h_nom, h_denom, name='foo')
        self.assertEqual('foo', eff.GetName())
        self.assertIsInstance(eff, ROOT.TH2F)
        for i in range(1, 9):
            self.assertAlmostEqual(float(i)/(i*i), eff.GetBinContent(i), delta=1.e-4)
