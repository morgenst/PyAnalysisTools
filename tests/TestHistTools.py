__author__ = 'marcusmorgenstern'
__mail__ = ''

import unittest
import ROOT
from PyAnalysisTools.PlottingUtils import HistTools as HT


class TestHistTools(unittest.TestCase):
    def setUp(self):
        root_file = ROOT.TFile.Open("test_data/test.root", "READ")
        self.unformatted_hist_1d = root_file.Get("test_hist_1")
        self.unformatted_hist_1d.SetDirectory(0)
        ROOT.SetOwnership(self.unformatted_hist_1d, False)

    def test_normalise(self):
            HT.normalise(self.unformatted_hist_1d)
            self.assertEqual(self.unformatted_hist_1d.Integral(), 1.)

    def test_normalise_list(self):
        h_clone = self.unformatted_hist_1d.Clone("cloned_unformatted_hist")
        HT.normalise([self.unformatted_hist_1d, h_clone])
        self.assertEqual(self.unformatted_hist_1d.Integral(), 1.)
        self.assertEqual(h_clone.Integral(), 1.)

    unittest.skip("Not implemented")
    def test_normalise_dict(self):
        pass
    
    def test_scale_hist(self):
        integral = self.unformatted_hist_1d.Integral()
        HT.scale(self.unformatted_hist_1d, 101.)
        self.assertEqual(self.unformatted_hist_1d.Integral(), integral * 101.)
