import inspect

__author__ = 'marcusmorgenstern'
__mail__ = ''

import unittest
import ROOT
import os
from PyAnalysisTools.PlottingUtils import HistTools as ht
from mock import MagicMock

cwd = os.path.dirname(__file__)

class TestHistTools(unittest.TestCase):
    def setUp(self):
        pass
        root_file = ROOT.TFile.Open(os.path.join(cwd, "test.root"), "READ")
        self.unformatted_hist_1d = root_file.Get("test_hist_1")
        self.unformatted_hist_1d.SetDirectory(0)
        ROOT.SetOwnership(self.unformatted_hist_1d, False)

    def test_normalise(self):
        h = self.unformatted_hist_1d.Clone(inspect.currentframe().f_code.co_name)
        ht.normalise(h)
        self.assertEqual(h.Integral(), 1.)

    def test_normalise_list(self):
        h_clone = self.unformatted_hist_1d.Clone("cloned_unformatted_hist")
        ht.normalise([self.unformatted_hist_1d, h_clone])
        self.assertEqual(self.unformatted_hist_1d.Integral(), 1.)
        self.assertEqual(h_clone.Integral(), 1.)

    unittest.skip("Not implemented")
    def test_normalise_dict(self):
        pass
    
    def test_scale_hist(self):
        h = self.unformatted_hist_1d.Clone("test_scale_hist")
        integral = h.Integral()
        ht.scale(h, 101.)
        self.assertEqual(h.Integral(), integral * 101.)
