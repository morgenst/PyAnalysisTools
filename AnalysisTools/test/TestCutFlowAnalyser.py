__author__ = 'marcusmorgenstern'
__mail__ = ''

import unittest
from ROOT import TFile


class TestCutFlowAnalyser(unittest.TestCase):
    def setUp(self):
        self.test_input_file_name = "CutflowTestInput.root"
        f =  TFile(self.test_input_file_name, "READ")
        self.cutflow_raw_hist = f.Get("cutflow_raw")
        self.cutflow_raw_hist.SetDirectory(0)

