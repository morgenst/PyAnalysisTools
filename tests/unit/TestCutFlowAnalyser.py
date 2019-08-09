import unittest
import numpy as np
from ROOT import TFile

from PyAnalysisTools.base import InvalidInputError
from PyAnalysisTools.AnalysisTools.CutFlowAnalyser import CutflowAnalyser as ca


class TestCutFlowAnalyser(unittest.TestCase):
    def setUp(self):
        self.test_input_file_name = "CutflowTestInput.root"
        f = TFile(self.test_input_file_name, "READ")
        self.cutflow_raw_hist = f.Get("cutflow_raw")
        self.cutflow_raw_hist.SetDirectory(0)
        self.ca = ca(file_list=[self.test_input_file_name], dataset_config=None, systematics='')

    @unittest.skip("Old interface")
    def test_analyse_cutflow(self):
        expected = np.array([("cutflow%i" % i, pow(10. - i, 2)) for i in range(10)],
                            dtype=[("cut", "S"), ("yield", "f4")])
        parsed_info = self.ca._analyse_cutflow(self.cutflow_raw_hist, self.cutflow_raw_hist)
        self.assertTrue(all(parsed_info == expected))

    @unittest.skip("Old interface")
    def test_print_cutflow(self):
        self.ca.make_cutflow_table('')

    def test_read_cutflow_default_name(self):
        self.ca.read_cutflows()
        #self.assertEqual(self.ca.cutflow_hists, )

    @unittest.skip("Old interface")
    def test_execution(self):
        self.ca.execute()

    @unittest.skip("Not catching right exception")
    def test_non_existing_file(self):
        self.assertRaises(ValueError, ca(**{'file_list': ['NonExistingFile'],
                                                   'dataset_config': None, 'systematics': ''}))

    def test_capture_missing_cutflow(self):
        pass
        """
        test e.g. for missing cutflow Nominal/cutflow_DxAOD
        """