import os
import unittest
# from mock import MagicMock, PropertyMock, patch
from PyAnalysisTools.AnalysisTools.BDTAnalyser import SklearnBDTTrainer, BDTConfig
# from PyAnalysisTools.base import InvalidInputError


class TestBDTConfig(unittest.TestCase):
    def test_default_ctor(self):
        cfg = BDTConfig()
        self.assertEqual(4, cfg.num_layers)


class TestSklearnBDTTrainer(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        fixture_path = os.path.join(os.path.dirname(__file__), 'fixtures')
        self.bdt_analyser = SklearnBDTTrainer(training_config_file=os.path.join(fixture_path,'training_config.yml'),
                                              bdt_config_file=os.path.join(fixture_path, 'simple_bdt_config.yml'),
                                              var_list=os.path.join(fixture_path, 'varlist_training.yml'),
                                              input_file_list=[os.path.join(fixture_path,
                                                                            'files/410470_1.MC16e_Prompt.json'),
                                                               os.path.join(fixture_path,
                                                                            'files/410470_1.MC16e_NonPrompt.json')])

    def test_ctor(self):
        self.assertIsNone(self.bdt_analyser.signal_df)
        self.assertIsNone(self.bdt_analyser.bkg_df)
        self.assertIsNone(self.bdt_analyser.labels)

    def test_ctor_exception(self):
        self.assertRaises(KeyError, SklearnBDTTrainer)

    def test_load_train_data(self):
        self.bdt_analyser.load_train_data()
        self.assertIsNotNone(self.bdt_analyser.signal_df)
        self.assertIsNotNone(self.bdt_analyser.bkg_df)
        self.assertIsNotNone(self.bdt_analyser.labels)

    def test_train_bdt(self):
        self.bdt_analyser.load_train_data()
        self.bdt_analyser.train_bdt()
        self.assertTrue(os.path.exists("test.pkl"))
        os.remove("test.pkl")
