import re
import unittest
import os
from PyAnalysisTools.base.ProcessConfig import Process
from PyAnalysisTools.base.YAMLHandle import YAMLLoader as yl

cwd = os.path.dirname(__file__)


class TestProcess(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.data_set_info = yl.read_yaml(os.path.join(os.path.dirname(__file__), 'fixtures/dataset_info_pmg.yml'))

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_process_file_name_ntuple(self):
        process = Process('tmp/ntuple-311570_0.MC16a.root', self.data_set_info)
        self.assertTrue(process.is_mc)
        self.assertFalse(process.is_data)
        self.assertEqual('TBbLQmumu1300l1', process.process_name)
        self.assertEqual('311570', process.dsid)
        self.assertEqual('mc16a', process.mc_campaign)

    def test_process_file_name_hist(self):
        process = Process('tmp/hist-311570_0.MC16d.root', self.data_set_info)
        self.assertTrue(process.is_mc)
        self.assertFalse(process.is_data)
        self.assertEqual('TBbLQmumu1300l1', process.process_name)
        self.assertEqual('311570', process.dsid)
        self.assertEqual('mc16d', process.mc_campaign)

    def test_process_file_name_arbitrary_tag(self):
        process = Process('tmp/foo-311570_0.MC16e.root', self.data_set_info, tags=['foo'])
        self.assertTrue(process.is_mc)
        self.assertFalse(process.is_data)
        self.assertEqual('TBbLQmumu1300l1', process.process_name)
        self.assertEqual('311570', process.dsid)
        self.assertEqual('mc16e', process.mc_campaign)

    def test_process_file_name_data(self):
        process = Process('v21/ntuple-data18_13TeV_periodO_0.root', self.data_set_info, tags=['foo'])
        self.assertFalse(process.is_mc)
        self.assertTrue(process.is_data)
        self.assertEqual('data18.periodO', process.process_name)
        self.assertIsNone(process.dsid)
        self.assertIsNone(process.mc_campaign)

    def test_process_no_file_name(self):
        process = Process(None, self.data_set_info, tags=['foo'])
        self.assertFalse(process.is_mc)
        self.assertFalse(process.is_data)
        self.assertIsNone(process.process_name)
        self.assertIsNone(process.dsid)

    def test_process_unconvential_file_name(self):
        process = Process('tmp/hist-333311570_0.MC16e.root', self.data_set_info)
        self.assertTrue(process.is_mc)
        self.assertFalse(process.is_data)
        self.assertEqual(None, process.process_name)
        self.assertEqual('333311570', process.dsid)

    def test_str_operator(self):
        process = Process('tmp/hist-311570_0.MC16d.root', self.data_set_info)
        self.assertEqual('TBbLQmumu1300l1 parsed from file name tmp/hist-311570_0.MC16d.root', process.__str__())

    def test_equality(self):
        process1 = Process('tmp/hist-311570_0.MC16d.root', self.data_set_info)
        process2 = Process('tmp/hist-311570_0.MC16d.root', self.data_set_info)
        self.assertEqual(process1, process2)

    def test_equality_different_files(self):
        process1 = Process('tmp/hist-311570_0.MC16d.root', self.data_set_info)
        process2 = Process('tmp/ntuple-311570_0.MC16d.root', self.data_set_info)
        self.assertEqual(process1, process2)

    def test_inequality(self):
        process1 = Process('tmp/hist-311570_0.MC16d.root', self.data_set_info)
        process2 = Process('tmp/hist-311570_0.MC16e.root', self.data_set_info)
        self.assertNotEqual(process1, process2)
    def test_inequality_type(self):
        process = Process('tmp/hist-311570_0.MC16d.root', self.data_set_info)
        self.assertNotEqual(process, None)

    def test_match_true(self):
        process = Process('tmp/hist-311570_0.MC16d.root', self.data_set_info)
        self.assertTrue(process.match('TBbLQmumu1300l1'))

    def test_match_false(self):
        process = Process('tmp/hist-311570_0.MC16d.root', self.data_set_info)
        self.assertFalse(process.match('TBbLQmumu1400l1'))

    def test_match_fals_no_process(self):
        process = Process('tmp/hist-333311570_0.MC16e.root', self.data_set_info)
        self.assertFalse(process.match('TBbLQmumu1400l1'))

    def test_match_any_true(self):
        process = Process('tmp/hist-311570_0.MC16d.root', self.data_set_info)
        self.assertTrue(process.matches_any(['TBbLQmumu1300l1']))

    def test_match_any_false(self):
        process = Process('tmp/hist-311570_0.MC16d.root', self.data_set_info)
        self.assertFalse(process.matches_any(['TBbLQmumu1400l1']))

    def test_match_any_false_invalid_input(self):
        process = Process('tmp/hist-311570_0.MC16d.root', self.data_set_info)
        self.assertFalse(process.matches_any('TBbLQmumu1400l1'))

    def test_with_cut(self):
        process1 = Process('tmp/hist-311570_0.MC16d.root', self.data_set_info, cut='foo')
        process2 = Process('tmp/hist-311570_0.MC16d.root', self.data_set_info, cut='bar')
        self.assertNotEqual(process1, process2)
        self.assertNotEqual(process1.process_name, process2.process_name)
        self.assertEqual(process1.dsid, process2.dsid)

    def test_process_file_name_data(self):
        process = Process('v21/ntuple-mc16_311570_MC16e.root', self.data_set_info, tags=['foo'])
        self.assertTrue(process.is_mc)
        self.assertFalse(process.is_data)
        self.assertEqual('TBbLQmumu1300l1', process.process_name)
        self.assertEqual('311570', process.dsid)
        self.assertEqual('mc16e', process.mc_campaign)

    def test_process_file_name_hist_full_path(self):
        process = Process('/Users/foo/tmp/test/hists_20200206_18-04-21/hist-364106_1.MC16d.root',
                          self.data_set_info, tags=['foo'])
        self.assertTrue(process.is_mc)
        self.assertFalse(process.is_data)
        self.assertEqual('ZmumuHT140280CVetoBVeto', process.process_name)
        self.assertEqual('364106', process.dsid)
        self.assertEqual('mc16d', process.mc_campaign)

    def test_process_file_name_data_user(self):
        process = Process('~/user.foo.data18_13TeV.periodAllYear.physics_Late.pro24_v01.v8_hist/user.foo.2._000001.hist-output.root', self.data_set_info, tags=['foo'])
        self.assertFalse(process.is_mc)
        self.assertTrue(process.is_data)
        self.assertTrue(re.match(r'.*data.*', process.process_name))

    def test_process_file_name_data_cos(self):
        process = Process('hist-data16_cos.00306147.physics_Main.cosmicsStandardOFCs.root', None)
        self.assertTrue(process.is_data)
        self.assertFalse(process.is_mc)
        self.assertEqual('data16_00306147', process.process_name)
        self.assertIsNone(process.period)
        self.assertIsNone(process.weight)

    def test_process_weight(self):
        process = Process('v21/ntuple-mc16_311570_MC16e.root', self.data_set_info, weight='foo')
        self.assertEqual('foo', process.weight)
