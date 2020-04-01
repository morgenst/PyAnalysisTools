import os
import unittest
import re
from mock import MagicMock
from PyAnalysisTools.base.ProcessConfig import ProcessConfig, parse_and_build_process_config, find_process_config, \
    Process
from PyAnalysisTools.base.YAMLHandle import YAMLLoader as yl


cwd = os.path.dirname(__file__)


class TestProcess(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.data_set_info = yl.read_yaml(os.path.join(cwd, 'fixtures/dataset_info_pmg.yml'))

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_str(self):
        process = Process('tmp/ntuple-311570_0.MC16a.root', self.data_set_info)
        self.assertEqual("TBbLQmumu1300l1 parsed from file name tmp/ntuple-311570_0.MC16a.root", process.__str__())

    def test_unicode(self):
        process = Process('tmp/ntuple-311570_0.MC16a.root', self.data_set_info)
        self.assertEqual("TBbLQmumu1300l1 parsed from file name tmp/ntuple-311570_0.MC16a.root", process.__unicode__())

    def test_format(self):
        process = Process('tmp/ntuple-311570_0.MC16a.root', self.data_set_info)
        self.assertEqual("TBbLQmumu1300l1 parsed from file name tmp/ntuple-311570_0.MC16a.root", "{:s}".format(process))

    def test_hash(self):
        process = Process('tmp/ntuple-311570_0.MC16a.root', self.data_set_info)
        self.assertEqual(hash("TBbLQmumu1300l1"), hash(process))

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

    def test_process_file_name_mc(self):
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

    def test_process_file_name_ntuple_data_full_path(self):
        process = Process('/storage/hepgrp/morgens/LQ/ntuples/v29_merged/ntuple-data17_13TeV_periodK_0.root',
                          self.data_set_info, tags=['foo'])
        self.assertFalse(process.is_mc)
        self.assertTrue(process.is_data)
        self.assertEqual('data17.periodK', process.process_name)

    def test_process_file_name_data_user(self):
        fname = '~/user.foo.data18_13TeV.periodAllYear.physics_Late.pro24_v01.v8_hist/user.foo.2._001.hist-output.root'
        process = Process(fname, self.data_set_info, tags=['foo'])
        self.assertFalse(process.is_mc)
        self.assertTrue(process.is_data)
        self.assertTrue(re.match(r'.*data.*', process.process_name))

    def test_process_file_name_data_run(self):
        fname = '~/v8/ntuple-data16_cos_306147_physics_Main_cosmicsReco.root'
        process = Process(fname, self.data_set_info, tags=['foo'])
        self.assertFalse(process.is_mc)
        self.assertTrue(process.is_data)
        self.assertTrue(re.match(r'.*data16.*306147.*', process.process_name))

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


class TestProcessConfig(unittest.TestCase):
    def setUp(self):
        self.pc = ProcessConfig(name='foo', type='data')
        self.cfg_file = os.path.join(os.path.dirname(__file__), 'fixtures/process_merge_config.yml')

    def test_ctor(self):
        self.assertIsNone(self.pc.parent_process)
        self.assertIsNone(self.pc.scale_factor)
        self.assertIsNone(self.pc.regions_only)
        self.assertIsNone(self.pc.weight)
        self.assertIsNone(self.pc.assoc_process)
        self.assertTrue(self.pc.is_data)
        self.assertFalse(self.pc.is_syst_process)
        self.assertFalse(self.pc.is_mc)

    def test_str(self):
        process_cfg_str = self.pc.__str__()
        self.assertTrue("Process config: foo \n" in process_cfg_str)
        self.assertTrue("name=foo \n" in process_cfg_str)
        self.assertTrue("type=data \n" in process_cfg_str)
        self.assertTrue("is_syst_process=False \n" in process_cfg_str)
        self.assertTrue("parent_process=None \n" in process_cfg_str)
        self.assertTrue("scale_factor=None \n" in process_cfg_str)
        self.assertTrue("is_mc=False \n" in process_cfg_str)
        self.assertTrue("weight=None \n" in process_cfg_str)
        self.assertTrue("is_data=True \n" in process_cfg_str)
        self.assertTrue("regions_only=None \n" in process_cfg_str)
        self.assertTrue("assoc_process=None \n" in process_cfg_str)
        self.assertTrue("type=data \n" in process_cfg_str)

    def test_unicode(self):
        process_cfg_str = self.pc.__unicode__()
        self.assertTrue("Process config: foo \n" in process_cfg_str)
        self.assertTrue("name=foo \n" in process_cfg_str)
        self.assertTrue("type=data \n" in process_cfg_str)
        self.assertTrue("is_syst_process=False \n" in process_cfg_str)
        self.assertTrue("parent_process=None \n" in process_cfg_str)
        self.assertTrue("scale_factor=None \n" in process_cfg_str)
        self.assertTrue("is_mc=False \n" in process_cfg_str)
        self.assertTrue("weight=None \n" in process_cfg_str)
        self.assertTrue("is_data=True \n" in process_cfg_str)
        self.assertTrue("regions_only=None \n" in process_cfg_str)
        self.assertTrue("assoc_process=None \n" in process_cfg_str)
        self.assertTrue("type=data \n" in process_cfg_str)

    def test_repr(self):
        process_cfg_str = self.pc.__repr__()
        self.assertTrue("Process config: foo \n" in process_cfg_str)
        self.assertTrue("name=foo \n" in process_cfg_str)
        self.assertTrue("type=data \n" in process_cfg_str)
        self.assertTrue("is_syst_process=False \n" in process_cfg_str)
        self.assertTrue("parent_process=None \n" in process_cfg_str)
        self.assertTrue("scale_factor=None \n" in process_cfg_str)
        self.assertTrue("is_mc=False \n" in process_cfg_str)
        self.assertTrue("weight=None \n" in process_cfg_str)
        self.assertTrue("is_data=True \n" in process_cfg_str)
        self.assertTrue("regions_only=None \n" in process_cfg_str)
        self.assertTrue("assoc_process=None \n" in process_cfg_str)
        self.assertTrue("type=data \n" in process_cfg_str)

    def test_parse_and_build_process_config(self):
        cfgs = parse_and_build_process_config(self.cfg_file)
        self.assertTrue('Data' in cfgs)

    def test_parse_and_build_process_config_lsit(self):
        cfgs = parse_and_build_process_config([self.cfg_file])
        self.assertTrue('Data' in cfgs)

    def test_parse_and_build_process_config_no_file(self):
        self.assertIsNone(parse_and_build_process_config(None))

    def test_parse_and_build_process_config_non_existing_file_exception(self):
        try:
            self.assertRaises(FileNotFoundError, parse_and_build_process_config, 'foo')
        except NameError:
            self.assertRaises(IOError, parse_and_build_process_config, 'foo')

    def test_find_process_config_missing_input(self):
        self.assertIsNone(find_process_config(None, MagicMock()))
        self.assertIsNone(find_process_config(MagicMock(), None))

    def test_find_process_config(self):
        cfgs = parse_and_build_process_config(self.cfg_file)
        self.assertEqual(cfgs['Data'], find_process_config('data18_13TeV_periodB', cfgs))

    def test_find_process_config_direct_cfg_match(self):
        cfgs = parse_and_build_process_config(self.cfg_file)
        self.assertEqual(cfgs['Data'], find_process_config('Data', cfgs))

    def test_find_process_config_no_regex(self):
        cfgs = parse_and_build_process_config(self.cfg_file)
        cfgs['Data'].subprocesses = ['data18_13TeV_periodB']
        self.assertEqual(cfgs['Data'], find_process_config('data18_13TeV_periodB', cfgs))

    def test_find_process_config_no_subprocess(self):
        cfgs = parse_and_build_process_config(self.cfg_file)
        delattr(cfgs['Data'], 'subprocesses')
        self.assertIsNone(find_process_config('data18_13TeV_periodB', cfgs))

    def test_find_process_config_multiple_matches(self):
        cfgs = parse_and_build_process_config(self.cfg_file)
        cfgs['tmp'] = cfgs['Data']
        self.assertIsNone(find_process_config('data18_13TeV_periodB', cfgs))

    def test_find_process_config_process(self):
        cfgs = parse_and_build_process_config(self.cfg_file)
        self.assertEqual(cfgs['Data'], find_process_config(Process('data18_13TeV_periodB', None), cfgs))

    def test_find_process_config_direct_cfg_match_process(self):
        cfgs = parse_and_build_process_config(self.cfg_file)
        self.assertEqual(cfgs['Data'], find_process_config(Process('Data', None), cfgs))

    def test_find_process_config_no_regex_process(self):
        cfgs = parse_and_build_process_config(self.cfg_file)
        cfgs['Data'].subprocesses = ['data18.periodB']
        self.assertEqual(cfgs['Data'], find_process_config(Process('data18_13TeV_periodB', None), cfgs))

    def test_find_process_config_no_subprocess_process(self):
        cfgs = parse_and_build_process_config(self.cfg_file)
        delattr(cfgs['Data'], 'subprocesses')
        self.assertIsNone(find_process_config(Process('data18_13TeV_periodB', None), cfgs))

    def test_find_process_config_multiple_matches_process(self):
        cfgs = parse_and_build_process_config(self.cfg_file)
        cfgs['tmp'] = cfgs['Data']
        self.assertIsNone(find_process_config(Process('data18_13TeV_periodB', None), cfgs))

