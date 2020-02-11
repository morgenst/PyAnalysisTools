import os
import sys  # noqa: F401
import unittest
from copy import deepcopy
import mock
import ROOT  # noqa: F401
from PyAnalysisTools.base.FileHandle import FileHandle
from PyAnalysisTools.base.YAMLHandle import YAMLLoader
from tabulate import tabulate  # noqa: F401


@classmethod
def patch(*args):
    return {}


with mock.patch.dict('sys.modules', {'pyAMI': mock.MagicMock()}):
    from PyAnalysisTools.AnalysisTools.NTupleAnalyser import NTupleAnalyser


    class TestNTupleAnalyser(unittest.TestCase):
        def setUp(self):
            self.patcher = mock.patch.object(NTupleAnalyser, 'check_valid_proxy')
            self.patcher.start()
            self.analyser = NTupleAnalyser(dataset_list=os.path.join(os.path.dirname(__file__),
                                                                     'fixtures/datasetlist.yml'),
                                           merge_mode='foo')

            self.addCleanup(self.patcher.stop)

        def test_ctor(self):
            analyser = NTupleAnalyser(dataset_list=os.path.join(os.path.dirname(__file__), 'fixtures/datasetlist.yml'))
            self.assertIsNone(analyser.filter)
            self.assertIsNone(analyser.merge_mode)
            self.assertFalse(analyser.resubmit)
            self.assertEqual('.', analyser.input_path)
            self.assertEqual('$$in[2].in[6]', analyser.grid_name_pattern)
            self.assertIsInstance(analyser.datasets, dict)

        def test_ctor_no_ds_keys(self):
            with mock.patch.object(YAMLLoader, 'read_yaml', patch):
                analyser = NTupleAnalyser(dataset_list=os.path.join(os.path.dirname(__file__), 'fixtures/datasetlist2.yml'))
                self.assertIsNone(analyser.grid_name_pattern)

        def test_ctor_filter(self):
            analyser = NTupleAnalyser(dataset_list=os.path.join(os.path.dirname(__file__), 'fixtures/datasetlist.yml'),
                                      filter='data')
            expected_ds = {'mc16a_13TeV':
                               ['mc16_13TeV.364100.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV0_70_CVetoBVeto.deriv.'
                                'DAOD_EXOT12.e5271_s3126_r9364_r9315_p3978',]}

            self.assertEqual('data', analyser.filter)
            self.assertEqual(expected_ds, analyser.datasets)

        def test_transform_dataset_list(self):
            self.analyser.transform_dataset_list()
            expected_ds_list = [['data15_13TeV.periodD.physics_Main.PhysCont.DAOD_EXOT12.grp15_v01_p3987',
                                 'periodD.grp15_v01_p3987'],
                                ['mc16_13TeV.364100.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV0_70_CVetoBVeto.deriv.'
                                 'DAOD_EXOT12.e5271_s3126_r9364_r9315_p3978',
                                 '364100.e5271_s3126_r9364_r9315_p3978']]
            self.assertEqual(expected_ds_list, self.analyser.datasets)

        @mock.patch.object(os, 'listdir', lambda _: ['foo/user.364100.e5271_s3126_r9364_r9315_p3978LQ_v12'])
        def test_add_path(self):
            self.analyser.transform_dataset_list()
            self.analyser.add_path()
            expected_ds_list = [['data15_13TeV.periodD.physics_Main.PhysCont.DAOD_EXOT12.grp15_v01_p3987',
                                 'periodD.grp15_v01_p3987', None],
                                ['mc16_13TeV.364100.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV0_70_CVetoBVeto.deriv.'
                                 'DAOD_EXOT12.e5271_s3126_r9364_r9315_p3978',
                                 '364100.e5271_s3126_r9364_r9315_p3978',
                                 ['foo/user.364100.e5271_s3126_r9364_r9315_p3978LQ_v12']]]
            self.assertEqual(expected_ds_list, self.analyser.datasets)

        @mock.patch.object(os, 'listdir', lambda _: list(range(1)))
        @mock.patch.object(os.path, 'exists', lambda _: True)
        @mock.patch.object(os.path, 'join', lambda *_: 'foo')
        @mock.patch.object(FileHandle, 'get_daod_events', lambda _: 10)
        @mock.patch.object(FileHandle, '__init__', lambda *args, **kwargs: None)
        @mock.patch.object(FileHandle, '__del__', lambda _: None)
        def test_get_events(self):
            self.analyser.transform_dataset_list()
            expected_ds_list = [['data15_13TeV.periodD.physics_Main.PhysCont.DAOD_EXOT12.grp15_v01_p3987',
                                 'periodD.grp15_v01_p3987', None],
                                ['mc16_13TeV.364100.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV0_70_CVetoBVeto.deriv.'
                                 'DAOD_EXOT12.e5271_s3126_r9364_r9315_p3978',
                                 '364100.e5271_s3126_r9364_r9315_p3978',
                                 ['foo/user.364100.e5271_s3126_r9364_r9315_p3978LQ_v12']]]
            test_ds_list = deepcopy(expected_ds_list)
            self.analyser.get_events(test_ds_list[1])
            expected_ds_list[-1] += [10, 1]
            self.assertEqual(expected_ds_list, test_ds_list)

        def test_print_summary(self):
            missing = [['data15_13TeV.periodD.physics_Main.PhysCont.DAOD_EXOT12.grp15_v01_p3987',
                       'periodD.grp15_v01_p3987', None]]
            incomplete = [['mc16_13TeV.364100.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV0_70_CVetoBVeto.deriv.'
                                 'DAOD_EXOT12.e5271_s3126_r9364_r9315_p3978',
                                 '364100.e5271_s3126_r9364_r9315_p3978',
                                 ['foo/user.364100.e5271_s3126_r9364_r9315_p3978LQ_v12'], 10, 1]]
            duplicated = [['mc16_13TeV.364100.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV0_70_CVetoBVeto.deriv.'
                          'DAOD_EXOT12.e5271_s3126_r9364_r9315_p3978',
                          '364100.e5271_s3126_r9364_r9315_p3978',
                          ['foo/user.364100.e5271_s3126_r9364_r9315_p3978LQ_v12',
                           'foo/user.364100.e5271_s3126_r9364_r9315_p3978LQ_v13'],
                          10, 1]]
            #self.assertIsNone(self.analyser.print_summary([], [], []))
            self.assertIsNone(self.analyser.print_summary(missing, [], []))
            self.assertIsNone(self.analyser.print_summary([], incomplete, []))
            self.assertIsNone(self.analyser.print_summary(missing, incomplete, duplicated))

        def test_prepare_resubmit(self):
            missing = [['data15_13TeV.periodD.physics_Main.PhysCont.DAOD_EXOT12.grp15_v01_p3987',
                       'periodD.grp15_v01_p3987', ['foo/user.data']]]
            incomplete = [['mc16_13TeV.364100.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV0_70_CVetoBVeto.deriv.'
                                 'DAOD_EXOT12.e5271_s3126_r9364_r9315_p3978',
                                 '364100.e5271_s3126_r9364_r9315_p3978',
                                 ['foo/user.364100.e5271_s3126_r9364_r9315_p3978LQ_v12'], 10, 1]]
            self.assertIsNone(self.analyser.prepare_resubmit(missing, incomplete))
            resub_file = os.path.join(os.path.dirname(__file__), 'fixtures/datasetlist_resubmit.yml')
            self.assertTrue(os.path.exists(resub_file))
            resub = YAMLLoader.read_yaml(resub_file)
            self.assertListEqual(list(resub.keys()), ['incomplete$$in[2].in[6]', 'missing$$in[2].in[6]'])
            self.assertListEqual(list(resub.values()),
                                 [['data15_13TeV.periodD.physics_Main.PhysCont.DAOD_EXOT12.grp15_v01_p3987'],
                                 ['mc16_13TeV.364100.Sherpa_221_NNPDF30NNLO_Zmumu_MAXHTPTV0_70_CVetoBVeto.deriv.'
                                 'DAOD_EXOT12.e5271_s3126_r9364_r9315_p3978']])

        def test_run(self):
            self.assertIsNone(self.analyser.run())
