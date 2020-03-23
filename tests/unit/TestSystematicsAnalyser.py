import os
import unittest
from copy import deepcopy
from functools import partial

import six
from mock import MagicMock, patch, Mock

import ROOT
from PyAnalysisTools.AnalysisTools import SystematicsAnalyser as sa, MLHelper
from PyAnalysisTools.AnalysisTools.LimitHelpers import Yield
from PyAnalysisTools.AnalysisTools.SystematicsAnalyser import SystematicsCategory
from PyAnalysisTools.PlottingUtils.BasePlotter import BasePlotter
from PyAnalysisTools.PlottingUtils.PlotConfig import PlotConfig
from PyAnalysisTools.base.FileHandle import FileHandle
from PyAnalysisTools.base.ProcessConfig import Process
from PyAnalysisTools.base.YAMLHandle import YAMLLoader as yl
from .Mocks import hist
import numpy as np

cwd = os.path.dirname(__file__)


def entity_patch(*args, **kwargs):
    return args, kwargs


@classmethod
def mock_read_yaml(*args):
    return {'foo': 'bar'}


@classmethod
def mock_read_yaml2(*args):
    return {'foo': {'bar': 'foobar'}}


class TestSystematicsCategory(unittest.TestCase):
    def test_ctor(self):
        category = sa.SystematicsCategory()
        self.assertEqual("total", category.name)
        self.assertEqual("", category.systematics)
        self.assertEqual(ROOT.kBlue, category.color)

    def test_contain_check(self):
        category = sa.SystematicsCategory(systematics=['foo', 'bar'])
        self.assertTrue(category.contains_syst('total'))
        self.assertTrue(category.contains_syst('foo'))
        self.assertTrue(category.contains_syst('missing'))
        category = sa.SystematicsCategory(name='single', systematics=['foo', 'bar'])
        self.assertTrue(category.contains_syst('foo'))
        self.assertFalse(category.contains_syst('missing'))


class TestSystematics(unittest.TestCase):
    def setUp(self):
        self.syst = sa.Systematic('foo', variation='updown')
        self.scale_syst = sa.Systematic('foo', type='scale', variation='single:up')

    def tearDown(self):
        del self.syst
        del self.scale_syst

    def test_ctor_default(self):
        syst = sa.Systematic('foo')
        self.assertEqual('', syst.prefix)
        self.assertFalse(syst.symmetrise)
        self.assertIsNone(syst.symmetrise_option)
        self.assertIsNone(syst.call)
        self.assertIsNone(syst.affects)
        self.assertIsNone(syst.samples)
        self.assertIsNone(syst.group)
        self.assertIsNone(syst.title)
        self.assertIsNone(syst.type)
        self.assertIsNone(syst.expand)
        self.assertIsNone(syst.hist_name)
        self.assertIsNone(syst.variation)
        self.assertIsNone(syst.envelope)

    def test_ctor_scale(self):
        self.assertEqual('weight_', self.scale_syst.prefix)

    def test_str_op(self):
        self.assertEqual('Systematic uncertainty: foo', self.syst.__str__())

    def test_repr_op(self):
        self.assertEqual('Systematic uncertainty: foo\n', self.syst.__repr__())

    def test_get_variations_updown(self):
        self.assertEqual(['foo__1up', 'foo__1down'], self.syst.get_variations())

    def test_get_variations_scale(self):
        self.assertEqual(['weight_foo__1up'], self.scale_syst.get_variations())

    def test_get_variations_empty(self):
        self.assertEqual([], sa.Systematic('foo').get_variations())

    def test_get_variations_unsupported(self):
        syst = sa.Systematic('foo', type='scale', variation='foo', symmetrise=True)
        self.assertEqual([], syst.get_variations())

    def test_get_symmetrised_name_nosym(self):
        self.assertIsNone(self.syst.get_symmetrised_name())

    def test_get_symmetrised_name_invalid(self):
        syst = sa.Systematic('foo', type='scale', variation='single:foo', symmetrise=True)
        self.assertIsNone(syst.get_symmetrised_name())

    def test_get_symmetrised_name_up(self):
        scale_syst = sa.Systematic('foo', type='scale', variation='single:up', symmetrise=True)
        self.assertEqual(['weight_foo__1down'], scale_syst.get_symmetrised_name())

    def test_get_symmetrised_name_down(self):
        scale_syst = sa.Systematic('foo', type='scale', variation='single:down', symmetrise=True)
        self.assertEqual(['weight_foo__1up'], scale_syst.get_symmetrised_name())

    def test_eq(self):
        syst1 = sa.Systematic('foo', type='scale', variation='single:down', symmetrise=True)
        syst2 = sa.Systematic('foo', type='scale', variation='single:down', symmetrise=True)
        self.assertEqual(syst1, syst2)

    def test_inequal(self):
        syst1 = sa.Systematic('foo', type='scale', variation='single:down', symmetrise=True)
        syst2 = sa.Systematic('foo', type='shape', variation='single:down', symmetrise=True)
        self.assertNotEqual(syst1, syst2)

    def test_inequal_missing_key(self):
        syst1 = sa.Systematic('foo', type='shape', variation='single:down', symmetrise=True, foo='bar')
        syst2 = sa.Systematic('foo', type='shape', variation='single:down', symmetrise=True)
        self.assertNotEqual(syst1, syst2)

    def test_inequal_missing_key_swap(self):
        syst1 = sa.Systematic('foo', type='scale', variation='single:down', symmetrise=True)
        syst2 = sa.Systematic('foo', type='shape', variation='single:down', symmetrise=True, foo='bar')
        self.assertNotEqual(syst1, syst2)

    def test_inequal_type(self):
        syst1 = sa.Systematic('foo', type='scale', variation='single:down', symmetrise=True)
        self.assertNotEqual(1., syst1)


syst = sa.Systematic('foo', bar='foobar', type='shape', variation='updown')
scale_syst = sa.Systematic('foo_scale', bar='foobar_scale', type='scale', variation='updown')


class TestSystematicsAnalyser(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_ctor_default(self):
        analyser = sa.SystematicsAnalyser(xs_handle='foo')
        self.assertEqual(1, analyser.ncpu)
        self.assertFalse(analyser.dump_hists)
        self.assertFalse(analyser.cluster_mode)
        self.assertFalse(analyser.store_abs_syst)
        self.assertIsNone(analyser.systematics)
        self.assertEqual({}, analyser.systematic_hists)
        self.assertEqual({}, analyser.systematic_variations)
        self.assertEqual({}, analyser.total_systematics)
        self.assertEqual([], analyser.scale_uncerts)
        self.assertEqual([], analyser.shape_uncerts)
        self.assertEqual([], analyser.file_handles)
        self.assertEqual([], analyser.custom_uncerts)
        self.assertEqual('foo', analyser.xs_handle)
        self.assertTrue(analyser.disable)

    @patch.object(yl, 'read_yaml', mock_read_yaml2)
    def test_parse_syst_cfg(self):
        systs = sa.SystematicsAnalyser.parse_syst_config('foo')
        self.assertEqual([sa.Systematic('foo', bar='foobar')], systs)

    @patch.object(FileHandle, 'get_object_by_name',
                  lambda *args: ROOT.TH1F('h_syst', '', 10, 0., 10.))
    def test_load_dumped_hists(self):
        fh = FileHandle(file_name='foo', process=Process('foo', dataset_info=None))
        pc, process, hist = sa.SystematicsAnalyser.load_dumped_hist((fh, PlotConfig()), 'foo')
        self.assertEqual(PlotConfig(), pc)
        self.assertEqual('foo', process.process_name)
        self.assertEqual('h_syst', hist.GetName())

    @patch.object(FileHandle, 'get_object_by_name',
                  lambda *args: None)
    def test_load_dumped_hists_tuple(self):
        fh = FileHandle(file_name='foo', process=Process('foo', dataset_info=None))
        self.assertEqual([(PlotConfig(), None, None)], sa.SystematicsAnalyser(xs_handle='foo').load_dumped_hists([fh],
                                                                                                         [PlotConfig()],
                                                                                                         'foo'))

    def test_load_dumped_hists_theory(self):
        fh = FileHandle(file_name='foo', process=Process('foo', dataset_info=None))
        pc, process, hist = sa.SystematicsAnalyser.load_dumped_hist((fh, PlotConfig()),
                                                                    'weight_pdf_uncert_MUR0p5_MUF0p5_PDF261000')
        self.assertIsNone(pc)
        self.assertIsNone(process)
        self.assertIsNone(hist)

    def test_retrieve_sys_hists_no_input_files(self):
        fh = FileHandle(file_name='foo', process=Process('data', dataset_info=None))
        analyser = sa.SystematicsAnalyser(file_handles=[fh], xs_handle='foo')
        self.assertEqual(1, len(analyser.file_handles))
        self.assertIsNone(analyser.retrieve_sys_hists())

    def test_retrieve_sys_hists_disable(self):
        self.assertIsNone(sa.SystematicsAnalyser(xs_handle='foo').retrieve_sys_hists())

    def test_retrieve_sys_hists_data_only(self):
        fh = FileHandle(file_name='foo', process=Process('foo', dataset_info=None))
        self.assertIsNone(sa.SystematicsAnalyser(file_handles=[fh], xs_handle='foo').retrieve_sys_hists())

    @patch.object(BasePlotter, 'read_histograms', lambda *args, **kwargs: [(PlotConfig(), 'foo',
                                                                            ROOT.TH1F('h_syst', '', 10, 0., 10.))])
    @patch.object(BasePlotter, 'apply_lumi_weights', lambda *args, **kwargs: None)
    @patch.object(sa.SystematicsAnalyser, 'parse_syst_config', lambda *args: [syst, scale_syst])
    @patch.object(BasePlotter, 'merge_histograms', lambda _: None)
    def test_retrieve_sys_hists(self):
        fh = FileHandle(file_name='foo', process=Process('foo_311011', dataset_info=None))
        print(sa.SystematicsAnalyser(file_handles=[fh], xs_handle='foo',
                                     plot_configs=[PlotConfig()]).retrieve_sys_hists())

    def test_get_variation_for_process_no_pc(self):
        dummy_process = Process('foo', dataset_info=None)
        fh = FileHandle(file_name='foo', process=dummy_process)
        analyser = sa.SystematicsAnalyser(file_handles=[fh], xs_handle='foo')
        analyser.systematic_hists = {'my_syst_foo': {PlotConfig(): [hist]}}
        self.assertEqual('Nominal', analyser.get_variation_for_process(dummy_process, 'Nominal', PlotConfig(name='bar'),
                                                                       'my_syst_foo'))

    def test_get_variation_for_process_theory_envelop_store_abs(self):
        dummy_process = Process('foo', dataset_info=None)
        pc = PlotConfig()
        fh = FileHandle(file_name='foo', process=dummy_process)
        analyser = sa.SystematicsAnalyser(file_handles=[fh], xs_handle='foo', store_abs_syst=True)
        analyser.systematic_hists = {'theory_envelop': {pc: [hist]}}
        new_hist = analyser.get_variation_for_process(dummy_process, hist, pc, 'theory_envelop')
        self.assertIsInstance(new_hist, ROOT.TH1)
        self.assertEqual(2, new_hist.GetNbinsX())
        self.assertEqual('Clone', new_hist.GetName())

    def test_get_variation_for_process_store_abs(self):
        dummy_process = Process('foo', dataset_info=None)
        pc = PlotConfig()
        fh = FileHandle(file_name='foo', process=dummy_process)
        analyser = sa.SystematicsAnalyser(file_handles=[fh], xs_handle='foo', store_abs_syst=True)
        analyser.systematic_hists = {'pdf_uncert_MUR0p5_MUF0p5_PDF261000': {pc: {dummy_process: [hist.Clone()]}}}

        new_hist = analyser.get_variation_for_process(dummy_process, hist, pc, 'pdf_uncert_MUR0p5_MUF0p5_PDF261000')[0]
        self.assertIsInstance(new_hist, ROOT.TH1)
        self.assertEqual(2, new_hist.GetNbinsX())
        self.assertEqual('Clone', new_hist.GetName())

    def test_get_variation_for_process_theory_envelop(self):
        dummy_process = Process('foo', dataset_info=None)
        pc = PlotConfig()
        fh = FileHandle(file_name='foo', process=dummy_process)
        analyser = sa.SystematicsAnalyser(file_handles=[fh], xs_handle='foo', store_abs_syst=True)
        analyser.systematic_hists = {'theory_envelop': {pc: [hist]}}
        new_hist = analyser.get_variation_for_process(dummy_process, hist, pc, 'theory_envelop')
        self.assertIsInstance(new_hist, ROOT.TH1)
        self.assertEqual(2, new_hist.GetNbinsX())
        self.assertEqual('Clone', new_hist.GetName())
        for i in range(new_hist.GetNbinsX() + 1):
            self.assertEqual(0., new_hist.GetBinContent(i))

    def test_get_variation_for_process_theory_envelop_process_check(self):
        dummy_process = Process('foo', dataset_info=None)
        pc = PlotConfig()
        fh = FileHandle(file_name='foo', process=dummy_process)
        analyser = sa.SystematicsAnalyser(file_handles=[fh], xs_handle='foo')
        analyser.systematic_hists = {'pdf_uncert_MUR0p5_MUF0p5_PDF261000': {pc: [hist]}}
        self.assertIsNone(analyser.get_variation_for_process(dummy_process, hist, pc,
                                                              'pdf_uncert_MUR0p5_MUF0p5_PDF261000'))
        self.assertIsNone(analyser.get_variation_for_process('Zjets', hist, pc,
                                                             'pdf_uncert_MUR0p5_MUF0p5_PDF261000'))

    def test_get_symmetrised_hists(self):
        def clone_and_rename(name):
            h = ROOT.TH1F(name, '', 2, 0., 1.)
            return h
        fh = FileHandle(file_name='foo', process=Process('foo_311011', dataset_info=None))
        analyser = sa.SystematicsAnalyser(file_handles=[fh], xs_handle='foo')
        hnom = deepcopy(hist)
        hnom.GetNbinsX = MagicMock(return_value=1)
        hnom.GetBinContent = MagicMock(return_value=2)
        hsys = deepcopy(hist)
        hsys.Clone = MagicMock(side_effect=clone_and_rename)
        hsys.GetNbinsX = MagicMock(return_value=1)
        hsys.GetBinContent = MagicMock(return_value=5)
        hist.SetBinContent = MagicMock(side_effect=lambda x, y: x)
        hsym = analyser.get_symmetrised_hists(hsys, hnom, 'foo')
        self.assertEqual('foo', hsym.GetName())
        self.assertEqual(-1., hsym.GetBinContent(1))

    @patch.object(BasePlotter, 'read_histograms', lambda *args, **kwargs: [(PlotConfig(), 'foo',
                                                                            ROOT.TH1F('h_syst', '', 10, 0., 10.))])
    @patch.object(BasePlotter, 'apply_lumi_weights', lambda *args, **kwargs: None)
    @patch.object(BasePlotter, 'merge_histograms', lambda _: None)
    def test_get_scale_uncertainty(self):
        fh = FileHandle(file_name='foo', process=Process('foo_311011', dataset_info=None))
        analyser = sa.SystematicsAnalyser(file_handles=[fh], xs_handle='foo', plot_configs=[PlotConfig(),
                                          PlotConfig(weight='weight')])
        self.assertIsNone(analyser.get_scale_uncertainties([fh], ['foo']))
        self.assertEqual(['foo'], list(analyser.systematic_hists.keys()))
        data = list(analyser.systematic_hists['foo'].items())[0][1]
        self.assertTrue('foo' in data)
        self.assertEqual('h_syst', data['foo'].GetName())

    @patch.object(sa.SystematicsAnalyser, 'load_dumped_hists', lambda *args, **kwargs: [(PlotConfig(), 'foo',
                                                                            ROOT.TH1F('h_syst', '', 10, 0., 10.))])
    @patch.object(BasePlotter, 'apply_lumi_weights', lambda *args, **kwargs: None)
    @patch.object(BasePlotter, 'merge_histograms', lambda _: None)
    def test_get_scale_uncertainty_load_dumped(self):
        fh = FileHandle(file_name='foo', process=Process('foo_311011', dataset_info=None))
        analyser = sa.SystematicsAnalyser(file_handles=[fh], xs_handle='foo', plot_configs=[PlotConfig(),
                                          PlotConfig(weight='weight')])
        self.assertIsNone(analyser.get_scale_uncertainties([fh], ['foo'], dumped_hist_path='foo'))
        self.assertEqual(['foo'], list(analyser.systematic_hists.keys()))
        data = list(analyser.systematic_hists['foo'].items())[0][1]
        self.assertTrue('foo' in data)
        self.assertEqual('h_syst', data['foo'].GetName())

    @patch.object(BasePlotter, 'read_histograms', lambda *args, **kwargs: [(PlotConfig(), 'foo',
                                                                            ROOT.TH1F('h_syst', '', 10, 0., 10.))])
    @patch.object(BasePlotter, 'apply_lumi_weights', lambda *args, **kwargs: None)
    @patch.object(BasePlotter, 'merge_histograms', lambda _: None)
    def test_get_scale_uncertainty_dump_hists(self):
        fh = FileHandle(file_name='foo', process=Process('foo_311011', dataset_info=None))
        analyser = sa.SystematicsAnalyser(file_handles=[fh], xs_handle='foo', plot_configs=[PlotConfig(),
                                          PlotConfig(weight='weight')], output_handle=MagicMock(), dump_hists=True)
        self.assertIsNone(analyser.get_scale_uncertainties([fh], ['foo']))
        self.assertEqual([], list(analyser.systematic_hists.keys()))

    @patch.object(sa.SystematicsAnalyser, 'load_dumped_hists', lambda *args, **kwargs: [(PlotConfig(),
                                                                                        Process('foo_311011',
                                                                                                dataset_info=None),
                                                                                        hist)])
    @patch.object(BasePlotter, 'apply_lumi_weights', lambda *args, **kwargs: None)
    @patch.object(BasePlotter, 'merge_histograms', lambda _: None)
    def test_get_scale_uncertainty_dump(self):
        fh = FileHandle(file_name='foo', process=Process('foo_311011', dataset_info=None))
        analyser = sa.SystematicsAnalyser(file_handles=[fh], xs_handle='foo', plot_configs=[PlotConfig()],
                                          dump_hist=True)
        self.assertIsNone(analyser.get_scale_uncertainties([fh], ['foo'], dumped_hist_path='foo'))
        self.assertTrue(isinstance(analyser.systematic_hists['foo'][PlotConfig()][Process('foo_311011',
                                                                                          dataset_info=None)], Mock))

    @patch.object(sa.SystematicsAnalyser, 'load_dumped_hists', lambda *args, **kwargs: [(PlotConfig(),
                                                                                      Process('foo_410472', dataset_info=None), hist)])
    @patch.object(BasePlotter, 'apply_lumi_weights', lambda *args, **kwargs: None)
    @patch.object(BasePlotter, 'merge_histograms', lambda _: None)
    def test_get_fixed_scale_uncertainties_load_from_dump(self):
        fh = FileHandle(file_name='foo', process=Process('foo_311011', dataset_info=None))
        analyser = sa.SystematicsAnalyser(file_handles=[fh], xs_handle='foo', plot_configs=[PlotConfig()],
                                          dump_hist=True)
        self.assertIsNone(analyser.get_fixed_scale_uncertainties([fh], {'foo': {410472: "(19.77/831.76, -29.20/831.76)"}},
                                                                 dumped_hist_path='foo'))

    @patch.object(sa.SystematicsAnalyser, 'read_histograms', lambda *args, **kwargs: [(PlotConfig(),
                                                                                      Process('foo_410472', dataset_info=None), hist)])
    def test_get_fixed_scale_uncertainties(self):
        fh = FileHandle(file_name='foo', process=Process('foo_311011', dataset_info=None))
        analyser = sa.SystematicsAnalyser(file_handles=[fh], xs_handle='foo', plot_configs=[PlotConfig()],
                                          dump_hist=True, output_handle=MagicMock())
        self.assertIsNone(analyser.get_fixed_scale_uncertainties([fh], {'foo':
                                                                            {410472: "(19.77/831.76, -29.20/831.76)"}}))


    def test_calculate_total_systematics(self):
        fh = FileHandle(file_name='foo', process=Process('foo_311011', dataset_info=None))
        hist = ROOT.TH1F("h", "", 100, -1., 1.)
        analyser = sa.SystematicsAnalyser(file_handles=[fh], xs_handle='foo', plot_configs=[PlotConfig()],
                                          dump_hist=True)
        analyser.systematic_variations = {'my_syst_down': {PlotConfig(): {'foo_311011': hist}}}
        self.assertIsNone(analyser.calculate_total_systematics())
        self.assertEqual('Total', list(analyser.total_systematics.keys())[0].name)

    @patch.object(sa.SystematicsAnalyser, 'get_variations_single_systematic', lambda *args: 0)
    def test_calculate_variations(self):
        dummy_process = Process('foo_311011', dataset_info=None)
        fh = FileHandle(file_name='foo', process=dummy_process)
        analyser = sa.SystematicsAnalyser(file_handles=[fh], xs_handle='foo', plot_configs=[PlotConfig()])
        hist = ROOT.TH1F("h", "", 100, -1., 1.)
        analyser.systematic_hists = {'my_syst_down': {dummy_process.process_name: {PlotConfig(): hist}}}
        self.assertIsNone(analyser.calculate_variations(analyser.systematic_hists))
        self.assertEqual({'my_syst_down': 0}, analyser.systematic_variations)

    @patch.object(BasePlotter, 'read_histograms', lambda *args, **kwargs: [(PlotConfig(), 'foo',
                                                                            ROOT.TH1F('h_syst', '', 10, 0., 10.))])
    @patch.object(BasePlotter, 'apply_lumi_weights', lambda *args, **kwargs: None)
    @patch.object(BasePlotter, 'merge_histograms', lambda _: None)
    def test_get_shape_uncertainty(self):
        fh = FileHandle(file_name='foo', process=Process('foo_311011', dataset_info=None))
        analyser = sa.SystematicsAnalyser(file_handles=[fh], xs_handle='foo', plot_configs=[PlotConfig()])
        self.assertIsNone(analyser.get_shape_uncertainty([fh], scale_syst))
        six.assertCountEqual(self, ['weight_foo_scale__1up', 'weight_foo_scale__1down'],
                             list(analyser.systematic_hists.keys()))
        data = list(analyser.systematic_hists['weight_foo_scale__1up'].items())[0][1]
        self.assertTrue('foo' in data)
        self.assertEqual('h_syst', data['foo'].GetName())

    @patch.object(BasePlotter, 'read_histograms', lambda *args, **kwargs: [(PlotConfig(), 'foo',
                                                                            ROOT.TH1F('h_syst', '', 10, 0., 10.))])
    @patch.object(BasePlotter, 'apply_lumi_weights', lambda *args, **kwargs: None)
    @patch.object(BasePlotter, 'merge_histograms', lambda _: None)
    def test_get_shape_uncertainty_dump_hisys(self):
        dummy_process = Process('foo_311011', dataset_info=None)
        fh = FileHandle(file_name='foo', process=dummy_process)
        analyser = sa.SystematicsAnalyser(file_handles=[fh], xs_handle='foo', plot_configs=[PlotConfig()],
                                          dump_hists=True, output_dir='test', output_handle=MagicMock())
        self.assertIsNone(analyser.get_shape_uncertainty([fh], scale_syst))
        self.assertEqual([], list(analyser.systematic_hists.keys()))

    @patch.object(sa.SystematicsAnalyser, 'load_dumped_hists', lambda *args, **kwargs: [(PlotConfig(), 'foo',
                                                                            ROOT.TH1F('h_syst', '', 10, 0., 10.))])
    @patch.object(BasePlotter, 'apply_lumi_weights', lambda *args, **kwargs: None)
    @patch.object(BasePlotter, 'merge_histograms', lambda _: None)
    def test_get_shape_uncertainty_load_dumped(self):
        fh = FileHandle(file_name='foo', process=Process('foo_311011', dataset_info=None))
        analyser = sa.SystematicsAnalyser(file_handles=[fh], xs_handle='foo', plot_configs=[PlotConfig()])
        self.assertIsNone(analyser.get_shape_uncertainty([fh], scale_syst, dumped_hist_path='foo'))
        six.assertCountEqual(self, ['weight_foo_scale__1up', 'weight_foo_scale__1down'],
                             list(analyser.systematic_hists.keys()))
        data = list(analyser.systematic_hists['weight_foo_scale__1up'].items())[0][1]
        self.assertTrue('foo' in data)
        self.assertEqual('h_syst', data['foo'].GetName())

    def test_get_variations_single_systematic(self):
        fh = FileHandle(file_name='foo', process=Process('foo_311011', dataset_info=None))
        pc = PlotConfig(name='foo')
        analyser = sa.SystematicsAnalyser(file_handles=[fh], xs_handle='foo', plot_configs=[pc])
        hnom = ROOT.TH1F('hnom', '', 10, 0., 10.)
        hsys = ROOT.TH1F('hsys', '', 10, 0., 10.)
        for i in range(1, 11):
            hnom.SetBinContent(i, i)
            hsys.SetBinContent(i, i*i)
        nom = {pc: {'foo': hnom}}
        analyser.systematic_hists['dummy_sys'] = {pc: {'foo': hsys}}
        variations = analyser.get_variations_single_systematic('dummy_sys', nom)
        self.assertEqual(1, len(variations))
        self.assertTrue(pc in variations)
        for i in range(1, 11):
            self.assertEqual((i*i - i)/i, variations[pc]['foo'].GetBinContent(i))

    def test_make_overview_plot(self):
        fh = FileHandle(file_name='foo', process=Process('foo_311011', dataset_info=None))
        pc = PlotConfig(name='foo')
        analyser = sa.SystematicsAnalyser(file_handles=[fh], xs_handle='foo', plot_configs=[pc])
        analyser.output_handle = MagicMock()
        hnom = ROOT.TH1F('hnom', '', 10, 0., 10.)
        hsys = ROOT.TH1F('hsys', '', 10, 0., 10.)
        for i in range(1, 11):
            hnom.SetBinContent(i, i)
            hsys.SetBinContent(i, i*i)
        analyser.systematic_variations['dummy_sys'] = {pc: {'foo': hsys}}
        self.assertIsNone(analyser.make_overview_plots(pc))

    def test_get_sm_total(self):
        fh = FileHandle(file_name='foo', process=Process('foo_311011', dataset_info=None))
        pc = PlotConfig(name='foo')
        analyser = sa.SystematicsAnalyser(file_handles=[fh], xs_handle='foo', plot_configs=[pc])
        hnom = ROOT.TH1F('hnom', '', 10, 0., 10.)
        hsys = ROOT.TH1F('hsys', '', 10, 0., 10.)
        for i in range(1, 10):
            hnom.SetBinContent(i, i)
            hsys.SetBinContent(i, 1. / (i*i))
        hnom.SetBinContent(10, 10.)
        hsys.SetBinContent(10, 3.)

        nom = {'foo': hnom}
        analyser.total_systematics[SystematicsCategory()] = {'up': {pc: {'foo': hsys}}}
        data = analyser.get_relative_unc_on_SM_total(pc, nom)
        self.assertEqual(3, len(data))
        self.assertEqual([], data[1])
        self.assertIsInstance(data[0][0], ROOT.TH1)
        self.assertEqual([ROOT.kBlue, ROOT.kBlue], data[2])
        for i in range(1, 10):
            self.assertAlmostEqual((1. + 1. / (i*i)) * i, data[0][0].GetBinContent(i), delta=1.e-4)
        self.assertAlmostEqual(10., data[0][0].GetBinContent(10), delta=1.e-4)

    def test_get_sm_total_nan_value(self):
        fh = FileHandle(file_name='foo', process=Process('foo_311011', dataset_info=None))
        pc = PlotConfig(name='foo')
        analyser = sa.SystematicsAnalyser(file_handles=[fh], xs_handle='foo', plot_configs=[pc])
        hnom = ROOT.TH1F('hnom', '', 10, 0., 10.)
        hsys = ROOT.TH1F('hsys', '', 10, 0., 10.)
        hnom.SetBinContent(1, np.nan)
        hsys.SetBinContent(1, np.nan)
        nom = {'foo': hnom}
        analyser.total_systematics[SystematicsCategory()] = {'up': {pc: {'foo': hsys}}}
        self.assertRaises(SystemExit, analyser.get_relative_unc_on_SM_total, pc, nom)

class TestTheoryUncertProvider(unittest.TestCase):
    def setUp(self):
        self.analyser = None

    def test_sherpa_uncerts(self):
        all_uncerts = ['weight_pdf_uncert_MUR0p5_MUF0p5_PDF261000',
                       'weight_pdf_uncert_MUR0p5_MUF1_PDF261000',
                       'weight_pdf_uncert_MUR1_MUF0p5_PDF261000',
                       'weight_pdf_uncert_MUR1_MUF2_PDF261000',
                       'weight_pdf_uncert_MUR2_MUF1_PDF261000',
                       'weight_pdf_uncert_MUR2_MUF2_PDF261000',
                       'weight_pdf_uncert_MUR1_MUF1_PDF25300',
                       'weight_pdf_uncert_MUR1_MUF1_PDF13000']
        self.assertEqual(all_uncerts, sa.TheoryUncertaintyProvider.get_sherpa_uncerts())

    def test_top_pdf_uncerts(self):
        all_uncerts = ['weight_pdf_uncert_PDF_set_{:d}'.format(i) for i in range(90901, 90931)]
        self.assertEqual(all_uncerts, sa.TheoryUncertaintyProvider.get_top_pdf_uncerts())

    def test_top_Var3c_uncerts(self):
        all_uncerts = ['weight_pdf_uncert_Var3cUp', 'weight_pdf_uncert_Var3cDown']
        self.assertEqual(all_uncerts, sa.TheoryUncertaintyProvider.get_top_Var3c_uncerts())

    def test_get_top_renorm_scale_uncerts(self):
        all_uncerts = ['weight_pdf_uncert_muR_0p5_muF_0p5',
                       'weight_pdf_uncert_muR_0p5_muF_1p0',
                       'weight_pdf_uncert_muR_0p5_muF_2p0',
                       'weight_pdf_uncert_muR_1p0_muF_0p5',
                       'weight_pdf_uncert_muR_1p0_muF_2p0',
                       'weight_pdf_uncert_muR_2p0_muF_0p5',
                       'weight_pdf_uncert_muR_2p0_muF_1p0',
                       'weight_pdf_uncert_muR_2p0_muF_2p0']
        self.assertEqual(all_uncerts, sa.TheoryUncertaintyProvider.get_top_renorm_scale_uncerts())

    def test_get_top_isr_fsr_uncerts(self):
        all_uncerts = ['weight_pdf_uncert_isr_muRfac=0p5_fsr_muRfac=0p5',
                       'weight_pdf_uncert_isr_muRfac=0p5_fsr_muRfac=1p0',
                       'weight_pdf_uncert_isr_muRfac=0p5_fsr_muRfac=2p0',
                       'weight_pdf_uncert_isr_muRfac=1p0_fsr_muRfac=0p5',
                       'weight_pdf_uncert_isr_muRfac=1p0_fsr_muRfac=2p0',
                       'weight_pdf_uncert_isr_muRfac=2p0_fsr_muRfac=1p0',
                       'weight_pdf_uncert_isr_muRfac=2p0_fsr_muRfac=2p0']
        self.assertEqual(all_uncerts, sa.TheoryUncertaintyProvider.get_top_isr_fsr_uncerts())

    def test_get_top_scale_uncerts(self):
        all_uncerts = []
        all_uncerts += sa.TheoryUncertaintyProvider.get_top_Var3c_uncerts()
        all_uncerts += sa.TheoryUncertaintyProvider.get_top_renorm_scale_uncerts()
        all_uncerts += sa.TheoryUncertaintyProvider.get_top_isr_fsr_uncerts()
        all_uncerts += sa.TheoryUncertaintyProvider.get_top_pdf_uncerts()
        self.assertListEqual(all_uncerts, sa.TheoryUncertaintyProvider.get_top_scale_uncerts())

    def test_top_uncertainy_names_invalid(self):
        self.assertIsNone(sa.TheoryUncertaintyProvider().get_top_uncert_names())

    @patch('PyAnalysisTools.base.YAMLHandle.YAMLLoader.read_yaml', mock_read_yaml)
    def test_top_uncertainy_names(self):
        self.assertEqual(['foo'], sa.TheoryUncertaintyProvider(fixed_top_unc_file='foo').get_top_uncert_names())

    @patch.object(FileHandle, 'get_object_by_name', MagicMock)
    def test_is_affected(self):
        self.assertTrue(sa.TheoryUncertaintyProvider().get_sherpa_uncerts())

    @patch.object(sa.SystematicsAnalyser, 'get_fixed_scale_uncertainties', lambda *args: None)
    @patch.object(MLHelper.Root2NumpyConverter, 'convert_to_array',
                  lambda *args: {name: [1.] for name in ['weight'] + sa.TheoryUncertaintyProvider.get_top_pdf_uncerts()})
    def test_get_top_uncert(self):
        analyser = sa.SystematicsAnalyser(xs_handle='foo', plot_configs=MagicMock(), tree_name='tree_name')
        process = MagicMock()
        process.dsid = MagicMock(return_value=311011)
        fh = MagicMock()
        fh.process = process
        analyser.file_handles = [fh]
        provider = sa.TheoryUncertaintyProvider()
        provider.top_unc = {'foo': {311011: 'foo'}}
        self.assertIsNone(provider.get_top_theory_unc(analyser))

    @patch.object(sa.SystematicsAnalyser, 'get_fixed_scale_uncertainties', lambda *args: None)
    @patch.object(MLHelper.Root2NumpyConverter, 'convert_to_array',
                  lambda *args: {name: [1.] for name in ['weight'] + sa.TheoryUncertaintyProvider.get_top_pdf_uncerts()})
    def test_get_top_uncert_load_dumped(self):
        analyser = sa.SystematicsAnalyser(xs_handle='foo', plot_configs=MagicMock(), tree_name='tree_name')
        process = MagicMock()
        process.dsid = MagicMock(return_value=311011)
        fh = MagicMock()
        fh.process = process
        analyser.file_handles = [fh]
        provider = sa.TheoryUncertaintyProvider()
        provider.top_unc = {'foo': {311011: 'foo'}}
        self.assertIsNone(provider.get_top_theory_unc(analyser, dump_hist_path='foo'))

    def test_calculate_envelop_count(self):
        sherpa_uncerts = sa.TheoryUncertaintyProvider.get_sherpa_uncerts()
        yields = {syst: Yield(weights=[i, i+1.]) for i, syst in enumerate(sherpa_uncerts)}
        yields['foo'] = Yield(weights=[13., 14.])
        self.assertIsNone(sa.TheoryUncertaintyProvider().calculate_envelop_count(yields))
        self.assertTrue('theory_envelop' in yields)
        i = len(sherpa_uncerts) - 1
        np.testing.assert_equal([i, i+1.], yields['theory_envelop'].weights)

    def test_calculate_envelop_count_missing_uncerts(self):
        sherpa_uncerts = sa.TheoryUncertaintyProvider.get_sherpa_uncerts()
        yields = {syst: Yield(weights=[i, i+1.]) for i, syst in enumerate(sherpa_uncerts[:2])}
        self.assertIsNone(sa.TheoryUncertaintyProvider().calculate_envelop_count(yields))
        self.assertFalse('theory_envelop' in yields)

    def test_get_envelop(self):
        analyser = Mock()
        analyser.file_handles = MagicMock(return_value=[])
        self.assertIsNone(sa.TheoryUncertaintyProvider().get_envelop(analyser))

    def test_get_uncertainties(self):
        analyser = Mock()
        analyser.file_handles = MagicMock(return_value=[])
        self.assertIsNone(sa.TheoryUncertaintyProvider().fetch_uncertainties(analyser))

    def test_get_uncertainties_load_dumped(self):
        analyser = Mock()
        analyser.file_handles = MagicMock(return_value=[])
        self.assertIsNone(sa.TheoryUncertaintyProvider().fetch_uncertainties(analyser, dump_hist_path='foo'))
