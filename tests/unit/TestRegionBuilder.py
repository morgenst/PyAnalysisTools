import unittest
import os

from PyAnalysisTools.AnalysisTools.RegionBuilder import Region, RegionBuilder, Cut
from PyAnalysisTools.PlottingUtils.PlotConfig import parse_and_build_plot_config
from PyAnalysisTools.base.YAMLHandle import YAMLLoader as yl

cwd = os.path.dirname(__file__)


class TestCut(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.base_selection = "muon_pt > 20 GeV. && electron_pt>10 GeV"
        self.cut = Cut(self.base_selection)

    def test_basic_cuts(self):
        self.assertFalse(self.cut.is_mc)
        self.assertFalse(self.cut.is_data)
        self.assertEqual(self.base_selection, self.cut.name)
        self.assertEqual(self.base_selection, self.cut.selection)

    def test_equality(self):
        c2 = Cut(self.base_selection)
        self.assertEqual(self.cut, c2)

    def test_inequality(self):
        c2 = Cut("muon_pt > 10 GeV. && electron_pt>20 GeV")
        self.assertNotEqual(self.cut, c2)

    def test_inequality_type(self):
        self.assertNotEqual(self.cut, 1)

    def test_print(self):
        self.assertEqual("Cut object named {:s} and selection {:s}".format(self.base_selection, self.base_selection),
                         self.cut.__str__())

    def test_print_list(self):
        self.assertEqual("[Cut object named {:s} and selection {:s}\n]".format(self.base_selection,
                                                                               self.base_selection),
                         str([self.cut]))

    def test_named_cut(self):
        c = Cut(self.base_selection + ":::myCut")
        self.assertFalse(c.is_mc)
        self.assertFalse(c.is_data)
        self.assertEqual("myCut", c.name)
        self.assertEqual(self.base_selection, c.selection)

    def test_mc_cut(self):
        c = Cut("MC:" + self.base_selection)
        self.assertTrue(c.is_mc)
        self.assertFalse(c.is_data)
        self.assertEqual(self.base_selection, c.selection)

    def test_data_cut(self):
        c = Cut("DATA:" + self.base_selection)
        self.assertFalse(c.is_mc)
        self.assertTrue(c.is_data)
        self.assertEqual(self.base_selection, c.selection)

    def test_process_specific_cut(self):
        c = Cut("TYPE_MY_PROCESS:" + self.base_selection)
        self.assertFalse(c.is_mc)
        self.assertFalse(c.is_data)
        self.assertEqual('my_process', c.process_type)
        self.assertEqual(self.base_selection, c.selection)


class TestRegion(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.region = Region(name='testRegion', event_cuts=["jet_n > 1", "Sum$(jet_has_btag) == 2"], n_lep=2,
                             n_muon=2)

    def test_defaults(self):
        region = Region(name='test')
        self.assertFalse(region.disable_leptons)
        self.assertFalse(region.disable_taus)
        self.assertFalse(region.disable_electrons)
        self.assertFalse(region.disable_muons)
        self.assertFalse(region.split_mc_data)
        self.assertFalse(region.norm_region)
        self.assertFalse(region.val_region)
        self.assertIsNone(region.is_on_z)
        self.assertIsNone(region.label_position)
        self.assertIsNone(region.good_muon)
        self.assertIsNone(region.fake_muon)
        self.assertIsNone(region.inverted_muon)
        self.assertIsNone(region.good_electron)
        self.assertIsNone(region.inverted_electron)
        self.assertIsNone(region.event_cuts)
        self.assertIsNone(region.common_selection)
        self.assertIsNone(region.weight)
        self.assertIsNone(region.binning)
        self.assertEqual('==', region.operator)
        self.assertEqual('==', region.muon_operator)
        self.assertEqual('test', region.name)
        self.assertEqual('', region.label)
        self.assertEqual({}, region.norm_backgrounds)

    def test_operators(self):
        region = Region(name='test', operator='leq', muon_operator='leq')
        self.assertEqual('<=', region.operator)
        self.assertEqual('<=', region.muon_operator)
        region2 = Region(name='test', operator='geq', muon_operator='geq')
        self.assertEqual('>=', region2.operator)
        self.assertEqual('>=', region2.muon_operator)
        self.assertRaises(ValueError, Region, name='test', operator='foo', muon_operator='geq')
        self.assertRaises(ValueError, Region, name='test', operator='geq', muon_operator='>=')

    def test_equality(self):
        reg = Region(name='testRegion', event_cuts=["jet_n > 1", "Sum$(jet_has_btag) == 2"], n_lep=2,
                     n_muon=2)
        self.assertEqual(reg, self.region)

    def test_inequality_name(self):
        reg = Region(name='fooRegion', event_cuts=["jet_n > 1", "Sum$(jet_has_btag) == 2"], n_lep=2,
                     n_muon=2)
        self.assertNotEqual(reg, self.region)

    def test_inequality_evt_cuts(self):
        reg = Region(name='testRegion', event_cuts=["jet_n > 2", "Sum$(jet_has_btag) == 2"], n_lep=2,
                     n_muon=2)
        self.assertNotEqual(reg, self.region)

    def test_inequality_lep_cut(self):
        reg = Region(name='testRegion', event_cuts=["jet_n > 1", "Sum$(jet_has_btag) == 2"], n_lep=2,
                     n_muon=1)
        self.assertNotEqual(reg, self.region)

    def test_inequality_type(self):
        self.assertNotEqual('foo', self.region)

    def test_build_cuts(self):
        self.region.build_cuts()
        self.assertEqual(self.region.cut_list, [Cut("jet_n > 1"), Cut('Sum$(jet_has_btag) == 2')])

    def test_hash(self):
        self.assertEqual(hash('testRegion'), self.region.__hash__())

    def test_str(self):
        self.assertEqual('Region: testRegion \n'
                         'n_lep=2 n_electron=-1 n_muon=2 n_tau=0 event_cuts=[\'jet_n > 1\', '
                         '\'Sum$(jet_has_btag) == 2\'] norm_region=False val_region=False norm_backgrounds={} '
                         'disable_leptons=False disable_taus=False disable_electrons=False disable_muons=False '
                         'is_on_z=None operator=== muon_operator=== electron_operator=== label=#mu^{#pm}#mu^{#pm} '
                         'label_position=None good_muon=None fake_muon=None inverted_muon=None good_electron=None '
                         'inverted_electron=None split_mc_data=False common_selection=None weight=None binning=None '
                         'cut_list=[Cut object named jet_n > 1 and selection jet_n > 1\n, Cut object named '
                         'Sum$(jet_has_btag) == 2 and selection Sum$(jet_has_btag) == 2\n] ', str(self.region))

    def test_str_list(self):
        self.assertEqual('[Region: testRegion \n'
                         'n_lep=2 n_electron=-1 n_muon=2 n_tau=0 event_cuts=[\'jet_n > 1\','
                         ' \'Sum$(jet_has_btag) == 2\'] norm_region=False val_region=False norm_backgrounds={} '
                         'disable_leptons=False disable_taus=False disable_electrons=False disable_muons=False '
                         'is_on_z=None operator=== muon_operator=== electron_operator=== label=#mu^{#pm}#mu^{#pm} '
                         'label_position=None good_muon=None fake_muon=None inverted_muon=None good_electron=None '
                         'inverted_electron=None split_mc_data=False common_selection=None weight=None binning=None '
                         'cut_list=[Cut object named jet_n > 1 and selection jet_n > 1\n, Cut object named '
                         'Sum$(jet_has_btag) == 2 and selection Sum$(jet_has_btag) == 2\n] \n]', str([self.region]))

    def test_cut_validation_data_mcinput(self):
        reg = Region(name='testRegion', event_cuts=["DATA:jet_n > 1", "Sum$(jet_has_btag) == 2"], n_lep=2,
                     n_muon=1)
        self.assertEqual(['1', 'Sum$(jet_has_btag) == 2'], [c.selection for c in reg.get_cut_list()])

    def test_cut_validation_data_datainput(self):
        reg = Region(name='testRegion', event_cuts=["DATA:jet_n > 1", "Sum$(jet_has_btag) == 2"], n_lep=2,
                     n_muon=1)
        self.assertEqual([Cut("DATA:jet_n > 1"), Cut('Sum$(jet_has_btag) == 2')], reg.get_cut_list(True))

    def test_cut_validation_mc_mcinput(self):
        reg = Region(name='testRegion', event_cuts=["MC:jet_n > 1", "Sum$(jet_has_btag) == 2"], n_lep=2,
                     n_muon=1)
        self.assertEqual([Cut("MC:jet_n > 1"), Cut('Sum$(jet_has_btag) == 2')], reg.get_cut_list())

    def test_cut_validation_mc_datainput(self):
        reg = Region(name='testRegion', event_cuts=["MC:jet_n > 1", "Sum$(jet_has_btag) == 2"], n_lep=2,
                     n_muon=1)
        self.assertEqual(['1', 'Sum$(jet_has_btag) == 2'], [c.selection for c in reg.get_cut_list(True)])

    def test_build_particle_cut_empty(self):
        self.assertEqual(Cut('muon_n > 2'), self.region.build_particle_cut(None, None, '>', 'muon_n', 2))

    def test_muon_cuts(self):
        reg = Region(name='testRegion2', good_muon=["muon_pt > 60."], n_lep=2, n_muon=2)
        self.assertEqual([Cut('Sum$(muon_pt > 60.) == muon_n && muon_n == 2')], reg.cut_list)

    def test_electron_cuts(self):
        reg = Region(name='testRegion2', good_electron=["electron_pt > 60."], n_lep=2, n_electron=2)
        self.assertEqual([Cut('Sum$(electron_pt > 60.) == electron_n && electron_n == 2')], reg.cut_list)

    def test_build_label(self):
        self.assertEqual('#mu^{#pm}#mu^{#pm}', self.region.label)

    def test_build_Zlabel(self):
        reg = Region(name='testRegion2', good_electron=["electron_pt > 60."], n_lep=2, n_electron=2, is_on_z=True)
        self.assertEqual('e^{#pm}e^{#pm} on-Z', reg.label)


class TestRegionBuilder(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        print('TRY: ', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures/module_config.yml'))
        cfg = yl.read_yaml(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures/module_config.yml'))
        self.reg_builder = RegionBuilder(**cfg['RegionBuilder'])

    def test_defaults(self):
        reg_builder = RegionBuilder()
        self.assertEqual(0, len(reg_builder.regions))

    def test_n_regions(self):
        self.assertEqual(2, len(self.reg_builder.regions))

    def test_plot_cfg_modification_cut(self):
        plot_configs, _ = parse_and_build_plot_config(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                   'fixtures/plot_config.yml'))
        plot_configs = self.reg_builder.execute(plot_configs)
        self.assertEqual(4, len(plot_configs))
        index_no_cut = plot_configs.index([pc for pc in plot_configs if 'no_cut' in pc.name][0])
        index_cut = plot_configs.index([pc for pc in plot_configs if 'no_cut' not in pc.name][0])
        self.assertEqual(['1 ::: dummy_cut']
                         + [c.selection for c in self.reg_builder.regions[0].get_cut_list()],
                         plot_configs[index_cut].cuts)
        self.assertEqual(['1 ::: dummy_cut']
                         + [c.selection for c in self.reg_builder.regions[1].get_cut_list()],
                         plot_configs[index_cut+2].cuts)
        self.assertEqual([c.selection for c in self.reg_builder.regions[0].get_cut_list()],
                         plot_configs[index_no_cut].cuts)
        self.assertEqual([c.selection for c in self.reg_builder.regions[1].get_cut_list()],
                         plot_configs[index_no_cut+2].cuts)

    def test_plot_cfg_modification_weight(self):
        plot_configs, _ = parse_and_build_plot_config(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                   'fixtures/plot_config.yml'))
        self.reg_builder.regions[0].weight = 'dummy_weight'
        self.reg_builder.regions[1].weight = None
        plot_configs = self.reg_builder.execute(plot_configs)
        self.assertEqual(4, len(plot_configs))
        index_no_cut = plot_configs.index([pc for pc in plot_configs if 'no_cut' in pc.name][0])
        index_cut = plot_configs.index([pc for pc in plot_configs if 'no_cut' not in pc.name][0])
        self.assertEqual('weight * dummy_weight', plot_configs[index_cut].weight)
        self.assertEqual('weight', plot_configs[index_cut+2].weight)
        self.assertEqual('dummy_weight', plot_configs[index_no_cut].weight)
        self.assertIsNone(plot_configs[index_no_cut+2].weight)
