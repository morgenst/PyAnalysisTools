import math
import os
import unittest
from collections import OrderedDict

import ROOT
from PyAnalysisTools.PlottingUtils.PlotConfig import PlotConfig
import PyAnalysisTools.PlottingUtils.PlotConfig as pcm
from PyAnalysisTools.base import InvalidInputError
from PyAnalysisTools.base.ProcessConfig import ProcessConfig
from .Mocks import hist
from mock import patch


class TestPlotConfig(unittest.TestCase):
    def setUp(self):
        pass

    def test_math_range(self):
        pc = PlotConfig(ymin='math.pi')
        self.assertEqual(math.pi, pc.ymin)

    def test_significance_and_ratio(self):
        pc = PlotConfig(calcsig=True, ratio=True)
        self.assertFalse(pc.calcsig)
        self.assertTrue(pc.ratio)

    def test_set_to_value(self):
        pc = PlotConfig(calcsig=True, ratio=True)
        self.assertFalse(pc.is_set_to_value('foo', 'bar'))
        self.assertFalse(pc.is_set_to_value('calcsig', True))
        self.assertTrue(pc.is_set_to_value('calcsig', False))
        self.assertFalse(pc.is_set_to_value('ratio', False))
        self.assertTrue(pc.is_set_to_value('ratio', True))

    def test_set_additional_config(self):
        pc = PlotConfig(ymin='math.pi')
        pc.set_additional_config('foo', bar='foobar')
        pc_res = PlotConfig(name='ratio', dist='ratio', bar='foobar')
        self.assertEqual(pc_res, pc.foo)

    def test_get_overwritable_options(self):
        pc = PlotConfig()
        opts = ["outline", "make_plot_book", "no_data", "draw", "ordering", "signal_scale", "lumi", "normalise",
                "merge_mc_campaigns", "signal_extraction", "ratio", "cuts", "enable_legend", 'total_lumi']
        self.assertTrue(opts, pc.get_overwritable_options())

    @patch('PyAnalysisTools.PlottingUtils.PlotConfig.input', create=True)
    def test_merging(self, mock_input):
        mock_input.side_effect = ['1', '2', '']
        pc1 = PlotConfig(ymin='math.pi')
        pc2 = PlotConfig(ymax='math.pi')
        pc1.merge_configs(pc2)
        self.assertAlmostEqual(math.pi, pc1.ymin, delta=1.e-5)
        self.assertAlmostEqual(math.pi, pc1.ymax, delta=1.e-5)

    def test_get_default(self):
        pc = pcm.get_default_plot_config(hist)
        pc_def = PlotConfig()
        self.assertEqual('foo', pc.name)
        for i in list(pc_def.__dict__.keys()):
            if i == 'name':
                continue
            self.assertEqual(getattr(pc, i), getattr(pc_def, i))

    def test_default_color_scheme(self):
        default = [ROOT.kGray + 3, ROOT.kRed + 2, ROOT.kAzure + 4,
                   ROOT.kSpring - 6,
                   ROOT.kOrange - 3,
                   ROOT.kCyan - 3,
                   ROOT.kPink - 2,
                   ROOT.kSpring - 9,
                   ROOT.kMagenta - 5,
                   ROOT.kOrange,
                   ROOT.kCyan + 3,
                   ROOT.kPink + 4,
                   ROOT.kGray + 3,
                   ROOT.kRed + 2,
                   ROOT.kAzure + 4,
                   ROOT.kSpring - 6,
                   ROOT.kOrange - 3,
                   ROOT.kCyan - 3,
                   ROOT.kPink - 2,
                   ROOT.kSpring - 9,
                   ROOT.kMagenta - 5,
                   ROOT.kOrange,
                   ROOT.kCyan + 3,
                   ROOT.kPink + 4]
        self.assertEqual(default, pcm.get_default_color_scheme())

    def test_get_lumi_float(self):
        pc = PlotConfig(lumi=100.)
        self.assertEqual(100., pc.get_lumi())

    def test_get_lumi_dict_all_processed(self):
        pc = PlotConfig(lumi={'MC16a': 100., 'MC16d': 120.})
        pc.used_mc_campaigns = ['MC16a', 'MC16d']
        self.assertEqual(220., pc.get_lumi())

    def test_get_lumi_dict_part_processed(self):
        pc = PlotConfig(lumi={'MC16a': 100., 'MC16d': 120.})
        pc.used_mc_campaigns = ['MC16a']
        self.assertEqual(100., pc.get_lumi())

    def test_get_lumi_ordereddict_all_processed(self):
        pc = PlotConfig(lumi=OrderedDict([('MC16a', 100.), ('MC16d', 120.)]))
        pc.used_mc_campaigns = ['MC16a', 'MC16d']
        self.assertEqual(220., pc.get_lumi())

    def test_mc_campaign_parsing(self):
        self.assertEqual('mc16a', pcm.parse_mc_campaign('LQ_MC16a'))
        self.assertEqual('mc16c', pcm.parse_mc_campaign('LQ_MC16c'))
        self.assertEqual('mc16d', pcm.parse_mc_campaign('LQ_MC16d'))
        self.assertEqual('mc16e', pcm.parse_mc_campaign('LQ_MC16e'))
        self.assertIsNone(pcm.parse_mc_campaign('LQ'))

    def test_parse_draw_option_default(self):
        pc = PlotConfig()
        self.assertEqual('hist', pcm._parse_draw_option(pc))

    def test_parse_draw_option(self):
        pc = PlotConfig(draw='p')
        self.assertEqual('p', pcm._parse_draw_option(pc))

    def test_parse_draw_option_process(self):
        pc = PlotConfig(draw='p')
        prcfg = ProcessConfig(draw='l', type='data')
        self.assertEqual('l', pcm._parse_draw_option(pc, prcfg))

    def test_draw_opt_as_root_str_default(self):
        pc = PlotConfig()
        self.assertEqual('HIST', pcm.get_draw_option_as_root_str(pc))

    def test_draw_opt_as_root_str_marker(self):
        pc = PlotConfig(draw='Marker')
        self.assertEqual('p', pcm.get_draw_option_as_root_str(pc))

    def test_draw_opt_as_root_str_line(self):
        pc = PlotConfig(draw='Line')
        self.assertEqual('l', pcm.get_draw_option_as_root_str(pc))

    def test_draw_opt_as_root_str_marker_error(self):
        pc = PlotConfig(draw='MarkerError')
        self.assertEqual('E', pcm.get_draw_option_as_root_str(pc))

    def test_draw_opt_as_root_str_draw_option(self):
        pc = PlotConfig(draw_option='lpe')
        self.assertEqual('lpe', pcm.get_draw_option_as_root_str(pc))

    def test_transform_color_str(self):
        self.assertEqual(634, pcm.transform_color('kRed + 2'))

    def test_transform_color_unicode(self):
        self.assertEqual(634, pcm.transform_color(u'kRed + 2'))

    def test_transform_color_str_2(self):
        self.assertEqual(630, pcm.transform_color('kRed - 2'))

    def test_transform_color_unicode_2(self):
        self.assertEqual(630, pcm.transform_color(u'kRed - 2'))

    def test_transform_color_str_3(self):
        self.assertEqual(632, pcm.transform_color('kRed'))

    def test_transform_color_unicode_3(self):
        self.assertEqual(632, pcm.transform_color(u'kRed'))

    def test_transform_color_root_str(self):
        self.assertEqual(634, pcm.transform_color('ROOT.kRed + 2'))

    def test_transform_color_root_unicode(self):
        self.assertEqual(634, pcm.transform_color(u'ROOT.kRed + 2'))

    def test_transform_color_list(self):
        self.assertEqual(634, pcm.transform_color(['ROOT.kRed + 2'], 0))

    def test_transform_color_list_unicode(self):
        self.assertEqual(634, pcm.transform_color([u'ROOT.kRed + 2'], 0))

    def test_transform_color_list_out_of_range(self):
        self.assertEqual(1, pcm.transform_color(['ROOT.kRed + 2'], 1))

    def test_get_hist_def_1d(self):
        pc = PlotConfig(name='foo', dist='foo / 1000.', bins=10, xmin=0., xmax=10., draw='Line')
        hist = pcm.get_histogram_definition(pc)
        self.assertEqual('foo%%Nominal_%%', hist.GetName())
        self.assertEqual(10, hist.GetNbinsX())

    def test_get_hist_def_1d_logx(self):
        pc = PlotConfig(name='foo', dist='foo / 1000.', bins=2, xmin=1., xmax=100., logx=True, draw='Line')
        hist = pcm.get_histogram_definition(pc)
        self.assertEqual('foo%%Nominal_%%', hist.GetName())
        self.assertEqual(2, hist.GetNbinsX())
        self.assertEqual(1., hist.GetXaxis().GetBinLowEdge(1))
        self.assertEqual(10., hist.GetXaxis().GetBinLowEdge(2))
        self.assertEqual(100., hist.GetXaxis().GetBinLowEdge(3))

    def test_get_hist_def_2d(self):
        pc = PlotConfig(name='foo', dist='foo / 1000. : bar', xbins=10, xmin=0., xmax=10., ybins=15, ymin=20.,
                        ymax=100., draw='Line')
        hist = pcm.get_histogram_definition(pc)
        self.assertEqual('foo%%Nominal_%%', hist.GetName())
        self.assertIsInstance(hist, ROOT.TH2F)
        self.assertEqual(10, hist.GetNbinsX())
        self.assertEqual(15, hist.GetYaxis().GetNbins())

    def test_get_hist_def_invalid(self):
        pc = PlotConfig(name='foo', nbins=10, xmin=0., xmax=10., draw='Line')
        self.assertRaises(InvalidInputError, pcm.get_histogram_definition, pc)

    def test_style_setter(self):
        pc = PlotConfig(name='foo', draw='Line', style=1001)
        self.assertEqual((['Line'], 1001, None), pcm.get_style_setters_and_values(pc))

    def test_plot_config_build(self):
        plot_cfgs, common_cfg = pcm.parse_and_build_plot_config(os.path.join(os.path.dirname(__file__),
                                                                             'fixtures/plot_config.yml'))
        self.assertEqual(2, len(plot_cfgs))
        self.assertEqual('lq_mass_max', plot_cfgs[0].name)
        self.assertEqual('lq_mass_max_no_cut', plot_cfgs[1].name)
        self.assertFalse(common_cfg.ratio)

    def test_plot_config_build_exception(self):
        try:
            self.assertRaises(FileNotFoundError, pcm.parse_and_build_plot_config, 'rndm_file')
        except NameError:
            self.assertRaises(IOError, pcm.parse_and_build_plot_config, 'rndm_file')

    def test_common_cfg_propagation(self):
        plot_cfgs, common_cfg = pcm.parse_and_build_plot_config(os.path.join(os.path.dirname(__file__),
                                                                             'fixtures/plot_config.yml'))
        self.assertEqual(2, len(plot_cfgs))
        self.assertEqual('lq_mass_max', plot_cfgs[0].name)
        self.assertEqual('hist', plot_cfgs[0].outline)
        self.assertEqual('lq_mass_max_no_cut', plot_cfgs[1].name)
        self.assertFalse(common_cfg.ratio)
        pcm.propagate_common_config(common_cfg, plot_cfgs)
        self.assertEqual('stack', plot_cfgs[0].outline)

    def test_defaults(self):
        pc = PlotConfig()
        self.assertIsNone(pc.process_weight)
        self.assertIsNone(pc.cuts)
        self.assertIsNone(pc.add_text)
        self.assertIsNone(pc.weight)
        self.assertIsNone(pc.dist)
        self.assertIsNone(pc.style)
        self.assertIsNone(pc.rebin)
        self.assertIsNone(pc.ratio)
        self.assertIsNone(pc.blind)
        self.assertIsNone(pc.ordering)
        self.assertIsNone(pc.ymin)
        self.assertIsNone(pc.xmin)
        self.assertIsNone(pc.draw_option)
        self.assertIsNone(pc.ymax)
        self.assertIsNone(pc.normalise_range)
        self.assertIsNone(pc.ratio_config)
        self.assertIsNone(pc.signal_scale)
        self.assertIsNone(pc.ytitle)
        self.assertIsNone(pc.ztitle)
        self.assertIsNone(pc.total_lumi)
        self.assertIsNone(pc.xtitle_offset)
        self.assertIsNone(pc.ytitle_offset)
        self.assertIsNone(pc.ztitle_offset)
        self.assertIsNone(pc.xtitle_size)
        self.assertIsNone(pc.ytitle_size)
        self.assertIsNone(pc.axis_labels)
        self.assertIsNone(pc.ztitle_size)
        self.assertIsNone(pc.labels)
        self.assertIsNone(pc.color)
        self.assertIsNone(pc.lumi_text)
        self.assertIsNone(pc.ybins)
        self.assertFalse(pc.stat_box)
        self.assertFalse(pc.normalise)
        self.assertFalse(pc.no_data)
        self.assertFalse(pc.ignore_style)
        self.assertFalse(pc.calcsig)
        self.assertFalse(pc.ignore_rebin)
        self.assertFalse(pc.enable_legend)
        self.assertFalse(pc.make_plot_book)
        self.assertFalse(pc.is_multidimensional)
        self.assertFalse(pc.is_common)
        self.assertFalse(pc.grid)
        self.assertFalse(pc.logy)
        self.assertFalse(pc.logx)
        self.assertFalse(pc.logz)
        self.assertFalse(pc.decor_text)
        self.assertFalse(pc.disable_bin_merge)
        self.assertFalse(pc.enable_range_arrows)
        self.assertFalse(pc.disable_bin_width_division)
        self.assertFalse(pc.ignore_process_labels)
        self.assertTrue(pc.merge)
        self.assertTrue(pc.signal_extraction)
        self.assertTrue(pc.merge_mc_campaigns)
        self.assertEqual('default_plot_config', pc.name)
        self.assertEqual('hist', pc.draw)
        self.assertEqual('hist', pc.outline)
        self.assertEqual({}, pc.legend_options)
        self.assertEqual([], pc.used_mc_campaigns)
        self.assertEqual(1, pc.lumi_precision)
        self.assertEqual(1.2, pc.yscale)
        self.assertEqual(100.0, pc.yscale_log)
        self.assertEqual(0.1, pc.ymin_log)
        self.assertEqual(1.0, pc.norm_scale)
        self.assertEqual(1.0, pc.lumi)
        self.assertEqual(0.065, pc.watermark_size)
        self.assertEqual(0.12, pc.watermark_offset)
        self.assertEqual(0.12, pc.watermark_offset_ratio)
        self.assertEqual(0.2, pc.watermark_x)
        self.assertEqual(0.2, pc.watermark_x_ratio)
        self.assertEqual(0.2, pc.decor_text_x)
        self.assertEqual(0.2, pc.decor_text_x_ratio)
        self.assertEqual(0.2, pc.lumi_text_x)
        self.assertEqual(0.2, pc.lumi_text_x_ratio)
        self.assertEqual(0.86, pc.watermark_y)
        self.assertEqual(0.8, pc.decor_text_y)
        self.assertEqual(0.8, pc.decor_text_y_ratio)
        self.assertEqual(0.8, pc.lumi_text_y)
        self.assertEqual(0.88, pc.watermark_y_ratio)
        self.assertEqual(0.04875, pc.watermark_size_ratio)
        self.assertEqual(0.835, pc.lumi_text_y_ratio)
        self.assertEqual(0.0375, pc.lumi_text_size_ratio)
        self.assertEqual(0.05, pc.decor_text_size)
        self.assertEqual(0.05, pc.decor_text_size_ratio)
        self.assertEqual(0.05, pc.lumi_text_size)
        self.assertEqual(0.25, pc.ratio_rel_size)
        self.assertEqual('', pc.xtitle)
        self.assertEqual('', pc.title)
        self.assertEqual('Internal', pc.watermark)
