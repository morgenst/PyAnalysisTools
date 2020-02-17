import os
import unittest

from array import array
from copy import deepcopy

import mock

import ROOT
from PyAnalysisTools.PlottingUtils import RatioPlotter as rp, Formatting
from PyAnalysisTools.PlottingUtils.PlotConfig import PlotConfig
from PyAnalysisTools.ROOTUtils.ObjectHandle import get_objects_from_canvas_by_type
from PyAnalysisTools.base import InvalidInputError

cwd = os.path.dirname(__file__)
ROOT.gROOT.SetBatch(True)


class TestRatioCalculator(unittest.TestCase):
    def test_ctor(self):
        calc = rp.RatioCalculator(reference='foo', compare='bar')
        self.assertEqual('foo', calc.reference)
        self.assertEqual('bar', calc.compare)
        self.assertIsNone(calc.rebin)

    def test_calculate_ratio_th1(self):
        ref = ROOT.TH1F("href", "", 10, 0., 10.)
        comp = ROOT.TH1F("hcomp", "", 10, 0., 10.)
        for i in range(1, 10):
            ref.SetBinContent(i, i)
            comp.SetBinContent(i, i*i)
        calc = rp.RatioCalculator(reference=ref, compare=comp)
        ratio = calc.calculate_ratio()[0]
        calc_list = rp.RatioCalculator(reference=[ref], compare=[comp])
        ratio_list = calc_list.calculate_ratio()[0]
        for i in range(1, 10):
            self.assertEqual(i * i / float(i), ratio.GetBinContent(i))
            self.assertEqual(i * i / float(i), ratio_list.GetBinContent(i))

    def test_calculate_ratio_th1_rebin(self):
        ref = ROOT.TH1D("href", "", 10, 0., 10.)
        comp = ROOT.TH1D("hcomp", "", 10, 0., 10.)
        for i in range(1, 10):
            ref.SetBinContent(i, i)
            comp.SetBinContent(i, i*i)
        calc = rp.RatioCalculator(reference=ref, compare=comp, rebin=2)
        ratio = calc.calculate_ratio()[0]
        for j, i in enumerate(range(1, 9, 2)):
            self.assertEqual((i * i + (i+1.) * (i+1.)) / (2. * i + 1), ratio.GetBinContent(j+1))

    def test_calculate_ratio_teff(self):
        ref = ROOT.TH1F("href", "", 10, 0., 10.)
        comp = ROOT.TH1F("hcomp", "", 10, 0., 10.)
        comp2 = ROOT.TH1F("hcomp2", "", 10, 0., 10.)
        for i in range(2, 10):
            ref.SetBinContent(i, i)
            comp.SetBinContent(i, i*i)
            comp2.SetBinContent(i, 2. * i * i)
        eff_ref = ROOT.TEfficiency(ref, comp)
        eff_comp = ROOT.TEfficiency(ref, comp2)
        c = ROOT.TCanvas()
        c.cd()
        eff_ref.Paint('ap')
        eff_comp.Paint('psame')
        calc = rp.RatioCalculator(reference=eff_ref, compare=eff_comp)
        ratio = calc.calculate_ratio()[0]
        self.assertEqual(array('f', [1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), array('f', ratio.GetY()))


class TestRatioPlotter(unittest.TestCase):
    def test_ctor(self):
        plotter = rp.RatioPlotter(reference='foo', compare='bar')
        self.assertEqual('foo', plotter.reference)
        self.assertEqual(['bar'], plotter.compare)
        self.assertIsNone(plotter.plot_config)

    def test_ctor_missing_input(self):
        self.assertRaises(InvalidInputError, rp.RatioPlotter)

    def test_make_ratio_plot(self):
        ref = ROOT.TH1F("href", "", 10, 0., 10.)
        comp = ROOT.TH1F("hcomp", "", 10, 0., 10.)
        for i in range(1, 10):
            ref.SetBinContent(i, i)
            comp.SetBinContent(i, i * i)
        plotter = rp.RatioPlotter(reference=ref, compare=comp)
        ratio_canvas = plotter.make_ratio_plot()
        ratio = get_objects_from_canvas_by_type(ratio_canvas, 'TH1')[0]
        for i in range(1, 10):
            self.assertEqual(i * i / float(i), ratio.GetBinContent(i))

    def test_make_ratio_plot_with_pc(self):
        pc = PlotConfig()
        ref = ROOT.TH1F("href", "", 10, 0., 10.)
        comp = ROOT.TH1F("hcomp", "", 10, 0., 10.)
        comp2 = ROOT.TH1F("hcomp2", "", 10, 0., 10.)
        for i in range(1, 10):
            ref.SetBinContent(i, i)
            comp.SetBinContent(i, i * i)
        pc.enable_range_arrows = True
        plotter = rp.RatioPlotter(reference=ref, compare=[comp, comp2], plot_config=pc)
        ratio_canvas = plotter.make_ratio_plot()
        ratio = get_objects_from_canvas_by_type(ratio_canvas, 'TH1')[0]
        for i in range(1, 10):
            self.assertEqual(i * i / float(i), ratio.GetBinContent(i))

    def test_make_ratio_plot_tefficiency(self):
        ref = ROOT.TH1F("href", "", 10, 0., 10.)
        comp = ROOT.TH1F("hcomp", "", 10, 0., 10.)
        comp2 = ROOT.TH1F("hcomp2", "", 10, 0., 10.)
        for i in range(2, 10):
            ref.SetBinContent(i, i)
            comp.SetBinContent(i, i*i)
            comp2.SetBinContent(i, 2. * i * i)
        eff_ref = ROOT.TEfficiency(ref, comp)
        eff_comp = ROOT.TEfficiency(ref, comp2)
        c = ROOT.TCanvas()
        c.cd()
        eff_ref.Paint('ap')
        eff_comp.Paint('psame')
        plotter = rp.RatioPlotter(reference=eff_ref, compare=eff_comp)
        ratio_canvas = plotter.make_ratio_plot()
        ratio = get_objects_from_canvas_by_type(ratio_canvas, 'TGraphAsymmErrors')[0]
        self.assertEqual(array('f', [1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), array('f', ratio.GetY()))

    def test_make_ratio_plot_tefficiency_list_input(self):
        ref = ROOT.TH1F("href", "", 10, 0., 10.)
        comp = ROOT.TH1F("hcomp", "", 10, 0., 10.)
        comp2 = ROOT.TH1F("hcomp2", "", 10, 0., 10.)
        for i in range(2, 10):
            ref.SetBinContent(i, i)
            comp.SetBinContent(i, i*i)
            comp2.SetBinContent(i, 2. * i * i)
        eff_ref = ROOT.TEfficiency(ref, comp)
        eff_comp = ROOT.TEfficiency(ref, comp2)
        c = ROOT.TCanvas()
        c.cd()
        eff_ref.Paint('ap')
        eff_comp.Paint('psame')
        plotter = rp.RatioPlotter(reference=[eff_ref], compare=[eff_comp])
        ratio_canvas = plotter.make_ratio_plot()
        ratio = get_objects_from_canvas_by_type(ratio_canvas, 'TGraphAsymmErrors')[0]
        self.assertEqual(array('f', [1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), array('f', ratio.GetY()))

    def test_overlay(self):
        hist = ROOT.TH1F("Data", "", 10, 0., 10.)
        for i in range(2, 10):
            hist.SetBinContent(i, i)
        hist.SetMinimum(3)
        hist.SetMaximum(5)
        c = ROOT.TCanvas()
        c.cd()
        hist.Draw()
        canvas = rp.RatioPlotter.overlay_out_of_range_arrow(c)
        self.assertEqual(8, len(get_objects_from_canvas_by_type(canvas, 'TArrow')))

    def test_overlay_missing_data(self):
        hist = ROOT.TH1F("MC", "", 10, 0., 10.)
        c = ROOT.TCanvas()
        c.cd()
        hist.Draw()
        canvas = rp.RatioPlotter.overlay_out_of_range_arrow(c)
        self.assertEqual(0, len(get_objects_from_canvas_by_type(canvas, 'TArrow')))

    def test_add_ratio_to_canvas(self):
        hist = ROOT.TH1F("Data", "", 10, 0., 10.)
        for i in range(2, 10):
            hist.SetBinContent(i, i)
        c = ROOT.TCanvas()
        c.cd()
        hist.Draw()
        c2 = deepcopy(c)
        c2.SetLogx()
        canvas = rp.RatioPlotter.add_ratio_to_canvas(c, c2)
        self.assertEqual(2, len(get_objects_from_canvas_by_type(canvas, 'TPad')))

    def test_add_ratio_to_canvas_invalid_type(self):
        hist = ROOT.TH1I("Data", "", 10, 0., 10.)
        for i in range(2, 10):
            hist.SetBinContent(i, i)
        c = ROOT.TCanvas()
        c.cd()
        hist.Draw()
        c2 = deepcopy(c)
        canvas = rp.RatioPlotter.add_ratio_to_canvas(c, c2)
        self.assertEqual(0, len(get_objects_from_canvas_by_type(canvas, 'TPad')))

    def test_add_ratio_to_canvas_hist_input(self):
        hist = ROOT.TH1F("Data", "", 10, 0., 10.)
        for i in range(2, 10):
            hist.SetBinContent(i, i)
        c = ROOT.TCanvas()
        c.cd()
        hist.Draw()
        canvas = rp.RatioPlotter.add_ratio_to_canvas(c, hist)
        self.assertEqual(2, len(get_objects_from_canvas_by_type(canvas, 'TPad')))

    def test_add_ratio_to_canvas_exception_invalid_input(self):
        c = ROOT.TCanvas()
        self.assertRaises(InvalidInputError, rp.RatioPlotter.add_ratio_to_canvas, c, None)
        self.assertRaises(InvalidInputError, rp.RatioPlotter.add_ratio_to_canvas, None, c)

    def test_add_uncertainty_to_canvas(self):
        hist = ROOT.TH1F("Data", "", 10, 0., 10.)
        for i in range(2, 10):
            hist.SetBinContent(i, i)
        c = ROOT.TCanvas()
        c.cd()
        hist.Draw()
        canvas = rp.RatioPlotter(reference=hist, compare=hist,
                                 plot_config=PlotConfig()).add_uncertainty_to_canvas(c, [hist, hist],
                                                                                     plot_config=PlotConfig())
        self.assertEqual(3, len(get_objects_from_canvas_by_type(canvas, 'TH1F')))

    @staticmethod
    def patch_leg(c, **_):
        leg = ROOT.TLegend(0., 1., 0., 1.)
        ROOT.SetOwnership(leg, False)
        leg.Draw("same")

    @mock.patch.object(Formatting, 'add_legend_to_canvas', patch_leg.__func__)
    def test_decorate(self):
        hist = ROOT.TH1F("Data", "", 10, 0., 10.)
        for i in range(2, 10):
            hist.SetBinContent(i, i)
        c = ROOT.TCanvas()
        c.cd()
        rp.RatioPlotter(reference=hist, compare=hist,
                        plot_config=PlotConfig(enable_legend=True)).decorate_ratio_canvas(c)
        self.assertEqual(1, len(get_objects_from_canvas_by_type(c, 'TLegend')))
