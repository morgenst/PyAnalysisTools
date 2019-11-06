import os
import unittest
from builtins import object
from builtins import range
from random import random as rndm

# import nosedep
from mock import MagicMock

import ROOT
from PyAnalysisTools.PlottingUtils import PlottingTools as pt
from PyAnalysisTools.PlottingUtils.PlotConfig import PlotConfig
from PyAnalysisTools.base import InvalidInputError

cwd = os.path.dirname(__file__)
ROOT.gROOT.SetBatch(True)


class TestPlottingTools(unittest.TestCase):
    def setUp(self):
        self.hist = ROOT.TH1F('h', '', 10, -1., 1.)
        self.hist.FillRandom('gaus', 10000)
        self.hist_2d = ROOT.TH2F('h2', '', 10, -1., 1., 10, -1., 1.)
        self.graph = ROOT.TGraph(10)
        list([self.hist_2d.Fill(rndm(), rndm()) for i in list(range(10000))])
        self.plot_config = PlotConfig(name='foo')
        self.plot_config.ymin = 1.
        self.plot_config.ymax = 1.
        self.plot_config.logx = True
        self.plot_config.logy = True
        self.plot_config.normalise = True

    def tearDown(self):
        del self.hist
        del self.hist_2d
        del self.graph
        list([c.Close() for c in ROOT.gROOT.GetListOfCanvases()])

    def test_new_canvas(self):
        """
        ROOT redefines canvas size to add menu bar - that's why the original size has to be corrected in the check
        """
        c = pt.retrieve_new_canvas('foo', '', 1000, 1200)
        self.assertEqual('foo', c.GetName())
        self.assertEqual(1000 - 4, c.GetWw())
        self.assertEqual(1200 - 28, c.GetWh())

    def test_plot_1d(self):
        c = pt.plot_obj(self.hist, self.plot_config)
        self.assertEqual(2, len(c.GetListOfPrimitives()))

    def test_plot_1d_multiple(self):
        c = pt.plot_objects([self.hist], self.plot_config)
        self.assertEqual(2, len(c.GetListOfPrimitives()))

    def test_add_plot_1d(self):
        c = pt.plot_obj(self.hist, self.plot_config)
        pt.add_object_to_canvas(c, self.hist, self.plot_config)
        self.assertEqual(3, len(c.GetListOfPrimitives()))

    def test_plot_2d(self):
        self.plot_config.draw_option = 'COLZ'
        self.plot_config.ztitle = 'ztitle'
        c = pt.plot_obj(self.hist_2d, self.plot_config)
        self.assertEqual(2, len(c.GetListOfPrimitives()))

    def test_plot_graph(self):
        self.plot_config.draw_option = 'ap'
        c = pt.plot_obj(self.graph, self.plot_config)
        self.assertEqual(2, len(c.GetListOfPrimitives()))

    def test_add_graph(self):
        self.plot_config.draw_option = 'ap'
        c = pt.plot_obj(self.graph, self.plot_config)
        pt.add_object_to_canvas(c, self.graph.Clone(), self.plot_config)
        self.assertEqual(3, len(c.GetListOfPrimitives()))

    def test_plot_stack_exception_input(self):
        self.assertRaises(InvalidInputError, pt.plot_stack, self.hist, self.plot_config)

    def test_plot_stack(self):
        c = pt.plot_stack([self.hist], self.plot_config)
        self.assertEqual(2, len(c.GetListOfPrimitives()))

    def test_add_to_stack(self):
        c = pt.plot_stack([self.hist], self.plot_config)
        pt.add_signal_to_stack(c, [self.hist])
        self.assertEqual(3, len(c.GetListOfPrimitives()))

    def test_blind(self):
        htmp = self.hist.Clone()
        pt.blind_data(htmp, 0.)
        for i in range(11):
            if i < 6:
                self.assertEqual(self.hist.GetBinContent(i), htmp.GetBinContent(i))
            else:
                self.assertEqual(0., htmp.GetBinContent(i))

    def test_plot_graphs(self):
        c = pt.plot_graphs([self.graph], self.plot_config)
        self.assertEqual(1, len([p for p in c.GetListOfPrimitives() if p.InheritsFrom('TGraph')]))

    def test_plot_graphs_dict(self):
        c = pt.plot_graphs({'foo': self.graph}, self.plot_config)
        self.assertEqual(1, len([p for p in c.GetListOfPrimitives() if p.InheritsFrom('TGraph')]))

    # @nosedep.depends(after=test_add_plot_1d)
    @unittest.skip("API change")
    def test_ratio(self):
        c = pt.plot_obj(self.hist, self.plot_config).Clone()
        c_ratio = pt.plot_obj(self.hist, self.plot_config)
        c = pt.add_ratio_to_canvas(c, c_ratio)
        self.assertEqual(2, len([p for p in c.GetListOfPrimitives() if p.InheritsFrom('TPad')]))

    # @nosedep.depends(after=test_add_plot_1d)
    @unittest.skip("API change")
    def test_ratio_hist(self):
        c = pt.plot_obj(self.hist, self.plot_config)
        c = pt.add_ratio_to_canvas(c, self.hist)
        self.assertEqual(2, len([p for p in c.GetListOfPrimitives() if p.InheritsFrom('TPad')]))

    # @nosedep.depends(after=test_add_plot_1d)
    @unittest.skip("API change")
    def test_ratio_empty(self):
        c = pt.plot_obj(self.hist, self.plot_config)
        c = pt.add_ratio_to_canvas(c, ROOT.TCanvas('ratio'))
        self.assertEqual(0, len([p for p in c.GetListOfPrimitives() if p.InheritsFrom('TPad')]))

    def test_add_fit_result(self):
        class FitResult(object):
            pass

        class RooVar(object):
            pass

        rv = RooVar()
        rv.GetName = MagicMock(return_value='roo_var')
        rv.getValV = MagicMock(return_value=2)
        rv.getError = MagicMock(return_value=0.2)

        fr = FitResult()
        fr.floatParsFinal = MagicMock(return_value=[rv, rv])

        c = ROOT.TCanvas()
        pt.add_fit_to_canvas(c, fr)
        self.assertEqual(1, len([p for p in c.GetListOfPrimitives() if p.InheritsFrom('TLatex')]))

    def test_apply_style(self):
        pt.apply_style(self.hist, ['Line'], style_attr=5, color=ROOT.kRed)
        self.assertEqual(ROOT.kRed, self.hist.GetLineColor())
        self.assertEqual(5, self.hist.GetLineStyle())

    def test_project(self):
        class Tree(object):
            pass

        tree = Tree()
        tree.GetName = MagicMock(return_value='mock_tree')
        tree.Project = MagicMock(return_value=self.hist.GetEntries())
        weight = 'DATA: w_data * MC: w_MC * weight'
        self.assertEqual(self.hist, pt.project_hist(tree, self.hist, 'foo', weight=weight, is_data=True))
