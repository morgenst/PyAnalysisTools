import ROOT
from PyAnalysisTools.base import _logger, InvalidInputError
from PyAnalysisTools.PlottingUtils import Formatting as FM
from PyAnalysisTools.PlottingUtils import PlottingTools as PT


class RatioCalculator(object):
    def __init__(self, **kwargs):
        self.reference = kwargs["reference"]
        self.compare = kwargs["compare"]

    def calculate_ratio(self):
        if isinstance(self.reference, ROOT.TEfficiency):
            calc_func = "calculate_ratio_tefficiency"
        elif isinstance(self.reference, ROOT.TH1):
            calc_func = "calculate_ratio_hist"
        ratios = []
        if isinstance(self.compare, list):
            for obj in self.compare:
                ratios.append(getattr(self, calc_func)(obj))
        else:
            ratios.append(getattr(self, calc_func)(self.compare))
        return ratios

    def calculate_ratio_tefficiency(self, compare):
        ratio_graph = self.reference.GetPaintedGraph().Clone("ratio_" + compare.GetName())
        nbins = ratio_graph.GetN()
        for b in range(nbins):
            eff_compare = compare.GetEfficiency(b + 1)
            eff_reference = self.reference.GetEfficiency(b + 1)
            if eff_reference == 0. and eff_compare == 0.:
                ratio = 1.
            elif eff_reference == 0. and eff_compare != 0.:
                ratio = 0.
            else:
                ratio = eff_compare / eff_reference
            x = ROOT.Double(0.)
            y = ROOT.Double(0.)
            ratio_graph.GetPoint(b, x, y)
            ratio_graph.SetPoint(b, x, ratio)
        return ratio_graph

    def calculate_ratio_hist(self, compare):
        ratio_hist = compare.Clone("ratio_" + compare.GetName())
        ratio_hist.Divide(self.reference)
        FM.set_title_y(ratio_hist, "ratio")
        return ratio_hist


class RatioPlotter(object):
    def __init__(self, **kwargs):
        if not "reference" in kwargs:
            _logger.error("Missing reference")
            raise InvalidInputError("Missing reference")
        self.reference = kwargs["reference"]
        self.plot_config = kwargs["plot_config"]
        self.ratio_calculator = RatioCalculator(**kwargs)

    def make_ratio_plot(self):
        ratios = self.ratio_calculator.calculate_ratio()
        if not isinstance(self.reference, ROOT.TEfficiency):
            return self.make_ratio_histogram(ratios)
        elif isinstance(self.reference, ROOT.TEfficiency):
            return self.make_ratio_tefficiency(ratios)

    def make_ratio_histogram(self, ratios):
        self.plot_config.xtitle = self.reference.GetXaxis().GetTitle()
        return PT.plot_histograms(ratios, self.plot_config)

    def make_ratio_tefficiency(self, ratios):
        c = ROOT.TCanvas("C", "C")
        c.cd()
        self.reference.Draw("ap")
        self.plot_config.xtitle = self.reference.GetPaintedGraph().GetXaxis().GetTitle()
        self.plot_config.ytitle = "ratio"
        index = ROOT.gROOT.GetListOfCanvases().IndexOf(c)
        ROOT.gROOT.GetListOfCanvases().RemoveAt(index)
        return PT.plot_graph(ratios[0], self.plot_config)
