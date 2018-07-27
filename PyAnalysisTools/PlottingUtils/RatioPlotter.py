import ROOT
from PyAnalysisTools.base import _logger, InvalidInputError
from PyAnalysisTools.PlottingUtils import Formatting as fm
from PyAnalysisTools.PlottingUtils import PlottingTools as pt
from PyAnalysisTools.PlottingUtils import HistTools as ht
from PyAnalysisTools.PlottingUtils.HistTools import get_colors
from PyAnalysisTools.ROOTUtils.ObjectHandle import get_objects_from_canvas_by_type


class RatioCalculator(object):
    def __init__(self, **kwargs):
        kwargs.setdefault("rebin", None)
        self.reference = kwargs["reference"]
        self.compare = kwargs["compare"]
        self.rebin = kwargs["rebin"]

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
        if self.rebin:
            compare = ht.rebin(compare, self.rebin)
            self.reference = ht.rebin(self.reference, self.rebin)
        ratio_hist = compare.Clone("ratio_" + compare.GetName())
        ratio_hist.Divide(self.reference)
        fm.set_title_y(ratio_hist, "ratio")
        return ratio_hist


class RatioPlotter(object):
    def __init__(self, **kwargs):
        kwargs.setdefault("plot_config", None)
        if not "reference" in kwargs:
            _logger.error("Missing reference")
            raise InvalidInputError("Missing reference")
        self.reference = kwargs["reference"]
        self.plot_config = kwargs["plot_config"]
        self.compare = kwargs["compare"]
        if not isinstance(self.compare, list):
            self.compare = [self.compare]
        if self.plot_config is not None:
            if not self.plot_config.name.startswith("ratio"):
                self.plot_config.name = "ratio_" + self.plot_config.name
        self.ratio_calculator = RatioCalculator(**kwargs)

    def make_ratio_plot(self):
        ratios = self.ratio_calculator.calculate_ratio()
        if not isinstance(self.reference, ROOT.TEfficiency):
            return self.make_ratio_histogram(ratios)
        elif isinstance(self.reference, ROOT.TEfficiency):
            return self.make_ratio_tefficiency(ratios)

    def make_ratio_histogram(self, ratios):
        if self.plot_config:
            self.plot_config.xtitle = self.reference.GetXaxis().GetTitle()
            if len(self.compare) > 1:
                colors = get_colors(self.compare)
                self.plot_config.color = colors
            self.plot_config.ordering = None
        canvas = pt.plot_histograms(ratios, self.plot_config)
        return canvas

    def add_uncertainty_to_canvas(self, canvas, hist, plot_config, n_systematics = 1):
        ratio_hist = fm.get_objects_from_canvas_by_type(canvas, "TH1F")[0]
        if not isinstance(hist, list):
            hist = [hist]
            plot_config = [plot_config]
        canvas = pt.plot_hist(hist[0], plot_config[0], index=0)
        if len(hist) > 1:
            for i, unc_hist in enumerate(hist[1:]):
                pc = plot_config[i/n_systematics]
                pt.add_histogram_to_canvas(canvas, unc_hist, pc, index=i - i/n_systematics * n_systematics)
        pt.add_histogram_to_canvas(canvas, ratio_hist, self.plot_config)
        return canvas

    def decorate_ratio_canvas(self, canvas):
        if self.plot_config.enable_legend:
            fm.add_legend_to_canvas(canvas,  **self.plot_config.__dict__)

    def make_ratio_tefficiency(self, ratios):
        c = ROOT.TCanvas("C", "C")
        c.cd()
        colors = None
        if len(self.compare) > 1:
            colors = get_colors(self.compare)
        self.reference.Draw("ap")
        self.plot_config.xtitle = self.reference.GetPaintedGraph().GetXaxis().GetTitle()
        self.plot_config.ytitle = "ratio"
        index = ROOT.gROOT.GetListOfCanvases().IndexOf(c)
        ROOT.gROOT.GetListOfCanvases().RemoveAt(index)
        if colors is not None:
            self.plot_config.color = colors[0]
        ratio_canvas = pt.plot_graph(ratios[0], self.plot_config)
        for ratio in ratios[1:]:
            if colors is not None:
                self.plot_config.color = colors[ratios.index(ratio)]
            pt.add_graph_to_canvas(ratio_canvas, ratio, self.plot_config)
        return ratio_canvas

    @staticmethod
    def overlay_out_of_range_arrow(canvas):
        hist = pt.get_objects_from_canvas_by_name(canvas, "Data")[0]
        y_min, y_max = hist.GetMinimum(), hist.GetMaximum()
        canvas.cd()
        for b in range(hist.GetNbinsX() + 1):
            bin_content = hist.GetBinContent(b)
            if bin_content > y_min and bin_content < y_max:
                continue
            bin_center = hist.GetXaxis().GetBinCenter(b)
            if bin_content < y_min:
                a = ROOT.TArrow(bin_center, y_min, bin_center, y_min+0.3, 0.03, "<")
                print "Draw arraow at ", bin_center, y_min, bin_center, y_min+0.3, 0.03, "<"
            elif bin_content < y_min:
                a = ROOT.TArrow(bin_center, y_max - 0.3, bin_center, y_max, 0.03, ">")
                print "Draw arraow at ", bin_center, y_max - 0.3, bin_center, y_max, 0.03, ">"
            ROOT.SetOwnership(a, False)
            a.Draw("sames")
        canvas.Modified()
        canvas.Update()
        return canvas
