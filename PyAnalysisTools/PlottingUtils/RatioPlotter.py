from __future__ import division
from builtins import range
from builtins import object
from past.utils import old_div
import ROOT
from PyAnalysisTools.PlottingUtils.PlotConfig import PlotConfig
from PyAnalysisTools.ROOTUtils.ObjectHandle import get_objects_from_canvas_by_name
from PyAnalysisTools.base import _logger, InvalidInputError
from PyAnalysisTools.PlottingUtils import Formatting as fm
from PyAnalysisTools.PlottingUtils import PlottingTools as pt
from PyAnalysisTools.PlottingUtils import HistTools as ht
from PyAnalysisTools.PlottingUtils.HistTools import get_colors
from PyAnalysisTools.ROOTUtils import ObjectHandle as object_handle


class RatioCalculator(object):
    def __init__(self, **kwargs):
        kwargs.setdefault("rebin", None)
        self.reference = kwargs["reference"]
        self.compare = kwargs["compare"]
        self.rebin = kwargs["rebin"]

    def calculate_ratio(self):
        def calc_ratio(self, ref, comp):
            calc_func = ""
            if isinstance(ref, ROOT.TEfficiency):
                calc_func = "calculate_ratio_tefficiency"
            elif isinstance(ref, ROOT.TH1):
                calc_func = "calculate_ratio_hist"
            return getattr(self, calc_func)(ref, comp)

        ratios = []
        if isinstance(self.reference, list):
            for i, ref in enumerate(self.reference):
                ratios.append(calc_ratio(self, ref, self.compare[i]))

        elif isinstance(self.compare, list):
            for obj in self.compare:
                ratios.append(calc_ratio(self, self.reference, obj))
        else:
            ratios.append(calc_ratio(self, self.reference, self.compare))
        return ratios

    def calculate_ratio_tefficiency(self, reference, compare):
        ratio_graph = reference.GetPaintedGraph().Clone("ratio_" + compare.GetName())
        nbins = ratio_graph.GetN()
        for b in range(nbins):
            eff_compare = compare.GetEfficiency(b + 1)
            eff_reference = reference.GetEfficiency(b + 1)
            if eff_reference == 0. and eff_compare == 0.:
                ratio = 1.
            elif eff_reference == 0. and eff_compare != 0.:
                ratio = 0.
            else:
                ratio = old_div(eff_compare, eff_reference)
            x = ROOT.Double(0.)
            y = ROOT.Double(0.)
            ratio_graph.GetPoint(b, x, y)
            ratio_graph.SetPoint(b, x, ratio)
        return ratio_graph

    def calculate_ratio_hist(self, reference, compare):
        if self.rebin:
            compare = ht.rebin(compare, self.rebin)
            reference = ht.rebin(reference, self.rebin)
        ratio_hist = compare.Clone("ratio_" + compare.GetName())
        ratio_hist.Divide(reference)
        fm.set_title_y(ratio_hist, "ratio")
        return ratio_hist


class RatioPlotter(object):
    def __init__(self, **kwargs):
        kwargs.setdefault("plot_config", None)
        if "reference" not in kwargs:
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
            if isinstance(self.reference, list):
                self.plot_config.xtitle = self.reference[0].GetXaxis().GetTitle()
            else:
                self.plot_config.xtitle = self.reference.GetXaxis().GetTitle()
            if len(self.compare) > 1:
                colors = get_colors(self.compare)
                self.plot_config.color = colors
            self.plot_config.ordering = None
        self.plot_config.normalise = False
        _logger.debug('Plotting now ratio histograms')
        canvas = pt.plot_histograms(ratios, self.plot_config)
        if self.plot_config.enable_range_arrows:
            RatioPlotter.overlay_out_of_range_arrow(canvas)
        return canvas

    def add_uncertainty_to_canvas(self, canvas, hist, plot_config, n_systematics=1):
        ratio_hist = fm.get_objects_from_canvas_by_type(canvas, "TH1F")[0]
        arrows = object_handle.get_objects_from_canvas_by_type(canvas, "TArrow")
        if not isinstance(hist, list):
            hist = [hist]
            plot_config = [plot_config]
        canvas = pt.plot_hist(hist[0], plot_config[0], index=0)
        if len(hist) > 1:
            for i, unc_hist in enumerate(hist[1:]):
                if n_systematics == 0:
                    _logger.warning("Detected 0 systematics for hist {:s}".format(unc_hist.GetName()))
                    continue
                pc = plot_config[old_div(i, n_systematics)]
                pt.add_histogram_to_canvas(canvas, unc_hist, pc, index=i+1)
        pt.add_histogram_to_canvas(canvas, ratio_hist, self.plot_config)
        for a in arrows:
            a.Draw()
        return canvas

    def decorate_ratio_canvas(self, canvas):
        if self.plot_config.enable_legend:
            fm.add_legend_to_canvas(canvas, **self.plot_config.__dict__)

    def make_ratio_tefficiency(self, ratios):
        c = ROOT.TCanvas("C", "C")
        c.cd()
        colors = None
        if len(self.compare) > 1:
            colors = get_colors(self.compare)
        if isinstance(self.reference, list):
            self.reference[0].Draw("ap")
            for i in range(len(self.reference)):
                self.reference[i+1].Draw("psame")
                self.plot_config.xtitle = self.reference[0].GetPaintedGraph().GetXaxis().GetTitle()
        else:
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
        hist = get_objects_from_canvas_by_name(canvas, "Data")[0]
        y_min, y_max = hist.GetMinimum(), hist.GetMaximum()
        canvas.cd()
        for b in range(1, hist.GetNbinsX() + 1):
            bin_content = hist.GetBinContent(b)
            if bin_content >= y_min and bin_content < y_max:
                continue
            bin_center = hist.GetXaxis().GetBinCenter(b)
            if bin_content < y_min:
                a = ROOT.TArrow(bin_center, y_min+0.05*(y_max-y_min), bin_center, y_min, 0.03/(canvas.GetWw()/596.),
                                "|>")
            elif bin_content > y_max:
                a = ROOT.TArrow(bin_center, y_max-0.05*(y_max-y_min), bin_center, y_max, 0.03/(canvas.GetWw()/596.),
                                "|>")

            ROOT.SetOwnership(a, False)
            a.SetFillColor(10)
            a.SetFillStyle(1001)
            a.SetLineColor(ROOT.kBlue - 7)
            a.SetLineWidth(2)
            a.SetAngle(40)
            a.Draw()
        canvas.Modified()
        canvas.Update()
        return canvas

    @staticmethod
    def add_ratio_to_canvas(canvas, ratio, y_min=None, y_max=None, y_title=None, name=None, title=''):
        def scale_frame_text(fr, scale):
            x_axis = fr.GetXaxis()
            y_axis = fr.GetYaxis()
            y_axis.SetTitleSize(y_axis.GetTitleSize() * scale)
            y_axis.SetLabelSize(y_axis.GetLabelSize() * scale)
            y_axis.SetTitleOffset(1.1 * y_axis.GetTitleOffset() / scale)
            y_axis.SetLabelOffset(0.01)
            x_axis.SetTitleSize(x_axis.GetTitleSize() * scale)
            x_axis.SetLabelSize(x_axis.GetLabelSize() * scale)
            x_axis.SetTickLength(x_axis.GetTickLength() * scale)
            x_axis.SetTitleOffset(2.5 * x_axis.GetTitleOffset() / scale)
            x_axis.SetLabelOffset(2.5 * x_axis.GetLabelOffset() / scale)

        def reset_frame_text(fr):
            x_axis = fr.GetXaxis()
            y_axis = fr.GetYaxis()
            gs = ROOT.gStyle
            y_axis.SetTitleSize(gs.GetTitleSize('Y'))
            y_axis.SetLabelSize(gs.GetLabelSize('Y'))
            y_axis.SetTitleOffset(gs.GetTitleOffset('Y'))
            y_axis.SetLabelOffset(gs.GetLabelOffset('Y'))
            x_axis.SetTitleSize(gs.GetTitleSize('X'))
            x_axis.SetLabelSize(gs.GetLabelSize('X'))
            x_axis.SetTickLength(gs.GetTickLength('X'))

        if not canvas or not ratio:
            raise InvalidInputError("Either canvas or ratio not provided.")
        y_frac = 0.25
        if isinstance(ratio, ROOT.TCanvas):
            supported_types = ["TH1F", "TH1D", "TGraph", "TGraphAsymmErrors", "TEfficiency"]
            try:
                hratio = object_handle.get_objects_from_canvas_by_type(ratio, supported_types)[0]
            except IndexError:
                _logger.error("Could not find any supported hist type in canvas {:s}".format(ratio.GetName()))
                return canvas
        else:
            hratio = ratio
            ratio = pt.plot_obj(ratio, PlotConfig())

        if name is None:
            name = canvas.GetName() + "_ratio"
        c = pt.retrieve_new_canvas(name, title)
        c.Draw()
        pad1 = ROOT.TPad("pad1", "top pad", 0., y_frac, 1., 1.)
        pad1.SetBottomMargin(0.05)
        pad1.Draw()
        pad2 = ROOT.TPad("pad2", "bottom pad", 0., 0., 1,
                         (old_div((1 - y_frac) * canvas.GetBottomMargin(), y_frac) + 1) * y_frac - 0.009)
        pad2.SetBottomMargin(0.1)
        pad2.Draw()
        pad1.cd()
        object_handle.get_objects_from_canvas(canvas)
        try:
            stack = object_handle.get_objects_from_canvas_by_type(canvas, "THStack")[0]
        except IndexError:
            try:
                stack = object_handle.get_objects_from_canvas_by_type(canvas, "TEfficiency")[0]
            except IndexError:
                try:
                    stack = object_handle.get_objects_from_canvas_by_type(canvas, "TH1")[0]
                except IndexError:
                    stack = object_handle.get_objects_from_canvas_by_type(canvas, "TGraph")[0]
        stack.GetXaxis().SetTitleSize(0)
        stack.GetXaxis().SetLabelSize(0)
        scale = 1. / (1. - y_frac)
        scale_frame_text(stack, scale)
        canvas.DrawClonePad()

        pad2.cd()
        hratio.GetYaxis().SetNdivisions(505)
        scale = 1. / ((old_div((1 - y_frac) * (canvas.GetBottomMargin()), y_frac) + 1) * y_frac)

        reset_frame_text(hratio)
        scale_frame_text(hratio, scale)
        ratio.Update()
        ratio.SetBottomMargin(0.4)
        ratio.DrawClonePad()
        pad2.Update()
        xlow = pad2.GetUxmin()
        xup = pad2.GetUxmax()
        if ratio.GetLogx():
            stack = object_handle.get_objects_from_canvas_by_type(canvas, "TH1")[0]
            xlow = stack.GetXaxis().GetXmin()
            xup = stack.GetXaxis().GetXmax()
        line = ROOT.TLine(xlow, 1, xup, 1)
        line.Draw('same')
        pad2.Update()
        c.line = line
        pad = c.cd(1)
        if y_min is not None or y_max is not None:
            efficiency_obj = object_handle.get_objects_from_canvas_by_type(pad1, "TEfficiency")
            first = efficiency_obj[0]
            fm.set_title_y(first, y_title)
            for obj in efficiency_obj:
                fm.set_range_y(obj, y_min, y_max)
            pad1.Update()
        pad.Update()
        pad.Modified()
        return c
