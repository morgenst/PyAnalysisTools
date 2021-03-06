from __future__ import division
from builtins import str
from builtins import range
from past.utils import old_div
import re
from copy import deepcopy
from operator import itemgetter

from math import sqrt

import ROOT
from PyAnalysisTools.base import _logger, InvalidInputError
import PyAnalysisTools.PlottingUtils.Formatting as fm
import PyAnalysisTools.PlottingUtils.PlottingTools as pt
from PyAnalysisTools.PlottingUtils.PlotConfig import PlotConfig, get_default_color_scheme


def consistency_check_bins(obj1, obj2):
    try:
        return obj1.GetNbinsX() == obj2.GetNbinsX()
    except AttributeError:
        _logger.error('Try comparing no. of bins, but no histogram provided')
        raise InvalidInputError


def calculate_significance(signal, background):
    """
    Calculate significance as s/sqrt(b)
    :param signal: signal yield
    :param background: background yield
    :return: significance (0 if background=0)
    """
    try:
        return float(signal) / sqrt(float(background))
    except (ZeroDivisionError, ValueError):
        return 0.


def get_significance(signal, background, plot_config, canvas=None, upper_cut=False):
    """
    Calculate significance for cutting on some variable
    :param signal: signal histogram
    :param background: background histogram
    :param plot_config:
    :param canvas:
    :param upper_cut:
    :return:
    """
    significance_hist = signal.Clone("significance")
    if not consistency_check_bins(signal, background):
        _logger.error("Signal and background have different binnings.")
        raise InvalidInputError("Inconsistent binning")
    for ibin in range(signal.GetNbinsX() + 1):
        try:
            if not upper_cut:
                significance_hist.SetBinContent(ibin, calculate_significance(signal.Integral(-1, ibin),
                                                                             background.Integral(-1, ibin)))
            else:
                significance_hist.SetBinContent(ibin, calculate_significance(signal.Integral(ibin, -1),
                                                                             background.Integral(ibin, -1)))
        except ValueError:
            pass
    fm.set_title_y(significance_hist, "S/#sqrt{B}")
    if canvas is None:
        canvas = pt.plot_obj(significance_hist, plot_config)
    else:
        pt.add_object_to_canvas(canvas, significance_hist, plot_config)
    return canvas


def get_statistical_uncertainty_hist(hists):
    """
    Sum all histograms in hists and rename as stat.unc for statistical uncertainty overlay
    """

    if len(hists) == 0:
        return None
    statistical_uncertainty_hist = hists[0].Clone("stat.unc")
    for hist in hists[1:]:
        statistical_uncertainty_hist.Add(hist)
    return statistical_uncertainty_hist


def get_statistical_uncertainty_from_stack(stack):
    """
    Retrieve total statistical uncertainty histogram from THStack
    :param stack: stack plots
    :type stack:  ROOT.THStack
    :return: stat. uncertainty histogram
    :rtype: TH1F
    """
    return get_statistical_uncertainty_hist([h for h in stack.GetHists()])


def get_statistical_uncertainty_ratio(stat_unc_hist):
    try:
        stat_unc_hist_ratio = stat_unc_hist.Clone("stat.unc.ratio")
        for b in range(0, stat_unc_hist.GetNbinsX() + 1):
            stat_unc_hist_ratio.SetBinContent(b, 1.)
            if stat_unc_hist.GetBinContent(b) > 0.:
                stat_unc_hist_ratio.SetBinError(b, old_div(stat_unc_hist.GetBinError(b),
                                                           stat_unc_hist.GetBinContent(b)))
            else:
                stat_unc_hist_ratio.SetBinError(b, 0.)
        stat_unc_hist_ratio.SetMarkerStyle(20)
        stat_unc_hist_ratio.SetMarkerSize(0)
    except AttributeError:
        _logger.error('Stat. uncertainty input cannot be cloned. Likely invalid input {:s}'.format(str(stat_unc_hist)))
        return None
    return stat_unc_hist_ratio


def get_single_relative_systematics_ratio(nominal, stat_unc, systematic, color=None):
    ratio_hist = nominal.Clone("ratio_{:s}".format(systematic.GetName()))
    for b in range(nominal.GetNbinsX() + 1):
        nominal_yield = nominal.GetBinContent(b)
        if nominal_yield == 0.:
            ratio_hist.SetBinContent(b, stat_unc.GetBinContent(b) - 1.)
            continue
        uncertainty = old_div((systematic.GetBinContent(b) - nominal_yield), nominal_yield)
        uncertainty += stat_unc.GetBinError(b)
        ratio_hist.SetBinContent(b, 1.)
        ratio_hist.SetBinError(b, uncertainty)
    ratio_hist.SetMarkerStyle(20)
    ratio_hist.SetMarkerSize(0)
    if color:
        ratio_hist.SetMarkerColor(color)
        ratio_hist.SetMarkerColorAlpha(color, 0)
    return ratio_hist


def get_relative_systematics_ratio(nominal, stat_unc, systematic_category_hists):
    total_per_category_hist = None
    relative_syst_ratios = []
    default_colors = [6, 3, 4]
    for index, hist in enumerate(systematic_category_hists):
        if total_per_category_hist is None:
            total_per_category_hist = hist
        else:
            total_per_category_hist.Add(hist)
        relative_syst_ratios.append(get_single_relative_systematics_ratio(nominal, stat_unc, total_per_category_hist,
                                                                          color=default_colors[index]))
    return relative_syst_ratios


def get_KS(reference, compare):
    return reference.KolmogorovTest(compare)


def get_signal_acceptance(signal_yields, generated_events, plot_config=None):
    """
    Calculate signal acceptance
    :param signal_yields: process and signal yields after cut
    :type signal_yields: dict
    :param generated_events: generated MC statistics
    :type generated_events: dict
    :return: hist of signal acceptance
    :rtype: TH1
    """

    def make_acceptance_graph(data):
        data.sort(key=itemgetter(0))
        graph = ROOT.TGraph(len(data))
        for i, signal in enumerate(data):
            graph.SetPoint(i, signal[0], signal[1])
        ROOT.SetOwnership(graph, False)
        return graph

    acceptance_hists = []
    for process, yields in list(signal_yields.items()):
        yields['yield'] /= generated_events[process]
    for cut in list(signal_yields.values())[0]['cut']:
        yields = [(float(re.findall(r"\d{3,4}", process)[0]), eff[eff['cut'] == cut]['yield'])
                  for process, eff in signal_yields.items()]
        acceptance_hists.append((cut, make_acceptance_graph(yields)))
        acceptance_hists[-1][-1].SetName(cut)

    if plot_config is None:
        plot_config = PlotConfig(name="acceptance_all_cuts", color=get_default_color_scheme(),
                                 labels=[data[0] for data in acceptance_hists], xtitle="x-title",
                                 ytitle="efficiency [%]", draw="Marker", lumi=-1, watermark="Internal", )

    pc_log = deepcopy(plot_config)
    pc_log.name += "_log"
    pc_log.logy = True
    pc_log.ymin = 0.1
    canvas = pt.plot_objects([data[1] for data in acceptance_hists], plot_config)
    fm.decorate_canvas(canvas, plot_config=plot_config)
    canvas_log = pt.plot_objects([data[1] for data in acceptance_hists], pc_log)
    fm.decorate_canvas(canvas_log, plot_config=pc_log)
    acceptance_hists[-1][1].SetName("acceptance_final")
    pc_final = deepcopy(plot_config)
    pc_final.name = "acceptance_final_cuts"
    canvas_final = pt.plot_graph(deepcopy(acceptance_hists[-1][1]), pc_final)
    fm.decorate_canvas(canvas_final, plot_config=plot_config)
    return canvas, canvas_log, canvas_final
