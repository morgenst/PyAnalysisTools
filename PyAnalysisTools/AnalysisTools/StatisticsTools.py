import re
from copy import deepcopy
from operator import itemgetter

import numpy as np
from math import sqrt

import ROOT
from PyAnalysisTools.base import _logger, InvalidInputError
import PyAnalysisTools.PlottingUtils.Formatting as fm
import PyAnalysisTools.PlottingUtils.PlottingTools as pt
from PyAnalysisTools.PlottingUtils.PlotConfig import PlotConfig, get_default_color_scheme


def consistency_check_bins(obj1, obj2):
    return obj1.GetNbinsX() == obj2.GetNbinsX()


def calculate_significance(signal, background):
    try:
        print float(signal)/sqrt(float(background))

        return float(signal)/sqrt(float(background))
    except ZeroDivisionError:
        print 0
        return 0.


def get_significance(signal, background, plot_config, canvas=None):
    #significance_hist_up = signal.Clone("significance_up")
    significance_hist_down = signal.Clone("significance_down")

    if not consistency_check_bins(signal, background):
        _logger.error("Signal and background have different binnings.")
        raise InvalidInputError("Inconsistent binning")
    for ibin in range(signal.GetNbinsX() + 1):
        try:
            significance_hist_down.SetBinContent(ibin, calculate_significance(signal.Integral(-1, ibin),
                                                                         background.Integral(-1, ibin)))
        except ValueError:
            pass
        # try:
        #     significance_hist_up.SetBinContent(ibin, calculate_significance(signal.Integral(ibin, -1),
        #                                                                  background.Integral(ibin, -1)))
        # except ValueError:
        #     pass
    fm.set_title_y(significance_hist_down, "S/#sqrt{B}")
    if canvas is None:
        
        canvas = pt.plot_obj(significance_hist_down, plot_config)
    else:
        pt.add_object_to_canvas(canvas, significance_hist, plot_config)
    return canvas


def get_statistical_uncertainty_hist(hists):
    statistical_uncertainty_hist = hists[0].Clone("stat.unc")
    for hist in hists[1:]:
        statistical_uncertainty_hist.Add(hist)
    return statistical_uncertainty_hist


def get_statistical_uncertainty_from_stack(stack):
    return get_statistical_uncertainty_hist([h for h in stack.GetHists()])


def get_statistical_uncertainty_ratio(stat_unc_hist):
    stat_unc_hist_ratio = stat_unc_hist.Clone("stat.unc.ratio")
    for b in range(0, stat_unc_hist.GetNbinsX() + 1):
        stat_unc_hist_ratio.SetBinContent(b, 1.)
        if stat_unc_hist.GetBinContent(b) > 0.:
            stat_unc_hist_ratio.SetBinError(b, stat_unc_hist.GetBinError(b) / stat_unc_hist.GetBinContent(b))
        else:
            stat_unc_hist_ratio.SetBinError(b, 0.)
    stat_unc_hist_ratio.SetMarkerStyle(20)
    stat_unc_hist_ratio.SetMarkerSize(0)
    return stat_unc_hist_ratio


def get_single_relative_systematics_ratio(nominal, stat_unc, systematic, color=None):
    ratio_hist = nominal.Clone("ratio_{:s}".format(systematic.GetName()))
    for b in range(nominal.GetNbinsX()+1):
        nominal_yield = nominal.GetBinContent(b)
        if nominal_yield == 0.:
            ratio_hist.SetBinContent(b, stat_unc.GetBinContent(b) - 1.)
            continue
        uncertainty = (systematic.GetBinContent(b) - nominal_yield) / nominal_yield
        uncertainty += stat_unc.GetBinError(b)
        ratio_hist.SetBinContent(b, 1.)
        ratio_hist.SetBinError(b, uncertainty)
    ratio_hist.SetMarkerStyle(20)
    ratio_hist.SetMarkerSize(0)
    if color:
        ratio_hist.SetMarkerColor(color)
        ratio_hist.SetMarkerColorAlpha(color, 0)
    return ratio_hist


def get_relative_systematics_ratio(nominal, stat_unc, systematic_category_hists, color=None):
    total_per_category_hist = None
    relative_syst_ratios = []
    default_colors = [6,3,4]
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


def get_signal_acceptance(signal_yields, generated_events, plot_config, process_config):
    """
    Calculate signal acceptance
    :param signal_yields: process and signal yields after cut
    :type signal_yields: dict
    :param generated_events: generated MC statistics
    :type generated_events: dict
    :param process_config: process configs
    :type process_config: dict
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

    #TODO: refactoring required
    acceptance = [(float(re.findall("\d{3,4}", process)[0]), process,
                   yields) for process, yields in signal_yields.iteritems()]
    if isinstance(acceptance[0][2], (np.ndarray, np.generic)):
        acceptance = [[(mass, cut_yield["cut"],
                        cut_yield["yield"] / generated_events[process] * 100.) for cut_yield in yields]
                      for mass, process, yields in acceptance]
    acceptance_hists = []
    if isinstance(acceptance[0], list):
        for icut in range(len(acceptance[0])):
            cut_name = acceptance[0][icut][1]
            acceptance_hists.append((cut_name, make_acceptance_graph([(signal[icut][0],
                                                                      signal[icut][2]) for signal in acceptance])))
            acceptance_hists[-1][-1].SetName(cut_name)
    plot_config = PlotConfig(name="acceptance_all_cuts", color=get_default_color_scheme(),
                    labels=[data[0] for data in acceptance_hists],
                    xtitle="Gluino mass [GeV]", ytitle="efficiency [%]", draw="Marker", lumi=-1, watermark="Internal", watermark_size=0.02, watermark_offset = 1,
                    ymin=0., ymax=100.)

    pc_log = deepcopy(plot_config)
    pc_log.name += "_log"
    pc_log.logy = True
    pc_log.ymin = 0.1
    canvas = pt.plot_objects([data[1] for data in acceptance_hists], plot_config)
    #fm.add_legend_to_canvas(canvas, labels=plot_config.labels)
    fm.decorate_canvas(canvas, plot_config=plot_config)
    canvas_log = pt.plot_objects([data[1] for data in acceptance_hists], pc_log)
    #fm.add_legend_to_canvas(canvas_log, labels=pc_log.labels)
    fm.decorate_canvas(canvas_log, plot_config=pc_log)
    acceptance_hists[-1][1].SetName("acceptance_final")
    pc_final = deepcopy(plot_config)
    pc_final.name = "acceptance_final_cuts"
    canvas_final = pt.plot_graph(deepcopy(acceptance_hists[-1][1]), pc_final)
    fm.decorate_canvas(canvas_final, plot_config=plot_config)
    return canvas, canvas_log, canvas_final
