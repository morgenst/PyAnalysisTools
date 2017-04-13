from math import sqrt
from PyAnalysisTools.base import _logger, InvalidInputError
import PyAnalysisTools.PlottingUtils.Formatting as FM
import PyAnalysisTools.PlottingUtils.PlottingTools as PT


def consistency_check_bins(obj1, obj2):
    return obj1.GetNbinsX() == obj2.GetNbinsX()


def calculate_significance(signal, background):
    try:
        return float(signal)/sqrt(float(signal) + float(background))
    except ZeroDivisionError:
        return 0.


def get_significance(signal, background):
    _logger.error("Not implemented yet. Uses just bin contents rather than integrals")
    significance_hist = signal.Clone("significance")
    if not consistency_check_bins(signal, background):
        _logger.error("Signal and background have different binnings.")
        raise InvalidInputError("Inconsistent binning")
    for ibin in range(signal.GetNbinsX() + 1):
        significance_hist.SetBinContent(ibin, calculate_significance(signal.Integral(-1,ibin),
                                                                     background.Integral(-1,ibin)))
    FM.set_title_y(significance_hist, "S/#sqrt(S + B)")
    canvas = PT.retrieve_new_canvas("significance", "")
    canvas.cd()
    #todo: call addHistogram to canvas (see MMPP-160)
    significance_hist.SetLineColor(2)
    significance_hist.SetFillColor(0)
    significance_hist.Draw("l")
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
    for b in range(0, stat_unc_hist.GetNbinsX()):
        stat_unc_hist_ratio.SetBinContent(b, 1.)
        if stat_unc_hist.GetBinContent(b) > 0.:
            stat_unc_hist_ratio.SetBinError(b, stat_unc_hist.GetBinError(b) / stat_unc_hist.GetBinContent(b))
        else:
            stat_unc_hist_ratio.SetBinError(b, 0.)
    return stat_unc_hist_ratio


def get_KS(reference, compare):
    return reference.KolmogorovTest(compare)
