import re
import ROOT
from array import array
from PyAnalysisTools.base import InvalidInputError, _logger


def rebin(histograms, factor=None):
    if factor is None or factor == 1:
        return histograms
    if type(histograms) == dict:
        for key, hist in histograms.items():
            try:
                histograms[key].append(_rebin_hist(hist, factor))
            except KeyError:
                histograms[key] = [_rebin_hist(hist, factor)]
        else:
            raise InvalidInputError('Invalid binning: ' + str(factor))
    else:
        histograms = _rebin_hist(histograms, factor)
    return histograms


def _rebin_hist(hist, factor):
    if type(factor) is int:
        hist = _rebin_1d_hist(hist, factor)
    elif type(factor) == list:
        binning = array('d', factor)
        _logger.debug('rebin histogram %s asymmetrically' % (hist.GetName()))
        hist = __rebin_asymmetric_1d_hist(hist, len(factor), binning)
    else:
        raise InvalidInputError('Invalid binning: ' + str(factor))
    return hist


def _parse_y_title(hist):
    binning = None
    y_title = hist.GetYaxis().GetTitle()
    try:
        binning = float(re.findall('[0-9].+', y_title)[0])
    except IndexError:
        try:
            binning = int(re.findall('[0-9]+', y_title)[0])
        except IndexError:
            pass
    return binning


def _rebin_1d_hist(hist, factor):
    y_title = hist.GetYaxis().GetTitle()
    if not factor:
        return
    try:
        try:
            binning = float(re.findall('[0-9].+', y_title)[0])
        except ValueError:
            binning = int(re.findall('[0-9]+', y_title)[0])
        y_title = y_title.replace(str(binning), str(binning * factor))
    except IndexError, KeyError:
        pass
    hist.GetYaxis().SetTitle(y_title)
    return hist.Rebin(factor)


def __rebin_asymmetric_1d_hist(hist, n_bins, bins):
    binning = _parse_y_title(hist)
    if binning is not None:
        hist.GetYaxis().SetTitle(hist.GetYaxis().GetTitle() + ' x %i' % n_bins)
    return hist.Rebin(n_bins - 1, hist.GetName(), bins)


def __rebin_asymmetric_2d_hist(hist, n_binsx, bins_x):
    hist.GetYaxis().SetTitle(hist.GetYaxis().GetTitle() + ' x %i' % n_bins)
    return hist.Rebin(n_binsx - 1, hist.GetName(), bins_x)


def merge_overflow_bins(hists, x_max=None):
    if type(hists) == dict:
        for item in hists.values():
            _merge_overflow_bins_1d(item, x_max)
    else:
        _merge_overflow_bins_1d(hists, x_max)


def _merge_overflow_bins_1d(hist, x_max=None):
    if x_max:
        last_visible_bin = hist.FindBin(x_max)
    else:
        last_visible_bin = hist.GetNbinsX()
    hist.SetBinContent(last_visible_bin, hist.GetBinContent(last_visible_bin) + hist.Integral(last_visible_bin, -1))


def scale(hist, weight):
    hist.Scale(weight)


def normalise(histograms, integration_range=None):
    if integration_range is None:
        integration_range = [-1, -1]
    if type(histograms) == dict:
        for h in histograms.keys():
            histograms[h] = _normalise_1d_hist(histograms[h], integration_range)
    elif type(histograms) == list:
        for h in histograms:
            h = _normalise_1d_hist(h, integration_range)
    else:
        histograms = _normalise_1d_hist(histograms, integration_range)


def _normalise_1d_hist(hist, integration_range=[-1, -1]):
    if isinstance(hist, ROOT.THStack):
        return hist
    integral = hist.Integral(*integration_range)
    if integral == 0:
        return hist
    hist.Scale(1. / integral)
    return hist


def read_bin_from_label(hist, label):
    labels = [hist.GetXaxis().GetBinLabel(i) for i in range(hist.GetNbinsX() + 1)]
    matched_labels = filter(lambda l: re.search(label, l) is not None, labels)
    if len(matched_labels) == 0:
        _logger.error("Could not find label matching {:s} in {:s}".format(label, hist.GetName()))
        return None
    if len(matched_labels) > 1:
        _logger.warning("Found multiple matches for label {:s} in {:s}".format(label, hist.GetName()))
    return labels.index(matched_labels[0])


def get_colors(hists):
    def get_color():
        draw_option = hist.GetDrawOption()
        if isinstance(hist, ROOT.TEfficiency):
            return max(hist.GetPaintedGraph().GetLineColor(), hist.GetPaintedGraph().GetMarkerColor())
        #todo: refactor this part as information parsing is implemented in PlotConfig
        if "hist" in draw_option.lower():
            return hist.GetLineColor()
    return [get_color() for hist in hists]
