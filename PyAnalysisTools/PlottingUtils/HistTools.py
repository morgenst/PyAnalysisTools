import re
from math import log10

import ROOT
from array import array
from PyAnalysisTools.base import InvalidInputError, _logger


def rebin(histograms, factor=None):
    """
    Rebin a histogram. Can be either symmetric, i.e. just merging X bins, or asymmetric
    :param histograms: hists to be rebinned
    :type histograms: list, dict, TH1
    :param factor: rebin factor. If integer a symmetric rebinning will be performed, if a list of bin boarders are
    provided an asymmetric rebinning will be performed
    :type factor: int, list
    :return: rebinned hists
    :rtype: type(hists), i.e. list, dict, TH1
    """
    if factor is None or factor == 1:
        return histograms
    if type(histograms) == dict:
        for key, hist in histograms.iteritems():
            if issubclass(hist.__class__, ROOT.TH1):
                histograms[key] = _rebin_hist(hist, factor)
            elif isinstance(hist, list):
                histograms[key] = [_rebin_hist(h, factor) for h in hist]
            else:
                raise InvalidInputError('Invalid binning: ' + str(factor))
    elif isinstance(histograms, list):
        histograms = [_rebin_hist(h, factor) for h in histograms]
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
    hist.GetYaxis().SetTitle(+ '{:s} x {:d}'.format(hist.GetYaxis().GetTitle(), n_binsx))
    return hist.Rebin(n_binsx - 1, hist.GetName(), bins_x)


def merge_overflow_bins(hists, x_max=None, y_max=None):
    """
    Merge overflow bins
    :param hists: hists for which merging should be applied
    :type hists: dict, list, TH1
    :param x_max: optional parameter to perform merging from a given x-value
    :type x_max: float
    :param y_max: optional parameter to perform merging from a given y-value (2D hists only)
    :type y_max: float
    :return: nothing
    :rtype: None
    """
    if type(hists) == dict:
        for item in hists.values():
            if isinstance(item, list):
                for i in item:
                    _merge_overflow_bins_1d(i, x_max)
            else:
                _merge_overflow_bins_1d(item, x_max)
    else:
        if isinstance(hists, ROOT.TH2):
            _merge_overflow_bins_2d(hists, x_max, y_max)
        else:
            _merge_overflow_bins_1d(hists, x_max)


def _merge_overflow_bins_1d(hist, x_max=None):
    if isinstance(hist, ROOT.TH2):
        return
    if x_max is not None:
        last_visible_bin = hist.FindBin(x_max)
    else:
        last_visible_bin = hist.GetNbinsX()
    hist.SetBinContent(last_visible_bin, hist.Integral(last_visible_bin, -1))
    for b in range(last_visible_bin+1, hist.GetNbinsX()+2):
        hist.SetBinContent(b, 0)

    
def _merge_overflow_bins_2d(hist, x_max=None, y_max=None):
    if x_max:
        last_visible_bin_x = hist.GetXaxis().FindBin(x_max)
    else:
        last_visible_bin_x = hist.GetNbinsX()
    if y_max:
        last_visible_bin_y = hist.GetYaxis().FindBin(y_max)
    else:
        last_visible_bin_y = hist.GetNbinsY()
    for i in range(hist.GetXaxis().GetNbins()):
        hist.SetBinContent(i+1, last_visible_bin_y, hist.Integral(i+1, i+1, last_visible_bin_y, -1))
    for i in range(hist.GetYaxis().GetNbins()):
        hist.SetBinContent(last_visible_bin_x, i+1, hist.Integral(last_visible_bin_x, -1, i+1, i+1))

        
def merge_underflow_bins(hists, x_min=None):
    """
    Merge underflow bins
    :param hists: hists for which merging should be applied
    :type hists: dict, list, TH1
    :param x_min: optional parameter to perform merging up to a given x-value
    :type x_min: float
    :return: nothing
    :rtype: None
    """
    if type(hists) == dict:
        for item in hists.values():
            if isinstance(item, list):
                for i in item:
                    _merge_underflow_bins_1d(i, x_min)
            else:
                _merge_underflow_bins_1d(item, x_min)
    else:
        _merge_underflow_bins_1d(hists, x_min)


def _merge_underflow_bins_1d(hist, x_min=None):
    if isinstance(hist, ROOT.TH2):
        return
    if x_min:
        first_visible_bin = hist.FindBin(x_min)
    else:
        first_visible_bin = 1
    hist.SetBinContent(first_visible_bin, hist.Integral(0, first_visible_bin))


def scale(hist, weight):
    """
    Wrapper around TH1::Scale. Scales each bin content by weight
    :param hist: histogram
    :type hist: TH1
    :param weight: scale factor
    :type weight: float
    :return: nothing
    :rtype: None
    """
    hist.Scale(weight)


def normalise(hists, integration_range=None, norm_scale=1.):
    """
    Wrapper for normalisation of histograms to a given scale in a given interval
    :param hists: histograms
    :type hists: list of dictionary of histograms
    :param integration_range: range in which integration should be performed (default fill range)
    :type integration_range: list (default: None)
    :param norm_scale: normalisation scale (default 1.)
    :type norm_scale: float
    :return: nothing
    :rtype: None
    """
    if integration_range is None:
        integration_range = [-1, -1]
    if type(hists) == dict:
        for h in hists.keys():
            hists[h] = normalise_hist(hists[h], integration_range, norm_scale)
    elif type(hists) == list:
        for h in hists:
            h = normalise_hist(h, integration_range, norm_scale)
    else:
        hists = normalise_hist(hists, integration_range, norm_scale)


def normalise_hist(hist, integration_range=[-1, -1], norm_scale=1.):
    """
    Perform histogram normalisation
    :param hist: single histogram
    :type hist: TH1
    :param integration_range: range in which integration should be performed (default fill range)
    :type integration_range: list (default: None)
    :param norm_scale: normalisation scale (default 1.)
    :type norm_scale: float
    :return: nothing
    :rtype: None
    """
    if isinstance(hist, ROOT.TH2):
        return _normalise_2d_hist(hist, integration_range, norm_scale)
    if isinstance(hist, ROOT.TH1):
        return _normalise_1d_hist(hist, integration_range, norm_scale)


def has_asymmetric_binning(hist):
    """
    Check if graph object has asymmetric x-binning
    :param hist: histogram
    :type hist: TH1
    :return: true/false on asymmetric binning
    :rtype: bool
    """
    return set([hist.GetBinWidth(b) for b in range(hist.GetNbinsX())]) > 1


def _normalise_1d_hist(hist, integration_range=[-1, -1], norm_scale=1.):
    if isinstance(hist, ROOT.THStack):
        return hist
    if has_asymmetric_binning(hist):
        for b in range(hist.GetNbinsX() + 1):
            hist.SetBinContent(b, hist.GetBinContent(b)/ hist.GetBinWidth(b))
    integral = hist.Integral(*integration_range)
    if integral == 0:
        return hist
    hist.Scale(norm_scale / integral)
    return hist


def _normalise_2d_hist(hist, integration_range=[-1,-1], norm_scale=1.):
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
    """
    Get colors of plotted objects from draw option
    :param hists: histograms
    :type hists: list
    :return: list of colors
    :rtype: list
    """
    def get_color():
        draw_option = hist.GetDrawOption()
        if isinstance(hist, ROOT.TEfficiency):
            return max(hist.GetPaintedGraph().GetLineColor(), hist.GetPaintedGraph().GetMarkerColor())
        if "hist" in draw_option.lower():
            return hist.GetLineColor()
    return [get_color() for hist in hists]


def set_axis_labels(obj, plot_config):
    """
    Set bin labels for x axis
    :param obj: plot object
    :type obj: TH1, TGraph
    :param plot_config: plot configuration
    :type plot_config: PlotConfig
    :return: nothing
    :rtype: None
    """
    if plot_config.axis_labels is None:
        return
    for b, label in enumerate(plot_config.axis_labels):
        obj.GetXaxis().SetBinLabel(b + 1, label)


def get_log_scale_x_bins(nbins, xmin, xmax):
    """
    Calculate bin boarders for equidistant bins for log scale x-axis binning
    :param nbins: number of bins
    :type nbins: int
    :param xmin: minimum of x-axis
    :type xmin: float
    :param xmax: maximum of x-axis
    :type xmax: float
    :return: binning
    :rtype:
    """
    log_min = log10(xmin)
    log_max = log10(xmax)
    bin_width = (log_max - log_min) / int(nbins)
    return [pow(10, log_min + i * bin_width) for i in range(0, int(nbins) + 1)]
