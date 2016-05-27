__author__ = 'marcusmorgenstern'
__mail__ = ''

import numpy as np
import re
import ROOT
from base import InvalidInputError, _logger


def set_title_x(obj, title):
    if not hasattr(obj, "GetXaxis"):
        raise TypeError
    obj.GetXaxis().SetTitle(title)


def set_title_y(obj, title):
    if not hasattr(obj, "GetYaxis"):
        raise TypeError
    obj.GetYaxis().SetTitle(title)


def set_style_options(obj, style):
    allowed_attributes = ["marker", "line"]
    if not isinstance(style, dict):
        raise InvalidInputError("Invalid style config. Needs to be dictionary")
    for style_object, style_options in style.items():
        if style_object.lower() not in allowed_attributes:
            continue
        if not isinstance(style_options, dict):
            raise InvalidInputError("Invalid style option for " + style_object + ". Requires dict, but received " +
                                    str(type(style_options)))
        for style_option, value in style_options.items():
            try:
                getattr(obj, "Set" + style_object.capitalize() + style_option.capitalize())(value)
            except AttributeError:
                _logger.warning("Could not set rquested style " + style_object.capitalize() + style_option.capitalize()
                                + " for object " + str(obj))


def make_text(x, y, text, size=0.05, angle=0, font=42, color=ROOT.kBlack, NDC=True):
    t = ROOT.TLatex(x, y, text)
    t.SetTextSize(size)
    t.SetTextAngle(angle)
    t.SetTextFont(font)
    t.SetTextColor(color)
    t.SetNDC()
    return t


def rebin(self, histograms, factor=None):
    if factor is None:
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
        binning = np.array('d', factor)
        _logger.debug('rebin histogram %s asymmetrically' % (hist.GetName()))
        hist = __rebin_asymmetric_1d_hist(hist, len(factor), binning)
    else:
        raise InvalidInputError('Invalid binning: ' + str(factor))
    return hist


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
    hist.GetYaxis().SetTitle(hist.GetYaxis().GetTitle() + ' x %i' % n_bins)
    return hist.Rebin(n_bins - 1, hist.GetName(), bins)


def merge_overflow_bins(self, hists, xmax=None):
    if type(hists) == dict:
        for item in hists.values():
            self.__mergeOverflowBinsTH1F(item, xmax)
    else:
        self.__mergeOverflowBinsTH1F(hists, xmax)