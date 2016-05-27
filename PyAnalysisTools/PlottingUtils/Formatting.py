__author__ = 'marcusmorgenstern'
__mail__ = ''

import re

import ROOT
import numpy as np

from PyAnalysisTools.base import InvalidInputError, _logger


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


