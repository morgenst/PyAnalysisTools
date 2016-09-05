__author__ = 'marcusmorgenstern'
__mail__ = ''

import re

import ROOT
import numpy as np
import os

from PyAnalysisTools.base import InvalidInputError, _logger
from PyAnalysisTools.ROOTUtils.ObjectHandle import get_objects_from_canvas_by_type


def load_atlas_style():
    try:
        base_path = os.path.dirname(os.path.join(os.path.realpath(__file__)))
        ROOT.gROOT.LoadMacro(os.path.join(base_path, 'AtlasStyle/AtlasStyle.C'))
        ROOT.SetAtlasStyle()
    except Exception as e:
        _logger.error("Could not find Atlas style files in %s" % os.path.join(base_path, 'AtlasStyle'))


def decorate_canvas(canvas, config):
    if hasattr(config, "watermark"):
        add_atlas_label(canvas, config.watermark, {"x": 0.15, "y": 0.96}, size=0.03, offset=0.08)
    if hasattr(config, "lumi"):
        add_lumi_text(canvas, config.lumi)


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


def make_text(x, y, text, size=0.05, angle=0, font=42, color=ROOT.kBlack, ndc=True):
    t = ROOT.TLatex(x, y, text)
    ROOT.SetOwnership(t, False)
    t.SetTextSize(size)
    t.SetTextAngle(angle)
    t.SetTextFont(font)
    t.SetTextColor(color)
    t.SetNDC(ndc)
    return t


def add_lumi_text(canvas, lumi, pos={'x': 0.6, 'y': 0.79}, size=0.04, split_lumi_text=False):
    canvas.cd()
    text_lumi = '#scale[0.7]{#int}dt L = %.1f fb^{-1}' % (float(lumi))
    text_energy = '#sqrt{s} = 13 TeV'
    if split_lumi_text:
        label_lumi = make_text(x=pos['x'], y=pos['y'] - 0.05, text=text_energy, size=size)
        label_energy = make_text(x=pos['x'], y=pos['y'] - 0.05, text=text_energy, size=size)
        label_energy.Draw('sames')
    else:
        label_lumi = make_text(x=pos['x'], y=pos['y'], text=','.join([text_lumi, text_energy]), size=size)
    label_lumi.Draw('sames')
    canvas.Update()


def add_atlas_label(canvas, description='', pos={'x': 0.6, 'y': 0.87}, size=0.05, offset=0.125):
    label_atlas = make_text(x=pos['x'], y=pos['y'], text='ATLAS', size=size, font=72)
    label_descr = make_text(x=pos['x'] + offset, y=pos['y'], text=description, size=size, font=42)
    canvas.cd()
    label_atlas.Draw('sames')
    label_descr.Draw('sames')
    canvas.Update()


def set_title(self, hist, title, axis='x'):
    if title is None:
        return
    if type(hist) == dict:
        for h in hist.keys():
            hist[h] = __setTitle(hist[h], title, axis)
    else:
        if isinstance(hist, ROOT.TEfficiency):
            # hist.Draw('ap')
            ROOT.gPad.Update()
            graph = hist.GetPaintedGraph()
            self.__setTitle(graph, title, axis)
        else:
            hist = setTitle(hist, title, axis)


def add_text_to_canvas(canvas, text, pos={'x': 0.6, 'y': 0.79}, size=0.04, color=None):
    label = make_text(x=pos['x'], y=pos['y'], text=text, size=size, color=color)
    label.Draw('sames')
    canvas.Update()


def set_maximum(graph_obj, maximum, axis='y'):
    _logger.debug("Set maximum for %s to %f" % (graph_obj.GetName(), maximum))
    if axis == 'y':
        set_maximum_y(graph_obj, maximum)
    elif axis == 'x':
        set_maximum_x(graph_obj, maximum)


def set_maximum_y(graph_obj, maximum):
    minimum = get_min_y(graph_obj)
    set_range_y(graph_obj, minimum, maximum)


def set_maximum_x(graph_obj, maximum):
    graph_obj.GetXaxis().SetRangeUser(0, maximum)


def set_minimum(graph_obj, minimum, axis='y'):
    _logger.debug("Set minimum for %s to %f" % (graph_obj.GetName(), minimum))
    if axis == 'y':
        set_minimum_y(graph_obj, minimum)
    elif axis == 'x':
        graph_obj.GetXaxis().SetRangeUser(minimum, graph_obj.GetXaxis().GetXmax())


def set_minimum_y(graph_obj, minimum):
    maximum = get_max_y(graph_obj)
    set_range_y(graph_obj, minimum, maximum)


def set_range_y(graph_obj, minimum, maximum):
    if isinstance(graph_obj, ROOT.THStack):
        graph_obj.SetMinimum(minimum)
        graph_obj.SetMaximum(maximum)
    elif isinstance(graph_obj, ROOT.TH1):
        graph_obj.GetYaxis().SetRangeUser(minimum, maximum)
    elif isinstance(graph_obj, ROOT.TEfficiency):
        graph_obj.GetPaintedGraph().GetYaxis().SetRangeUser(minimum, maximum)


def get_min_y(graph_obj):
    if isinstance(graph_obj, ROOT.TH1):
        return graph_obj.GetMinimum()
    if isinstance(graph_obj, ROOT.TEfficiency):
        return graph_obj.GetPaintedGraph().GetMinimum()
    return None


def get_max_y(graph_obj):
    if isinstance(graph_obj, ROOT.TH1):
        return graph_obj.GetMaximum()
    if isinstance(graph_obj, ROOT.TEfficiency):
        return graph_obj.GetPaintedGraph().GetMaximum()
    return None


def set_range(graph_obj, minimum=None, maximum=None, axis='y'):
    if minimum is None:
        set_maximum(graph_obj, maximum, axis)
        return
    if maximum is None:
        set_minimum(graph_obj, minimum, axis)
        return
    set_range_y(graph_obj, minimum, maximum)


def add_legend_to_canvas(canvas, xl=0.6, yl=0.7, xh=0.9, yh=0.9, **kwargs):
    def convert_draw_option():
        draw_option = plot_obj.GetDrawOption()
        legend_option = ""
        if "hist" in draw_option.lower():
            if plot_obj.GetFillStyle() == 1001:
                legend_option += "L"
            else:
                legend_option += "F"
        if "l" in draw_option:
            legend_option += "L"
        if "p" in draw_option:
            legend_option += "P"
        if not legend_option:
            _logger.error("Unable to parse legend option from " % draw_option)
        return legend_option

    legend = ROOT.TLegend(xl, yl, xh, yh)
    ROOT.SetOwnership(legend, False)
    plot_objects = get_objects_from_canvas_by_type(canvas, "TH1F")
    for plot_obj in plot_objects:
        if "process_configs" in kwargs:
            label = kwargs["process_configs"][plot_obj.GetName().split("_")[-1]].label
        if "labels" in kwargs:
            label = kwargs["labels"][plot_objects.index(plot_obj)]
        legend.AddEntry(plot_obj, label, convert_draw_option())
    canvas.cd()
    legend.Draw("sames")
    canvas.Update()
