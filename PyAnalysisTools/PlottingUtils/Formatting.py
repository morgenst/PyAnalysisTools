import re
import ROOT
import numpy as np
import os
from PyAnalysisTools.base import InvalidInputError, _logger
from PyAnalysisTools.ROOTUtils.ObjectHandle import get_objects_from_canvas_by_type
from PyAnalysisTools.PlottingUtils.PlotConfig import get_style_setters_and_values

def load_atlas_style():
    try:
        base_path = os.path.dirname(os.path.join(os.path.realpath(__file__)))
        ROOT.gROOT.LoadMacro(os.path.join(base_path, 'AtlasStyle/AtlasStyle.C'))
        ROOT.SetAtlasStyle()
    except Exception as e:
        _logger.error("Could not find Atlas style files in %s" % os.path.join(base_path, 'AtlasStyle'))


def apply_style(obj, plot_config, process_config):
    style_setter, style_attr, color = get_style_setters_and_values(plot_config, process_config)
    if style_attr is not None:
        getattr(obj, "Set" + style_setter + "Style")(style_attr)
    if color is not None:
        getattr(obj, "Set" + style_setter + "Color")(color)


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


def add_stat_box_to_canvas(canvas):
    def retrieve_stat_box(hist):
        ctmp = ROOT.TCanvas("c_tmp", "c_tmp")
        ctmp.cd()
        ROOT.gStyle.SetOptStat(111111)
        hist.SetStats(1)
        hist.Draw()
        ROOT.gPad.Update()
        stat_box = hist.FindObject("stats").Clone()
        ROOT.SetOwnership(stat_box, False)
        ROOT.gStyle.SetOptStat(0)
        return stat_box

    hists = get_objects_from_canvas_by_type(canvas, "TH1F")

    stat_boxes = [retrieve_stat_box(hist) for hist in hists]
    canvas.cd()
    height = min(0.15, 1. / len(stat_boxes))
    offset = 0.01
    for stat_box in stat_boxes:
        index = stat_boxes.index(stat_box)
        color = hists[index].GetLineColor()
        stat_box.SetTextColor(color)
        stat_box.SetY1NDC(1. - (index + 1.) * height)
        stat_box.SetY2NDC(1. - index * (height + offset))
        stat_box.Draw("sames")
    canvas.Update()


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


def auto_scale_y_axis(canvas, offset=1.1):
    graph_objects = get_objects_from_canvas_by_type(canvas, "TH1F")
    max_y = 1.1 * max([graph_obj.GetMaximum() for graph_obj in graph_objects])
    draw_options = [graph_objects.GetDrawOption() for graph_obj in graph_objects]
    first_index = draw_options.index(filter(lambda draw_option: draw_option.count("same") == 0)[0])
    first_graph_obj = graph_objects[first_index]
    set_maximum_y(first_graph_obj, max_y)
    canvas.Update()


def add_legend_to_canvas(canvas, **kwargs):
    kwargs.setdefault("xl", 0.6)
    kwargs.setdefault("yl", 0.7)
    kwargs.setdefault("xh", 0.9)
    kwargs.setdefault("yh", 0.9)
    def convert_draw_option():
        draw_option = plot_obj.GetDrawOption()
        if is_stacked:
            draw_option = "Hist"
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
    legend = ROOT.TLegend(kwargs["xl"], kwargs["yl"], kwargs["xh"], kwargs["yh"])
    ROOT.SetOwnership(legend, False)
    plot_objects = get_objects_from_canvas_by_type(canvas, "TH1F")
    stack = get_objects_from_canvas_by_type(canvas, "THStack")
    stacked_objects = None
    if stack is not None:
        stacked_objects = stack[0].GetHists()
        plot_objects += stacked_objects
    for plot_obj in plot_objects:
        if "process_configs" in kwargs:
            label = kwargs["process_configs"][plot_obj.GetName().split("_")[-1]].label
        if "labels" in kwargs:
            label = kwargs["labels"][plot_objects.index(plot_obj)]
        is_stacked = False
        if stacked_objects and plot_obj in stacked_objects:
            is_stacked = True
        legend.AddEntry(plot_obj, label, convert_draw_option())
    canvas.cd()
    legend.Draw("sames")
    canvas.Update()