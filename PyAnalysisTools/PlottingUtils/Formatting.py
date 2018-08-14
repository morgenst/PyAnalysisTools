import re
import traceback

import ROOT
import os
from PyAnalysisTools.base import InvalidInputError, _logger
from PyAnalysisTools.ROOTUtils.ObjectHandle import get_objects_from_canvas_by_type, get_objects_from_canvas_by_name
from PyAnalysisTools.PlottingUtils.PlotConfig import get_style_setters_and_values, find_process_config


def load_atlas_style():
    try:
        base_path = os.path.dirname(os.path.join(os.path.realpath(__file__)))
        ROOT.gROOT.LoadMacro(os.path.join(base_path, 'AtlasStyle/AtlasStyle.C'))
        ROOT.SetAtlasStyle()
    except Exception as e:
        print traceback.print_exc()
        _logger.error("Could not find Atlas style files in %s" % os.path.join(base_path, 'AtlasStyle'))


def apply_style(obj, plot_config, process_config, index=None):
    style_setter, style_attr, color = get_style_setters_and_values(plot_config, process_config, index)
    if style_attr is not None:
        for setter in style_setter:
            getattr(obj, "Set" + setter + "Style")(style_attr)
    if color is not None:
        for setter in style_setter:
            getattr(obj, "Set" + setter + "Color")(color)


def decorate_canvas(canvas, plot_config):
    """
    Canvas decoration for ATLAS label, luminosity, grid settings and additional texts

    :param canvas: input canvas
    :type canvas: TCanvas
    :param plot_config: config containing settings for decoration
    :type plot_config: PlotConfig
    :return: None
    :rtype: None
    """
    if hasattr(plot_config, "watermark"):
        add_atlas_label(canvas, plot_config.watermark, {"x": 0.15, "y": 0.96}, size=0.03, offset=0.08)
    if hasattr(plot_config, "lumi") and plot_config.lumi is not None and plot_config.lumi >= 0:
        add_lumi_text(canvas, plot_config.lumi, {"x": 0.2, "y": 0.9})
    if hasattr(plot_config, "grid") and plot_config.grid is True:
        canvas.SetGrid()
    if hasattr(plot_config, "decor_text"):
        add_text_to_canvas(canvas, plot_config.decor_text, {"x": 0.2, "y": 0.8})


def set_title_x(obj, title):
    if not hasattr(obj, "GetXaxis"):
        raise TypeError
    try:
        obj.GetXaxis().SetTitle(title)
    except ReferenceError:
        _logger.error("Nil object {:s}".format(obj.GetName()))


def set_title_y(obj, title):
    if not hasattr(obj, "GetYaxis"):
        raise TypeError
    try:
        obj.GetYaxis().SetTitle(title)
    except ReferenceError:
        _logger.error("Nil object {:s}".format(obj.GetName()))

def set_title_z(obj, title):
    if not hasattr(obj, "GetZaxis"):
        raise TypeError
    try:
        obj.GetZaxis().SetTitle(title)
    except ReferenceError:
        _logger.error("Nil object {:s}".format(obj.GetName()))

def set_title_x_offset(obj, offset):
    if not hasattr(obj, "GetXaxis"):
        raise TypeError
    try:
        obj.GetXaxis().SetTitleOffset(offset)
    except ReferenceError:
        _logger.error("Nil object {:s}".format(obj.GetName()))

def set_title_y_offset(obj, offset):
    if not hasattr(obj, "GetYaxis"):
        raise TypeError
    try:
        obj.GetYaxis().SetTitleOffset(offset)
    except ReferenceError:
        _logger.error("Nil object {:s}".format(obj.GetName()))

def set_title_z_offset(obj, offset):
    if not hasattr(obj, "GetZaxis"):
        raise TypeError
    try:
        obj.GetZaxis().SetTitleOffset(offset)
    except ReferenceError:
        _logger.error("Nil object {:s}".format(obj.GetName()))

def set_title_x_size(obj, size):
    if not hasattr(obj, "GetXaxis"):
        raise TypeError
    try:
        obj.GetXaxis().SetTitleSize(size)
    except ReferenceError:
        _logger.error("Nil object {:s}".format(obj.GetName()))

def set_title_y_size(obj, size):
    if not hasattr(obj, "GetYaxis"):
        raise TypeError
    try:
        obj.GetYaxis().SetTitleSize(size)
    except ReferenceError:
        _logger.error("Nil object {:s}".format(obj.GetName()))

def set_title_z_size(obj, size):
    if not hasattr(obj, "GetZaxis"):
        raise TypeError
    try:
        obj.GetZaxis().SetTitleSize(size)
    except ReferenceError:
        _logger.error("Nil object {:s}".format(obj.GetName()))

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
    if color is not None:
        t.SetTextColor(color)
    t.SetNDC(ndc)
    return t


def add_lumi_text(canvas, lumi, pos={'x': 0.6, 'y': 0.85}, size=0.04, split_lumi_text=False):
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
    if isinstance(graph_obj, ROOT.THStack):
        graph_obj.SetMaximum(maximum)
        return
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
    if isinstance(graph_obj, ROOT.THStack):
        graph_obj.SetMinimum(minimum)
        return
    maximum = get_max_y(graph_obj)
    set_range_y(graph_obj, minimum, maximum)


def set_range_y(graph_obj, minimum, maximum):
    if isinstance(graph_obj, ROOT.THStack):
        graph_obj.SetMinimum(minimum)
        graph_obj.SetMaximum(maximum)
    elif isinstance(graph_obj, ROOT.TH1) or isinstance(graph_obj, ROOT.TGraph):
        graph_obj.SetMaximum(maximum)
        graph_obj.GetYaxis().SetRangeUser(minimum, maximum)
    elif isinstance(graph_obj, ROOT.TEfficiency):
        graph_obj.GetPaintedGraph().GetYaxis().SetRangeUser(minimum, maximum)


def set_range_z(graph_obj, minimum, maximum):
    if isinstance(graph_obj, ROOT.TH1):
        graph_obj.SetMaximum(maximum)
        graph_obj.GetZaxis().SetRangeUser(minimum, maximum)


def set_range_x(graph_obj, minimum, maximum):
    if isinstance(graph_obj, ROOT.TEfficiency):
        graph_obj.GetPaintedGraph().GetXaxis().SetRangeUser(minimum, maximum)
    else:
        graph_obj.GetXaxis().SetRangeUser(minimum, maximum)


def get_min_y(graph_obj):
    if isinstance(graph_obj, ROOT.TH1) or isinstance(graph_obj, ROOT.THStack):
        return graph_obj.GetMinimum()
    if isinstance(graph_obj, ROOT.TEfficiency):
        return graph_obj.GetPaintedGraph().GetMinimum()
    return None


def get_max_y(graph_obj):
    if isinstance(graph_obj, ROOT.TH1) or isinstance(graph_obj, ROOT.THStack):
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
    if axis == "y":
        set_range_y(graph_obj, minimum, maximum)
    elif axis == "x":
        set_range_x(graph_obj, minimum, maximum)
    else:
        _logger.error("Invalid axis choice: {:s}".format(axis))


def auto_scale_y_axis(canvas, offset=1.1):
    graph_objects = get_objects_from_canvas_by_type(canvas, "TH1F")
    max_y = 1.1 * max([graph_obj.GetMaximum() for graph_obj in graph_objects])
    draw_options = [graph_obj.GetDrawOption() for graph_obj in graph_objects]
    first_index = draw_options.index(filter(lambda draw_option: draw_option.count("same") == 0)[0])
    first_graph_obj = graph_objects[first_index]
    set_maximum_y(first_graph_obj, max_y)
    canvas.Update()


def add_legend_to_canvas(canvas, **kwargs):
    kwargs.setdefault("xl", 0.7)
    kwargs.setdefault("yl", 0.6)
    kwargs.setdefault("xh", 0.9)
    kwargs.setdefault("yh", 0.9)
    kwargs.setdefault("format", None)
    kwargs.setdefault("columns", None)

    def convert_draw_option(process_config=None, plot_config=None):
        def parse_option_from_format():
            if kwargs["format"] == "line":
                return "L"
            elif kwargs["format"] == "marker":
                return "P"

        draw_option = plot_obj.GetDrawOption()
        if (draw_option is None or draw_option == "") and isinstance(plot_obj, ROOT.TF1):
            draw_option = ROOT.gROOT.GetFunction(plot_obj.GetName()).GetDrawOption()

        if is_stacked:
            draw_option = "Hist"
        legend_option = ""
        if "hist" in draw_option.lower():
            # if plot_obj.GetFillStyle() == 1001:
            #     legend_option += "L"
            # else:
            if process_config is not None and (hasattr(process_config, "format") or hasattr(plot_config, "format")) or kwargs["format"]:
                if process_config is not None and process_config.format.lower() == "line":
                    legend_option += "L"
                elif plot_config is not None and plot_config.format.lower() == "line":
                    legend_option += "L"
                elif kwargs["format"]:
                    legend_option += parse_option_from_format()
            else:
                legend_option += "F"
        if "l" in draw_option:
            legend_option += "L"
        if "p" in draw_option or "E" in draw_option:
            legend_option += "P"
        if re.match(r"e\d", draw_option.lower()):
            legend_option += "F"
        if not legend_option and kwargs["format"]:
            legend_option = parse_option_from_format()
        if not legend_option:
            _logger.error("Unable to parse legend option from {:s} for object {:s}".format(draw_option,
                                                                                           plot_obj.GetName()))
        return legend_option
    legend = ROOT.TLegend(kwargs["xl"], kwargs["yl"], kwargs["xh"], kwargs["yh"])
    ROOT.SetOwnership(legend, False)
    legend.SetTextSize(0.025)
    if kwargs["columns"]:
        legend.SetNColumns(kwargs["columns"])
    legend.SetFillStyle(0)
    labels = None
    stacks = []
    if "labels" in kwargs:
        labels = kwargs["labels"]
    if "labels" not in kwargs or not isinstance(kwargs["labels"], dict):
        if not "plot_objects" in kwargs:
            plot_objects = get_objects_from_canvas_by_type(canvas, "TH1F")
            plot_objects += get_objects_from_canvas_by_type(canvas, "TH1D")
            plot_objects += get_objects_from_canvas_by_type(canvas, "TF1")
            plot_objects += get_objects_from_canvas_by_type(canvas, "TGraph")
            #plot_objects += get_objects_from_canvas_by_type(canvas, "TProfile")
            stacks = get_objects_from_canvas_by_type(canvas, "THStack")
            plot_objects += get_objects_from_canvas_by_type(canvas, "TEfficiency")
        else:
            plot_objects = kwargs["plot_objects"]
    else:
        labels = {}
        plot_objects = []
        for hist_pattern, lab in kwargs["labels"].iteritems():
            plot_objects.append(get_objects_from_canvas_by_name(canvas, hist_pattern)[0])
            labels[get_objects_from_canvas_by_name(canvas, hist_pattern)[0].GetName()] = lab
    stacked_objects = None
    if len(stacks) is not 0:
        stacked_objects = stacks[0].GetHists()
        plot_objects += stacked_objects
    for plot_obj in plot_objects:
        label = None
        process_config = None
        if "stat.unc" in plot_obj.GetName() and plot_obj != plot_objects[-1]:
            plot_objects.append(plot_obj)
            continue
        if "process_configs" in kwargs and kwargs["process_configs"] is not None:
            try:
                process_config = find_process_config(plot_obj.GetName().split("_")[-1], kwargs["process_configs"])
                label = process_config.label
            except AttributeError:
                pass
        if "labels" is not None:
            if isinstance(labels, list):
                label = labels[plot_objects.index(plot_obj)]
            if isinstance(labels, dict):
                if plot_obj.GetName() in labels:
                    label = labels[plot_obj.GetName()]
        is_stacked = False
        if stacked_objects and plot_obj in stacked_objects:
            is_stacked = True
        if label is None:
            continue
        plot_config = kwargs["plot_config"] if "plot_config" in kwargs else None
        if not "format" in kwargs or not isinstance(kwargs["format"], list):
            legend.AddEntry(plot_obj, label, convert_draw_option(process_config, plot_config))
        else:
            legend.AddEntry(plot_obj, label, kwargs["format"][plot_objects.index(plot_obj)])
    canvas.cd()
    if "fill_style" in kwargs:
        print "yes, got fill style"
        legend.SetFillStyle(kwargs["fill_style"])
    legend.SetBorderSize(0)
    legend.Draw("sames")
    canvas.Update()

def format_canvas(canvas, **kwargs):
    if "margin" in kwargs:
        for side, margin in kwargs["margin"].iteritems():
            getattr(canvas, "Set{:s}Margin".format(side.capitalize()))(margin)
    canvas.Modified()
    canvas.Update()
    return canvas

