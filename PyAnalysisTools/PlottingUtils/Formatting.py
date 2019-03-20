import re
import traceback

import ROOT
import os
from PyAnalysisTools.base import InvalidInputError, _logger
from PyAnalysisTools.ROOTUtils.ObjectHandle import get_objects_from_canvas_by_type, get_objects_from_canvas_by_name
from PyAnalysisTools.PlottingUtils.PlotConfig import get_style_setters_and_values, find_process_config
from PyAnalysisTools.PlottingUtils.PlotableObject import PlotableObject


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


def apply_style_plotableObject(plotable_object):
    plotable_object.plot_object.SetMarkerColor(plotable_object.marker_color)
    plotable_object.plot_object.SetMarkerSize(plotable_object.marker_size)
    plotable_object.plot_object.SetMarkerStyle(plotable_object.marker_style)
    plotable_object.plot_object.SetLineColor(plotable_object.line_color)
    plotable_object.plot_object.SetLineWidth(plotable_object.line_width)
    plotable_object.plot_object.SetLineStyle(plotable_object.line_style)
    plotable_object.plot_object.SetFillColor(plotable_object.fill_color)
    plotable_object.plot_object.SetFillStyle(plotable_object.fill_style)


def decorate_canvas(canvas, plot_config, **kwargs):
    """
    Canvas decoration for ATLAS label, luminosity, grid settings and additional texts

    :param canvas: input canvas
    :type canvas: TCanvas
    :param plot_config: config containing settings for decoration
    :type plot_config: PlotConfig
    :return: None
    :rtype: None
    """

    if plot_config.ratio is not None and plot_config.ratio is not False:
        kwargs.setdefault('watermark_x', plot_config.watermark_x_ratio)
        kwargs.setdefault('watermark_y', plot_config.watermark_y_ratio)
        kwargs.setdefault('watermark_size', plot_config.watermark_size_ratio)
        kwargs.setdefault('watermark_offset', plot_config.watermark_offset_ratio)
        kwargs.setdefault('lumi_text_x', plot_config.lumi_text_x_ratio)
        kwargs.setdefault('lumi_text_y', plot_config.lumi_text_y_ratio)
        kwargs.setdefault('lumi_text_size', plot_config.lumi_text_size_ratio)
    else:
        kwargs.setdefault('watermark_x', plot_config.watermark_x)
        kwargs.setdefault('watermark_y', plot_config.watermark_y)
        kwargs.setdefault('watermark_size', plot_config.watermark_size)
        kwargs.setdefault('watermark_offset', plot_config.watermark_offset)
        kwargs.setdefault('lumi_text_x', plot_config.lumi_text_x)
        kwargs.setdefault('lumi_text_y', plot_config.lumi_text_y)
        kwargs.setdefault('lumi_text_size', plot_config.lumi_text_size)

    kwargs.setdefault('decor_text_x', plot_config.decor_text_x)
    kwargs.setdefault('decor_text_y', plot_config.decor_text_y)
    kwargs.setdefault('decor_text_size', plot_config.decor_text_size)
    kwargs.setdefault('lumi_text', plot_config.lumi_text)
    kwargs.setdefault('lumi_precision', plot_config.lumi_precision)
    kwargs.setdefault('add_text', plot_config.add_text)

    if plot_config.watermark is not None:
        add_atlas_label(canvas, plot_config.watermark, {"x": kwargs['watermark_x'],
                                                        "y": kwargs['watermark_y']},
                        size=kwargs['watermark_size'], offset=kwargs['watermark_offset'])
    if plot_config.get_lumi() is not None and plot_config.get_lumi() >= 0 or kwargs['lumi_text'] is not None:
        add_lumi_text(canvas, plot_config.get_lumi(), {"x": kwargs['lumi_text_x'], "y": kwargs['lumi_text_y']},
                      size=kwargs['lumi_text_size'], lumi_text=kwargs['lumi_text'], precision=kwargs['lumi_precision'])

    if plot_config.grid:
        canvas.SetGrid()

    if plot_config.decor_text is not None:
        add_text_to_canvas(canvas, plot_config.decor_text, {"x": kwargs['decor_text_x'], "y": kwargs['decor_text_y']},
                           size=kwargs['decor_text_size'])

    if plot_config.add_text is not None:
        add_text_to_canvas(canvas, kwargs['add_text'][0], {'x': kwargs['add_text'][1], 'y': kwargs['add_text'][2]}, size=kwargs['add_text'][3])


def check_valid_axis(axis):
    if axis.lower() not in ['x', 'y', 'z']:
        _logger.error('Request axis setting for {:s} which is not supported'.format(str(axis)))
        raise InvalidInputError('Request axis setting for {:s} which is not supported'.format(str(axis)))


def set_axis_title(obj, title, axis):
    check_valid_axis(axis)
    if title is None:
        return
    if not hasattr(obj, "Get{:s}axis".format(axis.capitalize())):
        raise TypeError
    try:
        getattr(obj, 'Get{:s}axis'.format(axis.capitalize()))().SetTitle(title)
    except ReferenceError:
        _logger.error("Nil object {:s}".format(obj.GetName()))


def set_axis_title_offset(obj, offset, axis):
    check_valid_axis(axis)
    if not hasattr(obj, "Get{:s}axis".format(axis.capitalize())):
        raise TypeError
    try:
        getattr(obj, 'Get{:s}axis'.format(axis.capitalize()))().SetTitleOffset(offset)
    except ReferenceError:
        _logger.error("Nil object {:s}".format(obj.GetName()))


def set_axis_title_size(obj, size, axis):
    check_valid_axis(axis)
    if not hasattr(obj, "Get{:s}axis".format(axis.capitalize())):
        raise TypeError
    try:
        getattr(obj, 'Get{:s}axis'.format(axis.capitalize()))().SetTitleSize(size)
    except ReferenceError:
        _logger.error("Nil object {:s}".format(obj.GetName()))


def set_title_x(obj, title):
    set_axis_title(obj, title, 'x')


def set_title_y(obj, title):
    set_axis_title(obj, title, 'y')


def set_title_z(obj, title):
    set_axis_title(obj, title, 'z')


def set_title_x_offset(obj, offset):
    set_axis_title_offset(obj, offset, 'x')


def set_title_y_offset(obj, offset):
    set_axis_title_offset(obj, offset, 'y')


def set_title_z_offset(obj, offset):
    set_axis_title_offset(obj, offset, 'z')


def set_title_x_size(obj, size):
    """
    Set x-axis title size for plotable object
    :param obj: plot object
    :type obj: ROOT.TH1, ROOT.TGraph, etc
    :param size: font size
    :type size: int
    :return: nothing
    :rtype: None
    """
    set_axis_title_size(obj, size, 'x')


def set_title_y_size(obj, size):
    """
    Set y-axis title size for plotable object
    :param obj: plot object
    :type obj: ROOT.TH1, ROOT.TGraph, etc
    :param size: font size
    :type size: int
    :return: nothing
    :rtype: None
    """
    set_axis_title_size(obj, size, 'y')


def set_title_z_size(obj, size):
    """
    Set z-axis title size for plotable object
    :param obj: plot object
    :type obj: ROOT.TH1, ROOT.TGraph, etc
    :param size: font size
    :type size: int
    :return: nothing
    :rtype: None
    """
    set_axis_title_size(obj, size, 'z')


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


def add_lumi_text(canvas, lumi, pos={'x': 0.6, 'y': 0.87}, size=0.04, split_lumi_text=False, energy=13, precision=2,
                  lumi_text=None):
    canvas.cd()

    if lumi_text:
        text_lumi = lumi_text
        text_energy = ''
    else:
        text_lumi = '#scale[0.7]{{#int}}dt L = {:.{:d}f} fb^{{-1}}'.format(float(lumi), precision)
        text_energy = '#sqrt{{s}} = {:d} TeV'.format(energy)

    #     text_lumi = '#scale[0.7]{#int}dt L = %.2f fb^{-1},' % (float(lumi))
    #     text_energy = '#sqrt{s} = 13 TeV'

    if split_lumi_text:
        label_lumi = make_text(x=pos['x'], y=pos['y'], text=text_lumi.rstrip(','), size=size)
        label_energy = make_text(x=pos['x'], y=pos['y'] - 0.05, text=text_energy, size=size)
        label_energy.Draw('sames')
    else:
        label_lumi = make_text(x=pos['x'], y=pos['y'], text=' '.join([text_lumi, text_energy]), size=size)
    label_lumi.Draw('sames')
    canvas.Update()


def add_atlas_label(canvas, description='', pos={'x': 0.6, 'y': 0.87}, size=0.05, offset=0.05):
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
        hist_name = hist.GetName()
        hist.SetName('hist')
        # ROOT.gStyle.SetOptStat(111111)
        ROOT.gStyle.SetOptStat('emruo')
        hist.SetStats(1)
        hist.Draw()
        ROOT.gPad.Update()
        stat_box = hist.FindObject("stats").Clone()
        ROOT.SetOwnership(stat_box, False)
        ROOT.gStyle.SetOptStat(0)
        hist.SetName(hist_name)
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
        stat_box.SetY1NDC(1. - (index + 1.) * height - 0.2)
        stat_box.SetY2NDC(1. - index * (height + offset) - 0.2)
        stat_box.SetX1NDC(0.7)
        stat_box.SetX2NDC(0.9)
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
        if not isinstance(graph_obj, ROOT.TH2):
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
        if isinstance(graph_obj, ROOT.TH2):
            return graph_obj.GetYaxis().GetXmax()
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


def get_legend(lines, max_length_label, **kwargs):
    columns = kwargs['columns']
    position = kwargs['position']
    if kwargs['ratio'] is True:
        scale = 0.0625
        if lines > 9 or max_length_label > (25 / columns) or columns > 3:
            if lines > 9 or max_length_label > (30 / columns) or columns > 3:
                text_size = 0.04
                x_min = 0.55
            else:
                text_size = 0.05
                x_min = 0.55
        else:
            text_size = 0.05
            x_min = 0.6
    else:
        scale = 0.04687
        if lines > 9 or max_length_label > (25 / columns) or columns > 3:
            if lines > 9 or max_length_label > (30 / columns) or columns > 3:
                text_size = 0.03
                x_min = 0.55
            else:
                text_size = 0.03759
                x_min = 0.55
        else:
            text_size = 0.03759
            x_min = 0.6
    if position == 0 or position == 'c':
        leg_x = (max(0.3, 0.5 - columns * 0.1), min(0.7, 0.5 + columns * 0.1))
        leg_y = (max(0.3, 0.5 - lines * scale / 2), min(0.7, 0.5 + lines * scale / 2))
    elif position == 1 or position == 'ur':
        if max_length_label > (25 / columns) or columns > 2:
            leg_x = (x_min, 0.72)
        else:
            leg_x = (max(x_min, 0.72 - columns * 0.1), min(0.92, 0.72 + columns * 0.1))
        leg_y = (max(0.6, 0.92 - lines * scale), min(0.92, 0.92))
        # leg_y = (max(0.52, 0.72 - lines * 0.0625 / 2), min(0.92, 0.72 + lines * 0.0625 / 2))
    elif position == 2 or position == 'ul':  # upper left
        leg_x = (max(0.08, 0.28 - columns * 0.1), min(0.48, 0.28 + columns * 0.1))
        leg_y = (max(0.52, 0.72 - lines * scale / 2), min(0.92, 0.72 + lines * scale / 2))
    elif position == 3 or position == 'll':
        leg_x = (max(0.08, 0.28 - columns * 0.1), min(0.48, 0.28 + columns * 0.1))
        leg_y = (max(0.08, 0.28 - lines * scale / 2), min(0.48, 0.28 + lines * scale / 2))
    elif position == 4 or position == 'lr':
        leg_x = (max(0.52, 0.72 - columns * 0.1), min(0.92, 0.72 + columns * 0.1))
        leg_y = (max(0.08, 0.28 - lines * scale / 2), min(0.48, 0.28 + lines * scale / 2))
    else:
        leg_x = (kwargs["xl"], kwargs["xh"])
        leg_y = (kwargs["yl"], kwargs["yh"])
        # leg_x = (position[0], position[2])
        # leg_y = (position[1], position[3])
    leg = ROOT.TLegend(leg_x[0], leg_y[0], leg_x[1], leg_y[1])
    leg.SetNColumns(columns)
    leg.SetMargin(0.2)
    # leg.SetMargin(min(0.2,0.1*max(columns,lines)))
    leg.SetLineColor(0)
    leg.SetLineStyle(0)
    leg.SetFillStyle(0)
    leg.SetFillColorAlpha(0, 0)
    leg.SetBorderSize(0)
    leg.SetTextSize(text_size)
    leg.SetTextFont(42)
    ROOT.SetOwnership(leg, False)
    return leg


def add_legend_to_canvas(canvas, **kwargs):
    # kwargs.setdefault("xl", 0.52)
    # kwargs.setdefault("yl", 0.52)
    # kwargs.setdefault("xh", 0.92)
    # kwargs.setdefault("yh", 0.92)
    # default MMM
    # kwargs.setdefault("xl", 0.7)
    # kwargs.setdefault("yl", 0.6)
    # kwargs.setdefault("xh", 0.9)
    # kwargs.setdefault("yh", 0.9)
    kwargs.setdefault("xl", 0.55)
    kwargs.setdefault("yl", 0.55)
    kwargs.setdefault("xh", 0.95)
    kwargs.setdefault("yh", 0.9)
    kwargs.setdefault("format", None)
    # kwargs.setdefault("columns", None)
    # kwargs.setdefault('text_size', 0.025)
    kwargs.setdefault("columns", 1)
    kwargs.setdefault("position", 'ur')
    kwargs.setdefault("fill_style", 0)
    kwargs.setdefault("process_configs", 0)
    kwargs.setdefault('ratio', False)
    kwargs.setdefault('plot_config', None)

    def convert_draw_option(process_config=None, plot_config=None):
        def parse_option_from_format():
            """
            Converts format option string to ROOT compatible string
            :return: ROOT format string
            :rtype: str
            """
            if kwargs["format"].lower() == "line":
                return "L"
            elif kwargs["format"].lower() == "marker":
                return "P"

        draw_option = plot_obj.GetDrawOption()
        if (draw_option is None or draw_option == "") and isinstance(plot_obj, ROOT.TF1):
            draw_option = ROOT.gROOT.GetFunction(plot_obj.GetName()).GetDrawOption()

        if is_stacked:
            draw_option = "Hist"
        legend_option = ""
        if "hist" in draw_option.lower():
            if process_config is not None and (hasattr(process_config, "format") or hasattr(plot_config, "format")) or \
                    kwargs["format"]:
                if process_config is not None and process_config.format.lower() == "line":
                    legend_option += "L"
                elif plot_config is not None and plot_config.format.lower() == "line":
                    legend_option += "L"
                elif kwargs["format"]:
                    legend_option += parse_option_from_format()
            else:
                legend_option += "F"
        if "l" in draw_option.lower():
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

    labels = None
    stacks = []
    is_stacked = False
    if "labels" in kwargs:
        labels = kwargs["labels"]
    if "labels" not in kwargs or not isinstance(kwargs["labels"], dict):
        if not "plot_objects" in kwargs:
            plot_objects = get_objects_from_canvas_by_type(canvas, "TH1F")
            plot_objects += get_objects_from_canvas_by_type(canvas, "TH1D")
            plot_objects += get_objects_from_canvas_by_type(canvas, "TF1")
            plot_objects += get_objects_from_canvas_by_type(canvas, "TGraph")
            # plot_objects += get_objects_from_canvas_by_type(canvas, "TProfile")
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

    if labels is None and kwargs["process_configs"] is not None:
        formats = []
        for plot_obj in plot_objects:
            label = None
            is_stacked = False
            if stacked_objects and plot_obj in stacked_objects:
                is_stacked = True
            try:
                process_config = find_process_config(plot_obj.GetName().split("_")[-1], kwargs["process_configs"])
                if process_config is None:
                    process_config = filter(lambda pn: pn[0] in plot_obj.GetName(), kwargs['process_configs'].iteritems())[0][1]
                label = process_config.label
                formats.append(convert_draw_option(process_config, kwargs['plot_config']))
            except AttributeError:
                print 'Could not find process label for ', plot_obj.GetName().split("_")[-1]
                pass
            if label is None:
                continue
            if labels is None:
                labels = []
            labels.append(label)
        if kwargs['format'] is None:
            kwargs['format'] = formats
    if isinstance(labels, list):
        legend = get_legend(len(plot_objects), len(max(labels, key=len)), **kwargs)
    else:
        #TODO: For what is this needed?
        legend = get_legend(len(plot_objects), labels, **kwargs)

    plot_config = kwargs["plot_config"] if "plot_config" in kwargs else None
    canvas.cd()
    legend.SetFillStyle(kwargs["fill_style"])
    for plot_obj, label in zip(plot_objects, labels):
        if 'format' not in kwargs or not isinstance(kwargs["format"], list):
            legend.AddEntry(plot_obj, label, convert_draw_option(None, plot_config))
        else:
            legend.AddEntry(plot_obj, label, kwargs["format"][plot_objects.index(plot_obj)])
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
