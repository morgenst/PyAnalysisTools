__author__ = 'marcusmorgenstern'
__mail__ = ''

import ROOT
import numpy as np
from PyAnalysisTools.base import InvalidInputError, _logger
from PyAnalysisTools.PlottingUtils import Formatting as FM
from PyAnalysisTools.ROOTUtils import ObjectHandle as object_handle
from functools import partial


def retrieve_new_canvas(name, title):
    return ROOT.TCanvas(name, title, 800, 600)


def plot_hist(hist,
              plot_options=None,
              draw_options=None,
              canvas_name='c',
              canvas_title=''):
    #ROOT.SetOwnership(hist, False)

    if plot_options is not None:
        plot_options.configure(hist)
    canvas = retrieve_new_canvas(canvas_name, canvas_title)
    canvas.cd()
    if draw_options is None:
        draw_options = 'hist'
    hist.SetFillColor(ROOT.kRed)
    hist.Draw(draw_options)
    return canvas, hist


# todo: memoise
def fetch_process_config(process, process_config):
    if process not in process_config:
        _logger.warning("Could not find process %s in process config" % process)
        return None
    return process_config[process]


def plot_histograms(hist_dict, plot_config, common_config, process_configs):
    canvas = retrieve_new_canvas(plot_config.name, "")
    canvas.cd()
    is_first = True
    for process, hist in hist_dict.iteritems():
        process_config = fetch_process_config(process, process_configs)
        draw_option = "Hist"
        style_attr, color = None, None
        if hasattr(process_config, "draw"):
            draw_option = process_config.draw
        if hasattr(process_config, "style"):
            style_attr = process_config.style
        if hasattr(process_config, "color"):
            color = process_config.color
            if isinstance(color, str):
                color = getattr(ROOT, color)
        style_setter = None
        if draw_option == "Hist":
            style_setter = "Fill"
        elif draw_option == "Marker":
            style_setter = "Marker"
            draw_option = "p"
        elif draw_option == "Line":
            style_setter = "Line"
            draw_option = "l"
        if not is_first:
            draw_option += " sames"
        #todo: refactoring of configs to be moved to plotHist
        hist.Draw(draw_option)
        if style_attr is not None:
            getattr(hist, "Set"+style_setter+"Style")(style_attr)
        if color is not None:
            getattr(hist, "Set" + style_setter + "Color")(color)
        is_first = False
    return canvas


def add_histogram_to_canvas(canvas, hist, plot_option=None, draw_option=None):
    canvas.cd()
    if "same" not in draw_option:
        draw_option += "sames"
    hist.Draw(draw_option)


def plot_graph(graph, plot_options=None, draw_options=None, **kwargs):
    kwargs.setdefault("canvas_name", graph.GetName())
    kwargs.setdefault("canvas_title", "")
    canvas = retrieve_new_canvas(kwargs["canvas_name"], kwargs["canvas_title"])
    canvas.cd()
    if draw_options is None:
        draw_options = 'ap'
    graph.Draw(draw_options)
    return canvas


def add_graph_to_canvas(canvas, graph, plot_options=None, draw_options=None):
    canvas.cd()
    #if plot_options is not None:
    #    plot_options.configure(graph)
    if draw_options is None:
        draw_options = 'psame'
    if not draw_options.endswith('same'):
        draw_options += 'same'
    graph.Draw(draw_options)


def plot_stack(histograms, plot_options=None, draw_options=None, canvas_name='name', canvas_title='',
               ordering=None, y_minimum=None, y_maximum=None):

    if plot_options is not None:
        if not len(histograms) == len(plot_options):
            raise InvalidInputError("No of histograms does not match to no of provided plot_options")
        for key in histograms.keys():
            plot_options[key].configure(histograms[key])

    canvas = retrieve_new_canvas(canvas_name, canvas_title)
    canvas.cd()
    if draw_options is None:
        draw_options = {}
        for key in histograms.keys():
            draw_options[key] = 'hist'
    stack = ROOT.THStack('', '')
    FM.set_range_y(stack, y_minimum, y_maximum)
    x_title = None
    y_title = None
    if ordering is None:
        ordering = histograms.keys()
    for key in reversed(ordering):
        try:
            stack.Add(histograms[key], draw_options[key])
        except KeyError:
            _logger.debug('Could not add %s to stack' % (key))
        try:
            if x_title is None:
                x_title = histograms[key].GetXaxis().GetTitle()
                y_title = histograms[key].GetYaxis().GetTitle()
        except KeyError:
            continue

    stack.Draw()
    FM.set_title_x(stack, title=x_title, axis='x')
    FM.set_title_y(stack, title=y_title, axis='y')
    return canvas


def add_data_to_stack(canvas, data, blind=None):
    if blind:
        blind_data(data, blind)
    canvas.cd()
    data.Draw("psames")


def blind_data(data, blind):
    for b in range(data.GetNbinsX() + 1):
        if data.GetBinCenter(b) < blind:
            continue
        else:
            data.SetBinContent(b, 0.)
            data.SetBinError(b, 0.)


def add_signal_to_stack(canvas, signal, signal_strength=1., overlay=False, stack=None):
    if overlay:
        if not stack:
            raise InvalidInputError("Requested overlay of signal, but no stack provided.")
        clone = None
        for h in stack.GetHists():
            if clone is None:
                clone = h.Clone()
            else:
                clone.Add(h)
    canvas.cd()
    for process in signal:
        process.SetLineColor(ROOT.kRed)
        process.Scale(signal_strength)
        if overlay:
            process.Add(clone)
        process.Draw("histsames")


def add_hist_to_canvas(canvas, hist, plot_options=None, draw_options=None):
    canvas.cd()
    if plot_options is not None:
        plot_options.configure(hist)
    if draw_options is None:
        draw_options = 'hist'
    hist.Draw(draw_options + 'sames')


def add_ratio_to_canvas(canvas, ratio, y_min=None, y_max=None, y_title=None, name=None, title=''):
    def scale_frame_text(fr, scale):
        x_axis = fr.GetXaxis()
        y_axis = fr.GetYaxis()
        y_axis.SetTitleSize(y_axis.GetTitleSize() * scale)
        y_axis.SetLabelSize(y_axis.GetLabelSize() * scale)
        y_axis.SetTitleOffset(1.1*y_axis.GetTitleOffset() / scale)
        y_axis.SetLabelOffset(y_axis.GetLabelOffset() * scale)
        x_axis.SetTitleSize(x_axis.GetTitleSize() * scale)
        x_axis.SetLabelSize(x_axis.GetLabelSize() * scale)
        x_axis.SetTickLength(x_axis.GetTickLength() * scale)
        x_axis.SetTitleOffset(2.5*x_axis.GetTitleOffset() / scale)
        x_axis.SetLabelOffset(2.5*x_axis.GetLabelOffset() / scale)

    def reset_frame_text(fr):
        x_axis = fr.GetXaxis()
        y_axis = fr.GetYaxis()
        gs = ROOT.gStyle
        y_axis.SetTitleSize(gs.GetTitleSize('Y'))
        y_axis.SetLabelSize(gs.GetLabelSize('Y'))
        y_axis.SetTitleOffset(gs.GetTitleOffset('Y'))
        y_axis.SetLabelOffset(gs.GetLabelOffset('Y'))
        x_axis.SetTitleSize(gs.GetTitleSize('X'))
        x_axis.SetLabelSize(gs.GetLabelSize('X'))
        x_axis.SetTickLength(gs.GetTickLength('X'))

    if not canvas or not ratio:
        raise InvalidInputError("Either canvas or ratio not provided.")
    y_frac = 0.25
    try:
        hratio = object_handle.get_objects_from_canvas_by_type(ratio, "TH1F")[0]
    except IndexError:
        try:
            hratio = object_handle.get_objects_from_canvas_by_type(ratio, "TH1D")[0]
        except IndexError:
            hratio = object_handle.get_objects_from_canvas_by_type(ratio, "TGraph")[0]
    if name is None:
        name = canvas.GetName() + "_ratio"
    c = retrieve_new_canvas(name, title)
    #ROOT.SetOwnership(c, False)
    c.Draw()
    pad1 = ROOT.TPad("pad1", "top pad", 0.0, y_frac, 1., 1.)
    pad1.SetBottomMargin(0.05)
    pad1.Draw()
    pad2 = ROOT.TPad("pad2", "bottom pad", 0, 0., 1, ((1 - y_frac) * canvas.GetBottomMargin() / y_frac + 1) * y_frac)
    pad2.SetTopMargin(0.5)
    pad2.SetBottomMargin(0.1)
    pad2.Draw()
    pad1.cd()
    try:
        stack = object_handle.get_objects_from_canvas_by_type(canvas, "THStack")[0]
        stack.GetXaxis().SetTitleSize(0)
        stack.GetXaxis().SetLabelSize(0)
        scale = 1. / (1. - y_frac)
        scale_frame_text(stack, scale)
    except IndexError:
        try:
            stack = object_handle.get_objects_from_canvas_by_type(canvas, "TEfficiency")[0]
        except IndexError:
            stack = object_handle.get_objects_from_canvas_by_type(canvas, "TH1")[0]
    canvas.DrawClonePad()
    pad2.cd()
    hratio.GetYaxis().SetNdivisions(505)
    hratio.GetXaxis().SetNdivisions(505)
    scale = 1. / y_frac - 1.5
    reset_frame_text(hratio)
    scale_frame_text(hratio, scale)
    ratio.Update()
    ratio.SetBottomMargin(0.4)
    ratio.DrawClonePad()
    xlow = pad2.GetUxmin()
    xup = pad2.GetUxmax()
    line = ROOT.TLine(xlow, 1, xup, 1)
    #ROOT.SetOwnership(line, False)
    line.Draw('same')
    pad2.Update()
    c.line = line
    pad = c.cd(1)
    if y_min is not None or y_max is not None:
        isFirst = True
        efficiency_obj = object_handle.get_objects_from_canvas_by_type(pad1, "TEfficiency")
        first = efficiency_obj[0]
        FM.set_title_y(first, y_title)
        for obj in efficiency_obj:
            FM.set_range_y(obj, y_min, y_max)
        pad1.Update()
    pad.Update()
    pad.Modified()
    return c