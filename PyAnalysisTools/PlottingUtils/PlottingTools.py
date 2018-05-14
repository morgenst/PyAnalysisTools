import ROOT
from collections import defaultdict
from operator import itemgetter
from PyAnalysisTools.base import InvalidInputError, _logger
from PyAnalysisTools.PlottingUtils import Formatting as FM
from PyAnalysisTools.PlottingUtils import HistTools as HT
from PyAnalysisTools.ROOTUtils import ObjectHandle as object_handle
from PyAnalysisTools.PlottingUtils.PlotConfig import get_draw_option_as_root_str, get_style_setters_and_values
from PyAnalysisTools.ROOTUtils.ObjectHandle import get_objects_from_canvas_by_name
from PyAnalysisTools.PlottingUtils.PlotConfig import get_default_plot_config


def retrieve_new_canvas(name, title, size_x=800, size_y=600):
    canvas = ROOT.TCanvas(name, title, size_x, size_y)
    ROOT.SetOwnership(canvas, False)
    return canvas


def plot_obj(hist, plot_config, **kwargs):
    if isinstance(hist, ROOT.TH2):
        return plot_2d_hist(hist, plot_config, **kwargs)
    if isinstance(hist, ROOT.TH1):
        return plot_hist(hist, plot_config, **kwargs)
    if isinstance(hist, ROOT.TEfficiency) or isinstance(hist, ROOT.TGraph):
        return plot_graph(hist, plot_config, **kwargs)


def plot_objects(objects, plot_config, process_configs=None):
    """
    Base interface to plot multiple objects

    :param objects: objects to be plotted, e.g. TH1, TEfficiency
    :type objects: list or dict
    :param plot_config: plot configuration
    :type plot_config: PlotConfig
    :param process_configs: physics processes configuration containing e.g. colors and plot styles
    :type process_configs: ProcessConfig
    :return: canvas with plotted objects
    :rtype: TCanvas
    """
    if len(objects) == 0:
        _logger.warning("Requested plot objects with zero objects")
        return
    if isinstance(objects, dict):
        first_obj = objects.values()[0]
    elif isinstance(objects, list):
        first_obj = objects[0]
    if isinstance(first_obj, ROOT.TH1):
        return plot_histograms(objects, plot_config, process_configs)
    if isinstance(first_obj, ROOT.TEfficiency):
        return plot_graphs(objects.values(), plot_config)
    _logger.error("Unsupported type {:s} passed for plot_objects".format(type(objects.values()[0])))


def add_object_to_canvas(canvas, obj, plot_config, process_config=None, index=None):
    if isinstance(obj, ROOT.TH1):
        add_histogram_to_canvas(canvas, obj, plot_config, process_config, index)
    if isinstance(obj, ROOT.TGraphAsymmErrors) or isinstance(obj, ROOT.TEfficiency):
        add_graph_to_canvas(canvas, obj, plot_config)


def plot_hist(hist, plot_config, **kwargs):
    kwargs.setdefault("y_max", 1.1 * hist.GetMaximum())
    kwargs.setdefault("index", None)
    ymax = kwargs["y_max"]
    canvas = retrieve_new_canvas(plot_config.name, "")
    canvas.cd()
    ROOT.SetOwnership(hist, False)
    process_config = None
    draw_option = get_draw_option_as_root_str(plot_config, process_config)
    hist = format_obj(hist, plot_config)
    hist.Draw(draw_option)
    hist.SetMarkerSize(0.7)
    FM.apply_style(hist, plot_config, process_config, kwargs["index"])
    if ymax:
        _logger.info("Deprecated. Use plot_config.ymax")
        FM.set_maximum_y(hist, ymax)
    if hasattr(plot_config, "ymin"):
        FM.set_minimum_y(hist, plot_config.ymin)
    if plot_config.ymax:
        FM.set_maximum_y(hist, plot_config.ymax)
    if hasattr(plot_config, "logy") and plot_config.logy:
        canvas.SetLogy()
    if hasattr(plot_config, "logx") and plot_config.logx:
        canvas.SetLogx()
    if hasattr(plot_config, "axis_labels") and plot_config.axis_labels is not None:
        for b in range(len(plot_config.axis_labels)):
            hist.GetXaxis().SetBinLabel(b+1, plot_config.axis_labels[b])
    canvas.Update()
    return canvas


def plot_2d_hist(hist, plot_config, **kwargs):
    title = ""
    if hasattr(plot_config, "title"):
        title = plot_config.title
    canvas = retrieve_new_canvas(plot_config.name, title)
    canvas.cd()
    hist = format_obj(hist, plot_config)
    ROOT.SetOwnership(hist, False)
    hist.Draw(plot_config.draw_option)
    canvas.SetRightMargin(0.2)
    canvas.Modified()
    canvas.Update()
    return canvas


# todo: memoise
def fetch_process_config(process, process_config):
    if process is None or process_config is None:
        return None
    if process not in process_config:
        _logger.warning("Could not find process %s in process config" % process)
        return None
    return process_config[process]


def format_obj(obj, plot_config):
    if isinstance(obj, ROOT.TH1):
        return format_hist(obj, plot_config)
    if isinstance(obj, ROOT.TGraphAsymmErrors) or isinstance(obj, ROOT.TGraph):
        return format_hist(obj, plot_config)
    if isinstance(obj, ROOT.TEfficiency):
        return format_tefficiency(obj, plot_config)


def get_title_from_plot_config(plot_config):
    xtitle = None
    if hasattr(plot_config, "xtitle"):
        xtitle = plot_config.xtitle
        if hasattr(plot_config, "unit"):
            xtitle += " [" + plot_config.unit + "]"
    y_title = "Entries"
    if hasattr(plot_config, "ytitle"):
        y_title = plot_config.ytitle
    return xtitle, y_title


def format_tefficiency(obj, plot_config):
    xtitle, ytitle = get_title_from_plot_config(plot_config)
    if xtitle is None:
        xtitle = ""
    obj.SetTitle(";{:s};{:s}".format(xtitle, ytitle))
    if plot_config.xmin is not None and plot_config.xmax is not None:
        ROOT.gPad.Update()
        obj.GetPaintedGraph().GetXaxis().SetRangeUser(plot_config.xmin, plot_config.xmax)
        obj.GetPaintedGraph().Set(0)
    return obj


def format_hist(hist, plot_config):
    xtitle, ytitle = get_title_from_plot_config(plot_config)
    if xtitle:
        FM.set_title_x(hist, xtitle)
    if hasattr(plot_config, "unit"):
        ytitle += " / %.1f %s" % (hist.GetXaxis().GetBinWidth(0), plot_config.unit)
    FM.set_title_y(hist, ytitle)
    yscale = 1.1
    if plot_config.logy:
        yscale = 100.
    if isinstance(hist, ROOT.TH2):
        if hasattr(plot_config, "ztitle"):
            hist.GetZaxis().SetTitle(plot_config.ztitle)
        if hasattr(plot_config, "rebinX") and hasattr(plot_config.rebinY):
            hist = HT.rebin2D(hist, plot_config.rebinX, plot_config.rebinY)
    if hasattr(plot_config, "normalise") and plot_config.normalise:
        HT.normalise(hist, plot_config.normalise_range)
        yscale = 1.1
        if hasattr(plot_config, "yscale"):
            yscale = yscale
        ymax = yscale * hist.GetMaximum()
        if hasattr(plot_config, "ymax"):
            plot_config.ymax = max(plot_config.ymax, ymax)
        else:
            plot_config.ymax = ymax
    if plot_config.rebin and not isinstance(hist, ROOT.THStack) and not plot_config.ignore_rebin:
        hist = HT.rebin(hist, plot_config.rebin)
        if hasattr(plot_config, "yscale"):
            yscale = yscale
        ymax = yscale*hist.GetMaximum()
        if hasattr(plot_config, "ymax"):
            plot_config.ymax = max(plot_config.ymax, ymax)
        else:
            plot_config.ymax = ymax
    return hist


def plot_graphs(graphs, plot_config):
    canvas = plot_graph(graphs[0], plot_config)
    for graph in graphs[1:]:
        add_graph_to_canvas(canvas, graph, plot_config)
    return canvas


def add_signal_to_canvas(signal, canvas, plot_config, process_configs):
    add_histogram_to_canvas(canvas, signal[1], plot_config, process_configs[signal[0]])


def plot_histograms(hists, plot_config, process_configs=None):
    if plot_config is None:
        plot_config = get_default_plot_config(hists[0])
    canvas = retrieve_new_canvas(plot_config.name, "")
    canvas.cd()
    is_first = True
    if isinstance(hists, dict):
        hist_defs = hists.items()
    elif isinstance(hists, list):
        hist_defs = zip([None] * len(hists), hists)
    max_y = 1.1 * max([item[1].GetMaximum() for item in hist_defs])
    if plot_config.ordering is not None:
        sorted(hist_defs, key=lambda k: plot_config.ordering.index(k[0]))
    for process, hist in hist_defs:
        index = map(itemgetter(1), hist_defs).index(hist)
        hist = format_hist(hist, plot_config)
        process_config = fetch_process_config(process, process_configs)
        if not (plot_config.is_set_to_value("ignore_style", True)) and \
                plot_config.is_set_to_value("ignore_style", False):
            draw_option = get_draw_option_as_root_str(plot_config, process_config)
        else:
            draw_option = "hist"
        if not is_first and "same" not in draw_option:
            draw_option += "sames"
        hist.Draw(draw_option)
        #todo: might break something upstream
        # if common_config is None or common_config.ignore_style:
        #     style_setter = "Line"
        if plot_config.ignore_style:
            style_setter = "Line"
        FM.apply_style(hist, plot_config, process_config, index=index)
        if is_first:
            if isinstance(hist, ROOT.TH2) and draw_option.lower() == "colz":
                canvas.SetRightMargin(0.15)
            FM.set_minimum_y(hist, plot_config.ymin)
            FM.set_maximum_y(hist, max_y)
            if plot_config.xmin and not plot_config.xmax:
                FM.set_minimum(hist, plot_config.xmin, "x")
            elif plot_config.xmin and plot_config.xmax:
                FM.set_range(hist, plot_config.xmin, plot_config.xmax, "x")
            if plot_config.logy:
                if hasattr(plot_config, "ymin"):
                    hist.SetMinimum(max(1., plot_config.ymin))
                else:
                    hist.SetMinimum(0.0001)
                canvas.SetLogy()
            if plot_config.logx:
                canvas.SetLogx()
            format_hist(hist, plot_config)
            if plot_config.ymax:
                 hist.SetMaximum(plot_config.ymax)
            canvas.Update()
        is_first = False
    if hasattr(plot_config, "normalise") and plot_config.normalise is True:
        hist_defs[0][1].SetMaximum(plot_config.ymax)
    canvas.Update()
    return canvas


def add_fit_to_canvas(canvas, fit_result, pdf=None, frame=None):
    canvas.cd()
    if frame:
        pdf.paramOn(frame, ROOT.RooFit.Layout(0.50, 0.9, 0.8))
        chi2 = frame.chiSquare("model", "data", 3)
        txt = ROOT.TText(2, 100, "#chi^{2} = " + "{:.2f}".format(chi2))
        ROOT.SetOwnership(txt, False)
        txt.SetTextSize(0.04)
        txt.SetTextColor(ROOT.kRed)
        frame.addObject(txt)
    else:
        for i in range(len(fit_result.floatParsFinal()) - 1):
            var = fit_result.floatParsFinal()[i]
            var_string = "{:s} = {:.2f} \pm {:.2f}".format(var.GetName(), var.getValV(), var.getError())
            FM.add_text_to_canvas(canvas, var_string, pos={'x': 0.15, 'y': 0.9 - i * 0.05}, size=0.04, color=None)
    canvas.Update()


def apply_style(obj, style_setter, style_attr, color):
    """
    Apply defined styles to plottable object

    :param obj: plot object to be styled
    :type obj: TGraph, TH1, ...
    :param style_setter: attribute to be set, e.g. Fill, Marker, Line
    :type style_setter: str
    :param style_attr: attribute value
    :type style_attr: str
    :param color: color for attribute
    :type color: int
    :return: None
    :rtype: None
    """
    if style_attr is not None:
        for ss in style_setter:
            getattr(obj, "Set" + ss + "Style")(style_attr)
    if color is not None:
        for ss in style_setter:
            getattr(obj, "Set" + ss + "Color")(color)


def add_histogram_to_canvas(canvas, hist, plot_config, process_config=None, index=None):
    canvas.cd()
    draw_option = get_draw_option_as_root_str(plot_config, process_config)
    hist = format_obj(hist, plot_config)
    apply_style(hist, *get_style_setters_and_values(plot_config, process_config, index))
    if "same" not in draw_option:
        draw_option += "sames"
    hist.Draw(draw_option)
    canvas.Update()


def plot_graph(graph, plot_config=None, **kwargs):
    """
    Plot a TGraph object

    :param graph: object to be plotted
    :type graph: TGraph
    :param plot_config: plot configuration defining style
    :type plot_config: PlotConfig
    :param kwargs: additional arguments like canvas name and title
    :type kwargs:
    :return: canvas containing plotted and formatted TGraph
    :rtype: TCanvas
    """
    kwargs.setdefault("canvas_name", graph.GetName())
    kwargs.setdefault("canvas_title", "")
    canvas = retrieve_new_canvas(kwargs["canvas_name"], kwargs["canvas_title"])
    canvas.cd()
    draw_option = "a" + get_draw_option_as_root_str(plot_config)
    graph.Draw(draw_option)
    if not "same" in draw_option:
        draw_option += "same"
    apply_style(graph, *get_style_setters_and_values(plot_config))
    ROOT.SetOwnership(graph, False)
    if plot_config:
        graph = format_obj(graph, plot_config)
    if hasattr(plot_config, "logy") and plot_config.logy:
        canvas.SetLogy()
    canvas.Update()
    return canvas


def add_graph_to_canvas(canvas, graph, plot_config):
    canvas.cd()
    draw_option = get_draw_option_as_root_str(plot_config)
    if not "same" in draw_option:
        draw_option += "same"
    apply_style(graph, *get_style_setters_and_values(plot_config))
    graph.Draw(draw_option)
    ROOT.SetOwnership(graph, False)
    canvas.Update()


def apply_ordering(hist_defs, ordering):
    for process, _ in hist_defs:
        if process not in ordering:
            ordering.append(process)
    return sorted(hist_defs, key=lambda k: ordering.index(k[0]))


def plot_stack(hists, plot_config, **kwargs):
    """
    Plot THStack
    :param hists: histogram list to be stacked
    :param plot_config:
    :param kwargs:
    :return:
    """
    kwargs.setdefault("process_configs", None)
    process_configs = kwargs["process_configs"]
    canvas = retrieve_new_canvas(plot_config.name, "")
    canvas.Clear()
    canvas.cd()
    is_first = True
    if isinstance(hists, dict) or isinstance(hists, defaultdict):
        hist_defs = hists.items()
    elif isinstance(hists, list):
        hist_defs = zip([None] * len(hists), hists)
    stack = ROOT.THStack('hs', '')
    ROOT.SetOwnership(stack, False)
    data = None
    if plot_config.ordering is not None:
        hist_defs = apply_ordering(hist_defs, plot_config.ordering)
    for process, hist in hist_defs:
        if "data" in process.lower():
            #todo: problem if two distinct data sets
            data = (process, hist)
            continue
        hist = format_hist(hist, plot_config)
        process_config = fetch_process_config(process, process_configs)
        draw_option = get_draw_option_as_root_str(plot_config, process_config)
        FM.apply_style(hist, plot_config, process_config)
        stack.Add(hist, draw_option)
    stack.Draw()
    canvas.Update()
    format_hist(stack, plot_config)
    y_scale_offset = 1.1
    if plot_config.logy:
        y_scale_offset = 100.
    max_y = y_scale_offset * stack.GetMaximum()
    if data is not None:
        add_data_to_stack(canvas, data[1], plot_config)
        max_y = max(max_y, y_scale_offset * data[1].GetMaximum())
        if plot_config.rebin and not plot_config.ignore_rebin:
            max_y = max(max_y, 1.3 * get_objects_from_canvas_by_name(canvas, data[1].GetName())[0].GetMaximum())
    if plot_config.ymax:
        max_y = plot_config.ymax
        if isinstance(max_y, str):
            max_y = eval(max_y)
    FM.set_maximum_y(stack, max_y)
    if hasattr(plot_config, "ymin"):
        FM.set_minimum_y(stack, plot_config.ymin)
    if hasattr(plot_config, "logy") and plot_config.logy:
        stack.SetMinimum(0.1)
        canvas.SetLogy()
    if plot_config.logx:
        canvas.SetLogx()
    return canvas


def add_data_to_stack(canvas, data, plot_config=None, blind=None):
    if blind:
        blind_data(data, blind)
    canvas.cd()
    ROOT.SetOwnership(data, False)
    data = format_hist(data, plot_config)
    data.Draw("Esames")


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
    if isinstance(ratio, ROOT.TCanvas):
        supported_types = ["TH1F", "TH1D", "TGraph", "TGraphAsymmErrors", "TEfficiency"]
        try:
            hratio = object_handle.get_objects_from_canvas_by_type(ratio, supported_types)[0]
        except:
            _logger.error("Could not find any supported hist type in canvas ", ratio.GetName())
            exit(0)
    else:
        hratio = ratio

    if name is None:
        name = canvas.GetName() + "_ratio"
    c = retrieve_new_canvas(name, title)
    c.Draw()
    pad1 = ROOT.TPad("pad1", "top pad", 0., y_frac, 1., 1.)
    pad1.SetBottomMargin(0.05)
    pad1.Draw()
    pad2 = ROOT.TPad("pad2", "bottom pad", 0., 0., 1, ((1 - y_frac) * canvas.GetBottomMargin() / y_frac + 1) * y_frac)
    pad2.SetBottomMargin(0.1)
    pad2.Draw()
    pad1.cd()
    object_handle.get_objects_from_canvas(canvas)
    try:
        stack = object_handle.get_objects_from_canvas_by_type(canvas, "THStack")[0]
    except IndexError:
        try:
            stack = object_handle.get_objects_from_canvas_by_type(canvas, "TEfficiency")[0]
        except IndexError:
            stack = object_handle.get_objects_from_canvas_by_type(canvas, "TH1")[0]
    stack.GetXaxis().SetTitleSize(0)
    stack.GetXaxis().SetLabelSize(0)
    stack.SetMinimum(max(stack.GetMinimum(), 0.1))
    scale = 1. / (1. - y_frac)
    scale_frame_text(stack, scale)
    canvas.DrawClonePad()

    pad2.cd()
    hratio.GetYaxis().SetNdivisions(505)
    scale = 1. / (((1 - y_frac) * (canvas.GetBottomMargin()) / y_frac + 1) * y_frac)

    reset_frame_text(hratio)
    scale_frame_text(hratio, scale)
    ratio.Update()
    ratio.SetBottomMargin(0.4)
    ratio.DrawClonePad()
    pad2.Update()
    xlow = pad2.GetUxmin()
    xup = pad2.GetUxmax()
    if ratio.GetLogx():
        stack = object_handle.get_objects_from_canvas_by_type(canvas, "TH1")[0]
        xlow = stack.GetXaxis().GetXmin()
        xup = stack.GetXaxis().GetXmax()
    line = ROOT.TLine(xlow, 1, xup, 1)
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
