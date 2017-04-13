import ROOT
from PyAnalysisTools.base import InvalidInputError, _logger
from PyAnalysisTools.PlottingUtils import Formatting as FM
from PyAnalysisTools.PlottingUtils import HistTools as HT
from PyAnalysisTools.ROOTUtils import ObjectHandle as object_handle
from PyAnalysisTools.PlottingUtils.PlotConfig import get_draw_option_as_root_str, get_style_setters_and_values


def retrieve_new_canvas(name, title, size_x=800, size_y=600):
    return ROOT.TCanvas(name, title, size_x, size_y)


def plot_obj(hist, plot_config, **kwargs):
    if isinstance(hist, ROOT.TH2):
        return plot_2d_hist(hist, plot_config, **kwargs)
    if isinstance(hist, ROOT.TH1):
        return plot_hist(hist, plot_config, **kwargs)
    if isinstance(hist, ROOT.TEfficiency):
        return plot_graph(hist, plot_config, **kwargs)


def plot_objects(objects, plot_config, process_configs=None):
    if len(objects) == 0:
        _logger.warning("Requested plot objects with zero objects")
        return
    if isinstance(objects.values()[0], ROOT.TH1):
        return plot_histograms(objects, plot_config, process_configs)
    _logger.error("Unsupported type {:s} passed for plot_objects".format(type(objects[0])))


def add_object_to_canvas(canvas, obj, plot_config):
    if isinstance(obj, ROOT.TH1):
        add_histogram_to_canvas(canvas, obj, plot_config)
    if isinstance(obj, ROOT.TGraphAsymmErrors) or isinstance(obj, ROOT.TEfficiency):
        add_graph_to_canvas(canvas, obj, plot_config)


def plot_hist(hist, plot_config, **kwargs):
    kwargs.setdefault("y_max", None)
    ymax = kwargs["y_max"]
    canvas = retrieve_new_canvas(plot_config.name, "")
    canvas.cd()
    ROOT.SetOwnership(hist, False)
    process_config = None
    draw_option = get_draw_option_as_root_str(plot_config, process_config)
    style_setter, style_attr, color = get_style_setters_and_values(plot_config, process_config)
    hist = format_obj(hist, plot_config)
    hist.Draw(draw_option)
    hist.SetMarkerSize(0.7)
    if style_attr is not None:
        getattr(hist, "Set"+style_setter+"Style")(style_attr)
    if color is not None:
        getattr(hist, "Set" + style_setter + "Color")(color)
    if ymax:
        _logger.info("Deprecated. Use plot_config.ymax")
        FM.set_maximum_y(hist, ymax)
    if hasattr(plot_config, "ymin"):
        FM.set_minimum_y(hist, plot_config.ymin)
    if hasattr(plot_config, "ymax"):
        FM.set_maximum_y(hist, plot_config.ymax)
    if hasattr(plot_config, "logy") and plot_config.logy:
        canvas.SetLogy()
    canvas.Update()
    return canvas


def plot_2d_hist(hist, plot_config, **kwargs):
    title = ""
    if hasattr(plot_config, "title"):
        title = plot_config.title
    canvas = retrieve_new_canvas(plot_config.name, title)
    canvas.cd()
    ROOT.SetOwnership(hist, False)
    hist.Draw(plot_config.draw_option)
    canvas.SetRightMargin(0.1)
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
    if isinstance(obj, ROOT.TGraphAsymmErrors):
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
    return obj


def format_hist(hist, plot_config):
    xtitle, ytitle = get_title_from_plot_config(plot_config)
    if xtitle:
        FM.set_title_x(hist, xtitle)
    if hasattr(plot_config, "unit"):
        ytitle += " / %.1f %s" % (hist.GetXaxis().GetBinWidth(0), plot_config.unit)
    FM.set_title_y(hist, ytitle)
    if hasattr(plot_config, "rebin"):
        hist = HT.rebin(hist, plot_config.rebin)
        yscale = 1.1
        if hasattr(plot_config, "yscale"):
            yscale = yscale
        ymax = yscale*hist.GetMaximum()
        if hasattr(plot_config, "ymax"):
            plot_config.ymax = max(plot_config.ymax, ymax)
        else:
            plot_config.ymax = ymax
    return hist


def plot_histograms(hists, plot_config, process_configs=None):
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
        hist = format_hist(hist, plot_config)
        process_config = fetch_process_config(process, process_configs)
        if not (plot_config.is_set_to_value("ignore_style", True)) and \
                plot_config.is_set_to_value("ignore_style", False):
            draw_option = get_draw_option_as_root_str(plot_config, process_config)
        else:
            draw_option = "hist"
        style_setter, style_attr, color = get_style_setters_and_values(plot_config, process_config)
        if not is_first and "same" not in draw_option:
            draw_option += "sames"
        hist.Draw(draw_option)
        #todo: might break something upstream
        # if common_config is None or common_config.ignore_style:
        #     style_setter = "Line"
        if plot_config.ignore_style:
            style_setter = "Line"
        if style_attr is not None and not plot_config.ignore_style:
            getattr(hist, "Set"+style_setter+"Style")(style_attr)
        if color is not None:
            hist_color = color
            if isinstance(color, list):
                hist_color = color[hists.index(hist)]
            getattr(hist, "Set" + style_setter + "Color")(hist_color)
        if is_first:
            FM.set_minimum_y(hist, plot_config.y_min)
            FM.set_maximum_y(hist, max_y)
            if plot_config.logy:
                if hasattr(plot_config, "ymin"):
                    hist.SetMinimum(max(1., plot_config.ymin))
                else:
                    hist.SetMinimum(0.0001)
                canvas.SetLogy()

                if hasattr(plot_config, "ymax"):
                    hist.SetMaximum(plot_config.ymax)
            format_hist(hist, plot_config)
            canvas.Update()
        is_first = False
    return canvas


def add_histogram_to_canvas(canvas, hist, plot_config):
    canvas.cd()
    draw_option = get_draw_option_as_root_str(plot_config)
    style_setter, style_attr, color = get_style_setters_and_values(plot_config)
    hist = format_obj(hist, plot_config)
    if style_attr is not None:
        getattr(hist, "Set" + style_setter + "Style")(style_attr)
    if color is not None:
        getattr(hist, "Set" + style_setter + "Color")(color)
    if "same" not in draw_option:
        draw_option += "sames"
    hist.Draw(draw_option)
    canvas.Update()


def plot_graph(graph, plot_config=None, **kwargs):
    kwargs.setdefault("canvas_name", graph.GetName())
    kwargs.setdefault("canvas_title", "")
    canvas = retrieve_new_canvas(kwargs["canvas_name"], kwargs["canvas_title"])
    canvas.cd()
    graph.Draw("ap")
    ROOT.SetOwnership(graph, False)
    if plot_config:
        graph = format_obj(graph, plot_config)
    canvas.Update()
    return canvas


def add_graph_to_canvas(canvas, graph, plot_config):
    canvas.cd()
    draw_option = get_draw_option_as_root_str(plot_config)
    if not "same" in draw_option:
        draw_option += "same"
    style_setter, style_attr, color = get_style_setters_and_values(plot_config)
    if style_attr is not None:
        getattr(graph, "Set" + style_setter + "Style")(style_attr)
    if color is not None:
        getattr(graph, "Set" + style_setter + "Color")(color)
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
    if isinstance(hists, dict):
        hist_defs = hists.items()
    elif isinstance(hists, list):
        hist_defs = zip([None] * len(hists), hists)
    stack = ROOT.THStack('hs', '')
    ROOT.SetOwnership(stack, False)
    data = None
    if plot_config.ordering is not None:
        hist_defs = apply_ordering(hist_defs, plot_config.ordering)
    for process, hist in hist_defs:
        if process == "Data":
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
    max_y = 1.1 * stack.GetMaximum()
    if data is not None:
        add_data_to_stack(canvas, *data)
        max_y = max(max_y, 1.1 * data[1].GetMaximum())
    FM.set_maximum_y(stack, max_y)
    if hasattr(plot_config, "ymin"):
        FM.set_minimum_y(stack, plot_config.ymin)
    if hasattr(plot_config, "logy") and plot_config.logy:
        stack.SetMinimum(0.1)
        canvas.SetLogy()
    return canvas


def add_data_to_stack(canvas, process, data, blind=None):
    if blind:
        blind_data(data, blind)
    canvas.cd()
    ROOT.SetOwnership(data, False)
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
        try:
            hratio = object_handle.get_objects_from_canvas_by_type(ratio, "TH1F")[0]
        except IndexError:
            try:
                hratio = object_handle.get_objects_from_canvas_by_type(ratio, "TH1D")[0]
            except IndexError:
                try:
                    hratio = object_handle.get_objects_from_canvas_by_type(ratio, "TGraph")[0]
                except IndexError:
                    try:
                        hratio = object_handle.get_objects_from_canvas_by_type(ratio, "TGraphAsymmErrors")[0]
                    except IndexError:
                        hratio = object_handle.get_objects_from_canvas_by_type(ratio, "TEfficiency")[0]
    else:
        hratio = ratio
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
    object_handle.get_objects_from_canvas(canvas)

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