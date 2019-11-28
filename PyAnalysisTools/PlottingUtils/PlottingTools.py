from __future__ import division
from builtins import map
from builtins import zip
from builtins import range
import ROOT
from collections import defaultdict
from operator import itemgetter
from PyAnalysisTools.base import InvalidInputError, _logger
from PyAnalysisTools.PlottingUtils import Formatting as fm
from PyAnalysisTools.PlottingUtils import HistTools as ht
from PyAnalysisTools.PlottingUtils.PlotConfig import get_draw_option_as_root_str, get_style_setters_and_values
from PyAnalysisTools.PlottingUtils.PlotConfig import get_default_plot_config
from PyAnalysisTools.base.ProcessConfig import find_process_config
import PyAnalysisTools.PlottingUtils.PlotableObject as PO


def retrieve_new_canvas(name, title='', size_x=800, size_y=600):
    canvas = ROOT.TCanvas(name, title, size_x, size_y)
    canvas.SetRightMargin(0.07)
    ROOT.SetOwnership(canvas, False)
    return canvas


def plot_obj(hist, plot_config, **kwargs):
    if isinstance(hist, ROOT.TH2):
        return plot_2d_hist(hist, plot_config, **kwargs)
    if isinstance(hist, ROOT.TH1):
        return plot_hist(hist, plot_config, **kwargs)
    if isinstance(hist, ROOT.TEfficiency) or isinstance(hist, ROOT.TGraph):
        return plot_graph(hist, plot_config, **kwargs)


def project_hist(tree, hist, var_name, cut_string='', weight=None, is_data=False):
    if cut_string is None:
        cut_string = ''
    if weight:
        if 'MC:' in weight:
            weight = weight.split('*')
            mc_weights = [w for w in weight if 'MC:' in w]
            for mc_w in mc_weights:
                weight.remove(mc_w)
            weight = '*'.join(weight)
            if not is_data:
                mc_weights = [mc_w.replace('MC:', '') for mc_w in mc_weights]
                for mc_w in mc_weights:
                    weight += '* {:s}'.format(mc_w)
        if 'DATA:' in weight:
            weight = weight.split('*')
            data_weights = [w for w in weight if 'DATA:' in w]
            for data_w in data_weights:
                weight.remove(data_w)
            weight = '*'.join(weight)
            if is_data:
                data_weights = [data_w.replace('DATA:', '') for data_w in data_weights]
                for data_w in data_weights:
                    weight += '* {:s}'.format(data_w)
        if cut_string == '':
            cut_string = weight
        else:
            cut_string = '%s * (%s)' % (weight, cut_string)
    n_selected_events = tree.Project(hist.GetName(), var_name, cut_string)
    _logger.debug("Selected %i events from tree %s for distribution %s and cut %s." % (n_selected_events,
                                                                                       tree.GetName(),
                                                                                       var_name,
                                                                                       cut_string))
    if n_selected_events != hist.GetEntries():
        _logger.error("No of selected events does not match histogram entries. Probably FileHandle has been "
                      "initialised after histogram definition has been received")
        raise RuntimeError("Inconsistency in TTree::Project")
    if n_selected_events == -1:
        _logger.error("Unable to project {:s} from tree {:s} with cut {:s}".format(var_name, tree.GetName(),
                                                                                   cut_string))
        raise RuntimeError("TTree::Project failed")
    return hist


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
        first_obj = list(objects.values())[0]
    elif isinstance(objects, list):
        first_obj = objects[0]
    if isinstance(first_obj, PO.PlotableObject):
        if isinstance(first_obj.plot_object, ROOT.TH1):
            return plot_histograms(objects, plot_config, process_configs)
        # if isinstance(first_obj, ROOT.TEfficiency) or isinstance(first_obj, ROOT.TGraph):
        #     return plot_graphs(objects, plot_config)
    if isinstance(first_obj, ROOT.TH1):
        return plot_histograms(objects, plot_config, process_configs)
    if isinstance(first_obj, ROOT.TEfficiency) or isinstance(first_obj, ROOT.TGraph):
        return plot_graphs(objects, plot_config)
    _logger.error("Unsupported type {:s} passed for plot_objects".format(type(list(objects.values())[0])))


def add_object_to_canvas(canvas, obj, plot_config, process_config=None, index=None):
    """
    Add an object to a canvas
    :param canvas: canvas to which the object should be added
    :type canvas: ROOT.TCanvas
    :param obj: plot object
    :type obj: ROOT.TH1, ROOT.TGraph, etc
    :param plot_config: plot configuration defining outline
    :type plot_config: PlotConfig
    :param process_config: specific process configuration (optional)
    :type process_config: ProcessConfig
    :param index: index of plot object in list of all plot objects to set specific styles like colors attached to
    plot_config (optional)
    :type index: int
    :return: nothing
    :rtype: None
    """
    if isinstance(obj, ROOT.TH1):
        add_histogram_to_canvas(canvas, obj, plot_config, process_config, index)
    if isinstance(obj, ROOT.TGraphAsymmErrors) or isinstance(obj, ROOT.TEfficiency) or isinstance(obj, ROOT.TGraph):
        add_graph_to_canvas(canvas, obj, plot_config)


def plot_hist(hist, plot_config, **kwargs):
    kwargs.setdefault("y_max", plot_config.yscale * hist.GetMaximum())
    # kwargs.setdefault("y_max", 1.1 * hist[0].GetMaximum()) - sm dev
    kwargs.setdefault("index", None)
    canvas = retrieve_new_canvas(plot_config.name, "")
    canvas.cd()
    ROOT.SetOwnership(hist, False)
    process_config = None
    draw_option = get_draw_option_as_root_str(plot_config, process_config)
    hist = format_obj(hist, plot_config)
    hist.Draw(draw_option)
    hist.SetMarkerSize(0.7)
    fm.apply_style(hist, plot_config, process_config, index=kwargs["index"])
    if plot_config.ymin:
        fm.set_minimum_y(hist, plot_config.ymin)
    if plot_config.ymax:
        fm.set_maximum_y(hist, plot_config.ymax)
    if plot_config.logy:
        hist.SetMaximum(hist.GetMaximum() * plot_config.yscale_log)
        if plot_config.ymin:
            hist.SetMinimum(max(0.1, plot_config.ymin))
        if hist.GetMinimum() == 0.:
            hist.SetMinimum(0.9)
        if plot_config.normalise:
            # hist.SetMinimum(0.000001)
            hist.SetMinimum(0.)
            fm.set_minimum_y(hist, plot_config.ymin)
        canvas.SetLogy()
    if plot_config.logx:
        canvas.SetLogx()
    ht.set_axis_labels(hist, plot_config)
    canvas.Update()
    return canvas


def plot_2d_hist(hist, plot_config, **kwargs):
    title = plot_config.title
    canvas = retrieve_new_canvas(plot_config.name, title)
    canvas.cd()
    hist = format_obj(hist, plot_config)
    ROOT.SetOwnership(hist, False)
    draw_option = plot_config.draw_option
    if draw_option is None:
        _logger.warning("No draw option provided for TH2 for pc: {:s}. "
                        "Fall back to default: COLZ".format(plot_config.name))
        draw_option = 'COLZ'
    hist.Draw(draw_option)
    if plot_config.logx:
        canvas.SetLogx()
    if plot_config.logy:
        canvas.SetLogy()
    if plot_config.logz:
        canvas.SetLogz()
    if plot_config.style is not None:
        hist.SetMarkerStyle(plot_config.style)
    canvas.SetRightMargin(0.2)
    canvas.Modified()
    canvas.Update()
    return canvas


# todo: remove
# def fetch_process_config_old(process, process_config):
#     if process is None or process_config is None:
#         return None
#     if process not in process_config:
#         _logger.warning("Could not find process %s in process config" % process)
#         return None
#     return process_config[process]


def format_obj(obj, plot_config):
    if isinstance(obj, ROOT.TH1):
        return format_hist(obj, plot_config)
    if isinstance(obj, ROOT.TGraphAsymmErrors) or isinstance(obj, ROOT.TGraph) or isinstance(obj, ROOT.TMultiGraph):
        return format_hist(obj, plot_config)
    if isinstance(obj, ROOT.TEfficiency):
        return format_tefficiency(obj, plot_config)
    _logger.warn('Could not find implementation for ', obj)
    return obj


def get_title_from_plot_config(plot_config):
    xtitle = None
    if plot_config.xtitle is not None:
        xtitle = plot_config.xtitle
        if hasattr(plot_config, "unit"):
            xtitle += " [" + plot_config.unit + "]"
    y_title = "Entries"
    if plot_config.ytitle is not None:
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
    if plot_config is None:
        return hist
    xtitle, ytitle = get_title_from_plot_config(plot_config)
    if xtitle:
        fm.set_title_x(hist, xtitle)
    if plot_config.xtitle_offset is not None:
        fm.set_title_x_offset(hist, plot_config.xtitle_offset)
    if plot_config.xtitle_size is not None:
        fm.set_title_x_size(hist, plot_config.xtitle_size)
    if hasattr(plot_config, "unit"):
        ytitle += " / %.1f %s" % (hist.GetXaxis().GetBinWidth(0), plot_config.unit)
    fm.set_title_y(hist, ytitle)
    if plot_config.ytitle_offset is not None:
        fm.set_title_y_offset(hist, plot_config.ytitle_offset)
    if plot_config.ytitle_size is not None:
        fm.set_title_y_size(hist, plot_config.ytitle_size)
    if isinstance(hist, ROOT.TH2):
        if plot_config.ztitle is not None:
            hist.GetZaxis().SetTitle(plot_config.ztitle)
        if plot_config.ztitle_offset is not None:
            fm.set_title_z_offset(hist, plot_config.ztitle_offset)
        if plot_config.ztitle_size is not None:
            fm.set_title_z_size(hist, plot_config.ztitle_size)
        if hasattr(plot_config, "rebinX") and hasattr(plot_config.rebinY):
            hist = ht.rebin2D(hist, plot_config.rebinX, plot_config.rebinY)
        if hasattr(plot_config, "zmin") and hasattr(plot_config, "zmax"):
            fm.set_range_z(hist, plot_config.zmin, plot_config.zmax)
    if plot_config.normalise:
        ht.normalise(hist, plot_config.normalise_range, plot_config.norm_scale)
        ymax = plot_config.yscale * hist.GetMaximum()
        if plot_config.ymax is not None:
            plot_config.ymax = max(plot_config.ymax, ymax)
        else:
            plot_config.ymax = ymax
    if plot_config.rebin and not isinstance(hist, ROOT.THStack) and not plot_config.ignore_rebin:
        hist = ht.rebin(hist, plot_config.rebin)
        ymax = plot_config.yscale*hist.GetMaximum()
        if plot_config.ymax is not None:
            plot_config.ymax = min(plot_config.ymax, ymax)
        else:
            plot_config.ymax = ymax

    ROOT.SetOwnership(hist, False)
    return hist


def make_graph(name, x_vals, y_vals):
    """
    Create a TGraph based on x and y-values
    :param name: name of graph
    :type name: str
    :param x_vals: x-values
    :type x_vals: list
    :param y_vals: y-values
    :type y_vals: lists
    :return: graph
    :rtype: ROOT.TGraph
    """
    g = ROOT.TGraph(len(x_vals))
    g.SetName(name)
    for i, x in enumerate(x_vals):
        g.SetPoint(i, x, y_vals[i])
    ROOT.SetOwnership(g, False)
    return g


def plot_graphs(graphs, plot_config):
    """
    Plot function for graphs
    :param graphs: graphs to plot
    :type graphs: list or dict of TGraphs
    :param plot_config: plot configuration
    :type plot_config: PlotConfig
    :return: canvas with drawn graphs
    :rtype: ROOT.TCanvas
    """
    if isinstance(graphs, dict):
        graphs = list(graphs.values())
    canvas = plot_graph(graphs[0], plot_config)
    for index, graph in enumerate(graphs[1:]):
        add_graph_to_canvas(canvas, graph, plot_config, index+1)
    return canvas


def add_signal_to_canvas(signal, canvas, plot_config, process_configs):
    """
    Overlay a signal histogram to an existing canvas
    :param signal: tuple of signal name, histogram
    :type signal: list
    :param canvas: existing canvas to which signal should be added
    :type canvas: ROOT.TCanvas
    :param plot_config: plot configuration
    :type plot_config: PlotConfig
    :param process_configs: process configuration
    :type process_configs: ProcessConfig
    :return: nothing
    :rtype: None
    """
    add_histogram_to_canvas(canvas, signal[1], plot_config, process_configs[signal[0]])


def plot_histograms(hists, plot_config, process_configs=None, switchOff=False):
    """
    Plot histograms in canvas and apply styles according to plot and process configuration
    :param hists: histograms to plots
    :type hists: list or dict
    :param plot_config: plot configuration
    :type plot_config: PlotConfig
    :param process_configs: process configuration
    :type process_configs: ProcessConfig
    :param switchOff: switch off style application (optional)
    :type switchOff: bool
    :return: canvas with plotted histograms
    :rtype: ROOT.TCanvas
    """
    if plot_config is None:
        plot_config = get_default_plot_config(hists[0])
    canvas = retrieve_new_canvas(plot_config.name, "")
    canvas.cd()
    is_first = True
    max_y = None
    if isinstance(hists, dict):
        hist_defs = list(hists.items())
        # import re
        # hist_defs.sort(key=lambda s: int(re.findall('\d+', s[0])[0]))
    elif isinstance(hists, list):
        hist_defs = list(zip([None] * len(hists), hists))

    if isinstance(hist_defs[0][1], PO.PlotableObject):
        if not switchOff and not isinstance(hist_defs[0][1].plot_object, ROOT.TH2):
            max_y = 1.4 * max([item[1].plot_object.GetMaximum() for item in hist_defs])
        elif isinstance(hist_defs[0][1].plot_object, ROOT.TH2):
            max_y = plot_config.ymax
        # if plot_config.ordering is not None:
        #     sorted(hist_defs, key=lambda k: plot_config.ordering.index(k[0]))

        for process, hist in hist_defs:
            hist.plot_object = format_hist(hist.plot_object, plot_config)
            process_config = find_process_config(process, process_configs)
            if not (plot_config.is_set_to_value("ignore_style", True)) and \
                    plot_config.is_set_to_value("ignore_style", False):
                setattr(plot_config, 'draw', hist.draw_option)
                draw_option = get_draw_option_as_root_str(plot_config, process_config)
            else:
                draw_option = "hist"
            if not is_first and "same" not in draw_option:
                draw_option += "sames"
            hist.plot_object.Draw(draw_option)
            fm.apply_style_plotableObject(hist)

            if is_first:
                if isinstance(hist.plot_object, ROOT.TH2) and draw_option.lower() == "colz":
                    canvas.SetRightMargin(0.15)
                    fm.set_minimum_y(hist.plot_object, plot_config.ymin)
                    if plot_config.logz:
                        canvas.SetLogz()
                if switchOff:
                    fm.set_maximum_y(hist.plot_object, plot_config.ymax)
                else:
                    fm.set_maximum_y(hist.plot_object, max_y)
                    fm.set_minimum_y(hist.plot_object, plot_config.ymin)
                if plot_config.xmin and not plot_config.xmax:
                    fm.set_minimum(hist.plot_object, plot_config.xmin, "x")
                elif plot_config.xmin and plot_config.xmax:
                    fm.set_range(hist.plot_object, plot_config.xmin, plot_config.xmax, "x")
                if plot_config.logx:
                    canvas.SetLogx()
                format_hist(hist.plot_object, plot_config)
                if not isinstance(hist.plot_object, ROOT.TH2):
                    if plot_config.ymax:
                        hist.plot_object.SetMaximum(plot_config.ymax)
                    else:
                        hist.plot_object.SetMaximum(hist.plot_object.GetMaximum() * 1.2)
            if plot_config.logy:
                hist.plot_object.SetMaximum(hist.plot_object.GetMaximum() * 100.)
                if plot_config.ymin > 0.:
                    hist.plot_object.SetMinimum(plot_config.ymin)
                # if hasattr(plot_config, "ymin"):
                #     hist.plot_object.SetMinimum(max(0.1, plot_config.ymin))
                else:
                    hist.plot_object.SetMinimum(0.9)
                if hist.plot_object.GetMinimum() == 0.:
                    hist.plot_object.SetMinimum(0.9)
                canvas.SetLogy()
                canvas.Update()
            is_first = False
            if plot_config.normalise:
                hist_defs[0][1].plot_object.SetMaximum(plot_config.ymax)
        canvas.Update()
        return canvas

    if not switchOff and plot_config.ymax is None:
        max_y = 1.4 * max([item[1].GetMaximum() for item in hist_defs])
    if plot_config.ordering is not None:
        hist_defs = sorted(hist_defs, key=lambda k: plot_config.ordering.index(k[0]))
    for process, hist in hist_defs:
        index = list(map(itemgetter(1), hist_defs)).index(hist)
        hist = format_hist(hist, plot_config)

        try:
            process_config = find_process_config(process, process_configs)
        except AttributeError:
            process_config = None
        if not (plot_config.is_set_to_value("ignore_style", True)) and \
                plot_config.is_set_to_value("ignore_style", False):
            draw_option = get_draw_option_as_root_str(plot_config, process_config)
        else:
            draw_option = "hist"
        if not is_first and "same" not in draw_option:
            draw_option += "sames"
        hist.Draw(draw_option)
        fm.apply_style(hist, plot_config, process_config, index=index)
        if is_first:
            if isinstance(hist, ROOT.TH2) and draw_option.lower() == "colz":
                canvas.SetRightMargin(0.15)
            format_hist(hist, plot_config)
            if plot_config.ymax:
                fm.set_maximum_y(hist, plot_config.ymax)  # hist.SetMaximum(plot_config.ymax)
            else:
                hist.SetMaximum(hist.GetMaximum() * 1.2)
            if plot_config.ymin:
                fm.set_minimum_y(hist, plot_config.ymin)
        if plot_config.logy:
            hist.SetMaximum(hist.GetMaximum() * 100.)
            if plot_config.ymin > 0.:
                hist.SetMinimum(plot_config.ymin)
            else:
                hist.SetMinimum(0.9)
            if hist.GetMinimum() == 0.:
                hist.SetMinimum(0.9)
            canvas.SetLogy()
            canvas.Update()
        is_first = False
    if plot_config.normalise:
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
            var_string = "{:s} = {:.2f} \\pm {:.2f}".format(var.GetName(), var.getValV(), var.getError())
            fm.add_text_to_canvas(canvas, var_string, pos={'x': 0.15, 'y': 0.9 - i * 0.05}, size=0.04, color=None)
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
        if isinstance(style_attr, dict):
            for ss, attr in style_attr.items():
                getattr(obj, "Set" + ss + "Style")(attr)
        else:
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


def plot_graph(graph, plot_config, **kwargs):
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
    kwargs.setdefault('index', 0)
    kwargs.setdefault('canvas_name', plot_config.name)
    kwargs.setdefault('canvas_title', '')
    canvas = retrieve_new_canvas(kwargs['canvas_name'], kwargs['canvas_title'])
    canvas.cd()
    draw_option = 'a' + get_draw_option_as_root_str(plot_config)
    if plot_config.ymax is not None:
        fm.set_range_y(graph, plot_config.ymin, plot_config.ymax)
    graph.Draw(draw_option)
    graph = format_obj(graph, plot_config)
    apply_style(graph, *get_style_setters_and_values(plot_config, index=kwargs['index']))
    ROOT.SetOwnership(graph, False)
    if plot_config.logy:
        canvas.SetLogy()

    canvas.Update()
    return canvas


def add_graph_to_canvas(canvas, graph, plot_config, index=None):
    """
    Add a TGraph or associated object to existing canvas
    :param canvas: canvas containing at least on graph
    :type canvas: ROOT.TCanvas
    :param graph: graph object to be added
    :type graph: TGraph (or in Inheritance scheme)
    :param plot_config: plot style configuration
    :type plot_config: PlotConfig
    :param index: index identifier for ith graph object in canvas for style choice (optional)
    :type index: int
    :return: nothing
    :rtype: None
    """
    canvas.cd()
    draw_option = get_draw_option_as_root_str(plot_config)
    if "same" not in draw_option:
        draw_option += "same"
    draw_option = draw_option.lstrip('a')
    apply_style(graph, *get_style_setters_and_values(plot_config, index=index))
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
    if isinstance(hists, dict) or isinstance(hists, defaultdict):
        hist_defs = list(hists.items())
    elif isinstance(hists, list):
        hist_defs = list(zip([None] * len(hists), hists))
    else:
        _logger.error('Cannot deal with provided input {:s}'.format(hists.__str__()))
        raise InvalidInputError()
    stack = ROOT.THStack('hs', '')

    ROOT.SetOwnership(stack, False)
    data = None
    if plot_config.ordering is not None:
        hist_defs = apply_ordering(hist_defs, plot_config.ordering)
    for index, histograms in enumerate(hist_defs):
        process, hist = histograms
        try:
            if "data" in process.lower():
                # todo: problem if two distinct data sets
                data = (process, hist)
                continue
        except AttributeError:
            pass
        hist = format_hist(hist, plot_config)
        process_config = find_process_config(process, process_configs)
        draw_option = get_draw_option_as_root_str(plot_config, process_config)
        fm.apply_style(hist, plot_config, process_config, index)
        stack.Add(hist, draw_option)
    stack.Draw()
    canvas.Update()
    format_hist(stack, plot_config)
    min_y, max_y = fm.get_min_max_y(canvas, plot_config)
    fm.set_range(stack, min_y, max_y)
    if data is not None:
        add_data_to_stack(canvas, data[1], plot_config)
    if plot_config.logy:
        canvas.SetLogy()
    if plot_config.logx:
        try:
            xmin, xmax = plot_config.xmin, plot_config.xmax
        except AttributeError:
            xmin, xmax = max(0.0001, stack.GetXaxis().GetXmin()), stack.GetXaxis().GetXmax()
        fm.set_range_x(stack, xmin, xmax)
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
    """
    Apply blinding to a given distribution above blind cut value
    :param data: histogram
    :type data: TH1X
    :param blind: cut value above which distribution should be blinded
    :type blind: float
    :return: nothing
    :rtype: None
    """
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
    _logger.error('add_ratio_to_canvas moved to RatioPlotter. Please update your code')
    return None  # RatioPlotter.add_ratio_to_canvas(canvas, ratio, y_min, y_max, y_title, name, title)
