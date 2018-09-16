import ROOT
import math
import re
from math import log10
from array import array
from copy import copy, deepcopy
from PyAnalysisTools.base import _logger, InvalidInputError
from PyAnalysisTools.base.YAMLHandle import YAMLLoader as yl


class PlotConfig(object):
    def __init__(self, **kwargs):
        if "dist" not in kwargs and "is_common" not in kwargs:
            _logger.debug("Plot config does not contain distribution. Add dist key")
        kwargs.setdefault("cuts", None)
        if not "draw" in kwargs:
            kwargs.setdefault("Draw", "hist")
        kwargs.setdefault("outline", "hist")
        kwargs.setdefault("stat_box", False)
        kwargs.setdefault("weight", None)
        kwargs.setdefault("normalise", False)
        kwargs.setdefault("dist", None)
        kwargs.setdefault("merge", True)
        kwargs.setdefault("no_data", False)
        kwargs.setdefault("ignore_style", False)
        kwargs.setdefault("style", None)
        kwargs.setdefault("rebin", None)
        kwargs.setdefault("ratio", None)
        kwargs.setdefault("ignore_rebin", False)
        kwargs.setdefault("weight", False)
        kwargs.setdefault("enable_legend", False)
        kwargs.setdefault("blind", None)
        kwargs.setdefault("legend_options", dict())
        kwargs.setdefault("make_plot_book", False)
        kwargs.setdefault("is_multidimensional", False)
        kwargs.setdefault("ordering", None)
        kwargs.setdefault("y_min", 0.)
        kwargs.setdefault("ymin", 0.)
        kwargs.setdefault("xmin", None)
        kwargs.setdefault("draw_option", None)
        kwargs.setdefault("ymax", None)
        kwargs.setdefault("is_common", False)
        kwargs.setdefault("normalise_range", None)
        kwargs.setdefault("ytitle", None)
        kwargs.setdefault("ratio_config", None)
        kwargs.setdefault("grid", False)
        kwargs.setdefault("logy", False)
        kwargs.setdefault("logx", False)
        kwargs.setdefault("signal_scale", None)
        kwargs.setdefault("Lumi", 1.)
        kwargs.setdefault("signal_extraction", True)
        kwargs.setdefault("xtitle", None)
        kwargs.setdefault("merge_mc_campaigns", True)
        kwargs.setdefault("watermark", "Internal")
        kwargs.setdefault("watermark_x", 0.15)
        kwargs.setdefault("watermark_y", 0.96)
        for k, v in kwargs.iteritems():
            if k == "y_min" or k == "y_max":
                _logger.info("Deprecated. Use ymin or ymax")
            if k == "ratio_config" and v is not None:
                v["logx"] = kwargs["logx"]
                self.set_additional_config("ratio_config", **v)
                continue
            if k == "significance_config":
                self.set_additional_config("significance_config", **v)
                continue
            if "xmin" in k or "xmax" in k:
                v = eval(str(v))
            if (k == "ymax" or k == "ymin") and v is not None and (re.match("[1-9].*[e][1-9]*", str(v)) or "math." in str(v)):
                setattr(self, k.lower(), eval(v))
                continue
            setattr(self, k.lower(), v)
        self.is_multidimensional = False
        self.auto_decorate()

    def is_set_to_value(self, attr, value):
        """
        Checks if attribute is set to value
        :param attr: attribute
        :type attr: str
        :param value: value of attribute
        :type value: any
        :return: True/False
        :rtype: boolean
        """
        if not hasattr(self, attr):
            return False
        return getattr(self, attr) == value

    def set_additional_config(self, attr_name, **kwargs):
        kwargs.setdefault("name", "ratio")
        kwargs.setdefault("dist", "ratio")
        kwargs.setdefault("ignore_style", False)
        kwargs.setdefault("enable_legend", False)
        setattr(self, attr_name, PlotConfig(**kwargs))

    def __str__(self):
        """
        Overloaded str operator. Get's called if object is printed
        :return: formatted string with name and attributes
        :rtype: str
        """
        obj_str = "Plot config: {:s} \n".format(self.name)
        for attribute, value in self.__dict__.items():
            obj_str += '{}={} '.format(attribute, value)
        return obj_str

    def __eq__(self, other):
        """
        Comparison operator
        :param other: plot config object to compare to
        :type other: PlotConfig
        :return: True/False
        :rtype: boolean
        """
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return False

    def __ne__(self, other):
        """
        Comparison operator (negative)
        :param other: plot config object to compare to
        :type other: PlotConfig
        :return: True/False
        :rtype: boolean
        """
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.name)

    @staticmethod
    def get_overwritable_options():
        """
        Get properties which can be overwriten by specific plot config

        :return: overwritable properties
        :rtype: list
        """
        return ["outline", "make_plot_book", "no_data", "draw", "ordering", "signal_scale", "lumi", "normalise",
                "merge_mc_campaigns", "signal_extraction", "ratio", "cuts", "enable_legend"]

    def auto_decorate(self):
        if hasattr(self, "dist") and self.dist:
            self.is_multidimensional = True if ":" in self.dist.replace("::", "") else False
        if self.xtitle is None and hasattr(self, "dist") and self.dist:
            self.xtitle = self.dist
        elif self.xtitle is None:
            self.xtitle = ""

    def merge_configs(self, other):
        """
        Merge two plot configs

        :param other: plot config to be merged in self
        :type other: PlotConfig
        :return: None
        :rtype: None
        """
        for attr, val in other.__dict__.iteritems():
            if not hasattr(self, attr):
                setattr(self, attr, val)
                continue
            if getattr(self, attr) != val:
                dec = raw_input("Different settings for attribute {:s} in common configs. "
                                "Please choose 1) {:s} or 2) {:s}: ".format(attr, str(val), str(getattr(self, attr))))
                if dec == "1":
                    setattr(self, attr, val)
                elif dec == "2":
                    continue
                else:
                    _logger.warn("Invalid choice {:s}. Take {:s}".format(str(dec), str(getattr(self, attr))))


default_plot_config = PlotConfig(name=None)


def get_default_plot_config(hist):
    """
    Get plot config with default arguments and name according to histogram name
    :param hist: histogram object
    :type hist: THX
    :return: plot configuration
    :rtype: PlotConfig
    """
    return PlotConfig(name=hist.GetName())


def get_default_color_scheme():
    return [ROOT.kBlack, ROOT.kYellow-3, ROOT.kRed+2, ROOT.kTeal - 2, ROOT.kSpring-8, ROOT.kCyan, ROOT.kBlue-6,
            ROOT.kRed, ROOT.kGreen, ROOT.kBlue, ROOT.kGray]


class ProcessConfig(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.iteritems():
            setattr(self, k.lower(), v)
        self.transform_type()

    def transform_type(self):
        if "data" in self.type.lower():
            self.is_data = True
            self.is_mc = False
        else:
            self.is_data = False
            self.is_mc = True

    def retrieve_subprocess_config(self):
        tmp = {}
        if not hasattr(self, "subprocesses"):
            return tmp
        for sub_process in self.subprocesses:
            tmp[sub_process] = ProcessConfig(**dict((k, v) for (k, v) in self.__dict__.iteritems() if not k == "subprocesses"))
        return tmp

    def add_subprocess(self, subprocess_name):
        self.subprocesses.append(subprocess_name)
        return ProcessConfig(**dict((k, v) for (k, v) in self.__dict__.iteritems() if not k == "subprocesses"))


def expand_plot_config(plot_config):
    # if not isinstance(plot_config.dist, list):
    #     _logger.debug("tried to expand plot config with single distribution")
    #     return [plot_config]
    plot_configs = []
    if hasattr(plot_config, "cuts_ref"):
        if "dummy1" in plot_config.cuts_ref:
            #for cuts in plot_config.cuts_ref.values():
            for item in ["dummy1", "dummy2", "dummy3", "dummy4", "dummy5", "dummy6", "dummy7", "dummy8"]:
                if item not in plot_config.cuts_ref:
                    continue
                cuts = plot_config.cuts_ref[item]
                tmp_config = copy(plot_config)
                tmp_config.cuts = cuts
                plot_configs.append(tmp_config)
        else:
            for cut_name, cut in plot_config.cuts_ref.iteritems():
                cuts = plot_config.cuts_ref[cut_name]
                tmp_config = copy(plot_config)
                tmp_config.cuts = plot_config.cuts + cuts
                tmp_config.enable_cut_ref_merge = True
                tmp_config.name += "_{:s}_".format(cut_name)
                plot_configs.append(tmp_config)
    else:
        for dist in plot_config.dist:
            tmp_config = copy(plot_config)
            tmp_config.dist = dist
            plot_configs.append(tmp_config)
    return plot_configs


def parse_and_build_plot_config(config_file):
    try:
        parsed_config = yl.read_yaml(config_file)
        common_plot_config = None
        if "common" in parsed_config:
            common_plot_config = PlotConfig(name="common", is_common=True, **(parsed_config["common"]))
        plot_configs = [PlotConfig(name=k, **v) for k, v in parsed_config.iteritems() if not k=="common"]
        _logger.debug("Successfully parsed %i plot configurations." % len(plot_configs))
        return plot_configs, common_plot_config
    except Exception as e:
        raise


def parse_and_build_process_config(process_config_files):
    """
    Parse yml file containing process definition and build ProcessConfig object
    :param process_config_files: process configuration yml files
    :type process_config_files: list
    :return: Process config
    :rtype: ProcessConfig
    """
    if process_config_files is None:
        return None
    try:
        _logger.debug("Parsing process configs")
        if not isinstance(process_config_files, list):
            parsed_process_config = yl.read_yaml(process_config_files)
            process_configs = {k: ProcessConfig(name=k, **v) for k, v in parsed_process_config.iteritems()}
        else:
            parsed_process_configs = [yl.read_yaml(pcf) for pcf in process_config_files]
            process_configs = {k: ProcessConfig(name=k, **v) for parsed_config in parsed_process_configs
                               for k, v in parsed_config.iteritems()}
        for process_config in process_configs.values():
            process_configs.update(process_config.retrieve_subprocess_config())
        _logger.debug("Successfully parsed %i process items." % len(process_configs))
        return process_configs
    except Exception as e:
        raise e


def merge_plot_configs(plot_configs):
    merged_plot_config = None
    merged_common_config = None
    for plot_config, common_config in plot_configs:
        if merged_plot_config is None:
            merged_plot_config = plot_config
            merged_common_config = common_config
            continue
        merged_plot_config += plot_config
        merged_common_config.merge_configs(common_config)
    return merged_plot_config, merged_common_config


def propagate_common_config(common_config, plot_configs):
    """
    Propagate common config settings to all plot configs
    :param common_config: Common settings shared among plot configs
    :type common_config: PlotConfig
    :param plot_configs: all defined plot configs
    :type plot_configs: PlotConfig
    :return: Nothing
    :rtype: None
    """
    def integrate(plot_config, attr, value):
        if attr == "weight":
            if plot_config.weight is not None and not plot_config.weight.lower() == "none":
                plot_config.weight += " * {:s}".format(value)
            else:
                plot_config.weight = value
        if hasattr(plot_config, attr) and getattr(plot_config, attr) != getattr(default_plot_config, attr) and \
                attr not in PlotConfig.get_overwritable_options() or attr is None:
            return
        if value == getattr(default_plot_config, attr):
            return
        if attr == "ratio_config":
            plot_config.ratio_config = deepcopy(value)
            return
        setattr(plot_config, attr, value)

    for attr, value in common_config.__dict__.iteritems():
        for plot_config in plot_configs:
            integrate(plot_config, attr, value)


def _parse_draw_option(plot_config, process_config=None):
    draw_option = "Hist"
    if hasattr(plot_config, "draw"):
        draw_option = plot_config.draw
    if process_config and hasattr(process_config, "draw"):
        draw_option = process_config.draw
    return draw_option


def get_draw_option_as_root_str(plot_config, process_config=None):
    if plot_config.draw_option is not None:
        return plot_config.draw_option
    draw_option = _parse_draw_option(plot_config, process_config)
    if draw_option == "Marker":
        draw_option = "p"
    elif draw_option == "MarkerError":
        draw_option = "E"
    elif draw_option == "Line":
        draw_option = "l"
    elif draw_option == "hist":
        draw_option = "HIST"
    return draw_option


def transform_color(color, index=None):
    if isinstance(color, str):
        offset = 0
        if "+" in color:
            color, offset = color.split("+")
        if "-" in color:
            color, offset = color.split("-")
            offset = "-" + offset
        color = getattr(ROOT, color.rstrip()) + int(offset)

    if isinstance(color, list):
        return transform_color(color[index])
    return color


def get_style_setters_and_values(plot_config, process_config=None, index=None):
    style_setter = None
    style_attr, color = None, None
    draw_option = _parse_draw_option(plot_config, process_config)
    if hasattr(process_config, "style"):
        style_attr = process_config.style
    if hasattr(plot_config, "styles") and index is not None:
        style_attr = plot_config.styles[index]
    if hasattr(plot_config, "style"):
        style_attr = plot_config.style
    if hasattr(process_config, "color"):
        color = transform_color(process_config.color)
    if hasattr(plot_config, "color"):
        color = transform_color(plot_config.color, index)
        
    if draw_option.lower() == "hist" or re.match(r"e\d", draw_option.lower()):
        if hasattr(process_config, "format"):
            style_setter = process_config.format.capitalize()
        elif style_attr:
            style_setter = "Line"
        else:
            #style_setter = ["Line", "Marker", "Fill"]
            style_setter = ["Line"]
    elif draw_option.lower() == "marker" or draw_option.lower() == "markererror":
        style_setter = "Marker"
    elif draw_option.lower() == "line":
        style_setter = "Line"
    if hasattr(plot_config, "style_setter"):
        style_setter = plot_config.style_setter
    # else:
    #     style_attr = None
    if not isinstance(style_setter, list):
        style_setter = [style_setter]
    return style_setter, style_attr, color


def get_histogram_definition(plot_config):
    """
    Create histogram defintion based on plot configuration. Dimension is parsed counting : in the distribution. If no
    distribution is provided by default a one dimension histogram will be created
    :param plot_config: plot configuration with binning and name
    :type plot_config: PlotConfig
    :return: histogram
    :rtype: ROOT.THXF
    """
    if plot_config.dist is not None:
        dimension = plot_config.dist.replace("::", "").count(":")
    else:
        dimension = 0
    hist = None
    hist_name = plot_config.name
    if dimension == 0:
        if not plot_config.logx:
            hist = ROOT.TH1F(hist_name, "", plot_config.bins, plot_config.xmin, plot_config.xmax)
        else:
            logxmin = log10(plot_config.xmin)
            logxmax = log10(plot_config.xmax)
            binwidth = (logxmax - logxmin) / plot_config.bins
            xbins = []
            for i in range(0, plot_config.bins+1):
                xbins.append(plot_config.xmin + pow(10, logxmin + i * binwidth))
            hist = ROOT.TH1F(hist_name, "", plot_config.bins, array('d', xbins))
    elif dimension == 1:
        if isinstance(plot_config.xbins, list):
            hist = ROOT.TH2F(hist_name, "", len(plot_config.xbins) - 1, array("d", plot_config.xbins),
                             plot_config.ybins, plot_config.ymin, plot_config.ymax)
        else:
            hist = ROOT.TH2F(hist_name, "", plot_config.xbins, plot_config.xmin, plot_config.xmax,
                             plot_config.ybins, plot_config.ymin, plot_config.ymax)
    elif dimension == 2:
        hist = ROOT.TH3F(hist_name, "", plot_config.xbins, plot_config.xmin, plot_config.xmax,
                         plot_config.ybins, plot_config.ymin, plot_config.ymax,
                         plot_config.zbins, plot_config.zmin, plot_config.zmax)
    if not hist:
        _logger.error("Unable to create histogram for plot_config %s for variable %s" % (plot_config.name,
                                                                                         plot_config.dist))
        raise InvalidInputError("Invalid plot configuration")
    hist.Sumw2()
    return hist


def find_process_config(process_name, process_configs):
    if process_configs is None or process_name is None:
        return None
    if process_name in process_configs:
        return process_configs[process_name]
    regex_configs = dict(filter(lambda kv: hasattr(kv[1], "subprocesses") and
                                              any(map(lambda i: i.startswith("re."), kv[1].subprocesses)),
                           process_configs.iteritems()))
    for process_config in regex_configs.values():
        for sub_process in process_config.subprocesses:
            if not sub_process.startswith("re."):
                continue
            match = re.match(sub_process.replace("re.", ""), process_name)
            if not match:
                continue
            process_configs[match.group()] = process_config.add_subprocess(match.group())
            return process_configs[match.group()]
    return None


def expand_process_configs(processes, process_configs):
    for process in processes:
        _ = find_process_config(process, process_configs)
    return process_configs
