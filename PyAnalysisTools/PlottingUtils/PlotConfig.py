import ROOT
import re
from copy import copy
from PyAnalysisTools.base import _logger, InvalidInputError
from PyAnalysisTools.base.YAMLHandle import YAMLLoader


class PlotConfig(object):
    def __init__(self, **kwargs):
        if "dist" not in kwargs and "is_common" not in kwargs:
            _logger.error("Plot config does not contain distribution. Add dist key")
            InvalidInputError("No distribution provided")
        kwargs.setdefault("cuts", None)
        kwargs.setdefault("draw", "hist")
        kwargs.setdefault("outline", "hist")
        kwargs.setdefault("stat_box", False)
        kwargs.setdefault("weight", None)
        kwargs.setdefault("normalise", False)
        kwargs.setdefault("merge", True)
        kwargs.setdefault("no_data", False)
        kwargs.setdefault("ignore_style", False)
        kwargs.setdefault("weight", False)
        kwargs.setdefault("blind", None)
        kwargs.setdefault("legend_options", None)
        kwargs.setdefault("make_plot_book", False)
        kwargs.setdefault("is_multidimensional", False)
        kwargs.setdefault("ordering", None)
        kwargs.setdefault("y_min", 0.)
        kwargs.setdefault("ymin", 0.)
        for k,v in kwargs.iteritems():
            if k == "y_min" or k == "ymax":
                _logger.info("Deprecated. Use ymin or ymax")
            if k == "ratio_config":
                self.set_ratio_config(**v)
                continue

            setattr(self, k.lower(), v)
        self.auto_decorate()

    def is_set_to_value(self, attr, value):
        if not hasattr(self, attr):
            return False
        return getattr(self, attr) == value

    def set_ratio_config(self, **kwargs):
        kwargs.setdefault("name", "ratio")
        kwargs.setdefault("dist", "ratio")
        kwargs.setdefault("ignore_style", False)
        self.ratio_config = PlotConfig(**kwargs)

    def auto_decorate(self):
        if hasattr(self, "dist") and self.dist:
            self.is_multidimensional = True if ":" in self.dist else False

    def _merge(self, other):
        for attr, val in other.__dict__.iteritems():
            if not hasattr(self, attr):
                setattr(self, attr, val)
                continue
            if getattr(self, attr) != val:
                _logger.warn("Different settings for attrinute {:s} in common configs: {:s} vs. {:s}".format(attr, val,
                                                                                                             getattr(self, attr)))


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
    if not isinstance(plot_config.dist, list):
        _logger.debug("tried to expand plot config with single distribution")
        return [plot_config]
    plot_configs = []
    for dist in plot_config.dist:
        tmp_config = copy(plot_config)
        tmp_config.dist = dist
        plot_configs.append(tmp_config)
    return plot_configs

def parse_and_build_plot_config(config_file):
    try:
        parsed_config = YAMLLoader.read_yaml(config_file)
        plot_configs = [PlotConfig(name=k, **v) for k, v in parsed_config.iteritems() if not k=="common"]
        common_plot_config = None
        if "common" in parsed_config:
            common_plot_config = PlotConfig(name="common", is_common=True, **(parsed_config["common"]))
        _logger.debug("Successfully parsed %i plot configurations." % len(plot_configs))
        return plot_configs, common_plot_config
    except Exception as e:
        raise


def parse_and_build_process_config(process_config_file):
    try:
        _logger.debug("Parsing process config")
        parsed_process_config = YAMLLoader.read_yaml(process_config_file)
        process_configs = {k: ProcessConfig(name=k, **v) for k, v in parsed_process_config.iteritems()}
        for process_config in process_configs.values():
            process_configs.update(process_config.retrieve_subprocess_config())
        _logger.debug("Successfully parsed %i process items." % len(process_configs))
        return process_configs
    except Exception as e:
        print e
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
        merged_common_config._merge(common_config)
    return merged_plot_config, merged_common_config


def _parse_draw_option(plot_config, process_config):
    draw_option = "Hist"
    if hasattr(plot_config, "draw"):
        draw_option = plot_config.draw
    if process_config and hasattr(process_config, "draw"):
        draw_option = process_config.draw
    return draw_option


def get_draw_option_as_root_str(plot_config, process_config=None):
    draw_option = _parse_draw_option(plot_config, process_config)
    if draw_option == "Marker":
        draw_option = "p"
    elif draw_option == "Line":
        draw_option = "l"
    return draw_option


def get_style_setters_and_values(plot_config, process_config=None):
    def transform_color(color):
        if isinstance(color, str):
            offset = 0
            if "+" in color:
                color, offset = color.split("+")
            if "-" in color:
                color, offset = color.split("-")
            color = getattr(ROOT, color.rstrip()) + int(offset)
        return color

    style_setter = None
    style_attr, color = None, None
    draw_option = _parse_draw_option(plot_config, process_config)
    if hasattr(process_config, "style"):
        style_attr = process_config.style
    if hasattr(plot_config, "style"):
        style_attr = plot_config.style
    if hasattr(process_config, "color"):
        color = transform_color(process_config.color)
    if hasattr(plot_config, "color"):
        color = transform_color(plot_config.color)
    if draw_option.lower() == "hist" or re.match(r"e\d",draw_option.lower()):
        if style_attr:
            style_setter = "Fill"
        else:
            style_setter = "Line"
    elif draw_option.lower() == "marker":
        style_setter = "Marker"
    elif draw_option.lower() == "line":
        style_setter = "Line"
    # else:
    #     style_attr = None
    return style_setter, style_attr, color


def get_histogram_definition(plot_config):
    dimension = plot_config.dist.count(":")
    hist = None
    hist_name = plot_config.name
    if dimension == 0:
        hist = ROOT.TH1F(hist_name, "", plot_config.bins, plot_config.xmin, plot_config.xmax)
    elif dimension == 1:
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
    return hist


def find_process_config(process_name, process_configs):
    if process_configs is None:
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