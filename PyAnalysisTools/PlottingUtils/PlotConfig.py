import ROOT
import re
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
        kwargs.setdefault("make_plot_book", False)
        kwargs.setdefault("is_multidimensional", False)
        for k,v in kwargs.iteritems():
            setattr(self, k.lower(), v)
        self.auto_decorate()

    def auto_decorate(self):
        if hasattr(self, "dist") and self.dist:
            self.is_multidimensional = True if ":" in self.dist else False


class ProcessConfig(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.iteritems():
            setattr(self, k.lower(), v)

    def retrieve_subprocess_config(self):
        tmp = {}
        if not hasattr(self, "subprocesses"):
            return tmp
        for sub_process in self.subprocesses:
            tmp[sub_process] = ProcessConfig(**dict((k, v) for (k, v) in self.__dict__.iteritems() if not k == "subprocesses"))
        return tmp


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


def _parse_draw_option(plot_config, process_config):
    draw_option = "Hist"
    if hasattr(plot_config, "draw"):
        draw_option = plot_config.draw
    if process_config and hasattr(process_config, "draw"):
        draw_option = process_config.draw
    return draw_option


def get_draw_option_as_root_str(plot_config, process_config = None):
    draw_option = _parse_draw_option(plot_config, process_config)
    if draw_option == "Marker":
        draw_option = "p"
    elif draw_option == "Line":
        draw_option = "l"
    return draw_option


def get_style_setters_and_values(plot_config, process_config = None):
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