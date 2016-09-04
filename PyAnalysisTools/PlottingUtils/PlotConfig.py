import ROOT
from PyAnalysisTools.base import _logger, InvalidInputError
from PyAnalysisTools.base.YAMLHandle import YAMLLoader


class PlotConfig(object):
    def __init__(self, **kwargs):
        if "dist" not in kwargs and "is_common" not in kwargs:
            _logger.error("Plot config does not contain distribution. Add dist key")
            InvalidInputError("No distribution provided")
        kwargs.setdefault("cuts", None)
        kwargs.setdefault("draw", "hist")
        for k,v in kwargs.iteritems():
            setattr(self, k.lower(), v)


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
    print "parse and buld"
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
    if process_config and hasattr(process_config, "draw"):
        draw_option = process_config.draw
    if hasattr(plot_config, "draw"):
        draw_option = plot_config.draw
    return draw_option


def get_draw_option_as_root_str(plot_config, process_config = None):
    draw_option = _parse_draw_option(plot_config, process_config)
    if draw_option == "Marker":
        draw_option = "p"
    elif draw_option == "Line":
        draw_option = "l"
    return draw_option

def get_style_setters_and_values(plot_config, process_config = None):
    style_setter = None
    style_attr, color = None, None
    draw_option = _parse_draw_option(plot_config, process_config)
    if hasattr(process_config, "style"):
        style_attr = process_config.style
    if hasattr(process_config, "color"):
        color = process_config.color
        if isinstance(color, str):
            color = getattr(ROOT, color)
    if hasattr(plot_config, "color"):
        color = plot_config.color
        if isinstance(color, str):
            color = getattr(ROOT, color)
    if draw_option == "Hist":
        style_setter = "Fill"
    elif draw_option == "Marker":
        style_setter = "Marker"
    elif draw_option == "Line":
        style_setter = "Line"
    return style_setter, style_attr, color