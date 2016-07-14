__author__ = 'marcusmorgenstern'
__mail__ = ''

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
            setattr(self, k, v)


class ProcessConfig(object):
    def __init__(self, **kwargs):
        self.decorate(**kwargs)

    def decorate(self, **kwargs):
        for k,v in kwargs.iteritems():
            setattr(self, k.lower(), v)


def parse_and_build_plot_config(config_file):
    try:
        parsed_config = YAMLLoader().read_yaml(config_file)
        plot_configs = [PlotConfig(name=k, **v) for k, v in parsed_config.iteritems() if not k=="common"]
        common_plot_config = PlotConfig(name="common", is_common=True, **(parsed_config["common"]))
        _logger.debug("Successfully parsed %i plot configurations." % len(plot_configs))
        return plot_configs, common_plot_config
    except Exception as e:
        raise


def parse_and_build_process_config(process_config_file, xs_config_file):
    try:
        _logger.debug("Parsing process config")
        parsed_process_config = YAMLLoader().read_yaml(process_config_file)
        parsed_xs_config = YAMLLoader().read_yaml(xs_config_file)
        process_configs = {k: ProcessConfig(name=k, **v) for k, v in parsed_process_config.iteritems()}
        for config in process_configs.values():
            if config.name not in parsed_process_config:
                _logger.warning("Could not find cross section entry for process %s" % config.name)
                continue
            config.decorate(**(parsed_xs_config[config.name]))
        _logger.debug("Successfully parsed %i process items." % len(process_configs))
        return process_configs
    except Exception as e:
        raise e