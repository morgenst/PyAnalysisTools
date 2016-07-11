__author__ = 'marcusmorgenstern'
__mail__ = ''

from PyAnalysisTools.base import _logger, InvalidInputError
from PyAnalysisTools.base.YAMLHandle import YAMLLoader


class PlotConfig(object):
    def __init__(self, **kwargs):
        if "dist" not in kwargs:
            _logger.error("Plot config does not contain distribution. Add dist key")
            InvalidInputError("No distribution provided")
        kwargs.setdefault("cuts", None)
        for k,v in kwargs.iteritems():
            setattr(self, k, v)


def parse_and_build_plot_config(config_file):
    try:
        parsed_config = YAMLLoader().read_yaml(config_file)
        plot_configs = [PlotConfig(name=k, **v) for k,v in parsed_config.iteritems()]
        return plot_configs
    except Exception as e:
        raise