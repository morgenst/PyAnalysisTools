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