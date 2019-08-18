from PyAnalysisTools.base import _logger
from PyAnalysisTools.base.YAMLHandle import YAMLLoader
from PyAnalysisTools.AnalysisTools.FakeEstimator import MuonFakeEstimator
from PyAnalysisTools.AnalysisTools.RegionBuilder import RegionBuilder
from PyAnalysisTools.AnalysisTools.SubtractionHandle import SubtractionHandle
from PyAnalysisTools.AnalysisTools.ProcessFilter import ProcessFilter
from PyAnalysisTools.AnalysisTools.RegionSummaryModule import RegionSummaryModule
from PyAnalysisTools.AnalysisTools.ExtrapolationModule import ExtrapolationModule, TopExtrapolationModule, QCDExtrapolationModule


class Module(object):
    pass


def load_modules(config_files, callee):
    modules = []
    try:
        for cfg_file in config_files:
            if cfg_file is None:
                continue
            config = YAMLLoader.read_yaml(cfg_file)
            modules += build_module_instances(config, callee)
    except TypeError:
        _logger.warning("Config files not iterable")
    return modules


def build_module_instances(config, callee):
    modules = []
    for module, mod_config in config.iteritems():
        mod_name = globals()[module]
        for key, val in mod_config.iteritems():
            if isinstance(val, str) and "callee" in val:
                mod_config[key] = eval(val)
        instance = mod_name(**mod_config)
        modules.append(instance)
    return modules