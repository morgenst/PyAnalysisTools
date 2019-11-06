from builtins import object
from PyAnalysisTools.base import _logger
from PyAnalysisTools.base.YAMLHandle import YAMLLoader
from PyAnalysisTools.AnalysisTools.FakeEstimator import MuonFakeEstimator  # noqa: F401
from PyAnalysisTools.AnalysisTools.RegionBuilder import RegionBuilder  # noqa: F401
from PyAnalysisTools.AnalysisTools.SubtractionHandle import SubtractionHandle  # noqa: F401
from PyAnalysisTools.AnalysisTools.ProcessFilter import ProcessFilter  # noqa: F401
from PyAnalysisTools.AnalysisTools.RegionSummaryModule import RegionSummaryModule  # noqa: F401
from PyAnalysisTools.AnalysisTools.ExtrapolationModule import ExtrapolationModule, TopExtrapolationModule  # noqa: F401
from PyAnalysisTools.AnalysisTools.ExtrapolationModule import QCDExtrapolationModule  # noqa: F401


class Module(object):
    pass


def load_modules(config_files, callee):
    """
    Load all modules defined in configuration files
    :param config_files: list of configuration file names
    :type config_files: list
    :param callee: instance of calling object
    :type callee: object
    :return: list of initialised modules
    :rtype: list
    """
    modules = []
    if not isinstance(config_files, list):
        config_files = [config_files]
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
    """
    Construct a specific module instance from a config. If the callee is needed for the initialisation needs to be
    passed and reflected in config
    :param config: module configuration
    :type config: dict
    :param callee: optional calling class
    :type callee: object
    :return:
    :rtype:
    """
    modules = []
    for module, mod_config in list(config.items()):
        mod_name = globals()[module]
        for key, val in list(mod_config.items()):
            if isinstance(val, str) and 'callee' in val:
                mod_config[key] = eval(val)
        instance = mod_name(**mod_config)
        modules.append(instance)
    return modules
