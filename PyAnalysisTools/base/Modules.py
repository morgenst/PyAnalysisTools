from PyAnalysisTools.base.YAMLHandle import YAMLLoader
from PyAnalysisTools.AnalysisTools.FakeEstimator import MuonFakeEstimator
from PyAnalysisTools.AnalysisTools.RegionBuilder import RegionBuilder


class Module(object):
    pass


def load_modules(config_file, callee):
    config = YAMLLoader.read_yaml(config_file)
    return build_module_instances(config, callee)


def build_module_instances(config, callee):
    modules = []
    for module, mod_config in config.iteritems():
        mod_name = globals()[module]
        for key, val in mod_config.iteritems():
            if "callee" in val:
                mod_config[key] = eval(val)
        instance = mod_name(**mod_config)
        modules.append(instance)
    return modules