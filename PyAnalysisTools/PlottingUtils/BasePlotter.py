from PyAnalysisTools.base import _logger, InvalidInputError
from PyAnalysisTools.PlottingUtils.PlotConfig import parse_and_build_plot_config, parse_and_build_process_config,\
    merge_plot_configs, propagate_common_config


class BasePlotter(object):
    def __init__(self, **kwargs):
        for attr, value in kwargs.iteritems():
            setattr(self, attr.lower(), value)
        self.process_configs = self.parse_process_config()
        self.parse_plot_config()
        #todo: temporary assignment for naming collision in Plotter and ComparisionPlotter
        self.plot_config = self.plot_configs

    def parse_process_config(self):
        if self.process_config_file is None:
            return None
        process_config = parse_and_build_process_config(self.process_config_file)
        return process_config

    def parse_plot_config(self):
        _logger.debug("Try to parse plot config file")
        unmerged_plot_configs = []
        for plot_config_file in self.plot_config_files:
            unmerged_plot_configs.append(parse_and_build_plot_config(plot_config_file))
        self.plot_configs, common_config = merge_plot_configs(unmerged_plot_configs)
        if not hasattr(self, "lumi") and hasattr(common_config, "lumi"):
            self.lumi = common_config.lumi
        propagate_common_config(common_config, self.plot_configs)
