from PyAnalysisTools.base import _logger, InvalidInputError
from PyAnalysisTools.PlottingUtils.PlotConfig import parse_and_build_plot_config, parse_and_build_process_config,\
    merge_plot_configs, propagate_common_config
from PyAnalysisTools.PlottingUtils import Formatting as FM
from PyAnalysisTools.PlottingUtils import set_batch_mode


class BasePlotter(object):
    def __init__(self, **kwargs):
        self.plot_configs = None
        self.lumi = None
        kwargs.setdefault("batch", True)
        kwargs.setdefault("process_config_file", None)
        for attr, value in kwargs.iteritems():
            setattr(self, attr.lower(), value)
        set_batch_mode(kwargs["batch"])
        self.process_configs = self.parse_process_config()
        self.parse_plot_config()
        self.load_atlas_style()
        self.event_yields = {}

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
        if self.lumi is None and hasattr(common_config, "lumi"):
            self.lumi = common_config.lumi
        if common_config is not None:
            propagate_common_config(common_config, self.plot_configs)

    @staticmethod
    def load_atlas_style():
        FM.load_atlas_style()

    def read_cutflows(self):
        for file_handle in self.file_handles:
            if file_handle.process in self.event_yields:
                self.event_yields[file_handle.process] += file_handle.get_number_of_total_events()
            else:
                self.event_yields[file_handle.process] = file_handle.get_number_of_total_events()
