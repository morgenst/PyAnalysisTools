import ROOT
import PyAnalysisTools.PlottingUtils.PlottingTools as PT
import PyAnalysisTools.PlottingUtils.Formatting as FM
from PyAnalysisTools.base import _logger
from PyAnalysisTools.ROOTUtils.FileHandle import FileHandle
from PyAnalysisTools.PlottingUtils.PlotConfig import parse_and_build_plot_config


class ComparisonPlotter(object):
    def __init__(self, input_files, reference_file, config):
        self.input_files = input_files
        self.reference_file = reference_file
        self.config_file = config
        self.reference_file_handle = FileHandle(reference_file)
        self.file_handles = [FileHandle(file_name) for file_name in input_files]
        self.color_palette = [ROOT.kRed, ROOT.kBlue, ROOT.kGreen, ROOT.kCyan]

    def parse_config(self):
        self.plot_configs, self.common_config = parse_and_build_plot_config(self.config_file)

    def update_color_palette(self):
        if isinstance(self.common_config.colors[0], str):
            self.color_palette = [getattr(ROOT, "k" + color.capitalize()) for color in self.common_config.colors]
        elif isinstance(self.common_config.colors[0], int):
            self.color_palette = [color for color in self.common_config.colors]
        else:
            _logger.warning("Unsuppored type %s for colors in common_config" % type(self.common_config.colors[0]))

    def make_comparison_plot(self, plot_config):
        reference_hist = self.reference_file_handle.get_object_by_name(plot_config.dist)
        hists = [fh.get_object_by_name(plot_config.dist) for fh in self.file_handles]
        canvas = PT.plot_hist(reference_hist, plot_config)
        for hist in hists:
            hist.SetName(hist.GetName() + "_%i" % hists.index(hist))
            plot_config.color = self.color_palette[hists.index(hist)]
            PT.add_histogram_to_canvas(canvas, hist, plot_config)
        labels = ["reference"] + [""] * len(hists)
        if hasattr(self.common_config, "labels"):
            labels = self.common_config.labels
        if len(labels) != len(hists) + 1:
            _logger.error("Not enough labels provied. Received %i labels for %i histograms" % (len(labels),
                                                                                               len(hists) + 1))
            labels += [""] * (len(hists) - len(labels))
        FM.add_legend_to_canvas(canvas, labels=labels)

    def compare_objects(self):
        self.parse_config()
        if hasattr(self.common_config, "colors"):
            self.update_color_palette()

        for plot_config in self.plot_configs:
            self.make_comparison_plot(plot_config)