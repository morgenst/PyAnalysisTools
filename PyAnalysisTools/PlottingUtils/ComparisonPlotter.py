import ROOT
import PyAnalysisTools.PlottingUtils.PlottingTools as PT
import PyAnalysisTools.PlottingUtils.Formatting as FM
from PyAnalysisTools.base import _logger, InvalidInputError
from PyAnalysisTools.ROOTUtils.FileHandle import FileHandle
from PyAnalysisTools.base.OutputHandle import OutputFileHandle
from PyAnalysisTools.PlottingUtils.PlotConfig import parse_and_build_plot_config


class ComparisonPlotter(object):
    def __init__(self, **kwargs):
        if not "input_files" in kwargs:
            _logger.error("No input files provided")
            raise InvalidInputError("Missing input files")
        if not "reference_file" in kwargs:
            _logger.error("No reference file provided")
            raise InvalidInputError("Missing reference")
        if not "config_file" in kwargs:
            _logger.error("No config file provided")
            raise InvalidInputError("Missing config")
        if not "output_dir" in kwargs:
            _logger.warning("No output directory given. Using ./")
        kwargs.setdefault("output_dir", "./")
        self.input_files = kwargs["input_files"]
        self.reference_file = kwargs["reference_file"]
        self.config_file = kwargs["config_file"]
        self.reference_file_handle = FileHandle(self.reference_file)
        self.file_handles = [FileHandle(file_name) for file_name in self.input_files]
        self.output_handle = OutputFileHandle(overload="comparison", output_file_name="Compare.root", **kwargs)
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

    @staticmethod
    def calculate_ratio(hist, reference):
        ratio_hist = hist.Clone("ratio_" + hist.GetName())
        ratio_hist.Divide(reference)
        FM.set_title_y(ratio_hist, "ratio")
        return ratio_hist

    def calculate_ratios(self, hists, reference, plot_config):
        plot_config.name = "ratio_" + plot_config.name
        plot_config.draw = "Marker"
        ratios = [self.calculate_ratio(hist, reference) for hist in hists]
        canvas = PT.plot_histograms_simple(ratios, plot_config)
        return canvas

    def make_comparison_plot(self, plot_config):
        reference_hist = self.reference_file_handle.get_object_by_name(plot_config.dist)
        hists = [fh.get_object_by_name(plot_config.dist) for fh in self.file_handles]
        canvas = PT.plot_hist(reference_hist, plot_config)
        for hist in hists:
            hist.SetName(hist.GetName() + "_%i" % hists.index(hist))
            plot_config.color = self.color_palette[hists.index(hist)]
            PT.add_histogram_to_canvas(canvas, hist, plot_config)
        FM.decorate_canvas(canvas, self.common_config)
        labels = ["reference"] + [""] * len(hists)
        if hasattr(self.common_config, "labels"):
            labels = self.common_config.labels
        if len(labels) != len(hists) + 1:
            _logger.error("Not enough labels provided. Received %i labels for %i histograms" % (len(labels),
                                                                                               len(hists) + 1))
            labels += [""] * (len(hists) - len(labels))
        FM.add_legend_to_canvas(canvas, labels=labels, xl=0.3, xh=0.5)
        if self.common_config.stat_box:
            FM.add_stat_box_to_canvas(canvas)
        canvas_ratio = self.calculate_ratios(hists, reference_hist, plot_config)
        canvas_combined = PT.add_ratio_to_canvas(canvas, canvas_ratio)
        self.output_handle.register_object(canvas)
        self.output_handle.register_object(canvas_combined)

    def compare_objects(self):
        self.parse_config()
        if hasattr(self.common_config, "colors"):
            self.update_color_palette()
        for plot_config in self.plot_configs:
            self.make_comparison_plot(plot_config)