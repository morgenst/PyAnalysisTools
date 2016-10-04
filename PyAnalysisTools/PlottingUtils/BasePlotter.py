import ROOT
from PyAnalysisTools.base import _logger, InvalidInputError
from PyAnalysisTools.PlottingUtils.PlotConfig import parse_and_build_plot_config, parse_and_build_process_config, \
    get_histogram_definition
from PyAnalysisTools.base.YAMLHandle import YAMLLoader
from PyAnalysisTools.ROOTUtils.FileHandle import FileHandle
from PyAnalysisTools.PlottingUtils import set_batch_mode
from PyAnalysisTools.PlottingUtils import Formatting as FM
from PyAnalysisTools.PlottingUtils import HistTools as HT
from PyAnalysisTools.PlottingUtils import PlottingTools as PT
from PyAnalysisTools.AnalysisTools.XSHandle import XSHandle
from PyAnalysisTools.ROOTUtils.ObjectHandle import merge_objects_by_process_type
from PyAnalysisTools.AnalysisTools.StatisticsTools import get_significance
from PyAnalysisTools.base.OutputHandle import OutputFileHandle


class BasePlotter(object):
    def __init__(self, **kwargs):
        if "input_files" not in kwargs:
            _logger.error("No input files provided")
            raise  InvalidInputError("No input files")
        if "plot_config_file" not in kwargs:
            _logger.error("No plot config file provided. Nothing to parse")
            raise InvalidInputError("No plot config file")
        if "process_config_file" not in kwargs:
            _logger.warning("No process config file provided. Unable to read process specific options.")
        if "xs_config_file" not in kwargs:
            _logger.error("No cross section file provided. No scaling will be applied.")
            kwargs.setdefault("xs_config_file", None)
        kwargs.setdefault("systematics", None)
        kwargs.setdefault("process_config_file", None)
        kwargs.setdefault("xs_config_file", None)
        kwargs.setdefault("batch", True)
        kwargs.setdefault("output_file_name", "plots.root")
        for k,v in kwargs.iteritems():
            setattr(self, k, v)
        set_batch_mode(kwargs["batch"])
        self.file_handles = [FileHandle(file_name=input_file, dataset_info=kwargs["xs_config_file"]) for input_file in self.input_files]
        self.xs_handle = XSHandle(kwargs["xs_config_file"])
        self.process_config = self.parse_process_config()
        FM.load_atlas_style()
        self.histograms = {}
        self.output_handle = OutputFileHandle(**kwargs)

    def parse_plot_config(self):
        _logger.debug("Try to parse plot config file")
        self.plot_configs, self.common_config = parse_and_build_plot_config(self.plot_config_file)
        if not hasattr(self, "lumi"):
            self.lumi = self.common_config.lumi

    def parse_process_config(self):
        if self.process_config_file is None:
            return None
        process_config = parse_and_build_process_config(self.process_config_file)
        return process_config

    def initialise(self):
        self.parse_plot_config()

    def retrieve_histogram(self, file_handle, plot_config):
        hist = get_histogram_definition(plot_config)
        try:
            weight = None
            if not file_handle.process == "Data":
                weight="pileup_weight"
            file_handle.fetch_and_link_hist_to_tree(self.tree_name, hist, plot_config.dist, plot_config.cuts,
                                                    tdirectory=self.systematics, weight=weight)
            hist.SetName(hist.GetName() + "_" + file_handle.process)
            _logger.debug("try to access config for process %s" % file_handle.process)
            if not self.process_config[file_handle.process].type == "Data":
                cross_section_weight = self.xs_handle.get_lumi_scale_factor(file_handle.process, self.lumi,
                                                                            file_handle.get_number_of_total_events())
                HT.scale(hist, cross_section_weight)
        except Exception as e:
            raise e
        return hist

    def merge_histograms(self):
        def merge(histograms):
            for process, process_config in self.process_config.iteritems():
                if not hasattr(process_config, "subprocesses"):
                    continue
                for sub_process in process_config.subprocesses:
                    if sub_process not in histograms.keys():
                        continue
                    if process not in histograms.keys():
                        new_hist_name = histograms[sub_process].GetName().replace(sub_process, process)
                        histograms[process] = histograms[sub_process].Clone(new_hist_name)
                    else:
                        histograms[process].Add(histograms[sub_process])
                    histograms.pop(sub_process)
        for plot_config, histograms in self.histograms.iteritems():
            merge(histograms)

    def calculate_ratios(self, hists, plot_config):
        ratio = hists["Data"].Clone("ratio_%s" % plot_config.dist)
        mc = None
        for key, hist in hists.iteritems():
            if key == "Data":
                continue
            if mc is None:
                mc = hist.Clone("mc_total_%s" % plot_config.dist)
                continue
            mc.Add(hist)
        ratio.Divide(mc)
        plot_config.name = "ratio_" + plot_config.name
        plot_config.ytitle = "data/MC"
        if "unit" in plot_config.__dict__.keys():
            plot_config.__dict__.pop("unit")
        plot_config.draw = "Marker"
        #ratios = [self.calculate_ratio(hist, reference) for hist in hists]
        canvas = PT.plot_hist(ratio, plot_config)
        return canvas

    def make_plots(self):
        for file_handle in self.file_handles:
            for plot_config in self.plot_configs:
                try:
                    if file_handle.process not in self.histograms[plot_config].keys():
                        self.histograms[plot_config][file_handle.process] = self.retrieve_histogram(file_handle,
                                                                                                    plot_config)
                    else:
                        self.histograms[plot_config][file_handle.process].Add(self.retrieve_histogram(file_handle,
                                                                                                      plot_config))
                except KeyError:
                    self.histograms[plot_config] = {file_handle.process: self.retrieve_histogram(file_handle,
                                                                                                 plot_config)}
        self.merge_histograms()
        for plot_config, data in self.histograms.iteritems():
            if self.common_config.outline == "hist":
                canvas = PT.plot_histograms(data, plot_config, self.common_config, self.process_config)
            elif self.common_config.outline == "stack":
                canvas = PT.plot_stack(data, plot_config, self.common_config, self.process_config)
            else:
                _logger.error("Unsupported outline option %s" % self.common_config.outline)
                raise InvalidInputError("Unsupported outline option")
            FM.decorate_canvas(canvas, self.common_config)
            FM.add_legend_to_canvas(canvas, process_configs=self.process_config)
            if hasattr(plot_config, "calcsig"):
                #todo: "Background" should be an actual type
                signal_hist = merge_objects_by_process_type(canvas, self.process_config, "Signal")
                background_hist = merge_objects_by_process_type(canvas, self.process_config, "Background")
                significance_hist = get_significance(signal_hist, background_hist)
                canvas_significance_ratio = PT.add_ratio_to_canvas(canvas, significance_hist)
            self.output_handle.register_object(canvas)
            if hasattr(plot_config, "ratio") or hasattr(self.common_config, "ratio"):
                if self.common_config.ratio:
                    canvas_ratio = self.calculate_ratios(data, plot_config)
                    canvas_combined = PT.add_ratio_to_canvas(canvas, canvas_ratio)
                    self.output_handle.register_object(canvas_combined)
