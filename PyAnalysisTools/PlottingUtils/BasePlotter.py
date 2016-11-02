import ROOT
import copy
import dill
import pathos.multiprocessing as mp
from functools import partial
from PyAnalysisTools.base import _logger, InvalidInputError
from PyAnalysisTools.PlottingUtils.PlotConfig import parse_and_build_plot_config, parse_and_build_process_config, \
    get_histogram_definition
from PyAnalysisTools.base.YAMLHandle import YAMLLoader
from PyAnalysisTools.ROOTUtils.FileHandle import FileHandle
from PyAnalysisTools.PlottingUtils import set_batch_mode
from PyAnalysisTools.PlottingUtils import Formatting as FM
from PyAnalysisTools.PlottingUtils import HistTools as HT
from PyAnalysisTools.PlottingUtils import PlottingTools as PT
from PyAnalysisTools.PlottingUtils.PlotConfig import PlotConfig
from PyAnalysisTools.AnalysisTools.XSHandle import XSHandle
from PyAnalysisTools.ROOTUtils.ObjectHandle import merge_objects_by_process_type
from PyAnalysisTools.AnalysisTools import StatisticsTools as ST
from PyAnalysisTools.base.OutputHandle import OutputFileHandle
from PyAnalysisTools.ROOTUtils.ObjectHandle import get_objects_from_canvas_by_type


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
        self.statistical_uncertainty_hist = None
        self.histograms = {}
        self.initialise()
        self.output_handle = OutputFileHandle(make_plotbook=self.common_config.make_plot_book, **kwargs)

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
        self.ncpu = min(self.ncpu, len(self.plot_configs))

    def retrieve_histogram(self, file_handle, plot_config):
        file_handle.open()
        hist = get_histogram_definition(plot_config)
        hist.SetName(hist.GetName() + file_handle.process)
        try:
            weight = None
            selection_cuts = ""
            if self.common_config.weight and plot_config.weight is not None:
                weight = self.common_config.weight
            if self.common_config.cuts:
                selection_cuts += "&&".join(self.common_config.cuts)
            if self.common_config.blind and self.process_config[file_handle.process].type == "Data":
                if len(selection_cuts) != 0:
                    selection_cuts += " && "
                selection_cuts += " !({:s})".format(" && ".join(self.common_config.blind))
            file_handle.fetch_and_link_hist_to_tree(self.tree_name, hist, plot_config.dist, selection_cuts,
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
        if self.statistical_uncertainty_hist:
            plot_config_stat_unc_ratio = copy.copy(plot_config)
            plot_config_stat_unc_ratio.color = ROOT.kYellow
            plot_config_stat_unc_ratio.style = 1001
            plot_config_stat_unc_ratio.draw = "E2"
            plot_config_stat_unc_ratio.logy = False
            statistical_uncertainty_ratio = self.statistical_uncertainty_hist.Clone("stat.unc.ratio")
            statistical_uncertainty_ratio.Divide(mc)
            canvas = PT.plot_hist(statistical_uncertainty_ratio, plot_config_stat_unc_ratio)
            PT.add_histogram_to_canvas(canvas, ratio, plot_config)
        else:
            canvas = PT.plot_hist(ratio, plot_config)
        return canvas

    def fetch_histograms(self, file_handle, plot_config):
        if "data" in file_handle.process.lower() and (plot_config.no_data or self.common_config.no_data):
            return
        return file_handle.process, self.retrieve_histogram(file_handle, plot_config)

    def read_histograms(self, plot_config):
        histograms = mp.ThreadPool(min(self.nfile_handles,
                                       len(self.file_handles))).map(partial(self.fetch_histograms,
                                                                            plot_config=plot_config), self.file_handles)
        return plot_config, histograms

    def categorise_histograms(self, plot_config, histograms):
        for process, hist in histograms:
            try:
                if process not in self.histograms[plot_config].keys():
                    self.histograms[plot_config][process] = hist
                else:
                    self.histograms[plot_config][process].Add(hist)
            except KeyError:
                self.histograms[plot_config] = {process: hist}

    def make_multidimensional_plot(self, plot_config, data):
        for process, histogram in data.iteritems():
            canvas = PT.plot_hist(histogram, plot_config)
            canvas.SetName("{:s}_{:s}".format(canvas.GetName(), process))
            canvas.SetRightMargin(0.15)
            FM.decorate_canvas(canvas, self.common_config, plot_config)
            self.output_handle.register_object(canvas)

    def make_plots(self):
        fetched_histograms = mp.ThreadPool(min(self.ncpu, len(self.plot_configs))).map(self.read_histograms,
                                                                                       self.plot_configs)
        for plot_config, histograms in fetched_histograms:
            histograms = filter(lambda hist: hist is not None, histograms)
            self.categorise_histograms(plot_config, histograms)
        if self.common_config.merge:
            self.merge_histograms()
        for plot_config, data in self.histograms.iteritems():
            if self.common_config.normalise or plot_config.normalise:
                HT.normalise(data)
            if self.common_config.outline == "hist" and not plot_config.is_multidimensional:
                canvas = PT.plot_histograms(data, plot_config, self.common_config, self.process_config)
            elif self.common_config.outline == "stack" and not plot_config.is_multidimensional:
                canvas = PT.plot_stack(data, plot_config, self.common_config, self.process_config)
                stack = get_objects_from_canvas_by_type(canvas, "THStack")[0]
                self.statistical_uncertainty_hist = ST.get_statistical_uncertainty_from_stack(stack)
                #todo: temporary fix
                self.statistical_uncertainty_hist.SetMarkerStyle(1)
                plot_config_stat_unc = PlotConfig(name="stat.unc", dist=None, label="stat unc", draw="E2", style=3244,
                                                  color=ROOT.kBlack)
                PT.add_histogram_to_canvas(canvas, self.statistical_uncertainty_hist, plot_config_stat_unc)
                self.process_config[plot_config_stat_unc.name] = plot_config_stat_unc
            elif plot_config.is_multidimensional:
                self.make_multidimensional_plot(plot_config, data)
                continue
            else:
                _logger.error("Unsupported outline option %s" % self.common_config.outline)
                raise InvalidInputError("Unsupported outline option")
            FM.decorate_canvas(canvas, self.common_config, plot_config)
            FM.add_legend_to_canvas(canvas, process_configs=self.process_config)
            if hasattr(plot_config, "calcsig"):
                #todo: "Background" should be an actual type
                signal_hist = merge_objects_by_process_type(canvas, self.process_config, "Signal")
                background_hist = merge_objects_by_process_type(canvas, self.process_config, "Background")
                significance_hist = ST.get_significance(signal_hist, background_hist)
                canvas_significance_ratio = PT.add_ratio_to_canvas(canvas, significance_hist)
            self.output_handle.register_object(canvas)
            if hasattr(plot_config, "ratio") or hasattr(self.common_config, "ratio"):
                if plot_config.no_data or self.common_config.no_data or plot_config.is_multidimensional:
                    continue
                if self.common_config.ratio:
                    canvas_ratio = self.calculate_ratios(data, plot_config)
                    canvas_combined = PT.add_ratio_to_canvas(canvas, canvas_ratio)
                    self.output_handle.register_object(canvas_combined)
