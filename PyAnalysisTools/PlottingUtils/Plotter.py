import re

import ROOT
import copy
from PyAnalysisTools.base import _logger, InvalidInputError
from PyAnalysisTools.PlottingUtils.PlotConfig import find_process_config, find_process_config_new, ProcessConfig, expand_process_configs_new
from PyAnalysisTools.PlottingUtils.BasePlotter import BasePlotter
from PyAnalysisTools.PlottingUtils import Formatting as FM
from PyAnalysisTools.PlottingUtils import HistTools as HT
from PyAnalysisTools.PlottingUtils import PlottingTools as pt
from PyAnalysisTools.PlottingUtils import RatioPlotter as RP
from PyAnalysisTools.PlottingUtils.PlotConfig import PlotConfig
from PyAnalysisTools.AnalysisTools.XSHandle import XSHandle
from PyAnalysisTools.ROOTUtils.ObjectHandle import merge_objects_by_process_type
from PyAnalysisTools.AnalysisTools.SystematicsAnalyser import SystematicsAnalyser
from PyAnalysisTools.AnalysisTools import StatisticsTools as ST
from PyAnalysisTools.base.OutputHandle import OutputFileHandle
from PyAnalysisTools.ROOTUtils.ObjectHandle import get_objects_from_canvas_by_type
from PyAnalysisTools.AnalysisTools.RegionBuilder import RegionBuilder
from PyAnalysisTools.AnalysisTools.FakeEstimator import MuonFakeEstimator
from PyAnalysisTools.base.Modules import load_modules
from collections import OrderedDict


class Plotter(BasePlotter):
    def __init__(self, **kwargs):
        if "input_files" not in kwargs:
            _logger.error("No input files provided")
            raise InvalidInputError("No input files")
        if "plot_config_files" not in kwargs:
            _logger.error("No plot config files provided. Nothing to parse")
            raise InvalidInputError("No plot config file")
        if "process_config_files" not in kwargs:
            _logger.warning("No process config file provided. Unable to read process specific options.")
        if "xs_config_file" not in kwargs:
            _logger.error("No cross section file provided. No scaling will be applied.")
            kwargs.setdefault("xs_config_file", None)
        kwargs.setdefault("systematics", "Nominal")
        kwargs.setdefault("process_config_files", None)
        kwargs.setdefault("process_config_file", None)
        kwargs.setdefault("xs_config_file", None)
        kwargs.setdefault("ncpu", 1)
        kwargs.setdefault("nfile_handles", 1)
        kwargs.setdefault("output_file_name", "plots.root")
        kwargs.setdefault("enable_systematics", False)
        kwargs.setdefault("module_config_file", None)
        kwargs.setdefault("read_hist", False)
        kwargs.setdefault('file_extension', ['.pdf'])

        super(Plotter, self).__init__(**kwargs)
        for k, v in kwargs.iteritems():
            setattr(self, k, v)
        self.xs_handle = XSHandle(kwargs["xs_config_file"])
        self.stat_unc_hist = None
        self.histograms = {}
        self.output_handle = OutputFileHandle(make_plotbook=self.plot_configs[0].make_plot_book, extension=kwargs['file_extension'], **kwargs)
        self.syst_analyser = None
        if kwargs["enable_systematics"]:
            self.syst_analyser = SystematicsAnalyser(**self.__dict__)

        self.file_handles = filter(lambda fh: fh.process is not None, self.file_handles)
        self.process_configs = expand_process_configs_new(map(lambda fh: fh.process, self.file_handles),
                                                          self.process_configs,
                                                          any(map(lambda pc: pc.merge_mc_campaigns, self.plot_configs)))

        self.file_handles = self.filter_process_configs(self.file_handles, self.process_configs)
        if not self.read_hist:
            self.filter_empty_trees()
        self.modules = load_modules(kwargs["module_config_file"], self)
        self.modules_pc_modifiers = [m for m in self.modules if m.type == "PCModifier"]
        self.modules_data_providers = [m for m in self.modules if m.type == "DataProvider"]
        self.modules_hist_fetching = [m for m in self.modules if m.type == "HistFetching"]
        #self.fake_estimator = MuonFakeEstimator(self, file_handles=self.file_handles)

    def add_mc_campaigns(self):
        if self.process_configs is None:
            return
        for process_config_name in self.process_configs.keys():
            process_config = self.process_configs[process_config_name]
            if process_config.is_data:
                continue
            for i, campaign in enumerate(["mc16a", "mc16c", "mc16d", 'mc16e']):
                new_config = copy.copy(process_config)
                new_config.name += campaign
                new_config.label += " {:s}".format(campaign)
                #TODO fix proper calculation
                if not '+' in new_config.color and not '-' in new_config.color:
                    new_config.color = new_config.color + " + {:d}".format(3*pow(-1, i))
                if hasattr(process_config, "subprocesses"):
                    new_config.subprocesses = ["{:s}.{:s}".format(sb, campaign) for sb in process_config.subprocesses]
                self.process_configs["{:s}.{:s}".format(process_config_name, campaign)] = new_config
            self.process_configs.pop(process_config_name)

    @staticmethod
    def filter_process_configs(file_handles, process_configs=None, split_mc_campaigns=False):
        if process_configs is None:
            return file_handles
        unavailable_process = map(lambda fh: fh.process,
                                  filter(lambda fh: find_process_config_new(fh.process, process_configs) is None,
                                         file_handles))
        for process in unavailable_process:
            _logger.error("Unable to find merge process config for {:s}".format(str(process)))
        return filter(lambda fh: find_process_config_new(fh.process, process_configs) is not None,
                      file_handles)

    def initialise(self):
        self.ncpu = min(self.ncpu, len(self.plot_configs))

    def filter_empty_trees(self):
        def is_empty(file_handle, tree_name):
            return file_handle.get_object_by_name(tree_name, "Nominal").GetEntries() > 0
        self.file_handles = filter(lambda fh: is_empty(fh, self.tree_name), self.file_handles)

    #todo: why is RatioPlotter not called?
    def calculate_ratios(self, hists, plot_config):
        if not "Data" in hists:
            _logger.error("Ratio requested but no data found")
            raise InvalidInputError("Missing data")
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
        if hasattr(plot_config, "ratio_config"):
            plot_config = plot_config.ratio_config
        else:
            plot_config.name = "ratio_" + plot_config.name
            plot_config.ytitle = "data/MC"
        if "unit" in plot_config.__dict__.keys():
            plot_config.__dict__.pop("unit")
            plot_config.draw = "Marker"
        plot_config.logy = False

        if self.stat_unc_hist:
            plot_config_stat_unc_ratio = copy.copy(plot_config)
            plot_config_stat_unc_ratio.color = ROOT.kYellow
            plot_config_stat_unc_ratio.style = 1001
            plot_config_stat_unc_ratio.draw = "E2"
            plot_config_stat_unc_ratio.logy = False
            statistical_uncertainty_ratio = ST.get_statistical_uncertainty_ratio(self.stat_unc_hist)
            canvas = pt.plot_hist(statistical_uncertainty_ratio, plot_config_stat_unc_ratio)
            pt.add_histogram_to_canvas(canvas, ratio, plot_config)
        else:
            canvas = pt.plot_hist(ratio, plot_config)
        return canvas

    def apply_lumi_weights(self, histograms):
        for hist_set in histograms.values():
            for process, hist in hist_set.iteritems():
                if hist is None:
                    _logger.error("Histogram for process {:s} is None".format(process))
                    continue
                if "data" in process.lower():
                    continue
                cross_section_weight = self.xs_handle.get_lumi_scale_factor(process.split(".")[0], self.lumi,
                                                                            self.event_yields[process])
                HT.scale(hist, cross_section_weight)

    def apply_lumi_weights_new(self, histograms):
        provided_wrong_info = False
        for plot_config, hist_set in histograms.iteritems():
            for process, hist in hist_set.iteritems():
                if hist is None:
                    _logger.error("Histogram for process {:s} is None".format(process))
                    continue
                if "data" in process.lower():
                    continue
                lumi = self.lumi
                if isinstance(self.lumi, OrderedDict):
                    if re.search('mc16[acde]$', process) is None:
                        if provided_wrong_info is False:
                            _logger.error('Could not find MC campaign informaiton, but lumi was provided per MC '
                                          'campaing. Not clear what to do. It will be assumed that you meant to scale '
                                          'to total lumi. Please update and acknowledge once.')
                            raw_input('Hit enter to continue or Ctrl+c to quit...')
                            provided_wrong_info = True
                            plot_config.used_mc_campaigns = self.lumi.keys()
                        lumi = sum(self.lumi.values())
                    else:
                        lumi = self.lumi[process.split('.')[-1]]
                        plot_config.used_mc_campaigns.append(process.split('.')[-1])
                cross_section_weight = self.xs_handle.get_lumi_scale_factor(process.split(".")[0], lumi,
                                                                            self.event_yields[process])
                HT.scale(hist, cross_section_weight)

    def make_multidimensional_plot(self, plot_config, data):
        for process, histogram in data.iteritems():
            canvas = pt.plot_obj(histogram, plot_config)
            canvas.SetName("{:s}_{:s}".format(canvas.GetName(), process))
            canvas.SetRightMargin(0.2)
            FM.decorate_canvas(canvas, plot_config)
            self.output_handle.register_object(canvas)

    def cut_based_normalise(self, cut):
        event_yields = {}
        for file_handle in self.file_handles:
            cutflow = file_handle.get_object_by_name("Nominal/cutflow_BaseSelection")
            process_config = find_process_config_new(file_handle.process, self.process_configs)
            if process_config is None:
                continue
            if process_config.is_data:
                process = "data"
                cross_section_weight = 1.
            elif process_config.is_mc and not hasattr(process_config, "is_signal"):
                process = "mc"
                cross_section_weight = self.xs_handle.get_lumi_scale_factor(file_handle.process, self.lumi,
                                                                            self.event_yields[file_handle.process])
            else:
                continue
            i_bin = HT.read_bin_from_label(cutflow, cut)
            if process in event_yields:
                event_yields[process] += cutflow.GetBinContent(i_bin) * cross_section_weight
            else:
                event_yields[process] = cutflow.GetBinContent(i_bin) * cross_section_weight
        if event_yields["mc"] > 0.:
            scale_factor = event_yields["data"] / event_yields["mc"]
        else:
            scale_factor = 0.
        _logger.debug("Calculated scale factor {:.2f} after cut {:s}".format(scale_factor, cut))
        for hist_set in self.histograms.values():
            for process_name, hist in hist_set.iteritems():
                process_config = find_process_config_new(process_name, self.process_configs)
                if process_config is None:
                    _logger.error("Could not find process config for {:s}".format(process_name))
                    continue
                if not process_config.is_mc or hasattr(process_config, "is_signal"):
                    continue
                HT.scale(hist, scale_factor)

    def get_signal_hists(self, data):
        signals = {}
        if self.process_configs is None:
            return signals
        signal_process_configs = filter(lambda pc: isinstance(pc[1], ProcessConfig) and
                                                   pc[1].type.lower() == "signal", self.process_configs.iteritems())
        print signal_process_configs
        print data.keys()
        if len(signal_process_configs) == 0:
            return signals
        for process in signal_process_configs:
            print 'check prcoess ', process[0], ' is in data?: ', process[0] in data
            try:
                signals[process[0]] = data.pop(process[0])
            except KeyError:
                continue
        return signals

    def scale_signals(self, signals, plot_config):
        for process, signal_hist in signals.iteritems():
            #label_postfix = "(x {:.0f})".format(plot_config.signal_scale)
            # if label_postfix not in self.process_configs[process].label:
            #     self.process_configs[process].label += label_postfix
            #HT.scale(signal_hist, plot_config.signal_scale)
            if hasattr(self.process_configs[process], 'signal_scale'):
                HT.scale(signal_hist, self.process_configs[process].signal_scale)

    def make_plot(self, plot_config, data):
        for mod in self.modules_data_providers:
            data.update([mod.execute(plot_config)])
        data = {k: v for k, v in data.iteritems() if v}
        if plot_config.normalise:
            HT.normalise(data, integration_range=[0, -1])
        HT.merge_overflow_bins(data)
        HT.merge_underflow_bins(data)
        if plot_config.signal_extraction:
            signals = self.get_signal_hists(data)
        # todo: need proper fix for this
        if plot_config.signal_scale is not None:
            self.scale_signals(signals, plot_config)
        signal_only = False
        # print 'SIGNALS: ', signals
        # if len(signals) > 0 and len(data) == 0:
        #     signal_only = True
        if plot_config.outline == "stack" and not plot_config.is_multidimensional and not signal_only:
            canvas = pt.plot_stack(data, plot_config=plot_config,
                                   process_configs=self.process_configs)
            stack = get_objects_from_canvas_by_type(canvas, "THStack")[0]
            canvas.Update()

            self.stat_unc_hist = ST.get_statistical_uncertainty_from_stack(stack)
            # todo: temporary fix
            self.stat_unc_hist.SetMarkerStyle(1)
            plot_config_stat_unc = PlotConfig(name="stat.unc", dist=None, label="stat unc", draw="E2", style=3244,
                                              color=ROOT.kBlack)
            pt.add_object_to_canvas(canvas, self.stat_unc_hist, plot_config_stat_unc)
            if plot_config.signal_extraction:
                for signal in signals.iteritems():
                    pt.add_signal_to_canvas(signal, canvas, plot_config, self.process_configs)
            self.process_configs[plot_config_stat_unc.name] = plot_config_stat_unc
        elif plot_config.is_multidimensional:
            self.make_multidimensional_plot(plot_config, data)
            return
        elif signal_only:
            canvas = pt.plot_objects(signals, plot_config, process_configs=self.process_configs)
        else:
            print data.values()[0].GetEntries()
            c = ROOT.TCanvas("c","", 800, 600)
            c.cd()
            data.values()[0].Draw()
            c.SaveAs("test.pdf")
            canvas = pt.plot_objects(data, plot_config, process_configs=self.process_configs)
            if plot_config.signal_extraction:
                for signal in signals.iteritems():
                    pt.add_signal_to_canvas(signal, canvas, plot_config, self.process_configs)
                    
        FM.decorate_canvas(canvas, plot_config)
        if plot_config.enable_legend:
            if plot_config.legend_options is not None:
                FM.add_legend_to_canvas(canvas, ratio=plot_config.ratio, process_configs=self.process_configs,
                                        **plot_config.legend_options)
            else:
                FM.add_legend_to_canvas(canvas, ratio=plot_config.ratio, process_configs=self.process_configs)
        if hasattr(plot_config, "calcsig"):
            # todo: "Background" should be an actual type
            merged_process_configs = dict(filter(lambda pc: hasattr(pc[1], "type"),
                                                 self.process_configs.iteritems()))
            #signal_hist = merge_objects_by_process_type(canvas, merged_process_configs, "Signal")
            signal_hist = signals.values()[0]
            background_hist = merge_objects_by_process_type(canvas, merged_process_configs, "Background")
            print background_hist.GetEntries()
            print signal_hist.GetEntries()
            if hasattr(plot_config, "significance_config"):
                sig_plot_config = plot_config.significance_config
            else:
                sig_plot_config = copy.copy(plot_config)
                sig_plot_config.name = "sig_" + plot_config.name
                sig_plot_config.ytitle = "S/#sqrt{S + B}"
                sig_plot_config.normalise = False
            significance_canvas = None
            for process, signal_hist in signals.iteritems():
                sig_plot_config.color = self.process_configs[process].color
                sig_plot_config.name = "significance_{:s}".format(process)
                sig_plot_config.ymin = 0.00001
                sig_plot_config.ymax = 10.
                sig_plot_config.ytitle = "S/#sqrt{B}"

                significance_canvas = ST.get_significance(signal_hist, background_hist, sig_plot_config,
                                                          significance_canvas)
                canvas_significance_ratio = pt.add_ratio_to_canvas(canvas, significance_canvas,
                                                                   name=canvas.GetName() + "_significance")
            if significance_canvas is not None:
                self.output_handle.register_object(canvas_significance_ratio)
                
        self.output_handle.register_object(canvas)
        if plot_config.ratio:
            if plot_config.no_data or plot_config.is_multidimensional:
                return
            if "Data" not in data:
                _logger.error("Requested ratio, but no data provided. Cannot build ratio.")
                return
            mc_total = None
            for key, hist in data.iteritems():
                if key == "Data":
                    continue
                if mc_total is None:
                    mc_total = hist.Clone("mc_total_%s" % plot_config.name)
                    continue
                mc_total.Add(hist)
            if hasattr(plot_config, "ratio_config"):
                ratio_plot_config = plot_config.ratio_config
                if plot_config.logx:
                    ratio_plot_config.logx = True
            else:
                ratio_plot_config = copy.copy(plot_config)
                ratio_plot_config.name = "ratio_" + plot_config.name
                ratio_plot_config.ytitle = "ratio"
            ratio_plot_config.name = "ratio_" + plot_config.name
            ratio_plotter = RP.RatioPlotter(reference=mc_total, compare=data["Data"],
                                            plot_config=ratio_plot_config)
            canvas_ratio = ratio_plotter.make_ratio_plot()
            if self.stat_unc_hist:
                plot_config_stat_unc_ratio = copy.copy(ratio_plot_config)
                plot_config_stat_unc_ratio.name = ratio_plot_config.name.replace("ratio", "stat_unc")
                plot_config_stat_unc_ratio.color = ROOT.kBlack
                plot_config_stat_unc_ratio.style = 3244
                plot_config_stat_unc_ratio.draw = "E2"
                plot_config_stat_unc_ratio.logy = False
                stat_unc_ratio = ST.get_statistical_uncertainty_ratio(self.stat_unc_hist)
                stat_unc_ratio.SetMarkerColor(ROOT.kYellow)
                stat_unc_ratio.SetMarkerStyle(1)

                if self.syst_analyser is None:
                    canvas_ratio = ratio_plotter.add_uncertainty_to_canvas(canvas_ratio, stat_unc_ratio,
                                                                           plot_config_stat_unc_ratio)
            if self.syst_analyser is not None:
                syst_color = ROOT.kRed
                plot_config_syst_unc_ratio = copy.copy(ratio_plot_config)
                plot_config_syst_unc_ratio.name = ratio_plot_config.name.replace("ratio", "syst_unc")
                plot_config_syst_unc_ratio.color = syst_color
                plot_config_syst_unc_ratio.style = 1001
                plot_config_syst_unc_ratio.draw = "E2"
                plot_config_syst_unc_ratio.logy = False
                syst_sm_total_up_categotised, syst_sm_total_down_categotised, colors = self.syst_analyser.get_relative_unc_on_SM_total(
                    plot_config, data)
                ratio_syst_up = ST.get_relative_systematics_ratio(mc_total, stat_unc_ratio, syst_sm_total_up_categotised)
                ratio_syst_down = ST.get_relative_systematics_ratio(mc_total, stat_unc_ratio, syst_sm_total_down_categotised)
                plot_config_syst_unc_ratio.color = colors
                canvas_ratio = ratio_plotter.add_uncertainty_to_canvas(canvas_ratio,
                                                                       ratio_syst_up + ratio_syst_down + [stat_unc_ratio],
                                                                       [plot_config_syst_unc_ratio,
                                                                        plot_config_syst_unc_ratio,
                                                                        plot_config_stat_unc_ratio],
                                                                       n_systematics=len(ratio_syst_up))


            ratio_plotter.decorate_ratio_canvas(canvas_ratio)
            #canvas_ratio = ratio_plotter.overlay_out_of_range_arrow(canvas_ratio)
            #canvas_ratio.SaveAs("foo_"+canvas_ratio.GetName()+".pdf")
            canvas_combined = pt.add_ratio_to_canvas(canvas, canvas_ratio)
            self.output_handle.register_object(canvas_combined)

    def make_plots(self):
        # self.read_cutflows()
        for mod in self.modules_pc_modifiers:
            self.plot_configs = mod.execute(self.plot_configs)
            if self.syst_analyser is not None:
                self.syst_analyser.plot_configs = self.plot_configs
        if not self.read_hist:
            if len(self.modules_hist_fetching) == 0:
                fetched_histograms = self.read_histograms(file_handles=self.file_handles, plot_configs=self.plot_configs)
            else:
                fetched_histograms = self.modules_hist_fetching[0].fetch()
        else:
            fetched_histograms = self.read_histograms_plain(file_handle=self.file_handles,
                                                            plot_configs=self.plot_configs)
        fetched_histograms = filter(lambda hist_set: all(hist_set), fetched_histograms)
        self.categorise_histograms(fetched_histograms)
        if not self.lumi < 0:
            self.apply_lumi_weights_new(self.histograms)

        if hasattr(self.plot_configs, "normalise_after_cut"):
            self.cut_based_normalise(self.plot_configs.normalise_after_cut)
        #workaround due to missing worker node communication of regex process parsing
        if self.process_configs is not None:
            self.merge_histograms()
        if self.syst_analyser is not None:
            self.syst_analyser.retrieve_sys_hists(self.file_handles)
            self.syst_analyser.calculate_variations(self.histograms)
            self.syst_analyser.calculate_total_systematics()
        for plot_config, data in self.histograms.iteritems():
            self.make_plot(plot_config, data)
        self.output_handle.write_and_close()

