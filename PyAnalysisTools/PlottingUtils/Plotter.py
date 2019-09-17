from __future__ import division
from builtins import map
from builtins import str
from past.utils import old_div
from builtins import object
import glob
import re
from itertools import product

import ROOT
import copy
#required to get deepcopy working for compiled regex. Should be fixed in python 3.7
copy._deepcopy_dispatch[type(re.compile(''))] = lambda r, _: r

import os
from PyAnalysisTools.ROOTUtils.FileHandle import FileHandle
from PyAnalysisTools.base import _logger, InvalidInputError
from PyAnalysisTools.PlottingUtils.PlotConfig import find_process_config
from PyAnalysisTools.base.ProcessConfig import ProcessConfig
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
from PyAnalysisTools.AnalysisTools.FakeEstimator import ElectronFakeEstimator
from PyAnalysisTools.base.Modules import load_modules
from collections import OrderedDict


class PlotArgs(object):
    def __init__(self, **kwargs):
        for attr, val in list(kwargs.items()):
            setattr(self, attr, val)

    def __str__(self):
        """
        Overloaded str operator. Get's called if object is printed
        :return: formatted string with name and attributes
        :rtype: str
        """
        obj_str = "Plot arguments \n"
        for attribute, value in list(self.__dict__.items()):
            obj_str += '{}={} \n'.format(attribute, value)
        return obj_str

    def __repr__(self):
        """
        Overloads representation operator. Get's called e.g. if list of objects are printed
        :return: formatted string with name and attributes
        :rtype: str
        """
        return self.__str__() + '\n'


class Plotter(BasePlotter):
    def __init__(self, **kwargs):
        kwargs.setdefault('cluster_config', None)
        kwargs.setdefault('ncpu', 1)
        kwargs.setdefault('cluster_mode', False)
        kwargs.setdefault('redraw_hists', None)

        _logger.debug("Initialise Plotter")
        if kwargs['cluster_config'] is not None:
            self.cluster_init(kwargs['cluster_config'])
            self.cluster_mode = True
            self.ncpu = kwargs['ncpu']
            return
        self.cluster_mode = kwargs['cluster_mode']
        if kwargs['redraw_hists'] is not None:
            self.redraw_init(**kwargs)
            return

        if "input_files" not in kwargs and not 'input_file_list' in kwargs:
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
        kwargs.setdefault("nfile_handles", 1)
        kwargs.setdefault("output_file_name", "plots.root")
        kwargs.setdefault("enable_systematics", False)
        kwargs.setdefault("module_config_files", [])
        kwargs.setdefault("read_hist", False)
        kwargs.setdefault('file_extension', ['.pdf'])

        super(Plotter, self).__init__(**kwargs)
        for k, v in list(kwargs.items()):
            if hasattr(self, k) and v is None:
                continue
            setattr(self, k, v)

        self.xs_handle = XSHandle(kwargs["xs_config_file"])
        self.stat_unc_hist = None
        self.histograms = {}
        self.output_handle = OutputFileHandle(make_plotbook=self.plot_configs[0].make_plot_book,
                                              extension=kwargs['file_extension'], **kwargs)
        self.syst_analyser = None
        if kwargs["enable_systematics"]:
            self.syst_analyser = SystematicsAnalyser(**self.__dict__)

        self.file_handles = [fh for fh in self.file_handles if fh.process is not None]
        self.file_handles = self.filter_unavailable_processes(self.file_handles, self.process_configs)
        if not self.read_hist:
            self.filter_empty_trees()
        self.modules = load_modules(kwargs['module_config_files'], self)
        self.fake_estimator = ElectronFakeEstimator(self, file_handles=self.file_handles)
        #self.modules.append(self.fake_estimator)
        self.init_modules()
        self.expand_plot_configs()
        if kwargs["enable_systematics"]:
            self.syst_analyser = SystematicsAnalyser(**self.__dict__)

    def expand_plot_configs(self):
        """
        Expand raw plot configs for each specified region provided via module config files
        :return:
        :rtype:
        """
        if len(self.modules_pc_modifiers) == 0:
            return
        original_plot_configs = copy.deepcopy(self.plot_configs)
        self.plot_configs = []
        for mod in self.modules_pc_modifiers:
            self.plot_configs += mod.execute(original_plot_configs)

    def init_modules(self):
        self.modules_pc_modifiers = [m for m in self.modules if m.type == "PCModifier"]
        self.modules_data_providers = [m for m in self.modules if m.type == "DataProvider"]
        self.modules_data_modifiers = [m for m in self.modules if m.type == 'DataModifier']
        self.modules_hist_fetching = [m for m in self.modules if m.type == "HistFetching"]

    def redraw_init(self, **kwargs):
        super(Plotter, self).__init__(cluster_mode=False, redraw=True)
        config = kwargs['config']
        self.histograms = {}
        self.modules = load_modules(kwargs['module_config_files'], self)
        self.init_modules()
        self.read_hist = True
        self.xs_handle = XSHandle(config.extra_args["xs_config_file"])
        self.ncpu = 1
        self.nfile_handles = 1
        self.plot_config_files = kwargs['plot_config_files']
        self.parse_plot_config()
        self.expand_plot_configs()
        self.xs_config_file = config.extra_args['xs_config_file']
        if len(kwargs['process_config_files']) > 0:
            self.process_config_files = kwargs['process_config_files']
            self.process_config_file = None
            self.process_configs = self.parse_process_config()
        else:
            self.process_configs = config.extra_args['process_configs']
        self.syst_analyser = config.extra_args['syst_analyser']
        if self.syst_analyser is not None:
            self.syst_analyser.file_handles = self.file_handles
            self.syst_analyser.event_yields = self.event_yields
            self.syst_analyser.dump_hists = False
            self.syst_analyser.plot_configs = self.plot_configs
        #config.output_dir = "/Users/morgens/tmp/test"
        self.output_handle = OutputFileHandle(make_plotbook=self.plot_configs[0].make_plot_book,
                                              extension=['.pdf'], output_dir=kwargs['output_dir'])
        self.output_handle.reinitialise_output_dir()

    def cluster_init(self, config):
        super(Plotter, self).__init__(cluster_config=config)
        for attr, val in list(config.extra_args.items()):
            setattr(self, attr, val)
        self.histograms = {}
        self.plot_configs = config.plot_config
        self.modules_pc_modifiers = []
        self.modules_data_providers = []
        self.modules_data_modifiers = []
        self.modules_hist_fetching = []
        self.read_hist = False
        self.ncpu = 1
        self.nfile_handles = 1
        if self.syst_analyser is not None:
            self.syst_analyser.file_handles = self.file_handles
            self.syst_analyser.event_yields = self.event_yields
            self.syst_analyser.dump_hists = True
            self.syst_analyser.plot_configs = self.plot_configs
        self.output_handle = OutputFileHandle(make_plotbook=self.plot_configs[0].make_plot_book,
                                              extension=None, output_dir=config.output_dir, sub_dir_name='hists',
                                              output_file='hist-{:s}'.format(config.file_name.split('-')[1]))

    def add_mc_campaigns(self):
        if self.process_configs is None:
            return
        for process_config_name in list(self.process_configs.keys()):
            process_config = self.process_configs[process_config_name]
            if process_config.is_data:
                continue
            for i, campaign in enumerate(['mc16a', 'mc16c', 'mc16d', 'mc16e']):
                new_config = copy.copy(process_config)
                new_config.name += campaign
                new_config.label += ' {:s}'.format(campaign)
                #TODO fix proper calculation
                if not '+' in new_config.color and not '-' in new_config.color:
                    new_config.color = new_config.color + ' + {:d}'.format(3*pow(-1, i))
                if hasattr(process_config, 'subprocesses'):
                    new_config.subprocesses = ['{:s}.{:s}'.format(sb, campaign) for sb in process_config.subprocesses]
                self.process_configs['{:s}.{:s}'.format(process_config_name, campaign)] = new_config
            self.process_configs.pop(process_config_name)

    @staticmethod
    def filter_unavailable_processes(file_handles, process_configs):
        if process_configs is None:
            return file_handles
        unavailable_process = [fh.process for fh in [fh for fh in file_handles if find_process_config(fh.process, process_configs) is None]]
        for process in unavailable_process:
            _logger.debug("Unable to find merge process config for {:s}".format(str(process)))
        failed_file_handles = [fh for fh in file_handles if find_process_config(fh.process, process_configs) is None]
        if len(failed_file_handles) > 0:
            _logger.debug("failed file handles {:s}.".format(', '.join([fh.file_name for fh in failed_file_handles])))
        list([fh.close() for fh in failed_file_handles])
        return [fh for fh in file_handles if find_process_config(fh.process, process_configs) is not None]

    @staticmethod
    def filter_processes_new(file_handles, process_configs):
        _logger.error("This function is deprecated. Switch to filter_unavailable_processes.")
        return Plotter.filter_unavailable_processes(file_handles, process_configs)

    def initialise(self):
        self.ncpu = min(self.ncpu, len(self.plot_configs))

    def filter_empty_trees(self):
        def is_empty(file_handle, tree_name, syst_tree_name):
            tn = tree_name
            if syst_tree_name is not None and file_handle.is_mc:
                tn = syst_tree_name
            return file_handle.get_object_by_name(tn, "Nominal").GetEntries() > 0

        empty_files = [fh for fh in self.file_handles if not is_empty(fh, self.tree_name, self.syst_tree_name)]
        self.file_handles = [fh for fh in self.file_handles if is_empty(fh, self.tree_name, self.syst_tree_name)]
        list([fh.close() for fh in empty_files])

    #todo: why is RatioPlotter not called?
    def calculate_ratios(self, hists, plot_config):
        if not "Data" in hists:
            _logger.error("Ratio requested but no data found")
            raise InvalidInputError("Missing data")
        ratio = hists["Data"].Clone("ratio_%s" % plot_config.dist)
        mc = None
        for key, hist in list(hists.items()):
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
        if "unit" in list(plot_config.__dict__.keys()):
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

    def make_multidimensional_plot(self, plot_config, data):
        for process, histogram in list(data.items()):
            canvas = pt.plot_obj(histogram, plot_config)
            canvas.SetName("{:s}_{:s}".format(canvas.GetName(), process))
            canvas.SetRightMargin(0.2)
            FM.decorate_canvas(canvas, plot_config)
            self.output_handle.register_object(canvas)

    def cut_based_normalise(self, cut):
        event_yields = {}
        for file_handle in self.file_handles:
            cutflow = file_handle.get_object_by_name("Nominal/cutflow_BaseSelection")
            process_config = find_process_config(file_handle.process, self.process_configs)
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
            scale_factor = old_div(event_yields["data"], event_yields["mc"])
        else:
            scale_factor = 0.
        _logger.debug("Calculated scale factor {:.2f} after cut {:s}".format(scale_factor, cut))
        for hist_set in list(self.histograms.values()):
            for process_name, hist in list(hist_set.items()):
                process_config = find_process_config(process_name, self.process_configs)
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
        signal_process_configs = [pc for pc in iter(list(self.process_configs.items())) if isinstance(pc[1], ProcessConfig) and
                                                   pc[1].type.lower() == "signal"]
        if len(signal_process_configs) == 0:
            return signals
        for process in signal_process_configs:
            try:
                signals[process[0]] = data.pop(process[0])
            except KeyError:
                continue
        return signals

    def scale_signals(self, signals, plot_config):
        for process, signal_hist in list(signals.items()):
            #label_postfix = "(x {:.0f})".format(plot_config.signal_scale)
            # if label_postfix not in self.process_configs[process].label:
            #     self.process_configs[process].label += label_postfix
            #HT.scale(signal_hist, plot_config.signal_scale)
            if hasattr(self.process_configs[process], 'signal_scale'):
                HT.scale(signal_hist, self.process_configs[process].signal_scale)

    def scale_process(self, data):
        for process, hist in list(data.items()):
            if self.process_configs[process].scale_factor is None:
                continue
            HT.scale(hist, self.process_configs[process].scale_factor)

    def make_plot(self, plot_config, data):
        _logger.debug("Making single plot")
        for mod in self.modules_data_providers:
            data.update([mod.execute(plot_config)])
        for mod in self.modules_data_modifiers:
            try:
                mod.execute(data, self.output_handle, plot_config, self.syst_analyser)
            except TypeError:
                mod.execute(data)
        data = {k: v for k, v in list(data.items()) if v}
        if plot_config.normalise:
            HT.normalise(data, integration_range=[0, -1], norm_scale=plot_config.norm_scale)
        HT.merge_overflow_bins(data)
        HT.merge_underflow_bins(data)
        signals = None
        if plot_config.signal_extraction:
            signals = self.get_signal_hists(data)
        #todo: need proper fix for this
        if plot_config.signal_scale is not None and signals is not None:
            self.scale_signals(signals, plot_config)
        self.scale_process(data)

        signal_only = False
        if signals is not None and len(signals) > 0 and len(data) == 0:
            signal_only = True
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
                for signal in list(signals.items()):
                    pt.add_signal_to_canvas(signal, canvas, plot_config, self.process_configs)
            self.process_configs[plot_config_stat_unc.name] = plot_config_stat_unc
        elif plot_config.is_multidimensional:
            self.make_multidimensional_plot(plot_config, data)
            return
        elif signal_only:
            canvas = pt.plot_objects(signals, plot_config, process_configs=self.process_configs)
        else:
            canvas = pt.plot_objects(data, plot_config, process_configs=self.process_configs)
            if plot_config.signal_extraction:
                for signal in list(signals.items()):
                    pt.add_signal_to_canvas(signal, canvas, plot_config, self.process_configs)
        FM.decorate_canvas(canvas, plot_config)
        if not plot_config.disable_legend or plot_config.enable_legend:
            if plot_config.legend_options is not None:
                FM.add_legend_to_canvas(canvas, ratio=plot_config.ratio, process_configs=self.process_configs,
                                        **plot_config.legend_options)
            else:
                FM.add_legend_to_canvas(canvas, ratio=plot_config.ratio, process_configs=self.process_configs)
        if hasattr(plot_config, "calcsig"):
            # todo: "Background" should be an actual type
            merged_process_configs = dict([pc for pc in iter(list(self.process_configs.items())) if hasattr(pc[1], "type")])
            #signal_hist = merge_objects_by_process_type(canvas, merged_process_configs, "Signal")
            signal_hist = list(signals.values())[0]
            background_hist = merge_objects_by_process_type(canvas, merged_process_configs, "Background")
            if hasattr(plot_config, "significance_config"):
                sig_plot_config = plot_config.significance_config
            else:
                sig_plot_config = copy.copy(plot_config)
                sig_plot_config.name = "sig_" + plot_config.name
                sig_plot_config.ytitle = "S/#sqrt{S + B}"
                sig_plot_config.normalise = False
            significance_canvas = None
            for process, signal_hist in list(signals.items()):
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
            for key, hist in list(data.items()):
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
                    ratio_plot_config.xmin = plot_config.xmin
                    ratio_plot_config.xmax = plot_config.xmax
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
                plot_config_syst_unc_ratio.style = 3244
                plot_config_syst_unc_ratio.draw = "E2"
                plot_config_syst_unc_ratio.logy = False
                syst_sm_total_up_categotised, syst_sm_total_down_categotised, \
                colors = self.syst_analyser.get_relative_unc_on_SM_total(plot_config, data)
                ratio_syst_up = ST.get_relative_systematics_ratio(mc_total, stat_unc_ratio,
                                                                  syst_sm_total_up_categotised)
                ratio_syst_down = ST.get_relative_systematics_ratio(mc_total, stat_unc_ratio,
                                                                    syst_sm_total_down_categotised)
                list([h.SetMarkerStyle(1) for h in ratio_syst_up])
                list([h.SetMarkerStyle(1) for h in ratio_syst_down])
                colors.append(ROOT.kBlack)
                plot_config_syst_unc_ratio.color = colors
                canvas_ratio = ratio_plotter.add_uncertainty_to_canvas(canvas_ratio,
                                                                       ratio_syst_up + ratio_syst_down + [stat_unc_ratio],
                                                                       [plot_config_syst_unc_ratio,
                                                                        plot_config_syst_unc_ratio,
                                                                        plot_config_stat_unc_ratio],
                                                                       n_systematics=len(ratio_syst_up))
                self.syst_analyser.make_overview_plots(plot_config)

            ratio_plotter.decorate_ratio_canvas(canvas_ratio)
            canvas_combined = pt.add_ratio_to_canvas(canvas, canvas_ratio)
            _logger.debug('register canvas {:s}'.format(canvas_combined.GetName()))
            self.output_handle.register_object(canvas_combined)

    def build_fetched_histograms(self):
        """
        Wrapper function to read all histograms from provided file handles and plot configs
        :return: list of fetched histograms objects
        :rtype: list
        """
        hists = []
        for arg in product(self.file_handles, self.plot_configs):
            hists.append(self.get_fetched_hist(arg))
        return hists

    @staticmethod
    def get_fetched_hist(args):
        """
        Read histogram from canvas stored in root file
        :param args: input arguments containing pair of file handle and plot config
        :type args: tuple (size 2)
        :return: plot config, file handle process and histograms from canvas
        :rtype: list
        """

        fh = args[0]
        pc = args[1]
        c = fh.get_object_by_name(pc.name)
        return pc, fh.process, get_objects_from_canvas_by_type(c, 'TH1F')[0]

    def project_hists(self):
        self.read_cutflows() #disabled in susy
        if self.syst_analyser is not None:
            self.syst_analyser.plot_configs = self.plot_configs

        if not self.read_hist:
            if len(self.modules_hist_fetching) == 0:
                fetched_histograms = self.read_histograms(file_handles=self.file_handles,
                                                          plot_configs=self.plot_configs)
            else:
                fetched_histograms = self.modules_hist_fetching[0].fetch()
        else:
            fetched_histograms = self.read_histograms_plain(file_handle=self.file_handles,
                                                            plot_configs=self.plot_configs)

        return [hist_set for hist_set in fetched_histograms if all(hist_set)]

    def read_hists(self, path):
        input_files = glob.glob(os.path.join(path, '*.root'))
        _logger.debug("Reading {:d} files now from path".format(len(input_files)))
        self.file_handles = [FileHandle(file_name=fn, dataset_info=self.xs_config_file,
                                        split_mc=False) for fn in input_files]
        self.file_handles = self.filter_unavailable_processes(self.file_handles, self.process_configs)
        if self.syst_analyser is not None:
            self.syst_analyser.file_handles = self.file_handles
        return self.build_fetched_histograms()

    def consistency_check(self):
        """
        Run consistency checks on plot configs and provided inputs prior to executing plotting
        :return: decision if all checks are passed
        :rtype: bool
        """
        if len(self.file_handles) == 0:
            return True
        if all([pc.outline == 'stack' for pc in self.plot_configs]):
            if len([fh for fh in self.file_handles if fh.process.is_mc]) == 0:
                _logger.error("Requested only stack plots but only data inputs provided. Giving up")
                return False
        if len([pc for pc in self.plot_configs if pc.outline == 'stack']) > 0:
            if len([fh for fh in self.file_handles if fh.process.is_mc]) == 0:
                _logger.error("Requested at least stack plot but only data inputs provided. "
                              "Removing corresponding plot configs")
                self.plot_configs = [pc for pc in self.plot_configs if pc.outline != 'stack']
        return True

    def make_plots(self, dumped_hist_path=None):
        """
        Entry point to make plots. Reads all histograms either from provided path or projects them out of trees
        :param dumped_hist_path: path to previously dumped histograms - enables hist reading mode (optional)
        :type dumped_hist_path: str
        :return: nothing
        :rtype: None
        """
        if not self.consistency_check():
            _logger.debug('Fail consistency check')
            exit()
        if dumped_hist_path is None:
            fetched_histograms = self.project_hists()
        else:
            fetched_histograms = self.read_hists(dumped_hist_path)
        self.categorise_histograms(fetched_histograms)
        if not self.cluster_mode:
            if not self.lumi < 0:
                self.apply_lumi_weights(self.histograms)
            if hasattr(self.plot_configs, "normalise_after_cut"):
                self.cut_based_normalise(self.plot_configs.normalise_after_cut)
        #workaround due to missing worker node communication of regex process parsing

        if self.process_configs is not None:
            self.merge_histograms()
            if not self.cluster_mode:
                for pc, hists in list(self.histograms.items()):
                    for h in list(hists.values()):
                        self.output_handle.register_object(h)
        if self.syst_analyser is not None:
            self.syst_analyser.nominal_hists = self.histograms
            self.syst_analyser.retrieve_sys_hists(dumped_hist_path)
            self.syst_analyser.calculate_variations(self.histograms)
            if not self.cluster_mode:
                for syst_name in list(self.syst_analyser.systematic_variations.keys()):
                    for pc, data in list(self.syst_analyser.systematic_variations[syst_name].items()):
                        for h in list(data.values()):
                            if h is None:
                                continue
                            if syst_name not in h.GetName():
                                h.SetName(h.GetName() + '_' + syst_name)
                            self.output_handle.register_object(h)
            self.syst_analyser.calculate_total_systematics()
            #For some reason need to transfer histograms

            if self.cluster_mode:
                for tdir, obj in list(self.syst_analyser.output_handle.objects.items()):
                    self.output_handle.register_object(obj, tdir=tdir[0])
        for plot_config, data in list(self.histograms.items()):
            self.make_plot(plot_config, data)
        self.output_handle.write_and_close()

