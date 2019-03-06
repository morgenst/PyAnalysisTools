import re
from collections import OrderedDict

import pathos.multiprocessing as mp
import traceback
from functools import partial
from itertools import product
from PyAnalysisTools.base import _logger
from PyAnalysisTools.ROOTUtils.FileHandle import FileHandle
from PyAnalysisTools.PlottingUtils.PlotConfig import propagate_common_config
from PyAnalysisTools.PlottingUtils import Formatting as fm
from PyAnalysisTools.PlottingUtils import HistTools as ht
from PyAnalysisTools.PlottingUtils import set_batch_mode
from PyAnalysisTools.PlottingUtils.PlotConfig import parse_and_build_plot_config, parse_and_build_process_config, \
    get_histogram_definition, find_process_config, merge_plot_configs


class BasePlotter(object):
    def __init__(self, **kwargs):
        self.plot_configs = None
        self.lumi = None
        kwargs.setdefault("batch", True)
        kwargs.setdefault("process_config_file", None) #deprecated, for now kept for backwards compatibility
        kwargs.setdefault("process_config_files", None)
        kwargs.setdefault("xs_config_file", None)
        kwargs.setdefault("read_hist", False)
        kwargs.setdefault("friend_directory", None)
        kwargs.setdefault("friend_tree_names", None)
        kwargs.setdefault("friend_file_pattern", None)
        kwargs.setdefault("plot_config_files", [])
        kwargs.setdefault("nfile_handles", 1)
        kwargs.setdefault('syst_tree_name', None)
        kwargs.setdefault('cluster_config', None)
        kwargs.setdefault('redraw', False)

        self.event_yields = {}
        set_batch_mode(kwargs["batch"])

        if kwargs['cluster_config'] is not None:
            self.cluster_setup(kwargs['cluster_config'])
            return
        if kwargs['redraw']:
            self.file_handles = []
            return

        for attr, value in kwargs.iteritems():
            setattr(self, attr.lower(), value)
        set_batch_mode(kwargs["batch"])
        self.process_configs = self.parse_process_config()
        self.parse_plot_config()
        self.split_mc_campaigns = False
        if self.plot_configs is not None and any([not pc.merge_mc_campaigns for pc in self.plot_configs]) \
                and self.process_config_files is not None:
            self.split_mc_campaigns = True
            self.add_mc_campaigns()

        self.event_yields = {}
        self.file_handles = [FileHandle(file_name=input_file, dataset_info=kwargs["xs_config_file"],
                                        split_mc=self.split_mc_campaigns, friend_directory=kwargs["friend_directory"],
                                        switch_off_process_name_analysis=False,
                                        friend_tree_names=kwargs["friend_tree_names"],
                                        friend_pattern=kwargs["friend_file_pattern"])
                             for input_file in self.input_files]
        self.filter_missing_friends()

    def cluster_setup(self, config):
        self.file_handles = [FileHandle(file_name=config.file_name, dataset_info=config.extra_args["xs_config_file"])]

    def parse_process_config(self):
        """
        Parse process config file and build process configs
        :return: list of build process configs from config file
        :rtype: list
        """
        if self.process_config_files is None and self.process_config_file is None:
            return None
        if self.process_config_file is not None:
            _logger.error("Single Process configs are deprecated. Please update you argument parser to "
                          "process_config_files (NOTE the additional s).")
        if self.process_config_files is None:
            process_config = parse_and_build_process_config(self.process_config_file)
        else:
            process_config = parse_and_build_process_config(self.process_config_files)
        return process_config

    def parse_plot_config(self):
        _logger.debug("Try to parse plot config file")
        unmerged_plot_configs = []
        for plot_config_file in self.plot_config_files:
            config = parse_and_build_plot_config(plot_config_file)
            unmerged_plot_configs.append(config)
        self.plot_configs, common_config = merge_plot_configs(unmerged_plot_configs)
        if self.lumi is None:
            if hasattr(common_config, "lumi"):
                self.lumi = common_config.lumi
            else:
                _logger.warning("No lumi information provided. Using 1 fb-1")
                self.lumi = 1.
        if common_config is not None:
            propagate_common_config(common_config, self.plot_configs)

    def filter_missing_friends(self, load_friend=False):
        """
        Remove files from file list for which no friend tree was found

        :param load_friend: enable/disable filter
        :type load_friend: bool
        :return:
        :rtype: None
        """

        if not load_friend:
            return
        self.file_handles = filter(lambda fh: len(fh.friends) > 0 or fh.is_data, self.file_handles)

    @staticmethod
    def load_atlas_style():
        """
        Load ATLAS plotting style

        :return: None
        :rtype: None
        """
        fm.load_atlas_style()

    def apply_lumi_weights(self, histograms):
        """
        Weight histograms according to process cross section and luminosity. If MC samples are split in several
        production campaigns and the luminosity information is provided as a dictionary with the campaign name as key
        and luminosity as value each campaign will be scaled to this luminosity and processes will be added up later
        :param histograms: all plottable objects
        :type histograms: dict
        :return: nothing
        :rtype: None
        """
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
                            _logger.error('Could not find MC campaign information, but lumi was provided per MC '
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
                ht.scale(hist, cross_section_weight)

    def read_cutflows(self):
        """
        Read cutflow histograms from input file and store total number of events in event yields dictionary

        :return: None
        :rtype: None
        """
        for file_handle in self.file_handles:
            process = file_handle.process
            if process is None:
                _logger.warning("Could not parse process for file {:s}". format(file_handle.file_name))
                continue
            if file_handle.process in self.event_yields:
                self.event_yields[process] += file_handle.get_number_of_total_events()
            else:
                self.event_yields[process] = file_handle.get_number_of_total_events()

    def fetch_histograms(self, data, systematic="Nominal"):
        file_handle, plot_config = data
        if file_handle.process is None or "data" in file_handle.process.lower() and plot_config.no_data:
            return [None, None, None]
        tmp = self.retrieve_histogram(file_handle, plot_config, systematic)
        if not plot_config.merge_mc_campaigns:
            return plot_config, file_handle.process, tmp
        return plot_config, file_handle.process, tmp

    def fetch_histograms_new(self, data, systematic="Nominal", factor_syst=''):
        file_handle, plot_config = data
        if file_handle.process is None or "data" in file_handle.process.lower() and plot_config.no_data:
            return [None, None, None]
        tmp = self.retrieve_histogram(file_handle, plot_config, systematic, factor_syst)
        tmp.SetName(tmp.GetName().split('%%')[0]+tmp.GetName().split('%%')[-1])
        return plot_config, file_handle.process, tmp

    def fetch_plain_histograms(self, data, systematic="Nominal"):
        file_handle, plot_config = data
        if "data" in file_handle.process.lower() and plot_config.no_data:
            return
        try:
            tn = self.tree_name
            if self.syst_tree_name is not None and file_handle.is_mc:
                tn = self.syst_tree_name
            hist = file_handle.get_object_by_name("{:s}/{:s}".format(tn, plot_config.dist), systematic)
        except ValueError:
            #This happens if cut is not passed
            return [None, None, None]
        hist.SetName("{:s}_{:s}".format(hist.GetName(), file_handle.process))
        return plot_config, file_handle.process, hist

    def retrieve_histogram(self, file_handle, plot_config, systematic="Nominal", factor_syst = ''):
        """
        Read data from ROOT file and build histogram according to definition in plot_config

        :param file_handle: ROOT file handle
        :type file_handle: FileHandle
        :param plot_config: plot configuration including distribution and axis definitions
        :type plot_config: PlotConfig
        :param systematic:
        :type systematic: str
        :return: filled histogram - dimension depends on request in plot config
        :rtype: THX
        """
        file_handle.open()
        file_handle.reset_friends()
        try:
            hist = get_histogram_definition(plot_config, systematic, factor_syst)
        except ValueError as e:
            _logger.error("Could not build histogram for {:s}. Likely issue with log-scale and \
            range settings.".format(plot_config.name))
            print traceback.print_exc()
            return None
        try:
            weight = None
            selection_cuts = ""
            if plot_config.weight is not None:
                weight = plot_config.weight
            if plot_config.process_weight is not None:
                match = filter(lambda k: re.match(k.replace('re.', ''), file_handle.process),
                               plot_config.process_weight.keys())
                if len(match) > 0:
                    process_weight = plot_config.process_weight[match[0]]
                    if weight is not None:
                        weight += '* ({:s})'.format(process_weight)
                    else:
                        weight = process_weight
            if plot_config.cuts:
                if isinstance(plot_config.cuts, str):
                    plot_config.cuts = plot_config.split("&&")
                mc_cuts = filter(lambda cut: "MC:" in cut, plot_config.cuts)
                data_cuts = filter(lambda cut: "DATA:" in cut, plot_config.cuts)
                for mc_cut in mc_cuts:
                    plot_config.cuts.pop(plot_config.cuts.index(mc_cut))
                    if not "data" in file_handle.process:
                        selection_cuts += "{:s} && ".format(mc_cut.replace("MC:", ""))
                for data_cut in data_cuts:
                    plot_config.cuts.pop(plot_config.cuts.index(data_cut))
                    if "data" in file_handle.process:
                        selection_cuts += "{:s} && ".format(data_cut.replace("DATA:", ""))
                if len(plot_config.cuts) > 0:
                    selection_cuts += "&&".join(plot_config.cuts)

            if plot_config.blind and find_process_config(file_handle.process, self.process_configs).type == "Data":
                if selection_cuts == "":
                    selection_cuts = "!({:s})".format(" && ".join(plot_config.blind))
                else:
                    selection_cuts = "({:s}) && !({:s})".format(selection_cuts, " && ".join(plot_config.blind))
            try:
                if plot_config.merge_mc_campaigns:
                    hist.SetName("{:s}_{:s}".format(hist.GetName(), file_handle.process))
                else:
                    hist.SetName("{:s}_{:s}".format(hist.GetName(), file_handle.process))
                selection_cuts = selection_cuts.rstrip().rstrip("&&")
                tn = self.tree_name
                if self.syst_tree_name is not None and file_handle.is_mc:
                    tn = self.syst_tree_name
                file_handle.fetch_and_link_hist_to_tree(tn, hist, plot_config.dist, selection_cuts,
                                                        tdirectory=systematic, weight=weight)
            except TypeError: #RuntimeError:
                _logger.error("Unable to retrieve hist {:s} for {:s}.".format(hist.GetName(), file_handle.file_name))
                _logger.error("Dist: {:s} and cuts: {:s}.".format(plot_config.dist, selection_cuts))
                return None
            except Exception as e:
                _logger.error("Catched exception for process "
                              "{:s} and plot_config {:s}".format(file_handle, file_handle.process, plot_config.name))
                print traceback.print_exc()
                return None
            #hist.SetName(hist.GetName() + "_" + file_handle.process)
            _logger.debug("try to access config for process {:s}".format(file_handle.process))
            if self.process_configs is None:
                return hist
            process_config = find_process_config(file_handle.process, self.process_configs).name
            if process_config is None:
                _logger.error("Could not find process config for {:s}".format(file_handle.process))
                return None

        except Exception as e:
            _logger.error("Catched exception for "
                          "process {:s} and plot_config {:s}".format(file_handle.process, plot_config.name))
            print traceback.print_exc()
            return None
        return hist

    def read_histograms(self, file_handles, plot_configs, systematic="Nominal", factor_syst=''):
        cpus = min(self.ncpu, len(plot_configs)) * min(self.nfile_handles, len(file_handles))
        comb = product(file_handles, plot_configs)
        if cpus > 0:
            pool = mp.ProcessPool(nodes=cpus)
            histograms = pool.map(partial(self.fetch_histograms_new, systematic=systematic), comb)
        else:
            histograms = []
            for i in comb:
                hist = self.fetch_histograms_new(i, systematic=systematic, factor_syst=factor_syst)
                histograms.append(hist)
                #histograms.append(self.fetch_histograms(i, systematic=systematic))
        return histograms

    def read_histograms_plain(self, file_handle, plot_configs, systematic="Nominal"):
        cpus = min(self.ncpu, len(plot_configs)) * min(self.nfile_handles, len(file_handle))
        comb = product(file_handle, plot_configs)
        if cpus > 1:
            pool = mp.ProcessPool(nodes=cpus)
            histograms = pool.map(partial(self.fetch_plain_histograms, systematic=systematic), comb)
        else:
            histograms = []
            for i in comb:
                histograms.append(self.fetch_plain_histograms(i, systematic=systematic))
        return histograms

    def categorise_histograms(self, histograms):
        _logger.debug("categorising {:d} histograms".format(len(histograms)))
        for plot_config, process, hist in histograms:
            if hist is None:
                _logger.warning("hist for process {:s} is None".format(process))
                continue
            try:
                if process not in self.histograms[plot_config].keys():
                    self.histograms[plot_config][process] = hist
                else:
                    self.histograms[plot_config][process].Add(hist)
            except KeyError:
                self.histograms[plot_config] = {process: hist}

    @staticmethod
    def merge_old(histograms, process_configs):
        for process, process_config in process_configs.iteritems():
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

    @staticmethod
    def merge(histograms, process_configs):
        for process in histograms.keys():
            parent_process = find_process_config(process, process_configs).name
            if parent_process not in histograms.keys():
                new_hist_name = histograms[process].GetName().replace(process, parent_process)
                histograms[parent_process] = histograms[process].Clone(new_hist_name)
            else:
                histograms[parent_process].Add(histograms[process])
            histograms.pop(process)

    def merge_histograms(self):
        for plot_config, histograms in self.histograms.iteritems():
            if plot_config.merge:
                self.merge(histograms, self.process_configs)
