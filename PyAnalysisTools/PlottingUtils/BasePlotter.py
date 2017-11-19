import pathos.multiprocessing as mp
import traceback
from functools import partial
from PyAnalysisTools.base import _logger
from PyAnalysisTools.ROOTUtils.FileHandle import FileHandle
from PyAnalysisTools.PlottingUtils.PlotConfig import propagate_common_config
from PyAnalysisTools.PlottingUtils import Formatting as fm
from PyAnalysisTools.PlottingUtils import set_batch_mode
from PyAnalysisTools.PlottingUtils.PlotConfig import parse_and_build_plot_config, parse_and_build_process_config, \
    get_histogram_definition, find_process_config, merge_plot_configs


class BasePlotter(object):
    def __init__(self, **kwargs):
        self.plot_configs = None
        self.lumi = None
        kwargs.setdefault("batch", True)
        kwargs.setdefault("process_config_file", None)
        kwargs.setdefault("xs_config_file", None)
        kwargs.setdefault("read_hist", False)
        for attr, value in kwargs.iteritems():
            setattr(self, attr.lower(), value)
        set_batch_mode(kwargs["batch"])
        self.process_configs = self.parse_process_config()
        self.parse_plot_config()
        self.load_atlas_style()
        self.event_yields = {}
        self.file_handles = [FileHandle(file_name=input_file, dataset_info=kwargs["xs_config_file"])
                             for input_file in self.input_files]

    def parse_process_config(self):
        if self.process_config_file is None:
            return None
        process_config = parse_and_build_process_config(self.process_config_file)
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

    @staticmethod
    def load_atlas_style():
        fm.load_atlas_style()

    def read_cutflows(self):
        for file_handle in self.file_handles:
            if file_handle.process in self.event_yields:
                self.event_yields[file_handle.process] += file_handle.get_number_of_total_events()
            else:
                self.event_yields[file_handle.process] = file_handle.get_number_of_total_events()

    def fetch_histograms(self, file_handle, plot_config, systematic="Nominal"):
        if "data" in file_handle.process.lower() and plot_config.no_data:
            return
        tmp = self.retrieve_histogram(file_handle, plot_config, systematic)
        return file_handle.process, tmp

    def fetch_plain_histograms(self, file_handle, plot_config, systematic="Nominal"):
        if "data" in file_handle.process.lower() and plot_config.no_data:
            return
        hist = file_handle.get_object_by_name("{:s}/{:s}".format(self.tree_name, plot_config.dist), systematic)
        hist.SetName("{:s}_{:s}".format(hist.GetName(), file_handle.process))
        return file_handle.process, hist

    def retrieve_histogram(self, file_handle, plot_config, systematic="Nominal"):
        file_handle.open()
        hist = get_histogram_definition(plot_config)
        try:
            weight = None
            selection_cuts = ""
            if plot_config.weight is not None:
                weight = plot_config.weight
            if plot_config.cuts:
                mc_cuts = filter(lambda cut: "MC:" in cut, plot_config.cuts)
                for mc_cut in mc_cuts:
                    plot_config.cuts.pop(plot_config.cuts.index(mc_cut))
                    if not "data" in file_handle.process:
                        selection_cuts += "{:s} && ".format(mc_cut.replace("MC:", ""))
                selection_cuts += "&&".join(plot_config.cuts)
            if plot_config.blind and self.process_configs[file_handle.process].type == "Data":
                selection_cuts = "({:s}) && !({:s})".format(selection_cuts, " && ".join(plot_config.blind))
            file_handle.fetch_and_link_hist_to_tree(self.tree_name, hist, plot_config.dist, selection_cuts,
                                                    tdirectory=systematic, weight=weight)
            hist.SetName(hist.GetName() + "_" + file_handle.process)
            _logger.debug("try to access config for process %s" % file_handle.process)
            if self.process_configs is None:
                return hist
            process_config = find_process_config(file_handle.process, self.process_configs)
            if process_config is None:
                _logger.error("Could not find process config for {:s}".format(file_handle.process))
                return None

        except Exception as e:
            print traceback.print_exc()
            raise e

        return hist

    def read_histograms(self, plot_config, file_handles, systematic="Nominal"):
        if not self.read_hist:
            histograms = mp.ThreadPool(min(self.nfile_handles,
                                           len(file_handles))).map(partial(self.fetch_histograms,
                                                                           plot_config=plot_config,
                                                                           systematic=systematic), file_handles)
        else:
            histograms = mp.ThreadPool(min(self.nfile_handles,
                                           len(file_handles))).map(partial(self.fetch_plain_histograms,
                                                                           plot_config=plot_config,
                                                                           systematic=systematic), file_handles)
        return plot_config, histograms

    def categorise_histograms(self, plot_config, histograms):
        _logger.debug("categorising {:d} histograms".format(len(histograms)))
        for process, hist in histograms:
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
    def merge(histograms, process_configs):
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

    def merge_histograms(self):
        for plot_config, histograms in self.histograms.iteritems():
            if plot_config.merge:
                self.merge(histograms, self.process_configs)
