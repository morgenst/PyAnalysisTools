import pathos.multiprocessing as mp
from math import sqrt
from copy import deepcopy
from functools import partial
from PyAnalysisTools.PlottingUtils.BasePlotter import BasePlotter
from PyAnalysisTools.PlottingUtils.PlotConfig import find_process_config


class SystematicsAnalyser(BasePlotter):
    def __init__(self, **kwargs):
        kwargs.setdefault("ncpu", 1)
        self.file_handles = None
        for attr, value in kwargs.iteritems():
            setattr(self, attr.lower(), value)
        self.systematics = None
        self.systematic_hists = {}
        self.systematic_variations = {}
        self.total_systematics = {}

    def parse_systematics(self, file_handle):
        if self.systematics is not None:
            return
        self.systematics = map(lambda o: o.GetName(),file_handle.get_objects_by_type("TDirectoryFile"))
        self.systematics.remove("Nominal")

    def retrieve_sys_hists(self, file_handles):
        self.file_handles = file_handles
        self.parse_systematics(self.file_handles[0])
        for systematic in self.systematics:
            self.histograms = {}
            fetched_histograms = mp.ThreadPool(min(self.ncpu,
                                                   len(self.plot_configs))).map(partial(self.read_histograms,
                                                                                        file_handles=self.file_handles,
                                                                                        systematic=systematic),
                                                                                self.plot_configs)
            for plot_config, histograms in fetched_histograms:
                histograms = filter(lambda hist: hist is not None, histograms)
                self.categorise_histograms(plot_config, histograms)
            if self.process_configs is not None:
                for hist_set in self.histograms.values():
                    for process_name in hist_set.keys():
                        _ = find_process_config(process_name, self.process_configs)
            self.merge_histograms()
            self.systematic_hists[systematic] = deepcopy(self.histograms)

    def retrieve_total_systematics(self):
        def rearrange_dict(keys):
            tmp = {}
            for systematic in keys:
                for plot_config, data in self.systematic_variations[systematic].iteritems():
                    for process, hist in data.iteritems():
                        if plot_config not in tmp:
                            tmp[plot_config] = {}
                        if process not in tmp[plot_config]:
                            tmp[plot_config][process] = {}
                        tmp[plot_config][process][systematic] = hist
            return tmp

        def get_total_relative_systematics(systematic_hists, name):
            self.total_systematics[name] = {}
            for plot_config, data in systematic_hists.iteritems():
                for process, systematics in data.iteritems():
                    hist = systematics.values()[0]
                    for b in range(hist.GetNbinsX() + 1):
                        total_uncertainty = sqrt(sum([pow(hist.GetBinContent(b), 2) for hist in systematics.values()]))
                        hist.SetBinContent(b, total_uncertainty)
                    if plot_config not in self.total_systematics[name]:
                        self.total_systematics[name][plot_config] = {}
                    if process not in self.total_systematics[name][plot_config]:
                        self.total_systematics[name][plot_config][process] = {}
                    self.total_systematics[name][plot_config][process] = hist

        up_variation_names = filter(lambda systematic: "up" in systematic.lower(), self.systematic_variations.keys())
        down_variation_names = filter(lambda systematic: "down" in systematic.lower(), self.systematic_variations.keys())
        up_variation = rearrange_dict(up_variation_names)
        down_variation = rearrange_dict(down_variation_names)
        get_total_relative_systematics(up_variation, "up")
        get_total_relative_systematics(down_variation, "down")

    def calculate_variations(self, nominal):
        for systematic in self.systematic_hists.keys():
            self.systematic_variations[systematic] = self.get_variations_single_systematic(systematic, nominal)

    def get_variations_single_systematic(self, systematic, nominal):
        variations = {}
        for plot_config, nominal_hists in nominal.iteritems():
            variations[plot_config] = self.get_variation_for_dist(plot_config, nominal_hists, systematic)
        return variations

    def get_variation_for_dist(self, plot_config, nominal_hists, systematic):
        variations = {}
        for process, hists in nominal_hists.iteritems():
            variations[process] = self.get_variation_for_process(process, hists, plot_config, systematic)
        return variations

    def get_variation_for_process(self, process, nominal, plot_config, systematic):
        def calculate_diff(nominal_hist, systematic_hist):
            hist = nominal_hist.Clone()
            for b in range(nominal_hist.GetNbinsX() + 1):
                nominal = nominal_hist.GetBinContent(b)
                variation = systematic_hist.GetBinContent(b)
                if nominal != 0:
                    hist.SetBinContent(b, (variation - nominal) / nominal)
                else:
                    hist.SetBinContent(b, 0.)
            return hist

        def find_plot_config():
            return filter(lambda pc: pc.name == plot_config.name, self.systematic_hists[systematic].keys())[0]

        systematic_plot_config = find_plot_config()
        systematic_hist = self.systematic_hists[systematic][systematic_plot_config][process]
        return calculate_diff(nominal, systematic_hist)
