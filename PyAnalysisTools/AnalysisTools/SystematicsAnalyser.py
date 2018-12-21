import imp

import pathos.multiprocessing as mp
from math import sqrt
from copy import deepcopy
from functools import partial
from PyAnalysisTools.PlottingUtils.BasePlotter import BasePlotter
from PyAnalysisTools.PlottingUtils.PlotConfig import find_process_config
from PyAnalysisTools.base.YAMLHandle import YAMLLoader as yl
import ROOT
from PyAnalysisTools.PlottingUtils import HistTools as HT


class SystematicsCategory(object):
    def __init__(self, **kwargs):
        kwargs.setdefault("name", "total")
        kwargs.setdefault("systematics", "")
        kwargs.setdefault("color", ROOT.kBlue)
        self.name = kwargs["name"]
        self.systematics = kwargs["systematics"]
        self.color = kwargs["color"]

    def contains_syst(self, systematic):
        if self.name.lower() == "total":
            return True
        return any(map(lambda all_syst: all_syst in systematic, self.systematics))


class FixedSystematics(object):
    def __init__(self, **kwargs):
        kwargs.setdefault('variation', None)
        self.name = kwargs["name"]
        if kwargs['variation'] is not None:
            if kwargs['variation'] == 1:
                self.weights = ["weight_{:s}__1up".format(kwargs["weight"])]
            else:
                self.weights = ["weight_{:s}__1down".format(kwargs["weight"])]
        else:
            self.weights = ["weight_{:s}__1down".format(kwargs["weight"]),
                            "weight_{:s}__1up".format(kwargs["weight"])]


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
        _, self.shape_syst_config, self.scale_syst_config = \
                imp.load_source('config_systematics',
                                kwargs['systematics_config']).config_systematics('1:1')
        self.xs_handle = kwargs["xs_handle"]
        # SystematicsCategory(name="Muon", systematics=["MUON_MS"]),
        # SystematicsCategory(name="Electron", systematics=["EG_RESOLUTION_ALL"], color=ROOT.kYellow),
        shape_syst = ['{:s}__1{:s}'.format(sn, 'up' if svar == 1 else 'down') for sn, svar in self.shape_syst_config]
        scale_syst = map(lambda s: s[0], self.scale_syst_config)
        single_direction_sys = filter(lambda sn: scale_syst.count(sn) != 2, set(scale_syst))
        self.syst_categories = [SystematicsCategory(name="Total", systematics=shape_syst, color=ROOT.kRed)]
        self.fixed_systematics = [FixedSystematics(name=sn, weight=sn)
                                  for sn, _ in self.scale_syst_config if sn not in single_direction_sys]
        for syst_name in single_direction_sys:
            self.fixed_systematics.append(FixedSystematics(name=syst_name, weight=syst_name,
                                                           variation=filter(lambda s: s[0] == syst_name,
                                                                            self.scale_syst_config)[0][1]))
    def parse_systematics(self, file_handle):
        if self.systematics is not None:
            return
        self.systematics = map(lambda o: o.GetName(), file_handle.get_objects_by_type("TDirectoryFile"))
        self.systematics.remove("Nominal")

    def retrieve_sys_hists(self, file_handles):
        def process_histograms(fetched_histograms):
            self.histograms = {}
            fetched_histograms = filter(lambda hist_set: all(hist_set), fetched_histograms)
            self.categorise_histograms(fetched_histograms)
            self.apply_lumi_weights_new(self.histograms)
            if self.process_configs is not None:
                for hist_set in self.histograms.values():
                    for process_name in hist_set.keys():
                        _ = find_process_config(process_name, self.process_configs)
            self.merge_histograms()
            map(lambda hists: HT.merge_overflow_bins(hists), self.histograms.values())
            map(lambda hists: HT.merge_underflow_bins(hists), self.histograms.values())
            self.systematic_hists[systematic] = deepcopy(self.histograms)

        file_handles = filter(lambda fh: fh.is_mc, self.file_handles)
        self.parse_systematics(file_handles[0])
        for systematic in self.systematics:
            fetched_histograms = self.read_histograms(file_handles=file_handles, plot_configs=self.plot_configs,
                                                      systematic=systematic)
            process_histograms(fetched_histograms)
        for fixed_systematic in self.fixed_systematics:
            for weight in fixed_systematic.weights:
                plot_configs = deepcopy(self.plot_configs)
                for pc in plot_configs:
                    pc.weight = weight
                fetched_histograms = self.read_histograms(file_handles=file_handles, plot_configs=plot_configs,
                                                          systematic="Nominal")
                process_histograms(fetched_histograms)

    def calculate_total_systematics(self):
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
            for category in self.syst_categories:
                if category in self.total_systematics:
                    self.total_systematics[category][name] = {}
                else:
                    self.total_systematics[category] = {name: {}}
                for plot_config, data in systematic_hists.iteritems():
                    for process, systematics in data.iteritems():
                        systematics = dict(filter(lambda s: category.contains_syst(s[0]), systematics.iteritems()))
                        hist = systematics.values()[0]
                        for b in range(hist.GetNbinsX() + 1):
                            total_uncertainty = sqrt(sum([pow(hist.GetBinContent(b), 2) for hist in systematics.values()]))
                            hist.SetBinContent(b, total_uncertainty)
                        if plot_config not in self.total_systematics[category][name]:
                            self.total_systematics[category][name][plot_config] = {}
                        if process not in self.total_systematics[category][name][plot_config]:
                            self.total_systematics[category][name][plot_config][process] = {}
                        self.total_systematics[category][name][plot_config][process] = hist
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
            if process.lower() == 'data':
                continue
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

    def get_relative_unc_on_SM_total(self, plot_config, nominal):
        def get_sm_total(nominal_hists, variation):
            sm_total_hists_syst = []
            colors = []
            for category in self.total_systematics.keys():
                colors.append(category.color)
                syst_hists = self.total_systematics[category][variation][plot_config]
                sm_total_hist_syst = None
                for process, nominal_hist in nominal_hists.iteritems():
                    if "data" in process.lower():
                        continue
                    tmp_hist = nominal_hist.Clone("_".join([nominal_hist.GetName(), variation]))
                    tmp_syst = syst_hists[process]
                    for b in range(tmp_syst.GetNbinsX()):
                        tmp_syst.SetBinContent(b, 1. + tmp_syst.GetBinContent(b))
                    tmp_hist.Multiply(tmp_syst)
                    if sm_total_hist_syst is None:
                        sm_total_hist_syst = tmp_hist.Clone(
                            "SM_{:s}_{:s}_{:s}".format(category.name, nominal_hist.GetName(), variation))
                        continue
                    sm_total_hist_syst.Add(tmp_hist)
                sm_total_hists_syst.append(sm_total_hist_syst)
            return sm_total_hists_syst, colors

        sm_total_up_categorised, color_up = get_sm_total(nominal, "up")
        sm_total_down_categorised, colors_down = get_sm_total(nominal, "down")
        colors = color_up + colors_down
        return sm_total_up_categorised, sm_total_down_categorised, colors
