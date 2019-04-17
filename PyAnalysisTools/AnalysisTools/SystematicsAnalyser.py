import imp
import math
from itertools import product
from math import sqrt
from copy import deepcopy
from PyAnalysisTools.PlottingUtils.BasePlotter import BasePlotter
from PyAnalysisTools.PlottingUtils.PlotConfig import find_process_config, get_default_color_scheme
import PyAnalysisTools.PlottingUtils.PlottingTools as pt
import PyAnalysisTools.PlottingUtils.Formatting as fm
import ROOT
from PyAnalysisTools.PlottingUtils import HistTools as HT
from PyAnalysisTools.base import _logger


def parse_syst_config(config_file):
    _, shape_syst, scale_syst = imp.load_source('config_systematics', config_file).config_systematics('1:1')
    return shape_syst, scale_syst


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
        kwargs.setdefault("dump_hists", False)
        kwargs.setdefault("cluster_mode", False)
        self.file_handles = None
        for attr, value in kwargs.iteritems():
            setattr(self, attr.lower(), value)
        self.systematics = None
        self.systematic_hists = {}
        self.systematic_variations = {}
        self.total_systematics = {}
        self.shape_syst_config, self.scale_syst_config = parse_syst_config(kwargs['systematics_config'])
        self.xs_handle = kwargs["xs_handle"]
        # SystematicsCategory(name="Muon", systematics=["MUON_MS"]),
        # SystematicsCategory(name="Electron", systematics=["EG_RESOLUTION_ALL"], color=ROOT.kYellow),
        shape_syst = ['{:s}__1{:s}'.format(sn, 'up' if svar == 1 else 'down') for sn, svar in self.shape_syst_config]
        scale_syst = map(lambda s: s[0], self.scale_syst_config)
        single_direction_sys = filter(lambda sn: scale_syst.count(sn) != 2, set(scale_syst))
        self.syst_categories = [SystematicsCategory(name="Total", systematics=shape_syst, color=ROOT.kRed)]

        self.fixed_systematics = [FixedSystematics(name=sn, weight=sn)
                                  for sn in set(scale_syst) if sn not in single_direction_sys]
        for syst_name in single_direction_sys:
            self.fixed_systematics.append(FixedSystematics(name=syst_name, weight=syst_name,
                                                           variation=filter(lambda s: s[0] == syst_name,
                                                                            self.scale_syst_config)[0][1]))
        file_handles = filter(lambda fh: fh.process.is_mc, self.file_handles)
        self.disable = False
        if len(file_handles) == 0:
            self.disable = True
        try:
            self.parse_systematics(filter(lambda fh: fh.process.is_mc, file_handles)[0])
        except IndexError:
            _logger.error("Could not parse any systematics as no MC file handles are provided")

    def parse_systematics(self, file_handle):
        if self.systematics is not None:
            return
        self.systematics = map(lambda o: o.GetName(), file_handle.get_objects_by_type("TDirectoryFile"))
        self.systematics.remove("Nominal")

    @staticmethod
    def load_dumped_hist(arg, systematic):
        fh = arg[0]
        pc = arg[1]
        hist_name = '{:s}_{:s}_{:s}_clone_clone'.format(pc.name, fh.process.process_name, systematic)
        try:
            return pc, fh.process, fh.get_object_by_name(hist_name)
        except ValueError:
            _logger.error('Could not find histogram: {:s}'.format(hist_name))
            return None, None, None

    def load_dumped_hists(self, file_handles, plot_configs, systematic):
        l = []
        for arg in product(file_handles, plot_configs):
            l.append(self.load_dumped_hist(arg, systematic))
        return l

    def retrieve_sys_hists(self, dumped_hist_path=None):
        def process_histograms(fetched_histograms, syst):
            self.histograms = {}
            fetched_histograms = filter(lambda hist_set: all(hist_set), fetched_histograms)
            self.categorise_histograms(fetched_histograms)
            if not self.cluster_mode:
                self.apply_lumi_weights(self.histograms)
            self.merge_histograms()
            map(lambda hists: HT.merge_overflow_bins(hists), self.histograms.values())
            map(lambda hists: HT.merge_underflow_bins(hists), self.histograms.values())
            self.systematic_hists[syst] = deepcopy(self.histograms)

        if self.disable:
            return
        file_handles = filter(lambda fh: fh.process.is_mc, self.file_handles)
        if len(file_handles) == 0:
            return
        for systematic in self.systematics:
            if dumped_hist_path is None:
                fetched_histograms = self.read_histograms(file_handles=file_handles, plot_configs=self.plot_configs,
                                                          systematic=systematic)
            else:
                fetched_histograms = self.load_dumped_hists(file_handles, self.plot_configs, systematic)
            if self.dump_hists:
                histograms = map(lambda it: it[-1], fetched_histograms)
                for h in histograms:
                    h.SetName("{:s}_{:s}".format(h.GetName(), systematic))
                    self.output_handle.register_object(h)
                continue
            process_histograms(fetched_histograms, systematic)
        for fixed_systematic in self.fixed_systematics:
            for weight in fixed_systematic.weights:
                # if weight == 'weight_MUON_EFF_TTVA_SYS__1down' or 'weight_MUON_EFF_ISO' in weight or 'weight_MUON_EFF_RECO' in weight or 'JvtEffic' in weight:
                #     continue
                plot_configs = deepcopy(self.plot_configs)
                for pc in plot_configs:
                    pc.weight = pc.weight.replace('weight', "{:s}*{:s}".format(pc.weight, weight))
                if dumped_hist_path is None:
                    fetched_histograms = self.read_histograms(file_handles=file_handles, plot_configs=plot_configs,
                                                              systematic="Nominal", factor_syst=weight)
                else:
                    fetched_histograms = self.load_dumped_hists(file_handles, self.plot_configs, weight)

                if self.dump_hists:
                    histograms = map(lambda it: it[-1], fetched_histograms)
                    for h in histograms:
                        h.SetName("{:s}_{:s}".format(h.GetName(), weight))
                        self.output_handle.register_object(deepcopy(h))
                    continue
                process_histograms(fetched_histograms, weight.replace('weight_', ''))

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
                if math.isnan(hist.GetBinContent(b)):
                    _logger.error('FOUND NAN om calc diff nom: {:f} and var: {:f} in hist {:s}'.format(nominal,
                                                                                                       variation,
                                                                                                       systematic_hist.GetName()))
                    exit(0)
            return hist

        def find_plot_config():
            return filter(lambda pc: pc.name == plot_config.name, self.systematic_hists[systematic].keys())[0]

        systematic_plot_config = find_plot_config()
        if systematic_plot_config is None:
            return nominal
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
                    tmp_nominal_hist = nominal_hist.Clone("_".join([nominal_hist.GetName(), variation]))
                    tmp_syst = syst_hists[process]
                    for b in range(tmp_syst.GetNbinsX()):
                        if tmp_syst.GetBinContent(b) > 1. or tmp_syst.GetBinContent(b) < 0.:
                            _logger.warn('FOUND BIN {:d} larger than 1 in {:s} with {:f}'.format(b, tmp_syst.GetName(),
                                                                                                 tmp_syst.GetBinContent(b)))
                            tmp_syst.SetBinContent(b, 1.)
                            continue
                        tmp_syst.SetBinContent(b, 1. + tmp_syst.GetBinContent(b))
                    for b in range(tmp_nominal_hist.GetNbinsX()):
                        if math.isnan(tmp_syst.GetBinContent(b)):
                            _logger.error('FOUND NAN in {:s} at bin {:d}. Exit now'.format(tmp_syst.GetName(), b))
                            exit(0)

                    tmp_nominal_hist.Multiply(tmp_syst)
                    if sm_total_hist_syst is None:
                        sm_total_hist_syst = tmp_nominal_hist.Clone(
                            "SM_{:s}_{:s}_{:s}".format(category.name, nominal_hist.GetName(), variation))
                        continue
                    sm_total_hist_syst.Add(tmp_nominal_hist)
                sm_total_hists_syst.append(sm_total_hist_syst)
            return sm_total_hists_syst, colors

        sm_total_up_categorised, color_up = get_sm_total(nominal, "up")
        sm_total_down_categorised, colors_down = get_sm_total(nominal, "down")
        colors = color_up + colors_down
        return sm_total_up_categorised, sm_total_down_categorised, colors

    def make_overview_plots(self, plot_config):
        """
        Make summary plot of each single systematic uncertainty for variable defined in plot_config for a single process
        :param plot_config:
        :type plot_config:
        :return:
        :rtype:
        """
        def format_plot(canvas, labels, **kwargs):
            fm.decorate_canvas(canvas, plot_config)
            fm.add_legend_to_canvas(canvas, labels=labels, **kwargs)

        overview_hists = {}
        syst_plot_config = deepcopy(plot_config)
        syst_plot_config.name = "syst_overview_{:s}".format(plot_config.name)
        syst_plot_config.logy = False
        syst_plot_config.ymin = -50.
        syst_plot_config.ymax = 50.
        syst_plot_config.ytitle = 'variation [%]'
        labels = []
        syst_plot_config.color = get_default_color_scheme()
        skipped = 0
        for index, variation in enumerate(self.systematic_variations.keys()):
            labels.append(variation)
            sys_hists = self.systematic_variations[variation][plot_config]
            for process, hist in sys_hists.iteritems():
                hist_base_name = "sys_overview_{:s}_{:s}".format(plot_config.name, process)
                syst_plot_config.name = hist_base_name
                if hist_base_name not in overview_hists:
                    overview_hists[hist_base_name] = pt.plot_obj(hist.Clone(hist_base_name), syst_plot_config, index=0)
                    continue
                overview_canvas = overview_hists[hist_base_name]
                if hist.GetMaximum() < 0.1 and abs(hist.GetMinimum()) < 0.1:
                    if len(labels) > 0:
                        labels.pop(-1)
                    skipped += 1
                    continue
                pt.add_object_to_canvas(overview_canvas,
                                        hist.Clone("{:s}_{:s}".format(hist_base_name, variation)),
                                        syst_plot_config,
                                        index=index - skipped)
        if len(labels) == 0:
            return

        map(lambda c: format_plot(c, labels, format = 'Line'), overview_hists.values())
        map(lambda h: self.output_handle.register_object(h), overview_hists.values())