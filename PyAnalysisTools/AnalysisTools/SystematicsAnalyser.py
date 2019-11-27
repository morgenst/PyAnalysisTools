from __future__ import division
from __future__ import print_function

import math
from builtins import object
from builtins import range
from copy import deepcopy
from itertools import product
from math import sqrt

from past.utils import old_div

import PyAnalysisTools.PlottingUtils.Formatting as fm
import PyAnalysisTools.PlottingUtils.PlottingTools as pt
import ROOT
from PyAnalysisTools.PlottingUtils import HistTools as HT
from PyAnalysisTools.PlottingUtils.BasePlotter import BasePlotter
from PyAnalysisTools.PlottingUtils.PlotConfig import get_default_color_scheme
from PyAnalysisTools.base import _logger
from PyAnalysisTools.base.YAMLHandle import YAMLLoader as yl


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
        return any([all_syst in systematic for all_syst in self.systematics])


class Systematic(object):
    """
    Class to define singe systematic uncertainty
    """

    def __init__(self, name, **kwargs):
        """
        Constructor
        :param kwargs: configuration arguments
        :type kwargs: dict
        """
        kwargs.setdefault('prefix', '')
        kwargs.setdefault('symmetrise', False)
        kwargs.setdefault('symmetrise_option', None)
        kwargs.setdefault('call', None)
        kwargs.setdefault('affects', None)
        kwargs.setdefault('samples', None)
        kwargs.setdefault('group', None)
        kwargs.setdefault('title', None)
        kwargs.setdefault('expand', None)
        kwargs.setdefault('hist_name', None)
        self.name = name
        for k, v in list(kwargs.items()):
            setattr(self, k, v)
        if self.type == 'scale' and self.prefix == '':
            self.prefix = 'weight_'

    def __str__(self):
        """
        Overloaded string operator providing formatted output
        :return: formatted string
        :rtype: str
        """
        obj_str = "Systematic uncertainty: {:s}".format(self.name)
        return obj_str

    def __repr__(self):
        """
        Overloads representation operator. Get's called e.g. if list of objects are printed
        :return: formatted string with name and attributes
        :rtype: str
        """
        return self.__str__() + '\n'

    def get_variations(self):
        if self.variation == 'updown':
            return [self.prefix + self.name + var for var in ['__1up', '__1down']]
        if 'single' in self.variation:
            if len(self.variation.split(':')) == 2:
                return [self.prefix + self.name + '__1' + self.variation.split(':')[-1]]
        return []

    def get_symmetrised_name(self):
        if not self.symmetrise:
            _logger.error("Request name of symmetrised systematic, but symmetrise set to False. Something went wrong")
            return None
        if 'up' in self.variation:
            return [self.prefix + self.name + '__1down']
        if 'down' in self.variation:
            return [self.prefix + self.name + '__1up']
        _logger.error("Catched undefined behaviour in systematics symmetrisation")


class SystematicsAnalyser(BasePlotter):
    def __init__(self, **kwargs):
        kwargs.setdefault("ncpu", 1)
        kwargs.setdefault("dump_hists", False)
        kwargs.setdefault("cluster_mode", False)
        kwargs.setdefault('store_abs_syst', False)
        self.file_handles = None
        for attr, value in list(kwargs.items()):
            setattr(self, attr.lower(), value)
        self.systematics = None
        self.systematic_hists = {}
        self.systematic_variations = {}
        self.total_systematics = {}
        self.systematic_configs = self.parse_syst_config(kwargs['systematics_config'])
        self.scale_uncerts = [s for s in self.systematic_configs if s.type == 'scale']
        self.shape_uncerts = [s for s in self.systematic_configs if s.type == 'shape']
        self.custom_uncerts = [s for s in self.systematic_configs if s.type == 'custom']
        self.xs_handle = kwargs["xs_handle"]
        self.syst_categories = [SystematicsCategory(name="Total", systematics=self.shape_uncerts, color=ROOT.kRed)]
        file_handles = [fh for fh in self.file_handles if fh.process.is_mc]
        self.disable = False
        if len(file_handles) == 0:
            self.disable = True

    @staticmethod
    def parse_syst_config(cfg_file):
        cfg = yl.read_yaml(cfg_file)
        return [Systematic(k, **v) for k, v in list(cfg.items())]

    @staticmethod
    def load_dumped_hist(arg, systematic):
        fh = arg[0]
        pc = arg[1]
        hist_name = '{:s}_{:s}_{:s}_clone_clone'.format(pc.name, fh.process.process_name, systematic)
        try:
            return pc, fh.process, fh.get_object_by_name(hist_name)
        except ValueError:
            try:
                if systematic in TheoryUncertaintyProvider.get_sherpa_uncerts():
                    return None, None, None
            except AttributeError:
                _logger.error('Could not find histogram: {:s} and process {:s}'.format(hist_name, fh.process))
            return None, None, None

    def load_dumped_hists(self, file_handles, plot_configs, systematic):
        hists = []
        for arg in product(file_handles, plot_configs):
            hists.append(self.load_dumped_hist(arg, systematic))
        return hists

    def process_histograms(self, fetched_histograms, syst):
        self.histograms = {}
        fetched_histograms = [hist_set for hist_set in fetched_histograms if all(hist_set)]
        self.categorise_histograms(fetched_histograms)
        if not self.cluster_mode:
            self.apply_lumi_weights(self.histograms)
        self.merge_histograms()
        for pc, hists in list(self.histograms.items()):
            if pc.disable_bin_merge:
                continue
            HT.merge_overflow_bins(hists)
            HT.merge_underflow_bins(hists)
        if len(self.histograms) == 0:
            return
        self.systematic_hists[syst] = deepcopy(self.histograms)

    def retrieve_sys_hists(self, dumped_hist_path=None):
        if self.disable:
            return
        file_handles = [fh for fh in self.file_handles if fh.process.is_mc]
        if len(file_handles) == 0:
            _logger.debug("Could not find any MC file handle. Nothing to do for systematics")
            return
        for systematic in self.shape_uncerts:
            self.get_shape_uncertainty(file_handles, systematic, dumped_hist_path)
        for systematic in self.scale_uncerts:
            self.get_scale_uncertainties(file_handles, systematic.get_variations(), dumped_hist_path)
        for systematic in self.custom_uncerts:
            if systematic.call is not None:
                eval(systematic.call)

    def get_symmetrised_hists(self, sys_hist, nominal_hist, new_hist_name):
        h_tmp = sys_hist.Clone(new_hist_name)
        for i in range(sys_hist.GetNbinsX() + 1):
            h_tmp.SetBinContent(i, 2. * nominal_hist.GetBinContent(i) - sys_hist.GetBinContent(i))
        return h_tmp

    def get_shape_uncertainty(self, file_handles, systematic, dumped_hist_path=None):
        for syst in systematic.get_variations():
            if dumped_hist_path is None:
                fetched_histograms = self.read_histograms(file_handles=file_handles, plot_configs=self.plot_configs,
                                                          tree_dir_name=syst)
            else:
                fetched_histograms = self.load_dumped_hists(file_handles, self.plot_configs, syst)
            if self.dump_hists:
                histograms = [it[-1] for it in fetched_histograms]
                for h in histograms:
                    h.SetName("{:s}_{:s}".format(h.GetName(), syst))
                    self.output_handle.register_object(h)
                continue
            self.process_histograms(fetched_histograms, syst)

    def get_scale_uncertainties(self, file_handles, weights, dumped_hist_path=None, disable_relative=False):
        for weight in weights:
            plot_configs = deepcopy(self.plot_configs)
            for pc in plot_configs:
                if pc.weight is None:
                    _logger.error("Try to modify plot config {:s} for systematic weights {:s} which has no weight "
                                  "set. Cannot do anything here.".format(pc.name, weight))
                    continue
                new_weight = '{:s} * ({:s} != -1111.) + ({:s}==-1111.)*1.'.format(weight, weight, weight)
                if not disable_relative:
                    pc.weight = pc.weight.replace('weight', '{:s}*({:s})'.format(pc.weight, new_weight))
                else:
                    pc.weight = pc.weight.replace('weight', '({:s})'.format(new_weight))
            if dumped_hist_path is None:
                fetched_histograms = self.read_histograms(file_handles=file_handles, plot_configs=plot_configs,
                                                          tree_dir_name="Nominal", factor_syst=weight)
            else:
                fetched_histograms = self.load_dumped_hists(file_handles, self.plot_configs, weight)

            if self.dump_hists:
                histograms = [it[-1] for it in fetched_histograms]
                for h in histograms:
                    h.SetName("{:s}_{:s}".format(h.GetName(), weight))
                    self.output_handle.register_object(deepcopy(h))
                continue
            self.process_histograms(fetched_histograms, weight.replace('weight_', ''))

    def get_fixed_scale_uncertainties(self, file_handles, scale_unc, dumped_hist_path=None, disable_relative=False):
        for unc_name, data in list(scale_unc.items()):
            if dumped_hist_path is None:
                nominal_hists = self.read_histograms(file_handles=file_handles, plot_configs=self.plot_configs,
                                                     tree_dir_name="Nominal", factor_syst='fixed_scale')
                fetched_histograms = []
                for pc, process_cfg, h in nominal_hists:
                    htmp_up = h.Clone('{:s}_{:s}_{:s}__1up'.format(pc.name, process_cfg.process_name, unc_name))
                    htmp_down = h.Clone('{:s}_{:s}_{:s}__1down'.format(pc.name, process_cfg.process_name, unc_name))
                    unc = eval(data[int(process_cfg.dsid)])
                    htmp_up.Scale(1. + unc[0])
                    htmp_down.Scale(1. + unc[1])
                    fetched_histograms += [(pc, process_cfg, htmp_up), (pc, process_cfg, htmp_down)]
                    histograms = [it[-1] for it in fetched_histograms]
                    for hist in histograms:
                        self.output_handle.register_object(deepcopy(hist))
            else:
                for var in ['__1up', '__1down']:
                    fetched_histograms = self.load_dumped_hists(file_handles, self.plot_configs, unc_name + var)
                    self.process_histograms(fetched_histograms, unc_name + var)

    def calculate_total_systematics(self):
        def rearrange_dict(keys):
            tmp = {}
            for systematic in keys:
                for plot_config, data in list(self.systematic_variations[systematic].items()):
                    for process, hist in list(data.items()):
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
                for plot_config, data in list(systematic_hists.items()):
                    for process, systematics in list(data.items()):
                        systematics = dict([s for s in iter(list(systematics.items())) if category.contains_syst(s[0])])
                        hist = list(systematics.values())[0].Clone('_'.join([plot_config.name, process, 'total_syst',
                                                                             name]))
                        if hist is None:
                            _logger.error("Systematics histogram for systematic {:s} and process {:s} is None. "
                                          "Continue.".format(name, process))
                            continue
                        for b in range(hist.GetNbinsX() + 1):
                            total_uncertainty = sqrt(
                                sum([pow(h.GetBinContent(b), 2) for h in list(systematics.values()) if h is not None]))
                            hist.SetBinContent(b, total_uncertainty)
                        if plot_config not in self.total_systematics[category][name]:
                            self.total_systematics[category][name][plot_config] = {}
                        if process not in self.total_systematics[category][name][plot_config]:
                            self.total_systematics[category][name][plot_config][process] = {}
                        self.total_systematics[category][name][plot_config][process] = hist

        # up_variation_names = [systematic for systematic in list(self.systematic_variations.keys()) if
        #                       "up" in systematic.lower()]
        down_variation_names = [systematic for systematic in list(self.systematic_variations.keys()) if
                                "down" in systematic.lower()]
        # up_variation = rearrange_dict(up_variation_names)
        down_variation = rearrange_dict(down_variation_names)
        # get_total_relative_systematics(up_variation, "up")
        get_total_relative_systematics(down_variation, "down")

    def calculate_variations(self, nominal):
        for systematic in list(self.systematic_hists.keys()):
            self.systematic_variations[systematic] = self.get_variations_single_systematic(systematic, nominal)
        if len(self.systematic_variations) == 0:
            return
        for syst in [s for s in self.systematic_configs if s.symmetrise]:
            self.systematic_variations[syst.get_symmetrised_name()[0]] = {}
            for pc in self.systematic_variations[syst.get_variations()[0]]:
                self.systematic_variations[syst.get_symmetrised_name()[0]][pc] = {}
                for process, hist in list(self.systematic_variations[syst.get_variations()[0]][pc].items()):
                    self.systematic_variations[syst.get_symmetrised_name()[0]][pc][
                        process] = self.get_symmetrised_hists(hist, nominal[pc][process],
                                                              hist.GetName().replace(syst.get_variations()[0],
                                                                                     syst.get_symmetrised_name()[0]))

    def get_variations_single_systematic(self, systematic, nominal):
        variations = {}
        for plot_config, nominal_hists in list(nominal.items()):
            variations[plot_config] = self.get_variation_for_dist(plot_config, nominal_hists, systematic)
        return variations

    def get_variation_for_dist(self, plot_config, nominal_hists, systematic):
        variations = {}
        for process, hists in list(nominal_hists.items()):
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
                    hist.SetBinContent(b, old_div((variation - nominal), nominal))
                    _logger.debug('set variation to: {:.4f} with nominal: {:.2f} '
                                  'variation: {:.2f} for syst: {:s}'.format(old_div((variation - nominal), nominal),
                                                                            nominal,
                                                                            variation,
                                                                            systematic_hist.GetName()))
                else:
                    hist.SetBinContent(b, 0.)
                if math.isnan(hist.GetBinContent(b)):
                    _logger.error('FOUND NAN om calc diff nom: {:f} and var: {:f} in hist '
                                  '{:s}'.format(nominal, variation, systematic_hist.GetName()))
                    hist.SetBinContent(b, 0.)
            return hist

        def find_plot_config():
            try:
                return [pc for pc in list(self.systematic_hists[systematic].keys()) if pc.name == plot_config.name][0]
            except IndexError:
                return None

        systematic_plot_config = find_plot_config()
        if systematic_plot_config is None:
            return nominal
        if systematic == 'theory_envelop' and process not in self.systematic_hists[systematic][systematic_plot_config]:
            hist = nominal.Clone()
            if self.store_abs_syst:
                return hist
            for b in range(nominal.GetNbinsX() + 1):
                hist.SetBinContent(b, 1.)
            return hist
        if process not in self.systematic_hists[systematic][systematic_plot_config]:
            if 'theory_envelop' in systematic and process != 'Zjets' or \
                    'weight_' + systematic in TheoryUncertaintyProvider.get_sherpa_uncerts() and process != 'Zjets':
                return None
            _logger.warning("Could not find systematic {:s} for process {:s}".format(systematic, process))
            return None
        systematic_hist = self.systematic_hists[systematic][systematic_plot_config][process]
        if self.store_abs_syst:
            return systematic_hist
        return calculate_diff(nominal, systematic_hist)

    def get_relative_unc_on_SM_total(self, plot_config, nominal):
        def get_sm_total(nominal_hists, variation):
            sm_total_hists_syst = []
            colors = []
            for category in list(self.total_systematics.keys()):
                colors.append(category.color)
                try:
                    syst_hists = self.total_systematics[category][variation][plot_config]
                except KeyError:
                    _logger.error('Could not find category {:s} for variation {:s} for plot '
                                 '{:s}'.format(category.name, variation, plot_config.name))
                    continue
                sm_total_hist_syst = None
                for process, nominal_hist in list(nominal_hists.items()):
                    if "data" in process.lower():
                        continue
                    tmp_nominal_hist = nominal_hist.Clone("_".join([nominal_hist.GetName(), variation]))
                    tmp_syst = syst_hists[process]
                    for b in range(tmp_syst.GetNbinsX()):
                        if tmp_syst.GetBinContent(b) > 1. or tmp_syst.GetBinContent(b) < 0.:
                            _logger.warn('FOUND BIN {:d} larger than 1 in {:s} with {:f}'.format(b, tmp_syst.GetName(),
                                                                                                 tmp_syst.GetBinContent(
                                                                                                     b)))
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
        syst_plot_config.ymin = -30.
        syst_plot_config.ymax = 30.
        syst_plot_config.ytitle = 'variation [%]'
        labels = {}
        syst_plot_config.color = get_default_color_scheme()
        skipped = {}

        for index, variation in enumerate(self.systematic_variations.keys()):
            sys_hists = self.systematic_variations[variation][plot_config]
            for process, hist in list(sys_hists.items()):
                if process not in labels:
                    labels[process] = []
                if process not in skipped:
                    skipped[process] = 0
                labels[process].append(variation)
                hist_base_name = "sys_overview_{:s}_{:s}".format(plot_config.name, process)
                syst_plot_config.name = hist_base_name
                if hist_base_name not in overview_hists:
                    overview_hists[hist_base_name] = pt.plot_obj(hist.Clone(hist_base_name), syst_plot_config, index=0)
                    continue
                overview_canvas = overview_hists[hist_base_name]
                if hist.GetMaximum() < 1.0 and abs(hist.GetMinimum()) < 1.0:
                    if len(labels[process]) > 0:
                        labels[process].pop(-1)
                    skipped[process] += 1
                    continue
                pt.add_object_to_canvas(overview_canvas,
                                        hist.Clone("{:s}_{:s}".format(hist_base_name, variation)),
                                        syst_plot_config,
                                        index=index - skipped[process])
        if len(labels) == 0:
            return

        list([format_plot(kv[1], labels[kv[0].split('_')[-1]], format='Line') for kv in
              iter(list(overview_hists.items()))])
        list([self.output_handle.register_object(h) for h in list(overview_hists.values())])


class TheoryUncertaintyProvider(object):
    def __init__(self, fixed_top_unc_file=None):
        self.sherpa_pdf_uncert = self.get_sherpa_uncerts()
        self.all_uneffected = False
        self.top_unc = None
        if fixed_top_unc_file is not None:
            self.top_unc = yl.read_yaml(fixed_top_unc_file)

    @staticmethod
    def get_sherpa_uncerts():
        return ['weight_pdf_uncert_MUR0.5_MUF0.5_PDF261000',
                'weight_pdf_uncert_MUR0.5_MUF1_PDF261000',
                'weight_pdf_uncert_MUR1_MUF0.5_PDF261000',
                'weight_pdf_uncert_MUR1_MUF2_PDF261000',
                'weight_pdf_uncert_MUR2_MUF1_PDF261000',
                'weight_pdf_uncert_MUR2_MUF2_PDF261000',
                'weight_pdf_uncert_MUR1_MUF1_PDF25300',
                'weight_pdf_uncert_MUR1_MUF1_PDF13000']

    @staticmethod
    def is_affected(file_handle, tree_name):
        """
        Check if file is affected by Sherpa uncertainties
        :param file_handle: input file
        :type file_handle: FileHandle
        :param tree_name: name of nominal tree
        :type tree_name: str
        :return: yes/no decision
        :rtype: bool
        """
        tree = file_handle.get_object_by_name(tree_name, tdirectory='Nominal')
        return TheoryUncertaintyProvider.check_is_affected(tree)

    @staticmethod
    def check_is_affected(tree):
        """
        Check if file is affected by Sherpa uncertainties
        :param file_handle: input file
        :type file_handle: FileHandle
        :param tree_name: name of nominal tree
        :type tree_name: str
        :return: yes/no decision
        :rtype: bool
        """
        return hasattr(tree, "weight_pdf_uncert_MUR0.5_MUF0.5_PDF261000")

    def get_envelop(self, analyser, dump_hist_path=None):
        self.fetch_uncertainties(analyser, dump_hist_path)
        self.calculate_envelop(analyser)

    def fetch_uncertainties(self, analyser, dump_hist_path=None):
        if dump_hist_path is None:
            file_handles = [fh for fh in analyser.file_handles if self.is_affected(fh, analyser.tree_name)]
        else:
            file_handles = analyser.file_handles
        if len(file_handles) == 0:
            _logger.debug("Could not find any file handle affected by theory uncertainty. Will do nothing")
            self.all_uneffected = True
            return
        analyser.get_scale_uncertainties(file_handles, self.sherpa_pdf_uncert, dump_hist_path, disable_relative=True)

    def get_top_theory_unc(self, analyser, dump_hist_path=None):
        if dump_hist_path is None:
            file_handles = [fh for fh in analyser.file_handles if
                            int(fh.process.dsid) in self.top_unc.values()[0].keys()]
        else:
            file_handles = analyser.file_handles
        analyser.get_fixed_scale_uncertainties(file_handles, self.top_unc, dump_hist_path, True)

    def get_top_uncert_names(self):
        return self.top_unc.keys()

    def calculate_envelop(self, analyser):
        def get_pc(hists, plot_config):
            """
            Required as dedicated plot config matching is needed since the weights are different and thus the equality
            check won't work
            :param hists: dictionary with plot configs and process, syst_hist dictionary for given systematic uncert
            :type hists: dict
            :param plot_config: (nominal) plot config
            :type plot_config: PlotConfig
            :return: plot config for given systematic uncertainty
            :rtype: PlotConfig
            """
            return [pc for pc in list(hists.keys()) if pc.name == plot_config.name][0]

        return
        if self.all_uneffected:
            return
        try:
            pdf_name = 'pdf_uncert_MUR1_MUF0.5_PDF261000'
            for plot_config in list(analyser.systematic_hists[pdf_name].keys()):
                for process, hist in list(analyser.systematic_hists[pdf_name][plot_config].items()):
                    nominal_hist = analyser.nominal_hists[get_pc(analyser.nominal_hists, plot_config)][process]
                    new_hist_name = hist.GetName().replace('weight_{:s}'.format(pdf_name), '') + '_theory_envelop'
                    new_hist_name = new_hist_name.replace('_clone', '')
                    envelop_up = hist.Clone(new_hist_name + '__1up')
                    envelop_down = hist.Clone(new_hist_name + '__1down')
                    for b in range(envelop_up.GetNbinsX() + 1):
                        unc_max = max([analyser.systematic_hists[sys.replace('weight_', '')]
                                       [get_pc(analyser.systematic_hists[sys.replace('weight_', '')], plot_config)]
                                       [process].GetBinContent(b) - nominal_hist.GetBinContent(b)
                                       for sys in self.sherpa_pdf_uncert])
                        nom = nominal_hist.GetBinContent(b)
                        if unc_max > 0:
                            up, down = unc_max + nom, nom - unc_max
                        else:
                            up, down = nom - unc_max, nom + unc_max
                        envelop_up.SetBinContent(b, up)
                        envelop_down.SetBinContent(b, down)
                    if 'theory_envelop__1up' not in analyser.systematic_hists:
                        analyser.systematic_hists['theory_envelop__1up'] = {}
                        analyser.systematic_hists['theory_envelop__1down'] = {}
                    if plot_config not in analyser.systematic_hists['theory_envelop__1up']:
                        analyser.systematic_hists['theory_envelop__1up'][plot_config] = {}
                        analyser.systematic_hists['theory_envelop__1down'][plot_config] = {}

                    analyser.systematic_hists['theory_envelop__1up'][plot_config][process] = deepcopy(envelop_up)
                    analyser.systematic_hists['theory_envelop__1down'][plot_config][process] = deepcopy(envelop_down)
            # for sys in self.sherpa_pdf_uncert:
            #     analyser.systematic_hists.pop(sys.replace('weight_', ''))
        except KeyError:
            _logger.debug("Could not find theory uncertainties")
            pass

    def calculate_envelop_count(self, yields):
        try:
            tmp = {syst: yields[syst].sum() for syst in self.sherpa_pdf_uncert}
            max_key, _ = max(iter(list(tmp.items())), key=lambda x: x[1])
            yields['theory_envelop'] = yields[max_key]
            # yields['theory_envelop'].extrapolated = True
        except KeyError:
            pass
