import operator
from operator import add

import numpy as np
import PyAnalysisTools.PlottingUtils.PlottingTools as Pt
import PyAnalysisTools.PlottingUtils.Formatting as Ft
try:
    from tabulate.tabulate import tabulate
except ImportError:
    from tabulate import tabulate
from collections import defaultdict
from PyAnalysisTools.base import _logger, InvalidInputError
from PyAnalysisTools.ROOTUtils.FileHandle import FileHandle as FH
from PyAnalysisTools.PlottingUtils import set_batch_mode
from PyAnalysisTools.PlottingUtils.HistTools import scale
from PyAnalysisTools.AnalysisTools.XSHandle import XSHandle
from PyAnalysisTools.PlottingUtils.PlotConfig import parse_and_build_process_config, find_process_config, PlotConfig
from PyAnalysisTools.base.OutputHandle import OutputFileHandle
from PyAnalysisTools.PlottingUtils.Plotter import Plotter as pl
from numpy.lib.recfunctions import rec_append_fields
from PyAnalysisTools.AnalysisTools.RegionBuilder import RegionBuilder
from PyAnalysisTools.base.YAMLHandle import YAMLLoader
from PyAnalysisTools.AnalysisTools.MLHelper import Root2NumpyConverter
from PyAnalysisTools.AnalysisTools.StatisticsTools import get_signal_acceptance


class CommonCutFlowAnalyser(object):
    def __init__(self, **kwargs):
        kwargs.setdefault("lumi", None)
        kwargs.setdefault("process_config", None)
        self.event_numbers = dict()
        self.lumi = kwargs["lumi"]
        self.xs_handle = XSHandle(kwargs["dataset_config"])
        self.file_handles = [FH(file_name=fn, dataset_info=kwargs["dataset_config"])for fn in kwargs["file_list"]]
        if kwargs["process_config"] is not None:
            self.process_configs = parse_and_build_process_config(kwargs["process_config"])
        self.expand_process_configs()
        map(self.load_dxaod_cutflows, self.file_handles)
        self.dtype = [("cut", "S300"), ("yield", "f4"), ("yield_unc", "f4"), ("eff", float), ("eff_total", float)]
        if kwargs["output_dir"] is not None:
            self.output_handle = OutputFileHandle(output_dir=kwargs["output_dir"])

    def load_dxaod_cutflows(self, file_handle):
        process = file_handle.process
        if process is None:
            _logger.error("Parsed NoneType process from {:s}".format(file_handle.file_name))
            return
        if process not in self.event_numbers:
            self.event_numbers[process] = file_handle.get_number_of_total_events()
        else:
            self.event_numbers[process] += file_handle.get_number_of_total_events()

    def expand_process_configs(self):
        if self.process_configs is not None:
            for fh in self.file_handles:
                _ = find_process_config(fh.process_with_mc_campaign, self.process_configs)

    def stringify(self, cutflow):
        def format_yield(value, uncertainty):
            if value > 10000.:
                return "{:.3e}".format(value)
            else:
                return "{:.2f}".format(value)
        if not self.raw:
            cutflow = np.array([(cutflow[i]["cut"],
                                 format_yield(cutflow[i]["yield"], cutflow[i]["yield_unc"]),
                                 #cutflow[i]["eff"],
                                 cutflow[i]["eff_total"]) for i in range(len(cutflow))],
                               dtype=[("cut", "S100"), ("yield", "S100"), ("eff_total", float)]) #("eff", float),
        else:
            cutflow = np.array([(cutflow[i]["cut"],
                                 format_yield(cutflow[i]["yield_raw"], cutflow[i]["yield_unc_raw"]),
                                 #cutflow[i]["eff"],
                                 cutflow[i]["eff_total"]) for i in range(len(cutflow))],
                               dtype=[("cut", "S100"), ("yield_raw", "S100"), ("eff_total", float)]) #("eff", float),
        return cutflow

    def print_cutflow_table(self):
        available_cutflows = self.cutflow_tables.keys()
        print "######## Selection menu  ########"
        print "Available cutflows for printing: "
        print "--------------------------------"
        for i, region in enumerate(available_cutflows):
            print i, ")", region
        print "a) all"
        user_input = raw_input("Please enter your selection (space or comma separated). Hit enter to select default (BaseSelection) ")
        if user_input == "":
            selections = ["BaseSelection"]
        elif user_input.lower() == "a":
            selections = available_cutflows
        elif "," in user_input:
            selections = [available_cutflows[i] for i in map(int, user_input.split(","))]
        elif "," not in user_input:
            selections = [available_cutflows[i] for i in map(int, user_input.split())]
        else:
            print "{:s}Invalid input {:s}. Going for default.\033[0m".format("\033[91m", user_input)
            selections = ["BaseSelection"]
        for selection, cutflow in self.cutflow_tables.iteritems():
            if selection not in selections:
                continue
            print
            print "Cutflow for region %s" % selection
            print cutflow

    def make_cutflow_tables(self):
        for systematic in self.systematics:
            self.make_cutflow_table(systematic)


class ExtendedCutFlowAnalyser(CommonCutFlowAnalyser):
    """
    Extended cutflow analyser building additional cuts not stored in cutflow hist
    """
    def __init__(self, **kwargs):
        kwargs.setdefault("output_file_name", None)
        kwargs.setdefault("no_merge", False)
        kwargs.setdefault("raw", False)
        kwargs.setdefault("output_dir", None)
        kwargs.setdefault("format", "plain")
        super(ExtendedCutFlowAnalyser, self).__init__(**kwargs)
        self.event_yields = {}

        self.selection = RegionBuilder(**YAMLLoader.read_yaml(kwargs["selection_config"])["RegionBuilder"])
        self.converter = Root2NumpyConverter(["weight"])
        self.cutflow_tables = {}
        self.cutflows = {}
        if kwargs["output_dir"] is not None:
            self.output_handle = OutputFileHandle(output_dir=kwargs["output_dir"])
        for k, v in kwargs.iteritems():
            if not hasattr(self, k):
                setattr(self, k, v)
        self.region_selections = {}

    def get_cross_section_weight(self, process):
        if process is None:
            _logger.error("Process is None")
            raise InvalidInputError("Process is NoneType")
        if self.lumi is None or "data" in process.lower() or self.lumi == -1:
            return 1.
        lumi_weight = self.xs_handle.get_lumi_scale_factor(process, self.lumi, self.event_numbers[process])
        _logger.debug("Retrieved %.2f as cross section weight for process %s and lumi %.2f" % (lumi_weight, process,
                                                                                               self.lumi))
        return lumi_weight

    def read_event_yields(self, systematic="Nominal"):
        if systematic not in self.cutflows:
            self.cutflows[systematic] = {}
        for region in self.selection.regions:
            self.region_selections[region] = region.get_cut_list()
            if region.name not in self.cutflows[systematic]:
                self.cutflows[systematic][region.name] = {}
            for file_handle in self.file_handles:
                process = file_handle.process
                tree = file_handle.get_object_by_name(self.tree_name, systematic)
                yields = []
                for i, cut in enumerate(region.get_cut_list()):
                    yields.append((cut, self.converter.convert_to_array(tree, "&&".join(region.get_cut_list()[:i+1]))['weight'].flatten().sum(),
                                   0, -1., -1.))
                self.cutflows[systematic][region.name][process] = np.array(yields,
                                                                      dtype=[("cut", "S300"), ("yield", "f4"),
                                                                              ("yield_unc", "f4"),
                                                                              ("eff", float),
                                                                              ("eff_total", float)])  # todo: array dtype for string not a good choiceyields

    def apply_cross_section_weight(self, systematic, region):
        for process in self.cutflows[systematic][region].keys():
            try:
                lumi_weight = self.get_cross_section_weight(process.split(".")[0])
            except InvalidInputError:
                _logger.error("None type parsed for ", process)
                continue
            self.cutflows[systematic][region][process]["yield"] *= lumi_weight

    def make_cutflow_table(self, systematic):
        cutflow_tables = dict()

        for region in self.cutflows[systematic].keys():
            for process, cutflow in self.cutflows[systematic][region].items():
                cutflow_tmp = self.stringify(cutflow)
                if region not in cutflow_tables.keys():
                    cutflow_tables[region] = cutflow_tmp
                    continue
                cutflow_tables[region] = rec_append_fields(cutflow_tables[region],
                                                              [i + process for i in cutflow_tmp.dtype.names[1:]],
                                                              [cutflow_tmp[n] for n in cutflow_tmp.dtype.names[1:]])
            headers = ["Cut"] + [x for elem in self.cutflows[systematic][region].keys() for x in (elem, "")]
            self.cutflow_tables = {k: tabulate(v.transpose(),
                                               headers=headers,
                                               tablefmt=self.format,
                                               floatfmt=".2f")
                                   for k, v in cutflow_tables.iteritems()}

    def calculate_sm_total(self):
        def add(yields):
            sm_yield = []

            for process, evn_yields in yields.iteritems():
                if "data" in process.lower():
                    continue
                if len(sm_yield) == 0:
                    sm_yield = list(evn_yields)
                    continue
                for icut, cut_item in enumerate(evn_yields):
                    sm_yield[icut] = tuple([sm_yield[icut][0]]+map(operator.add,
                                                                   list(sm_yield[icut])[1:],
                                                                   list(evn_yields[icut])[1:]))
            return np.array(sm_yield, dtype=self.dtype)

        for systematics, regions_data in self.cutflows.iteritems():
            for region, yields in regions_data.iteritems():
                self.cutflows[systematics][region]["SMTotal"] = add(yields)

    def merge_yields(self):
        def merge(yields):
            """
            Merge event yields for subprocesses
            :param yields: pair of process and yields
            :type yields: dict
            :return: merged yields
            :rtype: dict
            """
            for process, process_config in self.process_configs.iteritems():
                if not hasattr(process_config, "subprocesses"):
                    continue
                for sub_process in process_config.subprocesses:
                    if sub_process not in yields.keys():
                        continue
                    if not process in yields.keys():
                        yields[process] = yields[sub_process]
                    else:
                        map(add, yields[process], yields[sub_process])
                    yields.pop(sub_process)
            return yields

        for systematics, regions_data in self.cutflows.iteritems():
            for region, yields in regions_data.iteritems():
                self.cutflows[systematics][region] = merge(yields)

    def plot_signal_yields(self):
        signal_processes = filter(lambda prc: prc.type.lower() == "signal", self.process_configs.values())
        signal_generated_events = dict(filter(lambda cf: cf[0] in map(lambda prc: prc.name, signal_processes),
                                         self.event_numbers.iteritems()))
        for region, cutflows in self.cutflows["Nominal"].iteritems():
            signal_yields = dict(filter(lambda cf: cf[0] in map(lambda prc: prc.name, signal_processes),
                                        cutflows.iteritems()))
            canvas_cuts, canvas_final = get_signal_acceptance(signal_yields, signal_generated_events, None)
            canvas_cuts.SetName(canvas_cuts.GetName().replace("/","").replace(" ","").replace(">","_gt_").replace("<","_lt_").replace(".","") + "_{:s}".format(region))
            canvas_final.SetName(canvas_final.GetName().replace("/","").replace(" ","").replace(">","_gt_").replace("<","_lt_").replace(".","") + "_{:s}".format(region))
            self.output_handle.register_object(canvas_cuts)
            self.output_handle.register_object(canvas_final)

    def execute(self):
        self.read_event_yields()
        self.plot_signal_yields()

        for systematic in self.cutflows.keys():
            for region in self.cutflows[systematic].keys():
                self.apply_cross_section_weight(systematic, region)
        self.merge_yields()
        self.calculate_sm_total()

        self.make_cutflow_tables()
        self.output_handle.write_and_close()


class CutflowAnalyser(object):
    """
    Cutflow analyser
    """
    def __init__(self, **kwargs):
        kwargs.setdefault("output_file_name", None)
        kwargs.setdefault("lumi", None)
        kwargs.setdefault("process_config", None)
        kwargs.setdefault("no_merge", False)
        kwargs.setdefault("raw", False)
        kwargs.setdefault("output_dir", None)
        kwargs.setdefault("format", "plain")
        self.file_list = kwargs["file_list"]
        self.cutflow_hists = dict()
        self.cutflow_hists = dict()
        self.cutflow_tables = dict()
        self.dataset_config_file = kwargs["dataset_config"]
        self.lumi = kwargs["lumi"]
        self.output_file_name = kwargs["output_file_name"]
        self.systematics = kwargs["systematics"]
        self.cutflow_hists = dict()
        self.cutflows = dict()
        self.xs_handle = XSHandle(kwargs["dataset_config"])
        self.event_numbers = dict()
        self.process_configs = None
        self.raw = kwargs["raw"]
        self.format = kwargs["format"]
        self.merge = True if not kwargs["no_merge"] else False
        if kwargs["output_dir"] is not None:
            self.output_handle = OutputFileHandle(output_dir=kwargs["output_dir"])
        if kwargs["process_config"] is not None:
            self.process_configs = parse_and_build_process_config(kwargs["process_config"])
        self.file_handles = [FH(file_name=fn, dataset_info=self.dataset_config_file)for fn in self.file_list]
        if self.process_configs is not None:
            self.file_handles = pl.filter_process_configs(self.file_handles, self.process_configs)

    def apply_cross_section_weight(self):
        for process in self.cutflow_hists.keys():
            try:
                lumi_weight = self.get_cross_section_weight(process.split(".")[0])
            except InvalidInputError:
                _logger.error("None type parsed for ", self.cutflow_hists[process])
                continue
            for systematic in self.cutflow_hists[process].keys():
                for cutflow in self.cutflow_hists[process][systematic].values():
                    scale(cutflow, lumi_weight)

    def analyse_cutflow(self):
        self.apply_cross_section_weight()
        if self.process_configs is not None and self.merge:
            for process_name in self.cutflow_hists.keys():
                _ = find_process_config(process_name, self.process_configs)
            self.merge_histograms(self.cutflow_hists)
        self.calculate_sm_total()
        self.cutflow_hists = dict(filter(lambda kv: len(kv[1]) > 0, self.cutflow_hists.iteritems()))
        for systematic in self.systematics:
            self.cutflows[systematic] = dict()
            for process in self.cutflow_hists.keys():
                self.cutflows[systematic][process] = dict()
                for k, v in self.cutflow_hists[process][systematic].iteritems():
                    if k.endswith("_raw"):
                        continue
                    raw_cutflow = self.cutflow_hists[process][systematic][k+"_raw"]
                    self.cutflows[systematic][process][k] = self._analyse_cutflow(v, raw_cutflow, process)
        self.calculate_cut_efficiencies()

    def merge_histograms(self, hist_dict):

        def merge(histograms):
            for process, process_config in self.process_configs.iteritems():
                if not hasattr(process_config, "subprocesses"):
                    continue
                for sub_process in process_config.subprocesses:
                    if sub_process not in histograms.keys():
                        continue
                    for systematic in histograms[sub_process].keys():
                        for selection in histograms[sub_process][systematic].keys():
                            if not process in histograms.keys():
                                histograms[process] = dict((syst,
                                                            dict((sel, None) for sel in histograms[sub_process][syst].keys()))
                                                           for syst in histograms[sub_process].keys())
                            if selection not in histograms[process][systematic]:
                                print "could not find selection ", selection, " for process ", process
                                continue
                            if histograms[process][systematic][selection] is None:
                                new_hist_name = histograms[sub_process][systematic][selection].GetName().replace(sub_process, process)
                                histograms[process][systematic][selection] = histograms[sub_process][systematic][selection].Clone(new_hist_name)
                            else:
                                histograms[process][systematic][selection].Add(histograms[sub_process][systematic][selection].Clone(new_hist_name))
                    histograms.pop(sub_process)
        merge(hist_dict)

    def calculate_sm_total(self):
        sm_total_cutflows = {}
        for process, systematics in self.cutflow_hists.iteritems():
            if "data" in process.lower():
                continue
            for systematic, regions in systematics.iteritems():
                if systematic not in sm_total_cutflows.keys():
                    sm_total_cutflows[systematic] = {}
                for region, cutflow_hist in regions.iteritems():
                    if region not in sm_total_cutflows[systematic].keys():
                        sm_total_cutflows[systematic][region] = cutflow_hist.Clone()
                        continue
                    sm_total_cutflows[systematic][region].Add(cutflow_hist)
        self.cutflow_hists["SMTotal"] = sm_total_cutflows

    def get_cross_section_weight(self, process):
        if process is None:
            _logger.error("Process is None")
            raise InvalidInputError("Process is NoneType")
        if self.lumi is None or "data" in process.lower() or self.lumi == -1:
            return 1.
        lumi_weight = self.xs_handle.get_lumi_scale_factor(process, self.lumi, self.event_numbers[process])
        _logger.debug("Retrieved %.2f as cross section weight for process %s and lumi %.2f" % (lumi_weight, process,
                                                                                               self.lumi))
        return lumi_weight

    def _analyse_cutflow(self, cutflow_hist, raw_cutflow_hist, process=None):
        if not self.raw:
            parsed_info = np.array([(cutflow_hist.GetXaxis().GetBinLabel(b),
                                     cutflow_hist.GetBinContent(b),
                                     #raw_cutflow_hist.GetBinContent(b),
                                     cutflow_hist.GetBinError(b),
                                     # #raw_cutflow_hist.GetBinError(b),
                                     -1.,
                                     -1.) for b in range(1, cutflow_hist.GetNbinsX() + 1)],
                                   dtype=[("cut", "S100"), ("yield", "f4"), #("yield_raw", "f4"),
                                          ("yield_unc", "f4"),
                                          ("eff", float),
                                          ("eff_total", float)])  # todo: array dtype for string not a good choice
        else:
            parsed_info = np.array([(cutflow_hist.GetXaxis().GetBinLabel(b),
                                     raw_cutflow_hist.GetBinContent(b),
                                     raw_cutflow_hist.GetBinError(b),
                                     -1.,
                                     -1.) for b in range(1, cutflow_hist.GetNbinsX() + 1)],
                                   dtype=[("cut", "S100"), ("yield_raw", "f4"),
                                          ("yield_unc_raw", "f4"),
                                          ("eff", float),
                                          ("eff_total", float)])  # todo: array dtype for string not a good choice
        return parsed_info

    def calculate_cut_efficiencies(self):
        for systematic in self.systematics:
            for process in self.cutflows[systematic].keys():
                for cutflow in self.cutflows[systematic][process].values():
                    self.calculate_cut_efficiency(cutflow)

    def calculate_cut_efficiency(self, cutflow):
        yield_str = "yield"
        if self.raw:
            yield_str = "yield_raw"
        for i in range(len(cutflow["cut"])):
            if i == 0:
                cutflow[i]["eff"] = 100.
                cutflow[i]["eff_total"] = 100.
                continue
            if cutflow[i-1][yield_str] != 0.:
                cutflow[i]["eff"] = round(100.0*cutflow[i][yield_str]/cutflow[i-1][yield_str], 3)
            else:
                cutflow[i]["eff"] = -1.
            if cutflow[0][yield_str] != 0.:
                cutflow[i]["eff_total"] = round(100.0*cutflow[i][yield_str]/cutflow[0][yield_str], 3)
            else:
                cutflow[i]["eff_total"] = -1.

    def make_cutflow_tables(self):
        for systematic in self.systematics:
            self.make_cutflow_table(systematic)

    def make_cutflow_table(self, systematic):
        cutflow_tables = dict()
        for process in self.cutflows[systematic].keys():
            for selection, cutflow in self.cutflows[systematic][process].items():
                cutflow_tmp = self.stringify(cutflow)
                if selection not in cutflow_tables.keys():
                    cutflow_tables[selection] = cutflow_tmp
                    continue
                cutflow_tables[selection] = rec_append_fields(cutflow_tables[selection],
                                                              [i+process for i in cutflow_tmp.dtype.names[1:]],
                                                              [cutflow_tmp[n] for n in cutflow_tmp.dtype.names[1:]])
            headers = ["Cut"] + [x for elem in self.cutflows[systematic].keys() for x in (elem, "")]
            self.cutflow_tables = {k: tabulate(v.transpose(),
                                               headers=headers,
                                               tablefmt=self.format,
                                               floatfmt=".2f")
                                   for k, v in cutflow_tables.iteritems()}

    def stringify(self, cutflow):
        def format_yield(value, uncertainty):
            if value > 10000.:
                return "{:.3e}".format(value)
            else:
                return "{:.2f}".format(value)
            # if value > 10000.:
            #     return "{:.3e} +- {:.3e}".format(value, uncertainty)
            # else:
            #     return "{:.2f} +- {:.2f}".format(value, uncertainty)

        if not self.raw:
            cutflow = np.array([(cutflow[i]["cut"],
                                 format_yield(cutflow[i]["yield"], cutflow[i]["yield_unc"]),
                                 #cutflow[i]["eff"],
                                 cutflow[i]["eff_total"]) for i in range(len(cutflow))],
                               dtype=[("cut", "S100"), ("yield", "S100"), ("eff_total", float)]) #("eff", float),
        else:
            cutflow = np.array([(cutflow[i]["cut"],
                                 format_yield(cutflow[i]["yield_raw"], cutflow[i]["yield_unc_raw"]),
                                 #cutflow[i]["eff"],
                                 cutflow[i]["eff_total"]) for i in range(len(cutflow))],
                               dtype=[("cut", "S100"), ("yield_raw", "S100"), ("eff_total", float)]) #("eff", float),

        return cutflow

    def print_cutflow_table(self):
        available_cutflows = self.cutflow_tables.keys()
        print "######## Selection menu  ########"
        print "Available cutflows for printing: "
        print "--------------------------------"
        for i, region in enumerate(available_cutflows):
            print i, ")", region
        print "a) all"
        user_input = raw_input("Please enter your selection (space or comma separated). Hit enter to select default (BaseSelection) ")
        if user_input == "":
            selections = ["BaseSelection"]
        elif user_input.lower() == "a":
            selections = available_cutflows
        elif "," in user_input:
            selections = [available_cutflows[i] for i in map(int, user_input.split(","))]
        elif "," not in user_input:
            selections = [available_cutflows[i] for i in map(int, user_input.split())]
        else:
            print "{:s}Invalid input {:s}. Going for default.\033[0m".format("\033[91m", user_input)
            selections = ["BaseSelection"]
        for selection, cutflow in self.cutflow_tables.iteritems():
            if selection not in selections:
                continue
            print
            print "Cutflow for region %s" % selection
            print cutflow

    def store_cutflow_table(self):
        pass

    def load_cutflows(self, file_handle):
        process = file_handle.process
        if process is None:
            _logger.error("Parsed NoneType process from {:s}".format(file_handle.file_name))
            return
        if process not in self.event_numbers:
            self.event_numbers[process] = file_handle.get_number_of_total_events()
        else:
            self.event_numbers[process] += file_handle.get_number_of_total_events()
        if process not in self.cutflow_hists.keys():
            self.cutflow_hists[process] = dict()
        for systematic in self.systematics:
            cutflow_hists = file_handle.get_objects_by_pattern("^(cutflow_)", systematic)
            if systematic not in self.cutflow_hists[process]:
                self.cutflow_hists[process][systematic] = dict()
            for cutflow_hist in cutflow_hists:
                cutflow_hist.SetDirectory(0)
                try:
                    self.cutflow_hists[process][systematic][cutflow_hist.GetName().replace("cutflow_", "")].Add(cutflow_hist)
                except KeyError:
                    self.cutflow_hists[process][systematic][cutflow_hist.GetName().replace("cutflow_", "")] = cutflow_hist

    def read_cutflows(self):
        for file_handle in self.file_handles:
            self.load_cutflows(file_handle)

    def plot_cutflow(self):
        set_batch_mode(True)
        flipped = defaultdict(lambda: defaultdict(dict))
        for process, systematics in self.cutflow_hists.items():
            for systematic, cutflows in systematics.items():
                if "smtotal" in process.lower():
                    continue
                for region, cutflow_hist in cutflows.items():
                    flipped[systematic][region][process] = cutflow_hist
        plot_config = PlotConfig(name=None, dist=None, ytitle="Events", logy=True)
        for region in flipped['Nominal'].keys():
            plot_config.name = "{:s}_cutflow".format(region)
            cutflow_hists = {process: hist for process, hist in flipped["Nominal"][region].iteritems()
                             if "smtotal" not in process.lower()}
            for process, cutflow_hist in cutflow_hists.iteritems():
                cutflow_hist.SetName("{:s}_{:s}".format(cutflow_hist.GetName(), process))
            cutflow_canvas = Pt.plot_stack(cutflow_hists, plot_config, process_configs=self.process_configs)
            Ft.add_legend_to_canvas(cutflow_canvas, process_configs=self.process_configs)
            self.output_handle.register_object(cutflow_canvas)
        self.output_handle.write_and_close()

    def execute(self):
        self.read_cutflows()
        self.analyse_cutflow()
        self.make_cutflow_tables()
        if hasattr(self, "output_handle"):
            self.plot_cutflow()
