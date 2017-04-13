import numpy as np
from tabulate import tabulate
from PyAnalysisTools.base import _logger
from PyAnalysisTools.base.YAMLHandle import YAMLLoader
from PyAnalysisTools.ROOTUtils.FileHandle import FileHandle as FH
from PyAnalysisTools.PlottingUtils.HistTools import scale
from PyAnalysisTools.AnalysisTools.XSHandle import XSHandle
from PyAnalysisTools.PlottingUtils.PlotConfig import parse_and_build_process_config
from numpy.lib.recfunctions import rec_append_fields
import ROOT


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
        self.process_config = None
        self.raw = kwargs["raw"]
        self.merge = True if not kwargs["no_merge"] else False
        if kwargs["process_config"] is not None:
            self.process_config = parse_and_build_process_config(kwargs["process_config"])

    def apply_cross_section_weight(self):
        for process in self.cutflow_hists.keys():
            lumi_weight = self.get_cross_section_weight(process)
            for systematic in self.cutflow_hists[process].keys():
                for cutflow in self.cutflow_hists[process][systematic].values():
                    scale(cutflow, lumi_weight)

    def analyse_cutflow(self):
        self.apply_cross_section_weight()
        if self.process_config is not None and self.merge:
            self.merge_histograms(self.cutflow_hists)
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
            for process, process_config in self.process_config.iteritems():
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
                            if histograms[process][systematic][selection] is None:
                                new_hist_name = histograms[sub_process][systematic][selection].GetName().replace(sub_process, process)
                                histograms[process][systematic][selection] = histograms[sub_process][systematic][selection].Clone(new_hist_name)
                            else:
                                histograms[process][systematic][selection].Add(histograms[sub_process][systematic][selection].Clone(new_hist_name))
                    histograms.pop(sub_process)

        merge(hist_dict)

    def get_cross_section_weight(self, process):
        if self.lumi is None or "data" in process.lower():
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
                                 #raw_cutflow_hist.GetBinError(b),
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

        #("yield_raw_unc", float),
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
            self.cutflow_tables= {k: tabulate(v.transpose(), headers=self.cutflows[systematic].keys()) for k, v in cutflow_tables.iteritems()}

    def stringify(self, cutflow):
        def format_yield(value, uncertainty):
            if value > 10000.:
                return "{:.3e} +- {:.3e}".format(value, uncertainty)
            else:
                return "{:.2f} +- {:.2f}".format(value, uncertainty)

        if not self.raw:
            cutflow = np.array([(cutflow[i]["cut"],
                                 format_yield(cutflow[i]["yield"], cutflow[i]["yield_unc"]),
                                 cutflow[i]["eff"],
                                 cutflow[i]["eff_total"]) for i in range(len(cutflow))],
                               dtype=[("cut", "S100"), ("yield", "S100"), ("eff", float), ("eff_total", float)])
        else:
            cutflow = np.array([(cutflow[i]["cut"],
                                 format_yield(cutflow[i]["yield_raw"], cutflow[i]["yield_unc_raw"]),
                                 cutflow[i]["eff"],
                                 cutflow[i]["eff_total"]) for i in range(len(cutflow))],
                               dtype=[("cut", "S100"), ("yield_raw", "S100"), ("eff", float), ("eff_total", float)])

        return cutflow

    def print_cutflow_table(self):
        for selection, cutflow in self.cutflow_tables.iteritems():
            if not selection == "BaseSelection":
                continue
            print
            print "Cutflow for region %s" % selection
            print cutflow

    def store_cutflow_table(self):
        pass

    def load_cutflows(self, file_name):
        file_handle = FH(file_name=file_name, dataset_info=self.dataset_config_file)
        process = file_handle.process
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
        for file_name in self.file_list:
            self.load_cutflows(file_name)

    def execute(self):
        self.read_cutflows()
        self.analyse_cutflow()
        self.make_cutflow_tables()
