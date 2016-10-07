import numpy as np
from tabulate import tabulate
from PyAnalysisTools.ROOTUtils.FileHandle import FileHandle as FH
from numpy.lib.recfunctions import rec_append_fields


class CutflowAnalyser(object):
    """
    Cutflow analyser
    """
    def __init__(self, **kwargs):
        kwargs.setdefault("output_file_name", None)
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

    def analyse_cutflow(self):
        for systematic in self.systematics:
            self.cutflows[systematic] = dict()
            for process in self.cutflow_hists.keys():
                self.cutflows[systematic][process] = dict()
                for k, v in self.cutflow_hists[process][systematic].iteritems():
                    if k.endswith("_raw"):
                        continue
                    raw_cutflow = self.cutflow_hists[process][systematic][k+"_raw"]
                    self.cutflows[systematic][process][k] = CutflowAnalyser._analyse_cutflow(v, raw_cutflow)
        self.calculate_cut_efficiencies()

    @staticmethod
    def _analyse_cutflow(cutflow_hist, raw_cutflow_hist):
        parsed_info = np.array([(cutflow_hist.GetXaxis().GetBinLabel(b),
                                 cutflow_hist.GetBinContent(b),
                                 raw_cutflow_hist.GetBinContent(b),
                                 cutflow_hist.GetBinError(b),
                                 raw_cutflow_hist.GetBinError(b),
                                 -1.,
                                 -1.) for b in range(1, cutflow_hist.GetNbinsX() + 1)],
                               dtype=[("cut", "S100"), ("yield", "f4"), ("yield_raw", "f4"), ("yield_unc", "f4"),
                                      ("yield_raw_unc", float), ("eff", float), ("eff_total", float)]) # todo: array dtype for string not a good choice
        return parsed_info

    def calculate_cut_efficiencies(self):
        for systematic in self.systematics:
            for process in self.cutflows[systematic].keys():
                for cutflow in self.cutflows[systematic][process].values():
                    CutflowAnalyser.calculate_cut_efficiency(cutflow)

    @staticmethod
    def calculate_cut_efficiency(cutflow):
        for i in range(len(cutflow["cut"])):
            if i == 0:
                cutflow[i]["eff"] = 100.
                cutflow[i]["eff_total"] = 100.
                continue
            if cutflow[i-1]["yield"] != 0.:
                cutflow[i]["eff"] = round(100.0*cutflow[i]["yield"]/cutflow[i-1]["yield"], 3)
            else:
                cutflow[i]["eff"] = -1.
            if cutflow[0]["yield"] != 0.:
                cutflow[i]["eff_total"] = round(100.0*cutflow[i]["yield"]/cutflow[0]["yield"], 3)
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

    @staticmethod
    def stringify(cutflow):
        cutflow = np.array([(cutflow[i]["cut"],
                             "%.2f +- %.2f" % (cutflow[i]["yield"], cutflow[i]["yield_unc"]),
                             "%.2f +- %.2f" % (cutflow[i]["yield_raw"], cutflow[i]["yield_raw_unc"]),
                             cutflow[i]["eff"],
                             cutflow[i]["eff_total"]) for i in range(len(cutflow))],
                           dtype=[("cut", "S100"), ("yield", "S100"), ("yield_raw", "S100"), ("eff", float), ("eff_total", float)])
        return cutflow

    def print_cutflow_table(self):
        for selection, cutflow in self.cutflow_tables.iteritems():
            print
            print "Cutflow for region %s" % selection
            print cutflow

    def store_cutflow_table(self):
        pass

    def load_cutflows(self, file_name):
        file_handle = FH(file_name=file_name, dataset_info=self.dataset_config_file)
        process = file_handle.process
        if process not in self.cutflow_hists.keys():
            self.cutflow_hists[process] = dict()
        for systematic in self.systematics:
            self.cutflow_hists[process][systematic] = dict()
            cutflow_hists = file_handle.get_objects_by_pattern("^(cutflow_)", systematic)

            for cutflow_hist in cutflow_hists:
                self.cutflow_hists[process][systematic][cutflow_hist.GetName().replace("cutflow_", "")] = cutflow_hist

    def read_cutflows(self):
        for file_name in self.file_list:
            self.load_cutflows(file_name)

    def execute(self):
        self.read_cutflows()
        self.analyse_cutflow()
        self.make_cutflow_tables()
