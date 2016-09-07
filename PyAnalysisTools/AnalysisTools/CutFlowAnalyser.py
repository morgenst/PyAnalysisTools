import numpy as np
from tabulate import tabulate
from PyAnalysisTools.ROOTUtils.FileHandle import FileHandle as FH


class CutflowAnalyser(object):
    """
    Cutflow analyser
    """
    def __init__(self, file_list, systematics=None, output_file_name=None):
        self.file_list = file_list
        self.cutflow_hists = dict()
        self.cutflow_hists = dict()
        self.cutflow_tables = dict()
        self.output_file_name = output_file_name
        self.systematics = systematics
        self.cutflow_hists = dict()
        self.cutflows = dict()

    def analyse_cutflow(self):
        for systematic in self.systematics:
            self.cutflows[systematic] = dict()
            for k, v in self.cutflow_hists[systematic].items():
                if k.endswith("_raw"):
                    continue
                raw_cutflow = self.cutflow_hists[systematic][k+"_raw"]
                self.cutflows[systematic][k] = CutflowAnalyser._analyse_cutflow(v, raw_cutflow)
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
                               dtype=[("cut", "S100"), ("yield", "f4"), ("yield_raw", "f4"), ("yield_unc", "f4"), ("yield_raw_unc", "f4"), ("eff", "f3"), ("eff_total", "f3")]) # todo: array dtype for string not a good choice
        return parsed_info

    def calculate_cut_efficiencies(self):
        for systematic in self.systematics:
            for cutflow in self.cutflows[systematic].values():
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
        for selection, cutflow in self.cutflows[systematic].items():
            cutflow = self.stringify(cutflow)
            self.cutflow_tables[selection] = tabulate(cutflow.transpose())

    @staticmethod
    def stringify(cutflow):
        cutflow = np.array([(cutflow[i]["cut"],
                             "%f +- %f" % (cutflow[i]["yield"], cutflow[i]["yield_unc"]),
                             "%f +- %f" % (cutflow[i]["yield_raw"], cutflow[i]["yield_raw_unc"]),
                             cutflow[i]["eff"],
                             cutflow[i]["eff_total"]) for i in range(len(cutflow))],
                           dtype=[("cut", "S100"), ("yield", "S100"), ("yield_raw", "S100"), ("eff", "f3"), ("eff_total", "f3")])
        return cutflow

    def print_cutflow_table(self):
        for selection, cutflow in self.cutflow_tables.iteritems():
            print
            print "Cutflow for region %s" % selection
            print cutflow

    def store_cutflow_table(self):
        pass

    def load_cutflows(self, file_name):
        file_handle = FH(file_name)
        for systematic in self.systematics:
            self.cutflow_hists[systematic] = dict()
            cutflow_hists = file_handle.get_objects_by_pattern("^(cutflow_)", systematic)
            for cutflow_hist in cutflow_hists:
                self.cutflow_hists[systematic][cutflow_hist.GetName().replace("cutflow_", "")] = cutflow_hist

    def read_cutflows(self):
        for file_name in self.file_list:
            self.load_cutflows(file_name)

    def execute(self):
        self.read_cutflows()
        self.analyse_cutflow()
        self.make_cutflow_tables()
