__author__ = 'marcusmorgenstern'
__mail__ = ''

import numpy as np
from tabulate import tabulate

from PyAnalysisTools.ROOTUtils.FileHandle import FileHandle as FH


class CutflowAnalyser(object):
    """
    Cutflow analyser
    """
    def __init__(self, file_list, regions=None, output_file_name=None):
        self.file_list = file_list
        self.cutflow_hists = dict()
        self.cutflows = dict()
        self.cutflow_table = None
        self.output_file_name = output_file_name
        self.regions = regions

    def analyse_cutflow(self):
        for k,v in self.cutflow_hists.items():
            self.cutflows[k] = CutflowAnalyser._analyse_cutflow(v)

    @staticmethod
    def _analyse_cutflow(cutflow_hist):
        parsed_info = np.array([(cutflow_hist.GetXaxis().GetBinLabel(b),
                                 cutflow_hist.GetBinContent(b)) for b in range(cutflow_hist.GetNbinsX())],
                               dtype=[("cut", "S"), ("yield", "f4")])
        return parsed_info

    def make_cutflow_table(self):
        headers = ["Cut"] + self.cutflows.keys()
        cut_strings = self.cutflows.values()[0]["cut"]
        event_yields = [map(lambda y: "%.4f" % y, l["yield"]) for l in self.cutflows.values()]
        self.cutflow_table = tabulate(np.array([cut_strings] + event_yields).transpose())

    def print_cutflow_table(self):
        print self.cutflow_table

    def store_cutflow_table(self):
        pass

    def read_cutflow(self):
        for file_name in self.file_list:
            self._read_cutflow(file_name)

    def _read_cutflow(self, file_name, cutflow_hist_name="cutflow_raw"):
        file_handle = FH(file_name)
        self.cutflow_hists[file_name] = file_handle.get_object_by_name(cutflow_hist_name)

    def execute(self):
        self.read_cutflow()
        self.analyse_cutflow()
        self.make_cutflow_table()
