import itertools
import numpy as np
from tabulate import tabulate
from PyAnalysisTools.base.YAMLHandle import YAMLLoader
from PyAnalysisTools.AnalysisTools.XSHandle import XSHandle


class DatasetPrinter(object):
    def __init__(self, **kwargs):
        self.datasets = list(itertools.chain.from_iterable(filter(lambda v: "mc" in v[0],
                                                                  YAMLLoader.read_yaml(kwargs["dataset_list"]).values())))
        self.xs_handle = XSHandle(kwargs["xs_info_file"], read_dsid=True)
        self.data = {}

    def compile_info(self):
        for file_name in self.datasets:
            dsid = file_name.split(".")[1]
            self.data[dsid] = self.get_ds_info(int(dsid))

    def get_ds_info(self, dsid):
        info = list(self.xs_handle.retrieve_xs_info(dsid))
        info.insert(0, self.xs_handle.get_ds_info(dsid, "process_name"))
        #info.insert(1, self.xs_handle.get_ds_info(dsid, "generator"))
        return info

    def pprint(self):
        self.compile_info()
        data = np.array(self.data.values(), dtype=object)
        print tabulate.tabulate(data, headers=["Sample", "#sigma [pb-1]", "eff_filter", "k-factor"])

    @staticmethod
    def get_supported_formats():
        return ["plain"]