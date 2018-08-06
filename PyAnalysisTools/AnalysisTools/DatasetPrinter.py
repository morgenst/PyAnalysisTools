import numpy as np
from tabulate import tabulate
from itertools import chain
from PyAnalysisTools.base.YAMLHandle import YAMLLoader
from PyAnalysisTools.AnalysisTools.XSHandle import XSHandle
import tabulate as tb
tb.LATEX_ESCAPE_RULES = {}


class DatasetPrinter(object):
    def __init__(self, **kwargs):
        self.datasets = list(chain.from_iterable(filter(lambda v: 'mc' in v[0],
                                                        YAMLLoader.read_yaml(kwargs['dataset_list']).values())))
        self.xs_handle = XSHandle(kwargs['xs_info_file'], read_dsid=True)
        self.data = {}
        self.format = kwargs['format']

    def compile_info(self):
        for file_name in self.datasets:
            dsid = file_name.split('.')[1]
            self.data[dsid] = self.get_ds_info(int(dsid))

    def get_ds_info(self, dsid):
        info = list(self.xs_handle.retrieve_xs_info(dsid))
        try:
            try:
                info.insert(0, self.xs_handle.get_ds_info(dsid, 'latex_label'))
            except AttributeError:
                info.insert(0, self.xs_handle.get_ds_info(dsid, 'process_name'))
        except KeyError:
            #_logger.warning("cout not find xsec for " + dsid)
            info.insert(0, -1.)
        info.insert(0, dsid)
        return info

    def pprint(self):
        self.compile_info()
        data = np.array(sorted(self.data.values()), dtype=object)
        print tabulate(data, headers=["DSID", "Sample", "#sigma [pb-1]", "eff_filter", "k-factor"],
                       tablefmt=self.format, floatfmt='.2f')

