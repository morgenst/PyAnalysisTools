from __future__ import print_function

import os
import re
import sys
from builtins import filter
from builtins import map
from builtins import object
from copy import copy
from subprocess import check_output, CalledProcessError

import pathos.multiprocessing as mp

from PyAnalysisTools.base import InvalidInputError, _logger
from PyAnalysisTools.base.FileHandle import FileHandle
from PyAnalysisTools.base.YAMLHandle import YAMLLoader, YAMLDumper

try:
    from tabulate.tabulate import tabulate
except ImportError:
    from tabulate import tabulate
try:
    ModuleNotFoundError
except NameError:
    ModuleNotFoundError = ImportError
try:
    import pyAMI.client
except ModuleNotFoundError:
    _logger.error("pyAMI not loaded")
    sys.exit(1)


class NTupleAnalyser(object):
    def __init__(self, **kwargs):
        """
        Constructor

        :param kwargs: arguments
        :type kwargs: dict
        dataset_list: yml file containing datasetlist of expected processed samples
        input_path: path containing ntuples to cross check against AMI
        """
        kwargs.setdefault('filter', None)
        kwargs.setdefault('merge_mode', None)
        self.check_valid_proxy()
        if "dataset_list" not in kwargs:
            raise InvalidInputError("No dataset list provided")
        self.dataset_list_file = kwargs["dataset_list"]
        self.datasets = YAMLLoader.read_yaml(self.dataset_list_file)
        self.datasets = dict([kv for kv in iter(list(self.datasets.items())) if "pilot" not in kv[0]])
        self.datasets = dict([kv for kv in iter(list(self.datasets.items())) if "resubmit" not in kv[0]])
        keys = [k for k in self.datasets.keys() if '$$' in k]
        if len(keys):
            self.grid_name_pattern = '$$' + keys[0].split('$$')[-1]
        else:
            self.grid_name_pattern = None
        self.input_path = kwargs["input_path"]
        self.resubmit = kwargs["resubmit"]
        self.filter = kwargs['filter']
        self.merge_mode = kwargs['merge_mode']
        if self.filter is not None:
            for pattern in self.filter:
                self.datasets = dict([kv for kv in iter(list(self.datasets.items())) if not re.match(pattern, kv[0])])

    @staticmethod
    def check_valid_proxy():
        """
        Checks if valid voms proxy is setup and exits if not

        :return: None
        :rtype: None
        """
        try:
            info = check_output(["voms-proxy-info"])
        except CalledProcessError:
            _logger.error("voms not setup. Please run voms-proxy-init -voms atlas. Giving up now...")
            exit(-1)
        time_left = list(map(int, list(filter(lambda e: e[0].startswith("timeleft"),
                                              [tag.split(":") for tag in info.split("\n")]))[0][1:]))
        if not all([i == 0 for i in time_left]):
            return
        _logger.error("No valid proxy found. Please run voms-proxy-init -voms atlas. Giving up now...")
        exit(-1)

    def transform_dataset_list(self):
        self.datasets = [ds for campaign in list(self.datasets.values()) for ds in campaign]
        if self.merge_mode is None:
            self.datasets = [[ds, ".".join(ds.split(".")[1:3])] for ds in self.datasets]
        else:
            self.datasets = [[ds, ".".join([ds.split(".")[1], ds.split(".")[5]])] for ds in self.datasets]
        # self.datasets = map(lambda ds: [ds, ".".join([ds.split(".")[1], ds.split(".")[2],
        #ds.split(".")[3], ds.split(".")[4], ds.split(".")[5]])], self.datasets)

    def add_path(self):
        """
        Add all dataset in all paths to list of processed datasets
        :return:
        :rtype:
        """
        processed_datasets = []
        for path in self.input_path:
            processed_datasets += os.listdir(path)
        for ds in self.datasets:
            matches = [pds for pds in processed_datasets if ds[1] in pds]
            if len(matches) > 0:
                ds.append(matches)
            else:
                ds.append(None)

    def get_events(self, ds):
        n_processed_events = 0
        processed_datasets = []
        for path in self.input_path:
            processed_datasets += os.listdir(path)
            for rf in os.listdir(os.path.join(path, ds[2][0])):
                n_processed_events += int(FileHandle(file_name=os.path.join(path, ds[2][0], rf),
                                                     switch_off_process_name_analysis=True).get_daod_events())

        ds.append(n_processed_events)
        client = pyAMI.client.Client('atlas')
        try:
            n_expected_events = int(client.execute("GetDatasetInfo  -logicalDatasetName=%s" % ds[0],
                                                   format="dict_object").get_rows()[0]["totalEvents"])
        except pyAMI.exception.Error:
            _logger.error("Could not find dataset: {:s}".format(ds[0]))
            return
        ds.append(n_expected_events)

    @staticmethod
    def print_summary(missing, incomplete, duplicated):
        """
        Print summary of missing and incomplete datasets
        Values: dataset, events processed, events expected, missing fraction, complete fraction

        :param missing: missing datasets
        :type missing: list
        :param incomplete: incomplete datasets including expected and processed events
        :type incomplete: list(tuples)
        :param duplicated: duplicated datasets
        :type incomplete: list(tuples)
        :return: None
        :rtype: None
        """
        def calc_fractions(dataset):
            data = []
            for ds in dataset:
                missing_fraction = float(ds[-2]) / float(ds[-1]) * 100.
                data.append((ds[2][0], ds[-2], ds[-1], missing_fraction, 100. - missing_fraction))
            return data
        print("--------------- Missing datasets ---------------")
        print(tabulate([[ds[0]] for ds in missing], tablefmt='rst'))
        print("--------------- Duplicated datasets ---------------")
        print(tabulate(calc_fractions(duplicated), tablefmt='rst', floatfmt='.2f',
                       headers=["Dataset", "Processed event", "Total avail. events", "available fraction [%]",
                                "missing fraction [%]"]))
        print("------------------------------------------------")
        print('\n\n\n')
        print("--------------- Incomplete datasets ---------------")
        print(tabulate(calc_fractions(incomplete), tablefmt='rst', floatfmt='.2f',
                       headers=["Dataset", "Processed event", "Total avail. events", "available fraction [%]",
                                "missing fraction [%]"]))

    def prepare_resubmit(self, incomplete, missing):
        """
        add failed or incomplete samples back to new dataset input file
        :param incomplete: list of incomplete datasets
        :type incomplete: list
        :param missing: list of missing aka failed datasets
        :type missing: list
        :return: None
        :rtype: None
        """
        resubmit_ds = {}
        for ds in incomplete:
            for grid_ds in ds[2]:
                try:
                    version = '_' + re.search(r'.v\d+.\d+', grid_ds).group().replace('.v', 'v')
                except AttributeError:
                    version = ''
                if self.grid_name_pattern is not None:
                    version += self.grid_name_pattern
                try:
                    resubmit_ds['incomplete{:s}'.format(version)].append(ds[0])
                except KeyError:
                    resubmit_ds['incomplete{:s}'.format(version)] = [ds[0]]
        missing_key = 'missing'
        if self.grid_name_pattern is not None:
            missing_key += self.grid_name_pattern
        resubmit_ds[missing_key] = [ds[0] for ds in missing]
        YAMLDumper.dump_yaml(resubmit_ds, self.dataset_list_file.replace('.yml', '_resubmit.yml'),
                             default_flow_style=False)

    def run(self):
        """
        Entry point executing ntuple completeness check

        :return: None
        :rtype: None
        """
        self.transform_dataset_list()
        self.add_path()
        missing_datasets = [ds for ds in self.datasets if ds[2] is None]
        self.datasets = [ds for ds in self.datasets if ds not in missing_datasets]
        duplicated_tmp = [ds for ds in self.datasets if len(ds[-1]) > 1]
        self.datasets = [ds for ds in self.datasets if ds not in duplicated_tmp]
        duplicated_datasets = []
        for i in duplicated_tmp:
            for j in i[-1]:
                duplicated_datasets.append(copy(i))
                duplicated_datasets[-1][-1] = [j]
        mp.ThreadPool(10).map(self.get_events, self.datasets)
        mp.ThreadPool(10).map(self.get_events, duplicated_datasets)
        incomplete_datasets = [ds for ds in self.datasets if not ds[-2] == ds[-1]]
        self.print_summary(missing_datasets, incomplete_datasets, duplicated_datasets)
        if self.resubmit:
            self.prepare_resubmit(incomplete_datasets, missing_datasets)
