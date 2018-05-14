import os
from subprocess import check_output, CalledProcessError
from PyAnalysisTools.base import InvalidInputError, _logger
from PyAnalysisTools.base.YAMLHandle import YAMLLoader
from PyAnalysisTools.ROOTUtils.FileHandle import FileHandle
import pathos.multiprocessing as mp
try:
    import pyAMI.client
except Exception as e:
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
        self.check_valid_proxy()
        if not "dataset_list" in kwargs:
            raise InvalidInputError("No dataset list provided")
        self.datasets = YAMLLoader.read_yaml(kwargs["dataset_list"])
        self.datasets = dict(filter(lambda kv: "pilot" not in kv[0], self.datasets.iteritems()))
        self.datasets = dict(filter(lambda kv: "resubmit" not in kv[0], self.datasets.iteritems()))
        self.input_path = kwargs["input_path"]

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
        time_left = map(int, filter(lambda e: e[0].startswith("timeleft"),
                                    map(lambda tag: tag.split(":"), info.split("\n")))[0][1:])
        if not all(map(lambda i: i == 0, time_left)):
            return
        _logger.error("No valid proxy found. Please run voms-proxy-init -voms atlas. Giving up now...")
        exit(-1)

    def transform_dataset_list(self):
        self.datasets = [ds for campaign in self.datasets.values() for ds in campaign]
        self.datasets = map(lambda ds: [ds, ".".join([ds.split(".")[1], ds.split(".")[5]])], self.datasets)

    def add_path(self):
        processed_datasets = os.listdir(self.input_path)
        for ds in self.datasets:
            match = [ds[1] in pds for pds in processed_datasets]
            try:
                index = match.index(True)
                ds.append(processed_datasets[index])
            except ValueError:
                ds.append(None)

    def get_events(self, ds):
        n_processed_events = 0
        for rf in os.listdir(os.path.join(self.input_path, ds[2])):

            n_processed_events += int(FileHandle(file_name=os.path.join(self.input_path, ds[2], rf),
                                                 switch_off_process_name_analysis=True).get_daod_events())
        ds.append(n_processed_events)
        client = pyAMI.client.Client('atlas')
        try:
            n_expected_events = int(client.execute("GetDatasetInfo  -logicalDatasetName=%s" % ds[0],
                                                   format="dict_object").get_rows()[0]["totalEvents"])
        except pyAMI.exception.Error:
            _logger.error("Could not find dataset: ", ds[0])
            return
        ds.append(n_expected_events)

    @staticmethod
    def print_summary(missing, incomplete):
        """
        Print summary of missing and incomplete datasets
        Values: dataset, events processed, events expected, missing fraction, complete fraction

        :param missing: missing datasets
        :type missing: list
        :param incomplete: incomplete datasets including expected and processed events
        :type incomplete: list(tuples)
        :return: None
        :rtype: None
        """
        print "--------------- Missing datasets ---------------"
        for ds in missing:
            print ds[0]
        print "------------------------------------------------"
        print
        print
        print
        print "--------------- Incomplete datasets ---------------"
        for ds in incomplete:
            missing_fraction = float(ds[-2])/float(ds[-1]) * 100.
            print ds[2], ds[-2], ds[-1], missing_fraction, 100. - missing_fraction

    def run(self):
        """
        Entry point executing ntuple completeness check

        :return: None
        :rtype: None
        """
        self.transform_dataset_list()
        self.add_path()
        missing_datasets = filter(lambda ds: ds[2] is None, self.datasets)
        self.datasets = filter(lambda ds: ds not in missing_datasets, self.datasets)
        mp.ThreadPool(10).map(self.get_events, self.datasets)
        incomplete_datasets = filter(lambda ds: not ds[-2] ==ds[-1], self.datasets)
        self.print_summary(missing_datasets, incomplete_datasets)
