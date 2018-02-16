import os
from PyAnalysisTools.base import InvalidInputError
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
        if not "dataset_list" in kwargs:
            raise InvalidInputError("No dataset list provided")
        self.datasets = YAMLLoader.read_yaml(kwargs["dataset_list"])
        self.input_path = kwargs["input_path"]

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
                                                 switch_off_process_name_analysis=True).get_number_of_total_events(True))
        ds.append(n_processed_events)
        client = pyAMI.client.Client('atlas')
        n_expected_events = int(client.execute("GetDatasetInfo  -logicalDatasetName=%s" % ds[0],
                                               format="dict_object").get_rows()[0]["totalEvents"])
        ds.append(n_expected_events)

    @staticmethod
    def print_summary(missing, incomplete):
        print "--------------- Missing datasets ---------------"
        for ds in missing:
            print ds[0]
        print "------------------------------------------------"
        print
        print
        print
        print "--------------- Incomplete datasets ---------------"
        for ds in incomplete:
            print ds[2], ds[-2], ds[-1]

    def run(self):
        self.transform_dataset_list()
        self.add_path()
        missing_datasets = filter(lambda ds: ds[2] is None, self.datasets)
        self.datasets = filter(lambda ds: ds not in missing_datasets, self.datasets)
        mp.ThreadPool(10).map(self.get_events, self.datasets)
        incomplete_datasets = filter(lambda ds: not ds[-2] ==ds[-1], self.datasets)
        self.print_summary(missing_datasets, incomplete_datasets)