import argparse
import itertools
import sys
from PyAnalysisTools.base import _logger
from PyAnalysisTools.base.YAMLHandle import YAMLLoader
from PyAnalysisTools.ROOTUtils.FileHandle import FileHandle


try:
    import pyAMI.client
except Exception as e:
    _logger.error("pyAMI not loaded")
    sys.exit(1)


def get_size(dataset_name):
    client = pyAMI.client.Client('atlas')
    _logger.debug("Parsing info for %s" % dataset_name)
    dataset_info = client.execute("GetDatasetInfo  -logicalDatasetName=%s" % dataset_name,
                                  format="dict_object").get_rows()[0]
    return float(dataset_info[u'totalEvents'])


def get_process(dataset_name):
    if "data" in dataset_name:
        return dataset_name
    return dataset_name.split("_")[0]


def main(argv):
    parser = argparse.ArgumentParser(description="Script to compare no. of processed events to AMI")
    parser.add_argument("input_files", nargs="+", help="input files")
    parser.add_argument("--dataset_list", "-dl", type=str, required=True, help="file containing dataset list")
    parser.add_argument("--dataset_info_config", "-ds", type=str, required=True, help="dataset info config file")
    args = parser.parse_args()

    file_handles = [FileHandle(file_name=fn, switch_off_process_name_analysis=True) for fn in args.input_files]
    dataset_list = YAMLLoader.read_yaml(args.dataset_list)
    expected_yields = {get_process(dataset_name): get_size(dataset_name) for dataset_name in list(itertools.chain.from_iterable(dataset_list.values()))}
    yields = {get_process(fh.process): fh.get_number_of_total_events(True) for fh in file_handles}
    difference = {process: yields[process] - expected_yields[process]
    if process in expected_yields else yields[process] for process in yields.keys()}
    difference = dict(filter(lambda kv: kv[1] != 0., difference.iteritems()))
    for k, v in difference.iteritems():
        print "{:s}: {:.0f}".format(k, v)


if __name__ == '__main__':
    main(sys.argv[1:])

