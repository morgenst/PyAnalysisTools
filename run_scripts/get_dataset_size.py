import argparse
import itertools
import sys
from PyAnalysisTools.base import _logger
from PyAnalysisTools.base.YAMLHandle import YAMLLoader

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
    return float(dataset_info[u'totalSize']) / 1024. / 1024. / 1024.


def main(argv):
    parser = argparse.ArgumentParser(description="PyAMI script to get file size of sample list")
    parser.add_argument("dataset_list", type=str, help="dataset list file")

    args = parser.parse_args()
    dataset_list = YAMLLoader.read_yaml(args.dataset_list)
    size = sum([get_size(dataset_name) for dataset_name in list(itertools.chain.from_iterable(dataset_list.values()))])
    print "Total size {:.2f} GB".format(size)


if __name__ == '__main__':
    main(sys.argv[1:])