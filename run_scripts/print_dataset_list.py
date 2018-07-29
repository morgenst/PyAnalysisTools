import argparse
import sys
from tabulate.tabulate import tabulate_formats
from PyAnalysisTools.AnalysisTools.DatasetPrinter import DatasetPrinter


def main(argv):
    parser = argparse.ArgumentParser(description="Steering script for dataset printer")
    parser.add_argument("dataset_list", type=str, help="dataset list input file")
    parser.add_argument("xs_info_file", type=str, help="cross section input file")
    parser.add_argument("--format", "-f", type=str, choices=map(str, tabulate_formats), default="plain",
                        help="output format {:s}".format(DatasetPrinter.get_supported_formats()))
    args = parser.parse_args()
    printer = DatasetPrinter(**vars(args))
    printer.pprint()


if __name__ == '__main__':
    main(sys.argv[1:])
