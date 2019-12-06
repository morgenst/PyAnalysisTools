#!/usr/bin/env python
import sys

from builtins import map

try:
    from tabulate.tabulate import tabulate_formats
except ImportError:
    from tabulate import tabulate_formats
from PyAnalysisTools.AnalysisTools.DatasetPrinter import DatasetPrinter
from PyAnalysisTools.base import default_init, get_default_argparser


def main(argv):
    parser = get_default_argparser(description="Steering script for dataset printer")
    parser.add_argument("dataset_list", help="dataset list input file")
    parser.add_argument("xs_info_file", help="cross section input file")
    parser.add_argument("--format", "-f", choices=list(map(str, tabulate_formats)), default="plain",
                        help="output format")
    args = default_init(parser)
    printer = DatasetPrinter(**vars(args))
    printer.pprint()


if __name__ == '__main__':
    main(sys.argv[1:])
