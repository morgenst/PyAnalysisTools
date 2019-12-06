#!/usr/bin/env python
import sys

import root_numpy
from pandas import DataFrame
from tabulate import tabulate

from PyAnalysisTools.base import get_default_argparser, default_init
from PyAnalysisTools.base.FileHandle import FileHandle


def print_content(h):
    content, edges = root_numpy.hist2array(h, include_overflow=True, return_edges=True)
    df = DataFrame((edges[0], content))
    print('Dumping bin content for {:s}'.format(h.GetName()))
    print(tabulate(df.transpose(), headers=['bin number', 'bin edge', 'bin content']))
    print('\n\n')


def main(argv):
    parser = get_default_argparser('Tablise bin contents of histogram')
    parser.add_argument("input_file", help="input file list")
    parser.add_argument("--directory", '-d', default=None, help="input file list")
    args = default_init(parser)

    fh = FileHandle(file_name=args.input_file)
    hists = fh.get_objects_by_type('TH1', args.directory)
    for h in hists:
        print_content(h)


if __name__ == '__main__':
    main(sys.argv[1:])
