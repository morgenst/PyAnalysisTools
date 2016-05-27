__author__ = 'marcusmorgenstern'
__mail__ = ''

import argparse
import sys

from PyAnalysisTools.ROOTFileComparator import FileComparator


def main(argv):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('lhs', help = 'lhs file')
    parser.add_argument('rhs', help = 'rhs')
    parser.add_argument('--outdir', '-o', type = str, default = 'out', help = 'output directory')

    args = parser.parse_args()

    comparator = FileComparator(args)

if __name__ == '__main__':
    main(sys.argv[1:])