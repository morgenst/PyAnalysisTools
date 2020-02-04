#!/usr/bin/env python

import sys
from PyAnalysisTools.base import default_init, get_default_argparser
from PyAnalysisTools.AnalysisTools.NTupleAnalyser import NTupleAnalyser


def main(argv):
    parser = get_default_argparser("Cross check ntuple production against ami")
    parser.add_argument("input_path", nargs='+', help="input paths containing ntuples")
    parser.add_argument("--dataset_list", "-ds", required=True, help="list of dataset processed")
    parser.add_argument("--resubmit", "-r", action="store_true", default=False, help="add samples for resubmission to "
                                                                                     "dataset_list")
    parser.add_argument('--filter', '-f', nargs='+', default=None, help='ignore datasets matching filter requirement')
    parser.add_argument('--merge_mode', '-mm', default=None, help='switch between dataset merge mode')
    args = default_init(parser)
    analyser = NTupleAnalyser(**vars(args))
    analyser.run()


if __name__ == '__main__':
    main(sys.argv[1:])
