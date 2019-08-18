#!/usr/bin/env python

from PyAnalysisTools.base import default_init, get_default_argparser
from PyAnalysisTools.AnalysisTools.NTupleAnalyser import NTupleAnalyser


if __name__ == '__main__':
    parser = get_default_argparser("Cross check ntuple production against ami")
    parser.add_argument("input_path", nargs='+', help="input paths containing ntuples")
    parser.add_argument("--dataset_list", "-ds", required=True, help="list of dataset processed")
    parser.add_argument("--resubmit", "-r", action="store_true", default=False, help="add samples for resubmission to "
                                                                                     "dataset_list")
    parser.add_argument('--filter', '-f', nargs='+', default=None, help='ignore datasets matching filter requirement')
    args = default_init(parser)
    analyser = NTupleAnalyser(**vars(args))
    analyser.run()
