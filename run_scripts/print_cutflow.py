#!/usr/bin/env python

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
import os
import sys
from PyAnalysisTools.AnalysisTools.CutFlowAnalyser import CutflowAnalyser as ca
from PyAnalysisTools.AnalysisTools.CutFlowAnalyser import ExtendedCutFlowAnalyser as eca
from PyAnalysisTools.base import *
try:
    from tabulate.tabulate import tabulate_formats
except ImportError:
    from tabulate import tabulate_formats


def main(argv):
    parser = get_default_argparser("Cutflow printer")
    add_input_args(parser)
    add_output_args(parser)
    add_process_args(parser)
    add_friend_args(parser)
    add_selection_args(parser)
    parser.add_argument("--format", "-f", type=str, choices=map(str, tabulate_formats),
                        help="format of printed table")
    parser.add_argument('--config_file', '-cf', default=None, help='config file')
    parser.add_argument("--systematics", "-s", nargs="+", default=["Nominal"], help="systematics")
    parser.add_argument("--no_merge", "-n", action='store_true', default=False, help="switch off merging")
    parser.add_argument("--raw", "-r", action="store_true", default=False, help="print raw cutflow")
    parser.add_argument('--disable_sm_total', '-dsm', default=False, action='store_true',
                        help="disable summing sm total")
    parser.add_argument('--enable_eff', '-ee', action='store_true', default=False, help='Enable cut efficiencies')
    parser.add_argument('--percent_eff', '-p', action='store_true', default=False,
                        help='Calculate cut efficiencies in percent')

    parser.add_argument('--disable_signal_plots', '-dsp', action='store_true', default=False, help='Disable plots for '
                                                                                                   'signal efficiency')
    parser.add_argument('-disable_interactive', '-di', action='store_true', default=False, help="Disable interactive"
                                                                                                "mode")

    args = default_init(parser)
    args.file_list = [os.path.abspath(f) for f in args.input_file_list]
    if args.selection_config is None:
        cutflow_analyser = ca(**vars(args))
    else:
        cutflow_analyser = eca(**vars(args))
    cutflow_analyser.execute()
    cutflow_analyser.print_cutflow_table()


if __name__ == '__main__':
    main(sys.argv[1:])
