#!/usr/bin/env python

from builtins import map

import os
import sys
from PyAnalysisTools.AnalysisTools.CutFlowAnalyser import CutflowAnalyser as ca
from PyAnalysisTools.AnalysisTools.CutFlowAnalyser import ExtendedCutFlowAnalyser as eca
from PyAnalysisTools import base
try:
    from tabulate.tabulate import tabulate_formats
except ImportError:
    from tabulate import tabulate_formats
import ROOT

ROOT.PyConfig.IgnoreCommandLineOptions = True


def main(argv):
    parser = base.get_default_argparser("Cutflow printer")
    base.add_input_args(parser)
    base.add_output_args(parser)
    base.add_process_args(parser)
    base.add_friend_args(parser)
    base.add_selection_args(parser)
    parser.add_argument('--config_file', '-cf', default=None, help='config file')
    parser.add_argument('-disable_interactive', '-di', action='store_true', default=False, help="Disable interactive"
                                                                                                "mode")
    parser.add_argument('--enable_signal_plots', '-esp', action='store_true', default=False, help='Enable plots for '
                                                                                                  'signal efficiency')
    parser.add_argument('--disable_sm_total', '-dsm', default=False, action='store_true',
                        help="disable summing sm total")
    parser.add_argument('--format', '-f', choices=list(map(str, tabulate_formats)),
                        help="format of printed table")
    parser.add_argument('--no_merge', '-n', action='store_true', default=False, help="switch off merging")
    parser.add_argument('--precision', '-p', type=int, default=3, help="precision of printed numbers")
    parser.add_argument('--raw', '-r', action='store_true', default=False, help="print raw cutflow")
    parser.add_argument('--systematics', '-s', nargs='+', default=['Nominal'], help="systematics")
    parser.add_argument('--module_config_files', '-mcf', nargs='+', default=None,
                        help='config of additional modules to apply')
    parser.add_argument('--enable_eff', '-ee', action='store_true', default=False, help='Enable cut efficiencies')
    parser.add_argument('--percent_eff', '-per', action='store_true', default=False,
                        help='Calculate cut efficiencies in percent')
    parser.add_argument('--save_table', '-st', action='store_true', default=False, help='store cutflow to file')
    parser.add_argument('--output_tag', default=None, help='additional tag for file names storing enabled')
    parser.add_argument("--disable_cutflow_reading", "-dcr", action='store_true', default=False,
                        help="disable reading of initial cutflows. Lumi weighting won't work apparently.")

    args = base.default_init(parser)
    args.file_list = [os.path.abspath(f) for f in args.input_file_list]
    if args.selection_config is None:
        cutflow_analyser = ca(**vars(args))
    else:
        cutflow_analyser = eca(**vars(args))
    cutflow_analyser.execute()
    cutflow_analyser.print_cutflow_table()


if __name__ == '__main__':
    main(sys.argv[1:])
