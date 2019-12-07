#!/usr/bin/env python
from __future__ import print_function

import sys
import os
from PyAnalysisTools.PlottingUtils.Plotter import Plotter as BP
from PyAnalysisTools import base
from PyAnalysisTools.base.ShellUtils import resolve_path_from_symbolic_links
import ROOT

ROOT.PyConfig.IgnoreCommandLineOptions = True


def main(argv):
    parser = base.get_default_argparser('Run script for BasePlotter')
    base.add_input_args(parser)
    base.add_process_args(parser)
    base.add_output_args(parser)
    parser.add_argument("--output_file", "-of", default="plots.root", help="output file name")
    parser.add_argument("--tree_dir_name", "-tdn", default="Nominal", help="directory name in root file containing "
                                                                           "tree")
    parser.add_argument("--plot_config_files", "-pcf", nargs="+", required=True, help="plot configuration file")
    parser.add_argument("--ncpu", "-n", type=int, default=1, help="number of parallel jobs")
    parser.add_argument("--nfile_handles", "-nf", type=int, default=1, help="number of parallel file handles")
    parser.add_argument("--enable_systematics", "-s", action="store_true", help="run systematics")
    parser.add_argument("--systematics_config", "-sc", default=None, help="systematics config")
    parser.add_argument("--syst_tree_name", "-stn", default=None, help="tree name for systematics")
    parser.add_argument("--read_hist", "-rh", action="store_true", default=False, help="read histograms")
    parser.add_argument("--module_config_files", "-mcf", nargs='+', default=[None], help="module config files")
    parser.add_argument("--disable_cutflow_reading", "-dcr", action='store_true', default=False,
                        help="disable reading of initial cutflows. Lumi weighting won't work apparently.")

    args = base.default_init(parser)

    cwd = os.getenv("PWD")
    args.output_dir = resolve_path_from_symbolic_links(cwd, args.output_dir)
    arguments = vars(args)
    plotter = BP(**arguments)
    plotter.make_plots()


if __name__ == '__main__':
    main(sys.argv[1:])
