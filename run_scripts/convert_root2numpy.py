#!/usr/bin/env LD_LIBRARY_PATH=$LD_LIBRARY_PATH python

import os
import sys
from functools import partial
from pathos.multiprocessing import Pool
import pandas as pd

from PyAnalysisTools.AnalysisTools.RegionBuilder import RegionBuilder

from PyAnalysisTools.base import get_default_argparser, default_init, _logger
from PyAnalysisTools.base.ShellUtils import make_dirs
from PyAnalysisTools.base.YAMLHandle import YAMLLoader as yl
from PyAnalysisTools.AnalysisTools.MLHelper import Root2NumpyConverter
from PyAnalysisTools.base.FileHandle import FileHandle
import feather


def convert_and_dump(file_handle, output_path, tree_name, region=None, branches=None, format='json'):
    converter = Root2NumpyConverter(branches)
    output_file_name = file_handle.file_name.split("/")[-1].replace(".root", "")
    if region is None:
        selection = ''
    else:
        region.build_cuts()
        selection = region.convert2cut_string()
        output_file_name += '_' + region.name
    data = converter.convert_to_array(file_handle.get_object_by_name(tree_name, "Nominal"), selection=selection)
    df_data = pd.DataFrame(data)

    if format == 'json':
        df_data.to_json(os.path.join(output_path, output_file_name) + '.json')
    elif format == 'feather':
        feather.write_dataframe(df_data, os.path.join(output_path, output_file_name) + '.feather')


def main(argv):
    parser = get_default_argparser(description="convert root files to numpy arrays")
    parser.add_argument("input_files", nargs="+", help="input files")
    parser.add_argument('--selection', '-sc', default=None, help="optional region file for selection")
    parser.add_argument("--tree_name", "-tn", required=True, help="input tree name")
    parser.add_argument("--output_path", "-o", required=True, help="output directory")
    parser.add_argument('--var_list', '-vl', default=None, help='config file with reduced variable list')
    parser.add_argument('--format', '-f', default='json', choices=['json', 'feather'], help='format of output file')
    args = default_init(parser)

    file_handles = [FileHandle(file_name=fn) for fn in args.input_files]
    args.output_path = os.path.abspath(args.output_path)
    make_dirs(args.output_path)
    regions = None
    branches = None
    if args.selection is not None:
        regions = RegionBuilder(**yl.read_yaml(args.selection)["RegionBuilder"])

    if args.var_list is not None:
        branches = yl.read_yaml(args.var_list)

    if regions is None:
        Pool().map(partial(convert_and_dump, output_path=args.output_path, tree_name=args.tree_name, branches=branches,
                           format=args.format),
                   file_handles)
    else:
        for region in regions.regions:
            Pool().map(partial(convert_and_dump, output_path=args.output_path, tree_name=args.tree_name,
                               region=region, branches=branches, format=args.format), file_handles)
    _logger.info('Wrote output file to {:s}'.format(args.output_path))


if __name__ == '__main__':
    main(sys.argv[1:])
