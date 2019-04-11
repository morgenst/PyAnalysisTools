#!/usr/bin/env python

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
import os
from subprocess import check_call
from PyAnalysisTools.base import _logger, InvalidInputError, default_init, get_default_argparser
from PyAnalysisTools.base.Utilities import recursive_glob
from PyAnalysisTools.base.ShellUtils import move, remove_file
from PyAnalysisTools.ROOTUtils.FileHandle import FileHandle


def resolve_overlap(output_path, output_file_name, file_list):
    file_handle = FileHandle(file_name=os.path.join(output_path, output_file_name))
    existing_channels = file_handle.get_objects_by_type("TH1", "PileupReweighting")
    existing_channels = map(lambda h: h.GetName(), existing_channels)
    remove_list = []
    for file_name in file_list:
        file_handle = FileHandle(file_name=file_name)
        new_channels = map(lambda h: h.GetName(), file_handle.get_objects_by_type("TH1", "PileupReweighting"))
        if set(existing_channels).issuperset(set(new_channels)):
            remove_list.append(file_name)
    for file_name in remove_list:
        file_list.remove(file_name)


def merge(args):
    if not hasattr(args, 'input_path'):
        _logger.error("No input path given but chosen merge option")
        raise InvalidInputError("Missing input path")
    file_list = recursive_glob(args.input_path, '*.root*')
    out_file_name = 'prw.root'
    if not args.output_path:
        args.output_path = './'
    if args.output_path.endswith('.root'):
        out_file_name = os.path.basename(args.output_path)
    output_dir = os.path.dirname(args.output_path)
    os.chdir(output_dir)
    if out_file_name in os.listdir(output_dir):
        resolve_overlap(output_dir, out_file_name, file_list)
        move(out_file_name, 'prw_tmp.root')
        file_list.append('prw_tmp.root')
    check_call(['hadd', out_file_name] + file_list)
    if os.path.exists('prw_tmp.root'):
        remove_file('prw_tmp.root')


if __name__ == '__main__':
    parser = get_default_argparser("Pileup reweigthing input generator")
    parser.add_argument("--input_path", "-i", default=None, help="input path for merging")
    parser.add_argument("--output_path", "-o", default=None, help="output path for merging")

    args = default_init(parser)
    merge(args)

