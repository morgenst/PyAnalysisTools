#!/usr/bin/env python

from __future__ import print_function
import sys
import re
import os
from glob import glob
from PyAnalysisTools.base import _logger, get_default_argparser, default_init
from PyAnalysisTools.base.YAMLHandle import YAMLLoader
from PyAnalysisTools.base.IOTools import parallel_merge
from PyAnalysisTools.AnalysisTools.Utilities import find_data_period


def match_files_to_dataset(directory_list, base_path, data_summary, filter_dsid):
    def add(key):
        if key in result:
            result[key] += glob(os.path.join(base_path, ds, "*.root*"))
        else:
            result[key] = glob(os.path.join(base_path, ds, "*.root*"))
    result = {}
    for ds in directory_list:
        if filter_dsid is not None:
            if not re.search(filter_dsid, ds):
                continue
        if re.search(r'\d{6,8}', ds):
            ds_number = re.search(r'\d{6,8}', ds).group()
        else:
            ds_number = ds
        is_data = len(ds_number) > 6
        if is_data:
            try:
                period, year = find_data_period(int(ds_number), data_summary)
            except ValueError:
                try:
                    year, period = ds_number.split("_")
                except ValueError:
                    try:
                        info = re.match(r".*period[A-Z].grp[0-9]{2}", ds).group().split(".")
                        year, period = "data" + info[-1][-2:], info[2]
                    except AttributeError:
                        if 'AllYear' in ds:
                            year = re.search(r"grp[0-9]{2}", ds).group().replace('grp', '')
                            period = 'AllYear'
            if period is None or year is None:
                _logger.error("Could not find data taking information for run {:d}".format(int(ds_number)))
                continue
            add("{:s}_13TeV_{:s}".format(year, period))
        elif "period" in ds:
            period = ds.split(".")[1]
            year = ds.split(".")[0]
            if year == "user":
                try:
                    year = int(re.findall(r"\d+", re.findall(r"grp1[5678]", ds)[0])[0])
                    year = "data{:d}".format(year)
                    period = re.findall(r"period[A-Z]", ds)[0]
                except IndexError:
                    _logger.error("Could not resolve year from ", ds)
                    continue
            add("{:s}_13TeV_{:s}".format(year, period))
        else:
            add(ds_number)
    return result


def load_input_direcories(input_path):
    dirs = []
    for path in os.listdir(input_path):
        tmp = [os.path.join(path, fname) for fname in os.listdir(os.path.join(input_path, path))
               if os.path.isdir(os.path.join(input_path, path, fname))]
        if len(tmp):
            dirs += tmp
        else:
            dirs.append(path)
    return dirs


def main(argv):
    parser = get_default_argparser(description="Grid output merging and renaming script")
    parser.add_argument('input_path', type=str, help="input path")
    parser.add_argument('--output_path', '-o', required=True, help="output path")
    parser.add_argument('--data_summary', '-ds', default=None, help="data taking summary config")
    parser.add_argument('--force', '-f', action='store_true', default=False, help="force creation")
    parser.add_argument('--merge_dir', '-md', default=None, help="directory where merging is performed")
    parser.add_argument('--filter', default=None, help="filter to merge only given dsid")
    parser.add_argument('--ncpu', '-n', default=1, help="Number of CPUs to be used")
    parser.add_argument('--tag', '-t', default=None, help="Additional name tag, e.g. MC campaign")
    args = default_init(parser)

    data_summary = None
    if args.data_summary is not None:
        data_summary = YAMLLoader.read_yaml(args.data_summary)

    input_directories = load_input_direcories(args.input_path)

    match = match_files_to_dataset(input_directories, args.input_path, data_summary, args.filter)
    if len(match) == 0:
        _logger.info("Could not find any match")
        return
    parallel_merge(match, args.output_path,
                   "ntuple-", merge_dir=args.merge_dir, force=args.force, postfix=args.tag)


if __name__ == '__main__':
    main(sys.argv[1:])
