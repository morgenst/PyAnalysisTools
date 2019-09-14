from __future__ import print_function

import argparse
import os
import sys
from subprocess import check_call

from PyAnalysisTools.base import _logger
from PyAnalysisTools.base.ShellUtils import make_dirs


def generate_file_list(**kwargs):
    def expand(input):
        if os.path.isfile(input):
            file_list.append(input)
        for root, subdir, file_names in os.walk(os.path.abspath(input)):
            for file_name in file_names:
                file_list.append(os.path.join(root, file_name))

    def clean_mount():
        return [file_name.replace('/afs/cern.ch/user/m/morgens/eos_atlas/', '') for file_name in file_list]

    output_file = None
    if not "inputs" in kwargs:
        _logger.error("No inputs provided, but generation of file list requested.")
        exit(1)
    if not "output_file" in kwargs:
        _logger.warning("No output file provided")
    else:
        output_file = open(kwargs["output_file"], "w")

    file_list = []
    for f in kwargs["inputs"]:
        expand(f)
    file_list = clean_mount()
    for file_name in file_list:
        print(file_name, file=output_file)


def copy(**kwargs):
    output_path = os.path.abspath(kwargs["output_path"])
    if not os.path.exists(output_path):
        make_dirs(output_path)

    recursive_level = kwargs["recursive"]
    with open(kwargs["filelist"]) as f:
        for file in f.readlines():
            sub_dir = "."
            if recursive_level > 0:
                sub_dir = file.split("/")[-(recursive_level+1):-1][0]
                make_dirs(os.path.join(output_path, sub_dir))
            cmd = "xrdcp root://eosatlas.cern.ch//eos/%s %s" %(file, os.path.join(output_path, sub_dir))
            check_call(cmd.split())


def main(argv):
    parser = argparse.ArgumentParser(description="script download eos data to stoomboot")
    parser.add_argument("--filelist", "-f", type=str, default=None, help="input file containing files")
    parser.add_argument("--output_path", "-o", type=str, default=None, help="output path")
    parser.add_argument("--recursive", "-R", type=int, default=0, help="depth for recursive copy")
    parser.add_argument("--generate_file_list", "-g", action="store_true", default=False, help="generate file list")
    parser.add_argument("--output_file", "-of", type=str, default=None, help="output file for file list")
    parser.add_argument("--inputs", "-i", nargs="+", default=None, help="input file/directory list")

    args = parser.parse_args()

    if not args.generate_file_list:
        copy(**vars(args))
    else:
        generate_file_list(**vars(args))


if __name__ == '__main__':
    main(sys.argv[1:])
