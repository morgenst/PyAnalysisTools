import sys

from PyAnalysisTools.base import default_init, get_default_argparser, add_input_args
from PyAnalysisTools.base.FileHandle import FileHandle


def dump_branch_list(args):
    f = FileHandle(file_name=args.input_file_list[0])
    t = f.get_object_by_name(args.tree_name, 'Nominal')
    with open(args.output_file_name, 'w') as of:
        for b in t.GetListOfBranches():
            print(b.GetName(), file=of)


def main(_):
    parser = get_default_argparser('dump branch names from tree to file')
    add_input_args(parser)
    parser.add_argument('--output_file_name', '-of', default='tree_branches.txt', help='output file name')
    args = default_init(parser)

    dump_branch_list(args)


if __name__ == '__main__':
    main(sys.argv[1:])
