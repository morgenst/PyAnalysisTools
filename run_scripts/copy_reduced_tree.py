import sys
import ROOT
from PyAnalysisTools.base import default_init, get_default_argparser, add_input_args, add_selection_args
from PyAnalysisTools.AnalysisTools.RegionBuilder import RegionBuilder
from PyAnalysisTools.base.FileHandle import FileHandle
from PyAnalysisTools.base.YAMLHandle import YAMLLoader
from PyAnalysisTools.base.OutputHandle import OutputFileHandle


def select_and_copy(args):
    output_handle = OutputFileHandle(output_file=args.output_file_name)
    selection = RegionBuilder(**YAMLLoader.read_yaml(args.selection_config)).regions[0]
    cuts = selection.convert2cut_string()
    fh = FileHandle(file_name=args.input_file_list[0])
    tree = fh.get_object_by_name(args.tree_name, 'Nominal')
    cf_hists = fh.get_objects_by_pattern('cutflow', 'Nominal')
    new_tree = tree.CopyTree(cuts)
    new_cf_hists = []
    for cf in cf_hists:
        if 'DxAOD' in cf.GetName():
            continue
        h = ROOT.TH1F(cf.GetName() + '_new', cf.GetTitle(), cf.GetNbinsX()+1, 0., cf.GetNbinsX()+1)
        for i in range(cf.GetNbinsX()):
            h.GetXaxis().SetBinLabel(i+1, cf.GetXaxis().GetBinLabel(i))
            h.SetBinContent(i+1, cf.GetBinContent(i))
        h.GetXaxis().SetBinLabel(cf.GetNbinsX()+1, selection.name)
        h.SetBinContent(cf.GetNbinsX() + 2, new_tree.GetEntries())
        new_cf_hists.append(h)
    output_handle.register_object(new_tree, 'Nominal')
    for cf in cf_hists:
        output_handle.register_object(cf, 'Nominal')
    for cf in cf_hists:
        output_handle.register_object(cf, 'Nominal')
    for cf in new_cf_hists:
        output_handle.register_object(cf, 'Nominal')
    output_handle.write_and_close()


def main(_):
    parser = get_default_argparser('Copy tree with additional selection and update cutflows')
    add_input_args(parser)
    add_selection_args(parser)
    parser.add_argument('--output_file_name', '-of', default='tree_branches.txt', help='output file name')
    args = default_init(parser)
    select_and_copy(args)


if __name__ == '__main__':
    main(sys.argv[1:])
