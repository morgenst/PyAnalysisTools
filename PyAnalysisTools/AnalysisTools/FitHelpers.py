import ROOT
from ROOT import RooFit


def convert(file_handles, tree_name, quantity, blind, selection=None):
    var = ROOT.RooRealVar(quantity[0], quantity[0], quantity[1], quantity[2])
    if blind:
        var.setRange("left", quantity[1], blind[0])
        var.setRange("right", blind[1], quantity[2])
    var_arg = ROOT.RooArgSet(var)
    data = ROOT.RooDataSet("data", "data", var_arg)
    for file_handle in file_handles:
        tree = file_handle.get_object_by_name(tree_name, "Nominal")
        for i in range(tree.GetEntries()):
            tree.GetEntry(i)
            try:
                for t in range(len(getattr(tree, quantity[0]))):
                    value = getattr(tree, quantity[0])[t]
                    if quantity[1]<value and value<quantity[2] :
                        ROOT.RooAbsRealLValue.__assign__(var, value)
                        data.add(var_arg, 1.0)
            except TypeError:
                value = getattr(tree, quantity[0])
                ROOT.RooAbsRealLValue.__assign__(var, value)
                data.add(var_arg, 1.0)
    if selection is not None:
        data = data.reduce(RooFit.Cut(selection))
    return data, var
