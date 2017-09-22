import ROOT


def convert(file_handles, tree_name, quantity):
    var = ROOT.RooRealVar(quantity, quantity, 1500, 2250)
    var_arg = ROOT.RooArgSet(var)
    data = ROOT.RooDataSet("data", "data", var_arg)
    for file_handle in file_handles:
        tree = file_handle.get_object_by_name(tree_name, "Nominal")
        for i in range(tree.GetEntries()):
            tree.GetEntry(i)
            for t in range(len(getattr(tree, quantity))):
                ROOT.RooAbsRealLValue.__assign__(var, getattr(tree, quantity)[t])
                data.add(var_arg, 1.0)
    return data, var
