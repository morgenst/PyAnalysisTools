import ROOT
from ROOT import RooFit
from PyAnalysisTools.PlottingUtils.Formatting import add_text_to_canvas, add_atlas_label

def convert(file_handles, tree_name, quantity, blind, selections, extra_selection, weight_name=None):
    var = ROOT.RooRealVar(quantity[0], quantity[0], quantity[1], quantity[2])
    weight = ROOT.RooRealVar("weight", "weight", 1)
    if blind:
       var.setRange("left", quantity[1], blind[0])
       var.setRange("right", blind[1], quantity[2])
    var_arg = ROOT.RooArgSet(var, weight)
    data = ROOT.RooDataSet("data", "data", var_arg, "weight")
    cut = "1"
    selections += extra_selection
    if selections:
       for selection in selections:
           cut += "&&" + selection
    for file_handle in file_handles:
        tree = file_handle.get_object_by_name(tree_name, "Nominal")
        entrylist = ROOT.TEntryList("entrylist", "entrylist")
        tree.Draw(">>entrylist", cut, "entrylist")
        for i in range(entrylist.GetN()):
            tree.GetEntry(entrylist.GetEntry(i))
            try:
                for t in range(len(getattr(tree, quantity[0]))):
                    value = getattr(tree, quantity[0])[t]
                    if quantity[1]<value and value<quantity[2] :
                        ROOT.RooAbsRealLValue.__assign__(var, value)
                        if weight_name:
                           data.add(var_arg, getattr(tree, weight_name))
                        else:
                           data.add(var_arg, 1.0)
            except TypeError:
                value = getattr(tree, quantity[0])
                ROOT.RooAbsRealLValue.__assign__(var, value)
                data.add(var_arg, 1.0)
    return data, var

def get_integral(model, var, min=-1, max=-1):
    var.setRange("integral",min ,max)
    var_set = ROOT.RooArgSet(var)
    integral = model.createIntegral(var_set, RooFit.NormSet(var_set), RooFit.Range("integral"))
    return integral

def get_parameter_value_and_error(model, parameter):
    model_variables = model.getVaraibles().find(parameter)
    return model_variables.getVal(), model_variables.getError()

def add_chi2_to_canvas(canvas, frame):
    chi2 = frame.chiSquare()
    add_text_to_canvas(canvas, "#chi^{2}: " + "%.2g"%(chi2), pos={"x": 0.66, "y": 0.57})

def rename(name):
    list = {'decayrate':'c_1', 'decayrate2':'c_2', 'mean':'m_{Ds}','nBkg':'N_{Bkg}', 'nDs':'N_{Ds}','sigma':'#sigma_{Ds}', 'nD':'N_{D}', 'alpha':'N_{D}/N_{Ds}', 'm_diff':"#Deltam"}
    return list.get(name, "undef")

def add_parameters_to_canvas(canvas, model):
    it = model.getVariables().createIterator()
    pos = {"x": 0.66, "y": 0.87}
    for parameter in iter(it.Next, None):
        if "_m" not in parameter.GetName() and "decay" not in parameter.GetName():
           add_text_to_canvas(canvas, rename(parameter.GetName())+ ": "+"%.1f"%(parameter.getVal())+"("+"%.1f"%(parameter.getError())+")", pos)
           pos["y"] -= 0.05

def format_and_draw_frame(canvas, frame, xtitle):
    frame.GetXaxis().SetTitle(xtitle)
    frame.Draw()
    add_atlas_label(canvas, "Internal", pos={'x': 0.2, 'y': 0.87})

def plot_all_components(model, frame):
    pdflist = model.pdfList()
    for i in range(0, pdflist.getSize()):
        model.plotOn(frame, RooFit.Components(pdflist.at(i).GetName()), RooFit.LineColor(ROOT.kRed))
    model.plotOn(frame)
