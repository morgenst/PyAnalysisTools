import ROOT
from ROOT import RooFit
from PyAnalysisTools.PlottingUtils.Formatting import add_text_to_canvas

def convert(file_handles, tree_name, quantity, blind, selections=None):
    var = ROOT.RooRealVar(quantity[0], quantity[0], quantity[1], quantity[2])
    if blind:
        var.setRange("left", quantity[1], blind[0])
        var.setRange("right", blind[1], quantity[2])     
    var_arg = ROOT.RooArgSet(var)
    data = ROOT.RooDataSet("data", "data", var_arg)
    cut = "1"
    if selections:
       for selection in selections:
          cut += "&&" + selection
    for file_handle in file_handles:
        file_temp=ROOT.TFile("tmp.root")
        tree_temp = file_handle.get_object_by_name(tree_name, "Nominal")
        tree=tree_temp.CopyTree(cut)
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
    return data, var

def get_list_of_slices(slicing_variable):
    variable_name = slicing_variable[0]
    n_bin = slicing_variable[1]
    variable_min = slicing_variable[2]
    variable_max = slicing_variable[3]
    interval = (variable_max - variable_min)/n_bin
    list_of_slices = []
    bin_center = 0
    for i in range(0, n_bin):
        bin_lower = "{:.3f}".format(variable_min + i*interval)
        bin_upper = "{:.3f}".format(variable_min + (i+1)*interval)
        bin_center = variable_min + (i+0.5)*interval
        list_of_slices.append([bin_lower + "<" + variable_name + "&&" + variable_name+ "<" + bin_upper, bin_center])
    return list_of_slices

def get_integral(model, var, min=-1, max=-1):
    var.setRange("integral",min ,max)
    var_set = ROOT.RooArgSet(var)
    integral = model.createIntegral(var_set, RooFit.NormSet(var_set), RooFit.Range("integral"))
    return integral.getVal()

def get_Ds_and_Bkg_count(model, min=-1, max=-1):
    x = model.getVariables().find("triplet_refitted_m")
    gauss = model.getComponents().find("g2")
    gauss_coef = model.getVariables().find("coef2")
    background = model.getComponents().find("exp")
    background_coef = model.getVariables().find("coef3")
    n_Ds =  gauss_coef.getVal()*get_integral(gauss, x, min, max)
    n_Bkg = background_coef.getVal()*get_integral(background, x, min, max)
    return n_Ds, n_Bkg

def add_fit_parameters_to_canvas(canvas, model):
    parameters = model.getVariables()
    gauss_mean = parameters.find("mean2").getVal()
    gauss_error = parameters.find("sigma2").getError()
    add_text_to_canvas(canvas, "M_{D_{s}}: " + "{:.2f}".format(gauss_mean), pos={"x": 0.72, "y": 0.82})

def convert_to_valid_name(string_in):
    string_in=string_in.replace("<","")
    string_in=string_in.replace(">","")
    string_in=string_in.replace("&","")
    print(string_in)
    return string_in
