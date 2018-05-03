import ROOT
import math
from ROOT import RooFit
from PyAnalysisTools.PlottingUtils.Formatting import add_text_to_canvas, add_atlas_label

def convert(file_handles, tree_name, quantity, blind, selections, extra_selection):
    var = ROOT.RooRealVar(quantity[0], quantity[0], quantity[1], quantity[2])
    if blind:
       var.setRange("left", quantity[1], blind[0])
       var.setRange("right", blind[1], quantity[2])
    var_arg = ROOT.RooArgSet(var)
    data = ROOT.RooDataSet("data", "data", var_arg)
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

def get_Ds_count(model):
    gauss_coef = model.getVariables().find("coef2")
    n_Ds = gauss_coef.getVal()
    n_Ds_error = gauss_coef.getError()
    if math.isnan(n_Ds):
       n_Ds = 0
    if math.isnan(n_Ds_error):
       n_Ds_error = 0
    return n_Ds, n_Ds_error

def get_D_count(model):
    gauss_coef = model.getVariables().find("coef1")
    n_D = gauss_coef.getVal()
    n_D_error = gauss_coef.getError()
    if math.isnan(n_D):
       n_D = 0
    if math.isnan(n_D_error):
       n_D_error = 0
    return n_D, n_D_error

def get_Ds_width(model):
    gauss_sigma = model.getVariables().find("sigma2")
    w_Ds = gauss_sigma.getVal()
    w_Ds_error = gauss_sigma.getError()
    return w_Ds, w_Ds_error

def get_Ds_mass(model):
    gauss_mean = model.getVariables().find("mean2")
    m_Ds = gauss_mean.getVal()
    m_Ds_error = gauss_mean.getError()
    return m_Ds, m_Ds_error

def get_background_count(model):
    bkg_coef = model.getVariables().find("coef3")
    n_Bkg = bkg_coef.getVal()
    n_Bkg_error = bkg_coef.getError()
    return n_Bkg, n_Bkg_error

def add_chi2_to_canvas(canvas, frame):
    chi2 = frame.chiSquare()
    add_text_to_canvas(canvas, "#chi^{2}: " + "{:.2f}".format(chi2), pos={"x": 0.72, "y": 0.52})

def add_parameters_to_canvas(canvas, model, n_D, n_D_error, n_Ds, n_Ds_error, n_Bkg, n_Bkg_error):
    parameters = model.getVariables()
    add_text_to_canvas(canvas, "M_{D}: " + "{:.1f}".format(parameters.find("mean1").getVal())+"("+"{:.1f}".format(parameters.find("mean1").getError())+")", pos={"x": 0.72, "y": 0.87})
    add_text_to_canvas(canvas, "M_{Ds}: " + "{:.1f}".format(parameters.find("mean2").getVal())+"("+"{:.1f}".format(parameters.find("mean2").getError())+")", pos={"x": 0.72, "y": 0.82})
    add_text_to_canvas(canvas, "#sigma_{D}: " + "{:.1f}".format(parameters.find("sigma2").getVal())+"("+"{:.1f}".format(parameters.find("sigma2").getError())+")", pos={"x": 0.72, "y": 0.77})
    add_text_to_canvas(canvas, "#sigma_{Ds}: " + "{:.1f}".format(parameters.find("sigma2").getVal())+"("+"{:.1f}".format(parameters.find("sigma2").getError())+")", pos={"x": 0.72, "y": 0.72})
    add_text_to_canvas(canvas, "N_{D}: " + "{:.0f}".format(n_D)+"("+"{:.0f}".format(n_D_error)+")", pos={"x": 0.72, "y": 0.67})
    add_text_to_canvas(canvas, "N_{Ds}: " + "{:.0f}".format(n_Ds)+"("+"{:.0f}".format(n_Ds_error)+")", pos={"x": 0.72, "y": 0.62})
    add_text_to_canvas(canvas, "N_{Bkg}: " + "{:.0f}".format(n_Bkg)+"("+"{:.0f}".format(n_Bkg_error)+")", pos={"x": 0.72, "y": 0.57})

def format_and_draw_frame(canvas, frame, xtitle):
    frame.GetXaxis().SetTitle(xtitle)
    frame.Draw()
    add_atlas_label(canvas, "Internal", pos={'x': 0.2, 'y': 0.87})

def plot_all_components(model, frame):
    pdflist = model.pdfList()
    for i in range(0, pdflist.getSize()):
        model.plotOn(frame, RooFit.Components(pdflist.at(i).GetName()), RooFit.LineColor(ROOT.kRed))
    model.plotOn(frame)
