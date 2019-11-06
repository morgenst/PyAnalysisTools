import ROOT
from ROOT import RooFit
from PyAnalysisTools.PlottingUtils.Formatting import add_text_to_canvas, add_atlas_label
from PyAnalysisTools.ROOTUtils.ObjectHandle import get_objects_from_canvas_by_name


def create_roodata(file_handles, tree_name, var, quantity, blind, selections, extra_selection, weight_name=None):
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
                    if quantity[1] < value and value < quantity[2]:
                        ROOT.RooAbsRealLValue.__assign__(var, value)
                        if weight_name:
                            data.add(var_arg, getattr(tree, weight_name))
                        else:
                            data.add(var_arg, 1.0)
            except TypeError:
                value = getattr(tree, quantity[0])
                ROOT.RooAbsRealLValue.__assign__(var, value)
                data.add(var_arg, 1.0)
    return data


def get_hist_from_canvas(file_name, canvas_name, hist_name):
    file = ROOT.TFile(file_name)
    canvas = file.Get(canvas_name)
    hist = get_objects_from_canvas_by_name(canvas, hist_name)[0]
    return hist


def get_integral(model, var, min=-1, max=-1):
    var.setRange("integral", min, max)
    var_set = ROOT.RooArgSet(var)
    integral = model.createIntegral(var_set, RooFit.NormSet(var_set), RooFit.Range("integral"))
    return integral


def get_parameter_value_and_error(model, parameter):
    model_variables = model.getVaraibles().find(parameter)
    return model_variables.getVal(), model_variables.getError()


def rename(name):
    list = {'decayrate': 'c_{1}', 'decayrate2': 'c_{2}', 'mean': 'm_{Ds}', 'nBkg': 'N_{Bkg}', 'nDs': 'N_{Ds}',
            'sigma': '#sigma_{Ds}', 'nD': 'N_{D}', 'alpha': 'N_{D}/N_{Ds}', 'm_diff': "#Deltam", 'n_bkg': 'N_{Bkg}',
            'n_sig': 'N_{Sig}', 'BFraction': 'f_{bb}'}
    return list.get(name, name)


def add_parameters_to_canvas(canvas, model, frame):
    it = model.getVariables().createIterator()
    pos = {"x": 0.66, "y": 0.87}
    exclude = ["triplet_slxy", "triplet_refitted_m", "decayrate", "decayrate2", "triplet_muon_mass"]
    for parameter in iter(it.Next, None):
        name = parameter.GetName()
        val = parameter.getVal()
        err = parameter.getError()
        if name not in exclude:
            if val > 2000.:
                val = "%.0f" % val
                err = "%.0f" % err
            elif val > 1.:
                val = "%.1f" % val
                err = "%.1f" % err
            else:
                val = "%.2f" % val
                err = "%.2f" % err
            add_text_to_canvas(canvas, rename(name) + ": " + val + "(" + err + ")", pos)
            pos["y"] -= 0.05
    add_text_to_canvas(canvas, "#chi^{2}: " + "%.2f" % (frame.chiSquare()), pos)


def format_and_draw_frame(canvas, frame, xtitle):
    frame.GetXaxis().SetTitle(xtitle)
    frame.Draw()
    add_atlas_label(canvas, "Internal", pos={'x': 0.2, 'y': 0.87})


def plot_all_components(model, frame):
    pdflist = model.pdfList()
    list = [3, 4, 6]
    for i in range(0, pdflist.getSize()):
        model.plotOn(frame, RooFit.Components(pdflist.at(i).GetName()), RooFit.LineColor(list[i]))
    model.plotOn(frame, RooFit.LineColor(2))


def scan_parameter_likelihood(data, model, parameter_name):
    canvas_scan = ROOT.TCanvas(parameter_name, "", 800, 600)
    nll = model.createNLL(data, RooFit.NumCPU(5))
    ROOT.RooMinuit(nll).migrad()
    it = model.getVariables().createIterator()
    for parameter in iter(it.Next, None):
        if (parameter.GetName() == parameter_name) and False:
            frame_scan = parameter.frame()
            frac = nll.createProfile(ROOT.RooArgSet(parameter))
            frac.plotOn(frame_scan, RooFit.LineColor(2))
            canvas_scan.cd()
            frame_scan.Draw()
            return canvas_scan
