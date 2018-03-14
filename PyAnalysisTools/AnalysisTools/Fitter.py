import ROOT
import PyAnalysisTools.PlottingUtils.Formatting as fm
from ROOT import RooFit
from PyAnalysisTools.ROOTUtils.FileHandle import FileHandle
from PyAnalysisTools.AnalysisTools.FitHelpers import *
from PyAnalysisTools.base.YAMLHandle import YAMLLoader
from PyAnalysisTools.base.OutputHandle import OutputFileHandle
from PyAnalysisTools.PlottingUtils import set_batch_mode
from PyAnalysisTools.PlottingUtils.Formatting import add_text_to_canvas


class PDFConfig(object):
    def __init__(self, **kwargs):
        kwargs.setdefault("blind", False)
        if "config_file" in kwargs:
            config = YAMLLoader.read_yaml(kwargs["config_file"])
            self.pdf = config.keys()[0]
            self.set_attr("blind", False)
            for attr, val in config[self.pdf].iteritems():
                self.set_attr(attr, val)
        else:
            for attr, val in kwargs.iteritems():
                self.set_attr(attr, val)

    def set_attr(self, attr, val):
        if isinstance(val, dict):
            setattr(self, attr, PDFConfig(**val))
            return
        try:
            setattr(self, attr, eval(val))
        except (TypeError, NameError) as e:
            try:
                if isinstance(val, list):
                    setattr(self, attr, map(eval, val))
                else:
                    raise e
            except (TypeError, NameError):
                setattr(self, attr, val)


class PDF(object):
    def __init__(self):
        pass


class PDFGauss(PDF):
    def __init__(self, **kwargs):
        kwargs.setdefault("pdf_name", "gauss")
        self.name = kwargs["pdf_name"]
        self.quantity = kwargs["var"]
        self.mean = kwargs["pdf_config"].mean
        self.sigma = kwargs["pdf_config"].sigma

    def build(self):
        mean = ROOT.RooRealVar("mean", "mean", *self.mean)
        sigma = ROOT.RooRealVar("sigma", "sigma", *self.sigma)
        ROOT.SetOwnership(mean, False)
        ROOT.SetOwnership(sigma, False)
        return ROOT.RooGaussian(self.name, self.name, self.quantity, mean, sigma)


class PDFLinear(PDF):
    def __init__(self, **kwargs):
        kwargs.setdefault("pdf_name", "linear")
        self.name = kwargs["pdf_name"]

    #def build(self):


class PDFChebychev(PDF):
    def __init__(self, **kwargs):
        kwargs.setdefault("pdf_name", "cheb")
        self.name = kwargs["pdf_name"]
        self.coefficients = kwargs["pdf_config"].coefficients
        self.quantity = kwargs["var"]

    def build(self):
        coefficients = ROOT.RooArgList()
        for coeff in enumerate(self.coefficients):
            name = "coeff_{:d}".format(coeff[0])
            roo_coeff = ROOT.RooRealVar(name, name, *coeff[1])
            coefficients.add(roo_coeff)
            ROOT.SetOwnership(roo_coeff, False)
        ROOT.SetOwnership(coefficients, False)
        return ROOT.RooChebychev(self.name, self.name, self.quantity, coefficients)


class PDFBernstein(PDF):
    def __init__(self, **kwargs):
        kwargs.setdefault("pdf_name", "bernstein")
        self.name = kwargs["pdf_name"]
        self.coefficients = kwargs["pdf_config"].coefficients
        self.quantity = kwargs["var"]

    def build(self):
        coefficients = ROOT.RooArgList()
        for coeff in enumerate(self.coefficients):
            name = "coeff_{:d}".format(coeff[0])
            roo_coeff = ROOT.RooRealVar(name, name, *coeff[1])
            coefficients.add(roo_coeff)
            ROOT.SetOwnership(roo_coeff, False)
        ROOT.SetOwnership(coefficients, False)
        return ROOT.RooBernstein(self.name, self.name, self.quantity, coefficients)


class PDFArgus(PDF):
    def __init__(self, **kwargs):
        print kwargs
        kwargs.setdefault("pdf_name", "argus")
        self.name = kwargs["pdf_name"]
        self.kappa = kwargs["pdf_config"].kappa
        self.const = kwargs["pdf_config"].const
        self.quantity = kwargs["var"]

    def build(self):
        kappa = ROOT.RooRealVar("kappa", "kappa", *self.kappa)
        ROOT.SetOwnership(kappa, False)
        return ROOT.RooArgusBG(self.name, self.name, self.quantity, RooFit.RooConst(self.const), kappa)


class PDFAdd(PDF):
    def __init__(self, **kwargs):
        kwargs.setdefault("pdf_name", "argus")
        self.name = kwargs["pdf_name"]
        self.pdf1_config = kwargs["pdf_config"].pdf1
        self.pdf2_config = kwargs["pdf_config"].pdf2
        self.quantity = kwargs["var"]

    def build(self):
        print self.pdf1_config
        self.pdf1 = PDFArgus(var=self.quantity, **{"pdf_config": self.pdf1_config}).build()
        self.pdf2 = PDFBernstein(var=self.quantity, **{"pdf_config": self.pdf2_config}).build()
        nsig = ROOT.RooRealVar("nsig", "#signal events", 200, 0., 10000)
        nbkg = ROOT.RooRealVar("nbkg", "#background events", 800, 0., 10000)
        ROOT.SetOwnership(nsig, False)
        ROOT.SetOwnership(nbkg, False)
        return ROOT.RooAddPdf(self.name, self.name, ROOT.RooArgList(self.pdf1, self.pdf2), ROOT.RooArgList(nbkg, nsig))


class PDF2Gauss(PDF):
    def __init__(self, **kwargs):
        kwargs.setdefault("pdf_name", "2gauss")
        self.name = kwargs["pdf_name"]
        self.quantity = kwargs["var"]
        self.mean1 = kwargs["pdf_config"].mean1
        self.sigma1 = kwargs["pdf_config"].sigma1
        self.mean2 = kwargs["pdf_config"].mean2
        self.sigma2 = kwargs["pdf_config"].sigma2

    def build(self):
        mean1 = ROOT.RooRealVar("mean1", "mean1", *self.mean1)
        sigma1 = ROOT.RooRealVar("sigma1", "sigma1", *self.sigma1)
        mean2 = ROOT.RooRealVar("mean2", "mean2", *self.mean2)
        sigma2 = ROOT.RooRealVar("sigma2", "sigma2", *self.sigma2)
        decayrate = ROOT.RooRealVar("decayrate", "decayrate", -0.2, -5., 0)
        ROOT.SetOwnership(mean1, False)
        ROOT.SetOwnership(sigma1, False)
        ROOT.SetOwnership(mean2, False)
        ROOT.SetOwnership(sigma2, False)
        ROOT.SetOwnership(decayrate, False)
        self.pdf1 = ROOT.RooGaussian("g1", "g1", self.quantity, mean1, sigma1)
        self.pdf2 = ROOT.RooGaussian("g2", "g2", self.quantity, mean2, sigma2)
        self.pdf3 = ROOT.RooExponential("exp", "exp", self.quantity, decayrate)
        coef1 = ROOT.RooRealVar("coef1", "coef1", 2000, 0, 10000.)
        coef2 = ROOT.RooRealVar("coef2", "coef2", 4000, 0., 10000.)
        coef3 = ROOT.RooRealVar("coef3", "coef3", 20000, 0., 100000.)
        ROOT.SetOwnership(coef1, False)
        ROOT.SetOwnership(coef2, False)
        ROOT.SetOwnership(coef3, False)
        return ROOT.RooAddPdf(self.name, self.name, ROOT.RooArgList(self.pdf1,  self.pdf2 ,self.pdf3), ROOT.RooArgList(coef1, coef2, coef3))


class PDFGeneric(PDF):
    def __init__(self, **kwargs):
        kwargs.setdefault("pdf_name", "generic")
        self.name = kwargs["pdf_name"]
        self.parameters = kwargs["pdf_config"].parameters
        self.quantity = kwargs["var"]
        self.formula = kwargs["pdf_config"].formula

    def build(self):
        var_list = ROOT.RooArgList(self.quantity)
        for parameter in self.parameters:
            name = parameter[0]
            setattr(self, name, ROOT.RooRealVar(name, name, *parameter[1:]))
            ROOT.SetOwnership(getattr(self, name), False)
            var_list.add(getattr(self, name))
        return ROOT.RooGenericPdf(self.name, self.name, self.formula, var_list)


class Fitter(object):
    def __init__(self, **kwargs):
        kwargs.setdefault("batch", True)
        kwargs.setdefault("blind", False)
        kwargs.setdefault("selection", None)
        self.file_handles = [FileHandle(file_name=fn) for fn in kwargs["input_files"]]
        self.tree_name = kwargs["tree_name"]
        self.quantity = kwargs["quantity"]
        self.selection = kwargs["selection"]
        self.data = None
        self.slicing_variables = []
        self.pdf_config = PDFConfig(config_file=kwargs["config_file"]) if "config_file" in kwargs else kwargs["config"]
        if hasattr(self.pdf_config, "quantity"):
            self.quantity = self.pdf_config.quantity
        if hasattr(self.pdf_config, "selection"):
            self.selection=self.pdf_config.selection
            print(self.selection)
        if hasattr(self.pdf_config, "slicing_variables"):
            self.slicing_variables=self.pdf_config.slicing_variables
            print(self.slicing_variables)
        self.output_handle = OutputFileHandle(output_dir=kwargs["output_dir"])
        self.blind = self.pdf_config.blind
        fm.load_atlas_style()
        set_batch_mode(kwargs["batch"])

    def build_model(self):
        if self.pdf_config.pdf == "gauss":
            self.pdf = PDFGauss(**self.__dict__)
        if self.pdf_config.pdf == "cheb":
            self.pdf = PDFChebychev(**self.__dict__)
        if self.pdf_config.pdf == "generic":
            self.pdf = PDFGeneric(**self.__dict__)
        if self.pdf_config.pdf == "add":
            self.pdf = PDFAdd(**self.__dict__)
        if self.pdf_config.pdf == "linear":
            self.pdf = PDFLinear(**self.__dict__)
        if self.pdf_config.pdf == "2gauss":
            self.pdf = PDF2Gauss(**self.__dict__)
        self.model = self.pdf.build()

    def fit(self, return_fit=False, return_count=False, extra_selection="1"):
        selection_temp=list(self.selection)
        selection_temp.append(extra_selection)
        print self.file_handles, self.tree_name, self.quantity, self.blind, self.selection, extra_selection
        self.data, self.var = convert(self.file_handles, self.tree_name, self.quantity, self.blind, selection_temp)
        self.build_model()
        if self.blind:
            # region = ROOT.RooThresholdCategory("region", "Region of {:s}".format(self.quantity),
            #                                      self.var, "SideBand")
            # region.addThreshold(self.blind[0], "SideBand")
            # region.addThreshold(self.blind[1], "Signal")
            fit_result = self.model.fitTo(self.data, RooFit.Range(("left,right")), RooFit.Save(),
                                          RooFit.PrintEvalErrors(-1))#RooFit.Cut("region==region::SideBand"), RooFit.Save())
        else:
            fit_result = self.model.fitTo(self.data, RooFit.Save())
        canvas = ROOT.TCanvas("c"+convert_to_valid_name(extra_selection), "", 800, 600)
        canvas.cd()
        frame = self.var.frame()
        binning = ROOT.RooBinning(30, self.quantity[1], self.quantity[2])
        self.data.plotOn(frame, ROOT.RooFit.Binning(binning), RooFit.CutRange("left,right"))
        self.model.plotOn(frame)
        frame.SetTitle(extra_selection)
        frame.Draw()
        self.frame=frame
        if return_count:
            canvas.SetTitle(convert_to_valid_name(extra_selection))
            n_Ds, n_Bkg = get_Ds_and_Bkg_count(self.model, 1950, 2000)
            add_fit_parameters_to_canvas(canvas, self.model)
            add_text_to_canvas(canvas, extra_selection, pos={"x": 0.2, "y": 0.9})
            add_text_to_canvas(canvas, "N_{D_{s}}: " + "{:.0f}".format(n_Ds), pos={"x": 0.72, "y": 0.77})
            add_text_to_canvas(canvas, "N_{Bkg}: " + "{:.0f}".format(n_Bkg), pos={"x": 0.72, "y": 0.72})
        chi2 = frame.chiSquare()
        add_text_to_canvas(canvas, "#chi^{2}: " + "{:.2f}".format(chi2), pos={"x": 0.72, "y": 0.67})
        if return_fit:
            return frame, fit_result
        self.output_handle.register_object(canvas)
        if return_count:
            return n_Ds, n_Bkg
        self.output_handle.write_and_close()

    def fit_per_bin(self):
        for slicing_variable in self.slicing_variables:
            list_of_slices_and_bin_center = get_list_of_slices(slicing_variable)
            canvas = ROOT.TCanvas(slicing_variable[0], slicing_variable[0], 800, 600)
            hist_Ds_count = ROOT.TH1F("", "",
                                      slicing_variable[1], slicing_variable[2], slicing_variable[3])
            hist_Bkg_count = ROOT.TH1F("", "",
                                       slicing_variable[1], slicing_variable[2], slicing_variable[3])
            for individual_slice in list_of_slices_and_bin_center:
                n_Ds, n_Bkg = self.fit(False, True, individual_slice[0])
                if n_Ds:
                  hist_Ds_count.Fill(individual_slice[1],n_Ds)
                if n_Bkg:
                  hist_Bkg_count.Fill(individual_slice[1],n_Bkg)
            canvas.cd()
            hist_Ds_count.SetLineColor(8)
            hist_Ds_count.SetMarkerColor(8)
            hist_Ds_count.GetXaxis().SetTitle(slicing_variable[0])
            hist_Ds_count.GetYaxis().SetTitle("Ds/Bkg count")
            hist_Ds_count.Draw("E0P")
            hist_Bkg_count.Draw("E0Psame")
            self.output_handle.register_object(hist_Ds_count)
            self.output_handle.register_object(canvas)
        self.output_handle.write_and_close()
