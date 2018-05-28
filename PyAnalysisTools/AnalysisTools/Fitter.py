import ROOT
import PyAnalysisTools.PlottingUtils.Formatting as fm
import copy
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
        self.mean = kwargs["pdf_config"].mean
        self.sigma = kwargs["pdf_config"].sigma

    def build(self, mode = ""):
        w = ROOT.RooWorkspace("w")
        w.add = getattr(w, "import")
        #build gaussian models
        mean = ROOT.RooRealVar("mean", "mean", *self.mean)
        sigma = ROOT.RooRealVar("sigma", "sigma", *self.sigma)
        gauss2 = ROOT.RooGaussian("gauss2", "gauss2", self.quantity, mean, sigma)
        w.add(gauss2)
        w.factory("EDIT::gauss1(gauss2, mean=expr('mean-m_diff',mean,m_diff[98.5,97,100]))")
        #build background model
        decayrate = ROOT.RooRealVar("decayrate", "decayrate", -1.5e-3, -0.01, -1e-4)
        decayrate2 = ROOT.RooRealVar("decayrate2", "decayrate2", 3e-7, 1e-10, 8e-7)
        background = ROOT.RooGenericPdf("background", "exp(decayrate*triplet_refitted_m + decayrate2*triplet_refitted_m*triplet_refitted_m)", ROOT.RooArgList(self.quantity, decayrate, decayrate2))
        w.add(background)
        #build final model
        if "isBackground" in mode:
           w.factory("SUM::model(nBkg[10000,0,500000]*background)")
        if "isMC" in mode:
           w.factory("SUM::model(nDs[2000,0,20000]*gauss2)")
        else:
           w.factory("SUM::model(nD[500,0,20000]*gauss1, nDs[2000,0,20000]*gauss2, nBkg[10000,0,500000]*background)")
           w.factory("EDIT::model(model, nD=expr('alpha*nDs', alpha[0.3,0.2,0.4], nDs))")
        w.Print()
        ROOT.SetOwnership(w, False)
        return w.pdf("model")


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
        self.file_handles = [FileHandle(file_name=fn) for fn in kwargs["input_files"]]
        self.tree_name = kwargs["tree_name"]
        self.quantity = kwargs["quantity"]
        self.selection = []
        self.data = None
        self.weight = None
        self.pdf_config = PDFConfig(config_file=kwargs["config_file"]) if "config_file" in kwargs else kwargs["config"]
        self.mode = kwargs["mode"]
        if hasattr(self.pdf_config, "quantity"):
            self.quantity = self.pdf_config.quantity
        if hasattr(self.pdf_config, "selection"):
            self.selection=self.pdf_config.selection
        if hasattr(self.pdf_config, "weight"):
            self.weight=self.pdf_config.weight
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
        self.model = self.pdf.build(self.mode)

    def fit(self, return_fit=False, extra_selection=[]):
        print self.file_handles, self.tree_name, self.quantity, self.blind, self.selection, extra_selection
        self.data, self.var = convert(self.file_handles, self.tree_name, self.quantity, self.blind, copy.deepcopy(self.selection), extra_selection, self.weight)
        self.build_model()
        if self.blind:
            # region = ROOT.RooThresholdCategory("region", "Region of {:s}".format(self.quantity),
            #                                      self.var, "SideBand")
            # region.addThreshold(self.blind[0], "SideBand")
            # region.addThreshold(self.blind[1], "Signal")
            fit_result = self.model.fitTo(self.data, RooFit.Range(("left,right")), RooFit.Save(),
                                          RooFit.PrintEvalErrors(-1), RooFit.NumCPU(2))#RooFit.Cut("region==region::SideBand"), RooFit.Save())
        else:
            fit_result = self.model.fitTo(self.data, RooFit.Save(), RooFit.NumCPU(2))
        canvas = ROOT.TCanvas("c", "", 800, 600)
        frame = self.var.frame()
        binning = ROOT.RooBinning(30, self.quantity[1], self.quantity[2])
        self.data.plotOn(frame, ROOT.RooFit.Binning(binning), RooFit.CutRange("left,right"))
        plot_all_components(self.model, frame)
        format_and_draw_frame(canvas, frame, "m_{triplet} [MeV]")
        add_chi2_to_canvas(canvas, frame)
        if "DsPhiPi" in self.mode:
           add_parameters_to_canvas(canvas, self.model)
        if "PerSlice" in self.mode:
           return self.model, fit_result, canvas
        if return_fit:
           return frame, fit_result
        self.output_handle.register_object(canvas)
        self.output_handle.write_and_close()
