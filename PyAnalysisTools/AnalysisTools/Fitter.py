import ROOT
import PyAnalysisTools.PlottingUtils.Formatting as fm
from ROOT import RooFit
from PyAnalysisTools.ROOTUtils.FileHandle import FileHandle
from PyAnalysisTools.AnalysisTools.FitHelpers import convert
from PyAnalysisTools.base.YAMLHandle import YAMLLoader
from PyAnalysisTools.base.OutputHandle import OutputFileHandle
from PyAnalysisTools.PlottingUtils import set_batch_mode
from PyAnalysisTools.PlottingUtils.Formatting import add_text_to_canvas


class PDFConfig(object):
    def __init__(self, **kwargs):
        if "config_file" in kwargs:
            config = YAMLLoader.read_yaml(kwargs["config_file"])
            self.pdf = config.keys()[0]
            for attr, val in config[self.pdf].iteritems():
                self.set_attr(attr, val)
        else:
            for attr, val in kwargs.iteritems():
                self.set_attr(attr, val)

    def set_attr(self, attr, val):
        try:
            setattr(self, attr, eval(val))
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


class PDFChebychev(PDF):
    def __init__(self, **kwargs):
        kwargs.setdefault("pdf_name", "linear")
        self.name = kwargs["pdf_name"]
        self.coefficients = kwargs["pdf_config"].coefficients
        self.quantity = kwargs["var"]

    def build(self):
        coefficients = ROOT.RooArgList()
        for coeff in enumerate(self.coefficients):
            name = "coeff_{:d}".format(coeff[0])
            print "add coeff: ", name
            roo_coeff = ROOT.RooRealVar(name, name, *coeff[1])
            coefficients.add(roo_coeff)
            ROOT.SetOwnership(roo_coeff, False)
        ROOT.SetOwnership(coefficients, False)
        return ROOT.RooChebychev(self.name, self.name, self.quantity, coefficients)


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
        self.data = None
        self.pdf_config = PDFConfig(kwargs["config_file"]) if "config_file" in kwargs else kwargs["config"]
        self.output_handle = OutputFileHandle(output_dir=kwargs["output_dir"])
        self.selection = kwargs["selection"]
        self.blind = kwargs["blind"]
        fm.load_atlas_style()
        set_batch_mode(kwargs["batch"])

    def get_integral(self, min=-1, max=-1):
        var_set = ROOT.RooArgSet(self.var)
        self.var.setRange("integral", min, max)
        return self.model.createIntegral(var_set, RooFit.NormSet(var_set), RooFit.Range("integral"))

    def build_model(self):
        if self.pdf_config.pdf == "gauss":
            self.pdf = PDFGauss(**self.__dict__)
        if self.pdf_config.pdf == "cheb":
            self.pdf = PDFChebychev(**self.__dict__)
        if self.pdf_config.pdf == "generic":
            self.pdf = PDFGeneric(**self.__dict__)
        self.model = self.pdf.build()

    def fit(self, return_fit=False):
        self.data, self.var = convert(self.file_handles, self.tree_name, self.quantity, self.blind, self.selection)
        self.build_model()
        if self.blind:
            print self.quantity
            region = ROOT.RooThresholdCategory("region", "Region of {:s}".format(self.quantity),
                                                 self.var, "SideBand")
            region.addThreshold(self.blind[0], "SideBand")
            region.addThreshold(self.blind[1], "Signal")
            fit_result = self.model.fitTo(self.data, RooFit.Cut("region==region::SideBand"), RooFit.Save())
        else:
            fit_result = self.model.fitTo(self.data, RooFit.Save())
        canvas = ROOT.TCanvas("c", "", 800, 600)
        canvas.cd()
        frame = self.var.frame()
        binning = ROOT.RooBinning(25, 1600., 2250.)
        self.data.plotOn(frame, ROOT.RooFit.Binning(binning))
        self.model.plotOn(frame)
        frame.SetTitle("")
        frame.Draw()
        chi2 = frame.chiSquare()
        frame.SetTitle("")
        frame.Draw()
        chi2 = frame.chiSquare()
        #add_text_to_canvas(canvas, "mean: {:.2f} #pm {:.2f}".format(ean.getVal(), mean.getError()),
        #                    pos={"x": 0.65, "y": 0.87})
        # add_text_to_canvas(canvas, "sigma: {:.2f} #pm {:.2f}".format(sigma.getVal(), sigma.getError()),
        #                    pos={"x": 0.65, "y": 0.82})
        add_text_to_canvas(canvas, "#chi^{2}: " + "{:.2f}".format(chi2), pos={"x": 0.65, "y": 0.77})

        if return_fit:
            return frame, fit_result
        self.output_handle.register_object(canvas)
        self.output_handle.write_and_close()