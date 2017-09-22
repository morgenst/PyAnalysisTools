import ROOT
import PyAnalysisTools.PlottingUtils.Formatting as fm
from PyAnalysisTools.ROOTUtils.FileHandle import FileHandle
from PyAnalysisTools.AnalysisTools.FitHelpers import convert
from PyAnalysisTools.base.YAMLHandle import YAMLLoader
from PyAnalysisTools.base.OutputHandle import OutputFileHandle
from PyAnalysisTools.PlottingUtils import set_batch_mode
from PyAnalysisTools.PlottingUtils.Formatting import add_text_to_canvas


class PDFConfig(object):
    def __init__(self, config_file):
        raw_config = YAMLLoader.read_yaml(config_file)
        self.pdf = raw_config.keys()[0]
        for attr, val in raw_config[self.pdf].iteritems():
            setattr(self, attr, eval(val))


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


class Fitter(object):
    def __init__(self, **kwargs):
        kwargs.setdefault("batch", True)
        self.file_handles = [FileHandle(file_name=fn) for fn in kwargs["input_files"]]
        self.tree_name = kwargs["tree_name"]
        self.quantity = kwargs["quantity"]
        self.data = None
        self.pdf_config = PDFConfig(kwargs["config_file"])
        self.output_handle = OutputFileHandle(output_dir=kwargs["output_dir"])
        fm.load_atlas_style()
        set_batch_mode(kwargs["batch"])

    def build_model(self):
        if self.pdf_config.pdf == "gauss":
            self.pdf = PDFGauss(**self.__dict__)
        self.model = self.pdf.build()

    def fit(self):
        self.data, self.var = convert(self.file_handles, self.tree_name, self.quantity)
        self.build_model()
        self.model.fitTo(self.data)
        canvas = ROOT.TCanvas("c", "", 800, 600)
        canvas.cd()
        frame = self.var.frame()
        binning = ROOT.RooBinning(25, 1500., 2250.)
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

        self.output_handle.register_object(canvas)
        self.output_handle.write_and_close()