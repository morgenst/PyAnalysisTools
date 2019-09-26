from __future__ import print_function

import copy

from builtins import map
from builtins import object

from PyAnalysisTools.AnalysisTools.FitHelpers import *
from PyAnalysisTools.PlottingUtils import set_batch_mode
from PyAnalysisTools.ROOTUtils.FileHandle import FileHandle
from PyAnalysisTools.base.OutputHandle import OutputFileHandle
from PyAnalysisTools.base.YAMLHandle import YAMLLoader
from ROOT import RooFit


class PDFConfig(object):
    def __init__(self, **kwargs):
        kwargs.setdefault("blind", False)
        if "fit_config_file" in kwargs:
            config = YAMLLoader.read_yaml(kwargs["fit_config_file"])
            self.pdf = list(config.keys())[0]
            self.set_attr("blind", False)
            for attr, val in list(config[self.pdf].items()):
                self.set_attr(attr, val)
        else:
            for attr, val in list(kwargs.items()):
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
                    setattr(self, attr, list(map(eval, val)))
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
        print(self.pdf1_config)
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
        self.mode = kwargs["mode"]

    def build(self):
        w = ROOT.RooWorkspace("w")
        w.add = getattr(w, "import")
        #build gaussian models
        mean = ROOT.RooRealVar("mean", "mean", *self.mean)
        sigma = ROOT.RooRealVar("sigma", "sigma", *self.sigma)
        gauss2 = ROOT.RooGaussian("gauss2", "gauss2", self.quantity, mean, sigma)
        w.add(gauss2)
        w.factory("EDIT::gauss1(gauss2, mean=expr('mean-m_diff',mean,m_diff[98,92,104]))")
        #build background and final model
        if "Ds" in self.mode:
            w.factory("EXPR::background('exp(decayrate*triplet_refitted_m+decayrate2*triplet_refitted_m*triplet_refitted_m)',decayrate[-1.5e-3, -0.01, -1e-4], decayrate2[3e-7, 1e-10, 3e-6], triplet_refitted_m)")
            if "Background" in self.mode:
                w.factory("SUM::model(nBkg[10000,0,10000000]*background)")
            if "MC" in self.mode:
                w.factory("SUM::model(nDs[2000,0,20000]*gauss2)")
            if "Data" in self.mode:
                w.factory("SUM::model(nD[500,0,20000]*gauss1, nDs[2000,0,20000]*gauss2, nBkg[10000,0,10000000]*background)")
                w.factory("EDIT::model(model, nD=expr('alpha*nDs', alpha[0.3,0.1,0.5], nDs))")
        if "Phi" in self.mode:
            w.factory("EXPR::background('exp(decayrate*triplet_muon_mass+decayrate2*triplet_muon_mass*triplet_muon_mass)',decayrate[1e-7, 1.], decayrate2[1e-7, -1e-6, 1e-6], triplet_muon_mass)")
            w.factory("SUM::model(nDs[0,300000]*gauss2, nBkg[0,10000000]*background)")
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


class PDFBFraction(PDF):
     def __init__(self, **kwargs):
         kwargs.setdefault("pdf_name", "BFraction")
         self.name = kwargs["pdf_name"]
         self.var = kwargs["var"]
         self.templatepath = kwargs["templatepath"]

     def build(self):
         w = ROOT.RooWorkspace("w")
         w.add = getattr(w, "import")
         LXY_set = ROOT.RooArgSet(self.var)
         LXY_list = ROOT.RooArgList(self.var)
         #Define bin for rebinning
         #binlist = array.array('d', [-4+j*0.25 for j in xrange(24)]+[2. + i*0.5 for i in xrange(4)]+[4. + i*1 for i in xrange(11)])
         #Make pdf for bb contribution
         hist_bb = get_hist_from_canvas(self.templatepath, "triplet_slxy_bb",
                                        "triplet_slxy_bb_HFbbcc*")
         #hist_bb = hist_bb.Rebin(len(binlist)-1,"bb",binlist)
         #hist_bb.Scale(1., "width")
         datahist_bb = ROOT.RooDataHist("datahist_bb","datahist_bb",LXY_list,hist_bb)
         w.add(ROOT.RooHistPdf("histpdf_bb","histpdf_bb",LXY_set,datahist_bb,2))
         #Make pdf for cc contribution
         hist_cc = get_hist_from_canvas(self.templatepath, "triplet_slxy_cc",
                                        "triplet_slxy_cc_HFbbcc*")
         #hist_cc = hist_cc.Rebin(len(binlist)-1,"cc",binlist)
         #hist_cc.Scale(1., "width")
         datahist_cc = ROOT.RooDataHist("datahist_cc","datahist_cc",LXY_list,hist_cc)
         w.add(ROOT.RooHistPdf("histpdf_cc","histpdf_cc",LXY_set,datahist_cc,2))
         #Make pdf for background contribution
         hist_bkg = get_hist_from_canvas(self.templatepath, "triplet_slxy_bkg",
                                         "triplet_slxy_bkg_Data*")
         #hist_bkg = hist_bkg.Rebin(len(binlist)-1,"bkg",binlist)
         #hist_bkg.Scale(1., "width")
         datahist_bkg = ROOT.RooDataHist("datahist_bkg","datahist_bkg",LXY_list,hist_bkg)
         w.add(ROOT.RooHistPdf("histpdf_bkg","histpdf_bkg",LXY_set,datahist_bkg,2))
         #Add up pdf
         w.factory("BFraction[0.05, 0.5]")
         w.factory("n_bkg[0., 80000.]")
         w.factory("n_sig[0., 80000.]")
         w.factory("SUM::tmp1(n_bb[0]*histpdf_bb, n_cc[0]*histpdf_cc, n_bkg*histpdf_bkg)")
         w.factory("EDIT::tmp2(tmp1, n_bb=expr('BFraction*n_sig', BFraction, n_sig))")
         w.factory("EDIT::model(tmp2, n_cc=expr('(1.0-BFraction)*n_sig', BFraction, n_sig))")
         ROOT.SetOwnership(w, False)
         return w.pdf("model")


class Fitter(object):
    def __init__(self, **kwargs):
        kwargs.setdefault("batch", True)
        kwargs.setdefault("blind", False)
        self.file_handles = [FileHandle(file_name=fn) for fn in kwargs["input_files"]]
        self.tree_name = kwargs["tree_name"]
        self.quantity = kwargs["quantity"]
        self.templatepath = kwargs["templatepath"] if "templatepath" in kwargs else None
        self.selection = []
        self.data = None
        self.weight = None
        self.nbin = 40
        self.xtitle = "variable"
        self.logy = False
        self.pdf_config = PDFConfig(fit_config_file=kwargs["fit_config_file"]) if "fit_config_file" in kwargs else kwargs["fit_config_config"]
        self.mode = kwargs["mode"]
        if hasattr(self.pdf_config, "quantity"):
            self.quantity = self.pdf_config.quantity
            self.var = ROOT.RooRealVar(self.quantity[0], self.quantity[0],
                                       self.quantity[1], self.quantity[2])
        if hasattr(self.pdf_config, "selection"):
            self.selection=self.pdf_config.selection
        if hasattr(self.pdf_config, "weight"):
            self.weight=self.pdf_config.weight
        if hasattr(self.pdf_config, "nbin"):
            self.nbin=self.pdf_config.nbin
        if hasattr(self.pdf_config, "xtitle"):
            self.xtitle=self.pdf_config.xtitle
        if hasattr(self.pdf_config, "logy"):
            self.logy=self.pdf_config.logy
        if hasattr(self.pdf_config, "blind"):
            self.blind=self.pdf_config.blind
        self.output_handle = OutputFileHandle(output_dir=kwargs["output_dir"])
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
        if self.pdf_config.pdf == "doublegauss":
            self.pdf = PDF2Gauss(**self.__dict__)
        if self.pdf_config.pdf == "BFraction":
            self.pdf = PDFBFraction(**self.__dict__)
        self.model = self.pdf.build()

    def fit(self, return_fit=False, extra_selection=[]):
        print(self.file_handles, self.tree_name, self.var, self.quantity, self.blind, self.selection, extra_selection)
        self.data = create_roodata(self.file_handles, self.tree_name, self.var, self.quantity, self.blind, copy.deepcopy(self.selection), extra_selection, self.weight)
        self.build_model()
        if self.blind:
             #region = ROOT.RooThresholdCategory("region", "Region of {:s}".format(self.quantity),
             #                                     self.var, "SideBand")
             #region.addThreshold(self.blind[0], "SideBand")
             #region.addThreshold(self.blind[1], "Signal")
             fit_result = self.model.fitTo(self.data,  RooFit.Save(), RooFit.Range("left,right"),
                                           RooFit.NumCPU(5))#, RooFit.Cut("region==region::Signal"))
        else:
            fit_result = self.model.fitTo(self.data, RooFit.Save(), RooFit.NumCPU(5))
        canvas = ROOT.TCanvas("c", "", 800, 600)
        frame = self.var.frame()
        canvas.cd()
        if self.logy:
           ROOT.gPad.SetLogy()
           frame.SetMaximum(1.5e4)
        binning = ROOT.RooBinning(self.nbin, self.quantity[1], self.quantity[2])
        self.data.plotOn(frame, ROOT.RooFit.Binning(binning))
        plot_all_components(self.model, frame)
        format_and_draw_frame(canvas, frame, self.xtitle)
        add_parameters_to_canvas(canvas, self.model, frame)#Chi-squared wrt binning
        if "PerSlice" in self.mode:
           return self.model, fit_result, canvas
        if return_fit:
           return frame, fit_result
        self.output_handle.register_object(canvas)
        self.output_handle.write_and_close()
