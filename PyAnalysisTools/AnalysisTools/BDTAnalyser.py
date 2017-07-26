import ROOT
import PyAnalysisTools.PlottingUtils.PlottingTools as PT
import PyAnalysisTools.PlottingUtils.Formatting as FT
import PyAnalysisTools.PlottingUtils.Formatting as FM
from PyAnalysisTools.base import _logger, InvalidInputError
from PyAnalysisTools.base.OutputHandle import OutputFileHandle
from PyAnalysisTools.ROOTUtils.FileHandle import FileHandle
from PyAnalysisTools.PlottingUtils.PlotConfig import PlotConfig as PC
from PyAnalysisTools.AnalysisTools.StatisticsTools import get_KS


class BDTAnalyser(object):
    def __init__(self, **kwargs):
        if not "input_files" in kwargs:
            raise InvalidInputError("No input files provided")
        kwargs.setdefault("output_path", "./")
        self.file_handles = [FileHandle(file_name=file_name) for file_name in kwargs["input_files"]]
        self.output_handle = OutputFileHandle(output_dir=kwargs["output_path"])
        for arg, val in kwargs.iteritems():
            if not hasattr(self, arg):
                setattr(self, arg, val)
        ROOT.gROOT.SetBatch(True)

    def analyse(self):
        self.analyse_train_variables()
        self.perform_overtraining_check()
        self.perform_correlation_analysis()
        self.output_handle.write_and_close()

    def perform_overtraining_check(self):
        for file_handle in self.file_handles:
            self.analyse_overtraining(file_handle)

    def analyse_train_variables(self):
        for file_handle in self.file_handles:
            self.plot_train_variables(file_handle)

    def plot_train_variables(self, file_handle):
        def classify():
            variables = {}
            for signal_hist in signal_hists:
                variables[signal_hist.GetName().replace("__Signal", "")] = [signal_hist]
            for background_hist in background_hists:
                variables[background_hist.GetName().replace("__Background", "")].append(background_hist)
            return variables
        signal_hists = file_handle.get_objects_by_pattern("[A-z]*__Signal",
                                                          "Method_BDT/BDTG")
        background_hists = file_handle.get_objects_by_pattern("[A-z]*__Background",
                                                              "Method_BDT/BDTG")
        variables_hists = classify()
        for variable_name, variable_hists in variables_hists.iteritems():
            plot_config = PC(name="{:s}_{:d}".format(variable_name, self.file_handles.index(file_handle)),
                             color=[ROOT.kRed, ROOT.kBlue],
                             draw="Hist",
                             watermark="Internal",
                             normalise=True,
                             ymax=0.2)
            canvas = PT.plot_histograms(variable_hists, plot_config)
            FM.decorate_canvas(canvas, plot_config)
            self.output_handle.register_object(canvas, tdir="train_variables")

    def analyse_overtraining(self, file_handle):
        training_score_signal = file_handle.get_object_by_name("MVA_BDTG_Train_S", "Method_BDT/BDTG")
        training_score_background = file_handle.get_object_by_name("MVA_BDTG_Train_B", "Method_BDT/BDTG")
        eval_score_signal = file_handle.get_object_by_name("MVA_BDTG_S", "Method_BDT/BDTG")
        eval_score_background = file_handle.get_object_by_name("MVA_BDTG_B", "Method_BDT/BDTG")

        ymax = 1.6 * max([training_score_signal.GetMaximum(), training_score_background.GetMaximum(),
                          eval_score_signal.GetMaximum(), eval_score_background.GetMaximum()])

        kolmogorov_signal = get_KS(training_score_signal, eval_score_signal)
        kolmogorov_background = get_KS(training_score_background, eval_score_background)
        plot_config = PC(name="overtrain_{:d}".format(self.file_handles.index(file_handle)),
                         color=ROOT.kRed,
                         draw="Marker",
                         style=20,
                         ymax=ymax,
                         watermark="Internal")
        canvas = PT.plot_obj(training_score_signal, plot_config)
        plot_config.style = 24
        PT.add_object_to_canvas(canvas, eval_score_signal, plot_config)
        plot_config.style = 20
        plot_config.color = ROOT.kBlue
        PT.add_object_to_canvas(canvas, training_score_background, plot_config)
        plot_config.style = 24
        PT.add_object_to_canvas(canvas, eval_score_background, plot_config)
        FM.decorate_canvas(canvas, plot_config)
        FT.add_text_to_canvas(canvas, "KS (signal): {:.2f}".format(kolmogorov_signal), pos={'x': 0.18, 'y': 0.9},
                              color=ROOT.kRed)
        FT.add_text_to_canvas(canvas, "KS (bkg): {:.2f}".format(kolmogorov_background), pos={'x': 0.18, 'y': 0.85},
                              color=ROOT.kBlue)
        labels = ["signal (train)", "signal (eval)", "background (train)", "background (eval)"]
        FT.add_legend_to_canvas(canvas, labels=labels, xl=0.18, xh=0.3, yl=0.6, yh=0.82)
        self.output_handle.register_object(canvas, tdir="overtrain")

    def perform_correlation_analysis(self):
        for file_handle in self.file_handles:
            self.analyse_correlations(file_handle)

    def analyse_correlations(self, file_handle):
        index = self.file_handles.index(file_handle)
        linear_corr_coeff_signal = file_handle.get_object_by_name("CorrelationMatrixS")
        linear_corr_coeff_background = file_handle.get_object_by_name("CorrelationMatrixB")
        plot_config = PC(name="linear_corr_coeff_signal_{:d}".format(index), title="signal", dist=None,
                         draw_option="COLZTEXT", ytitle="", ztitle="lin. correlation [%]")
        canvas_corr_coeff_signal = PT.plot_obj(linear_corr_coeff_signal, plot_config)
        plot_config.title = "background"
        plot_config.name = plot_config.name.replace("signal", "background")
        canvas_corr_coeff_background = PT.plot_obj(linear_corr_coeff_background, plot_config)
        self.output_handle.register_object(canvas_corr_coeff_signal)
        self.output_handle.register_object(canvas_corr_coeff_background)
        correlation_hists_signal = file_handle.get_objects_by_pattern("scat_.*_Signal_Id",
                                                                      "InputVariables_Id/CorrelationPlots")
        correlation_hists_background = file_handle.get_objects_by_pattern("scat_.*_Background_Id",
                                                                          "InputVariables_Id/CorrelationPlots")
        plot_config_corr = PC(name="correlation_hist", dist=None, draw_option="COLZ", watermark="Internal")
        for hist in correlation_hists_signal:
            variable_info = hist.GetName().split("_")[1:-2]
            plot_config_corr.name = "corr_" + "_".join(variable_info) + "_signal_{:d}".format(index)
            split_index = variable_info.index("vs")
            variable_x = "_".join(variable_info[:split_index])
            variable_y = "_".join(variable_info[split_index+1:])
            plot_config_corr.xtitle = variable_x
            plot_config_corr.ytitle = variable_y
            plot_config_corr.ztitle = "Entries"
            canvas = PT.plot_obj(hist, plot_config_corr)
            FM.decorate_canvas(canvas, plot_config_corr)
            self.output_handle.register_object(canvas)
        for hist in correlation_hists_background:
            plot_config_corr.name = "corr_" + "_".join(hist.GetName().split("_")[1:-2]) + "_background_{:d}".format(index)
            canvas = PT.plot_obj(hist, plot_config_corr)
            FM.decorate_canvas(canvas, plot_config_corr)
            self.output_handle.register_object(canvas)

    def fit_score(self):
        bdt_score = ROOT.RooRealVar(self.branch_name, "BDT score", -0.9, 1.)
        chain = ROOT.TChain("Nominal/" + self.tree_name)
        for file_handle in self.file_handles[1:]:
            chain.Add(file_handle.file_name)
        p0 = ROOT.RooRealVar("p0", "p0", 1, -10., 10.)
        p1 = ROOT.RooRealVar("p1", "p1", 1, -10., 10.)
        p2 = ROOT.RooRealVar("p2", "p2", 1, -100., 100.)
        p3 = ROOT.RooRealVar("p3", "p3", 1, -10., 10.)
        p4 = ROOT.RooRealVar("p4", "p4", 1, -10., 10.)
        norm = ROOT.RooRealVar("norm", "norm", chain.GetEntries(), 0., chain.GetEntries() * 2)
        mass = ROOT.RooRealVar("object_m", "object_m", 0., 100000.)
        genpdf = ROOT.RooGenericPdf("genpdf", "genpdf",
                                    "norm * (p0 + p1 * exp(({:s} + 1.) *p2)  + p3 * abs({:s})^(({:s} + 1.)*p4))".format(self.branch_name,
                                                                                                         self.branch_name,
                                                                                                         self.branch_name),
                                    ROOT.RooArgList(bdt_score, p0, p1, p2, p3, p4, norm))
        data = ROOT.RooDataSet("data", "BDT_170526", chain, ROOT.RooArgSet(bdt_score, mass),
                               "object_m/1000. < 1713. || object_m/1000. > 1841.")
        frame = bdt_score.frame()
        data.plotOn(frame, ROOT.RooFit.Name("data"), ROOT.RooFit.Binning(25))
        fit_result = genpdf.fitTo(data, ROOT.RooFit.Save())
        canvas = ROOT.TCanvas("c", "c", 800, 600)
        canvas.cd()
        genpdf.plotOn(frame, ROOT.RooFit.Name("model"))
        PT.add_fit_to_canvas(canvas, fit_result, genpdf, frame)
        FM.add_atlas_label(canvas, "Internal")
        frame.Draw()
        canvas.Modified()
        self.output_handle.register_object(canvas)
        self.output_handle.write_and_close()
