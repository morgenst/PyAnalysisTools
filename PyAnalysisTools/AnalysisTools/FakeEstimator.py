import ROOT
from copy import copy
from PyAnalysisTools.base import _logger, InvalidInputError, Utilities
from PyAnalysisTools.PlottingUtils.Plotter import Plotter
import PyAnalysisTools.PlottingUtils.PlottingTools as PT
import PyAnalysisTools.PlottingUtils.HistTools as HT
import PyAnalysisTools.PlottingUtils.Formatting as FT
from PyAnalysisTools.ROOTUtils.FileHandle import FileHandle
from PyAnalysisTools.PlottingUtils.PlotConfig import parse_and_build_plot_config, find_process_config, PlotConfig
from PyAnalysisTools.ROOTUtils.ObjectHandle import get_objects_from_canvas_by_type
import pathos.multiprocessing as mp


class MuonFakeEstimator(object):
    def __init__(self, **kwargs):
        if not "input_files" in kwargs:
            raise InvalidInputError("No input files provided")
        kwargs.setdefault("lumi", 36.3)
        self.file_handles = [FileHandle(file_name=file_name) for file_name in kwargs["input_files"]]
        self.tree_name = kwargs["tree_name"]
        self.ncpu = 10
        self.plotter = Plotter(plot_config_files=[kwargs["plot_config"]], **kwargs)
        self.plot_config = self.plotter.plot_configs #parse_and_build_plot_config(kwargs["plot_config"])
        self.plotter.read_cutflows()
        self.jet_binned_histograms = {}
        self.histograms = {}
        self.enable_bjets = kwargs["enable_bjets"]

    def get_plots(self, dist, relation_op, enable_dr=True):
        plot_config_base = filter(lambda pc: pc.name == dist, self.plot_config)[0]
        hists = {}
        for n_jet in range(3):
            plot_config = copy(plot_config_base)
            if self.enable_bjets:
                jet_selector = "Sum$(muon_bjet_dr > 0.3)" if enable_dr else "jet_n"
            else:
                jet_selector = "Sum$(muon_jet_dr > 0.3)" if enable_dr else "jet_n"
            if "numerator" in dist:
                cut = ["({:s} {:s} {:d}) && muon_d0sig > 3 && muon_isolGradient==1".format(jet_selector,
                                                                                           relation_op, n_jet)]
                cut.append("MC:(muon_truth_mother_pdg==12 || muon_truth_mother_pdg==13)")
            elif "denominator" in dist:
                cut = ["({:s} {:s} {:d}) && fake_muon_d0sig > 3 && fake_muon_isolGradient==0".format(jet_selector.replace("muon", "fake_muon"),
                                                                                                     relation_op, n_jet)]
                cut.append("MC:(fake_muon_truth_mother_pdg==12 || fake_muon_truth_mother_pdg==13)")
            plot_config.cuts = cut
            suffix = "eq" if relation_op == "==" else "geq"
            jet_selector_name = "" if not enable_dr else "_dR0.3"
            name = "{:s}_{:s}{:d}_jets{:s}".format(dist, suffix, n_jet, jet_selector_name)
            plot_config.name = name
            numerator_hists = mp.ThreadPool(min(self.ncpu, 1)).map(self.plotter.read_histograms, [plot_config])
            for plot_config, histograms in numerator_hists:
                self.plotter.histograms = {}
                histograms = filter(lambda hist: hist is not None, histograms)
                self.plotter.categorise_histograms(plot_config, histograms)
            self.plotter.apply_lumi_weights()
            for hist_set in self.plotter.histograms.values():
                for process_name in hist_set.keys():
                    _ = find_process_config(process_name, self.plotter.process_configs)
            self.plotter.merge_histograms()
            hists[name] = self.plotter.histograms
        return hists

    def make_plot(self, hists, plot_config):
        data = hists[0].pop("Data")
        canvas = PT.plot_objects(hists[0], plot_config, process_configs=self.plotter.process_configs)
        PT.add_data_to_stack(canvas, data, plot_config)
        ymax = 1.3 * max(map(lambda h: h.GetMaximum(), get_objects_from_canvas_by_type(canvas, "TH1F")))
        ymin = 0.
        if plot_config.logy:
            ymin = 0.1
        get_objects_from_canvas_by_type(canvas, "TH1F")[0].GetYaxis().SetRangeUser(ymin, ymax)
        canvas.Update()
        canvas.Modified()
        hists[0]["Data"] = data
        labels = ["Prompt MC", "Data"]
        FT.add_legend_to_canvas(canvas, labels=labels)
        FT.decorate_canvas(canvas, plot_config)
        self.plotter.output_handle.register_object(canvas)

    def plot_jet_bins(self):
        self.histograms = Utilities.merge_dictionaries(self.get_plots("numerator_pt", "=="))
        self.histograms = Utilities.merge_dictionaries(self.histograms, self.get_plots("numerator_pt", ">="))
        self.histograms = Utilities.merge_dictionaries(self.histograms, self.get_plots("denominator_pt", "=="))
        self.histograms = Utilities.merge_dictionaries(self.histograms, self.get_plots("denominator_pt", ">="))
        self.histograms = Utilities.merge_dictionaries(self.histograms, self.get_plots("numerator_pt", "==", False))
        self.histograms = Utilities.merge_dictionaries(self.histograms, self.get_plots("numerator_pt", ">=", False))
        self.histograms = Utilities.merge_dictionaries(self.histograms, self.get_plots("denominator_pt", "==", False))
        self.histograms = Utilities.merge_dictionaries(self.histograms, self.get_plots("denominator_pt", ">=", False))
        for jet_bin, hist_set in self.histograms.iteritems():
            self.make_plot(hist_set.values(), hist_set.keys()[0])

    def subtract_prompt(self):
        data_hists = {}
        for jet_bin, data in self.histograms.iteritems():
            data_hists[jet_bin] = {}
            for plot_config, hists in data.iteritems():
                data_hists[jet_bin][plot_config] = hists["Data"]
                data_hists[jet_bin][plot_config].Add(hists["Prompt"], -1.0)
                if hasattr(plot_config, "rebin") and plot_config.rebin is not None:
                    data_hists[jet_bin][plot_config] = HT.rebin(data_hists[jet_bin][plot_config], plot_config.rebin)
        return data_hists

    @staticmethod
    def calculate_fake_factor(numerator, denominator, name):
        fake_factor = numerator.Clone(name)
        fake_factor.Divide(denominator)
        return fake_factor

    def plot_fake_factors(self):
        fake_factors = {}
        data_histograms = self.subtract_prompt()
        ordering = []
        for jet_selector in ["_dR0.3", ""]:
            for n_jet in range(3):
                for op in ["eq", "geq"]:
                    numerator_name = "numerator_pt_{:s}{:d}_jets{:s}".format(op, n_jet, jet_selector)
                    numerator_pt = data_histograms[numerator_name]
                    denominator_pt = data_histograms[numerator_name.replace("numerator", "denominator")]
                    ff_name = numerator_name.replace("numerator", "ff")
                    fake_factors[ff_name] = self.calculate_fake_factor(numerator_pt.values()[0], denominator_pt.values()[0],
                                                                       ff_name)
                    ordering.append(ff_name)
        plot_config = PlotConfig(draw="MarkerError", color=[ROOT.kBlack, ROOT.kRed, ROOT.kBlue, ROOT.kGreen,
                                                            ROOT.kCyan, ROOT.kGray] * 2, name="fake_factor_pt",
                                 watermark="Internal", xtitle="p_{T} [GeV]", ytitle="Fake factor", ordering=ordering,
                                 ymin=0., ymax=1., styles=[20] * int(len(fake_factors) / 2) + [21]*int(len(fake_factors) / 2))
        canvas = PT.plot_objects(fake_factors, plot_config)
        labels = ["=0 jet (dR > 0.3)", ">= 0 jet (dR > 0.3)", "=1 jet (dR > 0.3)", ">=1 jet (dR > 0.3)",
                  "=2 jet (dR > 0.3)", ">=2 jet  (dR > 0.3)", "=0 jet", ">= 0 jet", "=1 jet", ">=1 jet", "=2 jet",
                  ">=2 jet"]
        FT.add_legend_to_canvas(canvas, labels=labels)
        FT.decorate_canvas(canvas, plot_config)
        self.plotter.output_handle.register_object(canvas)

    def plot_fake_factors_2D(self):
        self.histograms = Utilities.merge_dictionaries(self.get_plots("numerator_pt_eta", "=="))
        self.histograms = Utilities.merge_dictionaries(self.histograms, self.get_plots("numerator_pt_eta", ">="))
        self.histograms = Utilities.merge_dictionaries(self.histograms, self.get_plots("denominator_pt_eta", "=="))
        self.histograms = Utilities.merge_dictionaries(self.histograms, self.get_plots("denominator_pt_eta", ">="))
        self.histograms = Utilities.merge_dictionaries(self.histograms, self.get_plots("numerator_pt_eta", "==", False))
        self.histograms = Utilities.merge_dictionaries(self.histograms, self.get_plots("numerator_pt_eta", ">=", False))
        self.histograms = Utilities.merge_dictionaries(self.histograms, self.get_plots("denominator_pt_eta", "==", False))
        self.histograms = Utilities.merge_dictionaries(self.histograms, self.get_plots("denominator_pt_eta", ">=", False))
        data_histograms = self.subtract_prompt()
        for jet_selector in ["_dR0.3", ""]:
            for n_jet in range(3):
                for op in ["eq", "geq"]:
                    name = "fake_factor_pt_eta_{:s}{:d}_jets{:s}".format(op, n_jet, jet_selector)
                    plot_config = PlotConfig(name=name, draw_option="COLZ",
                                             xtitle="p_{T} [GeV]", ytitle="#eta", ztitle="fake factor", watermark="Internal")
                    numerator_pt = data_histograms[name.replace("fake_factor", "numerator")]
                    denominator_pt = data_histograms[name.replace("fake_factor", "denominator")]

                    fake_factor_hist = self.calculate_fake_factor(numerator_pt.values()[0], denominator_pt.values()[0],
                                                                  name)
                    canvas = PT.plot_2d_hist(fake_factor_hist, plot_config)
                    fake_factor_hist.GetZaxis().SetRangeUser(0., 1.)
                    FT.decorate_canvas(canvas, plot_config)
                self.plotter.output_handle.register_object(canvas)

    def run(self, bjet=False):
        self.plot_jet_bins()
        self.plot_fake_factors()
        self.plot_fake_factors_2D()
        self.plotter.output_handle.write_and_close()
