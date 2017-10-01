import ROOT
from itertools import permutations
from copy import copy
from functools import partial
from PyAnalysisTools.base import _logger, InvalidInputError, Utilities
import PyAnalysisTools.PlottingUtils.PlottingTools as PT
import PyAnalysisTools.PlottingUtils.HistTools as HT
import PyAnalysisTools.PlottingUtils.Formatting as FT
from PyAnalysisTools.ROOTUtils.FileHandle import FileHandle
from PyAnalysisTools.PlottingUtils.PlotConfig import find_process_config, PlotConfig
from PyAnalysisTools.ROOTUtils.ObjectHandle import get_objects_from_canvas_by_type, get_objects_from_canvas_by_name
import pathos.multiprocessing as mp


class MuonFakeEstimator(object):
    def __init__(self, plotter_instance, **kwargs):
        kwargs.setdefault("sample_name", "Fakes")
        self.plotter = plotter_instance
        self.file_handles = filter(lambda fh: "data" not in fh.process.lower(), kwargs["file_handles"])
        self.sample_name = kwargs["sample_name"]
        self.type = "DataProvider"

    def execute(self, plot_config):
        return "Fakes", self.get_fake_background(plot_config)

    def get_fake_background(self, plot_config):
        def rebuild_dict_structure():
            for key, data in fake_histograms.iteritems():
                hist_data = {k: v for k, v in data[0][1]}
                fake_histograms[key] = hist_data

        fake_plot_configs = self.retrieve_fake_plot_configs(plot_config)
        if len(fake_plot_configs) == 0:
            return None
        fake_histograms = dict()
        for key, plot_config in fake_plot_configs.iteritems():
            fake_histograms[key] = mp.ThreadPool(self.plotter.ncpu).map(partial(self.plotter.read_histograms,
                                                                        file_handles=self.file_handles),
                                                                        [plot_config])
        rebuild_dict_structure()
        self.plotter.apply_lumi_weights(fake_histograms)
        fake_histograms = self.merge(fake_histograms)
        fake_hist = self.build_fake_contribution(fake_histograms)
        fake_hist.SetName("{:s}_{:s}".format(plot_config.name, self.sample_name))
        return fake_hist

    @staticmethod
    def build_fake_contribution(fake_histograms):
        single_lepton_fake_contribution = filter(lambda (k, v): k[0] == 1, fake_histograms.iteritems())
        fake_contribution = single_lepton_fake_contribution.pop()[-1]
        while len(single_lepton_fake_contribution) > 0:
            fake_contribution.Add(single_lepton_fake_contribution.pop()[-1])
        di_lepton_fake_contribution = filter(lambda (k, v): k[0] == 2, fake_histograms.iteritems())
        while len(di_lepton_fake_contribution) > 0:
            fake_contribution.Add(di_lepton_fake_contribution.pop()[-1], -1)
        tri_lepton_fake_contribution = filter(lambda (k, v): k[0] == 3, fake_histograms.iteritems())
        while len(tri_lepton_fake_contribution) > 0:
            fake_contribution.Add(tri_lepton_fake_contribution.pop()[-1])
        return fake_contribution

    def merge(self, fake_histograms):
        for key, histograms in fake_histograms.iteritems():
            hist = histograms.pop(histograms.keys()[0])
            for process in histograms.keys():
                hist.Add(histograms.pop(process))
            fake_histograms[key] = hist
        return fake_histograms

    def retrieve_fake_plot_configs(self, plot_config):
        n_muon = plot_config.region.n_muon
        fake_plot_configs = dict()
        l = range(1, n_muon + 1)
        combinations = [zip(x, l) for x in permutations(l, len(l))]
        combinations = [i for comb in combinations for i in comb]
        combinations = filter(lambda e: e[1] >= e[0], combinations)
        for combination in combinations:
            fake_plot_configs[combination] = self.retrieve_single_fake_plot_config(plot_config, *combination)
        return fake_plot_configs

    @staticmethod
    def retrieve_single_fake_plot_config(plot_config, n_fake_muon, n_total_muon):
        pc = copy(plot_config)
        pc.name = "fake_single_lep"
        cut = filter(lambda cut: "muon_prompt_n" in cut, pc.cuts)[0]
        cut_index = pc.cuts.index(cut)
        pc.cuts[cut_index] = cut.replace("muon_prompt_n == {:d}".format(n_total_muon),
                                         "(muon_n - muon_prompt_n) == {:d}".format(n_fake_muon))

        return pc


class MuonFakeProvider(object):
    def __init__(self, **kwargs):
        self.file_handle = FileHandle(file_name=kwargs["fake_factor_file"])
        self.fake_factor = {}
        self.read_fake_factors()

    def read_fake_factors(self):
        for i in range(3):
            name = "fake_factor_pt_eta_geq{:d}_jets_dR0.3".format(i)
            canvas_fake_factor = self.file_handle.get_object_by_name(name)
            self.fake_factor[i] = get_objects_from_canvas_by_name(canvas_fake_factor, name)[0]

    def retrieve_fake_factor(self, pt, eta, is_non_prompt, n_jets):
        if is_non_prompt == 0:
            return 1.
        if n_jets > 2:
            n_jets = 2
        b = self.fake_factor[n_jets].FindBin(pt / 1000., eta)
        return self.fake_factor[n_jets].GetBinContent(b)


class MuonFakeDecorator(object):
    def __init__(self, **kwargs):
        input_files = filter(lambda fn: "data" not in fn, kwargs["input_files"])
        self.input_file_handles = [FileHandle(file_name=input_file, open_option="UPDATE",
                                              run_dir=kwargs["run_dir"]) for input_file in input_files]
        self.estimator = MuonFakeProvider(**kwargs)
        self.tree_name = kwargs["tree_name"]
        self.fake_factors = ROOT.std.vector('float')()
        self.branch_name = kwargs["branch_name"]
        self.tree = None
        self.branch = None

    def decorate_event(self):
        def get_n_jets_dr(n_muon, dR):
            n_jets_dr = 0
            #todo: fix to new branch name: muon_jet_dr (v15 onwards)
            for jet_dr in self.tree.muon_all_jet_dr:#[n_muon]:
                if jet_dr > dR:
                    continue
                n_jets_dr += 1
            return n_jets_dr
        self.fake_factors.clear()
        print "decorating event"
        for n_muon in range(self.tree.muon_n):
            self.fake_factors.push_back(self.estimator.retrieve_fake_factor(self.tree.muon_pt[n_muon],
                                                                            self.tree.muon_eta[n_muon],
                                                                            self.tree.muon_is_non_prompt[n_muon],
                                                                            get_n_jets_dr(n_muon, 0.3)))

    def event_loop(self):
        total_entries = self.tree.GetEntries()
        for entry in range(total_entries):
            _logger.debug("Process event {:d}".format(entry))
            self.tree.GetEntry(entry)
            self.decorate_event()
            self.branch.Fill()

    def dump(self, file_handle):
        tdir = file_handle.get_directory("Nominal")
        tdir.cd()
        self.tree.Write()
        file_handle.close()

    def initialise(self, file_handle):
        self.tree = file_handle.get_objects_by_pattern(self.tree_name, "Nominal")[0]
        self.branch = self.tree.Branch(self.branch_name, self.fake_factors)

    def execute(self):
        for file_handle in self.input_file_handles:
            try:
                self.initialise(file_handle)
                self.event_loop()
                self.dump(file_handle)
            except Exception as e:
                raise e
            finally:
                file_handle.close()


class MuonFakeCalculator(object):
    def __init__(self, **kwargs):
        from PyAnalysisTools.PlottingUtils.Plotter import Plotter
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
            self.plotter.apply_lumi_weights(self.histograms)
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

    def run(self):
        self.plot_jet_bins()
        self.plot_fake_factors()
        self.plot_fake_factors_2D()
        self.plotter.output_handle.write_and_close()
