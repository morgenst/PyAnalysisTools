from __future__ import print_function

import traceback
from array import array
from copy import deepcopy
from functools import partial
from itertools import permutations

import pathos.multiprocessing as mp
from builtins import filter
from builtins import object
from builtins import range
from builtins import zip

import ROOT
# from PyAnalysisTools.PlottingUtils.Plotter import Plotter
from PyAnalysisTools.base.FileHandle import FileHandle
from PyAnalysisTools.ROOTUtils.ObjectHandle import get_objects_from_canvas_by_name
from PyAnalysisTools.base import _logger


class MuonFakeEstimator(object):
    def __init__(self, plotter_instance, **kwargs):
        kwargs.setdefault("sample_name", "Fakes")
        self.plotter = plotter_instance
        self.file_handles = [fh for fh in kwargs["file_handles"] if "data" in fh.process.lower()]
        self.sample_name = kwargs["sample_name"]
        self.type = "DataProvider"

    def execute(self, plot_config):
        return "Fakes", self.get_fake_background(plot_config)

    def get_fake_background(self, plot_config):
        def rebuild_dict_structure():
            for key, data in list(fake_histograms.items()):
                hist_data = {k: v for k, v in data[0][1]}
                fake_histograms[key] = hist_data

        fake_plot_configs = self.retrieve_fake_plot_configs(plot_config)
        if len(fake_plot_configs) == 0:
            return None
        fake_histograms = dict()
        for key, plot_config in list(fake_plot_configs.items()):
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
        single_lepton_fake_contribution = [k_v1 for k_v1 in iter(list(fake_histograms.items())) if k_v1[0][0] == 1]
        fake_contribution = single_lepton_fake_contribution.pop()[-1]
        while len(single_lepton_fake_contribution) > 0:
            fake_contribution.Add(single_lepton_fake_contribution.pop()[-1])
        di_lepton_fake_contribution = [k_v2 for k_v2 in iter(list(fake_histograms.items())) if k_v2[0][0] == 2]
        while len(di_lepton_fake_contribution) > 0:
            fake_contribution.Add(di_lepton_fake_contribution.pop()[-1], -1)
        tri_lepton_fake_contribution = [k_v3 for k_v3 in iter(list(fake_histograms.items())) if k_v3[0][0] == 3]
        while len(tri_lepton_fake_contribution) > 0:
            fake_contribution.Add(tri_lepton_fake_contribution.pop()[-1])
        return fake_contribution

    def merge(self, fake_histograms):
        for key, histograms in list(fake_histograms.items()):
            hist = histograms.pop(list(histograms.keys())[0])
            for process in list(histograms.keys()):
                hist.Add(histograms.pop(process))
            fake_histograms[key] = hist
        return fake_histograms

    def retrieve_fake_plot_configs(self, plot_config):
        n_muon = plot_config.region.n_muon
        fake_plot_configs = dict()
        l = list(range(1, n_muon + 1))
        combinations = [list(zip(x, l)) for x in permutations(l, len(l))]
        combinations = [i for comb in combinations for i in comb]
        combinations = [e for e in combinations if e[1] >= e[0]]
        for combination in combinations:
            fake_plot_configs[combination] = self.retrieve_single_fake_plot_config(plot_config, *combination)
        return fake_plot_configs

    @staticmethod
    def retrieve_single_fake_plot_config(plot_config, n_fake_muon, n_total_muon):
        pc = deepcopy(plot_config)
        pc.name = "fake_single_lep"
        cut_name = "Sum$(muon_isolFixedCutTight == 1 && muon_is_prompt == 1 && abs(muon_d0sig) < 3)" #""Sum$(muon_is_num == 1 && muon_is_prompt_fix == 1)" #"Sum$(muon_isolFixedCutLoose == 0)"

        good_muon = "muon_isolFixedCutTight == 1 && muon_is_prompt == 1 && abs(muon_d0sig) < 3"
        bad_muon = "muon_isolFixedCutLoose == 0 && abs(muon_d0sig) < 3"
        muon_selector = "Sum$({:s})".format(good_muon)
        cut = filter(lambda cut: cut_name in cut, pc.cuts)[0]
        cut_index = pc.cuts.index(cut)
        cut = cut.replace("{:s} == muon_n".format(muon_selector),
                          "{:s} == (muon_n - {:d})".format(muon_selector, n_fake_muon))
        cut = cut.replace("muon_n == {:d}".format(n_total_muon),
                          "muon_n == {:d} && Sum$({:s}) == {:d}".format(n_total_muon, bad_muon, n_fake_muon))
        pc.cuts[cut_index] = cut
        pc.weight += " * muon_fake_factor_20171117"
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

    def retrieve_fake_factor(self, pt, eta, is_denom, n_jets):
        if not is_denom or pt > 50000.:
            return 1.
        if n_jets > 2:
            n_jets = 2
        b = self.fake_factor[n_jets].FindBin(pt / 1000., eta)
        return self.fake_factor[n_jets].GetBinContent(b)


class MuonFakeDecorator(object):
    def __init__(self, **kwargs):
        input_files = [fn for fn in kwargs["input_files"] if "data" in fn]
        self.input_file_handles = [FileHandle(file_name=input_file, open_option="UPDATE",
                                              run_dir=kwargs["run_dir"]) for input_file in input_files]
        self.estimator = MuonFakeProvider(**kwargs)
        self.tree_name = kwargs["tree_name"]
        self.fake_factors = ROOT.std.vector('float')()
        self.branch_name = kwargs["branch_name"]
        self.branch_name_total = "{:s}_total".format(kwargs["branch_name"])
        self.tree = None
        self.branch = None
        self.branch_total = None
        self.total_sf = array('f', [1.])

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
        self.total_sf[0] = 1.
        for n_muon in range(self.tree.muon_n):
            #muon_isolFixedCutLoose == 0 & & muon_is_prompt == 1 & & abs(muon_d0sig) > 3 & & mc_weight >= 0
            fake_factor = self.estimator.retrieve_fake_factor(self.tree.muon_pt[n_muon],
                                                              self.tree.muon_eta[n_muon],
                                                              self.tree.muon_isolFixedCutLoose[n_muon] == 0,
                                                              0)
                                                              #self.tree.muon_n_jet_dr2[n_muon])
            self.total_sf[0] *= fake_factor
            self.fake_factors.push_back(fake_factor)
                                                                            #get_n_jets_dr(n_muon, 0.3)))

    def event_loop(self):
        total_entries = self.tree.GetEntries()
        for entry in range(total_entries):
            _logger.debug("Process event {:d}".format(entry))
            self.tree.GetEntry(entry)
            self.decorate_event()
            self.branch.Fill()
            self.branch_total.Fill()

    def dump(self, file_handle):
        tdir = file_handle.get_directory("Nominal")
        tdir.cd()
        self.tree.Write()
        file_handle.close()

    def initialise(self, file_handle):
        self.tree = file_handle.get_objects_by_pattern(self.tree_name, "Nominal")[0]
        self.branch = self.tree.Branch(self.branch_name, self.fake_factors)
        self.branch_total = self.tree.Branch(self.branch_name_total,
                                             self.total_sf,
                                             "{:s}/F".format(self.branch_name_total))

    def execute(self):
        for file_handle in self.input_file_handles:
            try:
                self.initialise(file_handle)
                self.event_loop()
                self.dump(file_handle)
            except Exception as e:
                print(traceback.format_exc())
                raise e
            finally:
                file_handle.close()


class ElectronFakeProvider(object):
    def __init__(self, **kwargs):
        self.file_handle = FileHandle(file_name=kwargs["fake_factor_file"])
        self.fake_factor = {}
        self.read_fake_factors()

    def read_fake_factors(self):
        for i in range(3):
            name = "fake_factor_pt_eta_geq{:d}_jets_dR0.3".format(i)
            canvas_fake_factor = self.file_handle.get_object_by_name(name)
            self.fake_factor[i] = get_objects_from_canvas_by_name(canvas_fake_factor, name)[0]

    def retrieve_fake_factor(self, pt, eta, is_denom, n_jets):
        if not is_denom or pt > 50000.:
            return 1.
        if n_jets > 2:
            n_jets = 2
        b = self.fake_factor[n_jets].FindBin(pt / 1000., eta)
        return self.fake_factor[n_jets].GetBinContent(b)


class ElectronFakeDecorator(object):
    """
    Apply electron fake factors
    """
    def __init__(self, **kwargs):
        input_files = [fn for fn in kwargs["input_files"] if "data" in fn]
        self.input_file_handles = [FileHandle(file_name=input_file, open_option="UPDATE",
                                              run_dir=kwargs["run_dir"]) for input_file in input_files]
        self.estimator = ElectronFakeProvider(**kwargs)
        self.tree_name = kwargs["tree_name"]
        self.fake_factors = ROOT.std.vector('float')()
        self.branch_name = kwargs["branch_name"]
        self.branch_name_total = "{:s}_total".format(kwargs["branch_name"])
        self.tree = None
        self.branch = None
        self.branch_total = None
        self.total_sf = array('f', [1.])

    def decorate_event(self):
        self.fake_factors.clear()
        self.total_sf[0] = 1.
        for n_electron in range(self.tree.electron_n):
            #muon_isolFixedCutLoose == 0 & & muon_is_prompt == 1 & & abs(muon_d0sig) > 3 & & mc_weight >= 0
            fake_factor = self.estimator.retrieve_fake_factor(self.tree.muon_pt[n_muon],
                                                              self.tree.muon_eta[n_muon],
                                                              self.tree.muon_isolFixedCutLoose[n_muon] == 0,
                                                              0)
                                                              #self.tree.muon_n_jet_dr2[n_muon])
            self.total_sf[0] *= fake_factor
            self.fake_factors.push_back(fake_factor)
                                                                            #get_n_jets_dr(n_muon, 0.3)))

    def event_loop(self):
        total_entries = self.tree.GetEntries()
        for entry in range(total_entries):
            _logger.debug("Process event {:d}".format(entry))
            self.tree.GetEntry(entry)
            self.decorate_event()
            self.branch.Fill()
            self.branch_total.Fill()

    def dump(self, file_handle):
        tdir = file_handle.get_directory("Nominal")
        tdir.cd()
        self.tree.Write()
        file_handle.close()

    def initialise(self, file_handle):
        self.tree = file_handle.get_objects_by_pattern(self.tree_name, "Nominal")[0]
        self.branch = self.tree.Branch(self.branch_name, self.fake_factors)
        self.branch_total = self.tree.Branch(self.branch_name_total,
                                             self.total_sf,
                                             "{:s}/F".format(self.branch_name_total))

    def execute(self):
        for file_handle in self.input_file_handles:
            try:
                self.initialise(file_handle)
                self.event_loop()
                self.dump(file_handle)
            except Exception as e:
                print(traceback.format_exc())
                raise e
            finally:
                file_handle.close()


class ElectronFakeEstimator(object):
    def __init__(self, plotter_instance, **kwargs):
        kwargs.setdefault("sample_name", "Fakes")
        self.plotter = plotter_instance
        self.file_handles = [fh for fh in kwargs["file_handles"] if fh.process.is_data]
        self.sample_name = kwargs["sample_name"]
        self.type = "DataProvider"

    def execute(self, plot_config):
        return "Fakes", self.get_fake_background(plot_config)

    def get_fake_background(self, plot_config):
        def rebuild_dict_structure():
            for key, data in list(fake_histograms.items()):
                data = [hist_set for hist_set in data if all(hist_set)]
                hist_data = {}
                for plot_config, process, hist in data:
                    if hist is None:
                        _logger.warning("hist for process {:s} is None".format(process))
                        continue
                    try:
                        if process not in list(hist_data[plot_config].keys()):
                            hist_data[plot_config][process] = hist
                        else:
                            hist_data[plot_config][process].Add(hist)
                    except KeyError:
                        hist_data[plot_config] = {process: hist}
                print(data)
                #hist_data = {k: v for k, v in data[0][1]}
                fake_histograms[key] = hist_data

        fake_plot_configs = self.retrieve_fake_plot_configs(plot_config)
        if len(fake_plot_configs) == 0:
            return None
        fake_histograms = dict()

        for key, plot_config in list(fake_plot_configs.items()):
            fake_histograms[key] = self.plotter.read_histograms(self.file_handles, [plot_config], systematic="Nominal")
        rebuild_dict_structure()
        for key, data in list(fake_histograms.items()):
            self.plotter.apply_lumi_weights(data)
            #print self.plotter.process_configs
            #print hists
            #self.plotter.merge(hists, self.plotter.process_configs)
            for plot_config, hists in list(data.items()):
                fake_histograms[key][plot_config] = self.merge(hists, self.plotter.process_configs)
        #fake_histograms = self.merge(fake_histograms)
        fake_hist = self.build_fake_contribution(fake_histograms)
        fake_hist.SetName("{:s}_{:s}".format(plot_config.name, self.sample_name))
        return fake_hist

    @staticmethod
    def build_fake_contribution(fake_histograms):
        fake_contribution = None
        single_lepton_fake_contribution = dict([k_v for k_v in iter(list(fake_histograms.items())) if k_v[0][0] == 1])
        for key, data in list(single_lepton_fake_contribution.items()):
            for pc, hist in list(data.items()):
                if fake_contribution is None:
                    fake_contribution = list(hist.values())[0]
        # print fake_contribution
        # exit()
        return fake_contribution

        # single_lepton_fake_contribution = filter(lambda (k, v): k[0] == 1, fake_histograms.iteritems())
        # #print single_lepton_fake_contribution
        # #print single_lepton_fake_contribution[1]
        # fake_contribution = single_lepton_fake_contribution.values()
        # # print single_lepton_fake_contribution
        # # while len(single_lepton_fake_contribution) > 0:
        # #     fake_contribution.Add(single_lepton_fake_contribution[1].pop()[-1])
        # di_lepton_fake_contribution = filter(lambda (k, v): k[0] == 2, fake_histograms.iteritems())
        # while len(di_lepton_fake_contribution) > 0:
        #     fake_contribution.Add(di_lepton_fake_contribution[0][1]..values(), -1)
        # tri_lepton_fake_contribution = filter(lambda (k, v): k[0] == 3, fake_histograms.iteritems())
        # while len(tri_lepton_fake_contribution) > 0:
        #     fake_contribution.Add(tri_lepton_fake_contribution.pop()[-1])
        # return fake_contribution

    # def merge(self, fake_histograms):
    #     for key, histograms in fake_histograms.iteritems():
    #         hist = histograms.pop(histograms.keys()[0])
    #         for process in histograms.keys():
    #             hist.Add(histograms.pop(process))
    #         fake_histograms[key] = hist
    #     return fake_histograms

    def merge(self, histograms, process_configs):
        for process, process_config in list(process_configs.items()):
            if not hasattr(process_config, "subprocesses"):
                continue
            print(process_config)
            for sub_process in process_config.subprocesses:
                if sub_process not in list(histograms.keys()):
                    continue
                if process not in list(histograms.keys()):
                    new_hist_name = histograms[sub_process].GetName().replace(sub_process, process)
                    histograms[process] = histograms[sub_process].Clone(new_hist_name)
                else:
                    histograms[process].Add(histograms[sub_process])
                histograms.pop(sub_process)
        return histograms

    def retrieve_fake_plot_configs(self, plot_config):
        n_electron = plot_config.region.n_electron
        fake_plot_configs = dict()
        l = list(range(1, n_electron + 1))
        combinations = [list(zip(x, [n_electron])) for x in permutations(l, len(l))]
        combinations = [i for comb in combinations for i in comb]
        combinations = [e for e in combinations if e[1] >= e[0]]
        combinations = [e for e in combinations if e[1] >= e[0]]
        for combination in combinations:
            fake_plot_configs[combination] = self.retrieve_single_fake_plot_config(plot_config, *combination)
        return fake_plot_configs

    @staticmethod
    def retrieve_single_fake_plot_config(plot_config, n_fake_electron, n_total_electron):
        pc = deepcopy(plot_config)
        pc.name = "fake_single_lep"
        cut_name = "Sum$(electron_is_num==1)"

        good_electron = "electron_is_num==1"
        bad_electron = "electron_is_denom==1"
        electron_selector = "Sum$({:s})".format(good_electron)
        cut = filter(lambda cut: cut_name in cut, pc.cuts)[0]
        cut_index = pc.cuts.index(cut)
        cut = cut.replace("{:s} == electron_n".format(electron_selector),
                          "{:s} == (electron_n - {:d})".format(electron_selector, n_fake_electron))
        cut = cut.replace("electron_n == {:d}".format(n_total_electron),
                          "electron_n == {:d} && Sum$({:s}) == {:d}".format(n_total_electron, bad_electron, n_fake_electron))
        pc.cuts[cut_index] = cut
        pc.weight += " * 0.5"
        return pc
