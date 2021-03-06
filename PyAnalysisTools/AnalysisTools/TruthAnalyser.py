from __future__ import print_function

import traceback
from copy import copy
from math import sqrt, pi

from builtins import filter
from builtins import object
from builtins import range
from builtins import str

import PyAnalysisTools.PlottingUtils.Formatting as FT
import ROOT
from PyAnalysisTools.PlottingUtils import PlottingTools as PT
from PyAnalysisTools.PlottingUtils.PlotConfig import PlotConfig
from PyAnalysisTools.base.OutputHandle import OutputFileHandle
from PyAnalysisTools.base.YAMLHandle import YAMLLoader

particle_map = dict()
particle_map["Ds-"] = [-431, "D_{s}^{-}"]
particle_map["Ds+"] = [431, "D_{s}^{+}"]
particle_map["D-"] = [-411, "D^{-}"]
particle_map["D+"] = [411, "D^{+}"]
particle_map["mu-"] = [13, "#mu^{-}"]
particle_map["mu+"] = [-13, "#mu^{+}"]
particle_map["phi1020"] = [333, "#phi"]
particle_map["pi0"] = [111, "#pi^{0}"]
particle_map["anti_mu_nu"] = [14, "#nu_{#mu}"]
particle_map["mu_nu"] = [-14, "#nu_{#mu}"]
particle_map["e_nu"] = [-12, "#nu_{e}"]
particle_map["gamma"] = [22, "#gamma"]
particle_map["rho0"] = [113, "#rho"]
particle_map["eta"] = [221, "#eta"]
particle_map["eta'"] = [331, "#eta'"]
particle_map["omega"] = [223, "#omega"]
particle_map["LQ"] = [1102, "LQ"]
particle_map["e-"] = [11, "e^{-}"]
particle_map["e+"] = [-11, "e^{+}"]
particle_map["q"] = [list(range(1, 6)), "q"]
particle_map["d"] = [1, "d"]
particle_map["s"] = [3, "s"]
particle_map["b"] = [5, "b"]
particle_map["tau-"] = [15, "tau^{-}"]
particle_map["tau+"] = [-15, "tau^{+}"]
particle_map["anti_tau_nu"] = [16, "#nu_{#tau}"]
particle_map["tau_nu"] = [-16, "#nu_{#tau}"]
particle_map["Bc+"] = [521, "B_{c}^{+}"]
particle_map["B_c+"] = [541, "B_{c}^{+}"]
particle_map["JPsi"] = [443, "J/#Psi"]

particle_map["B_s0"] = [531, "B_{s}^{0}"]
particle_map["B0"] = [511, "B^{0}"]
particle_map["B+"] = [521, "B^{+}"]
particle_map["anti-Lambda_b0"] = [-5122, "#Lambda_{b}^{0}"]

particle_map["anti-D*0"] = [-10421, "#bar{D}^{*,0}"]
particle_map["anti-D0"] = [-421, "#bar{D}^{0}"]
particle_map["anti-D_10"] = [-10423, "D_{1}^{*}"]
particle_map["anti-D_0*0"] = [-10421, "D_{0}^{*,0}(2400)"]
particle_map["anti-D'_10"] = [-20423, "D_{1}^{',0}(2400)"]
particle_map["anti-D_2*0"] = [-425, "D_{2}^{*,0}"]
# particle_map["anti-D*0"] = [-10411, "D_{0}^{*,-}"] #?
particle_map["D*-"] = [-413, "D^{*,-}(2010)"]
particle_map["D_1-"] = [-10413, "D_{1}^{-}"]
particle_map["D_0*-"] = [10411, "D_{0}^{*,-}"]
particle_map["D'_1-"] = [20413, "D\'_{1}^{-}"]
particle_map["D_2*-"] = [-415, "D_{2}^{*,-}"]
particle_map["anti-Lambda_c-"] = [-4122, "#Lambda_{c}^{-}"]
particle_map["anti-Lambda_c(2593)-"] = [-14122, "#Lambda_{c}^{-}(2593)"]
particle_map["anti-Lambda_c(2625)-"] = [-4124, "#Lambda_{c}^{-}(2625)"]
particle_map["D_s-"] = [-431, "D_{s}^{-}"]
particle_map["D_s*-"] = [-433, "D_{s}^{*}"]
particle_map["D_s1-"] = [-10433, "D_{s1}^{-} (2536)"]
particle_map["D_s0*-"] = [-10431, "D_{s0}^{*} (2317)"]
particle_map["D'_s1-"] = [-20433, "D_{s1}^{-} (2460)"]
particle_map["D_s2*-"] = [-435, "D_{s2}^{*} (2573)"]


class Process(object):
    def __init__(self, config):
        self.decay1_str = config["decay1"]
        self.decay2 = config["decay2"]
        self.decay1_pdgid = [particle_map[name][0] if isinstance(name, str) else name for name in self.decay1_str]
        self.decay2_pdgid = [[particle_map[name][0] if isinstance(name, str) else name for name in sub]
                             for sub in self.decay2]
        self.decay_2_initial_resonance = [decay[0] for decay in self.decay2_pdgid]
        self.decay2_sorted = [sorted(i[:-1], reverse=True) for i in self.decay2_pdgid]

    def get_bin_labels(self):
        labels = ["undefined"]
        for decay in self.decay2:
            labels.append(particle_map[decay[0]][1] + "#rightarrow" + "+".join([particle_map[decay[i]][1] for i in
                                                                                range(1, len(decay) - 1)]))
        return labels


class TruthAnalyer(object):
    def __init__(self, **kwargs):
        self.input_files = kwargs["input_files"]
        self.output_handle = OutputFileHandle(output_dir=kwargs["output_dir"])
        self.histograms = dict()
        self.plot_configs = dict()
        self.references = dict()
        self.tree_name = "CollectionTree"
        process_configs = YAMLLoader.read_yaml(kwargs["config_file"])
        self.processes = {int(channel): Process(process_config) for channel, process_config in
                          list(process_configs.items())}
        self.current_process_config = None
        self.setup()
        self.book_plot_configs()
        self.build_references()

    @staticmethod
    def setup():
        ROOT.gROOT.Macro('$ROOTCOREDIR/scripts/load_packages.C')
        ROOT.xAOD.Init().ignore()

    def book_histograms(self, process_id):
        def book_histogram(name, n_bins, x_min, x_max):
            if process_id not in self.histograms:
                self.histograms[process_id] = dict()
            self.histograms[process_id][name] = ROOT.TH1F("{:s}_{:d}".format(name, process_id), "", n_bins, x_min,
                                                          x_max)
            ROOT.SetOwnership(self.histograms[process_id][name], False)
            self.histograms[process_id][name].SetDirectory(0)

        book_histogram("resonance_counter_decay1", 2, -0.5, 1.5)
        book_histogram("resonance_decay1_child_pdg_ids", 501, -500.5, 500.5)
        book_histogram("decay2_mode", 4, -1.5, 2.5)
        book_histogram("decay1_lepton_e", 20, 0., 50.)
        book_histogram("muon_e", 20, 0., 50.)
        book_histogram("muon_eta", 50, -2.5, 2.5)
        book_histogram("muon_phi", 50, -3.2, 3.2)
        book_histogram("lead_muon_e", 20, 0., 20.)
        book_histogram("lead_muon_eta", 50, -2.5, 2.5)
        book_histogram("lead_muon_phi", 50, -3.2, 3.2)
        book_histogram("sub_lead_muon_eta", 50, -2.5, 2.5)
        book_histogram("sub_lead_muon_phi", 50, -3.2, 3.2)
        book_histogram("sub_lead_muon_e", 20, 0., 20.)
        book_histogram("third_lead_muon_e", 20, 0., 20.)
        book_histogram("third_lead_muon_eta", 50, -2.5, 2.5)
        book_histogram("third_lead_muon_phi", 50, -3.2, 3.2)
        book_histogram("gamma_e", 20, 0., 50.)
        book_histogram("decay2_particle", 999, -499.5, 499.5)
        book_histogram("decay2_particle_eta", 50, -2.5, 2.5)
        book_histogram("decay2_particle_phi", 50, -3.2, 3.2)
        book_histogram("decay2_particle_status", 10, -0.5, 9.5)
        book_histogram("decay2_non_muon_particle", 999, -499.5, 499.5)
        book_histogram("decay2_child_pdgid", 999, -499.5, 499.5)
        book_histogram("unidentified_pdgid", 999, -499.5, 499.5)
        book_histogram("unidentified_gamma_e", 100, 0., 1.)
        book_histogram("decay2_muon_e", 100, 0., 20.)
        book_histogram("decay2_gamma_e", 100, 0., 20.)
        book_histogram("decay2_gamma_eta", 50, -2.5, 2.5)
        book_histogram("decay2_gamma_phi", 50, -3.2, 3.2)
        book_histogram("decay2_gamma_e_low_pt", 100, 0., 1.)
        book_histogram("decay2_gamma_status", 100, -0.5, 99.5)
        book_histogram("decay2_gamma_e_after_veto", 100, 0., 20.)
        book_histogram("decay2_gamma_status_after_veto", 100, -0.5, 99.5)
        book_histogram("decay2_gamma_e_after_veto_low_pt", 100, 0., 1.)
        book_histogram("decay2_gamma_mu_dr", 30, 0., 0.4)
        book_histogram("decay2_gamma_mu_dr_after_veto", 30, 0., 0.4)
        book_histogram("control_photon_mother_pdgid", 501, -499.5, 499.5)

    def book_plot_configs(self):
        def book_plot_config(name, xtitle, **kwargs):
            pc = PlotConfig(dist=None, name=name, xtitle=xtitle, ytitle="Entries", watermark="Simulation Internal",
                            color=ROOT.kBlue, lumi=-1)
            for args, val in list(kwargs.items()):
                setattr(pc, name, val)
            self.plot_configs[name] = pc

        book_plot_config("resonance_counter_decay1", "Number of 1st decay particles")
        book_plot_config("resonance_decay1_child_pdg_ids", "pdg ID")
        book_plot_config("decay1_lepton_e", "#tau E [GeV]")
        book_plot_config("muon_e", "all #mu E [GeV]")
        book_plot_config("lead_muon_e", "lead. #mu E [GeV]")
        book_plot_config("lead_muon_eta", "lead. #mu #eta")
        book_plot_config("lead_muon_phi", "lead. #mu #phi")
        book_plot_config("sub_lead_muon_e", "sub-lead. #mu E [GeV]")
        book_plot_config("sub_lead_muon_eta", "sub-lead. #mu #eta")
        book_plot_config("sub_lead_muon_phi", "sub-lead. #mu #phi")
        book_plot_config("third_lead_muon_e", "third-lead. #mu E [GeV]")
        book_plot_config("third_lead_muon_eta", "third-lead. #mu #eta")
        book_plot_config("third_lead_muon_phi", "third-lead. #mu #phi")
        book_plot_config("gamma_e", "#gamma E [GeV]")
        book_plot_config("decay2_particle", "2^{nd} decay particle pdg ID")
        book_plot_config("decay2_particle_eta", "2^{nd} decay particle #eta")
        book_plot_config("decay2_particle_phi", "2^{nd} decay particle #phi")
        book_plot_config("decay2_particle_status", "2^{nd} decay particle status")
        book_plot_config("decay2_non_muon_particle", "non #mu pdg ID")
        book_plot_config("decay2_child_pdgid", "decay2 child pdg IDs")
        book_plot_config("unidentified_pdgid", "unidentified pdg IDs")
        book_plot_config("unidentified_gamma_e", "unidentified #gamma E [GeV]")
        book_plot_config("decay2_muon_e", "decay 2 #mu E [GeV]")
        book_plot_config("muon_eta", "#mu #eta")
        book_plot_config("muon_phi", "#mu #phi")
        book_plot_config("decay2_gamma_e", "decay 2 #gamma E [GeV]")
        book_plot_config("decay2_gamma_e_after_veto", "decay 2 #gamma E [GeV] (after E cut)")
        book_plot_config("decay2_gamma_eta", "decay 2 #gamma #eta")
        book_plot_config("decay2_gamma_phi", "decay 2 #gamma #phi")
        book_plot_config("decay2_gamma_e_low_pt", "decay 2 #gamma E [GeV]")
        book_plot_config("decay2_gamma_e_after_veto_low_pt", "decay 2 #gamma E [GeV] (after E cut)")
        book_plot_config("control_photon_mother_pdgid", "#gamma mother pdgID")
        book_plot_config("decay2_gamma_status", "decay 2 #gamma status")
        book_plot_config("decay2_gamma_mu_dr", "decay 2 #DeltaR(#gamma, #mu)")
        book_plot_config("decay2_gamma_mu_dr_after_veto", "decay 2 #DeltaR(#gamma, #mu) (after E cut)")
        book_plot_config("decay2_gamma_status_after_veto", "decay 2 #gamma status (after E cut)")
        book_plot_config("decay2_mode", "decay mode", ymax=1.1)

    def run(self):
        for input_file in self.input_files:
            self.analyse_file(input_file)
        self.plot_histograms()
        self.output_handle.write_and_close()

    def build_references(self):
        def find_br(process, mode):
            decay2_mode = filter(lambda decay: sorted(decay[:-1], reverse=True) == mode, process.decay2_pdgid)[0]
            return decay2_mode[-1]

        for process_id, process in list(self.processes.items()):
            reference_hist = ROOT.TH1F("decay_mode_reference_{:d}".format(process_id), "", 4, -1.5, 2.5)
            reference_hist.SetBinContent(1, 0)
            for mode_index in range(len(process.decay2_sorted)):
                br = find_br(process, process.decay2_sorted[mode_index])
                reference_hist.SetBinContent(mode_index + 2, br)
            self.references[process_id] = reference_hist

    def plot_histograms(self):
        for process_id, histograms in list(self.histograms.items()):
            for hist_name, hist in list(histograms.items()):
                pc = self.plot_configs[hist_name]
                if hist_name == "decay2_mode":
                    pc.axis_labels = self.processes[process_id].get_bin_labels()
                    pc.normalise = True
                canvas = PT.plot_obj(hist, pc)
                if hist_name == "decay2_mode":
                    pc_ref = copy(pc)
                    pc_ref.color = ROOT.kRed
                    PT.add_histogram_to_canvas(canvas, self.references[process_id], pc_ref)
                FT.decorate_canvas(canvas, pc)
                self.output_handle.register_object(canvas, str(process_id))

    @staticmethod
    def deltaR(particle1, particle2):
        def shift_phi(phi):
            if phi >= pi:
                return phi - 2. * pi
            if phi < -1. * pi:
                return phi + 2. * pi
            return phi

        phi1 = shift_phi(particle1[2])
        phi2 = shift_phi(particle2[2])
        return sqrt(pow(particle1[1] - particle2[1], 2) + pow(phi1 - phi2, 2))

    def analyse_file(self, input_file):
        f = ROOT.TFile.Open(input_file)
        tree = ROOT.xAOD.MakeTransientTree(f, self.tree_name)
        self.current_process_config = None
        for entry in range(tree.GetEntries()):
            tree.GetEntry(entry)
            process_id = tree.EventInfo.runNumber()
            if process_id not in self.histograms:
                self.book_histograms(process_id)
            if self.current_process_config is None:
                self.current_process_config = self.processes[process_id]
            truth_particles = tree.TruthParticles
            resonance_decay1 = [p for p in truth_particles if p.pdgId() == self.current_process_config.decay1_pdgid[0]]
            self.histograms[process_id]["resonance_counter_decay1"].Fill(len(resonance_decay1))
            self.histograms[process_id]["decay1_lepton_e"].Fill(resonance_decay1[0].e() / 1000.)

            if len(resonance_decay1) == 0:
                print("Suspicious event. Could not find ", self.current_process_config.decay1_pdgid[0], " for process ",
                      process_id)
                continue
            resonance1_vertex = resonance_decay1[0].decayVtxLink().outgoingParticleLinks()
            muon_pts = list()
            photons_pts = list()
            all_photons_pts = list()
            try:
                if -13 in self.current_process_config.decay1_pdgid:
                    muon_pts.append(
                        [filter(lambda particle: abs(particle.pdgId()) == 13, resonance1_vertex)[0].e() / 1000.,
                         filter(lambda particle: abs(particle.pdgId()) == 13, resonance1_vertex)[0].eta(),
                         filter(lambda particle: abs(particle.pdgId()) == 13, resonance1_vertex)[0].phi()])
            except IndexError:
                print("Could not find any muon for first resonance decay in process", process_id)
                continue
            mode = list()
            for resonance1_child in resonance1_vertex:
                self.histograms[process_id]["resonance_decay1_child_pdg_ids"].Fill(resonance1_child.pdgId())
                if abs(resonance1_child.pdgId() == 13):
                    self.histograms[process_id]["muon_e"].Fill(resonance1_child.e() / 1000.)
                    self.histograms[process_id]["muon_eta"].Fill(resonance1_child.eta())
                    self.histograms[process_id]["muon_phi"].Fill(resonance1_child.phi())
                if resonance1_child.pdgId() not in self.current_process_config.decay_2_initial_resonance:
                    continue
                mode.append((resonance1_child.pdgId(), resonance1_child.e() / 1000.))
                resonance2_vertex = resonance1_child.decayVtxLink().outgoingParticleLinks()
                for resonance2_child in resonance2_vertex:
                    if abs(resonance2_child.pdgId()) == 13:
                        self.histograms[process_id]["decay2_muon_e"].Fill(resonance2_child.e() / 1000.)
                        self.histograms[process_id]["muon_e"].Fill(resonance2_child.e() / 1000.)
                        self.histograms[process_id]["muon_eta"].Fill(resonance2_child.eta())
                        self.histograms[process_id]["muon_phi"].Fill(resonance2_child.phi())
                        muon_pts.append([resonance2_child.e() / 1000., resonance2_child.eta(), resonance2_child.phi()])
                    if abs(resonance2_child.pdgId()) == 22:
                        self.histograms[process_id]["decay2_gamma_e"].Fill(resonance2_child.e() / 1000.)
                        self.histograms[process_id]["decay2_gamma_e_low_pt"].Fill(resonance2_child.e() / 1000.)
                        self.histograms[process_id]["decay2_gamma_status"].Fill(resonance2_child.status())
                    if not abs(resonance2_child.pdgId()) == 13:
                        self.histograms[process_id]["decay2_non_muon_particle"].Fill(resonance2_child.pdgId())
                    if resonance2_child.pdgId() == 22:
                        all_photons_pts.append([resonance2_child.e() / 1000., resonance2_child.eta(),
                                                resonance2_child.phi()])
                    if resonance2_child.pdgId() == 22 and resonance2_child.e() < 100.:
                        continue
                    # if resonance2_child.pdgId() == 22 and self.deltaR(muon_pts[0], [resonance2_child.e() / 1000.,
                    #                                                                 resonance2_child.eta(),
                    #                                                                 resonance2_child.phi()]) > 0.4:
                    #     continue
                    mode.append((resonance2_child.pdgId(), resonance2_child.e() / 1000.))
                    self.histograms[process_id]["decay2_particle"].Fill(resonance2_child.pdgId())
                    self.histograms[process_id]["decay2_particle_eta"].Fill(resonance2_child.eta())
                    self.histograms[process_id]["decay2_particle_phi"].Fill(resonance2_child.phi())
                    self.histograms[process_id]["decay2_particle_status"].Fill(resonance2_child.status())
                    if abs(resonance2_child.pdgId()) == 22:
                        self.histograms[process_id]["decay2_gamma_e_after_veto"].Fill(resonance2_child.e() / 1000.)
                        self.histograms[process_id]["decay2_gamma_eta"].Fill(resonance2_child.eta())
                        self.histograms[process_id]["decay2_gamma_phi"].Fill(resonance2_child.phi())
                        self.histograms[process_id]["decay2_gamma_e_after_veto_low_pt"].Fill(
                            resonance2_child.e() / 1000.)
                        self.histograms[process_id]["decay2_gamma_status_after_veto"].Fill(resonance2_child.status())
                        mother = resonance2_child.prodVtx().incomingParticleLinks()[0]
                        self.histograms[process_id]["control_photon_mother_pdgid"].Fill(mother.pdgId())
                        photons_pts.append([resonance2_child.e() / 1000., resonance2_child.eta(),
                                            resonance2_child.phi()])
            mode.sort(key=lambda i: i[0], reverse=True)
            try:
                decay_mode = self.current_process_config.decay2_sorted.index([i[0] for i in mode])
            except ValueError:
                decay_mode = -1
            if decay_mode == -1:
                for pdg_id, pt in mode:
                    self.histograms[process_id]["unidentified_pdgid"].Fill(pdg_id)
                    if pdg_id == 22:
                        self.histograms[process_id]["unidentified_gamma_e"].Fill(pt)

            self.histograms[process_id]["decay2_mode"].Fill(decay_mode)
            muon_pts.sort(key=lambda x: x[0], reverse=True)
            if len(muon_pts) < 3:
                print("did not find 3 muons!")
                continue
            self.histograms[process_id]["lead_muon_e"].Fill(muon_pts[0][0])
            self.histograms[process_id]["sub_lead_muon_e"].Fill(muon_pts[1][0])
            self.histograms[process_id]["third_lead_muon_e"].Fill(muon_pts[2][0])
            self.histograms[process_id]["lead_muon_eta"].Fill(muon_pts[0][1])
            self.histograms[process_id]["sub_lead_muon_eta"].Fill(muon_pts[1][1])
            self.histograms[process_id]["third_lead_muon_eta"].Fill(muon_pts[2][1])
            self.histograms[process_id]["lead_muon_phi"].Fill(muon_pts[0][2])
            self.histograms[process_id]["sub_lead_muon_phi"].Fill(muon_pts[1][2])
            self.histograms[process_id]["third_lead_muon_phi"].Fill(muon_pts[2][2])
            for photon in photons_pts:
                for muon in muon_pts:
                    self.histograms[process_id]["decay2_gamma_mu_dr_after_veto"].Fill(self.deltaR(photon, muon))
            for photon in all_photons_pts:
                for muon in muon_pts:
                    self.histograms[process_id]["decay2_gamma_mu_dr"].Fill(self.deltaR(photon, muon))
        f.Close()


class LQTruthAnalyser(object):
    def __init__(self, **kwargs):
        kwargs.setdefault('max_events', None)
        self.input_files = kwargs["input_files"]
        self.output_handle = OutputFileHandle(output_dir=kwargs["output_dir"])
        self.histograms = dict()
        self.plot_configs = dict()
        self.references = dict()
        self.tree_name = "CollectionTree"
        self.max_events = kwargs['max_events']
        process_configs = YAMLLoader.read_yaml(kwargs["config_file"])
        self.processes = {int(channel): Process(process_config) for channel, process_config in
                          list(process_configs.items())}
        self.current_process_config = None
        self.setup()
        self.book_plot_configs()
        # self.build_references()

    @staticmethod
    def setup():
        ROOT.gROOT.Macro('$ROOTCOREDIR/scripts/load_packages.C')
        ROOT.xAOD.Init().ignore()

    def book_histograms(self, process_id):
        def book_histogram(name, n_bins, x_min, x_max, labels=None):
            if process_id not in self.histograms:
                self.histograms[process_id] = dict()
            self.histograms[process_id][name] = ROOT.TH1F("{:s}_{:d}".format(name, process_id), "", n_bins, x_min,
                                                          x_max)
            ROOT.SetOwnership(self.histograms[process_id][name], False)
            self.histograms[process_id][name].SetDirectory(0)
            if labels is not None:
                for i, label in enumerate(labels):
                    self.histograms[process_id][name].GetXaxis().SetBinLabel(i + 1, label)

        # book_histogram("LQ_counter", 2, -0.5, 1.5)
        # book_histogram("decay2_mode", 4, -1.5, 2.5)
        book_histogram("lepton1_status", 150, -0.5, 149.5)
        book_histogram("lepton1_e", 100, 0., 2500.)
        book_histogram("lepton1_pt", 100, 0., 2500.)
        book_histogram("lepton1_eta", 50, -2.5, 2.5)
        book_histogram("lepton1_phi", 50, -3.2, 3.2)
        book_histogram("lepton1_e_resonant", 100, 0., 2500.)
        book_histogram("lepton1_pt_resonant", 100, 0., 2500.)
        book_histogram("lepton1_eta_resonant", 50, -2.5, 2.5)
        book_histogram("lepton1_phi_resonant", 50, -3.2, 3.2)
        book_histogram("lepton1_e_non_resonant", 100, 0., 2500.)
        book_histogram("lepton1_pt_non_resonant", 100, 0., 2500.)
        book_histogram("lepton1_eta_non_resonant", 50, -2.5, 2.5)
        book_histogram("lepton1_phi_non_resonant", 50, -3.2, 3.2)
        book_histogram("lepton2_status", 150, -0.5, 149.5)
        book_histogram("lepton2_e", 100, 0., 2500.)
        book_histogram("lepton2_pt", 100, 0., 2500.)
        book_histogram("lepton2_eta", 50, -2.5, 2.5)
        book_histogram("lepton2_phi", 50, -3.2, 3.2)
        book_histogram("lepton2_e_resonant", 100, 0., 2500.)
        book_histogram("lepton2_pt_resonant", 100, 0., 2500.)
        book_histogram("lepton2_eta_resonant", 50, -2.5, 2.5)
        book_histogram("lepton2_phi_resonant", 50, -3.2, 3.2)
        book_histogram("lepton2_e_non_resonant", 100, 0., 2500.)
        book_histogram("lepton2_pt_non_resonant", 100, 0., 2500.)
        book_histogram("lepton2_eta_non_resonant", 50, -2.5, 2.5)
        book_histogram("lepton2_phi_non_resonant", 50, -3.2, 3.2)
        book_histogram("quark_status", 150, -0.5, 149.5)
        book_histogram("quark_e", 100, 0., 2500.)
        book_histogram("quark_pt", 100, 0., 2500.)
        book_histogram("quark_eta", 50, -2.5, 2.5)
        book_histogram("quark_phi", 50, -3.2, 3.2)
        book_histogram("quark_pdgId", 10, -0.5, 9.5)
        book_histogram("quark_e_resonant", 100, 0., 2500.)
        book_histogram("quark_pt_resonant", 100, 0., 2500.)
        book_histogram("quark_eta_resonant", 50, -2.5, 2.5)
        book_histogram("quark_phi_resonant", 50, -3.2, 3.2)
        book_histogram("quark_e_non_resonant", 100, 0., 2500.)
        book_histogram("quark_pt_non_resonant", 100, 0., 2500.)
        book_histogram("quark_eta_non_resonant", 50, -2.5, 2.5)
        book_histogram("quark_phi_non_resonant", 50, -3.2, 3.2)
        book_histogram("lq_mass", 100, 500, 4500)
        book_histogram("inv_mass", 100, 500, 4500)
        book_histogram("inv_mass_min", 100, 500, 4500)
        book_histogram("inv_mass_max", 100, 500, 4500)
        book_histogram("inv_mass_schan", 100, 500, 4500)
        book_histogram("inv_mass_min_schan", 100, 500, 4500)
        book_histogram("inv_mass_max_schan", 100, 500, 4500)
        book_histogram("inv_mass_tchan", 100, 500, 4500)
        book_histogram("inv_mass_min_tchan", 100, 500, 4500)
        book_histogram("inv_mass_max_tchan", 100, 500, 4500)
        book_histogram("event_yields", 3, 0., 3., labels=['all', 'resonant', 'non-resonant'])

    def book_plot_configs(self):
        def book_plot_config(name, xtitle, **kwargs):
            pc = PlotConfig(dist=None, name=name, xtitle=xtitle, ytitle="Entries", watermark="Simulation Internal",
                            color=ROOT.kBlue, draw="HIST", no_fill=True, lumi=None)
            for args, val in list(kwargs.items()):
                setattr(pc, name, val)
            self.plot_configs[name] = pc

        book_plot_config("resonance_counter_decay1", "Number of events")
        book_plot_config("resonance_decay1_child_pdg_ids", "pdg ID")
        book_plot_config("lepton1_status", "lepton 1 status")
        book_plot_config("lepton1_e", "lepton 1 E [GeV]")
        book_plot_config("lepton1_pt", "lepton 1 p_{T} [GeV]")
        book_plot_config("lepton1_eta", "lepton1 #eta")
        book_plot_config("lepton1_phi", "lepton1 #phi")

        book_plot_config("lepton1_e_resonant", "lepton 1 E [GeV] (resonant)")
        book_plot_config("lepton1_pt_resonant", "lepton 1 p_{T} [GeV] (resonant)")
        book_plot_config("lepton1_eta_resonant", "lepton 1 #eta (resonant)")
        book_plot_config("lepton1_phi_resonant", "lepton 1 #phi (resonant)")

        book_plot_config("lepton1_e_non_resonant", "lepton 1 E [GeV] (non-resonant)")
        book_plot_config("lepton1_pt_non_resonant", "lepton 1 p_{T} [GeV] (non-resonant)")
        book_plot_config("lepton1_eta_non_resonant", "lepton 1 #eta (non-resonant)")
        book_plot_config("lepton1_phi_non_resonant", "lepton 1 #phi (non-resonant)")

        book_plot_config("lepton2_status", "lepton 2 status")
        book_plot_config("lepton2_e", "lepton 2 E [GeV]")
        book_plot_config("lepton2_pt", "lepton 2 p_{T} [GeV]")
        book_plot_config("lepton2_eta", "lepton 2 #eta")
        book_plot_config("lepton2_phi", "lepton 2 #phi")

        book_plot_config("lepton2_e_resonant", "lepton 2 E [GeV] (resonant)")
        book_plot_config("lepton2_pt_resonant", "lepton 2 p_{T} [GeV] (resonant)")
        book_plot_config("lepton2_eta_resonant", "lepton 2 #eta (resonant)")
        book_plot_config("lepton2_phi_resonant", "lepton 2 #phi (resonant)")

        book_plot_config("lepton2_e_non_resonant", "lepton 2 E [GeV] (non-resonant)")
        book_plot_config("lepton2_pt_non_resonant", "lepton 2 p_{T} [GeV] (non-resonant)")
        book_plot_config("lepton2_eta_non_resonant", "lepton 2 #eta (non-resonant)")
        book_plot_config("lepton2_phi_non_resonant", "lepton 2 #phi (non-resonant)")

        book_plot_config("quark_status", "quark status")
        book_plot_config("quark_e", "quark E [GeV]")
        book_plot_config("quark_pt", "quark p_{T} [GeV]")
        book_plot_config("quark_eta", "quark #eta")
        book_plot_config("quark_phi", "quark #phi")

        book_plot_config("quark_e_resonant", "quark E [GeV] (resonant)")
        book_plot_config("quark_pt_resonant", "quark p_{T} [GeV] (resonant)")
        book_plot_config("quark_eta_resonant", "quark #eta (resonant)")
        book_plot_config("quark_phi_resonant", "quark #phi (resonant)")

        book_plot_config("quark_e_non_resonant", "quark E [GeV] (non-resonant)")
        book_plot_config("quark_pt_non_resonant", "quark p_{T} [GeV] (non-resonant)")
        book_plot_config("quark_eta_non_resonant", "quark #eta (non-resonant)")
        book_plot_config("quark_phi_non_resonant", "quark #phi (non-resonant)")

        book_plot_config("quark_pdgId", "quark pdg ID")

        book_plot_config("inv_mass", "M_{lq} [GeV]")
        book_plot_config("inv_mass_min", "M_{lq}^{min} [GeV]")
        book_plot_config("inv_mass_max", "M_{lq}^{max} [GeV]")

        book_plot_config("inv_mass_schan", "M_{lq} [GeV] (resonant)")
        book_plot_config("inv_mass_min_schan", "M_{lq}^{min} [GeV] (resonant)")
        book_plot_config("inv_mass_max_schan", "M_{lq}^{max} [GeV] (resonant)")

        book_plot_config("lq_mass", "LQ truth mass [GeV]")

        book_plot_config("inv_mass_tchan", "M_{lq} [GeV] (non resonant)")
        book_plot_config("inv_mass_min_tchan", "M_{lq}^{min} [GeV] (non resonant)")
        book_plot_config("inv_mass_max_tchan", "M_{lq}^{max} [GeV] (non resonant)")
        book_plot_config("event_yields", "")

    def run(self):
        for input_file in self.input_files:
            self.analyse_file(input_file)
        self.plot_histograms()
        # exit(0)
        self.output_handle.write_and_close()

    def build_references(self):
        def find_br(process, mode):
            decay2_mode = filter(lambda decay: sorted(decay[:-1], reverse=True) == mode, process.decay2_pdgid)[0]
            return decay2_mode[-1]

        for process_id, process in list(self.processes.items()):
            reference_hist = ROOT.TH1F("decay_mode_reference_{:d}".format(process_id), "", 4, -1.5, 2.5)
            reference_hist.SetBinContent(1, 0)
            for mode_index in range(len(process.decay2_sorted)):
                br = find_br(process, process.decay2_sorted[mode_index])
                reference_hist.SetBinContent(mode_index + 2, br)
            self.references[process_id] = reference_hist

    def plot_histograms(self):
        for process_id, histograms in list(self.histograms.items()):
            for hist_name, hist in list(histograms.items()):
                pc = self.plot_configs[hist_name]
                if hist_name == "decay2_mode":
                    pc.axis_labels = self.processes[process_id].get_bin_labels()
                    pc.normalise = True
                canvas = PT.plot_obj(hist, pc)
                if hist_name == "decay2_mode":
                    pc_ref = copy(pc)
                    pc_ref.color = ROOT.kRed
                    PT.add_histogram_to_canvas(canvas, self.references[process_id], pc_ref)
                print(canvas, hist_name, hist, process_id)
                FT.decorate_canvas(canvas, pc)
                self.output_handle.register_object(canvas, str(process_id))

    @staticmethod
    def deltaR(particle1, particle2):
        def shift_phi(phi):
            if phi >= pi:
                return phi - 2. * pi
            if phi < -1. * pi:
                return phi + 2. * pi
            return phi

        phi1 = shift_phi(particle1[2])
        phi2 = shift_phi(particle2[2])
        return sqrt(pow(particle1[1] - particle2[1], 2) + pow(phi1 - phi2, 2))

    def analyse_file(self, input_file):
        def run_schannel():
            self.is_schan = True
            self.histograms[process_id]["event_yields"].Fill(1)
            resonance1_vertex = LQ[-1].decayVtxLink().outgoingParticleLinks()
            try:
                lepton_1 = filter(lambda particle: abs(particle.pdgId()) == 11 or abs(particle.pdgId()) == 13
                                                   or abs(particle.pdgId()) == 15, resonance1_vertex)[0]  # noqa:E127
                quark_1 = filter(lambda particle: abs(particle.pdgId()) in range(1, 6), resonance1_vertex)[0]
                self.histograms[process_id]["lepton1_e"].Fill(lepton_1.e() / 1000.)
                self.histograms[process_id]["lepton1_eta"].Fill(lepton_1.eta())
                self.histograms[process_id]["lepton1_phi"].Fill(lepton_1.phi())
                self.histograms[process_id]["lepton1_e_resonant"].Fill(lepton_1.e() / 1000.)
                self.histograms[process_id]["lepton1_eta_resonant"].Fill(lepton_1.eta())
                self.histograms[process_id]["lepton1_phi_resonant"].Fill(lepton_1.phi())
                self.histograms[process_id]["quark_e"].Fill(quark_1.e() / 1000.)
                self.histograms[process_id]["quark_eta"].Fill(quark_1.eta())
                self.histograms[process_id]["quark_phi"].Fill(quark_1.phi())
                self.histograms[process_id]["quark_e_resonant"].Fill(quark_1.e() / 1000.)
                self.histograms[process_id]["quark_eta_resonant"].Fill(quark_1.eta())
                self.histograms[process_id]["quark_phi_resonant"].Fill(quark_1.phi())

            except IndexError:
                print("Could not find any first lepton for first resonance decay in process", process_id)
                return

            try:
                lepton_2 = filter(lambda particle: particle.pdgId() == -1 * lepton_1.pdgId(), truth_particles)[0]
                self.histograms[process_id]["lepton2_e"].Fill(lepton_2.e() / 1000.)
                self.histograms[process_id]["lepton2_eta"].Fill(lepton_2.eta())
                self.histograms[process_id]["lepton2_phi"].Fill(lepton_2.phi())
                self.histograms[process_id]["lepton2_e_resonant"].Fill(lepton_2.e() / 1000.)
                self.histograms[process_id]["lepton2_eta_resonant"].Fill(lepton_2.eta())
                self.histograms[process_id]["lepton2_phi_resonant"].Fill(lepton_2.phi())
            except IndexError:
                print("Could not find any second lepton for first resonance decay in process", process_id)
                return
            except KeyError:
                print("Could not add process {:f}. Check if it is defined in process defintion.".format(process_id))
                return
            tlv_lepton_1 = ROOT.TLorentzVector()
            tlv_lepton_1.SetPxPyPzE(lepton_1.px(), lepton_1.py(), lepton_1.pz(), lepton_1.e())
            tlv_lepton_2 = ROOT.TLorentzVector()
            tlv_lepton_2.SetPxPyPzE(lepton_2.px(), lepton_2.py(), lepton_2.pz(), lepton_2.e())
            tlv_quark_1 = ROOT.TLorentzVector()
            tlv_quark_1.SetPxPyPzE(quark_1.px(), quark_1.py(), quark_1.pz(), quark_1.e())
            self.histograms[process_id]["lepton1_pt"].Fill(tlv_lepton_1.Pt() / 1000.)
            self.histograms[process_id]["lepton1_pt_resonant"].Fill(tlv_lepton_1.Pt() / 1000.)
            self.histograms[process_id]["lepton2_pt"].Fill(tlv_lepton_2.Pt() / 1000.)
            self.histograms[process_id]["lepton2_pt_resonant"].Fill(tlv_lepton_2.Pt() / 1000.)
            self.histograms[process_id]["quark_pt"].Fill(tlv_quark_1.Pt() / 1000.)
            self.histograms[process_id]["quark_pt_resonant"].Fill(tlv_quark_1.Pt() / 1000.)
            self.histograms[process_id]["inv_mass"].Fill((tlv_lepton_1 + tlv_quark_1).M() / 1000.)
            self.histograms[process_id]["lepton1_status"].Fill(lepton_1.status())
            self.histograms[process_id]["lepton2_status"].Fill(lepton_2.status())
            self.histograms[process_id]["quark_status"].Fill(quark_1.status())
            self.histograms[process_id]["quark_pdgId"].Fill(abs(quark_1.pdgId()))

            mass1 = (tlv_lepton_1 + tlv_quark_1).M() / 1000.
            mass2 = (tlv_lepton_2 + tlv_quark_1).M() / 1000.
            if mass1 > mass2:
                self.histograms[process_id]["inv_mass_max"].Fill(mass1)
                self.histograms[process_id]["inv_mass_min"].Fill(mass2)
                self.histograms[process_id]["inv_mass_max_schan"].Fill(mass1)
                self.histograms[process_id]["inv_mass_min_schan"].Fill(mass2)
            else:
                self.histograms[process_id]["inv_mass_max"].Fill(mass2)
                self.histograms[process_id]["inv_mass_min"].Fill(mass1)
                self.histograms[process_id]["inv_mass_max_schan"].Fill(mass2)
                self.histograms[process_id]["inv_mass_min_schan"].Fill(mass1)

        def run_tchannel():
            self.is_tchan = True
            self.histograms[process_id]["event_yields"].Fill(2)
            LQ_decay_particles = [p for p in truth_particles if p.status() == 23]
            prod_vtx = LQ_decay_particles[0].prodVtxLink().outgoingParticleLinks()

            leptons = [particle for particle in prod_vtx if abs(particle.pdgId()) == 11
                       or abs(particle.pdgId()) == 13 or abs(particle.pdgId()) == 15]
            if len(leptons) < 2:
                print("PROBLEM")
                return
            lepton_1 = leptons[0]
            lepton_2 = leptons[1]
            quark_1 = filter(lambda particle: abs(particle.pdgId()) in range(1, 6), prod_vtx)[0]
            self.histograms[process_id]["lepton1_e"].Fill(lepton_1.e() / 1000.)
            self.histograms[process_id]["lepton1_eta"].Fill(lepton_1.eta())
            self.histograms[process_id]["lepton1_phi"].Fill(lepton_1.phi())
            self.histograms[process_id]["quark_e"].Fill(quark_1.e() / 1000.)
            self.histograms[process_id]["quark_eta"].Fill(quark_1.eta())
            self.histograms[process_id]["quark_phi"].Fill(quark_1.phi())
            self.histograms[process_id]["lepton2_e"].Fill(lepton_2.e() / 1000.)
            self.histograms[process_id]["lepton2_eta"].Fill(lepton_2.eta())
            self.histograms[process_id]["lepton2_phi"].Fill(lepton_2.phi())
            self.histograms[process_id]["lepton1_status"].Fill(lepton_1.status())
            self.histograms[process_id]["lepton2_status"].Fill(lepton_2.status())
            self.histograms[process_id]["quark_status"].Fill(quark_1.status())
            self.histograms[process_id]["quark_pdgId"].Fill(abs(quark_1.pdgId()))
            self.histograms[process_id]["lepton1_e_non_resonant"].Fill(lepton_1.e() / 1000.)
            self.histograms[process_id]["lepton1_eta_non_resonant"].Fill(lepton_1.eta())
            self.histograms[process_id]["lepton1_phi_non_resonant"].Fill(lepton_1.phi())
            self.histograms[process_id]["lepton2_e_non_resonant"].Fill(lepton_2.e() / 1000.)
            self.histograms[process_id]["lepton2_eta_non_resonant"].Fill(lepton_2.eta())
            self.histograms[process_id]["lepton2_phi_non_resonant"].Fill(lepton_2.phi())
            self.histograms[process_id]["quark_e_non_resonant"].Fill(quark_1.e() / 1000.)
            self.histograms[process_id]["quark_eta_non_resonant"].Fill(quark_1.eta())
            self.histograms[process_id]["quark_phi_non_resonant"].Fill(quark_1.phi())
            tlv_lepton_1 = ROOT.TLorentzVector()
            tlv_lepton_1.SetPxPyPzE(lepton_1.px(), lepton_1.py(), lepton_1.pz(), lepton_1.e())
            tlv_lepton_2 = ROOT.TLorentzVector()
            tlv_lepton_2.SetPxPyPzE(lepton_2.px(), lepton_2.py(), lepton_2.pz(), lepton_2.e())
            tlv_quark_1 = ROOT.TLorentzVector()
            tlv_quark_1.SetPxPyPzE(quark_1.px(), quark_1.py(), quark_1.pz(), quark_1.e())

            self.histograms[process_id]["lepton1_pt_non_resonant"].Fill(tlv_lepton_1.Pt() / 1000.)
            self.histograms[process_id]["lepton1_pt"].Fill(tlv_lepton_1.Pt() / 1000.)
            self.histograms[process_id]["lepton2_pt"].Fill(tlv_lepton_2.Pt() / 1000.)
            self.histograms[process_id]["lepton2_pt_non_resonant"].Fill(tlv_lepton_2.Pt() / 1000.)
            self.histograms[process_id]["quark_pt"].Fill(tlv_quark_1.Pt() / 1000.)
            self.histograms[process_id]["quark_pt_non_resonant"].Fill(tlv_quark_1.Pt() / 1000.)

            mass1 = (tlv_lepton_1 + tlv_quark_1).M() / 1000.
            mass2 = (tlv_lepton_2 + tlv_quark_1).M() / 1000.
            if mass1 > mass2:
                self.histograms[process_id]["inv_mass_max"].Fill(mass1)
                self.histograms[process_id]["inv_mass_min"].Fill(mass2)
                self.histograms[process_id]["inv_mass_max_tchan"].Fill(mass1)
                self.histograms[process_id]["inv_mass_min_tchan"].Fill(mass2)
            else:
                self.histograms[process_id]["inv_mass_max_tchan"].Fill(mass2)
                self.histograms[process_id]["inv_mass_min_tchan"].Fill(mass1)

        f = ROOT.TFile.Open(input_file)
        tree = ROOT.xAOD.MakeTransientTree(f, self.tree_name)
        self.current_process_config = None
        max_entries = tree.GetEntries()
        if self.max_events is not None:
            max_entries = self.max_events

        for entry in range(max_entries):
            tree.GetEntry(entry)
            process_id = tree.EventInfo.runNumber()
            if process_id not in self.histograms:
                self.book_histograms(process_id)
            truth_particles = tree.TruthParticles
            LQ = [p for p in truth_particles if
                  p.pdgId() == 1102 or p.pdgId() == 1101 or p.pdgId() == 42 or p.pdgId() == 9000005
                  or abs(p.pdgId()) == 9000002]
            list([self.histograms[process_id]["lq_mass"].Fill(p.m() / 1000.) for p in LQ])
            self.histograms[process_id]["event_yields"].Fill(0)
            if len(LQ) == 0:
                run_tchannel()
            else:
                run_schannel()


class BcTruthAnalyser(object):
    def __init__(self, **kwargs):
        self.input_files = kwargs["input_files"]
        self.output_handle = OutputFileHandle(output_dir=kwargs["output_dir"])
        self.output_handle_hists = OutputFileHandle(output_dir=kwargs["output_dir"], output_file="output_hist.root")
        self.histograms = dict()
        self.plot_configs = dict()
        self.references = dict()
        self.tree_name = "CollectionTree"
        process_configs = YAMLLoader.read_yaml(kwargs["config_file"])
        self.processes = {int(channel): Process(process_config) for channel, process_config in
                          list(process_configs.items())}
        self.current_process_config = None
        self.setup()
        # self.book_histograms()
        self.book_plot_configs()
        self.processed_ids = []

    @staticmethod
    def setup():
        ROOT.gROOT.Macro('$ROOTCOREDIR/scripts/load_packages.C')
        ROOT.xAOD.Init().ignore()

    def book_histograms(self, process_id):
        def book_histogram(name, n_bins, x_min, x_max):
            if process_id not in self.histograms:
                self.histograms[process_id] = dict()
            self.histograms[process_id][name] = ROOT.TH1F("{:s}_{:d}".format(name, process_id), "", n_bins, x_min,
                                                          x_max)
            ROOT.SetOwnership(self.histograms[process_id][name], False)
            self.histograms[process_id][name].SetDirectory(0)

        book_histogram("leading_lepton_e", 50, 0., 50.)
        book_histogram("subleading_lepton_e", 50, 0., 50.)
        book_histogram("thirdleading_lepton_e", 50, 0., 50.)
        book_histogram("lepton_e", 50, 0., 50.)
        book_histogram("lepton_eta", 50, -3.2, 3.2)
        book_histogram("lepton_phi", 50, -3.2, 3.2)
        book_histogram("lepton_pdgId", 41, -20.5, 20.5)
        book_histogram("third_lepton_e", 50, 0., 50.)
        book_histogram("third_lepton_eta", 50, -3.2, 3.2)
        book_histogram("third_lepton_phi", 50, -3.2, 3.2)
        book_histogram("neutrino_e", 25, 0., 50.)
        book_histogram("neutrino_eta", 50, -3.2, 3.2)
        book_histogram("neutrino_phi", 50, -3.2, 3.2)
        book_histogram("jpsi_e", 25, 0., 50.)
        book_histogram("jpsi_eta", 50, -2.5, 2.5)
        book_histogram("jpsi_phi", 50, -3.2, 3.2)
        book_histogram("jpsi_lepton1_e", 50, 0., 50.)
        book_histogram("jpsi_lepton1_eta", 50, -3.2, 3.2)
        book_histogram("jpsi_lepton1_phi", 50, -3.2, 3.2)
        book_histogram("jpsi_lepton1_pdgId", 41, -20.5, 20.5)
        book_histogram("jpsi_lepton2_e", 50, 0., 50.)
        book_histogram("jpsi_lepton2_eta", 50, -3.2, 3.2)
        book_histogram("jpsi_lepton2_phi", 50, -3.2, 3.2)
        book_histogram("jpsi_lepton2_pdgId", 41, -20.5, 20.5)
        book_histogram("jpsi_muon_deta", 50, 0., 2.5)
        book_histogram("jpsi_muon_dphi", 50, 0., 3.2)
        book_histogram("jpsi_muon_dR", 50, 0., 0.8)
        book_histogram("jpsi_tau_deta", 50, 0., 2.5)
        book_histogram("jpsi_tau_dphi", 50, 0., 3.2)
        book_histogram("jpsi_tau_dR", 50, 0., 0.8)
        book_histogram("jpsi_mass", 50, 0., 5.)
        book_histogram("B_mass_visible", 25, 2., 7.)
        book_histogram("B_mass", 50, 0., 10.)
        book_histogram("B_mass_direct", 50, 0., 10.)
        book_histogram("Bc_status", 100, 0., 100.)
        book_histogram("Bc_pt", 50, 0., 50.)
        book_histogram("Bc_eta", 50, -3.2, 3.2)
        book_histogram("Bc_phi", 50, -3.2, 3.2)
        book_histogram("Bc_init_pt", 50, 0., 50.)
        book_histogram("Bc_init_eta", 50, -3.2, 3.2)
        book_histogram("Bc_init_phi", 50, -3.2, 3.2)
        book_histogram("tau_decay_length", 50, 0., 5.)

    def book_plot_configs(self):
        def book_plot_config(name, xtitle, **kwargs):
            pc = PlotConfig(dist=None, name=name, xtitle=xtitle, ytitle="Entries", watermark="Simulation Internal",
                            color=ROOT.kBlue, draw="HIST", no_fill=True, lumi=None)
            for args, val in list(kwargs.items()):
                setattr(pc, name, val)
            self.plot_configs[name] = pc

        book_plot_config("resonance_counter_decay1", "pdg ID")
        book_plot_config("resonance_decay1_child_pdg_ids", "pdg ID")
        book_plot_config("leading_lepton_e", "leading lepton E [GeV]")
        book_plot_config("subleading_lepton_e", "sub-leading lepton E [GeV]")
        book_plot_config("thirdleading_lepton_e", "third-leading lepton E [GeV]")
        book_plot_config("lepton_e", "lepton from B_{c} decay E [GeV]")
        book_plot_config("lepton_eta", "lepton from B_{c} decay #eta")
        book_plot_config("lepton_phi", "lepton from B_{c} decay #phi")
        book_plot_config("lepton_pdgId", "lepton from B_{c} decay PDG ID")
        book_plot_config("third_lepton_e", "lepton from tau decay E [GeV]")
        book_plot_config("third_lepton_eta", "lepton from tau decay #eta")
        book_plot_config("third_lepton_phi", "lepton from tau decay #phi")
        book_plot_config("third_lepton_pdgId", "lepton from tau decay PDG Id")
        book_plot_config("neutrino_e", "neutrino from B_{c} decay E [GeV]")
        book_plot_config("neutrino_eta", "neutrino from B_{c} decay #eta")
        book_plot_config("neutrino_phi", "neutrino from B_{c} decay #phi")
        book_plot_config("jpsi_e", "J/#Psi E [GeV]")
        book_plot_config("jpsi_eta", "J/#Psi #eta")
        book_plot_config("jpsi_phi", "J/#Psi #phi")
        book_plot_config("jpsi_lepton1_e", "1^{st} muon from J/#Psi decay E [GeV]")
        book_plot_config("jpsi_lepton1_eta", "1^{st} muon from J/#Psi decay #eta")
        book_plot_config("jpsi_lepton1_phi", "1^{st} muon from J/#Psi decay #phi")
        book_plot_config("jpsi_lepton1_pdgId", "1^{st} muon from J/#Psi decay PDG ID")
        book_plot_config("jpsi_lepton2_e", "2^{nd} muon from J/#Psi decay E [GeV]")
        book_plot_config("jpsi_lepton2_eta", "2^{nd} muon from J/#Psi decay #eta")
        book_plot_config("jpsi_lepton2_phi", "2^{nd} muon from J/#Psi decay #phi")
        book_plot_config("jpsi_lepton2_pdgId", "2^{nd} muon from J/#Psi decay PDG ID")
        book_plot_config("jpsi_muon_deta", "#Delta#eta(J/#Psi, e/#mu)")
        book_plot_config("jpsi_muon_dphi", "#Delta#phi(J/#Psi, e/#mu)")
        book_plot_config("jpsi_muon_dR", "#DeltaR(J/#Psi, e/#mu)")
        book_plot_config("jpsi_tau_deta", "#Delta#eta(J/#Psi, #tau)")
        book_plot_config("jpsi_tau_dphi", "#Delta#phi(J/#Psi, #tau)")
        book_plot_config("jpsi_tau_dR", "#DeltaR(J/#Psi, #tau)")
        book_plot_config("jpsi_mass", "M_{J/#Psi} [GeV]")
        book_plot_config("B_mass_visible", "visible M_{B} [GeV]")
        book_plot_config("B_mass", "M_{B} [GeV]")
        book_plot_config("B_mass_direct", "M_{B} [GeV]")
        book_plot_config("Bc_status", "B_{c} status")
        book_plot_config("Bc_pt", "p_{T} (B_{c}) [GeV]")
        book_plot_config("Bc_eta", "#eta (B_{c})")
        book_plot_config("Bc_phi", "#phi (B_{c})")
        book_plot_config("Bc_init_pt", "p_{T} (B_{c}) [GeV]")
        book_plot_config("Bc_init_eta", "#eta (B_{c})")
        book_plot_config("Bc_init_phi", "#phi (B_{c})")
        book_plot_config("tau_decay_length", "#tau decay length [mm]")

    def run(self):
        for input_file in self.input_files:
            self.analyse_file(input_file)
        self.plot_histograms()
        self.output_handle.write_and_close()
        self.output_handle_hists.write_and_close()

    def plot_histograms(self):
        for process_id, histograms in list(self.histograms.items()):
            if process_id not in self.processed_ids:
                continue
            for hist_name, hist in list(histograms.items()):
                pc = self.plot_configs[hist_name]
                if hist_name == "decay2_mode":
                    pc.axis_labels = self.processes[process_id].get_bin_labels()
                    pc.normalise = True
                canvas = PT.plot_obj(hist, pc)
                if hist_name == "decay2_mode":
                    pc_ref = copy(pc)
                    pc_ref.color = ROOT.kRed
                    PT.add_histogram_to_canvas(canvas, self.references[process_id], pc_ref)
                FT.decorate_canvas(canvas, pc)
                self.output_handle.register_object(canvas, str(process_id))
                self.output_handle_hists.register_object(hist, str(process_id))

    @staticmethod
    def deltaR(particle1, particle2):
        def shift_phi(phi):
            if phi >= pi:
                return phi - 2. * pi
            if phi < -1. * pi:
                return phi + 2. * pi
            return phi

        phi1 = shift_phi(particle1[2])
        phi2 = shift_phi(particle2[2])
        return sqrt(pow(particle1[1] - particle2[1], 2) + pow(phi1 - phi2, 2))

    def analyse_file(self, input_file):
        f = ROOT.TFile.Open(input_file)
        tree = ROOT.xAOD.MakeTransientTree(f, self.tree_name)
        self.current_process_config = None
        n_entries = 0
        for entry in range(tree.GetEntries()):
            n_entries += 1
            # for entry in xrange(100):
            tree.GetEntry(entry)
            process_id = tree.EventInfo.runNumber()
            if process_id not in self.histograms:
                self.book_histograms(process_id)
            if process_id not in self.processed_ids:
                self.processed_ids.append(process_id)
            # if self.current_process_config is None:
            #     self.current_process_config = self.processes[process_id]
            truth_particles = tree.TruthParticles
            Bc = [p for p in truth_particles if p.pdgId() == 521 or p.pdgId() == 541]
            # self.histograms[process_id]["resonance_counter_decay1"].Fill(1)
            if len(Bc) == 0:
                print("Suspicious event. Could not find Bc")
                continue
            if process_id in [400571, 400572, 400573, 400574]:
                Bc_init = filter(lambda p: p.status() == 23, Bc)[0]
                Bc_tlv_init = ROOT.TLorentzVector()
                Bc_tlv_init.SetPxPyPzE(Bc_init.px(), Bc_init.py(), Bc_init.pz(), Bc_init.e())
                self.histograms[process_id]["Bc_init_pt"].Fill(Bc_tlv_init.Pt() / 1000.)
                self.histograms[process_id]["Bc_init_eta"].Fill(Bc_tlv_init.Eta())
                self.histograms[process_id]["Bc_init_phi"].Fill(Bc_tlv_init.Phi())
                Bc = [p for p in Bc if p.status() == 2]
            Bc_tlv = ROOT.TLorentzVector()
            Bc_tlv.SetPxPyPzE(Bc[0].px(), Bc[0].py(), Bc[0].pz(), Bc[0].e())
            for bc in Bc:
                self.histograms[process_id]["Bc_status"].Fill(bc.status())
            self.histograms[process_id]["B_mass_direct"].Fill(Bc_tlv.M() / 1000.)
            self.histograms[process_id]["Bc_pt"].Fill(Bc_tlv.Pt() / 1000.)
            self.histograms[process_id]["Bc_eta"].Fill(Bc_tlv.Eta())
            self.histograms[process_id]["Bc_phi"].Fill(Bc_tlv.Phi())

            prod_vtx = Bc[0].decayVtxLink().outgoingParticleLinks()
            try:
                jpsi = filter(lambda particle: abs(particle.pdgId()) == 443, prod_vtx)[0]
            except IndexError:
                print("Could not find J/Psi")
                continue
            self.histograms[process_id]["jpsi_e"].Fill(jpsi.e() / 1000.)
            self.histograms[process_id]["jpsi_eta"].Fill(jpsi.eta())
            self.histograms[process_id]["jpsi_phi"].Fill(jpsi.phi())
            lepton = filter(lambda particle: abs(particle.pdgId()) == 15 or abs(particle.pdgId()) == 13 or abs(
                particle.pdgId()) == 11, prod_vtx)[0]
            resonance_neutrino = filter(
                lambda particle: abs(particle.pdgId()) == 16 or abs(particle.pdgId()) == 14 or abs(
                    particle.pdgId()) == 12,
                prod_vtx)[0]

            self.histograms[process_id]["lepton_e"].Fill(lepton.e() / 1000.)
            self.histograms[process_id]["lepton_eta"].Fill(lepton.eta())
            self.histograms[process_id]["lepton_phi"].Fill(lepton.phi())
            self.histograms[process_id]["lepton_pdgId"].Fill(lepton.pdgId())
            self.histograms[process_id]["neutrino_e"].Fill(resonance_neutrino.e() / 1000.)
            self.histograms[process_id]["neutrino_eta"].Fill(resonance_neutrino.eta())
            self.histograms[process_id]["neutrino_phi"].Fill(resonance_neutrino.phi())

            jpsi_tlv = ROOT.TLorentzVector()
            jpsi_tlv.SetPxPyPzE(jpsi.px(), jpsi.py(), jpsi.pz(), jpsi.e() / 1000.)

            if abs(lepton.pdgId()) == 15:
                third_muon = filter(lambda particle: abs(particle.pdgId()) == 13 or abs(particle.pdgId()) == 11,
                                    lepton.decayVtxLink().outgoingParticleLinks())[0]
                self.histograms[process_id]["third_lepton_e"].Fill(third_muon.e() / 1000.)
                self.histograms[process_id]["third_lepton_eta"].Fill(third_muon.eta())
                self.histograms[process_id]["third_lepton_phi"].Fill(third_muon.phi())

                thrird_tlv = ROOT.TLorentzVector()
                thrird_tlv.SetPxPyPzE(third_muon.px(), third_muon.py(), third_muon.pz(), third_muon.e() / 1000.)

                self.histograms[process_id]["jpsi_muon_deta"].Fill(abs(jpsi_tlv.Eta() - thrird_tlv.Eta()))
                self.histograms[process_id]["jpsi_muon_dphi"].Fill(jpsi_tlv.DeltaPhi(thrird_tlv))
                self.histograms[process_id]["jpsi_muon_dR"].Fill(jpsi_tlv.DeltaR(thrird_tlv))

                tau_prod_vertex_link = lepton.prodVtxLink()
                tau_decay_vertex_link = lepton.decayVtxLink()
                tau_prod_vertex = ROOT.TVector3(tau_prod_vertex_link.x(),
                                                tau_prod_vertex_link.y(),
                                                tau_prod_vertex_link.z())
                tau_decay_vertex = ROOT.TVector3(tau_decay_vertex_link.x(),
                                                 tau_decay_vertex_link.y(),
                                                 tau_decay_vertex_link.z())
                tau_tlv = ROOT.TLorentzVector()
                tau_tlv.SetPxPyPzE(lepton.px(), lepton.py(), lepton.pz(), lepton.e() / 1000.)
                self.histograms[process_id]["tau_decay_length"].Fill((tau_decay_vertex - tau_prod_vertex).Mag())
                self.histograms[process_id]["jpsi_tau_deta"].Fill(abs(jpsi_tlv.Eta() - tau_tlv.Eta()))
                self.histograms[process_id]["jpsi_tau_dphi"].Fill(jpsi_tlv.DeltaPhi(tau_tlv))
                self.histograms[process_id]["jpsi_tau_dR"].Fill(jpsi_tlv.DeltaR(tau_tlv))
            else:
                third_muon = lepton
                thrird_tlv = ROOT.TLorentzVector()
                thrird_tlv.SetPxPyPzE(lepton.px(), lepton.py(), lepton.pz(), lepton.e() / 1000.)
                self.histograms[process_id]["jpsi_muon_deta"].Fill(abs(jpsi_tlv.Eta() - thrird_tlv.Eta()))
                self.histograms[process_id]["jpsi_muon_dphi"].Fill(jpsi_tlv.DeltaPhi(thrird_tlv))
                self.histograms[process_id]["jpsi_muon_dR"].Fill(jpsi_tlv.DeltaR(thrird_tlv))

            jpsi_decay_vtx = jpsi.decayVtxLink().outgoingParticleLinks()

            jpsi_muon1 = filter(lambda particle: abs(particle.pdgId()) == 13, jpsi_decay_vtx)[0]
            jpsi_muon2 = filter(lambda particle: abs(particle.pdgId()) == 13, jpsi_decay_vtx)[1]

            self.histograms[process_id]["jpsi_lepton1_e"].Fill(jpsi_muon1.e() / 1000.)
            self.histograms[process_id]["jpsi_lepton1_eta"].Fill(jpsi_muon1.eta())
            self.histograms[process_id]["jpsi_lepton1_phi"].Fill(jpsi_muon1.phi())
            self.histograms[process_id]["jpsi_lepton1_pdgId"].Fill(jpsi_muon1.pdgId())
            self.histograms[process_id]["jpsi_lepton2_e"].Fill(jpsi_muon2.e() / 1000.)
            self.histograms[process_id]["jpsi_lepton2_eta"].Fill(jpsi_muon2.eta())
            self.histograms[process_id]["jpsi_lepton2_phi"].Fill(jpsi_muon2.phi())
            self.histograms[process_id]["jpsi_lepton2_phi"].Fill(jpsi_muon2.phi())
            self.histograms[process_id]["jpsi_lepton2_pdgId"].Fill(jpsi_muon2.pdgId())

            pts = [jpsi_muon1.e(), jpsi_muon2.e(), third_muon.e()]
            pts.sort(reverse=True)
            self.histograms[process_id]["leading_lepton_e"].Fill(pts[0] / 1000.)
            self.histograms[process_id]["subleading_lepton_e"].Fill(pts[1] / 1000.)
            self.histograms[process_id]["thirdleading_lepton_e"].Fill(pts[2] / 1000.)

            jpsi_muon1_tlv = ROOT.TLorentzVector()
            jpsi_muon1_tlv.SetPxPyPzE(jpsi_muon1.px(), jpsi_muon1.py(), jpsi_muon1.pz(), jpsi_muon1.e())
            jpsi_muon2_tlv = ROOT.TLorentzVector()
            jpsi_muon2_tlv.SetPxPyPzE(jpsi_muon2.px(), jpsi_muon2.py(), jpsi_muon2.pz(), jpsi_muon2.e())
            lepton_tlv = ROOT.TLorentzVector()
            lepton_tlv.SetPxPyPzE(lepton.px(), lepton.py(), lepton.pz(), lepton.e())
            resonance_neutrino_tlv = ROOT.TLorentzVector()
            resonance_neutrino_tlv.SetPxPyPzE(resonance_neutrino.px(), resonance_neutrino.py(),
                                              resonance_neutrino.pz(), resonance_neutrino.e())
            self.histograms[process_id]["jpsi_mass"].Fill((jpsi_muon1_tlv + jpsi_muon2_tlv).M() / 1000.)
            self.histograms[process_id]["B_mass_visible"].Fill(
                (jpsi_muon1_tlv + jpsi_muon2_tlv + lepton_tlv).M() / 1000.)
            self.histograms[process_id]["B_mass"].Fill(
                (jpsi_muon1_tlv + jpsi_muon2_tlv + lepton_tlv + resonance_neutrino_tlv).M() / 1000.)

        print("N processed entries: ", n_entries, " and hist entries: ",
              self.histograms[process_id]["B_mass"].GetEntries())


class TruthAnalyerT3M(object):
    def __init__(self, **kwargs):
        kwargs.setdefault("selection", "None")
        self.input_files = kwargs["input_files"]
        self.output_handle = OutputFileHandle(output_dir=kwargs["output_dir"])
        self.histograms = dict()
        self.plot_configs = dict()
        self.references = dict()
        self.tree_name = "CollectionTree"
        process_configs = YAMLLoader.read_yaml(kwargs["config_file"])
        self.processes = {int(channel): Process(process_config) for channel, process_config in
                          list(process_configs.items())}
        self.current_process_config = None
        self.setup()
        self.book_plot_configs()
        self.build_references()
        self.decay_chains = {}
        if kwargs["selection"] == "None":
            self.pattern = ""
        elif kwargs["selection"] == "low_pt":
            self.pattern = "low_pt"
        elif kwargs["selection"] == "low_p":
            self.pattern = "low_p"

    @staticmethod
    def setup():
        ROOT.gROOT.Macro('$ROOTCOREDIR/scripts/load_packages.C')
        ROOT.xAOD.Init().ignore()

    def book_histograms(self, process_id):
        def book_histogram(name, n_bins, x_min, x_max):
            if process_id not in self.histograms:
                self.histograms[process_id] = dict()
            self.histograms[process_id][name] = ROOT.TH1F("{:s}{:s}_{:d}".format(self.pattern, name, process_id), "",
                                                          n_bins, x_min, x_max)
            ROOT.SetOwnership(self.histograms[process_id][name], False)
            self.histograms[process_id][name].SetDirectory(0)

        def book_histogram_2d(name, n_bins_x, x_min, x_max, n_bins_y, y_min, y_max):
            if process_id not in self.histograms:
                self.histograms[process_id] = dict()
            self.histograms[process_id][name] = ROOT.TH2F("{:s}{:s}_{:d}".format(self.pattern, name, process_id), "",
                                                          n_bins_x, x_min, x_max,
                                                          n_bins_y, y_min, y_max)
            ROOT.SetOwnership(self.histograms[process_id][name], False)
            self.histograms[process_id][name].SetDirectory(0)

        book_histogram("tau_e", 50, 0., 100.)
        book_histogram("tau_phi", 50, -3.2, 3.2)
        book_histogram("tau_eta", 50, -4.0, 4.0)
        book_histogram("tau_parent_pdgid", 4000, -2000, 2000)
        book_histogram("tau_lifetime", 100, 0., 10.)
        book_histogram("muon_e", 20, 0., 20.)
        book_histogram("muon_eta", 50, -4.0, 4.0)
        book_histogram("muon_phi", 50, -3.2, 3.2)
        book_histogram("lead_muon_e", 20, 0., 20.)
        book_histogram("lead_muon_eta", 50, -4.0, 4.0)
        book_histogram("lead_muon_phi", 50, -3.2, 3.2)
        book_histogram("sub_lead_muon_eta", 50, -4.0, 4.0)
        book_histogram("sub_lead_muon_phi", 50, -3.2, 3.2)
        book_histogram("sub_lead_muon_e", 20, 0., 20.)
        book_histogram("third_lead_muon_e", 20, 0., 20.)
        book_histogram("third_lead_muon_eta", 50, -4.0, 4.0)
        book_histogram("third_lead_muon_phi", 50, -3.2, 3.2)
        book_histogram_2d("tau_e_eta", 50, 0., 100., 50, -4.0, 4.0)
        book_histogram_2d("muon_e_eta", 50, 0., 100., 50, -4.0, 4.0)
        book_histogram_2d("lead_muon_e_eta", 50, 0., 100., 50, -4.0, 4.0)
        book_histogram_2d("sub_lead_muon_e_eta", 50, 0., 100., 50, -4.0, 4.0)
        book_histogram_2d("third_lead_muon_e_eta", 50, 0., 100., 50, -4.0, 4.0)

    def book_plot_configs(self):
        def book_plot_config(name, xtitle, **kwargs):
            pc = PlotConfig(dist=None, name=name, xtitle=xtitle, ytitle="Entries", watermark="Simulation Internal",
                            color=ROOT.kBlue, lumi=-1)
            for args, val in list(kwargs.items()):
                setattr(pc, args, val)
            self.plot_configs[name] = pc

        book_plot_config("tau_e", "#tau E [GeV]")
        book_plot_config("tau_eta", "#tau #eta")
        book_plot_config("tau_phi", "#tau #phi")
        book_plot_config('tau_parent_pdgid', "PDG ID #tau parent")
        book_plot_config('tau_lifetime', "#tau lifetime [10^{-13}s]")
        book_plot_config("muon_e", "#mu E [GeV]")
        book_plot_config("muon_eta", "#mu #eta")
        book_plot_config("muon_phi", "#mu #phi")
        book_plot_config("lead_muon_e", "lead. #mu E [GeV]")
        book_plot_config("lead_muon_eta", "lead. #mu #eta")
        book_plot_config("lead_muon_phi", "lead. #mu #phi")
        book_plot_config("sub_lead_muon_e", "sub-lead. #mu E [GeV]")
        book_plot_config("sub_lead_muon_eta", "sub-lead. #mu #eta")
        book_plot_config("sub_lead_muon_phi", "sub-lead. #mu #phi")
        book_plot_config("third_lead_muon_e", "third-lead. #mu E [GeV]")
        book_plot_config("third_lead_muon_eta", "third-lead. #mu #eta")
        book_plot_config("third_lead_muon_phi", "third-lead. #mu #phi")
        book_plot_config("tau_e_eta", "#tau E [GeV]", ytitle="#tau #eta", ztitle="Events",
                         draw_option="COLZ")
        book_plot_config("muon_e_eta", "#mu E [GeV]", ytitle="#mu #eta", ztitle="Events",
                         draw_option="COLZ")
        book_plot_config("lead_muon_e_eta", "lead. #mu E [GeV]", ytitle="lead. #mu #eta", ztitle="Events",
                         draw_option="COLZ")
        book_plot_config("sub_lead_muon_e_eta", "sub-lead. #mu E [GeV]", ytitle="sub-lead. #mu #eta", ztitle="Events",
                         draw_option="COLZ")
        book_plot_config("third_lead_muon_e_eta", "third-lead. #mu E [GeV]", ytitle="third=lead. #mu #eta",
                         ztitle="Events", draw_option="COLZ")

    def run(self):
        for input_file in self.input_files:
            print(input_file)
            try:
                self.analyse_file(input_file)
            except TypeError:
                print(traceback.print_exc())
                print("Could not analyse {:s}".format(input_file))
                continue
        self.print_decay_chains()
        self.plot_histograms()
        self.output_handle.write_and_close()

    def print_decay_chains(self):
        def convert_pdgid(pdgid):
            try:
                return filter(lambda d: d[0] == pdgid, list(particle_map.values()))[0][1]
            except IndexError:
                return str(pdgid)

        for chain, count in list(self.decay_chains.items()):
            print([convert_pdgid(part) for part in chain], " \t & \t", count)

    def build_references(self):
        def find_br(process, mode):
            decay2_mode = filter(lambda decay: sorted(decay[:-1], reverse=True) == mode, process.decay2_pdgid)[0]
            return decay2_mode[-1]

        for process_id, process in list(self.processes.items()):
            reference_hist = ROOT.TH1F("decay_mode_reference_{:d}".format(process_id), "", 4, -1.5, 2.5)
            reference_hist.SetBinContent(1, 0)
            for mode_index in range(len(process.decay2_sorted)):
                br = find_br(process, process.decay2_sorted[mode_index])
                reference_hist.SetBinContent(mode_index + 2, br)
            self.references[process_id] = reference_hist

    def plot_histograms(self):
        for process_id, histograms in list(self.histograms.items()):
            for hist_name, hist in list(histograms.items()):
                pc = self.plot_configs[hist_name]
                if hist_name == "decay2_mode":
                    pc.axis_labels = self.processes[process_id].get_bin_labels()
                    pc.normalise = True
                canvas = PT.plot_obj(hist, pc)
                if hist_name == "decay2_mode":
                    pc_ref = copy(pc)
                    pc_ref.color = ROOT.kRed
                    PT.add_histogram_to_canvas(canvas, self.references[process_id], pc_ref)
                FT.decorate_canvas(canvas, pc)
                self.output_handle.register_object(canvas, str(process_id))

    def analyse_file(self, input_file):
        def get_parent(tau):
            for part in tau.prodVtxLink().incomingParticleLinks():
                if abs(part.pdgId()) != 15:
                    return part
            return get_parent(part)

        def get_decay_chain(tau):
            parent = get_parent(tau)
            parent_childs = [part.pdgId() for part in parent.decayVtxLink().outgoingParticleLinks()]
            parent_childs = [pdgid for pdgid in parent_childs if pdgid != particle_map['gamma'][0]]
            return tuple([parent.pdgId()] + parent_childs)

        f = ROOT.TFile.Open(input_file)
        tree = ROOT.xAOD.MakeTransientTree(f, self.tree_name)
        self.current_process_config = None
        for entry in range(tree.GetEntries()):
            tree.GetEntry(entry)
            process_id = tree.EventInfo.runNumber()
            if process_id not in self.histograms:
                self.book_histograms(process_id)
            if self.current_process_config is None:
                self.current_process_config = self.processes[process_id]
            truth_particles = tree.TruthParticles
            tau_decay = [p for p in truth_particles if abs(p.pdgId()) == 15]
            if len(tau_decay) == 0:
                print("Suspicious event. Could not find tau for process ", process_id)
                continue
            muon_kinematics = list()
            tau_decay_vertex = tau_decay[0].decayVtxLink().outgoingParticleLinks()
            for dec in tau_decay:
                if len([particle for particle in dec.decayVtxLink().outgoingParticleLinks() if
                        abs(particle.pdgId()) == 13]) != 3:
                    continue
                tau_decay_vertex = dec.decayVtxLink().outgoingParticleLinks()
                lfv_decay = dec
            decay_chain = get_decay_chain(lfv_decay)
            if decay_chain not in self.decay_chains:
                self.decay_chains[decay_chain] = 1
            else:
                self.decay_chains[decay_chain] += 1

            self.histograms[process_id]['tau_parent_pdgid'].Fill(get_parent(lfv_decay).pdgId())
            vec_prod = ROOT.TVector3(lfv_decay.prodVtxLink().x(), lfv_decay.prodVtxLink().y(),
                                     lfv_decay.prodVtxLink().z())
            vec_decay = ROOT.TVector3(lfv_decay.decayVtxLink().x(), lfv_decay.decayVtxLink().y(),
                                      lfv_decay.decayVtxLink().z())
            decay_length = (vec_decay - vec_prod).Mag()
            self.histograms[process_id]['tau_lifetime'].Fill(decay_length)
            try:
                muons = [particle for particle in tau_decay_vertex if abs(particle.pdgId()) == 13]
                for i in range(3):
                    tlv = ROOT.TLorentzVector()
                    tlv.SetPxPyPzE(muons[i].px() / 1000., muons[i].py() / 1000., muons[i].pz() / 1000.,
                                   muons[i].e() / 1000.)
                    muon_kinematics.append([muons[i].e() / 1000.,
                                            muons[i].eta(),
                                            muons[i].phi(),
                                            tlv.Pt()])
            except IndexError:
                print("Could not find any muon for first resonance decay in process", process_id)
                continue
            muon_kinematics.sort(key=lambda i: i[0], reverse=True)

            if self.pattern == "low_pt":
                if muon_kinematics[1][3] < 4.:
                    continue
            if self.pattern == "low_p":
                if abs(muon_kinematics[1][1]) > 2.7 and muon_kinematics[1][0] < 20.:
                    continue
                if abs(muon_kinematics[1][1]) < 2.7 and muon_kinematics[1][3] < 4.:
                    continue
            self.histograms[process_id]["tau_e"].Fill(lfv_decay.e() / 1000.)
            self.histograms[process_id]["tau_eta"].Fill(lfv_decay.eta())
            self.histograms[process_id]["tau_phi"].Fill(lfv_decay.phi())
            self.histograms[process_id]["tau_e_eta"].Fill(lfv_decay.e() / 1000., lfv_decay.eta())

            for i in range(3):
                self.histograms[process_id]["muon_e"].Fill(muon_kinematics[i][0])
                self.histograms[process_id]["muon_eta"].Fill(muon_kinematics[i][1])
                self.histograms[process_id]["muon_phi"].Fill(muon_kinematics[i][2])
                self.histograms[process_id]["muon_e_eta"].Fill(muon_kinematics[i][0], muon_kinematics[i][1])
            self.histograms[process_id]["lead_muon_e"].Fill(muon_kinematics[0][0])
            self.histograms[process_id]["lead_muon_eta"].Fill(muon_kinematics[0][1])
            self.histograms[process_id]["lead_muon_phi"].Fill(muon_kinematics[0][2])
            self.histograms[process_id]["lead_muon_e_eta"].Fill(muon_kinematics[0][0], muon_kinematics[0][1])
            self.histograms[process_id]["sub_lead_muon_e"].Fill(muon_kinematics[1][0])
            self.histograms[process_id]["sub_lead_muon_eta"].Fill(muon_kinematics[1][1])
            self.histograms[process_id]["sub_lead_muon_phi"].Fill(muon_kinematics[1][2])
            self.histograms[process_id]["sub_lead_muon_e_eta"].Fill(muon_kinematics[1][0], muon_kinematics[1][1])
            self.histograms[process_id]["third_lead_muon_e"].Fill(muon_kinematics[2][0])
            self.histograms[process_id]["third_lead_muon_eta"].Fill(muon_kinematics[2][1])
            self.histograms[process_id]["third_lead_muon_phi"].Fill(muon_kinematics[2][2])
            self.histograms[process_id]["third_lead_muon_e_eta"].Fill(muon_kinematics[2][0], muon_kinematics[2][1])
        f.Close()
