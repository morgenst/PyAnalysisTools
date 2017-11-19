from PyAnalysisTools.base.YAMLHandle import YAMLLoader
from PyAnalysisTools.base.OutputHandle import OutputFileHandle
from PyAnalysisTools.PlottingUtils import PlottingTools as PT
from PyAnalysisTools.PlottingUtils.PlotConfig import PlotConfig
from copy import copy
from math import sqrt, pi
import PyAnalysisTools.PlottingUtils.Formatting as FT
import ROOT


particle_map = dict()
particle_map["Ds-"] = [-431, "D_{s}^{-}"]
particle_map["Ds+"] = [431, "D_{s}^{+}"]
particle_map["D-"] = [-411, "D^{-}"]
particle_map["D+"] = [411, "D^{+}"]
particle_map["mu-"] = [-13, "#mu^{-}"]
particle_map["mu+"] = [13, "#mu^{+}"]
particle_map["phi1020"] = [333, "#phi"]
particle_map["pi0"] = [111, "#pi^{0}"]
particle_map["anti_mu_nu"] = [14, "#nu_{#mu}"]
particle_map["mu_nu"] = [-14, "#nu_{#mu}"]
particle_map["gamma"] = [22, "#gamma"]
particle_map["rho0"] = [113, "#rho"]
particle_map["eta"] = [221, "#eta"]
particle_map["eta'"] = [331, "#eta'"]
particle_map["omega"] = [223, "#omega"]
particle_map["LQ"] = [1102, "LQ"]
particle_map["e-"] = [-11, "e^{-}"]
particle_map["e+"] = [11, "e^{+}"]
particle_map["q"] = [range(1,6), "q"]


class Process(object):
    def __init__(self, config):
        self.decay1_str = config["decay1"]
        self.decay2 = config["decay2"]
        self.decay1_pdgid = map(lambda name: particle_map[name][0], self.decay1_str)
        print self.decay2
        self.decay2_pdgid = [map(lambda name: particle_map[name][0] if isinstance(name, str) else name, sub)
                             for sub in self.decay2]
        self.decay_2_initial_resonance = [decay[0] for decay in self.decay2_pdgid]
        self.decay2_sorted = map(lambda i: sorted(i[:-1], reverse=True), self.decay2_pdgid)

    def get_bin_labels(self):
        labels = ["undefined"]
        for decay in self.decay2:
            labels.append(particle_map[decay[0]][1] + "#rightarrow" + "+".join([particle_map[decay[i]][1] for i in
                                                                                range(1, len(decay)-1)]))
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
        self.processes = {int(channel): Process(process_config) for channel, process_config in process_configs.iteritems()}
        self.current_process_config = None
        self.setup()
        self.book_histograms()
        self.book_plot_configs()
        self.build_references()

    @staticmethod
    def setup():
        ROOT.gROOT.Macro('$ROOTCOREDIR/scripts/load_packages.C')
        ROOT.xAOD.Init().ignore()

    def book_histograms(self):
        def book_histogram(name, n_bins, x_min, x_max):
            for process_id in self.processes.keys():
                if process_id not in self.histograms:
                    self.histograms[process_id] = dict()
                self.histograms[process_id][name] = ROOT.TH1F("{:s}_{:d}".format(name, process_id), "", n_bins, x_min,
                                                              x_max)
        book_histogram("resonance_counter_decay1", 2, -0.5, 1.5)
        book_histogram("resonance_decay1_child_pdg_ids", 501, -500.5, 500.5)
        book_histogram("decay2_mode", 4, -1.5, 2.5)
        book_histogram("muon_e", 20, 0., 50.)
        book_histogram("muon_eta", 50, -2.5, 2.5)
        book_histogram("muon_phi", 50, -3.2, 3.2)
        book_histogram("lead_muon_e", 20, 0., 20.)
        book_histogram("sub_lead_muon_e", 20, 0., 20.)
        book_histogram("third_lead_muon_e", 20, 0., 20.)
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
                            color=ROOT.kBlue)
            for args, val in kwargs.iteritems():
                setattr(pc, name, val)
            self.plot_configs[name] = pc

        book_plot_config("resonance_counter_decay1", "pdg ID")
        book_plot_config("resonance_decay1_child_pdg_ids", "pdg ID")
        book_plot_config("muon_e", "all #mu E [GeV]")
        book_plot_config("lead_muon_e", "lead. #mu E [GeV]")
        book_plot_config("sub_lead_muon_e", "sub-lead. #mu E [GeV]")
        book_plot_config("third_lead_muon_e", "third-lead. #mu E [GeV]")
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

        for process_id, process in self.processes.iteritems():
            reference_hist = ROOT.TH1F("decay_mode_reference_{:d}".format(process_id), "", 4, -1.5, 2.5)
            reference_hist.SetBinContent(1, 0)
            for mode_index in range(len(process.decay2_sorted)):
                br = find_br(process, process.decay2_sorted[mode_index])
                reference_hist.SetBinContent(mode_index + 2, br)
            self.references[process_id] = reference_hist

    def plot_histograms(self):
        for process_id, histograms in self.histograms.iteritems():
            for hist_name, hist in histograms.iteritems():
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
        for entry in xrange(tree.GetEntries()):
            tree.GetEntry(entry)
            process_id = tree.EventInfo.runNumber()
            if self.current_process_config is None:
                self.current_process_config = self.processes[process_id]
            truth_particles = tree.TruthParticles
            resonance_decay1 = filter(lambda p: p.pdgId() == self.current_process_config.decay1_pdgid[0],
                                      truth_particles)
            self.histograms[process_id]["resonance_counter_decay1"].Fill(len(resonance_decay1))
            if len(resonance_decay1) == 0:
                print "Suspicious event. Could not find ", self.current_process_config.decay1_pdgid[0], " for process ", process_id
                continue
            resonance1_vertex = resonance_decay1[0].decayVtxLink().outgoingParticleLinks()
            muon_pts = list()
            photons_pts = list()
            all_photons_pts = list()
            try:
                muon_pts.append([filter(lambda particle: abs(particle.pdgId()) == 13, resonance1_vertex)[0].e() / 1000.,
                                 filter(lambda particle: abs(particle.pdgId()) == 13, resonance1_vertex)[0].eta(),
                                 filter(lambda particle: abs(particle.pdgId()) == 13, resonance1_vertex)[0].phi()])
            except IndexError:
                print "Could not find any muon for first resonance decay in process", process_id
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
                        self.histograms[process_id]["decay2_gamma_e_after_veto_low_pt"].Fill(resonance2_child.e() / 1000.)
                        self.histograms[process_id]["decay2_gamma_status_after_veto"].Fill(resonance2_child.status())
                        mother = resonance2_child.prodVtx().incomingParticleLinks()[0]
                        self.histograms[process_id]["control_photon_mother_pdgid"].Fill(mother.pdgId())
                        photons_pts.append([resonance2_child.e() / 1000., resonance2_child.eta(),
                                            resonance2_child.phi()])
            mode.sort(key=lambda i: i[0], reverse=True)
            try:
                decay_mode = self.current_process_config.decay2_sorted.index(map(lambda i: i[0], mode))
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
                print "did not find 3 muons!"
                continue
            self.histograms[process_id]["lead_muon_e"].Fill(muon_pts[0][0])
            self.histograms[process_id]["sub_lead_muon_e"].Fill(muon_pts[1][0])
            self.histograms[process_id]["third_lead_muon_e"].Fill(muon_pts[2][0])
            for photon in photons_pts:
                for muon in muon_pts:
                    self.histograms[process_id]["decay2_gamma_mu_dr_after_veto"].Fill(self.deltaR(photon, muon))
            for photon in all_photons_pts:
                for muon in muon_pts:
                    self.histograms[process_id]["decay2_gamma_mu_dr"].Fill(self.deltaR(photon, muon))

        f.Close()


class LQTruthAnalyser(object):
    def __init__(self, **kwargs):
        self.input_files = kwargs["input_files"]
        self.output_handle = OutputFileHandle(output_dir=kwargs["output_dir"])
        self.histograms = dict()
        self.plot_configs = dict()
        self.references = dict()
        self.tree_name = "CollectionTree"
        process_configs = YAMLLoader.read_yaml(kwargs["config_file"])
        self.processes = {int(channel): Process(process_config) for channel, process_config in process_configs.iteritems()}
        self.current_process_config = None
        self.setup()
        self.book_histograms()
        self.book_plot_configs()
        #self.build_references()

    @staticmethod
    def setup():
        ROOT.gROOT.Macro('$ROOTCOREDIR/scripts/load_packages.C')
        ROOT.xAOD.Init().ignore()

    def book_histograms(self):
        def book_histogram(name, n_bins, x_min, x_max):
            for process_id in self.processes.keys():
                if process_id not in self.histograms:
                    self.histograms[process_id] = dict()
                self.histograms[process_id][name] = ROOT.TH1F("{:s}_{:d}".format(name, process_id), "", n_bins, x_min,
                                                              x_max)
        #book_histogram("LQ_counter", 2, -0.5, 1.5)
        #book_histogram("decay2_mode", 4, -1.5, 2.5)
        book_histogram("lepton1_e", 50, 0., 1500.)
        book_histogram("lepton1_eta", 50, -2.5, 2.5)
        book_histogram("lepton1_phi", 50, -3.2, 3.2)
        book_histogram("lepton2_e", 50, 0., 1500.)
        book_histogram("lepton2_eta", 50, -2.5, 2.5)
        book_histogram("lepton2_phi", 50, -3.2, 3.2)
        book_histogram("quark_e", 50, 0., 1500.)
        book_histogram("quark_eta", 50, -2.5, 2.5)
        book_histogram("quark_phi", 50, -3.2, 3.2)
        book_histogram("inv_mass", 100, 500, 4500)

    def book_plot_configs(self):
        def book_plot_config(name, xtitle, **kwargs):
            pc = PlotConfig(dist=None, name=name, xtitle=xtitle, ytitle="Entries", watermark="Simulation Internal",
                            color=ROOT.kBlue, draw="HIST", no_fill=True)
            for args, val in kwargs.iteritems():
                setattr(pc, name, val)
            self.plot_configs[name] = pc

        book_plot_config("resonance_counter_decay1", "pdg ID")
        book_plot_config("resonance_decay1_child_pdg_ids", "pdg ID")
        book_plot_config("lepton1_e", "lepton 1 E [GeV]")
        book_plot_config("lepton1_eta", "lepton1 #eta")
        book_plot_config("lepton1_phi", "lepton1 #phi")

        book_plot_config("lepton2_e", "lepton 2 E [GeV]")
        book_plot_config("lepton2_eta", "lepton 2 #eta")
        book_plot_config("lepton2_phi", "lepton 2 #phi")

        book_plot_config("quark_e", "quark E [GeV]")
        book_plot_config("quark_eta", "quark #eta")
        book_plot_config("quark_phi", "quark #phi")

        book_plot_config("inv_mass", "M_{lq} [GeV]")

    def run(self):
        for input_file in self.input_files:
            self.analyse_file(input_file)
        self.plot_histograms()
        self.output_handle.write_and_close()

    def build_references(self):
        def find_br(process, mode):
            decay2_mode = filter(lambda decay: sorted(decay[:-1], reverse=True) == mode, process.decay2_pdgid)[0]
            return decay2_mode[-1]

        for process_id, process in self.processes.iteritems():
            reference_hist = ROOT.TH1F("decay_mode_reference_{:d}".format(process_id), "", 4, -1.5, 2.5)
            reference_hist.SetBinContent(1, 0)
            for mode_index in range(len(process.decay2_sorted)):
                br = find_br(process, process.decay2_sorted[mode_index])
                reference_hist.SetBinContent(mode_index + 2, br)
            self.references[process_id] = reference_hist

    def plot_histograms(self):
        for process_id, histograms in self.histograms.iteritems():
            for hist_name, hist in histograms.iteritems():
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
        no_LQ_counter = 0
        #for entry in xrange(tree.GetEntries()):
        for entry in xrange(100):
            tree.GetEntry(entry)
            process_id = tree.EventInfo.runNumber()
            # if self.current_process_config is None:
            #     self.current_process_config = self.processes[process_id]
            truth_particles = tree.TruthParticles
            LQ = filter(lambda p: p.pdgId() == 1102, truth_particles)
            #self.histograms[process_id]["resonance_counter_decay1"].Fill(1)
            if len(LQ) == 0:
                no_LQ_counter += 1
                print "Suspicious event. Could not find LQ"
                continue
            resonance1_vertex = LQ[-1].decayVtxLink().outgoingParticleLinks()
            prod_vtx = LQ[0].prodVtxLink().outgoingParticleLinks()
            try:
                lepton_2 = filter(lambda particle: abs(particle.pdgId()) == 13, prod_vtx)[0]
                self.histograms[process_id]["lepton2_e"].Fill(lepton_2.e() / 1000.)
                self.histograms[process_id]["lepton2_eta"].Fill(lepton_2.eta())
                self.histograms[process_id]["lepton2_phi"].Fill(lepton_2.phi())
            except IndexError:
                print "Could not find any second lepton for first resonance decay in process", process_id
                continue
            try:
                lepton_1 = filter(lambda particle: abs(particle.pdgId()) == 13, resonance1_vertex)[0]
                quark_1 = filter(lambda particle: abs(particle.pdgId())in range(1, 6), resonance1_vertex)[0]
                self.histograms[process_id]["lepton1_e"].Fill(lepton_1.e() / 1000.)
                self.histograms[process_id]["lepton1_eta"].Fill(lepton_1.eta())
                self.histograms[process_id]["lepton1_phi"].Fill(lepton_1.phi())
                self.histograms[process_id]["quark_e"].Fill(quark_1.e() / 1000.)
                self.histograms[process_id]["quark_eta"].Fill(quark_1.eta())
                self.histograms[process_id]["quark_phi"].Fill(quark_1.phi())
    
                tlv_lepton_1 = ROOT.TLorentzVector()
                tlv_lepton_1.SetPtEtaPhiM(lepton_1.e(), lepton_1.eta(), lepton_1.phi(), lepton_1.m())
                tlv_quark_1 = ROOT.TLorentzVector()
                tlv_quark_1.SetPtEtaPhiM(quark_1.e(), quark_1.eta(), quark_1.phi(), quark_1.m())
                self.histograms[process_id]["inv_mass"].Fill((tlv_lepton_1 + tlv_quark_1).M() / 1000.)
            except IndexError:
                print "Could not find any first lepton for first resonance decay in process", process_id
                continue
        print "no LQ counter: ", no_LQ_counter


            # mode = list()
            # for resonance1_child in resonance1_vertex:
            #     self.histograms[process_id]["resonance_decay1_child_pdg_ids"].Fill(resonance1_child.pdgId())
            #     if abs(resonance1_child.pdgId() == 13):
            #         self.histograms[process_id]["muon_e"].Fill(resonance1_child.e() / 1000.)
            #         self.histograms[process_id]["muon_eta"].Fill(resonance1_child.eta())
            #         self.histograms[process_id]["muon_phi"].Fill(resonance1_child.phi())
            #     if resonance1_child.pdgId() not in self.current_process_config.decay_2_initial_resonance:
            #         continue
            #     mode.append((resonance1_child.pdgId(), resonance1_child.e() / 1000.))
            #     resonance2_vertex = resonance1_child.decayVtxLink().outgoingParticleLinks()
            #     for resonance2_child in resonance2_vertex:
            #         if abs(resonance2_child.pdgId()) == 13:
            #             self.histograms[process_id]["decay2_muon_e"].Fill(resonance2_child.e() / 1000.)
            #             self.histograms[process_id]["muon_e"].Fill(resonance2_child.e() / 1000.)
            #             self.histograms[process_id]["muon_eta"].Fill(resonance2_child.eta())
            #             self.histograms[process_id]["muon_phi"].Fill(resonance2_child.phi())
            #             muon_pts.append([resonance2_child.e() / 1000., resonance2_child.eta(), resonance2_child.phi()])
            #         if abs(resonance2_child.pdgId()) == 22:
            #             self.histograms[process_id]["decay2_gamma_e"].Fill(resonance2_child.e() / 1000.)
            #             self.histograms[process_id]["decay2_gamma_e_low_pt"].Fill(resonance2_child.e() / 1000.)
            #             self.histograms[process_id]["decay2_gamma_status"].Fill(resonance2_child.status())
            #         if not abs(resonance2_child.pdgId()) == 13:
            #             self.histograms[process_id]["decay2_non_muon_particle"].Fill(resonance2_child.pdgId())
            #         if resonance2_child.pdgId() == 22:
            #             all_photons_pts.append([resonance2_child.e() / 1000., resonance2_child.eta(),
            #                                     resonance2_child.phi()])
            #         if resonance2_child.pdgId() == 22 and resonance2_child.e() < 100.:
            #             continue
            #         # if resonance2_child.pdgId() == 22 and self.deltaR(muon_pts[0], [resonance2_child.e() / 1000.,
            #         #                                                                 resonance2_child.eta(),
            #         #                                                                 resonance2_child.phi()]) > 0.4:
            #         #     continue
            #         mode.append((resonance2_child.pdgId(), resonance2_child.e() / 1000.))
            #         self.histograms[process_id]["decay2_particle"].Fill(resonance2_child.pdgId())
            #         self.histograms[process_id]["decay2_particle_eta"].Fill(resonance2_child.eta())
            #         self.histograms[process_id]["decay2_particle_phi"].Fill(resonance2_child.phi())
            #         self.histograms[process_id]["decay2_particle_status"].Fill(resonance2_child.status())
            #         if abs(resonance2_child.pdgId()) == 22:
            #             self.histograms[process_id]["decay2_gamma_e_after_veto"].Fill(resonance2_child.e() / 1000.)
            #             self.histograms[process_id]["decay2_gamma_eta"].Fill(resonance2_child.eta())
            #             self.histograms[process_id]["decay2_gamma_phi"].Fill(resonance2_child.phi())
            #             self.histograms[process_id]["decay2_gamma_e_after_veto_low_pt"].Fill(resonance2_child.e() / 1000.)
            #             self.histograms[process_id]["decay2_gamma_status_after_veto"].Fill(resonance2_child.status())
            #             mother = resonance2_child.prodVtx().incomingParticleLinks()[0]
            #             self.histograms[process_id]["control_photon_mother_pdgid"].Fill(mother.pdgId())
            #             photons_pts.append([resonance2_child.e() / 1000., resonance2_child.eta(),
            #                                 resonance2_child.phi()])
            # mode.sort(key=lambda i: i[0], reverse=True)
            # try:
            #     decay_mode = self.current_process_config.decay2_sorted.index(map(lambda i: i[0], mode))
            # except ValueError:
            #     decay_mode = -1
            # if decay_mode == -1:
            #     for pdg_id, pt in mode:
            #         self.histograms[process_id]["unidentified_pdgid"].Fill(pdg_id)
            #         if pdg_id == 22:
            #             self.histograms[process_id]["unidentified_gamma_e"].Fill(pt)
            #
            # self.histograms[process_id]["decay2_mode"].Fill(decay_mode)
            # muon_pts.sort(key=lambda x: x[0], reverse=True)
            # if len(muon_pts) < 3:
            #     print "did not find 3 muons!"
            #     continue
            # self.histograms[process_id]["lead_muon_e"].Fill(muon_pts[0][0])
            # self.histograms[process_id]["sub_lead_muon_e"].Fill(muon_pts[1][0])
            # self.histograms[process_id]["third_lead_muon_e"].Fill(muon_pts[2][0])
            # for photon in photons_pts:
            #     for muon in muon_pts:
            #         self.histograms[process_id]["decay2_gamma_mu_dr_after_veto"].Fill(self.deltaR(photon, muon))
            # for photon in all_photons_pts:
            #     for muon in muon_pts:
            #         self.histograms[process_id]["decay2_gamma_mu_dr"].Fill(self.deltaR(photon, muon))
