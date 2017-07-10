from PyAnalysisTools.base.YAMLHandle import YAMLLoader
from PyAnalysisTools.base.OutputHandle import OutputFileHandle
from PyAnalysisTools.PlottingUtils import PlottingTools as PT
from PyAnalysisTools.PlottingUtils.PlotConfig import PlotConfig
from copy import copy
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
particle_map["pi0"] = [111, "#pi"]
particle_map["anti_mu_nu"] = [14, "#nu_{#mu}"]
particle_map["mu_nu"] = [-14, "#nu_{#mu}"]
particle_map["gamma"] = [22, "#gamma"]
particle_map["rho0"] = [113, "#rho"]
particle_map["eta"] = [221, "#eta"]
particle_map["omega"] = [223, "#omega"]


class Process(object):
    def __init__(self, config):
        self.decay1_str = config["decay1"]
        self.decay2 = config["decay2"]
        self.decay1_pdgid = map(lambda name: particle_map[name][0], self.decay1_str)
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
        for process_id in self.processes.keys():
            if process_id not in self.histograms:
                self.histograms[process_id] = dict()
            self.histograms[process_id]["resonance_counter_decay1"] = ROOT.TH1F("resonance_counter_decay1_{:d}".format(process_id),
                                                                                "", 2, -0.5, 1.5)
            self.histograms[process_id]["resonance_decay1_child_pdg_ids"] = ROOT.TH1F("resonance_decay1_child_pdg_ids_{:d}".format(process_id),
                                                                                      "", 501, -500.5, 500.5)
            self.histograms[process_id]["decay2_mode"] = ROOT.TH1F("hist_decay2_mode_{:d}".format(process_id), "", 4, -1.5, 2.5)
            self.histograms[process_id]["muon_e"] = ROOT.TH1F("hist_muon_e_{:d}".format(process_id), "", 20, 0., 50.)
            self.histograms[process_id]["lead_muon_e"] = ROOT.TH1F("lead_muon_e_{:d}".format(process_id), "", 20, 0., 20.)
            self.histograms[process_id]["sub_lead_muon_e"] = ROOT.TH1F("sub_lead_muon_e_{:d}".format(process_id), "", 20, 0., 20.)
            self.histograms[process_id]["third_lead_muon_e"] = ROOT.TH1F("third_lead_muon_e_{:d}".format(process_id), "", 20, 0., 20.)
            self.histograms[process_id]["gamma_e"] = ROOT.TH1F("hist_gamma_e_{:d}".format(process_id), "", 20, 0., 50.)
            self.histograms[process_id]["decay2_third_particle"] = ROOT.TH1F("decay2_third_particle_{:d}".format(process_id), "", 501, -499.5, 499.5)
            self.histograms[process_id]["decay2_non_muon_particle"] = ROOT.TH1F("decay2_non_muon_particle_{:d}".format(process_id), "", 501, -499.5, 499.5)
            self.histograms[process_id]["decay2_child_pdgid"] = ROOT.TH1F("decay2_child_pdgid_{:d}".format(process_id), "", 501, -499.5, 499.5)
            self.histograms[process_id]["decay2_muon_e"] = ROOT.TH1F("decay2_muon_e_{:d}".format(process_id), "", 20, 0., 20.)
            self.histograms[process_id]["decay2_gamma_e"] = ROOT.TH1F("decay2_gamma_e_{:d}".format(process_id), "", 20, 0., 20.)
            self.histograms[process_id]["control_photon_mother_pdgid"] = ROOT.TH1F("control_photon_mother_pdgid_{:d}".format(process_id), "", 239, -119.5, 119.5)

    def book_plot_configs(self):
        pc = PlotConfig(dist=None, name="", xtitle="", ytitle="Entries", watermark="Simulation Internal",
                        color=ROOT.kBlue)
        pc.name = "resonance_counter_decay1"
        pc.xtitle = "pdg ID"
        self.plot_configs["resonance_counter_decay1"] = copy(pc)
        pc.name = "resonance_decay1_child_pdg_ids"
        pc.xtitle = "pdg ID"
        self.plot_configs["resonance_decay1_child_pdg_ids"] = copy(pc)
        pc.name = "decay2_mode"
        pc.xtitle = "decay mode"
        pc.ymax = 1.1
        self.plot_configs["decay2_mode"] = copy(pc)
        pc.name = "muon_e"
        pc.xtitle = "all #mu E [GeV]"
        self.plot_configs["muon_e"] = copy(pc)
        pc.name = "lead_muon_e"
        pc.xtitle = "lead. #mu E [GeV]"
        self.plot_configs["lead_muon_e"] = copy(pc)
        pc.name = "sub_lead_muon_e"
        pc.xtitle = "sub-lead. #mu E [GeV]"
        self.plot_configs["sub_lead_muon_e"] = copy(pc)
        pc.name = "third_lead_muon_e"
        pc.xtitle = "third-lead. #mu E [GeV]"
        self.plot_configs["third_lead_muon_e"] = copy(pc)
        pc.name = "gamma_e"
        pc.xtitle = "#gamma E [GeV]"
        self.plot_configs["gamma_e"] = copy(pc)
        pc.name = "decay2_third_particle"
        pc.xtitle = "3^{rd} particle pdg ID"
        self.plot_configs["decay2_third_particle"] = copy(pc)
        pc.name = "decay2_non_muon_particle"
        pc.xtitle = "non #mu pdg ID"
        self.plot_configs["decay2_non_muon_particle"] = copy(pc)
        pc.name = "decay2_child_pdgid"
        pc.xtitle = "decay2 child pdg IDs"
        self.plot_configs["decay2_child_pdgid"] = copy(pc)
        pc.name = "decay2_muon_e"
        pc.xtitle = "decay 2 #mu E [GeV]"
        self.plot_configs["decay2_muon_e"] = copy(pc)
        pc.name = "decay2_gamma_e"
        pc.xtitle = "decay 2 #gamma E [GeV]"
        self.plot_configs["decay2_gamma_e"] = copy(pc)
        pc.name = "control_photon_mother_pdgid"
        pc.xtitle = "#gamma mother pdgID"
        self.plot_configs["control_photon_mother_pdgid"] = copy(pc)
        
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
                    pc.labels = self.processes[process_id].get_bin_labels()
                    pc.normalise = True
                canvas = PT.plot_obj(hist, pc)
                if hist_name == "decay2_mode":
                    pc_ref = copy(pc)
                    pc_ref.color = ROOT.kRed
                    PT.add_histogram_to_canvas(canvas, self.references[process_id], pc_ref)
                FT.decorate_canvas(canvas, pc)
                self.output_handle.register_object(canvas, str(process_id))

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
            try:
                muon_pts.append(filter(lambda particle: abs(particle.pdgId()) == 13, resonance1_vertex)[0].e() / 1000.)
            except IndexError:
                print "Could not find any muon for first resonance decay in process", process_id
                continue
            mode = list()
            for resonance1_child in resonance1_vertex:
                self.histograms[process_id]["resonance_decay1_child_pdg_ids"].Fill(resonance1_child.pdgId())
                if abs(resonance1_child.pdgId() == 13):
                    self.histograms[process_id]["muon_e"].Fill(resonance1_child.e() / 1000.)
                if resonance1_child.pdgId() not in self.current_process_config.decay_2_initial_resonance:
                    continue
                mode.append(resonance1_child.pdgId())
                resonance2_vertex = resonance1_child.decayVtxLink().outgoingParticleLinks()
                for resonance2_child in resonance2_vertex:
                    if abs(resonance2_child.pdgId()) == 13:
                        self.histograms[process_id]["decay2_muon_e"].Fill(resonance2_child.e() / 1000.)
                        self.histograms[process_id]["muon_e"].Fill(resonance2_child.e() / 1000.)
                        muon_pts.append(resonance2_child.e() / 1000.)
                    if abs(resonance2_child.pdgId()) == 22:
                        self.histograms[process_id]["decay2_gamma_e"].Fill(resonance2_child.e() / 1000.)
                    if not abs(resonance2_child.pdgId()) == 13:
                        self.histograms[process_id]["decay2_non_muon_particle"].Fill(resonance2_child.pdgId())
                    if resonance2_child.e() < 1.:
                        continue
                    mode.append(resonance2_child.pdgId())
                    self.histograms[process_id]["decay2_third_particle"].Fill(resonance2_child.pdgId())
            mode.sort(reverse=True)
            try:
                decay_mode = self.current_process_config.decay2_sorted.index(mode)
            except ValueError:
                decay_mode = -1
            self.histograms[process_id]["decay2_mode"].Fill(decay_mode)
            muon_pts.sort(reverse=True)
            self.histograms[process_id]["lead_muon_e"].Fill(muon_pts[0])
            self.histograms[process_id]["sub_lead_muon_e"].Fill(muon_pts[1])
            self.histograms[process_id]["third_lead_muon_e"].Fill(muon_pts[2])
        f.Close()
