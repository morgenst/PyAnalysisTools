from copy import deepcopy
from itertools import product


class Region(object):
    def __init__(self, **kwargs):
        self.name = kwargs["name"]
        self.n_lep = kwargs["n_lep"]
        self.n_electron = kwargs["n_electron"]
        self.n_muon = kwargs["n_muon"]
        try:
            self.n_tau = kwargs["n_tau"]
        except KeyError:
            self.n_tau = 0
        kwargs.setdefault("disable_taus", False)
        kwargs.setdefault("is_on_z", None)
        kwargs.setdefault("operator", "eq")
        kwargs.setdefault("label", None)
        kwargs.setdefault("good_muon", None)
        kwargs.setdefault("fake_muon", None)
        kwargs.setdefault("inverted_muon", None)
        kwargs.setdefault("good_electron", None)
        kwargs.setdefault("inverted_electron", None)
        kwargs.setdefault("event_cuts", None)
        for k, v in kwargs.iteritems():
            setattr(self, k.lower(), v)
        if kwargs["operator"] == "eq":
            self.operator = "=="
        elif kwargs["operator"] == "leq":
            self.operator = ">="
        else:
            raise ValueError("Invalid operator provided. Currently supported: eq(==) and leq(>=)")
        if self.label is None:
            self.build_label()
        self.convert_lepton_selections()

    def convert_lepton_selections(self):
        def convert_cut_list_to_string(cut_list):
            return " && ".join(cut_list)

        if self.good_muon:
            # good_muon = "muon_isolFixedCutTight == 1 && muon_is_prompt == 1 && abs(muon_d0sig) < 3"
            self.good_muon_cut_string = convert_cut_list_to_string(self.good_muon)
        if self.fake_muon:
            self.inverted_muon_cut_string = convert_cut_list_to_string(self.inverted_muon)
        if self.good_electron:
            self.good_electron_cut_string = convert_cut_list_to_string(self.good_electron)
        if self.fake_muon:
            self.inverted_muon_cut_string = convert_cut_list_to_string(self.inverted_muon)
        if self.event_cuts:
            self.event_cut_string = convert_cut_list_to_string(self.event_cuts)

    def convert2cut_string(self):
        electron_selector = "electron_n == electron_n"
        muon_selector = "muon_n == muon_n"
        if self.good_muon:
            muon_selector = "Sum$({:s}) == muon_n".format(self.good_muon_cut_string)
        if self.good_electron:
            electron_selector = "Sum$({:s}) == electron_n".format(self.good_electron_cut_string)

        cut = ""
        if self.event_cuts is not None:
            cut = self.event_cut_string + " && "
        if self.is_on_z is not None:
            cut = "Sum$(inv_Z_mask==1) > 0 && " if self.is_on_z else "Sum$(inv_Z_mask==1) == 0 && "
        if self.disable_taus:
            return cut + "{:s} && electron_n {:s} {:d} && {:s} && muon_n {:s} {:d}".format(electron_selector,
                                                                                           self.operator,
                                                                                           self.n_electron,
                                                                                           muon_selector,
                                                                                           self.operator,
                                                                                           self.n_muon)
        if self.n_lep > sum([self.n_muon, self.n_electron, self.n_tau]):
            return cut + "{:s} + {:s} + tau_n {:s} {:d}".format(electron_selector, muon_selector, self.operator,
                                                               self.n_lep)
        return cut + "{:s} {:s} {:d} && {:s} {:s} {:d} && tau_n {:s} {:d}".format(electron_selector, self.operator,
                                                                                  self.n_electron, muon_selector,
                                                                                  self.operator, self.n_muon,
                                                                                  self.operator, self.n_tau)

    def build_label(self):
        self.label = "".join([a*b for a, b in zip(["e^{#pm}", "#mu^{#pm}", "#tau^{#pm}"],
                                                  [self.n_electron, self.n_muon, self.n_tau])])
        if self.is_on_z is not None:
            self.label += " on-Z" if self.is_on_z else " off-Z"


class RegionBuilder(object):
    def __init__(self, **kwargs):
        self.regions = []
        kwargs.setdefault("auto_generate", False)
        kwargs.setdefault("disable_taus", False)
        kwargs.setdefault("split_z_mass", False)
        kwargs.setdefault("same_flavour_only", False)
        if kwargs["auto_generate"]:
            self.auto_generate_region(kwargs["nleptons"], kwargs["disable_taus"], kwargs["split_z_mass"],
                                      kwargs["same_flavour_only"])
        if "regions" in kwargs:
            for region_name, region_def in kwargs["regions"].iteritems():
                self.regions.append(Region(name=region_name, **region_def))
        self.type = "PCModifier"

    def auto_generate_region(self, n_leptons, disable_taus, split_z_mass, same_flavour_only):
        for digits in product("".join(map(str, range(n_leptons+1))), repeat=3):
            comb = map(int, digits)
            if sum(comb) == n_leptons:
                if same_flavour_only and not comb.count(0) == 2:
                    continue
                name = "".join([a*b for a, b in zip(["e", "m", "t"], comb)])
                if disable_taus and comb[2] > 0:
                    continue
                if split_z_mass:
                    self.regions.append(Region(name=name + "_onZ", n_lep=n_leptons, n_electron=comb[0], n_muon=comb[1],
                                               n_tau=comb[2], disable_taus=disable_taus, is_on_z=True))
                    self.regions.append(Region(name=name + "_offZ", n_lep=n_leptons, n_electron=comb[0], n_muon=comb[1],
                                               n_tau=comb[2], disable_taus=disable_taus, is_on_z=False))
                else:
                    self.regions.append(Region(name=name, n_lep=n_leptons, n_electron=comb[0], n_muon=comb[1],
                                               n_tau=comb[2], disable_taus=disable_taus))

    def build_custom_region(self):
        pass

    def modify_plot_configs(self, plot_configs):
        tmp = []
        for region in self.regions:
            for pc in plot_configs:
                region_pc = deepcopy(pc)
                if region_pc.cuts is None:
                    region_pc.cuts = [region.convert2cut_string()]
                else:
                    region_pc.cuts.append(region.convert2cut_string())
                region_pc.name = "{:s}_{:s}".format(region.name, pc.name)
                region_pc.decor_text = region.label
                region_pc.region = region
                tmp.append(region_pc)
        return tmp

    def execute(self, plot_configs):
        return self.modify_plot_configs(plot_configs)
