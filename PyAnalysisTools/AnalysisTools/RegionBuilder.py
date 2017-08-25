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

    def convert2cut_string(self):
        cut = ""
        if self.is_on_z is not None:
            cut = "Sum$(inv_Z_mask==1) > 0 && " if self.is_on_z else "Sum$(inv_Z_mask==1) == 0 && "
        if self.disable_taus:
            return cut + "electron_prompt_n {:s} {:d} && muon_prompt_n {:s} {:d}".format(self.operator, self.n_electron,
                                                                                         self.operator, self.n_muon)
        if self.n_lep > sum([self.n_muon, self.n_electron, self.n_tau]):
            return cut + "electron_prompt_n + muon_prompt_n + tau_n {:s} {:d}".format(self.operator,
                                                                                                self.n_lep)
        return cut + "electron_prompt_n {:s} {:d} && muon_prompt_n {:s} {:d} && tau_n {:s} {:d}".format(self.operator,
                                                                                                               self.n_electron,
                                                                                                               self.operator,
                                                                                                               self.n_muon,
                                                                                                               self.operator,
                                                                                                               self.n_tau)

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
        if kwargs["auto_generate"]:
            self.auto_generate_region(kwargs["nleptons"], kwargs["disable_taus"], kwargs["split_z_mass"])
        if "regions" in kwargs:
            for region_name, region_def in kwargs["regions"].iteritems():
                self.regions.append(Region(name=region_name, **region_def))
        self.type = "PCModifier"

    def auto_generate_region(self, n_leptons, disable_taus, split_z_mass):
        for digits in product("".join(map(str, range(n_leptons+1))), repeat=n_leptons):
            comb = map(int, digits)
            if sum(comb) == n_leptons:
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
