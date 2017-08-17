from copy import deepcopy
from itertools import product


class Region(object):
    def __init__(self, *args, **kwargs):
        self.name = args[0]
        self.n_lep = args[1]
        self.n_electron = args[2]
        self.n_muon = args[3]
        try:
            self.n_tau = args[4]
        except IndexError:
            self.n_tau = 0
        kwargs.setdefault("disable_taus", False)
        for k, v in kwargs.iteritems():
            setattr(self, k.lower(), v)

    def convert2cut_string(self):
        if self.disable_taus:
            return "electron_prompt_n == {:d} && muon_prompt_n == {:d}".format(self.n_electron, self.n_muon)
        return "electron_prompt_n == {:d} && muon_prompt_n == {:d} && tau_prompt_n == {:d}".format(self.n_electron,
                                                                                                   self.n_muon,
                                                                                                   self.n_tau)

    def convert2cut_decor_string(self):
        return "".join([a*b for a, b in zip(["e^{#pm}", "#mu^{#pm}", "#tau^{#pm}"],
                                            [self.n_electron, self.n_muon, self.n_tau])])


class RegionBuilder(object):
    def __init__(self, **kwargs):
        self.regions = []
        kwargs.setdefault("auto_generate", False)
        kwargs.setdefault("disable_taus", False)
        if kwargs["auto_generate"]:
            self.auto_generate_region(kwargs["nleptons"], kwargs["disable_taus"])
        self.type = "PCModifier"

    def auto_generate_region(self, n_leptons, disable_taus):
        for digits in product("".join(map(str, range(n_leptons+1))), repeat=n_leptons):
            comb = map(int, digits)
            if sum(comb) == n_leptons:
                name = "".join([a*b for a, b in zip(["e", "m", "t"], comb)])
                self.regions.append(Region(name, n_leptons, *comb, disable_taus=disable_taus))

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
                region_pc.decor_text = region.convert2cut_decor_string()
                region_pc.region = region
                tmp.append(region_pc)
        return tmp

    def execute(self, plot_configs):
        return self.modify_plot_configs(plot_configs)
