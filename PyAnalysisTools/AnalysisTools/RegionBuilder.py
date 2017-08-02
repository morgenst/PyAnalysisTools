from copy import deepcopy
from itertools import product


class Region(object):
    def __init__(self, *args):
        self.name = args[0]
        self.n_lep = args[1]
        self.n_electron = args[2]
        self.n_muon = args[3]
        self.n_tau = args[4]

    def convert2cut_string(self):
        return "electron_n == {:d} && muon_n == {:d} && tau_n == {:d}".format(self.n_electron,
                                                                              self.n_muon,
                                                                              self.n_tau)

    def convert2cut_decor_string(self):
        return "".join([a*b for a, b in zip(["e^{#pm}", "#mu^{#pm}", "#tau^{#pm}"],
                                            [self.n_electron, self.n_muon, self.n_tau])])


class RegionBuilder(object):
    def __init__(self):
        self.regions = []
        self.auto_generate_region(3)

    def auto_generate_region(self, n_leptons):
        for digits in product('0123', repeat=n_leptons):
            comb = map(int, digits)
            if sum(comb) == 3:
                name = "".join([a*b for a, b in zip(["e", "m", "t"], comb)])
                self.regions.append(Region(name, n_leptons, *comb))

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
                print "decor string: ", region_pc.decor_text
                tmp.append(region_pc)
        return tmp

    def execute(self, plot_configs):
        return self.modify_plot_configs(plot_configs)