from copy import deepcopy
from itertools import product


class Region(object):
    def __init__(self, **kwargs):
        kwargs.setdefault("n_lep", -1)
        kwargs.setdefault("n_electron", -1)
        kwargs.setdefault("n_muon", -1)
        self.name = kwargs["name"]
        self.n_lep = kwargs["n_lep"]
        self.n_electron = kwargs["n_electron"]
        self.n_muon = kwargs["n_muon"]
        try:
            self.n_tau = kwargs["n_tau"]
        except KeyError:
            self.n_tau = 0
        kwargs.setdefault("disable_leptons", False)
        kwargs.setdefault("disable_taus", False)
        kwargs.setdefault("disable_electrons", False)
        kwargs.setdefault("disable_muons", False)
        kwargs.setdefault("is_on_z", None)
        kwargs.setdefault("operator", "eq")
        kwargs.setdefault("muon_operator", "eq")
        kwargs.setdefault("label", None)
        kwargs.setdefault("good_muon", None)
        kwargs.setdefault("fake_muon", None)
        kwargs.setdefault("inverted_muon", None)
        kwargs.setdefault("good_electron", None)
        kwargs.setdefault("inverted_electron", None)
        kwargs.setdefault("event_cuts", None)
        kwargs.setdefault("split_mc_data", False)
        kwargs.setdefault("weight", None)
        for k, v in kwargs.iteritems():
            setattr(self, k.lower(), v)
        if self.label is None:
            self.build_label()
        if self.event_cuts:
            self.event_cut_string = self.convert_cut_list_to_string(self.event_cuts)
        if self.disable_leptons:
            return
        if kwargs["operator"] == "eq":
            self.operator = "=="
        elif kwargs["operator"] == "leq":
            self.operator = ">="
        if kwargs["muon_operator"] == "eq":
            self.muon_operator = "=="
        elif kwargs["muon_operator"] == "leq":
            self.muon_operator = ">="
        else:
            raise ValueError("Invalid operator provided. Currently supported: eq(==) and leq(>=)")
        if self.label is None:
            self.build_label()
        self.convert_lepton_selections()

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.name)

    def convert_cut_list_to_string(self, cut_list):
        """
        Convert list of cuts into proper selection string which can be parsed by ROOT

        :param cut_list: list of cuts
        :type cut_list: list
        :return: selection string
        :rtype: string
        """
        if not self.split_mc_data:
            return " && ".join(cut_list)
        return " && ".join(filter(lambda cut: "data:" not in cut.lower(), cut_list)), \
               " && ".join(cut_list).replace("Data:", "").replace("data:", "")

    def convert_lepton_selections(self):
        """
        build lepton selection depending on available definitions for good (signal-like) and bad (background side-band)
        lepton definitions

        :return: None
        :rtype: None
        """

        if self.good_muon:
            # good_muon = "muon_isolFixedCutTight == 1 && muon_is_prompt == 1 && abs(muon_d0sig) < 3"
            self.good_muon_cut_string = self.convert_cut_list_to_string(self.good_muon)
        if self.fake_muon:
            self.inverted_muon_cut_string = self.convert_cut_list_to_string(self.inverted_muon)
        if self.good_electron:
            self.good_electron_cut_string = self.convert_cut_list_to_string(self.good_electron)
        if self.fake_muon:
            self.inverted_muon_cut_string = self.convert_cut_list_to_string(self.inverted_muon)

    def convert2cut_string(self):
        """
        Build cut string from configuration

        :return: cut selection as ROOT compatible string
        :rtype: string
        """
        # electron_selector = "electron_n == electron_n"
        # muon_selector = "muon_n == muon_n"
        # if self.good_muon:
        #     muon_selector = "Sum$({:s}) == muon_n".format(self.good_muon_cut_string)
        # if self.good_electron:
        #     electron_selector = "Sum$({:s}) == electron_n".format(self.good_electron_cut_string)
        #
        # cut = ""
        # if self.event_cuts is not None:
        #     cut = self.event_cut_string + " && "
        # if self.is_on_z is not None:
        #     cut = "Sum$(inv_Z_mask==1) > 0 && " if self.is_on_z else "Sum$(inv_Z_mask==1) == 0 && "
        # if self.n_lep > sum([self.n_muon, self.n_electron, self.n_tau]):
        #     return cut + "{:s} + {:s} + tau_n {:s} {:d}".format(electron_selector, muon_selector, self.operator,
        #                                                        self.n_lep)
        # if not self.disable_muons:
        #     cut += "{:s} && muon_n {:s} {:d}".format(muon_selector, self.muon_operator, self.n_muon)
        # if not self.disable_electrons:
        #     cut += " {:s} && electron_n {:s} {:d}".format(electron_selector, self.operator, self.n_electron)
        # return cut
        if self.disable_leptons:
            return self.event_cut_string

        cut_list = []
        electron_selector = "electron_n == electron_n"
        muon_selector = "muon_n == muon_n"
        if self.good_muon:
            muon_selector = "Sum$({:s}) == muon_n".format(self.good_muon_cut_string)
        if self.good_electron:
            electron_selector = "Sum$({:s}) == electron_n".format(self.good_electron_cut_string)

        cut = ""
        if self.event_cuts is not None:
            cut_list += self.event_cuts
        if self.is_on_z is not None:
            cut_list.append("Sum$(inv_Z_mask==1) > 0" if self.is_on_z else "Sum$(inv_Z_mask==1) == 0")
        if self.n_lep > sum([self.n_muon, self.n_electron, self.n_tau]):
            return cut + "{:s} + {:s} + tau_n {:s} {:d}".format(electron_selector, muon_selector, self.operator,
                                                               self.n_lep)
        if not self.disable_muons:
            cut_list.append("{:s} && muon_n {:s} {:d}".format(muon_selector, self.muon_operator, self.n_muon))
        if not self.disable_electrons:
            cut_list.append(" {:s} && electron_n {:s} {:d}".format(electron_selector, self.operator, self.n_electron))
        return " && ".join(cut_list)



        # if self.disable_taus:
        #     return cut + "{:s} && electron_n {:s} {:d} && {:s} && muon_n {:s} {:d}".format(electron_selector,
        #                                                                                    self.operator,
        #                                                                                    self.n_electron,
        #                                                                                    muon_selector,
        #                                                                                    self.muon_operator,
        #                                                                                    self.n_muon)

        return cut + "{:s} {:s} {:d} && {:s} {:s} {:d} && tau_n {:s} {:d}".format(electron_selector, self.operator,
                                                                                  self.n_electron, muon_selector,
                                                                                  self.muon_operator, self.n_muon,
                                                                                  self.operator, self.n_tau)

    def build_label(self):
        """
        Constructs optional label for region from number of leptons and

        :return:
        :rtype:
        """
        self.label = "".join([a*b for a, b in zip(["e^{#pm}", "#mu^{#pm}", "#tau^{#pm}"],
                                                  [self.n_electron, self.n_muon, self.n_tau])])
        if self.is_on_z is not None:
            self.label += " on-Z" if self.is_on_z else " off-Z"


class RegionBuilder(object):
    def __init__(self, **kwargs):
        """
        contructor

        :param kwargs: see below
        :type kwargs:

        :Keyword Arguments:
            * *auto_generate* (bool): enable automatic generation of regions based on possible combinations.
        """

        self.regions = []
        kwargs.setdefault("auto_generate", False)
        kwargs.setdefault("disable_taus", False)
        kwargs.setdefault("split_z_mass", False)
        kwargs.setdefault("same_flavour_only", False)
        if kwargs["auto_generate"]:
            # self.auto_generate_region(kwargs["nleptons"], kwargs["disable_taus"], kwargs["split_z_mass"],
            #                           kwargs["same_flavour_only"])
            self.auto_generate_region(**kwargs)
        if "regions" in kwargs:
            for region_name, region_def in kwargs["regions"].iteritems():
                self.regions.append(Region(name=region_name, **region_def))
        self.type = "PCModifier"

    def auto_generate_region(self, **kwargs):
        n_leptons = kwargs["nleptons"]
        for digits in product("".join(map(str, range(n_leptons+1))), repeat=3):
            comb = map(int, digits)
            if sum(comb) == n_leptons:
                if kwargs["same_flavour_only"] and not comb.count(0) == 2:
                    continue
                name = "".join([a*b for a, b in zip(["e", "m", "t"], comb)])
                if kwargs["disable_taus"] and comb[2] > 0:
                    continue
                if kwargs["split_z_mass"]:
                    self.regions.append(Region(name=name + "_onZ", n_lep=n_leptons, n_electron=comb[0], n_muon=comb[1],
                                               n_tau=comb[2], is_on_z=True, **kwargs))
                    self.regions.append(Region(name=name + "_offZ", n_lep=n_leptons, n_electron=comb[0], n_muon=comb[1],
                                               n_tau=comb[2], is_on_z=False, **kwargs))
                else:
                    self.regions.append(Region(name=name, n_lep=n_leptons, n_electron=comb[0], n_muon=comb[1],
                                               n_tau=comb[2], **kwargs))

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
                if region.weight:
                    if region_pc.weight is not None and not region_pc.weight.lower() == "none":
                        region_pc.weight += " * {:s}".format(region.weight)
                    else:
                        region_pc.weight = region.weight

                region_pc.name = "{:s}_{:s}".format(region.name, pc.name)
                region_pc.decor_text = region.label
                region_pc.region = region
                tmp.append(region_pc)
        return tmp

    def execute(self, plot_configs):
        return self.modify_plot_configs(plot_configs)
