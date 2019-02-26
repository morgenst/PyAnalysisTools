from collections import OrderedDict
from copy import deepcopy
from itertools import product


class Cut(object):
    def __init__(self, selection):
        self.is_data = False
        self.is_mc = False
        if '::' in selection:
            self.selection, self.name = selection.split('::')
        else:
            self.name = selection
            self.selection = selection
        if 'DATA:' in self.selection:
            self.selection.replace('Data:', '')
            self.is_data = True
        if 'MC:' in self.selection:
            self.selection.replace('MC:', '')
            self.is_mc = True

    def __eq__(self, other):
        """
        Comparison operator
        :param other: Cut object to compare to
        :type other: Cut
        :return: True/False
        :rtype: boolean
        """
        return self.__dict__ == other.__dict__

    def __eq__(self, other):
        """
        Comparison operator
        :param other: Cut object to compare to
        :type other: Cut
        :return: True/False
        :rtype: boolean
        """
        return self.__dict__ == other.__dict__

    def __str__(self):
        """
        Overloaded str operator. Get's called if object is printed
        :return: formatted string with name and attributes
        :rtype: str
        """
        obj_str = "Cut object named {:s} and selection {:s}".format(self.name, self.selection)
        return obj_str

    def __repr__(self):
        """
        Overloads representation operator. Get's called e.g. if list of objects are printed
        :return: formatted string with name and attributes
        :rtype: str
        """
        return self.__str__() + '\n'


class NewRegion(object):
    def __init__(self, **kwargs):
        kwargs.setdefault("n_lep", -1)
        kwargs.setdefault("n_electron", -1)
        kwargs.setdefault("n_muon", -1)
        # limit specific settings to help HistFactory setup
        kwargs.setdefault("norm_region", False)
        kwargs.setdefault("val_region", False)
        kwargs.setdefault("norm_backgrounds", [])

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
        kwargs.setdefault('label_position', None)
        kwargs.setdefault("good_muon", None)
        kwargs.setdefault("fake_muon", None)
        kwargs.setdefault("inverted_muon", None)
        kwargs.setdefault("good_electron", None)
        kwargs.setdefault("inverted_electron", None)
        kwargs.setdefault("event_cuts", None)
        kwargs.setdefault("split_mc_data", False)
        kwargs.setdefault("common_selection", None)
        kwargs.setdefault("weight", None)
        for k, v in kwargs.iteritems():
            setattr(self, k.lower(), v)
        if self.label is None:
            self.build_label()
        self.cut_list = []
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
        self.build_cuts()

    def __eq__(self, other):
        """
        Comparison operator
        :param other: Region object to compare to
        :type other: Region
        :return: True/False
        :rtype: boolean
        """
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        """
        Overloaded str operator. Get's called if object is printed
        :return: formatted string with name and attributes
        :rtype: str
        """
        obj_str = "Region: {:s} \n".format(self.name)
        for attribute, value in self.__dict__.items():
            if attribute == 'name':
                continue
            obj_str += '{}={} '.format(attribute, value)
        return obj_str

    def build_cuts(self):
        self.cut_list = self.build_cut_list(self.event_cuts, 'event_cuts')
        if self.disable_leptons:
            return
        self.convert_lepton_selections()

    def build_cut_list(self, cut_list, selection=None):
        tmp_cut_list = []

        if self.common_selection is not None and selection is not None:
            if selection in self.common_selection:
                tmp_cut_list += [Cut(sel) for sel in self.common_selection[selection]]
        if cut_list is not None:
            tmp_cut_list += [Cut(c) for c in cut_list]
        return tmp_cut_list

    def build_particle_cut(self, cut_list, selection, operator, particle, count):
        if cut_list is None:
            cut_list = []
        if self.common_selection is not None and selection is not None:
            if selection in self.common_selection:
                cut_list += self.common_selection[selection]
        return Cut('Sum$({:s}) == {:s} && {:s} {:s} {:d}'.format('&& '.join(cut_list), particle,
                                                                 particle, operator, count))

    def get_cut_list(self, is_data=False):
        """
        Retrieve cut list for region. Replace data/MC-only selections according to is_data flag
        :param is_data: flag if cut list should be retrieved for data or MC
        :type is_data: boolean
        :return: cut list
        :rtype: list
        """
        def validate_cut(cut):
            if cut.is_data and not is_data:
                cut = deepcopy(cut)
                cut.selection = '1'
            if cut.is_mc and is_data:
                cut = deepcopy(cut)
                cut.selection = '1'
            return cut
        return map(lambda c: validate_cut(c), self.cut_list)

    def convert_lepton_selections(self):
        """
        build lepton selection depending on available definitions for good (signal-like) and bad (background side-band)
        lepton definitions

        :return: None
        :rtype: None
        """
        if self.good_muon or self.common_selection and "good_muon" in self.common_selection:
            self.cut_list.append(self.build_particle_cut(self.good_muon, "good_muon", self.muon_operator,
                                                         'muon_n', self.n_muon))
        if self.fake_muon:
            self.inverted_muon_cut_string = self.convert_cut_list_to_string(self.inverted_muon)
        if self.good_electron or self.common_selection and "good_electron" in self.common_selection:
            self.cut_list.append(self.build_particle_cut(self.good_electron, "good_electron", self.electron_operator,
                                                         'electron_n', self.n_electron))
        if self.fake_muon:
            self.inverted_muon_cut_string = self.convert_cut_list_to_string(self.inverted_muon)

    def convert2cut_string(self):
        """
        Build cut string from configuration

        :return: cut selection as ROOT compatible string
        :rtype: string
        """
        if self.split_mc_data:
            self.split_mc_data = False
            self.convert_lepton_selections()
            self.event_cut_string = self.convert_cut_list_to_string(self.event_cuts)
        if self.disable_leptons:
            return self.event_cut_string

        cut_list = []
        electron_selector = "electron_n == electron_n"
        muon_selector = "muon_n == muon_n"
        if hasattr(self, "good_muon_cut_string"):
            muon_selector = "Sum$({:s}) == muon_n".format(self.good_muon_cut_string)
        if hasattr(self, "good_electron_cut_string"):
            electron_selector = "Sum$({:s}) == electron_n".format(self.good_electron_cut_string)

        cut = ""
        if hasattr(self, "event_cut_string"):
            cut_list.append(self.event_cut_string)
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

    def build_label(self):
        """
        Constructs optional label for region from number of leptons and

        :return:
        :rtype:
        """
        self.label = "".join([a * b for a, b in zip(["e^{#pm}", "#mu^{#pm}", "#tau^{#pm}"],
                                                    [self.n_electron, self.n_muon, self.n_tau])])
        if self.is_on_z is not None:
            self.label += " on-Z" if self.is_on_z else " off-Z"


class NewRegionBuilder(object):
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
        kwargs.setdefault("modify_mc_data_split", False)
        kwargs.setdefault("common_selection", None)
        if kwargs["auto_generate"]:
            self.auto_generate_region(**kwargs)

        if "regions" in kwargs:
            for region_name, region_def in kwargs["regions"].iteritems():
                if kwargs["modify_mc_data_split"]:
                    region_def["split_mc_data"] = False
                self.regions.append(NewRegion(name=region_name,
                                              common_selection=kwargs["common_selection"],
                                              **region_def))
        self.type = "PCModifier"

    def auto_generate_region(self, **kwargs):
        n_leptons = kwargs["nleptons"]
        for digits in product("".join(map(str, range(n_leptons + 1))), repeat=3):
            comb = map(int, digits)
            if sum(comb) == n_leptons:
                if kwargs["same_flavour_only"] and not comb.count(0) == 2:
                    continue
                name = "".join([a * b for a, b in zip(["e", "m", "t"], comb)])
                if kwargs["disable_taus"] and comb[2] > 0:
                    continue
                if kwargs["split_z_mass"]:
                    self.regions.append(NewRegion(name=name + "_onZ", n_lep=n_leptons, n_electron=comb[0], n_muon=comb[1],
                                               n_tau=comb[2], is_on_z=True, **kwargs))
                    self.regions.append(NewRegion(name=name + "_offZ", n_lep=n_leptons, n_electron=comb[0], n_muon=comb[1],
                                               n_tau=comb[2], is_on_z=False, **kwargs))
                else:
                    self.regions.append(NewRegion(name=name, n_lep=n_leptons, n_electron=comb[0], n_muon=comb[1],
                                               n_tau=comb[2], **kwargs))

    def build_custom_region(self):
        pass

    def modify_plot_configs(self, plot_configs):
        tmp = []
        print self.regions
        for region in self.regions:
            for pc in plot_configs:
                region_pc = deepcopy(pc)
                cuts = map(lambda c: c.selection, region.get_cut_list())
                if region_pc.cuts is None:
                    region_pc.cuts = cuts  # [region.convert2cut_string()]
                else:
                    region_pc.cuts += cuts  # .append(region.convert2cut_string())
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


class Region(NewRegion):
    def __init__(self, **kwargs):
        super(Region, self).__init__(**kwargs)
        return
        kwargs.setdefault("n_lep", -1)
        kwargs.setdefault("n_electron", -1)
        kwargs.setdefault("n_muon", -1)
        #limit specific settings to help HistFactory setup
        kwargs.setdefault("norm_region", False)
        kwargs.setdefault("val_region", False)
        kwargs.setdefault("norm_backgrounds", [])


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
        kwargs.setdefault("common_selection", None)
        kwargs.setdefault("weight", None)
        for k, v in kwargs.iteritems():
            setattr(self, k.lower(), v)
        if self.label is None:
            self.build_label()
        if self.event_cuts or self.common_selection and "event_cuts" in self.common_selection:
            self.event_cut_string = self.convert_cut_list_to_string(self.event_cuts, "event_cuts")
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
        raw_input('Deprecated. Please try to switch to NewRegionBuilder. Acknowledge by hitting enter')

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.name)

    def convert_cut_list_to_string(self, cut_list, selection=None):
        """
        Convert list of cuts into proper selection string which can be parsed by ROOT

        :param cut_list: list of cuts
        :type cut_list: list
        :return: selection string
        :rtype: string
        """
        new_cut_list = []
        if self.common_selection is not None and selection is not None:
            if selection in self.common_selection:
                new_cut_list += self.common_selection[selection]
        if cut_list is not None:
            new_cut_list += cut_list
        if not self.split_mc_data:
            return " && ".join(new_cut_list)
        return " && ".join(filter(lambda cut: "data:" not in cut.lower(), new_cut_list)), \
               " && ".join(new_cut_list).replace("Data:", "").replace("data:", "").replace("DATA:", "")

    def get_cut_list(self):
        cuts = []
        electron_selector = "electron_n == electron_n"
        muon_selector = "muon_n == muon_n"
        if hasattr(self, "good_muon_cut_string"):
            muon_selector = "Sum$({:s}) == muon_n".format(self.good_muon_cut_string)
        if hasattr(self, "good_electron_cut_string"):
            electron_selector = "Sum$({:s}) == electron_n".format(self.good_electron_cut_string)
        if not self.disable_muons:
            cuts.append("{:s} && muon_n {:s} {:d}".format(muon_selector, self.muon_operator, self.n_muon))
        if not self.disable_electrons:
            cuts.append(" {:s} && electron_n {:s} {:d}".format(electron_selector, self.operator, self.n_electron))
        if self.event_cuts and self.common_selection is None:
            return cuts + self.event_cuts
        if self.common_selection and "event_cuts" in self.common_selection:
            cuts += deepcopy(self.common_selection["event_cuts"])
            if self.event_cuts:
                cuts += self.event_cuts

            return cuts
        
    def convert_lepton_selections(self):
        """
        build lepton selection depending on available definitions for good (signal-like) and bad (background side-band)
        lepton definitions

        :return: None
        :rtype: None
        """
        if self.good_muon or self.common_selection and "good_muon" in self.common_selection:
            # good_muon = "muon_isolFixedCutTight == 1 && muon_is_prompt == 1 && abs(muon_d0sig) < 3"
            self.good_muon_cut_string = self.convert_cut_list_to_string(self.good_muon, "good_muon")
        if self.fake_muon:
            self.inverted_muon_cut_string = self.convert_cut_list_to_string(self.inverted_muon)
        if self.good_electron or self.common_selection and "good_electron" in self.common_selection:
            self.good_electron_cut_string = self.convert_cut_list_to_string(self.good_electron, "good_electron")
        if self.fake_muon:
            self.inverted_muon_cut_string = self.convert_cut_list_to_string(self.inverted_muon)

    def convert2cut_string(self):
        """
        Build cut string from configuration

        :return: cut selection as ROOT compatible string
        :rtype: string
        """
        if self.split_mc_data:
            self.split_mc_data = False
            self.convert_lepton_selections()
            self.event_cut_string = self.convert_cut_list_to_string(self.event_cuts)
        if self.disable_leptons:
            return self.event_cut_string

        cut_list = []
        electron_selector = "electron_n == electron_n"
        muon_selector = "muon_n == muon_n"
        if hasattr(self, "good_muon_cut_string"):
            muon_selector = "Sum$({:s}) == muon_n".format(self.good_muon_cut_string)
        if hasattr(self, "good_electron_cut_string"):
            electron_selector = "Sum$({:s}) == electron_n".format(self.good_electron_cut_string)

        cut = ""
        # if self.event_cuts is not None:
        #     cut_list += self.event_cuts
        # if self.event_cuts is None and hasattr(self, "event_cut_string"):
        #     cut_list.append(self.event_cut_string)
        if hasattr(self, "event_cut_string"):
            cut_list.append(self.event_cut_string)
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


class RegionBuilder(NewRegionBuilder):
    def __init__(self, **kwargs):
        super(RegionBuilder, self).__init__(**kwargs)
        return
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
        kwargs.setdefault("modify_mc_data_split", False)
        kwargs.setdefault("common_selection", None)
        if kwargs["auto_generate"]:
            self.auto_generate_region(**kwargs)

        if "regions" in kwargs:
            for region_name, region_def in kwargs["regions"].iteritems():
                if kwargs["modify_mc_data_split"]:
                    region_def["split_mc_data"] = False
                self.regions.append(Region(name=region_name, common_selection=kwargs["common_selection"], **region_def))
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
    #
    # def modify_plot_configs(self, plot_configs):
    #     tmp = []
    #     for region in self.regions:
    #         for pc in plot_configs:
    #             region_pc = deepcopy(pc)
    #             if region_pc.cuts is None:
    #                 region_pc.cuts = region.get_cut_list()#[region.convert2cut_string()]
    #             else:
    #                 region_pc.cuts += region.get_cut_list()#.append(region.convert2cut_string())
    #             if region.weight:
    #                 if isinstance(region.weight, OrderedDict):
    #                     region_pc.process_weight = region.weight
    #                 elif region_pc.weight is not None and not region_pc.weight.lower() == "none":
    #                     region_pc.weight += " * {:s}".format(region.weight)
    #                 else:
    #                     region_pc.weight = region.weight
    #
    #             region_pc.name = "{:s}_{:s}".format(region.name, pc.name)
    #             region_pc.decor_text = region.label
    #             region_pc.region = region
    #             tmp.append(region_pc)
    #     return tmp

    # def execute(self, plot_configs):
    #     return self.modify_plot_configs(plot_configs)
