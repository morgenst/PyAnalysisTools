from __future__ import print_function

import re
from copy import deepcopy
from itertools import product

from builtins import map
from builtins import object
from builtins import range
from builtins import zip


class Cut(object):
    def __init__(self, selection):
        self.is_data = False
        self.is_mc = False
        self.process_type = None
        if ':::' in selection:
            self.selection, self.name = selection.split(':::')
        else:
            self.name = selection
            self.selection = selection
        if 'DATA:' in self.selection:
            self.selection = self.selection.replace('DATA:', '')
            self.is_data = True
        if 'MC:' in self.selection:
            self.selection = self.selection.replace('MC:', '')
            self.is_mc = True
        elif re.match(r'TYPE_[A-Z].*:', self.selection):
            process_type = re.match(r'TYPE_[A-Z].*:', self.selection).group(0)
            self.selection = self.selection.replace(process_type, '')
            self.process_type = process_type.replace('TYPE_', '').rstrip(':').lower()

    def __eq__(self, other):
        """
        Comparison operator
        :param other: Cut object to compare to
        :type other: Cut
        :return: True/False
        :rtype: boolean
        """
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return False

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


class Region(object):
    def __init__(self, **kwargs):
        kwargs.setdefault("n_lep", -1)
        kwargs.setdefault("n_electron", -1)
        kwargs.setdefault("n_muon", -1)
        # limit specific settings to help HistFactory setup
        kwargs.setdefault("norm_region", False)
        kwargs.setdefault("val_region", False)
        kwargs.setdefault("norm_backgrounds", {})

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
        kwargs.setdefault("electron_operator", "eq")
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
        kwargs.setdefault("binning", None)
        for k, v in list(kwargs.items()):
            setattr(self, k.lower(), v)
        if self.label is None:
            self.build_label()
        self.cut_list = []
        self.parse_operators('operator', kwargs)
        self.parse_operators('muon_operator', kwargs)
        self.parse_operators('electron_operator', kwargs)
        self.build_cuts()

    def parse_operators(self, name, kwargs):
        kwargs.setdefault(name, 'eq')
        if kwargs[name] == "eq":
            setattr(self, name, '==')
        elif kwargs[name] == 'geq':
            setattr(self, name, '>=')
        elif kwargs[name] == 'leq':
            setattr(self, name, '<=')
        else:
            raise ValueError("Invalid operator provided for {:s}. "
                             "Currently supported: eq(==), geq(>=) and leq(<=)".format(name))

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
        for attribute, value in list(self.__dict__.items()):
            if attribute == 'name':
                continue
            obj_str += '{}={} '.format(attribute, value)
        return obj_str

    def __repr__(self):
        """
        Overloads representation operator. Get's called e.g. if list of objects are printed
        :return: formatted string with name and attributes
        :rtype: str
        """
        return self.__str__() + '\n'

    def build_cuts(self):
        self.cut_list = self.build_cut_list(self.event_cuts, 'event_cuts')
        if not self.disable_leptons:
            self.convert_lepton_selections()
        self.cut_list += self.build_cut_list(None, 'post_sel_cuts')

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
        if not cut_list:
            return Cut('{:s} {:s} {:d}'.format(particle, operator, count))
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
        return [validate_cut(c) for c in self.cut_list]

    def convert_lepton_selections(self):
        """
        build lepton selection depending on available definitions for good (signal-like) and bad (background side-band)
        lepton definitions

        :return: None
        :rtype: None
        """
        # found_particle_cut = False
        if self.good_muon or self.common_selection and "good_muon" in self.common_selection:
            self.cut_list.append(self.build_particle_cut(self.good_muon, "good_muon", self.muon_operator,
                                                         'muon_n', self.n_muon))
            # found_particle_cut = True
        if self.good_electron or self.common_selection and "good_electron" in self.common_selection:
            self.cut_list.append(self.build_particle_cut(self.good_electron, "good_electron", self.electron_operator,
                                                         'electron_n', self.n_electron))
            # found_particle_cut = True
        if self.fake_muon:
            self.inverted_muon_cut_string = self.convert_cut_list_to_string(self.inverted_muon)
        # if not found_particle_cut and self.n_electron > 0 or self.n_muon > 0:
        #     self.cut_list.append(['Sum$({:s}) == {:s}')

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
        kwargs.setdefault("modify_mc_data_split", False)
        kwargs.setdefault("common_selection", None)
        if kwargs["auto_generate"]:
            self.auto_generate_region(**kwargs)

        if "regions" in kwargs:
            for region_name, region_def in list(kwargs["regions"].items()):
                if kwargs["modify_mc_data_split"]:
                    region_def["split_mc_data"] = False
                self.regions.append(Region(name=region_name,
                                           common_selection=kwargs["common_selection"],
                                           **region_def))
        self.type = "PCModifier"

    def __str__(self):
        """
        Overloaded str operator. Get's called if object is printed
        :return: formatted string for all regions
        :rtype: str
        """
        print(self.regions)

    def auto_generate_region(self, **kwargs):
        n_leptons = kwargs["nleptons"]
        for digits in product("".join(map(str, list(range(n_leptons + 1)))), repeat=3):
            comb = list(map(int, digits))
            if sum(comb) == n_leptons:
                if kwargs["same_flavour_only"] and not comb.count(0) == 2:
                    continue
                name = "".join([a * b for a, b in zip(["e", "m", "t"], comb)])
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

    def modify_plot_configs(self, plot_configs):
        tmp = []
        for region in self.regions:
            for pc in plot_configs:
                region_pc = deepcopy(pc)
                cuts = [c.selection for c in region.get_cut_list()]
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
