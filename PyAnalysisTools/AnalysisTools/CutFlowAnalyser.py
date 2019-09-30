from __future__ import print_function

import operator
import os
import re

import numpy as np
import pandas as pd
from builtins import input
from builtins import map
from builtins import object
from builtins import range

import PyAnalysisTools.PlottingUtils.Formatting as Ft
import PyAnalysisTools.PlottingUtils.PlottingTools as Pt
from PyAnalysisTools.base.Modules import load_modules
from PyAnalysisTools.base.ProcessConfig import Process

try:
    from tabulate.tabulate import tabulate
except ImportError:
    from tabulate import tabulate
tabulate.LATEX_ESCAPE_RULES = {}

from collections import defaultdict, OrderedDict
from PyAnalysisTools.base import _logger, InvalidInputError
from PyAnalysisTools.base.FileHandle import FileHandle as FH
from PyAnalysisTools.PlottingUtils.HistTools import scale
from PyAnalysisTools.AnalysisTools.XSHandle import XSHandle, get_xsec_weight
from PyAnalysisTools.PlottingUtils.PlotConfig import parse_and_build_process_config, find_process_config, PlotConfig, \
    parse_and_build_plot_config, get_default_color_scheme
from PyAnalysisTools.base.OutputHandle import OutputFileHandle
from PyAnalysisTools.PlottingUtils.Plotter import Plotter as pl
from numpy.lib.recfunctions import rec_append_fields
from PyAnalysisTools.AnalysisTools.RegionBuilder import RegionBuilder
from PyAnalysisTools.base.YAMLHandle import YAMLLoader
from PyAnalysisTools.AnalysisTools.MLHelper import Root2NumpyConverter
from PyAnalysisTools.AnalysisTools.StatisticsTools import get_signal_acceptance
from PyAnalysisTools.PlottingUtils import set_batch_mode


class CommonCutFlowAnalyser(object):
    def __init__(self, **kwargs):
        kwargs.setdefault('lumi', None)
        kwargs.setdefault('process_configs', None)
        kwargs.setdefault('disable_sm_total', False)
        kwargs.setdefault('plot_config_file', None)
        kwargs.setdefault('config_file', None)
        kwargs.setdefault('module_config_files', None)
        kwargs.setdefault('disable_interactive', False)
        kwargs.setdefault('save_table', False)
        kwargs.setdefault('batch', True)
        kwargs.setdefault('friend_directory', None)
        kwargs.setdefault('friend_tree_names', None)
        kwargs.setdefault('friend_file_pattern', None)
        kwargs.setdefault('precision', 3)
        self.event_numbers = dict()
        self.lumi = kwargs['lumi']
        self.interactive = not kwargs['disable_interactive']
        self.disable_sm_total = kwargs['disable_sm_total']
        if 'dataset_config' in kwargs:
            _logger.error('The property "dataset_config" is not supported anymore. Please use xs_config_file')
            kwargs.setdefault('xs_config_file', kwargs['dataset_config'])
        self.xs_handle = XSHandle(kwargs["xs_config_file"])
        self.file_handles = [FH(file_name=fn, dataset_info=kwargs['xs_config_file'],
                                friend_directory=kwargs['friend_directory'],
                                friend_tree_names=kwargs['friend_tree_names'],
                                friend_pattern=kwargs['friend_file_pattern']) for fn in kwargs['file_list']]
        self.process_configs = None
        self.save_table = kwargs['save_table']
        if kwargs['process_config_files'] is not None:
            self.process_configs = parse_and_build_process_config(kwargs['process_config_files'])

        #self.dtype = [('cut', 'S300'), ('yield', 'f4'), ('yield_unc', 'f4'), ('eff', float), ('eff_total', float)]
        self.dtype = [('cut', 'S300'), ('yield', 'f4')]
        self.output_handle = None
        self.plot_config = None
        self.config = None
        if kwargs['plot_config_file'] is not None:
            self.plot_config = parse_and_build_plot_config(kwargs['plot_config_file'])
        if kwargs['output_dir'] is not None:
            self.output_handle = OutputFileHandle(output_dir=kwargs['output_dir'])
        if kwargs['config_file'] is not None:
            self.config = YAMLLoader.read_yaml(kwargs['config_file'])
            if 'Lumi' in self.config:
                self.lumi = self.config['Lumi']
        if self.process_configs is not None:
            self.file_handles = pl.filter_unavailable_processes(self.file_handles, self.process_configs)
        self.modules = []
        if kwargs['module_config_files'] is not None:
            modules = load_modules(kwargs['module_config_files'], self)
            self.modules = [m for m in modules]

        list(map(self.load_dxaod_cutflows, self.file_handles))
        set_batch_mode(kwargs['batch'])

    def load_dxaod_cutflows(self, file_handle):
        #return
        process = file_handle.process
        if process is None:
            _logger.error("Parsed NoneType process from {:s}".format(file_handle.file_name))
            return
        if process not in self.event_numbers:
            self.event_numbers[process] = file_handle.get_number_of_total_events()
        else:
            self.event_numbers[process] += file_handle.get_number_of_total_events()

    def get_cross_section_weight(self, process):
        """
        Calculates weight according to process cross section and luminosity. If MC samples are split in several
        production campaigns and the luminosity information is provided as a dictionary with the campaign name as key
        and luminosity as value each campaign will be scaled to this luminosity and processes will be added up later
        :param process: process information
        :type process: Process
        :return: luminosity weight
        :rtype: float
        """
        cross_section_weight = get_xsec_weight(self.lumi, process, self.xs_handle, self.event_numbers)
        return cross_section_weight

    @staticmethod
    def format_yield(value, uncertainty=None):
        if value > 10000.:
            yld_string = '{:.3e}'.format(value)
            if uncertainty is not None:
                yld_string += ' +- {:.3e}'.format(uncertainty)
        else:
            yld_string = '{:.2f}'.format(value)
        return yld_string

    def stringify(self, cutflow):
        def format_yield(value, uncertainty=None):
            if value > 10000.:
                return "{:.2e}".format(value)
            else:
                return "{:.2f} ".format(value)

        name = 'yield'
        if self.raw:
            name = 'yield_raw'
        cutflow = np.array([(cutflow[i]["cut"],
                             format_yield(cutflow[i][name])) for i in range(len(cutflow))],
                           dtype=[("cut", "S100"), (name, "S100")])
        return cutflow

    def print_cutflow_table(self):
        available_cutflows = list(self.cutflow_tables.keys())
        if self.interactive:
            print("######## Selection menu  ########")
            print("Available cutflows for printing: ")
            print("--------------------------------")
            for i, region in enumerate(available_cutflows):
                print(i, ")", region)
            print("a) all")
            user_input = input(
                "Please enter your selection (space or comma separated). Hit enter to select default (BaseSelection) ")
            if user_input == "":
                selections = ["BaseSelection"]
            elif user_input.lower() == "a":
                selections = available_cutflows
            elif "," in user_input:
                selections = [available_cutflows[i] for i in map(int, user_input.split(","))]
            elif "," not in user_input:
                selections = [available_cutflows[i] for i in map(int, user_input.split())]
            else:
                print("{:s}Invalid input {:s}. Going for default.\033[0m".format("\033[91m", user_input))
                selections = ["BaseSelection"]
        else:
            selections = available_cutflows
        if self.save_table:
            if self.output_tag:
                f = open(os.path.join(self.output_dir, 'cutflow_' + self.output_tag + '.txt'), 'w')
            else:
                f = open(os.path.join(self.output_dir, 'cutflow.txt'), 'w')
        for selection, cutflow in list(self.cutflow_tables.items()):
            if selection not in selections:
                continue
            print()
            print("Cutflow for region %s" % selection)
            print(cutflow)
            if self.save_table:
                print("Cutflow for region %s" % selection, file=f)
                print(cutflow, file=f)

    def make_cutflow_tables(self):
        for systematic in self.systematics:
            self.make_cutflow_table(systematic)


class ExtendedCutFlowAnalyser(CommonCutFlowAnalyser):
    """
    Extended cutflow analyser building additional cuts not stored in cutflow hist
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("output_file_name", None)
        kwargs.setdefault("no_merge", False)
        kwargs.setdefault("raw", False)
        kwargs.setdefault("output_dir", None)
        kwargs.setdefault("format", "plain")
        kwargs.setdefault('enable_eff', False)
        kwargs.setdefault('percent_eff', False)
        kwargs.setdefault('disable_signal_plots', False)
        kwargs.setdefault('friend_tree_names', None)
        super(ExtendedCutFlowAnalyser, self).__init__(**kwargs)
        for k, v in list(kwargs.items()):
            if not hasattr(self, k):
                setattr(self, k, v)
        if not 'syst_tree_name' in kwargs:
            self.syst_tree_name = self.tree_name
        self.filter_empty_trees()
        self.event_yields = {}
        self.selection = RegionBuilder(**YAMLLoader.read_yaml(kwargs["selection_config"])["RegionBuilder"])
        self.converter = Root2NumpyConverter(["weight"])
        self.cutflow_tables = {}
        self.cutflows = {}
        print("setup extended cfa")
        if kwargs["output_dir"] is not None:
            self.output_handle = OutputFileHandle(output_dir=kwargs["output_dir"])
        for k, v in list(kwargs.items()):
            if not hasattr(self, k):
                setattr(self, k, v)
        if kwargs['friend_tree_names'] is not None:
            list([fh.reset_friends() for fh in self.file_handles])
            list([fh.link_friend_trees(self.tree_name, 'Nominal') for fh in self.file_handles])

        self.region_selections = {}
        if self.plot_config is None:
            self.plot_config = PlotConfig(name="acceptance_all_cuts", color=get_default_color_scheme(),
                                          #labels=[data[0] for data in acceptance_hists],
                                          xtitle="LQ mass [GeV]", ytitle="acceptance [%]", draw="Marker",
                                          lumi=self.lumi, watermark="Internal", ymin=0., ymax=100.)

    def filter_empty_trees(self):
        def is_empty(file_handle, tree_name, syst_tree_name):
            tn = tree_name
            if syst_tree_name is not None and file_handle.is_mc:
                tn = syst_tree_name
            return file_handle.get_object_by_name(tn, "Nominal").GetEntries() > 0

        empty_files = [fh for fh in self.file_handles if not is_empty(fh, self.tree_name, self.syst_tree_name)]
        self.file_handles = [fh for fh in self.file_handles if is_empty(fh, self.tree_name, self.syst_tree_name)]
        list([fh.close() for fh in empty_files])

    def read_event_yields(self, systematic="Nominal"):
        _logger.info("Read event yields in directory {:s}".format(systematic))
        if systematic not in self.cutflows:
            self.cutflows[systematic] = {}
        for region in self.selection.regions:
            self.region_selections[region] = region.get_cut_list()
            if region.name not in self.cutflows[systematic]:
                self.cutflows[systematic][region.name] = {}
            for file_handle in self.file_handles:
                process = file_handle.process
                process_config = find_process_config(process, process_configs=self.process_configs)
                tree = file_handle.get_object_by_name(self.tree_name, systematic)
                yields = []
                cut_list = region.get_cut_list(file_handle.process.is_data)
                cut_string = ""
                for i, cut in enumerate(cut_list):
                    if cut.process_type is not None:
                        if process_config.type.lower() in cut.process_type:
                           current_cut = cut.selection
                        else:
                            current_cut = "1"
                    else:
                        current_cut = cut.selection
                    cut_string = '&&'.join([cut_string, current_cut]).lstrip('&&')
                    cut_string = cut_string.replace(' ', '')
                    if not self.raw:
                        yields.append([cut.name,
                                       self.converter.convert_to_array(tree, cut_string)['weight'].flatten().sum()])
                                       #0, -1., -1.))
                    else:
                        yields.append([cut.name,
                                       len([y for y in self.converter.convert_to_array(tree, cut_string)['weight'].flatten() if y != 0.])])
                        # 0, -1., -1.))
                if process not in self.cutflows[systematic][region.name]:
                    self.cutflows[systematic][region.name][process] = yields
                else:
                    for icut, y in enumerate(yields):
                        self.cutflows[systematic][region.name][process][icut][1] += y[1]
            name = 'yield'
            if self.raw:
                name = 'yield_raw'
            for process in list(self.cutflows[systematic][region.name].keys()):
                list(map(tuple, self.cutflows[systematic][region.name][process]))
                self.cutflows[systematic][region.name][process] = list(map(tuple, self.cutflows[systematic][region.name][process]))
                self.cutflows[systematic][region.name][process] = np.array(self.cutflows[systematic][region.name][process],
                                                                           dtype=[('cut', 'S300'), (name, 'f4')])

    def apply_cross_section_weight(self, systematic, region):
        for process in list(self.cutflows[systematic][region].keys()):
            if process.is_data:
                continue
            try:
                lumi_weight = self.get_cross_section_weight(process)
            except InvalidInputError:
                _logger.error("None type parsed for ", process)
                continue
            self.cutflows[systematic][region][process]['yield'] *= lumi_weight

    def make_cutflow_table(self, systematic):
        cutflow_tables = dict()

        def keep_process(process, signals):
            if process == "SMTotal":
                return True

            prcf = find_process_config(process, self.process_configs)
            if prcf is None:
                _logger.error("Could not find process config for {:s}. This is not expected. Removing process. "
                              "Please investigate".format(process))
                return False
            if prcf.type.lower() != "signal":
                return True
            if prcf in signals:
                return True
            return False

        for region in list(self.cutflows[systematic].keys()):
            process_configs = [(process,
                                find_process_config(process,
                                                    self.process_configs)) for process in list(self.cutflows[systematic][region].keys())]
            if self.process_configs is None:
                process_configs = []
            if len([pc for pc in process_configs if pc[0] == "SMTotal" or pc[1].type.lower() == "signal"]) > 3:
                signals = [pc for pc in process_configs if pc[0] == "SMTotal" or pc[1].type.lower() == "signal"]
                try:
                    signals.sort(key=lambda i: int(re.findall('\d{2,4}', i[0])[0]))
                except IndexError:
                    _logger.error("Problem sorting signals")
                if self.config is not None:
                    if 'ordering' in self.config:
                        for sig in signals:
                            sig_name = sig[0]
                            if sig_name in self.config['ordering']:
                                continue
                            self.config['ordering'].append(sig_name)
                    else:
                        self.config['ordering'] = [s[0] for s in signals]
                else:
                    self.config = OrderedDict({'ordering': [s[0] for s in signals]})
                choices = None
                if self.interactive:
                    for i, process in enumerate(signals):
                        print("{:d}, {:s}".format(i, process[0]))
                    print("a) All")
                    choice = eval(input("What processes do you like to have shown (comma/space seperated)?"))
                    try:
                        if choice.lower() == "a":
                            choices = None
                        elif "," in choice:
                            choices = list(map(int, choice.split(",")))
                        else:
                            choices = list(map(int, choice.split(",")))
                    except ValueError:
                        pass
                if choices is not None:
                    signals = [process[1] for process in signals if signals.index(process) in choices]
                    self.cutflows[systematic][region] = OrderedDict([kv for kv in iter(list(self.cutflows[systematic][region].items())) if keep_process(kv[0], signals)])

            for process, cutflow in list(self.cutflows[systematic][region].items()):
                cutflow_tmp = self.stringify(cutflow)
                if region not in list(cutflow_tables.keys()):
                    cutflow_tables[region] = pd.DataFrame(cutflow_tmp, dtype=str)
                    if self.enable_eff:
                        cutflow_tables[region] = self.calculate_cut_efficiencies(cutflow_tables[region])
                        cutflow_tables[region].columns = ["cut", process, 'eff_{:s}'.format(process)]
                    else:
                        cutflow_tables[region].columns = ["cut", process]
                    continue
                d = {process: cutflow_tmp['yield']}
                cutflow_tables[region] = cutflow_tables[region].assign(**d)
                if self.enable_eff:
                    cutflow_tables[region] = self.calculate_cut_efficiencies(cutflow_tables[region], cutflow_tmp,
                                                                             process)

            self.cutflow_tables = {}
            ordering = None
            if self.config is not None and 'ordering' in self.config:
                ordering = self.config['ordering']
            for k, v in list(cutflow_tables.items()):
                if ordering is not None:
                    processes = list(v.keys())
                    if 'cut' not in ordering:
                        ordering.insert(0, 'cut')
                    ordering = [p for p in ordering if p in processes]
                    ordering += [p for p in processes if p not in ordering]
                    v = v[ordering]
                fct = 'to_csv'
                default_args = {'sep': ','}
                if self.format == 'plain':
                    fct = 'to_string'
                    default_args = {}
                if self.format == 'latex':
                    fct = 'to_latex'
                    default_args = {'index': False, 'escape': False}

                self.cutflow_tables[k] = getattr(v, fct)(**default_args)

    def calculate_sm_total(self):
        def add(yields):
            sm_yield = []
            for process, evn_yields in list(yields.items()):
                if 'data' in process.lower():
                    continue
                if find_process_config(process, self.process_configs).type.lower() == "signal":
                    continue
                if len(sm_yield) == 0:
                    sm_yield = list(evn_yields)
                    continue
                for icut, cut_item in enumerate(evn_yields):
                    sm_yield[icut] = tuple([sm_yield[icut][0]] + list(map(operator.add,
                                                                     list(sm_yield[icut])[1:],
                                                                     list(evn_yields[icut])[1:])))
            return np.array(sm_yield, dtype=self.dtype)

        for systematics, regions_data in list(self.cutflows.items()):
            for region, yields in list(regions_data.items()):
                self.cutflows[systematics][region]['SMTotal'] = add(yields)

    def merge(self, yields):
        """
        Merge event yields for subprocesses
        :param yields: pair of process and yields
        :type yields: dict
        :return: merged yields
        :rtype: dict
        """
        if self.process_configs is None:
            return yields
        for process in list(yields.keys()):
            parent_process = find_process_config(process, self.process_configs).name
            if parent_process is None:
                continue
            if not parent_process in list(yields.keys()):
                yields[parent_process] = yields[process]
            else:
                try:
                    yields[parent_process]["yield"] += yields[process]["yield"]
                except TypeError:
                    yields[parent_process] += yields[process]
            yields.pop(process)
        return yields

    def merge_yields(self):
        for systematics, regions_data in list(self.cutflows.items()):
            for region, yields in list(regions_data.items()):
                self.cutflows[systematics][region] = self.merge(yields)

    def update_top_background(self, module):
        def calc_inclusive():
            stitch = module.get_stitch_point(region)

        for region in list(self.cutflows['Nominal'].keys()):
            inclusive = None
            stitch = module.get_stitch_point(region)
            for cut, yld in self.cutflows['Nominal'][region]['ttbar']:
                yld = 10000.
                if not 'mLQmax' in cut:
                    continue
                mass = float(re.findall('\d+', cut)[0])
                if mass < stitch:
                    continue
                yld = module.get_extrapolated_bin_content(region, mass, lumi=140.)

    def plot_signal_yields(self):
        """
        Make plots of signal yields after each cut summarised per signal sample
        :return: nothing
        :rtype: None
        """
        if self.output_handle is None:
            _logger.error("Request to plot signal yields, but output handle not initialised. Please provide output"
                          "directory.")
            return
        replace_items = [('/', ''), (' ', ''), ('>', '_gt_'), ('<', '_lt_'), ('$', ''), ('.', '')]
        signal_processes = [prc for prc in list(self.process_configs.values()) if prc.type.lower() == "signal"]

        signal_generated_events = self.merge(self.event_numbers)
        signal_generated_events = dict([cf for cf in iter(list(signal_generated_events.items())) if cf[0] in [prc.name for prc in signal_processes]])

        for region, cutflows in list(self.cutflows["Nominal"].items()):
            signal_yields = dict([cf for cf in iter(list(cutflows.items())) if cf[0] in [prc.name for prc in signal_processes]])
            if len(signal_yields) == 0:
                continue
            canvas_cuts, canvas_cuts_log, canvas_final = get_signal_acceptance(signal_yields,
                                                                               signal_generated_events,
                                                                               self.plot_config, None)

            new_name_cuts = canvas_cuts.GetName()
            new_name_cuts_log = canvas_cuts_log.GetName()
            new_name_final = canvas_final.GetName()
            for item in replace_items:
                new_name_cuts = new_name_cuts.replace(item[0], item[1])
                new_name_cuts_log = new_name_cuts_log.replace(item[0], item[1])
                new_name_final = new_name_final.replace(item[0], item[1])
            canvas_cuts.SetName('{:s}_{:s}'.format(new_name_cuts, region))
            canvas_cuts_log.SetName('{:s}_{:s}'.format(new_name_cuts_log, region))
            canvas_final.SetName('{:s}_{:s}'.format(new_name_final, region))
            self.output_handle.register_object(canvas_cuts)
            self.output_handle.register_object(canvas_cuts_log)
            self.output_handle.register_object(canvas_final)

    def calculate_cut_efficiencies(self, cutflow, np_cutflow = None, process = None):
        """
        Calculate cut efficiencies w.r.t to first yield
        :param cutflow: cutflow yields
        :type cutflow: pandas.DataFrame
        :param cutflow: numpy array cutflow to be added to overall cutflow (cutflow)
        :type cutflow: numpy.ndarray
        :return: cutflow yields with efficiencies
        :rtype: pandas.DataFrame
        """

        def get_reference():
            return current_process_cf['yield'][0]

        current_process_cf = cutflow
        if np_cutflow is not None:
            current_process_cf = np_cutflow
        try:
            cut_efficiencies = [float(i)/float(get_reference()) for i in current_process_cf['yield']]
        except ZeroDivisionError:
            cut_efficiencies = [1.] * len(current_process_cf['yield'])
        if self.percent_eff:
            cut_efficiencies = [val*100. for val in cut_efficiencies]
        cut_efficiencies = ['{:.2f}'.format(val) for val in cut_efficiencies]
        tag = 'eff'
        if process is not None:
            tag = 'eff_{:s}'.format(process)
        return cutflow.assign(**{tag: cut_efficiencies})

    def execute(self):
        for systematic in self.systematics:
            self.read_event_yields(systematic)
        #self.plot_signal_yields()

        if not self.raw:
            for systematic in list(self.cutflows.keys()):
                for region in list(self.cutflows[systematic].keys()):
                    self.apply_cross_section_weight(systematic, region)
        if self.no_merge is False:
            self.merge_yields()
        #Need to remap names
        for systematics in list(self.cutflows.keys()):
            for region in list(self.cutflows[systematics].keys()):
                for process in list(self.cutflows[systematics][region].keys()):
                    if not isinstance(process, Process):
                        continue
                    self.cutflows[systematics][region][process.process_name] = self.cutflows[systematics][region].pop(
                        process)
        if len(self.modules) > 0:
            self.update_top_background(self.modules[0])
        if not self.disable_signal_plots:
            self.plot_signal_yields()

        if not self.disable_sm_total:
            self.calculate_sm_total()

        self.make_cutflow_tables()
        if self.output_handle is not None:
            self.output_handle.write_and_close()


class CutflowAnalyser(CommonCutFlowAnalyser):
    """
    Cutflow analyser
    """

    def __init__(self, **kwargs):
        kwargs.setdefault('output_file_name', None)
        kwargs.setdefault('lumi', None)
        kwargs.setdefault('process_configs', None)
        kwargs.setdefault('no_merge', False)
        kwargs.setdefault('raw', False)
        kwargs.setdefault('output_dir', None)
        kwargs.setdefault('format', 'plain')
        super(CutflowAnalyser, self).__init__(**kwargs)
        self.precision = 2 #TODO: quick term fix
        self.cutflow_hists = dict()
        self.cutflow_hists = dict()
        self.cutflow_tables = dict()
        self.output_file_name = kwargs['output_file_name']
        self.systematics = kwargs['systematics']
        self.cutflow_hists = dict()
        self.cutflows = dict()
        self.event_numbers = dict()
        self.raw = kwargs['raw']
        self.format = kwargs['format']
        self.merge = True if not kwargs['no_merge'] else False

    def apply_cross_section_weight(self):
        for process in list(self.cutflow_hists.keys()):
            if process.is_data:
                continue
            try:
                lumi_weight = self.get_cross_section_weight(process)
            except InvalidInputError:
                _logger.error("None type parsed for ", self.cutflow_hists[process])
                continue
            for systematic in list(self.cutflow_hists[process].keys()):
                for cutflow in list(self.cutflow_hists[process][systematic].values()):
                    scale(cutflow, lumi_weight)

    def analyse_cutflow(self):
        self.apply_cross_section_weight()
        if self.process_configs is not None and self.merge:
            self.merge_histograms(self.cutflow_hists)
        if not self.disable_sm_total:
            self.calculate_sm_total()
        self.cutflow_hists = dict([kv for kv in iter(list(self.cutflow_hists.items())) if len(kv[1]) > 0])
        for systematic in self.systematics:
            self.cutflows[systematic] = dict()
            for process in list(self.cutflow_hists.keys()):
                self.cutflows[systematic][process] = dict()
                for k, v in list(self.cutflow_hists[process][systematic].items()):
                    if k.endswith('_raw'):
                        continue
                    raw_cutflow = self.cutflow_hists[process][systematic][k + '_raw']
                    self.cutflows[systematic][process][k] = self._analyse_cutflow(v, raw_cutflow)
        self.calculate_cut_efficiencies()

    def merge_histograms(self, histograms):
        for process in list(histograms.keys()):
            parent_process = find_process_config(process, self.process_configs).name
            if parent_process is None:
                continue
            for systematic in list(histograms[process].keys()):
                for selection in list(histograms[process][systematic].keys()):
                    if not parent_process in list(histograms.keys()):
                        histograms[parent_process] = dict((syst,
                                                    dict((sel, None) for sel in
                                                         list(histograms[process][syst].keys())))
                                                   for syst in list(histograms[process].keys()))
                    if selection not in histograms[process][systematic]:
                        _logger.warning("Could not find selection {:s} for process {:s}".format(selection,
                                                                                                process.process_name))
                        continue
                    new_hist_name = histograms[process][systematic][selection].GetName().replace(process.process_name, parent_process)
                    if histograms[parent_process][systematic][selection] is None:
                        new_hist_name = histograms[process][systematic][selection].GetName().replace(
                            process.process_name, parent_process)
                        histograms[parent_process][systematic][selection] = histograms[process][systematic][
                            selection].Clone(new_hist_name)
                    else:
                        histograms[parent_process][systematic][selection].Add(
                            histograms[process][systematic][selection].Clone(new_hist_name))
            histograms.pop(process)

    def calculate_sm_total(self):
        sm_total_cutflows = {}
        for process, systematics in list(self.cutflow_hists.items()):
            if 'data' in process.lower():
                continue
            for systematic, regions in list(systematics.items()):
                if systematic not in list(sm_total_cutflows.keys()):
                    sm_total_cutflows[systematic] = {}
                for region, cutflow_hist in list(regions.items()):
                    if region not in list(sm_total_cutflows[systematic].keys()):
                        sm_total_cutflows[systematic][region] = cutflow_hist.Clone()
                        continue
                    sm_total_cutflows[systematic][region].Add(cutflow_hist)
        self.cutflow_hists['SMTotal'] = sm_total_cutflows

    def _analyse_cutflow(self, cutflow_hist, raw_cutflow_hist):
        if not self.raw:
            parsed_info = np.array([(cutflow_hist.GetXaxis().GetBinLabel(b),
                                     cutflow_hist.GetBinContent(b),
                                     # raw_cutflow_hist.GetBinContent(b),
                                     cutflow_hist.GetBinError(b),
                                     # #raw_cutflow_hist.GetBinError(b),
                                     -1.,
                                     -1.) for b in range(1, cutflow_hist.GetNbinsX() + 1)],
                                   dtype=[('cut', 'S100'), ('yield', 'f4'),  # ('yield_raw', 'f4'),
                                          ('yield_unc', 'f4'),
                                          ('eff', float),
                                          ('eff_total', float)])  # todo: array dtype for string not a good choice
        else:
            parsed_info = np.array([(cutflow_hist.GetXaxis().GetBinLabel(b),
                                     raw_cutflow_hist.GetBinContent(b),
                                     raw_cutflow_hist.GetBinError(b),
                                     -1.,
                                     -1.) for b in range(1, cutflow_hist.GetNbinsX() + 1)],
                                   dtype=[('cut', 'S100'), ('yield_raw', 'f4'),
                                          ('yield_unc_raw', 'f4'),
                                          ('eff', float),
                                          ('eff_total', float)])  # todo: array dtype for string not a good choice
        return parsed_info

    def calculate_cut_efficiencies(self):
        for systematic in self.systematics:
            for process in list(self.cutflows[systematic].keys()):
                for cutflow in list(self.cutflows[systematic][process].values()):
                    self.calculate_cut_efficiency(cutflow)

    def calculate_cut_efficiency(self, cutflow):
        yield_str = 'yield'
        if self.raw:
            yield_str = 'yield_raw'
        for i in range(len(cutflow['cut'])):
            if i == 0:
                cutflow[i]['eff'] = 100.
                cutflow[i]['eff_total'] = 100.
                continue
            if cutflow[i - 1][yield_str] != 0.:
                cutflow[i]['eff'] = round(100.0 * cutflow[i][yield_str] / cutflow[i - 1][yield_str], 3)
            else:
                cutflow[i]['eff'] = -1.
            if cutflow[0][yield_str] != 0.:
                cutflow[i]['eff_total'] = round(100.0 * cutflow[i][yield_str] / cutflow[0][yield_str], 3)
            else:
                cutflow[i]['eff_total'] = -1.

    def make_cutflow_tables(self):
        for systematic in self.systematics:
            self.make_cutflow_table(systematic)

    def make_cutflow_table(self, systematic):
        cutflow_tables = OrderedDict()
        # signal_yields = dict(filter(lambda cf: cf[0] in map(lambda prc: prc.name, signal_processes),
        #                             cutflows.iteritems()))
        for process in list(self.cutflows[systematic].keys()):
            for selection, cutflow in list(self.cutflows[systematic][process].items()):
                cutflow_tmp = self.stringify(cutflow)
                if selection not in list(cutflow_tables.keys()):
                    cutflow_tables[selection] = cutflow_tmp
                    continue
                cutflow_tables[selection] = rec_append_fields(cutflow_tables[selection],
                                                              [i + process for i in cutflow_tmp.dtype.names[1:]],
                                                              [cutflow_tmp[n] for n in cutflow_tmp.dtype.names[1:]])
            headers = ['Cut'] + [x for elem in list(self.cutflows[systematic].keys()) for x in (elem, '')]
            self.cutflow_tables = {k: tabulate(v.transpose(),
                                               headers=headers,
                                               tablefmt=self.format) #floatfmt='.2f'
                                   for k, v in list(cutflow_tables.items())}

    def stringify(self, cutflow):
        def format_yield(value, uncertainty):
            if value > 10000.:
                return '{:.{:d}e}'.format(value, self.precision)
            else:
                return '{:.{:d}f}'.format(value, self.precision)
            # if value > 10000.:
            #     return "{:.3e} +- {:.3e}".format(value, uncertainty)
            # else:
            #     return "{:.2f} +- {:.2f}".format(value, uncertainty)

        if not self.raw:
            cutflow = np.array([(cutflow[i]['cut'],
                                 format_yield(cutflow[i]['yield'], cutflow[i]['yield_unc']),
                                 # cutflow[i]['eff'],
                                 cutflow[i]['eff_total']) for i in range(len(cutflow))],
                               dtype=[('cut', 'S100'), ('yield', 'S100'), ('eff_total', float)])  # ('eff', float),
        else:
            cutflow = np.array([(cutflow[i]['cut'],
                                 format_yield(cutflow[i]['yield_raw'], cutflow[i]['yield_unc_raw']),
                                 # cutflow[i]['eff'],
                                 cutflow[i]['eff_total']) for i in range(len(cutflow))],
                               dtype=[('cut', 'S100'), ('yield_raw', 'S100'), ('eff_total', float)])  # ('eff', float),
        return cutflow

    def print_cutflow_table(self):
        available_cutflows = list(self.cutflow_tables.keys())
        print("######## Selection menu  ########")
        print("Available cutflows for printing: ")
        print("--------------------------------")
        for i, region in enumerate(available_cutflows):
            print(i, ")", region)
        print("a) all")
        user_input = input(
            "Please enter your selection (space or comma separated). Hit enter to select default (BaseSelection) ")
        print('user INPUT: ', user_input)
        if user_input == "":
            selections = ["BaseSelection"]
        elif user_input.lower() == "a":
            selections = available_cutflows
        elif "," in user_input:
            selections = [available_cutflows[i] for i in map(int, user_input.split(","))]
        elif "," not in user_input:
            selections = [available_cutflows[i] for i in map(int, user_input.split())]
        else:
            print("{:s}Invalid input {:s}. Going for default.\033[0m".format("\033[91m", user_input))
            selections = ["BaseSelection"]
        for selection, cutflow in list(self.cutflow_tables.items()):
            if selection not in selections:
                continue
            print()
            print("Cutflow for region %s" % selection)
            print(cutflow)

    def store_cutflow_table(self):
        pass

    def load_cutflows(self, file_handle):
        process = file_handle.process
        if process is None:
            _logger.error("Parsed NoneType process from {:s}".format(file_handle.file_name))
            return
        if process not in self.event_numbers:
            self.event_numbers[process] = file_handle.get_number_of_total_events()
        else:
            self.event_numbers[process] += file_handle.get_number_of_total_events()
        if process not in list(self.cutflow_hists.keys()):
            self.cutflow_hists[process] = dict()
        for systematic in self.systematics:
            cutflow_hists = file_handle.get_objects_by_pattern("^(cutflow_)", systematic)
            if systematic not in self.cutflow_hists[process]:
                self.cutflow_hists[process][systematic] = dict()
            for cutflow_hist in cutflow_hists:
                cutflow_hist.SetDirectory(0)
                try:
                    self.cutflow_hists[process][systematic][cutflow_hist.GetName().replace("cutflow_", "")].Add(
                        cutflow_hist)
                except KeyError:
                    self.cutflow_hists[process][systematic][
                        cutflow_hist.GetName().replace("cutflow_", "")] = cutflow_hist

    def read_cutflows(self):
        for file_handle in self.file_handles:
            self.load_cutflows(file_handle)

    def plot_cutflow(self):
        if self.output_handle is None:
            return
        set_batch_mode(True)
        flipped = defaultdict(lambda: defaultdict(dict))
        for process, systematics in list(self.cutflow_hists.items()):
            for systematic, cutflows in list(systematics.items()):
                if "smtotal" in process.lower():
                    continue
                for region, cutflow_hist in list(cutflows.items()):
                    flipped[systematic][region][process] = cutflow_hist
        plot_config = PlotConfig(name=None, dist=None, ytitle="Events", logy=True)
        for region in list(flipped['Nominal'].keys()):
            plot_config.name = "{:s}_cutflow".format(region)
            cutflow_hists = {process: hist for process, hist in list(flipped["Nominal"][region].items())
                             if "smtotal" not in process.lower()}
            for process, cutflow_hist in list(cutflow_hists.items()):
                cutflow_hist.SetName("{:s}_{:s}".format(cutflow_hist.GetName(), process))
            cutflow_canvas = Pt.plot_stack(cutflow_hists, plot_config, process_configs=self.process_configs)
            Ft.add_legend_to_canvas(cutflow_canvas, process_configs=self.process_configs)
            self.output_handle.register_object(cutflow_canvas)
        self.output_handle.write_and_close()

    def execute(self):
        self.read_cutflows()
        self.analyse_cutflow()
        self.make_cutflow_tables()
        if hasattr(self, "output_handle"):
            self.plot_cutflow()
