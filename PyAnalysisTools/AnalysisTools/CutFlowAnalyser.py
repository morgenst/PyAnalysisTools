import operator
import re
from copy import deepcopy
from operator import add
import numpy as np
import pandas as pd
import PyAnalysisTools.PlottingUtils.PlottingTools as Pt
import PyAnalysisTools.PlottingUtils.Formatting as Ft

try:
    from tabulate.tabulate import tabulate
except ImportError:
    from tabulate import tabulate
tabulate.LATEX_ESCAPE_RULES={}

from collections import defaultdict, OrderedDict
from PyAnalysisTools.base import _logger, InvalidInputError
from PyAnalysisTools.ROOTUtils.FileHandle import FileHandle as FH
from PyAnalysisTools.PlottingUtils import set_batch_mode
from PyAnalysisTools.PlottingUtils.HistTools import scale
from PyAnalysisTools.AnalysisTools.XSHandle import XSHandle
from PyAnalysisTools.PlottingUtils.PlotConfig import parse_and_build_process_config, find_process_config, PlotConfig, \
    parse_and_build_plot_config, get_default_color_scheme
from PyAnalysisTools.base.OutputHandle import OutputFileHandle
from PyAnalysisTools.PlottingUtils.Plotter import Plotter as pl
from numpy.lib.recfunctions import rec_append_fields
from PyAnalysisTools.AnalysisTools.RegionBuilder import NewRegionBuilder
from PyAnalysisTools.base.YAMLHandle import YAMLLoader
from PyAnalysisTools.AnalysisTools.MLHelper import Root2NumpyConverter
from PyAnalysisTools.AnalysisTools.StatisticsTools import get_signal_acceptance
from PyAnalysisTools.PlottingUtils import set_batch_mode


class CommonCutFlowAnalyser(object):
    def __init__(self, **kwargs):
        kwargs.setdefault("lumi", None)
        kwargs.setdefault("process_config", None)
        kwargs.setdefault("disable_sm_total", False)
        kwargs.setdefault('plot_config_file', None)
        kwargs.setdefault('config_file', None)
        kwargs.setdefault('batch', True)
        kwargs.setdefault('friend_directory', None)
        kwargs.setdefault('friend_tree_names', None)
        kwargs.setdefault('friend_file_pattern', None)
        self.event_numbers = dict()
        self.lumi = kwargs['lumi']
        self.disable_sm_total = kwargs['disable_sm_total']
        self.xs_handle = XSHandle(kwargs['xs_config_file'])
        self.file_handles = [FH(file_name=fn, dataset_info=kwargs['xs_config_file'],
                                friend_directory=kwargs['friend_directory'],
                                friend_tree_names=kwargs['friend_tree_names'],
                                friend_pattern=kwargs['friend_file_pattern']) for fn in kwargs['file_list']]
        self.process_configs = None
        if "process_configs" in kwargs and not "process_config_files" in kwargs:
            raw_input("Single process config deprecated. Please update to process_configs option and appreiate by "
                      "hitting enter.")
            kwargs['process_config_files'] = kwargs['process_configs']
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
            self.file_handles = pl.filter_processes_new(self.file_handles, self.process_configs)
        map(self.load_dxaod_cutflows, self.file_handles)
        set_batch_mode(kwargs['batch'])

    def load_dxaod_cutflows(self, file_handle):
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
        Get lumi weighting factor based on cross section, filter eff, k-factor and lumi.
        :param process: process name. In general contains MC campaign information which is removed to retrieve xsec
        :type process: string
        :return: weighting factor
        :rtype: float
        """

        if process is None:
            _logger.error("Process is None")
            raise InvalidInputError("Process is NoneType")
        if self.lumi is None or "data" in process.lower() or self.lumi == -1:
            return 1.
        lumi_weight = self.xs_handle.get_lumi_scale_factor(process.split('.')[0], self.lumi,
                                                           self.event_numbers[process])
        _logger.debug("Retrieved %.2f as cross section weight for process %s and lumi %.2f" % (lumi_weight, process,
                                                                                               self.lumi))
        return lumi_weight

    def get_cross_section_weight_new(self, process):
        """
        Weight histograms according to process cross section and luminosity. If MC samples are split in several
        production campaigns and the luminosity information is provided as a dictionary with the campaign name as key
        and luminosity as value each campaign will be scaled to this luminosity and processes will be added up later
        :param histograms: all plottable objects
        :type histograms: dict
        :return: nothing
        :rtype: None
        """
        provided_wrong_info = False
        lumi = self.lumi
        if isinstance(self.lumi, OrderedDict):
            if re.search('mc16[acde]$', process) is None:
                _logger.error('Could not find MC campaign information, but lumi was provided per MC '
                              'campaing. Not clear what to do. It will be assumed that you meant to scale '
                              'to total lumi. Please update and acknowledge once.')
                raw_input('Hit enter to continue or Ctrl+c to quit...')
                lumi = sum(self.lumi.values())
            else:
                lumi = self.lumi[process.split('.')[-1]]
        cross_section_weight = self.xs_handle.get_lumi_scale_factor(process.split(".")[0], lumi,
                                                                    self.event_numbers[process])
        return cross_section_weight

    def stringify(self, cutflow):
        def format_yield(value, uncertainty=None):
            if value > 10000.:
                return "{:.3e}".format(value)
            else:
                return "{:.2f} ".format(value)

        if not self.raw:
            # cutflow = np.array([(cutflow[i]["cut"],
            #                      format_yield(cutflow[i]["yield"], cutflow[i]["yield_unc"]),
            #                      # cutflow[i]["eff"],
            #                      cutflow[i]["eff_total"]) for i in range(len(cutflow))],
            #                    dtype=[("cut", "S100"), ("yield", "S100"), ("eff_total", float)])  # ("eff", float),
            cutflow = np.array([(cutflow[i]["cut"],
                                 format_yield(cutflow[i]["yield"])) for i in range(len(cutflow))],
                               dtype=[("cut", "S100"), ("yield", "S100")])
        else:
            cutflow = np.array([(cutflow[i]["cut"],
                                 format_yield(cutflow[i]["yield"])) for i in range(len(cutflow))],
                               dtype=[("cut", "S100"), ("yield", "S100")])
            # cutflow = np.array([(cutflow[i]["cut"],
            #                      format_yield(cutflow[i]["yield_raw"], cutflow[i]["yield_unc_raw"]),
            #                      # cutflow[i]["eff"],
            #                      #cutflow[i]["eff_total"]) for i in range(len(cutflow))],
            #                    dtype=[("cut", "S100"), ("yield_raw", "S100"), ("eff_total", float)])  # ("eff", float),
        return cutflow

    def print_cutflow_table(self):
        available_cutflows = self.cutflow_tables.keys()
        print "######## Selection menu  ########"
        print "Available cutflows for printing: "
        print "--------------------------------"
        for i, region in enumerate(available_cutflows):
            print i, ")", region
        print "a) all"
        user_input = raw_input(
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
            print "{:s}Invalid input {:s}. Going for default.\033[0m".format("\033[91m", user_input)
            selections = ["BaseSelection"]
        for selection, cutflow in self.cutflow_tables.iteritems():
            if selection not in selections:
                continue
            print
            print "Cutflow for region %s" % selection
            print cutflow

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
        self.event_yields = {}
        self.selection = NewRegionBuilder(**YAMLLoader.read_yaml(kwargs["selection_config"])["RegionBuilder"])
        self.converter = Root2NumpyConverter(["weight"])
        self.cutflow_tables = {}
        self.cutflows = {}

        if kwargs["output_dir"] is not None:
            self.output_handle = OutputFileHandle(output_dir=kwargs["output_dir"])
        for k, v in kwargs.iteritems():
            if not hasattr(self, k):
                setattr(self, k, v)
        if kwargs['friend_tree_names'] is not None:
            map(lambda fh: fh.reset_friends(), self.file_handles)
            map(lambda fh: fh.link_friend_trees(self.tree_name, 'Nominal'), self.file_handles)

        self.region_selections = {}
        if self.plot_config is None:
            self.plot_config = PlotConfig(name="acceptance_all_cuts", color=get_default_color_scheme(),
                                          #labels=[data[0] for data in acceptance_hists],
                                          xtitle="LQ mass [GeV]", ytitle="acceptance [%]", draw="Marker",
                                          lumi=self.lumi, watermark="Internal", ymin=0., ymax=100.)

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
                cut_list = region.get_cut_list(file_handle.is_data)
                cut_string = ""
                for i, cut in enumerate(cut_list):
                    if "TYPE_" in cut.selection:
                        if process_config.type.upper() in cut.selection:
                           current_cut = cut.selection.replace('TYPE_{:s}'.format(process_config.type.upper),'')
                        else:
                            current_cut = "1"
                    else:
                        current_cut = cut.selection
                    cut_string = '&&'.join([cut_string, current_cut]).lstrip('&&')
                    if not self.raw:
                        yields.append([cut.name,
                                       self.converter.convert_to_array(tree, cut_string)['weight'].flatten().sum()])
                                       #0, -1., -1.))
                    else:
                        yields.append([cut.name,
                                       len(filter(lambda y: y != 0., self.converter.convert_to_array(tree, cut_string)['weight'].flatten()))])
                        # 0, -1., -1.))
                if process not in self.cutflows[systematic][region.name]:
                    self.cutflows[systematic][region.name][process] = yields
                else:
                    for icut, y in enumerate(yields):
                        self.cutflows[systematic][region.name][process][icut][1] += y[1]
            for process in self.cutflows[systematic][region.name].keys():
                map(tuple, self.cutflows[systematic][region.name][process])
                self.cutflows[systematic][region.name][process] = map(tuple, self.cutflows[systematic][region.name][process])
                self.cutflows[systematic][region.name][process] = np.array(self.cutflows[systematic][region.name][process],
                                                                           dtype=[('cut', 'S300'), ('yield', 'f4')])

    def apply_cross_section_weight(self, systematic, region):
        for process in self.cutflows[systematic][region].keys():
            if 'data' in process:
                continue
            try:
                lumi_weight = self.get_cross_section_weight_new(process)
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

        for region in self.cutflows[systematic].keys():
            process_configs = [(process,
                                find_process_config(process,
                                                    self.process_configs)) for process in self.cutflows[systematic][region].keys()]
            if len(filter(lambda pc: pc[0] == "SMTotal" or pc[1].type.lower() == "signal", process_configs)) > 3:
                signals = filter(lambda pc: pc[0] == "SMTotal" or pc[1].type.lower() == "signal", process_configs)
                signals.sort(key=lambda i: int(re.findall('\d{2,4}', i[0])[0]))
                if self.config is not None:
                    if 'ordering' in self.config:
                        for sig in signals:
                            sig_name = sig[0]
                            if sig_name in self.config['ordering']:
                                continue
                            self.config['ordering'].append(sig_name)
                    else:
                        self.config['ordering'] = map(lambda s: s[0], signals)
                else:
                    self.config = OrderedDict({'ordering': map(lambda s: s[0], signals)})
                for i, process in enumerate(sorted(signals, key=lambda p: p[0])):
                    print "{:d}, {:s}".format(i, process[0])
                print "a) All"
                choice = raw_input("What processes do you like to have shown (comma/space seperated)?")
                try:
                    if choice.lower() == "a":
                        choices = None
                    elif "," in choice:
                        choices = map(int, choice.split(","))
                    else:
                        choices = map(int, choice.split(","))
                except ValueError:
                    choices = None
                if choices is not None:
                    choices.sort(key=lambda i: int(re.findall('\d{2,4}', i[0])[0]))
                    signals = [process[1] for process in signals if signals.index(process) in choices]
                    self.cutflows[systematic][region] = OrderedDict(filter(lambda kv: keep_process(kv[0], signals),
                                                                    self.cutflows[systematic][region].iteritems()))

            for process, cutflow in self.cutflows[systematic][region].items():
                cutflow_tmp = self.stringify(cutflow)
                if region not in cutflow_tables.keys():
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
            for k, v in cutflow_tables.iteritems():
                if ordering is not None:
                    processes = v.keys()
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
            for process, evn_yields in yields.iteritems():
                if 'data' in process.lower():
                    continue
                if find_process_config(process, self.process_configs).type.lower() == "signal":
                    continue
                if len(sm_yield) == 0:
                    sm_yield = list(evn_yields)
                    continue
                for icut, cut_item in enumerate(evn_yields):
                    sm_yield[icut] = tuple([sm_yield[icut][0]] + map(operator.add,
                                                                     list(sm_yield[icut])[1:],
                                                                     list(evn_yields[icut])[1:]))
            return np.array(sm_yield, dtype=self.dtype)

        for systematics, regions_data in self.cutflows.iteritems():
            for region, yields in regions_data.iteritems():
                self.cutflows[systematics][region]['SMTotal'] = add(yields)

    def merge(self, yields):
        """
        Merge event yields for subprocesses
        :param yields: pair of process and yields
        :type yields: dict
        :return: merged yields
        :rtype: dict
        """
        for process in yields.keys():
            parent_process = find_process_config(process, self.process_configs).name
            if parent_process is None:
                continue
            if not parent_process in yields.keys():
                yields[parent_process] = yields[process]
            else:
                try:
                    yields[parent_process]["yield"] += yields[process]["yield"]
                except TypeError:
                    yields[parent_process] += yields[process]
            yields.pop(process)
        return yields

    def merge_yields(self):
        for systematics, regions_data in self.cutflows.iteritems():
            for region, yields in regions_data.iteritems():
                self.cutflows[systematics][region] = self.merge(yields)

    def plot_signal_yields(self):
        """
        Make plots of signal yields after each cut summarised per signal sample
        :return: nothing
        :rtype: None
        """
        replace_items = [('/', ''), (' ', ''), ('>', '_gt_'), ('<', '_lt_'), ('$', ''), ('.', '')]
        signal_processes = filter(lambda prc: prc.type.lower() == "signal", self.process_configs.values())

        signal_generated_events = self.merge(self.event_numbers)
        signal_generated_events = dict(filter(lambda cf: cf[0] in map(lambda prc: prc.name, signal_processes),
                                              signal_generated_events.iteritems()))

        for region, cutflows in self.cutflows["Nominal"].iteritems():
            signal_yields = dict(filter(lambda cf: cf[0] in map(lambda prc: prc.name, signal_processes),
                                        cutflows.iteritems()))
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
            cut_efficiencies = map(lambda val: val*100., cut_efficiencies)
        cut_efficiencies = map(lambda val: '{:.2f}'.format(val), cut_efficiencies)
        tag = 'eff'
        if process is not None:
            tag = 'eff_{:s}'.format(process)
        return cutflow.assign(**{tag: cut_efficiencies})

    def execute(self):
        self.read_event_yields()
        if not self.raw:
            for systematic in self.cutflows.keys():
                for region in self.cutflows[systematic].keys():
                    self.apply_cross_section_weight(systematic, region)
        self.merge_yields()
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
        kwargs.setdefault('process_config', None)
        kwargs.setdefault('no_merge', False)
        kwargs.setdefault('raw', False)
        kwargs.setdefault('output_dir', None)
        kwargs.setdefault('format', 'plain')
        super(CutflowAnalyser, self).__init__(**kwargs)
        self.cutflow_hists = dict()
        self.cutflow_hists = dict()
        self.cutflow_tables = dict()
        self.lumi = kwargs['lumi']
        self.output_file_name = kwargs['output_file_name']
        self.systematics = kwargs['systematics']
        self.cutflow_hists = dict()
        self.cutflows = dict()
        self.xs_handle = XSHandle(kwargs['dataset_config'])
        self.event_numbers = dict()
        self.process_configs = None
        self.raw = kwargs['raw']
        self.format = kwargs['format']
        self.merge = True if not kwargs['no_merge'] else False
        if kwargs['output_dir'] is not None:
            self.output_handle = OutputFileHandle(output_dir=kwargs['output_dir'])
        if kwargs['process_configs'] is not None:
            self.process_configs = parse_and_build_process_config(kwargs['process_configs'])

    def apply_cross_section_weight(self):
        for process in self.cutflow_hists.keys():
            try:
                lumi_weight = self.get_cross_section_weight(process)
            except InvalidInputError:
                _logger.error("None type parsed for ", self.cutflow_hists[process])
                continue
            for systematic in self.cutflow_hists[process].keys():
                for cutflow in self.cutflow_hists[process][systematic].values():
                    scale(cutflow, lumi_weight)

    def analyse_cutflow(self):
        self.apply_cross_section_weight()
        if self.process_configs is not None and self.merge:
            self.merge_histograms(self.cutflow_hists)
        if not self.disable_sm_total:
            self.calculate_sm_total()
        self.cutflow_hists = dict(filter(lambda kv: len(kv[1]) > 0, self.cutflow_hists.iteritems()))
        for systematic in self.systematics:
            self.cutflows[systematic] = dict()
            for process in self.cutflow_hists.keys():
                self.cutflows[systematic][process] = dict()
                for k, v in self.cutflow_hists[process][systematic].iteritems():
                    if k.endswith('_raw'):
                        continue
                    raw_cutflow = self.cutflow_hists[process][systematic][k + '_raw']
                    self.cutflows[systematic][process][k] = self._analyse_cutflow(v, raw_cutflow, process)
        self.calculate_cut_efficiencies()

    def merge_histograms(self, histograms):
        for process in histograms.keys():
            parent_process = find_process_config(process, self.process_configs).name
            print 'PARENT ', parent_process
            if parent_process is None:
                continue
            for systematic in histograms[process].keys():
                for selection in histograms[process][systematic].keys():
                    if not parent_process in histograms.keys():
                        histograms[parent_process] = dict((syst,
                                                    dict((sel, None) for sel in
                                                         histograms[process][syst].keys()))
                                                   for syst in histograms[process].keys())
                    if selection not in histograms[process][systematic]:
                        print "could not find selection ", selection, " for process ", process
                        continue
                    print histograms[process][systematic][selection].GetName()
                    new_hist_name = histograms[process][systematic][selection].GetName().replace(process, parent_process)
                    if histograms[parent_process][systematic][selection] is None:
                        new_hist_name = histograms[process][systematic][selection].GetName().replace(
                            process, parent_process)
                        histograms[parent_process][systematic][selection] = histograms[process][systematic][
                            selection].Clone(new_hist_name)
                    else:
                        histograms[parent_process][systematic][selection].Add(
                            histograms[process][systematic][selection].Clone(new_hist_name))
            histograms.pop(process)

    def calculate_sm_total(self):
        sm_total_cutflows = {}
        for process, systematics in self.cutflow_hists.iteritems():
            if "data" in process.lower():
                continue
            for systematic, regions in systematics.iteritems():
                if systematic not in sm_total_cutflows.keys():
                    sm_total_cutflows[systematic] = {}
                for region, cutflow_hist in regions.iteritems():
                    if region not in sm_total_cutflows[systematic].keys():
                        sm_total_cutflows[systematic][region] = cutflow_hist.Clone()
                        continue
                    sm_total_cutflows[systematic][region].Add(cutflow_hist)
        self.cutflow_hists["SMTotal"] = sm_total_cutflows

    def _analyse_cutflow(self, cutflow_hist, raw_cutflow_hist, process=None):
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
            for process in self.cutflows[systematic].keys():
                for cutflow in self.cutflows[systematic][process].values():
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
        for process in self.cutflows[systematic].keys():

            for selection, cutflow in self.cutflows[systematic][process].items():
                cutflow_tmp = self.stringify(cutflow)
                if selection not in cutflow_tables.keys():
                    cutflow_tables[selection] = cutflow_tmp
                    continue
                cutflow_tables[selection] = rec_append_fields(cutflow_tables[selection],
                                                              [i + process for i in cutflow_tmp.dtype.names[1:]],
                                                              [cutflow_tmp[n] for n in cutflow_tmp.dtype.names[1:]])
            headers = ['Cut'] + [x for elem in self.cutflows[systematic].keys() for x in (elem, '')]
            self.cutflow_tables = {k: tabulate(v.transpose(),
                                               headers=headers,
                                               tablefmt=self.format,
                                               floatfmt='.2f')
                                   for k, v in cutflow_tables.iteritems()}

    def stringify(self, cutflow):
        def format_yield(value, uncertainty):
            if value > 10000.:
                return '{:.3e}'.format(value)
            else:
                return '{:.2f}'.format(value)
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
        available_cutflows = self.cutflow_tables.keys()
        print "######## Selection menu  ########"
        print "Available cutflows for printing: "
        print "--------------------------------"
        for i, region in enumerate(available_cutflows):
            print i, ")", region
        print "a) all"
        user_input = raw_input(
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
            print "{:s}Invalid input {:s}. Going for default.\033[0m".format("\033[91m", user_input)
            selections = ["BaseSelection"]
        for selection, cutflow in self.cutflow_tables.iteritems():
            if selection not in selections:
                continue
            print
            print "Cutflow for region %s" % selection
            print cutflow

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
        if process not in self.cutflow_hists.keys():
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
        for process, systematics in self.cutflow_hists.items():
            for systematic, cutflows in systematics.items():
                if "smtotal" in process.lower():
                    continue
                for region, cutflow_hist in cutflows.items():
                    flipped[systematic][region][process] = cutflow_hist
        plot_config = PlotConfig(name=None, dist=None, ytitle="Events", logy=True)
        for region in flipped['Nominal'].keys():
            plot_config.name = "{:s}_cutflow".format(region)
            cutflow_hists = {process: hist for process, hist in flipped["Nominal"][region].iteritems()
                             if "smtotal" not in process.lower()}
            for process, cutflow_hist in cutflow_hists.iteritems():
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
