from __future__ import print_function
from __future__ import division

from builtins import next
from builtins import str
from past.utils import old_div
from builtins import object
import collections
from copy import copy

import PyAnalysisTools.PlottingUtils.Formatting as FM
import PyAnalysisTools.PlottingUtils.PlotableObject as PO
import PyAnalysisTools.PlottingUtils.PlottingTools as PT
import ROOT
from PyAnalysisTools.PlottingUtils import HistTools as HT
from PyAnalysisTools.PlottingUtils import set_batch_mode
from PyAnalysisTools.PlottingUtils.BasePlotter import BasePlotter
from PyAnalysisTools.PlottingUtils.PlotConfig import get_histogram_definition
from PyAnalysisTools.base.ProcessConfig import find_process_config, parse_and_build_process_config
from PyAnalysisTools.PlottingUtils.RatioPlotter import RatioPlotter
from PyAnalysisTools.base.FileHandle import FileHandle
from PyAnalysisTools.base import _logger, InvalidInputError
from PyAnalysisTools.base.JSONHandle import JSONHandle
from PyAnalysisTools.base.Modules import load_modules
from PyAnalysisTools.base.OutputHandle import OutputFileHandle


class ComparisonReader(object):
    def __init__(self, **kwargs):
        if 'input_files' not in kwargs:
            _logger.error('No input file provided')
            raise InvalidInputError('Missing input files')
        kwargs.setdefault('compare_files', None)
        self.input_files = kwargs['input_files']
        self.compare_files = kwargs['compare_files']
        self.tree_name = kwargs['tree_name']
        for opt, val in list(kwargs.items()):
            if not hasattr(self, opt):
                setattr(self, opt, val)

    def get_instance(self, plot_config):
        if self.compare_files:
            if hasattr(plot_config, 'multi_ref') and plot_config.multi_ref:
                print("Using MultiFileMultiRefReader instance")
                _logger.debug("Using MultiFileMultiRefReader instance")
                return MultiFileMultiRefReader(plot_config=plot_config, **self.__dict__)
            else:
                print("Using MultiFileSingleRefReader instance")
                _logger.debug("Using MultiFileSingleRefReader instance")
                return MultiFileSingleRefReader(plot_config=plot_config, **self.__dict__)
        else:
            if hasattr(plot_config, 'multi_ref') and plot_config.multi_ref:
                print("Using SingleFileMultiRefReader instance")
                _logger.debug("Using SingleFileMultiRefReader instance")
                return SingleFileMultiRefReader(plot_config=plot_config, **self.__dict__)
            else:
                print("Using SingleFileSingleRefReader instance")
                _logger.debug("Using SingleFileSingleRefReader instance")
                return SingleFileSingleRefReader(plot_config=plot_config, **self.__dict__)

    def get_data(self):
        data = {}
        for plot_config in self.plot_configs:
            getter = self.get_instance(plot_config)
            data[plot_config] = getter.get_data()
        return data

    def make_hist(self, file_handle, plot_config, cut_name, cut_string, tree_name=None):
        if file_handle.is_data:
            cut_string = cut_string.replace('DATA:', '')
        else:
            cut_string = '&&'.join([ct for ct in cut_string.split("&&") if 'DATA' not in ct])
        file_handle.open()  # ?
        hist = get_histogram_definition(plot_config)
        hist.SetName('_'.join([hist.GetName(), file_handle.process.process_name, cut_name]))
        if tree_name is None:
            tree_name = self.tree_name
        try:
            file_handle.fetch_and_link_hist_to_tree(tree_name, hist, plot_config.dist, cut_string,
                                                    tdirectory=self.tree_dir_name)
            hist.SetName(hist.GetName() + '_' + file_handle.process.process_name)
            _logger.debug("try to access config for process %s" % file_handle.process.process_name)
        except Exception as e:
            raise e
        return hist

    def make_hists(self, file_handles, plot_config, cut_name, cut_string, tree_name=None):
        result = None
        hists = []
        for fh in file_handles:
            hist = self.make_hist(fh, plot_config, cut_name, cut_string, tree_name)
            hists.append(hist)
            if result is None:
                result = hist.Clone()
                ROOT.SetOwnership(result, False)
                continue
            result.Add(hist)
        result.SetDirectory(0)
        list([fh.close() for fh in file_handles])
        return result

    @staticmethod
    def merge_file_handles(file_handles, process_configs):
        # def find_parent_process(process):
        #     parent_process = filter(lambda c: hasattr(c[1], 'subprocesses') and process in c[1].subprocesses,
        #                             process_configs.iteritems())
        #     try:
        #         return parent_process[0][0]
        #     except IndexError:
        #         _logger.error("Could not find parent process for process {:s}".format(process))
        #         print "Available process configs:", process_configs
        #         exit(-1)

        tmp_file_handles = collections.OrderedDict()
        for fh in file_handles:
            parent_process = find_process_config(fh.process, process_configs)  # find_parent_process(fh.process)
            if parent_process not in tmp_file_handles:
                tmp_file_handles[parent_process] = [fh]
                continue
            tmp_file_handles[parent_process].append(fh)
        return tmp_file_handles


class SingleFileSingleRefReader(ComparisonReader):
    def __init__(self, **kwargs):
        input_files = kwargs['input_files']
        compare_files = kwargs['input_files']
        # self.file_handles = [FileHandle(file_name=fn, switch_off_process_name_analysis=True) for fn in input_files]
        self.file_handles = [FileHandle(file_name=fn, dataset_info=kwargs['xs_config_file']) for fn in input_files]
        self.compare_file_handles = [FileHandle(file_name=fn, dataset_info=kwargs['xs_config_file']) for fn in
                                     compare_files]
        self.plot_config = kwargs['plot_config']
        self.tree_name = kwargs['tree_name']
        for opt, value in list(kwargs.items()):
            if not hasattr(self, opt):
                setattr(self, opt, value)
        self.file_handles = self.merge_file_handles(self.file_handles, self.process_configs)
        self.compare_file_handles = self.merge_file_handles(self.compare_file_handles, self.process_configs)

    def get_data(self):
        cuts = collections.OrderedDict()
        if not self.plot_config.cuts:
            setattr(self.plot_config, 'cuts', [])
        if hasattr(self.plot_config, 'cuts_l1'):
            for k_l1, v_l1 in list(self.plot_config.cuts_l1.items()):
                if hasattr(self.plot_config, 'cuts_l2'):
                    for k_l2, v_l2 in list(self.plot_config.cuts_l2.items()):
                        cuts[' '.join([k_l1, k_l2])] = '&&'.join(
                            [str(v) for v in self.plot_config.cuts + v_l1 + v_l2])
                else:
                    cuts[' '.join([k_l1])] = '&&'.join([str(v) for v in self.plot_config.cuts + v_l1])
        elif hasattr(self.plot_config, 'cuts_l2'):
            for k_l2, v_l2 in list(self.plot_config.cuts_l2.items()):
                cuts[' '.join([k_l2])] = '&&'.join([str(v) for v in self.plot_config.cuts + v_l2])
        else:
            cuts['cut'] = '&&'.join([str(v) for v in self.plot_config.cuts])
        cuts_ref = collections.OrderedDict([list(cuts.items())[0]])
        if len(cuts) == 2:
            cuts_comp = collections.OrderedDict([list(cuts.items())[-1]])
        else:
            cuts_comp = collections.OrderedDict(list(cuts.items())[1:])

        plotable_objects = []
        for k_cuts, v_cuts in list(cuts_ref.items()):
            reference = collections.OrderedDict()
            for process, file_handles in list(self.file_handles.items()):
                reference[process] = self.make_hists(file_handles, self.plot_config, k_cuts, v_cuts, self.tree_name)
            if self.plot_config.labels is not None:
                label = self.plot_config.labels[list(cuts_ref.keys()).index(k_cuts)]
            else:
                if k_cuts == 'cut':
                    label = ''
                else:
                    label = k_cuts
            for k_ref, v_ref in list(reference.items()):
                v_ref.SetDirectory(0)
                plotable_objects.append(
                    PO.PlotableObject(plot_object=v_ref, is_ref=True, ref_id=1, label=label, cuts=v_cuts,
                                      process=k_ref))
        for k_cuts, v_cuts in list(cuts_comp.items()):
            compare = collections.OrderedDict()
            for process, compare_file_handles in list(self.compare_file_handles.items()):
                compare[process] = self.make_hists(compare_file_handles, self.plot_config, k_cuts, v_cuts,
                                                   self.tree_name)
            if self.plot_config.labels is not None:
                label = self.plot_config.labels[list(cuts_comp.keys()).index(k_cuts) + 1]
            else:
                if k_cuts == 'cut':
                    label = ""
                else:
                    label = k_cuts
            for k_comp, v_comp in list(compare.items()):
                v_comp.SetDirectory(0)
                plotable_objects.append(
                    PO.PlotableObject(plot_object=v_comp, is_ref=False, ref_id=1, label=label, cuts=v_cuts,
                                      process=k_comp))
        return plotable_objects


class SingleFileMultiRefReader(ComparisonReader):
    def __init__(self, **kwargs):
        input_files = kwargs['input_files']
        compare_files = kwargs['input_files']
        self.file_handles = [FileHandle(file_name=fn, switch_off_process_name_analysis=True) for fn in input_files]
        self.compare_file_handles = [FileHandle(file_name=fn, switch_off_process_name_analysis=True) for fn in
                                     compare_files]
        self.plot_config = kwargs['plot_config']
        self.tree_name = kwargs['tree_name']
        for opt, value in list(kwargs.items()):
            if not hasattr(self, opt):
                setattr(self, opt, value)
        self.file_handles = self.merge_file_handles(self.file_handles, self.process_configs)
        self.compare_file_handles = self.merge_file_handles(self.compare_file_handles, self.process_configs)

    def get_data(self):
        cuts_ref = collections.OrderedDict()
        if not self.plot_config.cuts:
            setattr(self.plot_config, 'cuts', [])
        if hasattr(self.plot_config, 'cuts_l1'):
            if hasattr(self.plot_config, 'cuts_l2'):
                for k_l2, v_l2 in list(self.plot_config.cuts_l2.items()):
                    cuts_ref[' '.join([list(self.plot_config.cuts_l1.keys())[0], k_l2])] = '&&'.join(
                        [str(v) for v in self.plot_config.cuts + list(self.plot_config.cuts_l1.values())[0] + v_l2])
            else:
                cuts_ref[list(self.plot_config.cuts_l1.keys())[0]] = '&&'.join(
                    [str(v) for v in self.plot_config.cuts + list(self.plot_config.cuts_l1.values())[0]])
        elif hasattr(self.plot_config, 'cuts_l2'):
            cuts_ref[list(self.plot_config.cuts_l2.keys())[0]] = '&&'.join(
                [str(v) for v in self.plot_config.cuts + list(self.plot_config.cuts_l2.values())[0]])
        else:
            cuts_ref['cut'] = '&&'.join([str(v) for v in self.plot_config.cuts])
        cuts_comp = collections.OrderedDict()
        if not self.plot_config.cuts:
            setattr(self.plot_config, 'cuts', [])
        if hasattr(self.plot_config, 'cuts_l1'):
            for k_l1, v_l1 in list(self.plot_config.cuts_l1.items()):
                if list(self.plot_config.cuts_l1.keys()).index(k_l1) == 0:
                    continue
                if hasattr(self.plot_config, 'cuts_l2'):
                    for k_l2, v_l2 in list(self.plot_config.cuts_l2.items()):
                        cuts_comp[' '.join([k_l1, k_l2])] = '&&'.join(
                            [str(v) for v in self.plot_config.cuts + v_l1 + v_l2])
                else:
                    cuts_comp[' '.join([k_l1])] = '&&'.join([str(v) for v in self.plot_config.cuts + v_l1])
        elif hasattr(self.plot_config, 'cuts_l2'):
            for k_l2, v_l2 in list(self.plot_config.cuts_l2.items()):
                if list(self.plot_config.cuts_l2.keys()).index(k_l2) == 0:
                    continue
                cuts_comp[' '.join([k_l2])] = '&&'.join([str(v) for v in self.plot_config.cuts + v_l2])
        else:
            cuts_comp['cut'] = '&&'.join([str(v) for v in self.plot_config.cuts])

        plotable_objects = []
        for k_cuts, v_cuts in list(cuts_ref.items()):
            reference = collections.OrderedDict()
            for process, file_handles in list(self.file_handles.items()):
                reference[process] = self.make_hists(file_handles, self.plot_config, k_cuts, v_cuts, self.tree_name)
            if self.plot_config.labels is not None:
                label = self.plot_config.labels[list(cuts_ref.keys()).index(k_cuts)]
            else:
                if k_cuts == 'cut':
                    label = ""
                else:
                    label = k_cuts
            for k_ref, v_ref in list(reference.items()):
                v_ref.SetDirectory(0)
                plotable_objects.append(
                    PO.PlotableObject(plot_object=v_ref, is_ref=True, ref_id=list(cuts_ref.keys()).index(k_cuts),
                                      label=label,
                                      cuts=v_cuts, process=k_ref))
        for k_cuts, v_cuts in list(cuts_comp.items()):
            compare = collections.OrderedDict()
            for process, compare_file_handles in list(self.compare_file_handles.items()):
                compare[process] = self.make_hists(file_handles, self.plot_config, k_cuts, v_cuts,
                                                   self.tree_name)
            if self.plot_config.labels is not None:
                label = self.plot_config.labels[list(cuts_comp.keys()).index(k_cuts)]
            else:
                if k_cuts == 'cut':
                    label = ""
                else:
                    label = k_cuts
            for k_comp, v_comp in list(compare.items()):
                v_comp.SetDirectory(0)
                ref_id = list(cuts_comp.keys()).index(k_cuts) % len(cuts_ref)
                plotable_objects.append(
                    PO.PlotableObject(plot_object=v_comp, is_ref=False, ref_id=ref_id, label=label, cuts=v_cuts,
                                      process=k_comp))
        return plotable_objects


class MultiFileSingleRefReader(ComparisonReader):
    def __init__(self, **kwargs):
        input_files = kwargs['input_files']
        compare_files = kwargs['compare_files']  # why was there this included here: kwargs['input_files'] +
        self.file_handles = [FileHandle(file_name=fn, switch_off_process_name_analysis=True) for fn in input_files]
        self.compare_file_handles = [FileHandle(file_name=fn, switch_off_process_name_analysis=True) for fn in
                                     compare_files]
        self.plot_config = kwargs['plot_config']
        self.tree_name = kwargs['tree_name']
        for opt, value in list(kwargs.items()):
            if not hasattr(self, opt):
                setattr(self, opt, value)
        self.file_handles = self.merge_file_handles(self.file_handles, self.process_configs)
        self.compare_file_handles = self.merge_file_handles(self.compare_file_handles, self.process_configs)

    def get_data(self):
        cuts = collections.OrderedDict()
        if not self.plot_config.cuts:
            setattr(self.plot_config, 'cuts', [])
        if hasattr(self.plot_config, 'cuts_l1'):
            for k_l1, v_l1 in list(self.plot_config.cuts_l1.items()):
                if hasattr(self.plot_config, 'cuts_l2'):
                    for k_l2, v_l2 in list(self.plot_config.cuts_l2.items()):
                        cuts[' '.join([k_l1, k_l2])] = '&&'.join(
                            [str(v) for v in self.plot_config.cuts + v_l1 + v_l2])
                else:
                    cuts[' '.join([k_l1])] = '&&'.join([str(v) for v in self.plot_config.cuts + v_l1])
        elif hasattr(self.plot_config, 'cuts_l2'):
            for k_l2, v_l2 in list(self.plot_config.cuts_l2.items()):
                cuts[' '.join([k_l2])] = '&&'.join([str(v) for v in self.plot_config.cuts + v_l2])
        else:
            cuts['cut'] = '&&'.join([str(v) for v in self.plot_config.cuts])
        cuts_ref = collections.OrderedDict([list(cuts.items())[0]])
        cuts_comp = cuts_ref  # cuts
        plotable_objects = []
        for k_cuts, v_cuts in list(cuts_ref.items()):
            reference = collections.OrderedDict()
            for process, file_handles in list(self.file_handles.items()):
                reference[process] = self.make_hists(file_handles, self.plot_config, k_cuts, v_cuts, self.tree_name)
            if self.plot_config.labels is not None:
                label = self.plot_config.labels[list(cuts_ref.keys()).index(k_cuts)]
            else:
                if k_cuts == 'cut':
                    label = ""
                else:
                    label = k_cuts
            for k_ref, v_ref in list(reference.items()):
                v_ref.SetDirectory(0)
                plotable_objects.append(
                    PO.PlotableObject(plot_object=v_ref, is_ref=True, ref_id=1, label=label, cuts=v_cuts,
                                      process=k_ref))

        for k_cuts, v_cuts in list(cuts_comp.items()):
            compare = collections.OrderedDict()
            for process, compare_file_handles in list(self.compare_file_handles.items()):
                compare[process] = self.make_hists(compare_file_handles, self.plot_config, k_cuts, v_cuts,
                                                   self.tree_name)
            if self.plot_config.labels is not None:
                label = self.plot_config.labels[list(cuts_comp.keys()).index(k_cuts)]
            else:
                if k_cuts == 'cut':
                    label = ""
                else:
                    label = k_cuts
            for k_comp, v_comp in list(compare.items()):
                v_comp.SetDirectory(0)
                plotable_objects.append(
                    PO.PlotableObject(plot_object=v_comp, is_ref=False, ref_id=1, label=label, cuts=v_cuts,
                                      process=k_comp))
        #del plotable_objects[1]
        return plotable_objects


class MultiFileMultiRefReader(ComparisonReader):
    def __init__(self, **kwargs):
        input_files = kwargs['input_files']
        compare_files = kwargs['compare_files']
        self.file_handles = [FileHandle(file_name=fn, switch_off_process_name_analysis=True) for fn in input_files]
        self.compare_file_handles = [FileHandle(file_name=fn, switch_off_process_name_analysis=True) for fn in
                                     compare_files]
        self.plot_config = kwargs['plot_config']
        self.tree_name = kwargs['tree_name']
        for opt, value in list(kwargs.items()):
            if not hasattr(self, opt):
                setattr(self, opt, value)
        self.file_handles = self.merge_file_handles(self.file_handles, self.process_configs)
        self.compare_file_handles = self.merge_file_handles(self.compare_file_handles, self.process_configs)

    def get_data(self):
        cuts = collections.OrderedDict()
        if not self.plot_config.cuts:
            setattr(self.plot_config, 'cuts', [])
        if hasattr(self.plot_config, 'cuts_l1'):
            for k_l1, v_l1 in list(self.plot_config.cuts_l1.items()):
                if hasattr(self.plot_config, 'cuts_l2'):
                    for k_l2, v_l2 in list(self.plot_config.cuts_l2.items()):
                        cuts[' '.join([k_l1, k_l2])] = '&&'.join(
                            [str(v) for v in self.plot_config.cuts + v_l1 + v_l2])
                else:
                    cuts[' '.join([k_l1])] = '&&'.join([str(v) for v in self.plot_config.cuts + v_l1])
        elif hasattr(self.plot_config, 'cuts_l2'):
            for k_l2, v_l2 in list(self.plot_config.cuts_l2.items()):
                cuts[' '.join([k_l2])] = '&&'.join([str(v) for v in self.plot_config.cuts + v_l2])
        else:
            cuts['cut'] = '&&'.join([str(v) for v in self.plot_config.cuts])

        plotable_objects = []
        for k_cuts, v_cuts in list(cuts.items()):
            reference = collections.OrderedDict()
            compare = collections.OrderedDict()
            for process, file_handles in list(self.file_handles.items()):
                reference[process] = self.make_hists(file_handles, self.plot_config, k_cuts, v_cuts, self.tree_name)
            for process, compare_file_handles in list(self.compare_file_handles.items()):
                compare[process] = self.make_hists(compare_file_handles, self.plot_config, k_cuts, v_cuts,
                                                   self.tree_name)
            if self.plot_config.labels is not None:
                label = self.plot_config.labels[list(cuts.keys()).index(k_cuts)]
            else:
                if k_cuts == 'cut':
                    label = ""
                else:
                    label = k_cuts
            for k_ref, v_ref in list(reference.items()):
                v_ref.SetDirectory(0)
                if len(reference) == len(compare):
                    ref_id = ((list(reference.keys()).index(k_ref) + 1) * 100) + \
                             ((list(reference.keys()).index(k_ref) + 1) * 10) + list(cuts.keys()).index(k_cuts)
                else:
                    ref_id = ((0 + 1) * 100) + ((0 + 1) * 10) + list(cuts.keys()).index(k_cuts)
                plotable_objects.append(
                    PO.PlotableObject(plot_object=v_ref, is_ref=True, ref_id=ref_id, label=label, cuts=v_cuts,
                                      process=k_ref))
            for k_comp, v_comp in list(compare.items()):
                if len(reference) == len(compare):
                    ref_id = ((list(compare.keys()).index(k_comp) + 1) * 100) + (
                            (list(compare.keys()).index(k_comp) + 1) * 10) + list(cuts.keys()).index(k_cuts)  # noqa: E126, E501
                else:
                    ref_id = ((0 + 1) * 100) + ((0 + 1) * 10) + list(cuts.keys()).index(k_cuts)
                v_comp.SetDirectory(0)
                plotable_objects.append(
                    PO.PlotableObject(plot_object=v_comp, is_ref=False, ref_id=ref_id, label=label, cuts=v_cuts,
                                      process=k_comp))
        return plotable_objects


class ComparisonPlotter(BasePlotter):
    def __init__(self, **kwargs):
        if 'input_files' not in kwargs:
            _logger.error("No input files provided")
            raise InvalidInputError("Missing input files")
        if 'plot_config_files' not in kwargs:
            _logger.error("No config file provided")
            raise InvalidInputError("Missing config")
        if 'output_dir' not in kwargs:
            _logger.warning("No output directory given. Using ./")
        kwargs.setdefault('batch', True)
        kwargs.setdefault('tree_name', None)
        kwargs.setdefault('output_dir', './')
        kwargs.setdefault('output_tag', None)
        kwargs.setdefault('process_config_files', None)
        kwargs.setdefault('ref_mod_modules', [])
        kwargs.setdefault('inp_mod_modules', None)
        kwargs.setdefault('read_hist', False)
        kwargs.setdefault('n_files_handles', 1)
        kwargs.setdefault('nfile_handles', 1)
        kwargs.setdefault('ref_module_config_file', None)
        kwargs.setdefault('module_config_file', [])
        kwargs.setdefault('json', False)
        kwargs.setdefault('file_extension', ['.pdf'])
        if kwargs['json']:
            kwargs = JSONHandle(kwargs['json']).load()
        set_batch_mode(kwargs['batch'])
        super(ComparisonPlotter, self).__init__(**kwargs)
        if self.tree_dir_name == "" or self.tree_dir_name.lower() == 'none':
            self.tree_dir_name = None
            kwargs['tree_dir_name'] = None
        self.input_files = kwargs['input_files']
        self.output_handle = OutputFileHandle(overload='comparison', output_file_name='Compare.root',
                                              extension=kwargs['file_extension'], **kwargs)
        # self.color_palette = [
        #     ROOT.kGray+3,
        #     ROOT.kPink+7,
        #     ROOT.kAzure+4,
        #     ROOT.kSpring-9,
        #     ROOT.kOrange-3,
        #     ROOT.kCyan-6,
        #     ROOT.kPink-7,
        #     ROOT.kSpring-7,
        #     ROOT.kPink-1,
        #     ROOT.kGray+3,
        #     ROOT.kPink+7,
        #     ROOT.kAzure+4,
        #     ROOT.kSpring-9,
        #     ROOT.kOrange-3,
        #     ROOT.kCyan-6,
        #     ROOT.kPink-7,
        #     ROOT.kSpring-7,
        #     ROOT.kPink-1,
        # ]
        # self.style_palette = [21,
        #                       20,
        #                       22,
        #                       23,
        #                       25,
        #                       24,
        #                       26,
        #                       32,
        #                       5,
        #                       2,
        #                       25,
        #                       24,
        #                       26,
        #                       32,
        #                       21,
        #                       20,
        #                       22,
        #                       23,
        #                       ]
        for attr, value in list(kwargs.items()):
            if not hasattr(self, attr):
                setattr(self, attr, value)
        # if self.systematics is None:
        #     self.systematics = 'Nominal'

        if 'process_config_files' in kwargs:
            self.process_configs = parse_and_build_process_config(kwargs['process_config_files'])

        self.ref_modules = load_modules(kwargs['ref_mod_modules'], self)
        self.modules = load_modules(kwargs['module_config_file'], self)
        self.modules_data_providers = [m for m in self.modules if m.type == 'DataProvider']
        self.module_filters = [m for m in self.modules if m.type == 'Filter']
        self.analyse_plot_config()
        # self.update_color_palette()
        self.getter = ComparisonReader(plot_configs=self.plot_configs, process_configs=self.process_configs, **kwargs)
        if not kwargs['json']:
            JSONHandle(kwargs['output_dir'], **kwargs).dump()

    def analyse_plot_config(self):
        if self.plot_configs is None:
            return None
        pc = next((pc for pc in self.plot_configs if pc.name == 'parse_from_file'), None)
        if pc is None:
            return
        if not hasattr(self, 'reference_files'):
            _logger.error("Request to parse plot configs from file, but no reference file given. Breaking up!")
            exit(0)
        file_handles = [FileHandle(file_name=reference_file) for reference_file in self.reference_files]
        objects = []
        for file_handle in file_handles:
            objects += file_handle.get_objects_by_type('TCanvas')
        self.plot_configs.remove(pc)
        for obj in objects:
            new_pc = copy(pc)
            new_pc.dist = obj.GetName()
            self.plot_configs.append(new_pc)

    def update_color_palette(self):
        if isinstance(self.common_config.colors[0], str):
            self.color_palette = [getattr(ROOT, 'k' + color.capitalize()) for color in self.common_config.colors]
        elif isinstance(self.common_config.colors[0], int):
            self.color_palette = [color for color in self.common_config.colors]
        else:
            _logger.warning("Unsuppored type %s for colors in common_config" % type(self.common_config.colors[0]))

    def make_comparison_plots(self):
        data = self.getter.get_data()
        for k, v in list(data.items()):
            self.make_comparison_plot(k, v)
        self.output_handle.write_and_close()

    def make_comparison_plot(self, plot_config, data):
        for i in data:
            HT.merge_overflow_bins(i.plot_object)
            HT.merge_underflow_bins(i.plot_object)
        reference_hists = [x for x in data if x.is_ref]
        compare_hists = [x for x in data if not x.is_ref]

        offset = len(reference_hists) if (len(reference_hists) != len(compare_hists) or len(reference_hists) == 1) \
            else 0
        for i, ref in enumerate(reference_hists):
            setattr(ref, 'draw_option', plot_config.draw)
            if plot_config.draw in ['Marker', 'marker', 'P', 'p']:
                setattr(ref, 'marker_color',
                        PO.color_palette[i - (int(old_div(i, len(PO.color_palette))) * len(PO.color_palette))])
                setattr(ref, 'marker_style', PO.marker_style_palette_filled[
                    i - (int(old_div(i, len(PO.marker_style_palette_filled))) * len(PO.marker_style_palette_filled))])
                setattr(ref, 'line_color',
                        PO.color_palette[i - (int(old_div(i, len(PO.color_palette))) * len(PO.color_palette))])
            elif plot_config.draw in ['Line', 'line', 'L', 'l']:
                setattr(ref, 'line_color',
                        PO.color_palette[i - (int(old_div(i, len(PO.color_palette))) * len(PO.color_palette))])
                setattr(ref, 'line_style', PO.line_style_palette_homogen[
                    i - (int(old_div(i, len(PO.line_style_palette_homogen))) * len(PO.line_style_palette_homogen))])
            elif plot_config.draw in ['Hist', 'hist', 'H', 'h']:
                setattr(ref, 'fill_color',
                        PO.color_palette[i - (int(old_div(i, len(PO.color_palette))) * len(PO.color_palette))])
                setattr(ref, 'fill_style',
                        PO.fill_style_palette_left[
                            i - (int(old_div(i, len(PO.color_palette))) * len(PO.color_palette))])
                setattr(ref, 'line_color',
                        PO.color_palette[i - (int(old_div(i, len(PO.color_palette))) * len(PO.color_palette))])
                setattr(ref, 'marker_color',
                        PO.color_palette[i - (int(old_div(i, len(PO.color_palette))) * len(PO.color_palette))])

        for i, comp in enumerate(compare_hists):
            setattr(comp, 'draw_option', plot_config.draw)
            index = (i + offset) - (int(old_div((i + offset), len(PO.color_palette))) * len(PO.color_palette))
            if plot_config.draw in ['Marker', 'marker', 'P', 'p']:
                setattr(comp, 'marker_color', PO.color_palette[index])
                setattr(comp, 'marker_style', PO.marker_style_palette_empty[(i + offset) - (
                        int(old_div((i + offset), len(PO.marker_style_palette_empty)))
                        * len(PO.marker_style_palette_empty))])
                setattr(comp, 'line_color', PO.color_palette[index])
            elif plot_config.draw in ['Line', 'line', 'L', 'l']:
                setattr(comp, 'line_color', PO.color_palette[index])
                setattr(comp, 'line_style', PO.line_style_palette_heterogen[(i + offset) - (
                        int(old_div((i + offset), len(PO.line_style_palette_heterogen)))
                        * len(PO.line_style_palette_heterogen))])
            elif plot_config.draw in ['Hist', 'hist', 'H', 'h']:
                setattr(comp, 'fill_color', PO.color_palette[index])
                setattr(comp, 'fill_style', PO.fill_style_palette_right[index])
                setattr(comp, 'line_color', PO.color_palette[index])
                setattr(comp, 'marker_color', PO.color_palette[index])

        # plot_config.color = self.color_palette
        # plot_config.styles = self.style_palette

        # canvas = PT.plot_objects(map(lambda x : x.plot_object, reference_hists+compare_hists),
        # plot_config, plotable_objects=reference_hists+compare_hists)
        canvas = PT.plot_objects(reference_hists + compare_hists, plot_config)
        canvas.SetName(plot_config.name.replace(' ', '_'))

        if self.process_configs:
            for ref in reference_hists:
                if hasattr(plot_config, 'ignore_process_labels') and plot_config.ignore_process_labels:
                    ref.label = '{:s}'.format(ref.label)
                else:
                    ref.label = '{:s} {:s}'.format(find_process_config(ref.process.name, self.process_configs).label,
                                                   ref.label)
            for comp in compare_hists:
                if hasattr(plot_config, 'ignore_process_labels') and plot_config.ignore_process_labels:
                    comp.label = '{:s}'.format(comp.label)
                else:
                    comp.label = '{:s} {:s}'.format(find_process_config(comp.process.name, self.process_configs).label,
                                                    comp.label)

        ROOT.SetOwnership(canvas, False)

        if plot_config.enable_legend:
            FM.add_legend_to_canvas(canvas, ratio=plot_config.ratio,
                                    labels=[x.label for x in reference_hists + compare_hists],
                                    plot_objects=[x.plot_object for x in reference_hists + compare_hists],
                                    **plot_config.legend_options)
        if plot_config.lumi:
            FM.decorate_canvas(canvas, plot_config)

        if plot_config.stat_box:
            FM.add_stat_box_to_canvas(canvas)

        if plot_config.ratio is False:
            canvas.SetName(plot_config.name.replace(' ', '_'))
            self.output_handle.register_object(canvas)
        else:
            if plot_config.ratio_config is not None:
                plot_config.ratio_config.draw = plot_config.draw
                plot_config = plot_config.ratio_config
            if not plot_config.name.startswith('ratio'):
                plot_config.name = 'ratio_' + plot_config.name
            canvas_ratio = None
            for ref in reference_hists:
                for comp in [x.plot_object for x in [y for y in compare_hists if y.ref_id == ref.ref_id]]:
                    if canvas_ratio:
                        ROOT.SetOwnership(canvas_ratio, False)
                        canvas_ratio.cd()
                        ratio_plotter = RatioPlotter(reference=ref.plot_object, compare=comp, plot_config=plot_config)
                        hist_ratio = ratio_plotter.ratio_calculator.calculate_ratio_hist(ref.plot_object, comp)
                        hist_ratio.Draw('same')
                        ROOT.SetOwnership(canvas_ratio, False)
                    else:
                        canvas_ratio = RatioPlotter(reference=ref.plot_object, compare=comp,
                                                    plot_config=plot_config).make_ratio_plot()
                        ROOT.SetOwnership(canvas_ratio, False)
            if canvas_ratio is not None:
                canvas_ratio.SetName(plot_config.name.replace(' ', '_') + '_ratio')
                # self.output_handle.register_object(canvas)
                # self.output_handle.register_object(canvas_ratio)
                canvas_combined = RatioPlotter.add_ratio_to_canvas(canvas, canvas_ratio)
                self.output_handle.register_object(canvas_combined)
            else:
                _logger.error('Ratio canvas was not created.')
                print('reference hists: ', reference_hists)
                print('compare hists: ', compare_hists)
