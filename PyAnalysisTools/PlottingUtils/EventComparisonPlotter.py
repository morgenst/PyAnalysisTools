from __future__ import print_function
from __future__ import division

from builtins import next
from builtins import str
from past.utils import old_div
from builtins import object
import collections
from copy import copy

import pandas as pd

import PyAnalysisTools.PlottingUtils.Formatting as FM
import PyAnalysisTools.PlottingUtils.PlotableObject as PO
import PyAnalysisTools.PlottingUtils.PlottingTools as PT
import ROOT
from PyAnalysisTools.AnalysisTools.MLHelper import Root2NumpyConverter
from PyAnalysisTools.PlottingUtils import HistTools as HT
from PyAnalysisTools.PlottingUtils import set_batch_mode
from PyAnalysisTools.PlottingUtils.BasePlotter import BasePlotter
from PyAnalysisTools.PlottingUtils.PlotConfig import get_histogram_definition, \
    parse_and_build_process_config, find_process_config
from PyAnalysisTools.ROOTUtils.FileHandle import FileHandle
from PyAnalysisTools.base import _logger, InvalidInputError
from PyAnalysisTools.base.JSONHandle import JSONHandle
from PyAnalysisTools.base.Modules import load_modules
from PyAnalysisTools.base.OutputHandle import OutputFileHandle


class EventComparisonReader(object):
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
        return Reader(plot_config=plot_config, **self.__dict__)

    
    def get_data(self):
        data = {}
        for plot_config in self.plot_configs:
            getter = self.get_instance(plot_config)
            data[plot_config] = getter.get_data()
        return data

    
    def make_hist(self, file_handle, compare_file_handle, plot_config, cut_name, cut_string, tree_name=None):
        hist = get_histogram_definition(plot_config)
        print(file_handle.process)
        print(cut_name)
        hist.SetName('_'.join([hist.GetName(), file_handle.process, cut_name]))
        if tree_name is None:
            tree_name = self.tree_name
        # try:

        t1 = file_handle.tfile.Get('Nominal/BaseSelection_tree_finalSelection')
        t2 = compare_file_handle.Get('Nominal/BaseSelection_tree_finalSelection')
        # t1 = file_handle.Get(tree_name)
        # t2 = compare_file_handle.Get(tree_name)

        
        if isinstance(hist, ROOT.TH1F):
            var = plot_config.dist
        else:
            var = plot_config.dist.split(":")[0]
            print(var)
            print(plot_config.dist)
            print(plot_config)
            
        branch_list = ['eventNumber', var]

        # cut_string = 'jet_n > 0 && jet_pt[0] > 60000. && MET_calo > 80000.'
        # cut_string = 'jet_n > 0'
        # if run == 'collisionRun':
        # branch_list.append('HLT_j55_0eta240_xe50_L1J30_EMPTYAcceptance')
        # cut_string += ' && passGRL==1 && HLT_j55_0eta240_xe50_L1J30_EMPTYAcceptance==0' # firstempty
        # cut_string += ' && passGRL==1 && HLT_j55_0eta240_xe50_L1J30_EMPTYAcceptance==1' # empty
        # else:
        #     branch_list.append('HLT_noalg_cosmiccalo_L1EM3_EMPTYAcceptance')
        #     branch_list.append('HLT_noalg_cosmiccalo_L1RD1_EMPTYAcceptance')
        #     branch_list.append('HLT_noalg_cosmiccalo_L1J12_EMPTYAcceptance')
        #     branch_list.append('HLT_noalg_cosmiccalo_L1J30_EMPTYAcceptance')
        #     branch_list.append('HLT_j0_L1J12_EMPTYAcceptance')
        #     branch_list.append('HLT_ht0_L1J12_EMPTYAcceptance')
        #     cut_string += ' && (HLT_noalg_cosmiccalo_L1EM3_EMPTYAcceptance==1 || HLT_noalg_cosmiccalo_L1RD1_EMPTYAcceptance==1 || HLT_noalg_cosmiccalo_L1J12_EMPTYAcceptance==1 || HLT_noalg_cosmiccalo_L1J30_EMPTYAcceptance==1 || HLT_j0_L1J12_EMPTYAcceptance==1 || HLT_ht0_L1J12_EMPTYAcceptance==1)'

        converter = Root2NumpyConverter(branch_list)
        
        data1 = pd.DataFrame(converter.convert_to_array(t1, cut_string))
        data2 = pd.DataFrame(converter.convert_to_array(t2, cut_string))
        if var.endswith('_n'):
            data1[[var]] = data1[[var]].astype(int)
            data2[[var]] = data2[[var]].astype(int)

        # if var.endswith('calo'):
        #     hist = ROOT.TH1D(var, '', 200, -200, 200.)
        # else:
        #     hist = ROOT.TH1D(var, '', 100, -100, 100.)

        print(type(hist))
        if isinstance(hist, ROOT.TH1F):
            for _, i in data1.iterrows():
                e2 = data2[data2['eventNumber'] == i['eventNumber']]
                if len(e2) == 0:
                    hist.Fill(-99999)
                    continue
                hist.Fill((i[var] - e2[var]))
                # if var.startswith('MET' or 'muon'):
                #     hist.Fill((i[var] - e2[var])/1000.)
                # elif var.startswith('muon'):
                #     hist.Fill((i[var] - e2[var]))
                # else:
                #     li_jet = list(i[var])
                #     le2_jet = list(e2[var])[0]
                #     if len(li_jet) != len(le2_jet):
                #         hist.Fill(-88888)
                #         continue
                #     for j in range(len(li_jet)):
                #         if var.endswith('Pt') or var.endswith('pt'):
                #             hist.Fill((li_jet[j] - le2_jet[j])/1000.)
                #         else:
                #             hist.Fill(li_jet[j] - le2_jet[j])

            hist.SetName('_'.join([hist.GetName(), file_handle.process, cut_name]))
            # _logger.debug("try to access config for process %s" % file_handle.process)
            # except Exception as e:
            #     raise e
            return hist

        if isinstance(hist, ROOT.TH2F):
            print('starting 2D')
            for _, i in data1.iterrows():
                e_cos = data2[data2['eventNumber'] == i['eventNumber']]
                # print 'Start new'
                # print e_cos, "\n"
                # print e_cos[var], "\n"
                # print i, "\n"
                # print i[var], "\n"
                # print len(e_cos[var])
                # print len(i[var])
                value_data1=i[var]
                value_data2=e_cos[var]
                try:
                    if len(value_data1) == 0 or len(value_data2) == 0:
                        hist.Fill(-99999., -99999.)
                        continue
                    # print type(value_data2), type(value_data1)
                    # if len(value_data2[var]) == 0:
                    #     hist.Fill(-99999., -99999.)
                    #     continue
                    hist.Fill(value_data1, value_data2)
                except TypeError:
                    #print 'Filling: ', value_data1, len(value_data1)
                    if len(value_data2) == 0:
                        hist.Fill(-99999., -99999.)
                        continue
                    # print 'val 1: ', value_data1
                    # print 'val 2: ', value_data2
                    hist.Fill(value_data1, value_data2)
                    #pass
                # print 'FOO'
                # print type(value_data1), value_data1
                # exit()
                # if len(e_cos[var]) == 0 or len(i[var]) == 0:
                #     hist.Fill(-99999., -99999.)
                #     continue
                # li_col = list(i[var])
                # le_cos = list(e_cos[var])
                # if len(le_cos[0]) == 0 or len(li_col) == 0:
                #     hist.Fill(-99999., -99999.)
                #     continue
                # print (li_col[0]/1000.), (le_cos[0][0]/1000.), (li_col[0] - le_cos[0][0])
                #hist.Fill(li_col[0], le_cos[0][0])
            hist.SetName('_'.join([hist.GetName(), file_handle.process, cut_name]))
            return hist
        
    def make_hists(self, file_handles, compare_file_handles, plot_config, cut_name, cut_string, tree_name=None):
        result = None
        for fh in file_handles:
            hist = self.make_hist(fh, compare_file_handles.tfile, plot_config, cut_name, cut_string, tree_name)
            if result is None:
                result = hist
                continue
            result.Add(hist)
        return result
    
    @staticmethod
    def merge_histograms(histograms, process_configs):
        def expand():
            if process_configs is not None:
                for process_name in list(histograms.keys()):
                    _ = find_process_config(process_name, process_configs)

        expand()
        for process, process_config in list(process_configs.items()):
            if not hasattr(process_config, 'subprocesses'):
                continue
            for sub_process in process_config.subprocesses:
                if sub_process not in list(histograms.keys()):
                    continue
                if process not in list(histograms.keys()):
                    new_hist_name = histograms[sub_process].GetName().replace(sub_process, process)
                    histograms[process] = histograms[sub_process].Clone(new_hist_name)
                else:
                    histograms[process].Add(histograms[sub_process])
                histograms.pop(sub_process)

        for process in list(histograms.keys()):
            histograms[find_process_config(process, process_configs)] = histograms.pop(process)


class Reader(EventComparisonReader):
    def __init__(self, **kwargs):
        self.process_configs = kwargs['process_configs']
        input_files = kwargs['input_files']
        self.file_handles = [FileHandle(file_name=fn, switch_off_process_name_analysis=True) for fn in input_files]
        self.file_handles = self.merge_file_handles(self.file_handles, self.process_configs)        
        # if hasattr(plot_config, 'compare_files'):
        compare_files = kwargs['compare_files']
        self.compare_file_handles = [FileHandle(file_name=fn, switch_off_process_name_analysis=True) for fn in compare_files]
        self.compare_file_handles = self.merge_file_handles(self.compare_file_handles, self.process_configs)
        self.plot_config = kwargs['plot_config']
        self.tree_name = kwargs['tree_name']
        for opt, value in list(kwargs.items()):
            if not hasattr(self, opt):
                setattr(self, opt, value)

    @staticmethod
    def merge_file_handles(file_handles, process_configs):
        def find_parent_process(process):
            parent_process = [c for c in iter(list(process_configs.items())) if hasattr(c[1], 'subprocesses') and process in c[1].subprocesses]
            return parent_process[0][0]
        
        def expand():
            if process_configs is not None:
                for fh in file_handles:
                    _ = find_process_config(fh.process, process_configs)

        expand()
        tmp_file_handles = collections.OrderedDict()
        for fh in file_handles:
            parent_process = find_parent_process(fh.process)
            if parent_process not in tmp_file_handles:
                tmp_file_handles[parent_process] = [fh]
                continue
            tmp_file_handles[parent_process].append(fh)
        return tmp_file_handles

                
    def get_data(self):
        plotable_objects = []

        cut_string = '&&'.join([str(v) for v in self.plot_config.cuts])
        print(cut_string)
        
        reference = collections.OrderedDict()
        for process, file_handles in list(self.file_handles.items()):
            # compare_file_handle = self.compare_file_handles['collisionRun_cosmicsReco_standardOFCs'][0]
            # compare_file_handle = self.compare_file_handles['cosmicRun_cosmicsReco_standardOFCs'][0]
            # compare_file_handle = self.compare_file_handles['collisionRun_cosmicsReco_iterativeOFCs'][0]
            # print self.compare_file_handles['cosmicRun_cosmicsReco_iterativeOFCs'][0]
            compare_file_handle = list(self.compare_file_handles.items())[0][1][0]
            reference[process] = self.make_hists(file_handles, compare_file_handle, self.plot_config, '', cut_string, self.tree_name)
            
        for k_ref, v_ref in list(reference.items()):
            v_ref.SetDirectory(0)
            plotable_objects.append(PO.PlotableObject(plot_object=v_ref, label='', process=k_ref))
        return plotable_objects

class EventComparisonPlotter(BasePlotter):
    def __init__(self, **kwargs):
        if not 'input_files' in kwargs:
            _logger.error("No input files provided")
            raise InvalidInputError("Missing input files")
        if not 'plot_config_files' in kwargs:
            _logger.error("No config file provided")
            raise InvalidInputError("Missing config")
        if not 'output_dir' in kwargs:
            _logger.warning("No output directory given. Using ./")
        kwargs.setdefault('batch', True)
        kwargs.setdefault('tree_name', None)
        kwargs.setdefault('output_dir', './')
        kwargs.setdefault('output_tag', None)
        kwargs.setdefault('process_config_files', None)
        kwargs.setdefault('systematics', 'Nominal')
        kwargs.setdefault('ref_mod_modules', None)
        kwargs.setdefault('inp_mod_modules', None)
        kwargs.setdefault('read_hist', False)
        kwargs.setdefault('n_files_handles', 1)
        kwargs.setdefault('nfile_handles', 1)
        kwargs.setdefault('ref_module_config_file', None)
        kwargs.setdefault('module_config_file', None)
        kwargs.setdefault('json', False)
        kwargs.setdefault('file_extension', ['.pdf'])

        if kwargs['json']:
            kwargs = JSONHandle(kwargs['json']).load()
        set_batch_mode(kwargs['batch'])
        super(EventComparisonPlotter, self).__init__(**kwargs)
        self.input_files = kwargs['input_files']
        self.output_handle = OutputFileHandle(overload='eventComparison', output_file_name='EventCompare.root',
                                              extension=kwargs['file_extension'],  **kwargs)
        for attr, value in list(kwargs.items()):
            if not hasattr(self, attr):
                setattr(self, attr, value)
        # if self.systematics is None:
        #     self.systematics = 'Nominal'
            
        if 'process_config_files' in kwargs:
            self.process_configs = parse_and_build_process_config(kwargs['process_config_files'])
            self.expand_process_configs()
            
        self.ref_modules = load_modules(kwargs['ref_mod_modules'], self)
        self.modules = load_modules(kwargs['module_config_file'], self)
        self.modules_data_providers = [m for m in self.modules if m.type == 'DataProvider']
        self.module_filters = [m for m in self.modules if m.type == 'Filter']
        self.analyse_plot_config()
        # self.update_color_palette()
        self.getter = EventComparisonReader(plot_configs=self.plot_configs, process_configs=self.process_configs, **kwargs)
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


    def expand_process_configs(self):
        if self.process_configs is not None:
            for fh in self.file_handles:
                _ = find_process_config(fh.process, self.process_configs)

                
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

        for i, ref in enumerate(data):
            setattr(ref, 'draw_option', plot_config.draw)
            if plot_config.draw in ['Marker', 'marker', 'P', 'p']:
                setattr(ref, 'marker_color', PO.color_palette[i-(int(old_div(i,len(PO.color_palette)))*len(PO.color_palette))])
                setattr(ref, 'marker_style', PO.marker_style_palette_filled[i-(int(old_div(i,len(PO.marker_style_palette_filled)))*len(PO.marker_style_palette_filled))])
                setattr(ref, 'line_color', PO.color_palette[i-(int(old_div(i,len(PO.color_palette)))*len(PO.color_palette))])
            elif plot_config.draw in ['Line', 'line', 'L', 'l']:
                setattr(ref, 'line_color', PO.color_palette[i-(int(old_div(i,len(PO.color_palette)))*len(PO.color_palette))])
                setattr(ref, 'line_style', PO.line_style_palette_homogen[i-(int(old_div(i,len(PO.line_style_palette_homogen)))*len(PO.line_style_palette_homogen))])
            elif plot_config.draw in ['Hist', 'hist', 'H', 'h']:
                setattr(ref, 'fill_color', PO.color_palette[i-(int(old_div(i,len(PO.color_palette)))*len(PO.color_palette))])
                # setattr(ref, 'fill_style', PO.fill_style_palette_left[i-(int(i/len(PO.color_palette))*len(PO.color_palette))])
                setattr(ref, 'fill_style', 0)
                setattr(ref, 'line_color', PO.color_palette[i-(int(old_div(i,len(PO.color_palette)))*len(PO.color_palette))])
                setattr(ref, 'marker_color', PO.color_palette[i-(int(old_div(i,len(PO.color_palette)))*len(PO.color_palette))])
                
        # canvas = PT.plot_objects(map(lambda x : x.plot_object, reference_hists+compare_hists), plot_config, plotable_objects=reference_hists+compare_hists)
        canvas = PT.plot_objects(data, plot_config)
        canvas.SetName(plot_config.name.replace(' ', '_'))


        if self.process_configs:
            for ref in data:
                if hasattr(plot_config, 'ignore_process_labels') and plot_config.ignore_process_labels:
                    ref.label = '{:s}'.format(ref.label)
                else:
                    ref.label = '{:s} {:s}'.format(find_process_config(ref.process, self.process_configs).label, ref.label)
            
        ROOT.SetOwnership(canvas, False)

        if plot_config.enable_legend:
            labels = {}
            FM.add_legend_to_canvas(canvas, plot_config.ratio, labels=[x.label for x in data], plot_objects=[x.plot_object for x in data], **plot_config.legend_options)
        if plot_config.lumi:
            FM.decorate_canvas(canvas, plot_config)
            
        if plot_config.stat_box:
            FM.add_stat_box_to_canvas(canvas)

        self.output_handle.register_object(canvas)
