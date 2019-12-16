from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import glob
import json
import os
import pickle
import re
import sys
import traceback
from builtins import filter
from builtins import map
from builtins import object
from builtins import range
from builtins import str
from collections import OrderedDict
from copy import deepcopy
from math import sqrt

import dill
import numpy as np

import PyAnalysisTools.PlottingUtils.Formatting as fm
import PyAnalysisTools.PlottingUtils.PlottingTools as pt
import ROOT
from PyAnalysisTools.AnalysisTools.MLHelper import Root2NumpyConverter
from PyAnalysisTools.AnalysisTools.RegionBuilder import RegionBuilder
from PyAnalysisTools.AnalysisTools.SystematicsAnalyser import SystematicsAnalyser, TheoryUncertaintyProvider
from PyAnalysisTools.AnalysisTools.XSHandle import XSHandle
from PyAnalysisTools.PlottingUtils import Plotter
from PyAnalysisTools.PlottingUtils.HistTools import get_log_scale_x_bins, rebin
from PyAnalysisTools.PlottingUtils.PlotConfig import PlotConfig, get_default_color_scheme, transform_color
from PyAnalysisTools.base.ProcessConfig import find_process_config, parse_and_build_process_config
from PyAnalysisTools.base import _logger
from PyAnalysisTools.base.FileHandle import FileHandle
from PyAnalysisTools.base.OutputHandle import OutputFileHandle
from PyAnalysisTools.base.OutputHandle import SysOutputHandle as soh
from PyAnalysisTools.base.ShellUtils import make_dirs
from PyAnalysisTools.base.Utilities import merge_dictionaries
from PyAnalysisTools.base.YAMLHandle import YAMLDumper as yd, YAMLDumper
from PyAnalysisTools.base.YAMLHandle import YAMLLoader as yl

try:
    from tabulate.tabulate import tabulate
except ImportError:
    from tabulate import tabulate
sys.modules[tabulate.__module__].LATEX_ESCAPE_RULES = {}


def dump_input_config(cfg, output_dir):
    if output_dir is None:
        return
    yd.dump_yaml(dict([kv for kv in iter(list(cfg.items())) if kv[0] != 'scan_info' and kv[0] != 'xsec_handle']),
                 os.path.join(output_dir, 'config.yml'))


class LimitArgs(object):
    def __init__(self, output_dir, **kwargs):
        kwargs.setdefault("ctrl_syst", None)
        kwargs.setdefault("skip_ws_build", False)
        kwargs.setdefault("base_output_dir", output_dir)
        kwargs.setdefault("fixed_xsec", None)
        kwargs.setdefault("run_pyhf", False)
        kwargs.setdefault("ranking", False)
        kwargs.setdefault("queue", 'short7')
        kwargs.setdefault("pruning", None)
        self.skip_ws_build = kwargs['skip_ws_build']
        self.output_dir = output_dir
        self.base_output_dir = kwargs['base_output_dir']
        self.job_id = kwargs["jobid"]
        self.sig_reg_name = kwargs["sig_reg_name"]
        self.run_pyhf = kwargs['run_pyhf']
        self.queue = kwargs['queue']
        self.log_level = kwargs['log_level']
        self.local = False
        self.kwargs = kwargs

    def __str__(self):
        """
        Overloaded str operator. Get's called if object is printed
        :return: formatted string with name and attributes
        :rtype: str
        """
        obj_str = "Limit args for: {:s} \n".format(self.job_id)
        obj_str += 'signal: {:s}\n'.format(self.kwargs['sig_name'])
        obj_str += 'signal region: {:s}\n'.format(self.sig_reg_name)
        obj_str += 'mass cut: {:.1f}\n'.format(self.kwargs['mass_cut'])
        obj_str += 'registered processes: \n'
        for process in list(self.kwargs['process_configs'].keys()):
            obj_str += '\t{:s}\n'.format(process)
        return obj_str

    def __repr__(self):
        """
        Overloads representation operator. Get's called e.g. if list of objects are printed
        :return: formatted string with name and attributes
        :rtype: str
        """
        return self.__str__() + '\n'

    def to_json(self, file_name=None):
        def add_signal_region(name):
            channel = OrderedDict({'name': name})
            sig_yield = self.kwargs['sig_yield'][name]
            if self.kwargs['signal_scale']:
                sig_yield *= self.kwargs['signal_scale']
            if self.kwargs['fixed_signal']:
                sig_yield = self.kwargs['fixed_signal']
            channel['samples'] = [build_sample(self.kwargs['sig_name'],
                                               sig_yield,
                                               'mu')]
            for sn, sy in list(self.kwargs['bkg_yields'][name].items()):
                channel['samples'].append(build_sample(sn, sy))
            return channel

        def add_channel(name, info):
            data = None
            channel = OrderedDict({'name': name, 'samples': []})
            for sn, sy in list(info.items()):
                if sn.lower() == 'data':
                    data = [sy]
                    continue
                if sn == self.kwargs['sig_name']:
                    continue
                channel['samples'].append(build_sample(sn, sy))
            return channel, data

        def build_sample(name, yld, norm_factor=None):
            sample = {'name': name, 'data': [yld]}
            if norm_factor is not None:
                sample['modifiers'] = [{'name': norm_factor, 'type': 'normfactor', 'data': None}]
            else:
                sample['modifiers'] = []
            return sample

        def add_systematics(channel_id, process, systematics):
            sample = filter(lambda s: s['name'] == process, specs['channels'][channel_id]['samples'])[0]
            syst_strings = set([sn.split('__')[0] for sn in list(systematics.keys())])
            for syst in syst_strings:
                try:
                    down, up = sorted([s for s in iter(list(systematics.items())) if syst in s[0]], key=lambda i: i[0])
                    sample['modifiers'].append({"type": "histosys", "data": {"lo_data": [down[1] * sample['data'][0]],
                                                                             "hi_data": [up[1] * sample['data'][0]]},
                                                'name': syst})
                except ValueError:
                    variation = sorted([s for s in iter(list(systematics.items())) if syst in s[0]],
                                       key=lambda i: i[0])[0]
                    if variation[1] > 1.:
                        sample['modifiers'].append({"type": "histosys",
                                                    "data": {"lo_data": [sample['data'][0]],
                                                             "hi_data": [variation[1] * sample['data'][0]]},
                                                    'name': syst})
                    else:
                        sample['modifiers'].append({"type": "histosys",
                                                    "data": {"lo_data": [variation[1] * sample['data'][0]],
                                                             "hi_data": [sample['data'][0]]},
                                                    'name': syst})

        specs = OrderedDict()
        specs['channels'] = [add_signal_region(sig_reg) for sig_reg in self.kwargs['sig_yield']]
        specs['data'] = OrderedDict()
        for region, info in list(self.kwargs['control_regions'].items()):
            channel, data = add_channel(region, info)
            specs['channels'].append(channel)
            if data is None:
                data = [0.]
            specs['data'][region] = data
        for sig_reg in list(self.kwargs['sig_yield'].keys()):
            specs['data'][sig_reg] = [0.]

        specs['measurements'] = [{'config': {'poi': 'mu', 'parameters': []}, 'name': self.sig_reg_name}]

        for sig_reg in list(self.kwargs['sr_syst'].keys()):
            for process, systematics in list(self.kwargs['sr_syst'][sig_reg].items()):
                channel_id = [i for i, c in enumerate(specs['channels']) if c['name'] == sig_reg][0]
                add_systematics(channel_id, process, systematics)
        for region in list(self.kwargs['ctrl_syst'].keys()):
            channel_id = [i for i, c in enumerate(specs['channels']) if c['name'] == region][0]
            add_systematics(channel_id, process, systematics)

        if file_name is None:
            return specs
        with open(file_name, 'w') as f:
            json.dump(specs, f)


def build_region_info(control_region_defs):
    limit_region_info = {}
    for region in control_region_defs.regions:
        limit_region_info[region.name] = {"is_norm_region": region.norm_region,
                                          "bgk_to_normalise": region.norm_backgrounds,
                                          'is_val_region': region.val_region,
                                          'binning': region.binning,
                                          'label': region.label}
    return limit_region_info


def read_workspace_from_file(file_name, ws_name="w"):
    f = ROOT.TFile.Open(file_name)
    return f.Get(ws_name)


def get_fit_quality(file_name, ws_name="w", fr_name="RooExpandedFitResult_afterFit"):
    ws = read_workspace_from_file(file_name, ws_name)
    fit_result = ws.obj(fr_name)
    return fit_result.status(), fit_result.covQual()


class LimitInfo(object):
    def __init__(self, **kwargs):
        self.add_info(**kwargs)

    def __str__(self):
        """
        Overloaded str operator. Get's called if object is printed
        :return: formatted string with name and attributes
        :rtype: str
        """
        obj_str = "Limit info for: {:s} \n".format(self.sig_name)
        for attribute, value in list(self.__dict__.items()):
            obj_str += '{}={} '.format(attribute, value)
        return obj_str

    def __repr__(self):
        """
        Overloads representation operator. Get's called e.g. if list of objects are printed
        :return: formatted string with name and attributes
        :rtype: str
        """
        return self.__str__() + '\n'

    def add_info(self, **kwargs):
        for k, v in list(kwargs.items()):
            setattr(self, k, v)


def get_expected_limit(file_name, name='hypo_Sig'):
    """
    read expected limit an 1 sigma band from file
    :param file_name: name of file containing HypoTestInverterResult
    :type file_name: string
    :param name: name of parameter under which fit result is stored
    :type name: string
    :return: expected limit, +1, -1
    :rtype: list
    """
    try:
        f = ROOT.TFile.Open(file_name, 'READ')
        result = f.Get(name)
        return result.GetExpectedUpperLimit(), result.GetExpectedUpperLimit(1), result.GetExpectedUpperLimit(-1)
    except AttributeError:
        return -1., 0., 0.


class LimitAnalyserCL(object):
    def __init__(self, input_path):
        self.input_path = input_path
        self.limit_info = LimitInfo()
        self.converter = Root2NumpyConverter(['exp_upperlimit', 'exp_upperlimit_plus1', 'exp_upperlimit_plus2',
                                              'exp_upperlimit_minus1', 'exp_upperlimit_minus2', 'fit_status'])

    def analyse_limit(self, signal_scale=1., pmg_xsec=None, fixed_sig_sf=None, enable_debug_plot=False):
        """

        :param signal_scale: signal scale factor applied on top of 1 pb fixed xsec in limit setting
        :type signal_scale: float (default 1.0
        :param fixed_signal: fixed signal yield input if used during limit setting
        :type fixed_signal: float (default None)
        :param sig_yield: signal yield scaled according to sigma=1pb
        :type sig_yield: float (default None)
        :return: limit info object containing parsed UL
        :rtype: LimitInfo
        """

        def get_scale_factor(signal_scale):
            if signal_scale is None:
                signal_scale = 1.
            scale_factor = 1.
            if fixed_sig_sf is not None:
                _logger.debug("Apply sf: {:.3f}; \t scale factor before: {:.3f}".format(fixed_sig_sf, scale_factor))
                scale_factor = fixed_sig_sf * 1000. * pmg_xsec
            elif pmg_xsec is not None:
                scale_factor = 1000. * pmg_xsec
            return scale_factor

        try:
            fh = FileHandle(file_name=os.path.join(self.input_path, 'asymptotics/myLimit_CL95.root'),
                            switch_off_process_name_analysis=True)
            tree = fh.get_object_by_name('stats')
            data = self.converter.convert_to_array(tree=tree)
            fit_status = data['fit_status']
            scale_factor = get_scale_factor(signal_scale)
            self.limit_info.add_info(fit_status=fit_status, fit_cov_quality=-1)
            _logger.debug("limits for {:s} exp upper limit: {:f} scale factor: {:f} exp upper limit: * "
                          "SF: {:f}".format(self.input_path, data['exp_upperlimit'][0], scale_factor,
                                            data['exp_upperlimit'][0] * scale_factor))
            self.limit_info.add_info(exp_limit=data['exp_upperlimit'] * scale_factor,
                                     exp_limit_up=data['exp_upperlimit_plus1'] * scale_factor,
                                     exp_limit_low=data['exp_upperlimit_minus1'] * scale_factor)
            if enable_debug_plot:
                self.limit_info.add_info(scale_factor=scale_factor,
                                         signal_strength=data['exp_upperlimit'],
                                         signal_strength_up=data['exp_upperlimit_plus1'],
                                         signal_strength_low=data['exp_upperlimit_minus1'])

        except ValueError:
            try:
                with open(os.path.join(self.input_path, 'limit.json'), 'r') as f:
                    data = json.load(f)
                scale_factor = get_scale_factor(signal_scale)
                self.limit_info.add_info(fit_status=1, fit_cov_quality=1, exp_limit=data['CLs_exp'][2] * scale_factor,
                                         exp_limit_up=data['CLs_exp'][3] * scale_factor,
                                         exp_limit_low=data['CLs_exp'][1] * scale_factor)
            except (ZeroDivisionError, IOError):
                _logger.error("Could not find limit in path {:s}".format(self.input_path))
                self.limit_info.add_info(fit_status=-1, fit_cov_quality=-1, exp_limit=-1, exp_limit_up=-1,
                                         exp_limit_low=-1)
        return self.limit_info


class LimitPlotter(object):
    def __init__(self, output_handle):
        self.output_handle = output_handle

    def make_cross_section_limit_plot(self, limits, plot_config, theory_xsec=None, model_par='mass',
                                      value_par='exp_limit', disable_bands=False):
        """
        make cross section limit plot based on expected limits as function of mass hypothesis

        :param limits: list of LimitInfo objects
        :type limits: list
        :param plot_config: dictionary containing plot configuration, such as lumi, titles, etc
        :type plot_config: Ordered dict/dict
        :param theory_xsec: theory predictions
        :type theory_xsec: TGraph (default = None)
        :param sig_reg_name: name of signal region
        :type sig_reg_name: string (default = None)
        :return: None
        :rtype: None
        """
        plot_config.setdefault('logy', True)
        plot_config.setdefault('ymin', 1e-6)
        plot_config.setdefault('ymax', 1.)

        limits.sort(key=lambda li: getattr(li, model_par))
        pc = PlotConfig(name=plot_config['name'], ytitle=plot_config['ytitle'], xtitle=plot_config['xtitle'],
                        draw='pLX', logy=plot_config['logy'], lumi=plot_config['lumi'],
                        watermark=plot_config['watermark'],
                        ymin=float(plot_config['ymin']), ymax=float(plot_config['ymax']), )
        pc_1sigma = deepcopy(pc)
        pc_2sigma = deepcopy(pc)
        pc_1sigma.color = ROOT.kGreen
        pc_2sigma.color = ROOT.kYellow
        pc_1sigma.draw = "3"
        pc_2sigma.draw = "3"
        pc_1sigma.style_setter = "Fill"
        pc_2sigma.style_setter = "Fill"

        graph = ROOT.TGraph(len(limits))
        if not disable_bands:
            graph_1sigma = ROOT.TGraphAsymmErrors(len(limits))
            graph_2sigma = ROOT.TGraphAsymmErrors(len(limits))
        for i, limit in sorted(enumerate(limits)):
            graph.SetPoint(i, getattr(limit, model_par), getattr(limit, value_par))
            if not disable_bands:
                graph_1sigma.SetPoint(i, getattr(limit, model_par), getattr(limit, value_par))
                graph_2sigma.SetPoint(i, getattr(limit, model_par), getattr(limit, value_par))
                graph_1sigma.SetPointEYhigh(i, getattr(limit, value_par + '_up') - getattr(limit, value_par))
                graph_1sigma.SetPointEYlow(i, getattr(limit, value_par) - getattr(limit, value_par + '_low'))
                graph_2sigma.SetPointEYhigh(i, 2. * (getattr(limit, value_par + '_up') - getattr(limit, value_par)))
                graph_2sigma.SetPointEYlow(i, 2. * (getattr(limit, value_par) - getattr(limit, value_par + '_low')))
        if theory_xsec is not None:
            graph_theory = []
            processed_model_pars = sorted([getattr(li, model_par) for li in limits])
            processes = set([n.split('_')[0] for n in theory_xsec.keys()])

            for process in processes:
                graph_theory.append(ROOT.TGraphAsymmErrors(len(processed_model_pars)))
                for i, par in enumerate(processed_model_pars):
                    xsec, unc_up, unc_down = theory_xsec['{:s}_{:.0f}'.format(process, par)]
                    graph_theory[-1].SetPoint(i, par, xsec)
                    graph_theory[-1].SetPointEYhigh(i, unc_up - xsec)
                    graph_theory[-1].SetPointEYlow(i, xsec - unc_down)
                graph_theory[-1].SetName('Theory_prediction_{:s}'.format(process))

        if not disable_bands:
            graph_2sigma.SetName(pc.name)
            canvas = pt.plot_obj(graph_2sigma, pc_2sigma)
            pt.add_graph_to_canvas(canvas, graph_1sigma, pc_1sigma)
            pt.add_graph_to_canvas(canvas, graph, pc)
            labels = ['expected limit', '#pm 1#sigma', '#pm 2#sigma']
            legend_format = ['PL', 'F', 'F']
            plot_objects = [graph, graph_1sigma, graph_2sigma]
        else:
            graph.SetName(pc.name)
            canvas = pt.plot_obj(graph, pc)
            labels = ['expected']
            plot_objects = [graph]
            legend_format = ['PL']

        if theory_xsec is not None:
            pc_theory = deepcopy(pc)
            pc_theory.draw_option = 'l3'
            pc_theory.style_setter = ['Line', 'Fill']
            pc_theory.style = {'Fill': 3002}

            colors = get_default_color_scheme()
            for i, g in enumerate(graph_theory):
                pc_theory.color = colors
                pt.add_graph_to_canvas(canvas, g, pc_theory, index=i)
                labels.append("Theory {:s}".format(g.GetName().split('_')[-1]))
                legend_format.append("L")
                plot_objects.append(g)
        fm.decorate_canvas(canvas, pc)
        fm.add_legend_to_canvas(canvas, plot_objects=plot_objects, labels=labels, format=legend_format)
        self.output_handle.register_object(canvas)

    def make_limit_plot_plane(self, limits, plot_config, theory_xsec, sr_name):
        def find_excluded_lambda(mass, xsec, excl_limit):
            xsecs = [xs for xs in xsec if xs[0] == mass]
            try:
                return sqrt(excl_limit / xsecs[0][-1])
            except IndexError:
                return None

        if theory_xsec is None:
            return
        graphs_contour = []
        for process, xsecs in list(theory_xsec.items()):
            excl_lambdas = [(limit.mass, find_excluded_lambda(limit.mass, xsecs, limit.exp_limit)) for limit in limits]
            excl_lambdas = [v for v in excl_lambdas if v[1] is not None]
            excl_lambdas = sorted(excl_lambdas, key=lambda i: i[0])
            graphs_contour.append(ROOT.TGraph(len(excl_lambdas)))
            for i, limit in enumerate(excl_lambdas):
                graphs_contour[-1].SetPoint(i, limit[0], limit[1])
            graphs_contour[-1].SetName('Limit_contour_{:s}'.format(process))

        pc = PlotConfig(name='limit_contour_{:s}'.format(sr_name), watermark=plot_config['watermark'], ymax=10.,
                        xtitle=plot_config['xtitle'], ytitle='#lambda', logy=False, ymin=0., draw='Line',
                        lumi=plot_config['lumi'], labels=[g.GetName().split('_')[-1] for g in graphs_contour],
                        color=get_default_color_scheme())
        canvas = pt.plot_graphs(graphs_contour, pc)
        fm.decorate_canvas(canvas, pc)
        fm.add_legend_to_canvas(canvas, labels=pc.labels)
        self.output_handle.register_object(canvas)


class XsecLimitAnalyser(object):
    """
    Class to analyse cross section limit
    """

    def __init__(self, **kwargs):
        """
        Constructor
        :param input_path: input path containing calculated limits for each scan point
        :type input_path: string
        :param output_dir: directory where to store results
        :type output_dir: string
        :param scan_info: details on performed scan (mass, signal, etc.)
        :type scan_info: list
        """
        kwargs.setdefault('scan_info', None)
        kwargs.setdefault('xsec_map', None)
        kwargs.setdefault('enable_debug_plot', False)
        self.input_path = kwargs['input_path']
        self.output_handle = OutputFileHandle(output_dir=kwargs['output_dir'])
        self.plotter = LimitPlotter(self.output_handle)
        self.xsec_handle = XSHandle("config/common/dataset_info_pmg.yml")
        self.plot_config = yl.read_yaml(kwargs["plot_config"])
        self.theory_xsec = {}
        self.prefit_yields = {}
        self.scanned_mass_cuts = None
        self.scanned_sig_masses = None
        # self.lumi = self.plot_config['lumi']
        self.enable_debug_plot = kwargs['enable_debug_plot']
        self.analysis_name = self.plot_config['analysis_name']
        self.xsec_map = self.read_theory_cross_sections(kwargs['xsec_map'])
        if kwargs['scan_info'] is None:
            tmp = yl.read_yaml(os.path.join(self.input_path, "scan_info.yml"), None)
            self.scan_info = tmp['configs']
            self.scale_factors = tmp['scale_factors']
        dump_input_config(self.__dict__, self.output_handle.output_dir)

    def read_theory_cross_sections(self, file_name):
        if file_name is None:
            return None
        xsec = yl.read_yaml(file_name)
        return xsec

    def parse_limits(self):
        parsed_data = []
        for scan in self.scan_info:
            if 'mc' in scan.kwargs['sig_name']:
                continue
            self.sig_reg_name = scan.sig_reg_name
            analyser = LimitAnalyserCL(os.path.join(self.input_path, str(scan.kwargs['jobid']), self.analysis_name,
                                                    'Limits'))
            try:
                scale_factor = None
                if self.scale_factors is not None:
                    scale_factor = self.scale_factors[scan.kwargs['sig_name']]
                limit_info = analyser.analyse_limit(scan.kwargs['signal_scale'],
                                                    self.xsec_handle.get_xs_scale_factor(scan.kwargs['sig_name']),
                                                    fixed_sig_sf=scale_factor, enable_debug_plot=self.enable_debug_plot)
            except ReferenceError:
                _logger.error("Could not find info for scan {:s}".format(scan))
                continue
            limit_info.sig_name = scan.kwargs['sig_name']
            mass = float(re.findall(r'\d{3,4}', scan.kwargs['sig_name'])[0])
            self.theory_xsec[mass] = None
            limit_info.add_info(mass_cut=scan.kwargs["mass_cut"], mass=mass)
            parsed_data.append(limit_info)
        limits = LimitScanAnalyser.find_best_limit(parsed_data)
        theory_xsec = None
        if 'SR_mu_bveto' in self.sig_reg_name:
            chains = ['dLQmu', 'sLQmu']
        elif 'SR_mu_btag' in self.sig_reg_name:
            chains = ['bLQmu']
        elif 'SR_el_bveto' in self.sig_reg_name:
            chains = ['dLQe', 'sLQe']
        elif 'SR_el_btag' in self.sig_reg_name:
            chains = ['bLQe']
        else:
            chains = []
        if self.xsec_map is not None:
            theory_xsec = dict([kv for kv in iter(list(self.xsec_map.items())) if kv[0].split('_')[0] in chains])

        pc = deepcopy(self.plot_config)
        pc['name'] = 'xsec_limit{:s}'.format(scan.sig_reg_name)
        self.plotter.make_cross_section_limit_plot(limits, pc, theory_xsec)
        if self.enable_debug_plot:
            pc = deepcopy(self.plot_config)
            pc['name'] = 'signal_stength_limit_{:s}'.format(scan.sig_reg_name)
            pc['ytitle'] = 'Upper limit on signal strength'
            self.plotter.make_cross_section_limit_plot(limits, pc, None, value_par='signal_strength')
            pc = deepcopy(self.plot_config)
            pc['name'] = 'scale_factor_{:s}'.format(scan.sig_reg_name)
            pc['ytitle'] = 'Signal scale factor'
            self.plotter.make_cross_section_limit_plot(limits, pc, None, value_par='scale_factor', disable_bands=True)
            pc = deepcopy(self.plot_config)
            pc['name'] = 'fit_status_{:s}'.format(scan.sig_reg_name)
            pc['ytitle'] = 'Fit status'
            pc['logy'] = False
            pc['ymin'] = -0.5
            pc['ymax'] = 2.5
            self.plotter.make_cross_section_limit_plot(limits, pc, None, value_par='fit_status', disable_bands=True)

    def save(self):
        self.output_handle.write_and_close()


class LimitScanAnalyser(object):
    """
    Class to analyse limit scan over mass range and mass cuts
    """

    def __init__(self, **kwargs):
        """
        Constructor
        :param input_path: input path containing calculated limits for each scan point
        :type input_path: string
        :param output_dir: directory where to store results
        :type output_dir: string
        :param scan_info: details on performed scan (mass, signal, etc.)
        :type scan_info: list
        """
        kwargs.setdefault('scan_info', None)
        kwargs.setdefault('xsec_map', None)
        self.input_path = kwargs['input_path']
        self.output_handle = OutputFileHandle(output_dir=kwargs['output_dir'], sub_dir_name='plots')
        self.plotter = LimitPlotter(self.output_handle)
        self.xsec_handle = XSHandle("config/common/dataset_info_pmg.yml")
        self.plot_config = yl.read_yaml(kwargs["plot_config"])
        self.theory_xsec = {}
        self.prefit_yields = {}
        self.scanned_mass_cuts = None
        self.scanned_sig_masses = None
        self.lumi = self.plot_config['lumi']
        self.analysis_name = self.plot_config['analysis_name']
        self.xsec_map = self.read_theory_cross_sections(kwargs['xsec_map'])
        if kwargs['scan_info'] is None:
            tmp = yl.read_yaml(os.path.join(self.input_path, "scan_info.yml"), None)
            self.scan_info = tmp['configs']
            self.scale_factors = tmp['scale_factors']
        dump_input_config(self.__dict__, self.output_handle.output_dir)

    @staticmethod
    def read_theory_cross_sections(file_name):
        if file_name is None:
            return None
        with open(file_name, 'r') as f:
            xsec = pickle.load(f)
            return xsec

    def parse_limits(self):
        parsed_data = []
        for scan in self.scan_info:
            if 'mc' in scan.kwargs['sig_name']:
                continue
            self.sig_reg_name = scan.sig_reg_name
            analyser = LimitAnalyserCL(os.path.join(self.input_path, str(scan.kwargs['jobid']), self.analysis_name,
                                                    'Limits'))
            try:
                scale_factor = None
                if self.scale_factors is not None:
                    scale_factor = self.scale_factors[scan.kwargs['sig_name']]
                limit_info = analyser.analyse_limit(scan.kwargs['signal_scale'],
                                                    self.xsec_handle.get_xs_scale_factor(scan.kwargs['sig_name']),
                                                    fixed_sig_sf=scale_factor)
            except ReferenceError:
                _logger.error("Could not find info for scan {:s}".format(scan))
                continue
            limit_info.sig_name = scan.kwargs['sig_name']
            mass = float(re.findall(r'\d{3,4}', scan.kwargs['sig_name'])[0])
            self.theory_xsec[mass] = None
            limit_info.add_info(mass_cut=scan.kwargs["mass_cut"],
                                mass=mass)
            parsed_data.append(limit_info)
        best_limits = self.find_best_limit(parsed_data)
        self.make_scan_plot(parsed_data, self.plot_config, best_limits)
        self.tabulate_limits(best_limits)
        theory_xsec = None

        pc = deepcopy(self.plot_config)
        pc['name'] = 'xsec_limit{:s}'.format(scan.sig_reg_name)
        self.plotter.make_cross_section_limit_plot(best_limits, pc, theory_xsec)
        if theory_xsec is not None:
            self.plotter.make_limit_plot_plane(best_limits, self.plot_config, theory_xsec,
                                               scan.sig_reg_name)
        self.output_handle.write_and_close()

    def tabulate_limits(self, limits):
        limits.sort(key=lambda l: l.mass)
        data = []
        ordering = self.plot_config['ordering']
        for limit in limits:
            data_mass_point = [limit.mass, limit.mass_cut, limit.exp_limit]
            data.append(data_mass_point)
        headers = ['$m_{LQ}^{gen} [\\GeV{}]$', '\\mLQmax{} cut [\\GeV{}]', 'UL [pb]', 'Signal'] + ordering
        print(tabulate(data, headers=headers, tablefmt='latex_raw'))
        with open(os.path.join(self.output_handle.output_dir,
                               'limit_scan_table_best_{:s}.tex'.format(self.sig_reg_name)), 'w') as f:
            print(tabulate(data, headers=headers, tablefmt='latex_raw'), file=f)
        print('wrote to file ', os.path.join(self.output_handle.output_dir,
                                             'limit_scan_table_best_{:s}.tex'.format(self.sig_reg_name)))
        self.dump_best_limits_to_yaml(limits)

    def dump_best_limits_to_yaml(self, best_limits):
        data = {}
        for limit in best_limits:
            data[limit.sig_name] = {'threshold': limit.mass_cut}
        yd.dump_yaml(data, os.path.join(self.output_handle.output_dir, 'limit_thresholds.yml'),
                     default_flow_style=False)

    @staticmethod
    def find_best_limit(limits):
        limits = [li for li in limits if li.exp_limit > 0.]
        masses = set([li.mass for li in limits])
        best_limits = []
        for mass in masses:
            # if mass == 500:
            #     limits = [li for li in limits if li.mass_cut < 1000.]
            mass_limits = [li for li in limits if li.mass == mass]
            if len(mass_limits) == 0:
                _logger.warn("Could not find limit for mass {:.0f}".format(mass))
                continue
            best_limits.append(min(mass_limits, key=lambda li: li.exp_limit))
        return best_limits

    @staticmethod
    def read_yields(path, fname, analysis_name):
        data = {}
        rf = ROOT.TFile.Open(os.path.join(path, "results", analysis_name, fname))
        for item in rf.GetListOfKeys():
            name = item.GetName()
            obj = rf.Get(name)
            if not isinstance(obj, ROOT.TH1F):
                continue
            obj.SetDirectory(0)
            data[name] = obj
        return data

    def parse_prefit_yields(self, scan, mass):
        self.prefit_yields[(mass, scan.kwargs["mass_cut"])] = self.read_yields(scan.output_dir,
                                                                               "SR_yield_beforeFit.root",
                                                                               self.analysis_name)

    def plot_prefit_yields(self):
        def book_single_mass_hist(mass, process):
            hist_single_mass = ROOT.TH1F("yield_single_mass_{:s}_{:s}".format(str(mass), process), "",
                                         *self.scan_mass_binning)
            return hist_single_mass

        def book_single_cut_hist(mass_cut, process):
            hist_single_mass_cut = ROOT.TH1F("yield_single_cut_{:s}_{:s}".format(str(mass_cut), process), "",
                                             *self.sig_mass_binning)
            return hist_single_mass_cut

        ordering = self.plot_config['ordering']
        yields_single_cut_hists = {}
        yields_single_mass_hists = {}
        for mass_cut in self.scanned_mass_cuts:
            for process in ordering + ["signal"]:
                yields_single_cut_hists[(mass_cut, process)] = book_single_cut_hist(mass_cut, process)
        for mass in self.scanned_sig_masses:
            for process in ordering + ["signal"]:
                yields_single_mass_hists[(mass, process)] = book_single_mass_hist(mass, process)

        for mass, cut in sorted(self.prefit_yields.keys()):
            yields = self.prefit_yields[(mass, cut)]
            for process in ordering:
                hist_vs_cut = yields_single_cut_hists[(cut, process)]
                try:
                    hist_vs_cut.Fill(mass, yields[process].GetBinContent(1))
                except KeyError:
                    hist_vs_cut.Fill(mass, 0.)
                    print("Did not fine yields for {:s} and LQ mass {:.0f} and cut {:.0f}".format(process, mass, cut))
                hist_vs_mass = yields_single_mass_hists[(mass, process)]
                try:
                    hist_vs_mass.Fill(cut, yields[process].GetBinContent(1))
                except KeyError:
                    hist_vs_mass.Fill(cut, 0.)
            signal_process = filter(lambda p: "LQ" in p, list(yields.keys()))[0]
            hist_vs_cut = yields_single_cut_hists[(cut, "signal")]
            hist_vs_cut.Fill(mass, yields[signal_process].GetBinContent(1))
            hist_vs_mass = yields_single_mass_hists[(mass, "signal")]
            hist_vs_mass.Fill(cut, yields[signal_process].GetBinContent(1))
        pc_vs_cut = PlotConfig(name="", xtitle=self.plot_config['xtitle'], ytitle="Event yields",
                               watermark=self.plot_config['watermark'], lumi=self.plot_config['lumi'],
                               color=get_default_color_scheme(), style=1001)
        pc_vs_mass = PlotConfig(name="", xtitle=self.plot_config['ytitle'], ytitle="Event yields",
                                watermark=self.plot_config['watermark'], lumi=self.plot_config['lumi'],
                                color=get_default_color_scheme(), style=1001)
        for mass, cut in list(self.prefit_yields.keys()):
            pc_vs_cut.name = "yield_vs_cut_{:s}".format(str(mass))
            pc_vs_mass.name = "yield_vs_mass_{:s}".format(str(mass))
            pc_vs_cut_log = deepcopy(pc_vs_cut)
            pc_vs_cut_log.name += "_log"
            pc_vs_cut_log.logy = True
            pc_vs_cut_log.ymin = 0.1
            pc_vs_mass_log = deepcopy(pc_vs_mass)
            pc_vs_mass_log.name += "_log"
            pc_vs_mass_log.logy = True
            pc_vs_mass_log.ymin = 0.1
            canvas_vs_cut = pt.plot_stack([yields_single_cut_hists[(cut, p)] for p in ordering + ["signal"]],
                                          pc_vs_cut)
            canvas_vs_cut_log = pt.plot_stack([yields_single_cut_hists[(cut, p)] for p in ordering + ["signal"]],
                                              pc_vs_cut_log)
            pc_vs_cut.name = "yield_vs_cut_{:s}".format(str(mass))
            canvas_vs_mass = pt.plot_stack([yields_single_mass_hists[(mass, p)] for p in ordering + ["signal"]],
                                           pc_vs_mass)
            canvas_vs_mass_log = pt.plot_stack([yields_single_mass_hists[(mass, p)] for p in ordering + ["signal"]],
                                               pc_vs_mass_log)
            fm.decorate_canvas(canvas_vs_cut, pc_vs_cut)
            fm.decorate_canvas(canvas_vs_mass, pc_vs_mass)
            fm.decorate_canvas(canvas_vs_cut_log, pc_vs_cut_log)
            fm.decorate_canvas(canvas_vs_mass_log, pc_vs_mass_log)
            fm.add_legend_to_canvas(canvas_vs_mass, labels=ordering + ["signal"])
            fm.add_legend_to_canvas(canvas_vs_cut, labels=ordering + ["signal"])
            fm.add_legend_to_canvas(canvas_vs_mass_log, labels=ordering + ["signal"])
            fm.add_legend_to_canvas(canvas_vs_cut_log, labels=ordering + ["signal"])
            self.output_handle.register_object(canvas_vs_cut)
            self.output_handle.register_object(canvas_vs_mass)
            self.output_handle.register_object(canvas_vs_cut_log)
            self.output_handle.register_object(canvas_vs_mass_log)

    def make_scan_plot(self, parsed_data, plot_config, best_limits=None):
        self.scanned_mass_cuts = sorted(list(set([li.mass_cut for li in parsed_data])))
        self.scanned_sig_masses = sorted(list(set([li.mass for li in parsed_data])))
        if len(self.scanned_sig_masses) > 1:
            self.min_mass_diff = min(
                [self.scanned_sig_masses[i + 1] - self.scanned_sig_masses[i] for i in
                 range(len(self.scanned_sig_masses) - 1)]) / 2.
        else:
            self.min_mass_diff = 50
        if len(self.scanned_mass_cuts) > 1:
            self.mass_cut_offset = (self.scanned_mass_cuts[1] - self.scanned_mass_cuts[0]) / 2.
        else:
            self.mass_cut_offset = 50.

        self.sig_mass_binning = [
            int((self.scanned_sig_masses[-1] - self.scanned_sig_masses[0]) / (self.min_mass_diff * 2)) + 1,
            self.scanned_sig_masses[0] - self.min_mass_diff,
            self.scanned_sig_masses[-1] + self.min_mass_diff]
        self.scan_mass_binning = [
            int((self.scanned_mass_cuts[-1] - self.scanned_mass_cuts[0]) / (self.mass_cut_offset * 2)) + 1,
            self.scanned_mass_cuts[0] - self.mass_cut_offset,
            self.scanned_mass_cuts[-1] + self.mass_cut_offset]
        y_binning = self.scan_mass_binning
        y_binning[2] += (y_binning[2] - y_binning[1]) / y_binning[0] * 10
        y_binning[0] += 10
        hist = ROOT.TH2F("upper_limit", "", *(self.sig_mass_binning + y_binning))
        hist_norm = ROOT.TH2F("upper_limit_norm", "", *(self.sig_mass_binning + y_binning))
        hist_best = hist.Clone("best_limit")
        hist_fit_status = hist.Clone("fit_status")
        hist_fit_quality = hist.Clone("fit_quality")
        limit_scan_table = OrderedDict()
        mass_points = sorted(set([i.mass for i in parsed_data]))
        for limit_info in parsed_data:
            if limit_info.mass_cut not in limit_scan_table:
                limit_scan_table[limit_info.mass_cut] = [-1.] * len(mass_points)
            limit_scan_table[limit_info.mass_cut][mass_points.index(limit_info.mass)] = limit_info.exp_limit

            if limit_info.exp_limit > 0:
                hist.Fill(limit_info.mass, limit_info.mass_cut, limit_info.exp_limit * 1000.)
                best_limit = min([li.exp_limit for li in parsed_data if li.mass == limit_info.mass and li.exp_limit > 0.])
                hist_norm.Fill(limit_info.mass, limit_info.mass_cut, limit_info.exp_limit/best_limit)
            # else:
            #     hist.Fill(limit_info.mass, limit_info.mass_cut, -1.)
            #     hist_norm.Fill(limit_info.mass, limit_info.mass_cut, -1.)
            hist_fit_status.Fill(limit_info.mass, limit_info.mass_cut, limit_info.fit_status + 1)
            hist_fit_quality.Fill(limit_info.mass, limit_info.mass_cut, limit_info.fit_cov_quality)

        ROOT.gStyle.SetPalette(1)
        pc = PlotConfig(name="limit_scan_{:s}".format(self.sig_reg_name), draw_option="COLZ",
                        xtitle=plot_config['xtitle'], ytitle=plot_config['ytitle'], ztitle="95% CL U.L. #sigma [fb]",
                        watermark='Internal', lumi=139.0)
        ROOT.gStyle.SetPaintTextFormat(".2g")
        pc_norm = PlotConfig(name="limit_scan_norm_{:s}".format(self.sig_reg_name), draw_option="COLZTEXT", lumi=139.0,
                             xtitle=plot_config['xtitle'], ytitle=plot_config['ytitle'], watermark='Internal',
                             ztitle="#splitline{95% CL U.L. #sigma [fb]}{normalised to lowest U.L. per mass bin}")
        pc_status = PlotConfig(name="limit_status_{:s}".format(self.sig_reg_name), draw_option="COLZTEXT",
                               xtitle=plot_config['xtitle'], ytitle=plot_config['ytitle'],
                               ztitle="fit status + 1", zmin=-1., watermark='Internal', lumi=139.0)
        pc_cov_quality = PlotConfig(name="limit_cov_quality_{:s}".format(self.sig_reg_name), draw_option="COLZTEXT",
                                    xtitle=plot_config['xtitle'], ytitle=plot_config['ytitle'],
                                    ztitle="fit cov quality", watermark='Internal', lumi=139.0)
        canvas = pt.plot_obj(hist, pc)
        canvas_scan_norm = pt.plot_obj(hist_norm, pc_norm)
        if best_limits is not None:
            pc_best = PlotConfig(draw_option="BOX")
            for limit in best_limits:
                hist_best.Fill(limit.mass, limit.mass_cut, hist.GetMaximum())
            hist_best.SetLineColor(ROOT.kRed)
            pt.add_histogram_to_canvas(canvas, hist_best, pc_best)
            pt.add_histogram_to_canvas(canvas_scan_norm, hist_best, pc_best)
        fm.decorate_canvas(canvas, pc)
        self.output_handle.register_object(canvas)
        self.output_handle.register_object(canvas_scan_norm)
        canvas_status = pt.plot_obj(hist_fit_status, pc_status)
        self.output_handle.register_object(canvas_status)
        canvas_quality = pt.plot_obj(hist_fit_quality, pc_cov_quality)
        self.output_handle.register_object(canvas_quality)
        limit_scan_table = [['{:.0f}'.format(i[0])] + i[1] for i in list(limit_scan_table.items())]
        with open(os.path.join(self.output_handle.output_dir,
                               'limit_scan_table_{:s}.tex'.format(self.sig_reg_name)), 'w') as f:
            print(tabulate(limit_scan_table, headers=['LQ mass'] + mass_points, tablefmt='latex_raw'), file=f)


class CommonLimitOptimiser(object):
    def __init__(self, **kwargs):
        kwargs.setdefault('syst_config', None)
        kwargs.setdefault('skip_fh_reading', False)
        kwargs.setdefault('thresholds', None)
        kwargs.setdefault('thrs_cfg', None)
        kwargs.setdefault('mass_range', None)
        kwargs.setdefault('local', False)
        kwargs.setdefault('cluster_cfg_file', None)
        kwargs.setdefault('scan_bounds_per_mass', None)
        kwargs.setdefault('mass_pattern', r'\d{2,4}')
        kwargs.setdefault('resubmit', None)
        kwargs.setdefault('include_timeout', False)

        if kwargs['cluster_cfg_file'] is not None:
            self.log_level = kwargs['log_level']
            self.queue = kwargs['queue']
            self.local = False
            self.job_id = kwargs['job_id']
            self.cluster_cfg_file = kwargs['cluster_cfg_file']
            return

        self.signal_region_def = RegionBuilder(**yl.read_yaml(kwargs["sr_module_config"])["RegionBuilder"])
        self.control_region_defs = None
        self.process_configs = parse_and_build_process_config(kwargs['process_config_files'])

        if kwargs['thresholds'] is not None:
            self.thrs_cfg = yl.read_yaml(kwargs["thresholds"])
            self.mass_cuts = [(cfg[0], cfg[1]["threshold"]) for cfg in list(self.thrs_cfg.items())]

        if kwargs["cr_module_config"] is not None:
            self.control_region_defs = RegionBuilder(**yl.read_yaml(kwargs["cr_module_config"])["RegionBuilder"])
        self.output_dir = soh.resolve_output_dir(output_dir=kwargs["output_dir"], sub_dir_name="limit")
        if kwargs['resubmit'] is not None:
            self.output_dir = kwargs['resubmit']

        make_dirs(self.output_dir)
        self.input_hist_file = kwargs['input_hist_file']
        self.output_handle = OutputFileHandle(**kwargs)
        self.run_syst = False
        self.shape_syst_config, self.scale_syst_config = None, None
        self.systematics = []
        if kwargs['syst_config'] is not None:
            self.systematics = SystematicsAnalyser.parse_syst_config(kwargs['syst_config'])
            self.run_syst = True
        self.queue = kwargs['queue']
        if not kwargs['skip_fh_reading']:
            self.file_handles = Plotter.filter_unavailable_processes(self.file_handles, self.process_configs)
            self.file_handles = self.filter_mass_points()
            self.filter_empty_trees()
        for k, v in list(kwargs.items()):
            if hasattr(self, k):
                continue
            setattr(self, k, v)
        if kwargs['mass_range'] is not None:
            self.mass_scans = np.linspace(self.mass_range[0], self.mass_range[1], kwargs["nscans"])

    def filter_mass_points(self):
        """
        Keep all non-signal file handles and only signals which are defined in the limit config
        :return: filtered list of file handles
        :rtype: list
        """
        return [fh for fh in self.file_handles if find_process_config(fh.process, self.process_configs) is not None and
                (find_process_config(fh.process, self.process_configs).type == 'Signal'
                 and fh.process.process_name in self.limit_config) or
                find_process_config(fh.process, self.process_configs).type != 'Signal']

    @staticmethod
    def run_limit_hf(args, **kwargs):
        try:
            CommonLimitOptimiser.run_fit(args, analysis_name='LQAnalysis', **kwargs)
        except Exception:
            traceback.print_exc()
            _logger.error("Failed running limit fit")
        return 0

    def prepare(self):
        cr_config = build_region_info(self.control_region_defs)
        configs = []
        if self.resubmit is not None:
            jobs = yl.read_yaml(os.path.join(self.resubmit, 'scan_info.yml'))['configs']
            processed_jobs = [int(i) for i in os.listdir(self.resubmit) if re.match(r'\d+', i)]
            if self.include_timeout:
                processed_jobs = [i for i in processed_jobs if len(glob.glob(os.path.join(self.resubmit, str(i), '*',
                                                                                          'Limits'))) > 0]
            _logger.debug("Processed jobs: {:d} expected: {:d}".format(len(processed_jobs), len(jobs)))

        elif self.thrs_cfg is not None:
            for sig_name, config in list(self.thrs_cfg.items()):
                threshold = config["threshold"]
                configs.append(LimitArgs(sig_reg_name=self.signal_region_def.regions[0].name,
                                         sig_reg_cfg=self.signal_region_def.regions[0],
                                         output_dir=self.output_dir, sig_name=sig_name, limit_config=self.limit_config,
                                         process_configs=self.process_configs, systematics=self.systematics,
                                         ctrl_config=cr_config, jobid=str(len(configs)), fixed_signal=self.fixed_signal,
                                         queue=self.queue, mass_cut=threshold, signal_scale=self.signal_scale,
                                         ranking=self.ranking, log_level=self.log_level))
            thresholds = [(cfg.sig_reg_name, cfg.kwargs['mass_cut']) for cfg in configs]
            scale_factors = convert_hists(self.input_hist_file, self.process_configs,
                                          self.signal_region_def.regions + self.control_region_defs.regions,
                                          self.fixed_signal, self.output_dir, thresholds, self.systematics,
                                          cut_off=self.cut_off_scale)
        elif self.mass_scans is not None:
            signals = [pc.name for pc in self.process_configs.values() if pc.type.lower() == 'signal']
            for threshold in self.mass_scans:
                for sig_name in list(signals):
                    if self.scan_bounds_per_mass is not None:
                        mass = float(re.search(self.mass_pattern, sig_name).group())
                        if threshold < mass - self.scan_bounds_per_mass[0] or threshold > self.scan_bounds_per_mass[1] \
                                + mass:
                            continue
                    configs.append(LimitArgs(sig_reg_name=self.signal_region_def.regions[0].name,
                                             sig_reg_cfg=self.signal_region_def.regions[0],
                                             output_dir=self.output_dir, sig_name=sig_name,
                                             limit_config=self.limit_config, pruning=0.00011,
                                             process_configs=self.process_configs, systematics=self.systematics,
                                             ctrl_config=cr_config, jobid=str(len(configs)),
                                             fixed_signal=self.fixed_signal, disable_plots=True,
                                             queue=self.queue, mass_cut=threshold, signal_scale=self.signal_scale,
                                             ranking=False, log_level=self.log_level))
            thresholds = [(cfg.sig_reg_name, cfg.kwargs['mass_cut']) for cfg in configs]
            scale_factors = convert_hists(self.input_hist_file, self.process_configs,
                                          self.signal_region_def.regions + self.control_region_defs.regions,
                                          self.fixed_signal, self.output_dir, thresholds, self.systematics,
                                          cut_off=self.cut_off_scale)
        else:
            """
            Run a single signal with arbitrary fixed yield
            """
            configs.append(LimitArgs(sig_reg_name=self.signal_region_def.regions[0].name,
                                     output_dir=self.output_dir, sig_name='Signal', limit_config=self.limit_config,
                                     process_configs=self.process_configs, systematics=self.systematics,
                                     ctrl_config=cr_config, jobid=str(len(configs)), fixed_signal=self.fixed_signal,
                                     queue=self.queue, mass_cut=0., signal_scale=self.signal_scale))
            scale_factors = convert_hists(self.input_hist_file, self.process_configs,
                                          self.signal_region_def.regions + self.control_region_defs.regions,
                                          self.fixed_signal, self.output_dir, [0.], self.systematics,
                                          dummy_signal='Signal', cut_off=self.cut_off_scale,
                                          anking=False, log_level=self.log_level)
        cfg_file_name = os.path.join(self.output_dir, 'scan_info.yml')
        if self.resubmit:
            self.submit_args = cfg_file_name, os.path.basename(sys.argv[0]), len(jobs), self.output_dir, True
            return [i for i in range(len(jobs)) if i not in processed_jobs]
        YAMLDumper.dump_yaml({'configs': configs, 'scale_factors': scale_factors},
                             cfg_file_name)
        self.submit_args = cfg_file_name, os.path.basename(sys.argv[0]), len(configs), self.output_dir, True

        if self.test_mode:
            return [0]

    @staticmethod
    def run_fit(args, **kwargs):
        """
        Submit limit fit using trex fitter executable
        :param args: argument list for configuration
        :type args: arglist
        :param kwargs: named argument list for additional information
        :type kwargs: dict
        :return: nothing
        :rtype: None
        """
        kwargs.setdefault('scale_factors', None)
        def execute(options):
            trf_cmd = '&& '.join(['trex-fitter {:s} {:s}'.format(o, cfg_file) for o in options])
            os.system(trf_cmd)
        args.kwargs = merge_dictionaries(args.kwargs, kwargs)
        write_config(args)
        kwargs.setdefault('options', 'hbwdflp')

        cfg_file = os.path.join(args.output_dir, str(args.job_id), 'trex_fitter.config')
        execute(kwargs['options'])
        if args.kwargs['ranking']:
            args.output_dir = os.path.join(args.output_dir, 'ranking')
            cfg_file = os.path.join(args.output_dir, str(args.job_id), 'trex_fitter.config')
            execute('hbwfr')

    def finish(self):
        _logger.info('Wrote limits to {:s}'.format(self.output_dir))

    def run(self, cluster_cfg_file, job_id):
        """
        Entry point starting execution
        :return: nothing
        :rtype: None
        """
        _logger.debug('Start running limits')
        args = yl.read_yaml(cluster_cfg_file)['configs'][job_id]
        scale_factors = yl.read_yaml(cluster_cfg_file)['scale_factors']
        self.queue = args.queue
        if args.run_pyhf:
            return run_fit_pyhf(args)
        else:
            return CommonLimitOptimiser.run_limit_hf(args, scale_factors=scale_factors)


__file_path__ = os.path.realpath(__file__)


def run_fit_pyhf(args):
    """
    Submit limit fit using pyhf
    :param args: argument list for configuration
    :type args: arglist
    :param kwargs: named argument list for additional information
    :type kwargs: dict
    :return: nothing
    :rtype: None
    """
    job_dir = os.path.join(args.output_dir, 'limits', str(args.job_id))
    make_dirs(job_dir)
    ws = os.path.join(job_dir, 'workspace.json')
    args.to_json(ws)
    os.system('pyhf cls {:s} --output-file {:s}'.format(ws, ws.replace('workspace', 'limit')))


def write_config(args, ranking=False):
    if args.kwargs['ranking'] and not ranking:
        write_config(deepcopy(args), True)
    kwargs = args.kwargs
    kwargs.setdefault('stat_only', False)
    kwargs.setdefault('disable_plots', False)

    hist_path = os.path.join(args.output_dir, 'hists')
    if ranking:
        args.output_dir = os.path.join(args.output_dir, 'ranking')
        kwargs['disable_plots'] = True
    make_dirs(os.path.join(args.output_dir, str(args.job_id)))
    make_plots = "FALSE" if kwargs['disable_plots'] else "TRUE"
    config_name = os.path.join(args.output_dir, str(args.job_id), 'trex_fitter.config')
    make_dirs(hist_path)
    limit_config = yl.read_yaml(kwargs['limit_config'])
    limit_config['general'].setdefault('lumi_uncert', 0.017)
    limit_config['general'].setdefault('suffix', None)

    def convert_binning(parsed_info):
        if 'logx' in parsed_info:
            return ','.join(map(str, get_log_scale_x_bins(*list(map(float, re.findall(r'\d+', parsed_info))))))
        if 'eval' in parsed_info:
            return ','.join(map(str, eval(parsed_info.replace('eval', ''))))
        return parsed_info

    with open(config_name, 'w') as f:
        print('% --------------- %', file=f)
        print('% ---  JOB - -- %', file=f)
        print('% --------------- %', file=f)
        print('\n', file=f)

        print('Job: {:s}'.format(limit_config['general']['analysis_name']), file=f)
        print('\tCmeLabel: "13 TeV"', file=f)
        print('\tPOI: "mu_Sig"', file=f)
        print('\tReadFrom: HIST', file=f)
        print('\tHistoPath: {:s}'.format(os.path.join(hist_path, 'Nominal')), file=f)
        print('\tLabel: "{:s}"'.format(kwargs['sig_reg_cfg'].channel), file=f)
        print('\tLumiLabel: "139 fb^{-1}"', file=f)
        print('\tPlotOptions: NOXERR, NOENDERR', file=f)
        print('\tSystErrorBars: {:s}'.format(make_plots), file=f)
        print('\tSystControlPlots: {:s}'.format(make_plots), file=f)
        print('\tCorrelationThreshold: 0.1', file=f)
        print('\tAtlasLabel: Internal', file=f)
        print('\tDebugLevel: 1', file=f)
        print('\tHistoChecks: NOCRASH', file=f)
        print('\tImageFormat: "pdf"', file=f)
        print('\tRankingPlot: all', file=f)
        print('\tKeepPruning: {:s}'.format(make_plots), file=f)
        print('\tSystCategoryTables: {:s}'.format(make_plots), file=f)
        print('\tSystControlPlots: {:s}'.format(make_plots), file=f)
        print('\tDoSummaryPlot: {:s}'.format(make_plots), file=f)
        print('\tDoTables: {:s}'.format(make_plots), file=f)
        print('\tDoSignalRegionsPlot: {:s}'.format(make_plots), file=f)
        print('\tDoPieChartPlot: {:s}'.format(make_plots), file=f)
        print('\tMergeUnderOverFlow: FALSE', file=f)
        if limit_config['general']['suffix']:
            print('\tSuffix: {:s}'.format(limit_config['general']['suffix']), file=f)
        if kwargs['pruning'] is not None:
            print('\tSystPruningNorm: {:f}'.format(kwargs['pruning']), file=f)
            print('\tSystPruningShape: {:f}'.format(kwargs['pruning']), file=f)

        print('\tUseATLASRounding: TRUE', file=f)
        print('\tTableOptions: !STANDALONE', file=f)
        if ranking:
            print('\tRankingMaxNP: 25', file=f)
        if kwargs['stat_only']:
            print('\tStatOnly: TRUE', file=f)
        print('\tOutputDir: {:s}'.format(os.path.join(args.output_dir, str(args.job_id))), file=f)
        print('\n', file=f)
        print('% --------------- %', file=f)
        print('% ---  FIT - -- %', file=f)
        print('% --------------- %', file=f)
        print('\n', file=f)

        print('Fit: "fit"', file=f)
        if not ranking:
            print('\tFitType: BONLY', file=f)
            print('\tFitRegion: CRONLY', file=f)
        else:
            print('\tFitType: SPLUSB', file=f)
            print('\tFitRegion: CRSR', file=f)
            print('\tPOIAsimov: 5.', file=f)
        print('\tUseMinos: mu_Sig', file=f)
        print('\n', file=f)
        print('% --------------- %', file=f)
        print('% ---  LIMIT - -- %', file=f)
        print('% --------------- %', file=f)
        print('\n', file=f)
        print('Limit: "limit"', file=f)
        print('\tLimitType: ASYMPTOTIC', file=f)
        print('\tLimitBlind: FALSE', file=f)
        print('\t% POIAsimov: 1', file=f)
        print('\n', file=f)

        print('% --------------- %', file=f)
        print('% --- REGIONS - -- %', file=f)
        print('% --------------- %', file=f)
        print('\n', file=f)

        print('Region: "{:s}"'.format(kwargs['sig_reg_name']), file=f)
        print('\tType: SIGNAL', file=f)
        print('\tVariableTitle: "{:s}"'.format(limit_config['plotting']['xtitle']), file=f)
        if kwargs['sig_reg_cfg'].channel is not None:
            print('\tLabel: "{:s}"'.format(kwargs['sig_reg_cfg'].label), file=f)
        else:
            print('\tLabel: "{:s}"'.format(kwargs['sig_reg_name']), file=f)
        print('\tShortLabel: "{:s}"'.format(kwargs['sig_reg_name']), file=f)
        print('\tHistoName: h_{:s}'.format(kwargs['sig_reg_name']), file=f)
        print('\tDataType: ASIMOV', file=f)
        if 'signal_region' in limit_config and 'binning' in limit_config['signal_region']:
            print('\tBinning: {:s}'.format(convert_binning(limit_config['signal_region']['binning'])), file=f)
        else:
            print('\tBinning: {:f}, 8000.'.format(kwargs['mass_cut']), file=f)
        print('\n', file=f)

        if 'spectators' in limit_config['general']:
            reg_match = [r for r in list(limit_config['general']['spectators']['regions'].keys()) if
                         re.match(r, kwargs['sig_reg_name'])]
            if len(reg_match):
                spec_cfg = limit_config['general']['spectators']['regions'][reg_match[0]]
                print('Region: "{:s}_spectator"'.format(kwargs['sig_reg_name']), file=f)
                print('\tType: VALIDATION', file=f)
                print('\tVariableTitle: "{:s}"'.format(limit_config['plotting']['xtitle']), file=f)
                print('\tLabel: "{:s}"'.format(kwargs['sig_reg_name']), file=f)
                print('\tShortLabel: "{:s}"'.format(kwargs['sig_reg_name']), file=f)
                print('\tHistoName: h_{:s}'.format(kwargs['sig_reg_name']), file=f)
                print('\tDataType: ASIMOV', file=f)
                print('\tBinning: {:s}'.format(convert_binning(spec_cfg['binning'])), file=f)
                print('\n', file=f)
        norm_factors = []
        for reg, ctrl_reg_cfg in list(kwargs["ctrl_config"].items()):
            print('Region: "{:s}"'.format(reg), file=f)
            if ctrl_reg_cfg['is_norm_region']:
                print('\tType: CONTROL', file=f)
            elif ctrl_reg_cfg['is_val_region']:
                print('\tType: VALIDATION', file=f)
            print('\tVariableTitle: "{:s}"'.format(limit_config['plotting']['xtitle']), file=f)
            print('\tLabel: "{:s}"'.format(ctrl_reg_cfg['label']), file=f)
            print('\tShortLabel: "{:s}"'.format(reg), file=f)
            print('\tHistoName: "h_{:s}"'.format(reg), file=f)
            if ctrl_reg_cfg['binning'] is not None:
                print('\tBinning: {:s}'.format(convert_binning(ctrl_reg_cfg['binning'])), file=f)
            print('', file=f)
            for bkg, norm_param in list(ctrl_reg_cfg['bgk_to_normalise'].items()):
                if 'norm_factor' not in norm_param:
                    continue
                norm_factors.append((norm_param['norm_factor'], bkg))
            if 'spectators' in limit_config['general']:
                reg_match = [r for r in list(limit_config['general']['spectators']['regions'].keys()) if
                             re.match(r, reg)]
                if len(reg_match):
                    spec_cfg = limit_config['general']['spectators']['regions'][reg_match[0]]
                    print('Region: "{:s}_spectator"'.format(reg), file=f)
                    print('\tType: VALIDATION', file=f)
                    print('\tVariableTitle: "{:s}"'.format(limit_config['plotting']['xtitle']), file=f)
                    print('\tLabel: "{:s}"'.format(ctrl_reg_cfg['label']), file=f)
                    print('\tShortLabel: "{:s}"'.format(reg), file=f)
                    print('\tHistoName: "h_{:s}"'.format(reg), file=f)
                    if spec_cfg is not None:
                        print('\tBinning: {:s}'.format(convert_binning(spec_cfg['binning'])), file=f)
                    if 'log_scale' in spec_cfg:
                        print('\tLogScale: {:s}'.format(str(spec_cfg['log_scale']).upper()), file=f)
                    print('\n', file=f)
        norm_factors = set(norm_factors)
        print('% --------------- %', file=f)
        print('% --- SAMPLES - -- %', file=f)
        print('% --------------- %', file=f)

        print('% Normal samples', file=f)

        print('Sample: "Data"', file=f)
        print('\tTitle: "Data"', file=f)
        print('\tType: DATA', file=f)
        print('\tHistoFile: "Data"', file=f)
        print('\tSeparateGammas: TRUE', file=f)
        print('\n', file=f)

        print('Sample: "{:s}"'.format(kwargs['sig_name']), file=f)
        print('\tType: SIGNAL', file=f)
        print('\tTitle: "{:s}"'.format(kwargs['sig_name']), file=f)
        print('\tTexTitle: "$LQ$"', file=f)
        print('\tFillColor: 0', file=f)  # TODO
        print('\tLineColor: 2', file=f)  # TODO
        print('\tHistoFile: "{:s}"'.format(kwargs['sig_name']), file=f)
        print('\tNormFactor: "mu_Sig", 1, 0, 10000', file=f)
        print('\tSeparateGammas: TRUE', file=f)
        print('\tUseMCstat: TRUE', file=f)
        if kwargs['scale_factors'] is not None:
            print('\tLumiScale: {:f}'.format(kwargs['scale_factors'][kwargs['sig_name']][kwargs['mass_cut']]), file=f)
        print('\n', file=f)

        for bkg in [p.name for p in
                    [pc for pc in list(kwargs["process_configs"].values()) if pc.type.lower() == 'background']]:
            print('Sample: "{:s}"'.format(bkg), file=f)
            print('\tType: BACKGROUND', file=f)
            print('\tTitle: "{:s}"'.format(bkg), file=f)
            print('\tTexTitle: "{:s}"'.format(bkg), file=f)
            print('\tFillColor: {:d}'.format(transform_color(kwargs['process_configs'][bkg].color)), file=f)
            print('\tLineColor: {:d}'.format(transform_color(kwargs['process_configs'][bkg].color)), file=f)
            print('\tHistoFile: "{:s}";'.format(bkg), file=f)
            print('\tSmooth: TRUE', file=f)
            print('\tUseMCstat: TRUE', file=f)
            print('\tSeparateGammas: TRUE', file=f)
            print('\n', file=f)

        print('% --------------- %', file=f)
        print('% - NORMFACTORS - %', file=f)
        print('% --------------- %', file=f)

        for norm_fac in norm_factors:
            print('NormFactor: "{:s}"'.format(norm_fac[0]), file=f)
            print('\tTitle: "{:s}"'.format(norm_fac[0]), file=f)
            print('\tNominal: 1', file=f)
            print('\tMin: 0', file=f)
            print('\tMax: 4', file=f)
            print('\tSamples: {:s}'.format(norm_fac[1]), file=f)
            print('\n', file=f)

        print('% --------------- %', file=f)
        print('% - SYSTEMATICS - %', file=f)
        print('% --------------- %', file=f)

        print('% Normalization only', file=f)

        print('Systematic: "lumi"', file=f)
        print('\tTitle: "Luminosity"', file=f)
        print('\tType: OVERALL', file=f)
        print('\tOverallUp: {:f}'.format(limit_config['general']['lumi_uncert']), file=f)
        print('\tOverallDown: -{:f}'.format(limit_config['general']['lumi_uncert']), file=f)
        print('\tSamples: Zjets, ttbar, Others, {:s}'.format(kwargs['sig_name']), file=f)
        print('\tSymmetrisation: TwoSided', file=f)
        print('\tSmoothing: 40', file=f)
        print('\n', file=f)

        print('% Weight systematics', file=f)

        all_processes = [p.name for p in
                         [pc for pc in list(kwargs["process_configs"].values()) if pc.type.lower() == 'background']] + \
                        [kwargs['sig_name']]
        theory_unc = [syst for syst in kwargs['systematics'] if 'theory_split' in syst.name]
        if len(theory_unc) > 0:
            for component in TheoryUncertaintyProvider().get_sherpa_uncerts():
                new_unc = deepcopy(theory_unc[0])
                new_unc.name = component.replace('weight_', '')
                new_unc.variation = 'custom'
                kwargs['systematics'].append(new_unc)
            kwargs['systematics'].remove(theory_unc[0])
        custom_syst = [syst for syst in kwargs['systematics'] if syst.type == 'custom']
        for syst in custom_syst:
            if syst.expand is None:
                continue
            for unc_name in eval(syst.expand):
                new_unc = deepcopy(syst)
                if 'alpha' in unc_name:
                    new_unc.name = unc_name.replace('alpha', 'a_s')
                    new_unc.hist_name = unc_name
                else:
                    new_unc.name = unc_name
                kwargs['systematics'].append(new_unc)
            kwargs['systematics'].remove(syst)

        for syst in kwargs['systematics']:
            syst_name = syst.name
            affected_regions = 'all'
            if syst.affects is not None:
                affected_regions = ''
                re_region = re.compile(syst.affects)
                if re.match(re_region, kwargs['sig_reg_name']):
                    affected_regions += kwargs['sig_reg_name'] + ','
                ctrl_reg_matches = [rn for rn in list(kwargs["ctrl_config"].keys()) if re.match(re_region, rn)]
                affected_regions += ','.join(ctrl_reg_matches)
                affected_regions.rstrip(',')
            if affected_regions == '':
                continue
            affected_samples = ','.join(all_processes)
            if syst.samples is not None:
                if isinstance(syst.samples, list):
                    affected_samples = ','.join(syst.samples)
                if isinstance(syst.samples, dict) or isinstance(syst.samples, OrderedDict):
                    if 'type' in syst.samples:
                        pass
            # tmp fix
            affected_samples = affected_samples.replace('QCD,', '')
            print('Systematic: "{:s}"'.format(syst_name), file=f)
            title = syst.title
            if title is None:
                if syst.envelope is not None:
                    title = syst.envelope
                else:
                    title = syst_name
            print('\tTitle: "{:s}"'.format(title), file=f)
            if syst.type != 'fixed':
                print('\tType: HISTO', file=f)
            else:
                print('\tType: OVERALL', file=f)
            print('\tSamples: {:s}'.format(affected_samples), file=f)
            print('\tRegions: {:s}'.format(affected_regions), file=f)
            if syst.group:
                print('\tCategory: {:s}'.format('instrumental'), file=f)
                print('\tSubCategory: {:s}'.format(syst.group), file=f)
            if syst.type != 'fixed':
                if syst.variation == 'updown' or syst.symmetrise:
                    hist_name = syst_name
                    if syst.hist_name is not None:
                        hist_name = syst.hist_name
                    print('\tHistoPathUp: "{:s}"'.format(os.path.join(hist_path, hist_name + '__1up')), file=f)
                    print('\tHistoPathDown: "{:s}"'.format(os.path.join(hist_path, hist_name + '__1down')), file=f)
                # elif 'theory_envelop' in syst_name:
                #     print('\tHistoPathUp: "{:s}"'.format(os.path.join(hist_path,
                #                                                       syst_name + '_{:.0f}'.format(
                #                                                           kwargs['mass_cut']))),
                #           file=f)
                #     print('\tSymmetrisation: ABSMEAN', file=f)
                elif syst.variation == 'custom':
                    hist_name = syst_name
                    if syst.hist_name is not None:
                        hist_name = syst.hist_name
                    if syst.envelope is not None:
                        print('\tCombineName: {:s}'.format(syst.envelope), file=f)
                        print('\tCombineType: ENVELOPE', file=f)
                    print('\tHistoPathUp: "{:s}"'.format(os.path.join(hist_path, hist_name)), file=f)
                    print('\tHistoPathDown: "{:s}"'.format(os.path.join(hist_path, hist_name)), file=f)
                else:
                    print("SYST NOT UP DOWN: ", syst_name)
            else:
                print(syst.name)
                print('\tOverallUp: {:f}'.format(syst.value), file=f)
                print('\tOverallDown: -{:f}'.format(syst.value), file=f)
            if syst.symmetrise_option is not None:
                print('\tSymmetrisation: {:s}'.format(syst.symmetrise_option), file=f)
            print('\n', file=f)
    print('Wrote config to ', config_name)


def convert_hists(input_hist_file, process_configs, regions, fixed_signal, output_dir, thresholds, systematics,
                  dummy_signal=None, cut_off=None):
    def parse_hist_name(hist_name):
        try:
            region = list(filter(lambda r: r in hist_name, region_names))[0]
            try:
                process = list(filter(lambda p: p + '_' in hist_name, processes))[0]
            except IndexError:
                if 'QCD' not in hist_name:
                    raise IndexError
                process = 'QCD'
            unc = hist_name.split(process)[-1].replace('_clone', '').lstrip('_')
            if unc == '':
                unc = 'Nominal'
            return region, process, unc
        except IndexError:
            return None, None, None

    f = FileHandle(file_name=input_hist_file)
    processes = [pc.name for pc in list(process_configs.values())]
    region_names = [r.name for r in regions]

    scale_factors = None
    data = {}
    sr_thresholds = set(thresholds)
    for h in f.get_objects_by_type('TH1'):
        region, process, unc = parse_hist_name(h.GetName())
        if region is None:
            continue
        unc = unc.replace('weight_', '')
        if unc not in data:
            data[unc] = {process: {region: deepcopy(h)}}
            continue
        if process not in data[unc]:
            data[unc][process] = {}
        if cut_off is not None:
            for b in range(h.GetNbinsX()):
                if h.GetBinCenter(b) > cut_off:
                    break
                h.SetBinContent(b, 0.)
                h.SetBinError(b, 0.)
        data[unc][process][region] = deepcopy(h)

    if dummy_signal is not None:
        # take last process
        data['Nominal'][dummy_signal] = {}
        for reg in region_names:
            data['Nominal'][dummy_signal][reg] = deepcopy(data['Nominal'][process][reg])
            # simplified for now
            if reg.startswith('SR'):
                data['Nominal'][dummy_signal][reg].SetBinContent(1, fixed_signal)
            else:
                data['Nominal'][dummy_signal][reg].SetBinContent(1, 0.)

    if fixed_signal and dummy_signal is None:
        scale_factors = {}
        for _, thrs in sr_thresholds:
            for process in data['Nominal'].keys():
                if [pc for pc in process_configs.items() if pc[0] == process][0][1].type.lower() != "signal":
                    continue
                for reg, h in data['Nominal'][process].items():
                    if not reg.startswith('SR_'):
                        continue
                    lower_bin = h.FindBin(thrs)
                    try:
                        sf = fixed_signal / h.Integral(lower_bin, -1)
                    except ZeroDivisionError:
                        sf = -1.
                    if process not in scale_factors:
                        scale_factors[process] = {}
                    scale_factors[process][thrs] = sf
    make_dirs(os.path.join(output_dir, 'hists'))
    for unc in list(data.keys()):
        if 'theory_envelop' in unc:
            continue
        make_dirs(os.path.join(output_dir, 'hists', unc))
        for process in list(data[unc].keys()):
            f = ROOT.TFile.Open(os.path.join(output_dir, 'hists', unc, '{:s}.root'.format(process)), 'RECREATE')
            f.cd()
            for reg, h in list(data[unc][process].items()):
                _logger.debug('Apply blinding')
                if reg.startswith('SR_') and 'data' in process.lower():
                    for b in range(h.GetNbinsX() + 1):
                        h.SetBinContent(b, 0)
                        h.SetBinError(b, 0)
                h.SetName('h_{:s}'.format(reg))
                h.Write()
            f.Close()
    return scale_factors
