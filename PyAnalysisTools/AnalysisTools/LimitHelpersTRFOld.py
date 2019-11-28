from __future__ import division
from __future__ import print_function
import numbers
import pickle
from copy import deepcopy
from math import sqrt
import json
import numpy as np
import pandas as pd
import ROOT
import os
import re
import sys
import PyAnalysisTools.PlottingUtils.PlottingTools as pt
import PyAnalysisTools.PlottingUtils.Formatting as fm
from PyAnalysisTools.AnalysisTools.RegionBuilder import RegionBuilder
from PyAnalysisTools.AnalysisTools.SystematicsAnalyserOld import parse_syst_config, TheoryUncertaintyProvider
from PyAnalysisTools.PlottingUtils.BasePlotter import BasePlotter
from PyAnalysisTools.base.FileHandle import FileHandle
from PyAnalysisTools.AnalysisTools.XSHandle import XSHandle
from PyAnalysisTools.PlottingUtils.PlotConfig import PlotConfig, get_default_color_scheme, transform_color
from PyAnalysisTools.base.ProcessConfig import find_process_config
from PyAnalysisTools.base.OutputHandle import OutputFileHandle
from PyAnalysisTools.AnalysisTools.MLHelper import Root2NumpyConverter
from PyAnalysisTools.base.YAMLHandle import YAMLLoader as yl
from PyAnalysisTools.base.YAMLHandle import YAMLDumper as yd
from PyAnalysisTools.base import _logger
from PyAnalysisTools.base.ShellUtils import make_dirs, copy
from collections import OrderedDict
import dill
from PyAnalysisTools.base.OutputHandle import SysOutputHandle as soh

try:
    from tabulate.tabulate import tabulate
except ImportError:
    from tabulate import tabulate
sys.modules[tabulate.__module__].LATEX_ESCAPE_RULES = {}


def dump_input_config(cfg, output_dir):
    if output_dir is None:
        return
    yd.dump_yaml(dict(filter(lambda kv: kv[0] != 'scan_info' and kv[0] != 'xsec_handle', cfg.iteritems())),
                 os.path.join(output_dir, 'config.yml'))


class Yield(object):
    def __init__(self, weights):
        self.weights = weights
        self.original_weights = []
        self.scale_factor = []
        self.extrapolated = False
        try:
            self.weights.dtype = np.float64
        except ValueError:
            pass
        except AttributeError:
            self.weights = np.array(weights)
            self.weights.dtype = np.float64

    def __add__(self, other):
        self.append(other)
        return self

    def __radd__(self, other):
        """
        Overloaded swapped add. Needed for sum()
        :param other: number like object
        :type other: int or Yield
        :return: this with weights summed. For other == 0 (first operand in sum() return just self)
        :rtype: Yield
        """
        if other == 0:
            return self
        return self + other

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            self.weights *= other.weights
        else:
            self.original_weights += [deepcopy(self.weights)]
            self.weights *= other
            self.scale_factor.append(other)
        return self

    def __rmul__(self, other):
        return self * other

    def __imul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        tmp_sum = other.weights.sum()
        if tmp_sum == 0.:
            if self.weights.sum() == 0.:
                return 0.
            _logger.error('Summed other to null. Return Nan {:f}'.format(self.weights.sum()))
            return np.nan
        return self.weights.sum() / other.weights.sum()

    def reduce(self):
        init = len(self.weights)
        self.weights = self.weights[abs(self.weights) < 40.]
        if init != len(self.weights):
            print('removed weight')

    def append(self, other):
        if isinstance(other, self.__class__):
            self.weights = np.append(self.weights, other.weights)
            self.original_weights += other.original_weights
            self.scale_factor += other.scale_factor
        elif isinstance(other, numbers.Number):
            self.weights = np.append(self.weights, other)
            self.original_weights += [other]
            self.scale_factor.append(1.)
        else:
            _logger.error('Cannot add object of type {:s} to Yield'.format(type(other)))

    def sum(self):
        return np.sum(self.weights)

    def stat_unc(self):
        stat_unc = 0.
        try:  # temporary fix while attribute not available in already processed yields
            if self.extrapolated:
                return np.sqrt(np.sum(self.weights))
        except AttributeError:
            pass
        if len(self.original_weights) == 0:
            return np.sqrt(np.sum(self.weights * self.weights))
        for i in range(len(self.original_weights)):
            stat_unc += self.scale_factor[i] * self.scale_factor[i] * np.sum(self.original_weights[i]
                                                                             * self.original_weights[i])
        return np.sqrt(stat_unc)

    def is_null(self):
        return True in pd.isnull(self.weights)


class LimitArgs(object):
    def __init__(self, output_dir, fit_mode, **kwargs):
        kwargs.setdefault("ctrl_syst", None)
        kwargs.setdefault("skip_ws_build", False)
        kwargs.setdefault("base_output_dir", output_dir)
        kwargs.setdefault("fixed_xsec", None)
        kwargs.setdefault("run_pyhf", False)
        self.skip_ws_build = kwargs['skip_ws_build']
        self.fit_mode = fit_mode
        self.output_dir = output_dir
        self.base_output_dir = kwargs['base_output_dir']
        self.job_id = kwargs["jobid"]
        self.sig_reg_name = kwargs["sig_reg_name"]
        self.run_pyhf = kwargs['run_pyhf']
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
        for process in self.kwargs['process_configs'].keys():
            obj_str += '\t{:s}\n'.format(process)
        for sig_reg, yld in self.kwargs['sig_yield'].iteritems():
            obj_str += 'SR {:s} yield: {:.2f} +- {:.2f}\n'.format(sig_reg, *yld)

        if 'sr_syst' in self.kwargs:
            for sig_reg in self.kwargs['sr_syst'].keys():
                obj_str += 'SR systematics for {:s}: \n'.format(sig_reg)
                for process in self.kwargs['sr_syst'][sig_reg].keys():
                    obj_str += 'Process: {:s}\n'.format(process)
                    for name, unc in self.kwargs['sr_syst'][sig_reg][process].iteritems():
                        obj_str += '\t{:s}\t\t{:.2f}\n'.format(name, unc)
                    obj_str += '\n'
        for sig_reg in self.kwargs['bkg_yields'].keys():
            obj_str += 'Bkg yields in {:s}: \n'.format(sig_reg)
            for process, ylds in self.kwargs['bkg_yields'][sig_reg].iteritems():
                obj_str += '\t{:s}\t\t{:.2f} +- {:.2f}\n'.format(process, *ylds)
        if self.kwargs['ctrl_syst'] is not None:
            obj_str += 'CR systematics: \n'
            for region in self.kwargs['ctrl_syst'].keys():
                obj_str += 'CR: {:s}\n'.format(region)
                for process in self.kwargs['ctrl_syst'][region].keys():
                    obj_str += 'Process: {:s}\n'.format(process)
                    for name, unc in self.kwargs['ctrl_syst'][region][process].iteritems():
                        obj_str += '\t{:s}\t\t{:.2f}\n'.format(name, unc)
                    obj_str += '\n'
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
            for sn, sy in self.kwargs['bkg_yields'][name].iteritems():
                channel['samples'].append(build_sample(sn, sy))
            return channel

        def add_channel(name, info):
            data = None
            channel = OrderedDict({'name': name, 'samples': []})
            for sn, sy in info.iteritems():
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
            syst_strings = set(map(lambda sn: sn.split('__')[0], systematics.keys()))
            for syst in syst_strings:
                try:
                    down, up = sorted(filter(lambda s: syst in s[0], systematics.iteritems()), key=lambda i: i[0])
                    sample['modifiers'].append({"type": "histosys", "data": {"lo_data": [down[1] * sample['data'][0]],
                                                                             "hi_data": [up[1] * sample['data'][0]]},
                                                'name': syst})
                except ValueError:
                    variation = sorted(filter(lambda s: syst in s[0], systematics.iteritems()), key=lambda i: i[0])[0]
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
        for region, info in self.kwargs['control_regions'].iteritems():
            channel, data = add_channel(region, info)
            specs['channels'].append(channel)
            if data is None:
                data = [0.]
            specs['data'][region] = data
        for sig_reg in self.kwargs['sig_yield'].keys():
            specs['data'][sig_reg] = [0.]

        specs['measurements'] = [{'config': {'poi': 'mu', 'parameters': []}, 'name': self.sig_reg_name}]

        for sig_reg in self.kwargs['sr_syst'].keys():
            for process, systematics in self.kwargs['sr_syst'][sig_reg].iteritems():
                channel_id = [i for i, c in enumerate(specs['channels']) if c['name'] == sig_reg][0]
                add_systematics(channel_id, process, systematics)
        for region in self.kwargs['ctrl_syst'].keys():
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
                                          'is_val_region': region.val_region}
    return limit_region_info


def read_workspace_from_file(file_name, ws_name="w"):
    f = ROOT.TFile.Open(file_name)
    return f.Get(ws_name)


def get_fit_quality(file_name, ws_name="w", fr_name="RooExpandedFitResult_afterFit"):
    ws = read_workspace_from_file(file_name, ws_name)
    fit_result = ws.obj(fr_name)
    return fit_result.status(), fit_result.covQual()


def make_cross_section_limit_plot(data, plot_config, sig_reg_name=''):
    data.sort()
    if plot_config['ytitle'] is None:
        ytitle = "95% CL U.L on #sigma [pb]"
    pc = PlotConfig(name="xsec_limit_{:s}".format(sig_reg_name), ytitle=ytitle, xtitle=plot_config['xtitle'], draw='ap',
                    logy=True, lumi=plot_config.get_lumi(), watermark=plot_config['watermark'])
    graph = ROOT.TGraph(len(data))
    for i, item in enumerate(data):
        graph.SetPoint(i, item[0], item[1] * item[2] / (plot_config.get_lumi() * 1000.))
    graph.SetName('xsec_limit_{:s}'.format(sig_reg_name))
    canvas = pt.plot_obj(graph, pc)
    fm.decorate_canvas(canvas, pc)
    return canvas


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
        for attribute, value in self.__dict__.items():
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
        for k, v in kwargs.iteritems():
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


class LimitAnalyser(object):
    def __init__(self, input_path, analysis_name):
        self.input_path = input_path
        self.limit_fname = os.path.join(input_path, 'results/{:s}_Output_upperlimit.root'.format(analysis_name))
        self.fit_fname = os.path.join(input_path, 'results', analysis_name,
                                      'SPlusB_combined_NormalMeasurement_model_afterFit.root')
        self.limit_info = LimitInfo()

    def analyse_limit(self, sig_name="Sig"):
        fit_status, fit_cov_quality = get_fit_quality(self.fit_fname)
        self.limit_info.add_info(fit_status=fit_status, fit_cov_quality=fit_cov_quality)
        exp_limit, exp_limit_up, exp_limit_low = get_expected_limit(self.limit_fname,
                                                                    "hypo_{:s}".format(sig_name))
        self.limit_info.add_info(exp_limit=exp_limit, exp_limit_up=exp_limit_up, exp_limit_low=exp_limit_low)
        return self.limit_info


class LimitAnalyserCL(object):
    def __init__(self, input_path):
        self.input_path = input_path
        self.limit_info = LimitInfo()
        self.converter = Root2NumpyConverter(['exp_upperlimit', 'exp_upperlimit_plus1', 'exp_upperlimit_plus2',
                                              'exp_upperlimit_minus1', 'exp_upperlimit_minus2', 'fit_status'])

    def analyse_limit(self, signal_scale=1., fixed_signal=None, sig_yield=None, pmg_xsec=None):
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
            scale_factor = 1000. * signal_scale
            if fixed_signal is not None:
                if isinstance(sig_yield, OrderedDict):
                    if isinstance(sig_yield.values()[0], numbers.Number):
                        print('Calc SF ', scale_factor, fixed_signal, sum([yld for yld in sig_yield.values()]))
                        scale_factor = scale_factor * sum([fixed_signal / yld for yld in sig_yield.values()])
                    else:
                        print('Calc SF[0] ', scale_factor, fixed_signal, sum([yld[0] for yld in sig_yield.values()]))
                        scale_factor = scale_factor * sum([fixed_signal / yld[0] for yld in sig_yield.values()])
                else:
                    print('Calc SF with sig)yield ', scale_factor, fixed_signal, sig_yield)

                    scale_factor = scale_factor * fixed_signal / sig_yield
            if pmg_xsec is not None:
                scale_factor = 1000.
            return scale_factor

        try:
            fh = FileHandle(file_name=os.path.join(self.input_path, 'asymptotics/myLimit_CL95.root'),
                            switch_off_process_name_analysis=True)
            tree = fh.get_object_by_name('stats')
            data = self.converter.convert_to_array(tree=tree)
            fit_status = data['fit_status']  # , fit_cov_quality = get_fit_quality(self.fit_fname)
            scale_factor = get_scale_factor(signal_scale)
            print("Analysing ", self.input_path, scale_factor, data['exp_upperlimit'], signal_scale, fixed_signal)
            self.limit_info.add_info(fit_status=fit_status, fit_cov_quality=-1)
            self.limit_info.add_info(exp_limit=data['exp_upperlimit'] * scale_factor,
                                     exp_limit_up=data['exp_upperlimit_plus1'] * scale_factor,
                                     exp_limit_low=data['exp_upperlimit_minus1'] * scale_factor)

        except ValueError:
            try:
                with open(os.path.join(self.input_path, 'limit.json'), 'r') as f:
                    data = json.load(f)
                scale_factor = get_scale_factor(signal_scale)
                self.limit_info.add_info(fit_status=1, fit_cov_quality=1, exp_limit=data['CLs_exp'][2] * scale_factor,
                                         exp_limit_up=data['CLs_exp'][3] * scale_factor,
                                         exp_limit_low=data['CLs_exp'][1] * scale_factor)
            except (ZeroDivisionError, IOError):
                self.limit_info.add_info(fit_status=-1, fit_cov_quality=-1, exp_limit=-1, exp_limit_up=-1,
                                         exp_limit_low=-1)
        return self.limit_info


class LimitPlotter(object):
    def __init__(self, output_handle):
        self.output_handle = output_handle

    def make_cross_section_limit_plot(self, limits, plot_config, theory_xsec=None, sig_reg_name=None):
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
        if sig_reg_name is not None:
            if not sig_reg_name.startswith('_'):
                sig_reg_name = '_{:s}'.format(sig_reg_name)
        else:
            sig_reg_name = ''
        limits.sort(key=lambda li: li.mass)
        ytitle = "95% CL U.L on #sigma [pb]"
        pc = PlotConfig(name='xsec_limit{:s}'.format(sig_reg_name), ytitle=ytitle, xtitle=plot_config['xtitle'],
                        draw='pLX', logy=True,
                        lumi=plot_config['lumi'], watermark=plot_config['watermark'], ymin=float(1e-6),
                        ymax=float(1.), )
        pc_1sigma = deepcopy(pc)
        pc_2sigma = deepcopy(pc)
        pc_1sigma.color = ROOT.kGreen
        pc_2sigma.color = ROOT.kYellow
        pc_1sigma.draw = "3"
        pc_2sigma.draw = "3"
        pc_1sigma.style_setter = "Fill"
        pc_2sigma.style_setter = "Fill"

        graph = ROOT.TGraph(len(limits))
        graph_1sigma = ROOT.TGraphAsymmErrors(len(limits))
        graph_2sigma = ROOT.TGraphAsymmErrors(len(limits))
        for i, limit in sorted(enumerate(limits)):
            print("Set Point: ", limit.mass, limit.exp_limit)
            graph.SetPoint(i, limit.mass, limit.exp_limit)
            graph_1sigma.SetPoint(i, limit.mass, limit.exp_limit)
            graph_2sigma.SetPoint(i, limit.mass, limit.exp_limit)
            graph_1sigma.SetPointEYhigh(i, limit.exp_limit_up - limit.exp_limit)
            graph_1sigma.SetPointEYlow(i, limit.exp_limit - limit.exp_limit_low)
            graph_2sigma.SetPointEYhigh(i, 2. * (limit.exp_limit_up - limit.exp_limit))
            graph_2sigma.SetPointEYlow(i, 2. * (limit.exp_limit - limit.exp_limit_low))
        if theory_xsec is not None:
            graph_theory = []
            processed_mass_points = sorted(map(lambda li: li.mass, limits))

            for process, xsecs in theory_xsec.iteritems():
                masses = []
                cross_sections = []
                for mass, lam, xsec in xsecs:
                    if mass not in processed_mass_points:
                        continue
                    masses.append(mass)
                    cross_sections.append(xsec)
                graph_theory.append(ROOT.TGraph(len(masses)))
                for i in range(len(masses)):
                    graph_theory[-1].SetPoint(i, masses[i], cross_sections[i])
                graph_theory[-1].SetName('Theory_prediction_{:s}'.format(process))

        graph_2sigma.SetName('xsec_limit{:s}'.format(sig_reg_name))
        canvas = pt.plot_obj(graph_2sigma, pc_2sigma)
        pt.add_graph_to_canvas(canvas, graph_1sigma, pc_1sigma)
        pt.add_graph_to_canvas(canvas, graph, pc)
        labels = ['expected limit', '#pm 1#sigma', '#pm 2#sigma']
        legend_format = ['PL', 'F', 'F']
        plot_objects = [graph, graph_1sigma, graph_2sigma]
        if theory_xsec is not None:
            pc_theory = deepcopy(pc)
            pc_theory.draw = 'line'
            colors = get_default_color_scheme()
            for i, g in enumerate(graph_theory):
                pc_theory.color = colors[i]
                pt.add_graph_to_canvas(canvas, g, pc_theory)
                labels.append("Theory {:s}".format(g.GetName().split('_')[-1]))
                legend_format.append("L")
                plot_objects.append(g)
        fm.decorate_canvas(canvas, pc)
        fm.add_legend_to_canvas(canvas, plot_objects=plot_objects, labels=labels, format=legend_format)
        self.output_handle.register_object(canvas)

    def make_limit_plot_plane(self, limits, plot_config, theory_xsec, sr_name):
        def find_excluded_lambda(mass, xsec, excl_limit):
            xsecs = filter(lambda xs: xs[0] == mass, xsec)
            try:
                return sqrt(excl_limit / xsecs[0][-1])
            except IndexError:
                return None

        if theory_xsec is None:
            return
        graphs_contour = []
        for process, xsecs in theory_xsec.iteritems():
            excl_lambdas = [(limit.mass, find_excluded_lambda(limit.mass, xsecs, limit.exp_limit)) for limit in limits]
            excl_lambdas = filter(lambda v: v[1] is not None, excl_lambdas)
            excl_lambdas = sorted(excl_lambdas, key=lambda i: i[0])
            graphs_contour.append(ROOT.TGraph(len(excl_lambdas)))
            for i, limit in enumerate(excl_lambdas):
                graphs_contour[-1].SetPoint(i, limit[0], limit[1])
            graphs_contour[-1].SetName('Limit_contour_{:s}'.format(process))

        pc = PlotConfig(name='limit_contour_{:s}'.format(sr_name), watermark=plot_config['watermark'], ymax=10.,
                        xtitle=plot_config['xtitle'], ytitle='#lambda', logy=False, ymin=0., draw='Line',
                        lumi=plot_config['lumi'], labels=map(lambda g: g.GetName().split('_')[-1], graphs_contour),
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
        self.input_path = kwargs['input_path']
        self.output_handle = OutputFileHandle(output_dir=kwargs['output_dir'])
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
            self.scan_info = yl.read_yaml(os.path.join(self.input_path, "scan_info.yml"), None)
        dump_input_config(self.__dict__, self.output_handle.output_dir)

    def read_theory_cross_sections(self, file_name):
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
                limit_info = analyser.analyse_limit(scan.kwargs['signal_scale'], scan.kwargs['fixed_signal'],
                                                    scan.kwargs['sig_yield'])  # scan.kwargs['sig_name'])
            except ReferenceError:
                _logger.error("Could not find info for scan {:s}".format(scan))
                continue
            limit_info.sig_name = scan.kwargs['sig_name']
            mass = float(re.findall(r'\d{3,4}', scan.kwargs['sig_name'])[0])
            self.theory_xsec[mass] = None
            limit_info.add_info(mass_cut=scan.kwargs["mass_cut"],
                                mass=mass)
            parsed_data.append(limit_info)
        limits = LimitScanAnalyser.find_best_limit(parsed_data)
        theory_xsec = None
        if 'SR_mu_bveto' in self.sig_reg_name:
            chains = ['LQmud', 'LQmus']
        elif 'SR_mu_btag' in self.sig_reg_name:
            chains = ['LQmub']
        elif 'SR_el_bveto' in self.sig_reg_name:
            chains = ['LQed', 'LQes']
        elif 'SR_el_btag' in self.sig_reg_name:
            chains = ['LQeb']
        else:
            chains = []
        if self.xsec_map is not None:
            theory_xsec = dict(filter(lambda kv: kv[0] in chains, self.xsec_map.iteritems()))
        self.plotter.make_limit_plot_plane(limits, self.plot_config, theory_xsec, scan.sig_reg_name)

        self.plotter.make_cross_section_limit_plot(limits, self.plot_config, theory_xsec,
                                                   sig_reg_name=scan.sig_reg_name)

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
            self.scan_info = yl.read_yaml(os.path.join(self.input_path, "scan_info.yml"), None)
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
                fixed_xsec = None
                if 'fixed_xsec' in scan.kwargs:
                    fixed_xsec = scan.kwargs['fixed_xsec']
                limit_info = analyser.analyse_limit(scan.kwargs['signal_scale'], scan.kwargs['fixed_signal'],
                                                    scan.kwargs['sig_yield'], fixed_xsec)
            except ReferenceError:
                _logger.error("Could not find info for scan {:s}".format(scan))
                continue
            limit_info.sig_name = scan.kwargs['sig_name']
            mass = float(re.findall(r'\d{3,4}', scan.kwargs['sig_name'])[0])
            self.theory_xsec[mass] = None
            limit_info.add_info(mass_cut=scan.kwargs["mass_cut"],
                                mass=mass)
            parsed_data.append(limit_info)
            # self.parse_prefit_yields(scan, mass)
        best_limits = self.find_best_limit(parsed_data)
        self.make_scan_plot(parsed_data, self.plot_config, best_limits)
        self.tabulate_limits(best_limits)
        theory_xsec = None

        if self.xsec_map is not None:
            chains = ['LQmud', 'LQmus']
            # chains = ['LQeb']
            theory_xsec = OrderedDict()
            # TODO: needs proper implementation
            # for mode in chains:
            #     # self.plotter.make_limit_plot_plane(best_limits, self.plot_config, self.xsec_map[mode],
            #     #                                    scan.kwargs['sig_name'])
            theory_xsec = dict(filter(lambda kv: kv[0] in chains, self.xsec_map.iteritems()))
        self.plotter.make_cross_section_limit_plot(best_limits, self.plot_config, theory_xsec, self.sig_reg_name)
        if theory_xsec is not None:
            self.plotter.make_limit_plot_plane(best_limits, self.plot_config, theory_xsec,
                                               scan.sig_reg_name)
        self.output_handle.write_and_close()

    def tabulate_limits(self, limits):
        limits.sort(key=lambda l: l.mass)
        with open(os.path.join(self.input_path, 'event_yields_nom.pkl'), 'r') as f:
            event_yields = dill.load(f)
        data = []
        ordering = self.plot_config['ordering']
        for limit in limits:
            data_mass_point = [limit.mass, limit.mass_cut, limit.exp_limit]
            prefit_ylds_bkg = event_yields.retrieve_bkg_ylds(limit.mass_cut, event_yields.get_signal_region_names()[0])
            prefit_ylds_sig = event_yields.retrieve_signal_ylds(limit.sig_name, limit.mass_cut,
                                                                event_yields.get_signal_region_names()[0])[0] / 1000.
            data_mass_point.append(prefit_ylds_sig * limit.exp_limit)
            for process in ordering:
                data_mass_point.append("{:.2f} $\\pm$ {:.2f}".format(*prefit_ylds_bkg[process]))
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
        limits = filter(lambda li: li.exp_limit > 0., limits)
        masses = set(map(lambda li: li.mass, limits))
        best_limits = []
        for mass in masses:
            # if mass == 500:
            #     limits = filter(lambda li: li.mass_cut < 1000., limits)
            mass_limits = filter(lambda li: li.mass == mass, limits)
            best_limits.append(min(mass_limits, key=lambda li: li.exp_limit))
            print("FOUND ", mass, best_limits[-1])
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
            signal_process = filter(lambda p: "LQ" in p, yields.keys())[0]
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
        for mass, cut in self.prefit_yields.keys():
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
        hist_best = hist.Clone("best_limit")
        hist_fit_status = hist.Clone("fit_status")
        hist_fit_quality = hist.Clone("fit_quality")
        limit_scan_table = OrderedDict()
        mass_points = sorted(set(map(lambda i: i.mass, parsed_data)))
        for limit_info in parsed_data:
            if limit_info.mass_cut not in limit_scan_table:
                limit_scan_table[limit_info.mass_cut] = [-1.] * len(mass_points)
            limit_scan_table[limit_info.mass_cut][mass_points.index(limit_info.mass)] = limit_info.exp_limit

            if limit_info.exp_limit > 0:
                hist.Fill(limit_info.mass, limit_info.mass_cut, limit_info.exp_limit * 1000.)
            hist_fit_status.Fill(limit_info.mass, limit_info.mass_cut, limit_info.fit_status + 1)
            hist_fit_quality.Fill(limit_info.mass, limit_info.mass_cut, limit_info.fit_cov_quality)

        ROOT.gStyle.SetPalette(1)
        # ROOT.gStyle.SetPaintTextFormat(".2g")
        pc = PlotConfig(name="limit_scan_{:s}".format(self.sig_reg_name), draw_option="COLZ",
                        xtitle=plot_config['xtitle'], ytitle=plot_config['ytitle'], ztitle="95% CL U.L. #sigma [fb]",
                        watermark='Internal', lumi=plot_config['lumi'])
        pc_status = PlotConfig(name="limit_status_{:s}".format(self.sig_reg_name), draw_option="COLZTEXT",
                               xtitle=plot_config['xtitle'], ytitle=plot_config['ytitle'],
                               ztitle="fit status + 1", zmin=-1., watermark='Internal', lumi=plot_config['lumi'])
        pc_cov_quality = PlotConfig(name="limit_cov_quality_{:s}".format(self.sig_reg_name), draw_option="COLZTEXT",
                                    xtitle=plot_config['xtitle'], ytitle=plot_config['ytitle'],
                                    ztitle="fit cov quality", watermark='Internal', lumi=plot_config['lumi'])
        canvas = pt.plot_obj(hist, pc)
        if best_limits is not None:
            pc_best = PlotConfig(draw_option="BOX")
            for limit in best_limits:
                hist_best.Fill(limit.mass, limit.mass_cut, hist.GetMaximum())
            hist_best.SetLineColor(ROOT.kRed)
            pt.add_histogram_to_canvas(canvas, hist_best, pc_best)
        fm.decorate_canvas(canvas, pc)
        self.output_handle.register_object(canvas)
        canvas_status = pt.plot_obj(hist_fit_status, pc_status)
        self.output_handle.register_object(canvas_status)
        canvas_quality = pt.plot_obj(hist_fit_quality, pc_cov_quality)
        self.output_handle.register_object(canvas_quality)
        limit_scan_table = [['{:.0f}'.format(i[0])] + i[1] for i in limit_scan_table.items()]
        with open(os.path.join(self.output_handle.output_dir,
                               'limit_scan_table_{:s}.tex'.format(self.sig_reg_name)), 'w') as f:
            print(tabulate(limit_scan_table, headers=['LQ mass'] + mass_points, tablefmt='latex_raw'), file=f)


class Sample(object):
    def __init__(self, process, gen_ylds):
        if isinstance(process, str):
            self.name = process
            self.process = None
            self.is_data = 'data' in process
        else:
            self.name = process.process_name
            if process.mc_campaign is not None:
                self.name += '.{:s}'.format(process.mc_campaign)
            self.process = process
            self.is_data = process.is_data
        self.generated_ylds = gen_ylds
        self.nominal_evt_yields = {}
        self.shape_uncerts = {}
        self.scale_uncerts = {}
        self.ctrl_region_yields = {}
        self.ctrl_reg_scale_ylds = {}
        self.ctrl_reg_shape_ylds = {}
        self.is_signal = False
        self.theory_uncert_provider = TheoryUncertaintyProvider()

    def __str__(self):
        """
        Overloaded str operator. Get's called if object is printed
        :return: formatted string with name and attributes
        :rtype: str
        """
        obj_str = "Sample : {:s} \n".format(self.name)
        for attribute, value in self.__dict__.items():
            if attribute == 'name':
                continue
            obj_str += '{}={} \n'.format(attribute, value)
        return obj_str

    def __repr__(self):
        """
        Overloads representation operator. Get's called e.g. if list of objects are printed
        :return: formatted string with name and attributes
        :rtype: str
        """
        return self.__str__() + '\n'

    def clear_yields(self):
        for sr in self.nominal_evt_yields.keys():
            for cut in self.nominal_evt_yields[sr].keys():
                self.nominal_evt_yields[sr][cut] = None
                self.shape_uncerts[sr][cut] = {}
                self.scale_uncerts[sr][cut] = {}
        for cr in self.ctrl_region_yields.keys():
            self.ctrl_region_yields[cr] = {}
            self.ctrl_reg_shape_ylds[cr] = {}
            self.ctrl_reg_scale_ylds[cr] = {}

    def add_signal_region_yields(self, sr_name, cut, nom_yields, shape_uncerts=None):
        for syst in nom_yields.keys():
            if syst == 'weight' or 'pdf_uncert' in syst:
                continue
            nom_yields[syst] *= nom_yields['weight']
        if sr_name not in self.nominal_evt_yields:
            self.nominal_evt_yields[sr_name] = {}
            self.shape_uncerts[sr_name] = {}
            self.scale_uncerts[sr_name] = {}

        self.nominal_evt_yields[sr_name][cut] = nom_yields['weight']
        if shape_uncerts is not None:
            self.shape_uncerts[sr_name][cut] = {syst: yld for syst, yld in shape_uncerts.iteritems()}
        self.scale_uncerts[sr_name][cut] = {syst: yld for syst, yld in nom_yields.iteritems() if not syst == 'weight'}
        self.theory_uncert_provider.calculate_envelop_count(self.scale_uncerts[sr_name][cut])
        self.scale_uncerts[sr_name][cut] = dict(filter(lambda kv: 'pdf_uncert' not in kv[0],
                                                       self.scale_uncerts[sr_name][cut].iteritems()))

    def add_ctrl_region(self, region_name, nominal_evt_yields, shape_uncert_yields=None):
        for syst in nominal_evt_yields.keys():
            if syst == 'weight' or 'pdf_uncert' in syst:
                continue
            nominal_evt_yields[syst] *= nominal_evt_yields['weight']

        self.ctrl_reg_scale_ylds[region_name] = {syst: yld for syst, yld in nominal_evt_yields.iteritems() if
                                                 not syst == 'weight'}
        self.theory_uncert_provider.calculate_envelop_count(self.ctrl_reg_scale_ylds[region_name])
        self.ctrl_reg_scale_ylds[region_name] = dict(filter(lambda kv: 'pdf_uncert' not in kv[0],
                                                            self.ctrl_reg_scale_ylds[region_name].iteritems()))
        if shape_uncert_yields is not None:
            self.ctrl_reg_shape_ylds[region_name] = {syst: yld for syst, yld in
                                                     shape_uncert_yields.iteritems()}
        else:
            if self.ctrl_reg_shape_ylds is None:
                self.ctrl_reg_shape_ylds = {region_name: {}}
            else:
                self.ctrl_reg_shape_ylds[region_name] = {}
        self.ctrl_region_yields[region_name] = nominal_evt_yields['weight']

    def build_sum(self):
        for region in self.nominal_evt_yields.keys():
            for cut, yld in self.nominal_evt_yields[region].iteritems():
                self.nominal_evt_yields[region][cut] = yld.sum(), yld.stat_unc()
        for region, yld in self.ctrl_region_yields.iteritems():
            self.ctrl_region_yields[region] = yld.sum(), yld.stat_unc()

    def remove_empties(self):
        self.scale_uncerts = {cut: dict(filter(lambda ylds: ylds[1] > 0., syst.iteritems()))
                              for cut, syst in self.scale_uncerts.iteritems()}
        self.shape_uncerts = {cut: dict(filter(lambda ylds: ylds[1] > 0., syst.iteritems()))
                              for cut, syst in self.shape_uncerts.iteritems()}
        for region in self.ctrl_reg_scale_ylds.keys():
            self.ctrl_reg_scale_ylds[region] = dict(filter(lambda ylds: ylds[1] > 0.,
                                                           self.ctrl_reg_scale_ylds[region].iteritems()))
            self.ctrl_reg_shape_ylds[region] = dict(filter(lambda ylds: ylds[1] > 0.,
                                                           self.ctrl_reg_shape_ylds[region].iteritems()))

    def calculate_relative_uncert(self):
        for region in self.nominal_evt_yields.keys():
            for cut in self.shape_uncerts[region].keys():
                nominal = self.nominal_evt_yields[region][cut]
                for syst, yld in self.shape_uncerts[region][cut].iteritems():
                    self.shape_uncerts[region][cut][syst] = yld / nominal
                for syst, yld in self.scale_uncerts[region][cut].iteritems():
                    if 'theory' in syst and cut == 900.:
                        print("THEO yields ", yld.weights, nominal.weights)
                        print("THEO yields sum", np.sum(yld.weights), np.sum(nominal.weights), yld / nominal)
                    self.scale_uncerts[region][cut][syst] = yld / nominal
        for region in self.ctrl_region_yields.keys():
            ctrl_nom_ylds = self.ctrl_region_yields[region]
            for syst, yld in self.ctrl_reg_scale_ylds[region].iteritems():
                self.ctrl_reg_scale_ylds[region][syst] = yld / ctrl_nom_ylds
            for syst, yld in self.ctrl_reg_shape_ylds[region].iteritems():
                self.ctrl_reg_shape_ylds[region][syst] = yld / ctrl_nom_ylds

    def remove_unused(self):
        self.remove_low_systematics()

    def apply_xsec_weight(self, lumi, xs_handle, signal_xsec):
        if self.is_data:
            return
        if self.is_signal:
            weight = xs_handle.get_lumi_scale_factor(self.name.split('.')[0], lumi, self.generated_ylds,
                                                     fixed_xsec=signal_xsec)
        else:
            weight = xs_handle.get_lumi_scale_factor(self.name.split('.')[0], lumi, self.generated_ylds)
        for region in self.nominal_evt_yields.keys():
            for cut in self.nominal_evt_yields[region].keys():
                self.nominal_evt_yields[region][cut] *= weight
                for syst in self.scale_uncerts[region][cut].keys():
                    self.scale_uncerts[region][cut][syst] *= weight
                for syst in self.shape_uncerts[region][cut].keys():
                    self.shape_uncerts[region][cut][syst] *= weight

        for region in self.ctrl_region_yields.keys():
            self.ctrl_region_yields[region] *= weight
            for syst in self.ctrl_reg_scale_ylds[region].keys():
                self.ctrl_reg_scale_ylds[region][syst] *= weight
            for syst in self.ctrl_reg_shape_ylds[region].keys():
                self.ctrl_reg_shape_ylds[region][syst] *= weight

    def __add__(self, other):
        self.generated_ylds += other.generated_ylds
        for region in self.nominal_evt_yields.keys():
            for cut in self.nominal_evt_yields[region].keys():
                self.nominal_evt_yields[region][cut].append(other.nominal_evt_yields[region][cut])
            for cut in self.shape_uncerts[region].keys():
                for syst in self.shape_uncerts[region][cut].keys():
                    self.shape_uncerts[region][cut][syst].append(other.shape_uncerts[region][cut][syst])
            for cut in self.scale_uncerts[region].keys():
                for syst in self.scale_uncerts[region][cut].keys():
                    self.scale_uncerts[region][cut][syst].append(other.scale_uncerts[region][cut][syst])
        for region in self.ctrl_region_yields:
            self.ctrl_region_yields[region].append(other.ctrl_region_yields[region])
            for syst in self.ctrl_reg_scale_ylds[region].keys():
                self.ctrl_reg_scale_ylds[region][syst].append(other.ctrl_reg_scale_ylds[region][syst])
            try:
                for syst in self.ctrl_reg_shape_ylds[region].keys():
                    try:
                        self.ctrl_reg_shape_ylds[region][syst].append(other.ctrl_reg_shape_ylds[region][syst])
                    except KeyError as ke:
                        _logger.error('Could not find control region systematic {:s} for region {:s}'.format(syst,
                                                                                                             region))
                        _logger.error(
                            'Available systematics {:s}'.format(', '.join(self.ctrl_reg_shape_ylds[region].keys())))

                        raise ke
            except KeyError as ke:
                _logger.error('Could not find control region systematic b/c of missing region {:s}'.format(region))
                _logger.error('Available regions {:s}'.format(', '.join(self.ctrl_reg_shape_ylds.keys())))
                if self.ctrl_region_yields[region] == 0:
                    continue
                raise ke
        return self

    def __radd__(self, other):
        if other == 0:
            return self
        if other is None:
            _logger.warning('Found none, which is surprising')
            return self
        return self.__add__(other)

    def filter_systematics(self, shape_uncert=None, scale_uncert=None, threshold=0.001):
        """
        Filter systematic uncertainties based on input list of systematics and threshold (set to 1 per-mille)
        :param shape_uncert: shape uncertainty names
        :type shape_uncert: dict
        :param scale_uncert: scale uncertainty names
        :type scale_uncert: dict
        :param threshold: threshold (systematic variation must be larger)
        :type threshold: float
        :return: nothing
        :rtype: None
        """

        def affects_signal_reg(syst_name, syst_yields):
            for ylds in syst_yields.values():
                if syst_name in ylds[cut]:
                    return True
            return False

        def convert_name(name):
            new_name = name.replace('weight_', '').replace('__1up', '').replace('__1down', '')
            return new_name

        for region in self.shape_uncerts.keys():
            for cut in self.shape_uncerts[region].keys():
                if shape_uncert is not None:
                    self.shape_uncerts[region][cut] = dict(
                        filter(lambda kv: convert_name(kv[0]) in map(lambda i: i[0] if not isinstance(i, str) else i,
                                                                     shape_uncert),
                               self.shape_uncerts[region][cut].iteritems()))
                if scale_uncert is not None:
                    self.scale_uncerts[region][cut] = dict(
                        filter(lambda kv: convert_name(kv[0]) in map(lambda i: i[0] if not isinstance(i, str) else i,
                                                                     scale_uncert),
                               self.scale_uncerts[region][cut].iteritems()))
                # self.shape_uncerts[region][cut] = dict(filter(lambda kv: abs(1. - kv[1]) > threshold,
                #                                               self.shape_uncerts[region][cut].iteritems()))
                # self.scale_uncerts[region][cut] = dict(filter(lambda kv: abs(1. - kv[1]) > threshold,
                #                                               self.scale_uncerts[region][cut].iteritems()))
                # #temporary fix
                # self.shape_uncerts[region][cut] = dict(filter(lambda kv: kv[1] != 0.,
                #                                               self.shape_uncerts[region][cut].iteritems()))
                # self.scale_uncerts[region][cut] = dict(filter(lambda kv: kv[1] != 0.,
                #                                               self.scale_uncerts[region][cut].iteritems()))

        for reg in self.ctrl_reg_shape_ylds.keys():
            if shape_uncert is not None:
                self.ctrl_reg_shape_ylds[reg] = dict(
                    filter(lambda kv: convert_name(kv[0]) in map(lambda i: i[0] if not isinstance(i, str) else i,
                                                                 shape_uncert),
                           self.ctrl_reg_shape_ylds[reg].iteritems()))
            if scale_uncert is not None:
                self.ctrl_reg_scale_ylds[reg] = dict(
                    filter(lambda kv: convert_name(kv[0]) in map(lambda i: i[0] if not isinstance(i, str) else i,
                                                                 scale_uncert),
                           self.ctrl_reg_scale_ylds[reg].iteritems()))

            # self.ctrl_reg_shape_ylds[reg] = dict(
            #     filter(lambda kv: abs(1. - kv[1]) > threshold or affects_signal_reg(kv[0], self.shape_uncerts),
            #            self.ctrl_reg_shape_ylds[reg].iteritems()))
            # self.ctrl_reg_scale_ylds[reg] = dict(
            #     filter(lambda kv: abs(1. - kv[1]) > threshold or affects_signal_reg(kv[0], self.scale_uncerts),
            #            self.ctrl_reg_scale_ylds[reg].iteritems()))

    @staticmethod
    def yld_sum(syst):
        return sum([s[0] for s in syst]),  # s[1]

    @staticmethod
    def product(syst, nom):
        return [syst[0] * nom[0], nom[1]]

    def merge_child_processes(self, samples, has_syst=True):
        self.generated_ylds = sum(map(lambda s: s.generated_ylds, samples))
        self.is_data = samples[0].is_data
        self.is_signal = samples[0].is_signal
        for region in samples[0].nominal_evt_yields.keys():
            if region not in self.nominal_evt_yields:
                self.nominal_evt_yields[region] = {}
            for cut in samples[0].nominal_evt_yields[region].keys():
                self.nominal_evt_yields[region][cut] = sum(map(lambda s: s.nominal_evt_yields[region][cut], samples))
        if has_syst:
            for region in samples[0].shape_uncerts.keys():
                for cut in samples[0].shape_uncerts[region].keys():
                    if region not in self.shape_uncerts:
                        self.shape_uncerts[region] = {}
                    self.shape_uncerts[region][cut] = {}
                    for syst in samples[0].shape_uncerts[region][cut].keys():
                        self.shape_uncerts[region][cut][syst] = sum(
                            map(lambda s: s.shape_uncerts[region][cut][syst], samples))
                for cut in samples[0].scale_uncerts[region].keys():
                    if region not in self.scale_uncerts:
                        self.scale_uncerts[region] = {}
                    self.scale_uncerts[region][cut] = {}
                    for syst in samples[0].scale_uncerts[region][cut].keys():
                        self.scale_uncerts[region][cut][syst] = sum(
                            map(lambda s: s.scale_uncerts[region][cut][syst], samples))

        for region in samples[0].ctrl_region_yields:
            self.ctrl_region_yields[region] = sum(map(lambda s: s.ctrl_region_yields[region], samples))
            if not has_syst:
                continue
            self.ctrl_reg_scale_ylds[region] = {}
            self.ctrl_reg_shape_ylds[region] = {}
            for syst in samples[0].ctrl_reg_scale_ylds[region].keys():
                self.ctrl_reg_scale_ylds[region][syst] = sum(
                    map(lambda s: s.ctrl_reg_scale_ylds[region][syst], samples))

            for syst in samples[0].ctrl_reg_shape_ylds[region].keys():
                self.ctrl_reg_shape_ylds[region][syst] = sum(
                    map(lambda s: s.ctrl_reg_shape_ylds[region][syst], samples))


class SampleStore(object):
    def __init__(self, **kwargs):
        self.samples = None
        self.xs_handle = XSHandle(kwargs["xs_config_file"])
        self.process_configs = kwargs['process_configs']
        self.lumi = OrderedDict([('mc16a', 36.1), ('mc16d', 43.6), ('mc16e', 59.93)])

    def get_all_thresholds(self):
        evt_ylds = self.samples[0].nominal_evt_yields
        return evt_ylds.values()[0].keys()

    def register_samples(self, samples):
        self.samples = samples

    def register_sample(self, sample):
        self.samples.append(sample)

    def filter_entries(self, shape_uncert=None, scale_uncert=None):
        map(lambda s: s.filter_systematics(shape_uncert, scale_uncert, threshold=0), self.samples)

    def scale_signal(self, factor):
        signal_samples = filter(lambda s: s.is_signal, self.samples)
        for sample in signal_samples:
            for region in sample.nominal_evt_yields.keys():
                for threshold in sample.nominal_evt_yields[region].keys():
                    sample.nominal_evt_yields[region][threshold] *= factor
            for reg in sample.ctrl_region_yields.keys():
                sample.ctrl_region_yields[reg] *= factor

    def scale_signal_by_pmg_xsec(self):
        signal_samples = filter(lambda s: s.is_signal, self.samples)
        for sample in signal_samples:
            factor = self.xs_handle.cross_sections[sample.name]
            self.scale_signal(factor)

    def merge_single_process(self):
        """
        Merge the samples of the same process in case of multiple files
        :return:
        :rtype:
        """
        sample_names = map(lambda sample: sample.name, self.samples)
        duplicates_names = set([x for x in sample_names if sample_names.count(x) > 1])
        for name in duplicates_names:
            duplicate_samples = filter(lambda s: s.name == name, self.samples)
            for s in duplicate_samples[1:]:
                duplicate_samples[0] += s
            summed_sample = sum(duplicate_samples)
            for s in duplicate_samples:
                self.samples.remove(s)
            self.samples.append(summed_sample)

    def apply_xsec_weight(self, signal_xsec=1.):
        for sample in self.samples:
            if sample.is_data:
                continue
            lumi = self.lumi
            if isinstance(self.lumi, OrderedDict):
                lumi = self.lumi[sample.process.mc_campaign]
            sample.apply_xsec_weight(lumi, self.xs_handle, signal_xsec)

    def calculate_uncertainties(self):
        for sample in self.samples:
            sample.calculate_relative_uncert()

    def merge_mc_campaigns(self):
        samples_to_remove = []
        for sample in self.samples:
            if sample.is_data:
                continue
            if '.mc16' not in sample.name:
                continue
            if sample in samples_to_remove:
                continue
            base_sample_name = sample.name.split('.')[0]
            merged_sample = Sample(base_sample_name, None)
            samples_to_merge = filter(lambda s: base_sample_name == s.name.split('.')[0], self.samples)
            merged_sample.merge_child_processes(samples_to_merge)  # , self.with_syst)
            self.samples.append(merged_sample)
            samples_to_remove += samples_to_merge
        for s in set(samples_to_remove):
            self.samples.remove(s)

    def merge_processes(self):
        samples_to_remove = []
        samples_to_merge = {}
        for sample in self.samples:
            process_config = find_process_config(sample.name, self.process_configs)
            if process_config is None:
                continue
            if process_config.type.lower() == 'signal':
                continue
            if process_config.name not in samples_to_merge:
                samples_to_merge[process_config.name] = [sample]
            else:
                samples_to_merge[process_config.name].append(sample)
        for process, samples in samples_to_merge.iteritems():
            if samples[0] in samples_to_remove:
                continue
            merged_sample = Sample(process, None)
            if len(samples) == 1:
                samples[0].name = process
            merged_sample.merge_child_processes(samples)  # , self.with_syst)
            self.samples.append(merged_sample)
            samples_to_remove += samples
        for s in set(samples_to_remove):
            self.samples.remove(s)

    def retrieve_ctrl_region_yields(self, regions=None):
        ctrl_region_ylds = {}
        for s in self.samples:
            for region, yld in s.ctrl_region_yields.iteritems():
                if regions is not None and region not in regions:
                    continue
                if region not in ctrl_region_ylds:
                    ctrl_region_ylds[region] = {s.name: yld}
                    continue
                ctrl_region_ylds[region][s.name] = yld
        return ctrl_region_ylds

    def retrieve_ctrl_region_syst(self, regions=None):
        systematics = {}
        for s in self.samples:
            for region in s.ctrl_reg_scale_ylds.keys():
                if regions is not None and region not in regions:
                    continue
                if region not in systematics:
                    systematics[region] = {s.name: s.ctrl_reg_scale_ylds[region]}
                else:
                    systematics[region][s.name] = s.ctrl_reg_scale_ylds[region]
                for syst_name, syst_yld in s.ctrl_reg_scale_ylds[region].iteritems():
                    systematics[region][s.name][syst_name] = syst_yld
                for syst_name, syst_yld in s.ctrl_reg_shape_ylds[region].iteritems():
                    systematics[region][s.name][syst_name] = syst_yld
        return systematics

    def get_signal_region_names(self, regions=None):
        """
        Get list of signal region names
        :param regions: optional filter argument to select subset of available regions in store
        :type regions: list
        :return: stored signal regions * filter
        :rtype: list
        """
        available_regions = self.samples[0].nominal_evt_yields.keys()
        if regions is not None:
            available_regions = filter(lambda r: r in regions, available_regions)
        return available_regions

    def retrieve_signal_ylds(self, sig_name, cut, region):
        """
        Get signal event yields in specific region for given cut
        :param sig_name: name of signal sample
        :type sig_name: str
        :param cut: threshold
        :type cut: int/float
        :param region: name of signal region
        :type region: str
        :return: signal yield
        :rtype: float
        """

        # todo: need some protections for missing signal, cut
        try:
            signal_sample = filter(lambda s: s.name == sig_name, self.samples)[0]
        except Exception as e:
            raise e
        return signal_sample.nominal_evt_yields[region][cut]

    def retrieve_signal_names(self):
        return map(lambda s: s.name, filter(lambda s: s.is_signal, self.samples))

    def retrieve_all_signal_ylds(self, cut, regions=None):
        signal_samples = filter(lambda s: s.is_signal, self.samples)
        return {reg: {s.name: s.nominal_evt_yields[reg][cut]
                      for s in signal_samples} for reg in self.get_signal_region_names(regions)}

    def retrieve_bkg_ylds(self, cut, region):
        bkg_samples = filter(lambda s: not s.is_data and not s.is_signal, self.samples)
        return {s.name: s.nominal_evt_yields[region][cut] for s in bkg_samples}

    def build_event_sums(self):
        for sample in self.samples:
            sample.build_sum()

    def retrieve_signal_region_syst(self, cut, sig_name, region):
        """
        Retrieve dictionary of systematic uncertainties for all MC and signal for a given cut
        :param cut: cut value
        :type cut: float
        :param sig_name: name of signal sample
        :type sig_name: str
        :return: dictionary of systematics (empty if systematics have not been enabled)
        :rtype: dict
        """

        def get_syst_dict(s):
            if len(s.shape_uncerts) == 0:
                return {}
            systematics = s.shape_uncerts[region][cut]
            for syst_name, syst_yld in s.scale_uncerts[region][cut].iteritems():
                systematics[syst_name] = syst_yld
            return systematics

        mc_samples = filter(lambda s: not s.is_data and (not s.is_signal or s.name == sig_name), self.samples)
        return {s.name: get_syst_dict(s) for s in mc_samples}

    def get_sample(self, sample_name):
        sample = filter(lambda s: s.name == sample_name, self.samples)
        if len(sample) == 1:
            return sample[0]
        return None


class CommonLimitOptimiser(BasePlotter):
    def __init__(self, **kwargs):
        kwargs.setdefault('syst_config', None)
        kwargs.setdefault('skip_fh_reading', True)
        super(CommonLimitOptimiser, self).__init__(**kwargs)
        self.signal_region_def = RegionBuilder(**yl.read_yaml(kwargs["sr_module_config"])["RegionBuilder"])
        self.control_region_defs = None
        if kwargs["cr_module_config"] is not None:
            self.control_region_defs = RegionBuilder(**yl.read_yaml(kwargs["cr_module_config"])["RegionBuilder"])
        self.data_converter = Root2NumpyConverter(["weight"])
        self.output_dir = soh.resolve_output_dir(output_dir=kwargs["output_dir"], sub_dir_name="limit")
        os.makedirs(self.output_dir)
        self.output_handle = OutputFileHandle(**kwargs)
        self.sample_store = SampleStore(xs_config_file=kwargs['xs_config_file'], process_configs=self.process_configs)
        self.weight_branch_list = ["weight"]
        self.run_syst = False
        self.shape_syst_config, self.scale_syst_config = None, None
        if kwargs['syst_config'] is not None:
            self.shape_syst_config, self.scale_syst_config = parse_syst_config(kwargs['syst_config'])
            for syst in self.scale_syst_config:
                self.weight_branch_list.append("weight_{:s}__1{:s}".format(syst[0], "up" if syst[1] == 1 else "down"))
            self.run_syst = True
        self.queue = kwargs['queue']

    def filter_empty_trees(self):
        """
        Remove empty trees from file list
        :return: nothing
        :rtype: None
        """

        def is_empty(file_handle, tree_name):
            """
            Retrieve entries from nominal tree
            :param file_handle: current file handle
            :type file_handle: FileHandle
            :param tree_name: name of input tree
            :type tree_name: string
            :return: Number of entries > 0
            :rtype: bool
            """
            if self.syst_tree_name is not None and file_handle.is_mc:
                tree_name = self.syst_tree_name
            return file_handle.get_object_by_name(tree_name, "Nominal").GetEntries() > 0

        self.file_handles = filter(lambda fh: is_empty(fh, self.tree_name), self.file_handles)

    def run(self):
        """
        Entry point starting execution
        :return: nothing
        :rtype: None
        """
        _logger.debug('Start running limits')
        yield_cache_file_name = "event_yields_nom.pkl"
        cr_config = build_region_info(self.control_region_defs)
        if self.read_cache is None:
            self.read_yields()
            if self.store_yields:
                with open(os.path.join(self.output_dir, yield_cache_file_name), 'w') as f:
                    dill.dump(self.sample_store, f)
        else:
            with open(self.read_cache, 'r') as f:
                self.sample_store = dill.load(f)
                if self.shape_syst_config is None:
                    self.shape_syst_config = {}
                if self.scale_syst_config is None:
                    self.scale_syst_config = {}
                self.sample_store.filter_entries(self.shape_syst_config, self.scale_syst_config)
            copy(self.read_cache, os.path.join(self.output_dir, yield_cache_file_name))
        if self.signal_scale is not None:
            self.sample_store.scale_signal(self.signal_scale)
        if self.signal_scale is not None:
            self.sample_store.scale_signal_by_pmg_xsec()
        self.sample_store.filter_entries()
        self.run_limits(cr_config)
        self.output_handle.write_and_close()


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


def write_config(args):
    # nice_labels = {'SR_mu_btag': 'SR \\mu b-tag',
    #                'SR_mu_bveto': 'SR \\mu b-veto',
    #                'SR_el_btag': 'SR el b-tag',
    #                'SR_el_bveto': 'SR el b-veto',
    #                'ZCR_el_bveto': 'ZCR el b-veto',
    #                'ZCR_el_btag': 'ZCR el b-tag',
    #                'ZCR_mu_bveto': 'ZCR \\mu b-veto',
    #                'ZCR_mu_btag': 'ZCR \\mu b-tag',
    #                }

    kwargs = args.kwargs
    kwargs.setdefault('disable_plots', False)
    make_dirs(os.path.join(args.output_dir, str(args.job_id)))
    make_plots = "FALSE" if kwargs['disable_plots'] else "TRUE"
    config_name = os.path.join(args.output_dir, str(args.job_id), 'trex_fitter.config')
    hist_path = os.path.join(args.output_dir, 'hists')
    make_dirs(hist_path)
    with open(config_name, 'w') as f:
        print('% --------------- %', file=f)
        print('% ---  JOB - -- %', file=f)
        print('% --------------- %', file=f)
        print('\n', file=f)

        print('Job: {:s}'.format('LQAnalysis'), file=f)
        print('\tCmeLabel: "13 TeV"', file=f)
        print('\tPOI: "mu_Sig"', file=f)
        print('\tReadFrom: HIST', file=f)
        print('\tHistoPath: {:s}'.format(os.path.join(hist_path, 'Nominal')), file=f)
        print('\tLabel: "LQAnalysis"', file=f)
        print('\tLumiLabel: "139 fb^{-1}"', file=f)
        print('\tPlotOptions: OVERSIG, NOXERR, NOENDERR', file=f)
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
        print('\tMergeUnderOverFlow: {:s}'.format(make_plots), file=f)
        # print >> f, '\tSystPruningShape: 0.005'
        # print >> f, '\tSystPruningNorm: 0.005'
        print('\tUseATLASRounding: TRUE', file=f)
        print('\tTableOptions: !STANDALONE', file=f)
        if kwargs['stat_only']:
            print('\tStatOnly: TRUE', file=f)
        print('\tOutputDir: {:s}'.format(os.path.join(args.output_dir, str(args.job_id))), file=f)
        print('\n', file=f)
        print('% --------------- %', file=f)
        print('% ---  FIT - -- %', file=f)
        print('% --------------- %', file=f)
        print('\n', file=f)

        print('Fit: "fit"', file=f)
        print('\tFitType: BONLY', file=f)
        print('\tFitRegion: CRONLY', file=f)
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
        for reg in kwargs['sig_yield'].keys():
            print('Region: "{:s}"'.format(reg), file=f)
            print('\tType: SIGNAL', file=f)
            print('\tVariableTitle: "m_{LQ} [GeV]"', file=f)
            print('\tLabel: "{:s}"'.format(reg), file=f)
            print('\tShortLabel: "{:s}"'.format(reg), file=f)
            print('\tHistoName: h_{:s}_{:.0f}'.format(reg, kwargs['mass_cut']), file=f)
            print('\tDataType: ASIMOV', file=f)
            print('\n', file=f)
            # print >> f,'Binning: 1000, 14000'

        norm_factors = []
        for reg, ctrl_reg_cfg in kwargs["ctrl_config"].iteritems():
            print('Region: "{:s}"'.format(reg), file=f)
            if ctrl_reg_cfg['is_norm_region']:
                print('\tType: CONTROL', file=f)
            elif ctrl_reg_cfg['is_val_region']:
                print('\tType: VALIDATION', file=f)
            print('\tVariableTitle: "m_{LQ} [GeV]"', file=f)
            print('\tLabel: "{:s}"'.format(reg), file=f)
            print('\tShortLabel: "{:s}"'.format(reg), file=f)
            print('\tHistoName: "h_{:s}"'.format(reg), file=f)
            print('', file=f)
            # print >> f,'Binning: 130, 150, 170, 190, 210, 230, 250'
            for bkg, norm_param in ctrl_reg_cfg['bgk_to_normalise'].iteritems():
                if 'norm_factor' not in norm_param:
                    continue
                norm_factors.append((norm_param['norm_factor'], bkg))
        norm_factors = set(norm_factors)

        print('% --------------- %', file=f)
        print('% --- SAMPLES - -- %', file=f)
        print('% --------------- %', file=f)

        print('% Normal samples', file=f)

        print('Sample: "Data"', file=f)
        print('\tTitle: "Data"', file=f)
        print('\tType: DATA', file=f)
        print('\tHistoFile: "data"', file=f)
        print('\n', file=f)

        print('Sample: "{:s}"'.format(kwargs['sig_name']), file=f)
        print('\tType: SIGNAL', file=f)
        print('\tTitle: "{:s}"'.format(kwargs['sig_name']), file=f)
        print('\tTexTitle: "$LQ$"', file=f)
        print('\tFillColor: 0', file=f)  # TODO
        print('\tLineColor: 2', file=f)  # TODO
        print('\tHistoFile: "{:s}"'.format(kwargs['sig_name']), file=f)
        print('\tNormFactor: "mu_Sig", 1, 0, 100', file=f)
        print('\tUseMCstat: TRUE', file=f)
        print('\n', file=f)

        for bkg in kwargs["bkg_yields"].values()[0].keys():
            print('Sample: "{:s}"'.format(bkg), file=f)
            print('\tType: BACKGROUND', file=f)
            print('\tTitle: "{:s}"'.format(bkg), file=f)
            print('\tTexTitle: "{:s}"'.format(bkg), file=f)
            print('\tFillColor: {:d}'.format(transform_color(kwargs['process_configs'][bkg].color)), file=f)
            print('\tLineColor: {:d}'.format(transform_color(kwargs['process_configs'][bkg].color)), file=f)
            print('\tHistoFile: "{:s}";'.format(bkg), file=f)
            print('\tSmooth: TRUE', file=f)
            print('\tUseMCstat: TRUE', file=f)
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
        print('\tOverallUp: 0.017', file=f)
        print('\tOverallDown: -0.017', file=f)
        print('\tSamples: Zjets, ttbar, Others, {:s}'.format(kwargs['sig_name']), file=f)
        print('\tSymmetrisation: TwoSided', file=f)
        print('\tSmoothing: 40', file=f)
        print('\n', file=f)

        print('% Weight systematics', file=f)

        paired_systematics = {}
        for reg in kwargs['sr_syst'].keys():
            for process, syst_set in kwargs['sr_syst'][reg].iteritems():
                for syst in syst_set.keys():
                    up_down = False
                    if '__1down' in syst and syst.replace('1down', '1up') in syst_set:
                        up_down = True
                        syst = syst.replace('__1down', '')
                    if '__1up' in syst and syst.replace('1up', '1down') in syst_set:
                        up_down = True
                        syst = syst.replace('__1up', '')

                    if up_down is False:
                        if syst == 'theory_envelop':
                            try:
                                paired_systematics[(syst, True)].append(process)
                            except KeyError:
                                paired_systematics[(syst, True)] = [process]
                            continue
                        else:
                            print('WEIRDD: ', syst, process, reg)
                            print(syst.replace('__1up', ''), syst.replace('up', 'down'),
                                  syst.replace('up', 'down') in kwargs['sr_syst'][reg][process])
                            print(kwargs['sr_syst'][reg][process].keys())
                    if (syst, up_down) not in paired_systematics:
                        paired_systematics[(syst, up_down)] = [process]
                        continue
                    if process in paired_systematics[(syst, up_down)]:
                        continue
                    paired_systematics[(syst, up_down)].append(process)

        for syst, processes in paired_systematics.iteritems():
            syst_name, up_down = syst
            print('Systematic: "{:s}"'.format(syst_name), file=f)
            print('\tTitle: "{:s}"'.format(syst_name), file=f)
            print('\tType: HISTO', file=f)
            print('\tSamples: {:s}'.format(','.join(processes)), file=f)
            print('\tRegions: all', file=f)
            if up_down:
                print('\tHistoPathUp: "{:s}"'.format(os.path.join(hist_path, syst_name + '__1up')), file=f)
                print('\tHistoPathDown: "{:s}"'.format(os.path.join(hist_path, syst_name + '__1down')), file=f)
            else:
                print("SYST NOT UP DOWN: ", syst_name)
            print('\n', file=f)
    print('Wrote config to ', config_name)


def get_hist(name, yld, unc):
    h = ROOT.TH1F('h_' + name, name, 1, 0., 1.)
    h.SetBinContent(1, yld)
    h.SetBinError(1, unc)
    return h


def dump2rootfile(hist_path, directory, name, hists):
    make_dirs(os.path.join(hist_path, directory))
    f = ROOT.TFile.Open(os.path.join(hist_path, directory, name + '.root'), 'RECREATE')
    f.cd()
    map(lambda h: h.Write(), hists)
    f.Close()


# def dump_yld2hist(args):
#     kwargs = args.kwargs
#     hist_path = os.path.join(args.output_dir, str(args.job_id), 'hists')
#
#     hists = []
#     for sr, sig_yld in kwargs['sig_yield'].iteritems():
#         yld, unc = sig_yld
#         fixed_signal_sf = 1.
#         if kwargs['fixed_signal'] is not None:
#             yld = kwargs['fixed_signal']
#             unc = kwargs['fixed_signal'] * kwargs['sig_yield'][sr][1] / kwargs['sig_yield'][sr][0]
#             fixed_signal_sf = kwargs['fixed_signal'] / kwargs['sig_yield'][sr][0]
#         hists.append(get_hist(sr, yld, unc))
#         for cr, yields in kwargs['control_regions'].iteritems():
#             hists.append(get_hist(cr, yields[kwargs['sig_name']][0] * fixed_signal_sf,
#                                   yields[kwargs['sig_name']][1] * fixed_signal_sf))
#     dump2rootfile(hist_path, 'Nominal', kwargs['sig_name'], hists)
#     for sr, bkgs in kwargs['bkg_yields'].iteritems():
#         for bkg, yields in bkgs.iteritems():
#             hists = []
#             hists.append(get_hist(sr, yields[0], yields[1]))
#             for cr, data in kwargs['control_regions'].iteritems():
#                 hists.append(get_hist(cr, data[bkg][0], data[bkg][1]))
#             dump2rootfile(hist_path, 'Nominal', bkg, hists)
#         data_hists = [get_hist(sr, 5., 0.)]
#         for cr, yields in kwargs['control_regions'].iteritems():
#             data_hists.append(get_hist(cr, yields['Data'][0], yields['Data'][1]))
#         dump2rootfile(hist_path, 'Nominal', 'data', data_hists)
#
#     for sr in kwargs['sr_syst'].keys():
#         for process in kwargs['sr_syst'][sr].keys():
#             if 'data' in process.lower():
#                 continue
#             try:
#                 nom = kwargs['bkg_yields'][sr][process]
#             except KeyError:
#                 if kwargs['fixed_signal'] is None:
#                     nom = sig_yld
#                 else:
#                     nom = (kwargs['fixed_signal'], kwargs['fixed_signal'])
#             extra_unc = {}
#             for unc, scale in kwargs['sr_syst'][sr][process].iteritems():
#                 hists = []
#                 if scale == 0. and process == 'ttbar':
#                     scale = 1.
#                 if 'JER' in unc:
#                     if '__1up' in unc and scale < 1.:
#                         scale = 2. - scale
#                     if '__1down' in unc and scale > 1.:
#                         scale = 2. - scale
#                 if unc != 'theory_envelop':
#                     hists.append(get_hist(sr, nom[0] * scale, nom[1] * scale))
#                 else:
#                     extra_unc[unc] = {sr: (nom, scale)}
#                 for cr in kwargs['ctrl_syst'].keys():
#                     fixed_signal_sf = 1.
#                     nom_cr = kwargs['control_regions'][cr][process]
#                     if process == kwargs['sig_name']:
#                         if kwargs['fixed_signal']:
#                             fixed_signal_sf = kwargs['fixed_signal'] / kwargs['sig_yield'][sr][0]
#                     nom_cr = kwargs['control_regions'][cr][process]
#                     if 'JER' in unc:
#                         if '__1up' in unc and scale < 1.:
#                             scale = 2. - scale
#                         if '__1down' in unc and scale > 1.:
#                             scale = 2. - scale
#                     cr_scale = kwargs['ctrl_syst'][cr][process][unc]
#                     if unc != 'theory_envelop':
#                         hists.append(get_hist(cr, nom_cr[0] * cr_scale * fixed_signal_sf,
#                                               nom_cr[1] * cr_scale * fixed_signal_sf))
#                     else:
#                         extra_unc[unc][cr] = (nom_cr, scale)
#                 dump2rootfile(hist_path, unc, process, hists)
#             for unc in extra_unc.keys():
#                 hists_up = []
#                 hists_down = []
#                 for reg, val in extra_unc[unc].iteritems():
#                     hists_up.append(get_hist(reg, val[0][0] * val[1], val[0][1] * val[1]))
#                     hists_down.append(get_hist(reg, val[0][0] * (2. - val[1]), val[0][1] * (2. - val[1])))
#                 dump2rootfile(hist_path, unc + '__1up', process, hists_up)
#                 dump2rootfile(hist_path, unc + '__1down', process, hists_down)


def dump_yld2hist(nominal_yields, systematics_yields, sig_names, fixed_signal, ctrl_reg_yields, ctrl_reg_syst,
                  output_dir):
    def symmetrise(syst_name, scale):
        if '__1up' in syst_name and scale < 1.:
            scale = 2. - scale
        if '__1down' in syst_name and scale > 1.:
            scale = 2. - scale
        return scale

    hist_path = os.path.join(output_dir, 'hists')
    sig_regions = []
    for process in nominal_yields.keys():
        hists = []
        fixed_signal_sf = 1.
        for k, yld in nominal_yields[process].iteritems():
            # usage of yld[0] for signal scaling is not really ideal
            yields = yld[0], yld[1]
            if process in sig_names and fixed_signal is not None:
                if yld[0] == 0:
                    _logger.warning('Found zero signal for {:s}'.format('_'.join(map(str, k))))
                    yields = 0., 0.
                    fixed_signal_sf = 0.
                else:
                    yields = fixed_signal, fixed_signal * yld[1] / yld[0]
                    fixed_signal_sf = fixed_signal / yld[0]
            sr_name = '{:s}_{:.0f}'.format(*k)
            sig_regions.append(sr_name)
            hists.append(get_hist(sr_name, *yields))
        for reg, yields in ctrl_reg_yields.iteritems():
            hists.append(get_hist(reg, yields[process][0] * fixed_signal_sf, yields[process][1] * fixed_signal_sf))
        dump2rootfile(hist_path, 'Nominal', process, hists)
    data_hists = []
    sig_regions = set(sig_regions)
    for sr in sig_regions:
        data_hists.append(get_hist(sr, 5., 0.))
    for cr, yields in ctrl_reg_yields.iteritems():
        data_hists.append(get_hist(cr, yields['Data'][0], yields['Data'][1]))
    dump2rootfile(hist_path, 'Nominal', 'data', data_hists)

    for syst_name in systematics_yields.keys():
        for process in systematics_yields[syst_name].keys():
            extra_unc = {}
            hists = []
            for k, scale in systematics_yields[syst_name][process].iteritems():
                fixed_signal_sf = 1.
                nom = nominal_yields[process][k]
                if process in sig_names and fixed_signal is not None:
                    if nom[0] == 0:
                        _logger.warning('Found zero signal for {:s}'.format('_'.join(map(str, k))))
                        fixed_signal_sf = 0.
                    else:
                        fixed_signal_sf = fixed_signal / nom[0]
                if 'JER' in syst_name:
                    if scale != 0. or process != 'ttbar':
                        scale = symmetrise(syst_name, scale)
                if scale == 0. and process == 'ttbar':
                    scale = 1.
                scale *= fixed_signal_sf
                yields = scale * nom[0], scale * nom[1]
                if syst_name != 'theory_envelop':
                    hists.append(get_hist("{:s}_{:.0f}".format(*k), *yields))
                else:
                    try:
                        extra_unc[syst_name]['{:s}_{:.0f}'.format(*k)] = (nom, scale)
                    except KeyError:
                        extra_unc[syst_name] = {'{:s}_{:.0f}'.format(*k): (nom, scale)}

            for reg in ctrl_reg_syst.keys():
                fixed_signal_sf = 1.
                if process in sig_names and fixed_signal is not None:
                    fixed_signal_sf = fixed_signal / yld[0]
                nom = ctrl_reg_yields[reg][process]
                scale = ctrl_reg_syst[reg][process][syst_name]
                if 'JER' in syst_name:
                    if scale != 0. or process != 'ttbar':
                        scale = symmetrise(syst_name, scale)
                if scale == 0. and process == 'ttbar':
                    scale = 1.
                scale *= fixed_signal_sf
                yields = scale * nom[0], scale * nom[1]
                if syst_name != 'theory_envelop':
                    hists.append(get_hist(reg, *yields))
                else:
                    extra_unc[syst_name][reg] = (nom, scale)
            dump2rootfile(hist_path, syst_name, process, hists)
            for unc in extra_unc.keys():
                hists_up = []
                hists_down = []
                for reg, val in extra_unc[unc].iteritems():
                    hists_up.append(get_hist(reg, val[0][0] * val[1], val[0][1] * val[1]))
                    hists_down.append(get_hist(reg, val[0][0] * (2. - val[1]), val[0][1] * (2. - val[1])))
                dump2rootfile(hist_path, unc + '__1up', process, hists_up)
                dump2rootfile(hist_path, unc + '__1down', process, hists_down)


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

    write_config(args)
    kwargs.setdefault('options', 'hwdflp')
    base_dir = os.path.abspath(os.path.join(os.path.basename(__file_path__), '../../../'))
    analysis_pkg_name = os.path.abspath(os.curdir).split('/')[-2]

    cfg_file = os.path.join(args.output_dir, str(args.job_id), 'trex_fitter.config')
    trf_cmd = '&& '.join(map(lambda o: 'trex-fitter {:s} {:s}'.format(o, cfg_file), kwargs['options']))
    os.system("""echo 'source $HOME/.bashrc && cd {:s} && source setup_python_ana.sh &&
        source /user/mmorgens/workarea/devarea/rel21/TRExFitter/setup.sh && {:s}  
        ' | 
        qsub -q {:s} -o {:s}.txt -e {:s}.err""".format(os.path.join(base_dir, analysis_pkg_name),  # noqa: E501, W291
                                                       trf_cmd,
                                                       args.kwargs['queue'],
                                                       os.path.join(args.output_dir, str(args.job_id), 'fit'),
                                                       os.path.join(args.output_dir, str(args.job_id), 'fit')))
