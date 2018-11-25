import pickle
from copy import deepcopy
import numpy as np
import pandas as pd
import ROOT
import os
import re
import PyAnalysisTools.PlottingUtils.PlottingTools as pt
import PyAnalysisTools.PlottingUtils.Formatting as fm
from PyAnalysisTools.ROOTUtils.FileHandle import FileHandle
from PyAnalysisTools.AnalysisTools.XSHandle import XSHandle
from PyAnalysisTools.PlottingUtils.PlotConfig import PlotConfig, get_default_color_scheme
from PyAnalysisTools.base.OutputHandle import OutputFileHandle
from PyAnalysisTools.AnalysisTools.MLHelper import Root2NumpyConverter
from PyAnalysisTools.base.YAMLHandle import YAMLLoader as yl
from collections import OrderedDict
import dill
try:
    from tabulate.tabulate import tabulate
except ImportError:
    from tabulate import tabulate


class LimitArgs(object):
    def __init__(self, output_dir, fit_mode, **kwargs):
        kwargs.setdefault("ctrl_syst", None)
        self.fit_mode = fit_mode
        self.output_dir = output_dir
        self.job_id = kwargs["jobid"]
        self.sig_reg_name = kwargs["sig_reg_name"]
        self.kwargs = kwargs


def build_region_info(control_region_defs):
    limit_region_info = {}
    for region in control_region_defs:
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


def make_cross_section_limit_plot(data, plot_config):
    data.sort()
    if plot_config['ytitle'] is None:
        ytitle = "95% CL U.L on #sigma [pb]"
    pc = PlotConfig(name="xsec_limit", ytitle=ytitle, xtitle=plot_config['xtitle'], draw="ap", logy=True,
                    lumi=plot_config.get_lumi(), watermark=plot_config['watermark'])
    graph = ROOT.TGraph(len(data))
    for i, item in enumerate(data):
        graph.SetPoint(i, item[0], item[1] * item[2]/(plot_config.get_lumi() * 1000.))
    graph.SetName("xsec_limit")
    canvas = pt.plot_obj(graph, pc)
    fm.decorate_canvas(canvas, pc)
    return canvas


class LimitInfo(object):
    def __init__(self, **kwargs):
        self.add_info(**kwargs)

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
        exp_limit, exp_limit_up,  exp_limit_low = get_expected_limit(self.limit_fname,
                                                                     "hypo_{:s}".format(sig_name))
        self.limit_info.add_info(exp_limit=exp_limit, exp_limit_up=exp_limit_up,  exp_limit_low=exp_limit_low)
        return self.limit_info


class LimitAnalyserCL(object):
    def __init__(self, input_path):
        self.input_path = input_path
        self.limit_info = LimitInfo()
        self.converter = Root2NumpyConverter(['exp_upperlimit', 'exp_upperlimit_plus1', 'exp_upperlimit_plus2',
                                              'exp_upperlimit_minus1', 'exp_upperlimit_minus2', 'fit_status'])

    def analyse_limit(self):
        try:
            fh = FileHandle(file_name=os.path.join(self.input_path, 'asymptotics/test_BLIND_CL95.root'),
                            switch_off_process_name_analysis=True)
            tree = fh.get_object_by_name('stats')
            data = self.converter.convert_to_array(tree=tree)
            fit_status = data['fit_status']  # , fit_cov_quality = get_fit_quality(self.fit_fname)
            self.limit_info.add_info(fit_status=fit_status, fit_cov_quality=-1)
            self.limit_info.add_info(exp_limit=data['exp_upperlimit'], exp_limit_up=data['exp_upperlimit_plus1'],
                                     exp_limit_low=data['exp_upperlimit_minus1'])

        except ValueError:
            self.limit_info.add_info(fit_status=-1, fit_cov_quality=-1, exp_limit=-1, exp_limit_up=-1,
                                     exp_limit_low=-1)
        return self.limit_info


class LimitPlotter(object):
    def __init__(self, output_handle):
        self.output_handle = output_handle

    def make_cross_section_limit_plot(self, limits, plot_config, theory_xsec=None):
        """
        make cross section limit plot based on expected limits as function of mass hypothesis

        :param limits: list of LimitInfo objects
        :type limits: list
        :param plot_config: dictionary containing plot configuration, such as lumi, titles, etc
        :type plot_config: Ordered dict/dict
        :param theory_xsec: theory predictions
        :type theory_xsec: TGraph (default = None)
        :return: None
        :rtype: None
        """
        limits.sort(key=lambda li: li.mass)
        ytitle = "95% CL U.L on #sigma [pb]"
        pc = PlotConfig(name="xsec_limit", ytitle=ytitle, xtitle=plot_config['xtitle'], draw="pLX", logy=True,
                        lumi=plot_config['lumi'], watermark=plot_config['watermark'], ymin=float(1e-7),
                        ymax=float(1e-2), )
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
        for i, limit in enumerate(limits):
            graph.SetPoint(i, limit.mass, limit.exp_limit)
            graph_1sigma.SetPoint(i, limit.mass, limit.exp_limit)
            graph_2sigma.SetPoint(i, limit.mass, limit.exp_limit)
            graph_1sigma.SetPointEYhigh(i, limit.exp_limit_up - limit.exp_limit)
            graph_1sigma.SetPointEYlow(i, limit.exp_limit - limit.exp_limit_low)
            graph_2sigma.SetPointEYhigh(i, 2. * (limit.exp_limit_up - limit.exp_limit))
            graph_2sigma.SetPointEYlow(i, 2. * (limit.exp_limit - limit.exp_limit_low))
        if theory_xsec is not None:
            graph_theory = ROOT.TGraph(len(limits))
            for i, mass in enumerate(sorted(map(lambda l: l.mass,limits))):
                theory_xsec = filter(lambda xs: xs[0] == mass, theory_xsec)[0]
                graph_theory.SetPoint(i, mass, theory_xsec[-1])
            limits.sort(key=lambda li: li.mass)
        graph_2sigma.SetName('xsec_limit')
        canvas = pt.plot_obj(graph_2sigma, pc_2sigma)
        pt.add_graph_to_canvas(canvas, graph_1sigma, pc_1sigma)
        pt.add_graph_to_canvas(canvas, graph, pc)
        labels = ['expected limit', '#pm 1#sigma', '#pm 2#sigma']
        legend_format = ['PL', 'F', 'F']
        plot_objects = [graph, graph_1sigma, graph_2sigma]
        if theory_xsec is not None:
            pc_theory = deepcopy(pc)
            pc_theory.draw = 'l'
            pt.add_graph_to_canvas(canvas, graph_theory, pc_theory)
            labels.append("Theory")
            legend_format.append("L")
            plot_objects.append(graph_theory)
        fm.decorate_canvas(canvas, pc)
        fm.add_legend_to_canvas(canvas, plot_objects=plot_objects, labels=labels, format=legend_format)
        self.output_handle.register_object(canvas)

    def make_limit_plot_plane(self, limits, plot_config, xsec, sig_name):
        def find_excluded_lambda(mass, excl_limit):
            xsecs = filter(lambda xs: xs[0] == mass, xsec)
            xsecs.sort(key=lambda i: i[-1])
            try:
                return filter(lambda i: i[-1] > excl_limit, xsecs)[0][1]
            except IndexError:
                return 1.
        sig_type = re.split(r'(\d+)', sig_name)[0]
        excl_lambdas = [find_excluded_lambda(limit.mass, limit.exp_limit) for limit in limits]
        graph = ROOT.TGraph(len(excl_lambdas))
        for i, limit in enumerate(limits):
            graph.SetPoint(i, limit.mass, excl_lambdas[i])
        pc = PlotConfig(name='limit_contour', watermark=plot_config['watermark'], ymax=1.2, ymin=0., draw_option='AL',
                        xtitle=plot_config['xtitle'], ytitle=plot_config['ytitle'], logy=False,
                        lumi=plot_config['lumi'])
        canvas = pt.plot_graph(graph, pc)
        fm.decorate_canvas(canvas, pc)
        fm.add_legend_to_canvas(canvas, labels=['95% CL U.L'])
        self.output_handle.register_object(canvas)


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
        self.output_handle = OutputFileHandle(output_dir=kwargs['output_dir'])
        self.plotter = LimitPlotter(self.output_handle)
        self.xsec_handle = XSHandle("config/common/dataset_info_lq_new.yml")
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
            #analyser = LimitAnalyser(scan.output_dir, self.analysis_name)
            analyser = LimitAnalyserCL(os.path.join(scan.output_dir, 'limits', str(scan.kwargs['jobid'])))
            try:
                limit_info = analyser.analyse_limit()#scan.kwargs['sig_name'])
            except ReferenceError:
                print "Could not find info for scan ", scan
                continue
            limit_info.sig_name = scan.kwargs['sig_name']
            mass = float(re.findall('\d{3,4}', scan.kwargs['sig_name'])[0])
            self.theory_xsec[mass] = None
            limit_info.add_info(mass_cut=scan.kwargs["mass_cut"],
                                mass=mass)
            parsed_data.append(limit_info)
            #self.parse_prefit_yields(scan, mass)
        self.make_scan_plot(parsed_data, self.plot_config)
        #self.plot_prefit_yields()
        best_limits = self.find_best_limit(parsed_data)
        self.tabulate_limits(best_limits)
        theory_xsec = None
        if self.xsec_map is not None:
            # TODO: needs proper implementation
            self.plotter.make_limit_plot_plane(best_limits, self.plot_config, self.xsec_map['LQed'],
                                               scan.kwargs['sig_name'])
            theory_xsec = filter(lambda l: l[1] == 1.0, self.xsec_map['LQed'])
        self.plotter.make_cross_section_limit_plot(best_limits, self.plot_config, theory_xsec)
        self.output_handle.write_and_close()

    def tabulate_limits(self, limits):
        limits.sort(key=lambda l: l.mass)
        with open(os.path.join(self.input_path, 'event_yields_nominal.yml'), 'r') as f:
            event_yields = dill.load(f)
        data = []
        ordering = ['ttbar']
        for limit in limits:
            data_mass_point = [limit.mass, limit.mass_cut, limit.exp_limit]
            prefit_ylds_bkg = event_yields.retrieve_bkg_ylds(limit.mass_cut)
            prefit_ylds_sig = event_yields.retrieve_signal_ylds(limit.sig_name, limit.mass_cut)
            data_mass_point.append(prefit_ylds_sig * limit.exp_limit)
            for process in ordering:
                data_mass_point.append(prefit_ylds_bkg[process])
            data.append(data_mass_point)

        headers = ['mass', 'mass_cut', 'UL [pb]', 'Signal'] + ordering
        print tabulate(data, headers=headers)


    @staticmethod
    def find_best_limit(limits):
        limits = filter(lambda li: li.exp_limit > 0., limits)
        masses = set(map(lambda li: li.mass, limits))
        best_limits = []
        for mass in masses:
            mass_limits = filter(lambda li: li.mass == mass, limits)
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
                    print "Did not fine yields for {:s} and LQ mass {:.0f} and cut {:.0f}".format(process, mass, cut)
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
            fm.add_legend_to_canvas(canvas_vs_mass, labels=ordering+["signal"])
            fm.add_legend_to_canvas(canvas_vs_cut, labels=ordering+["signal"])
            fm.add_legend_to_canvas(canvas_vs_mass_log, labels=ordering+["signal"])
            fm.add_legend_to_canvas(canvas_vs_cut_log, labels=ordering+["signal"])
            self.output_handle.register_object(canvas_vs_cut)
            self.output_handle.register_object(canvas_vs_mass)
            self.output_handle.register_object(canvas_vs_cut_log)
            self.output_handle.register_object(canvas_vs_mass_log)

    def make_scan_plot(self, parsed_data, plot_config):
        self.scanned_mass_cuts = sorted(list(set([li.mass_cut for li in parsed_data])))
        self.scanned_sig_masses = sorted(list(set([li.mass for li in parsed_data])))
        if len(self.scanned_sig_masses) > 1:
            self.min_mass_diff = min(
                [self.scanned_sig_masses[i + 1] - self.scanned_sig_masses[i] for i in range(len(self.scanned_sig_masses) - 1)]) / 2.
        else:
            self.min_mass_diff = 50
        if len(self.scanned_mass_cuts) > 1:
            self.mass_cut_offset = (self.scanned_mass_cuts[1] - self.scanned_mass_cuts[0])  / 2.
        else:
            self.mass_cut_offset = 50.

        self.sig_mass_binning = [int((self.scanned_sig_masses[-1] - self.scanned_sig_masses[0]) / (self.min_mass_diff * 2)) + 1,
                                 self.scanned_sig_masses[0] - self.min_mass_diff,
                                 self.scanned_sig_masses[-1] + self.min_mass_diff]
        self.scan_mass_binning = [int((self.scanned_mass_cuts[-1] - self.scanned_mass_cuts[0]) / (self.mass_cut_offset * 2)) + 1,
                                  self.scanned_mass_cuts[0] - self.mass_cut_offset,
                                  self.scanned_mass_cuts[-1] + self.mass_cut_offset]

        hist = ROOT.TH2F("upper_limit", "", *(self.sig_mass_binning+self.scan_mass_binning))
        hist_fit_status = hist.Clone("fit_status")
        hist_fit_quality = hist.Clone("fit_quality")
        for limit_info in parsed_data:
            hist.Fill(limit_info.mass, limit_info.mass_cut, limit_info.exp_limit * 1000.)
            hist_fit_status.Fill(limit_info.mass, limit_info.mass_cut, limit_info.fit_status+1)
            hist_fit_quality.Fill(limit_info.mass, limit_info.mass_cut, limit_info.fit_cov_quality)
        ROOT.gStyle.SetPalette(1)
        ROOT.gStyle.SetPaintTextFormat(".2g")
        pc = PlotConfig(name="limit_scan_{:s}".format(self.sig_reg_name), draw_option="COLZTEXT",
                        xtitle=plot_config['xtitle'], ytitle=plot_config['ytitle'], ztitle="95 \% CL U.L. #sigma [fb]")
        pc_status = PlotConfig(name="limit_status_{:s}".format(self.sig_reg_name), draw_option="COLZTEXT",
                               xtitle=plot_config['xtitle'], ytitle=plot_config['ytitle'],
                               ztitle="fit status + 1", zmin=-1.)
        pc_cov_quality = PlotConfig(name="limit_cov_quality_{:s}".format(self.sig_reg_name), draw_option="COLZTEXT",
                                    xtitle=plot_config['xtitle'], ytitle=plot_config['ytitle'],
                                    ztitle="fit cov quality")
        canvas = pt.plot_obj(hist, pc)
        self.output_handle.register_object(canvas)
        canvas_status = pt.plot_obj(hist_fit_status, pc_status)
        self.output_handle.register_object(canvas_status)
        canvas_quality = pt.plot_obj(hist_fit_quality, pc_cov_quality)
        self.output_handle.register_object(canvas_quality)


def sum_ylds(ylds):
    ylds.dtype = np.float64
    if True in pd.isnull(ylds):
        print "found NONE"
    ylds = ylds[~pd.isnull(ylds)]
    return np.sum(ylds)


class Sample(object):
    def __init__(self, name, gen_ylds):
        self.name = name
        self.generated_ylds = gen_ylds
        self.nominal_evt_yields = {}
        self.shape_uncerts = {}
        self.scale_uncerts = {}
        self.ctrl_region_yields = {}
        self.ctrl_reg_scale_ylds = {}
        self.ctrl_reg_shape_ylds = {}
        self.is_data = 'data' in name
        self.is_signal = False

    def add_signal_region_yields(self, cut, nom_yields, shape_uncerts=None):
        for syst in nom_yields.keys():
            if syst == 'weight':
                continue
            nom_yields[syst] *= nom_yields['weight']
        self.nominal_evt_yields[cut] = sum_ylds(nom_yields['weight'])
        if shape_uncerts is not None:
            self.shape_uncerts[cut] = {syst: sum_ylds(yld) for syst, yld in shape_uncerts.iteritems()}
        self.scale_uncerts[cut] = {syst: sum_ylds(yld) for syst, yld in nom_yields.iteritems() if not syst == 'weight'}

    def add_ctrl_region(self, region_name, nominal_evt_yields, shape_uncert_yields=None):
        for syst in nominal_evt_yields.keys():
            if syst == 'weight':
                continue
                nominal_evt_yields[syst] *= nominal_evt_yields['weight']

        self.ctrl_reg_scale_ylds[region_name] = {syst: sum_ylds(yld) for syst, yld in nominal_evt_yields.iteritems() if not syst == 'weight'}
        if shape_uncert_yields is not None:
            self.ctrl_reg_shape_ylds[region_name] = {syst: sum_ylds(yld) for syst, yld in shape_uncert_yields.iteritems()}
        self.ctrl_region_yields[region_name] = sum_ylds(nominal_evt_yields['weight'])

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
        for cut in self.shape_uncerts.keys():
            for syst, yld in self.shape_uncerts[cut].iteritems():
                self.shape_uncerts[cut][syst] = yld / self.nominal_evt_yields[cut]
            for syst, yld in self.scale_uncerts[cut].iteritems():
                self.scale_uncerts[cut][syst] = yld / self.nominal_evt_yields[cut]
        for region in self.ctrl_region_yields.keys():
            ctrl_nom_ylds = self.ctrl_region_yields[region]
            for syst, yld in self.ctrl_reg_scale_ylds[region].iteritems():
                self.ctrl_reg_scale_ylds[region][syst] = yld / ctrl_nom_ylds
            for syst, yld in self.ctrl_reg_shape_ylds[region].iteritems():
                self.ctrl_reg_shape_ylds[region][syst] = yld / ctrl_nom_ylds

    def remove_unused(self):
        self.remove_low_systematics()

    def apply_xsec_weight(self, lumi, xs_handle):
        if self.is_data:
            return
        if self.is_signal:
            weight = xs_handle.get_lumi_scale_factor(self.name.split('.')[0], lumi, self.generated_ylds, fixed_xsec=1.)
        else:
            weight = xs_handle.get_lumi_scale_factor(self.name.split('.')[0], lumi, self.generated_ylds)
        for cut in self.nominal_evt_yields.keys():
            self.nominal_evt_yields[cut] *= weight
        for region in self.ctrl_region_yields.keys():
            self.ctrl_region_yields[region] *= weight

    def __add__(self, other):
        self.generated_ylds += other.generated_ylds
        for cut in self.nominal_evt_yields.keys():
            self.nominal_evt_yields[cut] += other.nominal_evt_yields[cut]
        for cut in self.shape_uncerts.keys():
            for syst in self.shape_uncerts[cut].keys():
                self.shape_uncerts[cut][syst] += other.shape_uncerts[cut][syst]
        for cut in self.scale_uncerts.keys():
            for syst in self.scale_uncerts[cut].keys():
                self.scale_uncerts[cut][syst] += other.scale_uncerts[cut][syst]
        for region in self.ctrl_region_yields:
            self.ctrl_region_yields[region] += other.ctrl_region_yields[region]
            for syst in self.ctrl_reg_scale_ylds[region].keys():
                self.ctrl_reg_scale_ylds[region][syst] += other.ctrl_reg_scale_ylds[region][syst]
            for syst in self.ctrl_reg_shape_ylds[region].keys():
                self.ctrl_reg_shape_ylds[region][syst] += other.ctrl_reg_shape_ylds[region][syst]
        return self

    def __radd__(self, other):
        if other == 0:
            return self
        if other is None:
            print 'Found none, which is surprising'
            return self
        return self.__add__(other)

    def filter_systematics(self):
        for cut in self.shape_uncerts.keys():
            self.shape_uncerts[cut] = dict(filter(lambda kv: abs(1.-kv[1]) > 0.01,
                                                  self.shape_uncerts[cut].iteritems()))
            self.scale_uncerts[cut] = dict(filter(lambda kv: abs(1.-kv[1]) > 0.01,
                                                  self.scale_uncerts[cut].iteritems()))
        for reg in self.ctrl_reg_shape_ylds.keys():
            self.ctrl_reg_shape_ylds[reg] = dict(filter(lambda kv: abs(1.-kv[1]) > 0.01,
                                                        self.ctrl_reg_shape_ylds[reg].iteritems()))
            self.ctrl_reg_scale_ylds[reg] = dict(filter(lambda kv: abs(1.-kv[1]) > 0.01,
                                                        self.ctrl_reg_scale_ylds[reg].iteritems()))

    def merge_child_processes(self, samples, has_syst=True):
        self.generated_ylds = sum(map(lambda s: s.generated_ylds, samples))
        self.is_data = samples[0].is_data
        self.is_signal = samples[0].is_signal
        for cut in samples[0].nominal_evt_yields.keys():
            self.nominal_evt_yields[cut] = sum(map(lambda s: s.nominal_evt_yields[cut], samples))
        if not has_syst:
            return
        for cut in samples[0].shape_uncerts.keys():
            self.shape_uncerts[cut] = {}
            for syst in samples[0].shape_uncerts[cut].keys():
                total_uncert = sum(map(lambda s: s.shape_uncerts[cut][syst] * s.nominal_evt_yields[cut], samples)) / \
                               self.nominal_evt_yields[cut]
                self.shape_uncerts[cut][syst] = total_uncert
        for cut in samples[0].scale_uncerts.keys():
            self.scale_uncerts[cut] = {}
            for syst in samples[0].scale_uncerts[cut].keys():
                total_uncert = sum(map(lambda s: s.scale_uncerts[cut][syst] * s.nominal_evt_yields[cut], samples)) / \
                               self.nominal_evt_yields[cut]
                self.scale_uncerts[cut][syst] = total_uncert

        for region in samples[0].ctrl_region_yields:
            self.ctrl_region_yields[region] = sum(map(lambda s: s.ctrl_region_yields[region], samples))
            self.ctrl_reg_scale_ylds[region] = {}
            self.ctrl_reg_shape_ylds[region] = {}
            for syst in samples[0].ctrl_reg_scale_ylds[region].keys():
                total_uncert = sum(
                    map(lambda s: s.ctrl_reg_scale_ylds[region][syst] * s.ctrl_region_yields[region], samples)) / \
                               self.ctrl_region_yields[region]
                self.ctrl_reg_scale_ylds[region][syst] = total_uncert

            for syst in samples[0].ctrl_reg_shape_ylds[region].keys():
                total_uncert = sum(
                    map(lambda s: s.ctrl_reg_shape_ylds[region][syst] * s.ctrl_region_yields[region], samples)) / \
                               self.ctrl_region_yields[region]
                self.ctrl_reg_shape_ylds[region][syst] = total_uncert


class SampleStore(object):
    def __init__(self, **kwargs):
        self.samples = None
        self.xs_handle = XSHandle(kwargs["xs_config_file"])
        self.process_configs = kwargs['process_configs']
        self.lumi = OrderedDict([('mc16a', 36.1), ('mc16d', 43.6), ('mc16e', 47.)])
        self.with_syst = False

    def register_samples(self, samples):
        self.samples = samples

    def filter_entries(self):
        map(lambda s: s.filter_systematics(), self.samples)

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
            for s in duplicate_samples:
                self.samples.remove(s)

    def apply_xsec_weight(self):
        for sample in self.samples:
            if sample.is_data:
                return
            lumi = self.lumi
            if isinstance(self.lumi, OrderedDict):
                lumi = self.lumi[sample.name.split('.')[-1]]
            sample.apply_xsec_weight(lumi, self.xs_handle)

    def calculate_uncertainties(self):
        for sample in self.samples:
            sample.calculate_relative_uncert()

    def merge_mc_campaigns(self):
        for sample in self.samples:
            if sample.is_data:
                continue
            if not '.mc16' in sample.name:
                continue
            base_sample_name = sample.name.split('.')[0]
            merged_sample = Sample(base_sample_name, None)
            samples_to_merge = filter(lambda s: base_sample_name == s.name.split('.')[0], self.samples)
            merged_sample.merge_child_processes(samples_to_merge, self.with_syst)
            self.samples.append(merged_sample)
            for s in samples_to_merge:
                self.samples.remove(s)

    def merge_processes(self):
        for process, process_config in self.process_configs.iteritems():
            if not hasattr(process_config, "subprocesses"):
                continue
            if process_config.type.lower() == "signal":
                continue
            samples_to_merge = filter(lambda sample: sample.name in process_config.subprocesses, self.samples)
            if len(samples_to_merge) == 0:
                continue
            if len(samples_to_merge) == 1:
                samples_to_merge[0].name = process
            merged_sample = Sample(process, None)
            merged_sample.merge_child_processes(samples_to_merge, self.with_syst)
            self.samples.append(merged_sample)
            for s in samples_to_merge:
                self.samples.remove(s)

    def retrieve_ctrl_region_yields(self):
        ctrl_region_ylds = {}
        for s in self.samples:
            for region, yld in s.ctrl_region_yields.iteritems():
                if region not in ctrl_region_ylds:
                    ctrl_region_ylds[region] = {s.name: yld}
                    continue
                ctrl_region_ylds[region][s.name] = yld
        return ctrl_region_ylds

    def retrieve_ctrl_region_syst(self):
        systematics = {}
        for s in self.samples:
            for region in s.ctrl_reg_scale_ylds.keys():
                if region not in systematics:
                    systematics[region] = {s.name: s.ctrl_reg_scale_ylds[region]}
                else:
                    systematics[region][s.name] = s.ctrl_reg_scale_ylds[region]
                for syst_name, syst_yld in s.ctrl_reg_scale_ylds[region].iteritems():
                    systematics[region][s.name][syst_name] = syst_yld
        return systematics

    def retrieve_signal_ylds(self, sig_name, cut):
        #todo: need some protections for missing signal, cut
        signal_sample = filter(lambda s: s.name == sig_name, self.samples)[0]
        return signal_sample.nominal_evt_yields[cut]

    def retrieve_all_signal_ylds(self, cut):
        signal_samples = filter(lambda s: s.is_signal, self.samples)
        return {s.name: s.nominal_evt_yields[cut] for s in signal_samples}

    def retrieve_bkg_ylds(self, cut):
        bkg_samples = filter(lambda s: not s.is_data and not s.is_signal, self.samples)
        return {s.name: s.nominal_evt_yields[cut] for s in bkg_samples}

    def retrieve_signal_region_syst(self, cut, sig_name):
        def get_syst_dict(s):
            systematics = s.shape_uncerts[cut]
            for syst_name, syst_yld in s.scale_uncerts[cut].iteritems():
                systematics[syst_name] = syst_yld
            return systematics
        mc_samples = filter(lambda s: not s.is_data and (not s.is_signal or s.name == sig_name), self.samples)
        return {s.name: get_syst_dict(s) for s in mc_samples}

