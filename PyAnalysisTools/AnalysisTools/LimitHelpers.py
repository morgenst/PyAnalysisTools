import ROOT
import os
import re
import PyAnalysisTools.PlottingUtils.PlottingTools as pt
import PyAnalysisTools.PlottingUtils.Formatting as fm
from PyAnalysisTools.PlottingUtils.PlotConfig import PlotConfig
from PyAnalysisTools.base.OutputHandle import OutputFileHandle
from PyAnalysisTools.base.YAMLHandle import YAMLLoader as yl


class LimitArgs(object):
    def __init__(self, output_dir, fit_mode, **kwargs):
        self.fit_mode = fit_mode
        self.output_dir = output_dir
        self.job_id = kwargs["jobid"]
        self.kwargs = kwargs


def build_region_info(control_region_defs):
    limit_region_info = {}
    for region in control_region_defs:
        limit_region_info[region.name] = {"is_norm_region": region.norm_region,
                                          "bgk_to_normalise": region.norm_backgrounds}

    return limit_region_info


def read_workspace_from_file(file_name, ws_name="w"):
    f = ROOT.TFile.Open(file_name)
    return f.Get(ws_name)


def get_fit_quality(file_name, ws_name="w", fr_name="RooExpandedFitResult_afterFit"):
    ws = read_workspace_from_file(file_name, ws_name)
    fit_result = ws.obj(fr_name)
    return fit_result.status(), fit_result.covQual()


def make_cross_section_limit_plot(data, lumi=80., ytitle=None):
    data.sort()
    if ytitle is None:
        ytitle = "95% CL U.L on #sigma [pb]"
    pc = PlotConfig(name="xsec_limit", ytitle=ytitle, xtitle="m [GeV]", draw="ap", logy=True, lumi=lumi,
                    watermark="Internal")
    graph = ROOT.TGraph(len(data))
    for i, item in enumerate(data):
        print item
        graph.SetPoint(i, item[0], item[1] * item[2]/(lumi*1000.))
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


class LimitPlotter(object):
    def __init__(self, output_handle):
        self.output_handle = output_handle

    def make_cross_section_limit_plot(self, limits, lumi):
        limits.sort(key=lambda li: li.mass)
        ytitle = "95% CL U.L on #sigma [pb]"
        pc = PlotConfig(name="xsec_limit", ytitle=ytitle, xtitle="m [GeV]", draw="ap", logy=True, lumi=lumi,
                        watermark="Internal")
        graph = ROOT.TGraph(len(limits))
        for i, limit in enumerate(limits):
            graph.SetPoint(i, limit.mass, limit.exp_limit)
        graph.SetName("xsec_limit")
        canvas = pt.plot_obj(graph, pc)
        fm.decorate_canvas(canvas, pc)
        self.output_handle.register_object(canvas)


class LimitScanAnalyser(object):
    """
    Class to analyse limit scan over mass range and mass cuts
    """
    def __init__(self, input_path, output_dir, scan_info=None):
        """
        Constructor
        :param input_path: input path containing calculated limits for each scan point
        :type input_path: string
        :param output_dir: directory where to store results
        :type output_dir: string
        :param scan_info: details on performed scan (mass, signal, etc.)
        :type scan_info: list
        """
        self.input_path = input_path
        self.output_handle = OutputFileHandle(output_dir=output_dir)
        self.plotter = LimitPlotter(self.output_handle)
        if scan_info is None:
            self.scan_info = yl.read_yaml(os.path.join(self.input_path, "scan_info.yml"), None)

    def parse_limits(self):
        parsed_data = []
        for scan in self.scan_info:
            analyser = LimitAnalyser(scan.output_dir, 'LQAnalysis')
            limit_info = analyser.analyse_limit(scan.kwargs['sig_name'])
            limit_info.add_info(mass_cut=scan.kwargs["mass_cut"],
                                mass=float(re.findall('\d{3,4}', scan.kwargs['sig_name'])[0]))
            parsed_data.append(limit_info)
        self.make_scan_plot(parsed_data)
        best_limits = self.find_best_limit(parsed_data)
        self.plotter.make_cross_section_limit_plot(best_limits, 80.)
        self.output_handle.write_and_close()

    @staticmethod
    def find_best_limit(limits):
        masses = set(map(lambda li: li.mass, limits))
        best_limits = []
        for mass in masses:
            mass_limits = filter(lambda li: li.mass == mass, limits)
            best_limits.append(min(mass_limits, key=lambda li: li.exp_limit))
        return best_limits

    def make_scan_plot(self, parsed_data):
        scanned_mass_cuts = sorted(list(set([li.mass_cut for li in parsed_data])))
        scanned_sig_masses = sorted(list(set([li.mass for li in parsed_data])))
        if len(scanned_sig_masses) > 1:
            min_mass_diff = min(
                [scanned_sig_masses[i + 1] - scanned_sig_masses[i] for i in range(len(scanned_sig_masses) - 1)]) / 2.
        else:
            min_mass_diff = 50
        if len(scanned_mass_cuts) > 1:
            mass_cut_offset = (scanned_mass_cuts[1] - scanned_mass_cuts[0]) / (len(scanned_sig_masses) - 1) / 2.
        else:
            mass_cut_offset = 50.

        hist = ROOT.TH2F("uppler_limit", "",
                         len(scanned_sig_masses),
                         scanned_sig_masses[0] - min_mass_diff, scanned_sig_masses[-1] + min_mass_diff,
                         len(scanned_mass_cuts),
                         scanned_mass_cuts[0] - mass_cut_offset, scanned_mass_cuts[-1] + mass_cut_offset)
        hist_fit_status = hist.Clone("fit_status")
        hist_fit_quality = hist.Clone("fit_quality")
        for limit_info in parsed_data:
            hist.Fill(limit_info.mass, limit_info.mass_cut, limit_info.exp_limit)
            hist_fit_status.Fill(limit_info.mass, limit_info.mass_cut, limit_info.fit_status+1)
            hist_fit_quality.Fill(limit_info.mass, limit_info.mass_cut, limit_info.fit_cov_quality)
        ROOT.gStyle.SetPalette(1)
        pc = PlotConfig(name="limit_scan_{:s}".format("test"), draw_option="COLZTEXT", xtitle="m_{LQ} [GeV]",
                        ytitle="m_{lq}^{max} cut [GeV]", ztitle="#mu_{sig}^{U.L. @ 95% CL}")
        pc_status = PlotConfig(name="limit_status_{:s}".format("test"), draw_option="COLZTEXT", xtitle="m_{LQ} [GeV]",
                               ytitle="m_{lq}^{max} cut [GeV]", ztitle="fit status + 1", zmin=-1.)
        pc_cov_quality = PlotConfig(name="limit_cov_quality_{:s}".format("test"), draw_option="COLZTEXT",
                                    xtitle="m_{LQ} [GeV]", ytitle="m_{lq}^{max} cut [GeV]", ztitle="fit cov quality")
        canvas = pt.plot_obj(hist, pc)
        self.output_handle.register_object(canvas)
        canvas_status = pt.plot_obj(hist_fit_status, pc_status)
        self.output_handle.register_object(canvas_status)
        canvas_quality = pt.plot_obj(hist_fit_quality, pc_cov_quality)
        self.output_handle.register_object(canvas_quality)

