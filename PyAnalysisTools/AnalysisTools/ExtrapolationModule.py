from PyAnalysisTools.PlottingUtils.PlotConfig import find_process_config
from PyAnalysisTools.base import _logger
from PyAnalysisTools.ROOTUtils.FileHandle import FileHandle
from PyAnalysisTools.AnalysisTools.LimitHelpers import Yield
import ROOT


class ExtrapolationModule(object):
    def __init__(self, **kwargs):
        kwargs.setdefault('input_hist_file', None)
        kwargs.setdefault('qcd_mode', False)
        _logger.debug('Initialising ExtrapolationModule')
        self.stich_points = {}
        self.functions = {}
        if kwargs['input_hist_file'] is None:
            self.build_functions(**kwargs)
        else:
            self.read_histograms(**kwargs)
        self.qcd_mode = kwargs['qcd_mode']
        self.type = "DataModifier"
        self.tag = 'top' if not self.qcd_mode else 'qcd'

    def build_functions(self, **kwargs):
        def build_fct(name, params):
            fcts = [kwargs['functions'][name]] * 3
            for i, param in enumerate(params):
                param = eval(param)
                fcts[0] = fcts[0].replace('[{:d}]'.format(i), str(param[0]))
                fcts[1] = fcts[1].replace('[{:d}]'.format(i), str(param[0] - param[1]))
                fcts[2] = fcts[2].replace('[{:d}]'.format(i), str(param[0] + param[1]))
            return map(lambda f: ROOT.TF1("", f), fcts)

        self.functions = {}
        for reg_name, cfg in kwargs.iteritems():
            if reg_name == 'functions':
                continue
            self.functions[reg_name] = build_fct(cfg['func'], cfg['params'])
            self.stich_points[reg_name] = cfg['stitch']

    def read_histograms(self, **kwargs):
        def find_stitch_point():
            i = 0
            while h.GetBinContent(i) == 0.:
                i += 1
            return h.GetXaxis().GetBinLowEdge(i)

        fh = FileHandle(file_name=kwargs['input_hist_file'])
        self.histograms = fh.get_objects_by_type('TH1')
        for h in self.histograms:
            if 'uncert' in h.GetName():
                continue
            self.stich_points[h.GetName()] = max(300., find_stitch_point())
        map(lambda h: h.SetDirectory(0), self.histograms)

    def get_stitch_point(self, region):
        return self.stich_points[region]

    def get_extrapolated_bin_content(self, region, xmin, xmax=None, lumi=1.):
        uncert = 0
        if xmax is None:
            xmax = 10000000.
        if xmin < self.stich_points[region]:
            return None
        if region in self.functions:
            new_yield = self.functions[region][0][0].Integral(xmin, xmax)
        elif len(self.histograms) > 0:
            h = filter(lambda h: h.GetName() == region, self.histograms)[0]
            new_yield = h.Integral(h.FindBin(xmin), -1)
            huncert = filter(lambda h: h.GetName() == region + '_uncert', self.histograms)[0].Clone()
            huncert.Multiply(h)
            uncert = huncert.Integral(huncert.FindBin(xmin), -1)
        else:
            _logger.error('Could not find information to extrapolate top')
            exit(-1)
        new_yield *= lumi
        _logger.debug('RESET bin content: {:.1f} {:.1f} {:.1f} {:.1f}'.format(xmin, xmax, new_yield, uncert))
        return new_yield, uncert

    # def execute(self, histograms):
    #     top_hist = histograms['ttbar']
    #     region = [r for r in self.functions.keys() if r in top_hist.GetName()][0]
    #     _logger.debug('Running top extrapolation in region {:s}'.format(region))
    #     for i in range(top_hist.GetNbinsX() + 1):
    #         bin_content = self.get_extrapolated_bin_content(region, top_hist.GetXaxis().GetBinLowEdge(i),
    #                                                         top_hist.GetXaxis().GetBinUpEdge(i), 139.)
    #         _logger.debug("old yield: {:.2f} new yield: {:.2f}".format(top_hist.GetBinContent(i), bin_content))
    #         if bin_content is None:
    #             continue
    #         top_hist.SetBinContent(i, bin_content)

    def execute_top(self, histograms, output_handle):
        top_hist = histograms['ttbar']
        #top_uncert_hist_down = top_hist.Clone('top_extrapol_unc__1down')
        region = [r for r in self.stich_points.keys() if r in top_hist.GetName()][0]
        top_uncert = top_hist.Clone('{:s}_lq_mass_max_ttbar_top_extrapol_unc'.format(region))
        _logger.debug('Running top extrapolation in region {:s}'.format(region))
        for i in range(top_hist.GetNbinsX() + 1):
            bin_content = self.get_extrapolated_bin_content(region, top_hist.GetXaxis().GetBinLowEdge(i),
                                                            top_hist.GetXaxis().GetBinUpEdge(i), 139.)
            if bin_content is None:
                continue
            _logger.debug("old yield: {:.2f} new yield: {:.2f}".format(top_hist.GetBinContent(i), bin_content[0]))
            top_hist.SetBinContent(i, bin_content[0])
            top_uncert.SetBinContent(i, bin_content[1])
        output_handle.register_object(top_uncert)

    def execute_qcd(self, histograms, output_handle):
        region = [r for r in self.stich_points.keys() if r in histograms.values()[0].GetName()][0]
        h_qcd = histograms.values()[0].Clone(region + "_lq_mass_max_QCD")
        for i in range(h_qcd.GetNbinsX() + 1):
            bin_content = self.get_extrapolated_bin_content(region, h_qcd.GetXaxis().GetBinLowEdge(i),
                                                            h_qcd.GetXaxis().GetBinUpEdge(i), 1.)
            if bin_content is None:
                continue
            h_qcd.SetBinContent(i, bin_content[0])
        h_unc = filter(lambda h: h.GetName() == region + '_uncert', self.histograms)[0].Clone()
        h_unc.SetName('{:s}_lq_mass_max_QCD_qcd_uncert'.format(region))
        output_handle.register_object(h_qcd)
        output_handle.register_object(h_unc)

    def execute(self, histograms, output_handle):
        if not self.qcd_mode:
            self.execute_top(histograms, output_handle)
        else:
            self.execute_qcd(histograms, output_handle)

    def modify_cut(self, cuts, region, process, process_configs):
        parent_process = find_process_config(process, process_configs).name

        if parent_process != 'ttbar':
            return cuts
        stitch_point = self.get_stitch_point(region.name)
        cuts += ' && lq_mass_max / 1000. < {:f}'.format(stitch_point)
        return cuts

    def add_extrapolated_yields(self, sample, lumi=139.):
        uncert_name = '{:s}_extrapol'.format(self.tag)
        for region in sample.ctrl_region_yields.keys():
            yld, uncert = self.get_extrapolated_bin_content(region,
                                                            self.stich_points[region],
                                                            lumi=lumi)
            if not self.qcd_mode:
                sample.ctrl_region_yields[region] += yld
                uncert = uncert / sample.ctrl_region_yields[region].sum()
            else:
                sample.ctrl_region_yields[region] = Yield(yld)
                uncert = uncert / yld
            sample.ctrl_region_yields[region].extrapolated = True
            sample.ctrl_reg_shape_ylds[region]['{:s}__1down'.format(uncert_name)] = 1. - uncert
            sample.ctrl_reg_shape_ylds[region]['{:s}__1up'.format(uncert_name)] = 1. + uncert

        for region in sample.nominal_evt_yields.keys():
            stitch_point = self.get_stitch_point(region)
            for cut in sample.nominal_evt_yields[region].keys():
                if cut < stitch_point:
                    yld, uncert = self.get_extrapolated_bin_content(region, stitch_point, lumi=lumi)
                    sample.nominal_evt_yields[region][cut] += yld
                else:
                    yld, uncert = self.get_extrapolated_bin_content(region, cut, lumi=lumi)
                    if not self.qcd_mode:
                        sample.nominal_evt_yields[region][cut] += yld
                        sample.nominal_evt_yields[region][cut].extrapolated = True
                    else:
                        sample.nominal_evt_yields[region][cut] = Yield(yld)

                sample.nominal_evt_yields[region][cut].extrapolated = True
                uncert = uncert / sample.nominal_evt_yields[region][cut].sum()
                sample.shape_uncerts[region][cut]['{:s}__1down'.format(uncert_name)] = 1. - uncert
                sample.shape_uncerts[region][cut]['{:s}__1up'.format(uncert_name)] = 1. + uncert


class TopExtrapolationModule(ExtrapolationModule):
    def __init__(self, **kwargs):
        kwargs['qcd_mode'] = False
        super(TopExtrapolationModule, self).__init__(**kwargs)


class QCDExtrapolationModule(ExtrapolationModule):
    def __init__(self, **kwargs):
        kwargs['qcd_mode'] = True
        super(QCDExtrapolationModule, self).__init__(**kwargs)
