from __future__ import division
from builtins import str
from builtins import range
from builtins import object

import ROOT
from PyAnalysisTools.base.ProcessConfig import find_process_config
from PyAnalysisTools.base.FileHandle import FileHandle
from PyAnalysisTools.base import _logger


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
            return [ROOT.TF1("", f) for f in fcts]

        self.functions = {}
        for reg_name, cfg in list(kwargs.items()):
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
        list([h.SetDirectory(0) for h in self.histograms])

    def get_stitch_point(self, region):
        return self.stich_points[region]

    def get_extrapolated_bin_content(self, region, xmin, xmax=None, lumi=1.):
        uncert = 0
        if xmax is None:
            xmax = 10000000.
        if xmin < self.stich_points[region]:
            return None
        if region in self.functions:
            exit()
            new_yield = self.functions[region][0][0].Integral(xmin, xmax)
        elif len(self.histograms) > 0:
            h_nom = [htmp for htmp in self.histograms if htmp.GetName() == region][0]
            max_bin = -1
            if xmax is not None:
                max_bin = h_nom.FindBin(xmax) - 1
            new_yield = h_nom.Integral(h_nom.FindBin(xmin), max_bin)
            huncert = [htmp for htmp in self.histograms if htmp.GetName() == region + '_uncert'][0].Clone()
            huncert.Multiply(h_nom)
            max_bin = -1
            if xmax is not None:
                max_bin = huncert.FindBin(xmax) - 1
            uncert = huncert.Integral(huncert.FindBin(xmin), max_bin)
        else:
            _logger.error('Could not find information to extrapolate top')
            exit(-1)
        new_yield *= lumi
        uncert *= lumi
        _logger.debug('RESET bin content: {:.1f} {:.1f} {:.1f} {:.5f} based on {:s} and {:s}'.format(xmin, xmax,
                                                                                                     new_yield,
                                                                                                     uncert,
                                                                                                     h_nom.GetName(),
                                                                                                     huncert.GetName()))
        return new_yield, uncert

    def execute_top(self, histograms, output_handle, plot_config, systematics_handle):
        if 'ttbar' not in histograms:
            _logger.warn("Requested top extrapolation, but could not find ttbar in histograms.")
            return
        top_hist = histograms['ttbar']
        region = [r for r in list(self.stich_points.keys()) if r.replace('ZVR', 'VR') in top_hist.GetName()][0]
        top_uncert_up = top_hist.Clone('{:s}_lq_mass_max_ttbar_top_extrapol_unc__1up'.format(region))
        top_uncert_down = top_hist.Clone('{:s}_lq_mass_max_ttbar_top_extrapol_unc__1down'.format(region))
        _logger.debug('Running top extrapolation in region {:s}'.format(region))
        uncert_sum = 0.
        for i in range(top_hist.GetNbinsX() + 1):
            bin_content = self.get_extrapolated_bin_content(region, top_hist.GetXaxis().GetBinLowEdge(i),
                                                            top_hist.GetXaxis().GetBinUpEdge(i), 139.)
            if bin_content is None:
                continue
            _logger.debug("old yield: {:.2f} new yield: {:.2f}".format(top_hist.GetBinContent(i), bin_content[0]))
            top_hist.SetBinContent(i, bin_content[0])
            if top_hist.GetBinLowEdge(i) > 1699.:
                uncert_sum += bin_content[1]
            if systematics_handle is not None:
                # need to modify all tail of the systematics histograms:
                for unc in list(systematics_handle.systematic_variations.keys()):
                    if 'ttbar' not in systematics_handle.systematic_variations[unc][plot_config]:
                        continue
                    if 'theory_envelop' in unc:
                        continue
                    if systematics_handle.systematic_variations[unc][plot_config]['ttbar'] is None:
                        if 'pdf_uncert_MU' not in unc:
                            _logger.error("Somehow ttbar unc is None for {:s}".format(unc))
                        continue
                    systematics_handle.systematic_variations[unc][plot_config]['ttbar'].SetBinContent(i, bin_content[0])
            # TODO: Need some way for relative and abs uncertainty
            top_uncert_up.SetBinContent(i, bin_content[0] + bin_content[1])
            top_uncert_down.SetBinContent(i, bin_content[0] - bin_content[1])
        output_handle.register_object(top_hist)
        output_handle.register_object(top_uncert_up)
        output_handle.register_object(top_uncert_down)
        if systematics_handle is not None:
            for unc in list(systematics_handle.systematic_variations.keys()):
                if 'ttbar' not in systematics_handle.systematic_variations[unc][plot_config]:
                    continue
                if systematics_handle.systematic_variations[unc][plot_config]['ttbar'] is None:
                    _logger.error("Somehow hist is None. Something likely went wrong")
                    continue
                output_handle.register_object(systematics_handle.systematic_variations[unc][plot_config]['ttbar'])

    def execute_qcd(self, histograms, output_handle):
        try:
            region = [r for r in list(self.stich_points.keys()) if r in list(histograms.values())[0].GetName()][0]
        except IndexError:
            _logger.error('Requested qcd but could not find corresponding region')
            return
        h_qcd = list(histograms.values())[0].Clone(region + "_lq_mass_max_QCD")
        qcd_uncert_up = h_qcd.Clone('{:s}_lq_mass_max_QCD_extrapol_unc__1up'.format(region))
        qcd_uncert_down = h_qcd.Clone('{:s}_lq_mass_max_QCD_extrapol_unc__1down'.format(region))
        for i in range(h_qcd.GetNbinsX() + 1):
            bin_content = self.get_extrapolated_bin_content(region, h_qcd.GetXaxis().GetBinLowEdge(i),
                                                            h_qcd.GetXaxis().GetBinUpEdge(i), 1.)
            if bin_content is None:
                continue
            h_qcd.SetBinContent(i, bin_content[0])
            h_qcd.SetBinError(i, 0.)
            qcd_uncert_up.SetBinContent(i, bin_content[0] + abs(bin_content[1]))
            qcd_uncert_down.SetBinContent(i, bin_content[0] - abs(bin_content[1]))
        output_handle.register_object(h_qcd)
        output_handle.register_object(qcd_uncert_up)
        output_handle.register_object(qcd_uncert_down)

    def execute(self, histograms, output_handle, plot_config=None, systematics_handle=None):
        if not self.qcd_mode:
            self.execute_top(histograms, output_handle, plot_config, systematics_handle)
        else:
            self.execute_qcd(histograms, output_handle)

    def modify_cut(self, cuts, region, process, process_configs):
        parent_process = find_process_config(process, process_configs).name

        if parent_process != 'ttbar':
            return cuts
        stitch_point = self.get_stitch_point(region.name)
        cuts += ' && lq_mass_max / 1000. < {:f}'.format(stitch_point)
        return cuts

    # def add_extrapolated_yields(self, sample, lumi=139.):
    #     uncert_name = '{:s}_extrapol'.format(self.tag)
    #     for region in list(sample.ctrl_region_yields.keys()):
    #         yld, uncert = self.get_extrapolated_bin_content(region,
    #                                                         self.stich_points[region],
    #                                                         lumi=lumi)
    #         if not self.qcd_mode:
    #             sample.ctrl_region_yields[region] += yld
    #             uncert = old_div(uncert, sample.ctrl_region_yields[region].sum())
    #         else:
    #             sample.ctrl_region_yields[region] = Yield(yld)
    #             uncert = old_div(uncert, yld)
    #         sample.ctrl_region_yields[region].extrapolated = True
    #         sample.ctrl_reg_shape_ylds[region]['{:s}__1down'.format(uncert_name)] = 1. - uncert
    #         sample.ctrl_reg_shape_ylds[region]['{:s}__1up'.format(uncert_name)] = 1. + uncert
    #
    #     for region in list(sample.nominal_evt_yields.keys()):
    #         stitch_point = self.get_stitch_point(region)
    #         for cut in list(sample.nominal_evt_yields[region].keys()):
    #             if cut < stitch_point:
    #                 yld, uncert = self.get_extrapolated_bin_content(region, stitch_point, lumi=lumi)
    #                 sample.nominal_evt_yields[region][cut] += yld
    #             else:
    #                 yld, uncert = self.get_extrapolated_bin_content(region, cut, lumi=lumi)
    #                 if not self.qcd_mode:
    #                     sample.nominal_evt_yields[region][cut] += yld
    #                     sample.nominal_evt_yields[region][cut].extrapolated = True
    #                 else:
    #                     sample.nominal_evt_yields[region][cut] = Yield(yld)
    #
    #             sample.nominal_evt_yields[region][cut].extrapolated = True
    #             uncert = old_div(uncert, sample.nominal_evt_yields[region][cut].sum())
    #             sample.shape_uncerts[region][cut]['{:s}__1down'.format(uncert_name)] = 1. - uncert
    #             sample.shape_uncerts[region][cut]['{:s}__1up'.format(uncert_name)] = 1. + uncert


class TopExtrapolationModule(ExtrapolationModule):
    def __init__(self, **kwargs):
        kwargs['qcd_mode'] = False
        super(TopExtrapolationModule, self).__init__(**kwargs)


class QCDExtrapolationModule(ExtrapolationModule):
    def __init__(self, **kwargs):
        kwargs['qcd_mode'] = True
        super(QCDExtrapolationModule, self).__init__(**kwargs)
