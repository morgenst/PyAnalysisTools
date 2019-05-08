from PyAnalysisTools.base import _logger
import ROOT


class TopExtrapolationModule(object):
    def __init__(self, **kwargs):
        _logger.debug('Initialising TopExtrapolationModule')
        self.build_functions(**kwargs)
        self.type = "DataModifier"

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
            self.functions[reg_name] = build_fct(cfg['func'], cfg['params']), cfg['stitch']

    def get_extrapolated_bin_content(self, region, xmin, xmax=None, lumi=1.):
        if xmax is None:
            xmax = 1e12
        if xmin < self.functions[region][1]:
            return None
        return lumi * self.functions[region][0][0].Integral(xmin, xmax)

    def execute(self, histograms):
        top_hist = histograms['ttbar']
        region = [r for r in self.functions.keys() if r in top_hist.GetName()][0]
        print region
        for i in range(top_hist.GetNbinsX() + 1):
            bin_content = self.get_extrapolated_bin_content(region, top_hist.GetXaxis().GetBinLowEdge(i),
                                                            top_hist.GetXaxis().GetBinUpEdge(i), 139.)
            if bin_content is None:
                continue
            top_hist.SetBinContent(i, bin_content)
